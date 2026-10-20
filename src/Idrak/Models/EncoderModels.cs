// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// A head over an encoder's output, made by a kind of <see cref="EncoderHeads"/>: a <see cref="Pooled"/> head reads one
/// vector per sequence [batch, width] (a classifier), the others the states [batch, time, width] (a masked-language-model
/// head); either gives [..., <see cref="Outputs"/>].
/// </summary>
public sealed class EncoderHead : Module
{
    /// <summary>A head of the kind <paramref name="kind"/> around <paramref name="network"/> (owned).</summary>
    /// <param name="kind">Its kind's name in <see cref="EncoderHeads"/>.</param>
    /// <param name="pooled">Whether it reads the pooled vectors (else the states).</param>
    /// <param name="outputs">Its outputs (the last dimension of its result).</param>
    /// <param name="network">Its layers.</param>
    public EncoderHead(string kind, bool pooled, int outputs, Module network)
    {
        ArgumentException.ThrowIfNullOrEmpty(kind);
        ArgumentNullException.ThrowIfNull(network);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputs);
        Kind = kind;
        Pooled = pooled;
        Outputs = outputs;
        Network = network;
        Name = kind;
    }

    /// <summary>Its kind's name in <see cref="EncoderHeads"/>.</summary>
    public string Kind { get; }

    /// <summary>Whether it reads the pooled vectors [batch, width] (else the states [batch, time, width]).</summary>
    public bool Pooled { get; }

    /// <summary>Its outputs: labels, or the vocabulary.</summary>
    public int Outputs { get; }

    /// <summary>Its layers.</summary>
    public Module Network { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Network.Forward(input);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [Network];

    /// <inheritdoc />
    public override string ToString() => $"EncoderHead({Kind}, {Outputs} outputs{(Pooled ? ", pooled" : "")})";

    // "sequence-classification": [dense, activation,] out over the pooled vector.
    internal static EncoderHead Classification(EncoderHeadRequest request)
    {
        var (device, random) = Begin(request);
        bool dense = Flag(request, "dense", false);
        var layers = new List<Module>();
        if (dense)
        {
            layers.Add(Layer(request, "dense", request.Width, request.Width, device, random));
            layers.Add(Activation(request, "tanh"));
        }

        layers.Add(Layer(request, "out", request.Width, request.Outputs, device, random));
        return new EncoderHead(EncoderHeads.SequenceClassification, pooled: true, request.Outputs, new Sequential(layers));
    }

    // "masked-lm": [dense, activation, layer norm,] out over every position's state.
    internal static EncoderHead MaskedLanguageModel(EncoderHeadRequest request)
    {
        var (device, random) = Begin(request);
        bool transform = Flag(request, "transform", false);
        var layers = new List<Module>();
        if (transform)
        {
            float epsilon = 1e-5f;
            if (request.Settings?["norm_epsilon"] is { } node)
            {
                epsilon = node.GetValueKind() == JsonValueKind.Number && node.GetValue<double>() is var e && e > 0 && double.IsFinite(e) ? (float)e
                    : throw new ArgumentException($"The head setting norm_epsilon is a positive number; got {node.ToJsonString()}.", nameof(request));
            }

            layers.Add(Layer(request, "dense", request.Width, request.Width, device, random));
            layers.Add(Activation(request, "gelu"));
            var gain = Read(request, "norm.weight", [request.Width]) ?? Enumerable.Repeat(1f, request.Width).ToArray();
            var shift = Read(request, "norm.bias", [request.Width]) ?? new float[request.Width];
            layers.Add(LayerNorm.FromWeights(Tensor.Persistent(gain, [request.Width], device, requiresGrad: true),
                Tensor.Persistent(shift, [request.Width], device, requiresGrad: true), epsilon));
        }

        layers.Add(Layer(request, "out", request.Width, request.Outputs, device, random));
        return new EncoderHead(EncoderHeads.MaskedLanguageModel, pooled: false, request.Outputs, new Sequential(layers));
    }

    private static (Device Device, Random Random) Begin(EncoderHeadRequest request)
    {
        if (request.Width <= 0)
        {
            throw new ArgumentException($"The head setting width is positive; got {request.Width}.", nameof(request));
        }

        if (request.Outputs <= 0)
        {
            throw new ArgumentException($"The head setting outputs (labels or vocabulary) is positive; got {request.Outputs}.", nameof(request));
        }

        return (request.Device ?? Device.Default, new Random(request.Seed));
    }

    private static bool Flag(EncoderHeadRequest request, string name, bool fallback) => request.Settings?[name] switch
    {
        null => fallback,
        var node when node.GetValueKind() is JsonValueKind.True or JsonValueKind.False => node.GetValue<bool>(),
        var node => throw new ArgumentException($"The head setting {name} is true or false; got {node.ToJsonString()}.", nameof(request)),
    };

    private static Module Activation(EncoderHeadRequest request, string fallback)
    {
        string name = request.Settings?["activation"] is { } node
            ? node.GetValueKind() == JsonValueKind.String ? node.GetValue<string>() : throw new ArgumentException($"The head setting activation is a name; got {node.ToJsonString()}.", nameof(request))
            : fallback;
        return name switch
        {
            "tanh" => new Tanh(),
            "gelu" => new ExactGELU(),
            "gelu-tanh" => new GELU(),
            "relu" => new ReLU(),
            _ => throw new ArgumentException($"The head setting activation is tanh, gelu, gelu-tanh or relu; got '{name}'.", nameof(request)),
        };
    }

    private static float[]? Read(EncoderHeadRequest request, string name, int[] shape)
    {
        if (request.Weights?.Invoke(name, shape) is not { } values)
        {
            return null;
        }

        long size = shape.Aggregate(1L, (a, b) => a * b);
        return values.Length == size ? values
            : throw new InvalidDataException($"The head tensor {name} has {values.Length} values; [{string.Join(", ", shape)}] is {size}.");
    }

    // A linear layer [in, out] with its bias: read when the weights hold them, else random (a missing bias: zeros).
    private static Linear Layer(EncoderHeadRequest request, string name, int inputs, int outputs, Device device, Random random)
    {
        if (Read(request, $"{name}.weight", [inputs, outputs]) is not { } weight)
        {
            return new Linear(inputs, outputs, bias: true, device, random);
        }

        var bias = Read(request, $"{name}.bias", [outputs]) ?? new float[outputs];
        var layer = Linear.FromWeights(Tensor.Persistent(weight, [inputs, outputs], device, requiresGrad: true), Tensor.Persistent(bias, [outputs], device, requiresGrad: true));
        layer.Name = name;
        return layer;
    }
}

/// <summary>
/// The library's <see cref="ISeq2SeqModel"/>: an <see cref="ISequenceEncoder"/> and a decoder of layers (token ids
/// [batch, time] → logits, its last layer the output head) whose blocks hold <see cref="CrossAttention"/>s over the
/// encoder's states (<see cref="DecoderBlock"/>s built with a cross-attention): the whole target at once through an
/// <see cref="EncoderMemory"/> scope, or step by step through the decoding context, the memory projected once per
/// sequence. A family plug-in builds one from its checkpoint; it takes ownership of the encoder and the decoder.
/// </summary>
public sealed class Seq2SeqNetwork : ISeq2SeqModel
{
    private readonly CrossAttention[] _crosses;

    /// <summary>Creates the model from its encoder and decoder.</summary>
    /// <param name="encoder">The source's encoder.</param>
    /// <param name="decoder">The decoder, ids to logits: its cached layers implement <see cref="Layers.Abstractions.ICachedModule"/> and its last one is the output head.</param>
    /// <param name="decoderStartToken">The token decoding starts from.</param>
    /// <param name="endToken">The token that ends a target.</param>
    /// <param name="maxTargetPositions">The longest target; null takes the decoder's position limit (its position embedding's or its attention's).</param>
    /// <exception cref="ArgumentException">The decoder has no cross-attention or no output head, a token is outside its vocabulary, or no limit is known.</exception>
    public Seq2SeqNetwork(ISequenceEncoder encoder, Sequential decoder, int decoderStartToken, int endToken, int? maxTargetPositions = null)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        ArgumentNullException.ThrowIfNull(decoder);
        _crosses = [.. decoder.Descendants().OfType<CrossAttention>()];
        if (_crosses.Length == 0)
        {
            throw new ArgumentException("An encoder-decoder's decoder attends to the encoder: it holds no CrossAttention (build its blocks with one).", nameof(decoder));
        }

        if (decoder.Count() == 0 || decoder.Last() is not Linear head)
        {
            throw new ArgumentException("The decoder's last layer is its output head (a Linear giving the logits).", nameof(decoder));
        }

        Vocabulary = head.OutFeatures;
        if ((uint)decoderStartToken >= (uint)Vocabulary || (uint)endToken >= (uint)Vocabulary)
        {
            throw new ArgumentException($"The start token {decoderStartToken} and end token {endToken} are tokens of the decoder's {Vocabulary}.");
        }

        int? limit = maxTargetPositions ?? decoder.Descendants().Select(m => m switch
        {
            PositionEmbedding p => p.MaxPositions,
            CausalSelfAttention a => a.MaxPositions,
            _ => (int?)null,
        }).Where(l => l is not null).Min();
        MaxTargetPositions = limit is > 0 ? limit.Value
            : throw new ArgumentException("The longest target is not known from the decoder's layers: pass maxTargetPositions.", nameof(maxTargetPositions));
        Encoder = encoder;
        Decoder = decoder;
        DecoderStartToken = decoderStartToken;
        EndToken = endToken;
        Network = new Parts(encoder, decoder);
    }

    /// <inheritdoc />
    public ISequenceEncoder Encoder { get; }

    /// <summary>The decoder: ids to logits, its blocks holding the cross-attentions.</summary>
    public Sequential Decoder { get; }

    /// <inheritdoc />
    public int Vocabulary { get; }

    /// <inheritdoc />
    public int DecoderStartToken { get; }

    /// <inheritdoc />
    public int EndToken { get; }

    /// <inheritdoc />
    public int MaxTargetPositions { get; }

    /// <inheritdoc />
    public Device Device => Encoder.Device;

    /// <inheritdoc />
    public Module Network { get; }

    /// <inheritdoc />
    public Tensor Decode(Tensor ids, Tensor memory, IReadOnlyList<int>? memoryLengths = null)
    {
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(memory);
        using (EncoderMemory.Use(memory, memoryLengths))
        {
            return Decoder.Forward(ids);
        }
    }

    /// <inheritdoc />
    public void SetMemory(Tensor memory, DecodingContext context, IReadOnlyList<int>? memoryLengths = null)
    {
        ArgumentNullException.ThrowIfNull(memory);
        ArgumentNullException.ThrowIfNull(context);
        foreach (var cross in _crosses)
        {
            cross.SetMemory(memory, context, memoryLengths);
        }
    }

    /// <inheritdoc />
    public Tensor DecodeCached(Tensor ids, DecodingContext context) => Decoder.ForwardCached(ids, context);

    /// <inheritdoc />
    public void Dispose() => Network.Dispose();

    /// <inheritdoc />
    public override string ToString() => $"Seq2SeqNetwork({Encoder}; decoder of {Decoder.Count()} layers, {Vocabulary} tokens)";

    // The trained layers: the encoder's, then the decoder's.
    private sealed class Parts(ISequenceEncoder encoder, Sequential decoder) : Module
    {
        public override IEnumerable<Module> Children() => [encoder.Network, decoder];

        protected override Tensor ForwardCore(Tensor input) =>
            throw new NotSupportedException("An encoder-decoder runs in two parts: Encoder.Encode, then Decode (or DecodeCached).");

        // Every layer through its own Dispose (attention layers' masks and key ranges); a tensor disposed twice is a no-op.
        public override void Dispose()
        {
            encoder.Dispose();
            foreach (var module in Enumerable.Reverse(decoder.Descendants().ToList()))
            {
                module.Dispose();
            }

            base.Dispose();
        }
    }
}

/// <summary>How <see cref="PretrainedEncoderModel.Load"/> and <see cref="PretrainedSeq2SeqModel.Load"/> read a model.</summary>
public sealed record PretrainedEncoderOptions
{
    /// <summary>Where the network is built (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>The architecture to read the checkpoint as, instead of config.json's first "architectures" name.</summary>
    public string? Architecture { get; init; }

    /// <summary>How the folder's tokenizer is read (the defaults when null).</summary>
    public TokenizerOptions? Tokenizer { get; init; }

    /// <summary>The pooler (a name in <see cref="SequencePoolers"/>) instead of the family's (encoders only).</summary>
    public string? Pooler { get; init; }
}

// What both loaders read first: the folder prepared by its checkpoint format, its config.json and architecture name.
internal static class EncoderCheckpoints
{
    public static (string Folder, JsonObject Config, string Name, ICheckpointFormat Format) Read(string folder, string? architecture)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(folder);
        var format = CheckpointFormats.For(folder);
        folder = format.Prepare(folder);
        var config = JsonNode.Parse(File.ReadAllText(Path.Combine(folder, "config.json")))!.AsObject();
        string name = architecture ?? (string?)config["architectures"]?[0]
            ?? throw new InvalidDataException("config.json names no architecture; pass the options' Architecture.");
        return (folder, config, name, format);
    }
}

/// <summary>
/// A pretrained encoder read from a model folder by its family (<see cref="PretrainedEncoders"/>): the encoder, the heads
/// its checkpoint holds, its pooler and its tokenizer. <see cref="Hidden"/> gives the states, <see cref="Pooled"/> one
/// vector per sequence, <see cref="Logits"/> a head's output; Idrak.Text's <c>TextEncoder</c> turns texts into vectors
/// with <see cref="Encoder"/>.
/// </summary>
public sealed class PretrainedEncoderModel : IDisposable
{
    private PretrainedEncoderModel(string folder, JsonObject config, string architecture, ISequenceEncoder encoder, IReadOnlyDictionary<string, EncoderHead> heads,
        string pooler, IPretrainedTokenizer? tokenizer, IReadOnlyList<string> notes)
    {
        (Folder, Config, Architecture, Encoder, Heads, Pooler, Tokenizer, Notes) = (folder, config, architecture, encoder, heads, pooler, tokenizer, notes);
    }

    /// <summary>The model folder.</summary>
    public string Folder { get; }

    /// <summary>The model's config.json.</summary>
    public JsonObject Config { get; }

    /// <summary>The architecture it was read as.</summary>
    public string Architecture { get; }

    /// <summary>The encoder.</summary>
    public ISequenceEncoder Encoder { get; }

    /// <summary>The heads the checkpoint holds, by kind.</summary>
    public IReadOnlyDictionary<string, EncoderHead> Heads { get; }

    /// <summary>The pooler of <see cref="Pooled"/> (a name in <see cref="SequencePoolers"/>).</summary>
    public string Pooler { get; }

    /// <summary>The folder's tokenizer, or null when it has none.</summary>
    public IPretrainedTokenizer? Tokenizer { get; }

    /// <summary>What was approximated while reading.</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>Where it runs.</summary>
    public Device Device => Encoder.Device;

    /// <summary>
    /// Reads the model in <paramref name="folder"/> with the family registered under its architecture name.
    /// </summary>
    /// <exception cref="NotSupportedException">No encoder family is registered under the architecture's name (the message says how to register one).</exception>
    public static PretrainedEncoderModel Load(string folder, PretrainedEncoderOptions? options = null)
    {
        options ??= new PretrainedEncoderOptions();
        var (prepared, config, name, format) = EncoderCheckpoints.Read(folder, options.Architecture);
        var family = PretrainedEncoders.Get(name);
        string pooler = options.Pooler ?? family.Pooler(config);
        SequencePoolers.Get(pooler);                                           // a pooler nobody registered is refused before anything is built
        var notes = new List<string>();
        notes.AddRange(format.Notes(prepared));
        ISequenceEncoder encoder;
        IReadOnlyList<EncoderHead> heads;
        using (var reader = format.Open(prepared))
        {
            var checkpoint = new PretrainedCheckpoint(name, config, reader, prepared, options.Device ?? Device.Default, notes);
            encoder = family.Build(checkpoint) ?? throw new InvalidOperationException($"The encoder family '{name}' built no encoder.");
            try
            {
                heads = family.Heads(checkpoint, encoder);
            }
            catch
            {
                encoder.Dispose();
                throw;
            }
        }

        var byKind = new Dictionary<string, EncoderHead>(StringComparer.Ordinal);
        foreach (var head in heads)
        {
            byKind[head.Kind] = head;
        }

        var tokenizer = PretrainedTokenizers.TryLoad(prepared, options.Tokenizer);
        return new PretrainedEncoderModel(prepared, config, name, encoder, byKind, pooler, tokenizer, notes);
    }

    /// <summary>The hidden states [batch, time, width] (see <see cref="ISequenceEncoder.Encode"/>).</summary>
    public Tensor Hidden(Tensor ids, IReadOnlyList<int>? lengths = null, Tensor? tokenTypes = null) => Encoder.Encode(ids, lengths, tokenTypes);

    /// <summary>One vector per sequence [batch, width]: the states pooled by <see cref="Pooler"/>.</summary>
    public Tensor Pooled(Tensor ids, IReadOnlyList<int>? lengths = null, Tensor? tokenTypes = null) =>
        SequencePoolers.Get(Pooler).Pool(Hidden(ids, lengths, tokenTypes), lengths);

    /// <summary>The output of the head <paramref name="kind"/>: over the pooled vectors or the states, as the head reads.</summary>
    /// <exception cref="ArgumentException">The checkpoint holds no head of that kind (the message lists those it holds).</exception>
    public Tensor Logits(string kind, Tensor ids, IReadOnlyList<int>? lengths = null, Tensor? tokenTypes = null)
    {
        if (!Heads.TryGetValue(kind, out var head))
        {
            throw new ArgumentException($"The model holds no '{kind}' head (it holds: {(Heads.Count == 0 ? "none" : string.Join(", ", Heads.Keys))}).", nameof(kind));
        }

        return head.Forward(head.Pooled ? Pooled(ids, lengths, tokenTypes) : Hidden(ids, lengths, tokenTypes));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (var head in Heads.Values)
        {
            head.Dispose();
        }

        Encoder.Dispose();
    }
}

/// <summary>
/// A pretrained text encoder-decoder read from a model folder by its family (<see cref="PretrainedSeq2Seq"/>): the model
/// and its tokenizer. <see cref="CreateGenerator"/> generates text with it through the library's samplers.
/// </summary>
public sealed class PretrainedSeq2SeqModel : IDisposable
{
    private PretrainedSeq2SeqModel(string folder, JsonObject config, string architecture, ISeq2SeqModel model, IPretrainedTokenizer? tokenizer, IReadOnlyList<string> notes)
    {
        (Folder, Config, Architecture, Model, Tokenizer, Notes) = (folder, config, architecture, model, tokenizer, notes);
    }

    /// <summary>The model folder.</summary>
    public string Folder { get; }

    /// <summary>The model's config.json.</summary>
    public JsonObject Config { get; }

    /// <summary>The architecture it was read as.</summary>
    public string Architecture { get; }

    /// <summary>The encoder-decoder.</summary>
    public ISeq2SeqModel Model { get; }

    /// <summary>The folder's tokenizer, or null when it has none.</summary>
    public IPretrainedTokenizer? Tokenizer { get; }

    /// <summary>What was approximated while reading.</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>Where it runs.</summary>
    public Device Device => Model.Device;

    /// <summary>Reads the model in <paramref name="folder"/> with the family registered under its architecture name.</summary>
    /// <exception cref="NotSupportedException">No encoder-decoder family is registered under the architecture's name (the message says how to register one).</exception>
    public static PretrainedSeq2SeqModel Load(string folder, PretrainedEncoderOptions? options = null)
    {
        options ??= new PretrainedEncoderOptions();
        var (prepared, config, name, format) = EncoderCheckpoints.Read(folder, options.Architecture);
        var family = PretrainedSeq2Seq.Get(name);
        var notes = new List<string>();
        notes.AddRange(format.Notes(prepared));
        ISeq2SeqModel model;
        using (var reader = format.Open(prepared))
        {
            model = family.Build(new PretrainedCheckpoint(name, config, reader, prepared, options.Device ?? Device.Default, notes))
                    ?? throw new InvalidOperationException($"The encoder-decoder family '{name}' built no model.");
        }

        var tokenizer = PretrainedTokenizers.TryLoad(prepared, options.Tokenizer);
        return new PretrainedSeq2SeqModel(prepared, config, name, model, tokenizer, notes);
    }

    /// <summary>A generator of text with this model, with <paramref name="tokenizer"/> or the folder's.</summary>
    /// <exception cref="InvalidOperationException">No tokenizer is given and the folder has none.</exception>
    public Seq2SeqGenerator CreateGenerator(ITokenizer? tokenizer = null) =>
        new(Model, tokenizer ?? Tokenizer ?? throw new InvalidOperationException($"The model in {Folder} has no tokenizer: pass one."));

    /// <inheritdoc />
    public void Dispose() => Model.Dispose();
}
