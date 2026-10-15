// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Audio;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.IdrakAudio;

/// <summary>
/// The projector of an IdrakAudio audio-language model: every <see cref="Stack"/> consecutive encoder steps joined into one
/// soft token (the last group padded with zeros), then into the text decoder's width: "mlp" (linear_1, the activation,
/// linear_2) or "linear" (linear_1 alone). Its keys are config.json's "audio_projector".
/// </summary>
public sealed record IdrakAudioProjectorConfig
{
    /// <summary>Encoder steps per soft token.</summary>
    public int Stack { get; init; } = 4;

    /// <summary>"mlp" or "linear".</summary>
    public string Type { get; init; } = "mlp";

    /// <summary>The text decoder's width (the soft tokens' embeddings).</summary>
    public int TextDim { get; init; }

    /// <summary>The settings of config.json's "audio_projector" (or a description's); <paramref name="textDim"/> when it gives none.</summary>
    /// <exception cref="InvalidDataException">A value is out of range.</exception>
    public static IdrakAudioProjectorConfig FromJson(JsonObject? json, int? textDim = null)
    {
        var d = new IdrakAudioProjectorConfig();
        var config = new IdrakAudioProjectorConfig
        {
            Stack = (int?)json?["stack"] ?? d.Stack,
            Type = (string?)json?["type"] ?? d.Type,
            TextDim = (int?)json?["text_dim"] ?? textDim ?? 0,
        };
        if (config.Stack < 1 || config.Type is not ("mlp" or "linear") || config.TextDim < 1)
        {
            throw new InvalidDataException($"An IdrakAudio projector stacks at least one step ({config.Stack}), is \"mlp\" or \"linear\" ({config.Type}) and has a text width ({config.TextDim}).");
        }

        return config;
    }

    /// <summary>The settings as config.json's "audio_projector".</summary>
    public JsonObject ToJson() => new() { ["stack"] = Stack, ["type"] = Type, ["text_dim"] = TextDim };

    /// <summary>The soft tokens of <paramref name="steps"/> encoder steps.</summary>
    public int TokensFor(int steps) => (steps + Stack - 1) / Stack;
}

/// <summary>The projector's layers (<see cref="IdrakAudioProjectorConfig"/>): [tokens, stack · dim] → [tokens, text width].</summary>
public sealed class IdrakAudioProjector : Module
{
    private readonly Linear _first;
    private readonly Linear? _second;
    private readonly string _activation;

    internal IdrakAudioProjector(IdrakAudioProjectorConfig config, int dim, string activation, IdrakAudioWeights weights)
    {
        Config = config;
        _activation = activation;
        _first = weights.Linear("linear_1", config.Stack * dim, config.TextDim);
        _first.Name = "linear_1";
        if (config.Type == "mlp")
        {
            _second = weights.Linear("linear_2", config.TextDim, config.TextDim);
            _second.Name = "linear_2";
        }
    }

    /// <summary>The settings.</summary>
    public IdrakAudioProjectorConfig Config { get; }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => _second is null ? [_first] : [_first, _second];

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        var x = _first.Forward(input);
        if (_second is null)
        {
            return x;
        }

        x = _activation == "gelu" ? ExactGELU.Apply(x) : x.Gelu();
        return _second.Forward(x);
    }
}

/// <summary>
/// The IdrakAudio audio-language family (<see cref="IAudioFamily"/>, under <see cref="Architecture"/>): an audio part made
/// from scratch for a text decoder (Llama-style, its tensors under "language_model.", its spec as "text_config"). The
/// part: the IdrakAudio encoder over log-mel features ("audio_config", its tensors under "audio_tower.") and the projector
/// ("audio_projector", tensors under "audio_projector."); a sound becomes one block of soft tokens, the rendered prompt's
/// audio marker expanded to the begin token, the soft tokens and the end token. Both train ("audio-encoder",
/// "audio-projector"), through the library's model part ("audio") and pretraining stages.
/// </summary>
public sealed class IdrakAudioLanguageFamily : IAudioFamily
{
    /// <summary>The architecture name config.json gives; the text decoder is registered under it too.</summary>
    public const string Architecture = "IdrakAudioForConditionalGeneration";

    /// <summary>Where the encoder's tensors are stored.</summary>
    public const string TowerPrefix = "audio_tower.";

    /// <summary>Where the projector's tensors are stored.</summary>
    public const string ProjectorPrefix = "audio_projector.";

    /// <summary>The trainable part: the encoder.</summary>
    public const string EncoderPart = "audio-encoder";

    /// <summary>The trainable part: the projector.</summary>
    public const string ProjectorPart = "audio-projector";

    /// <inheritdoc />
    public string Name => Architecture;

    /// <inheritdoc />
    public PretrainedAudio? Read(AudioCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var config = checkpoint.Config;
        var encoder = IdrakAudioConfig.FromJson(config["audio_config"] as JsonObject ?? throw new InvalidDataException("An IdrakAudio audio-language config.json has an \"audio_config\"."));
        var projector = IdrakAudioProjectorConfig.FromJson(config["audio_projector"] as JsonObject);
        var stored = checkpoint.Tensors.Names.Where(n => n.StartsWith(TowerPrefix, StringComparison.Ordinal) || n.StartsWith(ProjectorPrefix, StringComparison.Ordinal)).ToHashSet(StringComparer.Ordinal);
        if (stored.Count == 0)
        {
            checkpoint.Notes.Add($"the {Architecture} checkpoint holds no audio weights (audio_tower., audio_projector.).");
            return null;
        }

        foreach (string required in new[] { TowerPrefix + "norm.weight", ProjectorPrefix + "linear_1.weight" })
        {
            if (!stored.Contains(required))
            {
                throw new InvalidDataException($"The {Architecture} checkpoint has no '{required}'.");
            }
        }

        int token = Index(config, "audio_token_index") ?? throw new InvalidDataException("An IdrakAudio audio-language config.json gives \"audio_token_index\".");
        var attention = ImageAttentionRules.Get((string?)config["audio_attention"] ?? ImageAttentionRules.Causal);
        return new IdrakAudioLanguagePart(encoder, projector, stored, token, Index(config, "boa_token_index"), Index(config, "eoa_token_index"), attention, checkpoint.Open);
    }

    /// <inheritdoc />
    /// <remarks>
    /// The description: <c>{"audio_config": {IdrakAudioConfig's keys}, "audio_projector": {"stack": 4, "type": "mlp"},
    /// "audio_token": "&lt;audio&gt;", "begin_audio": null, "end_audio": null, "audio_attention": "causal", "init_scale": 1}</c>,
    /// every key optional; the tokens must be single tokens of the tokenizer (the audio token is also the marker the chat
    /// template writes). Weights: matrices a normal draw of init_scale / √inputs from <paramref name="seed"/>, biases 0,
    /// norms 1.
    /// </remarks>
    public CreatedAudio Create(JsonObject audio, int textDim, ITokenizer tokenizer, int seed)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(tokenizer);
        var encoderJson = audio["audio_config"]?.DeepClone() as JsonObject ?? new JsonObject();
        encoderJson["task"] = IdrakAudioFamily.AudioLanguageTask;
        var encoder = IdrakAudioConfig.FromJson(encoderJson);
        var projector = IdrakAudioProjectorConfig.FromJson(audio["audio_projector"] as JsonObject, textDim) with { TextDim = textDim };
        int? Token(string key, string? fallback)
        {
            string? text = audio.ContainsKey(key) ? (string?)audio[key] : fallback;
            if (text is null)
            {
                return null;
            }

            var ids = tokenizer.Encode(text);
            return ids.Count == 1 ? ids[0] : throw new InvalidDataException($"The tokenizer has no single token '{text}' ({key}); add it as a special token (idrak data tokenizer --special).");
        }

        int token = Token("audio_token", "<audio>")!.Value;
        int? begin = Token("begin_audio", null), end = Token("end_audio", null);
        string attention = (string?)audio["audio_attention"] ?? ImageAttentionRules.Causal;
        _ = ImageAttentionRules.Get(attention);
        float scale = (float?)audio["init_scale"] ?? encoder.InitScale;
        var tensors = new List<(string, int[], float[])>();
        var towerWeights = IdrakAudioWeights.Draw(seed, scale, Device.Cpu, TowerPrefix);
        var projectorWeights = IdrakAudioWeights.Draw(seed ^ 0x5eed, scale, Device.Cpu, ProjectorPrefix);
        using (var tower = new IdrakAudioEncoder(encoder, towerWeights))
        using (var layers = new IdrakAudioProjector(projector, encoder.Dim, encoder.Activation, projectorWeights))
        {
            foreach (var (name, tensor, shape, transposed) in towerWeights.Entries.Concat(projectorWeights.Entries))
            {
                tensors.Add((name, shape, IdrakAudioWeights.Stored(tensor, transposed)));
            }
        }

        var encoderConfig = encoder.ToJson();
        encoderConfig.Remove("architectures");
        encoderConfig.Remove("model_type");
        var keys = new JsonObject
        {
            ["audio_config"] = encoderConfig,
            ["audio_projector"] = projector.ToJson(),
            ["audio_token_index"] = token,
            ["boa_token_index"] = begin,
            ["eoa_token_index"] = end,
            ["audio_attention"] = attention,
        };
        return new CreatedAudio(keys, tensors, new Dictionary<string, string>());
    }

    private static int? Index(JsonObject config, string key) => config[key] is JsonValue value ? (int)value : null;

    // The text side: Llama's spec from text_config, its tensors under language_model., a spec written as text_config.
    internal static PretrainedArchitecture TextDecoder()
    {
        var llama = PretrainedFamilies.LlamaStyle((_, spec, _) => spec, "llama");
        return new PretrainedArchitecture
        {
            Spec = (config, notes) => llama.Spec(config["text_config"] as JsonObject
                ?? throw new InvalidDataException($"An {Architecture} config.json has a \"text_config\"."), notes),
            TensorName = name => PretrainedFamilies.LlamaTensorName(name) is { } stored ? "language_model." + stored : null,
            Config = spec => new JsonObject { ["text_config"] = llama.Config!(spec), ["tie_word_embeddings"] = spec.TieEmbeddings },
        };
    }
}

/// <summary>The IdrakAudio audio-language part as its family read it: settings, features, prompt format; it trains (<see cref="IAudioTuningPart"/>).</summary>
public sealed class IdrakAudioLanguagePart : PretrainedAudio, IAudioTuningPart
{
    private readonly IReadOnlySet<string> _stored;
    private readonly Func<ITensorStore> _open;

    internal IdrakAudioLanguagePart(IdrakAudioConfig encoder, IdrakAudioProjectorConfig projector, IReadOnlySet<string> stored, int token, int? begin, int? end,
        IImageAttentionRule attention, Func<ITensorStore> open)
    {
        Encoder = encoder;
        Projector = projector;
        _stored = stored;
        _open = open;
        Attention = attention;
        Features = AudioFeatureExtractors.Create(encoder.FeatureExtractor, encoder.Features);
        PromptFormat = new ImageTokenFormat("idrak-audio", token, token, begin, end);
    }

    /// <summary>The encoder's settings.</summary>
    public IdrakAudioConfig Encoder { get; }

    /// <summary>The projector's settings.</summary>
    public IdrakAudioProjectorConfig Projector { get; }

    /// <inheritdoc />
    public override string Family => IdrakAudioLanguageFamily.Architecture;

    /// <inheritdoc />
    public override int Width => Projector.TextDim;

    /// <inheritdoc />
    public override IReadOnlySet<string> StoredTensors => _stored;

    /// <inheritdoc />
    public override IAudioFeatureExtractor Features { get; }

    /// <inheritdoc />
    public override IImagePromptFormat PromptFormat { get; }

    /// <inheritdoc />
    public override IImageAttentionRule Attention { get; }

    /// <summary>The trainable parts: "audio-encoder" and "audio-projector".</summary>
    public IReadOnlyList<string> TrainableParts { get; } = [IdrakAudioLanguageFamily.EncoderPart, IdrakAudioLanguageFamily.ProjectorPart];

    /// <inheritdoc />
    /// <exception cref="InvalidDataException">A tensor is missing, of the wrong shape, or not one the configuration reads.</exception>
    public override IAudioLanguageEncoder CreateEncoder(AudioEncoderOptions? options = null)
    {
        options ??= new AudioEncoderOptions();
        var device = options.Device ?? Device.Default;
        using var store = _open();
        var towerWeights = IdrakAudioWeights.Read(store, device, IdrakAudioLanguageFamily.TowerPrefix, options.Trained);
        var projectorWeights = IdrakAudioWeights.Read(store, device, IdrakAudioLanguageFamily.ProjectorPrefix, options.Trained);
        IdrakAudioEncoder? tower = null;
        IdrakAudioProjector? projector = null;
        try
        {
            tower = new IdrakAudioEncoder(Encoder, towerWeights);
            projector = new IdrakAudioProjector(Projector, Encoder.Dim, Encoder.Activation, projectorWeights);
            if (towerWeights.Unread().Concat(projectorWeights.Unread()).ToList() is { Count: > 0 } unread)
            {
                throw new InvalidDataException($"The audio part holds tensors its config does not use ({string.Join(", ", unread.Take(5))}{(unread.Count > 5 ? ", …" : "")}).");
            }

            return new IdrakAudioLanguageEncoder(this, tower, projector, [.. towerWeights.Entries], [.. projectorWeights.Entries], device);
        }
        catch
        {
            tower?.Dispose();
            projector?.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Train(IAudioLanguageEncoder encoder, IReadOnlyCollection<string> parts)
    {
        var own = Own(encoder);
        if (parts.FirstOrDefault(p => !TrainableParts.Contains(p, StringComparer.OrdinalIgnoreCase)) is { } unknown)
        {
            throw new ArgumentException($"The IdrakAudio audio part has no part '{unknown}' ({string.Join(", ", TrainableParts)}).", nameof(parts));
        }

        var chosen = new List<Tensor>();
        foreach (var (module, part) in new (Module, string)[] { (own.Tower, IdrakAudioLanguageFamily.EncoderPart), (own.Projector, IdrakAudioLanguageFamily.ProjectorPart) })
        {
            bool trains = parts.Contains(part, StringComparer.OrdinalIgnoreCase);
            foreach (var p in module.Parameters())
            {
                p.RequiresGrad = trains;
                if (trains)
                {
                    chosen.Add(p);
                }
            }
        }

        return chosen;
    }

    /// <inheritdoc />
    public void SetTraining(IAudioLanguageEncoder encoder, bool training)
    {
        var own = Own(encoder);
        foreach (var module in new Module[] { own.Tower, own.Projector })
        {
            if (training)
            {
                module.Train();
            }
            else
            {
                module.Eval();
            }
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<ImageFeatures> EncodeTraining(IAudioLanguageEncoder encoder, IReadOnlyList<AudioData> audios, bool checkpointing) =>
        Own(encoder).Run(audios, checkpointing);

    /// <inheritdoc />
    public IReadOnlyList<(string Name, int[] Shape, float[] Values)> Export(IAudioLanguageEncoder encoder, IReadOnlyCollection<string> parts)
    {
        var own = Own(encoder);
        var result = new List<(string, int[], float[])>();
        foreach (var (entries, part) in new[] { (own.TowerEntries, IdrakAudioLanguageFamily.EncoderPart), (own.ProjectorEntries, IdrakAudioLanguageFamily.ProjectorPart) })
        {
            if (parts.Contains(part, StringComparer.OrdinalIgnoreCase))
            {
                foreach (var (name, tensor, shape, transposed) in entries)
                {
                    result.Add((name, shape, IdrakAudioWeights.Stored(tensor, transposed)));
                }
            }
        }

        return result;
    }

    private IdrakAudioLanguageEncoder Own(IAudioLanguageEncoder encoder) => encoder is IdrakAudioLanguageEncoder own && ReferenceEquals(own.Part, this)
        ? own : throw new ArgumentException("The encoder is not this IdrakAudio audio part's (CreateEncoder).", nameof(encoder));
}

/// <summary>
/// The IdrakAudio audio-language encoder: each sound mixed to one channel at the features' rate, its log-mel features, the
/// encoder (a batch padded, each sound's own steps attended to), every <see cref="IdrakAudioProjectorConfig.Stack"/> steps
/// joined and projected: one block of soft tokens per sound, the same alone and in a batch.
/// </summary>
public sealed class IdrakAudioLanguageEncoder : IAudioLanguageEncoder
{
    internal IdrakAudioLanguageEncoder(IdrakAudioLanguagePart part, IdrakAudioEncoder tower, IdrakAudioProjector projector,
        IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> towerEntries,
        IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> projectorEntries, Device device)
    {
        (Part, Tower, Projector, TowerEntries, ProjectorEntries, Device) = (part, tower, projector, towerEntries, projectorEntries, device);
        tower.Eval();
        projector.Eval();
    }

    internal IdrakAudioLanguagePart Part { get; }

    internal IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> TowerEntries { get; }

    internal IReadOnlyList<(string Name, Tensor Tensor, int[] Stored, bool Transposed)> ProjectorEntries { get; }

    /// <summary>The encoder.</summary>
    public IdrakAudioEncoder Tower { get; }

    /// <summary>The projector.</summary>
    public IdrakAudioProjector Projector { get; }

    /// <inheritdoc />
    public int Width => Part.Width;

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public IReadOnlyList<ImageTokenLayout> Blocks(AudioData audio)
    {
        ArgumentNullException.ThrowIfNull(audio);
        int frames = Part.Features.FramesFor(audio.SampleRate == Part.Features.Options.SampleRate ? audio.Length : Prepare(audio).Length);
        return [new ImageTokenLayout(Part.Projector.TokensFor(Part.Encoder.StepsFor(frames)))];
    }

    /// <inheritdoc />
    public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<AudioData> audios)
    {
        using var noGrad = Autograd.NoGrad();
        return Run(audios, checkpointing: false);
    }

    // The sounds' soft-token features, recording gradients when they are on.
    internal IReadOnlyList<ImageFeatures> Run(IReadOnlyList<AudioData> audios, bool checkpointing)
    {
        ArgumentNullException.ThrowIfNull(audios);
        if (audios.Count == 0)
        {
            return [];
        }

        int mels = Part.Encoder.Mels, n = audios.Count;
        var features = new Tensor[n];
        var frames = new int[n];
        for (int i = 0; i < n; i++)
        {
            features[i] = Part.Features.Extract(Prepare(audios[i]), Device);       // [frames, mels]
            frames[i] = features[i].Shape[0];
        }

        int longest = frames.Max();
        var rows = new Tensor[n];
        for (int i = 0; i < n; i++)
        {
            var own = frames[i] == longest ? features[i] : Tensor.Concat([features[i], Tensor.Zeros([longest - frames[i], mels], Device)]);
            rows[i] = own.Reshape(1, longest, mels);
        }

        var batch = n == 1 ? rows[0] : Tensor.Concat(rows);
        var states = Tower.Run(batch, frames, checkpointing);                      // [n, steps, dim]
        int dim = Part.Encoder.Dim, stack = Part.Projector.Stack;
        var result = new ImageFeatures[n];
        for (int i = 0; i < n; i++)
        {
            int steps = Part.Encoder.StepsFor(frames[i]), tokens = Part.Projector.TokensFor(steps);
            var own = (n == 1 ? states : states.Narrow(0, i, 1)).Reshape(states.Shape[1], dim);
            own = own.Shape[0] == steps ? own : own.Narrow(0, 0, steps);
            if (tokens * stack > steps)
            {
                own = Tensor.Concat([own, Tensor.Zeros([tokens * stack - steps, dim], Device)]);
            }

            result[i] = new ImageFeatures(Projector.Forward(own.Reshape(tokens, stack * dim)), new ImageTokenLayout(tokens));
        }

        return result;
    }

    // One channel at the features' rate.
    private AudioData Prepare(AudioData audio)
    {
        if (audio.Channels > 1)
        {
            var mono = new float[audio.Length];
            audio.ToMono(mono);
            audio = new AudioData(mono, 1, audio.SampleRate);
        }

        return audio.SampleRate == Part.Features.Options.SampleRate ? audio : Resampler.Resample(audio, Part.Features.Options.SampleRate);
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Tower.Dispose();
        Projector.Dispose();
    }
}
