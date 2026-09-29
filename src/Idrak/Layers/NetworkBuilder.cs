using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Layers;

/// <summary>
/// Entry points of the fluent network builder. Each builder method creates exactly one of the existing layers with the
/// same parameters and the same defaults as its constructor; the only value the builder supplies is a layer's input
/// size, taken from the output shape of the previous layer. <see cref="NetworkBuilder.Build"/> returns an ordinary
/// <see cref="Sequential"/>, so everything else (training, saving, devices) works unchanged.
/// </summary>
/// <example>
/// <code>
/// var model = Network.Input(9).Linear(64).ReLU().Linear(32).ReLU().Linear(1).Build();
/// // is the same network as
/// var same = new Sequential { new Linear(9, 64), new ReLU(), new Linear(64, 32), new ReLU(), new Linear(32, 1) };
/// </code>
/// </example>
public static class Network
{
    /// <summary>Rows of <paramref name="features"/> numbers: batches of shape [N, features].</summary>
    public static NetworkBuilder Input(int features) => new(InputKind.Features, [Positive(features)]);

    /// <summary>Images: batches of shape [N, channels, height, width].</summary>
    public static NetworkBuilder Image(int channels, int height, int width) =>
        new(InputKind.Image, [Positive(channels), Positive(height), Positive(width)]);

    /// <summary>Token ids: batches of shape [N, length] (for <see cref="NetworkBuilder.Embedding"/>).</summary>
    public static NetworkBuilder Tokens(int length) => new(InputKind.Tokens, [Positive(length)]);

    /// <summary>Sequences of feature vectors: batches of shape [N, length, features] (e.g. time-series windows).</summary>
    public static NetworkBuilder Sequence(int length, int features) => new(InputKind.Sequence, [Positive(length), Positive(features)]);

    /// <summary>Recreates a builder from the description written by <see cref="NetworkBuilder.ToJson"/>.</summary>
    public static NetworkBuilder FromJson(JsonNode description) => NetworkBuilder.Replay(description);

    /// <summary>The description of a network made by <see cref="NetworkBuilder.Build"/>, or null for networks built another way.</summary>
    public static JsonObject? ArchitectureOf(Module model) =>
        NetworkBuilder.Architectures.TryGetValue(model, out var description) ? (JsonObject)description.DeepClone() : null;

    private static int Positive(int value)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(value);
        return value;
    }
}

/// <summary>What the network's input batches contain.</summary>
public enum InputKind
{
    /// <summary>[N, features] numbers.</summary>
    Features,

    /// <summary>[N, channels, height, width] images.</summary>
    Image,

    /// <summary>[N, length] token ids.</summary>
    Tokens,

    /// <summary>[N, length, features] sequences.</summary>
    Sequence,
}

/// <summary>A recurrent cell type for <see cref="Architectures.Rnn"/>.</summary>
public enum RecurrentCell
{
    /// <summary><see cref="Layers.LSTM"/>.</summary>
    LSTM,

    /// <summary><see cref="Layers.GRU"/>.</summary>
    GRU,
}

/// <summary>An activation layer for the builder and <see cref="Architectures"/>.</summary>
public enum Activation
{
    /// <summary><see cref="Layers.ReLU"/>.</summary>
    ReLU,

    /// <summary><see cref="Layers.Tanh"/>.</summary>
    Tanh,

    /// <summary><see cref="Layers.Sigmoid"/>.</summary>
    Sigmoid,

    /// <summary><see cref="Layers.GELU"/>.</summary>
    GELU,
}

/// <summary>
/// Builds a <see cref="Sequential"/> step by step, tracking the shape of one sample (without the batch dimension) so
/// each layer's input size comes from the previous layer. Every method corresponds to one layer constructor; see
/// <see cref="Network"/> for the entry points. Builders made only of layer steps can be written to JSON and replayed
/// (<see cref="ToJson"/>, <see cref="Network.FromJson"/>); model packages use this to store the architecture.
/// </summary>
public sealed class NetworkBuilder
{
    internal static readonly ConditionalWeakTable<Module, JsonObject> Architectures = new();

    private readonly List<(Func<Module> Create, JsonObject? Step)> _steps = [];
    private readonly InputKind _kind;
    private readonly int[] _input;
    private int[] _shape;
    private Device? _device;
    private Random? _random;
    private int? _seed;
    private string? _name;
    private bool _describable = true;
    private Random? _buildRandom;

    internal NetworkBuilder(InputKind kind, int[] input)
    {
        _kind = kind;
        _input = input;
        _shape = input;
    }

    /// <summary>What the input batches contain.</summary>
    public InputKind InputKind => _kind;

    /// <summary>The shape of one input sample (without the batch dimension).</summary>
    public IReadOnlyList<int> InputShape => _input;

    /// <summary>The shape of one sample after the steps added so far (without the batch dimension).</summary>
    public IReadOnlyList<int> CurrentShape => _shape;

    /// <summary>Number of layers added so far.</summary>
    public int Count => _steps.Count;

    /// <summary>Whether every step can be written to JSON (false after <see cref="Lambda"/> or <see cref="Add"/>).</summary>
    public bool IsDescribable => _describable;

    // ------------------------------------------------------------------ settings passed to every layer

    /// <summary>Passes <paramref name="device"/> as the <c>device</c> argument of every layer (as writing it on each constructor).</summary>
    public NetworkBuilder OnDevice(Device device)
    {
        _device = device;
        return this;
    }

    /// <summary>Passes one <c>new Random(seed)</c> as the <c>random</c> argument of every layer, in order (reproducible weights).</summary>
    public NetworkBuilder Seed(int seed)
    {
        _seed = seed;
        _random = null;
        return this;
    }

    /// <summary>Passes <paramref name="random"/> as the <c>random</c> argument of every layer, in order.</summary>
    public NetworkBuilder WithRandom(Random random)
    {
        _random = random;
        _seed = null;
        return this;
    }

    /// <summary>Sets <see cref="Module.Name"/> of the built <see cref="Sequential"/>.</summary>
    public NetworkBuilder Named(string name)
    {
        _name = name;
        return this;
    }

    // ------------------------------------------------------------------ dense and activations

    /// <summary><c>new Layers.Linear(in, outFeatures, bias)</c>; <c>in</c> is the last dimension of the current shape (or <paramref name="inputs"/>, checked).</summary>
    public NetworkBuilder Linear(int outFeatures, bool bias = true, int? inputs = null)
    {
        int inFeatures = Last("Linear", inputs);
        return Push(r => new Layers.Linear(inFeatures, outFeatures, bias, _device, r), [.. _shape[..^1], outFeatures],
            Step("linear", ("out", outFeatures), ("bias", bias)));
    }

    /// <summary><c>new Layers.ReLU()</c>.</summary>
    public NetworkBuilder ReLU() => Push(_ => new Layers.ReLU(), _shape, Step("relu"));

    /// <summary><c>new Layers.Tanh()</c>.</summary>
    public NetworkBuilder Tanh() => Push(_ => new Layers.Tanh(), _shape, Step("tanh"));

    /// <summary><c>new Layers.Sigmoid()</c>.</summary>
    public NetworkBuilder Sigmoid() => Push(_ => new Layers.Sigmoid(), _shape, Step("sigmoid"));

    /// <summary><c>new Layers.GELU()</c>.</summary>
    public NetworkBuilder GELU() => Push(_ => new Layers.GELU(), _shape, Step("gelu"));

    /// <summary><c>new Layers.Softmax()</c>.</summary>
    public NetworkBuilder Softmax() => Push(_ => new Layers.Softmax(), _shape, Step("softmax"));

    /// <summary>One of the activation layers.</summary>
    public NetworkBuilder Activate(Activation activation) => activation switch
    {
        Layers.Activation.ReLU => ReLU(),
        Layers.Activation.Tanh => Tanh(),
        Layers.Activation.Sigmoid => Sigmoid(),
        Layers.Activation.GELU => GELU(),
        _ => throw new ArgumentOutOfRangeException(nameof(activation)),
    };

    /// <summary><c>new Layers.Dropout(probability, random)</c>.</summary>
    public NetworkBuilder Dropout(float probability = 0.5f) =>
        Push(r => new Layers.Dropout(probability, r), _shape, Step("dropout", ("p", probability)));

    // ------------------------------------------------------------------ normalization

    /// <summary><c>new Layers.BatchNorm(channels, momentum, epsilon)</c>; channels are the features of [F] or the channels of [C, H, W].</summary>
    public NetworkBuilder BatchNorm(float momentum = 0.1f, float epsilon = 1e-5f)
    {
        if (_shape.Length is not (1 or 3))
        {
            throw new InvalidOperationException($"BatchNorm needs [features] or [channels, height, width], the current shape is {Tensor.FormatShape(_shape)}.");
        }

        int channels = _shape[0];
        return Push(_ => new Layers.BatchNorm(channels, momentum, epsilon, _device), _shape, Step("batchnorm", ("momentum", momentum), ("epsilon", epsilon)));
    }

    /// <summary><c>new Layers.LayerNorm(features, epsilon)</c>; features are the last dimension.</summary>
    public NetworkBuilder LayerNorm(float epsilon = 1e-5f)
    {
        int features = Last("LayerNorm", null);
        return Push(_ => new Layers.LayerNorm(features, epsilon, _device), _shape, Step("layernorm", ("epsilon", epsilon)));
    }

    // ------------------------------------------------------------------ images

    /// <summary><c>new Layers.Conv2d(channels, outChannels, kernelSize, stride, padding, bias)</c>; channels from the current [C, H, W].</summary>
    public NetworkBuilder Conv2d(int outChannels, int kernelSize, int stride = 1, int padding = 0, bool bias = true)
    {
        var (c, h, w) = Image("Conv2d");
        int oh = (h + 2 * padding - kernelSize) / stride + 1, ow = (w + 2 * padding - kernelSize) / stride + 1;
        CheckSpatial("Conv2d", oh, ow);
        return Push(r => new Layers.Conv2d(c, outChannels, kernelSize, stride, padding, bias, _device, r), [outChannels, oh, ow],
            Step("conv2d", ("out", outChannels), ("kernel", kernelSize), ("stride", stride), ("padding", padding), ("bias", bias)));
    }

    /// <summary><c>new Layers.MaxPool2d(kernelSize, stride, padding)</c>.</summary>
    public NetworkBuilder MaxPool2d(int kernelSize, int? stride = null, int padding = 0)
    {
        var (c, h, w) = Image("MaxPool2d");
        int s = stride ?? kernelSize;
        int oh = (h + 2 * padding - kernelSize) / s + 1, ow = (w + 2 * padding - kernelSize) / s + 1;
        CheckSpatial("MaxPool2d", oh, ow);
        var step = Step("maxpool2d", ("kernel", kernelSize), ("padding", padding));
        if (stride is { } explicitStride)
        {
            step["stride"] = explicitStride;
        }

        return Push(_ => new Layers.MaxPool2d(kernelSize, stride, padding), [c, oh, ow], step);
    }

    /// <summary><c>new Layers.GlobalAveragePool2d()</c>: [C, H, W] → [C].</summary>
    public NetworkBuilder GlobalAveragePool2d()
    {
        var (c, _, _) = Image("GlobalAveragePool2d");
        return Push(_ => new Layers.GlobalAveragePool2d(), [c], Step("globalavgpool2d"));
    }

    /// <summary><c>new Layers.Flatten()</c>: any shape → [product of its dimensions].</summary>
    public NetworkBuilder Flatten() => Push(_ => new Layers.Flatten(), [_shape.Aggregate(1, (a, b) => a * b)], Step("flatten"));

    // ------------------------------------------------------------------ sequences and text

    /// <summary><c>new Layers.Embedding(vocabulary, dim)</c>: token ids [T] → [T, dim].</summary>
    public NetworkBuilder Embedding(int vocabulary, int dim)
    {
        if (_kind != InputKind.Tokens || _steps.Count > 0)
        {
            throw new InvalidOperationException("Embedding must be the first layer after Network.Tokens(length).");
        }

        return Push(r => new Layers.Embedding(vocabulary, dim, _device, r), [_shape[0], dim], Step("embedding", ("vocabulary", vocabulary), ("dim", dim)));
    }

    /// <summary><c>new Layers.PositionalEncoding(maxLength, dim)</c>; <c>maxLength</c> is the sequence length (or <paramref name="maxLength"/>), <c>dim</c> the last dimension.</summary>
    public NetworkBuilder PositionalEncoding(int? maxLength = null)
    {
        var (t, d) = Sequence("PositionalEncoding");
        int length = maxLength ?? t;
        var step = Step("positional");
        if (maxLength is { } explicitLength)
        {
            step["maxLength"] = explicitLength;
        }

        return Push(_ => new Layers.PositionalEncoding(length, d, _device), _shape, step);
    }

    /// <summary><c>new Layers.TransformerEncoderLayer(dim, heads, ffDim, dropout, causal)</c>; <c>dim</c> is the last dimension.</summary>
    public NetworkBuilder TransformerEncoderLayer(int heads, int? ffDim = null, float dropout = 0.1f, bool causal = false)
    {
        var (_, d) = Sequence("TransformerEncoderLayer");
        var step = Step("transformer", ("heads", heads), ("dropout", dropout), ("causal", causal));
        if (ffDim is { } ff)
        {
            step["ffDim"] = ff;
        }

        return Push(r => new Layers.TransformerEncoderLayer(d, heads, ffDim, dropout, causal, _device, r), _shape, step);
    }

    /// <summary><c>new Layers.MultiHeadAttention(dim, heads, causal, dropout)</c>; <c>dim</c> is the last dimension.</summary>
    public NetworkBuilder MultiHeadAttention(int heads, bool causal = false, float dropout = 0f)
    {
        var (_, d) = Sequence("MultiHeadAttention");
        return Push(r => new Layers.MultiHeadAttention(d, heads, causal, dropout, _device, r), _shape,
            Step("attention", ("heads", heads), ("causal", causal), ("dropout", dropout)));
    }

    /// <summary><c>new Layers.LSTM(features, hiddenSize, returnSequences)</c>: [T, F] → [hidden], or [T, hidden] with <paramref name="returnSequences"/>.</summary>
    public NetworkBuilder LSTM(int hiddenSize, bool returnSequences = false)
    {
        var (t, f) = Sequence("LSTM");
        return Push(r => new Layers.LSTM(f, hiddenSize, returnSequences, _device, r), returnSequences ? [t, hiddenSize] : [hiddenSize],
            Step("lstm", ("hidden", hiddenSize), ("returnSequences", returnSequences)));
    }

    /// <summary><c>new Layers.GRU(features, hiddenSize, returnSequences)</c>: [T, F] → [hidden], or [T, hidden] with <paramref name="returnSequences"/>.</summary>
    public NetworkBuilder GRU(int hiddenSize, bool returnSequences = false)
    {
        var (t, f) = Sequence("GRU");
        return Push(r => new Layers.GRU(f, hiddenSize, returnSequences, _device, r), returnSequences ? [t, hiddenSize] : [hiddenSize],
            Step("gru", ("hidden", hiddenSize), ("returnSequences", returnSequences)));
    }

    /// <summary>A <see cref="Layers.Lambda"/> averaging over the time dimension: [T, F] → [F] (<c>x.Mean(1)</c>).</summary>
    public NetworkBuilder MeanOverTime()
    {
        var (_, f) = Sequence("MeanOverTime");
        return Push(_ => new Layers.Lambda(x => x.Mean(1), "MeanOverTime"), [f], Step("meanOverTime"));
    }

    /// <summary>A <see cref="Layers.Lambda"/> keeping the last time step: [T, F] → [F].</summary>
    public NetworkBuilder LastStep()
    {
        var (t, f) = Sequence("LastStep");
        return Push(_ => new Layers.Lambda(x => x.Narrow(1, t - 1, 1).Reshape(-1, f), "LastStep"), [f], Step("lastStep"));
    }

    /// <summary>A <see cref="Layers.Lambda"/> keeping the first time step (e.g. a &lt;cls&gt; token): [T, F] → [F].</summary>
    public NetworkBuilder FirstStep()
    {
        var (_, f) = Sequence("FirstStep");
        return Push(_ => new Layers.Lambda(x => x.Narrow(1, 0, 1).Reshape(-1, f), "FirstStep"), [f], Step("firstStep"));
    }

    /// <summary>A <see cref="Layers.Lambda"/> reshaping each sample to <paramref name="shape"/> (the batch dimension is kept).</summary>
    public NetworkBuilder Reshape(params int[] shape)
    {
        if (shape.Aggregate(1, (a, b) => a * b) != _shape.Aggregate(1, (a, b) => a * b))
        {
            throw new InvalidOperationException($"Cannot reshape {Tensor.FormatShape(_shape)} to {Tensor.FormatShape(shape)}.");
        }

        int[] target = [.. shape];
        var step = Step("reshape");
        step["shape"] = new JsonArray([.. target.Select(v => (JsonNode)v)]);
        return Push(_ => new Layers.Lambda(x => x.Reshape([x.Shape[0], .. target]), "Reshape"), target, step);
    }

    // ------------------------------------------------------------------ anything else

    /// <summary>
    /// <c>new Layers.Lambda(function, name)</c>. The builder cannot know what the function does to the shape, so the shape of
    /// one output sample is required. A builder with a lambda cannot be written to JSON.
    /// </summary>
    public NetworkBuilder Lambda(Func<Tensor, Tensor> function, string name, int[] outputShape)
    {
        _describable = false;
        return Push(_ => new Layers.Lambda(function, name), [.. outputShape], null);
    }

    /// <summary>
    /// Appends an existing module (a custom layer, or a <see cref="Sequential"/> block). Its output shape is required.
    /// A builder with such a module cannot be written to JSON.
    /// </summary>
    public NetworkBuilder Add(Module module, int[] outputShape)
    {
        ArgumentNullException.ThrowIfNull(module);
        _describable = false;
        return Push(_ => module, [.. outputShape], null);
    }

    /// <summary>Applies a reusable block of builder steps (a function that adds layers and returns the builder).</summary>
    public NetworkBuilder Apply(Func<NetworkBuilder, NetworkBuilder> block) => block(this);

    /// <summary>Applies <paramref name="block"/> <paramref name="count"/> times, e.g. <c>.Repeat(3, b =&gt; b.TransformerEncoderLayer(4))</c>.</summary>
    public NetworkBuilder Repeat(int count, Func<NetworkBuilder, NetworkBuilder> block)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        for (int i = 0; i < count; i++)
        {
            block(this);
        }

        return this;
    }

    // ------------------------------------------------------------------ build and describe

    /// <summary>
    /// Creates the layers in order and returns them as a <see cref="Sequential"/>. Each call creates new layers (with
    /// <see cref="Seed"/>, the same initial weights every time).
    /// </summary>
    public Sequential Build()
    {
        if (_steps.Count == 0)
        {
            throw new InvalidOperationException("The network has no layers.");
        }

        _buildRandom = _seed is { } seed ? new Random(seed) : _random;
        var model = new Sequential(_steps.Select(s => s.Create()).ToList()) { Name = _name };
        _buildRandom = null;
        if (_describable)
        {
            Architectures.AddOrUpdate(model, ToJson());
        }

        return model;
    }

    /// <summary>The builder as JSON: input kind and shape, name, seed and every step with its arguments.</summary>
    public JsonObject ToJson()
    {
        if (!_describable)
        {
            throw new InvalidOperationException("This network contains a Lambda or an added module, which cannot be written to JSON.");
        }

        var json = new JsonObject
        {
            ["format"] = "idrak-network/1",
            ["input"] = _kind.ToString(),
            ["shape"] = new JsonArray([.. _input.Select(v => (JsonNode)v)]),
            ["steps"] = new JsonArray([.. _steps.Select(s => (JsonNode)s.Step!.DeepClone())]),
        };
        if (_name is not null)
        {
            json["name"] = _name;
        }

        if (_seed is { } seed)
        {
            json["seed"] = seed;
        }

        return json;
    }

    internal static NetworkBuilder Replay(JsonNode description)
    {
        if ((string?)description["format"] != "idrak-network/1")
        {
            throw new InvalidDataException("Not a Idrak network description (format idrak-network/1).");
        }

        var kind = Enum.Parse<InputKind>((string)description["input"]!);
        int[] shape = [.. description["shape"]!.AsArray().Select(v => (int)v!)];
        var b = kind switch
        {
            InputKind.Features => Network.Input(shape[0]),
            InputKind.Image => Network.Image(shape[0], shape[1], shape[2]),
            InputKind.Tokens => Network.Tokens(shape[0]),
            _ => Network.Sequence(shape[0], shape[1]),
        };
        if ((string?)description["name"] is { } name)
        {
            b.Named(name);
        }

        if ((int?)description["seed"] is { } seed)
        {
            b.Seed(seed);
        }

        foreach (var node in description["steps"]!.AsArray())
        {
            var s = node!.AsObject();
            int I(string key) => (int)s[key]!;
            float F(string key) => (float)s[key]!;
            bool B(string key) => (bool)s[key]!;
            int? N(string key) => (int?)s[key];
            _ = (string)s["op"]! switch
            {
                "linear" => b.Linear(I("out"), B("bias")),
                "relu" => b.ReLU(),
                "tanh" => b.Tanh(),
                "sigmoid" => b.Sigmoid(),
                "gelu" => b.GELU(),
                "softmax" => b.Softmax(),
                "dropout" => b.Dropout(F("p")),
                "batchnorm" => b.BatchNorm(F("momentum"), F("epsilon")),
                "layernorm" => b.LayerNorm(F("epsilon")),
                "conv2d" => b.Conv2d(I("out"), I("kernel"), I("stride"), I("padding"), B("bias")),
                "maxpool2d" => b.MaxPool2d(I("kernel"), N("stride"), I("padding")),
                "globalavgpool2d" => b.GlobalAveragePool2d(),
                "flatten" => b.Flatten(),
                "embedding" => b.Embedding(I("vocabulary"), I("dim")),
                "positional" => b.PositionalEncoding(N("maxLength")),
                "transformer" => b.TransformerEncoderLayer(I("heads"), N("ffDim"), F("dropout"), B("causal")),
                "attention" => b.MultiHeadAttention(I("heads"), B("causal"), F("dropout")),
                "lstm" => b.LSTM(I("hidden"), B("returnSequences")),
                "gru" => b.GRU(I("hidden"), B("returnSequences")),
                "meanOverTime" => b.MeanOverTime(),
                "lastStep" => b.LastStep(),
                "firstStep" => b.FirstStep(),
                "reshape" => b.Reshape([.. s["shape"]!.AsArray().Select(v => (int)v!)]),
                var op => throw new InvalidDataException($"Unknown network step '{op}'."),
            };
        }

        return b;
    }

    /// <summary>The steps so far, one per line (a quick check before <see cref="Build"/>).</summary>
    public override string ToString() =>
        $"Network({_kind} {Tensor.FormatShape(_input)} → {Tensor.FormatShape(_shape)}, {_steps.Count} layers)";

    private NetworkBuilder Push(Func<Random?, Module> create, int[] output, JsonObject? step)
    {
        _steps.Add((() => create(_buildRandom), step));
        _shape = output;
        return this;
    }

    private static JsonObject Step(string op, params (string Key, JsonNode? Value)[] args)
    {
        var step = new JsonObject { ["op"] = op };
        foreach (var (key, value) in args)
        {
            step[key] = value;
        }

        return step;
    }

    private int Last(string layer, int? inputs)
    {
        if (_kind == InputKind.Tokens && _steps.Count == 0)
        {
            throw new InvalidOperationException($"{layer} cannot read token ids; start with Embedding(vocabulary, dim).");
        }

        int last = _shape[^1];
        if (inputs is { } given && given != last)
        {
            throw new InvalidOperationException($"{layer} was given {given} inputs, but the previous layer produces {last} (shape {Tensor.FormatShape(_shape)}).");
        }

        return last;
    }

    private (int C, int H, int W) Image(string layer) => _shape.Length == 3
        ? (_shape[0], _shape[1], _shape[2])
        : throw new InvalidOperationException($"{layer} needs [channels, height, width], the current shape is {Tensor.FormatShape(_shape)}.");

    private (int T, int F) Sequence(string layer) => _shape.Length == 2 && !(_kind == InputKind.Tokens && _steps.Count == 0)
        ? (_shape[0], _shape[1])
        : throw new InvalidOperationException($"{layer} needs [length, features], the current shape is {Tensor.FormatShape(_shape)}.");

    private static void CheckSpatial(string layer, int h, int w)
    {
        if (h <= 0 || w <= 0)
        {
            throw new InvalidOperationException($"{layer} would produce an empty {h}x{w} output.");
        }
    }
}

/// <summary>Reusable groups of layers for the collection-initializer style (<c>new Sequential { ... }</c>).</summary>
public static class Blocks
{
    /// <summary>A <see cref="Sequential"/> of <paramref name="count"/> layers made by <paramref name="factory"/>.</summary>
    public static Sequential Repeat(int count, Func<Module> factory) => Repeat(count, _ => factory());

    /// <summary>A <see cref="Sequential"/> of <paramref name="count"/> layers; the factory receives the index.</summary>
    public static Sequential Repeat(int count, Func<int, Module> factory)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        return new Sequential(Enumerable.Range(0, count).Select(factory).ToList());
    }
}

/// <summary>
/// Ready-made recipes (the Director of the builder pattern): each is a fixed, documented sequence of builder steps
/// with every size given by the caller. They return the builder, so settings such as <see cref="NetworkBuilder.Seed"/>
/// can still be added before <see cref="NetworkBuilder.Build"/>.
/// </summary>
public static class Architectures
{
    /// <summary>Input(inputs) → for each hidden size: Linear, activation → Linear(outputs).</summary>
    public static NetworkBuilder Mlp(int inputs, int[] hidden, int outputs, Activation activation)
    {
        var b = Network.Input(inputs);
        foreach (int h in hidden)
        {
            b.Linear(h).Activate(activation);
        }

        return b.Linear(outputs);
    }

    /// <summary>As <see cref="Mlp(int, int[], int, Activation)"/> with Dropout(dropout) after each activation.</summary>
    public static NetworkBuilder Mlp(int inputs, int[] hidden, int outputs, Activation activation, float dropout)
    {
        var b = Network.Input(inputs);
        foreach (int h in hidden)
        {
            b.Linear(h).Activate(activation).Dropout(dropout);
        }

        return b.Linear(outputs);
    }

    /// <summary>
    /// Image(channels, height, width) → for each filter count: Conv2d(filters, kernelSize, padding: kernelSize / 2),
    /// BatchNorm, ReLU, MaxPool2d(2) → Flatten → Linear(outputs).
    /// </summary>
    public static NetworkBuilder Cnn(int channels, int height, int width, int[] filters, int kernelSize, int outputs)
    {
        var b = Network.Image(channels, height, width);
        foreach (int f in filters)
        {
            b.Conv2d(f, kernelSize, padding: kernelSize / 2).BatchNorm().ReLU().MaxPool2d(2);
        }

        return b.Flatten().Linear(outputs);
    }

    /// <summary>Tokens(length) → Embedding(vocabulary, embed) → LSTM or GRU(hidden) → Linear(outputs).</summary>
    public static NetworkBuilder Rnn(RecurrentCell cell, int vocabulary, int length, int embed, int hidden, int outputs)
    {
        var b = Network.Tokens(length).Embedding(vocabulary, embed);
        b = cell == RecurrentCell.LSTM ? b.LSTM(hidden) : b.GRU(hidden);
        return b.Linear(outputs);
    }

    /// <summary>
    /// Tokens(length) → Embedding(vocabulary, dim) → PositionalEncoding → layers × TransformerEncoderLayer(heads, ffDim,
    /// dropout) → LayerNorm → MeanOverTime → Linear(outputs).
    /// </summary>
    public static NetworkBuilder TransformerClassifier(int vocabulary, int length, int dim, int heads, int layers, int ffDim, float dropout, int outputs) =>
        Network.Tokens(length).Embedding(vocabulary, dim).PositionalEncoding()
            .Repeat(layers, b => b.TransformerEncoderLayer(heads, ffDim, dropout))
            .LayerNorm().MeanOverTime().Linear(outputs);

    /// <summary>
    /// A decoder-only language model: Tokens(context) → Embedding(vocabulary, dim) → PositionalEncoding → layers ×
    /// causal TransformerEncoderLayer(heads, ffDim, dropout) → LayerNorm → Linear(vocabulary). Same layers as the GPT
    /// samples.
    /// </summary>
    public static NetworkBuilder Gpt(int vocabulary, int context, int dim, int heads, int layers, int ffDim, float dropout) =>
        Network.Tokens(context).Embedding(vocabulary, dim).PositionalEncoding()
            .Repeat(layers, b => b.TransformerEncoderLayer(heads, ffDim, dropout, causal: true))
            .LayerNorm().Linear(vocabulary);
}
