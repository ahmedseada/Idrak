// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Formats;
using Idrak.Layers;

namespace Idrak.VitsSpeech;

/// <summary>
/// Reads a checkpoint's tensors into the library's layers on one device. A convolution over time is the library's
/// <see cref="Conv2d"/> with a (1, k) kernel over <c>[1, C, 1, T]</c>; a weight-normed one is folded at load
/// (<c>g · v / ‖v‖</c>, once); a linear layer's <c>[out, in]</c> weight is read transposed to the library's <c>[in, out]</c>.
/// Loaded to be trained (<see cref="Trainable"/>), every tensor read becomes a parameter that takes gradients, a weight
/// norm stays its gains and directions (folded per pass, so training updates what the checkpoint stores), and each is
/// listed in <see cref="Trained"/> under its stored name, so the model saves back to the checkpoint's own layout.
/// </summary>
internal sealed class VitsWeights(ITensorStore store, Device device, bool trainable = false)
{
    private readonly List<VitsParameter> _trained = [];

    public Device Device { get; } = device;

    /// <summary>Whether the tensors are read as trained parameters.</summary>
    public bool Trainable { get; } = trainable;

    /// <summary>Every tensor read as a parameter (when <see cref="Trainable"/>), in reading order.</summary>
    public IReadOnlyList<VitsParameter> Trained => _trained;

    public bool Contains(string name) => store.Contains(name);

    public float[] Read(string name) =>
        store.Contains(name) ? store.Read(name) : throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");

    public int[] Shape(string name) =>
        store.Contains(name) ? store.ShapeOf(name) : throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");

    public Tensor Tensor(string name, params int[] expected)
    {
        var shape = Shape(name);
        if (expected.Length > 0 && !shape.AsSpan().SequenceEqual(expected))
        {
            throw new InvalidDataException($"'{name}' is {Abstraction.Tensor.FormatShape(shape)}, the configuration says {Abstraction.Tensor.FormatShape(expected)}.");
        }

        return Parameter(name, Read(name), shape, shape);
    }

    /// <summary>A tensor of <paramref name="shape"/> holding the values stored as <paramref name="name"/> (of <paramref name="stored"/>), a parameter when trainable.</summary>
    public Tensor Parameter(string name, float[] values, int[] shape, int[] stored, bool transposed = false)
    {
        var tensor = Abstraction.Tensor.Persistent(values, shape, Device, requiresGrad: Trainable);
        if (Trainable)
        {
            _trained.Add(new VitsParameter(name, tensor, stored, transposed));
        }

        return tensor;
    }

    /// <summary>A linear layer from <c>{prefix}.weight</c> [out, in] and <c>{prefix}.bias</c> when present.</summary>
    public Linear Linear(string prefix, int inFeatures, int outFeatures)
    {
        var shape = Shape(prefix + ".weight");
        if (shape.Length != 2 || shape[0] != outFeatures || shape[1] != inFeatures)
        {
            throw new InvalidDataException($"'{prefix}.weight' is {Abstraction.Tensor.FormatShape(shape)}, the configuration says [{outFeatures}, {inFeatures}].");
        }

        var weight = Parameter(prefix + ".weight", store.ReadTransposed(prefix + ".weight"), [inFeatures, outFeatures], shape, transposed: true);
        return Layers.Linear.FromWeights(weight, store.Contains(prefix + ".bias") ? Tensor(prefix + ".bias", outFeatures) : null);
    }

    public LayerNorm Norm(string prefix, int features, float epsilon) =>
        LayerNorm.FromWeights(Tensor(prefix + ".weight", features), Tensor(prefix + ".bias", features), epsilon);

    /// <summary>
    /// A convolution over time from <c>{prefix}.weight</c> [out, in / groups, k], or a weight norm's gains and directions
    /// when the checkpoint stores those (and <c>{prefix}.bias</c> when present).
    /// </summary>
    public Conv1d Conv(string prefix, int inChannels, int outChannels, int kernel, int padding, int dilation = 1, int groups = 1) =>
        Conv2(prefix, inChannels, outChannels, (1, kernel), (1, 1), (0, padding), (1, dilation), groups);

    /// <summary>A weight-normed convolution (<c>parametrizations.weight.original0</c> the gains, <c>original1</c> the directions, or the older <c>weight_g</c> and <c>weight_v</c>).</summary>
    public Conv1d WeightNormConv(string prefix, int inChannels, int outChannels, int kernel, int padding, int dilation = 1) =>
        Conv(prefix, inChannels, outChannels, kernel, padding, dilation);

    /// <summary>A 2-D convolution (the discriminators' (k, 1) filters over a period's rows) from <c>{prefix}</c>'s filters [out, in / groups, kh, kw].</summary>
    public Conv1d Conv2(string prefix, int inChannels, int outChannels, (int Height, int Width) kernel, (int Height, int Width) stride, (int Height, int Width) padding,
        (int Height, int Width) dilation, int groups = 1)
    {
        int columns = inChannels / groups * kernel.Height * kernel.Width;
        int[] expected = kernel.Height == 1 ? [outChannels, inChannels / groups, kernel.Width] : [outChannels, inChannels / groups, kernel.Height, kernel.Width];
        var (weight, gain) = Filters(prefix, expected, outChannels, columns);
        var bias = store.Contains(prefix + ".bias") ? Tensor(prefix + ".bias", outChannels) : null;
        return new Conv1d(Conv2d.FromWeights(weight, bias, kernel, stride, padding, dilation, groups), gain);
    }

    /// <summary>w[o] = g[o] · v[o] / ‖v[o]‖ over every dimension but the first (PyTorch's weight_norm with dim 0).</summary>
    public static float[] FoldWeightNorm(float[] gains, float[] directions, int[] shape)
    {
        int rows = shape[0], columns = directions.Length / rows;
        if (gains.Length != rows)
        {
            throw new InvalidDataException($"A weight norm's gains hold {gains.Length} values for {rows} filters.");
        }

        var folded = new float[directions.Length];
        for (int o = 0; o < rows; o++)
        {
            double norm = 0;
            var row = directions.AsSpan(o * columns, columns);
            for (int i = 0; i < columns; i++)
            {
                norm += (double)row[i] * row[i];
            }

            double scale = gains[o] / Math.Sqrt(norm);
            var target = folded.AsSpan(o * columns, columns);
            for (int i = 0; i < columns; i++)
            {
                target[i] = (float)(row[i] * scale);
            }
        }

        return folded;
    }

    /// <summary>A transposed convolution over time from <c>{prefix}.weight</c> [in, out, k] (or a weight norm's gains and directions) and <c>{prefix}.bias</c>.</summary>
    public ConvTranspose1d ConvTranspose(string prefix, int inChannels, int outChannels, int kernel, int stride, int padding)
    {
        var (weight, gain) = Filters(prefix, [inChannels, outChannels, kernel], inChannels, outChannels * kernel);
        var bias = store.Contains(prefix + ".bias") ? Tensor(prefix + ".bias", outChannels) : null;
        return new ConvTranspose1d(ConvTranspose2d.FromWeights(weight, bias, (1, kernel), (1, stride), (0, padding)), gain);
    }

    // A convolution's filters as [rows, columns] (rows the first stored dimension): {prefix}.weight, else a weight norm's
    // gains and directions, folded once (speaking) or kept as the directions and the gains [rows, 1] (training).
    private (Tensor Weight, Tensor? Gain) Filters(string prefix, int[] expected, int rows, int columns)
    {
        string? g = null, v = null;
        if (store.Contains(prefix + ".parametrizations.weight.original1"))
        {
            (g, v) = (prefix + ".parametrizations.weight.original0", prefix + ".parametrizations.weight.original1");
        }
        else if (store.Contains(prefix + ".weight_v"))
        {
            (g, v) = (prefix + ".weight_g", prefix + ".weight_v");
        }

        string name = v ?? prefix + ".weight";
        var shape = Shape(name);
        if (!shape.AsSpan().SequenceEqual(expected))
        {
            throw new InvalidDataException($"'{name}' is {Abstraction.Tensor.FormatShape(shape)}, the configuration says {Abstraction.Tensor.FormatShape(expected)}.");
        }

        if (g is null)
        {
            return (Parameter(name, Read(name), [rows, columns], shape), null);
        }

        if (!Trainable)
        {
            return (Abstraction.Tensor.Persistent(FoldWeightNorm(Read(g), Read(v!), shape), [rows, columns], Device), null);
        }

        var gains = Read(g);
        if (gains.Length != rows)
        {
            throw new InvalidDataException($"'{g}' holds {gains.Length} gains for {rows} filters.");
        }

        var direction = Parameter(v!, Read(v!), [rows, columns], shape);
        return (direction, Parameter(g, gains, [rows, 1], Shape(g)));
    }
}

/// <summary>A tensor a model trains, under the name and in the layout its checkpoint stores it (a linear weight transposed back to [out, in]).</summary>
/// <param name="Name">The stored name.</param>
/// <param name="Tensor">The parameter.</param>
/// <param name="Stored">The stored shape.</param>
/// <param name="Transposed">Whether the parameter is the stored [out, in] matrix transposed.</param>
internal sealed record VitsParameter(string Name, Tensor Tensor, int[] Stored, bool Transposed)
{
    /// <summary>The values in the stored layout.</summary>
    public float[] Values(Tensor? of = null)
    {
        var values = (of ?? Tensor).ToArray();
        return Transposed ? HostParallel.Transpose(values, Tensor.Shape[0], Tensor.Shape[1]) : values;
    }
}

/// <summary>A model's trained tensors as a <see cref="Module"/> (what the trainers take); its layers compute, it does not, and they own the tensors.</summary>
internal sealed class VitsParameterSet(string name, IReadOnlyList<VitsParameter> parameters) : Module
{
    private readonly HashSet<string> _frozen = new(StringComparer.Ordinal);

    /// <summary>Every tensor, frozen or not (what a checkpoint saves).</summary>
    public IReadOnlyList<VitsParameter> Named => parameters;

    /// <summary>The parts: the first segment of the tensors' names ("text_encoder", "flow", "decoder", …), in reading order.</summary>
    public IReadOnlyList<string> Parts => [.. parameters.Select(p => Part(p.Name)).Distinct(StringComparer.Ordinal)];

    /// <summary>Keeps these parts' tensors as they are: no gradient, and out of <see cref="Parameters"/>.</summary>
    public void Freeze(IReadOnlyCollection<string> parts)
    {
        var known = Parts;
        foreach (string part in parts)
        {
            if (!known.Contains(part, StringComparer.Ordinal))
            {
                throw new ArgumentException($"VITS has no part '{part}' to freeze; its parts are {string.Join(", ", known)}.", nameof(parts));
            }
        }

        foreach (var p in parameters.Where(p => parts.Contains(Part(p.Name))))
        {
            p.Tensor.RequiresGrad = false;
            _frozen.Add(p.Name);
        }
    }

    protected override Tensor ForwardCore(Tensor input) => throw new NotSupportedException($"{name} lists the trained tensors; the model's layers run them.");

    public override IEnumerable<Tensor> Parameters() => parameters.Where(p => !_frozen.Contains(p.Name)).Select(p => p.Tensor);

    private static string Part(string tensor) => tensor.IndexOf('.') is int dot and > 0 ? tensor[..dot] : tensor;

    public override string ToString() => name;

    public override void Dispose() => GC.SuppressFinalize(this);
}

/// <summary>
/// A convolution through the library's <see cref="Conv2d"/>: over time on <c>[C, T]</c> (as <c>[1, C, 1, T]</c>), or on
/// <c>[N, C, H, W]</c>. With a weight norm's gains (training), the filters are <c>g · v / ‖v‖</c> of the layer's
/// directions, made per pass so their gradients reach both.
/// </summary>
internal sealed class Conv1d(Conv2d conv, Tensor? gain = null) : IDisposable
{
    public int InChannels => conv.InChannels;

    public int OutChannels => conv.OutChannels;

    public Tensor Forward(Tensor x)
    {
        var y = Forward4(x.Reshape(1, x.Shape[0], 1, x.Shape[1]));
        return y.Reshape(y.Shape[1], y.Shape[3]);
    }

    public Tensor Forward4(Tensor x)
    {
        if (gain is null)
        {
            return conv.Forward(x);
        }

        return x.Convolution(VitsOps.WeightNorm(gain, conv.Weight), conv.Bias, conv.Geometry(x.Shape[0], x.Shape[2], x.Shape[3]), conv.Groups);
    }

    public void Dispose()
    {
        conv.Dispose();
        gain?.Dispose();
    }
}

/// <summary>A transposed convolution over time on <c>[C, T]</c>, through the library's <see cref="ConvTranspose2d"/> (weight-normed per pass in training, as <see cref="Conv1d"/>).</summary>
internal sealed class ConvTranspose1d(ConvTranspose2d conv, Tensor? gain = null) : IDisposable
{
    public int OutChannels => conv.OutChannels;

    public Tensor Forward(Tensor x)
    {
        var x4 = x.Reshape(1, x.Shape[0], 1, x.Shape[1]);
        var y = gain is null ? conv.Forward(x4) : x4.ConvolutionTranspose(VitsOps.WeightNorm(gain, conv.Weight), conv.Bias, conv.Geometry(1, 1, x.Shape[1]), conv.Groups);
        return y.Reshape(y.Shape[1], y.Shape[3]);
    }

    public void Dispose()
    {
        conv.Dispose();
        gain?.Dispose();
    }
}

/// <summary>Element-wise pieces written from public tensor operations, so they run on every device.</summary>
internal static class VitsOps
{
    /// <summary>max(x, slope · x): the leaky ReLU for a slope in [0, 1).</summary>
    public static Tensor LeakyRelu(Tensor x, float slope) => slope == 0 ? x.Relu() : x.Maximum(x * slope);

    /// <summary>The feed-forward activation a config.json names.</summary>
    public static Tensor Activation(string name, Tensor x) => name switch
    {
        "relu" => x.Relu(),
        "gelu" => ExactGELU.Apply(x),
        "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => x.Gelu(),
        "silu" or "swish" => x.Silu(),
        _ => throw new NotSupportedException($"hidden_act '{name}' is not supported (relu, gelu, gelu_new, silu)."),
    };

    /// <summary>x [C, T] padded with zeros along time: <paramref name="left"/> columns before, <paramref name="right"/> after.</summary>
    public static Tensor PadTime(Tensor x, int left, int right)
    {
        if (left == 0 && right == 0)
        {
            return x;
        }

        var parts = new List<Tensor>(3);
        if (left > 0)
        {
            parts.Add(Tensor.Zeros([x.Shape[0], left], x.Device));
        }

        parts.Add(x);
        if (right > 0)
        {
            parts.Add(Tensor.Zeros([x.Shape[0], right], x.Device));
        }

        return Tensor.Concat(parts, 1);
    }

    /// <summary>A column [C, 1] repeated over <paramref name="columns"/> columns: [C, columns].</summary>
    public static Tensor ExpandColumns(Tensor column, int columns) => column.MatMul(Tensor.Ones([1, columns], column.Device));

    /// <summary>g · v / ‖v‖ per row: the directions [rows, columns] scaled by the gains [rows, 1] over each row's norm (PyTorch's weight_norm, dim 0).</summary>
    public static Tensor WeightNorm(Tensor gain, Tensor direction)
    {
        var scale = gain * direction.Square().Sum(1, keepDim: true).Sqrt().Pow(-1f);   // [rows, 1]
        return direction * ExpandColumns(scale, direction.Shape[1]);
    }

    /// <summary>
    /// The samples [1, L] padded at the end by reflection to <paramref name="length"/> (torch's <c>F.pad(x, (0, n), "reflect")</c>),
    /// as a gather, so the padding's gradient reaches the samples it copies.
    /// </summary>
    public static Tensor ReflectEnd(Tensor samples, int length)
    {
        int n = samples.Shape[1];
        if (length == n)
        {
            return samples;
        }

        var rows = new int[length];
        for (int i = 0; i < length; i++)
        {
            rows[i] = i < n ? i : 2 * (n - 1) - i;
        }

        return samples.Reshape(n, 1).EmbeddingLookup(Indices(rows, samples.Device)).Reshape(1, length);
    }

    /// <summary>The samples [1, L] reflect-padded by <paramref name="pad"/> at both ends (the edge sample not repeated), as a gather.</summary>
    public static Tensor Reflect(Tensor samples, int pad)
    {
        int n = samples.Shape[1];
        var rows = new int[n + 2 * pad];
        for (int i = 0; i < rows.Length; i++)
        {
            int j = i - pad;
            rows[i] = j < 0 ? -j : j >= n ? 2 * (n - 1) - j : j;
        }

        return samples.Reshape(n, 1).EmbeddingLookup(Indices(rows, samples.Device)).Reshape(1, rows.Length);
    }

    /// <summary>A float tensor of integer indices [count] on <paramref name="device"/> (what the library's gathers take).</summary>
    public static Tensor Indices(ReadOnlySpan<int> indices, Device device)
    {
        var values = new float[indices.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = indices[i];
        }

        return Tensor.From(values, [values.Length], device);
    }

    /// <summary>The rows of x [R, C] in reverse order (torch.flip over the channel axis for channels-first data).</summary>
    public static Tensor FlipRows(Tensor x)
    {
        int rows = x.Shape[0];
        var reversed = new int[rows];
        for (int r = 0; r < rows; r++)
        {
            reversed[r] = rows - 1 - r;
        }

        return x.EmbeddingLookup(Indices(reversed, x.Device));
    }
}

/// <summary>
/// transformers' <c>VitsDilatedDepthSeparableConv</c>: per layer a depthwise dilated convolution, LayerNorm over the
/// channels, GELU, a pointwise convolution, LayerNorm, GELU, added to the input. The conditioning is added to the
/// input first.
/// </summary>
internal sealed class VitsDilatedDepthSeparableConv : IDisposable
{
    private readonly Conv1d[] _dilated, _pointwise;
    private readonly LayerNorm[] _norms1, _norms2;

    public VitsDilatedDepthSeparableConv(VitsWeights weights, string prefix, VitsConfig config)
    {
        int channels = config.HiddenSize, kernel = config.DurationKernelSize, layers = config.DepthSeparableLayers;
        _dilated = new Conv1d[layers];
        _pointwise = new Conv1d[layers];
        _norms1 = new LayerNorm[layers];
        _norms2 = new LayerNorm[layers];
        for (int i = 0; i < layers; i++)
        {
            int dilation = (int)Math.Pow(kernel, i), padding = (kernel * dilation - dilation) / 2;
            _dilated[i] = weights.Conv($"{prefix}.convs_dilated.{i}", channels, channels, kernel, padding, dilation, groups: channels);
            _pointwise[i] = weights.Conv($"{prefix}.convs_pointwise.{i}", channels, channels, 1, 0);
            _norms1[i] = weights.Norm($"{prefix}.norms_1.{i}", channels, config.LayerNormEps);
            _norms2[i] = weights.Norm($"{prefix}.norms_2.{i}", channels, config.LayerNormEps);
        }
    }

    /// <summary>x [C, T] (+ conditioning [C, T]) through the layers: [C, T].</summary>
    public Tensor Forward(Tensor x, Tensor? conditioning)
    {
        if (conditioning is not null)
        {
            x = x + conditioning;
        }

        for (int i = 0; i < _dilated.Length; i++)
        {
            var h = _dilated[i].Forward(x);
            h = ExactGELU.Apply(_norms1[i].Forward(h.Transpose()).Transpose());
            h = _pointwise[i].Forward(h);
            h = ExactGELU.Apply(_norms2[i].Forward(h.Transpose()).Transpose());
            x = x + h;
        }

        return x;
    }

    public void Dispose()
    {
        foreach (var c in _dilated) c.Dispose();
        foreach (var c in _pointwise) c.Dispose();
        foreach (var n in _norms1) n.Dispose();
        foreach (var n in _norms2) n.Dispose();
    }
}

/// <summary>
/// transformers' <c>VitsWaveNet</c>: per layer a dilated convolution to 2C channels (weight-normed, folded at load),
/// the speaker conditioning's slice added, tanh of the first half times sigmoid of the second, a pointwise convolution
/// whose first C channels go back into the input and whose rest (or all of it, in the last layer) sum into the output.
/// </summary>
internal sealed class VitsWaveNet : IDisposable
{
    private readonly int _channels;
    private readonly Conv1d[] _in, _resSkip;
    private readonly Conv1d? _cond;

    public VitsWaveNet(VitsWeights weights, string prefix, VitsConfig config, int layers)
    {
        _channels = config.HiddenSize;
        int kernel = config.WaveNetKernelSize;
        _in = new Conv1d[layers];
        _resSkip = new Conv1d[layers];
        for (int i = 0; i < layers; i++)
        {
            int dilation = (int)Math.Pow(config.WaveNetDilationRate, i), padding = (kernel * dilation - dilation) / 2;
            _in[i] = weights.WeightNormConv($"{prefix}.in_layers.{i}", _channels, 2 * _channels, kernel, padding, dilation);
            int outChannels = i < layers - 1 ? 2 * _channels : _channels;
            _resSkip[i] = weights.WeightNormConv($"{prefix}.res_skip_layers.{i}", _channels, outChannels, 1, 0);
        }

        if (config.SpeakerEmbeddingSize > 0)
        {
            _cond = weights.WeightNormConv($"{prefix}.cond_layer", config.SpeakerEmbeddingSize, 2 * _channels * layers, 1, 0);
        }
    }

    /// <summary>x [C, T] with the speaker's embedding [E, 1] (null for one speaker): the skip sum [C, T].</summary>
    public Tensor Forward(Tensor x, Tensor? speaker)
    {
        int t = x.Shape[1];
        Tensor? conditioning = speaker is not null && _cond is not null ? _cond.Forward(speaker) : null;   // [2C · layers, 1]
        Tensor? output = null;
        for (int i = 0; i < _in.Length; i++)
        {
            var h = _in[i].Forward(x);                                           // [2C, T]
            if (conditioning is not null)
            {
                h = h + VitsOps.ExpandColumns(conditioning.Narrow(0, i * 2 * _channels, 2 * _channels), t);
            }

            var acts = h.Narrow(0, 0, _channels).Tanh() * h.Narrow(0, _channels, _channels).Sigmoid();
            var resSkip = _resSkip[i].Forward(acts);
            if (i < _in.Length - 1)
            {
                x = x + resSkip.Narrow(0, 0, _channels);
                var skip = resSkip.Narrow(0, _channels, _channels);
                output = output is null ? skip : output + skip;
            }
            else
            {
                output = output is null ? resSkip : output + resSkip;
            }
        }

        return output!;
    }

    public void Dispose()
    {
        foreach (var c in _in) c.Dispose();
        foreach (var c in _resSkip) c.Dispose();
        _cond?.Dispose();
    }
}
