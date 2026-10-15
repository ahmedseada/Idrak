// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Formats;
using Idrak.Models;

namespace Idrak.VitsSpeech;

/// <summary>
/// The discriminators' shapes (a model folder's discriminator.json): VITS's multi-period discriminator by default (one
/// scale discriminator, one per period 2, 3, 5, 7, 11, at its widths); <see cref="Scales"/> above 1 adds HiFi-GAN's
/// multi-scale discriminators, each reading the sound average-pooled once more. Every value is a setting.
/// </summary>
public sealed record VitsDiscriminatorConfig
{
    /// <summary>The periods of the period discriminators (the sound folded into rows of this many samples).</summary>
    public int[] Periods { get; init; } = [2, 3, 5, 7, 11];

    /// <summary>The period discriminators' channels per convolution (each a (k, 1) filter at stride (s, 1), the last at stride 1).</summary>
    public int[] PeriodChannels { get; init; } = [32, 128, 512, 1024, 1024];

    /// <summary>The period convolutions' kernel (k).</summary>
    public int PeriodKernel { get; init; } = 5;

    /// <summary>The period convolutions' stride (s).</summary>
    public int PeriodStride { get; init; } = 3;

    /// <summary>The output convolution's kernel (both kinds).</summary>
    public int PostKernel { get; init; } = 3;

    /// <summary>The scale discriminators (the first reads the sound, each next one the sound pooled once more).</summary>
    public int Scales { get; init; } = 1;

    /// <summary>The scale discriminators' channels per convolution.</summary>
    public int[] ScaleChannels { get; init; } = [16, 64, 256, 1024, 1024, 1024];

    /// <summary>Their kernels.</summary>
    public int[] ScaleKernels { get; init; } = [15, 41, 41, 41, 41, 5];

    /// <summary>Their strides.</summary>
    public int[] ScaleStrides { get; init; } = [1, 4, 4, 4, 4, 1];

    /// <summary>Their channel groups.</summary>
    public int[] ScaleGroups { get; init; } = [1, 4, 16, 64, 256, 1];

    /// <summary>The average pooling between scales: kernel, stride and padding (counted in the average, as PyTorch's default).</summary>
    public int PoolKernel { get; init; } = 4;

    /// <inheritdoc cref="PoolKernel"/>
    public int PoolStride { get; init; } = 2;

    /// <inheritdoc cref="PoolKernel"/>
    public int PoolPadding { get; init; } = 2;

    /// <summary>The leaky ReLU's slope after each convolution.</summary>
    public float LeakyReluSlope { get; init; } = 0.1f;

    /// <summary>The description as discriminator.json writes it.</summary>
    public JsonObject ToJson() => new()
    {
        ["periods"] = Ints(Periods), ["period_channels"] = Ints(PeriodChannels), ["period_kernel"] = PeriodKernel, ["period_stride"] = PeriodStride,
        ["post_kernel"] = PostKernel, ["scales"] = Scales, ["scale_channels"] = Ints(ScaleChannels), ["scale_kernels"] = Ints(ScaleKernels),
        ["scale_strides"] = Ints(ScaleStrides), ["scale_groups"] = Ints(ScaleGroups), ["pool_kernel"] = PoolKernel, ["pool_stride"] = PoolStride,
        ["pool_padding"] = PoolPadding, ["leaky_relu_slope"] = LeakyReluSlope,
    };

    /// <summary>A description from its JSON; a key missing keeps the default.</summary>
    /// <exception cref="InvalidDataException">A list is not one, the scale lists differ in length, or a size is not positive.</exception>
    public static VitsDiscriminatorConfig FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var d = new VitsDiscriminatorConfig();
        var c = new VitsDiscriminatorConfig
        {
            Periods = List(json, "periods", d.Periods),
            PeriodChannels = List(json, "period_channels", d.PeriodChannels),
            PeriodKernel = (int?)json["period_kernel"] ?? d.PeriodKernel,
            PeriodStride = (int?)json["period_stride"] ?? d.PeriodStride,
            PostKernel = (int?)json["post_kernel"] ?? d.PostKernel,
            Scales = (int?)json["scales"] ?? d.Scales,
            ScaleChannels = List(json, "scale_channels", d.ScaleChannels),
            ScaleKernels = List(json, "scale_kernels", d.ScaleKernels),
            ScaleStrides = List(json, "scale_strides", d.ScaleStrides),
            ScaleGroups = List(json, "scale_groups", d.ScaleGroups),
            PoolKernel = (int?)json["pool_kernel"] ?? d.PoolKernel,
            PoolStride = (int?)json["pool_stride"] ?? d.PoolStride,
            PoolPadding = (int?)json["pool_padding"] ?? d.PoolPadding,
            LeakyReluSlope = (float?)json["leaky_relu_slope"] ?? d.LeakyReluSlope,
        };
        int n = c.ScaleChannels.Length;
        if (c.ScaleKernels.Length != n || c.ScaleStrides.Length != n || c.ScaleGroups.Length != n)
        {
            throw new InvalidDataException($"The scale discriminators' channels, kernels, strides and groups pair up: {n}, {c.ScaleKernels.Length}, {c.ScaleStrides.Length}, {c.ScaleGroups.Length}.");
        }

        if (c.Periods.Any(p => p < 1) || c.PeriodChannels.Length == 0 || c.PeriodKernel < 1 || c.PeriodStride < 1 || c.PostKernel < 1 || c.Scales < 0 || (c.Scales > 0 && n == 0)
            || c.PoolKernel < 1 || c.PoolStride < 1 || c.PoolPadding < 0 || c.Periods.Length + c.Scales == 0)
        {
            throw new InvalidDataException($"A discriminator description has a size that is not positive (or no discriminator): {c.ToJson().ToJsonString()}.");
        }

        return c;
    }

    private static JsonArray Ints(int[] values) => new([.. values.Select(v => (JsonNode)v)]);

    private static int[] List(JsonObject json, string key, int[] fallback) => json[key] switch
    {
        null => fallback,
        JsonArray a => [.. a.Select(v => (int)v!)],
        _ => throw new InvalidDataException($"discriminator.json's \"{key}\" is not a list."),
    };
}

/// <summary>
/// The discriminators VITS trains its generator against (its multi-period discriminator: a scale discriminator of
/// grouped convolutions over the sound and period discriminators of (k, 1) convolutions over the sound folded into rows
/// of a period; more scales read it average-pooled, as HiFi-GAN's multi-scale discriminator), every convolution
/// weight-normed and followed by a leaky ReLU. Each gives its scores and the feature maps of each layer (the
/// feature-matching loss compares them). Stored as discriminator.safetensors under VITS's names
/// (<c>discriminators.{i}.convs.{j}</c>, <c>discriminators.{i}.conv_post</c>; the scales first) with discriminator.json.
/// </summary>
internal sealed class VitsDiscriminators : IDisposable
{
    /// <summary>The weights' file in a model folder.</summary>
    public const string WeightsFile = "discriminator.safetensors";

    /// <summary>The description's file in a model folder.</summary>
    public const string ConfigFile = "discriminator.json";

    private readonly Stack[] _scales, _periods;
    private readonly VitsWeights _weights;

    public VitsDiscriminators(VitsDiscriminatorConfig config, ITensorStore store, Device device)
    {
        Config = config;
        _weights = new VitsWeights(store, device, trainable: true);
        _scales = new Stack[config.Scales];
        for (int s = 0; s < config.Scales; s++)
        {
            _scales[s] = Stack.Scale(_weights, $"discriminators.{s}", config);
        }

        _periods = new Stack[config.Periods.Length];
        for (int p = 0; p < _periods.Length; p++)
        {
            _periods[p] = Stack.Period(_weights, $"discriminators.{config.Scales + p}", config, config.Periods[p]);
        }

        Module = new VitsParameterSet("VITS discriminators", _weights.Trained);
    }

    public VitsDiscriminatorConfig Config { get; }

    /// <summary>The trained tensors (what the discriminators' optimizer steps).</summary>
    public VitsParameterSet Module { get; }

    /// <summary>
    /// New discriminators of <paramref name="config"/>: each convolution's directions and bias drawn as PyTorch's
    /// default does (uniform within 1 / √fan-in), its gains the directions' norms (weight_norm's start), from <paramref name="seed"/>.
    /// </summary>
    public static VitsDiscriminators Create(VitsDiscriminatorConfig config, Device device, int seed)
    {
        var random = new Random(seed);
        var tensors = new VitsTensorDictionary();
        void Draw(string prefix, int[] shape)
        {
            int rows = shape[0], columns = shape.Skip(1).Aggregate(1, (a, b) => a * b);
            double bound = 1 / Math.Sqrt(columns);
            var v = new float[rows * columns];
            var g = new float[rows];
            var b = new float[rows];
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = (float)((random.NextDouble() * 2 - 1) * bound);
            }

            for (int r = 0; r < rows; r++)
            {
                double norm = 0;
                for (int i = 0; i < columns; i++)
                {
                    norm += (double)v[r * columns + i] * v[r * columns + i];
                }

                g[r] = (float)Math.Sqrt(norm);
                b[r] = (float)((random.NextDouble() * 2 - 1) * bound);
            }

            tensors.Add(prefix + ".parametrizations.weight.original0", [rows, .. Enumerable.Repeat(1, shape.Length - 1)], g);
            tensors.Add(prefix + ".parametrizations.weight.original1", shape, v);
            tensors.Add(prefix + ".bias", [rows], b);
        }

        for (int s = 0; s < config.Scales; s++)
        {
            int c = 1;
            for (int j = 0; j < config.ScaleChannels.Length; j++)
            {
                Draw($"discriminators.{s}.convs.{j}", [config.ScaleChannels[j], c / config.ScaleGroups[j], config.ScaleKernels[j]]);
                c = config.ScaleChannels[j];
            }

            Draw($"discriminators.{s}.conv_post", [1, c, config.PostKernel]);
        }

        for (int p = 0; p < config.Periods.Length; p++)
        {
            int c = 1, i = config.Scales + p;
            for (int j = 0; j < config.PeriodChannels.Length; j++)
            {
                Draw($"discriminators.{i}.convs.{j}", [config.PeriodChannels[j], c, config.PeriodKernel, 1]);
                c = config.PeriodChannels[j];
            }

            Draw($"discriminators.{i}.conv_post", [1, c, config.PostKernel, 1]);
        }

        return new VitsDiscriminators(config, tensors, device);
    }

    /// <summary>The scores of each discriminator for the sound [1, L] and each one's feature maps (every layer's output, the scores' layer last).</summary>
    public (List<Tensor> Scores, List<List<Tensor>> Maps) Forward(Tensor sound)
    {
        var scores = new List<Tensor>(_scales.Length + _periods.Length);
        var maps = new List<List<Tensor>>(scores.Capacity);
        var x = sound;
        for (int s = 0; s < _scales.Length; s++)
        {
            if (s > 0)
            {
                int length = x.Shape[1];
                x = x.Reshape(1, 1, 1, length).AvgPool(new ConvGeometry(1, 1, 1, length, 1, Config.PoolKernel, 1, Config.PoolStride, 0, Config.PoolPadding), countIncludePad: true);
                x = x.Reshape(1, x.Shape[3]);
            }

            var (score, map) = _scales[s].Forward(x.Reshape(1, 1, 1, x.Shape[1]), Config.LeakyReluSlope);
            scores.Add(score);
            maps.Add(map);
        }

        foreach (var (stack, period) in _periods.Zip(Config.Periods))
        {
            int length = sound.Shape[1], padded = (length + period - 1) / period * period;
            var folded = VitsOps.ReflectEnd(sound, padded).Reshape(1, 1, padded / period, period);
            var (score, map) = stack.Forward(folded, Config.LeakyReluSlope);
            scores.Add(score);
            maps.Add(map);
        }

        return (scores, maps);
    }

    /// <summary>Writes discriminator.safetensors and discriminator.json into <paramref name="folder"/>.</summary>
    public void Save(string folder)
    {
        SafeTensorsWriter.Write(Path.Combine(folder, WeightsFile), _weights.Trained.Select(p => (p.Name, p.Stored, p.Values())), SafeTensorType.F32);
        File.WriteAllText(Path.Combine(folder, ConfigFile), Config.ToJson().ToJsonString(new System.Text.Json.JsonSerializerOptions { WriteIndented = true }) + "\n");
    }

    public void Dispose()
    {
        foreach (var s in _scales) s.Dispose();
        foreach (var p in _periods) p.Dispose();
    }

    // One discriminator: its convolutions, each followed by the leaky ReLU, and the output convolution; [1, C, H, W] in.
    private sealed class Stack(Conv1d[] convs, Conv1d post) : IDisposable
    {
        public static Stack Scale(VitsWeights weights, string prefix, VitsDiscriminatorConfig c)
        {
            var convs = new Conv1d[c.ScaleChannels.Length];
            int channels = 1;
            for (int j = 0; j < convs.Length; j++)
            {
                int k = c.ScaleKernels[j];
                convs[j] = weights.Conv2($"{prefix}.convs.{j}", channels, c.ScaleChannels[j], (1, k), (1, c.ScaleStrides[j]), (0, (k - 1) / 2), (1, 1), c.ScaleGroups[j]);
                channels = c.ScaleChannels[j];
            }

            return new Stack(convs, weights.Conv2($"{prefix}.conv_post", channels, 1, (1, c.PostKernel), (1, 1), (0, (c.PostKernel - 1) / 2), (1, 1)));
        }

        public static Stack Period(VitsWeights weights, string prefix, VitsDiscriminatorConfig c, int period)
        {
            _ = period;
            var convs = new Conv1d[c.PeriodChannels.Length];
            int channels = 1, k = c.PeriodKernel;
            for (int j = 0; j < convs.Length; j++)
            {
                int stride = j < convs.Length - 1 ? c.PeriodStride : 1;
                convs[j] = weights.Conv2($"{prefix}.convs.{j}", channels, c.PeriodChannels[j], (k, 1), (stride, 1), ((k - 1) / 2, 0), (1, 1));
                channels = c.PeriodChannels[j];
            }

            return new Stack(convs, weights.Conv2($"{prefix}.conv_post", channels, 1, (c.PostKernel, 1), (1, 1), ((c.PostKernel - 1) / 2, 0), (1, 1)));
        }

        public (Tensor Score, List<Tensor> Maps) Forward(Tensor x, float slope)
        {
            var maps = new List<Tensor>(convs.Length + 1);
            foreach (var conv in convs)
            {
                x = VitsOps.LeakyRelu(conv.Forward4(x), slope);
                maps.Add(x);
            }

            x = post.Forward4(x);
            maps.Add(x);
            return (x.Reshape(1, x.Size), maps);
        }

        public void Dispose()
        {
            foreach (var conv in convs) conv.Dispose();
            post.Dispose();
        }
    }
}

/// <summary>Tensors held in memory by name, as a <see cref="ITensorStore"/> (new discriminators before their first save).</summary>
internal sealed class VitsTensorDictionary : ITensorStore
{
    private readonly Dictionary<string, (int[] Shape, float[] Values)> _tensors = new(StringComparer.Ordinal);

    public IEnumerable<string> Names => _tensors.Keys;

    public void Add(string name, int[] shape, float[] values)
    {
        if (shape.Aggregate(1, (a, b) => a * b) != values.Length)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture, $"{name}: {values.Length} values for {Tensor.FormatShape(shape)}."));
        }

        _tensors[name] = (shape, values);
    }

    public bool Contains(string name) => _tensors.ContainsKey(name);

    public int[] ShapeOf(string name) => _tensors[name].Shape;

    public float[] Read(string name) => _tensors[name].Values;

    public void Dispose()
    {
    }
}
