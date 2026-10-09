// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

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
/// (<see cref="ToJson"/>, <see cref="Network.FromJson"/>); model packages use this to store the architecture. Steps of
/// your own are registered in <see cref="NetworkOps"/> and added with <see cref="Op"/>; they see the builder through
/// <see cref="INetworkBuilder"/>.
/// </summary>
public sealed class NetworkBuilder : INetworkBuilder
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

    /// <summary>Whether every step can be written to JSON (false after <see cref="Lambda"/> or <c>Add</c> outside a registered step, <see cref="Op"/>).</summary>
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

    /// <summary>
    /// <c>new Layers.ChannelNormalize(mean, std)</c>: (x - mean[c]) / std[c] per channel of [C, H, W] (or feature of
    /// [F]). Put it first so the network takes plain [0, 1] images; <c>ChannelStatistics</c> (Idrak.Vision) has ImageNet's
    /// values and computes a data set's.
    /// </summary>
    public NetworkBuilder Normalize(IReadOnlyList<float> mean, IReadOnlyList<float> std)
    {
        if (_shape.Length is not (1 or 3) || _shape[0] != mean.Count)
        {
            throw new InvalidOperationException($"Normalize needs one mean and std per channel of [channels, height, width] (or feature of [features]); the shape is {Tensor.FormatShape(_shape)}, with {mean.Count} means.");
        }

        float[] m = [.. mean], s = [.. std];
        return Push(_ => new Layers.ChannelNormalize(m, s, _device), _shape,
            Step("normalize", ("mean", new JsonArray([.. m.Select(v => JsonValue.Create(v))])), ("std", new JsonArray([.. s.Select(v => JsonValue.Create(v))]))));
    }

    /// <summary>
    /// <c>new Layers.GroupNorm(groups, channels, epsilon, affine)</c>; channels are the features of [F] or the channels of
    /// [C, H, W] (normalized per sample and group, whatever the batch size).
    /// </summary>
    public NetworkBuilder GroupNorm(int groups, float epsilon = 1e-5f, bool affine = true)
    {
        if (_shape.Length is not (1 or 3))
        {
            throw new InvalidOperationException($"GroupNorm needs [features] or [channels, height, width], the current shape is {Tensor.FormatShape(_shape)}.");
        }

        int channels = _shape[0];
        if (groups <= 0 || channels % groups != 0)
        {
            throw new ArgumentException($"GroupNorm's {groups} groups divide the {channels} channels.", nameof(groups));
        }

        var step = Step("groupnorm", ("groups", groups), ("epsilon", epsilon));
        if (!affine)
        {
            step["affine"] = false;
        }

        return Push(_ => new Layers.GroupNorm(groups, channels, epsilon, affine, _device), _shape, step);
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

    /// <summary>
    /// <c>new Layers.Conv2d(channels, outChannels, kernelSize, stride, padding, dilation, groups, bias)</c> with (height, width)
    /// pairs (a text line's tall or wide filters, dilated or grouped convolutions); channels from the current [C, H, W].
    /// </summary>
    public NetworkBuilder Conv2d(int outChannels, (int Height, int Width) kernelSize, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null,
        (int Height, int Width)? dilation = null, int groups = 1, bool bias = true)
    {
        var (c, h, w) = Image("Conv2d");
        var (sh, sw) = stride ?? (1, 1);
        var (ph, pw) = padding ?? (0, 0);
        var (dh, dw) = dilation ?? (1, 1);
        if (sh <= 0 || sw <= 0 || dh <= 0 || dw <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(stride), "Conv2d strides and dilations are positive.");
        }

        int oh = (h + 2 * ph - dh * (kernelSize.Height - 1) - 1) / sh + 1, ow = (w + 2 * pw - dw * (kernelSize.Width - 1) - 1) / sw + 1;
        CheckSpatial("Conv2d", oh, ow);
        var step = Step("conv2d", ("out", outChannels), ("kernel", Pair(kernelSize)), ("stride", Pair((sh, sw))), ("padding", Pair((ph, pw))), ("bias", bias));
        if (dh != 1 || dw != 1)
        {
            step["dilation"] = Pair((dh, dw));
        }

        if (groups != 1)
        {
            step["groups"] = groups;
        }

        return Push(r => new Layers.Conv2d(c, outChannels, kernelSize, (sh, sw), (ph, pw), (dh, dw), groups, bias, _device, r), [outChannels, oh, ow], step);
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

    /// <summary>
    /// <c>new Layers.MaxPool2d(kernelSize, stride, padding, ceilMode, paddingEnd)</c> with (height, width) pairs (a text
    /// line's tall or wide windows; ONNX's padding below and right; PyTorch's ceil mode).
    /// </summary>
    public NetworkBuilder MaxPool2d((int Height, int Width) kernelSize, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null, bool ceilMode = false,
        (int Height, int Width)? paddingEnd = null)
    {
        var (c, oh, ow) = Pooled("MaxPool2d", kernelSize, stride, padding, paddingEnd, ceilMode);
        var step = Step("maxpool2d", ("kernel", Pair(kernelSize)), ("padding", Pair(padding ?? (0, 0))));
        if (stride is { } explicitStride)
        {
            step["stride"] = Pair(explicitStride);
        }

        PoolingExtras(step, padding, paddingEnd, ceilMode);
        return Push(_ => new Layers.MaxPool2d(kernelSize, stride, padding, ceilMode, paddingEnd), [c, oh, ow], step);
    }

    /// <summary><c>new Layers.AvgPool2d(kernelSize, stride, padding, countIncludePad)</c>.</summary>
    public NetworkBuilder AvgPool2d(int kernelSize, int? stride = null, int padding = 0, bool countIncludePad = true) =>
        AvgPool2d((kernelSize, kernelSize), stride is { } s ? (s, s) : null, (padding, padding), countIncludePad);

    /// <summary><c>new Layers.AvgPool2d(kernelSize, stride, padding, countIncludePad, ceilMode, paddingEnd)</c> with (height, width) pairs.</summary>
    public NetworkBuilder AvgPool2d((int Height, int Width) kernelSize, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null, bool countIncludePad = true,
        bool ceilMode = false, (int Height, int Width)? paddingEnd = null)
    {
        var (c, oh, ow) = Pooled("AvgPool2d", kernelSize, stride, padding, paddingEnd, ceilMode);
        var step = Step("avgpool2d", ("kernel", Pair(kernelSize)), ("padding", Pair(padding ?? (0, 0))), ("countIncludePad", countIncludePad));
        if (stride is { } explicitStride)
        {
            step["stride"] = Pair(explicitStride);
        }

        PoolingExtras(step, padding, paddingEnd, ceilMode);
        return Push(_ => new Layers.AvgPool2d(kernelSize, stride, padding, countIncludePad, ceilMode, paddingEnd), [c, oh, ow], step);
    }

    // A pooling window's output size over the current [C, H, W] (PyTorch's and ONNX's count, ceil mode included).
    private (int C, int OH, int OW) Pooled(string layer, (int Height, int Width) kernel, (int Height, int Width)? stride, (int Height, int Width)? padding,
        (int Height, int Width)? paddingEnd = null, bool ceilMode = false)
    {
        var (c, h, w) = Image(layer);
        var (_, _, sh, sw, ph, pw, pb, pr) = Layers.Pool.Check(layer, kernel, stride, padding, paddingEnd);
        int oh = Layers.Pool.Windows(h, kernel.Height, sh, ph, pb, ceilMode), ow = Layers.Pool.Windows(w, kernel.Width, sw, pw, pr, ceilMode);
        CheckSpatial(layer, oh, ow);
        return (c, oh, ow);
    }

    // The ceil mode and padding below and right, written only when they are not the defaults.
    private static void PoolingExtras(JsonObject step, (int Height, int Width)? padding, (int Height, int Width)? paddingEnd, bool ceilMode)
    {
        if (paddingEnd is { } end && end != (padding ?? (0, 0)))
        {
            step["paddingEnd"] = Pair(end);
        }

        if (ceilMode)
        {
            step["ceilMode"] = true;
        }
    }

    /// <summary>
    /// <c>new Layers.ConvTranspose2d(channels, outChannels, kernelSize, stride, padding, outputPadding, bias)</c>; channels from the
    /// current [C, H, W]: [C, H, W] → [out, (H - 1)·stride - 2·padding + kernel + outputPadding, ...] (a decoder's upsampling step).
    /// </summary>
    public NetworkBuilder ConvTranspose2d(int outChannels, int kernelSize, int stride = 1, int padding = 0, int outputPadding = 0, bool bias = true) =>
        ConvTranspose2d(outChannels, (kernelSize, kernelSize), (stride, stride), (padding, padding), (outputPadding, outputPadding), null, 1, bias);

    /// <summary>
    /// <c>new Layers.ConvTranspose2d(channels, outChannels, kernelSize, stride, padding, outputPadding, dilation, groups, bias)</c>
    /// with (height, width) pairs; channels from the current [C, H, W].
    /// </summary>
    public NetworkBuilder ConvTranspose2d(int outChannels, (int Height, int Width) kernelSize, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null,
        (int Height, int Width)? outputPadding = null, (int Height, int Width)? dilation = null, int groups = 1, bool bias = true)
    {
        var (c, h, w) = Image("ConvTranspose2d");
        var (sh, sw) = stride ?? (1, 1);
        var (ph, pw) = padding ?? (0, 0);
        var (oph, opw) = outputPadding ?? (0, 0);
        var (dh, dw) = dilation ?? (1, 1);
        Layers.ConvTranspose2d.Validate(c, outChannels, kernelSize, (sh, sw), (ph, pw), (oph, opw), (dh, dw), groups);
        int oh = (h - 1) * sh - 2 * ph + dh * (kernelSize.Height - 1) + oph + 1, ow = (w - 1) * sw - 2 * pw + dw * (kernelSize.Width - 1) + opw + 1;
        CheckSpatial("ConvTranspose2d", oh, ow);
        var step = Step("convtranspose2d", ("out", outChannels), ("kernel", Pair(kernelSize)), ("stride", Pair((sh, sw))), ("padding", Pair((ph, pw))), ("bias", bias));
        if (oph != 0 || opw != 0)
        {
            step["outputPadding"] = Pair((oph, opw));
        }

        if (dh != 1 || dw != 1)
        {
            step["dilation"] = Pair((dh, dw));
        }

        if (groups != 1)
        {
            step["groups"] = groups;
        }

        return Push(r => new Layers.ConvTranspose2d(c, outChannels, kernelSize, (sh, sw), (ph, pw), (oph, opw), (dh, dw), groups, bias, _device, r), [outChannels, oh, ow], step);
    }

    /// <summary><c>new Layers.Upsample(scaleFactor, mode, alignCorners)</c>: [C, H, W] → [C, floor(H · factor), floor(W · factor)].</summary>
    public NetworkBuilder Upsample(float scaleFactor, InterpolationMode mode = InterpolationMode.Nearest, bool alignCorners = false) =>
        Upsample((scaleFactor, scaleFactor), mode, alignCorners);

    /// <summary><c>new Layers.Upsample(scaleFactor, mode, alignCorners)</c> with a (height, width) factor.</summary>
    public NetworkBuilder Upsample((float Height, float Width) scaleFactor, InterpolationMode mode = InterpolationMode.Nearest, bool alignCorners = false)
    {
        var layer = new Layers.Upsample(scaleFactor, mode, alignCorners);
        var step = Step("upsample", ("scale", scaleFactor.Height == scaleFactor.Width ? JsonValue.Create(scaleFactor.Height) : new JsonArray(scaleFactor.Height, scaleFactor.Width)));
        return Upsampled(layer, step, () => new Layers.Upsample(scaleFactor, mode, alignCorners));
    }

    /// <summary><c>Layers.Upsample.ToSize(size, mode, alignCorners)</c>: [C, H, W] → [C, size.Height, size.Width].</summary>
    public NetworkBuilder UpsampleToSize((int Height, int Width) size, InterpolationMode mode = InterpolationMode.Nearest, bool alignCorners = false)
    {
        var layer = Layers.Upsample.ToSize(size, mode, alignCorners);
        return Upsampled(layer, Step("upsample", ("size", Pair(size))), () => Layers.Upsample.ToSize(size, mode, alignCorners));
    }

    private NetworkBuilder Upsampled(Layers.Upsample layer, JsonObject step, Func<Module> create)
    {
        var (c, h, w) = Image("Upsample");
        var (oh, ow) = layer.OutputSize(h, w);
        CheckSpatial("Upsample", oh, ow);
        step["mode"] = layer.Mode == InterpolationMode.Nearest ? "nearest" : "bilinear";
        if (layer.AlignCorners)
        {
            step["alignCorners"] = true;
        }

        return Push(_ => create(), [c, oh, ow], step);
    }

    /// <summary><c>new Layers.AdaptiveAvgPool2d(outputSize)</c>: [C, H, W] → [C, outputSize, outputSize].</summary>
    public NetworkBuilder AdaptiveAvgPool2d(int outputSize) => AdaptiveAvgPool2d((outputSize, outputSize));

    /// <summary><c>new Layers.AdaptiveAvgPool2d(outputSize)</c>: [C, H, W] → [C, outputSize.Height, outputSize.Width].</summary>
    public NetworkBuilder AdaptiveAvgPool2d((int Height, int Width) outputSize) =>
        Adaptive("adaptiveavgpool2d", "AdaptiveAvgPool2d", outputSize, () => new Layers.AdaptiveAvgPool2d(outputSize));

    /// <summary><c>new Layers.AdaptiveMaxPool2d(outputSize)</c>: [C, H, W] → [C, outputSize, outputSize].</summary>
    public NetworkBuilder AdaptiveMaxPool2d(int outputSize) => AdaptiveMaxPool2d((outputSize, outputSize));

    /// <summary><c>new Layers.AdaptiveMaxPool2d(outputSize)</c>: [C, H, W] → [C, outputSize.Height, outputSize.Width].</summary>
    public NetworkBuilder AdaptiveMaxPool2d((int Height, int Width) outputSize) =>
        Adaptive("adaptivemaxpool2d", "AdaptiveMaxPool2d", outputSize, () => new Layers.AdaptiveMaxPool2d(outputSize));

    private NetworkBuilder Adaptive(string op, string layer, (int Height, int Width) size, Func<Module> create)
    {
        var (c, _, _) = Image(layer);
        Layers.AdaptivePooling.Check(layer, size);
        return Push(_ => create(), [c, size.Height, size.Width], Step(op, ("size", Pair(size))));
    }

    // A (height, width) pair as the builder's JSON writes it: one number when both are equal, else [height, width].
    private static JsonNode Pair((int Height, int Width) pair) =>
        pair.Height == pair.Width ? JsonValue.Create(pair.Height) : new JsonArray(pair.Height, pair.Width);

    /// <summary>
    /// A <see cref="Layers.Lambda"/> reading an image's columns as a sequence: [C, H, W] → [W, C · H], step t holding column t
    /// of every channel (channel-major), as a convolutional feature map enters a recurrent layer (a text line read left to
    /// right).
    /// </summary>
    public NetworkBuilder ColumnsToSequence()
    {
        var (c, h, w) = Image("ColumnsToSequence");
        return Push(_ => new Layers.Lambda(ColumnsAsSequence, "ColumnsToSequence"), [w, c * h], Step("columnsToSequence"));
    }

    // [N, C, H, W] → [N, W, C·H].
    internal static Tensor ColumnsAsSequence(Tensor x) => x.Permute(0, 3, 1, 2).Reshape(x.Shape[0], x.Shape[3], x.Shape[1] * x.Shape[2]);

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

    /// <summary>
    /// <c>new Layers.LSTM(features, hiddenSize, returnSequences, bidirectional, layers)</c>: [T, F] → [hidden · directions], or
    /// [T, hidden · directions] with <paramref name="returnSequences"/>.
    /// </summary>
    public NetworkBuilder LSTM(int hiddenSize, bool returnSequences, bool bidirectional, int layers = 1)
    {
        var (t, f) = Sequence("LSTM");
        int outputs = hiddenSize * (bidirectional ? 2 : 1);
        return Push(r => new Layers.LSTM(f, hiddenSize, returnSequences, bidirectional, layers, _device, r), returnSequences ? [t, outputs] : [outputs],
            Recurrent("lstm", hiddenSize, returnSequences, bidirectional, layers, false));
    }

    /// <summary><c>new Layers.GRU(features, hiddenSize, returnSequences)</c>: [T, F] → [hidden], or [T, hidden] with <paramref name="returnSequences"/>.</summary>
    public NetworkBuilder GRU(int hiddenSize, bool returnSequences = false)
    {
        var (t, f) = Sequence("GRU");
        return Push(r => new Layers.GRU(f, hiddenSize, returnSequences, _device, r), returnSequences ? [t, hiddenSize] : [hiddenSize],
            Step("gru", ("hidden", hiddenSize), ("returnSequences", returnSequences)));
    }

    /// <summary>
    /// <c>new Layers.GRU(features, hiddenSize, returnSequences, bidirectional, layers, candidateBias)</c>: [T, F] → [hidden ·
    /// directions], or [T, hidden · directions] with <paramref name="returnSequences"/>.
    /// </summary>
    public NetworkBuilder GRU(int hiddenSize, bool returnSequences, bool bidirectional, int layers = 1, bool candidateBias = false)
    {
        var (t, f) = Sequence("GRU");
        int outputs = hiddenSize * (bidirectional ? 2 : 1);
        return Push(r => new Layers.GRU(f, hiddenSize, returnSequences, bidirectional, layers, candidateBias, _device, r), returnSequences ? [t, outputs] : [outputs],
            Recurrent("gru", hiddenSize, returnSequences, bidirectional, layers, candidateBias));
    }

    private static JsonObject Recurrent(string op, int hidden, bool returnSequences, bool bidirectional, int layers, bool candidateBias)
    {
        var step = Step(op, ("hidden", hidden), ("returnSequences", returnSequences), ("bidirectional", bidirectional), ("layers", layers));
        if (candidateBias)
        {
            step["candidateBias"] = true;
        }

        return step;
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

    /// <summary>
    /// Appends a layer that <paramref name="create"/> makes anew on every <see cref="Build"/>, from the builder's device
    /// and random (as the built-in steps make theirs). Its output shape is required. A builder with such a layer cannot
    /// be written to JSON, unless a registered step added it (<see cref="Op"/>).
    /// </summary>
    public NetworkBuilder Add(Func<Device?, Random?, Module> create, int[] outputShape)
    {
        ArgumentNullException.ThrowIfNull(create);
        _describable = false;
        return Push(r => create(_device, r), [.. outputShape], null);
    }

    /// <summary>
    /// Adds the step registered as <paramref name="name"/> in <see cref="NetworkOps"/>, with <paramref name="arguments"/>
    /// (the keys of its JSON besides "op"). A step that adds its layers with <see cref="Lambda"/> or <see cref="Add(Func{Device?, Random?, Module}, int[])"/>
    /// is still written to JSON, as <c>{"op": name, ...arguments}</c>, and <see cref="Network.FromJson"/> replays it through
    /// the same registration.
    /// </summary>
    public NetworkBuilder Op(string name, JsonObject? arguments = null)
    {
        var op = NetworkOps.Get(name);
        var step = new JsonObject { ["op"] = name };
        foreach (var (key, value) in arguments ?? [])
        {
            if (key != "op")
            {
                step[key] = value?.DeepClone();
            }
        }

        bool describable = _describable;
        int count = _steps.Count;
        op(this, new NetworkOpArguments(step));
        if (_describable == describable && _steps.Skip(count).All(s => s.Step is not null))
        {
            return this;                                                    // made of described steps (the built-ins): they describe it
        }

        // Layers JSON cannot describe: the step's own JSON stands for all of them (written once, with the first).
        for (int i = count; i < _steps.Count; i++)
        {
            _steps[i] = (_steps[i].Create, i == count ? step : null);
        }

        _describable = describable;
        return this;
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

    // ------------------------------------------------------------------ what registered steps see (INetworkBuilder)

    /// <inheritdoc />
    INetworkBuilder INetworkBuilder.Lambda(Func<Tensor, Tensor> function, string name, int[] outputShape) => Lambda(function, name, outputShape);

    /// <inheritdoc />
    INetworkBuilder INetworkBuilder.Add(Func<Device?, Random?, Module> create, int[] outputShape) => Add(create, outputShape);

    /// <inheritdoc />
    INetworkBuilder INetworkBuilder.Op(string name, JsonObject? arguments) => Op(name, arguments);

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
            ["steps"] = new JsonArray([.. _steps.Where(s => s.Step is not null).Select(s => (JsonNode)s.Step!.DeepClone())]),
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
            var step = node!.AsObject();
            string op = (string)step["op"]!;
            if (!NetworkOps.Contains(op))
            {
                throw new InvalidDataException($"Unknown network step '{op}' (registered: {string.Join(", ", NetworkOps.Names)}); add it with NetworkOps.Register.");
            }

            b.Op(op, step);
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
