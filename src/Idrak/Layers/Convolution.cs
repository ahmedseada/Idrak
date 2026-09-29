using Idrak.Backends;

namespace Idrak.Layers;

/// <summary>
/// 2-D convolution over [N, C, H, W] images, producing [N, outChannels, OH, OW].
/// Implemented as patch unfolding (im2col) followed by one large matrix product, so it runs on the same
/// optimized GEMM as <see cref="Linear"/> on both CPU and GPU. Weights start He-uniform (suited to ReLU).
/// </summary>
public sealed class Conv2d : Module
{
    /// <summary>Creates the layer.</summary>
    /// <param name="inChannels">Input channels (1 for grayscale, 3 for RGB).</param>
    /// <param name="outChannels">Number of filters.</param>
    /// <param name="kernelSize">Filter height and width.</param>
    /// <param name="stride">Step between filter positions.</param>
    /// <param name="padding">Zero padding on each border; kernelSize / 2 keeps the size for odd kernels and stride 1.</param>
    /// <param name="bias">Whether to learn a per-filter bias.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    public Conv2d(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, bool bias = true, Device? device = null, Random? random = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inChannels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outChannels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kernelSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(stride);
        ArgumentOutOfRangeException.ThrowIfNegative(padding);
        InChannels = inChannels;
        OutChannels = outChannels;
        KernelSize = kernelSize;
        Stride = stride;
        Padding = padding;
        device ??= Device.Default;
        int fanIn = inChannels * kernelSize * kernelSize;
        Weight = CreateParameter(UniformValues(outChannels * fanIn, MathF.Sqrt(6f / fanIn), random ?? Random.Shared), [outChannels, fanIn], device);
        Bias = bias ? CreateParameter(new float[outChannels], [outChannels], device) : null;
    }

    /// <summary>Input channels.</summary>
    public int InChannels { get; }

    /// <summary>Output channels (filters).</summary>
    public int OutChannels { get; }

    /// <summary>Filter size.</summary>
    public int KernelSize { get; }

    /// <summary>Filter step.</summary>
    public int Stride { get; }

    /// <summary>Zero padding per border.</summary>
    public int Padding { get; }

    /// <summary>Filters as [outChannels, inChannels · k · k].</summary>
    public Tensor Weight { get; private set; }

    /// <summary>Per-filter bias, or null.</summary>
    public Tensor? Bias { get; private set; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 4 || input.Shape[1] != InChannels)
        {
            throw new ArgumentException($"Conv2d expects [N, {InChannels}, H, W], got {Tensor.FormatShape(input.Shape)}.");
        }

        var g = new ConvGeometry(input.Shape[0], InChannels, input.Shape[2], input.Shape[3], KernelSize, KernelSize, Stride, Stride, Padding, Padding);
        if (g.OH <= 0 || g.OW <= 0)
        {
            throw new ArgumentException($"A {KernelSize}x{KernelSize} kernel does not fit a {g.H}x{g.W} input with padding {Padding}.");
        }

        var columns = input.Im2Col(g);                                   // [N·OH·OW, C·k·k]
        var rows = columns.MatMul(Weight, transposeB: true);             // [N·OH·OW, OC]
        var output = rows.Reshape(g.N, g.OH * g.OW, OutChannels)
            .Permute(0, 2, 1)
            .Reshape(g.N, OutChannels, g.OH, g.OW);
        return Bias is null ? output : output.GroupAffine(null, Bias, OutChannels, g.OH * g.OW);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => Bias is null ? [Weight] : [Weight, Bias];

    /// <inheritdoc />
    protected internal override void MoveTo(Device device)
    {
        Weight = MoveTensor(Weight, device);
        Bias = Bias is null ? null : MoveTensor(Bias, device);
    }

    /// <inheritdoc />
    public override string ToString() => $"Conv2d({InChannels} -> {OutChannels}, {KernelSize}x{KernelSize}, stride {Stride}, padding {Padding})";
}

/// <summary>2-D max pooling over [N, C, H, W]: keeps the largest value of each window.</summary>
/// <param name="kernelSize">Window height and width.</param>
/// <param name="stride">Step between windows; defaults to the window size (non-overlapping).</param>
/// <param name="padding">Border padding (padded positions never win).</param>
public sealed class MaxPool2d(int kernelSize, int? stride = null, int padding = 0) : Module
{
    /// <summary>Window size.</summary>
    public int KernelSize { get; } = kernelSize;

    /// <summary>Window step.</summary>
    public int Stride { get; } = stride ?? kernelSize;

    /// <summary>Border padding.</summary>
    public int Padding { get; } = padding;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException($"MaxPool2d expects [N, C, H, W], got {Tensor.FormatShape(input.Shape)}.");
        }

        var s = input.Shape;
        return input.MaxPool(new ConvGeometry(s[0], s[1], s[2], s[3], KernelSize, KernelSize, Stride, Stride, Padding, Padding));
    }

    /// <inheritdoc />
    public override string ToString() => $"MaxPool2d({KernelSize}x{KernelSize}, stride {Stride})";
}

/// <summary>Averages each channel over all positions: [N, C, H, W] → [N, C]. A common head before the classifier.</summary>
public sealed class GlobalAveragePool2d : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        var s = input.Shape;
        return input.Reshape(s[0], s[1], -1).Mean(2);
    }

    /// <inheritdoc />
    public override string ToString() => "GlobalAveragePool2d";
}

/// <summary>Flattens everything after the batch dimension: [N, ...] → [N, features].</summary>
public sealed class Flatten : Module
{
    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Flatten(1);

    /// <inheritdoc />
    public override string ToString() => "Flatten";
}
