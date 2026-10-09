// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.


namespace Idrak.Layers;

/// <summary>
/// 2-D convolution over [N, C, H, W] images, producing [N, outChannels, OH, OW]. Kernels, strides, padding and dilation
/// may differ between height and width (a text line's tall or wide filters); <c>groups</c> splits the channels into
/// groups convolved separately (depthwise when it equals the input channels). Runs as one device operation
/// (<see cref="Tensor.Convolution"/>): on the CPU as patch unfolding (im2col) and the tiled products, depthwise as direct
/// loops; on GPUs as the device's own kernels (implicit products, depthwise) or the unfolded patches on its matrix kernels,
/// whichever is measured faster for the shape. In inference, a following <see cref="BatchNorm"/> (evaluation mode) and
/// activation fold into the same pass (see <see cref="Sequential"/>). Weights start He-uniform (suited to ReLU).
/// </summary>
public sealed partial class Conv2d : Module
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
        : this(inChannels, outChannels, (kernelSize, kernelSize), (stride, stride), (padding, padding), null, 1, bias, device, random)
    {
    }

    /// <summary>Creates the layer with a filter, stride, padding and dilation each given as (height, width), and channel groups.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Number of filters.</param>
    /// <param name="kernelSize">Filter (height, width).</param>
    /// <param name="stride">Step between filter positions (default (1, 1)).</param>
    /// <param name="padding">Zero padding above and below, left and right (default none).</param>
    /// <param name="dilation">Step between the rows and columns a filter reads (default (1, 1): adjacent).</param>
    /// <param name="groups">Channel groups: each group of inChannels / groups inputs feeds outChannels / groups filters
    /// (1: every filter sees every channel; inChannels: depthwise).</param>
    /// <param name="bias">Whether to learn a per-filter bias.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    public Conv2d(int inChannels, int outChannels, (int Height, int Width) kernelSize, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null,
        (int Height, int Width)? dilation = null, int groups = 1, bool bias = true, Device? device = null, Random? random = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inChannels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outChannels);
        InChannels = inChannels;
        OutChannels = outChannels;
        (KernelHeight, KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth, DilationHeight, DilationWidth, Groups) =
            Check(inChannels, outChannels, kernelSize, stride ?? (1, 1), padding ?? (0, 0), dilation ?? (1, 1), groups);
        device ??= Device.Default;
        int fanIn = inChannels / groups * KernelHeight * KernelWidth;
        Weight = CreateParameter(UniformValues(outChannels * fanIn, MathF.Sqrt(6f / fanIn), random ?? Random.Shared), [outChannels, fanIn], device);
        Bias = bias ? CreateParameter(new float[outChannels], [outChannels], device) : null;
    }

    private Conv2d(Tensor weight, Tensor? bias, int inChannels, (int, int, int, int, int, int, int, int, int) geometry)
    {
        InChannels = inChannels;
        OutChannels = weight.Shape[0];
        (KernelHeight, KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth, DilationHeight, DilationWidth, Groups) = geometry;
        Weight = weight;
        Bias = bias;
    }

    private static (int, int, int, int, int, int, int, int, int) Check(int inChannels, int outChannels, (int Height, int Width) kernel, (int Height, int Width) stride,
        (int Height, int Width) padding, (int Height, int Width) dilation, int groups)
    {
        if (kernel.Height <= 0 || kernel.Width <= 0 || stride.Height <= 0 || stride.Width <= 0 || padding.Height < 0 || padding.Width < 0
            || dilation.Height <= 0 || dilation.Width <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(kernel), $"Conv2d needs positive kernel sizes, strides and dilations and no negative padding; got kernel {kernel}, "
                + $"stride {stride}, padding {padding}, dilation {dilation}.");
        }

        if (groups <= 0 || inChannels % groups != 0 || outChannels % groups != 0)
        {
            throw new ArgumentException($"Conv2d groups divide both the {inChannels} input and the {outChannels} output channels; got {groups}.", nameof(groups));
        }

        return (kernel.Height, kernel.Width, stride.Height, stride.Width, padding.Height, padding.Width, dilation.Height, dilation.Width, groups);
    }

    /// <summary>
    /// A layer around existing filters (for example loaded weights); the layer takes ownership. The weight is
    /// [outChannels, inChannels, k, k] (PyTorch's layout) or the same values as [outChannels, inChannels · k · k].
    /// </summary>
    public static Conv2d FromWeights(Tensor weight, Tensor? bias, int kernelSize, int stride = 1, int padding = 0) =>
        FromWeights(weight, bias, (kernelSize, kernelSize), (stride, stride), (padding, padding));

    /// <summary>
    /// A layer around existing filters with a (height, width) filter, stride, padding and dilation, and channel groups; the
    /// layer takes ownership. The weight is [outChannels, inChannels / groups, kh, kw] (PyTorch's layout) or the same values
    /// as [outChannels, inChannels / groups · kh · kw].
    /// </summary>
    public static Conv2d FromWeights(Tensor weight, Tensor? bias, (int Height, int Width) kernelSize, (int Height, int Width)? stride = null,
        (int Height, int Width)? padding = null, (int Height, int Width)? dilation = null, int groups = 1)
    {
        ArgumentNullException.ThrowIfNull(weight);
        var shape = weight.Shape;
        int area = Math.Max(kernelSize.Height, 1) * Math.Max(kernelSize.Width, 1);
        bool fits = shape.Length switch
        {
            4 => shape[2] == kernelSize.Height && shape[3] == kernelSize.Width,
            2 => shape[1] % area == 0,
            _ => false,
        };
        if (!fits || (bias is not null && (bias.Rank != 1 || bias.Shape[0] != shape[0])))
        {
            var (kh, kw) = kernelSize;
            throw new ArgumentException($"Conv2d filters of {kh}x{kw} are [out, in / groups, {kh}, {kw}] or [out, in / groups · {kh * kw}] with a bias [out]; "
                + $"got {Tensor.FormatShape(shape)} and {(bias is null ? "no bias" : Tensor.FormatShape(bias.Shape))}.");
        }

        int perGroup = shape.Length == 4 ? shape[1] : shape[1] / area;
        var geometry = Check(perGroup * Math.Max(groups, 1), shape[0], kernelSize, stride ?? (1, 1), padding ?? (0, 0), dilation ?? (1, 1), groups);
        var flat = weight;
        if (shape.Length == 4)
        {
            // The same values as [out, in / groups · kh · kw] (the order Im2Col writes a patch in), in a tensor of its own.
            flat = Tensor.Persistent(weight.ToArray(), [shape[0], perGroup * area], weight.Device, weight.RequiresGrad);
            weight.Dispose();
        }

        return new Conv2d(flat, bias, perGroup * groups, geometry);
    }

    /// <summary>Input channels.</summary>
    public int InChannels { get; }

    /// <summary>Output channels (filters).</summary>
    public int OutChannels { get; }

    /// <summary>Filter size: its height, which is also its width for a square filter (see <see cref="KernelWidth"/>).</summary>
    public int KernelSize => KernelHeight;

    /// <summary>Filter height.</summary>
    public int KernelHeight { get; }

    /// <summary>Filter width.</summary>
    public int KernelWidth { get; }

    /// <summary>Filter step: the vertical one, which is also the horizontal one unless they differ (see <see cref="StrideWidth"/>).</summary>
    public int Stride => StrideHeight;

    /// <summary>Vertical filter step.</summary>
    public int StrideHeight { get; }

    /// <summary>Horizontal filter step.</summary>
    public int StrideWidth { get; }

    /// <summary>Zero padding per border: above and below, which is also left and right unless they differ (see <see cref="PaddingWidth"/>).</summary>
    public int Padding => PaddingHeight;

    /// <summary>Zero padding above and below.</summary>
    public int PaddingHeight { get; }

    /// <summary>Zero padding left and right.</summary>
    public int PaddingWidth { get; }

    /// <summary>Step between the rows a filter reads (1: adjacent).</summary>
    public int DilationHeight { get; }

    /// <summary>Step between the columns a filter reads (1: adjacent).</summary>
    public int DilationWidth { get; }

    /// <summary>Channel groups (1: every filter sees every channel).</summary>
    public int Groups { get; }

    /// <summary>Whether the filter, stride, padding and dilation are each the same for height and width, with dilation 1 and one group: the layer the square constructor makes.</summary>
    public bool IsSquare => KernelHeight == KernelWidth && StrideHeight == StrideWidth && PaddingHeight == PaddingWidth && DilationHeight == 1 && DilationWidth == 1 && Groups == 1;

    /// <summary>Filters as [outChannels, inChannels / groups · kh · kw].</summary>
    public Tensor Weight { get; private set; }

    /// <summary>Per-filter bias, or null.</summary>
    public Tensor? Bias { get; private set; }

    /// <summary>The window geometry of this layer over an [N, C, H, W] input.</summary>
    public ConvGeometry Geometry(int batch, int height, int width) =>
        new(batch, InChannels, height, width, KernelHeight, KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth)
        {
            DH = DilationHeight, DW = DilationWidth,
        };

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 4 || input.Shape[1] != InChannels)
        {
            throw new ArgumentException($"Conv2d expects [N, {InChannels}, H, W], got {Tensor.FormatShape(input.Shape)}.");
        }

        var g = Geometry(input.Shape[0], input.Shape[2], input.Shape[3]);
        if (g.OH <= 0 || g.OW <= 0)
        {
            throw new ArgumentException($"A {KernelHeight}x{KernelWidth} kernel (dilation {DilationHeight}x{DilationWidth}) does not fit a {g.H}x{g.W} input with padding {PaddingHeight}x{PaddingWidth}.");
        }

        return input.Convolution(Weight, Bias, g, Groups);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => Bias is null ? [Weight] : [Weight, Bias];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        Weight = MoveTensor(Weight, device);
        Bias = Bias is null ? null : MoveTensor(Bias, device);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        static string Pair(int h, int w) => h == w ? $"{h}" : $"{h}x{w}";
        string text = $"Conv2d({InChannels} -> {OutChannels}, {KernelHeight}x{KernelWidth}, stride {Pair(StrideHeight, StrideWidth)}, padding {Pair(PaddingHeight, PaddingWidth)}";
        if (DilationHeight != 1 || DilationWidth != 1)
        {
            text += $", dilation {Pair(DilationHeight, DilationWidth)}";
        }

        return text + (Groups != 1 ? $", groups {Groups})" : ")");
    }
}

/// <summary>
/// 2-D transposed convolution (PyTorch's <c>ConvTranspose2d</c>, the gradient of a convolution with respect to its input):
/// [N, inChannels, H, W] → [N, outChannels, OH, OW] with OH = (H - 1)·strideH - 2·paddingH + dilationH·(kh - 1) +
/// outputPaddingH + 1, and the same for the width. Each input position's products with the filters are spread over the
/// window it scatters to and added where windows overlap. Runs as <see cref="Conv2d"/>'s input gradient
/// (<see cref="Tensor.ConvolutionTranspose"/>, one device operation: the products and fold, or the device's implicit
/// product, whichever is measured faster), and its gradient as the convolution. The weight is PyTorch's [inChannels,
/// outChannels / groups, kh, kw], flattened.
/// </summary>
public sealed class ConvTranspose2d : Module
{
    /// <summary>Creates the layer with a square filter.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="kernelSize">Filter height and width.</param>
    /// <param name="stride">Step between the windows the input positions scatter to (the upsampling factor).</param>
    /// <param name="padding">Rows and columns removed from each border of the full output.</param>
    /// <param name="outputPadding">Rows and columns added below and right (less than the stride), to choose between the
    /// output sizes a strided convolution maps to the same input size.</param>
    /// <param name="bias">Whether to learn a per-channel bias.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    public ConvTranspose2d(int inChannels, int outChannels, int kernelSize, int stride = 1, int padding = 0, int outputPadding = 0, bool bias = true, Device? device = null,
        Random? random = null)
        : this(inChannels, outChannels, (kernelSize, kernelSize), (stride, stride), (padding, padding), (outputPadding, outputPadding), null, 1, bias, device, random)
    {
    }

    /// <summary>Creates the layer with a filter, stride, padding, output padding and dilation each given as (height, width), and channel groups.</summary>
    /// <param name="inChannels">Input channels.</param>
    /// <param name="outChannels">Output channels.</param>
    /// <param name="kernelSize">Filter (height, width).</param>
    /// <param name="stride">Step between the windows the input positions scatter to (default (1, 1)).</param>
    /// <param name="padding">Rows and columns removed from each border of the full output (default none).</param>
    /// <param name="outputPadding">Rows and columns added below and right, each less than its stride (default none).</param>
    /// <param name="dilation">Step between the rows and columns a filter writes (default (1, 1)).</param>
    /// <param name="groups">Channel groups: each group of inChannels / groups inputs feeds outChannels / groups outputs.</param>
    /// <param name="bias">Whether to learn a per-channel bias.</param>
    /// <param name="device">Where the parameters live.</param>
    /// <param name="random">Seed source for the initial weights.</param>
    public ConvTranspose2d(int inChannels, int outChannels, (int Height, int Width) kernelSize, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null,
        (int Height, int Width)? outputPadding = null, (int Height, int Width)? dilation = null, int groups = 1, bool bias = true, Device? device = null, Random? random = null)
    {
        Shape = Check(inChannels, outChannels, kernelSize, stride ?? (1, 1), padding ?? (0, 0), outputPadding ?? (0, 0), dilation ?? (1, 1), groups);
        InChannels = inChannels;
        OutChannels = outChannels;
        device ??= Device.Default;
        int columns = outChannels / groups * KernelHeight * KernelWidth, fanIn = inChannels / groups * KernelHeight * KernelWidth;
        Weight = CreateParameter(UniformValues(inChannels * columns, MathF.Sqrt(6f / fanIn), random ?? Random.Shared), [inChannels, columns], device);
        Bias = bias ? CreateParameter(new float[outChannels], [outChannels], device) : null;
    }

    private ConvTranspose2d(Tensor weight, Tensor? bias, int outChannels, Window shape)
    {
        InChannels = weight.Shape[0];
        OutChannels = outChannels;
        Shape = shape;
        Weight = weight;
        Bias = bias;
    }

    // Kernel, stride, padding, output padding and dilation as (height, width), and the groups.
    private readonly record struct Window((int H, int W) Kernel, (int H, int W) Stride, (int H, int W) Padding, (int H, int W) OutputPadding, (int H, int W) Dilation, int Groups);

    private Window Shape { get; }

    // The constructor's checks, for the network builder.
    internal static void Validate(int inChannels, int outChannels, (int Height, int Width) kernel, (int Height, int Width) stride, (int Height, int Width) padding,
        (int Height, int Width) outputPadding, (int Height, int Width) dilation, int groups) =>
        Check(inChannels, outChannels, kernel, stride, padding, outputPadding, dilation, groups);

    private static Window Check(int inChannels, int outChannels, (int Height, int Width) kernel, (int Height, int Width) stride, (int Height, int Width) padding,
        (int Height, int Width) outputPadding, (int Height, int Width) dilation, int groups)
    {
        if (kernel.Height <= 0 || kernel.Width <= 0 || stride.Height <= 0 || stride.Width <= 0 || padding.Height < 0 || padding.Width < 0 || dilation.Height <= 0
            || dilation.Width <= 0 || outputPadding.Height < 0 || outputPadding.Width < 0 || outputPadding.Height >= stride.Height || outputPadding.Width >= stride.Width)
        {
            throw new ArgumentOutOfRangeException(nameof(kernel), $"ConvTranspose2d needs positive kernel sizes, strides and dilations, no negative padding and an output "
                + $"padding below the stride; got kernel {kernel}, stride {stride}, padding {padding}, output padding {outputPadding}, dilation {dilation}.");
        }

        if (inChannels <= 0 || outChannels <= 0 || groups <= 0 || inChannels % groups != 0 || outChannels % groups != 0)
        {
            throw new ArgumentException($"ConvTranspose2d groups divide both the {inChannels} input and the {outChannels} output channels; got {groups}.", nameof(groups));
        }

        return new(kernel, stride, padding, outputPadding, dilation, groups);
    }

    /// <summary>
    /// A layer around existing filters (for example loaded weights); the layer takes ownership. The weight is
    /// [inChannels, outChannels / groups, kh, kw] (PyTorch's layout) or the same values as [inChannels, outChannels / groups · kh · kw].
    /// </summary>
    public static ConvTranspose2d FromWeights(Tensor weight, Tensor? bias, (int Height, int Width) kernelSize, (int Height, int Width)? stride = null,
        (int Height, int Width)? padding = null, (int Height, int Width)? outputPadding = null, (int Height, int Width)? dilation = null, int groups = 1)
    {
        ArgumentNullException.ThrowIfNull(weight);
        var shape = weight.Shape;
        int area = Math.Max(kernelSize.Height, 1) * Math.Max(kernelSize.Width, 1);
        bool fits = shape.Length switch
        {
            4 => shape[2] == kernelSize.Height && shape[3] == kernelSize.Width,
            2 => shape[1] % area == 0,
            _ => false,
        };
        int perGroup = shape.Length == 4 ? shape[1] : shape.Length == 2 ? shape[1] / area : 0;
        int outChannels = perGroup * Math.Max(groups, 1);
        if (!fits || (bias is not null && (bias.Rank != 1 || bias.Shape[0] != outChannels)))
        {
            var (kh, kw) = kernelSize;
            throw new ArgumentException($"ConvTranspose2d filters of {kh}x{kw} are [in, out / groups, {kh}, {kw}] or [in, out / groups · {kh * kw}] with a bias [out]; "
                + $"got {Tensor.FormatShape(shape)} and {(bias is null ? "no bias" : Tensor.FormatShape(bias.Shape))}.");
        }

        var window = Check(shape[0], outChannels, kernelSize, stride ?? (1, 1), padding ?? (0, 0), outputPadding ?? (0, 0), dilation ?? (1, 1), groups);
        var flat = weight;
        if (shape.Length == 4)
        {
            flat = Tensor.Persistent(weight.ToArray(), [shape[0], perGroup * area], weight.Device, weight.RequiresGrad);
            weight.Dispose();
        }

        return new ConvTranspose2d(flat, bias, outChannels, window);
    }

    /// <summary>Input channels.</summary>
    public int InChannels { get; }

    /// <summary>Output channels.</summary>
    public int OutChannels { get; }

    /// <summary>Filter height.</summary>
    public int KernelHeight => Shape.Kernel.H;

    /// <summary>Filter width.</summary>
    public int KernelWidth => Shape.Kernel.W;

    /// <summary>Vertical stride (the vertical upsampling factor).</summary>
    public int StrideHeight => Shape.Stride.H;

    /// <summary>Horizontal stride.</summary>
    public int StrideWidth => Shape.Stride.W;

    /// <summary>Rows removed from the top and the bottom of the full output.</summary>
    public int PaddingHeight => Shape.Padding.H;

    /// <summary>Columns removed from the left and the right of the full output.</summary>
    public int PaddingWidth => Shape.Padding.W;

    /// <summary>Rows added below.</summary>
    public int OutputPaddingHeight => Shape.OutputPadding.H;

    /// <summary>Columns added right.</summary>
    public int OutputPaddingWidth => Shape.OutputPadding.W;

    /// <summary>Step between the rows a filter writes.</summary>
    public int DilationHeight => Shape.Dilation.H;

    /// <summary>Step between the columns a filter writes.</summary>
    public int DilationWidth => Shape.Dilation.W;

    /// <summary>Channel groups.</summary>
    public int Groups => Shape.Groups;

    /// <summary>Filters as [inChannels, outChannels / groups · kh · kw].</summary>
    public Tensor Weight { get; private set; }

    /// <summary>Per-channel bias, or null.</summary>
    public Tensor? Bias { get; private set; }

    /// <summary>The output height and width for an input of <paramref name="height"/> x <paramref name="width"/>.</summary>
    public (int Height, int Width) OutputSize(int height, int width) =>
        ((height - 1) * StrideHeight - 2 * PaddingHeight + DilationHeight * (KernelHeight - 1) + OutputPaddingHeight + 1,
         (width - 1) * StrideWidth - 2 * PaddingWidth + DilationWidth * (KernelWidth - 1) + OutputPaddingWidth + 1);

    /// <summary>
    /// The window geometry of the fold over the [N, outChannels, OH, OW] output: the convolution whose gradient this layer
    /// computes, whose windows are the input's H x W positions.
    /// </summary>
    public ConvGeometry Geometry(int batch, int height, int width)
    {
        var (oh, ow) = OutputSize(height, width);
        return new(batch, OutChannels, oh, ow, KernelHeight, KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth) { DH = DilationHeight, DW = DilationWidth };
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 4 || input.Shape[1] != InChannels)
        {
            throw new ArgumentException($"ConvTranspose2d expects [N, {InChannels}, H, W], got {Tensor.FormatShape(input.Shape)}.");
        }

        int n = input.Shape[0], h = input.Shape[2], w = input.Shape[3];
        var g = Geometry(n, h, w);
        if (g.H <= 0 || g.W <= 0 || g.OH != h || g.OW != w)
        {
            throw new ArgumentException($"{this} gives an empty output for a {h}x{w} input.");
        }

        // The input gradient of the convolution whose windows are the input's positions, as one device operation.
        return input.ConvolutionTranspose(Weight, Bias, g, Groups);
    }

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => Bias is null ? [Weight] : [Weight, Bias];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        Weight = MoveTensor(Weight, device);
        Bias = Bias is null ? null : MoveTensor(Bias, device);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        static string Pair(int h, int w) => h == w ? $"{h}" : $"{h}x{w}";
        string text = $"ConvTranspose2d({InChannels} -> {OutChannels}, {KernelHeight}x{KernelWidth}, stride {Pair(StrideHeight, StrideWidth)}, padding {Pair(PaddingHeight, PaddingWidth)}";
        if (OutputPaddingHeight != 0 || OutputPaddingWidth != 0)
        {
            text += $", output padding {Pair(OutputPaddingHeight, OutputPaddingWidth)}";
        }

        if (DilationHeight != 1 || DilationWidth != 1)
        {
            text += $", dilation {Pair(DilationHeight, DilationWidth)}";
        }

        return text + (Groups != 1 ? $", groups {Groups})" : ")");
    }
}

/// <summary>
/// 2-D max pooling over [N, C, H, W]: keeps the largest value of each window. Padding (which never wins) may be larger
/// below and right than above and left (ONNX's four pads), and <see cref="CeilMode"/> keeps a last, partial window as
/// PyTorch's <c>ceil_mode</c>.
/// </summary>
public sealed class MaxPool2d : Module
{
    /// <summary>Creates the layer with a square window.</summary>
    /// <param name="kernelSize">Window height and width.</param>
    /// <param name="stride">Step between windows; defaults to the window size (non-overlapping).</param>
    /// <param name="padding">Border padding (padded positions never win).</param>
    public MaxPool2d(int kernelSize, int? stride = null, int padding = 0)
        : this((kernelSize, kernelSize), stride is { } s ? (s, s) : null, (padding, padding))
    {
    }

    /// <summary>Creates the layer with a (height, width) window, stride and padding.</summary>
    /// <param name="kernelSize">Window (height, width).</param>
    /// <param name="stride">Step between windows; defaults to the window size (non-overlapping).</param>
    /// <param name="padding">Border padding above and left (and below and right unless <paramref name="paddingEnd"/> is given); padded positions never win.</param>
    /// <param name="ceilMode">Whether a last window that starts inside the input but runs past its padded end counts (PyTorch's ceil_mode).</param>
    /// <param name="paddingEnd">Padding below and right, when it differs from <paramref name="padding"/>.</param>
    public MaxPool2d((int Height, int Width) kernelSize, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null, bool ceilMode = false,
        (int Height, int Width)? paddingEnd = null)
    {
        (KernelHeight, KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth, PaddingBottom, PaddingRight) = Pool.Check("MaxPool2d", kernelSize, stride, padding, paddingEnd);
        CeilMode = ceilMode;
    }

    /// <summary>Window size: its height, which is also its width for a square window (see <see cref="KernelWidth"/>).</summary>
    public int KernelSize => KernelHeight;

    /// <summary>Window height.</summary>
    public int KernelHeight { get; }

    /// <summary>Window width.</summary>
    public int KernelWidth { get; }

    /// <summary>Window step: the vertical one (see <see cref="StrideWidth"/>).</summary>
    public int Stride => StrideHeight;

    /// <summary>Vertical window step.</summary>
    public int StrideHeight { get; }

    /// <summary>Horizontal window step.</summary>
    public int StrideWidth { get; }

    /// <summary>Border padding: above (see <see cref="PaddingWidth"/>, <see cref="PaddingBottom"/>).</summary>
    public int Padding => PaddingHeight;

    /// <summary>Padding above (and below unless <see cref="PaddingBottom"/> differs).</summary>
    public int PaddingHeight { get; }

    /// <summary>Padding left (and right unless <see cref="PaddingRight"/> differs).</summary>
    public int PaddingWidth { get; }

    /// <summary>Padding below.</summary>
    public int PaddingBottom { get; }

    /// <summary>Padding right.</summary>
    public int PaddingRight { get; }

    /// <summary>Whether a last window running past the padded end counts (PyTorch's ceil_mode).</summary>
    public bool CeilMode { get; }

    /// <summary>Whether window, stride and padding are each the same for height and width, padding the same on both sides and no ceil mode.</summary>
    public bool IsSquare => KernelHeight == KernelWidth && StrideHeight == StrideWidth && PaddingHeight == PaddingWidth && !Pool.Uneven(this) && !CeilMode;

    /// <summary>The window geometry of this layer over an [N, C, H, W] input (its padding below and right includes what the ceil mode adds).</summary>
    public ConvGeometry Geometry(int batch, int channels, int height, int width) =>
        Pool.Geometry(batch, channels, height, width, (KernelHeight, KernelWidth), (StrideHeight, StrideWidth), (PaddingHeight, PaddingWidth), (PaddingBottom, PaddingRight), CeilMode);

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException($"MaxPool2d expects [N, C, H, W], got {Tensor.FormatShape(input.Shape)}.");
        }

        var s = input.Shape;
        var g = Geometry(s[0], s[1], s[2], s[3]);
        if (g.OH <= 0 || g.OW <= 0)
        {
            throw new ArgumentException($"A {KernelHeight}x{KernelWidth} window does not fit a {s[2]}x{s[3]} input with padding {PaddingHeight}x{PaddingWidth}.");
        }

        return input.MaxPool(g);
    }

    /// <inheritdoc />
    public override string ToString() => IsSquare
        ? $"MaxPool2d({KernelHeight}x{KernelWidth}, stride {StrideHeight})"
        : $"MaxPool2d({KernelHeight}x{KernelWidth}, stride {StrideHeight}x{StrideWidth}, padding {PaddingHeight}x{PaddingWidth}{Pool.Extras(PaddingBottom, PaddingRight, Pool.Uneven(this), CeilMode)})";
}

/// <summary>
/// 2-D average pooling over [N, C, H, W]: the mean of each window. As PyTorch's <c>AvgPool2d</c>, padded positions count as
/// zeros in the mean unless <see cref="CountIncludePad"/> is false (then each window is divided by the input positions it
/// covers); with <see cref="CeilMode"/> a last, partial window counts, divided by the positions it covers up to the padded
/// end. Padding may be larger below and right than above and left (ONNX's four pads).
/// </summary>
public sealed class AvgPool2d : Module
{
    /// <summary>Creates the layer with a square window.</summary>
    /// <param name="kernelSize">Window height and width.</param>
    /// <param name="stride">Step between windows; defaults to the window size (non-overlapping).</param>
    /// <param name="padding">Zero padding on each border.</param>
    /// <param name="countIncludePad">Whether padded zeros count in each window's mean.</param>
    public AvgPool2d(int kernelSize, int? stride = null, int padding = 0, bool countIncludePad = true)
        : this((kernelSize, kernelSize), stride is { } s ? (s, s) : null, (padding, padding), countIncludePad)
    {
    }

    /// <summary>Creates the layer with a (height, width) window, stride and padding.</summary>
    /// <param name="kernelSize">Window (height, width).</param>
    /// <param name="stride">Step between windows; defaults to the window size (non-overlapping).</param>
    /// <param name="padding">Zero padding above and left (and below and right unless <paramref name="paddingEnd"/> is given).</param>
    /// <param name="countIncludePad">Whether padded zeros count in each window's mean.</param>
    /// <param name="ceilMode">Whether a last window that starts inside the input but runs past its padded end counts (PyTorch's ceil_mode).</param>
    /// <param name="paddingEnd">Zero padding below and right, when it differs from <paramref name="padding"/>.</param>
    public AvgPool2d((int Height, int Width) kernelSize, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null, bool countIncludePad = true,
        bool ceilMode = false, (int Height, int Width)? paddingEnd = null)
    {
        (KernelHeight, KernelWidth, StrideHeight, StrideWidth, PaddingHeight, PaddingWidth, PaddingBottom, PaddingRight) = Pool.Check("AvgPool2d", kernelSize, stride, padding, paddingEnd);
        CountIncludePad = countIncludePad;
        CeilMode = ceilMode;
    }

    /// <summary>Window height.</summary>
    public int KernelHeight { get; }

    /// <summary>Window width.</summary>
    public int KernelWidth { get; }

    /// <summary>Vertical window step.</summary>
    public int StrideHeight { get; }

    /// <summary>Horizontal window step.</summary>
    public int StrideWidth { get; }

    /// <summary>Zero padding above (and below unless <see cref="PaddingBottom"/> differs).</summary>
    public int PaddingHeight { get; }

    /// <summary>Zero padding left (and right unless <see cref="PaddingRight"/> differs).</summary>
    public int PaddingWidth { get; }

    /// <summary>Zero padding below.</summary>
    public int PaddingBottom { get; }

    /// <summary>Zero padding right.</summary>
    public int PaddingRight { get; }

    /// <summary>Whether padded zeros count in each window's mean (PyTorch's default) or only the input positions it covers.</summary>
    public bool CountIncludePad { get; }

    /// <summary>Whether a last window running past the padded end counts (PyTorch's ceil_mode).</summary>
    public bool CeilMode { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank != 4)
        {
            throw new ArgumentException($"AvgPool2d expects [N, C, H, W], got {Tensor.FormatShape(input.Shape)}.");
        }

        var s = input.Shape;
        var g = Pool.Geometry(s[0], s[1], s[2], s[3], (KernelHeight, KernelWidth), (StrideHeight, StrideWidth), (PaddingHeight, PaddingWidth), (PaddingBottom, PaddingRight), CeilMode);
        if (g.OH <= 0 || g.OW <= 0)
        {
            throw new ArgumentException($"A {KernelHeight}x{KernelWidth} window does not fit a {s[2]}x{s[3]} input with padding {PaddingHeight}x{PaddingWidth}.");
        }

        return input.AvgPool(g, CountIncludePad, PaddingBottom, PaddingRight);
    }

    /// <inheritdoc />
    public override string ToString() => $"AvgPool2d({KernelHeight}x{KernelWidth}, stride {StrideHeight}x{StrideWidth}, padding {PaddingHeight}x{PaddingWidth}"
        + Pool.Extras(PaddingBottom, PaddingRight, PaddingBottom != PaddingHeight || PaddingRight != PaddingWidth, CeilMode) + (CountIncludePad ? ")" : ", padding not counted)");
}

// The checks and window geometry pooling layers share.
internal static class Pool
{
    public static (int, int, int, int, int, int, int, int) Check(string layer, (int Height, int Width) kernel, (int Height, int Width)? stride, (int Height, int Width)? padding,
        (int Height, int Width)? paddingEnd = null)
    {
        var (sh, sw) = stride ?? kernel;
        var (ph, pw) = padding ?? (0, 0);
        var (pb, pr) = paddingEnd ?? (ph, pw);
        if (kernel.Height <= 0 || kernel.Width <= 0 || sh <= 0 || sw <= 0 || ph < 0 || pw < 0 || pb < 0 || pr < 0
            || 2 * ph > kernel.Height || 2 * pw > kernel.Width || 2 * pb > kernel.Height || 2 * pr > kernel.Width)
        {
            throw new ArgumentOutOfRangeException(nameof(kernel), $"{layer} needs a positive window and stride and padding of at most half the window on each side; "
                + $"got window {kernel}, stride ({sh}, {sw}), padding ({ph}, {pw})" + (paddingEnd is null ? "." : $" above and left, ({pb}, {pr}) below and right."));
        }

        return (kernel.Height, kernel.Width, sh, sw, ph, pw, pb, pr);
    }

    /// <summary>
    /// The windows along one axis, as PyTorch and ONNX count them: (size + padBegin + padEnd - kernel) / stride + 1, rounded
    /// up in ceil mode, where a last window must still start inside the input or the padding before it.
    /// </summary>
    public static int Windows(int size, int kernel, int stride, int padBegin, int padEnd, bool ceilMode)
    {
        int span = size + padBegin + padEnd - kernel;
        if (span < 0)
        {
            return 0;
        }

        int windows = (ceilMode ? (span + stride - 1) / stride : span / stride) + 1;
        return ceilMode && (windows - 1) * stride >= size + padBegin ? windows - 1 : windows;
    }

    // The geometry whose padding below and right gives exactly Windows(...) windows (the ceil mode's last one included).
    public static ConvGeometry Geometry(int n, int c, int h, int w, (int H, int W) kernel, (int H, int W) stride, (int H, int W) padding, (int H, int W) paddingEnd, bool ceilMode)
    {
        int End(int size, int k, int s, int pb, int pe)
        {
            int windows = Windows(size, k, s, pb, pe, ceilMode);
            return ceilMode && windows > 0 ? Math.Max((windows - 1) * s + k - size - pb, 0) : pe;
        }

        return new ConvGeometry(n, c, h, w, kernel.H, kernel.W, stride.H, stride.W, padding.H, padding.W)
        {
            PadBottom = End(h, kernel.H, stride.H, padding.H, paddingEnd.H), PadRight = End(w, kernel.W, stride.W, padding.W, paddingEnd.W),
        };
    }

    public static bool Uneven(MaxPool2d m) => m.PaddingBottom != m.PaddingHeight || m.PaddingRight != m.PaddingWidth;

    // What a pooling layer's text adds for padding below and right and the ceil mode.
    public static string Extras(int bottom, int right, bool uneven, bool ceilMode) =>
        (uneven ? $" ({bottom}x{right} below and right)" : "") + (ceilMode ? ", ceil mode" : "");
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
