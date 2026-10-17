// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Layers;

/// <summary>
/// A convolution over time: <c>[C, T]</c>, <c>[B, C, T]</c> or <c>[B, C, 1, T]</c> in, the same rank out, through the
/// library's <see cref="Conv2d"/> with a (1, k) filter (every device's convolution kernels, one launch for a batch).
/// Optionally weight-normed (<see cref="WeightNorm"/>): the weight g · v / ‖v‖ is made per pass, or once per step when
/// bound, with the gradients reaching g and v.
/// </summary>
public sealed class Conv1d : Module
{
    /// <summary>
    /// Over an existing 2-D convolution, with the parametrization of its weight when it has one (v is then the
    /// convolution's weight). A (1, k) filter takes <c>[C, T]</c> and <c>[B, C, T]</c>; any filter takes <c>[B, C, H, W]</c>
    /// (a discriminator folding a signal into rows of a period runs its (k, 1) filters this way, weight-normed).
    /// </summary>
    public Conv1d(Conv2d convolution, WeightNorm? weightNorm = null)
    {
        ArgumentNullException.ThrowIfNull(convolution);
        if (weightNorm is not null && !ReferenceEquals(weightNorm.Direction, convolution.Weight))
        {
            throw new ArgumentException("The weight norm's direction is the convolution's weight.", nameof(weightNorm));
        }

        Convolution = convolution;
        WeightNorm = weightNorm;
    }

    /// <summary>A new convolution of <paramref name="kernel"/> taps, drawn as <see cref="Conv2d"/> draws its filters.</summary>
    public Conv1d(int inChannels, int outChannels, int kernel, int stride = 1, int padding = 0, int dilation = 1, int groups = 1, bool bias = true, Device? device = null, Random? random = null)
        : this(new Conv2d(inChannels, outChannels, (1, kernel), (1, stride), (0, padding), (1, dilation), groups, bias, device, random))
    {
    }

    /// <summary>The 2-D convolution underneath (its weight is the direction when weight-normed).</summary>
    public Conv2d Convolution { get; }

    /// <summary>The weight's parametrization, when it has one.</summary>
    public WeightNorm? WeightNorm { get; }

    /// <summary>Input channels.</summary>
    public int InChannels => Convolution.InChannels;

    /// <summary>Output channels.</summary>
    public int OutChannels => Convolution.OutChannels;

    /// <summary>The filter's taps.</summary>
    public int Kernel => Convolution.KernelWidth;

    /// <summary>The stride over time.</summary>
    public int Stride => Convolution.StrideWidth;

    /// <summary>The zero padding on each side.</summary>
    public int Padding => Convolution.PaddingWidth;

    /// <summary>The filter's dilation.</summary>
    public int Dilation => Convolution.DilationWidth;

    /// <summary>The groups.</summary>
    public int Groups => Convolution.Groups;

    /// <summary>The bias, or null.</summary>
    public Tensor? Bias => Convolution.Bias;

    /// <summary>The weight the next pass uses: the normed one (bound or made now) or the convolution's own.</summary>
    public Tensor Weight => WeightNorm?.Weight ?? Convolution.Weight;

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => WeightNorm is null ? Convolution.Parameters() : [.. Convolution.Parameters(), WeightNorm.Gain];

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Shape.Length is 2 or 3 && (Convolution.KernelHeight != 1 || Convolution.StrideHeight != 1 || Convolution.PaddingHeight != 0))
        {
            throw new ArgumentException($"A convolution over time of {Tensor.FormatShape(input.Shape)} needs a (1, k) filter with a stride of 1 and no padding over the rows; this one is ({Convolution.KernelHeight}, {Convolution.KernelWidth}): give it [B, C, H, W].", nameof(input));
        }

        switch (input.Shape.Length)
        {
            case 2:
            {
                var y = Run(input.Reshape(1, input.Shape[0], 1, input.Shape[1]));
                return y.Reshape(y.Shape[1], y.Shape[3]);
            }

            case 3:
            {
                var y = Run(input.Reshape(input.Shape[0], input.Shape[1], 1, input.Shape[2]));
                return y.Reshape(y.Shape[0], y.Shape[1], y.Shape[3]);
            }

            case 4:
                return Run(input);
            default:
                throw new ArgumentException($"A convolution over time takes [C, T], [B, C, T] or [B, C, H, W], not {Tensor.FormatShape(input.Shape)}.", nameof(input));
        }
    }

    private Tensor Run(Tensor x) => WeightNorm is null
        ? Convolution.Forward(x)
        : x.Convolution(WeightNorm.Weight, Convolution.Bias, Convolution.Geometry(x.Shape[0], x.Shape[2], x.Shape[3]), Convolution.Groups);

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("A convolution over time is built on its device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        WeightNorm?.Dispose();
        WeightNorm?.Gain.Dispose();
        Convolution.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"Conv1d({InChannels} -> {OutChannels}, k{Kernel}{(Stride != 1 ? $" s{Stride}" : "")}{(Dilation != 1 ? $" d{Dilation}" : "")}{(WeightNorm is null ? "" : ", weight-normed")})";
}

/// <summary>
/// A transposed convolution over time (upsampling by its stride): <c>[C, T]</c> or <c>[B, C, T]</c> in, the same rank out,
/// through the library's <see cref="ConvTranspose2d"/> with a (1, k) filter; optionally weight-normed as <see cref="Conv1d"/>.
/// </summary>
public sealed class ConvTranspose1d : Module
{
    /// <summary>Over an existing 2-D transposed convolution whose filters are (1, k), with the parametrization of its weight when it has one.</summary>
    public ConvTranspose1d(ConvTranspose2d convolution, WeightNorm? weightNorm = null)
    {
        ArgumentNullException.ThrowIfNull(convolution);
        if (convolution.KernelHeight != 1 || convolution.StrideHeight != 1 || convolution.PaddingHeight != 0)
        {
            throw new ArgumentException($"A transposed convolution over time has a (1, k) filter, a stride of 1 and no padding over the rows; this one is ({convolution.KernelHeight}, {convolution.KernelWidth}).", nameof(convolution));
        }

        if (weightNorm is not null && !ReferenceEquals(weightNorm.Direction, convolution.Weight))
        {
            throw new ArgumentException("The weight norm's direction is the convolution's weight.", nameof(weightNorm));
        }

        Convolution = convolution;
        WeightNorm = weightNorm;
    }

    /// <summary>A new transposed convolution of <paramref name="kernel"/> taps and <paramref name="stride"/>, drawn as <see cref="ConvTranspose2d"/> draws its filters.</summary>
    public ConvTranspose1d(int inChannels, int outChannels, int kernel, int stride = 1, int padding = 0, int outputPadding = 0, bool bias = true, Device? device = null, Random? random = null)
        : this(new ConvTranspose2d(inChannels, outChannels, (1, kernel), (1, stride), (0, padding), (0, outputPadding), null, 1, bias, device, random))
    {
    }

    /// <summary>The 2-D transposed convolution underneath.</summary>
    public ConvTranspose2d Convolution { get; }

    /// <summary>The weight's parametrization, when it has one.</summary>
    public WeightNorm? WeightNorm { get; }

    /// <summary>Input channels.</summary>
    public int InChannels => Convolution.InChannels;

    /// <summary>Output channels.</summary>
    public int OutChannels => Convolution.OutChannels;

    /// <summary>The filter's taps.</summary>
    public int Kernel => Convolution.KernelWidth;

    /// <summary>The stride over time (the upsampling factor).</summary>
    public int Stride => Convolution.StrideWidth;

    /// <summary>The bias, or null.</summary>
    public Tensor? Bias => Convolution.Bias;

    /// <summary>The weight the next pass uses: the normed one (bound or made now) or the convolution's own.</summary>
    public Tensor Weight => WeightNorm?.Weight ?? Convolution.Weight;

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => WeightNorm is null ? Convolution.Parameters() : [.. Convolution.Parameters(), WeightNorm.Gain];

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        switch (input.Shape.Length)
        {
            case 2:
            {
                var y = Run(input.Reshape(1, input.Shape[0], 1, input.Shape[1]));
                return y.Reshape(y.Shape[1], y.Shape[3]);
            }

            case 3:
            {
                var y = Run(input.Reshape(input.Shape[0], input.Shape[1], 1, input.Shape[2]));
                return y.Reshape(y.Shape[0], y.Shape[1], y.Shape[3]);
            }

            case 4:
                return Run(input);
            default:
                throw new ArgumentException($"A transposed convolution over time takes [C, T], [B, C, T] or [B, C, 1, T], not {Tensor.FormatShape(input.Shape)}.", nameof(input));
        }
    }

    private Tensor Run(Tensor x) => WeightNorm is null
        ? Convolution.Forward(x)
        : x.ConvolutionTranspose(WeightNorm.Weight, Convolution.Bias, Convolution.Geometry(x.Shape[0], x.Shape[2], x.Shape[3]), Convolution.Groups);

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("A transposed convolution over time is built on its device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        WeightNorm?.Dispose();
        WeightNorm?.Gain.Dispose();
        Convolution.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"ConvTranspose1d({InChannels} -> {OutChannels}, k{Kernel} s{Stride}{(WeightNorm is null ? "" : ", weight-normed")})";
}
