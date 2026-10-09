// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;

namespace Idrak.Layers;

/// <summary>
/// Resizes the height and width of [N, C, H, W] (or any [..., H, W]) as PyTorch's <c>nn.Upsample</c> /
/// <c>F.interpolate</c>: by a scale factor (output floor(H · factor)) or to a fixed size (<see cref="ToSize"/>), with the
/// nearest input position or bilinear weights, corners aligned or not. A decoder's or a feature pyramid's upsampling step.
/// </summary>
public sealed class Upsample : Module
{
    /// <summary>Scales height and width by the same factor.</summary>
    /// <param name="scaleFactor">Output positions per input position (2 doubles the size; below 1 shrinks it).</param>
    /// <param name="mode">Nearest or bilinear.</param>
    /// <param name="alignCorners">Bilinear only: the corner positions of input and output coincide.</param>
    public Upsample(float scaleFactor, InterpolationMode mode = InterpolationMode.Nearest, bool alignCorners = false)
        : this((scaleFactor, scaleFactor), mode, alignCorners)
    {
    }

    /// <summary>Scales height and width by their own factors.</summary>
    /// <param name="scaleFactor">Output positions per input position, (height, width).</param>
    /// <param name="mode">Nearest or bilinear.</param>
    /// <param name="alignCorners">Bilinear only: the corner positions of input and output coincide.</param>
    public Upsample((float Height, float Width) scaleFactor, InterpolationMode mode = InterpolationMode.Nearest, bool alignCorners = false)
        : this(scaleFactor, null, mode, alignCorners)
    {
    }

    private Upsample((float Height, float Width)? scaleFactor, (int Height, int Width)? size, InterpolationMode mode, bool alignCorners)
    {
        if (scaleFactor is { } f && !(f.Height > 0 && f.Width > 0 && float.IsFinite(f.Height) && float.IsFinite(f.Width)))
        {
            throw new ArgumentOutOfRangeException(nameof(scaleFactor), f, "Upsample scale factors are positive.");
        }

        if (size is { } s && (s.Height <= 0 || s.Width <= 0))
        {
            throw new ArgumentOutOfRangeException(nameof(size), s, "An Upsample size is positive.");
        }

        if (alignCorners && mode == InterpolationMode.Nearest)
        {
            throw new ArgumentException("alignCorners applies to bilinear upsampling, not nearest (as PyTorch).", nameof(alignCorners));
        }

        if (!Enum.IsDefined(mode))
        {
            throw new ArgumentOutOfRangeException(nameof(mode), mode, "Upsampling is nearest or bilinear.");
        }

        ScaleFactor = scaleFactor;
        Size = size;
        Mode = mode;
        AlignCorners = alignCorners;
    }

    /// <summary>Resizes to a fixed (height, width) whatever the input's size.</summary>
    /// <param name="size">The output (height, width).</param>
    /// <param name="mode">Nearest or bilinear.</param>
    /// <param name="alignCorners">Bilinear only: the corner positions of input and output coincide.</param>
    public static Upsample ToSize((int Height, int Width) size, InterpolationMode mode = InterpolationMode.Nearest, bool alignCorners = false) =>
        new(null, size, mode, alignCorners);

    /// <summary>The scale factors, or null when the layer resizes to a fixed <see cref="Size"/>.</summary>
    public (float Height, float Width)? ScaleFactor { get; }

    /// <summary>The fixed output size, or null when the layer scales by <see cref="ScaleFactor"/>.</summary>
    public (int Height, int Width)? Size { get; }

    /// <summary>Nearest or bilinear.</summary>
    public InterpolationMode Mode { get; }

    /// <summary>Whether the corner positions of input and output coincide (bilinear).</summary>
    public bool AlignCorners { get; }

    /// <summary>
    /// The output height and width for an input of <paramref name="height"/> x <paramref name="width"/>: the fixed size, or
    /// floor(size · factor) with the factor as written (its shortest decimal form, as Python reads it).
    /// </summary>
    public (int Height, int Width) OutputSize(int height, int width) => Size ?? (Scaled(height, ScaleFactor!.Value.Height), Scaled(width, ScaleFactor!.Value.Width));

    private static int Scaled(int size, float factor) => (int)Math.Floor(size * Written(factor));

    // A factor as written (1.7f as 1.7, not 1.70000005), as Python passes it to F.interpolate.
    private static double Written(float factor) => (double)(decimal)factor;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank < 3)
        {
            throw new ArgumentException($"Upsample expects [N, C, H, W] (or [..., H, W] with a batch), got {Tensor.FormatShape(input.Shape)}.");
        }

        var size = OutputSize(input.Shape[^2], input.Shape[^1]);
        if (size.Height <= 0 || size.Width <= 0)
        {
            throw new ArgumentException($"{this} gives an empty output for a {input.Shape[^2]}x{input.Shape[^1]} input.");
        }

        // As F.interpolate: a scale factor (not the size ratio) sets where an output position reads.
        return input.Interpolate(size, Mode, AlignCorners, ScaleFactor is { } f ? (Written(f.Height), Written(f.Width)) : null);
    }

    /// <inheritdoc />
    public override string ToString()
    {
        string how = Size is { } s ? $"to {s.Height}x{s.Width}" : ScaleFactor!.Value.Height == ScaleFactor.Value.Width ? $"x{F(ScaleFactor.Value.Height)}"
            : $"x{F(ScaleFactor.Value.Height)}x{F(ScaleFactor.Value.Width)}";
        return $"Upsample({how}, {(Mode == InterpolationMode.Nearest ? "nearest" : "bilinear")}{(AlignCorners ? ", corners aligned" : "")})";

        static string F(float v) => v.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
    }
}

/// <summary>
/// Adaptive average pooling over [N, C, H, W] (or [..., H, W]) to a fixed output size whatever the input's, as PyTorch's
/// <c>AdaptiveAvgPool2d</c>: output row o averages input rows floor(o · H / OH) to ceil((o + 1) · H / OH), exclusive (so
/// windows may differ in size and overlap); columns likewise. Lets a classifier head take images of any size.
/// </summary>
public sealed class AdaptiveAvgPool2d : Module
{
    /// <summary>Pools to a square <paramref name="outputSize"/> x <paramref name="outputSize"/>.</summary>
    public AdaptiveAvgPool2d(int outputSize) : this((outputSize, outputSize))
    {
    }

    /// <summary>Pools to <paramref name="outputSize"/> (height, width).</summary>
    public AdaptiveAvgPool2d((int Height, int Width) outputSize) => (OutputHeight, OutputWidth) = AdaptivePooling.Check("AdaptiveAvgPool2d", outputSize);

    /// <summary>Output height.</summary>
    public int OutputHeight { get; }

    /// <summary>Output width.</summary>
    public int OutputWidth { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => AdaptivePooling.Pool(input, (OutputHeight, OutputWidth), max: false);

    /// <inheritdoc />
    public override string ToString() => $"AdaptiveAvgPool2d({OutputHeight}x{OutputWidth})";
}

/// <summary>
/// Adaptive max pooling over [N, C, H, W] (or [..., H, W]) to a fixed output size, over the windows of
/// <see cref="AdaptiveAvgPool2d"/> (PyTorch's <c>AdaptiveMaxPool2d</c>).
/// </summary>
public sealed class AdaptiveMaxPool2d : Module
{
    /// <summary>Pools to a square <paramref name="outputSize"/> x <paramref name="outputSize"/>.</summary>
    public AdaptiveMaxPool2d(int outputSize) : this((outputSize, outputSize))
    {
    }

    /// <summary>Pools to <paramref name="outputSize"/> (height, width).</summary>
    public AdaptiveMaxPool2d((int Height, int Width) outputSize) => (OutputHeight, OutputWidth) = AdaptivePooling.Check("AdaptiveMaxPool2d", outputSize);

    /// <summary>Output height.</summary>
    public int OutputHeight { get; }

    /// <summary>Output width.</summary>
    public int OutputWidth { get; }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => AdaptivePooling.Pool(input, (OutputHeight, OutputWidth), max: true);

    /// <inheritdoc />
    public override string ToString() => $"AdaptiveMaxPool2d({OutputHeight}x{OutputWidth})";
}

internal static class AdaptivePooling
{
    public static (int, int) Check(string layer, (int Height, int Width) size) => size.Height > 0 && size.Width > 0
        ? size
        : throw new ArgumentOutOfRangeException(nameof(size), size, $"{layer} needs a positive output size.");

    // When the output divides the input the windows are ordinary ones (k = s = input / output); a device with no adaptive
    // pooling kernel of its own (it would run on the host) pools them with its own window kernels instead. Everything else,
    // and every device that has the kernel (the CPU among them), runs the adaptive operation.
    public static Tensor Pool(Tensor input, (int Height, int Width) size, bool max)
    {
        if (input.Rank < 3)
        {
            throw new ArgumentException($"Adaptive pooling expects [N, C, H, W] (or [..., H, W] with a batch), got {Tensor.FormatShape(input.Shape)}.");
        }

        int h = input.Shape[^2], w = input.Shape[^1];
        if (h % size.Height == 0 && w % size.Width == 0 && !RunsItself(input.Backend, max ? Ops.AdaptiveMaxPool : Ops.AdaptiveAvgPool))
        {
            int planes = input.Size / (h * w), kh = h / size.Height, kw = w / size.Width;
            var g = new ConvGeometry(planes, 1, h, w, kh, kw, kh, kw, 0, 0);
            var pooled = max ? input.Reshape(planes, 1, h, w).MaxPool(g) : input.Reshape(planes, 1, h, w).Im2Col(g).Mean(1);
            return pooled.Reshape([.. input.Shape[..^2], size.Height, size.Width]);
        }

        return max ? input.AdaptiveMaxPool(size) : input.AdaptiveAvgPool(size);
    }

    private static bool RunsItself(Backend backend, Operation operation) =>
        Kernels.Chain(backend)[operation.Index].Source is KernelSource.Registered or KernelSource.Device;
}
