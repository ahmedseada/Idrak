// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

// Image operations as single device operations: convolution (and its two gradients), average pooling and its gradient,
// and resampling with per-channel normalization. Each device runs them with its own kernels (or the composed or host
// path where it has none); a convolution keeps nothing but its input and weights for the backward pass.
public sealed partial class Tensor
{
    /// <summary>
    /// 2-D convolution of this [N, C, H, W] tensor with <paramref name="weight"/> ([filters, C / groups · KH · KW], or
    /// PyTorch's [filters, C / groups, KH, KW]) and an optional <paramref name="bias"/> [filters], giving [N, filters, OH, OW]
    /// over the windows of <paramref name="geometry"/> (its N, C, H and W are this tensor's); each group of C / groups
    /// channels feeds filters / groups filters (depthwise when groups is C). With an <paramref name="activation"/> the
    /// device applies it to each output in the same pass when nothing is recorded for the gradient (inference), else as a
    /// separate operation.
    /// </summary>
    public Tensor Convolution(Tensor weight, Tensor? bias, in ConvGeometry geometry, int groups = 1, ConvActivation activation = ConvActivation.None)
    {
        ThrowIfDisposed();
        ArgumentNullException.ThrowIfNull(weight);
        weight.ThrowIfDisposed();
        bias?.ThrowIfDisposed();
        CheckSameDevice(this, weight);
        if (bias is not null)
        {
            CheckSameDevice(this, bias);
        }

        var g0 = geometry;
        int filters = weight.Rank > 0 ? weight._shape[0] : 0;
        if (Rank != 4 || _shape[0] != g0.N || _shape[1] != g0.C || _shape[2] != g0.H || _shape[3] != g0.W)
        {
            throw new ArgumentException($"The convolution's geometry reads [{g0.N}, {g0.C}, {g0.H}, {g0.W}]; the input is {FormatShape(_shape)}.");
        }

        if (groups <= 0 || g0.C % groups != 0 || filters <= 0 || filters % groups != 0 || weight.Size != filters * (g0.PatchSize / groups)
            || bias is not null && bias.Size != filters)
        {
            throw new ArgumentException($"A convolution of {g0.C} channels in {groups} groups with {g0.KH}x{g0.KW} windows takes weights [filters, "
                + $"{(groups > 0 ? g0.C / groups : 0) * g0.KH * g0.KW}] with filters a multiple of the groups and a bias [filters]; got {FormatShape(weight._shape)} and "
                + $"{(bias is null ? "no bias" : FormatShape(bias._shape))}.");
        }

        if (g0.OH <= 0 || g0.OW <= 0)
        {
            throw new ArgumentException($"A {g0.KH}x{g0.KW} window (dilation {g0.DH}x{g0.DW}) does not fit a {g0.H}x{g0.W} input with padding {g0.PH}x{g0.PW}.");
        }

        var x = this;
        bool record = Autograd.IsEnabled && (RequiresGrad || weight.RequiresGrad || bias?.RequiresGrad == true);
        if (record && Devices.Backend.ActivationOp(activation) is { } op)
        {
            return Convolution(weight, bias, g0, groups).Unary(op);
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([g0.N, filters, g0.OH, g0.OW], Device);
        Backend.Convolution(Storage, weight.Storage, bias?.Storage, y.Storage, g0, filters, groups, activation);
        if (record)
        {
            Tensor[] inputs = bias is null ? [x, weight] : [x, weight, bias];
            y.Record("conv2d", g =>
            {
                var backend = x.Backend;
                if (x.RequiresGrad)
                {
                    backend.ConvolutionBackwardInput(g.Storage, weight.Storage, x.GradStorage(), g0, filters, groups);
                }

                if (weight.RequiresGrad)
                {
                    backend.ConvolutionBackwardWeight(x.Storage, g.Storage, weight.GradStorage(), g0, filters, groups);
                }

                if (bias?.RequiresGrad == true)
                {
                    backend.GroupReduce(g.Storage, null, bias.GradStorage(), null, g0.N, filters, g0.OH * g0.OW);
                }
            }, inputs);
        }

        return Traced("conv2d", y, start);
    }

    /// <summary>
    /// 2-D average pooling of this [N, C, H, W] tensor over the windows of <paramref name="geometry"/> (no dilation): each
    /// window's sum divided by KH·KW, or with <paramref name="countIncludePad"/> false by the input positions it covers.
    /// </summary>
    public Tensor AvgPool(in ConvGeometry geometry, bool countIncludePad = true)
    {
        ThrowIfDisposed();
        var g0 = geometry;
        if (g0.Dilated)
        {
            throw new ArgumentException("Average pooling takes no dilation.", nameof(geometry));
        }

        if (Rank != 4 || _shape[0] != g0.N || _shape[1] != g0.C || _shape[2] != g0.H || _shape[3] != g0.W)
        {
            throw new ArgumentException($"The pooling's geometry reads [{g0.N}, {g0.C}, {g0.H}, {g0.W}]; the input is {FormatShape(_shape)}.");
        }

        if (g0.OH <= 0 || g0.OW <= 0)
        {
            throw new ArgumentException($"A {g0.KH}x{g0.KW} window does not fit a {g0.H}x{g0.W} input with padding {g0.PH}x{g0.PW}.");
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty([g0.N, g0.C, g0.OH, g0.OW], Device);
        Backend.AvgPool(Storage, y.Storage, g0, countIncludePad);
        if (WillRecord(this))
        {
            var x = this;
            y.Record("avgpool", g => x.Backend.AvgPoolBackward(g.Storage, x.GradStorage(), g0, countIncludePad), x);
        }

        return Traced("avgpool", y, start);
    }

    /// <summary>
    /// Resizes this [N, C, H, W] (or [C, H, W]) float image tensor to <paramref name="height"/> x <paramref name="width"/>
    /// by bilinear interpolation and normalizes each channel, in one operation on the tensor's device:
    /// y = resized · scale / std[c] - mean[c] / std[c], that is (resized · scale - mean[c]) / std[c] in affine form. The
    /// weights are those of PyTorch's <c>interpolate(mode="bilinear", align_corners=False, antialias=...)</c>: with
    /// <paramref name="antialias"/> a shrink averages over the pixels it covers (a triangle filter widened by the shrink
    /// factor, as Pillow), without it each output reads its two nearest pixels per axis. No gradient is recorded
    /// (preprocessing).
    /// </summary>
    /// <param name="height">The output height.</param>
    /// <param name="width">The output width.</param>
    /// <param name="mean">One mean per channel, or one for all.</param>
    /// <param name="std">One standard deviation per channel, or one for all.</param>
    /// <param name="scale">What the resized values are multiplied by first (1 / 255 for values 0 ... 255).</param>
    /// <param name="antialias">Whether a shrink averages over every pixel it covers.</param>
    public Tensor ResizeNormalize(int height, int width, ReadOnlySpan<float> mean, ReadOnlySpan<float> std, float scale = 1f, bool antialias = true)
    {
        ThrowIfDisposed();
        if (Rank is not (3 or 4))
        {
            throw new ArgumentException($"ResizeNormalize takes [N, C, H, W] or [C, H, W] images, not {FormatShape(_shape)}.");
        }

        int channels = _shape[^3], inHeight = _shape[^2], inWidth = _shape[^1], planes = Size / Math.Max(1, inHeight * inWidth);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (mean.Length is not 1 && mean.Length != channels || std.Length is not 1 && std.Length != channels)
        {
            throw new ArgumentException($"{mean.Length} means and {std.Length} standard deviations for {channels} channels: give one per channel, or one for all.");
        }

        var values = new float[2 * channels];
        for (int c = 0; c < channels; c++)
        {
            float m = mean[mean.Length == 1 ? 0 : c], s = std[std.Length == 1 ? 0 : c];
            values[2 * c] = scale / s;
            values[2 * c + 1] = -m / s;
        }

        var (across, xTaps) = inWidth == width ? ([], 0) : BilinearTaps(inWidth, width, antialias);
        var (down, yTaps) = inHeight == height ? ([], 0) : BilinearTaps(inHeight, height, antialias);
        int[] shape = [.. _shape[..^2], height, width];
        long start = Telemetry.Start(TelemetryLevel.Operations);
        var y = Empty(shape, Device);
        using var coefficients = From([.. across, .. down], [Math.Max(1, across.Length + down.Length)], Device);
        using var table = From(values, [values.Length], Device);
        Backend.ResizeNormalize(Storage, coefficients.Storage, table.Storage, y.Storage, planes, channels, inHeight, inWidth, height, width, xTaps, yTaps, bytes: false);
        return Traced("resize_normalize", y, start);
    }

    /// <summary>
    /// The coefficients of a bilinear resize of one axis from <paramref name="inSize"/> to <paramref name="outSize"/> as
    /// <see cref="Backend.ResizeNormalize"/> reads them (per output: first input, tap count, then the float weights of
    /// <c>Taps</c> taps): a triangle filter centred on each output's position, widened by the shrink factor when
    /// <paramref name="antialias"/>, its weights normalized to sum 1 (PyTorch's and Pillow's construction, in double).
    /// </summary>
    public static (float[] Coefficients, int Taps) BilinearTaps(int inSize, int outSize, bool antialias = true)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inSize);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outSize);
        double scale = (double)inSize / outSize, filterScale = antialias ? Math.Max(scale, 1.0) : 1.0, support = filterScale;
        int taps = (int)Math.Ceiling(support) * 2 + 1;
        var result = new float[outSize * (2 + taps)];
        var weights = new double[taps];
        for (int o = 0; o < outSize; o++)
        {
            double center = (o + 0.5) * scale;
            int first = Math.Max((int)(center - support + 0.5), 0), last = Math.Min((int)(center + support + 0.5), inSize);
            int count = Math.Max(0, last - first);
            double total = 0;
            for (int t = 0; t < count; t++)
            {
                double w = Math.Max(0, 1 - Math.Abs((first + t - center + 0.5) / filterScale));
                weights[t] = w;
                total += w;
            }

            int at = o * (2 + taps);
            result[at] = BitConverter.Int32BitsToSingle(first);
            result[at + 1] = BitConverter.Int32BitsToSingle(count);
            for (int t = 0; t < count; t++)
            {
                result[at + 2 + t] = total > 0 ? (float)(weights[t] / total) : 0f;
            }
        }

        return (result, taps);
    }
}
