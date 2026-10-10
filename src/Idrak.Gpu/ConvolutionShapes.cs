// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu;

/// <summary>
/// What the GPU backends share about their convolution choices: the shape of a convolution as the numbers of a tuning key,
/// and the formula for the weight gradient's splits of the sum. Which path runs (the composed patches and products, an
/// implicit product at a tile and width, the depthwise kernels) is measured per shape on the device in use.
/// </summary>
internal static class ConvolutionShapes
{
    /// <summary>The forward pass, the input gradient and the weight gradient (a key's pass).</summary>
    public const int Forward = 0, Input = 1, Weight = 2;

    /// <summary>
    /// The shape as six numbers (a, b, c, d, e, f) for a tuning key: images, channels, filters, height and width, window,
    /// stride and dilation, padding (above, left, below, right); false when a size passes its field (the formula is used then, not measured).
    /// The images, height and width follow the data and are keyed by their size classes (<see cref="TuneSizes"/>): batches
    /// of lines or images of every size share the choices of their class instead of measuring nearly every batch.
    /// </summary>
    public static bool Key(in ConvGeometry g, int filters, int groups, out (int A, int B, int C, int D, int E, int F) key)
    {
        key = default;
        int images = TuneSizes.Class(g.N), height = TuneSizes.Class(g.H), width = TuneSizes.Class(g.W);
        if (height >= 1 << 16 || width >= 1 << 16 || g.KH >= 1 << 8 || g.KW >= 1 << 8 || g.SH >= 1 << 4 || g.SW >= 1 << 4 || g.DH >= 1 << 4 || g.DW >= 1 << 4
            || g.PH >= 1 << 8 || g.PW >= 1 << 8 || g.PadBottom >= 1 << 8 || g.PadRight >= 1 << 7 || groups >= 1 << 22)
        {
            return false;
        }

        key = (images, g.C, filters, height << 16 | width, g.KH << 24 | g.KW << 16 | g.SH << 12 | g.SW << 8 | g.DH << 4 | g.DW,
            g.PH << 23 | g.PW << 15 | g.PadBottom << 7 | g.PadRight);
        return true;
    }

    /// <summary>A key's variant: the pass, the matrix precision (MixedPrecision), whether an activation is applied, the groups.</summary>
    public static int Variant(int pass, ConvActivation activation, int groups) =>
        pass | (int)MixedPrecision.Current << 2 | (activation == ConvActivation.None ? 0 : 1 << 4) | groups << 5;

    /// <summary>Whether every group reads one input channel (depthwise; any number of filters a channel).</summary>
    public static bool Depthwise(in ConvGeometry g, int groups) => groups > 1 && g.C == groups;

    /// <summary>The split counts of the weight gradient's sum over positions tried: 1, 4, 16 and 64, each at least 256 positions a split.</summary>
    public static int[] SplitCounts(long positions)
    {
        var counts = new List<int>(SplitChoices.Length);
        foreach (int s in SplitChoices)
        {
            if (SplitTried(s, positions))
            {
                counts.Add(s);
            }
        }

        return [.. counts];
    }

    /// <summary>The split counts <see cref="SplitCounts"/> picks from, in its order (the choices run per pass read them without an array).</summary>
    public static ReadOnlySpan<int> SplitChoices => [1, 4, 16, 64];

    /// <summary>Whether <see cref="SplitCounts"/> tries <paramref name="splits"/> for <paramref name="positions"/>: at least 256 positions a split.</summary>
    public static bool SplitTried(int splits, long positions) => splits == 1 || positions / splits >= 256;

    /// <summary>
    /// The splits of the weight gradient while nothing is measured: the most of <see cref="SplitCounts"/> that leave each
    /// split at least 4,096 positions (the product's blocks alone are few where filters and patches are small).
    /// </summary>
    public static int FormulaSplits(long positions)
    {
        int most = 1;
        foreach (int s in SplitChoices)
        {
            if (SplitTried(s, positions) && (s == 1 || positions / s >= 4096))
            {
                most = Math.Max(most, s);
            }
        }

        return most;
    }
}
