// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;

namespace Idrak.Abstraction.Data;

/// <summary>
/// Decoded pixels: [Channels, Height, Width] values in [0, 1], grey (1 channel) or red, green and blue (3 channels).
/// </summary>
public sealed class ImageData
{
    /// <summary>Wraps decoded pixels (used directly, not copied).</summary>
    public ImageData(float[] pixels, int channels, int height, int width)
    {
        ArgumentNullException.ThrowIfNull(pixels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (pixels.Length != (long)channels * height * width)
        {
            throw new ArgumentException($"{pixels.Length} values are not {channels} x {height} x {width}.", nameof(pixels));
        }

        Pixels = pixels;
        Channels = channels;
        Height = height;
        Width = width;
    }

    /// <summary>[Channels, Height, Width] values in [0, 1].</summary>
    public float[] Pixels { get; }

    /// <summary>1 (grey) or 3 (red, green, blue).</summary>
    public int Channels { get; }

    /// <summary>Rows.</summary>
    public int Height { get; }

    /// <summary>Pixels per row.</summary>
    public int Width { get; }

    /// <summary>
    /// The image as <paramref name="channels"/> x <paramref name="height"/> x <paramref name="width"/> values written
    /// to <paramref name="destination"/>: grey repeated to colour, colour averaged to grey, and the size changed per
    /// axis: an axis that grows (or keeps its size) by bilinear sampling with the corners aligned, an axis that shrinks
    /// by the mean of the pixels each output pixel covers, so that no thin line falls between the samples.
    /// </summary>
    public void Resize(int channels, int height, int width, Span<float> destination)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(channels);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(height);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(width);
        if (destination.Length != channels * height * width)
        {
            throw new ArgumentException($"The destination holds {destination.Length} values, not {channels} x {height} x {width}.", nameof(destination));
        }

        // In two passes with the same arithmetic as the two-axis sum: each source row along x first (its value at every
        // output column), then each output row as the weighted sum of its source rows, in the same order. Output channels
        // that read the same source (grey repeated, colour averaged) are computed once and copied.
        int c = Channels, h = Height, w = Width, plane = height * width;
        var pixels = Pixels;
        var rows = Taps(h, height);
        var columns = Taps(w, width);
        bool shared = c == 1 || channels != c;
        float[] across = ArrayPool<float>.Shared.Rent(h * width);
        try
        {
            for (int oc = 0; oc < channels; oc++)
            {
                var output = destination.Slice(oc * plane, plane);
                if (shared && oc > 0)
                {
                    destination[..plane].CopyTo(output);
                    continue;
                }

                if (shared && c > 1)
                {
                    AcrossGrey(pixels, c, h, w, columns, across.AsSpan(0, h * width));
                }
                else
                {
                    Across(pixels.AsSpan((c == 1 ? 0 : oc) * h * w, h * w), w, columns, across.AsSpan(0, h * width));
                }

                for (int y = 0; y < height; y++)
                {
                    var line = output.Slice(y * width, width);
                    line.Clear();
                    foreach (var (yy, wy) in rows[y])
                    {
                        AddScaled(line, across.AsSpan(yy * width, width), wy);
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(across);
        }
    }

    // Each row of one source plane resampled along x: rows of `columns.Length` values.
    private static void Across(ReadOnlySpan<float> source, int w, (int Index, float Weight)[][] columns, Span<float> across)
    {
        int width = columns.Length;
        for (int yy = 0; yy < source.Length / w; yy++)
        {
            var pixelRow = source.Slice(yy * w, w);
            var line = across.Slice(yy * width, width);
            for (int x = 0; x < line.Length; x++)
            {
                float row = 0;
                foreach (var (xx, wx) in columns[x])
                {
                    row += pixelRow[xx] * wx;
                }

                line[x] = row;
            }
        }
    }

    // As Across, on the grey of a colour image: each sample the mean of its channels.
    private static void AcrossGrey(float[] pixels, int c, int h, int w, (int Index, float Weight)[][] columns, Span<float> across)
    {
        int width = columns.Length, size = h * w;
        for (int yy = 0; yy < h; yy++)
        {
            var line = across.Slice(yy * width, width);
            for (int x = 0; x < line.Length; x++)
            {
                float row = 0;
                foreach (var (xx, wx) in columns[x])
                {
                    float sum = 0;
                    for (int ic = 0, at = yy * w + xx; ic < c; ic++, at += size)
                    {
                        sum += pixels[at];
                    }

                    row += sum / c * wx;
                }

                line[x] = row;
            }
        }
    }

    // target += source · scale, element by element (vectors and a scalar tail compute the same values).
    private static void AddScaled(Span<float> target, ReadOnlySpan<float> source, float scale)
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && target.Length >= Vector<float>.Count)
        {
            var s = new Vector<float>(scale);
            for (; i <= target.Length - Vector<float>.Count; i += Vector<float>.Count)
            {
                (new Vector<float>(target[i..]) + new Vector<float>(source[i..]) * s).CopyTo(target[i..]);
            }
        }

        for (; i < target.Length; i++)
        {
            target[i] += source[i] * scale;
        }
    }

    // For each output position along an axis of `from` pixels resized to `to`, the source pixels it reads and their
    // weights (summing to 1). Growing or keeping the size: the two neighbours of bilinear sampling with the corners
    // aligned. Shrinking: every source pixel the output pixel's span [i, i + 1) * from / to covers, weighted by how much.
    private static (int Index, float Weight)[][] Taps(int from, int to)
    {
        var taps = new (int Index, float Weight)[to][];
        for (int i = 0; i < to; i++)
        {
            if (to >= from)
            {
                float s = to == 1 ? 0 : i * (from - 1) / (float)(to - 1);
                int i0 = (int)s, i1 = Math.Min(i0 + 1, from - 1);
                float f = s - i0;
                taps[i] = [(i0, 1 - f), (i1, f)];
                continue;
            }

            double start = i * (double)from / to, end = (i + 1) * (double)from / to;
            var covered = new List<(int Index, float Weight)>();
            for (int j = (int)start; j < Math.Min(from, (int)Math.Ceiling(end)); j++)
            {
                double overlap = Math.Min(end, j + 1) - Math.Max(start, j);
                if (overlap > 0)
                {
                    covered.Add((j, (float)(overlap / (end - start))));
                }
            }

            taps[i] = [.. covered];
        }

        return taps;
    }

    /// <summary>The image resized as <see cref="Resize(int, int, int, Span{float})"/> says, in a new array.</summary>
    public float[] Resize(int channels, int height, int width)
    {
        var result = new float[channels * height * width];
        Resize(channels, height, width, result);
        return result;
    }
}
