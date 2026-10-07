// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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

        int c = Channels, h = Height, w = Width;
        var pixels = Pixels;
        var rows = Taps(h, height);
        var columns = Taps(w, width);
        for (int oc = 0; oc < channels; oc++)
        {
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    float value = 0;
                    foreach (var (yy, wy) in rows[y])
                    {
                        float row = 0;
                        foreach (var (xx, wx) in columns[x])
                        {
                            row += Sample(yy, xx) * wx;
                        }

                        value += row * wy;
                    }

                    destination[oc * height * width + y * width + x] = value;

                    float Sample(int yy, int xx)
                    {
                        if (channels == c || c == 1)
                        {
                            return pixels[(c == 1 ? 0 : oc) * h * w + yy * w + xx];
                        }

                        float sum = 0;                                                  // colour to grey: the mean of the channels
                        for (int ic = 0; ic < c; ic++)
                        {
                            sum += pixels[ic * h * w + yy * w + xx];
                        }

                        return sum / c;
                    }
                }
            }
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
