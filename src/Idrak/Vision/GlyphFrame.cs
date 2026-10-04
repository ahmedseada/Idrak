// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Data;

namespace Idrak.Vision;

/// <summary>
/// Frames a character for a classifier the way EMNIST framed its images: the character's ink, centred in a square that
/// keeps its aspect ratio, <c>border</c> pixels from the edge, averaged down (or up) to <c>size</c> x <c>size</c>, at
/// full contrast (its brightest pixel 1). Characters cut from a page and the images of a data set framed alike look
/// alike to a model, whatever size and position they started at.
/// </summary>
/// <remarks>
/// Frames are written into the caller's span, so a page's characters go straight into one batch buffer without an
/// array each. Use <see cref="Reframe"/> as a <see cref="DataLoader"/> transform to frame training images the same way.
/// </remarks>
public static class GlyphFrame
{
    /// <summary>Frames the character in <paramref name="box"/> of a segmented page into <paramref name="destination"/> (size x size values).</summary>
    public static void Extract(PageLayout layout, PixelBox box, Span<float> destination, int size = 28, int border = 1)
    {
        ArgumentNullException.ThrowIfNull(layout);
        Frame(layout.Ink, layout.Width, layout.Threshold, box, destination, size, border);
    }

    /// <summary>
    /// Frames a data set's character image (<paramref name="rows"/> x <paramref name="columns"/>, ink high) into
    /// <paramref name="destination"/>: cropped to the pixels above <paramref name="threshold"/>, then framed as
    /// <see cref="Extract"/> frames characters from a page. An image without ink gives zeros.
    /// </summary>
    public static void Fit(ReadOnlySpan<float> image, int rows, int columns, Span<float> destination, int size = 28, int border = 1, float threshold = 0.25f)
    {
        if (image.Length != rows * columns)
        {
            throw new ArgumentException($"The image holds {image.Length} values, not {rows} x {columns}.", nameof(image));
        }

        int left = columns, right = -1, top = rows, bottom = -1;
        for (int y = 0; y < rows; y++)
        {
            for (int x = 0; x < columns; x++)
            {
                if (image[y * columns + x] > threshold)
                {
                    (left, right) = (Math.Min(left, x), Math.Max(right, x));
                    (top, bottom) = (Math.Min(top, y), Math.Max(bottom, y));
                }
            }
        }

        if (right < 0)
        {
            destination[..(size * size)].Clear();
            return;
        }

        Frame(image, columns, threshold, new PixelBox(left, top, right - left + 1, bottom - top + 1), destination, size, border);
    }

    /// <summary>A <see cref="DataLoader"/> transform that frames each image as <see cref="Fit"/> does, in place (the image stays size x size).</summary>
    public static ISampleTransform Reframe(int border = 1, float threshold = 0.25f) => new ReframeTransform(border, threshold);

    // The ink of `box` (values at or below the threshold read as paper), centred in a square frame, averaged down (or up).
    private static void Frame(ReadOnlySpan<float> ink, int stride, float threshold, PixelBox box, Span<float> destination, int size, int border)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 1);
        if (border < 0 || 2 * border >= size)
        {
            throw new ArgumentOutOfRangeException(nameof(border), $"The border must leave room in a {size}-pixel frame.");
        }

        if (destination.Length < size * size)
        {
            throw new ArgumentException($"The destination holds {destination.Length} values, fewer than {size} x {size}.", nameof(destination));
        }

        var output = destination[..(size * size)];
        // The frame is fractional, so a character is centred exactly whatever its size: its larger side spans
        // size - 2 * border output pixels, and each output pixel is the mean over the source pixels it covers.
        double scale = Math.Max(box.Width, box.Height) / (double)(size - 2 * border);   // source pixels per output pixel
        double offsetX = (size * scale - box.Width) / 2, offsetY = (size * scale - box.Height) / 2;
        double area = scale * scale;
        float max = 0;
        for (int oy = 0; oy < size; oy++)
        {
            double y0 = oy * scale - offsetY, y1 = y0 + scale;   // in the box's rows
            int gyStart = Math.Max((int)Math.Floor(y0), 0), gyEnd = Math.Min((int)Math.Ceiling(y1), box.Height);
            for (int ox = 0; ox < size; ox++)
            {
                double x0 = ox * scale - offsetX, x1 = x0 + scale, sum = 0;
                int gxStart = Math.Max((int)Math.Floor(x0), 0), gxEnd = Math.Min((int)Math.Ceiling(x1), box.Width);
                for (int gy = gyStart; gy < gyEnd; gy++)
                {
                    double wy = Math.Min(y1, gy + 1) - Math.Max(y0, gy);
                    int row = (box.Y + gy) * stride + box.X;
                    for (int gx = gxStart; gx < gxEnd; gx++)
                    {
                        float v = ink[row + gx];
                        if (v > threshold)
                        {
                            sum += wy * (Math.Min(x1, gx + 1) - Math.Max(x0, gx)) * v;
                        }
                    }
                }

                float value = (float)(sum / area);
                output[oy * size + ox] = value;
                max = Math.Max(max, value);
            }
        }

        if (max > 0)
        {
            float inverse = 1f / max;
            for (int i = 0; i < output.Length; i++)
            {
                output[i] *= inverse;
            }
        }
    }

    private sealed class ReframeTransform(int border, float threshold) : ISampleTransform
    {
        public void Apply(Span<float> features, Span<float> targets, IReadOnlyList<int> featureShape, Random random)
        {
            var (planes, h, w) = ImageShape.Of(featureShape, features.Length);
            if (planes != 1 || h != w)
            {
                throw new ArgumentException($"GlyphFrame.Reframe takes square single-channel images, not {string.Join(" x ", featureShape)}.");
            }

            var copy = ArrayPool<float>.Shared.Rent(h * w);
            try
            {
                features[..(h * w)].CopyTo(copy);
                Fit(copy.AsSpan(0, h * w), h, w, features, h, border, threshold);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(copy);
            }
        }
    }
}
