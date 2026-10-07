// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Data;
using Idrak.Vision.Abstractions;

namespace Idrak.Vision;

/// <summary>
/// Frames one object for a classifier: its foreground, centred in a square that keeps its aspect ratio, <c>border</c>
/// pixels from the edge, averaged down (or up) to <c>size</c> x <c>size</c>, at full contrast (its brightest pixel 1).
/// This is how EMNIST framed its characters. Objects found in a large image and the images of a data set, framed alike,
/// look alike to a model whatever size and position they started at: characters, digits, symbols, sketches, cells.
/// </summary>
/// <remarks>
/// Frames are written into the caller's span, so many objects go straight into one batch buffer without an array each
/// (<see cref="RegionClassifier"/> does that). Use <see cref="Reframe"/> as a <see cref="DataLoader"/> transform to frame
/// training images the same way.
/// </remarks>
public static class ContentFrame
{
    /// <summary>Frames the object in <paramref name="box"/> of an image's foreground into <paramref name="destination"/> (size x size values).</summary>
    public static void Extract(ForegroundImage image, PixelBox box, Span<float> destination, int size = 28, int border = 1)
    {
        ArgumentNullException.ThrowIfNull(image);
        if (box.X < 0 || box.Y < 0 || box.Right > image.Width || box.Bottom > image.Height || box.Width <= 0 || box.Height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(box), $"{box} is not inside the {image.Width} x {image.Height} image.");
        }

        Frame(image.Values, image.Width, image.Threshold, box, destination, size, border);
    }

    /// <summary>
    /// Frames a data set's image (<paramref name="rows"/> x <paramref name="columns"/>, foreground high) into
    /// <paramref name="destination"/>: cropped to the pixels above <paramref name="threshold"/>, then framed as
    /// <see cref="Extract"/> frames objects of a larger image. An image without foreground gives zeros.
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

    // The foreground of `box` (values at or below the threshold read as background), centred in a square frame, averaged down (or up).
    private static void Frame(ReadOnlySpan<float> values, int stride, float threshold, PixelBox box, Span<float> destination, int size, int border)
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
        // The frame is fractional, so an object is centred exactly whatever its size: its larger side spans
        // size - 2 * border output pixels. Shrinking, each output pixel is the mean over the source pixels it covers.
        double scale = Math.Max(box.Width, box.Height) / (double)(size - 2 * border);   // source pixels per output pixel
        double offsetX = (size * scale - box.Width) / 2, offsetY = (size * scale - box.Height) / 2;
        double area = scale * scale;
        float max = 0;
        if (scale < 1)
        {
            // Enlarging: the output samples the frame bilinearly with its corners aligned (as ImageData.Resize does),
            // so a small object grows smooth, as a data set's enlarged images do, not in blocks of repeated pixels.
            // Outside the box reads as 0.
            double step = (size * scale - 1) / (size - 1);   // source pixels between neighbouring output pixels
            for (int oy = 0; oy < size; oy++)
            {
                double sy = oy * step - offsetY;
                int y0 = (int)Math.Floor(sy);
                float fy = (float)(sy - y0);
                for (int ox = 0; ox < size; ox++)
                {
                    double sx = ox * step - offsetX;
                    int x0 = (int)Math.Floor(sx);
                    float fx = (float)(sx - x0);
                    float top = (1 - fx) * At(values, stride, threshold, box, x0, y0) + fx * At(values, stride, threshold, box, x0 + 1, y0);
                    float bottom = (1 - fx) * At(values, stride, threshold, box, x0, y0 + 1) + fx * At(values, stride, threshold, box, x0 + 1, y0 + 1);
                    float value = (1 - fy) * top + fy * bottom;
                    output[oy * size + ox] = value;
                    max = Math.Max(max, value);
                }
            }

            Stretch(output, max);
            return;
        }

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
                        float v = values[row + gx];
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

        Stretch(output, max);
    }

    // A pixel of the box (x, y within it): 0 outside the box or at or below the threshold.
    private static float At(ReadOnlySpan<float> values, int stride, float threshold, PixelBox box, int x, int y)
    {
        if (x < 0 || y < 0 || x >= box.Width || y >= box.Height)
        {
            return 0;
        }

        float v = values[(box.Y + y) * stride + box.X + x];
        return v > threshold ? v : 0;
    }

    // Full contrast: the brightest value becomes 1.
    private static void Stretch(Span<float> output, float max)
    {
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
                throw new ArgumentException($"ContentFrame.Reframe takes square single-channel images, not {string.Join(" x ", featureShape)}.");
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
