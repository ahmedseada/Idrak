// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using Idrak.Data;
using Idrak.Data.Abstractions;

namespace Idrak.Samples.Ocr;

/// <summary>
/// Reads images for the reader (any format the library decodes: PNG, BMP, PGM, PPM) and writes PGM (portable graymap),
/// a trivial format every image editor can export (GIMP, Photoshop, IrfanView, ImageMagick: <c>magick input.jpg output.png</c>).
/// </summary>
internal static class Pgm
{
    /// <summary>Loads an image as ink intensities in [0, 1] (colour averaged to grey); light backgrounds are inverted.</summary>
    public static (float[] Pixels, int Width, int Height) ReadImage(string path)
    {
        var image = ImageCodecs.Decode(path);
        var pixels = image.Resize(1, image.Height, image.Width);

        // Ink should be bright: invert dark-on-light pages.
        if (pixels.Average() > 0.5f)
        {
            for (int i = 0; i < pixels.Length; i++)
            {
                pixels[i] = 1f - pixels[i];
            }
        }

        return (pixels, image.Width, image.Height);
    }

    /// <summary>Writes the image as a binary PGM: dark ink on a white page, or bright ink as given (for training images).</summary>
    public static void Write(string path, float[] pixels, int width, int height, bool inkIsDark = true)
    {
        using var stream = File.Create(path);
        stream.Write(Encoding.ASCII.GetBytes(string.Create(CultureInfo.InvariantCulture, $"P5\n{width} {height}\n255\n")));
        var bytes = new byte[pixels.Length];                                              // one write, not a call a pixel
        for (int i = 0; i < pixels.Length; i++)
        {
            int value = Math.Clamp((int)MathF.Round(pixels[i] * 255), 0, 255);
            bytes[i] = (byte)(inkIsDark ? 255 - value : value);
        }

        stream.Write(bytes);
    }
}
