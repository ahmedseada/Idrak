// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Samples.Ocr;

/// <summary>Draws characters and text lines as grayscale images (ink = 1, paper = 0).</summary>
internal static class Renderer
{
    /// <summary>Side of the square image the classifier sees.</summary>
    public const int Cell = 20;

    /// <summary>Glyph height, in pixels, that segmented characters are normalized to.</summary>
    public const float NormalHeight = Font.Height * 2.2f;

    /// <summary>
    /// Renders one character into a Cell×Cell image with random size, slant, position, stroke weight,
    /// brightness and noise, so the classifier learns to tolerate real-world variation.
    /// </summary>
    public static void RenderCharacter(int index, Span<float> image, Random random)
    {
        float scaleX = 2.0f + random.NextSingle() * 0.6f;
        float scaleY = 2.0f + random.NextSingle() * 0.4f;
        float shear = (random.NextSingle() - 0.5f) * 0.5f;
        float weight = 0.4f + random.NextSingle() * 0.3f;
        float offsetX = (Cell - Font.Width * scaleX) / 2 + (random.NextSingle() - 0.5f) * 3f;
        float offsetY = (Cell - Font.Height * scaleY) / 2 + (random.NextSingle() - 0.5f) * 3f;
        float brightness = 0.65f + random.NextSingle() * 0.35f;
        Draw(index, image, Cell, Cell, offsetX, offsetY, scaleX, scaleY, shear, weight, brightness);
        for (int i = 0; i < image.Length; i++)
        {
            image[i] = Math.Clamp(image[i] + (random.NextSingle() - 0.5f) * 0.25f, 0f, 1f);
        }
    }

    /// <summary>Renders a text line (characters from <see cref="Font.Characters"/> and spaces) at a fixed size.</summary>
    public static (float[] Pixels, int Width) RenderLine(string text, Random random, float noise = 0.15f)
    {
        const float Scale = 2.2f;
        const int Gap = 3, SpaceWidth = 9, Margin = 4;
        int glyphWidth = (int)MathF.Ceiling(Font.Width * Scale);
        int width = Margin * 2 + text.Sum(c => c == ' ' ? SpaceWidth : glyphWidth + Gap);
        var pixels = new float[width * Cell];
        float x = Margin;
        foreach (char c in text.ToUpperInvariant())
        {
            int index = Font.Characters.IndexOf(c);
            if (index < 0)
            {
                x += SpaceWidth;
                continue;
            }

            float y = (Cell - Font.Height * Scale) / 2 + (random.NextSingle() - 0.5f);
            Draw(index, pixels, width, Cell, x, y, Scale, Scale, 0f, 0.55f, 0.9f);
            x += glyphWidth + Gap;
        }

        for (int i = 0; i < pixels.Length; i++)
        {
            pixels[i] = Math.Clamp(pixels[i] + (random.NextSingle() - 0.5f) * noise * 2, 0f, 1f);
        }

        return (pixels, width);
    }

    /// <summary>
    /// Anti-aliased rendering by 3×3 supersampling. A sample is inked when it lies inside a square of
    /// half-size <paramref name="weight"/> (in glyph cells) around an inked cell's center: 0.5 draws the
    /// exact glyph, less gives thin dot-matrix strokes, more gives bold ones.
    /// </summary>
    private static void Draw(int index, Span<float> image, int width, int height, float offsetX, float offsetY,
        float scaleX, float scaleY, float shear, float weight, float brightness)
    {
        for (int py = 0; py < height; py++)
        {
            for (int px = 0; px < width; px++)
            {
                int hits = 0;
                for (int sy = 0; sy < 3; sy++)
                {
                    for (int sx = 0; sx < 3; sx++)
                    {
                        float v = (py + (sy + 0.5f) / 3 - offsetY) / scaleY;
                        float u = (px + (sx + 0.5f) / 3 - offsetX) / scaleX - shear * (v - Font.Height / 2f);
                        if (Inked(index, u, v, weight))
                        {
                            hits++;
                        }
                    }
                }

                if (hits > 0)
                {
                    int i = py * width + px;
                    image[i] = MathF.Max(image[i], brightness * hits / 9f);
                }
            }
        }
    }

    private static bool Inked(int index, float u, float v, float weight)
    {
        int cu = (int)MathF.Floor(u), cv = (int)MathF.Floor(v);
        for (int dv = -1; dv <= 1; dv++)
        {
            for (int du = -1; du <= 1; du++)
            {
                int gu = cu + du, gv = cv + dv;
                if (Font.Ink(index, gu, gv) && MathF.Abs(u - (gu + 0.5f)) <= weight && MathF.Abs(v - (gv + 0.5f)) <= weight)
                {
                    return true;
                }
            }
        }

        return false;
    }
}
