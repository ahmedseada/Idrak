// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data.Pdf;

/// <summary>
/// A colour given in a colour space (ISO 32000-2, 8.6) as light, by the rules images are decoded with (D4): grey and RGB
/// as they are (8 bits rounded half to even), the calibrated and ICC-based spaces by their component count (profiles not
/// applied), CMYK through 8 bits and the library's JPEG rule, Lab through XYZ (D65) into sRGB, an index through its table
/// and base, Separation and DeviceN through their tint transform into the alternate space.
/// </summary>
internal static class PdfColours
{
    /// <summary>The colour of <paramref name="components"/> (as the space takes them) as 8-bit red, green and blue (grey repeated).</summary>
    public static void ToBytes(PdfFile file, PdfColourSpace space, ReadOnlySpan<double> components, Span<byte> rgb, int depth = 0)
    {
        switch (space.Kind)
        {
            case PdfImageColour.Grey:
                rgb[0] = rgb[1] = rgb[2] = Level(At(components, 0));
                return;
            case PdfImageColour.Rgb:
                (rgb[0], rgb[1], rgb[2]) = (Level(At(components, 0)), Level(At(components, 1)), Level(At(components, 2)));
                return;
            case PdfImageColour.Cmyk:
                PdfPixelConverter.CmykToRgb(Level(At(components, 0)), Level(At(components, 1)), Level(At(components, 2)), Level(At(components, 3)), rgb);
                return;
            case PdfImageColour.Lab:
            {
                var r = space.Range;
                PdfPixelConverter.LabToRgb(Math.Clamp(At(components, 0), 0, 100), Math.Clamp(At(components, 1), r[0], r[1]), Math.Clamp(At(components, 2), r[2], r[3]), space.White, rgb);
                return;
            }

            case PdfImageColour.Indexed when space.Base is { } baseSpace && depth < 4:
            {
                var table = space.Table(file);
                int index = (int)Math.Clamp(Math.Round(At(components, 0), MidpointRounding.ToEven), 0, space.Highest);
                int n = baseSpace.Components;
                Span<double> entry = stackalloc double[4];
                for (int c = 0; c < n && c < 4; c++)
                {
                    int at = index * n + c;
                    double v = at < table.Length ? table[at] : 0;
                    entry[c] = baseSpace.Kind == PdfImageColour.Lab
                        ? c == 0 ? v * 100.0 / 255 : baseSpace.Range[2 * (c - 1)] + v * (baseSpace.Range[2 * c - 1] - baseSpace.Range[2 * (c - 1)]) / 255
                        : v / 255;
                }

                ToBytes(file, baseSpace, entry[..Math.Min(n, 4)], rgb, depth + 1);
                return;
            }

            case PdfImageColour.Tint when space.Base is { } alternate && space.Function is { } function && depth < 4:
            {
                Span<double> outputs = stackalloc double[32];
                int n = Math.Clamp(alternate.Components, 1, 32);
                function.Evaluate(components, outputs[..n]);
                ToBytes(file, alternate, outputs[..n], rgb, depth + 1);
                return;
            }

            default:
                rgb.Clear();
                return;
        }
    }

    /// <summary>
    /// The colour as a <see cref="DocumentColour"/> at opacity <paramref name="alpha"/>: a grey, RGB or CMYK space's
    /// components as given (clamped to 0-1), any other space by its light (<see cref="ToBytes"/>) as RGB.
    /// </summary>
    public static DocumentColour ToColour(PdfFile file, PdfColourSpace space, ReadOnlySpan<double> components, float alpha)
    {
        var colour = space.Kind switch
        {
            PdfImageColour.Grey => DocumentColour.Grey(Unit(At(components, 0))),
            PdfImageColour.Rgb => DocumentColour.Rgb(Unit(At(components, 0)), Unit(At(components, 1)), Unit(At(components, 2))),
            PdfImageColour.Cmyk => DocumentColour.Cmyk(Unit(At(components, 0)), Unit(At(components, 1)), Unit(At(components, 2)), Unit(At(components, 3))),
            _ => Light(file, space, components),
        };
        return alpha >= 1f ? colour : colour.WithAlpha(Math.Clamp(alpha, 0f, 1f));
    }

    /// <summary>The colour a space starts with when it is set (8.6.5.3 and the others): black, index 0, tints of 1, Lab's (0, 0, 0) within its ranges.</summary>
    public static double[] Initial(PdfColourSpace space) => space.Kind switch
    {
        PdfImageColour.Cmyk => [0, 0, 0, 1],
        PdfImageColour.Tint => [.. Enumerable.Repeat(1.0, space.Components)],
        PdfImageColour.Lab => [0, Math.Clamp(0, space.Range[0], space.Range[1]), Math.Clamp(0, space.Range[2], space.Range[3])],
        _ => new double[Math.Max(1, space.Components)],
    };

    private static DocumentColour Light(PdfFile file, PdfColourSpace space, ReadOnlySpan<double> components)
    {
        Span<byte> rgb = stackalloc byte[3];
        try
        {
            ToBytes(file, space, components, rgb);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            rgb.Clear();   // a damaged table: black, as an unknown space is
        }

        return DocumentColour.Rgb(rgb[0] / 255f, rgb[1] / 255f, rgb[2] / 255f);
    }

    private static double At(ReadOnlySpan<double> components, int i) => i < components.Length && double.IsFinite(components[i]) ? components[i] : 0;

    private static float Unit(double v) => (float)Math.Clamp(v, 0, 1);

    private static byte Level(double v) => (byte)Math.Round(Math.Clamp(v, 0, 1) * 255, MidpointRounding.ToEven);
}
