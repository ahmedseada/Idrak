// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Data.Pdf;

/// <summary>
/// A shading dictionary or stream (ISO 32000-2, 8.7.4.5) as a <see cref="DocumentShading"/> placed on the page: types 1
/// (function-based, its /Matrix and domain), 2 (axial) and 3 (radial) with their colour from the function (one or several,
/// through the colour space into light); types 4 and 5 (free-form and lattice meshes) as their triangles; types 6 and 7
/// (Coons and tensor patches, edges shared by their flags) cut into a grid of triangles, each corner's colour blended
/// across the patch. Every point is mapped by the matrix given (the CTM and the page's display map) onto the page.
/// </summary>
internal static class PdfShadings
{
    private const int PatchSteps = 16;          // a patch is cut into this many by this many cells of two triangles
    private const int MaxTriangles = 1 << 20;   // a mesh's triangles at most (more is damage or a bomb)

    /// <summary>The shading <paramref name="value"/> holds, placed by <paramref name="ctm"/>.</summary>
    /// <exception cref="InvalidDataException">It is damaged, or of an unknown type (named).</exception>
    public static DocumentShading Read(PdfFile file, PdfValue value, PdfDictionary? resources, PdfMatrix ctm)
    {
        var resolved = file.Resolve(value);
        var stream = resolved.AsStream;
        var dictionary = stream?.Dictionary ?? resolved.AsDictionary ?? throw new InvalidDataException($"a shading that is {resolved}");
        long type = file.Resolve(dictionary["ShadingType"u8]).TryInteger(out long t) ? t : 0;
        var space = PdfColourSpace.Read(file, dictionary["ColorSpace"u8], resources);
        if (space.Problem is { } problem)
        {
            throw new InvalidDataException($"a shading's colour space: {problem}");
        }

        PdfFunction? function = dictionary.Contains("Function"u8) ? PdfFunction.Read(file, dictionary["Function"u8]) : null;
        DocumentColour? background = PdfFunction.Numbers(file, dictionary["Background"u8]) is { Length: > 0 } b
            ? PdfColours.ToColour(file, space, b, 1f)
            : null;
        int n = Math.Clamp(space.Components, 1, 32);
        DocumentColour Colour(ReadOnlySpan<double> input)
        {
            Span<double> components = stackalloc double[32];
            if (function is null)
            {
                return PdfColours.ToColour(file, space, input, 1f);
            }

            function.Evaluate(input, components[..n]);
            return PdfColours.ToColour(file, space, components[..n], 1f);
        }

        var matrix = Document(ctm);
        switch (type)
        {
            case 1:
            {
                double[] domain = PdfFunction.Numbers(file, dictionary["Domain"u8]) is { Length: 4 } d ? d : [0, 1, 0, 1];
                var placed = Matrix(file, dictionary["Matrix"u8]).Then(ctm);
                _ = function ?? throw new InvalidDataException("a function-based shading without its function");
                return new DocumentShading(DocumentShadingKind.Function, (x, y) => Colour([x, y]))
                {
                    Coords = new float[] { (float)domain[0], (float)domain[1], (float)domain[2], (float)domain[3] },
                    Matrix = Document(placed), Background = background,
                };
            }

            case 2 or 3:
            {
                int count = type == 2 ? 4 : 6;
                double[] coords = PdfFunction.Numbers(file, dictionary["Coords"u8]) is { } c && c.Length == count ? c
                    : throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture, $"a type {type} shading without its {count} /Coords"));
                double[] domain = PdfFunction.Numbers(file, dictionary["Domain"u8]) is { Length: 2 } dd ? dd : [0, 1];
                bool[] extend = file.Resolve(dictionary["Extend"u8]).AsArray is { Count: 2 } e ? [file.Resolve(e[0]).AsBoolean, file.Resolve(e[1]).AsBoolean] : [false, false];
                _ = function ?? throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture, $"a type {type} shading without its function"));
                return new DocumentShading(type == 2 ? DocumentShadingKind.Axial : DocumentShadingKind.Radial, (s, _) => Colour([s]))
                {
                    Coords = coords.Select(v => (float)v).ToArray(), DomainStart = (float)domain[0], DomainEnd = (float)domain[1],
                    ExtendStart = extend[0], ExtendEnd = extend[1], Matrix = matrix, Background = background,
                };
            }

            case >= 4 and <= 7 when stream is not null:
            {
                var triangles = Mesh(file, stream, (int)type, function is null ? n : 1, ctm, c => Colour(c));
                return new DocumentShading(DocumentShadingKind.Mesh, null) { Triangles = triangles, Background = background };
            }

            default:
                throw new InvalidDataException(string.Create(CultureInfo.InvariantCulture, $"a shading of type {type} (types 1 to 7; 4 to 7 are streams)"));
        }
    }

    internal static DocumentMatrix Document(PdfMatrix m) => new((float)m.A, (float)m.B, (float)m.C, (float)m.D, (float)m.E, (float)m.F);

    internal static PdfMatrix Matrix(PdfFile file, PdfValue value) =>
        PdfFunction.Numbers(file, value) is { Length: 6 } m ? new PdfMatrix(m[0], m[1], m[2], m[3], m[4], m[5]) : PdfMatrix.Identity;

    // A mesh's triangles on the page: x, y, then red, green and blue, for each corner.
    private static float[] Mesh(PdfFile file, PdfStream stream, int type, int components, PdfMatrix ctm, Func<double[], DocumentColour> colour)
    {
        var d = stream.Dictionary;
        int coordinateBits = Bits(file, d["BitsPerCoordinate"u8]), componentBits = Bits(file, d["BitsPerComponent"u8]);
        int flagBits = type is 4 or 6 or 7 ? Bits(file, d["BitsPerFlag"u8]) : 0;
        double[] decode = PdfFunction.Numbers(file, d["Decode"u8]) ?? [];
        if (coordinateBits == 0 || componentBits == 0 || decode.Length < 4 + 2 * components || (type != 5 && flagBits == 0))
        {
            throw new InvalidDataException("a mesh shading without its bits a coordinate, a component and a flag, or its /Decode");
        }

        int perRow = type == 5 ? (file.Resolve(d["VerticesPerRow"u8]).TryInteger(out long v) && v >= 2 ? (int)Math.Min(v, 1 << 16) : 0) : 0;
        if (type == 5 && perRow < 2)
        {
            throw new InvalidDataException("a lattice shading without /VerticesPerRow (2 or more)");
        }

        using var data = file.Decode(stream);
        var reader = new BitReader(data.Span.ToArray());
        var output = new List<float>();
        Vertex ReadVertex(bool flag, out int f)
        {
            f = flag ? (int)reader.Read(flagBits) : 0;
            double x = Decode(reader.Read(coordinateBits), coordinateBits, decode[0], decode[1]);
            double y = Decode(reader.Read(coordinateBits), coordinateBits, decode[2], decode[3]);
            var c = new double[components];
            for (int k = 0; k < components; k++)
            {
                c[k] = Decode(reader.Read(componentBits), componentBits, decode[4 + 2 * k], decode[5 + 2 * k]);
            }

            reader.Align();
            return new Vertex(x, y, c);
        }

        void Triangle(Vertex a, Vertex b, Vertex c)
        {
            if (output.Count / 15 >= MaxTriangles)
            {
                throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture, $"a mesh shading of more than {MaxTriangles:N0} triangles"));
            }

            foreach (var p in (Vertex[])[a, b, c])
            {
                var (px, py) = Apply(ctm, p.X, p.Y);
                var rgb = colour(p.C);
                output.AddRange([(float)px, (float)py, rgb.Red, rgb.Green, rgb.Blue]);
            }
        }

        switch (type)
        {
            case 4:
            {
                Vertex? va = null, vb = null, vc = null;
                while (reader.Left >= flagBits + 2 * coordinateBits)
                {
                    var vertex = ReadVertex(true, out int flag);
                    if (flag == 0 || va is null)
                    {
                        if (reader.Left < flagBits + 2 * coordinateBits)
                        {
                            break;
                        }

                        var second = ReadVertex(true, out _);
                        if (reader.Left < flagBits + 2 * coordinateBits)
                        {
                            break;
                        }

                        var third = ReadVertex(true, out _);
                        (va, vb, vc) = (vertex, second, third);
                    }
                    else if (flag == 1)
                    {
                        (va, vb, vc) = (vb, vc, vertex);
                    }
                    else
                    {
                        (vb, vc) = (vc, vertex);
                    }

                    Triangle(va!, vb!, vc!);
                }

                break;
            }

            case 5:
            {
                var rows = new List<Vertex[]>();
                while (reader.Left >= 2 * coordinateBits)
                {
                    var row = new Vertex[perRow];
                    int k = 0;
                    for (; k < perRow && reader.Left >= 2 * coordinateBits; k++)
                    {
                        row[k] = ReadVertex(false, out _);
                    }

                    if (k < perRow)
                    {
                        break;
                    }

                    if (rows.Count > 0)
                    {
                        var above = rows[^1];
                        for (int i = 0; i + 1 < perRow; i++)
                        {
                            Triangle(above[i], above[i + 1], row[i]);
                            Triangle(above[i + 1], row[i + 1], row[i]);
                        }
                    }

                    rows.Add(row);
                }

                break;
            }

            default:
                Patches(reader, type == 7, flagBits, coordinateBits, componentBits, components, decode, Triangle);
                break;
        }

        return [.. output];
    }

    // Coons (12 points) and tensor (16) patches, each cut into a grid of triangles.
    private static void Patches(BitReader reader, bool tensor, int flagBits, int coordinateBits, int componentBits, int components, double[] decode, Action<Vertex, Vertex, Vertex> triangle)
    {
        int own = tensor ? 16 : 12;
        var previous = new (double X, double Y)[4, 4];
        var previousColours = new double[4][];
        bool have = false;
        while (reader.Left >= flagBits + 2 * coordinateBits)
        {
            int flag = (int)reader.Read(flagBits);
            var p = new (double X, double Y)[4, 4];
            var c = new double[4][];   // c00, c03, c33, c30
            int points = flag == 0 || !have ? own : own - 4, colours = flag == 0 || !have ? 4 : 2;
            if (flag != 0 && have)
            {
                // The shared edge: p00, p01, p02, p03 and c00, c03 from the previous patch's edge the flag names.
                (double X, double Y)[] edge = flag switch
                {
                    1 => [previous[0, 3], previous[1, 3], previous[2, 3], previous[3, 3]],
                    2 => [previous[3, 3], previous[3, 2], previous[3, 1], previous[3, 0]],
                    _ => [previous[3, 0], previous[2, 0], previous[1, 0], previous[0, 0]],
                };
                (p[0, 0], p[0, 1], p[0, 2], p[0, 3]) = (edge[0], edge[1], edge[2], edge[3]);
                (c[0], c[1]) = flag switch
                {
                    1 => (previousColours[1], previousColours[2]),
                    2 => (previousColours[2], previousColours[3]),
                    _ => (previousColours[3], previousColours[0]),
                };
            }

            // The points in the order the specification lists them, the shared four first when there are.
            ReadOnlySpan<(int I, int J)> order = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 3), (2, 3), (3, 3), (3, 2), (3, 1), (3, 0), (2, 0), (1, 0), (1, 1), (1, 2), (2, 2), (2, 1)];
            int start = points == own ? 0 : 4;
            for (int k = start; k < own; k++)
            {
                if (reader.Left < 2 * coordinateBits)
                {
                    return;
                }

                double x = Decode(reader.Read(coordinateBits), coordinateBits, decode[0], decode[1]);
                double y = Decode(reader.Read(coordinateBits), coordinateBits, decode[2], decode[3]);
                p[order[k].I, order[k].J] = (x, y);
            }

            for (int k = 4 - colours; k < 4; k++)
            {
                var value = new double[components];
                for (int m = 0; m < components; m++)
                {
                    value[m] = Decode(reader.Read(componentBits), componentBits, decode[4 + 2 * m], decode[5 + 2 * m]);
                }

                c[k] = value;
            }

            reader.Align();
            if (!tensor)
            {
                // The Coons patch's inner points from its edges (8.7.4.5.8).
                p[1, 1] = Inner(p[0, 0], p[0, 1], p[1, 0], p[0, 3], p[3, 0], p[3, 1], p[1, 3], p[3, 3]);
                p[1, 2] = Inner(p[0, 3], p[0, 2], p[1, 3], p[0, 0], p[3, 3], p[3, 2], p[1, 0], p[3, 0]);
                p[2, 1] = Inner(p[3, 0], p[3, 1], p[2, 0], p[3, 3], p[0, 0], p[0, 1], p[2, 3], p[0, 3]);
                p[2, 2] = Inner(p[3, 3], p[3, 2], p[2, 3], p[3, 0], p[0, 3], p[0, 2], p[2, 0], p[0, 0]);
            }

            Grid(p, c, components, triangle);
            (previous, previousColours, have) = (p, c, true);
        }
    }

    // (-4 a + 6 (b + c) - 2 (d + e) + 3 (f + g) - h) / 9
    private static (double X, double Y) Inner((double X, double Y) a, (double X, double Y) b, (double X, double Y) c, (double X, double Y) d, (double X, double Y) e,
        (double X, double Y) f, (double X, double Y) g, (double X, double Y) h) =>
        ((-4 * a.X + 6 * (b.X + c.X) - 2 * (d.X + e.X) + 3 * (f.X + g.X) - h.X) / 9, (-4 * a.Y + 6 * (b.Y + c.Y) - 2 * (d.Y + e.Y) + 3 * (f.Y + g.Y) - h.Y) / 9);

    // The tensor surface S(u, v) = sum of p[i, j] B_i(u) B_j(v) at a grid of (u, v), its colours bilinear in the corners'.
    private static void Grid((double X, double Y)[,] p, double[][] c, int components, Action<Vertex, Vertex, Vertex> triangle)
    {
        var at = new Vertex[PatchSteps + 1, PatchSteps + 1];
        for (int a = 0; a <= PatchSteps; a++)
        {
            double u = (double)a / PatchSteps;
            for (int b = 0; b <= PatchSteps; b++)
            {
                double v = (double)b / PatchSteps;
                double x = 0, y = 0;
                for (int i = 0; i < 4; i++)
                {
                    double bu = Bernstein(i, u);
                    for (int j = 0; j < 4; j++)
                    {
                        double w = bu * Bernstein(j, v);
                        x += w * p[i, j].X;
                        y += w * p[i, j].Y;
                    }
                }

                var colour = new double[components];
                for (int k = 0; k < components; k++)
                {
                    colour[k] = (1 - u) * (1 - v) * c[0][k] + (1 - u) * v * c[1][k] + u * v * c[2][k] + u * (1 - v) * c[3][k];
                }

                at[a, b] = new Vertex(x, y, colour);
            }
        }

        for (int a = 0; a < PatchSteps; a++)
        {
            for (int b = 0; b < PatchSteps; b++)
            {
                triangle(at[a, b], at[a + 1, b], at[a, b + 1]);
                triangle(at[a + 1, b], at[a + 1, b + 1], at[a, b + 1]);
            }
        }
    }

    private static double Bernstein(int i, double t) => i switch
    {
        0 => (1 - t) * (1 - t) * (1 - t),
        1 => 3 * t * (1 - t) * (1 - t),
        2 => 3 * t * t * (1 - t),
        _ => t * t * t,
    };

    private static int Bits(PdfFile file, PdfValue value) =>
        file.Resolve(value).TryInteger(out long bits) && bits is 1 or 2 or 4 or 8 or 12 or 16 or 24 or 32 ? (int)bits : 0;

    private static double Decode(ulong value, int bits, double low, double high) => low + value * (high - low) / (bits == 64 ? ulong.MaxValue : (1UL << bits) - 1);

    internal static (double X, double Y) Apply(PdfMatrix m, double x, double y) => (m.A * x + m.C * y + m.E, m.B * x + m.D * y + m.F);

    private sealed record Vertex(double X, double Y, double[] C);

    // Bits read from the most significant first.
    private sealed class BitReader(byte[] data)
    {
        private long _bit;

        public long Left => 8L * data.Length - _bit;

        public ulong Read(int bits)
        {
            ulong v = 0;
            for (int k = 0; k < bits; k++, _bit++)
            {
                long at = _bit >> 3;
                v = v << 1 | (uint)(at < data.Length ? (data[at] >> (7 - (int)(_bit & 7))) & 1 : 0);
            }

            return v;
        }

        public void Align() => _bit = (_bit + 7) & ~7L;
    }
}
