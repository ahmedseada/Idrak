// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;

namespace Idrak.Data.Pdf;

/// <summary>How an image's samples become pixels: its colour space by what is done with it.</summary>
internal enum PdfImageColour : byte
{
    /// <summary>One component, grey (DeviceGray, CalGray, ICCBased with one).</summary>
    Grey,

    /// <summary>Three components, red, green and blue (DeviceRGB, CalRGB, ICCBased with three).</summary>
    Rgb,

    /// <summary>Four components, inks (DeviceCMYK, ICCBased with four).</summary>
    Cmyk,

    /// <summary>One index into a table of a base space's colours.</summary>
    Indexed,

    /// <summary>CIE L*a*b*, under the space's white point.</summary>
    Lab,

    /// <summary>An image mask: one bit, painted or not.</summary>
    Stencil,

    /// <summary>Separation or DeviceN: its components through a tint transform into an alternate space (<see cref="PdfColourSpace.Base"/>).</summary>
    Tint,

    /// <summary>A space an image cannot be painted in (Pattern): named, not decoded.</summary>
    Unsupported,

    /// <summary>Not a colour space (damaged).</summary>
    Unknown,
}

/// <summary>
/// A colour space as an image uses it (ISO 32000-2, 8.6): its kind, its components, and for an indexed space its base,
/// highest index and table, for Lab its white point and ranges. Profiles and calibrations are not applied: ICC-based and
/// calibrated spaces are read by their component count.
/// </summary>
internal sealed class PdfColourSpace
{
    public static PdfColourSpace Grey { get; } = new() { Kind = PdfImageColour.Grey, Components = 1, Name = "DeviceGray" };

    public static PdfColourSpace Rgb { get; } = new() { Kind = PdfImageColour.Rgb, Components = 3, Name = "DeviceRGB" };

    public static PdfColourSpace Cmyk { get; } = new() { Kind = PdfImageColour.Cmyk, Components = 4, Name = "DeviceCMYK" };

    public PdfImageColour Kind { get; init; }

    public int Components { get; init; }

    /// <summary>Its name as the file gives it ("DeviceRGB", "ICCBased", "Separation").</summary>
    public string Name { get; init; } = "";

    /// <summary>An indexed space's base (grey, RGB, CMYK, Lab or a tint), or a tint's alternate space.</summary>
    public PdfColourSpace? Base { get; init; }

    /// <summary>A Separation's or DeviceN's tint transform (its components to the alternate space's).</summary>
    public PdfFunction? Function { get; init; }

    private byte[]? _table;

    /// <summary>An indexed space's table read once ((highest + 1) x the base's components bytes, zeros past what the file holds).</summary>
    public byte[] Table(PdfFile file)
    {
        if (Volatile.Read(ref _table) is { } known)
        {
            return known;
        }

        byte[] table = new byte[(Highest + 1) * (Base?.Components ?? 1)];
        var lookup = file.Resolve(Lookup);
        if (lookup.AsString is { } text)
        {
            text.Bytes.AsSpan(0, Math.Min(text.Bytes.Length, table.Length)).CopyTo(table);
        }
        else if (lookup.AsStream is { } stream)
        {
            using var decoded = file.Decode(stream);
            decoded.Span[..Math.Min(decoded.Length, table.Length)].CopyTo(table);
        }
        else
        {
            throw new InvalidDataException($"an Indexed space's table is {lookup}, not a string or a stream");
        }

        Volatile.Write(ref _table, table);
        return table;
    }

    /// <summary>An indexed space's highest index (0-255).</summary>
    public int Highest { get; init; }

    /// <summary>An indexed space's table: a string or a stream of (highest + 1) x the base's components bytes.</summary>
    public PdfValue Lookup { get; init; }

    /// <summary>Lab's white point (X, Y, Z).</summary>
    public double[] White { get; init; } = [0.9505, 1, 1.089];

    /// <summary>Lab's ranges of a* and b* (amin, amax, bmin, bmax).</summary>
    public double[] Range { get; init; } = [-100, 100, -100, 100];

    /// <summary>Why it cannot be decoded, or null.</summary>
    public string? Problem { get; init; }

    /// <summary>The channels its pixels have: 1 for grey (and an index into grey), else 3.</summary>
    public int Channels => Kind switch
    {
        PdfImageColour.Grey => 1,
        PdfImageColour.Indexed or PdfImageColour.Tint => Base?.Channels ?? 3,
        _ => 3,
    };

    /// <summary>The default /Decode of one component at <paramref name="bits"/> bits (8.9.5.2, table 89).</summary>
    public (double Low, double High) DefaultDecode(int component, int bits) => Kind switch
    {
        PdfImageColour.Indexed => (0, (1 << bits) - 1),
        PdfImageColour.Lab when component == 0 => (0, 100),
        PdfImageColour.Lab => (Range[2 * (component - 1)], Range[2 * (component - 1) + 1]),
        _ => (0, 1),
    };

    /// <summary>
    /// The colour space <paramref name="value"/> names: a device space (or an inline image's abbreviation), an array
    /// (ICCBased, CalGray, CalRGB, Lab, Indexed, and the spaces read later), or a name from the resources' /ColorSpace.
    /// </summary>
    public static PdfColourSpace Read(PdfFile file, PdfValue value, PdfDictionary? resources, int depth = 0)
    {
        if (depth > 4)
        {
            return Damaged("colour spaces that name each other");
        }

        var resolved = file.Resolve(value);
        if (resolved.AsName is { } name)
        {
            return Named(file, name, resources, depth);
        }

        if (resolved.AsArray is not { Count: > 0 } array || file.Resolve(array[0]).AsName is not { } family)
        {
            return Damaged(resolved.IsNull ? "no /ColorSpace" : $"the colour space {resolved}");
        }

        if (family.Is("ICCBased"u8))
        {
            var profile = array.Count > 1 ? file.Dictionary(array[1]) : null;
            if (profile is not null && file.Resolve(profile["N"u8]).TryInteger(out long n))
            {
                return n switch
                {
                    1 => new PdfColourSpace { Kind = PdfImageColour.Grey, Components = 1, Name = "ICCBased" },
                    3 => new PdfColourSpace { Kind = PdfImageColour.Rgb, Components = 3, Name = "ICCBased" },
                    4 => new PdfColourSpace { Kind = PdfImageColour.Cmyk, Components = 4, Name = "ICCBased" },
                    _ => Damaged($"an ICC profile of {n} components"),
                };
            }

            return profile is not null && profile.Contains("Alternate"u8) ? Read(file, profile["Alternate"u8], resources, depth + 1) : Damaged("an ICCBased space without /N");
        }

        if (family.Is("CalGray"u8))
        {
            return new PdfColourSpace { Kind = PdfImageColour.Grey, Components = 1, Name = "CalGray" };
        }

        if (family.Is("CalRGB"u8))
        {
            return new PdfColourSpace { Kind = PdfImageColour.Rgb, Components = 3, Name = "CalRGB" };
        }

        if (family.Is("Lab"u8))
        {
            var parameters = array.Count > 1 ? file.Dictionary(array[1]) : null;
            double[] white = Numbers(file, parameters?["WhitePoint"u8] ?? PdfValue.Null, 3) ?? [0.9505, 1, 1.089];
            double[] range = Numbers(file, parameters?["Range"u8] ?? PdfValue.Null, 4) ?? [-100, 100, -100, 100];
            return white[0] > 0 && white[1] > 0 && white[2] > 0
                ? new PdfColourSpace { Kind = PdfImageColour.Lab, Components = 3, Name = "Lab", White = white, Range = range }
                : Damaged("a Lab space whose white point is not positive");
        }

        if (family.Is("Indexed"u8) || family.Is("I"u8))
        {
            if (array.Count < 4)
            {
                return Damaged("an Indexed space without its base, highest index and table");
            }

            var baseSpace = Read(file, array[1], resources, depth + 1);
            if (baseSpace.Kind is not (PdfImageColour.Grey or PdfImageColour.Rgb or PdfImageColour.Cmyk or PdfImageColour.Lab or PdfImageColour.Tint))
            {
                return baseSpace.Kind == PdfImageColour.Unsupported
                    ? new PdfColourSpace { Kind = PdfImageColour.Unsupported, Components = 1, Name = $"Indexed over {baseSpace.Name}", Problem = baseSpace.Problem }
                    : Damaged($"an Indexed space over {baseSpace.Name}");
            }

            long highest = file.Resolve(array[2]).TryInteger(out long h) ? h : -1;
            return highest >= 0
                ? new PdfColourSpace { Kind = PdfImageColour.Indexed, Components = 1, Name = "Indexed", Base = baseSpace, Highest = (int)Math.Min(highest, 255), Lookup = array[3] }
                : Damaged("an Indexed space whose highest index is not a number");
        }

        if (array.Count == 1)
        {
            return Named(file, family, resources, depth);
        }

        if (family.Is("Separation"u8) || family.Is("DeviceN"u8))
        {
            return Tint(file, array, resources, depth, family.ToString(), family.Is("DeviceN"u8));
        }

        return family.Is("Pattern"u8) ? Later(family.ToString()) : Damaged($"the colour space /{family}");
    }

    // [/Separation name alternate tintTransform] or [/DeviceN names alternate tintTransform attributes]: one component a
    // colourant, through the transform into the alternate space (a device, ICC-based, calibrated, Lab or indexed one).
    private static PdfColourSpace Tint(PdfFile file, PdfArray array, PdfDictionary? resources, int depth, string family, bool deviceN)
    {
        if (array.Count < 4)
        {
            return Damaged($"a {family} space without its alternate space and tint transform");
        }

        int components = deviceN ? file.Resolve(array[1]).AsArray?.Count ?? 0 : 1;
        if (components is < 1 or > 32)
        {
            return Damaged($"a DeviceN space of {components} colourants");
        }

        var alternate = Read(file, array[2], resources, depth + 1);
        if (alternate.Kind is not (PdfImageColour.Grey or PdfImageColour.Rgb or PdfImageColour.Cmyk or PdfImageColour.Lab or PdfImageColour.Indexed))
        {
            return Damaged($"a {family} space over {alternate.Name}");
        }

        PdfFunction function;
        try
        {
            function = PdfFunction.Read(file, array[3]);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            return Damaged($"a {family} space's tint transform: {e.Message}");
        }

        return new PdfColourSpace { Kind = PdfImageColour.Tint, Components = components, Name = family, Base = alternate, Function = function };
    }

    private static PdfColourSpace Named(PdfFile file, PdfName name, PdfDictionary? resources, int depth)
    {
        if (name.Is("DeviceGray"u8) || name.Is("G"u8) || name.Is("CalGray"u8))
        {
            return Grey;
        }

        if (name.Is("DeviceRGB"u8) || name.Is("RGB"u8) || name.Is("CalRGB"u8))
        {
            return Rgb;
        }

        if (name.Is("DeviceCMYK"u8) || name.Is("CMYK"u8))
        {
            return Cmyk;
        }

        if (name.Is("Pattern"u8))
        {
            return Later(name.ToString());
        }

        if (name.Is("Separation"u8) || name.Is("DeviceN"u8))
        {
            return Damaged($"the colour space /{name} without its colourants, alternate space and tint transform");
        }

        // A name the resources give (an inline image's colour space, or a writer's habit).
        if (resources is not null && file.Dictionary(resources["ColorSpace"u8]) is { } spaces && spaces.TryGet(name.Bytes, out var named) && !named.IsNull)
        {
            return Read(file, named, resources, depth + 1);
        }

        return Damaged($"the colour space /{name}, which is neither a device space nor in the page's resources");
    }

    private static PdfColourSpace Later(string name) => new()
    {
        Kind = PdfImageColour.Unsupported, Components = 1, Name = name,
        Problem = $"a raw image in the colour space {name}, which images are not painted in (a pattern paints a path)",
    };

    private static PdfColourSpace Damaged(string what) => new() { Kind = PdfImageColour.Unknown, Components = 1, Name = "unknown", Problem = what };

    // An array of exactly `count` numbers, or null.
    private static double[]? Numbers(PdfFile file, PdfValue value, int count)
    {
        if (file.Resolve(value).AsArray is not { } array || array.Count != count)
        {
            return null;
        }

        var numbers = new double[count];
        for (int i = 0; i < count; i++)
        {
            if (!file.Resolve(array[i]).TryNumber(out numbers[i]) || !double.IsFinite(numbers[i]))
            {
                return null;
            }
        }

        return numbers;
    }
}

/// <summary>
/// An image's dictionary read (ISO 32000-2, 8.9.5; an inline image's with its keys spelled out): its size, bits a
/// component, colour space, /Decode, encoding (the first image filter in its chain, else raw) and masks. Nothing is
/// decoded.
/// </summary>
internal sealed class PdfImageInfo
{
    public int Width { get; init; }

    public int Height { get; init; }

    public int Bits { get; init; }

    /// <summary>The encoding as <see cref="DocumentImage.Encoding"/> names it.</summary>
    public string Encoding { get; init; } = DocumentImage.Raw;

    public PdfColourSpace Space { get; init; } = PdfColourSpace.Grey;

    public bool Stencil { get; init; }

    /// <summary>Two numbers a component: what sample 0 and the largest sample stand for.</summary>
    public double[] Decode { get; init; } = [];

    /// <summary>/SMask (a stream) and /Mask (a stream or a colour key array), as written.</summary>
    public PdfValue SoftMask { get; init; }

    public PdfValue Mask { get; init; }

    /// <summary>Why a raw image cannot be decoded (null when it can), and whether that is a feature not read yet rather than damage.</summary>
    public string? Problem { get; init; }

    public bool NotSupported { get; init; }

    /// <summary>The components each pixel's samples hold.</summary>
    public int Components => Stencil ? 1 : Space.Components;

    /// <summary>The channels of the decoded pixels (1 grey, 3 RGB).</summary>
    public int Channels => Stencil ? 1 : Space.Channels;

    /// <summary>Whether the pixels are kept at 16 bits (grey and RGB of 16 bits: float values, not 8-bit planes).</summary>
    public bool Sixteen => Bits == 16 && !Stencil && Space.Kind is PdfImageColour.Grey or PdfImageColour.Rgb;

    /// <summary>
    /// Reads an image's dictionary (an inline image's after <see cref="Spelled"/>); null, with <paramref name="skipped"/>
    /// saying why, when its size cannot be read. <paramref name="stencil"/> reads it as an image mask whatever it says (an
    /// explicit /Mask).
    /// </summary>
    public static PdfImageInfo? Read(PdfFile file, PdfDictionary dictionary, PdfDictionary? resources, bool stencil, out string? skipped)
    {
        skipped = null;
        if (!Whole(file, dictionary["Width"u8], out long width) || !Whole(file, dictionary["Height"u8], out long height) || width <= 0 || height <= 0
            || width > int.MaxValue || height > int.MaxValue)
        {
            skipped = $"its size is not two whole numbers above 0 (/Width {dictionary["Width"u8]}, /Height {dictionary["Height"u8]})";
            return null;
        }

        string encoding = DocumentImage.Raw;
        string? problem = null;
        try
        {
            foreach (var filter in file.Filters(dictionary))
            {
                if (PdfFilters.ImageEncoding(filter.Name) is { } image)
                {
                    encoding = image;
                    break;
                }
            }
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            problem = $"its filters: {e.Message}";
        }

        stencil |= file.Resolve(dictionary["ImageMask"u8]).AsBoolean;
        var space = stencil ? PdfColourSpace.Grey : PdfColourSpace.Read(file, dictionary["ColorSpace"u8], resources);
        int bits = stencil ? 1 : Whole(file, dictionary["BitsPerComponent"u8], out long b) && b is 1 or 2 or 4 or 8 or 16 ? (int)b : 0;
        bool notSupported = false;
        if (encoding == DocumentImage.Raw && problem is null)
        {
            if (space.Problem is { } spaceProblem)
            {
                (problem, notSupported) = (spaceProblem, space.Kind == PdfImageColour.Unsupported);
            }
            else if (bits == 0)
            {
                problem = $"/BitsPerComponent {dictionary["BitsPerComponent"u8]}, not 1, 2, 4, 8 or 16";
            }
        }

        // /Decode as written when it gives two numbers a component, else the default.
        int components = stencil ? 1 : space.Components;
        var decode = new double[2 * components];
        var written = file.Resolve(dictionary["Decode"u8]).AsArray;
        bool given = written is not null && written.Count == decode.Length;
        for (int c = 0; c < components; c++)
        {
            var (low, high) = stencil ? (0.0, 1.0) : space.DefaultDecode(c, Math.Max(bits, 1));
            if (given && file.Resolve(written![2 * c]).TryNumber(out double l) && file.Resolve(written[2 * c + 1]).TryNumber(out double h) && double.IsFinite(l) && double.IsFinite(h))
            {
                (low, high) = (l, h);
            }

            (decode[2 * c], decode[2 * c + 1]) = (low, high);
        }

        return new PdfImageInfo
        {
            Width = (int)width, Height = (int)height, Bits = bits, Encoding = encoding, Space = space, Stencil = stencil, Decode = decode,
            SoftMask = dictionary["SMask"u8], Mask = dictionary["Mask"u8], Problem = problem, NotSupported = notSupported,
        };
    }

    private static bool Whole(PdfFile file, PdfValue value, out long number) => file.Resolve(value).TryInteger(out number);

    /// <summary>An inline image's entries (8.9.7) with their keys spelled out, so they read as an image XObject's.</summary>
    public static PdfDictionary Spelled(PdfDictionary inline)
    {
        var keys = new PdfName[inline.Count];
        var values = new PdfValue[inline.Count];
        for (int i = 0; i < keys.Length; i++)
        {
            var key = inline.KeyAt(i);
            keys[i] = key.Bytes switch
            {
                [(byte)'W'] => PdfName.From("Width"u8),
                [(byte)'H'] => PdfName.From("Height"u8),
                [(byte)'B', (byte)'P', (byte)'C'] => PdfName.From("BitsPerComponent"u8),
                [(byte)'C', (byte)'S'] => PdfName.From("ColorSpace"u8),
                [(byte)'D'] => PdfName.From("Decode"u8),
                [(byte)'D', (byte)'P'] => PdfName.From("DecodeParms"u8),
                [(byte)'F'] => PdfName.From("Filter"u8),
                [(byte)'I', (byte)'M'] => PdfName.From("ImageMask"u8),
                [(byte)'I'] => PdfName.From("Interpolate"u8),
                [(byte)'L'] => PdfName.From("Length"u8),
                _ => key,
            };
            values[i] = inline.ValueAt(i);
        }

        return new PdfDictionary(keys, values);
    }
}

/// <summary>
/// One image of a page (an XObject, or an inline image's entries and data): its bytes as stored, its pixels and its mask,
/// each read when asked, bounded by <see cref="DocumentOptions.MaxStreamBytes"/>. Raw images are decoded here: samples of
/// 1 to 16 bits (rows padded to whole bytes) mapped through /Decode, grey and RGB (and the calibrated and ICC-based
/// spaces by their component count) into 8-bit planes, 16-bit grey and RGB into values; CMYK to RGB as the library's JPEG
/// codec turns ink into light ((255 - K) - X (255 - K) / 255); indexed colours through their table; Lab through XYZ (its
/// white point scaled to D65) into sRGB; an image mask as grey (0 where it paints, 255 where it does not). Other encodings
/// (JPEG, JPEG 2000, CCITT fax, JBIG2) go to the image codec registered by their name, whose decode is theirs. Safe from
/// several threads.
/// </summary>
internal sealed class PdfImage
{
    private const int RowsGrain = 1 << 16;   // pixels a chunk of rows spread over the cores holds at least

    private static readonly float[] Levels = [.. Enumerable.Range(0, 256).Select(k => k * (1f / 255f))];   // as the image codecs give 8-bit values

    private readonly PdfFile _file;
    private readonly PdfStream? _stream;
    private readonly byte[]? _inline;
    private readonly PdfFilter[] _inlineFilters = [];
    private readonly string _where;

    private PdfImage(PdfFile file, PdfImageInfo info, PdfStream? stream, byte[]? inline, PdfFilter[] inlineFilters, string where)
    {
        (_file, Info, _stream, _inline, _inlineFilters, _where) = (file, info, stream, inline, inlineFilters, where);
    }

    public PdfImageInfo Info { get; }

    /// <summary>The image XObject <paramref name="stream"/>, or null (with <paramref name="skipped"/>) when its size cannot be read.</summary>
    public static PdfImage? FromStream(PdfFile file, PdfStream stream, PdfDictionary? resources, string where, bool stencil, out string? skipped)
    {
        var info = PdfImageInfo.Read(file, stream.Dictionary, resources, stencil, out skipped);
        return info is null ? null : new PdfImage(file, info, stream, null, [], where);
    }

    /// <summary>An inline image: its entries (keys spelled out) and its data as the content holds it (a copy, now the image's).</summary>
    public static PdfImage? FromInline(PdfFile file, PdfDictionary entries, byte[] data, PdfDictionary? resources, string where, out string? skipped)
    {
        var info = PdfImageInfo.Read(file, entries, resources, stencil: false, out skipped);
        if (info is null)
        {
            return null;
        }

        PdfFilter[] filters = [];
        try
        {
            filters = file.Filters(entries);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            // Read already counted it among the image's problems.
        }

        return new PdfImage(file, info, null, data, filters, where);
    }

    /// <summary>
    /// The image placed by <paramref name="ctm"/> (its unit square in the file's user space, mapped onto the page as
    /// shown): its matrix from its top-left corner, the box around it, its place in the content.
    /// </summary>
    public DocumentImage Place(PdfMatrix ctm, int order, PdfPartStyle style)
    {
        // The image's rows run down from its top: (u, v) from its top-left corner is (u, 1 - v) of its unit square.
        var m = new PdfMatrix(1, 0, 0, -1, 0, 1).Then(ctm);
        double minX = Math.Min(Math.Min(m.E, m.A + m.E), Math.Min(m.C + m.E, m.A + m.C + m.E));
        double maxX = Math.Max(Math.Max(m.E, m.A + m.E), Math.Max(m.C + m.E, m.A + m.C + m.E));
        double minY = Math.Min(Math.Min(m.F, m.B + m.F), Math.Min(m.D + m.F, m.B + m.D + m.F));
        double maxY = Math.Max(Math.Max(m.F, m.B + m.F), Math.Max(m.D + m.F, m.B + m.D + m.F));
        bool finite = double.IsFinite(minX) && double.IsFinite(maxX) && double.IsFinite(minY) && double.IsFinite(maxY) && maxX - minX < float.MaxValue && maxY - minY < float.MaxValue;
        return new DocumentImage(Info.Encoding, Info.Width, Info.Height, Bytes, Decode, Mask)
        {
            Box = finite ? new DocumentBox((float)minX, (float)minY, (float)(maxX - minX), (float)(maxY - minY)) : DocumentBox.Empty,
            Matrix = finite ? new DocumentMatrix((float)m.A, (float)m.B, (float)m.C, (float)m.D, (float)m.E, (float)m.F) : null,
            Order = order,
            Clip = style.Clip, Alpha = style.Alpha, Stencil = Info.Stencil ? style.Colour : null, Blend = style.Blend, Group = style.Group, SoftMask = style.SoftMask,
        };
    }

    /// <summary>The bytes as stored, through every filter before the image encoding: a raw image's samples, a JPEG's file.</summary>
    public ReadOnlyMemory<byte> Bytes() => Guard(() =>
    {
        using var data = Data();
        return (ReadOnlyMemory<byte>)data.Span.ToArray();
    });

    /// <summary>The pixels (see the class).</summary>
    public ImageData Decode() => Guard(() => Info.Encoding == DocumentImage.Raw ? DecodeRaw(PdfPixelMode.Pixels, null) : DecodeEncoded());

    /// <summary>
    /// The mask as grey, how much of the image shows at each of its pixels (255 fully, 0 not at all), at its own size: a
    /// soft mask's levels; an explicit mask or an image mask 255 where it paints; a colour key 0 where every component
    /// falls in its range. Null when the image has none.
    /// </summary>
    public ImageData? Mask() => Guard<ImageData?>(() =>
    {
        if (Info.Stencil)
        {
            return Info.Encoding == DocumentImage.Raw ? DecodeRaw(PdfPixelMode.Coverage, null) : Inverted(DecodeEncoded());
        }

        if (_file.Resolve(Info.SoftMask).AsStream is { } soft)
        {
            var image = FromStream(_file, soft, null, $"{_where}'s soft mask (object {soft.Number})", stencil: false, out string? skipped)
                        ?? throw new InvalidDataException($"{_where}'s soft mask (object {soft.Number}): {skipped}");
            var grey = image.Decode();
            return grey.Channels == 1 ? grey : new ImageData(grey.Resize(1, grey.Height, grey.Width), 1, grey.Height, grey.Width);
        }

        var mask = _file.Resolve(Info.Mask);
        if (mask.AsStream is { } explicitMask)
        {
            var image = FromStream(_file, explicitMask, null, $"{_where}'s mask (object {explicitMask.Number})", stencil: true, out string? skipped)
                        ?? throw new InvalidDataException($"{_where}'s mask (object {explicitMask.Number}): {skipped}");
            return image.Info.Encoding == DocumentImage.Raw ? image.DecodeRaw(PdfPixelMode.Coverage, null) : Inverted(image.DecodeEncoded());
        }

        if (mask.AsArray is { } key && key.Count == 2 * Info.Components && !Info.Stencil)
        {
            var ranges = new int[key.Count];
            for (int i = 0; i < ranges.Length; i++)
            {
                ranges[i] = _file.Resolve(key[i]).TryInteger(out long v) ? (int)Math.Clamp(v, -1, 65536) : -1;
            }

            return Info.Encoding == DocumentImage.Raw ? DecodeRaw(PdfPixelMode.ColourKey, ranges) : KeyOf(DecodeEncoded(), ranges);
        }

        return null;
    });

    // ---- reading ---------------------------------------------------------------------------------------------------

    // The stored data through every filter before the image encoding (decrypted first), bounded by MaxStreamBytes.
    private PdfDecoded Data()
    {
        if (_stream is { } stream)
        {
            return _file.Decode(stream);
        }

        return PdfFilters.Decode(_inline, null, _inlineFilters, _file.MaxStreamBytes,
            string.Create(CultureInfo.InvariantCulture,
                $"{_where}: its data decodes to more than the limit {nameof(DocumentOptions.MaxStreamBytes)} allows (DocumentOptions.MaxStreamBytes, or {DocumentOptions.MaxStreamBytesVariable})."));
    }

    // The pixels' bytes within MaxStreamBytes, checked before anything is decoded.
    private void CheckPixels(long width, long height, long channels, long bytesEach)
    {
        long bytes = width * height * channels * bytesEach;
        if (bytes > _file.MaxStreamBytes || bytes > Array.MaxLength || width * channels > Array.MaxLength)
        {
            throw PdfLimit.Error(string.Create(CultureInfo.InvariantCulture,
                $"{_where}: {width:N0} x {height:N0} pixels decode to {bytes:N0} bytes, more than the limit {nameof(DocumentOptions.MaxStreamBytes)} allows (DocumentOptions.MaxStreamBytes, or {DocumentOptions.MaxStreamBytesVariable}); raise it to decode this image."));
        }
    }

    private ImageData DecodeRaw(PdfPixelMode mode, int[]? key)
    {
        if (Info.Problem is { } problem)
        {
            throw Info.NotSupported ? new NotSupportedException($"{_where}: {problem}.") : new InvalidDataException($"{_where}: {problem}.");
        }

        int channels = mode == PdfPixelMode.Pixels ? Info.Channels : 1;
        CheckPixels(Info.Width, Info.Height, channels, mode == PdfPixelMode.Pixels && Info.Sixteen ? sizeof(float) : 1);
        CheckPixels(Info.Width, 1, Info.Components, sizeof(ushort));   // a row's samples, unpacked
        if (mode == PdfPixelMode.Pixels && Info.Space.Kind == PdfImageColour.Tint && !Info.Stencil)
        {
            return DecodeTint();
        }

        byte[]? palette = mode == PdfPixelMode.Pixels && Info.Space.Kind == PdfImageColour.Indexed && !Info.Stencil ? Palette() : null;
        using var data = Data();
        var converter = new PdfPixelConverter(Info, data.Memory, mode, key, palette);
        long pixels = (long)Info.Width * Info.Height;
        int grain = (int)Math.Max(1, RowsGrain / Math.Max(1, Info.Width));
        HostParallel.For(Info.Height, pixels >= 2 * RowsGrain ? grain : Info.Height, converter.Rows);
        return converter.Sixteen ? new ImageData(converter.Values!, channels, Info.Height, Info.Width) : new ImageData(converter.Planes!, Info.Height, Info.Width, Levels);
    }

    // A Separation or DeviceN image: each pixel's samples through /Decode and the tint transform into the alternate
    // space, as 8-bit planes (grey, or RGB); a pixel's samples met before are not transformed again.
    private ImageData DecodeTint()
    {
        int width = Info.Width, height = Info.Height, components = Info.Components, bits = Info.Bits, channels = Info.Channels;
        var planes = new byte[channels][];
        for (int c = 0; c < channels; c++)
        {
            planes[c] = new byte[width * height];
        }

        using var data = Data();
        var bytes = data.Span;
        long rowBits = (long)width * components * bits, rowBytes = (rowBits + 7) / 8;
        double max = (1L << bits) - 1;
        var cache = new Dictionary<ulong, int>();
        Span<double> tint = stackalloc double[32];
        Span<byte> rgb = stackalloc byte[3];
        for (int y = 0; y < height; y++)
        {
            long rowStart = y * rowBytes * 8;
            for (int x = 0; x < width; x++)
            {
                ulong key = 0;
                long bit = rowStart + (long)x * components * bits;
                for (int c = 0; c < components; c++)
                {
                    int v = 0;
                    for (int k = 0; k < bits; k++, bit++)
                    {
                        long at = bit >> 3;
                        v = v << 1 | (at < bytes.Length ? (bytes[(int)at] >> (7 - (int)(bit & 7))) & 1 : 0);
                    }

                    tint[c] = Info.Decode[2 * c] + v * (Info.Decode[2 * c + 1] - Info.Decode[2 * c]) / max;
                    key = key * 65537 + (ulong)v;
                }

                if (!cache.TryGetValue(key, out int packed) || cache.Count > 1 << 16)
                {
                    PdfColours.ToBytes(_file, Info.Space, tint[..components], rgb);
                    packed = rgb[0] << 16 | rgb[1] << 8 | rgb[2];
                    if (cache.Count <= 1 << 16)
                    {
                        cache[key] = packed;
                    }
                }

                int i = y * width + x;
                if (channels == 1)
                {
                    planes[0][i] = (byte)(packed >> 16);
                }
                else
                {
                    (planes[0][i], planes[1][i], planes[2][i]) = ((byte)(packed >> 16), (byte)(packed >> 8), (byte)packed);
                }
            }
        }

        return new ImageData(planes, height, width, Levels);
    }

    // An indexed space's colours, each as the decoded pixels hold it (one byte, or three), (highest + 1) of them.
    private byte[] Palette()
    {
        var space = Info.Space;
        var baseSpace = space.Base!;
        int count = space.Highest + 1, components = baseSpace.Components, channels = space.Channels;
        byte[] table;
        try
        {
            table = space.Table(_file);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            throw new InvalidDataException($"{_where}: {e.Message}.", e);
        }

        byte[] palette = new byte[count * channels];
        Span<double> tint = stackalloc double[32];
        Span<byte> rgb = stackalloc byte[3];
        for (int i = 0; i < count; i++)
        {
            var entry = table.AsSpan(i * components, components);
            var colour = palette.AsSpan(i * channels, channels);
            switch (baseSpace.Kind)
            {
                case PdfImageColour.Grey or PdfImageColour.Rgb:
                    entry.CopyTo(colour);
                    break;
                case PdfImageColour.Cmyk:
                    PdfPixelConverter.CmykToRgb(entry[0], entry[1], entry[2], entry[3], colour);
                    break;
                case PdfImageColour.Lab:
                    PdfPixelConverter.LabToRgb(entry[0] * 100.0 / 255,
                        baseSpace.Range[0] + entry[1] * (baseSpace.Range[1] - baseSpace.Range[0]) / 255,
                        baseSpace.Range[2] + entry[2] * (baseSpace.Range[3] - baseSpace.Range[2]) / 255, baseSpace.White, colour);
                    break;
                case PdfImageColour.Tint:
                {
                    for (int c = 0; c < components; c++)
                    {
                        tint[c] = entry[c] / 255.0;
                    }

                    PdfColours.ToBytes(_file, baseSpace, tint[..components], rgb);
                    if (channels == 1)
                    {
                        colour[0] = rgb[0];
                    }
                    else
                    {
                        rgb.CopyTo(colour);
                    }

                    break;
                }
            }
        }

        return palette;
    }

    // A JPEG, JPEG 2000, CCITT or JBIG2 image through the codec registered by its encoding's name: its own decode,
    // bounded by its header's size, which must be the dictionary's.
    private ImageData DecodeEncoded()
    {
        IImageCodec codec;
        try
        {
            codec = ImageCodecs.Get(Info.Encoding);
        }
        catch (NotRegisteredException e)
        {
            throw new NotRegisteredException(e.Registry, e.Name, e.Registered,
                $"{_where} is {Info.Encoding}, which no registered image codec reads (registered: {string.Join(", ", e.Registered)}); register one named \"{Info.Encoding}\" with ImageCodecs.Register (DocumentImage.Bytes gives its data as stored).");
        }

        using var data = Data();
        var bytes = data.Span;
        var header = codec.ReadInfo(bytes[..Math.Min(bytes.Length, ImageCodecs.HeaderBytes)]);
        if (header is { Width: 0 } && bytes.Length > ImageCodecs.HeaderBytes)
        {
            header = codec.ReadInfo(bytes);
        }

        if (header is { Width: > 0, Height: > 0 } stated)
        {
            if (stated.Width != Info.Width || stated.Height != Info.Height)
            {
                throw new InvalidDataException($"{_where}: its {Info.Encoding} data is {stated.Width} x {stated.Height} pixels, its dictionary says {Info.Width} x {Info.Height}.");
            }

            CheckPixels(stated.Width, stated.Height, Math.Max(1, stated.Channels), 1);
        }
        else
        {
            CheckPixels(Info.Width, Info.Height, Math.Max(1, Info.Channels), 1);
        }

        ImageData image;
        try
        {
            image = codec.Decode(bytes);
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException($"{_where}: its {Info.Encoding} data: {e.Message}", e);
        }

        if (image.Width != Info.Width || image.Height != Info.Height)
        {
            throw new InvalidDataException($"{_where}: its {Info.Encoding} data decodes to {image.Width} x {image.Height} pixels, its dictionary says {Info.Width} x {Info.Height}.");
        }

        return image;
    }

    // A grey image turned over: an encoded stencil's ink as coverage.
    private static ImageData Inverted(ImageData grey)
    {
        var values = grey.Channels == 1 ? (float[])grey.Pixels.Clone() : grey.Resize(1, grey.Height, grey.Width);
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = 1f - values[i];
        }

        return new ImageData(values, 1, grey.Height, grey.Width);
    }

    // A colour key over an encoded image's decoded channels (its samples, once the codec has turned them into colour).
    private ImageData KeyOf(ImageData image, int[] ranges)
    {
        if (ranges.Length != 2 * image.Channels)
        {
            throw new InvalidDataException($"{_where}: a colour key of {ranges.Length / 2} components over {image.Channels} decoded channels.");
        }

        var pixels = image.Pixels;
        int size = image.Width * image.Height;
        byte[] plane = new byte[size];
        for (int i = 0; i < size; i++)
        {
            bool inside = true;
            for (int c = 0; c < image.Channels && inside; c++)
            {
                int level = (int)MathF.Round(pixels[c * size + i] * 255f);
                inside = level >= ranges[2 * c] && level <= ranges[2 * c + 1];
            }

            plane[i] = inside ? (byte)0 : (byte)255;
        }

        return new ImageData([plane], image.Height, image.Width, Levels);
    }

    // What a damaged image may throw that is not an InvalidDataException becomes one naming the image; a limit, an
    // encoding no codec reads and a disposed document pass as they are.
    private T Guard<T>(Func<T> read)
    {
        try
        {
            return read();
        }
        catch (InvalidDataException e) when (!e.Message.StartsWith(_file.Name, StringComparison.Ordinal))
        {
            string message = $"{_where}: {e.Message}";
            throw PdfLimit.Is(e) ? PdfLimit.Error(message) : new InvalidDataException(message, e);
        }
        catch (Exception e) when (e is not (InvalidDataException or NotSupportedException or OperationCanceledException or OutOfMemoryException or ObjectDisposedException
                                      or InsufficientExecutionStackException))
        {
            throw new InvalidDataException($"{_where}: damaged ({e.GetType().Name}: {e.Message}).", e);
        }
    }
}

/// <summary>What <see cref="PdfPixelConverter"/> makes of an image's samples.</summary>
internal enum PdfPixelMode : byte
{
    /// <summary>The pixels.</summary>
    Pixels,

    /// <summary>An image mask's coverage: 255 where it paints.</summary>
    Coverage,

    /// <summary>A colour key's coverage: 0 where every component lies in its range.</summary>
    ColourKey,
}

/// <summary>
/// Turns a raw image's samples into pixels, a row at a time (rows of several chunks at once on the cores): each row's
/// samples unpacked from their bits, mapped through tables made once from /Decode (a component's every sample value; a
/// 16-bit sample is mapped on its own, as 65,536 entries a component would cost more than most such images), then into
/// the colour space's grey or RGB. Rows the data does not hold are read as zeros (a damaged image shows what it has).
/// Rounding follows the samples' reference: half to even, on the same arithmetic in the same order.
/// </summary>
internal sealed class PdfPixelConverter
{
    private static readonly double[] SrgbMatrix = [3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.0570];
    private static readonly double[] D65 = [0.9505, 1.0, 1.089];

    private readonly ReadOnlyMemory<byte> _samples;
    private readonly int _width, _components, _bits, _rowBytes, _size;
    private readonly PdfImageColour _colour;
    private readonly PdfPixelMode _mode;
    private readonly byte[][] _bytes = [];        // per component: sample value to an 8-bit level (grey, RGB, CMYK, the image mask)
    private readonly double[][] _lab = [];        // per component: sample value to L*, a* or b*
    private readonly byte[] _index = [];          // sample value to an index of the palette
    private readonly byte[]? _palette;
    private readonly int[]? _key;
    private readonly double[] _white;
    private readonly double[] _decode;
    private readonly int _max, _highest;
    private readonly bool _identity;              // 8-bit grey whose samples are its levels: the rows are copied
    private readonly bool _plainDecode;           // /Decode [0 1] for every component

    public PdfPixelConverter(PdfImageInfo info, ReadOnlyMemory<byte> samples, PdfPixelMode mode, int[]? key, byte[]? palette)
    {
        (_samples, _mode, _key, _palette) = (samples, mode, key, palette);
        (_width, _components, _bits, _colour) = (info.Width, info.Components, info.Bits, info.Stencil ? PdfImageColour.Stencil : info.Space.Kind);
        _rowBytes = (int)(((long)_width * _components * _bits + 7) / 8);
        _size = _width * info.Height;
        int values = 1 << _bits, max = values - 1;
        var decode = info.Decode;
        (_decode, _max, _highest, _white) = (decode, max, info.Space.Highest, info.Space.White);
        _plainDecode = true;
        for (int c = 0; c < decode.Length; c += 2)
        {
            _plainDecode &= decode[c] == 0 && decode[c + 1] == 1;
        }

        int channels = mode == PdfPixelMode.Pixels ? info.Channels : 1;
        Sixteen = mode == PdfPixelMode.Pixels && info.Sixteen;
        if (Sixteen)
        {
            Values = GC.AllocateUninitializedArray<float>(channels * _size);   // every value written by the rows
        }
        else
        {
            Planes = new byte[channels][];
            for (int c = 0; c < channels; c++)
            {
                Planes[c] = GC.AllocateUninitializedArray<byte>(_size);   // every byte written by the rows
            }
        }

        if (mode == PdfPixelMode.ColourKey || _bits == 16)
        {
            return;   // the samples themselves are compared, or each 16-bit sample is mapped on its own
        }

        switch (_colour)
        {
            case PdfImageColour.Stencil:
                _bytes = [new byte[values]];
                for (int v = 0; v < values; v++)
                {
                    bool paints = Sample(decode, 0, v, max) == 0;
                    _bytes[0][v] = (paints ^ (mode == PdfPixelMode.Coverage)) ? (byte)0 : (byte)255;
                }

                break;
            case PdfImageColour.Grey or PdfImageColour.Rgb or PdfImageColour.Cmyk:
                _bytes = new byte[_components][];
                _identity = _bits == 8 && _components == 1;
                for (int c = 0; c < _components; c++)
                {
                    _bytes[c] = new byte[values];
                    for (int v = 0; v < values; v++)
                    {
                        _bytes[c][v] = Level(c, v);
                        _identity &= _bytes[c][v] == v;
                    }
                }

                break;
            case PdfImageColour.Indexed:
                _index = new byte[values];
                for (int v = 0; v < values; v++)
                {
                    _index[v] = (byte)Index(v);
                }

                break;
            case PdfImageColour.Lab:
                _lab = new double[3][];
                for (int c = 0; c < 3; c++)
                {
                    _lab[c] = new double[values];
                    for (int v = 0; v < values; v++)
                    {
                        _lab[c][v] = Sample(decode, c, v, max);
                    }
                }

                break;
        }
    }

    /// <summary>Whether the pixels are 16-bit values (<see cref="Values"/>) rather than 8-bit planes (<see cref="Planes"/>).</summary>
    public bool Sixteen { get; }

    public byte[][]? Planes { get; }

    public float[]? Values { get; }

    // What sample v of component c stands for: /Decode's low end plus v of the largest sample's share of the range.
    private static double Sample(double[] decode, int c, int v, int max) => decode[2 * c] + v * (decode[2 * c + 1] - decode[2 * c]) / max;

    // Sample v of component c as an 8-bit level (grey, RGB, CMYK ink), a 16-bit level's value, an index of the palette.
    private byte Level(int c, int v) => (byte)Math.Round(Math.Clamp(Sample(_decode, c, v, _max), 0, 1) * 255, MidpointRounding.ToEven);

    private float Wide(int c, int v) =>
        _plainDecode ? v / 65535f : (float)Math.Round(Math.Clamp(Sample(_decode, c, v, _max), 0, 1) * 65535, MidpointRounding.ToEven) / 65535f;

    private int Index(int v) => (int)Math.Clamp(Math.Round(Sample(_decode, 0, v, _max), MidpointRounding.ToEven), 0, _highest);

    /// <summary>Rows [first, last), in order: unpacked, then turned into pixels.</summary>
    public void Rows(int first, int last)
    {
        var samples = _samples.Span;
        int count = _width * _components;
        ushort[] row = _bits == 8 ? [] : System.Buffers.ArrayPool<ushort>.Shared.Rent(count);
        byte[]? padded = null;
        try
        {
            for (int y = first; y < last; y++)
            {
                long start = (long)y * _rowBytes;
                ReadOnlySpan<byte> bytes;
                if (start + _rowBytes <= samples.Length)
                {
                    bytes = samples.Slice((int)start, _rowBytes);
                }
                else
                {
                    // A row the data holds only part of (or none of): the rest read as zeros.
                    padded ??= System.Buffers.ArrayPool<byte>.Shared.Rent(_rowBytes);
                    padded.AsSpan(0, _rowBytes).Clear();
                    if (start < samples.Length)
                    {
                        samples[(int)start..].CopyTo(padded);
                    }

                    bytes = padded.AsSpan(0, _rowBytes);
                }

                int offset = y * _width;
                if (_bits == 8)
                {
                    if (_identity && _mode == PdfPixelMode.Pixels)
                    {
                        bytes.CopyTo(Planes![0].AsSpan(offset, _width));
                    }
                    else
                    {
                        Convert(new EightBits(bytes), offset);
                    }

                    continue;
                }

                var values = row.AsSpan(0, count);
                Unpack(bytes, values);
                if (_bits == 16 && _mode == PdfPixelMode.Pixels)
                {
                    ConvertSixteen(values, offset);
                }
                else
                {
                    Convert(new Unpacked(values), offset);
                }
            }
        }
        finally
        {
            if (row.Length > 0)
            {
                System.Buffers.ArrayPool<ushort>.Shared.Return(row);
            }

            if (padded is not null)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(padded);
            }
        }
    }

    // A row's samples of 1, 2, 4 or 16 bits as numbers, most significant bit first.
    private void Unpack(ReadOnlySpan<byte> bytes, Span<ushort> values)
    {
        if (_bits == 16)
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = (ushort)(bytes[2 * i] << 8 | bytes[2 * i + 1]);
            }

            return;
        }

        int bits = _bits, mask = (1 << bits) - 1, perByte = 8 / bits;
        for (int i = 0; i < values.Length; i++)
        {
            int at = i / perByte, shift = 8 - bits - (i % perByte) * bits;
            values[i] = (ushort)((bytes[at] >> shift) & mask);
        }
    }

    private void Convert<T>(T s, int offset) where T : ISamples, allows ref struct
    {
        int width = _width;
        if (_mode == PdfPixelMode.ColourKey)
        {
            var plane = Planes![0].AsSpan(offset, width);
            var key = _key!;
            for (int x = 0; x < width; x++)
            {
                bool inside = true;
                for (int c = 0; c < _components && inside; c++)
                {
                    int v = s[x * _components + c];
                    inside = v >= key[2 * c] && v <= key[2 * c + 1];
                }

                plane[x] = inside ? (byte)0 : (byte)255;
            }

            return;
        }

        switch (_colour)
        {
            case PdfImageColour.Stencil:
            case PdfImageColour.Grey:
            {
                var table = _bytes[0];
                var plane = Planes![0].AsSpan(offset, width);
                for (int x = 0; x < width; x++)
                {
                    plane[x] = table[s[x]];
                }

                break;
            }

            case PdfImageColour.Rgb:
            {
                byte[] t0 = _bytes[0], t1 = _bytes[1], t2 = _bytes[2];
                Span<byte> red = Planes![0].AsSpan(offset, width), green = Planes[1].AsSpan(offset, width), blue = Planes[2].AsSpan(offset, width);
                for (int x = 0; x < width; x++)
                {
                    red[x] = t0[s[3 * x]];
                    green[x] = t1[s[3 * x + 1]];
                    blue[x] = t2[s[3 * x + 2]];
                }

                break;
            }

            case PdfImageColour.Cmyk:
            {
                byte[] t0 = _bytes[0], t1 = _bytes[1], t2 = _bytes[2], t3 = _bytes[3];
                Span<byte> red = Planes![0].AsSpan(offset, width), green = Planes[1].AsSpan(offset, width), blue = Planes[2].AsSpan(offset, width);
                Span<byte> rgb = stackalloc byte[3];
                for (int x = 0; x < width; x++)
                {
                    CmykToRgb(t0[s[4 * x]], t1[s[4 * x + 1]], t2[s[4 * x + 2]], t3[s[4 * x + 3]], rgb);
                    (red[x], green[x], blue[x]) = (rgb[0], rgb[1], rgb[2]);
                }

                break;
            }

            case PdfImageColour.Indexed:
            {
                var palette = _palette!;
                var index = _index;
                if (Planes!.Length == 1)
                {
                    var plane = Planes[0].AsSpan(offset, width);
                    for (int x = 0; x < width; x++)
                    {
                        plane[x] = palette[index[s[x]]];
                    }

                    break;
                }

                Span<byte> red = Planes[0].AsSpan(offset, width), green = Planes[1].AsSpan(offset, width), blue = Planes[2].AsSpan(offset, width);
                for (int x = 0; x < width; x++)
                {
                    int entry = 3 * index[s[x]];
                    (red[x], green[x], blue[x]) = (palette[entry], palette[entry + 1], palette[entry + 2]);
                }

                break;
            }

            case PdfImageColour.Lab:
            {
                double[] l = _lab[0], a = _lab[1], b = _lab[2];
                Span<byte> red = Planes![0].AsSpan(offset, width), green = Planes[1].AsSpan(offset, width), blue = Planes[2].AsSpan(offset, width);
                Span<byte> rgb = stackalloc byte[3];
                for (int x = 0; x < width; x++)
                {
                    LabToRgb(l[s[3 * x]], a[s[3 * x + 1]], b[s[3 * x + 2]], _white, rgb);
                    (red[x], green[x], blue[x]) = (rgb[0], rgb[1], rgb[2]);
                }

                break;
            }
        }
    }

    // A row of 16-bit samples into pixels, each sample mapped on its own: grey and RGB kept at 16 bits (values), the other
    // spaces through their 8-bit levels as at fewer bits.
    private void ConvertSixteen(ReadOnlySpan<ushort> s, int offset)
    {
        int width = _width;
        if (Sixteen)
        {
            var values = Values!;
            for (int c = 0; c < _components; c++)
            {
                var plane = values.AsSpan(c * _size + offset, width);
                for (int x = 0; x < width; x++)
                {
                    plane[x] = Wide(c, s[x * _components + c]);
                }
            }

            return;
        }

        var planes = Planes!;
        Span<byte> rgb = stackalloc byte[3];
        for (int x = 0; x < width; x++)
        {
            switch (_colour)
            {
                case PdfImageColour.Cmyk:
                    CmykToRgb(Level(0, s[4 * x]), Level(1, s[4 * x + 1]), Level(2, s[4 * x + 2]), Level(3, s[4 * x + 3]), rgb);
                    break;
                case PdfImageColour.Lab:
                    LabToRgb(Sample(_decode, 0, s[3 * x], _max), Sample(_decode, 1, s[3 * x + 1], _max), Sample(_decode, 2, s[3 * x + 2], _max), _white, rgb);
                    break;
                case PdfImageColour.Indexed:
                    int entry = Index(s[x]) * planes.Length;
                    _palette.AsSpan(entry, planes.Length).CopyTo(rgb);
                    break;
                default:
                    rgb[0] = Level(0, s[x]);   // not reached: grey and RGB of 16 bits are kept as values above
                    break;
            }

            for (int c = 0; c < planes.Length; c++)
            {
                planes[c][offset + x] = rgb[c];
            }
        }
    }

    /// <summary>Ink to light as the library's JPEG codec turns CMYK (Pillow's rule): (255 - K) - X (255 - K) / 255, rounded as its MULDIV255.</summary>
    public static void CmykToRgb(int cyan, int magenta, int yellow, int black, Span<byte> rgb)
    {
        int light = 255 - black;
        rgb[0] = (byte)Math.Clamp(light - MulDiv255(cyan, light), 0, 255);
        rgb[1] = (byte)Math.Clamp(light - MulDiv255(magenta, light), 0, 255);
        rgb[2] = (byte)Math.Clamp(light - MulDiv255(yellow, light), 0, 255);

        static int MulDiv255(int a, int b)
        {
            int t = a * b + 128;
            return ((t >> 8) + t) >> 8;
        }
    }

    /// <summary>L*a*b* under <paramref name="white"/> to XYZ, scaled to D65 per component, to sRGB (IEC 61966-2-1's matrix and curve), rounded half to even.</summary>
    public static void LabToRgb(double l, double a, double b, double[] white, Span<byte> rgb)
    {
        double fy = (l + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
        double x = white[0] * Inverse(fx) * D65[0] / white[0];
        double y = white[1] * Inverse(fy) * D65[1] / white[1];
        double z = white[2] * Inverse(fz) * D65[2] / white[2];
        for (int k = 0; k < 3; k++)
        {
            double v = SrgbMatrix[3 * k] * x + SrgbMatrix[3 * k + 1] * y + SrgbMatrix[3 * k + 2] * z;
            v = v <= 0.0031308 ? 12.92 * v : 1.055 * Math.Pow(Math.Max(v, 0), 1 / 2.4) - 0.055;
            rgb[k] = (byte)Math.Clamp(Math.Round(v * 255, MidpointRounding.ToEven), 0, 255);
        }

        static double Inverse(double t) => t > 6.0 / 29 ? Math.Pow(t, 3) : 3 * Math.Pow(6.0 / 29, 2) * (t - 4.0 / 29);
    }

    // A row's samples read by position: 8-bit bytes as they are, other depths unpacked.
    private interface ISamples
    {
        int this[int index] { get; }
    }

    private readonly ref struct EightBits(ReadOnlySpan<byte> bytes) : ISamples
    {
        private readonly ReadOnlySpan<byte> _bytes = bytes;

        public int this[int index] => _bytes[index];
    }

    private readonly ref struct Unpacked(ReadOnlySpan<ushort> values) : ISamples
    {
        private readonly ReadOnlySpan<ushort> _values = values;

        public int this[int index] => _values[index];
    }
}

/// <summary>
/// The images of one page as its content draws them (<c>Do</c> of an image XObject, at any depth of forms, and inline
/// images), each placed by the CTM in the content's order; an XObject drawn again is read once. What cannot be read is
/// counted in the page's issues: an image whose size is not given (left out), a colour space read later or damaged, an
/// encoding no registered codec reads.
/// </summary>
internal sealed class PdfPageImages(PdfFile file, int page, PdfIssues issues)
{
    private readonly Dictionary<int, PdfImage?> _byObject = [];

    private delegate PdfImage? ImageReader(out string? skipped);

    public List<DocumentImage> Images { get; } = [];

    /// <summary>The image XObject <paramref name="stream"/>, drawn by the name <paramref name="name"/>.</summary>
    public void Add(PdfStream stream, ReadOnlySpan<byte> name, PdfDictionary? resources, PdfMatrix ctm, int order, PdfPartStyle style)
    {
        if (!_byObject.TryGetValue(stream.Number, out var image))
        {
            string label = string.Create(CultureInfo.InvariantCulture, $"/{PdfText.Utf8OrLatin1(name)} (object {stream.Number})");
            image = Read(label, (out string? skipped) => PdfImage.FromStream(file, stream, resources, Where(label), stencil: false, out skipped));
            _byObject[stream.Number] = image;
        }

        if (image is not null)
        {
            Images.Add(image.Place(ctm, order, style));
        }
    }

    /// <summary>An inline image: the bytes of its entries (between BI and ID) and of its data (between ID and EI).</summary>
    public void AddInline(ReadOnlySpan<byte> entries, ReadOnlySpan<byte> data, PdfDictionary? resources, PdfMatrix ctm, int order, PdfPartStyle style)
    {
        string label = string.Create(CultureInfo.InvariantCulture, $"inline image {order}");
        PdfDictionary dictionary;
        try
        {
            dictionary = PdfImageInfo.Spelled(Entries(entries));
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            issues.Add(DocumentIssue.DamagedStream, $"page {page}: the {label}: its entries: {e.Message}");
            return;
        }

        byte[] copy = data.ToArray();   // the content's buffer is the reader's
        var image = Read(label, (out string? skipped) => PdfImage.FromInline(file, dictionary, copy, resources, Where(label), out skipped));
        if (image is not null)
        {
            Images.Add(image.Place(ctm, order, style));
        }
    }

    private string Where(string label) => string.Create(CultureInfo.InvariantCulture, $"{file.Name}: page {page}: the image {label}");

    // An image read, with what is wrong with it counted (a limit fails the page).
    private PdfImage? Read(string label, ImageReader read)
    {
        PdfImage? image;
        string? skipped;
        try
        {
            image = read(out skipped);
        }
        catch (InvalidDataException e) when (!PdfLimit.Is(e))
        {
            (image, skipped) = (null, e.Message);
        }

        if (image is null)
        {
            issues.Add(DocumentIssue.DamagedStream, $"page {page}: the image {label} is left out: {skipped}");
            return null;
        }

        var info = image.Info;
        if (info.Encoding != DocumentImage.Raw)
        {
            if (!ImageCodecs.Names.Contains(info.Encoding, StringComparer.OrdinalIgnoreCase))
            {
                issues.Add(DocumentIssue.ImageEncoding,
                    $"page {page}: {info.Encoding} images, which no registered image codec reads (register one named \"{info.Encoding}\" with ImageCodecs.Register; DocumentImage.Bytes gives their data as stored)");
            }
        }
        else if (info.Problem is { } problem)
        {
            if (info.NotSupported)
            {
                issues.Add(DocumentIssue.NotSupported, $"page {page}: images in the colour space {info.Space.Name}, which images are not painted in: DocumentImage.Decode refuses them");
            }
            else
            {
                issues.Add(DocumentIssue.DamagedStream, $"page {page}: the image {label} cannot be decoded: {problem}");
            }
        }

        return image;
    }

    // The entries between BI and ID: names and values, as a dictionary's body.
    private PdfDictionary Entries(ReadOnlySpan<byte> entries)
    {
        var parser = new PdfParser(entries, file.MaxDepth, complete: true);
        var keys = new List<PdfName>(8);
        var values = new List<PdfValue>(8);
        while (true)
        {
            parser.SkipWhitespaceToEnd();
            if (parser.Position >= entries.Length)
            {
                break;
            }

            if (parser.ReadValue().AsName is not { } key)
            {
                throw new InvalidDataException("an entry whose key is not a name");
            }

            keys.Add(key);
            values.Add(parser.ReadValue());
        }

        return new PdfDictionary([.. keys], [.. values]);
    }
}
