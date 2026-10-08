// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Abstraction.Devices;
using Idrak.Data.Abstractions;

namespace Idrak.Data;

/// <summary>
/// JPEG (ITU-T T.81, as JFIF, EXIF and Adobe files use it), decoded without dependencies: baseline and extended
/// sequential (SOF0, SOF1) and progressive (SOF2) Huffman-coded frames at 8 bits, one component (grey) or three
/// (YCbCr, or RGB when an Adobe APP14 segment says transform 0 or the component ids are 'R', 'G', 'B'), any whole
/// sampling ratio (4:4:4, 4:2:2, 4:2:0, 4:4:0, 4:1:1, ...), restart intervals, interleaved or not. APPn and comment
/// segments (EXIF, ICC profiles, XMP) are skipped: the EXIF orientation is not applied.
/// Not read (a <see cref="NotSupportedException"/> naming the variant): CMYK and YCCK (four components), 12- and
/// 16-bit samples, arithmetic coding, lossless and hierarchical frames.
/// <para>
/// The pixels follow libjpeg-turbo's defaults (and so Pillow's, which transformers uses): the "islow" integer inverse
/// DCT of jidctint.c, the "fancy" (triangle) upsampling of jdsample.c for 2:1 ratios (h2v1, h1v2, h2v2; other ratios
/// repeat samples), and the fixed-point YCbCr to RGB tables of jdcolor.c, computed the same way, so the files of the
/// tests decode to Pillow's bytes exactly. The integer IDCT is used rather than a float AAN one because it is what the
/// reference decoder runs: matching it bit for bit costs nothing in speed here, and a float IDCT differs by 1 in a few
/// percent of pixels. Interblock smoothing (libjpeg's, for a progressive file whose scans stop before the last bit) is
/// not done; a complete progressive file does not use it.
/// </para>
/// </summary>
internal sealed class JpegCodec : IImageCodec
{
    public string Name => "jpeg";

    public IReadOnlyCollection<string> Extensions { get; } = [".jpg", ".jpeg", ".jpe", ".jfif"];

    public ImageInfo? ReadInfo(ReadOnlySpan<byte> header)
    {
        if (header.Length < 4 || header[0] != 0xFF || header[1] != 0xD8 || header[2] != 0xFF)
        {
            return null;
        }

        int position = 2;
        while (true)
        {
            while (position < header.Length && header[position] != 0xFF)
            {
                position++;
            }

            while (position < header.Length && header[position] == 0xFF)
            {
                position++;
            }

            if (position + 3 > header.Length)
            {
                return new ImageInfo(0, 0, 3, Name);                                    // the frame header lies beyond these bytes
            }

            byte marker = header[position++];
            if (marker is 0x01 or 0xD8 or >= 0xD0 and <= 0xD7)
            {
                continue;
            }

            if (marker is 0xD9 or 0xDA)
            {
                return new ImageInfo(0, 0, 3, Name);                                    // no frame header before the image data
            }

            int length = header[position] << 8 | header[position + 1];
            if (JpegDecoder.IsFrame(marker) && position + 8 <= header.Length)
            {
                int h = header[position + 3] << 8 | header[position + 4], w = header[position + 5] << 8 | header[position + 6];
                return new ImageInfo(w, h, header[position + 7] == 1 ? 1 : 3, Name);
            }

            position += length;
        }
    }

    public ImageData Decode(ReadOnlySpan<byte> file)
    {
        var (planes, h, w) = JpegDecoder.Decode(file);
        var pixels = new float[checked(planes.Length * h * w)];
        HostParallel.For(planes.Length * h, 64, (first, last) =>
        {
            for (int row = first; row < last; row++)
            {
                var source = planes[row / h].AsSpan(row % h * w, w);
                var target = pixels.AsSpan(row * w, w);
                for (int x = 0; x < source.Length; x++)
                {
                    target[x] = source[x] * (1f / 255f);
                }
            }
        });
        return new ImageData(pixels, planes.Length, h, w);
    }
}

/// <summary>The JPEG decoder of <see cref="JpegCodec"/>: whole files to 8-bit planes (grey, or red, green and blue).</summary>
internal sealed class JpegDecoder
{
    // The natural (row-major) index of each zig-zag position, with 16 extra entries so a damaged run cannot index past
    // the block (libjpeg's jpeg_natural_order does the same).
    private static readonly byte[] Natural =
    [
        0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63,
        63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63, 63,
    ];

    // libjpeg's post-IDCT range limit (jdmaster.c prepare_range_limit_table, offset by CENTERJSAMPLE), indexed by a
    // descaled value masked to 10 bits: -128..127 map to 0..255, larger values to 255, smaller to 0 (and wild values
    // wrap as libjpeg's do).
    private static readonly byte[] IdctLimit = BuildIdctLimit();

    // jdcolor.c's YCbCr to RGB tables (16 fractional bits, rounded as libjpeg rounds them).
    private static readonly int[] CrToR = new int[256], CbToB = new int[256], CrToG = new int[256], CbToG = new int[256];

    private readonly byte[] data;
    private readonly int[][] quant = new int[4][];                                       // natural order
    private readonly Huffman?[] dcTables = new Huffman?[4], acTables = new Huffman?[4];
    private Component[] components = [];
    private int width, height, maxH, maxV, mcusX, mcusY, restartInterval;
    private bool progressive, sawJfif, sawAdobe;
    private int adobeTransform = -1;

    // The bit reader: bits are taken from the top of `bits`, `count` of them valid; bytes come from `position`. A marker
    // stops the reader there and feeds zero bits, as libjpeg does on damaged or short data.
    private ulong bits;
    private int count, position, eobRun;

    static JpegDecoder()
    {
        const int scale = 16, half = 1 << (scale - 1);
        for (int i = 0, x = -128; i < 256; i++, x++)
        {
            CrToR[i] = (Fix(1.40200) * x + half) >> scale;
            CbToB[i] = (Fix(1.77200) * x + half) >> scale;
            CrToG[i] = -Fix(0.71414) * x;
            CbToG[i] = -Fix(0.34414) * x + half;
        }

        static int Fix(double v) => (int)(v * (1 << scale) + 0.5);
    }

    private JpegDecoder(byte[] data) => this.data = data;

    /// <summary>Whether <paramref name="marker"/> starts a frame (SOF0 to SOF15, but DHT, JPG and DAC).</summary>
    public static bool IsFrame(byte marker) => marker is >= 0xC0 and <= 0xCF and not 0xC4 and not 0xC8 and not 0xCC;

    /// <summary>Decodes a whole file to one 8-bit plane (grey) or three (red, green, blue), each height x width.</summary>
    public static (byte[][] Planes, int Height, int Width) Decode(ReadOnlySpan<byte> file)
    {
        var decoder = new JpegDecoder(file.ToArray());
        decoder.ReadSegments();
        return (decoder.Output(), decoder.height, decoder.width);
    }

    // ---------------------------------------------------------------- segments

    private void ReadSegments()
    {
        if (data.Length < 4 || data[0] != 0xFF || data[1] != 0xD8)
        {
            throw new InvalidDataException("not a JPEG file (no start-of-image marker).");
        }

        int at = 2;
        bool scanned = false;
        while (true)
        {
            int marker = NextMarker(ref at);
            if (marker < 0 || marker == 0xD9)
            {
                break;                                                                  // the end, or a file cut short after its last scan
            }

            if (marker is 0x01 or 0xD8 or >= 0xD0 and <= 0xD7)
            {
                continue;
            }

            if (at + 2 > data.Length)
            {
                break;
            }

            int length = data[at] << 8 | data[at + 1];
            if (length < 2 || at + length > data.Length)
            {
                if (scanned)
                {
                    break;
                }

                throw new InvalidDataException($"JPEG segment 0x{marker:X2} runs past the end of the file.");
            }

            var segment = data.AsSpan(at + 2, length - 2);
            switch (marker)
            {
                case 0xC0 or 0xC1 or 0xC2:
                    Frame(segment, marker == 0xC2);
                    break;
                case 0xC3 or 0xC7 or 0xCB or 0xCF:
                    throw new NotSupportedException("lossless JPEG is not read.");
                case 0xC5 or 0xC6 or 0xCD or 0xCE:
                    throw new NotSupportedException("hierarchical JPEG is not read.");
                case 0xC9 or 0xCA or 0xCC:
                    throw new NotSupportedException("arithmetic-coded JPEG is not read; re-save it with Huffman coding.");
                case 0xC4:
                    HuffmanTables(segment);
                    break;
                case 0xDB:
                    QuantizationTables(segment);
                    break;
                case 0xDD:
                    restartInterval = segment.Length >= 2 ? segment[0] << 8 | segment[1] : 0;
                    break;
                case 0xE0:
                    sawJfif |= segment.StartsWith("JFIF\0"u8);
                    break;
                case 0xEE:
                    if (segment.Length >= 12 && segment.StartsWith("Adobe"u8))
                    {
                        sawAdobe = true;
                        adobeTransform = segment[11];
                    }

                    break;
                case 0xDA:
                    if (components.Length == 0)
                    {
                        throw new InvalidDataException("a JPEG scan before its frame header.");
                    }

                    at = Scan(segment, at + length);
                    scanned = true;
                    continue;
            }

            at += length;
        }

        if (!scanned)
        {
            throw new InvalidDataException("a JPEG file without image data.");
        }
    }

    // The next marker at or after `at`, skipping fill bytes and stray data; `at` is left after the marker byte. -1 at the end.
    private int NextMarker(ref int at)
    {
        while (at + 1 < data.Length)
        {
            if (data[at] != 0xFF || data[at + 1] is 0x00 or 0xFF)
            {
                at++;
                continue;
            }

            at += 2;
            return data[at - 1];
        }

        return -1;
    }

    private void Frame(ReadOnlySpan<byte> s, bool isProgressive)
    {
        if (components.Length > 0)
        {
            throw new NotSupportedException("a JPEG file with more than one frame is not read.");
        }

        if (s.Length < 6)
        {
            throw new InvalidDataException("a JPEG frame header too short.");
        }

        if (s[0] != 8)
        {
            throw new NotSupportedException($"{s[0]}-bit JPEG is not read (only 8-bit samples).");
        }

        height = s[1] << 8 | s[2];
        width = s[3] << 8 | s[4];
        int n = s[5];
        if (n is 4)
        {
            throw new NotSupportedException("CMYK and YCCK JPEG (four components) is not read; convert it to RGB.");
        }

        if (n is not (1 or 3))
        {
            throw new NotSupportedException($"JPEG with {n} components is not read (only grey and colour).");
        }

        if (height == 0)
        {
            throw new NotSupportedException("a JPEG whose height comes after the first scan (a DNL marker) is not read.");
        }

        if (width == 0 || s.Length < 6 + 3 * n)
        {
            throw new InvalidDataException($"a JPEG frame of {width} x {height} with {n} components.");
        }

        progressive = isProgressive;
        components = new Component[n];
        for (int i = 0; i < n; i++)
        {
            int h = s[7 + 3 * i] >> 4, v = s[7 + 3 * i] & 15;
            if (h is < 1 or > 4 || v is < 1 or > 4)
            {
                throw new InvalidDataException($"JPEG sampling factors {h} x {v}.");
            }

            components[i] = new Component { Id = s[6 + 3 * i], H = h, V = v, Table = s[8 + 3 * i] & 3 };
            maxH = Math.Max(maxH, h);
            maxV = Math.Max(maxV, v);
        }

        mcusX = (width + 8 * maxH - 1) / (8 * maxH);
        mcusY = (height + 8 * maxV - 1) / (8 * maxV);
        foreach (var c in components)
        {
            if (maxH % c.H != 0 || maxV % c.V != 0)
            {
                throw new NotSupportedException($"JPEG sampling {c.H} x {c.V} against {maxH} x {maxV} (not a whole ratio) is not read.");
            }

            c.Width = (width * c.H + maxH - 1) / maxH;
            c.Height = (height * c.V + maxV - 1) / maxV;
            c.BlocksPerLine = mcusX * c.H;
            c.BlockRows = mcusY * c.V;
            c.Coefficients = new short[checked(c.BlocksPerLine * c.BlockRows * 64)];
        }
    }

    private void HuffmanTables(ReadOnlySpan<byte> s)
    {
        while (s.Length >= 17)
        {
            int kind = s[0] >> 4, index = s[0] & 15;
            if (kind > 1 || index > 3)
            {
                throw new InvalidDataException($"a JPEG Huffman table of class {kind}, index {index}.");
            }

            var counts = s.Slice(1, 16);
            int total = 0;
            foreach (byte c in counts)
            {
                total += c;
            }

            if (total > 256 || s.Length < 17 + total)
            {
                throw new InvalidDataException("a JPEG Huffman table runs past its segment.");
            }

            var table = new Huffman(counts, s.Slice(17, total));
            (kind == 0 ? dcTables : acTables)[index] = table;
            s = s[(17 + total)..];
        }
    }

    private void QuantizationTables(ReadOnlySpan<byte> s)
    {
        while (s.Length >= 65)
        {
            int precision = s[0] >> 4, index = s[0] & 15;
            int size = precision == 0 ? 64 : 128;
            if (index > 3 || s.Length < 1 + size)
            {
                throw new InvalidDataException("a JPEG quantization table runs past its segment.");
            }

            var table = new int[64];
            for (int k = 0; k < 64; k++)
            {
                table[Natural[k]] = precision == 0 ? s[1 + k] : s[1 + 2 * k] << 8 | s[2 + 2 * k];
            }

            quant[index] = table;
            s = s[(1 + size)..];
        }
    }

    // ---------------------------------------------------------------- entropy-coded data

    // Decodes one scan whose header is `s` and whose data starts at `start`; returns where the data ends.
    private int Scan(ReadOnlySpan<byte> s, int start)
    {
        int n = s.Length > 0 ? s[0] : 0;
        if (n is < 1 or > 4 || s.Length < 4 + 2 * n)
        {
            throw new InvalidDataException("a JPEG scan header too short.");
        }

        var scan = new Component[n];
        for (int i = 0; i < n; i++)
        {
            int id = s[1 + 2 * i];
            scan[i] = Array.Find(components, c => c.Id == id) ?? throw new InvalidDataException($"a JPEG scan names component {id}, which the frame has not.");
            scan[i].DcTable = s[2 + 2 * i] >> 4 & 3;
            scan[i].AcTable = s[2 + 2 * i] & 3;
            scan[i].Predictor = 0;
        }

        int ss = s[1 + 2 * n], se = s[2 + 2 * n], ah = s[3 + 2 * n] >> 4, al = s[3 + 2 * n] & 15;
        if (progressive && (se > 63 || ss > se || ss == 0 && se != 0 || ss > 0 && n != 1 || al > 13))
        {
            throw new InvalidDataException($"a progressive JPEG scan of coefficients {ss} to {se}, bits {ah}/{al}.");
        }

        var mode = !progressive ? Mode.Sequential
            : ss == 0 ? ah == 0 ? Mode.DcFirst : Mode.DcRefine
            : ah == 0 ? Mode.AcFirst : Mode.AcRefine;
        foreach (var c in scan)
        {
            bool needsDc = mode is Mode.Sequential or Mode.DcFirst, needsAc = mode is Mode.Sequential or Mode.AcFirst or Mode.AcRefine;
            if (needsDc && dcTables[c.DcTable] is null || needsAc && acTables[c.AcTable] is null)
            {
                throw new InvalidDataException("a JPEG scan uses a Huffman table that was not defined.");
            }

            if (quant[c.Table] is null)
            {
                throw new InvalidDataException("a JPEG component uses a quantization table that was not defined.");
            }
        }

        position = start;
        bits = 0;
        count = 0;
        eobRun = 0;
        int done = 0;
        if (n == 1)
        {
            // One component: blocks in raster order over the component's own size, one block per MCU.
            var c = scan[0];
            int across = (c.Width + 7) / 8, down = (c.Height + 7) / 8;
            for (int by = 0; by < down; by++)
            {
                for (int bx = 0; bx < across; bx++)
                {
                    Block(c, (by * c.BlocksPerLine + bx) * 64, mode, ss, se, ah, al);
                    if (restartInterval > 0 && ++done % restartInterval == 0 && done < across * down)
                    {
                        Restart(scan);
                    }
                }
            }
        }
        else
        {
            for (int my = 0; my < mcusY; my++)
            {
                for (int mx = 0; mx < mcusX; mx++)
                {
                    foreach (var c in scan)
                    {
                        for (int v = 0; v < c.V; v++)
                        {
                            for (int h = 0; h < c.H; h++)
                            {
                                Block(c, ((my * c.V + v) * c.BlocksPerLine + mx * c.H + h) * 64, mode, ss, se, ah, al);
                            }
                        }
                    }

                    if (restartInterval > 0 && ++done % restartInterval == 0 && done < mcusX * mcusY)
                    {
                        Restart(scan);
                    }
                }
            }
        }

        return position;
    }

    private enum Mode
    {
        Sequential,
        DcFirst,
        DcRefine,
        AcFirst,
        AcRefine,
    }

    private void Block(Component c, int offset, Mode mode, int ss, int se, int ah, int al)
    {
        var block = c.Coefficients.AsSpan(offset, 64);
        switch (mode)
        {
            case Mode.Sequential:
            {
                int t = dcTables[c.DcTable]!.Decode(this);
                c.Predictor += t == 0 ? 0 : Extend(Receive(t), t);
                block[0] = (short)c.Predictor;
                var ac = acTables[c.AcTable]!;
                for (int k = 1; k < 64;)
                {
                    int rs = ac.Decode(this), r = rs >> 4, size = rs & 15;
                    if (size == 0)
                    {
                        if (r != 15)
                        {
                            break;
                        }

                        k += 16;
                        continue;
                    }

                    k += r;
                    block[Natural[k]] = (short)Extend(Receive(size), size);
                    k++;
                }

                break;
            }

            case Mode.DcFirst:
            {
                int t = dcTables[c.DcTable]!.Decode(this);
                c.Predictor += t == 0 ? 0 : Extend(Receive(t), t);
                block[0] = (short)(c.Predictor << al);
                break;
            }

            case Mode.DcRefine:
                if (Receive(1) != 0)
                {
                    block[0] |= (short)(1 << al);
                }

                break;
            case Mode.AcFirst:
            {
                if (eobRun > 0)
                {
                    eobRun--;
                    break;
                }

                var ac = acTables[c.AcTable]!;
                for (int k = ss; k <= se;)
                {
                    int rs = ac.Decode(this), r = rs >> 4, size = rs & 15;
                    if (size == 0)
                    {
                        if (r < 15)
                        {
                            eobRun = (1 << r) - 1 + (r > 0 ? Receive(r) : 0);
                            break;
                        }

                        k += 16;
                        continue;
                    }

                    k += r;
                    block[Natural[k]] = (short)(Extend(Receive(size), size) * (1 << al));
                    k++;
                }

                break;
            }

            case Mode.AcRefine:
                AcRefine(c, block, ss, se, al);
                break;
        }
    }

    // A refinement scan of AC coefficients (T.81 G.1.2.3, as jdphuff.c decode_mcu_AC_refine): one more bit of every
    // coefficient already non-zero, and new coefficients of magnitude 1 << al.
    private void AcRefine(Component c, Span<short> block, int ss, int se, int al)
    {
        int p1 = 1 << al, m1 = -1 << al;
        int k = ss;
        if (eobRun == 0)
        {
            var ac = acTables[c.AcTable]!;
            for (; k <= se; k++)
            {
                int rs = ac.Decode(this), r = rs >> 4, size = rs & 15, value = 0;
                if (size != 0)
                {
                    value = Receive(1) != 0 ? p1 : m1;                                  // size is always 1 here
                }
                else if (r != 15)
                {
                    eobRun = (1 << r) + (r > 0 ? Receive(r) : 0);
                    break;
                }

                do
                {
                    ref short coefficient = ref block[Natural[k]];
                    if (coefficient != 0)
                    {
                        Refine(ref coefficient, p1, m1);
                    }
                    else if (--r < 0)
                    {
                        break;                                                          // the zero the new value goes to
                    }

                    k++;
                }
                while (k <= se);

                if (value != 0)
                {
                    block[Natural[k]] = (short)value;
                }
            }
        }

        if (eobRun > 0)
        {
            for (; k <= se; k++)
            {
                ref short coefficient = ref block[Natural[k]];
                if (coefficient != 0)
                {
                    Refine(ref coefficient, p1, m1);
                }
            }

            eobRun--;
        }
    }

    private void Refine(ref short coefficient, int p1, int m1)
    {
        if (Receive(1) != 0 && (coefficient & p1) == 0)
        {
            coefficient += (short)(coefficient >= 0 ? p1 : m1);
        }
    }

    // After `restartInterval` MCUs: the bits left are padding; the next marker should be RSTn; predictions restart.
    private void Restart(Component[] scan)
    {
        bits = 0;
        count = 0;
        eobRun = 0;
        foreach (var c in scan)
        {
            c.Predictor = 0;
        }

        int at = position;
        int marker = NextMarker(ref at);
        if (marker is >= 0xD0 and <= 0xD7)
        {
            position = at;
        }
    }

    // Keeps at least 57 bits in the buffer (zeros past a marker or the end).
    private void Fill()
    {
        while (count <= 56)
        {
            int b = 0;
            if (position < data.Length)
            {
                b = data[position];
                if (b == 0xFF)
                {
                    int next = position + 1 < data.Length ? data[position + 1] : 0xD9;
                    if (next == 0x00)
                    {
                        position += 2;
                    }
                    else
                    {
                        b = 0;                                                          // a marker: stay before it
                    }
                }
                else
                {
                    position++;
                }
            }

            bits |= (ulong)b << (56 - count);
            count += 8;
        }
    }

    private int Receive(int n)
    {
        if (count < n)
        {
            Fill();
        }

        int value = (int)(bits >> (64 - n));
        bits <<= n;
        count -= n;
        return value;
    }

    // The value of `size` received bits (T.81 F.2.2.1's EXTEND).
    private static int Extend(int value, int size) => value < 1 << (size - 1) ? value - (1 << size) + 1 : value;

    private sealed class Component
    {
        public int Id, H, V, Table, DcTable, AcTable, Predictor;
        public int Width, Height, BlocksPerLine, BlockRows;
        public short[] Coefficients = [];
    }

    // A Huffman table: codes of up to 9 bits from one lookup, longer ones from the canonical code limits.
    private sealed class Huffman
    {
        private const int LookBits = 9;
        private readonly ushort[] fast = new ushort[1 << LookBits];                     // (length << 8 | symbol), 0 when longer
        private readonly int[] maxCode = new int[17], offset = new int[17];
        private readonly byte[] symbols;

        public Huffman(ReadOnlySpan<byte> counts, ReadOnlySpan<byte> values)
        {
            symbols = values.ToArray();
            int code = 0, k = 0;
            for (int length = 1; length <= 16; length++)
            {
                offset[length] = k - code;                                              // symbol index = code + offset
                for (int i = 0; i < counts[length - 1]; i++, k++, code++)
                {
                    if (length <= LookBits)
                    {
                        int shift = LookBits - length;
                        for (int fill = 0; fill < 1 << shift; fill++)
                        {
                            fast[code << shift | fill] = (ushort)(length << 8 | symbols[k]);
                        }
                    }
                }

                maxCode[length] = counts[length - 1] > 0 ? code - 1 : -1;
                code <<= 1;
            }

        }

        public int Decode(JpegDecoder reader)
        {
            if (reader.count < 16)
            {
                reader.Fill();
            }

            int entry = fast[(int)(reader.bits >> (64 - LookBits))];
            if (entry != 0)
            {
                int length = entry >> 8;
                reader.bits <<= length;
                reader.count -= length;
                return entry & 0xFF;
            }

            int look = (int)(reader.bits >> 48);
            for (int length = LookBits + 1; length <= 16; length++)
            {
                int code = look >> (16 - length);
                if (code <= maxCode[length])
                {
                    reader.bits <<= length;
                    reader.count -= length;
                    int index = code + offset[length];
                    return (uint)index < (uint)symbols.Length ? symbols[index] : 0;
                }
            }

            reader.bits <<= 16;                                                         // not a code (damaged data): a zero, as libjpeg gives
            reader.count -= 16;
            return 0;
        }
    }

    // ---------------------------------------------------------------- pixels

    private byte[][] Output()
    {
        // Each component's samples at its own resolution (whole blocks; the rows and columns past its size unused).
        var samples = new byte[components.Length][];
        for (int i = 0; i < components.Length; i++)
        {
            var c = components[i];
            int stride = c.BlocksPerLine * 8, rows = (c.Height + 7) / 8, across = (c.Width + 7) / 8;
            var plane = samples[i] = new byte[stride * rows * 8];
            var table = quant[c.Table] ?? throw new InvalidDataException("a JPEG component without a quantization table.");
            HostParallel.For(rows, 1, (first, last) =>
            {
                Span<int> workspace = stackalloc int[64];
                for (int by = first; by < last; by++)
                {
                    for (int bx = 0; bx < across; bx++)
                    {
                        Idct(c.Coefficients.AsSpan((by * c.BlocksPerLine + bx) * 64, 64), table, plane.AsSpan(by * 8 * stride + bx * 8), stride, workspace);
                    }
                }
            });
        }

        // Every component at full size, then to RGB.
        var full = new byte[components.Length][];
        for (int i = 0; i < components.Length; i++)
        {
            full[i] = Upsample(components[i], samples[i]);
        }

        if (components.Length == 1)
        {
            return full;
        }

        bool rgb = sawJfif ? false
            : sawAdobe ? adobeTransform == 0
            : components[0].Id == 'R' && components[1].Id == 'G' && components[2].Id == 'B';
        if (!rgb)
        {
            YccToRgb(full[0], full[1], full[2]);
        }

        return full;
    }

    private void YccToRgb(byte[] y, byte[] cb, byte[] cr)
    {
        HostParallel.For(height, 16, (first, last) =>
        {
            for (int i = first * width; i < last * width; i++)
            {
                int luma = y[i], blue = cb[i], red = cr[i];
                y[i] = Clamp(luma + CrToR[red]);
                cb[i] = Clamp(luma + ((CbToG[blue] + CrToG[red]) >> 16));
                cr[i] = Clamp(luma + CbToB[blue]);
            }
        });

        static byte Clamp(int v) => (byte)Math.Clamp(v, 0, 255);
    }

    // A component's samples at the image's size: as they are (the same sampling as the largest), with libjpeg's fancy
    // (triangle) upsampling for 2:1 in either direction or both, and by repeating samples for other ratios.
    private byte[] Upsample(Component c, byte[] plane)
    {
        int stride = c.BlocksPerLine * 8, cw = c.Width, ch = c.Height, w = width, h = height;
        int rx = maxH / c.H, ry = maxV / c.V;
        var output = new byte[checked(w * h)];
        if (rx == 1 && ry == 1)
        {
            for (int y = 0; y < h; y++)
            {
                plane.AsSpan(y * stride, w).CopyTo(output.AsSpan(y * w));
            }

            return output;
        }

        bool fancy = rx == 2 && ry is 1 or 2 || rx == 1 && ry == 2;
        if (!fancy || cw < 2 && rx == 2)
        {
            HostParallel.For(h, 16, (first, last) =>
            {
                for (int y = first; y < last; y++)
                {
                    var source = plane.AsSpan(Math.Min(y / ry, ch - 1) * stride);
                    var target = output.AsSpan(y * w, w);
                    for (int x = 0; x < w; x++)
                    {
                        target[x] = source[Math.Min(x / rx, cw - 1)];
                    }
                }
            });
            return output;
        }

        HostParallel.For(ch, 8, (first, last) =>
        {
            var row = ArrayPool<int>.Shared.Rent(cw);
            var wide = ArrayPool<byte>.Shared.Rent(2 * cw);
            try
            {
                for (int r = first; r < last; r++)
                {
                    var current = plane.AsSpan(r * stride, cw);
                    for (int v = 0; v < ry; v++)
                    {
                        int y = r * ry + v;
                        if (y >= h)
                        {
                            break;
                        }

                        var target = output.AsSpan(y * w, w);
                        if (ry == 1)
                        {
                            H2V1(current, target, wide);
                            continue;
                        }

                        // Vertically: three parts of this row and one of the row above (for the upper output row) or
                        // below (for the lower), the edges repeated.
                        var other = plane.AsSpan(Math.Clamp(v == 0 ? r - 1 : r + 1, 0, ch - 1) * stride, cw);
                        for (int x = 0; x < cw; x++)
                        {
                            row[x] = current[x] * 3 + other[x];
                        }

                        if (rx == 1)
                        {
                            int bias = v == 0 ? 1 : 2;                                  // jdsample.c h1v2_fancy_upsample
                            for (int x = 0; x < w; x++)
                            {
                                target[x] = (byte)((row[x] + bias) >> 2);
                            }
                        }
                        else
                        {
                            H2V2(row.AsSpan(0, cw), target, wide);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<int>.Shared.Return(row);
                ArrayPool<byte>.Shared.Return(wide);
            }
        });
        return output;
    }

    // jdsample.c h2v1_fancy_upsample: each output sample is 3/4 of the nearer input sample and 1/4 of the further one.
    private static void H2V1(ReadOnlySpan<byte> input, Span<byte> target, byte[] wide)
    {
        int n = input.Length;
        wide[0] = input[0];
        wide[1] = (byte)((input[0] * 3 + input[1] + 2) >> 2);
        for (int i = 1; i < n - 1; i++)
        {
            int value = input[i] * 3;
            wide[2 * i] = (byte)((value + input[i - 1] + 1) >> 2);
            wide[2 * i + 1] = (byte)((value + input[i + 1] + 2) >> 2);
        }

        wide[2 * n - 2] = (byte)((input[n - 1] * 3 + input[n - 2] + 1) >> 2);
        wide[2 * n - 1] = input[n - 1];
        wide.AsSpan(0, target.Length).CopyTo(target);
    }

    // jdsample.c h2v2_fancy_upsample, horizontally, on column sums already 3/4 this row and 1/4 the other.
    private static void H2V2(ReadOnlySpan<int> sums, Span<byte> target, byte[] wide)
    {
        int n = sums.Length;
        wide[0] = (byte)((sums[0] * 4 + 8) >> 4);
        wide[1] = (byte)((sums[0] * 3 + sums[1] + 7) >> 4);
        for (int i = 1; i < n - 1; i++)
        {
            int value = sums[i] * 3;
            wide[2 * i] = (byte)((value + sums[i - 1] + 8) >> 4);
            wide[2 * i + 1] = (byte)((value + sums[i + 1] + 7) >> 4);
        }

        wide[2 * n - 2] = (byte)((sums[n - 1] * 3 + sums[n - 2] + 8) >> 4);
        wide[2 * n - 1] = (byte)((sums[n - 1] * 4 + 7) >> 4);
        wide.AsSpan(0, target.Length).CopyTo(target);
    }

    private static byte[] BuildIdctLimit()
    {
        var table = new byte[1024];
        for (int i = 0; i < 1024; i++)
        {
            table[i] = (byte)(i < 128 ? i + 128 : i < 512 ? 255 : i < 896 ? 0 : i - 896);
        }

        return table;
    }

    // jidctint.c jpeg_idct_islow: the "slow but accurate" integer inverse DCT (13 fractional bits for the constants,
    // 2 extra bits between the passes), dequantizing as it reads. Columns first into `workspace`, then rows to samples.
    private static void Idct(ReadOnlySpan<short> input, int[] q, Span<byte> output, int stride, Span<int> ws)
    {
        const int constBits = 13, pass1Bits = 2;
        const int fix0298631336 = 2446, fix0390180644 = 3196, fix0541196100 = 4433, fix0765366865 = 6270, fix0899976223 = 7373,
            fix1175875602 = 9633, fix1501321110 = 12299, fix1847759065 = 15137, fix1961570560 = 16069, fix2053119869 = 16819,
            fix2562915447 = 20995, fix3072711026 = 25172;
        var limit = IdctLimit;
        for (int col = 0; col < 8; col++)
        {
            if (input[8 + col] == 0 && input[16 + col] == 0 && input[24 + col] == 0 && input[32 + col] == 0
                && input[40 + col] == 0 && input[48 + col] == 0 && input[56 + col] == 0)
            {
                int dc = input[col] * q[col] << pass1Bits;
                for (int r = 0; r < 8; r++)
                {
                    ws[r * 8 + col] = dc;
                }

                continue;
            }

            int z2 = input[16 + col] * q[16 + col], z3 = input[48 + col] * q[48 + col];
            int z1 = (z2 + z3) * fix0541196100;
            int tmp2 = z1 + z3 * -fix1847759065, tmp3 = z1 + z2 * fix0765366865;
            z2 = input[col] * q[col];
            z3 = input[32 + col] * q[32 + col];
            int tmp0 = (z2 + z3) << constBits, tmp1 = (z2 - z3) << constBits;
            int tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

            tmp0 = input[56 + col] * q[56 + col];
            tmp1 = input[40 + col] * q[40 + col];
            tmp2 = input[24 + col] * q[24 + col];
            tmp3 = input[8 + col] * q[8 + col];
            z1 = tmp0 + tmp3;
            z2 = tmp1 + tmp2;
            z3 = tmp0 + tmp2;
            int z4 = tmp1 + tmp3, z5 = (z3 + z4) * fix1175875602;
            tmp0 *= fix0298631336;
            tmp1 *= fix2053119869;
            tmp2 *= fix3072711026;
            tmp3 *= fix1501321110;
            z1 *= -fix0899976223;
            z2 *= -fix2562915447;
            z3 = z3 * -fix1961570560 + z5;
            z4 = z4 * -fix0390180644 + z5;
            tmp0 += z1 + z3;
            tmp1 += z2 + z4;
            tmp2 += z2 + z3;
            tmp3 += z1 + z4;

            const int shift1 = constBits - pass1Bits, round1 = 1 << (shift1 - 1);
            ws[col] = (tmp10 + tmp3 + round1) >> shift1;
            ws[56 + col] = (tmp10 - tmp3 + round1) >> shift1;
            ws[8 + col] = (tmp11 + tmp2 + round1) >> shift1;
            ws[48 + col] = (tmp11 - tmp2 + round1) >> shift1;
            ws[16 + col] = (tmp12 + tmp1 + round1) >> shift1;
            ws[40 + col] = (tmp12 - tmp1 + round1) >> shift1;
            ws[24 + col] = (tmp13 + tmp0 + round1) >> shift1;
            ws[32 + col] = (tmp13 - tmp0 + round1) >> shift1;
        }

        const int shift2 = constBits + pass1Bits + 3, round2 = 1 << (shift2 - 1);
        for (int row = 0; row < 8; row++)
        {
            var w = ws.Slice(row * 8, 8);
            var o = output.Slice(row * stride, 8);
            if (w[1] == 0 && w[2] == 0 && w[3] == 0 && w[4] == 0 && w[5] == 0 && w[6] == 0 && w[7] == 0)
            {
                o.Fill(limit[((w[0] + (1 << (pass1Bits + 2))) >> (pass1Bits + 3)) & 1023]);
                continue;
            }

            int z2 = w[2], z3 = w[6];
            int z1 = (z2 + z3) * fix0541196100;
            int tmp2 = z1 + z3 * -fix1847759065, tmp3 = z1 + z2 * fix0765366865;
            int tmp0 = (w[0] + w[4]) << constBits, tmp1 = (w[0] - w[4]) << constBits;
            int tmp10 = tmp0 + tmp3, tmp13 = tmp0 - tmp3, tmp11 = tmp1 + tmp2, tmp12 = tmp1 - tmp2;

            tmp0 = w[7];
            tmp1 = w[5];
            tmp2 = w[3];
            tmp3 = w[1];
            z1 = tmp0 + tmp3;
            z2 = tmp1 + tmp2;
            z3 = tmp0 + tmp2;
            int z4 = tmp1 + tmp3, z5 = (z3 + z4) * fix1175875602;
            tmp0 *= fix0298631336;
            tmp1 *= fix2053119869;
            tmp2 *= fix3072711026;
            tmp3 *= fix1501321110;
            z1 *= -fix0899976223;
            z2 *= -fix2562915447;
            z3 = z3 * -fix1961570560 + z5;
            z4 = z4 * -fix0390180644 + z5;
            tmp0 += z1 + z3;
            tmp1 += z2 + z4;
            tmp2 += z2 + z3;
            tmp3 += z1 + z4;

            o[0] = limit[((tmp10 + tmp3 + round2) >> shift2) & 1023];
            o[7] = limit[((tmp10 - tmp3 + round2) >> shift2) & 1023];
            o[1] = limit[((tmp11 + tmp2 + round2) >> shift2) & 1023];
            o[6] = limit[((tmp11 - tmp2 + round2) >> shift2) & 1023];
            o[2] = limit[((tmp12 + tmp1 + round2) >> shift2) & 1023];
            o[5] = limit[((tmp12 - tmp1 + round2) >> shift2) & 1023];
            o[3] = limit[((tmp13 + tmp0 + round2) >> shift2) & 1023];
            o[4] = limit[((tmp13 - tmp0 + round2) >> shift2) & 1023];
        }
    }
}
