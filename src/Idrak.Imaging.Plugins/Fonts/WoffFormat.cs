// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Runtime.InteropServices;

namespace Idrak.Imaging.Plugins.Fonts;

/// <summary>
/// "woff": WOFF 1.0 fonts (.woff; the W3C Recommendation of 2012), an sfnt font (TrueType or OpenType) whose tables are
/// each zlib-compressed or stored as they are. <see cref="Read"/> unpacks the font inside (<see cref="Decode"/>) and
/// reads it as the WOFF header's flavor says: TrueType (0x00010000 or true) and OpenType with CFF outlines (OTTO) by the
/// library's OpenType reader (<see cref="OpenTypeFormat"/>, reached directly), another flavor refused; so a face read
/// from a WOFF is that font's face: the same glyphs, outlines, advances, characters and tables, and its
/// <see cref="IFontFace.Format"/> names the reader of the font inside ("opentype"). Inflating goes through
/// <see cref="ZLibStream"/> (the base library's zlib, on every platform).
/// <para>
/// A WOFF file is untrusted input. Every offset and length is checked before anything is inflated: the header's file
/// length, the table directory in tag order, each table on a 4-byte boundary, inside the file and overlapping no other
/// block, its stored bytes no more than its own and no fewer than zlib can inflate it from (1,032 to 1 at most), and the
/// font inside within <see cref="MaxFontBytes"/> and the size the header gives it. A table inflates to exactly the
/// length it declares: inflating stops one byte past it, so a decompression bomb costs no more than the font it claims to
/// hold. A damaged or hostile file is an <see cref="InvalidDataException"/> naming the file and the cause. Metadata and
/// private data blocks are checked for place and left out (a face has no use for them).
/// </para>
/// </summary>
public sealed class WoffFormat : IFontFormat
{
    /// <summary>The format's name in <c>FontFormats</c>: "woff".</summary>
    public const string FormatName = "woff";

    /// <summary>The variable that sets <see cref="MaxFontBytes"/> (bytes, or with k, m or g): "IDRAK_WOFF_MAX_BYTES".</summary>
    public const string MaxFontBytesVariable = "IDRAK_WOFF_MAX_BYTES";

    // deflate gives at most 1,032 bytes for one (a 258-byte match coded in two bits): a table that claims more from the
    // bytes it stores cannot be, and is refused before anything is inflated.
    private const long MostInflated = 1032;

    // The flavors of the fonts a WOFF 1.0 file holds: TrueType (0x00010000, or Apple's "true") and OpenType with CFF
    // outlines ("OTTO").
    private const uint TrueTypeFlavor = 0x00010000, AppleTrueTypeFlavor = 0x74727565, CffFlavor = 0x4F54544F;

    // The reader of the font inside, whatever its flavor: the library's OpenType reader, reached directly.
    private static readonly OpenTypeFormat OpenType = new();

    private readonly long _maxFontBytes = WoffSettings.Bytes(MaxFontBytesVariable) ?? 0;

    /// <inheritdoc/>
    public string Name => FormatName;

    /// <inheritdoc/>
    public IReadOnlyCollection<string> Extensions { get; } = [".woff"];

    /// <summary>
    /// The most bytes the font inside a WOFF may take once unpacked (its header, directory and tables): a quarter of the
    /// memory free when a file is read (within <see cref="ComputeResources.CpuMemoryLimit"/>), at most the largest array
    /// .NET makes, unless set here or by <see cref="MaxFontBytesVariable"/> (read when the format is made). A file whose
    /// tables claim more is refused before anything is inflated.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not positive.</exception>
    public long MaxFontBytes
    {
        get => _maxFontBytes > 0 ? _maxFontBytes : WoffSettings.MeasuredLimit();
        init => _maxFontBytes = value > 0 ? value : throw new ArgumentOutOfRangeException(nameof(MaxFontBytes), value, $"A WOFF font limit is a number of bytes, more than 0 ({MaxFontBytesVariable}).");
    }

    /// <inheritdoc/>
    public bool Accepts(ReadOnlySpan<byte> header) => header.Length >= 4 && BinaryPrimitives.ReadUInt32BigEndian(header) == WoffSettings.Woff;

    /// <inheritdoc/>
    /// <exception cref="InvalidDataException">The file is damaged or hostile (the message says how), or its flavor is not a
    /// font WOFF 1.0 holds (TrueType, 0x00010000 or true; OpenType with CFF outlines, OTTO).</exception>
    /// <exception cref="NotSupportedException">The font inside is one the OpenType reader does not read (CFF2 outlines, no
    /// Unicode character map: the message says which).</exception>
    public IReadOnlyList<IFontFace> Read(ReadOnlyMemory<byte> file, string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        byte[] sfnt = Decode(file, name);
        // The header's flavor, the font inside's sfnt version, says what it is: TrueType outlines (0x00010000, or "true")
        // and CFF outlines ("OTTO") are both read by the library's OpenType reader, reached directly. The faces keep the
        // unpacked font, not copied.
        uint flavor = BinaryPrimitives.ReadUInt32BigEndian(sfnt);
        return flavor is TrueTypeFlavor or AppleTrueTypeFlavor or CffFlavor
            ? OpenType.Read(sfnt, name)
            : throw Damaged(name, $"the WOFF's flavor is {WoffSettings.Tag(flavor)}: not a font WOFF 1.0 holds (TrueType, 0x00010000 or true; OpenType with CFF outlines, OTTO)");
    }

    /// <summary>
    /// The font inside a WOFF file, unpacked: the sfnt its encoder held, its tables listed by tag and laid out in the order
    /// the WOFF stores them, each padded to 4 bytes. A WOFF that <see cref="WoffWriter"/> wrote unpacks to exactly the bytes
    /// its sfnt writer wrote.
    /// </summary>
    /// <param name="file">The WOFF file's bytes.</param>
    /// <param name="name">A name for messages (the file's).</param>
    /// <exception cref="InvalidDataException">The file is damaged or hostile, or the font inside would pass <see cref="MaxFontBytes"/>; the message names the cause.</exception>
    public byte[] Decode(ReadOnlyMemory<byte> file, string name = "font")
    {
        ArgumentNullException.ThrowIfNull(name);
        var woff = file.Span;
        if (woff.Length < WoffSettings.HeaderSize)
        {
            throw Damaged(name, $"the file is cut: {woff.Length:N0} bytes, fewer than a WOFF header's {WoffSettings.HeaderSize}");
        }

        uint signature = U32(woff, 0), flavor = U32(woff, 4), length = U32(woff, 8), declaredSfnt = U32(woff, 16);
        int count = U16(woff, 12), reserved = U16(woff, 14);
        if (signature != WoffSettings.Woff)
        {
            throw Damaged(name, $"not a WOFF file: it starts with {WoffSettings.Tag(signature)}, not wOFF");
        }

        if (length != woff.Length)
        {
            throw length > woff.Length
                ? Damaged(name, $"the file is cut: its WOFF header gives {length:N0} bytes, the file holds {woff.Length:N0}")
                : Damaged(name, $"{woff.Length - length:N0} bytes follow the {length:N0} its WOFF header gives: a damaged file");
        }

        if (reserved != 0)
        {
            throw Damaged(name, $"the WOFF header's reserved field is {reserved}, not 0: a damaged file");
        }

        if (count == 0)
        {
            throw Damaged(name, $"a WOFF file of no tables: a damaged file");
        }

        if (flavor is WoffSettings.Woff or WoffSettings.Woff2 or WoffSettings.Collection)
        {
            throw Damaged(name, $"the WOFF's flavor is {WoffSettings.Tag(flavor)}: a WOFF 1.0 file holds one sfnt font (0x00010000, OTTO or true), never a collection or another WOFF");
        }

        long directoryEnd = WoffSettings.HeaderSize + (long)WoffSettings.EntrySize * count;
        if (directoryEnd > woff.Length)
        {
            throw Damaged(name, $"the file is cut: the directory of its {count} tables ends at {directoryEnd:N0}, past its {woff.Length:N0} bytes");
        }

        // The directory, every entry checked before anything is allocated or inflated.
        var tables = new Table[count];
        long sfntSize = WoffSettings.SfntHeaderSize + (long)WoffSettings.SfntEntrySize * count;
        for (int i = 0; i < count; i++)
        {
            var entry = woff.Slice(WoffSettings.HeaderSize + WoffSettings.EntrySize * i, WoffSettings.EntrySize);
            var table = new Table(U32(entry, 0), U32(entry, 4), U32(entry, 8), U32(entry, 12), U32(entry, 16));
            if (i > 0 && table.Tag <= tables[i - 1].Tag)
            {
                throw table.Tag == tables[i - 1].Tag
                    ? Damaged(name, $"the WOFF table '{WoffSettings.Tag(table.Tag)}' is listed twice: a damaged file")
                    : Damaged(name, $"the WOFF table directory is not in tag order ('{WoffSettings.Tag(table.Tag)}' after '{WoffSettings.Tag(tables[i - 1].Tag)}'): a damaged file");
            }

            if (table.Offset % 4 != 0)
            {
                throw Damaged(name, $"the WOFF table '{WoffSettings.Tag(table.Tag)}' starts at {table.Offset:N0}, not on a 4-byte boundary: a damaged file");
            }

            if (table.Offset < directoryEnd)
            {
                throw Damaged(name, $"the WOFF table '{WoffSettings.Tag(table.Tag)}' starts at {table.Offset:N0}, inside the header and directory (they end at {directoryEnd:N0}): a damaged file");
            }

            if ((long)table.Offset + table.Stored > woff.Length)
            {
                throw Damaged(name, $"the WOFF table '{WoffSettings.Tag(table.Tag)}' ({table.Stored:N0} bytes at {table.Offset:N0}) runs past the end of the file ({woff.Length:N0} bytes): the file is cut or damaged");
            }

            if (table.Stored > table.Length)
            {
                throw Damaged(name, $"the WOFF table '{WoffSettings.Tag(table.Tag)}' stores {table.Stored:N0} bytes for its {table.Length:N0}: more than the table itself, a damaged file");
            }

            if (table.Stored < table.Length && table.Length > MostInflated * table.Stored)
            {
                throw Damaged(name, $"the WOFF table '{WoffSettings.Tag(table.Tag)}' claims {table.Length:N0} bytes from {table.Stored:N0} compressed, more than zlib can give ({MostInflated:N0} to 1): a damaged or hostile file");
            }

            sfntSize += WoffSettings.Padded(table.Length);
            tables[i] = table;
        }

        long limit = MaxFontBytes;
        if (sfntSize > limit)
        {
            throw Damaged(name, $"the font inside the WOFF would take {sfntSize:N0} bytes, more than the limit of {limit:N0} (WoffFormat.MaxFontBytes, {MaxFontBytesVariable}): refused before anything was inflated");
        }

        if (declaredSfnt != sfntSize)
        {
            throw Damaged(name, $"its WOFF header gives the font inside {declaredSfnt:N0} bytes, its tables make {sfntSize:N0}: a damaged file");
        }

        CheckBlocks(woff, tables, directoryEnd, name);
        return Unpack(file, flavor, tables, (int)sfntSize, name);
    }

    // The tables, the metadata and the private data each in the file, none overlapping another.
    private static void CheckBlocks(ReadOnlySpan<byte> woff, Table[] tables, long directoryEnd, string name)
    {
        const int Metadata = -1, Private = -2;
        uint metaOffset = U32(woff, 24), metaLength = U32(woff, 28), privateOffset = U32(woff, 36), privateLength = U32(woff, 40);
        var blocks = new List<(long Start, long End, int Index)>(tables.Length + 2);
        for (int i = 0; i < tables.Length; i++)
        {
            blocks.Add((tables[i].Offset, (long)tables[i].Offset + tables[i].Stored, i));
        }

        foreach (var (offset, size, index) in ((uint, uint, int)[])[(metaOffset, metaLength, Metadata), (privateOffset, privateLength, Private)])
        {
            if (size == 0)
            {
                continue;
            }

            if (offset < directoryEnd || (long)offset + size > woff.Length)
            {
                throw Damaged(name, $"the WOFF's {Shown(index)} ({size:N0} bytes at {offset:N0}) lies outside the file's data (from {directoryEnd:N0} to {woff.Length:N0}): a damaged file");
            }

            blocks.Add((offset, (long)offset + size, index));
        }

        blocks.Sort((a, b) => a.Start != b.Start ? a.Start.CompareTo(b.Start) : a.End.CompareTo(b.End));
        for (int i = 1; i < blocks.Count; i++)
        {
            if (blocks[i].Start < blocks[i - 1].End)
            {
                throw Damaged(name, $"the WOFF's {Shown(blocks[i - 1].Index)} and {Shown(blocks[i].Index)} overlap (from {blocks[i].Start:N0}): a damaged file");
            }
        }

        string Shown(int index) => index switch { Metadata => "metadata", Private => "private data", _ => $"table '{WoffSettings.Tag(tables[index].Tag)}'" };
    }

    // The sfnt: its header, its directory by tag, then each table where the WOFF's order puts it, inflated or copied.
    private static byte[] Unpack(ReadOnlyMemory<byte> file, uint flavor, Table[] tables, int size, string name)
    {
        int count = tables.Length, power = 1, log = 0;
        while (power * 2 <= count)
        {
            power *= 2;
            log++;
        }

        var sfnt = new byte[size];
        BinaryPrimitives.WriteUInt32BigEndian(sfnt, flavor);
        BinaryPrimitives.WriteUInt16BigEndian(sfnt.AsSpan(4), (ushort)count);
        BinaryPrimitives.WriteUInt16BigEndian(sfnt.AsSpan(6), (ushort)(16 * power));
        BinaryPrimitives.WriteUInt16BigEndian(sfnt.AsSpan(8), (ushort)log);
        BinaryPrimitives.WriteUInt16BigEndian(sfnt.AsSpan(10), (ushort)(16 * count - 16 * power));

        // The data in the order the WOFF stores it (its encoder kept its font's order), each table padded with zeros.
        var order = new ulong[count];
        for (int i = 0; i < count; i++)
        {
            order[i] = ((ulong)tables[i].Offset << 32) | (uint)i;
        }

        Array.Sort(order);
        var at = new int[count];
        int next = WoffSettings.SfntHeaderSize + WoffSettings.SfntEntrySize * count;
        foreach (ulong key in order)
        {
            int i = (int)(uint)key;
            at[i] = next;
            next += (int)WoffSettings.Padded(tables[i].Length);
        }

        for (int i = 0; i < count; i++)
        {
            var entry = sfnt.AsSpan(WoffSettings.SfntHeaderSize + WoffSettings.SfntEntrySize * i, WoffSettings.SfntEntrySize);
            BinaryPrimitives.WriteUInt32BigEndian(entry, tables[i].Tag);
            BinaryPrimitives.WriteUInt32BigEndian(entry[4..], tables[i].Checksum);
            BinaryPrimitives.WriteUInt32BigEndian(entry[8..], (uint)at[i]);
            BinaryPrimitives.WriteUInt32BigEndian(entry[12..], tables[i].Length);
        }

        foreach (ulong key in order)
        {
            var table = tables[(int)(uint)key];
            var stored = file.Slice((int)table.Offset, (int)table.Stored);
            var into = sfnt.AsSpan(at[(int)(uint)key], (int)table.Length);
            if (table.Stored == table.Length)
            {
                stored.Span.CopyTo(into);
            }
            else
            {
                Inflate(stored, into, table.Tag, name);
            }
        }

        return sfnt;
    }

    // One zlib-compressed table into its place: exactly its length, or refused; nothing past one byte more is inflated.
    private static void Inflate(ReadOnlyMemory<byte> stored, Span<byte> into, uint tag, string name)
    {
        int read;
        bool more;
        try
        {
            using var source = MemoryMarshal.TryGetArray(stored, out ArraySegment<byte> segment)
                ? new MemoryStream(segment.Array!, segment.Offset, segment.Count, writable: false)
                : new MemoryStream(stored.ToArray(), writable: false);
            using var zlib = new ZLibStream(source, CompressionMode.Decompress);
            read = zlib.ReadAtLeast(into, into.Length, throwOnEndOfStream: false);
            more = read == into.Length && zlib.ReadByte() >= 0;
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException(Message(name, $"the WOFF table '{WoffSettings.Tag(tag)}' does not inflate ({e.Message}): a damaged file"), e);
        }

        if (more)
        {
            throw Damaged(name, $"the WOFF table '{WoffSettings.Tag(tag)}' inflates past the {into.Length:N0} bytes it declares (nothing more was inflated): a damaged or hostile file");
        }

        if (read < into.Length)
        {
            throw Damaged(name, $"the WOFF table '{WoffSettings.Tag(tag)}' inflates to {read:N0} bytes, fewer than the {into.Length:N0} it declares: a damaged file");
        }
    }

    private static uint U32(ReadOnlySpan<byte> data, int at) => BinaryPrimitives.ReadUInt32BigEndian(data[at..]);

    private static int U16(ReadOnlySpan<byte> data, int at) => BinaryPrimitives.ReadUInt16BigEndian(data[at..]);

    private static string Message(string name, FormattableString what) => $"{name}: {what.ToString(CultureInfo.InvariantCulture)}";

    private static InvalidDataException Damaged(string name, FormattableString what) => new(Message(name, what));

    // A table directory entry: its tag, where its stored bytes are in the WOFF and how many, its length and checksum in the font.
    private readonly record struct Table(uint Tag, uint Offset, uint Stored, uint Length, uint Checksum);
}
