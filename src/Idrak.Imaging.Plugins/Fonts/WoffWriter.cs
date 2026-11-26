// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.IO.Compression;

namespace Idrak.Imaging.Plugins.Fonts;

/// <summary>
/// "woff": writes a face as a WOFF 1.0 font (.woff). The face is first written as an sfnt by the font writer
/// <see cref="SfntWriter"/> names (the library's "truetype", <see cref="TrueTypeWriter"/>, reached directly), with
/// the options given (a subset, the layout tables, names, the curve tolerance and hinting all pass through); then
/// <see cref="Encode"/> packs it: each table zlib-compressed through <see cref="ZLibStream"/> at
/// <see cref="CompressionLevel"/> and kept so only when that is smaller (else stored as it is), every table on a 4-byte
/// boundary and padded with zeros, the directory in tag order and the tables' data in the order the sfnt lays them out
/// (the order the OpenType specification recommends, which the decoder keeps), no metadata or private data. The same
/// face and options write the same bytes, and the WOFF unpacks (<see cref="WoffFormat.Decode"/>) to exactly the sfnt
/// written first.
/// </summary>
public sealed class WoffWriter : IFontWriter
{
    /// <summary>The writer's name: "woff".</summary>
    public const string WriterName = "woff";

    /// <summary>The variable that sets <see cref="CompressionLevel"/> (0 to 9): "IDRAK_WOFF_COMPRESSION".</summary>
    public const string CompressionLevelVariable = "IDRAK_WOFF_COMPRESSION";

    /// <summary>The font writer a face goes through before it is packed, unless set: the library's "truetype".</summary>
    public const string DefaultSfntWriter = "truetype";

    private readonly int _compressionLevel = WoffSettings.Int(CompressionLevelVariable, 0, 9) ?? 9;
    private readonly string _sfntWriter = DefaultSfntWriter;

    /// <inheritdoc/>
    public string Name => WriterName;

    /// <inheritdoc/>
    public IReadOnlyCollection<string> Extensions { get; } = [".woff"];

    /// <summary>
    /// The zlib level each table is compressed at, 0 (none: every table stored as it is) to 9 (the smallest, the default
    /// unless <see cref="CompressionLevelVariable"/> sets it when the writer is made). A table stays stored when its
    /// compressed bytes would not be fewer.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not 0 to 9.</exception>
    public int CompressionLevel
    {
        get => _compressionLevel;
        init => _compressionLevel = value is >= 0 and <= 9 ? value : throw new ArgumentOutOfRangeException(nameof(CompressionLevel), value, $"A zlib level is 0 to 9 ({CompressionLevelVariable}).");
    }

    /// <summary>
    /// The name of the writer that writes a face as an sfnt before it is packed
    /// (<see cref="DefaultSfntWriter"/> unless set; any writer of TrueType or OpenType fonts).
    /// </summary>
    /// <exception cref="ArgumentException">Empty, or this writer's own name.</exception>
    public string SfntWriter
    {
        get => _sfntWriter;
        init
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(value);
            _sfntWriter = !value.Equals(WriterName, StringComparison.OrdinalIgnoreCase) ? value
                : throw new ArgumentException($"The woff writer packs what another writer writes; it cannot be its own sfnt writer ('{value}').", nameof(SfntWriter));
        }
    }

    // Picking the font writers by name is not on the new design yet. The default, the library's "truetype"
    // (TrueTypeWriter), is reached directly; another name stops.
    private static readonly TrueTypeWriter TrueType = new();

    private static IFontWriter WriterNamed(string name) =>
        name.Equals(TrueTypeWriter.WriterName, StringComparison.OrdinalIgnoreCase) ? TrueType
        : throw new NotSupportedException($"The font writer '{name}' by name is not on the new design yet.");

    /// <inheritdoc/>
    /// <exception cref="NotSupportedException"><see cref="SfntWriter"/> names a writer other than the default, "truetype": a font writer by name is not on the new design yet.</exception>
    /// <exception cref="NotSupportedException">The sfnt writer refuses the face (its message says what it cannot hold).</exception>
    /// <exception cref="InvalidDataException">The sfnt writer wrote no sfnt font.</exception>
    public void Write(IFontFace face, Stream destination, FontWriteOptions options)
    {
        ArgumentNullException.ThrowIfNull(face);
        ArgumentNullException.ThrowIfNull(destination);
        ArgumentNullException.ThrowIfNull(options);
        using var sfnt = new MemoryStream();
        WriterNamed(SfntWriter).Write(face, sfnt, options);
        if (!sfnt.TryGetBuffer(out ArraySegment<byte> written))
        {
            written = sfnt.ToArray();
        }

        Encode(written, destination);
    }

    /// <summary>
    /// Packs an sfnt font (a TrueType or OpenType file, one font) as a WOFF 1.0 file into <paramref name="destination"/>
    /// (the stream stays the caller's; it is written once, at the end). Its tables are kept byte for byte, with their
    /// checksums; the WOFF header's version is the font's revision (head), as fontTools sets it.
    /// </summary>
    /// <exception cref="InvalidDataException"><paramref name="sfnt"/> is not one sfnt font (a collection, a WOFF, or a damaged directory).</exception>
    public void Encode(ReadOnlySpan<byte> sfnt, Stream destination)
    {
        ArgumentNullException.ThrowIfNull(destination);
        if (sfnt.Length < WoffSettings.SfntHeaderSize)
        {
            throw new InvalidDataException($"An sfnt font is at least {WoffSettings.SfntHeaderSize} bytes; {sfnt.Length} were given.");
        }

        uint flavor = BinaryPrimitives.ReadUInt32BigEndian(sfnt);
        int count = BinaryPrimitives.ReadUInt16BigEndian(sfnt[4..]);
        if (flavor is WoffSettings.Woff or WoffSettings.Woff2 or WoffSettings.Collection)
        {
            throw new InvalidDataException($"A WOFF 1.0 file packs one sfnt font, not a {WoffSettings.Tag(flavor)} file.");
        }

        if (count == 0 || WoffSettings.SfntHeaderSize + WoffSettings.SfntEntrySize * count > sfnt.Length)
        {
            throw new InvalidDataException($"An sfnt font of {count} tables in {sfnt.Length} bytes: its directory is damaged.");
        }

        // The sfnt's tables: the directory by tag (WOFF's order), the data in the sfnt's own order (by offset).
        var tables = new SfntTable[count];
        for (int i = 0; i < count; i++)
        {
            var entry = sfnt.Slice(WoffSettings.SfntHeaderSize + WoffSettings.SfntEntrySize * i, WoffSettings.SfntEntrySize);
            var table = new SfntTable(BinaryPrimitives.ReadUInt32BigEndian(entry), BinaryPrimitives.ReadUInt32BigEndian(entry[4..]),
                BinaryPrimitives.ReadUInt32BigEndian(entry[8..]), BinaryPrimitives.ReadUInt32BigEndian(entry[12..]));
            if ((long)table.Offset + table.Length > sfnt.Length)
            {
                throw new InvalidDataException($"The sfnt font's table '{WoffSettings.Tag(table.Tag)}' ({table.Length} bytes at {table.Offset}) runs past its {sfnt.Length} bytes.");
            }

            tables[i] = table;
        }

        Array.Sort(tables, static (a, b) => a.Tag.CompareTo(b.Tag));
        for (int i = 1; i < count; i++)
        {
            if (tables[i].Tag == tables[i - 1].Tag)
            {
                throw new InvalidDataException($"The sfnt font lists the table '{WoffSettings.Tag(tables[i].Tag)}' twice.");
            }
        }

        var order = new ulong[count];
        for (int i = 0; i < count; i++)
        {
            order[i] = ((ulong)tables[i].Offset << 32) | (uint)i;
        }

        Array.Sort(order);

        // The header and directory are written last, into the room left for them at the start.
        int directoryEnd = WoffSettings.HeaderSize + WoffSettings.EntrySize * count;
        using var woff = new MemoryStream(directoryEnd + sfnt.Length + 64);
        woff.SetLength(directoryEnd);
        woff.Position = directoryEnd;
        var placed = new (int Offset, int Stored)[count];
        var compression = new ZLibCompressionOptions { CompressionLevel = CompressionLevel };
        long sfntSize = WoffSettings.SfntHeaderSize + (long)WoffSettings.SfntEntrySize * count;
        foreach (ulong key in order)
        {
            int i = (int)(uint)key;
            int at = (int)woff.Position;
            placed[i] = (at, Store(woff, sfnt.Slice((int)tables[i].Offset, (int)tables[i].Length), compression));
            while (woff.Position % 4 != 0)
            {
                woff.WriteByte(0);
            }

            sfntSize += WoffSettings.Padded(tables[i].Length);
        }

        int head = Array.FindIndex(tables, t => t.Tag == WoffSettings.Head);
        uint revision = head >= 0 && tables[head].Length >= 8 ? BinaryPrimitives.ReadUInt32BigEndian(sfnt[((int)tables[head].Offset + 4)..]) : 0;
        var file = woff.GetBuffer().AsSpan(0, (int)woff.Length);
        BinaryPrimitives.WriteUInt32BigEndian(file, WoffSettings.Woff);
        BinaryPrimitives.WriteUInt32BigEndian(file[4..], flavor);
        BinaryPrimitives.WriteUInt32BigEndian(file[8..], (uint)file.Length);
        BinaryPrimitives.WriteUInt16BigEndian(file[12..], (ushort)count);
        BinaryPrimitives.WriteUInt16BigEndian(file[14..], 0);                       // reserved
        BinaryPrimitives.WriteUInt32BigEndian(file[16..], checked((uint)sfntSize));
        BinaryPrimitives.WriteUInt32BigEndian(file[20..], revision);                // majorVersion, minorVersion: head's fontRevision
        file[24..WoffSettings.HeaderSize].Clear();                                   // no metadata, no private data
        for (int i = 0; i < count; i++)
        {
            var entry = file.Slice(WoffSettings.HeaderSize + WoffSettings.EntrySize * i, WoffSettings.EntrySize);
            BinaryPrimitives.WriteUInt32BigEndian(entry, tables[i].Tag);
            BinaryPrimitives.WriteUInt32BigEndian(entry[4..], (uint)placed[i].Offset);
            BinaryPrimitives.WriteUInt32BigEndian(entry[8..], (uint)placed[i].Stored);
            BinaryPrimitives.WriteUInt32BigEndian(entry[12..], tables[i].Length);
            BinaryPrimitives.WriteUInt32BigEndian(entry[16..], tables[i].Checksum);
        }

        destination.Write(file);
    }

    // One table at the stream's end: compressed when that is smaller, else as it is; how many bytes it stores.
    private static int Store(MemoryStream woff, ReadOnlySpan<byte> table, ZLibCompressionOptions compression)
    {
        long start = woff.Position;
        if (compression.CompressionLevel > 0 && table.Length > 0)
        {
            using (var zlib = new ZLibStream(woff, compression, leaveOpen: true))
            {
                zlib.Write(table);
            }

            long stored = woff.Position - start;
            if (stored < table.Length)
            {
                return (int)stored;
            }

            woff.SetLength(start);                                                    // not smaller: the table is kept as it is
            woff.Position = start;
        }

        woff.Write(table);
        return table.Length;
    }

    // An sfnt table directory entry: its tag, checksum, offset and length.
    private readonly record struct SfntTable(uint Tag, uint Checksum, uint Offset, uint Length);
}
