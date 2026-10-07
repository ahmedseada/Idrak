// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Numerics;
using System.Text.Json.Nodes;
using System.Text;
using Idrak.Data.Abstractions;
using Idrak.Data.Parquet;

namespace Idrak.Data;

/// <summary>
/// Reads Apache Parquet files into rows, without dependencies: nested lists, structs and maps (a chat dataset's
/// <c>messages: [{role, content}]</c> comes back as a JSON array of objects), dictionary, RLE, delta and byte-stream-split
/// encodings, data pages v1 and v2, Snappy / Gzip / Brotli / LZ4 compression; strings, numbers, booleans, dates,
/// timestamps, decimals and UUIDs become their JSON forms, other binary values base64.
/// </summary>
public static class ParquetFile
{
    /// <summary>The number of rows and the column names of a Parquet file (read from its footer only).</summary>
    public static (long Rows, IReadOnlyList<string> Columns) Describe(Stream stream)
    {
        var (metadata, root) = ReadFooter(stream);
        return (metadata.Long(3) ?? 0, [.. root.Children.Select(c => c.Name)]);
    }

    /// <summary>The rows of a seekable Parquet stream, row group by row group; only <paramref name="columns"/> (top-level names) when given.</summary>
    public static IEnumerable<JsonObject> ReadRows(Stream stream, IReadOnlyCollection<string>? columns = null)
    {
        if (!stream.CanSeek)
        {
            throw new ArgumentException("Parquet needs a seekable stream.", nameof(stream));
        }

        var (metadata, root) = ReadFooter(stream);
        var leaves = new List<Leaf>();
        CollectLeaves(root, leaves);
        var wanted = leaves.Where(l => columns is null || columns.Contains(l.Steps[0].Name)).ToList();
        foreach (var node in metadata.List(4))
        {
            var group = (ThriftStruct)node!;
            long rows = group.Long(3) ?? 0;
            // Each chunk's column path decoded once per row group (not once per wanted leaf and chunk).
            var chunks = new Dictionary<string, ThriftStruct>();
            foreach (var c in group.List(1).Cast<ThriftStruct>())
            {
                if (c.Struct(3) is { } m)
                {
                    chunks.TryAdd(string.Join('\0', m.List(3).Select(p => Encoding.UTF8.GetString((byte[])p!))), c);
                }
            }

            var data = new List<ColumnData>();
            foreach (var leaf in wanted)
            {
                var chunk = chunks.GetValueOrDefault(string.Join('\0', leaf.Path))
                            ?? throw new InvalidDataException($"Parquet row group has no column {string.Join('.', leaf.Path)}.");
                data.Add(ReadColumn(stream, chunk.Struct(3)!, leaf));
            }

            var cursors = new int[data.Count];
            var valueCursors = new int[data.Count];
            for (long r = 0; r < rows; r++)
            {
                var row = new JsonObject();
                for (int c = 0; c < data.Count; c++)
                {
                    var column = data[c];
                    var leaf = column.Leaf;
                    if (leaf.Steps.Count == 1 && column.Rep is null)
                    {
                        // A top-level column (no nesting, no repetition): its value goes straight into the row.
                        int i = cursors[c]++;
                        row[leaf.Steps[0].Name] = column.Def is null || column.Def[i] == leaf.MaxDef ? ToNode(leaf.Node, column.Values[valueCursors[c]++]) : null;
                        continue;
                    }

                    var tree = new JsonObject();
                    do
                    {
                        int i = cursors[c];
                        int rep = column.Rep is null ? 0 : column.Rep[i];
                        int def = column.Def is null ? leaf.MaxDef : column.Def[i];
                        var value = def == leaf.MaxDef ? ToNode(leaf.Node, column.Values[valueCursors[c]++]) : null;
                        new Assembler(leaf.Steps, rep, def, value).FillObject(tree, 0);
                        cursors[c]++;
                    }
                    while (cursors[c] < column.Count && column.Rep is not null && column.Rep[cursors[c]] != 0);

                    Merge(row, tree);
                }

                yield return row;
            }
        }
    }

    // ------------------------------------------------------------------ schema

    private sealed class SchemaNode
    {
        public required string Name { get; init; }

        public int Repetition { get; init; }                        // 0 required, 1 optional, 2 repeated

        public int? Type { get; init; }

        public int TypeLength { get; init; }

        public int? Converted { get; init; }

        public ThriftStruct? Logical { get; init; }

        public int Scale { get; init; }

        public SchemaNode? Parent { get; set; }

        public List<SchemaNode> Children { get; } = [];

        public int Def { get; set; }

        public int Rep { get; set; }

        public bool IsLeaf => Children.Count == 0;

        public bool IsListWrapper => !IsLeaf && Children.Count == 1 && Children[0].Repetition == 2
                                     && (Converted is 1 or 2 or 3 || Logical?.Fields.ContainsKey(2) == true || Logical?.Fields.ContainsKey(3) == true);

        public bool IsList => Converted == 3 || Logical?.Fields.ContainsKey(3) == true;
    }

    private enum StepKind
    {
        Named,
        Repeated,
        Anonymous,
    }

    private sealed record Step(StepKind Kind, string Name, int Def, int Rep, bool IsLeaf, bool IsList);

    private sealed record Leaf(SchemaNode Node, List<string> Path, List<Step> Steps, int MaxDef, int MaxRep);

    private sealed record ColumnData(Leaf Leaf, int Count, int[]? Rep, int[]? Def, List<object?> Values);

    private static (ThriftStruct Metadata, SchemaNode Root) ReadFooter(Stream stream)
    {
        Span<byte> tail = stackalloc byte[8];
        if (stream.Length < 12)
        {
            throw new InvalidDataException("Not a Parquet file (too short).");
        }

        stream.Seek(-8, SeekOrigin.End);
        stream.ReadExactly(tail);
        if (!tail[4..].SequenceEqual("PAR1"u8))
        {
            throw new InvalidDataException(tail[4..].SequenceEqual("PARE"u8) ? "Encrypted Parquet files are not supported." : "Not a Parquet file (no PAR1 footer).");
        }

        int length = BinaryPrimitives.ReadInt32LittleEndian(tail);
        if (length <= 0 || length > stream.Length - 12)
        {
            throw new InvalidDataException("Corrupt Parquet footer.");
        }

        var footer = new byte[length];
        stream.Seek(-8 - length, SeekOrigin.End);
        stream.ReadExactly(footer);
        var metadata = new ThriftReader(footer).ReadStruct();

        var elements = metadata.List(2).Cast<ThriftStruct>().ToList();
        if (elements.Count == 0)
        {
            throw new InvalidDataException("Parquet file without a schema.");
        }

        int index = 0;
        var root = Build(elements, ref index, null);
        return (metadata, root);
    }

    private static SchemaNode Build(List<ThriftStruct> elements, ref int index, SchemaNode? parent)
    {
        var e = elements[index++];
        var node = new SchemaNode
        {
            Name = e.String(4) ?? "",
            Repetition = parent is null ? 0 : e.Int(3) ?? 0,
            Type = e.Int(1),
            TypeLength = e.Int(2) ?? 0,
            Converted = e.Int(6),
            Logical = e.Struct(10),
            Scale = e.Int(7) ?? (e.Struct(10)?.Struct(5)?.Int(1) ?? 0),
            Parent = parent,
        };
        node.Def = (parent?.Def ?? 0) + (node.Repetition > 0 ? 1 : 0);
        node.Rep = (parent?.Rep ?? 0) + (node.Repetition == 2 ? 1 : 0);
        int children = e.Int(5) ?? 0;
        for (int i = 0; i < children; i++)
        {
            node.Children.Add(Build(elements, ref index, node));
        }

        return node;
    }

    private static void CollectLeaves(SchemaNode node, List<Leaf> leaves)
    {
        foreach (var child in node.Children)
        {
            if (!child.IsLeaf)
            {
                CollectLeaves(child, leaves);
                continue;
            }

            var chain = new List<SchemaNode>();
            for (var n = child; n.Parent is not null; n = n.Parent)
            {
                chain.Insert(0, n);
            }

            var steps = new List<Step>();
            foreach (var n in chain)
            {
                var p = n.Parent!;
                if (n.Repetition == 2)
                {
                    if (!p.IsListWrapper)
                    {
                        // A bare repeated field (the legacy list layout): a named array of its values.
                        steps.Add(new Step(StepKind.Named, n.Name, p.Def, p.Rep, false, true));
                    }

                    steps.Add(new Step(StepKind.Repeated, n.Name, n.Def, n.Rep, n.IsLeaf, false));
                }
                else if (p.Repetition == 2 && p.Parent is { } wrapper && wrapper.IsListWrapper && wrapper.IsList && p.Children.Count == 1
                         && p.Name != "array" && p.Name != wrapper.Name + "_tuple")
                {
                    steps.Add(new Step(StepKind.Anonymous, n.Name, n.Def, n.Rep, n.IsLeaf, n.IsListWrapper));
                }
                else
                {
                    steps.Add(new Step(StepKind.Named, n.Name, n.Def, n.Rep, n.IsLeaf, n.IsListWrapper));
                }
            }

            leaves.Add(new Leaf(child, [.. chain.Select(n => n.Name)], steps, child.Def, child.Rep));
        }
    }

    // ------------------------------------------------------------------ record assembly

    // Places one (repetition level, definition level, value) of a leaf column into a row's JSON (Dremel assembly).
    private readonly struct Assembler(List<Step> steps, int rep, int def, JsonNode? value)
    {
        public void FillObject(JsonObject obj, int i)
        {
            var s = steps[i];
            if (def < s.Def)
            {
                if (!obj.ContainsKey(s.Name))
                {
                    obj[s.Name] = null;
                }

                return;
            }

            if (s.IsLeaf)
            {
                obj[s.Name] = value;
                return;
            }

            var existing = obj[s.Name];
            var filled = FillGroup(existing, i);
            if (!ReferenceEquals(filled, existing))
            {
                obj[s.Name] = filled;
            }
        }

        private JsonNode FillGroup(JsonNode? existing, int i)
        {
            if (steps[i].IsList)
            {
                var array = existing as JsonArray ?? [];
                FillArray(array, i + 1);
                return array;
            }

            var obj = existing as JsonObject ?? [];
            FillObject(obj, i + 1);
            return obj;
        }

        private void FillArray(JsonArray array, int i)
        {
            var s = steps[i];
            if (def < s.Def)
            {
                return;                                             // an empty list
            }

            bool append = rep <= s.Rep || array.Count == 0;
            var element = append ? null : array[^1];
            var filled = FillElement(element, i);
            if (append)
            {
                array.Add(filled);
            }
            else if (!ReferenceEquals(filled, element))
            {
                array[^1] = filled;
            }
        }

        private JsonNode? FillElement(JsonNode? existing, int i)
        {
            var s = steps[i];
            if (s.IsLeaf)
            {
                return value;
            }

            if (i + 1 < steps.Count && steps[i + 1].Kind == StepKind.Anonymous)
            {
                var e = steps[i + 1];
                if (def < e.Def)
                {
                    return null;
                }

                return e.IsLeaf ? value : FillGroup(existing, i + 1);
            }

            var obj = existing as JsonObject ?? [];
            FillObject(obj, i + 1);
            return obj;
        }
    }

    // Leaves of the same struct or list build separate trees; merging joins them (objects by key, lists by position).
    private static void Merge(JsonObject target, JsonObject source)
    {
        foreach (var key in source.Select(p => p.Key).ToList())
        {
            var value = source[key];
            source.Remove(key);
            if (!target.TryGetPropertyValue(key, out var existing) || existing is null)
            {
                target[key] = value;
            }
            else
            {
                MergeNode(existing, value);
            }
        }
    }

    private static void MergeNode(JsonNode target, JsonNode? source)
    {
        switch (target, source)
        {
            case (JsonObject t, JsonObject s):
                Merge(t, s);
                break;
            case (JsonArray t, JsonArray s):
                var items = s.ToList();
                s.Clear();
                for (int i = 0; i < items.Count; i++)
                {
                    if (i >= t.Count)
                    {
                        t.Add(items[i]);
                    }
                    else if (t[i] is null)
                    {
                        t[i] = items[i];
                    }
                    else
                    {
                        MergeNode(t[i]!, items[i]);
                    }
                }

                break;
        }
    }

    // ------------------------------------------------------------------ column chunks and pages

    private static ColumnData ReadColumn(Stream stream, ThriftStruct meta, Leaf leaf)
    {
        int codec = meta.Int(4) ?? 0;
        IParquetCodec? decoder = null;                              // looked up once, at the chunk's first compressed page
        byte[] Decompress(ReadOnlySpan<byte> input, int size) => (decoder ??= ParquetCodecs.Get(codec)).Decompress(input, size);
        long total = meta.Long(5) ?? 0;
        long dataOffset = meta.Long(9) ?? throw new InvalidDataException("Parquet column without a data page offset.");
        long start = meta.Long(11) is { } dict && dict > 0 && dict < dataOffset ? dict : dataOffset;
        long size = meta.Long(7) ?? throw new InvalidDataException("Parquet column without a size.");
        var bytes = new byte[size];
        stream.Seek(start, SeekOrigin.Begin);
        stream.ReadExactly(bytes);

        var reps = leaf.MaxRep > 0 ? new List<int>((int)Math.Min(total, 1 << 20)) : null;
        var defs = leaf.MaxDef > 0 ? new List<int>((int)Math.Min(total, 1 << 20)) : null;
        var values = new List<object?>();
        object?[]? dictionary = null;
        int levels = 0, pos = 0;
        while (levels < total && pos < bytes.Length)
        {
            var reader = new ThriftReader(bytes.AsSpan(pos));
            var header = reader.ReadStruct();
            pos += reader.Position;
            int type = header.Int(1) ?? -1;
            int uncompressed = header.Int(2) ?? 0;
            int compressed = header.Int(3) ?? 0;
            var page = bytes.AsSpan(pos, compressed);
            pos += compressed;
            switch (type)
            {
                case 2:                                             // dictionary page
                {
                    var h = header.Struct(7)!;
                    var raw = Decompress(page, uncompressed);
                    var list = new List<object?>();
                    int offset = 0;
                    Plain(raw, ref offset, leaf.Node, h.Int(1) ?? 0, list);
                    dictionary = [.. list];

                    // Entries read as text decoded once here, not again for every row that refers to them.
                    for (int e = 0; e < dictionary.Length; e++)
                    {
                        if (dictionary[e] is byte[] entry && ToNode(leaf.Node, entry) is JsonValue text && text.TryGetValue(out string? decoded))
                        {
                            dictionary[e] = decoded;
                        }
                    }
                    break;
                }

                case 0:                                             // data page v1
                {
                    var h = header.Struct(5)!;
                    int count = h.Int(1) ?? 0;
                    var raw = Decompress(page, uncompressed);
                    int offset = 0;
                    int present = ReadLevels(raw, ref offset, count, leaf, reps, defs, lengthPrefixed: true, repLength: 0, defLength: 0);
                    DecodeValues(raw.AsSpan(offset), h.Int(2) ?? 0, leaf.Node, present, dictionary, values);
                    levels += count;
                    break;
                }

                case 3:                                             // data page v2
                {
                    var h = header.Struct(8)!;
                    int count = h.Int(1) ?? 0;
                    int defLength = h.Int(5) ?? 0, repLength = h.Int(6) ?? 0;
                    bool isCompressed = h.Bool(7) ?? true;
                    var levelBytes = page[..(repLength + defLength)].ToArray();
                    int offset = 0;
                    int present = ReadLevels(levelBytes, ref offset, count, leaf, reps, defs, lengthPrefixed: false, repLength, defLength);
                    var body = page[(repLength + defLength)..];
                    var raw = isCompressed && codec != 0 ? Decompress(body, uncompressed - repLength - defLength) : body.ToArray();
                    DecodeValues(raw, h.Int(4) ?? 0, leaf.Node, present, dictionary, values);
                    levels += count;
                    break;
                }

                default:
                    break;                                          // index pages and others carry no rows
            }
        }

        return new ColumnData(leaf, levels, reps?.ToArray(), defs?.ToArray(), values);
    }

    private static int ReadLevels(byte[] data, ref int offset, int count, Leaf leaf, List<int>? reps, List<int>? defs, bool lengthPrefixed, int repLength, int defLength)
    {
        if (leaf.MaxRep > 0)
        {
            int length = lengthPrefixed ? BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset)) : repLength;
            offset += lengthPrefixed ? 4 : 0;
            Hybrid(data.AsSpan(offset, length), BitWidth(leaf.MaxRep), count, reps!);
            offset += length;
        }

        if (leaf.MaxDef == 0)
        {
            return count;
        }

        int defBytes = lengthPrefixed ? BinaryPrimitives.ReadInt32LittleEndian(data.AsSpan(offset)) : defLength;
        offset += lengthPrefixed ? 4 : 0;
        int before = defs!.Count;
        Hybrid(data.AsSpan(offset, defBytes), BitWidth(leaf.MaxDef), count, defs);
        offset += defBytes;
        int present = 0;
        for (int i = before; i < defs.Count; i++)
        {
            present += defs[i] == leaf.MaxDef ? 1 : 0;
        }

        return present;
    }

    private static int BitWidth(int max) => max == 0 ? 0 : 32 - BitOperations.LeadingZeroCount((uint)max);

    // The RLE / bit-packing hybrid encoding.
    private static void Hybrid(ReadOnlySpan<byte> data, int width, int count, List<int> output)
    {
        int pos = 0, target = output.Count + count;
        int byteWidth = (width + 7) / 8;
        while (output.Count < target)
        {
            if (pos >= data.Length)
            {
                throw new InvalidDataException("A Parquet RLE run ends early.");
            }

            ulong header = UVarint(data, ref pos);
            if ((header & 1) == 0)
            {
                int run = (int)(header >> 1);
                int value = 0;
                for (int b = 0; b < byteWidth; b++)
                {
                    value |= data[pos++] << (8 * b);
                }

                for (int i = 0; i < run && output.Count < target; i++)
                {
                    output.Add(value);
                }
            }
            else
            {
                int values = (int)(header >> 1) * 8;
                int bytes = values * width / 8;
                var packed = data.Slice(pos, Math.Min(bytes, data.Length - pos));
                pos += bytes;
                for (int i = 0; i < values && output.Count < target; i++)
                {
                    output.Add(Bits(packed, i * width, width));
                }
            }
        }
    }

    private static int Bits(ReadOnlySpan<byte> data, long bit, int width)
    {
        if (width == 0)
        {
            return 0;
        }

        long value = 0;
        for (int got = 0; got < width;)
        {
            int index = (int)(bit >> 3), shift = (int)(bit & 7);
            int take = Math.Min(8 - shift, width - got);
            int b = index < data.Length ? data[index] : 0;
            value |= (long)((b >> shift) & ((1 << take) - 1)) << got;
            got += take;
            bit += take;
        }

        return (int)value;
    }

    private static ulong UVarint(ReadOnlySpan<byte> data, ref int pos)
    {
        ulong value = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            byte b = data[pos++];
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return value;
            }
        }

        throw new InvalidDataException("Bad varint in a Parquet page.");
    }

    private static long ZigZag(ulong n) => (long)(n >> 1) ^ -(long)(n & 1);

    // ------------------------------------------------------------------ value encodings

    private static void DecodeValues(ReadOnlySpan<byte> data, int encoding, SchemaNode node, int count, object?[]? dictionary, List<object?> output)
    {
        if (count == 0)
        {
            return;
        }

        switch (encoding)
        {
            case 0:                                                 // PLAIN
                int offset = 0;
                Plain(data, ref offset, node, count, output);
                break;
            case 2 or 8:                                            // PLAIN_DICTIONARY, RLE_DICTIONARY
                if (dictionary is null)
                {
                    throw new InvalidDataException("A dictionary-encoded Parquet page without a dictionary.");
                }

                var indices = new List<int>(count);
                Hybrid(data[1..], data[0], count, indices);
                foreach (int index in indices)
                {
                    output.Add(dictionary[index]);
                }

                break;
            case 3 when node.Type == 0:                             // RLE booleans (length-prefixed)
                var bits = new List<int>(count);
                Hybrid(data.Slice(4, BinaryPrimitives.ReadInt32LittleEndian(data)), 1, count, bits);
                output.AddRange(bits.Select(b => (object?)(b != 0)));
                break;
            case 5:                                                 // DELTA_BINARY_PACKED
                int used = 0;
                foreach (long v in DeltaBinaryPacked(data, ref used))
                {
                    output.Add(node.Type == 1 ? (int)v : v);
                }

                break;
            case 6:                                                 // DELTA_LENGTH_BYTE_ARRAY
            {
                int lengthBytes = 0;
                var lengths = DeltaBinaryPacked(data, ref lengthBytes);
                int pos = lengthBytes;
                var kind = BytesKind(node);
                foreach (long length in lengths)
                {
                    output.Add(Bytes(data.Slice(pos, (int)length), kind));
                    pos += (int)length;
                }

                break;
            }

            case 7:                                                 // DELTA_BYTE_ARRAY
            {
                int prefixBytes = 0;
                var prefixes = DeltaBinaryPacked(data, ref prefixBytes);
                int suffixLengthBytes = 0;
                var suffixLengths = DeltaBinaryPacked(data[prefixBytes..], ref suffixLengthBytes);
                int pos = prefixBytes + suffixLengthBytes;
                byte[] previous = [];
                for (int i = 0; i < prefixes.Count; i++)
                {
                    int prefix = (int)prefixes[i], suffix = (int)suffixLengths[i];
                    var value = new byte[prefix + suffix];
                    previous.AsSpan(0, prefix).CopyTo(value);
                    data.Slice(pos, suffix).CopyTo(value.AsSpan(prefix));
                    pos += suffix;
                    output.Add(value);
                    previous = value;
                }

                break;
            }

            case 9:                                                 // BYTE_STREAM_SPLIT
            {
                int width = node.Type switch { 1 or 4 => 4, 2 or 5 => 8, 7 => node.TypeLength, _ => throw new NotSupportedException("BYTE_STREAM_SPLIT on this type.") };
                var joined = new byte[count * width];
                for (int i = 0; i < count; i++)
                {
                    for (int k = 0; k < width; k++)
                    {
                        joined[i * width + k] = data[k * count + i];
                    }
                }

                int o = 0;
                Plain(joined, ref o, node, count, output);
                break;
            }

            default:
                throw new NotSupportedException($"Parquet encoding {encoding} is not supported.");
        }
    }

    private static List<long> DeltaBinaryPacked(ReadOnlySpan<byte> data, ref int used)
    {
        int pos = 0;
        int blockSize = (int)UVarint(data, ref pos);
        int miniblocks = (int)UVarint(data, ref pos);
        int total = (int)UVarint(data, ref pos);
        long last = ZigZag(UVarint(data, ref pos));
        var values = new List<long>(total);
        if (total > 0)
        {
            values.Add(last);
        }

        int perMiniblock = blockSize / miniblocks;
        while (values.Count < total)
        {
            long minDelta = ZigZag(UVarint(data, ref pos));
            var widths = data.Slice(pos, miniblocks).ToArray();
            pos += miniblocks;
            for (int m = 0; m < miniblocks && values.Count < total; m++)
            {
                int width = widths[m];
                var packed = data.Slice(pos, Math.Min(perMiniblock * width / 8, data.Length - pos));
                for (int i = 0; i < perMiniblock && values.Count < total; i++)
                {
                    long delta = width == 0 ? 0 : BitsLong(packed, (long)i * width, width);
                    last = unchecked(last + minDelta + delta);
                    values.Add(last);
                }

                pos += perMiniblock * width / 8;
            }
        }

        used += pos;
        return values;
    }

    private static long BitsLong(ReadOnlySpan<byte> data, long bit, int width)
    {
        ulong value = 0;
        for (int got = 0; got < width;)
        {
            int index = (int)(bit >> 3), shift = (int)(bit & 7);
            int take = Math.Min(8 - shift, width - got);
            ulong b = index < data.Length ? data[index] : 0UL;
            value |= ((b >> shift) & ((1UL << take) - 1)) << got;
            got += take;
            bit += take;
        }

        return (long)value;
    }

    private static void Plain(ReadOnlySpan<byte> data, ref int offset, SchemaNode node, int count, List<object?> output)
    {
        switch (node.Type)
        {
            case 0:
                for (int i = 0; i < count; i++)
                {
                    output.Add(((data[offset + (i >> 3)] >> (i & 7)) & 1) != 0);
                }

                offset += (count + 7) / 8;
                break;
            case 1:
                for (int i = 0; i < count; i++, offset += 4)
                {
                    output.Add(BinaryPrimitives.ReadInt32LittleEndian(data[offset..]));
                }

                break;
            case 2:
                for (int i = 0; i < count; i++, offset += 8)
                {
                    output.Add(BinaryPrimitives.ReadInt64LittleEndian(data[offset..]));
                }

                break;
            case 3:
                for (int i = 0; i < count; i++, offset += 12)
                {
                    long nanos = BinaryPrimitives.ReadInt64LittleEndian(data[offset..]);
                    int day = BinaryPrimitives.ReadInt32LittleEndian(data[(offset + 8)..]);
                    output.Add(DateTime.UnixEpoch.AddDays(day - 2440588).AddTicks(nanos / 100).ToString("yyyy-MM-ddTHH:mm:ss.FFFFFFF", CultureInfo.InvariantCulture));
                }

                break;
            case 4:
                for (int i = 0; i < count; i++, offset += 4)
                {
                    output.Add(BinaryPrimitives.ReadSingleLittleEndian(data[offset..]));
                }

                break;
            case 5:
                for (int i = 0; i < count; i++, offset += 8)
                {
                    output.Add(BinaryPrimitives.ReadDoubleLittleEndian(data[offset..]));
                }

                break;
            case 6:
                var kind = BytesKind(node);
                for (int i = 0; i < count; i++)
                {
                    int length = BinaryPrimitives.ReadInt32LittleEndian(data[offset..]);
                    output.Add(Bytes(data.Slice(offset + 4, length), kind));
                    offset += 4 + length;
                }

                break;
            case 7:
                for (int i = 0; i < count; i++, offset += node.TypeLength)
                {
                    output.Add(data.Slice(offset, node.TypeLength).ToArray());
                }

                break;
            default:
                throw new NotSupportedException($"Parquet physical type {node.Type} is not supported.");
        }
    }

    private enum ByteArray
    {
        Text,                                                       // UTF8 / STRING, ENUM, JSON: always text
        TextIfValid,                                                // plain binary: text when it is valid UTF-8, else base64
        Bytes,                                                      // decimals, UUIDs, float16s: left to ToNode
    }

    // How ToNode turns a BYTE_ARRAY column's values into JSON, so text is decoded straight from the page (no byte[] per value).
    private static ByteArray BytesKind(SchemaNode node)
    {
        var logical = node.Logical;
        bool Is(short id) => logical?.Fields.ContainsKey(id) == true;
        return node.Converted is 0 or 4 or 19 || Is(1) || Is(4) || Is(12) ? ByteArray.Text
             : node.Converted == 5 || Is(5) || Is(14) || Is(15) || node.Type != 6 ? ByteArray.Bytes
             : ByteArray.TextIfValid;
    }

    private static object Bytes(ReadOnlySpan<byte> value, ByteArray kind) => kind switch
    {
        ByteArray.Text => Encoding.UTF8.GetString(value),
        ByteArray.TextIfValid when System.Text.Unicode.Utf8.IsValid(value) => Encoding.UTF8.GetString(value),
        _ => value.ToArray(),
    };

    // ------------------------------------------------------------------ values as JSON

    private static JsonNode? ToNode(SchemaNode node, object? value)
    {
        var logical = node.Logical;
        bool Is(short id) => logical?.Fields.ContainsKey(id) == true;
        switch (value)
        {
            case null:
                return null;
            case bool b:
                return b;
            case string s:
                return s;
            case int i:
                if (node.Converted == 6 || Is(6))
                {
                    return DateOnly.FromDayNumber(new DateOnly(1970, 1, 1).DayNumber + i).ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
                }

                if (node.Converted == 5 || Is(5))
                {
                    return (decimal)i / Pow10(node.Scale);
                }

                if (node.Converted == 7 || Is(7))
                {
                    return TimeOnly.FromTimeSpan(TimeSpan.FromMilliseconds(i)).ToString("HH:mm:ss.FFF", CultureInfo.InvariantCulture);
                }

                if (node.Converted is 11 or 12 or 13 || logical?.Struct(10)?.Bool(2) == false)
                {
                    return (uint)i;
                }

                return i;
            case long l:
                if (node.Converted is 9 or 10 || Is(8))
                {
                    int unit = node.Converted == 9 ? 1 : node.Converted == 10 ? 2 : TimeUnit(logical!.Struct(8)!.Struct(2));
                    var time = unit switch
                    {
                        1 => DateTime.UnixEpoch.AddTicks(l * TimeSpan.TicksPerMillisecond),
                        2 => DateTime.UnixEpoch.AddTicks(l * 10),
                        _ => DateTime.UnixEpoch.AddTicks(l / 100),
                    };
                    return time.ToString("yyyy-MM-ddTHH:mm:ss.FFFFFFF", CultureInfo.InvariantCulture);
                }

                if (node.Converted == 5 || Is(5))
                {
                    return (decimal)l / Pow10(node.Scale);
                }

                if (node.Converted == 14 || logical?.Struct(10)?.Bool(2) == false)
                {
                    return (ulong)l;
                }

                return l;
            case float f:
                return float.IsFinite(f) ? f : null;
            case double d:
                return double.IsFinite(d) ? d : null;
            case byte[] bytes:
                if (node.Converted is 0 or 4 or 19 || Is(1) || Is(4) || Is(12))
                {
                    return Encoding.UTF8.GetString(bytes);
                }

                if (node.Converted == 5 || Is(5))
                {
                    return (decimal)new BigInteger(bytes, isUnsigned: false, isBigEndian: true) / Pow10(node.Scale);
                }

                if (Is(14) && bytes.Length == 16)
                {
                    return new Guid(bytes, bigEndian: true).ToString();
                }

                if (Is(15) && bytes.Length == 2)
                {
                    return (float)BinaryPrimitives.ReadHalfLittleEndian(bytes);
                }

                if (node.Type == 6 && System.Text.Unicode.Utf8.IsValid(bytes))
                {
                    return Encoding.UTF8.GetString(bytes);   // valid UTF-8: checked first, as throwing per binary value is slow
                }

                return Convert.ToBase64String(bytes);
            default:
                return value.ToString();
        }
    }

    private static int TimeUnit(ThriftStruct? unit) => unit?.Fields.ContainsKey(1) == true ? 1 : unit?.Fields.ContainsKey(2) == true ? 2 : 3;

    private static decimal Pow10(int scale)
    {
        decimal result = 1;
        for (int i = 0; i < scale; i++)
        {
            result *= 10;
        }

        return result;
    }
}
