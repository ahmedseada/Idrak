// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Datasets.Parquet;

/// <summary>A Thrift struct read with the compact protocol: field id → value (bool, long, double, byte[], List, ThriftStruct).</summary>
internal sealed class ThriftStruct
{
    public Dictionary<short, object?> Fields { get; } = [];

    public long? Long(short id) => Fields.TryGetValue(id, out var v) && v is long l ? l : null;

    public int? Int(short id) => Long(id) is { } l ? (int)l : null;

    public bool? Bool(short id) => Fields.TryGetValue(id, out var v) && v is bool b ? b : null;

    public string? String(short id) => Fields.TryGetValue(id, out var v) && v is byte[] b ? Encoding.UTF8.GetString(b) : null;

    public ThriftStruct? Struct(short id) => Fields.TryGetValue(id, out var v) ? v as ThriftStruct : null;

    public List<object?> List(short id) => Fields.TryGetValue(id, out var v) && v is List<object?> l ? l : [];
}

/// <summary>Reads Thrift's compact protocol (the encoding of Parquet's metadata and page headers).</summary>
internal ref struct ThriftReader(ReadOnlySpan<byte> data)
{
    private readonly ReadOnlySpan<byte> _data = data;

    public int Position { get; private set; }

    public ThriftStruct ReadStruct(int depth = 0)
    {
        if (depth > 64)
        {
            throw new InvalidDataException("Parquet metadata nests too deeply.");
        }

        var result = new ThriftStruct();
        short last = 0;
        while (true)
        {
            byte header = Next();
            if (header == 0)
            {
                return result;                                      // stop
            }

            int type = header & 0x0F;
            int delta = header >> 4;
            short id = delta != 0 ? (short)(last + delta) : (short)ZigZag(Varint());
            last = id;
            result.Fields[id] = type switch
            {
                1 => true,
                2 => false,
                _ => ReadValue(type, depth),
            };
        }
    }

    private object? ReadValue(int type, int depth)
    {
        switch (type)
        {
            case 1:
            case 2:
                return Next() == 1;                                 // a bool inside a list or map
            case 3:
                return (long)(sbyte)Next();
            case 4:
            case 5:
            case 6:
                return ZigZag(Varint());
            case 7:
                double d = BitConverter.ToDouble(_data.Slice(Position, 8));
                Position += 8;
                return d;
            case 8:
                int length = checked((int)Varint());
                var bytes = _data.Slice(Position, length).ToArray();
                Position += length;
                return bytes;
            case 9:
            case 10:
                byte header = Next();
                int count = header >> 4;
                if (count == 15)
                {
                    count = checked((int)Varint());
                }

                int elementType = header & 0x0F;
                var list = new List<object?>(Math.Min(count, 1 << 16));
                for (int i = 0; i < count; i++)
                {
                    list.Add(ReadValue(elementType, depth + 1));
                }

                return list;
            case 11:
                int size = checked((int)Varint());
                if (size == 0)
                {
                    return new List<object?>();
                }

                byte types = Next();
                var pairs = new List<object?>(size);
                for (int i = 0; i < size; i++)
                {
                    pairs.Add(new KeyValuePair<object?, object?>(ReadValue(types >> 4, depth + 1), ReadValue(types & 0x0F, depth + 1)));
                }

                return pairs;
            case 12:
                return ReadStruct(depth + 1);
            default:
                throw new InvalidDataException($"Unknown Thrift type {type} in Parquet metadata.");
        }
    }

    private byte Next()
    {
        if (Position >= _data.Length)
        {
            throw new InvalidDataException("Parquet metadata ends early.");
        }

        return _data[Position++];
    }

    private ulong Varint()
    {
        ulong value = 0;
        for (int shift = 0; shift < 64; shift += 7)
        {
            byte b = Next();
            value |= (ulong)(b & 0x7F) << shift;
            if ((b & 0x80) == 0)
            {
                return value;
            }
        }

        throw new InvalidDataException("Bad varint in Parquet metadata.");
    }

    private static long ZigZag(ulong n) => (long)(n >> 1) ^ -(long)(n & 1);
}
