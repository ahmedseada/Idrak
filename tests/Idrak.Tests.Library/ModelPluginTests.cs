// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

// GGUF files written for the tests (the GGUF and tokenizer groups' fixtures made in memory).
internal abstract partial class Tests
{
    // A GGUF (version 3) file: metadata (long, double, string, bool and arrays of them, and float, byte and sbyte arrays) and tensors of raw bytes.
    internal static void WriteGguf(string path, IEnumerable<KeyValuePair<string, object>> metadata, IReadOnlyList<(string Name, int Type, long[] Dimensions, byte[] Data)> tensors)
    {
        const int Alignment = 32;
        using var stream = new MemoryStream();
        using var w = new BinaryWriter(stream, Encoding.UTF8);
        void Str(string s)
        {
            var bytes = Encoding.UTF8.GetBytes(s);
            w.Write((ulong)bytes.Length);
            w.Write(bytes);
        }

        uint TypeOf(object value) => value switch
        {
            long => 11, ulong => 10, double => 12, string => 8, bool => 7, Array => 9,
            _ => throw new NotSupportedException(value.GetType().Name),
        };

        void Value(object value)
        {
            switch (value)
            {
                case long l: w.Write(l); break;
                case ulong u: w.Write(u); break;
                case double d: w.Write(d); break;
                case float f: w.Write(f); break;
                case byte b: w.Write(b); break;
                case sbyte b: w.Write(b); break;
                case string s: Str(s); break;
                case bool b: w.Write(b); break;
                case Array a:
                    w.Write(a switch { string[] => 8u, long[] => 11u, double[] => 12u, bool[] => 7u, float[] => 6u, byte[] => 0u, sbyte[] => 1u, _ => throw new NotSupportedException(a.GetType().Name) });
                    w.Write((ulong)a.Length);
                    foreach (var item in a)
                    {
                        Value(item!);
                    }

                    break;
            }
        }

        var entries = metadata.ToList();
        w.Write("GGUF"u8);
        w.Write(3u);
        w.Write((ulong)tensors.Count);
        w.Write((ulong)entries.Count);
        foreach (var (key, value) in entries)
        {
            Str(key);
            w.Write(TypeOf(value));
            Value(value);
        }

        long offset = 0;
        foreach (var (name, type, dimensions, data) in tensors)
        {
            Str(name);
            w.Write((uint)dimensions.Length);
            foreach (long d in dimensions)
            {
                w.Write((ulong)d);
            }

            w.Write((uint)type);
            w.Write((ulong)offset);
            offset += (data.Length + Alignment - 1) / Alignment * Alignment;
        }

        foreach (var (_, _, _, data) in tensors)
        {
            w.Write(new byte[(Alignment - stream.Length % Alignment) % Alignment]);
            w.Write(data);
        }

        w.Flush();
        File.WriteAllBytes(path, stream.ToArray());
    }
}
