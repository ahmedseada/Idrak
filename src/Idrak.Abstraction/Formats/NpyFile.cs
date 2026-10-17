// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Abstraction.Formats;

/// <summary>NumPy .npy files: writes version 1.0 little-endian float32 arrays (readable with numpy.load) and reads float32 and int64 ones in C order.</summary>
public static class NpyFile
{
    /// <summary>A little-endian float32 array in C order, and its shape.</summary>
    public static (float[] Values, int[] Shape) ReadFloat32(string path)
    {
        var (bytes, start, shape) = Read(path, "<f4");
        var values = new float[(bytes.Length - start) / 4];
        var source = bytes.AsSpan(start, values.Length * 4);
        if (BitConverter.IsLittleEndian)
        {
            source.CopyTo(MemoryMarshal.AsBytes(values.AsSpan()));
        }
        else
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = BinaryPrimitives.ReadSingleLittleEndian(source[(i * 4)..]);
            }
        }

        return (values, shape);
    }

    /// <summary>A little-endian int64 array in C order, and its shape.</summary>
    public static (long[] Values, int[] Shape) ReadInt64(string path)
    {
        var (bytes, start, shape) = Read(path, "<i8");
        var values = new long[(bytes.Length - start) / 8];
        var source = bytes.AsSpan(start, values.Length * 8);
        if (BitConverter.IsLittleEndian)
        {
            source.CopyTo(MemoryMarshal.AsBytes(values.AsSpan()));
        }
        else
        {
            for (int i = 0; i < values.Length; i++)
            {
                values[i] = BinaryPrimitives.ReadInt64LittleEndian(source[(i * 8)..]);
            }
        }

        return (values, shape);
    }

    private static (byte[] Bytes, int Start, int[] Shape) Read(string path, string type)
    {
        var bytes = File.ReadAllBytes(path);
        if (bytes.Length < 12 || bytes[0] != 0x93 || Encoding.ASCII.GetString(bytes, 1, 5) != "NUMPY")
        {
            throw new InvalidDataException($"{path} is not a .npy file.");
        }

        int major = bytes[6];
        int length = major == 1 ? BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(8)) : BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(8));
        int start = (major == 1 ? 10 : 12) + length;
        string header = Encoding.ASCII.GetString(bytes, major == 1 ? 10 : 12, length);
        if (!header.Contains($"'descr': '{type}'", StringComparison.Ordinal) || !header.Contains("'fortran_order': False", StringComparison.Ordinal))
        {
            throw new InvalidDataException($"{path}: expected a {type} array in C order, found {header.Trim()}");
        }

        var match = System.Text.RegularExpressions.Regex.Match(header, @"'shape':\s*\(([^)]*)\)");
        int[] shape = [.. match.Groups[1].Value.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
            .Select(d => int.Parse(d, System.Globalization.CultureInfo.InvariantCulture))];
        return (bytes, start, shape);
    }

    /// <summary>Writes <paramref name="rows"/> as a [rows, columns] float32 array.</summary>
    public static void Write(string path, IReadOnlyList<float[]> rows, int columns)
    {
        string header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows.Count}, {columns}), }}";
        int total = 10 + header.Length + 1;
        header = header.PadRight(header.Length + (64 - total % 64) % 64) + "\n";   // the data starts on a 64-byte boundary
        using var stream = File.Create(path);
        WriteHeader(stream, header);
        foreach (var row in rows)
        {
            WriteFloat32(stream, row);
        }
    }

    /// <summary>Writes <paramref name="values"/> as a float32 array of <paramref name="shape"/> in C order.</summary>
    public static void Write(string path, ReadOnlySpan<float> values, IReadOnlyList<int> shape)
    {
        string dims = shape.Count == 1 ? $"{shape[0]}," : string.Join(", ", shape);
        string header = $"{{'descr': '<f4', 'fortran_order': False, 'shape': ({dims}), }}";
        int total = 10 + header.Length + 1;
        header = header.PadRight(header.Length + (64 - total % 64) % 64) + "\n";   // the data starts on a 64-byte boundary
        using var stream = File.Create(path);
        WriteHeader(stream, header);
        WriteFloat32(stream, values);
    }

    /// <summary>The magic, version 1.0, the header's length (little-endian) and the header (padded, ending in a newline).</summary>
    public static void WriteHeader(Stream stream, string header)
    {
        Span<byte> start = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y', 1, 0, 0, 0];
        BinaryPrimitives.WriteUInt16LittleEndian(start[8..], (ushort)header.Length);
        stream.Write(start);
        stream.Write(Encoding.ASCII.GetBytes(header));
    }

    /// <summary>
    /// <paramref name="values"/> as little-endian float32 bytes: one block write on a little-endian machine, converted
    /// through a pooled buffer on a big-endian one (the file's byte order never follows the machine's).
    /// </summary>
    public static void WriteFloat32(Stream stream, ReadOnlySpan<float> values)
    {
        if (BitConverter.IsLittleEndian)
        {
            stream.Write(MemoryMarshal.AsBytes(values));
            return;
        }

        byte[] buffer = ArrayPool<byte>.Shared.Rent(Math.Min(values.Length, 16 * 1024) * 4);
        try
        {
            int per = buffer.Length / 4;
            for (int i = 0; i < values.Length; i += per)
            {
                int n = Math.Min(per, values.Length - i);
                for (int k = 0; k < n; k++)
                {
                    BinaryPrimitives.WriteSingleLittleEndian(buffer.AsSpan(k * 4), values[i + k]);
                }

                stream.Write(buffer, 0, n * 4);
            }
        }
        finally
        {
            ArrayPool<byte>.Shared.Return(buffer);
        }
    }
}
