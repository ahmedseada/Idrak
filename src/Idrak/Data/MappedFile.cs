// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Text;
using System.Text.RegularExpressions;

namespace Idrak.Data;

/// <summary>A file mapped read-only into memory: the operating system pages it in as it is read, so its size is not a limit.</summary>
internal sealed unsafe class MappedFile : IDisposable
{
    private readonly MemoryMappedFile _file;
    private readonly MemoryMappedViewAccessor _view;
    private byte* _pointer;

    public MappedFile(string path)
    {
        Length = new FileInfo(path).Length;
        if (Length == 0)
        {
            throw new InvalidDataException($"{path} is empty.");
        }

        _file = MemoryMappedFile.CreateFromFile(path, FileMode.Open, null, 0, MemoryMappedFileAccess.Read);
        _view = _file.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);
        _view.SafeMemoryMappedViewHandle.AcquirePointer(ref _pointer);
        _pointer += _view.PointerOffset;
        Path = path;
    }

    /// <summary>The file.</summary>
    public string Path { get; }

    /// <summary>Its size in bytes.</summary>
    public long Length { get; }

    /// <summary><paramref name="length"/> bytes from <paramref name="offset"/>.</summary>
    public ReadOnlySpan<byte> Bytes(long offset, int length)
    {
        ObjectDisposedException.ThrowIf(_pointer == null, this);
        if (offset < 0 || length < 0 || offset + length > Length)
        {
            throw new ArgumentOutOfRangeException(nameof(offset), $"Bytes {offset} to {offset + length} are outside {Path} ({Length} bytes).");
        }

        return new ReadOnlySpan<byte>(_pointer + offset, length);
    }

    public void Dispose()
    {
        if (_pointer != null)
        {
            _pointer = null;
            _view.SafeMemoryMappedViewHandle.ReleasePointer();
        }

        _view.Dispose();
        _file.Dispose();
    }
}

/// <summary>The element types of packed numbers in files (.npy arrays, token files), converted to float32 when read.</summary>
internal enum ElementType
{
    Float32, Float64, Float16, Int8, UInt8, Int16, UInt16, Int32, UInt32, Int64, UInt64, Bool,
}

internal static class Elements
{
    public static int Size(ElementType type) => type switch
    {
        ElementType.Int8 or ElementType.UInt8 or ElementType.Bool => 1,
        ElementType.Float16 or ElementType.Int16 or ElementType.UInt16 => 2,
        ElementType.Float32 or ElementType.Int32 or ElementType.UInt32 => 4,
        _ => 8,
    };

    /// <summary>Converts little-endian values of <paramref name="type"/> to float32.</summary>
    public static void ToFloat(ReadOnlySpan<byte> bytes, ElementType type, Span<float> destination)
    {
        int n = destination.Length;
        switch (type)
        {
            case ElementType.Float32:
                for (int i = 0; i < n; i++)
                {
                    destination[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes[(i * 4)..]);
                }

                break;
            case ElementType.UInt16:
                for (int i = 0; i < n; i++)
                {
                    destination[i] = BinaryPrimitives.ReadUInt16LittleEndian(bytes[(i * 2)..]);
                }

                break;
            case ElementType.Int32:
                for (int i = 0; i < n; i++)
                {
                    destination[i] = BinaryPrimitives.ReadInt32LittleEndian(bytes[(i * 4)..]);
                }

                break;
            case ElementType.UInt8 or ElementType.Bool:
                for (int i = 0; i < n; i++)
                {
                    destination[i] = bytes[i];
                }

                break;
            default:
                int size = Size(type);
                for (int i = 0; i < n; i++)
                {
                    var b = bytes.Slice(i * size, size);
                    destination[i] = type switch
                    {
                        ElementType.Float64 => (float)BinaryPrimitives.ReadDoubleLittleEndian(b),
                        ElementType.Float16 => (float)BinaryPrimitives.ReadHalfLittleEndian(b),
                        ElementType.Int8 => (sbyte)b[0],
                        ElementType.Int16 => BinaryPrimitives.ReadInt16LittleEndian(b),
                        ElementType.UInt32 => BinaryPrimitives.ReadUInt32LittleEndian(b),
                        ElementType.Int64 => BinaryPrimitives.ReadInt64LittleEndian(b),
                        _ => BinaryPrimitives.ReadUInt64LittleEndian(b),
                    };
                }

                break;
        }
    }
}

/// <summary>A NumPy .npy array file's header: element type, shape, and where the data starts.</summary>
internal sealed record NpyHeader(ElementType Type, int[] Shape, long DataOffset)
{
    /// <summary>Whether the file starts with the .npy magic.</summary>
    public static bool IsNpy(ReadOnlySpan<byte> start) => start.Length >= 6 && start[0] == 0x93 && start[1..6].SequenceEqual("NUMPY"u8);

    /// <summary>
    /// Reads the header (format versions 1 to 3): <c>{'descr': '&lt;f4', 'fortran_order': False, 'shape': (60000, 28, 28), }</c>.
    /// Little-endian numbers and booleans in C order are read; a big-endian, Fortran-order or structured array is an error.
    /// </summary>
    public static NpyHeader Read(MappedFile file)
    {
        var start = file.Bytes(0, (int)Math.Min(file.Length, 12));
        if (!IsNpy(start) || start.Length < 10)
        {
            throw new InvalidDataException($"{file.Path} is not a .npy file.");
        }

        int major = start[6];
        int headerLength = major == 1 ? BinaryPrimitives.ReadUInt16LittleEndian(start[8..]) : BinaryPrimitives.ReadInt32LittleEndian(start[8..]);
        int prefix = major == 1 ? 10 : 12;
        string text = (major == 3 ? Encoding.UTF8 : Encoding.Latin1).GetString(file.Bytes(prefix, headerLength));
        string descr = Field(text, "descr", "'([^']*)'") ?? throw new InvalidDataException($"{file.Path}: no 'descr' in the header {text}.");
        string order = Field(text, "fortran_order", "(True|False)") ?? "False";
        string shape = Field(text, "shape", @"\(([^)]*)\)") ?? throw new InvalidDataException($"{file.Path}: no 'shape' in the header {text}.");
        if (order == "True")
        {
            throw new InvalidDataException($"{file.Path} is in Fortran order; save it in C order (numpy.ascontiguousarray).");
        }

        var type = descr switch
        {
            "<f4" or "=f4" => ElementType.Float32,
            "<f8" or "=f8" => ElementType.Float64,
            "<f2" or "=f2" => ElementType.Float16,
            "|i1" or "<i1" or "=i1" => ElementType.Int8,
            "|u1" or "<u1" or "=u1" => ElementType.UInt8,
            "|b1" or "<b1" => ElementType.Bool,
            "<i2" or "=i2" => ElementType.Int16,
            "<u2" or "=u2" => ElementType.UInt16,
            "<i4" or "=i4" => ElementType.Int32,
            "<u4" or "=u4" => ElementType.UInt32,
            "<i8" or "=i8" => ElementType.Int64,
            "<u8" or "=u8" => ElementType.UInt64,
            _ => throw new InvalidDataException($"{file.Path}: element type '{descr}' is not read (little-endian numbers and booleans are)."),
        };
        int[] dims = [.. shape.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(d => int.Parse(d, CultureInfo.InvariantCulture))];
        long offset = prefix + headerLength;
        long expected = dims.Aggregate(1L, (a, b) => a * b) * Elements.Size(type);
        if (offset + expected > file.Length)
        {
            throw new InvalidDataException($"{file.Path}: the shape ({shape}) needs {expected} bytes of data; the file has {file.Length - offset}.");
        }

        return new NpyHeader(type, dims, offset);
    }

    private static string? Field(string header, string name, string value) =>
        Regex.Match(header, $"'{name}'\\s*:\\s*{value}") is { Success: true } m ? m.Groups[1].Value : null;
}
