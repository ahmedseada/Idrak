// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

namespace Idrak.Datasets;

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

    /// <summary>Little-endian float32 values from <paramref name="offset"/> into <paramref name="destination"/>.</summary>
    public void Floats(long offset, Span<float> destination)
    {
        var bytes = Bytes(offset, checked(destination.Length * sizeof(float)));
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.Cast<byte, float>(bytes).CopyTo(destination);
            return;
        }

        for (int i = 0; i < destination.Length; i++)
        {
            destination[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes[(i * 4)..]);
        }
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

/// <summary>A NumPy .npy array file's header of float32 values: shape, and where the data starts.</summary>
internal sealed record NpyHeader(int[] Shape, long DataOffset)
{
    /// <summary>Whether the file starts with the .npy magic.</summary>
    public static bool IsNpy(ReadOnlySpan<byte> start) => start.Length >= 6 && start[0] == 0x93 && start[1..6].SequenceEqual("NUMPY"u8);

    /// <summary>
    /// Reads the header (format versions 1 to 3): <c>{'descr': '&lt;f4', 'fortran_order': False, 'shape': (60000, 28, 28), }</c>.
    /// Little-endian float32 values in C order are read; another element type is refused (not on the new design yet), and a
    /// Fortran-order array is an error.
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

        if (descr is not ("<f4" or "=f4"))
        {
            throw new NotSupportedException($"{file.Path}: element type '{descr}': type not on the new design yet.");
        }

        int[] dims = [.. shape.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).Select(d => int.Parse(d, CultureInfo.InvariantCulture))];
        long offset = prefix + headerLength;
        long expected = dims.Aggregate(1L, (a, b) => a * b) * sizeof(float);
        if (offset + expected > file.Length)
        {
            throw new InvalidDataException($"{file.Path}: the shape ({shape}) needs {expected} bytes of data; the file has {file.Length - offset}.");
        }

        return new NpyHeader(dims, offset);
    }

    private static string? Field(string header, string name, string value) =>
        Regex.Match(header, $"'{name}'\\s*:\\s*{value}") is { Success: true } m ? m.Groups[1].Value : null;
}
