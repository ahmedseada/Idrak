// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>The element type of a stored tensor.</summary>
public enum SafeTensorType
{
    /// <summary>32-bit float.</summary>
    F32,

    /// <summary>16-bit IEEE float.</summary>
    F16,

    /// <summary>bfloat16.</summary>
    BF16,
}

/// <summary>A tensor's entry in a safetensors file: its type, shape and byte range.</summary>
public sealed record SafeTensorInfo(string Name, SafeTensorType Type, int[] Shape, long Offset, long Length, string File)
{
    /// <summary>Number of elements.</summary>
    public long Count => Shape.Aggregate(1L, (a, b) => a * b);
}

/// <summary>
/// Reads safetensors weights: one file, or a sharded checkpoint (<c>model.safetensors.index.json</c> naming the file of
/// each tensor). Tensors are read one at a time, straight from disk, and converted to float32, or, for bfloat16 ones, also
/// copied as their 16-bit values (<see cref="ReadBFloat16"/>). Tensors of integer or boolean types (buffers a checkpoint
/// may carry beside its weights, such as position ids) are left out (<see cref="LeftOut"/>), not refused.
/// </summary>
public sealed class SafeTensorsReader : IDisposable, ITensorStore
{
    IEnumerable<string> ITensorStore.Names => _tensors.Keys;

    int[] ITensorStore.ShapeOf(string name) => _tensors[name].Shape;

    WeightFormat? ITensorStore.FormatOf(string name) => _tensors[name].Type switch
    {
        SafeTensorType.F32 => WeightFormat.Float32,
        SafeTensorType.F16 => WeightFormat.Float16,
        SafeTensorType.BF16 => WeightFormat.BFloat16,
        _ => null,
    };

    private readonly Dictionary<string, SafeTensorInfo> _tensors = [];
    private readonly Dictionary<string, string> _leftOut = new(StringComparer.Ordinal);
    private readonly Dictionary<string, FileStream> _files = [];
    private long _readTicks;                                            // Stopwatch ticks spent reading tensor bytes
    private long _bytesRead;                                            // bytes of tensor values read

    private SafeTensorsReader()
    {
    }

    /// <summary>The tensors, by name.</summary>
    public IReadOnlyDictionary<string, SafeTensorInfo> Tensors => _tensors;

    /// <summary>
    /// The tensors of integer or boolean types ("I64", "I32", "U8", "BOOL", ...), by name with their type: not weights
    /// (buffers such as position ids), so they are not among <see cref="Tensors"/> and are not read.
    /// </summary>
    public IReadOnlyDictionary<string, string> LeftOut => _leftOut;

    /// <summary>Metadata of the files (the "__metadata__" entries), merged.</summary>
    public Dictionary<string, string> Metadata { get; } = [];

    /// <summary>
    /// Opens a .safetensors file, a sharded checkpoint's index (.index.json), or a folder containing either
    /// (model.safetensors or model.safetensors.index.json).
    /// </summary>
    public static SafeTensorsReader Open(string path)
    {
        if (Directory.Exists(path))
        {
            string index = Path.Combine(path, "model.safetensors.index.json"), single = Path.Combine(path, "model.safetensors");
            path = File.Exists(index) ? index : File.Exists(single) ? single
                : Directory.GetFiles(path, "*.safetensors") is [var only] ? only
                : throw new FileNotFoundException($"{path} has no model.safetensors, model.safetensors.index.json or single .safetensors file.");
        }

        var reader = new SafeTensorsReader();
        try
        {
            if (path.EndsWith(".json", StringComparison.OrdinalIgnoreCase))
            {
                var map = JsonNode.Parse(File.ReadAllText(path))!["weight_map"]!.AsObject();
                string folder = Path.GetDirectoryName(Path.GetFullPath(path))!;
                foreach (var file in map.Select(p => (string)p.Value!).Distinct())
                {
                    reader.ReadHeader(Path.Combine(folder, file));
                }
            }
            else
            {
                reader.ReadHeader(path);
            }

            return reader;
        }
        catch
        {
            reader.Dispose();
            throw;
        }
    }

    /// <summary>Whether a tensor named <paramref name="name"/> exists.</summary>
    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <summary>The tensor <paramref name="name"/> as float32 values (row-major, as stored).</summary>
    public float[] Read(string name)
    {
        var info = _tensors.TryGetValue(name, out var t) ? t : throw Missing(name);
        // Positional reads (no shared stream position, so tensors can be read concurrently) straight into the result:
        // float32 bytes land in place; 16-bit types are widened chunk by chunk through one pooled buffer.
        int count = checked((int)info.Count);
        var handle = _files[info.File].SafeFileHandle;
        var values = GC.AllocateUninitializedArray<float>(count);
        if (info.Type == SafeTensorType.F32 && BitConverter.IsLittleEndian)
        {
            ReadTimed(handle, MemoryMarshal.AsBytes(values.AsSpan()), info.Offset);
            return values;
        }

        const int ChunkValues = 1 << 22;                                            // 8 MB of 16-bit values
        byte[] buffer = System.Buffers.ArrayPool<byte>.Shared.Rent((int)Math.Min(info.Length, 2L * ChunkValues));
        try
        {
            for (int first = 0; first < count; first += ChunkValues)
            {
                int n = Math.Min(ChunkValues, count - first);
                int width = info.Type == SafeTensorType.F32 ? 4 : 2;
                ReadTimed(handle, buffer.AsSpan(0, n * width), info.Offset + (long)first * width);
                DecodeInto(buffer, info.Type, values, first, n);
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }

        return values;
    }

    /// <inheritdoc cref="ITensorStore.SameValues"/>
    /// <remarks>
    /// Both tensors are read and widened a chunk at a time (1M values each, through pooled buffers) and compared chunk by
    /// chunk: the memory is a few chunks whatever the tensors' size, and the reading stops at the first chunk that differs.
    /// </remarks>
    public bool SameValues(string first, string second)
    {
        var a = _tensors.TryGetValue(first, out var t1) ? t1 : throw Missing(first);
        var b = _tensors.TryGetValue(second, out var t2) ? t2 : throw Missing(second);
        if (!a.Shape.AsSpan().SequenceEqual(b.Shape))
        {
            return false;
        }

        const int ChunkValues = 1 << 20;
        long count = a.Count;
        int chunk = (int)Math.Min(count, ChunkValues);
        int widthA = a.Type == SafeTensorType.F32 ? 4 : 2, widthB = b.Type == SafeTensorType.F32 ? 4 : 2;
        byte[] bytesA = System.Buffers.ArrayPool<byte>.Shared.Rent(Math.Max(1, chunk * widthA));
        byte[] bytesB = System.Buffers.ArrayPool<byte>.Shared.Rent(Math.Max(1, chunk * widthB));
        float[] valuesA = System.Buffers.ArrayPool<float>.Shared.Rent(Math.Max(1, chunk));
        float[] valuesB = System.Buffers.ArrayPool<float>.Shared.Rent(Math.Max(1, chunk));
        try
        {
            var handleA = _files[a.File].SafeFileHandle;
            var handleB = _files[b.File].SafeFileHandle;
            for (long start = 0; start < count; start += ChunkValues)
            {
                int n = (int)Math.Min(ChunkValues, count - start);
                ReadTimed(handleA, bytesA.AsSpan(0, n * widthA), a.Offset + start * widthA);
                ReadTimed(handleB, bytesB.AsSpan(0, n * widthB), b.Offset + start * widthB);
                DecodeInto(bytesA, a.Type, valuesA, 0, n);
                DecodeInto(bytesB, b.Type, valuesB, 0, n);
                if (!valuesA.AsSpan(0, n).SequenceEqual(valuesB.AsSpan(0, n)))
                {
                    return false;
                }
            }

            return true;
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(bytesA);
            System.Buffers.ArrayPool<byte>.Shared.Return(bytesB);
            System.Buffers.ArrayPool<float>.Shared.Return(valuesA);
            System.Buffers.ArrayPool<float>.Shared.Return(valuesB);
        }
    }

    /// <summary>The 2-D tensor <paramref name="name"/> [rows, columns] as float32 values transposed to [columns, rows].</summary>
    public float[] ReadTransposed(string name)
    {
        var info = _tensors.TryGetValue(name, out var t) ? t : throw Missing(name);
        if (info.Shape.Length != 2)
        {
            throw new ArgumentException($"'{name}' is [{string.Join(", ", info.Shape)}], not a matrix.", nameof(name));
        }

        if (!BitConverter.IsLittleEndian)
        {
            return Idrak.Abstraction.Devices.HostParallel.Transpose(Read(name), info.Shape[0], info.Shape[1]);
        }

        // Chunks of stored rows are read through one pooled buffer and scattered, widened, into the transposed result.
        int rows = info.Shape[0], columns = info.Shape[1], width = info.Type == SafeTensorType.F32 ? 4 : 2;
        var handle = _files[info.File].SafeFileHandle;
        var values = GC.AllocateUninitializedArray<float>(checked(rows * columns));
        int chunkRows = Math.Max(1, (1 << 22) / Math.Max(1, columns));
        byte[] buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(Math.Max(1, (int)Math.Min(info.Length, (long)chunkRows * columns * width)));
        try
        {
            for (int r0 = 0; r0 < rows; r0 += chunkRows)
            {
                int n = Math.Min(chunkRows, rows - r0), first = r0;
                ReadTimed(handle, buffer.AsSpan(0, n * columns * width), info.Offset + (long)r0 * columns * width);
                var type = info.Type;
                Idrak.Abstraction.Devices.HostParallel.For(columns, Math.Max(1, (1 << 14) / n), (c0, c1) =>
                {
                    const int Tile = 64;
                    for (int t0 = 0; t0 < n; t0 += Tile)
                    {
                        int t1 = Math.Min(n, t0 + Tile);
                        for (int c = c0; c < c1; c++)
                        {
                            var column = values.AsSpan(c * rows + first, n);
                            switch (type)
                            {
                                case SafeTensorType.F32:
                                    var floats = MemoryMarshal.Cast<byte, float>(buffer.AsSpan(0, n * columns * 4));
                                    for (int r = t0; r < t1; r++)
                                    {
                                        column[r] = floats[r * columns + c];
                                    }

                                    break;
                                case SafeTensorType.F16:
                                    var halves = MemoryMarshal.Cast<byte, Half>(buffer.AsSpan(0, n * columns * 2));
                                    for (int r = t0; r < t1; r++)
                                    {
                                        column[r] = (float)halves[r * columns + c];
                                    }

                                    break;
                                default:
                                    var bits = MemoryMarshal.Cast<byte, ushort>(buffer.AsSpan(0, n * columns * 2));
                                    var target = MemoryMarshal.Cast<float, uint>(column);
                                    for (int r = t0; r < t1; r++)
                                    {
                                        target[r] = (uint)bits[r * columns + c] << 16;
                                    }

                                    break;
                            }
                        }
                    }
                });
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }

        return values;
    }

    /// <inheritdoc cref="ITensorStore.ReadBFloat16"/>
    /// <exception cref="ArgumentException">The tensor is not a matrix, or the destination or stride cannot hold the result.</exception>
    public unsafe bool ReadBFloat16(string name, bool transposed, Span<ushort> destination, int stride)
    {
        var info = _tensors.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"No tensor named '{name}'.");
        if (info.Type != SafeTensorType.BF16 || !BitConverter.IsLittleEndian)
        {
            return false;
        }

        if (info.Shape.Length != 2)
        {
            throw new ArgumentException($"'{name}' is [{string.Join(", ", info.Shape)}], not a matrix.", nameof(name));
        }

        int rows = info.Shape[0], columns = info.Shape[1];
        int resultRows = transposed ? columns : rows, resultColumns = transposed ? rows : columns;
        if (stride < resultColumns)
        {
            throw new ArgumentOutOfRangeException(nameof(stride), stride, $"A row of the result holds {resultColumns} values.");
        }

        if (resultRows > 0 && destination.Length < (long)(resultRows - 1) * stride + resultColumns)
        {
            throw new ArgumentException($"{destination.Length} values cannot hold {resultRows} rows of {resultColumns} at a stride of {stride}.", nameof(destination));
        }

        if (rows == 0 || columns == 0)
        {
            return true;
        }

        // Chunks of stored rows are read through one pooled buffer and copied (or scattered, transposed) into the result
        // as 16-bit values: the stored order is never held whole, and nothing is widened to float32 and rounded back.
        var handle = _files[info.File].SafeFileHandle;
        int chunkRows = Math.Max(1, (1 << 22) / columns);                              // 8 MB of 16-bit values
        byte[] buffer = System.Buffers.ArrayPool<byte>.Shared.Rent((int)Math.Min(info.Length, (long)chunkRows * columns * 2));
        try
        {
            fixed (ushort* result = destination)
            {
                for (int r0 = 0; r0 < rows; r0 += chunkRows)
                {
                    int n = Math.Min(chunkRows, rows - r0);
                    ReadTimed(handle, buffer.AsSpan(0, n * columns * 2), info.Offset + (long)r0 * columns * 2);
                    PlaceBFloat16(buffer, n, columns, result, r0, stride, transposed);
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }

        return true;
    }

    /// <summary>The time spent reading tensor bytes from the files so far (all threads together).</summary>
    public TimeSpan ReadTime => Stopwatch.GetElapsedTime(0, Interlocked.Read(ref _readTicks));

    TimeSpan? ITensorStore.ReadTime => ReadTime;

    // n stored rows of `columns` bfloat16 values from the start of buffer, stored rows first.. of the tensor, into the
    // result: as its rows first.. (each at its stride), or transposed, as columns first.. of every result row. Every value
    // goes through CopyStoredBits (a NaN made quiet, as rounding its float32 copy does), in parallel over rows or columns.
    private static unsafe void PlaceBFloat16(byte[] buffer, int n, int columns, ushort* result, int first, int stride, bool transposed)
    {
        if (!transposed)
        {
            Idrak.Abstraction.Devices.HostParallel.For(n, Math.Max(1, (1 << 16) / columns), (r0, r1) =>
            {
                var bits = MemoryMarshal.Cast<byte, ushort>(buffer.AsSpan(0, n * columns * 2));
                for (int r = r0; r < r1; r++)
                {
                    BFloat16Weight.CopyStoredBits(bits.Slice(r * columns, columns), new Span<ushort>(result + (long)(first + r) * stride, columns));
                }
            });
            return;
        }

        // NaNs made quiet in place, then each result row (a stored column) gets the chunk's n values, tiles of stored rows
        // at a time so the strided reads stay in cache, as ReadTransposed scatters.
        Idrak.Abstraction.Devices.HostParallel.For(n * columns, 1 << 16, (i0, i1) =>
        {
            var bits = MemoryMarshal.Cast<byte, ushort>(buffer.AsSpan(i0 * 2, (i1 - i0) * 2));
            BFloat16Weight.CopyStoredBits(bits, bits);
        });
        Idrak.Abstraction.Devices.HostParallel.For(columns, Math.Max(1, (1 << 14) / n), (c0, c1) =>
        {
            var bits = MemoryMarshal.Cast<byte, ushort>(buffer.AsSpan(0, n * columns * 2));
            const int Tile = 64;
            for (int t0 = 0; t0 < n; t0 += Tile)
            {
                int t1 = Math.Min(n, t0 + Tile);
                for (int c = c0; c < c1; c++)
                {
                    ushort* row = result + (long)c * stride + first;
                    for (int r = t0; r < t1; r++)
                    {
                        row[r] = bits[r * columns + c];
                    }
                }
            }
        });
    }

    // ReadAt, its time added to ReadTime (tensors may be read on several threads at once).
    private void ReadTimed(Microsoft.Win32.SafeHandles.SafeFileHandle handle, Span<byte> destination, long offset)
    {
        long start = Stopwatch.GetTimestamp();
        ReadAt(handle, destination, offset);
        Interlocked.Add(ref _readTicks, Stopwatch.GetTimestamp() - start);
        Interlocked.Add(ref _bytesRead, destination.Length);
    }

    /// <summary>The bytes of tensor values read from the files so far (each tensor's own byte range only).</summary>
    public long BytesRead => Interlocked.Read(ref _bytesRead);

    long? ITensorStore.BytesRead => BytesRead;

    private static void ReadAt(Microsoft.Win32.SafeHandles.SafeFileHandle handle, Span<byte> destination, long offset)
    {
        while (!destination.IsEmpty)
        {
            int read = RandomAccess.Read(handle, destination, offset);
            if (read <= 0)
            {
                throw new EndOfStreamException("The safetensors file ends before the tensor data.");
            }

            destination = destination[read..];
            offset += read;
        }
    }

    private void ReadHeader(string file)
    {
        var stream = File.OpenRead(file);
        _files[file] = stream;
        Span<byte> size = stackalloc byte[8];
        stream.ReadExactly(size);
        long headerLength = BinaryPrimitives.ReadInt64LittleEndian(size);
        if (headerLength <= 0 || headerLength > 100_000_000)
        {
            throw new InvalidDataException($"{file} is not a safetensors file (header size {headerLength}).");
        }

        var header = new byte[headerLength];
        stream.ReadExactly(header);
        long dataStart = 8 + headerLength;
        foreach (var (name, node) in JsonNode.Parse(header)!.AsObject())
        {
            if (name == "__metadata__")
            {
                foreach (var (key, value) in node!.AsObject())
                {
                    Metadata[key] = value?.ToString() ?? "";
                }

                continue;
            }

            string dtype = (string)node!["dtype"]!;
            if (dtype is "I64" or "I32" or "I16" or "I8" or "U64" or "U32" or "U16" or "U8" or "BOOL")
            {
                _leftOut[name] = dtype;                                     // a buffer (position ids, a mask), not a weight
                continue;
            }

            var type = dtype switch
            {
                "F32" => SafeTensorType.F32,
                "F16" => SafeTensorType.F16,
                "BF16" => SafeTensorType.BF16,
                _ => throw new NotSupportedException($"Tensor '{name}' in {file} has type {dtype}; F32, F16 and BF16 are supported."),
            };
            int[] shape = [.. node["shape"]!.AsArray().Select(d => (int)d!)];
            var offsets = node["data_offsets"]!.AsArray();
            long begin = (long)offsets[0]!, end = (long)offsets[1]!;
            _tensors[name] = new SafeTensorInfo(name, type, shape, dataStart + begin, end - begin, file);
        }
    }

    private Exception Missing(string name) => _leftOut.TryGetValue(name, out string? type)
        ? new NotSupportedException($"The tensor '{name}' is of type {type}, a buffer the reader leaves out (it reads F32, F16 and BF16 tensors).")
        : new KeyNotFoundException($"No tensor named '{name}'.");

    internal static float[] Decode(byte[] bytes, SafeTensorType type, int count)
    {
        var values = GC.AllocateUninitializedArray<float>(count);
        DecodeInto(bytes, type, values, 0, count);
        return values;
    }

    // Widens `count` stored values from the start of bytes into values[offset..] (all cores for large chunks).
    private static void DecodeInto(byte[] bytes, SafeTensorType type, float[] values, int offset, int count)
    {
        switch (type)
        {
            case SafeTensorType.F32:
                StoredFloats.ReadFloat32(bytes.AsSpan(0, count * 4), values.AsSpan(offset, count));
                break;
            case SafeTensorType.F16:
                Idrak.Abstraction.Devices.HostParallel.For(count, 1 << 16, (first, last) =>
                {
                    for (int i = first; i < last; i++)
                    {
                        values[offset + i] = (float)BinaryPrimitives.ReadHalfLittleEndian(bytes.AsSpan(i * 2));
                    }
                });
                break;
            default:
                // bfloat16 is the top half of a float32.
                Idrak.Abstraction.Devices.HostParallel.For(count, 1 << 16, (first, last) =>
                    StoredFloats.WidenBFloat16(bytes.AsSpan(first * 2, (last - first) * 2), values.AsSpan(offset + first, last - first)));
                break;
        }
    }

    /// <inheritdoc />
    public void Dispose()
    {
        foreach (var stream in _files.Values)
        {
            stream.Dispose();
        }

        _files.Clear();
    }
}

/// <summary>Writes safetensors files (for example fine-tuned weights, for other tools).</summary>
public static class SafeTensorsWriter
{
    /// <summary>Writes <paramref name="tensors"/> (name, shape, values) to <paramref name="path"/> in <paramref name="type"/>.</summary>
    public static void Write(string path, IEnumerable<(string Name, int[] Shape, float[] Values)> tensors, SafeTensorType type = SafeTensorType.BF16,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        var list = tensors.ToList();
        foreach (var (name, shape, values) in list)
        {
            Check(name, shape, values);
        }

        Write(path, [.. list.Select(t => (t.Name, t.Shape, (Func<float[]>)(() => t.Values)))], type, metadata);
    }

    /// <summary>
    /// Writes tensors whose values are produced one at a time by <c>Values</c>, as each is written: the header comes from
    /// the shapes, so only one tensor's values (and one chunk of its bytes) exist at once (a model's weights are never all
    /// on the host together).
    /// </summary>
    public static void Write(string path, IReadOnlyList<(string Name, int[] Shape, Func<float[]> Values)> tensors, SafeTensorType type,
        IReadOnlyDictionary<string, string>? metadata = null)
    {
        int size = type == SafeTensorType.F32 ? 4 : 2;
        var header = new JsonObject();
        if (metadata is not null)
        {
            var meta = new JsonObject();
            foreach (var (key, value) in metadata)
            {
                meta[key] = value;
            }

            header["__metadata__"] = meta;
        }

        long offset = 0;
        foreach (var (name, shape, _) in tensors)
        {
            long count = shape.Aggregate(1L, (a, b) => a * b);
            header[name] = new JsonObject
            {
                ["dtype"] = type.ToString(),
                ["shape"] = new JsonArray([.. shape.Select(d => (JsonNode)d)]),
                ["data_offsets"] = new JsonArray(offset, offset + count * size),
            };
            offset += count * size;
        }

        var headerJson = new System.Buffers.ArrayBufferWriter<byte>();
        using (var writer = new Utf8JsonWriter(headerJson))                   // the UTF-8 of header.ToJsonString()
        {
            header.WriteTo(writer);
        }

        var headerBytes = headerJson.WrittenSpan;
        int padding = (8 - headerBytes.Length % 8) % 8;                      // the data starts 8-byte aligned
        using var stream = File.Create(path);
        Span<byte> length = stackalloc byte[8];
        BinaryPrimitives.WriteInt64LittleEndian(length, headerBytes.Length + padding);
        stream.Write(length);
        stream.Write(headerBytes);
        length.Fill((byte)' ');
        stream.Write(length[..padding]);
        const int ChunkValues = 1 << 22;
        byte[] bytes = System.Buffers.ArrayPool<byte>.Shared.Rent(ChunkValues * size);
        try
        {
            foreach (var (name, shape, produce) in tensors)
            {
                var values = produce();
                Check(name, shape, values);
                for (int first = 0; first < values.Length; first += ChunkValues)
                {
                    int n = Math.Min(ChunkValues, values.Length - first), start = first;
                    Idrak.Abstraction.Devices.HostParallel.For(n, 1 << 16, (lo, hi) => Encode(values, start + lo, bytes, lo, hi - lo, type));
                    stream.Write(bytes, 0, n * size);
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(bytes);
        }
    }

    private static void Check(string name, int[] shape, float[] values)
    {
        if (values.Length != shape.Aggregate(1L, (a, b) => a * b))
        {
            throw new ArgumentException($"'{name}': {values.Length} values do not fill [{string.Join(", ", shape)}].");
        }
    }

    // values[from..from+count] as little-endian `type` into bytes, starting at value index `at` (bfloat16 rounds to nearest
    // even; NaN stays NaN).
    private static void Encode(float[] values, int from, byte[] bytes, int at, int count, SafeTensorType type)
    {
        for (int k = 0; k < count; k++)
        {
            float value = values[from + k];
            int i = at + k;
            switch (type)
            {
                case SafeTensorType.F32:
                    BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), value);
                    break;
                case SafeTensorType.F16:
                    BinaryPrimitives.WriteHalfLittleEndian(bytes.AsSpan(i * 2), (Half)value);
                    break;
                default:
                    uint bits = BitConverter.SingleToUInt32Bits(value);
                    ushort b16 = float.IsNaN(value) ? (ushort)0x7FC0 : (ushort)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
                    BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(i * 2), b16);
                    break;
            }
        }
    }
}

// Stored little-endian floats read into float32 (the checkpoint readers and the weight codecs): on a little-endian machine
// float32 bytes are copied as they are and bfloat16 is widened whole vectors at a time (its bits are the high half of a
// float32) with a scalar tail; elsewhere value by value. The same values either way.
internal static class StoredFloats
{
    public static void ReadFloat32(ReadOnlySpan<byte> bytes, Span<float> values)
    {
        if (BitConverter.IsLittleEndian)
        {
            MemoryMarshal.Cast<byte, float>(bytes[..(values.Length * 4)]).CopyTo(values);
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes[(i * 4)..]);
        }
    }

    public static void WidenBFloat16(ReadOnlySpan<byte> bytes, Span<float> values)
    {
        int i = 0;
        if (BitConverter.IsLittleEndian && System.Numerics.Vector.IsHardwareAccelerated)
        {
            var halves = MemoryMarshal.Cast<byte, ushort>(bytes[..(values.Length * 2)]);
            var bits = MemoryMarshal.Cast<float, uint>(values);
            for (; i <= values.Length - System.Numerics.Vector<ushort>.Count; i += System.Numerics.Vector<ushort>.Count)
            {
                System.Numerics.Vector.Widen(new System.Numerics.Vector<ushort>(halves[i..]), out var low, out var high);
                (low << 16).CopyTo(bits[i..]);
                (high << 16).CopyTo(bits[(i + System.Numerics.Vector<uint>.Count)..]);
            }
        }

        for (; i < values.Length; i++)
        {
            values[i] = BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(bytes[(i * 2)..]) << 16);
        }
    }
}
