// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Models;

/// <summary>A float32 tensor's entry in a safetensors file: its shape and byte range.</summary>
public sealed record SafeTensorInfo(string Name, int[] Shape, long Offset, long Length, string File)
{
    /// <summary>Number of elements.</summary>
    public long Count => Shape.Aggregate(1L, (a, b) => a * b);
}

/// <summary>
/// Reads safetensors weights: one file, or a sharded checkpoint (<c>model.safetensors.index.json</c> naming the file of
/// each tensor). Float32 tensors are read one at a time, straight from disk. A tensor stored in any other type is left out
/// (<see cref="LeftOut"/>) and refused when read: its type is not on the new design yet.
/// </summary>
public sealed class SafeTensorsReader : IDisposable, ITensorStore
{
    IEnumerable<string> ITensorStore.Names => _tensors.Keys;

    int[] ITensorStore.ShapeOf(string name) => _tensors[name].Shape;

    private readonly Dictionary<string, SafeTensorInfo> _tensors = [];
    private readonly Dictionary<string, string> _leftOut = new(StringComparer.Ordinal);
    private readonly Dictionary<string, FileStream> _files = [];
    private long _readTicks;                                            // Stopwatch ticks spent reading tensor bytes
    private long _bytesRead;                                            // bytes of tensor values read

    private SafeTensorsReader()
    {
    }

    /// <summary>The float32 tensors, by name.</summary>
    public IReadOnlyDictionary<string, SafeTensorInfo> Tensors => _tensors;

    /// <summary>
    /// The tensors stored in any type but float32, by name with their stored type ("BF16", "I64", ...): not among
    /// <see cref="Tensors"/>, and refused when read (their type is not on the new design yet).
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

    /// <summary>Whether a float32 tensor named <paramref name="name"/> exists.</summary>
    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <summary>The tensor <paramref name="name"/> as float32 values (row-major, as stored).</summary>
    public float[] Read(string name)
    {
        var info = _tensors.TryGetValue(name, out var t) ? t : throw Missing(name);
        // A positional read (no shared stream position, so tensors can be read concurrently) straight into the result.
        var values = GC.AllocateUninitializedArray<float>(checked((int)info.Count));
        ReadFloats(_files[info.File].SafeFileHandle, values, info.Offset);
        return values;
    }

    /// <inheritdoc cref="ITensorStore.ReadRows"/>
    /// <remarks>The rows' byte range alone is read (one positional read).</remarks>
    public bool ReadRows(string name, int first, int count, Span<float> destination)
    {
        var info = _tensors.TryGetValue(name, out var t) ? t : throw Missing(name);
        if (info.Shape.Length == 0)
        {
            return false;
        }

        long rowValues = info.Count / Math.Max(1, info.Shape[0]);
        if (first < 0 || count < 0 || first + count > info.Shape[0] || destination.Length < rowValues * count)
        {
            throw new ArgumentOutOfRangeException(nameof(count), $"Rows [{first}, {first + count}) of '{name}' [{string.Join(", ", info.Shape)}] into {destination.Length} values.");
        }

        int values = checked((int)(rowValues * count));
        ReadFloats(_files[info.File].SafeFileHandle, destination[..values], info.Offset + first * rowValues * sizeof(float));
        return true;
    }

    /// <inheritdoc cref="ITensorStore.IdentityOf"/>
    /// <remarks>The file's full path, its size and time of last change (UTC ticks), and the tensor's byte range in it.</remarks>
    public string? IdentityOf(string name)
    {
        if (!_tensors.TryGetValue(name, out var info))
        {
            return null;
        }

        var file = new FileInfo(info.File);
        return string.Create(CultureInfo.InvariantCulture, $"safetensors|{file.FullName}|{file.Length}|{file.LastWriteTimeUtc.Ticks}|{info.Offset}|{info.Length}");
    }

    /// <inheritdoc cref="ITensorStore.SameValues"/>
    /// <remarks>
    /// Both tensors are read a chunk at a time (1M values each, through pooled buffers) and compared chunk by chunk: the
    /// memory is a few chunks whatever the tensors' size, and the reading stops at the first chunk that differs.
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
        float[] valuesA = System.Buffers.ArrayPool<float>.Shared.Rent(Math.Max(1, chunk));
        float[] valuesB = System.Buffers.ArrayPool<float>.Shared.Rent(Math.Max(1, chunk));
        try
        {
            var handleA = _files[a.File].SafeFileHandle;
            var handleB = _files[b.File].SafeFileHandle;
            for (long start = 0; start < count; start += ChunkValues)
            {
                int n = (int)Math.Min(ChunkValues, count - start);
                ReadFloats(handleA, valuesA.AsSpan(0, n), a.Offset + start * sizeof(float));
                ReadFloats(handleB, valuesB.AsSpan(0, n), b.Offset + start * sizeof(float));
                if (!valuesA.AsSpan(0, n).SequenceEqual(valuesB.AsSpan(0, n)))
                {
                    return false;
                }
            }

            return true;
        }
        finally
        {
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

        // Chunks of stored rows are read through one pooled buffer and scattered into the transposed result.
        int rows = info.Shape[0], columns = info.Shape[1];
        var handle = _files[info.File].SafeFileHandle;
        var values = GC.AllocateUninitializedArray<float>(checked(rows * columns));
        int chunkRows = Math.Max(1, (1 << 22) / Math.Max(1, columns));
        float[] buffer = System.Buffers.ArrayPool<float>.Shared.Rent(Math.Max(1, (int)Math.Min(info.Count, (long)chunkRows * columns)));
        try
        {
            for (int r0 = 0; r0 < rows; r0 += chunkRows)
            {
                int n = Math.Min(chunkRows, rows - r0), first = r0;
                ReadFloats(handle, buffer.AsSpan(0, n * columns), info.Offset + (long)r0 * columns * sizeof(float));
                Idrak.Abstraction.Devices.HostParallel.For(columns, Math.Max(1, (1 << 14) / n), (c0, c1) =>
                {
                    const int Tile = 64;
                    for (int t0 = 0; t0 < n; t0 += Tile)
                    {
                        int t1 = Math.Min(n, t0 + Tile);
                        for (int c = c0; c < c1; c++)
                        {
                            var column = values.AsSpan(c * rows + first, n);
                            for (int r = t0; r < t1; r++)
                            {
                                column[r] = buffer[r * columns + c];
                            }
                        }
                    }
                });
            }
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(buffer);
        }

        return values;
    }

    /// <summary>The time spent reading tensor bytes from the files so far (all threads together).</summary>
    public TimeSpan ReadTime => Stopwatch.GetElapsedTime(0, Interlocked.Read(ref _readTicks));

    TimeSpan? ITensorStore.ReadTime => ReadTime;

    // Little-endian float32 values at offset into values (read in place, then their bytes turned on a big-endian
    // machine), the time added to ReadTime (tensors may be read on several threads at once).
    private void ReadFloats(Microsoft.Win32.SafeHandles.SafeFileHandle handle, Span<float> values, long offset)
    {
        long start = Stopwatch.GetTimestamp();
        var destination = MemoryMarshal.AsBytes(values);
        ReadAt(handle, destination, offset);
        Interlocked.Add(ref _readTicks, Stopwatch.GetTimestamp() - start);
        Interlocked.Add(ref _bytesRead, destination.Length);
        if (!BitConverter.IsLittleEndian)
        {
            var bits = MemoryMarshal.Cast<float, uint>(values);
            BinaryPrimitives.ReverseEndianness(bits, bits);
        }
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
            if (dtype != "F32")
            {
                _leftOut[name] = dtype;                                     // not on the new design yet: refused when read
                continue;
            }

            int[] shape = [.. node["shape"]!.AsArray().Select(d => (int)d!)];
            var offsets = node["data_offsets"]!.AsArray();
            long begin = (long)offsets[0]!, end = (long)offsets[1]!;
            _tensors[name] = new SafeTensorInfo(name, shape, dataStart + begin, end - begin, file);
        }
    }

    private Exception Missing(string name) => _leftOut.TryGetValue(name, out string? type)
        ? new NotSupportedException($"The tensor '{name}' is stored as {type}: type not on the new design yet.")
        : new KeyNotFoundException($"No tensor named '{name}'.");

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

/// <summary>Writes safetensors files of float32 tensors (for example fine-tuned weights, for other tools).</summary>
public static class SafeTensorsWriter
{
    /// <summary>Writes <paramref name="tensors"/> (name, shape, values) to <paramref name="path"/>.</summary>
    public static void Write(string path, IEnumerable<(string Name, int[] Shape, float[] Values)> tensors, IReadOnlyDictionary<string, string>? metadata = null)
    {
        var list = tensors.ToList();
        foreach (var (name, shape, values) in list)
        {
            Check(name, shape, values);
        }

        Write(path, [.. list.Select(t => (t.Name, t.Shape, (Func<float[]>)(() => t.Values)))], metadata);
    }

    /// <summary>
    /// Writes tensors whose values are produced one at a time by <c>Values</c>, as each is written: the header comes from
    /// the shapes, so only one tensor's values (and one chunk of its bytes) exist at once (a model's weights are never all
    /// on the host together).
    /// </summary>
    public static void Write(string path, IReadOnlyList<(string Name, int[] Shape, Func<float[]> Values)> tensors, IReadOnlyDictionary<string, string>? metadata = null)
    {
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
                ["dtype"] = "F32",
                ["shape"] = new JsonArray([.. shape.Select(d => (JsonNode)d)]),
                ["data_offsets"] = new JsonArray(offset, offset + count * sizeof(float)),
            };
            offset += count * sizeof(float);
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
        byte[] bytes = System.Buffers.ArrayPool<byte>.Shared.Rent(ChunkValues * sizeof(float));
        try
        {
            foreach (var (name, shape, produce) in tensors)
            {
                var values = produce();
                Check(name, shape, values);
                for (int first = 0; first < values.Length; first += ChunkValues)
                {
                    int n = Math.Min(ChunkValues, values.Length - first), start = first;
                    Idrak.Abstraction.Devices.HostParallel.For(n, 1 << 16, (lo, hi) =>
                    {
                        for (int i = lo; i < hi; i++)
                        {
                            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), values[start + i]);
                        }
                    });
                    stream.Write(bytes, 0, n * sizeof(float));
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
}
