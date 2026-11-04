// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Diagnostics;
using System.Globalization;
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

    /// <summary>8-bit float, 4 exponent and 3 mantissa bits, no infinities (OCP E4M3FN, PyTorch's float8_e4m3fn); read only.</summary>
    F8_E4M3,

    /// <summary>8-bit float, 5 exponent and 2 mantissa bits (OCP E5M2, PyTorch's float8_e5m2); read only.</summary>
    F8_E5M2,

    /// <summary>A signed byte: a weight quantized to int8, read with the scale stored beside it; read only.</summary>
    I8,
}

/// <summary>A tensor's entry in a safetensors file: its type, shape and byte range.</summary>
public sealed record SafeTensorInfo(string Name, SafeTensorType Type, int[] Shape, long Offset, long Length, string File)
{
    /// <summary>Number of elements.</summary>
    public long Count => Shape.Aggregate(1L, (a, b) => a * b);

    /// <summary>Stored bytes per value: 4 (F32), 2 (F16, BF16) or 1 (F8_E4M3, F8_E5M2, I8).</summary>
    public int BytesPerValue => Type switch
    {
        SafeTensorType.F32 => 4,
        SafeTensorType.F16 or SafeTensorType.BF16 => 2,
        _ => 1,
    };
}

/// <summary>
/// Reads safetensors weights: one file, or a sharded checkpoint (<c>model.safetensors.index.json</c> naming the file of
/// each tensor). Tensors are read one at a time, straight from disk, and converted to float32, or, for bfloat16 ones, also
/// copied as their 16-bit values (<see cref="ReadBFloat16"/>). Tensors of integer or boolean types (buffers a checkpoint
/// may carry beside its weights, such as position ids) are left out (<see cref="LeftOut"/>), not refused.
/// </summary>
/// <remarks>
/// Quantized checkpoints: F8_E4M3 and F8_E5M2 tensors are read as their float values, and an I8 tensor with a scale beside
/// it as a quantized weight. A weight's scale, <c>{name}_scale_inv</c> (DeepSeek and Hugging Face fine-grained FP8) or
/// <c>{name}_scale</c> (compressed-tensors, FBGEMM), and its zero point <c>{name}_zero_point</c>, are applied as the weight
/// is read, value = (w − zero point) · scale in float32: one scale for the tensor, one per row, or one per block of rows and
/// columns (per matrix of a stacked [experts, rows, columns] weight). The block's size is config.json's
/// <c>quantization_config.weight_block_size</c> when it fits, else the smallest power of two the scale's shape gives (128 for
/// DeepSeek's), else rows / scale rows when they divide. The scales, and the activation scales such a checkpoint carries
/// (<c>input_scale</c>, <c>k_scale</c>, …), are not among <see cref="Tensors"/>: see <see cref="QuantizationScales"/>.
/// </remarks>
public sealed class SafeTensorsReader : IDisposable, IBFloat16TensorStore
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
    private readonly Dictionary<string, Entry> _entries = new(StringComparer.Ordinal);       // every stored tensor, as listed
    private readonly Dictionary<string, Scaling> _scalings = new(StringComparer.Ordinal);    // a quantized weight's scale
    private readonly Dictionary<string, string> _scales = new(StringComparer.Ordinal);       // scale → what it scales
    private int[]? _blockSize;                                                               // config.json's weight_block_size
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

    /// <summary>
    /// The scales of a quantized checkpoint, by name, with the weight each scales (applied when it is read), or
    /// "activations" for the scales of a layer's inputs or cached keys and values: not among <see cref="Tensors"/>.
    /// </summary>
    public IReadOnlyDictionary<string, string> QuantizationScales => _scales;

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

            reader._blockSize = WeightBlockSize(Path.GetDirectoryName(Path.GetFullPath(path))!);
            reader.LinkScales();
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
        if (info.BytesPerValue == 1)
        {
            ReadQuantized(info, 0, values);
            return values;
        }

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

    /// <inheritdoc cref="ITensorStore.ReadRows"/>
    /// <remarks>The rows' byte range alone is read (positional reads, chunks of 8 MB widened in place for 16-bit types).</remarks>
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
        if (info.BytesPerValue == 1)
        {
            ReadQuantized(info, first * rowValues, destination[..values]);
            return true;
        }

        int width = info.BytesPerValue;
        long offset = info.Offset + first * rowValues * width;
        var handle = _files[info.File].SafeFileHandle;
        if (info.Type == SafeTensorType.F32 && BitConverter.IsLittleEndian)
        {
            ReadTimed(handle, MemoryMarshal.AsBytes(destination[..values]), offset);
            return true;
        }

        const int ChunkValues = 1 << 22;                                            // 8 MB of 16-bit values
        byte[] buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(Math.Min(values, ChunkValues) * width);
        try
        {
            for (int at = 0; at < values; at += ChunkValues)
            {
                int n = Math.Min(ChunkValues, values - at);
                ReadTimed(handle, buffer.AsSpan(0, n * width), offset + (long)at * width);
                Widen(buffer.AsSpan(0, n * width), info.Type, destination.Slice(at, n));
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }

        return true;
    }

    /// <inheritdoc cref="ITensorStore.IdentityOf"/>
    /// <remarks>The file's full path, its size and time of last change (UTC ticks), and the tensor's byte range and type in it.</remarks>
    public string? IdentityOf(string name)
    {
        if (!_tensors.TryGetValue(name, out var info))
        {
            return null;
        }

        var file = new FileInfo(info.File);
        string identity = string.Create(CultureInfo.InvariantCulture, $"safetensors|{file.FullName}|{file.Length}|{file.LastWriteTimeUtc.Ticks}|{info.Offset}|{info.Length}|{info.Type}");
        if (_scalings.TryGetValue(name, out var scaling))
        {
            // The scale (and zero point) change the values too: their files and byte ranges are part of the identity.
            foreach (var entry in new[] { scaling.Scale, scaling.Zero })
            {
                if (entry is not null)
                {
                    var of = new FileInfo(entry.File);
                    identity += string.Create(CultureInfo.InvariantCulture, $"|{entry.Name}|{of.FullName}|{of.Length}|{of.LastWriteTimeUtc.Ticks}|{entry.Offset}|{entry.Length}|{entry.DType}");
                }
            }
        }

        return identity;
    }

    // Widens stored values into float32 (a span: the rows' destination).
    private static void Widen(ReadOnlySpan<byte> bytes, SafeTensorType type, Span<float> values)
    {
        switch (type)
        {
            case SafeTensorType.F32:
                StoredFloats.ReadFloat32(bytes, values);
                break;
            case SafeTensorType.F16:
                StoredFloats.WidenFloat16(bytes, values);
                break;
            default:
                StoredFloats.WidenBFloat16(bytes, values);
                break;
        }
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

        if (a.BytesPerValue == 1 || b.BytesPerValue == 1)
        {
            return Read(first).AsSpan().SequenceEqual(Read(second));   // scaled values: compared whole
        }

        const int ChunkValues = 1 << 20;
        long count = a.Count;
        int chunk = (int)Math.Min(count, ChunkValues);
        int widthA = a.BytesPerValue, widthB = b.BytesPerValue;
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

        if (!BitConverter.IsLittleEndian || info.BytesPerValue == 1)
        {
            return Idrak.Abstraction.Devices.HostParallel.Transpose(Read(name), info.Shape[0], info.Shape[1]);
        }

        // Chunks of stored rows are read through one pooled buffer and scattered, widened, into the transposed result.
        int rows = info.Shape[0], columns = info.Shape[1], width = info.BytesPerValue;
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

    /// <inheritdoc cref="IBFloat16TensorStore.ReadBFloat16"/>
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
            int[] shape = [.. node["shape"]!.AsArray().Select(d => (int)d!)];
            var offsets = node["data_offsets"]!.AsArray();
            long begin = (long)offsets[0]!, end = (long)offsets[1]!;
            _entries[name] = new Entry(name, dtype, shape, dataStart + begin, end - begin, file);
            if (dtype is "I64" or "I32" or "I16" or "I8" or "U64" or "U32" or "U16" or "U8" or "BOOL")
            {
                _leftOut[name] = dtype;                                     // a buffer (position ids, a mask), not a weight; an
                continue;                                                   // I8 with a scale becomes one in LinkScales
            }

            var type = dtype switch
            {
                "F32" => SafeTensorType.F32,
                "F16" => SafeTensorType.F16,
                "BF16" => SafeTensorType.BF16,
                "F8_E4M3" => SafeTensorType.F8_E4M3,
                "F8_E5M2" => SafeTensorType.F8_E5M2,
                _ => throw new NotSupportedException($"Tensor '{name}' in {file} has type {dtype}; F32, F16, BF16, F8_E4M3 and F8_E5M2 are supported, and I8 with a scale."),
            };
            _tensors[name] = new SafeTensorInfo(name, type, shape, dataStart + begin, end - begin, file);
        }
    }

    // A stored tensor as the header lists it (scales and zero points are read from these).
    private sealed record Entry(string Name, string DType, int[] Shape, long Offset, long Length, string File);

    // A quantized weight's scale (and zero point): the scale grid's [a, b] blocks of [blockRows, blockColumns] values per
    // matrix of `rows` × `columns` (perMatrix: the scale has a grid per matrix of a stacked weight). Values read on first use.
    private sealed class Scaling(Entry scale, Entry? zero, int rows, int columns, int a, int b, int blockRows, int blockColumns, bool perMatrix)
    {
        public Entry Scale => scale;

        public Entry? Zero => zero;

        public float[]? ScaleValues;

        public float[]? ZeroValues;

        // values (the weight's values first.. first + values.Length) = (values − zero) · scale, in float32.
        public void Apply(long first, Span<float> values, float[] s, float[]? z)
        {
            long v = first;
            int k = 0;
            while (k < values.Length)
            {
                long row = v / columns;
                int c0 = (int)(v % columns), n = (int)Math.Min(columns - c0, values.Length - k);
                long matrix = row / rows;
                int r = (int)(row % rows);
                int baseIndex = (perMatrix ? (int)(matrix * a * b) : 0) + r / blockRows * b;
                var span = values.Slice(k, n);
                for (int j = 0; j < span.Length; j++)
                {
                    int i = baseIndex + (c0 + j) / blockColumns;
                    span[j] = z is null ? span[j] * s[i] : (span[j] - z[i]) * s[i];
                }

                k += n;
                v += n;
            }
        }
    }

    // config.json's quantization_config.weight_block_size beside the checkpoint ([rows, columns]), or null.
    private static int[]? WeightBlockSize(string folder)
    {
        string config = Path.Combine(folder, "config.json");
        if (!File.Exists(config))
        {
            return null;
        }

        try
        {
            return JsonNode.Parse(File.ReadAllText(config))?["quantization_config"]?["weight_block_size"] is JsonArray { Count: 2 } size
                ? [(int)size[0]!, (int)size[1]!]
                : null;
        }
        catch (Exception e) when (e is JsonException or InvalidOperationException or FormatException)
        {
            return null;                                                    // the model's loader reports a broken config
        }
    }

    // Activation scales a quantized checkpoint carries (static input quantization, an FP8 KV cache): not weights.
    private static readonly string[] ActivationScales = [".input_scale", ".input_zero_point", ".output_scale", ".k_scale", ".v_scale"];

    // Each F8 / I8 tensor's scale and zero point, found by name once every file's header is read (a shard may hold a scale
    // apart from its weight): an I8 tensor with a scale becomes a weight, the scales leave Tensors and LeftOut.
    private void LinkScales()
    {
        bool quantized = false;
        foreach (var entry in _entries.Values.ToList())
        {
            bool float8 = entry.DType is "F8_E4M3" or "F8_E5M2";
            if (!float8 && entry.DType != "I8")
            {
                continue;
            }

            var scale = Find(entry.Name + "_scale_inv") ?? Find(entry.Name + "_scale");
            if (scale is null)
            {
                continue;                                                   // F8: its values; I8: a buffer, left out
            }

            if (entry.DType == "I8")
            {
                _leftOut.Remove(entry.Name);
                _tensors[entry.Name] = new SafeTensorInfo(entry.Name, SafeTensorType.I8, entry.Shape, entry.Offset, entry.Length, entry.File);
            }

            var zero = Find(entry.Name + "_zero_point");
            _scalings[entry.Name] = Geometry(entry, scale, zero);
            foreach (var part in new[] { scale, zero })
            {
                if (part is not null)
                {
                    _tensors.Remove(part.Name);
                    _leftOut.Remove(part.Name);
                    _scales[part.Name] = entry.Name;
                }
            }

            quantized = true;
        }

        quantized |= _tensors.Values.Any(t => t.Type is SafeTensorType.F8_E4M3 or SafeTensorType.F8_E5M2);
        if (quantized)
        {
            foreach (var name in _entries.Keys.Where(n => ActivationScales.Any(s => n.EndsWith(s, StringComparison.Ordinal))))
            {
                _tensors.Remove(name);
                _leftOut.Remove(name);
                _scales[name] = "activations";
            }
        }

        Entry? Find(string name) => _entries.TryGetValue(name, out var found) && !_scales.ContainsKey(name) ? found : null;
    }

    // Where a weight's values find their scale: one for the tensor, one per row ([rows] or [rows, 1] of a matrix), or a grid
    // of blocks per matrix (the scale's leading dimensions the weight's).
    private Scaling Geometry(Entry weight, Entry scale, Entry? zero)
    {
        int rank = weight.Shape.Length;
        int columns = rank == 0 ? 1 : weight.Shape[^1], rows = rank >= 2 ? weight.Shape[^2] : 1;
        long scales = scale.Shape.Aggregate(1L, (x, y) => x * y);
        int a, b;
        bool perMatrix = false;
        if (scales == 1)
        {
            (a, b) = (1, 1);
        }
        else if (rank == 2 && scale.Shape.Length == 1 && scale.Shape[0] == rows)
        {
            (a, b) = (rows, 1);
        }
        else if (rank >= 2 && scale.Shape.Length == rank && scale.Shape.AsSpan(0, rank - 2).SequenceEqual(weight.Shape.AsSpan(0, rank - 2)))
        {
            (a, b) = (scale.Shape[^2], scale.Shape[^1]);
            perMatrix = rank > 2;
        }
        else
        {
            throw Unfit($"a scale of [{string.Join(", ", scale.Shape)}] for a weight of [{string.Join(", ", weight.Shape)}]");
        }

        if (zero is not null && zero.Shape.Aggregate(1L, (x, y) => x * y) != scales)
        {
            throw Unfit($"a zero point of [{string.Join(", ", zero.Shape)}] beside a scale of [{string.Join(", ", scale.Shape)}]");
        }

        int blockRows = Block(rows, a, _blockSize?[0]) ?? throw Unfit($"{a} scale rows for {rows} rows");
        int blockColumns = Block(columns, b, _blockSize?[1]) ?? throw Unfit($"{b} scale columns for {columns} columns");
        return new Scaling(scale, zero, rows, columns, a, b, blockRows, blockColumns, perMatrix);

        NotSupportedException Unfit(string what) => new($"The quantized weight '{weight.Name}' ({weight.DType}) in {weight.File} has {what}: "
            + "one scale, one per row, or one per block of rows and columns are supported (config.json's quantization_config.weight_block_size names the block).");
    }

    // Values per block of `size` values cut into `blocks` scales: the configured block when it gives that many, else the
    // smallest power of two that does, else size / blocks when it divides; null when none fits.
    private static int? Block(int size, int blocks, int? configured)
    {
        static int Count(int size, int block) => (size + block - 1) / block;
        if (blocks <= 0)
        {
            return null;
        }

        if (configured is > 0 and var c && Count(size, c) == blocks)
        {
            return c;
        }

        for (long p = 1; p <= Math.Max(1L, 2L * size); p *= 2)
        {
            if (Count(size, (int)p) == blocks)
            {
                return (int)p;
            }
        }

        return size % blocks == 0 ? size / blocks : null;
    }

    // F8_E4M3 and F8_E5M2 codes as float32: every code's value exactly, a NaN code as the quiet NaN of its sign (as PyTorch
    // and ml_dtypes widen them).
    private static readonly float[] E4M3Values = Float8Table(e4m3: true), E5M2Values = Float8Table(e4m3: false);

    private static float[] Float8Table(bool e4m3)
    {
        var table = new float[256];
        for (int code = 0; code < 256; code++)
        {
            uint sign = (uint)(code >> 7) << 31;
            float value;
            if (e4m3)
            {
                int e = (code >> 3) & 15, m = code & 7;
                value = e == 15 && m == 7 ? float.NaN
                    : e == 0 ? m * (1f / 512f)                                     // subnormal: m · 2^-9
                    : BitConverter.UInt32BitsToSingle((uint)(e - 7 + 127) << 23 | (uint)m << 20);
            }
            else
            {
                value = (float)BitConverter.UInt16BitsToHalf((ushort)((code & 0x7F) << 8));   // E5M2 is a half's top byte
            }

            table[code] = float.IsNaN(value) ? BitConverter.UInt32BitsToSingle(sign | 0x7FC00000u)
                : BitConverter.UInt32BitsToSingle(BitConverter.SingleToUInt32Bits(value) | sign);
        }

        return table;
    }

    // Values first.. of an F8 or I8 tensor into destination, through one pooled buffer of their bytes, then its scale.
    private void ReadQuantized(SafeTensorInfo info, long first, Span<float> destination)
    {
        var handle = _files[info.File].SafeFileHandle;
        var table = info.Type switch { SafeTensorType.F8_E4M3 => E4M3Values, SafeTensorType.F8_E5M2 => E5M2Values, _ => null };
        const int ChunkValues = 1 << 23;                                            // 8 MB of codes
        byte[] buffer = System.Buffers.ArrayPool<byte>.Shared.Rent(Math.Max(1, Math.Min(destination.Length, ChunkValues)));
        try
        {
            for (int at = 0; at < destination.Length; at += ChunkValues)
            {
                int n = Math.Min(ChunkValues, destination.Length - at);
                ReadTimed(handle, buffer.AsSpan(0, n), info.Offset + first + at);
                var codes = buffer.AsSpan(0, n);
                var values = destination.Slice(at, n);
                if (table is null)
                {
                    for (int i = 0; i < codes.Length; i++)
                    {
                        values[i] = (sbyte)codes[i];
                    }
                }
                else
                {
                    for (int i = 0; i < codes.Length; i++)
                    {
                        values[i] = table[codes[i]];
                    }
                }
            }
        }
        finally
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(buffer);
        }

        if (_scalings.TryGetValue(info.Name, out var scaling))
        {
            var s = scaling.ScaleValues ??= ReadSmall(scaling.Scale);
            var z = scaling.Zero is null ? null : scaling.ZeroValues ??= ReadSmall(scaling.Zero);
            scaling.Apply(first, destination, s, z);
        }
    }

    // A scale or zero point (a small tensor) as float32.
    private float[] ReadSmall(Entry entry)
    {
        int width = entry.DType switch { "F32" or "I32" => 4, "F16" or "BF16" => 2, "I8" or "U8" or "F8_E4M3" or "F8_E5M2" => 1, _ => 0 };
        long count = entry.Shape.Aggregate(1L, (x, y) => x * y);
        if (width == 0 || count * width != entry.Length)
        {
            throw new NotSupportedException($"The scale '{entry.Name}' in {entry.File} is {entry.DType} [{string.Join(", ", entry.Shape)}] in {entry.Length} bytes; "
                + "a scale or zero point is F32, F16, BF16, F8, I32, I8 or U8.");
        }

        var bytes = new byte[entry.Length];
        ReadTimed(_files[entry.File].SafeFileHandle, bytes, entry.Offset);
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = entry.DType switch
            {
                "F32" => BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(4 * i)),
                "I32" => BinaryPrimitives.ReadInt32LittleEndian(bytes.AsSpan(4 * i)),
                "F16" => (float)BinaryPrimitives.ReadHalfLittleEndian(bytes.AsSpan(2 * i)),
                "BF16" => BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(bytes.AsSpan(2 * i)) << 16),
                "I8" => (sbyte)bytes[i],
                "U8" => bytes[i],
                "F8_E4M3" => E4M3Values[bytes[i]],
                _ => E5M2Values[bytes[i]],
            };
        }

        return values;
    }

    private Exception Missing(string name) => _leftOut.TryGetValue(name, out string? type)
        ? new NotSupportedException($"The tensor '{name}' is of type {type}, a buffer the reader leaves out (it reads F32, F16, BF16 and F8 tensors, and I8 ones with a scale).")
        : _scales.TryGetValue(name, out string? scaled)
            ? new NotSupportedException(scaled == "activations"
                ? $"The tensor '{name}' is an activation scale of a quantized checkpoint, not a weight (QuantizationScales)."
                : $"The tensor '{name}' is the quantization scale of '{scaled}', applied when '{scaled}' is read (QuantizationScales).")
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
                    StoredFloats.WidenFloat16(bytes.AsSpan(first * 2, (last - first) * 2), values.AsSpan(offset + first, last - first)));
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
        if (type is not (SafeTensorType.F32 or SafeTensorType.F16 or SafeTensorType.BF16))
        {
            throw new NotSupportedException($"safetensors are written as F32, F16 or BF16; {type} is read only (a quantized checkpoint's weights are read with their scales).");
        }

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
// float32 bytes are copied as they are, and bfloat16 and float16 are widened whole vectors at a time by HalfFloats (as
// the CPU device widens its bfloat16 and float16 weights) with a scalar tail; elsewhere value by value. The same values
// either way.
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

    // bfloat16 stored little-endian, widened by HalfFloats (whole vectors at a time); on a big-endian machine value by
    // value, the same values.
    public static void WidenBFloat16(ReadOnlySpan<byte> bytes, Span<float> values)
    {
        if (BitConverter.IsLittleEndian)
        {
            HalfFloats.WidenBFloat16(MemoryMarshal.Cast<byte, ushort>(bytes[..(values.Length * 2)]), values);
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = BitConverter.UInt32BitsToSingle((uint)BinaryPrimitives.ReadUInt16LittleEndian(bytes[(i * 2)..]) << 16);
        }
    }

    // IEEE halves stored little-endian, widened by HalfFloats (whole vectors at a time); on a big-endian machine value by
    // value, the same values.
    public static void WidenFloat16(ReadOnlySpan<byte> bytes, Span<float> values)
    {
        if (BitConverter.IsLittleEndian)
        {
            HalfFloats.WidenFloat16(MemoryMarshal.Cast<byte, ushort>(bytes[..(values.Length * 2)]), values);
            return;
        }

        for (int i = 0; i < values.Length; i++)
        {
            values[i] = HalfFloats.Float16ToSingle(BinaryPrimitives.ReadUInt16LittleEndian(bytes[(i * 2)..]));
        }
    }
}
