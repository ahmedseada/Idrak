// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Models;

/// <summary>A tensor's entry in a GGUF file.</summary>
/// <param name="Name">Its name (llama.cpp's naming, e.g. blk.0.attn_q.weight).</param>
/// <param name="Type">The ggml type id the file stores it in (GGUF's own table, <see cref="GgufFile.TypeOfId"/>, says which type it is).</param>
/// <param name="Dimensions">Sizes, innermost first (ggml's ne: [columns, rows, …]).</param>
/// <param name="Offset">Byte offset of its data in the file.</param>
public sealed record GgufTensorInfo(string Name, int Type, long[] Dimensions, long Offset)
{
    /// <summary>Number of elements.</summary>
    public long Count => Dimensions.Aggregate(1L, (a, b) => a * b);

    /// <summary>The shape outermost first, as PyTorch / safetensors write it ([rows, columns]).</summary>
    public int[] Shape => [.. Dimensions.Reverse().Select(d => checked((int)d))];

    /// <summary>The type its ggml type id names (<see cref="GgufFile.TypeOfId"/>), or null for an id GGUF's table here does not hold.</summary>
    public IElementType? ElementType => GgufFile.TypeOfId(Type);
}

/// <summary>
/// Reads GGUF files (the format of llama.cpp and of the local model store): metadata, the tensor index, and the tensors,
/// each as the type its ggml type id names (GGUF's own table, <see cref="TypeOfId"/>): its stored bytes as they lie
/// (<see cref="ReadStored"/>, never widened), or its values read through its type as float32 (<see cref="Read"/>). An id
/// the table does not hold is refused when read.
/// </summary>
public sealed class GgufFile : IDisposable
{
    private readonly FileStream _file;
    private readonly Dictionary<string, GgufTensorInfo> _tensors = new(StringComparer.Ordinal);

    private GgufFile(string path)
    {
        Path = System.IO.Path.GetFullPath(path);
        _file = new FileStream(Path, FileMode.Open, FileAccess.Read, FileShare.Read, 1, FileOptions.RandomAccess);
    }

    /// <summary>The file.</summary>
    public string Path { get; }

    /// <summary>The GGUF version (2 or 3).</summary>
    public int Version { get; private set; }

    /// <summary>
    /// Metadata by key: numbers as long / ulong / double, strings, booleans, and arrays: uint8 as byte[], int8 as sbyte[],
    /// float32 as float[], float64 as double[], the other integers as long[], and string[], bool[] or object[] (arrays of
    /// arrays). Numeric arrays are read in one block (a tokenizer's 250,000 scores, or its charsmap, take one read, at
    /// their own width). <see cref="Get{T}"/> converts byte[] and sbyte[] to long[] and float[] to double[] for callers that
    /// ask for those.
    /// </summary>
    public Dictionary<string, object> Metadata { get; } = new(StringComparer.Ordinal);

    /// <summary>The tensors by name, in file order.</summary>
    public IReadOnlyDictionary<string, GgufTensorInfo> Tensors => _tensors;

    /// <summary>The value of <paramref name="key"/> converted to T, or <paramref name="fallback"/>.</summary>
    public T Get<T>(string key, T fallback)
    {
        if (!Metadata.TryGetValue(key, out var value))
        {
            return fallback;
        }

        // The CLR lets a byte[] pass for an sbyte[] (and back), so those two are told apart by their exact type.
        bool bytes = value.GetType() == typeof(byte[]), signedBytes = value.GetType() == typeof(sbyte[]);
        return value switch
        {
            T t when !(bytes || signedBytes) || value.GetType() == typeof(T) || !typeof(T).IsArray => t,
            long l when typeof(T) == typeof(int) => (T)(object)checked((int)l),
            ulong u when typeof(T) == typeof(int) => (T)(object)checked((int)u),
            ulong u when typeof(T) == typeof(long) => (T)(object)checked((long)u),
            long l when typeof(T) == typeof(double) => (T)(object)(double)l,
            double d when typeof(T) == typeof(float) => (T)(object)(float)d,
            long[] a when typeof(T) == typeof(int) && a.Length > 0 => (T)(object)checked((int)a[0]),   // per-layer values: the first
            byte[] a when bytes && typeof(T) == typeof(int) && a.Length > 0 => (T)(object)(int)a[0],
            sbyte[] a when signedBytes && typeof(T) == typeof(int) && a.Length > 0 => (T)(object)(int)a[0],
            byte[] a when bytes && typeof(T) == typeof(long[]) => (T)(object)Array.ConvertAll(a, x => (long)x),
            sbyte[] a when signedBytes && typeof(T) == typeof(long[]) => (T)(object)Array.ConvertAll(a, x => (long)x),
            float[] a when typeof(T) == typeof(double[]) => (T)(object)Array.ConvertAll(a, x => (double)x),
            _ => fallback,
        };
    }

    /// <summary>Opens <paramref name="path"/> and reads its header, metadata and tensor index.</summary>
    public static GgufFile Open(string path)
    {
        var file = new GgufFile(path);
        try
        {
            file.ReadHeader();
            return file;
        }
        catch
        {
            file.Dispose();
            throw;
        }
    }

    /// <summary>
    /// The type ggml's type id <paramref name="id"/> names (GGUF's own table: 0 F32, 1 F16, 2 Q4_0, 3 Q4_1, 6 Q5_0, 7 Q5_1,
    /// 8 Q8_0, 10 Q2_K, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K, 16 IQ2_XXS, 17 IQ2_XS, 18 IQ3_XXS, 19 IQ1_S, 20 IQ4_NL, 21 IQ3_S,
    /// 22 IQ2_S, 23 IQ4_XS, 24 I8, 25 I16, 26 I32, 27 I64, 28 F64, 29 IQ1_M, 30 BF16, 34 TQ1_0, 35 TQ2_0, 39 MXFP4), or null.
    /// </summary>
    public static IElementType? TypeOfId(int id) => id switch
    {
        0 => Abstraction.Types.F32.Type,
        1 => Abstraction.Types.F16.Type,
        2 => Types.Q4_0.Type,
        3 => Types.Q4_1.Type,
        6 => Types.Q5_0.Type,
        7 => Types.Q5_1.Type,
        8 => Types.Q8_0.Type,
        10 => Types.Q2_K.Type,
        11 => Types.Q3_K.Type,
        12 => Types.Q4_K.Type,
        13 => Types.Q5_K.Type,
        14 => Types.Q6_K.Type,
        16 => Types.IQ2_XXS.Type,
        17 => Types.IQ2_XS.Type,
        18 => Types.IQ3_XXS.Type,
        19 => Types.IQ1_S.Type,
        20 => Types.IQ4_NL.Type,
        21 => Types.IQ3_S.Type,
        22 => Types.IQ2_S.Type,
        23 => Types.IQ4_XS.Type,
        24 => Abstraction.Types.I8.Type,
        25 => Abstraction.Types.I16.Type,
        26 => Abstraction.Types.I32.Type,
        27 => Abstraction.Types.I64.Type,
        28 => Abstraction.Types.F64.Type,
        29 => Types.IQ1_M.Type,
        30 => Abstraction.Types.BF16.Type,
        34 => Types.TQ1_0.Type,
        35 => Types.TQ2_0.Type,
        39 => Types.MXFP4.Type,
        _ => null,
    };

    /// <summary>The type the tensor <paramref name="name"/> is stored in (<see cref="TypeOfId"/> of its id).</summary>
    public IElementType TypeOf(string name) => Typed(name).Type;

    /// <summary>The tensor <paramref name="name"/>'s values as its type stores them: its bytes as they lie, never widened.</summary>
    public byte[] ReadStored(string name)
    {
        var (info, type) = Typed(name);
        var bytes = GC.AllocateUninitializedArray<byte>(checked((int)type.ByteCount(info.Count)));
        ReadAt(bytes, info.Offset);
        return bytes;
    }

    /// <summary>The tensor <paramref name="name"/> as float32 values (row-major, as stored; another type read through its type).</summary>
    /// <exception cref="NotSupportedException">The tensor's ggml type id is not one GGUF's table here holds.</exception>
    public float[] Read(string name)
    {
        var (info, type) = Typed(name);
        var values = GC.AllocateUninitializedArray<float>(checked((int)info.Count));
        if (info.Type == 0)
        {
            ReadFloats(values, info.Offset);
        }
        else
        {
            type.Read(ReadStored(name), values);
        }

        return values;
    }

    /// <summary>
    /// Rows [<paramref name="firstRow"/>, + n) of the 2-D float32 tensor <paramref name="name"/> (shape [rows, columns])
    /// into <paramref name="values"/> (n whole rows).
    /// </summary>
    internal void ReadRows(string name, long firstRow, Span<float> values)
    {
        var (info, type) = Typed(name);
        long columns = info.Dimensions[0];
        if (columns == 0 || values.Length % columns != 0)
        {
            throw new InvalidDataException($"{name}: {values.Length} values are not whole rows of {columns}.");
        }

        if (info.Type == 0)
        {
            ReadFloats(values, info.Offset + firstRow * columns * sizeof(float));
            return;
        }

        // Rows of another type: their bytes (each row whole groups of the type's blocks, as ggml stores rows).
        long rowBytes = type.ByteCount(columns);
        var bytes = new byte[checked((int)(rowBytes * (values.Length / columns)))];
        ReadAt(bytes, info.Offset + firstRow * rowBytes);
        type.Read(bytes, values);
    }

    // The entry of the tensor name and its type; an id GGUF's table here does not hold is refused.
    private (GgufTensorInfo Info, IElementType Type) Typed(string name)
    {
        var info = _tensors.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"No tensor named '{name}' in {Path}.");
        return info.ElementType is { } type ? (info, type)
            : throw new NotSupportedException($"{name} in {Path} is stored as ggml type #{info.Type}, an id GGUF's table here does not hold.");
    }

    // Little-endian float32 values from the file at offset into values.
    private void ReadFloats(Span<float> values, long offset)
    {
        ReadAt(MemoryMarshal.AsBytes(values), offset);
        if (!BitConverter.IsLittleEndian)
        {
            var bits = MemoryMarshal.Cast<float, uint>(values);
            BinaryPrimitives.ReverseEndianness(bits, bits);
        }
    }

    /// <inheritdoc />
    public void Dispose() => _file.Dispose();

    // ------------------------------------------------------------------ header

    private long _position;

    private void ReadHeader()
    {
        Span<byte> magic = stackalloc byte[4];
        ReadAt(magic, 0);
        _position = 4;
        if (!magic.SequenceEqual("GGUF"u8))
        {
            throw new InvalidDataException($"{Path} is not a GGUF file.");
        }

        Version = (int)U32();
        if (Version is < 2 or > 3)
        {
            throw new NotSupportedException($"{Path}: GGUF version {Version} is not supported (2 and 3 are).");
        }

        long tensors = checked((long)U64()), keys = checked((long)U64());
        // Each count bounded by the bytes left before anything is sized by it: a key has at least its name's length,
        // a type and one byte; a tensor at least its name's length, a dimension count, a type and an offset.
        CheckEntries(keys, 13, "metadata keys");
        CheckEntries(tensors, 24, "tensors");
        for (long i = 0; i < keys; i++)
        {
            string key = Str();
            Metadata[key] = Value((int)U32(), 0);
        }

        var infos = new List<(string Name, int Type, long[] Dims, long Offset)>();
        var names = new HashSet<string>(StringComparer.Ordinal);
        for (long i = 0; i < tensors; i++)
        {
            string name = Str();
            if (!names.Add(name))
            {
                throw new InvalidDataException($"{Path}: the tensor '{name}' is listed twice (a GGUF names each tensor once).");
            }

            uint dims = U32();
            if (dims > MaxDimensions)
            {
                throw new InvalidDataException($"{Path}: the tensor '{name}' has {dims} dimensions; GGUF tensors have at most {MaxDimensions}.");
            }

            var shape = new long[dims];
            for (int d = 0; d < shape.Length; d++)
            {
                ulong size = U64();
                shape[d] = size <= long.MaxValue ? (long)size
                    : throw new InvalidDataException($"{Path}: the tensor '{name}' has a dimension of {size} elements.");
            }

            int type = (int)U32();
            long offset = checked((long)U64());
            infos.Add((name, type, shape, offset));
        }

        long alignment = Get("general.alignment", 32L);
        alignment = alignment > 0 ? alignment : 32;
        long dataStart = (_position + alignment - 1) / alignment * alignment;
        foreach (var (name, type, dims, offset) in infos)
        {
            _tensors[name] = new GgufTensorInfo(name, type, dims, dataStart + offset);
        }
    }

    private object Value(int type, int depth)
    {
        switch (type)
        {
            case 0: return (long)Bytes(1)[0];
            case 1: return (long)(sbyte)Bytes(1)[0];
            case 2: return (long)BinaryPrimitives.ReadUInt16LittleEndian(Bytes(2));
            case 3: return (long)BinaryPrimitives.ReadInt16LittleEndian(Bytes(2));
            case 4: return (long)U32();
            case 5: return (long)BinaryPrimitives.ReadInt32LittleEndian(Bytes(4));
            case 6: return (double)BinaryPrimitives.ReadSingleLittleEndian(Bytes(4));
            case 7: return Bytes(1)[0] != 0;
            case 8: return Str();
            case 10: return U64() is var u && u <= long.MaxValue ? (long)u : (object)u;
            case 11: return BinaryPrimitives.ReadInt64LittleEndian(Bytes(8));
            case 12: return BinaryPrimitives.ReadDoubleLittleEndian(Bytes(8));
            case 9:
                if (depth > 8)
                {
                    throw new InvalidDataException($"{Path}: metadata nests too deeply.");
                }

                int itemType = (int)U32();
                long count = checked((long)U64());
                switch (itemType)
                {
                    case 0: return ReadArray<byte>(count);
                    case 1: return ReadArray<sbyte>(count);
                    case 2: return Array.ConvertAll(ReadArray<ushort>(count), x => (long)x);
                    case 3: return Array.ConvertAll(ReadArray<short>(count), x => (long)x);
                    case 4: return Array.ConvertAll(ReadArray<uint>(count), x => (long)x);
                    case 5: return Array.ConvertAll(ReadArray<int>(count), x => (long)x);
                    case 6: return ReadArray<float>(count);
                    case 7: return Array.ConvertAll(ReadArray<byte>(count), x => x != 0);
                    case 10: return Array.ConvertAll(ReadArray<ulong>(count), x => unchecked((long)x));
                    case 11: return ReadArray<long>(count);
                    case 12: return ReadArray<double>(count);
                    case 8:
                        CheckCount(count, 8);                      // each string has at least its 8-byte length
                        var strings = new string[count];
                        for (long i = 0; i < count; i++)
                        {
                            strings[i] = Str();
                        }

                        return strings;
                    case 9:
                        CheckCount(count, 12);                     // each array has at least its type and count
                        var items = new object[count];
                        for (long i = 0; i < count; i++)
                        {
                            items[i] = Value(itemType, depth + 1);
                        }

                        return items;
                    default:
                        throw new InvalidDataException($"{Path}: unknown metadata array item type {itemType}.");
                }

            default:
                throw new InvalidDataException($"{Path}: unknown metadata type {type}.");
        }
    }

    private readonly byte[] _scratch = new byte[8];

    // A metadata array of fixed-width items, read in one block straight into the array (GGUF is little-endian).
    private T[] ReadArray<T>(long count) where T : unmanaged
    {
        int size = System.Runtime.CompilerServices.Unsafe.SizeOf<T>();
        CheckCount(count, size);
        var array = new T[count];
        var bytes = MemoryMarshal.AsBytes(array.AsSpan());
        ReadAt(bytes, _position);
        _position += bytes.Length;
        if (!BitConverter.IsLittleEndian && size > 1)
        {
            switch (size)
            {
                case 2: BinaryPrimitives.ReverseEndianness(MemoryMarshal.Cast<byte, ushort>(bytes), MemoryMarshal.Cast<byte, ushort>(bytes)); break;
                case 4: BinaryPrimitives.ReverseEndianness(MemoryMarshal.Cast<byte, uint>(bytes), MemoryMarshal.Cast<byte, uint>(bytes)); break;
                default: BinaryPrimitives.ReverseEndianness(MemoryMarshal.Cast<byte, ulong>(bytes), MemoryMarshal.Cast<byte, ulong>(bytes)); break;
            }
        }

        return array;
    }

    // The most dimensions a GGUF tensor has (ggml's GGML_MAX_DIMS).
    private const int MaxDimensions = 4;

    // A header count of entries of at least itemBytes bytes each must fit in the rest of the file.
    private void CheckEntries(long count, int itemBytes, string what)
    {
        long left = _file.Length - _position;
        if (count < 0 || count > left / itemBytes)
        {
            throw new InvalidDataException($"{Path}: the header lists {count} {what}, which do not fit in the {left} bytes left.");
        }
    }

    // An array of count items of at least itemBytes bytes each must fit in the rest of the file.
    private void CheckCount(long count, int itemBytes)
    {
        long left = _file.Length - _position;
        if (count < 0 || count > left / itemBytes || count > Array.MaxLength)
        {
            throw new InvalidDataException($"{Path}: a metadata array of {count} items at byte {_position} does not fit in the {left} bytes left.");
        }
    }

    private ReadOnlySpan<byte> Bytes(int count)
    {
        var span = _scratch.AsSpan(0, count);
        ReadAt(span, _position);
        _position += count;
        return span;
    }

    private uint U32() => BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4));

    private ulong U64() => BinaryPrimitives.ReadUInt64LittleEndian(Bytes(8));

    private string Str()
    {
        ulong length = U64();
        if (length > 1 << 28 || length > (ulong)(_file.Length - _position))
        {
            throw new InvalidDataException($"{Path}: a string of {length} bytes at byte {_position} (the file has {_file.Length - _position} bytes left).");
        }

        // Read into the stack or a pooled buffer: only the string is kept.
        int count = (int)length;
        byte[]? rented = null;
        Span<byte> bytes = count <= 256 ? stackalloc byte[256] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(count));
        try
        {
            ReadAt(bytes[..count], _position);
            _position += count;
            return Encoding.UTF8.GetString(bytes[..count]);
        }
        finally
        {
            if (rented is not null)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(rented);
            }
        }
    }

    private void ReadAt(Span<byte> destination, long offset)
    {
        while (!destination.IsEmpty)
        {
            int read = RandomAccess.Read(_file.SafeFileHandle, destination, offset);
            if (read <= 0)
            {
                throw new EndOfStreamException($"{Path} ends early.");
            }

            destination = destination[read..];
            offset += read;
        }
    }
}
