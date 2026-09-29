using System.Buffers.Binary;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Pretrained;

/// <summary>A tensor's entry in a GGUF file.</summary>
/// <param name="Name">Its name (llama.cpp's naming, e.g. blk.0.attn_q.weight).</param>
/// <param name="Type">The ggml type (0 F32, 1 F16, 8 Q8_0, 12 Q4_K, 14 Q6_K, 30 BF16 …).</param>
/// <param name="Dimensions">Sizes, innermost first (ggml's ne: [columns, rows, …]).</param>
/// <param name="Offset">Byte offset of its data in the file.</param>
public sealed record GgufTensorInfo(string Name, int Type, long[] Dimensions, long Offset)
{
    /// <summary>Number of elements.</summary>
    public long Count => Dimensions.Aggregate(1L, (a, b) => a * b);

    /// <summary>The shape outermost first, as PyTorch / safetensors write it ([rows, columns]).</summary>
    public int[] Shape => [.. Dimensions.Reverse().Select(d => checked((int)d))];
}

/// <summary>
/// Reads GGUF files (the format of llama.cpp and Ollama): metadata, the tensor index, and tensors dequantized to
/// float32 from F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL and IQ4_XS.
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
    /// Metadata by key: numbers as long / ulong / double, strings, booleans, and arrays (string[], long[], double[],
    /// bool[] or object[]).
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

        return value switch
        {
            T t => t,
            long l when typeof(T) == typeof(int) => (T)(object)checked((int)l),
            ulong u when typeof(T) == typeof(int) => (T)(object)checked((int)u),
            ulong u when typeof(T) == typeof(long) => (T)(object)checked((long)u),
            long l when typeof(T) == typeof(double) => (T)(object)(double)l,
            double d when typeof(T) == typeof(float) => (T)(object)(float)d,
            long[] a when typeof(T) == typeof(int) && a.Length > 0 => (T)(object)checked((int)a[0]),   // per-layer values: the first
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

    /// <summary>The tensor <paramref name="name"/> dequantized to float32 (row-major, as stored).</summary>
    public float[] Read(string name)
    {
        var info = _tensors.TryGetValue(name, out var t) ? t : throw new KeyNotFoundException($"No tensor named '{name}' in {Path}.");
        long count = info.Count;
        var (blockValues, blockBytes) = BlockSize(info.Type, name);
        if (count % blockValues != 0)
        {
            throw new InvalidDataException($"{name}: {count} values is not a whole number of {blockValues}-value blocks.");
        }

        long bytes = count / blockValues * blockBytes;
        var raw = GC.AllocateUninitializedArray<byte>(checked((int)bytes));
        ReadAt(raw, info.Offset);
        var values = GC.AllocateUninitializedArray<float>(checked((int)count));
        Dequantize(info.Type, raw, values);
        return values;
    }

    /// <summary>Values per block and bytes per block of a ggml type.</summary>
    public static (int Values, int Bytes) BlockSize(int type, string? name = null) => type switch
    {
        0 => (1, 4),
        1 or 30 => (1, 2),
        2 => (32, 18),
        3 => (32, 20),
        6 => (32, 22),
        7 => (32, 24),
        8 => (32, 34),
        10 => (256, 84),
        11 => (256, 110),
        12 => (256, 144),
        13 => (256, 176),
        14 => (256, 210),
        20 => (32, 18),
        23 => (256, 136),
        _ => throw new NotSupportedException($"{(name is null ? "" : name + ": ")}ggml type {TypeName(type)} is not supported yet "
            + "(supported: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K–Q6_K, IQ4_NL, IQ4_XS)."),
    };

    /// <summary>A ggml type's name.</summary>
    public static string TypeName(int type) => type switch
    {
        0 => "F32", 1 => "F16", 2 => "Q4_0", 3 => "Q4_1", 6 => "Q5_0", 7 => "Q5_1", 8 => "Q8_0", 9 => "Q8_1", 10 => "Q2_K", 11 => "Q3_K",
        12 => "Q4_K", 13 => "Q5_K", 14 => "Q6_K", 15 => "Q8_K", 16 => "IQ2_XXS", 17 => "IQ2_XS", 18 => "IQ3_XXS", 19 => "IQ1_S", 20 => "IQ4_NL",
        21 => "IQ3_S", 22 => "IQ2_S", 23 => "IQ4_XS", 24 => "I8", 25 => "I16", 26 => "I32", 27 => "I64", 28 => "F64", 29 => "IQ1_M", 30 => "BF16",
        34 => "TQ1_0", 35 => "TQ2_0", 39 => "MXFP4", _ => $"#{type}",
    };

    /// <summary>Dequantizes blocks of <paramref name="type"/> into float32 values (in parallel over blocks).</summary>
    public static void Dequantize(int type, ReadOnlySpan<byte> raw, Span<float> values)
    {
        var (blockValues, blockBytes) = BlockSize(type);
        switch (type)
        {
            case 0:
                MemoryMarshal.Cast<byte, float>(raw)[..values.Length].CopyTo(values);
                return;
            case 1:
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(raw[(2 * i)..]);
                }

                return;
            case 30:
                for (int i = 0; i < values.Length; i++)
                {
                    values[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(raw[(2 * i)..]) << 16);
                }

                return;
        }

        // Blocks are independent: split them across threads (spans cannot be captured, so work on arrays).
        int blocks = values.Length / blockValues;
        var input = raw.ToArray();
        var output = new float[values.Length];
        Parallel.For(0, (blocks + 1023) / 1024, chunk =>
        {
            int first = chunk * 1024, last = Math.Min(blocks, first + 1024);
            for (int b = first; b < last; b++)
            {
                DequantizeBlock(type, input.AsSpan(b * blockBytes, blockBytes), output.AsSpan(b * blockValues, blockValues));
            }
        });
        output.CopyTo(values);
    }

    private static readonly sbyte[] Iq4Values = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

    private static float Half(ReadOnlySpan<byte> b) => (float)BinaryPrimitives.ReadHalfLittleEndian(b);

    // One block, as ggml's dequantize_row_* functions do it.
    private static void DequantizeBlock(int type, ReadOnlySpan<byte> x, Span<float> y)
    {
        switch (type)
        {
            case 2:                                                 // Q4_0: d, 16 bytes of nibbles
            {
                float d = Half(x);
                var qs = x[2..];
                for (int j = 0; j < 16; j++)
                {
                    y[j] = ((qs[j] & 0xF) - 8) * d;
                    y[j + 16] = ((qs[j] >> 4) - 8) * d;
                }

                return;
            }

            case 3:                                                 // Q4_1: d, m, nibbles
            {
                float d = Half(x), m = Half(x[2..]);
                var qs = x[4..];
                for (int j = 0; j < 16; j++)
                {
                    y[j] = (qs[j] & 0xF) * d + m;
                    y[j + 16] = (qs[j] >> 4) * d + m;
                }

                return;
            }

            case 6:                                                 // Q5_0: d, 32 high bits, nibbles
            {
                float d = Half(x);
                uint qh = BinaryPrimitives.ReadUInt32LittleEndian(x[2..]);
                var qs = x[6..];
                for (int j = 0; j < 16; j++)
                {
                    int h0 = (int)(((qh >> j) << 4) & 0x10), h1 = (int)((qh >> (j + 12)) & 0x10);
                    y[j] = (((qs[j] & 0x0F) | h0) - 16) * d;
                    y[j + 16] = (((qs[j] >> 4) | h1) - 16) * d;
                }

                return;
            }

            case 7:                                                 // Q5_1: d, m, high bits, nibbles
            {
                float d = Half(x), m = Half(x[2..]);
                uint qh = BinaryPrimitives.ReadUInt32LittleEndian(x[4..]);
                var qs = x[8..];
                for (int j = 0; j < 16; j++)
                {
                    int h0 = (int)(((qh >> j) << 4) & 0x10), h1 = (int)((qh >> (j + 12)) & 0x10);
                    y[j] = ((qs[j] & 0x0F) | h0) * d + m;
                    y[j + 16] = ((qs[j] >> 4) | h1) * d + m;
                }

                return;
            }

            case 8:                                                 // Q8_0: d, 32 signed bytes
            {
                float d = Half(x);
                for (int j = 0; j < 32; j++)
                {
                    y[j] = (sbyte)x[2 + j] * d;
                }

                return;
            }

            case 10:                                                // Q2_K: scales[16], qs[64], d, dmin
            {
                var scales = x[..16];
                var q = x[16..80];
                float d = Half(x[80..]), min = Half(x[82..]);
                int o = 0, s = 0;
                for (int n = 0; n < 256; n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; j++)
                    {
                        byte sc = scales[s++];
                        float dl = d * (sc & 0xF), ml = min * (sc >> 4);
                        for (int l = 0; l < 16; l++)
                        {
                            y[o++] = dl * ((q[l] >> shift) & 3) - ml;
                        }

                        sc = scales[s++];
                        dl = d * (sc & 0xF);
                        ml = min * (sc >> 4);
                        for (int l = 0; l < 16; l++)
                        {
                            y[o++] = dl * ((q[l + 16] >> shift) & 3) - ml;
                        }

                        shift += 2;
                    }

                    q = q[32..];
                }

                return;
            }

            case 11:                                                // Q3_K: hmask[32], qs[64], scales[12], d
            {
                var hm = x[..32];
                var q = x[32..96];
                float dAll = Half(x[108..]);
                const uint kmask1 = 0x03030303, kmask2 = 0x0f0f0f0f;
                Span<uint> aux = stackalloc uint[4];
                aux[0] = BinaryPrimitives.ReadUInt32LittleEndian(x[96..]);
                aux[1] = BinaryPrimitives.ReadUInt32LittleEndian(x[100..]);
                aux[2] = BinaryPrimitives.ReadUInt32LittleEndian(x[104..]);
                uint tmp = aux[2];
                aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
                var scales = MemoryMarshal.Cast<uint, sbyte>(aux);
                int o = 0, s = 0;
                byte m = 1;
                for (int n = 0; n < 256; n += 128)
                {
                    int shift = 0;
                    for (int j = 0; j < 4; j++)
                    {
                        float dl = dAll * (scales[s++] - 32);
                        for (int l = 0; l < 16; l++)
                        {
                            y[o++] = dl * (((q[l] >> shift) & 3) - ((hm[l] & m) != 0 ? 0 : 4));
                        }

                        dl = dAll * (scales[s++] - 32);
                        for (int l = 0; l < 16; l++)
                        {
                            y[o++] = dl * (((q[l + 16] >> shift) & 3) - ((hm[l + 16] & m) != 0 ? 0 : 4));
                        }

                        shift += 2;
                        m <<= 1;
                    }

                    q = q[32..];
                }

                return;
            }

            case 12:                                                // Q4_K: d, dmin, scales[12], qs[128]
            {
                float d = Half(x), min = Half(x[2..]);
                var scales = x[4..16];
                var q = x[16..];
                int o = 0, s = 0;
                for (int j = 0; j < 256; j += 64)
                {
                    var (sc1, m1) = ScaleMin(s, scales);
                    var (sc2, m2) = ScaleMin(s + 1, scales);
                    float d1 = d * sc1, mm1 = min * m1, d2 = d * sc2, mm2 = min * m2;
                    for (int l = 0; l < 32; l++)
                    {
                        y[o++] = d1 * (q[l] & 0xF) - mm1;
                    }

                    for (int l = 0; l < 32; l++)
                    {
                        y[o++] = d2 * (q[l] >> 4) - mm2;
                    }

                    q = q[32..];
                    s += 2;
                }

                return;
            }

            case 13:                                                // Q5_K: d, dmin, scales[12], qh[32], qs[128]
            {
                float d = Half(x), min = Half(x[2..]);
                var scales = x[4..16];
                var qh = x[16..48];
                var ql = x[48..];
                int o = 0, s = 0;
                int u1 = 1, u2 = 2;
                for (int j = 0; j < 256; j += 64)
                {
                    var (sc1, m1) = ScaleMin(s, scales);
                    var (sc2, m2) = ScaleMin(s + 1, scales);
                    float d1 = d * sc1, mm1 = min * m1, d2 = d * sc2, mm2 = min * m2;
                    for (int l = 0; l < 32; l++)
                    {
                        y[o++] = d1 * ((ql[l] & 0xF) + ((qh[l] & u1) != 0 ? 16 : 0)) - mm1;
                    }

                    for (int l = 0; l < 32; l++)
                    {
                        y[o++] = d2 * ((ql[l] >> 4) + ((qh[l] & u2) != 0 ? 16 : 0)) - mm2;
                    }

                    ql = ql[32..];
                    s += 2;
                    u1 <<= 2;
                    u2 <<= 2;
                }

                return;
            }

            case 14:                                                // Q6_K: ql[128], qh[64], scales[16], d
            {
                var ql = x[..128];
                var qh = x[128..192];
                var sc = MemoryMarshal.Cast<byte, sbyte>(x[192..208]);
                float d = Half(x[208..]);
                int o = 0;
                for (int n = 0; n < 256; n += 128)
                {
                    for (int l = 0; l < 32; l++)
                    {
                        int s = l / 16;
                        int q1 = ((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        int q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        int q3 = ((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        int q4 = ((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        y[o + l] = d * sc[s] * q1;
                        y[o + l + 32] = d * sc[s + 2] * q2;
                        y[o + l + 64] = d * sc[s + 4] * q3;
                        y[o + l + 96] = d * sc[s + 6] * q4;
                    }

                    o += 128;
                    ql = ql[64..];
                    qh = qh[32..];
                    sc = sc[8..];
                }

                return;
            }

            case 20:                                                // IQ4_NL: d, nibbles into a non-linear table
            {
                float d = Half(x);
                var qs = x[2..];
                for (int j = 0; j < 16; j++)
                {
                    y[j] = d * Iq4Values[qs[j] & 0xF];
                    y[j + 16] = d * Iq4Values[qs[j] >> 4];
                }

                return;
            }

            case 23:                                                // IQ4_XS: d, scales_h, scales_l[4], qs[128]
            {
                float d = Half(x);
                int scalesH = BinaryPrimitives.ReadUInt16LittleEndian(x[2..]);
                var scalesL = x[4..8];
                var qs = x[8..];
                for (int ib = 0; ib < 8; ib++)
                {
                    int ls = ((scalesL[ib / 2] >> (4 * (ib % 2))) & 0xF) | (((scalesH >> (2 * ib)) & 3) << 4);
                    float dl = d * (ls - 32);
                    for (int j = 0; j < 16; j++)
                    {
                        y[ib * 32 + j] = dl * Iq4Values[qs[j] & 0xF];
                        y[ib * 32 + j + 16] = dl * Iq4Values[qs[j] >> 4];
                    }

                    qs = qs[16..];
                }

                return;
            }

            default:
                throw new NotSupportedException($"ggml type {TypeName(type)} is not supported.");
        }
    }

    // The 6-bit scale and minimum of sub-block j in a K-quant's 12 packed bytes.
    private static (int Scale, int Min) ScaleMin(int j, ReadOnlySpan<byte> q) => j < 4
        ? (q[j] & 63, q[j + 4] & 63)
        : ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4));

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
        for (long i = 0; i < keys; i++)
        {
            string key = Str();
            Metadata[key] = Value((int)U32(), 0);
        }

        var infos = new List<(string Name, int Type, long[] Dims, long Offset)>();
        for (long i = 0; i < tensors; i++)
        {
            string name = Str();
            int dims = (int)U32();
            var shape = new long[dims];
            for (int d = 0; d < dims; d++)
            {
                shape[d] = checked((long)U64());
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
                    case 8:
                        var strings = new string[count];
                        for (long i = 0; i < count; i++)
                        {
                            strings[i] = Str();
                        }

                        return strings;
                    case 6 or 12:
                        var doubles = new double[count];
                        for (long i = 0; i < count; i++)
                        {
                            doubles[i] = (double)Value(itemType, depth + 1);
                        }

                        return doubles;
                    case 7:
                        var bools = new bool[count];
                        for (long i = 0; i < count; i++)
                        {
                            bools[i] = (bool)Value(itemType, depth + 1);
                        }

                        return bools;
                    case 9:
                        var items = new object[count];
                        for (long i = 0; i < count; i++)
                        {
                            items[i] = Value(itemType, depth + 1);
                        }

                        return items;
                    default:
                        var longs = new long[count];
                        for (long i = 0; i < count; i++)
                        {
                            var item = Value(itemType, depth + 1);
                            longs[i] = item is long l ? l : unchecked((long)(ulong)item);
                        }

                        return longs;
                }

            default:
                throw new InvalidDataException($"{Path}: unknown metadata type {type}.");
        }
    }

    private readonly byte[] _scratch = new byte[8];

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
        long length = checked((long)U64());
        if (length > 1 << 28)
        {
            throw new InvalidDataException($"{Path}: a string of {length} bytes in the metadata.");
        }

        var bytes = new byte[length];
        ReadAt(bytes, _position);
        _position += length;
        return Encoding.UTF8.GetString(bytes);
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
