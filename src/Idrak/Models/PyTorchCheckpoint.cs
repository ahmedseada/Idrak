// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.IO.Compression;
using System.Text;

namespace Idrak.Models;

/// <summary>
/// The tensors of a PyTorch checkpoint (<c>torch.save</c>'s zip format: <c>.pth</c>, <c>.pt</c>, <c>.bin</c>,
/// <c>.ckpt</c>), read without running anything in it. The pickle that describes the tensors is interpreted by a
/// restricted machine: data opcodes only (dictionaries, lists, tuples, numbers, strings), and of the names a pickle can
/// call, only those that rebuild a tensor or a dictionary (<c>collections.OrderedDict</c>,
/// <c>torch._utils._rebuild_tensor_v2</c>, <c>_rebuild_parameter</c>, the <c>torch.*Storage</c> types). Any other
/// name or opcode is refused, naming it: a pickle that would run code is never run. The values are read from the zip's
/// storage entries when a tensor is read, one tensor at a time.
/// </summary>
/// <remarks>
/// Names are the dictionary keys joined by '.'. A checkpoint that wraps its weights (<c>{"model": state_dict, ...}</c>
/// or <c>"state_dict"</c>, as training scripts save them) is read from that entry unless a root is given; the other
/// entries (an optimizer's state, the step) are left out. The legacy (pre-zip) format and big-endian storages are refused.
/// </remarks>
public sealed class PyTorchCheckpointReader : ITensorStore
{
    /// <summary>The file extensions read as PyTorch checkpoints.</summary>
    public static IReadOnlyList<string> Extensions { get; } = [".pth", ".pt", ".bin", ".ckpt"];

    /// <summary>The entries a wrapped checkpoint keeps its weights under, asked in this order when no root is given.</summary>
    public static IReadOnlyList<string> WrapperKeys { get; } = ["model", "state_dict", "module"];

    private readonly ZipArchive _zip;
    private readonly string _prefix;
    private readonly Dictionary<string, TensorEntry> _tensors;
    private readonly Lock _gate = new();

    private PyTorchCheckpointReader(string path, ZipArchive zip, string prefix, Dictionary<string, TensorEntry> tensors, string? root)
    {
        Path = path;
        _zip = zip;
        _prefix = prefix;
        _tensors = tensors;
        Root = root;
    }

    /// <summary>The checkpoint file.</summary>
    public string Path { get; }

    /// <summary>The entry the tensors were read from ("model" for <c>{"model": state_dict}</c>), or null for the whole checkpoint.</summary>
    public string? Root { get; }

    /// <summary>Whether <paramref name="path"/> is a PyTorch zip checkpoint (a zip holding a <c>data.pkl</c>), without reading its pickle.</summary>
    public static bool IsCheckpoint(string path)
    {
        if (!File.Exists(path) || !Extensions.Contains(System.IO.Path.GetExtension(path), StringComparer.OrdinalIgnoreCase))
        {
            return false;
        }

        try
        {
            using var zip = ZipFile.OpenRead(path);
            return zip.Entries.Any(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal) || e.FullName == "data.pkl");
        }
        catch (InvalidDataException)
        {
            return false;
        }
    }

    /// <summary>
    /// Opens <paramref name="path"/>. <paramref name="root"/>: the dotted entry the weights are under ("model"), "" for
    /// the whole checkpoint, or null to take the first of <see cref="WrapperKeys"/> that holds tensors, else the whole.
    /// </summary>
    /// <exception cref="InvalidDataException">It is not a zip checkpoint, the pickle uses an opcode or a name outside the allowlist, or a storage is missing.</exception>
    public static PyTorchCheckpointReader Open(string path, string? root = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        ZipArchive zip;
        try
        {
            zip = ZipFile.OpenRead(path);
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException($"{path} is not a zip PyTorch checkpoint (torch.save's format since PyTorch 1.6); the legacy format is not read: "
                + "load and save it again with a current PyTorch, or convert it to safetensors.", e);
        }

        try
        {
            var pickle = zip.Entries.FirstOrDefault(e => e.FullName.EndsWith("/data.pkl", StringComparison.Ordinal) || e.FullName == "data.pkl")
                ?? throw new InvalidDataException($"{path} holds no data.pkl: not a PyTorch checkpoint.");
            string prefix = pickle.FullName[..^"data.pkl".Length];
            if (zip.GetEntry(prefix + "byteorder") is { } order)
            {
                using var reader = new StreamReader(order.Open());
                if (reader.ReadToEnd().Trim() is { } byteOrder && byteOrder != "little")
                {
                    throw new InvalidDataException($"{path} stores its tensors {byteOrder}-endian; only little-endian checkpoints are read.");
                }
            }

            byte[] bytes;
            using (var stream = pickle.Open())
            using (var memory = new MemoryStream())
            {
                stream.CopyTo(memory);
                bytes = memory.ToArray();
            }

            object? value = new Unpickler(bytes, path).Load();
            var (chosen, rootName) = ChooseRoot(value, root, path);
            var tensors = new Dictionary<string, TensorEntry>(StringComparer.Ordinal);
            Flatten(chosen, "", tensors);
            if (tensors.Count == 0)
            {
                throw new InvalidDataException($"{path} holds no tensors{(rootName is null ? "" : $" under '{rootName}'")}.");
            }

            return new PyTorchCheckpointReader(path, zip, prefix, tensors, rootName);
        }
        catch
        {
            zip.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    public IEnumerable<string> Names => _tensors.Keys;

    /// <inheritdoc />
    public bool Contains(string name) => _tensors.ContainsKey(name);

    /// <inheritdoc />
    public int[] ShapeOf(string name) => (int[])Entry(name).Shape.Clone();

    /// <inheritdoc />
    public WeightFormat? FormatOf(string name) => Entry(name).Storage.Type switch
    {
        StorageType.Float32 => WeightFormat.Float32,
        StorageType.Float16 => WeightFormat.Float16,
        StorageType.BFloat16 => WeightFormat.BFloat16,
        _ => null,
    };

    /// <inheritdoc />
    public float[] Read(string name)
    {
        var entry = Entry(name);
        long count = 1;
        foreach (int d in entry.Shape)
        {
            count *= d;
        }

        var values = new float[count];
        if (count == 0)
        {
            return values;
        }

        int size = Size(entry.Storage.Type);
        long first = entry.Offset, last = entry.Offset;                         // the span of storage elements the view touches
        for (int d = 0; d < entry.Shape.Length; d++)
        {
            long reach = (entry.Shape[d] - 1) * entry.Stride[d];
            if (reach < 0)
            {
                first += reach;
            }
            else
            {
                last += reach;
            }
        }

        if (first < 0 || last >= entry.Storage.Elements)
        {
            throw new InvalidDataException($"{Path}: tensor '{name}' reaches elements {first} to {last} of a storage of {entry.Storage.Elements}.");
        }

        var raw = new byte[(last - first + 1) * size];
        lock (_gate)
        {
            var zipEntry = _zip.GetEntry(_prefix + "data/" + entry.Storage.Key)
                ?? throw new InvalidDataException($"{Path} has no storage '{entry.Storage.Key}' for tensor '{name}'.");
            using var stream = zipEntry.Open();
            Skip(stream, first * size);
            stream.ReadExactly(raw);
        }

        if (IsContiguous(entry.Shape, entry.Stride))
        {
            Convert(raw, 0, entry.Storage.Type, values, 0, values.Length);
            return values;
        }

        // A strided view: each element gathered by its index's offset.
        var index = new int[entry.Shape.Length];
        for (long i = 0; i < count; i++)
        {
            long at = entry.Offset - first;
            for (int d = 0; d < index.Length; d++)
            {
                at += index[d] * entry.Stride[d];
            }

            Convert(raw, at * size, entry.Storage.Type, values, i, 1);
            for (int d = index.Length - 1; d >= 0 && ++index[d] == entry.Shape[d]; d--)
            {
                index[d] = 0;
            }
        }

        return values;
    }

    /// <inheritdoc />
    public void Dispose() => _zip.Dispose();

    private TensorEntry Entry(string name) =>
        _tensors.TryGetValue(name, out var entry) ? entry : throw new InvalidDataException($"{Path} has no tensor '{name}'.");

    private static bool IsContiguous(int[] shape, long[] stride)
    {
        long expected = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            if (shape[d] != 1 && stride[d] != expected)
            {
                return false;
            }

            expected *= shape[d];
        }

        return true;
    }

    private static void Skip(Stream stream, long bytes)
    {
        if (stream.CanSeek)
        {
            stream.Seek(bytes, SeekOrigin.Current);
            return;
        }

        var buffer = new byte[Math.Min(bytes, 1 << 16)];
        while (bytes > 0)
        {
            int n = stream.Read(buffer, 0, (int)Math.Min(buffer.Length, bytes));
            if (n == 0)
            {
                throw new EndOfStreamException("A storage ends before the tensor's values.");
            }

            bytes -= n;
        }
    }

    private static int Size(StorageType type) => type switch
    {
        StorageType.Float64 or StorageType.Int64 => 8,
        StorageType.Float32 or StorageType.Int32 => 4,
        StorageType.Float16 or StorageType.BFloat16 or StorageType.Int16 => 2,
        _ => 1,
    };

    // `count` values of `type` from raw[byteOffset..] into target[at..].
    private static void Convert(byte[] raw, long byteOffset, StorageType type, float[] target, long at, long count)
    {
        var source = raw.AsSpan((int)byteOffset);
        var destination = target.AsSpan((int)at, (int)count);
        switch (type)
        {
            case StorageType.Float32:
                System.Runtime.InteropServices.MemoryMarshal.Cast<byte, float>(source[..(destination.Length * 4)]).CopyTo(destination);
                break;
            case StorageType.Float16:
                for (int i = 0; i < destination.Length; i++) destination[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(source[(2 * i)..]);
                break;
            case StorageType.BFloat16:
                for (int i = 0; i < destination.Length; i++) destination[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(source[(2 * i)..]) << 16);
                break;
            case StorageType.Float64:
                for (int i = 0; i < destination.Length; i++) destination[i] = (float)BinaryPrimitives.ReadDoubleLittleEndian(source[(8 * i)..]);
                break;
            case StorageType.Int64:
                for (int i = 0; i < destination.Length; i++) destination[i] = BinaryPrimitives.ReadInt64LittleEndian(source[(8 * i)..]);
                break;
            case StorageType.Int32:
                for (int i = 0; i < destination.Length; i++) destination[i] = BinaryPrimitives.ReadInt32LittleEndian(source[(4 * i)..]);
                break;
            case StorageType.Int16:
                for (int i = 0; i < destination.Length; i++) destination[i] = BinaryPrimitives.ReadInt16LittleEndian(source[(2 * i)..]);
                break;
            case StorageType.Int8:
                for (int i = 0; i < destination.Length; i++) destination[i] = (sbyte)source[i];
                break;
            default:                                                             // uint8 and bool
                for (int i = 0; i < destination.Length; i++) destination[i] = source[i];
                break;
        }
    }

    // The entry the tensors are taken from: the given dotted path, or the first wrapper key holding tensors, or the whole.
    private static (object? Value, string? Name) ChooseRoot(object? value, string? root, string path)
    {
        if (root is not null)
        {
            if (root.Length == 0)
            {
                return (value, null);
            }

            object? at = value;
            foreach (string part in root.Split('.'))
            {
                at = at is PickleDict d && d.TryGet(part, out var next) ? next
                    : throw new InvalidDataException($"{path} has no entry '{root}' (its keys: {Keys(value)}).");
            }

            return (at, root);
        }

        if (value is PickleDict top && !HoldsTensorsDirectly(top))
        {
            foreach (string key in WrapperKeys)
            {
                if (top.TryGet(key, out var inner) && inner is PickleDict d && CountTensors(d) > 0)
                {
                    return (inner, key);
                }
            }
        }

        return (value, null);
    }

    private static string Keys(object? value) => value is PickleDict d ? string.Join(", ", d.Items.Select(i => i.Key?.ToString()).Take(12)) : "none";

    private static bool HoldsTensorsDirectly(PickleDict d) => d.Items.Any(i => i.Value is TensorEntry);

    private static int CountTensors(object? value) => value switch
    {
        TensorEntry => 1,
        PickleDict d => d.Items.Sum(i => CountTensors(i.Value)),
        List<object?> l => l.Sum(CountTensors),
        object?[] t => t.Sum(CountTensors),
        _ => 0,
    };

    private static void Flatten(object? value, string prefix, Dictionary<string, TensorEntry> into)
    {
        switch (value)
        {
            case TensorEntry tensor:
                into[prefix.Length == 0 ? "tensor" : prefix] = tensor;
                break;
            case PickleDict d:
                foreach (var (key, item) in d.Items)
                {
                    string part = key switch
                    {
                        string s => s,
                        long n => n.ToString(CultureInfo.InvariantCulture),
                        _ => key?.ToString() ?? "None",
                    };
                    Flatten(item, prefix.Length == 0 ? part : prefix + "." + part, into);
                }

                break;
            case List<object?> list:
                for (int i = 0; i < list.Count; i++) Flatten(list[i], $"{prefix}{(prefix.Length == 0 ? "" : ".")}{i}", into);
                break;
            case object?[] tuple:
                for (int i = 0; i < tuple.Length; i++) Flatten(tuple[i], $"{prefix}{(prefix.Length == 0 ? "" : ".")}{i}", into);
                break;
        }
    }

    internal enum StorageType { Float32, Float16, BFloat16, Float64, Int64, Int32, Int16, Int8, UInt8, Bool }

    internal sealed record StorageRef(StorageType Type, string Key, long Elements);

    internal sealed record TensorEntry(StorageRef Storage, long Offset, int[] Shape, long[] Stride);

    // An ordered dictionary as the pickle builds it (a dict or an OrderedDict).
    internal sealed class PickleDict
    {
        public List<KeyValuePair<object?, object?>> Items { get; } = [];

        public void Set(object? key, object? value)
        {
            for (int i = 0; i < Items.Count; i++)
            {
                if (Equals(Items[i].Key, key))
                {
                    Items[i] = new(key, value);
                    return;
                }
            }

            Items.Add(new(key, value));
        }

        public bool TryGet(string key, out object? value)
        {
            foreach (var item in Items)
            {
                if (item.Key is string s && s == key)
                {
                    value = item.Value;
                    return true;
                }
            }

            value = null;
            return false;
        }
    }

    // A name the pickle refers to, already checked against the allowlist.
    private sealed record Global(string Module, string Name)
    {
        public string Full => Module + "." + Name;
    }

    private sealed class Mark
    {
        public static readonly Mark Instance = new();
    }

    // The restricted pickle machine (protocols 2 to 5, the opcodes torch.save writes).
    private sealed class Unpickler(byte[] data, string path)
    {
        private static readonly Dictionary<string, StorageType> Storages = new(StringComparer.Ordinal)
        {
            ["FloatStorage"] = StorageType.Float32, ["HalfStorage"] = StorageType.Float16, ["BFloat16Storage"] = StorageType.BFloat16,
            ["DoubleStorage"] = StorageType.Float64, ["LongStorage"] = StorageType.Int64, ["IntStorage"] = StorageType.Int32,
            ["ShortStorage"] = StorageType.Int16, ["CharStorage"] = StorageType.Int8, ["ByteStorage"] = StorageType.UInt8, ["BoolStorage"] = StorageType.Bool,
        };

        private static readonly HashSet<string> Allowed = new(StringComparer.Ordinal)
        {
            "collections.OrderedDict", "torch._utils._rebuild_tensor_v2", "torch._utils._rebuild_tensor", "torch._utils._rebuild_parameter",
            "torch._utils._rebuild_parameter_with_state", "builtins.dict", "builtins.list", "builtins.tuple",
        };

        private readonly List<object?> _stack = [];
        private readonly Dictionary<long, object?> _memo = [];
        private int _at;

        public object? Load()
        {
            while (true)
            {
                byte op = Byte();
                switch (op)
                {
                    case 0x80: Byte(); break;                                    // PROTO
                    case 0x95: _at += 8; break;                                  // FRAME
                    case (byte)'.': return Pop();                                // STOP
                    case (byte)'(': _stack.Add(Mark.Instance); break;            // MARK
                    case (byte)'}': _stack.Add(new PickleDict()); break;         // EMPTY_DICT
                    case (byte)']': _stack.Add(new List<object?>()); break;      // EMPTY_LIST
                    case (byte)')': _stack.Add(Array.Empty<object?>()); break;   // EMPTY_TUPLE
                    case (byte)'t': _stack.Add(PopMark().ToArray()); break;      // TUPLE
                    case 0x85: _stack.Add(new[] { Pop() }); break;               // TUPLE1
                    case 0x86: { var b = Pop(); var a = Pop(); _stack.Add(new[] { a, b }); break; }
                    case 0x87: { var c = Pop(); var b = Pop(); var a = Pop(); _stack.Add(new[] { a, b, c }); break; }
                    case (byte)'J': _stack.Add((long)BinaryPrimitives.ReadInt32LittleEndian(Bytes(4))); break;
                    case (byte)'K': _stack.Add((long)Byte()); break;
                    case (byte)'M': _stack.Add((long)BinaryPrimitives.ReadUInt16LittleEndian(Bytes(2))); break;
                    case 0x8a: _stack.Add(Long(Bytes(Byte()))); break;           // LONG1
                    case 0x8b: _stack.Add(Long(Bytes(BinaryPrimitives.ReadInt32LittleEndian(Bytes(4))))); break;   // LONG4
                    case (byte)'N': _stack.Add(null); break;
                    case 0x88: _stack.Add(true); break;
                    case 0x89: _stack.Add(false); break;
                    case (byte)'G': _stack.Add(BinaryPrimitives.ReadDoubleBigEndian(Bytes(8))); break;
                    case (byte)'X': _stack.Add(Encoding.UTF8.GetString(Bytes(BinaryPrimitives.ReadInt32LittleEndian(Bytes(4))))); break;
                    case 0x8c: _stack.Add(Encoding.UTF8.GetString(Bytes(Byte()))); break;           // SHORT_BINUNICODE
                    case 0x8d: _stack.Add(Encoding.UTF8.GetString(Bytes(checked((int)BinaryPrimitives.ReadInt64LittleEndian(Bytes(8)))))); break;
                    case (byte)'U': _stack.Add(Encoding.Latin1.GetString(Bytes(Byte()))); break;     // SHORT_BINSTRING
                    case (byte)'T': _stack.Add(Encoding.Latin1.GetString(Bytes(BinaryPrimitives.ReadInt32LittleEndian(Bytes(4))))); break;
                    case (byte)'B': _stack.Add(Bytes(BinaryPrimitives.ReadInt32LittleEndian(Bytes(4))).ToArray()); break;
                    case (byte)'C': _stack.Add(Bytes(Byte()).ToArray()); break;
                    case (byte)'q': _memo[Byte()] = Top(); break;                // BINPUT
                    case (byte)'r': _memo[BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4))] = Top(); break;
                    case 0x94: _memo[_memo.Count] = Top(); break;                // MEMOIZE
                    case (byte)'h': _stack.Add(Memo(Byte())); break;             // BINGET
                    case (byte)'j': _stack.Add(Memo(BinaryPrimitives.ReadUInt32LittleEndian(Bytes(4)))); break;
                    case (byte)'0': Pop(); break;                                // POP
                    case (byte)'1': PopMark(); break;                            // POP_MARK
                    case (byte)'2': _stack.Add(Top()); break;                    // DUP
                    case (byte)'c': _stack.Add(Name(Line(), Line())); break;     // GLOBAL
                    case 0x93: { var name = Pop() as string; var module = Pop() as string; _stack.Add(Name(module ?? "", name ?? "")); break; }   // STACK_GLOBAL
                    case (byte)'Q': _stack.Add(Persistent(Pop())); break;        // BINPERSID
                    case (byte)'R': { var args = Pop(); var callable = Pop(); _stack.Add(Call(callable, args)); break; }   // REDUCE
                    case 0x81: { var args = Pop(); var cls = Pop(); _stack.Add(Call(cls, args)); break; }                  // NEWOBJ
                    case (byte)'b': { var state = Pop(); Build(Top(), state); break; }                                      // BUILD
                    case (byte)'s': { var value = Pop(); var key = Pop(); Dict(Top()).Set(key, value); break; }             // SETITEM
                    case (byte)'u': { var items = PopMark(); var dict = Dict(Top()); for (int i = 0; i + 1 < items.Count; i += 2) dict.Set(items[i], items[i + 1]); break; }
                    case (byte)'a': { var value = Pop(); List(Top()).Add(value); break; }                                   // APPEND
                    case (byte)'e': { var items = PopMark(); List(Top()).AddRange(items); break; }                          // APPENDS
                    default:
                        throw new InvalidDataException($"{path}: the pickle uses opcode 0x{op:x2} at byte {_at - 1}, which the restricted reader does not run "
                            + "(it reads tensors and their dictionaries only).");
                }
            }
        }

        private object Name(string module, string name)
        {
            if (module == "torch" && Storages.TryGetValue(name, out var type))
            {
                return type;
            }

            string full = module + "." + name;
            if (!Allowed.Contains(full))
            {
                throw new InvalidDataException($"{path}: the pickle names '{full}', which is not on the reader's allowlist (tensors, their storages and "
                    + "dictionaries); it is not run. Convert the checkpoint to safetensors if it is one you trust.");
            }

            return new Global(module, name);
        }

        private StorageRef Persistent(object? id)
        {
            if (id is not object?[] { Length: >= 5 } t || t[0] is not string kind || kind != "storage" || t[1] is not StorageType type || t[2] is not string key)
            {
                throw new InvalidDataException($"{path}: a persistent id the reader does not know ({Describe(id)}).");
            }

            return new StorageRef(type, key, t[4] is long n ? n : 0);
        }

        private object? Call(object? callable, object? args)
        {
            var a = args as object?[] ?? throw new InvalidDataException($"{path}: a call's arguments are not a tuple.");
            switch ((callable as Global)?.Full)
            {
                case "collections.OrderedDict" or "builtins.dict":
                    return new PickleDict();
                case "builtins.list":
                    return a.Length == 0 ? new List<object?>() : new List<object?>(a[0] as List<object?> ?? []);
                case "builtins.tuple":
                    return a.Length == 0 ? Array.Empty<object?>() : (a[0] as List<object?>)?.ToArray() ?? a[0] as object?[] ?? [];
                case "torch._utils._rebuild_tensor_v2" or "torch._utils._rebuild_tensor":
                    if (a.Length < 4 || a[0] is not StorageRef storage || a[1] is not long offset || a[2] is not object?[] size || a[3] is not object?[] stride)
                    {
                        throw new InvalidDataException($"{path}: a tensor's arguments are not (storage, offset, size, stride).");
                    }

                    return new TensorEntry(storage, offset, [.. size.Select(s => checked((int)(long)s!))], [.. stride.Select(s => (long)s!)]);
                case "torch._utils._rebuild_parameter" or "torch._utils._rebuild_parameter_with_state":
                    return a.Length > 0 && a[0] is TensorEntry tensor ? tensor : throw new InvalidDataException($"{path}: a parameter holds no tensor.");
                default:
                    throw new InvalidDataException($"{path}: the pickle calls {Describe(callable)}, which the reader does not run.");
            }
        }

        private void Build(object? target, object? state)
        {
            if (target is PickleDict dict && state is PickleDict items)
            {
                foreach (var (k, v) in items.Items) dict.Set(k, v);
            }

            // A tensor's or an OrderedDict's other state (hooks, metadata) is not needed to read values.
        }

        private static string Describe(object? value) => value switch
        {
            null => "None",
            Global g => g.Full,
            object?[] t => "(" + string.Join(", ", t.Select(Describe)) + ")",
            _ => value.GetType().Name,
        };

        private static long Long(ReadOnlySpan<byte> bytes)
        {
            if (bytes.Length == 0)
            {
                return 0;
            }

            if (bytes.Length > 8)
            {
                throw new InvalidDataException("An integer of more than 64 bits.");
            }

            long value = 0;
            for (int i = bytes.Length - 1; i >= 0; i--)
            {
                value = (value << 8) | bytes[i];
            }

            int shift = 64 - 8 * bytes.Length;
            return shift == 0 ? value : (value << shift) >> shift;               // sign-extended
        }

        private byte Byte() => _at < data.Length ? data[_at++] : throw new InvalidDataException($"{path}: the pickle ends early.");

        private ReadOnlySpan<byte> Bytes(int count)
        {
            if (count < 0 || _at + count > data.Length)
            {
                throw new InvalidDataException($"{path}: the pickle ends early.");
            }

            var span = data.AsSpan(_at, count);
            _at += count;
            return span;
        }

        private string Line()
        {
            int end = Array.IndexOf(data, (byte)'\n', _at);
            if (end < 0)
            {
                throw new InvalidDataException($"{path}: the pickle ends early.");
            }

            string line = Encoding.UTF8.GetString(data, _at, end - _at);
            _at = end + 1;
            return line;
        }

        private object? Memo(long key) => _memo.TryGetValue(key, out var v) ? v : throw new InvalidDataException($"{path}: the pickle refers to memo {key}, never stored.");

        private object? Top() => _stack.Count > 0 ? _stack[^1] : throw new InvalidDataException($"{path}: the pickle's stack is empty.");

        private object? Pop()
        {
            var value = Top();
            _stack.RemoveAt(_stack.Count - 1);
            return value;
        }

        private List<object?> PopMark()
        {
            int mark = _stack.FindLastIndex(v => ReferenceEquals(v, Mark.Instance));
            if (mark < 0)
            {
                throw new InvalidDataException($"{path}: the pickle has no mark to pop to.");
            }

            var items = _stack.GetRange(mark + 1, _stack.Count - mark - 1);
            _stack.RemoveRange(mark, _stack.Count - mark);
            return items;
        }

        private PickleDict Dict(object? value) => value as PickleDict ?? throw new InvalidDataException($"{path}: the pickle sets an item on {Describe(value)}, not a dictionary.");

        private List<object?> List(object? value) => value as List<object?> ?? throw new InvalidDataException($"{path}: the pickle appends to {Describe(value)}, not a list.");
    }
}

// A folder holding one PyTorch zip checkpoint (and no safetensors), or the checkpoint file itself (beside its config.json).
internal sealed class PyTorchCheckpointFormat : ICheckpointFormat
{
    public string Name => "pytorch";

    public bool CanOpen(string path) => PyTorchCheckpointReader.IsCheckpoint(path)
        || Directory.Exists(path) && !Directory.EnumerateFiles(path, "*.safetensors").Any() && Checkpoints(path).Count == 1;

    public string Prepare(string path) => File.Exists(path) ? System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(path))! : path;

    public ITensorStore Open(string folder)
    {
        var found = Checkpoints(folder);
        return found.Count == 1 ? PyTorchCheckpointReader.Open(found[0])
            : throw new InvalidDataException(found.Count == 0
                ? $"{folder} holds no PyTorch checkpoint ({string.Join(", ", PyTorchCheckpointReader.Extensions)})."
                : $"{folder} holds several PyTorch checkpoints ({string.Join(", ", found.Select(System.IO.Path.GetFileName))}); keep one, or open it with PyTorchCheckpointReader.Open.");
    }

    public IEnumerable<string> Notes(string folder) =>
        Checkpoints(folder) is [var file] ? [$"weights read from the PyTorch checkpoint {System.IO.Path.GetFileName(file)} by the restricted reader (nothing in it is run)"] : [];

    private static List<string> Checkpoints(string folder) =>
        [.. Directory.EnumerateFiles(folder).Where(PyTorchCheckpointReader.IsCheckpoint).Order(StringComparer.Ordinal)];
}
