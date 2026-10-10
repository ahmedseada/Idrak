// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;

namespace Idrak.Onnx;

/// <summary>Reads Protocol Buffers fields: each call to <see cref="Next"/> yields (field number, wire type) and the value is read with the matching method.</summary>
internal ref struct ProtoReader(ReadOnlySpan<byte> data)
{
    private readonly ReadOnlySpan<byte> _data = data;
    private int _position;

    public bool Next(out int field, out int wireType)
    {
        if (_position >= _data.Length)
        {
            field = wireType = 0;
            return false;
        }

        ulong tag = Varint();
        field = (int)(tag >> 3);
        wireType = (int)(tag & 7);
        return true;
    }

    public ulong Varint()
    {
        ulong result = 0;
        for (int shift = 0; ; shift += 7)
        {
            if (_position >= _data.Length || shift > 63)
            {
                throw new InvalidDataException("Truncated or invalid ONNX file (bad varint).");
            }

            byte b = _data[_position++];
            result |= (ulong)(b & 0x7F) << shift;
            if (b < 0x80)
            {
                return result;
            }
        }
    }

    public float Fixed32()
    {
        float value = BinaryPrimitives.ReadSingleLittleEndian(_data.Slice(_position, 4));
        _position += 4;
        return value;
    }

    public double Fixed64()
    {
        double value = BinaryPrimitives.ReadDoubleLittleEndian(_data.Slice(_position, 8));
        _position += 8;
        return value;
    }

    public ReadOnlySpan<byte> Bytes()
    {
        int length = checked((int)Varint());
        if (length < 0 || _position + length > _data.Length)
        {
            throw new InvalidDataException("Truncated ONNX file.");
        }

        var bytes = _data.Slice(_position, length);
        _position += length;
        return bytes;
    }

    public string String() => Encoding.UTF8.GetString(Bytes());

    public void Skip(int wireType)
    {
        switch (wireType)
        {
            case 0: Varint(); break;
            case 1: _position += 8; break;
            case 2: Bytes(); break;
            case 5: _position += 4; break;
            default: throw new InvalidDataException($"Unsupported protobuf wire type {wireType}.");
        }
    }

    // A repeated integer field, packed (wire type 2) or not (wire type 0).
    public void Ints(int wireType, List<long> into)
    {
        if (wireType == 0)
        {
            into.Add((long)Varint());
            return;
        }

        var packed = new ProtoReader(Bytes());
        while (packed._position < packed._data.Length)
        {
            into.Add((long)packed.Varint());
        }
    }

    // A repeated float field, packed (wire type 2) or not (wire type 5).
    public void Floats(int wireType, List<float> into)
    {
        if (wireType == 5)
        {
            into.Add(Fixed32());
            return;
        }

        var packed = Bytes();
        if (BitConverter.IsLittleEndian)
        {
            // The values as stored, in one copy (the machine's byte order is the file's, rule 48): the list grown once.
            int count = packed.Length / 4, start = into.Count;
            CollectionsMarshal.SetCount(into, start + count);
            MemoryMarshal.Cast<byte, float>(packed[..(count * 4)]).CopyTo(CollectionsMarshal.AsSpan(into)[start..]);
            return;
        }

        for (int i = 0; i + 4 <= packed.Length; i += 4)
        {
            into.Add(BinaryPrimitives.ReadSingleLittleEndian(packed.Slice(i, 4)));
        }
    }
}

/// <summary>A constant tensor of an ONNX file (an initializer or a Constant node), as floats or integers.</summary>
internal sealed class OnnxTensor
{
    public string Name { get; init; } = "";

    public int[] Dims { get; init; } = [];

    public float[]? Floats { get; init; }

    public long[]? Longs { get; init; }

    public int Size => Dims.Aggregate(1, (a, b) => a * b);

    public float[] AsFloats() => Floats ?? Longs!.Select(v => (float)v).ToArray();

    public long[] AsLongs() => Longs ?? Floats!.Select(v => (long)v).ToArray();

    public static OnnxTensor Read(ReadOnlySpan<byte> data)
    {
        var r = new ProtoReader(data);
        var dims = new List<long>();
        var floats = new List<float>();
        var longs = new List<long>();
        var doubles = new List<double>();
        int type = 0;
        string name = "";
        ReadOnlySpan<byte> raw = default;
        bool hasRaw = false;
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1: r.Ints(wire, dims); break;
                case 2: type = (int)r.Varint(); break;
                case 4: r.Floats(wire, floats); break;
                case 5 or 7: r.Ints(wire, longs); break;                  // int32_data (also float16 bits) / int64_data
                case 8: name = r.String(); break;
                case 9: raw = r.Bytes(); hasRaw = true; break;                 // read in place: no copy of the weights
                case 10:
                    if (wire == 1)
                    {
                        doubles.Add(r.Fixed64());
                    }
                    else
                    {
                        var packed = r.Bytes();
                        for (int i = 0; i + 8 <= packed.Length; i += 8)
                        {
                            doubles.Add(BinaryPrimitives.ReadDoubleLittleEndian(packed.Slice(i, 8)));
                        }
                    }

                    break;
                case 14:
                    if (r.Varint() == 1)
                    {
                        throw new NotSupportedException($"Tensor '{name}' is stored in an external file; save the model with its weights embedded.");
                    }

                    break;
                default: r.Skip(wire); break;
            }
        }

        int[] shape = [.. dims.Select(d => checked((int)d))];
        int count = shape.Aggregate(1, (a, b) => a * b);
        return type switch
        {
            1 => new OnnxTensor { Name = name, Dims = shape, Floats = !hasRaw ? [.. floats] : Copy(raw, count, b => BinaryPrimitives.ReadSingleLittleEndian(b)) },
            11 => new OnnxTensor { Name = name, Dims = shape, Floats = !hasRaw ? [.. doubles.Select(d => (float)d)] : Decode(raw, 8, count, b => (float)BinaryPrimitives.ReadDoubleLittleEndian(b)) },
            10 => new OnnxTensor { Name = name, Dims = shape, Floats = !hasRaw ? [.. longs.Select(v => (float)BitConverter.UInt16BitsToHalf((ushort)v))] : Halves(raw, count) },
            16 => new OnnxTensor { Name = name, Dims = shape, Floats = !hasRaw ? [.. longs.Select(v => BitConverter.Int32BitsToSingle((int)v << 16))] : BFloat16s(raw, count) },
            7 => new OnnxTensor { Name = name, Dims = shape, Longs = !hasRaw ? [.. longs] : Copy(raw, count, b => BinaryPrimitives.ReadInt64LittleEndian(b)) },
            6 or 5 or 3 or 2 or 4 or 12 or 13 => new OnnxTensor { Name = name, Dims = shape, Longs = !hasRaw ? [.. longs] : DecodeInt(raw, type, count) },
            _ => throw new NotSupportedException($"Tensor '{name}' has ONNX data type {type}, which the importer does not read."),
        };
    }

    private delegate T Reader<T>(ReadOnlySpan<byte> bytes);

    private static T[] Decode<T>(ReadOnlySpan<byte> raw, int size, int count, Reader<T> read)
    {
        var values = new T[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = read(raw.Slice(i * size, size));
        }

        return values;
    }

    // Little-endian values as stored: one copy of the bytes where the machine's byte order is the file's (rule 48), else
    // read one by one.
    private static T[] Copy<T>(ReadOnlySpan<byte> raw, int count, Reader<T> read)
        where T : unmanaged
    {
        if (!BitConverter.IsLittleEndian)
        {
            return Decode(raw, Unsafe.SizeOf<T>(), count, read);
        }

        var values = new T[count];
        raw[..(count * Unsafe.SizeOf<T>())].CopyTo(MemoryMarshal.AsBytes(values.AsSpan()));
        return values;
    }

    private static float[] Halves(ReadOnlySpan<byte> raw, int count)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = (float)BinaryPrimitives.ReadHalfLittleEndian(raw.Slice(i * 2, 2));
        }

        return values;
    }

    private static float[] BFloat16s(ReadOnlySpan<byte> raw, int count)
    {
        var values = new float[count];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = BitConverter.Int32BitsToSingle(BinaryPrimitives.ReadUInt16LittleEndian(raw.Slice(i * 2, 2)) << 16);
        }

        return values;
    }

    private static long[] DecodeInt(ReadOnlySpan<byte> raw, int type, int count) => type switch
    {
        6 => Decode(raw, 4, count, b => (long)BinaryPrimitives.ReadInt32LittleEndian(b)),
        12 => Decode(raw, 4, count, b => (long)BinaryPrimitives.ReadUInt32LittleEndian(b)),
        13 => Decode(raw, 8, count, b => (long)BinaryPrimitives.ReadUInt64LittleEndian(b)),
        5 => Decode(raw, 2, count, b => (long)BinaryPrimitives.ReadInt16LittleEndian(b)),
        4 => Decode(raw, 2, count, b => (long)BinaryPrimitives.ReadUInt16LittleEndian(b)),
        3 => Decode(raw, 1, count, b => (long)(sbyte)b[0]),
        _ => Decode(raw, 1, count, b => (long)b[0]),
    };
}

/// <summary>An attribute of a node.</summary>
internal sealed class OnnxAttributeValue
{
    public long? Int { get; set; }

    public float? Float { get; set; }

    public string? String { get; set; }

    public long[]? Ints { get; set; }

    public float[]? Floats { get; set; }

    public OnnxTensor? Tensor { get; set; }

    public static (string Name, OnnxAttributeValue Value) Read(ReadOnlySpan<byte> data)
    {
        var r = new ProtoReader(data);
        var value = new OnnxAttributeValue();
        var ints = new List<long>();
        var floats = new List<float>();
        string name = "";
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1: name = r.String(); break;
                case 2: value.Float = r.Fixed32(); break;
                case 3: value.Int = (long)r.Varint(); break;
                case 4: value.String = r.String(); break;
                case 5: value.Tensor = OnnxTensor.Read(r.Bytes()); break;
                case 7: r.Floats(wire, floats); break;
                case 8: r.Ints(wire, ints); break;
                default: r.Skip(wire); break;
            }
        }

        if (ints.Count > 0)
        {
            value.Ints = [.. ints];
        }

        if (floats.Count > 0)
        {
            value.Floats = [.. floats];
        }

        return (name, value);
    }
}

/// <summary>A node of the graph.</summary>
internal sealed class OnnxNode
{
    public string Op { get; init; } = "";

    public string Name { get; init; } = "";

    public string Domain { get; init; } = "";

    public List<string> Inputs { get; } = [];

    public List<string> Outputs { get; } = [];

    public Dictionary<string, OnnxAttributeValue> Attributes { get; } = [];

    public long Int(string name, long fallback) => Attributes.TryGetValue(name, out var a) && a.Int is { } i ? i : fallback;

    public float Float(string name, float fallback) => Attributes.TryGetValue(name, out var a) && a.Float is { } f ? f : fallback;

    public long[]? Ints(string name) => Attributes.TryGetValue(name, out var a) ? a.Ints : null;

    public string? String(string name) => Attributes.TryGetValue(name, out var a) ? a.String : null;

    public override string ToString() => $"{Op} '{Name}'";

    public static OnnxNode Read(ReadOnlySpan<byte> data)
    {
        var r = new ProtoReader(data);
        string op = "", name = "", domain = "";
        var inputs = new List<string>();
        var outputs = new List<string>();
        var attributes = new List<(string, OnnxAttributeValue)>();
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1: inputs.Add(r.String()); break;
                case 2: outputs.Add(r.String()); break;
                case 3: name = r.String(); break;
                case 4: op = r.String(); break;
                case 5: attributes.Add(OnnxAttributeValue.Read(r.Bytes())); break;
                case 7: domain = r.String(); break;
                default: r.Skip(wire); break;
            }
        }

        var node = new OnnxNode { Op = op, Name = name, Domain = domain };
        node.Inputs.AddRange(inputs);
        node.Outputs.AddRange(outputs);
        foreach (var (key, value) in attributes)
        {
            node.Attributes[key] = value;
        }

        return node;
    }
}

/// <summary>A graph input or output: name and shape (-1 for dynamic or symbolic dimensions).</summary>
internal sealed record OnnxValueInfo(string Name, int[]? Shape)
{
    public static OnnxValueInfo Read(ReadOnlySpan<byte> data)
    {
        var r = new ProtoReader(data);
        string name = "";
        int[]? shape = null;
        while (r.Next(out int field, out int wire))
        {
            if (field == 1)
            {
                name = r.String();
            }
            else if (field == 2)
            {
                var type = new ProtoReader(r.Bytes());
                while (type.Next(out int tf, out int tw))
                {
                    if (tf != 1)
                    {
                        type.Skip(tw);
                        continue;
                    }

                    var tensor = new ProtoReader(type.Bytes());
                    while (tensor.Next(out int f, out int w))
                    {
                        if (f != 2)
                        {
                            tensor.Skip(w);
                            continue;
                        }

                        var dims = new List<int>();
                        var shapeReader = new ProtoReader(tensor.Bytes());
                        while (shapeReader.Next(out int sf, out int sw))
                        {
                            if (sf != 1)
                            {
                                shapeReader.Skip(sw);
                                continue;
                            }

                            int value = -1;
                            var dim = new ProtoReader(shapeReader.Bytes());
                            while (dim.Next(out int df, out int dw))
                            {
                                if (df == 1)
                                {
                                    value = (int)dim.Varint();
                                }
                                else
                                {
                                    dim.Skip(dw);
                                }
                            }

                            dims.Add(value);
                        }

                        shape = [.. dims];
                    }
                }
            }
            else
            {
                r.Skip(wire);
            }
        }

        return new OnnxValueInfo(name, shape);
    }
}

/// <summary>An ONNX model as read from a file: nodes, constants, inputs, outputs, opset and metadata.</summary>
internal sealed class OnnxModel
{
    public List<OnnxNode> Nodes { get; } = [];

    public Dictionary<string, OnnxTensor> Constants { get; } = [];

    public List<OnnxValueInfo> Inputs { get; } = [];

    public List<OnnxValueInfo> Outputs { get; } = [];

    public Dictionary<string, string> Metadata { get; } = [];

    public long Opset { get; private set; }

    public string GraphName { get; private set; } = "";

    public static OnnxModel Read(ReadOnlySpan<byte> data)
    {
        var model = new OnnxModel();
        var r = new ProtoReader(data);
        bool sawGraph = false;
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 7:
                    model.ReadGraph(r.Bytes());
                    sawGraph = true;
                    break;
                case 8:
                    var opset = new ProtoReader(r.Bytes());
                    string domain = "";
                    long version = 0;
                    while (opset.Next(out int f, out int w))
                    {
                        if (f == 1)
                        {
                            domain = opset.String();
                        }
                        else if (f == 2)
                        {
                            version = (long)opset.Varint();
                        }
                        else
                        {
                            opset.Skip(w);
                        }
                    }

                    if (domain is "" or "ai.onnx")
                    {
                        model.Opset = version;
                    }

                    break;
                case 14:
                    var entry = new ProtoReader(r.Bytes());
                    string key = "", value = "";
                    while (entry.Next(out int f, out int w))
                    {
                        if (f == 1)
                        {
                            key = entry.String();
                        }
                        else if (f == 2)
                        {
                            value = entry.String();
                        }
                        else
                        {
                            entry.Skip(w);
                        }
                    }

                    model.Metadata[key] = value;
                    break;
                default:
                    r.Skip(wire);
                    break;
            }
        }

        return sawGraph ? model : throw new InvalidDataException("Not an ONNX model (no graph).");
    }

    private void ReadGraph(ReadOnlySpan<byte> data)
    {
        var r = new ProtoReader(data);
        while (r.Next(out int field, out int wire))
        {
            switch (field)
            {
                case 1:
                    var node = OnnxNode.Read(r.Bytes());
                    if (node.Op == "Constant" && node.Attributes.TryGetValue("value", out var constant) && constant.Tensor is { } tensor)
                    {
                        Constants[node.Outputs[0]] = new OnnxTensor { Name = node.Outputs[0], Dims = tensor.Dims, Floats = tensor.Floats, Longs = tensor.Longs };
                    }
                    else if (node.Op == "Constant")
                    {
                        var a = node.Attributes.Values.First();
                        Constants[node.Outputs[0]] = a.Float is { } f ? new OnnxTensor { Name = node.Outputs[0], Floats = [f] }
                            : a.Int is { } i ? new OnnxTensor { Name = node.Outputs[0], Longs = [i] }
                            : a.Floats is { } fs ? new OnnxTensor { Name = node.Outputs[0], Dims = [fs.Length], Floats = fs }
                            : new OnnxTensor { Name = node.Outputs[0], Dims = [a.Ints!.Length], Longs = a.Ints };
                    }
                    else
                    {
                        Nodes.Add(node);
                    }

                    break;
                case 2: GraphName = r.String(); break;
                case 5:
                    var initializer = OnnxTensor.Read(r.Bytes());
                    Constants[initializer.Name] = initializer;
                    break;
                case 11: Inputs.Add(OnnxValueInfo.Read(r.Bytes())); break;
                case 12: Outputs.Add(OnnxValueInfo.Read(r.Bytes())); break;
                default: r.Skip(wire); break;
            }
        }

        Inputs.RemoveAll(i => Constants.ContainsKey(i.Name));           // old models list initializers as inputs too
    }
}
