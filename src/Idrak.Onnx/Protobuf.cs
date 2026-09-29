using System.Buffers.Binary;
using System.Text;

namespace Idrak.Onnx;

/// <summary>Writes Protocol Buffers messages (the encoding of .onnx files): varints, fixed32 floats, length-delimited fields.</summary>
internal sealed class ProtoWriter
{
    private readonly MemoryStream _buffer = new();

    public ProtoWriter Int(int field, long value)
    {
        Tag(field, 0);
        Varint((ulong)value);
        return this;
    }

    public ProtoWriter Float(int field, float value)
    {
        Tag(field, 5);
        Span<byte> bytes = stackalloc byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(bytes, value);
        _buffer.Write(bytes);
        return this;
    }

    public ProtoWriter String(int field, string value) => Bytes(field, Encoding.UTF8.GetBytes(value));

    public ProtoWriter Bytes(int field, ReadOnlySpan<byte> value)
    {
        Tag(field, 2);
        Varint((ulong)value.Length);
        _buffer.Write(value);
        return this;
    }

    public ProtoWriter Message(int field, ProtoWriter message) => Bytes(field, message._buffer.GetBuffer().AsSpan(0, (int)message._buffer.Length));

    public ProtoWriter PackedInts(int field, IEnumerable<long> values)
    {
        var packed = new ProtoWriter();
        foreach (long v in values)
        {
            packed.Varint((ulong)v);
        }

        return Message(field, packed);
    }

    public ProtoWriter PackedFloats(int field, IReadOnlyList<float> values)
    {
        var bytes = new byte[values.Count * 4];
        for (int i = 0; i < values.Count; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), values[i]);
        }

        return Bytes(field, bytes);
    }

    public byte[] ToArray() => _buffer.ToArray();

    private void Tag(int field, int wireType) => Varint((ulong)((field << 3) | wireType));

    private void Varint(ulong value)
    {
        while (value >= 0x80)
        {
            _buffer.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }

        _buffer.WriteByte((byte)value);
    }
}
