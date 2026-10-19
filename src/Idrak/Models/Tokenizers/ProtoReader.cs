// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;

namespace Idrak.Models;

/// <summary>The wire type of a Protocol Buffers field: the low three bits of its header.</summary>
internal enum ProtoWireType
{
    /// <summary>A base-128 varint (int32, int64, uint32, uint64, bool, enum).</summary>
    Varint = 0,

    /// <summary>Eight little-endian bytes (fixed64, sfixed64, double).</summary>
    Fixed64 = 1,

    /// <summary>A varint length, then that many bytes (string, bytes, an embedded message, a packed repeated field).</summary>
    LengthDelimited = 2,

    /// <summary>The start of a group (proto2's deprecated nested form), ended by <see cref="EndGroup"/> with the same field number.</summary>
    StartGroup = 3,

    /// <summary>The end of a group.</summary>
    EndGroup = 4,

    /// <summary>Four little-endian bytes (fixed32, sfixed32, float).</summary>
    Fixed32 = 5,
}

/// <summary>
/// Reads Protocol Buffers wire data from a span, field by field and without copying: <see cref="Next"/> reads a field's
/// header, then the matching method reads its value (or <see cref="Skip"/> passes over a field the caller does not know).
/// Every error is an <see cref="InvalidDataException"/> that names the byte where the bad item starts, counted from the
/// start of the outermost data (a reader made by <see cref="Message"/> keeps counting from there), so a message about a
/// broken file points into the file.
/// </summary>
/// <remarks>
/// Byte order is explicit (<see cref="BinaryPrimitives"/>), so the reader gives the same values on every platform. A varint
/// is at most ten bytes and its tenth byte may only carry the 64th bit; an int32 field is the low 32 bits of its varint,
/// as protobuf writes a negative int32 sign-extended to ten bytes.
/// </remarks>
internal ref struct ProtoReader
{
    private const int MaxGroupDepth = 64;
    private static readonly UTF8Encoding StrictUtf8 = new(encoderShouldEmitUTF8Identifier: false, throwOnInvalidBytes: true);

    private readonly ReadOnlySpan<byte> _data;
    private readonly int _origin;
    private int _position;

    /// <summary>A reader over <paramref name="data"/>, whose first byte is byte <paramref name="origin"/> of the whole input.</summary>
    public ProtoReader(ReadOnlySpan<byte> data, int origin = 0)
    {
        _data = data;
        _origin = origin;
        _position = 0;
    }

    /// <summary>The offset of the next byte to read, counted from the start of the outermost data.</summary>
    public readonly int Offset => _origin + _position;

    /// <summary>True when every byte has been read.</summary>
    public readonly bool End => _position >= _data.Length;

    /// <summary>
    /// Reads the next field's header. False at the end of the data. A field number of 0 or above 2^29 - 1, or a wire
    /// type of 6 or 7, is refused.
    /// </summary>
    public bool Next(out int field, out ProtoWireType wireType)
    {
        if (_position >= _data.Length)
        {
            field = 0;
            wireType = default;
            return false;
        }

        int start = Offset;
        ulong tag = Varint();
        int type = (int)(tag & 7);
        ulong number = tag >> 3;
        if (number == 0 || number > 0x1FFF_FFFF)
        {
            throw Error($"field number {number} (valid: 1 to 536870911)", start);
        }

        if (type > 5)
        {
            throw Error($"wire type {type} (valid: 0 to 5) in field {number}", start);
        }

        field = (int)number;
        wireType = (ProtoWireType)type;
        return true;
    }

    /// <summary>Reads a varint (at most ten bytes, at most 64 bits).</summary>
    public ulong Varint()
    {
        int start = _position;
        ulong result = 0;
        for (int i = 0; ; i++)
        {
            if (_position >= _data.Length)
            {
                throw Error($"truncated varint (the data ends after {i} of its bytes)", _origin + start);
            }

            byte b = _data[_position++];
            if (i == 9)
            {
                if (b > 1)
                {
                    throw Error(b >= 0x80 ? "varint longer than ten bytes" : "varint above 64 bits", _origin + start);
                }

                return result | ((ulong)b << 63);
            }

            result |= (ulong)(b & 0x7F) << (7 * i);
            if (b < 0x80)
            {
                return result;
            }
        }
    }

    /// <summary>An int32 or enum field: the low 32 bits of its varint.</summary>
    public int Int32() => unchecked((int)Varint());

    /// <summary>An int64 field.</summary>
    public long Int64() => unchecked((long)Varint());

    /// <summary>A bool field: any non-zero varint is true.</summary>
    public bool Bool() => Varint() != 0;

    /// <summary>Reads four little-endian bytes.</summary>
    public uint Fixed32()
    {
        var bytes = Take(4, "fixed32");
        return BinaryPrimitives.ReadUInt32LittleEndian(bytes);
    }

    /// <summary>Reads eight little-endian bytes.</summary>
    public ulong Fixed64()
    {
        var bytes = Take(8, "fixed64");
        return BinaryPrimitives.ReadUInt64LittleEndian(bytes);
    }

    /// <summary>A float field (fixed32), bit for bit.</summary>
    public float Float() => BitConverter.UInt32BitsToSingle(Fixed32());

    /// <summary>A double field (fixed64), bit for bit.</summary>
    public double Double() => BitConverter.UInt64BitsToDouble(Fixed64());

    /// <summary>Reads a length-delimited field's bytes, a slice of the input (nothing is copied).</summary>
    public ReadOnlySpan<byte> LengthDelimited()
    {
        int start = _position;
        ulong length = Varint();
        int left = _data.Length - _position;
        if (length > (ulong)left)
        {
            throw Error($"a length-delimited field of {length} bytes runs past the end of its data (byte {_origin + _data.Length}, {left} bytes left)", _origin + start);
        }

        var bytes = _data.Slice(_position, (int)length);
        _position += (int)length;
        return bytes;
    }

    /// <summary>An embedded message: a reader over its bytes that keeps counting offsets from the outermost data.</summary>
    public ProtoReader Message()
    {
        var bytes = LengthDelimited();
        return new ProtoReader(bytes, _origin + _position - bytes.Length);
    }

    /// <summary>A string field: strict UTF-8 (an invalid sequence is an error, not a replacement character).</summary>
    public string String()
    {
        int start = Offset;
        var bytes = LengthDelimited();
        try
        {
            return StrictUtf8.GetString(bytes);
        }
        catch (DecoderFallbackException)
        {
            throw Error("invalid UTF-8 in a string field", start);
        }
    }

    /// <summary>
    /// Passes over a field whose header <see cref="Next"/> just read: any wire type, a group up to its matching end
    /// (nested groups too). An end-group with nothing open is refused.
    /// </summary>
    public void Skip(int field, ProtoWireType wireType) => SkipField(field, wireType, 0);

    private void SkipField(int field, ProtoWireType wireType, int depth)
    {
        switch (wireType)
        {
            case ProtoWireType.Varint:
                Varint();
                break;
            case ProtoWireType.Fixed64:
                Take(8, "fixed64");
                break;
            case ProtoWireType.LengthDelimited:
                LengthDelimited();
                break;
            case ProtoWireType.Fixed32:
                Take(4, "fixed32");
                break;
            case ProtoWireType.StartGroup:
                SkipGroup(field, depth);
                break;
            default:
                throw Error($"an end-group (field {field}) with no group open", Offset - TagSize(field, wireType));
        }
    }

    private void SkipGroup(int field, int depth)
    {
        int start = Offset - TagSize(field, ProtoWireType.StartGroup);
        if (depth >= MaxGroupDepth)
        {
            throw Error($"groups nested deeper than {MaxGroupDepth}", start);
        }

        while (Next(out int inner, out var type))
        {
            if (type == ProtoWireType.EndGroup)
            {
                if (inner != field)
                {
                    throw Error($"the group of field {field} is ended by field {inner}", Offset - TagSize(inner, type));
                }

                return;
            }

            SkipField(inner, type, depth + 1);
        }

        throw Error($"the group of field {field} has no end", start);
    }

    private ReadOnlySpan<byte> Take(int count, string what)
    {
        if (_data.Length - _position < count)
        {
            throw Error($"truncated {what} (the data ends after {_data.Length - _position} of its {count} bytes)", Offset);
        }

        var bytes = _data.Slice(_position, count);
        _position += count;
        return bytes;
    }

    // The size of a field header's varint (Next has consumed it; errors point at its first byte).
    private static int TagSize(int field, ProtoWireType wireType)
    {
        ulong tag = ((ulong)(uint)field << 3) | (uint)wireType;
        int size = 1;
        while (tag >= 0x80)
        {
            tag >>= 7;
            size++;
        }

        return size;
    }

    private static InvalidDataException Error(string what, int offset) =>
        new($"Protocol Buffers data: {what} at byte {offset}.");
}
