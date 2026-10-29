// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;

namespace Idrak.Gpu.Hip;

/// <summary>
/// One kernel as a code object's metadata declares it: its name, its arguments (offset, size, kind), the bytes of
/// kernel arguments a launch passes (<c>.kernarg_segment_size</c>) and the block it runs in (<c>.reqd_workgroup_size</c>'s
/// x, else <c>.max_flat_workgroup_size</c>). What a launch is checked against, as CUDA's <c>_signatures</c> are.
/// </summary>
internal sealed record GcnKernelLayout(string Name, IReadOnlyList<GcnArgument> Arguments, int ArgumentBytes, int Block);

// Reading a code object's metadata back (a code object Idrak wrote, from the assembler or from the kernel cache): the
// section headers, the note of type NT_AMDGPU_METADATA owned by "AMDGPU", and its MessagePack's amdhsa.kernels, each
// kernel's name, arguments and kernel argument bytes. Anything malformed or out of bounds is refused with
// InvalidDataException, never read past.
internal static partial class GcnCodeObject
{
    /// <summary>The kernels <paramref name="image"/>'s metadata declares, in its order.</summary>
    /// <exception cref="InvalidDataException">Not a 64-bit little-endian ELF, no metadata note, or malformed MessagePack.</exception>
    internal static GcnKernelLayout[] ReadKernels(ReadOnlySpan<byte> image)
    {
        if (image.Length < HeaderBytes || !image[..4].SequenceEqual("\u007fELF"u8) || image[4] != ElfClass64 || image[5] != ElfData2Lsb)
        {
            throw new InvalidDataException("not a 64-bit little-endian ELF code object");
        }

        ulong sectionOffset = BinaryPrimitives.ReadUInt64LittleEndian(image[0x28..]);
        int entrySize = BinaryPrimitives.ReadUInt16LittleEndian(image[0x3A..]);
        int count = BinaryPrimitives.ReadUInt16LittleEndian(image[0x3C..]);
        if (entrySize < SectionHeaderBytes || sectionOffset > (ulong)image.Length || (ulong)count * (ulong)entrySize > (ulong)image.Length - sectionOffset)
        {
            throw new InvalidDataException("the section headers lie outside the code object");
        }

        for (int i = 0; i < count; i++)
        {
            var header = image.Slice((int)sectionOffset + i * entrySize, SectionHeaderBytes);
            if (BinaryPrimitives.ReadUInt32LittleEndian(header[4..]) != ShtNote)
            {
                continue;
            }

            var notes = Within(image, BinaryPrimitives.ReadUInt64LittleEndian(header[0x18..]), BinaryPrimitives.ReadUInt64LittleEndian(header[0x20..]));
            while (notes.Length >= 12)
            {
                int nameBytes = (int)Math.Min(BinaryPrimitives.ReadUInt32LittleEndian(notes), int.MaxValue / 2);
                int descBytes = (int)Math.Min(BinaryPrimitives.ReadUInt32LittleEndian(notes[4..]), int.MaxValue / 2);
                uint type = BinaryPrimitives.ReadUInt32LittleEndian(notes[8..]);
                int nameEnd = 12 + (nameBytes + 3) / 4 * 4, descEnd = nameEnd + (descBytes + 3) / 4 * 4;
                if (descEnd > notes.Length || nameEnd - 12 < nameBytes)
                {
                    throw new InvalidDataException("a note runs past its section");
                }

                if (type == NtAmdgpuMetadata && notes.Slice(12, nameBytes).SequenceEqual(MetadataOwner()))
                {
                    return ReadMetadata(notes.Slice(nameEnd, descBytes));
                }

                notes = notes[descEnd..];
            }
        }

        throw new InvalidDataException("the code object has no AMDGPU metadata note");
    }

    // The note's owner as written: "AMDGPU" and its terminator.
    private static ReadOnlySpan<byte> MetadataOwner() => "AMDGPU\0"u8;

    private static ReadOnlySpan<byte> Within(ReadOnlySpan<byte> image, ulong offset, ulong size) =>
        offset <= (ulong)image.Length && size <= (ulong)image.Length - offset ? image.Slice((int)offset, (int)size)
            : throw new InvalidDataException("a section lies outside the code object");

    // The metadata map: amdhsa.kernels read, every other key skipped.
    private static GcnKernelLayout[] ReadMetadata(ReadOnlySpan<byte> metadata)
    {
        var reader = new MessagePackReader(metadata);
        GcnKernelLayout[]? kernels = null;
        for (int entries = reader.Map(); entries > 0; entries--)
        {
            if (reader.String().SequenceEqual("amdhsa.kernels"u8))
            {
                kernels = new GcnKernelLayout[reader.Array()];
                for (int k = 0; k < kernels.Length; k++)
                {
                    kernels[k] = ReadKernel(ref reader);
                }
            }
            else
            {
                reader.Skip();
            }
        }

        return kernels ?? throw new InvalidDataException("the metadata lists no amdhsa.kernels");
    }

    private static GcnKernelLayout ReadKernel(ref MessagePackReader reader)
    {
        string? name = null;
        int argumentBytes = -1, required = 0, most = 0;
        var arguments = new List<GcnArgument>();
        for (int entries = reader.Map(); entries > 0; entries--)
        {
            var key = reader.String();
            if (key.SequenceEqual(".name"u8))
            {
                name = Encoding.UTF8.GetString(reader.String());
            }
            else if (key.SequenceEqual(".kernarg_segment_size"u8))
            {
                argumentBytes = checked((int)reader.UInt());
            }
            else if (key.SequenceEqual(".reqd_workgroup_size"u8))
            {
                int dimensions = reader.Array();
                for (int d = 0; d < dimensions; d++)
                {
                    int size = checked((int)reader.UInt());
                    required = d == 0 ? size : required;
                }
            }
            else if (key.SequenceEqual(".max_flat_workgroup_size"u8))
            {
                most = checked((int)reader.UInt());
            }
            else if (key.SequenceEqual(".args"u8))
            {
                for (int a = reader.Array(); a > 0; a--)
                {
                    arguments.Add(ReadArgument(ref reader));
                }
            }
            else
            {
                reader.Skip();
            }
        }

        if (name is null || argumentBytes < 0)
        {
            throw new InvalidDataException("a kernel's metadata lacks its .name or .kernarg_segment_size");
        }

        foreach (var argument in arguments)
        {
            if (argument.Offset < 0 || argument.Size <= 0 || argument.Offset + argument.Size > argumentBytes)
            {
                throw new InvalidDataException($"{name}: argument {argument.Name} at {argument.Offset} ({argument.Size} bytes) lies outside its {argumentBytes} bytes of kernel arguments");
            }
        }

        return new GcnKernelLayout(name, arguments, argumentBytes, required > 0 ? required : most);
    }

    private static GcnArgument ReadArgument(ref MessagePackReader reader)
    {
        int offset = -1, size = -1;
        string? name = null;
        var kind = GcnArgumentKind.ByValue;
        var space = GcnAddressSpace.None;
        for (int entries = reader.Map(); entries > 0; entries--)
        {
            var key = reader.String();
            if (key.SequenceEqual(".offset"u8))
            {
                offset = checked((int)reader.UInt());
            }
            else if (key.SequenceEqual(".size"u8))
            {
                size = checked((int)reader.UInt());
            }
            else if (key.SequenceEqual(".name"u8))
            {
                name = Encoding.UTF8.GetString(reader.String());
            }
            else if (key.SequenceEqual(".value_kind"u8))
            {
                kind = reader.String().SequenceEqual("global_buffer"u8) ? GcnArgumentKind.GlobalBuffer : GcnArgumentKind.ByValue;
            }
            else if (key.SequenceEqual(".address_space"u8))
            {
                var value = reader.String();
                space = value.SequenceEqual("global"u8) ? GcnAddressSpace.Global : value.SequenceEqual("constant"u8) ? GcnAddressSpace.Constant : GcnAddressSpace.None;
            }
            else
            {
                reader.Skip();
            }
        }

        return offset >= 0 && size > 0 ? new GcnArgument(offset, size, kind, space, name)
            : throw new InvalidDataException("a kernel argument lacks its .offset or .size");
    }

    /// <summary>A MessagePack reader over a span: maps, arrays, strings and unsigned integers read; any value skipped.</summary>
    private ref struct MessagePackReader(ReadOnlySpan<byte> data)
    {
        private readonly ReadOnlySpan<byte> _data = data;
        private int _at;

        public int Map() => Header(0x80, 0xDE, 0xDF, "a map");

        public int Array() => Header(0x90, 0xDC, 0xDD, "an array");

        public ReadOnlySpan<byte> String()
        {
            byte b = Byte();
            int length = b switch
            {
                >= 0xA0 and <= 0xBF => b & 0x1F,
                0xD9 => Byte(),
                0xDA => Big(2),
                0xDB => Big(4),
                _ => throw new InvalidDataException($"MessagePack: a string expected at {_at - 1}, 0x{b:X2} found"),
            };
            return Take(length);
        }

        public ulong UInt()
        {
            byte b = Byte();
            return b switch
            {
                <= 0x7F => b,
                0xCC => Byte(),
                0xCD => (ulong)Big(2),
                0xCE => (ulong)(uint)Big(4),
                0xCF => BinaryPrimitives.ReadUInt64BigEndian(Take(8)),
                _ => throw new InvalidDataException($"MessagePack: an unsigned integer expected at {_at - 1}, 0x{b:X2} found"),
            };
        }

        /// <summary>Skips one value of any type (nested maps and arrays whole).</summary>
        public void Skip()
        {
            for (long pending = 1; pending > 0; pending--)
            {
                byte b = Byte();
                switch (b)
                {
                    case <= 0x7F or >= 0xE0 or 0xC0 or 0xC2 or 0xC3:
                        break;
                    case >= 0x80 and <= 0x8F:
                        pending += 2L * (b & 0x0F);
                        break;
                    case >= 0x90 and <= 0x9F:
                        pending += b & 0x0F;
                        break;
                    case >= 0xA0 and <= 0xBF:
                        Take(b & 0x1F);
                        break;
                    case 0xC4 or 0xD9:
                        Take(Byte());
                        break;
                    case 0xC5 or 0xDA:
                        Take(Big(2));
                        break;
                    case 0xC6 or 0xDB:
                        Take(Big(4));
                        break;
                    case 0xCC or 0xD0:
                        Take(1);
                        break;
                    case 0xCD or 0xD1:
                        Take(2);
                        break;
                    case 0xCA or 0xCE or 0xD2:
                        Take(4);
                        break;
                    case 0xCB or 0xCF or 0xD3:
                        Take(8);
                        break;
                    case 0xDC:
                        pending += Big(2);
                        break;
                    case 0xDD:
                        pending += Big(4);
                        break;
                    case 0xDE:
                        pending += 2L * Big(2);
                        break;
                    case 0xDF:
                        pending += 2L * Big(4);
                        break;
                    default:
                        throw new InvalidDataException($"MessagePack: type 0x{b:X2} at {_at - 1} is not read here");
                }
            }
        }

        private int Header(byte fixedForm, byte form16, byte form32, string what)
        {
            byte b = Byte();
            if ((b & 0xF0) == fixedForm)
            {
                return b & 0x0F;
            }

            return b == form16 ? Big(2) : b == form32 ? Big(4) : throw new InvalidDataException($"MessagePack: {what} expected at {_at - 1}, 0x{b:X2} found");
        }

        private byte Byte() => _at < _data.Length ? _data[_at++] : throw new InvalidDataException("MessagePack: the metadata ends early");

        private int Big(int bytes)
        {
            var span = Take(bytes);
            uint value = bytes == 2 ? BinaryPrimitives.ReadUInt16BigEndian(span) : BinaryPrimitives.ReadUInt32BigEndian(span);
            return value <= int.MaxValue ? (int)value : throw new InvalidDataException("MessagePack: a length past 2^31");
        }

        private ReadOnlySpan<byte> Take(int bytes)
        {
            if (bytes < 0 || bytes > _data.Length - _at)
            {
                throw new InvalidDataException("MessagePack: the metadata ends early");
            }

            var span = _data.Slice(_at, bytes);
            _at += bytes;
            return span;
        }
    }
}
