// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;

namespace Idrak.Gpu.Hip;

// The code object's metadata note (NT_AMDGPU_METADATA): the kernels as the HIP runtime reads them, in MessagePack written
// by hand, the keys of each map in LLVM's order (sorted), each number and string in its shortest form as LLVM writes them.
internal static partial class GcnCodeObject
{
    /// <summary>
    /// The metadata: <c>amdhsa.kernels</c> (each kernel's arguments, segment sizes, workgroup sizes, registers and their
    /// spills, name, descriptor symbol and wave size), <c>amdhsa.target</c> (the triple and target id) and <c>amdhsa.version</c> (1.1 for format 4, 1.2 for 5).
    /// Written twice: once to measure (an empty writer), once into the note.
    /// </summary>
    private static void WriteMetadata(ref MessagePack pack, ReadOnlySpan<GcnKernelCode> kernels, ReadOnlySpan<int> sgprs, string targetId, int format)
    {
        pack.Map(3);
        pack.String("amdhsa.kernels"u8);
        pack.Array(kernels.Length);
        for (int i = 0; i < kernels.Length; i++)
        {
            var kernel = kernels[i];
            pack.Map(12 + (kernel.Arguments.Count > 0 ? 1 : 0) + (kernel.RequiredWorkgroupSize is null ? 0 : 1));
            if (kernel.Arguments.Count > 0)
            {
                pack.String(".args"u8);
                pack.Array(kernel.Arguments.Count);
                foreach (var argument in kernel.Arguments)
                {
                    pack.Map(3 + (argument.AddressSpace == GcnAddressSpace.None ? 0 : 1) + (argument.Name is null ? 0 : 1));
                    if (argument.AddressSpace != GcnAddressSpace.None)
                    {
                        pack.String(".address_space"u8);
                        pack.String(argument.AddressSpace == GcnAddressSpace.Global ? "global"u8 : "constant"u8);
                    }

                    if (argument.Name is not null)
                    {
                        pack.String(".name"u8);
                        pack.String(argument.Name, ""u8);
                    }

                    pack.String(".offset"u8);
                    pack.UInt((ulong)argument.Offset);
                    pack.String(".size"u8);
                    pack.UInt((ulong)argument.Size);
                    pack.String(".value_kind"u8);
                    pack.String(argument.Kind == GcnArgumentKind.GlobalBuffer ? "global_buffer"u8 : "by_value"u8);
                }
            }

            pack.String(".group_segment_fixed_size"u8);
            pack.UInt((ulong)kernel.LdsBytes);
            pack.String(".kernarg_segment_align"u8);
            pack.UInt((ulong)KernargAlign(kernel));
            pack.String(".kernarg_segment_size"u8);
            pack.UInt((ulong)KernargBytes(kernel));
            pack.String(".max_flat_workgroup_size"u8);
            pack.UInt((ulong)kernel.MaxFlatWorkgroupSize);
            pack.String(".name"u8);
            pack.String(kernel.Name, ""u8);
            pack.String(".private_segment_fixed_size"u8);
            pack.UInt((ulong)kernel.PrivateBytes);
            if (kernel.RequiredWorkgroupSize is { } shape)
            {
                pack.String(".reqd_workgroup_size"u8);
                pack.Array(3);
                pack.UInt((ulong)shape.X);
                pack.UInt((ulong)shape.Y);
                pack.UInt((ulong)shape.Z);
            }

            pack.String(".sgpr_count"u8);
            pack.UInt((ulong)sgprs[i]);
            pack.String(".sgpr_spill_count"u8);
            pack.UInt((ulong)kernel.SgprSpills);
            pack.String(".symbol"u8);
            pack.String(kernel.Name, ".kd"u8);
            pack.String(".vgpr_count"u8);
            pack.UInt((ulong)kernel.Vgprs);
            pack.String(".vgpr_spill_count"u8);
            pack.UInt((ulong)kernel.VgprSpills);
            pack.String(".wavefront_size"u8);
            pack.UInt((ulong)kernel.WaveSize);
        }

        pack.String("amdhsa.target"u8);
        pack.String(targetId, ""u8);
        pack.String("amdhsa.version"u8);
        pack.Array(2);
        pack.UInt(1);
        pack.UInt(format == 4 ? 1u : 2u);
    }

    /// <summary>
    /// A MessagePack writer into a span, or, given an empty one, a counter of the bytes it would write. Maps, arrays,
    /// strings and unsigned integers, each in its shortest form (fixmap, fixarray, fixstr then str8, positive fixint then
    /// uint8 to uint64), big-endian as the format is.
    /// </summary>
    private ref struct MessagePack(Span<byte> destination)
    {
        private readonly Span<byte> _destination = destination;
        private readonly bool _measure = destination.IsEmpty;

        /// <summary>The bytes written (or counted) so far.</summary>
        public int Length { get; private set; }

        public void Map(int entries) => Header(entries, 0x80, 15, 0xDE, 0xDF);

        public void Array(int items) => Header(items, 0x90, 15, 0xDC, 0xDD);

        public void String(ReadOnlySpan<byte> utf8)
        {
            StringHeader(utf8.Length);
            Bytes(utf8);
        }

        /// <summary>The string <paramref name="text"/> followed by <paramref name="suffix"/>, as one string.</summary>
        public void String(string text, ReadOnlySpan<byte> suffix)
        {
            int bytes = Encoding.UTF8.GetByteCount(text);
            StringHeader(bytes + suffix.Length);
            if (!_measure)
            {
                Encoding.UTF8.GetBytes(text, _destination[Length..]);
            }

            Length += bytes;
            Bytes(suffix);
        }

        public void UInt(ulong value)
        {
            if (value <= 0x7F)
            {
                Byte((byte)value);
            }
            else if (value <= byte.MaxValue)
            {
                Byte(0xCC);
                Byte((byte)value);
            }
            else if (value <= ushort.MaxValue)
            {
                Byte(0xCD);
                Big16((ushort)value);
            }
            else if (value <= uint.MaxValue)
            {
                Byte(0xCE);
                Big32((uint)value);
            }
            else
            {
                Byte(0xCF);
                if (!_measure)
                {
                    BinaryPrimitives.WriteUInt64BigEndian(_destination[Length..], value);
                }

                Length += 8;
            }
        }

        private void StringHeader(int bytes)
        {
            if (bytes <= 31)
            {
                Byte((byte)(0xA0 | bytes));
            }
            else if (bytes <= byte.MaxValue)
            {
                Byte(0xD9);
                Byte((byte)bytes);
            }
            else if (bytes <= ushort.MaxValue)
            {
                Byte(0xDA);
                Big16((ushort)bytes);
            }
            else
            {
                Byte(0xDB);
                Big32((uint)bytes);
            }
        }

        private void Header(int count, byte fixedForm, int fixedMax, byte form16, byte form32)
        {
            if (count <= fixedMax)
            {
                Byte((byte)(fixedForm | count));
            }
            else if (count <= ushort.MaxValue)
            {
                Byte(form16);
                Big16((ushort)count);
            }
            else
            {
                Byte(form32);
                Big32((uint)count);
            }
        }

        private void Byte(byte value)
        {
            if (!_measure)
            {
                _destination[Length] = value;
            }

            Length++;
        }

        private void Big16(ushort value)
        {
            if (!_measure)
            {
                BinaryPrimitives.WriteUInt16BigEndian(_destination[Length..], value);
            }

            Length += 2;
        }

        private void Big32(uint value)
        {
            if (!_measure)
            {
                BinaryPrimitives.WriteUInt32BigEndian(_destination[Length..], value);
            }

            Length += 4;
        }

        private void Bytes(ReadOnlySpan<byte> bytes)
        {
            if (!_measure)
            {
                bytes.CopyTo(_destination[Length..]);
            }

            Length += bytes.Length;
        }
    }
}
