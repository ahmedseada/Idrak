// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Text;

namespace Idrak.Tests.Gcn;

/// <summary>
/// A kernel's descriptor (the 64-byte amdhsa kernel descriptor of gfx9), as the interpreter reads it: what the dispatch
/// sets up (register counts, the user and system SGPRs it enables, the work-item ids, LDS and the kernel arguments).
/// </summary>
internal sealed record GcnKernelDescriptor
{
    /// <summary>Bytes of LDS a workgroup has (group_segment_fixed_size).</summary>
    public int LdsBytes { get; init; }

    /// <summary>Bytes of scratch a work-item has (private_segment_fixed_size); the interpreter models none.</summary>
    public int PrivateBytes { get; init; }

    /// <summary>Bytes of the kernel arguments (kernarg_size).</summary>
    public int KernargBytes { get; init; }

    public uint Rsrc1 { get; init; }

    public uint Rsrc2 { get; init; }

    public ushort Properties { get; init; }

    /// <summary>VGPRs a wave has: the granulated count of RSRC1 in blocks of 4 (wave64).</summary>
    public int Vgprs => (int)((Rsrc1 & 0x3F) + 1) * 4;

    /// <summary>SGPRs a wave is allocated: the granulated count of RSRC1 in blocks of 8 (VCC and the xnack mask among them).</summary>
    public int SgprBlocks => (int)(((Rsrc1 >> 6) & 0xF) + 1) * 8;

    public int FloatDenormMode32 => (int)(Rsrc1 >> 16) & 3;

    public bool Dx10Clamp => (Rsrc1 & (1u << 21)) != 0;

    public bool IeeeMode => (Rsrc1 & (1u << 23)) != 0;

    public bool PrivateSegmentWaveOffset => (Rsrc2 & 1) != 0;

    public int UserSgprCount => (int)(Rsrc2 >> 1) & 0x1F;

    public bool WorkgroupIdX => (Rsrc2 & (1u << 7)) != 0;

    public bool WorkgroupIdY => (Rsrc2 & (1u << 8)) != 0;

    public bool WorkgroupIdZ => (Rsrc2 & (1u << 9)) != 0;

    public bool WorkgroupInfo => (Rsrc2 & (1u << 10)) != 0;

    /// <summary>The work-item ids set: 0 x (v0), 1 x and y (v0, v1), 2 x, y and z (v0, v1, v2).</summary>
    public int WorkitemIds => (int)(Rsrc2 >> 11) & 3;

    public bool PrivateSegmentBuffer => (Properties & 1) != 0;

    public bool DispatchPtr => (Properties & 2) != 0;

    public bool QueuePtr => (Properties & 4) != 0;

    public bool KernargSegmentPtr => (Properties & 8) != 0;

    public bool DispatchId => (Properties & 16) != 0;

    public bool FlatScratchInit => (Properties & 32) != 0;

    public bool PrivateSegmentSize => (Properties & 64) != 0;

    /// <summary>A descriptor from its 64 bytes.</summary>
    public static GcnKernelDescriptor Read(ReadOnlySpan<byte> bytes) => new()
    {
        LdsBytes = (int)BinaryPrimitives.ReadUInt32LittleEndian(bytes),
        PrivateBytes = (int)BinaryPrimitives.ReadUInt32LittleEndian(bytes[4..]),
        KernargBytes = (int)BinaryPrimitives.ReadUInt32LittleEndian(bytes[8..]),
        Rsrc1 = BinaryPrimitives.ReadUInt32LittleEndian(bytes[48..]),
        Rsrc2 = BinaryPrimitives.ReadUInt32LittleEndian(bytes[52..]),
        Properties = BinaryPrimitives.ReadUInt16LittleEndian(bytes[56..]),
    };

    /// <summary>
    /// A descriptor for hand-written code: <paramref name="vgprs"/> and <paramref name="sgprs"/> rounded up to their
    /// blocks (with VCC's and the xnack mask's four), the kernel arguments' address in s[0:1], the workgroup ids asked
    /// for after it, and work-item ids up to <paramref name="workitemIds"/> (0 x, 1 x and y, 2 x, y and z).
    /// </summary>
    public static GcnKernelDescriptor For(int vgprs, int sgprs, int ldsBytes = 0, int kernargBytes = 0, int workgroupIds = 1, int workitemIds = 0)
    {
        int vgprBlocks = Math.Max(1, (vgprs + 3) / 4) - 1;
        int sgprBlocks = Math.Max(1, (sgprs + 4 + 7) / 8) - 1;
        uint rsrc1 = (uint)vgprBlocks | (uint)sgprBlocks << 6 | 3u << 16 | 3u << 18 | 1u << 21 | 1u << 23;
        uint rsrc2 = 2u << 1 | (uint)(workgroupIds & 7) << 7 | (uint)workitemIds << 11;
        return new GcnKernelDescriptor { LdsBytes = ldsBytes, KernargBytes = kernargBytes, Rsrc1 = rsrc1, Rsrc2 = rsrc2, Properties = 8 };
    }
}

/// <summary>A kernel ready to run: its processor, its code, the offset of its entry in the code, and its descriptor.</summary>
internal sealed record GcnKernelImage(string Name, GcnCpu Cpu, byte[] Code, int Entry, GcnKernelDescriptor Descriptor, bool Xnack)
{
    /// <summary>
    /// The SGPRs the code may use as s0..: the allocated blocks less VCC's two and, where xnack may be on, the xnack
    /// mask's two (gfx9 keeps them at the top of the wave's allocation).
    /// </summary>
    public int UsableSgprs => Descriptor.SgprBlocks - (Xnack ? 4 : 2);
}

/// <summary>Reads a gfx9 code object (an ELF shared object for EM_AMDGPU) by itself: its kernels' code and descriptors.</summary>
internal static class GcnCodeObjectReader
{
    private readonly record struct Section(string Name, uint Type, ulong Address, ulong Offset, ulong Size, uint Link);

    /// <summary>The processor of a gfx9 code object's ELF flags.</summary>
    /// <exception cref="InvalidDataException">The flags name a processor the interpreter does not decode for.</exception>
    public static GcnCpu CpuOf(uint flags) => (flags & 0xFF) switch
    {
        0x2C => GcnCpu.Gfx900,
        0x2F => GcnCpu.Gfx906,
        0x30 => GcnCpu.Gfx908,
        0x32 => GcnCpu.Gfx90c,
        var m => throw new InvalidDataException($"ELF flags 0x{flags:X}: machine 0x{m:X2} is not a processor the interpreter decodes for (gfx900, gfx906, gfx908, gfx90c)"),
    };

    /// <summary>The kernel <paramref name="name"/> of a code object.</summary>
    /// <exception cref="InvalidDataException">The bytes are not a gfx9 code object, or hold no such kernel.</exception>
    public static GcnKernelImage Read(byte[] elf, string name)
    {
        var span = elf.AsSpan();
        if (elf.Length < 64 || !span[..4].SequenceEqual((ReadOnlySpan<byte>)[0x7F, (byte)'E', (byte)'L', (byte)'F']) || elf[4] != 2 || elf[5] != 1)
        {
            throw new InvalidDataException("not a 64-bit little-endian ELF file");
        }

        if (BinaryPrimitives.ReadUInt16LittleEndian(span[18..]) != 224)
        {
            throw new InvalidDataException($"ELF machine {BinaryPrimitives.ReadUInt16LittleEndian(span[18..])} is not EM_AMDGPU (224)");
        }

        uint flags = BinaryPrimitives.ReadUInt32LittleEndian(span[48..]);
        var cpu = CpuOf(flags);
        bool xnack = (flags & 0x300) != 0x200;
        var sections = Sections(span);
        var dynsym = sections.FirstOrDefault(s => s.Type == 11);
        if (dynsym.Name is null)
        {
            throw new InvalidDataException("the code object has no .dynsym");
        }

        var strings = sections[(int)dynsym.Link];
        ulong? kd = null;
        for (ulong at = dynsym.Offset; at + 24 <= dynsym.Offset + dynsym.Size; at += 24)
        {
            var entry = span[(int)at..];
            string symbol = CString(span, (int)(strings.Offset + BinaryPrimitives.ReadUInt32LittleEndian(entry)));
            if (symbol == name + ".kd")
            {
                kd = BinaryPrimitives.ReadUInt64LittleEndian(entry[8..]);
            }
        }

        if (kd is not { } descriptorAddress)
        {
            throw new InvalidDataException($"the code object has no kernel '{name}' (no symbol {name}.kd)");
        }

        var holder = sections.First(s => s.Type != 8 && descriptorAddress >= s.Address && descriptorAddress + 64 <= s.Address + s.Size && s.Address != 0);
        var descriptorBytes = span.Slice((int)(holder.Offset + (descriptorAddress - holder.Address)), 64);
        var descriptor = GcnKernelDescriptor.Read(descriptorBytes);
        long entryAddress = (long)descriptorAddress + BinaryPrimitives.ReadInt64LittleEndian(descriptorBytes[16..]);
        var text = sections.FirstOrDefault(s => s.Name == ".text");
        if (text.Name is null || entryAddress < (long)text.Address || entryAddress >= (long)(text.Address + text.Size))
        {
            throw new InvalidDataException($"'{name}': the descriptor's entry (0x{entryAddress:X}) is not in .text");
        }

        byte[] code = span.Slice((int)text.Offset, (int)text.Size).ToArray();
        return new GcnKernelImage(name, cpu, code, (int)(entryAddress - (long)text.Address), descriptor, xnack);
    }

    private static List<Section> Sections(ReadOnlySpan<byte> elf)
    {
        ulong table = BinaryPrimitives.ReadUInt64LittleEndian(elf[40..]);
        int size = BinaryPrimitives.ReadUInt16LittleEndian(elf[58..]);
        int count = BinaryPrimitives.ReadUInt16LittleEndian(elf[60..]);
        int names = BinaryPrimitives.ReadUInt16LittleEndian(elf[62..]);
        var raw = new List<(uint Name, uint Type, ulong Address, ulong Offset, ulong Size, uint Link)>();
        for (int i = 0; i < count; i++)
        {
            var h = elf[(int)(table + (ulong)(i * size))..];
            raw.Add((BinaryPrimitives.ReadUInt32LittleEndian(h), BinaryPrimitives.ReadUInt32LittleEndian(h[4..]), BinaryPrimitives.ReadUInt64LittleEndian(h[16..]),
                BinaryPrimitives.ReadUInt64LittleEndian(h[24..]), BinaryPrimitives.ReadUInt64LittleEndian(h[32..]), BinaryPrimitives.ReadUInt32LittleEndian(h[40..])));
        }

        ulong nameOffset = raw[names].Offset;
        var result = new List<Section>(count);
        foreach (var s in raw)
        {
            result.Add(new Section(CString(elf, (int)(nameOffset + s.Name)), s.Type, s.Address, s.Offset, s.Size, s.Link));
        }

        return result;
    }

    private static string CString(ReadOnlySpan<byte> bytes, int at)
    {
        int end = bytes[at..].IndexOf((byte)0);
        return Encoding.UTF8.GetString(bytes.Slice(at, end));
    }
}
