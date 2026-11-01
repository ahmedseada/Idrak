// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Frozen;

namespace Idrak.Tests.Gcn;

// The RDNA families' words (plan 35, G8a), read from the RDNA 1, RDNA 2 and RDNA 3 ISA references' layouts and checked
// against what llvm-mc 18 wrote for every wave64 form of tests/Idrak.Tests/data/gcn/gfx10 and gfx11: the scalar and
// VOP1, VOP2, VOPC words are gfx9's; SMEM, VOP3, VOP3P, DS, FLAT and MUBUF have their own prefixes and fields. The
// registers are held in gfx9's numbering, so the machine and the semantics are shared: M0 124 and null 125 (ISA 11
// swaps the two codes; Sr maps them), s0-s105 below VCC (106).
internal sealed partial class GcnDecoder
{
    // The null register's code once mapped (Sr): what an RDNA operand of 125 is.
    internal const int Null = 125;

    /// <summary>A scalar operand's code in gfx9's numbering: ISA 11's M0 (125) and null (124) swapped to 124 and 125.</summary>
    private int Sr(int value) => isa11 && value is 124 or 125 ? 249 - value : value;

    // Whether a VOP3 or VOP3P source is the literal in a third word (RDNA only; one literal value for every such source).
    private bool SourceLiteral(int offset, uint w0, uint w1, bool hasW2, string kind, int s0, int s1, int s2)
    {
        bool literal = false;
        foreach (int s in (ReadOnlySpan<int>)[s0, s1, s2])
        {
            if (!rdna && s is 255 or 0xF9 or 0xFA)
            {
                throw Refuse(offset, $"{w0:X8} {w1:X8}", $"a {kind} source on gfx9 cannot be a literal, DPP or SDWA");
            }

            if (rdna && s is 0xE9 or 0xEA or 0xF9 or 0xFA)
            {
                throw Refuse(offset, $"{w0:X8} {w1:X8}", $"a {kind} source cannot be DPP or SDWA");
            }

            literal |= s == 255;
        }

        if (literal && !hasW2)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "a literal constant follows, but the code ends");
        }

        return literal;
    }

    // The words that are neither scalar nor VOP1, VOP2 or VOPC: located by GcnOpcodeTable.Locate, then read by format.
    private GcnInstruction DecodeRdna(int offset, uint w0, uint w1, bool hasW1, uint w2, bool hasW2)
    {
        if (GcnOpcodeTable.Locate(isa11, w0) is not { } at)
        {
            string kind = (w0 >> 26) switch
            {
                0x32 => isa11 ? "VOPD" : "VINTRP",
                0x3A => "MTBUF",
                0x3C => "MIMG",
                0x3E => "EXP",
                _ => "a reserved encoding",
            };
            throw Refuse(offset, $"{w0:X8}", $"{kind}: not decoded by the interpreter");
        }

        Need(hasW1, offset, w0, at.Format.ToString().ToUpperInvariant());
        return at.Format switch
        {
            GcnFormat.Smem => RdnaSmem(offset, w0, w1, at.Code),
            GcnFormat.Vop3 => DecodeVop3(offset, w0, w1, w2, hasW2),
            GcnFormat.Vop3p => DecodeVop3p(offset, w0, w1, w2, hasW2),
            GcnFormat.Ds => DecodeDs(offset, w0, w1),
            GcnFormat.Mubuf => RdnaMubuf(offset, w0, w1, at.Code),
            _ => RdnaFlat(offset, w0, w1, at.Format, at.Code),
        };
    }

    // SMEM (0xF4): sbase/2 5:0, sdata 12:6, glc 16 and dlc 14 (ISA 10) or glc 14 and dlc 13 (ISA 11), opcode 25:18; the
    // second word's offset 20:0 and soffset 31:25 (null when the offset is the immediate alone).
    private GcnInstruction RdnaSmem(int offset, uint w0, uint w1, int code)
    {
        var opcode = Opcode(GcnFormat.Smem, code, offset, $"{w0:X8} {w1:X8}");
        uint glcBit = isa11 ? 1u << 14 : 1u << 16, dlcBit = isa11 ? 1u << 13 : 1u << 14;
        uint known = 0x3Fu | 0x7Fu << 6 | glcBit | dlcBit | 0xFFu << 18 | 0x3Fu << 26;
        if ((w0 & ~known) != 0 || (w1 & 0x01E00000) != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "SMEM bits the family does not define");
        }

        int soffset = Sr((int)(w1 >> 25) & 0x7F), immediate = (int)(w1 & 0x1FFFFF);
        bool register = soffset != Null;
        if (register && immediate != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", "SMEM with both a register and an immediate offset is not decoded by the interpreter");
        }

        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = w1,
            A = ((int)w0 & 0x3F) << 1,
            D = Sr((int)(w0 >> 6) & 0x7F),
            Glc = (w0 & glcBit) != 0,
            Dlc = (w0 & dlcBit) != 0,
            Imm = !register,
            MemoryOffset = register ? soffset : immediate,
        };
    }

    // FLAT (0xDC): opcode 24:18; ISA 10 offset 11:0, dlc 12, lds 13, seg 15:14, glc 16, slc 17; ISA 11 offset 12:0, dlc
    // 13, glc 14, slc 15, seg 17:16. The offset is signed for GLOBAL and SCRATCH (12 bits on ISA 10, 13 on ISA 11), and
    // unsigned and a bit shorter for FLAT. The second word is gfx9's, saddr's off being null (0x7D, ISA 11 0x7C).
    private GcnInstruction RdnaFlat(int offset, uint w0, uint w1, GcnFormat format, int code)
    {
        string words = $"{w0:X8} {w1:X8}";
        if (format == GcnFormat.Global && (isa11 ? (w0 >> 16) & 3 : (w0 >> 14) & 3) == 3)
        {
            throw Refuse(offset, words, "FLAT segment 3 is reserved");
        }

        if ((!isa11 && (w0 & (1u << 13)) != 0) || (w0 & (1u << 25)) != 0 || (w1 & (1u << 23)) != 0)
        {
            throw Refuse(offset, words, "FLAT with lds or a reserved bit set is not decoded by the interpreter");
        }

        var opcode = Opcode(format, code, offset, words);
        int bits = isa11 ? 13 : 12;
        int field = (int)w0 & ((1 << bits) - 1);
        int imm = format == GcnFormat.Flat ? field : field << (32 - bits) >> (32 - bits);
        if (format == GcnFormat.Flat && (field >> (bits - 1)) != 0)
        {
            throw Refuse(offset, words, $"FLAT's offset is {bits - 1} bits, unsigned, on this family");
        }

        int saddr = (int)(w1 >> 16) & 0x7F;
        return new GcnInstruction
        {
            Offset = offset,
            Size = 8,
            Opcode = opcode,
            Cpu = cpu,
            Word0 = w0,
            Word1 = w1,
            MemoryOffset = imm,
            Glc = (w0 & (isa11 ? 1u << 14 : 1u << 16)) != 0,
            Slc = (w0 & (isa11 ? 1u << 15 : 1u << 17)) != 0,
            Dlc = (w0 & (isa11 ? 1u << 13 : 1u << 12)) != 0,
            A = 256 + ((int)w1 & 0xFF),
            S0 = 256 + ((int)(w1 >> 8) & 0xFF),
            D = 256 + ((int)(w1 >> 24) & 0xFF),
            SAddr = format == GcnFormat.Flat || Sr(saddr) == Null ? -1 : saddr,
        };
    }

    // MUBUF (0xE0): the cache invalidations alone (buffer_gl0_inv, buffer_gl1_inv), which take no operands.
    private GcnInstruction RdnaMubuf(int offset, uint w0, uint w1, int code)
    {
        var opcode = Opcode(GcnFormat.Mubuf, code, offset, $"{w0:X8} {w1:X8}");
        if ((w0 & 0xFFFF) != 0 || w1 != 0)
        {
            throw Refuse(offset, $"{w0:X8} {w1:X8}", $"{opcode.Name} with operands or modifiers is not decoded by the interpreter");
        }

        return new GcnInstruction { Offset = offset, Size = 8, Opcode = opcode, Cpu = cpu, Word0 = w0, Word1 = w1 };
    }
}

internal static partial class GcnOpcodeTable
{
    // gfx10's and gfx11's renamed mnemonics, as gfx9 names the same operation (the semantics' keys); the memory names
    // (the width in bits for gfx9's dwords, load and store for read and write) by the rules of Gfx9Stem. A class of their
    // own, so they are set before Build (in another file of this partial class) reads them.
    private static class Stems
    {
        internal static readonly FrozenDictionary<string, string> Renamed = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["v_add_nc_u32"] = "v_add_u32",
            ["v_sub_nc_u32"] = "v_sub_u32",
            ["v_subrev_nc_u32"] = "v_subrev_u32",
            ["global_load_u8"] = "global_load_ubyte",
            ["global_load_i8"] = "global_load_sbyte",
            ["global_load_u16"] = "global_load_ushort",
            ["global_load_i16"] = "global_load_sshort",
            ["global_store_b8"] = "global_store_byte",
            ["global_store_b16"] = "global_store_short",
            ["flat_load_u8"] = "flat_load_ubyte",
            ["flat_load_i8"] = "flat_load_sbyte",
            ["flat_load_u16"] = "flat_load_ushort",
            ["flat_load_i16"] = "flat_load_sshort",
            ["flat_store_b8"] = "flat_store_byte",
            ["flat_store_b16"] = "flat_store_short",
            ["ds_load_u8"] = "ds_read_u8",
            ["ds_load_i8"] = "ds_read_i8",
            ["ds_load_u16"] = "ds_read_u16",
            ["ds_load_i16"] = "ds_read_i16",
            ["ds_store_b8"] = "ds_write_b8",
            ["ds_store_b16"] = "ds_write_b16",
        }.ToFrozenDictionary(StringComparer.Ordinal);

        internal static readonly FrozenDictionary<string, string> Atomics = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            ["add_u32"] = "add",
            ["sub_u32"] = "sub",
            ["swap_b32"] = "swap",
            ["cmpswap_b32"] = "cmpswap",
            ["and_b32"] = "and",
            ["or_b32"] = "or",
            ["xor_b32"] = "xor",
            ["inc_u32"] = "inc",
            ["dec_u32"] = "dec",
            ["min_i32"] = "smin",
            ["max_i32"] = "smax",
            ["min_u32"] = "umin",
            ["max_u32"] = "umax",
            ["add_u64"] = "add_x2",
            ["swap_b64"] = "swap_x2",
            ["cmpswap_b64"] = "cmpswap_x2",
        }.ToFrozenDictionary(StringComparer.Ordinal);
    }

    /// <summary>
    /// The gfx9 name of an opcode's stem (unchanged for gfx9's own): gfx10's v_add_nc_u32, v_sub_nc_u32 and
    /// v_subrev_nc_u32; gfx11's s_load_b32 to b512 (s_load_dword to dwordx16), global_, flat_ and scratch_ load_b32 to
    /// b128 and store_b32 to b128 (dword to dwordx4), the byte and short loads and stores, ds_load_b and ds_store_b
    /// (ds_read_b, ds_write_b), ds_load_2addr and ds_store_2addr (ds_read2, ds_write2, and their stride64 forms), the
    /// atomics' typed names (global_atomic_add_u32 is global_atomic_add), and s_and_not1 and s_or_not1 (s_andn2, s_orn2).
    /// </summary>
    internal static string Gfx9Stem(string stem)
    {
        if (Stems.Renamed.TryGetValue(stem, out string? old))
        {
            return old;
        }

        foreach (string prefix in (string[])["s_load_b", "s_buffer_load_b"])
        {
            if (stem.StartsWith(prefix, StringComparison.Ordinal))
            {
                string dwords = stem[prefix.Length..] switch { "32" => "", "64" => "x2", "128" => "x4", "256" => "x8", "512" => "x16", _ => "?" };
                return dwords == "?" ? stem : prefix[..^1] + "dword" + dwords;
            }
        }

        foreach (string space in (string[])["global_", "flat_", "scratch_"])
        {
            foreach (string kind in (string[])["load_b", "store_b"])
            {
                if (stem.StartsWith(space + kind, StringComparison.Ordinal))
                {
                    string dwords = stem[(space.Length + kind.Length)..] switch { "32" => "", "64" => "x2", "96" => "x3", "128" => "x4", _ => "?" };
                    return dwords == "?" ? stem : space + kind[..^1] + "dword" + dwords;
                }
            }

            if (stem.StartsWith(space + "atomic_", StringComparison.Ordinal) && Stems.Atomics.TryGetValue(stem[(space.Length + 7)..], out string? atomic))
            {
                return space + "atomic_" + atomic;
            }
        }

        if (stem.StartsWith("ds_", StringComparison.Ordinal))
        {
            string name = stem.Replace("ds_load_2addr_stride64_", "ds_read2st64_", StringComparison.Ordinal)
                .Replace("ds_store_2addr_stride64_", "ds_write2st64_", StringComparison.Ordinal)
                .Replace("ds_load_2addr_", "ds_read2_", StringComparison.Ordinal)
                .Replace("ds_store_2addr_", "ds_write2_", StringComparison.Ordinal);
            if (name.StartsWith("ds_load_b", StringComparison.Ordinal))
            {
                return "ds_read_b" + name["ds_load_b".Length..];
            }

            return name.StartsWith("ds_store_b", StringComparison.Ordinal) ? "ds_write_b" + name["ds_store_b".Length..] : name;
        }

        return stem.StartsWith("s_", StringComparison.Ordinal)
            ? stem.Replace("_and_not1_", "_andn2_", StringComparison.Ordinal).Replace("_or_not1_", "_orn2_", StringComparison.Ordinal)
            : stem;
    }
}
