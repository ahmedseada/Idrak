// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;

namespace Idrak.Tests.Gcn;

/// <summary>What an instruction does to a wave (an opcode's semantics).</summary>
internal delegate void GcnExecute(GcnMachine m, GcnWave w, GcnInstruction i);

/// <summary>
/// The interpreter's semantics, by mnemonic (without _e32 or _e64): the scalar ALU, program control and the memory
/// instructions here, the vector ALU in GcnInterpreter.Vector.cs. A mnemonic not here is decoded and printed but refused
/// when run, so a kernel never passes through an instruction the interpreter does not model.
/// </summary>
internal static partial class GcnSemantics
{
    // LaneForms first: Build fills it (static fields start in the order they are written).
    private static readonly HashSet<string> LaneForms = new(StringComparer.Ordinal);

    private static readonly Dictionary<string, GcnExecute> Table = Build();

    internal static GcnExecute? Find(string stem) => Table.GetValueOrDefault(stem);

    /// <summary>Whether the DPP and SDWA forms of <paramref name="stem"/> are modelled.</summary>
    internal static bool TakesLaneForms(string stem) => LaneForms.Contains(stem);

    private static Dictionary<string, GcnExecute> Build()
    {
        var table = new Dictionary<string, GcnExecute>(StringComparer.Ordinal);
        AddScalar(table);
        AddControl(table);
        AddMemory(table);
        AddVector(table);
        return table;
    }

    // ---- SOP1, SOP2, SOPK, SOPC -----------------------------------------------------------------------------------

    private static void Unary32(Dictionary<string, GcnExecute> t, string name, Func<uint, uint> f, bool scc) =>
        t[name] = (m, w, i) =>
        {
            uint d = f(m.Scalar32(w, i.S0, i));
            m.WriteScalar(w, i.D, d, i);
            if (scc)
            {
                w.Scc = d != 0;
            }
        };

    private static void Unary64(Dictionary<string, GcnExecute> t, string name, Func<ulong, ulong> f, bool scc) =>
        t[name] = (m, w, i) =>
        {
            ulong d = f(m.Scalar64(w, i.S0, i));
            m.WriteScalar64(w, i.D, d, i);
            if (scc)
            {
                w.Scc = d != 0;
            }
        };

    private static void Count64(Dictionary<string, GcnExecute> t, string name, Func<ulong, uint> f, bool scc) =>
        t[name] = (m, w, i) =>
        {
            uint d = f(m.Scalar64(w, i.S0, i));
            m.WriteScalar(w, i.D, d, i);
            if (scc)
            {
                w.Scc = d != 0;
            }
        };

    private static void Binary32(Dictionary<string, GcnExecute> t, string name, Func<uint, uint, uint> f, bool scc) =>
        t[name] = (m, w, i) =>
        {
            uint d = f(m.Scalar32(w, i.S0, i), m.Scalar32(w, i.S1, i));
            m.WriteScalar(w, i.D, d, i);
            if (scc)
            {
                w.Scc = d != 0;
            }
        };

    private static void Binary64(Dictionary<string, GcnExecute> t, string name, Func<ulong, ulong, ulong> f, bool scc, bool shiftCount = false) =>
        t[name] = (m, w, i) =>
        {
            ulong a = m.Scalar64(w, i.S0, i);
            ulong b = shiftCount ? m.Scalar32(w, i.S1, i) : m.Scalar64(w, i.S1, i);
            ulong d = f(a, b);
            m.WriteScalar64(w, i.D, d, i);
            if (scc)
            {
                w.Scc = d != 0;
            }
        };

    private static void Compare32(Dictionary<string, GcnExecute> t, string name, Func<uint, uint, bool> f) =>
        t[name] = (m, w, i) => w.Scc = f(m.Scalar32(w, i.S0, i), m.Scalar32(w, i.S1, i));

    private static void SaveExec(Dictionary<string, GcnExecute> t, string name, Func<ulong, ulong, ulong> f) =>
        t[name] = (m, w, i) =>
        {
            ulong source = m.Scalar64(w, i.S0, i);
            ulong exec = w.Exec;
            m.WriteScalar64(w, i.D, exec, i);
            w.Exec = f(source, exec);
            w.Scc = w.Exec != 0;
        };

    private static uint FirstBitFromTop32(uint v) => v == 0 ? uint.MaxValue : (uint)BitOperations.LeadingZeroCount(v);

    private static void AddScalar(Dictionary<string, GcnExecute> t)
    {
        Unary32(t, "s_mov_b32", a => a, false);
        Unary64(t, "s_mov_b64", a => a, false);
        t["s_cmov_b32"] = (m, w, i) =>
        {
            uint a = m.Scalar32(w, i.S0, i);
            if (w.Scc)
            {
                m.WriteScalar(w, i.D, a, i);
            }
        };
        t["s_cmov_b64"] = (m, w, i) =>
        {
            ulong a = m.Scalar64(w, i.S0, i);
            if (w.Scc)
            {
                m.WriteScalar64(w, i.D, a, i);
            }
        };
        Unary32(t, "s_not_b32", a => ~a, true);
        Unary64(t, "s_not_b64", a => ~a, true);
        Unary32(t, "s_wqm_b32", Wqm32, true);
        Unary64(t, "s_wqm_b64", a => Wqm32((uint)a) | (ulong)Wqm32((uint)(a >> 32)) << 32, true);
        Unary32(t, "s_brev_b32", ReverseBits32, false);
        Unary64(t, "s_brev_b64", a => (ulong)ReverseBits32((uint)a) << 32 | ReverseBits32((uint)(a >> 32)), false);
        Unary32(t, "s_bcnt0_i32_b32", a => (uint)BitOperations.PopCount(~a), true);
        Count64(t, "s_bcnt0_i32_b64", a => (uint)BitOperations.PopCount(~a), true);
        Unary32(t, "s_bcnt1_i32_b32", a => (uint)BitOperations.PopCount(a), true);
        Count64(t, "s_bcnt1_i32_b64", a => (uint)BitOperations.PopCount(a), true);
        Unary32(t, "s_ff0_i32_b32", a => ~a == 0 ? uint.MaxValue : (uint)BitOperations.TrailingZeroCount(~a), false);
        Count64(t, "s_ff0_i32_b64", a => ~a == 0 ? uint.MaxValue : (uint)BitOperations.TrailingZeroCount(~a), false);
        Unary32(t, "s_ff1_i32_b32", a => a == 0 ? uint.MaxValue : (uint)BitOperations.TrailingZeroCount(a), false);
        Count64(t, "s_ff1_i32_b64", a => a == 0 ? uint.MaxValue : (uint)BitOperations.TrailingZeroCount(a), false);
        Unary32(t, "s_flbit_i32_b32", FirstBitFromTop32, false);
        Count64(t, "s_flbit_i32_b64", a => a == 0 ? uint.MaxValue : (uint)BitOperations.LeadingZeroCount(a), false);
        Unary32(t, "s_flbit_i32", a => (int)a < 0 ? FirstBitFromTop32(~a) : FirstBitFromTop32(a), false);
        Count64(t, "s_flbit_i32_i64", a =>
        {
            ulong v = (long)a < 0 ? ~a : a;
            return v == 0 ? uint.MaxValue : (uint)BitOperations.LeadingZeroCount(v);
        }, false);
        Unary32(t, "s_sext_i32_i8", a => (uint)(sbyte)a, false);
        Unary32(t, "s_sext_i32_i16", a => (uint)(short)a, false);
        t["s_bitset0_b32"] = (m, w, i) => m.WriteScalar(w, i.D, m.ReadScalar(w, i.D, i) & ~(1u << (int)(m.Scalar32(w, i.S0, i) & 31)), i);
        t["s_bitset1_b32"] = (m, w, i) => m.WriteScalar(w, i.D, m.ReadScalar(w, i.D, i) | 1u << (int)(m.Scalar32(w, i.S0, i) & 31), i);
        t["s_bitset0_b64"] = (m, w, i) => m.WriteScalar64(w, i.D, m.ReadScalar64(w, i.D, i) & ~(1UL << (int)(m.Scalar32(w, i.S0, i) & 63)), i);
        t["s_bitset1_b64"] = (m, w, i) => m.WriteScalar64(w, i.D, m.ReadScalar64(w, i.D, i) | 1UL << (int)(m.Scalar32(w, i.S0, i) & 63), i);
        Unary32(t, "s_abs_i32", a => (uint)Math.Abs((long)(int)a), true);
        t["s_getpc_b64"] = (m, w, i) => m.WriteScalar64(w, i.D, m.CodeAddress + (ulong)(i.Offset + 4), i);
        t["s_setpc_b64"] = (m, w, i) => w.Next = m.CodeOffset(w, i, m.Scalar64(w, i.S0, i));
        t["s_swappc_b64"] = (m, w, i) =>
        {
            ulong target = m.Scalar64(w, i.S0, i);
            m.WriteScalar64(w, i.D, m.CodeAddress + (ulong)(i.Offset + 4), i);
            w.Next = m.CodeOffset(w, i, target);
        };
        SaveExec(t, "s_and_saveexec_b64", (s, e) => s & e);
        SaveExec(t, "s_or_saveexec_b64", (s, e) => s | e);
        SaveExec(t, "s_xor_saveexec_b64", (s, e) => s ^ e);
        SaveExec(t, "s_andn2_saveexec_b64", (s, e) => s & ~e);
        SaveExec(t, "s_orn2_saveexec_b64", (s, e) => s | ~e);
        SaveExec(t, "s_nand_saveexec_b64", (s, e) => ~(s & e));
        SaveExec(t, "s_nor_saveexec_b64", (s, e) => ~(s | e));
        SaveExec(t, "s_xnor_saveexec_b64", (s, e) => ~(s ^ e));
        SaveExec(t, "s_andn1_saveexec_b64", (s, e) => ~s & e);
        SaveExec(t, "s_orn1_saveexec_b64", (s, e) => ~s | e);
        t["s_movrels_b32"] = (m, w, i) => m.WriteScalar(w, i.D, m.ReadScalar(w, i.S0 + (int)m.ReadScalar(w, 124, i), i), i);
        t["s_movreld_b32"] = (m, w, i) => m.WriteScalar(w, i.D + (int)m.ReadScalar(w, 124, i), m.Scalar32(w, i.S0, i), i);

        t["s_add_u32"] = (m, w, i) => Add(m, w, i, 0, false);
        t["s_addc_u32"] = (m, w, i) => Add(m, w, i, w.Scc ? 1u : 0, false);
        t["s_sub_u32"] = (m, w, i) => Add(m, w, i, 0, true);
        t["s_subb_u32"] = (m, w, i) => Add(m, w, i, w.Scc ? 1u : 0, true);
        t["s_add_i32"] = (m, w, i) =>
        {
            int a = (int)m.Scalar32(w, i.S0, i), b = (int)m.Scalar32(w, i.S1, i);
            long d = (long)a + b;
            m.WriteScalar(w, i.D, (uint)d, i);
            w.Scc = d != (int)d;
        };
        t["s_sub_i32"] = (m, w, i) =>
        {
            int a = (int)m.Scalar32(w, i.S0, i), b = (int)m.Scalar32(w, i.S1, i);
            long d = (long)a - b;
            m.WriteScalar(w, i.D, (uint)d, i);
            w.Scc = d != (int)d;
        };
        foreach (var (name, signed, max) in (ReadOnlySpan<(string, bool, bool)>)[("s_min_i32", true, false), ("s_min_u32", false, false), ("s_max_i32", true, true), ("s_max_u32", false, true)])
        {
            t[name] = (m, w, i) =>
            {
                uint a = m.Scalar32(w, i.S0, i), b = m.Scalar32(w, i.S1, i);
                bool first = signed ? (max ? (int)a > (int)b : (int)a < (int)b) : (max ? a > b : a < b);
                m.WriteScalar(w, i.D, first ? a : b, i);
                w.Scc = first;
            };
        }

        t["s_cselect_b32"] = (m, w, i) =>
        {
            uint a = m.Scalar32(w, i.S0, i), b = m.Scalar32(w, i.S1, i);
            m.WriteScalar(w, i.D, w.Scc ? a : b, i);
        };
        t["s_cselect_b64"] = (m, w, i) =>
        {
            ulong a = m.Scalar64(w, i.S0, i), b = m.Scalar64(w, i.S1, i);
            m.WriteScalar64(w, i.D, w.Scc ? a : b, i);
        };
        Binary32(t, "s_and_b32", (a, b) => a & b, true);
        Binary64(t, "s_and_b64", (a, b) => a & b, true);
        Binary32(t, "s_or_b32", (a, b) => a | b, true);
        Binary64(t, "s_or_b64", (a, b) => a | b, true);
        Binary32(t, "s_xor_b32", (a, b) => a ^ b, true);
        Binary64(t, "s_xor_b64", (a, b) => a ^ b, true);
        Binary32(t, "s_andn2_b32", (a, b) => a & ~b, true);
        Binary64(t, "s_andn2_b64", (a, b) => a & ~b, true);
        Binary32(t, "s_orn2_b32", (a, b) => a | ~b, true);
        Binary64(t, "s_orn2_b64", (a, b) => a | ~b, true);
        Binary32(t, "s_nand_b32", (a, b) => ~(a & b), true);
        Binary64(t, "s_nand_b64", (a, b) => ~(a & b), true);
        Binary32(t, "s_nor_b32", (a, b) => ~(a | b), true);
        Binary64(t, "s_nor_b64", (a, b) => ~(a | b), true);
        Binary32(t, "s_xnor_b32", (a, b) => ~(a ^ b), true);
        Binary64(t, "s_xnor_b64", (a, b) => ~(a ^ b), true);
        Binary32(t, "s_lshl_b32", (a, b) => a << (int)(b & 31), true);
        Binary64(t, "s_lshl_b64", (a, b) => a << (int)(b & 63), true, shiftCount: true);
        Binary32(t, "s_lshr_b32", (a, b) => a >> (int)(b & 31), true);
        Binary64(t, "s_lshr_b64", (a, b) => a >> (int)(b & 63), true, shiftCount: true);
        Binary32(t, "s_ashr_i32", (a, b) => (uint)((int)a >> (int)(b & 31)), true);
        Binary64(t, "s_ashr_i64", (a, b) => (ulong)((long)a >> (int)(b & 63)), true, shiftCount: true);
        Binary32(t, "s_bfm_b32", (a, b) => (uint)((1UL << (int)(a & 31)) - 1) << (int)(b & 31), false);
        t["s_bfm_b64"] = (m, w, i) =>
        {
            uint a = m.Scalar32(w, i.S0, i), b = m.Scalar32(w, i.S1, i);
            ulong field = (a & 63) == 0 ? 0 : ulong.MaxValue >> (64 - (int)(a & 63));
            m.WriteScalar64(w, i.D, field << (int)(b & 63), i);
        };
        Binary32(t, "s_mul_i32", (a, b) => a * b, false);
        Binary32(t, "s_mul_hi_u32", (a, b) => (uint)((ulong)a * b >> 32), false);
        Binary32(t, "s_mul_hi_i32", (a, b) => (uint)((long)(int)a * (int)b >> 32), false);
        Binary32(t, "s_bfe_u32", (a, b) => BitField(a, (int)(b & 31), (int)(b >> 16) & 0x7F, false), true);
        Binary32(t, "s_bfe_i32", (a, b) => BitField(a, (int)(b & 31), (int)(b >> 16) & 0x7F, true), true);
        Binary64(t, "s_bfe_u64", (a, b) => BitField64(a, (int)(b & 63), (int)(b >> 16) & 0x7F, false), true, shiftCount: true);
        Binary64(t, "s_bfe_i64", (a, b) => BitField64(a, (int)(b & 63), (int)(b >> 16) & 0x7F, true), true, shiftCount: true);
        Binary32(t, "s_absdiff_i32", (a, b) => (uint)Math.Abs((long)(int)a - (int)b), true);
        for (int shift = 1; shift <= 4; shift++)
        {
            int s = shift;
            t[$"s_lshl{s}_add_u32"] = (m, w, i) =>
            {
                ulong d = ((ulong)m.Scalar32(w, i.S0, i) << s) + m.Scalar32(w, i.S1, i);
                m.WriteScalar(w, i.D, (uint)d, i);
                w.Scc = d >> 32 != 0;
            };
        }

        Binary32(t, "s_pack_ll_b32_b16", (a, b) => (a & 0xFFFF) | b << 16, false);
        Binary32(t, "s_pack_lh_b32_b16", (a, b) => (a & 0xFFFF) | (b & 0xFFFF0000), false);
        Binary32(t, "s_pack_hh_b32_b16", (a, b) => a >> 16 | (b & 0xFFFF0000), false);

        // SOPK: the 16-bit immediate, sign- or zero-extended.
        t["s_movk_i32"] = (m, w, i) => m.WriteScalar(w, i.D, (uint)(short)i.Simm16, i);
        t["s_cmovk_i32"] = (m, w, i) =>
        {
            if (w.Scc)
            {
                m.WriteScalar(w, i.D, (uint)(short)i.Simm16, i);
            }
        };
        t["s_addk_i32"] = (m, w, i) =>
        {
            long d = (long)(int)m.ReadScalar(w, i.D, i) + (short)i.Simm16;
            m.WriteScalar(w, i.D, (uint)d, i);
            w.Scc = d != (int)d;
        };
        t["s_mulk_i32"] = (m, w, i) => m.WriteScalar(w, i.D, (uint)((int)m.ReadScalar(w, i.D, i) * (short)i.Simm16), i);
        foreach (var (cond, f) in Conditions())
        {
            var test = f;
            t[$"s_cmpk_{cond}_i32"] = (m, w, i) => w.Scc = test((int)m.ReadScalar(w, i.D, i), (short)i.Simm16);
            t[$"s_cmpk_{cond}_u32"] = (m, w, i) => w.Scc = test(m.ReadScalar(w, i.D, i), (ushort)i.Simm16);
            Compare32(t, $"s_cmp_{cond}_i32", (a, b) => test((int)a, (int)b));
            Compare32(t, $"s_cmp_{cond}_u32", (a, b) => test(a, b));
        }

        t["s_call_b64"] = (m, w, i) =>
        {
            m.WriteScalar64(w, i.D, m.CodeAddress + (ulong)(i.Offset + 4), i);
            w.Next = i.Offset + 4 + 4 * (short)i.Simm16;
        };
        t["s_cmp_eq_u64"] = (m, w, i) => w.Scc = m.Scalar64(w, i.S0, i) == m.Scalar64(w, i.S1, i);
        t["s_cmp_lg_u64"] = (m, w, i) => w.Scc = m.Scalar64(w, i.S0, i) != m.Scalar64(w, i.S1, i);
        Compare32(t, "s_bitcmp0_b32", (a, b) => (a >> (int)(b & 31) & 1) == 0);
        Compare32(t, "s_bitcmp1_b32", (a, b) => (a >> (int)(b & 31) & 1) == 1);
        t["s_bitcmp0_b64"] = (m, w, i) => w.Scc = (m.Scalar64(w, i.S0, i) >> (int)(m.Scalar32(w, i.S1, i) & 63) & 1) == 0;
        t["s_bitcmp1_b64"] = (m, w, i) => w.Scc = (m.Scalar64(w, i.S0, i) >> (int)(m.Scalar32(w, i.S1, i) & 63) & 1) == 1;
    }

    private static (string, Func<long, long, bool>)[] Conditions() =>
    [
        ("eq", (a, b) => a == b), ("lg", (a, b) => a != b), ("gt", (a, b) => a > b), ("ge", (a, b) => a >= b), ("lt", (a, b) => a < b), ("le", (a, b) => a <= b),
    ];

    private static void Add(GcnMachine m, GcnWave w, GcnInstruction i, uint carry, bool subtract)
    {
        ulong a = m.Scalar32(w, i.S0, i), b = m.Scalar32(w, i.S1, i);
        ulong d = subtract ? a - b - carry : a + b + carry;
        m.WriteScalar(w, i.D, (uint)d, i);
        w.Scc = subtract ? b + carry > a : d >> 32 != 0;
    }

    private static uint Wqm32(uint a)
    {
        uint d = 0;
        for (int q = 0; q < 8; q++)
        {
            if ((a >> (4 * q) & 0xF) != 0)
            {
                d |= 0xFu << (4 * q);
            }
        }

        return d;
    }

    internal static uint ReverseBits32(uint v)
    {
        v = (v >> 1 & 0x55555555) | (v & 0x55555555) << 1;
        v = (v >> 2 & 0x33333333) | (v & 0x33333333) << 2;
        v = (v >> 4 & 0x0F0F0F0F) | (v & 0x0F0F0F0F) << 4;
        return BinaryReverse(v);
    }

    private static uint BinaryReverse(uint v) => System.Buffers.Binary.BinaryPrimitives.ReverseEndianness(v);

    internal static uint BitField(uint value, int offset, int width, bool signed)
    {
        if (width == 0)
        {
            return 0;
        }

        width = Math.Min(width, 32);
        uint field = (uint)((ulong)value >> offset) & (width == 32 ? uint.MaxValue : (1u << width) - 1);
        return signed && width < 32 && (field >> (width - 1) & 1) != 0 ? field | ~((1u << width) - 1) : field;
    }

    private static ulong BitField64(ulong value, int offset, int width, bool signed)
    {
        if (width == 0)
        {
            return 0;
        }

        width = Math.Min(width, 64);
        ulong field = (value >> offset) & (width == 64 ? ulong.MaxValue : (1UL << width) - 1);
        return signed && width < 64 && (field >> (width - 1) & 1) != 0 ? field | ~((1UL << width) - 1) : field;
    }

    // ---- SOPP: program control ------------------------------------------------------------------------------------

    private static void Branch(Dictionary<string, GcnExecute> t, string name, Func<GcnMachine, GcnWave, GcnInstruction, bool> taken) =>
        t[name] = (m, w, i) =>
        {
            if (taken(m, w, i))
            {
                w.Next = i.Offset + 4 + 4 * (short)i.Simm16;
            }
        };

    private static void AddControl(Dictionary<string, GcnExecute> t)
    {
        foreach (string name in (string[])["s_nop", "s_sleep", "s_setprio", "s_icache_inv", "s_incperflevel", "s_decperflevel"])
        {
            t[name] = static (_, _, _) => { };
        }

        t["s_endpgm"] = (m, w, i) =>
        {
            m.RetireAll(w);
            w.Ended = true;
        };
        Branch(t, "s_branch", (_, _, _) => true);
        Branch(t, "s_cbranch_scc0", (_, w, _) => !w.Scc);
        Branch(t, "s_cbranch_scc1", (_, w, _) => w.Scc);
        Branch(t, "s_cbranch_vccz", (m, w, i) => (m.ReadScalar(w, 106, i) | m.ReadScalar(w, 107, i)) == 0);
        Branch(t, "s_cbranch_vccnz", (m, w, i) => (m.ReadScalar(w, 106, i) | m.ReadScalar(w, 107, i)) != 0);
        Branch(t, "s_cbranch_execz", (m, w, i) => m.ExecOf(w, i) == 0);
        Branch(t, "s_cbranch_execnz", (m, w, i) => m.ExecOf(w, i) != 0);
        t["s_barrier"] = (_, w, _) => w.AtBarrier = true;
        t["s_waitcnt"] = (m, w, i) =>
        {
            int k = i.Simm16;
            m.Wait(w, (k & 0xF) | ((k >> 14) & 3) << 4, (k >> 8) & 0xF);
        };
    }

    // ---- Memory ---------------------------------------------------------------------------------------------------

    private static void AddMemory(Dictionary<string, GcnExecute> t)
    {
        foreach (string size in (string[])["", "x2", "x4", "x8", "x16"])
        {
            t["s_load_dword" + size] = GcnMachine.ScalarLoad;
            t["s_buffer_load_dword" + size] = GcnMachine.ScalarLoad;
        }

        t["s_memtime"] = GcnMachine.ScalarTime;
        t["s_memrealtime"] = GcnMachine.ScalarTime;
        foreach (string name in (string[])["s_dcache_wb", "s_dcache_inv", "s_dcache_wb_vol", "s_dcache_inv_vol"])
        {
            t[name] = static (_, _, _) => { };
        }

        foreach (string name in (string[])["ds_read_b32", "ds_read_b64", "ds_read_b96", "ds_read_b128", "ds_read_u8", "ds_read_i8", "ds_read_u16", "ds_read_i16",
                     "ds_read_u8_d16", "ds_read_u8_d16_hi", "ds_read_i8_d16", "ds_read_i8_d16_hi", "ds_read_u16_d16", "ds_read_u16_d16_hi",
                     "ds_read2_b32", "ds_read2_b64", "ds_read2st64_b32", "ds_read2st64_b64"])
        {
            t[name] = GcnMachine.LdsLoad;
        }

        foreach (string name in (string[])["ds_write_b32", "ds_write_b64", "ds_write_b96", "ds_write_b128", "ds_write_b8", "ds_write_b16", "ds_write_b8_d16_hi", "ds_write_b16_d16_hi",
                     "ds_write2_b32", "ds_write2_b64", "ds_write2st64_b32", "ds_write2st64_b64"])
        {
            t[name] = GcnMachine.LdsStore;
        }

        foreach (string op in (string[])["add", "sub", "rsub", "inc", "dec", "min", "max", "and", "or", "xor", "mskor", "wrxchg", "cmpst"])
        {
            foreach (string type in (string[])["u32", "i32", "b32", "f32", "u64", "i64", "b64"])
            {
                foreach (string rtn in (string[])["", "_rtn"])
                {
                    t[$"ds_{op}{rtn}_{type}"] = GcnMachine.LdsAtomic;
                }
            }
        }

        foreach (string name in (string[])["ds_swizzle_b32", "ds_permute_b32", "ds_bpermute_b32"])
        {
            t[name] = GcnMachine.LdsLanes;
        }

        foreach (string segment in (string[])["global", "flat"])
        {
            foreach (string size in (string[])["dword", "dwordx2", "dwordx3", "dwordx4", "ubyte", "sbyte", "ushort", "sshort", "short_d16", "short_d16_hi", "ubyte_d16", "ubyte_d16_hi", "sbyte_d16", "sbyte_d16_hi"])
            {
                t[$"{segment}_load_{size}"] = GcnMachine.VectorLoad;
            }

            foreach (string size in (string[])["dword", "dwordx2", "dwordx3", "dwordx4", "byte", "short", "byte_d16_hi", "short_d16_hi"])
            {
                t[$"{segment}_store_{size}"] = GcnMachine.VectorStore;
            }

            foreach (string op in (string[])["swap", "cmpswap", "add", "sub", "smin", "umin", "smax", "umax", "and", "or", "xor", "inc", "dec"])
            {
                t[$"{segment}_atomic_{op}"] = GcnMachine.VectorAtomic;
                t[$"{segment}_atomic_{op}_x2"] = GcnMachine.VectorAtomic;
            }

            t[$"{segment}_atomic_add_f32"] = GcnMachine.VectorAtomic;
        }
    }
}

internal sealed partial class GcnMachine
{
    /// <summary>The address the code is at, as s_getpc_b64 reads it (a fixed address, apart from every storage).</summary>
    internal ulong CodeAddress => 0x7F00_0000_0000;

    /// <summary>The code offset of a jump target address (s_setpc_b64, s_swappc_b64).</summary>
    internal int CodeOffset(GcnWave wave, GcnInstruction i, ulong target)
    {
        ulong offset = target - CodeAddress;
        if (target < CodeAddress || offset >= (ulong)kernel.Code.Length || offset % 4 != 0)
        {
            throw Fail(GcnFailure.Hang, wave, i, $"a jump to 0x{target:X}, outside the code at 0x{CodeAddress:X} ({kernel.Code.Length} bytes)");
        }

        return (int)offset;
    }
}
