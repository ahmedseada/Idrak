// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using static Idrak.Tests.Gcn.GcnFloat;

namespace Idrak.Tests.Gcn;

/// <summary>The type a vector operand is read or written as: its width, and whether float modifiers and output modifiers apply.</summary>
internal enum GcnType : byte
{
    B32,
    U32,
    I32,
    F32,
    U16,
    I16,
    F16,
    U64,
    I64,
    F64,
}

/// <summary>A vector ALU operation: its operand types and what it computes on one lane's bits.</summary>
internal sealed class GcnVectorOp
{
    public required GcnType D { get; init; }

    public required GcnType[] S { get; init; }

    /// <summary>One lane: sources' bits in, the result's bits out.</summary>
    public Func<ulong, ulong, ulong, ulong>? F { get; init; }

    /// <summary>One lane, knowing its index (v_mbcnt).</summary>
    public Func<ulong, ulong, ulong, int, ulong>? Lane { get; init; }

    /// <summary>The integer operation's clamp form (saturating), when it has one.</summary>
    public Func<ulong, ulong, ulong, ulong>? Saturate { get; init; }

    /// <summary>Float32 denormals flushed in and out whatever the mode (v_mad_f32, v_mac_f32 and their K forms on gfx9).</summary>
    public bool Mad { get; init; }

    /// <summary>Where each source comes from: 0-2 the source fields, 3 the destination (mac), 4 the literal (madmk, madak).</summary>
    public int[] Fields { get; init; } = [0, 1, 2];
}

internal static partial class GcnSemantics
{
    // Properties, not fields: Build runs from Table's initializer in another file, before this file's fields would be set.
    private static GcnType[] F1 => [GcnType.F32];

    private static GcnType[] F2 => [GcnType.F32, GcnType.F32];

    private static GcnType[] F3 => [GcnType.F32, GcnType.F32, GcnType.F32];

    private static GcnType[] U1 => [GcnType.U32];

    private static GcnType[] U2 => [GcnType.U32, GcnType.U32];

    private static GcnType[] U3 => [GcnType.U32, GcnType.U32, GcnType.U32];

    private static GcnType[] I2 => [GcnType.I32, GcnType.I32];

    private static GcnType[] I3 => [GcnType.I32, GcnType.I32, GcnType.I32];

    private static GcnType[] H1 => [GcnType.F16];

    private static GcnType[] H2 => [GcnType.F16, GcnType.F16];

    private static GcnType[] H3 => [GcnType.F16, GcnType.F16, GcnType.F16];

    private static GcnType[] D1 => [GcnType.F64];

    private static GcnType[] D2 => [GcnType.F64, GcnType.F64];

    private static GcnType[] D3 => [GcnType.F64, GcnType.F64, GcnType.F64];

    private static void Op(Dictionary<string, GcnExecute> t, string name, GcnVectorOp op)
    {
        t[name] = (m, w, i) => m.RunVector(w, i, op);
        if (op.D < GcnType.U64 && op.S.All(s => s < GcnType.U64))
        {
            LaneForms.Add(name);
        }
    }

    private static void Op(Dictionary<string, GcnExecute> t, string name, GcnType d, GcnType[] s, Func<ulong, ulong, ulong, ulong> f, Func<ulong, ulong, ulong, ulong>? saturate = null) =>
        Op(t, name, new GcnVectorOp { D = d, S = s, F = f, Saturate = saturate });

    private static void F32(Dictionary<string, GcnExecute> t, string name, Func<float, float> f) => Op(t, name, GcnType.F32, F1, (a, _, _) => B(f(F(a))));

    private static void F32(Dictionary<string, GcnExecute> t, string name, Func<float, float, float> f) => Op(t, name, GcnType.F32, F2, (a, b, _) => B(f(F(a), F(b))));

    private static void F32(Dictionary<string, GcnExecute> t, string name, Func<float, float, float, float> f) => Op(t, name, GcnType.F32, F3, (a, b, c) => B(f(F(a), F(b), F(c))));

    private static void F16(Dictionary<string, GcnExecute> t, string name, Func<Half, Half> f) => Op(t, name, GcnType.F16, H1, (a, _, _) => B(f(H(a))));

    private static void F16(Dictionary<string, GcnExecute> t, string name, Func<Half, Half, Half> f) => Op(t, name, GcnType.F16, H2, (a, b, _) => B(f(H(a), H(b))));

    private static void F16(Dictionary<string, GcnExecute> t, string name, Func<Half, Half, Half, Half> f) => Op(t, name, GcnType.F16, H3, (a, b, c) => B(f(H(a), H(b), H(c))));

    private static void F64(Dictionary<string, GcnExecute> t, string name, Func<double, double> f) => Op(t, name, GcnType.F64, D1, (a, _, _) => B(f(D(a))));

    private static void F64(Dictionary<string, GcnExecute> t, string name, Func<double, double, double> f) => Op(t, name, GcnType.F64, D2, (a, b, _) => B(f(D(a), D(b))));

    private static void F64(Dictionary<string, GcnExecute> t, string name, Func<double, double, double, double> f) => Op(t, name, GcnType.F64, D3, (a, b, c) => B(f(D(a), D(b), D(c))));

    private static void U32(Dictionary<string, GcnExecute> t, string name, Func<uint, uint> f) => Op(t, name, GcnType.U32, U1, (a, _, _) => f((uint)a));

    private static void U32(Dictionary<string, GcnExecute> t, string name, Func<uint, uint, uint> f, Func<uint, uint, uint>? saturate = null) =>
        Op(t, name, GcnType.U32, U2, (a, b, _) => f((uint)a, (uint)b), saturate is null ? null : (a, b, _) => saturate((uint)a, (uint)b));

    private static void U32(Dictionary<string, GcnExecute> t, string name, Func<uint, uint, uint, uint> f) => Op(t, name, GcnType.U32, U3, (a, b, c) => f((uint)a, (uint)b, (uint)c));

    private static void I32(Dictionary<string, GcnExecute> t, string name, Func<int, int, int> f) => Op(t, name, GcnType.I32, I2, (a, b, _) => (uint)f((int)a, (int)b));

    private static void I32(Dictionary<string, GcnExecute> t, string name, Func<int, int, int, int> f) => Op(t, name, GcnType.I32, I3, (a, b, c) => (uint)f((int)a, (int)b, (int)c));

    private static void Int16(Dictionary<string, GcnExecute> t, string name, bool signed, int sources, Func<int, int, int, int> f)
    {
        var type = signed ? GcnType.I16 : GcnType.U16;
        GcnType[] s = sources == 1 ? [type] : sources == 2 ? [type, type] : [type, type, type];
        Op(t, name, type, s, signed
            ? (a, b, c) => (ushort)f((short)a, (short)b, (short)c)
            : (a, b, c) => (ushort)f((ushort)a, (ushort)b, (ushort)c));
    }

    private static uint Sext24(uint v) => (uint)((int)(v << 8) >> 8);

    private static uint Perm(uint s0, uint s1, uint selector)
    {
        ulong both = (ulong)s0 << 32 | s1;
        uint d = 0;
        for (int k = 0; k < 4; k++)
        {
            uint sel = selector >> (8 * k) & 0xFF;
            uint b = sel switch
            {
                <= 7 => (uint)(both >> (int)(8 * sel)) & 0xFF,
                8 => (s1 >> 15 & 1) * 0xFF,
                9 => (s1 >> 31 & 1) * 0xFF,
                10 => (s0 >> 15 & 1) * 0xFF,
                11 => (s0 >> 31 & 1) * 0xFF,
                12 => 0,
                _ => 0xFF,
            };
            d |= b << (8 * k);
        }

        return d;
    }

    private static uint FirstFromTop(uint v) => v == 0 ? uint.MaxValue : (uint)BitOperations.LeadingZeroCount(v);

    private static void AddVector(Dictionary<string, GcnExecute> t)
    {
        // Moves, lanes and the rest that is not arithmetic.
        Op(t, "v_mov_b32", GcnType.B32, [GcnType.B32], (a, _, _) => a);
        t["v_nop"] = static (_, _, _) => { };
        t["v_clrexcp"] = static (_, _, _) => { };
        t["v_readfirstlane_b32"] = GcnMachine.ReadFirstLane;
        t["v_readlane_b32"] = GcnMachine.ReadLane;
        t["v_writelane_b32"] = GcnMachine.WriteLane;
        t["v_swap_b32"] = GcnMachine.Swap;
        t["v_cndmask_b32"] = GcnMachine.Cndmask;
        LaneForms.Add("v_cndmask_b32");
        t["v_accvgpr_read_b32"] = GcnMachine.AccRead;
        t["v_accvgpr_write_b32"] = GcnMachine.AccWrite;
        Op(t, "v_mbcnt_lo_u32_b32", new GcnVectorOp { D = GcnType.U32, S = U2, Lane = (a, b, _, lane) => (uint)(BitOperations.PopCount(a & (uint)((1UL << lane) - 1)) + (uint)b) });
        Op(t, "v_mbcnt_hi_u32_b32", new GcnVectorOp { D = GcnType.U32, S = U2, Lane = (a, b, _, lane) => (uint)(BitOperations.PopCount(a & (uint)(((1UL << lane) - 1) >> 32)) + (uint)b) });

        // Float32.
        F32(t, "v_add_f32", (a, b) => a + b);
        F32(t, "v_sub_f32", (a, b) => a - b);
        F32(t, "v_subrev_f32", (a, b) => b - a);
        F32(t, "v_mul_f32", (a, b) => a * b);
        F32(t, "v_mul_legacy_f32", MulLegacy);
        F32(t, "v_min_f32", GcnFloat.Min);
        F32(t, "v_max_f32", GcnFloat.Max);
        F32(t, "v_min3_f32", (a, b, c) => GcnFloat.Min(GcnFloat.Min(a, b), c));
        F32(t, "v_max3_f32", (a, b, c) => GcnFloat.Max(GcnFloat.Max(a, b), c));
        F32(t, "v_med3_f32", Med3);
        F32(t, "v_fma_f32", MathF.FusedMultiplyAdd);
        Op(t, "v_mad_f32", new GcnVectorOp { D = GcnType.F32, S = F3, F = (a, b, c) => B(Mad(F(a), F(b), F(c))), Mad = true });
        Op(t, "v_mad_legacy_f32", new GcnVectorOp { D = GcnType.F32, S = F3, F = (a, b, c) => B(Flush(Flush(MulLegacy(Flush(F(a)), Flush(F(b)))) + Flush(F(c)))), Mad = true });
        Op(t, "v_mac_f32", new GcnVectorOp { D = GcnType.F32, S = F3, F = (a, b, c) => B(Mad(F(a), F(b), F(c))), Mad = true, Fields = [0, 1, 3] });
        Op(t, "v_fmac_f32", new GcnVectorOp { D = GcnType.F32, S = F3, F = (a, b, c) => B(MathF.FusedMultiplyAdd(F(a), F(b), F(c))), Fields = [0, 1, 3] });
        Op(t, "v_madmk_f32", new GcnVectorOp { D = GcnType.F32, S = F3, F = (a, b, c) => B(Mad(F(a), F(b), F(c))), Mad = true, Fields = [0, 4, 1] });
        Op(t, "v_madak_f32", new GcnVectorOp { D = GcnType.F32, S = F3, F = (a, b, c) => B(Mad(F(a), F(b), F(c))), Mad = true, Fields = [0, 1, 4] });
        F32(t, "v_fract_f32", Fract);
        F32(t, "v_trunc_f32", MathF.Truncate);
        F32(t, "v_ceil_f32", MathF.Ceiling);
        F32(t, "v_floor_f32", MathF.Floor);
        F32(t, "v_rndne_f32", x => MathF.Round(x, MidpointRounding.ToEven));
        F32(t, "v_exp_f32", Exp);
        F32(t, "v_log_f32", Log);
        F32(t, "v_rcp_f32", Rcp);
        F32(t, "v_rcp_iflag_f32", Rcp);
        F32(t, "v_rsq_f32", Rsq);
        F32(t, "v_sqrt_f32", MathF.Sqrt);
        F32(t, "v_sin_f32", Sin);
        F32(t, "v_cos_f32", Cos);
        F32(t, "v_frexp_mant_f32", FrexpMant);
        Op(t, "v_frexp_exp_i32_f32", GcnType.I32, F1, (a, _, _) => (uint)FrexpExp(F(a)));
        Op(t, "v_ldexp_f32", GcnType.F32, [GcnType.F32, GcnType.I32], (a, b, _) => B(Ldexp(F(a), (int)b)));
        F32(t, "v_div_fixup_f32", DivFixup);
        t["v_div_scale_f32"] = GcnMachine.DivScale;
        t["v_div_fmas_f32"] = GcnMachine.DivFmas;
        t["v_div_scale_f64"] = GcnMachine.DivScale;
        t["v_div_fmas_f64"] = GcnMachine.DivFmas;

        // Conversions.
        Op(t, "v_cvt_f32_i32", GcnType.F32, [GcnType.I32], (a, _, _) => B((float)(int)a));
        Op(t, "v_cvt_f32_u32", GcnType.F32, U1, (a, _, _) => B((float)(uint)a));
        Op(t, "v_cvt_i32_f32", GcnType.I32, F1, (a, _, _) => (uint)ToInt(F(a)));
        Op(t, "v_cvt_u32_f32", GcnType.U32, F1, (a, _, _) => ToUInt(F(a)));
        Op(t, "v_cvt_rpi_i32_f32", GcnType.I32, F1, (a, _, _) => (uint)ToInt(Math.Floor(F(a) + 0.5)));
        Op(t, "v_cvt_flr_i32_f32", GcnType.I32, F1, (a, _, _) => (uint)ToInt(Math.Floor(F(a))));
        Op(t, "v_cvt_f16_f32", GcnType.F16, F1, (a, _, _) => B((Half)F(a)));
        Op(t, "v_cvt_f32_f16", GcnType.F32, H1, (a, _, _) => B((float)H(a)));
        for (int k = 0; k < 4; k++)
        {
            int shift = 8 * k;
            Op(t, $"v_cvt_f32_ubyte{k}", GcnType.F32, U1, (a, _, _) => B((float)((uint)a >> shift & 0xFF)));
        }

        Op(t, "v_cvt_f16_u16", GcnType.F16, [GcnType.U16], (a, _, _) => B((Half)(ushort)a));
        Op(t, "v_cvt_f16_i16", GcnType.F16, [GcnType.I16], (a, _, _) => B((Half)(short)a));
        Op(t, "v_cvt_u16_f16", GcnType.U16, H1, (a, _, _) => ToUShort((double)H(a)));
        Op(t, "v_cvt_i16_f16", GcnType.I16, H1, (a, _, _) => (ushort)ToShort((double)H(a)));
        Op(t, "v_cvt_pkrtz_f16_f32", GcnType.B32, F2, (a, b, _) => B(ToHalfTowardZero(F(a))) | B(ToHalfTowardZero(F(b))) << 16);
        Op(t, "v_cvt_pknorm_i16_f32", GcnType.B32, F2, (a, b, _) => (ushort)ToShort(Math.Round(Math.Clamp(F(a), -1f, 1f) * 32767.0, MidpointRounding.ToEven))
                                                                   | (ulong)(ushort)ToShort(Math.Round(Math.Clamp(F(b), -1f, 1f) * 32767.0, MidpointRounding.ToEven)) << 16);
        Op(t, "v_cvt_pknorm_u16_f32", GcnType.B32, F2, (a, b, _) => ToUShort(Math.Round(Math.Clamp(F(a), 0f, 1f) * 65535.0, MidpointRounding.ToEven))
                                                                   | (ulong)ToUShort(Math.Round(Math.Clamp(F(b), 0f, 1f) * 65535.0, MidpointRounding.ToEven)) << 16);
        U32(t, "v_cvt_pk_u16_u32", (a, b) => Math.Min(a, 0xFFFFu) | Math.Min(b, 0xFFFFu) << 16);
        I32(t, "v_cvt_pk_i16_i32", (a, b) => (int)((ushort)Math.Clamp(a, short.MinValue, short.MaxValue) | (uint)(ushort)Math.Clamp(b, short.MinValue, short.MaxValue) << 16));
        Op(t, "v_cvt_pk_u8_f32", GcnType.U32, [GcnType.F32, GcnType.U32, GcnType.U32], (a, b, c) =>
        {
            int at = 8 * ((int)b & 3);
            uint value = ToUInt(Math.Round(F(a), MidpointRounding.ToEven));
            return ((uint)c & ~(0xFFu << at)) | Math.Min(value, 255u) << at;
        });
        Op(t, "v_pack_b32_f16", GcnType.B32, H2, (a, b, _) => (a & 0xFFFF) | (b & 0xFFFF) << 16);
        Op(t, "v_cvt_f32_f64", GcnType.F32, D1, (a, _, _) => B((float)D(a)));
        Op(t, "v_cvt_f64_f32", GcnType.F64, F1, (a, _, _) => B((double)F(a)));
        Op(t, "v_cvt_i32_f64", GcnType.I32, D1, (a, _, _) => (uint)ToInt(D(a)));
        Op(t, "v_cvt_u32_f64", GcnType.U32, D1, (a, _, _) => ToUInt(D(a)));
        Op(t, "v_cvt_f64_i32", GcnType.F64, [GcnType.I32], (a, _, _) => B((double)(int)a));
        Op(t, "v_cvt_f64_u32", GcnType.F64, U1, (a, _, _) => B((double)(uint)a));

        // Integers.
        U32(t, "v_not_b32", a => ~a);
        U32(t, "v_bfrev_b32", ReverseBits32);
        U32(t, "v_ffbh_u32", FirstFromTop);
        U32(t, "v_ffbl_b32", a => a == 0 ? uint.MaxValue : (uint)BitOperations.TrailingZeroCount(a));
        U32(t, "v_ffbh_i32", a => (int)a < 0 ? FirstFromTop(~a) : FirstFromTop(a));
        U32(t, "v_add_u32", (a, b) => a + b, (a, b) => (uint)Math.Min((ulong)a + b, uint.MaxValue));
        U32(t, "v_sub_u32", (a, b) => a - b, (a, b) => b > a ? 0 : a - b);
        U32(t, "v_subrev_u32", (a, b) => b - a, (a, b) => a > b ? 0 : b - a);
        Op(t, "v_add_i32", GcnType.I32, I2, (a, b, _) => (uint)((int)a + (int)b), (a, b, _) => (uint)(int)Math.Clamp((long)(int)a + (int)b, int.MinValue, int.MaxValue));
        Op(t, "v_sub_i32", GcnType.I32, I2, (a, b, _) => (uint)((int)a - (int)b), (a, b, _) => (uint)(int)Math.Clamp((long)(int)a - (int)b, int.MinValue, int.MaxValue));
        U32(t, "v_mul_lo_u32", (a, b) => a * b);
        U32(t, "v_mul_hi_u32", (a, b) => (uint)((ulong)a * b >> 32));
        I32(t, "v_mul_hi_i32", (a, b) => (int)((long)a * b >> 32));
        U32(t, "v_mul_u32_u24", (a, b) => (a & 0xFFFFFF) * (b & 0xFFFFFF));
        U32(t, "v_mul_hi_u32_u24", (a, b) => (uint)((ulong)(a & 0xFFFFFF) * (b & 0xFFFFFF) >> 32));
        U32(t, "v_mul_i32_i24", (a, b) => (uint)((int)Sext24(a) * (int)Sext24(b)));
        U32(t, "v_mul_hi_i32_i24", (a, b) => (uint)((long)(int)Sext24(a) * (int)Sext24(b) >> 32));
        U32(t, "v_mad_u32_u24", (a, b, c) => (a & 0xFFFFFF) * (b & 0xFFFFFF) + c);
        U32(t, "v_mad_i32_i24", (a, b, c) => (uint)((int)Sext24(a) * (int)Sext24(b)) + c);
        Op(t, "v_mad_u32_u16", GcnType.U32, [GcnType.U16, GcnType.U16, GcnType.U32], (a, b, c) => (uint)((uint)a * (uint)b + (uint)c));
        Op(t, "v_mad_i32_i16", GcnType.I32, [GcnType.I16, GcnType.I16, GcnType.I32], (a, b, c) => (uint)((short)a * (short)b + (int)c));
        I32(t, "v_min_i32", Math.Min);
        I32(t, "v_max_i32", Math.Max);
        U32(t, "v_min_u32", Math.Min);
        U32(t, "v_max_u32", Math.Max);
        I32(t, "v_min3_i32", (a, b, c) => Math.Min(Math.Min(a, b), c));
        I32(t, "v_max3_i32", (a, b, c) => Math.Max(Math.Max(a, b), c));
        I32(t, "v_med3_i32", (a, b, c) => Math.Max(Math.Min(a, b), Math.Min(Math.Max(a, b), c)));
        U32(t, "v_min3_u32", (a, b, c) => Math.Min(Math.Min(a, b), c));
        U32(t, "v_max3_u32", (a, b, c) => Math.Max(Math.Max(a, b), c));
        U32(t, "v_med3_u32", (a, b, c) => Math.Max(Math.Min(a, b), Math.Min(Math.Max(a, b), c)));
        U32(t, "v_lshrrev_b32", (a, b) => b >> (int)(a & 31));
        U32(t, "v_ashrrev_i32", (a, b) => (uint)((int)b >> (int)(a & 31)));
        U32(t, "v_lshlrev_b32", (a, b) => b << (int)(a & 31));
        U32(t, "v_and_b32", (a, b) => a & b);
        U32(t, "v_or_b32", (a, b) => a | b);
        U32(t, "v_xor_b32", (a, b) => a ^ b);
        U32(t, "v_xnor_b32", (a, b) => ~(a ^ b));
        U32(t, "v_bfe_u32", (a, b, c) => BitField(a, (int)(b & 31), (int)(c & 31), false));
        U32(t, "v_bfe_i32", (a, b, c) => BitField(a, (int)(b & 31), (int)(c & 31), true));
        U32(t, "v_bfi_b32", (a, b, c) => (a & b) | (~a & c));
        U32(t, "v_bfm_b32", (a, b) => (uint)((1UL << (int)(a & 31)) - 1) << (int)(b & 31));
        U32(t, "v_alignbit_b32", (a, b, c) => (uint)(((ulong)a << 32 | b) >> (int)(c & 31)));
        U32(t, "v_alignbyte_b32", (a, b, c) => (uint)(((ulong)a << 32 | b) >> (int)(8 * (c & 3))));
        U32(t, "v_perm_b32", Perm);
        U32(t, "v_lshl_add_u32", (a, b, c) => (a << (int)(b & 31)) + c);
        U32(t, "v_add_lshl_u32", (a, b, c) => (a + b) << (int)(c & 31));
        U32(t, "v_add3_u32", (a, b, c) => a + b + c);
        U32(t, "v_lshl_or_b32", (a, b, c) => (a << (int)(b & 31)) | c);
        U32(t, "v_and_or_b32", (a, b, c) => (a & b) | c);
        U32(t, "v_or3_b32", (a, b, c) => a | b | c);
        U32(t, "v_xad_u32", (a, b, c) => (a ^ b) + c);
        U32(t, "v_bcnt_u32_b32", (a, b) => (uint)BitOperations.PopCount(a) + b);
        U32(t, "v_sad_u32", (a, b, c) => (a > b ? a - b : b - a) + c);
        Op(t, "v_lshlrev_b64", GcnType.U64, [GcnType.U32, GcnType.U64], (a, b, _) => b << (int)(a & 63));
        Op(t, "v_lshrrev_b64", GcnType.U64, [GcnType.U32, GcnType.U64], (a, b, _) => b >> (int)(a & 63));
        Op(t, "v_ashrrev_i64", GcnType.I64, [GcnType.U32, GcnType.I64], (a, b, _) => (ulong)((long)b >> (int)(a & 63)));
        t["v_add_co_u32"] = (m, w, i) => m.Carry(w, i, subtract: false, reverse: false, carryIn: false);
        t["v_sub_co_u32"] = (m, w, i) => m.Carry(w, i, subtract: true, reverse: false, carryIn: false);
        t["v_subrev_co_u32"] = (m, w, i) => m.Carry(w, i, subtract: true, reverse: true, carryIn: false);
        t["v_addc_co_u32"] = (m, w, i) => m.Carry(w, i, subtract: false, reverse: false, carryIn: true);
        t["v_subb_co_u32"] = (m, w, i) => m.Carry(w, i, subtract: true, reverse: false, carryIn: true);
        t["v_subbrev_co_u32"] = (m, w, i) => m.Carry(w, i, subtract: true, reverse: true, carryIn: true);
        foreach (string name in (string[])["v_add_co_u32", "v_sub_co_u32", "v_subrev_co_u32", "v_addc_co_u32", "v_subb_co_u32", "v_subbrev_co_u32"])
        {
            LaneForms.Add(name);
        }

        t["v_mad_u64_u32"] = (m, w, i) => m.Mad64(w, i, signed: false);
        t["v_mad_i64_i32"] = (m, w, i) => m.Mad64(w, i, signed: true);

        // 16-bit integers.
        Int16(t, "v_add_u16", false, 2, (a, b, _) => a + b);
        Int16(t, "v_sub_u16", false, 2, (a, b, _) => a - b);
        Int16(t, "v_subrev_u16", false, 2, (a, b, _) => b - a);
        Int16(t, "v_add_i16", true, 2, (a, b, _) => a + b);
        Int16(t, "v_sub_i16", true, 2, (a, b, _) => a - b);
        Int16(t, "v_mul_lo_u16", false, 2, (a, b, _) => a * b);
        Int16(t, "v_lshlrev_b16", false, 2, (a, b, _) => b << (a & 15));
        Int16(t, "v_lshrrev_b16", false, 2, (a, b, _) => b >> (a & 15));
        Int16(t, "v_ashrrev_i16", true, 2, (a, b, _) => b >> (a & 15));
        Int16(t, "v_max_u16", false, 2, (a, b, _) => Math.Max(a, b));
        Int16(t, "v_min_u16", false, 2, (a, b, _) => Math.Min(a, b));
        Int16(t, "v_max_i16", true, 2, (a, b, _) => Math.Max(a, b));
        Int16(t, "v_min_i16", true, 2, (a, b, _) => Math.Min(a, b));
        Int16(t, "v_mad_u16", false, 3, (a, b, c) => a * b + c);
        Int16(t, "v_mad_i16", true, 3, (a, b, c) => a * b + c);
        Int16(t, "v_max3_u16", false, 3, (a, b, c) => Math.Max(Math.Max(a, b), c));
        Int16(t, "v_min3_u16", false, 3, (a, b, c) => Math.Min(Math.Min(a, b), c));
        Int16(t, "v_max3_i16", true, 3, (a, b, c) => Math.Max(Math.Max(a, b), c));
        Int16(t, "v_min3_i16", true, 3, (a, b, c) => Math.Min(Math.Min(a, b), c));

        // Float16 (computed in float32 and rounded once to float16, exact for one add, multiply or divide).
        F16(t, "v_add_f16", (a, b) => a + b);
        F16(t, "v_sub_f16", (a, b) => a - b);
        F16(t, "v_subrev_f16", (a, b) => b - a);
        F16(t, "v_mul_f16", (a, b) => a * b);
        F16(t, "v_max_f16", (a, b) => (Half)GcnFloat.Max((float)a, (float)b));
        F16(t, "v_min_f16", (a, b) => (Half)GcnFloat.Min((float)a, (float)b));
        F16(t, "v_max3_f16", (a, b, c) => (Half)GcnFloat.Max(GcnFloat.Max((float)a, (float)b), (float)c));
        F16(t, "v_min3_f16", (a, b, c) => (Half)GcnFloat.Min(GcnFloat.Min((float)a, (float)b), (float)c));
        F16(t, "v_med3_f16", (a, b, c) => (Half)Med3((float)a, (float)b, (float)c));
        F16(t, "v_fma_f16", (a, b, c) => (Half)Math.FusedMultiplyAdd((double)a, (double)b, (double)c));
        F16(t, "v_mad_f16", (a, b, c) => (Half)((float)(Half)((float)a * (float)b) + (float)c));
        Op(t, "v_mac_f16", new GcnVectorOp { D = GcnType.F16, S = H3, F = (a, b, c) => B((Half)((float)(Half)((float)H(a) * (float)H(b)) + (float)H(c))), Fields = [0, 1, 3] });
        Op(t, "v_ldexp_f16", GcnType.F16, [GcnType.F16, GcnType.I16], (a, b, _) => B((Half)Ldexp((float)H(a), (short)b)));
        F16(t, "v_div_fixup_f16", (a, b, c) => (Half)DivFixup((float)a, (float)b, (float)c));
        F16(t, "v_rcp_f16", a => (Half)Rcp((float)a));
        F16(t, "v_rsq_f16", a => (Half)Rsq((float)a));
        F16(t, "v_sqrt_f16", a => (Half)MathF.Sqrt((float)a));
        F16(t, "v_log_f16", a => (Half)Log((float)a));
        F16(t, "v_exp_f16", a => (Half)Exp((float)a));
        F16(t, "v_sin_f16", a => (Half)Sin((float)a));
        F16(t, "v_cos_f16", a => (Half)Cos((float)a));
        F16(t, "v_floor_f16", Half.Floor);
        F16(t, "v_ceil_f16", Half.Ceiling);
        F16(t, "v_trunc_f16", Half.Truncate);
        F16(t, "v_rndne_f16", a => Half.Round(a, MidpointRounding.ToEven));
        F16(t, "v_fract_f16", a => (Half)Fract((float)a));
        F16(t, "v_frexp_mant_f16", a => (Half)FrexpMant((float)a));
        Op(t, "v_frexp_exp_i16_f16", GcnType.I16, H1, (a, _, _) => (ushort)FrexpExp((float)H(a)));

        // Float64.
        F64(t, "v_add_f64", (a, b) => a + b);
        F64(t, "v_mul_f64", (a, b) => a * b);
        F64(t, "v_min_f64", GcnFloat.Min);
        F64(t, "v_max_f64", GcnFloat.Max);
        F64(t, "v_fma_f64", Math.FusedMultiplyAdd);
        F64(t, "v_div_fixup_f64", DivFixup);
        F64(t, "v_rcp_f64", a => 1.0 / a);
        F64(t, "v_rsq_f64", a => 1.0 / Math.Sqrt(a));
        F64(t, "v_sqrt_f64", Math.Sqrt);
        F64(t, "v_trunc_f64", Math.Truncate);
        F64(t, "v_ceil_f64", Math.Ceiling);
        F64(t, "v_floor_f64", Math.Floor);
        F64(t, "v_rndne_f64", a => Math.Round(a, MidpointRounding.ToEven));
        F64(t, "v_fract_f64", Fract);
        F64(t, "v_frexp_mant_f64", FrexpMant);
        Op(t, "v_frexp_exp_i32_f64", GcnType.I32, D1, (a, _, _) => (uint)FrexpExp(D(a)));
        Op(t, "v_ldexp_f64", GcnType.F64, [GcnType.F64, GcnType.I32], (a, b, _) => B(Math.ScaleB(D(a), Math.Clamp((int)b, -3000, 3000))));

        AddCompares(t);
        AddPacked(t);
    }

    private static void AddCompares(Dictionary<string, GcnExecute> t)
    {
        (string, Func<double, double, bool>)[] floats =
        [
            ("f", (_, _) => false), ("lt", (a, b) => a < b), ("eq", (a, b) => a == b), ("le", (a, b) => a <= b), ("gt", (a, b) => a > b), ("lg", (a, b) => a < b || a > b),
            ("ge", (a, b) => a >= b), ("o", (a, b) => !double.IsNaN(a) && !double.IsNaN(b)), ("u", (a, b) => double.IsNaN(a) || double.IsNaN(b)), ("nge", (a, b) => !(a >= b)),
            ("nlg", (a, b) => !(a < b || a > b)), ("ngt", (a, b) => !(a > b)), ("nle", (a, b) => !(a <= b)), ("neq", (a, b) => !(a == b)), ("nlt", (a, b) => !(a < b)),
            ("tru", (_, _) => true),
        ];
        (string, Func<long, long, bool>)[] integers =
        [
            ("f", (_, _) => false), ("lt", (a, b) => a < b), ("eq", (a, b) => a == b), ("le", (a, b) => a <= b), ("gt", (a, b) => a > b), ("ne", (a, b) => a != b),
            ("ge", (a, b) => a >= b), ("t", (_, _) => true),
        ];
        foreach (string x in (string[])["", "x"])
        {
            bool writesExec = x == "x";
            foreach (var (cond, f) in floats)
            {
                var test = f;
                Compare(t, $"v_cmp{x}_{cond}_f32", GcnType.F32, (a, b) => test(F(a), F(b)), writesExec);
                Compare(t, $"v_cmp{x}_{cond}_f16", GcnType.F16, (a, b) => test((double)H(a), (double)H(b)), writesExec);
                Compare(t, $"v_cmp{x}_{cond}_f64", GcnType.F64, (a, b) => test(D(a), D(b)), writesExec);
            }

            foreach (var (cond, f) in integers)
            {
                var test = f;
                Compare(t, $"v_cmp{x}_{cond}_i32", GcnType.I32, (a, b) => test((int)a, (int)b), writesExec);
                Compare(t, $"v_cmp{x}_{cond}_u32", GcnType.U32, (a, b) => test((uint)a, (uint)b), writesExec);
                Compare(t, $"v_cmp{x}_{cond}_i16", GcnType.I16, (a, b) => test((short)a, (short)b), writesExec);
                Compare(t, $"v_cmp{x}_{cond}_u16", GcnType.U16, (a, b) => test((ushort)a, (ushort)b), writesExec);
                Compare(t, $"v_cmp{x}_{cond}_i64", GcnType.I64, (a, b) => test((long)a, (long)b), writesExec);
                Compare(t, $"v_cmp{x}_{cond}_u64", GcnType.U64, (a, b) => cond switch
                {
                    "lt" => a < b,
                    "le" => a <= b,
                    "gt" => a > b,
                    "ge" => a >= b,
                    _ => test((long)a, (long)b),
                }, writesExec);
            }

            Compare(t, $"v_cmp{x}_class_f32", GcnType.F32, (a, b) => (b >> ClassOf32((uint)a) & 1) != 0, writesExec, classMask: true);
            Compare(t, $"v_cmp{x}_class_f16", GcnType.F16, (a, b) => (b >> ClassOf16((ushort)a) & 1) != 0, writesExec, classMask: true);
            Compare(t, $"v_cmp{x}_class_f64", GcnType.F64, (a, b) => (b >> ClassOf64(a) & 1) != 0, writesExec, classMask: true);
        }
    }

    private static int ClassOf32(uint bits)
    {
        float x = BitConverter.UInt32BitsToSingle(bits);
        int c = GcnFloat.Class(x, (bits & 0x00400000) == 0);
        return float.IsSubnormal(x) ? (float.IsNegative(x) ? 4 : 7) : c;
    }

    private static int ClassOf16(ushort bits)
    {
        var x = BitConverter.UInt16BitsToHalf(bits);
        int c = GcnFloat.Class((double)x, (bits & 0x0200) == 0);
        return Half.IsSubnormal(x) ? (Half.IsNegative(x) ? 4 : 7) : c;
    }

    private static int ClassOf64(ulong bits)
    {
        double x = BitConverter.UInt64BitsToDouble(bits);
        int c = GcnFloat.Class(x, (bits & 0x0008000000000000) == 0);
        return double.IsSubnormal(x) ? (double.IsNegative(x) ? 4 : 7) : c;
    }

    private static void Compare(Dictionary<string, GcnExecute> t, string name, GcnType type, Func<ulong, ulong, bool> f, bool writesExec, bool classMask = false)
    {
        t[name] = (m, w, i) => m.Compare(w, i, type, classMask ? GcnType.U32 : type, f, writesExec);
        if (type < GcnType.U64)
        {
            LaneForms.Add(name);
        }
    }

    private static void AddPacked(Dictionary<string, GcnExecute> t)
    {
        // VOP3P: two 16-bit halves, each from the half op_sel (low result) or op_sel_hi (high result) picks.
        Packed(t, "v_pk_add_f16", true, (a, b, _) => a + b);
        Packed(t, "v_pk_mul_f16", true, (a, b, _) => a * b);
        Packed(t, "v_pk_fma_f16", true, (a, b, c) => (float)(Half)Math.FusedMultiplyAdd((double)a, (double)b, (double)c));
        Packed(t, "v_pk_max_f16", true, (a, b, _) => GcnFloat.Max(a, b));
        Packed(t, "v_pk_min_f16", true, (a, b, _) => GcnFloat.Min(a, b));
        PackedInt(t, "v_pk_add_u16", false, (a, b, _) => a + b);
        PackedInt(t, "v_pk_sub_u16", false, (a, b, _) => a - b);
        PackedInt(t, "v_pk_add_i16", true, (a, b, _) => a + b);
        PackedInt(t, "v_pk_sub_i16", true, (a, b, _) => a - b);
        PackedInt(t, "v_pk_mul_lo_u16", false, (a, b, _) => a * b);
        PackedInt(t, "v_pk_mad_u16", false, (a, b, c) => a * b + c);
        PackedInt(t, "v_pk_mad_i16", true, (a, b, c) => a * b + c);
        PackedInt(t, "v_pk_lshlrev_b16", false, (a, b, _) => b << (a & 15));
        PackedInt(t, "v_pk_lshrrev_b16", false, (a, b, _) => b >> (a & 15));
        PackedInt(t, "v_pk_ashrrev_i16", true, (a, b, _) => b >> (a & 15));
        PackedInt(t, "v_pk_max_u16", false, (a, b, _) => Math.Max(a, b));
        PackedInt(t, "v_pk_min_u16", false, (a, b, _) => Math.Min(a, b));
        PackedInt(t, "v_pk_max_i16", true, (a, b, _) => Math.Max(a, b));
        PackedInt(t, "v_pk_min_i16", true, (a, b, _) => Math.Min(a, b));
        foreach (string name in (string[])["v_mad_mix_f32", "v_mad_mixlo_f16", "v_mad_mixhi_f16", "v_fma_mix_f32", "v_fma_mixlo_f16", "v_fma_mixhi_f16"])
        {
            bool fused = name.StartsWith("v_fma", StringComparison.Ordinal);
            int part = name.EndsWith("lo_f16", StringComparison.Ordinal) ? 1 : name.EndsWith("hi_f16", StringComparison.Ordinal) ? 2 : 0;
            t[name] = (m, w, i) => m.Mix(w, i, fused, part);
        }

        t["v_dot2_f32_f16"] = (m, w, i) => m.Dot(w, i, (a, b, c) => B((float)((double)H(a) * (double)H(b) + (double)H(a >> 16) * (double)H(b >> 16) + F(c))));
        t["v_dot2_i32_i16"] = (m, w, i) => m.Dot(w, i, (a, b, c) => (uint)((short)a * (short)b + (short)(a >> 16) * (short)(b >> 16) + (int)c));
        t["v_dot2_u32_u16"] = (m, w, i) => m.Dot(w, i, (a, b, c) => (uint)((ushort)a * (ushort)b + (ushort)(a >> 16) * (ushort)(b >> 16) + (uint)c));
        t["v_dot4_i32_i8"] = (m, w, i) => m.Dot(w, i, (a, b, c) => Dot((uint)a, (uint)b, (uint)c, 8, true));
        t["v_dot4_u32_u8"] = (m, w, i) => m.Dot(w, i, (a, b, c) => Dot((uint)a, (uint)b, (uint)c, 8, false));
        t["v_dot8_i32_i4"] = (m, w, i) => m.Dot(w, i, (a, b, c) => Dot((uint)a, (uint)b, (uint)c, 4, true));
        t["v_dot8_u32_u4"] = (m, w, i) => m.Dot(w, i, (a, b, c) => Dot((uint)a, (uint)b, (uint)c, 4, false));
    }

    private static ulong Dot(uint a, uint b, uint c, int bits, bool signed)
    {
        int sum = (int)c;
        int shift = 32 - bits;
        for (int k = 0; k < 32 / bits; k++)
        {
            int x = signed ? (int)(a << (shift - k * bits)) >> shift : (int)(a >> (k * bits) & ((1u << bits) - 1));
            int y = signed ? (int)(b << (shift - k * bits)) >> shift : (int)(b >> (k * bits) & ((1u << bits) - 1));
            sum += x * y;
        }

        return (uint)sum;
    }

    private static void Packed(Dictionary<string, GcnExecute> t, string name, bool isFloat, Func<float, float, float, float> f) =>
        t[name] = (m, w, i) => m.Packed(w, i, isFloat, (a, b, c) => B((Half)f((float)H(a), (float)H(b), (float)H(c))));

    private static void PackedInt(Dictionary<string, GcnExecute> t, string name, bool signed, Func<int, int, int, int> f) =>
        t[name] = (m, w, i) => m.Packed(w, i, false, signed ? (a, b, c) => (ushort)f((short)a, (short)b, (short)c) : (a, b, c) => (ushort)f((ushort)a, (ushort)b, (ushort)c));
}

internal sealed partial class GcnMachine
{
    private readonly ulong[] a0 = new ulong[64], a1 = new ulong[64], a2 = new ulong[64], result = new ulong[64], moved = new ulong[64];

    private static bool Wide(GcnType type) => type >= GcnType.U64;

    private static bool Narrow(GcnType type) => type is GcnType.U16 or GcnType.I16 or GcnType.F16;

    private static bool IsFloat(GcnType type) => type is GcnType.F32 or GcnType.F16 or GcnType.F64;

    private static ulong Sign(GcnType type) => type switch { GcnType.F16 => 0x8000, GcnType.F64 => 1UL << 63, _ => 0x80000000 };

    private bool Flushes32 => descriptor.FloatDenormMode32 != 3;

    /// <summary>An SDWA source selection: a byte or word, zero- or sign-extended, or the whole dword.</summary>
    private static uint Select(uint v, int sel, bool sext) => sel switch
    {
        <= 3 => sext ? (uint)(sbyte)(v >> (8 * sel)) : v >> (8 * sel) & 0xFF,
        4 or 5 => sext ? (uint)(short)(v >> (16 * (sel - 4))) : v >> (16 * (sel - 4)) & 0xFFFF,
        _ => v,
    };

    /// <summary>Source <paramref name="k"/> of a vector instruction as <paramref name="type"/>: fetched, selected (SDWA, op_sel) and modified (abs, neg).</summary>
    private void Typed(GcnWave w, GcnInstruction i, int value, GcnType type, int k, ulong[] into, bool floatModifiers)
    {
        if (Wide(type))
        {
            Source64(w, value, i, L0, L1, type == GcnType.F64);
            for (int lane = 0; lane < 64; lane++)
            {
                into[lane] = L0[lane] | (ulong)L1[lane] << 32;
            }
        }
        else
        {
            Source32(w, value, i, L0, type == GcnType.F16);
            for (int lane = 0; lane < 64; lane++)
            {
                into[lane] = L0[lane];
            }
        }

        if (i.Variant == GcnVariant.Sdwa && k < 2)
        {
            int sel = k == 0 ? i.Src0Sel : i.Src1Sel;
            bool sext = (i.Sext >> k & 1) != 0;
            if (sel != 6 || sext)
            {
                for (int lane = 0; lane < 64; lane++)
                {
                    into[lane] = Select((uint)into[lane], sel, sext);
                }
            }
        }

        if (Narrow(type))
        {
            int shift = i.Opcode.Format == GcnFormat.Vop3 && k < 3 && (i.OpSel >> k & 1) != 0 ? 16 : 0;
            for (int lane = 0; lane < 64; lane++)
            {
                into[lane] = into[lane] >> shift & 0xFFFF;
            }
        }

        bool abs = (i.Abs >> k & 1) != 0, neg = (i.Neg >> k & 1) != 0;
        if (abs || neg)
        {
            if (!floatModifiers || !IsFloat(type))
            {
                throw Fail(GcnFailure.NotModelled, w, i, $"abs or neg on source {k}, which {i.Opcode.Base} reads as an integer");
            }

            ulong sign = Sign(type);
            for (int lane = 0; lane < 64; lane++)
            {
                ulong v = into[lane];
                if (abs)
                {
                    v &= ~sign;
                }

                if (neg)
                {
                    v ^= sign;
                }

                into[lane] = v;
            }
        }
    }

    private static int DppSource(int c, int lane) => c switch
    {
        <= 0xFF => (lane & ~3) | (c >> (2 * (lane & 3)) & 3),
        >= 0x101 and <= 0x10F => (lane & 15) + (c & 15) < 16 ? lane + (c & 15) : -1,
        >= 0x111 and <= 0x11F => (lane & 15) >= (c & 15) ? lane - (c & 15) : -1,
        >= 0x121 and <= 0x12F => (lane & ~15) | ((lane - (c & 15)) & 15),
        0x130 => lane < 63 ? lane + 1 : -1,
        0x134 => (lane + 1) & 63,
        0x138 => lane > 0 ? lane - 1 : -1,
        0x13C => (lane - 1) & 63,
        0x140 => (lane & ~15) | (15 - (lane & 15)),
        0x141 => (lane & ~7) | (7 - (lane & 7)),
        0x142 => lane >= 16 ? (lane & ~15) - 1 : -1,
        _ => lane >= 32 ? 31 : -1,
    };

    /// <summary>
    /// DPP: moves source 0 between lanes; returns the lanes written. A lane is not written where its row or bank is
    /// masked off; where its source lane is out of range or inactive, it gets 0 with bound_ctrl, else is not written.
    /// </summary>
    private ulong Dpp(GcnInstruction i, ulong exec, ulong[] source)
    {
        ulong write = 0;
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            if ((i.RowMask >> (lane >> 4) & 1) == 0 || (i.BankMask >> ((lane >> 2) & 3) & 1) == 0)
            {
                continue;
            }

            int from = DppSource(i.DppControl, lane);
            if (from < 0 || (exec >> from & 1) == 0)
            {
                if (i.BoundControl)
                {
                    moved[lane] = 0;
                    write |= 1UL << lane;
                }

                continue;
            }

            moved[lane] = source[from];
            write |= 1UL << lane;
        }

        for (ulong mask = write; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            source[lane] = moved[lane];
        }

        return write;
    }

    private int FieldValue(GcnInstruction i, int field) => field switch
    {
        0 => i.S0,
        1 => i.S1,
        2 => i.S2,
        3 => i.D,
        _ => 255,
    };

    private ulong[] Buffer(int k) => k == 0 ? a0 : k == 1 ? a1 : a2;

    /// <summary>A vector ALU instruction (VOP1, VOP2, VOP3, with DPP or SDWA): every active lane computed, output modifiers applied, written.</summary>
    internal void RunVector(GcnWave w, GcnInstruction i, GcnVectorOp op)
    {
        ulong exec = ExecOf(w, i);
        for (int k = 0; k < op.S.Length; k++)
        {
            Typed(w, i, FieldValue(i, op.Fields[k]), op.S[k], k, Buffer(k), true);
        }

        ulong active = i.Variant == GcnVariant.Dpp ? Dpp(i, exec, a0) : exec;
        bool flush = op.Mad || Flushes32;
        for (int k = 0; k < op.S.Length; k++)
        {
            if (op.S[k] == GcnType.F32 && flush)
            {
                var buffer = Buffer(k);
                for (int lane = 0; lane < 64; lane++)
                {
                    buffer[lane] = B(Flush(F(buffer[lane])));
                }
            }
        }

        var compute = op.F;
        if (i.Clamp && !IsFloat(op.D))
        {
            compute = op.Saturate ?? throw Fail(GcnFailure.NotModelled, w, i, $"clamp on {i.Opcode.Base}");
        }

        for (ulong mask = active; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            result[lane] = op.Lane is { } perLane ? perLane(a0[lane], a1[lane], a2[lane], lane) : compute!(a0[lane], a1[lane], a2[lane]);
        }

        if (IsFloat(op.D) && (i.Omod != 0 || i.Clamp || flush && op.D == GcnType.F32))
        {
            OutputModifiers(i, op.D, active, flush && op.D == GcnType.F32);
        }

        WriteResult(w, i, op.D, active);
    }

    private void OutputModifiers(GcnInstruction i, GcnType type, ulong active, bool flush)
    {
        double scale = i.Omod switch { 1 => 2, 2 => 4, 3 => 0.5, _ => 1 };
        for (ulong mask = active; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            ulong bits = result[lane];
            switch (type)
            {
                case GcnType.F32:
                    float x = F(bits);
                    if (scale != 1)
                    {
                        x = (float)(x * scale);
                    }

                    if (i.Clamp)
                    {
                        x = float.IsNaN(x) ? 0 : Math.Clamp(x, 0f, 1f);
                    }

                    result[lane] = B(flush ? Flush(x) : x);
                    break;
                case GcnType.F16:
                    float h = (float)H(bits);
                    if (scale != 1)
                    {
                        h = (float)(h * scale);
                    }

                    if (i.Clamp)
                    {
                        h = float.IsNaN(h) ? 0 : Math.Clamp(h, 0f, 1f);
                    }

                    result[lane] = B((Half)h);
                    break;
                default:
                    double d = D(bits) * scale;
                    if (i.Clamp)
                    {
                        d = double.IsNaN(d) ? 0 : Math.Clamp(d, 0, 1);
                    }

                    result[lane] = B(d);
                    break;
            }
        }
    }

    /// <summary>Writes the result lanes: 64-bit to a register pair; 16-bit to the low half (the high zeroed), or the high half with VOP3's op_sel; SDWA's dst_sel.</summary>
    private void WriteResult(GcnWave w, GcnInstruction i, GcnType type, ulong mask)
    {
        int r = i.D - 256;
        if (Wide(type))
        {
            for (int lane = 0; lane < 64; lane++)
            {
                R0[lane] = (uint)result[lane];
                R1[lane] = (uint)(result[lane] >> 32);
            }

            CheckVector(w, r, 2, i);
            WriteVector(w, r, i, R0, mask);
            WriteVector(w, r + 1, i, R1, mask);
            return;
        }

        for (int lane = 0; lane < 64; lane++)
        {
            R0[lane] = Narrow(type) ? (uint)result[lane] & 0xFFFF : (uint)result[lane];
        }

        if (i.Variant == GcnVariant.Sdwa && i.DstSel != 6)
        {
            ReadVector(w, r, i, R1);
            int sel = i.DstSel, size = sel <= 3 ? 8 : 16, shift = sel <= 3 ? 8 * sel : 16 * (sel - 4);
            uint field = (1u << size) - 1;
            for (int lane = 0; lane < 64; lane++)
            {
                uint value = (R0[lane] & field) << shift;
                R0[lane] = i.DstUnused switch
                {
                    0 => value,
                    1 => (R0[lane] >> (size - 1) & 1) != 0 ? value | (uint)((ulong)uint.MaxValue << (shift + size)) : value,
                    _ => (R1[lane] & ~(field << shift)) | value,
                };
            }
        }
        else if (Narrow(type) && i.Opcode.Format == GcnFormat.Vop3 && (i.OpSel & 8) != 0)
        {
            ReadVector(w, r, i, R1);
            for (int lane = 0; lane < 64; lane++)
            {
                R0[lane] = (R1[lane] & 0xFFFF) | R0[lane] << 16;
            }
        }

        WriteVector(w, r, i, R0, mask);
    }

    /// <summary>VOPC: a compare per active lane into VCC (VOPC), the SGPR pair (VOP3) or SDWA's destination; inactive lanes read 0. v_cmpx also writes EXEC.</summary>
    internal void Compare(GcnWave w, GcnInstruction i, GcnType type, GcnType second, Func<ulong, ulong, bool> f, bool writesExec)
    {
        ulong exec = ExecOf(w, i);
        Typed(w, i, i.S0, type, 0, a0, true);
        Typed(w, i, i.S1, second, 1, a1, true);
        ulong active = i.Variant == GcnVariant.Dpp ? Dpp(i, exec, a0) : exec;
        ulong bits = 0;
        for (ulong mask = active; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            if (f(a0[lane], a1[lane]))
            {
                bits |= 1UL << lane;
            }
        }

        int destination = i.Variant == GcnVariant.Sdwa ? i.SD : i.Opcode.Format == GcnFormat.Vop3 ? i.D : 106;
        WriteScalar64(w, destination, bits, i);
        if (writesExec)
        {
            w.Exec = bits;
        }
    }

    /// <summary>v_add_co_u32 and the rest: the carry (or borrow) per lane into VCC or SD; carry in from VCC or the third source.</summary>
    internal void Carry(GcnWave w, GcnInstruction i, bool subtract, bool reverse, bool carryIn)
    {
        ulong exec = ExecOf(w, i);
        Typed(w, i, i.S0, GcnType.U32, 0, a0, false);
        Typed(w, i, i.S1, GcnType.U32, 1, a1, false);
        bool vop3 = i.Opcode.Format == GcnFormat.Vop3;
        ulong carries = carryIn ? (vop3 ? Scalar64(w, i.S2, i) : ReadScalar64(w, 106, i)) : 0;
        ulong active = i.Variant == GcnVariant.Dpp ? Dpp(i, exec, a0) : exec;
        ulong output = 0;
        for (ulong mask = active; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            ulong a = reverse ? a1[lane] : a0[lane], b = reverse ? a0[lane] : a1[lane], c = carries >> lane & 1;
            bool carry;
            ulong value;
            if (subtract)
            {
                value = a - b - c;
                carry = b + c > a;
            }
            else
            {
                value = a + b + c;
                carry = value >> 32 != 0;
            }

            if (i.Clamp && carry)
            {
                value = subtract ? 0 : uint.MaxValue;
            }

            result[lane] = (uint)value;
            if (carry)
            {
                output |= 1UL << lane;
            }
        }

        WriteResult(w, i, GcnType.U32, active);
        WriteScalar64(w, vop3 ? i.SD : 106, output, i);
    }

    /// <summary>v_cndmask_b32: source 1 where the lane's bit of VCC (or the third source's pair) is set, else source 0.</summary>
    internal static void Cndmask(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        ulong exec = m.ExecOf(w, i);
        m.Typed(w, i, i.S0, GcnType.F32, 0, m.a0, true);
        m.Typed(w, i, i.S1, GcnType.F32, 1, m.a1, true);
        ulong select = i.Opcode.Format == GcnFormat.Vop3 ? m.Scalar64(w, i.S2, i) : m.ReadScalar64(w, 106, i);
        ulong active = i.Variant == GcnVariant.Dpp ? m.Dpp(i, exec, m.a0) : exec;
        for (ulong mask = active; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            m.result[lane] = (select >> lane & 1) != 0 ? m.a1[lane] : m.a0[lane];
        }

        m.WriteResult(w, i, GcnType.B32, active);
    }

    /// <summary>v_mad_u64_u32, v_mad_i64_i32: a 32 × 32 product plus a 64-bit addend; the carry out of the add into SD.</summary>
    internal void Mad64(GcnWave w, GcnInstruction i, bool signed)
    {
        ulong exec = ExecOf(w, i);
        Typed(w, i, i.S0, GcnType.U32, 0, a0, false);
        Typed(w, i, i.S1, GcnType.U32, 1, a1, false);
        Typed(w, i, i.S2, GcnType.U64, 2, a2, false);
        ulong carries = 0;
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            if (signed)
            {
                Int128 sum = (Int128)((long)(int)a0[lane] * (int)a1[lane]) + (long)a2[lane];
                result[lane] = (ulong)(long)sum;
                carries |= sum != (long)sum ? 1UL << lane : 0;
            }
            else
            {
                UInt128 sum = (UInt128)(a0[lane] * a1[lane]) + a2[lane];
                result[lane] = (ulong)sum;
                carries |= sum >> 64 != 0 ? 1UL << lane : 0;
            }
        }

        WriteResult(w, i, GcnType.U64, exec);
        WriteScalar64(w, i.SD, carries, i);
    }

    /// <summary>v_div_scale_f32/f64: the scaled operand per lane, the flag per lane into SD.</summary>
    internal static void DivScale(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        bool wide = i.Opcode.Base.EndsWith("f64", StringComparison.Ordinal);
        var type = wide ? GcnType.F64 : GcnType.F32;
        ulong exec = m.ExecOf(w, i);
        m.Typed(w, i, i.S0, type, 0, m.a0, true);
        m.Typed(w, i, i.S1, type, 1, m.a1, true);
        m.Typed(w, i, i.S2, type, 2, m.a2, true);
        ulong flags = 0;
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            bool flag;
            m.result[lane] = wide ? B(GcnFloat.DivScale(D(m.a0[lane]), D(m.a1[lane]), D(m.a2[lane]), out flag)) : B(GcnFloat.DivScale(F(m.a0[lane]), F(m.a1[lane]), F(m.a2[lane]), out flag));
            flags |= flag ? 1UL << lane : 0;
        }

        m.WriteResult(w, i, type, exec);
        m.WriteScalar64(w, i.SD, flags, i);
    }

    /// <summary>v_div_fmas_f32/f64: fma, with the quotient's scale undone in the lanes whose VCC bit div_scale set.</summary>
    internal static void DivFmas(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        bool wide = i.Opcode.Base.EndsWith("f64", StringComparison.Ordinal);
        var type = wide ? GcnType.F64 : GcnType.F32;
        ulong exec = m.ExecOf(w, i);
        m.Typed(w, i, i.S0, type, 0, m.a0, true);
        m.Typed(w, i, i.S1, type, 1, m.a1, true);
        m.Typed(w, i, i.S2, type, 2, m.a2, true);
        ulong vcc = m.ReadScalar64(w, 106, i);
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            bool flag = (vcc >> lane & 1) != 0;
            m.result[lane] = wide ? B(GcnFloat.DivFmas(D(m.a0[lane]), D(m.a1[lane]), D(m.a2[lane]), flag)) : B(GcnFloat.DivFmas(F(m.a0[lane]), F(m.a1[lane]), F(m.a2[lane]), flag));
        }

        if (i.Omod != 0 || i.Clamp)
        {
            m.OutputModifiers(i, type, exec, false);
        }

        m.WriteResult(w, i, type, exec);
    }

    /// <summary>v_readfirstlane_b32: the first active lane's value (lane 0 when EXEC is 0) into an SGPR.</summary>
    internal static void ReadFirstLane(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        ulong exec = m.ExecOf(w, i);
        int lane = exec == 0 ? 0 : BitOperations.TrailingZeroCount(exec);
        m.Source32(w, i.S0, i, m.L0);
        m.WriteScalar(w, i.D, m.L0[lane], i);
    }

    /// <summary>v_readlane_b32: lane S1 (mod 64) of a VGPR into an SGPR, whatever EXEC is.</summary>
    internal static void ReadLane(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        int lane = (int)(m.Scalar32(w, i.S1, i) & 63);
        if (i.S0 < 256)
        {
            throw m.Fail(GcnFailure.NotModelled, w, i, "v_readlane_b32 reads a VGPR");
        }

        m.ReadVector(w, i.S0 - 256, i, m.L0);
        m.WriteScalar(w, i.D, m.L0[lane], i);
    }

    /// <summary>v_writelane_b32: an SGPR (or constant) into lane S1 (mod 64) of a VGPR, whatever EXEC is.</summary>
    internal static void WriteLane(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        uint value = m.Scalar32(w, i.S0, i);
        int lane = (int)(m.Scalar32(w, i.S1, i) & 63);
        m.L0[lane] = value;
        m.WriteVector(w, i.D - 256, i, m.L0, 1UL << lane);
    }

    /// <summary>v_swap_b32: two VGPRs exchanged in the active lanes.</summary>
    internal static void Swap(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        ulong exec = m.ExecOf(w, i);
        m.ReadVector(w, i.D - 256, i, m.L0);
        m.ReadVector(w, i.S0 - 256, i, m.L1);
        m.WriteVector(w, i.D - 256, i, m.L1, exec);
        m.WriteVector(w, i.S0 - 256, i, m.L0, exec);
    }

    internal static void AccRead(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        ulong exec = m.ExecOf(w, i);
        m.ReadVector(w, i.S0 - 256, i, m.L0, acc: true);
        m.WriteVector(w, i.D - 256, i, m.L0, exec);
    }

    internal static void AccWrite(GcnMachine m, GcnWave w, GcnInstruction i)
    {
        ulong exec = m.ExecOf(w, i);
        m.Source32(w, i.S0, i, m.L0);
        m.WriteVector(w, i.D - 256, i, m.L0, exec, acc: true);
    }

    /// <summary>VOP3P packed 16-bit: the low result from the halves op_sel picks (neg_lo), the high from those op_sel_hi picks (neg_hi).</summary>
    internal void Packed(GcnWave w, GcnInstruction i, bool isFloat, Func<ulong, ulong, ulong, ulong> f)
    {
        ulong exec = ExecOf(w, i);
        int sources = i.Opcode.Sources;
        for (int k = 0; k < sources; k++)
        {
            Source32(w, k == 0 ? i.S0 : k == 1 ? i.S1 : i.S2, i, L0, isFloat);
            var buffer = Buffer(k);
            for (int lane = 0; lane < 64; lane++)
            {
                buffer[lane] = L0[lane];
            }
        }

        if (!isFloat && (i.Neg | i.NegHi) != 0)
        {
            throw Fail(GcnFailure.NotModelled, w, i, "neg_lo or neg_hi on integer halves");
        }

        Span<ulong> x = stackalloc ulong[3];
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            ulong low = 0, high = 0;
            for (int part = 0; part < 2; part++)
            {
                for (int k = 0; k < sources; k++)
                {
                    int pick = part == 0 ? i.OpSel : i.OpSelHi;
                    int negate = part == 0 ? i.Neg : i.NegHi;
                    ulong half = Buffer(k)[lane] >> (16 * (pick >> k & 1)) & 0xFFFF;
                    x[k] = (negate >> k & 1) != 0 ? half ^ 0x8000 : half;
                }

                ulong value = f(x[0], x[1], x[2]) & 0xFFFF;
                if (i.Clamp)
                {
                    if (!isFloat)
                    {
                        throw Fail(GcnFailure.NotModelled, w, i, "clamp on integer halves");
                    }

                    float v = (float)H(value);
                    value = B((Half)(float.IsNaN(v) ? 0 : Math.Clamp(v, 0f, 1f)));
                }

                if (part == 0)
                {
                    low = value;
                }
                else
                {
                    high = value;
                }
            }

            result[lane] = low | high << 16;
        }

        WriteResult(w, i, GcnType.B32, exec);
    }

    /// <summary>
    /// v_mad_mix and v_fma_mix: each source float32 (op_sel_hi 0) or the float16 half op_sel picks (op_sel_hi 1),
    /// neg_lo negates and neg_hi takes the absolute value; the result float32, or float16 into the low or high half.
    /// </summary>
    internal void Mix(GcnWave w, GcnInstruction i, bool fused, int part)
    {
        ulong exec = ExecOf(w, i);
        for (int k = 0; k < 3; k++)
        {
            Source32(w, k == 0 ? i.S0 : k == 1 ? i.S1 : i.S2, i, L0, (i.OpSelHi >> k & 1) != 0);
            var buffer = Buffer(k);
            for (int lane = 0; lane < 64; lane++)
            {
                buffer[lane] = L0[lane];
            }
        }

        if (part != 0)
        {
            ReadVector(w, i.D - 256, i, R1);
        }

        Span<float> x = stackalloc float[3];
        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            for (int k = 0; k < 3; k++)
            {
                ulong bits = Buffer(k)[lane];
                float v = (i.OpSelHi >> k & 1) != 0 ? (float)H(bits >> (16 * (i.OpSel >> k & 1))) : F(bits);
                if ((i.NegHi >> k & 1) != 0)
                {
                    v = Math.Abs(v);
                }

                if ((i.Neg >> k & 1) != 0)
                {
                    v = -v;
                }

                x[k] = v;
            }

            float r = fused ? MathF.FusedMultiplyAdd(x[0], x[1], x[2]) : Mad(x[0], x[1], x[2]);
            if (i.Clamp)
            {
                r = float.IsNaN(r) ? 0 : Math.Clamp(r, 0f, 1f);
            }

            result[lane] = part switch
            {
                0 => B(r),
                1 => (R1[lane] & 0xFFFF0000) | B((Half)r),
                _ => (R1[lane] & 0xFFFF) | B((Half)r) << 16,
            };
        }

        WriteResult(w, i, GcnType.B32, exec);
    }

    /// <summary>v_dot*: per lane over the packed parts (with op_sel and op_sel_hi at their defaults).</summary>
    internal void Dot(GcnWave w, GcnInstruction i, Func<ulong, ulong, ulong, ulong> f)
    {
        if (i.OpSel != 0 || i.OpSelHi != 7 || i.Neg != 0 || i.NegHi != 0 || i.Clamp)
        {
            throw Fail(GcnFailure.NotModelled, w, i, "a dot product with op_sel, neg or clamp");
        }

        ulong exec = ExecOf(w, i);
        for (int k = 0; k < 3; k++)
        {
            Source32(w, k == 0 ? i.S0 : k == 1 ? i.S1 : i.S2, i, L0);
            var buffer = Buffer(k);
            for (int lane = 0; lane < 64; lane++)
            {
                buffer[lane] = L0[lane];
            }
        }

        for (ulong mask = exec; mask != 0; mask &= mask - 1)
        {
            int lane = BitOperations.TrailingZeroCount(mask);
            result[lane] = (uint)f(a0[lane], a1[lane], a2[lane]);
        }

        WriteResult(w, i, GcnType.B32, exec);
    }
}
