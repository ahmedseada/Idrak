// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Gpu.Cuda;

// PTX for the element-wise operations beyond the activations, as the CPU computes them (CpuBackend.Pointwise.cs): square
// roots, sine and cosine, SiLU and sign with their gradients, element-wise maximum and minimum and the gradient of either,
// powers, clamping and selection by a mask with their gradients. One thread per element (Elementwise). Every product,
// quotient and sum is rounded on its own (.rn, no contraction) in the CPU's order, so where the CPU is exact (square roots,
// sign, extremes, clamping, selection, the accumulation of every gradient) the bits are the CPU's.
//
// Sine and cosine: x is reduced by quarter turns, r = x - j·π/2 with j = rint(x·2/π) and π/2 in three floats (Cody-Waite,
// each step one fused multiply-add; a second step corrects j where x·2/π rounded to a neighbour), then minimax polynomials
// on [-π/4, π/4] and the quadrant j mod 4 pick ±sin r or ±cos r. Emulated against MathF on the host: within 2 ulp for
// |x| < 1e6, 1.2e-7 for |x| < 1e7 and 5e-6 for |x| < 1e9; beyond that the float x no longer carries its turn. (The
// hardware sin.approx alone is accurate only near zero.)
//
// Optional storages are passed as another valid pointer with their flag bit clear; the kernels never touch them.
internal static partial class PtxKernels
{
    /// <summary>The kernels of this file (in the main module).</summary>
    public static readonly string[] PointwiseNames =
    [
        "sqrt_f32", "sin_f32", "cos_f32", "silu_f32", "sign_f32", "sqrt_bwd_f32", "sin_bwd_f32", "cos_bwd_f32", "silu_bwd_f32",
        "max_f32", "min_f32", "extremum_bwd_f32", "pow_f32", "pow_bwd_f32", "clamp_f32", "clamp_bwd_f32", "where_f32", "where_bwd_f32",
    ];

    /// <summary>pow_f32 / pow_bwd_f32 flags (PowFlags): the exponent is an odd integer.</summary>
    public const uint PowOdd = 1;

    /// <summary>The exponent is finite and not an integer (a finite negative base gives NaN).</summary>
    public const uint PowFractional = 2;

    /// <summary>The exponent is NaN (only x = 1 gives 1).</summary>
    public const uint PowNaN = 4;

    /// <summary>The exponent is 0 (1 everywhere, NaN too).</summary>
    public const uint PowZero = 8;

    /// <summary>The exponent is 1 (x itself).</summary>
    public const uint PowOne = 16;

    /// <summary>The exponent is 2 (x · x, correctly rounded).</summary>
    public const uint PowTwo = 32;

    // π/2 in three floats (C1 = π/2 rounded, C2 = π/2 - C1 rounded, C3 = the rest rounded; checked against 62 digits of π),
    // negated for the fused multiply-adds; 2/π rounded.
    private const string NegHalfPi1 = "0fBFC90FDB";
    private const string NegHalfPi2 = "0f333BBD2E";
    private const string NegHalfPi3 = "0f26F72CED";
    private const string TwoOverPi = "0f3F22F983";
    private const string QuietNaN = "0fFFC00000";

    // The flags of a power's exponent p (the same tests MathF.Pow makes on it).
    public static uint PowFlags(float p)
    {
        bool integer = float.IsFinite(p) && MathF.Floor(p) == p;
        uint flags = 0;
        if (integer && MathF.IEEERemainder(p, 2f) != 0f)
        {
            flags |= PowOdd;
        }

        if (float.IsFinite(p) && !integer)
        {
            flags |= PowFractional;
        }

        if (float.IsNaN(p))
        {
            flags |= PowNaN;
        }

        flags |= p == 0f ? PowZero : p == 1f ? PowOne : p == 2f ? PowTwo : 0u;
        return flags;
    }

    private static void BuildPointwise(StringBuilder sb)
    {
        // ---------------------------------------------------------------- unary functions

        // sqrt.rn is correctly rounded, as MathF.Sqrt (NaN for negative x, -0 for -0).
        Elementwise(sb, "sqrt_f32", ["x", "y"], [],
            """
            ld.global.f32 %f1, [%a_x];
            sqrt.rn.f32 %f2, %f1;
            st.global.f32 [%a_y], %f2;
            """);

        Elementwise(sb, "sin_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            {SinCosOf("%f2", "%f1", cosine: false)}
            st.global.f32 [%a_y], %f2;
            """);

        Elementwise(sb, "cos_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            {SinCosOf("%f2", "%f1", cosine: true)}
            st.global.f32 [%a_y], %f2;
            """);

        // y = x · s, s = 1 / (1 + e^-x) (the CPU's SigmoidOf).
        Elementwise(sb, "silu_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            {SigmoidOf("%f2", "%f1")}
            mul.rn.f32 %f3, %f1, %f2;
            st.global.f32 [%a_y], %f3;
            """);

        // 1 for x > 0, -1 for x < 0, else x itself (±0, and NaN for NaN).
        Elementwise(sb, "sign_f32", ["x", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            setp.gt.f32 %p1, %f1, {Zero};
            setp.lt.f32 %p2, %f1, {Zero};
            selp.f32 %f2, {One}, %f1, %p1;
            selp.f32 %f2, {F(-1f)}, %f2, %p2;
            st.global.f32 [%a_y], %f2;
            """);

        // Backward: dx += dy · op'(x) in the CPU's order (MathBackwardLoop); sign's gradient is zero and needs no kernel.
        // dx += dy · 0.5 / y.
        Elementwise(sb, "sqrt_bwd_f32", ["x", "y", "dy", "dx"], [],
            $"""
            ld.global.f32 %f1, [%a_y];
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            mul.rn.f32 %f4, %f2, {F(0.5f)};
            div.rn.f32 %f4, %f4, %f1;
            add.rn.f32 %f3, %f3, %f4;
            st.global.f32 [%a_dx], %f3;
            """);

        // dx += dy · cos(x).
        Elementwise(sb, "sin_bwd_f32", ["x", "y", "dy", "dx"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            {SinCosOf("%f4", "%f1", cosine: true)}
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            mul.rn.f32 %f4, %f2, %f4;
            add.rn.f32 %f3, %f3, %f4;
            st.global.f32 [%a_dx], %f3;
            """);

        // dx -= dy · sin(x).
        Elementwise(sb, "cos_bwd_f32", ["x", "y", "dy", "dx"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            {SinCosOf("%f4", "%f1", cosine: false)}
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            mul.rn.f32 %f4, %f2, %f4;
            sub.rn.f32 %f3, %f3, %f4;
            st.global.f32 [%a_dx], %f3;
            """);

        // dx += dy · s · (1 + x · (1 - s)), s = sigmoid(x).
        Elementwise(sb, "silu_bwd_f32", ["x", "y", "dy", "dx"], [],
            $"""
            ld.global.f32 %f1, [%a_x];
            {SigmoidOf("%f2", "%f1")}
            ld.global.f32 %f3, [%a_dy];
            ld.global.f32 %f4, [%a_dx];
            sub.rn.f32 %f5, {One}, %f2;
            mul.rn.f32 %f5, %f1, %f5;
            add.rn.f32 %f5, {One}, %f5;
            mul.rn.f32 %f6, %f3, %f2;
            mul.rn.f32 %f6, %f6, %f5;
            add.rn.f32 %f4, %f4, %f6;
            st.global.f32 [%a_dx], %f4;
            """);

        // ---------------------------------------------------------------- extremes

        foreach (var (name, minimum) in new[] { ("max_f32", false), ("min_f32", true) })
        {
            Elementwise(sb, name, ["a", "b", "c"], [],
                $"""
                ld.global.f32 %f1, [%a_a];
                ld.global.f32 %f2, [%a_b];
                {Extreme(minimum, "%f3", "%f1", "%f2")}
                st.global.f32 [%a_c], %f3;
                """);
        }

        // The gradient of max(a, b) or (flags & 4) min(a, b): da += dy where a wins (a ≥ b, or a ≤ b; ties to a; never for
        // NaN), else db += dy; flags & 1: da given, flags & 2: db given.
        Elementwise(sb, "extremum_bwd_f32", ["a", "b", "dy", "da", "db"], [("u32", "flags")],
            """
            ld.global.f32 %f1, [%a_a];
            ld.global.f32 %f2, [%a_b];
            ld.global.f32 %f3, [%a_dy];
            setp.ge.f32 %p1, %f1, %f2;
            setp.le.f32 %p2, %f1, %f2;
            and.b32 %r5, %s_flags, 4;
            setp.ne.u32 %p3, %r5, 0;
            @%p3 mov.pred %p1, %p2;
            and.b32 %r5, %s_flags, 1;
            setp.ne.u32 %p4, %r5, 0;
            and.pred %p4, %p4, %p1;
            @!%p4 bra EXT_SECOND;
            ld.global.f32 %f4, [%a_da];
            add.rn.f32 %f4, %f4, %f3;
            st.global.f32 [%a_da], %f4;
            EXT_SECOND:
            not.pred %p1, %p1;
            and.b32 %r5, %s_flags, 2;
            setp.ne.u32 %p5, %r5, 0;
            and.pred %p5, %p5, %p1;
            @!%p5 bra DONE;
            ld.global.f32 %f5, [%a_db];
            add.rn.f32 %f5, %f5, %f3;
            st.global.f32 [%a_db], %f5;
            """);

        // ---------------------------------------------------------------- powers

        // y = x^p as MathF.Pow (flags: PowFlags(p)).
        Elementwise(sb, "pow_f32", ["x", "y"], [("f32", "p"), ("u32", "flags")],
            $"""
            ld.global.f32 %f1, [%a_x];
            {PowOf("%f2", "%f1", "%s_p", "%s_flags")}
            st.global.f32 [%a_y], %f2;
            """);

        // dx += dy · e · x^p, p = e - 1 (flags: PowFlags(p)); the CPU's order. The host skips e = 0.
        Elementwise(sb, "pow_bwd_f32", ["x", "dy", "dx"], [("f32", "e"), ("f32", "p"), ("u32", "flags")],
            $"""
            ld.global.f32 %f1, [%a_x];
            {PowOf("%f2", "%f1", "%s_p", "%s_flags")}
            ld.global.f32 %f3, [%a_dy];
            ld.global.f32 %f4, [%a_dx];
            mul.rn.f32 %f3, %f3, %s_e;
            mul.rn.f32 %f3, %f3, %f2;
            add.rn.f32 %f4, %f4, %f3;
            st.global.f32 [%a_dx], %f4;
            """);

        // ---------------------------------------------------------------- clamping and selection

        // y = MathF.Min(MathF.Max(x, lo), hi).
        Elementwise(sb, "clamp_f32", ["x", "y"], [("f32", "lo"), ("f32", "hi")],
            $"""
            ld.global.f32 %f1, [%a_x];
            {Extreme(false, "%f2", "%f1", "%s_lo")}
            {Extreme(true, "%f3", "%f2", "%s_hi")}
            st.global.f32 [%a_y], %f3;
            """);

        // dx += dy where lo ≤ x ≤ hi (never for NaN).
        Elementwise(sb, "clamp_bwd_f32", ["x", "dy", "dx"], [("f32", "lo"), ("f32", "hi")],
            """
            ld.global.f32 %f1, [%a_x];
            setp.ge.f32 %p1, %f1, %s_lo;
            setp.le.f32 %p2, %f1, %s_hi;
            and.pred %p1, %p1, %p2;
            @!%p1 bra DONE;
            ld.global.f32 %f2, [%a_dy];
            ld.global.f32 %f3, [%a_dx];
            add.rn.f32 %f3, %f3, %f2;
            st.global.f32 [%a_dx], %f3;
            """);

        // y = condition ≠ 0 ? a : b (NaN selects a, ±0 selects b).
        Elementwise(sb, "where_f32", ["condition", "a", "b", "y"], [],
            $"""
            ld.global.f32 %f1, [%a_condition];
            ld.global.f32 %f2, [%a_a];
            ld.global.f32 %f3, [%a_b];
            setp.neu.f32 %p1, %f1, {Zero};
            selp.f32 %f4, %f2, %f3, %p1;
            st.global.f32 [%a_y], %f4;
            """);

        // da += dy where condition ≠ 0 (flags & 1: da given), db += dy elsewhere (flags & 2: db given).
        Elementwise(sb, "where_bwd_f32", ["condition", "dy", "da", "db"], [("u32", "flags")],
            $"""
            ld.global.f32 %f1, [%a_condition];
            ld.global.f32 %f2, [%a_dy];
            setp.neu.f32 %p1, %f1, {Zero};
            and.b32 %r5, %s_flags, 1;
            setp.ne.u32 %p2, %r5, 0;
            and.pred %p2, %p2, %p1;
            @!%p2 bra WHERE_SECOND;
            ld.global.f32 %f3, [%a_da];
            add.rn.f32 %f3, %f3, %f2;
            st.global.f32 [%a_da], %f3;
            WHERE_SECOND:
            not.pred %p1, %p1;
            and.b32 %r5, %s_flags, 2;
            setp.ne.u32 %p3, %r5, 0;
            and.pred %p3, %p3, %p1;
            @!%p3 bra DONE;
            ld.global.f32 %f4, [%a_db];
            add.rn.f32 %f4, %f4, %f2;
            st.global.f32 [%a_db], %f4;
            """);
    }

    // target = sin(x) or cos(x) (see the file's comment). Uses %f20-%f24, %r20-%r21, %p10; target is not one of them.
    private static string SinCosOf(string target, string x, bool cosine) => $"""
        mul.rn.f32 %f20, {x}, {TwoOverPi};
        cvt.rni.f32.f32 %f20, %f20;
        cvt.rzi.s32.f32 %r20, %f20;
        fma.rn.f32 %f21, %f20, {NegHalfPi1}, {x};
        fma.rn.f32 %f21, %f20, {NegHalfPi2}, %f21;
        fma.rn.f32 %f21, %f20, {NegHalfPi3}, %f21;
        mul.rn.f32 %f20, %f21, {TwoOverPi};
        cvt.rni.f32.f32 %f20, %f20;
        cvt.rzi.s32.f32 %r21, %f20;
        add.s32 %r20, %r20, %r21;
        fma.rn.f32 %f21, %f20, {NegHalfPi1}, %f21;
        fma.rn.f32 %f21, %f20, {NegHalfPi2}, %f21;
        fma.rn.f32 %f21, %f20, {NegHalfPi3}, %f21;
        {(cosine ? "add.s32 %r20, %r20, 1;" : "")}
        mul.rn.f32 %f22, %f21, %f21;
        fma.rn.f32 %f23, %f22, {F(-1.9515295891e-4f)}, {F(8.3321608736e-3f)};
        fma.rn.f32 %f23, %f23, %f22, {F(-1.6666654611e-1f)};
        mul.rn.f32 %f23, %f23, %f22;
        fma.rn.f32 %f23, %f23, %f21, %f21;
        fma.rn.f32 %f24, %f22, {F(2.443315711809948e-5f)}, {F(-1.388731625493765e-3f)};
        fma.rn.f32 %f24, %f24, %f22, {F(4.166664568298827e-2f)};
        fma.rn.f32 %f24, %f24, %f22, {F(-0.5f)};
        fma.rn.f32 %f24, %f24, %f22, {One};
        and.b32 %r21, %r20, 1;
        setp.ne.u32 %p10, %r21, 0;
        selp.f32 {target}, %f24, %f23, %p10;
        and.b32 %r21, %r20, 2;
        setp.ne.u32 %p10, %r21, 0;
        @%p10 neg.f32 {target}, {target};
        {(cosine ? "" : $"setp.eq.f32 %p10, {x}, {Zero};\n@%p10 mov.f32 {target}, {x};")}
        """;

    // target = 1 / (1 + e^-x) (rcp.rn: the correctly rounded quotient, as the CPU's division), e^-x = 2^t with t = -x·log2 e rounded, corrected by t's rounding error and log2 e's second
    // float (the hardware 2^t is then the only approximation). Uses %f25-%f28, %p11; target is not one of them.
    private static string SigmoidOf(string target, string x) => $"""
        neg.f32 %f25, {x};
        mul.rn.f32 %f26, %f25, {F(1.4426950408889634f)};
        neg.f32 %f27, %f26;
        fma.rn.f32 %f27, %f25, {F(1.4426950408889634f)}, %f27;
        fma.rn.f32 %f27, %f25, {F((float)(1.4426950408889634 - (double)1.4426950408889634f))}, %f27;
        mul.rn.f32 %f27, %f27, {Ln2};
        ex2.approx.f32 %f28, %f26;
        testp.finite.f32 %p11, %f26;
        @%p11 fma.rn.f32 %f28, %f28, %f27, %f28;
        add.rn.f32 %f28, %f28, {One};
        rcp.rn.f32 {target}, %f28;
        """;

    // target = MathF.Max(a, b) or MathF.Min(a, b): a NaN a gives a, else a NaN b gives b; equal values give the one with
    // the sign the IEEE 754 maximum/minimum wants (+0 above -0). Uses %f29, %r29, %p13-%p14; target is neither a nor b.
    private static string Extreme(bool minimum, string target, string a, string b) => minimum
        ? $"""
            setp.lt.f32 %p13, {a}, {b};
            selp.f32 {target}, {a}, {b}, %p13;
            mov.b32 %r29, {a};
            setp.lt.s32 %p14, %r29, 0;
            selp.f32 %f29, {a}, {b}, %p14;
            setp.eq.f32 %p13, {a}, {b};
            @%p13 mov.f32 {target}, %f29;
            setp.nan.f32 %p13, {a}, {a};
            @%p13 mov.f32 {target}, {a};
            """
        : $"""
            setp.lt.f32 %p13, {b}, {a};
            selp.f32 {target}, {a}, {b}, %p13;
            mov.b32 %r29, {b};
            setp.lt.s32 %p14, %r29, 0;
            selp.f32 %f29, {a}, {b}, %p14;
            setp.eq.f32 %p13, {a}, {b};
            @%p13 mov.f32 {target}, %f29;
            setp.nan.f32 %p13, {a}, {a};
            @%p13 mov.f32 {target}, {a};
            """;

    // target = x^p as MathF.Pow: |x|^p = 2^(p·log2|x|), t = p·log2|x| rounded and corrected by its rounding error, then the
    // special cases (flags = PowFlags(p)): |x| = 1 gives 1 (for an infinite p too) unless p is NaN and x is not 1; a negative
    // x (-0 and -∞ too) to an odd integer power is negative; a finite negative x to a fractional power is NaN; p = 0, 1
    // and 2 give 1, x and x·x exactly. 0 and ∞ follow from log2 (±∞) and 2^±∞. Uses %f15-%f18, %r15-%r16, %p7-%p9.
    private static string PowOf(string target, string x, string p, string flags) => $"""
        abs.f32 %f15, {x};
        lg2.approx.f32 %f16, %f15;
        mul.rn.f32 %f17, %f16, {p};
        neg.f32 %f18, %f17;
        fma.rn.f32 %f18, %f16, {p}, %f18;
        mul.rn.f32 %f18, %f18, {Ln2};
        ex2.approx.f32 {target}, %f17;
        testp.finite.f32 %p7, %f17;
        @%p7 fma.rn.f32 {target}, {target}, %f18, {target};
        setp.eq.f32 %p7, %f15, {One};
        setp.eq.f32 %p8, {x}, {One};
        and.b32 %r15, {flags}, {PowNaN};
        setp.ne.u32 %p9, %r15, 0;
        @%p9 mov.pred %p7, %p8;
        @%p7 mov.f32 {target}, {One};
        mov.b32 %r15, {x};
        setp.lt.s32 %p7, %r15, 0;
        and.b32 %r16, {flags}, {PowOdd};
        setp.ne.u32 %p8, %r16, 0;
        and.pred %p7, %p7, %p8;
        @%p7 neg.f32 {target}, {target};
        setp.lt.f32 %p7, {x}, {Zero};
        testp.finite.f32 %p8, {x};
        and.pred %p7, %p7, %p8;
        and.b32 %r16, {flags}, {PowFractional};
        setp.ne.u32 %p8, %r16, 0;
        and.pred %p7, %p7, %p8;
        @%p7 mov.f32 {target}, {QuietNaN};
        and.b32 %r16, {flags}, {PowZero};
        setp.ne.u32 %p7, %r16, 0;
        @%p7 mov.f32 {target}, {One};
        and.b32 %r16, {flags}, {PowOne};
        setp.ne.u32 %p7, %r16, 0;
        @%p7 mov.f32 {target}, {x};
        and.b32 %r16, {flags}, {PowTwo};
        setp.ne.u32 %p7, %r16, 0;
        @%p7 mul.rn.f32 {target}, {x}, {x};
        """;
}
