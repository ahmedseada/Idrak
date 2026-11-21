// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

/// <summary>
/// What the generic products widen a weight type from (<see cref="ProductRoutes"/>): the
/// type's own description, never its name: a float's bits (<see cref="ElementType.Float"/>), an integer's bits and sign,
/// or a block of codes and one scale (<see cref="ElementType.Scaled"/>). Null for a type these kernels cannot widen: per
/// value 8, 16 or 32 bits; a block of 32 values whose codes are 8 bits, with a 16-bit float scale, every part on an even
/// byte.
/// </summary>
/// <param name="Kind">'f' a float, 'i' an integer, 'b' a block of codes.</param>
/// <param name="Bits">Bits a value (a code, for a block).</param>
/// <param name="Signed">Whether an integer or a code is two's complement.</param>
/// <param name="Float">A float's bits ('f').</param>
/// <param name="Block">A block's layout ('b').</param>
internal sealed record TypedDecode(char Kind, int Bits, bool Signed, FloatBits Float, ScaledBlock? Block)
{
    /// <summary>The description of <paramref name="type"/> these kernels widen, or null.</summary>
    public static TypedDecode? Of(ElementType type)
    {
        if (type.Scaled is { } block)
        {
            // Integer codes and a float scale only: codes or a scale of another element type (MXFP4's E2M1 and E8M0) are not read here.
            bool codes = block is { CodeType: null, ScaleType: null } && (block.CodeBits == 8 && block.CodeTable is null
                || block.CodeBits == 4 && block.NibblesSplit && (block.CodeTable is null || block.CodeTable is { Count: 16 } grid && grid.All(e => e is >= -128 and <= 127)));
            bool fits = block.Values == PtxKernels.TypedTileDepth && codes && block.Scale.Bits == 16 && block.Scale.Exponent <= 8
                        && block.Bytes % 2 == 0 && block.CodeOffset % 2 == 0 && block.ScaleOffset % 2 == 0
                        && block.CodeOffset + block.Values * block.CodeBits / 8 <= block.Bytes && block.ScaleOffset + 2 <= block.Bytes;
            return fits ? new TypedDecode('b', block.CodeBits, block.SignedCodes, block.Scale, block) : null;
        }

        if (type.IsBlock || type.BlockBits is not (8 or 16 or 32))
        {
            return null;
        }

        if (type.Float is { } bits)
        {
            return bits.Bits == type.BlockBits && bits.Exponent <= 8 && bits.Mantissa <= 23 ? new TypedDecode('f', bits.Bits, true, bits, null) : null;
        }

        return type.IsInteger ? new TypedDecode('i', type.BlockBits, type.IsSigned, default, null) : null;
    }

    /// <summary>Whether values are one a lane's word holds several of (a float or an integer), not blocks.</summary>
    public bool PerValue => Kind != 'b';

    /// <summary>
    /// Whether the matrix-unit lane takes the type (PtxKernels.TypedMma): its values widen into bfloat16 tiles, exactly for
    /// codes of 8 bits (an integer, a block's codes, the block's scale applied in float32) and a float of bfloat16's bits,
    /// rounded to nearest for any other float, as a bfloat16 run rounds its products' operands.
    /// </summary>
    public bool MatrixUnits => Kind == 'f' || Bits <= 8;

    /// <summary>Whether a block's codes are 4-bit halves in ggml's order (byte i: value i low, value i + Values / 2 high).</summary>
    public bool Nibbles => Kind == 'b' && Bits == 4;

    /// <summary>Values a 32-bit word holds (a float or an integer).</summary>
    public int PerWord => 32 / Bits;

    /// <summary>The description as a kernel's name part ("f8m7", "s8", "b32x34c8s+0o2f5m10at0"): the same description, the same kernels.</summary>
    public string Code => Kind switch
    {
        'f' => string.Create(CultureInfo.InvariantCulture, $"f{Float.Exponent}m{Float.Mantissa}"),
        'i' => string.Create(CultureInfo.InvariantCulture, $"{(Signed ? 's' : 'u')}{Bits}"),
        _ => string.Create(CultureInfo.InvariantCulture,
            $"b{Block!.Values}x{Block.Bytes}c{Block.CodeBits}{(Block.NibblesSplit ? "h" : "")}{(Block.CodeTable is { } table ? "t" + string.Join("_", table.Select(e => e < 0 ? "n" + -e : e.ToString(CultureInfo.InvariantCulture))) : "")}{(Block.SignedCodes ? 's' : 'u')}{(Block.CodeBias < 0 ? "n" : "p")}{Math.Abs(Block.CodeBias)}o{Block.CodeOffset}f{Block.Scale.Exponent}m{Block.Scale.Mantissa}at{Block.ScaleOffset}"),
    };
}

// The generic products of the experiment (ProductRoutes, CudaBackend.Typed.cs): for one weight description, a tiled
// product (64 × 64 outputs a block of 256 threads, 32 of k a step, every output one FMA chain over k from zero, as the
// CPU's tiled products compute) and the few-row products (1, 2, 4 or 8 rows), for weights stored inputs-major ([k, ld]:
// k rows of ld ≥ n values, as the int8 and bfloat16 packed weights lie) or outputs-major (n rows of k values, as a file
// stores them and StoredWeight keeps them). Every value is widened from the description inside the kernel (WordValues,
// FloatFrom, the block's codes and scale) and times its scale where one is given, then multiplied: one kernel per
// description, none per type's name.
internal static partial class PtxKernels
{
    /// <summary>The tiled product's output rows and columns a block, and its step of k (a block of a block format).</summary>
    internal const int TypedTileRows = 64, TypedTileColumns = 64, TypedTileDepth = 32, TypedTileThreads = 256;

    /// <summary>Threads of a few-row inputs-major block (one word of columns each) and output columns of an outputs-major block (a warp each).</summary>
    internal const int TypedGemvThreads = 128, TypedWarpColumns = 8;

    /// <summary>The row counts the few-row products are compiled for.</summary>
    internal static ReadOnlySpan<int> TypedGemvRows => [1, 2, 4, 8];

    internal const string TypedSumName = "typed_sum_parts";

    internal static string TypedTileName(TypedDecode d, bool outputsMajor) =>
        string.Create(CultureInfo.InvariantCulture, $"typed_tile_{d.Code}_{(outputsMajor ? 'n' : 'k')}");

    internal static string TypedGemvName(TypedDecode d, bool outputsMajor, int rows, bool wide = false) =>
        string.Create(CultureInfo.InvariantCulture, $"typed_gemv_{d.Code}_{(outputsMajor ? (wide ? "n4" : "n") : "k")}_r{rows}");

    /// <summary>The module of <paramref name="d"/>'s generic products: its kernels' names and its PTX.</summary>
    internal static (string[] Names, string Source) TypedModule(TypedDecode d)
    {
        var sb = new StringBuilder();
        sb.AppendLine(".version 6.0");
        sb.AppendLine(".target sm_50");
        sb.AppendLine(".address_size 64");
        sb.AppendLine();
        var names = new List<string>();
        foreach (bool outputsMajor in d.PerValue ? new[] { false, true } : [true])
        {
            TypedTile(sb, d, outputsMajor);
            names.Add(TypedTileName(d, outputsMajor));
            foreach (int rows in TypedGemvRows)
            {
                if (outputsMajor)
                {
                    TypedGemvOutputs(sb, d, rows);
                    foreach (bool w in d.PerValue ? new[] { false, true } : [false])
                    {
                        if (w)
                        {
                            TypedGemvOutputs(sb, d, rows, wide: true);
                            names.Add(TypedGemvName(d, true, rows, wide: true));
                        }

                        TypedGemvOutputs(sb, d, rows, wide: w, many: true);
                        names.Add(TypedGemvName(d, true, rows, wide: w) + "_many");
                        TypedGemvOutputs(sb, d, rows, wide: w, many: true, epilogue: Epilogue.Gated);
                        names.Add(TypedGemvName(d, true, rows, wide: w) + "_many_gated");
                        TypedGemvOutputs(sb, d, rows, wide: w, epilogue: Epilogue.AddNorm);
                        names.Add(TypedGemvName(d, true, rows, wide: w) + "_addnorm");
                    }
                }
                else
                {
                    TypedGemvInputs(sb, d, rows);
                }

                names.Add(TypedGemvName(d, outputsMajor, rows));
            }
        }

        TypedSum(sb);
        names.Add(TypedSumName);
        return ([.. names], sb.ToString());
    }

    // ------------------------------------------------------------------ widening from the description

    // dest = the float `bits` describes, from its raw bits in the low bits of `raw` (zero-extended). Exponent 8: the bits
    // shifted into float32's place (bfloat16, float32: exact, specials and subnormals included). Otherwise: the sign; a
    // normal value rebiased; the largest exponent an infinity or a NaN with its mantissa kept in place (as the CPU widens
    // a half: sign | 0x7F800000 | mantissa << (23 − M)); the smallest a zero or a subnormal, mantissa · 2^(1 − bias − M)
    // (exact). Temporaries %q1-%q6, %qp0, %qp1 and %qf0.
    private static void FloatFrom(StringBuilder s, string raw, string dest, FloatBits bits)
    {
        int e = bits.Exponent, m = bits.Mantissa;
        if (e == 8)
        {
            if (m == 23)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    mov.b32 {dest}, {raw};");
                return;
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"    shl.b32 %q1, {raw}, {23 - m};");
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.b32 {dest}, %q1;");
            return;
        }

        uint expMask = (1u << e) - 1, manMask = (1u << m) - 1;
        uint subnormalScale = (uint)(1 - bits.Bias - m + 127) << 23;      // 2^(1 − bias − M), a normal float32
        s.AppendLine(CultureInfo.InvariantCulture, $"""
                shr.u32 %q1, {raw}, {e + m};
                and.b32 %q1, %q1, 1;
                shl.b32 %q1, %q1, 31;
                shr.u32 %q2, {raw}, {m};
                and.b32 %q2, %q2, {expMask};
                and.b32 %q3, {raw}, {manMask};
                shl.b32 %q4, %q3, {23 - m};
                add.u32 %q5, %q2, {127 - bits.Bias};
                shl.b32 %q5, %q5, 23;
                or.b32 %q5, %q5, %q4;
                or.b32 %q5, %q5, %q1;
                or.b32 %q6, %q4, 0x7F800000;
                or.b32 %q6, %q6, %q1;
                setp.eq.u32 %qp0, %q2, {expMask};
                selp.b32 %q5, %q6, %q5, %qp0;
                cvt.rn.f32.u32 %qf0, %q3;
                mul.rn.f32 %qf0, %qf0, 0f{subnormalScale:X8};
                mov.b32 %q6, %qf0;
                or.b32 %q6, %q6, %q1;
                setp.eq.u32 %qp1, %q2, 0;
                selp.b32 %q5, %q6, %q5, %qp1;
                mov.b32 {dest}, %q5;
            """);
    }

    // The values of a float or an integer the 32-bit word `word` holds (the first in its lowest bits), widened into `dest`.
    private static void WordValues(StringBuilder s, TypedDecode d, string word, IReadOnlyList<string> dest)
    {
        for (int i = 0; i < dest.Count; i++)
        {
            int at = i * d.Bits;
            if (d.Kind == 'i')
            {
                string type = d.Signed ? "s32" : "u32";
                if (d.Bits == 32)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    cvt.rn.f32.{type} {dest[i]}, {word};");
                    continue;
                }

                s.AppendLine(CultureInfo.InvariantCulture, $"    bfe.{type} %q0, {word}, {at}, {d.Bits};");
                s.AppendLine(CultureInfo.InvariantCulture, $"    cvt.rn.f32.{type} {dest[i]}, %q0;");
                continue;
            }

            if (d.Bits == 32)
            {
                FloatFrom(s, word, dest[i], d.Float);
                continue;
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"    bfe.u32 %q0, {word}, {at}, {d.Bits};");
            FloatFrom(s, "%q0", dest[i], d.Float);
        }
    }

    // The two codes of a block's 16-bit unit `unit` (the first in its low byte) times `scale`, into `first`, `second`; 4-bit
    // codes (TypedDecode.Nibbles) from the half of each byte that %ns selects (NibbleAt).
    private static void UnitCodes(StringBuilder s, TypedDecode d, string unit, string scale, string first, string second)
    {
        var block = d.Block!;
        string type = d.Signed ? "s32" : "u32";
        if (d.Nibbles)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    shr.b32 %qn, {unit}, %ns;");
            unit = "%qn";
        }

        foreach (var (at, dest) in new[] { (0, first), (8, second) })
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    bfe.{type} %q0, {unit}, {at}, {d.Bits};");
            TableEntry(s, d, "%q0");
            if (block.CodeBias != 0)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    add.s32 %q0, %q0, {block.CodeBias};");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"    cvt.rn.f32.s32 {dest}, %q0;");
            s.AppendLine(CultureInfo.InvariantCulture, $"    mul.rn.f32 {dest}, {dest}, {scale};");
        }
    }

    private static string Shift(int bits) => bits switch { 8 => "", 16 => "1", _ => "2" };

    // A block's 4-bit code (in `reg`, 0..15) as its table entry where the description has a code table (ggml's IQ4_NL
    // grid), sign-extended: the 16 int8 entries in four words, a byte chosen from each half, the half by the code's top bit.
    private static void TableEntry(StringBuilder s, TypedDecode d, string reg)
    {
        if (d.Block?.CodeTable is not { } table)
        {
            return;
        }

        foreach (var (name, word) in TableWords(table))
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.b32 {name}, {word};");
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"    and.b32 %qt, {reg}, 7;");
        s.AppendLine("    prmt.b32 %qa, %ta, %tb, %qt;");
        s.AppendLine("    prmt.b32 %qb, %tc, %td, %qt;");
        s.AppendLine(CultureInfo.InvariantCulture, $"    setp.ge.u32 %tp, {reg}, 8;");
        s.AppendLine(CultureInfo.InvariantCulture, $"    selp.b32 {reg}, %qb, %qa, %tp;");
        s.AppendLine(CultureInfo.InvariantCulture, $"    bfe.s32 {reg}, {reg}, 0, 8;");
    }

    // A 16-entry table's int8 entries as four little-endian words (%ta .. %td).
    internal static (string Name, uint Word)[] TableWords(IReadOnlyList<int> table) =>
        [.. new[] { "%ta", "%tb", "%tc", "%td" }.Select((name, w) => (name, (uint)((byte)(sbyte)table[4 * w] | (byte)(sbyte)table[4 * w + 1] << 8
            | (byte)(sbyte)table[4 * w + 2] << 16 | (byte)(sbyte)table[4 * w + 3] << 24)))];

    // For 4-bit codes in ggml's order: the byte of the codes of values [v, ...) (v in register `value`, the group within
    // one half of the block) into %nb, the shift of their half into %ns.
    private static void NibbleAt(StringBuilder s, TypedDecode d, string value)
    {
        int half = d.Block!.Values / 2;
        s.AppendLine(CultureInfo.InvariantCulture, $"    and.b32 %nb, {value}, {half - 1};");
        s.AppendLine(CultureInfo.InvariantCulture, $"    shr.u32 %ns, {value}, {(int)Math.Log2(half)};");
        s.AppendLine("    shl.b32 %ns, %ns, 2;");
    }

    // A numbered register's name ("%v3"), formatted alike on every machine.
    private static string R(string prefix, int number) => string.Create(CultureInfo.InvariantCulture, $"{prefix}{number}");

    // The temporaries every widening uses.
    private const string WideningRegisters = """
            .reg .pred %qp<2>;
            .reg .b32 %q<7>;
            .reg .f32 %qf<1>;
            .reg .b32 %nb, %ns, %qn;
            .reg .b32 %ta, %tb, %tc, %td, %qt, %qa, %qb;
            .reg .pred %tp;
        """;

    // ------------------------------------------------------------------ the tiled product

    // y [m, n] = x [m, k] · W: each block 64 × 64 outputs (16 × 16 threads, 4 × 4 each), k in steps of 32: the step's x tile
    // (64 rows × 32) and W tile (32 × 64 columns, widened and scaled) in shared memory, then 32 FMAs a thread per output.
    // Each output's sum is one FMA chain over k from zero in k's order. Inputs-major W (k rows of ld values, value (r, c)
    // at r · ld + c, p_s a scale per column or 0) or outputs-major (n rows of k values; p_s a scale per row or 0; a block
    // format's blocks along k, k a multiple of 32).
    private static void TypedTile(StringBuilder sb, TypedDecode d, bool outputsMajor)
    {
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{TypedTileName(d, outputsMajor)}}(
                .param .u64 p_x, .param .u64 p_w, .param .u64 p_s, .param .u64 p_y,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .u32 p_ld
            )
            .maxntid {{TypedTileThreads}}, 1, 1
            {
                .reg .pred %p<24>;
                .reg .b32 %r<40>;
                .reg .b32 %u<8>;
                .reg .f32 %c<16>;
                .reg .f32 %a<4>;
                .reg .f32 %b<4>;
                .reg .f32 %v<8>;
                .reg .f32 %xv<8>;
                .reg .f32 %sv<8>;
                .reg .f32 %sc;
                .reg .b64 %rd<16>;
            {{WideningRegisters}}
                .shared .align 16 .f32 typed_xs[{{TypedTileDepth * TypedTileRows}}];
                .shared .align 16 .f32 typed_ws[{{TypedTileDepth * TypedTileColumns}}];
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_w];
                ld.param.u64 %rd3, [p_s];
                ld.param.u64 %rd4, [p_y];
                setp.ne.u64 %p20, %rd3, 0;
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.u32 %r4, [p_ld];
                mov.u32 %r5, %tid.x;
                and.b32 %r6, %r5, 15;
                shr.u32 %r7, %r5, 4;
                mov.u32 %r8, %ctaid.x;
                shl.b32 %r8, %r8, 6;
                mov.u32 %r9, %ctaid.y;
                shl.b32 %r9, %r9, 6;
                shr.u32 %r10, %r5, 3;
                and.b32 %r11, %r5, 7;
                shl.b32 %r11, %r11, 2;
                mov.u32 %r16, typed_xs;
                mad.lo.u32 %r17, %r11, {{TypedTileRows}}, %r10;
                shl.b32 %r17, %r17, 2;
                add.u32 %r16, %r16, %r17;
                mov.u32 %r18, typed_xs;
                shl.b32 %r17, %r7, 4;
                add.u32 %r18, %r18, %r17;
                mov.u32 %r19, typed_ws;
                shl.b32 %r17, %r6, 4;
                add.u32 %r19, %r19, %r17;
            """);

        // Where this thread widens W into the tile, and the scales it applies (read once, before the steps).
        if (!outputsMajor)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    mov.u32 %r21, %r10;
                    shl.b32 %r22, %r11, 1;
                    mov.u32 %r20, typed_ws;
                    mad.lo.u32 %r17, %r21, {{TypedTileColumns}}, %r22;
                    shl.b32 %r17, %r17, 2;
                    add.u32 %r20, %r20, %r17;
                    add.u32 %r23, %r8, %r22;
                """);
            for (int v = 0; v < 8; v++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        add.u32 %r24, %r23, {{v}};
                        setp.lt.u32 %p21, %r24, %r2;
                        and.pred %p21, %p21, %p20;
                        mov.f32 %sv{{v}}, 0f3F800000;
                        mul.wide.u32 %rd5, %r24, 4;
                        add.u64 %rd5, %rd5, %rd3;
                        @%p21 ld.global.f32 %sv{{v}}, [%rd5];
                    """);
            }
        }
        else
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    shr.u32 %r21, %r5, 2;
                    and.b32 %r22, %r5, 3;
                    shl.b32 %r22, %r22, 3;
                    mov.u32 %r20, typed_ws;
                    mad.lo.u32 %r17, %r22, {{TypedTileColumns}}, %r21;
                    shl.b32 %r17, %r17, 2;
                    add.u32 %r20, %r20, %r17;
                    add.u32 %r23, %r8, %r21;
                    setp.lt.u32 %p21, %r23, %r2;
                    and.pred %p21, %p21, %p20;
                    mov.f32 %sv0, 0f3F800000;
                    mul.wide.u32 %rd5, %r23, 4;
                    add.u64 %rd5, %rd5, %rd3;
                    @%p21 ld.global.f32 %sv0, [%rd5];
                """);
        }

        for (int c = 0; c < 16; c++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %c{c}, 0f00000000;");
        }

        // The step: x's tile (rows i0 + xr and i0 + xr + 32, k0 + xc ... + 3), each value its own register.
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
                mov.u32 %r12, 0;
            TILE:
                setp.ge.u32 %p0, %r12, %r3;
                @%p0 bra TILES_DONE;
                add.u32 %r13, %r12, %r11;
                add.u32 %r14, %r9, %r10;
                setp.lt.u32 %p1, %r14, %r1;
                mul.wide.u32 %rd6, %r14, %r3;
                cvt.u64.u32 %rd7, %r13;
                add.u64 %rd6, %rd6, %rd7;
                shl.b64 %rd6, %rd6, 2;
                add.u64 %rd6, %rd6, %rd1;
                add.u32 %r15, %r14, 32;
                setp.lt.u32 %p2, %r15, %r1;
                mul.wide.u32 %rd8, %r15, %r3;
                add.u64 %rd8, %rd8, %rd7;
                shl.b64 %rd8, %rd8, 2;
                add.u64 %rd8, %rd8, %rd1;
            """);
        for (int q = 0; q < 4; q++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r24, %r13, {{q}};
                    setp.lt.u32 %p3, %r24, %r3;
                    and.pred %p4, %p3, %p1;
                    and.pred %p5, %p3, %p2;
                    mov.f32 %xv{{q}}, 0f00000000;
                    mov.f32 %xv{{q + 4}}, 0f00000000;
                    @%p4 ld.global.f32 %xv{{q}}, [%rd6+{{4 * q}}];
                    @%p5 ld.global.f32 %xv{{q + 4}}, [%rd8+{{4 * q}}];
                """);
        }

        for (int q = 0; q < 4; q++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    st.shared.f32 [%r16+{q * TypedTileRows * 4}], %xv{q};");
            s.AppendLine(CultureInfo.InvariantCulture, $"    st.shared.f32 [%r16+{q * TypedTileRows * 4 + 128}], %xv{q + 4};");
        }

        // W's tile, widened.
        if (!outputsMajor)
        {
            // Inputs-major: row k0 + wr, columns j0 + wc ... + 7, in words of PerWord values.
            int words = 8 / d.PerWord;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    add.u32 %r25, %r12, %r21;
                    setp.lt.u32 %p6, %r25, %r3;
                    add.u32 %r26, %r8, %r22;
                    mul.wide.u32 %rd9, %r25, %r4;
                    cvt.u64.u32 %rd10, %r26;
                    add.u64 %rd9, %rd9, %rd10;
                """);
            if (Shift(d.Bits).Length > 0)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    shl.b64 %rd9, %rd9, {Shift(d.Bits)};");
            }

            s.AppendLine("    add.u64 %rd9, %rd9, %rd2;");
            for (int w = 0; w < words; w++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        add.u32 %r27, %r26, {{w * d.PerWord}};
                        setp.lt.u32 %p7, %r27, %r4;
                        and.pred %p7, %p7, %p6;
                        mov.b32 %u{{w}}, 0;
                        @%p7 ld.global.u32 %u{{w}}, [%rd9+{{4 * w}}];
                    """);
            }

            for (int w = 0; w < words; w++)
            {
                WordValues(s, d, R("%u", w), [.. Enumerable.Range(w * d.PerWord, d.PerWord).Select(v => R("%v", v))]);
            }

            for (int v = 0; v < 8; v++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    @%p20 mul.rn.f32 %v{v}, %v{v}, %sv{v};");
            }

            s.AppendLine("    st.shared.v4.f32 [%r20], {%v0, %v1, %v2, %v3};");
            s.AppendLine("    st.shared.v4.f32 [%r20+16], {%v4, %v5, %v6, %v7};");
        }
        else if (d.PerValue)
        {
            // Outputs-major: row j0 + wcol, values k0 + wp ... + 7.
            int words = 8 / d.PerWord;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    add.u32 %r25, %r8, %r21;
                    setp.lt.u32 %p6, %r25, %r2;
                    add.u32 %r26, %r12, %r22;
                    mul.wide.u32 %rd9, %r25, %r3;
                    cvt.u64.u32 %rd10, %r26;
                    add.u64 %rd9, %rd9, %rd10;
                """);
            if (Shift(d.Bits).Length > 0)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    shl.b64 %rd9, %rd9, {Shift(d.Bits)};");
            }

            s.AppendLine("    add.u64 %rd9, %rd9, %rd2;");
            for (int w = 0; w < words; w++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        add.u32 %r27, %r26, {{w * d.PerWord}};
                        setp.lt.u32 %p7, %r27, %r3;
                        and.pred %p7, %p7, %p6;
                        mov.b32 %u{{w}}, 0;
                        @%p7 ld.global.u32 %u{{w}}, [%rd9+{{4 * w}}];
                    """);
            }

            for (int w = 0; w < words; w++)
            {
                WordValues(s, d, R("%u", w), [.. Enumerable.Range(w * d.PerWord, d.PerWord).Select(v => R("%v", v))]);
            }

            for (int v = 0; v < 8; v++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    @%p20 mul.rn.f32 %v{v}, %v{v}, %sv0;");
                s.AppendLine(CultureInfo.InvariantCulture, $"    st.shared.f32 [%r20+{v * TypedTileColumns * 4}], %v{v};");
            }
        }
        else
        {
            // A block format: the step is one block of row j0 + wcol; this thread's 8 codes from byte CodeOffset + wp.
            var block = d.Block!;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    add.u32 %r25, %r8, %r21;
                    setp.lt.u32 %p6, %r25, %r2;
                    shr.u32 %r27, %r3, 5;
                    shr.u32 %r28, %r12, 5;
                    mul.wide.u32 %rd9, %r25, %r27;
                    cvt.u64.u32 %rd10, %r28;
                    add.u64 %rd9, %rd9, %rd10;
                    mul.lo.u64 %rd9, %rd9, {block.Bytes};
                    add.u64 %rd9, %rd9, %rd2;
                    mov.b32 %u6, 0;
                    @%p6 ld.global.u16 %u6, [%rd9+{block.ScaleOffset}];
                """);
            FloatFrom(s, "%u6", "%sc", block.Scale);
            if (d.Nibbles)
            {
                NibbleAt(s, d, "%r22");
            }

            s.AppendLine(d.Nibbles ? "    cvt.u64.u32 %rd11, %nb;" : "    cvt.u64.u32 %rd11, %r22;");
            s.AppendLine("    add.u64 %rd11, %rd11, %rd9;");
            for (int unit = 0; unit < 4; unit++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    mov.b32 %u{unit}, 0;");
                s.AppendLine(CultureInfo.InvariantCulture, $"    @%p6 ld.global.u16 %u{unit}, [%rd11+{block.CodeOffset + 2 * unit}];");
            }

            for (int unit = 0; unit < 4; unit++)
            {
                UnitCodes(s, d, R("%u", unit), "%sc", R("%v", 2 * unit), R("%v", 2 * unit + 1));
            }

            for (int v = 0; v < 8; v++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    st.shared.f32 [%r20+{v * TypedTileColumns * 4}], %v{v};");
            }
        }

        s.AppendLine("    bar.sync 0;");
        for (int kk = 0; kk < TypedTileDepth; kk++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v4.f32 {{%a0, %a1, %a2, %a3}}, [%r18+{kk * TypedTileRows * 4}];");
            s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.v4.f32 {{%b0, %b1, %b2, %b3}}, [%r19+{kk * TypedTileColumns * 4}];");
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < 4; j++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %c{i * 4 + j}, %a{i}, %b{j}, %c{i * 4 + j};");
                }
            }
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
                bar.sync 0;
                add.u32 %r12, %r12, {TypedTileDepth};
                bra TILE;
            TILES_DONE:
                shl.b32 %r29, %r7, 2;
                add.u32 %r29, %r29, %r9;
                shl.b32 %r30, %r6, 2;
                add.u32 %r30, %r30, %r8;
            """);
        for (int i = 0; i < 4; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r31, %r29, {{i}};
                    setp.lt.u32 %p8, %r31, %r1;
                    mul.wide.u32 %rd12, %r31, %r2;
                    cvt.u64.u32 %rd13, %r30;
                    add.u64 %rd12, %rd12, %rd13;
                    shl.b64 %rd12, %rd12, 2;
                    add.u64 %rd12, %rd12, %rd4;
                """);
            for (int j = 0; j < 4; j++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        add.u32 %r32, %r30, {{j}};
                        setp.lt.u32 %p9, %r32, %r2;
                        and.pred %p9, %p9, %p8;
                        @%p9 st.global.f32 [%rd12+{{4 * j}}], %c{{i * 4 + j}};
                    """);
            }
        }

        s.AppendLine("    ret;");
        s.AppendLine("}");
        sb.Append(s);
        sb.AppendLine();
    }

    // ------------------------------------------------------------------ the few-row products

    // Inputs-major W (k rows of ld values), `rows` rows of x at most (p_m ≤ rows): each thread one word of PerWord columns,
    // each block row chunk p_chunk of k (block y); its sums written to y when p_splits is 1, else to its part of p_part
    // ([splits, m, n]), which typed_sum_parts adds in split order. Each part one FMA chain over its rows in order, four rows
    // a pass with every load its own register.
    private static void TypedGemvInputs(StringBuilder sb, TypedDecode d, int rows)
    {
        int per = d.PerWord, accumulators = rows * per;
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{TypedGemvName(d, false, rows)}}(
                .param .u64 p_x, .param .u64 p_w, .param .u64 p_s, .param .u64 p_y, .param .u64 p_part,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .u32 p_ld, .param .u32 p_chunk, .param .u32 p_splits
            )
            .maxntid {{TypedGemvThreads}}, 1, 1
            {
                .reg .pred %p<24>;
                .reg .b32 %r<32>;
                .reg .b32 %u<4>;
                .reg .f32 %acc<{{accumulators}}>;
                .reg .f32 %x<{{4 * rows}}>;
                .reg .f32 %v<{{4 * per}}>;
                .reg .f32 %sv<{{per}}>;
                .reg .b64 %rd<{{24 + rows}}>;
            {{WideningRegisters}}
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_w];
                ld.param.u64 %rd3, [p_s];
                ld.param.u64 %rd4, [p_y];
                ld.param.u64 %rd5, [p_part];
                setp.ne.u64 %p20, %rd3, 0;
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                cvta.to.global.u64 %rd5, %rd5;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                ld.param.u32 %r4, [p_ld];
                mov.u32 %r5, %ctaid.x;
                mov.u32 %r6, %tid.x;
                mad.lo.u32 %r7, %r5, {{TypedGemvThreads}}, %r6;
                shr.u32 %r8, %r4, {{(int)Math.Log2(per)}};
                setp.ge.u32 %p0, %r7, %r8;
                @%p0 bra DONE;
                shl.b32 %r9, %r7, {{(int)Math.Log2(per)}};
                mov.u32 %r10, %ctaid.y;
                ld.param.u32 %r11, [p_chunk];
                mul.lo.u32 %r12, %r10, %r11;
                add.u32 %r13, %r12, %r11;
                min.u32 %r13, %r13, %r3;
                cvt.u64.u32 %rd10, %r4;
            """);
        if (Shift(d.Bits).Length > 0)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    shl.b64 %rd10, %rd10, {Shift(d.Bits)};");
        }

        // The column scales, read once; the first row's word and each row of x.
        for (int v = 0; v < per; v++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r14, %r9, {{v}};
                    setp.lt.u32 %p21, %r14, %r2;
                    and.pred %p21, %p21, %p20;
                    mov.f32 %sv{{v}}, 0f3F800000;
                    mul.wide.u32 %rd6, %r14, 4;
                    add.u64 %rd6, %rd6, %rd3;
                    @%p21 ld.global.f32 %sv{{v}}, [%rd6];
                """);
        }

        s.AppendLine(CultureInfo.InvariantCulture, $"""
                mul.wide.u32 %rd11, %r12, %r4;
                cvt.u64.u32 %rd6, %r9;
                add.u64 %rd11, %rd11, %rd6;
            """);
        if (Shift(d.Bits).Length > 0)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    shl.b64 %rd11, %rd11, {Shift(d.Bits)};");
        }

        s.AppendLine("    add.u64 %rd11, %rd11, %rd2;");
        for (int i = 0; i < rows; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    setp.gt.u32 %p{{4 + i}}, %r1, {{i}};
                    mul.lo.u32 %r15, %r3, {{i}};
                    add.u32 %r15, %r15, %r12;
                    mul.wide.u32 %rd{{24 + i}}, %r15, 4;
                    add.u64 %rd{{24 + i}}, %rd{{24 + i}}, %rd1;
                """);
        }

        for (int a = 0; a < accumulators; a++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %acc{a}, 0f00000000;");
        }

        // Four rows a pass, then one at a time.
        foreach (int step in new[] { 4, 1 })
        {
            string label = step == 4 ? "PASS4" : "PASS1", next = step == 4 ? "PASS1" : "WRITE";
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                {label}:
                    add.u32 %r16, %r12, {step};
                    setp.gt.u32 %p1, %r16, %r13;
                    @%p1 bra {next};
                    mov.u64 %rd12, %rd11;
                """);
            for (int u = 0; u < step; u++)
            {
                if (u > 0)
                {
                    s.AppendLine("    add.u64 %rd12, %rd12, %rd10;");
                }

                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.global.u32 %u{u}, [%rd12];");
            }

            for (int i = 0; i < rows; i++)
            {
                for (int u = 0; u < step; u++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %x{i * 4 + u}, 0f00000000;");
                    s.AppendLine(CultureInfo.InvariantCulture, $"    @%p{4 + i} ld.global.f32 %x{i * 4 + u}, [%rd{24 + i}+{4 * u}];");
                }
            }

            for (int u = 0; u < step; u++)
            {
                WordValues(s, d, R("%u", u), [.. Enumerable.Range(u * per, per).Select(v => R("%v", v))]);
                for (int v = 0; v < per; v++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    @%p20 mul.rn.f32 %v{u * per + v}, %v{u * per + v}, %sv{v};");
                }
            }

            for (int u = 0; u < step; u++)
            {
                for (int i = 0; i < rows; i++)
                {
                    for (int v = 0; v < per; v++)
                    {
                        s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %acc{i * per + v}, %x{i * 4 + u}, %v{u * per + v}, %acc{i * per + v};");
                    }
                }
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    add.u64 %rd12, %rd12, %rd10;
                    mov.u64 %rd11, %rd12;
                    add.u32 %r12, %r12, {step};
                """);
            for (int i = 0; i < rows; i++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    add.u64 %rd{24 + i}, %rd{24 + i}, {4 * step};");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"    bra {label};");
        }

        // y, or this split's part: [split, m, n].
        s.AppendLine(CultureInfo.InvariantCulture, $"""
            WRITE:
                ld.param.u32 %r17, [p_splits];
                setp.eq.u32 %p2, %r17, 1;
                mul.lo.u32 %r18, %r1, %r2;
                mul.wide.u32 %rd13, %r18, %r10;
                shl.b64 %rd13, %rd13, 2;
                add.u64 %rd13, %rd13, %rd5;
                selp.b64 %rd13, %rd4, %rd13, %p2;
            """);
        for (int i = 0; i < rows; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    mul.lo.u32 %r19, %r2, {i};
                    add.u32 %r19, %r19, %r9;
                    mul.wide.u32 %rd14, %r19, 4;
                    add.u64 %rd14, %rd14, %rd13;
                """);
            for (int v = 0; v < per; v++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $$"""
                        add.u32 %r20, %r9, {{v}};
                        setp.lt.u32 %p3, %r20, %r2;
                        and.pred %p3, %p3, %p{{4 + i}};
                        @%p3 st.global.f32 [%rd14+{{4 * v}}], %acc{{i * per + v}};
                    """);
            }
        }

        s.AppendLine("""
            DONE:
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }

    // The grouped few-row product's further weights (TypedGemvOutputs with many): weight, scales, output and rows of the
    // second and third products (the third's rows 0 for two).
    private const string ManyParameters = ",\n    .param .u64 p_w1, .param .u64 p_s1, .param .u64 p_y1, .param .u32 p_n1, .param .u64 p_w2, .param .u64 p_s2, .param .u64 p_y2, .param .u32 p_n2";

    // The grouped few-row product's own registers (declared with the kernel's).
    private const string ManyRegisters = ".reg .b64 %mw<3>; .reg .b32 %mn<1>;";

    // The column of the grouped products (%r7, over all their rows) to its product's own: past the first product's rows,
    // the next product's weight, scales, output and rows replace the first's (%rd2, %rd3 and %p20, %rd4, %r2), twice.
    private const string ManyRemap = """
            ld.param.u64 %mw0, [p_w1];
            ld.param.u64 %mw1, [p_s1];
            ld.param.u64 %mw2, [p_y1];
            ld.param.u32 %mn0, [p_n1];
            setp.ge.u32 %p22, %r7, %r2;
            @%p22 sub.u32 %r7, %r7, %r2;
            @%p22 cvta.to.global.u64 %rd2, %mw0;
            @%p22 setp.ne.u64 %p20, %mw1, 0;
            @%p22 cvta.to.global.u64 %rd3, %mw1;
            @%p22 cvta.to.global.u64 %rd4, %mw2;
            @%p22 mov.u32 %r2, %mn0;
            ld.param.u64 %mw0, [p_w2];
            ld.param.u64 %mw1, [p_s2];
            ld.param.u64 %mw2, [p_y2];
            ld.param.u32 %mn0, [p_n2];
            setp.ge.u32 %p23, %r7, %r2;
            and.pred %p23, %p23, %p22;
            @%p23 sub.u32 %r7, %r7, %r2;
            @%p23 cvta.to.global.u64 %rd2, %mw0;
            @%p23 setp.ne.u64 %p20, %mw1, 0;
            @%p23 cvta.to.global.u64 %rd3, %mw1;
            @%p23 cvta.to.global.u64 %rd4, %mw2;
            @%p23 mov.u32 %r2, %mn0;
        """;

    // Outputs-major W (n rows of k values; p_s a scale per row or 0), `rows` rows of x at most: a warp a column (8 a
    // block), its lanes along the row: a float's or an integer's words lane by lane, a block format's blocks a
    // quarter warp each (each lane 4 codes). Each lane one FMA chain, then the warp's lanes added by a butterfly of a fixed
    // shape; lane 0 writes the column.
    private static void TypedGemvOutputs(StringBuilder sb, TypedDecode d, int rows, bool wide = false, bool many = false, Epilogue epilogue = Epilogue.None)
    {
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{TypedGemvName(d, true, rows, wide)}}{{(many ? "_many" : "")}}{{EpilogueSuffix(epilogue)}}(
                .param .u64 p_x, .param .u64 p_w, .param .u64 p_s, .param .u64 p_y,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k{{(many ? ManyParameters : "")}}{{EpilogueParameters(epilogue)}}
            )
            .maxntid {{TypedWarpColumns * 32}}, 1, 1
            {
                .reg .pred %p<24>;
                .reg .b32 %r<32>;
                .reg .b32 %u<4>;
                .reg .f32 %acc<{{rows}}>;
                .reg .f32 %t<{{rows}}>;
                .reg .f32 %x<16>;
                .reg .f32 %v<16>;
                .reg .f32 %sv;
                .reg .f32 %sc;
                .reg .b64 %rd<{{24 + rows}}>;
                {{(many ? ManyRegisters : "")}}
                {{(epilogue == Epilogue.None ? "" : EpilogueRegisters)}}
            {{WideningRegisters}}
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_w];
                ld.param.u64 %rd3, [p_s];
                ld.param.u64 %rd4, [p_y];
                setp.ne.u64 %p20, %rd3, 0;
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                mov.u32 %r4, %tid.x;
                shr.u32 %r5, %r4, 5;
                and.b32 %r6, %r4, 31;
                mov.u32 %r7, %ctaid.x;
                mad.lo.u32 %r7, %r7, {{TypedWarpColumns}}, %r5;
            {{(many ? ManyRemap : "")}}
                setp.ge.u32 %p0, %r7, %r2;
                @%p0 bra DONE;
                mov.f32 %sv, 0f3F800000;
                mul.wide.u32 %rd6, %r7, 4;
                add.u64 %rd6, %rd6, %rd3;
                @%p20 ld.global.f32 %sv, [%rd6];
            """);
        for (int i = 0; i < rows; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    mov.f32 %acc{{i}}, 0f00000000;
                    setp.gt.u32 %p{{4 + i}}, %r1, {{i}};
                    mul.lo.u32 %r15, %r3, {{i}};
                    mul.wide.u32 %rd{{24 + i}}, %r15, 4;
                    add.u64 %rd{{24 + i}}, %rd{{24 + i}}, %rd1;
                """);
        }

        if (d.PerValue)
        {
            // Word e of the row (PerWord values from value e · PerWord), lanes e = lane, lane + 32, ...; wide: words 4e ... 4e
            // + 3 by one 16-byte load (rows of whole 16-byte groups), lanes e = lane, lane + 32, ...
            int per = d.PerWord, words = wide ? 4 : 1, values = words * per;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shr.u32 %r8, %r3, {(int)Math.Log2(per)};
                    mul.wide.u32 %rd11, %r7, %r8;
                    shl.b64 %rd11, %rd11, 2;
                    add.u64 %rd11, %rd11, %rd2;
                    shl.b32 %r9, %r6, {(int)Math.Log2(words)};
                WORD:
                    setp.ge.u32 %p1, %r9, %r8;
                    @%p1 bra REDUCE;
                    mul.wide.u32 %rd12, %r9, 4;
                    add.u64 %rd12, %rd12, %rd11;
                """);
            s.AppendLine(wide ? "    ld.global.v4.u32 {%u0, %u1, %u2, %u3}, [%rd12];" : "    ld.global.u32 %u0, [%rd12];");
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shl.b32 %r10, %r9, {(int)Math.Log2(per)};
                    mul.wide.u32 %rd13, %r10, 4;
                """);
            for (int w = 0; w < words; w++)
            {
                WordValues(s, d, R("%u", w), [.. Enumerable.Range(w * per, per).Select(v => R("%v", v))]);
            }

            for (int i = 0; i < rows; i++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    add.u64 %rd14, %rd13, %rd{24 + i};");
                if (wide)
                {
                    for (int q = 0; q < values / 4; q++)
                    {
                        s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %x{4 * q}, 0f00000000;");
                        s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %x{4 * q + 1}, 0f00000000;");
                        s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %x{4 * q + 2}, 0f00000000;");
                        s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %x{4 * q + 3}, 0f00000000;");
                        s.AppendLine(CultureInfo.InvariantCulture, $"    @%p{4 + i} ld.global.v4.f32 {{%x{4 * q}, %x{4 * q + 1}, %x{4 * q + 2}, %x{4 * q + 3}}}, [%rd14+{16 * q}];");
                    }
                }
                else
                {
                    for (int v = 0; v < per; v++)
                    {
                        s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %x{v}, 0f00000000;");
                        s.AppendLine(CultureInfo.InvariantCulture, $"    @%p{4 + i} ld.global.f32 %x{v}, [%rd14+{4 * v}];");
                    }
                }

                for (int v = 0; v < values; v++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %acc{i}, %x{v}, %v{v}, %acc{i};");
                }
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    add.u32 %r9, %r9, {32 * words};
                    bra WORD;
                """);
        }
        else
        {
            // Block b of the row: lanes 8b' ... 8b' + 7 of a quarter warp, each its 4 codes from byte CodeOffset + 4 · sub.
            var block = d.Block!;
            int lanesPerBlock = block.Values / 4, blocksPerStep = 32 / lanesPerBlock;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shr.u32 %r8, %r3, {(int)Math.Log2(block.Values)};
                    mul.wide.u32 %rd11, %r7, %r8;
                    mul.lo.u64 %rd11, %rd11, {block.Bytes};
                    add.u64 %rd11, %rd11, %rd2;
                    shr.u32 %r9, %r6, {(int)Math.Log2(lanesPerBlock)};
                    and.b32 %r11, %r6, {lanesPerBlock - 1};
                    shl.b32 %r11, %r11, 2;
                {(d.Nibbles ? $"    and.b32 %nb, %r11, {block.Values / 2 - 1};\n    shr.u32 %ns, %r11, {(int)Math.Log2(block.Values / 2)};\n    shl.b32 %ns, %ns, 2;" : "    mov.u32 %nb, %r11;")}
                BLOCK:
                    setp.ge.u32 %p1, %r9, %r8;
                    @%p1 bra REDUCE;
                    mul.wide.u32 %rd12, %r9, {block.Bytes};
                    add.u64 %rd12, %rd12, %rd11;
                    ld.global.u16 %u2, [%rd12+{block.ScaleOffset}];
                    cvt.u64.u32 %rd15, %nb;
                    add.u64 %rd15, %rd15, %rd12;
                    ld.global.u16 %u0, [%rd15+{block.CodeOffset}];
                    ld.global.u16 %u1, [%rd15+{block.CodeOffset + 2}];
                """);
            FloatFrom(s, "%u2", "%sc", block.Scale);
            UnitCodes(s, d, "%u0", "%sc", "%v0", "%v1");
            UnitCodes(s, d, "%u1", "%sc", "%v2", "%v3");
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shl.b32 %r10, %r9, {(int)Math.Log2(block.Values)};
                    add.u32 %r10, %r10, %r11;
                    mul.wide.u32 %rd13, %r10, 4;
                """);
            for (int i = 0; i < rows; i++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    add.u64 %rd14, %rd13, %rd{24 + i};");
                for (int q = 0; q < 4; q++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %x{q}, 0f00000000;");
                    s.AppendLine(CultureInfo.InvariantCulture, $"    @%p{4 + i} ld.global.f32 %x{q}, [%rd14+{4 * q}];");
                }

                for (int q = 0; q < 4; q++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %acc{i}, %x{q}, %v{q}, %acc{i};");
                }
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    add.u32 %r9, %r9, {blocksPerStep};
                    bra BLOCK;
                """);
        }

        s.AppendLine("REDUCE:");
        foreach (int offset in new[] { 16, 8, 4, 2, 1 })
        {
            for (int i = 0; i < rows; i++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    shfl.sync.bfly.b32 %t{i}, %acc{i}, {offset}, 31, 0xffffffff;");
                s.AppendLine(CultureInfo.InvariantCulture, $"    add.rn.f32 %acc{i}, %acc{i}, %t{i};");
            }
        }

        s.AppendLine("""
                setp.ne.u32 %p2, %r6, 0;
                @%p2 bra DONE;
            """);
        for (int i = 0; i < rows; i++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    @%p20 mul.rn.f32 %acc{i}, %acc{i}, %sv;");
            s.AppendLine(CultureInfo.InvariantCulture, $$"""
                    mul.lo.u32 %r12, %r2, {{i}};
                    add.u32 %r12, %r12, %r7;
                    mul.wide.u32 %rd16, %r12, 4;
                    add.u64 %rd16, %rd16, %rd4;
                    @%p{{4 + i}} st.global.f32 [%rd16], %acc{{i}};
                """);
        }

        s.AppendLine("DONE:");
        s.Append(EpilogueCode(epilogue));
        s.AppendLine("""
                ret;
            }
            """);
        sb.Append(s);
        sb.AppendLine();
    }

    internal static string TypedMmaName(TypedDecode d) => string.Create(CultureInfo.InvariantCulture, $"typed_mma_{d.Code}_n");

    /// <summary>The matrix-unit lane's module of <paramref name="d"/> (PTX 7.0 for the bfloat16 tensor cores: TensorCoreTarget).</summary>
    internal static (string[] Names, string Source) TypedMmaModule(TypedDecode d)
    {
        var sb = new StringBuilder();
        sb.AppendLine(".version 7.0");
        sb.AppendLine(CultureInfo.InvariantCulture, $".target {TensorCoreTarget}");
        sb.AppendLine(".address_size 64");
        sb.AppendLine();
        TypedMma(sb, d);
        return ([TypedMmaName(d)], sb.ToString());
    }

    // The matrix-unit lane (chosen by the type's description): y [m, n] = x [m, k] · W for W
    // outputs-major (n rows of k values), k a multiple of 32. A block of 128 threads (4 warps, 2 × 2) computes 64 × 64
    // outputs, each warp 32 × 32 as 2 × 4 tiles of mma.sync m16n8k16 (bfloat16 operands, float32 sums), k in steps of 32:
    // x's tile rounded to bfloat16 pairs, W's tile widened from the description into bfloat16 pairs (8-bit codes and
    // bfloat16 values exactly), both in shared memory rows of 40 halves (80 bytes: the fragments' reads fall in distinct
    // banks). A block format's step is one block: its two k16 products into partial sums, then each column's partial times
    // its block's scale added to the column's sum in float32. p_s: a scale per output row (column of y) times the sum, or 0.
    private static void TypedMma(StringBuilder sb, TypedDecode d)
    {
        const int Row = 80;                                                // bytes of a shared tile row (32 halves + 8 of padding)
        bool block = !d.PerValue;
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{TypedMmaName(d)}}(
                .param .u64 p_x, .param .u64 p_w, .param .u64 p_s, .param .u64 p_y,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k
            )
            .maxntid 128, 1, 1
            {
                .reg .pred %p<24>;
                .reg .b32 %r<48>;
                .reg .b32 %u<16>;
                .reg .b32 %h<16>;
                .reg .b32 %fa<8>;
                .reg .b32 %fb<8>;
                .reg .f32 %c<32>;
                .reg .f32 %pt<32>;
                .reg .f32 %xv<16>;
                .reg .f32 %v<16>;
                .reg .f32 %sc;
                .reg .f32 %s<2>;
                .reg .b64 %rd<16>;
            {{WideningRegisters}}
                .shared .align 16 .b8 typed_mx[{{64 * Row}}];
                .shared .align 16 .b8 typed_mw[{{64 * Row}}];
                .shared .align 16 .f32 typed_ms[64];
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_w];
                ld.param.u64 %rd3, [p_s];
                ld.param.u64 %rd4, [p_y];
                setp.ne.u64 %p20, %rd3, 0;
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                cvta.to.global.u64 %rd3, %rd3;
                cvta.to.global.u64 %rd4, %rd4;
                ld.param.u32 %r1, [p_m];
                ld.param.u32 %r2, [p_n];
                ld.param.u32 %r3, [p_k];
                mov.u32 %r5, %tid.x;
                shr.u32 %r6, %r5, 5;
                and.b32 %r7, %r5, 31;
                shr.u32 %r8, %r7, 2;
                and.b32 %r9, %r7, 3;
                shr.u32 %r10, %r6, 1;
                and.b32 %r11, %r6, 1;
                mov.u32 %r12, %ctaid.x;
                shl.b32 %r12, %r12, 6;
                mov.u32 %r13, %ctaid.y;
                shl.b32 %r13, %r13, 6;
                shr.u32 %r14, %r5, 1;
                and.b32 %r15, %r5, 1;
                mul.lo.u32 %r16, %r14, {{Row}};
                shl.b32 %r17, %r15, 5;
                add.u32 %r16, %r16, %r17;
                mov.u32 %r17, typed_mw;
                add.u32 %r17, %r17, %r16;
                mov.u32 %r18, typed_mx;
                add.u32 %r16, %r16, %r18;
                shl.b32 %r19, %r10, 5;
                add.u32 %r19, %r19, %r8;
                mul.lo.u32 %r19, %r19, {{Row}};
                shl.b32 %r20, %r9, 2;
                add.u32 %r19, %r19, %r20;
                add.u32 %r18, %r18, %r19;
                shl.b32 %r19, %r11, 5;
                add.u32 %r19, %r19, %r8;
                mul.lo.u32 %r19, %r19, {{Row}};
                add.u32 %r19, %r19, %r20;
                mov.u32 %r21, typed_mw;
                add.u32 %r19, %r19, %r21;
                shl.b32 %r20, %r11, 5;
                shl.b32 %r21, %r9, 1;
                add.u32 %r20, %r20, %r21;
                shl.b32 %r20, %r20, 2;
                mov.u32 %r21, typed_ms;
                add.u32 %r20, %r20, %r21;
                add.u32 %r22, %r13, %r14;
                setp.lt.u32 %p1, %r22, %r1;
                mul.wide.u32 %rd6, %r22, %r3;
                cvt.u64.u32 %rd7, %r15;
                shl.b64 %rd7, %rd7, 4;
                add.u64 %rd6, %rd6, %rd7;
                shl.b64 %rd6, %rd6, 2;
                add.u64 %rd6, %rd6, %rd1;
                add.u32 %r23, %r12, %r14;
                setp.lt.u32 %p2, %r23, %r2;
            """);
        // W's row base: per value, row j's bytes and this thread's 16 values; a block format, row j's first block.
        if (block)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shr.u32 %r24, %r3, 5;
                    mul.wide.u32 %rd8, %r23, %r24;
                    mul.lo.u64 %rd8, %rd8, {d.Block!.Bytes};
                    add.u64 %rd8, %rd8, %rd2;
                    cvt.u64.u32 %rd9, %r15;
                    shl.b64 %rd9, %rd9, 4;
                """);
        }
        else
        {
            int bytes = d.Bits / 8;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    mul.wide.u32 %rd8, %r23, %r3;
                    cvt.u64.u32 %rd9, %r15;
                    shl.b64 %rd9, %rd9, 4;
                    add.u64 %rd8, %rd8, %rd9;
                    mul.lo.u64 %rd8, %rd8, {bytes};
                    add.u64 %rd8, %rd8, %rd2;
                """);
        }

        for (int c = 0; c < 32; c++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %c{c}, 0f00000000;");
        }

        s.AppendLine("""
                mov.u32 %r30, 0;
            STEP:
                setp.ge.u32 %p0, %r30, %r3;
                @%p0 bra STEPS_DONE;
                cvt.u64.u32 %rd10, %r30;
                shl.b64 %rd11, %rd10, 2;
                add.u64 %rd11, %rd11, %rd6;
            """);
        // x's 16 values of row i0 + lr from k0 + 16 · lh, rounded to bfloat16 pairs (the first in the low half).
        for (int q = 0; q < 4; q++)
        {
            for (int e = 0; e < 4; e++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %xv{4 * q + e}, 0f00000000;");
            }

            s.AppendLine(CultureInfo.InvariantCulture, $"    @%p1 ld.global.v4.f32 {{%xv{4 * q}, %xv{4 * q + 1}, %xv{4 * q + 2}, %xv{4 * q + 3}}}, [%rd11+{16 * q}];");
        }

        for (int w = 0; w < 8; w++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    cvt.rn.bf16x2.f32 %h{w}, %xv{2 * w + 1}, %xv{2 * w};");
        }

        s.AppendLine("    st.shared.v4.b32 [%r16], {%h0, %h1, %h2, %h3};");
        s.AppendLine("    st.shared.v4.b32 [%r16+16], {%h4, %h5, %h6, %h7};");

        // W's 16 values of row j0 + lr from k0 + 16 · lh, widened into bfloat16 pairs.
        if (block)
        {
            var b = d.Block!;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shr.u32 %r25, %r30, 5;
                    mul.wide.u32 %rd12, %r25, {b.Bytes};
                    add.u64 %rd12, %rd12, %rd8;
                    mov.b32 %u8, 0;
                    @%p2 ld.global.u16 %u8, [%rd12+{b.ScaleOffset}];
                """);
            FloatFrom(s, "%u8", "%sc", b.Scale);
            s.AppendLine("""
                    setp.eq.u32 %p3, %r15, 0;
                    shl.b32 %r26, %r14, 2;
                    mov.u32 %r27, typed_ms;
                    add.u32 %r26, %r26, %r27;
                    @%p3 st.shared.f32 [%r26], %sc;
                """);
            s.AppendLine(d.Nibbles ? "    mov.b64 %rd13, %rd12;\n    shl.b32 %ns, %r15, 2;" : "    add.u64 %rd13, %rd12, %rd9;");
            for (int unit = 0; unit < 8; unit++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    mov.b32 %u{unit}, 0;");
                s.AppendLine(CultureInfo.InvariantCulture, $"    @%p2 ld.global.u16 %u{unit}, [%rd13+{b.CodeOffset + 2 * unit}];");
            }

            string type = d.Signed ? "s32" : "u32";
            for (int unit = 0; unit < 8; unit++)
            {
                string from = R("%u", unit);
                if (d.Nibbles)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    shr.b32 %qn, %u{unit}, %ns;");
                    from = "%qn";
                }

                foreach (var (at, dest) in new[] { (0, 2 * unit), (8, 2 * unit + 1) })
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    bfe.{type} %q0, {from}, {at}, {d.Bits};");
                    TableEntry(s, d, "%q0");
                    if (b.CodeBias != 0)
                    {
                        s.AppendLine(CultureInfo.InvariantCulture, $"    add.s32 %q0, %q0, {b.CodeBias};");
                    }

                    s.AppendLine(CultureInfo.InvariantCulture, $"    cvt.rn.f32.s32 %v{dest}, %q0;");
                }

                s.AppendLine(CultureInfo.InvariantCulture, $"    cvt.rn.bf16x2.f32 %h{8 + unit}, %v{2 * unit + 1}, %v{2 * unit};");
            }
        }
        else
        {
            int bytes = d.Bits / 8, words = 4 * bytes;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    mul.lo.u64 %rd12, %rd10, {bytes};
                    add.u64 %rd12, %rd12, %rd8;
                """);
            for (int q = 0; q < bytes; q++)
            {
                for (int e = 0; e < 4; e++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    mov.b32 %u{4 * q + e}, 0;");
                }

                s.AppendLine(CultureInfo.InvariantCulture, $"    @%p2 ld.global.v4.u32 {{%u{4 * q}, %u{4 * q + 1}, %u{4 * q + 2}, %u{4 * q + 3}}}, [%rd12+{16 * q}];");
            }

            if (d.Kind == 'f' && d.Float is { Exponent: 8, Mantissa: 7 })
            {
                for (int w = 0; w < 8; w++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    mov.b32 %h{8 + w}, %u{w};");                   // bfloat16 pairs as stored
                }
            }
            else
            {
                for (int w = 0; w < words; w++)
                {
                    WordValues(s, d, R("%u", w), [.. Enumerable.Range(w * d.PerWord, d.PerWord).Select(v => R("%v", v))]);
                }

                for (int w = 0; w < 8; w++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    cvt.rn.bf16x2.f32 %h{8 + w}, %v{2 * w + 1}, %v{2 * w};");
                }
            }
        }

        s.AppendLine("""
                st.shared.v4.b32 [%r17], {%h8, %h9, %h10, %h11};
                st.shared.v4.b32 [%r17+16], {%h12, %h13, %h14, %h15};
                bar.sync 0;
            """);
        if (block)
        {
            for (int c = 0; c < 32; c++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %pt{c}, 0f00000000;");
            }
        }

        string target = block ? "%pt" : "%c";
        for (int ks = 0; ks < 2; ks++)
        {
            for (int mt = 0; mt < 2; mt++)
            {
                int at = mt * 16 * Row + ks * 32;
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.b32 %fa{4 * mt}, [%r18+{at}];");
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.b32 %fa{4 * mt + 1}, [%r18+{at + 8 * Row}];");
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.b32 %fa{4 * mt + 2}, [%r18+{at + 16}];");
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.b32 %fa{4 * mt + 3}, [%r18+{at + 8 * Row + 16}];");
            }

            for (int nt = 0; nt < 4; nt++)
            {
                int at = nt * 8 * Row + ks * 32;
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.b32 %fb{2 * nt}, [%r19+{at}];");
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.b32 %fb{2 * nt + 1}, [%r19+{at + 16}];");
            }

            for (int mt = 0; mt < 2; mt++)
            {
                for (int nt = 0; nt < 4; nt++)
                {
                    int c = (mt * 4 + nt) * 4;
                    s.AppendLine(CultureInfo.InvariantCulture,
                        $"    mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{target}{c}, {target}{c + 1}, {target}{c + 2}, {target}{c + 3}}}, {{%fa{4 * mt}, %fa{4 * mt + 1}, %fa{4 * mt + 2}, %fa{4 * mt + 3}}}, {{%fb{2 * nt}, %fb{2 * nt + 1}}}, {{{target}{c}, {target}{c + 1}, {target}{c + 2}, {target}{c + 3}}};");
                }
            }
        }

        if (block)
        {
            // Each column's partial sum times its block's scale, added to the column's sum.
            for (int nt = 0; nt < 4; nt++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.f32 %s0, [%r20+{32 * nt}];");
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.shared.f32 %s1, [%r20+{32 * nt + 4}];");
                for (int mt = 0; mt < 2; mt++)
                {
                    int c = (mt * 4 + nt) * 4;
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %c{c}, %pt{c}, %s0, %c{c};");
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %c{c + 1}, %pt{c + 1}, %s1, %c{c + 1};");
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %c{c + 2}, %pt{c + 2}, %s0, %c{c + 2};");
                    s.AppendLine(CultureInfo.InvariantCulture, $"    fma.rn.f32 %c{c + 3}, %pt{c + 3}, %s1, %c{c + 3};");
                }
            }
        }

        s.AppendLine("""
                bar.sync 0;
                add.u32 %r30, %r30, 32;
                bra STEP;
            STEPS_DONE:
            """);
        // y: tile (mt, nt)'s value e at row i0 + 32 · warp_m + 16 · mt + g (+ 8 for e ≥ 2), column j0 + 32 · warp_n + 8 · nt + 2 · tq (+ 1 for odd e).
        s.AppendLine("""
                shl.b32 %r31, %r10, 5;
                add.u32 %r31, %r31, %r8;
                add.u32 %r31, %r31, %r13;
                shl.b32 %r32, %r11, 5;
                shl.b32 %r33, %r9, 1;
                add.u32 %r32, %r32, %r33;
                add.u32 %r32, %r32, %r12;
            """);
        for (int nt = 0; nt < 4; nt++)
        {
            for (int half = 0; half < 2; half++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"""
                        add.u32 %r34, %r32, {8 * nt + half};
                        setp.lt.u32 %p4, %r34, %r2;
                        and.pred %p5, %p4, %p20;
                        mov.f32 %s{half}, 0f3F800000;
                        mul.wide.u32 %rd14, %r34, 4;
                        add.u64 %rd14, %rd14, %rd3;
                        @%p5 ld.global.f32 %s{half}, [%rd14];
                    """);
            }

            for (int mt = 0; mt < 2; mt++)
            {
                for (int e = 0; e < 4; e++)
                {
                    int c = (mt * 4 + nt) * 4 + e;
                    s.AppendLine(CultureInfo.InvariantCulture, $"""
                            add.u32 %r35, %r31, {16 * mt + (e >= 2 ? 8 : 0)};
                            add.u32 %r36, %r32, {8 * nt + (e & 1)};
                            setp.lt.u32 %p6, %r35, %r1;
                            setp.lt.u32 %p7, %r36, %r2;
                            and.pred %p6, %p6, %p7;
                            @%p20 mul.rn.f32 %c{c}, %c{c}, %s{e & 1};
                            mul.wide.u32 %rd15, %r35, %r2;
                            cvt.u64.u32 %rd14, %r36;
                            add.u64 %rd15, %rd15, %rd14;
                            shl.b64 %rd15, %rd15, 2;
                            add.u64 %rd15, %rd15, %rd4;
                            @%p6 st.global.f32 [%rd15], %c{c};
                        """);
                }
            }
        }

        s.AppendLine("    ret;");
        s.AppendLine("}");
        sb.Append(s);
        sb.AppendLine();
    }

    // y[e] = part[0, e] + part[1, e] + ... in split order (a few-row product's splits, [splits, count]).
    private static void TypedSum(StringBuilder sb) => sb.AppendLine(CultureInfo.InvariantCulture, $$"""
        .visible .entry {{TypedSumName}}(
            .param .u64 p_part, .param .u64 p_y, .param .u32 p_count, .param .u32 p_splits
        )
        .maxntid 256, 1, 1
        {
            .reg .pred %p<4>;
            .reg .b32 %r<8>;
            .reg .f32 %f<4>;
            .reg .b64 %rd<8>;
            ld.param.u64 %rd1, [p_part];
            ld.param.u64 %rd2, [p_y];
            cvta.to.global.u64 %rd1, %rd1;
            cvta.to.global.u64 %rd2, %rd2;
            ld.param.u32 %r1, [p_count];
            ld.param.u32 %r2, [p_splits];
            mov.u32 %r3, %ctaid.x;
            mov.u32 %r4, %tid.x;
            mad.lo.u32 %r3, %r3, 256, %r4;
            setp.ge.u32 %p0, %r3, %r1;
            @%p0 bra DONE;
            mul.wide.u32 %rd3, %r3, 4;
            add.u64 %rd4, %rd3, %rd1;
            mul.wide.u32 %rd5, %r1, 4;
            ld.global.f32 %f1, [%rd4];
            mov.u32 %r5, 1;
        SPLIT:
            setp.ge.u32 %p1, %r5, %r2;
            @%p1 bra STORE;
            add.u64 %rd4, %rd4, %rd5;
            ld.global.f32 %f2, [%rd4];
            add.rn.f32 %f1, %f1, %f2;
            add.u32 %r5, %r5, 1;
            bra SPLIT;
        STORE:
            add.u64 %rd6, %rd3, %rd2;
            st.global.f32 [%rd6], %f1;
        DONE:
            ret;
        }

        """);
}

/// <summary>
/// How CUDA's described lane reads a type: its <see cref="TypeDescription"/> (the description the CPU's described lane
/// reads, <see cref="ElementType.Description"/>), compiled once into what the generated kernels decode, never the type's
/// name. A row's values are read a pattern block at a time (<see cref="PatternValues"/>: whole blocks, a multiple of 16
/// values); in each step of 16 values thread t of a group of four (t = 0 .. 3) reads values 2t, 2t + 1, 2t + 8 and 2t + 9
/// (the k positions of an mma.sync m16n8k16 fragment, so the tensor-core product reads x as stored and the rows product
/// reads x in pairs): each value's code from its planes' fields at thread 0's byte and bit plus the thread's own offset
/// of the plane (<see cref="DeltaBytes"/>, <see cref="DeltaShifts"/>), its scales and minimums (one group, the same for
/// the four threads) from their bits. Every value is (number + CodeOffset) · the scales' product ∓ the minimums', its
/// number by its kind (an integer, a float by its bits, a table's entry); a 64-bit code (or an integer of more than 24
/// bits) in float64, never narrowed. Null, with the reason, where the description does not read so.
/// </summary>
internal sealed class DescribedPlan
{
    /// <summary>How a code's or a factor's bits become a number in the kernels.</summary>
    internal enum Read : byte
    {
        /// <summary>A signed integer of at most 24 bits (exact in float32).</summary>
        Signed,

        /// <summary>An unsigned integer of at most 24 bits.</summary>
        Unsigned,

        /// <summary>A float of 8 exponent bits, IEEE's: its bits are float32's high bits (bfloat16).</summary>
        High,

        /// <summary>IEEE 754's binary16 (float16).</summary>
        Half,

        /// <summary>A code of at most 8 bits through a table of its numbers (in shared memory).</summary>
        Table,

        /// <summary>float32 as stored.</summary>
        Float32,

        /// <summary>A signed integer of 25 to 32 bits, in float64.</summary>
        WideSigned,

        /// <summary>An unsigned integer of 25 to 32 bits, in float64.</summary>
        WideUnsigned,

        /// <summary>A signed 64-bit integer, in float64.</summary>
        Long,

        /// <summary>An unsigned 64-bit integer, in float64.</summary>
        ULong,

        /// <summary>IEEE 754's binary64 (float64) as stored.</summary>
        Double,
    }

    /// <summary>A field of a value's code: plane <paramref name="Plane"/>'s <paramref name="Bits"/> bits at thread 0's byte (from the pattern block's first) and bit, landing at the code's bit <paramref name="Low"/>.</summary>
    internal readonly record struct Field(int Plane, int Byte, int Shift, int Bits, int Low);

    /// <summary>A piece of a factor's bits: <paramref name="Bits"/> bits at bit <paramref name="Shift"/> of byte <paramref name="Byte"/> (from the pattern block's first), landing at bit <paramref name="Low"/>.</summary>
    internal readonly record struct Piece(int Byte, int Shift, int Bits, int Low);

    /// <summary>A value a thread reads in a step: its code's fields and its factors' group.</summary>
    internal sealed record Slot(Field[] Fields, int Group);

    /// <summary>A scale or minimum of the description: how its bits read, its table (a <see cref="Read.Table"/>), its bits, whether a signed integer, its bias, whether a minimum.</summary>
    internal sealed record Factor(Read Read, double[]? Table, int Bits, bool Signed, double Bias, bool Min);

    /// <summary>The k positions of a step a thread reads, after its own 2t: the mma fragment's.</summary>
    internal static readonly int[] Columns = [0, 1, 8, 9];

    private DescribedPlan(TypeDescription description, string code, Read read, double[]? table, int patternValues, int patternBytes, int[][] deltaBytes,
        int[][] deltaShifts, Slot[] slots, Piece[][][] groups, Factor[] factors)
    {
        (Description, Code, CodeRead, CodeTable, PatternValues, PatternBytes, DeltaBytes, DeltaShifts, Slots, Groups, Factors) =
            (description, code, read, table, patternValues, patternBytes, deltaBytes, deltaShifts, slots, groups, factors);
        Wide = read is Read.WideSigned or Read.WideUnsigned or Read.Long or Read.ULong or Read.Double;
    }

    /// <summary>The description read.</summary>
    public TypeDescription Description { get; }

    /// <summary>The description as a kernel's name part: a stable hash of its canonical text (the same description, the same kernels).</summary>
    public string Code { get; }

    /// <summary>How a value's code becomes a number.</summary>
    public Read CodeRead { get; }

    /// <summary>The numbers of a <see cref="Read.Table"/> code, or null.</summary>
    public double[]? CodeTable { get; }

    /// <summary>Whether the values multiply in float64 (a 64-bit code, an integer of more than 24 bits): never narrowed to float32.</summary>
    public bool Wide { get; }

    /// <summary>Values a pattern block holds: whole blocks, a multiple of 16.</summary>
    public int PatternValues { get; }

    /// <summary>Bytes a pattern block takes.</summary>
    public int PatternBytes { get; }

    /// <summary>Steps of 16 values a pattern block holds.</summary>
    public int Steps => PatternValues / 16;

    /// <summary>Each plane's byte offset of thread t's fields from thread 0's ([plane][t]).</summary>
    public int[][] DeltaBytes { get; }

    /// <summary>Each plane's bit offset of thread t's fields from thread 0's, within the bytes ([plane][t]).</summary>
    public int[][] DeltaShifts { get; }

    /// <summary>The values a thread reads: step s's at s · 4 + i, i for its 2t + <see cref="Columns"/>[i].</summary>
    public Slot[] Slots { get; }

    /// <summary>Each factor group's pieces: [group][factor][piece].</summary>
    public Piece[][][] Groups { get; }

    /// <summary>The description's scales, then its minimums.</summary>
    public Factor[] Factors { get; }

    /// <summary>The largest bit offset of a plane's fields over the four threads.</summary>
    public int MaxDeltaShift(int plane) => DeltaShifts[plane].Max();

    /// <summary>The largest power of two (at most 16) dividing every thread's byte offset of a plane.</summary>
    public int DeltaAlign(int plane)
    {
        int align = 16;
        foreach (int d in DeltaBytes[plane])
        {
            while (align > 1 && d % align != 0)
            {
                align /= 2;
            }
        }

        return align;
    }

    /// <summary>The plan of <paramref name="d"/>, or null with <paramref name="reason"/> where its values do not read so.</summary>
    public static DescribedPlan? Of(TypeDescription d, out string? reason)
    {
        reason = null;
        if (!d.Valid)
        {
            reason = "the description does not hold together";
            return null;
        }

        if (ReadOf(d.Code, out var table) is not { } read)
        {
            reason = $"a code of {d.Code.Bits} bits read as {d.Code.Kind} has no reading here (a float of 9 to 63 bits other than float16, bfloat16 and float32, a table of more than 8 bits)";
            return null;
        }

        if (d.Code.Bits > 32 && !(d.Planes.Count == 1 && d.Code.Bits == 64))
        {
            reason = "a code of more than 32 bits in more than one plane";
            return null;
        }

        var factors = new List<Factor>();
        foreach (var (scale, min) in (d.Scales ?? []).Select(s => (s, false)).Concat((d.Mins ?? []).Select(s => (s, true))))
        {
            var factorRead = ReadOf(scale.Type, out var factorTable);
            if (factorRead is not (Read.Signed or Read.Unsigned or Read.High or Read.Half or Read.Table or Read.Float32))
            {
                reason = $"a scale of {scale.Type.Bits} bits read as {scale.Type.Kind} has no float32 reading here";
                return null;
            }

            factors.Add(new Factor(factorRead.Value, factorTable, scale.Type.Bits, scale.Type.Kind == NumberKind.Integer && scale.Type.Signed, scale.Bias, min));
        }

        int patternValues = d.Values / Gcd(d.Values, 16) * 16;
        long patternBytes = (long)(patternValues / d.Values) * d.Bytes;
        if (patternValues > 1024 || patternBytes > 2048)
        {
            reason = $"a pattern block of {patternValues} values takes {patternBytes} bytes (at most 1024 values in 2048 bytes)";
            return null;
        }

        int planes = d.Planes.Count, steps = patternValues / 16;
        var deltaBytes = new int[planes][];
        var deltaShifts = new int[planes][];
        for (int p = 0; p < planes; p++)
        {
            (deltaBytes[p], deltaShifts[p]) = (new int[4], new int[4]);
        }

        var slots = new Slot[steps * 4];
        var groups = new List<Piece[][]>();
        var groupOf = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int s = 0; s < steps; s++)
        {
            for (int i = 0; i < 4; i++)
            {
                Field[] fields = [];
                Piece[][] pieces = [];
                string key = "";
                for (int t = 0; t < 4; t++)
                {
                    int u = 16 * s + 2 * t + Columns[i], block = u / d.Values, v = u % d.Values;
                    var own = new Field[planes];
                    int low = 0;
                    for (int p = 0; p < planes; p++)
                    {
                        var plane = d.Planes[p];
                        var (at, shift) = plane.Place(v);
                        own[p] = new Field(p, block * d.Bytes + at, shift, plane.Bits, low);
                        low += plane.Bits;
                    }

                    var ownPieces = FactorPieces(d, block, v);
                    string ownKey = string.Join(";", ownPieces.Select(f => string.Join(",", f.Select(x => x.ToString()))));
                    if (t == 0)
                    {
                        (fields, pieces, key) = (own, ownPieces, ownKey);
                        continue;
                    }

                    if (ownKey != key)
                    {
                        reason = "a scale or minimum differs between the four threads' values of a step (a block of fewer than 16 values with factors)";
                        return null;
                    }

                    for (int p = 0; p < planes; p++)
                    {
                        int db = own[p].Byte - fields[p].Byte, ds = own[p].Shift - fields[p].Shift;
                        if (s == 0 && i == 0)
                        {
                            (deltaBytes[p][t], deltaShifts[p][t]) = (db, ds);
                        }
                        else if (deltaBytes[p][t] != db || deltaShifts[p][t] != ds)
                        {
                            reason = $"plane {p}'s fields do not follow one offset a thread (step {s}, value {Columns[i]}, thread {t})";
                            return null;
                        }
                    }
                }

                if (!groupOf.TryGetValue(key, out int group))
                {
                    group = groups.Count;
                    groups.Add(pieces);
                    groupOf[key] = group;
                }

                slots[s * 4 + i] = new Slot(fields, group);
            }
        }

        return new DescribedPlan(d, Canonical(d), read, table, patternValues, (int)patternBytes, deltaBytes, deltaShifts, slots, [.. groups], [.. factors]);
    }

    // How a number type's bits read in the kernels, with its table where it is read through one.
    private static Read? ReadOf(NumberType n, out double[]? table)
    {
        table = null;
        switch (n.Kind)
        {
            case NumberKind.Integer:
                return n.Bits switch
                {
                    <= 24 => n.Signed ? Read.Signed : Read.Unsigned,
                    <= 32 => n.Signed ? Read.WideSigned : Read.WideUnsigned,
                    64 => n.Signed ? Read.Long : Read.ULong,
                    _ => null,
                };
            case NumberKind.Float:
                bool ieee = n.Signed && n.Subnormals && n.Specials == FloatSpecials.Ieee;
                if (ieee && n.Exponent == 8 && n.ExponentBias == 127 && n.Mantissa <= 23 && n.Bits > 8)
                {
                    return n.Bits == 32 ? Read.Float32 : Read.High;
                }

                if (ieee && n is { Exponent: 5, Mantissa: 10, ExponentBias: 15 })
                {
                    return Read.Half;
                }

                if (ieee && n is { Exponent: 11, Mantissa: 52, ExponentBias: 1023 })
                {
                    return Read.Double;
                }

                break;
        }

        if (n.Bits > 8)
        {
            return null;
        }

        table = [.. Enumerable.Range(0, 1 << n.Bits).Select(c => n.Value((ulong)c))];
        return Read.Table;
    }

    // Each factor's pieces for value v of block `block` of a pattern block (the scales, then the minimums).
    private static Piece[][] FactorPieces(TypeDescription d, int block, int v)
    {
        var all = (d.Scales ?? []).Concat(d.Mins ?? []).ToList();
        var result = new Piece[all.Count][];
        for (int q = 0; q < all.Count; q++)
        {
            var scale = all[q];
            int index = v / scale.Values, origin = block * d.Bytes;
            var pieces = new List<Piece>();
            int low = 0;
            if (scale.Pieces is { } own)
            {
                foreach (var piece in own[index])
                {
                    pieces.Add(new Piece(origin + piece.Offset, piece.Shift, piece.Bits, low));
                    low += piece.Bits;
                }
            }
            else
            {
                long bit = 8L * scale.Offset + (long)index * scale.Type.Bits;
                for (int left = scale.Type.Bits; left > 0;)
                {
                    int shift = (int)(bit % 8), take = Math.Min(left, 8 - shift);
                    pieces.Add(new Piece(origin + (int)(bit / 8), shift, take, low));
                    (bit, left, low) = (bit + take, left - take, low + take);
                }
            }

            result[q] = [.. pieces];
        }

        return result;
    }

    // The description's canonical text, hashed (FNV-1a, 64 bits) into a name part.
    private static string Canonical(TypeDescription d)
    {
        static string Number(NumberType n) => string.Create(CultureInfo.InvariantCulture,
            $"{n.Bits}{n.Kind}{(n.Signed ? 's' : 'u')}e{n.Exponent}m{n.Mantissa}b{n.ExponentBias}{(n.Subnormals ? 'n' : 'x')}{n.Specials}[{string.Join(",", (n.Table ?? []).Select(e => e.ToString("R", CultureInfo.InvariantCulture)))}]");
        static string Scale(BlockScale s) => string.Create(CultureInfo.InvariantCulture,
            $"{Number(s.Type)}/{s.Values}/{s.Offset}/{s.Bias.ToString("R", CultureInfo.InvariantCulture)}/{string.Join("|", (s.Pieces ?? []).Select(p => string.Join(",", p.Select(x => string.Create(CultureInfo.InvariantCulture, $"{x.Offset}.{x.Shift}.{x.Bits}")))))}");
        string planes = string.Join(",", d.Planes.Select(p => string.Create(CultureInfo.InvariantCulture, $"{p.Offset}.{p.Bits}.{p.Group}")));
        string text = string.Create(CultureInfo.InvariantCulture,
            $"{d.Values}x{d.Bytes};{Number(d.Code)};{planes};{d.CodeOffset.ToString("R", CultureInfo.InvariantCulture)};{string.Join(";", (d.Scales ?? []).Select(Scale))};-{string.Join(";", (d.Mins ?? []).Select(Scale))};{d.MinsAdded}");
        ulong hash = 14695981039346656037UL;
        foreach (char c in text)
        {
            hash = (hash ^ c) * 1099511628211UL;
        }

        return hash.ToString("x16", CultureInfo.InvariantCulture);
    }

    internal static int Gcd(int a, int b) => b == 0 ? Math.Abs(a) : Gcd(b, a % b);
}

// The described lane's kernels (CudaBackend.Typed.cs: the measured route's candidate of every type a DescribedPlan reads),
// generated from the plan alone, one module per plan and alignment of its rows: the rows product (rows of x a block of
// 256 threads, 8 weight rows a block, a group of four lanes a weight row, the block's 8 warps over the rows' spans of
// pattern blocks: each warp's spans copied as stored into its shared memory by 16-byte cp.async, the next while this one
// is read, each value decoded in registers and multiplied with x's float32 values, or float64 ones for a wide plan; the
// parts added in a fixed order: the four lanes, then the warps) and the tensor-core product (a bfloat16 run's prompts:
// 64 × 64 outputs a block of 4 warps, each 32 × 32 as mma.sync m16n8k16 tiles; x packed to bfloat16 once and staged by
// cp.async, W's spans staged as stored; each B fragment's four values decoded in registers from the stored bytes and
// rounded to bfloat16 pairs; k split over the grid's z with the parts added in order). Nothing of a weight is written
// anywhere decoded: not in memory, not in shared memory.
internal static partial class PtxKernels
{
    /// <summary>The rows product's threads and weight rows a block.</summary>
    internal const int DescribedRowsThreads = 256, DescribedRowsBlock = 8;

    /// <summary>The rows of x the rows product is compiled for (more: the grid's y over chunks of the last).</summary>
    internal static ReadOnlySpan<int> DescribedRowCounts => [1, 2, 4, 8, 16];

    /// <summary>The tensor-core product's threads, and its tile of x rows and weight rows.</summary>
    internal const int DescribedMmaThreads = 128, DescribedMmaTile = 64;

    // x's staged rows in the tensor-core product: 64 bfloat16 values and 16 bytes of padding (ldmatrix rows in distinct banks).
    private const int DescribedXRow = 144, DescribedXStage = DescribedMmaTile * DescribedXRow;

    /// <summary>The rows product's span: pattern blocks a warp stages at once (about 128 bytes a row).</summary>
    internal static int DescribedSpan(DescribedPlan p) => 1 << (int)Math.Floor(Math.Log2(Math.Max(1, 128 / p.PatternBytes)));

    /// <summary>The tensor-core product's span of W: values staged at once (whole pattern blocks, a multiple of 64).</summary>
    internal static int DescribedMmaSpan(DescribedPlan p) => p.PatternValues / DescribedPlan.Gcd(p.PatternValues, 64) * 64;

    /// <summary>The module's name of a plan's kernels for rows aligned to <paramref name="align"/> bytes, x read in pairs or not.</summary>
    internal static string DescribedModuleName(DescribedPlan p, int align, bool pairs) =>
        string.Create(CultureInfo.InvariantCulture, $"described_{p.Code}_a{align}{(pairs ? "" : "_s")}");

    internal static string DescribedRowsName(DescribedPlan p, int align, bool pairs, int rows) =>
        DescribedModuleName(p, align, pairs) + string.Create(CultureInfo.InvariantCulture, $"_rows{rows}");

    internal static string DescribedMmaName(DescribedPlan p, int align) => DescribedModuleName(p, align, true) + "_mma";

    internal static string DescribedWidenName(DescribedPlan p, int align, bool pairs) => DescribedModuleName(p, align, pairs) + "_widen";

    // The plan's tables in shared memory (the code's, then each factor's), 4 bytes an entry, 16-byte aligned.
    private static int DescribedTablesBytes(DescribedPlan p) => (4 * ((p.CodeTable?.Length ?? 0) + p.Factors.Sum(f => f.Table?.Length ?? 0)) + 15) / 16 * 16;

    // A staged row of `bytes` bytes from an offset a multiple of `align`: whole 16-byte chunks from the window's start (the
    // offset at most 16 - align), 8 bytes past the last (a load wider than a field), an odd number of 16-byte units (8 rows'
    // reads in distinct banks).
    private static (int Chunks, int Stride) DescribedRow(int bytes, int align)
    {
        int chunks = ((align >= 16 ? 0 : 16 - align) + bytes + 15) / 16;
        int stride = (chunks * 16 + 8 + 15) / 16 * 16;
        if (stride / 16 % 2 == 0)
        {
            stride += 16;
        }

        return (chunks, stride);
    }

    /// <summary>Dynamic shared memory of the rows product of <paramref name="rows"/> rows of x.</summary>
    internal static int DescribedRowsShared(DescribedPlan p, int align, int rows) =>
        DescribedTablesBytes(p) + 8 * 2 * DescribedRowsBlock * DescribedRow(DescribedSpan(p) * p.PatternBytes, align).Stride + 8 * DescribedRowsBlock * rows * (p.Wide ? 8 : 4);

    /// <summary>Dynamic shared memory of the tensor-core product.</summary>
    internal static int DescribedMmaShared(DescribedPlan p, int align) =>
        DescribedTablesBytes(p) + 2 * DescribedXStage + 2 * DescribedMmaTile * DescribedRow(DescribedMmaSpan(p) / p.PatternValues * p.PatternBytes, align).Stride;

    /// <summary>The module of a plan's kernels: the rows products, the tensor-core product (a float32 plan, x in pairs), the widening of x to float64 (a wide plan).</summary>
    internal static (string[] Names, string Source) DescribedModule(DescribedPlan p, int align, bool pairs)
    {
        var sb = new StringBuilder(NativeHeader);
        sb.AppendLine();
        sb.AppendLine(".extern .shared .align 16 .b8 described_smem[];");
        var words = (p.CodeTable ?? []).Concat(p.Factors.SelectMany(f => f.Table ?? [])).Select(v => BitConverter.SingleToUInt32Bits((float)v)).ToArray();
        if (words.Length > 0)
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $".const .align 4 .b32 described_tables[{words.Length}] = {{{string.Join(", ", words.Select(w => w.ToString(CultureInfo.InvariantCulture)))}}};");
        }

        sb.AppendLine();
        var names = new List<string>();
        foreach (int rows in DescribedRowCounts)
        {
            DescribedRows(sb, p, align, pairs, rows, words.Length);
            names.Add(DescribedRowsName(p, align, pairs, rows));
        }

        if (!p.Wide && pairs)
        {
            DescribedMma(sb, p, align, words.Length);
            names.Add(DescribedMmaName(p, align));
        }

        if (p.Wide)
        {
            DescribedWiden(sb, p, align, pairs);
            names.Add(DescribedWidenName(p, align, pairs));
        }

        return ([.. names], sb.ToString());
    }

    // A kernel body's instructions with registers named as it asks for them (declared once it is written).
    private sealed class DescribedEmit
    {
        private readonly StringBuilder _s = new();
        private int _b, _f, _w, _g, _p, _h, _label;

        public string B() => R("%e", _b++);

        public string F() => R("%f", _f++);

        public string W() => R("%w", _w++);

        public string G() => R("%g", _g++);

        public string P() => R("%q", _p++);

        public string H() => R("%s", _h++);

        public string Label(string name) => string.Create(CultureInfo.InvariantCulture, $"{name}_{_label++}");

        public void L(FormattableString line) => _s.Append("    ").AppendLine(line.ToString(CultureInfo.InvariantCulture));

        public void Mark(string label) => _s.Append(label).AppendLine(":");

        public string Declarations() => string.Create(CultureInfo.InvariantCulture,
            $"    .reg .b32 %e<{Math.Max(1, _b)}>;\n    .reg .f32 %f<{Math.Max(1, _f)}>;\n    .reg .b64 %w<{Math.Max(1, _w)}>;\n    .reg .f64 %g<{Math.Max(1, _g)}>;\n    .reg .pred %q<{Math.Max(1, _p)}>;\n    .reg .b16 %s<{Math.Max(1, _h)}>;\n");

        public override string ToString() => _s.ToString();

        // A per-thread constant of t (0 .. 3): `values[t]`, by a product where they are a multiple of t, else selections;
        // null where every thread's is 0.
        public string? OfThread(string t, int[] values)
        {
            if (values.All(v => v == 0))
            {
                return null;
            }

            string r = B();
            if (values[0] == 0 && values.Select((v, i) => (v, i)).All(x => x.v == values[1] * x.i))
            {
                L($"mul.lo.u32 {r}, {t}, {values[1]};");
                return r;
            }

            L($"mov.u32 {r}, {values[0]};");
            for (int i = 1; i < 4; i++)
            {
                string q = P();
                L($"setp.eq.u32 {q}, {t}, {i};");
                L($"selp.u32 {r}, {values[i]}, {r}, {q};");
            }

            return r;
        }

        // A float32 or float64 immediate.
        public static string Immediate(double value, bool wide) => wide
            ? string.Create(CultureInfo.InvariantCulture, $"0d{BitConverter.DoubleToUInt64Bits(value):X16}")
            : string.Create(CultureInfo.InvariantCulture, $"0f{BitConverter.SingleToUInt32Bits((float)value):X8}");
    }

    // One pattern block's values for this thread of one weight row, decoded in registers from its staged bytes: each
    // plane's fields from `planeBase[p] + origin` (the row's staged pattern block plus the thread's offset of the plane),
    // the factors from `groupBase + origin`; a load of 1, 2, 4 or 8 bytes as the bases' alignment allows, each kept for
    // the values that read the same bytes.
    private sealed class DescribedBlock(DescribedEmit e, DescribedPlan plan, string[] planeBase, string groupBase, int origin, int[] planeAlign, int groupAlign,
        string?[] deltaShift, string? codeTable, string?[] factorTables)
    {
        private readonly List<(int Which, int Start, int Width, string Reg)> _loads = [];
        private readonly Dictionary<int, (string? S, string? B)> _groups = [];

        // Bytes [b, b + count) of base `which` (a plane, or -1: the factors') in one register: the register, the bit of byte b in it, whether it is 64-bit.
        private (string Reg, int Bit, bool Wide) Bytes(int which, int b, int count)
        {
            foreach (var (w, start, width, reg) in _loads)
            {
                if (w == which && start <= b && b + count <= start + width)
                {
                    return (reg, 8 * (b - start), width == 8);
                }
            }

            string address = which < 0 ? groupBase : planeBase[which];
            int align = which < 0 ? groupAlign : planeAlign[which];
            for (int width = count > 4 ? 8 : 4; width >= count; width /= 2)
            {
                int start = b / width * width;
                if (width <= align && b + count <= start + width)
                {
                    string reg = width == 8 ? e.W() : e.B();
                    string op = width switch { 1 => "u8", 2 => "u16", 4 => "u32", _ => "u64" };
                    e.L($"ld.shared.{op} {reg}, [{address}+{origin + start}];");
                    _loads.Add((which, start, width, reg));
                    return (reg, 8 * (b - start), width == 8);
                }
            }

            // Not aligned for one load: the bytes one at a time.
            bool wide = count > 4;
            string into = wide ? e.W() : e.B();
            e.L($"mov.{(wide ? "b64" : "b32")} {into}, 0;");
            for (int j = 0; j < count; j++)
            {
                string one = e.B();
                e.L($"ld.shared.u8 {one}, [{address}+{origin + b + j}];");
                if (wide)
                {
                    string w64 = e.W();
                    e.L($"cvt.u64.u32 {w64}, {one};");
                    e.L($"shl.b64 {w64}, {w64}, {8 * j};");
                    e.L($"or.b64 {into}, {into}, {w64};");
                }
                else
                {
                    e.L($"bfi.b32 {into}, {one}, {into}, {8 * j}, 8;");
                }
            }

            _loads.Add((which, b, count, into));
            return (into, 0, wide);
        }

        // A field's bits in a 32-bit register (a 64-bit field: its 64-bit register), sign-extended where `signed`.
        private string FieldBits(DescribedPlan.Field f, bool signed)
        {
            int count = (f.Shift + plan.MaxDeltaShift(f.Plane) + f.Bits + 7) / 8;
            var (reg, bit, wide) = Bytes(f.Plane, f.Byte, count);
            if (f.Bits == 64)
            {
                return reg;
            }

            string at = (bit + f.Shift).ToString(CultureInfo.InvariantCulture);
            if (deltaShift[f.Plane] is { } shift)
            {
                at = e.B();
                e.L($"add.u32 {at}, {shift}, {bit + f.Shift};");
            }

            string field = e.B();
            if (wide)
            {
                string w64 = e.W();
                e.L($"bfe.{(signed ? "s64" : "u64")} {w64}, {reg}, {at}, {f.Bits};");
                e.L($"cvt.u32.u64 {field}, {w64};");
            }
            else if (f.Bits == 32 && at == "0")
            {
                return reg;
            }
            else
            {
                e.L($"bfe.{(signed ? "s32" : "u32")} {field}, {reg}, {at}, {f.Bits};");
            }

            return field;
        }

        // The number of bits `bits` read as `read` (float32; float64 for a wide read).
        private string Number(string bits, DescribedPlan.Read read, int width, string? table)
        {
            string f;
            switch (read)
            {
                case DescribedPlan.Read.Signed or DescribedPlan.Read.Unsigned:
                    f = e.F();
                    e.L($"cvt.rn.f32.{(read == DescribedPlan.Read.Signed ? "s32" : "u32")} {f}, {bits};");
                    return f;
                case DescribedPlan.Read.High or DescribedPlan.Read.Float32:
                    f = e.F();
                    if (width < 32)
                    {
                        string high = e.B();
                        e.L($"shl.b32 {high}, {bits}, {32 - width};");
                        bits = high;
                    }

                    e.L($"mov.b32 {f}, {bits};");
                    return f;
                case DescribedPlan.Read.Half:
                    string h = e.H();
                    f = e.F();
                    e.L($"cvt.u16.u32 {h}, {bits};");
                    e.L($"cvt.f32.f16 {f}, {h};");
                    return f;
                case DescribedPlan.Read.Table:
                    string address = e.B();
                    f = e.F();
                    e.L($"shl.b32 {address}, {bits}, 2;");
                    e.L($"add.u32 {address}, {address}, {table};");
                    e.L($"ld.shared.f32 {f}, [{address}];");
                    return f;
                case DescribedPlan.Read.WideSigned or DescribedPlan.Read.WideUnsigned:
                    f = e.G();
                    e.L($"cvt.rn.f64.{(read == DescribedPlan.Read.WideSigned ? "s32" : "u32")} {f}, {bits};");
                    return f;
                case DescribedPlan.Read.Long or DescribedPlan.Read.ULong:
                    f = e.G();
                    e.L($"cvt.rn.f64.{(read == DescribedPlan.Read.Long ? "s64" : "u64")} {f}, {bits};");
                    return f;
                default:
                    f = e.G();
                    e.L($"mov.b64 {f}, {bits};");
                    return f;
            }
        }

        // Group g's factors: S the product of its scales (null: none), B the code offset times S less (or plus) the product of
        // its minimums (null: 0), float32; float64 for a wide plan.
        private (string? S, string? B) Factors(int g)
        {
            if (_groups.TryGetValue(g, out var known))
            {
                return known;
            }

            var pieces = plan.Groups[g];
            string? scale = null, min = null;
            for (int q = 0; q < plan.Factors.Length; q++)
            {
                var factor = plan.Factors[q];
                string bits = "";
                foreach (var piece in pieces[q])
                {
                    var (reg, bit, wide) = Bytes(-1, piece.Byte, 1);
                    string part = e.B();
                    if (wide)
                    {
                        string w64 = e.W();
                        e.L($"bfe.u64 {w64}, {reg}, {bit + piece.Shift}, {piece.Bits};");
                        e.L($"cvt.u32.u64 {part}, {w64};");
                    }
                    else
                    {
                        e.L($"bfe.u32 {part}, {reg}, {bit + piece.Shift}, {piece.Bits};");
                    }

                    if (bits.Length == 0)
                    {
                        bits = part;
                        continue;
                    }

                    string joined = e.B();
                    e.L($"bfi.b32 {joined}, {part}, {bits}, {piece.Low}, {piece.Bits};");
                    bits = joined;
                }

                if (factor.Signed && factor.Bits < 32)
                {
                    string extended = e.B();
                    e.L($"bfe.s32 {extended}, {bits}, 0, {factor.Bits};");
                    bits = extended;
                }

                string number = Number(bits, factor.Read, factor.Bits, factorTables[q]);
                if (factor.Bias != 0)
                {
                    string biased = e.F();
                    e.L($"add.rn.f32 {biased}, {number}, {DescribedEmit.Immediate(factor.Bias, false)};");
                    number = biased;
                }

                ref string? product = ref factor.Min ? ref min : ref scale;
                if (product is null)
                {
                    product = number;
                }
                else
                {
                    string times = e.F();
                    e.L($"mul.rn.f32 {times}, {product}, {number};");
                    product = times;
                }
            }

            double offset = plan.Description.CodeOffset;
            bool added = plan.Description.MinsAdded;
            string? sum = null;
            if (min is not null)
            {
                sum = e.F();
                if (offset != 0)
                {
                    string signedMin = min;
                    if (!added)
                    {
                        signedMin = e.F();
                        e.L($"neg.f32 {signedMin}, {min};");
                    }

                    e.L($"fma.rn.f32 {sum}, {scale ?? DescribedEmit.Immediate(1, false)}, {DescribedEmit.Immediate(offset, false)}, {signedMin};");
                }
                else if (added)
                {
                    e.L($"mov.f32 {sum}, {min};");
                }
                else
                {
                    e.L($"neg.f32 {sum}, {min};");
                }
            }
            else if (offset != 0)
            {
                if (scale is null)
                {
                    sum = DescribedEmit.Immediate(offset, plan.Wide);
                }
                else
                {
                    sum = e.F();
                    e.L($"mul.rn.f32 {sum}, {scale}, {DescribedEmit.Immediate(offset, false)};");
                }
            }

            if (plan.Wide)
            {
                if (scale is not null)
                {
                    string wideScale = e.G();
                    e.L($"cvt.f64.f32 {wideScale}, {scale};");
                    scale = wideScale;
                }

                if (sum is not null && !sum.StartsWith("0d", StringComparison.Ordinal))
                {
                    string wideSum = e.G();
                    e.L($"cvt.f64.f32 {wideSum}, {sum};");
                    sum = wideSum;
                }
            }

            return _groups[g] = (scale, sum);
        }

        /// <summary>Slot <paramref name="index"/>'s value in a register (float32; float64 for a wide plan).</summary>
        public string Value(int index)
        {
            var slot = plan.Slots[index];
            bool signed = plan.CodeRead is DescribedPlan.Read.Signed or DescribedPlan.Read.WideSigned;
            string code;
            if (slot.Fields.Length == 1)
            {
                code = FieldBits(slot.Fields[0], signed);
            }
            else
            {
                code = FieldBits(slot.Fields[0], false);
                for (int i = 1; i < slot.Fields.Length; i++)
                {
                    string field = FieldBits(slot.Fields[i], false), joined = e.B();
                    e.L($"bfi.b32 {joined}, {field}, {code}, {slot.Fields[i].Low}, {slot.Fields[i].Bits};");
                    code = joined;
                }

                if (signed)
                {
                    string extended = e.B();
                    e.L($"bfe.s32 {extended}, {code}, 0, {plan.Description.Code.Bits};");
                    code = extended;
                }
            }

            string number = Number(code, plan.CodeRead, plan.Description.Code.Bits, codeTable);
            var (scale, sum) = Factors(slot.Group);
            string type = plan.Wide ? "f64" : "f32";
            if (scale is null && sum is null)
            {
                return number;
            }

            string value = plan.Wide ? e.G() : e.F();
            if (scale is null)
            {
                e.L($"add.rn.{type} {value}, {number}, {sum};");
            }
            else if (sum is null)
            {
                e.L($"mul.rn.{type} {value}, {number}, {scale};");
            }
            else
            {
                e.L($"fma.rn.{type} {value}, {number}, {scale}, {sum};");
            }

            return value;
        }
    }

    // Copies the plan's tables (`words` 32-bit entries) from constant memory into the first bytes of shared memory: each
    // thread every `threads`-th entry, then a barrier.
    private static void DescribedTables(DescribedEmit e, string tid, string smem, int words, int threads)
    {
        if (words == 0)
        {
            return;
        }

        string i = e.B(), p = e.P(), at = e.B(), value = e.B(), address = e.B(), table = e.W(), wideAt = e.W();
        string loop = e.Label("TABLES"), done = e.Label("TABLES_DONE");
        e.L($"mov.u64 {table}, described_tables;");
        e.L($"mov.u32 {i}, {tid};");
        e.Mark(loop);
        e.L($"setp.ge.u32 {p}, {i}, {words};");
        e.L($"@{p} bra {done};");
        e.L($"shl.b32 {at}, {i}, 2;");
        e.L($"cvt.u64.u32 {wideAt}, {at};");
        e.L($"add.u64 {wideAt}, {wideAt}, {table};");
        e.L($"ld.const.b32 {value}, [{wideAt}];");
        e.L($"add.u32 {address}, {smem}, {at};");
        e.L($"st.shared.b32 [{address}], {value};");
        e.L($"add.u32 {i}, {i}, {threads};");
        e.L($"bra {loop};");
        e.Mark(done);
        e.L($"bar.sync 0;");
    }

    // The tables' shared addresses: the code's, then each factor's (null where it has none).
    private static (string? Code, string?[] Factors) DescribedTableBases(DescribedEmit e, DescribedPlan p, string smem)
    {
        int at = 0;
        string? code = null;
        if (p.CodeTable is { } table)
        {
            code = smem;
            at += 4 * table.Length;
        }

        var factors = new string?[p.Factors.Length];
        for (int q = 0; q < p.Factors.Length; q++)
        {
            if (p.Factors[q].Table is { } own)
            {
                factors[q] = e.B();
                e.L($"add.u32 {factors[q]}, {smem}, {at};");
                at += 4 * own.Length;
            }
        }

        return (code, factors);
    }

    // The thread's offset of each plane (null: 0), its shift (null: 0) and the planes' alignment there.
    private static (string?[] Bytes, string?[] Shifts, int[] Align) DescribedDeltas(DescribedEmit e, DescribedPlan p, string t, int align)
    {
        int planes = p.Description.Planes.Count;
        var bytes = new string?[planes];
        var shifts = new string?[planes];
        var aligns = new int[planes];
        for (int q = 0; q < planes; q++)
        {
            bytes[q] = e.OfThread(t, p.DeltaBytes[q]);
            shifts[q] = e.OfThread(t, p.DeltaShifts[q]);
            aligns[q] = Math.Min(Math.Min(align, 16), p.DeltaAlign(q));
        }

        return (bytes, shifts, aligns);
    }

    // Each plane's base for a row's pattern block at `block`: the block plus the thread's offset of the plane.
    private static string[] DescribedPlaneBases(DescribedEmit e, string?[] deltaBytes, string block)
    {
        var bases = new string[deltaBytes.Length];
        for (int q = 0; q < bases.Length; q++)
        {
            bases[q] = block;
            if (deltaBytes[q] is { } delta)
            {
                bases[q] = e.B();
                e.L($"add.u32 {bases[q]}, {block}, {delta};");
            }
        }

        return bases;
    }

    // The 16-byte cp.async copies of a staged row's window: each copy (rr, cc) of `rows` rows of `chunks` chunks, from
    // `count` copies a thread (`threads` threads), with row rr at n0 + rr of W (rows past n not copied), its address, the
    // row's end and the copy's place in a stage.
    private static (string Valid, string At, string End, string Dst)[] DescribedCopies(DescribedEmit e, string thread, int threads, int rows, int chunks, int stride,
        string first, string n, string lastRow, string rb, string w)
    {
        int count = (rows * chunks + threads - 1) / threads;
        var copies = new (string, string, string, string)[count];
        for (int j = 0; j < count; j++)
        {
            string index = e.B(), rr = e.B(), cc = e.B(), valid = e.P(), inside = e.P(), r = e.B(), at = e.W(), end = e.W(), dst = e.B(), ccBytes = e.W();
            e.L($"add.u32 {index}, {thread}, {threads * j};");
            e.L($"div.u32 {rr}, {index}, {chunks};");
            e.L($"rem.u32 {cc}, {index}, {chunks};");
            e.L($"setp.lt.u32 {valid}, {index}, {rows * chunks};");
            e.L($"add.u32 {r}, {first}, {rr};");
            e.L($"setp.lt.u32 {inside}, {r}, {n};");
            e.L($"and.pred {valid}, {valid}, {inside};");
            e.L($"min.u32 {r}, {r}, {lastRow};");
            e.L($"mul.wide.u32 {at}, {r}, {rb};");
            e.L($"add.u64 {at}, {at}, {w};");
            e.L($"cvt.u64.u32 {end}, {rb};");
            e.L($"add.u64 {end}, {end}, {at};");
            e.L($"mul.wide.u32 {ccBytes}, {cc}, 16;");
            e.L($"add.u64 {at}, {at}, {ccBytes};");
            e.L($"mul.lo.u32 {dst}, {rr}, {stride};");
            e.L($"shl.b32 {cc}, {cc}, 4;");
            e.L($"add.u32 {dst}, {dst}, {cc};");
            copies[j] = (valid, at, end, dst);
        }

        return copies;
    }

    // Issues the copies of the span at byte `spanAt` of each row (a 64-bit register) into the stage at `stage`: each chunk
    // of the row's 16-byte window from the span's first byte rounded down; bytes past the row zero (src-size).
    private static void DescribedIssue(DescribedEmit e, (string Valid, string At, string End, string Dst)[] copies, string spanAt, string stage)
    {
        foreach (var (valid, at, end, dst) in copies)
        {
            string src = e.W(), left = e.W(), size = e.B(), to = e.B();
            e.L($"add.u64 {src}, {at}, {spanAt};");
            e.L($"and.b64 {src}, {src}, -16;");
            e.L($"sub.s64 {left}, {end}, {src};");
            e.L($"max.s64 {left}, {left}, 0;");
            e.L($"min.s64 {left}, {left}, 16;");
            e.L($"cvt.u32.u64 {size}, {left};");
            e.L($"add.u32 {to}, {stage}, {dst};");
            e.L($"@{valid} cp.async.cg.shared.global [{to}], [{src}], 16, {size};");
        }
    }

    // The kernel's parameters loaded: the pointers (global), then the 32-bit values, and whether p_rs is given.
    private static (string[] Pointers, string[] Values, string HasScales) DescribedParameters(DescribedEmit e, string[] pointers, string[] values)
    {
        var p = new string[pointers.Length];
        for (int i = 0; i < p.Length; i++)
        {
            p[i] = e.W();
            e.L($"ld.param.u64 {p[i]}, [{pointers[i]}];");
        }

        string hasScales = e.P();
        e.L($"setp.ne.u64 {hasScales}, {p[2]}, 0;");
        for (int i = 0; i < p.Length; i++)
        {
            e.L($"cvta.to.global.u64 {p[i]}, {p[i]};");
        }

        var v = new string[values.Length];
        for (int i = 0; i < v.Length; i++)
        {
            v[i] = e.B();
            e.L($"ld.param.u32 {v[i]}, [{values[i]}];");
        }

        return (p, v, hasScales);
    }

    // The rows product: y [m, n] = x [m, k] · Wᵀ (× a scale an output row where p_rs is given) for up to `rows` rows of x a
    // block (the grid's y over chunks of rows), W n rows of p_rb bytes as stored. A warp stages its spans of 8 weight rows
    // (DescribedSpan pattern blocks a span) into its two shared stages in turn; lane (g, t) reads weight row g's values of
    // each step at 2t + 0, 1, 8, 9, with x's values there (pairs where k is even). Each lane's sums in k's order; the four
    // lanes added by a butterfly; the 8 warps' sums added in warp order by one thread an output.
    private static void DescribedRows(StringBuilder sb, DescribedPlan p, int align, bool pairs, int rows, int tableWords)
    {
        var e = new DescribedEmit();
        bool wide = p.Wide;
        string ft = wide ? "f64" : "f32";
        int eb = wide ? 8 : 4, span = DescribedSpan(p), spanBytes = span * p.PatternBytes;
        var (chunks, stride) = DescribedRow(spanBytes, align);
        int stageBytes = DescribedRowsBlock * stride, tables = DescribedTablesBytes(p), redAt = tables + 8 * 2 * stageBytes;

        var (pointers, values, hasScales) = DescribedParameters(e, ["p_x", "p_w", "p_rs", "p_y"], ["p_m", "p_n", "p_k", "p_rb"]);
        var (x, w, rs, y) = (pointers[0], pointers[1], pointers[2], pointers[3]);
        var (m, n, k, rb) = (values[0], values[1], values[2], values[3]);
        string tid = e.B(), lane = e.B(), warp = e.B(), g = e.B(), t = e.B(), row0 = e.B(), xr0 = e.B(), smem = e.B();
        e.L($"mov.u32 {tid}, %tid.x;");
        e.L($"and.b32 {lane}, {tid}, 31;");
        e.L($"shr.u32 {warp}, {tid}, 5;");
        e.L($"shr.u32 {g}, {lane}, 2;");
        e.L($"and.b32 {t}, {lane}, 3;");
        e.L($"mov.u32 {row0}, %ctaid.x;");
        e.L($"shl.b32 {row0}, {row0}, 3;");
        e.L($"mov.u32 {xr0}, %ctaid.y;");
        e.L($"mul.lo.u32 {xr0}, {xr0}, {rows};");
        e.L($"mov.u32 {smem}, described_smem;");
        DescribedTables(e, tid, smem, tableWords, DescribedRowsThreads);
        var (codeTable, factorTables) = DescribedTableBases(e, p, smem);

        // Pattern blocks a row, spans a row.
        string blocks = e.B(), spans = e.B();
        e.L($"add.u32 {blocks}, {k}, {p.PatternValues - 1};");
        e.L($"div.u32 {blocks}, {blocks}, {p.PatternValues};");
        e.L($"add.u32 {spans}, {blocks}, {span - 1};");
        e.L($"div.u32 {spans}, {spans}, {span};");

        // This lane's weight row (clamped into the matrix: rows past n read row n - 1 and are not written).
        string own = e.B(), ownAt = e.W(), last = e.B();
        e.L($"add.u32 {own}, {row0}, {g};");
        e.L($"sub.u32 {last}, {n}, 1;");
        e.L($"min.u32 {own}, {own}, {last};");
        e.L($"mul.wide.u32 {ownAt}, {own}, {rb};");
        e.L($"add.u64 {ownAt}, {ownAt}, {w};");
        var copies = DescribedCopies(e, lane, 32, DescribedRowsBlock, chunks, stride, row0, n, last, rb, w);

        // The warp's ring of two stages.
        string ring = e.B();
        e.L($"mul.lo.u32 {ring}, {warp}, {2 * stageBytes};");
        e.L($"add.u32 {ring}, {ring}, {smem};");
        e.L($"add.u32 {ring}, {ring}, {tables};");
        var (deltaByte, deltaShift, planeAlign) = DescribedDeltas(e, p, t, align);

        // x's rows of this block (clamped into x: rows past m read row m - 1 and are not written).
        var xRow = new string[rows];
        string lastX = e.B();
        e.L($"sub.u32 {lastX}, {m}, 1;");
        for (int r = 0; r < rows; r++)
        {
            string xr = e.B();
            xRow[r] = e.W();
            e.L($"add.u32 {xr}, {xr0}, {r};");
            e.L($"min.u32 {xr}, {xr}, {lastX};");
            e.L($"mul.wide.u32 {xRow[r]}, {xr}, {k};");
            e.L($"mul.lo.u64 {xRow[r]}, {xRow[r]}, {eb};");
            e.L($"add.u64 {xRow[r]}, {xRow[r]}, {x};");
        }

        var acc = new string[rows];
        for (int r = 0; r < rows; r++)
        {
            acc[r] = wide ? e.G() : e.F();
            e.L($"mov.{ft} {acc[r]}, {DescribedEmit.Immediate(0, wide)};");
        }

        // Span `which`'s copies into stage offset `so`.
        void Issue(string which, string so)
        {
            string first = e.W(), stage = e.B();
            e.L($"mul.wide.u32 {first}, {which}, {spanBytes};");
            e.L($"add.u32 {stage}, {ring}, {so};");
            DescribedIssue(e, copies, first, stage);
        }

        string i = e.B(), so = e.B(), next = e.B(), soNext = e.B(), any = e.P(), more = e.P();
        string loop = e.Label("SPAN"), end = e.Label("SPANS_DONE"), noIssue = e.Label("NO_ISSUE"), noFirst = e.Label("NO_FIRST");
        e.L($"mov.u32 {i}, {warp};");
        e.L($"mov.u32 {so}, 0;");
        e.L($"setp.ge.u32 {any}, {i}, {spans};");
        e.L($"@{any} bra {noFirst};");
        Issue(i, so);
        e.Mark(noFirst);
        e.L($"cp.async.commit_group;");
        e.Mark(loop);
        e.L($"setp.ge.u32 {any}, {i}, {spans};");
        e.L($"@{any} bra {end};");
        e.L($"add.u32 {next}, {i}, 8;");
        e.L($"sub.u32 {soNext}, {stageBytes}, {so};");
        e.L($"setp.ge.u32 {more}, {next}, {spans};");
        e.L($"@{more} bra {noIssue};");
        Issue(next, soNext);
        e.Mark(noIssue);
        e.L($"cp.async.commit_group;");
        e.L($"cp.async.wait_group 1;");
        e.L($"bar.warp.sync -1;");

        // Span i: its pattern blocks j = 0 .. count - 1 at this row's offset in the stage.
        string pb0 = e.B(), count = e.B(), firstAt = e.W(), offset = e.B(), rowBase = e.B();
        e.L($"mul.lo.u32 {pb0}, {i}, {span};");
        e.L($"sub.u32 {count}, {blocks}, {pb0};");
        e.L($"min.u32 {count}, {count}, {span};");
        e.L($"mul.wide.u32 {firstAt}, {pb0}, {p.PatternBytes};");
        e.L($"add.u64 {firstAt}, {firstAt}, {ownAt};");
        e.L($"cvt.u32.u64 {offset}, {firstAt};");
        e.L($"and.b32 {offset}, {offset}, 15;");
        e.L($"mad.lo.u32 {rowBase}, {g}, {stride}, {ring};");
        e.L($"add.u32 {rowBase}, {rowBase}, {so};");
        e.L($"add.u32 {rowBase}, {rowBase}, {offset};");
        string j0 = e.B(), blockLoop = e.Label("BLOCK"), blockDone = e.Label("BLOCKS_DONE"), maskedLabel = e.Label("MASKED"), blockNext = e.Label("BLOCK_NEXT");
        e.L($"mov.u32 {j0}, 0;");
        e.Mark(blockLoop);
        e.L($"setp.ge.u32 {any}, {j0}, {count};");
        e.L($"@{any} bra {blockDone};");
        string block = e.B(), kIndex = e.B(), kThread = e.B();
        e.L($"mad.lo.u32 {block}, {j0}, {p.PatternBytes}, {rowBase};");
        e.L($"add.u32 {kIndex}, {pb0}, {j0};");
        e.L($"mul.lo.u32 {kIndex}, {kIndex}, {p.PatternValues};");
        e.L($"shl.b32 {kThread}, {t}, 1;");
        e.L($"add.u32 {kThread}, {kThread}, {kIndex};");
        var planeBase = DescribedPlaneBases(e, deltaByte, block);
        var xAt = new string[rows];
        string kBytes = e.W();
        e.L($"mul.wide.u32 {kBytes}, {kThread}, {eb};");
        for (int r = 0; r < rows; r++)
        {
            xAt[r] = e.W();
            e.L($"add.u64 {xAt[r]}, {xRow[r]}, {kBytes};");
        }

        string kEnd = e.B(), partial = e.P();
        e.L($"add.u32 {kEnd}, {kIndex}, {p.PatternValues};");
        e.L($"setp.gt.u32 {partial}, {kEnd}, {k};");
        e.L($"@{partial} bra {maskedLabel};");
        RowsBody(false);
        e.L($"bra {blockNext};");
        e.Mark(maskedLabel);
        RowsBody(true);
        e.Mark(blockNext);
        e.L($"add.u32 {j0}, {j0}, 1;");
        e.L($"bra {blockLoop};");
        e.Mark(blockDone);
        e.L($"bar.warp.sync -1;");
        e.L($"mov.u32 {so}, {soNext};");
        e.L($"mov.u32 {i}, {next};");
        e.L($"bra {loop};");
        e.Mark(end);
        e.L($"cp.async.wait_group 0;");

        // A pattern block's values times x's, into each row's sum; `masked`: x's values past k read as 0.
        void RowsBody(bool masked)
        {
            var read = new DescribedBlock(e, p, planeBase, block, 0, planeAlign, Math.Min(align, 16), deltaShift, codeTable, factorTables);
            for (int s = 0; s < p.Steps; s++)
            {
                for (int half = 0; half < 2; half++)
                {
                    string v0 = read.Value(4 * s + 2 * half), v1 = read.Value(4 * s + 2 * half + 1);
                    int at = 16 * s + 8 * half;
                    string inK = "", inK1 = "";
                    if (masked)
                    {
                        string idx = e.B();
                        inK = e.P();
                        e.L($"add.u32 {idx}, {kThread}, {at};");
                        e.L($"setp.lt.u32 {inK}, {idx}, {k};");
                        if (!pairs)
                        {
                            string idx1 = e.B();
                            inK1 = e.P();
                            e.L($"add.u32 {idx1}, {idx}, 1;");
                            e.L($"setp.lt.u32 {inK1}, {idx1}, {k};");
                        }
                    }

                    for (int r = 0; r < rows; r++)
                    {
                        string xa = wide ? e.G() : e.F(), xb = wide ? e.G() : e.F();
                        if (masked)
                        {
                            e.L($"mov.{ft} {xa}, {DescribedEmit.Immediate(0, wide)};");
                            e.L($"mov.{ft} {xb}, {DescribedEmit.Immediate(0, wide)};");
                        }

                        string guard = masked ? $"@{inK} " : "", guard1 = masked ? $"@{(pairs ? inK : inK1)} " : "";
                        if (pairs)
                        {
                            e.L($"{guard}ld.global.nc.v2.{ft} {{{xa}, {xb}}}, [{xAt[r]}+{eb * at}];");
                        }
                        else
                        {
                            e.L($"{guard}ld.global.nc.{ft} {xa}, [{xAt[r]}+{eb * at}];");
                            e.L($"{guard1}ld.global.nc.{ft} {xb}, [{xAt[r]}+{eb * (at + 1)}];");
                        }

                        e.L($"fma.rn.{ft} {acc[r]}, {xa}, {v0}, {acc[r]};");
                        e.L($"fma.rn.{ft} {acc[r]}, {xb}, {v1}, {acc[r]};");
                    }
                }
            }
        }

        // The four lanes of a row added (a butterfly), then each warp's sums into shared memory, added in warp order.
        foreach (int width in new[] { 1, 2 })
        {
            for (int r = 0; r < rows; r++)
            {
                if (wide)
                {
                    string lo = e.B(), hi = e.B(), lo2 = e.B(), hi2 = e.B(), other = e.G();
                    e.L($"mov.b64 {{{lo}, {hi}}}, {acc[r]};");
                    e.L($"shfl.sync.bfly.b32 {lo2}, {lo}, {width}, 31, -1;");
                    e.L($"shfl.sync.bfly.b32 {hi2}, {hi}, {width}, 31, -1;");
                    e.L($"mov.b64 {other}, {{{lo2}, {hi2}}};");
                    e.L($"add.rn.f64 {acc[r]}, {acc[r]}, {other};");
                }
                else
                {
                    string other = e.F();
                    e.L($"shfl.sync.bfly.b32 {other}, {acc[r]}, {width}, 31, -1;");
                    e.L($"add.rn.f32 {acc[r]}, {acc[r]}, {other};");
                }
            }
        }

        string skip = e.Label("NO_PART"), notFirst = e.P(), part = e.B();
        e.L($"setp.ne.u32 {notFirst}, {t}, 0;");
        e.L($"@{notFirst} bra {skip};");
        e.L($"mad.lo.u32 {part}, {warp}, {DescribedRowsBlock}, {g};");
        e.L($"mul.lo.u32 {part}, {part}, {rows * eb};");
        e.L($"add.u32 {part}, {part}, {smem};");
        for (int r = 0; r < rows; r++)
        {
            e.L($"st.shared.{ft} [{part}+{redAt + r * eb}], {acc[r]};");
        }

        e.Mark(skip);
        e.L($"bar.sync 0;");
        string done = e.Label("DONE"), outside = e.P(), gg = e.B(), xi = e.B(), outRow = e.B(), outX = e.B();
        e.L($"setp.ge.u32 {outside}, {tid}, {DescribedRowsBlock * rows};");
        e.L($"@{outside} bra {done};");
        e.L($"div.u32 {gg}, {tid}, {rows};");
        e.L($"rem.u32 {xi}, {tid}, {rows};");
        e.L($"add.u32 {outRow}, {row0}, {gg};");
        e.L($"add.u32 {outX}, {xr0}, {xi};");
        e.L($"setp.ge.u32 {outside}, {outRow}, {n};");
        e.L($"@{outside} bra {done};");
        e.L($"setp.ge.u32 {outside}, {outX}, {m};");
        e.L($"@{outside} bra {done};");
        string sumAt = e.B(), total = wide ? e.G() : e.F();
        e.L($"mad.lo.u32 {sumAt}, {gg}, {rows}, {xi};");
        e.L($"mul.lo.u32 {sumAt}, {sumAt}, {eb};");
        e.L($"add.u32 {sumAt}, {sumAt}, {smem};");
        e.L($"ld.shared.{ft} {total}, [{sumAt}+{redAt}];");
        for (int wi = 1; wi < 8; wi++)
        {
            string add = wide ? e.G() : e.F();
            e.L($"ld.shared.{ft} {add}, [{sumAt}+{redAt + wi * DescribedRowsBlock * rows * eb}];");
            e.L($"add.rn.{ft} {total}, {total}, {add};");
        }

        string scaleAt = e.W(), scale = e.F(), result = e.F(), yAt = e.W(), col = e.W(), noScale = e.Label("NO_SCALE");
        e.L($"@!{hasScales} bra {noScale};");
        e.L($"mul.wide.u32 {scaleAt}, {outRow}, 4;");
        e.L($"add.u64 {scaleAt}, {scaleAt}, {rs};");
        e.L($"ld.global.f32 {scale}, [{scaleAt}];");
        if (wide)
        {
            string wideScale = e.G();
            e.L($"cvt.f64.f32 {wideScale}, {scale};");
            e.L($"mul.rn.f64 {total}, {total}, {wideScale};");
        }
        else
        {
            e.L($"mul.rn.f32 {total}, {total}, {scale};");
        }

        e.Mark(noScale);
        if (wide)
        {
            e.L($"cvt.rn.f32.f64 {result}, {total};");
        }
        else
        {
            e.L($"mov.f32 {result}, {total};");
        }

        e.L($"mul.wide.u32 {yAt}, {outX}, {n};");
        e.L($"cvt.u64.u32 {col}, {outRow};");
        e.L($"add.u64 {yAt}, {yAt}, {col};");
        e.L($"shl.b64 {yAt}, {yAt}, 2;");
        e.L($"add.u64 {yAt}, {yAt}, {y};");
        e.L($"st.global.f32 [{yAt}], {result};");
        e.Mark(done);
        e.L($"ret;");

        sb.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{DescribedRowsName(p, align, pairs, rows)}}(
                .param .u64 p_x, .param .u64 p_w, .param .u64 p_rs, .param .u64 p_y,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .u32 p_rb
            )
            .maxntid {{DescribedRowsThreads}}, 1, 1
            {
            """);
        sb.Append(e.Declarations());
        sb.Append(e.ToString());
        sb.AppendLine("}");
        sb.AppendLine();
    }

    // The tensor-core product (a bfloat16 run's prompts): y [m, n] = x16 [m, kp] · Wᵀ (× a scale an output row where p_rs
    // is given), x16 x's bfloat16 rows of kp (zero past k), W n rows of p_rb bytes as stored, k split over the grid's z
    // (each split p_per of W's spans, its sums into y's plane z: p_y holds as many planes of [m, n] as the grid's z). A
    // block of 4 warps (2 × 2) computes 64 × 64 outputs; per W span (whole pattern blocks, a multiple of 64 values) its rows
    // are staged as stored, per 64 values x's rows; both in two stages by cp.async, the next while this one is read. In each
    // k16 step a lane's B fragment of each of its 4 tiles is weight row 8 nt + g's values 2t + 0, 1, 8, 9 of the step,
    // decoded in registers and rounded to bfloat16 pairs; A by ldmatrix.
    private static void DescribedMma(StringBuilder sb, DescribedPlan p, int align, int tableWords)
    {
        var e = new DescribedEmit();
        int spanValues = DescribedMmaSpan(p), subs = spanValues / 64, spanBytes = spanValues / p.PatternValues * p.PatternBytes;
        var (chunks, stride) = DescribedRow(spanBytes, align);
        int tables = DescribedTablesBytes(p), xBase = tables, wBase = tables + 2 * DescribedXStage, wStage = DescribedMmaTile * stride;

        var (pointers, values, hasScales) = DescribedParameters(e, ["p_x", "p_w", "p_rs", "p_y"], ["p_m", "p_n", "p_k", "p_kp", "p_rb", "p_per"]);
        var (x, w, rs, y) = (pointers[0], pointers[1], pointers[2], pointers[3]);
        var (m, n, kp, rb, per) = (values[0], values[1], values[3], values[4], values[5]);
        string tid = e.B(), lane = e.B(), warp = e.B(), g = e.B(), t = e.B(), wm = e.B(), wn = e.B(), n0 = e.B(), m0 = e.B(), z = e.B(), smem = e.B();
        e.L($"mov.u32 {tid}, %tid.x;");
        e.L($"and.b32 {lane}, {tid}, 31;");
        e.L($"shr.u32 {warp}, {tid}, 5;");
        e.L($"shr.u32 {g}, {lane}, 2;");
        e.L($"and.b32 {t}, {lane}, 3;");
        e.L($"and.b32 {wm}, {warp}, 1;");
        e.L($"shr.u32 {wn}, {warp}, 1;");
        e.L($"mov.u32 {n0}, %ctaid.x;");
        e.L($"shl.b32 {n0}, {n0}, 6;");
        e.L($"mov.u32 {m0}, %ctaid.y;");
        e.L($"shl.b32 {m0}, {m0}, 6;");
        e.L($"mov.u32 {z}, %ctaid.z;");
        e.L($"mov.u32 {smem}, described_smem;");
        DescribedTables(e, tid, smem, tableWords, DescribedMmaThreads);
        var (codeTable, factorTables) = DescribedTableBases(e, p, smem);

        // This split's W spans [sp0, sp1).
        string all = e.B(), sp0 = e.B(), sp1 = e.B(), lastRow = e.B();
        e.L($"div.u32 {all}, {kp}, {spanValues};");
        e.L($"mul.lo.u32 {sp0}, {z}, {per};");
        e.L($"add.u32 {sp1}, {sp0}, {per};");
        e.L($"min.u32 {sp1}, {sp1}, {all};");
        e.L($"sub.u32 {lastRow}, {n}, 1;");
        var wCopies = DescribedCopies(e, tid, DescribedMmaThreads, DescribedMmaTile, chunks, stride, n0, n, lastRow, rb, w);

        // x's copies: 4 a thread, chunk (rr, cc) of the tile's 64 rows of 64 values.
        var xCopy = new (string Size, string At, string Dst)[4];
        string lastX = e.B();
        e.L($"sub.u32 {lastX}, {m}, 1;");
        for (int j = 0; j < 4; j++)
        {
            string index = e.B(), rr = e.B(), cc = e.B(), inside = e.P(), r = e.B(), at = e.W(), size = e.B(), dst = e.B(), ccv = e.W();
            e.L($"add.u32 {index}, {tid}, {DescribedMmaThreads * j};");
            e.L($"shr.u32 {rr}, {index}, 3;");
            e.L($"and.b32 {cc}, {index}, 7;");
            e.L($"add.u32 {r}, {m0}, {rr};");
            e.L($"setp.lt.u32 {inside}, {r}, {m};");
            e.L($"selp.u32 {size}, 16, 0, {inside};");
            e.L($"min.u32 {r}, {r}, {lastX};");
            e.L($"mul.wide.u32 {at}, {r}, {kp};");
            e.L($"mul.wide.u32 {ccv}, {cc}, 8;");
            e.L($"add.u64 {at}, {at}, {ccv};");
            e.L($"shl.b64 {at}, {at}, 1;");
            e.L($"add.u64 {at}, {at}, {x};");
            e.L($"mul.lo.u32 {dst}, {rr}, {DescribedXRow};");
            e.L($"shl.b32 {cc}, {cc}, 4;");
            e.L($"add.u32 {dst}, {dst}, {cc};");
            xCopy[j] = (size, at, dst);
        }

        string xs = e.B(), ws = e.B();
        e.L($"add.u32 {xs}, {smem}, {xBase};");
        e.L($"add.u32 {ws}, {smem}, {wBase};");

        // W span `which` into W stage `wso`.
        void IssueW(string which, string wso)
        {
            string first = e.W(), stage = e.B();
            e.L($"mul.wide.u32 {first}, {which}, {spanBytes};");
            e.L($"add.u32 {stage}, {ws}, {wso};");
            DescribedIssue(e, wCopies, first, stage);
        }

        // x's 64 values from k index `kAt` into x stage `xso`.
        void IssueX(string kAt, string xso)
        {
            string kb = e.W(), stage = e.B();
            e.L($"mul.wide.u32 {kb}, {kAt}, 2;");
            e.L($"add.u32 {stage}, {xs}, {xso};");
            foreach (var (size, at, dst) in xCopy)
            {
                string src = e.W(), to = e.B();
                e.L($"add.u64 {src}, {at}, {kb};");
                e.L($"add.u32 {to}, {stage}, {dst};");
                e.L($"cp.async.cg.shared.global [{to}], [{src}], 16, {size};");
            }
        }

        // The lane's ldmatrix offset of A, and its B rows (nt) with their global rows (for the span's offset).
        string aOff = e.B(), tmp = e.B();
        e.L($"and.b32 {aOff}, {lane}, 15;");
        e.L($"mad.lo.u32 {aOff}, {wm}, 32, {aOff};");
        e.L($"mul.lo.u32 {aOff}, {aOff}, {DescribedXRow};");
        e.L($"shr.u32 {tmp}, {lane}, 4;");
        e.L($"shl.b32 {tmp}, {tmp}, 4;");
        e.L($"add.u32 {aOff}, {aOff}, {tmp};");
        var bRow = new string[4];
        var bAt = new string[4];
        for (int nt = 0; nt < 4; nt++)
        {
            string r = e.B();
            bRow[nt] = e.B();
            bAt[nt] = e.W();
            e.L($"mad.lo.u32 {bRow[nt]}, {wn}, 32, {g};");
            e.L($"add.u32 {bRow[nt]}, {bRow[nt]}, {8 * nt};");
            e.L($"add.u32 {r}, {n0}, {bRow[nt]};");
            e.L($"min.u32 {r}, {r}, {lastRow};");
            e.L($"mul.wide.u32 {bAt[nt]}, {r}, {rb};");
            e.L($"add.u64 {bAt[nt]}, {bAt[nt]}, {w};");
            e.L($"mul.lo.u32 {bRow[nt]}, {bRow[nt]}, {stride};");
        }

        var (deltaByte, deltaShift, planeAlign) = DescribedDeltas(e, p, t, align);
        var acc = new string[32];
        for (int c = 0; c < 32; c++)
        {
            acc[c] = e.F();
            e.L($"mov.f32 {acc[c]}, 0f00000000;");
        }

        // The pipeline over sub-spans of 64 values: x's stage toggles each sub-span, W's each span.
        string i = e.B(), xso = e.B(), wso = e.B(), any = e.P(), kAt = e.B();
        string spanLoop = e.Label("SPAN"), spansDone = e.Label("SPANS_DONE"), noFirst = e.Label("NO_FIRST");
        e.L($"mov.u32 {i}, {sp0};");
        e.L($"mov.u32 {xso}, 0;");
        e.L($"mov.u32 {wso}, 0;");
        e.L($"setp.ge.u32 {any}, {i}, {sp1};");
        e.L($"@{any} bra {noFirst};");
        IssueW(i, wso);
        e.L($"mul.lo.u32 {kAt}, {i}, {spanValues};");
        IssueX(kAt, xso);
        e.Mark(noFirst);
        e.L($"cp.async.commit_group;");
        e.Mark(spanLoop);
        e.L($"setp.ge.u32 {any}, {i}, {sp1};");
        e.L($"@{any} bra {spansDone};");

        // The span's base of each B row: the W stage, the row, the row's offset in its 16-byte window.
        var rowBase = new string[4];
        var planeBases = new string[4][];
        for (int nt = 0; nt < 4; nt++)
        {
            string first = e.W(), offset = e.B();
            rowBase[nt] = e.B();
            e.L($"mul.wide.u32 {first}, {i}, {spanBytes};");
            e.L($"add.u64 {first}, {first}, {bAt[nt]};");
            e.L($"cvt.u32.u64 {offset}, {first};");
            e.L($"and.b32 {offset}, {offset}, 15;");
            e.L($"add.u32 {rowBase[nt]}, {ws}, {wso};");
            e.L($"add.u32 {rowBase[nt]}, {rowBase[nt]}, {bRow[nt]};");
            e.L($"add.u32 {rowBase[nt]}, {rowBase[nt]}, {offset};");
            planeBases[nt] = DescribedPlaneBases(e, deltaByte, rowBase[nt]);
        }

        string next = e.B(), xsoNext = e.B(), wsoNext = e.B(), last = e.P();
        e.L($"add.u32 {next}, {i}, 1;");
        e.L($"sub.u32 {wsoNext}, {wStage}, {wso};");
        for (int sub = 0; sub < subs; sub++)
        {
            // The next sub-span's copies: x's next 64 values, and at the span's last, the next span's W.
            e.L($"sub.u32 {xsoNext}, {DescribedXStage}, {xso};");
            if (sub < subs - 1)
            {
                e.L($"mul.lo.u32 {kAt}, {i}, {spanValues};");
                e.L($"add.u32 {kAt}, {kAt}, {64 * (sub + 1)};");
                IssueX(kAt, xsoNext);
            }
            else
            {
                string noNext = e.Label("NO_NEXT");
                e.L($"setp.ge.u32 {last}, {next}, {sp1};");
                e.L($"@{last} bra {noNext};");
                IssueW(next, wsoNext);
                e.L($"mul.lo.u32 {kAt}, {next}, {spanValues};");
                IssueX(kAt, xsoNext);
                e.Mark(noNext);
            }

            e.L($"cp.async.commit_group;");
            e.L($"cp.async.wait_group 1;");
            e.L($"bar.sync 0;");
            string aAt = e.B();
            e.L($"add.u32 {aAt}, {xs}, {xso};");
            e.L($"add.u32 {aAt}, {aAt}, {aOff};");
            var reads = new Dictionary<(int, int), DescribedBlock>();
            for (int ks = 0; ks < 4; ks++)
            {
                var a = new string[2][];
                for (int mt = 0; mt < 2; mt++)
                {
                    a[mt] = [e.B(), e.B(), e.B(), e.B()];
                    e.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{a[mt][0]}, {a[mt][1]}, {a[mt][2]}, {a[mt][3]}}}, [{aAt}+{mt * 16 * DescribedXRow + ks * 32}];");
                }

                int u = 64 * sub + 16 * ks, pb = u / p.PatternValues, step = u % p.PatternValues / 16;
                var b = new string[4][];
                for (int nt = 0; nt < 4; nt++)
                {
                    if (!reads.TryGetValue((nt, pb), out var read))
                    {
                        read = new DescribedBlock(e, p, planeBases[nt], rowBase[nt], pb * p.PatternBytes, planeAlign, Math.Min(align, 16), deltaShift, codeTable, factorTables);
                        reads[(nt, pb)] = read;
                    }

                    string v0 = read.Value(4 * step), v1 = read.Value(4 * step + 1), v8 = read.Value(4 * step + 2), v9 = read.Value(4 * step + 3);
                    b[nt] = [e.B(), e.B()];
                    e.L($"cvt.rn.bf16x2.f32 {b[nt][0]}, {v1}, {v0};");
                    e.L($"cvt.rn.bf16x2.f32 {b[nt][1]}, {v9}, {v8};");
                }

                for (int mt = 0; mt < 2; mt++)
                {
                    for (int nt = 0; nt < 4; nt++)
                    {
                        int c = (mt * 4 + nt) * 4;
                        e.L($"mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {{{acc[c]}, {acc[c + 1]}, {acc[c + 2]}, {acc[c + 3]}}}, {{{a[mt][0]}, {a[mt][1]}, {a[mt][2]}, {a[mt][3]}}}, {{{b[nt][0]}, {b[nt][1]}}}, {{{acc[c]}, {acc[c + 1]}, {acc[c + 2]}, {acc[c + 3]}}};");
                    }
                }
            }

            e.L($"bar.sync 0;");
            e.L($"mov.u32 {xso}, {xsoNext};");
        }

        e.L($"mov.u32 {wso}, {wsoNext};");
        e.L($"mov.u32 {i}, {next};");
        e.L($"bra {spanLoop};");
        e.Mark(spansDone);
        e.L($"cp.async.wait_group 0;");

        // y's plane z: tile (mt, nt)'s value c at row m0 + 32 wm + 16 mt + g (+ 8 for c ≥ 2), column n0 + 32 wn + 8 nt + 2t
        // (+ 1 for odd c), times the column's scale.
        string plane = e.W(), zWide = e.W(), rowOut = e.B(), colOut = e.B(), even = e.P();
        e.L($"mul.wide.u32 {plane}, {m}, {n};");
        e.L($"cvt.u64.u32 {zWide}, {z};");
        e.L($"mul.lo.u64 {plane}, {plane}, {zWide};");
        e.L($"shl.b64 {plane}, {plane}, 2;");
        e.L($"add.u64 {y}, {y}, {plane};");
        e.L($"mad.lo.u32 {rowOut}, {wm}, 32, {m0};");
        e.L($"add.u32 {rowOut}, {rowOut}, {g};");
        e.L($"mad.lo.u32 {colOut}, {wn}, 32, {n0};");
        e.L($"shl.b32 {tmp}, {t}, 1;");
        e.L($"add.u32 {colOut}, {colOut}, {tmp};");
        e.L($"and.b32 {tmp}, {n}, 1;");
        e.L($"setp.eq.u32 {even}, {tmp}, 0;");
        for (int nt = 0; nt < 4; nt++)
        {
            string col = e.B(), col1 = e.B(), in0 = e.P(), in1 = e.P(), s0 = e.F(), s1 = e.F(), sAt = e.W(), q0 = e.P(), q1 = e.P();
            e.L($"add.u32 {col}, {colOut}, {8 * nt};");
            e.L($"add.u32 {col1}, {col}, 1;");
            e.L($"setp.lt.u32 {in0}, {col}, {n};");
            e.L($"setp.lt.u32 {in1}, {col1}, {n};");
            e.L($"mov.f32 {s0}, 0f3F800000;");
            e.L($"mov.f32 {s1}, 0f3F800000;");
            e.L($"mul.wide.u32 {sAt}, {col}, 4;");
            e.L($"add.u64 {sAt}, {sAt}, {rs};");
            e.L($"and.pred {q0}, {in0}, {hasScales};");
            e.L($"and.pred {q1}, {in1}, {hasScales};");
            e.L($"@{q0} ld.global.f32 {s0}, [{sAt}];");
            e.L($"@{q1} ld.global.f32 {s1}, [{sAt}+4];");
            for (int mt = 0; mt < 2; mt++)
            {
                for (int half = 0; half < 2; half++)
                {
                    int c = (mt * 4 + nt) * 4 + 2 * half;
                    string row = e.B(), inRow = e.P(), yAt = e.W(), colWide = e.W(), both = e.P(), single = e.P(), s0p = e.P(), s1p = e.P();
                    e.L($"add.u32 {row}, {rowOut}, {16 * mt + 8 * half};");
                    e.L($"setp.lt.u32 {inRow}, {row}, {m};");
                    e.L($"mul.rn.f32 {acc[c]}, {acc[c]}, {s0};");
                    e.L($"mul.rn.f32 {acc[c + 1]}, {acc[c + 1]}, {s1};");
                    e.L($"mul.wide.u32 {yAt}, {row}, {n};");
                    e.L($"cvt.u64.u32 {colWide}, {col};");
                    e.L($"add.u64 {yAt}, {yAt}, {colWide};");
                    e.L($"shl.b64 {yAt}, {yAt}, 2;");
                    e.L($"add.u64 {yAt}, {yAt}, {y};");
                    e.L($"and.pred {both}, {inRow}, {in1};");
                    e.L($"and.pred {both}, {both}, {even};");
                    e.L($"@{both} st.global.v2.f32 [{yAt}], {{{acc[c]}, {acc[c + 1]}}};");
                    e.L($"not.pred {single}, {both};");
                    e.L($"and.pred {single}, {single}, {inRow};");
                    e.L($"and.pred {s0p}, {single}, {in0};");
                    e.L($"and.pred {s1p}, {single}, {in1};");
                    e.L($"@{s0p} st.global.f32 [{yAt}], {acc[c]};");
                    e.L($"@{s1p} st.global.f32 [{yAt}+4], {acc[c + 1]};");
                }
            }
        }

        e.L($"ret;");
        sb.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{DescribedMmaName(p, align)}}(
                .param .u64 p_x, .param .u64 p_w, .param .u64 p_rs, .param .u64 p_y,
                .param .u32 p_m, .param .u32 p_n, .param .u32 p_k, .param .u32 p_kp, .param .u32 p_rb, .param .u32 p_per
            )
            .maxntid {{DescribedMmaThreads}}, 1, 1
            {
            """);
        sb.Append(e.Declarations());
        sb.Append(e.ToString());
        sb.AppendLine("}");
        sb.AppendLine();
    }

    // x's float32 values widened to float64 (a wide plan's x, once a call): y[i] = x[i] for i below p_count.
    private static void DescribedWiden(StringBuilder sb, DescribedPlan p, int align, bool pairs) => sb.AppendLine(CultureInfo.InvariantCulture, $$"""
        .visible .entry {{DescribedWidenName(p, align, pairs)}}(
            .param .u64 p_x, .param .u64 p_y, .param .u32 p_count
        )
        .maxntid 256, 1, 1
        {
            .reg .pred %p<2>;
            .reg .b32 %r<4>;
            .reg .f32 %f<2>;
            .reg .f64 %d<2>;
            .reg .b64 %rd<6>;
            ld.param.u64 %rd1, [p_x];
            ld.param.u64 %rd2, [p_y];
            cvta.to.global.u64 %rd1, %rd1;
            cvta.to.global.u64 %rd2, %rd2;
            ld.param.u32 %r1, [p_count];
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %tid.x;
            mad.lo.u32 %r2, %r2, 256, %r3;
            setp.ge.u32 %p0, %r2, %r1;
            @%p0 bra DONE;
            mul.wide.u32 %rd3, %r2, 4;
            add.u64 %rd3, %rd3, %rd1;
            ld.global.f32 %f1, [%rd3];
            cvt.f64.f32 %d1, %f1;
            mul.wide.u32 %rd4, %r2, 8;
            add.u64 %rd4, %rd4, %rd2;
            st.global.f64 [%rd4], %d1;
        DONE:
            ret;
        }

        """);
}
