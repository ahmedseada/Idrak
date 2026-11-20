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
            bool codes = block.CodeBits == 8 && block.CodeTable is null
                || block.CodeBits == 4 && block.NibblesSplit && (block.CodeTable is null || block.CodeTable is { Count: 16 } grid && grid.All(e => e is >= -128 and <= 127));
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
