// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// The experiment's block route on CUDA (ProductRoute.Adad), ported from Adad and driven by the weights' description: x
// quantized into q8_0's blocks (Adad's Ptx.Blocks EncodeQ0: a warp a block, the block's largest |x| by butterflies, the
// scale stored as float16 and read back, each value rounded to the nearest code, ties to even, within ±127), then
// Σ over blocks of dW·dX·(Σ of 32 integer products), the weights' codes read as their layout says (BlockLayout: 8-bit
// codes, or 4-bit codes in ggml's order, signed or biased, a 16-bit float scale anywhere in the block, or none and a scale
// a row): a decoding-sized product a thread a weight row (Adad's BlockProducts GemvBlocks; dp4a for 8-bit codes with no
// bias), a prompt-sized one by the s8 tensor instruction at k = 32 (Adad's MatMulBlocks: 64 × 64 of C a block of 128
// threads, four k-blocks staged in shared memory by 16-bit loads, codes widened to int8 on the way, rows 48 bytes apart,
// the s32 tile results scaled into float32 sums). Nothing is decoded to a float per value; C comes out as float32.
internal static partial class PtxKernels
{
    /// <summary>The block product's tile: threads a block, rows and columns of C a block, bytes of a staged row, k-blocks a stage.</summary>
    internal const int BlocksThreads = 128, BlocksTile = 64, BlocksRow = 48, BlocksStage = 4;

    /// <summary>x's blocks: 32 values, a float16 scale then 32 int8 codes (q8_0).</summary>
    internal const int Q8Values = 32, Q8Bytes = 34;

    /// <summary>
    /// A block route operand's layout, from a description: bytes a block of 32 values takes, its 16-bit float scale's byte
    /// and bits (ScaleOffset −1: none, the row's scale applies), its codes' byte and bits (8, or 4 in ggml's order: byte i
    /// holds value i low, value i + 16 high), their sign and the bias added before the scale.
    /// </summary>
    internal sealed record BlockLayout(int Bytes, int ScaleOffset, FloatBits Scale, int CodeOffset, int CodeBits, bool Signed, int Bias, IReadOnlyList<int>? Table = null)
    {
        /// <summary>q8_0's layout (x's).</summary>
        public static readonly BlockLayout Q8 = new(Q8Bytes, 0, new FloatBits(5, 10), 2, 8, true, 0);

        public bool HasScale => ScaleOffset >= 0;

        /// <summary>16-bit units of codes a block holds.</summary>
        public int CodeUnits => 32 * CodeBits / 16;

        /// <summary>16-bit units a block's staging reads: the scale's, then the codes'.</summary>
        public int Units => (HasScale ? 1 : 0) + CodeUnits;

        /// <summary>The codes are int8 as stored (no widening on the way).</summary>
        public bool Direct => CodeBits == 8 && Signed && Bias == 0 && Table is null;

        /// <summary>The layout as a kernel's name part.</summary>
        public string Code => string.Create(CultureInfo.InvariantCulture,
            $"b{Bytes}{(HasScale ? $"s{ScaleOffset}f{Scale.Exponent}m{Scale.Mantissa}" : "r")}c{CodeOffset}x{CodeBits}{(Signed ? 's' : 'u')}{(Bias < 0 ? 'n' : 'p')}{Math.Abs(Bias)}{(Table is null ? "" : "t" + string.Join("_", Table.Select(e => e < 0 ? "n" + -e : e.ToString(CultureInfo.InvariantCulture))))}");

        /// <summary>
        /// The block route's layout of <paramref name="type"/>'s weights (<paramref name="rowScales"/>: a scale a row is given),
        /// or null: 32-value blocks of 8-bit codes or 4-bit codes in ggml's order with a float16 or bfloat16 scale, and no
        /// scale a row; or an 8-bit integer with a scale a row (rows of codes). Each code plus its bias within int8.
        /// </summary>
        public static BlockLayout? Of(ElementType type, bool rowScales)
        {
            if (type.Scaled is { Values: 32 } b)
            {
                bool scale = b.Scale is { Exponent: 5, Mantissa: 10 } or { Exponent: 8, Mantissa: 7 };
                bool codes = b.CodeBits == 8 && b.CodeTable is null || b.CodeBits == 4 && b.NibblesSplit && b.CodeTable is null or { Count: 16 };
                bool fits = b.CodeTable is { } table ? table.All(e => e + b.CodeBias is >= -128 and <= 127) : WithinInt8(b.CodeBits, b.SignedCodes, b.CodeBias);
                return !rowScales && scale && codes && b.Bytes % 2 == 0 && b.ScaleOffset % 2 == 0 && b.CodeOffset % 2 == 0 && fits
                    ? new BlockLayout(b.Bytes, b.ScaleOffset, b.Scale, b.CodeOffset, b.CodeBits, b.SignedCodes, b.CodeBias, b.CodeTable)
                    : null;
            }

            return rowScales && !type.IsBlock && type.Float is null && type.IsInteger && type.BlockBits == 8 && WithinInt8(8, type.IsSigned, 0)
                ? new BlockLayout(32, -1, default, 0, 8, type.IsSigned, 0)
                : null;
        }

        private static bool WithinInt8(int bits, bool signed, int bias) =>
            (signed ? -(1 << (bits - 1)) : 0) + bias >= -128 && (signed ? (1 << (bits - 1)) - 1 : (1 << bits) - 1) + bias <= 127;
    }

    /// <summary>The block module's kernels for <paramref name="w"/>: x into q8_0 blocks, the row product, the s8 tensor tile.</summary>
    internal static string[] BlockNames(BlockLayout w) => ["blocks_quantize_q8_0", $"gemv_blocks_{w.Code}", $"matmul_blocks_{w.Code}"];

    /// <summary>The block route's module for weights of layout <paramref name="w"/> (sm_80: the s8 tensor instruction at k = 32).</summary>
    internal static string BlockModule(BlockLayout w)
    {
        var names = BlockNames(w);
        var sb = new StringBuilder(BlockQuantize(NativeHeader, names[0]));
        sb.Append(GemvBlocks(w, names[1]));
        sb.Append(MatMulBlocks(w, names[2]));
        return sb.ToString();
    }

    private static string Param64(PtxBuilder b, string name)
    {
        string r = b.RD();
        b.L($"ld.param.u64 {r}, [{name}];");
        return r;
    }

    private static string Param32(PtxBuilder b, string name)
    {
        string r = b.R();
        b.L($"ld.param.u32 {r}, [{name}];");
        return r;
    }

    // A 4-bit code in `reg` as its table entry (sign-extended) where the layout has a code table (TypedDecode's TableEntry,
    // on the builder's registers).
    private static void BlockTable(PtxBuilder b, BlockLayout l, string reg)
    {
        if (l.Table is { } table)
        {
            TableLookup(b, table, reg);
        }
    }

    // A 4-bit code in `reg` as its entry of a 16-entry int8 table, sign-extended (two byte selections, the half by bit 3).
    internal static void TableLookup(PtxBuilder b, IReadOnlyList<int> table, string reg)
    {

        var words = TableWords(table).Select(_ => b.R()).ToArray();
        var values = TableWords(table);
        for (int i = 0; i < 4; i++)
        {
            b.L(string.Create(CultureInfo.InvariantCulture, $"mov.b32 {words[i]}, {values[i].Word};"));
        }

        string sel = b.R(), lo = b.R(), hi = b.R(), p = b.P();
        b.L($"and.b32 {sel}, {reg}, 7;");
        b.L($"prmt.b32 {lo}, {words[0]}, {words[1]}, {sel};");
        b.L($"prmt.b32 {hi}, {words[2]}, {words[3]}, {sel};");
        b.L($"setp.ge.u32 {p}, {reg}, 8;");
        b.L($"selp.b32 {reg}, {hi}, {lo}, {p};");
        b.L($"bfe.s32 {reg}, {reg}, 0, 8;");
    }

    // A 16-bit float of `bits` held in the low half of `hv` (a .b32 register), as float32.
    private static string BlockScale(PtxBuilder b, string hv, FloatBits bits)
    {
        string d = b.F();
        if (bits.Exponent == 8)
        {
            string t = b.R();
            b.L($"shl.b32 {t}, {hv}, 16;");
            b.L($"mov.b32 {d}, {t};");
            return d;
        }

        string hh = b.H();
        b.L($"cvt.u16.u32 {hh}, {hv};");
        b.L($"cvt.f32.f16 {d}, {hh};");
        return d;
    }

    // Adad's LanesToBlock with EncodeQ0 for q8_0: block (gid / 32) of p_n float32 values (p_n whole blocks), lane l
    // holding value l.
    private static string BlockQuantize(string header, string entry)
    {
        var b = new PtxBuilder();
        string x = Param64(b, "p_x"), outp = Param64(b, "p_out"), n = Param32(b, "p_n");
        string tid = b.R(), ctaid = b.R(), ntid = b.R(), gid = b.R(), blk = b.R(), lane = b.R(), blocks = b.R(), pDone = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {ctaid}, %ctaid.x;");
        b.L($"mov.u32 {ntid}, %ntid.x;");
        b.L($"mad.lo.u32 {gid}, {ctaid}, {ntid}, {tid};");
        b.L($"shr.u32 {blk}, {gid}, 5;");
        b.L($"and.b32 {lane}, {gid}, 31;");
        b.L($"shr.u32 {blocks}, {n}, 5;");
        b.L($"setp.ge.u32 {pDone}, {blk}, {blocks};");
        b.L($"@{pDone} bra DONE;");
        string addr = b.RD(), v = b.F(), bAddr = b.RD();
        b.L($"mul.wide.u32 {addr}, {gid}, 4;");
        b.L($"add.u64 {addr}, {x}, {addr};");
        b.L($"ld.global.f32 {v}, [{addr}];");
        b.L($"mul.wide.u32 {bAddr}, {blk}, {Q8Bytes};");
        b.L($"add.u64 {bAddr}, {outp}, {bAddr};");

        // The block's largest |x| (NaN as 0, as max drops it), over the warp by butterflies.
        string amax = b.F();
        b.L($"abs.f32 {amax}, {v};");
        b.L($"max.f32 {amax}, {amax}, 0f00000000;");
        for (int off = 16; off >= 1; off >>= 1)
        {
            string o = b.F(), m = b.F();
            b.L($"shfl.sync.bfly.b32 {o}, {amax}, {off}, 0x1f, 0xffffffff;");
            b.L($"max.f32 {m}, {amax}, {o};");
            amax = m;
        }

        string d = b.F(), h = b.H(), dBack = b.F(), id = b.F(), pZero = b.P(), pFirst = b.P();
        b.L($"div.rn.f32 {d}, {amax}, 0f42FE0000;");                       // / 127
        b.L($"cvt.rn.f16.f32 {h}, {d};");
        b.L($"cvt.f32.f16 {dBack}, {h};");
        b.L($"setp.eq.f32 {pZero}, {dBack}, 0f00000000;");
        b.L($"div.rn.f32 {id}, 0f3F800000, {dBack};");
        b.L($"selp.f32 {id}, 0f00000000, {id}, {pZero};");
        b.L($"setp.eq.u32 {pFirst}, {lane}, 0;");
        b.L($"@{pFirst} st.global.b16 [{bAddr}], {h};");
        string t = b.F(), q = b.R(), hq = b.H(), at = b.RD();
        b.L($"mul.rn.f32 {t}, {v}, {id};");
        b.L($"cvt.rni.s32.f32 {q}, {t};");
        b.L($"max.s32 {q}, {q}, -127;");
        b.L($"min.s32 {q}, {q}, 127;");
        b.L($"cvt.u16.u32 {hq}, {q};");
        b.L($"cvt.u64.u32 {at}, {lane};");
        b.L($"add.u64 {at}, {bAddr}, {at};");
        b.L($"st.global.b8 [{at}+2], {hq};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish(header, entry, ".param .u64 p_x, .param .u64 p_out, .param .u32 p_n");
    }

    // Adad's GemvBlocks over weights of layout w: y[row] = Σ_blocks d_w·d_x·(Σ q_w·q_x), a thread a row of W against one row
    // of x (q8_0), times the row's scale where the layout has none of its own.
    private static string GemvBlocks(BlockLayout w, string entry)
    {
        var b = new PtxBuilder();
        string tid = b.R(), bid = b.R(), row = b.R(), a = Param64(b, "p_a"), x = Param64(b, "p_x"), y = Param64(b, "p_y"), rows = Param32(b, "p_rows"), blocks = Param32(b, "p_blocks"), ps = Param64(b, "p_s"), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bid}, %ctaid.x;");
        b.L($"mad.lo.u32 {row}, {bid}, 256, {tid};");
        b.L($"setp.ge.u32 {pEnd}, {row}, {rows};");
        b.L($"@{pEnd} bra DONE;");
        string acc = b.F(), blk = b.R(), pB = b.P(), aAddr = b.RD(), xAddr = b.RD(), rowBase = b.RD();
        b.L($"mov.f32 {acc}, 0f00000000;");
        b.L($"mul.wide.u32 {rowBase}, {row}, {blocks};");
        b.L($"mul.lo.u64 {rowBase}, {rowBase}, {w.Bytes};");
        b.L($"add.u64 {rowBase}, {a}, {rowBase};");
        b.L($"mov.u32 {blk}, 0;");
        b.Label("BLK");
        b.L($"setp.ge.u32 {pB}, {blk}, {blocks};");
        b.L($"@{pB} bra BLK_END;");
        b.L($"mul.wide.u32 {aAddr}, {blk}, {w.Bytes};");
        b.L($"add.u64 {aAddr}, {rowBase}, {aAddr};");
        b.L($"mul.wide.u32 {xAddr}, {blk}, {Q8Bytes};");
        b.L($"add.u64 {xAddr}, {x}, {xAddr};");
        string hx = b.H(), dx = b.F(), dot = b.R(), da;
        if (w.HasScale)
        {
            string ha = b.R();
            b.L($"ld.global.u16 {ha}, [{aAddr}+{w.ScaleOffset}];");
            da = BlockScale(b, ha, w.Scale);
        }
        else
        {
            da = b.F();
            b.L($"mov.f32 {da}, 0f3F800000;");
        }

        b.L($"ld.global.u16 {hx}, [{xAddr}];");
        b.L($"cvt.f32.f16 {dx}, {hx};");
        b.L($"mov.u32 {dot}, 0;");

        // A block is 2-byte aligned only: a 4-byte group is read as two halves.
        void Word(string reg, string addr, int offset)
        {
            string lo = b.H(), hi = b.H();
            b.L($"ld.global.u16 {lo}, [{addr}+{offset}];");
            b.L($"ld.global.u16 {hi}, [{addr}+{offset + 2}];");
            b.L($"mov.b32 {reg}, {{{lo}, {hi}}};");
        }

        // W's code at bit `at` of `word` (its field `bits` wide), plus the bias, into a fresh register.
        string CodeOf(string word, int at, int bits)
        {
            string c = b.R();
            b.L($"bfe.{(w.Signed ? "s32" : "u32")} {c}, {word}, {at}, {bits};");
            BlockTable(b, w, c);
            if (w.Bias != 0)
            {
                b.L($"add.s32 {c}, {c}, {w.Bias};");
            }

            return c;
        }

        if (w.CodeBits == 8)
        {
            for (int g = 0; g < 8; g++)
            {
                string wa = b.R(), wx = b.R();
                Word(wa, aAddr, w.CodeOffset + g * 4);
                Word(wx, xAddr, 2 + g * 4);
                if (w.Bias == 0)
                {
                    b.L($"dp4a.{(w.Signed ? "s32" : "u32")}.s32 {dot}, {wa}, {wx}, {dot};");
                    continue;
                }

                for (int i = 0; i < 4; i++)
                {
                    string qx = b.R();
                    b.L($"bfe.s32 {qx}, {wx}, {i * 8}, 8;");
                    b.L($"mad.lo.s32 {dot}, {CodeOf(wa, i * 8, 8)}, {qx}, {dot};");
                }
            }
        }
        else
        {
            // 16 bytes of 4-bit codes: byte j holds value j (low half) and value j + 16 (high half).
            for (int g = 0; g < 4; g++)
            {
                string wa = b.R(), xl = b.R(), xh = b.R();
                Word(wa, aAddr, w.CodeOffset + g * 4);
                Word(xl, xAddr, 2 + g * 4);
                Word(xh, xAddr, 2 + 16 + g * 4);
                for (int i = 0; i < 4; i++)
                {
                    string ql = b.R(), qh = b.R();
                    b.L($"bfe.s32 {ql}, {xl}, {i * 8}, 8;");
                    b.L($"bfe.s32 {qh}, {xh}, {i * 8}, 8;");
                    b.L($"mad.lo.s32 {dot}, {CodeOf(wa, i * 8, 4)}, {ql}, {dot};");
                    b.L($"mad.lo.s32 {dot}, {CodeOf(wa, i * 8 + 4, 4)}, {qh}, {dot};");
                }
            }
        }

        string dotF = b.F(), scale = b.F();
        b.L($"cvt.rn.f32.s32 {dotF}, {dot};");
        b.L($"mul.rn.f32 {scale}, {da}, {dx};");
        b.L($"fma.rn.f32 {acc}, {scale}, {dotF}, {acc};");
        b.L($"add.u32 {blk}, {blk}, 1;");
        b.L("bra BLK;");
        b.Label("BLK_END");
        string yAddr = b.RD();
        if (!w.HasScale)
        {
            string sAddr = b.RD(), rs = b.F();
            b.L($"mul.wide.u32 {sAddr}, {row}, 4;");
            b.L($"add.u64 {sAddr}, {ps}, {sAddr};");
            b.L($"ld.global.f32 {rs}, [{sAddr}];");
            b.L($"mul.rn.f32 {acc}, {acc}, {rs};");
        }

        b.L($"mul.wide.u32 {yAddr}, {row}, 4;");
        b.L($"add.u64 {yAddr}, {y}, {yAddr};");
        b.L($"st.global.f32 [{yAddr}], {acc};");
        b.Label("DONE");
        b.L("ret;");
        return b.Finish("", entry, ".param .u64 p_a, .param .u64 p_x, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_blocks, .param .u64 p_s");
    }

    // Adad's MatMulBlocks: C (f32, m × n) = A·Bᵀ for A x's q8_0 rows and B weight rows of layout w (k whole blocks), the s8
    // tensor instruction at k = 32; each operand's codes widened to int8 as they are staged.
    private static string MatMulBlocks(BlockLayout w, string entry)
    {
        const int tile = BlocksTile, rowB = BlocksRow, ks = BlocksStage, T = BlocksThreads, slab = tile * rowB;
        const int aOff = 0, bOff = ks * slab, saOff = 2 * ks * slab, sbOff = saOff + ks * tile * 4, shared = sbOff + ks * tile * 4;
        var b = new PtxBuilder();
        string tid = b.R(), bx = b.R(), by = b.R(), sm = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"mov.u32 {sm}, smem;");
        string pa = Param64(b, "p_a"), pb = Param64(b, "p_b"), pc = Param64(b, "p_c"), M = Param32(b, "p_m"), N = Param32(b, "p_n"), blocks = Param32(b, "p_blocks"), ps = Param64(b, "p_s");
        string m0 = b.R(), n0 = b.R(), strideA = b.RD(), strideB = b.RD();
        b.L($"mul.lo.u32 {m0}, {by}, {tile};");
        b.L($"mul.lo.u32 {n0}, {bx}, {tile};");
        b.L($"mul.wide.u32 {strideA}, {blocks}, {Q8Bytes};");
        b.L($"mul.wide.u32 {strideB}, {blocks}, {w.Bytes};");

        // The warp's 32 × 32 and the lane's place in the fragments.
        string warp = b.R(), lane = b.R(), gid = b.R(), tig = b.R(), wm = b.R(), wn = b.R();
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {lane}, {tid}, 31;");
        b.L($"shr.u32 {gid}, {lane}, 2;");
        b.L($"and.b32 {tig}, {lane}, 3;");
        b.L($"shr.u32 {wm}, {warp}, 1;");
        b.L($"mul.lo.u32 {wm}, {wm}, 32;");
        b.L($"and.b32 {wn}, {warp}, 1;");
        b.L($"mul.lo.u32 {wn}, {wn}, 32;");
        var acc = new string[2, 4, 4];
        for (int mt = 0; mt < 2; mt++)
        {
            for (int nt = 0; nt < 4; nt++)
            {
                for (int e = 0; e < 4; e++)
                {
                    acc[mt, nt, e] = b.F();
                    b.L($"mov.f32 {acc[mt, nt, e]}, 0f00000000;");
                }
            }
        }

        // Fragment byte offsets in shared memory (row · 48 + tig · 4), fixed across k-blocks.
        string aFrag0 = b.R(), bFrag0 = b.R(), sa0 = b.R(), sb0 = b.R(), t4 = b.R();
        b.L($"shl.b32 {t4}, {tig}, 2;");
        b.L($"add.u32 {aFrag0}, {wm}, {gid};");
        b.L($"mad.lo.u32 {aFrag0}, {aFrag0}, {rowB}, {t4};");
        b.L($"add.u32 {aFrag0}, {aFrag0}, {sm};");
        b.L($"add.u32 {bFrag0}, {wn}, {gid};");
        b.L($"mad.lo.u32 {bFrag0}, {bFrag0}, {rowB}, {t4};");
        b.L($"add.u32 {bFrag0}, {bFrag0}, {sm};");
        b.L($"add.u32 {sa0}, {wm}, {gid};");
        b.L($"mad.lo.u32 {sa0}, {sa0}, 4, {sm};");                               // sA[wm + gid]
        b.L($"shl.b32 {sb0}, {tig}, 1;");
        b.L($"add.u32 {sb0}, {sb0}, {wn};");
        b.L($"mad.lo.u32 {sb0}, {sb0}, 4, {sm};");                               // sB[wn + tig·2]

        string kb = b.R(), pK = b.P();
        b.L($"mov.u32 {kb}, 0;");
        b.Label("KB");
        b.L($"setp.ge.u32 {pK}, {kb}, {blocks};");
        b.L($"@{pK} bra KB_END;");

        // Stage k-blocks kb .. kb + 3 of an operand's tile: unit i of the 64 × (4 · Units) goes to thread i % T; a unit is
        // the block's scale (into the scales) or two bytes of its codes (widened to int8 into the tile's row).
        void Stage(BlockLayout l, string gPtr, string limit, string rowBase0, int tileOff, int scaleOff, string stride, string tag)
        {
            int units = l.Units, rowUnits = ks * units;
            for (int it = 0; it < tile * rowUnits / T; it++)
            {
                string idx = b.R(), row = b.R(), hw = b.R(), blk = b.R(), h = b.R(), gRow = b.R(), kAt = b.R(), valid = b.P(), kValid = b.P(), addr = b.RD(), hv = b.R();
                b.L($"add.u32 {idx}, {tid}, {it * T};");
                b.L($"div.u32 {row}, {idx}, {rowUnits};");
                b.L($"rem.u32 {hw}, {idx}, {rowUnits};");
                b.L($"div.u32 {blk}, {hw}, {units};");
                b.L($"rem.u32 {h}, {hw}, {units};");
                b.L($"add.u32 {gRow}, {rowBase0}, {row};");
                b.L($"add.u32 {kAt}, {kb}, {blk};");
                b.L($"setp.lt.u32 {valid}, {gRow}, {limit};");
                b.L($"setp.lt.and.u32 {kValid}, {kAt}, {blocks}, {valid};");

                // The unit's byte in the block: the scale's (unit 0 where there is one), else the codes' unit h (− 1).
                string off = b.R(), first = b.P(), kBytes = b.RD(), offWide = b.RD();
                b.L($"setp.eq.u32 {first}, {h}, 0;");
                b.L($"shl.b32 {off}, {h}, 1;");
                b.L($"add.u32 {off}, {off}, {l.CodeOffset - (l.HasScale ? 2 : 0)};");
                if (l.HasScale)
                {
                    b.L($"selp.u32 {off}, {l.ScaleOffset}, {off}, {first};");
                }

                b.L($"mul.wide.u32 {addr}, {gRow}, 1;");
                b.L($"mul.lo.u64 {addr}, {addr}, {stride};");
                b.L($"mul.wide.u32 {kBytes}, {kAt}, {l.Bytes};");
                b.L($"add.u64 {addr}, {addr}, {kBytes};");
                b.L($"cvt.u64.u32 {offWide}, {off};");
                b.L($"add.u64 {addr}, {addr}, {offWide};");
                b.L($"add.u64 {addr}, {addr}, {gPtr};");
                b.L($"mov.u32 {hv}, 0;");
                b.L($"@{kValid} ld.global.u16 {hv}, [{addr}];");

                string sAddr = b.R();
                b.L($"mad.lo.u32 {sAddr}, {blk}, {tile}, {row};");
                b.L($"mad.lo.u32 {sAddr}, {sAddr}, 4, {sm};");
                if (l.HasScale)
                {
                    b.L($"@!{first} bra VAL_{tag}_{it};");
                    string dScale = BlockScale(b, hv, l.Scale);
                    b.L($"st.shared.f32 [{sAddr}+{scaleOff}], {dScale};");
                    b.L($"bra NEXT_{tag}_{it};");
                }
                else
                {
                    string one = b.F();
                    b.L($"mov.f32 {one}, 0f3F800000;");
                    b.L($"@{first} st.shared.f32 [{sAddr}+{scaleOff}], {one};");   // the row's scale applies after
                }

                b.Label($"VAL_{tag}_{it}");
                string dRow = b.R(), j = b.R();
                b.L($"mad.lo.u32 {dRow}, {blk}, {tile}, {row};");
                b.L($"mad.lo.u32 {dRow}, {dRow}, {rowB}, {sm};");                       // the k-block's tile, the row
                b.L($"sub.u32 {j}, {h}, {(l.HasScale ? 1 : 0)};");
                b.L($"shl.b32 {j}, {j}, 1;");                                             // the unit's byte in the codes
                b.L($"add.u32 {dRow}, {dRow}, {j};");
                string type = l.Signed ? "s32" : "u32";

                // Two codes at bits `lo` and `hi` of hv (fields `bits` wide), plus the bias, as two int8 bytes.
                string Pair(int lo, int hi, int bits)
                {
                    string c0 = b.R(), c1 = b.R(), pair = b.R();
                    b.L($"bfe.{type} {c0}, {hv}, {lo}, {bits};");
                    b.L($"bfe.{type} {c1}, {hv}, {hi}, {bits};");
                    BlockTable(b, l, c0);
                    BlockTable(b, l, c1);
                    if (l.Bias != 0)
                    {
                        b.L($"add.s32 {c0}, {c0}, {l.Bias};");
                        b.L($"add.s32 {c1}, {c1}, {l.Bias};");
                    }

                    b.L($"and.b32 {c0}, {c0}, 255;");
                    b.L($"and.b32 {c1}, {c1}, 255;");
                    b.L($"shl.b32 {c1}, {c1}, 8;");
                    b.L($"or.b32 {pair}, {c0}, {c1};");
                    return pair;
                }

                void Store(string value, int extra)
                {
                    string hq = b.H();
                    b.L($"cvt.u16.u32 {hq}, {value};");
                    b.L($"st.shared.u16 [{dRow}+{tileOff + extra}], {hq};");
                }

                if (l.CodeBits == 8)
                {
                    Store(l.Direct ? hv : Pair(0, 8, 8), 0);
                }
                else
                {
                    // Bytes 2j, 2j + 1 of 4-bit codes: values 2j, 2j + 1 (low halves) and 2j + 16, 2j + 17 (high halves).
                    Store(Pair(0, 8, 4), 0);
                    Store(Pair(4, 12, 4), 16);
                }

                b.Label($"NEXT_{tag}_{it}");
            }
        }

        Stage(BlockLayout.Q8, pa, M, m0, aOff, saOff, strideA, "A");
        Stage(w, pb, N, n0, bOff, sbOff, strideB, "B");
        b.L("bar.sync 0;");

        // Per staged k-block, the warp's 2 × 4 tiles: one m16n8k32 each, its s32 result scaled into the f32 accumulators.
        for (int kk = 0; kk < ks; kk++)
        {
            string kAt = b.R(), pSkip = b.P();
            b.L($"add.u32 {kAt}, {kb}, {kk};");
            b.L($"setp.ge.u32 {pSkip}, {kAt}, {blocks};");
            b.L($"@{pSkip} bra COMPUTED;");
            var sbv = new string[4, 2];
            for (int nt = 0; nt < 4; nt++)
            {
                for (int e = 0; e < 2; e++)
                {
                    sbv[nt, e] = b.F();
                    b.L($"ld.shared.f32 {sbv[nt, e]}, [{sb0}+{sbOff + (kk * tile + nt * 8 + e) * 4}];");
                }
            }

            for (int mt = 0; mt < 2; mt++)
            {
                string a0 = b.R(), a1 = b.R(), a2 = b.R(), a3 = b.R(), s0 = b.F(), s1 = b.F();
                int ro = aOff + kk * slab + mt * 16 * rowB;
                b.L($"ld.shared.b32 {a0}, [{aFrag0}+{ro}];");
                b.L($"ld.shared.b32 {a1}, [{aFrag0}+{ro + 8 * rowB}];");
                b.L($"ld.shared.b32 {a2}, [{aFrag0}+{ro + 16}];");
                b.L($"ld.shared.b32 {a3}, [{aFrag0}+{ro + 8 * rowB + 16}];");
                b.L($"ld.shared.f32 {s0}, [{sa0}+{saOff + (kk * tile + mt * 16) * 4}];");
                b.L($"ld.shared.f32 {s1}, [{sa0}+{saOff + (kk * tile + mt * 16 + 8) * 4}];");
                for (int nt = 0; nt < 4; nt++)
                {
                    string b0 = b.R(), b1 = b.R(), d0 = b.R(), d1 = b.R(), d2 = b.R(), d3 = b.R();
                    int co = bOff + kk * slab + nt * 8 * rowB;
                    b.L($"ld.shared.b32 {b0}, [{bFrag0}+{co}];");
                    b.L($"ld.shared.b32 {b1}, [{bFrag0}+{co + 16}];");
                    b.L($"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{d0}, {d1}, {d2}, {d3}}}, {{{a0}, {a1}, {a2}, {a3}}}, {{{b0}, {b1}}}, {{0, 0, 0, 0}};");
                    var d = new[] { d0, d1, d2, d3 };
                    for (int e = 0; e < 4; e++)
                    {
                        string df = b.F(), sc = b.F();
                        b.L($"cvt.rn.f32.s32 {df}, {d[e]};");
                        b.L($"mul.rn.f32 {sc}, {(e < 2 ? s0 : s1)}, {sbv[nt, e & 1]};");
                        b.L($"fma.rn.f32 {acc[mt, nt, e]}, {df}, {sc}, {acc[mt, nt, e]};");
                    }
                }
            }
        }

        b.Label("COMPUTED");
        b.L("bar.sync 0;");
        b.L($"add.u32 {kb}, {kb}, {ks};");
        b.L("bra KB;");
        b.Label("KB_END");

        // The accumulators to C (inside m × n only), times the column's row scale where the weights' layout has none.
        for (int mt = 0; mt < 2; mt++)
        {
            for (int nt = 0; nt < 4; nt++)
            {
                for (int e = 0; e < 4; e++)
                {
                    string row = b.R(), col = b.R(), pr = b.P(), pcl = b.P(), idx = b.R(), addr = b.RD();
                    b.L($"add.u32 {row}, {m0}, {wm};");
                    b.L($"add.u32 {row}, {row}, {gid};");
                    b.L($"add.u32 {row}, {row}, {mt * 16 + (e < 2 ? 0 : 8)};");
                    b.L($"shl.b32 {col}, {tig}, 1;");
                    b.L($"add.u32 {col}, {col}, {n0};");
                    b.L($"add.u32 {col}, {col}, {wn};");
                    b.L($"add.u32 {col}, {col}, {nt * 8 + (e & 1)};");
                    b.L($"setp.lt.u32 {pr}, {row}, {M};");
                    b.L($"setp.lt.and.u32 {pcl}, {col}, {N}, {pr};");
                    string value = acc[mt, nt, e];
                    if (!w.HasScale)
                    {
                        string sAddr = b.RD(), rs = b.F(), scaled = b.F();
                        b.L($"mul.wide.u32 {sAddr}, {col}, 4;");
                        b.L($"add.u64 {sAddr}, {ps}, {sAddr};");
                        b.L($"mov.f32 {rs}, 0f00000000;");
                        b.L($"@{pcl} ld.global.f32 {rs}, [{sAddr}];");
                        b.L($"mul.rn.f32 {scaled}, {value}, {rs};");
                        value = scaled;
                    }

                    b.L($"mad.lo.u32 {idx}, {row}, {N}, {col};");
                    b.L($"mul.wide.u32 {addr}, {idx}, 4;");
                    b.L($"add.u64 {addr}, {pc}, {addr};");
                    b.L($"@{pcl} st.global.f32 [{addr}], {value};");
                }
            }
        }

        b.L("ret;");
        return b.Finish("", entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_blocks, .param .u64 p_s", shared);
    }
}
