// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

// The experiment's kernels for super-block formats (ElementType.Super: ggml's k-quants), generated from the description
// alone: a thread widens 8 consecutive values of a block (SuperEight: the super-scale and minimum, the sub-block's scale and
// minimum assembled from their pieces, read from a table in the module's constant memory by the sub-block, each code from
// its planes' fields), then either writes them (SuperWiden: rows of float32 for the float32 product, or of bfloat16 padded
// to the native kind's rows) or multiplies them with up to 8 rows of x (SuperGemv: a warp a weight row, lanes over 8-value
// chunks, the sums added across the warp in a fixed order). Every value is (d · scale) · code − dmin · min, rounded at
// each step as the CPU's description decoder rounds it (SuperBlock.Decode).
internal static partial class PtxKernels
{
    // A description's name part: a stable hash of its canonical text (the same description, the same kernels).
    internal static string SuperCode(SuperBlock s)
    {
        var text = new StringBuilder(string.Create(CultureInfo.InvariantCulture, $"{s.Values}x{s.Bytes}b{s.CodeBias}f{s.Scale.Exponent}m{s.Scale.Mantissa}s{s.ScaleOffset}n{s.MinOffset}v{s.SubValues}q{s.SubScaleBias}{s.SubScalesSigned}"));
        foreach (var p in s.Planes)
        {
            text.Append(CultureInfo.InvariantCulture, $"|{p.Offset},{p.Bits},{p.Group}");
        }

        foreach (var list in s.SubMins is null ? new[] { s.SubScales } : new[] { s.SubScales, s.SubMins })
        {
            foreach (var pieces in list)
            {
                text.Append(';');
                foreach (var piece in pieces)
                {
                    text.Append(CultureInfo.InvariantCulture, $"{piece.Offset}.{piece.Shift}.{piece.Bits},");
                }
            }
        }

        uint hash = 2166136261;
        foreach (char c in text.ToString())
        {
            hash = (hash ^ c) * 16777619;
        }

        return string.Create(CultureInfo.InvariantCulture, $"k{s.Values}x{s.Bytes}_{hash:x8}");
    }

    // The most pieces a sub-block's scale or minimum takes (the table's entries a sub-block, each).
    private static int SuperPieces(SuperBlock s) => Math.Max(s.SubScales.Max(p => p.Count), s.SubMins?.Max(p => p.Count) ?? 0);

    /// <summary>Whether these kernels widen the description: powers of two where they shift, even offsets for 16-bit loads, small pieces.</summary>
    internal static bool SuperFits(SuperBlock s)
    {
        static bool Pow2(int v) => v > 0 && (v & (v - 1)) == 0;
        bool planes = s.Planes.All(p => p.Bits is 1 or 2 or 4 or 8 && Pow2(p.Group) && p.Group * p.Bits / 8 is var per && per >= 8 && Pow2(per) && p.Offset % 2 == 0);
        bool pieces = (s.SubMins is null ? s.SubScales : s.SubScales.Concat(s.SubMins)).All(list => list.All(q => q.Offset is >= 0 and < 256 && q.Shift < 8 && q.Bits is > 0 and <= 8)
            && list.Sum(q => q.Bits) < 32);
        bool signedWidth = !s.SubScalesSigned || s.SubScales.Select(list => list.Sum(q => q.Bits)).Distinct().Count() == 1;
        return Pow2(s.Values) && Pow2(s.SubValues) && s.SubValues >= 8 && s.Values % s.SubValues == 0 && s.Bytes % 2 == 0 && s.ScaleOffset % 2 == 0
            && (s.MinOffset < 0 || s.MinOffset % 2 == 0) && s.Scale.Bits == 16 && planes && pieces && signedWidth && s.Planes.Sum(p => p.Bits) <= 16;
    }

    /// <summary>The kernels of a description's module: the float32 and bfloat16 widenings, then the few-row products (TypedGemvRows).</summary>
    internal static string[] SuperNames(SuperBlock s)
    {
        string c = SuperCode(s);
        return [$"super_widen_{c}_f32", $"super_widen_{c}_bf16", .. TypedGemvRows.ToArray().Select(r => string.Create(CultureInfo.InvariantCulture, $"super_gemv_{c}_r{r}")),
            .. TypedGemvRows.ToArray().Select(r => string.Create(CultureInfo.InvariantCulture, $"super_gemv_{c}_r{r}_many")),
            .. TypedGemvRows.ToArray().Select(r => string.Create(CultureInfo.InvariantCulture, $"super_gemv_{c}_r{r}_many_gated")),
            .. TypedGemvRows.ToArray().Select(r => string.Create(CultureInfo.InvariantCulture, $"super_gemv_{c}_r{r}_addnorm"))];
    }

    /// <summary>The module for <paramref name="s"/>: its pieces' table in constant memory, the widenings and the few-row products.</summary>
    internal static string SuperModule(SuperBlock s)
    {
        var names = SuperNames(s);
        var sb = new StringBuilder(NativeHeader);
        sb.Append(SuperPiecesDeclaration(s));
        sb.Append(SuperWiden(s, names[0], bf16: false));
        sb.Append(SuperWiden(s, names[1], bf16: true));
        for (int i = 0; i < TypedGemvRows.Length; i++)
        {
            sb.Append(SuperGemv(s, names[2 + i], TypedGemvRows[i]));
        }

        for (int i = 0; i < TypedGemvRows.Length; i++)
        {
            sb.Append(SuperGemv(s, names[2 + TypedGemvRows.Length + i], TypedGemvRows[i], many: true));
            sb.Append(SuperGemv(s, names[2 + 2 * TypedGemvRows.Length + i], TypedGemvRows[i], many: true, epilogue: Epilogue.Gated));
            sb.Append(SuperGemv(s, names[2 + 3 * TypedGemvRows.Length + i], TypedGemvRows[i], epilogue: Epilogue.AddNorm));
        }

        return sb.ToString();
    }

    /// <summary>The module's table of the sub-blocks' scale and minimum pieces in constant memory (super_pieces), as SuperEight reads it.</summary>
    internal static string SuperPiecesDeclaration(SuperBlock s)
    {
        int p = SuperPieces(s), subs = s.Values / s.SubValues;
        var entries = new List<uint>();
        for (int j = 0; j < subs; j++)
        {
            foreach (var list in new[] { s.SubScales[j], s.SubMins?[j] ?? [] })
            {
                int at = 0;
                for (int i = 0; i < p; i++)
                {
                    if (i < list.Count)
                    {
                        var q = list[i];
                        entries.Add((uint)(q.Offset | q.Shift << 8 | q.Bits << 12 | at << 16));
                        at += q.Bits;
                    }
                    else
                    {
                        entries.Add(0);                                     // no bits: adds nothing
                    }
                }
            }
        }

        return string.Create(CultureInfo.InvariantCulture, $".const .align 4 .b32 super_pieces[{entries.Count}] = {{{string.Join(", ", entries)}}};\n");
    }

    private static int Log2(int v) => System.Numerics.BitOperations.Log2((uint)v);

    // The float of the description's 16-bit scale bits at byte `offset` of the block at `block`.
    private static string SuperFloat(PtxBuilder b, SuperBlock s, string block, int offset)
    {
        string h = b.R(), f = b.F();
        b.L($"ld.global.u16 {h}, [{block}+{offset}];");
        if (s.Scale.Exponent == 8)
        {
            b.L($"shl.b32 {h}, {h}, 16;");
            b.L($"mov.b32 {f}, {h};");
            return f;
        }

        string hh = b.H();
        b.L($"cvt.u16.u32 {hh}, {h};");
        b.L($"cvt.f32.f16 {f}, {hh};");
        return f;
    }

    // Values [w, w + 8) of the block at `block` (w a u32 register, a multiple of 8, below Values) as float32 registers.
    private static string[] SuperEight(PtxBuilder b, SuperBlock s, string block, string w)
    {
        string d = SuperFloat(b, s, block, s.ScaleOffset);
        string? dmin = s.MinOffset >= 0 ? SuperFloat(b, s, block, s.MinOffset) : null;
        int p = SuperPieces(s);
        string j = b.R(), table = b.RD();
        b.L($"shr.u32 {j}, {w}, {Log2(s.SubValues)};");
        b.L($"mov.u64 {table}, super_pieces;");

        // Sub-block j's scale (which 0) or minimum (1) from its table entries: each a byte, a shift, bits and their place.
        string Assemble(int which, int signedBits)
        {
            string value = b.R(), first = b.R();
            b.L($"mov.u32 {value}, 0;");
            b.L($"mad.lo.u32 {first}, {j}, {2 * p}, {which * p};");
            for (int i = 0; i < p; i++)
            {
                string e = b.R(), at = b.RD(), off = b.R(), sh = b.R(), bits = b.R(), place = b.R(), wide = b.RD(), by = b.R(), f = b.R();
                b.L($"add.u32 {e}, {first}, {i};");
                b.L($"mul.wide.u32 {at}, {e}, 4;");
                b.L($"add.u64 {at}, {table}, {at};");
                b.L($"ld.const.u32 {e}, [{at}];");
                b.L($"and.b32 {off}, {e}, 255;");
                b.L($"bfe.u32 {sh}, {e}, 8, 4;");
                b.L($"bfe.u32 {bits}, {e}, 12, 4;");
                b.L($"bfe.u32 {place}, {e}, 16, 5;");
                b.L($"cvt.u64.u32 {wide}, {off};");
                b.L($"add.u64 {wide}, {block}, {wide};");
                b.L($"ld.global.u8 {by}, [{wide}];");
                b.L($"bfe.u32 {f}, {by}, {sh}, {bits};");
                b.L($"shl.b32 {f}, {f}, {place};");
                b.L($"or.b32 {value}, {value}, {f};");
            }

            if (signedBits > 0)
            {
                b.L($"bfe.s32 {value}, {value}, 0, {signedBits};");
            }

            return value;
        }

        string sc = Assemble(0, s.SubScalesSigned ? s.SubScales[0].Sum(q => q.Bits) : 0), scF = b.F(), scale = b.F();
        if (s.SubScaleBias != 0)
        {
            b.L($"add.s32 {sc}, {sc}, {s.SubScaleBias};");
        }

        b.L($"cvt.rn.f32.s32 {scF}, {sc};");
        b.L($"mul.rn.f32 {scale}, {d}, {scF};");
        string? min = null;
        if (s.SubMins is not null)
        {
            string mn = Assemble(1, 0), mnF = b.F();
            min = b.F();
            b.L($"cvt.rn.f32.s32 {mnF}, {mn};");
            b.L($"mul.rn.f32 {min}, {dmin}, {mnF};");
        }

        var code = new string[8];
        for (int e = 0; e < 8; e++)
        {
            code[e] = b.R();
            b.L($"mov.u32 {code[e]}, 0;");
        }

        int low = 0;
        foreach (var plane in s.Planes)
        {
            int per = plane.Group * plane.Bits / 8;
            string g = b.R(), ww = b.R(), byte0 = b.R(), shift = b.R(), tmp = b.R(), addr = b.RD();
            b.L($"shr.u32 {g}, {w}, {Log2(plane.Group)};");
            b.L($"and.b32 {ww}, {w}, {plane.Group - 1};");
            b.L($"mul.lo.u32 {byte0}, {g}, {per};");
            b.L($"and.b32 {tmp}, {ww}, {per - 1};");
            b.L($"add.u32 {byte0}, {byte0}, {tmp};");
            b.L($"add.u32 {byte0}, {byte0}, {plane.Offset};");
            b.L($"shr.u32 {shift}, {ww}, {Log2(per)};");
            b.L($"mul.lo.u32 {shift}, {shift}, {plane.Bits};");
            b.L($"cvt.u64.u32 {addr}, {byte0};");
            b.L($"add.u64 {addr}, {block}, {addr};");

            // The 8 values' fields lie in 8 consecutive bytes at the same shift: four 16-bit loads (a block is 2-byte aligned).
            var units = new string[4];
            for (int u = 0; u < 4; u++)
            {
                units[u] = b.R();
                b.L($"ld.global.u16 {units[u]}, [{addr}+{2 * u}];");
            }

            for (int e = 0; e < 8; e++)
            {
                string pos = b.R(), f = b.R();
                b.L($"add.u32 {pos}, {shift}, {8 * (e % 2)};");
                b.L($"bfe.u32 {f}, {units[e / 2]}, {pos}, {plane.Bits};");
                if (low > 0)
                {
                    b.L($"shl.b32 {f}, {f}, {low};");
                }

                b.L($"or.b32 {code[e]}, {code[e]}, {f};");
            }

            low += plane.Bits;
        }

        var values = new string[8];
        for (int e = 0; e < 8; e++)
        {
            string cf = b.F();
            values[e] = b.F();
            if (s.CodeBias != 0)
            {
                b.L($"add.s32 {code[e]}, {code[e]}, {s.CodeBias};");
            }

            b.L($"cvt.rn.f32.s32 {cf}, {code[e]};");
            b.L($"mul.rn.f32 {values[e]}, {scale}, {cf};");
            if (min is not null)
            {
                b.L($"sub.rn.f32 {values[e]}, {values[e]}, {min};");
            }
        }

        return values;
    }

    // Rows of the weights widened: out[row · ld + v] for v below p_k, 0 for v in [p_k, p_ld) (the native kind's padding); a
    // thread 8 values; float32, or bfloat16 pairs.
    /// <summary>A block format's decoder for the generated kernels: values and bytes a block, 8 values' decode (a block's address, the first value), the module's tables.</summary>
    internal sealed record BlockDecode(int Values, int Bytes, Func<PtxBuilder, string, string, string[]> Eight);

    private static string SuperWiden(SuperBlock s, string entry, bool bf16) => BlockWiden(new BlockDecode(s.Values, s.Bytes, (b, block, w) => SuperEight(b, s, block, w)), entry, bf16);

    private static string BlockWiden(BlockDecode s, string entry, bool bf16)
    {
        var b = new PtxBuilder();
        string pw = Param64(b, "p_w"), po = Param64(b, "p_out"), rows = Param32(b, "p_rows"), k = Param32(b, "p_k"), ld = Param32(b, "p_ld");
        string tid = b.R(), ctaid = b.R(), t = b.R(), groups = b.R(), total = b.R(), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {ctaid}, %ctaid.x;");
        b.L($"mad.lo.u32 {t}, {ctaid}, 256, {tid};");
        b.L($"shr.u32 {groups}, {ld}, 3;");
        b.L($"mul.lo.u32 {total}, {rows}, {groups};");
        b.L($"setp.ge.u32 {pEnd}, {t}, {total};");
        b.L($"@{pEnd} bra DONE;");
        string row = b.R(), gi = b.R(), v0 = b.R(), pZero = b.P();
        b.L($"div.u32 {row}, {t}, {groups};");
        b.L($"rem.u32 {gi}, {t}, {groups};");
        b.L($"shl.b32 {v0}, {gi}, 3;");
        b.L($"setp.ge.u32 {pZero}, {v0}, {k};");
        b.L($"@{pZero} bra ZERO;");
        string per = b.R(), blk = b.R(), w = b.R(), at = b.RD(), blockAddr = b.RD();
        b.L($"shr.u32 {per}, {k}, {Log2(s.Values)};");
        b.L($"shr.u32 {blk}, {v0}, {Log2(s.Values)};");
        b.L($"and.b32 {w}, {v0}, {s.Values - 1};");
        b.L($"mad.lo.u32 {blk}, {row}, {per}, {blk};");
        b.L($"mul.wide.u32 {at}, {blk}, {s.Bytes};");
        b.L($"add.u64 {blockAddr}, {pw}, {at};");
        var values = s.Eight(b, blockAddr, w);
        b.L("bra STORE;");
        b.Label("ZERO");
        foreach (string v in values)
        {
            b.L($"mov.f32 {v}, 0f00000000;");
        }

        b.Label("STORE");
        string index = b.R(), addr = b.RD();
        b.L($"mad.lo.u32 {index}, {row}, {ld}, {v0};");
        if (bf16)
        {
            var pairs = new string[4];
            for (int i = 0; i < 4; i++)
            {
                pairs[i] = b.R();
                b.L($"cvt.rn.bf16x2.f32 {pairs[i]}, {values[2 * i + 1]}, {values[2 * i]};");
            }

            b.L($"mul.wide.u32 {addr}, {index}, 2;");
            b.L($"add.u64 {addr}, {po}, {addr};");
            b.L($"st.global.v4.b32 [{addr}], {{{pairs[0]}, {pairs[1]}, {pairs[2]}, {pairs[3]}}};");
        }
        else
        {
            b.L($"mul.wide.u32 {addr}, {index}, 4;");
            b.L($"add.u64 {addr}, {po}, {addr};");
            b.L($"st.global.v4.f32 [{addr}], {{{values[0]}, {values[1]}, {values[2]}, {values[3]}}};");
            b.L($"st.global.v4.f32 [{addr}+16], {{{values[4]}, {values[5]}, {values[6]}, {values[7]}}};");
        }

        b.Label("DONE");
        b.L("ret;");
        return b.Finish("", entry, ".param .u64 p_w, .param .u64 p_out, .param .u32 p_rows, .param .u32 p_k, .param .u32 p_ld");
    }

    // y [m, n] = x [m, k] · Wᵀ for up to `rows` rows of x: a warp a weight row (8 a block of 256 threads), each lane over
    // 8-value chunks c = lane, lane + 32, ..., the sums added across the warp by butterflies.
    private static string SuperGemv(SuperBlock s, string entry, int rows, bool many = false, Epilogue epilogue = Epilogue.None) =>
        BlockGemv(new BlockDecode(s.Values, s.Bytes, (b, block, w) => SuperEight(b, s, block, w)), entry, rows, many, epilogue);

    private static string BlockGemv(BlockDecode s, string entry, int rows, bool many = false, Epilogue epilogue = Epilogue.None)
    {
        var b = new PtxBuilder();
        if (epilogue != Epilogue.None)
        {
            b.Declare(EpilogueRegisters);
        }

        string px = Param64(b, "p_x"), pw = Param64(b, "p_w"), py = Param64(b, "p_y"), m = Param32(b, "p_m"), n = Param32(b, "p_n"), k = Param32(b, "p_k");
        string tid = b.R(), ctaid = b.R(), warp = b.R(), lane = b.R(), col = b.R(), pEnd = b.P();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {ctaid}, %ctaid.x;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {lane}, {tid}, 31;");
        b.L($"mad.lo.u32 {col}, {ctaid}, 8, {warp};");
        if (many)
        {
            // The grouped products' column to its product's own: past a product's rows, the next one's weight, output, rows.
            string p1 = b.P(), p2 = b.P();
            string w1 = Param64(b, "p_w1"), y1 = Param64(b, "p_y1"), n1 = Param32(b, "p_n1");
            b.L($"setp.ge.u32 {p1}, {col}, {n};");
            b.L($"@{p1} sub.u32 {col}, {col}, {n};");
            b.L($"@{p1} mov.u64 {pw}, {w1};");
            b.L($"@{p1} mov.u64 {py}, {y1};");
            b.L($"@{p1} mov.u32 {n}, {n1};");
            string w2 = Param64(b, "p_w2"), y2 = Param64(b, "p_y2"), n2 = Param32(b, "p_n2");
            b.L($"setp.ge.u32 {p2}, {col}, {n};");
            b.L($"and.pred {p2}, {p2}, {p1};");
            b.L($"@{p2} sub.u32 {col}, {col}, {n};");
            b.L($"@{p2} mov.u64 {pw}, {w2};");
            b.L($"@{p2} mov.u64 {py}, {y2};");
            b.L($"@{p2} mov.u32 {n}, {n2};");
        }

        b.L($"setp.ge.u32 {pEnd}, {col}, {n};");
        b.L($"@{pEnd} bra DONE;");
        string per = b.R(), rowBase = b.RD(), chunks = b.R(), c = b.R();
        b.L($"shr.u32 {per}, {k}, {Log2(s.Values)};");
        b.L($"mul.wide.u32 {rowBase}, {col}, {per};");
        b.L($"mul.lo.u64 {rowBase}, {rowBase}, {s.Bytes};");
        b.L($"add.u64 {rowBase}, {pw}, {rowBase};");
        b.L($"shr.u32 {chunks}, {k}, 3;");
        var acc = new string[rows];
        var has = new string[rows];
        for (int i = 0; i < rows; i++)
        {
            acc[i] = b.F();
            has[i] = b.P();
            b.L($"mov.f32 {acc[i]}, 0f00000000;");
            b.L($"setp.lt.u32 {has[i]}, {i}, {m};");
        }

        b.L($"mov.u32 {c}, {lane};");
        b.Label("CHUNK");
        string pDone = b.P();
        b.L($"setp.ge.u32 {pDone}, {c}, {chunks};");
        b.L($"@{pDone} bra SUM;");
        string v0 = b.R(), blk = b.R(), w = b.R(), at = b.RD(), blockAddr = b.RD();
        b.L($"shl.b32 {v0}, {c}, 3;");
        b.L($"shr.u32 {blk}, {v0}, {Log2(s.Values)};");
        b.L($"and.b32 {w}, {v0}, {s.Values - 1};");
        b.L($"mul.wide.u32 {at}, {blk}, {s.Bytes};");
        b.L($"add.u64 {blockAddr}, {rowBase}, {at};");
        var values = s.Eight(b, blockAddr, w);
        for (int i = 0; i < rows; i++)
        {
            string index = b.R(), addr = b.RD();
            var xv = new string[8];
            for (int e = 0; e < 8; e++)
            {
                xv[e] = b.F();
                b.L($"mov.f32 {xv[e]}, 0f00000000;");
            }

            b.L($"mad.lo.u32 {index}, {i}, {k}, {v0};");
            b.L($"mul.wide.u32 {addr}, {index}, 4;");
            b.L($"add.u64 {addr}, {px}, {addr};");
            b.L($"@{has[i]} ld.global.v4.f32 {{{xv[0]}, {xv[1]}, {xv[2]}, {xv[3]}}}, [{addr}];");
            b.L($"@{has[i]} ld.global.v4.f32 {{{xv[4]}, {xv[5]}, {xv[6]}, {xv[7]}}}, [{addr}+16];");
            for (int e = 0; e < 8; e++)
            {
                b.L($"fma.rn.f32 {acc[i]}, {values[e]}, {xv[e]}, {acc[i]};");
            }
        }

        b.L($"add.u32 {c}, {c}, 32;");
        b.L("bra CHUNK;");
        b.Label("SUM");
        for (int i = 0; i < rows; i++)
        {
            for (int off = 16; off >= 1; off >>= 1)
            {
                string o = b.F();
                b.L($"shfl.sync.bfly.b32 {o}, {acc[i]}, {off}, 0x1f, 0xffffffff;");
                b.L($"add.rn.f32 {acc[i]}, {acc[i]}, {o};");
            }
        }

        string first = b.P();
        b.L($"setp.eq.u32 {first}, {lane}, 0;");
        b.L($"@!{first} bra DONE;");
        for (int i = 0; i < rows; i++)
        {
            string index = b.R(), addr = b.RD(), store = b.P();
            b.L($"mad.lo.u32 {index}, {i}, {n}, {col};");
            b.L($"mul.wide.u32 {addr}, {index}, 4;");
            b.L($"add.u64 {addr}, {py}, {addr};");
            b.L($"setp.lt.u32 {store}, {i}, {m};");
            b.L($"@{store} st.global.f32 [{addr}], {acc[i]};");
        }

        b.Label("DONE");
        b.L(EpilogueCode(epilogue));
        b.L("ret;");
        return b.Finish("", entry, ".param .u64 p_x, .param .u64 p_w, .param .u64 p_y, .param .u32 p_m, .param .u32 p_n, .param .u32 p_k"
            + (many ? ", .param .u64 p_w1, .param .u64 p_y1, .param .u32 p_n1, .param .u64 p_w2, .param .u64 p_y2, .param .u32 p_n2" : "") + EpilogueParameters(epilogue));
    }
}

// The experiment's kernels for grid-codebook formats (ElementType.Grid: ggml's IQ2_XXS, IQ3_XXS), generated from the
// description alone, on the super-block formats' widening and few-row generators (BlockWiden, BlockGemv): 8 values a
// thread from the module's read-only global tables (the sub-scale, entry and sign pieces; the grid's magnitudes; the sign masks:
// global, not constant memory, since each lane reads its own entry, which constant memory serves one address at a time),
// each (scale · magnitude) · sign as GridBlock.Decode computes it.
internal static partial class PtxKernels
{
    internal static string GridCode(GridBlock g)
    {
        var text = new StringBuilder(string.Create(CultureInfo.InvariantCulture, $"{g.Values}x{g.Bytes}s{g.ScaleOffset}v{g.SubValues}o{g.SubScaleOffset}f{g.SubScaleFactor}e{g.EntryValues}g{g.SignValues}"));
        foreach (var lists in new[] { g.SubScales, g.Entries, g.Signs })
        {
            foreach (var pieces in lists)
            {
                text.Append(';');
                foreach (var q in pieces)
                {
                    text.Append(CultureInfo.InvariantCulture, $"{q.Offset}.{q.Shift}.{q.Bits},");
                }
            }
        }

        text.Append(string.Join(",", g.Grid)).Append('|').Append(string.Join(",", g.SignTable));
        uint hash = 2166136261;
        foreach (char c in text.ToString())
        {
            hash = (hash ^ c) * 16777619;
        }

        return string.Create(CultureInfo.InvariantCulture, $"g{g.Values}x{g.Bytes}_{hash:x8}");
    }

    /// <summary>Whether these kernels widen the grid description: powers of two where they shift, 8 values within one sign group and whole entries.</summary>
    internal static bool GridFits(GridBlock g)
    {
        static bool Pow2(int v) => v > 0 && (v & (v - 1)) == 0;
        bool pieces = g.SubScales.Concat(g.Entries).Concat(g.Signs).All(list => list.Count <= 2 && list.All(q => q.Offset is >= 0 and < 256 && q.Shift < 8 && q.Bits is > 0 and <= 8));
        return Pow2(g.Values) && Pow2(g.SubValues) && g.SubValues >= 8 && Pow2(g.EntryValues) && g.EntryValues <= 8 && g.SignValues == 8 && g.Scale.Bits == 16
            && g.ScaleOffset % 2 == 0 && g.Grid.All(v => v is >= -128 and <= 127) && g.SignTable.All(v => v is >= 0 and <= 255) && pieces;
    }

    internal static string[] GridNames(GridBlock g)
    {
        string c = GridCode(g);
        return [$"grid_widen_{c}_f32", $"grid_widen_{c}_bf16", .. TypedGemvRows.ToArray().Select(r => string.Create(CultureInfo.InvariantCulture, $"grid_gemv_{c}_r{r}")),
            .. TypedGemvRows.ToArray().Select(r => string.Create(CultureInfo.InvariantCulture, $"grid_gemv_{c}_r{r}_many")),
            .. TypedGemvRows.ToArray().Select(r => string.Create(CultureInfo.InvariantCulture, $"grid_gemv_{c}_r{r}_many_gated")),
            .. TypedGemvRows.ToArray().Select(r => string.Create(CultureInfo.InvariantCulture, $"grid_gemv_{c}_r{r}_addnorm"))];
    }

    // Pieces packed as SuperModule's table packs them: offset | shift << 8 | bits << 12 | place << 16, two a value.
    private static IEnumerable<uint> PieceWords(IReadOnlyList<IReadOnlyList<ScalePiece>> lists)
    {
        foreach (var list in lists)
        {
            int at = 0;
            for (int i = 0; i < 2; i++)
            {
                if (i < list.Count)
                {
                    var q = list[i];
                    yield return (uint)(q.Offset | q.Shift << 8 | q.Bits << 12 | at << 16);
                    at += q.Bits;
                }
                else
                {
                    yield return 0;
                }
            }
        }
    }

    /// <summary>The module for <paramref name="g"/>: its tables in read-only global memory, the widenings and the few-row products.</summary>
    internal static string GridModule(GridBlock g)
    {
        var names = GridNames(g);
        var sb = new StringBuilder(NativeHeader);
        sb.Append(CultureInfo.InvariantCulture, $".global .align 4 .b32 grid_sub[{g.SubScales.Count * 2}] = {{{string.Join(", ", PieceWords(g.SubScales))}}};\n");
        sb.Append(CultureInfo.InvariantCulture, $".global .align 4 .b32 grid_entry[{g.Entries.Count * 2}] = {{{string.Join(", ", PieceWords(g.Entries))}}};\n");
        sb.Append(CultureInfo.InvariantCulture, $".global .align 4 .b32 grid_sign[{g.Signs.Count * 2}] = {{{string.Join(", ", PieceWords(g.Signs))}}};\n");
        sb.Append(CultureInfo.InvariantCulture, $".global .align 8 .b8 grid_values[{g.Grid.Count}] = {{{string.Join(", ", g.Grid.Select(v => (byte)(sbyte)v))}}};\n");
        sb.Append(CultureInfo.InvariantCulture, $".global .align 4 .b8 grid_masks[{g.SignTable.Count}] = {{{string.Join(", ", g.SignTable)}}};\n");
        var decode = new BlockDecode(g.Values, g.Bytes, (b, block, w) => GridEight(b, g, block, w));
        sb.Append(BlockWiden(decode, names[0], bf16: false));
        sb.Append(BlockWiden(decode, names[1], bf16: true));
        int rows = TypedGemvRows.Length;
        for (int i = 0; i < rows; i++)
        {
            sb.Append(BlockGemv(decode, names[2 + i], TypedGemvRows[i]));
            sb.Append(BlockGemv(decode, names[2 + rows + i], TypedGemvRows[i], many: true));
            sb.Append(BlockGemv(decode, names[2 + 2 * rows + i], TypedGemvRows[i], many: true, epilogue: Epilogue.Gated));
            sb.Append(BlockGemv(decode, names[2 + 3 * rows + i], TypedGemvRows[i], epilogue: Epilogue.AddNorm));
        }

        return sb.ToString();
    }

    // A field from its two pieces at table `table`'s entry `index` (a register): each piece a byte, shift and bits.
    private static string GridField(PtxBuilder b, string table, string index, string block)
    {
        string value = b.R(), base64 = b.RD();
        b.L($"mov.u32 {value}, 0;");
        b.L($"mov.u64 {base64}, {table};");
        for (int i = 0; i < 2; i++)
        {
            string e = b.R(), at = b.RD(), off = b.R(), sh = b.R(), bits = b.R(), place = b.R(), wide = b.RD(), by = b.R(), f = b.R();
            b.L($"mad.lo.u32 {e}, {index}, 2, {i};");
            b.L($"mul.wide.u32 {at}, {e}, 4;");
            b.L($"add.u64 {at}, {base64}, {at};");
            b.L($"ld.global.nc.u32 {e}, [{at}];");
            b.L($"and.b32 {off}, {e}, 255;");
            b.L($"bfe.u32 {sh}, {e}, 8, 4;");
            b.L($"bfe.u32 {bits}, {e}, 12, 4;");
            b.L($"bfe.u32 {place}, {e}, 16, 5;");
            b.L($"cvt.u64.u32 {wide}, {off};");
            b.L($"add.u64 {wide}, {block}, {wide};");
            b.L($"ld.global.u8 {by}, [{wide}];");
            b.L($"bfe.u32 {f}, {by}, {sh}, {bits};");
            b.L($"shl.b32 {f}, {f}, {place};");
            b.L($"or.b32 {value}, {value}, {f};");
        }

        return value;
    }

    // Values [w, w + 8) of the grid block at `block` (w a multiple of 8): one sign group, 8 / EntryValues entries, one sub-block.
    private static string[] GridEight(PtxBuilder b, GridBlock g, string block, string w)
    {
        string h = b.R(), d = b.F();
        b.L($"ld.global.u16 {h}, [{block}+{g.ScaleOffset}];");
        if (g.Scale.Exponent == 8)
        {
            b.L($"shl.b32 {h}, {h}, 16;");
            b.L($"mov.b32 {d}, {h};");
        }
        else
        {
            string hh = b.H();
            b.L($"cvt.u16.u32 {hh}, {h};");
            b.L($"cvt.f32.f16 {d}, {hh};");
        }

        string j = b.R(), sf = b.F(), scale = b.F();
        b.L($"shr.u32 {j}, {w}, {Log2(g.SubValues)};");
        string s = GridField(b, "grid_sub", j, block);
        b.L($"cvt.rn.f32.u32 {sf}, {s};");
        b.L(string.Create(CultureInfo.InvariantCulture, $"add.rn.f32 {sf}, {sf}, 0f{BitConverter.SingleToInt32Bits(g.SubScaleOffset):X8};"));
        b.L($"mul.rn.f32 {scale}, {d}, {sf};");
        b.L(string.Create(CultureInfo.InvariantCulture, $"mul.rn.f32 {scale}, {scale}, 0f{BitConverter.SingleToInt32Bits(g.SubScaleFactor):X8};"));
        string group = b.R(), maskBase = b.RD(), maskAt = b.RD(), mask = b.R();
        b.L($"shr.u32 {group}, {w}, 3;");
        string signIndex = GridField(b, "grid_sign", group, block);
        b.L($"mov.u64 {maskBase}, grid_masks;");
        b.L($"cvt.u64.u32 {maskAt}, {signIndex};");
        b.L($"add.u64 {maskAt}, {maskBase}, {maskAt};");
        b.L($"ld.global.nc.u8 {mask}, [{maskAt}];");
        var values = new string[8];
        string gridBase = b.RD();
        b.L($"mov.u64 {gridBase}, grid_values;");
        for (int e = 0; e < 8 / g.EntryValues; e++)
        {
            string entry = b.R(), entryIndex = b.R(), at = b.RD();
            b.L($"shr.u32 {entry}, {w}, {Log2(g.EntryValues)};");
            if (e > 0)
            {
                b.L($"add.u32 {entry}, {entry}, {e};");
            }

            string index = GridField(b, "grid_entry", entry, block);
            b.L($"mul.lo.u32 {entryIndex}, {index}, {g.EntryValues};");
            b.L($"cvt.u64.u32 {at}, {entryIndex};");
            b.L($"add.u64 {at}, {gridBase}, {at};");
            // The entry's magnitudes in one read: 8 bytes as two words, 4 as one.
            var words = new string[(g.EntryValues + 3) / 4];
            if (g.EntryValues == 8)
            {
                string pair = b.RD();
                words[0] = b.R();
                words[1] = b.R();
                b.L($"ld.global.nc.u64 {pair}, [{at}];");
                b.L($"mov.b64 {{{words[0]}, {words[1]}}}, {pair};");
            }
            else
            {
                words[0] = b.R();
                b.L(g.EntryValues == 4 ? $"ld.global.nc.u32 {words[0]}, [{at}];" : g.EntryValues == 2 ? $"ld.global.nc.u16 {words[0]}, [{at}];" : $"ld.global.nc.u8 {words[0]}, [{at}];");
            }

            for (int i = 0; i < g.EntryValues; i++)
            {
                int v = e * g.EntryValues + i;
                string mag = b.R(), magF = b.F(), bit = b.R(), neg = b.P();
                values[v] = b.F();
                b.L($"bfe.s32 {mag}, {words[i / 4]}, {8 * (i % 4)}, 8;");
                b.L($"cvt.rn.f32.s32 {magF}, {mag};");
                b.L($"mul.rn.f32 {values[v]}, {scale}, {magF};");
                b.L($"bfe.u32 {bit}, {mask}, {v}, 1;");
                b.L($"setp.ne.u32 {neg}, {bit}, 0;");
                b.L($"@{neg} neg.f32 {values[v]}, {values[v]};");
            }
        }

        return values;
    }
}
