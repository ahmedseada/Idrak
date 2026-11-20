// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using Idrak.Adad.Types;

namespace Idrak.Adad.Devices.Cuda;

// The widening of a super-block description's rows (copied from Idrak's PtxKernels.Super, the parts the control lane runs):
// a thread widens 8 consecutive values of a block (SuperEight: the super-scale and minimum, the sub-block's scale and minimum
// assembled from their pieces, read from a table in the module's constant memory by the sub-block, each code from its
// planes' fields), then writes them as bfloat16 rows padded to the native kind's rows. Every value is (d · scale) · code −
// dmin · min, rounded at each step as the CPU's description decoder rounds it (SuperBlock.Decode).
internal static class SuperKernels
{
    /// <summary>A description's name part: k{Values}x{Bytes}_{FNV-1a of its canonical text} (made once a description).</summary>
    internal static string SuperCode(SuperBlock s) => string.Create(CultureInfo.InvariantCulture, $"k{s.Values}x{s.Bytes}_{s.Hash:x8}");

    // The most pieces a sub-block's scale or minimum takes (the table's entries a sub-block, each).
    internal static int SuperPieces(SuperBlock s) => Math.Max(s.SubScales.Max(p => p.Count), s.SubMins?.Max(p => p.Count) ?? 0);

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

    /// <summary>The bfloat16 widening's kernel name.</summary>
    internal static string WidenBf16Name(SuperBlock s) => $"super_widen_{SuperCode(s)}_bf16";

    /// <summary>The module of the bfloat16 widening: the pieces' table in constant memory, then the kernel.</summary>
    internal static string WidenModule(SuperBlock s) =>
        NativeKernels.NativeHeader + SuperPiecesDeclaration(s) + SuperWiden(s, WidenBf16Name(s), bf16: true);

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

    internal static int Log2(int v) => System.Numerics.BitOperations.Log2((uint)v);

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
    internal static string[] SuperEight(PtxBuilder b, SuperBlock s, string block, string w)
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
    internal static string SuperWiden(SuperBlock s, string entry, bool bf16)
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
        var values = SuperEight(b, s, blockAddr, w);
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
}
