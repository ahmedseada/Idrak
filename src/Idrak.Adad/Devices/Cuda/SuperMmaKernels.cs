// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Adad.Types;
using static Idrak.Adad.Devices.Cuda.NativeKernels;

namespace Idrak.Adad.Devices.Cuda;

// The prompt product's kernels that read a super-block weight as stored (decision 23), built on the native kernel's
// machinery (its block of 8 warps, 2 × 4, over 128 × BN of C; the ring of cp.async stages; ldmatrix; the swizzle; the k split
// by grid z; the store), generated from the description alone:
//   A (DecodeMma): x as the control packs it (bfloat16 rows), W's stored bytes staged as they are (each row's 16-byte
//     header and the 32 code bytes of the 64-value group a stage reads), each fragment's four values decoded in registers
//     just before the multiply with exactly the widening's float32 steps and one cvt.rn.bf16x2, so the multiply's inputs
//     are the control's bit for bit;
//   B (Int8Mma): x rounded once per call to int8 blocks of 32 with their scales and block sums (QuantizeX), W's 4-bit codes
//     widened to int8 in registers (eight a 32-bit operation pair), int8 tensor cores (mma m16n8k32 s8.s8.s32) giving one
//     int32 sum per 32-value sub-block, then sums += (d·sc·xs)·isum − (dmin·m)·xsum in float32.
// In both, a sub-block's scale d·(sc + bias) and minimum dmin·m are unpacked once per sub-block a block (not per value): the
// thread that staged a row's header computes them for the stage's sub-blocks into the stage's scale area, after its own
// copies land and before the barrier that hands the stage to the warps.
internal static class SuperMmaKernels
{
    /// <summary>Threads of a block: the native kernel's 8 warps.</summary>
    internal const int Threads = NativeMmaThreads;

    /// <summary>The header a row stages for every stage: the block's first 16 bytes (d, dmin, the sub-blocks' scale and minimum pieces).</summary>
    internal const int HeaderBytes = 16;

    /// <summary>
    /// Whether A and B read <paramref name="s"/>: one plane of 4-bit codes in groups of 64 right after a 16-byte header (a
    /// group's 32 bytes hold its first 32 values in their low halves and the next 32 in their high halves), sub-blocks of 32,
    /// 16-bit d and dmin and every scale and minimum piece within the header, blocks a whole number of 16-byte chunks; the
    /// widening's own conditions (SuperFits). Matched by these fields alone, never by a type's name.
    /// </summary>
    internal static bool Fits(SuperBlock s) =>
        SuperKernels.SuperFits(s) && s.Planes.Count == 1 && s.Planes[0] is { Bits: 4, Group: 64, Offset: HeaderBytes }
        && s.SubValues == 32 && s.Values % 64 == 0 && s.Bytes == HeaderBytes + s.Values / 2 && s.Bytes % 16 == 0 && s.Scale.Bits == 16
        && s.ScaleOffset + 2 <= HeaderBytes && s.MinOffset + 2 <= HeaderBytes
        && (s.SubMins is null ? s.SubScales : s.SubScales.Concat(s.SubMins)).All(list => list.All(q => q.Offset < HeaderBytes));

    /// <summary>B's further condition: the codes are the unsigned 4-bit fields as they are (no bias), which int8 holds exactly.</summary>
    internal static bool FitsInt8(SuperBlock s) => Fits(s) && s.CodeBias == 0;

    /// <summary>A's shapes (bytes of x's bfloat16 row a stage: 64 or 128, i.e. 32 or 64 values; stages; columns; swizzle).</summary>
    internal static readonly NativeMmaConfig[] DecodeConfigs =
    [
        new(64, 3, 128, false), new(64, 4, 128, false), new(128, 2, 128, false), new(64, 3, 256, false), new(64, 3, 256, true),
        new(64, 4, 256, true), new(128, 2, 256, true), new(128, 3, 128, true), new(64, 6, 128, true), new(128, 4, 128, true),
    ];

    /// <summary>B's shapes (bytes of x's int8 row a stage = its values: 64 or 128; stages; columns; swizzle).</summary>
    internal static readonly NativeMmaConfig[] Int8Configs =
    [
        new(64, 3, 128, true), new(64, 4, 128, true), new(64, 5, 128, true), new(64, 4, 128, false), new(64, 2, 256, true),
        new(64, 3, 256, true), new(128, 2, 128, true), new(128, 2, 256, true),
    ];

    /// <summary>Values of k a stage of A takes (x's bfloat16 row bytes / 2).</summary>
    internal static int DecodeStageValues(NativeMmaConfig c) => c.RowBytes / 2;

    /// <summary>Values of k a stage of B takes (x's int8 row bytes).</summary>
    internal static int Int8StageValues(NativeMmaConfig c) => c.RowBytes;

    // A stage's areas: x's rows, (B) x's scales, W's rows (header + code bytes), W's sub-block scales and minimums.
    private readonly record struct Layout(int XBytes, int XScaleBytes, int WStride, int WBytes, int ScaleBytes)
    {
        public int XScaleBase => XBytes;

        public int WBase => XBytes + XScaleBytes;

        public int ScaleBase => WBase + WBytes;

        public int StageBytes => ScaleBase + ScaleBytes;
    }

    private static Layout DecodeLayout(NativeMmaConfig c)
    {
        int subs = DecodeStageValues(c) / 32;
        return new Layout(128 * (c.RowBytes + (c.Swizzle ? 0 : 16)), 0, HeaderBytes + 32, c.Bn * (HeaderBytes + 32), c.Bn * subs * 8);
    }

    private static Layout Int8Layout(NativeMmaConfig c)
    {
        int values = Int8StageValues(c), subs = values / 32, wStride = HeaderBytes + values / 2;
        return new Layout(128 * (c.RowBytes + (c.Swizzle ? 0 : 16)), 128 * subs * 8, wStride, c.Bn * wStride, c.Bn * subs * 8);
    }

    /// <summary>Shared bytes A's shape takes.</summary>
    internal static int DecodeShared(NativeMmaConfig c) => c.Stages * DecodeLayout(c).StageBytes;

    /// <summary>Shared bytes B's shape takes.</summary>
    internal static int Int8Shared(NativeMmaConfig c) => c.Stages * Int8Layout(c).StageBytes;

    internal static string DecodeName(SuperBlock s, NativeMmaConfig c) =>
        string.Create(CultureInfo.InvariantCulture, $"super_mma_bf16_{SuperKernels.SuperCode(s)}_r{c.RowBytes}_s{c.Stages}_n{c.Bn}{(c.Swizzle ? "_x" : "")}");

    internal static string Int8Name(SuperBlock s, NativeMmaConfig c) =>
        string.Create(CultureInfo.InvariantCulture, $"super_mma_s8_{SuperKernels.SuperCode(s)}_r{c.RowBytes}_s{c.Stages}_n{c.Bn}{(c.Swizzle ? "_x" : "")}");

    /// <summary>A's module for shape <paramref name="c"/>: the pieces' table and the kernel.</summary>
    internal static string DecodeModule(SuperBlock s, NativeMmaConfig c) => Lf(Generate(s, c, int8: false));

    /// <summary>B's module for shape <paramref name="c"/>: the pieces' table and the kernel.</summary>
    internal static string Int8Module(SuperBlock s, NativeMmaConfig c) => Lf(Generate(s, c, int8: true));

    // The kernel. A's parameters: p_a (x packed bfloat16, rows of p_ld values), p_b (W as stored, rows of p_wrow bytes),
    // p_c, p_m, p_n, p_kp (k), p_ld, p_kblocks (the split's blocks), p_wrow. B's: p_a (x int8, rows of k bytes), p_xs (x's
    // [m, k / 32] pairs of scale and block sum), p_b, p_c, p_m, p_n, p_kp (k), p_kblocks, p_wrow.
    private static string Generate(SuperBlock s, NativeMmaConfig cfg, bool int8)
    {
        int rowBytes = cfg.RowBytes, stages = cfg.Stages, bn = cfg.Bn;
        bool swizzle = cfg.Swizzle;
        string entry = int8 ? Int8Name(s, cfg) : DecodeName(s, cfg);
        const int BM = 128;
        int wn = bn / 4, nt = wn / 8;
        var layout = int8 ? Int8Layout(cfg) : DecodeLayout(cfg);
        int stageValues = int8 ? Int8StageValues(cfg) : DecodeStageValues(cfg), blockShift = SuperKernels.Log2(stageValues);
        int subs = stageValues / 32, groups = Math.Max(1, stageValues / 64), codeChunks = 2 * groups;
        int stride = rowBytes + (swizzle ? 0 : 16), wStride = layout.WStride, stageBytes = layout.StageBytes, totalShared = stages * stageBytes;
        int chunks = rowBytes / 16, copiesA = BM * chunks / Threads, codeCopies = bn * codeChunks / Threads;
        int ksteps = int8 ? subs : rowBytes / 32;                          // k32 steps (B) or k16 steps (A) a stage
        int pieces = SuperKernels.SuperPieces(s);
        if (chunks is not (4 or 8) || bn * codeChunks % Threads != 0 || bn > Threads)
        {
            throw new NotSupportedException($"shape {cfg}");
        }

        var b = new PtxBuilder();
        string tid = b.R(), bx = b.R(), by = b.R(), warp = b.R(), lane = b.R(), wmR = b.R(), wnR = b.R();
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mov.u32 {bx}, %ctaid.x;");
        b.L($"mov.u32 {by}, %ctaid.y;");
        b.L($"shr.u32 {warp}, {tid}, 5;");
        b.L($"and.b32 {lane}, {tid}, 31;");
        b.L($"shr.u32 {wmR}, {warp}, 2;");
        b.L($"and.b32 {wnR}, {warp}, 3;");
        string Param64(string name)
        {
            string r = b.RD();
            b.L($"ld.param.u64 {r}, [{name}];");
            return r;
        }

        string Param32(string name)
        {
            string r = b.R();
            b.L($"ld.param.u32 {r}, [{name}];");
            return r;
        }

        string pa = Param64("p_a"), pxs = int8 ? Param64("p_xs") : "", pb = Param64("p_b"), pc = Param64("p_c");
        string m = Param32("p_m"), n = Param32("p_n"), kp = Param32("p_kp"), ld = int8 ? kp : Param32("p_ld");
        string kblocks = Param32("p_kblocks"), wrow = Param32("p_wrow");
        string split = b.R(), kb0 = b.R(), partOff = b.RD(), mn = b.R();
        b.L($"mov.u32 {split}, %ctaid.z;");
        b.L($"mul.lo.u32 {kb0}, {split}, {kblocks};");
        b.L($"mul.lo.u32 {mn}, {m}, {n};");
        b.L($"mul.wide.u32 {partOff}, {mn}, {split};");
        b.L($"shl.b64 {partOff}, {partOff}, 2;");
        b.L($"add.u64 {pc}, {pc}, {partOff};");
        string row0 = b.R(), col0 = b.R(), smem = b.R();
        b.L($"mul.lo.u32 {row0}, {by}, {BM};");
        b.L($"mul.lo.u32 {col0}, {bx}, {bn};");
        b.L($"mov.u32 {smem}, smem;");

        var c = new string[4, nt, 4];
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < nt; j++)
            {
                for (int q = 0; q < 4; q++)
                {
                    c[i, j, q] = b.F();
                    b.L($"mov.f32 {c[i, j, q]}, 0f00000000;");
                }
            }
        }

        // x's row bytes: p_ld bfloat16 values (A) or k bytes (B).
        string xRowBytes = b.RD();
        if (int8)
        {
            b.L($"cvt.u64.u32 {xRowBytes}, {ld};");
        }
        else
        {
            b.L($"mul.wide.u32 {xRowBytes}, {ld}, 2;");
        }

        // x's copies, as the native kernel's: chunk j·256 + tid of the stage's 128 rows.
        var xg = new string[copiesA];
        var xs = new string[copiesA];
        var xsz = new string[copiesA];
        for (int j = 0; j < copiesA; j++)
        {
            string chunk = b.R(), row = b.R(), part = b.R(), gRow = b.R(), p = b.P(), t64 = b.RD(), off = b.RD(), rowSel = b.R();
            b.L($"add.u32 {chunk}, {tid}, {j * Threads};");
            b.L($"shr.u32 {row}, {chunk}, {SuperKernels.Log2(chunks)};");
            b.L($"and.b32 {part}, {chunk}, {chunks - 1};");
            b.L($"add.u32 {gRow}, {row0}, {row};");
            b.L($"setp.lt.u32 {p}, {gRow}, {m};");
            b.L($"selp.u32 {rowSel}, {gRow}, 0, {p};");
            xg[j] = b.RD();
            xs[j] = b.R();
            xsz[j] = b.R();
            b.L($"cvt.u64.u32 {t64}, {rowSel};");
            b.L($"mul.lo.u64 {t64}, {t64}, {xRowBytes};");
            b.L($"mul.wide.u32 {off}, {part}, 16;");
            b.L($"add.u64 {t64}, {t64}, {off};");
            b.L($"add.u64 {xg[j]}, {pa}, {t64};");
            b.L($"mad.lo.u32 {xs[j]}, {row}, {stride}, {smem};");
            if (swizzle)
            {
                string swz = b.R(), partX = b.R();
                NativeSwizzle(b, swz, row, rowBytes);
                b.L($"xor.b32 {partX}, {part}, {swz};");
                b.L($"mad.lo.u32 {xs[j]}, {partX}, 16, {xs[j]};");
            }
            else
            {
                b.L($"mad.lo.u32 {xs[j]}, {part}, 16, {xs[j]};");
            }

            b.L($"selp.u32 {xsz[j]}, 16, 0, {p};");
        }

        // B: x's scales, chunk q of the stage's 128 rows × subs pairs (16 bytes: two sub-blocks' pairs).
        int scaleChunksPerRow = subs / 2, xScaleChunks = BM * scaleChunksPerRow, xScaleCopies = (xScaleChunks + Threads - 1) / Threads;
        var sg = new string[int8 ? xScaleCopies : 0];
        var ss = new string[sg.Length];
        var ssz = new string[sg.Length];
        var sp = new string[sg.Length];
        string xScaleRowBytes = "";
        if (int8)
        {
            xScaleRowBytes = b.RD();
            string pairs = b.R();
            b.L($"shr.u32 {pairs}, {kp}, 5;");
            b.L($"mul.wide.u32 {xScaleRowBytes}, {pairs}, 8;");
            for (int j = 0; j < xScaleCopies; j++)
            {
                string chunk = b.R(), row = b.R(), part = b.R(), gRow = b.R(), p = b.P(), inside = b.P(), t64 = b.RD(), off = b.RD(), rowSel = b.R();
                b.L($"add.u32 {chunk}, {tid}, {j * Threads};");
                b.L($"setp.lt.u32 {inside}, {chunk}, {xScaleChunks};");
                b.L($"shr.u32 {row}, {chunk}, {SuperKernels.Log2(scaleChunksPerRow)};");
                b.L($"and.b32 {part}, {chunk}, {scaleChunksPerRow - 1};");
                b.L($"add.u32 {gRow}, {row0}, {row};");
                b.L($"setp.lt.and.u32 {p}, {gRow}, {m}, {inside};");
                b.L($"selp.u32 {rowSel}, {gRow}, 0, {p};");
                sg[j] = b.RD();
                ss[j] = b.R();
                ssz[j] = b.R();
                b.L($"cvt.u64.u32 {t64}, {rowSel};");
                b.L($"mul.lo.u64 {t64}, {t64}, {xScaleRowBytes};");
                b.L($"mul.wide.u32 {off}, {part}, 16;");
                b.L($"add.u64 {t64}, {t64}, {off};");
                b.L($"add.u64 {sg[j]}, {pxs}, {t64};");
                b.L($"mad.lo.u32 {ss[j]}, {row}, {subs * 8}, {smem};");
                b.L($"mad.lo.u32 {ss[j]}, {part}, 16, {ss[j]};");
                b.L($"add.u32 {ss[j]}, {ss[j]}, {layout.XScaleBase};");
                b.L($"selp.u32 {ssz[j]}, 16, 0, {p};");
                sp[j] = inside;                                             // a chunk past the stage's rows is not copied at all
            }
        }

        // W's copies: row tid's header (threads below BN), and the code chunks q = j·256 + tid (row q / codeChunks).
        string hasHeader = b.P(), hdrRow = b.R(), hdrOk = b.P(), hdrSel = b.R(), hdrG = b.RD(), hdrS = b.R(), hdrSz = b.R(), t = b.RD();
        b.L($"setp.lt.u32 {hasHeader}, {tid}, {bn};");
        b.L($"add.u32 {hdrRow}, {col0}, {tid};");
        b.L($"setp.lt.and.u32 {hdrOk}, {hdrRow}, {n}, {hasHeader};");
        b.L($"selp.u32 {hdrSel}, {hdrRow}, 0, {hdrOk};");
        b.L($"mul.wide.u32 {t}, {hdrSel}, {wrow};");
        b.L($"add.u64 {hdrG}, {pb}, {t};");
        b.L($"mad.lo.u32 {hdrS}, {tid}, {wStride}, {smem};");
        b.L($"add.u32 {hdrS}, {hdrS}, {layout.WBase};");
        b.L($"selp.u32 {hdrSz}, 16, 0, {hdrOk};");
        var cg = new string[codeCopies];
        var cs = new string[codeCopies];
        var csz = new string[codeCopies];
        for (int j = 0; j < codeCopies; j++)
        {
            string chunk = b.R(), row = b.R(), part = b.R(), gRow = b.R(), p = b.P(), rowSel = b.R(), t64 = b.RD();
            b.L($"add.u32 {chunk}, {tid}, {j * Threads};");
            b.L($"shr.u32 {row}, {chunk}, {SuperKernels.Log2(codeChunks)};");
            b.L($"and.b32 {part}, {chunk}, {codeChunks - 1};");
            b.L($"add.u32 {gRow}, {col0}, {row};");
            b.L($"setp.lt.u32 {p}, {gRow}, {n};");
            b.L($"selp.u32 {rowSel}, {gRow}, 0, {p};");
            cg[j] = b.RD();
            cs[j] = b.R();
            csz[j] = b.R();
            b.L($"mul.wide.u32 {t64}, {rowSel}, {wrow};");
            b.L($"add.u64 {cg[j]}, {pb}, {t64};");
            b.L($"mul.wide.u32 {t64}, {part}, 16;");
            b.L($"add.u64 {t64}, {t64}, {HeaderBytes};");
            b.L($"add.u64 {cg[j]}, {cg[j]}, {t64};");
            b.L($"mad.lo.u32 {cs[j]}, {row}, {wStride}, {smem};");
            b.L($"mad.lo.u32 {cs[j]}, {part}, 16, {cs[j]};");
            b.L($"add.u32 {cs[j]}, {cs[j]}, {layout.WBase + HeaderBytes};");
            b.L($"selp.u32 {csz[j]}, 16, 0, {p};");
        }

        // The split's blocks: those of k from kb0 on, at most p_kblocks, none when the split starts past k.
        string blocksReg = b.R();
        b.L($"shr.u32 {blocksReg}, {kp}, {blockShift};");
        b.L($"sub.s32 {blocksReg}, {blocksReg}, {kb0};");
        b.L($"max.s32 {blocksReg}, {blocksReg}, 0;");
        b.L($"min.u32 {blocksReg}, {blocksReg}, {kblocks};");

        // Issues block `block` into stage `stageOff` (bytes) when the block exists; commits a group either way.
        int labels = 0;
        void Issue(string block, string stageOff)
        {
            string label = string.Create(CultureInfo.InvariantCulture, $"ISSUED{labels++}");
            string pHas = b.P(), kBytes = b.RD(), absolute = b.R(), k0 = b.R(), blk = b.R(), group = b.R(), blockOff = b.RD(), codeOff = b.RD();
            b.L($"setp.ge.u32 {pHas}, {block}, {blocksReg};");
            b.L($"@{pHas} bra {label};");
            b.L($"add.u32 {absolute}, {block}, {kb0};");
            b.L($"mul.wide.u32 {kBytes}, {absolute}, {rowBytes};");
            for (int j = 0; j < copiesA; j++)
            {
                string src = b.RD(), dst = b.R();
                b.L($"add.u64 {src}, {xg[j]}, {kBytes};");
                b.L($"add.u32 {dst}, {xs[j]}, {stageOff};");
                b.L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {xsz[j]};");
            }

            b.L($"mul.lo.u32 {k0}, {absolute}, {stageValues};");
            if (int8)
            {
                // x's scale pairs of the stage's sub-blocks: k0 / 32 pairs on.
                string pairOff = b.RD(), pairs = b.R();
                b.L($"shr.u32 {pairs}, {k0}, 5;");
                b.L($"mul.wide.u32 {pairOff}, {pairs}, 8;");
                for (int j = 0; j < xScaleCopies; j++)
                {
                    string src = b.RD(), dst = b.R();
                    b.L($"add.u64 {src}, {sg[j]}, {pairOff};");
                    b.L($"add.u32 {dst}, {ss[j]}, {stageOff};");
                    b.L($"@{sp[j]} cp.async.cg.shared.global [{dst}], [{src}], 16, {ssz[j]};");
                }
            }

            // W: the block holding k0, its header and the code bytes of the 64-value group(s) the stage reads.
            b.L($"shr.u32 {blk}, {k0}, {SuperKernels.Log2(s.Values)};");
            b.L($"and.b32 {group}, {k0}, {s.Values - 1};");
            b.L($"shr.u32 {group}, {group}, 6;");
            b.L($"mul.wide.u32 {blockOff}, {blk}, {s.Bytes};");
            b.L($"mul.wide.u32 {codeOff}, {group}, 32;");
            b.L($"add.u64 {codeOff}, {codeOff}, {blockOff};");
            {
                string src = b.RD(), dst = b.R();
                b.L($"add.u64 {src}, {hdrG}, {blockOff};");
                b.L($"add.u32 {dst}, {hdrS}, {stageOff};");
                b.L($"@{hasHeader} cp.async.cg.shared.global [{dst}], [{src}], 16, {hdrSz};");
            }

            for (int j = 0; j < codeCopies; j++)
            {
                string src = b.RD(), dst = b.R();
                b.L($"add.u64 {src}, {cg[j]}, {codeOff};");
                b.L($"add.u32 {dst}, {cs[j]}, {stageOff};");
                b.L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {csz[j]};");
            }

            b.Label(label);
            b.L("cp.async.commit_group;");
        }

        // The stage's sub-block scales and minimums, by the thread that staged the row's header (its own copies landed):
        // row tid's d · (sc_j + bias) and dmin · m_j for the stage's sub-blocks j, as the widening rounds them, into the
        // stage's scale area [sub][row] (pairs of float32).
        string table = b.RD();
        b.L($"mov.u64 {table}, super_pieces;");
        void Scales(string block, string stageOff)
        {
            string label = string.Create(CultureInfo.InvariantCulture, $"SCALED{labels++}");
            string pSkip = b.P(), absolute = b.R(), w0 = b.R(), jBase = b.R(), hdr = b.R();
            b.L($"setp.ge.u32 {pSkip}, {tid}, {bn};");
            b.L($"setp.ge.or.u32 {pSkip}, {block}, {blocksReg}, {pSkip};");
            b.L($"@{pSkip} bra {label};");
            b.L($"add.u32 {absolute}, {block}, {kb0};");
            b.L($"mul.lo.u32 {w0}, {absolute}, {stageValues};");
            b.L($"and.b32 {w0}, {w0}, {s.Values - 1};");
            b.L($"shr.u32 {jBase}, {w0}, {SuperKernels.Log2(s.SubValues)};");
            b.L($"add.u32 {hdr}, {hdrS}, {stageOff};");
            string FloatAt(int offset)
            {
                string h = b.R(), f = b.F();
                b.L($"ld.shared.u16 {h}, [{hdr}+{offset}];");
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

            string d = FloatAt(s.ScaleOffset);
            string? dmin = s.MinOffset >= 0 ? FloatAt(s.MinOffset) : null;
            for (int sub = 0; sub < subs; sub++)
            {
                string j = b.R();
                b.L($"add.u32 {j}, {jBase}, {sub};");
                string Assemble(int which, int signedBits)
                {
                    string value = b.R(), first = b.R();
                    b.L($"mov.u32 {value}, 0;");
                    b.L($"mad.lo.u32 {first}, {j}, {2 * pieces}, {which * pieces};");
                    for (int i = 0; i < pieces; i++)
                    {
                        string e = b.R(), at = b.RD(), off = b.R(), sh = b.R(), bits = b.R(), place = b.R(), by = b.R(), f = b.R();
                        b.L($"add.u32 {e}, {first}, {i};");
                        b.L($"mul.wide.u32 {at}, {e}, 4;");
                        b.L($"add.u64 {at}, {table}, {at};");
                        b.L($"ld.const.u32 {e}, [{at}];");
                        b.L($"and.b32 {off}, {e}, 255;");
                        b.L($"bfe.u32 {sh}, {e}, 8, 4;");
                        b.L($"bfe.u32 {bits}, {e}, 12, 4;");
                        b.L($"bfe.u32 {place}, {e}, 16, 5;");
                        b.L($"add.u32 {off}, {off}, {hdr};");
                        b.L($"ld.shared.u8 {by}, [{off}];");
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

                string sc = Assemble(0, s.SubScalesSigned ? s.SubScales[0].Sum(q => q.Bits) : 0), scF = b.F(), scale = b.F(), min = b.F();
                if (s.SubScaleBias != 0)
                {
                    b.L($"add.s32 {sc}, {sc}, {s.SubScaleBias};");
                }

                b.L($"cvt.rn.f32.s32 {scF}, {sc};");
                b.L($"mul.rn.f32 {scale}, {d}, {scF};");
                if (s.SubMins is not null)
                {
                    string mnI = Assemble(1, 0), mnF = b.F();
                    b.L($"cvt.rn.f32.s32 {mnF}, {mnI};");
                    b.L($"mul.rn.f32 {min}, {dmin}, {mnF};");
                }
                else
                {
                    b.L($"mov.f32 {min}, 0f00000000;");
                }

                string at2 = b.R();
                b.L($"mad.lo.u32 {at2}, {tid}, 8, {smem};");
                b.L($"add.u32 {at2}, {at2}, {stageOff};");
                b.L($"st.shared.v2.f32 [{at2}+{layout.ScaleBase + sub * bn * 8}], {{{scale}, {min}}};");
            }

            b.Label(label);
        }

        // Fragment addresses: x's as the native kernel's (each lane's row, the chunk it supplies within a k-step).
        string aAddr = b.R(), t1 = b.R(), t2 = b.R(), aChunk = b.R(), aRow = b.R(), g4 = b.R(), t4 = b.R();
        b.L($"and.b32 {t1}, {lane}, 15;");
        b.L($"mad.lo.u32 {aRow}, {wmR}, 64, {t1};");
        b.L($"mad.lo.u32 {aAddr}, {aRow}, {stride}, {smem};");
        b.L($"shr.u32 {aChunk}, {lane}, 4;");
        b.L($"shr.u32 {g4}, {lane}, 2;");
        b.L($"and.b32 {t4}, {lane}, 3;");
        if (swizzle)
        {
            string sa = b.R();
            NativeSwizzle(b, sa, aRow, rowBytes);
            b.L($"xor.b32 {aChunk}, {aChunk}, {sa};");
        }
        else
        {
            b.L($"mad.lo.u32 {aAddr}, {aChunk}, 16, {aAddr};");
        }

        string FragAddr(string stageAddr, int rowOffset, int chunk)
        {
            if (!swizzle)
            {
                return string.Create(CultureInfo.InvariantCulture, $"[{stageAddr}+{rowOffset + chunk * 16}]");
            }

            string tt = b.R(), a2 = b.R();
            b.L($"xor.b32 {tt}, {aChunk}, {chunk};");
            b.L($"mad.lo.u32 {a2}, {tt}, 16, {stageAddr};");
            return string.Create(CultureInfo.InvariantCulture, $"[{a2}+{rowOffset}]");
        }

        // W's lane addresses. A: row wn·WN + lane/4 of the warp's tiles, bytes 2·(lane%4) of the code bytes, and its scale pair.
        // B: the ldmatrix rows of a pair of tiles (lane%8 + 8·(lane/16)), chunk (lane/8)%2; its columns' scale pairs.
        string wLane = b.R(), scLane = b.R();
        if (int8)
        {
            string r = b.R(), ch = b.R();
            b.L($"and.b32 {r}, {lane}, 7;");
            b.L($"shr.u32 {t2}, {lane}, 4;");
            b.L($"mad.lo.u32 {r}, {t2}, 8, {r};");
            b.L($"mad.lo.u32 {r}, {wnR}, {wn}, {r};");
            b.L($"shr.u32 {ch}, {lane}, 3;");
            b.L($"and.b32 {ch}, {ch}, 1;");
            b.L($"mad.lo.u32 {wLane}, {r}, {wStride}, {smem};");
            b.L($"mad.lo.u32 {wLane}, {ch}, 16, {wLane};");
            b.L($"add.u32 {wLane}, {wLane}, {layout.WBase + HeaderBytes};");
            // Columns wn·WN + 2·(lane%4) (+1) of tile 0: their pairs (scale, min) side by side, 16 bytes.
            string col = b.R();
            b.L($"mad.lo.u32 {col}, {wnR}, {wn}, 0;");
            b.L($"mad.lo.u32 {col}, {t4}, 2, {col};");
            b.L($"mad.lo.u32 {scLane}, {col}, 8, {smem};");
            b.L($"add.u32 {scLane}, {scLane}, {layout.ScaleBase};");
        }
        else
        {
            string r = b.R();
            b.L($"mad.lo.u32 {r}, {wnR}, {wn}, {g4};");
            b.L($"mad.lo.u32 {wLane}, {r}, {wStride}, {smem};");
            b.L($"mad.lo.u32 {wLane}, {t4}, 2, {wLane};");
            b.L($"add.u32 {wLane}, {wLane}, {layout.WBase + HeaderBytes};");
            b.L($"mad.lo.u32 {scLane}, {r}, 8, {smem};");
            b.L($"add.u32 {scLane}, {scLane}, {layout.ScaleBase};");
        }

        // B: x's scale rows of the lane, wm·64 + i·16 + lane/4 (+8): their pairs at (row · subs + sub) · 8.
        string xsLane = "";
        if (int8)
        {
            xsLane = b.R();
            string r = b.R();
            b.L($"mad.lo.u32 {r}, {wmR}, 64, {g4};");
            b.L($"mad.lo.u32 {xsLane}, {r}, {subs * 8}, {smem};");
            b.L($"add.u32 {xsLane}, {xsLane}, {layout.XScaleBase};");
        }

        var af = new string[2][][];
        var bf = new string[2][][];
        for (int set = 0; set < 2; set++)
        {
            af[set] = new string[4][];
            bf[set] = new string[nt][];
            for (int i = 0; i < 4; i++)
            {
                af[set][i] = [b.R(), b.R(), b.R(), b.R()];
            }

            for (int j = 0; j < nt; j++)
            {
                bf[set][j] = [b.R(), b.R()];
            }
        }

        // A's scale pairs of the current sub-block, a tile each; B's raw code words of a 64-value group, 4 a pair of tiles.
        var wScale = new string[nt];
        var wMin = new string[nt];
        for (int j = 0; j < nt; j++)
        {
            (wScale[j], wMin[j]) = (b.F(), b.F());
        }

        var raw = new string[int8 ? nt / 2 : 0][];
        for (int jp = 0; jp < raw.Length; jp++)
        {
            raw[jp] = [b.R(), b.R(), b.R(), b.R()];
        }

        string shiftReg = b.R();                                           // A, 32-value stages: the nibble's shift, 0 or 4
        string aStage = b.R(), wStage = b.R(), scStage = b.R(), xsStage = int8 ? b.R() : "";
        int loadedSub = -1;

        // A: tile j's four values at k-step ks (k = 16·ks + 2t, 2t + 1, 2t + 8, 2t + 9 of the stage), decoded.
        void DecodeTile(int set, int j, int ks)
        {
            int groupByte = 16 * (ks % 2), sub = ks * 16 / 32;
            if (sub != loadedSub)
            {
                for (int jj = 0; jj < nt; jj++)
                {
                    b.L($"ld.shared.v2.f32 {{{wScale[jj]}, {wMin[jj]}}}, [{scStage}+{(sub * bn + jj * 8) * 8}];");
                }

                loadedSub = sub;
            }

            string h0 = b.R(), h1 = b.R(), w = b.R();
            b.L($"ld.shared.u16 {h0}, [{wStage}+{j * 8 * wStride + groupByte}];");
            b.L($"ld.shared.u16 {h1}, [{wStage}+{j * 8 * wStride + groupByte + 8}];");
            b.L($"prmt.b32 {w}, {h0}, {h1}, 0x5410;");
            if (stageValues == 32)
            {
                b.L($"shr.b32 {w}, {w}, {shiftReg};");
            }
            else if (ks / 2 == 1)
            {
                b.L($"shr.b32 {w}, {w}, 4;");
            }

            b.L($"and.b32 {w}, {w}, 0x0F0F0F0F;");
            var f = new string[4];
            for (int e = 0; e < 4; e++)
            {
                string u = b.R();
                f[e] = b.F();
                b.L(string.Create(CultureInfo.InvariantCulture, $"prmt.b32 {u}, {w}, 0x4B000000, 0x{0x7440 + e:X4};"));
                b.L($"mov.b32 {f[e]}, {u};");
                b.L($"sub.rn.f32 {f[e]}, {f[e]}, 0f4B000000;");               // the code, exactly (2^23 + code − 2^23)
                if (s.CodeBias != 0)
                {
                    b.L(string.Create(CultureInfo.InvariantCulture, $"add.rn.f32 {f[e]}, {f[e]}, 0f{BitConverter.SingleToInt32Bits(s.CodeBias):X8};"));
                }

                b.L($"mul.rn.f32 {f[e]}, {wScale[j]}, {f[e]};");
                if (s.SubMins is not null)
                {
                    b.L($"sub.rn.f32 {f[e]}, {f[e]}, {wMin[j]};");
                }
            }

            b.L($"cvt.rn.bf16x2.f32 {bf[set][j][0]}, {f[1]}, {f[0]};");
            b.L($"cvt.rn.bf16x2.f32 {bf[set][j][1]}, {f[3]}, {f[2]};");
        }

        void LoadFrags(int set, int ks)
        {
            for (int i = 0; i < 4; i++)
            {
                b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{string.Join(", ", af[set][i])}}}, {FragAddr(aStage, i * 16 * stride, ks * 2)};");
            }

            if (!int8)
            {
                for (int j = 0; j < nt; j++)
                {
                    DecodeTile(set, j, ks);
                }

                return;
            }

            // B: a 64-value group's code words once (its first sub-block), then each sub-block's nibbles as int8.
            int gi = ks / 2, nibble = ks % 2;
            if (nibble == 0)
            {
                for (int jp = 0; jp < nt / 2; jp++)
                {
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{string.Join(", ", raw[jp])}}}, [{wStage}+{jp * 16 * wStride + gi * 32}];");
                }
            }

            for (int j = 0; j < nt; j++)
            {
                for (int r = 0; r < 2; r++)
                {
                    string word = raw[j / 2][(j % 2) * 2 + r];
                    if (nibble == 1)
                    {
                        b.L($"shr.b32 {bf[set][j][r]}, {word}, 4;");
                        b.L($"and.b32 {bf[set][j][r]}, {bf[set][j][r]}, 0x0F0F0F0F;");
                    }
                    else
                    {
                        b.L($"and.b32 {bf[set][j][r]}, {word}, 0x0F0F0F0F;");
                    }
                }
            }

        }

        // B's sums: per sub-block an int32 sum a value of C, then C += (d·sc·xs)·isum − (dmin·m)·xsum.
        var isum = new string[nt, 4];
        string zero = "";
        if (int8)
        {
            zero = b.R();
            b.L($"mov.u32 {zero}, 0;");
            for (int j = 0; j < nt; j++)
            {
                for (int q = 0; q < 4; q++)
                {
                    isum[j, q] = b.R();
                }
            }
        }

        var xScale = new string[4, 2];
        var xSum = new string[4, 2];
        var cScale = new string[nt, 2];
        var cMin = new string[nt, 2];
        if (int8)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int h = 0; h < 2; h++)
                {
                    (xScale[i, h], xSum[i, h]) = (b.F(), b.F());
                }
            }

            for (int j = 0; j < nt; j++)
            {
                for (int e = 0; e < 2; e++)
                {
                    (cScale[j, e], cMin[j, e]) = (b.F(), b.F());
                }
            }
        }

        string factor = b.F(), isf = b.F();
        void Compute(int set, int ks)
        {
            if (!int8)
            {
                for (int i = 0; i < 4; i++)
                {
                    for (int j = 0; j < nt; j++)
                    {
                        string d = $"{{{c[i, j, 0]}, {c[i, j, 1]}, {c[i, j, 2]}, {c[i, j, 3]}}}";
                        b.L($"{NativeKind.BF16.Mma} {d}, {{{string.Join(", ", af[set][i])}}}, {{{string.Join(", ", bf[set][j])}}}, {d};");
                    }
                }

                return;
            }

            // The sub-block's scales: x's rows (pairs of scale and block sum), W's columns (pairs of scale and minimum, negated).
            for (int i = 0; i < 4; i++)
            {
                for (int h = 0; h < 2; h++)
                {
                    b.L($"ld.shared.v2.f32 {{{xScale[i, h]}, {xSum[i, h]}}}, [{xsStage}+{((i * 16 + h * 8) * subs + ks) * 8}];");
                }
            }

            for (int j = 0; j < nt; j++)
            {
                b.L($"ld.shared.v4.f32 {{{cScale[j, 0]}, {cMin[j, 0]}, {cScale[j, 1]}, {cMin[j, 1]}}}, [{scStage}+{(ks * bn + j * 8) * 8}];");
                b.L($"neg.f32 {cMin[j, 0]}, {cMin[j, 0]};");
                b.L($"neg.f32 {cMin[j, 1]}, {cMin[j, 1]};");
            }

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < nt; j++)
                {
                    b.L($"mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {{{isum[j, 0]}, {isum[j, 1]}, {isum[j, 2]}, {isum[j, 3]}}}, {{{string.Join(", ", af[set][i])}}}, {{{string.Join(", ", bf[set][j])}}}, {{{zero}, {zero}, {zero}, {zero}}};");
                }

                for (int j = 0; j < nt; j++)
                {
                    for (int q = 0; q < 4; q++)
                    {
                        int h = q / 2, e = q % 2;
                        b.L($"mul.rn.f32 {factor}, {cScale[j, e]}, {xScale[i, h]};");
                        b.L($"cvt.rn.f32.s32 {isf}, {isum[j, q]};");
                        b.L($"fma.rn.f32 {c[i, j, q]}, {factor}, {isf}, {c[i, j, q]};");
                        b.L($"fma.rn.f32 {c[i, j, q]}, {cMin[j, e]}, {xSum[i, h]}, {c[i, j, q]};");
                    }
                }
            }
        }

        // The pipeline: stages 0 .. S-2 issued ahead, block 0's scales; each block waits for its own, computes, issues
        // block + S - 1, then (its own copies of block + 1 landed) computes block + 1's scales for the next barrier to hand on.
        string kb = b.R(), stage = b.R(), issueBlock = b.R(), issueStage = b.R(), next = b.R(), nextStage = b.R(), pLoop = b.P(), pWrap = b.P(), pWrap2 = b.P();
        for (int sIdx = 0; sIdx < stages - 1; sIdx++)
        {
            string blk = b.R(), off = b.R();
            b.L($"mov.u32 {blk}, {sIdx};");
            b.L($"mov.u32 {off}, {sIdx * stageBytes};");
            Issue(blk, off);
        }

        b.L($"cp.async.wait_group {stages - 2};");
        {
            string blk = b.R(), off = b.R();
            b.L($"mov.u32 {blk}, 0;");
            b.L($"mov.u32 {off}, 0;");
            Scales(blk, off);
        }

        b.L($"mov.u32 {kb}, 0;");
        b.L($"mov.u32 {stage}, 0;");
        b.L($"mov.u32 {issueBlock}, {stages - 1};");
        b.L($"mov.u32 {issueStage}, {(stages - 1) * stageBytes};");
        b.Label("KLOOP");
        b.L($"setp.ge.u32 {pLoop}, {kb}, {blocksReg};");
        b.L($"@{pLoop} bra KDONE;");
        b.L($"cp.async.wait_group {stages - 2};");
        b.L("bar.sync 0;");
        b.L($"add.u32 {aStage}, {aAddr}, {stage};");
        b.L($"add.u32 {wStage}, {wLane}, {stage};");
        b.L($"add.u32 {scStage}, {scLane}, {stage};");
        if (int8)
        {
            b.L($"add.u32 {xsStage}, {xsLane}, {stage};");
        }
        else if (stageValues == 32)
        {
            // The stage's 32 values are a 64-value group's low halves (first) or high halves (second).
            string w0 = b.R();
            b.L($"add.u32 {w0}, {kb}, {kb0};");
            b.L($"shl.b32 {w0}, {w0}, 5;");
            b.L($"and.b32 {w0}, {w0}, 32;");
            b.L($"shr.u32 {shiftReg}, {w0}, 3;");
        }

        loadedSub = -1;
        LoadFrags(0, 0);
        for (int ks = 0; ks < ksteps; ks++)
        {
            if (ks + 1 < ksteps)
            {
                LoadFrags((ks + 1) % 2, ks + 1);
            }

            Compute(ks % 2, ks);
        }

        b.L("bar.sync 0;");
        Issue(issueBlock, issueStage);
        b.L($"cp.async.wait_group {stages - 2};");
        b.L($"add.u32 {next}, {kb}, 1;");
        b.L($"add.u32 {nextStage}, {stage}, {stageBytes};");
        b.L($"setp.ge.u32 {pWrap2}, {nextStage}, {totalShared};");
        b.L($"@{pWrap2} mov.u32 {nextStage}, 0;");
        Scales(next, nextStage);
        b.L($"add.u32 {issueBlock}, {issueBlock}, 1;");
        b.L($"add.u32 {issueStage}, {issueStage}, {stageBytes};");
        b.L($"setp.ge.u32 {pWrap}, {issueStage}, {totalShared};");
        b.L($"@{pWrap} mov.u32 {issueStage}, 0;");
        b.L($"mov.u32 {stage}, {nextStage};");
        b.L($"mov.u32 {kb}, {next};");
        b.L("bra KLOOP;");
        b.Label("KDONE");
        b.L("cp.async.wait_group 0;");

        string l4 = b.R(), l3 = b.R(), rBase = b.R(), cBase = b.R();
        b.L($"shr.u32 {l4}, {lane}, 2;");
        b.L($"and.b32 {l3}, {lane}, 3;");
        b.L($"mad.lo.u32 {rBase}, {wmR}, 64, {row0};");
        b.L($"add.u32 {rBase}, {rBase}, {l4};");
        b.L($"mad.lo.u32 {cBase}, {wnR}, {wn}, {col0};");
        b.L($"mad.lo.u32 {cBase}, {l3}, 2, {cBase};");
        StoreC(b, c, nt, rBase, cBase, m, n, pc);
        b.L("ret;");
        string parameters = int8
            ? ".param .u64 p_a, .param .u64 p_xs, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_kp, .param .u32 p_kblocks, .param .u32 p_wrow"
            : ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_kp, .param .u32 p_ld, .param .u32 p_kblocks, .param .u32 p_wrow";
        return b.Finish(NativeHeader + SuperKernels.SuperPiecesDeclaration(s), entry, parameters, totalShared);
    }

    // ------------------------------------------------------------------ B's x, rounded once per call

    internal const string QuantizeName = "quantize_x_s8_32";

    /// <summary>
    /// B's x (m × k float32, k a multiple of 32) into int8 rows of k and, per 32-value block, a pair (d, d · Σq) at
    /// [row, k / 32]: d = max |x| / 127, q = round-to-even(x · (1 / d)) within ±127 (0 where d is 0). A thread per 8
    /// values, a block's 4 threads a quad of lanes (their maximum and sum by butterflies), grid as the x pack's
    /// (<see cref="NativePackGrid"/>).
    /// </summary>
    internal static string QuantizeModule()
    {
        var b = new PtxBuilder();
        string px = b.RD(), pq = b.RD(), ps = b.RD(), rows = b.R(), k = b.R();
        b.L($"ld.param.u64 {px}, [p_x];");
        b.L($"ld.param.u64 {pq}, [p_q];");
        b.L($"ld.param.u64 {ps}, [p_s];");
        b.L($"ld.param.u32 {rows}, [p_rows];");
        b.L($"ld.param.u32 {k}, [p_k];");
        string chunk = b.R(), tid = b.R(), chunksReg = b.R(), pOut = b.P(), row = b.R(), stepY = b.R(), lane = b.R(), quadLead = b.P();
        b.L($"mov.u32 {chunk}, %ctaid.x;");
        b.L($"mov.u32 {tid}, %tid.x;");
        b.L($"mad.lo.u32 {chunk}, {chunk}, 256, {tid};");
        b.L($"shr.u32 {chunksReg}, {k}, 3;");
        b.L($"setp.ge.u32 {pOut}, {chunk}, {chunksReg};");
        b.L($"@{pOut} bra DONE;");
        b.L($"and.b32 {lane}, {tid}, 3;");
        b.L($"setp.eq.u32 {quadLead}, {lane}, 0;");
        b.L($"mov.u32 {row}, %ctaid.y;");
        b.L($"mov.u32 {stepY}, %nctaid.y;");
        b.Label("ROW");
        string pRow = b.P();
        b.L($"setp.ge.u32 {pRow}, {row}, {rows};");
        b.L($"@{pRow} bra DONE;");
        string at = b.RD(), idx = b.RD();
        b.L($"mul.wide.u32 {at}, {row}, {k};");
        b.L($"cvt.u64.u32 {idx}, {chunk};");
        b.L($"shl.b64 {idx}, {idx}, 3;");
        b.L($"add.u64 {at}, {at}, {idx};");                                 // the chunk's first value
        string xa = b.RD();
        b.L($"shl.b64 {xa}, {at}, 2;");
        b.L($"add.u64 {xa}, {xa}, {px};");
        var v = new string[8];
        for (int e = 0; e < 8; e++)
        {
            v[e] = b.F();
        }

        b.L($"ld.global.v4.f32 {{{v[0]}, {v[1]}, {v[2]}, {v[3]}}}, [{xa}];");
        b.L($"ld.global.v4.f32 {{{v[4]}, {v[5]}, {v[6]}, {v[7]}}}, [{xa}+16];");
        string amax = b.F(), tmp = b.F();
        b.L($"abs.f32 {amax}, {v[0]};");
        for (int e = 1; e < 8; e++)
        {
            b.L($"abs.f32 {tmp}, {v[e]};");
            b.L($"max.f32 {amax}, {amax}, {tmp};");
        }

        foreach (int mask in new[] { 1, 2 })
        {
            b.L($"shfl.sync.bfly.b32 {tmp}, {amax}, {mask}, 0x1f, 0xffffffff;");
            b.L($"max.f32 {amax}, {amax}, {tmp};");
        }

        string d = b.F(), inv = b.F(), pZero = b.P();
        b.L($"div.rn.f32 {d}, {amax}, 0f42FE0000;");                      // / 127
        b.L($"setp.eq.f32 {pZero}, {d}, 0f00000000;");
        b.L($"rcp.rn.f32 {inv}, {d};");
        b.L($"@{pZero} mov.f32 {inv}, 0f00000000;");
        string sum = b.R(), lo = b.R(), hi = b.R();
        b.L($"mov.u32 {sum}, 0;");
        b.L($"mov.u32 {lo}, 0;");
        b.L($"mov.u32 {hi}, 0;");
        for (int e = 0; e < 8; e++)
        {
            string q = b.R(), qf = b.F();
            b.L($"mul.rn.f32 {qf}, {v[e]}, {inv};");
            b.L($"cvt.rni.s32.f32 {q}, {qf};");
            b.L($"min.s32 {q}, {q}, 127;");
            b.L($"max.s32 {q}, {q}, -127;");
            b.L($"add.s32 {sum}, {sum}, {q};");
            string word = e < 4 ? lo : hi;
            b.L($"bfi.b32 {word}, {q}, {word}, {8 * (e % 4)}, 8;");
        }

        string qa = b.RD();
        b.L($"add.u64 {qa}, {pq}, {at};");
        b.L($"st.global.v2.b32 [{qa}], {{{lo}, {hi}}};");
        string other = b.R();
        foreach (int mask in new[] { 1, 2 })
        {
            b.L($"shfl.sync.bfly.b32 {other}, {sum}, {mask}, 0x1f, 0xffffffff;");
            b.L($"add.s32 {sum}, {sum}, {other};");
        }

        string sumF = b.F(), xsum = b.F(), sa = b.RD();
        b.L($"cvt.rn.f32.s32 {sumF}, {sum};");
        b.L($"mul.rn.f32 {xsum}, {d}, {sumF};");
        b.L($"shr.u64 {sa}, {at}, 5;");                                     // the block's pair: value / 32
        b.L($"shl.b64 {sa}, {sa}, 3;");
        b.L($"add.u64 {sa}, {sa}, {ps};");
        b.L($"@{quadLead} st.global.v2.f32 [{sa}], {{{d}, {xsum}}};");
        b.L($"add.u32 {row}, {row}, {stepY};");
        b.L("bra ROW;");
        b.Label("DONE");
        b.L("ret;");
        return Lf(b.Finish(NativeHeader, QuantizeName, ".param .u64 p_x, .param .u64 p_q, .param .u64 p_s, .param .u32 p_rows, .param .u32 p_k"));
    }
}
