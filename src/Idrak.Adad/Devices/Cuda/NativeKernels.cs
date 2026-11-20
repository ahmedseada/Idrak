// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Adad.Devices.Cuda;

// The native matrix-unit kernel (Adad's GemmMma, as the proof built it; copied from Idrak's PtxKernels.NativeMma, its
// packed-rows mode only): a device's tensor kind (bfloat16: mma.sync m16n8k16, float32 sums) runs one kernel, C = A · Bᵀ over
// operands packed in the kind's rows (k padded to whole blocks of 128). The kernel's shape (bytes of a row a stage, stages,
// columns a block, swizzled chunks or padded rows) is chosen per device by measurement among NativeMmaConfigs; the device
// has the kind natively when its driver accepts the kernel. With it, the parts of the control lane copied as they are: the
// x pack (native_pack_x), the k-splits' sum (typed_sum_parts).
internal static class NativeKernels
{
    /// <summary>
    /// A tensor kind, as data: its name, operand bits, the float bits its values hold (exponent, mantissa), the instruction
    /// and the values of k one instruction takes.
    /// </summary>
    internal sealed record NativeKind(string Name, int Bits, int Exponent, int Mantissa, string Mma, int KStep)
    {
        public static readonly NativeKind BF16 = new("bf16", 16, 8, 7, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 16);

        public static readonly NativeKind F16 = new("f16", 16, 5, 10, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", 16);

        /// <summary>16-byte chunks of a row one instruction consumes.</summary>
        public int StepChunks => KStep * Bits / 128;

        /// <summary>The conversion of two float32 values into the kind's pair (cvt.rn.{Pair}.f32).</summary>
        public string Pair => Name + "x2";
    }

    /// <summary>A shape of the native kernel: bytes of a row a stage, pipeline stages, columns a block, swizzled chunks.</summary>
    internal readonly record struct NativeMmaConfig(int RowBytes, int Stages, int Bn, bool Swizzle)
    {
        public override string ToString() => string.Create(CultureInfo.InvariantCulture, $"r{RowBytes} s{Stages} n{Bn}{(Swizzle ? " x" : "")}");
    }

    /// <summary>The shapes a device measures (Adad's).</summary>
    internal static readonly NativeMmaConfig[] NativeMmaConfigs =
    [
        new(64, 3, 128, false), new(64, 4, 128, false), new(128, 2, 128, false), new(64, 3, 256, false),
        new(64, 3, 256, true), new(64, 4, 256, true), new(128, 2, 256, true), new(128, 3, 128, true), new(64, 6, 128, true),
    ];

    /// <summary>Threads of a native kernel's block (8 warps, 2 × 4).</summary>
    internal const int NativeMmaThreads = 256;

    /// <summary>k padded to the packed rows' length: whole blocks of 128 values.</summary>
    internal static int NativePadded(int k) => (k + 127) / 128 * 128;

    /// <summary>Shared bytes a shape takes: (128 + BN) rows a stage, padded by 16 bytes unless swizzled.</summary>
    internal static int NativeMmaShared(NativeMmaConfig c) =>
        c.Stages * (128 * (c.RowBytes + (c.Swizzle ? 0 : 16)) + c.Bn * (c.RowBytes + (c.Swizzle ? 0 : 16)));

    internal const string NativeHeader = ".version 7.0\n.target sm_80\n.address_size 64\n";

    // The chunk swizzle of a row: 64-byte rows (4 chunks) by (row / 2) mod 4, 128-byte rows (8 chunks) by row mod 8.
    internal static void NativeSwizzle(PtxBuilder b, string dst, string row, int rowBytes)
    {
        if (rowBytes == 64)
        {
            b.L($"shr.u32 {dst}, {row}, 1;");
            b.L($"and.b32 {dst}, {dst}, 3;");
        }
        else
        {
            b.L($"and.b32 {dst}, {row}, 7;");
        }
    }

    /// <summary>
    /// C (m × n, float32, rows of n) = A · Bᵀ for A m × kp and B n × kp packed rows of the kind (ld values a row): 128 × BN
    /// of C a block of 8 warps (2 × 4; each 64 × BN/4 as 4 × BN/32 m16n8 tiles); k walks in blocks of RowBytes a row through
    /// a ring of shared-memory stages filled by asynchronous copies (the blocks ahead land while this one computes);
    /// fragments from shared memory by ldmatrix, the next k-step's loaded while this step's instructions run. Rows past m
    /// or n copy zeros; their C is not stored. Grid z splits k: blocks [z · p_kblocks, ...), sums into part z of C.
    /// </summary>
    internal static string NativeMma(NativeKind kind, NativeMmaConfig cfg, out string entry)
    {
        int rowBytes = cfg.RowBytes, stages = cfg.Stages, bn = cfg.Bn;
        bool swizzle = cfg.Swizzle;
        entry = string.Create(CultureInfo.InvariantCulture, $"native_mma_{kind.Name}_r{rowBytes}_s{stages}_n{bn}{(swizzle ? "_x" : "")}");
        const int BM = 128;
        int wn = bn / 4, nt = wn / 8;
        int blockValues = rowBytes * 8 / kind.Bits, blockShift = (int)Math.Log2(blockValues);
        int stride = rowBytes + (swizzle ? 0 : 16);
        int stageBytes = BM * stride + bn * stride, totalShared = stages * stageBytes, bBase = BM * stride;
        int chunks = rowBytes / 16, copiesA = BM * chunks / NativeMmaThreads, copiesB = bn * chunks / NativeMmaThreads;
        _ = chunks switch { 4 => 2, 8 => 3, _ => throw new NotSupportedException("row bytes") };
        int stepBytes = kind.KStep * kind.Bits / 8, ksteps = rowBytes / stepBytes;
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

        string pa = Param64("p_a"), pb = Param64("p_b"), pc = Param64("p_c"), m = Param32("p_m"), n = Param32("p_n"), kp = Param32("p_kp"), ld = Param32("p_ld");
        string kblocks = Param32("p_kblocks");
        // The split (grid z): blocks of k [z · p_kblocks, ...), its sums written to part z of C (z · m · n values on).
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

        string rowBytesK = b.RD();
        b.L($"mul.wide.u32 {rowBytesK}, {ld}, {kind.Bits};");
        b.L($"shr.u64 {rowBytesK}, {rowBytesK}, 3;");

        (string[] G, string[] S, string[] Sz) Copies(int count, string basePtr, string origin, string limit, int sharedBase)
        {
            var g = new string[count];
            var s = new string[count];
            var sz = new string[count];
            for (int j = 0; j < count; j++)
            {
                string chunk = b.R(), row = b.R(), part = b.R(), gRow = b.R(), p = b.P(), t64 = b.RD(), partOff = b.RD(), rowSel = b.R();
                b.L($"add.u32 {chunk}, {tid}, {j * NativeMmaThreads};");
                b.L($"shr.u32 {row}, {chunk}, {(int)Math.Log2(chunks)};");
                b.L($"and.b32 {part}, {chunk}, {chunks - 1};");
                b.L($"add.u32 {gRow}, {origin}, {row};");
                b.L($"setp.lt.u32 {p}, {gRow}, {limit};");
                b.L($"selp.u32 {rowSel}, {gRow}, 0, {p};");
                g[j] = b.RD();
                s[j] = b.R();
                sz[j] = b.R();
                b.L($"cvt.u64.u32 {t64}, {rowSel};");
                b.L($"mul.lo.u64 {t64}, {t64}, {rowBytesK};");
                b.L($"mul.wide.u32 {partOff}, {part}, 16;");
                b.L($"add.u64 {t64}, {t64}, {partOff};");
                b.L($"add.u64 {g[j]}, {basePtr}, {t64};");
                b.L($"mad.lo.u32 {s[j]}, {row}, {stride}, {smem};");
                if (swizzle)
                {
                    string swz = b.R(), partX = b.R();
                    NativeSwizzle(b, swz, row, rowBytes);
                    b.L($"xor.b32 {partX}, {part}, {swz};");
                    b.L($"mad.lo.u32 {s[j]}, {partX}, 16, {s[j]};");
                }
                else
                {
                    b.L($"mad.lo.u32 {s[j]}, {part}, 16, {s[j]};");
                }

                if (sharedBase != 0)
                {
                    b.L($"add.u32 {s[j]}, {s[j]}, {sharedBase};");
                }

                b.L($"selp.u32 {sz[j]}, 16, 0, {p};");
            }

            return (g, s, sz);
        }

        var ca = Copies(copiesA, pa, row0, m, 0);
        var cb = Copies(copiesB, pb, col0, n, bBase);

        string blocksReg = b.R();
        b.L($"shr.u32 {blocksReg}, {kp}, {blockShift};");
        b.L($"sub.u32 {blocksReg}, {blocksReg}, {kb0};");
        b.L($"min.u32 {blocksReg}, {blocksReg}, {kblocks};");

        // Issues block `block` into stage `stageOff` (bytes) when the block exists; commits a group either way.
        int labels = 0;
        void Issue(string block, string stageOff)
        {
            string label = string.Create(CultureInfo.InvariantCulture, $"ISSUED{labels++}");
            string pHas = b.P(), kBytes = b.RD(), absolute = b.R();
            b.L($"setp.ge.u32 {pHas}, {block}, {blocksReg};");
            b.L($"@{pHas} bra {label};");
            b.L($"add.u32 {absolute}, {block}, {kb0};");
            b.L($"mul.wide.u32 {kBytes}, {absolute}, {rowBytes};");
            foreach (var (g, s, sz) in new[] { (ca.G, ca.S, ca.Sz), (cb.G, cb.S, cb.Sz) })
            {
                for (int j = 0; j < g.Length; j++)
                {
                    string src = b.RD(), dst = b.R();
                    b.L($"add.u64 {src}, {g[j]}, {kBytes};");
                    b.L($"add.u32 {dst}, {s[j]}, {stageOff};");
                    b.L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {sz[j]};");
                }
            }

            b.Label(label);
            b.L("cp.async.commit_group;");
        }

        // Fragment addresses for ldmatrix: each lane's row address, plus the chunk its lane supplies within a k-step.
        string aAddr = b.R(), bAddr = b.R(), t1 = b.R(), t2 = b.R(), aChunk = b.R(), bChunk = b.R(), aRow = b.R(), bRow = b.R();
        b.L($"and.b32 {t1}, {lane}, 15;");
        b.L($"mad.lo.u32 {aRow}, {wmR}, 64, {t1};");
        b.L($"mad.lo.u32 {aAddr}, {aRow}, {stride}, {smem};");
        b.L($"shr.u32 {aChunk}, {lane}, 4;");
        if (kind.StepChunks == 2)
        {
            b.L($"and.b32 {t1}, {lane}, 7;");
            b.L($"shr.u32 {t2}, {lane}, 4;");
            b.L($"mad.lo.u32 {t1}, {t2}, 8, {t1};");
            b.L($"mad.lo.u32 {bRow}, {wnR}, {wn}, {t1};");
            b.L($"shr.u32 {bChunk}, {lane}, 3;");
            b.L($"and.b32 {bChunk}, {bChunk}, 1;");
        }
        else
        {
            b.L($"mad.lo.u32 {bRow}, {wnR}, {wn}, {lane};");
            b.L($"mov.u32 {bChunk}, 0;");
        }

        b.L($"mad.lo.u32 {bAddr}, {bRow}, {stride}, {smem};");
        b.L($"add.u32 {bAddr}, {bAddr}, {bBase};");
        if (swizzle)
        {
            string sa = b.R(), sb2 = b.R();
            NativeSwizzle(b, sa, aRow, rowBytes);
            NativeSwizzle(b, sb2, bRow, rowBytes);
            b.L($"xor.b32 {aChunk}, {aChunk}, {sa};");
            b.L($"xor.b32 {bChunk}, {bChunk}, {sb2};");
        }
        else
        {
            b.L($"mad.lo.u32 {aAddr}, {aChunk}, 16, {aAddr};");
            b.L($"mad.lo.u32 {bAddr}, {bChunk}, 16, {bAddr};");
        }

        string FragAddr(string stageAddr, string chunkTerm, int rowOffset, int chunk)
        {
            if (!swizzle)
            {
                return string.Create(CultureInfo.InvariantCulture, $"[{stageAddr}+{rowOffset + chunk * 16}]");
            }

            string t = b.R(), a2 = b.R();
            b.L($"xor.b32 {t}, {chunkTerm}, {chunk};");
            b.L($"mad.lo.u32 {a2}, {t}, 16, {stageAddr};");
            return string.Create(CultureInfo.InvariantCulture, $"[{a2}+{rowOffset}]");
        }

        var af = new string[2][][];
        var bf = new string[2][][];
        for (int set = 0; set < 2; set++)
        {
            af[set] = new string[4][];
            bf[set] = new string[nt][];
            for (int i = 0; i < 4; i++)
            {
                af[set][i] = kind.StepChunks == 2 ? [b.R(), b.R(), b.R(), b.R()] : [b.R(), b.R()];
            }

            for (int j = 0; j < nt; j++)
            {
                bf[set][j] = kind.StepChunks == 2 ? [b.R(), b.R()] : [b.R()];
            }
        }

        void LoadFrags(int set, string aStage, string bStage, int ks)
        {
            if (kind.StepChunks == 2)
            {
                for (int i = 0; i < 4; i++)
                {
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{string.Join(", ", af[set][i])}}}, {FragAddr(aStage, aChunk, i * 16 * stride, ks * 2)};");
                }

                for (int j = 0; j < nt; j += 2)
                {
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{bf[set][j][0]}, {bf[set][j][1]}, {bf[set][j + 1][0]}, {bf[set][j + 1][1]}}}, {FragAddr(bStage, bChunk, j * 8 * stride, ks * 2)};");
                }
            }
            else
            {
                for (int i = 0; i < 4; i++)
                {
                    b.L($"ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{{string.Join(", ", af[set][i])}}}, {FragAddr(aStage, aChunk, i * 16 * stride, ks)};");
                }

                for (int j = 0; j < nt; j += 4)
                {
                    b.L($"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{bf[set][j][0]}, {bf[set][j + 1][0]}, {bf[set][j + 2][0]}, {bf[set][j + 3][0]}}}, {FragAddr(bStage, bChunk, j * 8 * stride, ks)};");
                }
            }
        }

        void Compute(int set)
        {
            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < nt; j++)
                {
                    string d = $"{{{c[i, j, 0]}, {c[i, j, 1]}, {c[i, j, 2]}, {c[i, j, 3]}}}";
                    b.L($"{kind.Mma} {d}, {{{string.Join(", ", af[set][i])}}}, {{{string.Join(", ", bf[set][j])}}}, {d};");
                }
            }
        }

        // The pipeline: stages 0 .. S-2 issued ahead; each block waits for its own, computes, then issues block + S - 1.
        string kb = b.R(), stage = b.R(), issueBlock = b.R(), issueStage = b.R(), pLoop = b.P(), pWrap = b.P(), pWrap2 = b.P();
        for (int sIdx = 0; sIdx < stages - 1; sIdx++)
        {
            string blk = b.R(), off = b.R();
            b.L($"mov.u32 {blk}, {sIdx};");
            b.L($"mov.u32 {off}, {sIdx * stageBytes};");
            Issue(blk, off);
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
        string aStage = b.R(), bStage = b.R();
        b.L($"add.u32 {aStage}, {aAddr}, {stage};");
        b.L($"add.u32 {bStage}, {bAddr}, {stage};");
        LoadFrags(0, aStage, bStage, 0);
        for (int ks = 0; ks < ksteps; ks++)
        {
            if (ks + 1 < ksteps)
            {
                LoadFrags((ks + 1) % 2, aStage, bStage, ks + 1);
            }

            Compute(ks % 2);
        }

        b.L("bar.sync 0;");
        Issue(issueBlock, issueStage);
        b.L($"add.u32 {issueBlock}, {issueBlock}, 1;");
        b.L($"add.u32 {issueStage}, {issueStage}, {stageBytes};");
        b.L($"setp.ge.u32 {pWrap}, {issueStage}, {totalShared};");
        b.L($"@{pWrap} mov.u32 {issueStage}, 0;");
        b.L($"add.u32 {stage}, {stage}, {stageBytes};");
        b.L($"setp.ge.u32 {pWrap2}, {stage}, {totalShared};");
        b.L($"@{pWrap2} mov.u32 {stage}, 0;");
        b.L($"add.u32 {kb}, {kb}, 1;");
        b.L("bra KLOOP;");
        b.Label("KDONE");
        b.L("cp.async.wait_group 0;");

        // C: rows row0 + wm·64 + i·16 + lane/4 (+8), columns col0 + wn·WN + j·8 + 2·(lane%4) (+1).
        string l4 = b.R(), l3 = b.R(), rBase = b.R(), cBase = b.R();
        b.L($"shr.u32 {l4}, {lane}, 2;");
        b.L($"and.b32 {l3}, {lane}, 3;");
        b.L($"mad.lo.u32 {rBase}, {wmR}, 64, {row0};");
        b.L($"add.u32 {rBase}, {rBase}, {l4};");
        b.L($"mad.lo.u32 {cBase}, {wnR}, {wn}, {col0};");
        b.L($"mad.lo.u32 {cBase}, {l3}, 2, {cBase};");
        StoreC(b, c, nt, rBase, cBase, m, n, pc);
        b.L("ret;");
        return b.Finish(NativeHeader, entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_kp, .param .u32 p_ld, .param .u32 p_kblocks", totalShared);
    }

    /// <summary>
    /// The native kernel's store of a warp's C tiles (4 × nt m16n8 tiles of accumulators c[i, j, 0..3]): rows rBase + i·16
    /// (+8), columns cBase + j·8 (+1), within m and n; a pair in one 8-byte store where rows are whole pairs (n even).
    /// </summary>
    internal static void StoreC(PtxBuilder b, string[,,] c, int nt, string rBase, string cBase, string m, string n, string pc)
    {
        for (int i = 0; i < 4; i++)
        {
            for (int half = 0; half < 2; half++)
            {
                string row = b.R(), pr = b.P(), rowOff = b.RD();
                b.L($"add.u32 {row}, {rBase}, {i * 16 + half * 8};");
                b.L($"setp.lt.u32 {pr}, {row}, {m};");
                b.L($"mul.wide.u32 {rowOff}, {row}, {n};");
                for (int j = 0; j < nt; j++)
                {
                    string col = b.R(), pcol = b.P(), p2 = b.P(), idx = b.RD(), addr = b.RD(), col2 = b.R(), pOnly = b.P();
                    b.L($"add.u32 {col}, {cBase}, {j * 8};");
                    b.L($"add.u32 {col2}, {col}, 1;");
                    b.L($"setp.lt.u32 {pcol}, {col}, {n};");
                    b.L($"and.pred {pcol}, {pcol}, {pr};");
                    b.L($"setp.lt.u32 {p2}, {col2}, {n};");
                    b.L($"and.pred {p2}, {p2}, {pcol};");
                    b.L($"not.pred {pOnly}, {p2};");
                    b.L($"and.pred {pOnly}, {pOnly}, {pcol};");
                    b.L($"cvt.u64.u32 {idx}, {col};");
                    b.L($"add.u64 {idx}, {idx}, {rowOff};");
                    b.L($"shl.b64 {idx}, {idx}, 2;");
                    b.L($"add.u64 {addr}, {pc}, {idx};");
                    // A pair in one 8-byte store where rows are whole pairs (n even); with n odd, two 4-byte stores.
                    string nOdd = b.R(), pEven = b.P(), pPair = b.P(), pSplit = b.P();
                    b.L($"and.b32 {nOdd}, {n}, 1;");
                    b.L($"setp.eq.u32 {pEven}, {nOdd}, 0;");
                    b.L($"and.pred {pPair}, {p2}, {pEven};");
                    b.L($"not.pred {pSplit}, {pEven};");
                    b.L($"and.pred {pSplit}, {pSplit}, {p2};");
                    b.L($"@{pPair} st.global.v2.f32 [{addr}], {{{c[i, j, half * 2]}, {c[i, j, half * 2 + 1]}}};");
                    b.L($"@{pSplit} st.global.f32 [{addr}], {c[i, j, half * 2]};");
                    b.L($"@{pSplit} st.global.f32 [{addr}+4], {c[i, j, half * 2 + 1]};");
                    b.L($"@{pOnly} st.global.f32 [{addr}], {c[i, j, half * 2]};");
                }
            }
        }
    }

    // ------------------------------------------------------------------ the x pack

    internal static string NativePackXName(NativeKind kind) => $"native_pack_x_{kind.Name}";

    /// <summary>The pack kernels' grid for <paramref name="rows"/> rows of kp values: a row's chunks of 8 by 256 threads along x, rows along y (looped past 65,535).</summary>
    internal static (uint X, uint Y) NativePackGrid(int rows, int kp) => ((uint)((kp / 8 + 255) / 256), (uint)Math.Clamp(rows, 1, 65535));

    /// <summary>
    /// The x pack (the native lane's packing module's x kernel, as it is): x (m × k float32) into the kind's rows of kp (zeros
    /// past k). A thread per 8 values of a row (grid x: the row's chunks, grid y: rows, looped past the grid's), no division;
    /// x read 16 bytes a load where the chunk lies within k and k is a multiple of 4 (<see cref="NativePackGrid"/>).
    /// </summary>
    internal static string NativePackX(NativeKind kind)
    {
        string xName = NativePackXName(kind);
        var sb = new StringBuilder(NativeHeader);
        sb.AppendLine();
        sb.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{xName}}(
                .param .u64 p_x, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_k, .param .u32 p_kp
            )
            .maxntid 256, 1, 1
            {
                .reg .pred %p<8>;
                .reg .b32 %r<16>;
                .reg .b32 %h<4>;
                .reg .f32 %v<8>;
                .reg .b64 %rd<8>;
                ld.param.u64 %rd1, [p_x];
                ld.param.u64 %rd2, [p_y];
                ld.param.u32 %r1, [p_rows];
                ld.param.u32 %r2, [p_k];
                ld.param.u32 %r3, [p_kp];
                mov.u32 %r4, %ctaid.x;
                mov.u32 %r5, %tid.x;
                mad.lo.u32 %r4, %r4, 256, %r5;
                shr.u32 %r6, %r3, 3;
                setp.ge.u32 %p0, %r4, %r6;
                @%p0 bra DONE;
                shl.b32 %r9, %r4, 3;
                add.u32 %r11, %r9, 8;
                and.b32 %r12, %r2, 3;
                setp.le.u32 %p2, %r11, %r2;
                setp.eq.u32 %p3, %r12, 0;
                and.pred %p2, %p2, %p3;
                mov.u32 %r8, %ctaid.y;
                mov.u32 %r13, %nctaid.y;
            ROW:
                setp.ge.u32 %p4, %r8, %r1;
                @%p4 bra DONE;
                mul.wide.u32 %rd3, %r8, %r2;
                cvt.u64.u32 %rd4, %r9;
                add.u64 %rd3, %rd3, %rd4;
                shl.b64 %rd3, %rd3, 2;
                add.u64 %rd3, %rd3, %rd1;
                @!%p2 bra SCALAR;
                ld.global.v4.f32 {%v0, %v1, %v2, %v3}, [%rd3];
                ld.global.v4.f32 {%v4, %v5, %v6, %v7}, [%rd3+16];
                bra CONVERT;
            SCALAR:
            """);
        for (int v = 0; v < 8; v++)
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r10, %r9, {{v}};
                    setp.lt.u32 %p1, %r10, %r2;
                    mov.f32 %v{{v}}, 0f00000000;
                    @%p1 ld.global.f32 %v{{v}}, [%rd3+{{4 * v}}];
                """);
        }

        sb.AppendLine("""
            CONVERT:
                cvt.rn.bf16x2.f32 %h0, %v1, %v0;
                cvt.rn.bf16x2.f32 %h1, %v3, %v2;
                cvt.rn.bf16x2.f32 %h2, %v5, %v4;
                cvt.rn.bf16x2.f32 %h3, %v7, %v6;
                mul.wide.u32 %rd5, %r8, %r6;
                cvt.u64.u32 %rd6, %r4;
                add.u64 %rd5, %rd5, %rd6;
                shl.b64 %rd5, %rd5, 4;
                add.u64 %rd5, %rd5, %rd2;
                st.global.v4.b32 [%rd5], {%h0, %h1, %h2, %h3};
                add.u32 %r8, %r8, %r13;
                bra ROW;
            DONE:
                ret;
            }
            """);

        // The kind's conversion: bfloat16 pairs, or the kind's own (float16).
        return sb.ToString().Replace("cvt.rn.bf16x2.f32", $"cvt.rn.{kind.Pair}.f32", StringComparison.Ordinal);
    }

    // ------------------------------------------------------------------ the k splits' sum

    internal const string TypedSumName = "typed_sum_parts";

    /// <summary>y[e] = part[0, e] + part[1, e] + ... in split order ([splits, count]); its module, as it is.</summary>
    internal static string TypedSumModule()
    {
        var sb = new StringBuilder();
        sb.AppendLine(".version 6.0");
        sb.AppendLine(".target sm_50");
        sb.AppendLine(".address_size 64");
        sb.AppendLine();
        TypedSum(sb);
        return sb.ToString();
    }

    // y[e] = part[0, e] + part[1, e] + ... in split order (a few-row product's splits, [splits, count]).
    internal static void TypedSum(StringBuilder sb) => sb.AppendLine(CultureInfo.InvariantCulture, $$"""
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

    /// <summary>A module's text with its line ends as LF (generated text carries the source file's line ends; the PTX is the same).</summary>
    internal static string Lf(string text) => text.Replace("\r\n", "\n", StringComparison.Ordinal);
}
