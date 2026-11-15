// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Cuda;

/// <summary>A kernel's text with numbered registers (ported from Adad's PtxBuilder): each R / RD / F / P a new register, declared in Finish.</summary>
internal sealed class PtxBuilder
{
    private readonly StringBuilder _body = new();
    private int _r, _rd, _f, _p, _h;

    public string R() => string.Create(CultureInfo.InvariantCulture, $"%r{_r++}");

    public string RD() => string.Create(CultureInfo.InvariantCulture, $"%rd{_rd++}");

    public string F() => string.Create(CultureInfo.InvariantCulture, $"%f{_f++}");

    public string P() => string.Create(CultureInfo.InvariantCulture, $"%p{_p++}");

    /// <summary>A 16-bit register (a half's bits, a byte to store).</summary>
    public string H() => string.Create(CultureInfo.InvariantCulture, $"%h{_h++}");

    public void L(string line) => _body.Append("    ").Append(line).Append('\n');

    public void Label(string name) => _body.Append(name).Append(":\n");

    private readonly StringBuilder _declarations = new();

    /// <summary>Declarations of the kernel's own beyond the builder's registers (an epilogue's), emitted with them.</summary>
    public void Declare(string line) => _declarations.Append("    ").Append(line).Append('\n');

    /// <summary>The module: <paramref name="header"/>, the entry with its parameters and registers, shared memory (static up to 48 KiB, else the module's dynamic block).</summary>
    public string Finish(string header, string entry, string parameters, int sharedBytes = 0)
    {
        var sb = new StringBuilder(header);
        if (sharedBytes > 48 * 1024)
        {
            sb.Append(".extern .shared .align 16 .b8 smem[];\n");
        }

        sb.Append(".visible .entry ").Append(entry).Append('(').Append(parameters).Append(")\n{\n");
        sb.Append(CultureInfo.InvariantCulture, $"    .reg .pred %p<{Math.Max(1, _p)}>;\n    .reg .b32 %r<{Math.Max(1, _r)}>;\n    .reg .b64 %rd<{Math.Max(1, _rd)}>;\n");
        sb.Append(CultureInfo.InvariantCulture, $"    .reg .f32 %f<{Math.Max(1, _f)}>;\n");
        if (_h > 0)
        {
            sb.Append(CultureInfo.InvariantCulture, $"    .reg .b16 %h<{_h}>;\n");
        }

        sb.Append(_declarations);
        if (sharedBytes is > 0 and <= 48 * 1024)
        {
            sb.Append(CultureInfo.InvariantCulture, $"    .shared .align 16 .b8 smem[{sharedBytes}];\n");
        }

        sb.Append(_body).Append("}\n");
        return sb.ToString();
    }
}

// The experiment's native matrix-unit lane (Adad's design, ported from Adad's Ptx.GemmMma): a device's tensor kind (bfloat16:
// mma.sync m16n8k16, float32 sums) runs one kernel, C = A · Bᵀ over operands packed in the kind's rows (k padded to whole
// blocks of 128), whatever type they came from: a type reaches it by packing its values into the kind (NativePackX,
// NativePackW: each value widened from the type's description). The kernel's shape (bytes of a row a stage, stages, columns
// a block, swizzled chunks or padded rows) is chosen per device by measurement among NativeMmaConfigs, as Adad's MmaShape
// does; the device has the kind natively when its driver accepts the kernel.
internal static partial class PtxKernels
{
    /// <summary>
    /// A tensor kind, as data (Adad's Ptx.Kind table): its name, operand bits, the float bits its values hold (exponent,
    /// mantissa), the instruction and the values of k one instruction takes. A weight is multiplied in the kind its own
    /// description matches (<see cref="For"/>): float16 weights in float16, bfloat16 in bfloat16, as they are.
    /// </summary>
    internal sealed record NativeKind(string Name, int Bits, int Exponent, int Mantissa, string Mma, int KStep)
    {
        public static readonly NativeKind BF16 = new("bf16", 16, 8, 7, "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32", 16);

        public static readonly NativeKind F16 = new("f16", 16, 5, 10, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32", 16);

        /// <summary>The kinds the native kernels are made in, in order (a kind's place numbers its tuning keys).</summary>
        public static readonly NativeKind[] All = [BF16, F16];

        /// <summary>16-byte chunks of a row one instruction consumes.</summary>
        public int StepChunks => KStep * Bits / 128;

        /// <summary>The kind's place in <see cref="All"/>.</summary>
        public int Index => Array.IndexOf(All, this);

        /// <summary>The conversion of two float32 values into the kind's pair (cvt.rn.{Pair}.f32).</summary>
        public string Pair => Name + "x2";

        /// <summary>Whether <paramref name="d"/>'s values are this kind's (a float type of the same bits: read as they are).</summary>
        public bool Holds(TypedDecode d) => d.Kind == 'f' && d.Float is { } f && f.Exponent == Exponent && f.Mantissa == Mantissa;

        /// <summary>The kind <paramref name="d"/>'s weights are multiplied in: a float type's own where a kind holds it, else bfloat16 (8-bit codes exactly).</summary>
        public static NativeKind For(TypedDecode d) => All.FirstOrDefault(k => k.Holds(d)) ?? BF16;
    }

    /// <summary>A shape of the native kernel (Adad's MmaConfig): bytes of a row a stage, pipeline stages, columns a block, swizzled chunks.</summary>
    internal readonly record struct NativeMmaConfig(int RowBytes, int Stages, int Bn, bool Swizzle);

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

    /// <summary>
    /// Shared bytes a shape takes: (128 + BN) rows a stage, padded by 16 bytes unless swizzled; with <paramref name="raw"/>
    /// int8 weights, W's BN rows a stage hold the block's codes as they are (half a 16-bit row, padded by 16 bytes).
    /// </summary>
    internal static int NativeMmaShared(NativeMmaConfig c, bool raw = false) =>
        c.Stages * (128 * (c.RowBytes + (c.Swizzle ? 0 : 16)) + c.Bn * (raw ? c.RowBytes / 2 + 16 : c.RowBytes + (c.Swizzle ? 0 : 16)));

    internal const string NativeHeader = ".version 7.0\n.target sm_80\n.address_size 64\n";

    // The chunk swizzle of a row: 64-byte rows (4 chunks) by (row / 2) mod 4, 128-byte rows (8 chunks) by row mod 8.
    private static void NativeSwizzle(PtxBuilder b, string dst, string row, int rowBytes)
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
    /// or n copy zeros; their C is not stored. (Adad's Ptx.GemmMma, its packed-rows path.)
    /// </summary>
    /// <summary>
    /// The direct native kernel's operands (widened inside the product, no packing pass): x as float32, converted to the
    /// kind as it is staged, and W decoded from its description as it is staged (a super-block's, or a TypedDecode's:
    /// float16 or bfloat16, an 8-bit integer with a scale a row or none, a scaled block of 8- or 4-bit codes).
    /// </summary>
    internal sealed record NativeSource(TypedDecode? Typed, SuperBlock? Super)
    {
        public string Code => Super is { } s ? SuperCode(s) : Typed!.Code;

        /// <summary>The source of <paramref name="type"/>'s weights, or null where the direct kernel does not decode it.</summary>
        public static NativeSource? Of(ElementType type)
        {
            if (type.Super is { } s)
            {
                return SuperFits(s) ? new NativeSource(null, s) : null;
            }

            return TypedDecode.Of(type) is { } d && (d.Kind == 'f' && d.Float is { Exponent: 5, Mantissa: 10 } or { Exponent: 8, Mantissa: 7 } || d.Kind == 'i' && d.Bits == 8 || d.Kind == 'b')
                ? new NativeSource(d, null)
                : null;
        }

        /// <summary>Whether W's bfloat16 rows are copied as they are (the source is bfloat16).</summary>
        public bool CopiesRows => Typed is { Kind: 'f', Float.Exponent: 8 };

        /// <summary>Values a decoded unit spans: a block's (k must be whole units), else 8.</summary>
        public int Unit => Super?.Values ?? (Typed!.Kind == 'b' ? Typed.Block!.Values : 8);
    }

    /// <summary>
    /// The raw int8 mode (<paramref name="raw"/>): W's signed 8-bit codes copied into shared memory as they are (a byte a
    /// value, cp.async), each fragment's codes widened to the kind in registers as they are loaded (ldmatrix gives a lane
    /// four codes in a row, k = 4t .. 4t + 3, so x is packed with its k permuted to match: NativePackXPermuted), and C
    /// times each output's row scale as it is stored: no packing pass over W, and its codes exact.
    /// </summary>
    internal static string NativeMma(NativeKind kind, NativeMmaConfig cfg, out string entry, NativeSource? source = null, bool raw = false)
    {
        int rowBytes = cfg.RowBytes, stages = cfg.Stages, bn = cfg.Bn;
        bool swizzle = cfg.Swizzle, direct = source is not null;
        entry = string.Create(CultureInfo.InvariantCulture, $"native_mma_{kind.Name}_r{rowBytes}_s{stages}_n{bn}{(swizzle ? "_x" : "")}{(direct ? "_d" + source!.Code : "")}{(raw ? "_ri8" : "")}");
        const int BM = 128;
        int wn = bn / 4, nt = wn / 8;
        int blockValues = rowBytes * 8 / kind.Bits, blockShift = (int)Math.Log2(blockValues);
        // The raw mode's W rows: the block's codes (a byte a value), padded by 16 bytes (ldmatrix's rows land in distinct banks).
        int rawRowBytes = blockValues, rawChunks = rawRowBytes / 16;
        int stride = rowBytes + (swizzle ? 0 : 16), strideB = raw ? rawRowBytes + 16 : stride;
        int stageBytes = BM * stride + bn * strideB, totalShared = stages * stageBytes, bBase = BM * stride;
        int chunks = rowBytes / 16, copiesA = BM * chunks / NativeMmaThreads, copiesB = bn * (raw ? rawChunks : chunks) / NativeMmaThreads;
        int chunkShift = chunks switch { 4 => 2, 8 => 3, _ => throw new NotSupportedException("row bytes") };
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
        // The direct and raw variants: the true k (x's rows, W's values a row), W's bytes a row and its row scales (or 0).
        string kTrue = direct || raw ? Param32("p_k") : "", wRow = direct || raw ? Param32("p_wrow") : "", ps = direct || raw ? Param64("p_s") : "";
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
        // The direct variant's rows: x's of k floats, W's of p_wrow bytes (each chunk's value position kept, its offset added
        // as it is staged).
        string aStride = rowBytesK, bStride = rowBytesK;
        if (direct)
        {
            aStride = b.RD();
            bStride = b.RD();
            b.L($"mul.wide.u32 {aStride}, {kTrue}, 4;");
            b.L($"cvt.u64.u32 {bStride}, {wRow};");
        }
        else if (raw)
        {
            bStride = b.RD();
            b.L($"cvt.u64.u32 {bStride}, {wRow};");
        }

        var parts = new List<string>[2] { [], [] };
        var valid = new List<string>[2] { [], [] };
        var rows = new List<string>[2] { [], [] };
        int copySet = 0;
        (string[] G, string[] S, string[] Sz) Copies(int count, string basePtr, string origin, string limit, int sharedBase)
        {
            var g = new string[count];
            var s = new string[count];
            var sz = new string[count];
            bool rawB = raw && sharedBase != 0;                            // the raw mode's W: its codes' rows as stored
            string strideBytes = rawB ? bStride : !direct ? rowBytesK : sharedBase == 0 ? aStride : bStride;
            bool chunkOffset = !direct;                                    // direct: the chunk's place comes from its value index
            int rowChunks = rawB ? rawChunks : chunks, rowStride = rawB ? strideB : stride;
            for (int j = 0; j < count; j++)
            {
                string chunk = b.R(), row = b.R(), part = b.R(), gRow = b.R(), p = b.P(), t64 = b.RD(), partOff = b.RD(), rowSel = b.R();
                b.L($"add.u32 {chunk}, {tid}, {j * NativeMmaThreads};");
                b.L($"shr.u32 {row}, {chunk}, {(int)Math.Log2(rowChunks)};");
                b.L($"and.b32 {part}, {chunk}, {rowChunks - 1};");
                b.L($"add.u32 {gRow}, {origin}, {row};");
                b.L($"setp.lt.u32 {p}, {gRow}, {limit};");
                b.L($"selp.u32 {rowSel}, {gRow}, 0, {p};");
                g[j] = b.RD();
                s[j] = b.R();
                sz[j] = b.R();
                b.L($"cvt.u64.u32 {t64}, {rowSel};");
                b.L($"mul.lo.u64 {t64}, {t64}, {strideBytes};");
                if (chunkOffset)
                {
                    b.L($"mul.wide.u32 {partOff}, {part}, 16;");
                    b.L($"add.u64 {t64}, {t64}, {partOff};");
                }

                b.L($"add.u64 {g[j]}, {basePtr}, {t64};");
                parts[copySet].Add(part);
                valid[copySet].Add(p);
                rows[copySet].Add(rowSel);
                b.L($"mad.lo.u32 {s[j]}, {row}, {rowStride}, {smem};");
                if (swizzle && !rawB)
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

            copySet++;
            return (g, s, sz);
        }

        var ca = Copies(copiesA, pa, row0, m, 0);
        var cb = Copies(copiesB, pb, col0, n, bBase);

        // The direct variant's row scales of W's copies (1 without), loaded once: a row's scale is the same along k.
        var rowScales = new string[copiesB];
        if (direct)
        {
            string hasScales = b.P();
            b.L($"setp.ne.u64 {hasScales}, {ps}, 0;");
            for (int j = 0; j < copiesB; j++)
            {
                string at = b.RD();
                rowScales[j] = b.F();
                b.L($"mov.f32 {rowScales[j]}, 0f3F800000;");
                b.L($"mul.wide.u32 {at}, {rows[1][j]}, 4;");
                b.L($"add.u64 {at}, {ps}, {at};");
                b.L($"@{hasScales} ld.global.f32 {rowScales[j]}, [{at}];");
            }
        }
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
            if (direct)
            {
                IssueDirect(absolute, stageOff);
            }
            else if (raw)
            {
                // x's packed rows as usual; W's codes [k0, k0 + blockValues) of each row, a chunk's bytes past k zero.
                for (int j = 0; j < ca.G.Length; j++)
                {
                    string src = b.RD(), dst = b.R();
                    b.L($"add.u64 {src}, {ca.G[j]}, {kBytes};");
                    b.L($"add.u32 {dst}, {ca.S[j]}, {stageOff};");
                    b.L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {ca.Sz[j]};");
                }

                string k0 = b.R();
                b.L($"mul.lo.u32 {k0}, {absolute}, {blockValues};");
                for (int j = 0; j < cb.G.Length; j++)
                {
                    string src = b.RD(), dst = b.R(), at = b.R(), left = b.R(), size = b.R(), off = b.RD();
                    b.L($"mad.lo.u32 {at}, {parts[1][j]}, 16, {k0};");
                    b.L($"sub.s32 {left}, {kTrue}, {at};");
                    b.L($"max.s32 {left}, {left}, 0;");
                    b.L($"min.s32 {left}, {left}, 16;");
                    b.L($"selp.u32 {size}, {left}, 0, {valid[1][j]};");
                    b.L($"cvt.u64.u32 {off}, {k0};");
                    b.L($"add.u64 {src}, {cb.G[j]}, {off};");
                    b.L($"add.u32 {dst}, {cb.S[j]}, {stageOff};");
                    b.L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {size};");
                }
            }
            else
            {
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
            }

            b.Label(label);
            b.L("cp.async.commit_group;");
        }

        // The direct variant's staging of block `absolute`: each chunk's 8 values at k = absolute · blockValues + 8 · part,
        // zeros past k or past the rows: x's floats converted, W's bfloat16 copied, its other types decoded.
        void IssueDirect(string absolute, string stageOff)
        {
            string kStart = b.R();
            b.L($"mul.lo.u32 {kStart}, {absolute}, {blockValues};");
            for (int set = 0; set < 2; set++)
            {
                var (g, s) = set == 0 ? (ca.G, ca.S) : (cb.G, cb.S);
                for (int j = 0; j < g.Length; j++)
                {
                    string idx = b.R(), ok = b.P(), dst = b.R();
                    b.L($"shl.b32 {idx}, {parts[set][j]}, 3;");
                    b.L($"add.u32 {idx}, {idx}, {kStart};");
                    b.L($"setp.lt.and.u32 {ok}, {idx}, {kTrue}, {valid[set][j]};");
                    b.L($"add.u32 {dst}, {s[j]}, {stageOff};");
                    if (set == 1 && source!.CopiesRows)
                    {
                        string src = b.RD(), off = b.RD(), size = b.R();
                        b.L($"mul.wide.u32 {off}, {idx}, 2;");
                        b.L($"add.u64 {src}, {g[j]}, {off};");
                        b.L($"selp.u32 {size}, 16, 0, {ok};");
                        b.L($"cp.async.cg.shared.global [{dst}], [{src}], 16, {size};");
                        continue;
                    }

                    // Branch-free, so the copies' loads issue back to back: a chunk past k or the rows reads value 0 of its row
                    // (valid memory: rows past the end read row 0) and its values are then 0.
                    string at = b.R();
                    b.L($"selp.u32 {at}, {idx}, 0, {ok};");
                    var v = new string[8];
                    var got = set == 0 ? DirectFloats(b, g[j], at) : DirectEight(b, source!, g[j], at);
                    for (int e = 0; e < 8; e++)
                    {
                        v[e] = b.F();
                        if (set == 1)
                        {
                            b.L($"mul.rn.f32 {got[e]}, {got[e]}, {rowScales[j]};");
                        }

                        b.L($"selp.f32 {v[e]}, {got[e]}, 0f00000000, {ok};");
                    }

                    var h = new string[4];
                    for (int q = 0; q < 4; q++)
                    {
                        h[q] = b.R();
                        b.L($"cvt.rn.{kind.Pair}.f32 {h[q]}, {v[2 * q + 1]}, {v[2 * q]};");
                    }

                    b.L($"st.shared.v4.b32 [{dst}], {{{h[0]}, {h[1]}, {h[2]}, {h[3]}}};");
                }
            }
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

        if (raw)
        {
            // The raw codes' fragments: ldmatrix.x4 over four n8 tiles at once, lane L the row L of the warp's 32.
            b.L($"mad.lo.u32 {bRow}, {wnR}, {wn}, {lane};");
            b.L($"mad.lo.u32 {bAddr}, {bRow}, {strideB}, {smem};");
        }
        else
        {
            b.L($"mad.lo.u32 {bAddr}, {bRow}, {stride}, {smem};");
        }

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
            if (!raw)
            {
                b.L($"mad.lo.u32 {bAddr}, {bChunk}, 16, {bAddr};");
            }
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

                if (raw)
                {
                    // Four tiles' codes, k = 4t .. 4t + 3 a lane (t its place in a quad), each widened exactly with the
                    // full-rate pipes only (no conversion unit): code + 128 (each byte's sign bit flipped) as the low byte of
                    // float32 2^23 + 128 + code (bits 0x4B0000xx), less 2^23 + 128; a small integer's bfloat16 is its float32's
                    // high half, so (4t, 4t + 1) are the fragment's first register, (4t + 2, 4t + 3) its second.
                    for (int j = 0; j < nt; j += 4)
                    {
                        var q = new[] { b.R(), b.R(), b.R(), b.R() };
                        b.L(string.Create(CultureInfo.InvariantCulture, $"ldmatrix.sync.aligned.m8n8.x4.shared.b16 {{{string.Join(", ", q)}}}, [{bStage}+{j * 8 * strideB + ks * 16}];"));
                        for (int qi = 0; qi < 4; qi++)
                        {
                            string biased = b.R();
                            b.L($"xor.b32 {biased}, {q[qi]}, 0x80808080;");
                            var f = new string[4];
                            for (int e = 0; e < 4; e++)
                            {
                                string word = b.R();
                                f[e] = b.F();
                                b.L(string.Create(CultureInfo.InvariantCulture, $"prmt.b32 {word}, {biased}, 0x4B000000, 0x{0x7540 + e:X4};"));
                                b.L($"mov.b32 {f[e]}, {word};");
                                b.L($"sub.f32 {f[e]}, {f[e]}, 0f4B000080;");
                            }

                            if (kind == NativeKind.BF16)
                            {
                                b.L($"prmt.b32 {bf[set][j + qi][0]}, {f[0]}, {f[1]}, 0x7632;");
                                b.L($"prmt.b32 {bf[set][j + qi][1]}, {f[2]}, {f[3]}, 0x7632;");
                            }
                            else
                            {
                                b.L($"cvt.rn.{kind.Pair}.f32 {bf[set][j + qi][0]}, {f[1]}, {f[0]};");
                                b.L($"cvt.rn.{kind.Pair}.f32 {bf[set][j + qi][1]}, {f[3]}, {f[2]};");
                            }
                        }
                    }

                    return;
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

        // The raw mode's row scales: each output column's (1 without), read once for the rows the thread stores.
        var colScale = new string[nt, 2];
        if (raw)
        {
            string hasScales = b.P();
            b.L($"setp.ne.u64 {hasScales}, {ps}, 0;");
            for (int j = 0; j < nt; j++)
            {
                for (int e = 0; e < 2; e++)
                {
                    string col = b.R(), inside = b.P(), at = b.RD();
                    colScale[j, e] = b.F();
                    b.L($"mov.f32 {colScale[j, e]}, 0f3F800000;");
                    b.L($"add.u32 {col}, {cBase}, {j * 8 + e};");
                    b.L($"setp.lt.and.u32 {inside}, {col}, {n}, {hasScales};");
                    b.L($"mul.wide.u32 {at}, {col}, 4;");
                    b.L($"add.u64 {at}, {ps}, {at};");
                    b.L($"@{inside} ld.global.f32 {colScale[j, e]}, [{at}];");
                }
            }

            for (int i = 0; i < 4; i++)
            {
                for (int j = 0; j < nt; j++)
                {
                    for (int q = 0; q < 4; q++)
                    {
                        b.L($"mul.rn.f32 {c[i, j, q]}, {c[i, j, q]}, {colScale[j, q % 2]};");
                    }
                }
            }
        }

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

        b.L("ret;");
        string header = source?.Super is { } superSource ? NativeHeader + SuperPiecesDeclaration(superSource) : NativeHeader;
        return b.Finish(header, entry, ".param .u64 p_a, .param .u64 p_b, .param .u64 p_c, .param .u32 p_m, .param .u32 p_n, .param .u32 p_kp, .param .u32 p_ld, .param .u32 p_kblocks"
            + (direct || raw ? ", .param .u32 p_k, .param .u32 p_wrow, .param .u64 p_s" : ""), totalShared);
    }

    // x's 8 floats at value `idx` of the row at `rowBase` (k a multiple of 8: two 16-byte loads).
    private static string[] DirectFloats(PtxBuilder b, string rowBase, string idx)
    {
        string addr = b.RD(), off = b.RD();
        var v = new string[8];
        for (int e = 0; e < 8; e++)
        {
            v[e] = b.F();
        }

        b.L($"mul.wide.u32 {off}, {idx}, 4;");
        b.L($"add.u64 {addr}, {rowBase}, {off};");
        b.L($"ld.global.v4.f32 {{{v[0]}, {v[1]}, {v[2]}, {v[3]}}}, [{addr}];");
        b.L($"ld.global.v4.f32 {{{v[4]}, {v[5]}, {v[6]}, {v[7]}}}, [{addr}+16];");
        return v;
    }

    // W's 8 values at value `idx` of the row at `rowBase`, decoded from the source's description (before the row's scale).
    private static string[] DirectEight(PtxBuilder b, NativeSource source, string rowBase, string idx)
    {
        if (source.Super is { } super)
        {
            string blk = b.R(), w = b.R(), at = b.RD(), blockAddr = b.RD();
            b.L($"shr.u32 {blk}, {idx}, {Log2(super.Values)};");
            b.L($"and.b32 {w}, {idx}, {super.Values - 1};");
            b.L($"mul.wide.u32 {at}, {blk}, {super.Bytes};");
            b.L($"add.u64 {blockAddr}, {rowBase}, {at};");
            return SuperEight(b, super, blockAddr, w);
        }

        var d = source.Typed!;
        var v = new string[8];
        for (int e = 0; e < 8; e++)
        {
            v[e] = b.F();
        }

        if (d.Kind == 'f')
        {
            // float16: 8 halves in 16 bytes.
            string addr = b.RD(), off = b.RD();
            var words = new[] { b.R(), b.R(), b.R(), b.R() };
            b.L($"mul.wide.u32 {off}, {idx}, 2;");
            b.L($"add.u64 {addr}, {rowBase}, {off};");
            b.L($"ld.global.v4.b32 {{{words[0]}, {words[1]}, {words[2]}, {words[3]}}}, [{addr}];");
            for (int e = 0; e < 8; e++)
            {
                string t = b.R(), hh = b.H();
                b.L(e % 2 == 0 ? $"mov.b32 {t}, {words[e / 2]};" : $"shr.b32 {t}, {words[e / 2]}, 16;");
                b.L($"cvt.u16.u32 {hh}, {t};");
                b.L($"cvt.f32.f16 {v[e]}, {hh};");
            }

            return v;
        }

        if (d.Kind == 'i')
        {
            // An 8-bit integer: 8 codes in 8 bytes.
            string addr = b.RD(), off = b.RD(), w0 = b.R(), w1 = b.R();
            b.L($"cvt.u64.u32 {off}, {idx};");
            b.L($"add.u64 {addr}, {rowBase}, {off};");
            b.L($"ld.global.v2.b32 {{{w0}, {w1}}}, [{addr}];");
            for (int e = 0; e < 8; e++)
            {
                string q = b.R();
                b.L($"bfe.{(d.Signed ? "s32" : "u32")} {q}, {(e < 4 ? w0 : w1)}, {8 * (e % 4)}, 8;");
                b.L($"cvt.rn.f32.s32 {v[e]}, {q};");
            }

            return v;
        }

        // A scaled block: its scale, then 8 codes (8-bit from byte w, or 4-bit halves in ggml's order), through the table,
        // plus the bias, times the scale.
        var block = d.Block!;
        string bi = b.R(), wv = b.R(), boff = b.RD(), baddr = b.RD(), hs = b.R(), scale = b.F();
        b.L($"shr.u32 {bi}, {idx}, {Log2(block.Values)};");
        b.L($"and.b32 {wv}, {idx}, {block.Values - 1};");
        b.L($"mul.wide.u32 {boff}, {bi}, {block.Bytes};");
        b.L($"add.u64 {baddr}, {rowBase}, {boff};");
        b.L($"ld.global.u16 {hs}, [{baddr}+{block.ScaleOffset}];");
        if (block.Scale.Exponent == 8)
        {
            b.L($"shl.b32 {hs}, {hs}, 16;");
            b.L($"mov.b32 {scale}, {hs};");
        }
        else
        {
            string hh = b.H();
            b.L($"cvt.u16.u32 {hh}, {hs};");
            b.L($"cvt.f32.f16 {scale}, {hh};");
        }

        string byte0 = b.R(), shift = b.R(), caddr = b.RD(), cw = b.RD();
        if (block.CodeBits == 4)
        {
            int half = block.Values / 2;
            b.L($"and.b32 {byte0}, {wv}, {half - 1};");
            b.L($"shr.u32 {shift}, {wv}, {Log2(half)};");
            b.L($"shl.b32 {shift}, {shift}, 2;");
        }
        else
        {
            b.L($"mov.u32 {byte0}, {wv};");
            b.L($"mov.u32 {shift}, 0;");
        }

        b.L($"cvt.u64.u32 {cw}, {byte0};");
        b.L($"add.u64 {caddr}, {baddr}, {cw};");
        var units = new[] { b.R(), b.R(), b.R(), b.R() };
        for (int u = 0; u < 4; u++)
        {
            b.L($"ld.global.u16 {units[u]}, [{caddr}+{block.CodeOffset + 2 * u}];");
        }

        string kind = d.Signed && block.CodeTable is null ? "s32" : "u32";
        for (int e = 0; e < 8; e++)
        {
            string pos = b.R(), q = b.R();
            b.L($"add.u32 {pos}, {shift}, {8 * (e % 2)};");
            b.L($"bfe.{kind} {q}, {units[e / 2]}, {pos}, {block.CodeBits};");
            if (block.CodeTable is { } table)
            {
                TableLookup(b, table, q);
            }

            if (block.CodeBias != 0)
            {
                b.L($"add.s32 {q}, {q}, {block.CodeBias};");
            }

            b.L($"cvt.rn.f32.s32 {v[e]}, {q};");
            b.L($"mul.rn.f32 {v[e]}, {v[e]}, {scale};");
        }

        return v;
    }

    internal static string NativePackXName(NativeKind kind) => $"native_pack_x_{kind.Name}";

    /// <summary>The raw int8 mode's x pack: rows of kp in the kind, each 16 values' k permuted as its fragments read W's codes.</summary>
    internal static string NativePackXPermutedName(NativeKind kind) => $"native_pack_x_{kind.Name}_p16";

    /// <summary>
    /// x (m × k float32) into the kind's rows of kp (zeros past k), each 16 values [16g, 16g + 16) laid out so that slots
    /// 2t, 2t + 1, 2t + 8, 2t + 9 (a fragment's k for lane quad place t) hold x at 16g + 4t .. 16g + 4t + 3: the order in which
    /// ldmatrix hands the raw mode's lanes W's codes (NativeMma with raw). A thread per 8 slots (half a group), the
    /// grid as <see cref="NativePackGrid"/>.
    /// </summary>
    internal static string NativePackXPermuted(NativeKind kind)
    {
        var sb = new StringBuilder(NativeHeader);
        sb.AppendLine();
        sb.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{NativePackXPermutedName(kind)}}(
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
                shr.u32 %r9, %r4, 1;
                shl.b32 %r9, %r9, 4;
                and.b32 %r10, %r4, 1;
                shl.b32 %r10, %r10, 1;
                add.u32 %r9, %r9, %r10;
                add.u32 %r11, %r9, 14;
                and.b32 %r12, %r2, 1;
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
                ld.global.v2.f32 {%v0, %v1}, [%rd3];
                ld.global.v2.f32 {%v2, %v3}, [%rd3+16];
                ld.global.v2.f32 {%v4, %v5}, [%rd3+32];
                ld.global.v2.f32 {%v6, %v7}, [%rd3+48];
                bra CONVERT;
            SCALAR:
            """);
        int[] at = [0, 1, 4, 5, 8, 9, 12, 13];
        for (int v = 0; v < 8; v++)
        {
            sb.AppendLine(CultureInfo.InvariantCulture, $$"""
                    add.u32 %r14, %r9, {{at[v]}};
                    setp.lt.u32 %p1, %r14, %r2;
                    mov.f32 %v{{v}}, 0f00000000;
                    @%p1 ld.global.f32 %v{{v}}, [%rd3+{{4 * at[v]}}];
                """);
        }

        sb.AppendLine(CultureInfo.InvariantCulture, $$"""
            CONVERT:
                cvt.rn.{{kind.Pair}}.f32 %h0, %v1, %v0;
                cvt.rn.{{kind.Pair}}.f32 %h1, %v3, %v2;
                cvt.rn.{{kind.Pair}}.f32 %h2, %v5, %v4;
                cvt.rn.{{kind.Pair}}.f32 %h3, %v7, %v6;
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
        return sb.ToString();
    }

    /// <summary>The pack kernels' grid for <paramref name="rows"/> rows of kp values: a row's chunks of 8 by 256 threads along x, rows along y (looped past 65,535).</summary>
    internal static (uint X, uint Y) NativePackGrid(int rows, int kp) => ((uint)((kp / 8 + 255) / 256), (uint)Math.Clamp(rows, 1, 65535));

    internal static string NativePackWName(TypedDecode d, NativeKind kind) => string.Create(CultureInfo.InvariantCulture, $"native_pack_w_{d.Code}_{kind.Name}");

    /// <summary>
    /// The native lane's packing module for <paramref name="d"/>: x (m × k float32) into bfloat16 rows of kp (zeros past k),
    /// and W stored outputs-major (n rows of k values of the type, a scale per row or none) into bfloat16 rows of kp, each
    /// value widened from the description, times its block's or its row's scale, rounded to nearest. A thread per 8 values
    /// of a row (grid x: the row's chunks, grid y: rows, looped past the grid's), no division; x read 16 bytes a load where
    /// the chunk lies within k and the rows start 16-byte aligned, a type's words in one load (<see cref="NativePackGrid"/>).
    /// </summary>
    internal static (string[] Names, string Source) NativePackModule(TypedDecode d, NativeKind kind)
    {
        string xName = NativePackXName(kind), wName = NativePackWName(d, kind);
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

        // W: row j's values [8c, 8c + 8), zeros past k; a per-value type's in words (k a multiple of 8), a block's in its codes.
        var s = new StringBuilder();
        s.AppendLine(CultureInfo.InvariantCulture, $$"""
            .visible .entry {{wName}}(
                .param .u64 p_w, .param .u64 p_s, .param .u64 p_y, .param .u32 p_rows, .param .u32 p_k, .param .u32 p_kp
            )
            .maxntid 256, 1, 1
            {
                .reg .pred %p<24>;
                .reg .b32 %r<24>;
                .reg .b32 %u<8>;
                .reg .b32 %h<4>;
                .reg .f32 %v<8>;
                .reg .f32 %sc;
                .reg .f32 %sv;
                .reg .b64 %rd<12>;
            {{WideningRegisters}}
                ld.param.u64 %rd1, [p_w];
                ld.param.u64 %rd2, [p_s];
                ld.param.u64 %rd3, [p_y];
                setp.ne.u64 %p20, %rd2, 0;
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
                setp.lt.u32 %p1, %r9, %r2;
                mov.u32 %r8, %ctaid.y;
                mov.u32 %r20, %nctaid.y;
            ROW:
                setp.ge.u32 %p21, %r8, %r1;
                @%p21 bra DONE;
                mov.f32 %sv, 0f3F800000;
                mul.wide.u32 %rd4, %r8, 4;
                add.u64 %rd4, %rd4, %rd2;
                @%p20 ld.global.f32 %sv, [%rd4];
            """);
        for (int v = 0; v < 8; v++)
        {
            s.AppendLine(CultureInfo.InvariantCulture, $"    mov.f32 %v{v}, 0f00000000;");
        }

        s.AppendLine("    @!%p1 bra PACK;");
        if (d.PerValue)
        {
            int words = 8 / d.PerWord;
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    mul.wide.u32 %rd5, %r8, %r2;
                    cvt.u64.u32 %rd6, %r9;
                    add.u64 %rd5, %rd5, %rd6;
                """);
            if (Shift(d.Bits).Length > 0)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    shl.b64 %rd5, %rd5, {Shift(d.Bits)};");
            }

            s.AppendLine("    add.u64 %rd5, %rd5, %rd1;");
            // The words in one load (k a multiple of 8: a row's chunk starts on its words' size).
            if (words is 2 or 4)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.global.v{words}.u32 {{{string.Join(", ", Enumerable.Range(0, words).Select(w => $"%u{w}"))}}}, [%rd5];");
            }
            else
            {
                for (int w = 0; w < words; w++)
                {
                    s.AppendLine(CultureInfo.InvariantCulture, $"    ld.global.u32 %u{w}, [%rd5+{4 * w}];");
                }
            }

            for (int w = 0; w < words; w++)
            {
                WordValues(s, d, R("%u", w), [.. Enumerable.Range(w * d.PerWord, d.PerWord).Select(v => R("%v", v))]);
            }

            for (int v = 0; v < 8; v++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    @%p20 mul.rn.f32 %v{v}, %v{v}, %sv;");
            }
        }
        else
        {
            var block = d.Block!;
            int shift = (int)Math.Log2(block.Values);
            s.AppendLine(CultureInfo.InvariantCulture, $"""
                    shr.u32 %r10, %r2, {shift};
                    shr.u32 %r11, %r9, {shift};
                    and.b32 %r12, %r9, {block.Values - 1};
                    mul.wide.u32 %rd5, %r8, %r10;
                    cvt.u64.u32 %rd6, %r11;
                    add.u64 %rd5, %rd5, %rd6;
                    mul.lo.u64 %rd5, %rd5, {block.Bytes};
                    add.u64 %rd5, %rd5, %rd1;
                    ld.global.u16 %u4, [%rd5+{block.ScaleOffset}];
                """);
            if (d.Nibbles)
            {
                NibbleAt(s, d, "%r12");
            }

            s.AppendLine(d.Nibbles ? "    cvt.u64.u32 %rd7, %nb;" : "    cvt.u64.u32 %rd7, %r12;");
            s.AppendLine("    add.u64 %rd7, %rd7, %rd5;");
            FloatFrom(s, "%u4", "%sc", block.Scale);
            for (int unit = 0; unit < 4; unit++)
            {
                s.AppendLine(CultureInfo.InvariantCulture, $"    ld.global.u16 %u{unit}, [%rd7+{block.CodeOffset + 2 * unit}];");
            }

            for (int unit = 0; unit < 4; unit++)
            {
                UnitCodes(s, d, R("%u", unit), "%sc", R("%v", 2 * unit), R("%v", 2 * unit + 1));
            }
        }

        s.AppendLine("""
            PACK:
                cvt.rn.bf16x2.f32 %h0, %v1, %v0;
                cvt.rn.bf16x2.f32 %h1, %v3, %v2;
                cvt.rn.bf16x2.f32 %h2, %v5, %v4;
                cvt.rn.bf16x2.f32 %h3, %v7, %v6;
                mul.wide.u32 %rd8, %r8, %r6;
                cvt.u64.u32 %rd9, %r4;
                add.u64 %rd8, %rd8, %rd9;
                shl.b64 %rd8, %rd8, 4;
                add.u64 %rd8, %rd8, %rd3;
                st.global.v4.b32 [%rd8], {%h0, %h1, %h2, %h3};
                add.u32 %r8, %r8, %r20;
                bra ROW;
            DONE:
                ret;
            }
            """);
        sb.Append(s);
        // The kind's conversion: bfloat16 pairs, or the kind's own (float16).
        return ([xName, wName], sb.ToString().Replace("cvt.rn.bf16x2.f32", $"cvt.rn.{kind.Pair}.f32", StringComparison.Ordinal));
    }
}
