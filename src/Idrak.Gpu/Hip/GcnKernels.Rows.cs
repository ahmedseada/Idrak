// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers.Binary;
using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// HIP's row kernels in GCN: norms, softmax and their kin (plan 35, G3b), written for gfx9's wave of 64 as CUDA's are
// in PTX (Cuda/PtxKernels.Decoder.cs, .Decoding.cs, .Advanced.cs, .Rows.cs): the same names, arguments and operations,
// the CPU backend's results within the device-against-CPU tolerance.
//
// A row kernel takes one block of RowThreads (256, four waves; declared with .reqd_workgroup_size, which bounds the
// registers) per row and walks the rows in a grid-stride loop (row = workgroup id x, then + the grid's workgroups while
// row < n), so the backend may launch min(rows, what the device allows) blocks. A thread reads columns tid, tid + 256,
// ... (coalesced): four a step while four fit (four loads in flight, rule 80), then one at a time, so each thread's
// partial is over its columns in column order. The block's partials are combined by one tree of a fixed shape in LDS
// (256 -> 128 between two s_barriers, -> 64 between two more, then 64 -> 1 within the first wave, whose LDS accesses
// are in order), so a row's sum is the same on every device, every grid and every run (rule 86; no atomics).
// Consecutive reductions use two LDS areas in turn, so no barrier is needed after a result is read. Per-call work (the
// arguments, 1 / cols, the mask row's remainder) is done once at the start (rule 85).
//
// For G2b (--profile labels and --bench-hip rows): each kernel, its arguments (CUDA's order: pointers, then 32-bit
// scalars, the float ones as their bits; PackArguments packs them), its launch, and its floating-point operations and
// global-memory bytes per element of a row (an exp, log, rsq or rcp counted as one operation; k = the top-k's picks;
// per-row values such as inv or stats left out):
//
// | Kernel                   | Arguments                                                | Launch (blocks of 256)     | Flops/el | Bytes/el            |
// |--------------------------|----------------------------------------------------------|----------------------------|----------|---------------------|
// | rms_norm_f32             | x, y, inv; cols, eps, n                                  | rows (or fewer)            | 3        | 12                  |
// | rms_norm_affine_f32      | x, gain, y; cols, eps, offset, n                         | rows (or fewer)            | 5        | 16                  |
// | layernorm_fused_f32      | x, gamma, beta, y; cols, eps, n                          | rows (or fewer)            | 8        | 24                  |
// | layernorm_train_f32      | x, gamma, beta, y, stats; cols, eps, n                   | rows (or fewer)            | 8        | 24                  |
// | layernorm_bwd_f32        | x, gamma, dy, stats, dx; cols, n                         | rows (or fewer)            | 14       | 32                  |
// | layernorm_bwd_params_f32 | x, dy, stats, dgamma, dbeta, partials; rows, cols, chunk | ceil(cols/256) x chunks    | 5        | 8                   |
// | fold_chunks_f32          | partials, y; cols, chunks, n                             | ceil(n/256) (or fewer)     | 1        | 4                   |
// | softmax_f32              | x, y; cols, log, n                                       | rows (or fewer)            | 6        | 20 (log: 16)        |
// | scale_mask_softmax_f32   | x, mask, y; cols, maskrows, scale, hasmask, n            | rows (or fewer)            | 8        | 28 (no mask: 24)    |
// | log_softmax_topk_f32     | x, values, indices; cols, k, n                           | rows (or fewer)            | 5 + 4k   | 8 + 4k              |
//
// layernorm_bwd_params_f32 counts per element of x (rows x cols), fold_chunks_f32 per element of partials (chunks x n).
// layernorm_bwd_params_f32 sums `chunk` rows of a column (CUDA's LayerNormBackwardKernel takes chunks of 64): with one
// chunk it adds the sums into dgamma and dbeta (a pointer of 0 skipped), with several it stores them in partials
// (dgamma's at [chunk · cols + j], dbeta's after chunks · cols) for fold_chunks_f32. Its grid's y must be the chunk
// count; its x, and every other kernel's grid, may be fewer blocks than the work (grid-stride loops).
internal static partial class GcnKernels
{
    /// <summary>
    /// Threads of a row kernel's block (four waves of 64), as CUDA's <c>PtxKernels.RowThreads</c>; also the block of
    /// <c>layernorm_bwd_params_f32</c> and <c>fold_chunks_f32</c>. Fixed: the reduction tree's shape is the block's, so
    /// a fixed block gives every device the same order of summation (rule 86).
    /// </summary>
    internal const int RowThreads = 256;

    /// <summary>The row kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Rows { get; } =
    [
        RmsNorm(affine: false),
        RmsNorm(affine: true),
        LayerNorm(train: false),
        LayerNorm(train: true),
        LayerNormBackward(),
        LayerNormBackwardParams(),
        FoldChunks(),
        Softmax(),
        ScaleMaskSoftmax(),
        LogSoftmaxTopK(),
    ];

    /// <summary>
    /// Packs a kernel's arguments as its metadata lays them out (each at its offset, of its size, little-endian), from
    /// one 64-bit value per argument as CUDA's launches pass them: a pointer, a 32-bit value, or a float's bits.
    /// </summary>
    /// <exception cref="ArgumentException">The values do not match the arguments, or an argument is neither 4 nor 8 bytes.</exception>
    internal static void PackArguments(IReadOnlyList<GcnArgument> arguments, Span<byte> destination, params ReadOnlySpan<ulong> values)
    {
        ArgumentOutOfRangeException.ThrowIfNotEqual(values.Length, arguments.Count);
        for (int i = 0; i < values.Length; i++)
        {
            var argument = arguments[i];
            var at = destination.Slice(argument.Offset, argument.Size);
            switch (argument.Size)
            {
                case 8:
                    BinaryPrimitives.WriteUInt64LittleEndian(at, values[i]);
                    break;
                case 4:
                    BinaryPrimitives.WriteUInt32LittleEndian(at, (uint)values[i]);
                    break;
                default:
                    throw new ArgumentException($"argument {i} ({argument.Name}) is {argument.Size} bytes; only 4- and 8-byte arguments are packed", nameof(arguments));
            }
        }
    }

    // ---- The kernels -----------------------------------------------------------------------------------------------

    // y = x · inv, inv = 1 / sqrt(mean(x²) + eps) per row, inv stored (rms_norm_f32); or y = x · inv · (gain + offset)
    // (rms_norm_affine_f32, inference).
    private static GcnKernelSource RmsNorm(bool affine)
    {
        var w = new RowWriter(affine ? "rms_norm_affine_f32" : "rms_norm_f32", arrays: 1);
        var a = affine
            ? w.Arguments(["x", "gain", "y"], ["cols", "eps", "offset", "n"])
            : w.Arguments(["x", "y", "inv"], ["cols", "eps", "n"]);
        string x = a.Pointer("x"), y = a.Pointer("y");
        string cols = a.Scalar("cols"), eps = a.Scalar("eps"), n = a.Scalar("n");
        string rc = w.ReciprocalOf(cols);
        w.BeginRows(n);
        int offset = w.RowOffset(cols);
        string xb = w.Base(x, offset), yb = w.Base(y, offset);
        string acc = w.V();
        w.L($"v_mov_b32_e32 {acc}, 0");
        var xs = w.Vs();
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"v_fma_f32 {acc}, {xs[e.K]}, {xs[e.K]}, {acc}");
        w.Reduce(RowWriter.Add, acc);
        string r = w.V();
        w.L($"v_mul_f32_e32 {r}, {acc}, {rc}");
        w.L($"v_add_f32_e32 {r}, {eps}, {r}");
        w.L($"v_rsq_f32_e32 {r}, {r}");
        var ys = w.Vs();
        if (affine)
        {
            string gain = a.Pointer("gain"), shift = a.Scalar("offset");
            var gs = w.Vs();
            w.Columns(cols,
                e => $"""
                    global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}
                    global_load_dword {gs[e.K]}, {e.Off}, {gain}{e.Imm}
                    """,
                e => $"""
                    v_add_f32_e32 {gs[e.K]}, {shift}, {gs[e.K]}
                    v_mul_f32_e32 {ys[e.K]}, {xs[e.K]}, {r}
                    v_mul_f32_e32 {ys[e.K]}, {ys[e.K]}, {gs[e.K]}
                    global_store_dword {e.Off}, {ys[e.K]}, {yb}{e.Imm}
                    """);
        }
        else
        {
            string inv = a.Pointer("inv");
            w.StoreFirstLane((w.ElementAddress(inv, w.Row), r));
            w.Columns(cols,
                e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}",
                e => $"""
                    v_mul_f32_e32 {ys[e.K]}, {xs[e.K]}, {r}
                    global_store_dword {e.Off}, {ys[e.K]}, {yb}{e.Imm}
                    """);
        }

        w.EndRows();
        return w.Source();
    }

    // LayerNorm over a row with gamma and beta: the mean, then the variance around it (two passes), y = (x - mean) ·
    // rstd · gamma + beta (layernorm_fused_f32); layernorm_train_f32 also stores stats[row] = mean, stats[n + row] = rstd.
    private static GcnKernelSource LayerNorm(bool train)
    {
        var w = new RowWriter(train ? "layernorm_train_f32" : "layernorm_fused_f32", arrays: 1);
        var a = train
            ? w.Arguments(["x", "gamma", "beta", "y", "stats"], ["cols", "eps", "n"])
            : w.Arguments(["x", "gamma", "beta", "y"], ["cols", "eps", "n"]);
        string x = a.Pointer("x"), y = a.Pointer("y"), gamma = a.Pointer("gamma"), beta = a.Pointer("beta");
        string cols = a.Scalar("cols"), eps = a.Scalar("eps"), n = a.Scalar("n");
        string rc = w.ReciprocalOf(cols);
        w.BeginRows(n);
        int offset = w.RowOffset(cols);
        string xb = w.Base(x, offset), yb = w.Base(y, offset);
        var xs = w.Vs();
        string sum = w.V();
        w.L($"v_mov_b32_e32 {sum}, 0");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"v_add_f32_e32 {sum}, {sum}, {xs[e.K]}");
        w.Reduce(RowWriter.Add, sum);
        string mean = w.V();
        w.L($"v_mul_f32_e32 {mean}, {sum}, {rc}");
        string squares = w.V();
        var ds = w.Vs();
        w.L($"v_mov_b32_e32 {squares}, 0");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"""
            v_sub_f32_e32 {ds[e.K]}, {xs[e.K]}, {mean}
            v_fma_f32 {squares}, {ds[e.K]}, {ds[e.K]}, {squares}
            """);
        w.Reduce(RowWriter.Add, squares);
        string rstd = w.V();
        w.L($"v_mul_f32_e32 {rstd}, {squares}, {rc}");
        w.L($"v_add_f32_e32 {rstd}, {eps}, {rstd}");
        w.L($"v_rsq_f32_e32 {rstd}, {rstd}");
        if (train)
        {
            string stats = a.Pointer("stats");
            string upper = w.S();
            w.L($"s_add_u32 {upper}, {n}, {w.Row}");
            w.StoreFirstLane((w.ElementAddress(stats, w.Row), mean), (w.ElementAddress(stats, upper), rstd));
        }

        var gs = w.Vs();
        var bs = w.Vs();
        w.Columns(cols,
            e => $"""
                global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}
                global_load_dword {gs[e.K]}, {e.Off}, {gamma}{e.Imm}
                global_load_dword {bs[e.K]}, {e.Off}, {beta}{e.Imm}
                """,
            e => $"""
                v_sub_f32_e32 {ds[e.K]}, {xs[e.K]}, {mean}
                v_mul_f32_e32 {ds[e.K]}, {ds[e.K]}, {rstd}
                v_fma_f32 {ds[e.K]}, {ds[e.K]}, {gs[e.K]}, {bs[e.K]}
                global_store_dword {e.Off}, {ds[e.K]}, {yb}{e.Imm}
                """);
        w.EndRows();
        return w.Source();
    }

    // LayerNorm's input gradient per row: g = dy · gamma, x̂ = (x - mean) · rstd, s1 = Σ g / cols, s2 = Σ g · x̂ / cols
    // (one tree for both), dx += rstd · (g - s1 - x̂ · s2).
    private static GcnKernelSource LayerNormBackward()
    {
        var w = new RowWriter("layernorm_bwd_f32", arrays: 2);
        var a = w.Arguments(["x", "gamma", "dy", "stats", "dx"], ["cols", "n"]);
        string x = a.Pointer("x"), gamma = a.Pointer("gamma"), dy = a.Pointer("dy"), stats = a.Pointer("stats"), dx = a.Pointer("dx");
        string cols = a.Scalar("cols"), n = a.Scalar("n");
        string rc = w.ReciprocalOf(cols);
        w.BeginRows(n);
        int offset = w.RowOffset(cols);
        string xb = w.Base(x, offset), dyb = w.Base(dy, offset), dxb = w.Base(dx, offset);
        string upper = w.S();
        w.L($"s_add_u32 {upper}, {n}, {w.Row}");
        string meanAt = w.ElementAddress(stats, w.Row), rstdAt = w.ElementAddress(stats, upper);
        string mean = w.V(), rstd = w.V();
        w.L($"global_load_dword {mean}, {w.Zero}, {meanAt}");
        w.L($"global_load_dword {rstd}, {w.Zero}, {rstdAt}");
        string s1 = w.V(), s2 = w.V();
        w.L($"v_mov_b32_e32 {s1}, 0");
        w.L($"v_mov_b32_e32 {s2}, 0");
        var xs = w.Vs();
        var gs = w.Vs();
        var dys = w.Vs();
        var hs = w.Vs();
        string Loads(RowWriter.Element e) => $"""
            global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}
            global_load_dword {gs[e.K]}, {e.Off}, {gamma}{e.Imm}
            global_load_dword {dys[e.K]}, {e.Off}, {dyb}{e.Imm}
            """;
        string Terms(RowWriter.Element e) => $"""
            v_mul_f32_e32 {gs[e.K]}, {dys[e.K]}, {gs[e.K]}
            v_sub_f32_e32 {hs[e.K]}, {xs[e.K]}, {mean}
            v_mul_f32_e32 {hs[e.K]}, {hs[e.K]}, {rstd}
            """;
        w.Columns(cols, Loads, e => Terms(e) + $"""

            v_add_f32_e32 {s1}, {s1}, {gs[e.K]}
            v_fma_f32 {s2}, {gs[e.K]}, {hs[e.K]}, {s2}
            """);
        w.Reduce(RowWriter.Add, s1, s2);
        w.L($"v_mul_f32_e32 {s1}, {s1}, {rc}");
        w.L($"v_mul_f32_e32 {s2}, {s2}, {rc}");
        var dxs = w.Vs();
        w.Columns(cols, e => Loads(e) + $"\nglobal_load_dword {dxs[e.K]}, {e.Off}, {dxb}{e.Imm}", e => Terms(e) + $"""

            v_sub_f32_e32 {gs[e.K]}, {gs[e.K]}, {s1}
            v_mul_f32_e32 {hs[e.K]}, {hs[e.K]}, {s2}
            v_sub_f32_e32 {gs[e.K]}, {gs[e.K]}, {hs[e.K]}
            v_fma_f32 {dxs[e.K]}, {gs[e.K]}, {rstd}, {dxs[e.K]}
            global_store_dword {e.Off}, {dxs[e.K]}, {dxb}{e.Imm}
            """);
        w.EndRows();
        return w.Source();
    }

    // LayerNorm's gamma and beta gradients: thread = column j (blocks of 256 over the columns, grid-stride), grid y =
    // chunks of `chunk` rows; Σ over the chunk's rows of dy · x̂ and of dy, in row order. One chunk adds the sums to
    // dgamma and dbeta (a pointer of 0 skipped); several store them in partials for fold_chunks_f32 (rule 86: no atomics).
    private static GcnKernelSource LayerNormBackwardParams()
    {
        var w = new RowWriter("layernorm_bwd_params_f32", arrays: 0, workgroupY: true);
        var a = w.Arguments(["x", "dy", "stats", "dgamma", "dbeta", "partials"], ["rows", "cols", "chunk"]);
        string x = a.Pointer("x"), dy = a.Pointer("dy"), stats = a.Pointer("stats"), dgamma = a.Pointer("dgamma"), dbeta = a.Pointer("dbeta");
        string partials = a.Pointer("partials");
        string rows = a.Scalar("rows"), cols = a.Scalar("cols"), chunk = a.Scalar("chunk");
        string grid = w.SPair(), stride = w.S(), first = w.S(), last = w.S(), block = w.S();
        string chunks = RowWriter.Hi(grid);
        w.L($"s_load_dwordx2 {grid}, s[0:1], 0xc                ; grid_size_x, grid_size_y (the chunks: a block is 1 high)");
        w.L($"s_lshr_b32 {stride}, {RowWriter.Lo(grid)}, 8");
        w.L($"s_mul_i32 {first}, s5, {chunk}");
        w.L($"s_add_u32 {last}, {first}, {chunk}");
        w.L($"s_min_u32 {last}, {last}, {rows}");
        string rowBytes = w.Bytes64(cols);
        w.L($"s_mov_b32 {block}, s4");
        w.Label(".Lblocks");
        string column0 = w.S();
        w.L($"s_lshl_b32 {column0}, {block}, 8");
        w.L($"s_cmp_ge_u32 {column0}, {cols}");
        w.L($"s_cbranch_scc1 .Ldone");
        string column = w.V(), off = w.V(), saved = w.SPair();
        w.L($"v_add_u32_e32 {column}, {column0}, {w.Tid}");
        w.L($"v_lshlrev_b32_e32 {off}, 2, {column}");
        w.L($"v_cmp_gt_u32_e32 vcc, {cols}, {column}");
        w.L($"s_and_saveexec_b64 {saved}, vcc");
        string sg = w.V(), sb = w.V();
        w.L($"v_mov_b32_e32 {sg}, 0");
        w.L($"v_mov_b32_e32 {sb}, 0");
        int start = w.Product64(first, cols);
        string xb = w.Base(x, start), dyb = w.Base(dy, start);
        string upper = w.S(), row = w.S();
        w.L($"s_add_u32 {upper}, {rows}, {first}");
        string meanAt = w.ElementAddress(stats, first), rstdAt = w.ElementAddress(stats, upper);
        w.L($"s_mov_b32 {row}, {first}");
        w.Label(".Lrows");
        w.L($"s_cmp_ge_u32 {row}, {last}");
        w.L($"s_cbranch_scc1 .Lrows_end");
        string xv = w.V(), dv = w.V(), mean = w.S(), rstd = w.S();
        w.L($"global_load_dword {xv}, {off}, {xb}");
        w.L($"global_load_dword {dv}, {off}, {dyb}");
        w.L($"s_load_dword {mean}, {meanAt}, 0x0");
        w.L($"s_load_dword {rstd}, {rstdAt}, 0x0");
        w.L($"v_subrev_f32_e32 {xv}, {mean}, {xv}");
        w.L($"v_mul_f32_e32 {xv}, {rstd}, {xv}");
        w.L($"v_fma_f32 {sg}, {dv}, {xv}, {sg}");
        w.L($"v_add_f32_e32 {sb}, {sb}, {dv}");
        w.Add64(xb, rowBytes);
        w.Add64(dyb, rowBytes);
        w.Add64(meanAt, "4");
        w.Add64(rstdAt, "4");
        w.L($"s_add_u32 {row}, {row}, 1");
        w.L($"s_branch .Lrows");
        w.Label(".Lrows_end");
        w.L($"s_cmp_eq_u32 {chunks}, 1");
        w.L($"s_cbranch_scc0 .Lpartial");
        string t = w.V();
        w.L($"s_cmp_eq_u64 {dgamma}, 0");
        w.L($"s_cbranch_scc1 .Lno_dgamma");
        w.L($"global_load_dword {t}, {off}, {dgamma}");
        w.L($"v_add_f32_e32 {t}, {t}, {sg}");
        w.L($"global_store_dword {off}, {t}, {dgamma}");
        w.Label(".Lno_dgamma");
        w.L($"s_cmp_eq_u64 {dbeta}, 0");
        w.L($"s_cbranch_scc1 .Lnext");
        w.L($"global_load_dword {t}, {off}, {dbeta}");
        w.L($"v_add_f32_e32 {t}, {t}, {sb}");
        w.L($"global_store_dword {off}, {t}, {dbeta}");
        w.L($"s_branch .Lnext");
        w.Label(".Lpartial");
        string pg = w.Base(partials, w.Product64("s5", cols));
        w.L($"global_store_dword {off}, {sg}, {pg}");
        string pb = w.Base(pg, w.Product64(chunks, cols));
        w.L($"global_store_dword {off}, {sb}, {pb}");
        w.Label(".Lnext");
        w.L($"s_or_b64 exec, exec, {saved}");
        w.L($"s_add_u32 {block}, {block}, {stride}");
        w.L($"s_branch .Lblocks");
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // y[i] += Σ over c < chunks of partials[c · cols + i], in chunk order, for i < n (blocks of 256, grid-stride).
    private static GcnKernelSource FoldChunks()
    {
        var w = new RowWriter("fold_chunks_f32", arrays: 0);
        var a = w.Arguments(["partials", "y"], ["cols", "chunks", "n"]);
        string partials = a.Pointer("partials"), y = a.Pointer("y");
        string cols = a.Scalar("cols"), chunks = a.Scalar("chunks"), n = a.Scalar("n");
        string grid = w.S(), stride = w.S(), block = w.S();
        w.L($"s_load_dword {grid}, s[0:1], 0xc                  ; grid_size_x");
        string chunkBytes = w.Bytes64(cols);
        w.L($"s_lshr_b32 {stride}, {grid}, 8");
        w.L($"s_mov_b32 {block}, s4");
        w.Label(".Lblocks");
        string first = w.S();
        w.L($"s_lshl_b32 {first}, {block}, 8");
        w.L($"s_cmp_ge_u32 {first}, {n}");
        w.L($"s_cbranch_scc1 .Ldone");
        string index = w.V(), off = w.V(), saved = w.SPair();
        w.L($"v_add_u32_e32 {index}, {first}, {w.Tid}");
        w.L($"v_lshlrev_b32_e32 {off}, 2, {index}");
        w.L($"v_cmp_gt_u32_e32 vcc, {n}, {index}");
        w.L($"s_and_saveexec_b64 {saved}, vcc");
        string sum = w.V(), value = w.V(), at = w.SPair(), c = w.S();
        w.L($"v_mov_b32_e32 {sum}, 0");
        w.L($"s_mov_b64 {at}, {partials}");
        w.L($"s_mov_b32 {c}, 0");
        w.Label(".Lchunks");
        w.L($"s_cmp_ge_u32 {c}, {chunks}");
        w.L($"s_cbranch_scc1 .Lchunks_end");
        w.L($"global_load_dword {value}, {off}, {at}");
        w.L($"v_add_f32_e32 {sum}, {sum}, {value}");
        w.Add64(at, chunkBytes);
        w.L($"s_add_u32 {c}, {c}, 1");
        w.L($"s_branch .Lchunks");
        w.Label(".Lchunks_end");
        w.L($"global_load_dword {value}, {off}, {y}");
        w.L($"v_add_f32_e32 {value}, {value}, {sum}");
        w.L($"global_store_dword {off}, {value}, {y}");
        w.L($"s_or_b64 exec, exec, {saved}");
        w.L($"s_add_u32 {block}, {block}, {stride}");
        w.L($"s_branch .Lblocks");
        w.Label(".Ldone");
        w.L($"s_endpgm");
        return w.Source();
    }

    // Softmax or log-softmax per row: the max, Σ exp(x - max) (the exponentials stored in y for softmax; not for
    // log-softmax, which reads x again and may be writing over it), then y = e / Σ or y = x - (max + log Σ). Each
    // thread revisits only the columns it wrote.
    private static GcnKernelSource Softmax()
    {
        var w = new RowWriter("softmax_f32", arrays: 1);
        var a = w.Arguments(["x", "y"], ["cols", "log", "n"]);
        string x = a.Pointer("x"), y = a.Pointer("y");
        string cols = a.Scalar("cols"), log = a.Scalar("log"), n = a.Scalar("n");
        string ln2 = w.S();
        w.L($"s_mov_b32 {ln2}, {RowWriter.Ln2}");
        w.BeginRows(n);
        int offset = w.RowOffset(cols);
        string xb = w.Base(x, offset), yb = w.Base(y, offset);
        var xs = w.Vs();
        string max = w.RowMax(cols, xb, xs);
        var es = w.Vs();
        string sum = w.V();
        w.L($"v_mov_b32_e32 {sum}, 0");
        w.L($"s_cmp_lg_u32 {log}, 0");
        w.L($"s_cbranch_scc1 .Llog");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => RowWriter.Exp(es[e.K], xs[e.K], max) + $"""

            global_store_dword {e.Off}, {es[e.K]}, {yb}{e.Imm}
            v_add_f32_e32 {sum}, {sum}, {es[e.K]}
            """);
        w.Reduce(RowWriter.Add, sum);
        string r = w.V();
        w.L($"v_rcp_f32_e32 {r}, {sum}");
        w.Columns(cols, e => $"global_load_dword {es[e.K]}, {e.Off}, {yb}{e.Imm}", e => $"""
            v_mul_f32_e32 {es[e.K]}, {es[e.K]}, {r}
            global_store_dword {e.Off}, {es[e.K]}, {yb}{e.Imm}
            """);
        w.L($"s_branch .Lrow_end");
        w.Label(".Llog");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => RowWriter.Exp(es[e.K], xs[e.K], max) + $"""

            v_add_f32_e32 {sum}, {sum}, {es[e.K]}
            """);
        w.Reduce(RowWriter.Add, sum);
        string shift = w.V();
        w.L($"v_log_f32_e32 {shift}, {sum}");
        w.L($"v_fma_f32 {shift}, {shift}, {ln2}, {max}");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"""
            v_sub_f32_e32 {es[e.K]}, {xs[e.K]}, {shift}
            global_store_dword {e.Off}, {es[e.K]}, {yb}{e.Imm}
            """);
        w.Label(".Lrow_end");
        w.EndRows();
        return w.Source();
    }

    // softmax(scale · x + mask[row % maskrows]) per row (no mask when hasmask is 0). The mask row's remainder is found
    // once at the start (the workgroup id and the grid's stride modulo maskrows, LLVM's sequence) and stepped with the rows.
    private static GcnKernelSource ScaleMaskSoftmax()
    {
        var w = new RowWriter("scale_mask_softmax_f32", arrays: 1);
        var a = w.Arguments(["x", "mask", "y"], ["cols", "maskrows", "scale", "hasmask", "n"]);
        string x = a.Pointer("x"), mask = a.Pointer("mask"), y = a.Pointer("y");
        string cols = a.Scalar("cols"), maskRows = a.Scalar("maskrows"), scale = a.Scalar("scale"), hasMask = a.Scalar("hasmask"), n = a.Scalar("n");
        w.BeginRows(n, loop: false);
        string maskRow = w.Remainder(w.Row, maskRows), maskStep = w.Remainder(w.Stride, maskRows);
        w.L($"s_branch .Lrow_body");
        w.Label(".Lrow_next");
        string wrapped = w.S();
        w.L($"s_add_u32 {maskRow}, {maskRow}, {maskStep}");
        w.L($"s_sub_u32 {wrapped}, {maskRow}, {maskRows}");
        w.L($"s_cmp_ge_u32 {maskRow}, {maskRows}");
        w.L($"s_cselect_b32 {maskRow}, {wrapped}, {maskRow}");
        w.Label(".Lrow_body");
        int offset = w.RowOffset(cols);
        string xb = w.Base(x, offset), yb = w.Base(y, offset);
        string mb = w.Base(mask, w.Product64(maskRow, cols));
        var xs = w.Vs();
        var ms = w.Vs();
        string max = w.V();
        w.L($"v_mov_b32_e32 {max}, {RowWriter.NegativeInfinity}");
        w.L($"s_cmp_lg_u32 {hasMask}, 0");
        w.L($"s_cbranch_scc0 .Lno_mask");
        w.Columns(cols,
            e => $"""
                global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}
                global_load_dword {ms[e.K]}, {e.Off}, {mb}{e.Imm}
                """,
            e => $"""
                v_mul_f32_e32 {xs[e.K]}, {scale}, {xs[e.K]}
                v_add_f32_e32 {xs[e.K]}, {xs[e.K]}, {ms[e.K]}
                global_store_dword {e.Off}, {xs[e.K]}, {yb}{e.Imm}
                v_max_f32_e32 {max}, {max}, {xs[e.K]}
                """);
        w.L($"s_branch .Lscaled");
        w.Label(".Lno_mask");
        w.Columns(cols,
            e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}",
            e => $"""
                v_mul_f32_e32 {xs[e.K]}, {scale}, {xs[e.K]}
                global_store_dword {e.Off}, {xs[e.K]}, {yb}{e.Imm}
                v_max_f32_e32 {max}, {max}, {xs[e.K]}
                """);
        w.Label(".Lscaled");
        w.Reduce(RowWriter.Max, max);
        string sum = w.V();
        var es = w.Vs();
        w.L($"v_mov_b32_e32 {sum}, 0");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {yb}{e.Imm}", e => RowWriter.Exp(es[e.K], xs[e.K], max) + $"""

            global_store_dword {e.Off}, {es[e.K]}, {yb}{e.Imm}
            v_add_f32_e32 {sum}, {sum}, {es[e.K]}
            """);
        w.Reduce(RowWriter.Add, sum);
        string r = w.V();
        w.L($"v_rcp_f32_e32 {r}, {sum}");
        w.Columns(cols, e => $"global_load_dword {es[e.K]}, {e.Off}, {yb}{e.Imm}", e => $"""
            v_mul_f32_e32 {es[e.K]}, {es[e.K]}, {r}
            global_store_dword {e.Off}, {es[e.K]}, {yb}{e.Imm}
            """);
        w.EndRows(".Lrow_next");
        return w.Source();
    }

    // The best k log-probabilities of a row and their columns (Backend.LogSoftmaxTopK): the max (NaN ignored), Σ exp(x -
    // max) (a NaN makes it NaN, so the row has no candidate), shift = max + log Σ; then k passes, each the block's
    // arg-max of v = x - shift over the entries after the previous pick in the order (larger v, then the smaller column),
    // -∞ and NaN never picked; the first thread writes each pick (-∞ and -1 once none is left).
    private static GcnKernelSource LogSoftmaxTopK()
    {
        var w = new RowWriter("log_softmax_topk_f32", arrays: 2);
        var a = w.Arguments(["x", "values", "indices"], ["cols", "k", "n"]);
        string x = a.Pointer("x"), values = a.Pointer("values"), indices = a.Pointer("indices");
        string cols = a.Scalar("cols"), k = a.Scalar("k"), n = a.Scalar("n");
        string ln2 = w.S();
        w.L($"s_mov_b32 {ln2}, {RowWriter.Ln2}");
        w.BeginRows(n);
        int offset = w.RowOffset(cols);
        string xb = w.Base(x, offset);
        int picks = w.Product64(w.Row, k);
        string vb = w.Base(values, picks), ib = w.Base(indices, picks);
        var xs = w.Vs();
        string max = w.RowMax(cols, xb, xs);
        var es = w.Vs();
        string sum = w.V();
        w.L($"v_mov_b32_e32 {sum}, 0");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => RowWriter.Exp(es[e.K], xs[e.K], max) + $"""

            v_add_f32_e32 {sum}, {sum}, {es[e.K]}
            """);
        w.Reduce(RowWriter.Add, sum);
        string shift = w.V(), previous = w.V(), previousColumn = w.V(), at = w.V(), pass = w.S();
        w.L($"v_log_f32_e32 {shift}, {sum}");
        w.L($"v_fma_f32 {shift}, {shift}, {ln2}, {max}");
        w.L($"v_mov_b32_e32 {previous}, {RowWriter.PositiveInfinity}");
        w.L($"v_mov_b32_e32 {previousColumn}, 0");
        w.L($"v_mov_b32_e32 {at}, 0");
        w.L($"s_mov_b32 {pass}, 0");
        w.Label(".Lpass");
        w.L($"s_cmp_ge_u32 {pass}, {k}");
        w.L($"s_cbranch_scc1 .Lpass_end");
        string best = w.V(), bestColumn = w.V();
        string below = w.SPair(), tie = w.SPair(), later = w.SPair(), better = w.SPair();
        w.L($"v_mov_b32_e32 {best}, {RowWriter.NegativeInfinity}");
        w.L($"v_mov_b32_e32 {bestColumn}, -1");
        var vs = w.Vs();
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => $"""
            v_sub_f32_e32 {vs[e.K]}, {xs[e.K]}, {shift}
            v_cmp_lt_f32_e64 {below}, {vs[e.K]}, {previous}
            v_cmp_eq_f32_e64 {tie}, {vs[e.K]}, {previous}
            v_cmp_gt_u32_e64 {later}, {e.Column}, {previousColumn}
            s_and_b64 {tie}, {tie}, {later}
            s_or_b64 {below}, {below}, {tie}
            v_cmp_gt_f32_e64 {better}, {vs[e.K]}, {best}
            s_and_b64 vcc, {below}, {better}
            v_cndmask_b32_e32 {best}, {best}, {vs[e.K]}, vcc
            v_cndmask_b32_e32 {bestColumn}, {bestColumn}, {e.Column}, vcc
            """, columns: true);
        w.Reduce(RowWriter.ArgMax, best, bestColumn);
        string index = w.V();
        w.L($"v_cvt_f32_i32_e32 {index}, {bestColumn}");
        w.StoreFirstLane((vb, best, at), (ib, index, at));
        w.L($"v_mov_b32_e32 {previous}, {best}");
        w.L($"v_mov_b32_e32 {previousColumn}, {bestColumn}");
        w.L($"v_add_u32_e32 {at}, 4, {at}");
        w.L($"s_add_u32 {pass}, {pass}, 1");
        w.L($"s_branch .Lpass");
        w.Label(".Lpass_end");
        w.EndRows();
        return w.Source();
    }

    // ---- The writer ------------------------------------------------------------------------------------------------

    /// <summary>
    /// Writes a row kernel's GCN text with virtual registers: its directives (the block of 256, two LDS areas, the
    /// dispatch packet for the grid's size), its arguments loaded once, the grid-stride row loop, coalesced column loops
    /// and the block's fixed reduction tree. Each kernel's text is written once a process, with the class.
    /// </summary>
    private sealed class RowWriter
    {
        internal const string NegativeInfinity = "0xff800000";
        internal const string PositiveInfinity = "0x7f800000";
        internal const string Log2E = "0x3fb8aa3b";
        internal const string Ln2 = "0x3f317218";

        // Columns a thread takes a step in the column loops' main part (independent loads in flight); the offsets of
        // the step's later columns (1,024 bytes apart) fit a global instruction's 13-bit offset.
        private const int Unroll = 4;

        private readonly StringBuilder _text = new();
        private readonly string _name;
        private readonly int _areaBytes;
        private readonly bool _workgroupY;
        private int _s = 100, _v = 100, _labels;
        private string _rows = "";

        public RowWriter(string name, int arrays, bool workgroupY = false)
        {
            _name = name;
            _areaBytes = 1024 * arrays;
            _workgroupY = workgroupY;
            Raw($".kernel {name}");
        }

        /// <summary>The thread's index in the block.</summary>
        public string Tid => "%v1";

        /// <summary>A zero in every lane (an address offset).</summary>
        public string Zero => "%v3";

        // The thread's index · 4; the current LDS area's address of this thread (base + index · 4), and its base.
        private static string Tid4 => "%v2";

        private static string Area => "%v4";

        private static string AreaBase => "%v5";

        /// <summary>The current row, and the grid's workgroups (the row loop's stride).</summary>
        public string Row => "%s90";

        public string Stride => "%s91";

        internal static string Lo(string pair) => F($"%s{Id(pair)}");

        internal static string Hi(string pair) => F($"%s{Id(pair) + 1}");

        private static int Id(string pair) => int.Parse(pair.AsSpan(3, pair.IndexOf(':', StringComparison.Ordinal) - 3), CultureInfo.InvariantCulture);

        /// <summary>A fresh scalar register.</summary>
        public string S() => F($"%s{_s++}");

        /// <summary>A fresh scalar pair.</summary>
        public string SPair()
        {
            int id = _s;
            _s += 2;
            return F($"%s[{id}:{id + 1}]");
        }

        /// <summary>A fresh vector register.</summary>
        public string V() => F($"%v{_v++}");

        /// <summary>Fresh vector registers: one per column of a main step, and one for the tail's.</summary>
        public string[] Vs()
        {
            var registers = new string[Unroll + 1];
            for (int k = 0; k < registers.Length; k++)
            {
                registers[k] = V();
            }

            return registers;
        }

        /// <summary>Instruction lines (several split at their line breaks), each indented.</summary>
        public void L(FormattableString lines) => Lines(lines.ToString(CultureInfo.InvariantCulture));

        private void Lines(string lines)
        {
            foreach (var part in lines.Split('\n'))
            {
                string line = part.Trim();
                if (line.Length > 0)
                {
                    _text.Append("  ").Append(line).Append('\n');
                }
            }
        }

        public void Label(string label) => _text.Append(label).Append(":\n");

        private void Raw(FormattableString line) => _text.Append(line.ToString(CultureInfo.InvariantCulture)).Append('\n');

        private static string F(FormattableString text) => text.ToString(CultureInfo.InvariantCulture);

        private string NewLabel() => F($".L{_labels++}");

        public GcnKernelSource Source() => new(_name, Encoding.UTF8.GetBytes(_text.ToString()));

        /// <summary>
        /// Declares the arguments (8-byte pointers, then 4-byte scalars), the block, LDS and the system registers, and
        /// loads every argument once into one scalar range: s[0:1] holds the dispatch packet's address, s[2:3] the
        /// arguments', s4 the workgroup id x (and s5 y).
        /// </summary>
        public Args Arguments(string[] pointers, string[] scalars)
        {
            foreach (var p in pointers)
            {
                Raw($".arg {p}, 8, global");
            }

            foreach (var s in scalars)
            {
                Raw($".arg {s}, 4, value");
            }

            Raw($".reqd_workgroup_size {RowThreads}, 1, 1");
            if (_areaBytes > 0)
            {
                Raw($".lds {2 * _areaBytes}");
            }

            Raw($".user_sgpr dispatch_ptr");
            Raw($".user_sgpr kernarg_segment_ptr");
            Raw($".system_sgpr {(_workgroupY ? "workgroup_id_y" : "workgroup_id_x")}");
            int words = 2 * pointers.Length + scalars.Length;
            int first = _s;
            _s += words;
            for (int w = 0; w < words;)
            {
                int take = words - w >= 8 ? 8 : words - w >= 4 ? 4 : words - w >= 2 ? 2 : 1;
                string load = take == 1 ? F($"s_load_dword %s{first + w}") : F($"s_load_dwordx{take} %s[{first + w}:{first + w + take - 1}]");
                L($"{load}, s[2:3], 0x{4 * w:x}");
                w += take;
            }

            L($"v_mov_b32_e32 {Tid}, v0");
            L($"v_lshlrev_b32_e32 {Tid4}, 2, v0");
            L($"v_mov_b32_e32 {Zero}, 0");
            if (_areaBytes > 0)
            {
                L($"v_lshlrev_b32_e32 {Area}, 2, v0");
                L($"v_mov_b32_e32 {AreaBase}, 0");
            }

            return new Args(pointers, scalars, first);
        }

        /// <summary>The arguments' registers: a pointer's pair, a scalar's register.</summary>
        public sealed class Args(string[] pointers, string[] scalars, int first)
        {
            public string Pointer(string name)
            {
                int p = Array.IndexOf(pointers, name);
                return F($"%s[{first + 2 * p}:{first + 2 * p + 1}]");
            }

            public string Scalar(string name) => F($"%s{first + 2 * pointers.Length + Array.IndexOf(scalars, name)}");
        }

        /// <summary>1 / count as a float in a vector register.</summary>
        public string ReciprocalOf(string count)
        {
            string r = V();
            L($"v_cvt_f32_u32_e32 {r}, {count}");
            L($"v_rcp_f32_e32 {r}, {r}");
            return r;
        }

        /// <summary>
        /// The grid-stride row loop's head: row = workgroup id x, stride = grid_size_x / 256 (from the dispatch packet),
        /// the end when row ≥ n; then <c>.Lrow</c>, where each row starts (unless <paramref name="loop"/> is false: the
        /// caller places its own).
        /// </summary>
        public void BeginRows(string n, bool loop = true)
        {
            _rows = n;
            string grid = S();
            L($"s_load_dword {grid}, s[0:1], 0xc                ; grid_size_x");
            L($"s_lshr_b32 {Stride}, {grid}, 8");
            L($"s_mov_b32 {Row}, s4");
            L($"s_cmp_ge_u32 {Row}, {n}");
            L($"s_cbranch_scc1 .Lrows_done");
            if (loop)
            {
                Label(".Lrow");
            }
        }

        /// <summary>The row loop's tail: row += stride, back to <paramref name="next"/> while row &lt; n; the kernel's end.</summary>
        public void EndRows(string next = ".Lrow")
        {
            L($"s_add_u32 {Row}, {Row}, {Stride}");
            L($"s_cmp_ge_u32 {Row}, {_rows}");
            L($"s_cbranch_scc0 {next}");
            Label(".Lrows_done");
            L($"s_endpgm");
        }

        /// <summary>a · b · 4 (bytes) as a 64-bit value in a fresh scalar pair; returns the pair's first register's id.</summary>
        public int Product64(string a, string b)
        {
            int id = _s;
            _s += 2;
            L($"s_mul_i32 %s{id}, {a}, {b}");
            L($"s_mul_hi_u32 %s{id + 1}, {a}, {b}");
            L($"s_lshl_b64 %s[{id}:{id + 1}], %s[{id}:{id + 1}], 2");
            return id;
        }

        /// <summary>The current row's first element as a byte offset: row · cols · 4.</summary>
        public int RowOffset(string cols) => Product64(Row, cols);

        /// <summary>value · 4 as a 64-bit byte count in a fresh pair.</summary>
        public string Bytes64(string value)
        {
            string pair = SPair();
            L($"s_lshl_b32 {Lo(pair)}, {value}, 2");
            L($"s_lshr_b32 {Hi(pair)}, {value}, 30");
            return pair;
        }

        /// <summary>A pointer plus a 64-bit offset (a pair <see cref="Product64"/> made), in a fresh pair.</summary>
        public string Base(string pointer, int offset)
        {
            string pair = SPair();
            L($"s_add_u32 {Lo(pair)}, {Lo(pointer)}, %s{offset}");
            L($"s_addc_u32 {Hi(pair)}, {Hi(pointer)}, %s{offset + 1}");
            return pair;
        }

        /// <summary>The address of element <paramref name="index"/> of a float array, in a fresh pair.</summary>
        public string ElementAddress(string pointer, string index)
        {
            string bytes = Bytes64(index);
            L($"s_add_u32 {Lo(bytes)}, {Lo(pointer)}, {Lo(bytes)}");
            L($"s_addc_u32 {Hi(bytes)}, {Hi(pointer)}, {Hi(bytes)}");
            return bytes;
        }

        /// <summary>pair += a 64-bit value (a pair) or a small constant.</summary>
        public void Add64(string pair, string value)
        {
            bool constant = !value.StartsWith('%');
            L($"s_add_u32 {Lo(pair)}, {Lo(pair)}, {(constant ? value : Lo(value))}");
            L($"s_addc_u32 {Hi(pair)}, {Hi(pair)}, {(constant ? "0" : Hi(value))}");
        }

        /// <summary>num % den (both scalar, 32-bit unsigned) in a fresh register: LLVM's sequence (a float reciprocal, refined, then two corrections).</summary>
        public string Remainder(string num, string den)
        {
            string f = V(), u = V(), nd = S(), r = S(), q = S(), m = S(), t = S();
            L($"""
                v_cvt_f32_u32_e32 {f}, {den}
                s_sub_i32 {nd}, 0, {den}
                v_rcp_iflag_f32_e32 {f}, {f}
                v_mul_f32_e32 {f}, 0x4f7ffffe, {f}
                v_cvt_u32_f32_e32 {u}, {f}
                v_readfirstlane_b32 {r}, {u}
                s_mul_i32 {nd}, {nd}, {r}
                s_mul_hi_u32 {nd}, {r}, {nd}
                s_add_i32 {r}, {r}, {nd}
                s_mul_hi_u32 {q}, {num}, {r}
                s_mul_i32 {q}, {q}, {den}
                s_sub_i32 {m}, {num}, {q}
                s_sub_i32 {t}, {m}, {den}
                s_cmp_ge_u32 {m}, {den}
                s_cselect_b32 {m}, {t}, {m}
                s_sub_i32 {t}, {m}, {den}
                s_cmp_ge_u32 {m}, {den}
                s_cselect_b32 {m}, {t}, {m}
                """);
            return m;
        }

        /// <summary>The row's max: each thread's over its columns (from -∞; v_max_f32 passes over NaN), then the block's.</summary>
        public string RowMax(string cols, string xb, string[] xs)
        {
            string max = V();
            L($"v_mov_b32_e32 {max}, {NegativeInfinity}");
            Columns(cols, e => F($"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}"), e => F($"v_max_f32_e32 {max}, {max}, {xs[e.K]}"));
            Reduce(Max, max);
            return max;
        }

        /// <summary>e = exp(x - max) as 2^((x - max) · log2 e), as CUDA's ex2 computes it.</summary>
        public static string Exp(string e, string x, string max) => F($"""
            v_sub_f32_e32 {e}, {x}, {max}
            v_mul_f32_e32 {e}, {Log2E}, {e}
            v_exp_f32_e32 {e}, {e}
            """);

        /// <summary>A column of the row as a column loop's body sees it.</summary>
        /// <param name="K">Which column of a main step (0 to 3), or the tail's (4): the index of the body's registers.</param>
        /// <param name="Off">The vector register holding the step's byte offset in the row (its first column · 4).</param>
        /// <param name="Imm">This column's instruction offset (" offset:1024" for a step's second), or empty.</param>
        /// <param name="Column">The vector register holding the column's index (only when the loop is asked for it).</param>
        public readonly record struct Element(int K, string Off, string Imm, string? Column);

        /// <summary>
        /// A loop over the row's columns, columns tid + 256 i for each thread: four a step while four fit (every load
        /// first, then every use), then one a step under EXEC narrowed to the columns below cols. Each thread so takes
        /// its columns in order.
        /// </summary>
        public void Columns(string cols, Func<Element, string> load, Func<Element, string> use, bool columns = false)
        {
            string label = NewLabel();
            string b = S(), end = S(), b4 = S(), off = V();
            string?[] column = columns ? Vs() : new string?[Unroll + 1];
            L($"s_mov_b32 {b}, 0");
            Label(label + "_main");
            L($"s_add_u32 {end}, {b}, 0x{Unroll * RowThreads:x}");
            L($"s_cmp_gt_u32 {end}, {cols}");
            L($"s_cbranch_scc1 {label}_tail");
            L($"s_lshl_b32 {b4}, {b}, 2");
            L($"v_add_u32_e32 {off}, {b4}, {Tid4}");
            if (columns)
            {
                L($"v_add_u32_e32 {column[0]}, {b}, {Tid}");
                for (int k = 1; k < Unroll; k++)
                {
                    L($"v_add_u32_e32 {column[k]}, 0x{k * RowThreads:x}, {column[0]}");
                }
            }

            for (int k = 0; k < Unroll; k++)
            {
                Lines(load(new Element(k, off, Offset(4 * RowThreads * k), column[k])));
            }

            for (int k = 0; k < Unroll; k++)
            {
                Lines(use(new Element(k, off, Offset(4 * RowThreads * k), column[k])));
            }

            L($"s_mov_b32 {b}, {end}");
            L($"s_branch {label}_main");
            Label(label + "_tail");
            L($"s_cmp_ge_u32 {b}, {cols}");
            L($"s_cbranch_scc1 {label}_end");
            string c = column[Unroll] ?? V(), saved = SPair();
            L($"s_lshl_b32 {b4}, {b}, 2");
            L($"v_add_u32_e32 {off}, {b4}, {Tid4}");
            L($"v_add_u32_e32 {c}, {b}, {Tid}");
            L($"v_cmp_gt_u32_e32 vcc, {cols}, {c}");
            L($"s_and_saveexec_b64 {saved}, vcc");
            var tail = new Element(Unroll, off, "", c);
            Lines(load(tail));
            Lines(use(tail));
            L($"s_or_b64 exec, exec, {saved}");
            L($"s_add_u32 {b}, {b}, 0x{RowThreads:x}");
            L($"s_branch {label}_tail");
            Label(label + "_end");
        }

        /// <summary>A reduction's combining step: the registers held, the registers read from the other half, its lines.</summary>
        public delegate string Combine(RowWriter writer, string[] held, string[] other);

        public static string Add(RowWriter writer, string[] held, string[] other)
        {
            var text = new StringBuilder();
            for (int i = 0; i < held.Length; i++)
            {
                text.Append(F($"v_add_f32_e32 {held[i]}, {held[i]}, {other[i]}\n"));
            }

            return text.ToString();
        }

        public static string Max(RowWriter writer, string[] held, string[] other) => F($"v_max_f32_e32 {held[0]}, {held[0]}, {other[0]}");

        // The larger value wins; on equal values the smaller column (unsigned, so -1 loses).
        public static string ArgMax(RowWriter writer, string[] held, string[] other)
        {
            string larger = writer.SPair(), equal = writer.SPair(), smaller = writer.SPair();
            return F($"""
                v_cmp_gt_f32_e64 {larger}, {other[0]}, {held[0]}
                v_cmp_eq_f32_e64 {equal}, {other[0]}, {held[0]}
                v_cmp_lt_u32_e64 {smaller}, {other[1]}, {held[1]}
                s_and_b64 {equal}, {equal}, {smaller}
                s_or_b64 vcc, {larger}, {equal}
                v_cndmask_b32_e32 {held[0]}, {held[0]}, {other[0]}, vcc
                v_cndmask_b32_e32 {held[1]}, {held[1]}, {other[1]}, vcc
                """);
        }

        /// <summary>
        /// Combines <paramref name="held"/> (one register per LDS array, the arrays 1,024 bytes apart in the area) over
        /// the block by a tree of a fixed shape: each thread's value to LDS; threads below 128 take the one 128 above
        /// (an s_barrier before and after); threads below 64, the first wave, the one 64 above, then 32, ..., 1 above,
        /// with no barrier (one wave's LDS accesses are in order; its lanes past the stride compute values nobody
        /// reads); an s_barrier; every thread reads the result. Then the other area for the next reduction, so this
        /// one's result is read before anything overwrites it.
        /// </summary>
        public void Reduce(Combine combine, params string[] held)
        {
            string[] other = new string[held.Length];
            for (int i = 0; i < held.Length; i++)
            {
                other[i] = V();
                L($"ds_write_b32 {Area}, {held[i]}{Offset(1024 * i)}");
            }

            L($"s_barrier");
            string label = NewLabel();
            string upper = SPair(), wave = SPair();
            L($"v_cmp_gt_u32_e32 vcc, 0x80, {Tid}");
            L($"s_and_saveexec_b64 {upper}, vcc");
            L($"s_cbranch_execz {label}_upper");
            Step(combine, held, other, 128);
            Label(label + "_upper");
            L($"s_or_b64 exec, exec, {upper}");
            L($"s_barrier");
            L($"v_cmp_gt_u32_e32 vcc, 64, {Tid}");
            L($"s_and_saveexec_b64 {wave}, vcc");
            L($"s_cbranch_execz {label}_wave");
            for (int stride = 64; stride >= 1; stride /= 2)
            {
                Step(combine, held, other, stride);
            }

            Label(label + "_wave");
            L($"s_or_b64 exec, exec, {wave}");
            L($"s_barrier");
            for (int i = 0; i < held.Length; i++)
            {
                L($"ds_read_b32 {held[i]}, {AreaBase}{Offset(1024 * i)}");
            }

            L($"v_xor_b32_e32 {Area}, 0x{_areaBytes:x}, {Area}");
            L($"v_xor_b32_e32 {AreaBase}, 0x{_areaBytes:x}, {AreaBase}");
        }

        private void Step(Combine combine, string[] held, string[] other, int stride)
        {
            for (int i = 0; i < held.Length; i++)
            {
                L($"ds_read_b32 {other[i]}, {Area}{Offset(1024 * i + 4 * stride)}");
            }

            Lines(combine(this, held, other));
            for (int i = 0; i < held.Length; i++)
            {
                L($"ds_write_b32 {Area}, {held[i]}{Offset(1024 * i)}");
            }
        }

        private static string Offset(int bytes) => bytes == 0 ? "" : F($" offset:{bytes}");

        /// <summary>The first thread stores each value at its address.</summary>
        public void StoreFirstLane(params (string Address, string Value)[] stores)
        {
            var at = new (string, string, string)[stores.Length];
            for (int i = 0; i < stores.Length; i++)
            {
                at[i] = (stores[i].Address, stores[i].Value, Zero);
            }

            StoreFirstLane(at);
        }

        /// <summary>The first thread stores each value at its address plus a vector register's byte offset.</summary>
        public void StoreFirstLane(params (string Address, string Value, string Offset)[] stores)
        {
            string saved = SPair();
            L($"v_cmp_eq_u32_e32 vcc, 0, {Tid}");
            L($"s_and_saveexec_b64 {saved}, vcc");
            foreach (var (address, value, offset) in stores)
            {
                L($"global_store_dword {offset}, {value}, {address}");
            }

            L($"s_or_b64 exec, exec, {saved}");
        }
    }
}
