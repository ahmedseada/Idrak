// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;

namespace Idrak.Gpu.Hip;

// The kernels CUDA has and HIP lacked (plan 35, HIP parity), with CUDA's names and arguments (PtxKernels.Decoder.cs) and
// the CPU backend's results (CpuBackend.Quantized.cs, CpuBackend.Decoder.cs) within the device-against-CPU tolerance:
// - softmax_ce_rows_f32 (Backend.SoftmaxCrossEntropyRows, a language model's token loss): a block of 1,024 a row, its sums
//   grouped as CUDA's (each thread's columns in order, each warp's butterfly, the warps in order; plan 35's CE order):
//   the row's max, Σ exp(x - max), lse = max + log Σ; the first thread writes
//   losses[r] = w_r · (lse - x[t_r]) (the target logit read before any column is rewritten); then every column is rewritten
//   with its gradient scale · w_r · (exp(x - lse) - [column = t_r]).
//
// | Kernel              | Arguments                                                        | Launch                        | Flops | Bytes |
// |---------------------|------------------------------------------------------------------|-------------------------------|-------|-------|
// | softmax_ce_rows_f32 | x, targets, weights, losses; cols, scale, n (n = rows)           | rows (or the grid), 1,024     | 8     | 12    |
internal static partial class GcnKernels
{
    /// <summary>The kernels of HIP's parity with CUDA, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Parity => _parity ??= [SoftmaxCrossEntropyRows()];

    private static GcnKernelSource[]? _parity;

    /// <summary>Work-items of a <c>softmax_ce_rows_f32</c> block: CUDA's (PtxKernels.SoftmaxCrossEntropyThreads), so its sums group as CUDA's.</summary>
    internal const int CrossEntropyThreads = 1024;

    // CUDA's softmax_ce_rows_f32 (PtxKernels.Decoder.cs, RowBlock with StridedLoop and BlockReduce) in CUDA's order: a block
    // of 1,024 a row (rows grid-stride); thread t takes columns t, t + 1,024, … in order (its max from -∞, its Σ exp(x - max)
    // from 0); the block's sum as CUDA's BlockReduce makes it: each 32 threads (a CUDA warp: half a wave) combined by the
    // butterfly over offsets 16, 8, 4, 2, 1 (lane l with lane l + offset, in LDS), then every thread adds the 32 warps' sums in
    // warp order to 0. So the sums group exactly as CUDA's, and only the exponentials (v_exp_f32 against ex2.approx) and the
    // logarithm can differ from it.
    private static GcnKernelSource SoftmaxCrossEntropyRows()
    {
        const string Name = "softmax_ce_rows_f32", Label = ".Lsoftmax_ce_rows_f32";
        var w = new PointwiseWriter();
        string[] buffers = ["x", "targets", "weights", "losses"];
        string x = "%s[0:1]", targets = "%s[2:3]", weights = "%s[4:5]", losses = "%s[6:7]", cols = "%s8", scale = "%s9", n = "%s10", gridItems = "%s60";
        var body = new StringBuilder();
        w.Into(body);
        string tid4 = w.V(), lane = w.V(), zero = w.V();
        string stride = CvScalar(w), row = CvScalar(w), ln2 = CvScalar(w);
        string xb = w.Mask(), at = w.Mask(), targetBits = CvScalar(w), weight = CvScalar(w), target = CvScalar(w), logit = CvScalar(w), converted = w.V();
        CvLines(w, $"""
            v_lshlrev_b32_e32 {tid4}, 2, v0
            v_and_b32_e32 {lane}, 31, v0
            v_mov_b32_e32 {zero}, 0
            s_mov_b32 {ln2}, {RowWriter.Ln2}
            s_lshr_b32 {stride}, {gridItems}, 10
            s_mov_b32 {row}, s4
            s_cmp_ge_u32 {row}, {n}
            s_cbranch_scc1 {Label}_done
            {Label}_row:
            s_mul_i32 {RowWriter.Lo(xb)}, {row}, {cols}
            s_mul_hi_u32 {RowWriter.Hi(xb)}, {row}, {cols}
            s_lshl_b64 {xb}, {xb}, 2
            s_add_u32 {RowWriter.Lo(xb)}, {RowWriter.Lo(x)}, {RowWriter.Lo(xb)}
            s_addc_u32 {RowWriter.Hi(xb)}, {RowWriter.Hi(x)}, {RowWriter.Hi(xb)}
            s_lshl_b32 {RowWriter.Lo(at)}, {row}, 2
            s_mov_b32 {RowWriter.Hi(at)}, 0
            s_add_u32 {RowWriter.Lo(at)}, {RowWriter.Lo(targets)}, {RowWriter.Lo(at)}
            s_addc_u32 {RowWriter.Hi(at)}, {RowWriter.Hi(targets)}, {RowWriter.Hi(at)}
            s_load_dword {targetBits}, {at}, 0x0
            s_lshl_b32 {RowWriter.Lo(at)}, {row}, 2
            s_mov_b32 {RowWriter.Hi(at)}, 0
            s_add_u32 {RowWriter.Lo(at)}, {RowWriter.Lo(weights)}, {RowWriter.Lo(at)}
            s_addc_u32 {RowWriter.Hi(at)}, {RowWriter.Hi(weights)}, {RowWriter.Hi(at)}
            s_load_dword {weight}, {at}, 0x0
            s_waitcnt lgkmcnt(0)
            v_cvt_u32_f32_e32 {converted}, {targetBits}
            v_readfirstlane_b32 {target}, {converted}
            s_lshl_b32 {RowWriter.Lo(at)}, {target}, 2
            s_mov_b32 {RowWriter.Hi(at)}, 0
            s_add_u32 {RowWriter.Lo(at)}, {RowWriter.Lo(xb)}, {RowWriter.Lo(at)}
            s_addc_u32 {RowWriter.Hi(at)}, {RowWriter.Hi(xb)}, {RowWriter.Hi(at)}
            s_load_dword {logit}, {at}, 0x0
            s_waitcnt lgkmcnt(0)
            """);

        // The thread's max and Σ exp(x - max) over its columns in order, each combined over the block.
        string max = w.V(), sum = w.V();
        CvLines(w, $"v_mov_b32_e32 {max}, {RowWriter.NegativeInfinity}");
        CrossEntropyColumns(w, Label + "_max", cols, xb, (value, _, _) => CvLines(w, $"v_max_f32_e32 {max}, {max}, {value}"));
        CrossEntropyReduce(w, Label + "_rmax", "v_max_f32_e32", RowWriter.NegativeInfinity, max, tid4, lane, zero);
        CvLines(w, $"v_mov_b32_e32 {sum}, 0");
        CrossEntropyColumns(w, Label + "_sum", cols, xb, (value, _, _) =>
        {
            string e = w.V();
            CvLines(w, $"""
                v_sub_f32_e32 {e}, {value}, {max}
                v_mul_f32_e32 {e}, {RowWriter.Log2E}, {e}
                v_exp_f32_e32 {e}, {e}
                v_add_f32_e32 {sum}, {sum}, {e}
                """);
        });
        CrossEntropyReduce(w, Label + "_rsum", "v_add_f32_e32", "0", sum, tid4, lane, zero);

        // lse = max + log Σ; the first thread writes the row's loss; every column becomes its gradient.
        string lse = w.V(), loss = w.V(), gradientScale = w.V(), saved = w.Mask();
        CvLines(w, $"""
            v_log_f32_e32 {lse}, {sum}
            v_fma_f32 {lse}, {lse}, {ln2}, {max}
            v_subrev_f32_e32 {loss}, {logit}, {lse}
            v_mul_f32_e32 {loss}, {weight}, {loss}
            s_lshl_b32 {RowWriter.Lo(at)}, {row}, 2
            s_mov_b32 {RowWriter.Hi(at)}, 0
            s_add_u32 {RowWriter.Lo(at)}, {RowWriter.Lo(losses)}, {RowWriter.Lo(at)}
            s_addc_u32 {RowWriter.Hi(at)}, {RowWriter.Hi(losses)}, {RowWriter.Hi(at)}
            v_cmp_eq_u32_e32 vcc, 0, v0
            s_and_saveexec_b64 {saved}, vcc
            global_store_dword {zero}, {loss}, {at}
            s_mov_b64 exec, {saved}
            v_mov_b32_e32 {gradientScale}, {scale}
            v_mul_f32_e32 {gradientScale}, {weight}, {gradientScale}
            """);
        CrossEntropyColumns(w, Label + "_grad", cols, xb, (value, column, offset) =>
        {
            string e = w.V(), less = w.V(), hit = w.Mask();
            CvLines(w, $"""
                v_sub_f32_e32 {e}, {value}, {lse}
                v_mul_f32_e32 {e}, {RowWriter.Log2E}, {e}
                v_exp_f32_e32 {e}, {e}
                v_cmp_eq_u32_e64 {hit}, {target}, {column}
                v_add_f32_e32 {less}, -1.0, {e}
                v_cndmask_b32_e64 {e}, {e}, {less}, {hit}
                v_mul_f32_e32 {e}, {e}, {gradientScale}
                global_store_dword {offset}, {e}, {xb}
                """);
        });
        CvLines(w, $"""
            s_add_u32 {row}, {row}, {stride}
            s_cmp_ge_u32 {row}, {n}
            s_cbranch_scc0 {Label}_row
            {Label}_done:
              s_endpgm
            """);

        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $".kernel {Name}\n");
        foreach (string buffer in buffers)
        {
            text.Append(CultureInfo.InvariantCulture, $".arg {buffer}, 8, global\n");
        }

        text.Append(CultureInfo.InvariantCulture, $"""
            .arg cols, 4, value
            .arg scale, 4, value
            .arg n, 4, value
            .reqd_workgroup_size {CrossEntropyThreads}, 1, 1
            .lds {CrossEntropyThreads * 4}
            .user_sgpr dispatch_ptr
            .user_sgpr kernarg_segment_ptr
            .system_sgpr workgroup_id_x
              s_load_dword {gridItems}, s[0:1], 0xc

            """);
        AppendArgumentLoads(text, 2 * buffers.Length + 3, "s[2:3]");
        text.Append(w.Held).Append(body);
        return new GcnKernelSource(Name, Encoding.UTF8.GetBytes(text.ToString()));
    }

    // The thread's columns c = tid, tid + 1,024, … below cols, in order (CUDA's StridedLoop): `use` with the column's
    // value, its index and its byte offset in the row, the lanes past cols off.
    private static void CrossEntropyColumns(PointwiseWriter w, string label, string cols, string rowBase, Action<string, string, string> use)
    {
        string first = CvScalar(w), column = w.V(), offset = w.V(), value = w.V(), saved = w.Mask();
        CvLines(w, $"""
            s_mov_b32 {first}, 0
            {label}:
            s_cmp_ge_u32 {first}, {cols}
            s_cbranch_scc1 {label}_end
            v_add_u32_e32 {column}, {first}, v0
            v_lshlrev_b32_e32 {offset}, 2, {column}
            v_mov_b32_e32 {value}, 0
            v_cmp_gt_u32_e32 vcc, {cols}, {column}
            s_and_saveexec_b64 {saved}, vcc
            global_load_dword {value}, {offset}, {rowBase}
            """);
        use(value, column, offset);
        CvLines(w, $"""
            s_mov_b64 exec, {saved}
            s_add_u32 {first}, {first}, {CrossEntropyThreads}
            s_branch {label}
            {label}_end:
            """);
    }

    // `held` combined over the block as CUDA's BlockReduce: each 32 work-items' butterfly over offsets 16 … 1 (lane l with
    // lane l + offset, in LDS), then every work-item combines the 32 warps' results in order, from `identity`.
    private static void CrossEntropyReduce(PointwiseWriter w, string label, string op, string identity, string held, string tid4, string lane, string zero)
    {
        string own = w.V(), partner = w.V(), part = w.V(), saved = w.Mask();
        CvLines(w, $"""
            ds_write_b32 {tid4}, {held}
            s_barrier
            v_mov_b32_e32 {own}, 0
            v_mov_b32_e32 {partner}, 0
            """);
        for (int offset = 16; offset >= 1; offset /= 2)
        {
            CvLines(w, $"""
                v_cmp_gt_u32_e32 vcc, {offset}, {lane}
                s_and_saveexec_b64 {saved}, vcc
                ds_read_b32 {own}, {tid4}
                ds_read_b32 {partner}, {tid4} offset:{4 * offset}
                {op} {own}, {own}, {partner}
                ds_write_b32 {tid4}, {own}
                s_mov_b64 exec, {saved}
                """);
        }

        CvLines(w, $"""
            s_barrier
            v_mov_b32_e32 {held}, {identity}
            """);
        for (int warp = 0; warp < CrossEntropyThreads / 32; warp++)
        {
            CvLines(w, $"""
                ds_read_b32 {part}, {zero}{(warp == 0 ? "" : TrainingText($" offset:{warp * 128}"))}
                {op} {held}, {held}, {part}
                """);
        }

        CvLines(w, $"s_barrier");
    }

}
