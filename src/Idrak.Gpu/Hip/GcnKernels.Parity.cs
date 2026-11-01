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
// - gated_act_bf16_f32 (Backend.GatedActivationPacked): y = act(gate) · up, a work-item a word of two elements (n =
//   ⌈count / 2⌉), gate and up read as floats or as packed bfloat16 (flags bit 0), y written (bit 2), gate and up packed
//   (bit 1, when not read packed) and y packed (bit 3), each packed to nearest with ties to even and NaN as 0x7FC0 (the
//   CPU's PackBFloat16); the activation is gated_act_bwd_f32's (GatedValues).
//
// | Kernel              | Arguments                                                        | Launch                        | Flops | Bytes |
// |---------------------|------------------------------------------------------------------|-------------------------------|-------|-------|
// | softmax_ce_rows_f32 | x, targets, weights, losses; cols, scale, n (n = rows)           | rows (or the grid), 1,024     | 8     | 12    |
// | gated_act_bf16_f32  | gate, up, pg, pu, y, py; kind, flags, count, n (n = ⌈count/2⌉)   | single, a work-item a word    | 30    | 16    |
internal static partial class GcnKernels
{
    /// <summary>The kernels of HIP's parity with CUDA, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Parity => _parity ??= [SoftmaxCrossEntropyRows(), GatedActivationPacked()];

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

    private static GcnKernelSource GatedActivationPacked() =>
        TrainingElementwise("gated_act_bf16_f32", new(["gate", "up", "pg", "pu", "y", "py"], ["kind", "flags", "count"], [], [])
        {
            Index = true,
            Single = true,
        }, (w, e) =>
        {
            string kind = e.Values[0], flags = e.Values[1], count = e.Values[2];
            string word = w.V(), packedGate = w.V(), packedUp = w.V(), bias = CvHeld(w, "s_movk_i32 {0}, 0x7fff"), quiet = w.V();
            w.I(TrainingText($"""
                v_mov_b32_e32 {quiet}, 0x7fc0
                v_lshlrev_b32_e32 {word}, 2, {e.Index}
                v_mov_b32_e32 {packedGate}, 0
                v_mov_b32_e32 {packedUp}, 0
                """));
            TrainingGate(w, flags, 0, e.Label + "_no_packed_in");
            w.I(TrainingText($"global_load_dword {packedGate}, {word}, {e.Pointers[2]}"));
            w.I(TrainingText($"global_load_dword {packedUp}, {word}, {e.Pointers[3]}"));
            w.I(TrainingText($"{e.Label}_no_packed_in:"));

            // Each half's bfloat16 bits (0 for an element past count, as bf16_pack_f32 leaves it).
            var gateBits = new string[2];
            var upBits = new string[2];
            var yBits = new string[2];
            for (int half = 0; half < 2; half++)
            {
                gateBits[half] = w.V();
                upBits[half] = w.V();
                yBits[half] = w.V();
                string element = w.V(), at = w.V(), g = w.V(), u = w.V(), y = w.V(), saved = w.Mask();
                string label = TrainingText($"{e.Label}_h{half}");
                w.I(TrainingText($"""
                    v_mov_b32_e32 {gateBits[half]}, 0
                    v_mov_b32_e32 {upBits[half]}, 0
                    v_mov_b32_e32 {yBits[half]}, 0
                    v_lshl_add_u32 {element}, {e.Index}, 1, {half}
                    v_cmp_gt_u32_e32 vcc, {count}, {element}
                    s_and_saveexec_b64 {saved}, vcc
                    s_cbranch_execz {label}_out
                    v_lshlrev_b32_e32 {at}, 2, {element}
                    """));
                w.I(half == 0 ? TrainingText($"v_lshlrev_b32_e32 {g}, 16, {packedGate}") : TrainingText($"v_and_b32_e32 {g}, 0xffff0000, {packedGate}"));
                w.I(half == 0 ? TrainingText($"v_lshlrev_b32_e32 {u}, 16, {packedUp}") : TrainingText($"v_and_b32_e32 {u}, 0xffff0000, {packedUp}"));
                ParitySkipIfSet(w, flags, 0, label + "_floats");
                w.I(TrainingText($"global_load_dword {g}, {at}, {e.Pointers[0]}"));
                w.I(TrainingText($"global_load_dword {u}, {at}, {e.Pointers[1]}"));
                w.I(TrainingText($"{label}_floats:"));
                var (value, _, upValue, _) = GatedValues(w, label, g, u, kind);
                w.I(TrainingText($"v_mul_f32_e32 {y}, {value}, {upValue}"));
                TrainingGate(w, flags, 2, label + "_no_y");
                w.I(TrainingText($"global_store_dword {at}, {y}, {e.Pointers[4]}"));
                w.I(TrainingText($"{label}_no_y:"));
                ParityRound(w, gateBits[half], g, bias, quiet);
                ParityRound(w, upBits[half], u, bias, quiet);
                ParityRound(w, yBits[half], y, bias, quiet);
                w.I(TrainingText($"{label}_out:"));
                w.I(TrainingText($"s_mov_b64 exec, {saved}"));
            }

            // The words: gate and up packed (bit 1, unless they were read packed), y packed (bit 3).
            string gateWord = w.V(), upWord = w.V(), yWord = w.V();
            w.I(TrainingText($"""
                v_lshl_or_b32 {gateWord}, {gateBits[1]}, 16, {gateBits[0]}
                v_lshl_or_b32 {upWord}, {upBits[1]}, 16, {upBits[0]}
                v_lshl_or_b32 {yWord}, {yBits[1]}, 16, {yBits[0]}
                """));
            ParitySkipIfSet(w, flags, 0, e.Label + "_no_pack");
            TrainingGate(w, flags, 1, e.Label + "_no_pack");
            w.I(TrainingText($"global_store_dword {word}, {gateWord}, {e.Pointers[2]}"));
            w.I(TrainingText($"global_store_dword {word}, {upWord}, {e.Pointers[3]}"));
            w.I(TrainingText($"{e.Label}_no_pack:"));
            TrainingGate(w, flags, 3, e.Label + "_no_py");
            w.I(TrainingText($"global_store_dword {word}, {yWord}, {e.Pointers[5]}"));
            w.I(TrainingText($"{e.Label}_no_py:"));
        });

    // To `skip` when the value's bit is set (TrainingGate's inverse).
    private static void ParitySkipIfSet(PointwiseWriter w, string value, int bit, string skip)
    {
        w.I(TrainingText($"s_bitcmp1_b32 {value}, {bit}"));
        w.I(TrainingText($"s_cbranch_scc1 {skip}"));
    }

    // target = value's bfloat16 bits: (u + 0x7FFF + ((u >> 16) & 1)) >> 16, or 0x7FC0 for a NaN (bf16_pack_f32's rounding).
    private static void ParityRound(PointwiseWriter w, string target, string value, string bias, string quiet)
    {
        string odd = w.V();
        w.I(TrainingText($"""
            v_bfe_u32 {odd}, {value}, 16, 1
            v_add3_u32 {target}, {value}, {bias}, {odd}
            v_lshrrev_b32_e32 {target}, 16, {target}
            v_cmp_u_f32_e32 vcc, {value}, {value}
            v_cndmask_b32_e32 {target}, {target}, {quiet}, vcc
            """));
    }
}
