// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// The kernels CUDA has and HIP lacked (plan 35, HIP parity), with CUDA's names and arguments (PtxKernels.Decoder.cs) and
// the CPU backend's results (CpuBackend.Quantized.cs, CpuBackend.Decoder.cs) within the device-against-CPU tolerance:
// - softmax_ce_rows_f32 (Backend.SoftmaxCrossEntropyRows, a language model's token loss): a block a row (RowWriter's
//   frame, blocks of 256 and one fixed LDS tree for each sum, as every HIP row kernel; CUDA's block takes 1,024, so the
//   sums group differently there): the row's max, Σ exp(x - max), lse = max + log Σ; the first thread writes
//   losses[r] = w_r · (lse - x[t_r]) (the target logit read before any column is rewritten); then every column is rewritten
//   with its gradient scale · w_r · (exp(x - lse) - [column = t_r]).
// - gated_act_bf16_f32 (Backend.GatedActivationPacked): y = act(gate) · up, a work-item a word of two elements (n =
//   ⌈count / 2⌉), gate and up read as floats or as packed bfloat16 (flags bit 0), y written (bit 2), gate and up packed
//   (bit 1, when not read packed) and y packed (bit 3), each packed to nearest with ties to even and NaN as 0x7FC0 (the
//   CPU's PackBFloat16); the activation is gated_act_bwd_f32's (GatedValues).
//
// | Kernel              | Arguments                                                        | Launch                        | Flops | Bytes |
// |---------------------|------------------------------------------------------------------|-------------------------------|-------|-------|
// | softmax_ce_rows_f32 | x, targets, weights, losses; cols, scale, n (n = rows)           | rows (or the grid), 256       | 8     | 12    |
// | gated_act_bf16_f32  | gate, up, pg, pu, y, py; kind, flags, count, n (n = ⌈count/2⌉)   | single, a work-item a word    | 30    | 16    |
internal static partial class GcnKernels
{
    /// <summary>The kernels of HIP's parity with CUDA, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Parity => _parity ??= [SoftmaxCrossEntropyRows(), GatedActivationPacked()];

    private static GcnKernelSource[]? _parity;

    private static GcnKernelSource SoftmaxCrossEntropyRows()
    {
        var w = new RowWriter("softmax_ce_rows_f32", arrays: 1);
        var a = w.Arguments(["x", "targets", "weights", "losses"], ["cols", "scale", "n"]);
        string x = a.Pointer("x"), targets = a.Pointer("targets"), weights = a.Pointer("weights"), losses = a.Pointer("losses");
        string cols = a.Scalar("cols"), scale = a.Scalar("scale"), n = a.Scalar("n");
        string ln2 = w.S();
        w.L($"s_mov_b32 {ln2}, {RowWriter.Ln2}");
        w.BeginRows(n);
        int offset = w.RowOffset(cols);
        string xb = w.Base(x, offset);

        // The row's weight and target class, and the target's logit, before any column is rewritten.
        string targetAt = w.ElementAddress(targets, w.Row), weightAt = w.ElementAddress(weights, w.Row);
        string targetBits = w.S(), weight = w.S(), target = w.S(), converted = w.V();
        w.L($"""
            s_load_dword {targetBits}, {targetAt}, 0x0
            s_load_dword {weight}, {weightAt}, 0x0
            s_waitcnt lgkmcnt(0)
            v_cvt_u32_f32_e32 {converted}, {targetBits}
            v_readfirstlane_b32 {target}, {converted}
            """);
        string logitAt = w.ElementAddress(xb, target), logit = w.S();
        w.L($"""
            s_load_dword {logit}, {logitAt}, 0x0
            s_waitcnt lgkmcnt(0)
            """);

        var xs = w.Vs();
        string max = w.RowMax(cols, xb, xs);
        var es = w.Vs();
        string sum = w.V();
        w.L($"v_mov_b32_e32 {sum}, 0");
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => RowWriter.Exp(es[e.K], xs[e.K], max) + $"""

            v_add_f32_e32 {sum}, {sum}, {es[e.K]}
            """);
        w.Reduce(RowWriter.Add, sum);
        string lse = w.V(), loss = w.V(), gradientScale = w.V(), saved = w.SPair(), lossAt = w.ElementAddress(losses, w.Row);
        w.L($"""
            v_log_f32_e32 {lse}, {sum}
            v_fma_f32 {lse}, {lse}, {ln2}, {max}
            v_subrev_f32_e32 {loss}, {logit}, {lse}
            v_mul_f32_e32 {loss}, {weight}, {loss}
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {saved}, vcc
            global_store_dword {w.Zero}, {loss}, {lossAt}
            s_mov_b64 exec, {saved}
            v_mov_b32_e32 {gradientScale}, {scale}
            v_mul_f32_e32 {gradientScale}, {weight}, {gradientScale}
            """);
        string[] hits = [w.SPair(), w.SPair(), w.SPair(), w.SPair(), w.SPair()];
        var less = w.Vs();
        w.Columns(cols, e => $"global_load_dword {xs[e.K]}, {e.Off}, {xb}{e.Imm}", e => RowWriter.Exp(es[e.K], xs[e.K], lse) + $"""

            v_cmp_eq_u32_e64 {hits[e.K]}, {target}, {e.Column}
            v_add_f32_e32 {less[e.K]}, -1.0, {es[e.K]}
            v_cndmask_b32_e64 {es[e.K]}, {es[e.K]}, {less[e.K]}, {hits[e.K]}
            v_mul_f32_e32 {es[e.K]}, {es[e.K]}, {gradientScale}
            global_store_dword {e.Off}, {es[e.K]}, {xb}{e.Imm}
            """, columns: true);
        w.EndRows();
        return w.Source();
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
