// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// HIP's connectionist temporal classification kernels in GCN (plan 35, G5c), CUDA's (PtxKernels.Ctc.cs) with its names,
// arguments and arithmetic: a block per sequence (grid-stride over the batch) runs α (and for the gradient β) over the
// S = 2L + 1 extended states in log space, in float, the block's threads over the states in order (tid, tid + 256, …),
// the block's fixed LDS tree (RowWriter.Reduce) between steps. The rows (stride `states`: the forward's 2 rows of α; the
// gradient's 2 rows of β, a row of terms and the label groups) live in LDS when they fit CtcRowsBytes (declared with the
// kernel; the backend checks it against the block's reported LDS, rule 84), else ("_global") in a scratch buffer; the
// gradient keeps α of every step in global memory ([batch, steps, states], then each step's row maximum, [batch, steps]).
//
// The rows stay near zero so float keeps its precision over long sequences: row t is kept as a_t = α_t - P_t, each step
// subtracting the previous row's maximum m_{t-1} (an all -∞ row counts as 0) and P_t = m_0 + … + m_{t-1} summed in double
// (v_add_f64); β likewise against the later rows' maxima. The loss is -(log(e^a(S-1) + e^a(S-2)) + P_{T-1}) in double,
// rounded once. The gradient's per-class sums go in a fixed order: the blank's through the block's tree (even states in
// order), each label's over its occurrences in label order (the label groups, made once a sequence: each label's class
// and its next occurrence, the top bit set on all but a class's first), so the bits do not change run to run; one work-item
// writes each class's gradient of a step (dlogprobs -= scale · sum · e^(top + nll - logprob)): no atomics. Labels are
// clamped to the classes (the CPU checks them). e^x and log x are the hardware's 2^x and log2 x (CUDA's ex2 and lg2).
//
// Arguments (CUDA's): logprobs, targets, meta (per sequence: input length, target length, target offset as ints), the
// outputs and scratch; steps, batch, classes, blank, batchfirst, zeroinf, states (the rows' stride: the longest sequence's
// 2L + 1). For --profile and --bench-hip (HipBackend.CtcCost): per state and step, about 12 floating-point operations
// forward (24 for the gradient) and 12 bytes (24).
internal static partial class GcnKernels
{
    /// <summary>The CTC kernels, in the order they join <see cref="All"/>.</summary>
    internal static IReadOnlyList<GcnKernelSource> Ctc => _ctc ??= [CtcLoss(shared: true), CtcBackward(shared: true), CtcLoss(shared: false), CtcBackward(shared: false)];

    private static GcnKernelSource[]? _ctc;

    /// <summary>The LDS the shared-row CTC kernels declare for their rows (bytes): rows · states floats must fit it.</summary>
    internal const int CtcRowsBytes = 32768;

    /// <summary>The LDS a shared-row CTC kernel declares in all: the reduction tree's two areas and the rows.</summary>
    internal const int CtcLdsBytes = 2 * 1024 + CtcRowsBytes;

    /// <summary>Rows (floats, stride states) a sequence keeps: the forward's 2, the gradient's 3 and the label groups (≤ states).</summary>
    internal static int CtcRows(bool backward) => backward ? 4 : 2;

    private static readonly string[] CtcScalarNames = ["steps", "batch", "classes", "blank", "batchfirst", "zeroinf", "states"];

    /// <summary>
    /// One CTC kernel's writer and the sequence's registers: the rows (LDS or a scratch pair), the meta values, and the
    /// helpers for the states' classes, the log-probabilities and the log-sums.
    /// </summary>
    private sealed class CtcWriter
    {
        public required RowWriter W { get; init; }

        public required RowWriter.Args A { get; init; }

        public required bool Shared { get; init; }

        public string Rows { get; set; } = "";

        public string Length { get; set; } = "";

        public string Labels { get; set; } = "";

        public string Offset { get; set; } = "";

        public string States { get; set; } = "";

        public string ClassesMinus1 { get; set; } = "";

        public string Ln2 { get; set; } = "";

        public string NegInf { get; set; } = "";

        private int _labels;

        public string NewLabel(string prefix) => TrainingText($".L{prefix}{_labels++}");

        // The rows' element `index` (a vector register) into a fresh register (or `target`).
        public string Read(string index, string? target = null)
        {
            string at = W.V(), value = target ?? W.V();
            W.L($"v_lshlrev_b32_e32 {at}, 2, {index}");
            W.L(Shared ? (FormattableString)$"ds_read_b32 {value}, {at} offset:{2 * 1024}" : $"global_load_dword {value}, {at}, {Rows}");
            return value;
        }

        // Stores `value` at the rows' element `index`.
        public void Write(string index, string value)
        {
            string at = W.V();
            W.L($"v_lshlrev_b32_e32 {at}, 2, {index}");
            W.L(Shared ? (FormattableString)$"ds_write_b32 {at}, {value} offset:{2 * 1024}" : $"global_store_dword {at}, {value}, {Rows}");
        }

        // The class of state s: the blank, or (odd s) label (s - 1) / 2 clamped to the classes.
        public string StateClass(string s)
        {
            string c = W.V(), odd = W.V(), j = W.V(), saved = W.SPair(), skip = NewLabel("class");
            W.L($"""
                v_mov_b32_e32 {c}, {A.Scalar("blank")}
                v_and_b32_e32 {odd}, 1, {s}
                v_cmp_ne_u32_e32 vcc, 0, {odd}
                s_and_saveexec_b64 {saved}, vcc
                s_cbranch_execz {skip}
                v_lshrrev_b32_e32 {j}, 1, {s}
                """);
            TargetLabel(j, c);
            W.Label(skip);
            W.L($"s_or_b64 exec, exec, {saved}");
            return c;
        }

        // Label j (a vector register) of the sequence into `target`, clamped to the classes.
        public void TargetLabel(string j, string target)
        {
            string at = W.V(), value = W.V();
            W.L($"""
                v_add_u32_e32 {at}, {Offset}, {j}
                v_lshlrev_b32_e32 {at}, 2, {at}
                global_load_dword {value}, {at}, {A.Pointer("targets")}
                v_cvt_i32_f32_e32 {value}, {value}
                v_max_i32_e32 {value}, 0, {value}
                v_min_i32_e32 {target}, {ClassesMinus1}, {value}
                """);
        }

        // The byte offset of the log-probability of class c (a vector register) at step t (a scalar) into a fresh register.
        public string LogProbAt(string t, string c)
        {
            string first = W.S(), other = W.S(), at = W.V();
            W.L($"""
                s_mul_i32 {first}, {W.Row}, {A.Scalar("steps")}
                s_add_u32 {first}, {first}, {t}
                s_mul_i32 {other}, {t}, {A.Scalar("batch")}
                s_add_u32 {other}, {other}, {W.Row}
                s_cmp_lg_u32 {A.Scalar("batchfirst")}, 0
                s_cselect_b32 {first}, {first}, {other}
                s_mul_i32 {first}, {first}, {A.Scalar("classes")}
                v_add_u32_e32 {at}, {first}, {c}
                v_lshlrev_b32_e32 {at}, 2, {at}
                """);
            return at;
        }

        // The log-probability of class c at step t into a fresh register.
        public string LogProb(string t, string c)
        {
            string at = LogProbAt(t, c), value = W.V();
            W.L($"global_load_dword {value}, {at}, {A.Pointer("logprobs")}");
            return value;
        }

        // log(e^a + e^b + e^c) with one logarithm into a fresh register; -∞ when all three are.
        public string LogSum3(string a, string b, string c)
        {
            string m = W.V(), d1 = W.V(), d2 = W.V(), d3 = W.V(), l = W.V(), r = W.V(), target = W.V();
            W.L($"""
                v_max_f32_e32 {m}, {a}, {b}
                v_max_f32_e32 {m}, {m}, {c}
                v_sub_f32_e32 {d1}, {a}, {m}
                v_mul_f32_e32 {d1}, {RowWriter.Log2E}, {d1}
                v_exp_f32_e32 {d1}, {d1}
                v_sub_f32_e32 {d2}, {b}, {m}
                v_mul_f32_e32 {d2}, {RowWriter.Log2E}, {d2}
                v_exp_f32_e32 {d2}, {d2}
                v_add_f32_e32 {d1}, {d1}, {d2}
                v_sub_f32_e32 {d3}, {c}, {m}
                v_mul_f32_e32 {d3}, {RowWriter.Log2E}, {d3}
                v_exp_f32_e32 {d3}, {d3}
                v_add_f32_e32 {d1}, {d1}, {d3}
                v_log_f32_e32 {l}, {d1}
                v_fma_f32 {r}, {l}, {Ln2}, {m}
                v_cmp_eq_f32_e32 vcc, {NegInf}, {m}
                v_cndmask_b32_e32 {target}, {r}, {m}, vcc
                """);
            return target;
        }

        // e^x as 2^(x · log2 e) into a fresh register.
        public string Exp(string x)
        {
            string target = W.V();
            W.L($"""
                v_mul_f32_e32 {target}, {RowWriter.Log2E}, {x}
                v_exp_f32_e32 {target}, {target}
                """);
            return target;
        }

        // The block's maximum of `value` in every thread, 0 when it is -∞ (unless `keepInfinity`): the step's barrier too.
        public void BlockMax(string value, bool keepInfinity = false)
        {
            W.Reduce(RowWriter.Max, value);
            if (!keepInfinity)
            {
                W.L($"""
                    v_cmp_eq_f32_e32 vcc, {NegInf}, {value}
                    v_cndmask_b32_e64 {value}, {value}, 0, vcc
                    """);
            }
        }

        // A fresh pair of vector registers (a double).
        public string Pair()
        {
            string a = W.V(), b = W.V();
            return TrainingText($"%v[{a[2..]}:{b[2..]}]");
        }

        // α of step t at state s from the previous row at element `from` (read by `read`), less `shift`, into a fresh register.
        public string AlphaStep(string s, string from, string shift, string t, Func<string, string> read)
        {
            string c = StateClass(s);
            string i0 = W.V(), i1 = W.V(), i2 = W.V(), s1 = W.V(), s2 = W.V(), b = W.V(), skip = W.V(), skipOk = W.SPair();
            W.L($"""
                v_add_u32_e32 {i0}, {from}, {s}
                v_add_u32_e32 {s1}, -1, {s}
                v_max_i32_e32 {s1}, 0, {s1}
                v_add_u32_e32 {i1}, {from}, {s1}
                v_add_u32_e32 {s2}, -2, {s}
                v_max_i32_e32 {s2}, 0, {s2}
                v_add_u32_e32 {i2}, {from}, {s2}
                """);
            string a0 = read(i0), a1 = read(i1), a2 = read(i2);
            string c2 = StateClass(s2);
            W.L($"""
                v_cmp_eq_u32_e32 vcc, 0, {s}
                v_cndmask_b32_e32 {a1}, {a1}, {NegInf}, vcc
                v_cmp_lt_u32_e32 vcc, 1, {s}
                v_cmp_ne_u32_e64 {skipOk}, {A.Scalar("blank")}, {c}
                s_and_b64 vcc, vcc, {skipOk}
                v_cmp_ne_u32_e64 {skipOk}, {c2}, {c}
                s_and_b64 vcc, vcc, {skipOk}
                v_cndmask_b32_e32 {skip}, {NegInf}, {a2}, vcc
                """);
            string sum = LogSum3(a0, a1, skip), lp = LogProb(t, c), result = W.V();
            W.L($"""
                v_sub_f32_e32 {b}, {sum}, {shift}
                v_add_f32_e32 {b}, {b}, {lp}
                v_cmp_eq_f32_e32 vcc, {NegInf}, {sum}
                v_cndmask_b32_e32 {result}, {b}, {sum}, vcc
                """);
            return result;
        }

        // α of step 0 at state s: the log-probability of its class for s < 2, else -∞.
        public string AlphaStart(string s)
        {
            string c = StateClass(s), zero = W.S(), result = W.V();
            W.L($"s_mov_b32 {zero}, 0");
            string lp = LogProb(zero, c);
            W.L($"""
                v_cmp_gt_u32_e32 vcc, 2, {s}
                v_cndmask_b32_e32 {result}, {NegInf}, {lp}, vcc
                """);
            return result;
        }
    }

    // The common start of a CTC kernel's sequence: the meta values, the states, the rows' place, the constants.
    private static CtcWriter CtcStart(string name, string[] pointers, bool shared, bool backward)
    {
        var w = new RowWriter(name, arrays: 1, extraLds: shared ? CtcRowsBytes : 0);
        var a = w.Arguments(pointers, CtcScalarNames);
        var c = new CtcWriter { W = w, A = a, Shared = shared };
        c.ClassesMinus1 = w.S();
        c.Ln2 = w.S();
        c.NegInf = w.V();
        w.L($"""
            s_sub_u32 {c.ClassesMinus1}, {a.Scalar("classes")}, 1
            s_mov_b32 {c.Ln2}, {RowWriter.Ln2}
            v_mov_b32_e32 {c.NegInf}, {RowWriter.NegativeInfinity}
            """);
        w.BeginRows(a.Scalar("batch"));
        string metaOffset = w.S();
        w.L($"s_mul_i32 {metaOffset}, {w.Row}, 12");
        string meta = w.SPair();
        c.Length = w.S();
        c.Labels = w.S();
        c.Offset = w.S();
        c.States = w.S();
        w.L($"""
            s_add_u32 {RowWriter.Lo(meta)}, {RowWriter.Lo(a.Pointer("meta"))}, {metaOffset}
            s_addc_u32 {RowWriter.Hi(meta)}, {RowWriter.Hi(a.Pointer("meta"))}, 0
            s_load_dword {c.Length}, {meta}, 0x0
            s_load_dword {c.Labels}, {meta}, 0x4
            s_load_dword {c.Offset}, {meta}, 0x8
            s_lshl_b32 {c.States}, {c.Labels}, 1
            s_add_u32 {c.States}, {c.States}, 1
            """);
        if (!shared)
        {
            string perSequence = w.S();
            w.L($"s_mul_i32 {perSequence}, {a.Scalar("states")}, {CtcRows(backward)}");
            c.Rows = w.Base(a.Pointer("work"), w.Product64(w.Row, perSequence));
        }

        return c;
    }

    // losses[n] = -log p(labels | sequence n): two rows of α, alternating.
    private static GcnKernelSource CtcLoss(bool shared)
    {
        var c = CtcStart(shared ? "ctc_loss_f32" : "ctc_loss_global_f32", ["logprobs", "targets", "meta", "losses", "work"], shared, backward: false);
        var (w, a) = (c.W, c.A);
        string lossAt = w.ElementAddress(a.Pointer("losses"), w.Row), empty = w.S(), emptyLoss = w.V();
        w.L($"""
            s_cmp_lg_u32 {c.Length}, 0
            s_cbranch_scc1 .Lrun
            s_cmp_eq_u32 {c.Labels}, 0
            s_cselect_b32 {empty}, 0, {RowWriter.PositiveInfinity}
            s_cmp_lg_u32 {a.Scalar("zeroinf")}, 0
            s_cselect_b32 {empty}, 0, {empty}
            v_mov_b32_e32 {emptyLoss}, {empty}
            """);
        w.StoreFirstLane((lossAt, emptyLoss));
        w.L($"s_branch .Lnext");
        w.Label(".Lrun");

        // Step 0, then each step from the previous row less its maximum, the maxima summed in double.
        string local = w.V(), m = w.V(), offset = c.Pair(), m64 = c.Pair(), t = w.S(), from = w.S(), to = w.S(), one = w.S();
        w.L($"v_mov_b32_e32 {local}, {RowWriter.NegativeInfinity}");
        BlockElements(w, ".Linit", "0", c.States, s =>
        {
            string alpha = c.AlphaStart(s);
            c.Write(s, alpha);
            w.L($"v_max_f32_e32 {local}, {local}, {alpha}");
        });
        w.L($"v_mov_b32_e32 {m}, {local}");
        c.BlockMax(m);
        CtcZero(w, offset);
        w.L($"s_mov_b32 {t}, 1");
        w.Label(".Lstep");
        w.L($"""
            s_cmp_ge_u32 {t}, {c.Length}
            s_cbranch_scc1 .Lsteps_end
            v_cvt_f64_f32_e32 {m64}, {m}
            v_add_f64 {offset}, {offset}, {m64}
            s_sub_u32 {from}, {t}, 1
            s_and_b32 {from}, {from}, 1
            s_mul_i32 {from}, {from}, {a.Scalar("states")}
            s_and_b32 {to}, {t}, 1
            s_mul_i32 {to}, {to}, {a.Scalar("states")}
            v_mov_b32_e32 {local}, {RowWriter.NegativeInfinity}
            """);
        BlockElements(w, ".Ladvance", "0", c.States, s =>
        {
            string alpha = c.AlphaStep(s, from, m, t, index => c.Read(index));
            string at = w.V();
            w.L($"v_add_u32_e32 {at}, {to}, {s}");
            c.Write(at, alpha);
            w.L($"v_max_f32_e32 {local}, {local}, {alpha}");
        });
        w.L($"v_mov_b32_e32 {m}, {local}");
        c.BlockMax(m);
        w.L($"""
            s_add_u32 {t}, {t}, 1
            s_branch .Lstep
            """);
        w.Label(".Lsteps_end");

        // The end states' log-sum of the last row, the offsets added back in double; ∞ (no alignment) is 0 with zeroinf.
        string last = w.S(), endAt = w.V(), endAt2 = w.V(), lse64 = c.Pair(), loss = w.V(), zeroInfinity = w.SPair();
        w.L($"""
            s_sub_u32 {last}, {c.Length}, 1
            s_and_b32 {last}, {last}, 1
            s_mul_i32 {last}, {last}, {a.Scalar("states")}
            s_add_u32 {last}, {last}, {c.States}
            s_sub_u32 {one}, {last}, 1
            v_mov_b32_e32 {endAt}, {one}
            s_sub_u32 {one}, {last}, 2
            s_max_i32 {one}, {one}, 0
            v_mov_b32_e32 {endAt2}, {one}
            """);
        string e1 = c.Read(endAt), e2 = c.Read(endAt2);
        string twoStates = w.S();
        w.L($"""
            s_cmp_lt_u32 {c.States}, 2
            s_cselect_b32 {twoStates}, 0, -1
            v_cmp_ne_u32_e64 vcc, 0, {twoStates}
            v_cndmask_b32_e32 {e2}, {c.NegInf}, {e2}, vcc
            """);
        string lse = c.LogSum3(e1, e2, c.NegInf);
        w.L($"""
            v_cvt_f64_f32_e32 {lse64}, {lse}
            v_add_f64 {lse64}, {lse64}, {offset}
            v_cvt_f32_f64_e32 {loss}, {lse64}
            v_xor_b32_e32 {loss}, 0x80000000, {loss}
            v_cmp_eq_f32_e32 vcc, {RowWriter.PositiveInfinity}, {loss}
            s_cmp_lg_u32 {a.Scalar("zeroinf")}, 0
            s_cselect_b64 {zeroInfinity}, -1, 0
            s_and_b64 vcc, vcc, {zeroInfinity}
            v_cndmask_b32_e64 {loss}, {loss}, 0, vcc
            """);
        w.StoreFirstLane((lossAt, loss));
        w.Label(".Lnext");
        w.L($"s_barrier");
        w.EndRows();
        return w.Source();
    }

    // Zeroes a double (a vector pair).
    private static void CtcZero(RowWriter w, string pair) => w.L($"v_cvt_f64_f32_e32 {pair}, 0");

    // dlogprobs += lossgrads[n] · ∂losses[n] / ∂logprobs: the label groups once, α of every step into `alpha`, then β
    // backwards a row at a time with each step's per-class sums of α·β (scaled by the step's largest).
    private static GcnKernelSource CtcBackward(bool shared)
    {
        var c = CtcStart(shared ? "ctc_loss_bwd_f32" : "ctc_loss_bwd_global_f32", ["logprobs", "targets", "meta", "lossgrads", "dlogprobs", "alpha", "work"], shared,
            backward: true);
        var (w, a) = (c.W, c.A);
        string states = a.Scalar("states");

        // The sequence's scale; nothing to do without one or without steps.
        string scaleAt = w.ElementAddress(a.Pointer("lossgrads"), w.Row), scale = w.S(), magnitude = w.S(), scaleV = w.V();
        w.L($"""
            s_load_dword {scale}, {scaleAt}, 0x0
            s_and_b32 {magnitude}, {scale}, 0x7fffffff
            s_cmp_eq_u32 {magnitude}, 0
            s_cbranch_scc1 .Lnext
            s_cmp_eq_u32 {c.Length}, 0
            s_cbranch_scc1 .Lnext
            v_mov_b32_e32 {scaleV}, {scale}
            """);

        // The label groups: each label's class at element classRow + j, its next occurrence of the class (or the label
        // count) at nextRow + j, the top bit set when an earlier label has the class.
        string classRow = w.S(), nextRow = w.S(), top = w.V();
        w.L($"""
            s_mul_i32 {classRow}, {states}, 3
            s_add_u32 {nextRow}, {classRow}, {c.Labels}
            v_mov_b32_e32 {top}, 0x80000000
            """);
        BlockElements(w, ".Lgclass", "0", c.Labels, j =>
        {
            string label = w.V(), at = w.V();
            c.TargetLabel(j, label);
            w.L($"v_add_u32_e32 {at}, {classRow}, {j}");
            c.Write(at, label);
        });
        w.L($"s_barrier");
        BlockElements(w, ".Lgnext", "0", c.Labels, j =>
        {
            string at = w.V(), own = w.V(), flag = w.V(), k = w.V(), next = w.V(), saved = w.SPair(), kAt = w.V(), linked = w.V();
            w.L($"v_add_u32_e32 {at}, {classRow}, {j}");
            c.Read(at, own);
            w.L($"""
                v_mov_b32_e32 {flag}, 0
                v_mov_b32_e32 {k}, 0
                s_mov_b64 {saved}, exec
                v_cmp_gt_u32_e32 vcc, {j}, {k}
                s_and_b64 exec, exec, vcc
                s_cbranch_execz .Learlier_end
                """);
            w.Label(".Learlier");
            w.L($"v_add_u32_e32 {kAt}, {classRow}, {k}");
            string other = c.Read(kAt);
            w.L($"""
                v_cmp_eq_u32_e32 vcc, {other}, {own}
                v_cndmask_b32_e32 {flag}, {flag}, {top}, vcc
                s_andn2_b64 exec, exec, vcc
                v_add_u32_e32 {k}, 1, {k}
                v_cmp_gt_u32_e32 vcc, {j}, {k}
                s_and_b64 exec, exec, vcc
                s_cbranch_execnz .Learlier
                """);
            w.Label(".Learlier_end");
            w.L($"""
                s_mov_b64 exec, {saved}
                v_mov_b32_e32 {next}, {c.Labels}
                v_add_u32_e32 {k}, 1, {j}
                v_cmp_gt_u32_e32 vcc, {c.Labels}, {k}
                s_and_b64 exec, exec, vcc
                s_cbranch_execz .Llater_end
                """);
            w.Label(".Llater");
            w.L($"v_add_u32_e32 {kAt}, {classRow}, {k}");
            string later = c.Read(kAt);
            w.L($"""
                v_cmp_eq_u32_e32 vcc, {later}, {own}
                v_cndmask_b32_e32 {next}, {next}, {k}, vcc
                s_andn2_b64 exec, exec, vcc
                v_add_u32_e32 {k}, 1, {k}
                v_cmp_gt_u32_e32 vcc, {c.Labels}, {k}
                s_and_b64 exec, exec, vcc
                s_cbranch_execnz .Llater
                """);
            w.Label(".Llater_end");
            w.L($"""
                s_mov_b64 exec, {saved}
                v_or_b32_e32 {linked}, {next}, {flag}
                v_add_u32_e32 {at}, {nextRow}, {j}
                """);
            c.Write(at, linked);
        });
        w.L($"s_barrier");

        // α of every step into the alpha buffer (this sequence's [steps, states] block), each row's maximum after all of them.
        string stepsStates = w.S(), all = w.S();
        w.L($"""
            s_mul_i32 {stepsStates}, {a.Scalar("steps")}, {states}
            s_mul_i32 {all}, {a.Scalar("batch")}, {stepsStates}
            """);
        string alpha = w.Base(a.Pointer("alpha"), w.Product64(w.Row, stepsStates));
        string maxima = w.Base(a.Pointer("alpha"), w.Product64(all, "1"));
        w.Add64(maxima, w.Bytes64(StepsOf(w, a)));
        string local = w.V(), m = w.V(), t = w.S(), from = w.S(), to = w.S();

        string AlphaRead(string index)
        {
            string at = w.V(), value = w.V();
            w.L($"""
                v_lshlrev_b32_e32 {at}, 2, {index}
                global_load_dword {value}, {at}, {alpha}
                """);
            return value;
        }

        void AlphaWrite(string index, string value)
        {
            string at = w.V();
            w.L($"""
                v_lshlrev_b32_e32 {at}, 2, {index}
                global_store_dword {at}, {value}, {alpha}
                """);
        }

        void StoreMaximum(string step, string value)
        {
            string at = w.V();
            w.L($"v_mov_b32_e32 {at}, {step}");
            w.L($"v_lshlrev_b32_e32 {at}, 2, {at}");
            w.StoreFirstLane((maxima, value, at));
        }

        w.L($"v_mov_b32_e32 {local}, {RowWriter.NegativeInfinity}");
        BlockElements(w, ".Linit", "0", c.States, s =>
        {
            string value = c.AlphaStart(s);
            AlphaWrite(s, value);
            w.L($"v_max_f32_e32 {local}, {local}, {value}");
        });
        w.L($"""
            v_mov_b32_e32 {m}, {local}
            s_mov_b32 {t}, 0
            """);
        c.BlockMax(m);
        StoreMaximum(t, m);
        w.L($"s_mov_b32 {t}, 1");
        w.Label(".Lastep");
        w.L($"""
            s_cmp_ge_u32 {t}, {c.Length}
            s_cbranch_scc1 .Lasteps_end
            s_sub_u32 {from}, {t}, 1
            s_mul_i32 {from}, {from}, {states}
            s_mul_i32 {to}, {t}, {states}
            v_mov_b32_e32 {local}, {RowWriter.NegativeInfinity}
            """);
        BlockElements(w, ".Laadvance", "0", c.States, s =>
        {
            string value = c.AlphaStep(s, from, m, t, AlphaRead);
            string at = w.V();
            w.L($"v_add_u32_e32 {at}, {to}, {s}");
            AlphaWrite(at, value);
            w.L($"v_max_f32_e32 {local}, {local}, {value}");
        });
        w.L($"v_mov_b32_e32 {m}, {local}");
        c.BlockMax(m);
        StoreMaximum(t, m);
        w.L($"""
            s_add_u32 {t}, {t}, 1
            s_branch .Lastep
            """);
        w.Label(".Lasteps_end");

        // The end states' log-sum of the last α row (-nll less the offsets); nothing more for ∞ with zeroinf.
        string lastRow = w.S(), endAt = w.V(), endAt2 = w.V(), one = w.S(), twoStates = w.S();
        w.L($"""
            s_sub_u32 {lastRow}, {c.Length}, 1
            s_mul_i32 {lastRow}, {lastRow}, {states}
            s_add_u32 {lastRow}, {lastRow}, {c.States}
            s_sub_u32 {one}, {lastRow}, 1
            v_mov_b32_e32 {endAt}, {one}
            s_sub_u32 {one}, {lastRow}, 2
            s_max_i32 {one}, {one}, 0
            v_mov_b32_e32 {endAt2}, {one}
            """);
        string e1 = AlphaRead(endAt), e2 = AlphaRead(endAt2);
        w.L($"""
            s_cmp_lt_u32 {c.States}, 2
            s_cselect_b32 {twoStates}, 0, -1
            v_cmp_ne_u32_e64 vcc, 0, {twoStates}
            v_cndmask_b32_e32 {e2}, {c.NegInf}, {e2}, vcc
            """);
        string lseEnd = c.LogSum3(e1, e2, c.NegInf);
        w.L($"""
            s_cmp_eq_u32 {a.Scalar("zeroinf")}, 0
            s_cbranch_scc1 .Lfinite
            v_cmp_eq_f32_e32 vcc, {c.NegInf}, {lseEnd}
            s_cbranch_vccnz .Lnext
            """);
        w.Label(".Lfinite");
        w.L($"s_barrier");

        // β backwards: R the later rows' maxima, Pdiff = m_t + … + m_{T-2}, both in double; each step's terms and sums.
        string later64 = c.Pair(), pdiff = c.Pair(), temp64 = c.Pair(), other64 = c.Pair(), betaMax = w.V(), topV = w.V(), topNll = w.V(), cur = w.S(), nxt = w.S(),
            alphaRow = w.S(), termRow = w.S(), lastStep = w.S(), pairs = w.S();
        CtcZero(w, later64);
        CtcZero(w, pdiff);
        w.L($"""
            v_mov_b32_e32 {betaMax}, 0
            s_lshl_b32 {termRow}, {states}, 1
            s_sub_u32 {lastStep}, {c.Length}, 1
            s_add_u32 {pairs}, {c.Labels}, 1
            s_mov_b32 {t}, {lastStep}
            """);
        w.Label(".Lbstep");
        w.L($"""
            s_cmp_lt_i32 {t}, 0
            s_cbranch_scc1 .Lnext
            s_and_b32 {cur}, {t}, 1
            s_mul_i32 {cur}, {cur}, {states}
            s_add_u32 {nxt}, {t}, 1
            s_and_b32 {nxt}, {nxt}, 1
            s_mul_i32 {nxt}, {nxt}, {states}
            s_mul_i32 {alphaRow}, {t}, {states}
            s_cmp_eq_u32 {t}, {lastStep}
            s_cbranch_scc1 .Loffsets_end
            v_cvt_f64_f32_e32 {temp64}, {betaMax}
            v_add_f64 {later64}, {later64}, {temp64}
            """);
        string maxAt = w.V(), stepMax = w.V();
        w.L($"""
            v_mov_b32_e32 {maxAt}, {t}
            v_lshlrev_b32_e32 {maxAt}, 2, {maxAt}
            global_load_dword {stepMax}, {maxAt}, {maxima}
            v_cvt_f64_f32_e32 {temp64}, {stepMax}
            v_add_f64 {pdiff}, {pdiff}, {temp64}
            """);
        w.Label(".Loffsets_end");
        w.L($"v_mov_b32_e32 {local}, {RowWriter.NegativeInfinity}");
        BlockElements(w, ".Lbeta", "0", c.States, s =>
        {
            string beta = CtcBeta(c, s, nxt, betaMax, t, lastStep), at = w.V();
            w.L($"v_add_u32_e32 {at}, {cur}, {s}");
            c.Write(at, beta);
            w.L($"v_max_f32_e32 {local}, {local}, {beta}");
        });
        w.L($"v_mov_b32_e32 {betaMax}, {local}");
        c.BlockMax(betaMax);
        w.L($"v_mov_b32_e32 {local}, {RowWriter.NegativeInfinity}");
        BlockElements(w, ".Ltop", "0", c.States, s =>
        {
            string at = w.V(), atRow = w.V();
            w.L($"""
                v_add_u32_e32 {at}, {alphaRow}, {s}
                v_add_u32_e32 {atRow}, {cur}, {s}
                """);
            string av = AlphaRead(at), bv = c.Read(atRow), sum = w.V();
            w.L($"""
                v_add_f32_e32 {sum}, {av}, {bv}
                v_max_f32_e32 {local}, {local}, {sum}
                """);
        });
        w.L($"v_mov_b32_e32 {topV}, {local}");
        c.BlockMax(topV, keepInfinity: true);
        w.L($"""
            v_cvt_f64_f32_e32 {temp64}, {topV}
            v_cvt_f64_f32_e32 {other64}, {lseEnd}
            v_add_f64 {temp64}, {temp64}, -{other64}
            v_add_f64 {temp64}, {temp64}, {later64}
            v_add_f64 {temp64}, {temp64}, -{pdiff}
            v_cvt_f32_f64_e32 {topNll}, {temp64}
            """);
        string noTop = w.SPair();
        w.L($"v_cmp_eq_f32_e64 {noTop}, {c.NegInf}, {topV}");
        BlockElements(w, ".Lterm", "0", c.States, s =>
        {
            string at = w.V(), atRow = w.V(), termAt = w.V();
            w.L($"""
                v_add_u32_e32 {at}, {alphaRow}, {s}
                v_add_u32_e32 {atRow}, {cur}, {s}
                """);
            string av = AlphaRead(at), bv = c.Read(atRow), sum = w.V();
            w.L($"""
                v_add_f32_e32 {sum}, {av}, {bv}
                v_sub_f32_e32 {sum}, {sum}, {topV}
                """);
            string term = c.Exp(sum);
            w.L($"""
                v_cndmask_b32_e64 {term}, {term}, 0, {noTop}
                v_add_u32_e32 {termAt}, {termRow}, {s}
                """);
            c.Write(termAt, term);
        });
        w.L($"s_barrier");

        // The blank's sum over the even states (state 2j, j = tid, tid + 256, …), the block's tree; the first thread's gradient.
        string blankSum = w.V();
        w.L($"v_mov_b32_e32 {blankSum}, 0");
        BlockElements(w, ".Lblank", "0", pairs, j =>
        {
            string at = w.V();
            w.L($"v_lshl_add_u32 {at}, {j}, 1, {termRow}");
            string term = c.Read(at);
            w.L($"v_add_f32_e32 {blankSum}, {blankSum}, {term}");
        });
        w.Reduce(RowWriter.Add, blankSum);
        string firstThread = w.SPair(), blank = w.V();
        w.L($"""
            v_cmp_eq_u32_e32 vcc, 0, {w.Tid}
            s_and_saveexec_b64 {firstThread}, vcc
            v_mov_b32_e32 {blank}, {a.Scalar("blank")}
            """);
        CtcClassGradient(c, t, blank, blankSum, topNll, scaleV);
        w.L($"s_or_b64 exec, exec, {firstThread}");

        // Each label's class at its first occurrence: the sum over its occurrences in label order (the chain).
        BlockElements(w, ".Llabels", "0", c.Labels, j =>
        {
            string at = w.V(), link = w.V(), first = w.SPair(), saved = w.SPair(), cls = w.V(), sum = w.V(), k = w.V(), termAt = w.V(), walking = w.SPair();
            w.L($"v_add_u32_e32 {at}, {nextRow}, {j}");
            c.Read(at, link);
            w.L($"""
                v_cmp_gt_u32_e64 {first}, {top}, {link}
                s_and_saveexec_b64 {saved}, {first}
                s_cbranch_execz .Llabel_done
                v_add_u32_e32 {at}, {classRow}, {j}
                """);
            c.Read(at, cls);
            w.L($"""
                v_mov_b32_e32 {sum}, 0
                v_mov_b32_e32 {k}, {j}
                s_mov_b64 {walking}, exec
                """);
            w.Label(".Loccurrence");
            w.L($"""
                v_lshl_add_u32 {termAt}, {k}, 1, {termRow}
                v_add_u32_e32 {termAt}, 1, {termAt}
                """);
            string term = c.Read(termAt);
            w.L($"""
                v_add_f32_e32 {sum}, {sum}, {term}
                v_add_u32_e32 {at}, {nextRow}, {k}
                """);
            c.Read(at, k);
            w.L($"""
                v_and_b32_e32 {k}, 0x7fffffff, {k}
                v_cmp_gt_u32_e32 vcc, {c.Labels}, {k}
                s_and_b64 exec, exec, vcc
                s_cbranch_execnz .Loccurrence
                s_mov_b64 exec, {walking}
                """);
            CtcClassGradient(c, t, cls, sum, topNll, scaleV);
            w.Label(".Llabel_done");
            w.L($"s_or_b64 exec, exec, {saved}");
        });
        w.L($"""
            s_barrier
            s_sub_u32 {t}, {t}, 1
            s_branch .Lbstep
            """);
        w.Label(".Lnext");
        w.L($"s_barrier");
        w.EndRows();
        return w.Source();
    }

    // The sequence's step count (steps) as a scalar of the writer: the steps argument times the row (for the maxima's place).
    private static string StepsOf(RowWriter w, RowWriter.Args a)
    {
        string value = w.S();
        w.L($"s_mul_i32 {value}, {w.Row}, {a.Scalar("steps")}");
        return value;
    }

    // β of step t at state s from the later row at element `nxt`, less the later row's maximum; at the last step the end
    // states' log-probabilities (states S - 2 and S - 1), else -∞.
    private static string CtcBeta(CtcWriter c, string s, string nxt, string laterMax, string t, string lastStep)
    {
        var w = c.W;
        string cls = c.StateClass(s), lp = c.LogProb(t, cls);
        string i0 = w.V(), i1 = w.V(), i2 = w.V(), s1 = w.V(), s2 = w.V(), limit = w.V(), skip = w.V(), b = w.V(), recur = w.V(), result = w.V(), ok = w.SPair(),
            endState = w.V();
        w.L($"""
            v_mov_b32_e32 {limit}, {c.States}
            v_add_u32_e32 {limit}, -1, {limit}
            v_add_u32_e32 {s1}, 1, {s}
            v_min_u32_e32 {s1}, {limit}, {s1}
            v_add_u32_e32 {s2}, 2, {s}
            v_min_u32_e32 {s2}, {limit}, {s2}
            v_add_u32_e32 {i0}, {nxt}, {s}
            v_add_u32_e32 {i1}, {nxt}, {s1}
            v_add_u32_e32 {i2}, {nxt}, {s2}
            """);
        string b0 = c.Read(i0), b1 = c.Read(i1), b2 = c.Read(i2);
        string cls2 = c.StateClass(s2);
        w.L($"""
            v_add_u32_e32 {endState}, 1, {s}
            v_cmp_gt_u32_e32 vcc, {c.States}, {endState}
            v_cndmask_b32_e32 {b1}, {c.NegInf}, {b1}, vcc
            v_add_u32_e32 {endState}, 2, {s}
            v_cmp_gt_u32_e32 vcc, {c.States}, {endState}
            v_cmp_ne_u32_e64 {ok}, {c.A.Scalar("blank")}, {cls}
            s_and_b64 vcc, vcc, {ok}
            v_cmp_ne_u32_e64 {ok}, {cls2}, {cls}
            s_and_b64 vcc, vcc, {ok}
            v_cndmask_b32_e32 {skip}, {c.NegInf}, {b2}, vcc
            """);
        string sum = c.LogSum3(b0, b1, skip);
        string endFirst = w.S();
        w.L($"""
            v_sub_f32_e32 {b}, {sum}, {laterMax}
            v_add_f32_e32 {b}, {b}, {lp}
            v_cmp_eq_f32_e32 vcc, {c.NegInf}, {sum}
            v_cndmask_b32_e32 {recur}, {b}, {sum}, vcc
            s_sub_u32 {endFirst}, {c.States}, 3
            v_cmp_lt_i32_e32 vcc, {endFirst}, {s}
            v_cndmask_b32_e32 {endState}, {c.NegInf}, {lp}, vcc
            s_cmp_eq_u32 {t}, {lastStep}
            s_cselect_b64 {ok}, -1, 0
            v_cndmask_b32_e64 {result}, {recur}, {endState}, {ok}
            """);
        return result;
    }

    // dlogprobs[t, class] -= scale · sum · e^(top + nll - logprobs[t, class]) for the lanes on (one a class).
    private static void CtcClassGradient(CtcWriter c, string t, string cls, string sum, string topNll, string scale)
    {
        var w = c.W;
        string at = c.LogProbAt(t, cls), lp = w.V(), d = w.V(), old = w.V();
        w.L($"""
            global_load_dword {lp}, {at}, {c.A.Pointer("logprobs")}
            v_sub_f32_e32 {d}, {topNll}, {lp}
            """);
        string e = c.Exp(d);
        w.L($"""
            v_mul_f32_e32 {e}, {e}, {sum}
            v_mul_f32_e32 {e}, {e}, {scale}
            global_load_dword {old}, {at}, {c.A.Pointer("dlogprobs")}
            v_sub_f32_e32 {old}, {old}, {e}
            global_store_dword {at}, {old}, {c.A.Pointer("dlogprobs")}
            """);
    }
}
