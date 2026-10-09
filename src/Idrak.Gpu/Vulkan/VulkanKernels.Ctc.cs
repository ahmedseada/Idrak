// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Connectionist temporal classification on the device (Backend.CtcLoss and CtcLossBackward): a workgroup per sequence
// runs α (and for the gradient β) over the S = 2L + 1 extended states in log space, in float, the workgroup's invocations
// over the states (a fixed stride) and a barrier between steps. Rows of α and β live in workgroup memory where a row fits
// the workgroup's width (the "_global" variants keep them in a scratch buffer instead: longer label sequences). Each
// sequence writes only its own loss and gradient rows; the per-class sums of the gradient go in a fixed order (the blank's
// through the workgroup's reduction, each label's over its occurrences in label order), so the bits do not change run to
// run. Buffers: log-probabilities, targets (ids as floats), meta (per sequence: input length, target length, target
// offset, as ints), then the outputs and scratch; push constants: steps, batch, classes, blank, batchFirst, zeroInfinity,
// states (the scratch rows' stride: the longest sequence's 2L + 1).
internal static partial class VulkanKernels
{
    private static IEnumerable<(string, Func<SpirvKernel>)> CtcKernels()
    {
        foreach (bool shared in new[] { true, false })
        {
            string suffix = shared ? "" : "_global";
            yield return ("ctc_loss" + suffix, () => CtcLoss("ctc_loss" + suffix, shared));
            yield return ("ctc_loss_backward" + suffix, () => CtcLossBackward("ctc_loss_backward" + suffix, shared));
        }
    }

    // The values every CTC kernel reads per sequence.
    private sealed class CtcSequence
    {
        public required KernelBuilder K;
        public required Buf LogProbs, Targets, Meta;
        public required Val Steps, Batch, Classes, Blank, BatchFirst, ZeroInfinity, Stride;
        public Val N, Length, Labels, Offset, States;

        public void Begin(Val n)
        {
            N = n;
            Length = Meta.Int(3 * n);
            Labels = Meta.Int(3 * n + 1);
            Offset = Meta.Int(3 * n + 2);
            States = 2 * Labels + 1;
        }

        // The log-probability of class c at step t.
        public Val LogProb(Val t, Val c) => LogProbs[(K.Select(BatchFirst.Ne(0), N * Steps + t, t * Batch + N)) * Classes + c];

        // The class of extended state s: the blank on even states, label (s - 1) / 2 on odd ones (read only there: an
        // empty sequence's offset may be past the targets), clamped to the classes (the CPU checks the labels; the device
        // reads nothing outside the log-probabilities whatever they hold).
        public Val Class(Val s)
        {
            var c = K.Local(Blank);
            K.If((s & 1).Eq(1), () => c.V = Label(s >> 1));
            return c.V;
        }

        // Label j, clamped to the classes.
        public Val Label(Val j) => K.Clamp(Targets[Offset + j].ToInt(), K.Int(0), Classes - 1);
    }

    private static readonly float NegInf = float.NegativeInfinity;

    // log(e^a + e^b + e^c) with one logarithm; -∞ when all three are.
    private static Val LogSum3(KernelBuilder k, Val a, Val b, Val c)
    {
        var m = k.Max(a, k.Max(b, c));
        var finite = m > NegInf;
        var safe = k.Select(finite, m, k.Float(0f));
        var sum = k.Exp(a - safe) + k.Exp(b - safe) + k.Exp(c - safe);
        return k.Select(finite, safe + k.Log(sum), k.Float(NegInf));
    }

    // α of step t at state s from the row of step t - 1 (read through `previous`).
    private static Val AlphaStep(KernelBuilder k, CtcSequence q, Val t, Val s, Func<Val, Val> previous)
    {
        var c = q.Class(s);
        var a = previous(s);
        var b = k.Select(s > 0, previous(k.Max(s - 1, k.Int(0))), k.Float(NegInf));
        var skipAllowed = (s > 1) & c.Ne(q.Blank) & c.Ne(q.Class(k.Max(s - 2, k.Int(0))));
        var skip = k.Select(skipAllowed, previous(k.Max(s - 2, k.Int(0))), k.Float(NegInf));
        var sum = LogSum3(k, a, b, skip);
        return k.Select(sum > NegInf, sum + q.LogProb(t, c), k.Float(NegInf));
    }

    // α of step 0 at state s: the blank and the first label, -∞ elsewhere.
    private static Val AlphaStart(KernelBuilder k, CtcSequence q, Val s)
    {
        var v = k.Local(NegInf);
        k.If(s < 2, () => v.V = q.LogProb(k.Int(0), q.Class(s)));
        return v.V;
    }

    private static CtcSequence CtcInputs(KernelBuilder k, out Buf[] outputs, params string[] names)
    {
        var (logProbs, targets, meta) = (k.Buffer("logProbs"), k.Buffer("targets"), k.Buffer("meta"));
        outputs = [.. names.Select(name => k.Buffer(name))];
        return new CtcSequence
        {
            K = k, LogProbs = logProbs, Targets = targets, Meta = meta,
            Steps = k.PushInt("steps"), Batch = k.PushInt("batch"), Classes = k.PushInt("classes"), Blank = k.PushInt("blank"),
            BatchFirst = k.PushInt("batchFirst"), ZeroInfinity = k.PushInt("zeroInfinity"), Stride = k.PushInt("states"),
        };
    }

    // losses[n] = -log p(labels | sequence n): two rows of α, alternating.
    private static SpirvKernel CtcLoss(string name, bool shared)
    {
        var k = new KernelBuilder(name, Block);
        var q = CtcInputs(k, out var outputs, "losses", "work");
        var (losses, work) = (outputs[0], outputs[1]);
        var rows = shared ? k.Shared("rows", 2 * Block) : null;
        EachRow(k, q.Batch, n =>
        {
            q.Begin(n);
            var rowBase = n * 2 * q.Stride;
            Val Read(Val i) => shared ? rows![i] : work[rowBase + i];
            void Write(Val i, Val v)
            {
                if (shared)
                {
                    rows![i] = v;
                }
                else
                {
                    work[rowBase + i] = v;
                }
            }

            var stride = shared ? k.Int(Block) : q.Stride;
            k.If(q.Length > 0, () =>
            {
                k.For(k.LocalX, q.States, Block, s => Write(s, AlphaStart(k, q, s)));
                Barrier(k, shared);
                k.For(k.Int(1), q.Length, 1, t =>
                {
                    var (from, to) = ((t - 1 & 1) * stride, (t & 1) * stride);
                    k.For(k.LocalX, q.States, Block, s => Write(to + s, AlphaStep(k, q, t, s, i => Read(from + i))));
                    Barrier(k, shared);
                });
                var last = (q.Length - 1 & 1) * stride;
                k.If(k.LocalX.Eq(0), () =>
                {
                    var end = Read(last + q.States - 1);
                    var before = k.Select(q.States > 1, Read(last + k.Max(q.States - 2, k.Int(0))), k.Float(NegInf));
                    var nll = -LogSum3(k, end, before, k.Float(NegInf));
                    losses[n] = k.Select(q.ZeroInfinity.Ne(0) & (nll >= float.PositiveInfinity), k.Float(0f), nll);
                });
                Barrier(k, shared);
            }, () => k.If(k.LocalX.Eq(0), () => losses[n] = k.Select(q.Labels.Eq(0) | q.ZeroInfinity.Ne(0), k.Float(0f), k.Float(float.PositiveInfinity))));
        });
        return k.Build();
    }

    // dLogProbs += lossGrads[n] · ∂losses[n] / ∂logProbs: α of every step into `alpha` ([batch, steps, states]), then β
    // backwards a row at a time with each step's per-class sums of α·β (scaled by the step's largest).
    private static SpirvKernel CtcLossBackward(string name, bool shared)
    {
        var k = new KernelBuilder(name, Block);
        var q = CtcInputs(k, out var outputs, "lossGrads", "dLogProbs", "alpha", "work");
        var (lossGrads, gradient, alpha, work) = (outputs[0], outputs[1], outputs[2], outputs[3]);
        var rows = shared ? k.Shared("rows", 2 * Block) : null;
        var terms = shared ? k.Shared("terms", Block) : null;
        var scratch = k.Shared("scratch", Block);
        EachRow(k, q.Batch, n =>
        {
            q.Begin(n);
            var scale = lossGrads[n];
            var workBase = n * 3 * q.Stride;
            var stride = shared ? k.Int(Block) : q.Stride;
            Val Beta(Val i) => shared ? rows![i] : work[workBase + i];
            void SetBeta(Val i, Val v)
            {
                if (shared)
                {
                    rows![i] = v;
                }
                else
                {
                    work[workBase + i] = v;
                }
            }

            Val Term(Val s) => shared ? terms![s] : work[workBase + 2 * q.Stride + s];
            void SetTerm(Val s, Val v)
            {
                if (shared)
                {
                    terms![s] = v;
                }
                else
                {
                    work[workBase + 2 * q.Stride + s] = v;
                }
            }

            var alphaBase = n * q.Steps * q.Stride;
            k.If((q.Length > 0) & scale.Ne(k.Float(0f)), () =>
            {
                // α of every step.
                k.For(k.LocalX, q.States, Block, s => alpha[alphaBase + s] = AlphaStart(k, q, s));
                k.BufferBarrier();
                k.For(k.Int(1), q.Length, 1, t =>
                {
                    k.For(k.LocalX, q.States, Block, s =>
                        alpha[alphaBase + t * q.Stride + s] = AlphaStep(k, q, t, s, i => alpha[alphaBase + (t - 1) * q.Stride + i]));
                    k.BufferBarrier();
                });
                var lastRow = alphaBase + (q.Length - 1) * q.Stride;
                var nll = -LogSum3(k, alpha[lastRow + q.States - 1], k.Select(q.States > 1, alpha[lastRow + k.Max(q.States - 2, k.Int(0))], k.Float(NegInf)),
                    k.Float(NegInf));
                k.If(!(q.ZeroInfinity.Ne(0) & (nll >= float.PositiveInfinity)), () =>
                {
                    // β backwards, a row at a time (alternating), with the gradient of each step.
                    var t = k.Local(q.Length - 1);
                    k.While(() => t.V >= 0, () =>
                    {
                        var step = t.V;
                        var (current, next) = ((step & 1) * stride, (step + 1 & 1) * stride);
                        k.For(k.LocalX, q.States, Block, s =>
                        {
                            var c = q.Class(s);
                            var last = step.Eq(q.Length - 1);
                            var start = k.Select((s >= q.States - 2), q.LogProb(step, c), k.Float(NegInf));
                            var a = Beta(next + s);
                            var b = k.Select(s + 1 < q.States, Beta(next + k.Min(s + 1, q.States - 1)), k.Float(NegInf));
                            var skipAllowed = (s + 2 < q.States) & c.Ne(q.Blank) & c.Ne(q.Class(k.Min(s + 2, q.States - 1)));
                            var skip = k.Select(skipAllowed, Beta(next + k.Min(s + 2, q.States - 1)), k.Float(NegInf));
                            var value = k.Local(start);
                            k.If(!last, () =>
                            {
                                var sum = LogSum3(k, a, b, skip);
                                value.V = k.Select(sum > NegInf, sum + q.LogProb(step, c), k.Float(NegInf));
                            });
                            SetBeta(current + s, value.V);
                        });
                        Barrier(k, shared);

                        // The step's largest α·β (log), then each state's share e^(α + β - top).
                        var row = alphaBase + step * q.Stride;
                        var mine = k.Local(NegInf);
                        k.For(k.LocalX, q.States, Block, s => mine.V = k.Max(mine.V, alpha[row + s] + Beta(current + s)));
                        var top = k.ReduceMax(scratch, mine.V);
                        var finite = top > NegInf;
                        k.For(k.LocalX, q.States, Block, s => SetTerm(s, k.Select(finite, k.Exp(alpha[row + s] + Beta(current + s) - top), k.Float(0f))));
                        Barrier(k, shared);

                        // The blank's sum over the even states (the workgroup's reduction), each label's over its occurrences in
                        // label order (by the invocation of its first occurrence).
                        var blankPart = k.Local(0f);
                        k.For(k.LocalX * 2, q.States, 2 * Block, s => blankPart.V = blankPart.V + Term(s));
                        var blankSum = k.ReduceSum(scratch, blankPart.V);
                        k.If(k.LocalX.Eq(0), () =>
                        {
                            var at = (k.Select(q.BatchFirst.Ne(0), q.N * q.Steps + step, step * q.Batch + q.N)) * q.Classes + q.Blank;
                            gradient[at] = gradient[at] - scale * (blankSum * k.Exp(top + nll - q.LogProbs[at]));
                        });
                        k.For(k.LocalX, q.Labels, Block, j =>
                        {
                            var c = q.Label(j);
                            var earlier = k.Local(0);
                            k.For(k.Int(0), j, 1, i => earlier.V = earlier.V | k.Select(q.Label(i).Eq(c), k.Int(1), k.Int(0)));
                            k.If(earlier.V.Eq(0), () =>
                            {
                                var sum = k.Local(0f);
                                k.For(j, q.Labels, 1, i => k.If(q.Label(i).Eq(c), () => sum.V = sum.V + Term(2 * i + 1)));
                                var at = (k.Select(q.BatchFirst.Ne(0), q.N * q.Steps + step, step * q.Batch + q.N)) * q.Classes + c;
                                gradient[at] = gradient[at] - scale * (sum.V * k.Exp(top + nll - q.LogProbs[at]));
                            });
                        });
                        Barrier(k, shared);
                        t.V = step - 1;
                    });
                });
            });
        });
        return k.Build();
    }

    // A barrier that orders the rows' memory: workgroup memory, or the scratch buffer too.
    private static void Barrier(KernelBuilder k, bool shared)
    {
        if (shared)
        {
            k.Barrier();
        }
        else
        {
            k.BufferBarrier();
        }
    }
}
