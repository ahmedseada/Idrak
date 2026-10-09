// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Abstraction.Devices.Cpu;

// Connectionist temporal classification (Graves, Fernández, Gomez and Schmidhuber 2006, "Connectionist Temporal
// Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"), in log space. A sequence of L labels
// is extended to S = 2L + 1 states (a blank before, between and after the labels); α and β run over the states, in double
// precision. The loss keeps two rows of α (O(S) memory a sequence); the gradient recomputes α, keeps all of it
// (O(T·S)) and runs β backwards a row at a time. Sequences run in parallel; each writes only its own gradient rows.
internal sealed partial class CpuBackend
{
    public override void CtcLossKernel(Storage logProbs, Storage targets, Storage losses, ReadOnlySpan<int> inputLengths, ReadOnlySpan<int> targetLengths,
        ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        float[] lp = D(logProbs), lv = D(losses);
        var problem = new CtcProblem(lp, ReadLabels(D(targets), targetLengths, targetOffsets, batch, classes, blank), inputLengths.ToArray(), steps, batch, classes, blank, batchFirst);
        For(batch, problem.Work, (start, end) =>
        {
            for (int n = start; n < end; n++)
            {
                double nll = problem.NegativeLogLikelihood(n, null);
                lv[n] = double.IsPositiveInfinity(nll) && zeroInfinity ? 0f : (float)nll;
            }
        });
    }

    public override void CtcLossBackwardKernel(Storage logProbs, Storage targets, Storage lossGrads, Storage dLogProbs, ReadOnlySpan<int> inputLengths,
        ReadOnlySpan<int> targetLengths, ReadOnlySpan<int> targetOffsets, int steps, int batch, int classes, int blank, bool batchFirst, bool zeroInfinity)
    {
        float[] lp = D(logProbs), gv = D(lossGrads), dv = D(dLogProbs);
        var problem = new CtcProblem(lp, ReadLabels(D(targets), targetLengths, targetOffsets, batch, classes, blank), inputLengths.ToArray(), steps, batch, classes, blank, batchFirst);
        For(batch, problem.Work * 2, (start, end) =>
        {
            for (int n = start; n < end; n++)
            {
                if (gv[n] != 0f)
                {
                    problem.Gradient(n, gv[n], zeroInfinity, dv);
                }
            }
        });
    }

    // The labels of each sequence as integers, checked once (a label outside the classes, or the blank, is an error).
    private static int[][] ReadLabels(float[] targets, ReadOnlySpan<int> lengths, ReadOnlySpan<int> offsets, int batch, int classes, int blank)
    {
        var labels = new int[batch][];
        for (int n = 0; n < batch; n++)
        {
            var row = labels[n] = new int[lengths[n]];
            for (int i = 0; i < row.Length; i++)
            {
                float value = targets[offsets[n] + i];
                int label = (int)value;
                if (label != value || (uint)label >= (uint)classes || label == blank)
                {
                    throw new ArgumentException($"CTC target {i} of sequence {n} is {value}: labels are whole numbers in [0, {classes}) other than the blank ({blank}).");
                }

                row[i] = label;
            }
        }

        return labels;
    }

    private sealed class CtcProblem(float[] logProbs, int[][] labels, int[] inputLengths, int steps, int batch, int classes, int blank, bool batchFirst)
    {
        // About ten operations a state and step; the cut-over to several cores reads it.
        public long Work
        {
            get
            {
                long work = 0;
                for (int n = 0; n < batch; n++)
                {
                    work += 10L * inputLengths[n] * (2 * labels[n].Length + 1);
                }

                return work;
            }
        }

        // Where the log-probability of class c at step t of sequence n is.
        private int At(int t, int n, int c) => (batchFirst ? n * steps + t : t * batch + n) * classes + c;

        // The class of extended state s: the blank on even states, label (s - 1) / 2 on odd ones.
        private int State(int[] label, int s) => (s & 1) == 0 ? blank : label[s >> 1];

        private static double LogAdd(double a, double b)
        {
            if (double.IsNegativeInfinity(a))
            {
                return b;
            }

            if (double.IsNegativeInfinity(b))
            {
                return a;
            }

            return a > b ? a + Math.Log(1 + Math.Exp(b - a)) : b + Math.Log(1 + Math.Exp(a - b));
        }

        // log(e^a + e^b + e^c) with one logarithm (c is -∞ when a state has only two predecessors).
        private static double LogSum(double a, double b, double c)
        {
            double m = Math.Max(a, Math.Max(b, c));
            return double.IsNegativeInfinity(m) ? m : m + Math.Log(Math.Exp(a - m) + Math.Exp(b - m) + (double.IsNegativeInfinity(c) ? 0 : Math.Exp(c - m)));
        }

        // α of step t from α of step t - 1 (log space). Only the states an alignment can be in at step t are computed: at
        // most 2(t + 1) from the start, and at least S - 2(T - t) so the end is still reachable.
        private void Advance(int[] label, int n, int t, int length, ReadOnlySpan<double> previous, Span<double> next)
        {
            int states = next.Length, first = Math.Max(0, states - 2 * (length - t)), end = Math.Min(states, 2 * (t + 1));
            next[..first].Fill(double.NegativeInfinity);
            next[end..].Fill(double.NegativeInfinity);
            int row = At(t, n, 0);
            for (int s = first; s < end; s++)
            {
                int c = State(label, s);
                double skip = s > 1 && c != blank && c != State(label, s - 2) ? previous[s - 2] : double.NegativeInfinity;
                double sum = LogSum(previous[s], s > 0 ? previous[s - 1] : double.NegativeInfinity, skip);
                next[s] = double.IsNegativeInfinity(sum) ? sum : sum + logProbs[row + c];
            }
        }

        private void Start(int[] label, int n, Span<double> first)
        {
            first.Fill(double.NegativeInfinity);
            first[0] = logProbs[At(0, n, blank)];
            if (first.Length > 1)
            {
                first[1] = logProbs[At(0, n, label[0])];
            }
        }

        private static double End(ReadOnlySpan<double> last) => last.Length > 1 ? LogAdd(last[^1], last[^2]) : last[^1];

        /// <summary>-log p(labels | sequence n); with <paramref name="alpha"/> every step's α is kept there ([T, S]).</summary>
        public double NegativeLogLikelihood(int n, double[]? alpha)
        {
            var label = labels[n];
            int states = 2 * label.Length + 1, length = inputLengths[n];
            if (length == 0)
            {
                return label.Length == 0 ? 0 : double.PositiveInfinity;
            }

            double[]? rows = alpha is null ? ArrayPool<double>.Shared.Rent(2 * states) : null;
            try
            {
                Span<double> Row(int t) => alpha is not null ? alpha.AsSpan(t * states, states) : rows.AsSpan((t & 1) * states, states);
                Start(label, n, Row(0));
                for (int t = 1; t < length; t++)
                {
                    Advance(label, n, t, length, Row(t - 1), Row(t));
                }

                return -End(Row(length - 1));
            }
            finally
            {
                if (rows is not null)
                {
                    ArrayPool<double>.Shared.Return(rows);
                }
            }
        }

        /// <summary>gradient += scale · ∂(-log p) / ∂logProbs for sequence n.</summary>
        public void Gradient(int n, float scale, bool zeroInfinity, float[] gradient)
        {
            var label = labels[n];
            int states = 2 * label.Length + 1, length = inputLengths[n];
            if (length == 0)
            {
                return;
            }

            var alpha = ArrayPool<double>.Shared.Rent(length * states);
            var beta = ArrayPool<double>.Shared.Rent(2 * states);
            try
            {
                double nll = NegativeLogLikelihood(n, alpha);
                if (double.IsPositiveInfinity(nll) && zeroInfinity)
                {
                    return;
                }

                // The distinct classes of the states, and which of them each state is (the blank is class 0).
                var distinct = new List<int> { blank };
                var slot = new int[states];
                for (int s = 1; s < states; s += 2)
                {
                    int index = distinct.IndexOf(label[s >> 1], 1);
                    slot[s] = index > 0 ? index : distinct.Count;
                    if (index <= 0)
                    {
                        distinct.Add(label[s >> 1]);
                    }
                }

                var sums = new double[distinct.Count];
                Span<double> Beta(int t) => beta.AsSpan((t & 1) * states, states);
                for (int t = length - 1; t >= 0; t--)
                {
                    var current = Beta(t);
                    if (t == length - 1)
                    {
                        current.Fill(double.NegativeInfinity);
                        current[states - 1] = logProbs[At(t, n, State(label, states - 1))];
                        if (states > 1)
                        {
                            current[states - 2] = logProbs[At(t, n, State(label, states - 2))];
                        }
                    }
                    else
                    {
                        var next = Beta(t + 1);
                        int row = At(t, n, 0);
                        for (int s = 0; s < states; s++)
                        {
                            int c = State(label, s);
                            double skip = s + 2 < states && c != blank && c != State(label, s + 2) ? next[s + 2] : double.NegativeInfinity;
                            double sum = LogSum(next[s], s + 1 < states ? next[s + 1] : double.NegativeInfinity, skip);
                            current[s] = double.IsNegativeInfinity(sum) ? sum : sum + logProbs[row + c];
                        }
                    }

                    // ∂(-log p) / ∂logProbs[t, c] = -Σ over the states s of class c of α(s)·β(s) / (p · y(c)), both α and β holding
                    // y(c); the sums run in linear space scaled by the step's largest α·β, one exponential a state.
                    var a = alpha.AsSpan(t * states, states);
                    double top = double.NegativeInfinity;
                    for (int s = 0; s < states; s++)
                    {
                        top = Math.Max(top, a[s] + current[s]);
                    }

                    Array.Clear(sums);
                    if (!double.IsNegativeInfinity(top))
                    {
                        for (int s = 0; s < states; s++)
                        {
                            sums[slot[s]] += Math.Exp(a[s] + current[s] - top);
                        }
                    }

                    for (int k = 0; k < sums.Length; k++)
                    {
                        int at = At(t, n, distinct[k]);
                        gradient[at] -= scale * (float)(sums[k] * Math.Exp(top + nll - logProbs[at]));
                    }
                }
            }
            finally
            {
                ArrayPool<double>.Shared.Return(alpha);
                ArrayPool<double>.Shared.Return(beta);
            }
        }
    }
}
