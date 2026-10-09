// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Testing;

// The cases of sequence losses: CTC against every alignment counted one by one.
public static partial class DeviceCases
{
    private static void SequenceLosses(DeviceCaseContext c)
    {
        var b = c.Backend;

        // Small enough to list every alignment (classes^steps paths): the loss and its gradient against plain loops, in both
        // layouts, with an empty label, a repeated label (which needs a blank between) and a sequence no alignment fits.
        foreach (bool batchFirst in new[] { false, true })
        {
            int steps = 5, classes = 4, blank = batchFirst ? 3 : 0;
            int[][] labels = [[1, 2], [2, 2], [], [1, 2, 1], [1, 1, 1]];
            int[] inputLengths = [5, 4, 3, 5, 4];                           // the last needs 5 steps for 1, -, 1, -, 1: impossible in 4
            if (blank != 0)
            {
                labels = [.. labels.Select(l => l.Select(v => v - 1).ToArray())];
            }

            int batch = labels.Length;
            var (targets, offsets) = Concatenated(labels);
            int[] lengths = [.. labels.Select(l => l.Length)];
            var logProbs = LogSoftmaxRows(c.Values(steps * batch * classes, 2f), classes);
            Storage lp = c.Storage(logProbs), tv = c.Storage(targets), losses = c.Zeros(batch);
            foreach (bool zeroInfinity in new[] { false, true })
            {
                b.CtcLoss(lp, tv, losses, inputLengths, lengths, offsets, steps, batch, classes, blank, batchFirst, zeroInfinity);
                var expected = new float[batch];
                var expectedGrad = new float[logProbs.Length];
                var scales = c.Values(batch, 2f);
                for (int n = 0; n < batch; n++)
                {
                    var (nll, grad) = CtcByAlignments(logProbs, labels[n], inputLengths[n], n, steps, batch, classes, blank, batchFirst);
                    bool impossible = double.IsPositiveInfinity(nll);
                    expected[n] = impossible && zeroInfinity ? 0f : (float)nll;
                    for (int i = 0; i < grad.Length && !impossible; i++)
                    {
                        expectedGrad[i] += scales[n] * grad[i];
                    }
                }

                var actual = Read(losses);
                c.Expect(float.IsPositiveInfinity(actual[^1]) == !zeroInfinity, $"CTC: an impossible alignment gives {actual[^1]}");
                c.ExpectClose(expected[..^1], actual[..^1], 1e-4f, $"CTC losses (batch first {batchFirst})");
                if (zeroInfinity)
                {
                    c.ExpectClose([0f], actual[^1..], 0f, "CTC loss of an impossible alignment with zero infinity");
                    var grad = c.Zeros(logProbs.Length);
                    b.CtcLossBackward(lp, tv, c.Storage(scales), grad, inputLengths, lengths, offsets, steps, batch, classes, blank, batchFirst, zeroInfinity);
                    c.ExpectClose(expectedGrad, Read(grad), 1e-4f, $"CTC gradient (batch first {batchFirst})");
                }
            }
        }

        // Random sizes: longer sequences and many classes (compared between devices).
        {
            int steps = c.Size(1, 40), batch = c.Size(1, 6), classes = c.Size(2, 40), blank = c.Random.Next(classes);
            var labels = Enumerable.Range(0, batch).Select(_ => Enumerable.Range(0, c.Random.Next(0, Math.Max(1, steps / 2)))
                .Select(_ => (c.Random.Next(1, classes) + blank) % classes).ToArray()).ToArray();
            int[] inputLengths = [.. labels.Select(_ => c.Random.Next(0, steps + 1))];
            var (targets, offsets) = Concatenated(labels);
            int[] lengths = [.. labels.Select(l => l.Length)];
            Storage lp = c.Storage(LogSoftmaxRows(c.Values(steps * batch * classes, 3f), classes)), tv = c.Storage(targets);
            bool batchFirst = c.Random.Next(2) == 0;
            b.CtcLoss(lp, tv, c.Zeros(batch), inputLengths, lengths, offsets, steps, batch, classes, blank, batchFirst, zeroInfinity: true);
            b.CtcLossBackward(lp, tv, Random(c, batch), Random(c, steps * batch * classes), inputLengths, lengths, offsets, steps, batch, classes, blank, batchFirst, zeroInfinity: true);
        }
    }

    private static (float[] Targets, int[] Offsets) Concatenated(int[][] labels)
    {
        var offsets = new int[labels.Length];
        for (int n = 1; n < labels.Length; n++)
        {
            offsets[n] = offsets[n - 1] + labels[n - 1].Length;
        }

        return ([.. labels.SelectMany(l => l).Select(v => (float)v)], offsets);
    }

    private static float[] LogSoftmaxRows(float[] values, int classes)
    {
        for (int row = 0; row < values.Length / classes; row++)
        {
            var span = values.AsSpan(row * classes, classes);
            double max = span.ToArray().Max(), sum = 0;
            foreach (float v in span)
            {
                sum += Math.Exp(v - max);
            }

            for (int i = 0; i < span.Length; i++)
            {
                span[i] = (float)(span[i] - max - Math.Log(sum));
            }
        }

        return values;
    }

    // -log p(label) by visiting every path of the first `length` steps, and its gradient: -Σ over the paths that read as the
    // label and pass class k at step t of p(path) / p(label).
    private static (double Nll, float[] Gradient) CtcByAlignments(float[] logProbs, int[] label, int length, int n, int steps, int batch, int classes, int blank, bool batchFirst)
    {
        int At(int t, int k) => (batchFirst ? n * steps + t : t * batch + n) * classes + k;
        var gradient = new float[logProbs.Length];
        var weights = new double[logProbs.Length];
        var path = new int[length];
        double total = 0;
        long count = (long)Math.Pow(classes, length);
        for (long code = 0; code < count; code++)
        {
            long rest = code;
            for (int t = 0; t < length; t++, rest /= classes)
            {
                path[t] = (int)(rest % classes);
            }

            var read = new List<int>();
            for (int t = 0; t < length; t++)
            {
                if (path[t] != blank && (t == 0 || path[t] != path[t - 1]))
                {
                    read.Add(path[t]);
                }
            }

            if (!read.SequenceEqual(label))
            {
                continue;
            }

            double logP = 0;
            for (int t = 0; t < length; t++)
            {
                logP += logProbs[At(t, path[t])];
            }

            double p = Math.Exp(logP);
            total += p;
            for (int t = 0; t < length; t++)
            {
                weights[At(t, path[t])] += p;
            }
        }

        for (int i = 0; i < gradient.Length; i++)
        {
            gradient[i] = total > 0 ? (float)(-weights[i] / total) : 0f;
        }

        return (total > 0 ? -Math.Log(total) : double.PositiveInfinity, gradient);
    }
}
