// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Diagnostics;

namespace Idrak.Abstraction;

/// <summary>
/// A divergence between a student's and a teacher's next-token distributions, row by row, for the chunked divergence
/// passes (<see cref="Tensor.TokenDivergences(Tensor, Func{Tensor, Tensor}, int[], Func{int, int, Tensor}, float, int, TokenDivergenceRule?)"/>):
/// from the student's log-probabilities log p = log_softmax(u) of the softened logits u = z / T [rows, vocabulary] and
/// the teacher's probabilities q [rows, vocabulary] (same device), the divergence of each row [rows] and, when
/// <paramref name="gradient"/> is set, its gradient with respect to u [rows, vocabulary] (the caller multiplies by each
/// row's weight and 1 / T). Runs without gradients; returns new tensors.
/// </summary>
public delegate (Tensor Values, Tensor? Gradient) TokenDivergenceRule(Tensor studentLogProbabilities, Tensor teacherProbabilities, bool gradient);

// Knowledge distillation over the trained rows of a language model: the student's softened distribution against a
// teacher's, the output head run a chunk of rows at a time.
public sealed partial class Tensor
{
    /// <summary>
    /// KL(q_r ‖ softmax(head(h_r) / T)) for each listed row r of <paramref name="hidden"/> [positions, dim], where q_r is the
    /// teacher's distribution at the same temperature: <paramref name="teacher"/>(start, count) returns the probabilities
    /// of listed rows start … start + count − 1 as [count, vocabulary] on the hidden states' device. The head runs on
    /// <paramref name="chunkRows"/> rows at a time; only one value per row leaves the device. Not recorded.
    /// </summary>
    public static float[] TokenDivergences(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, Func<int, int, Tensor> teacher, float temperature, int chunkRows) =>
        Divergences(hidden, head, rows, null, teacher, temperature, chunkRows, null).Values;

    /// <summary>
    /// <see cref="TokenDivergences(Tensor, Func{Tensor, Tensor}, int[], Func{int, int, Tensor}, float, int)"/> with the
    /// divergence <paramref name="rule"/> computes (a reverse KL, a Jensen-Shannon divergence) instead of KL(q ‖ p); null:
    /// KL(q ‖ p).
    /// </summary>
    public static float[] TokenDivergences(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, Func<int, int, Tensor> teacher, float temperature, int chunkRows,
        TokenDivergenceRule? rule) =>
        Divergences(hidden, head, rows, null, teacher, temperature, chunkRows, rule).Values;

    /// <summary>
    /// Σ_r w_r · KL(q_r ‖ softmax(head(h_r) / T)) over the listed rows (see <see cref="TokenDivergences(Tensor, Func{Tensor, Tensor}, int[], Func{int, int, Tensor}, float, int)"/>), recorded: the
    /// gradient with respect to row r's logits is w_r · (softmax(z_r / T) − q_r) / T, back-propagated through the head
    /// chunk by chunk during this call (parameters inside the head receive theirs now, as with
    /// <see cref="TokenCrossEntropyRows(Tensor, Func{Tensor, Tensor}, int[], float[], float[], float, int)"/>), and the
    /// rows' gradient is scattered into <paramref name="hidden"/>'s by the result's backward pass. Back-propagate the
    /// result unscaled (scale through <paramref name="weights"/>). The [rows, vocabulary] logits never exist together.
    /// </summary>
    public static Tensor TokenDivergenceRows(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, float[] weights, Func<int, int, Tensor> teacher, float temperature,
        int chunkRows) =>
        Divergences(hidden, head, rows, weights, teacher, temperature, chunkRows, null).Loss!;

    /// <summary>
    /// <see cref="TokenDivergenceRows(Tensor, Func{Tensor, Tensor}, int[], float[], Func{int, int, Tensor}, float, int)"/>
    /// for the divergence <paramref name="rule"/> computes: row r's logits receive w_r · (the rule's gradient) / T; null:
    /// KL(q ‖ p).
    /// </summary>
    public static Tensor TokenDivergenceRows(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, float[] weights, Func<int, int, Tensor> teacher, float temperature,
        int chunkRows, TokenDivergenceRule? rule) =>
        Divergences(hidden, head, rows, weights, teacher, temperature, chunkRows, rule).Loss!;

    /// <summary>The listed rows of x [rows, dim] as [count, dim], on x's device. Not recorded.</summary>
    public static Tensor GatherRows(Tensor x, int[] rows)
    {
        x.ThrowIfDisposed();
        if (x.Rank != 2)
        {
            throw new ArgumentException($"GatherRows needs [rows, dim], got {FormatShape(x._shape)}.");
        }

        int total = x._shape[0], dim = x._shape[1];
        if (rows.Any(r => (uint)r >= (uint)total))
        {
            throw new ArgumentOutOfRangeException(nameof(rows), $"A row is outside the {total} rows.");
        }

        var y = Empty([rows.Length, dim], x.Device);
        if (rows.Length > 0)
        {
            using var index = From([.. rows.Select(r => (float)r)], [rows.Length], x.Device);
            x.Backend.Gather(x.Storage, index.Storage, y.Storage, rows.Length, dim, total);
        }

        return y;
    }

    private static (float[] Values, Tensor? Loss) Divergences(Tensor hidden, Func<Tensor, Tensor> head, int[] rows, float[]? weights, Func<int, int, Tensor> teacher,
        float temperature, int chunkRows, TokenDivergenceRule? rule)
    {
        hidden.ThrowIfDisposed();
        if (hidden.Rank != 2 || weights is not null && weights.Length != rows.Length)
        {
            throw new ArgumentException("The divergences need hidden [positions, dim] (and one weight per listed row).");
        }

        if (!(temperature > 0f) || float.IsInfinity(temperature))
        {
            throw new ArgumentOutOfRangeException(nameof(temperature), "The temperature is a positive number.");
        }

        int total = hidden._shape[0], dim = hidden._shape[1], count = rows.Length;
        foreach (int row in rows)
        {
            if ((uint)row >= (uint)total)
            {
                throw new ArgumentOutOfRangeException(nameof(rows), $"Row {row} is outside the {total} positions.");
            }
        }

        long start = Telemetry.Start(TelemetryLevel.Operations);
        var (device, backend) = (hidden.Device, hidden.Backend);
        bool record = weights is not null && WillRecord(hidden);
        var values = new float[count];
        var gradient = record && count > 0 ? Empty([count, dim], device, zeroed: true) : null;
        float inverse = 1f / temperature;
        chunkRows = Math.Max(1, chunkRows);
        for (int r0 = 0; r0 < count; r0 += chunkRows)
        {
            int n = Math.Min(chunkRows, count - r0);
            using var scope = new TensorScope();
            var index = From([.. rows.AsSpan(r0, n).ToArray().Select(r => (float)r)], [n], device);
            var chunk = Empty([n, dim], device);
            backend.Gather(hidden.Storage, index.Storage, chunk.Storage, n, dim, total);
            chunk.RequiresGrad = record;
            Tensor logits;
            using (record ? default : Autograd.NoGrad())
            {
                logits = head(chunk);
            }

            int vocabulary = logits._shape[^1];
            if (logits.Size != n * vocabulary)
            {
                throw new ArgumentException($"The head must map [{n}, {dim}] to [{n}, vocabulary], got {FormatShape(logits._shape)}.");
            }

            Tensor? logitGradient = null;
            using (Autograd.NoGrad())
            {
                var logp = (logits.Reshape(n, vocabulary) * inverse).LogSoftmax();
                var q = teacher(r0, n);
                if (q.Rank != 2 || q._shape[0] != n || q._shape[1] != vocabulary || q.Device != device)
                {
                    throw new ArgumentException($"The teacher's probabilities must be [{n}, {vocabulary}] on {device}, got {FormatShape(q._shape)} on {q.Device}.");
                }

                if (rule is null)
                {
                    // Σ_v q (log q − log p); entries with q = 0 add nothing (q · log max(q, tiny) is 0 there).
                    var divergence = (q * (q.Maximum(1e-30f).Log() - logp)).Sum(1);
                    divergence.ToArray().CopyTo(values, r0);
                    if (gradient is not null)
                    {
                        // Row r's weight w_r / T on each of its logits: an outer product with a row of ones (no broadcasting).
                        var rowWeights = From([.. weights!.AsSpan(r0, n).ToArray().Select(w => w * inverse)], [n, 1], device).MatMul(Full([1, vocabulary], 1f, device));
                        logitGradient = (logp.Exp() - q) * rowWeights;
                    }
                }
                else
                {
                    var (divergence, ruleGradient) = rule(logp, q, gradient is not null);
                    if (divergence.Size != n || gradient is not null && (ruleGradient is null || ruleGradient.Size != n * vocabulary))
                    {
                        throw new InvalidOperationException($"A divergence rule gave {divergence.Size} values (and a gradient of {ruleGradient?.Size ?? 0}) for {n} rows of {vocabulary}.");
                    }

                    divergence.ToArray().CopyTo(values, r0);
                    if (gradient is not null)
                    {
                        var rowWeights = From([.. weights!.AsSpan(r0, n).ToArray().Select(w => w * inverse)], [n, 1], device).MatMul(Full([1, vocabulary], 1f, device));
                        logitGradient = ruleGradient!.Reshape(n, vocabulary) * rowWeights;
                    }
                }
            }

            if (logitGradient is not null && logits.RequiresGrad)
            {
                logits.Backward(logitGradient.Reshape(logits._shape));
                backend.Copy2D(chunk.GradStorage(), 0, dim, gradient!.Storage, r0 * dim, dim, n, dim, accumulate: false);
            }
        }

        if (weights is null)
        {
            return (values, null);
        }

        double sum = 0;
        for (int i = 0; i < count; i++)
        {
            sum += (double)weights[i] * values[i];
        }

        var loss = From([(float)sum], [1], device);
        if (gradient is not null)
        {
            var gradientRows = From([.. rows.Select(r => (float)r)], [count], device);
            // dhidden[rows] += gradient · g, with g read on the device.
            loss.Record("token_divergence", g =>
            {
                backend.GroupScaleShift(gradient.Storage, g.Storage, null, gradient.Storage, count * dim, 1, count * dim, false);
                backend.ScatterAdd(gradient.Storage, gradientRows.Storage, hidden.GradStorage(), count, dim, total);
            }, hidden);
        }

        return (values, Traced("token_divergence", loss, start));
    }
}
