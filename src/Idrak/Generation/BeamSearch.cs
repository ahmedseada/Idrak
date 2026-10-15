// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Generation;

/// <summary>The settings of a <see cref="BeamSearch"/>.</summary>
public sealed record BeamOptions
{
    /// <summary>Beams kept at every step (1: greedy, the most likely token each step).</summary>
    public int Beams { get; init; } = 4;

    /// <summary>
    /// The exponent a finished hypothesis's log-probability sum is divided by its length to the power of (1: the mean
    /// log-probability; 0: the sum, favouring short answers; above 1: longer ones), as transformers' length_penalty.
    /// </summary>
    public float LengthPenalty { get; init; } = 1f;

    /// <summary>New tokens at most.</summary>
    public int MaxTokens { get; init; } = 448;

    /// <summary>New tokens at least: the end token is not chosen before this many (transformers' min_new_tokens).</summary>
    public int MinTokens { get; init; }

    /// <summary>
    /// Stop as soon as <see cref="Beams"/> hypotheses have ended (true, transformers' early_stopping); otherwise go on
    /// while a live beam could still beat the worst finished one.
    /// </summary>
    public bool EarlyStopping { get; init; } = true;

    /// <summary>An n-gram size no sequence may repeat (0: none), over the prompt and the generated tokens.</summary>
    public int NoRepeatNgram { get; init; }

    /// <summary>The token that ends a hypothesis, or -1 for none (every beam runs to <see cref="MaxTokens"/>).</summary>
    public int EndToken { get; init; } = -1;

    /// <summary>Tokens never chosen, at any step.</summary>
    public IReadOnlyList<int> SuppressTokens { get; init; } = [];

    /// <summary>Tokens never chosen at the first step.</summary>
    public IReadOnlyList<int> BeginSuppressTokens { get; init; } = [];
}

/// <summary>
/// Changes one row's logits before the next token is chosen (a family's rules, such as which tokens may follow which):
/// <paramref name="step"/> counts the generated tokens so far (0 at the first), <paramref name="generated"/> are the
/// row's generated tokens, <paramref name="logits"/> its scores over the vocabulary (set one to -∞ to forbid it).
/// </summary>
public delegate void LogitsFilter(int step, ReadOnlySpan<int> generated, Span<float> logits);

/// <summary>One answer of a <see cref="BeamSearch"/>.</summary>
/// <param name="Tokens">The generated tokens (the end token, when reached, not among them).</param>
/// <param name="Score">The log-probability sum divided by the length to the power of <see cref="BeamOptions.LengthPenalty"/> (the hypotheses are ranked by it).</param>
/// <param name="LogProb">The log-probability sum of the tokens (the end token's included when reached).</param>
public sealed record BeamHypothesis(IReadOnlyList<int> Tokens, float Score, float LogProb);

/// <summary>
/// Beam search over any decoder that runs step by step on a <see cref="DecodingContext"/> (a text decoder, an
/// encoder-decoder's decoder with its memory set): the <see cref="BeamOptions.Beams"/> most likely continuations are kept
/// at every step as transformers' beam search keeps them (each step's 2·beams best candidates over beams × vocabulary,
/// a hypothesis ending when the end token is among the first beams of them, the search done by
/// <see cref="BeamOptions.EarlyStopping"/>), the key/value caches' rows reordered to the beams kept
/// (<see cref="DecodingContext.ReorderRows"/>, any cache format). The logits of a step are read back once (beams ×
/// vocabulary floats) for the candidates and the filters. A beam of 1 is greedy decoding.
/// </summary>
/// <example>
/// <code>
/// using var context = new DecodingContext(device, batch: 4, capacity: 448) { LastPositionOnly = true };
/// foreach (var cross in decoder.Descendants().OfType&lt;CrossAttention&gt;()) cross.SetMemory(encoded, context);
/// var best = BeamSearch.Decode((ids, ctx) => decoder.ForwardCached(ids, ctx), context, prompt, new BeamOptions { Beams = 4, EndToken = eos })[0];
/// </code>
/// </example>
public static class BeamSearch
{
    /// <summary>
    /// Decodes from <paramref name="prompt"/> with <paramref name="step"/> (the decoder's cached pass: ids [beams, steps] →
    /// logits [beams, 1, vocabulary] or [beams, vocabulary]) on <paramref name="context"/>, whose batch is the beams; the
    /// context is reset first. The hypotheses, best first (as many as finished or live, at most the beams).
    /// </summary>
    /// <param name="step">The decoder's step.</param>
    /// <param name="context">The decoding context (batch = <see cref="BeamOptions.Beams"/>; a memory set on it stays).</param>
    /// <param name="prompt">The tokens every beam starts from (at least one).</param>
    /// <param name="options">The search's settings.</param>
    /// <param name="filter">A rule applied to each row's logits before the choice, or null.</param>
    /// <param name="cancellationToken">Stops the search; what was decoded so far is returned.</param>
    /// <exception cref="ArgumentException">The context's batch is not the beams, or the prompt is empty.</exception>
    public static IReadOnlyList<BeamHypothesis> Decode(Func<Tensor, DecodingContext, Tensor> step, DecodingContext context, IReadOnlyList<int> prompt, BeamOptions options,
        LogitsFilter? filter = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(step);
        ArgumentNullException.ThrowIfNull(context);
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(options);
        int beams = options.Beams;
        if (beams <= 0 || context.Batch != beams)
        {
            throw new ArgumentException($"The context decodes {context.Batch} rows; the search keeps {beams} beams (make the context with batch = Beams).");
        }

        if (prompt.Count == 0)
        {
            throw new ArgumentException("The prompt holds at least one token.", nameof(prompt));
        }

        if (options.MaxTokens <= 0)
        {
            return [];
        }

        var device = context.Device;
        var sequences = new List<int>[beams];
        var scores = new double[beams];
        for (int b = 0; b < beams; b++)
        {
            sequences[b] = new List<int>(options.MaxTokens);
            scores[b] = b == 0 ? 0 : double.NegativeInfinity;                  // one live beam at the start (transformers' -1e9)
        }

        var finished = new List<(List<int> Tokens, double Score, double LogProb)>();
        var next = new (int Beam, int Token, double Score)[beams];
        var sources = new int[beams];
        var ids = new float[beams];
        context.Reset();

        // The prompt on every row (the logits of its last position).
        var promptIds = new float[beams * prompt.Count];
        for (int b = 0; b < beams; b++)
        {
            for (int i = 0; i < prompt.Count; i++)
            {
                promptIds[b * prompt.Count + i] = prompt[i];
            }
        }

        int vocabulary = 0;
        float[]? logits = null;
        try
        {
            long prefillStart = Telemetry.Start(TelemetryLevel.Inference);
            logits = Logits(step, context, promptIds, [beams, prompt.Count], beams, ref vocabulary);
            Telemetry.InferenceStage(InferenceStageKind.Prefill, device, prefillStart, prompt.Count);
            long decodeStart = Telemetry.Start(TelemetryLevel.Inference);
            int produced = 0;
            bool done = false;
            for (int s = 0; s < options.MaxTokens && !done && !cancellationToken.IsCancellationRequested; s++)
            {
                // Each live beam's candidates (its score plus each token's log-probability), the best 2·beams of all.
                var candidates = new List<(int Beam, int Token, double Score)>(2 * beams * beams);
                for (int b = 0; b < beams; b++)
                {
                    if (double.IsNegativeInfinity(scores[b]))
                    {
                        continue;
                    }

                    var row = logits.AsSpan(b * vocabulary, vocabulary);
                    Filter(row, s, prompt, sequences[b], options, filter);
                    TopCandidates(row, b, scores[b], 2 * beams, candidates);
                }

                candidates.Sort((x, y) => y.Score != x.Score ? y.Score.CompareTo(x.Score) : x.Beam != y.Beam ? x.Beam.CompareTo(y.Beam) : x.Token.CompareTo(y.Token));
                int kept = 0;
                for (int rank = 0; rank < candidates.Count && kept < beams; rank++)
                {
                    var (beam, token, score) = candidates[rank];
                    if (token == options.EndToken)
                    {
                        if (rank < beams)                                      // a hypothesis ends only from the first beams
                        {
                            AddFinished(finished, [.. sequences[beam]], score, s + 1, options.LengthPenalty, beams);
                        }

                        continue;
                    }

                    next[kept++] = (beam, token, score);
                }

                if (kept == 0)
                {
                    done = true;                                                   // every candidate ended
                    break;
                }

                produced = s + 1;
                done = finished.Count >= beams && (options.EarlyStopping || Finished(finished, candidates[0].Score, s + 1, options.LengthPenalty));
                if (done || s + 1 == options.MaxTokens)
                {
                    // The live beams as they stand (transformers adds them at the end when the search is not done).
                    if (!done)
                    {
                        for (int k = 0; k < kept; k++)
                        {
                            var (beam, token, score) = next[k];
                            AddFinished(finished, new List<int>(sequences[beam]) { token }, score, s + 1, options.LengthPenalty, beams);
                        }
                    }

                    break;
                }

                // The beams kept: their sequences, scores and cache rows (a source row may be taken by several).
                var previous = (List<int>[])sequences.Clone();
                for (int k = 0; k < beams; k++)
                {
                    var (beam, token, score) = k < kept ? next[k] : (next[0].Beam, next[0].Token, double.NegativeInfinity);
                    sources[k] = beam;
                    sequences[k] = new List<int>(previous[beam]) { token };
                    scores[k] = score;
                    ids[k] = token;
                }

                context.ReorderRows(sources);
                logits = Logits(step, context, ids, [beams, 1], beams, ref vocabulary);
            }

            Telemetry.InferenceStage(InferenceStageKind.DecodeChunk, device, decodeStart, produced);
        }
        finally
        {
            if (logits is not null)
            {
                ArrayPool<float>.Shared.Return(logits);
            }
        }

        if (finished.Count == 0)
        {
            // Stopped before any step: the live beams as they are.
            for (int b = 0; b < beams; b++)
            {
                if (!double.IsNegativeInfinity(scores[b]) && sequences[b].Count > 0)
                {
                    AddFinished(finished, sequences[b], scores[b], sequences[b].Count, options.LengthPenalty, beams);
                }
            }
        }

        finished.Sort((x, y) => y.Score.CompareTo(x.Score));
        return [.. finished.Select(f => new BeamHypothesis(f.Tokens, (float)f.Score, (float)f.LogProb))];
    }

    // The step's logits of every row read back into a pooled buffer (returned by the caller), the step's tensors freed.
    private static float[] Logits(Func<Tensor, DecodingContext, Tensor> step, DecodingContext context, float[] ids, int[] shape, int beams, ref int vocabulary)
    {
        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        using var input = Tensor.From(ids, shape, context.Device);
        var output = step(input, context);
        int v = output.Shape[^1];
        if (output.Size != beams * v)
        {
            throw new InvalidOperationException($"The step gave {Tensor.FormatShape(output.Shape)}; the search reads one row of logits per beam ([{beams}, vocabulary]).");
        }

        vocabulary = v;
        var values = ArrayPool<float>.Shared.Rent(beams * v);
        try
        {
            output.CopyTo(values.AsSpan(0, beams * v));
            return values;
        }
        catch
        {
            ArrayPool<float>.Shared.Return(values);
            throw;
        }
    }

    // The suppressions, the least length, the n-gram rule and the caller's filter on one row, then log-softmax in place.
    private static void Filter(Span<float> row, int step, IReadOnlyList<int> prompt, List<int> generated, BeamOptions options, LogitsFilter? filter)
    {
        foreach (int token in options.SuppressTokens)
        {
            if ((uint)token < (uint)row.Length)
            {
                row[token] = float.NegativeInfinity;
            }
        }

        if (step == 0)
        {
            foreach (int token in options.BeginSuppressTokens)
            {
                if ((uint)token < (uint)row.Length)
                {
                    row[token] = float.NegativeInfinity;
                }
            }
        }

        if (step < options.MinTokens && (uint)options.EndToken < (uint)row.Length)
        {
            row[options.EndToken] = float.NegativeInfinity;
        }

        if (options.NoRepeatNgram > 0)
        {
            BanRepeatedNgrams(row, prompt, generated, options.NoRepeatNgram);
        }

        filter?.Invoke(step, System.Runtime.InteropServices.CollectionsMarshal.AsSpan(generated), row);

        // log-softmax: the max and the sum of exponentials in double (the ranking of close scores holds on every device).
        float max = float.NegativeInfinity;
        foreach (float x in row)
        {
            if (x > max)
            {
                max = x;
            }
        }

        if (float.IsNegativeInfinity(max))
        {
            return;                                                             // every token forbidden: no candidate from this row
        }

        double sum = 0;
        foreach (float x in row)
        {
            if (!float.IsNegativeInfinity(x))
            {
                sum += Math.Exp(x - max);
            }
        }

        float shift = (float)(max + Math.Log(sum));
        for (int i = 0; i < row.Length; i++)
        {
            row[i] -= shift;
        }
    }

    // Tokens that would complete an n-gram the sequence (prompt and generated) already holds.
    private static void BanRepeatedNgrams(Span<float> row, IReadOnlyList<int> prompt, List<int> generated, int n)
    {
        int length = prompt.Count + generated.Count;
        if (length < n)
        {
            return;
        }

        int At(int i) => i < prompt.Count ? prompt[i] : generated[i - prompt.Count];
        for (int start = 0; start + n <= length; start++)
        {
            bool same = true;
            for (int j = 0; j < n - 1 && same; j++)
            {
                same = At(start + j) == At(length - (n - 1) + j);
            }

            if (same && (uint)At(start + n - 1) < (uint)row.Length)
            {
                row[At(start + n - 1)] = float.NegativeInfinity;
            }
        }
    }

    // The row's `count` best tokens (score + log-probability), added to the candidates: a small ordered list, one pass.
    private static void TopCandidates(ReadOnlySpan<float> row, int beam, double score, int count, List<(int Beam, int Token, double Score)> candidates)
    {
        Span<int> tokens = count <= 64 ? stackalloc int[count] : new int[count];
        Span<float> values = count <= 64 ? stackalloc float[count] : new float[count];
        int held = 0;
        for (int t = 0; t < row.Length; t++)
        {
            float v = row[t];
            if (float.IsNegativeInfinity(v) || float.IsNaN(v) || held == count && v <= values[held - 1])
            {
                continue;
            }

            // Insert in descending order (ties keep the earlier token first).
            int at = held < count ? held : held - 1;
            while (at > 0 && values[at - 1] < v)
            {
                values[at] = values[at - 1];
                tokens[at] = tokens[at - 1];
                at--;
            }

            values[at] = v;
            tokens[at] = t;
            if (held < count)
            {
                held++;
            }
        }

        for (int i = 0; i < held; i++)
        {
            candidates.Add((beam, tokens[i], score + values[i]));
        }
    }

    // A finished hypothesis, ranked by its length-normalized score; at most `beams` are kept (the worst dropped).
    private static void AddFinished(List<(List<int> Tokens, double Score, double LogProb)> finished, List<int> tokens, double logProb, int length, float penalty, int beams)
    {
        double score = logProb / Math.Pow(Math.Max(length, 1), penalty);
        if (finished.Count >= beams)
        {
            int worst = 0;
            for (int i = 1; i < finished.Count; i++)
            {
                if (finished[i].Score < finished[worst].Score)
                {
                    worst = i;
                }
            }

            if (score <= finished[worst].Score)
            {
                return;
            }

            finished.RemoveAt(worst);
        }

        finished.Add((tokens, score, logProb));
    }

    // Without early stopping: done when no live beam (the best candidate's score, at the length it would have) can beat
    // the worst finished hypothesis (transformers' BeamHypotheses.is_done).
    private static bool Finished(List<(List<int> Tokens, double Score, double LogProb)> finished, double bestLogProb, int length, float penalty)
    {
        double highest = penalty > 0 ? bestLogProb / Math.Pow(length, penalty) : bestLogProb;
        double worst = double.PositiveInfinity;
        foreach (var f in finished)
        {
            worst = Math.Min(worst, f.Score);
        }

        return worst >= highest;
    }
}
