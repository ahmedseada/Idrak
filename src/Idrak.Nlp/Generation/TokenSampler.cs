// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Generation.Abstractions;

namespace Idrak.Generation;

/// <summary>
/// Samples the next token for a batch of sequences on the device that holds the logits, so the chosen ids can feed
/// the next decoding step without a round trip to the host. Randomness is counter-based (seed, step, row), so
/// results are reproducible and identical on CPU and GPU up to floating-point rounding. Per-token statistics are
/// written to a device buffer and read back in chunks with <see cref="Read"/>. Logits that are not finite follow one rule on
/// every device: NaN is never sampled, +∞ wins (shared evenly among the +∞ tokens), and a row with neither a finite logit
/// nor a +∞ one is reported by <see cref="Read"/>.
/// </summary>
public sealed class TokenSampler : ITokenSampler
{
    /// <summary>The sampler a generation uses by default: this class, set from <paramref name="request"/>'s options.</summary>
    public static ITokenSampler Create(SamplerRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        var options = request.Options;
        return new TokenSampler(request.Device, request.Rows, request.Vocabulary, request.MaxSteps, request.HistoryCapacity)
        {
            Temperature = options.Temperature,
            TopK = options.TopK,
            TopP = options.TopP,
            MinP = options.MinP,
            RepeatPenalty = options.RepeatPenalty,
            RepeatLastN = options.RepeatLastN,
            PresencePenalty = options.PresencePenalty,
            FrequencyPenalty = options.FrequencyPenalty,
            Seed = (uint)(options.Seed ?? Random.Shared.Next()),
        };
    }

    /// <summary>True unless <see cref="RejectNonFinite"/>: sampling, penalties and statistics run on the device.</summary>
    public bool Recordable => !RejectNonFinite;

    /// <summary>
    /// Refuse logits that are not finite with an error before sampling, as version 1 of the default sampler did (the
    /// sampler of Idrak 0.3 failed on them; <c>TokenSamplers.Default(TokenSamplers.DefaultName, version: 1)</c> makes one).
    /// Reads the logits on the host every step, so a step is not recorded as a graph. Off by default: NaN logits are never
    /// sampled and +∞ ones win.
    /// </summary>
    public bool RejectNonFinite { get; set; }

    private const int StatsPerToken = 13;
    private readonly Tensor _stats;
    private readonly Tensor _step;
    private readonly Tensor _history;
    private readonly Tensor _historyLength;
    private readonly Tensor _work;

    /// <summary>Creates a sampler for <paramref name="rows"/> sequences over <paramref name="vocabulary"/> tokens, for up to <paramref name="maxSteps"/> steps.</summary>
    /// <param name="device">Where sampling runs (the logits' device).</param>
    /// <param name="rows">Sequences sampled per step.</param>
    /// <param name="vocabulary">Vocabulary size.</param>
    /// <param name="maxSteps">Capacity of the per-token statistics buffer.</param>
    /// <param name="historyCapacity">Tokens remembered per row for repetition penalties (the largest usable <see cref="RepeatLastN"/>).</param>
    public TokenSampler(Device device, int rows, int vocabulary, int maxSteps, int historyCapacity = 256)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(historyCapacity);
        Rows = rows;
        Vocabulary = vocabulary;
        MaxSteps = maxSteps;
        HistoryCapacity = historyCapacity;
        Ids = Tensor.Persistent(new float[rows], [rows], device, requiresGrad: false);
        _stats = Tensor.Persistent(new float[maxSteps * rows * StatsPerToken], [maxSteps, rows, StatsPerToken], device, requiresGrad: false);
        _step = Tensor.Persistent([0f], [1], device, requiresGrad: false);
        _history = Tensor.Persistent(new float[rows * historyCapacity], [rows, historyCapacity], device, requiresGrad: false);
        _historyLength = Tensor.Persistent([0f], [1], device, requiresGrad: false);
        _work = Tensor.Persistent(new float[rows * vocabulary], [rows, vocabulary], device, requiresGrad: false);
    }

    /// <summary>Tokens remembered per row for repetition penalties.</summary>
    public int HistoryCapacity { get; }

    /// <summary>Nucleus sampling: keep the smallest set of most likely tokens holding this share of the probability (1 = off).</summary>
    public float TopP { get; set; } = 1f;

    /// <summary>Keep only tokens at least this fraction as likely as the most likely one (0 = off).</summary>
    public float MinP { get; set; }

    /// <summary>Divides positive (multiplies negative) scores of tokens seen in the last <see cref="RepeatLastN"/> tokens (1 = off).</summary>
    public float RepeatPenalty { get; set; } = 1f;

    /// <summary>How many recent tokens the penalties look at (at most <see cref="HistoryCapacity"/>; 0 disables penalties).</summary>
    public int RepeatLastN { get; set; } = 64;

    /// <summary>Subtracted from the score of every token present in the recent window (0 = off).</summary>
    public float PresencePenalty { get; set; }

    /// <summary>Subtracted from a token's score once per occurrence in the recent window (0 = off).</summary>
    public float FrequencyPenalty { get; set; }

    /// <summary>
    /// Sets the recent-token history of every row (e.g. the prompt), which the repetition penalties look at, and which each
    /// sampled token then extends. Call before sampling (and not while recording a graph).
    /// </summary>
    public void SetHistory(IReadOnlyList<int> tokens)
    {
        var values = new float[Rows * HistoryCapacity];
        for (int t = Math.Max(0, tokens.Count - HistoryCapacity); t < tokens.Count; t++)
        {
            for (int r = 0; r < Rows; r++)
            {
                values[r * HistoryCapacity + t % HistoryCapacity] = tokens[t];
            }
        }

        _history.Load(values);
        _historyLength.FillInPlace(tokens.Count);
    }

    /// <summary>Sequences sampled per step.</summary>
    public int Rows { get; }

    /// <summary>Vocabulary size.</summary>
    public int Vocabulary { get; }

    /// <summary>Capacity of the statistics buffer.</summary>
    public int MaxSteps { get; }

    /// <summary>The most recently sampled ids, [rows] (device); reshape to [rows, 1] to feed the next step.</summary>
    public Tensor Ids { get; }

    /// <summary>Softmax temperature (fixed when a step using this sampler is recorded as a graph).</summary>
    public float Temperature { get; set; } = 1f;

    /// <summary>Restrict sampling to the k most likely tokens (0 = all).</summary>
    public int TopK { get; set; }

    /// <summary>Random seed.</summary>
    public uint Seed { get; set; }

    /// <summary>
    /// Samples one token per row from the last position of <paramref name="logits"/> ([rows, vocabulary] or
    /// [rows, steps, vocabulary]), stores ids and statistics, and advances the device-side step counter.
    /// </summary>
    public void Sample(Tensor logits)
    {
        int vocabulary = logits.Shape[^1];
        int steps = logits.Rank == 3 ? logits.Shape[1] : 1;
        if (vocabulary != Vocabulary || logits.Shape[0] != Rows)
        {
            throw new ArgumentException($"Expected [{Rows}, ..., {Vocabulary}] logits, got {Tensor.FormatShape(logits.Shape)}.");
        }

        if (RejectNonFinite)
        {
            var values = logits.ToArray();
            int bad = Array.FindIndex(values, v => !float.IsFinite(v));
            if (bad >= 0)
            {
                throw new ArgumentException($"Logit {bad % vocabulary} of row {bad / (steps * vocabulary)} is {values[bad]}: this sampler (version 1) "
                    + "refuses logits that are not finite. The default sampler (version 2) never samples NaN and lets +∞ win.", nameof(logits));
            }
        }

        var backend = logits.Backend;
        var source = logits.Storage;
        int rowStride = steps * vocabulary, rowOffset = (steps - 1) * vocabulary;
        int lastN = Math.Min(RepeatLastN, HistoryCapacity);
        if (lastN > 0 && (RepeatPenalty != 1f || PresencePenalty != 0f || FrequencyPenalty != 0f))
        {
            backend.PenalizeRows(source, _work.Storage, _history.Storage, _historyLength.Storage, Rows, Vocabulary,
                rowStride, rowOffset, HistoryCapacity, lastN, RepeatPenalty, PresencePenalty, FrequencyPenalty);
            source = _work.Storage;
            rowStride = vocabulary;
            rowOffset = 0;
        }

        backend.SampleRows(source, Ids.Storage, _stats.Storage, _step.Storage, Rows, Vocabulary,
            rowStride, rowOffset, Temperature, TopK, TopP, MinP, Seed);
        backend.HistoryPush(Ids.Storage, _history.Storage, _historyLength.Storage, Rows, HistoryCapacity);
        _historyLength.AddInPlace(1f);
        _step.AddInPlace(1f);
    }

    /// <summary>Restarts step counting (and the random stream) from step 0 and clears the penalty history.</summary>
    public void Reset()
    {
        _step.FillInPlace(0f);
        _historyLength.FillInPlace(0f);
    }

    /// <summary>
    /// Downloads the statistics of steps [<paramref name="fromStep"/>, <paramref name="toStep"/>) as [step][row] tokens (one
    /// synchronization).
    /// </summary>
    /// <exception cref="InvalidOperationException">A row had no logit to sample (all NaN or -∞; see <c>Backend.SampleRowsKernel</c>).</exception>
    public SampledToken[][] Read(int fromStep, int toStep)
    {
        int count = toStep - fromStep;
        var raw = new float[count * Rows * StatsPerToken];
        _stats.CopyTo(raw, fromStep * Rows * StatsPerToken);
        var result = new SampledToken[count][];
        for (int s = 0; s < count; s++)
        {
            result[s] = new SampledToken[Rows];
            for (int r = 0; r < Rows; r++)
            {
                int o = (s * Rows + r) * StatsPerToken;
                var alternatives = new (int, float)[5];
                for (int a = 0; a < 5; a++)
                {
                    alternatives[a] = ((int)raw[o + 3 + 2 * a], raw[o + 4 + 2 * a]);
                }

                if (raw[o] < 0)
                {
                    throw new InvalidOperationException($"Nothing to sample at step {fromStep + s}, row {r}: every logit was NaN or -∞, so the "
                        + "model's output is not finite (an overflow in its weights or activations, or a bad input). NaN logits are never "
                        + "sampled and +∞ ones win; a row needs at least one of either.");
                }

                result[s][r] = new SampledToken((int)raw[o], raw[o + 1], raw[o + 2], alternatives);
            }
        }

        return result;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Ids.Dispose();
        _stats.Dispose();
        _step.Dispose();
        _history.Dispose();
        _historyLength.Dispose();
        _work.Dispose();
    }
}
