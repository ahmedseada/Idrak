// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

/// <summary>One sampled token with the statistics recorded by <see cref="TokenSampler"/>.</summary>
/// <param name="Id">The chosen token id.</param>
/// <param name="Probability">Its probability after temperature and top-k.</param>
/// <param name="Entropy">Entropy of the sampling distribution, in bits.</param>
/// <param name="Alternatives">The five most likely tokens (id, probability), most likely first; id -1 when fewer exist.</param>
public readonly record struct SampledToken(int Id, float Probability, float Entropy, (int Id, float Probability)[] Alternatives);

/// <summary>
/// Chooses the next token of each sequence from the model's logits. <see cref="TokenSampler"/> is the built-in one
/// (temperature, top-k, top-p, min-p and penalties, on the device); implement this to plug in another strategy
/// (constrained or grammar-guided decoding, a custom sampling rule) through <c>Generation.TextGenerator.CreateSampler</c>.
/// </summary>
public interface ITokenSampler : IDisposable
{
    /// <summary>Sequences sampled per step.</summary>
    int Rows { get; }

    /// <summary>Vocabulary size.</summary>
    int Vocabulary { get; }

    /// <summary>The most recently sampled ids, [rows], on the logits' device; the next decoding step reads them from here.</summary>
    Tensor Ids { get; }

    /// <summary>
    /// True when <see cref="Sample"/> only queues work on the device (no reads back to the host), so a decoding step that
    /// samples can be recorded once as a CUDA graph and replayed. A sampler that decides on the host returns false, and
    /// generation then runs every step as it is.
    /// </summary>
    bool Recordable { get; }

    /// <summary>
    /// Samples one token per row from the last position of <paramref name="logits"/> ([rows, vocabulary] or
    /// [rows, steps, vocabulary]) into <see cref="Ids"/>, and records the step's statistics.
    /// </summary>
    void Sample(Tensor logits);

    /// <summary>Sets the recent tokens of every row (the prompt), which penalties look at. Called before sampling.</summary>
    void SetHistory(IReadOnlyList<int> tokens);

    /// <summary>Restarts step counting (and any random stream) from step 0.</summary>
    void Reset();

    /// <summary>The statistics of steps [<paramref name="fromStep"/>, <paramref name="toStep"/>) as [step][row] tokens.</summary>
    SampledToken[][] Read(int fromStep, int toStep);
}

/// <summary>What a generation asks a sampler for: where it runs, its shape, and the settings to sample with.</summary>
/// <param name="Device">The logits' device.</param>
/// <param name="Rows">Sequences sampled per step.</param>
/// <param name="Vocabulary">Vocabulary size.</param>
/// <param name="MaxSteps">Steps whose statistics are kept.</param>
/// <param name="HistoryCapacity">Recent tokens per row the penalties may look at.</param>
/// <param name="Options">Temperature, top-k, top-p, min-p, penalties and seed.</param>
public sealed record SamplerRequest(Device Device, int Rows, int Vocabulary, int MaxSteps, int HistoryCapacity, Generation.GenerationOptions Options);

/// <summary>
/// Samples the next token for a batch of sequences on the device that holds the logits, so the chosen ids can feed
/// the next decoding step without a round trip to the host. Randomness is counter-based (seed, step, row), so
/// results are reproducible and identical on CPU and GPU up to floating-point rounding. Per-token statistics are
/// written to a device buffer and read back in chunks with <see cref="Read"/>.
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

    /// <summary>Always true: sampling, penalties and statistics run on the device.</summary>
    public bool Recordable => true;

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

    /// <summary>Downloads the statistics of steps [<paramref name="fromStep"/>, <paramref name="toStep"/>) as [step][row] tokens (one synchronization).</summary>
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
