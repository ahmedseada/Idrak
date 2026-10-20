// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

/// <summary>
/// Sampling and length settings for <c>TextGenerator</c>. Names and defaults follow the options of common
/// local LLM servers (temperature, top_k, top_p, min_p, repeat_penalty, repeat_last_n, presence_penalty,
/// frequency_penalty, seed, num_ctx, num_predict, stop), so requests written for them map one to one.
/// </summary>
public sealed record GenerationOptions
{
    /// <summary>Softmax temperature: below 1 is more predictable, above 1 more varied.</summary>
    public float Temperature { get; init; } = 0.8f;

    /// <summary>Sample only among the k most likely tokens (0 = all).</summary>
    public int TopK { get; init; } = 40;

    /// <summary>Nucleus sampling: the smallest set of likely tokens holding this probability (1 = off).</summary>
    public float TopP { get; init; } = 0.9f;

    /// <summary>Drop tokens less than this fraction as likely as the best token (0 = off).</summary>
    public float MinP { get; init; }

    /// <summary>Penalty for tokens seen in the last <see cref="RepeatLastN"/> tokens (1 = off).</summary>
    public float RepeatPenalty { get; init; } = 1.1f;

    /// <summary>Window of recent tokens (prompt included) that the penalties look at (0 = off).</summary>
    public int RepeatLastN { get; init; } = 64;

    /// <summary>Subtracted from the score of every token in the window.</summary>
    public float PresencePenalty { get; init; }

    /// <summary>Subtracted from a token's score once per occurrence in the window.</summary>
    public float FrequencyPenalty { get; init; }

    /// <summary>Random seed for reproducible output (null = random).</summary>
    public int? Seed { get; init; }

    /// <summary>Context window in tokens; the model's own context length caps it.</summary>
    public int NumCtx { get; init; } = 2048;

    /// <summary>Maximum tokens to generate; -1 means until a stop sequence or the context window is full.</summary>
    public int NumPredict { get; init; } = -1;

    /// <summary>Generation ends when the text contains any of these; the stop text itself is not returned.</summary>
    public IReadOnlyList<string> Stop { get; init; } = [];

    /// <summary>Incremental decoding with a KV cache (false recomputes the whole window every token).</summary>
    public bool UseCache { get; init; } = true;

    /// <summary>Record the decoding step once and replay it (CUDA graphs on the GPU; not for models with experts, whose routing is read back every step).</summary>
    public bool UseGraph { get; init; } = true;

    /// <summary>Tokens generated between host synchronizations while streaming.</summary>
    public int ChunkSize { get; init; } = 8;

    /// <summary>
    /// Ends the answer when its last tokens are one run of <see cref="RepeatStopMinTokens"/> to 64 tokens repeated this
    /// many times in a row (0 = off; at least 2): a model caught in a loop stops (done reason "repetition") instead of
    /// filling its token limit. Checked as the tokens reach the host (every <see cref="ChunkSize"/>).
    /// </summary>
    public int RepeatStop { get; init; }

    /// <summary>The shortest run <see cref="RepeatStop"/> counts (shorter ones, such as a row of dots or a table's rule, are text).</summary>
    public int RepeatStopMinTokens { get; init; } = 4;

    /// <summary>
    /// Beam search: 1 (the default) chooses each token with the sampler (the settings above); above 1, an encoder-decoder's
    /// generation (Idrak's <c>Seq2SeqGenerator</c>) keeps this many hypotheses at every step, as transformers'
    /// <c>num_beams</c>, and answers with the best (ranked by <see cref="LengthPenalty"/>), the sampling settings unread.
    /// Decoder-only generation samples and does not read it.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Below 1.</exception>
    public int NumBeams
    {
        get => _numBeams;
        init => _numBeams = value >= 1 ? value
            : throw new ArgumentOutOfRangeException(nameof(NumBeams), value, "The generation setting NumBeams is 1 (no beam search) or more.");
    }

    private readonly int _numBeams = 1;

    /// <summary>
    /// Beam search: a finished hypothesis's log-probability sum is divided by its length to this power before the
    /// hypotheses are ranked (1, the default: the mean; 0: the sum, favouring short answers; above 1: longer ones), as
    /// transformers' <c>length_penalty</c>.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">Not finite.</exception>
    public float LengthPenalty
    {
        get => _lengthPenalty;
        init => _lengthPenalty = float.IsFinite(value) ? value
            : throw new ArgumentOutOfRangeException(nameof(LengthPenalty), value, "The generation setting LengthPenalty is a finite number (1: the mean log-probability).");
    }

    private readonly float _lengthPenalty = 1f;

    /// <summary>
    /// Beam search: stop as soon as <see cref="NumBeams"/> hypotheses have ended (true, the default, transformers'
    /// <c>early_stopping=True</c>), or go on while a live beam could still beat the worst finished one (false).
    /// </summary>
    public bool EarlyStopping { get; init; } = true;
}
