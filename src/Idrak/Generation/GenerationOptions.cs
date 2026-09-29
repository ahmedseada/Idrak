namespace Idrak.Generation;

/// <summary>
/// Sampling and length settings for <see cref="TextGenerator"/>. Names and defaults follow the options of common
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

    /// <summary>Maximum tokens to generate; -1 means until a stop sequence or <see cref="TextGenerator.MaxTokens"/>.</summary>
    public int NumPredict { get; init; } = -1;

    /// <summary>Generation ends when the text contains any of these; the stop text itself is not returned.</summary>
    public IReadOnlyList<string> Stop { get; init; } = [];

    /// <summary>Incremental decoding with a KV cache (false recomputes the whole window every token).</summary>
    public bool UseCache { get; init; } = true;

    /// <summary>Record the decoding step once and replay it (CUDA graphs on the GPU).</summary>
    public bool UseGraph { get; init; } = true;

    /// <summary>Tokens generated between host synchronizations while streaming.</summary>
    public int ChunkSize { get; init; } = 8;
}
