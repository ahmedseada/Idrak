// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

/// <summary>Timing and token counts of one generation (durations as in common LLM server APIs).</summary>
/// <param name="PromptTokens">Tokens of the (possibly truncated) prompt that were processed.</param>
/// <param name="PromptDuration">Time to process the prompt (prefill), including the first sampled token.</param>
/// <param name="GeneratedTokens">Tokens sampled (including any that form a stop sequence).</param>
/// <param name="GenerationDuration">Time spent generating after the prompt.</param>
/// <param name="TotalDuration">Wall time of the whole call.</param>
/// <param name="ContextResets">Times the context window filled up and was re-read from its last half.</param>
public sealed record GenerationStats(int PromptTokens, TimeSpan PromptDuration, int GeneratedTokens, TimeSpan GenerationDuration,
    TimeSpan TotalDuration, int ContextResets)
{
    /// <summary>Generated tokens per second.</summary>
    public double TokensPerSecond => GenerationDuration.TotalSeconds > 0 ? GeneratedTokens / GenerationDuration.TotalSeconds : 0;
}

/// <summary>A chat request: the conversation, optional tools, reasoning mode and generation options.</summary>
/// <param name="Messages">The conversation so far (system, user, assistant and tool messages).</param>
/// <param name="Tools">Functions the model may call.</param>
/// <param name="Think">true: return reasoning separately; false: suppress it; null: model default (returned separately if produced).</param>
/// <param name="Options">Sampling and length options; the template's stop sequences are added to <see cref="GenerationOptions.Stop"/>.</param>
public sealed record ChatRequest(IReadOnlyList<ChatMessage> Messages, IReadOnlyList<ToolDefinition>? Tools = null, bool? Think = null,
    GenerationOptions? Options = null);

/// <summary>A streamed piece of the assistant's reply; the final one carries the reason, the full message and statistics.</summary>
/// <param name="Delta">What this piece added (content, reasoning, completed tool calls).</param>
/// <param name="Done">True for the final piece.</param>
/// <param name="DoneReason">"stop" or "length" on the final piece.</param>
/// <param name="Message">The complete assistant message, on the final piece.</param>
/// <param name="Stats">Statistics, on the final piece.</param>
public sealed record ChatChunk(ChatDelta Delta, bool Done = false, string? DoneReason = null, ChatMessage? Message = null, GenerationStats? Stats = null);

/// <summary>Anything that answers chat requests: <c>ChatGenerator</c>, a model hosted by the inference engine, or <c>FakeChatModel</c> in tests.</summary>
public interface IChatModel
{
    /// <summary>Streams the reply to <paramref name="request"/>; the last chunk has <see cref="ChatChunk.Done"/> set and carries the full message.</summary>
    IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, CancellationToken cancellationToken = default);
}
