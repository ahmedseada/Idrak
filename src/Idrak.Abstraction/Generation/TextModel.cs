// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

/// <summary>A piece of streamed output. The last chunk has <see cref="Done"/> set, a reason and the statistics.</summary>
/// <param name="Text">New text since the previous chunk (may be empty).</param>
/// <param name="Done">True for the final chunk.</param>
/// <param name="DoneReason">"stop" (a stop sequence was produced) or "length" (the token limit was reached); null until done.</param>
/// <param name="Stats">Statistics, on the final chunk only.</param>
public sealed record GenerationChunk(string Text, bool Done = false, string? DoneReason = null, GenerationStats? Stats = null);

/// <summary>Anything that continues a text prompt: <c>TextGenerator</c>, or a text or chat model hosted by the inference engine.</summary>
public interface ITextModel
{
    /// <summary>Streams the continuation of <paramref name="prompt"/>; the last chunk has <see cref="GenerationChunk.Done"/> set and carries the statistics.</summary>
    IAsyncEnumerable<GenerationChunk> StreamAsync(string prompt, GenerationOptions options, CancellationToken cancellationToken = default);
}

/// <summary>Whole answers from the streaming contracts <see cref="ITextModel"/> and <see cref="IChatModel"/>.</summary>
public static class GenerationModelExtensions
{
    /// <summary>The whole continuation of <paramref name="prompt"/>, with why it ended and the statistics.</summary>
    public static async Task<(string Text, string DoneReason, GenerationStats Stats)> GenerateAsync(this ITextModel model, string prompt,
        GenerationOptions options, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        var text = new System.Text.StringBuilder();
        await foreach (var chunk in model.StreamAsync(prompt, options, cancellationToken).ConfigureAwait(false))
        {
            text.Append(chunk.Text);
            if (chunk.Done)
            {
                return (text.ToString(), chunk.DoneReason!, chunk.Stats!);
            }
        }

        throw new InvalidOperationException("Generation ended without a final chunk.");
    }

    /// <summary>The final chunk of the reply to <paramref name="request"/>: the full message, why it ended and the statistics.</summary>
    public static async Task<ChatChunk> ChatAsync(this IChatModel model, ChatRequest request, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(model);
        ChatChunk? last = null;
        await foreach (var chunk in model.StreamAsync(request, cancellationToken).ConfigureAwait(false))
        {
            last = chunk;
        }

        return last ?? throw new InvalidOperationException("The chat ended without a final chunk.");
    }
}
