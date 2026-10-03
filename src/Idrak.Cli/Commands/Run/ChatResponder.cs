// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Generation;

namespace Idrak.Cli.Commands.Run;

/// <summary>What one answer took: the final message, every tool call run on the way, and the summed figures.</summary>
/// <param name="Message">The assistant's final message.</param>
/// <param name="DoneReason">"stop" or "length" for the final reply.</param>
/// <param name="Rounds">Model replies (1 without tool calls).</param>
/// <param name="ToolResults">The tool calls run, in order.</param>
/// <param name="PromptTokens">Prompt tokens of the last reply (the context it read).</param>
/// <param name="GeneratedTokens">Tokens generated over all rounds.</param>
/// <param name="PromptSeconds">Time spent reading prompts, over all rounds.</param>
/// <param name="GenerationSeconds">Time spent generating, over all rounds.</param>
/// <param name="ContextResets">Times the context window filled up and was re-read.</param>
internal sealed record ChatAnswer(ChatMessage Message, string? DoneReason, int Rounds, IReadOnlyList<ToolResult> ToolResults,
    int PromptTokens, int GeneratedTokens, double PromptSeconds, double GenerationSeconds, int ContextResets)
{
    /// <summary>Generated tokens per second.</summary>
    public double TokensPerSecond => GenerationSeconds > 0 ? GeneratedTokens / GenerationSeconds : 0;

    /// <summary>The figures as JSON (run --json, chat --json, batch lines).</summary>
    public JsonObject Figures() => new()
    {
        ["done_reason"] = DoneReason,
        ["prompt_tokens"] = PromptTokens,
        ["generated_tokens"] = GeneratedTokens,
        ["prompt_seconds"] = Math.Round(PromptSeconds, 4),
        ["generation_seconds"] = Math.Round(GenerationSeconds, 4),
        ["tokens_per_second"] = Math.Round(TokensPerSecond, 2),
        ["rounds"] = Rounds,
        ["tool_calls"] = ToolResults.Count,
    };

    /// <summary>One line for people: "[12 tokens, 34.5 tokens/s, prompt 20 tokens in 0.05 s]".</summary>
    public string Summary(int context) =>
        $"[{GeneratedTokens} tokens, {TokensPerSecond:F1} tokens/s, prompt {PromptTokens} tokens in {PromptSeconds:F2} s, "
        + $"context {Math.Min(PromptTokens + GeneratedTokens, context)}/{context}"
        + $"{(DoneReason == "length" ? ", stopped at the token limit" : "")}{(ContextResets > 0 ? $", context re-read {ContextResets}×" : "")}]";
}

/// <summary>
/// Answers a conversation with a chat model, streaming the reply to a writer as it is generated, and runs the tools the
/// model calls (sending their results back) for at most <see cref="MaxToolRounds"/> rounds. Shared by chat, run and agent.
/// </summary>
internal sealed class ChatResponder(IChatModel model, ToolRegistry? tools)
{
    /// <summary>The chat model.</summary>
    public IChatModel Model { get; } = model;

    /// <summary>The tools, or null.</summary>
    public ToolRegistry? Tools { get; } = tools;

    /// <summary>Tool rounds allowed per answer.</summary>
    public int MaxToolRounds { get; init; } = 8;

    /// <summary>Where the answer streams to (null: nowhere).</summary>
    public TextWriter? Stream { get; init; }

    /// <summary>Where reasoning streams to (null: not shown).</summary>
    public TextWriter? ThinkingStream { get; init; }

    /// <summary>Dims reasoning and colours tool calls with terminal escape codes.</summary>
    public bool Color { get; init; }

    /// <summary>
    /// Answers <paramref name="messages"/> (the reply, tool calls and tool results are appended to it) with the
    /// sampling of <paramref name="settings"/> and a context window of <paramref name="context"/> tokens.
    /// </summary>
    public ChatAnswer Answer(List<ChatMessage> messages, GenerationSettings settings, int context, CancellationToken cancellationToken = default)
    {
        var results = new List<ToolResult>();
        int rounds = 0, generated = 0, resets = 0;
        double promptSeconds = 0, generationSeconds = 0;
        while (true)
        {
            var request = new ChatRequest([.. messages], Tools?.Definitions, settings.Think, settings.ToOptions(context));
            ChatChunk? final = null;
            bool thinking = false, wrote = false;
            foreach (var chunk in Model.StreamAsync(request, cancellationToken).ToBlockingEnumerable(cancellationToken))
            {
                if (chunk.Delta.Thinking.Length > 0 && ThinkingStream is not null)
                {
                    if (!thinking)
                    {
                        ThinkingStream.Write(Color ? "\u001b[2m" : "(thinking) ");
                        thinking = true;
                    }

                    ThinkingStream.Write(chunk.Delta.Thinking);
                    ThinkingStream.Flush();
                }

                if (chunk.Delta.Content.Length > 0 && Stream is not null)
                {
                    EndThinking(ref thinking);
                    Stream.Write(wrote ? chunk.Delta.Content : chunk.Delta.Content.TrimStart());
                    Stream.Flush();
                    wrote = true;
                }

                if (chunk.Done)
                {
                    final = chunk;
                }
            }

            EndThinking(ref thinking);
            if (final?.Message is not { } message)
            {
                throw new InvalidOperationException("The model ended without a final message.");
            }

            if (final.Stats is { } stats)
            {
                generated += stats.GeneratedTokens;
                promptSeconds += stats.PromptDuration.TotalSeconds;
                generationSeconds += stats.GenerationDuration.TotalSeconds;
                resets += stats.ContextResets;
            }

            messages.Add(message);
            rounds++;
            if (message.ToolCalls is not { Count: > 0 } calls || Tools is null || rounds > MaxToolRounds)
            {
                if (wrote)
                {
                    Stream?.WriteLine();
                }

                return new ChatAnswer(message, final.DoneReason, rounds, results, final.Stats?.PromptTokens ?? 0, generated, promptSeconds,
                    generationSeconds, resets);
            }

            if (wrote)
            {
                Stream?.WriteLine();
            }

            foreach (var call in calls)
            {
                Stream?.WriteLine(Paint($"> {call.Name} {Shorten(call.Arguments.ToJsonString(), 160)}", "36"));
                var result = Tools.InvokeAsync(call, cancellationToken).GetAwaiter().GetResult();
                results.Add(result);
                messages.Add(result.ToMessage());
                var lines = (result.Succeeded ? result.Content : "error: " + result.Error).Split('\n');
                Stream?.WriteLine(Paint(string.Join('\n', lines.Take(6)) + (lines.Length > 6 ? $"\n... {lines.Length - 6} more lines" : ""), "2"));
            }
        }
    }

    private void EndThinking(ref bool thinking)
    {
        if (thinking)
        {
            ThinkingStream!.Write(Color ? "\u001b[0m\n" : "\n");
            thinking = false;
        }
    }

    private string Paint(string text, string code) => Color ? $"\u001b[{code}m{text}\u001b[0m" : text;

    private static string Shorten(string text, int max) => text.Length > max ? text[..max] + "..." : text;
}
