// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A chat model that comes with tools the server may run for it, such as an inference engine chat model configured
/// with tools. Hosts look for it to run tool calls on the server (<see cref="ChatTools.WithTools"/>):
/// <c>engine.Model&lt;IToolChatModel&gt;("my-gpt").Tools</c>.
/// </summary>
public interface IToolChatModel : IChatModel
{
    /// <summary>The model's tools, or null when it has none.</summary>
    IToolRegistry? Tools { get; }
}

/// <summary>Server-side tool execution for any <see cref="IChatModel"/>.</summary>
public static class ChatTools
{
    /// <summary>
    /// <paramref name="model"/> with its tool calls run on the server. Each request is sent with the definitions of
    /// <paramref name="tools"/> (in place of the request's own); when a reply asks for tools, they run, their results go
    /// back to the model and it is asked again, for at most <paramref name="maxToolRounds"/> rounds. The stream carries
    /// the content and thinking of every round; the final chunk holds the last reply, its statistics, and tool calls
    /// only when the rounds ran out while the model still asked for tools. The request's messages are not changed.
    /// </summary>
    /// <param name="model">The chat model.</param>
    /// <param name="tools">The tools to run.</param>
    /// <param name="maxToolRounds">How many times tool results may be sent back to the model for one request.</param>
    public static IChatModel WithTools(this IChatModel model, IToolRegistry tools, int maxToolRounds)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(tools);
        ArgumentOutOfRangeException.ThrowIfNegative(maxToolRounds);
        return new ToolRunningChatModel(model, tools, maxToolRounds);
    }

    private sealed class ToolRunningChatModel(IChatModel model, IToolRegistry tools, int maxToolRounds) : IChatModel
    {
        public async IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var messages = new List<ChatMessage>(request.Messages);
            for (int rounds = 1; ; rounds++)
            {
                ChatChunk? final = null;
                await foreach (var chunk in model.StreamAsync(request with { Messages = [.. messages], Tools = tools.Definitions }, cancellationToken)
                    .ConfigureAwait(false))
                {
                    if (chunk.Done)
                    {
                        final = chunk;
                    }
                    else if (chunk.Delta.Content.Length > 0 || chunk.Delta.Thinking.Length > 0)
                    {
                        yield return new ChatChunk(chunk.Delta with { ToolCalls = [] });     // calls are reported once, at the end
                    }
                }

                if (final?.Message is not { } message)
                {
                    throw new InvalidOperationException("The model ended without a final message.");
                }

                var calls = message.ToolCalls ?? [];
                if (calls.Count == 0 || rounds > maxToolRounds)
                {
                    yield return final with { Delta = final.Delta with { ToolCalls = calls } };
                    yield break;
                }

                if (final.Delta.Content.Length > 0 || final.Delta.Thinking.Length > 0)
                {
                    yield return new ChatChunk(final.Delta with { ToolCalls = [] });
                }

                messages.Add(message);
                foreach (var result in await tools.InvokeAsync(calls, cancellationToken).ConfigureAwait(false))
                {
                    messages.Add(result.ToMessage());
                }
            }
        }
    }
}
