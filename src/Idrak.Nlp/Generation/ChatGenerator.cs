// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Generation;

/// <summary>
/// Chat on top of a <see cref="TextGenerator"/>: renders the conversation with a <see cref="ChatTemplate"/>, generates
/// until the end-of-turn marker, and splits the output into reasoning, answer and tool calls as it streams.
/// </summary>
public sealed class ChatGenerator(TextGenerator generator, ChatTemplate? template = null) : IChatModel
{
    /// <summary>The underlying text generator.</summary>
    public TextGenerator Generator { get; } = generator;

    /// <summary>The prompt format.</summary>
    public ChatTemplate Template { get; } = template ?? new ChatMLTemplate();

    /// <summary>The prompt text for a request (useful for debugging templates).</summary>
    public string RenderPrompt(ChatRequest request) => Template.Render(request.Messages, request.Tools ?? [], request.Think);

    /// <summary>Generates the complete reply.</summary>
    public ChatChunk Chat(ChatRequest request, CancellationToken cancellationToken = default) =>
        Stream(request, cancellationToken).Last();

    /// <summary><see cref="Stream"/> on a background thread, as an <c>await foreach</c> stream.</summary>
    public IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, CancellationToken cancellationToken = default) =>
        BackgroundStream.Run(token => Stream(request, token), cancellationToken);

    /// <summary><see cref="Chat"/> on a background thread.</summary>
    public Task<ChatChunk> ChatAsync(ChatRequest request, CancellationToken cancellationToken = default) =>
        Task.Run(() => Chat(request, cancellationToken), CancellationToken.None);

    /// <summary>
    /// The complete replies to several requests, generated together (<see cref="TextGenerator.GenerateBatch"/>): the
    /// options of the first request apply to all. Each reply is what <see cref="Chat"/> returns for its request.
    /// </summary>
    public IReadOnlyList<ChatChunk> ChatBatch(IReadOnlyList<ChatRequest> requests, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0)
        {
            return [];
        }

        var options = requests[0].Options ?? new GenerationOptions();
        options = options with { Stop = [.. options.Stop, .. Template.StopSequences] };
        var outputs = Generator.GenerateBatch([.. requests.Select(RenderPrompt)], options, cancellationToken);
        var replies = new List<ChatChunk>(requests.Count);
        for (int i = 0; i < requests.Count; i++)
        {
            var request = requests[i];
            var (text, reason, stats) = outputs[i];
            var parser = new ChatOutputParser(Template, request.Tools, separateThinking: request.Think != false);
            var first = parser.Feed(text);
            var last = parser.Finish();
            var calls = first.ToolCalls.Concat(last.ToolCalls).ToList();
            string thinking = first.Thinking + last.Thinking;
            var message = new ChatMessage("assistant", (first.Content + last.Content).Trim(), thinking.Length > 0 ? thinking.Trim() : null,
                calls.Count > 0 ? calls : null);
            replies.Add(new ChatChunk(last, true, reason, message, stats));
        }

        return replies;
    }

    /// <summary>
    /// <see cref="ChatBatch"/> streamed: each request's reply as it is generated (content, thinking and tool calls parsed
    /// as <see cref="Stream"/> parses them), tagged with the request's index, then each request's final chunk (with the
    /// whole message and statistics) once all are done. The options of the first request apply to all.
    /// </summary>
    public IEnumerable<(int Index, ChatChunk Chunk)> StreamBatch(IReadOnlyList<ChatRequest> requests, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(requests);
        if (requests.Count == 0)
        {
            yield break;
        }

        var options = requests[0].Options ?? new GenerationOptions();
        options = options with { Stop = [.. options.Stop, .. Template.StopSequences] };
        var parsers = requests.Select(r => new ChatOutputParser(Template, r.Tools, separateThinking: r.Think != false)).ToArray();
        var content = requests.Select(_ => new System.Text.StringBuilder()).ToArray();
        var thinking = requests.Select(_ => new System.Text.StringBuilder()).ToArray();
        var calls = requests.Select(_ => new List<ToolCall>()).ToArray();
        foreach (var chunk in Generator.StreamBatch([.. requests.Select(RenderPrompt)], options, cancellationToken))
        {
            int i = chunk.Index;
            var delta = chunk.Done ? parsers[i].Finish() : parsers[i].Feed(chunk.Text);
            content[i].Append(delta.Content);
            thinking[i].Append(delta.Thinking);
            calls[i].AddRange(delta.ToolCalls);
            if (!chunk.Done)
            {
                if (delta.Content.Length > 0 || delta.Thinking.Length > 0 || delta.ToolCalls.Count > 0)
                {
                    yield return (i, new ChatChunk(delta));
                }

                continue;
            }

            string thought = thinking[i].ToString();
            var message = Reply(parsers[i], content[i].ToString(), thought, calls[i]);
            yield return (i, new ChatChunk(delta, true, chunk.DoneReason, message, chunk.Stats));
        }
    }

    /// <summary><see cref="StreamBatch"/> on a background thread, as an <c>await foreach</c> stream.</summary>
    public IAsyncEnumerable<(int Index, ChatChunk Chunk)> StreamBatchAsync(IReadOnlyList<ChatRequest> requests, CancellationToken cancellationToken = default) =>
        BackgroundStream.Run(token => StreamBatch(requests, token), cancellationToken);

    /// <summary>Streams the reply.</summary>
    public IEnumerable<ChatChunk> Stream(ChatRequest request, CancellationToken cancellationToken = default)
    {
        var options = request.Options ?? new GenerationOptions();
        options = options with { Stop = [.. options.Stop, .. Template.StopSequences] };
        var parser = new ChatOutputParser(Template, request.Tools, separateThinking: request.Think != false);
        var content = new System.Text.StringBuilder();
        var thinking = new System.Text.StringBuilder();
        var calls = new List<ToolCall>();

        foreach (var chunk in Generator.Stream(RenderPrompt(request), options, cancellationToken))
        {
            var delta = chunk.Done ? parser.Finish() : parser.Feed(chunk.Text);
            content.Append(delta.Content);
            thinking.Append(delta.Thinking);
            calls.AddRange(delta.ToolCalls);
            if (chunk.Done)
            {
                var message = Reply(parser, content.ToString(), thinking.ToString(), calls);
                yield return new ChatChunk(delta, true, chunk.DoneReason, message, chunk.Stats);
            }
            else if (!delta.IsEmpty)
            {
                yield return new ChatChunk(delta);
            }
        }
    }

    // The finished reply; a call written as JSON in the answer (not in the template's format) becomes that call.
    private static ChatMessage Reply(ChatOutputParser parser, string content, string thinking, List<ToolCall> calls)
    {
        if (calls.Count == 0 && parser.CallsInAnswer(content) is { } written)
        {
            calls = [.. written.Calls];
            content = written.Text;
        }

        return new ChatMessage("assistant", content.Trim(), thinking.Length > 0 ? thinking.Trim() : null, calls.Count > 0 ? [.. calls] : null);
    }
}
