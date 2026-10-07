// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Generation;

namespace Idrak.LanguageModels;

/// <summary>Settings for <see cref="TeacherData.Generate"/>.</summary>
public sealed record TeacherDataOptions
{
    /// <summary>
    /// Keep the teacher's reasoning: it is asked to think (the template's thinking mode on) and its reasoning goes into
    /// each answer as <c>reasoning_content</c>, so the student learns to reason as well; false (the default) asks for the
    /// answer alone.
    /// </summary>
    public bool Reasoning { get; init; }

    /// <summary>The longest answer, in tokens (reasoning included).</summary>
    public int MaxNewTokens { get; init; } = 1024;

    /// <summary>Sampling temperature (0, the default: greedy, the teacher's most likely answer).</summary>
    public float Temperature { get; init; }

    /// <summary>The seed for sampling (null: random), when <see cref="Temperature"/> is above 0.</summary>
    public int? Seed { get; init; }

    /// <summary>Prompts answered together, when the generator can batch.</summary>
    public int BatchSize { get; init; } = 8;

    /// <summary>The context window for generating (prompt and answer).</summary>
    public int Context { get; init; } = 4096;

    /// <summary>A system message for prompts without one.</summary>
    public string? System { get; init; }

    /// <summary>Keep answers cut off at <see cref="MaxNewTokens"/> (false, the default: they are left out, so the student does not learn to stop mid-answer).</summary>
    public bool KeepCutOff { get; init; }
}

/// <summary>
/// Teacher-generated data for sequence-level distillation: a teacher model answers prompts, and its answers become chat
/// rows (<c>{"messages": [...]}</c>) for supervised fine-tuning of a student (<see cref="ChatTranscriptEncoder"/>,
/// <see cref="FineTuner"/>). Unlike logit distillation (<see cref="DistillationTeacher"/>) it needs no shared vocabulary:
/// the student learns from the teacher's text.
/// </summary>
/// <example>
/// <code>
/// using var teacher = PretrainedModel.Load(teacherFolder);
/// var prompts = File.ReadLines("prompts.jsonl").Select(line => JsonNode.Parse(line)!.AsObject());
/// var rows = TeacherData.Generate(teacher.CreateChat(), prompts, new TeacherDataOptions { Reasoning = true });
/// File.WriteAllLines("teacher.jsonl", rows.Select(r => r.ToJsonString()));
/// </code>
/// </example>
public static class TeacherData
{
    // Columns read as the prompt of a row that has no answer.
    private static readonly string[] PromptColumns = ["prompt", "question", "instruction", "query", "input", "problem"];

    /// <summary>
    /// The teacher's answer to each row's prompt, as a chat row: the prompt's messages (a conversation up to its last user
    /// message, a question and answer row's question, or a prompt column), then the teacher's assistant message (with
    /// <c>reasoning_content</c> when <see cref="TeacherDataOptions.Reasoning"/>). Rows without a prompt, empty answers and
    /// (unless <see cref="TeacherDataOptions.KeepCutOff"/>) answers cut off by the length limit are left out. Answers are
    /// generated <see cref="TeacherDataOptions.BatchSize"/> prompts at a time, in order, as the rows are enumerated.
    /// </summary>
    public static IEnumerable<JsonObject> Generate(ChatGenerator teacher, IEnumerable<JsonObject> rows, TeacherDataOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(teacher);
        ArgumentNullException.ThrowIfNull(rows);
        options ??= new TeacherDataOptions();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.BatchSize);
        var generation = options.Temperature <= 0f
            ? new GenerationOptions { Temperature = 0f, TopK = 1, TopP = 1f, RepeatPenalty = 1f, NumPredict = options.MaxNewTokens, NumCtx = options.Context }
            : new GenerationOptions { Temperature = options.Temperature, TopK = 40, TopP = 0.95f, RepeatPenalty = 1f, NumPredict = options.MaxNewTokens, NumCtx = options.Context, Seed = options.Seed };
        var prompts = rows.Select(r => Prompt(r, options.System)).OfType<JsonObject>();
        foreach (var group in prompts.Chunk(options.BatchSize))
        {
            cancellationToken.ThrowIfCancellationRequested();
            var requests = group.Select(p =>
            {
                var transcript = ChatTranscript.FromJson(p);
                return new ChatRequest(transcript.Messages, transcript.Tools.Count > 0 ? transcript.Tools : null, options.Reasoning, generation);
            }).ToList();
            IReadOnlyList<ChatChunk> replies = requests.Count > 1 && teacher.Generator.SupportsBatches
                ? teacher.ChatBatch(requests, cancellationToken)
                : [.. requests.Select(r => teacher.Chat(r, cancellationToken))];
            for (int i = 0; i < group.Length; i++)
            {
                var message = replies[i].Message;
                string content = message?.Content.Trim() ?? "";
                if (content.Length == 0 && message?.ToolCalls is not { Count: > 0 } || replies[i].DoneReason == "length" && !options.KeepCutOff)
                {
                    continue;
                }

                var answer = new JsonObject { ["role"] = "assistant", ["content"] = content };
                if (options.Reasoning && message?.Thinking is { Length: > 0 } thinking)
                {
                    answer["reasoning_content"] = thinking.Trim();
                }

                if (message?.ToolCalls is { Count: > 0 } calls)
                {
                    answer["tool_calls"] = new JsonArray([.. calls.Select(c => (JsonNode)new JsonObject
                    {
                        ["type"] = "function",
                        ["function"] = new JsonObject { ["name"] = c.Name, ["arguments"] = c.Arguments.DeepClone() },
                    })]);
                }

                var row = (JsonObject)group[i].DeepClone();
                row["messages"]!.AsArray().Add((JsonNode)answer);
                yield return row;
            }
        }
    }

    /// <summary>
    /// The prompt of <paramref name="row"/> as <c>{"messages": [...]}</c> ending with a user message (with the row's
    /// "tools"), or null when it has none: a conversation in any layout <see cref="ChatRows"/> reads, cut after its last
    /// user message; or a prompt column (prompt, question, instruction, query, input, problem) as text or as messages.
    /// <paramref name="system"/> starts a prompt that has no system message.
    /// </summary>
    public static JsonObject? Prompt(JsonObject row, string? system = null)
    {
        ArgumentNullException.ThrowIfNull(row);
        JsonArray? messages = null;
        JsonNode? tools = row["tools"]?.DeepClone();
        if (ChatRows.Normalize(row, RowKind.Chat) is { } chat && chat["messages"] is JsonArray normalized)
        {
            messages = normalized;
            tools ??= chat["tools"]?.DeepClone();
        }
        else if (PromptColumns.Select(c => row[c]).FirstOrDefault(n => n is JsonArray or JsonValue) is { } prompt)
        {
            messages = prompt switch
            {
                JsonArray list when ChatRows.Normalize(new JsonObject { ["messages"] = list.DeepClone() }, RowKind.Chat)?["messages"] is JsonArray m => m,
                JsonValue v when v.TryGetValue<string>(out var text) && text.Trim().Length > 0 => [new JsonObject { ["role"] = "user", ["content"] = text }],
                _ => null,
            };
        }

        if (messages is null)
        {
            return null;
        }

        int lastUser = -1;
        for (int i = 0; i < messages.Count; i++)
        {
            if ((string?)messages[i]?["role"] == "user")
            {
                lastUser = i;
            }
        }

        if (lastUser < 0)
        {
            return null;
        }

        var cut = new JsonArray([.. messages.Take(lastUser + 1).Select(m => m!.DeepClone())]);
        if (system is not null && (string?)cut[0]?["role"] != "system")
        {
            cut.Insert(0, new JsonObject { ["role"] = "system", ["content"] = system });
        }

        var result = new JsonObject { ["messages"] = cut };
        if (tools is not null)
        {
            result["tools"] = tools;
        }

        return result;
    }
}
