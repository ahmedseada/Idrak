// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// Tools for a model whose chat template cannot carry them (it renders no tool definitions: many instruct models'
/// templates): the tools and the call format read back from text (<see cref="ToolCallText.CallsInText"/>: a fenced JSON
/// block <c>{"name": …, "arguments": {…}}</c> per call) are described in the system message instead, tool results are
/// given as user messages, and an assistant's calls are written into its text, so any instruct model can act as an
/// agent. Family-neutral: the template is probed (<see cref="RendersTools"/>), never named.
/// </summary>
public static class ToolPrompts
{
    // A tool no template knows: a template renders tools when its prompt with this tool differs and names it.
    private const string ProbeName = "idrak_probe_tool_4f9c";

    private static readonly ConditionalWeakTable<IChatTemplate, StrongBox<bool>> Probed = new();

    /// <summary>
    /// Whether <paramref name="template"/> renders the tools it is given: a one-message conversation rendered with a probe
    /// tool names the tool and differs from the one rendered without. A template that fails with the tool renders none.
    /// Probed once per template.
    /// </summary>
    public static bool RendersTools(IChatTemplate template)
    {
        ArgumentNullException.ThrowIfNull(template);
        return Probed.GetValue(template, t => new StrongBox<bool>(Probe(t))).Value;
    }

    private static bool Probe(IChatTemplate template)
    {
        IReadOnlyList<ChatMessage> messages = [new("user", "hello")];
        var probe = new ToolDefinition(ProbeName, "A probe.", new JsonObject { ["type"] = "object", ["properties"] = new JsonObject() });
        try
        {
            string with = template.Render(messages, [probe], null);
            return with.Contains(ProbeName, StringComparison.Ordinal) && with != template.Render(messages, [], null);
        }
        catch (Exception e) when (e is InvalidOperationException or NotSupportedException or ArgumentException or FormatException)
        {
            return false;
        }
    }

    private static readonly ConditionalWeakTable<IChatTemplate, StrongBox<bool>> ProbedCalls = new();

    /// <summary>
    /// Whether <paramref name="template"/> renders an assistant message holding two tool calls (each with its result
    /// after it): false when it renders one call but fails on two (a template that allows a single call at once), true
    /// otherwise (also when one call already fails: the probe then tells nothing). Probed once per template.
    /// </summary>
    public static bool RendersSeveralCalls(IChatTemplate template)
    {
        ArgumentNullException.ThrowIfNull(template);
        return ProbedCalls.GetValue(template, t => new StrongBox<bool>(ProbeCalls(t))).Value;
    }

    private static bool ProbeCalls(IChatTemplate template)
    {
        var probe = new ToolDefinition(ProbeName, "A probe.", new JsonObject { ["type"] = "object", ["properties"] = new JsonObject() });
        IReadOnlyList<ChatMessage> Conversation(int calls)
        {
            var messages = new List<ChatMessage> { new("user", "hello"), new("assistant", "", ToolCalls: [.. Enumerable.Range(0, calls).Select(_ => new ToolCall(ProbeName, []))]) };
            messages.AddRange(Enumerable.Range(0, calls).Select(i => new ChatMessage("tool", i == 0 ? "one" : "two", ToolName: ProbeName)));
            return messages;
        }

        bool Renders(int calls)
        {
            try
            {
                _ = template.Render(Conversation(calls), [probe], null);
                return true;
            }
            catch (Exception e) when (e is InvalidOperationException or NotSupportedException or ArgumentException or FormatException)
            {
                return false;
            }
        }

        return !Renders(1) || Renders(2);
    }

    /// <summary>
    /// <paramref name="messages"/> with each assistant message of several tool calls split into consecutive assistant
    /// messages of one call each, each followed by its own result (the tool messages after it, in order), for a template
    /// that renders one call at once (<see cref="RendersSeveralCalls"/>). The text stays on the first; calls without a
    /// result yet follow each other. The same list comes back when no message has more than one call.
    /// </summary>
    public static IReadOnlyList<ChatMessage> OneCallEach(IReadOnlyList<ChatMessage> messages)
    {
        ArgumentNullException.ThrowIfNull(messages);
        if (!messages.Any(m => m.Role == "assistant" && m.ToolCalls is { Count: > 1 }))
        {
            return messages;
        }

        var result = new List<ChatMessage>(messages.Count + 4);
        for (int i = 0; i < messages.Count; i++)
        {
            var message = messages[i];
            if (message.Role != "assistant" || message.ToolCalls is not { Count: > 1 } calls)
            {
                result.Add(message);
                continue;
            }

            int results = 0;
            while (i + 1 + results < messages.Count && messages[i + 1 + results].Role == "tool")
            {
                results++;
            }

            for (int c = 0; c < calls.Count; c++)
            {
                result.Add(c == 0 ? message with { ToolCalls = [calls[0]] } : new ChatMessage("assistant", "", ToolCalls: [calls[c]]));
                if (c < results)
                {
                    result.Add(messages[i + 1 + c]);
                }
            }

            for (int extra = calls.Count; extra < results; extra++)
            {
                result.Add(messages[i + 1 + extra]);                         // more results than calls: kept after them, in order
            }

            i += results;
        }

        return result;
    }

    /// <summary>
    /// The instructions for <paramref name="tools"/>: how to call them (one fenced json block per call, as
    /// <see cref="ToolCallText.CallsInText"/> reads them), how results come back, and each tool's name, description and
    /// parameters' schema.
    /// </summary>
    public static string Describe(IReadOnlyList<ToolDefinition> tools)
    {
        ArgumentNullException.ThrowIfNull(tools);
        var text = new StringBuilder();
        text.Append("You can call tools. To call one, write a fenced json block holding only the call, one block per call:\n")
            .Append("```json\n{\"name\": \"TOOL_NAME\", \"arguments\": {\"ARGUMENT\": \"VALUE\"}}\n```\n")
            .Append("Write the calls and stop: each result comes back in the next message. Answer without a call when you are done.\n")
            .Append("The tools:\n");
        foreach (var tool in tools)
        {
            text.Append("- ").Append(tool.Name);
            if (!string.IsNullOrWhiteSpace(tool.Description))
            {
                text.Append(": ").Append(tool.Description.Trim());
            }

            if (tool.Parameters is { } parameters)
            {
                text.Append("\n  arguments (JSON schema): ").Append(parameters.ToJsonString());
            }

            text.Append('\n');
        }

        return text.ToString();
    }

    /// <summary>
    /// <paramref name="messages"/> for a template that renders no tools: <see cref="Describe"/> of
    /// <paramref name="tools"/> after the first system message's text (a system message of its own first when there is
    /// none), each tool result as a user message ("Result of TOOL:" and its text), and each assistant call its text does
    /// not already hold written after it as a fenced json block. Text and other parts are kept as they are.
    /// </summary>
    public static IReadOnlyList<ChatMessage> WithTools(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools)
    {
        ArgumentNullException.ThrowIfNull(messages);
        ArgumentNullException.ThrowIfNull(tools);
        string described = Describe(tools);
        var result = new List<ChatMessage>(messages.Count + 1);
        bool placed = false;
        foreach (var message in messages)
        {
            switch (message.Role)
            {
                case "system" when !placed:
                    placed = true;
                    result.Add(message with { Parts = [.. message.Parts, new ChatText((message.Content.Length > 0 ? "\n\n" : "") + described)] });
                    break;
                case "tool":
                    result.Add(new ChatMessage("user", [new ChatText($"Result of {message.ToolName ?? "the tool"}:\n"), .. message.Parts]));
                    break;
                case "assistant" when message.ToolCalls is { Count: > 0 } calls:
                {
                    var written = new StringBuilder();
                    foreach (var call in calls)
                    {
                        string json = new JsonObject { ["name"] = call.Name, ["arguments"] = call.Arguments.DeepClone() }.ToJsonString();
                        if (!message.Content.Contains($"\"{call.Name}\"", StringComparison.Ordinal))
                        {
                            written.Append(written.Length > 0 || message.Content.Length > 0 ? "\n\n" : "").Append("```json\n").Append(json).Append("\n```");
                        }
                    }

                    result.Add(message with { Parts = written.Length == 0 ? message.Parts : [.. message.Parts, new ChatText(written.ToString())], ToolCalls = null });
                    break;
                }

                default:
                    result.Add(message);
                    break;
            }
        }

        if (!placed)
        {
            result.Insert(0, new ChatMessage("system", described));
        }

        return result;
    }
}
