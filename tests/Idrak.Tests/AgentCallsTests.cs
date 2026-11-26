// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Generation;

// From Craft's agent bench (Llama-3.1-8B-Instruct, Qwen2.5-7B, deepseek-coder-6.7b-instruct): tool arguments written as
// strings, a template that renders one call at once, an edit written in the same reply as the file's first read, the
// error of a missed edit on a short file, the prompt's language line, the reminder of an answer without tools, and the
// tools described in deepseek's real template.
internal abstract partial class DesktopTests
{
    // A template that renders tools and one call per assistant message, as Llama 3.1's: two calls raise its error.
    private sealed class OneCallTemplate : ChatTemplate
    {
        public override IReadOnlyList<string> StopSequences { get; } = ["<end>"];

        public override void Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think, System.Text.StringBuilder prompt)
        {
            foreach (var tool in tools)
            {
                prompt.Append("tool ").Append(tool.Name).Append('\n');
            }

            foreach (var message in messages)
            {
                if (message.ToolCalls is { Count: > 1 })
                {
                    throw new InvalidOperationException("Chat template error: This model only supports single tool-calls at once!");
                }

                prompt.Append(message.Role).Append(": ").Append(message.Content);
                foreach (var call in message.ToolCalls ?? [])
                {
                    prompt.Append("[call ").Append(call.Name).Append(' ').Append(call.Arguments.ToJsonString()).Append(']');
                }

                prompt.Append("<end>\n");
            }
        }
    }

    private static void ToolArgumentsCoerced(Device device)
    {
        _ = device;
        JsonObject? seen = null;
        var schema = new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject
            {
                ["flag"] = new JsonObject { ["type"] = "boolean" }, ["count"] = new JsonObject { ["type"] = "integer" }, ["ratio"] = new JsonObject { ["type"] = "number" },
                ["name"] = new JsonObject { ["type"] = "string" },
            },
        };
        var registry = ToolRegistry.Create().Add(Tool.Create("probe", "A probe.", schema, (args, _) =>
        {
            seen = args;
            return Task.FromResult("ok");
        })).Build();
        ToolResult Run(JsonObject arguments) => registry.InvokeAsync(new ToolCall("probe", arguments)).GetAwaiter().GetResult();

        var sent = new JsonObject { ["flag"] = " FALSE ", ["count"] = "12", ["ratio"] = "2.5", ["name"] = "true" };
        var result = Run(sent);
        Check(result.Succeeded && seen is { } args && args["flag"]!.GetValue<bool>() == false && args["count"]!.GetValue<long>() == 12 && args["ratio"]!.GetValue<double>() == 2.5
              && args["name"]!.GetValue<string>() == "true" && sent["flag"]!.GetValue<string>() == " FALSE " && result.Call.Arguments["count"]!.GetValue<string>() == "12",
            $"\"false\" (any case, spaced), \"12\" and \"2.5\" read as the schema's boolean, integer and number; a string argument kept; the call as sent ({seen?.ToJsonString()})");
        Check(Run(new JsonObject { ["flag"] = "True" }).Succeeded && seen!["flag"]!.GetValue<bool>() && Run(new JsonObject { ["count"] = "-3" }).Succeeded && seen!["count"]!.GetValue<long>() == -3,
            "\"True\" and \"-3\"");
        foreach (var (arguments, says) in new[]
        {
            (new JsonObject { ["flag"] = "maybe" }, "argument 'flag' must be a boolean, not \"maybe\""),
            (new JsonObject { ["count"] = "1.5" }, "argument 'count' must be an integer, not \"1.5\""),
            (new JsonObject { ["ratio"] = "NaN" }, "argument 'ratio' must be a number, not \"NaN\""),
        })
        {
            var refused = Run(arguments);
            Check(!refused.Succeeded && refused.Error == says, $"{arguments.ToJsonString()}: '{says}' ({refused.Error})");
        }
    }
}
