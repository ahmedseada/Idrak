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

    // A model's folder in the local download store, or null (the test then says it was skipped).
    private static string? LocalModelFolder(string repository)
    {
        string folder = Path.Combine(ComputeResources.CacheFolder, "downloads", "huggingface", "models", repository);
        return Directory.Exists(folder) ? Directory.GetDirectories(folder).FirstOrDefault(d => File.Exists(Path.Combine(d, "tokenizer_config.json"))) : null;
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

    private static void ToolsOneCallAtOnce(Device device)
    {
        if (device != Device.Cpu)
        {
            return;
        }

        var one = new OneCallTemplate();
        Check(!ToolPrompts.RendersSeveralCalls(one) && ToolPrompts.RendersSeveralCalls(new ChatMLTemplate()),
            "the probe: a template that fails on two calls renders one at once; ChatML renders several");
        var read = new ToolCall("read_file", new JsonObject { ["path"] = "A.cs" });
        var list = new ToolCall("list_files", new JsonObject());
        IReadOnlyList<ChatMessage> messages =
        [
            new("user", "Look"), new("assistant", "Reading.", ToolCalls: [read, list]), new("tool", "A", ToolName: "read_file"), new("tool", "B", ToolName: "list_files"),
            new("assistant", "", ToolCalls: [list, read]),
        ];
        var split = ToolPrompts.OneCallEach(messages);
        Check(split is
            [
                { Role: "user" }, { Role: "assistant", Content: "Reading.", ToolCalls: [{ Name: "read_file" }] }, { Role: "tool", Content: "A" },
                { Role: "assistant", Content: "", ToolCalls: [{ Name: "list_files" }] }, { Role: "tool", Content: "B" },
                { Role: "assistant", ToolCalls: [{ Name: "list_files" }] }, { Role: "assistant", ToolCalls: [{ Name: "read_file" }] },
            ],
            $"each call its own assistant message, its result after it; calls without results follow each other ({string.Join(" | ", split.Select(m => $"{m.Role}:{m.Content}:{m.ToolCalls?.Count}"))})");
        IReadOnlyList<ChatMessage> single = [new("user", "x"), new("assistant", "", ToolCalls: [read])];
        Check(ReferenceEquals(ToolPrompts.OneCallEach(single), single), "no message of several calls: the same list");

        using var model = PretrainedModel.Load(CliModel);
        var generator = model.CreateGenerator();
        var tools = new[] { new ToolDefinition("read_file", "Read.", new JsonObject { ["type"] = "object" }), new ToolDefinition("list_files", "List.", new JsonObject { ["type"] = "object" }) };
        string prompt = new ChatGenerator(generator, one).RenderPrompt(new ChatRequest(messages, tools));
        int first = prompt.IndexOf("[call read_file", StringComparison.Ordinal), result = prompt.IndexOf("tool: A", StringComparison.Ordinal),
            second = prompt.IndexOf("[call list_files", StringComparison.Ordinal);
        Check(first >= 0 && first < result && result < second, $"ChatGenerator renders the split conversation for that template ({prompt})");

        // Llama 3.1's real template, from the local store: refuses two calls, renders them split.
        if (LocalModelFolder("meta-llama/Llama-3.1-8B-Instruct") is not { } llama || JinjaChatTemplate.Load(llama) is not { } template)
        {
            Console.WriteLine("    (skipped: Llama-3.1-8B-Instruct is not in the local store)");
            return;
        }

        Check(!ToolPrompts.RendersSeveralCalls(template), "Llama 3.1's template renders one call at once");
        string llamaPrompt = new ChatGenerator(generator, template).RenderPrompt(new ChatRequest(messages, tools));
        Check(llamaPrompt.Contains("read_file", StringComparison.Ordinal) && llamaPrompt.IndexOf("\"name\": \"read_file\"", StringComparison.Ordinal) < llamaPrompt.IndexOf("\"name\": \"list_files\"", StringComparison.Ordinal),
            $"Llama 3.1's template renders the split calls in order ({llamaPrompt.Length} characters)");
    }

    private static void DeepSeekPromptCarriesTools(Device device)
    {
        if (device != Device.Cpu)
        {
            return;
        }

        if (LocalModelFolder("deepseek-ai/deepseek-coder-6.7b-instruct") is not { } folder || JinjaChatTemplate.Load(folder) is not { } template)
        {
            Console.WriteLine("    (skipped: deepseek-coder-6.7b-instruct is not in the local store)");
            return;
        }

        string root = Path.Combine(Path.GetTempPath(), "idrak-deepseek-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var definitions = new CodingTools(root).All.Select(t => t.Definition).ToList();
            using var model = PretrainedModel.Load(CliModel);
            var chat = new ChatGenerator(model.CreateGenerator(), template);
            string prompt = chat.RenderPrompt(new ChatRequest([new("system", CodingAgent.DefaultSystemPrompt), new("user", "Add a Calculator class")], definitions));
            Check(!ToolPrompts.RendersTools(template) && prompt.Contains("You can call tools.", StringComparison.Ordinal)
                  && definitions.All(d => prompt.Contains($"- {d.Name}: ", StringComparison.Ordinal)) && prompt.Contains("### Instruction:", StringComparison.Ordinal),
                $"deepseek's template renders no tools, so ChatGenerator describes every tool in its system text ({prompt[..Math.Min(prompt.Length, 300)]} …)");
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }

    private static void AgentEditsAfterTheReadIsSeen(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            string program = Path.Combine(root, "Program.cs");
            File.WriteAllText(program, "Console.WriteLine(1);\n");
            var read = FakeChatModel.ToolCall("read_file", new JsonObject { ["path"] = "Program.cs" });
            var edit = new JsonObject { ["path"] = "Program.cs", ["old_text"] = "Console.WriteLine(1);", ["new_text"] = "Console.WriteLine(2);" };
            var both = new ChatMessage("assistant", "", ToolCalls: [read.ToolCalls![0], new ToolCall("edit_file", edit)]);
            var told = new List<ToolResult>();
            var fake = FakeChatModel.Script(both, FakeChatModel.ToolCall("edit_file", (JsonObject)edit.DeepClone()), FakeChatModel.Answer("Prints 2."));
            var run = new CodingAgent(fake) { OnToolResult = told.Add }.RunAsync("Print 2", new CodingTools(root)).GetAwaiter().GetResult();
            Check(told is [{ Succeeded: true }, { Succeeded: false } blind, { Succeeded: true }]
                  && blind.Error == "edit_file was written in the same reply as read_file for Program.cs, before its text was seen: edit it in your next reply, from read_file's result."
                  && File.ReadAllText(program) == "Console.WriteLine(2);\n" && run.End == CodingAgentEnd.Answered,
                $"an edit in the read's own reply is refused, the next reply's is made ({string.Join("; ", told.Select(r => r.Error ?? "ok"))})");

            // Read in an earlier reply: an edit beside a new read of it is made; a new file written and edited in one reply too.
            File.WriteAllText(program, "Console.WriteLine(1);\n");
            var again = FakeChatModel.Script(read, new ChatMessage("assistant", "", ToolCalls: [read.ToolCalls![0], new ToolCall("edit_file", (JsonObject)edit.DeepClone())]),
                new ChatMessage("assistant", "", ToolCalls: [new ToolCall("write_file", new JsonObject { ["path"] = "New.cs", ["content"] = "class A { }\n" }),
                    new ToolCall("edit_file", new JsonObject { ["path"] = "New.cs", ["old_text"] = "class A", ["new_text"] = "class B" })]),
                FakeChatModel.Answer("Done."));
            var second = new CodingAgent(again).RunAsync("Print 2", new CodingTools(root)).GetAwaiter().GetResult();
            Check(second.ToolErrors == 0 && File.ReadAllText(program) == "Console.WriteLine(2);\n" && File.ReadAllText(Path.Combine(root, "New.cs")) == "class B { }\n",
                $"a file read in an earlier reply, and one written in this reply, are edited ({second.ToolErrors} errors)");

            // Without replies told (the tools called directly), a read counts at once, as before.
            File.WriteAllText(program, "Console.WriteLine(1);\n");
            var direct = new CodingTools(root);
            _ = direct.ReadFile("Program.cs");
            var registry = direct.Registry();
            _ = registry.InvokeAsync(new ToolCall("read_file", new JsonObject { ["path"] = "Program.cs" })).GetAwaiter().GetResult();
            Check(registry.InvokeAsync(new ToolCall("edit_file", (JsonObject)edit.DeepClone())).GetAwaiter().GetResult().Succeeded, "no BeginReply: a read counts at once");
        });
    }

    private static void CodingEditMissedTwice(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            File.WriteAllText(Path.Combine(root, "Program.cs"), "Console.WriteLine(1);\n");
            var tools = new CodingTools(root);
            _ = tools.ReadFile("Program.cs");
            string first = tools.EditFile("Program.cs", "using System;", "x");
            Check(first.StartsWith("Error: old_text was not found in Program.cs. old_text was: \"using System;\".", StringComparison.Ordinal)
                  && first.Contains("The whole file is quoted here (without line numbers):\n```\nConsole.WriteLine(1);\n```", StringComparison.Ordinal)
                  && first.EndsWith("Or give write_file the file's complete new content instead of editing it.", StringComparison.Ordinal),
                $"a short file: the whole file quoted, write_file offered ({first})");
            string second = tools.EditFile("Program.cs", "using System.IO;", "x");
            Check(second.StartsWith("Error: old_text was not found in Program.cs again: use write_file with Program.cs's complete new content instead", StringComparison.Ordinal)
                  && second.Contains("Console.WriteLine(1);", StringComparison.Ordinal),
                $"the second miss on the file leads with write_file ({second})");
            _ = tools.EditFile("Program.cs", "Console.WriteLine(1);", "Console.WriteLine(2);");
            Check(tools.EditFile("Program.cs", "nothing", "x").StartsWith("Error: old_text was not found in Program.cs. old_text", StringComparison.Ordinal),
                "a successful edit starts the count again");
            tools.ForgetReads();
            _ = tools.ReadFile("Program.cs");
            Check(tools.EditFile("Program.cs", "nothing", "x").StartsWith("Error: old_text was not found in Program.cs. old_text", StringComparison.Ordinal), "a new task starts it again");
        });
    }

    private static void AgentRemindsAnswerWithoutTools(Device device)
    {
        _ = device;
        InAgentWorkspace(root =>
        {
            Check(CodingAgent.DefaultSystemPrompt.Contains("- Write every reply in the language of the user's request.\n", StringComparison.Ordinal),
                "the default prompt asks for the user's language");

            // deepseek's runs: the code in the answer, no call. Reminded once; it then writes the file.
            var fake = FakeChatModel.Script(FakeChatModel.Answer("```csharp\nclass Calculator { }\n```"),
                FakeChatModel.ToolCall("write_file", new JsonObject { ["path"] = "Calculator.cs", ["content"] = "class Calculator { }\n" }), FakeChatModel.Answer("Wrote Calculator.cs."));
            var run = new CodingAgent(fake).RunAsync("Add a Calculator class", new CodingTools(root)).GetAwaiter().GetResult();
            Check(run is { End: CodingAgentEnd.Answered, NoToolNudges: 1, ToolCalls: 1 } && fake.Requests[1].Messages[^1] is { Role: "user" } note && note.Content == CodingAgent.NoToolReminder
                  && File.Exists(Path.Combine(root, "Calculator.cs")) && run.ToJson()["no_tool_nudges"]?.GetValue<int>() == 1,
                $"an answer without any call is reminded once, in English, and the model then acts ({run.End}, {run.NoToolNudges})");

            // At most one a run; a model that answers again ends the run; 0 sends none.
            var words = FakeChatModel.Script(FakeChatModel.Answer("Nothing to change."), FakeChatModel.Answer("Nothing to change."));
            Check(new CodingAgent(words).RunAsync("Explain", new CodingTools(root)).GetAwaiter().GetResult() is { End: CodingAgentEnd.Answered, NoToolNudges: 1, Rounds: 2 },
                "the repeated answer ends the run");
            var off = FakeChatModel.Script(FakeChatModel.Answer("Nothing to change."));
            Check(new CodingAgent(off, new AgentOptions { NoToolNudges = 0 }).RunAsync("Explain", new CodingTools(root)).GetAwaiter().GetResult() is { End: CodingAgentEnd.Answered, NoToolNudges: 0 },
                "NoToolNudges 0: none");
        });
    }
}
