// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;
using Idrak.Text;
using Idrak.Text.Generation;

// Tools for a template that cannot carry them (ToolPrompts): described in the system message, read back from fenced json.
internal abstract partial class DesktopTests : Tests
{
    // A template with no tool support: roles and text only, the tools it is given ignored.
    private sealed class NoToolsTemplate : ChatTemplate
    {
        public override IReadOnlyList<string> StopSequences { get; } = ["<end>"];

        public override void Render(IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think, System.Text.StringBuilder prompt)
        {
            foreach (var message in messages)
            {
                prompt.Append(message.Role).Append(": ").Append(message.Content).Append("<end>\n");
            }

            prompt.Append("assistant: ");
        }
    }

    private static void ToolsWithoutTemplateSupport(Device device)
    {
        if (device != Device.Cpu)
        {
            return;
        }

        var plain = new NoToolsTemplate();
        Check(!ToolPrompts.RendersTools(plain) && ToolPrompts.RendersTools(new ChatMLTemplate()), "the probe: a template that ignores tools renders none; ChatML renders them");
        string root = Path.Combine(Path.GetTempPath(), "idrak-notools-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(root);
        try
        {
            var definitions = new CodingTools(root).All.Select(t => t.Definition).ToList();
            var call = new ToolCall("read_file", new JsonObject { ["path"] = "Program.cs" });
            IReadOnlyList<ChatMessage> messages =
            [
                new("system", "You are a coding agent."), new("user", "Read Program.cs"),
                new("assistant", ToolCalls: [call]), new("tool", "1| class Program { }", ToolName: "read_file"),
            ];
            var rewritten = ToolPrompts.WithTools(messages, definitions);
            Check(rewritten.Count == 4 && rewritten[0].Content.StartsWith("You are a coding agent.\n\nYou can call tools.", StringComparison.Ordinal)
                  && definitions.All(d => rewritten[0].Content.Contains($"- {d.Name}: ", StringComparison.Ordinal)),
                $"the system message keeps its text, then the call format and every tool: {rewritten[0].Content}");
            Check(rewritten[3] is { Role: "user" } result && result.Content == "Result of read_file:\n1| class Program { }", $"a tool result as a user message: {rewritten[3].Content}");
            var context = CodingAgent.CallContext(plain, definitions);
            Check(rewritten[2] is { Role: "assistant", ToolCalls: null } && ToolCallText.CallsInText(rewritten[2].Content, context) is [{ Name: "read_file" } back]
                  && JsonNode.DeepEquals(back.Arguments, call.Arguments),
                $"an assistant's call written as the fenced json the agent reads back: {rewritten[2].Content}");
            string example = ToolPrompts.Describe(definitions);
            int fence = example.IndexOf("```json", StringComparison.Ordinal);
            int close = example.IndexOf("```\n", fence + 7, StringComparison.Ordinal) + 3;
            Check(fence > 0 && ToolCallText.CallsInText(example[fence..close].Replace("TOOL_NAME", "read_file", StringComparison.Ordinal), context) is [{ Name: "read_file" }],
                "the described format is the one CallsInText reads");
            Check(ToolPrompts.WithTools([new("user", "hi")], definitions) is [{ Role: "system" }, { Role: "user" }], "no system message: one of its own first");

            // ChatGenerator: tools described for the plain template, given to one that renders them, never when turned off.
            using var model = PretrainedModel.Load(CliModel);
            var generator = model.CreateGenerator();
            var request = new ChatRequest(messages, definitions);
            string described = new ChatGenerator(generator, plain).RenderPrompt(request);
            Check(described.Contains("You can call tools.", StringComparison.Ordinal) && described.Contains("Result of read_file:", StringComparison.Ordinal)
                  && described.Contains("\"name\":\"read_file\"", StringComparison.Ordinal),
                $"a template without tools: the prompt describes them ({described.Length} characters)");
            var chatml = new ChatMLTemplate();
            Check(new ChatGenerator(generator, chatml).RenderPrompt(request) == chatml.Render(messages, definitions, null), "a template with tools renders them as before");
            Check(!new ChatGenerator(generator, plain) { ToolsInPrompt = false }.RenderPrompt(request).Contains("You can call tools.", StringComparison.Ordinal),
                "ToolsInPrompt false: not described");
        }
        finally
        {
            Directory.Delete(root, true);
        }
    }
}
