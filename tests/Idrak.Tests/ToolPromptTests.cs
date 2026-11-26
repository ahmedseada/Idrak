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
}
