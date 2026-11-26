// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Generation;

/// <summary>What a chat template reader is asked with: the model folder and the model's tokenizer (or null).</summary>
/// <param name="Folder">The model folder.</param>
/// <param name="Tokenizer">The model's tokenizer, or null.</param>
public sealed record ChatTemplateRequest(string Folder, ITokenizer? Tokenizer = null);

/// <summary>
/// The readers of a model folder's own chat template, which loading a pretrained model asks (Idrak's
/// <c>PretrainedModel.Load</c>). Idrak.Text registers "jinja": the Jinja template of tokenizer_config.json,
/// chat_template.jinja or chat_template.json, rendered as Hugging Face renders it. Without a reader, a loaded model has
/// no chat template.
/// </summary>
public static class ChatTemplates
{
    /// <summary>The prompt <paramref name="template"/> renders for the next assistant turn, as a string (see <see cref="IChatTemplate.Render"/>).</summary>
    /// <exception cref="NotSupportedException">A message part whose kind the template does not render.</exception>
    public static string Render(this IChatTemplate template, IReadOnlyList<ChatMessage> messages, IReadOnlyList<ToolDefinition> tools, bool? think)
    {
        ArgumentNullException.ThrowIfNull(template);
        var prompt = new StringBuilder();
        template.Render(messages, tools, think, prompt);
        return prompt.ToString();
    }

}
