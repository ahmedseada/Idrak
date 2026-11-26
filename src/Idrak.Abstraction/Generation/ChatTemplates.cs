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
/// A reader of a model folder's own chat template: the checkpoint's own file (Hugging Face's tokenizer_config.json
/// "chat_template", chat_template.jinja or chat_template.json; a GGUF file's <c>tokenizer.chat_template</c>, written there
/// when the file is prepared), read where the folder is loaded. It is handed over to loading directly (Idrak's
/// <c>PretrainedOptions.ChatTemplateReader</c>; Idrak.Text's <c>JinjaChatTemplate.Reader</c> renders the Jinja templates as
/// Hugging Face does); nothing looks one up.
/// </summary>
public interface IChatTemplateReader
{
    /// <summary>The template the folder holds, or null when it holds none.</summary>
    /// <param name="request">The folder and its tokenizer.</param>
    IChatTemplate? Read(ChatTemplateRequest request);
}

/// <summary>Helpers for chat templates (<see cref="IChatTemplate"/>).</summary>
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
