// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Models.Abstractions;

/// <summary>
/// A chat template for GGUF files that carry none (no <c>tokenizer.chat_template</c>): the Jinja template written into the
/// prepared folder, and which files it is for, told by their tokens.
/// </summary>
public interface IGgufChatTemplate
{
    /// <summary>
    /// Whether a file without a template gets this one: the file and its <c>tokenizer.ggml.tokens</c> (a template is for
    /// the files whose vocabulary holds the tokens it writes).
    /// </summary>
    bool Matches(GgufFile file, IReadOnlyList<string> tokens);

    /// <summary>The Jinja chat template, as tokenizer_config.json's <c>chat_template</c> holds one.</summary>
    string Template { get; }
}

/// <summary>A chat template for GGUF files without one (<see cref="IGgufChatTemplate"/>), from its text and a test of the file's tokens.</summary>
public sealed class GgufChatTemplate : IGgufChatTemplate
{
    /// <summary>Whether a file without a template gets this one: the file and its <c>tokenizer.ggml.tokens</c>.</summary>
    public required Func<GgufFile, IReadOnlyList<string>, bool> Matches { get; init; }

    /// <inheritdoc />
    public required string Template { get; init; }

    bool IGgufChatTemplate.Matches(GgufFile file, IReadOnlyList<string> tokens) => Matches(file, tokens);
}
