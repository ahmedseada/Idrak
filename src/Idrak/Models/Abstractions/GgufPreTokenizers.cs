// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// How the tokenizer of a GGUF file splits text before byte-level BPE, by the file's <c>tokenizer.ggml.pre</c> name: a
/// regular expression (the Hugging Face "Split" pre-tokenizer, isolated matches), or null for GPT-2's own rule.
/// Idrak registers the patterns of llama.cpp's llama-vocab.cpp for the llama3, qwen2, tekken and gpt2
/// families; add others with <see cref="Register"/>. A name nobody registered uses Llama 3's rule, and the prepared model
/// notes it.
/// </summary>
public static class GgufPreTokenizers
{
    private static readonly SlotTable<string, string?> Registry = new(nameof(GgufPreTokenizers), comparer: StringComparer.Ordinal,
        unguarded: "a pattern is data, not a call");

    static GgufPreTokenizers() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterGgufPreTokenizers);   // the built-in pre-tokenizers, on first use

    /// <summary>
    /// Registers the split pattern of the pre-tokenizer named <paramref name="name"/> in GGUF files; null for GPT-2's own
    /// rule. Under a built-in name it overrides the library's pattern until <see cref="Unregister"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, string? pattern)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (pattern is not null)
        {
            _ = new System.Text.RegularExpressions.Regex(pattern);  // a malformed pattern fails here, not at the first encode
        }

        Registry.Register(name, pattern, System.Reflection.Assembly.GetCallingAssembly(), pattern ?? "GPT-2's rule");
    }

    /// <summary>Removes the app's pre-tokenizer <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered pre-tokenizer names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The pattern registered as <paramref name="name"/> (null: GPT-2's rule); false when none is registered.</summary>
    public static bool TryGet(string name, out string? pattern) => Registry.TryGet(name, out pattern);

    /// <summary>The library's pattern for <paramref name="name"/>, whatever an app registered over it; null when the library has none (or uses GPT-2's rule).</summary>
    public static string? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the pre-tokenizer <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);
}
