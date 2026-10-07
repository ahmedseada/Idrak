// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

/// <summary>
/// How the tokenizer of a GGUF file splits text before byte-level BPE, by the file's <c>tokenizer.ggml.pre</c> name: a
/// regular expression (the Hugging Face "Split" pre-tokenizer, isolated matches), or null for GPT-2's own rule.
/// Idrak registers the patterns of llama.cpp's llama-vocab.cpp for the llama3, qwen2, tekken and gpt2
/// families; add others with <see cref="Register"/>. A name nobody registered uses Llama 3's rule, and the prepared model
/// notes it.
/// </summary>
public static class GgufPreTokenizers
{
    private static readonly Dictionary<string, string?> Registry = new(StringComparer.Ordinal);

    static GgufPreTokenizers() => LibraryDefaults.Ensure(typeof(GgufPreTokenizers));

    /// <summary>
    /// Registers (or replaces) the split pattern of the pre-tokenizer named <paramref name="name"/> in GGUF files; null
    /// for GPT-2's own rule.
    /// </summary>
    public static void Register(string name, string? pattern)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        if (pattern is not null)
        {
            _ = new System.Text.RegularExpressions.Regex(pattern);  // a malformed pattern fails here, not at the first encode
        }

        lock (Registry)
        {
            Registry[name] = pattern;
        }
    }

    /// <summary>Removes the pre-tokenizer registered as <paramref name="name"/>; false when there is none.</summary>
    public static bool Unregister(string name)
    {
        lock (Registry)
        {
            return Registry.Remove(name);
        }
    }

    /// <summary>The registered pre-tokenizer names.</summary>
    public static IReadOnlyCollection<string> Names
    {
        get
        {
            lock (Registry)
            {
                return [.. Registry.Keys];
            }
        }
    }

    /// <summary>The pattern registered as <paramref name="name"/> (null: GPT-2's rule); false when none is registered.</summary>
    public static bool TryGet(string name, out string? pattern)
    {
        lock (Registry)
        {
            return Registry.TryGetValue(name, out pattern);
        }
    }
}
