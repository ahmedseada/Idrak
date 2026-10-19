// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>
/// How the tokenizer of a GGUF file becomes a tokenizer definition (written as the prepared folder's tokenizer.json), by
/// the file's <c>tokenizer.ggml.model</c>. Built in: "gpt2" (byte-level BPE: Llama 3, Qwen, Mistral Nemo, …), "llama"
/// (SentencePiece BPE: Llama 2, Mistral 7B, TinyLlama, …), "t5" (SentencePiece Unigram) and "bert" (WordPiece). A converter
/// gets the file, its <c>tokenizer.ggml.tokens</c> and a list for notes on what it assumed. Add others with
/// <see cref="Register"/>; a file whose tokenizer nobody registered fails naming it.
/// </summary>
public static class GgufTokenizers
{
    private static readonly SlotTable<string, Func<GgufFile, IReadOnlyList<string>, List<string>, TokenizerDefinition>> Registry =
        new(nameof(GgufTokenizers), comparer: StringComparer.Ordinal, unguarded: "a converter's definition is its own file's reading");

    static GgufTokenizers() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterGgufTokenizers);   // the built-in converters, on first use

    /// <summary>Registers the converter of the GGUF tokenizer model <paramref name="name"/>; a built-in name overrides the library's until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string name, Func<GgufFile, IReadOnlyList<string>, List<string>, TokenizerDefinition> convert)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(convert);
        Registry.Register(name, convert, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's converter <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered tokenizer models.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The converter of <paramref name="name"/>, or null.</summary>
    public static Func<GgufFile, IReadOnlyList<string>, List<string>, TokenizerDefinition>? Find(string name) => Registry.Find(name);

    /// <summary>The converter of <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">No converter is registered under that name.</exception>
    public static Func<GgufFile, IReadOnlyList<string>, List<string>, TokenizerDefinition> Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"GGUF tokenizer '{name}' is not registered (registered: {string.Join(", ", Registry.Keys)}); add it with GgufTokenizers.Register.");

    /// <summary>The library's converter of <paramref name="name"/>, or null.</summary>
    public static Func<GgufFile, IReadOnlyList<string>, List<string>, TokenizerDefinition>? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the converter of <paramref name="name"/>.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>The registered tokenizer models, one a line, with who registered each.</summary>
    public static string Describe() => string.Join("\n", Registry.Keys.Select(name => $"{name}: {Registry.Origin(name)}"));
}
