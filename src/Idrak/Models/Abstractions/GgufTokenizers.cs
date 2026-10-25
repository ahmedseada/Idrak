// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>What a GGUF tokenizer converter reads (<see cref="IGgufTokenizer"/>): the file, its tokens, and a list for notes on what it assumed.</summary>
/// <param name="File">The GGUF file.</param>
/// <param name="Tokens">Its <c>tokenizer.ggml.tokens</c>.</param>
/// <param name="Notes">Where the converter notes what it assumed (shown in the prepared model's notes).</param>
public sealed record GgufTokenizerRequest(GgufFile File, IReadOnlyList<string> Tokens, List<string> Notes);

/// <summary>
/// The converter of one GGUF tokenizer model, registered by its <see cref="IKind.Name"/> (the file's
/// <c>tokenizer.ggml.model</c>) in <see cref="GgufTokenizers"/>: the tokenizer definition of a file
/// (<see cref="IKind{TRequest, T}.Create"/>), written as the prepared folder's tokenizer.json.
/// </summary>
public interface IGgufTokenizer : IKind<GgufTokenizerRequest, TokenizerDefinition>;

/// <summary>
/// How the tokenizer of a GGUF file becomes a tokenizer definition (written as the prepared folder's tokenizer.json), by
/// the file's <c>tokenizer.ggml.model</c>. Built in: "gpt2" (byte-level BPE: Llama 3, Qwen, Mistral Nemo, …), "llama"
/// (SentencePiece BPE: Llama 2, Mistral 7B, TinyLlama, …), "t5" (SentencePiece Unigram) and "bert" (WordPiece). Add others
/// with <see cref="Register"/>; a file whose tokenizer nobody registered fails naming it.
/// </summary>
public static class GgufTokenizers
{
    private static readonly KindTable<IGgufTokenizer> Registry = new(nameof(GgufTokenizers), "GGUF tokenizer", "add it with GgufTokenizers.Register",
        comparer: StringComparer.Ordinal, unguarded: "a converter's definition is its own file's reading");

    static GgufTokenizers() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterGgufTokenizers);   // the built-in converters, on first use

    /// <summary>Registers <paramref name="converter"/> under its <see cref="IKind.Name"/>; a built-in name overrides the library's until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IGgufTokenizer converter) => Registry.Register(converter, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's converter <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered tokenizer models.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Names;

    /// <summary>The converter of <paramref name="name"/>, or null.</summary>
    public static IGgufTokenizer? Find(ReadOnlySpan<char> name) => Registry.Find(name);

    /// <summary>The converter of <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">No converter is registered under that name.</exception>
    public static IGgufTokenizer Get(ReadOnlySpan<char> name) => Registry.Get(name);

    /// <summary>The library's converter of <paramref name="name"/>, or null.</summary>
    public static IGgufTokenizer? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the converter of <paramref name="name"/>.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>The registered tokenizer models, one a line, with who registered each.</summary>
    public static string Describe() => string.Join("\n", Registry.Names.Select(name => $"{name}: {Registry.Origin(name)}"));
}
