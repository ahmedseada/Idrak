// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>A tokenizer.json normalizer: rewrites the text between added tokens before it is split.</summary>
public interface ITokenizerNormalizer
{
    /// <summary>The normalized <paramref name="text"/>.</summary>
    string Normalize(string text);
}

/// <summary>A tokenizer.json pre-tokenizer: splits (and may rewrite) text into the pieces BPE encodes one by one.</summary>
public interface IPreTokenizer
{
    /// <summary>
    /// The pieces made of <paramref name="pieces"/> (the previous stage's, or the normalized text as one piece);
    /// <paramref name="atStart"/> says whether the text begins the input (not after an added token). The final pieces are
    /// looked up in the vocabulary as they are.
    /// </summary>
    IReadOnlyList<string> PreTokenize(IReadOnlyList<string> pieces, bool atStart);
}

/// <summary>A tokenizer.json decoder stage: turns token texts (or the previous stage's pieces) into pieces of the decoded text.</summary>
public interface ITokenizerDecoder
{
    /// <summary>The pieces made of <paramref name="tokens"/>; the decoded text is the last stage's pieces joined.</summary>
    IReadOnlyList<string> Decode(IReadOnlyList<string> tokens);
}

/// <summary>
/// The normalizer, pre-tokenizer and decoder types a tokenizer.json may name ("type": "NFC", "Split", "ByteLevel", …), for
/// the tokenizers that read tokenizer.json (Idrak's <c>BpeTokenizer</c>). The built-in types are listed too:
/// the Hugging Face types the BPE tokenizer runs itself, on its fast paths. Register a factory for another type (it
/// receives the component's JSON object) with <see cref="RegisterNormalizer"/>, <see cref="RegisterPreTokenizer"/> or
/// <see cref="RegisterDecoder"/>; registering a built-in name replaces the built-in. Components are made once, when a
/// tokenizer is read, and a tokenizer using a registered one encodes (or decodes) through the general string pipeline
/// instead of its fast paths.
/// </summary>
public static class TokenizerComponents
{
    // The types the BPE tokenizer runs itself.
    private static readonly string[] BuiltInNormalizers = ["Sequence", "NFC", "NFKC", "NFD", "NFKD", "Prepend", "Replace", "Lowercase"];
    private static readonly string[] BuiltInPreTokenizers = ["Sequence", "Split", "ByteLevel", "Metaspace", "Digits", "Whitespace"];
    private static readonly string[] BuiltInDecoders = ["Sequence", "ByteLevel", "Replace", "ByteFallback", "Fuse", "Strip", "Metaspace"];

    // Every type by name; null: built into BpeTokenizer.
    private static readonly Dictionary<string, Func<JsonObject, ITokenizerNormalizer>?> Normalizers = BuiltIn<ITokenizerNormalizer>(BuiltInNormalizers);
    private static readonly Dictionary<string, Func<JsonObject, IPreTokenizer>?> PreTokenizers = BuiltIn<IPreTokenizer>(BuiltInPreTokenizers);
    private static readonly Dictionary<string, Func<JsonObject, ITokenizerDecoder>?> Decoders = BuiltIn<ITokenizerDecoder>(BuiltInDecoders);

    static TokenizerComponents() => LibraryDefaults.Ensure();   // components a first-party assembly adds are registered before the first lookup

    private static Dictionary<string, Func<JsonObject, T>?> BuiltIn<T>(string[] types) =>
        types.ToDictionary(t => t, Func<JsonObject, T>? (_) => null, StringComparer.Ordinal);

    /// <summary>Registers (or replaces) the normalizer type <paramref name="type"/>, made by <paramref name="create"/> from its JSON.</summary>
    public static void RegisterNormalizer(string type, Func<JsonObject, ITokenizerNormalizer> create) => Register(Normalizers, type, create);

    /// <summary>Registers (or replaces) the pre-tokenizer type <paramref name="type"/>, made by <paramref name="create"/> from its JSON.</summary>
    public static void RegisterPreTokenizer(string type, Func<JsonObject, IPreTokenizer> create) => Register(PreTokenizers, type, create);

    /// <summary>Registers (or replaces) the decoder type <paramref name="type"/>, made by <paramref name="create"/> from its JSON.</summary>
    public static void RegisterDecoder(string type, Func<JsonObject, ITokenizerDecoder> create) => Register(Decoders, type, create);

    /// <summary>Removes the registered normalizer type <paramref name="type"/> (a built-in name gets the built-in back); false when none was registered.</summary>
    public static bool UnregisterNormalizer(string type) => Unregister(Normalizers, BuiltInNormalizers, type);

    /// <summary>Removes the registered pre-tokenizer type <paramref name="type"/> (a built-in name gets the built-in back); false when none was registered.</summary>
    public static bool UnregisterPreTokenizer(string type) => Unregister(PreTokenizers, BuiltInPreTokenizers, type);

    /// <summary>Removes the registered decoder type <paramref name="type"/> (a built-in name gets the built-in back); false when none was registered.</summary>
    public static bool UnregisterDecoder(string type) => Unregister(Decoders, BuiltInDecoders, type);

    /// <summary>The normalizer types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> NormalizerTypes => Types(Normalizers);

    /// <summary>The pre-tokenizer types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> PreTokenizerTypes => Types(PreTokenizers);

    /// <summary>The decoder types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> DecoderTypes => Types(Decoders);

    /// <summary>
    /// The factory registered for the normalizer type <paramref name="type"/>, or null for a built-in type nobody replaced
    /// (the tokenizer runs it itself) and for an unknown one.
    /// </summary>
    public static Func<JsonObject, ITokenizerNormalizer>? FindNormalizer(string? type) => Find(Normalizers, type);

    /// <summary>The factory registered for the pre-tokenizer type <paramref name="type"/>, or null (see <see cref="FindNormalizer"/>).</summary>
    public static Func<JsonObject, IPreTokenizer>? FindPreTokenizer(string? type) => Find(PreTokenizers, type);

    /// <summary>The factory registered for the decoder type <paramref name="type"/>, or null (see <see cref="FindNormalizer"/>).</summary>
    public static Func<JsonObject, ITokenizerDecoder>? FindDecoder(string? type) => Find(Decoders, type);

    private static void Register<T>(Dictionary<string, Func<JsonObject, T>?> registry, string type, Func<JsonObject, T> create)
    {
        ArgumentNullException.ThrowIfNull(type);
        ArgumentNullException.ThrowIfNull(create);
        lock (registry)
        {
            registry[type] = create;
        }
    }

    private static bool Unregister<T>(Dictionary<string, Func<JsonObject, T>?> registry, string[] builtIn, string type)
    {
        lock (registry)
        {
            if (registry.GetValueOrDefault(type) is null)
            {
                return false;
            }

            if (builtIn.Contains(type))
            {
                registry[type] = null;
            }
            else
            {
                registry.Remove(type);
            }

            return true;
        }
    }

    private static IReadOnlyCollection<string> Types<T>(Dictionary<string, Func<JsonObject, T>?> registry)
    {
        lock (registry)
        {
            return [.. registry.Keys];
        }
    }

    private static Func<JsonObject, T>? Find<T>(Dictionary<string, Func<JsonObject, T>?> registry, string? type)
    {
        if (type is null)
        {
            return null;
        }

        lock (registry)
        {
            return registry.GetValueOrDefault(type);
        }
    }
}
