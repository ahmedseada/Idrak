// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Models.Abstractions;

/// <summary>What a tokenizer model is built from (<see cref="ITokenizerModelKind"/>): what a reader found, and the read's settings.</summary>
/// <param name="Definition">The tokenizer's definition (its tables, components and special tokens).</param>
/// <param name="Options">The read's settings.</param>
public sealed record TokenizerModelRequest(TokenizerDefinition Definition, TokenizerOptions Options);

/// <summary>
/// A tokenizer model, registered by its <see cref="IKind.Name"/> (the type a tokenizer.json names, <c>model.type</c>) in
/// <see cref="TokenizerModels"/>: the tokenizer built from a definition (<see cref="IKind{TRequest, T}.Create"/>).
/// </summary>
public interface ITokenizerModelKind : IKind<TokenizerModelRequest, IPretrainedTokenizer>;

/// <summary>
/// The tokenizer models, by the type a tokenizer.json names (<c>model.type</c>): each builds a tokenizer from what a reader
/// (<see cref="TokenizerFiles"/>) found. Built in: "BPE" (<see cref="BpeTokenizer"/>: byte-level BPE, and SentencePiece-style
/// BPE with byte fallback), "WordPiece" (BERT) and "Unigram" (SentencePiece's unigram model, from a tokenizer.json or a .model file). Register a factory for another type with <see cref="Register"/>; an unregistered type fails
/// naming the registered ones.
/// <para>
/// Registering a built-in name overrides the built-in, which stays behind it as the library default (<see cref="Default"/>)
/// until the app unregisters it. A failure of the app's model reaches the caller and is reported (<see cref="SlotPolicy.Throw"/>,
/// the default); under <see cref="SlotPolicy.FallBack"/> (see <see cref="SetPolicy"/>) a model that fails to be built is
/// replaced by the built-in, and an encode or decode that throws is answered by it; under <see cref="SlotPolicy.Shadow"/>
/// the built-in answers and the app's model is compared with it on a sample of calls.
/// </para>
/// </summary>
public static class TokenizerModels
{
    /// <summary>The library's byte-pair-encoding model.</summary>
    public const string Bpe = "BPE";

    /// <summary>The library's WordPiece model (BERT).</summary>
    public const string WordPiece = "WordPiece";

    /// <summary>The library's unigram language model (SentencePiece's: T5, ALBERT, XLNet, XLM-R).</summary>
    public const string Unigram = "Unigram";

    private static readonly KindTable<ITokenizerModelKind> Table = new(nameof(TokenizerModels), "Tokenizer model", "add it with TokenizerModels.Register",
        (slot, app, library) => new GuardedKind(slot, app, library), StringComparer.Ordinal);

    static TokenizerModels() => Overrides.AsLibraryDefaults(LibraryModelFormats.RegisterTokenizerModels);   // the built-in models, on first use

    /// <summary>
    /// Registers <paramref name="kind"/> under its <see cref="IKind.Name"/> (the model type, as tokenizer.json's
    /// <c>model.type</c> names it), built from a definition and the read's settings; a built-in name overrides the built-in.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITokenizerModelKind kind) => Table.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's model type <paramref name="type"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool Unregister(string type) => Table.Unregister(type);

    /// <summary>The model types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The model type <paramref name="type"/> (guarded by its policy when an app's overrides a built-in), or null.</summary>
    public static ITokenizerModelKind? Find(ReadOnlySpan<char> type) => Table.Find(type);

    /// <summary>The model type <paramref name="type"/>.</summary>
    /// <exception cref="NotSupportedException">No model of that type is registered.</exception>
    public static ITokenizerModelKind Get(ReadOnlySpan<char> type) => Table.Find(type)
        ?? throw new NotSupportedException($"Tokenizer model '{type}' is not registered (registered: {string.Join(", ", Table.Names)}); add it with TokenizerModels.Register.");

    /// <summary>The library's model type <paramref name="type"/>, or null when it is not built in.</summary>
    public static ITokenizerModelKind? Default(string type) => Table.Default(type);

    /// <summary>Who registered the model type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string type) => Table.Origin(type);

    /// <summary>What happens when the app's model type <paramref name="type"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the built-in).</summary>
    public static void SetPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(type, policy, shadowRate);

    /// <summary>The model types, one a line, with who registered each.</summary>
    public static string Describe() => string.Join("\n", Table.Names.Select(type => $"{type}: {Table.Origin(type)}"));

    /// <summary>
    /// The tokenizer the model type <paramref name="definition"/> names builds from it (<see cref="TokenizerDefinition.ModelType"/>).
    /// </summary>
    /// <exception cref="NotSupportedException">No model of that type is registered.</exception>
    public static IPretrainedTokenizer Create(TokenizerDefinition definition, TokenizerOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(definition);
        string type = definition.ModelType;
        if (type.Length == 0)
        {
            throw new InvalidDataException($"{definition.Source}: the tokenizer names no model type (model.type).");
        }

        return Get(type).Create(new TokenizerModelRequest(definition, options ?? new TokenizerOptions()));
    }

    // Under Shadow the built-in answers anyway: an app's model that cannot be built is reported, if sampled.
    private static bool ShadowFailed(Slot slot, Exception error)
    {
        if (error is OperationCanceledException)
        {
            return false;
        }

        slot.Shadow()?.Failed(error);
        return true;
    }

    // An app's model type overriding a built-in: the app's tokenizer, guarded on each call (the built-in made only when needed).
    private sealed class GuardedKind(Slot slot, ITokenizerModelKind app, ITokenizerModelKind library) : ITokenizerModelKind
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public IPretrainedTokenizer Create(TokenizerModelRequest request)
        {
            // The built-in gets its own copy of the tables, made only when it is needed (the app's model may take them).
            var original = request.Definition.Copy();
            var builtIn = new Later(() => library.Create(request with { Definition = original }));
            try
            {
                return new GuardedTokenizer(slot, app.Create(request), builtIn);
            }
            catch (Exception e) when (slot.Policy == SlotPolicy.Shadow ? ShadowFailed(slot, e) : slot.Failed(e))
            {
                return builtIn.Value;
            }
        }
    }

    // The built-in tokenizer, built the first time it is needed (an app's model that works never needs it).
    private sealed class Later(Func<IPretrainedTokenizer> make)
    {
        private IPretrainedTokenizer? _value;

        public IPretrainedTokenizer Value => _value ??= make();
    }

    // An app's model overriding a built-in, guarded on each encode and decode; the rest is the app's.
    private sealed class GuardedTokenizer(Slot slot, IPretrainedTokenizer app, Later library) : IPretrainedTokenizer
    {
        public string Model => app.Model;

        public int VocabularySize => app.VocabularySize;

        public IReadOnlyList<int> Encode(string text) => slot.Call(() => app.Encode(text), () => library.Value.Encode(text), Comparisons.Sequences);

        public void EncodeRange(string text, int start, int length, List<int> ids) =>
            ids.AddRange(slot.Call(() => Range(app, text, start, length), () => Range(library.Value, text, start, length), Comparisons.Sequences));

        public string Decode(IEnumerable<int> ids)
        {
            int[] copy = [.. ids];
            return slot.Call(() => app.Decode(copy), () => library.Value.Decode(copy), Comparisons.Exact);
        }

        public string Decode(ReadOnlySpan<int> ids)
        {
            int[] copy = ids.ToArray();
            return slot.Call(() => app.Decode(copy.AsSpan()), () => library.Value.Decode(copy.AsSpan()), Comparisons.Exact);
        }

        public TokenizerEncoding Encode(string text, string? pair, TokenizerEncodeOptions options) =>
            slot.Call(() => app.Encode(text, pair, options), () => library.Value.Encode(text, pair, options),
                (expected, actual) => Comparisons.Sequences(expected.Ids, actual.Ids) ?? Comparisons.Sequences(expected.Offsets, actual.Offsets));

        // Each side's own batch encode, so its padding to the longest and its workers apply; compared text by text.
        public IReadOnlyList<TokenizerEncoding> EncodeBatch(IReadOnlyList<string> texts, IReadOnlyList<string>? pairs, TokenizerEncodeOptions options) =>
            slot.Call(() => app.EncodeBatch(texts, pairs, options), () => library.Value.EncodeBatch(texts, pairs, options), SameEncodings);

        private static string? SameEncodings(IReadOnlyList<TokenizerEncoding> expected, IReadOnlyList<TokenizerEncoding> actual)
        {
            if (expected.Count != actual.Count)
            {
                return $"{actual.Count} encodings, expected {expected.Count}";
            }

            for (int i = 0; i < expected.Count; i++)
            {
                if ((Comparisons.Sequences(expected[i].Ids, actual[i].Ids) ?? Comparisons.Sequences(expected[i].Offsets, actual[i].Offsets)) is { } difference)
                {
                    return $"text {i}: {difference}";
                }
            }

            return null;
        }

        public string TokenOf(int id) => app.TokenOf(id);

        public int? IdOf(string token) => app.IdOf(token);

        public bool IsSpecial(int id) => app.IsSpecial(id);

        public void PadVocabulary(int size) => app.PadVocabulary(size);

        public System.Text.Json.Nodes.JsonObject ToJson() => app.ToJson();

        public void WriteTo(Stream stream) => app.WriteTo(stream);

        public void Save(string path) => app.Save(path);

        // Each side's ids of the range in a list of its own, so the two can be compared before the caller's list grows.
        private static IReadOnlyList<int> Range(IPretrainedTokenizer tokenizer, string text, int start, int length)
        {
            var ids = new List<int>();
            tokenizer.EncodeRange(text, start, length, ids);
            return ids;
        }
    }
}
