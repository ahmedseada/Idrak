// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Models.Abstractions;

/// <summary>A tokenizer.json normalizer: rewrites the text between added tokens before it is split.</summary>
public interface ITokenizerNormalizer
{
    /// <summary>The normalized <paramref name="text"/>.</summary>
    string Normalize(string text);

    /// <summary>
    /// Normalizes <paramref name="text"/> in place, keeping where each character came from (the form the library's
    /// tokenizers run, with no string per call). By default the string form runs and the changed middle of the text
    /// stands for the old middle as a whole, so offsets through such a component are coarse; implement this to keep them
    /// exact.
    /// </summary>
    void Normalize(NormalizedText text)
    {
        ArgumentNullException.ThrowIfNull(text);
        string before = text.ToString();
        string after = Normalize(before);
        if (!string.Equals(before, after, StringComparison.Ordinal))
        {
            text.Set(after);
        }
    }
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

    /// <summary>
    /// Splits (and may rewrite) the pieces of <paramref name="text"/> in place, keeping where each character came from.
    /// By default the string form runs: a result piece found in order in the text keeps its characters' ranges, one that
    /// was rewritten stands for the piece it was found in as a whole.
    /// </summary>
    void PreTokenize(PreTokenizedText text, bool atStart)
    {
        ArgumentNullException.ThrowIfNull(text);
        TokenizerPipelineDefaults.PreTokenize(this, text, atStart);
    }
}

/// <summary>A tokenizer.json decoder stage: turns token texts (or the previous stage's pieces) into pieces of the decoded text.</summary>
public interface ITokenizerDecoder
{
    /// <summary>The pieces made of <paramref name="tokens"/>; the decoded text is the last stage's pieces joined.</summary>
    IReadOnlyList<string> Decode(IReadOnlyList<string> tokens);
}

/// <summary>
/// A tokenizer.json post-processor: adds a model's special tokens around an encoded text (or a pair of texts), with
/// their type ids, as <c>[CLS] … [SEP]</c> or <c>&lt;s&gt; … &lt;/s&gt;</c>.
/// </summary>
public interface ITokenizerPostProcessor
{
    /// <summary>How many tokens it adds to one text (<paramref name="pair"/> false) or to a pair (true), for truncation.</summary>
    int AddedTokens(bool pair);

    /// <summary>
    /// The encoding of <paramref name="encoding"/> (and <paramref name="pair"/>, when given) as the model takes it: with
    /// its special tokens when <paramref name="addSpecialTokens"/> is true, else the two joined as they are.
    /// </summary>
    TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens);
}

/// <summary>
/// A tokenizer.json component type (a normalizer, pre-tokenizer, decoder or post-processor), registered by its
/// <see cref="IKind.Name"/> (the "type" tokenizer.json names) in <see cref="TokenizerComponents"/>: the component made
/// from its JSON object (<see cref="IKind{TRequest, T}.Create"/>), once, when a tokenizer is read.
/// </summary>
/// <typeparam name="T">The component: <see cref="ITokenizerNormalizer"/>, <see cref="IPreTokenizer"/>, <see cref="ITokenizerDecoder"/> or <see cref="ITokenizerPostProcessor"/>.</typeparam>
public interface ITokenizerComponentKind<out T> : IKind<JsonObject, T> where T : class;

/// <summary>
/// The normalizer, pre-tokenizer, decoder and post-processor types a tokenizer.json may name ("type": "NFC", "Split", "ByteLevel", …), for
/// the tokenizers that read tokenizer.json (Idrak's <c>BpeTokenizer</c>). The built-in types are listed too: the Hugging
/// Face types the BPE tokenizer runs itself, on its fast paths. Register a kind for another type (it receives the
/// component's JSON object, <see cref="ITokenizerComponentKind{T}"/>) with <see cref="RegisterNormalizer"/>,
/// <see cref="RegisterPreTokenizer"/> or <see cref="RegisterDecoder"/>. Components are made once, when a tokenizer is read, and a tokenizer using a registered
/// one encodes (or decodes) through the general string pipeline instead of its fast paths.
/// <para>
/// Registering a built-in name overrides the built-in, which stays behind it as the library default
/// (<see cref="DefaultNormalizer"/>, …) until the app unregisters it. A failure of the app's component reaches the caller
/// and is reported (<see cref="SlotPolicy.Throw"/>, the default); under <see cref="SlotPolicy.FallBack"/> (see
/// <see cref="SetNormalizerPolicy"/>, …) a component that fails to be made is replaced by the built-in when the tokenizer
/// is read, and a call that throws is answered by the built-in; under <see cref="SlotPolicy.Shadow"/> the built-in answers
/// and the app's component is compared with it on a sample of calls.
/// </para>
/// </summary>
public static class TokenizerComponents
{
    // The types the BPE tokenizer runs itself, with their library defaults as components: what an app's component of
    // the same type falls back to or is compared with.
    private static readonly KindTable<ITokenizerComponentKind<ITokenizerNormalizer>> Normalizers = BuiltIn<ITokenizerNormalizer>(
        "TokenizerComponents.Normalizers", "Tokenizer normalizer", "RegisterNormalizer",
        ["Sequence", "NFC", "NFKC", "NFD", "NFKD", "Prepend", "Replace", "Lowercase", "Strip", "BertNormalizer", "Precompiled"],
        spec => new Normalizer(BpeTokenizer.NormalizerStep(spec), SpanComponents.BuiltInNormalizer(spec)),
        (slot, app, library) => new GuardedNormalizer(slot, app, library));

    private static readonly KindTable<ITokenizerComponentKind<IPreTokenizer>> PreTokenizers = BuiltIn<IPreTokenizer>(
        "TokenizerComponents.PreTokenizers", "Tokenizer pre-tokenizer", "RegisterPreTokenizer",
        ["Sequence", "Split", "ByteLevel", "Metaspace", "Digits", "Whitespace", "WhitespaceSplit", "Punctuation", "BertPreTokenizer"],
        spec => new PreTokenizer(BpeTokenizer.PreTokenizerStage(spec), SpanComponents.BuiltInPreTokenizer(spec)),
        (slot, app, library) => new GuardedPreTokenizer(slot, app, library));

    private static readonly KindTable<ITokenizerComponentKind<ITokenizerDecoder>> Decoders = BuiltIn<ITokenizerDecoder>(
        "TokenizerComponents.Decoders", "Tokenizer decoder", "RegisterDecoder",
        ["Sequence", "ByteLevel", "Replace", "ByteFallback", "Fuse", "Strip", "Metaspace", "WordPiece"],
        spec => new Decoder(BpeTokenizer.DecoderStage(spec)),
        (slot, app, library) => new GuardedDecoder(slot, app, library));

    private static readonly KindTable<ITokenizerComponentKind<ITokenizerPostProcessor>> PostProcessors = BuiltIn<ITokenizerPostProcessor>(
        "TokenizerComponents.PostProcessors", "Tokenizer post-processor", "RegisterPostProcessor", Idrak.Models.PostProcessors.Types, Idrak.Models.PostProcessors.BuiltIn,
        (slot, app, library) => new GuardedPostProcessor(slot, app, library));

    // A table of the built-in types ("TokenizerComponents.Normalizers": its policy is set by SetNormalizerPolicy), each type
    // a kind of its own. An app's kind is guarded when it makes a component (a failure falls back to the built-in, under
    // FallBack) and its component on each call (wrap).
    private static KindTable<ITokenizerComponentKind<T>> BuiltIn<T>(string registry, string kind, string register, string[] types, Func<JsonObject, T> library,
        Func<Slot, T, Later<T>, T> wrap) where T : class
    {
        var table = new KindTable<ITokenizerComponentKind<T>>(registry, kind, "add it with TokenizerComponents." + register,
            (slot, app, fallback) => new GuardedKind<T>(slot, app, fallback, wrap), StringComparer.Ordinal,
            setPolicy: "TokenizerComponents.Set" + registry[(registry.IndexOf('.') + 1)..^1] + "Policy");
        foreach (string type in types)
        {
            table.RegisterDefault(new BuiltInKind<T>(type, library));
        }

        return table;
    }

    // Under Shadow the built-in answers anyway: an app's component that cannot be made is reported, if sampled.
    private static bool ShadowFailed(Slot slot, Exception error)
    {
        if (error is OperationCanceledException)
        {
            return false;
        }

        slot.Shadow()?.Failed(error);
        return true;
    }

    /// <summary>Registers the normalizer type <paramref name="kind"/> (by its <see cref="IKind.Name"/>, the type tokenizer.json names); a built-in name overrides the built-in.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void RegisterNormalizer(ITokenizerComponentKind<ITokenizerNormalizer> kind) => Normalizers.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Registers the pre-tokenizer type <paramref name="kind"/> (by its <see cref="IKind.Name"/>); a built-in name overrides the built-in.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void RegisterPreTokenizer(ITokenizerComponentKind<IPreTokenizer> kind) => PreTokenizers.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Registers the decoder type <paramref name="kind"/> (by its <see cref="IKind.Name"/>); a built-in name overrides the built-in.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void RegisterDecoder(ITokenizerComponentKind<ITokenizerDecoder> kind) => Decoders.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Registers the post-processor type <paramref name="kind"/> (by its <see cref="IKind.Name"/>); a built-in name overrides the built-in.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void RegisterPostProcessor(ITokenizerComponentKind<ITokenizerPostProcessor> kind) => PostProcessors.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's post-processor type <paramref name="type"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool UnregisterPostProcessor(string type) => PostProcessors.Unregister(type);

    /// <summary>The post-processor types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> PostProcessorTypes => PostProcessors.Names;

    /// <summary>The kind registered for the post-processor type <paramref name="type"/> (guarded by its policy when it overrides a built-in), or null for a built-in type nobody replaced and for an unknown one.</summary>
    public static ITokenizerComponentKind<ITokenizerPostProcessor>? FindPostProcessor(string? type) => Find(PostProcessors, type);

    /// <summary>The built-in post-processor type <paramref name="type"/> as a component kind; null when it is not built in.</summary>
    public static ITokenizerComponentKind<ITokenizerPostProcessor>? DefaultPostProcessor(string type) => PostProcessors.Default(type);

    /// <summary>Who registered the post-processor type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? PostProcessorOrigin(string type) => PostProcessors.Origin(type);

    /// <summary>What happens when the app's post-processor type <paramref name="type"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the built-in).</summary>
    public static void SetPostProcessorPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) =>
        PostProcessors.SetPolicy(type, policy, shadowRate);

    /// <summary>Removes the app's normalizer type <paramref name="type"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool UnregisterNormalizer(string type) => Normalizers.Unregister(type);

    /// <summary>Removes the app's pre-tokenizer type <paramref name="type"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool UnregisterPreTokenizer(string type) => PreTokenizers.Unregister(type);

    /// <summary>Removes the app's decoder type <paramref name="type"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool UnregisterDecoder(string type) => Decoders.Unregister(type);

    /// <summary>The normalizer types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> NormalizerTypes => Normalizers.Names;

    /// <summary>The pre-tokenizer types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> PreTokenizerTypes => PreTokenizers.Names;

    /// <summary>The decoder types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> DecoderTypes => Decoders.Names;

    /// <summary>
    /// The kind an app registered for the normalizer type <paramref name="type"/> (guarded by its policy when it
    /// overrides a built-in), or null for a built-in type nobody replaced (the tokenizer runs it itself) and for an
    /// unknown one.
    /// </summary>
    public static ITokenizerComponentKind<ITokenizerNormalizer>? FindNormalizer(string? type) => Find(Normalizers, type);

    /// <summary>The kind an app registered for the pre-tokenizer type <paramref name="type"/>, or null (see <see cref="FindNormalizer"/>).</summary>
    public static ITokenizerComponentKind<IPreTokenizer>? FindPreTokenizer(string? type) => Find(PreTokenizers, type);

    /// <summary>The kind an app registered for the decoder type <paramref name="type"/>, or null (see <see cref="FindNormalizer"/>).</summary>
    public static ITokenizerComponentKind<ITokenizerDecoder>? FindDecoder(string? type) => Find(Decoders, type);

    /// <summary>The built-in normalizer type <paramref name="type"/> as a component kind (for an app's normalizer to delegate to); null when it is not built in.</summary>
    public static ITokenizerComponentKind<ITokenizerNormalizer>? DefaultNormalizer(string type) => Normalizers.Default(type);

    /// <summary>The built-in pre-tokenizer type <paramref name="type"/> as a component kind; null when it is not built in.</summary>
    public static ITokenizerComponentKind<IPreTokenizer>? DefaultPreTokenizer(string type) => PreTokenizers.Default(type);

    /// <summary>The built-in decoder type <paramref name="type"/> as a component kind; null when it is not built in.</summary>
    public static ITokenizerComponentKind<ITokenizerDecoder>? DefaultDecoder(string type) => Decoders.Default(type);

    /// <summary>Who registered the normalizer type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? NormalizerOrigin(string type) => Normalizers.Origin(type);

    /// <summary>Who registered the pre-tokenizer type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? PreTokenizerOrigin(string type) => PreTokenizers.Origin(type);

    /// <summary>Who registered the decoder type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? DecoderOrigin(string type) => Decoders.Origin(type);

    /// <summary>What happens when the app's normalizer type <paramref name="type"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the built-in).</summary>
    public static void SetNormalizerPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) =>
        Normalizers.SetPolicy(type, policy, shadowRate);

    /// <summary>What happens when the app's pre-tokenizer type <paramref name="type"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the built-in).</summary>
    public static void SetPreTokenizerPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) =>
        PreTokenizers.SetPolicy(type, policy, shadowRate);

    /// <summary>What happens when the app's decoder type <paramref name="type"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller; <see cref="SlotPolicy.FallBack"/> retries on the built-in).</summary>
    public static void SetDecoderPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) =>
        Decoders.SetPolicy(type, policy, shadowRate);

    private static ITokenizerComponentKind<T>? Find<T>(KindTable<ITokenizerComponentKind<T>> table, string? type) where T : class =>
        type is not null && table.Origin(type) is { } origin && origin != Overrides.Library ? table.Find(type) : null;

    // A built-in type: the library's component for its JSON.
    private sealed class BuiltInKind<T>(string name, Func<JsonObject, T> create) : ITokenizerComponentKind<T> where T : class
    {
        public string Name => name;

        public string Summary => "the library's " + name;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public T Create(JsonObject request) => create(request);
    }

    // An app's type over a built-in: its component made under the slot's policy (the built-in made when needed), then
    // guarded on each call.
    private sealed class GuardedKind<T>(Slot slot, ITokenizerComponentKind<T> app, ITokenizerComponentKind<T> library, Func<Slot, T, Later<T>, T> wrap)
        : ITokenizerComponentKind<T> where T : class
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public T Create(JsonObject request)
        {
            var builtIn = new Later<T>(() => library.Create(request));
            try
            {
                return wrap(slot, app.Create(request), builtIn);
            }
            catch (Exception e) when (slot.Policy == SlotPolicy.Shadow ? ShadowFailed(slot, e) : slot.Failed(e))
            {
                return builtIn.Value;
            }
        }
    }

    // The built-in types as components. Each call gets its own copy of the pieces or tokens: some stages change the
    // list they are given.
    private sealed class Normalizer(Func<string, string> step, Action<NormalizedText> span) : ITokenizerNormalizer
    {
        public string Normalize(string text) => step(text);

        public void Normalize(NormalizedText text) => span(text);
    }

    private sealed class PreTokenizer(Func<List<string>, bool, List<string>> stage, Action<PreTokenizedText, bool> span) : IPreTokenizer
    {
        public IReadOnlyList<string> PreTokenize(IReadOnlyList<string> pieces, bool atStart) => stage([.. pieces], atStart);

        public void PreTokenize(PreTokenizedText text, bool atStart) => span(text, atStart);
    }

    private sealed class Decoder(Func<List<string>, List<string>> stage) : ITokenizerDecoder
    {
        public IReadOnlyList<string> Decode(IReadOnlyList<string> tokens) => stage([.. tokens]);
    }

    // An app's component overriding a built-in, guarded on each call; the app's gets its own copy of the input (taken
    // before the built-in may change it).
    private sealed class GuardedNormalizer(Slot slot, ITokenizerNormalizer app, Later<ITokenizerNormalizer> library) : ITokenizerNormalizer
    {
        public string Normalize(string text) => slot.Call(() => app.Normalize(text), () => library.Value.Normalize(text), Comparisons.Exact);
    }

    private sealed class GuardedPreTokenizer(Slot slot, IPreTokenizer app, Later<IPreTokenizer> library) : IPreTokenizer
    {
        public IReadOnlyList<string> PreTokenize(IReadOnlyList<string> pieces, bool atStart)
        {
            List<string> copy = [.. pieces];
            return slot.Call(() => app.PreTokenize(copy, atStart), () => library.Value.PreTokenize(pieces, atStart), Comparisons.Sequences);
        }
    }

    private sealed class GuardedDecoder(Slot slot, ITokenizerDecoder app, Later<ITokenizerDecoder> library) : ITokenizerDecoder
    {
        public IReadOnlyList<string> Decode(IReadOnlyList<string> tokens)
        {
            List<string> copy = [.. tokens];
            return slot.Call(() => app.Decode(copy), () => library.Value.Decode(tokens), Comparisons.Sequences);
        }
    }

    private sealed class GuardedPostProcessor(Slot slot, ITokenizerPostProcessor app, Later<ITokenizerPostProcessor> library) : ITokenizerPostProcessor
    {
        public int AddedTokens(bool pair) => slot.Call(() => app.AddedTokens(pair), () => library.Value.AddedTokens(pair), Comparisons.Exact);

        public TokenizerEncoding Process(TokenizerEncoding encoding, TokenizerEncoding? pair, bool addSpecialTokens) =>
            slot.Call(() => app.Process(encoding, pair, addSpecialTokens), () => library.Value.Process(encoding, pair, addSpecialTokens),
                (expected, actual) => Comparisons.Sequences(expected.Ids, actual.Ids) ?? Comparisons.Sequences(expected.TypeIds, actual.TypeIds));
    }

    // The built-in component, made the first time it is needed (an app's component that works never needs it).
    private sealed class Later<T>(Func<T> make) where T : class
    {
        private T? _value;

        public T Value => _value ??= make();
    }
}
