// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models;

namespace Idrak.Models.Abstractions;

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
/// the tokenizers that read tokenizer.json (Idrak's <c>BpeTokenizer</c>). The built-in types are listed too: the Hugging
/// Face types the BPE tokenizer runs itself, on its fast paths. Register a factory for another type (it receives the
/// component's JSON object) with <see cref="RegisterNormalizer"/>, <see cref="RegisterPreTokenizer"/> or
/// <see cref="RegisterDecoder"/>. Components are made once, when a tokenizer is read, and a tokenizer using a registered
/// one encodes (or decodes) through the general string pipeline instead of its fast paths.
/// <para>
/// Registering a built-in name overrides the built-in, which stays behind it as the library default
/// (<see cref="DefaultNormalizer"/>, …) until the app unregisters it. Under <see cref="SlotPolicy.FallBack"/> (the
/// default; see <see cref="SetNormalizerPolicy"/>, …) a component that fails to be made is replaced by the built-in when
/// the tokenizer is read, and a call that throws is answered by the built-in; under <see cref="SlotPolicy.Shadow"/> the
/// built-in answers and the app's component is compared with it on a sample of calls.
/// </para>
/// </summary>
public static class TokenizerComponents
{
    // The types the BPE tokenizer runs itself, with their library defaults as components: what an app's component of
    // the same type falls back to or is compared with.
    private static readonly SlotTable<string, Func<JsonObject, ITokenizerNormalizer>> Normalizers = BuiltIn<ITokenizerNormalizer>(
        "TokenizerComponents.Normalizers", ["Sequence", "NFC", "NFKC", "NFD", "NFKD", "Prepend", "Replace", "Lowercase"],
        spec => new Normalizer(BpeTokenizer.NormalizerStep(spec)),
        (slot, app, library) => new GuardedNormalizer(slot, app, library));

    private static readonly SlotTable<string, Func<JsonObject, IPreTokenizer>> PreTokenizers = BuiltIn<IPreTokenizer>(
        "TokenizerComponents.PreTokenizers", ["Sequence", "Split", "ByteLevel", "Metaspace", "Digits", "Whitespace"],
        spec => new PreTokenizer(BpeTokenizer.PreTokenizerStage(spec)),
        (slot, app, library) => new GuardedPreTokenizer(slot, app, library));

    private static readonly SlotTable<string, Func<JsonObject, ITokenizerDecoder>> Decoders = BuiltIn<ITokenizerDecoder>(
        "TokenizerComponents.Decoders", ["Sequence", "ByteLevel", "Replace", "ByteFallback", "Fuse", "Strip", "Metaspace"],
        spec => new Decoder(BpeTokenizer.DecoderStage(spec)),
        (slot, app, library) => new GuardedDecoder(slot, app, library));

    // A table of the built-in types. An app's factory is guarded when it is made (a failure gives the built-in) and its
    // component on each call (wrap).
    private static SlotTable<string, Func<JsonObject, T>> BuiltIn<T>(string registry, string[] types, Func<JsonObject, T> library,
        Func<Slot, T, Later<T>, T> wrap) where T : class
    {
        var table = new SlotTable<string, Func<JsonObject, T>>(registry, (slot, app, fallback) => spec =>
        {
            var builtIn = new Later<T>(() => fallback(spec));
            try
            {
                return wrap(slot, app(spec), builtIn);
            }
            catch (Exception e) when (slot.Policy == SlotPolicy.Shadow ? ShadowFailed(slot, e) : slot.FallsBack(e))
            {
                return builtIn.Value;
            }
        }, StringComparer.Ordinal);
        foreach (string type in types)
        {
            table.RegisterDefault(type, library);
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

    /// <summary>Registers the normalizer type <paramref name="type"/>, made by <paramref name="create"/> from its JSON; a built-in name overrides the built-in.</summary>
    public static void RegisterNormalizer(string type, Func<JsonObject, ITokenizerNormalizer> create) => Register(Normalizers, type, create);

    /// <summary>Registers the pre-tokenizer type <paramref name="type"/>, made by <paramref name="create"/> from its JSON; a built-in name overrides the built-in.</summary>
    public static void RegisterPreTokenizer(string type, Func<JsonObject, IPreTokenizer> create) => Register(PreTokenizers, type, create);

    /// <summary>Registers the decoder type <paramref name="type"/>, made by <paramref name="create"/> from its JSON; a built-in name overrides the built-in.</summary>
    public static void RegisterDecoder(string type, Func<JsonObject, ITokenizerDecoder> create) => Register(Decoders, type, create);

    /// <summary>Removes the app's normalizer type <paramref name="type"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool UnregisterNormalizer(string type) => Normalizers.Unregister(type);

    /// <summary>Removes the app's pre-tokenizer type <paramref name="type"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool UnregisterPreTokenizer(string type) => PreTokenizers.Unregister(type);

    /// <summary>Removes the app's decoder type <paramref name="type"/> (a built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool UnregisterDecoder(string type) => Decoders.Unregister(type);

    /// <summary>The normalizer types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> NormalizerTypes => Normalizers.Keys;

    /// <summary>The pre-tokenizer types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> PreTokenizerTypes => PreTokenizers.Keys;

    /// <summary>The decoder types: the built-in ones and those registered.</summary>
    public static IReadOnlyCollection<string> DecoderTypes => Decoders.Keys;

    /// <summary>
    /// The factory an app registered for the normalizer type <paramref name="type"/> (guarded by its policy when it
    /// overrides a built-in), or null for a built-in type nobody replaced (the tokenizer runs it itself) and for an
    /// unknown one.
    /// </summary>
    public static Func<JsonObject, ITokenizerNormalizer>? FindNormalizer(string? type) => Find(Normalizers, type);

    /// <summary>The factory an app registered for the pre-tokenizer type <paramref name="type"/>, or null (see <see cref="FindNormalizer"/>).</summary>
    public static Func<JsonObject, IPreTokenizer>? FindPreTokenizer(string? type) => Find(PreTokenizers, type);

    /// <summary>The factory an app registered for the decoder type <paramref name="type"/>, or null (see <see cref="FindNormalizer"/>).</summary>
    public static Func<JsonObject, ITokenizerDecoder>? FindDecoder(string? type) => Find(Decoders, type);

    /// <summary>The built-in normalizer type <paramref name="type"/> as a component factory (for an app's normalizer to delegate to); null when it is not built in.</summary>
    public static Func<JsonObject, ITokenizerNormalizer>? DefaultNormalizer(string type) => Normalizers.Default(type);

    /// <summary>The built-in pre-tokenizer type <paramref name="type"/> as a component factory; null when it is not built in.</summary>
    public static Func<JsonObject, IPreTokenizer>? DefaultPreTokenizer(string type) => PreTokenizers.Default(type);

    /// <summary>The built-in decoder type <paramref name="type"/> as a component factory; null when it is not built in.</summary>
    public static Func<JsonObject, ITokenizerDecoder>? DefaultDecoder(string type) => Decoders.Default(type);

    /// <summary>Who registered the normalizer type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? NormalizerOrigin(string type) => Normalizers.Origin(type);

    /// <summary>Who registered the pre-tokenizer type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? PreTokenizerOrigin(string type) => PreTokenizers.Origin(type);

    /// <summary>Who registered the decoder type <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? DecoderOrigin(string type) => Decoders.Origin(type);

    /// <summary>What happens when the app's normalizer type <paramref name="type"/> fails (<see cref="SlotPolicy.FallBack"/> to the built-in unless set).</summary>
    public static void SetNormalizerPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) =>
        Normalizers.SetPolicy(type, policy, shadowRate);

    /// <summary>What happens when the app's pre-tokenizer type <paramref name="type"/> fails (<see cref="SlotPolicy.FallBack"/> to the built-in unless set).</summary>
    public static void SetPreTokenizerPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) =>
        PreTokenizers.SetPolicy(type, policy, shadowRate);

    /// <summary>What happens when the app's decoder type <paramref name="type"/> fails (<see cref="SlotPolicy.FallBack"/> to the built-in unless set).</summary>
    public static void SetDecoderPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) =>
        Decoders.SetPolicy(type, policy, shadowRate);

    private static void Register<T>(SlotTable<string, Func<JsonObject, T>> table, string type, Func<JsonObject, T> create)
    {
        ArgumentNullException.ThrowIfNull(type);
        ArgumentNullException.ThrowIfNull(create);
        table.Register(type, create);
    }

    private static Func<JsonObject, T>? Find<T>(SlotTable<string, Func<JsonObject, T>> table, string? type) =>
        type is not null && table.Origin(type) is { } origin && origin != Overrides.Library ? table.Find(type) : null;

    // The built-in types as components. Each call gets its own copy of the pieces or tokens: some stages change the
    // list they are given.
    private sealed class Normalizer(Func<string, string> step) : ITokenizerNormalizer
    {
        public string Normalize(string text) => step(text);
    }

    private sealed class PreTokenizer(Func<List<string>, bool, List<string>> stage) : IPreTokenizer
    {
        public IReadOnlyList<string> PreTokenize(IReadOnlyList<string> pieces, bool atStart) => stage([.. pieces], atStart);
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

    // The built-in component, made the first time it is needed (an app's component that works never needs it).
    private sealed class Later<T>(Func<T> make) where T : class
    {
        private T? _value;

        public T Value => _value ??= make();
    }
}
