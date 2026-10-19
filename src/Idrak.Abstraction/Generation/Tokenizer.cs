// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>Turns text into token ids and back.</summary>
public interface ITokenizer
{
    /// <summary>Number of distinct token ids (the model's output width).</summary>
    int VocabularySize { get; }

    /// <summary>Token ids for <paramref name="text"/>.</summary>
    IReadOnlyList<int> Encode(string text);

    /// <summary>The text of a sequence of token ids.</summary>
    string Decode(IEnumerable<int> ids);

    /// <summary>The text of a sequence of token ids given as a span (tokenizers may decode it without copying the ids).</summary>
    string Decode(ReadOnlySpan<int> ids) => Decode((IEnumerable<int>)ids.ToArray());

    /// <summary>
    /// The vocabulary entry of <paramref name="id"/> as the tokenizer stores it ("" past the end), for comparing
    /// vocabularies (two models share one when every id is the same entry). A byte-level BPE tokenizer returns the entry
    /// itself (which may be part of a character); by default, the id's decoded text.
    /// </summary>
    string TokenOf(int id) => (uint)id < (uint)VocabularySize ? Decode([id]) : "";
}

/// <summary>
/// A tokenizer a model ships with (a Hugging Face <c>tokenizer.json</c>, a SentencePiece model, a vocabulary file), or one
/// a trainer learned: a vocabulary of named tokens with special ones, written back as a <c>tokenizer.json</c>. The library's
/// models (BPE, and the others registered by name) implement it, so code that loads, pads, saves or looks tokens up works
/// with any of them.
/// </summary>
public interface IPretrainedTokenizer : ITokenizer
{
    /// <summary>The tokenizer model's type, as tokenizer.json names it ("BPE", "Unigram", "WordPiece", …).</summary>
    string Model { get; }

    /// <summary>The id of <paramref name="token"/> (a vocabulary entry or an added token), or null.</summary>
    int? IdOf(string token);

    /// <summary>Whether <paramref name="id"/> is a special token (false for any id outside the vocabulary).</summary>
    bool IsSpecial(int id);

    /// <summary>
    /// Extends the vocabulary to <paramref name="size"/> ids (models often have more embedding rows than the tokenizer has
    /// tokens, for alignment); the extra ids decode to nothing. A smaller size changes nothing.
    /// </summary>
    void PadVocabulary(int size);

    /// <summary>
    /// Appends the ids of <paramref name="text"/>[<paramref name="start"/>..<paramref name="start"/> + <paramref name="length"/>]
    /// to <paramref name="ids"/>: the ids <see cref="ITokenizer.Encode"/> gives for that substring, without copying it.
    /// </summary>
    void EncodeRange(string text, int start, int length, List<int> ids);

    /// <summary>The tokenizer as the JSON of a tokenizer.json (see <see cref="Save"/>).</summary>
    JsonObject ToJson();

    /// <summary>Writes the tokenizer.json (see <see cref="Save"/>) to <paramref name="stream"/>.</summary>
    void WriteTo(Stream stream);

    /// <summary>
    /// Writes the tokenizer as a tokenizer.json (UTF-8), from which the library's loader reads back a tokenizer that encodes
    /// every text to the same ids.
    /// </summary>
    /// <param name="path">A file, or a folder (then its tokenizer.json).</param>
    void Save(string path);

    /// <summary>
    /// <paramref name="text"/> (and <paramref name="pair"/>, when given) as the model takes it: ids with their type ids,
    /// offsets in the text each came from, attention and special-tokens masks and sequence ids, truncated, given the
    /// post-processor's special tokens and padded as <paramref name="options"/> say (as Hugging Face's <c>encode</c>).
    /// </summary>
    TokenizerEncoding Encode(string text, string? pair, TokenizerEncodeOptions options);

    /// <summary>
    /// <see cref="Encode(string, string?, TokenizerEncodeOptions)"/> of each text (and its pair), spread over the tokenizer's
    /// workers, in input order; padding "longest" pads to the batch's longest.
    /// </summary>
    IReadOnlyList<TokenizerEncoding> EncodeBatch(IReadOnlyList<string> texts, IReadOnlyList<string>? pairs, TokenizerEncodeOptions options);

    /// <summary>The text of <paramref name="ids"/>, leaving out the special tokens when <paramref name="skipSpecialTokens"/> is true.</summary>
    string Decode(ReadOnlySpan<int> ids, bool skipSpecialTokens)
    {
        if (!skipSpecialTokens)
        {
            return Decode(ids);
        }

        var kept = new List<int>(ids.Length);
        foreach (int id in ids)
        {
            if (!IsSpecial(id))
            {
                kept.Add(id);
            }
        }

        return Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(kept));
    }
}

/// <summary>
/// How <see cref="IPretrainedTokenizer.Encode(string, string?, TokenizerEncodeOptions)"/> shapes an encoding. A setting left
/// null takes the tokenizer.json's own (its <c>truncation</c> and <c>padding</c>), else none.
/// </summary>
public sealed record TokenizerEncodeOptions
{
    /// <summary>Whether the post-processor's special tokens are added (<c>[CLS] … [SEP]</c>, <c>&lt;s&gt;</c>): true by default, as Hugging Face's <c>encode</c>.</summary>
    public bool AddSpecialTokens { get; init; } = true;

    /// <summary>Truncate to this many tokens, special tokens included; null: the tokenizer.json's <c>truncation.max_length</c>, else no truncation.</summary>
    public int? MaxLength { get; init; }

    /// <summary>
    /// Which text loses tokens past <see cref="MaxLength"/>: "longest_first" (one token at a time from the longer),
    /// "only_first" or "only_second"; null: the tokenizer.json's strategy, else "longest_first".
    /// </summary>
    public string? Truncation { get; init; }

    /// <summary>
    /// "none", "longest" (a batch to its longest encoding) or "max_length" (to <see cref="MaxLength"/>); null: the
    /// tokenizer.json's padding, else none.
    /// </summary>
    public string? Padding { get; init; }

    /// <summary>The id padding adds; null: the tokenizer.json's <c>padding.pad_id</c>, else the tokenizer's padding token.</summary>
    public int? PadId { get; init; }
}


/// <summary>
/// One token per character of a fixed alphabet (the vocabulary string). Characters outside the alphabet are
/// dropped when encoding, or replaced by <see cref="UnknownCharacter"/> when it is part of the alphabet.
/// </summary>
public sealed class CharTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _index;
    private readonly int _unknownId = -1;                      // the id of UnknownCharacter (-1: none), looked up once

    /// <summary>Creates the tokenizer for the characters of <paramref name="vocabulary"/> (id = position).</summary>
    public CharTokenizer(string vocabulary, char? unknownCharacter = null)
    {
        Vocabulary = vocabulary;
        _index = vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
        if (unknownCharacter is { } u && _index.TryGetValue(u, out int unknownId))
        {
            UnknownCharacter = u;
            _unknownId = unknownId;
        }
    }

    /// <summary>The alphabet; character i has id i.</summary>
    public string Vocabulary { get; }

    /// <summary>Stand-in for characters outside the alphabet, or null to drop them.</summary>
    public char? UnknownCharacter { get; }

    /// <inheritdoc />
    public int VocabularySize => Vocabulary.Length;

    /// <summary>Whether every character of <paramref name="text"/> is in the alphabet.</summary>
    public bool Covers(string text) => text.All(_index.ContainsKey);

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>(text.Length);
        foreach (char c in text)
        {
            if (_index.TryGetValue(c, out int id))
            {
                ids.Add(id);
            }
            else if (_unknownId >= 0)
            {
                ids.Add(_unknownId);
            }
        }

        return ids;
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids) => ids switch
    {
        int[] array => DecodeSpan(array),
        List<int> list => DecodeSpan(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(list)),
        _ => string.Concat(ids.Select(i => (uint)i < (uint)Vocabulary.Length ? Vocabulary[i] : '�')),
    };

    string ITokenizer.Decode(ReadOnlySpan<int> ids) => DecodeSpan(ids);

    // One character per id, gathered in a stack or pooled buffer and copied once into the result.
    private string DecodeSpan(ReadOnlySpan<int> ids)
    {
        string vocabulary = Vocabulary;
        char[]? rented = null;
        Span<char> chars = ids.Length <= 256 ? stackalloc char[256] : (rented = System.Buffers.ArrayPool<char>.Shared.Rent(ids.Length));
        try
        {
            chars = chars[..ids.Length];
            for (int i = 0; i < chars.Length; i++)
            {
                int id = ids[i];
                chars[i] = (uint)id < (uint)vocabulary.Length ? vocabulary[id] : '�';
            }

            return new string(chars);
        }
        finally
        {
            if (rented is not null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(rented);
            }
        }
    }

    /// <summary>Saves the alphabet and unknown character as JSON.</summary>
    public void Save(string path)
    {
        using var stream = File.Create(path);
        Save(stream);
    }

    /// <summary>Writes the tokenizer as JSON to <paramref name="stream"/>; the stream stays open.</summary>
    public void Save(Stream stream) => Tokenizers.Write(stream, new JsonObject
    {
        ["type"] = "char",
        ["vocabulary"] = Vocabulary,
        ["unknown"] = UnknownCharacter?.ToString(),
    });

    /// <summary>Loads a tokenizer written by <see cref="Save(string)"/>.</summary>
    public static CharTokenizer Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Reads a tokenizer written by <see cref="Save(Stream)"/>.</summary>
    public static CharTokenizer Load(Stream stream) => (CharTokenizer)Tokenizers.Load(stream);
}

/// <summary>
/// One token per word, number or punctuation mark of a fixed vocabulary. Text is split into special tokens in angle
/// brackets (such as <c>&lt;sum&gt;</c>), runs of letters and digits, and single punctuation marks; words outside the
/// vocabulary become <see cref="UnknownToken"/>. Decoding puts a space before every token except closing punctuation,
/// so decoding a sequence piece by piece gives the same text as decoding it at once (streaming-safe).
/// </summary>
public sealed partial class WordTokenizer : ITokenizer
{
    private static readonly System.Buffers.SearchValues<char> ClosingPunctuation = System.Buffers.SearchValues.Create(".,!?;:%)'");

    private readonly Dictionary<string, int> _index;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _lookup;   // created once: words looked up by span
    private readonly int _unknownId;

    /// <summary>Creates the tokenizer for <paramref name="vocabulary"/> (id = position); the unknown token is added if missing.</summary>
    public WordTokenizer(IEnumerable<string> vocabulary, string unknownToken = "<unk>", bool lowercase = true)
    {
        var words = vocabulary.Distinct().ToList();
        if (!words.Contains(unknownToken))
        {
            words.Add(unknownToken);
        }

        Vocabulary = words;
        _index = words.Select((w, i) => (w, i)).ToDictionary(p => p.w, p => p.i);
        _lookup = _index.GetAlternateLookup<ReadOnlySpan<char>>();
        _unknownId = _index[unknownToken];
        UnknownToken = unknownToken;
        Lowercase = lowercase;
    }

    /// <summary>Builds a vocabulary from sample texts: <paramref name="specials"/> first, then words by frequency (at least <paramref name="minCount"/> uses).</summary>
    public static WordTokenizer FromTexts(IEnumerable<string> texts, IEnumerable<string> specials, int minCount = 1, bool lowercase = true)
    {
        var counts = new Dictionary<string, int>();
        var lookup = counts.GetAlternateLookup<ReadOnlySpan<char>>();
        foreach (var text in texts)
        {
            foreach (var word in SplitSpans(text, lowercase))
            {
                System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(lookup, word, out _)++;     // a string only for a new word
            }
        }

        var first = specials.ToList();
        return new WordTokenizer([.. first, .. counts.Where(p => p.Value >= minCount && !first.Contains(p.Key))
            .OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal).Select(p => p.Key)], lowercase: lowercase);
    }

    /// <summary>The words; word i has id i.</summary>
    public IReadOnlyList<string> Vocabulary { get; }

    /// <summary>The stand-in for words outside the vocabulary.</summary>
    public string UnknownToken { get; }

    /// <summary>Whether text is lower-cased before lookup.</summary>
    public bool Lowercase { get; }

    /// <inheritdoc />
    public int VocabularySize => Vocabulary.Count;

    /// <summary>The id of a word (or of the unknown token).</summary>
    public int this[string word] => _index.TryGetValue(word, out int id) ? id : _unknownId;

    /// <summary>Splits text into the tokenizer's units without looking them up.</summary>
    public static IEnumerable<string> Split(string text, bool lowercase = true) =>
        Pattern().Matches(lowercase ? text.ToLowerInvariant() : text).Select(m => m.Value);

    /// <summary>
    /// The units of <paramref name="text"/> as <see cref="Split"/> finds them, as slices of one pooled lower-cased copy
    /// (no string per unit). Enumerate it once with <c>foreach</c>, which returns the copy to the pool.
    /// </summary>
    public static Units SplitSpans(string text, bool lowercase = true) => new(text, lowercase);

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>();
        foreach (var word in SplitSpans(text, Lowercase))
        {
            ids.Add(_lookup.TryGetValue(word, out int id) ? id : _unknownId);
        }

        return ids;
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids)
    {
        var text = new System.Text.StringBuilder();
        foreach (int id in ids)
        {
            Append(text, id);
        }

        return text.ToString();
    }

    // The ids read in place (the interface's default would copy them to an array first).
    string ITokenizer.Decode(ReadOnlySpan<int> ids)
    {
        var text = new System.Text.StringBuilder();
        foreach (int id in ids)
        {
            Append(text, id);
        }

        return text.ToString();
    }

    private void Append(System.Text.StringBuilder text, int id)
    {
        string word = (uint)id < (uint)Vocabulary.Count ? Vocabulary[id] : UnknownToken;
        if (!(word.Length == 1 && ClosingPunctuation.Contains(word[0])))
        {
            text.Append(' ');
        }

        text.Append(word);
    }

    /// <summary>Saves the vocabulary, unknown token and lower-casing setting as JSON.</summary>
    public void Save(string path)
    {
        using var stream = File.Create(path);
        Save(stream);
    }

    /// <summary>Writes the tokenizer as JSON to <paramref name="stream"/>; the stream stays open.</summary>
    public void Save(Stream stream) => Tokenizers.Write(stream, new JsonObject
    {
        ["type"] = "word",
        ["words"] = new JsonArray([.. Vocabulary.Select(w => (JsonNode)w)]),
        ["unknown"] = UnknownToken,
        ["lowercase"] = Lowercase,
    });

    /// <summary>Loads a tokenizer written by <see cref="Save(string)"/>.</summary>
    public static WordTokenizer Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Reads a tokenizer written by <see cref="Save(Stream)"/>.</summary>
    public static WordTokenizer Load(Stream stream) => (WordTokenizer)Tokenizers.Load(stream);

    [System.Text.RegularExpressions.GeneratedRegex(@"<[\w|/]+>|\w+|[^\w\s]")]
    private static partial System.Text.RegularExpressions.Regex Pattern();

    /// <summary>The units <see cref="SplitSpans"/> finds, enumerated once with <c>foreach</c> (which returns the pooled copy).</summary>
    public ref struct Units
    {
        private readonly char[]? _rented;
        private readonly ReadOnlySpan<char> _text;
        private System.Text.RegularExpressions.Regex.ValueMatchEnumerator _matches;

        internal Units(string text, bool lowercase)
        {
            if (lowercase)
            {
                // Lower-cased as string.ToLowerInvariant does (same mapping, same length).
                _rented = System.Buffers.ArrayPool<char>.Shared.Rent(text.Length);
                _text = _rented.AsSpan(0, text.AsSpan().ToLowerInvariant(_rented));
            }
            else
            {
                _text = text;
            }

            _matches = Pattern().EnumerateMatches(_text);
        }

        /// <summary>The current unit.</summary>
        public readonly ReadOnlySpan<char> Current => _text.Slice(_matches.Current.Index, _matches.Current.Length);

        /// <summary>The enumerator itself (for <c>foreach</c>).</summary>
        public readonly Units GetEnumerator() => this;

        /// <summary>Moves to the next unit; false after the last.</summary>
        public bool MoveNext() => _matches.MoveNext();

        /// <summary>Returns the pooled lower-cased copy.</summary>
        public readonly void Dispose()
        {
            if (_rented is not null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(_rented);
            }
        }
    }
}

/// <summary>Reads either tokenizer from the JSON written by <see cref="CharTokenizer.Save(string)"/> or <see cref="WordTokenizer.Save(string)"/>.</summary>
public static class Tokenizers
{
    /// <summary>Loads a character or word tokenizer from a file.</summary>
    public static ITokenizer Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Reads a character or word tokenizer from <paramref name="stream"/>.</summary>
    public static ITokenizer Load(Stream stream)
    {
        var json = JsonNode.Parse(stream) as JsonObject ?? throw new InvalidDataException("A tokenizer file must contain a JSON object.");
        return (string?)json["type"] switch
        {
            "char" => new CharTokenizer((string)json["vocabulary"]!, ((string?)json["unknown"]) is { Length: 1 } u ? u[0] : null),
            "word" => new WordTokenizer([.. json["words"]!.AsArray().Select(w => (string)w!)], (string)json["unknown"]!, (bool)json["lowercase"]!),
            var type => throw new InvalidDataException($"Unknown tokenizer type '{type}'."),
        };
    }

    /// <summary>Saves a <see cref="CharTokenizer"/> or <see cref="WordTokenizer"/>.</summary>
    public static void Save(ITokenizer tokenizer, Stream stream)
    {
        switch (tokenizer)
        {
            case CharTokenizer c: c.Save(stream); break;
            case WordTokenizer w: w.Save(stream); break;
            default: throw new NotSupportedException($"{tokenizer.GetType().Name} cannot be saved; only CharTokenizer and WordTokenizer can.");
        }
    }

    internal static void Write(Stream stream, JsonObject json)
    {
        using var writer = new Utf8JsonWriter(stream, new JsonWriterOptions { Indented = true });
        json.WriteTo(writer);
    }
}

/// <summary>
/// A text as a model takes it: its token ids with, for each token, its type id (which text of a pair it belongs to),
/// whether it is a special token, where it comes from in the text, and whether the model attends to it. Every array has
/// one entry per token.
/// </summary>
public sealed class TokenizerEncoding
{
    /// <summary>The token ids.</summary>
    public required int[] Ids { get; init; }

    /// <summary>Each token's type id (0 for the first text of a pair, 1 for the second; as the post-processor sets them).</summary>
    public required int[] TypeIds { get; init; }

    /// <summary>
    /// Each token's range in its text, as UTF-16 indexes [Start, End); a token added by the tokenizer (a special token)
    /// has (0, 0).
    /// </summary>
    public required (int Start, int End)[] Offsets { get; init; }

    /// <summary>1 for a token the model attends to, 0 for padding.</summary>
    public required int[] AttentionMask { get; init; }

    /// <summary>1 for a special token the tokenizer added, else 0.</summary>
    public required int[] SpecialTokensMask { get; init; }

    /// <summary>Which text each token comes from: 0 or 1 for the texts of a pair, -1 for a token the tokenizer added.</summary>
    public required int[] SequenceIds { get; init; }

    /// <summary>Each token's vocabulary entry (as <see cref="ITokenizer.TokenOf"/> gives it), or empty when the encoder gives none.</summary>
    public string[] Tokens { get; init; } = [];

    /// <summary>The number of tokens.</summary>
    public int Count => Ids.Length;

    /// <summary>The encoding of ids from one text (sequence <paramref name="sequence"/>), all attended, none special.</summary>
    public static TokenizerEncoding Of(int[] ids, (int Start, int End)[] offsets, int sequence = 0, int typeId = 0)
    {
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentNullException.ThrowIfNull(offsets);
        if (offsets.Length != ids.Length)
        {
            throw new ArgumentException($"{offsets.Length} offsets for {ids.Length} ids.", nameof(offsets));
        }

        return new TokenizerEncoding
        {
            Ids = ids, Offsets = offsets, TypeIds = Filled(ids.Length, typeId), AttentionMask = Filled(ids.Length, 1),
            SpecialTokensMask = new int[ids.Length], SequenceIds = Filled(ids.Length, sequence),
        };

        static int[] Filled(int length, int value)
        {
            var values = new int[length];
            Array.Fill(values, value);
            return values;
        }
    }
}
