// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
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
    /// Which end a truncated text loses its tokens from: "right" (its last tokens go) or "left" (its first tokens go);
    /// null: the tokenizer.json's <c>truncation.direction</c>, else "right".
    /// </summary>
    public string? TruncationDirection { get; init; }

    /// <summary>
    /// "none", "longest" (a batch to its longest encoding) or "max_length" (to <see cref="MaxLength"/>); null: the
    /// tokenizer.json's padding, else none.
    /// </summary>
    public string? Padding { get; init; }

    /// <summary>The id padding adds; null: the tokenizer.json's <c>padding.pad_id</c>, else the tokenizer's padding token.</summary>
    public int? PadId { get; init; }

    /// <summary>
    /// Whether each token's offsets and vocabulary entry are found (true by default). False, for a caller that needs only
    /// ids, type ids and masks: a single text goes through the tokenizer's plain encode (its piece cache, no strings), every
    /// offset is (0, 0) and <see cref="TokenizerEncoding.Tokens"/> is empty; the ids, truncation, special tokens and
    /// padding are the same. A pair is always encoded in detail (tokenizers' pair truncation depends on its pre-tokens).
    /// </summary>
    public bool Offsets { get; init; } = true;
}


/// <summary>How a <see cref="CharTokenizer"/> reads characters, and what it does with those outside its alphabet.</summary>
public sealed record CharTokenizerOptions
{
    /// <summary>
    /// Whether a character is a whole Unicode code point: a surrogate pair is one character and one id, and a lone
    /// surrogate in a text reads as U+FFFD (as <see cref="string.EnumerateRunes"/> reads it). False (the default): one
    /// character per UTF-16 unit, as before.
    /// </summary>
    public bool CodePoints { get; init; }

    /// <summary>
    /// The stand-in for characters outside the alphabet (one character of the alphabet: one code point with
    /// <see cref="CodePoints"/>), or null to drop them (the default).
    /// </summary>
    public string? UnknownCharacter { get; init; }
}

/// <summary>
/// One token per character of a fixed alphabet (the vocabulary string): per UTF-16 unit by default, or per whole code
/// point (<see cref="CharTokenizerOptions.CodePoints"/>, a surrogate pair one token). Characters outside the alphabet are
/// dropped when encoding, or replaced by the unknown character when it is part of the alphabet.
/// </summary>
public sealed class CharTokenizer : ISavableTokenizer
{
    private readonly Dictionary<char, int> _index;              // per UTF-16 unit (empty with code points)
    private readonly Dictionary<int, int>? _points;             // per code point, or null
    private readonly string[]? _characters;                     // each id's code point as text, or null
    private readonly int _unknownId = -1;                      // the id of the unknown character (-1: none), looked up once

    /// <summary>Creates the tokenizer for the characters of <paramref name="vocabulary"/> (id = position).</summary>
    public CharTokenizer(string vocabulary, char? unknownCharacter = null)
    {
        Vocabulary = vocabulary;
        _index = vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
        Options = new CharTokenizerOptions();
        if (unknownCharacter is { } u && _index.TryGetValue(u, out int unknownId))
        {
            UnknownCharacter = u;
            _unknownId = unknownId;
            Options = Options with { UnknownCharacter = u.ToString() };
        }
    }

    /// <summary>
    /// Creates the tokenizer for the characters of <paramref name="vocabulary"/> read as <paramref name="options"/> say
    /// (id = the character's position: its code point's, with <see cref="CharTokenizerOptions.CodePoints"/>).
    /// </summary>
    /// <exception cref="ArgumentException">A character appears twice, or the unknown character is not one character of the alphabet.</exception>
    public CharTokenizer(string vocabulary, CharTokenizerOptions options)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        ArgumentNullException.ThrowIfNull(options);
        Vocabulary = vocabulary;
        Options = options;
        if (!options.CodePoints)
        {
            _index = vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
            if (options.UnknownCharacter is { } u)
            {
                _unknownId = u.Length == 1 && _index.TryGetValue(u[0], out int id) ? id
                    : throw new ArgumentException($"The unknown character '{u}' is not one character of the alphabet.", nameof(options));
                UnknownCharacter = u[0];
            }

            return;
        }

        _index = [];
        _points = [];
        var characters = new List<string>(vocabulary.Length);
        for (int i = 0; i < vocabulary.Length; i++)
        {
            int point = CodePointAt(vocabulary, ref i, vocabularyText: true);
            if (!_points.TryAdd(point, characters.Count))
            {
                throw new ArgumentException($"The character U+{point:X4} appears twice in the alphabet.", nameof(vocabulary));
            }

            characters.Add(char.IsSurrogate(vocabulary[i]) && point > 0xFFFF ? vocabulary.Substring(i - 1, 2) : vocabulary[i].ToString());
        }

        _characters = [.. characters];
        if (options.UnknownCharacter is { } unknown)
        {
            int at = 0;
            int point = unknown.Length is 1 or 2 ? CodePointAt(unknown, ref at, vocabularyText: true) : -1;
            _unknownId = point >= 0 && at == unknown.Length - 1 && _points.TryGetValue(point, out int id) ? id
                : throw new ArgumentException($"The unknown character '{unknown}' is not one code point of the alphabet.", nameof(options));
            UnknownCharacter = unknown.Length == 1 ? unknown[0] : null;
        }
    }

    /// <summary>
    /// Creates the tokenizer for <paramref name="characters"/>, one character each (id = position): a whole code point
    /// each with <see cref="CharTokenizerOptions.CodePoints"/>.
    /// </summary>
    /// <exception cref="ArgumentException">An entry that is not one character, a character that appears twice, or an unknown character outside the alphabet.</exception>
    public CharTokenizer(IEnumerable<string> characters, CharTokenizerOptions options)
        : this(Joined(characters, options), options)
    {
    }

    /// <summary>The alphabet; character i has id i (with code points, the i-th code point).</summary>
    public string Vocabulary { get; }

    /// <summary>Stand-in for characters outside the alphabet, or null to drop them (or when it is a code point beyond one UTF-16 unit: see <see cref="Options"/>).</summary>
    public char? UnknownCharacter { get; }

    /// <summary>How characters are read and what happens to unknown ones (the unknown character only when it is in the alphabet).</summary>
    public CharTokenizerOptions Options { get; }

    /// <inheritdoc />
    public int VocabularySize => _characters?.Length ?? Vocabulary.Length;

    /// <summary>Whether every character of <paramref name="text"/> is in the alphabet.</summary>
    public bool Covers(string text)
    {
        if (_points is null)
        {
            return text.All(_index.ContainsKey);
        }

        for (int i = 0; i < text.Length; i++)
        {
            if (!_points.ContainsKey(CodePointAt(text, ref i, vocabularyText: false)))
            {
                return false;
            }
        }

        return true;
    }

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        var ids = new List<int>(text.Length);
        if (_points is not null)
        {
            for (int i = 0; i < text.Length; i++)
            {
                if (_points.TryGetValue(CodePointAt(text, ref i, vocabularyText: false), out int id))
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
        _ when _characters is not null => DecodeSpan([.. ids]),
        _ => string.Concat(ids.Select(i => (uint)i < (uint)Vocabulary.Length ? Vocabulary[i] : '\uFFFD')),
    };

    string ITokenizer.Decode(ReadOnlySpan<int> ids) => DecodeSpan(ids);

    // The code point at text[i] (i moved to a pair's second half). A lone surrogate reads as U+FFFD in a text, and as
    // itself in an alphabet (where it can then never match).
    private static int CodePointAt(string text, ref int i, bool vocabularyText)
    {
        char c = text[i];
        if (char.IsHighSurrogate(c) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
        {
            return char.ConvertToUtf32(c, text[++i]);
        }

        return char.IsSurrogate(c) && !vocabularyText ? 0xFFFD : c;
    }

    // The alphabet of one-character entries as one string, each entry checked.
    private static string Joined(IEnumerable<string> characters, CharTokenizerOptions options)
    {
        ArgumentNullException.ThrowIfNull(characters);
        ArgumentNullException.ThrowIfNull(options);
        var text = new System.Text.StringBuilder();
        foreach (string character in characters)
        {
            bool one = character.Length == 1 || options.CodePoints && character.Length == 2 && char.IsSurrogatePair(character[0], character[1]);
            if (!one)
            {
                throw new ArgumentException($"'{character}' is not one {(options.CodePoints ? "code point" : "UTF-16 character")}.", nameof(characters));
            }

            text.Append(character);
        }

        return text.ToString();
    }

    // One character per id (two units for a code point beyond the first plane), gathered in a stack or pooled buffer and
    // copied once into the result.
    private string DecodeSpan(ReadOnlySpan<int> ids)
    {
        string vocabulary = Vocabulary;
        string[]? characters = _characters;
        int most = characters is null ? ids.Length : 2 * ids.Length;
        char[]? rented = null;
        Span<char> chars = most <= 256 ? stackalloc char[256] : (rented = System.Buffers.ArrayPool<char>.Shared.Rent(most));
        try
        {
            int n = 0;
            if (characters is null)
            {
                for (int i = 0; i < ids.Length; i++)
                {
                    int id = ids[i];
                    chars[n++] = (uint)id < (uint)vocabulary.Length ? vocabulary[id] : '\uFFFD';
                }
            }
            else
            {
                for (int i = 0; i < ids.Length; i++)
                {
                    int id = ids[i];
                    if ((uint)id < (uint)characters.Length)
                    {
                        string character = characters[id];
                        chars[n++] = character[0];
                        if (character.Length == 2)
                        {
                            chars[n++] = character[1];
                        }
                    }
                    else
                    {
                        chars[n++] = '\uFFFD';
                    }
                }
            }

            return new string(chars[..n]);
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

    /// <summary>The type its saved JSON names: "char".</summary>
    public string Type => "char";

    /// <summary>Writes the tokenizer as JSON to <paramref name="stream"/>; the stream stays open.</summary>
    public void Save(Stream stream)
    {
        var json = new JsonObject
        {
            ["type"] = "char",
            ["vocabulary"] = Vocabulary,
            ["unknown"] = _characters is null ? UnknownCharacter?.ToString() : _unknownId >= 0 ? _characters[_unknownId] : null,
        };
        if (_characters is not null)
        {
            json["code_points"] = true;                                    // written only when set: the default file is as before
        }

        Tokenizers.Write(stream, json);
    }

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
public sealed partial class WordTokenizer : ISavableTokenizer
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

    /// <summary>The type its saved JSON names: "word".</summary>
    public string Type => "word";

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

/// <summary>
/// A tokenizer that saves itself as a JSON object naming its <see cref="Type"/>, which <see cref="Tokenizers.Load(Stream)"/>
/// reads back through the loader registered for that type (<see cref="TokenizerLoaders"/>).
/// </summary>
public interface ISavableTokenizer : ITokenizer
{
    /// <summary>The type its saved JSON names ("char", "word"): the name of its loader in <see cref="TokenizerLoaders"/>.</summary>
    string Type { get; }

    /// <summary>Writes the tokenizer to <paramref name="stream"/> as one JSON object with "type": <see cref="Type"/>.</summary>
    void Save(Stream stream);
}

/// <summary>
/// A reader of saved tokenizers of one type (<see cref="ISavableTokenizer.Type"/>), registered by its
/// <see cref="IKind.Name"/> (that type) in <see cref="TokenizerLoaders"/>: the tokenizer from its saved JSON object.
/// </summary>
public interface ITokenizerLoader : IKind<JsonObject, ITokenizer>;

/// <summary>
/// The readers of saved tokenizers, by the "type" their JSON names: "char" (<see cref="CharTokenizer"/>) and "word"
/// (<see cref="WordTokenizer"/>) are the library's; a tokenizer of one's own that saves itself
/// (<see cref="ISavableTokenizer"/>) registers its reader with <see cref="Register"/>.
/// </summary>
public static class TokenizerLoaders
{
    private static readonly KindTable<ITokenizerLoader> Table = BuiltIn();

    private static KindTable<ITokenizerLoader> BuiltIn()
    {
        var table = new KindTable<ITokenizerLoader>(nameof(TokenizerLoaders), "Tokenizer type", "add its reader with TokenizerLoaders.Register", comparer: StringComparer.Ordinal,
            unguarded: "a saved tokenizer is read by its own type's reader, whose vocabulary another cannot stand in for");
        table.RegisterDefault(new BuiltInLoader("char", "CharTokenizer: one id per character (or code point)", json =>
            json["code_points"] is JsonValue points && (bool)points
                ? new CharTokenizer((string)json["vocabulary"]!, new CharTokenizerOptions { CodePoints = true, UnknownCharacter = (string?)json["unknown"] })
                : new CharTokenizer((string)json["vocabulary"]!, ((string?)json["unknown"]) is { Length: 1 } u ? u[0] : null)));
        table.RegisterDefault(new BuiltInLoader("word", "WordTokenizer: one id per word of its vocabulary", json =>
            new WordTokenizer([.. json["words"]!.AsArray().Select(w => (string)w!)], (string)json["unknown"]!, (bool)json["lowercase"]!)));
        return table;
    }

    /// <summary>Registers <paramref name="loader"/> under its <see cref="IKind.Name"/> (the type it reads); under a library type it takes the library's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ITokenizerLoader loader) => Table.Register(loader, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's reader of <paramref name="type"/> (a library type gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string type) => Table.Unregister(type);

    /// <summary>The types a reader is registered for.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The reader of <paramref name="type"/>, or null.</summary>
    public static ITokenizerLoader? Find(ReadOnlySpan<char> type) => Table.Find(type);

    /// <summary>Who registered the reader of <paramref name="type"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string type) => Table.Origin(type);

    private sealed class BuiltInLoader(string name, string summary, Func<JsonObject, ITokenizer> read) : ITokenizerLoader
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public ITokenizer Create(JsonObject request) => read(request);
    }
}

/// <summary>Reads a saved tokenizer (<see cref="ISavableTokenizer"/>) from its JSON, through the reader of its type (<see cref="TokenizerLoaders"/>).</summary>
public static class Tokenizers
{
    /// <summary>Loads a saved tokenizer from a file.</summary>
    public static ITokenizer Load(string path)
    {
        using var stream = File.OpenRead(path);
        return Load(stream);
    }

    /// <summary>Reads a saved tokenizer from <paramref name="stream"/>.</summary>
    /// <exception cref="InvalidDataException">The JSON is not an object, or names a type no reader is registered for.</exception>
    public static ITokenizer Load(Stream stream)
    {
        var json = JsonNode.Parse(stream) as JsonObject ?? throw new InvalidDataException("A tokenizer file must contain a JSON object.");
        var type = (string?)json["type"];
        return (type is null ? null : TokenizerLoaders.Find(type))?.Create(json) ?? throw new InvalidDataException($"Unknown tokenizer type '{type}'.");
    }

    /// <summary>Saves <paramref name="tokenizer"/>, one that saves itself (<see cref="ISavableTokenizer"/>: the library's <see cref="CharTokenizer"/> and <see cref="WordTokenizer"/>).</summary>
    /// <exception cref="NotSupportedException">The tokenizer does not save itself.</exception>
    public static void Save(ITokenizer tokenizer, Stream stream)
    {
        ArgumentNullException.ThrowIfNull(tokenizer);
        if (tokenizer is not ISavableTokenizer savable)
        {
            throw new NotSupportedException($"{tokenizer.GetType().Name} cannot be saved; only a tokenizer that saves itself (ISavableTokenizer: CharTokenizer, WordTokenizer) can.");
        }

        savable.Save(stream);
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
