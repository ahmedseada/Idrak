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
/// One token per character of a fixed alphabet (the vocabulary string). Characters outside the alphabet are
/// dropped when encoding, or replaced by <see cref="UnknownCharacter"/> when it is part of the alphabet.
/// </summary>
public sealed class CharTokenizer : ITokenizer
{
    private readonly Dictionary<char, int> _index;

    /// <summary>Creates the tokenizer for the characters of <paramref name="vocabulary"/> (id = position).</summary>
    public CharTokenizer(string vocabulary, char? unknownCharacter = null)
    {
        Vocabulary = vocabulary;
        _index = vocabulary.Select((c, i) => (c, i)).ToDictionary(p => p.c, p => p.i);
        UnknownCharacter = unknownCharacter is { } u && _index.ContainsKey(u) ? u : null;
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
            else if (UnknownCharacter is { } u)
            {
                ids.Add(_index[u]);
            }
        }

        return ids;
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids) => string.Concat(ids.Select(i => (uint)i < (uint)Vocabulary.Length ? Vocabulary[i] : '�'));

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
    private readonly Dictionary<string, int> _index;

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
    public int this[string word] => _index.TryGetValue(word, out int id) ? id : _index[UnknownToken];

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
        var lookup = _index.GetAlternateLookup<ReadOnlySpan<char>>();
        foreach (var word in SplitSpans(text, Lowercase))
        {
            ids.Add(lookup.TryGetValue(word, out int id) ? id : _index[UnknownToken]);
        }

        return ids;
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids)
    {
        var text = new System.Text.StringBuilder();
        foreach (int id in ids)
        {
            string word = (uint)id < (uint)Vocabulary.Count ? Vocabulary[id] : UnknownToken;
            if (!(word.Length == 1 && ".,!?;:%)'".Contains(word[0])))
            {
                text.Append(' ');
            }

            text.Append(word);
        }

        return text.ToString();
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
