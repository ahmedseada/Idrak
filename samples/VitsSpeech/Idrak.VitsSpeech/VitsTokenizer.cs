// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.VitsSpeech;

/// <summary>
/// The character tokenizer of a VITS checkpoint (transformers' <c>VitsTokenizer</c>): <c>vocab.json</c> maps each
/// symbol to its id; <c>tokenizer_config.json</c> (optional) says whether the text is lower-cased
/// (<c>normalize</c>), whether a blank is put between symbols (<c>add_blank</c>, the pad token's id), whether the
/// checkpoint expects phonemes (<c>phonemize</c>: refused, the library carries no phonemizer) or romanised text
/// (<c>is_uroman</c>: non-Latin text is romanised first by the <see cref="TextRomanizers"/> entry that covers its
/// script, Arabic built in; a script none covers is refused). Each flag can be changed after loading.
/// </summary>
public sealed class VitsTokenizer
{
    private readonly Dictionary<string, int> _ids;
    private readonly string[] _symbols;

    /// <summary>A tokenizer over <paramref name="vocabulary"/> (symbol to id).</summary>
    public VitsTokenizer(IReadOnlyDictionary<string, int> vocabulary)
    {
        ArgumentNullException.ThrowIfNull(vocabulary);
        _ids = new Dictionary<string, int>(vocabulary, StringComparer.Ordinal);
        int size = _ids.Count == 0 ? 0 : _ids.Values.Max() + 1;
        _symbols = new string[size];
        foreach (var (symbol, id) in _ids)
        {
            if (id < 0)
            {
                throw new InvalidDataException($"vocab.json gives '{symbol}' the id {id}.");
            }

            _symbols[id] = symbol;
        }
    }

    /// <summary>The symbols by id (null where no symbol has that id).</summary>
    public IReadOnlyList<string?> Symbols => _symbols;

    /// <summary>The vocabulary's size (the largest id + 1).</summary>
    public int Size => _symbols.Length;

    /// <summary>Lower-case the text and drop symbols the vocabulary lacks (transformers' <c>normalize</c>).</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>Put <see cref="BlankId"/> between the symbols and at both ends (transformers' <c>add_blank</c>; MMS-TTS checkpoints set it).</summary>
    public bool AddBlank { get; set; }

    /// <summary>The blank's id (the pad token's).</summary>
    public int BlankId { get; set; }

    /// <summary>The checkpoint expects phonemes: encoding is refused, since the library carries no phonemizer.</summary>
    public bool Phonemize { get; set; }

    /// <summary>The checkpoint expects romanised (uroman) text: non-Latin text is romanised before it is tokenized.</summary>
    public bool IsUroman { get; set; }

    /// <summary>
    /// The romanizer for <see cref="IsUroman"/> checkpoints: a name in <see cref="TextRomanizers"/>, or null (the default)
    /// for the one that covers the text's script.
    /// </summary>
    public string? Romanizer { get; set; }

    /// <summary>Reads <c>vocab.json</c> and, when present, <c>tokenizer_config.json</c> from <paramref name="folder"/>.</summary>
    /// <exception cref="FileNotFoundException">There is no vocab.json.</exception>
    public static VitsTokenizer Read(string folder)
    {
        ArgumentException.ThrowIfNullOrEmpty(folder);
        string vocabPath = Path.Combine(folder, "vocab.json");
        if (!File.Exists(vocabPath))
        {
            throw new FileNotFoundException($"{folder} has no vocab.json (a VITS checkpoint's symbol-to-id map).", vocabPath);
        }

        var vocab = JsonNode.Parse(File.ReadAllText(vocabPath))?.AsObject() ?? throw new InvalidDataException($"{vocabPath} is not a JSON object.");
        var tokenizer = new VitsTokenizer(vocab.ToDictionary(p => p.Key, p => p.Value!.GetValue<int>(), StringComparer.Ordinal));
        string configPath = Path.Combine(folder, "tokenizer_config.json");
        if (File.Exists(configPath))
        {
            var config = JsonNode.Parse(File.ReadAllText(configPath))?.AsObject() ?? throw new InvalidDataException($"{configPath} is not a JSON object.");
            tokenizer.Normalize = config["normalize"]?.GetValue<bool>() ?? true;
            tokenizer.AddBlank = config["add_blank"]?.GetValue<bool>() ?? true;            // transformers' default when the file exists
            tokenizer.Phonemize = config["phonemize"]?.GetValue<bool>() ?? false;
            tokenizer.IsUroman = config["is_uroman"]?.GetValue<bool>() ?? false;
            if (config["pad_token"] is JsonValue pad && pad.TryGetValue(out string? padToken) && padToken is not null && tokenizer._ids.TryGetValue(padToken, out int padId))
            {
                tokenizer.BlankId = padId;
            }
        }

        return tokenizer;
    }

    /// <summary>The ids of <paramref name="text"/> (normalised and interspersed with blanks as the flags say).</summary>
    /// <exception cref="NotSupportedException">The checkpoint expects phonemes, or romanised text and the text is not Latin.</exception>
    public int[] Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (Phonemize)
        {
            throw new NotSupportedException("This checkpoint's tokenizer_config.json sets \"phonemize\": the library carries no phonemizer; give it a checkpoint over characters (MMS-TTS), or set VitsTokenizer.Phonemize false to encode the characters as they are.");
        }

        if (IsUroman && HasNonLatin(text))
        {
            text = Romanize(text);
        }

        var symbols = new List<int>(text.Length);
        foreach (var rune in text.EnumerateRunes())
        {
            string symbol = rune.ToString();
            if (Normalize)
            {
                symbol = symbol.ToLowerInvariant();
                if (_ids.TryGetValue(symbol, out int lowered))
                {
                    symbols.Add(lowered);
                }

                continue;                                                        // a symbol outside the vocabulary is dropped
            }

            symbols.Add(_ids.TryGetValue(symbol, out int id) ? id
                : throw new ArgumentException($"'{symbol}' is not in the vocabulary (set Normalize to drop such symbols).", nameof(text)));
        }

        if (Normalize)
        {
            TrimBlanks(symbols);
        }

        if (!AddBlank)
        {
            return [.. symbols];
        }

        var ids = new int[symbols.Count * 2 + 1];
        Array.Fill(ids, BlankId);
        for (int i = 0; i < symbols.Count; i++)
        {
            ids[2 * i + 1] = symbols[i];
        }

        return ids;
    }

    // transformers strips the normalised text: leading and trailing spaces go.
    private void TrimBlanks(List<int> symbols)
    {
        if (!_ids.TryGetValue(" ", out int space))
        {
            return;
        }

        while (symbols.Count > 0 && symbols[^1] == space)
        {
            symbols.RemoveAt(symbols.Count - 1);
        }

        int start = 0;
        while (start < symbols.Count && symbols[start] == space)
        {
            start++;
        }

        symbols.RemoveRange(0, start);
    }

    /// <summary>
    /// <paramref name="text"/> as an <see cref="IsUroman"/> checkpoint reads it: romanised by <see cref="Romanizer"/> (or
    /// the registered romanizer that covers its script) when it holds non-Latin letters, else as it is.
    /// </summary>
    /// <exception cref="NotSupportedException">No registered romanizer covers the text's script.</exception>
    public string Romanize(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!HasNonLatin(text))
        {
            return text;
        }

        var romanizer = Romanizer is { } name ? TextRomanizers.Get(name) : TextRomanizers.For(text)
            ?? throw new NotSupportedException("This checkpoint's tokenizer_config.json sets \"is_uroman\": it expects romanised text, and no registered romanizer covers the "
                + $"text's script (registered: {string.Join(", ", TextRomanizers.Names)}); add one with TextRomanizers.Register, or give it Latin text.");
        return romanizer.Romanize(text);
    }

    // A letter past Latin Extended-B: a script uroman would romanise.
    internal static bool IsNonLatinLetter(Rune rune) => Rune.IsLetter(rune) && rune.Value > 0x024F;

    private static bool HasNonLatin(string text)
    {
        foreach (var rune in text.EnumerateRunes())
        {
            if (IsNonLatinLetter(rune))
            {
                return true;
            }
        }

        return false;
    }
}
