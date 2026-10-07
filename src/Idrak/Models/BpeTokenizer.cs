// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;

namespace Idrak.Models;

/// <summary>
/// A byte-pair-encoding tokenizer read from a Hugging Face <c>tokenizer.json</c>: byte-level BPE (GPT-2, Llama 3, Qwen,
/// …) and SentencePiece-style BPE with the "▁" word marker and byte fallback (Llama 2, Mistral, Gemma, …). Special
/// (added) tokens are matched exactly before anything else. <see cref="Encode"/> adds no special tokens itself (chat
/// templates write them), like <c>add_special_tokens=False</c>. Pieces of the file it does not know are reported, not
/// guessed; other normalizer, pre-tokenizer and decoder types plug in through <see cref="TokenizerComponents"/>.
/// </summary>
public sealed class BpeTokenizer : ITokenizer
{
    private static readonly string Gpt2Pattern = @"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+";
    private static readonly char[] ByteToChar = BuildByteMap();
    private static readonly Dictionary<char, byte> CharToByte = ByteToChar.Select((c, b) => (c, b)).ToDictionary(p => p.c, p => (byte)p.b);

    private readonly Dictionary<string, int> _vocabulary;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _vocabularySpans;
    private static readonly UTF8Encoding StrictUtf8 = new(false, throwOnInvalidBytes: true);
    // Tokenizer patterns run on every encode: compiled to IL where the runtime allows it (interpreted under native AOT).
    private const RegexOptions Fast = RegexOptions.Compiled | RegexOptions.CultureInvariant;
    private const int CacheLimit = 100_000;                                    // cached pieces
    private const int StackChars = 256;                                        // pieces up to this long stay on the stack

    private string[] _tokens;
    private readonly (string Left, string Right)[] _merges;
    private Dictionary<(string, string), int>? _ranks;                        // built only when merges must run on strings

    // The merges by token ids: (left id << 32 | right id) → (rank, merged id); null when some merge's parts or result
    // are not tokens (then merges run on strings).
    private readonly Dictionary<long, (int Rank, int Merged)>? _pairs;
    private readonly List<(string Content, int Id)> _added;
    private readonly HashSet<int> _special;
    private readonly HashSet<int> _addedIds;
    private readonly Regex? _addedPattern;
    private readonly List<Func<List<string>, bool, List<string>>> _preTokenizers = [];   // (pieces, text starts the input)
    private readonly List<Func<string, string>> _normalizers = [];
    private readonly List<(string Type, Func<List<string>, List<string>> Run, Func<string, string>? Map)> _decoders = [];

    // The normalizers when each prepends a string or replaces one character by another (SentencePiece's "▁" scheme): they
    // run in place on a buffer; null when some normalizer is anything else.
    private List<(string? Prepend, char From, char To)>? _normalizerSteps = [];

    // The pre-tokenizer when it is the only one and rewrites text: Metaspace, or ByteLevel with add_prefix_space (its
    // regex, null without use_regex); these run on spans.
    private (string Replacement, string Scheme, bool Split)? _metaspace;
    private (bool Prefix, Regex? Pattern)? _prefixByteLevel;

    // Decoding without token lists (see DecodeTable), or null; built on the first decode.
    private DecodeTable? _decodeTable;
    private readonly int _fusedDecoders = -1;                                 // decoders the table covers, -1: none

    // Pre-tokenizers that only split (no stage rewrites the text) as stages over ranges of one string, or null when
    // some stage rewrites text (then the string pipeline above runs).
    private List<(Regex Pattern, string Behavior, bool Invert)>? _splits = [];

    // Encoded pieces (shared by concurrent encodes); looked up by span, so a hit allocates nothing.
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, int[]> _cache = new();
    private readonly System.Collections.Concurrent.ConcurrentDictionary<string, int[]>.AlternateLookup<ReadOnlySpan<char>> _cacheSpans;

    // Byte-level decoding: the bytes of every token in one array (token id's bytes at [_byteStart[id], _byteStart[id + 1])).
    private byte[]? _tokenBytes;
    private int[]? _byteStart;

    // Byte-level encoding: the id of each byte's character (-1 when it is not a token).
    private readonly int[] _byteIds = new int[256];

    // Byte fallback: the id of each byte's <0xNN> token (-1 when it is missing).
    private readonly int[] _fallbackIds = new int[256];

    private readonly bool _byteLevel;
    private readonly bool _byteFallback;
    private readonly bool _ignoreMerges;
    private readonly int? _unknown;

    private BpeTokenizer(JsonObject json, Dictionary<string, int> vocabulary, (string Left, string Right)[] merges)
    {
        var model = json["model"]?.AsObject() ?? throw new InvalidDataException("tokenizer.json has no model.");
        if ((string?)model["type"] != "BPE")
        {
            throw new NotSupportedException($"Tokenizer model '{model["type"]}' is not supported (BPE).");
        }

        _vocabulary = vocabulary;
        _vocabularySpans = _vocabulary.GetAlternateLookup<ReadOnlySpan<char>>();
        _cacheSpans = _cache.GetAlternateLookup<ReadOnlySpan<char>>();
        _merges = merges;
        _byteFallback = (bool?)model["byte_fallback"] ?? false;
        _ignoreMerges = (bool?)model["ignore_merges"] ?? false;
        _unknown = model["unk_token"] is JsonValue unk && _vocabulary.TryGetValue((string)unk!, out int u) ? u : null;
        var addedTokens = json["added_tokens"]?.AsArray() ?? [];
        _added = [.. addedTokens.Select(t => ((string)t!["content"]!, (int)t["id"]!))];
        _special = [.. addedTokens.Where(t => (bool?)t!["special"] ?? false).Select(t => (int)t!["id"]!)];
        _addedIds = [.. _added.Select(a => a.Id)];
        foreach (var (content, id) in _added)
        {
            _vocabulary[content] = id;
        }

        int size = 0;
        foreach (int id in _vocabulary.Values)
        {
            size = Math.Max(size, id + 1);
        }

        _tokens = new string[size];
        foreach (var (token, id) in _vocabulary)
        {
            _tokens[id] = token;
        }

        // Merged tokens are looked up by span (the two parts written side by side), so no string is built per merge.
        _pairs = new Dictionary<long, (int Rank, int Merged)>(merges.Length);
        Span<char> joined = stackalloc char[StackChars];
        for (int r = 0; r < merges.Length; r++)
        {
            var (left, right) = merges[r];
            int length = left.Length + right.Length;
            var both = length <= StackChars ? joined[..length] : new char[length];
            left.CopyTo(both);
            right.CopyTo(both[left.Length..]);
            if (!_vocabulary.TryGetValue(left, out int a) || !_vocabulary.TryGetValue(right, out int b) || !_vocabularySpans.TryGetValue(both, out int merged))
            {
                _pairs = null;
                break;
            }

            _pairs.TryAdd(((long)a << 32) | (uint)b, (r, merged));                // the first (lowest-rank) duplicate wins
        }

        if (_added.Count > 0)
        {
            _addedPattern = new Regex(string.Join('|', _added.Select(a => a.Content).OrderByDescending(c => c.Length).Select(Regex.Escape)), Fast);
        }

        AddNormalizer(json["normalizer"]);
        _byteLevel = AddPreTokenizer(json["pre_tokenizer"]);
        if (_preTokenizers.Count != 1)
        {
            (_metaspace, _prefixByteLevel) = (null, null);
        }

        AddDecoder(json["decoder"]);
        _fusedDecoders = FusedDecoders();
        if (_byteFallback)
        {
            for (int b = 0; b < 256; b++)
            {
                _fallbackIds[b] = _vocabulary.TryGetValue($"<0x{b:X2}>", out int id) ? id : -1;
            }
        }

        if (_byteLevel)
        {
            Span<char> one = stackalloc char[1];
            for (int b = 0; b < 256; b++)
            {
                one[0] = ByteToChar[b];
                _byteIds[b] = _vocabularySpans.TryGetValue(one, out int id) ? id : -1;
            }

            BuildTokenBytes();
        }
    }

    /// <summary>Reads tokenizer.json from a file or a model folder.</summary>
    public static BpeTokenizer Load(string path)
    {
        if (Directory.Exists(path))
        {
            path = Path.Combine(path, "tokenizer.json");
        }

        // The vocabulary and merges (most of the file) are read straight from the document; only the small rest becomes
        // a JSON object tree.
        using var document = JsonDocument.Parse(File.ReadAllBytes(path), new JsonDocumentOptions { MaxDepth = 256 });
        var root = document.RootElement;
        var rest = new JsonObject();
        Dictionary<string, int>? vocabulary = null;
        (string, string)[] merges = [];
        foreach (var property in root.EnumerateObject())
        {
            if (property.NameEquals("model") && property.Value.ValueKind == JsonValueKind.Object)
            {
                var model = new JsonObject();
                foreach (var field in property.Value.EnumerateObject())
                {
                    if (field.NameEquals("vocab") && field.Value.ValueKind == JsonValueKind.Object)
                    {
                        vocabulary = new Dictionary<string, int>(CountProperties(field.Value));
                        foreach (var entry in field.Value.EnumerateObject())
                        {
                            vocabulary[entry.Name] = entry.Value.GetInt32();
                        }
                    }
                    else if (field.NameEquals("merges") && field.Value.ValueKind == JsonValueKind.Array)
                    {
                        merges = new (string, string)[field.Value.GetArrayLength()];
                        int i = 0;
                        foreach (var merge in field.Value.EnumerateArray())
                        {
                            merges[i++] = merge.ValueKind == JsonValueKind.Array
                                ? (merge[0].GetString()!, merge[1].GetString()!)
                                : Split(merge.GetString()!);
                        }
                    }
                    else
                    {
                        model[field.Name] = JsonNode.Parse(System.Runtime.InteropServices.JsonMarshal.GetRawUtf8Value(field.Value));
                    }
                }

                rest[property.Name] = model;
            }
            else
            {
                rest[property.Name] = JsonNode.Parse(System.Runtime.InteropServices.JsonMarshal.GetRawUtf8Value(property.Value));
            }
        }

        return new BpeTokenizer(rest, vocabulary ?? throw new InvalidDataException("tokenizer.json has no model vocabulary."), merges);
    }

    private static int CountProperties(JsonElement element)
    {
        int count = 0;
        foreach (var _ in element.EnumerateObject())
        {
            count++;
        }

        return count;
    }

    /// <summary>Reads a tokenizer from the JSON of a tokenizer.json.</summary>
    public static BpeTokenizer FromJson(JsonObject json)
    {
        var model = json["model"] as JsonObject;
        var vocabulary = model?["vocab"]?.AsObject().ToDictionary(p => p.Key, p => (int)p.Value!)
            ?? throw new InvalidDataException("tokenizer.json has no model vocabulary.");
        var merges = (model["merges"]?.AsArray() ?? [])
            .Select(m => m is JsonArray pair ? ((string)pair[0]!, (string)pair[1]!) : Split((string)m!)).ToArray();
        return new BpeTokenizer(json, vocabulary, merges);
    }

    /// <inheritdoc />
    public int VocabularySize => _tokens.Length;

    /// <summary>
    /// Extends the vocabulary to <paramref name="size"/> ids (models often have more embedding rows than the tokenizer has
    /// tokens, for alignment); the extra ids decode to nothing.
    /// </summary>
    public void PadVocabulary(int size)
    {
        if (size > _tokens.Length)
        {
            Array.Resize(ref _tokens, size);
        }
    }

    /// <summary>The id of <paramref name="token"/>, or null.</summary>
    public int? IdOf(string token) => _vocabulary.TryGetValue(token, out int id) ? id : null;

    /// <summary>The token with <paramref name="id"/>.</summary>
    public string TokenOf(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] ?? "" : "";

    /// <summary>Whether <paramref name="id"/> is a special token.</summary>
    public bool IsSpecial(int id) => _special.Contains(id);

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        text = WithoutLoneSurrogates(text);
        var ids = new List<int>(text.Length / 3 + 4);
        EncodeSegments(text, 0, text.Length, ids);
        return ids;
    }

    /// <summary>
    /// Appends the ids of <paramref name="text"/>[<paramref name="start"/>..<paramref name="start"/> + <paramref name="length"/>]
    /// to <paramref name="ids"/>: the ids <see cref="Encode"/> gives for that substring, without copying it.
    /// </summary>
    public void EncodeRange(string text, int start, int length, List<int> ids)
    {
        if (text.AsSpan(start, length).ContainsAnyInRange('\uD800', '\uDFFF'))
        {
            text = WithoutLoneSurrogates(text.Substring(start, length));         // the range may cut a pair in half
            start = 0;
        }

        EncodeSegments(text, start, length, ids);
    }

    // Added tokens, and the text between them, of text[start..start + length].
    private void EncodeSegments(string text, int start, int length, List<int> ids)
    {
        int last = start;
        if (_addedPattern is not null)
        {
            foreach (var match in _addedPattern.EnumerateMatches(text.AsSpan(start, length)))
            {
                int at = start + match.Index;
                EncodeText(text, last, at - last, ids, last == start);
                ids.Add(_vocabularySpans[text.AsSpan(at, match.Length)]);
                last = at + match.Length;
            }
        }

        EncodeText(text, last, start + length - last, ids, last == start);
    }

    // Half of a surrogate pair is not a character (it has no UTF-8 bytes, and Unicode normalization rejects it): it
    // becomes U+FFFD, as it does when such text is written as UTF-8.
    private static string WithoutLoneSurrogates(string text)
    {
        int at = text.AsSpan().IndexOfAnyInRange('\uD800', '\uDFFF');
        if (at < 0)
        {
            return text;
        }

        char[]? copy = null;
        for (int i = at; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                i++;
            }
            else if (char.IsSurrogate(text[i]))
            {
                copy ??= text.ToCharArray();
                copy[i] = '\uFFFD';
            }
        }

        return copy is null ? text : new string(copy);
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids) => ids switch
    {
        int[] array => Decode(array.AsSpan()),
        List<int> list => Decode(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(list)),
        _ => Decode([.. ids]),
    };

    /// <inheritdoc />
    public string Decode(ReadOnlySpan<int> ids)
    {
        if (_byteLevel)
        {
            // Byte-level tokens are bytes written as characters (added tokens are plain text): copy each token's bytes.
            var starts = _byteStart!;
            var all = _tokenBytes!;
            int count = 0;
            foreach (int id in ids)
            {
                if ((uint)id < (uint)starts.Length - 1)
                {
                    count += starts[id + 1] - starts[id];
                }
            }

            byte[]? rented = null;
            Span<byte> bytes = count <= 1024 ? stackalloc byte[count] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(count));
            int at = 0;
            foreach (int id in ids)
            {
                if ((uint)id < (uint)starts.Length - 1)
                {
                    var source = all.AsSpan(starts[id], starts[id + 1] - starts[id]);
                    source.CopyTo(bytes[at..]);
                    at += source.Length;
                }
            }

            string text = Encoding.UTF8.GetString(bytes[..count]);
            if (rented is not null)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(rented);
            }

            return text;
        }

        if (_fusedDecoders >= 0 && (_decodeTable ??= BuildDecodeTable()) is { Text: not null } table)
        {
            return DecodeFused(ids, table);
        }

        var tokens = new List<string>(ids.Length);
        foreach (int id in ids)
        {
            tokens.Add(TokenOf(id));
        }

        foreach (var decoder in _decoders)
        {
            tokens = decoder.Run(tokens);
        }

        return string.Concat(tokens);
    }

    // Every token's text after the decoders' Replace stages, and its byte when the ByteFallback stage reads it as one
    // (-1 otherwise); Empty* for ids without a token. Text is null when some token looks like <0xNN> but is not hex.
    private sealed record DecodeTable(string[]? Text, int[] Byte, string EmptyText, int EmptyByte);

    // How many decoders (from the first) DecodeFused reproduces: Replace stages, then at most one ByteFallback, up to the
    // first Fuse or Metaspace (from there the text is one piece, so the remaining stages run on it); -1 for other chains.
    private int FusedDecoders()
    {
        bool byteFallback = false;
        for (int i = 0; i < _decoders.Count; i++)
        {
            switch (_decoders[i].Type)
            {
                case "Replace" when !byteFallback:
                    break;
                case "ByteFallback" when !byteFallback:
                    byteFallback = true;
                    break;
                case "Fuse" or "Metaspace":
                    return i;
                default:
                    return -1;
            }
        }

        return _decoders.Count;
    }

    private DecodeTable BuildDecodeTable()
    {
        bool byteFallback = false;
        var maps = new List<Func<string, string>>();
        foreach (var (type, _, map) in _decoders.Take(_fusedDecoders))
        {
            if (map is not null)
            {
                maps.Add(map);
            }

            byteFallback |= type == "ByteFallback";
        }

        bool hex = true;
        (string, int) Entry(string token)
        {
            foreach (var map in maps)
            {
                token = map(token);
            }

            if (byteFallback && IsByteToken(token))
            {
                if (char.IsAsciiHexDigit(token[3]) && char.IsAsciiHexDigit(token[4]))
                {
                    return (token, byte.Parse(token.AsSpan(3, 2), System.Globalization.NumberStyles.AllowHexSpecifier));
                }

                hex = false;                                                     // left to the list stages (as they read it)
            }

            return (token, -1);
        }

        var tokens = _tokens;
        var text = new string[tokens.Length];
        var bytes = new int[tokens.Length];
        for (int id = 0; id < tokens.Length; id++)
        {
            (text[id], bytes[id]) = Entry(tokens[id] ?? "");
        }

        var (emptyText, emptyByte) = Entry("");
        return new DecodeTable(hex ? text : null, bytes, emptyText, emptyByte);
    }

    private static bool IsByteToken(string token) => token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>';

    // The decoders through the table: token texts copied side by side, runs of byte tokens decoded as UTF-8 (one U+FFFD
    // per byte when a run is not valid UTF-8, as the ByteFallback stage does), then the stages after the table's.
    private string DecodeFused(ReadOnlySpan<int> ids, DecodeTable table)
    {
        var texts = table.Text!;
        var tokenBytes = table.Byte;
        int most = 0;
        foreach (int id in ids)
        {
            bool known = (uint)id < (uint)texts.Length;
            most += (known ? tokenBytes[id] : table.EmptyByte) >= 0 ? 1 : (known ? texts[id] : table.EmptyText).Length;
        }

        char[]? rentedChars = null;
        byte[]? rentedBytes = null;
        Span<char> chars = most <= 1024 ? stackalloc char[1024] : (rentedChars = System.Buffers.ArrayPool<char>.Shared.Rent(most));
        Span<byte> run = ids.Length <= StackChars ? stackalloc byte[StackChars] : (rentedBytes = System.Buffers.ArrayPool<byte>.Shared.Rent(ids.Length));
        int at = 0, pending = 0;
        foreach (int id in ids)
        {
            bool known = (uint)id < (uint)texts.Length;
            int b = known ? tokenBytes[id] : table.EmptyByte;
            if (b >= 0)
            {
                run[pending++] = (byte)b;
                continue;
            }

            at += DecodeRun(run[..pending], chars[at..]);
            pending = 0;
            string token = known ? texts[id] : table.EmptyText;
            token.CopyTo(chars[at..]);
            at += token.Length;
        }

        at += DecodeRun(run[..pending], chars[at..]);
        string text = new(chars[..at]);
        if (rentedChars is not null)
        {
            System.Buffers.ArrayPool<char>.Shared.Return(rentedChars);
        }

        if (rentedBytes is not null)
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(rentedBytes);
        }

        List<string>? pieces = null;
        for (int i = _fusedDecoders; i < _decoders.Count; i++)
        {
            if (pieces is not null || _decoders[i].Type != "Fuse")              // fusing one piece changes nothing
            {
                pieces = _decoders[i].Run(pieces ?? [text]);
            }
        }

        return pieces is null ? text : string.Concat(pieces);
    }

    // A run of byte tokens as text: its UTF-8, or one U+FFFD per byte when it is not valid UTF-8.
    private static int DecodeRun(ReadOnlySpan<byte> run, Span<char> destination)
    {
        if (run.IsEmpty)
        {
            return 0;
        }

        if (System.Text.Unicode.Utf8.IsValid(run))
        {
            return Encoding.UTF8.GetChars(run, destination);
        }

        destination[..run.Length].Fill('\uFFFD');
        return run.Length;
    }

    // The bytes of every token (see _tokenBytes): characters of the byte map are bytes, others (and added tokens) UTF-8.
    private void BuildTokenBytes()
    {
        var starts = new int[_tokens.Length + 1];
        var bytes = new List<byte>(_tokens.Length * 8);
        Span<byte> utf8 = stackalloc byte[4];
        for (int id = 0; id < _tokens.Length; id++)
        {
            starts[id] = bytes.Count;
            string token = _tokens[id] ?? "";
            if (_addedIds.Contains(id))
            {
                bytes.AddRange(Encoding.UTF8.GetBytes(token));
                continue;
            }

            foreach (char c in token)
            {
                if (CharToByte.TryGetValue(c, out byte b))
                {
                    bytes.Add(b);
                }
                else
                {
                    int n = Encoding.UTF8.GetBytes([c], utf8);
                    bytes.AddRange(utf8[..n]);
                }
            }
        }

        starts[^1] = bytes.Count;
        _tokenBytes = [.. bytes];
        _byteStart = starts;
    }

    // Encodes text[start..start + length]. atStart: the text begins the input (Metaspace's "first" scheme only prepends
    // there, not after an added token).
    private void EncodeText(string text, int start, int length, List<int> ids, bool atStart)
    {
        if (length == 0)
        {
            return;
        }

        if (_normalizers.Count == 0)
        {
            EncodeNormalized(text.AsSpan(start, length), null, ids, atStart);
            return;
        }

        if (_normalizerSteps is { } steps)
        {
            // Prepends and character replacements in place: the text goes at the end of the buffer, each prepend before it.
            int total = length;
            foreach (var step in steps)
            {
                total += step.Prepend?.Length ?? 0;
            }

            char[]? rented = null;
            Span<char> buffer = total <= StackChars ? stackalloc char[StackChars] : (rented = System.Buffers.ArrayPool<char>.Shared.Rent(total));
            buffer = buffer[..total];
            int at = total - length;
            text.AsSpan(start, length).CopyTo(buffer[at..]);
            foreach (var (prepend, from, to) in steps)
            {
                if (prepend is not null)
                {
                    at -= prepend.Length;
                    prepend.CopyTo(buffer[at..]);
                }
                else
                {
                    buffer[at..].Replace(from, to);
                }
            }

            EncodeNormalized(buffer, null, ids, atStart);
            if (rented is not null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(rented);
            }

            return;
        }

        string segment = text.Substring(start, length);
        foreach (var normalize in _normalizers)
        {
            segment = normalize(segment);
        }

        EncodeNormalized(segment, segment, ids, atStart);
    }

    // Pre-tokenizes and encodes normalized text (segmentString: the same text as a string, when there is one).
    private void EncodeNormalized(ReadOnlySpan<char> segment, string? segmentString, List<int> ids, bool atStart)
    {
        if (_splits is not null)
        {
            EncodeRanges(segment, ids);
        }
        else if (_metaspace is { } metaspace)
        {
            EncodeMetaspace(segment, metaspace.Replacement, metaspace.Scheme == "always" || metaspace.Scheme == "first" && atStart, metaspace.Split, ids);
        }
        else if (_prefixByteLevel is { } byteLevel)
        {
            EncodePrefixedByteLevel(segment, byteLevel.Pattern, ids);
        }
        else
        {
            var pieces = new List<string> { segmentString ?? segment.ToString() };
            foreach (var preTokenize in _preTokenizers)
            {
                pieces = preTokenize(pieces, atStart);
            }

            foreach (var piece in pieces)
            {
                EncodePiece(piece, ids, mapBytes: false);
            }
        }
    }

    // The Metaspace pre-tokenizer on its own: spaces become the replacement, which is prepended (when the scheme says so
    // and the text does not start with it); with split, a piece starts at every replacement but the first character.
    private void EncodeMetaspace(ReadOnlySpan<char> segment, string replacement, bool prepend, bool split, List<int> ids)
    {
        int spaces = segment.Count(' ');
        int most = replacement.Length + segment.Length + spaces * (replacement.Length - 1);
        char[]? rented = null;
        Span<char> buffer = most <= StackChars ? stackalloc char[StackChars] : (rented = System.Buffers.ArrayPool<char>.Shared.Rent(most));
        int at = replacement.Length;
        for (var rest = segment; ;)
        {
            int space = rest.IndexOf(' ');
            if (space < 0)
            {
                rest.CopyTo(buffer[at..]);
                at += rest.Length;
                break;
            }

            rest[..space].CopyTo(buffer[at..]);
            at += space;
            replacement.CopyTo(buffer[at..]);
            at += replacement.Length;
            rest = rest[(space + 1)..];
        }

        int start = replacement.Length;
        if (prepend && !buffer[start..at].StartsWith(replacement, StringComparison.Ordinal))
        {
            start = 0;
            replacement.CopyTo(buffer);
        }

        ReadOnlySpan<char> s = buffer[start..at];
        if (!split)
        {
            EncodePiece(s, ids, mapBytes: false);
        }
        else
        {
            int from = 0;
            while (from < s.Length)
            {
                int next = from + 1 >= s.Length ? -1 : s[(from + 1)..].IndexOf(replacement, StringComparison.Ordinal);
                int to = next < 0 ? s.Length : from + 1 + next;
                EncodePiece(s[from..to], ids, mapBytes: false);
                from = to;
            }
        }

        if (rented is not null)
        {
            System.Buffers.ArrayPool<char>.Shared.Return(rented);
        }
    }

    // The ByteLevel pre-tokenizer on its own with add_prefix_space: a space before text that does not start with one,
    // then its regex's matches (pattern; or the whole text without use_regex) encoded from their UTF-8 bytes.
    private void EncodePrefixedByteLevel(ReadOnlySpan<char> segment, Regex? pattern, List<int> ids)
    {
        char[]? rented = null;
        scoped ReadOnlySpan<char> piece = segment;
        Span<char> buffer = segment.Length < StackChars ? stackalloc char[StackChars] : default;
        if (!segment.StartsWith(' '))
        {
            buffer = buffer.IsEmpty ? (rented = System.Buffers.ArrayPool<char>.Shared.Rent(segment.Length + 1)) : buffer;
            buffer[0] = ' ';
            segment.CopyTo(buffer[1..]);
            piece = buffer[..(segment.Length + 1)];
        }

        if (pattern is null)
        {
            EncodePiece(piece, ids, mapBytes: true);
        }
        else
        {
            foreach (var m in pattern.EnumerateMatches(piece))
            {
                EncodePiece(piece.Slice(m.Index, m.Length), ids, mapBytes: true);
            }
        }

        if (rented is not null)
        {
            System.Buffers.ArrayPool<char>.Shared.Return(rented);
        }
    }

    // Splits text with the split stages (as ranges: nothing is copied) and encodes each piece.
    private void EncodeRanges(ReadOnlySpan<char> text, List<int> ids)
    {
        var current = new List<(int Start, int Length)> { (0, text.Length) };
        var next = new List<(int Start, int Length)>();
        foreach (var (pattern, behavior, invert) in _splits!)
        {
            next.Clear();
            foreach (var range in current)
            {
                SplitRange(text, range, pattern, behavior, invert, next);
            }

            (current, next) = (next, current);
        }

        foreach (var (from, count) in current)
        {
            EncodePiece(text.Slice(from, count), ids, mapBytes: _byteLevel);
        }
    }

    // A pre-token piece: its cached ids, or the merges (mapBytes: the piece is raw text of a byte-level tokenizer, so its
    // UTF-8 bytes are written as the byte map's characters first).
    private void EncodePiece(ReadOnlySpan<char> piece, List<int> ids, bool mapBytes)
    {
        if (piece.IsEmpty)
        {
            return;
        }

        if (_cacheSpans.TryGetValue(piece, out var cached))
        {
            ids.AddRange(cached);
            return;
        }

        int[] encoded;
        if (mapBytes)
        {
            int most = Encoding.UTF8.GetMaxByteCount(piece.Length);
            byte[]? rented = null;
            Span<byte> bytes = most <= StackChars ? stackalloc byte[StackChars] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(most));
            int count = Encoding.UTF8.GetBytes(piece, bytes);
            encoded = BpeBytes(bytes[..count]);
            if (rented is not null)
            {
                System.Buffers.ArrayPool<byte>.Shared.Return(rented);
            }
        }
        else
        {
            encoded = Bpe(piece);
        }

        if (_cache.Count < CacheLimit && piece.Length <= StackChars)          // long pieces (unsplit SentencePiece text) rarely repeat
        {
            _cacheSpans.TryAdd(piece, encoded);
        }

        ids.AddRange(encoded);
    }

    // Byte-level merges straight from the bytes: each byte starts as its character's token.
    private int[] BpeBytes(ReadOnlySpan<byte> bytes)
    {
        int[]? rented = null;
        Span<int> symbols = bytes.Length <= StackChars ? stackalloc int[StackChars] : (rented = System.Buffers.ArrayPool<int>.Shared.Rent(bytes.Length));
        symbols = symbols[..bytes.Length];
        bool known = _pairs is not null;
        for (int i = 0; i < bytes.Length && known; i++)
        {
            symbols[i] = _byteIds[bytes[i]];
            known = symbols[i] >= 0;
        }

        int[] result;
        if (known && !_ignoreMerges)
        {
            result = Merge(symbols);
        }
        else
        {
            // The general path, on the mapped characters.
            char[]? rentedChars = null;
            Span<char> chars = bytes.Length <= StackChars ? stackalloc char[StackChars] : (rentedChars = System.Buffers.ArrayPool<char>.Shared.Rent(bytes.Length));
            for (int i = 0; i < bytes.Length; i++)
            {
                chars[i] = ByteToChar[bytes[i]];
            }

            result = Bpe(chars[..bytes.Length]);
            if (rentedChars is not null)
            {
                System.Buffers.ArrayPool<char>.Shared.Return(rentedChars);
            }
        }

        if (rented is not null)
        {
            System.Buffers.ArrayPool<int>.Shared.Return(rented);
        }

        return result;
    }

    // Byte-pair merges over one piece (characters of the token alphabet), best (lowest-rank) pair first, leftmost among equals.
    private int[] Bpe(ReadOnlySpan<char> piece)
    {
        if (_ignoreMerges && _vocabularySpans.TryGetValue(piece, out int whole))
        {
            return [whole];
        }

        if (_pairs is not null)
        {
            int[]? rented = null;
            Span<int> symbols = piece.Length <= StackChars ? stackalloc int[StackChars] : (rented = System.Buffers.ArrayPool<int>.Shared.Rent(piece.Length));
            int n = 0;
            bool known = true;
            for (int i = 0; i < piece.Length && known;)
            {
                int width = i + 1 < piece.Length && char.IsSurrogatePair(piece[i], piece[i + 1]) ? 2 : 1;
                known = _vocabularySpans.TryGetValue(piece.Slice(i, width), out symbols[n++]);
                i += width;
            }

            // Every merge joins two tokens into a token, so a character that is not one never merges: the runs of known
            // characters between such characters merge on their own.
            int[] merged = known ? Merge(symbols[..n]) : MergeAround(piece, symbols);
            if (rented is not null)
            {
                System.Buffers.ArrayPool<int>.Shared.Return(rented);
            }

            return merged;
        }

        return BpeStrings(piece.ToString());
    }

    // Merges the runs of known characters of piece (symbols: room for one id per character), each unknown character as
    // the string merges give it (its byte fallback tokens, or the unknown token).
    private int[] MergeAround(ReadOnlySpan<char> piece, Span<int> symbols)
    {
        var result = new List<int>(piece.Length);
        int n = 0;
        for (int i = 0; i < piece.Length;)
        {
            int width = i + 1 < piece.Length && char.IsSurrogatePair(piece[i], piece[i + 1]) ? 2 : 1;
            if (_vocabularySpans.TryGetValue(piece.Slice(i, width), out symbols[n]))
            {
                n++;
            }
            else
            {
                result.AddRange(n > 0 ? Merge(symbols[..n]) : []);
                n = 0;
                AddUnknown(piece.Slice(i, width), result);
            }

            i += width;
        }

        result.AddRange(n > 0 ? Merge(symbols[..n]) : []);
        return [.. result];
    }

    // A symbol that is not a token: its UTF-8 bytes' fallback tokens, or the unknown token.
    private void AddUnknown(ReadOnlySpan<char> symbol, List<int> result)
    {
        if (_byteFallback)
        {
            int most = Encoding.UTF8.GetMaxByteCount(symbol.Length);
            Span<byte> bytes = most <= StackChars ? stackalloc byte[StackChars] : new byte[most];
            foreach (byte b in bytes[..Encoding.UTF8.GetBytes(symbol, bytes)])
            {
                result.Add(_fallbackIds[b] >= 0 ? _fallbackIds[b]
                    : throw new InvalidDataException($"Byte fallback token <0x{b:X2}> is missing from the vocabulary."));
            }
        }
        else
        {
            result.Add(_unknown ?? throw new InvalidDataException($"'{symbol}' is not in the vocabulary and the tokenizer has no unknown token."));
        }
    }

    // The merges on token ids: short pieces merge in place (lowest-rank pair, leftmost among equals, repeatedly); long
    // ones use a heap (same order, O(n log n)).
    private int[] Merge(Span<int> ids)
    {
        if (ids.Length > 32)
        {
            return MergeLong(ids);
        }

        int count = ids.Length;
        while (count > 1)
        {
            int best = -1, bestRank = int.MaxValue, bestMerged = 0;
            for (int i = 0; i < count - 1; i++)
            {
                if (_pairs!.TryGetValue(((long)ids[i] << 32) | (uint)ids[i + 1], out var pair) && pair.Rank < bestRank)
                {
                    (best, bestRank, bestMerged) = (i, pair.Rank, pair.Merged);
                }
            }

            if (best < 0)
            {
                break;
            }

            ids[best] = bestMerged;
            ids[(best + 2)..count].CopyTo(ids[(best + 1)..]);
            count--;
        }

        return ids[..count].ToArray();
    }

    // Long pieces (SentencePiece-style tokenizers do not split text first): the symbols in a linked list and candidate
    // pairs in a heap ordered by (rank, position); a popped pair whose symbols changed since it was queued is skipped.
    private int[] MergeLong(Span<int> ids)
    {
        int n = ids.Length;
        var symbol = ids.ToArray();
        var next = new int[n];
        var previous = new int[n];
        var alive = new bool[n];
        for (int i = 0; i < n; i++)
        {
            next[i] = i + 1 < n ? i + 1 : -1;
            previous[i] = i - 1;
            alive[i] = true;
        }

        var queue = new PriorityQueue<(int Left, int LeftId, int RightId, int Merged), (int Rank, int Position)>(n);
        void Offer(int left)
        {
            int right = left < 0 ? -1 : next[left];
            if (right >= 0 && _pairs!.TryGetValue(((long)symbol[left] << 32) | (uint)symbol[right], out var pair))
            {
                queue.Enqueue((left, symbol[left], symbol[right], pair.Merged), (pair.Rank, left));
            }
        }

        for (int i = 0; i < n - 1; i++)
        {
            Offer(i);
        }

        int remaining = n;
        while (queue.TryDequeue(out var candidate, out _))
        {
            int left = candidate.Left, right = next[left];
            if (!alive[left] || right < 0 || symbol[left] != candidate.LeftId || symbol[right] != candidate.RightId)
            {
                continue;
            }

            symbol[left] = candidate.Merged;
            alive[right] = false;
            remaining--;
            next[left] = next[right];
            if (next[right] >= 0)
            {
                previous[next[right]] = left;
            }

            Offer(previous[left]);
            Offer(left);
        }

        var result = new int[remaining];
        for (int i = 0, j = 0; i >= 0; i = next[i])
        {
            result[j++] = symbol[i];
        }

        return result;
    }

    // Merges on strings: for tokenizers whose merges name pieces that are not tokens.
    private int[] BpeStrings(string piece)
    {
        if (_ranks is null)
        {
            var ranks = new Dictionary<(string, string), int>(_merges.Length);
            for (int r = 0; r < _merges.Length; r++)
            {
                ranks.TryAdd(_merges[r], r);
            }

            Interlocked.CompareExchange(ref _ranks, ranks, null);
        }

        var symbols = new List<string>();
        for (int i = 0; i < piece.Length; i += char.IsSurrogatePair(piece, i) ? 2 : 1)
        {
            symbols.Add(char.IsSurrogatePair(piece, i) ? piece.Substring(i, 2) : piece[i].ToString());
        }

        while (symbols.Count > 1)
        {
            int best = -1, bestRank = int.MaxValue;
            for (int i = 0; i < symbols.Count - 1; i++)
            {
                if (_ranks.TryGetValue((symbols[i], symbols[i + 1]), out int rank) && rank < bestRank)
                {
                    (best, bestRank) = (i, rank);
                }
            }

            if (best < 0)
            {
                break;
            }

            symbols[best] += symbols[best + 1];
            symbols.RemoveAt(best + 1);
        }

        var result = new List<int>();
        foreach (var symbol in symbols)
        {
            if (_vocabulary.TryGetValue(symbol, out int id))
            {
                result.Add(id);
            }
            else
            {
                AddUnknown(symbol, result);
            }
        }

        return [.. result];
    }

    // One split stage on a range: the matches and the text between them, combined as the behavior says (all adjacent,
    // so every result is again a range of the same string).
    private static void SplitRange(ReadOnlySpan<char> text, (int Start, int Length) range, Regex pattern, string behavior, bool invert, List<(int Start, int Length)> output)
    {
        int last = range.Start, end = range.Start + range.Length;
        int pending = -1;                                                       // MergedWithNext: start of matched text waiting
        bool previousOpen = false;                                              // MergedWithPrevious: a part to extend exists
        void Add(int from, int to, bool matched)
        {
            if (to <= from)
            {
                return;
            }

            switch (behavior)
            {
                case "Isolated":
                    output.Add((from, to - from));
                    break;
                case "Removed":
                    if (!matched)
                    {
                        output.Add((from, to - from));
                    }

                    break;
                case "MergedWithPrevious":
                    if (matched && previousOpen)
                    {
                        var last = output[^1];
                        output[^1] = (last.Start, to - last.Start);
                    }
                    else
                    {
                        output.Add((from, to - from));
                    }

                    previousOpen = true;
                    break;
                case "MergedWithNext":
                    if (matched)
                    {
                        pending = pending < 0 ? from : pending;
                    }
                    else
                    {
                        int first = pending < 0 ? from : pending;
                        output.Add((first, to - first));
                        pending = -1;
                    }

                    break;
                default:
                    throw new NotSupportedException($"Split behavior '{behavior}' is not supported.");
            }
        }

        foreach (var m in pattern.EnumerateMatches(text.Slice(range.Start, range.Length)))
        {
            if (m.Length == 0)
            {
                continue;
            }

            int from = range.Start + m.Index;
            Add(last, from, invert);
            Add(from, from + m.Length, !invert);
            last = from + m.Length;
        }

        Add(last, end, invert);
        if (behavior == "MergedWithNext" && pending >= 0)
        {
            output.Add((pending, end - pending));
        }
    }

    private void AddNormalizer(JsonNode? node)
    {
        switch (node)
        {
            case null:
                return;
            case JsonObject n when TokenizerComponents.FindNormalizer((string?)n["type"]) is { } create:
                _normalizerSteps = null;                                          // a registered type: the string pipeline runs
                _normalizers.Add(create(n).Normalize);
                return;
            case JsonObject n when (string?)n["type"] == "Sequence":
                foreach (var inner in n["normalizers"]!.AsArray())
                {
                    AddNormalizer(inner);
                }

                return;
            case JsonObject n:
                string type = (string)n["type"]!;
                if (type == "Prepend")
                {
                    _normalizerSteps?.Add(((string?)n["prepend"] ?? "", '\0', '\0'));
                }
                else if (type == "Replace" && n["pattern"]?["String"] is { } literal && (string)literal! is { Length: 1 } from
                         && (string)n["content"]! is { Length: 1 } to)
                {
                    _normalizerSteps?.Add((null, from[0], to[0]));
                }
                else
                {
                    _normalizerSteps = null;
                }

                _normalizers.Add(type switch
                {
                    "NFC" => s => s.Normalize(NormalizationForm.FormC),
                    "NFKC" => s => s.Normalize(NormalizationForm.FormKC),
                    "NFD" => s => s.Normalize(NormalizationForm.FormD),
                    "NFKD" => s => s.Normalize(NormalizationForm.FormKD),
                    "Prepend" => s => (string)n["prepend"]! + s,
                    "Replace" => Replacer(n),
                    "Lowercase" => s => s.ToLowerInvariant(),
                    _ => throw new NotSupportedException($"Tokenizer normalizer '{type}' is not supported; add it with TokenizerComponents.RegisterNormalizer."),
                });
                return;
        }
    }

    private static Func<string, string> Replacer(JsonObject n)
    {
        string content = (string)n["content"]!;
        if (n["pattern"]?["String"] is { } literal)
        {
            string from = (string)literal!;
            return s => s.Replace(from, content, StringComparison.Ordinal);
        }

        var regex = new Regex((string)n["pattern"]!["Regex"]!, Fast);
        return s => regex.Replace(s, content);
    }

    // Returns true when the tokenizer is byte-level (its pieces are bytes written as characters).
    private bool AddPreTokenizer(JsonNode? node)
    {
        if (node is not JsonObject p)
        {
            return false;
        }

        string type = (string)p["type"]!;
        if (TokenizerComponents.FindPreTokenizer(type) is { } create)
        {
            var custom = create(p);
            _splits = null;                                                       // a registered type: the string pipeline runs
            _preTokenizers.Add((pieces, atStart) => custom.PreTokenize(pieces, atStart) is var result && result is List<string> list ? list : [.. result]);
            return false;
        }

        switch (type)
        {
            case "Sequence":
                bool byteLevel = false;
                foreach (var inner in p["pretokenizers"]!.AsArray())
                {
                    byteLevel |= AddPreTokenizer(inner);
                }

                return byteLevel;
            case "Split":
                var pattern = p["pattern"]!["Regex"] is { } r ? new Regex((string)r!, Fast) : new Regex(Regex.Escape((string)p["pattern"]!["String"]!), Fast);
                string behavior = (string?)p["behavior"] ?? "Isolated";
                bool invert = (bool?)p["invert"] ?? false;
                _preTokenizers.Add((pieces, atStart) => SplitPieces(pieces, pattern, behavior, invert));
                _splits?.Add((pattern, behavior, invert));
                return false;
            case "ByteLevel":
                bool prefix = (bool?)p["add_prefix_space"] ?? false, useRegex = (bool?)p["use_regex"] ?? true;
                var gpt2 = new Regex(Gpt2Pattern, Fast);
                if (prefix)
                {
                    _splits = null;                                               // rewrites text: the string pipeline runs
                }
                else if (useRegex)
                {
                    _splits?.Add((gpt2, "Isolated", false));
                }

                _prefixByteLevel = prefix ? (true, useRegex ? gpt2 : null) : null;
                _preTokenizers.Add((pieces, atStart) =>
                {
                    var result = new List<string>(pieces.Count);
                    foreach (var text in pieces)
                    {
                        string piece = prefix && !text.StartsWith(' ') ? " " + text : text;
                        if (!useRegex)
                        {
                            result.Add(ToByteChars(piece));
                            continue;
                        }

                        foreach (var m in gpt2.EnumerateMatches(piece))
                        {
                            result.Add(ToByteChars(piece.AsSpan(m.Index, m.Length)));
                        }
                    }

                    return result;
                });
                return true;
            case "Metaspace":
                string replacement = (string?)p["replacement"] ?? "▁";
                string scheme = (string?)p["prepend_scheme"] ?? (((bool?)p["add_prefix_space"] ?? true) ? "always" : "never");
                bool split = (bool?)p["split"] ?? true;
                _splits = null;
                _metaspace = (replacement, scheme, split);
                _preTokenizers.Add((pieces, atStart) =>
                {
                    var result = new List<string>();
                    for (int i = 0; i < pieces.Count; i++)
                    {
                        string s = pieces[i].Replace(" ", replacement, StringComparison.Ordinal);
                        if ((scheme == "always" || scheme == "first" && i == 0 && atStart) && !s.StartsWith(replacement, StringComparison.Ordinal))
                        {
                            s = replacement + s;
                        }

                        if (!split)
                        {
                            result.Add(s);
                            continue;
                        }

                        int start = 0;
                        for (int j = 1; j <= s.Length; j++)
                        {
                            if (j == s.Length || s.AsSpan(j).StartsWith(replacement, StringComparison.Ordinal))
                            {
                                result.Add(s[start..j]);
                                start = j;
                            }
                        }
                    }

                    return result;
                });
                return false;
            case "Digits":
                bool individual = (bool?)p["individual_digits"] ?? false;
                var digits = new Regex(individual ? @"\p{Nd}" : @"\p{Nd}+", Fast);
                _preTokenizers.Add((pieces, atStart) => SplitPieces(pieces, digits, "Isolated", false));
                _splits?.Add((digits, "Isolated", false));
                return false;
            case "Whitespace":
                var words = new Regex(@"\w+|[^\w\s]+", Fast);
                _preTokenizers.Add((pieces, atStart) =>
                {
                    var result = new List<string>(pieces.Count);
                    foreach (var piece in pieces)
                    {
                        foreach (var m in words.EnumerateMatches(piece))
                        {
                            result.Add(piece.Substring(m.Index, m.Length));
                        }
                    }

                    return result;
                });
                _splits?.Add((words, "Removed", true));                           // keeps the matches only
                return false;
            default:
                throw new NotSupportedException($"Tokenizer pre-tokenizer '{type}' is not supported; add it with TokenizerComponents.RegisterPreTokenizer.");
        }
    }

    // A split stage on pieces of the string pipeline: SplitRange on each piece, then the ranges copied out.
    private static List<string> SplitPieces(List<string> pieces, Regex pattern, string behavior, bool invert)
    {
        var result = new List<string>(pieces.Count);
        var ranges = new List<(int Start, int Length)>();
        foreach (var piece in pieces)
        {
            if (behavior is not ("Isolated" or "Removed" or "MergedWithPrevious" or "MergedWithNext"))
            {
                throw new NotSupportedException($"Split behavior '{behavior}' is not supported.");
            }

            ranges.Clear();
            SplitRange(piece, (0, piece.Length), pattern, behavior, invert, ranges);
            foreach (var (start, length) in ranges)
            {
                result.Add(length == piece.Length ? piece : piece.Substring(start, length));
            }
        }

        return result;
    }

    // Text as the byte map's characters of its UTF-8 bytes.
    private static string ToByteChars(ReadOnlySpan<char> text)
    {
        int count = Encoding.UTF8.GetByteCount(text);
        byte[]? rented = null;
        char[]? rentedChars = null;
        Span<byte> bytes = count <= StackChars ? stackalloc byte[StackChars] : (rented = System.Buffers.ArrayPool<byte>.Shared.Rent(count));
        Span<char> chars = count <= StackChars ? stackalloc char[StackChars] : (rentedChars = System.Buffers.ArrayPool<char>.Shared.Rent(count));
        Encoding.UTF8.GetBytes(text, bytes);
        for (int i = 0; i < count; i++)
        {
            chars[i] = ByteToChar[bytes[i]];
        }

        string mapped = new(chars[..count]);
        if (rented is not null)
        {
            System.Buffers.ArrayPool<byte>.Shared.Return(rented);
            System.Buffers.ArrayPool<char>.Shared.Return(rentedChars!);
        }

        return mapped;
    }

    private void AddDecoder(JsonNode? node)
    {
        if (node is not JsonObject d)
        {
            return;
        }

        string type = (string)d["type"]!;
        if (TokenizerComponents.FindDecoder(type) is { } create)
        {
            // A registered type: named so that no built-in stage (or the fused decoding) takes it for one of its own.
            var custom = create(d);
            _decoders.Add(("Registered:" + type, tokens => custom.Decode(tokens) is var result && result is List<string> list ? list : [.. result], null));
            return;
        }

        switch (type)
        {
            case "Sequence":
                foreach (var inner in d["decoders"]!.AsArray())
                {
                    AddDecoder(inner);
                }

                break;
            case "ByteLevel":
                break;                                                           // handled in Decode
            case "Replace":
                var replace = Replacer(d);
                _decoders.Add((type, tokens => [.. tokens.Select(replace)], replace));
                break;
            case "ByteFallback":
                _decoders.Add((type, tokens =>
                {
                    var result = new List<string>();
                    var bytes = new List<byte>();
                    void Flush()
                    {
                        if (bytes.Count > 0)
                        {
                            // As the tokenizers library: bytes that are not valid UTF-8 become one U+FFFD each.
                            try
                            {
                                result.Add(StrictUtf8.GetString(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(bytes)));
                            }
                            catch (DecoderFallbackException)
                            {
                                result.Add(new string('\uFFFD', bytes.Count));
                            }

                            bytes.Clear();
                        }
                    }

                    foreach (var token in tokens)
                    {
                        if (IsByteToken(token))
                        {
                            bytes.Add(char.IsAsciiHexDigit(token[3]) && char.IsAsciiHexDigit(token[4])
                                ? byte.Parse(token.AsSpan(3, 2), System.Globalization.NumberStyles.AllowHexSpecifier)
                                : Convert.ToByte(token[3..5], 16));
                        }
                        else
                        {
                            Flush();
                            result.Add(token);
                        }
                    }

                    Flush();
                    return result;
                }, null));
                break;
            case "Fuse":
                _decoders.Add((type, tokens => [string.Concat(tokens)], null));
                break;
            case "Strip":
                string content = (string)d["content"]!;
                int start = (int?)d["start"] ?? 0, stop = (int?)d["stop"] ?? 0;
                _decoders.Add((type, tokens =>
                {
                    if (tokens.Count == 0)
                    {
                        return tokens;
                    }

                    string first = tokens[0];
                    for (int i = 0; i < start && first.StartsWith(content, StringComparison.Ordinal); i++)
                    {
                        first = first[content.Length..];
                    }

                    tokens[0] = first;
                    string last = tokens[^1];
                    for (int i = 0; i < stop && last.EndsWith(content, StringComparison.Ordinal); i++)
                    {
                        last = last[..^content.Length];
                    }

                    tokens[^1] = last;
                    return tokens;
                }, null));
                break;
            case "Metaspace":
                string replacement = (string?)d["replacement"] ?? "▁";
                _decoders.Add((type, tokens =>
                {
                    var text = string.Concat(tokens).Replace(replacement, " ", StringComparison.Ordinal);
                    return [text.StartsWith(' ') ? text[1..] : text];
                }, null));
                break;
            default:
                throw new NotSupportedException($"Tokenizer decoder '{type}' is not supported; add it with TokenizerComponents.RegisterDecoder.");
        }
    }

    private static (string, string) Split(string merge)
    {
        int space = merge.IndexOf(' ', 1);
        return (merge[..space], merge[(space + 1)..]);
    }

    // GPT-2's byte → printable character table: printable bytes map to themselves, the rest to 256 + n.
    private static char[] BuildByteMap()
    {
        var map = new char[256];
        var printable = Enumerable.Range('!', '~' - '!' + 1).Concat(Enumerable.Range('¡', '¬' - '¡' + 1)).Concat(Enumerable.Range('®', 'ÿ' - '®' + 1)).ToHashSet();
        int n = 0;
        for (int b = 0; b < 256; b++)
        {
            map[b] = printable.Contains(b) ? (char)b : (char)(256 + n++);
        }

        return map;
    }
}
