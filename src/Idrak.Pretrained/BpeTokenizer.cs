using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Generation;

namespace Idrak.Pretrained;

/// <summary>
/// A byte-pair-encoding tokenizer read from a Hugging Face <c>tokenizer.json</c>: byte-level BPE (GPT-2, Llama 3, Qwen,
/// …) and SentencePiece-style BPE with the "▁" word marker and byte fallback (Llama 2, Mistral, Gemma, …). Special
/// (added) tokens are matched exactly before anything else. <see cref="Encode"/> adds no special tokens itself (chat
/// templates write them), like <c>add_special_tokens=False</c>. Pieces of the file it does not know are reported, not
/// guessed.
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
    private readonly List<Func<List<string>, List<string>>> _decoders = [];

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
        AddDecoder(json["decoder"]);
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
                        model[field.Name] = JsonNode.Parse(field.Value.GetRawText());
                    }
                }

                rest[property.Name] = model;
            }
            else
            {
                rest[property.Name] = JsonNode.Parse(property.Value.GetRawText());
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
        var ids = new List<int>(text.Length / 3 + 4);
        int last = 0;
        if (_addedPattern is not null)
        {
            foreach (var match in _addedPattern.EnumerateMatches(text))
            {
                EncodeText(text, last, match.Index - last, ids, last == 0);
                ids.Add(_vocabularySpans[text.AsSpan(match.Index, match.Length)]);
                last = match.Index + match.Length;
            }
        }

        EncodeText(text, last, text.Length - last, ids, last == 0);
        return ids;
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

        var tokens = new List<string>(ids.Length);
        foreach (int id in ids)
        {
            tokens.Add(TokenOf(id));
        }

        foreach (var decoder in _decoders)
        {
            tokens = decoder(tokens);
        }

        return string.Concat(tokens);
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

        if (_normalizers.Count > 0 || _splits is null)
        {
            string segment = text.Substring(start, length);
            foreach (var normalize in _normalizers)
            {
                segment = normalize(segment);
            }

            if (_splits is not null)
            {
                EncodeRanges(segment, 0, segment.Length, ids);
                return;
            }

            var pieces = new List<string> { segment };
            foreach (var preTokenize in _preTokenizers)
            {
                pieces = preTokenize(pieces, atStart);
            }

            foreach (var piece in pieces)
            {
                EncodePiece(piece, ids, mapBytes: false);
            }

            return;
        }

        EncodeRanges(text, start, length, ids);
    }

    // Splits text[start..start + length] with the split stages (as ranges: nothing is copied) and encodes each piece.
    private void EncodeRanges(string text, int start, int length, List<int> ids)
    {
        var current = new List<(int Start, int Length)> { (start, length) };
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
            EncodePiece(text.AsSpan(from, count), ids, mapBytes: _byteLevel);
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

            int[]? merged = known ? Merge(symbols[..n]) : null;
            if (rented is not null)
            {
                System.Buffers.ArrayPool<int>.Shared.Return(rented);
            }

            if (merged is not null)
            {
                return merged;
            }
        }

        return BpeStrings(piece.ToString());
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
            else if (_byteFallback)
            {
                foreach (byte b in Encoding.UTF8.GetBytes(symbol))
                {
                    result.Add(_vocabulary.TryGetValue($"<0x{b:X2}>", out int byteId) ? byteId
                        : throw new InvalidDataException($"Byte fallback token <0x{b:X2}> is missing from the vocabulary."));
                }
            }
            else
            {
                result.Add(_unknown ?? throw new InvalidDataException($"'{symbol}' is not in the vocabulary and the tokenizer has no unknown token."));
            }
        }

        return [.. result];
    }

    // One split stage on a range: the matches and the text between them, combined as the behavior says (all adjacent,
    // so every result is again a range of the same string).
    private static void SplitRange(string text, (int Start, int Length) range, Regex pattern, string behavior, bool invert, List<(int Start, int Length)> output)
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

        foreach (var m in pattern.EnumerateMatches(text.AsSpan(range.Start, range.Length)))
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
            case JsonObject n when (string?)n["type"] == "Sequence":
                foreach (var inner in n["normalizers"]!.AsArray())
                {
                    AddNormalizer(inner);
                }

                return;
            case JsonObject n:
                string type = (string)n["type"]!;
                _normalizers.Add(type switch
                {
                    "NFC" => s => s.Normalize(NormalizationForm.FormC),
                    "NFKC" => s => s.Normalize(NormalizationForm.FormKC),
                    "NFD" => s => s.Normalize(NormalizationForm.FormD),
                    "NFKD" => s => s.Normalize(NormalizationForm.FormKD),
                    "Prepend" => s => (string)n["prepend"]! + s,
                    "Replace" => Replacer(n),
                    "Lowercase" => s => s.ToLowerInvariant(),
                    _ => throw new NotSupportedException($"Tokenizer normalizer '{type}' is not supported."),
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
                _preTokenizers.Add((pieces, atStart) => [.. pieces.SelectMany(piece => SplitPiece(piece, pattern, behavior, invert))]);
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

                _preTokenizers.Add((pieces, atStart) => [.. pieces.SelectMany(piece =>
                {
                    if (prefix && !piece.StartsWith(' '))
                    {
                        piece = " " + piece;
                    }

                    var parts = useRegex ? gpt2.Matches(piece).Select(m => m.Value) : [piece];
                    return parts.Select(part => new string([.. Encoding.UTF8.GetBytes(part).Select(b => ByteToChar[b])]));
                })]);
                return true;
            case "Metaspace":
                string replacement = (string?)p["replacement"] ?? "▁";
                string scheme = (string?)p["prepend_scheme"] ?? (((bool?)p["add_prefix_space"] ?? true) ? "always" : "never");
                bool split = (bool?)p["split"] ?? true;
                _splits = null;
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
                _preTokenizers.Add((pieces, atStart) => [.. pieces.SelectMany(piece => SplitPiece(piece, digits, "Isolated", false))]);
                _splits?.Add((digits, "Isolated", false));
                return false;
            case "Whitespace":
                var words = new Regex(@"\w+|[^\w\s]+", Fast);
                _preTokenizers.Add((pieces, atStart) => [.. pieces.SelectMany(piece => words.Matches(piece).Select(m => m.Value))]);
                _splits?.Add((words, "Removed", true));                           // keeps the matches only
                return false;
            default:
                throw new NotSupportedException($"Tokenizer pre-tokenizer '{type}' is not supported.");
        }
    }

    private static IEnumerable<string> SplitPiece(string piece, Regex pattern, string behavior, bool invert)
    {
        var parts = new List<(string Text, bool Matched)>();
        int last = 0;
        foreach (Match m in pattern.Matches(piece))
        {
            if (m.Length == 0)
            {
                continue;
            }

            if (m.Index > last)
            {
                parts.Add((piece[last..m.Index], invert));
            }

            parts.Add((m.Value, !invert));
            last = m.Index + m.Length;
        }

        if (last < piece.Length)
        {
            parts.Add((piece[last..], invert));
        }

        switch (behavior)
        {
            case "Isolated":
                return parts.Select(p => p.Text);
            case "Removed":
                return parts.Where(p => !p.Matched).Select(p => p.Text);
            case "MergedWithPrevious":
                var previous = new List<string>();
                foreach (var (text, matched) in parts)
                {
                    if (matched && previous.Count > 0)
                    {
                        previous[^1] += text;
                    }
                    else
                    {
                        previous.Add(text);
                    }
                }

                return previous;
            case "MergedWithNext":
                var next = new List<string>();
                string pending = "";
                foreach (var (text, matched) in parts)
                {
                    if (matched)
                    {
                        pending += text;
                    }
                    else
                    {
                        next.Add(pending + text);
                        pending = "";
                    }
                }

                if (pending.Length > 0)
                {
                    next.Add(pending);
                }

                return next;
            default:
                throw new NotSupportedException($"Split behavior '{behavior}' is not supported.");
        }
    }

    private void AddDecoder(JsonNode? node)
    {
        if (node is not JsonObject d)
        {
            return;
        }

        string type = (string)d["type"]!;
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
                _decoders.Add(tokens => [.. tokens.Select(replace)]);
                break;
            case "ByteFallback":
                _decoders.Add(tokens =>
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
                                result.Add(StrictUtf8.GetString([.. bytes]));
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
                        if (token.Length == 6 && token.StartsWith("<0x", StringComparison.Ordinal) && token[5] == '>')
                        {
                            bytes.Add(Convert.ToByte(token[3..5], 16));
                        }
                        else
                        {
                            Flush();
                            result.Add(token);
                        }
                    }

                    Flush();
                    return result;
                });
                break;
            case "Fuse":
                _decoders.Add(tokens => [string.Concat(tokens)]);
                break;
            case "Strip":
                string content = (string)d["content"]!;
                int start = (int?)d["start"] ?? 0, stop = (int?)d["stop"] ?? 0;
                _decoders.Add(tokens =>
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
                });
                break;
            case "Metaspace":
                string replacement = (string?)d["replacement"] ?? "▁";
                _decoders.Add(tokens =>
                {
                    var text = string.Concat(tokens).Replace(replacement, " ", StringComparison.Ordinal);
                    return [text.StartsWith(' ') ? text[1..] : text];
                });
                break;
            default:
                throw new NotSupportedException($"Tokenizer decoder '{type}' is not supported.");
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
