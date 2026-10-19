// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Abstraction.Generation;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The Unigram model (plan 24, T1: SentencePiece's unigram language model, the tokenizer of T5, ALBERT, XLNet, XLM-R and
/// many multilingual models), registered as "Unigram" in <see cref="TokenizerModels"/>. Added tokens are matched first;
/// the text between them runs through the span pipeline (<see cref="TokenizerPipeline"/>), and each piece is segmented by
/// Viterbi over a lattice of its UTF-8 bytes whose nodes start at character boundaries, the vocabulary's matches found by
/// one walk of a double-array trie per position.
/// <para>
/// Two references fix the result, and the model follows the one its file came from (the model's <c>mode</c> field):
/// Hugging Face's tokenizers for a tokenizer.json (scores summed in <c>double</c>, every piece in the trie, the unknown
/// score the lowest score minus 10), SentencePiece for a <c>.model</c> file, which the sentencepiece reader marks
/// <c>"mode": "sentencepiece"</c> (scores summed in <c>float</c>, control, unknown and byte pieces not matched in text,
/// user-defined pieces with SentencePiece's bonus, the unknown score the lowest normal score minus 10). Both sum left to
/// right and keep the first of equal scores, so the two differ only where float rounding makes a tie (a run of 300
/// <c>a</c>s). Runs of unknown characters fuse into one unknown id; with byte fallback an unknown run becomes the
/// <c>&lt;0xNN&gt;</c> pieces of its UTF-8 bytes.
/// </para>
/// <para>
/// Encoded pieces are cached by span (<see cref="TokenizerOptions.PieceCacheLimit"/>); the lattice lives in arrays the thread keeps,
/// so encoding into a reused list allocates nothing once warm. Subword regularization (<see cref="TokenizerOptions.SamplingAlpha"/>)
/// samples a segmentation instead, bypassing the cache.
/// </para>
/// </summary>
internal sealed class UnigramTokenizer : IPretrainedTokenizer
{
    /// <summary>The model type tokenizer.json names.</summary>
    public const string Type = TokenizerModels.Unigram;

    /// <summary>The model field that names the reference the lattice follows.</summary>
    public const string ModeField = "mode";

    /// <summary>The mode field's value for SentencePiece's lattice (what the sentencepiece reader writes).</summary>
    public const string SentencePieceMode = "sentencepiece";

    /// <summary>The mode field's value for Hugging Face's lattice (the default when the field is absent).</summary>
    public const string HuggingFaceMode = "huggingface";

    private const double UnknownPenalty = 10.0;              // both references' kUnkPenalty
    private const int CachedChars = 256;                     // longer pieces (unsplit text) rarely repeat

    private const int KeptBytes = 1 << 16;                   // a thread keeps lattice arrays for pieces up to this many bytes

    [ThreadStatic]
    private static PreTokenizedText? t_text;

    [ThreadStatic]
    private static Scratch? t_scratch;

    private readonly JsonObject _source;                     // the tokenizer.json without the model's vocab, for Save
    private readonly IReadOnlyList<(string Piece, double Score)> _pieces;
    private string[] _tokens;                                // id → token (pieces, added tokens over them), padded by PadVocabulary
    private readonly Dictionary<string, int> _ids;
    private readonly Dictionary<string, int>.AlternateLookup<ReadOnlySpan<char>> _idSpans;
    private readonly HashSet<int> _special;
    private readonly Regex? _addedPattern;
    private readonly TokenizerPipeline _pipeline;
    private readonly Func<List<string>, List<string>>? _decoder;

    private readonly DoubleArrayTrie? _trie;                 // the matchable pieces' UTF-8 → id
    private readonly int _longestPiece;                      // the longest key in bytes (bounds the matches at a position)
    private readonly bool _sentencePiece;
    private readonly double[] _scores;                       // by id (Hugging Face: summed in double)
    private readonly float[] _singleScores;                  // by id (SentencePiece: summed in float)
    private readonly bool[]? _userDefined;                   // SentencePiece: user-defined pieces take its bonus
    private readonly double _unknownScore, _maxScore;
    private readonly float _unknownSingle, _maxSingle;
    private readonly int _unknown;                           // the unknown piece's id, -1 when there is none
    private readonly bool _byteFallback;
    private readonly int[] _byteIds = new int[256];          // each byte's <0xNN> piece (-1 when missing)

    private readonly int _cacheLimit;
    private int _cached;                                     // pieces in the cache (counted on add)
    private readonly ConcurrentDictionary<string, int[]> _cache = new();
    private readonly ConcurrentDictionary<string, int[]>.AlternateLookup<ReadOnlySpan<char>> _cacheSpans;

    private readonly int _workers;                           // batch threads (TokenizerOptions.Workers)
    private TokenizerEncodings.RawEncoder? _raw;             // Raw, as Finish takes it (made once)
    private ITokenizerPostProcessor? _postProcessor;
    private bool _postProcessorRead;

    private readonly double? _alpha;                         // sampling (null: off)
    private readonly int _best;
    private readonly Random _random;
    private readonly Lock _randomLock = new();

    private UnigramTokenizer(string source, JsonObject json, IReadOnlyList<(string Piece, double Score)> pieces, TokenizerOptions options)
    {
        var model = json["model"] as JsonObject ?? throw new InvalidDataException($"{source}: the tokenizer has no model.");
        if ((string?)model["type"] != Type)
        {
            throw new NotSupportedException($"{source}: tokenizer model '{model["type"]}' is not Unigram.");
        }

        _source = json;
        _pieces = pieces;
        _cacheLimit = options.ResolvedPieceCacheLimit();
        _cacheSpans = _cache.GetAlternateLookup<ReadOnlySpan<char>>();
        _workers = options.ResolvedWorkers();
        _alpha = options.ResolvedSamplingAlpha();
        _best = options.ResolvedSamplingBest();
        if (_best == 1)
        {
            _alpha = null;                                   // the best one: SentencePiece's nbest_size 1 does not sample
        }

        _random = options.ResolvedSamplingSeed() is { } seed ? new Random(seed) : new Random();
        _sentencePiece = model[ModeField] switch
        {
            null => false,
            JsonValue v when v.TryGetValue(out string? name) && name == SentencePieceMode => true,
            JsonValue v when v.TryGetValue(out string? name) && name == HuggingFaceMode => false,
            var other => throw new InvalidDataException(
                $"{source}: the Unigram model's mode is {other.ToJsonString()}: give \"{SentencePieceMode}\" or \"{HuggingFaceMode}\" (or leave it out)."),
        };
        _byteFallback = (bool?)model["byte_fallback"] ?? false;
        _unknown = -1;
        if (model["unk_id"] is JsonValue unk)
        {
            int id = (int)unk;
            if ((uint)id >= (uint)pieces.Count)
            {
                throw new InvalidDataException($"{source}: the Unigram model's unk_id is {id}, outside its {pieces.Count} pieces.");
            }

            _unknown = id;
        }

        // The vocabulary: pieces by id (a repeated piece names its last id, as Hugging Face's map does), then the added
        // tokens over them.
        _ids = new Dictionary<string, int>(pieces.Count + 16, StringComparer.Ordinal);
        for (int id = 0; id < pieces.Count; id++)
        {
            _ids[pieces[id].Piece] = id;
        }

        Span<char> bytePiece = ['<', '0', 'x', '0', '0', '>'];
        var lookup = _ids.GetAlternateLookup<ReadOnlySpan<char>>();
        for (int b = 0; b < 256; b++)
        {
            bytePiece[3] = "0123456789ABCDEF"[b >> 4];
            bytePiece[4] = "0123456789ABCDEF"[b & 15];
            _byteIds[b] = lookup.TryGetValue(bytePiece, out int id) ? id : -1;
        }

        var added = (json["added_tokens"] as JsonArray ?? []).OfType<JsonObject>()
            .Select(t => (Content: (string)t["content"]!, Id: (int)t["id"]!, Special: (bool?)t["special"] ?? false)).ToList();
        int size = pieces.Count;
        foreach (var (_, id, _) in added)
        {
            size = Math.Max(size, id + 1);
        }

        _tokens = new string[size];
        for (int id = 0; id < pieces.Count; id++)
        {
            _tokens[id] = pieces[id].Piece;
        }

        _special = [];
        var control = new HashSet<int>();
        var userDefined = new HashSet<int>();
        foreach (var (content, id, special) in added)
        {
            if (id < 0)
            {
                throw new InvalidDataException($"{source}: added token '{content}' has id {id}.");
            }

            _tokens[id] = content;
            _ids[content] = id;
            if (special)
            {
                _special.Add(id);
                control.Add(id);
            }
            else if (id < pieces.Count)
            {
                userDefined.Add(id);
            }
        }

        _idSpans = _ids.GetAlternateLookup<ReadOnlySpan<char>>();
        if (added.Count > 0)
        {
            _addedPattern = new Regex(string.Join('|', added.Select(a => a.Content).Where(c => c.Length > 0)
                .Distinct(StringComparer.Ordinal).OrderByDescending(c => c.Length).Select(Regex.Escape)), BpeTokenizer.Fast);
        }

        // The trie and the scores. Hugging Face matches every piece; SentencePiece only normal, user-defined (and unused)
        // ones: control and unknown pieces (special added tokens) and byte pieces are reserved there.
        _scores = new double[pieces.Count];
        _singleScores = new float[pieces.Count];
        var keys = new Dictionary<string, int>(pieces.Count, StringComparer.Ordinal);
        double min = double.PositiveInfinity, max = double.NegativeInfinity;
        float minSingle = float.MaxValue, maxSingle = float.MinValue;
        if (_sentencePiece && userDefined.Count > 0)
        {
            _userDefined = new bool[pieces.Count];
        }

        for (int id = 0; id < pieces.Count; id++)
        {
            var (piece, score) = pieces[id];
            _scores[id] = score;
            _singleScores[id] = (float)score;
            bool reserved = _sentencePiece && (control.Contains(id) || (_byteFallback && IsBytePiece(piece)));
            if (!reserved && piece.Length > 0 && !piece.Contains('\0', StringComparison.Ordinal))
            {
                keys[piece] = id;
            }

            if (!_sentencePiece)
            {
                min = Math.Min(min, score);
            }
            else if (!reserved && !userDefined.Contains(id))
            {
                minSingle = Math.Min(minSingle, (float)score);    // SentencePiece: normal pieces only
                maxSingle = Math.Max(maxSingle, (float)score);
            }

            if (_userDefined is not null && userDefined.Contains(id))
            {
                _userDefined[id] = true;
            }
        }

        _unknownScore = min - UnknownPenalty;
        _maxScore = max;
        _unknownSingle = minSingle - (float)UnknownPenalty;
        _maxSingle = maxSingle;
        if (keys.Count > 0)
        {
            _trie = DoubleArrayTrie.Build([.. keys.Keys], [.. keys.Values]);
            foreach (var key in keys.Keys)
            {
                _longestPiece = Math.Max(_longestPiece, Encoding.UTF8.GetByteCount(key));
            }
        }

        _pipeline = new TokenizerPipeline(json);
        _decoder = json["decoder"] is JsonObject decoder ? DecoderOf(decoder) : null;
    }

    // The library's "Unigram" model (TokenizerModels): the tokenizer of a definition, read from its pieces and scores.
    internal static UnigramTokenizer FromDefinition(TokenizerDefinition definition, TokenizerOptions options) =>
        new(definition.Source, definition.Json,
            definition.Pieces ?? throw new InvalidDataException($"{definition.Source}: the Unigram model has no vocabulary of [piece, score] pairs (model.vocab)."),
            options);

    /// <inheritdoc />
    public string Model => Type;

    /// <inheritdoc />
    public int VocabularySize => _tokens.Length;

    /// <inheritdoc />
    public void PadVocabulary(int size)
    {
        if (size > _tokens.Length)
        {
            Array.Resize(ref _tokens, size);
        }
    }

    /// <inheritdoc />
    public int? IdOf(string token) => _ids.TryGetValue(token, out int id) ? id : null;

    /// <inheritdoc />
    public string TokenOf(int id) => (uint)id < (uint)_tokens.Length ? _tokens[id] ?? "" : "";

    /// <inheritdoc />
    public bool IsSpecial(int id) => _special.Contains(id);

    /// <inheritdoc />
    public IReadOnlyList<int> Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var ids = new List<int>(text.Length / 3 + 4);
        EncodeRange(text, 0, text.Length, ids);
        return ids;
    }

    /// <inheritdoc />
    public void EncodeRange(string text, int start, int length, List<int> ids)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(ids);
        ArgumentOutOfRangeException.ThrowIfNegative(start);
        ArgumentOutOfRangeException.ThrowIfGreaterThan(length, text.Length - start);
        if (HasLoneSurrogate(text.AsSpan(start, length)))
        {
            text = BpeTokenizer.WithoutLoneSurrogates(text.Substring(start, length));   // as UTF-8 writes it: U+FFFD
            start = 0;
        }

        int last = start;
        if (_addedPattern is not null)
        {
            foreach (var match in _addedPattern.EnumerateMatches(text.AsSpan(start, length)))
            {
                int at = start + match.Index;
                EncodeText(text, last, at - last, ids, last == start);
                ids.Add(_idSpans[text.AsSpan(at, match.Length)]);
                last = at + match.Length;
            }
        }

        EncodeText(text, last, start + length - last, ids, last == start);
    }

    /// <inheritdoc />
    public TokenizerEncoding Encode(string text, string? pair, TokenizerEncodeOptions options) =>
        TokenizerEncodings.Finish(text, pair, _raw ??= Raw, PostProcessor, options, _source, this);

    /// <inheritdoc />
    public IReadOnlyList<TokenizerEncoding> EncodeBatch(IReadOnlyList<string> texts, IReadOnlyList<string>? pairs, TokenizerEncodeOptions options) =>
        TokenizerEncodings.Batch(texts, pairs, options, _source, this, _workers);

    // The post-processor the tokenizer.json names, or null.
    internal ITokenizerPostProcessor? PostProcessor
    {
        get
        {
            if (!_postProcessorRead)
            {
                _postProcessor = PostProcessors.Of(_source["post_processor"] as JsonObject);
                _postProcessorRead = true;
            }

            return _postProcessor;
        }
    }

    // A text's ids (as Encode gives them, the piece cache included) with their offsets in it: each piece's tokens aligned
    // to the characters they came from.
    private TokenizerEncoding Raw(string text, int sequence, int limit, bool left)
    {
        ArgumentNullException.ThrowIfNull(text);
        text = BpeTokenizer.WithoutLoneSurrogates(text);                      // one character for one: offsets still index the text
        var ids = new List<int>(text.Length / 3 + 4);
        var offsets = new List<(int Start, int End)>(ids.Capacity);
        var ranges = new List<(int Start, int Length)>();
        var starts = left && limit < int.MaxValue ? new List<(int Start, bool Added)>() : null;  // each pre-token's first token
        bool full = false;                                                    // a right limit reached: the rest is not encoded
        var pieces = t_text ?? new PreTokenizedText();
        t_text = null;
        try
        {
            void Text(int start, int length, bool atStart)
            {
                if (length == 0 || full)
                {
                    return;
                }

                _pipeline.Run(text, start, length, atStart, pieces);
                for (int i = 0; i < pieces.Pieces.Count; i++)
                {
                    var (pieceStart, pieceLength) = pieces.Pieces[i];
                    if (pieceLength == 0)
                    {
                        continue;
                    }

                    starts?.Add((ids.Count, false));
                    int before = ids.Count;
                    var piece = pieces.Piece(i);
                    EncodePiece(piece, ids);
                    ranges.Clear();
                    TokenizerEncodings.Align(piece, CollectionsMarshal.AsSpan(ids)[before..], TokenOf, _byteFallback, ranges);
                    if (_byteFallback && !_sentencePiece)
                    {
                        TokenizerEncodings.FuseByteRuns(CollectionsMarshal.AsSpan(ids)[before..], TokenOf, ranges);
                    }

                    foreach (var (from, count) in ranges)
                    {
                        offsets.Add(pieces.Text.OriginOf(pieceStart + from, count));
                    }

                    if (starts is null && ids.Count >= limit)
                    {
                        full = true;
                        return;
                    }
                }
            }

            int last = 0;
            if (_addedPattern is not null)
            {
                foreach (var match in _addedPattern.EnumerateMatches(text))
                {
                    Text(last, match.Index - last, last == 0);
                    if (full)
                    {
                        break;
                    }

                    starts?.Add((ids.Count, true));
                    ids.Add(_idSpans[text.AsSpan(match.Index, match.Length)]);
                    offsets.Add((match.Index, match.Index + match.Length));
                    last = match.Index + match.Length;
                }
            }

            Text(last, text.Length - last, last == 0);
        }
        finally
        {
            t_text = pieces;
        }

        return TokenizerEncodings.Raw(ids, offsets, TokenOf, sequence, starts is null ? 0 : TokenizerEncodings.Limited(starts, ids.Count, limit));
    }

    // The text between added tokens: normalized and pre-tokenized on spans, then each piece.
    private void EncodeText(string text, int start, int length, List<int> ids, bool atStart)
    {
        if (length == 0)
        {
            return;
        }

        var pre = t_text ??= new PreTokenizedText();
        _pipeline.Run(text, start, length, atStart, pre);
        for (int i = 0; i < pre.Pieces.Count; i++)
        {
            EncodePiece(pre.Piece(i), ids);
        }
    }

    // A piece: its cached ids, or the lattice (a sampled encode bypasses the cache).
    private void EncodePiece(ReadOnlySpan<char> piece, List<int> ids)
    {
        if (piece.IsEmpty)
        {
            return;
        }

        bool sampled = _alpha is not null;
        if (!sampled && _cacheSpans.TryGetValue(piece, out var cached))
        {
            ids.AddRange(cached);
            return;
        }

        int before = ids.Count;
        int most = Encoding.UTF8.GetMaxByteCount(piece.Length);
        if (sampled)
        {
            byte[] rented = ArrayPool<byte>.Shared.Rent(most);
            try
            {
                Sample(rented.AsSpan(0, Encoding.UTF8.GetBytes(piece, rented)), ids);
            }
            finally
            {
                ArrayPool<byte>.Shared.Return(rented);
            }

            return;
        }

        // The lattice's arrays: the thread's own for pieces up to KeptBytes (no pool round trips per piece), pooled ones
        // returned at once for longer pieces (unsplit text), so a thread keeps no more than that.
        bool kept = most <= KeptBytes;
        var scratch = kept ? (t_scratch ??= new Scratch()) : Scratch.Rent(most, _longestPiece + 1);
        try
        {
            scratch.Ensure(most, _longestPiece + 1);
            var utf8 = scratch.Bytes.AsSpan(0, Encoding.UTF8.GetBytes(piece, scratch.Bytes));
            if (_sentencePiece)
            {
                Viterbi(utf8, _singleScores, _unknownSingle, _maxSingle, scratch.Single, scratch, ids);
            }
            else
            {
                Viterbi(utf8, _scores, _unknownScore, _maxScore, scratch.Double, scratch, ids);
            }
        }
        finally
        {
            if (!kept)
            {
                scratch.Return();
            }
        }

        // The cache's size is counted here (ConcurrentDictionary.Count takes every lock), so a miss costs no lock.
        if (piece.Length <= CachedChars && Volatile.Read(ref _cached) < _cacheLimit
            && _cacheSpans.TryAdd(piece, CollectionsMarshal.AsSpan(ids)[before..].ToArray()))
        {
            Interlocked.Increment(ref _cached);
        }
    }

    // The best segmentation of a piece's UTF-8 bytes, appended to ids. best[e]: the best score of a path ending at byte e;
    // from[e]: where its last node starts (-1: no path yet); node[e]: that node's id (-1: an unknown character). Scores sum
    // left to right in T (double for Hugging Face, float for SentencePiece); a later candidate replaces only a strictly
    // better one, as both references do.
    private void Viterbi<T>(ReadOnlySpan<byte> bytes, T[] scores, T unknownScore, T maxScore, T[] best, Scratch scratch, List<int> ids)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        int n = bytes.Length;
        int[] from = scratch.From, node = scratch.Node, path = scratch.Path;
        TrieMatch[] matches = scratch.Matches;
        from.AsSpan(0, n + 1).Fill(-1);
        best[0] = T.Zero;
        var userDefined = _userDefined;
        T bonus = T.CreateChecked(0.1);
        for (int start = 0; start < n;)
        {
            int width = Math.Min(CharWidth(bytes[start]), n - start);
            T here = best[start];
            bool single = false;
            int found = _trie is null ? 0 : _trie.CommonPrefixSearch(bytes[start..], matches);
            for (int m = 0; m < found; m++)
            {
                var (id, length) = matches[m];
                int end = start + length;
                T score = userDefined is not null && userDefined[id] ? T.CreateChecked(length) * maxScore - bonus : scores[id];
                T candidate = score + here;
                if (from[end] < 0 || candidate > best[end])
                {
                    best[end] = candidate;
                    from[end] = start;
                    node[end] = id;
                }

                single |= length == width;
            }

            if (!single)
            {
                int end = start + width;
                T candidate = unknownScore + here;
                if (from[end] < 0 || candidate > best[end])
                {
                    best[end] = candidate;
                    from[end] = start;
                    node[end] = -1;
                }
            }

            start += width;
        }

        // Back from the end, the path's node ends; then the nodes forward.
        int count = 0;
        for (int end = n; end > 0; end = from[end])
        {
            path[count++] = end;
        }

        var writer = new PathWriter(this, bytes, ids);
        for (int i = count - 1; i >= 0; i--)
        {
            int end = path[i];
            writer.Add(from[end], end, node[end]);
        }

        writer.Flush();
    }

    // A thread's lattice arrays (see EncodePiece): the piece's bytes, and per byte position the best score (double or
    // float), where the best node ending there starts, its id and the path; the trie's matches at one position.
    private sealed class Scratch
    {
        public byte[] Bytes = [];
        public double[] Double = [];
        public float[] Single = [];
        public int[] From = [], Node = [], Path = [];
        public TrieMatch[] Matches = [];
        private bool _rented;

        // Arrays from the pool for one long piece (Return gives them back).
        public static Scratch Rent(int bytes, int matches) => new()
        {
            _rented = true,
            Bytes = ArrayPool<byte>.Shared.Rent(bytes),
            Double = ArrayPool<double>.Shared.Rent(bytes + 1),
            Single = ArrayPool<float>.Shared.Rent(bytes + 1),
            From = ArrayPool<int>.Shared.Rent(bytes + 1),
            Node = ArrayPool<int>.Shared.Rent(bytes + 1),
            Path = ArrayPool<int>.Shared.Rent(bytes + 1),
            Matches = ArrayPool<TrieMatch>.Shared.Rent(matches),
        };

        // Room for a piece of `bytes` UTF-8 bytes and `matches` trie matches (a kept scratch grows, doubling).
        public void Ensure(int bytes, int matches)
        {
            if (Bytes.Length < bytes)
            {
                int size = Math.Max(bytes, Math.Max(256, Bytes.Length * 2));
                Bytes = new byte[size];
                Double = new double[size + 1];
                Single = new float[size + 1];
                From = new int[size + 1];
                Node = new int[size + 1];
                Path = new int[size + 1];
            }

            if (Matches.Length < matches)
            {
                Matches = new TrieMatch[matches];
            }
        }

        public void Return()
        {
            if (_rented)
            {
                ArrayPool<byte>.Shared.Return(Bytes);
                ArrayPool<double>.Shared.Return(Double);
                ArrayPool<float>.Shared.Return(Single);
                ArrayPool<int>.Shared.Return(From);
                ArrayPool<int>.Shared.Return(Node);
                ArrayPool<int>.Shared.Return(Path);
                ArrayPool<TrieMatch>.Shared.Return(Matches);
            }
        }
    }

    // The ids of a path's nodes, in order: a run of unknown nodes is one unknown id (fuse_unk), or with byte fallback the
    // <0xNN> pieces of its bytes (the unknown id when some byte has no piece, as Hugging Face's does).
    private ref struct PathWriter(UnigramTokenizer owner, ReadOnlySpan<byte> bytes, List<int> ids)
    {
        private readonly ReadOnlySpan<byte> _bytes = bytes;
        private int _unknownStart = -1;
        private int _unknownEnd;

        public void Add(int start, int end, int id)
        {
            if (id < 0 || id == owner._unknown)
            {
                if (_unknownStart < 0)
                {
                    _unknownStart = start;
                }

                _unknownEnd = end;
                return;
            }

            Flush();
            ids.Add(id);
        }

        public void Flush()
        {
            if (_unknownStart < 0)
            {
                return;
            }

            var run = _bytes[_unknownStart.._unknownEnd];
            _unknownStart = -1;
            if (owner._byteFallback)
            {
                bool all = true;
                foreach (byte b in run)
                {
                    all &= owner._byteIds[b] >= 0;
                }

                if (all)
                {
                    foreach (byte b in run)
                    {
                        ids.Add(owner._byteIds[b]);
                    }

                    return;
                }
            }

            if (owner._unknown < 0)
            {
                throw new InvalidOperationException(
                    $"The Unigram model has no unknown piece (unk_id), and '{Encoding.UTF8.GetString(run)}' is not covered by its pieces"
                    + (owner._byteFallback ? " or its byte pieces." : "."));
            }

            ids.Add(owner._unknown);
        }
    }

    // Subword regularization: the lattice's edges, then a segmentation drawn from the n best (SamplingBest > 1, each with
    // probability proportional to exp(alpha × its score)) or from the whole lattice (forward filtering, backward sampling).
    private void Sample(ReadOnlySpan<byte> bytes, List<int> ids)
    {
        int n = bytes.Length;
        var edges = new Edges(n * 2 + 16);
        int[] head = ArrayPool<int>.Shared.Rent(n + 1);                   // the last edge ending at each byte (-1: none)
        TrieMatch[] matches = ArrayPool<TrieMatch>.Shared.Rent(_longestPiece + 1);
        int[] path = ArrayPool<int>.Shared.Rent(n + 1);
        try
        {
            head.AsSpan(0, n + 1).Fill(-1);
            for (int start = 0; start < n;)
            {
                int width = Math.Min(CharWidth(bytes[start]), n - start);
                bool single = false;
                int found = _trie is null ? 0 : _trie.CommonPrefixSearch(bytes[start..], matches);
                for (int m = 0; m < found; m++)
                {
                    var (id, length) = matches[m];
                    double score = _userDefined is not null && _userDefined[id] ? length * (double)_maxSingle - 0.1
                        : _sentencePiece ? _singleScores[id] : _scores[id];
                    edges.Add(start, start + length, id, score, head);
                    single |= length == width;
                }

                if (!single)
                {
                    edges.Add(start, start + width, -1, _sentencePiece ? _unknownSingle : _unknownScore, head);
                }

                start += width;
            }

            double alpha = _alpha!.Value;
            int count;
            lock (_randomLock)
            {
                count = _best > 1 ? SampleBest(ref edges, n, alpha, path) : SampleLattice(ref edges, head, n, alpha, path);
            }

            var writer = new PathWriter(this, bytes, ids);
            for (int i = count - 1; i >= 0; i--)
            {
                int e = path[i];
                writer.Add(edges.Start[e], edges.End[e], edges.Id[e]);
            }

            writer.Flush();
        }
        finally
        {
            edges.Dispose();
            ArrayPool<int>.Shared.Return(head);
            ArrayPool<TrieMatch>.Shared.Return(matches);
            ArrayPool<int>.Shared.Return(path);
        }
    }

    // Forward filtering, backward sampling: forward[e] is the log of the summed exp(alpha × score) of every path ending at
    // e (edges come in order of their start, so forward[start] is complete when its edges are added); then from the end an
    // edge into the current position is drawn with its share of forward there. Writes the edges, last first; returns how many.
    private int SampleLattice(ref Edges edges, int[] head, int n, double alpha, int[] path)
    {
        double[] forward = ArrayPool<double>.Shared.Rent(n + 1);
        try
        {
            forward.AsSpan(0, n + 1).Fill(double.NegativeInfinity);
            forward[0] = 0;
            for (int e = 0; e < edges.Count; e++)
            {
                int end = edges.End[e];
                forward[end] = LogAdd(forward[end], forward[edges.Start[e]] + alpha * edges.Score[e]);
            }

            int count = 0;
            for (int at = n; at > 0;)
            {
                double draw = _random.NextDouble(), sum = 0;
                int chosen = -1;
                for (int e = head[at]; e >= 0; e = edges.Next[e])
                {
                    chosen = e;
                    sum += Math.Exp(forward[edges.Start[e]] + alpha * edges.Score[e] - forward[at]);
                    if (draw < sum)
                    {
                        break;
                    }
                }

                path[count++] = chosen;
                at = edges.Start[chosen];
            }

            return count;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(forward);
        }
    }

    // The n best paths (each position keeps its best k partial paths, best first, an equal score after the earlier one),
    // then one drawn with probability proportional to exp(alpha × score), as SentencePiece's SampleEncode with nbest_size k.
    private int SampleBest(ref Edges edges, int n, double alpha, int[] path)
    {
        int k = _best;
        int slots = (n + 1) * k;
        double[] score = ArrayPool<double>.Shared.Rent(slots);
        int[] edge = ArrayPool<int>.Shared.Rent(slots);
        int[] rank = ArrayPool<int>.Shared.Rent(slots);
        int[] kept = ArrayPool<int>.Shared.Rent(n + 1);
        try
        {
            kept.AsSpan(0, n + 1).Clear();
            kept[0] = 1;
            score[0] = 0;
            edge[0] = -1;
            for (int e = 0; e < edges.Count; e++)
            {
                int start = edges.Start[e], end = edges.End[e];
                for (int r = 0; r < kept[start]; r++)
                {
                    double candidate = score[start * k + r] + edges.Score[e];
                    int size = kept[end];
                    int at = size;
                    while (at > 0 && candidate > score[end * k + at - 1])
                    {
                        at--;
                    }

                    if (at >= k)
                    {
                        continue;
                    }

                    for (int j = Math.Min(size, k - 1); j > at; j--)
                    {
                        score[end * k + j] = score[end * k + j - 1];
                        edge[end * k + j] = edge[end * k + j - 1];
                        rank[end * k + j] = rank[end * k + j - 1];
                    }

                    score[end * k + at] = candidate;
                    edge[end * k + at] = e;
                    rank[end * k + at] = r;
                    kept[end] = Math.Min(size + 1, k);
                }
            }

            // Draw one of the n best: weights exp(alpha × (score − the best's)).
            int paths = kept[n];
            double top = score[n * k], total = 0;
            for (int r = 0; r < paths; r++)
            {
                total += Math.Exp(alpha * (score[n * k + r] - top));
            }

            double draw = _random.NextDouble() * total, sum = 0;
            int chosen = paths - 1;
            for (int r = 0; r < paths; r++)
            {
                sum += Math.Exp(alpha * (score[n * k + r] - top));
                if (draw < sum)
                {
                    chosen = r;
                    break;
                }
            }

            int count = 0;
            for (int at = n, r = chosen; at > 0;)
            {
                int e = edge[at * k + r];
                path[count++] = e;
                r = rank[at * k + r];
                at = edges.Start[e];
            }

            return count;
        }
        finally
        {
            ArrayPool<double>.Shared.Return(score);
            ArrayPool<int>.Shared.Return(edge);
            ArrayPool<int>.Shared.Return(rank);
            ArrayPool<int>.Shared.Return(kept);
        }
    }

    private static double LogAdd(double x, double y)
    {
        if (double.IsNegativeInfinity(x))
        {
            return y;
        }

        double high = Math.Max(x, y), low = Math.Min(x, y);
        return high + double.LogP1(Math.Exp(low - high));
    }

    // A sampled lattice's edges in pooled arrays, in the order they were found (by start), each linked to the previous
    // edge with the same end.
    private struct Edges(int capacity) : IDisposable
    {
        public int[] Start = ArrayPool<int>.Shared.Rent(capacity);
        public int[] End = ArrayPool<int>.Shared.Rent(capacity);
        public int[] Id = ArrayPool<int>.Shared.Rent(capacity);
        public int[] Next = ArrayPool<int>.Shared.Rent(capacity);
        public double[] Score = ArrayPool<double>.Shared.Rent(capacity);
        public int Count;

        public void Add(int start, int end, int id, double score, int[] head)
        {
            if (Count == Start.Length)
            {
                Grow(ref Start, Count);
                Grow(ref End, Count);
                Grow(ref Id, Count);
                Grow(ref Next, Count);
                Grow(ref Score, Count);
            }

            Start[Count] = start;
            End[Count] = end;
            Id[Count] = id;
            Score[Count] = score;
            Next[Count] = head[end];
            head[end] = Count;
            Count++;
        }

        public void Dispose()
        {
            ArrayPool<int>.Shared.Return(Start);
            ArrayPool<int>.Shared.Return(End);
            ArrayPool<int>.Shared.Return(Id);
            ArrayPool<int>.Shared.Return(Next);
            ArrayPool<double>.Shared.Return(Score);
        }

        private static void Grow<TItem>(ref TItem[] array, int used)
        {
            var larger = ArrayPool<TItem>.Shared.Rent(array.Length * 2);
            array.AsSpan(0, used).CopyTo(larger);
            ArrayPool<TItem>.Shared.Return(array);
            array = larger;
        }
    }

    /// <inheritdoc />
    public string Decode(IEnumerable<int> ids) => ids switch
    {
        null => throw new ArgumentNullException(nameof(ids)),
        int[] array => Decode(array.AsSpan()),
        List<int> list => Decode(CollectionsMarshal.AsSpan(list)),
        _ => Decode([.. ids]),
    };

    /// <inheritdoc />
    /// <remarks>
    /// The tokens run through the tokenizer.json's decoder chain (ids without a token are left out); with no decoder they
    /// are joined with spaces, as Hugging Face's tokenizers does.
    /// </remarks>
    public string Decode(ReadOnlySpan<int> ids)
    {
        var tokens = new List<string>(ids.Length);
        foreach (int id in ids)
        {
            if ((uint)id < (uint)_tokens.Length && _tokens[id] is { } token)
            {
                tokens.Add(token);
            }
        }

        return _decoder is null ? string.Join(' ', tokens) : string.Concat(_decoder(tokens));
    }

    /// <inheritdoc />
    public void Save(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        if (Directory.Exists(path))
        {
            path = Path.Combine(path, "tokenizer.json");
        }

        string? folder = Path.GetDirectoryName(Path.GetFullPath(path));
        if (folder is not null)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None, 1 << 16);
        WriteTo(stream);
    }

    /// <inheritdoc />
    public JsonObject ToJson()
    {
        using var stream = new MemoryStream();
        WriteTo(stream);
        return JsonNode.Parse(stream.ToArray())!.AsObject();
    }

    /// <inheritdoc />
    /// <remarks>
    /// Everything the tokenizer was read with, and the model's pieces in id order as <c>[piece, score]</c> pairs (scores
    /// written so they read back bit for bit), so the file loads back to the same ids.
    /// </remarks>
    public void WriteTo(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        var options = new JsonWriterOptions { Indented = true, Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };
        using var writer = new Utf8JsonWriter(stream, options);
        writer.WriteStartObject();
        foreach (var (name, value) in _source)
        {
            writer.WritePropertyName(name);
            if (name != "model" || value is not JsonObject model)
            {
                Write(writer, value);
                continue;
            }

            writer.WriteStartObject();
            foreach (var (field, fieldValue) in model)
            {
                if (field != "vocab")
                {
                    writer.WritePropertyName(field);
                    Write(writer, fieldValue);
                }
            }

            writer.WriteStartArray("vocab");
            foreach (var (piece, score) in _pieces)
            {
                writer.WriteStartArray();
                writer.WriteStringValue(piece);
                writer.WriteNumberValue(score);
                writer.WriteEndArray();
            }

            writer.WriteEndArray();
            writer.WriteEndObject();
        }

        writer.WriteEndObject();

        static void Write(Utf8JsonWriter writer, JsonNode? node)
        {
            if (node is null)
            {
                writer.WriteNullValue();
            }
            else
            {
                node.WriteTo(writer);
            }
        }
    }

    // The decoder a tokenizer.json names: the app's component when one is registered for its type, else the built-in.
    private static Func<List<string>, List<string>> DecoderOf(JsonObject spec)
    {
        if (TokenizerComponents.FindDecoder((string?)spec["type"]) is { } create)
        {
            var part = create(spec);
            return tokens => part.Decode(tokens) is var result && result is List<string> list ? list : [.. result];
        }

        return BpeTokenizer.DecoderStage(spec);
    }

    // The UTF-8 length of the character a lead byte starts (the text is valid UTF-8: .NET wrote it).
    private static int CharWidth(byte lead) => lead < 0x80 ? 1 : lead < 0xE0 ? 2 : lead < 0xF0 ? 3 : 4;

    private static bool IsBytePiece(string piece) => piece.Length == 6 && piece.StartsWith("<0x", StringComparison.Ordinal) && piece[5] == '>'
                                                     && char.IsAsciiHexDigitUpper(piece[3]) && char.IsAsciiHexDigitUpper(piece[4]);

    // Whether the text holds half of a surrogate pair (a range may cut a pair).
    private static bool HasLoneSurrogate(ReadOnlySpan<char> text)
    {
        int at = text.IndexOfAnyInRange('\uD800', '\uDFFF');
        if (at < 0)
        {
            return false;
        }

        for (int i = at; i < text.Length; i++)
        {
            if (char.IsHighSurrogate(text[i]) && i + 1 < text.Length && char.IsLowSurrogate(text[i + 1]))
            {
                i++;
            }
            else if (char.IsSurrogate(text[i]))
            {
                return true;
            }
        }

        return false;
    }
}
