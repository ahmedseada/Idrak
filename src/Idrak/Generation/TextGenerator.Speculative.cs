// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Runtime.InteropServices;
using Idrak.Abstraction.Diagnostics;

namespace Idrak.Generation;

// Speculative decoding (GenerationOptions.Draft): a draft model proposes a few tokens, this generator checks them all in one
// pass over its key/value cache and keeps the ones it agrees with, then both caches go back to what was kept.
public sealed partial class TextGenerator
{
    // The cache this generator keeps as another generator's draft between generations (StartDraft), apart from KeepCache's
    // own, so an inline completion's prompt on the same model is not thrown away by a chat's drafting; with the ids it holds.
    private DecodingContext? _draftKept;
    private List<int> _draftKeptIds = [];

    /// <summary>
    /// Whether a key/value cache is kept from this generator's last drafting for another (<see cref="StartDraft"/>, while
    /// <see cref="KeepCache"/> is on), apart from its own (<see cref="HoldsCache"/>); <see cref="ReleaseCache"/> frees it.
    /// </summary>
    public bool HoldsDraftCache
    {
        get
        {
            lock (_keptLock)
            {
                return _draftKept is not null;
            }
        }
    }

    /// <summary>
    /// Starts drafting for another generator of this vocabulary (speculative decoding, <see cref="GenerationOptions.Draft"/>):
    /// the session feeds <paramref name="prompt"/> through this model's own key/value cache on its first proposal (the
    /// prefix it shares with the last drafting's, kept while <see cref="KeepCache"/> is on, is not fed again) and samples as
    /// <paramref name="options"/> say, on the host (the sampler's rule, <see cref="TokenSampler.Probabilities"/>).
    /// </summary>
    public IDraftSession StartDraft(IReadOnlyList<int> prompt, GenerationOptions options)
    {
        ArgumentNullException.ThrowIfNull(prompt);
        ArgumentNullException.ThrowIfNull(options);
        return new DraftSession(this, prompt, options);
    }

    // A draft this generator cannot check: another vocabulary, or a sampler of its own (the check draws from the library's
    // sampling rule on the host, which another sampler would not follow).
    private void CheckDraft(IDraftModel draft)
    {
        if (SpeculativeDecoding.Refusal(Tokenizer, draft) is { } reason)
        {
            throw new InvalidOperationException(reason);
        }

        if (CreateSampler != (Func<SamplerRequest, ITokenSampler>)TokenSamplers.Create || TokenSamplers.Origin(TokenSamplers.DefaultName) != Overrides.Library)
        {
            throw new InvalidOperationException("Speculative decoding checks the draft's tokens by the library's sampling rule, and this generator samples with a "
                + "sampler of its own (TextGenerator.CreateSampler, or the app's TokenSamplers \"default\"): generate without GenerationOptions.Draft.");
        }
    }

    // The width of the distributions a round compares: the wider of this model's logits and the draft's (ids past a model's
    // own width have probability 0 under it, a vocabulary padded further in one of them).
    private static int CommonWidth(int target, IDraftModel draft) => Math.Max(target, draft.Vocabulary);

    // The most tokens a round may propose under `options`.
    private static int MostPerRound(GenerationOptions options) =>
        options.DraftTokens > 0 ? options.DraftTokens : Math.Max(options.DraftMinTokens, options.DraftMaxTokens);

    // Stream with a draft: the prompt's pass as without one, then rounds of the draft's proposals checked by one pass each.
    private IEnumerable<GenerationChunk> Speculate(string prompt, IDraftModel draft, GenerationOptions options, CancellationToken cancellationToken)
    {
        var total = Stopwatch.StartNew();
        int context = Math.Clamp(options.NumCtx, 2, ContextLength);
        int limit = options.NumPredict > 0 ? options.NumPredict : context;
        var history = new List<int>(Tokenizer.Encode(prompt));
        if (history.Count == 0)
        {
            history.Add(0);
        }

        if (history.Count > context - 1)
        {
            history.RemoveRange(0, history.Count - (context - 1));          // keep the most recent part of the prompt
        }

        if (!options.SlideContext)
        {
            limit = Math.Min(limit, context - history.Count);               // the reply ends where the window does
        }

        int promptTokens = history.Count;
        int[]? promptIds = options.ReturnTokens || options.ReturnLogProbabilities ? [.. history] : null;
        var reply = new ReplyText(Tokenizer, options, HiddenTokens(options));
        int vocabulary = SampledVocabulary(), width = CommonWidth(vocabulary, draft), most = MostPerRound(options);
        var rule = new HostRule(options, (uint)(options.Seed ?? Random.Shared.Next()));
        int cachedTokens = 0, resets = 0, proposedTotal = 0, acceptedTotal = 0, rounds = 0;
        TimeSpan promptDuration = TimeSpan.Zero;

        // Host buffers for a round, rented once per generation and returned at its end: the target's rows, the draft's
        // distributions (sampling only), and one row's working copy and distribution.
        float[] rows = ArrayPool<float>.Shared.Rent((most + 1) * vocabulary);
        float[] drafted = ArrayPool<float>.Shared.Rent(SpeculativeDecoding.IsGreedy(options) ? 1 : most * width);
        float[] work = ArrayPool<float>.Shared.Rent(width);
        float[] target = ArrayPool<float>.Shared.Rent(width);
        int[] proposals = ArrayPool<int>.Shared.Rent(most);
        IDraftSession? session = null;

        Model.Eval();
        TensorOffloading.BeforeInference(Device, Model);                          // with offloading: room for the pass on the device
        var (kept, keptIds) = KeepCache ? TakeCache(context) : (null, []);
        int capacity = KeepCache ? context : Math.Min(context, history.Count + limit);
        var decoding = kept ?? new DecodingContext(Device, 1, capacity, Layout);
        decoding.LastPositionOnly = true;
        decoding.Cancellation = cancellationToken;                          // the prompt's pass stops between layers
        bool keep = false;
        int window = 0;                                                     // history's index at the cache's position 0 (moves when a full window is re-read)
        try
        {
            // The cached pass over history[from..] (in pieces when long), its last position's logits copied to `rows`; the
            // pieces it went through kept for the telemetry.
            (int Pieces, int Size) fedShape = (1, 0);
            void Feed(int from)
            {
                int count = history.Count - from;
                int piece = PrefillPiece(options, decoding, count);
                fedShape = PieceShape(count, piece);
                float[] values = ArrayPool<float>.Shared.Rent(count);
                try
                {
                    var ids = CollectionsMarshal.AsSpan(history)[from..];
                    for (int i = 0; i < count; i++)
                    {
                        values[i] = ids[i];
                    }

                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    Tensor logits;
                    using (PrefillScope())
                    {
                        if (count <= piece)
                        {
                            using var input = Tensor.From(values.AsSpan(0, count), [1, count], Device);
                            logits = Model.ForwardCached(input, decoding);
                        }
                        else
                        {
                            logits = ForwardPieces(values.AsSpan(0, count), 1, count, decoding, piece);
                        }
                    }

                    logits.CopyTo(rows.AsSpan(0, vocabulary), logits.Size - vocabulary);
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(values);
                }
            }

            // The token chosen from the logits in rows[0..vocabulary) after the history so far (the prompt's pass, a re-read
            // window), with its probability.
            (int Id, float Probability) Choose()
            {
                rule.Distribution(rows.AsSpan(0, vocabulary), CollectionsMarshal.AsSpan(history), work, target.AsSpan(0, width));
                int id = rule.Pick(target.AsSpan(0, width));
                return (id, target[id]);
            }

            long prefillStart = Telemetry.StartStage();
            int shared = 0;
            while (shared < keptIds.Count && shared < history.Count && keptIds[shared] == history[shared])
            {
                shared++;
            }

            shared = Math.Min(shared, history.Count - 1);
            (int Id, float Probability) first;
            try
            {
                if (kept is not null && shared > 0 && shared <= decoding.Length)
                {
                    decoding.Truncate(shared);
                    Feed(shared);
                    cachedTokens = shared;
                }
                else
                {
                    decoding.Reset();
                    Feed(0);
                }

                first = Choose();
                decoding.Cancellation = CancellationToken.None;             // a round is never stopped part-way
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested && KeepCache && decoding.Length <= history.Count)
            {
                KeepCacheFor(decoding, history.GetRange(0, decoding.Length));   // as Stream: what was fed whole is kept
                keep = true;
                throw;
            }

            promptDuration = total.Elapsed;
            if (prefillStart != 0)
            {
                Telemetry.InferenceStage(InferenceStageKind.Prefill, Device, prefillStart, promptTokens, 0, 1, fedShape.Pieces, fedShape.Size);
            }

            reply.Add(first.Id, first.Probability);
            history.Add(first.Id);
            int produced = 1, chunkFrom = 1, round = SpeculativeDecoding.FirstRound(options);
            long chunkStart = Telemetry.StartStage();
            while (reply.DoneReason is null && !cancellationToken.IsCancellationRequested)
            {
                if (produced - chunkFrom >= Math.Max(1, options.ChunkSize) || produced >= limit)
                {
                    string piece = reply.Take(final: false);
                    Telemetry.InferenceStage(InferenceStageKind.DecodeChunk, Device, chunkStart, produced - chunkFrom);
                    (chunkFrom, chunkStart) = (produced, Telemetry.StartStage());
                    if (piece.Length > 0)
                    {
                        yield return new GenerationChunk(piece) { Generated = reply.Generated.Count };
                    }
                }

                // Cancelled while the reader held the chunk: no round after it.
                if (reply.DoneReason is not null || produced >= limit || cancellationToken.IsCancellationRequested)
                {
                    break;
                }

                int pending = history.Count - window - decoding.Length;     // fed by the round's pass before the proposals (1)
                if (decoding.Length + pending > decoding.Capacity)
                {
                    // A full window is re-read from its last half (as Stream), its next token the target's alone.
                    decoding.Reset();
                    decoding.Cancellation = cancellationToken;
                    window = history.Count - context / 2;
                    Feed(window);
                    decoding.Cancellation = CancellationToken.None;
                    resets++;
                    var next = Choose();
                    reply.Add(next.Id, next.Probability);
                    history.Add(next.Id);
                    session?.Accept(0, next.Id);
                    produced++;
                    continue;
                }

                if (session is null)
                {
                    session = draft.StartDraft(history.GetRange(0, history.Count - 1), options);
                    session.Accept(0, history[^1]);
                }

                int ask = Math.Min(Math.Min(round, limit - produced - 1), decoding.Capacity - decoding.Length - pending);
                long draftStart = Telemetry.StartStage();
                var ids = proposals.AsSpan(0, Math.Max(0, ask));
                int proposed = ask > 0 ? session.Propose(ids, rule.Greedy ? [] : drafted.AsSpan(0, ask * width), width, cancellationToken) : 0;
                if (proposed > 0)
                {
                    Telemetry.InferenceStage(InferenceStageKind.Draft, Device, draftStart, proposed, 0, 1, proposed);
                }

                if (cancellationToken.IsCancellationRequested)
                {
                    break;                                                   // within the draft's steps: no pass after them
                }

                // The target's one pass over the pending token and the proposals, every position's logits kept.
                long verifyStart = Telemetry.StartStage();
                int length = decoding.Length, fed = pending + proposed;
                float[] input = ArrayPool<float>.Shared.Rent(fed);
                try
                {
                    for (int i = 0; i < pending; i++)
                    {
                        input[i] = history[window + length + i];
                    }

                    for (int i = 0; i < proposed; i++)
                    {
                        input[pending + i] = ids[i];
                    }

                    using var noGrad = Autograd.NoGrad();
                    using var scope = new TensorScope();
                    using var tensor = Tensor.From(input.AsSpan(0, fed), [1, fed], Device);
                    decoding.LastPositionOnly = false;
                    try
                    {
                        var logits = Model.ForwardCached(tensor, decoding);
                        int count = (proposed + 1) * vocabulary;
                        logits.CopyTo(rows.AsSpan(0, count), logits.Size - count);
                    }
                    finally
                    {
                        decoding.LastPositionOnly = true;
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(input);
                }

                // Each proposal against the target's distribution after the history and the proposals before it.
                int before = history.Count, accepted = 0;
                for (int i = 0; i < proposed; i++)
                {
                    history.Add(ids[i]);
                }

                (int Id, float Probability) chosen = default;
                bool replaced = false;
                for (int i = 0; i < proposed && !replaced; i++)
                {
                    var p = target.AsSpan(0, width);
                    rule.Distribution(rows.AsSpan(i * vocabulary, vocabulary), CollectionsMarshal.AsSpan(history)[..(before + i)], work, p);
                    int x = ids[i];
                    if (rule.Keeps(x, p, rule.Greedy ? default : drafted.AsSpan(i * width, width)))
                    {
                        reply.Add(x, p[x]);
                        accepted++;
                        continue;
                    }

                    int id = rule.Greedy ? rule.Pick(p) : rule.Replace(p, drafted.AsSpan(i * width, width));
                    chosen = (id, p[id]);
                    replaced = true;
                }

                if (!replaced)
                {
                    // Every proposal kept: the target's own next token from the position after them.
                    var p = target.AsSpan(0, width);
                    rule.Distribution(rows.AsSpan(proposed * vocabulary, vocabulary), CollectionsMarshal.AsSpan(history)[..(before + proposed)], work, p);
                    int id = rule.Pick(p);
                    chosen = (id, p[id]);
                }

                history.RemoveRange(before + accepted, proposed - accepted);
                history.Add(chosen.Id);
                reply.Add(chosen.Id, chosen.Probability);
                decoding.Truncate(length + pending + accepted);              // the cache holds history[window..^1] again
                session.Accept(accepted, chosen.Id);
                reply.CheckLoop();
                produced += accepted + 1;
                (proposedTotal, acceptedTotal, rounds) = (proposedTotal + proposed, acceptedTotal + accepted, rounds + (proposed > 0 ? 1 : 0));
                if (verifyStart != 0 && proposed > 0)
                {
                    Telemetry.InferenceStage(InferenceStageKind.Verify, Device, verifyStart, accepted + 1, 0, 1, 1, proposed + 1);
                }

                round = SpeculativeDecoding.NextRound(round, proposed, accepted, options);
            }

            string rest = reply.Take(final: true);
            if (produced > chunkFrom)
            {
                Telemetry.InferenceStage(InferenceStageKind.DecodeChunk, Device, chunkStart, produced - chunkFrom);
            }

            if (rest.Length > 0)
            {
                yield return new GenerationChunk(rest) { Generated = reply.Generated.Count };
            }

            // The cache holds the keys and values of history[..Length] (the last chosen token was never fed).
            if (KeepCache && resets == 0 && decoding.Length <= history.Count)
            {
                KeepCacheFor(decoding, history.GetRange(0, decoding.Length));
                keep = true;
            }
        }
        finally
        {
            session?.Dispose();
            if (!keep)
            {
                decoding.Dispose();
            }

            ArrayPool<float>.Shared.Return(rows);
            ArrayPool<float>.Shared.Return(drafted);
            ArrayPool<float>.Shared.Return(work);
            ArrayPool<float>.Shared.Return(target);
            ArrayPool<int>.Shared.Return(proposals);
        }

        reply.DoneReason ??= "length";
        Device.Synchronize();
        var generated = reply.Generated;
        var stats = new GenerationStats(promptTokens, promptDuration, generated.Count, total.Elapsed - promptDuration, total.Elapsed, resets)
        {
            CachedTokens = cachedTokens, DraftTokens = proposedTotal, DraftAccepted = acceptedTotal, DraftRounds = rounds,
        };
        yield return new GenerationChunk("", Done: true, DoneReason: reply.DoneReason, Stats: stats)
        {
            Generated = generated.Count,
            Tokens = promptIds is null ? null : Sampled(promptIds, generated, reply.LogProbabilities, reply.DoneReason == "stop" ? reply.StopText : null),
        };
    }

    // Speculative decoding's choices on the host, by the library's sampling rule (TokenSampler.Probabilities after
    // TokenSampler.Penalize): the distribution after a history, the greedy pick or a draw, whether a proposal is kept
    // (greedy: it is the pick; sampling: with probability min(1, p/q)), and a refused one's replacement from max(0, p - q).
    // Draws come from a counter-based stream (seed, draw), so a seed repeats a reply.
    private sealed class HostRule
    {
        private readonly GenerationOptions _options;
        private readonly uint _seed;
        private readonly bool _penalized;
        private uint _draws;

        public HostRule(GenerationOptions options, uint seed)
        {
            (_options, _seed) = (options, seed);
            Greedy = SpeculativeDecoding.IsGreedy(options);
            _penalized = options.RepeatLastN > 0 && (options.RepeatPenalty != 1f || options.PresencePenalty != 0f || options.FrequencyPenalty != 0f);
        }

        public bool Greedy { get; }

        // The distribution `logits` sample from after `seen` (penalties over its last RepeatLastN ids), into `p` (its width
        // may pass the logits': those ids get 0); `work` holds the penalized copy.
        public void Distribution(ReadOnlySpan<float> logits, ReadOnlySpan<int> seen, Span<float> work, Span<float> p)
        {
            var scores = work[..logits.Length];
            logits.CopyTo(scores);
            if (_penalized)
            {
                TokenSampler.Penalize(scores, seen[Math.Max(0, seen.Length - _options.RepeatLastN)..], _options.RepeatPenalty, _options.PresencePenalty, _options.FrequencyPenalty);
            }

            if (!TokenSampler.Probabilities(scores, p, _options.Temperature, _options.TopK, _options.TopP, _options.MinP))
            {
                throw new InvalidOperationException("Nothing to sample: every logit was NaN or -∞, so the model's output is not finite (an overflow in its "
                    + "weights or activations, or a bad input). NaN logits are never sampled and +∞ ones win; a row needs at least one of either.");
            }

            p[logits.Length..].Clear();
        }

        // The greedy pick (the most likely id, the lowest on a tie) or a draw from `p`.
        public int Pick(ReadOnlySpan<float> p) => Greedy ? Best(p) : Draw(p, Next());

        // Whether proposal `x` is kept: greedy, when it is the pick; sampling, with probability min(1, p(x) / q(x)).
        public bool Keeps(int x, ReadOnlySpan<float> p, ReadOnlySpan<float> q)
        {
            if ((uint)x >= (uint)p.Length)
            {
                return false;
            }

            if (Greedy)
            {
                return Best(p) == x;
            }

            float u = Next();
            return q[x] > 0f && u * q[x] < p[x];
        }

        // A refused proposal's replacement: a draw from max(0, p - q), normalized (from p when that is empty: p and q agree).
        public int Replace(Span<float> p, ReadOnlySpan<float> q)
        {
            double sum = 0;
            for (int j = 0; j < p.Length; j++)
            {
                sum += Math.Max(0f, p[j] - q[j]);
            }

            if (!(sum > 0))
            {
                return Draw(p, Next());
            }

            double target = Next() * sum, cumulative = 0;
            int last = -1;
            for (int j = 0; j < p.Length; j++)
            {
                float w = Math.Max(0f, p[j] - q[j]);
                if (w > 0f)
                {
                    last = j;
                    cumulative += w;
                    if (cumulative > target)
                    {
                        return j;
                    }
                }
            }

            return last;
        }

        private float Next() => Uniform(_seed, _draws++);

        private static int Best(ReadOnlySpan<float> p)
        {
            int best = 0;
            for (int j = 1; j < p.Length; j++)
            {
                if (p[j] > p[best])
                {
                    best = j;
                }
            }

            return best;
        }

        // A draw from `p` (summing to 1) by `u` in [0, 1): the first id whose running sum passes it.
        internal static int Draw(ReadOnlySpan<float> p, float u)
        {
            float cumulative = 0f;
            int last = -1;
            for (int j = 0; j < p.Length; j++)
            {
                if (p[j] > 0f)
                {
                    last = j;
                    cumulative += p[j];
                    if (cumulative > u)
                    {
                        return j;
                    }
                }
            }

            return Math.Max(last, 0);
        }

        // A uniform float in [0, 1) from (seed, counter): the MurmurHash3 finalizer, as the sampler's stream.
        private static float Uniform(uint seed, uint counter)
        {
            uint h = seed ^ (counter * 0x9E3779B9u) ^ 0x5BD1E995u;
            h ^= h >> 16;
            h *= 0x85EBCA6Bu;
            h ^= h >> 13;
            h *= 0xC2B2AE35u;
            h ^= h >> 16;
            return (h >> 8) * (1f / 16777216f);
        }
    }

    // One generation's drafting on this generator (StartDraft): its own cache over the ids accepted so far (from history
    // index `_start`, which moves when a full window is re-read from its last half), the proposals fed after them, and the
    // host rule it draws with (its own stream).
    private sealed class DraftSession : IDraftSession
    {
        private readonly TextGenerator _owner;
        private readonly GenerationOptions _options;
        private readonly HostRule _rule;
        private readonly DecodingContext _decoding;
        private readonly List<int> _seq;                 // the ids accepted so far, then the current proposals
        private readonly List<int> _fed = [];            // the ids in the cache, from _seq[_start]
        private readonly float[] _row;
        private readonly float[] _work;
        private readonly float[] _p;
        private int _accepted;                           // _seq's ids the target accepted (the rest are proposals)
        private int _settled;                            // _seq's first ids the cache agrees with, as far as it holds them
        private int _start;
        private int _proposed;
        private bool _disposed;

        public DraftSession(TextGenerator owner, IReadOnlyList<int> prompt, GenerationOptions options)
        {
            (_owner, _options) = (owner, options);
            _rule = new HostRule(options, (uint)(options.Seed ?? Random.Shared.Next()) ^ 0xA5A5A5A5u);
            _seq = [.. prompt];
            _accepted = _seq.Count;
            int capacity = Math.Clamp(options.NumCtx, 2, owner.ContextLength);
            int width = owner.SampledVocabulary();
            _row = ArrayPool<float>.Shared.Rent(width);
            _work = ArrayPool<float>.Shared.Rent(width);
            _p = ArrayPool<float>.Shared.Rent(width);
            owner.Model.Eval();
            DecodingContext? kept = null;
            List<int> keptIds = [];
            lock (owner._keptLock)
            {
                if (owner._draftKept is { } held && held.Capacity == capacity && held.Layout == owner.Layout && held.Device == owner.Device)
                {
                    (kept, keptIds) = (held, owner._draftKeptIds);
                }
                else
                {
                    owner._draftKept?.Dispose();
                }

                (owner._draftKept, owner._draftKeptIds) = (null, []);
            }

            _decoding = kept ?? new DecodingContext(owner.Device, 1, capacity, owner.Layout);
            _decoding.LastPositionOnly = true;
            int shared = 0;
            while (kept is not null && shared < keptIds.Count && shared < _seq.Count && shared < _decoding.Length && keptIds[shared] == _seq[shared])
            {
                shared++;
            }

            if (shared > 0)
            {
                _decoding.Truncate(shared);
                _fed.AddRange(keptIds.Take(shared));
            }
            else
            {
                _decoding.Reset();
            }
        }

        private int Width => _owner.SampledVocabulary();

        public int Propose(Span<int> ids, Span<float> probabilities, int width, CancellationToken cancellationToken)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (!_rule.Greedy && probabilities.Length < ids.Length * width)
            {
                throw new ArgumentException($"{probabilities.Length} probabilities for {ids.Length} proposals of {width} ids.", nameof(probabilities));
            }

            int vocabulary = Width;
            if (width < vocabulary && !_rule.Greedy)
            {
                throw new ArgumentOutOfRangeException(nameof(width), width, $"The rows are {width} wide and the draft's logits {vocabulary}.");
            }

            _settled = Math.Min(_settled, _accepted);
            _seq.RemoveRange(_accepted, _seq.Count - _accepted);           // a proposal not answered is dropped
            Trim();
            _proposed = 0;
            for (int i = 0; i < ids.Length; i++)
            {
                if (cancellationToken.IsCancellationRequested || !Feed(cancellationToken))
                {
                    break;
                }

                _rule.Distribution(_row.AsSpan(0, vocabulary), CollectionsMarshal.AsSpan(_seq), _work, _p.AsSpan(0, vocabulary));
                var p = _p.AsSpan(0, vocabulary);
                int x = _rule.Pick(p);
                ids[i] = x;
                if (!_rule.Greedy)
                {
                    var row = probabilities.Slice(i * width, width);
                    p.CopyTo(row);
                    row[vocabulary..].Clear();
                }

                _seq.Add(x);
                _proposed++;
            }

            return _proposed;
        }

        // Feeds the ids of _seq not in the cache yet and reads the last position's logits into _row; a full window is
        // re-read from its last half first. False when cancelled part-way (the cache keeps what was fed whole).
        private bool Feed(CancellationToken cancellationToken)
        {
            int from = _start + _decoding.Length;
            int count = _seq.Count - from;
            if (count <= 0 && _decoding.Length > 0)
            {
                // Every id is in the cache already (a proposal asked again without an answer): the last one is fed again
                // for its logits.
                _decoding.Truncate(_decoding.Length - 1);
                _fed.RemoveAt(_fed.Count - 1);
                (from, count) = (from - 1, 1);
            }

            if (_decoding.Length + count > _decoding.Capacity)
            {
                int keep = Math.Min(_seq.Count, Math.Max(1, _decoding.Capacity / 2));
                _decoding.Reset();
                _fed.Clear();
                _start = _seq.Count - keep;
                (from, count) = (_start, keep);
            }

            int vocabulary = Width;
            int piece = _owner.PrefillPiece(_options, _decoding, count);
            float[] values = ArrayPool<float>.Shared.Rent(count);
            try
            {
                var source = CollectionsMarshal.AsSpan(_seq).Slice(from, count);
                for (int i = 0; i < count; i++)
                {
                    values[i] = source[i];
                }

                using var noGrad = Autograd.NoGrad();
                using var scope = new TensorScope();
                _decoding.Cancellation = count > 1 ? cancellationToken : CancellationToken.None;   // a long pass stops between layers
                Tensor logits;
                using (_owner.PrefillScope())
                {
                    if (count <= piece)
                    {
                        using var input = Tensor.From(values.AsSpan(0, count), [1, count], _owner.Device);
                        logits = _owner.Model.ForwardCached(input, _decoding);
                    }
                    else
                    {
                        logits = _owner.ForwardPieces(values.AsSpan(0, count), 1, count, _decoding, piece);
                    }
                }

                logits.CopyTo(_row.AsSpan(0, vocabulary), logits.Size - vocabulary);
                return true;
            }
            catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
            {
                return false;
            }
            finally
            {
                _decoding.Cancellation = CancellationToken.None;
                while (_fed.Count < _decoding.Length)
                {
                    _fed.Add(_seq[_start + _fed.Count]);
                }

                ArrayPool<float>.Shared.Return(values);
            }
        }

        public void Accept(int accepted, int next)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentOutOfRangeException.ThrowIfNegative(accepted);
            if (accepted > _proposed || _seq.Count - _accepted != _proposed)
            {
                throw new ArgumentOutOfRangeException(nameof(accepted), accepted, $"The last proposal was {_proposed} tokens.");
            }

            _accepted += accepted;
            _settled = Math.Min(_settled, _accepted);
            _seq.RemoveRange(_accepted, _seq.Count - _accepted);
            _seq.Add(next);
            _accepted++;
            _proposed = 0;
            Trim();
        }

        // Cuts the cache after the ids it holds that _seq still has in their places (the proposals after a refused one go;
        // the target's own token stays when it is the proposal fed there). Only ids from _settled on can differ.
        private void Trim()
        {
            int valid = Math.Clamp(_settled - _start, 0, _decoding.Length);
            while (valid < _decoding.Length && _start + valid < _seq.Count && _fed[valid] == _seq[_start + valid])
            {
                valid++;
            }

            if (valid < _decoding.Length)
            {
                _decoding.Truncate(valid);
                _fed.RemoveRange(valid, _fed.Count - valid);
            }

            _settled = _start + _decoding.Length;
        }

        public void Dispose()
        {
            if (_disposed)
            {
                return;
            }

            _disposed = true;
            ArrayPool<float>.Shared.Return(_row);
            ArrayPool<float>.Shared.Return(_work);
            ArrayPool<float>.Shared.Return(_p);
            if (_owner.KeepCache && _start == 0 && _fed.Count == _decoding.Length)
            {
                lock (_owner._keptLock)
                {
                    _owner._draftKept?.Dispose();
                    (_owner._draftKept, _owner._draftKeptIds) = (_decoding, [.. _fed]);
                }
            }
            else
            {
                _decoding.Dispose();
            }
        }
    }
}
