// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;

internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] SpeculativeGroup =
    [
        ("speculative decoding: greedy tokens with a draft are the target's alone (rule 71) over several prompts: a draft that is the target, a different one, penalties, stop sequences, a kept cache reused, fixed and adapted rounds, a window re-read past its end", GreedySameTokens),
        ("speculative decoding: the host rule is the sampler's: TokenSampler.Probabilities and Penalize give the CPU sampler's distribution (its five most likely ids and their probabilities) with and without penalties", HostRuleIsTheSamplers),
        ("speculative decoding: sampled with a different draft, the replies follow the target's own distribution on a tiny vocabulary (every 3-token reply's share against its exact probability, fixed seeds), proposals both kept and refused", SampledDistributionIsTheTargets),
        ("speculative decoding: a draft of another vocabulary is refused naming the id that differs; one padded further (empty entries, wider logits) drafts; a generator with a sampler of its own refuses a draft", MismatchRefused),
        ("speculative decoding: rounds adapt to the acceptance (up to DraftMaxTokens with a draft that agrees, down to DraftMinTokens with one that does not), a fixed DraftTokens is kept, and the Draft and Verify stages add up to the reply's statistics", RoundsAdapt),
        ("speculative decoding: a cancellation stops the reply within one round: from the reader (no round after its chunk) and from inside the draft's proposals (no pass after them)", CancelWithinARound),
        ("speculative decoding: memory stays flat over many requests (device memory in use and the managed heap), one cache kept for the target and one for the draft, ReleaseCache frees both", MemoryFlat),
        ("speculative decoding: tokens a second with and without a draft on a tiny pair on this device (cold and warm apart, rule 76), the same tokens", SpeedMeasured),
    ];

    // A target and two drafts over one character vocabulary: the target decoder, a smaller decoder of other weights.
    private static (Sequential Target, Sequential Draft, CharTokenizer Tokenizer, DecoderSpec Spec) SpeculativePair(Device device, int vocabularyPad = 0)
    {
        var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,<>|/_{}\":\n");
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 96,
            Rope = new RopeSettings(500f, null, false, null),
        };
        var target = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 3 });
        var draft = (spec with { Layers = 1, Vocabulary = tokenizer.VocabularySize + vocabularyPad }).Build(null, new DecoderBuildOptions { Device = device, Seed = 11 });
        return (target, draft, tokenizer, spec);
    }

    internal static void GreedySameTokens(Device device)
    {
        var (target, other, tokenizer, _) = SpeculativePair(device);
        var (tiny, _) = TinyLanguageModel(device, context: 96);
        using var _1 = target;
        using var _2 = other;
        using var _3 = tiny;
        string[] prompts = ["the quick brown fox", "a", "hello world, hello", "{\"key\": \"value\"}\n", "the lazy dog sleeps under the old tree by the river"];
        var greedy = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, Seed = 1, NumPredict = 30, NumCtx = 96, ChunkSize = 4, ReturnTokens = true };
        var cases = new (string Name, Sequential Model, Sequential Draft, GenerationOptions Options)[]
        {
            ("the target drafting for itself", target, target, greedy),
            ("a different draft", target, other, greedy),
            ("a different draft, penalties", target, other, greedy with { RepeatPenalty = 1.3f, PresencePenalty = 0.2f, FrequencyPenalty = 0.1f, RepeatLastN = 8 }),
            ("a different draft, a stop sequence", target, other, greedy with { Stop = ["e ", "\n"] }),
            ("a different draft, two tokens a round", target, other, greedy with { DraftTokens = 2, ChunkSize = 1 }),
            ("a different draft, at most 7 a round", target, other, greedy with { DraftMinTokens = 3, DraftMaxTokens = 7 }),
            ("a decoder drafting for the tiny model", tiny, target, greedy),
            ("a window re-read past its end", target, other, greedy with { NumCtx = 24, NumPredict = 60 }),
        };
        foreach (var (name, model, draftModel, options) in cases)
        {
            foreach (bool keepCache in new[] { false, true })
            {
                var alone = new TextGenerator(model, tokenizer, 96) { KeepCache = keepCache };
                var withDraft = new TextGenerator(model, tokenizer, 96) { KeepCache = keepCache };
                var draft = ReferenceEquals(draftModel, model) ? withDraft : new TextGenerator(draftModel, tokenizer, 96) { KeepCache = keepCache };
                int proposed = 0, kept = 0;
                foreach (string prompt in prompts.Concat(prompts.Take(2)))           // the last two reuse kept caches
                {
                    var expected = alone.Stream(prompt, options).ToList();
                    var got = withDraft.Stream(prompt, options with { Draft = draft }).ToList();
                    string what = $"{name}, KeepCache {keepCache}, '{prompt}'";
                    Check(string.Concat(got.Select(c => c.Text)) == string.Concat(expected.Select(c => c.Text)),
                        $"{what}: '{string.Concat(got.Select(c => c.Text))}' with the draft, '{string.Concat(expected.Select(c => c.Text))}' alone");
                    var (e, g) = (expected[^1], got[^1]);
                    Check(g.Tokens!.Reply.SequenceEqual(e.Tokens!.Reply) && g.Tokens.Prompt.SequenceEqual(e.Tokens.Prompt) && g.DoneReason == e.DoneReason,
                        $"{what}: ids [{string.Join(",", g.Tokens!.Reply)}] ({g.DoneReason}) with the draft, [{string.Join(",", e.Tokens!.Reply)}] ({e.DoneReason}) alone");
                    Check(g.Stats!.DraftAccepted <= g.Stats.DraftTokens && g.Stats.DraftRounds > 0 == g.Stats.DraftTokens > 0 && e.Stats!.DraftTokens == 0,
                        $"{what}: statistics {g.Stats.DraftAccepted} of {g.Stats.DraftTokens} in {g.Stats.DraftRounds} rounds");
                    (proposed, kept) = (proposed + g.Stats.DraftTokens, kept + g.Stats.DraftAccepted);
                }

                Check(proposed > 0, $"{name}, KeepCache {keepCache}: the draft proposed tokens");
                if (ReferenceEquals(draftModel, model))
                {
                    Check(kept >= proposed * 0.95, $"{name}, KeepCache {keepCache}: the target agrees with itself: {kept} of {proposed} kept");
                }
            }
        }
    }

    internal static void HostRuleIsTheSamplers(Device device)
    {
        int vocabulary = 50;
        var random = new Random(5);
        float[] logits = [.. Enumerable.Range(0, vocabulary).Select(_ => (float)(random.NextDouble() * 6 - 3))];
        int[] history = [3, 7, 7, 12, 3, 40, 7];
        var settings = new (float Temperature, int TopK, float TopP, float MinP, float Repeat, float Presence, float Frequency)[]
        {
            (1f, 0, 1f, 0f, 1f, 0f, 0f), (0.7f, 10, 0.9f, 0f, 1f, 0f, 0f), (1.3f, 0, 0.8f, 0.05f, 1.2f, 0.3f, 0.1f), (0.5f, 1, 1f, 0f, 1.5f, 0f, 0f),
        };
        foreach (var (temperature, topK, topP, minP, repeat, presence, frequency) in settings)
        {
            using var sampler = new TokenSampler(device, 1, vocabulary, 4, 16)
            {
                Temperature = temperature, TopK = topK, TopP = topP, MinP = minP, RepeatPenalty = repeat, PresencePenalty = presence, FrequencyPenalty = frequency, RepeatLastN = 16, Seed = 1,
            };
            sampler.SetHistory(history);
            using (var row = Tensor.From(logits, [1, vocabulary], device))
            {
                sampler.Sample(row);
            }

            var token = sampler.Read(0, 1)[0][0];
            float[] scores = [.. logits];
            if (repeat != 1f || presence != 0f || frequency != 0f)
            {
                TokenSampler.Penalize(scores, history, repeat, presence, frequency);
            }

            float[] p = new float[vocabulary];
            Check(TokenSampler.Probabilities(scores, p, temperature, topK, topP, minP), "a row with finite logits has something to sample");
            Check(Math.Abs(p.Sum() - 1f) < 1e-4f, $"the distribution sums to 1: {p.Sum()}");
            string what = $"T {temperature}, top-k {topK}, top-p {topP}, min-p {minP}, penalties {repeat}/{presence}/{frequency}";
            foreach (var (id, probability) in token.Alternatives.Where(a => a.Id >= 0))
            {
                Check(Math.Abs(p[id] - probability) <= 2e-5f, $"{what}: id {id} has {p[id]} on the host and {probability} in the sampler");
            }

            Check(p[token.Id] > 0f && Math.Abs(p[token.Id] - token.Probability) <= 2e-5f, $"{what}: the sampled id {token.Id} has {p[token.Id]} on the host, {token.Probability} in the sampler");
            Check(p.Count(v => v > 0f) >= token.Alternatives.Count(a => a.Id >= 0), $"{what}: the host keeps every id the sampler can draw");
        }

        float[] nothing = [float.NaN, float.NegativeInfinity];
        Check(!TokenSampler.Probabilities(nothing, new float[2], 1f, 0, 1f, 0f), "a row of NaN and -∞ has nothing to sample");
    }

    internal static void SampledDistributionIsTheTargets(Device device)
    {
        var tokenizer = new CharTokenizer("abcd");
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 8, Layers = 1, Heads = 2, KvHeads = 1, HeadDim = 4, FfDim = 12, MaxPositions = 16,
            Rope = new RopeSettings(500f, null, false, null),
        };
        using var targetModel = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 21 });
        using var draftModel = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 22 });
        // Sharper than the weights alone give, so the two models disagree: scale both heads.
        foreach (var model in new[] { targetModel, draftModel })
        {
            foreach (var p in model.Last().Parameters())
            {
                p.Load([.. p.ToArray().Select(v => v * 6f)]);
            }
        }

        var target = new TextGenerator(targetModel, tokenizer, 16);
        var draft = new TextGenerator(draftModel, tokenizer, 16);
        const string Prompt = "abca";
        const int Length = 3;
        var options = new GenerationOptions { Temperature = 1f, TopK = 0, TopP = 1f, RepeatPenalty = 1f, NumPredict = Length, NumCtx = 16, ReturnTokens = true, DraftTokens = 2 };

        // The exact probability of every reply of Length tokens under a model, by its own passes and the sampler's rule.
        Dictionary<string, double> Exact(Sequential model)
        {
            var exact = new Dictionary<string, double>(StringComparer.Ordinal);
            var ids = tokenizer.Encode(Prompt).ToList();
            void Walk(List<int> sequence, double probability, string reply)
            {
                if (reply.Length == Length)
                {
                    exact[reply] = probability;
                    return;
                }

                float[] logits;
                using (Autograd.NoGrad())
                using (var scope = new TensorScope())
                {
                    var output = model.Forward(Tensor.From([.. sequence.Select(i => (float)i)], [1, sequence.Count], device));
                    var all = output.ToArray();
                    logits = all[^tokenizer.VocabularySize..];
                }

                float[] p = new float[logits.Length];
                TokenSampler.Probabilities(logits, p, options.Temperature, options.TopK, options.TopP, options.MinP);
                for (int id = 0; id < p.Length; id++)
                {
                    Walk([.. sequence, id], probability * p[id], reply + tokenizer.Decode([id]));
                }
            }

            Walk(ids, 1, "");
            return exact;
        }

        var expected = Exact(targetModel);
        var draftOwn = Exact(draftModel);
        double apart = expected.Keys.Sum(k => Math.Abs(expected[k] - draftOwn[k])) / 2;
        Check(apart > 0.2, $"the draft's own distribution is far from the target's (total variation {apart:F3}), so the check can tell them apart");

        const int Runs = 3000;
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        int proposed = 0, kept = 0;
        for (int seed = 0; seed < Runs; seed++)
        {
            var (text, _, stats) = target.Generate(Prompt, options with { Seed = seed, Draft = draft });
            counts[text] = counts.GetValueOrDefault(text) + 1;
            (proposed, kept) = (proposed + stats.DraftTokens, kept + stats.DraftAccepted);
        }

        double distance = expected.Keys.Sum(k => Math.Abs(counts.GetValueOrDefault(k) / (double)Runs - expected[k])) / 2;
        Check(counts.Keys.All(expected.ContainsKey), $"every reply is {Length} characters of the vocabulary: {string.Join(", ", counts.Keys)}");
        Check(distance < 0.06, $"the replies' shares are within 0.06 (total variation) of the target's exact distribution: {distance:F4} "
            + $"(the draft's own is {apart:F3} away); kept {kept} of {proposed} proposals");
        Check(kept > 0 && kept < proposed, $"proposals both kept and refused: {kept} of {proposed}");
    }

    // A tokenizer equal to `inner` with `extra` padding ids past it (empty entries, decoded as nothing).
    private sealed class PaddedTokens(ITokenizer inner, int extra) : ITokenizer
    {
        public int VocabularySize => inner.VocabularySize + extra;

        public IReadOnlyList<int> Encode(string text) => inner.Encode(text);

        public string Decode(IEnumerable<int> ids) => inner.Decode(ids.Where(id => id < inner.VocabularySize));

        public string TokenOf(int id) => id < inner.VocabularySize ? inner.TokenOf(id) : "";
    }

    internal static void MismatchRefused(Device device)
    {
        var (targetModel, draftModel, tokenizer, spec) = SpeculativePair(device);
        using var _1 = targetModel;
        using var _2 = draftModel;
        var target = new TextGenerator(targetModel, tokenizer, 96);
        var options = new GenerationOptions { TopK = 1, Seed = 1, NumPredict = 12, NumCtx = 96, ReturnTokens = true };
        var shuffled = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,<>|/_{}\":\n".Replace("q", "Q", StringComparison.Ordinal));
        var stranger = new TextGenerator(draftModel, shuffled, 96);
        Check(SpeculativeDecoding.Refusal(tokenizer, stranger) is { } reason && reason.Contains("id 16", StringComparison.Ordinal), "the refusal names the id that differs");
        ThrowsNaming<InvalidOperationException>(() => target.Generate("the fox", options with { Draft = stranger }), "a draft of another vocabulary",
            "vocabulary is not the target's", "id 16", "'Q' in the draft", "'q' in the target");

        var longer = new TextGenerator(draftModel, new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,<>|/_{}\":\n#"), 96);
        ThrowsNaming<InvalidOperationException>(() => target.Generate("the fox", options with { Draft = longer }), "a draft with a real token past the target's",
            "vocabulary is not the target's", "not padding");

        using var widerModel = (spec with { Layers = 1, Vocabulary = tokenizer.VocabularySize + 2 }).Build(null, new DecoderBuildOptions { Device = device, Seed = 11 });
        var padded = new TextGenerator(widerModel, new PaddedTokens(tokenizer, 2), 96);
        Check(SpeculativeDecoding.Refusal(tokenizer, padded) is null, "a vocabulary padded further (empty entries) drafts");
        var alone = target.Stream("the fox", options).Last();
        var drafted = target.Stream("the fox", options with { Draft = padded }).Last();
        Check(drafted.Tokens!.Reply.SequenceEqual(alone.Tokens!.Reply) && drafted.Stats!.DraftTokens > 0,
            $"a draft with wider logits gives the target's tokens: [{string.Join(",", drafted.Tokens!.Reply)}] vs [{string.Join(",", alone.Tokens!.Reply)}]");

        var own = new TextGenerator(targetModel, tokenizer, 96) { CreateSampler = request => TokenSampler.Create(request) };
        ThrowsNaming<InvalidOperationException>(() => own.Generate("the fox", options with { Draft = target }), "a generator with a sampler of its own",
            "sampler of its own", "GenerationOptions.Draft");
    }

    internal static void RoundsAdapt(Device device)
    {
        Check(SpeculativeDecoding.FirstRound(new GenerationOptions()) == 4 && SpeculativeDecoding.FirstRound(new GenerationOptions { DraftTokens = 6 }) == 6,
            "the first round: DraftMinTokens + 3, or DraftTokens");
        var bounded = new GenerationOptions { DraftMinTokens = 2, DraftMaxTokens = 5 };
        Check(SpeculativeDecoding.NextRound(4, 4, 4, bounded) == 5 && SpeculativeDecoding.NextRound(3, 3, 3, bounded) == 5 && SpeculativeDecoding.NextRound(3, 3, 1, bounded) == 2
              && SpeculativeDecoding.NextRound(2, 2, 0, bounded) == 2 && SpeculativeDecoding.NextRound(4, 0, 0, bounded) == 4,
            "two more after a round kept whole, one fewer after a refusal, within the bounds");

        var (targetModel, otherModel, tokenizer, _) = SpeculativePair(device);
        using var _1 = targetModel;
        using var _2 = otherModel;
        var target = new TextGenerator(targetModel, tokenizer, 96) { KeepCache = false };
        var self = new TextGenerator(targetModel, tokenizer, 96) { KeepCache = false };
        var other = new TextGenerator(otherModel, tokenizer, 96) { KeepCache = false };
        var options = new GenerationOptions { TopK = 1, Seed = 1, NumPredict = 60, NumCtx = 96, RepeatPenalty = 1f, DraftMinTokens = 1, DraftMaxTokens = 6 };
        var hook = new StageList();
        using (Telemetry.Subscribe(hook))
        {
            var stats = target.Generate("the quick brown fox", options with { Draft = self }).Stats;
            var stages = hook.Take();
            var verify = stages.Where(e => e.Stage == InferenceStageKind.Verify).ToList();
            var drafts = stages.Where(e => e.Stage == InferenceStageKind.Draft).ToList();
            Check(verify.Count == stats.DraftRounds && verify.Sum(e => e.PieceSize - 1) == stats.DraftTokens && verify.Sum(e => e.Count - 1) == stats.DraftAccepted
                  && drafts.Sum(e => e.Count) == stats.DraftTokens,
                $"the stages add up to the statistics ({stats.DraftAccepted} of {stats.DraftTokens} in {stats.DraftRounds} rounds): {string.Join("; ", verify)}");
            Check(verify.Any(e => e.PieceSize == 7) && verify.Max(e => e.PieceSize) == 7, $"a draft that agrees grows the rounds to DraftMaxTokens 6: {string.Join(", ", verify.Select(e => e.PieceSize - 1))}");
            Check(stages.Count(e => e.Stage == InferenceStageKind.Prefill) == 1 && stages.Where(e => e.Stage == InferenceStageKind.DecodeChunk).Sum(e => e.Count) == stats.GeneratedTokens - 1,
                $"one prompt pass, and the decoding chunks count every token after the first: {string.Join("; ", stages.Where(e => e.Stage != InferenceStageKind.Verify && e.Stage != InferenceStageKind.Draft))}");

            stats = target.Generate("the quick brown fox", options with { Draft = other }).Stats;
            verify = [.. hook.Take().Where(e => e.Stage == InferenceStageKind.Verify)];
            Check(stats.DraftAcceptance < 0.5 && verify.Count(e => e.PieceSize == 2) >= verify.Count / 2,
                $"a draft that disagrees ({stats.DraftAccepted} of {stats.DraftTokens} kept) shrinks the rounds to DraftMinTokens 1: {string.Join(", ", verify.Select(e => e.PieceSize - 1))}");

            target.Generate("the quick brown fox", options with { Draft = other, DraftTokens = 3 });
            verify = [.. hook.Take().Where(e => e.Stage == InferenceStageKind.Verify)];
            Check(verify.Count > 3 && verify.SkipLast(2).All(e => e.PieceSize == 4) && verify.TakeLast(2).All(e => e.PieceSize is > 1 and <= 4),
                $"DraftTokens 3 is kept every round (the last ones cut by NumPredict): {string.Join(", ", verify.Select(e => e.PieceSize - 1))}");
        }

        target.Generate("the quick brown fox", options with { Draft = other });
        Check(hook.Take().Count == 0, "nothing reported once nobody listens");
    }

    // A draft that runs `inner` and calls `onPropose` with each proposal's number (from 1) before it.
    private sealed class WatchedDraft(IDraftModel inner, Action<int> onPropose) : IDraftModel
    {
        public ITokenizer Tokenizer => inner.Tokenizer;

        public int Vocabulary => inner.Vocabulary;

        public IDraftSession StartDraft(IReadOnlyList<int> prompt, GenerationOptions options) => new Session(inner.StartDraft(prompt, options), onPropose);

        private sealed class Session(IDraftSession inner, Action<int> onPropose) : IDraftSession
        {
            private int _calls;

            public int Propose(Span<int> ids, Span<float> probabilities, int width, CancellationToken cancellationToken)
            {
                onPropose(++_calls);
                return inner.Propose(ids, probabilities, width, cancellationToken);
            }

            public void Accept(int accepted, int next) => inner.Accept(accepted, next);

            public void Dispose() => inner.Dispose();
        }
    }

    internal static void CancelWithinARound(Device device)
    {
        var (targetModel, otherModel, tokenizer, _) = SpeculativePair(device);
        using var _1 = targetModel;
        using var _2 = otherModel;
        var target = new TextGenerator(targetModel, tokenizer, 96);
        var draft = new TextGenerator(otherModel, tokenizer, 96);
        var options = new GenerationOptions { TopK = 1, Seed = 1, NumPredict = 60, NumCtx = 96, ChunkSize = 1, Draft = draft };

        using (var reader = new CancellationTokenSource())
        {
            int atCancel = -1;
            GenerationChunk? last = null;
            foreach (var chunk in target.Stream("the quick brown fox", options, reader.Token))
            {
                if (atCancel < 0 && chunk.Text.Length > 0 && chunk.Generated >= 3)
                {
                    atCancel = chunk.Generated;
                    reader.Cancel();
                }

                last = chunk;
            }

            Check(last is { Done: true } && atCancel > 0 && last.Generated == atCancel,
                $"cancelled by the reader at {atCancel} tokens: the reply ends there, {last?.Generated} tokens");
        }

        using (var inside = new CancellationTokenSource())
        {
            var watched = new WatchedDraft(draft, call =>
            {
                if (call == 3)
                {
                    inside.Cancel();
                }
            });
            var final = target.Stream("the quick brown fox", options with { Draft = watched, ChunkSize = 100 }, inside.Token).Last();
            Check(final.Done && final.Stats!.DraftRounds == 2, $"cancelled in the draft's third proposal: two rounds checked, {final.Stats!.DraftRounds}");
        }

        var again = target.Generate("the quick brown fox", options with { ChunkSize = 8 });
        Check(again.Stats.GeneratedTokens == 60, $"the generator answers whole after the cancellations: {again.Stats.GeneratedTokens} tokens");
    }

    internal static void MemoryFlat(Device device)
    {
        var (targetModel, otherModel, tokenizer, _) = SpeculativePair(device);
        using var _1 = targetModel;
        using var _2 = otherModel;
        var target = new TextGenerator(targetModel, tokenizer, 96);
        var draft = new TextGenerator(otherModel, tokenizer, 96);
        var options = new GenerationOptions { Temperature = 0.8f, TopK = 20, Seed = 3, NumPredict = 30, NumCtx = 96, Draft = draft };
        string[] prompts = ["the quick brown fox", "jumps over the lazy dog", "the quick brown fox jumps", "a b c d e f"];
        long[] device_ = new long[3];
        long[] heap = new long[3];
        int[] marks = [6, 20, 40];
        int mark = 0;
        for (int request = 1; request <= marks[^1]; request++)
        {
            var reply = target.Generate(prompts[request % prompts.Length], options with { Seed = request, TopK = request % 2 == 0 ? 1 : 20 });
            Check(reply.Stats.DraftTokens > 0, $"request {request} drafted");
            if (request == marks[mark])
            {
                device.Synchronize();
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();
                device_[mark] = ComputeResources.GetMemoryUsage(device).InUse;
                heap[mark] = GC.GetTotalMemory(forceFullCollection: true);
                mark++;
            }
        }

        Check(device_[2] <= device_[0] && device_[1] <= device_[0], $"device memory in use after 6, 20 and 40 requests: {string.Join(", ", device_)} bytes");
        Check(heap[2] - heap[0] < 1_000_000, $"the managed heap after 6, 20 and 40 requests: {string.Join(", ", heap)} bytes");
        Check(target.HoldsCache && draft.HoldsDraftCache && !draft.HoldsCache, "one cache kept for the target, one for the draft (apart from the draft's own)");
        target.ReleaseCache();
        draft.ReleaseCache();
        Check(!target.HoldsCache && !draft.HoldsDraftCache, "ReleaseCache frees both");
    }

    // A target of `layers` blocks and a draft of its first block, its embedding, norm and head: the target's other blocks are
    // scaled by `scale` (near the identity), so the draft agrees with it most of the time and costs a fraction of a pass.
    private static (Sequential Target, Sequential Draft) MeasuredPair(Device device, CharTokenizer tokenizer, int dim, int layers, float scale)
    {
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = dim, Layers = layers, Heads = 8, KvHeads = 4, HeadDim = dim / 8, FfDim = dim * 3, MaxPositions = 512,
            Rope = new RopeSettings(10000f, null, false, null),
        };
        var target = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 31 });
        var draft = (spec with { Layers = 1 }).Build(null, new DecoderBuildOptions { Device = device, Seed = 32 });
        var big = target.ToList();
        var small = draft.ToList();
        Check(big.Count == small.Count + layers - 1, $"the decoder's modules: {big.Count} for {layers} blocks, {small.Count} for one");
        void Copy(Module from, Module to)
        {
            foreach (var (a, b) in from.Parameters().Zip(to.Parameters()))
            {
                b.Load(a.ToArray());
            }
        }

        int blocks = big.Count - small.Count + 1, first = big.FindIndex(m => m is DecoderBlock);
        for (int i = 0; i < small.Count; i++)
        {
            Copy(big[i < first + 1 ? i : i + blocks - 1], small[i]);
        }

        foreach (var block in big.Skip(first + 1).Take(blocks - 1))
        {
            foreach (var p in block.Parameters())
            {
                p.Load([.. p.ToArray().Select(v => v * scale)]);
            }
        }

        return (target, draft);
    }

    internal static void SpeedMeasured(Device device)
    {
        // Two pairs: the target's other blocks nearer the identity (a draft that agrees most of the time) and farther.
        foreach (float scale in new[] { 0.15f, 0.3f })
        {
            SpeedMeasured(device, scale);
        }
    }

    private static void SpeedMeasured(Device device, float scale)
    {
        var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,<>|/_{}\":\n");
        var (targetModel, draftModel) = MeasuredPair(device, tokenizer, dim: 512, layers: 8, scale);
        using var _1 = targetModel;
        using var _2 = draftModel;
        var target = new TextGenerator(targetModel, tokenizer, 512) { KeepCache = false };
        var draft = new TextGenerator(draftModel, tokenizer, 512) { KeepCache = false };
        var options = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, Seed = 0, NumPredict = 64, NumCtx = 512, ReturnTokens = true };
        const string Prompt = "the quick brown fox jumps over the lazy dog while the river runs past the old mill";
        (double Rate, GenerationChunk Final) Run(GenerationOptions o)
        {
            var final = target.Stream(Prompt, o).Last();
            return (final.Stats!.TokensPerSecond, final);
        }

        var (coldAlone, alone) = Run(options);
        var (coldDraft, first) = Run(options with { Draft = draft });
        var warmAlone = new List<double>();
        var warmDraft = new List<double>();
        GenerationStats last = first.Stats!;
        var hook = new StageList();
        List<InferenceStage> stages;
        using (Telemetry.Subscribe(hook))
        {
            for (int i = 0; i < 5; i++)
            {
                warmAlone.Add(Run(options).Rate);
                var (rate, final) = Run(options with { Draft = draft });
                warmDraft.Add(rate);
                last = final.Stats!;
                Check(final.Tokens!.Reply.SequenceEqual(alone.Tokens!.Reply), $"run {i}: the same tokens with the draft");
            }

            stages = hook.Take();
        }

        // Where a round's time goes (the library's telemetry, rule 78): the draft a proposal, the target's pass a round, and a
        // step of the target alone (its decoding chunks without a draft: every other chunk, the runs alternate).
        var drafts = stages.Where(e => e.Stage == InferenceStageKind.Draft).ToList();
        var verifies = stages.Where(e => e.Stage == InferenceStageKind.Verify).ToList();
        double draftMs = drafts.Sum(e => e.Duration.TotalMilliseconds) / Math.Max(1, drafts.Sum(e => e.Count));
        double verifyMs = verifies.Sum(e => e.Duration.TotalMilliseconds) / Math.Max(1, verifies.Count);
        double positions = verifies.Count == 0 ? 0 : verifies.Average(e => e.PieceSize);

        Check(first.Tokens!.Reply.SequenceEqual(alone.Tokens!.Reply), "the cold run: the same tokens with the draft");
        double Median(List<double> values) => values.Order().ElementAt(values.Count / 2);
        var c = CultureInfo.InvariantCulture;
        Console.WriteLine(string.Create(c, $"    speculative on {device}, other blocks x{scale}: target {targetModel.ParameterCount / 1e6:F2}M parameters, draft {draftModel.ParameterCount / 1e6:F2}M; "
            + $"alone {coldAlone:F0} tokens/s cold, {Median(warmAlone):F0} warm; with the draft {coldDraft:F0} cold, {Median(warmDraft):F0} warm "
            + $"(x{Median(warmDraft) / Median(warmAlone):F2}); {last.DraftAccepted} of {last.DraftTokens} proposals kept ({100 * last.DraftAcceptance:F0}%) in {last.DraftRounds} rounds; "
            + $"the draft {draftMs:F2} ms a proposal, the target's pass {verifyMs:F2} ms a round of {positions:F1} positions, the target alone {1000 / Median(warmAlone):F2} ms a token"));
    }
}
