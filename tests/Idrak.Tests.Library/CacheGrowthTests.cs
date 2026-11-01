// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models.Abstractions;

internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] CacheGrowthGroup =
    [
        ("kv cache growth: the tokens are the same with caches that grow as with caches of the whole window (rule 71): greedy and sampled, three models (multi-head attention, a decoder, a decoder with a sliding-window layer), float32, int8 and bfloat16 caches, a kept cache reused, a batch of rows of different lengths, speculative decoding; the caches grew", GrowthGivesTheSameTokens),
        ("kv cache growth: every layout grows in place: float32, int8 (with its scales) and bfloat16 caches keep each slot they held, the new slots zero", EveryLayoutGrows),
        ("kv cache growth: a recorded step is current until the caches grow; then it is recorded again (a replay of a stale one is refused naming the cause), a step re-run on a device without graphs grows them itself; GrowCaches false reserves the whole window and cannot change once caches are made", RecordedStepsFollowGrowth),
        ("kv cache growth: a short prompt on a 32768-position window of a 28-layer model holds its slots, not the window (before and after measured), a long reply grows the caches about log2 times, and each growth reaches telemetry", ShortPromptOnALongWindow),
    ];

    // Runs `run` with new contexts' caches growing (from `start` positions) or reserved whole.
    private static T WithGrowth<T>(bool grow, int start, Func<T> run)
    {
        var (was, wasStart) = (DecodingContext.DefaultGrowCaches, DecodingContext.DefaultGrowthStart);
        try
        {
            (DecodingContext.DefaultGrowCaches, DecodingContext.DefaultGrowthStart) = (grow, start);
            return run();
        }
        finally
        {
            (DecodingContext.DefaultGrowCaches, DecodingContext.DefaultGrowthStart) = (was, wasStart);
        }
    }

    internal static void GrowthGivesTheSameTokens(Device device)
    {
        var (tiny, tokenizer) = TinyLanguageModel(device, context: 96);
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 96,
            Rope = new RopeSettings(500f, null, false, null),
        };
        using var _1 = tiny;
        using var decoder = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 3 });
        using var windowed = (spec with { SlidingWindow = 12, SlidingWindowLayers = [true, false] }).Build(null, new DecoderBuildOptions { Device = device, Seed = 5 });
        using var draftModel = (spec with { Layers = 1 }).Build(null, new DecoderBuildOptions { Device = device, Seed = 11 });
        string[] prompts = ["the quick brown fox", "a", "the quick brown fox jumps over"];
        var greedy = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, Seed = 1, NumPredict = 60, NumCtx = 96, ChunkSize = 4, ReturnTokens = true };
        var sampled = greedy with { TopK = 10, Temperature = 0.8f, RepeatPenalty = 1.1f, Seed = 3 };
        var hook = new StageList();
        int growths = 0;

        // The replies of one generator setting over the prompts, with caches growing from 8 positions or reserved whole.
        List<(string Text, int[] Ids)> Replies(bool grow, Sequential model, KeyValueFormat format, bool keep, GenerationOptions options, TextGenerator? draft) =>
            WithGrowth(grow, 8, () =>
            {
                var generator = new TextGenerator(model, tokenizer, 96) { CacheFormat = format, KeepCache = keep };
                var replies = new List<(string, int[])>();
                foreach (string prompt in prompts)
                {
                    var chunks = generator.Stream(prompt, draft is null ? options : options with { Draft = draft }).ToList();
                    replies.Add((string.Concat(chunks.Select(c => c.Text)), chunks[^1].Tokens!.Reply));
                }

                return replies;
            });

        using (Telemetry.Subscribe(hook))
        {
            foreach (var (name, model) in new[] { ("multi-head attention", tiny), ("decoder", decoder), ("sliding-window decoder", windowed) })
            {
                foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
                {
                    foreach (bool keep in new[] { false, true })
                    {
                        foreach (var (sampling, options) in new[] { ("greedy", greedy), ("sampled", sampled) })
                        {
                            var whole = Replies(false, model, format, keep, options, null);
                            hook.Take();
                            var grown = Replies(true, model, format, keep, options, null);
                            growths += hook.Take().Count(e => e.Stage == InferenceStageKind.CacheGrowth);
                            for (int p = 0; p < prompts.Length; p++)
                            {
                                Check(grown[p].Ids.SequenceEqual(whole[p].Ids) && grown[p].Text == whole[p].Text,
                                    $"{name}, {format}, KeepCache {keep}, {sampling}, '{prompts[p]}': [{string.Join(",", grown[p].Ids)}] growing, [{string.Join(",", whole[p].Ids)}] reserved whole");
                            }
                        }
                    }
                }
            }

            var draft = new TextGenerator(draftModel, tokenizer, 96);
            var wholeDrafted = Replies(false, decoder, KeyValueFormat.Float32, true, greedy, draft);
            var grownDrafted = Replies(true, decoder, KeyValueFormat.Float32, true, greedy, draft);
            Check(wholeDrafted.Select(r => r.Ids).Zip(grownDrafted.Select(r => r.Ids)).All(p => p.First.SequenceEqual(p.Second)), "speculative decoding: the same tokens with growing caches (the target's and the draft's)");

            string[] rows = ["the quick brown fox", "a dog"];
            var batchWhole = WithGrowth(false, 8, () => new TextGenerator(decoder, tokenizer, 96) { KeepCache = false }.GenerateBatch(rows, greedy));
            var batchGrown = WithGrowth(true, 8, () => new TextGenerator(decoder, tokenizer, 96) { KeepCache = false }.GenerateBatch(rows, greedy));
            for (int r = 0; r < rows.Length; r++)
            {
                Check(batchGrown[r].Text == batchWhole[r].Text, $"batch row {r}: '{batchGrown[r].Text}' growing, '{batchWhole[r].Text}' reserved whole");
            }

            growths += hook.Take().Count(e => e.Stage == InferenceStageKind.CacheGrowth);
        }

        Check(growths > 0, $"the caches grew ({growths} growths reported)");
    }

    internal static void EveryLayoutGrows(Device device)
    {
        var random = new Random(9);
        foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
        {
            using var cache = new KeyValueCache(3, 5, 6, device, KeyValueLayouts.For(format));
            var tensors = new[] { cache.Keys, cache.Values, cache.KeyScales, cache.ValueScales }.OfType<Tensor>().ToList();
            var before = tensors.Select(t =>
            {
                float[] values = [.. Enumerable.Range(0, t.Size).Select(_ => (float)random.NextDouble())];
                t.Load(values);
                return (Shape: t.Shape.ToArray(), Values: values);
            }).ToList();
            cache.Grow(9);
            var after = new[] { cache.Keys, cache.Values, cache.KeyScales, cache.ValueScales }.OfType<Tensor>().ToList();
            Check(cache.Capacity == 9 && after.Count == before.Count, $"{format}: 9 slots, every tensor kept ({after.Count} of {before.Count})");
            for (int i = 0; i < after.Count; i++)
            {
                var (shape, values) = before[i];
                int rows = shape[0], inner = values.Length / (rows * 5);
                Check(after[i].Shape[0] == rows && after[i].Shape[1] == 9 && after[i].Size == rows * 9 * inner, $"{format} tensor {i}: [{string.Join(", ", after[i].Shape.ToArray())}]");
                float[] grown = after[i].ToArray();
                for (int r = 0; r < rows; r++)
                {
                    Check(grown.AsSpan(r * 9 * inner, 5 * inner).SequenceEqual(values.AsSpan(r * 5 * inner, 5 * inner)), $"{format} tensor {i}, row {r}: the slots it held");
                    Check(!grown.AsSpan(r * 9 * inner + 5 * inner, 4 * inner).ContainsAnyExcept(0f), $"{format} tensor {i}, row {r}: the new slots are zero");
                }
            }

            cache.Grow(7);
            Check(cache.Capacity == 9, $"{format}: a cache never shrinks");
        }
    }

    internal static void RecordedStepsFollowGrowth(Device device)
    {
        var (tiny, tokenizer) = TinyLanguageModel(device, context: 64);
        tiny.Dispose();
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 16, Layers = 2, Heads = 4, KvHeads = 2, HeadDim = 6, FfDim = 24, MaxPositions = 64,
            Rope = new RopeSettings(500f, null, false, null),
        };
        using var model = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 3 });
        model.Eval();
        float[] prompt = [3, 7, 1];
        float[] Logits(DecodingContext context, float[] ids)
        {
            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            using var input = Tensor.From(ids, [1, ids.Length], device);
            return model.ForwardCached(input, context).ToArray()[^tokenizer.VocabularySize..];
        }

        using var whole = new DecodingContext(device, 1, 64) { GrowCaches = false, LastPositionOnly = true };
        using var grown = new DecodingContext(device, 1, 64) { GrowCaches = true, GrowthStart = 4, LastPositionOnly = true };
        Check(Logits(whole, prompt).SequenceEqual(Logits(grown, prompt)), "the prompt's logits are the same");
        Check(whole.CacheSlots == 64 && grown.CacheSlots == 4 && grown.CacheGrowths == 0, $"slots: {whole.CacheSlots} reserved whole, {grown.CacheSlots} for 3 positions and a step");
        Check(grown.CacheBytes * 16 == whole.CacheBytes, $"the grown caches hold 4 of 64 slots: {grown.CacheBytes} and {whole.CacheBytes} bytes");

        using var token = Tensor.Persistent([5f], [1, 1], device, requiresGrad: false);
        var graph = grown.CaptureStep(() => model.ForwardCached(token, grown));
        Check(grown.IsCurrent(graph), "a step just recorded is current");
        grown.ReplayStep(graph);                                        // position 3: within the 4 slots
        Logits(whole, [5f]);
        Check(grown.Reserve(grown.Length + 1) && grown.CacheSlots == 8 && grown.CacheGrowths == 1, $"position 4 needs more: the caches double to {grown.CacheSlots}");
        if (graph.IsRecorded)
        {
            Check(!grown.IsCurrent(graph), "the step recorded before the growth reads the freed caches: not current");
            ThrowsNaming<InvalidOperationException>(() => grown.ReplayStep(graph), "replaying a stale step", "grown", "record the step again");
            graph.Dispose();
            graph = grown.CaptureStep(() => model.ForwardCached(token, grown));
        }
        else
        {
            Check(grown.IsCurrent(graph), "a step re-run at each replay (no device graph) is always current: it grows the caches itself");
        }

        for (int step = 0; step < 6; step++)
        {
            grown.Reserve(grown.Length + 1);
            if (!grown.IsCurrent(graph))
            {
                graph.Dispose();
                graph = grown.CaptureStep(() => model.ForwardCached(token, grown));
            }

            grown.ReplayStep(graph);
            Logits(whole, [5f]);
        }

        graph.Dispose();
        Check(grown.Length == whole.Length && grown.CacheSlots == 16 && grown.CacheGrowths == 2, $"10 positions in 16 slots after two growths ({grown.Length}, {grown.CacheSlots}, {grown.CacheGrowths})");
        Check(Logits(grown, [2f]).SequenceEqual(Logits(whole, [2f])), "the next logits are the same as with caches reserved whole");
        ThrowsNaming<InvalidOperationException>(() => grown.GrowCaches = false, "turning growth off once caches are made", "before the first step");
        using var fresh = new DecodingContext(device, 1, 64) { GrowCaches = false };
        Check(fresh.SlotsFor(3) == 64 && !fresh.Reserve(70) && fresh.CacheSlots == 64, "reserved whole: every slot from the first step");
    }

    internal static void ShortPromptOnALongWindow(Device device)
    {
        var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,<>|/_{}\":\n");
        const int Window = 32768;
        // A long window's cache with a 1.5B coder's key/value shape (28 layers, 2 key/value heads of 128), small weights.
        var spec = new DecoderSpec
        {
            Vocabulary = tokenizer.VocabularySize, Dim = 32, Layers = 28, Heads = 2, KvHeads = 2, HeadDim = 128, FfDim = 32, MaxPositions = Window,
            Rope = new RopeSettings(1_000_000f, null, false, null),
        };
        using var model = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 2 });
        model.Eval();
        long perPosition = 28L * 2 * 2 * 128 * sizeof(float);
        var ids = tokenizer.Encode("the quick brown fox jumps over the lazy dog").Select(i => (float)i).ToArray();
        long Bytes(bool grow, int steps, out int slots, out int growths)
        {
            using var context = new DecodingContext(device, 1, Window) { GrowCaches = grow, LastPositionOnly = true };
            using (Autograd.NoGrad())
            {
                using (var scope = new TensorScope())
                using (var input = Tensor.From(ids, [1, ids.Length], device))
                {
                    model.ForwardCached(input, context);
                }

                using var token = Tensor.From([4f], [1, 1], device);
                for (int s = 0; s < steps; s++)
                {
                    using var scope = new TensorScope();
                    model.ForwardCached(token, context);
                }
            }

            (slots, growths) = (context.CacheSlots, context.CacheGrowths);
            return context.CacheBytes;
        }

        long after = Bytes(grow: true, steps: 16, out int slotsAfter, out int growthsAfter);
        long before = Bytes(grow: false, steps: 16, out int slotsBefore, out _);
        Check(before == perPosition * Window && slotsBefore == Window, $"reserved whole: {before:N0} bytes for {slotsBefore} slots");
        Check(after == perPosition * 256 && slotsAfter == 256 && growthsAfter == 0, $"growing: {after:N0} bytes for {slotsAfter} slots, {growthsAfter} growths");

        var hook = new StageList();
        long longReply;
        int slotsLong, growthsLong;
        using (Telemetry.Subscribe(hook))
        {
            longReply = Bytes(grow: true, steps: 1000, out slotsLong, out growthsLong);
        }

        var stages = hook.Take();
        var growthEvents = stages.Where(e => e.Stage == InferenceStageKind.CacheGrowth).ToList();
        Check(slotsLong == 2048 && growthsLong == 3 && longReply == perPosition * 2048, $"a 1,043-position reply: {slotsLong} slots after {growthsLong} growths (256 → 512 → 1024 → 2048), {longReply:N0} bytes");
        Check(growthEvents.Count == 28 * growthsLong && growthEvents.All(e => e.Count is 512 or 1024 or 2048 && e.Bytes == perPosition / 28 * e.Count),
            $"each layer's growth reaches telemetry with its slots and bytes ({growthEvents.Count} events)");
        Check(stages.Where(e => e.Stage == InferenceStageKind.CacheAllocation).All(e => e.Count == 256), "each cache is made for the prompt and a step, rounded up to 256 slots");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    kv cache on {device}, window {Window}, 28 layers x 2 heads x 128: a {ids.Length}-token prompt and 16 tokens hold {before:N0} bytes reserved whole, "
            + $"{after:N0} growing ({slotsAfter} slots, x{(double)before / after:F0} less); a 1,000-token reply grows {growthsLong} times to {slotsLong} slots, {longReply:N0} bytes"));
    }
}
