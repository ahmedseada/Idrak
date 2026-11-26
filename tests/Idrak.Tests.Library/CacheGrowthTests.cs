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
