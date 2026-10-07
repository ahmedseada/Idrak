// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Generation;
using Idrak.Gpu.Vulkan;
using Idrak.Layers;
using Idrak.Models.Abstractions;

// Recorded graphs on Vulkan (VulkanBackend.Graphs.cs): commands recorded once into command buffers and replayed by
// executing them again, so the host records nothing per decoding step.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] VulkanGraphGroup =
    [
        ("vulkan graphs: recorded commands replay in queue order (zeroed and freed scratch, uploads during recording, several command buffers; mapped, staging, descriptor sets)", VulkanGraphCommands),
        ("vulkan graphs: recorded decoding steps give the tokens and logits of direct steps (float32, int8, bfloat16 caches; sampler on device) as the cache fills, with no host fallback or recording per replay", VulkanGraphDecoding),
        ("vulkan graphs: a failed or aborted recording and a destroyed graph give their memory back", VulkanGraphMemory),
        ("vulkan graphs: the host's time per decoding step drops when steps are replayed", VulkanGraphHostTime),
        ("vulkan graphs: text generation replays its recorded step and gives the text of direct steps", VulkanGraphGeneration),
    ];

    private static bool VulkanKernelsOff => Environment.GetEnvironmentVariable("IDRAK_VULKAN_KERNELS") is "0" or "false";

    // Commands recorded on the backend directly: y = a + k (k uploaded during the recording into a storage it allocates),
    // then enough additions of 1 to need several command buffers, then acc += y, with y zeroed when allocated and freed
    // during the recording. Each replay adds to acc; work queued between replays, and host writes to a storage the graph
    // reads, keep their order. Reading device memory while recording fails. Through mapped memory and staging, and with
    // descriptor sets instead of pushed descriptors.
    private static void VulkanGraphCommands(Device device)
    {
        if (device.Type != DeviceType.Vulkan || VulkanKernelsOff)
        {
            return;                                                            // run on the Vulkan device itself
        }

        var backend = (VulkanBackend)device.Backend;
        VulkanGraphCommands(backend, $"{device} ({(backend.UnifiedMemory ? "mapped" : "staging")}, {(backend.PushDescriptors ? "pushed descriptors" : "descriptor sets")})");
        foreach (var (mapped, push, what) in new[] { (false, (bool?)null, "staging"), (true, false, "descriptor sets") })
        {
            var separate = VulkanBackend.CreateSeparate(device.Ordinal, preferMapped: mapped, pushDescriptors: push);
            try
            {
                VulkanGraphCommands(separate, $"{device} ({what})");
            }
            finally
            {
                separate.Shutdown();
            }
        }
    }

    private static void VulkanGraphCommands(VulkanBackend backend, string label)
    {
        const int N = 1000;
        int additions = backend.MaxBatchCommands * 2 + 5;                    // more than two batches of commands
        var a = backend.Allocate(N, zeroed: false);
        var acc = backend.Allocate(N, zeroed: true);
        var other = backend.Allocate(N, zeroed: true);
        var values = new float[N];
        try
        {
            backend.Upload([.. Enumerable.Range(0, N).Select(i => (float)(i % 7))], a);
            long inUse = backend.GetMemoryUsage().InUse;

            // Reading while recording fails, and the recording is abandoned.
            backend.BeginCapture();
            try
            {
                backend.Download(a, values);
                throw new Exception($"{label}: a read during a recording went through");
            }
            catch (InvalidOperationException)
            {
            }
            finally
            {
                foreach (var s in backend.AbortCapture())
                {
                    s.Release();
                }
            }

            Check(!backend.Capturing, $"{label}: the aborted recording ended");
            long dispatches = backend.Dispatches;
            backend.BeginCapture();
            List<Storage> owned;
            IntPtr executable;
            try
            {
                var y = backend.Allocate(N, zeroed: true);
                var k = backend.Allocate(N, zeroed: false);
                backend.Upload(Enumerable.Repeat(10f, N).ToArray(), k);         // at once, outside the graph
                backend.Binary(BinaryOp.Add, y, a, y, N);                       // y = 0 + a
                backend.Binary(BinaryOp.Add, y, k, y, N);                       // y = a + 10
                for (int i = 0; i < additions; i++)
                {
                    backend.Affine(y, y, N, 1f, 1f);
                }

                backend.Binary(BinaryOp.Add, acc, y, acc, N);
                y.Release();                                                   // the graph owns them now
                k.Release();
                (executable, _, owned) = backend.EndCapture();
            }
            catch
            {
                foreach (var s in backend.AbortCapture())
                {
                    s.Release();
                }

                throw;
            }

            try
            {
                var (buffers, commands, _) = backend.DescribeGraph(executable);
                Check(buffers >= 3, $"{label}: {commands} commands in {buffers} command buffers (batches of {backend.MaxBatchCommands})");
                Check(owned.Count == 2, $"{label}: the graph owns the {owned.Count} storages the recording freed, expected 2");
                Check(backend.LiveGraphs >= 1, $"{label}: the graph is counted");
                backend.Download(acc, values);
                Check(values.All(v => v == 0f), $"{label}: recording ran nothing");

                // acc += a + 10 + additions per replay; a changes from the host between replays.
                var expected = new float[N];
                long replayDispatches = backend.Dispatches;
                for (int r = 0; r < 5; r++)
                {
                    backend.ReplayGraph(executable);
                    backend.ReplayGraph(executable);                            // twice in one batch
                    for (int i = 0; i < N; i++)
                    {
                        expected[i] += 2 * (i % 7 + r + 10 + additions);
                    }

                    backend.Upload([.. Enumerable.Range(0, N).Select(i => (float)(i % 7 + r + 1))], a);   // waits for the replays
                }

                Check(backend.Dispatches == replayDispatches, $"{label}: replays recorded {backend.Dispatches - replayDispatches} dispatches on the host");
                backend.Affine(acc, other, N, 2f, 0f);                          // after the replays in queue order
                backend.Download(acc, values);
                AssertClose(expected, values, 0f, $"{label}: acc after 10 replays");
                backend.Download(other, values);
                AssertClose([.. expected.Select(v => 2 * v)], values, 0f, $"{label}: work queued after the replays");
            }
            finally
            {
                backend.Synchronize();
                backend.DestroyGraph(executable, IntPtr.Zero);
                owned.ForEach(s => s.Release());
            }

            Check(backend.GetMemoryUsage().InUse == inUse, $"{label}: in use {backend.GetMemoryUsage().InUse:N0} bytes after the graph is destroyed, {inUse:N0} before");
            Check(backend.Dispatches > dispatches, $"{label}: the recording counted its dispatches");
        }
        finally
        {
            a.Release();
            acc.Release();
            other.Release();
        }
    }

    // One decoding run of `model` over a cache of `format`: a prompt, a direct step, then the remaining steps direct or
    // replayed from a recording; the logits of each step and the sampled tokens. Halfway through, the pool grows (storages
    // of their own and new pages) and cached blocks are freed, which a recorded step must survive.
    private static (List<float[]> Logits, int[] Tokens, string Note) VulkanDecodingRun(Device device, Sequential model, int vocabulary,
        KeyValueFormat format, int capacity, bool useGraph)
    {
        var backend = (VulkanBackend)device.Backend;
        using var noGrad = Autograd.NoGrad();
        using var context = new DecodingContext(device, 1, capacity, format) { LastPositionOnly = true };
        using var sampler = new TokenSampler(device, 1, vocabulary, capacity)
        {
            Temperature = 0.8f, TopK = 5, TopP = 0.9f, MinP = 0.02f, RepeatPenalty = 1.1f, PresencePenalty = 0.1f, Seed = 7,
        };
        sampler.SetHistory([1, 2, 3]);
        using (new TensorScope())
        {
            sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f, 3f], [1, 3], device), context));
        }

        var logits = new List<float[]>();
        Tensor? last = null;
        void Step()
        {
            last = model.ForwardCached(sampler.Ids.Reshape(1, 1), context);
            sampler.Sample(last);
        }

        using (new TensorScope())
        {
            Step();                                                            // measures the step's shapes
            logits.Add(last!.ToArray());
        }

        int steps = capacity - context.Length;
        void GrowPool()
        {
            var grown = new List<Storage>();
            for (int i = 0; i < 40; i++)
            {
                grown.Add(backend.Allocate(10_000 + 37 * i, zeroed: true));
            }

            grown.Add(backend.Allocate((int)Math.Min(backend.SubAllocationMax / 2, 1 << 22), zeroed: true));
            grown.ForEach(s => s.Release());
            backend.ReleaseCachedMemory();
        }

        string note = "";
        if (useGraph)
        {
            using var graph = context.CaptureStep(Step);
            Check(graph.IsRecorded, $"{format} cache: the step was not recorded: {graph.FailureReason}");
            var output = last!;                                                // the graph's output, written by each replay
            long hostCalls = Idrak.Abstraction.Operations.Kernels.HostCalls(backend), dispatches = backend.Dispatches, replays = backend.Replays;
            for (int s = 0; s < steps; s++)
            {
                if (s == steps / 2)
                {
                    GrowPool();
                }

                context.ReplayStep(graph);
                logits.Add(output.ToArray());
            }

            Check(Idrak.Abstraction.Operations.Kernels.HostCalls(backend) == hostCalls, $"{format} cache: {Idrak.Abstraction.Operations.Kernels.HostCalls(backend) - hostCalls} host fallbacks during replays");
            Check(backend.Dispatches == dispatches, $"{format} cache: replays recorded {backend.Dispatches - dispatches} dispatches on the host");
            Check(backend.Replays - replays == steps, $"{format} cache: {backend.Replays - replays} replays, expected {steps}");
            note = $"{steps} replays";
        }
        else
        {
            for (int s = 0; s < steps; s++)
            {
                if (s == steps / 2)
                {
                    GrowPool();
                }

                using var scope = new TensorScope();
                Step();
                logits.Add(last!.ToArray());
            }
        }

        Check(context.Length == capacity, $"{format} cache: filled to {context.Length} of {capacity}");
        var tokens = sampler.Read(0, steps + 2).Select(t => t[0].Id).ToArray();
        return (logits, tokens, note);
    }

    private static void VulkanGraphDecoding(Device device)
    {
        if (device.Type != DeviceType.Vulkan || VulkanKernelsOff)
        {
            return;                                                            // run on the Vulkan device itself
        }

        // The same kernels run either way, but not always the same sequence of them: a direct step frees each layer's
        // tensors as it goes, so a block's residual addition and the next block's normalization are not kept fused (the
        // norm runs again on its own), which a recorded step keeps. The logits then differ in the last bits; the tokens
        // are the same.
        const int Capacity = 40;
        var spec = SmallSpec with { QkNorm = true, MaxPositions = Capacity };
        foreach (var (name, options) in new[]
        {
            ("float32 weights", new DecoderBuildOptions { Device = device, Seed = 2 }),
            ("int8 weights", new DecoderBuildOptions { Device = device, Seed = 2, Int8 = true }),
        })
        {
            using var model = spec.Build(null, options);
            model.Eval();
            foreach (var format in new[] { KeyValueFormat.Float32, KeyValueFormat.Int8, KeyValueFormat.BFloat16 })
            {
                var direct = VulkanDecodingRun(device, model, spec.Vocabulary, format, Capacity, useGraph: false);
                var replayed = VulkanDecodingRun(device, model, spec.Vocabulary, format, Capacity, useGraph: true);
                Check(direct.Tokens.SequenceEqual(replayed.Tokens),
                    $"{name}, {format} cache: direct tokens [{string.Join(",", direct.Tokens)}], replayed [{string.Join(",", replayed.Tokens)}]");
                Check(direct.Logits.Count == replayed.Logits.Count, $"{name}, {format} cache: {replayed.Logits.Count} steps, expected {direct.Logits.Count}");
                for (int s = 0; s < direct.Logits.Count; s++)
                {
                    AssertClose(direct.Logits[s], replayed.Logits[s], 1e-5f, $"{name}, {format} cache, step {s}: replayed logits");
                }
            }
        }
    }

    private static void VulkanGraphMemory(Device device)
    {
        if (device.Type != DeviceType.Vulkan || VulkanKernelsOff)
        {
            return;
        }

        var backend = (VulkanBackend)device.Backend;
        var spec = SmallSpec with { MaxPositions = 16 };
        using var model = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 4 });
        model.Eval();
        using var noGrad = Autograd.NoGrad();
        using var context = new DecodingContext(device, 1, 16) { LastPositionOnly = true };
        using var sampler = new TokenSampler(device, 1, spec.Vocabulary, 16) { Temperature = 0f };
        using (new TensorScope())
        {
            sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f], [1, 2], device), context));
        }

        void Step() => sampler.Sample(model.ForwardCached(sampler.Ids.Reshape(1, 1), context));
        using (new TensorScope())
        {
            Step();
        }

        backend.Synchronize();
        long inUse = backend.GetMemoryUsage().InUse;
        int graphs = backend.LiveGraphs;

        // A step that reads device memory cannot be recorded: the recording is abandoned and its storages released.
        using (var failed = context.CaptureStep(() =>
        {
            Step();
            _ = sampler.Ids.ToArray();
        }))
        {
            Check(!failed.IsRecorded && failed.FailureReason is { } reason && reason.Contains("graph is recorded", StringComparison.Ordinal),
                $"a step reading device memory was recorded ({failed.FailureReason})");
            Check(!backend.Capturing && backend.LiveGraphs == graphs, "the failed recording ended without a graph");
        }

        Check(backend.GetMemoryUsage().InUse == inUse, $"after a failed recording {backend.GetMemoryUsage().InUse:N0} bytes in use, {inUse:N0} before");

        // A recorded graph holds memory until it is destroyed.
        using (var graph = context.CaptureStep(Step))
        {
            Check(graph.IsRecorded, $"the step was not recorded: {graph.FailureReason}");
            Check(backend.LiveGraphs == graphs + 1, "the graph is counted while it exists");
            for (int i = 0; i < 4; i++)
            {
                context.ReplayStep(graph);
            }

            backend.ReleaseCachedMemory();                                     // does not free what the graph uses
            context.ReplayStep(graph);
            var tokens = sampler.Read(0, context.Length - 1);
            Check(tokens.All(t => t[0].Id >= 0 && t[0].Id < spec.Vocabulary), "replayed tokens in range");
        }

        backend.Synchronize();
        Check(backend.LiveGraphs == graphs, "the destroyed graph is no longer counted");
        Check(backend.GetMemoryUsage().InUse == inUse, $"after the graph is destroyed {backend.GetMemoryUsage().InUse:N0} bytes in use, {inUse:N0} before");
    }

    // The host's time to queue a decoding step: recorded each time (layer code, then every dispatch recorded) against
    // replayed (one command recorded). Each queued step is timed with the device idle before it, so waiting for the
    // device does not count; medians of a few steps.
    private static void VulkanGraphHostTime(Device device)
    {
        if (device.Type != DeviceType.Vulkan || VulkanKernelsOff)
        {
            return;
        }

        var backend = (VulkanBackend)device.Backend;
        const int Capacity = 64, Timed = 15;
        var spec = SmallSpec with { MaxPositions = Capacity };
        using var model = spec.Build(null, new DecoderBuildOptions { Device = device, Seed = 5 });
        model.Eval();
        using var noGrad = Autograd.NoGrad();
        using var context = new DecodingContext(device, 1, Capacity) { LastPositionOnly = true };
        using var sampler = new TokenSampler(device, 1, spec.Vocabulary, Capacity) { Temperature = 0.7f, TopK = 8, Seed = 3 };
        using (new TensorScope())
        {
            sampler.Sample(model.ForwardCached(Tensor.From([1f, 2f, 3f], [1, 3], device), context));
        }

        void Step() => sampler.Sample(model.ForwardCached(sampler.Ids.Reshape(1, 1), context));
        var recorded = new List<double>();
        long dispatches = backend.Dispatches;
        for (int s = 0; s < Timed + 2; s++)
        {
            backend.Synchronize();
            using var scope = new TensorScope();
            long start = Stopwatch.GetTimestamp();
            Step();
            if (s >= 2)                                                        // after warm-up
            {
                recorded.Add(Stopwatch.GetElapsedTime(start).TotalMicroseconds);
            }
        }

        double perStep = (backend.Dispatches - dispatches) / (double)(Timed + 2);
        var replayed = new List<double>();
        using (var graph = context.CaptureStep(Step))
        {
            Check(graph.IsRecorded, $"the step was not recorded: {graph.FailureReason}");
            for (int s = 0; s < Timed + 2; s++)
            {
                backend.Synchronize();
                long start = Stopwatch.GetTimestamp();
                context.ReplayStep(graph);
                if (s >= 2)
                {
                    replayed.Add(Stopwatch.GetElapsedTime(start).TotalMicroseconds);
                }
            }
        }

        double r = Median(recorded), p = Median(replayed);
        Console.WriteLine($"    host time per step: recorded {r:F0} µs ({perStep:F0} dispatches), replayed {p:F1} µs");
        // A replay costs one submission whatever the step's length, recording costs per command: on this small step (a
        // few dozen dispatches) a driver's submission may take a third of the recording, so half is the bound.
        Check(p < r / 2, $"replaying a step took {p:F1} µs of host time, recording it {r:F0} µs");

        static double Median(List<double> values)
        {
            var sorted = values.Order().ToArray();
            return sorted[sorted.Length / 2];
        }
    }

    // TextGenerator records its decoding step on Vulkan (from the second step) and replays it: the same text as without
    // the graph, and replays counted.
    private static void VulkanGraphGeneration(Device device)
    {
        if (device.Type != DeviceType.Vulkan || VulkanKernelsOff)
        {
            return;
        }

        var backend = (VulkanBackend)device.Backend;
        var (model, tokenizer) = TinyLanguageModel(device);
        using var _ = model;
        var generator = new TextGenerator(model, tokenizer, contextLength: 32) { KeepCache = false };
        var options = new GenerationOptions { Seed = 11, NumPredict = 20, Temperature = 0.9f, TopK = 6, RepeatPenalty = 1.1f, ChunkSize = 4 };
        long replays = backend.Replays;
        string replayed = generator.Generate("hello", options).Text;
        Check(backend.Replays - replays >= 15, $"text generation replayed {backend.Replays - replays} steps");
        string direct = generator.Generate("hello", options with { UseGraph = false }).Text;
        Check(replayed == direct, $"replayed '{replayed}', direct '{direct}'");
        Check(backend.LiveGraphs == 0 && !backend.Capturing, "the generator destroyed its graph");
    }
}
