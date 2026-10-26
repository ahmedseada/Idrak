// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Gpu.Cuda;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Optimizers;

// Offloading to system memory (IMemoryOffload): the CPU optimizer step, moving tensors, staging layers' weights, cold
// data first, and bringing tensors back.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] OffloadingGroup =
    [
        ("offloading: HostOptimizer (AdamW, Adam, SGD with momentum, run on the CPU) gives the weights of the same optimizer on the device, with and without clipping", HostOptimizerMatches),
        ("offloading: a tensor moved to system memory and back keeps its values; nothing moves while a recorded graph exists", OffloadMoves),
        ("offloading: layers whose weights live in system memory give the same outputs and gradients, their weights staged on the GPU (forward and backward, the next layer prefetched) and back home after", OffloadStaging),
        ("offloading: a decoder whose base weights (float32 and int8) live in system memory gives the same outputs and LoRA gradients through its fused products (q/k/v, gate/up, down), staged as groups", OffloadDecoder),
        ("offloading: training with a full GPU moves optimizer state out first (then frozen weights), gives the weights of an ordinary run, and brings tensors back when memory frees up", OffloadColdFirst),
        ("offloading: before inference, a model that did not fit has its frozen weights marked cold before any pass and moved out, largest first, until the headroom is free; its outputs are unchanged", OffloadBeforeInference),
        ("memory chunks: long-lived blocks (Backend.LongLivedAllocations) the driver would round to its page share chunks: the driver takes about their bytes instead of a page more each; values and kernels unchanged; a freed block's room is reused; outside the scope a block takes the chunks' free room and makes none; with GpuChunkBytes 0 each block is its own; empty chunks go back to the driver", ChunkedBlocks),
        ("memory tuning: measurements on a device run one at a time (four threads measuring the same choice measure it once, with no event read across threads), and the attention path's scratch goes back to the driver, not to the pool", MeasuringAlone),
        ("memory stress: four threads take, check and free blocks of every kind the pool hands out (small, best-fit, chunk-sized in and out of the long-lived scope, a page and a little, whole pages) while others release the cache, record and replay graphs and measure the attention path; then again with the device at its limit and offloading on; then while another thread moves tensors they read and write between the device and system memory: no block handed out twice, every value and write kept, every graph recorded, no address read outside an operation or a move, no thread failing, and the pool's figures back where they started", MemoryStress),
        ("move gate: a move waits for other threads' operations and holds new ones back until it ends; one that may not wait, or is asked from inside an operation, runs only while no other thread is inside one; the mover's own operations and nested moves go through; a placement of lookup tables from inside an operation waits for the others, another inside one steps aside for it (counted again before it ends), and a general move still waiting gives way to it", MoveGateOrder),
        ("offloading: a lookup table placed while other threads keep reading it: an allocation at the limit outside any operation waits for their operations, moves the table and succeeds; their reads give the same sums before and after", LookupTablesUnderReaders),
        ("offloading: a frozen decoder's separate token table by LookupTables: host (system memory from the build), automatic (moved for good when an allocation finds the device at its limit, which then succeeds) and device (the allocation fails as before) give the same logits bit for bit; the table never counts as offloaded; auto, host and device parse", LookupTablePlacements),
    ];

    internal static Sequential OffloadModel(Device device, int seed) => new(
        new Linear(16, 96, device: device, random: new Random(seed)), new ReLU(),
        new Linear(96, 96, device: device, random: new Random(seed + 1)), new ReLU(),
        new Linear(96, 4, device: device, random: new Random(seed + 2)));

    // A few training steps (MSE on fixed data); returns every parameter's values.
    internal static float[][] TrainSteps(Sequential model, Optimizer optimizer, Device device, int steps, float clip, Action? afterStep = null)
    {
        var r = new Random(7);
        var xs = Enumerable.Range(0, 32 * 16).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        var ys = Enumerable.Range(0, 32 * 4).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        for (int step = 0; step < steps; step++)
        {
            using var scope = new TensorScope();
            optimizer.ZeroGrad();
            var x = Tensor.From(xs, [32, 16], device);
            var y = Tensor.From(ys, [32, 4], device);
            Losses.MeanSquaredError(model.Forward(x), y).Backward();
            if (clip > 0f)
            {
                optimizer.ClipAndStep(clip);
            }
            else
            {
                optimizer.Step();
            }

            afterStep?.Invoke();
        }

        return [.. model.Parameters().Select(p => p.ToArray())];
    }

    internal static void HostOptimizerMatches(Device device)
    {
        var makers = new (string Name, Func<IReadOnlyList<Tensor>, Optimizer> Create)[]
        {
            ("AdamW", ps => new AdamW(ps, 0.01f, weightDecay: 0.1f)),
            ("Adam", ps => new Adam(ps, 0.01f)),
            ("SGD", ps => new Sgd(ps, 0.05f, momentum: 0.9f)),
        };
        foreach (var (name, create) in makers)
        {
            foreach (float clip in new[] { 0f, 0.05f })
            {
                using var a = OffloadModel(device, 3);
                using var b = OffloadModel(device, 3);
                using var direct = create([.. a.Parameters()]);
                using var host = new HostOptimizer(b.Parameters(), create);
                var expected = TrainSteps(a, direct, device, 4, clip);
                var actual = TrainSteps(b, host, device, 4, clip);
                for (int i = 0; i < expected.Length; i++)
                {
                    AssertClose(expected[i], actual[i], 2e-4f, $"{name} (clip {clip}) parameter {i}");
                }

                Check(host.CpuOptimizers.All(o => o.Parameters.All(p => p.Device == Device.Cpu)), $"{name}: the inner optimizers run on the CPU");
            }
        }
    }

    internal static IMemoryOffload? OffloadOf(Device device) => device.Backend.Offload;

    internal static void OffloadMoves(Device device)
    {
        if (OffloadOf(device) is not { } offload)
        {
            Check(device.Type != DeviceType.Cuda, "every CUDA backend offloads");
            return;                                                              // the CPU's memory is system memory already
        }

        var values = Enumerable.Range(0, 300_001).Select(i => (float)Math.Sin(i)).ToArray();
        using var t = Tensor.From(values, [values.Length], device);
        long before = ComputeResources.GetMemoryUsage(device).Offloaded;
        Check(offload.MoveToHost(t.Storage, keep: true), "moved to system memory");
        Check(offload.IsOffloaded(t.Storage), "offloaded");
        Check(ComputeResources.GetMemoryUsage(device).Offloaded - before >= values.Length * 4L, "counted as offloaded");
        AssertClose(values, t.ToArray(), 0f, "values in system memory");
        using (var doubled = t * 2f)
        {
            AssertClose([.. values.Select(v => v * 2f)], doubled.ToArray(), 1e-6f, "kernels read system memory");
        }

        offload.Rebalance(makeRoom: false);
        Check(offload.IsOffloaded(t.Storage), "kept in system memory (keep = true)");
        Check(offload.MoveToDevice(t.Storage), "moved back");
        Check(!offload.IsOffloaded(t.Storage), "on the device");
        AssertClose(values, t.ToArray(), 0f, "values back on the device");

        // A recorded graph holds raw addresses: nothing moves until it is disposed.
        using var input = Tensor.From(values, [values.Length], device);
        using var output = Tensor.Zeros([values.Length], device);
        var graph = ComputeGraph.Capture(device, () => input.Backend.Affine(input.Storage, output.Storage, values.Length, 3f, 0f));
        Check(!offload.MoveToHost(input.Storage, keep: false), "no move while a graph exists");
        graph.Dispose();
        Check(offload.MoveToHost(input.Storage, keep: false), "moves once the graph is gone");
        Check(offload.MoveToDevice(input.Storage), "and back");
    }

    internal static void OffloadStaging(Device device)
    {
        if (OffloadOf(device) is not { } offload || device.Backend is not CudaBackend cuda)
        {
            return;
        }

        var r = new Random(11);
        var xs = Enumerable.Range(0, 8 * 16).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        (float[] Output, float[][] Grads, float[] InputGrad) Run(bool offloaded)
        {
            using var model = OffloadModel(device, 5);
            if (offloaded)
            {
                foreach (var p in model.Parameters())
                {
                    Check(offload.MoveToHost(p.Storage, keep: true), "weight moved");
                }
            }

            float[] output = [];
            float[] inputGrad = [];
            for (int pass = 0; pass < 2; pass++)                                // the second pass knows the layer order
            {
                using var scope = new TensorScope();
                foreach (var p in model.Parameters())
                {
                    p.ZeroGrad();
                }

                var x = Tensor.From(xs, [8, 16], device);
                x.RequiresGrad = true;
                var y = model.Forward(x);
                y.Square().Sum().Backward();
                (output, inputGrad) = (y.ToArray(), x.Grad!.ToArray());
            }

            if (offloaded)
            {
                Check(model.Parameters().All(p => offload.IsOffloaded(p.Storage)), "weights back in system memory after the passes");
            }

            return (output, [.. model.Parameters().Select(p => p.Grad!.ToArray())], inputGrad);
        }

        var expected = Run(offloaded: false);
        var (staged, prefetched) = (cuda.StagedStorages, cuda.PrefetchedStorages);
        var actual = Run(offloaded: true);
        AssertClose(expected.Output, actual.Output, 1e-5f, "outputs");
        AssertClose(expected.InputGrad, actual.InputGrad, 1e-5f, "input gradient");
        for (int i = 0; i < expected.Grads.Length; i++)
        {
            AssertClose(expected.Grads[i], actual.Grads[i], 1e-4f, $"gradient {i}");
        }

        // Forward and backward stage every layer on each pass; the second pass prefetches the next layer.
        Check(cuda.StagedStorages - staged >= 2 * 2 * 6, $"staged {cuda.StagedStorages - staged}");
        Check(cuda.PrefetchedStorages - prefetched > 0, $"prefetched {cuda.PrefetchedStorages - prefetched}");
    }

    internal static void OffloadDecoder(Device device)
    {
        if (OffloadOf(device) is not { } offload || device.Backend is not CudaBackend cuda)
        {
            return;
        }

        var spec = SmallSpec;
        var r = new Random(31);
        var inputValues = Enumerable.Range(0, 2 * 7 * spec.Dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        var weightValues = Enumerable.Range(0, 2 * 7 * spec.Dim).Select(_ => (float)(r.NextDouble() * 2 - 1)).ToArray();
        foreach (bool int8 in new[] { false, true })
        {
            (float[] Inference, float[] Adapters, long Staged) Run(bool offloaded)
            {
                using var model = spec.Build(new RandomWeights(33), new DecoderBuildOptions { Device = device });
                if (int8)
                {
                    model.QuantizeInt8();
                }

                model.AddLora(rank: 2, alpha: 4, targets: l => l.Name is "q" or "k" or "v" or "o" or "gate" or "up" or "down", freezeBase: true,
                    random: new Random(34));
                foreach (var adapter in model.Descendants().OfType<Linear>().Select(l => l.Adapter).OfType<LoraAdapter>())
                {
                    adapter.B.Load([.. Enumerable.Range(0, adapter.B.Size).Select(i => MathF.Sin(i))]);
                }

                if (offloaded)
                {
                    foreach (var l in model.Descendants().OfType<Linear>())
                    {
                        foreach (var t in l.Parameters().Where(p => !p.RequiresGrad).Concat(l.Buffers()))
                        {
                            Check(offload.MoveToHost(t.Storage, keep: true), "base weight moved");
                        }
                    }
                }

                long staged = cuda.StagedStorages;
                float[] inference;
                using (Autograd.NoGrad())
                using (var scope = new TensorScope())
                {
                    model.Eval();
                    var hidden = Tensor.From(inputValues.AsSpan(0, 7 * spec.Dim), [1, 7, spec.Dim], device);
                    foreach (var block in model.OfType<DecoderBlock>())
                    {
                        hidden = block.Forward(hidden);
                    }

                    inference = hidden.ToArray();
                }

                model.Train();
                using (var scope = new TensorScope())
                {
                    var hidden = Tensor.From(inputValues, [2, 7, spec.Dim], device);
                    foreach (var block in model.OfType<DecoderBlock>())
                    {
                        hidden = block.Forward(hidden);
                    }

                    (hidden * Tensor.From(weightValues, [2, 7, spec.Dim], device)).Sum().Backward();
                }

                var adapters = model.TrainableParameters().SelectMany(p => p.Grad!.ToArray()).ToArray();
                return (inference, adapters, cuda.StagedStorages - staged);
            }

            var expected = Run(offloaded: false);
            var actual = Run(offloaded: true);
            string name = int8 ? "int8 base" : "float32 base";
            AssertClose(expected.Inference, actual.Inference, 1e-4f, $"{name}: inference outputs");
            AssertClose(expected.Adapters, actual.Adapters, 1e-3f, $"{name}: adapter gradients");
            Check(actual.Staged > 0, $"{name}: weights staged ({actual.Staged})");
        }
    }

    internal static void OffloadColdFirst(Device device)
    {
        if (OffloadOf(device) is not { } offload || device.Backend is not CudaBackend cuda)
        {
            return;
        }

        float[][] Ordinary()
        {
            using var model = OffloadModel(device, 9);
            using var optimizer = new AdamW(model.Parameters(), 0.01f);
            return TrainSteps(model, optimizer, device, 5, 0f);
        }

        var expected = Ordinary();
        long? limit = ComputeResources.GpuMemoryLimit;
        bool offloading = ComputeResources.OffloadToHostMemory;
        var headroom = ComputeResources.OffloadReturnHeadroom;
        ComputeResources.ReleaseCachedMemory(device);
        try
        {
            ComputeResources.OffloadToHostMemory = true;
            using var model = OffloadModel(device, 9);
            model.Parameters().Skip(4).ToList().ForEach(p => p.RequiresGrad = false);          // the last layer frozen
            var trained = model.Parameters().Where(p => p.RequiresGrad).ToList();
            using var optimizer = new AdamW(trained, 0.01f);
            long movedOut = cuda.MovedToHost;

            // The weights fit and a little more: the optimizer state and the activations do not.
            ComputeResources.GpuMemoryLimit = ComputeResources.GetMemoryUsage(device).InUse + (64L << 10);
            var actual = TrainSteps(model, optimizer, device, 5, 0f);
            Check(cuda.MovedToHost > movedOut, $"cold data moved out ({cuda.MovedToHost - movedOut})");

            // Frozen layer: trained weights equal the ordinary run's only for the trained layers, so compare those, and
            // check the frozen ones did not change.
            using var reference = OffloadModel(device, 9);
            var frozen = reference.Parameters().Skip(4).Select(p => p.ToArray()).ToList();
            AssertClose(frozen[0], actual[4], 0f, "frozen weight unchanged");
            AssertClose(frozen[1], actual[5], 0f, "frozen bias unchanged");
            Check(offload.OffloadedCount > 0, "something lives in system memory under the limit");

            // Memory frees up: tensors come back.
            ComputeResources.GpuMemoryLimit = null;
            ComputeResources.OffloadReturnHeadroom = 0;
            int back = ComputeResources.ReturnOffloaded(device);
            Check(back > 0, $"tensors brought back ({back})");
            Check(model.Parameters().All(p => !offload.IsOffloaded(p.Storage)), "weights back on the device");
            var after = model.Parameters().Select(p => p.ToArray()).ToList();
            for (int i = 0; i < after.Count; i++)
            {
                AssertClose(actual[i], after[i], 0f, $"parameter {i} kept its values through the moves");
            }
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = limit;
            ComputeResources.OffloadToHostMemory = offloading;
            ComputeResources.OffloadReturnHeadroom = headroom;
            ComputeResources.ReleaseCachedMemory(device);
        }

        // The same steps with every layer trained, under a limit, match the ordinary run.
        try
        {
            ComputeResources.OffloadToHostMemory = true;
            using var model = OffloadModel(device, 9);
            using var optimizer = new AdamW(model.Parameters(), 0.01f);
            ComputeResources.GpuMemoryLimit = ComputeResources.GetMemoryUsage(device).InUse + (64L << 10);
            var actual = TrainSteps(model, optimizer, device, 5, 0f);
            for (int i = 0; i < expected.Length; i++)
            {
                AssertClose(expected[i], actual[i], 1e-4f, $"parameter {i} trained under the limit");
            }
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = limit;
            ComputeResources.OffloadToHostMemory = offloading;
            ComputeResources.ReleaseCachedMemory(device);
        }
    }

    internal static void OffloadBeforeInference(Device device)
    {
        if (OffloadOf(device) is not { } offload || device.Backend is not CudaBackend cuda)
        {
            return;
        }

        using var input = Tensor.From(Enumerable.Range(0, 8 * 16).Select(i => MathF.Sin(i * 0.37f)).ToArray(), [8, 16], device);
        float[] expected;
        using (var ordinary = OffloadModel(device, 5))
        using (Autograd.NoGrad())
        {
            expected = ordinary.Forward(input).ToArray();
        }

        long? limit = ComputeResources.GpuMemoryLimit;
        bool offloading = ComputeResources.OffloadToHostMemory;
        var headroom = ComputeResources.OffloadReturnHeadroom;
        ComputeResources.ReleaseCachedMemory(device);
        try
        {
            ComputeResources.OffloadToHostMemory = true;
            using var model = OffloadModel(device, 5);
            model.Parameters().ToList().ForEach(p => p.RequiresGrad = false);
            Check(offload.MoveToHost(model.Parameters().Last().Storage, keep: false), "a weight in system memory: the model did not fit");

            // A full GPU (16 KiB of room under the limit) and 32 KiB of headroom asked for: the largest frozen weight
            // (96 x 96) leaves, never having run.
            ComputeResources.GpuMemoryLimit = ComputeResources.GetMemoryUsage(device).InUse + (16L << 10);
            ComputeResources.OffloadReturnHeadroom = 32L << 10;
            long movedOut = cuda.MovedToHost;
            TensorOffloading.BeforeInference(device, model);
            var usage = ComputeResources.GetMemoryUsage(device);
            Check(cuda.MovedToHost > movedOut && offload.IsOffloaded(model.Parameters().ElementAt(2).Storage),
                $"the largest frozen weight moved out ({cuda.MovedToHost - movedOut} moved)");
            Check(usage.Limit - usage.InUse >= 32L << 10, $"{usage.Limit - usage.InUse} bytes free under the limit, 32 KiB asked");
            using (Autograd.NoGrad())
            {
                AssertClose(expected, model.Forward(input).ToArray(), 1e-5f, "outputs with the weights in system memory");
            }

            // Room to spare: nothing more moves.
            ComputeResources.OffloadReturnHeadroom = 0;
            long before = cuda.MovedToHost;
            Check(offload.KeepFree() == 0 && cuda.MovedToHost == before, "no move when the headroom is free");
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = limit;
            ComputeResources.OffloadToHostMemory = offloading;
            ComputeResources.OffloadReturnHeadroom = headroom;
            ComputeResources.ReleaseCachedMemory(device);
        }
    }

    // Training-step times with the weights on the GPU, in system memory (read directly, staged, staged with the next layer
    // prefetched), and with AdamW's update on the CPU.
    internal static int BenchOffload()
    {
        if (!Device.IsCudaAvailable)
        {
            Console.WriteLine("needs a CUDA device");
            return 1;
        }

        var device = Device.Cuda();
        var offload = device.Backend.Offload!;
        Console.WriteLine($"{device.Name}: 8 linear layers 2048 x 2048 (128 MiB of float32 weights), 256 rows, AdamW");
        var r = new Random(1);
        var xs = Enumerable.Range(0, 256 * 2048).Select(_ => r.NextSingle() - 0.5f).ToArray();
        double Time(string name, bool offloaded, bool stage, bool prefetch, bool hostOptimizer)
        {
            using var model = new Sequential(Enumerable.Range(0, 8).Select(i => (Module)new Linear(2048, 2048, device: device, random: new Random(i))));
            using Optimizer optimizer = hostOptimizer ? new HostOptimizer(model.Parameters(), ps => new AdamW(ps, 1e-4f)) : new AdamW(model.Parameters(), 1e-4f);
            if (offloaded)
            {
                model.Parameters().ToList().ForEach(p => offload.MoveToHost(p.Storage, keep: true));
            }

            (Offloading.StageWeights, ComputeResources.PrefetchOffloadedWeights) = (stage, prefetch);
            using var x = Tensor.From(xs, [256, 2048], device);
            void Step()
            {
                using var scope = new TensorScope();
                optimizer.ZeroGrad();
                model.Forward(x).Square().Mean().Backward();
                optimizer.ClipAndStep(1f);
            }

            try
            {
                for (int i = 0; i < 3; i++)
                {
                    Step();
                }

                device.Synchronize();
                var watch = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < 10; i++)
                {
                    Step();
                }

                device.Synchronize();
                double ms = watch.Elapsed.TotalMilliseconds / 10;
                Console.WriteLine($"  {name,-58} {ms,8:F1} ms/step");
                return ms;
            }
            finally
            {
                (Offloading.StageWeights, ComputeResources.PrefetchOffloadedWeights) = (true, true);
            }
        }

        double gpu = Time("weights and AdamW state on the GPU", false, true, true, false);
        Time("weights in system memory, read over PCIe as kernels run", true, false, false, false);
        Time("weights in system memory, staged per layer", true, true, false, false);
        double prefetched = Time("weights in system memory, staged + next layer prefetched", true, true, true, false);
        double host = Time("weights on the GPU, AdamW on the CPU (state in RAM)", false, true, true, true);
        Console.WriteLine($"  prefetch vs GPU-only: {prefetched / gpu:F2}x the time; CPU optimizer: {host / gpu:F2}x");
        return 0;
    }

    internal static void LookupTablePlacements(Device device)
    {
        Check(ComputeResources.ParseLookupTables(null) == LookupTablePlacement.Automatic && ComputeResources.ParseLookupTables("auto") == LookupTablePlacement.Automatic
            && ComputeResources.ParseLookupTables("Host") == LookupTablePlacement.HostMemory && ComputeResources.ParseLookupTables(" device ") == LookupTablePlacement.DeviceMemory,
            "auto, host and device parse");
        Expect<ArgumentException>(() => ComputeResources.ParseLookupTables("disk"), "auto, host or device", "an unknown placement");

        // int8 projections and a tied int8 head beside a bfloat16 table of their own (as int8 weights load a checkpoint).
        var spec = SmallSpec with { Vocabulary = 4096, TieEmbeddings = true };
        using var sequence = Tensor.From([.. Enumerable.Range(0, 7).Select(i => (float)(i * 577 % spec.Vocabulary))], [1, 7], device);
        Sequential Build(LookupTablePlacement placement) =>
            spec.Build(new RandomWeights(41), new DecoderBuildOptions { Device = device, Int8 = true, LookupTables = placement });
        float[] Logits(Sequential model)
        {
            using (Autograd.NoGrad())
            {
                model.Eval();
                return model.Predict(sequence).ToArray();
            }
        }

        static Storage TableOf(Sequential model) => model.Descendants().OfType<Embedding>().Single().BFloat16!.Packed.Storage;

        using var reference = Build(LookupTablePlacement.DeviceMemory);
        var expected = Logits(reference);
        if (OffloadOf(device) is not { } offload || device.Backend is not CudaBackend cuda)
        {
            using var host = Build(LookupTablePlacement.HostMemory);                 // the table is in system memory already
            AssertClose(expected, Logits(host), 0f, "a device without offloading: the same logits");
            return;
        }

        long tableBytes = TableOf(reference).Length * 4L;
        long placed = cuda.PlacedOnHost, offloaded = ComputeResources.GetMemoryUsage(device).Offloaded;
        using (var host = Build(LookupTablePlacement.HostMemory))
        {
            Check(cuda.PlacedOnHost == placed + 1, "host: the table in system memory from the build");
            Check(ComputeResources.GetMemoryUsage(device).Offloaded - offloaded >= tableBytes, $"host: counted in system memory ({ComputeResources.GetMemoryUsage(device)})");
            Check(offload.OffloadedCount == 0, "host: not offloaded (nothing staged, rebalanced or brought back)");
            AssertClose(expected, Logits(host), 0f, "host: the same logits");
        }

        Check(ComputeResources.GetMemoryUsage(device).Offloaded == offloaded, "host: the system block released with the model");
        long? limit = ComputeResources.GpuMemoryLimit;
        try
        {
            // The device at its limit: an allocation the size of the table fits only once the table has left.
            void AtLimit() => ComputeResources.GpuMemoryLimit = ComputeResources.GetMemoryUsage(device).InUse + tableBytes / 2;
            AtLimit();
            Expect<ResourceLimitExceededException>(() => Tensor.Zeros([(int)(tableBytes / 4)], device).Dispose(), "limit", "device: the table stays, the allocation fails");
            Check(cuda.PlacedOnHost == placed + 1, "device: nothing placed");
            ComputeResources.GpuMemoryLimit = limit;

            using var automatic = Build(LookupTablePlacement.Automatic);
            Check(cuda.PlacedOnHost == placed + 1, "automatic: on the device while it has room");
            AssertClose(expected, Logits(automatic), 0f, "automatic, on the device: the same logits");
            AtLimit();
            using (Tensor.Zeros([(int)(tableBytes / 4)], device))
            {
                Check(cuda.PlacedOnHost == placed + 2, "automatic: the full device moved the table out and the allocation succeeded");
            }

            ComputeResources.GpuMemoryLimit = limit;
            Check(offload.OffloadedCount == 0, "automatic: not offloaded");
            AssertClose(expected, Logits(automatic), 0f, "automatic, in system memory: the same logits");
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = limit;
        }
    }

    internal static void ChunkedBlocks(Device device)
    {
        if (device.Backend is not CudaBackend cuda)
        {
            Check(device.Backend.LongLivedAllocations() is null, "a device that places every block alike has no long-lived scope");
            return;
        }

        long chunk = cuda.ChunkSize;
        if (chunk <= 0)
        {
            Console.WriteLine("    the driver reports no allocation page: no chunks to check");
            return;
        }

        // Blocks of half a chunk's page count and a little more: alone, each takes the driver's next whole page.
        int floats = (int)((chunk / 32 + 4096) / 4), count = 24;
        var values = Enumerable.Range(0, floats).Select(i => MathF.Sin(i)).ToArray();
        ComputeResources.ReleaseCachedMemory(device);
        int chunksBefore = cuda.ChunkCount;
        long blocksBefore = cuda.ChunkedBlocks, free = cuda.FreeMemory()!.Value;
        var blocks = new List<Tensor>();
        try
        {
            using (device.Backend.LongLivedAllocations())
            {
                for (int i = 0; i < count; i++)
                {
                    blocks.Add(Tensor.From(values, [floats], device));
                }
            }

            long taken = free - cuda.FreeMemory()!.Value, asked = 4L * floats * count;
            Check(cuda.ChunkedBlocks - blocksBefore == count, $"{cuda.ChunkedBlocks - blocksBefore} of {count} blocks in chunks");
            Check(taken <= asked + chunk, $"the driver took {taken:N0} bytes for {asked:N0} (alone each would take {count * (chunk / 16):N0})");
            using (var doubled = blocks[^1] * 2f)
            {
                AssertClose([.. values.Select(v => v * 2f)], doubled.ToArray(), 0f, "a chunk's block read by a kernel");
            }

            AssertClose(values, blocks[0].ToArray(), 0f, "a chunk's block keeps its values");

            // A freed block's room is taken again by the next block of the scope.
            long made = cuda.ChunksAllocated;
            blocks[3].Dispose();
            blocks.RemoveAt(3);
            ComputeResources.ReleaseCachedMemory(device);
            using (device.Backend.LongLivedAllocations())
            {
                blocks.Add(Tensor.From(values, [floats], device));
            }

            Check(cuda.ChunksAllocated == made, "the freed block's room reused, no chunk added");

            // Outside the scope a block takes the chunks' free room (they are held anyway), and makes no chunk of its own.
            long chunked = cuda.ChunkedBlocks;
            using (var outside = Tensor.From(values, [floats], device))
            {
                Check(cuda.ChunkedBlocks == chunked + 1 && cuda.ChunksAllocated == made, "outside the scope: the chunks' free room, no new chunk");
                AssertClose(values, outside.ToArray(), 0f, "a block in a chunk's free room keeps its values");
            }

            ComputeResources.ReleaseCachedMemory(device);
            chunked = cuda.ChunkedBlocks;

            long? setting = ComputeResources.GpuChunkBytes;
            ComputeResources.GpuChunkBytes = 0;
            try
            {
                using (device.Backend.LongLivedAllocations())
                using (Tensor.From(values, [floats], device))
                {
                    Check(cuda.ChunkedBlocks == chunked, "GpuChunkBytes 0: a block of its own");
                }
            }
            finally
            {
                ComputeResources.GpuChunkBytes = setting;
            }
        }
        finally
        {
            blocks.ForEach(b => b.Dispose());
            ComputeResources.ReleaseCachedMemory(device);
        }

        Check(cuda.ChunkCount == chunksBefore, $"empty chunks back to the driver: {cuda.ChunkCount} held, {chunksBefore} before");
    }

    internal static void MeasuringAlone(Device device)
    {
        if (device.Backend is not CudaBackend cuda || !CudaBackend.Autotune)
        {
            return;                                                              // nothing is measured
        }

        // The attention path for a shape of its own: the scores and weights it measures with leave the cache as they came.
        const int heads = 2, rows = 512, dim = 64;
        var r = new Random(5);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
        using var q = Tensor.From(Values(heads * rows * dim), [heads, rows, dim], device);
        using var k = Tensor.From(Values(heads * rows * dim), [heads, rows, dim], device);
        using var v = Tensor.From(Values(heads * rows * dim), [heads, rows, dim], device);
        using var starts = Tensor.From(new float[heads * rows], [heads * rows], device);
        using var ends = Tensor.From(Enumerable.Repeat((float)rows, heads * rows).ToArray(), [heads * rows], device);
        cuda.PrefersComposedAttention(q.Storage, k.Storage, v.Storage, starts.Storage, ends.Storage, null, heads, rows, rows, dim, 0.125f);
        cuda.ForgetMeasurements(TuneOp.AttentionPath);                          // the choices its candidates need stay settled
        ComputeResources.ReleaseCachedMemory(device);
        long cached = ComputeResources.GetMemoryUsage(device).Cached, measured = cuda.MeasuredCount;
        cuda.PrefersComposedAttention(q.Storage, k.Storage, v.Storage, starts.Storage, ends.Storage, null, heads, rows, rows, dim, 0.125f);
        long kept = ComputeResources.GetMemoryUsage(device).Cached - cached, scores = 4L * heads * rows * rows;
        Check(cuda.MeasuredCount == measured + 1, $"the attention path measured: {cuda.MeasuredCount - measured} measurements");
        Check(kept < scores, $"the measurement's scratch back to the driver: the cache grew by {kept:N0} bytes (the scores alone are {scores:N0})");

        // Four threads asking for the same unmeasured choice at once: one measures, the others take its choice.
        cuda.ForgetMeasurements(TuneOp.AttentionPath);
        measured = cuda.MeasuredCount;
        var errors = new System.Collections.Concurrent.ConcurrentBag<string>();
        var choices = new System.Collections.Concurrent.ConcurrentBag<bool>();
        Parallel.For(0, 4, new ParallelOptions { MaxDegreeOfParallelism = 4 }, _ =>
        {
            try
            {
                choices.Add(cuda.PrefersComposedAttention(q.Storage, k.Storage, v.Storage, starts.Storage, ends.Storage, null, heads, rows, rows, dim, 0.125f));
            }
            catch (Exception e)
            {
                errors.Add($"{e.GetType().Name}: {e.Message}");
            }
        });
        Check(errors.IsEmpty, $"measuring from four threads: {string.Join("; ", errors)}");
        Check(cuda.MeasuredCount - measured == 1 && choices.Distinct().Count() == 1, $"{cuda.MeasuredCount - measured} measurements, choices {string.Join(", ", choices)}");
    }

    // The allocator's policies at once, from several threads on one device: the pool by size and best fit, chunks inside and
    // outside the long-lived scope, the cache released by another thread, blocks freed while a graph is recorded, the
    // attention path's scratch freed alone, and system blocks once the device is at its limit.
    internal static void MemoryStress(Device device)
    {
        var cuda = device.Backend as CudaBackend;
        long page = cuda is { Page: > 0 } ? cuda.Page : 2L << 20;               // sizes around the driver's page

        // A block's values: whole numbers below 251, exact in float32 and doubled exactly.
        static float[] Pattern(int length, int seed)
        {
            var values = new float[length];
            for (int i = 0; i < length; i++)
            {
                values[i] = (seed + i) % 251;
            }

            return values;
        }

        int Size(Random r) => r.Next(5) switch
        {
            0 => r.Next(1, 1024),                                              // below best fit: exact sizes only
            1 => r.Next(1024, 64 << 10),                                       // best fit: up to a quarter larger
            2 => (int)((page / 2 + r.Next(1, (int)(page / 8))) / 4),           // half a page and more: chunk-sized
            3 => (int)((page + 4096) / 4),                                     // a page and a little: rounded to two alone
            _ => (int)(page / 4),                                              // whole pages: never in a chunk
        };

        // One warm round, so what the device makes once (kernels, the tuning events, its zero position) is not counted.
        using (var warm = Tensor.From(Pattern(4096, 1), [4096], device))
        using (var doubled = warm * 2f)
        {
            _ = doubled.ToArray();
        }

        ComputeResources.ReleaseCachedMemory(device);
        var start = ComputeResources.GetMemoryUsage(device);
        int chunksBefore = cuda?.ChunkCount ?? 0;

        bool checking = CudaBackend.CheckAddressReads;
        CudaBackend.CheckAddressReads = cuda is not null;                    // every address read inside an operation or a move
        try
        {
            Run("the device as it is", offloading: false, moving: false);
            if (cuda is not null && OffloadOf(device) is not null)
            {
                RunAtLimit();
                if (cuda.LiveGraphs > 0)
                {
                    GC.Collect();                                              // graphs of earlier tests' generators
                    GC.WaitForPendingFinalizers();
                }

                if (cuda.LiveGraphs == 0)
                {
                    Run("tensors moved while read and written", offloading: false, moving: true);
                }
                else
                {
                    Console.WriteLine($"    {cuda.LiveGraphs} recorded graphs alive (nothing moves while one exists): the moves are not checked");
                }
            }
        }
        finally
        {
            CudaBackend.CheckAddressReads = checking;
        }

        void RunAtLimit()
        {
            long? limit = ComputeResources.GpuMemoryLimit;
            bool offloading = ComputeResources.OffloadToHostMemory;
            try
            {
                // A few pages above what is in use: most new blocks land in system memory, the cache and chunks still serve.
                ComputeResources.OffloadToHostMemory = true;
                ComputeResources.GpuMemoryLimit = ComputeResources.GetMemoryUsage(device).InUse + 4 * page;
                Run("at the limit, offloading on", offloading: true, moving: false);
            }
            finally
            {
                ComputeResources.GpuMemoryLimit = limit;
                ComputeResources.OffloadToHostMemory = offloading;
            }
        }

        ComputeResources.ReleaseCachedMemory(device);
        var end = ComputeResources.GetMemoryUsage(device);
        Check(end.InUse <= start.InUse, $"in use back where it started: {end.InUse:N0} bytes, {start.InUse:N0} before");
        Check(end.Cached <= start.Cached, $"nothing cached after the release: {end.Cached:N0} bytes, {start.Cached:N0} before");
        Check(end.Offloaded <= start.Offloaded, $"no system block left: {end.Offloaded:N0} bytes, {start.Offloaded:N0} before");
        Check(cuda is null || cuda.ChunkCount <= chunksBefore, $"empty chunks back to the driver: {cuda?.ChunkCount} held, {chunksBefore} before");

        void Run(string name, bool offloading, bool moving)
        {
            const int Workers = 4, Steps = 400, MostLive = 12;
            var errors = new System.Collections.Concurrent.ConcurrentQueue<string>();
            var live = new System.Collections.Concurrent.ConcurrentDictionary<ulong, int>();   // CUDA: block address → worker
            int finished = 0, releases = 0, graphs = 0, measurements = 0, onHost = 0, moves = 0, writes = 0;

            // Moving: tensors every worker reads (one of each size class) and one a writer fills again and again, moved
            // between the device and system memory by another thread meanwhile.
            var sharedSeeds = new[] { 7, 8, 9, 10 };
            Tensor Shared(int seed)
            {
                int length = Size(new Random(seed));
                return Tensor.From(Pattern(length, seed), [length], device);
            }

            var shared = moving ? sharedSeeds.Select(Shared).ToArray() : [];
            long movedOut = cuda?.MovedToHost ?? 0, movedIn = cuda?.MovedToDevice ?? 0;
            using var written = moving ? Tensor.Zeros([(int)(page / 4) + 1024], device) : null;
            long taken = 0;
            bool Done() => Volatile.Read(ref finished) == Workers;
            void Fail(string what, Exception e) => errors.Enqueue($"{what}: {e.GetType().Name}: {e.Message}");

            void Verify(Tensor t, int seed, string what)
            {
                using var doubled = t * 2f;                                    // a kernel reads the block
                var got = doubled.ToArray();
                for (int i = 0; i < got.Length; i++)
                {
                    if (got[i] != 2 * ((seed + i) % 251))
                    {
                        errors.Enqueue($"{what}: element {i} of {got.Length} is {got[i] / 2}, expected {(seed + i) % 251} (a block shared or overwritten)");
                        return;
                    }
                }
            }

            void Work(int worker)
            {
                var r = new Random(100 + worker);
                var held = new List<(Tensor Tensor, int Seed, ulong Address)>();
                void Free(int at)
                {
                    var (t, seed, address) = held[at];
                    held.RemoveAt(at);
                    Verify(t, seed, $"worker {worker}");
                    if (address != 0)
                    {
                        live.TryRemove(address, out _);                         // before the block can be handed out again
                    }

                    t.Dispose();
                }

                try
                {
                    for (int step = 0; step < Steps; step++)
                    {
                        if (moving)
                        {
                            int j = r.Next(shared.Length);
                            Verify(shared[j], sharedSeeds[j], $"worker {worker}, shared tensor {j} (moved meanwhile)");
                        }

                        if (held.Count == 0 || (held.Count < MostLive && r.Next(3) > 0))
                        {
                            int length = Size(r), seed = worker * 1_000 + step;
                            bool longLived = worker % 2 == 0 && r.Next(2) == 0;
                            Tensor t;
                            using (longLived ? device.Backend.LongLivedAllocations() : null)
                            {
                                t = Tensor.From(Pattern(length, seed), [length], device);
                            }

                            ulong address = t.Storage is CudaStorage block ? block.Pointer : 0;
                            if (address != 0 && !live.TryAdd(address, worker))
                            {
                                errors.Enqueue($"worker {worker}: block 0x{address:X} handed out while worker {live.GetValueOrDefault(address)} holds it");
                            }

                            held.Add((t, seed, address));
                            Interlocked.Increment(ref taken);
                            if (offloading && ComputeResources.GetMemoryUsage(device).Offloaded > start.Offloaded)
                            {
                                Volatile.Write(ref onHost, 1);
                            }
                        }
                        else
                        {
                            Free(r.Next(held.Count));
                        }
                    }
                }
                catch (Exception e)
                {
                    Fail($"worker {worker}", e);
                }
                finally
                {
                    while (held.Count > 0)
                    {
                        try
                        {
                            Free(held.Count - 1);
                        }
                        catch (Exception e)
                        {
                            Fail($"worker {worker} freeing", e);
                        }
                    }

                    Interlocked.Increment(ref finished);
                }
            }

            void Release()
            {
                try
                {
                    do                                                         // at least once, however soon the workers finish
                    {
                        ComputeResources.ReleaseCachedMemory(device);
                        Interlocked.Increment(ref releases);
                        Thread.Sleep(1);
                    }
                    while (!Done());
                }
                catch (Exception e)
                {
                    Fail("releasing the cache", e);
                }
            }

            void Record()
            {
                try
                {
                    var input = Pattern(4096, 3);
                    using var x = Tensor.From(input, [4096], device);
                    using var output = Tensor.Zeros([4096], device);
                    var expected = input.Select(v => 2 * v + 1).ToArray();
                    do
                    {
                        using var graph = ComputeGraph.Capture(device, () =>
                        {
                            using var doubled = x * 2f;                         // freed while recording: the graph's block
                            using var plus = doubled + 1f;
                            output.CopyFrom(plus);
                        });
                        if (device.Backend.SupportsGraphs && !graph.IsRecorded)
                        {
                            errors.Enqueue($"a graph not recorded: {graph.FailureReason}");
                        }

                        output.Fill(-1f);
                        graph.Replay();
                        var got = output.ToArray();
                        if (!got.AsSpan().SequenceEqual(expected))
                        {
                            errors.Enqueue($"a replayed graph wrote {got[0]}, {got[^1]}; expected {expected[0]}, {expected[^1]}");
                        }

                        Interlocked.Increment(ref graphs);
                    }
                    while (!Done());
                }
                catch (Exception e)
                {
                    Fail("recording graphs", e);
                }
            }

            void Measure()
            {
                // The attention path measured again and again: its scratch (the scores) is freed alone, past the cache.
                const int heads = 2, rows = 256, dim = 64;
                try
                {
                    var r = new Random(9);
                    float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => (float)(r.NextDouble() * 2 - 1))];
                    using var q = Tensor.From(Values(heads * rows * dim), [heads, rows, dim], device);
                    using var k = Tensor.From(Values(heads * rows * dim), [heads, rows, dim], device);
                    using var v = Tensor.From(Values(heads * rows * dim), [heads, rows, dim], device);
                    using var starts = Tensor.From(new float[heads * rows], [heads * rows], device);
                    using var ends = Tensor.From(Enumerable.Repeat((float)rows, heads * rows).ToArray(), [heads * rows], device);
                    while (!Done())
                    {
                        cuda!.ForgetMeasurements(TuneOp.AttentionPath);
                        cuda.PrefersComposedAttention(q.Storage, k.Storage, v.Storage, starts.Storage, ends.Storage, null, heads, rows, rows, dim, 0.125f);
                        Interlocked.Increment(ref measurements);
                    }
                }
                catch (Exception e)
                {
                    Fail("measuring the attention path", e);
                }
            }

            void Write()
            {
                try
                {
                    int k = 0;
                    do
                    {
                        float value = ++k % 251;
                        written!.Fill(value);                                  // a write a move must not lose
                        var got = written.ToArray();
                        int wrong = Array.FindIndex(got, v => v != value);
                        if (wrong >= 0)
                        {
                            errors.Enqueue($"a write lost to a move: element {wrong} is {got[wrong]}, {value} written");
                            return;
                        }

                        Interlocked.Increment(ref writes);
                    }
                    while (!Done());
                }
                catch (Exception e)
                {
                    Fail("writing a moved tensor", e);
                }
            }

            void Move()
            {
                try
                {
                    var offload = OffloadOf(device)!;
                    var targets = shared.Append(written!).ToArray();
                    int round = 0;
                    do
                    {
                        foreach (var t in targets)
                        {
                            offload.MoveToHost(t.Storage, keep: false);
                        }

                        Thread.Sleep(1);
                        if (++round % 2 == 0)
                        {
                            ComputeResources.ReturnOffloaded(device);                   // the step boundary's way back
                        }
                        else
                        {
                            foreach (var t in targets)
                            {
                                offload.MoveToDevice(t.Storage);
                            }
                        }
                    }
                    while (!Done());
                }
                catch (Exception e)
                {
                    Fail("moving tensors", e);
                }
            }

            var threads = new List<Thread>();
            for (int w = 0; w < Workers; w++)
            {
                int worker = w;
                threads.Add(new Thread(() => Work(worker)) { IsBackground = true });
            }

            threads.Add(new Thread(Release) { IsBackground = true });
            if (moving)
            {
                threads.Add(new Thread(Write) { IsBackground = true });
                threads.Add(new Thread(Move) { IsBackground = true });   // no graphs: nothing moves while one exists
            }
            else
            {
                threads.Add(new Thread(Record) { IsBackground = true });
            }

            if (cuda is not null && CudaBackend.Autotune && !offloading && !moving)
            {
                threads.Add(new Thread(Measure) { IsBackground = true });
            }

            var watch = System.Diagnostics.Stopwatch.StartNew();
            threads.ForEach(t => t.Start());
            threads.ForEach(t => t.Join());
            foreach (var t in shared)
            {
                t.Dispose();
            }

            if (cuda is not null)
            {
                moves = (int)(cuda.MovedToHost - movedOut + cuda.MovedToDevice - movedIn);
            }

            string recorded = device.Backend.SupportsGraphs ? "graphs recorded and replayed" : "graph steps replayed (no device graphs)";
            Console.WriteLine($"    {name}: {taken} blocks taken by {Workers} threads, {releases} cache releases, " +
                (moving ? $"{moves} moves, {writes} writes checked, " : $"{graphs} {recorded}, {measurements} attention-path measurements, ") +
                $"{watch.Elapsed.TotalSeconds:F1} s");
            Check(errors.IsEmpty, $"{name}: {errors.Count} failures; first: {string.Join(" | ", errors.Take(5))}");
            Check((moving ? moves > 0 && writes > 0 : graphs > 0) && releases > 0, $"{name}: the other threads ran ({graphs} graphs, {moves} moves, {writes} writes, {releases} releases)");
            Check(!offloading || Volatile.Read(ref onHost) == 1, $"{name}: blocks went to system memory at the limit");
        }
    }

    // The move guard's gate (a device that offloads), with threads and timeouts (a failure ends the wait instead of hanging the run).
    internal static void MoveGateOrder(Device device)
    {
        var gate = new MoveGate();
        var timeout = TimeSpan.FromSeconds(10);
        static bool Quiet(ManualResetEventSlim e) => !e.Wait(150);         // not set within a moment

        using var readerIn = new ManualResetEventSlim();
        using var readerOut = new ManualResetEventSlim();
        var reader = new Thread(() =>
        {
            gate.Enter();
            readerIn.Set();
            readerOut.Wait();
            gate.Exit();
        }) { IsBackground = true };
        reader.Start();
        Check(readerIn.Wait(timeout), "the reader entered");
        Check(!gate.Holds, "a thread outside any operation holds no address");
        Check(!gate.BeginMove(wait: false, lookupTablesOnly: false), "a move that may not wait does not run while another thread is inside an operation");

        // A waiting move: after the reader's operation; meanwhile, and while it runs, a new operation waits.
        using var moved = new ManualResetEventSlim();
        using var endMove = new ManualResetEventSlim();
        bool ownWork = false, nested = false;
        var mover = new Thread(() =>
        {
            if (!gate.BeginMove(wait: true, lookupTablesOnly: false))
            {
                return;
            }

            gate.Enter();                                                 // the mover's own operation goes through
            ownWork = gate.Holds;
            gate.Exit();
            nested = gate.BeginMove(wait: true, lookupTablesOnly: false);                           // and so does a nested move
            gate.EndMove();
            moved.Set();
            endMove.Wait();
            gate.EndMove();
        }) { IsBackground = true };
        mover.Start();
        Check(Quiet(moved), "the move waits for the reader's operation");

        using var lateIn = new ManualResetEventSlim();
        using var lateOut = new ManualResetEventSlim();
        var late = new Thread(() =>
        {
            gate.Enter();
            lateIn.Set();
            lateOut.Wait();
            gate.Exit();
        }) { IsBackground = true };
        late.Start();
        Check(Quiet(lateIn), "a new operation waits while a move waits");
        readerOut.Set();
        Check(moved.Wait(timeout), "the move runs once the reader's operation ended");
        Check(ownWork && nested, "the mover's own operation and its nested move went through");
        Check(Quiet(lateIn), "a new operation waits while the move runs");
        endMove.Set();
        Check(lateIn.Wait(timeout), "the new operation runs once the move ended");

        // From inside an operation a move never waits: with another thread inside one it does not run; alone it runs.
        gate.Enter();
        Check(gate.Holds, "inside an operation the addresses hold");
        var watch = System.Diagnostics.Stopwatch.StartNew();
        Check(!gate.BeginMove(wait: true, lookupTablesOnly: false), "from inside an operation, with another thread inside one: no move");
        Check(watch.Elapsed < TimeSpan.FromSeconds(1), $"and no waiting ({watch.Elapsed.TotalMilliseconds:F0} ms)");
        lateOut.Set();
        Check(late.Join(timeout), "the other operation ended");
        Check(gate.BeginMove(wait: true, lookupTablesOnly: false), "from inside an operation, alone: the move runs");
        gate.EndMove();
        gate.Exit();
        Check(reader.Join(timeout) && mover.Join(timeout), "every thread ended");
        Check(gate.BeginMove(wait: false, lookupTablesOnly: false), "with nobody inside an operation even a move that may not wait runs");
        gate.EndMove();

        // A placement of lookup tables from inside an operation (thread A) waits for the other operation (thread B); B,
        // needing one too, steps aside and gets false, counted again before A's placement ends.
        using var bIn = new ManualResetEventSlim();
        using var bGo = new ManualResetEventSlim();
        using var bParked = new ManualResetEventSlim();
        using var bLeave = new ManualResetEventSlim();
        bool parkedMoved = true;
        var b = new Thread(() =>
        {
            gate.Enter();
            bIn.Set();
            bGo.Wait();
            parkedMoved = gate.BeginMove(wait: true, lookupTablesOnly: true);
            bParked.Set();
            bLeave.Wait();
            gate.Exit();
        }) { IsBackground = true };
        b.Start();
        Check(bIn.Wait(timeout), "B entered");
        using var aMoved = new ManualResetEventSlim();
        using var aEnd = new ManualResetEventSlim();
        var a = new Thread(() =>
        {
            gate.Enter();
            if (gate.BeginMove(wait: true, lookupTablesOnly: true))
            {
                aMoved.Set();
                aEnd.Wait();
                gate.EndMove();
            }

            gate.Exit();
        }) { IsBackground = true };
        a.Start();
        Check(Quiet(aMoved), "a placement from inside an operation waits for the other operation");
        bGo.Set();
        Check(aMoved.Wait(timeout), "it runs once the other thread steps aside");
        Check(Quiet(bParked), "the thread that stepped aside waits while the placement runs");
        Check(!gate.BeginMove(wait: false, lookupTablesOnly: false), "no other move while the placement runs");
        aEnd.Set();
        Check(bParked.Wait(timeout) && !parkedMoved, "after the placement the thread that stepped aside gets false (its allocation tries again)");
        Check(a.Join(timeout), "A ended");
        Check(!gate.BeginMove(wait: false, lookupTablesOnly: false), "the thread that stepped aside counts again: a move must wait for it");
        bLeave.Set();
        Check(b.Join(timeout), "B ended");

        // A general move waiting for an operation gives way to that operation's placement of lookup tables.
        using var rIn = new ManualResetEventSlim();
        using var rGo = new ManualResetEventSlim();
        using var rMoved = new ManualResetEventSlim();
        using var rLeave = new ManualResetEventSlim();
        bool placement = false;
        var r = new Thread(() =>
        {
            gate.Enter();
            rIn.Set();
            rGo.Wait();
            placement = gate.BeginMove(wait: true, lookupTablesOnly: true);
            if (placement)
            {
                gate.EndMove();
            }

            rMoved.Set();
            rLeave.Wait();
            gate.Exit();
        }) { IsBackground = true };
        r.Start();
        Check(rIn.Wait(timeout), "R entered");
        using var gDone = new ManualResetEventSlim();
        bool general = true;
        var g = new Thread(() =>
        {
            general = gate.BeginMove(wait: true, lookupTablesOnly: false);
            if (general)
            {
                gate.EndMove();
            }

            gDone.Set();
        }) { IsBackground = true };
        g.Start();
        Check(Quiet(gDone), "the general move waits for R's operation");
        rGo.Set();
        Check(gDone.Wait(timeout) && !general, "the general move gave way (false: its caller does without)");
        Check(rMoved.Wait(timeout) && placement, "R's placement ran");
        rLeave.Set();
        Check(r.Join(timeout) && g.Join(timeout), "R and the general move ended");
    }

    // The allocation that found the device full with a lookup table on it, while other threads read the table without pause.
    internal static void LookupTablesUnderReaders(Device device)
    {
        if (OffloadOf(device) is not { } offload || device.Backend is not CudaBackend cuda)
        {
            return;                                                          // nothing moves on a device without offloading
        }

        const int Floats = 1 << 20, Readers = 3;
        var values = new float[Floats];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = i % 2;                                               // sums exact in float32
        }

        long? limit = ComputeResources.GpuMemoryLimit;
        try
        {
            for (int round = 0; round < 3; round++)
            {
                ComputeResources.ReleaseCachedMemory(device);
                using var table = Tensor.From(values, [Floats], device);
                offload.MarkLookupTable(table.Storage);
                float expected;
                using (var first = table.Sum())
                {
                    expected = first.Item();
                }

                var errors = new System.Collections.Concurrent.ConcurrentQueue<string>();
                int stop = 0, reads = 0, readsAfter = 0;
                long placed = cuda.PlacedOnHost;
                var readers = Enumerable.Range(0, Readers).Select(_ => new Thread(() =>
                {
                    try
                    {
                        while (Volatile.Read(ref stop) == 0)
                        {
                            using var sum = table.Sum();                    // an operation reading the whole table
                            float got = sum.Item();
                            if (got != expected)
                            {
                                errors.Enqueue($"a read gave {got}, expected {expected}");
                                return;
                            }

                            Interlocked.Increment(ref reads);
                            if (cuda.PlacedOnHost > placed)
                            {
                                Interlocked.Increment(ref readsAfter);
                            }
                        }
                    }
                    catch (Exception e)
                    {
                        errors.Enqueue($"{e.GetType().Name}: {e.Message}");
                    }
                }) { IsBackground = true }).ToList();
                readers.ForEach(t => t.Start());
                try
                {
                    SpinWait.SpinUntil(() => Volatile.Read(ref reads) >= Readers * 4, TimeSpan.FromSeconds(10));

                    // The device at its limit: a block of three quarters of the table fits only once the table has left.
                    ComputeResources.GpuMemoryLimit = ComputeResources.GetMemoryUsage(device).InUse + 4L * Floats / 2;
                    using (var block = Tensor.Zeros([Floats * 3 / 4], device))
                    {
                        Check(cuda.PlacedOnHost == placed + 1, $"round {round}: the table moved to system memory and the allocation succeeded");
                    }

                    ComputeResources.GpuMemoryLimit = limit;
                    SpinWait.SpinUntil(() => Volatile.Read(ref readsAfter) >= Readers * 4 || !errors.IsEmpty, TimeSpan.FromSeconds(10));
                }
                finally
                {
                    ComputeResources.GpuMemoryLimit = limit;
                    Volatile.Write(ref stop, 1);
                    readers.ForEach(t => t.Join());
                }

                Check(errors.IsEmpty, $"round {round}: {string.Join(" | ", errors.Take(3))}");
                Check(readsAfter > 0, $"round {round}: reads of the table in system memory ({reads} reads, {readsAfter} after the move)");
            }
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = limit;
            ComputeResources.ReleaseCachedMemory(device);
        }
    }
}
