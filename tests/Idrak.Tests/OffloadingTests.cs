using Idrak;
using Idrak.Backends;
using Idrak.Backends.Cuda;
using Idrak.Layers;
using Idrak.LanguageModels;
using Idrak.Optimizers;

// Offloading to system memory (IMemoryOffload): the CPU optimizer step, moving tensors, staging layers' weights, cold
// data first, and bringing tensors back.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] OffloadingGroup =
    [
        ("offloading: HostOptimizer (AdamW, Adam, SGD with momentum, run on the CPU) gives the weights of the same optimizer on the device, with and without clipping", HostOptimizerMatches),
        ("offloading: a tensor moved to system memory and back keeps its values; nothing moves while a recorded graph exists", OffloadMoves),
        ("offloading: layers whose weights live in system memory give the same outputs and gradients, their weights staged on the GPU (forward and backward, the next layer prefetched) and back home after", OffloadStaging),
        ("offloading: a decoder whose base weights (float32 and int8) live in system memory gives the same outputs and LoRA gradients through its fused products (q/k/v, gate/up, down), staged as groups", OffloadDecoder),
        ("offloading: training with a full GPU moves optimizer state out first (then frozen weights), gives the weights of an ordinary run, and brings tensors back when memory frees up", OffloadColdFirst),
    ];

    private static Sequential OffloadModel(Device device, int seed) => new(
        new Linear(16, 96, device: device, random: new Random(seed)), new ReLU(),
        new Linear(96, 96, device: device, random: new Random(seed + 1)), new ReLU(),
        new Linear(96, 4, device: device, random: new Random(seed + 2)));

    // A few training steps (MSE on fixed data); returns every parameter's values.
    private static float[][] TrainSteps(Sequential model, Optimizer optimizer, Device device, int steps, float clip, Action? afterStep = null)
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

    private static void HostOptimizerMatches(Device device)
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

    private static IMemoryOffload? OffloadOf(Device device) => device.Backend.Offload;

    private static void OffloadMoves(Device device)
    {
        if (OffloadOf(device) is not { } offload)
        {
            Check(device.Type == DeviceType.Cpu, "every GPU backend offloads");
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

    private static void OffloadStaging(Device device)
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

    private static void OffloadDecoder(Device device)
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

    private static void OffloadColdFirst(Device device)
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
}
