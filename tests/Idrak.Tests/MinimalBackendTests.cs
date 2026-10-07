// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Generation;
using Idrak.Layers;

// A backend with only the members every device must have (memory, copies, its name): every other operation runs through
// the host fallback. The whole test list runs on it with IDRAK_DEVICES=cpu,minimal; the tests below check it from any device.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] MinimalBackendGroup =
    [
        ("minimum backend: a device with only memory and copies (every operation through the host fallback) is listed by name, trains and decodes like the CPU", MinimalBackendMatches),
    ];

    // Plain arrays standing for device memory.
    private sealed class MinimalStorage(Backend backend, float[] data, int length) : Storage(backend, length)
    {
        public float[] Data = data;
    }

    internal sealed class MinimalBackend : Backend
    {
        public static readonly MinimalBackend Instance = new();
        private readonly MemoryAccountant _memory = new(() => ComputeResources.GpuMemoryLimit, "minimal:0");

        public override BackendCapabilities Capabilities => CpuBackend.Instance.Capabilities;

        public override string Kind => "minimal";

        public override string Name => "minimal (memory and copies only)";

        public int Calls;

        public override Storage Allocate(int length, bool zeroed)
        {
            _memory.MustReleaseCacheFor(length * 4L);                     // throws over the limit
            _memory.Allocated(length * 4L);
            return new MinimalStorage(this, new float[length], length);
        }

        public override void Return(Storage storage)
        {
            _memory.Returned(storage.Length * 4L);
            _memory.Freed(storage.Length * 4L);                           // nothing cached
        }

        protected override void Detach(Storage storage) => ((MinimalStorage)storage).Data = null!;

        protected override void Attach(Storage storage, Storage fresh) => ((MinimalStorage)storage).Data = ((MinimalStorage)fresh).Data;

        public override MemoryUsage GetMemoryUsage() => _memory.Usage;

        public override void ReleaseCachedMemory()
        {
        }

        public override void Upload(ReadOnlySpan<float> source, Storage destination)
        {
            Interlocked.Increment(ref Calls);
            source.CopyTo(((MinimalStorage)destination).Data);
        }

        public override void Download(Storage source, Span<float> destination)
        {
            Interlocked.Increment(ref Calls);
            ((MinimalStorage)source).Data.AsSpan(0, destination.Length).CopyTo(destination);
        }

        public override void DownloadRange(Storage source, int offset, Span<float> destination) =>
            ((MinimalStorage)source).Data.AsSpan(offset, destination.Length).CopyTo(destination);

        public override void Synchronize()
        {
        }
    }

    internal sealed class MinimalProvider : DeviceProvider
    {
        public override string Kind => "minimal";

        public override DeviceType Type => DeviceType.Other;

        public override int Count => 1;

        public override Backend Create(int ordinal) => MinimalBackend.Instance;

        public override bool IsStarted(int ordinal) => false;

        public override bool Listed(int ordinal) => false;                 // only by name

        public override string? Note(int ordinal) => "test backend (memory and copies only): by name only";
    }

    private static void MinimalBackendMatches(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                      // compared with the CPU once
        }

        var minimal = Device.Get("minimal");
        Check(minimal.ToString() == "minimal:0" && Device.Parse("minimal:0") == minimal && !Device.Available.Contains(minimal), "found by name, not listed");

        // A small network trained a few steps on each device.
        float[] Train(Device d)
        {
            var r = new Random(3);
            var x = Enumerable.Range(0, 32 * 8).Select(_ => r.NextSingle() * 2 - 1).ToArray();
            var t = Enumerable.Range(0, 32 * 3).Select(i => i % 3 == i / 3 % 3 ? 1f : 0f).ToArray();
            using var model = new Sequential(new Linear(8, 16, device: d, random: new Random(5)), new ReLU(), new LayerNorm(16, device: d), new Linear(16, 3, device: d, random: new Random(6)));
            using var optimizer = new Idrak.Abstraction.Training.AdamW(model.Parameters(), 1e-2f);
            var losses = new List<float>();
            for (int step = 0; step < 5; step++)
            {
                using var scope = new TensorScope();
                var loss = Losses.CrossEntropy(model.Forward(Tensor.From(x, [32, 8], d)), Tensor.From(t, [32, 3], d));
                optimizer.ZeroGrad();
                loss.Backward();
                optimizer.Step();
                losses.Add(loss.ToArray()[0]);
            }

            return [.. losses];
        }

        int before = MinimalBackend.Instance.Calls;
        AssertClose(Train(Device.Cpu), Train(minimal), 1e-4f, "training losses");
        Check(MinimalBackend.Instance.Calls > before, "the work went through the minimal device's copies");

        // A decoder generating greedily, with the KV cache.
        string Generate(Device d)
        {
            var (model, tokenizer) = TinyLanguageModel(d);
            var options = new GenerationOptions { TopK = 1, Temperature = 1f, RepeatPenalty = 1f, NumPredict = 8, UseCache = true, Seed = 1 };
            return new TextGenerator(model, tokenizer, 32) { KeepCache = false }.Generate("abc", options).Text;
        }

        Check(Generate(Device.Cpu) == Generate(minimal), "greedy generation matches the CPU");
    }
}
