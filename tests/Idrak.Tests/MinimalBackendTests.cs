// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;
using Idrak.Generation;
using Idrak.Layers;

// A backend with only the members every device must have (memory, copies, its name): every other operation runs through
// the host fallback. The whole test list runs on it with IDRAK_DEVICES=cpu,minimal; the tests below check it from any device.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] MinimalBackendGroup =
    [
        ("minimum backend: a device with only memory and copies (every operation through the host fallback) is listed by name, trains and decodes like the CPU", MinimalBackendMatches),
        ("minimum backend: the first host fallback of each operation is reported to telemetry once (a console logger prints it, JSON Lines writes it), every call counted", MinimalBackendReportsHostFallbacks),
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

        // The CPU's limits, and no attention over key ranges of its own (every head size takes the host fallback): attention
        // that knows its ranges composes on this device where the memory it reports holds the scores.
        public override BackendCapabilities Capabilities { get; } = CpuBackend.Instance.Capabilities with { SpanAttentionHeadDim = 0 };

        public override string Kind => "minimal";

        // Its memory is the process's: what the CPU reports available.
        public override long? AvailableMemory() => CpuBackend.Instance.AvailableMemory();

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

    // Keeps the HostFallback events of one kind of device.
    private sealed class HostFallbacks(string kind) : ITelemetryHook
    {
        private readonly List<HostFallback> _events = [];

        public TelemetryLevel Levels => TelemetryLevel.Devices;

        public List<HostFallback> Events
        {
            get
            {
                lock (_events)
                {
                    return [.. _events];
                }
            }
        }

        public void OnHostFallback(in HostFallback e)
        {
            if (e.Device == kind)
            {
                lock (_events)
                {
                    _events.Add(e);
                }
            }
        }
    }

    private static void MinimalBackendReportsHostFallbacks(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return;                                                      // a device of its own, once
        }

        var minimal = new MinimalBackend();                              // nothing reported on it yet
        var x = minimal.Allocate(6, zeroed: false);
        var y = minimal.Allocate(6, zeroed: true);
        minimal.Upload([1f, 2f, 3f, -1f, 0f, 1f], x);
        var fallbacks = new HostFallbacks("minimal");
        var printed = new StringWriter();
        long hostCalls = Kernels.HostCalls(minimal);
        using (Telemetry.Subscribe(fallbacks))
        using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Training, output: printed)))
        {
            for (int i = 0; i < 3; i++)
            {
                minimal.Softmax(x, y, 2, 3, log: false);
            }
        }

        Check(Kernels.HostCalls(minimal) - hostCalls == 3, $"every call counted: {Kernels.HostCalls(minimal) - hostCalls} of 3");
        var events = fallbacks.Events;
        Check(events is [{ Device: "minimal", Operation: "Softmax" }], $"one HostFallback event for three calls: {string.Join(", ", events)}");
        Check(printed.ToString().Contains("minimal Softmax runs on the host", StringComparison.Ordinal),
            $"a console logger subscribed for training prints the first host fallback: {printed}");

        var json = new MemoryStream();
        using (var writer = new Utf8JsonWriter(json))
        {
            TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.UtcNow, events[0]));
        }

        string line = Encoding.UTF8.GetString(json.ToArray());
        Check(line.Contains("\"event\":\"host_fallback\"", StringComparison.Ordinal) && line.Contains("\"operation\":\"Softmax\"", StringComparison.Ordinal)
              && line.Contains("\"device\":\"minimal\"", StringComparison.Ordinal), $"JSON Lines: {line}");
    }
}
