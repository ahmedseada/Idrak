// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Devices;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 37, W0 and W1: what crosses the bus per token (OffloadTraffic, printed by idrak run), and a decoder split between the
// device and the CPU (WeightPlacements), whose results are the all-device run's. W3: a split's CPU layers and lazily built
// layers read their weights in place from the checkpoint's file mapping.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] OffloadTelemetryGroup =
    [
        ("offload telemetry: the counts add up, a snapshot's difference is what crossed since, the line and the JSON give per-token figures; a model on one device moves nothing across the bus", OffloadTrafficCounts),
        ("offload telemetry: a split model's activations are counted at each crossing (both ways, a step's positions × width × 4 bytes), no weight crosses; idrak run -v -j prints the placement and a \"bus\" object", SplitTrafficCounted),
        ("offload telemetry: staging a layer's offloaded weights allocates nothing once warm (its storages found once, the forward's scope pooled), the weights staged on every pass and the next layer prefetched", StagingAllocatesNothing),
    ];

    private static readonly (string Name, Action<Device> Run)[] LazyLoadGroup =
    [
        ("lazy load: a lazily loaded Q8_0 GGUF model's matrices are read in place from the file as kept (W3), each block matrix mapped or its copy counted with why, the loss of the eager load bit for bit", LazyGgufReadsTheMapping),
    ];

    private static readonly (string Name, Action<Device> Run)[] WeightPlacementGroup =
    [
        ("weight placement: device, split, stream and auto plan from the request (the layers the free memory holds beside the fixed weights, the working memory and each layer's cache; the user's count; stream refused where the device cannot and the model does not fit; auto keeps a model that fits on the device); device layers parse", PlacementPlans),
        ("weight placement: layers follow the free memory at step boundaries (more on the device when it frees, fewer under pressure, a user's count kept), their weights and caches moved, the run going on with the all-CPU logits; FollowMemory off keeps them; lazy layers built where they then belong", LayersFollowMemory),
    ];

    private static string PlacementModel => PretrainData("tiny-llama");

    private static void OffloadTrafficCounts(Device device)
    {
        var before = OffloadTraffic.Read();
        OffloadTraffic.CountStaged(100);
        OffloadTraffic.CountPrefetched(50);
        OffloadTraffic.CountMapped(25);
        OffloadTraffic.CountLayer("", staged: true, mapped: false);
        OffloadTraffic.CountLayer("Linear up", staged: false, mapped: true);
        OffloadTraffic.CountBoundary(16);
        var counted = OffloadTraffic.Read().Since(before);
        Check(counted is { StagedBytes: 100, PrefetchedBytes: 50, MappedBytes: 25, BoundaryBytes: 16, BoundaryCrossings: 1, LayersStaged: 1, LayersMapped: 1, BusBytes: 191 }
              && counted.MappedLayers.Count == 1 && counted.MappedLayers["Linear up"] == 1 && !counted.IsEmpty, $"the counts: {counted}");
        var json = counted.ToJson(2);
        string line = counted.Describe(2);
        Check((long)json["bytes"]! == 191 && (long)json["bytes_per_token"]! == 95 && (double)json["layers_staged_per_token"]! == 0.5 && (long)json["mapped_layers"]!["Linear up"]! == 1
              && line.StartsWith("bus: 191 B, 95 B a token", StringComparison.Ordinal) && line.Contains("Linear up ×1", StringComparison.Ordinal), $"per token: {line} {json}");
        Check(OffloadTraffic.Read().Since(OffloadTraffic.Read()).IsEmpty, "nothing since a snapshot of now");

        // A model on one device: nothing crosses while it decodes.
        using var model = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device });
        var start = OffloadTraffic.Read();
        Decode(model, truncate: false);
        Check(OffloadTraffic.Read().Since(start).IsEmpty, $"{device}: a model on one device moves nothing across the bus ({OffloadTraffic.Read().Since(start).Describe(4)})");
    }

    private static void SplitTrafficCounted(Device device)
    {
        int dim = TinyLlamaSpec().Dim;
        if (device.Type != DeviceType.Cpu)
        {
            using var split = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Split, DeviceLayers = 1 });
            var start = OffloadTraffic.Read();
            Decode(split, truncate: false);                              // a plain pass and a prompt of 4 positions, then 3 steps of one
            var crossed = OffloadTraffic.Read().Since(start);
            long expected = 2L * (4 + 4 + 3) * dim * sizeof(float);      // to the CPU and back, at every position
            Check(crossed is { StagedBytes: 0, PrefetchedBytes: 0, MappedBytes: 0, BoundaryCrossings: 10 } && crossed.BoundaryBytes == expected,
                $"{device}: activations only, {expected} bytes in 10 crossings ({crossed.Describe(5)})");
        }

        var (code, text, error) = RunIdrakOn(device, null, "run", CliModel, "Hello", "--max-tokens", "3", "--temperature", "0", "-j", "-v", "--device-layers", "1");
        Check(code == 0, $"idrak run on {device}: {error}");
        var bus = JsonOf(text, "run -j -v")["bus"] as JsonObject;
        Check(bus is not null && (long)bus["staged_bytes"]! == 0 && (long)bus["mapped_bytes"]! == 0, $"a \"bus\" object, no weight staged or read in place: {text}");
        if (device.Type == DeviceType.Cpu)
        {
            Check((long)bus!["bytes"]! == 0 && !error.Contains("placement", StringComparison.Ordinal), $"on the CPU nothing is split and nothing crosses: {bus} {error}");
        }
        else
        {
            Check((long)bus!["boundary_bytes"]! > 0 && (long)bus["boundary_crossings"]! % 2 == 0 && error.Contains("placement auto: 1 of 2 layers on", StringComparison.Ordinal),
                $"{device}: the split said and its activations counted: {bus} {error}");
        }
    }

    private static void PlacementPlans(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(PlacementPlans)))
        {
            return;
        }

        var gpu = Device.Cpu;                                            // the plans read the request's figures, not the device's
        WeightPlacementRequest Request(long? free, int? asked = null, bool stream = false) =>
            new(gpu, 10, LayerBytes: 100, FixedBytes: 50, LayerCacheBytes: 10, WorkingBytes: 40, free, asked, stream);
        Check(Request(50 + 40 + 3 * 110 + 109).LayersThatFit == 3 && Request(10).LayersThatFit == 0 && Request(null).LayersThatFit == 10 && Request(5000).Fits,
            "the layers that fit: room after the fixed weights and the working memory, a layer and its cache each");
        var device_ = new WeightPlacements.DevicePlacement().Place(Request(200));
        var split = new WeightPlacements.SplitPlacement().Place(Request(50 + 40 + 4 * 110));
        var asked = new WeightPlacements.SplitPlacement().Place(Request(50 + 40 + 4 * 110, asked: 7));
        Check(device_ is { DeviceLayers: 10, Stream: false, Split: false } && split is { DeviceLayers: 4, Layers: 10, Split: true } && asked is { DeviceLayers: 7, Split: true }
              && split.ToString().StartsWith("split: 4 of 10 layers on cpu, 6 computed by the CPU", StringComparison.Ordinal),
            $"device all; split as the memory holds ({split}); split as asked ({asked})");
        var streamed = new WeightPlacements.StreamPlacement().Place(Request(300, stream: true));
        Check(streamed is { Stream: true, DeviceLayers: 10 } && new WeightPlacements.StreamPlacement().Place(Request(5000)) is { Stream: false, DeviceLayers: 10 }, $"stream: {streamed}");
        bool refused = false;
        try
        {
            new WeightPlacements.StreamPlacement().Place(Request(300));
        }
        catch (NotSupportedException)
        {
            refused = true;
        }

        Check(refused, "stream where the device cannot and the model does not fit: refused");
        Check(new WeightPlacements.AutoPlacement().Place(Request(5000)) is { DeviceLayers: 10, Split: false, Stream: false }
              && new WeightPlacements.AutoPlacement().Place(Request(300)) is { DeviceLayers: 1, Split: true }
              && new WeightPlacements.AutoPlacement().Place(Request(5000, asked: 3)) is { DeviceLayers: 3, Split: true },
            "auto: the device when it fits, a split where the device cannot stream, the user's count when given");
        bool bad = false;
        try
        {
            WeightPlacements.ParseDeviceLayers("many", "--device-layers");
        }
        catch (ArgumentException e)
        {
            bad = e.Message.Contains("--device-layers", StringComparison.Ordinal);
        }

        Check(WeightPlacements.ParseDeviceLayers("12", "x") == 12 && WeightPlacements.ParseDeviceLayers("auto", "x") is null && WeightPlacements.ParseDeviceLayers(null, "x") is null && bad,
            "device layers: a count, auto or nothing; anything else names where it came from");
    }

    private static void LayersFollowMemory(Device device)
    {
        if (FirstRun(nameof(LayersFollowMemory)))
        {
            // Follow: as many more as the room past the working memory holds; under pressure as many fewer as give it back.
            var r = new WeightPlacementRequest(Device.Cpu, 10, LayerBytes: 100, FixedBytes: 50, LayerCacheBytes: 10, WorkingBytes: 40, AvailableBytes: 40 + 2 * 110 + 5, DeviceLayers: null, CanStream: false);
            Check(WeightPlacements.Follow(r, 3) == 5 && WeightPlacements.Follow(r with { AvailableBytes = 0 }, 3) == 2 && WeightPlacements.Follow(r with { AvailableBytes = 100 }, 3) == 3
                  && WeightPlacements.Follow(r with { AvailableBytes = 0 }, 0) == 0 && WeightPlacements.Follow(r with { AvailableBytes = long.MaxValue / 4 }, 3) == 10
                  && WeightPlacements.Follow(r with { DeviceLayers = 4 }, 3) == 3 && WeightPlacements.Follow(r with { AvailableBytes = null }, 3) == 3
                  && ((IWeightPlacement)new WeightPlacements.DevicePlacement()).Adjust(r, 3) == 3 && new WeightPlacements.SplitPlacement().Adjust(r, 3) == 5 && new WeightPlacements.AutoPlacement().Adjust(r, 3) == 5,
                "grow by what the room holds, shrink by what the pressure asks, keep a user's count; device keeps its layers");
        }

        if (device.Type == DeviceType.Cpu || device.Backend.AvailableMemory() is null)
        {
            return;                                                      // nothing to move between on the CPU (or no free memory reported)
        }

        using var reference = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = Device.Cpu });
        var spec = reference.Spec;
        var request = AutoWeights.PlacementRequest(spec, "float32", Math.Min(AutoWeights.DefaultContext, spec.MaxPositions), device, null, null);
        long budget = request.FixedBytes + request.WorkingBytes + request.LayerBytes + request.LayerCacheBytes + request.LayerBytes / 2;
        long? reserveBefore = ComputeResources.GpuMemoryReserve;
        try
        {
            foreach (bool lazy in new[] { false, true })
            {
                ComputeResources.GpuMemoryReserve = 0;                   // the device's whole free memory: room to grow
                using var model = PretrainedModel.Load(PlacementModel, new PretrainedOptions
                {
                    Device = device, WeightPlacement = WeightPlacements.Auto, AutoWeightsBudget = budget, LazyLayers = lazy,
                });
                Check(model.Placement is { Split: true, DeviceLayers: 1 } && model.DeviceLayersNow == 1, $"lazy {lazy}: split at load ({model.Placement})");
                using var context = new DecodingContext(device, 1, 32);
                using var cpuContext = new DecodingContext(Device.Cpu, 1, 32);
                var moves = OffloadTraffic.Read();
                var counts = new List<int>();
                using (Autograd.NoGrad())
                {
                    void Step(float[] ids, string what)
                    {
                        using var scope = new TensorScope();
                        var got = model.Network.ForwardCached(Tensor.From(ids, [1, ids.Length], device), context).ToArray();
                        var expected = reference.Network.ForwardCached(Tensor.From(ids, [1, ids.Length], Device.Cpu), cpuContext).ToArray();
                        AssertClose(expected, got, 2e-3f, $"lazy {lazy}, {what} ({model.DeviceLayersNow} layers on {device})");
                        counts.Add(model.DeviceLayersNow ?? -1);
                    }

                    Step([1f, 5f, 9f, 3f], "the prompt");                   // its step's end: the room frees, layer 1 comes to the device
                    Step([7f], "a step on the device");
                    ComputeResources.GpuMemoryReserve = long.MaxValue / 4;   // nothing reported free: pressure
                    Step([2f], "a step under pressure");                   // its end: layers go back to the CPU
                    Step([11f], "a step after shrinking");
                    ComputeResources.GpuMemoryReserve = 0;
                    Step([4f], "a step with room again");
                    Step([6f], "a step after growing again");
                }

                var moved = OffloadTraffic.Read().Since(moves);
                Check(counts[0] == 2 && counts[1] == 2 && counts[2] < 2 && counts[3] <= counts[2] && counts[4] == 2 && counts[5] == 2 && moved.LayersMoved >= 3 && moved.MovedBytes > 0,
                    $"lazy {lazy}: device layers after each step {string.Join(", ", counts)}; {moved.LayersMoved} layers moved ({moved.MovedBytes} bytes)");
            }

            // Following off: the layers stay where the load put them.
            ComputeResources.GpuMemoryReserve = 0;
            using var still = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Auto, AutoWeightsBudget = budget, FollowMemory = false });
            Decode(still, truncate: false);
            Check(still.DeviceLayersNow == 1, $"FollowMemory off: still {still.DeviceLayersNow} layer on {device}");
        }
        finally
        {
            ComputeResources.GpuMemoryReserve = reserveBefore;
        }
    }

    // A device whose every storage counts as offloaded: Stage hands back one token, Prefetch counts (the staging hooks
    // measured without a GPU's offloading).
    private sealed class StagingBackend : Backend, IOffloadingBackend, IMemoryOffload
    {
        public static readonly StagingBackend Instance = new();

        public long Stages, Prefetches;
        private readonly Token _token = new();

        public override BackendCapabilities Capabilities { get; } = Device.Cpu.Backend.Capabilities;

        public override string Kind => "staging-test";

        public override string Name => "staging test (every storage offloaded)";

        public override Storage Allocate(int length, bool zeroed) => new MinimalStorage(this, new float[length], length);

        public override void Return(Storage storage)
        {
        }

        protected override void Detach(Storage storage) => ((MinimalStorage)storage).Data = null!;

        protected override void Attach(Storage storage, Storage fresh) => ((MinimalStorage)storage).Data = ((MinimalStorage)fresh).Data;

        public override MemoryUsage GetMemoryUsage() => new(0, 0, null);

        public override void ReleaseCachedMemory()
        {
        }

        public override void Upload(ReadOnlySpan<float> source, Storage destination) => source.CopyTo(((MinimalStorage)destination).Data);

        public override void Download(Storage source, Span<float> destination) => ((MinimalStorage)source).Data.AsSpan(0, destination.Length).CopyTo(destination);

        public override void DownloadRange(Storage source, int offset, Span<float> destination) =>
            ((MinimalStorage)source).Data.AsSpan(offset, destination.Length).CopyTo(destination);

        public override void Synchronize()
        {
        }

        IMemoryOffload IOffloadingBackend.Offload => this;

        public int OffloadedCount => 1;

        public bool IsOffloaded(Storage storage) => true;

        public void SetPriority(Storage storage, OffloadPriority priority)
        {
        }

        public bool MoveToHost(Storage storage, bool keep) => false;

        public bool MoveToDevice(Storage storage) => false;

        public int Rebalance(bool makeRoom = true) => 0;

        public void Prefetch(IReadOnlyList<Storage> storages) => Prefetches++;

        public IDisposable? Stage(IReadOnlyList<Storage> storages)
        {
            Stages++;
            return _token;
        }

        private sealed class Token : IDisposable
        {
            public void Dispose()
            {
            }
        }
    }

    private static void StagingAllocatesNothing(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(StagingAllocatesNothing)))
        {
            return;
        }

        var staging = NotOnTheNewDesign.Yet<Device>("A test device beside the CPU");
        var backend = StagingBackend.Instance;
        using var first = new Linear(8, 8, device: staging, random: new Random(1));
        using var second = new Linear(8, 8, device: staging, random: new Random(2));
        using var input = Tensor.From(new float[8], [1, 8], staging);
        var hooks = TensorOffloading.Hooks!;
        void Pass()
        {
            hooks.EnterForward(first, input)?.Dispose();
            hooks.EnterForward(second, input)?.Dispose();
        }

        for (int i = 0; i < 3; i++)
        {
            Pass();                                                      // warm: storages found, order learned, scopes pooled
        }

        long stages = backend.Stages, prefetches = backend.Prefetches;
        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++)
        {
            Pass();
        }

        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Check(allocated == 0 && backend.Stages - stages == 200 && backend.Prefetches - prefetches == 200,
            $"100 passes of two layers: {allocated} bytes allocated, {backend.Stages - stages} stagings, {backend.Prefetches - prefetches} prefetches");
    }

    // Logits of a plain pass, a prompt of 4 positions and 3 cached steps; with truncate, the context cut back to the prompt
    // and the first step fed again (its logits must be the same).
    private static List<float[]> Decode(PretrainedModel model, bool truncate)
    {
        var device = model.Device;
        var results = new List<float[]>();
        using var context = new DecodingContext(device, 1, 32);
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            results.Add(model.Network.Forward(Tensor.From([1f, 5f, 9f, 3f], [1, 4], device)).ToArray());
            results.Add(model.Network.ForwardCached(Tensor.From([1f, 5f, 9f, 3f], [1, 4], device), context).ToArray());
            foreach (float id in new[] { 7f, 2f, 11f })
            {
                results.Add(model.Network.ForwardCached(Tensor.From([id], [1, 1], device), context).ToArray());
            }

            if (truncate)
            {
                context.Truncate(4);
                results.Add(model.Network.ForwardCached(Tensor.From([7f], [1, 1], device), context).ToArray());
                Check(results[^1].SequenceEqual(results[2]), $"{device}: the step fed again after a truncation gives its logits");
            }
        }

        return results;
    }
}
