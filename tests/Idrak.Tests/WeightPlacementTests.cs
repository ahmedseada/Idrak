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
// layers read their weights in place from the checkpoint's file mapping (WeightMappings, MappedWeights).
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
        ("lazy load: a lazy load builds no decoder layer and reads none of their weights until its first use (counted), then each on first use; the logits of a pass, a prompt and cached steps equal an eager load's bit for bit; disposed, it closes its checkpoint", LazyLoadMatchesEager),
        ("lazy load: on the CPU a lazy load's layers read their float32 weights in place from the checkpoint's file mapping (W3): every projection mapped, none copied, the logits of the copied weights bit for bit, the CPU's own memory lower by at least the mapped bytes (its counters); IDRAK_WEIGHT_MAPPING=copy copies them; disposed, the mapping is closed; on another device nothing is mapped", LazyLayersReadTheMapping),
        ("lazy load: a lazily loaded Q8_0 GGUF model's matrices are read in place from the file as kept (W3), each block matrix mapped or its copy counted with why, the loss of the eager load bit for bit", LazyGgufReadsTheMapping),
    ];

    private static readonly (string Name, Action<Device> Run)[] WeightPlacementGroup =
    [
        ("weight placement: device, split, stream and auto plan from the request (the layers the free memory holds beside the fixed weights, the working memory and each layer's cache; the user's count; stream refused where the device cannot and the model does not fit; auto keeps a model that fits on the device); the names are registered, an unknown one is not; device layers parse", PlacementPlans),
        ("weight placement: a decoder split between the device and the CPU gives the all-device and the all-CPU logits (a pass, a prompt and cached steps, a truncated context), its later layers' weights and caches on the CPU; a free memory that holds part of it splits it at load (auto) instead of failing; on the CPU nothing is split", SplitMatchesOneDevice),
        ("weight placement: layers follow the free memory at step boundaries (more on the device when it frees, fewer under pressure, a user's count kept), their weights and caches moved, the run going on with the all-CPU logits; FollowMemory off keeps them; lazy layers built where they then belong", LayersFollowMemory),
        ("weight placement: the weight mappings are registered (map, copy; an unknown name is not), map reads a split's CPU layer or a lazy layer in place on a device that reads a mapping in place and never a layer kept on the device, copy never; IDRAK_WEIGHT_MAPPING names the default; the counts' difference and line", WeightMappingsPlan),
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
        var device_ = WeightPlacements.Get("device").Place(Request(200));
        var split = WeightPlacements.Get("SPLIT").Place(Request(50 + 40 + 4 * 110));
        var asked = WeightPlacements.Get("split").Place(Request(50 + 40 + 4 * 110, asked: 7));
        Check(device_ is { DeviceLayers: 10, Stream: false, Split: false } && split is { DeviceLayers: 4, Layers: 10, Split: true } && asked is { DeviceLayers: 7, Split: true }
              && split.ToString().StartsWith("split: 4 of 10 layers on cpu, 6 computed by the CPU", StringComparison.Ordinal),
            $"device all; split as the memory holds ({split}); split as asked ({asked})");
        var streamed = WeightPlacements.Get("stream").Place(Request(300, stream: true));
        Check(streamed is { Stream: true, DeviceLayers: 10 } && WeightPlacements.Get("stream").Place(Request(5000)) is { Stream: false, DeviceLayers: 10 }, $"stream: {streamed}");
        bool refused = false;
        try
        {
            WeightPlacements.Get("stream").Place(Request(300));
        }
        catch (NotSupportedException)
        {
            refused = true;
        }

        Check(refused, "stream where the device cannot and the model does not fit: refused");
        Check(WeightPlacements.Get("auto").Place(Request(5000)) is { DeviceLayers: 10, Split: false, Stream: false }
              && WeightPlacements.Get("auto").Place(Request(300)) is { DeviceLayers: 1, Split: true }
              && WeightPlacements.Get("auto").Place(Request(5000, asked: 3)) is { DeviceLayers: 3, Split: true },
            "auto: the device when it fits, a split where the device cannot stream, the user's count when given");
        bool unknown = false;
        try
        {
            WeightPlacements.Get("nowhere");
        }
        catch (NotSupportedException e)
        {
            unknown = e.Message.Contains("not registered", StringComparison.Ordinal);
        }

        Check(unknown && WeightPlacements.Names.Order(StringComparer.Ordinal).SequenceEqual(["auto", "device", "split", "stream"]), "the names; an unknown one is not registered");
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

    private static void SplitMatchesOneDevice(Device device)
    {
        using var whole = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.AllOnDevice });
        if (device.Type == DeviceType.Cpu)
        {
            using var cpu = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Split, DeviceLayers = 1 });
            Check(cpu.Placement is null && cpu.Network.Descendants().OfType<DecoderBlock>().All(b => b.PlacedOn is null)
                  && Decode(cpu, truncate: true).Zip(Decode(whole, truncate: true)).All(p => p.First.SequenceEqual(p.Second)),
                "on the CPU nothing is split: the same logits");
            return;
        }

        using var host = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = Device.Cpu });
        using var split = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Split, DeviceLayers = 1 });
        var blocks = split.Network.Descendants().OfType<DecoderBlock>().ToList();
        Check(split.Placement is { Split: true, DeviceLayers: 1, Layers: 2 } && blocks[0].PlacedOn is null && blocks[1].PlacedOn == Device.Cpu
              && blocks[0].Parameters().All(p => p.Device == device) && blocks[1].Parameters().All(p => p.Device == Device.Cpu)
              && split.Notes.Any(n => n.StartsWith("placement split: 1 of 2 layers on", StringComparison.Ordinal)),
            $"layer 0 on {device}, layer 1 on the CPU ({split.Placement})");
        // W3: the CPU layer's projections read in place from the file's mapping, the device layer's in its memory.
        var cpuLinears = blocks[1].Descendants().OfType<Linear>().ToList();
        Check(cpuLinears.Count > 0 && cpuLinears.All(l => l.PackedWeight is StoredWeight { Mapping: not null })
              && blocks[0].Descendants().OfType<Linear>().All(l => l.PackedWeight is null)
              && split.WeightsMapped is { } mappedLine && mappedLine.StartsWith($"weights mapped: {cpuLinears.Count} matrices read in place", StringComparison.Ordinal),
            $"the CPU layer's {cpuLinears.Count} projections read in place: {split.WeightsMapped}");

        var expected = Decode(whole, truncate: true);
        var onCpu = Decode(host, truncate: true);
        var both = Decode(split, truncate: true);
        for (int i = 0; i < expected.Count; i++)
        {
            AssertClose(expected[i], both[i], 2e-3f, $"pass {i}: split against all on {device}");
            AssertClose(onCpu[i], both[i], 2e-3f, $"pass {i}: split against all on the CPU");
        }

        // The caches: layer 1's on the CPU (the context's counterpart there), layer 0's on the device.
        using (var context = new DecodingContext(device, 1, 32))
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            split.Network.ForwardCached(Tensor.From([1f, 5f, 9f], [1, 3], device), context);
            var mirror = context.On(Device.Cpu);
            Check(mirror.Device == Device.Cpu && mirror.Length == 3 && context.Length == 3 && mirror.CacheBytes > 0 && context.CacheBytes > 0 && !context.Recordable,
                $"the CPU layer's cache in system memory, in step ({mirror.Length} positions, {mirror.CacheBytes} bytes); the step is not recordable");
        }

        // Auto with a free memory that holds the fixed weights, the working memory and one layer: split at load.
        var spec = whole.Spec;
        int contextLength = Math.Min(AutoWeights.DefaultContext, spec.MaxPositions);
        var request = AutoWeights.PlacementRequest(spec, "float32", contextLength, device, null, null);
        long budget = request.FixedBytes + request.WorkingBytes + request.LayerBytes + request.LayerCacheBytes + request.LayerBytes / 2;
        using var auto = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, WeightPlacement = WeightPlacements.Auto, AutoWeightsBudget = budget });
        Check(auto.Placement is { Split: true, DeviceLayers: 1 } && auto.Network.Descendants().OfType<DecoderBlock>().Last().PlacedOn == Device.Cpu,
            $"a budget of one layer: auto splits ({auto.Placement})");
        var autoLogits = Decode(auto, truncate: false);
        AssertClose(expected[0], autoLogits[0], 2e-3f, "auto's split: the all-device logits");
    }

    private static void LazyLoadMatchesEager(Device device)
    {
        using var eager = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, LazyLayers = false });
        var expected = Decode(eager, truncate: true);
        using (var lazy = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, LazyLayers = true }))
        {
            Check(lazy.LayersBuilt == 0 && eager.LayersBuilt == 2 && lazy.Network.Children().Count(m => m.GetType().Name == "LazyLayer") == 2,
                $"nothing built at load: {lazy.LayersBuilt} layers built");
            var got = Decode(lazy, truncate: true);
            Check(lazy.LayersBuilt == 2 && lazy.Network.Children().All(m => m.GetType().Name != "LazyLayer"), $"both built on first use ({lazy.LayersBuilt})");
            for (int i = 0; i < expected.Count; i++)
            {
                Check(got[i].Select(BitConverter.SingleToInt32Bits).SequenceEqual(expected[i].Select(BitConverter.SingleToInt32Bits)), $"pass {i}: the eager logits bit for bit");
            }

            // A projection whose weights are read in place from the file (W3: lazy layers on the CPU) holds them as buffers.
            int inPlace = lazy.Network.Descendants().OfType<Linear>().Count(l => l.PackedWeight is StoredWeight { Mapping: not null });
            Check(lazy.Network.Parameters().Count() + inPlace == eager.Network.Parameters().Count(),
                $"the same weights once built ({lazy.Network.Parameters().Count()} parameters, {inPlace} read in place from the file; {eager.Network.Parameters().Count()} eager)");
        }

        // One layer at a time: a pass through the first layer only builds that one.
        using var partial = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, LazyLayers = true });
        using (Autograd.NoGrad())
        using (new TensorScope())
        {
            var blocksBefore = partial.LayersBuilt;
            using var context = new DecodingContext(device, 1, 16);
            partial.Network.ForwardCached(Tensor.From([1f, 5f], [1, 2], device), context, 2);   // embedding and the first layer
            Check(blocksBefore == 0 && partial.LayersBuilt == 1, $"a pass through one layer builds one ({partial.LayersBuilt})");
        }
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
                  && WeightPlacements.Get("device").Adjust(r, 3) == 3 && WeightPlacements.Get("split").Adjust(r, 3) == 5 && WeightPlacements.Get("auto").Adjust(r, 3) == 5,
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

    private sealed class StagingProvider : IDeviceProvider
    {
        public string Name => "staging-test";

        public string Summary => "a test device whose storages all count as offloaded";

        public DeviceType Type => DeviceType.Other;

        public int Count => 1;

        public Backend Create(int ordinal) => StagingBackend.Instance;

        public bool IsStarted(int ordinal) => false;

        public bool Listed(int ordinal) => false;

        public string? Note(int ordinal) => "test backend: by name only";
    }

    private static void StagingAllocatesNothing(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(StagingAllocatesNothing)))
        {
            return;
        }

        if (DeviceProviders.Find("staging-test") is null)
        {
            DeviceProviders.Register(new StagingProvider());
        }

        var staging = Device.Get("staging-test");
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

    private static void WeightMappingsPlan(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(WeightMappingsPlan)))
        {
            return;
        }

        var map = WeightMappings.Get("MAP");
        var copy = WeightMappings.Get(WeightMappings.Copy);
        var host = Device.Cpu;
        Check(map.Maps(new WeightMappingRequest(host, CpuLayer: true, Lazy: false)) && map.Maps(new WeightMappingRequest(host, CpuLayer: false, Lazy: true))
              && !map.Maps(new WeightMappingRequest(host, CpuLayer: false, Lazy: false)) && host.Backend is IMappedStorageBackend
              && !copy.Maps(new WeightMappingRequest(host, CpuLayer: true, Lazy: true)),
            "map: a split's CPU layer or a lazy layer on the CPU, not a layer kept where it is built; copy: never");
        bool unknown = false;
        try
        {
            WeightMappings.Get("nowhere");
        }
        catch (NotSupportedException e)
        {
            unknown = e.Message.Contains("not registered", StringComparison.Ordinal);
        }

        Check(unknown && WeightMappings.Names.Order(StringComparer.Ordinal).SequenceEqual(["copy", "map"]), "the names; an unknown one is not registered");
        WithEnvironment(new Dictionary<string, string?> { [WeightMappings.EnvironmentVariable] = "copy" }, () =>
            Check(WeightMappings.DefaultName == WeightMappings.Copy, $"IDRAK_WEIGHT_MAPPING=copy: {WeightMappings.DefaultName}"));
        WithEnvironment(new Dictionary<string, string?> { [WeightMappings.EnvironmentVariable] = null }, () =>
            Check(WeightMappings.DefaultName == WeightMappings.Map, "map by default"));

        // The counts: a snapshot's difference, the bytes read in place now, the copies by why.
        var before = MappedWeights.Read();
        MappedWeights.CountMapped(300);
        MappedWeights.CountReleased(100);
        MappedWeights.CountCopied(40, "a test's copy");
        var counted = MappedWeights.Read().Since(before);
        Check(counted is { Mapped: 1, MappedBytes: 300, Released: 1, ReleasedBytes: 100, LiveBytes: 200, Copies: 1, CopiedBytes: 40 } && counted.CopiedFor["a test's copy"] == 40
              && counted.ToString().StartsWith("weights mapped: 1 read in place from the file", StringComparison.Ordinal),
            $"the counts: {counted}");
    }

    private static void LazyLayersReadTheMapping(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            // Layers built on a device that does not read a mapping in place are read into its memory as before.
            var start = MappedWeights.Read();
            using var onDevice = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = device, LazyLayers = true });
            Decode(onDevice, truncate: false);
            Check(MappedWeights.Read().Since(start).Mapped == 0 && onDevice.Network.Descendants().OfType<Linear>().All(l => l.PackedWeight is null),
                $"{device}: lazy layers built on the device read nothing in place ({MappedWeights.Read().Since(start)})");
            return;
        }

        var cpu = Device.Cpu;
        long InUse()
        {
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();
            return cpu.Backend.GetMemoryUsage().InUse;
        }

        int filesBefore = FileMapping.MappedFiles;
        // Each path run once first, so what the process makes once on a kernel's first use (and keeps) is in neither measure.
        foreach (var warm in new[] { WeightMappings.Copy, WeightMappings.Map })
        {
            using var once = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = cpu, LazyLayers = true, WeightMapping = warm });
            Decode(once, truncate: true);
        }

        // The same lazy load with its weights copied, then read in place: the CPU memory each holds once its layers are built.
        List<float[]> copiedLogits, mappedLogits;
        long copiedGrowth, mappedGrowth, projectionBytes;
        int projections;
        MappedWeightCounts copiedCounts, mappedCounts;
        long start0 = InUse();
        var counts0 = MappedWeights.Read();
        using (var copied = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = cpu, LazyLayers = true, WeightMapping = WeightMappings.Copy }))
        {
            copiedLogits = Decode(copied, truncate: true);
            copiedGrowth = InUse() - start0;
            copiedCounts = MappedWeights.Read().Since(counts0);
            Check(copied.Network.Descendants().OfType<DecoderBlock>().SelectMany(b => b.Descendants().OfType<Linear>()).All(l => l.PackedWeight is null),
                "copy: every projection's weights in the CPU's memory");
        }

        long start1 = InUse();
        var counts1 = MappedWeights.Read();
        using (var mapped = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = cpu, LazyLayers = true, WeightMapping = WeightMappings.Map }))
        {
            Check(MappedWeights.Read().Since(counts1).Mapped == 0, "nothing mapped before a layer is built");
            mappedLogits = Decode(mapped, truncate: true);
            mappedGrowth = InUse() - start1;
            mappedCounts = MappedWeights.Read().Since(counts1);
            var linears = mapped.Network.Descendants().OfType<DecoderBlock>().SelectMany(b => b.Descendants().OfType<Linear>()).ToList();
            projections = linears.Count;
            projectionBytes = linears.Sum(l => (long)l.InFeatures * l.OutFeatures * sizeof(float));
            Check(projections > 0 && linears.All(l => l.PackedWeight is StoredWeight { Mapping: not null } w && ReferenceEquals(w.Type, ElementType.Float32)),
                $"map: every projection reads its float32 weights in place ({string.Join(", ", linears.Select(l => l.ToString()))})");
            Check(FileMapping.MappedFiles >= Math.Max(1, filesBefore), $"the checkpoint is mapped ({FileMapping.MappedFiles} files)");
        }

        Check(copiedCounts.Mapped == 0 && copiedCounts.Copies == 0, $"copy: nothing mapped, nothing counted ({copiedCounts})");
        Check(mappedCounts.Mapped == projections && mappedCounts.MappedBytes == projectionBytes && mappedCounts.Copies == 0,
            $"map: {projections} projections of {projectionBytes} bytes read in place, no copy ({mappedCounts})");
        for (int i = 0; i < copiedLogits.Count; i++)
        {
            Check(mappedLogits[i].Select(BitConverter.SingleToInt32Bits).SequenceEqual(copiedLogits[i].Select(BitConverter.SingleToInt32Bits)),
                $"pass {i}: the logits of the weights read in place are the copied weights' bit for bit");
        }

        // The CPU's own memory (its accountant: what the process allocated for tensors) holds the mapped weights nowhere.
        Check(copiedGrowth - mappedGrowth >= projectionBytes,
            $"the CPU's memory in use grew by {copiedGrowth} bytes with the weights copied and {mappedGrowth} read in place: lower by {copiedGrowth - mappedGrowth}, expected at least the {projectionBytes} mapped");
        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    measured: CPU memory in use after the lazy layers are built: {copiedGrowth:N0} B copied, {mappedGrowth:N0} B read in place, {copiedGrowth - mappedGrowth:N0} B lower ({projections} projections, {projectionBytes:N0} B mapped)"));

        InUse();                                                         // the disposed model's tensors finalized, if any were left
        var released = MappedWeights.Read().Since(counts1);
        Check(released.ReleasedBytes == projectionBytes && released.LiveBytes == 0 && FileMapping.MappedFiles == filesBefore,
            $"disposed: every mapping closed ({released}; {FileMapping.MappedFiles} files mapped, {filesBefore} before)");

        // IDRAK_WEIGHT_MAPPING=copy: a load without the option copies.
        WithEnvironment(new Dictionary<string, string?> { [WeightMappings.EnvironmentVariable] = "copy" }, () =>
        {
            var before = MappedWeights.Read();
            using var byVariable = PretrainedModel.Load(PlacementModel, new PretrainedOptions { Device = cpu, LazyLayers = true });
            Decode(byVariable, truncate: false);
            Check(MappedWeights.Read().Since(before).Mapped == 0, $"IDRAK_WEIGHT_MAPPING=copy: nothing read in place ({MappedWeights.Read().Since(before)})");
        });
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
