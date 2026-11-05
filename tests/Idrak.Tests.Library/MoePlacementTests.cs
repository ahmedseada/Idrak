// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

// Plan 37, R1: a mixture-of-experts model's experts placed apart (ExpertPlacements, MixtureOfExperts.Cache).
// - On the CPU, the tests run on a test device: plain arrays with a memory limit (ComputeResources.GpuMemoryLimit), its
//   operations through the host fallback, the routing operations as kernels of its own. The CPU is the host.
// - On a GPU (Vulkan's software driver here), the device itself under the same limit.
// The limit is set from what the device holds, never a size of a card.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] MoePlacementGroup =
    [
        ("moe placement: ExpertPlacements registers resident, offload and auto; auto keeps every expert resident when the model fits the measured free memory and offloads when it does not, the device's experts sized by the free memory or IDRAK_MOE_DEVICE_EXPERTS_MB; IDRAK_MOE_COLD takes copy or cpu; an unknown placement is not registered", MoePlacementRegistry),
        ("moe placement: a layer whose experts are placed apart (caches of 1, 2 and every expert; cold experts copied in or computed by the CPU; routed on the device and on the host) gives the resident layer's results bit for bit (a GPU's cold rows computed by the CPU within 1e-5: the CPU's activation), keeps at most its capacity on the device, keeps the most picked, and routes on the device once every expert is resident", MoePlacementLayer),
        ("moe placement: under a device memory limit below every expert's bytes, the layer with its experts placed apart runs and holds less device memory, the time per token measured cold and warm", MoePlacementLimit),
        ("moe placement: a checkpoint loaded with the experts offloaded (each layer's cache) or lazily (each expert read the first time the router picks it, those never picked never read) predicts the whole load's logits bit for bit", MoePlacementLoad),
    ];

    // A device with a memory limit (ComputeResources.GpuMemoryLimit), every operation through the host fallback and the
    // routing operations as kernels of its own (so a mixture's experts go through ExpertGatedPair and ExpertMatMul).
    internal sealed class PlacementBackend : Backend, IMemoryReport
    {
        public static readonly PlacementBackend Instance = new();

        private readonly MemoryAccountant _memory = new(() => ComputeResources.GpuMemoryLimit, "moe-placement-test:0");

        public override BackendCapabilities Capabilities { get; } = CpuBackend.Instance.Capabilities;

        public override string Kind => "moe-placement-test";

        public override string Name => "expert placement test (memory with a limit, routing kernels of its own)";

        long? IMemoryReport.AvailableMemory() => GetMemoryUsage().Limit is { } limit ? Math.Max(0, limit - GetMemoryUsage().InUse) : CpuBackend.Instance.AvailableMemory();

        long? IMemoryReport.FreeMemory() => null;

        public override Storage Allocate(int length, bool zeroed)
        {
            _memory.MustReleaseCacheFor(length * 4L);                     // throws over the limit
            _memory.Allocated(length * 4L);
            return new MinimalStorage(this, new float[length], length);
        }

        public override void Return(Storage storage)
        {
            _memory.Returned(storage.Length * 4L);
            _memory.Freed(storage.Length * 4L);
        }

        protected override void Detach(Storage storage) => ((MinimalStorage)storage).Data = null!;

        protected override void Attach(Storage storage, Storage fresh) => ((MinimalStorage)storage).Data = ((MinimalStorage)fresh).Data;

        public override MemoryUsage GetMemoryUsage() => _memory.Usage;

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

        public override void RouteTopKKernel(Storage probabilities, Storage weights, Storage offsets, Storage tokenOfRow, Storage slotRow, int tokens, int experts, int k,
            bool normalize) => base.RouteTopKKernel(probabilities, weights, offsets, tokenOfRow, slotRow, tokens, experts, k, normalize);

        public override void ExpertGatedPairKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Gate, Storage? GateBias, Storage Up, Storage? UpBias)> experts,
            Storage hidden, int rows, int k, int n, int activation) => base.ExpertGatedPairKernel(x, offsets, experts, hidden, rows, k, n, activation);

        public override void ExpertMatMulKernel(Storage x, Storage offsets, ReadOnlySpan<(Storage Weight, Storage? Bias)> experts, Storage y, int rows, int k, int n) =>
            base.ExpertMatMulKernel(x, offsets, experts, y, rows, k, n);
    }

    internal sealed class PlacementProvider : IDeviceProvider
    {
        public string Name => "moe-placement-test";

        public string Summary => "a test device with a memory limit and the mixture-of-experts routing as kernels of its own";

        public DeviceType Type => DeviceType.Other;

        public int Count => 1;

        public Backend Create(int ordinal) => PlacementBackend.Instance;

        public bool IsStarted(int ordinal) => false;

        public bool Listed(int ordinal) => false;

        public string? Note(int ordinal) => "test backend: by name only";
    }

    // The device the placement tests run on: a GPU itself, or the test device beside the CPU.
    private static Device PlacementDevice(Device device)
    {
        if (device.Type != DeviceType.Cpu)
        {
            return device;
        }

        if (DeviceProviders.Find("moe-placement-test") is null)
        {
            DeviceProviders.Register(new PlacementProvider());
        }

        return Device.Get("moe-placement-test");
    }

    internal static void MoePlacementRegistry(Device device)
    {
        _ = device;
        Check(ExpertPlacements.Names.SequenceEqual([ExpertPlacements.Resident, ExpertPlacements.Offload, ExpertPlacements.Auto])
              && ExpertPlacements.Origin("auto") == Overrides.Library && ExpertPlacements.Get("OFFLOAD").Name == "offload",
            $"the library's placements: {string.Join(", ", ExpertPlacements.Names)}");
        try
        {
            ExpertPlacements.Get("everywhere");
            Check(false, "an unknown placement should be refused");
        }
        catch (NotRegisteredException ex)
        {
            Check(ex.Message.Contains("not registered", StringComparison.Ordinal) && ex.Message.Contains("offload", StringComparison.Ordinal), ex.Message);
        }

        // 2 layers of 8 experts of 1 MB beside 10 MB of the rest: 26 MB in all.
        const long Mb = 1L << 20;
        ExpertPlacementRequest Request(long? free, long? cache = null, bool cpu = false) => new(Device.Cpu, 2, 8, 2, Mb, 10 * Mb, free, cache, cpu);
        var fits = ExpertPlacements.Get("auto").Place(Request(26 * Mb));
        var tight = ExpertPlacements.Get("auto").Place(Request(15 * Mb));
        var limited = ExpertPlacements.Get("auto").Place(Request(26 * Mb, cache: 4 * Mb, cpu: true));
        var none = ExpertPlacements.Get("offload").Place(Request(5 * Mb));
        Check(fits is { Offload: false, DeviceExperts: 8 }, $"auto, the model fits: {fits}");
        Check(tight is { Offload: true, DeviceExperts: 2, ColdOnCpu: false }, $"auto, 5 MB for 16 MB of experts: 2 of each layer's 8 on the device ({tight})");
        Check(limited is { Offload: true, DeviceExperts: 2, ColdOnCpu: true }, $"a limit of 4 MB: 2 a layer, cold ones on the CPU ({limited})");
        Check(none is { Offload: true, DeviceExperts: 1 }, $"no room: still one a layer ({none})");
        Check(ExpertPlacements.Get("resident").Place(Request(1 * Mb)) is { Offload: false, DeviceExperts: 8 }, "resident keeps every expert on the device");
        Check(ExpertPlacements.ParseCacheMegabytes("64", "x") == 64 * Mb && ExpertPlacements.ParseCacheMegabytes(" ", "x") is null
              && !ExpertPlacements.ParseCold(null, "x") && !ExpertPlacements.ParseCold("copy", "x") && ExpertPlacements.ParseCold("CPU", "x"), "the settings' values");
        foreach (var (bad, parse) in new (string, Action)[] { ("-3", () => ExpertPlacements.ParseCacheMegabytes("-3", "IDRAK_MOE_DEVICE_EXPERTS_MB")), ("gpu", () => ExpertPlacements.ParseCold("gpu", "IDRAK_MOE_COLD")) })
        {
            try
            {
                parse();
                Check(false, $"'{bad}' should be refused");
            }
            catch (ArgumentException ex)
            {
                Check(ex.Message.Contains(bad, StringComparison.Ordinal), ex.Message);
            }
        }
    }

    // A layer of `experts` experts, made from `seed` on `device` (its biases drawn too where it has them).
    private static MixtureOfExperts PlacementLayer(Device device, int dim, int hidden, int experts, int k, FeedForwardActivation activation, bool bias, int seed)
    {
        var moe = new MixtureOfExperts(dim, hidden, experts, k, normalizeTopK: true, sharedHidden: 0, activation, bias, device, new Random(seed));
        if (bias)
        {
            var random = new Random(seed + 1);
            foreach (var layer in moe.Experts.SelectMany(e => new[] { e.Gate!, e.Up, e.Down }))
            {
                layer.Bias!.CopyFrom([.. Enumerable.Range(0, layer.OutFeatures).Select(_ => random.NextSingle() - 0.5f)]);
            }
        }

        return moe;
    }

    // The layer's output for `input` [1, tokens, dim] (inference), routed on the device or not.
    private static float[] PlacementRun(MixtureOfExperts moe, Tensor input, bool onDevice)
    {
        bool before = MixtureOfExperts.RoutesOnDevice;
        MixtureOfExperts.RoutesOnDevice = onDevice;
        try
        {
            using var scope = new TensorScope();
            using (Autograd.NoGrad())
            {
                return moe.Forward(input).ToArray();
            }
        }
        finally
        {
            MixtureOfExperts.RoutesOnDevice = before;
        }
    }

    private static int[] PlacementBits(float[] values) => [.. values.Select(BitConverter.SingleToInt32Bits)];

    internal static void MoePlacementLayer(Device device)
    {
        var target = PlacementDevice(device);
        var backend = target.Backend;
        const int Dim = 24, Hidden = 40, Experts = 9, K = 2, Tokens = 30;
        var random = new Random(61);
        using var input = Tensor.From([.. Enumerable.Range(0, Tokens * Dim).Select(_ => random.NextSingle() * 2 - 1)], [1, Tokens, Dim], target);
        foreach (var (activation, bias) in new[] { (FeedForwardActivation.Silu, false), (FeedForwardActivation.Gelu, true) })
        {
            using var resident = PlacementLayer(target, Dim, Hidden, Experts, K, activation, bias, 62);
            var expected = PlacementRun(resident, input, onDevice: true);
            foreach (int capacity in (int[])[1, 2, Experts])
            {
                foreach (bool cold in (bool[])[false, true])
                {
                    foreach (bool onDevice in (bool[])[true, false])
                    {
                        string what = $"{target}, {activation}{(bias ? " with biases" : "")}, a cache of {capacity}, cold experts {(cold ? "on the CPU" : "copied in")}, {(onDevice ? "device" : "host")} routing";
                        using var moe = PlacementLayer(target, Dim, Hidden, Experts, K, activation, bias, 62);
                        var cache = moe.OffloadExperts(target, capacity, cold);
                        Check(cache.Resident == 0 && cache.Capacity == capacity && moe.Experts.All(e => e.WeightsDevice == Device.Cpu), $"{what}: every expert moved to the host");
                        var first = PlacementRun(moe, input, onDevice);                // cold: experts copied in
                        var second = PlacementRun(moe, input, onDevice);               // warm
                        // Bit for bit wherever the experts compute on the device (copied in), and on the test device (whose
                        // operations are the CPU's); a GPU's cold rows computed by the CPU take the CPU's activation, within
                        // the device-against-CPU tolerance.
                        bool exact = !cold || cache.CpuRows == 0 || device.Type == DeviceType.Cpu;
                        float worst = first.Zip(expected, (a, b) => MathF.Abs(a - b) / MathF.Max(1f, MathF.Abs(b))).Concat(second.Zip(expected, (a, b) => MathF.Abs(a - b) / MathF.Max(1f, MathF.Abs(b)))).Max();
                        Check(exact ? PlacementBits(first).SequenceEqual(PlacementBits(expected)) && PlacementBits(second).SequenceEqual(PlacementBits(expected)) : worst <= 1e-5f,
                            $"{what}: the resident layer's values {(exact ? "bit for bit" : "within 1e-5")} (largest difference {worst})");
                        Check(cache.Resident <= capacity && moe.Experts.Count(e => e.WeightsDevice == target) == cache.Resident,
                            $"{what}: {cache.Resident} experts on the device, at most {capacity} ({cache})");
                        Check(cold ? cache.CpuRows > 0 || cache.Resident == capacity : cache.Misses > 0 && cache.CpuRows == 0, $"{what}: {cache}");

                        // The most picked stay: every resident expert is picked at least as often as every cold one.
                        int[] residentOnes = [.. Enumerable.Range(0, Experts).Where(cache.IsResident)];
                        int[] coldOnes = [.. Enumerable.Range(0, Experts).Where(e => !cache.IsResident(e) && cache.Uses(e) > 0)];
                        if (capacity < Experts && !cold && residentOnes.Length > 0 && coldOnes.Length > 0)
                        {
                            Check(residentOnes.Min(cache.Uses) >= coldOnes.Max(cache.Uses) || residentOnes.Length == capacity,
                                $"{what}: the resident experts are the most picked ({string.Join(", ", Enumerable.Range(0, Experts).Select(e => $"{e}:{cache.Uses(e)}{(cache.IsResident(e) ? "*" : "")}"))})");
                        }

                        if (capacity == Experts && !cold)
                        {
                            // Every expert picked once is resident: the layer routes on the device again, and a step can be recorded.
                            Check(cache.HoldsAll == (cache.Resident == Experts), $"{what}: {cache}");
                            if (cache.HoldsAll && onDevice)
                            {
                                using var trace = Kernels.Trace(backend);
                                var third = PlacementRun(moe, input, onDevice: true);
                                Check(trace.Calls(Ops.RouteTopK) == 1 && !moe.RoundTrips(backend) && PlacementBits(third).SequenceEqual(PlacementBits(expected)),
                                    $"{what}: every expert resident, routed on the device ({trace.Calls(Ops.RouteTopK)} RouteTopK, round trips {moe.RoundTrips(backend)})");
                            }
                        }
                        else
                        {
                            Check(moe.RoundTrips(backend) == !cache.HoldsAll,
                                $"{what}: the step reads its routing back (not recorded) exactly while not every expert is resident ({cache}; round trips {moe.RoundTrips(backend)})");
                        }
                    }
                }
            }
        }
    }

    internal static void MoePlacementLimit(Device device)
    {
        var target = PlacementDevice(device);
        var backend = target.Backend;
        const int Dim = 48, Hidden = 96, Experts = 16, K = 2, Tokens = 6, Capacity = 3;
        var random = new Random(71);
        float[] values = [.. Enumerable.Range(0, Tokens * Dim).Select(_ => random.NextSingle() * 2 - 1)];
        long? limitBefore = ComputeResources.GpuMemoryLimit;
        try
        {
            ComputeResources.GpuMemoryLimit = null;
            float[] expected;
            long residentBytes;
            using (var resident = PlacementLayer(target, Dim, Hidden, Experts, K, FeedForwardActivation.Silu, false, 72))
            using (var input = Tensor.From(values, [1, Tokens, Dim], target))
            {
                expected = PlacementRun(resident, input, onDevice: true);
                residentBytes = resident.Parameters().Concat(resident.Buffers()).Where(t => t.Device == target).Sum(t => (long)t.Storage.Length * sizeof(float));
            }

            backend.ReleaseCachedMemory();
            using var moe = PlacementLayer(target, Dim, Hidden, Experts, K, FeedForwardActivation.Silu, false, 72);
            moe.OffloadExperts(target, Capacity);
            backend.ReleaseCachedMemory();
            long expertBytes = ExpertCache.Bytes(moe.Experts[0]);
            // Room for the cache, a step's activations and the input: less than the experts not in it.
            long room = Capacity * expertBytes + 64L * Tokens * (Dim + Hidden) * K * sizeof(float);
            Check(room < (Experts - Capacity) * expertBytes, $"the test's room ({room} bytes) must be below the cold experts' ({(Experts - Capacity) * expertBytes})");
            ComputeResources.GpuMemoryLimit = backend.GetMemoryUsage().Reserved + room;
            Check(Experts * expertBytes > room, $"every expert ({Experts * expertBytes} bytes) exceeds the room the limit leaves ({room} bytes): a resident layer would not fit");

            using var input2 = Tensor.From(values, [1, Tokens, Dim], target);
            long start = Stopwatch.GetTimestamp();
            var first = PlacementRun(moe, input2, onDevice: true);
            double coldMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            start = Stopwatch.GetTimestamp();
            var second = PlacementRun(moe, input2, onDevice: true);
            double warmMs = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
            var cache = moe.Cache!;
            long deviceBytes = moe.Parameters().Concat(moe.Buffers()).Where(t => t.Device == target).Sum(t => (long)t.Storage.Length * sizeof(float));
            Check(PlacementBits(first).SequenceEqual(PlacementBits(expected)) && PlacementBits(second).SequenceEqual(PlacementBits(expected)),
                "under the limit, the resident layer's values bit for bit");
            Check(cache.Resident <= Capacity && moe.Experts.Where(e => e.WeightsDevice == target).Sum(ExpertCache.Bytes) < residentBytes,
                $"the device holds {cache.Resident} experts ({moe.Experts.Where(e => e.WeightsDevice == target).Sum(ExpertCache.Bytes)} bytes), the resident layer {residentBytes}");
            Console.WriteLine(string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"    {target}: {Experts} experts of {expertBytes} bytes, {Capacity} on the device under a limit of {ComputeResources.GpuMemoryLimit} bytes; resident layer {residentBytes} bytes, placed apart {deviceBytes} bytes; {Tokens} tokens: cold {coldMs:0.##} ms ({coldMs / Tokens:0.###} ms a token), warm {warmMs:0.##} ms ({warmMs / Tokens:0.###} ms a token); {cache}"));
        }
        finally
        {
            ComputeResources.GpuMemoryLimit = limitBefore;
        }
    }

    internal static void MoePlacementLoad(Device device)
    {
        var target = PlacementDevice(device);
        var spec = SmallSpec with { Experts = 12, ExpertsPerToken = 2 };
        int[] ids = [2, 9, 14];
        string folder = TempFolder();
        try
        {
            spec = PretrainedArchitectures.Get("MixtralForCausalLM").Spec(ExpertConfig("MixtralForCausalLM", spec), []);
            WriteExpertCheckpoint(folder, "MixtralForCausalLM", spec, new RandomWeights(81));
            float[] Predict(PretrainedOptions options, out PretrainedModel model)
            {
                model = PretrainedModel.Load(folder, options with { Device = target });
                using var input = Tensor.From([.. ids.Select(i => (float)i)], [1, ids.Length], target);
                using var scope = new TensorScope();
                using (Autograd.NoGrad())
                {
                    return model.Network.Predict(input).ToArray();
                }
            }

            var whole = Predict(new PretrainedOptions(), out var wholeModel);
            wholeModel.Dispose();
            long expertBytes = (long)spec.Dim * spec.ExpertHidden * 3 * sizeof(float);
            var offloaded = Predict(new PretrainedOptions { ExpertPlacement = "offload", DeviceExpertBytes = 2 * expertBytes * spec.Layers }, out var offloadedModel);
            using (offloadedModel)
            {
                var caches = offloadedModel.Network.Descendants().OfType<MixtureOfExperts>().Select(m => m.Cache).ToList();
                Check(offloadedModel.ExpertPlacement is { Offload: true, DeviceExperts: 2 } && caches.Count == spec.Layers && caches.All(c => c is { Capacity: 2 } && c.Resident <= 2),
                    $"offload: {offloadedModel.ExpertPlacement}; caches {string.Join("; ", caches)}");
                Check(PlacementBits(offloaded).SequenceEqual(PlacementBits(whole)), "offload: the whole load's logits bit for bit");
            }

            var lazy = Predict(new PretrainedOptions { LazyLayers = true }, out var lazyModel);
            using (lazyModel)
            {
                var caches = lazyModel.Network.Descendants().OfType<MixtureOfExperts>().Select(m => m.Cache).ToList();
                Check(caches.Count == spec.Layers && caches.All(c => c is { Lazy: true } && c.Built <= ids.Length * spec.ExpertsPerToken + 1 && c.Built < spec.Experts),
                    $"lazy: each layer read only the experts picked ({string.Join("; ", caches)})");
                Check(PlacementBits(lazy).SequenceEqual(PlacementBits(whole)), "lazy: the whole load's logits bit for bit");
            }
        }
        finally
        {
            Directory.Delete(folder, true);
        }
    }
}
