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
        ("moe placement: auto keeps every expert resident when the model fits the measured free memory and offloads when it does not, the device's experts sized by the free memory or IDRAK_MOE_DEVICE_EXPERTS_MB; IDRAK_MOE_COLD takes copy or cpu", MoePlacementPlans),
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

    internal static void MoePlacementPlans(Device device)
    {
        _ = device;
        // 2 layers of 8 experts of 1 MB beside 10 MB of the rest: 26 MB in all.
        const long Mb = 1L << 20;
        ExpertPlacementRequest Request(long? free, long? cache = null, bool cpu = false) => new(Device.Cpu, 2, 8, 2, Mb, 10 * Mb, free, cache, cpu);
        var fits = new ExpertPlacements.AutoPlacement().Place(Request(26 * Mb));
        var tight = new ExpertPlacements.AutoPlacement().Place(Request(15 * Mb));
        var limited = new ExpertPlacements.AutoPlacement().Place(Request(26 * Mb, cache: 4 * Mb, cpu: true));
        var none = new ExpertPlacements.OffloadPlacement().Place(Request(5 * Mb));
        Check(fits is { Offload: false, DeviceExperts: 8 }, $"auto, the model fits: {fits}");
        Check(tight is { Offload: true, DeviceExperts: 2, ColdOnCpu: false }, $"auto, 5 MB for 16 MB of experts: 2 of each layer's 8 on the device ({tight})");
        Check(limited is { Offload: true, DeviceExperts: 2, ColdOnCpu: true }, $"a limit of 4 MB: 2 a layer, cold ones on the CPU ({limited})");
        Check(none is { Offload: true, DeviceExperts: 1 }, $"no room: still one a layer ({none})");
        Check(new ExpertPlacements.ResidentPlacement().Place(Request(1 * Mb)) is { Offload: false, DeviceExperts: 8 }, "resident keeps every expert on the device");

        // Each placement states its own name; a person's text is matched against them where it comes in, and with none the
        // default (auto, or IDRAK_MOE_EXPERTS's) is reached directly.
        Check(ExpertPlacements.All.Select(p => p.Name).SequenceEqual([ExpertPlacements.Resident, ExpertPlacements.Offload, ExpertPlacements.Auto])
              && ExpertPlacements.Named(" Offload ", "--moe-experts") is ExpertPlacements.OffloadPlacement
              && (Environment.GetEnvironmentVariable(ExpertPlacements.PlacementVariable) is { Length: > 0 } || ExpertPlacements.Default is ExpertPlacements.AutoPlacement),
            "the placements by their own names, auto by default");
        ThrowsNaming<ArgumentException>(() => ExpertPlacements.Named("gpu", "--moe-experts"), "an expert placement no placement names", "resident, offload, auto");
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
}
