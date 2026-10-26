// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Abstraction.Operations;
using Idrak.Gpu.Vulkan;

// Plan 34, phase F: what the first runs on a phone and a tablet showed. The CPU's memory is the system's where the
// runtime's figure is its heap budget (Mono), held to the runtime's heap limit where it has one (CoreCLR); portable
// 128-bit vectors run where the runtime reports no NEON; the cores are named by their groups; a kernel the driver refuses
// falls back (the same kernel without its subgroup-size request, its next variant, its operation's fallback), never
// taking the device down, and is asked for once per process.
internal abstract partial class Tests
{
    // Writes `text` at root/relative, making its folders.
    private static void WriteAt(string root, string relative, string text)
    {
        string path = Path.Combine(root, relative);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        File.WriteAllText(path, text);
    }

    internal static void CpuSystemMemory(Device device)
    {
        if (device.Backend is not CpuBackend)
        {
            return;
        }

        string root = Path.Combine(Path.GetTempPath(), $"idrak-memory-{Environment.ProcessId}");
        try
        {
            // /proc/meminfo, as bytes.
            var (total, available) = SystemMemory.ParseMeminfo("MemTotal:       16477348 kB\nMemFree:         5673032 kB\nMemAvailable:   12563452 kB\n"u8);
            Check(total == 16477348L << 10 && available == 12563452L << 10, $"MemTotal and MemAvailable in bytes: {total}, {available}");
            Check(SystemMemory.ParseMeminfo("MemFree: 1 kB\n"u8) == (null, null), "no MemTotal or MemAvailable: none");
            Check(SystemMemory.ParseBytes("max\n"u8) is null && SystemMemory.ParseBytes("14341881856\n"u8) == 14341881856 && SystemMemory.ParseBytes("12x"u8) is null,
                "a group's number file: max is no limit");
            Check(SystemMemory.StatValue("cache 1\ninactive_file 3399880704\ntotal_inactive_file 7\n"u8, "inactive_file"u8) == 3399880704
                  && SystemMemory.StatValue("total_inactive_file 7\n"u8, "inactive_file"u8) is null, "memory.stat values by their whole key");

            // cgroup v1 (a container): the leaf's limit, lower than its parent's, the root unlimited; usage less inactive file.
            WriteAt(root, "proc/meminfo", "MemTotal:       16477348 kB\nMemAvailable:   12563452 kB\n");
            WriteAt(root, "proc/self/cgroup", "9:name=systemd:/\n4:memory:/outer/inner\n1:cpu:/\n0::/\n");
            WriteAt(root, "proc/self/mountinfo", "37 24 0:29 / /sys/fs/cgroup rw,relatime - tmpfs tmpfs rw,mode=755\n"
                + "38 37 0:30 / /sys/fs/cgroup/cpu rw,relatime - cgroup cgroup rw,cpu\n"
                + "41 37 0:33 / /sys/fs/cgroup/memory rw,relatime - cgroup cgroup rw,memory\n"
                + "47 37 0:39 / /sys/fs/cgroup/unified rw,relatime - cgroup2 cgroup2 rw\n");
            WriteAt(root, "sys/fs/cgroup/memory/memory.limit_in_bytes", "9223372036854771712\n");
            WriteAt(root, "sys/fs/cgroup/memory/outer/memory.limit_in_bytes", "20000000000\n");
            WriteAt(root, "sys/fs/cgroup/memory/outer/inner/memory.limit_in_bytes", "14341881856\n");
            WriteAt(root, "sys/fs/cgroup/memory/outer/inner/memory.usage_in_bytes", "8582230016\n");
            WriteAt(root, "sys/fs/cgroup/memory/outer/inner/memory.stat", "cache 1\ninactive_file 3399880704\ntotal_inactive_file 3399880704\n");
            var v1 = SystemMemory.Read(root);
            Check(v1.Total == 16477348L << 10 && v1.Available == 12563452L << 10, $"v1: the machine's memory ({v1})");
            Check(v1.Limit == 14341881856 && v1.LimitUsage == 8582230016 - 3399880704 && v1.LimitSource == "cgroup v1", $"v1: the lowest limit and its usage ({v1})");
            Check(v1.ProcessTotal == 14341881856 && v1.ProcessAvailable == 14341881856 - (8582230016 - 3399880704), $"v1: what the process may use ({v1})");

            // cgroup v2: the parent's limit binds (the leaf's is max), its usage less inactive file.
            Directory.Delete(root, recursive: true);
            WriteAt(root, "proc/meminfo", "MemTotal:        8000000 kB\nMemAvailable:    6000000 kB\n");
            WriteAt(root, "proc/self/cgroup", "0::/app/leaf\n");
            WriteAt(root, "proc/self/mountinfo", "30 1 0:26 / /sys/fs/cgroup rw,nosuid shared:4 - cgroup2 cgroup2 rw,nsdelegate\n");
            WriteAt(root, "sys/fs/cgroup/app/memory.max", "4294967296\n");
            WriteAt(root, "sys/fs/cgroup/app/memory.current", "1073741824\n");
            WriteAt(root, "sys/fs/cgroup/app/memory.stat", "anon 5\ninactive_file 73741824\n");
            WriteAt(root, "sys/fs/cgroup/app/leaf/memory.max", "max\n");
            var v2 = SystemMemory.Read(root);
            Check(v2.Limit == 4294967296 && v2.LimitUsage == 1_000_000_000 && v2.LimitSource == "cgroup v2", $"v2: the parent's limit ({v2})");
            Check(v2.ProcessTotal == 4294967296 && v2.ProcessAvailable == 4294967296 - 1_000_000_000, $"v2: what the process may use ({v2})");

            // A phone: /proc/meminfo, no group file readable.
            Directory.Delete(root, recursive: true);
            WriteAt(root, "proc/meminfo", "MemTotal:        7500000 kB\nMemAvailable:    3000000 kB\n");
            var phone = SystemMemory.Read(root);
            Check(phone.Limit is null && phone.ProcessTotal == 7500000L << 10 && phone.ProcessAvailable == 3000000L << 10, $"no groups: the machine's ({phone})");
            Directory.Delete(root, recursive: true);
            Check(SystemMemory.Read(root) == default, "nothing readable: nothing reported");
        }
        finally
        {
            if (Directory.Exists(root))
            {
                Directory.Delete(root, recursive: true);
            }
        }

        // The choice. Nothing from the system (Windows, macOS): the runtime's, as before.
        var none = CpuBackend.ChooseMemory(default, 10_000, 4_000, runtimeIsBudget: false);
        Check(none.Total == 10_000 && none.Available == 6_000, $"no system figures: the runtime's total and total less load ({none})");
        var monoNone = CpuBackend.ChooseMemory(default, 150, 10, runtimeIsBudget: true);
        Check(monoNone.Total == 150 && monoNone.Available == 140, "no system figures under Mono: still the runtime's (nothing else is known)");
        // Mono: its figure is the heap's budget, so the system's.
        var system = new SystemMemoryReport(8_000, 5_000, null, null, null);
        var mono = CpuBackend.ChooseMemory(system, 150, 10, runtimeIsBudget: true);
        Check(mono.Total == 8_000 && mono.Available == 5_000 && mono.RuntimeIsBudget && !mono.RuntimeCaps, $"Mono: the system's memory ({mono})");
        // CoreCLR with a heap limit (set, or implied by a container's): the limit stays; available the lower of both.
        var container = new SystemMemoryReport(16_000, 12_000, 13_000, 4_000, "cgroup v1");
        var capped = CpuBackend.ChooseMemory(container, 9_750, 1_000, runtimeIsBudget: false);
        Check(capped.Total == 9_750 && capped.Available == 8_750 && capped.RuntimeCaps, $"CoreCLR capped: its limit ({capped})");
        var busy = CpuBackend.ChooseMemory(container with { LimitUsage = 6_000 }, 9_750, 1_000, runtimeIsBudget: false);
        Check(busy.Total == 9_750 && busy.Available == 7_000, $"CoreCLR capped, the group nearly full: the group's room ({busy})");
        var plain = CpuBackend.ChooseMemory(new SystemMemoryReport(16_000, 12_000, null, null, null), 16_000, 2_000, runtimeIsBudget: false);
        Check(plain.Total == 16_000 && plain.Available == 12_000 && !plain.RuntimeCaps, $"CoreCLR without a limit: the machine's ({plain})");

        // This machine: on CoreCLR the total is the runtime's figure as before (rule 71), never above the system's.
        var gc = GC.GetGCMemoryInfo();
        var measured = CpuBackend.Memory;
        var hardware = Device.Cpu.Backend.Hardware;
        Console.WriteLine($"    the CPU's memory: {string.Join("; ", hardware.Facts!.Where(f => f.Key.Contains("memory", StringComparison.Ordinal)).Select(f => $"{f.Key}: {f.Value}"))}");
        Check(hardware.MemoryBytes == measured.Total, "the device's memory is the measured total");
        if (!CpuBackend.RuntimeIsMono)
        {
            Check(measured.Total == Math.Min(gc.TotalAvailableMemoryBytes, measured.System.ProcessTotal ?? long.MaxValue),
                $"CoreCLR: the runtime's figure ({gc.TotalAvailableMemoryBytes}) unless the system's is lower ({measured.Total})");
            Check(measured.System.ProcessTotal is null || measured.Total == gc.TotalAvailableMemoryBytes,
                "CoreCLR: the total as before (its figure is never above the machine's or the group's)");
        }

        if (OperatingSystem.IsLinux())
        {
            Check(measured.System.Total is > 0 && hardware.Facts!.Any(f => f.Key == "system memory") && hardware.Facts!.Any(f => f.Key == "runtime memory"),
                "Linux: the system's memory and the runtime's, labelled, in the facts");
        }

        Check(Device.Cpu.Backend.AvailableMemory() is { } free && free >= 0 && free <= measured.Total, "available within the total");
    }

    internal static void CpuCoreGroups(Device device)
    {
        if (device.Backend is not CpuBackend)
        {
            return;
        }

        // A phone's layout: four small cores, three large, one larger (one part number, two frequencies), and the system's name.
        var text = new System.Text.StringBuilder();
        string[] parts = ["0xd05", "0xd05", "0xd05", "0xd05", "0xd0d", "0xd0d", "0xd0d", "0xd0d"];
        long[] khz = [1_800_000, 1_800_000, 1_800_000, 1_800_000, 2_400_000, 2_400_000, 2_400_000, 2_840_000];
        for (int cpu = 0; cpu < 8; cpu++)
        {
            text.Append($"processor\t: {cpu}\nBogoMIPS\t: 38.40\nFeatures\t: fp asimd\nCPU implementer\t: 0x41\nCPU architecture: 8\nCPU variant\t: 0x1\nCPU part\t: {parts[cpu]}\nCPU revision\t: 0\n\n");
        }

        var noName = text.ToString();
        var (groups, model) = CpuInfo.ParseCores(System.Text.Encoding.UTF8.GetBytes(noName), cpu => khz[cpu]);
        Check(groups is [(4, "0x41:0xd05", 1_800_000), (3, "0x41:0xd0d", 2_400_000), (1, "0x41:0xd0d", 2_840_000)],
            $"three groups by part and frequency: {string.Join(", ", groups)}");
        Check(model == "8 cores: 4 × part 0x41:0xd05 up to 1.80 GHz, 3 × part 0x41:0xd0d up to 2.40 GHz, 1 × part 0x41:0xd0d up to 2.84 GHz",
            $"named by its groups: {model}");
        var (_, named) = CpuInfo.ParseCores(System.Text.Encoding.UTF8.GetBytes(noName + "Hardware\t: Example SoC 1\n"), cpu => khz[cpu]);
        Check(named.StartsWith("Example SoC 1 (8 cores: 4 × part", StringComparison.Ordinal), $"the system's name with the groups: {named}");
        var (unknownFrequency, _) = CpuInfo.ParseCores(System.Text.Encoding.UTF8.GetBytes(noName), _ => null);
        Check(unknownFrequency is [(4, "0x41:0xd05", null), (4, "0x41:0xd0d", null)], "no frequency reported: by part alone");

        // x86: the model name as before, the cores in one group.
        var x86 = "processor\t: 0\nvendor_id\t: Example\nmodel\t\t: 85\nmodel name\t: Example CPU @ 2.80GHz\n\nprocessor\t: 1\nvendor_id\t: Example\nmodel\t\t: 85\nmodel name\t: Example CPU @ 2.80GHz\n"u8;
        var (x86Groups, x86Model) = CpuInfo.ParseCores(x86, _ => null);
        Check(x86Model == "Example CPU @ 2.80GHz" && x86Groups is [(2, "model 85", null)], $"x86: {x86Model}, {string.Join(", ", x86Groups)}");

        // This machine: where /proc/cpuinfo has a model name, the name is the one read before (rule 71: the tuning key).
        if (OperatingSystem.IsLinux() && File.Exists("/proc/cpuinfo"))
        {
            string? before = File.ReadLines("/proc/cpuinfo").FirstOrDefault(l => l.StartsWith("model name", StringComparison.Ordinal));
            if (before is not null)
            {
                Check(CpuInfo.Current.Model == before[(before.IndexOf(':') + 1)..].Trim(), $"the model name as before: {CpuInfo.Current.Model}");
            }

            Console.WriteLine($"    this CPU: {CpuInfo.Current.Model}; {CpuInfo.DescribeGroups(CpuInfo.Current.CoreGroups)}");
        }
    }

    internal static void CpuVectorPathsChosen(Device device)
    {
        if (device.Backend is not CpuBackend)
        {
            return;
        }

        Check(ComputeResources.ParseCpuVectors(null) == CpuVectorPaths.Automatic && ComputeResources.ParseCpuVectors(" Portable ") == CpuVectorPaths.Portable
              && ComputeResources.ParseCpuVectors("native") == CpuVectorPaths.Native && ComputeResources.ParseCpuVectors("scalar") == CpuVectorPaths.Scalar,
            "IDRAK_CPU_VECTORS by name");
        bool threw = false;
        try
        {
            ComputeResources.ParseCpuVectors("fast");
        }
        catch (ArgumentException)
        {
            threw = true;
        }

        Check(threw, "another name is refused");

        // The description says what runs: Mono on ARM64 reports no AdvSimd but accelerates 128-bit vectors.
        Check(CpuInfo.VectorsReported(intrinsics: false, vector128Accelerated: true) == VectorSupport.Portable
              && CpuInfo.VectorsReported(true, true) == VectorSupport.Intrinsics && CpuInfo.VectorsReported(false, false) == VectorSupport.None, "what the runtime reports");
        Check(CpuInfo.DescribeSets([], VectorSupport.Portable, Architecture.Arm64) == "NEON (portable vectors: the runtime reports no NEON intrinsics)",
            "ARM64 under Mono: NEON through portable vectors, not \"no SIMD\"");
        Check(CpuInfo.DescribeSets([], VectorSupport.None, Architecture.Arm64) == "no SIMD" && CpuInfo.DescribeSets(["AVX2", "FMA"], VectorSupport.Intrinsics, Architecture.X64) == "AVX2 FMA",
            "no vectors: no SIMD; reported sets as before");

        // The tiled kernel per setting: native as before; portable the 128-bit kernel wherever 128-bit vectors are
        // accelerated; scalar none; automatic native's, or the portable kernel on ARM64 without reported intrinsics.
        var native = Vector512.IsHardwareAccelerated && Avx512F.IsSupported && Fma.IsSupported ? TiledKernel.Avx512
            : Vector256.IsHardwareAccelerated && Avx2.IsSupported && Fma.IsSupported ? TiledKernel.Avx2
            : AdvSimd.Arm64.IsSupported && Vector128.IsHardwareAccelerated ? TiledKernel.Neon : TiledKernel.None;
        Check(CpuTuning.KernelFor(CpuVectorPaths.Native) == native, $"native: the former selection ({native})");
        Check(CpuTuning.KernelFor(CpuVectorPaths.Portable) == (Vector128.IsHardwareAccelerated ? TiledKernel.Neon : TiledKernel.None), "portable: the 128-bit kernel");
        Check(CpuTuning.KernelFor(CpuVectorPaths.Scalar) == TiledKernel.None, "scalar: no tiled kernel");
        Check(CpuTuning.KernelFor(CpuVectorPaths.Automatic) == (native != TiledKernel.None ? native
            : Vector128.IsHardwareAccelerated && RuntimeInformation.ProcessArchitecture == Architecture.Arm64 ? TiledKernel.Neon : TiledKernel.None), "automatic");

        // The int4 lane shifts: a shift per lane and a multiply by 2^count give the same bits (rule 71), on every count used.
        var random = new Random(34);
        for (int trial = 0; trial < 256; trial++)
        {
            var lanes = new int[8];
            var counts = new int[8];
            for (int l = 0; l < 8; l++)
            {
                lanes[l] = random.Next(int.MinValue, int.MaxValue);
                counts[l] = 28 - 4 * random.Next(8);
            }

            var products = lanes.Select((v, l) => unchecked(v * (1 << counts[l]))).ToArray();
            Check(lanes.Select((v, l) => v << counts[l]).SequenceEqual(products), "a shift is a multiply by 2^count");
            if (Avx2.IsSupported)
            {
                var shifted = Avx2.ShiftLeftLogicalVariable(Vector256.Create(lanes), Vector256.Create(counts).AsUInt32());
                Check(shifted == Vector256.Create(lanes) * Vector256.Create(products.Select((p, l) => 1 << counts[l]).ToArray()), "AVX2's lane shift as the multiply");
            }

            if (AdvSimd.IsSupported)
            {
                var shifted = AdvSimd.ShiftLogical(Vector128.Create(lanes[..4]), Vector128.Create(counts[..4]));
                Check(shifted == Vector128.Create(products[..4]), "NEON's lane shift as the multiply");
            }
        }

        string described = CpuTuning.DescribeVectorPaths();
        Console.WriteLine($"    vector paths: {described}; instruction sets: {CpuInfo.Current.InstructionSets}");
        Check(described.Contains("IDRAK_CPU_VECTORS", StringComparison.Ordinal) && Device.Cpu.Backend.Hardware.Facts!.Any(f => f.Key == "vector paths"),
            "the vector paths are listed with the device");
    }

    // A device that builds its kernels when first used, whose Softmax kernel the "driver" refuses and whose Fill fails
    // otherwise, on request (the dispatcher's seam of a refused kernel, on the CPU).
    internal sealed class RefusingBackend(string kind) : Backend
    {
        public bool Refuse;
        public readonly List<string> FellBackTo = [];

        protected override bool BuildsKernelsWhenUsed => true;

        protected override void KernelFellBack(string operation, KernelUnavailableException refused, string fallback) =>
            FellBackTo.Add($"{operation}|{refused.Kernel}|{fallback}");

        public override BackendCapabilities Capabilities => CpuBackend.Instance.Capabilities;

        public override string Kind => kind;

        public override string Name => "refusing (Softmax refused, Fill failing, on request)";

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

        public override void SoftmaxKernel(Storage x, Storage y, int rows, int cols, bool log)
        {
            if (Refuse)
            {
                using (DeviceException.Handled())
                {
                    throw new KernelUnavailableException(Kind, "softmax", "injected: the driver refused 'softmax'");
                }
            }

            base.SoftmaxKernel(x, y, rows, cols, log);
        }

        public override bool MatMulBiasKernel(Storage a, Storage b, Storage bias, Storage c, int m, int n, int k)
        {
            if (Refuse)
            {
                using (DeviceException.Handled())
                {
                    throw new KernelUnavailableException(Kind, "matmul_bias", "injected: the driver refused 'matmul_bias'");
                }
            }

            return false;
        }

        public override void FillKernel(Storage y, int n, float value)
        {
            if (Refuse)
            {
                throw new DeviceException(Kind, "injected: another device error in Fill");
            }

            base.FillKernel(y, n, value);
        }
    }

    internal static void RefusedKernelFallsBack(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(RefusedKernelFallsBack)))
        {
            return;
        }

        var refusing = new RefusingBackend("refusing") { RetryOnHost = false, Refuse = true };
        float[] values = [1f, 2f, 3f, -1f, 0f, 1f, 0.5f, 7f, -3f];
        var x = refusing.Allocate(values.Length, zeroed: false);
        var y = refusing.Allocate(values.Length, zeroed: true);
        refusing.Upload(values, x);
        var cpuX = CpuBackend.Instance.Allocate(values.Length, zeroed: false);
        var cpuY = CpuBackend.Instance.Allocate(values.Length, zeroed: true);
        try
        {
            CpuBackend.Instance.Upload(values, cpuX);
            CpuBackend.Instance.Softmax(cpuX, cpuY, 3, 3, log: false);
            var expected = new float[values.Length];
            CpuBackend.Instance.Download(cpuY, expected);

            long hostCalls = Kernels.HostCalls(refusing);
            for (int call = 0; call < 3; call++)
            {
                refusing.Softmax(x, y, 3, 3, log: false);                     // the first call too: no exception
                var got = new float[values.Length];
                refusing.Download(y, got);
                Check(got.Select(BitConverter.SingleToInt32Bits).SequenceEqual(expected.Select(BitConverter.SingleToInt32Bits)),
                    $"call {call}: the host fallback gives the CPU's result, bit for bit");
            }

            Check(Kernels.HostCalls(refusing) == hostCalls + 3, $"each call ran on the host ({Kernels.HostCalls(refusing) - hostCalls})");
            Check(refusing.FellBackTo is ["Softmax|softmax|the host"], $"the device is told once what ran instead: {string.Join(", ", refusing.FellBackTo)}");

            // An operation with no kernel otherwise: "none", so the caller takes its own path.
            var c = refusing.Allocate(4, zeroed: true);
            Check(!refusing.MatMulBias(x, x, x, c, 1, 1, 1), "a refused kernel of an operation with no fallback returns its none");
            Check(refusing.FellBackTo.Contains("MatMulBias|matmul_bias|the caller's path"), "and is listed as the caller's path");

            // Any other device error still reaches the caller (RetryOnHost off: plan 10, decision 13).
            DeviceException? thrown = null;
            try
            {
                using (DeviceException.Handled())
                {
                    refusing.Fill(y, values.Length, 1f);
                }
            }
            catch (DeviceException e)
            {
                thrown = e;
            }

            Check(thrown is not null and not KernelUnavailableException, "another device error is not retried");
        }
        finally
        {
            cpuX.Release();
            cpuY.Release();
        }
    }

    // ------------------------------------------------------------------ Vulkan: refused kernels (any Vulkan device)

    internal static void VulkanRefusedKernels(Device device)
    {
        foreach (int ordinal in LimitOrdinals(device))
        {
            var shared = (VulkanBackend)device.Backend;
            var savedRefusals = VulkanBackend.RefuseKernels;
            string? savedCache = VulkanBackend.TuningCacheFile;
            var savedFormula = VulkanBackend.ProbeFormulaOverride;
            string cache = Path.Combine(Path.GetTempPath(), $"idrak-refused-{Environment.ProcessId}.tsv");
            var cpu = CpuBackend.Instance;
            const int M = 8, K = 96, N = 300;
            var w = Values(K * N, 5);
            var int8 = Int8Weight.Quantize(w, K, N, Device.Cpu);
            float[] q8 = int8.Packed.ToArray(), s8 = int8.Scales.ToArray(), xs = Values(M * K, 6), rows = Values(4 * 300, 7);

            float[] Product(Backend b, int m)
            {
                var (x, q, s, y) = (b.Allocate(m * K, false), b.Allocate(q8.Length, false), b.Allocate(s8.Length, false), b.Allocate(m * N, true));
                try
                {
                    b.Upload(xs.AsSpan(0, m * K), x);
                    b.Upload(q8, q);
                    b.Upload(s8, s);
                    b.Int8MatMul(x, q, s, y, m, N, K);
                    var result = new float[m * N];
                    b.Download(y, result);
                    return result;
                }
                finally
                {
                    x.Release();
                    q.Release();
                    s.Release();
                    y.Release();
                }
            }

            float[] Softmax(Backend b)
            {
                var (x, y) = (b.Allocate(rows.Length, false), b.Allocate(rows.Length, true));
                try
                {
                    b.Upload(rows, x);
                    b.Softmax(x, y, 4, 300, log: false);
                    var result = new float[rows.Length];
                    b.Download(y, result);
                    return result;
                }
                finally
                {
                    x.Release();
                    y.Release();
                }
            }

            static bool Same(float[] a, float[] b) => a.Select(BitConverter.SingleToInt32Bits).SequenceEqual(b.Select(BitConverter.SingleToInt32Bits));
            static string Fact(Backend b, string name) => b.Hardware.Facts!.First(f => f.Key == name).Value;

            float[] reference = Product(shared, M);                            // the device's own kernel, nothing refused
            float[] cpuProduct = Product(cpu, M);
            AssertClose(cpuProduct, reference, 1e-4f, "the int8 product matches the CPU");
            try
            {
                VulkanBackend.TuningCacheFile = cache;
                VulkanBackend.ProbeFormulaOverride = Math.Min(256, Math.Min(shared.Limits.MaxInvocations, shared.Limits.MaxSizeX));

                // 1. The 8-row int8 product kernel refused (both word variants): the device starts, the width probe
                //    included (it runs 64-row int8 products), and the products run the next variant with the same bits.
                File.Delete(cache);
                VulkanBackend.RefuseKernels = ["int8_gemv_8", "int8_gemv_8_w64"];
                var first = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                try
                {
                    Console.WriteLine($"    vulkan:{ordinal}: started with int8_gemv_8 refused: width {first.Width} ({first.WidthChoice})");
                    long hostCalls = Kernels.HostCalls(first);
                    var got = Product(first, M);
                    Check(Same(reference, got), "8 rows through the next variant: the bits of the device's own kernel");
                    int attempts = first.PipelineAttempts;
                    for (int again = 0; again < 3; again++)
                    {
                        Check(Same(reference, Product(first, M)), "again: the same bits");
                    }

                    Check(first.PipelineAttempts == attempts, $"the refusal is asked once: {first.PipelineAttempts - attempts} more pipeline attempts");
                    Check(Kernels.HostCalls(first) == hostCalls, "nothing ran on the host");
                    Check(first.RefusedKernels.Any(r => r.Kernel == "int8_gemv_8"), "the refusal is kept");
                    string refused = Fact(first, "refused kernels");
                    Console.WriteLine($"    vulkan:{ordinal}: refused kernels: {refused}");
                    Check(refused.Contains("int8_gemv_8 at width", StringComparison.Ordinal) && refused.Contains("VK_ERROR_UNKNOWN", StringComparison.Ordinal)
                          && refused.Contains("int8_gemv_4", StringComparison.Ordinal), "the facts list the refused kernel, how, and what ran instead");
                }
                finally
                {
                    first.Shutdown();
                }

                // 2. Every int8 product kernel and the softmax kernels refused: those operations run on the host with the
                //    CPU's bits, once asked.
                VulkanBackend.RefuseKernels = [.. VulkanKernels.GemvRowBlocks.SelectMany(r => VulkanKernels.GemvWordCounts
                    .Select(words => VulkanKernels.GemvName(VulkanKernels.PackedFormat.Int8, r, words))), "softmax", "softmax_narrow"];
                var second = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                try
                {
                    long hostCalls = Kernels.HostCalls(second);
                    Check(Same(cpuProduct, Product(second, M)), "every int8 product kernel refused: the host's result, the CPU's bits");
                    float[] cpuSoftmax = Softmax(cpu);
                    Check(Same(cpuSoftmax, Softmax(second)), "softmax refused: the host's result, the CPU's bits");
                    int attempts = second.PipelineAttempts;
                    Check(Same(cpuProduct, Product(second, M)) && Same(cpuSoftmax, Softmax(second)), "again");
                    Check(second.PipelineAttempts == attempts, $"asked once: {second.PipelineAttempts - attempts} more attempts");
                    Check(Kernels.HostCalls(second) > hostCalls, "they ran on the host");
                    string refused = Fact(second, "refused kernels");
                    Console.WriteLine($"    vulkan:{ordinal}: refused kernels: {refused}");
                    Check(refused.Contains("the host", StringComparison.Ordinal) && refused.Contains("softmax", StringComparison.Ordinal),
                        "the facts say they fell back to the host");
                    Check(Same(Product(cpu, 1), Product(second, 1)), "one row too");
                }
                finally
                {
                    second.Shutdown();
                }

                // 3. A kernel refused only with its subgroup-size request: built again without it, the same results.
                var probe = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                int[] sizes;
                try
                {
                    sizes = probe.SubgroupSizeControl ? probe.SubgroupSizeCandidates(probe.Width) : [];
                }
                finally
                {
                    probe.Shutdown();
                }

                if (sizes.Length == 0)
                {
                    Console.WriteLine($"    vulkan:{ordinal}: no subgroup size control: the retry without the request is not reachable here");
                }
                else
                {
                    VulkanBackend.RefuseKernels = ["fill:sized"];
                    var third = VulkanBackend.CreateSeparate(ordinal, preferMapped: true);
                    try
                    {
                        var built = VulkanKernels.Get("fill", third.Width);
                        var sized = new VulkanKernel(built.Words, built.Bindings, built.PushBytes, built.Name, built.Writes) { RequiredSubgroupSize = sizes[^1] };
                        var y = third.Allocate(1000, zeroed: true);
                        int before = third.PipelineAttempts;
                        var push = new byte[8];
                        BitConverter.TryWriteBytes(push.AsSpan(0, 4), 1000);
                        BitConverter.TryWriteBytes(push.AsSpan(4, 4), 2.5f);
                        third.Dispatch(sized, (uint)((1000 + third.Width - 1) / third.Width), 1, 1, [y], push);
                        var filled = new float[1000];
                        third.Download(y, filled);
                        y.Release();
                        Check(filled.All(v => v == 2.5f), "built without the request, the kernel's result");
                        Check(third.PipelineAttempts - before == 2, $"two attempts: with the request, then without ({third.PipelineAttempts - before})");
                        Check(Fact(third, "subgroup-size requests dropped").Contains("fill", StringComparison.Ordinal) && Fact(third, "refused kernels") == "none",
                            "listed as built without its request, not refused");
                    }
                    finally
                    {
                        third.Shutdown();
                    }
                }
            }
            finally
            {
                VulkanBackend.RefuseKernels = savedRefusals;
                VulkanBackend.TuningCacheFile = savedCache;
                VulkanBackend.ProbeFormulaOverride = savedFormula;
                File.Delete(cache);
            }
        }
    }
}
