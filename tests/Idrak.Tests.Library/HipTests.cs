// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;

// The HIP backend (the graphics driver's runtime on Windows, ROCm's on Linux; no part of the HIP SDK): found at run time
// or absent without harm, its devices named hip:N, its block size from the reported limits, its kernels (Idrak's own GCN
// code, plan 35) kept on disk per architecture and driver. Most tests here run anywhere (no runtime needed); the last
// runs on HIP devices, which a plain run includes when present.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] HipGroup =
    [
        ("hip runtime: library names per system; the system's search (the driver's runtime) first, ROCm's folders on Linux only, no SDK's folder anywhere; a missing runtime means no HIP devices, a reason and no exception", HipMissingRuntime),
        ("hip devices: named hip:N, parsed like the others, listed in --list-devices (with the reason when there are none)", HipDeviceNames),
        ("hip limits: the block size and the limits' plausibility come from what the device reports", HipBlockSize),
        ("hip architecture: the target string is read from the device properties by content; runtime versions read as major.minor.patch", HipArchitecture),
        ("hip kernel cache: the key changes with every input (a kernel set's with any kernel's text, the target, the device, the format and the versions); a stored code object comes back; IDRAK_HIP_KERNEL_CACHE moves or disables it", HipKernelCacheKeys),
        ("hip kernels: GcnKernels.All (fill_f32, the element-wise, row, product, attention, shape and decoding kernels) assembles into one code object for gfx900, gfx906, gfx908 and gfx90c (formats 5 and 4), whose metadata, read back by the library, declares every kernel once with the arguments, kernel argument bytes and block it was written with; malformed objects are refused", HipKernelSet),
        ("hip kernels: IDRAK_HIP_KERNELS=0 turns the kernels off with that reason; a target Idrak does not read or no family handles gives a reason naming it (not registered); every kernel has its profile cost; IDRAK_HIP_ATTENTION_SPLIT and attention's splits; IDRAK_HIP_GEMV_SPLITS and the k splits of the packed few-row products; the element-wise grid's blocks per compute unit come from IDRAK_HIP_POINTWISE_BLOCKS", HipKernelSettings),
        ("hip telemetry: KernelsLoaded (a device's kernel set, from the cache or assembled, its time, kernels and bytes) reaches the hooks at the Devices level, the console logger's line and the JSON lines' kernels_loaded", HipKernelsLoadedEvent),
        ("hip: no vendor ids or card names in the code of src/Idrak.Gpu/Hip, and GPU targets (gfx names) only in its instruction-set tables, GcnTargets*.cs (comments may name them)", HipNoVendorNames),
        ("hip profiler and telemetry: HipBackend times its kernels (IKernelProfiler) as CUDA does; a runtime without the event functions times nothing and says why; on HIP devices fill_f32's launches are timed under its name or a label with its floating-point operations, a refused code object reaches the telemetry, and a host fallback carries why the kernels are unavailable", HipProfilerAndTelemetry),
        ("hip device: memory, uploads, downloads, copies and fills on the device, and every operation matches the CPU: the row, element-wise, attention, shape and product operations (float32, few-row, packed int8, int4, bfloat16 and float16, the expansions and bfloat16 packing) and the decoding ones (the fused packed products, attention over rows and packed sequences, penalties, rotary embeddings, the element-wise operations with parameters) on the GCN kernels (kernels loaded: none of them through the host fallback; on HIP devices)", HipDeviceMatchesCpu),
    ];

    internal static void HipMissingRuntime(Device device)
    {
        _ = device;
        Check(HipRuntime.CandidateNames("linux").Contains("libamdhip64.so") && HipRuntime.CandidateNames("linux").Contains("libamdhip64.so.6"),
            "Linux: ROCm's runtime names");
        Check(HipRuntime.CandidateNames("windows").Contains("amdhip64_6.dll") && HipRuntime.CandidateNames("windows").Contains("amdhip64.dll"),
            "Windows: the graphics driver's runtime names");
        Check(HipRuntime.CandidateNames("macos").Length == 0, "no names elsewhere");
        Check(!HipRuntime.SearchFolders("windows", v => v is "HIP_PATH" or "ROCM_PATH" ? @"C:\HIP" : null).Any(),
            "Windows: only the system's search (the driver's runtime), never an SDK's folder");
        Check(HipRuntime.SearchFolders("linux", _ => null).SequenceEqual(["/opt/rocm/lib"]), "Linux: ROCm's default folder");
        Check(HipRuntime.SearchFolders("linux", v => v == "ROCM_PATH" ? "/r" : null).SequenceEqual([Path.Combine("/r", "lib"), "/opt/rocm/lib"]),
            "Linux: ROCM_PATH's lib folder first");

        // A runtime that is not there: no handle, no exception, a reason naming what was looked for.
        bool opened = HipRuntime.TryOpen(["libidrak-no-such-hip-runtime.so"], [Path.GetTempPath()], out var handle, out string reason, out string from);
        Check(!opened && handle == IntPtr.Zero && from.Length == 0 && reason.Contains("libidrak-no-such-hip-runtime.so", StringComparison.Ordinal), $"missing library: {reason}");

        // A library found opens and says from where (the system's search first, then the install's folders).
        string folder = System.Runtime.InteropServices.RuntimeEnvironment.GetRuntimeDirectory();
        string file = OperatingSystem.IsWindows() ? "clrjit.dll" : OperatingSystem.IsMacOS() ? "libclrjit.dylib" : "libclrjit.so";
        bool found = HipRuntime.TryOpen([file], [folder], out var both, out _, out string where);
        Check(found && where.EndsWith(file, StringComparison.Ordinal), $"opened from: {where}");
        if (found)
        {
            System.Runtime.InteropServices.NativeLibrary.Free(both);
        }

        var provider = DeviceProviders.Find("hip");
        Check(provider is { Type: DeviceType.Hip, Display: "HIP" }, "the HIP provider is registered");
        if (provider!.Count == 0)
        {
            Check(provider.UnavailableReason is { Length: > 0 }, "a reason when there are no devices");
            Check(!Device.Available.Any(d => d.Type == DeviceType.Hip), "no HIP device listed");
            try
            {
                Device.Get("hip");
                Check(false, "Device.Get(\"hip\") without devices throws");
            }
            catch (InvalidOperationException e)
            {
                Check(e.Message.Contains("HIP is not available", StringComparison.Ordinal), $"a clear message: {e.Message}");
            }

            Console.WriteLine($"    no HIP devices: {provider.UnavailableReason}");
        }
        else
        {
            Console.WriteLine($"    {provider.Count} HIP device(s), runtime from {HipRuntime.LoadedFrom}");
        }
    }

    internal static void HipDeviceNames(Device device)
    {
        _ = device;
        int count = DeviceProviders.Find("hip")!.Count;
        for (int i = 0; i < count; i++)
        {
            var hip = Device.Parse($"hip:{i}");
            Check(hip.Type == DeviceType.Hip && hip.Kind == "hip" && hip.Ordinal == i && hip.ToString() == $"hip:{i}" && hip.IsGpu, $"hip:{i}");
            Check(Device.Parse($"HIP:{i}") == hip && Device.Available.Contains(hip), $"hip:{i} is case-insensitive and listed (tested in plain runs)");
        }

        try
        {
            Device.Parse($"hip:{count}");
            Check(false, $"hip:{count} does not exist");
        }
        catch (InvalidOperationException)
        {
        }

        Check(Environment.GetEnvironmentVariable("IDRAK_HIP_DEFAULT") == "1" || Device.Default.Type != DeviceType.Hip,
            "not the default device unless IDRAK_HIP_DEFAULT=1");

        // The device listing has a HIP row: each device, or why there is none.
        var saved = Console.Out;
        var text = new StringWriter();
        Console.SetOut(text);
        try
        {
            ListDevices();
        }
        finally
        {
            Console.SetOut(saved);
        }

        string listing = text.ToString();
        Check(count == 0 ? listing.Contains("| hip:-", StringComparison.Ordinal) && listing.Contains("none found:", StringComparison.Ordinal)
                : listing.Contains("| hip:0", StringComparison.Ordinal),
            "--list-devices shows HIP");
    }

    // Device limits as a runtime might report them (no card in mind: the values vary independently).
    internal static HipDeviceLimits HipLimits(int warp, int threads, int shared, int gridX = int.MaxValue) => new()
    {
        Name = "test device",
        Architecture = "",
        TotalMemory = 1L << 30,
        ComputeUnits = 8,
        WarpSize = warp,
        MaxThreadsPerBlock = threads,
        MaxBlockDimX = threads,
        MaxGridDimX = gridX,
        MaxGridDimY = 65535,
        SharedMemoryPerBlock = shared,
        L2CacheBytes = 0,
        MemoryBusWidth = 0,
        Major = 0,
        Minor = 0,
        Integrated = false,
        PciAddress = "",
    };

    internal static void HipBlockSize(Device device)
    {
        _ = device;

        // A quarter of the largest block, a power of two rounded down, whole wavefronts.
        Check(HipLimits(64, 1024, 65536).BlockSize == 256, "64 lanes, 1024 threads: 256");
        Check(HipLimits(32, 1024, 65536).BlockSize == 256, "32 lanes, 1024 threads: 256");
        Check(HipLimits(64, 256, 65536).BlockSize == 64, "256 threads: one 64-lane wavefront");
        Check(HipLimits(32, 200, 65536).BlockSize == 32, "200 threads: rounded down to a power of two");
        foreach (int warp in new[] { 16, 32, 64 })
        {
            foreach (int threads in new[] { 64, 128, 256, 512, 1024, 2048 })
            {
                int block = HipLimits(warp, threads, 65536).BlockSize;
                Check(block % warp == 0 && block <= threads && (block & (block - 1)) == 0, $"warp {warp}, {threads} threads: {block}");
            }
        }

        Check(HipLimits(64, 1024, 65536).Problem() is null && HipLimits(32, 1024, 65536).Problem() is null, "plausible limits are used");
        Check(HipLimits(48, 1024, 65536).Problem() is not null && HipLimits(0, 1024, 65536).Problem() is not null
              && HipLimits(64, 32, 65536).Problem() is not null && HipLimits(64, 1024, 0).Problem() is not null
              && HipLimits(64, 1024, 65536, gridX: 0).Problem() is not null,
            "implausible limits (a runtime numbering its attributes differently) turn the kernels off");
    }

    internal static void HipArchitecture(Device device)
    {
        _ = device;
        byte[] properties = new byte[4096];
        "Some GPU gfx9000"u8.CopyTo(properties);                                // in the name: not the target
        "gfx90a:sramecc+:xnack-"u8.CopyTo(properties.AsSpan(1200));
        Check(HipDeviceLimits.ArchitectureFrom(properties) == "gfx90a:sramecc+:xnack-", "the target string after the name, with its features");
        Check(HipDeviceLimits.ArchitectureFrom(new byte[4096]) == "", "none found: empty");
        byte[] other = new byte[4096];
        "gfx 12"u8.CopyTo(other.AsSpan(800));
        Check(HipDeviceLimits.ArchitectureFrom(other) == "", "a prefix without a target is not one");

        Check(HipBackend.FormatVersion(60241134) == "6.2.41134", "runtime versions read as major.minor.patch");
    }

    internal static void HipKernelCacheKeys(Device device)
    {
        _ = device;
        string[] options = ["block=256", "target=gfx1100"];
        string key = HipKernelCache.Key("source", options, "gfx1100", "device", "runtime 1");
        Check(key.Length == 32 && key == HipKernelCache.Key("source", options, "gfx1100", "device", "runtime 1"), "the same inputs, the same key");
        Check(key != HipKernelCache.Key("source2", options, "gfx1100", "device", "runtime 1")
              && key != HipKernelCache.Key("source", ["block=128", "target=gfx1100"], "gfx1100", "device", "runtime 1")
              && key != HipKernelCache.Key("source", options, "gfx1101", "device", "runtime 1")
              && key != HipKernelCache.Key("source", options, "gfx1100", "device 2", "runtime 1")
              && key != HipKernelCache.Key("source", options, "gfx1100", "device", "runtime 2"),
            "source, options, architecture, device and versions each change the key");

        // A kernel set's key: every kernel's text, the options (the format), the target, the device and the versions.
        GcnKernelSource[] set = [new("a", "s_endpgm\n"u8.ToArray()), new("b", "s_nop 0\n"u8.ToArray())];
        string setKey = HipKernelCache.KeyOf(set, ["format=5"], "gfx90c", "device", "runtime 1");
        Check(setKey.Length == 32 && setKey == HipKernelCache.KeyOf([.. set], ["format=5"], "gfx90c", "device", "runtime 1"), "a kernel set: the same inputs, the same key");
        Check(setKey != HipKernelCache.KeyOf([set[0], new("b", "s_nop 1\n"u8.ToArray())], ["format=5"], "gfx90c", "device", "runtime 1")
              && setKey != HipKernelCache.KeyOf([set[0], new("c", set[1].Text)], ["format=5"], "gfx90c", "device", "runtime 1")
              && setKey != HipKernelCache.KeyOf([set[0]], ["format=5"], "gfx90c", "device", "runtime 1")
              && setKey != HipKernelCache.KeyOf(set, ["format=4"], "gfx90c", "device", "runtime 1")
              && setKey != HipKernelCache.KeyOf(set, ["format=5"], "gfx908", "device", "runtime 1")
              && setKey != HipKernelCache.KeyOf(set, ["format=5"], "gfx90c", "device 2", "runtime 1")
              && setKey != HipKernelCache.KeyOf(set, ["format=5"], "gfx90c", "device", "runtime 2"),
            "a kernel set: any kernel's text or name, the kernel count, the format, the target, the device and the versions each change the key");
        Check(HipKernelCache.KeyOf(GcnKernels.All, ["format=5"], "gfx90c", "d", "v") is { Length: 32 }, "HIP's own kernel set has a key");

        string folder = Path.Combine(Path.GetTempPath(), $"idrak-hip-cache-{Environment.ProcessId}");
        try
        {
            var cache = new HipKernelCache(folder);
            Check(cache.Load(key, "gfx1100") is null, "nothing kept yet");
            byte[] code = [1, 2, 3, 4, 5];
            cache.Store(key, "gfx90a:sramecc+:xnack-", code);
            Check(cache.Load(key, "gfx90a:sramecc+:xnack-") is { } back && back.SequenceEqual(code), "a stored code object comes back");
            Check(Path.GetFileName(cache.PathOf(key, "gfx90a:sramecc+:xnack-"))!.All(c => char.IsAsciiLetterOrDigit(c) || c is '-' or '.' or '_'),
                "file names are safe on every system");
            cache.Delete(key, "gfx90a:sramecc+:xnack-");
            Check(cache.Load(key, "gfx90a:sramecc+:xnack-") is null, "a refused code object is forgotten");
            Check(new HipKernelCache(null).PathOf(key, "gfx1100") is null && new HipKernelCache(null).Load(key, "gfx1100") is null, "no folder: nothing kept");
        }
        finally
        {
            if (Directory.Exists(folder))
            {
                Directory.Delete(folder, recursive: true);
            }
        }

        Check(HipKernelCache.DefaultFolder(v => v == "IDRAK_HIP_KERNEL_CACHE" ? "0" : null) is null, "IDRAK_HIP_KERNEL_CACHE=0 keeps nothing");
        Check(HipKernelCache.DefaultFolder(v => v == "IDRAK_HIP_KERNEL_CACHE" ? "/tmp/k" : null) == "/tmp/k", "IDRAK_HIP_KERNEL_CACHE=folder moves it");
        Check(HipKernelCache.DefaultFolder(v => v == "IDRAK_CACHE" ? "/c" : null) == Path.Combine("/c", "hip", "kernels"), "under IDRAK_CACHE");
    }

    internal static void HipKernelSet(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(HipKernelSet)))
        {
            return;
        }

        var all = GcnKernels.All;
        Check(all.Select(k => k.Name).Distinct().Count() == all.Count
              && all.Count == 1 + GcnKernels.Pointwise.Count + GcnKernels.Rows.Count + GcnKernels.Products.Count + GcnKernels.Attention.Count + GcnKernels.Shapes.Count
                 + GcnKernels.Decoding.Count + GcnKernels.Training.Count
              && all[0].Name == "fill_f32",
            $"GcnKernels.All: fill_f32, then the element-wise, row, product, attention, shape, decoding and training kernels, each once ({all.Count})");
        foreach (string processor in new[] { "gfx900", "gfx906", "gfx908", "gfx90c" })
        {
            var isa = GcnIsaVersion.Parse(processor);
            var target = GcnTargets.For(isa);
            var codes = GcnKernels.AssembleAll(target, isa);
            Check(codes.Length == all.Count && codes.Select(c => c.Name).SequenceEqual(all.Select(k => k.Name)), $"{processor}: every kernel assembled, in order");
            foreach (int format in new[] { 5, 4 })
            {
                byte[] image = GcnCodeObject.Write(target, isa, format, codes);
                var layouts = GcnCodeObject.ReadKernels(image);
                Check(layouts.Select(l => l.Name).SequenceEqual(all.Select(k => k.Name)), $"{processor}, format {format}: the metadata declares every kernel once, in order");
                for (int i = 0; i < codes.Length; i++)
                {
                    var (code, layout) = (codes[i], layouts[i]);
                    Check(layout.Arguments.SequenceEqual(code.Arguments) && layout.ArgumentBytes == GcnCodeObject.KernargBytes(code)
                          && layout.Block == (code.RequiredWorkgroupSize?.X ?? code.MaxFlatWorkgroupSize) && layout.Block is 256 or GcnKernels.PackedGemvWords,
                        $"{processor}, format {format}, {code.Name}: arguments, bytes and block as assembled ({layout.ArgumentBytes} bytes, block {layout.Block})");
                }

                // The test runner's own reader (tests/Gcn) finds every kernel's code too.
                Check(Idrak.Tests.Gcn.GcnCodeObjectReader.Read(image, "layernorm_bwd_params_f32") is not null, $"{processor}, format {format}: the reader of the tests finds a row kernel");
            }

            Check(GcnKernels.CodeObject(isa, 5).Length > 0, $"{processor}: GcnKernels.CodeObject writes the set");
        }

        // Malformed code objects are refused, never read past.
        bool Refused(byte[] bytes)
        {
            try
            {
                GcnCodeObject.ReadKernels(bytes);
                return false;
            }
            catch (InvalidDataException)
            {
                return true;
            }
        }

        byte[] good = GcnKernels.CodeObject(GcnIsaVersion.Parse("gfx90c"), 5);
        Check(Refused([]) && Refused(new byte[64]) && Refused(good[..(good.Length / 3)]), "an empty, a zeroed and a cut code object are refused");
        int note = good.AsSpan().IndexOf("amdhsa.kernels"u8);
        byte[] broken = (byte[])good.Clone();
        broken[note - 1] = 0xC1;                                                  // a MessagePack type that does not exist, before the key
        Check(note > 0 && Refused(broken), "a metadata note that does not parse is refused");
    }

    internal static void HipKernelSettings(Device device)
    {
        _ = device;
        Check(HipBackend.KernelsSettingReason("0") == "turned off by IDRAK_HIP_KERNELS=0" && HipBackend.KernelsSettingReason("false") == "turned off by IDRAK_HIP_KERNELS=false",
            $"IDRAK_HIP_KERNELS=0: {HipBackend.KernelsSettingReason("0")}");
        Check(HipBackend.KernelsSettingReason(null) is null && HipBackend.KernelsSettingReason("") is null && HipBackend.KernelsSettingReason("1") is null, "unset or 1: on");
        Check(HipBackend.TargetUnavailable("hip:0", "gfx90c:xnack-", out var isa, out var target) is null && target is { Family: "gfx9" } && isa.Minor == 0 && isa.Stepping == 12,
            "a target gfx9 handles: no reason, its family");
        string? unregistered = HipBackend.TargetUnavailable("hip:0", "gfx1100", out _, out var none);
        Check(none is null && unregistered is not null && unregistered.Contains("gfx1100", StringComparison.Ordinal) && unregistered.Contains("not registered", StringComparison.Ordinal)
              && unregistered.Contains("hip:0", StringComparison.Ordinal), $"a target no family handles: {unregistered}");
        string? unread = HipBackend.TargetUnavailable("hip:0", "sm_80", out _, out _);
        Check(unread is not null && unread.Contains("'sm_80'", StringComparison.Ordinal) && unread.Contains("not a target string", StringComparison.Ordinal), $"a target Idrak does not read: {unread}");

        var missing = GcnKernels.All.Where(k => HipBackend.Cost(k.Name).Bytes <= 0 && HipBackend.ProductCost(k.Name, 2, 3, 4).Bytes <= 0).Select(k => k.Name).ToList();
        Check(missing.Count == 0, $"every kernel has its profile cost (the tables of GcnKernels.Rows.cs, .Pointwise.cs, .Products.cs, .Attention.cs and .Shapes.cs): {string.Join(", ", missing)}");
        Check(HipBackend.GemvSplitsSetting(null) is null && HipBackend.GemvSplitsSetting("6") == 6 && HipBackend.GemvSplitsSetting("0") is null && HipBackend.GemvSplitsSetting("x") is null
              && HipBackend.GemvSplitCount(8, 4, 3072) == 4 && HipBackend.GemvSplitCount(8, 4, 3072, forced: 6) == 6 && HipBackend.GemvSplitCount(64, 1, 100) == 1
              && HipBackend.GemvSplitCount(1000, 1, 1 << 20) == 64, "IDRAK_HIP_GEMV_SPLITS forces the packed few-row products' k splits; else about two blocks a compute unit, a power of two, at most min(64, k / 64)");
        Check(HipBackend.AttentionSplitPositions(null) == 512 && HipBackend.AttentionSplitPositions("0") == 0 && HipBackend.AttentionSplitPositions("300") == 512
              && HipBackend.AttentionSplitPositions("1024") == 1024 && HipBackend.AttentionSplitPositions("x") == 512,
            "IDRAK_HIP_ATTENTION_SPLIT: positions a split, a multiple of the 256-key tile (512 unless given), 0 for none");
        Check(HipBackend.AttentionSplits(32, 4096, 128, 512) == (8, 512) && HipBackend.AttentionSplits(32, 400, 128, 512) == (1, 400)
              && HipBackend.AttentionSplits(32, 4096, 128, 0) == (1, 4096) && HipBackend.AttentionSplits(8192, 4096, 128, 512) == (1, 4096)
              && HipBackend.AttentionSplits(4, 1 << 20, 64, 512) == (64, 16384),
            "attention's splits follow the shapes alone: the positions a split takes, at most 64 splits and 4,096 blocks, chunks of whole tiles");
        Check(Enum.GetValues<UnaryOp>().Count(op => HipBackend.UnaryKernelName(op) is { } name && GcnKernels.All.Any(k => k.Name == name)) == 14
              && Enum.GetValues<BinaryOp>().All(op => HipBackend.BinaryKernelName(op) is { } name && GcnKernels.All.Any(k => k.Name == name)),
            "the activations and binary operations CUDA runs on kernels name kernels of the set");
        Check(HipBackend.PointwiseBlocksPerUnit(null) == 8 && HipBackend.PointwiseBlocksPerUnit("12") == 12 && HipBackend.PointwiseBlocksPerUnit("0") == 8
              && HipBackend.PointwiseBlocksPerUnit("x") == 8, "IDRAK_HIP_POINTWISE_BLOCKS: blocks per compute unit, 8 unless a positive number is given");
    }

    internal static void HipKernelsLoadedEvent(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(HipKernelsLoadedEvent)))
        {
            return;
        }

        var seen = new List<KernelsLoaded>();
        var text = new StringWriter();
        var loaded = new KernelsLoaded("hip:0", "hip", 34, FromCache: false, TimeSpan.FromMilliseconds(123.4), 98_304, "gfx90c, code object format 5");
        using (Telemetry.Subscribe(new KernelsLoadedHook(seen)))
        using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Devices, output: text)))
        {
            Telemetry.KernelsLoaded(loaded);
            Telemetry.KernelsLoaded(loaded with { FromCache = true, Duration = TimeSpan.FromMilliseconds(2.5) });
        }

        Check(seen.Count == 2 && seen[0] == loaded && seen[1].FromCache, $"the hooks at the Devices level get each load: {seen.Count}");
        string[] lines = text.ToString().Split('\n', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        Check(lines.Length == 2 && lines[0] == "Device hip:0 loaded 34 hip kernels made and loaded (cold) in 123.4 ms, 98,304 bytes (gfx90c, code object format 5)"
              && lines[1].Contains("from the kernel cache (warm) in 2.5 ms", StringComparison.Ordinal), $"the console logger's lines: {text}");

        var json = new MemoryStream();
        using (var writer = new Utf8JsonWriter(json))
        {
            TelemetryJson.Write(writer, new TelemetryRecord(DateTimeOffset.UnixEpoch, loaded));
        }

        using var parsed = JsonDocument.Parse(json.ToArray());
        var root = parsed.RootElement;
        Check(root.GetProperty("event").GetString() == "kernels_loaded" && root.GetProperty("device").GetString() == "hip:0" && root.GetProperty("backend").GetString() == "hip"
              && root.GetProperty("kernels").GetInt32() == 34 && !root.GetProperty("from_cache").GetBoolean() && root.GetProperty("bytes").GetInt64() == 98_304
              && Math.Abs(root.GetProperty("duration_ms").GetDouble() - 123.4) < 1e-9 && root.GetProperty("detail").GetString() == "gfx90c, code object format 5",
            $"the JSON lines' record: {Encoding.UTF8.GetString(json.ToArray())}");
    }

    private sealed class KernelsLoadedHook(List<KernelsLoaded> seen) : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Devices;

        public void OnKernelsLoaded(in KernelsLoaded e)
        {
            lock (seen)
            {
                seen.Add(e);
            }
        }
    }

    internal static void HipNoVendorNames(Device device)
    {
        _ = device;
        string folder = RepositoryPath("src", "Idrak.Gpu", "Hip");
        var vendorIds = new Regex(@"0x0*(10DE|1002|1022|8086|13B5|5143|106B)\b", RegexOptions.IgnoreCase);
        var names = new Regex(@"\b(NVIDIA|AMD|ATI|Intel|Radeon|Instinct|MI\d{2,3}X?|RDNA\d?|CDNA\d?|Vega|Navi|GeForce|RTX|GTX|Arc|Iris)\b", RegexOptions.IgnoreCase);
        var targets = new Regex(@"\bgfx\d", RegexOptions.IgnoreCase);
        var found = new List<string>();
        foreach (string path in Directory.EnumerateFiles(folder, "*.cs", SearchOption.AllDirectories))
        {
            // GPU targets (gfx9, gfx90c) are instruction-set families and the ELF machine values a code object's format
            // requires: the instruction-set tables (GcnTargets*.cs, in this folder itself) may hold them in code, and the
            // target triple (amdgcn-amd-amdhsa) the format names. Vendor ids and other vendor or card names stay out of
            // every file, those tables included.
            string relative = Path.GetRelativePath(folder, path);
            bool targetTable = Path.GetFileName(relative) == relative && relative.StartsWith("GcnTargets", StringComparison.Ordinal);
            string[] lines = File.ReadAllLines(path);
            for (int n = 0; n < lines.Length; n++)
            {
                string line = lines[n].TrimStart();
                if (line.StartsWith("//", StringComparison.Ordinal) || line.StartsWith("///", StringComparison.Ordinal) || line.StartsWith('*'))
                {
                    continue;                                                    // comments may say where something was measured
                }

                int comment = line.IndexOf(" // ", StringComparison.Ordinal);
                string code = comment >= 0 ? line[..comment] : line;
                if (vendorIds.Match(code) is { Success: true } id)
                {
                    found.Add($"{Path.GetFileName(path)}:{n + 1}: {id.Value}");
                }

                // The code object's target triple names the vendor as format data: allowed in the tables too.
                string named = targetTable ? code.Replace("amdgcn-amd-amdhsa", "", StringComparison.Ordinal) : code;
                if (names.Match(named) is { Success: true } name)
                {
                    found.Add($"{relative}:{n + 1}: {name.Value}");
                }

                if (!targetTable && targets.Match(code) is { Success: true } target)
                {
                    found.Add($"{relative}:{n + 1}: {target.Value} (GPU targets belong in GcnTargets*.cs)");
                }
            }
        }

        Check(found.Count == 0, $"vendor ids, names or targets in code: {string.Join("; ", found.Take(10))}");
    }

    internal static void HipDeviceMatchesCpu(Device device)
    {
        if (device.Type != DeviceType.Hip)
        {
            return;
        }

        var hip = (HipBackend)device.Backend;
        var cpu = CpuBackend.Instance;
        var random = new Random(11);
        var owned = new List<Storage>();
        Storage Keep(Storage s)
        {
            owned.Add(s);
            return s;
        }

        float[] Values(int n, float low = -2f, float high = 2f) => [.. Enumerable.Range(0, n).Select(_ => low + random.NextSingle() * (high - low))];
        Storage On(Backend backend, float[] values)
        {
            var s = Keep(backend.Allocate(values.Length, zeroed: false));
            backend.Upload(values, s);
            return s;
        }

        float[] Read(Backend backend, Storage s, int n)
        {
            var values = new float[n];
            backend.Download(s, values);
            return values;
        }

        try
        {
            // Memory and copies (always on the device).
            int n = 4099;
            float[] a = Values(n), b = Values(n);
            var da = On(hip, a);
            AssertClose(a, Read(hip, da, n), 0f, "upload and download");
            var part = new float[100];
            hip.DownloadRange(da, 1000, part);
            AssertClose(a[1000..1100], part, 0f, "download of a range");
            var zeros = Keep(hip.Allocate(n, zeroed: true));
            Check(Read(hip, zeros, n).All(v => v == 0f), "zeroed allocation");
            hip.Fill(zeros, n, 2.5f);
            Check(Read(hip, zeros, n).All(v => v == 2.5f), "fill");
            hip.Copy(da, zeros, n);
            AssertClose(a, Read(hip, zeros, n), 0f, "copy");
            long before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
            var block = Keep(hip.Allocate(64 * 40, zeroed: true));
            hip.Copy2D(da, 7, 50, block, 3, 40, 30, 33, accumulate: false);
            var expected = new float[64 * 40];
            for (int r = 0; r < 30; r++)
            {
                Array.Copy(a, 7 + r * 50, expected, 3 + r * 40, 33);
            }

            AssertClose(expected, Read(hip, block, 64 * 40), 0f, "strided copy");
            Check(Idrak.Abstraction.Operations.Kernels.HostCalls(hip) == before, "memory and copies took no host fallback");

            // The kernels against the CPU (or the host fallback, when they are unavailable: same results either way). The
            // operations on GCN kernels (rows and element-wise) are counted apart: with the kernels loaded, none of them may
            // take the host fallback; the others still do until their kernels land.
            bool kernels = ((IKernelAvailability)hip).KernelsUnavailable() is null;
            Console.WriteLine($"    {device}: kernels {(kernels ? "loaded" : $"unavailable ({hip.KernelsUnavailableReason})")}");
            long wiredFallbacks = 0;
            var wiredNames = new List<string>();
            void Wired(string what, Action run)
            {
                long start = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
                run();
                long taken = Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - start;
                wiredFallbacks += taken;
                if (taken > 0)
                {
                    wiredNames.Add(what);
                }
            }

            var db = On(hip, b);
            var ca = On(cpu, a);
            var cb = On(cpu, b);
            void Compare(string what, Action<Backend, Storage, Storage, Storage> op, float tolerance = 1e-5f)
            {
                var hy = On(hip, Values(n));
                var cy = On(cpu, Read(hip, hy, n));
                Wired(what, () => op(hip, da, db, hy));
                op(cpu, ca, cb, cy);
                AssertClose(Read(cpu, cy, n), Read(hip, hy, n), tolerance, what);
            }

            foreach (var op in new[] { BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul, BinaryOp.Maximum, BinaryOp.Minimum })
            {
                Compare($"{op}", (k, x, y, z) => k.Binary(op, x, y, z, n));
            }

            foreach (var op in new[] { UnaryOp.Relu, UnaryOp.Sigmoid, UnaryOp.Tanh, UnaryOp.Square, UnaryOp.Abs, UnaryOp.Exp, UnaryOp.Sign })
            {
                Compare($"{op}", (k, x, _, z) => k.Unary(op, x, z, n), 1e-4f);
            }

            Compare("axpy", (k, x, _, z) => k.Axpy(x, z, n, 0.75f));
            Compare("affine", (k, x, _, z) => k.Affine(x, z, n, -1.25f, 0.5f));
            Compare("multiply-add", (k, x, y, z) => k.MulAdd(x, y, z, n));
            Compare("row vector", (k, x, y, z) => k.AddRowVector(x, y, z, n / 17, 17));

            int rows = 5, cols = 515;
            float[] x2 = Values(rows * cols), gain = Values(cols);
            var hx = On(hip, x2);
            var hg = On(hip, gain);
            var hy2 = Keep(hip.Allocate(rows * cols, zeroed: true));
            var hinv = Keep(hip.Allocate(rows, zeroed: true));
            var cy2 = Keep(cpu.Allocate(rows * cols, zeroed: true));
            var cinv = Keep(cpu.Allocate(rows, zeroed: true));
            Wired("RMS norm", () => hip.RmsNorm(hx, hy2, hinv, rows, cols, 1e-6f));
            cpu.RmsNorm(On(cpu, x2), cy2, cinv, rows, cols, 1e-6f);
            AssertClose(Read(cpu, cy2, rows * cols), Read(hip, hy2, rows * cols), 1e-4f, "RMS norm");
            AssertClose(Read(cpu, cinv, rows), Read(hip, hinv, rows), 1e-4f, "RMS norm scales");
            Wired("RMS norm with gain", () => hip.RmsNormAffine(hx, hg, hy2, rows, cols, 1e-6f, 1f));
            cpu.RmsNormAffine(On(cpu, x2), On(cpu, gain), cy2, rows, cols, 1e-6f, 1f);
            AssertClose(Read(cpu, cy2, rows * cols), Read(hip, hy2, rows * cols), 1e-4f, "RMS norm with gain");

            // An int8 product with a column count that is not a multiple of four (padded weight rows).
            int m = 3, cols8 = 301, k8 = 96, stride = (cols8 + 3) / 4 * 4;
            var bytes = new byte[k8 * stride];
            random.NextBytes(bytes);
            var packed = new float[bytes.Length / 4];
            Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
            float[] x8 = Values(m * k8), scales = Values(cols8, 0.001f, 0.01f);
            var hy8 = Keep(hip.Allocate(m * cols8, zeroed: true));
            var cy8 = Keep(cpu.Allocate(m * cols8, zeroed: true));
            Wired("int8 product", () => hip.Int8MatMul(On(hip, x8), On(hip, packed), On(hip, scales), hy8, m, cols8, k8));
            cpu.Int8MatMul(On(cpu, x8), On(cpu, packed), On(cpu, scales), cy8, m, cols8, k8);
            AssertClose(Read(cpu, cy8, m * cols8), Read(hip, hy8, m * cols8), 1e-3f, "int8 product");

            // Every other kernel against the CPU: the same inputs on both, the outputs starting from the same values (for the
            // products' beta and the accumulating copies), every output compared.
            void Both(string what, float[][] inputs, int[] outputs, Action<Backend, Storage[]> op, float tolerance = 1e-4f, bool wired = false)
            {
                var hs = inputs.Select(v => On(hip, v)).Concat(outputs.Select(count => On(hip, Values(count)))).ToArray();
                var cs = inputs.Select(v => On(cpu, v)).Concat(outputs.Select((count, i) => On(cpu, Read(hip, hs[inputs.Length + i], count)))).ToArray();
                if (wired)
                {
                    Wired(what, () => op(hip, hs));
                }
                else
                {
                    op(hip, hs);
                }

                op(cpu, cs);
                for (int i = 0; i < outputs.Length; i++)
                {
                    AssertClose(Read(cpu, cs[inputs.Length + i], outputs[i]), Read(hip, hs[inputs.Length + i], outputs[i]), tolerance, $"{what} (output {i})");
                }
            }

            // Float32 products: few rows (as stored and transposed, with a bias), the tiled product in every transposition
            // with beta, batches.
            foreach (var (pm, pn, pk) in new[] { (3, 300, 257), (70, 45, 33), (1, 513, 129) })
            {
                foreach (bool ta in new[] { false, true })
                {
                    foreach (bool tb in new[] { false, true })
                    {
                        float beta = ta ? 1f : 0f;
                        Both($"product {pm}x{pn}x{pk} transA={ta} transB={tb}", [Values(pm * pk), Values(pk * pn)], [pm * pn],
                            (k, s) => k.MatMul(s[0], s[1], s[2], pm, pn, pk, ta, tb, beta), 1e-3f, wired: true);
                    }
                }
            }

            Both("batched product", [Values(3 * 17 * 5), Values(3 * 5 * 66)], [3 * 17 * 66], (k, s) => k.BatchedMatMul(s[0], s[1], s[2], 3, 17, 66, 5, false, true, 0f), wired: true);
            Both("product with a bias", [Values(4 * 300), Values(300 * 260), Values(260)], [4 * 260], (k, s) =>
            {
                if (!k.MatMulBias(s[0], s[1], s[2], s[3], 4, 260, 300))
                {
                    k.MatMul(s[0], s[1], s[3], 4, 260, 300, false, false, 0f);
                    k.AddRowVector(s[3], s[2], s[3], 4, 260);
                }
            }, 1e-3f, wired: true);

            // Packed products of few rows and the expansions (bfloat16 packed on the CPU; 4-bit words of random nibbles).
            int pr = 2, pcols = 300, pdepth = 70;
            var packedRows = new float[pdepth * pcols / 2];                          // an even width: rows of whole words
            var packedStorage = On(cpu, packedRows);
            cpu.PackBFloat16(On(cpu, Values(pdepth * pcols)), packedStorage, pdepth * pcols);
            cpu.Download(packedStorage, packedRows);
            Both("bfloat16 product", [Values(pr * pdepth), packedRows], [pr * pcols], (k, s) => k.BFloat16MatMul(s[0], s[1], s[2], pr, pcols, pdepth), 1e-3f, wired: true);
            Both("bfloat16 expansion", [packedRows], [pdepth * pcols], (k, s) => k.BFloat16Dequantize(s[0], s[1], pdepth, pcols), 0f, wired: true);
            Both("bfloat16 packing", [Values(301)], [151], (k, s) => k.PackBFloat16(s[0], s[1], 301), 0f, wired: true);
            Both("float16 product", [Values(pr * pdepth), packedRows], [pr * pcols], (k, s) => k.Float16MatMul(s[0], s[1], s[2], pr, pcols, pdepth), 1e-3f, wired: true);
            Both("float16 expansion", [packedRows], [pdepth * pcols], (k, s) => k.Float16Dequantize(s[0], s[1], pdepth, pcols), 0f, wired: true);
            Both("bfloat16 product of 11 rows (expanded)", [Values(11 * pdepth), packedRows], [11 * pcols], (k, s) => k.BFloat16MatMul(s[0], s[1], s[2], 11, pcols, pdepth), 1e-3f, wired: true);
            Both("int8 product of 11 rows", [Values(11 * k8), packed, scales], [11 * cols8], (k, s) => k.Int8MatMul(s[0], s[1], s[2], s[3], 11, cols8, k8), 1e-3f, wired: true);
            Both("int8 expansion", [packed, scales], [k8 * cols8], (k, s) => k.Int8Dequantize(s[0], s[1], s[2], k8, cols8), 0f, wired: true);
            int int4Words = (pcols + 7) / 8;                                           // rows of ⌈n / 8⌉ words, scale rows of 8 · that
            var nibbles = new byte[pdepth * int4Words * 4];
            random.NextBytes(nibbles);
            var int4 = new float[pdepth * int4Words];
            Buffer.BlockCopy(nibbles, 0, int4, 0, nibbles.Length);
            float[] int4Scales = Values((pdepth + 31) / 32 * int4Words * 8, 0.001f, 0.01f);
            Both("4-bit product", [Values(pr * pdepth), int4, int4Scales], [pr * pcols], (k, s) => k.Int4MatMul(s[0], s[1], s[2], s[3], pr, pcols, pdepth), 1e-3f, wired: true);
            Both("4-bit expansion", [int4, int4Scales], [pdepth * pcols], (k, s) => k.Int4Dequantize(s[0], s[1], s[2], pdepth, pcols), 0f, wired: true);

            // Rows: softmax and log-softmax, the masked softmax, layer norm with its statistics, GELU after a bias.
            int sr = 5, sc = 513;
            float[] logits = Values(sr * sc, -6f, 6f), mask = [.. Enumerable.Range(0, 2 * sc).Select(i => i % 4 == 0 ? -1e9f : random.NextSingle())];
            Both("softmax", [logits], [sr * sc], (k, s) => k.Softmax(s[0], s[1], sr, sc, false), wired: true);
            Both("log-softmax", [logits], [sr * sc], (k, s) => k.Softmax(s[0], s[1], sr, sc, true), wired: true);
            Both("masked softmax", [logits, mask], [sr * sc], (k, s) => k.ScaleMaskSoftmax(s[0], s[1], s[2], sr, sc, 2, 0.5f), wired: true);
            Both("scaled softmax without a mask", [logits], [sr * sc], (k, s) => k.ScaleMaskSoftmax(s[0], null, s[1], sr, sc, 1, 0.5f), wired: true);
            Both("layer norm, training", [logits, Values(sc), Values(sc)], [sr * sc, 2 * sr], (k, s) => k.LayerNormTrain(s[0], s[1], s[2], s[3], s[4], sr, sc, 1e-5f), wired: true);
            Both("layer norm", [logits, Values(sc), Values(sc)], [sr * sc], (k, s) => k.LayerNormFused(s[0], s[1], s[2], s[3], sr, sc, 1e-5f), wired: true);

            // Layer norm's gradients over one chunk of rows and over three (partials folded in chunk order), dgamma alone too.
            foreach (int lr in new[] { 5, 130 })
            {
                int lc = 300;
                float[] lx = Values(lr * lc), lgamma = Values(lc), lbeta = Values(lc), ldy = Values(lr * lc);
                var cstats = Keep(cpu.Allocate(2 * lr, zeroed: true));
                cpu.LayerNormTrain(On(cpu, lx), On(cpu, lgamma), On(cpu, lbeta), Keep(cpu.Allocate(lr * lc, zeroed: true)), cstats, lr, lc, 1e-5f);
                float[] stats = Read(cpu, cstats, 2 * lr);
                Both($"layer norm gradients, {lr} rows", [lx, lgamma, ldy, stats], [lr * lc, lc, lc],
                    (k, s) => k.LayerNormBackward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], lr, lc), 1e-3f, wired: true);
                Both($"layer norm gradients, {lr} rows, dgamma alone", [lx, lgamma, ldy, stats], [lc],
                    (k, s) => k.LayerNormBackward(s[0], s[1], s[2], s[3], null, s[4], null, lr, lc), 1e-3f, wired: true);
            }

            Both("GELU after a bias", [logits, Values(sc)], [sr * sc], (k, s) => k.BiasGelu(s[0], s[1], s[2], sr * sc, sc), wired: true);
            Both("best k log-probabilities", [logits], [sr * 8, sr * 8], (k, s) => k.LogSoftmaxTopK(s[0], s[1], s[2], sr, sc, 8), wired: true);
            foreach (var op in new[] { UnaryOp.Gelu, UnaryOp.GeluErf, UnaryOp.Silu, UnaryOp.Sqrt, UnaryOp.Sin, UnaryOp.Cos, UnaryOp.Sign, UnaryOp.Log })
            {
                Both($"{op}", [Values(1000, 0.01f, 4f)], [1000], (k, s) => k.Unary(op, s[0], s[1], 1000), wired: true);
            }

            // Attention over key ranges (grouped heads, two tables; with the log-sum-exp, then with an empty range, whose
            // log-sum-exp is -∞; a soft-cap; enough keys to split them), its strided form over packed rows, and over a cache up to
            // the device position (decoding and the tiled form with the log-sum-exp; a window, a soft-cap; bfloat16 and int8
            // caches), all on the GCN kernels.
            int heads = 4, kvHeads = 2, rowsA = 9, keyRows = 37, dimA = 64;
            float[] q = Values(heads * rowsA * dimA), keysA = Values(kvHeads * keyRows * dimA), valuesA = Values(kvHeads * keyRows * dimA);
            float[] starts = [.. Enumerable.Range(0, 2 * rowsA).Select(i => (float)(i % 3))], ends = [.. Enumerable.Range(0, 2 * rowsA).Select(i => (float)(keyRows - i))];
            Both("attention over key ranges", [q, keysA, valuesA, starts, ends], [heads * rowsA * dimA, heads * rowsA],
                (k, s) => k.AttentionSpans(s[0], s[1], s[2], s[3], s[4], s[5], s[6], heads, kvHeads, heads / 2, rowsA, keyRows, dimA, 0.125f), wired: true);
            ends[4] = 1f;                                                                // row 4 of the first table: from key 1 to key 1
            Both("attention over key ranges, an empty one", [q, keysA, valuesA, starts, ends], [heads * rowsA * dimA],
                (k, s) => k.AttentionSpans(s[0], s[1], s[2], s[3], s[4], s[5], null, heads, kvHeads, heads / 2, rowsA, keyRows, dimA, 0.125f), wired: true);
            Both("attention over key ranges, soft-capped", [q, keysA, valuesA, starts, ends], [heads * rowsA * dimA],
                (k, s) => k.AttentionSpans(s[0], s[1], s[2], s[3], s[4], s[5], null, heads, kvHeads, heads / 2, rowsA, keyRows, dimA, 0.125f, new AttentionVariant(0, 5f)), wired: true);
            int longKeys = 1500;
            float[] spanStarts = [.. Enumerable.Range(0, 2).Select(i => (float)(3 * i))], spanEnds = [.. Enumerable.Range(0, 2).Select(i => (float)(longKeys - i))];
            Both("attention over key ranges, splits merged", [Values(2 * dimA), Values(2 * longKeys * dimA), Values(2 * longKeys * dimA), spanStarts, spanEnds],
                [2 * dimA, 2], (k, s) => k.AttentionSpans(s[0], s[1], s[2], s[3], s[4], s[5], s[6], 2, 2, 1, 1, longKeys, dimA, 0.125f), 1e-3f, wired: true);
            int sb = 2, sh = 2, skv = 1, sRows = 7, sd = 24, sPitch = (sh + 2 * skv) * sd;
            Both("attention over key ranges, strided", [Values(sb * sRows * sPitch), [.. Enumerable.Range(0, sb * sh * sRows).Select(i => (float)Math.Max(0, i % sRows - 2))],
                    [.. Enumerable.Range(0, sb * sh * sRows).Select(i => (float)(i % sRows + 1))]], [sb * sh * sRows * sd],
                (k, s) => k.AttentionSpansStrided(s[0], 0, s[0], sh * sd, s[0], (sh + skv) * sd, sPitch, sPitch, s[1], s[2], s[3], sb, sh, skv, 1, sRows, sd, 0.2f), wired: true);
            foreach (int rowsPerHead in new[] { 1, 12 })
            {
                Both($"attention over a cache, {rowsPerHead} rows", [Values(heads * rowsPerHead * dimA), Values(heads * 50 * dimA), Values(heads * 50 * dimA), [6f]],
                    [heads * rowsPerHead * dimA], (k, s) => k.AttentionDecode(s[0], s[1], s[2], s[3], s[4], heads, rowsPerHead, Math.Min(rowsPerHead, 4), 50, dimA, 0.125f),
                    wired: true);
            }

            Both("tiled attention over a cache with the log-sum-exp, a window and a soft-cap", [Values(heads * 8 * dimA), Values(heads * 50 * dimA), Values(heads * 50 * dimA), [20f]],
                [heads * 8 * dimA, heads * 8], (k, s) => k.AttentionTiled(s[0], s[1], s[2], s[3], s[4], s[5], heads, 8, 8, 50, dimA, 0.125f, new AttentionVariant(9, 4f)),
                wired: true);
            Both("decoding over a long cache, splits merged", [Values(heads * dimA), Values(heads * 2000 * dimA), Values(heads * 2000 * dimA), [1990f]],
                [heads * dimA], (k, s) => k.AttentionDecode(s[0], s[1], s[2], s[3], s[4], heads, 1, 1, 2000, dimA, 0.125f), 1e-3f, wired: true);
            int dimP = 37, capP = 40;
            var bf16Cache = Keep(cpu.Allocate(heads * capP * ((dimP + 1) / 2), zeroed: true));
            cpu.KeyValueWriteBFloat16(On(cpu, Values(heads * capP * dimP)), bf16Cache, On(cpu, [0f]), heads, capP, capP, dimP);
            float[] bf16Keys = Read(cpu, bf16Cache, heads * capP * ((dimP + 1) / 2));
            Both("attention over a bfloat16 cache", [Values(heads * 3 * dimP), bf16Keys, bf16Keys, [30f]], [heads * 3 * dimP],
                (k, s) => k.AttentionBFloat16(s[0], s[1], s[2], s[3], s[4], heads, 3, 3, capP, dimP, 0.15f, tiled: false), wired: true);
            var int8Cache = Keep(cpu.Allocate(heads * capP * ((dimP + 3) / 4), zeroed: true));
            var int8Scales = Keep(cpu.Allocate(heads * capP, zeroed: true));
            cpu.KeyValueWriteInt8(On(cpu, Values(heads * capP * dimP)), int8Cache, int8Scales, On(cpu, [0f]), heads, capP, capP, dimP);
            float[] int8Keys = Read(cpu, int8Cache, heads * capP * ((dimP + 3) / 4)), int8KeyScales = Read(cpu, int8Scales, heads * capP);
            Both("attention over an int8 cache", [Values(heads * 2 * dimP), int8Keys, int8Keys, int8KeyScales, int8KeyScales, [30f]], [heads * 2 * dimP],
                (k, s) => k.AttentionInt8(s[0], s[1], s[2], s[3], s[4], s[5], s[6], heads, 2, 2, capP, dimP, 0.15f, tiled: false, new AttentionVariant(16, 0f)), wired: true);
            Both("int8 cache scores", [Values(2 * 3 * dimP), int8Keys, int8KeyScales], [2 * 3 * capP],
                (k, s) => k.AttentionScoresInt8(s[0], s[1], s[2], s[3], 2, 3, capP, dimP), wired: true);
            Both("int8 cache weighted values", [Values(2 * 3 * capP, 0f, 0.1f), int8Keys, int8KeyScales], [2 * 3 * dimP],
                (k, s) => k.AttentionContextInt8(s[0], s[1], s[2], s[3], 2, 3, capP, dimP), wired: true);

            // Gathers, the cache writes and mask, permutes and an accumulating strided copy, on the GCN kernels.
            float[] ids = [0f, 5f, 36f, 2f];
            Both("gather", [Values(37 * 13), ids], [4 * 13], (k, s) => k.Gather(s[0], s[1], s[2], 4, 13, 37), 0f, wired: true);
            float[] bf16Table = Read(cpu, bf16Cache, 37 * 7);                          // 37 rows of 13 halves: 7 words each
            Both("bfloat16 gather", [bf16Table, ids], [4 * 13], (k, s) => k.GatherBFloat16(s[0], s[1], s[2], 4, 13, 37), 0f, wired: true);
            Both("bfloat16 column gather", [bf16Table, ids], [4 * 7], (k, s) => k.GatherBFloat16Columns(s[0], s[1], s[2], 4, 7, 37), 0f, wired: true);
            Both("cache write", [Values(heads * 2 * dimA), [6f]], [heads * 50 * dimA], (k, s) => k.KeyValueWrite(s[0], s[2], s[1], heads, 2, 50, dimA), 0f, wired: true);
            Both("bfloat16 cache write", [Values(heads * 2 * dimP), [6f]], [heads * capP * ((dimP + 1) / 2)],
                (k, s) => k.KeyValueWriteBFloat16(s[0], s[2], s[1], heads, 2, capP, dimP), 0f, wired: true);
            Both("int8 cache write", [Values(heads * 2 * dimP), [6f]], [heads * capP * ((dimP + 3) / 4), heads * capP],
                (k, s) => k.KeyValueWriteInt8(s[0], s[2], s[3], s[1], heads, 2, capP, dimP), 0f, wired: true);
            Both("decoder mask", [[6f]], [3 * 50], (k, s) => k.DecoderMask(s[0], s[1], 3, 50), 0f, wired: true);
            Both("permute", [Values(2 * 3 * 4 * 5)], [120], (k, s) => k.Permute(s[0], s[1], [2, 4, 5, 3], [60, 5, 1, 20], accumulate: false), 0f, wired: true);
            Both("permute, adding", [Values(2 * 3 * 4 * 5)], [120], (k, s) => k.Permute(s[0], s[1], [2, 4, 5, 3], [60, 5, 1, 20], accumulate: true), wired: true);
            Both("strided copy, adding", [Values(200)], [200], (k, s) => k.Copy2D(s[0], 5, 13, s[1], 3, 20, 6, 9, accumulate: true), wired: true);

            // G5a: the element-wise operations with parameters, the rows of decoding and the decoding contracts.
            float[] mixed = Values(1000, -3f, 3f), condition = [.. Values(1000).Select((v, i) => i % 3 == 0 ? 0f : v)];
            Both("power", [Values(1000, 0.1f, 3f)], [1000], (k, s) => k.Pow(s[0], s[1], 1000, 1.5f), 1e-3f, wired: true);
            Both("cube of negative bases", [mixed], [1000], (k, s) => k.Pow(s[0], s[1], 1000, 3f), 1e-3f, wired: true);
            Both("clamp", [mixed], [1000], (k, s) => k.Clamp(s[0], s[1], 1000, -0.5f, 0.75f), 0f, wired: true);
            Both("leaky ReLU", [mixed], [1000], (k, s) => k.LeakyRelu(s[0], s[1], 1000, 0.1f), 0f, wired: true);
            Both("selection", [condition, mixed, Values(1000)], [1000], (k, s) => k.Where(s[0], s[1], s[2], s[3], 1000), 0f, wired: true);
            Both("inverse square root", [Values(1000, 0.1f, 4f)], [1000], (k, s) => k.InvSqrt(s[0], s[1], 1000, 1e-5f), wired: true);
            Both("a scalar added to every element", [[0.3f]], [1000], (k, s) => k.AddBroadcastScalar(s[0], s[1], 1000, -2f), wired: true);
            Both("a scalar added to one element", [[0.3f]], [1000], (k, s) => k.AxpyAt(s[0], s[1], 17, 0.5f), wired: true);
            for (int kind = 0; kind < 4; kind++)
            {
                int activation = kind;
                Both($"gated activation {kind}", [mixed, Values(1000)], [1000], (k, s) => k.GatedActivation(s[0], s[1], s[2], 1000, activation), wired: true);
            }

            float[] rowsX = Values(5 * 513, -4f, 4f);
            rowsX[513 + 7] = rowsX[513 + 9] = 50f;                                        // a tie: the first column
            Both("arg-max", [rowsX], [5], (k, s) => k.ArgMax(s[0], s[1], 5, 513), 0f, wired: true);
            Both("class match", [rowsX, Values(5 * 513)], [5], (k, s) => k.ClassMatch(s[0], s[1], s[2], 5, 513, 0.5f), 0f, wired: true);
            Both("residual addition and RMS norm", [rowsX, Values(5 * 513), Values(513)], [5 * 513, 5 * 513],
                (k, s) => k.AddRmsNormAffine(s[0], s[1], s[3], s[2], s[4], 5, 513, 1e-6f, 1f), wired: true);
            float[] ropeX = Values(6 * 40), ropeCos = Values(10 * 16), ropeSin = Values(10 * 16);
            Both("rotary embedding", [ropeX, ropeCos, ropeSin, [5f, 7f]], [6 * 40], (k, s) => k.Rope(s[0], s[4], s[1], s[2], s[3], 6, 3, 2, 40, 16, false, -1f), wired: true);
            Both("RMS norm and rotary embedding", [ropeX, Values(40), ropeCos, ropeSin, [5f, 7f]], [6 * 40],
                (k, s) => k.RmsNormRope(s[0], s[1], s[2], s[3], s[4], s[5], 6, 40, 1e-6f, 0f, 3, 2, 16, true), wired: true);
            Both("one-hot", [[0f, 12f, 5f, 3f]], [4 * 13], (k, s) => k.OneHot(s[0], s[1], 4, 13), 0f, wired: true);
            Both("history push", [[17f, 4f], [12f]], [2 * 5], (k, s) => k.HistoryPush(s[0], s[2], s[1], 2, 5), 0f, wired: true);
            Both("repetition penalties", [Values(2 * 310, -4f, 4f), [12, 40, 299, 12, 350, 7, 40, 12, 3, -1, 3, 3, 250, 0, 9, 3], [11f]], [2 * 300],
                (k, s) => k.PenalizeRows(s[0], s[3], s[1], s[2], 2, 300, 310, 5, 8, 6, 1.3f, 0.2f, 0.1f), wired: true);

            // Rows of different lengths over float32 and bfloat16 caches, and packed sequences (IRowAttentionBackend,
            // ISegmentedAttentionBackend): on the kernels, never the host.
            float[] rowStarts = [0f, 5f, 40f, 2f, 31f, 33f];
            Both("attention over rows of different lengths", [Values(heads * 3 * dimP), Values(heads * capP * dimP), Values(heads * capP * dimP), [30f], rowStarts],
                [heads * 3 * dimP], (k, s) => k.AttentionRows(s[0], s[1], s[2], s[3], s[5], s[4], heads, 2, 3, 3, capP, dimP, 0.15f, new AttentionVariant(9, 4f)), wired: true);
            Both("attention over rows of different lengths, a bfloat16 cache", [Values(heads * 3 * dimP), bf16Keys, bf16Keys, [30f], rowStarts], [heads * 3 * dimP],
                (k, s) => k.AttentionRowsBFloat16(s[0], s[1], s[2], s[3], s[5], s[4], heads, 2, 3, 3, capP, dimP, 0.15f), wired: true);
            float[] packedStarts = [0, 0, 0, 3, 3, 3, 3, 7, 7, 0, 0, 2, 2, 2, 2, 2, 2, 2], packedEnds = [3, 3, 3, 7, 7, 7, 7, 9, 9, 2, 2, 9, 9, 9, 9, 9, 9, 9];
            Both("attention over packed sequences", [Values(heads * 9 * 24), Values(heads * 9 * 24), Values(heads * 9 * 24), packedStarts, packedEnds], [heads * 9 * 24, heads * 9],
                (k, s) => k.AttentionSegmented(s[0], s[1], s[2], s[5], s[6], s[3], s[4], heads, 2, 9, 9, 24, 0.2f, new AttentionVariant(3, 5f)), wired: true);
            Check(hip.SupportsAttentionRows(dimP) && hip.SupportsSegmentedAttention(dimP), "HIP answers for rows and packed sequences as the CPU does");

            // The fused packed products (IFusedPackedKernels) against the CPU's composed operations: on HIP's kernels they run
            // (true) and take no host fallback.
            Check(hip.FusedKernels, "HIP implements the fused packed products");
            void PackedProduct(Backend b, PackedFormat f, Storage x, Storage w, Storage? sc, Storage y, int rows, int cols, int depth)
            {
                switch (f)
                {
                    case PackedFormat.Int8:
                        b.Int8MatMul(x, w, sc!, y, rows, cols, depth);
                        break;
                    case PackedFormat.Int4:
                        b.Int4MatMul(x, w, sc!, y, rows, cols, depth);
                        break;
                    default:
                        b.BFloat16MatMul(x, w, y, rows, cols, depth);
                        break;
                }
            }

            foreach (var (format, weightsOf, scalesOf, depth, width) in new (PackedFormat, float[], float[]?, int, int)[]
                     {
                         (PackedFormat.Int8, packed, scales, k8, cols8), (PackedFormat.Int4, int4, int4Scales, pdepth, pcols), (PackedFormat.BFloat16, packedRows, null, pdepth, pcols),
                     })
            {
                const int Rows = 3;
                float[] fx = Values(Rows * depth), fup = Values(Rows * depth), residual = Values(Rows * width), fgain = Values(width), bias = Values(width);
                float[][] Inputs() => scalesOf is null ? [fx, fup, weightsOf, [0f], residual, fgain, bias] : [fx, fup, weightsOf, scalesOf, residual, fgain, bias];
                Storage? Scales(Storage[] s) => scalesOf is null ? null : s[3];
                bool fused = true;
                void Fused(Backend b, bool done)
                {
                    if (ReferenceEquals(b, hip))
                    {
                        fused &= done || !kernels;
                    }
                }

                Both($"{format} gated product", Inputs(), [Rows * width], (k, s) =>
                {
                    if (ReferenceEquals(k, hip))
                    {
                        Fused(k, k.PackedMatMulGated(format, 0, s[0], s[1], s[2], Scales(s), s[7], Rows, width, depth));
                        return;
                    }

                    var hidden = k.Allocate(Rows * depth, zeroed: false);
                    k.GatedActivation(s[0], s[1], hidden, Rows * depth, 0);
                    PackedProduct(k, format, hidden, s[2], Scales(s), s[7], Rows, width, depth);
                    hidden.Release();
                }, 1e-3f, wired: true);
                Both($"{format} product, residual addition and norm", Inputs(), [Rows * width, Rows * width, Rows * width], (k, s) =>
                {
                    if (ReferenceEquals(k, hip))
                    {
                        Fused(k, k.PackedMatMulAddRmsNorm(format, s[0], s[2], Scales(s), s[7], Rows, width, depth, s[4], s[8], s[5], s[9], 1e-6f, 1f));
                        return;
                    }

                    PackedProduct(k, format, s[0], s[2], Scales(s), s[7], Rows, width, depth);
                    k.AddRmsNormAffine(s[4], s[7], s[8], s[5], s[9], Rows, width, 1e-6f, 1f);
                }, 1e-3f, wired: true);
                Both($"{format} product, its norm, residual addition and norm", Inputs(), [Rows * width, Rows * width, Rows * width],
                    (k, s) => Fused(k, k.PackedMatMulNormAddRmsNorm(format, s[0], s[2], Scales(s), s[7], Rows, width, depth, s[6], 1e-6f, 0f, s[4], s[8], s[5], s[9], 1e-6f, 1f)),
                    1e-3f, wired: true);
                Both($"{format} two products of one input, one with a bias", Inputs(), [Rows * width, Rows * width], (k, s) =>
                {
                    if (ReferenceEquals(k, hip))
                    {
                        Fused(k, k.PackedMatMulMany(format, s[0], Rows, depth, [(s[2], Scales(s), s[6], s[7], width), (s[2], Scales(s), null, s[8], width)]));
                        return;
                    }

                    PackedProduct(k, format, s[0], s[2], Scales(s), s[7], Rows, width, depth);
                    k.AddRowVector(s[7], s[6], s[7], Rows, width);
                    PackedProduct(k, format, s[0], s[2], Scales(s), s[8], Rows, width, depth);
                }, 1e-3f, wired: true);
                Both($"{format} gate/up pair with GELU", Inputs(), [Rows * width, Rows * width, Rows * width], (k, s) =>
                {
                    if (ReferenceEquals(k, hip))
                    {
                        Fused(k, k.PackedMatMulGatedPair(format, 1, s[0], Rows, depth, [(s[2], Scales(s), null, s[7], width), (s[2], Scales(s), null, s[8], width)], s[9]));
                        return;
                    }

                    PackedProduct(k, format, s[0], s[2], Scales(s), s[7], Rows, width, depth);
                    PackedProduct(k, format, s[0], s[2], Scales(s), s[8], Rows, width, depth);
                    k.GatedActivation(s[7], s[8], s[9], Rows * width, 1);
                }, 1e-3f, wired: true);
                Check(fused, $"{format}: with kernels, every fused packed product ran on HIP's kernels (none declined)");
            }
            // Training (plan 35, G5b): the backward passes, the optimizers and the reductions on the GCN kernels.
            int tn = 3001;
            float[] tx = Values(tn), tdy = Values(tn), tpos = Values(tn, 0.1f, 3f);
            foreach (var op in new[] { UnaryOp.Sigmoid, UnaryOp.Tanh, UnaryOp.Relu, UnaryOp.Square, UnaryOp.Abs, UnaryOp.Exp, UnaryOp.Gelu, UnaryOp.GeluErf, UnaryOp.Sin, UnaryOp.Cos, UnaryOp.Silu })
            {
                var cyo = Keep(cpu.Allocate(tn, zeroed: true));
                cpu.Unary(op, On(cpu, tx), cyo, tn);
                Both($"{op} gradient", [tx, Read(cpu, cyo, tn), tdy], [tn], (k, s) => k.UnaryBackward(op, s[0], s[1], s[2], s[3], tn), wired: true);
            }

            foreach (var op in new[] { UnaryOp.Log, UnaryOp.Sqrt })
            {
                var cyo = Keep(cpu.Allocate(tn, zeroed: true));
                cpu.Unary(op, On(cpu, tpos), cyo, tn);
                Both($"{op} gradient", [tpos, Read(cpu, cyo, tn), tdy], [tn], (k, s) => k.UnaryBackward(op, s[0], s[1], s[2], s[3], tn), wired: true);
            }

            Both("maximum gradient", [tx, tdy], [tn, tn], (k, s) => k.ExtremumBackward(BinaryOp.Maximum, s[0], s[1], s[1], s[2], s[3], tn), 0f, wired: true);
            Both("minimum gradient, a alone", [tx, tdy], [tn], (k, s) => k.ExtremumBackward(BinaryOp.Minimum, s[0], s[1], s[1], s[2], null, tn), 0f, wired: true);
            Both("power gradient", [tpos, tdy], [tn], (k, s) => k.PowBackward(s[0], s[1], s[2], tn, 2.5f), 1e-3f, wired: true);
            Both("clamp gradient", [tx, tdy], [tn], (k, s) => k.ClampBackward(s[0], s[1], s[2], tn, -1f, 0.5f), 0f, wired: true);
            Both("leaky ReLU gradient", [tx, tdy], [tn], (k, s) => k.LeakyReluBackward(s[0], s[1], s[2], tn, 0.1f), 0f, wired: true);
            Both("selection gradient", [tx, tdy], [tn, tn], (k, s) => k.WhereBackward(s[0], s[1], s[2], s[3], tn), 0f, wired: true);
            foreach (int kind in new[] { 0, 1, 2, 3 })
            {
                Both($"gated activation gradient, kind {kind}", [tx, tdy, Values(tn)], [tn, tn], (k, s) => k.GatedActivationBackward(s[0], s[1], s[2], s[3], s[4], tn, kind, 3), 1e-3f, wired: true);
            }

            Both("softmax gradient", [Values(sr * sc, 0f, 0.01f), Values(sr * sc)], [sr * sc], (k, s) => k.SoftmaxBackward(s[0], s[1], s[2], sr, sc, false), wired: true);
            Both("log-softmax gradient", [Values(sr * sc, -6f, -1f), Values(sr * sc)], [sr * sc], (k, s) => k.SoftmaxBackward(s[0], s[1], s[2], sr, sc, true), 1e-3f, wired: true);
            Both("RMS norm gradient", [Values(sr * sc), Values(sr * sc), Values(sr, 0.5f, 2f)], [sr * sc], (k, s) => k.RmsNormBackward(s[0], s[1], s[2], s[3], sr, sc), 1e-3f, wired: true);
            Both("SGD with momentum", [Values(tn)], [tn, tn], (k, s) => k.SgdStep(s[1], s[0], s[2], tn, 0.1f, 0.9f), wired: true);
            Both("Adam", [Values(tn), Values(tn, 0f, 0.01f)], [tn, tn], (k, s) => k.AdamStep(s[2], s[0], s[3], s[1], tn, 1e-3f, 0.9f, 0.999f, 1e-8f), 1e-3f, wired: true);
            Both("dropout", [tx], [tn], (k, s) => k.Dropout(s[0], s[1], tn, 0.3f, 99u), 0f, wired: true);
            Both("dropout gradient", [tx], [tn], (k, s) => k.DropoutBackward(s[0], s[1], tn, 0.3f, 99u), 0f, wired: true);
            Both("residual and dropout", [tx, tdy], [tn], (k, s) => k.AddDropout(s[0], s[1], s[2], tn, 0.3f, 99u), 0f, wired: true);
            Both("clipping factor", [[16f]], [1], (k, s) => k.ClipFactor(s[0], s[1], 2f), wired: true);
            Both("sums of rows", [Values(7 * 300)], [300], (k, s) => k.SumRows(s[0], s[1], 7, 300), wired: true);
            Both("sums of 300 rows (chunks folded)", [Values(300 * 77)], [77], (k, s) => k.SumRows(s[0], s[1], 300, 77), 1e-3f, wired: true);
            Both("sum", [Values(70_000)], [1], (k, s) => k.Sum(s[0], s[1], 70_000, 0.5f), 1e-3f, wired: true);
            Both("sum of squares", [Values(70_000)], [1], (k, s) => k.SumSquares(s[0], s[1], 70_000), 1e-3f, wired: true);
            Both("sum along an axis", [Values(3 * 9 * 13)], [3 * 13], (k, s) => k.SumAxis(s[0], s[1], 3, 9, 13, 0.5f, false), 1e-3f, wired: true);
            Both("sum along the last axis", [Values(4 * 300)], [4], (k, s) => k.SumAxis(s[0], s[1], 4, 300, 1, 1f, true), 1e-3f, wired: true);
            Both("broadcast along an axis", [Values(3 * 13)], [3 * 9 * 13], (k, s) => k.BroadcastAxis(s[0], s[1], 3, 9, 13, 0.5f), wired: true);
            float[] embedIds = [.. Enumerable.Range(0, 300).Select(i => (float)(i * 7 % 37))];
            Both("scatter-add", [Values(300 * 13), embedIds], [37 * 13], (k, s) => k.ScatterAdd(s[0], s[1], s[2], 300, 13, 37), 1e-3f, wired: true);
            foreach (var (no, ng, ni) in new[] { (3, 4, 50), (2, 2, 9000) })
            {
                int gn = no * ng * ni;
                Both($"group statistics {no}x{ng}x{ni}", [Values(gn)], [ng, ng, ng], (k, s) => k.NormStats(s[0], s[1], s[2], s[3], no, ng, ni, 1e-5f), 1e-3f, wired: true);
                Both($"group sums {no}x{ng}x{ni}", [Values(gn), Values(gn)], [ng, ng], (k, s) => k.GroupReduce(s[0], s[1], s[2], s[3], no, ng, ni), 1e-3f, wired: true);
                Both($"normalization {no}x{ng}x{ni}", [Values(gn), Values(ng), Values(ng, 0.5f, 2f)], [gn], (k, s) => k.NormApply(s[0], s[1], s[2], s[3], no, ng, ni), 1e-3f, wired: true);
                Both($"normalization gradient {no}x{ng}x{ni}", [Values(gn), Values(gn), Values(ng), Values(ng), Values(ng, 0.5f, 2f)], [gn],
                    (k, s) => k.NormBackward(s[0], s[1], s[2], s[3], s[4], s[5], no, ng, ni), 1e-3f, wired: true);
                Both($"group scale and shift {no}x{ng}x{ni}", [Values(gn), Values(ng), Values(ng)], [gn], (k, s) => k.GroupScaleShift(s[0], s[1], s[2], s[3], gn, ng, ni, true), 1e-3f, wired: true);
            }

            Both("weight norm", [Values(5, 0.5f, 2f), Values(5 * 300)], [5 * 300], (k, s) => k.WeightNorm(s[0], s[1], s[2], 5, 300), 1e-3f, wired: true);
            Both("weight norm gradient", [Values(5, 0.5f, 2f), Values(5 * 300), Values(5 * 300)], [5, 5 * 300],
                (k, s) => k.WeightNormBackward(s[0], s[1], s[2], s[3], s[4], 5, 300), 1e-3f, wired: true);
            var codes = new byte[2 * 700];
            random.NextBytes(codes);
            float[] moments = new float[2 * 175];
            Buffer.BlockCopy(codes, 0, moments, 0, codes.Length);
            float[] map = [.. Enumerable.Range(0, 256).Select(i => MathF.Pow((i - 127.5f) / 127.5f, 3)), .. Enumerable.Range(0, 256).Select(i => i * i / 65025f)];
            Both("8-bit Adam", [Values(700), moments[..175], moments[175..], Values(6, 0.01f, 0.1f), map], [700],
                (k, s) => k.AdamStep8Bit(s[5], s[0], s[1], s[2], s[3], s[4], 700, 1e-3f, 0.9f, 0.999f, 1e-8f, 1f, 1f), 1e-3f, wired: true);
            Both("fused AdamW", [Values(5000), Values(300), Values(5000, 0f, 0.01f), Values(300, 0f, 0.01f)], [5000, 5000, 300, 300], (k, s) =>
            {
                IDisposable? fused = null;
                var tensors = new (Storage, Storage, Storage, Storage, int)[] { (s[4], s[0], s[5], s[2], 5000), (s[6], s[1], s[7], s[3], 300) };
                if (!k.FusedAdamW(tensors, ref fused, 1f, 1e-3f, 0.99f, 0.9f, 0.999f, 1e-8f, false))
                {
                    throw new InvalidOperationException("no fused AdamW");
                }

                fused?.Dispose();
            }, 1e-3f, wired: true);

            // The attention gradients: tiled over a cache (a window and a soft-cap) and over key ranges (grouped heads).
            float[] aq = Values(heads * 8 * dimA), ak = Values(heads * 50 * dimA), av = Values(heads * 50 * dimA), ady = Values(heads * 8 * dimA);
            var cOut = Keep(cpu.Allocate(heads * 8 * dimA, zeroed: true));
            var cLse = Keep(cpu.Allocate(heads * 8, zeroed: true));
            var tiled = new AttentionVariant(5, 4f);
            cpu.AttentionTiled(On(cpu, aq), On(cpu, ak), On(cpu, av), On(cpu, [0f]), cOut, cLse, heads, 8, 8, 50, dimA, 0.125f, tiled);
            Both("tiled attention gradients", [aq, ak, av, Read(cpu, cOut, heads * 8 * dimA), Read(cpu, cLse, heads * 8), ady],
                [heads * 8 * dimA, heads * 50 * dimA, heads * 50 * dimA],
                (k, s) => k.AttentionTiledBackward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], heads, 8, 8, 50, dimA, 0.125f, tiled), 1e-3f, wired: true);
            var sOut = Keep(cpu.Allocate(heads * rowsA * dimA, zeroed: true));
            var sLse = Keep(cpu.Allocate(heads * rowsA, zeroed: true));
            cpu.AttentionSpans(On(cpu, q), On(cpu, keysA), On(cpu, valuesA), On(cpu, starts), On(cpu, ends), sOut, sLse, heads, kvHeads, heads / 2, rowsA, keyRows, dimA, 0.125f);
            Both("attention gradients over key ranges", [q, keysA, valuesA, starts, ends, Read(cpu, sOut, heads * rowsA * dimA), Read(cpu, sLse, heads * rowsA), Values(heads * rowsA * dimA)],
                [heads * rowsA * dimA, kvHeads * keyRows * dimA, kvHeads * keyRows * dimA],
                (k, s) => k.AttentionSpansBackward(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8], s[9], s[10], heads, kvHeads, heads / 2, rowsA, keyRows, dimA, 0.125f),
                1e-3f, wired: true);

            Check(!kernels || wiredFallbacks == 0,
                $"with kernels, no row, element-wise, product, attention, shape or training operation took the host fallback ({wiredFallbacks} calls did: {string.Join(", ", wiredNames)})");
            hip.Synchronize();
            if (!kernels)
            {
                // Unavailable kernels fail the test, unless the setting turned them off or no registered family handles the
                // device's target (its reason says "not registered"): then a skip, seen even with --failures-only, since a
                // run whose kernels never loaded is not a pass of the kernels.
                bool understood = HipBackend.KernelsSettingReason(Environment.GetEnvironmentVariable(HipBackend.KernelsVariable)) is not null
                                || HipBackend.TargetUnavailable(device.ToString(), hip.DeviceLimits.Architecture, out _, out _) is not null;
                Check(understood, $"{device}: kernels loaded, expected for its target {hip.DeviceLimits.Architecture}; unavailable: {hip.KernelsUnavailableReason}");
                Skip($"the kernels are unavailable ({hip.KernelsUnavailableReason}): memory, copies and the host fallback were checked against the CPU, the kernels were not");
            }
        }
        finally
        {
            foreach (var s in owned)
            {
                s.Release();
            }
        }
    }

    internal static void HipProfilerAndTelemetry(Device device)
    {
        if (FirstRun(nameof(HipProfilerAndTelemetry)))
        {
            Check(typeof(IKernelProfiler).IsAssignableFrom(typeof(HipBackend)), "HipBackend times its kernels (IKernelProfiler), so the profile's kernels probe and idrak profile take hip:N");
            Check(HipBackend.ProfileUnavailable("hip:0", "libamdhip64.so", []) is null, "a runtime with every event function: nothing to say");
            string? why = HipBackend.ProfileUnavailable("hip:0", "libamdhip64.so", ["hipEventCreate", "hipEventElapsedTime"]);
            Check(why is not null && why.Contains("hip:0 cannot time its kernels", StringComparison.Ordinal) && why.Contains("libamdhip64.so", StringComparison.Ordinal)
                  && why.Contains("hipEventCreate, hipEventElapsedTime", StringComparison.Ordinal), $"a runtime without events: the device, the runtime and what it lacks: {why}");
        }

        if (device.Type != DeviceType.Hip)
        {
            return;
        }

        var hip = (HipBackend)device.Backend;
        var failures = new DeviceFailures("hip");
        var fallbacks = new HostFallbacks("hip");
        using var listening = Telemetry.Subscribe(failures);
        using var listeningFallbacks = Telemetry.Subscribe(fallbacks);

        // A host fallback says why the kernels are unavailable (what --verbose prints), when it is the first of its operation.
        var a = hip.Allocate(64, zeroed: true);
        var b = hip.Allocate(64, zeroed: true);
        try
        {
            hip.Copy2D(a, 0, 8, b, 0, 8, 8, 8, accumulate: true);
            hip.Synchronize();
        }
        finally
        {
            a.Release();
            b.Release();
        }

        string? off = ((IKernelAvailability)hip).KernelsUnavailable();
        var first = fallbacks.Events.FirstOrDefault(e => e.Operation == "Copy2D");
        Check(first.Operation is null || first.KernelsUnavailable == off, $"the host fallback's reason: {first.KernelsUnavailable ?? "(reported earlier in the run)"} / {off ?? "kernels available"}");

        if (hip.ProfileUnavailableReason is { } cannot)
        {
            Idrak.Diagnostics.GpuProfiler.Start(device);
            var none = Idrak.Diagnostics.GpuProfiler.Stop(device);
            Check(none.Count == 0 && failures.Events.Any(e => e.Operation == "profile" && e.Message == cannot), $"no events: nothing timed, and the telemetry says why: {cannot}");
            Skip($"the HIP runtime cannot time kernels ({cannot})");
        }

        HipBackend.GcnFunction fill;
        try
        {
            fill = hip.LoadGcn(GcnKernels.Fill);
        }
        catch (HipException e)
        {
            Check(failures.Events.Any(f => f.Message == e.Message), $"a refused code object reaches the telemetry: {e.Message}");
            Skip($"the runtime refused Idrak's GCN code object, so no launch was timed: {e.Message}");
            return;
        }

        const int Count = 1 << 20;
        var storage = hip.Allocate(Count, zeroed: false);
        try
        {
            hip.FillGcn(fill, storage, Count, 1f);                              // a launch outside the profile: not timed
            Idrak.Diagnostics.GpuProfiler.Start(device);
            Check(hip.Profiling, "profiling while the profiler runs");
            for (int i = 0; i < 3; i++)
            {
                hip.FillGcn(fill, storage, Count, i);
            }

            HipBackend.LabelNextLaunch("fill_f32 labelled", 2.0 * Count);
            hip.FillGcn(fill, storage, Count, 4f);
            var entries = Idrak.Diagnostics.GpuProfiler.Stop(device);
            Check(!hip.Profiling, "not profiling after the profiler stopped");
            var named = entries.FirstOrDefault(e => e.Name == "fill_f32");
            var labelled = entries.FirstOrDefault(e => e.Name == "fill_f32 labelled");
            Check(entries.Count == 2 && named is { Calls: 3, Milliseconds: > 0, Tflops: null } && labelled is { Calls: 1, Milliseconds: > 0, Tflops: > 0 },
                $"fill_f32 timed three times under its name and once under its label with its operations: {string.Join("; ", entries)}");
            Console.WriteLine($"    {device}: fill_f32 over {Count:N0} floats, {named!.Milliseconds / 3 * 1000:F1} µs a launch on the device");
        }
        finally
        {
            storage.Release();
            fill.Dispose();
        }
    }
}
