// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.RegularExpressions;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
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
        ("hip kernel cache: the key changes with every input; a stored code object comes back; IDRAK_HIP_KERNEL_CACHE moves or disables it", HipKernelCacheKeys),
        ("hip: no vendor ids or card names in the code of src/Idrak.Gpu/Hip, and GPU targets (gfx names) only in its instruction-set tables, GcnTargets*.cs (comments may name them)", HipNoVendorNames),
        ("hip device: memory, uploads, downloads, copies and fills on the device, and every operation (products, rows, attention, gathers, permutes, through the host fallback until the GCN kernels land) match the CPU (on HIP devices)", HipDeviceMatchesCpu),
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

            // The kernels against the CPU (or the host fallback, when they are unavailable: same results either way).
            bool kernels = ((IKernelAvailability)hip).KernelsUnavailable() is null;
            Console.WriteLine($"    {device}: kernels {(kernels ? "loaded" : $"unavailable ({hip.KernelsUnavailableReason})")}");
            before = Idrak.Abstraction.Operations.Kernels.HostCalls(hip);
            var db = On(hip, b);
            var ca = On(cpu, a);
            var cb = On(cpu, b);
            void Compare(string what, Action<Backend, Storage, Storage, Storage> op, float tolerance = 1e-5f)
            {
                var hy = On(hip, Values(n));
                var cy = On(cpu, Read(hip, hy, n));
                op(hip, da, db, hy);
                op(cpu, ca, cb, cy);
                AssertClose(Read(cpu, cy, n), Read(hip, hy, n), tolerance, what);
            }

            foreach (var op in new[] { BinaryOp.Add, BinaryOp.Sub, BinaryOp.Mul })
            {
                Compare($"{op}", (k, x, y, z) => k.Binary(op, x, y, z, n));
            }

            foreach (var op in new[] { UnaryOp.Relu, UnaryOp.Sigmoid, UnaryOp.Tanh, UnaryOp.Square, UnaryOp.Abs, UnaryOp.Exp })
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
            hip.RmsNorm(hx, hy2, hinv, rows, cols, 1e-6f);
            cpu.RmsNorm(On(cpu, x2), cy2, cinv, rows, cols, 1e-6f);
            AssertClose(Read(cpu, cy2, rows * cols), Read(hip, hy2, rows * cols), 1e-4f, "RMS norm");
            AssertClose(Read(cpu, cinv, rows), Read(hip, hinv, rows), 1e-4f, "RMS norm scales");
            hip.RmsNormAffine(hx, hg, hy2, rows, cols, 1e-6f, 1f);
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
            hip.Int8MatMul(On(hip, x8), On(hip, packed), On(hip, scales), hy8, m, cols8, k8);
            cpu.Int8MatMul(On(cpu, x8), On(cpu, packed), On(cpu, scales), cy8, m, cols8, k8);
            AssertClose(Read(cpu, cy8, m * cols8), Read(hip, hy8, m * cols8), 1e-3f, "int8 product");

            // Every other kernel against the CPU: the same inputs on both, the outputs starting from the same values (for the
            // products' beta and the accumulating copies), every output compared.
            void Both(string what, float[][] inputs, int[] outputs, Action<Backend, Storage[]> op, float tolerance = 1e-4f)
            {
                var hs = inputs.Select(v => On(hip, v)).Concat(outputs.Select(count => On(hip, Values(count)))).ToArray();
                var cs = inputs.Select(v => On(cpu, v)).Concat(outputs.Select((count, i) => On(cpu, Read(hip, hs[inputs.Length + i], count)))).ToArray();
                op(hip, hs);
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
                            (k, s) => k.MatMul(s[0], s[1], s[2], pm, pn, pk, ta, tb, beta), 1e-3f);
                    }
                }
            }

            Both("batched product", [Values(3 * 17 * 5), Values(3 * 5 * 66)], [3 * 17 * 66], (k, s) => k.BatchedMatMul(s[0], s[1], s[2], 3, 17, 66, 5, false, true, 0f));
            Both("product with a bias", [Values(4 * 300), Values(300 * 260), Values(260)], [4 * 260], (k, s) =>
            {
                if (!k.MatMulBias(s[0], s[1], s[2], s[3], 4, 260, 300))
                {
                    k.MatMul(s[0], s[1], s[3], 4, 260, 300, false, false, 0f);
                    k.AddRowVector(s[3], s[2], s[3], 4, 260);
                }
            }, 1e-3f);

            // Packed products of few rows and the expansions (bfloat16 packed on the CPU; 4-bit words of random nibbles).
            int pr = 2, pcols = 300, pdepth = 70;
            var packedRows = new float[pdepth * pcols / 2];                          // an even width: rows of whole words
            var packedStorage = On(cpu, packedRows);
            cpu.PackBFloat16(On(cpu, Values(pdepth * pcols)), packedStorage, pdepth * pcols);
            cpu.Download(packedStorage, packedRows);
            Both("bfloat16 product", [Values(pr * pdepth), packedRows], [pr * pcols], (k, s) => k.BFloat16MatMul(s[0], s[1], s[2], pr, pcols, pdepth), 1e-3f);
            Both("bfloat16 expansion", [packedRows], [pdepth * pcols], (k, s) => k.BFloat16Dequantize(s[0], s[1], pdepth, pcols), 0f);
            Both("bfloat16 packing", [Values(301)], [151], (k, s) => k.PackBFloat16(s[0], s[1], 301), 0f);
            int int4Words = (pcols + 7) / 8;                                           // rows of ⌈n / 8⌉ words, scale rows of 8 · that
            var nibbles = new byte[pdepth * int4Words * 4];
            random.NextBytes(nibbles);
            var int4 = new float[pdepth * int4Words];
            Buffer.BlockCopy(nibbles, 0, int4, 0, nibbles.Length);
            float[] int4Scales = Values((pdepth + 31) / 32 * int4Words * 8, 0.001f, 0.01f);
            Both("4-bit product", [Values(pr * pdepth), int4, int4Scales], [pr * pcols], (k, s) => k.Int4MatMul(s[0], s[1], s[2], s[3], pr, pcols, pdepth), 1e-3f);
            Both("4-bit expansion", [int4, int4Scales], [pdepth * pcols], (k, s) => k.Int4Dequantize(s[0], s[1], s[2], pdepth, pcols), 0f);

            // Rows: softmax and log-softmax, the masked softmax, layer norm with its statistics, GELU after a bias.
            int sr = 5, sc = 513;
            float[] logits = Values(sr * sc, -6f, 6f), mask = [.. Enumerable.Range(0, 2 * sc).Select(i => i % 4 == 0 ? -1e9f : random.NextSingle())];
            Both("softmax", [logits], [sr * sc], (k, s) => k.Softmax(s[0], s[1], sr, sc, false));
            Both("log-softmax", [logits], [sr * sc], (k, s) => k.Softmax(s[0], s[1], sr, sc, true));
            Both("masked softmax", [logits, mask], [sr * sc], (k, s) => k.ScaleMaskSoftmax(s[0], s[1], s[2], sr, sc, 2, 0.5f));
            Both("layer norm", [logits, Values(sc), Values(sc)], [sr * sc, 2 * sr], (k, s) => k.LayerNormTrain(s[0], s[1], s[2], s[3], s[4], sr, sc, 1e-5f));
            Both("GELU after a bias", [logits, Values(sc)], [sr * sc], (k, s) => k.BiasGelu(s[0], s[1], s[2], sr * sc, sc));
            Both("best k log-probabilities", [logits], [sr * 8, sr * 8], (k, s) => k.LogSoftmaxTopK(s[0], s[1], s[2], sr, sc, 8));
            foreach (var op in new[] { UnaryOp.Gelu, UnaryOp.GeluErf, UnaryOp.Silu, UnaryOp.Sqrt, UnaryOp.Sin, UnaryOp.Cos, UnaryOp.Sign })
            {
                Both($"{op}", [Values(1000, 0.01f, 4f)], [1000], (k, s) => k.Unary(op, s[0], s[1], 1000));
            }

            // Attention over key ranges (grouped heads, two tables; with the log-sum-exp, then with an empty range, whose
            // log-sum-exp is -∞) and over a cache up to the device position.
            int heads = 4, kvHeads = 2, rowsA = 9, keyRows = 37, dimA = 64;
            float[] q = Values(heads * rowsA * dimA), keysA = Values(kvHeads * keyRows * dimA), valuesA = Values(kvHeads * keyRows * dimA);
            float[] starts = [.. Enumerable.Range(0, 2 * rowsA).Select(i => (float)(i % 3))], ends = [.. Enumerable.Range(0, 2 * rowsA).Select(i => (float)(keyRows - i))];
            Both("attention over key ranges", [q, keysA, valuesA, starts, ends], [heads * rowsA * dimA, heads * rowsA],
                (k, s) => k.AttentionSpans(s[0], s[1], s[2], s[3], s[4], s[5], s[6], heads, kvHeads, heads / 2, rowsA, keyRows, dimA, 0.125f));
            ends[4] = 1f;                                                                // row 4 of the first table: from key 1 to key 1
            Both("attention over key ranges, an empty one", [q, keysA, valuesA, starts, ends], [heads * rowsA * dimA],
                (k, s) => k.AttentionSpans(s[0], s[1], s[2], s[3], s[4], s[5], null, heads, kvHeads, heads / 2, rowsA, keyRows, dimA, 0.125f));
            foreach (int rowsPerHead in new[] { 1, 12 })
            {
                Both($"attention over a cache, {rowsPerHead} rows", [Values(heads * rowsPerHead * dimA), Values(heads * 50 * dimA), Values(heads * 50 * dimA), [6f]],
                    [heads * rowsPerHead * dimA], (k, s) => k.AttentionDecode(s[0], s[1], s[2], s[3], s[4], heads, rowsPerHead, Math.Min(rowsPerHead, 4), 50, dimA, 0.125f));
            }

            // Gathers, the cache write and mask, permutes and an accumulating strided copy.
            float[] ids = [0f, 5f, 36f, 2f];
            Both("gather", [Values(37 * 13), ids], [4 * 13], (k, s) => k.Gather(s[0], s[1], s[2], 4, 13, 37), 0f);
            Both("cache write", [Values(heads * 2 * dimA), [6f]], [heads * 50 * dimA], (k, s) => k.KeyValueWrite(s[0], s[2], s[1], heads, 2, 50, dimA), 0f);
            Both("decoder mask", [[6f]], [3 * 50], (k, s) => k.DecoderMask(s[0], s[1], 3, 50), 0f);
            Both("permute", [Values(2 * 3 * 4 * 5)], [120], (k, s) => k.Permute(s[0], s[1], [2, 4, 5, 3], [60, 5, 1, 20], accumulate: false), 0f);
            Both("permute, adding", [Values(2 * 3 * 4 * 5)], [120], (k, s) => k.Permute(s[0], s[1], [2, 4, 5, 3], [60, 5, 1, 20], accumulate: true));
            Both("strided copy, adding", [Values(200)], [200], (k, s) => k.Copy2D(s[0], 5, 13, s[1], 3, 20, 6, 9, accumulate: true));

            Check(!kernels || Idrak.Abstraction.Operations.Kernels.HostCalls(hip) == before, $"with kernels, none of these took the host fallback ({Idrak.Abstraction.Operations.Kernels.HostCalls(hip) - before} did)");
            hip.Synchronize();
            if (!kernels)
            {
                // Seen even with --failures-only: a run whose kernels never loaded is not a pass of the kernels.
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
}
