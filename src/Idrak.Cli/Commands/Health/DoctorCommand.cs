// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.Diagnostics;

namespace Idrak.Cli.Commands.Health;

/// <summary>
/// <c>idrak doctor</c> (<c>doc</c>): checks the runtime, every backend, each device, the caches, disk space, the config
/// and the environment, and says what is missing with how to fix it. Exits with 1 when a check fails (a missing GPU
/// backend is a warning: the CPU still works).
/// </summary>
internal sealed class DoctorCommand : Command
{
    public override string Name => "doctor";

    public override IReadOnlyCollection<string> Aliases => ["doc"];

    public override string Summary => "What works on this machine and what to fix: runtime, backends, devices, caches, disk, environment";

    public override string Usage => """
        [--android] [--network] [--explain] [--fix [--yes]]

          --android      also the phone checks: the Vulkan driver file (VK_ICD_FILENAMES), the GPU device node
                         (/dev/kgsl-3d0) and the .NET heap limit (DOTNET_GCHeapHardLimit)
          --network      also the network: the proxy settings, whether the model hub, the repository API and the
                         package feed answer, and which tokens are set (never their values)
          -e, --explain  why each check matters
          --fix          the exact commands that fix what failed or warned; with -y/--yes, the safe ones are done
                         (lines added to ~/.bashrc, or --rc FILE)
          --rc FILE      the shell start-up file --fix --yes adds lines to (default ~/.bashrc)

        Exits with 1 when a check fails; warnings (a backend not installed) leave it at 0.

        Examples:
          idrak doctor
          idrak doc --android
          idrak doctor -e -j
          idrak doctor --network
          idrak doc --android --fix -y
        """;

    public override IReadOnlyCollection<string> Flags => ["--android", "--explain", "--network", "--fix", "--yes"];

    public override IReadOnlyCollection<string> ValueOptions => ["--rc"];

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-e"] = "--explain", ["-y"] = "--yes" };

    /// <summary>A check's result.</summary>
    internal enum Status
    {
        Ok,
        Info,
        Warn,
        Fail,
    }

    /// <summary>One check: what was looked at, what was found, how to fix it and why it matters.</summary>
    internal sealed record Check(string Area, string Name, Status Status, string Detail, string? Fix, string Why)
    {
        /// <summary>The exact command that fixes it, when there is one.</summary>
        public string? Command { get; init; }

        /// <summary>A safe fix <c>--fix --yes</c> applies (returns what it did).</summary>
        public Func<string>? Repair { get; init; }
    }

    public override int Run(CommandContext context)
    {
        var checks = Checks(context, context.Flag("--android"));
        if (context.Flag("--network"))
        {
            Network(context, checks);
        }

        bool explain = context.Flag("--explain");
        if (context.Format is OutputFormat.Csv or OutputFormat.Markdown)
        {
            context.Table(explain ? ["Status", "Check", "Detail", "Fix", "Why"] : ["Status", "Check", "Detail", "Fix"], checks.Select(c => (IReadOnlyList<string>)
                [c.Status.ToString().ToLowerInvariant(), c.Name, c.Detail, c.Status is Status.Warn or Status.Fail ? c.Fix ?? "" : "", .. explain ? [c.Why ?? ""] : Array.Empty<string>()]));
        }

        foreach (var c in context.Format is OutputFormat.Csv or OutputFormat.Markdown ? [] : checks)
        {
            var (word, colour) = c.Status switch
            {
                Status.Ok => ("ok  ", Colour.Green),
                Status.Info => ("info", Colour.Dim),
                Status.Warn => ("warn", Colour.Yellow),
                _ => ("FAIL", Colour.Red),
            };
            context.Write($"{Terminal.Paint(context, word, colour)}  {c.Name.PadRight(18)} {c.Detail}");
            if (c.Fix is not null && c.Status is Status.Warn or Status.Fail)
            {
                context.Write($"      {"".PadRight(18)} fix: {c.Fix}");
            }

            if (explain)
            {
                context.Write($"      {"".PadRight(18)} why: {c.Why}");
            }
        }

        var repaired = new JsonArray();
        if (context.Flag("--fix"))
        {
            var fixable = checks.Where(c => c.Status is Status.Warn or Status.Fail && c.Command is not null).ToList();
            context.Write("");
            context.Write(fixable.Count == 0 ? "No command fixes what is left; see the fix lines above." : "To fix:");
            foreach (var c in fixable)
            {
                context.Write($"  {c.Command}    # {c.Name}{(c.Repair is not null ? context.Flag("--yes") ? "" : " (safe: --fix --yes does it)" : "")}");
            }

            if (context.Flag("--yes"))
            {
                foreach (var c in fixable.Where(c => c.Repair is not null))
                {
                    string done = c.Repair!();
                    context.Write($"Done: {done}");
                    repaired.Add(done);
                }
            }
        }

        int failed = checks.Count(c => c.Status == Status.Fail), warned = checks.Count(c => c.Status == Status.Warn);
        context.Write("");
        context.Write(failed > 0 ? $"{failed} check(s) failed, {warned} warning(s)." : warned > 0 ? $"Everything needed works; {warned} warning(s)." : "Everything works.");
        context.WriteJson(new JsonObject
        {
            ["failed"] = failed,
            ["warnings"] = warned,
            ["checks"] = new JsonArray([.. checks.Select(c => (JsonNode)new JsonObject
            {
                ["area"] = c.Area,
                ["name"] = c.Name,
                ["status"] = c.Status.ToString().ToLowerInvariant(),
                ["detail"] = c.Detail,
                ["fix"] = c.Fix,
                ["why"] = c.Why,
                ["command"] = c.Command,
            })]),
            ["repaired"] = repaired,
        });
        return failed > 0 ? ExitCodes.Failed : ExitCodes.Ok;
    }

    /// <summary>Every check, in the order printed.</summary>
    internal static List<Check> Checks(CommandContext context, bool android)
    {
        var checks = new List<Check>();
        Runtime(checks);
        Backends(checks);
        Devices(context, checks);
        Storage(context, checks);
        Settings(context, checks);
        if (android)
        {
            Android(checks, context.Option("--rc") ?? SetupAndroidCommand.DefaultRc);
        }

        return checks;
    }

    private static void Runtime(List<Check> checks)
    {
        var version = Environment.Version;
        checks.Add(new("runtime", ".NET runtime", version.Major >= 10 ? Status.Ok : Status.Fail,
            $"{RuntimeInformation.FrameworkDescription} ({RuntimeInformation.ProcessArchitecture}) on {RuntimeInformation.OSDescription}",
            "install the .NET 10 runtime or SDK (dotnet --version prints 10.x)",
            "Idrak is built for .NET 10; an older runtime cannot load it."));
        string simd = RuntimeInformation.ProcessArchitecture is Architecture.X64 or Architecture.X86
            ? string.Join(' ', new[] { ("AVX2", Avx2.IsSupported), ("FMA", Fma.IsSupported), ("AVX-512", Avx512F.IsSupported) }.Where(f => f.Item2).Select(f => f.Item1))
            : System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported ? "AdvSIMD" : "";
        checks.Add(new("runtime", "CPU", Status.Info,
            $"{Environment.ProcessorCount} logical processors, {System.Numerics.Vector<float>.Count} float lanes per vector{(simd.Length > 0 ? $" ({simd})" : "")}, " +
            $"{Units.Bytes(Machine.Memory)} memory for the process",
            null, "The CPU backend always works; its speed follows the vector width and the processor count."));
    }

    private static void Backends(List<Check> checks)
    {
        foreach (var backend in DeviceListing.Backends)
        {
            if (backend.Count > 0)
            {
                checks.Add(new("backend", backend.Display, Status.Ok, $"{backend.Count} device(s) found", null, Why(backend.Kind)));
                continue;
            }

            string reason = backend.UnavailableReason is { Length: > 0 } r ? r : "no devices found";
            bool disabled = Environment.GetEnvironmentVariable($"IDRAK_DISABLE_{backend.Kind.ToUpperInvariant()}") is "1" or "true";
            checks.Add(new("backend", backend.Display, disabled ? Status.Info : Status.Warn, $"none found: {reason}", disabled ? null : Fix(backend.Kind), Why(backend.Kind)));
        }

        if (OperatingSystem.IsLinux())
        {
            var icds = VulkanDriverFiles();
            checks.Add(icds.Count > 0
                ? new("backend", "Vulkan drivers", Status.Info, $"{icds.Count} driver file(s): {string.Join(", ", icds.Select(Path.GetFileName))}", null, VulkanDriversWhy)
                : new("backend", "Vulkan drivers", Status.Warn, "no Vulkan driver (ICD) files found",
                    "install the GPU's Vulkan driver (or a software one for tests); VK_ICD_FILENAMES can point at a driver file", VulkanDriversWhy));
        }
    }

    private const string VulkanDriversWhy = "The Vulkan loader reaches GPUs only through the driver (ICD) files it finds; without one there is no Vulkan device.";

    // The Vulkan driver files the loader would read on Linux: VK_DRIVER_FILES / VK_ICD_FILENAMES, else the standard folders.
    internal static List<string> VulkanDriverFiles()
    {
        if ((Environment.GetEnvironmentVariable("VK_DRIVER_FILES") ?? Environment.GetEnvironmentVariable("VK_ICD_FILENAMES")) is { Length: > 0 } listed)
        {
            return [.. listed.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries).Where(File.Exists)];
        }

        string home = Environment.GetFolderPath(Environment.SpecialFolder.UserProfile);
        string[] folders = ["/etc/vulkan/icd.d", "/usr/share/vulkan/icd.d", "/usr/local/share/vulkan/icd.d", Path.Combine(home, ".local", "share", "vulkan", "icd.d")];
        return [.. folders.Where(Directory.Exists).SelectMany(f => Directory.EnumerateFiles(f, "*.json")).Order(StringComparer.Ordinal)];
    }

    private static string Fix(string kind) => kind switch
    {
        "cuda" => "install a GPU driver with CUDA support (only for GPUs that have it); the other backends work without it",
        "vulkan" => "install the Vulkan loader (libvulkan.so.1, vulkan-1.dll) and the GPU's Vulkan driver",
        "hip" => "install the HIP runtime and hipRTC (ROCm on Linux, the HIP SDK on Windows; HIP_PATH or ROCM_PATH finds them)",
        _ => $"install what the {kind} backend needs (see the plug-in that registers it)",
    };

    private static string Why(string kind) => kind switch
    {
        "cuda" => "CUDA drives GPUs through their CUDA driver; without it those GPUs can still run through Vulkan.",
        "vulkan" => "Vulkan compute runs Idrak's generated kernels on most GPUs (and on phones); it needs the loader and a driver.",
        "hip" => "HIP runs kernels compiled at run time by hipRTC; without the runtime those GPUs can still run through Vulkan.",
        _ => "A registered backend adds devices of its kind.",
    };

    private static void Devices(CommandContext context, List<Check> checks)
    {
        foreach (var d in Machine.Devices(context))
        {
            if (d.Error is not null)
            {
                string disable = d.Kind is "cuda" or "vulkan" or "hip" ? $", or leave the backend out with IDRAK_DISABLE_{d.Kind.ToUpperInvariant()}=1" : "";
                checks.Add(new("device", d.Device, d.Listed ? Status.Fail : Status.Warn, $"cannot start: {d.Error}",
                    $"update the device's driver{disable}", "A device that is found but cannot start fails every model placed on it.")
                {
                    Command = disable.Length > 0 ? $"export IDRAK_DISABLE_{d.Kind.ToUpperInvariant()}=1" : null,
                });
                continue;
            }

            var facts = new List<string>();
            if (d.MemoryBytes is long memory)
            {
                facts.Add($"{Units.Bytes(memory)}");
            }

            if (d.ComputeUnits is int units && d.Kind != "cpu")
            {
                facts.Add($"{units} compute units");
            }

            if (d.SubgroupSize is int lanes && d.Kind != "cpu")
            {
                facts.Add($"subgroup {lanes}");
            }

            if (d.KernelWidth is int width)
            {
                facts.Add($"kernels width {width}");
            }

            facts.Add(d.MatrixUnits ? "matrix units" : "no matrix units");
            string use = d.IsDefault ? "default device" : d.Listed ? "listed" : d.Note is { Length: > 0 } n ? n : "by name only";
            checks.Add(new("device", d.Device, Status.Ok, $"{d.Name}; {string.Join(", ", facts)}; {use}", null,
                "Each device's limits (memory, subgroup size, matrix units) decide which kernels run and what models fit."));
        }

        checks.Add(new("device", "default device", Status.Info, Device.Default.ToString(), null,
            "Commands run on the default device unless --device (or the config's \"device\") names another."));
    }

    private static void Storage(CommandContext context, List<Check> checks)
    {
        string cache = Path.GetFullPath(context.CacheFolder);
        string? writable = Writable(cache);
        checks.Add(writable is null
            ? new("storage", "cache", Status.Ok, $"{cache} ({(Directory.Exists(cache) ? Units.Bytes(CacheLayout.Size(cache)) : "not created yet")})", null, CacheWhy)
            : new("storage", "cache", Status.Fail, $"{cache} is not writable: {writable}", "choose another folder with --cache DIR or IDRAK_CACHE", CacheWhy));

        try
        {
            string? existing = cache;
            while (existing is not null && !Directory.Exists(existing))
            {
                existing = Path.GetDirectoryName(existing);
            }

            var drive = new DriveInfo(existing ?? cache);
            long free = drive.AvailableFreeSpace;
            checks.Add(new("storage", "disk space", free < 5L << 30 ? Status.Warn : Status.Ok, $"{Units.Bytes(free)} free for the cache",
                "free some space, or move the cache to a larger disk with --cache DIR or IDRAK_CACHE",
                "Models take 0.5 to 20 GB each; a full disk stops downloads part way.") { Command = "idrak cache clear models --dry-run" });
        }
        catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException)
        {
            checks.Add(new("storage", "disk space", Status.Info, $"unknown ({e.Message})", null, "Models take 0.5 to 20 GB each."));
        }
    }

    private const string CacheWhy = "Downloaded models, measured kernel choices and compiled kernels are kept in the cache folder.";

    // Null when files can be created in the folder (or in the nearest existing parent, where it would be created).
    private static string? Writable(string folder)
    {
        string? existing = folder;
        while (existing is not null && !Directory.Exists(existing))
        {
            existing = Path.GetDirectoryName(existing);
        }

        if (existing is null)
        {
            return "no existing parent folder";
        }

        try
        {
            string probe = Path.Combine(existing, $".idrak-doctor-{Environment.ProcessId}");
            File.WriteAllText(probe, "");
            File.Delete(probe);
            return null;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            return e.Message;
        }
    }

    private static void Settings(CommandContext context, List<Check> checks)
    {
        string path = Path.GetFullPath(context.Config.Path);
        checks.Add(File.Exists(path)
            ? new("settings", "config", Status.Ok, $"{path}{(context.Config.Profile is { } p ? $" (profile {p})" : "")}", null, ConfigWhy)
            : new("settings", "config", Status.Info, $"none at {path} (defaults apply; idrak init writes one)", null, ConfigWhy));

        var set = EnvironmentVariables.All.Where(v => !v.Secret && v.Name.StartsWith("IDRAK_", StringComparison.Ordinal) && Environment.GetEnvironmentVariable(v.Name) is { Length: > 0 }).Select(v => v.Name).ToList();
        checks.Add(new("settings", "environment", Status.Info, set.Count == 0 ? "no Idrak variables set" : $"set: {string.Join(", ", set)} (idrak env shows them)", null,
            "Environment variables change devices, tuning and memory for every Idrak program; a forgotten one explains surprises."));
    }

    private const string ConfigWhy = "The config file gives defaults (device, cache, plug-ins, model aliases) for every command.";

    /// <summary>The phone checks; fixes add lines to <paramref name="rc"/> (the shell start-up file).</summary>
    internal static void Android(List<Check> checks, string rc)
    {
        string? icd = Environment.GetEnvironmentVariable("VK_ICD_FILENAMES") ?? Environment.GetEnvironmentVariable("VK_DRIVER_FILES");
        var files = icd?.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries) ?? [];
        var missing = files.Where(f => !File.Exists(f)).ToList();
        const string IcdFix = "build Turnip with -Dfreedreno-kmds=kgsl and export VK_ICD_FILENAMES=<its icd.d/freedreno_icd*.json> (installation/android-termux.md, step 7)";
        const string IcdWhy = "On the phone the Vulkan loader finds the GPU only through the Turnip driver file that VK_ICD_FILENAMES names.";
        string? turnip = SetupAndroidCommand.FindTurnip();
        string? icdLine = turnip is null ? null : $"export VK_ICD_FILENAMES={turnip}";
        Check IcdCheck(Status status, string detail) => new("android", "VK_ICD_FILENAMES", status, detail, IcdFix, IcdWhy)
        {
            Command = icdLine is null ? null : $"echo '{icdLine}' >> {rc}",
            Repair = icdLine is null ? null : () => SetupAndroidCommand.AddLine(rc, icdLine),
        };
        checks.Add(icd is null ? IcdCheck(Status.Warn, "not set")
            : missing.Count > 0 ? IcdCheck(Status.Fail, $"names missing file(s): {string.Join(", ", missing)}")
            : new("android", "VK_ICD_FILENAMES", Status.Ok, icd, null, IcdWhy));

        const string Node = "/dev/kgsl-3d0";
        const string NodeWhy = "The GPU driver reaches the phone's GPU through its kernel device node; without access there is no GPU.";
        string? access = null;
        if (File.Exists(Node))
        {
            try
            {
                using var stream = new FileStream(Node, FileMode.Open, FileAccess.ReadWrite);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                access = e.Message;
            }
        }

        checks.Add(!File.Exists(Node) ? new("android", Node, Status.Fail, "not found", "run on the phone (Termux, with proot sharing /dev); ls -la /dev/kgsl-3d0 should list it", NodeWhy)
            : access is not null ? new("android", Node, Status.Fail, $"not readable and writable: {access}", "give the user access to the device node (installation/android-termux.md)", NodeWhy)
            : new("android", Node, Status.Ok, "readable and writable", null, NodeWhy));

        string? heap = Environment.GetEnvironmentVariable("DOTNET_GCHeapHardLimit");
        checks.Add(heap is { Length: > 0 }
            ? new("android", "DOTNET_GCHeapHardLimit", Status.Ok, heap, null, HeapWhy)
            : new("android", "DOTNET_GCHeapHardLimit", Status.Warn, "not set", "export DOTNET_GCHeapHardLimit=0x100000000 (4 GiB)", HeapWhy)
            {
                Command = $"echo '{SetupAndroidCommand.HeapLine}' >> {rc}",
                Repair = () => SetupAndroidCommand.AddLine(rc, SetupAndroidCommand.HeapLine),
            });
    }

    /// <summary>The network checks: proxy settings, whether the hub, the repository API and the package feed answer, tokens set.</summary>
    internal static void Network(CommandContext context, List<Check> checks)
    {
        const string ProxyWhy = "Downloads go through the proxy these variables name (read by .NET's HTTP client); a wrong one stops every download.";
        var proxies = new[] { "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY" }.Where(v => Environment.GetEnvironmentVariable(v) is { Length: > 0 }).ToList();
        checks.Add(new("network", "proxy", Status.Info, proxies.Count == 0 ? "none set" : $"set: {string.Join(", ", proxies)}", null, ProxyWhy));
        if (context.Offline)
        {
            checks.Add(new("network", "reachability", Status.Info, "not checked (--offline)", null, "With --offline nothing is downloaded."));
        }
        else
        {
            string hub = (Environment.GetEnvironmentVariable("HF_ENDPOINT") is { Length: > 0 } endpoint ? endpoint : "https://huggingface.co").TrimEnd('/');
            string api = (Environment.GetEnvironmentVariable("GITHUB_API_URL") is { Length: > 0 } github ? github : "https://api.github.com").TrimEnd('/');
            using var http = new HttpClient { Timeout = context.Timeout ?? TimeSpan.FromSeconds(10) };
            http.DefaultRequestHeaders.UserAgent.ParseAdd($"idrak/{Machine.ToolVersion}");
            foreach (var (name, url, why) in new[]
            {
                ("model hub", $"{hub}/api/models?limit=1", "Models and datasets download from the hub (HF_ENDPOINT names a mirror)."),
                ("repository API", api, "github: data sources read repositories through this API (GITHUB_API_URL)."),
                ("package feed", UpdateCommand.DefaultIndex, "idrak update and new projects read the package feed."),
            })
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                try
                {
                    using var response = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, context.TimeoutToken).GetAwaiter().GetResult();
                    checks.Add(new("network", name, Status.Ok, $"{url} answered {(int)response.StatusCode} in {watch.ElapsedMilliseconds} ms", null, why));
                }
                catch (Exception e) when (e is HttpRequestException or TaskCanceledException or InvalidOperationException)
                {
                    checks.Add(new("network", name, Status.Warn, $"{url} did not answer: {e.Message}",
                        "check the connection and the proxy (HTTPS_PROXY); local models and --offline still work", why));
                }
            }
        }

        bool hf = Environment.GetEnvironmentVariable("HF_TOKEN") is { Length: > 0 } || Environment.GetEnvironmentVariable("HUGGING_FACE_HUB_TOKEN") is { Length: > 0 }
                  || File.Exists(LoginCommand.HuggingFaceTokenFile());
        bool gh = Environment.GetEnvironmentVariable("GITHUB_TOKEN") is { Length: > 0 } || Environment.GetEnvironmentVariable("GH_TOKEN") is { Length: > 0 };
        bool kaggle = Environment.GetEnvironmentVariable("KAGGLE_KEY") is { Length: > 0 } || File.Exists(LoginCommand.KaggleFile());
        checks.Add(new("network", "tokens", Status.Info, $"Hugging Face: {Set(hf)}, GitHub: {Set(gh)}, Kaggle: {Set(kaggle)} (idrak login stores one)", null,
            "Gated and private models, private repositories and Kaggle datasets need a token; public ones do not."));
        static string Set(bool set) => set ? "set" : "not set";
    }

    private const string HeapWhy = "Android's address space is too small for .NET's default heap reservation; without a limit dotnet fails at start.";
}
