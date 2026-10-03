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

        Options:
              --android  also the phone checks: the Vulkan driver file (VK_ICD_FILENAMES), the GPU device node
                         (/dev/kgsl-3d0) and the .NET heap limit (DOTNET_GCHeapHardLimit)
              --network  also the network: the proxy settings, whether the model hub, the repository API and the
                         package feed answer, and which tokens are set (never their values)
          -e, --explain  why each check matters
              --fix      the exact commands that fix what failed or warned; with --yes, the safe ones are done
                         (lines added to ~/.bashrc, or --rc FILE)
              --rc FILE  the shell start-up file --fix --yes adds lines to (default ~/.bashrc)
          -y, --yes      with --fix: do the safe fixes

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
                Status.Ok => (StatusWord(Status.Ok), Colour.Green),
                Status.Info => (StatusWord(Status.Info), Colour.Dim),
                Status.Warn => (StatusWord(Status.Warn), Colour.Yellow),
                _ => (StatusWord(Status.Fail), Colour.Red),
            };
            context.Write($"{Terminal.Paint(context, word, colour)}  {c.Name.PadRight(18)} {c.Detail}");
            string indent = new(' ', word.Length + 2);
            if (c.Fix is not null && c.Status is Status.Warn or Status.Fail)
            {
                context.Write($"{indent}{"".PadRight(18)} {Messages.T("fix: {0}", c.Fix)}");
            }

            if (explain)
            {
                context.Write($"{indent}{"".PadRight(18)} {Messages.T("why: {0}", c.Why)}");
            }
        }

        var repaired = new JsonArray();
        if (context.Flag("--fix"))
        {
            var fixable = checks.Where(c => c.Status is Status.Warn or Status.Fail && c.Command is not null).ToList();
            context.Write("");
            context.Write(fixable.Count == 0 ? Messages.T("No command fixes what is left; see the fix lines above.") : Messages.T("To fix:"));
            foreach (var c in fixable)
            {
                context.Write($"  {c.Command}    # {c.Name}{(c.Repair is not null ? context.Flag("--yes") ? "" : Messages.T(" (safe: --fix --yes does it)") : "")}");
            }

            if (context.Flag("--yes"))
            {
                foreach (var c in fixable.Where(c => c.Repair is not null))
                {
                    string done = c.Repair!();
                    context.Write(Messages.T("Done: {0}", done));
                    repaired.Add(done);
                }
            }
        }

        int failed = checks.Count(c => c.Status == Status.Fail), warned = checks.Count(c => c.Status == Status.Warn);
        context.Write("");
        context.Write(failed > 0 ? Messages.T("{0} check(s) failed, {1} warning(s).", failed, warned)
            : warned > 0 ? Messages.T("Everything needed works; {0} warning(s).", warned) : Messages.T("Everything works."));
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

    // The status word of a line, padded to the longest of the four in this language so the columns line up.
    private static string StatusWord(Status status)
    {
        string[] words = [Messages.T("ok"), Messages.T("info"), Messages.T("warn"), Messages.T("FAIL")];
        return words[(int)status].PadRight(words.Max(w => w.Length));
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
        checks.Add(new("runtime", Messages.T(".NET runtime"), version.Major >= 10 ? Status.Ok : Status.Fail,
            Messages.T("{0} ({1}) on {2}", RuntimeInformation.FrameworkDescription, RuntimeInformation.ProcessArchitecture, RuntimeInformation.OSDescription),
            Messages.T("install the .NET 10 runtime or SDK (dotnet --version prints 10.x)"),
            Messages.T("Idrak is built for .NET 10; an older runtime cannot load it.")));
        string simd = RuntimeInformation.ProcessArchitecture is Architecture.X64 or Architecture.X86
            ? string.Join(' ', new[] { ("AVX2", Avx2.IsSupported), ("FMA", Fma.IsSupported), ("AVX-512", Avx512F.IsSupported) }.Where(f => f.Item2).Select(f => f.Item1))
            : System.Runtime.Intrinsics.Arm.AdvSimd.IsSupported ? "AdvSIMD" : "";
        checks.Add(new("runtime", "CPU", Status.Info,
            Messages.T("{0} logical processors, {1} float lanes per vector{2}, {3} memory for the process", Environment.ProcessorCount,
                System.Numerics.Vector<float>.Count, simd.Length > 0 ? $" ({simd})" : "", Units.Bytes(Machine.Memory)),
            null, Messages.T("The CPU backend always works; its speed follows the vector width and the processor count.")));
    }

    private static void Backends(List<Check> checks)
    {
        foreach (var backend in DeviceListing.Backends)
        {
            if (backend.Count > 0)
            {
                checks.Add(new("backend", backend.Display, Status.Ok, Messages.T("{0} device(s) found", backend.Count), null, Why(backend.Kind)));
                continue;
            }

            string reason = backend.UnavailableReason is { Length: > 0 } r ? r : Messages.T("no devices found");
            bool disabled = Environment.GetEnvironmentVariable($"IDRAK_DISABLE_{backend.Kind.ToUpperInvariant()}") is "1" or "true";
            checks.Add(new("backend", backend.Display, disabled ? Status.Info : Status.Warn, Messages.T("none found: {0}", reason), disabled ? null : Fix(backend.Kind), Why(backend.Kind)));
        }

        if (OperatingSystem.IsLinux())
        {
            var icds = VulkanDriverFiles();
            checks.Add(icds.Count > 0
                ? new("backend", Messages.T("Vulkan drivers"), Status.Info, Messages.T("{0} driver file(s): {1}", icds.Count, string.Join(", ", icds.Select(Path.GetFileName))), null, VulkanDriversWhy)
                : new("backend", Messages.T("Vulkan drivers"), Status.Warn, Messages.T("no Vulkan driver (ICD) files found"),
                    Messages.T("install the GPU's Vulkan driver (or a software one for tests); VK_ICD_FILENAMES can point at a driver file"), VulkanDriversWhy));
        }
    }

    private static string VulkanDriversWhy => Messages.T("The Vulkan loader reaches GPUs only through the driver (ICD) files it finds; without one there is no Vulkan device.");

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
        "cuda" => Messages.T("install a GPU driver with CUDA support (only for GPUs that have it); the other backends work without it"),
        "vulkan" => Messages.T("install the Vulkan loader (libvulkan.so.1, vulkan-1.dll) and the GPU's Vulkan driver"),
        "hip" => Messages.T("install the HIP runtime and hipRTC (ROCm on Linux, the HIP SDK on Windows; HIP_PATH or ROCM_PATH finds them)"),
        _ => Messages.T("install what the {0} backend needs (see the plug-in that registers it)", kind),
    };

    private static string Why(string kind) => kind switch
    {
        "cuda" => Messages.T("CUDA drives GPUs through their CUDA driver; without it those GPUs can still run through Vulkan."),
        "vulkan" => Messages.T("Vulkan compute runs Idrak's generated kernels on most GPUs (and on phones); it needs the loader and a driver."),
        "hip" => Messages.T("HIP runs kernels compiled at run time by hipRTC; without the runtime those GPUs can still run through Vulkan."),
        _ => Messages.T("A registered backend adds devices of its kind."),
    };

    private static void Devices(CommandContext context, List<Check> checks)
    {
        foreach (var d in Machine.Devices(context))
        {
            if (d.Error is not null)
            {
                string disable = d.Kind is "cuda" or "vulkan" or "hip" ? Messages.T(", or leave the backend out with {0}", $"IDRAK_DISABLE_{d.Kind.ToUpperInvariant()}=1") : "";
                checks.Add(new("device", d.Device, d.Listed ? Status.Fail : Status.Warn, Messages.T("cannot start: {0}", d.Error),
                    Messages.T("update the device's driver{0}", disable), Messages.T("A device that is found but cannot start fails every model placed on it."))
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
                facts.Add(Messages.T("{0} compute units", units));
            }

            if (d.SubgroupSize is int lanes && d.Kind != "cpu")
            {
                facts.Add(Messages.T("subgroup {0}", lanes));
            }

            if (d.KernelWidth is int width)
            {
                facts.Add(Messages.T("kernels width {0}", width));
            }

            facts.Add(d.MatrixUnits ? Messages.T("matrix units") : Messages.T("no matrix units"));
            string use = d.IsDefault ? Messages.T("default device") : d.Listed ? Messages.T("listed") : d.Note is { Length: > 0 } n ? n : Messages.T("by name only");
            checks.Add(new("device", d.Device, Status.Ok, $"{d.Name}; {string.Join(", ", facts)}; {use}", null,
                Messages.T("Each device's limits (memory, subgroup size, matrix units) decide which kernels run and what models fit.")));
        }

        checks.Add(new("device", Messages.T("default device"), Status.Info, Device.Default.ToString(), null,
            Messages.T("Commands run on the default device unless --device (or the config's \"device\") names another.")));
    }

    private static void Storage(CommandContext context, List<Check> checks)
    {
        string cache = Path.GetFullPath(context.CacheFolder);
        string? writable = Writable(cache);
        checks.Add(writable is null
            ? new("storage", Messages.T("cache"), Status.Ok, $"{cache} ({(Directory.Exists(cache) ? Units.Bytes(CacheLayout.Size(cache)) : Messages.T("not created yet"))})", null, CacheWhy)
            : new("storage", Messages.T("cache"), Status.Fail, Messages.T("{0} is not writable: {1}", cache, writable), Messages.T("choose another folder with --cache DIR or IDRAK_CACHE"), CacheWhy));

        try
        {
            string? existing = cache;
            while (existing is not null && !Directory.Exists(existing))
            {
                existing = Path.GetDirectoryName(existing);
            }

            var drive = new DriveInfo(existing ?? cache);
            long free = drive.AvailableFreeSpace;
            checks.Add(new("storage", Messages.T("disk space"), free < 5L << 30 ? Status.Warn : Status.Ok, Messages.T("{0} free for the cache", Units.Bytes(free)),
                Messages.T("free some space, or move the cache to a larger disk with --cache DIR or IDRAK_CACHE"),
                Messages.T("Models take 0.5 to 20 GB each; a full disk stops downloads part way.")) { Command = "idrak cache clear models --dry-run" });
        }
        catch (Exception e) when (e is IOException or ArgumentException or UnauthorizedAccessException)
        {
            checks.Add(new("storage", Messages.T("disk space"), Status.Info, Messages.T("unknown ({0})", e.Message), null, Messages.T("Models take 0.5 to 20 GB each.")));
        }
    }

    private static string CacheWhy => Messages.T("Downloaded models, measured kernel choices and compiled kernels are kept in the cache folder.");

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
            return Messages.T("no existing parent folder");
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
            ? new("settings", Messages.T("config"), Status.Ok, $"{path}{(context.Config.Profile is { } p ? Messages.T(" (profile {0})", p) : "")}", null, ConfigWhy)
            : new("settings", Messages.T("config"), Status.Info, Messages.T("none at {0} (defaults apply; idrak init writes one)", path), null, ConfigWhy));

        var set = EnvironmentVariables.All.Where(v => !v.Secret && v.Name.StartsWith("IDRAK_", StringComparison.Ordinal) && Environment.GetEnvironmentVariable(v.Name) is { Length: > 0 }).Select(v => v.Name).ToList();
        checks.Add(new("settings", Messages.T("environment"), Status.Info, set.Count == 0 ? Messages.T("no Idrak variables set") : Messages.T("set: {0} (idrak env shows them)", string.Join(", ", set)), null,
            Messages.T("Environment variables change devices, tuning and memory for every Idrak program; a forgotten one explains surprises.")));
    }

    private static string ConfigWhy => Messages.T("The config file gives defaults (device, cache, plug-ins, model aliases) for every command.");

    /// <summary>The phone checks; fixes add lines to <paramref name="rc"/> (the shell start-up file).</summary>
    internal static void Android(List<Check> checks, string rc)
    {
        string? icd = Environment.GetEnvironmentVariable("VK_ICD_FILENAMES") ?? Environment.GetEnvironmentVariable("VK_DRIVER_FILES");
        var files = icd?.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries) ?? [];
        var missing = files.Where(f => !File.Exists(f)).ToList();
        string icdFix = Messages.T("build Turnip with -Dfreedreno-kmds=kgsl and export VK_ICD_FILENAMES=<its icd.d/freedreno_icd*.json> (installation/android-termux.md, step 7)");
        string icdWhy = Messages.T("On the phone the Vulkan loader finds the GPU only through the Turnip driver file that VK_ICD_FILENAMES names.");
        string? turnip = SetupAndroidCommand.FindTurnip();
        string? icdLine = turnip is null ? null : $"export VK_ICD_FILENAMES={turnip}";
        Check IcdCheck(Status status, string detail) => new("android", "VK_ICD_FILENAMES", status, detail, icdFix, icdWhy)
        {
            Command = icdLine is null ? null : $"echo '{icdLine}' >> {rc}",
            Repair = icdLine is null ? null : () => SetupAndroidCommand.AddLine(rc, icdLine),
        };
        checks.Add(icd is null ? IcdCheck(Status.Warn, Messages.T("not set"))
            : missing.Count > 0 ? IcdCheck(Status.Fail, Messages.T("names missing file(s): {0}", string.Join(", ", missing)))
            : new("android", "VK_ICD_FILENAMES", Status.Ok, icd, null, icdWhy));

        const string Node = "/dev/kgsl-3d0";
        string nodeWhy = Messages.T("The GPU driver reaches the phone's GPU through its kernel device node; without access there is no GPU.");
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

        checks.Add(!File.Exists(Node) ? new("android", Node, Status.Fail, Messages.T("not found"), Messages.T("run on the phone (Termux, with proot sharing /dev); ls -la /dev/kgsl-3d0 should list it"), nodeWhy)
            : access is not null ? new("android", Node, Status.Fail, Messages.T("not readable and writable: {0}", access), Messages.T("give the user access to the device node (installation/android-termux.md)"), nodeWhy)
            : new("android", Node, Status.Ok, Messages.T("readable and writable"), null, nodeWhy));

        string? heap = Environment.GetEnvironmentVariable("DOTNET_GCHeapHardLimit");
        checks.Add(heap is { Length: > 0 }
            ? new("android", "DOTNET_GCHeapHardLimit", Status.Ok, heap, null, HeapWhy)
            : new("android", "DOTNET_GCHeapHardLimit", Status.Warn, Messages.T("not set"), Messages.T("export DOTNET_GCHeapHardLimit=0x100000000 (4 GiB)"), HeapWhy)
            {
                Command = $"echo '{SetupAndroidCommand.HeapLine}' >> {rc}",
                Repair = () => SetupAndroidCommand.AddLine(rc, SetupAndroidCommand.HeapLine),
            });
    }

    /// <summary>The network checks: proxy settings, whether the hub, the repository API and the package feed answer, tokens set.</summary>
    internal static void Network(CommandContext context, List<Check> checks)
    {
        string proxyWhy = Messages.T("Downloads go through the proxy these variables name (read by .NET's HTTP client); a wrong one stops every download.");
        var proxies = new[] { "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY" }.Where(v => Environment.GetEnvironmentVariable(v) is { Length: > 0 }).ToList();
        checks.Add(new("network", Messages.T("proxy"), Status.Info, proxies.Count == 0 ? Messages.T("none set") : Messages.T("set: {0}", string.Join(", ", proxies)), null, proxyWhy));
        if (context.Offline)
        {
            checks.Add(new("network", Messages.T("reachability"), Status.Info, Messages.T("not checked (--offline)"), null, Messages.T("With --offline nothing is downloaded.")));
        }
        else
        {
            string hub = (Environment.GetEnvironmentVariable("HF_ENDPOINT") is { Length: > 0 } endpoint ? endpoint : "https://huggingface.co").TrimEnd('/');
            string api = (Environment.GetEnvironmentVariable("GITHUB_API_URL") is { Length: > 0 } github ? github : "https://api.github.com").TrimEnd('/');
            using var http = new HttpClient { Timeout = context.Timeout ?? TimeSpan.FromSeconds(10) };
            http.DefaultRequestHeaders.UserAgent.ParseAdd($"idrak/{Machine.ToolVersion}");
            foreach (var (name, url, why) in new[]
            {
                (Messages.T("model hub"), $"{hub}/api/models?limit=1", Messages.T("Models and datasets download from the hub (HF_ENDPOINT names a mirror).")),
                (Messages.T("repository API"), api, Messages.T("github: data sources read repositories through this API (GITHUB_API_URL).")),
                (Messages.T("package feed"), UpdateCommand.DefaultIndex, Messages.T("idrak update and new projects read the package feed.")),
            })
            {
                var watch = System.Diagnostics.Stopwatch.StartNew();
                try
                {
                    using var response = http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, context.TimeoutToken).GetAwaiter().GetResult();
                    checks.Add(new("network", name, Status.Ok, Messages.T("{0} answered {1} in {2} ms", url, (int)response.StatusCode, watch.ElapsedMilliseconds), null, why));
                }
                catch (Exception e) when (e is HttpRequestException or TaskCanceledException or InvalidOperationException)
                {
                    checks.Add(new("network", name, Status.Warn, Messages.T("{0} did not answer: {1}", url, e.Message),
                        Messages.T("check the connection and the proxy (HTTPS_PROXY); local models and --offline still work"), why));
                }
            }
        }

        bool hf = Environment.GetEnvironmentVariable("HF_TOKEN") is { Length: > 0 } || Environment.GetEnvironmentVariable("HUGGING_FACE_HUB_TOKEN") is { Length: > 0 }
                  || File.Exists(LoginCommand.HuggingFaceTokenFile());
        bool gh = Environment.GetEnvironmentVariable("GITHUB_TOKEN") is { Length: > 0 } || Environment.GetEnvironmentVariable("GH_TOKEN") is { Length: > 0 };
        bool kaggle = Environment.GetEnvironmentVariable("KAGGLE_KEY") is { Length: > 0 } || File.Exists(LoginCommand.KaggleFile());
        checks.Add(new("network", Messages.T("tokens"), Status.Info, Messages.T("Hugging Face: {0}, GitHub: {1}, Kaggle: {2} (idrak login stores one)", Set(hf), Set(gh), Set(kaggle)), null,
            Messages.T("Gated and private models, private repositories and Kaggle datasets need a token; public ones do not.")));
        static string Set(bool set) => set ? Messages.T("set") : Messages.T("not set");
    }

    private static string HeapWhy => Messages.T("Android's address space is too small for .NET's default heap reservation; without a limit dotnet fails at start.");
}
