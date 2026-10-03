// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Cli.Shared;

/// <summary>
/// Every environment variable the Idrak libraries and the tool read (plans/idrak-cli.md, "Environment variables"): the
/// one table <c>idrak env</c>, <c>idrak help env</c> and the end of each command's help are made from. A test scans the
/// sources for variable names read and fails when one is missing here. Each meaning is taken from the code that reads
/// the variable.
/// </summary>
internal static class EnvironmentVariables
{
    /// <summary>One variable.</summary>
    /// <param name="Name">The variable's name.</param>
    /// <param name="Group">Its group in listings ("Devices and backends", ...).</param>
    /// <param name="Default">What applies when it is not set.</param>
    /// <param name="Meaning">What it does.</param>
    /// <param name="Commands">The commands it affects (names as in <c>idrak help</c>); <see cref="Every"/>: every command.</param>
    /// <param name="Secret">A token or key: listings say only whether it is set.</param>
    public sealed record Variable(string Name, string Group, string Default, string Meaning, string[] Commands, bool Secret = false);

    /// <summary>Marks a variable that affects every command.</summary>
    public const string Every = "*";

    // Commands that run models or kernels on a device (the device and backend variables affect these).
    private static readonly string[] OnDevice =
    [
        "doctor", "devices", "version", "report", "init", "chat", "run", "batch", "compare", "complete", "embed", "agent", "serve", "ui",
        "memory", "quantize", "merge", "convert", "bench", "eval", "perplexity", "profile", "check", "tuning show", "tune", "train", "resume", "predict",
        "distill", "suggest", "explain", "rag index", "rag ask", "rag search", "rag eval", "test", "trace", "demo", "onnx import", "onnx export",
        "onnx check",
    ];

    // Commands that read models from the hub or the local stores (a model argument, or an index's embedding model).
    private static readonly string[] Models =
    [
        "pull", "list", "rm", "show", "search", "verify", "convert", "diff", "inspect", "chat", "run", "batch", "compare", "complete", "embed",
        "tokenize", "template", "agent", "serve", "ui", "memory", "quantize", "merge", "bench", "eval", "perplexity", "profile", "check", "tune",
        "distill", "data stats", "rag index", "rag ask", "rag search", "rag eval", "suggest",
    ];

    // Commands that read datasets from remote sources (the others read local files only).
    private static readonly string[] Data = ["data", "data mix", "tune"];

    // Commands that read or clear the tuning caches and compiled kernels besides those running on a device.
    private static readonly string[] Caches = [.. OnDevice, "cache info", "cache clear"];

    // Commands that read or store the hub and repository tokens.
    private static readonly string[] Logins = ["login", "logout", "doctor"];

    // Commands that call a running server (they go through the proxy settings unless NO_PROXY spares the host).
    private static readonly string[] Clients = ["ping", "api", "server ps", "server stop", "server load", "server unload"];

    // Commands that reach the network: model and data downloads, the package feed, the network checks, server calls.
    private static readonly string[] Network = [.. Models, .. Data, .. Clients, "update", "doctor"];

    // Commands whose child processes get the variables the coding tools set.
    private static readonly string[] ChildCommands = ["agent", "tools test"];

    private const string Child = "Set for the coding tools' commands";

    private const string Devices = "Devices and backends", Tuning = "Tuning and caches", Precision = "Precision and memory", Cpu = "CPU",
        Vulkan = "Vulkan", Hip = "HIP", Sources = "Model and data sources", Tool = "The tool", Tests = "Tests and diagnostics", DotNet = ".NET";

    /// <summary>The table, in listing order.</summary>
    public static IReadOnlyList<Variable> All { get; } =
    [
        new("IDRAK_DISABLE_CUDA", Devices, "not set", "1 or true: CUDA devices are not looked for (the backend reports itself disabled)", OnDevice),
        new("IDRAK_DISABLE_VULKAN", Devices, "not set", "1 or true: Vulkan devices are not looked for", OnDevice),
        new("IDRAK_DISABLE_HIP", Devices, "not set", "1 or true: HIP devices are not looked for", OnDevice),
        new("IDRAK_VULKAN_DEFAULT", Devices, "not set", "1: a Vulkan GPU may be the default device (otherwise only by name until its kernels are tuned)", OnDevice),
        new("IDRAK_HIP_DEFAULT", Devices, "not set", "1: a HIP GPU may be the default device (after CUDA, before Vulkan)", OnDevice),
        new("IDRAK_CUDA_DEBUG", Devices, "not set", "1: synchronize after every CUDA kernel and name the one that failed (slow)", OnDevice),
        new("IDRAK_WINDOW_KERNELS", Devices, "on", "0 or false: sliding-window and soft-capped attention through basic operations over the whole cache instead of the attention kernels (to compare or isolate them)", OnDevice),
        new("IDRAK_POWER_SOURCE", Devices, "as the system reports", "ac or battery: the power source tuning choices are measured and kept under", OnDevice),

        new("IDRAK_CACHE", Tuning, "~/.cache/idrak", "The cache folder: downloaded models, tuning choices, compiled kernels (--cache wins for the tool)", [Every]),
        new("IDRAK_AUTOTUNE", Tuning, "1 (measure)", "0 or false: the formulas only, no measuring of kernel choices on the device", OnDevice),
        new("IDRAK_TUNING_CACHE", Tuning, "on, under the cache folder", "0, false or off: measured choices are not kept; a path: the CUDA tuning folder", Caches),
        new("IDRAK_TUNE_LOG", Tuning, "not set", "1: print every tuning measurement (candidates, time ratios, the choice) to the error output", OnDevice),
        new("IDRAK_CPU_TUNING_FILE", Tuning, "<cache>/cpu/tuning.tsv", "The CPU tuning file (empty: none)", Caches),
        new("IDRAK_VULKAN_TUNING_CACHE", Tuning, "<cache>/vulkan/tuning.tsv", "0 or false: no Vulkan tuning file; a path: the file", Caches),
        new("IDRAK_HIP_KERNEL_CACHE", Tuning, "<cache>/hip/kernels", "0 or false: compiled HIP kernels are not kept; a path: the folder", Caches),

        new("IDRAK_MATMUL", Precision, "float32", "bf16 (bfloat16) or fp8 (float8): the precision of matrix products outside any precision scope", OnDevice),
        new("IDRAK_FP8_DELAYED", Precision, "not set", "1: delayed column scaling for float8 products on CUDA", OnDevice),
        new("IDRAK_OFFLOAD", Precision, "not set", "1 or true: move tensors to system memory when device memory runs out", OnDevice),

        new("IDRAK_CPU_KCHUNK", Cpu, "from the cache sizes", "Forces the k chunk of the CPU's few-row kernels (tests; every chunk gives the same results)", OnDevice),
        new("IDRAK_CPU_PARALLEL_ELEMENTS", Cpu, "measured", "Overrides the element-wise parallel cut-over (tests and benchmarks)", OnDevice),
        new("IDRAK_CPU_PARALLEL_FLOPS", Cpu, "measured", "Overrides the product parallel cut-over in m*n*k (tests and benchmarks)", OnDevice),

        new("IDRAK_VULKAN_KERNELS", Vulkan, "on", "0 or false: every operation through the host fallback (to compare or isolate a kernel)", OnDevice),
        new("IDRAK_VULKAN_MATRIX", Vulkan, "on where reported", "0 or false: cooperative matrices (matrix units) stay off", OnDevice),
        new("IDRAK_VULKAN_WIDTH", Vulkan, "measured", "Forces the kernels' workgroup width (a power of two within the device's limits; diagnostics only)", OnDevice),
        new("IDRAK_VULKAN_WIDTH_PROBE", Vulkan, "on", "0: keep the formula's width without measuring", OnDevice),
        new("IDRAK_VULKAN_SUBGROUPS", Vulkan, "on where reported", "0 or false: reductions through workgroup memory alone, as on devices without subgroup arithmetic", OnDevice),
        new("IDRAK_VULKAN_SUBGROUP_SIZE", Vulkan, "on where reported", "0 or false: no per-kernel subgroup size (every pipeline at the device's default)", OnDevice),
        new("IDRAK_VULKAN_STORAGE", Vulkan, "measured", "staging, mapped-device, mapped-cached or mapped-uncached: the memory tensors are kept in", OnDevice),
        new("IDRAK_VULKAN_STAGING", Vulkan, "not set", "1 or true: copies go through staging buffers everywhere", OnDevice),
        new("IDRAK_VULKAN_STAGING_BYTES", Vulkan, "from the heap", "The staging buffer's size in bytes (tests)", OnDevice),
        new("IDRAK_VULKAN_PAGE_BYTES", Vulkan, "from the heap", "The size of the memory pages storages are carved from, in bytes (tests)", OnDevice),
        new("IDRAK_VULKAN_MAX_STORAGE_BYTES", Vulkan, "maxStorageBufferRange", "Lowers the largest storage binding, to reach the windowed paths on any device (tests)", OnDevice),
        new("IDRAK_VULKAN_MAX_ALLOCATIONS", Vulkan, "the driver's cap", "Lowers the memory allocation count cap (to reproduce a low cap on any device)", OnDevice),
        new("IDRAK_VULKAN_PUSH_DESCRIPTORS", Vulkan, "measured", "1 or 0: pushed descriptors or descriptor sets", OnDevice),
        new("IDRAK_VULKAN_BATCH_COMMANDS", Vulkan, "measured", "Commands recorded per submitted batch", OnDevice),
        new("IDRAK_VULKAN_IN_FLIGHT", Vulkan, "measured", "Batches in flight at once", OnDevice),
        new("VK_ICD_FILENAMES", Vulkan, "the loader's search", "The Vulkan loader: which driver (ICD) files to use (needed for Turnip on a phone)", [.. OnDevice, "setup android"]),
        new("VK_DRIVER_FILES", Vulkan, "the loader's search", "The Vulkan loader: the newer name of VK_ICD_FILENAMES", [.. OnDevice, "setup android"]),
        new("VK_INSTANCE_LAYERS", Vulkan, "none", "The Vulkan loader: layers to enable (e.g. validation), read by the loader itself", OnDevice),

        new("IDRAK_HIP_KERNELS", Hip, "on", "0 or false: every HIP operation through the host fallback", OnDevice),
        new("HIP_PATH", Hip, "not set", "Where the HIP SDK is (Windows)", OnDevice),
        new("ROCM_PATH", Hip, "/opt/rocm", "Where ROCm is (Linux)", OnDevice),

        new("HF_TOKEN", Sources, "the saved login", "The Hugging Face access token for gated and private models and datasets", [.. Models, .. Data, .. Logins], Secret: true),
        new("HUGGING_FACE_HUB_TOKEN", Sources, "not set", "The older name of HF_TOKEN", [.. Models, .. Data, "doctor"], Secret: true),
        new("HF_TOKEN_PATH", Sources, "<HF_HOME>/token", "The file holding the saved Hugging Face token", [.. Models, .. Data, .. Logins]),
        new("HF_HOME", Sources, "~/.cache/huggingface", "The Hugging Face folder (its hub cache is searched for models; its token file is read)", [.. Models, .. Data, .. Logins]),
        new("HF_HUB_CACHE", Sources, "<HF_HOME>/hub", "The Hugging Face hub cache searched for already downloaded models", Models),
        new("HF_ENDPOINT", Sources, "https://huggingface.co", "The Hugging Face hub address (a mirror)", [.. Models, .. Data, "doctor"]),
        new("OLLAMA_MODELS", Sources, "the store's own folder in the home folder", "The local model store whose GGUF models are read by name", Models),
        new("GITHUB_TOKEN", Sources, "not set", "A token for private repositories and higher rate limits (github: data sources)", [.. Data, .. Logins], Secret: true),
        new("GH_TOKEN", Sources, "not set", "Read when GITHUB_TOKEN is not set", Data, Secret: true),
        new("GITHUB_API_URL", Sources, "https://api.github.com", "The repository API root (an enterprise server)", [.. Data, "doctor"]),
        new("KAGGLE_USERNAME", Sources, "from kaggle.json", "The Kaggle account name (kaggle: data sources)", [.. Data, .. Logins]),
        new("KAGGLE_KEY", Sources, "from kaggle.json", "The Kaggle API key", [.. Data, .. Logins], Secret: true),
        new("KAGGLE_CONFIG_DIR", Sources, "~/.kaggle", "The folder holding kaggle.json", [.. Data, .. Logins]),
        new("ZENODO_TOKEN", Sources, "not set", "A token for restricted Zenodo records (zenodo: data sources)", Data, Secret: true),

        new("IDRAK_CONFIG", Tool, "~/.idrak/config.json", "The config file (--config wins)", [Every]),
        new("IDRAK_PROFILE", Tool, "the config's \"profile\"", "The config profile to use (a name under \"profiles\" in the config)", [Every]),
        new("IDRAK_API_KEY", Tool, "not set", "The key serve requires and the client commands (api, ping, server ...) send when --api-key is not given", ["serve", "ui", "api", "ping", "server ps", "server stop", "server load", "server unload"], Secret: true),
        new("IDRAK_TRACE", Tool, "not set", "1 or true: errors print the stack as well", [Every]),
        new("NO_COLOR", Tool, "not set", "Set and not empty: no colour in the output", [Every]),
        new("TERM", Tool, "as the terminal sets it", "dumb: no colour in the output", [Every]),
        new("IDRAK_UPDATE_INDEX", Tool, "the package feed", "Where idrak update reads the published versions (a URL or a local file; mirrors and tests)", ["update"]),

        new("IDRAK_DEVICES", Tests, "every listed device", "Devices the test runner uses, comma-separated (cpu, cuda:0, vulkan:0, ...)", ["test", "report"]),
        new("IDRAK_FILTER", Tests, "every test", "Run only tests whose names contain this text", ["test", "report"]),
        new("IDRAK_TIMEOUT", Tests, "300", "Seconds before a running test is reported as HANG and the run stops", ["test", "report"]),
        new("IDRAK_SPIRV_VAL", Tests, "spirv-val on PATH", "The SPIR-V validator the kernel tests use", ["test"]),
        new("PATH", Tests, "as the system sets it", "Where the coding tools and the tests look for programs", ["agent", "tools test", "test"]),
        new("PATHEXT", Tests, "as the system sets it", "Program extensions on Windows (the coding tools' program search)", ["agent", "tools test"]),
        new("PROCESSOR_IDENTIFIER", Devices, "as the system sets it", "The processor's name on Windows (device listings, tuning keys)", OnDevice),

        new("HTTPS_PROXY", DotNet, "none", "The proxy for https downloads (read by .NET's HTTP client; also https_proxy)", Network),
        new("HTTP_PROXY", DotNet, "none", "The proxy for http downloads (read by .NET's HTTP client)", Network),
        new("NO_PROXY", DotNet, "none", "Hosts reached without the proxy (read by .NET's HTTP client)", Network),
        new("DOTNET_GCHeapHardLimit", DotNet, "not set", "Caps the .NET heap reservation; needed on Android, whose address space is too small for the default", [Every]),
        new("DOTNET_CLI_TELEMETRY_OPTOUT", Child, "1", "Set for the commands the coding tools run (no telemetry prompt)", ChildCommands),
        new("DOTNET_NOLOGO", Child, "1", "Set for the commands the coding tools run (no logo)", ChildCommands),
        new("DOTNET_SKIP_FIRST_TIME_EXPERIENCE", Child, "1", "Set for the commands the coding tools run (no first-run text)", ChildCommands),
        new("CI", Child, "1", "Set for the commands the coding tools run (non-interactive tools)", ChildCommands),
        new("FORCE_COLOR", Child, "0", "Set for the commands the coding tools run (no colour)", ChildCommands),
        new("NG_CLI_ANALYTICS", Child, "false", "Set for the commands the coding tools run (no analytics prompt)", ChildCommands),
        new("npm_config_yes", Child, "true", "Set for the commands the coding tools run (no npm questions)", ChildCommands),
        new("npm_config_fund", Child, "false", "Set for the commands the coding tools run (no funding messages)", ChildCommands),
        new("npm_config_audit", Child, "false", "Set for the commands the coding tools run (no audit)", ChildCommands),
    ];

    /// <summary>The variable named <paramref name="name"/>, or null.</summary>
    public static Variable? Find(string name) => All.FirstOrDefault(v => v.Name == name);

    /// <summary>What a listing shows for the variable's value: null when not set, "set" for a secret.</summary>
    public static string? Value(Variable variable) => Environment.GetEnvironmentVariable(variable.Name) is { Length: > 0 } value ? variable.Secret ? "set" : value : null;

    /// <summary>The variables that affect the command named <paramref name="command"/> (not those affecting every command).</summary>
    public static IEnumerable<Variable> For(string command) => All.Where(v => v.Commands.Contains(command) && !v.Commands.Contains(Every));

    /// <summary>
    /// The help section listing the variables that affect a command (the end of <c>idrak help COMMAND</c>); "" when none
    /// beyond those every command reads. Groups may call it for extra names too.
    /// </summary>
    public static string HelpSection(string command, params string[] extra)
    {
        var variables = For(command).Concat(extra.Select(Find).OfType<Variable>()).DistinctBy(v => v.Name).ToList();
        var text = new StringBuilder("\nEnvironment (idrak help env for all):\n");
        if (variables.Count > 8)
        {
            // Long lists (the device and tuning variables) by group, names only.
            foreach (var group in variables.GroupBy(v => v.Group))
            {
                Names(text, group.Key, group.Select(v => v.Name));
            }
        }
        else
        {
            int width = variables.Count == 0 ? 0 : variables.Max(v => v.Name.Length);
            foreach (var v in variables)
            {
                text.Append("  ").Append(v.Name.PadRight(width)).Append("  ").Append(v.Meaning).Append('\n');
            }
        }

        Names(text, "Every command", All.Where(v => v.Commands.Contains(Every)).Select(v => v.Name));
        return text.ToString();
    }

    // "  Label: A, B, C" wrapped at the help's width, continuation lines indented under the first name.
    private static void Names(StringBuilder text, string label, IEnumerable<string> names)
    {
        const int Width = 118;
        string indent = new(' ', label.Length + 4);
        var line = new StringBuilder("  ").Append(label).Append(": ");
        bool first = true;
        foreach (string name in names)
        {
            string piece = first ? name : ", " + name;
            if (!first && line.Length + piece.Length > Width)
            {
                text.Append(line.Append(',')).Append('\n');
                line.Clear().Append(indent).Append(name);
            }
            else
            {
                line.Append(piece);
            }

            first = false;
        }

        text.Append(line).Append('\n');
    }

    /// <summary>The whole reference without values (<c>idrak help env</c>): every variable by group, its default and meaning.</summary>
    public static string Reference()
    {
        var text = new StringBuilder();
        int width = All.Max(v => v.Name.Length);
        foreach (var group in All.GroupBy(v => v.Group))
        {
            text.Append('\n').Append(group.Key).Append(":\n");
            foreach (var v in group)
            {
                text.Append("  ").Append(v.Name.PadRight(width)).Append("  ").Append(v.Meaning).Append(" (default: ").Append(v.Default).Append(")\n");
            }
        }

        return text.Append("\nTokens and keys are shown as set or not set, never their values.\n").ToString();
    }
}
