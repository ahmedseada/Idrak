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
        "distill", "create", "pretrain", "suggest", "explain", "rag index", "rag ask", "rag search", "rag eval", "test", "trace", "demo", "onnx import", "onnx export",
        "onnx check", "vlm check", "transcribe", "speak", "voice add", "data mel",
    ];

    // Commands that read models from the hub or the local stores (a model argument, or an index's embedding model).
    private static readonly string[] Models =
    [
        "pull", "list", "rm", "show", "search", "verify", "convert", "diff", "inspect", "chat", "run", "batch", "compare", "complete", "embed",
        "tokenize", "template", "agent", "serve", "ui", "memory", "quantize", "merge", "bench", "eval", "perplexity", "profile", "check", "tune",
        "distill", "create", "pretrain", "data stats", "data tokenizer", "data tokens", "rag index", "rag ask", "rag search", "rag eval", "suggest", "vlm check",
        "transcribe", "speak", "voice add",
    ];

    // Commands that read or write the installed voices.
    private static readonly string[] VoiceStore = ["speak", "voice add", "voice list", "voice show", "voice remove", "voice export", "voice import"];

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

    /// <summary>The group of the variables the tool sets for the coding tools' commands (not settings).</summary>
    public const string ChildGroup = Child;

    private const string Devices = "Devices and backends", Tuning = "Tuning and caches", Precision = "Precision and memory", Cpu = "CPU",
        Vulkan = "Vulkan", Hip = "HIP", Sources = "Model and data sources", Audio = "Audio", Tool = "The tool", Tests = "Tests and diagnostics", DotNet = ".NET",
        Cluster = "Cluster";

    // Plan 23: read by ClusterOptions.FromEnvironment (the library and its tests) and by the commands that start ranks
    // (cluster sim from phase C2; cluster start and --cluster on the training commands from C5 and C8).
    private static readonly string[] ClusterRun = ["cluster sim", "test"];

    /// <summary>The table, in listing order.</summary>
    public static IReadOnlyList<Variable> All { get; } =
    [
        new("IDRAK_DISABLE_CUDA", Devices, "not set", "1 or true: CUDA devices are not looked for (the backend reports itself disabled)", OnDevice),
        new("IDRAK_DISABLE_VULKAN", Devices, "not set", "1 or true: Vulkan devices are not looked for", OnDevice),
        new("IDRAK_DISABLE_HIP", Devices, "not set", "1 or true: HIP devices are not looked for", OnDevice),
        new("IDRAK_VULKAN_DEFAULT", Devices, "not set", "1: a Vulkan GPU may be the default device (otherwise only by name until its kernels are tuned)", OnDevice),
        new("IDRAK_HIP_DEFAULT", Devices, "not set", "1: a HIP GPU may be the default device (after CUDA, before Vulkan)", OnDevice),
        new("IDRAK_RETRY_ON_HOST", Devices, "not set (a failing GPU kernel throws)", "1 or all: an operation whose GPU kernel fails runs again on the CPU (reported to telemetry, counted as a host call); a list of kinds (cuda,vulkan,hip): only those devices. Errors found after an operation returned can't be retried", OnDevice),
        new("IDRAK_CUDA_DEBUG", Devices, "not set", "1: synchronize after every CUDA kernel and name the one that failed (slow)", OnDevice),
        new("IDRAK_STFT_PATH", Devices, "auto (the host on the CPU, composed device operations elsewhere)", "host or composed: how the short-time Fourier transform of audio features is computed, over the options' auto", ["transcribe", "data mel"]),
        new("IDRAK_ATTENTION_PATH", Devices, "measured", "spans or composed: attention that can run either way (a bidirectional pass, ranges with a dense mask) through the key-range kernel or through the full scores, instead of the faster as measured on the device (composed only where the device's free memory holds the scores)", OnDevice),
        new("IDRAK_WINDOW_KERNELS", Devices, "on", "0 or false: sliding-window and soft-capped attention through basic operations over the whole cache instead of the attention kernels (to compare or isolate them)", OnDevice),
        new("IDRAK_POWER_SOURCE", Devices, "as the system reports", "ac or battery: the power source tuning choices are measured and kept under", OnDevice),

        new("IDRAK_CACHE", Tuning, "~/.cache/idrak", "The cache folder: downloaded models, tuning choices, compiled kernels (--cache wins for the tool)", [Every]),
        new("IDRAK_AUTOTUNE", Tuning, "1 (measure)", "0 or false: the formulas only, no measuring of kernel choices on the device", OnDevice),
        new("IDRAK_TUNING_CACHE", Tuning, "on, under the cache folder", "0, false or off: measured choices are not kept; a path: the CUDA tuning folder", Caches),
        new("IDRAK_TUNE_NEIGHBOUR_OCTAVES", Tuning, "1", "How far a measured kernel choice reaches, in octaves of the work that follows the data (a convolution's, product's or attention's batch and lengths): a size class none measured takes the nearest measured class's choice within it instead of measuring (ComputeResources.TuningNeighbourOctaves); 0: every class measured", OnDevice),
        new("IDRAK_TUNE_LOG", Tuning, "not set", "1: print every tuning measurement (candidates, time ratios, the choice) to the error output", OnDevice),
        new("IDRAK_CPU_TUNING_FILE", Tuning, "<cache>/cpu/tuning.tsv", "The CPU tuning file (empty: none)", Caches),
        new("IDRAK_VULKAN_TUNING_CACHE", Tuning, "<cache>/vulkan/tuning.tsv", "0 or false: no Vulkan tuning file; a path: the file", Caches),
        new("IDRAK_HIP_KERNEL_CACHE", Tuning, "<cache>/hip/kernels", "0 or false: compiled HIP kernels are not kept; a path: the folder", Caches),

        new("IDRAK_MATMUL", Precision, "float32", "bf16 (bfloat16) or fp8 (float8): the precision of matrix products outside any precision scope", OnDevice),
        new("IDRAK_FP8_DELAYED", Precision, "not set", "1: delayed column scaling for float8 products on CUDA", OnDevice),
        new("IDRAK_OFFLOAD", Precision, "not set", "1 or true: move tensors to system memory when device memory runs out", OnDevice),
        new("IDRAK_CPU_THREADS", Precision, "every core", "The most CPU threads the library's kernels, loaders and parsers use (ComputeResources.MaxCpuThreads; --threads wins for a command)", [Every]),
        new("IDRAK_CPU_MEMORY_LIMIT_MB", Precision, "unlimited", "The most MiB of tensor memory on the CPU (ComputeResources.CpuMemoryLimit)", [Every]),
        new("IDRAK_GPU_MEMORY_LIMIT_MB", Precision, "the card's memory", "The most MiB of tensor memory on each GPU (ComputeResources.GpuMemoryLimit)", OnDevice),
        new("IDRAK_GPU_MEMORY_RESERVE_MB", Precision, "a sixteenth of the card, at least 256", "MiB kept free on each GPU for the display and other programs (ComputeResources.GpuMemoryReserve); 0: none", OnDevice),
        new("IDRAK_OFFLOAD_RETURN_HEADROOM_MB", Precision, "an eighth of the card, at least 256", "MiB kept free when offloaded tensors come back to the GPU (ComputeResources.OffloadReturnHeadroom)", OnDevice),
        new("IDRAK_WORKSPACE_SHARE", Precision, "0.5", "The share of a device's free memory one operation's working buffers may take at once (unfolded convolution patches, composed attention scores, a measured batch); in (0, 1] (ComputeResources.WorkspaceShare)", OnDevice),
        new("IDRAK_GPU_CHUNK_MB", Precision, "32 of the device's allocation pages", "MiB of the chunks long-lived blocks (weights, attention caches) share on a GPU; 0: each block its own allocation", OnDevice),
        new("IDRAK_GPU_STACK_BYTES", Precision, "none beyond what kernels need", "The stack of each GPU thread, reserved for every thread the device can run at once", OnDevice),
        new("IDRAK_LOOKUP_TABLES", Precision, "auto", "auto, host or device: where a frozen model's separate token table lives (auto: on the GPU until it is full)", OnDevice),

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
        new("IDRAK_VULKAN_BARRIERS", Vulkan, "where storages are shared", "all: a barrier before every command (diagnostics: results that change with it point to a missed dependency)", OnDevice),
        new("IDRAK_VULKAN_GRAPHS", Vulkan, "on", "0 or false: no recorded graphs on Vulkan; steps run directly (diagnostics)", OnDevice),
        new("IDRAK_VULKAN_BARRIER_KERNELS", Vulkan, "none", "Kernel names (comma-separated; copy and fill for copies and fills): a barrier before each of their commands (diagnostics)", OnDevice),
        new("IDRAK_VULKAN_KERNEL_LOG", Vulkan, "none", "A file the names of the Vulkan commands recorded are appended to, once each (diagnostics)", OnDevice),
        new("VK_ICD_FILENAMES", Vulkan, "the loader's search", "The Vulkan loader: which driver (ICD) files to use (needed for Turnip on a phone)", [.. OnDevice, "setup android"]),
        new("VK_DRIVER_FILES", Vulkan, "the loader's search", "The Vulkan loader: the newer name of VK_ICD_FILENAMES", [.. OnDevice, "setup android"]),
        new("VK_INSTANCE_LAYERS", Vulkan, "none", "The Vulkan loader: layers to enable (e.g. validation), read by the loader itself", OnDevice),

        new("IDRAK_HIP_KERNELS", Hip, "on", "0 or false: every HIP operation through the host fallback", OnDevice),
        new("HIP_PATH", Hip, "not set", "Where the HIP SDK is (Windows)", OnDevice),
        new("ROCM_PATH", Hip, "/opt/rocm", "Where ROCm is (Linux)", OnDevice),

        new("IDRAK_VOICES", Sources, "voices/ under IDRAK_CACHE or ~/.cache/idrak", "The installed voices' folder (--voices wins; with --cache given and this unset: voices/ under it)", VoiceStore),
        new("IDRAK_ESPEAK", Sources, "espeak-ng on the PATH", "The espeak-ng program the phonemizer runs for a speech model whose text front end wants phonemes (Idrak.Audio.SpeechText.EspeakNgPhonemizer)", ["speak", "pretrain", "tune", "serve"]),
        new("IDRAK_ESPEAK_VOICE", Sources, "the checkpoint's language", "The espeak-ng voice the phonemizer uses instead of the model's language", ["speak", "pretrain", "tune", "serve"]),
        new("IDRAK_MP3_BITRATE", Audio, "128 for stereo, 64 for mono (96 and 64 below 32 kHz)", "The MP3 encoder's constant bit rate in kbit/s (Mp3EncodeOptions.Bitrate)", ["speak"]),
        new("IDRAK_MP3_RATE", Audio, "the sound's rate", "The rate MP3 is written at, in Hz: 32000, 44100, 48000, 16000, 22050 or 24000", ["speak"]),
        new("IDRAK_MP3_CHANNELS", Audio, "joint", "How two channels are written to MP3: joint, stereo or mono (the channels' mean)", ["speak"]),
        new("IDRAK_MP3_QUALITY", Audio, "standard", "The MP3 encoder's effort a granule: fast, standard or high", ["speak"]),
        new("IDRAK_MP3_LOWPASS", Audio, "by the bit rate", "The MP3 encoder's low-pass in Hz; 0: none", ["speak"]),
        new("IDRAK_MP3_FRAMES_PER_TASK", Audio, "by the length", "Frames an MP3 encoding task takes at once (the frames are encoded in parallel)", ["speak"]),
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
        new("IDRAK_PORT", Tool, "the config's serve.port, else 7317", "The port serve and ui listen on and the client commands call when --port is not given", ["serve", "ui", "api", "ping", "server ps", "server stop", "server load", "server unload"]),
        new("IDRAK_API_KEY", Tool, "not set", "The key serve requires and the client commands (api, ping, server ...) send when --api-key is not given", ["serve", "ui", "api", "ping", "server ps", "server stop", "server load", "server unload"], Secret: true),
        new("IDRAK_TRACE", Tool, "not set", "1 or true: errors print the stack as well", [Every]),
        new("IDRAK_LANG", Tool, "en", "The language of the tool's messages: en or ar (--lang and the config's \"lang\" win); JSON stays English", [Every]),
        new("IDRAK_LANG_RENDER", Tool, "auto", "How Arabic reaches the terminal: auto, visual (shaped and reordered by the tool), visual-right or logical (--lang-render and the config's \"lang-render\" win)", [Every]),
        new("NO_COLOR", Tool, "not set", "Set and not empty: no colour in the output", [Every]),
        new("TERM", Tool, "as the terminal sets it", "dumb: no colour in the output; mlterm...: the terminal shows Arabic itself (--lang-render auto)", [Every]),
        new("TERM_PROGRAM", Tool, "as the terminal sets it", "mintty or Apple_Terminal: the terminal shows Arabic right to left itself; vscode, iTerm.app, WezTerm: it does not (--lang-render auto)", [Every]),
        new("VTE_VERSION", Tool, "as the terminal sets it", "5800 or above (VTE 0.58, GNOME Terminal and others): the terminal shows Arabic right to left itself (--lang-render auto)", [Every]),
        new("KONSOLE_VERSION", Tool, "as the terminal sets it", "Set (Konsole): the terminal shows Arabic right to left itself (--lang-render auto)", [Every]),
        new("MLTERM", Tool, "as the terminal sets it", "Set (mlterm): the terminal shows Arabic right to left itself (--lang-render auto)", [Every]),
        new("WT_SESSION", Tool, "as the terminal sets it", "Set (Windows Terminal): Arabic is shaped and reordered by the tool (--lang-render auto)", [Every]),
        new("IDRAK_UPDATE_INDEX", Tool, "the package feed", "Where idrak update reads the published versions (a URL or a local file; mirrors and tests)", ["update"]),

        new("IDRAK_OVERRIDE_POLICY", Tests, "throw (an override's failure reaches the caller)", "What happens when an app's or plug-in's registration over a library default throws (until code sets a slot's policy): fallback (the library default answers), shadow or shadow:RATE (the library answers, the override is compared on RATE of the calls, 0.01 by default), throw; for every slot, and/or Registry/name=policy entries for single slots, separated by commas (RopeScalings/yarn=fallback). See idrak overrides", [Every]),
        new("IDRAK_DEVICES", Tests, "every listed device", "Devices the test runner uses, comma-separated (cpu, cuda:0, vulkan:0, ...)", ["test", "report"]),
        new("IDRAK_FILTER", Tests, "every test", "Run only tests whose names contain this text", ["test", "report"]),
        new("IDRAK_TIMEOUT", Tests, "300", "Seconds before a running test is reported as HANG and the run stops", ["test", "report"]),
        new("IDRAK_SPIRV_VAL", Tests, "spirv-val on PATH", "The SPIR-V validator the kernel tests use", ["test"]),
        new("IDRAK_PROFILE_COLD", Tests, "1", "1 or 0: a profile (ProfileOptions.FromEnvironment) runs its workload once cold first", ["test"]),
        new("IDRAK_PROFILE_WARM", Tests, "3", "The warm runs a profile measures", ["test"]),
        new("IDRAK_PROFILE_PROBES", Tests, "all", "The probes a profile runs, separated by commas (all, none, or names such as cpu,heap,memory)", ["test"]),
        new("IDRAK_PROFILE_STATISTIC", Tests, "median", "median, mean, min or max: the warm runs' figure a profile reports", ["test"]),
        new("IDRAK_PROFILE_FORMAT", Tests, "table", "The writer a profile is written with (table, markdown, csv, json, or one registered)", ["test"]),
        new("IDRAK_PROFILE_OUT", Tests, "the console", "A file a profile is appended to", ["test"]),

        new("IDRAK_CLUSTER_TOPOLOGY", Cluster, "none: one node with the devices given", "The topology file of a cluster run (idrak cluster init writes it)", ClusterRun),
        new("IDRAK_CLUSTER_NODE", Cluster, "the node named after this host", "Which node of the topology file this machine is", ClusterRun),
        new("IDRAK_RANK", Cluster, "from the topology file", "This process's rank; needed only on a node with several ranks when a process is started by hand", ClusterRun),
        new("IDRAK_WORLD_SIZE", Cluster, "the topology file's device count", "The number of ranks; with a file, only checked against it", ClusterRun),
        new("IDRAK_COORDINATOR", Cluster, "the topology's first node's address", "host:port where rank 0 listens", ClusterRun),
        new("IDRAK_CLUSTER_TRANSPORT", Cluster, "auto", "The transport (sim, shared-memory, pcie-host, nvlink, direct-link, tcp, or a plug-in's), or auto for the fastest available per pair", ClusterRun),
        new("IDRAK_CLUSTER_SELECTOR", Cluster, "fastest", "The transport selector", ClusterRun),
        new("IDRAK_CLUSTER_COLLECTIVE", Cluster, "hierarchical with several nodes, else ring", "The collective algorithm (naive, ring, hierarchical, or a plug-in's)", ClusterRun),
        new("IDRAK_CLUSTER_KEY", Cluster, "not set: nothing listens beyond this machine", "The shared secret every rank proves in its first message (at least 16 characters); never an option, never written", ClusterRun, Secret: true),
        new("IDRAK_CLUSTER_PORT", Cluster, "0 (the system picks)", "The port this rank listens on for its peers", ClusterRun),
        new("IDRAK_CLUSTER_TIMEOUT", Cluster, "120", "Seconds any wait for a peer lasts", ClusterRun),
        new("IDRAK_CLUSTER_RETRIES", Cluster, "3", "Connection attempts at the rendezvous", ClusterRun),
        new("IDRAK_CLUSTER_CHUNK_BYTES", Cluster, "measured at first use", "The bytes of a chunk in flight (B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_IN_FLIGHT", Cluster, "2", "Chunks in flight each way", ClusterRun),
        new("IDRAK_CLUSTER_SOCKET_BUFFER", Cluster, "the system's", "The socket buffers' size (B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_BUCKET_BYTES", Cluster, "16 MiB (measured against the step from phase C4)", "The size of a gradient bucket: gradients are reduced a bucket at a time (B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_WIRE", Cluster, "f32", "f32 or bf16: how gradients travel (sums are always made in float32)", ClusterRun),
        new("IDRAK_CLUSTER_TOPK", Cluster, "0 (every value)", "The share of gradient values sent, largest first, from 0 to 1", ClusterRun),
        new("IDRAK_CLUSTER_ERROR_FEEDBACK", Cluster, "1", "1 or 0: values top-k leaves out are added to the next step's", ClusterRun),
        new("IDRAK_CLUSTER_LOCAL_SGD", Cluster, "0 (every step)", "Synchronize the weights every this many steps instead of the gradients every step", ClusterRun),
        new("IDRAK_CLUSTER_SYNC_WEIGHTS", Cluster, "0 (never)", "Rank 0 sends its weights every this many steps (ranks of different device kinds round apart)", ClusterRun),
        new("IDRAK_CLUSTER_REBALANCE_EVERY", Cluster, "0 (the first warm steps only)", "Re-measure each rank's speed and re-split the batch every this many steps", ClusterRun),
        new("IDRAK_CLUSTER_REBALANCE_STEPS", Cluster, "3", "Warm steps each rank times before a step's rows are split by the ranks' measured speeds; 0 keeps equal splits", ClusterRun),
        new("IDRAK_CLUSTER_REBALANCE_TOLERANCE", Cluster, "0.1", "How far apart the ranks' measured speeds may be (fastest over slowest, minus 1) and still split equally", ClusterRun),
        new("IDRAK_CLUSTER_GRADIENT_SYNC", Cluster, "all-reduce", "How gradients are kept in step across the ranks (all-reduce, or a plug-in's)", ClusterRun),
        new("IDRAK_CLUSTER_START", Cluster, "broadcast", "broadcast or check: rank 0's starting values sent to every rank, or every rank's hash compared", ClusterRun),
        new("IDRAK_CLUSTER_SHARD", Cluster, "none", "none, optimizer, gradients or parameters: the state split across ranks", ClusterRun),
        new("IDRAK_CLUSTER_PLACEMENT", Cluster, "single", "single or pipeline: how the model is placed", ClusterRun),
        new("IDRAK_CLUSTER_MICROBATCHES", Cluster, "the gradient accumulation", "A pipeline's micro-batches in flight", ClusterRun),
        new("IDRAK_CLUSTER_SCHEDULE", Cluster, "gpipe", "A pipeline's schedule", ClusterRun),
        new("IDRAK_CLUSTER_SUGGEST_SHARE", Cluster, "0.2", "The share of a step spent on the network above which the advisor suggests sending less", ClusterRun),
        new("IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE", Cluster, "the reference's measured one", "The largest relative error of bfloat16 gradients the advisor accepts", ClusterRun),
        new("IDRAK_CLUSTER_PROFILES_FILE", Cluster, "the tuning cache's", "The calibrated link profiles' file", ClusterRun),
        new("IDRAK_CLUSTER_SEED", Cluster, "the topology file's", "The seed of a simulation's jitter and faults", ClusterRun),
        new("IDRAK_CLUSTER_MODE", Cluster, "in-process", "in-process or localhost: where a simulation's ranks run", ClusterRun),
        new("IDRAK_CLUSTER_THROTTLE", Cluster, "1", "1 or 0: localhost transfers are paced to the topology's links", ClusterRun),
        new("IDRAK_CLUSTER_FAILURES", Cluster, "the topology file's", "The failure model of a simulation (none, slow-worker, dropped-worker, link-loss, partition, late-join, mixed)", ClusterRun),
        new("IDRAK_CLUSTER_REPLAY_STEP", Cluster, "not set: the whole run", "The step a replay reruns", ClusterRun),
        new("IDRAK_CLUSTER_CLOCK", Cluster, "wall", "wall or simulated: the clock throughput figures use", ClusterRun),
        new("IDRAK_CLUSTER_PROBE_BYTES", Cluster, "32 MiB (at most an eighth of the free memory)", "The bytes a link measurement copies (same-device, shared-memory, pcie)", ["cluster init", "cluster inspect", "test"]),
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
        var text = new StringBuilder("\n").Append(Messages.T("Environment (idrak help env for all):")).Append('\n');
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

        Names(text, Messages.T("Every command"), All.Where(v => v.Commands.Contains(Every)).Select(v => v.Name));
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
