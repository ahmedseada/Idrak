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

    // Commands that read a model's tokenizer (TokenizerOptions' variables).
    private static readonly string[] Tokenizing = ["tokenize", "data tokens", "chat", "run", "serve", "create", "pretrain", "train"];

    // Commands whose child processes get the variables the coding tools set.
    private static readonly string[] ChildCommands = ["agent", "tools test"];

    private const string Child = "Set for the coding tools' commands";

    /// <summary>The group of the variables the tool sets for the coding tools' commands (not settings).</summary>
    public const string ChildGroup = Child;

    private const string Devices = "Devices and backends", Tuning = "Tuning and caches", Precision = "Precision and memory", Cpu = "CPU",
        Vulkan = "Vulkan", Hip = "HIP", Sources = "Model and data sources", Audio = "Audio", Tool = "The tool", Tests = "Tests and diagnostics", DotNet = ".NET",
        Cluster = "Cluster";

    // Plan 27, phase R2: read by ReinforcementOptions.FromEnvironment and ModulePolicyOptions.FromEnvironment in
    // Idrak.Reinforcement (the library and its tests), and by idrak reinforce (R5), whose command line wins.
    private const string Reinforcement = "Reinforcement learning";

    private static readonly string[] ReinforcementRun = ["reinforce"];

    // Plan 27, phases R3 and R5: the language-model policy's and on-policy distillation's settings (idrak reinforce, and
    // idrak distill --on-policy).
    private static readonly string[] OnPolicyRun = ["reinforce", "distill"];
    // Plan 27, phase R6: read by OffPolicyOptions.FromEnvironment in Idrak.Reinforcement (the off-policy trainer, DQN: the
    // library and its tests); no command of the tool trains off-policy yet.
    private static readonly string[] OffPolicyRun = ["test"];

    // Plan 27, phase R4: the command sandboxes (CommandSandboxes; the coding tools' commands, the coding environment's
    // builds and tests): idrak agent and the tests; idrak reinforce (R5) joins them.
    private const string Sandbox = "Command sandbox";

    private static readonly string[] SandboxRun = ["agent", "reinforce", "test"];

    // Plan 23: read by ClusterOptions.FromEnvironment (the library and its tests) and by the commands that start ranks
    // (cluster sim from phase C2; cluster start and --cluster on the training commands from C5 and C8).
    private static readonly string[] ClusterRun =
    [
        "cluster sim", "cluster start", "cluster bench", "cluster ui", "cluster replay", "cluster calibrate", "cluster inspect", "transports", "pretrain", "train", "tune",
        "resume", "distill", "profile", "tuning show", "test",
    ];

    // Plan 23, phase C9: the dashboard (cluster ui, and cluster sim --dashboard).
    private static readonly string[] ClusterUi = ["cluster ui", "cluster sim"];

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
        new("IDRAK_TRACK_STORAGES", Precision, "not set", "1 or true: record every storage a device makes with the method that made it, for Backend.LiveStorages (a debug listing)", OnDevice),
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
        new("IDRAK_ESPEAK", Sources, "espeak-ng on the PATH", "The espeak-ng program the phonemizer runs for a speech model whose text front end wants phonemes (Idrak.Audio.Plugins.EspeakNg.EspeakNgPhonemizer, registered with the VITS family)", ["speak", "pretrain", "tune", "serve"]),
        new("IDRAK_ESPEAK_VOICE", Sources, "the checkpoint's language", "The espeak-ng voice the phonemizer uses instead of the model's language", ["speak", "pretrain", "tune", "serve"]),
        new("IDRAK_MP3_BITRATE", Audio, "128 for stereo, 64 for mono (96 and 64 below 32 kHz)", "The MP3 encoder's constant bit rate in kbit/s (Mp3EncodeOptions.Bitrate)", ["speak"]),
        new("IDRAK_MP3_RATE", Audio, "the sound's rate", "The rate MP3 is written at, in Hz: 32000, 44100, 48000, 16000, 22050 or 24000", ["speak"]),
        new("IDRAK_MP3_CHANNELS", Audio, "joint", "How two channels are written to MP3: joint, stereo or mono (the channels' mean)", ["speak"]),
        new("IDRAK_MP3_QUALITY", Audio, "standard", "The MP3 encoder's effort a granule: fast, standard or high", ["speak"]),
        new("IDRAK_MP3_LOWPASS", Audio, "by the bit rate", "The MP3 encoder's low-pass in Hz; 0: none", ["speak"]),
        new("IDRAK_MP3_FRAMES_PER_TASK", Audio, "by the length", "Frames an MP3 encoding task takes at once (the frames are encoded in parallel)", ["speak"]),
        new("IDRAK_TOKENIZER_FORMAT", Sources, "the first file found: tokenizer.json, then a SentencePiece model, then vocab.txt", "Which tokenizer file a model folder is read from when it holds several: tokenizer.json, sentencepiece, vocab.txt or a plug-in's reader (TokenizerOptions.Format)", Tokenizing),
        new("IDRAK_TOKENIZER_CACHE", Sources, "100000", "Encoded words a tokenizer keeps, so a repeated word is looked up instead of encoded again; 0: none (TokenizerOptions.PieceCacheLimit)", Tokenizing),
        new("IDRAK_TOKENIZER_MAX_WORD_CHARS", Sources, "the tokenizer's own (100)", "WordPiece: a longer word (in characters) is one unknown token (TokenizerOptions.MaxInputCharsPerWord)", Tokenizing),
        new("IDRAK_TOKENIZER_ALPHA", Sources, "not set (sampling off)", "Unigram: subword regularization, each encode samples a segmentation with probability by exp(alpha x its score) (TokenizerOptions.SamplingAlpha)", Tokenizing),
        new("IDRAK_TOKENIZER_NBEST", Sources, "-1 (the whole lattice)", "Unigram sampling: draw from this many best segmentations; 1: the best; 0 or less: the whole lattice (TokenizerOptions.SamplingBest)", Tokenizing),
        new("IDRAK_TOKENIZER_SEED", Sources, "not set (unseeded)", "Unigram sampling: the seed of the tokenizer's draws, so the same encodes sample the same segmentations (TokenizerOptions.SamplingSeed)", Tokenizing),
        new("IDRAK_TOKENIZER_WORKERS", Sources, "0 (every core)", "Threads a batch encode spreads over (TokenizerOptions.Workers)", Tokenizing),
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
        new("IDRAK_API_KEY", Tool, "not set", "The key serve requires and the client commands (api, ping, server ...) send when --api-key is not given", ["serve", "ui", "api", "ping", "server ps", "server stop", "server load", "server unload", "cluster ui", "cluster sim"], Secret: true),
        new("IDRAK_PROGRESS", Tool, "console on a terminal, else none", "The progress reporter: console (one line redrawn in place), plain (a line at intervals), json (a JSON object a line) or quiet; also read by ProgressReporters.Start in the library (an app's choice wins)", [Every]),
        new("IDRAK_PLUGIN_PACKAGES", Tool, "text,vision,audio,reinforcement", "The plug-in packages the tool registers at startup: none, or a comma list of text, vision, audio and reinforcement (Idrak.Text.Plugins, Idrak.Vision.Plugins, Idrak.Audio.Plugins, Idrak.Reinforcement.Plugins); -P plug-ins load either way", [Every]),
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

        new("IDRAK_RL_ALGORITHM", Reinforcement, "ppo", "The algorithm: a PolicyLosses name (reinforce, ppo, grpo, or a plug-in's)", ReinforcementRun),
        new("IDRAK_RL_ESTIMATOR", Reinforcement, "the algorithm's (grpo: group; ppo: gae, returns without a critic; else returns)", "The advantage estimator: an AdvantageEstimators name (gae, group, leave-one-out, returns, or a plug-in's)", ReinforcementRun),
        new("IDRAK_RL_GROUP", Reinforcement, "auto (8, 4 or 2 dividing the episodes for group and leave-one-out, else 1)", "The episodes of a group: reset with the same seed and compared with each other", ReinforcementRun),
        new("IDRAK_RL_EPISODES", Reinforcement, "16", "The episodes of each iteration's rollout, played at once (a multiple of the group)", ReinforcementRun),
        new("IDRAK_RL_ENVIRONMENTS", Reinforcement, "every CPU thread (IDRAK_CPU_THREADS)", "The most environments stepped at once; results do not depend on it", ReinforcementRun),
        new("IDRAK_RL_PARALLEL_STEPS", Reinforcement, "auto (measured on the first rounds)", "on, off or auto: whether environments step in parallel", ReinforcementRun),
        new("IDRAK_RL_ITERATIONS", Reinforcement, "100", "The iterations of a run (a rollout, its advantages, the updates)", ReinforcementRun),
        new("IDRAK_RL_EPOCHS", Reinforcement, "the algorithm's (ppo 4, else 1)", "Passes over each rollout's episodes", ReinforcementRun),
        new("IDRAK_RL_MINIBATCH", Reinforcement, "0 (the whole rollout)", "Episodes per update (optimizer step)", ReinforcementRun),
        new("IDRAK_RL_GAMMA", Reinforcement, "0.99", "The discount γ of gae and returns", ReinforcementRun),
        new("IDRAK_RL_LAMBDA", Reinforcement, "0.95", "GAE's λ", ReinforcementRun),
        new("IDRAK_RL_GROUP_SCALE", Reinforcement, "the estimator's (true)", "true or false: the group estimator divides by the group's deviation", ReinforcementRun),
        new("IDRAK_RL_NORMALIZE_ADVANTAGES", Reinforcement, "the estimator's (false)", "true or false: gae and returns normalize the batch's advantages", ReinforcementRun),
        new("IDRAK_RL_CLIP", Reinforcement, "0.2", "The ratio's clip ε of ppo and grpo", ReinforcementRun),
        new("IDRAK_RL_KL", Reinforcement, "0", "The weight β of the KL penalty to the reference (above 0 needs a reference policy)", ReinforcementRun),
        new("IDRAK_RL_KL_ESTIMATOR", Reinforcement, "k3", "The KL estimator: k1, k2 or k3", ReinforcementRun),
        new("IDRAK_RL_ENTROPY", Reinforcement, "0", "The weight of the entropy bonus", ReinforcementRun),
        new("IDRAK_RL_VALUE", Reinforcement, "0.5", "The weight of the value head's squared error", ReinforcementRun),
        new("IDRAK_RL_NORMALIZE", Reinforcement, "the loss's (ppo, reinforce: token; grpo: sequence)", "How a policy loss divides the step's sum: token, sequence or constant", ReinforcementRun),
        new("IDRAK_RL_MAX_GRAD_NORM", Reinforcement, "1", "The largest gradient norm an update applies (0: no clip)", ReinforcementRun),
        new("IDRAK_RL_SCHEDULE", Reinforcement, "constant", "The learning-rate schedule over the run's updates: a LearningRateSchedules name", ReinforcementRun),
        new("IDRAK_RL_WARMUP", Reinforcement, "0", "The schedule's warm-up share of the updates", ReinforcementRun),
        new("IDRAK_RL_TEMPERATURE", Reinforcement, "1", "The temperature rollouts sample at (and idrak distill --on-policy's student)", OnPolicyRun),
        new("IDRAK_RL_MAX_STEPS", Reinforcement, "0 (the environment's own limit)", "The most steps of an episode before the trainer cuts it", ReinforcementRun),
        new("IDRAK_RL_SEED", Reinforcement, "0", "The run's seed: every episode, draw and shuffle derives from it", ReinforcementRun),
        new("IDRAK_RL_EVALUATE_EVERY", Reinforcement, "0 (no evaluation)", "Greedy evaluation episodes after every this many iterations", ReinforcementRun),
        new("IDRAK_RL_EVALUATION_EPISODES", Reinforcement, "8", "The episodes of an evaluation", ReinforcementRun),
        new("IDRAK_RL_SAVE_EVERY", Reinforcement, "0 (at the end only)", "A checkpoint after every this many iterations", ReinforcementRun),
        new("IDRAK_RL_CHECKPOINTS", Reinforcement, "none", "The folder checkpoints are written to (iteration-N folders)", ReinforcementRun),
        new("IDRAK_RL_RESUME", Reinforcement, "none", "A checkpoint to resume from: an iteration-N folder, or a checkpoint folder (its latest)", ReinforcementRun),
        new("IDRAK_RL_LOG", Reinforcement, "none", "A file each iteration is appended to as a JSON line", ReinforcementRun),
        new("IDRAK_RL_OPTIMIZER", Reinforcement, "adam", "A module policy's optimizer: an OptimizerKinds name", ReinforcementRun),
        new("IDRAK_RL_LEARNING_RATE", Reinforcement, "0.0003", "A module policy's learning rate (the schedule's base)", ReinforcementRun),
        new("IDRAK_RL_DQN_LOSS", Reinforcement, "dqn", "The off-policy trainer's algorithm: a QLosses name (dqn, or an app's)", OffPolicyRun),
        new("IDRAK_RL_DQN_DOUBLE", Reinforcement, "the loss's (off)", "on or off: double Q-learning (van Hasselt et al. 2016), the next action chosen by the online network and valued by the target one", OffPolicyRun),
        new("IDRAK_RL_DQN_HUBER", Reinforcement, "the loss's (1)", "The Huber loss's δ of DQN's temporal-difference error (0: the squared error)", OffPolicyRun),
        new("IDRAK_RL_DQN_STEPS", Reinforcement, "100000", "The environment steps of an off-policy run (every environment's together)", OffPolicyRun),
        new("IDRAK_RL_DQN_ENVIRONMENTS", Reinforcement, "1", "The environments an off-policy run plays side by side (a round steps each once)", OffPolicyRun),
        new("IDRAK_RL_DQN_REPLAY", Reinforcement, "100000", "The transitions the replay buffer keeps (fixed when the trainer is made; the oldest overwritten)", OffPolicyRun),
        new("IDRAK_RL_DQN_BATCH", Reinforcement, "32", "The transitions of each off-policy update's mini-batch", OffPolicyRun),
        new("IDRAK_RL_DQN_LEARNING_STARTS", Reinforcement, "100", "The steps taken with random actions, and before any update, to fill the replay buffer", OffPolicyRun),
        new("IDRAK_RL_DQN_TRAIN_EVERY", Reinforcement, "4", "The rounds between two off-policy trainings", OffPolicyRun),
        new("IDRAK_RL_DQN_GRADIENT_STEPS", Reinforcement, "1", "The updates of each off-policy training", OffPolicyRun),
        new("IDRAK_RL_DQN_TARGET_EVERY", Reinforcement, "10000", "The environment steps between two updates of DQN's target network", OffPolicyRun),
        new("IDRAK_RL_DQN_TAU", Reinforcement, "1", "How far each target update moves the target network to the online one (1: a copy; below 1: Polyak averaging)", OffPolicyRun),
        new("IDRAK_RL_DQN_EPSILON_START", Reinforcement, "1", "ε of the ε-greedy actions at the start of an off-policy run", OffPolicyRun),
        new("IDRAK_RL_DQN_EPSILON_END", Reinforcement, "0.05", "ε once its linear decay is over", OffPolicyRun),
        new("IDRAK_RL_DQN_EPSILON_FRACTION", Reinforcement, "0.1", "The share of the run's steps over which ε falls from its start to its end", OffPolicyRun),
        new("IDRAK_RL_DQN_LEARNING_RATE", Reinforcement, "0.0001", "The off-policy trainer's learning rate (the schedule's base)", OffPolicyRun),
        new("IDRAK_RL_DQN_MAX_GRAD_NORM", Reinforcement, "10", "The largest gradient norm an off-policy update applies (0: no clip)", OffPolicyRun),
        new("IDRAK_RL_DQN_REPORT_EVERY", Reinforcement, "1000", "The environment steps between two reports of an off-policy run (progress, telemetry, the log's line)", OffPolicyRun),
        new("IDRAK_RL_DQN_EVALUATE_EVERY", Reinforcement, "0 (no evaluation)", "Greedy evaluation episodes of an off-policy run after every this many steps", OffPolicyRun),
        new("IDRAK_RL_DQN_SAVE_EVERY", Reinforcement, "0 (at the end only)", "A checkpoint of an off-policy run (networks, optimizer, replay buffer, episodes under way) after every this many steps", OffPolicyRun),

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
        new("IDRAK_CLUSTER_CHUNK_BYTES", Cluster, "measured at the first collective over a real transport, kept in the tuning cache", "The bytes of a chunk in flight (B, KB, MB, GB); on the simulator unset is one message a piece", ClusterRun),
        new("IDRAK_CLUSTER_CHUNK_PROBE", Cluster, "4 MiB", "The all-reduce each candidate chunk size is timed with when the chunk is measured (B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_TUNING_FILE", Cluster, "the tuning cache's cluster/tuning.txt", "The file the measured chunk and bucket sizes are kept in (empty: none kept)", ClusterRun),
        new("IDRAK_CLUSTER_IN_FLIGHT", Cluster, "2", "Chunks in flight each way: messages a peer has not taken yet, per peer and tag", ClusterRun),
        new("IDRAK_CLUSTER_RECEIVE_BYTES", Cluster, "none (IDRAK_CLUSTER_IN_FLIGHT bounds it)", "The most bytes received from one peer under one tag and not yet taken; a larger message travels alone (B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_ASYNC_STAGING", Cluster, "1", "1 or 0: a finished gradient bucket is copied to the host in the background (pinned slots), not on the training thread", ClusterRun),
        new("IDRAK_CLUSTER_DEVICE_COLLECTIVES", Cluster, "1", "1 or 0: collectives over pcie-host or nvlink keep chunks on the device and sum them there (the same sums)", ClusterRun),
        new("IDRAK_CLUSTER_DEVICE_ACTIVATIONS", Cluster, "1", "1 or 0: a pipeline's activations cross between stages device to device over pcie-host or nvlink (the same values)", ClusterRun),
        new("IDRAK_CLUSTER_DIRECT_WAKE", Cluster, "1", "1 or 0: a waiting receive on a real transport is woken by the thread that queued its message, not through the thread pool", ClusterRun),
        new("IDRAK_CLUSTER_OWN_TENSORS", Cluster, "1", "1 or 0: a pipeline stage or a parameters shard reads only its own tensors from the checkpoint (0: every rank loads the whole model first)", ClusterRun),
        new("IDRAK_CLUSTER_STREAMED_SAVE", Cluster, "1", "1 or 0: a pipeline's or a sharded run's checkpoint is streamed to rank 0 a tensor at a time (0: every share gathered whole first)", ClusterRun),
        new("IDRAK_CLUSTER_SOCKET_BUFFER", Cluster, "the system's", "The socket buffers' size (B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_BUCKET_BYTES", Cluster, "measured on the first step (kept in the tuning cache over a real transport)", "The size of a gradient bucket: gradients are reduced a bucket at a time, each sent once the backward pass finished it (B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_MESSAGE_COST", Cluster, "measured on this machine", "Seconds one collective's messages cost beyond their bytes, which the measured bucket size counts", ClusterRun),
        new("IDRAK_CLUSTER_OVERLAP", Cluster, "1", "1 or 0: gradient buckets are reduced while the backward pass continues (the same result either way)", ClusterRun),
        new("IDRAK_CLUSTER_COMPRESSION", Cluster, "top-k with IDRAK_CLUSTER_TOPK, bf16 with IDRAK_CLUSTER_WIRE=bf16, else none", "How gradient buckets are compressed on their way: none, bf16, top-k, or a plug-in's", ClusterRun),
        new("IDRAK_CLUSTER_WIRE", Cluster, "f32", "f32 or bf16: how gradients travel (sums are always made in float32)", ClusterRun),
        new("IDRAK_CLUSTER_TOPK", Cluster, "0 (every value)", "The share of gradient values sent, largest first, from 0 to 1", ClusterRun),
        new("IDRAK_CLUSTER_ERROR_FEEDBACK", Cluster, "1", "1 or 0: values top-k leaves out are added to the next step's", ClusterRun),
        new("IDRAK_CLUSTER_LOCAL_SGD", Cluster, "0 (every step)", "Local SGD: each rank steps on its own gradients and the weights are averaged every this many steps", ClusterRun),
        new("IDRAK_CLUSTER_SYNC_WEIGHTS", Cluster, "0 (never)", "Rank 0 sends its weights every this many steps (ranks of different device kinds round apart)", ClusterRun),
        new("IDRAK_CLUSTER_REBALANCE_EVERY", Cluster, "0 (the first warm steps only)", "Re-measure each rank's speed and re-split the batch every this many steps", ClusterRun),
        new("IDRAK_CLUSTER_REBALANCE_STEPS", Cluster, "3", "Warm steps each rank times before a step's rows are split by the ranks' measured speeds; 0 keeps equal splits", ClusterRun),
        new("IDRAK_CLUSTER_REBALANCE_TOLERANCE", Cluster, "0.1", "How far apart the ranks' measured speeds may be (fastest over slowest, minus 1) and still split equally", ClusterRun),
        new("IDRAK_CLUSTER_GRADIENT_SYNC", Cluster, "local-sgd with IDRAK_CLUSTER_LOCAL_SGD, else all-reduce", "How gradients are kept in step across the ranks (all-reduce, local-sgd, or a plug-in's)", ClusterRun),
        new("IDRAK_CLUSTER_START", Cluster, "broadcast", "broadcast or check: rank 0's starting values sent to every rank, or every rank's hash compared", ClusterRun),
        new("IDRAK_CLUSTER_SHARD", Cluster, "none", "none, optimizer, gradients or parameters: the state split across ranks (ZeRO-style)", ClusterRun),
        new("IDRAK_CLUSTER_GATHER_BYTES", Cluster, "each rank's share of the trained parameters", "With IDRAK_CLUSTER_SHARD=parameters: the most bytes of weights one layer gathers whole before its forward (B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_PLACEMENT", Cluster, "single", "single or pipeline: how the model is placed", ClusterRun),
        new("IDRAK_CLUSTER_MICROBATCHES", Cluster, "the gradient accumulation", "A pipeline's micro-batches in flight in a step (the step's sequences shared out among them)", ClusterRun),
        new("IDRAK_CLUSTER_SCHEDULE", Cluster, "gpipe", "A pipeline's schedule: the order of each stage's forward and backward passes", ClusterRun),
        new("IDRAK_CLUSTER_STAGES", Cluster, "one per rank", "A pipeline's stages, dividing the ranks; each stage's ranks are data-parallel replicas", ClusterRun),
        new("IDRAK_CLUSTER_PIPELINE_BALANCE", Cluster, "auto", "auto, bytes or memory: cut a pipeline's stages to equal bytes or by the ranks' declared memory (auto: memory when every rank declares it and they differ)", ClusterRun),
        new("IDRAK_CLUSTER_PIPELINE_CUTS", Cluster, "measured", "A pipeline's cuts by hand: the first top-level module of each stage after the first (\"3,6\")", ClusterRun),
        new("IDRAK_CLUSTER_SUGGEST_SHARE", Cluster, "0.2", "The share of a step spent on the network above which the advisor suggests sending less", ClusterRun),
        new("IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE", Cluster, "2^-7 (0.0078125, bf16's rounding step)", "The largest relative error of bfloat16 gradients the advisor accepts (a slice reduced over both wires)", ClusterRun),
        new("IDRAK_CLUSTER_SUGGEST", Cluster, "while someone listens", "Whether a run measures its first warm steps for the cluster advisor's suggestions (1 or 0)", ClusterRun),
        new("IDRAK_CLUSTER_SUGGEST_STEPS", Cluster, "3", "The warm steps the cluster advisor measures before it suggests (the first, cold step aside)", ClusterRun),
        new("IDRAK_CLUSTER_SUGGEST_SLICE", Cluster, "65536", "The gradient values the cluster advisor reduces over both wires to measure bf16's error", ClusterRun),
        new("IDRAK_CLUSTER_PROFILES_FILE", Cluster, "a file per link in the tuning cache's cluster folder", "The one file every calibrated link profile is kept in and read from (empty: none kept)", ClusterRun),
        new("IDRAK_CLUSTER_SEED", Cluster, "the topology file's", "The seed of a simulation's jitter and faults", ClusterRun),
        new("IDRAK_CLUSTER_MODE", Cluster, "in-process", "in-process or localhost: where a simulation's ranks run", ClusterRun),
        new("IDRAK_CLUSTER_THROTTLE", Cluster, "1", "1 or 0: localhost transfers are paced to the topology's links", ClusterRun),
        new("IDRAK_CLUSTER_FAILURES", Cluster, "the topology file's", "The failure model of a simulation (none, slow-worker, dropped-worker, link-loss, partition, late-join, mixed)", ClusterRun),
        new("IDRAK_CLUSTER_REPLAY_STEP", Cluster, "not set: the whole run", "The step a replay reruns", ClusterRun),
        new("IDRAK_CLUSTER_CLOCK", Cluster, "wall", "wall or simulated: the clock throughput figures use", ClusterRun),
        new("IDRAK_CLUSTER_CARVE_FROM", Cluster, "each topology entry's own device", "The device on this machine every fake device of a simulation is carved from (cuda:0, cpu), so a topology written for other machines runs here", ClusterRun),
        new("IDRAK_CLUSTER_EVENT_LOG", Cluster, "the run folder's (cluster sim), else kept in memory", "The file a simulation's event log (faults, each rank's clock per step, state changes, transfers) is written to", ClusterRun),
        new("IDRAK_CLUSTER_EVENT_LOG_FLUSH", Cluster, "1", "Steps between the event log's writes to its file while a simulation runs (0: at the end only)", ClusterRun),
        new("IDRAK_CLUSTER_LOG_TRANSFERS", Cluster, "1", "1 or 0: a simulation's event log, and a run's log (transfers over real transports), record every transfer", ClusterRun),
        new("IDRAK_CLUSTER_FAULT_RATE", Cluster, "0.1", "The chance a simulated step draws a fault (the built-in failure models), from 0 to 1", ClusterRun),
        new("IDRAK_CLUSTER_SLOW_FACTOR", Cluster, "4", "How many times as long a slow worker's transfers take, 1 or more", ClusterRun),
        new("IDRAK_CLUSTER_FAULT_DURATION", Cluster, "0 (the rest of the run)", "Seconds of simulated time a slow worker, a lossy link or a partition lasts; a partition for the rest of the run, or the timeout or longer, times out", ClusterRun),
        new("IDRAK_CLUSTER_LOSS_SHARE", Cluster, "0.2", "The share of messages a lossy link loses on top of its profile's, from 0 to 1", ClusterRun),
        new("IDRAK_CLUSTER_LATE_STEPS", Cluster, "2", "How many steps late a late joiner connects", ClusterRun),
        new("IDRAK_CLUSTER_DEVICE", Cluster, "the rank's topology entry", "The device this rank runs on (cluster start sets it for each rank it starts)", ClusterRun),
        new("IDRAK_CLUSTER_RUN", Cluster, "derived from the coordinator", "A name the ranks of one run share, so two runs on one machine never meet (the launchers set a new one per run)", ClusterRun),
        new("IDRAK_CLUSTER_PARENT", Cluster, "not set", "The launcher's process id: a rank ends when its launcher is gone (the launchers set it)", ClusterRun),
        new("IDRAK_CLUSTER_RETRY_DELAY", Cluster, "1", "Seconds before the first retry at the rendezvous, doubled at each later one", ClusterRun),
        new("IDRAK_CLUSTER_SHM_BYTES", Cluster, "4 MiB", "The size of each shared-memory ring (one per pair of ranks and direction on a machine; B, KB, MB, GB)", ClusterRun),
        new("IDRAK_CLUSTER_SHM_FOLDER", Cluster, "the system's temporary folder", "The folder the shared-memory rings are mapped from", ClusterRun),
        new("IDRAK_CLUSTER_SHM_SPIN", Cluster, "one doorbell wake, measured", "Seconds a shared-memory reader or writer spins before it blocks on its doorbell", ClusterRun),
        new("IDRAK_CLUSTER_SHM_CHECK", Cluster, "1", "Seconds between an idle shared-memory reader's checks that its writers' processes still run", ClusterRun),
        new("IDRAK_CLUSTER_CONTROL", Cluster, "not set (the dashboard sets it for a localhost run it starts)", "The control file a launcher pauses, resumes and stops a run's processes through: rank 0 reads it as each step begins (run, pause or stop)", ClusterRun),
        new("IDRAK_CLUSTER_CONTROL_POLL", Cluster, "0.2", "Seconds between rank 0's reads of the control file while the run is paused", ClusterRun),
        new("IDRAK_CLUSTER_UI_PORT", Cluster, "0 (the system picks, printed)", "The port the cluster dashboard listens on when --port is not given", ClusterUi),
        new("IDRAK_CLUSTER_UI_HOST", Cluster, "127.0.0.1", "The address the cluster dashboard listens on; off this machine, requests need an API key (idrak server keys add, IDRAK_API_KEY)", ClusterUi),
        new("IDRAK_CLUSTER_UI_HISTORY", Cluster, "20000", "Records the dashboard keeps per run for a page that connects late", ClusterUi),
        new("IDRAK_CLUSTER_UI_POLL", Cluster, "0.5", "Seconds between the dashboard's reads of a followed run folder while a page listens", ClusterUi),
        new("IDRAK_CLUSTER_PROBE_BYTES", Cluster, "32 MiB (at most an eighth of the free memory)", "The bytes a link measurement copies (same-device, shared-memory, pcie)", ["cluster init", "cluster inspect", "cluster ui", "test"]),
        // Plan 27, R3: read by Idrak.Text's language-model policy (LanguageModelPolicyOptions) and on-policy distillation
        // (OnPolicyDistillationOptions); idrak reinforce (R5) carries them.
        new("IDRAK_RL_TURNS", Reinforcement, "auto", "How a multi-turn episode of a language model trains: auto (one sequence when its ids are prefix-stable, else a sequence per turn), episode or turn", OnPolicyRun),
        new("IDRAK_RL_BEHAVIOR", Reinforcement, "recompute", "Where a language-model policy's behaviour log-probabilities come from: recompute (the training forward, once per rollout batch) or sampler (as sampled)", OnPolicyRun),
        new("IDRAK_RL_TEACHER_WEIGHT", Reinforcement, "0 (idrak reinforce: 1 with a teacher)", "The weight of the on-policy distillation term beside a policy loss (0: none)", OnPolicyRun),
        new("IDRAK_RL_DIVERGENCE", Reinforcement, "reverse", "The on-policy distillation divergence: forward, reverse or jsd (TeacherDivergences)", OnPolicyRun),
        new("IDRAK_RL_WORKSPACES", Reinforcement, "the CPU threads, fewer when the free disk holds fewer than two copies of a task's workspace each", "The most coding episodes running tools, builds and tests at once (the coding environment's workspaces)", ReinforcementRun),
        new("IDRAK_RL_KEEP_WORKSPACES", Reinforcement, "false", "true: the coding environment keeps every episode's workspace (and the prepared copies) for inspection instead of deleting them", ReinforcementRun),
        new("IDRAK_SANDBOX", Sandbox, "process", "The sandbox the coding tools run commands in: process (a plain process: programs allowed, time, output and memory limits), container (Docker or Podman; Idrak.Reinforcement.Plugins), or a plug-in's", SandboxRun),
        new("IDRAK_SANDBOX_PROGRAM", Sandbox, "docker, else podman, found on the PATH", "The container sandbox's program (a path, or a name on the PATH)", SandboxRun),
        new("IDRAK_SANDBOX_IMAGE", Sandbox, "none: the container sandbox refuses", "The image the container sandbox runs every command in (it holds the programs the commands start)", SandboxRun),
        new("IDRAK_SANDBOX_NETWORK", Sandbox, "off", "The container sandbox's network: off (none), on (the program's default network), or a network's name", SandboxRun),
        new("IDRAK_RL_DISTILL_MIX", Reinforcement, "0", "λ of on-policy distillation: the share of a step's prompts trained on the data's answers instead of the student's samples, in [0, 1]", OnPolicyRun),
        new("IDRAK_RL_ENV", Reinforcement, "prompts", "The environment kind idrak reinforce trains in: an EnvironmentKinds name (prompts, coding, or a plug-in's)", ReinforcementRun),
        new("IDRAK_RL_REWARD", Reinforcement, "the environment's (prompts: exact; coding: tests-passed)", "The reward expression of idrak reinforce's environment: a reward or metric name, or weighted terms such as exact:1,length:0.1", ReinforcementRun),
        new("IDRAK_RL_MAX_NEW", Reinforcement, "256", "The longest reply a language-model policy samples, in tokens (idrak reinforce's --max-new)", ReinforcementRun),
        new("IDRAK_RL_TEACHER", Reinforcement, "none", "A teacher model for idrak reinforce: its distributions on the model's own replies, a term beside the reward (or alone with --algorithm distill)", ReinforcementRun),
        new("IDRAK_RL_TEACHER_DEVICE", Reinforcement, "the model's device", "The device idrak reinforce loads the teacher on (cpu, cuda:N, vulkan:N)", ReinforcementRun),
        new("IDRAK_CODING_LANGUAGES", Sandbox, "all", "The coding languages the coding tools and agent work with (programs, folders skipped, prompt lines): all registered, detected in the workspace, none, or names such as csharp,typescript", SandboxRun),
        new("IDRAK_CSHARP_PACKAGES", Reinforcement, "the machine's NuGet folder", "The csharp environment's package folder every episode restores from (NUGET_PACKAGES for its commands; Idrak.Reinforcement.Plugins)", ReinforcementRun),
        new("IDRAK_CSHARP_PROTECT", Reinforcement, "Directory.Build.props/.targets, Directory.Packages.props, global.json, *Tests and *Test projects", "The csharp environment's protected files: globs (separated by commas) whose edits count as test edits", ReinforcementRun),
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
        new("NUGET_PACKAGES", Child, "the csharp environment's package folder", "Set for the csharp environment's commands when it has a package folder (IDRAK_CSHARP_PACKAGES)", ReinforcementRun),
        new("MSBUILDDISABLENODEREUSE", Child, "1", "Set for the csharp environment's commands unless node-reuse=true (build nodes end with their build)", ReinforcementRun),
        new("UseSharedCompilation", Child, "unset", "Set to false for the csharp environment's commands with build-server=false (each build compiles in its own process)", ReinforcementRun),
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
