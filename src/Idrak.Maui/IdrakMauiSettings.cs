// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Abstraction.Devices;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>How a setting is edited.</summary>
public enum IdrakSettingKind
{
    /// <summary>Free text (a folder).</summary>
    Text,

    /// <summary>A number; empty means the library's own value.</summary>
    Number,

    /// <summary>One of <see cref="IdrakSetting.Choices"/>.</summary>
    Choice,

    /// <summary>A device name: "auto", "cpu" or a device such as "vulkan:0".</summary>
    Device,

    /// <summary>A secret (the token): kept in secure storage, shown hidden.</summary>
    Secret,
}

/// <summary>
/// One option of <see cref="IdrakMauiOptions"/> as a setting: its key in the settings store, its title and help for the
/// settings page, and its value as text (empty: the library's own value). <see cref="IdrakMauiSettings.Definitions"/>
/// lists them all; saving, loading, the settings page and the restart check all read this one list.
/// </summary>
public sealed class IdrakSetting
{
    private readonly Func<IdrakMauiOptions, string> _read;
    private readonly Action<IdrakMauiOptions, string> _write;

    internal IdrakSetting(string key, string title, string summary, IdrakSettingKind kind, Func<IdrakMauiOptions, string> read, Action<IdrakMauiOptions, string> write,
        bool needsRestart = false, string placeholder = "", IReadOnlyList<string>? choices = null, string group = "", (double Minimum, double Maximum, double Step)? range = null)
    {
        Group = group.Length > 0 ? group : GroupOf(key);
        Range = range;
        Key = key;
        Title = title;
        Summary = summary;
        Kind = kind;
        _read = read;
        _write = write;
        NeedsRestart = needsRestart;
        Placeholder = placeholder;
        Choices = choices ?? [];
    }

    /// <summary>The key in the settings store, after <see cref="IdrakMauiSettings.Prefix"/> ("cpu_threads").</summary>
    public string Key { get; }

    /// <summary>The title on the settings page.</summary>
    public string Title { get; }

    /// <summary>What it does, with the library's option and variable it maps to.</summary>
    public string Summary { get; }

    /// <summary>How it is edited.</summary>
    public IdrakSettingKind Kind { get; }

    /// <summary>Whether a change takes effect only at the next start.</summary>
    public bool NeedsRestart { get; }

    /// <summary>What an empty value means ("measured"), shown in an empty field.</summary>
    public string Placeholder { get; }

    /// <summary>The values a <see cref="IdrakSettingKind.Choice"/> takes.</summary>
    public IReadOnlyList<string> Choices { get; }

    /// <summary>The group the settings page shows it in ("General", "Storage", "Limits", "Models", "Chat", "Layout").</summary>
    public string Group { get; }

    /// <summary>For a number that has one, the range a slider shows (its least, its most, its step); null for the rest.</summary>
    public (double Minimum, double Maximum, double Step)? Range { get; }

    // The group of a setting by its key: the chat's, the GPU's and CPU's limits, storage, models in memory, the rest general.
    private static string GroupOf(string key) => key switch
    {
        _ when key.StartsWith("chat_", StringComparison.Ordinal) => "Chat",
        _ when key.StartsWith("terminal.", StringComparison.Ordinal) => "Terminal",
        "cache_folder" or "hf_token" or "storage_headroom_mb" or "packed_cache" or "packed_cache_folder" => "Storage",
        "model_keep_alive_s" or "max_loaded_models" or "unload_models_at" => "Models in memory",
        "wide_layout_dp" or "carousel_s" or "progress_interval_ms" or "prebuild_sections" or "section_fade_ms" or "frame_clock" or "live_fps" or "live_level_ms"
            or "window_keep" or "window_min_width_dp" or "window_min_height_dp" => "Layout and motion",
        "device" or "cpu_vectors" or "theme" => "General",
        _ => "Limits",
    };

    /// <summary>The value in <paramref name="options"/> as text ("" for the library's own).</summary>
    public string Get(IdrakMauiOptions options) => _read(options ?? throw new ArgumentNullException(nameof(options)));

    /// <summary>Sets the value in <paramref name="options"/> from text ("" or null for the library's own).</summary>
    /// <exception cref="FormatException">The text is not a value of the setting (the message says what is).</exception>
    public void Set(IdrakMauiOptions options, string? text) => _write(options ?? throw new ArgumentNullException(nameof(options)), text?.Trim() ?? "");
}

/// <summary>
/// The settings kept between runs: each <see cref="IdrakMauiOptions"/> value under <see cref="Prefix"/> and its key in an
/// <see cref="ISettingsStore"/> (MAUI's Preferences in an app), the token in an <see cref="ISecretStore"/> (secure storage).
/// </summary>
public sealed class IdrakMauiSettings(ISettingsStore store, ISecretStore secrets)
{
    /// <summary>What every key starts with.</summary>
    public const string Prefix = "idrak.";

    private const string DisabledFeaturesKey = Prefix + "disabled_features", SavedKey = Prefix + "saved", TokenKey = Prefix + "hf_token";

    /// <summary>Every setting, in the settings page's order.</summary>
    public static IReadOnlyList<IdrakSetting> Definitions { get; } = Define();

    /// <summary>Raised after <see cref="SaveAsync"/>, with the options saved.</summary>
    public event EventHandler<IdrakMauiOptions>? Saved;

    /// <summary>Whether settings were saved before (by <see cref="SaveAsync"/>).</summary>
    public bool HasSaved => store.Get(SavedKey) is not null;

    /// <summary>
    /// Applies what was saved over <paramref name="options"/>: each saved setting (an empty one is the library's own value),
    /// the features turned off and the token. Nothing saved: <paramref name="options"/> stays as it is. A saved value that
    /// no longer reads (an older app's) is skipped and named in the result.
    /// </summary>
    public async Task<IReadOnlyList<string>> LoadIntoAsync(IdrakMauiOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        var skipped = new List<string>();
        if (!HasSaved)
        {
            return skipped;
        }

        foreach (var setting in Definitions)
        {
            string? text = setting.Kind == IdrakSettingKind.Secret ? await secrets.GetAsync(TokenKey).ConfigureAwait(false) ?? "" : store.Get(Prefix + setting.Key);
            if (text is null)
            {
                continue;
            }

            try
            {
                setting.Set(options, text);
            }
            catch (FormatException e)
            {
                skipped.Add($"{setting.Title}: {e.Message}");
            }
        }

        if (store.Get(DisabledFeaturesKey) is { } disabled)
        {
            options.DisabledFeatures = new HashSet<string>(disabled.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries), StringComparer.OrdinalIgnoreCase);
        }

        return skipped;
    }

    /// <summary>Keeps every setting of <paramref name="options"/> (the token in secure storage), then raises <see cref="Saved"/>.</summary>
    public async Task SaveAsync(IdrakMauiOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        foreach (var setting in Definitions)
        {
            if (setting.Kind == IdrakSettingKind.Secret)
            {
                await secrets.SetAsync(TokenKey, setting.Get(options)).ConfigureAwait(false);
            }
            else
            {
                store.Set(Prefix + setting.Key, setting.Get(options));
            }
        }

        store.Set(DisabledFeaturesKey, string.Join(',', options.DisabledFeatures.Order(StringComparer.OrdinalIgnoreCase)));
        store.Set(SavedKey, "1");
        Saved?.Invoke(this, options);
    }

    /// <summary>Forgets every saved setting and the token: the next start takes the app's defaults.</summary>
    public async Task ClearAsync()
    {
        foreach (var setting in Definitions.Where(s => s.Kind != IdrakSettingKind.Secret))
        {
            store.Set(Prefix + setting.Key, null);
        }

        store.Set(DisabledFeaturesKey, null);
        store.Set(SavedKey, null);
        await secrets.SetAsync(TokenKey, null).ConfigureAwait(false);
    }

    /// <summary>The titles of the settings that differ between <paramref name="before"/> and <paramref name="after"/> and take effect only at the next start.</summary>
    public static IReadOnlyList<string> NeedRestart(IdrakMauiOptions before, IdrakMauiOptions after)
    {
        ArgumentNullException.ThrowIfNull(before);
        ArgumentNullException.ThrowIfNull(after);
        var titles = Definitions.Where(s => s.NeedsRestart && s.Get(before) != s.Get(after)).Select(s => s.Title).ToList();
        if (!IsAuto(before.Device) && IsAuto(after.Device))
        {
            titles.Add("Device (back to auto)");                                     // a chosen device stays the default until the app starts again
        }

        return titles;
    }

    internal static bool IsAuto(string device) => device.Length == 0 || device.Equals("auto", StringComparison.OrdinalIgnoreCase);

    private static IReadOnlyList<IdrakSetting> Define() =>
    [
        new("device", "Device", "The device models load on: auto (the best found), cpu, or a device such as vulkan:0 (Device.Default).", IdrakSettingKind.Device,
            o => o.Device, (o, t) => o.Device = DeviceName(t), placeholder: "auto"),
        new("cpu_vectors", "CPU vector paths", "Which vector paths the CPU's kernels take (ComputeResources.CpuVectors, IDRAK_CPU_VECTORS): automatic (the platform's intrinsics where the runtime reports them, "
            + "else portable 128-bit vectors), native (the intrinsics only), portable (the portable vectors only) or scalar (the plain loops). Every path gives the same results; "
            + "only the speed differs. The int4 kernels keep the lane shifts they took at their first run until the next start.", IdrakSettingKind.Choice,
            o => o.CpuVectors is { } paths ? VectorName(paths) : "", (o, t) => o.CpuVectors = VectorPaths(t), placeholder: "automatic", choices: ["automatic", "native", "portable", "scalar"]),
        new("theme", "Theme", "The phone's theme, light or dark.", IdrakSettingKind.Choice,
            o => o.Theme.ToString().ToLowerInvariant(), (o, t) => o.Theme = t.Length == 0 ? IdrakThemeChoice.System : Enum.TryParse<IdrakThemeChoice>(t, ignoreCase: true, out var theme) && Enum.IsDefined(theme)
                ? theme : throw new FormatException($"'{t}' is not a theme: system, light or dark."),
            placeholder: "system", choices: ["system", "light", "dark"]),
        new("cache_folder", "Cache and models folder", "Where models, prepared files and measured choices are kept (ComputeResources.CacheFolder, IDRAK_CACHE). Empty: the app's own storage.",
            IdrakSettingKind.Text, o => o.CacheFolder ?? "", (o, t) => o.CacheFolder = t.Length == 0 ? null : Path.IsPathFullyQualified(t) ? t
                : throw new FormatException($"'{t}' is not a full path."), needsRestart: true, placeholder: "the app's storage"),
        new("cpu_threads", "CPU threads", "The CPU threads the library may use (ComputeResources.MaxCpuThreads, IDRAK_CPU_THREADS).", IdrakSettingKind.Number,
            o => Text(o.MaxCpuThreads), (o, t) => o.MaxCpuThreads = (int?)Whole(t, 1, Environment.ProcessorCount), placeholder: "every core"),
        new("cpu_memory_limit_mb", "CPU memory limit (MiB)", "The most tensor memory on the CPU (ComputeResources.CpuMemoryLimit, IDRAK_CPU_MEMORY_LIMIT_MB).", IdrakSettingKind.Number,
            o => Text(o.CpuMemoryLimitMiB), (o, t) => o.CpuMemoryLimitMiB = Whole(t, 1), placeholder: "no limit"),
        new("gpu_memory_limit_mb", "GPU memory limit (MiB)", "The most tensor memory on each GPU (ComputeResources.GpuMemoryLimit, IDRAK_GPU_MEMORY_LIMIT_MB).", IdrakSettingKind.Number,
            o => Text(o.GpuMemoryLimitMiB), (o, t) => o.GpuMemoryLimitMiB = Whole(t, 1), placeholder: "what the device has"),
        new("gpu_memory_reserve_mb", "GPU memory kept free (MiB)", "GPU memory left for the system and other apps (ComputeResources.GpuMemoryReserve, IDRAK_GPU_MEMORY_RESERVE_MB).",
            IdrakSettingKind.Number, o => Text(o.GpuMemoryReserveMiB), (o, t) => o.GpuMemoryReserveMiB = Whole(t, 0), placeholder: "measured"),
        new("offload", "Offload to system memory", "Whether a full GPU places tensors in system memory (ComputeResources.OffloadToHostMemory, IDRAK_OFFLOAD).",
            IdrakSettingKind.Choice, o => Text(o.OffloadToHostMemory), (o, t) => o.OffloadToHostMemory = Switch(t), placeholder: "default", choices: ["default", "on", "off"]),
        new("offload_return_headroom_mb", "Offload return headroom (MiB)", "GPU memory kept free when offloaded tensors come back (ComputeResources.OffloadReturnHeadroom).",
            IdrakSettingKind.Number, o => Text(o.OffloadReturnHeadroomMiB), (o, t) => o.OffloadReturnHeadroomMiB = Whole(t, 0), placeholder: "measured"),
        new("gpu_chunk_mb", "GPU chunk (MiB)", "The chunk long-lived GPU blocks are placed in (ComputeResources.GpuChunkBytes, IDRAK_GPU_CHUNK_MB); 0: a block each.",
            IdrakSettingKind.Number, o => Text(o.GpuChunkMiB), (o, t) => o.GpuChunkMiB = Whole(t, 0), needsRestart: true, placeholder: "from the device's pages"),
        new("gpu_stack_bytes", "GPU thread stack (bytes)", "The stack each GPU thread gets (ComputeResources.GpuStackBytes, IDRAK_GPU_STACK_BYTES).",
            IdrakSettingKind.Number, o => Text(o.GpuStackBytes), (o, t) => o.GpuStackBytes = Whole(t, 0), needsRestart: true, placeholder: "what kernels need"),
        new("workspace_share", "Workspace share", "The share of available memory one operation's buffers may take, in (0, 1] (ComputeResources.WorkspaceShare).",
            IdrakSettingKind.Number, o => Text(o.WorkspaceShare), (o, t) => o.WorkspaceShare = Fraction(t), placeholder: "the library's", range: (0.05, 1, 0.05)),
        new("tuning_budget_ms", "Tuning budget (ms)", "How long a shape's first use may measure kernel choices (ComputeResources.TuningBudget, IDRAK_TUNE_BUDGET_MS).",
            IdrakSettingKind.Number, o => Text(o.TuningBudgetMs), (o, t) => o.TuningBudgetMs = Real(t, exclusiveMinimum: 0), placeholder: "the library's"),
        new("tuning_neighbour_octaves", "Tuning reach (octaves)", "How far a measured choice reaches (ComputeResources.TuningNeighbourOctaves, IDRAK_TUNE_NEIGHBOUR_OCTAVES).",
            IdrakSettingKind.Number, o => Text(o.TuningNeighbourOctaves), (o, t) => o.TuningNeighbourOctaves = Real(t, minimum: 0), placeholder: "the library's"),
        new("lookup_tables", "Lookup tables", "Where frozen lookup tables live (ComputeResources.LookupTables, IDRAK_LOOKUP_TABLES).", IdrakSettingKind.Choice,
            o => o.LookupTables ?? "", (o, t) => o.LookupTables = Placement(t), placeholder: "default", choices: ["default", "auto", "host", "device"]),
        new("packed_cache", "Keep packed weights", "Whether int8 and int4 weights are kept between loads, so the next load of a model reads them instead of quantizing again "
            + "(ComputeResources.PackedWeightCache, IDRAK_PACKED_CACHE). A kept entry is made again when its model's file changes.",
            IdrakSettingKind.Choice, o => Text(o.PackedWeightCache), (o, t) => o.PackedWeightCache = Switch(t), placeholder: "default", choices: ["default", "on", "off"]),
        new("packed_cache_folder", "Packed weights folder", "Where kept int8 and int4 weights live (ComputeResources.PackedWeightCacheFolder, IDRAK_PACKED_CACHE_DIR). Empty: \"packed\" in the cache folder.",
            IdrakSettingKind.Text, o => o.PackedWeightCacheFolder ?? "", (o, t) => o.PackedWeightCacheFolder = t.Length == 0 ? null : Path.IsPathFullyQualified(t) ? t
                : throw new FormatException($"'{t}' is not a full path."), placeholder: "packed, in the cache folder"),
        new("hf_token", "Hugging Face token", "For gated and private models, kept in the phone's secure storage (HF_TOKEN).", IdrakSettingKind.Secret,
            o => o.HuggingFaceToken ?? "", (o, t) => o.HuggingFaceToken = t.Length == 0 ? null : t, placeholder: "none"),
        new("storage_headroom_mb", "Storage kept free (MiB)", "The free space a model pull leaves; a pull that would leave less is refused before it starts.",
            IdrakSettingKind.Number, o => Text(o.StorageHeadroomMiB), (o, t) => o.StorageHeadroomMiB = Whole(t, 0), placeholder: "none"),
        new("progress_interval_ms", "Progress interval (ms)", "The least time between two updates of a progress view.", IdrakSettingKind.Number,
            o => Text(o.ProgressInterval.TotalMilliseconds), (o, t) => o.ProgressInterval = TimeSpan.FromMilliseconds(Real(t, exclusiveMinimum: 0) ?? 250), placeholder: "250"),
        new("wide_layout_dp", "Wide layout from (dp)", "The window width from which the app shows the side rail and pages in columns; below it, the bottom bar. It follows rotation and split screen.",
            IdrakSettingKind.Number, o => Text(o.WideLayoutWidth), (o, t) => o.WideLayoutWidth = Real(t, minimum: 320) ?? 840, placeholder: "840"),
        new("carousel_s", "Home carousel (s)", "How long Home's carousel shows a slide before the next; 0 keeps it still.", IdrakSettingKind.Number,
            o => o.CarouselInterval is { } c ? Text(c.TotalSeconds) : "0", (o, t) => o.CarouselInterval = Real(t, minimum: 0) is { } s && s > 0 ? TimeSpan.FromSeconds(s) : t.Length == 0 ? TimeSpan.FromSeconds(6) : null,
            placeholder: "6", range: (0, 20, 1)),
        new("prebuild_sections", "Make the tabs ahead", "Whether the tabs' views are made while the app is idle after Home first shows, so a first tap only shows a tab; off makes each when first opened.",
            IdrakSettingKind.Choice, o => o.PrebuildSections ? "on" : "off", (o, t) => o.PrebuildSections = Switch(t) ?? true, placeholder: "on", choices: ["on", "off"]),
        new("section_fade_ms", "Tab fade (ms)", "How long a tab's view fades in when shown; 0 shows it at once. The tap's command never waits for it.", IdrakSettingKind.Number,
            o => Text(o.SectionFade.TotalMilliseconds), (o, t) => o.SectionFade = TimeSpan.FromMilliseconds(Real(t, minimum: 0) ?? 0), placeholder: "0", range: (0, 400, 10)),
        new("frame_clock", "Live drawing clock", "What the live waveform and level meters draw on (FrameClocks): display (the display's own frames, on Android and Windows), timer (a timer at the display's rate), or auto.",
            IdrakSettingKind.Choice, o => o.FrameClock ?? "", (o, t) => o.FrameClock = FrameClockName(t), placeholder: "auto", choices: ["auto", .. FrameClocks.Names]),
        new("live_fps", "Live drawing rate (frames/s)", "The most frames a second the live waveform and meters draw. Empty: every frame the display shows.", IdrakSettingKind.Number,
            o => Text(o.LiveFrameRate), (o, t) => o.LiveFrameRate = Real(t, exclusiveMinimum: 0), placeholder: "the display's"),
        new("live_level_ms", "Waveform bar (ms)", "The sound one bar of the live waveform stands for (and one step of a level meter); taken at the next recording.", IdrakSettingKind.Number,
            o => Text(o.LiveLevelWindow.TotalMilliseconds), (o, t) => o.LiveLevelWindow = TimeSpan.FromMilliseconds(Real(t, exclusiveMinimum: 0) ?? 50), placeholder: "50", range: (10, 250, 5)),
        new("window_keep", "Keep the window's size and place", "Whether the main window opens at the size and place it had when the app last ran (a desktop's window; a phone's is the system's).",
            IdrakSettingKind.Choice, o => o.KeepWindowPlace ? "on" : "off", (o, t) => o.KeepWindowPlace = Switch(t) ?? true, placeholder: "on", choices: ["on", "off"]),
        new("window_min_width_dp", "Window: least width (dp)", "The narrowest the main window may be made on a desktop, in device-independent units.",
            IdrakSettingKind.Number, o => Text(o.WindowMinimumWidth), (o, t) => o.WindowMinimumWidth = Real(t, minimum: KeptWindowPlacement.LeastWidth) ?? 360, placeholder: "360"),
        new("window_min_height_dp", "Window: least height (dp)", "The shortest the main window may be made on a desktop, in device-independent units.",
            IdrakSettingKind.Number, o => Text(o.WindowMinimumHeight), (o, t) => o.WindowMinimumHeight = Real(t, minimum: KeptWindowPlacement.LeastHeight) ?? 480, placeholder: "480"),
        new("model_keep_alive_s", "Model keep-alive (s)", "How long a model stays loaded after its last use; empty: until memory pressure.", IdrakSettingKind.Number,
            o => o.ModelKeepAlive is { } k ? Text(k.TotalSeconds) : "", (o, t) => o.ModelKeepAlive = Real(t, minimum: 0) is { } s ? TimeSpan.FromSeconds(s) : null,
            placeholder: "until memory pressure"),
        new("max_loaded_models", "Models loaded at once", "The most models kept loaded in each host; loading another unloads the longest unused.", IdrakSettingKind.Number,
            o => Text(o.MaxLoadedModels), (o, t) => o.MaxLoadedModels = (int?)Whole(t, 1, 1024), placeholder: "no limit"),
        new("unload_models_at", "Unload models at", "The least memory pressure that unloads the models not in use.", IdrakSettingKind.Choice,
            o => o.UnloadModelsAt.ToString().ToLowerInvariant(), (o, t) => o.UnloadModelsAt = t.Length == 0 ? MemoryPressureLevel.Low
                : Enum.TryParse<MemoryPressureLevel>(t, ignoreCase: true, out var level) && Enum.IsDefined(level) ? level
                : throw new FormatException($"'{t}' is not a level: background, moderate, low or critical."),
            placeholder: "low", choices: ["background", "moderate", "low", "critical"]),
        new("chat_system_prompt", "Chat: system prompt", "The instructions every chat starts with, as a system message the model's own chat template renders. Empty: none.",
            IdrakSettingKind.Text, o => o.ChatSystemPrompt ?? "", (o, t) => o.ChatSystemPrompt = t.Length == 0 ? null : t, placeholder: "none"),
        new("chat_temperature", "Chat: temperature", "How varied the answers are (GenerationOptions.Temperature): 0 takes the likeliest token, above 1 more varied.",
            IdrakSettingKind.Number, o => Text(o.ChatTemperature), (o, t) => o.ChatTemperature = Real(t, minimum: 0), placeholder: Library.Temperature.ToString(CultureInfo.InvariantCulture),
            range: (0, 2, 0.05)),
        new("chat_top_p", "Chat: top-p", "Sample among the likeliest tokens holding this share, in (0, 1] (GenerationOptions.TopP); 1 is off.",
            IdrakSettingKind.Number, o => Text(o.ChatTopP), (o, t) => o.ChatTopP = Fraction(t), placeholder: Library.TopP.ToString(CultureInfo.InvariantCulture),
            range: (0.05, 1, 0.05)),
        new("chat_top_k", "Chat: top-k", "Sample among this many likeliest tokens (GenerationOptions.TopK); 0: all.",
            IdrakSettingKind.Number, o => Text(o.ChatTopK), (o, t) => o.ChatTopK = (int?)Whole(t, 0, int.MaxValue), placeholder: Text(Library.TopK)),
        new("chat_max_new_tokens", "Chat: max new tokens", "The most tokens one answer has (GenerationOptions.NumPredict). Empty: until the turn ends or the context is full.",
            IdrakSettingKind.Number, o => Text(o.ChatMaxNewTokens), (o, t) => o.ChatMaxNewTokens = (int?)Whole(t, 1, int.MaxValue), placeholder: "until the turn ends"),
        new("chat_context", "Chat: context (tokens)", "The context window (GenerationOptions.NumCtx), capped by the model's own; it sizes the key/value cache and the memory a model is checked against.",
            IdrakSettingKind.Number, o => Text(o.ChatContext), (o, t) => o.ChatContext = (int?)Whole(t, 16, int.MaxValue), placeholder: Text(Library.NumCtx)),
        new("chat_seed", "Chat: seed", "The random seed (GenerationOptions.Seed), for answers that repeat. Empty: random.",
            IdrakSettingKind.Number, o => Text(o.ChatSeed), (o, t) => o.ChatSeed = (int?)Whole(t, int.MinValue, int.MaxValue), placeholder: "random"),
        new("chat_thinking", "Chat: thinking", "Whether the chat asks for reasoning, where the model's template has a switch for it (ChatRequest.Think); the chat page switches it too.",
            IdrakSettingKind.Choice, o => Text(o.ChatThinking), (o, t) => o.ChatThinking = Switch(t), placeholder: "default", choices: ["default", "on", "off"]),
        new("chat_weights", "Chat: weights format", "The format a chat model's projections are kept in once loaded: bfloat16 (half of float32), int8, int4 or another registered packed format (PackedWeightFormats). Default: float32. Taken at the next load.",
            IdrakSettingKind.Choice, o => o.ChatWeights ?? "", (o, t) => o.ChatWeights = WeightFormat(t), placeholder: "default", choices: ["default", .. WeightFormats()]),
        new("chat_kv", "Chat: key/value cache format", "How the chat's key/value cache is kept (KeyValueLayouts): int8 or bfloat16 take less memory than float32. Default: float32. Taken at the next load.",
            IdrakSettingKind.Choice, o => o.ChatKeyValueFormat ?? "", (o, t) => o.ChatKeyValueFormat = CacheFormat(t), placeholder: "default",
            choices: ["default", .. KeyValueLayouts.Names.Order(StringComparer.Ordinal)]),
        .. TerminalSettings(),
    ];

    // The Terminal's limits (kept as idrak.terminal.*): an empty one is the app's own (TerminalLabOptions), shown in it.
    private static IEnumerable<IdrakSetting> TerminalSettings()
    {
        var own = new ViewModels.TerminalLabOptions();
        yield return new("terminal.lines", "Terminal: lines kept", "The most output lines the Terminal keeps; the oldest are dropped. Applied when the Terminal is next shown.",
            IdrakSettingKind.Number, o => Text(o.TerminalLines), (o, t) => o.TerminalLines = (int?)Whole(t, 1, 10_000_000), placeholder: Text(own.Lines));
        yield return new("terminal.history", "Terminal: history kept", "The most command lines the Terminal's history keeps between runs.",
            IdrakSettingKind.Number, o => Text(o.TerminalHistory), (o, t) => o.TerminalHistory = (int?)Whole(t, 1, 1_000_000), placeholder: Text(own.HistoryLimit));
        yield return new("terminal.history_shown", "Terminal: history shown", "The history's latest lines shown as chips in the commands pane (0: none).",
            IdrakSettingKind.Number, o => Text(o.TerminalHistoryShown), (o, t) => o.TerminalHistoryShown = (int?)Whole(t, 0, 10_000), placeholder: Text(own.HistoryShown));
        yield return new("terminal.suggestions", "Terminal: suggestions shown", "The most commands or options offered at once above the line.",
            IdrakSettingKind.Number, o => Text(o.TerminalSuggestions), (o, t) => o.TerminalSuggestions = (int?)Whole(t, 1, 10_000), placeholder: Text(own.Suggestions));
        yield return new("terminal.folder", "Terminal: working folder", "The folder the commands run in (relative paths read from it). Empty: the app's storage.",
            IdrakSettingKind.Text, o => o.TerminalFolder ?? "", (o, t) => o.TerminalFolder = t.Length == 0 ? null : Path.IsPathFullyQualified(t) ? t
                : throw new FormatException($"'{t}' is not a full path."), placeholder: "the app's storage");
    }

    // The library's generation defaults, which an empty chat setting keeps (shown as the placeholders).
    private static GenerationOptions Library => field ??= new GenerationOptions();

    // bfloat16, int8, int4 first, then the other registered packed formats.
    private static IEnumerable<string> WeightFormats() =>
        new[] { "bfloat16", "int8", "int4" }.Concat(PackedWeightFormats.Names.Order(StringComparer.Ordinal)).Distinct(StringComparer.OrdinalIgnoreCase);

    // "default" or "float32" is the library's own (null); bf16 is bfloat16's short name; anything else a registered packed format.
    private static string? WeightFormat(string text)
    {
        string name = text.ToLowerInvariant() switch { "" or "default" or "float32" or "f32" => "", "bf16" => "bfloat16", var other => other };
        if (name.Length == 0)
        {
            return null;
        }

        return PackedWeightFormats.Names.Contains(name, StringComparer.OrdinalIgnoreCase) || name is "bfloat16" or "int8" or "int4" ? name
            : throw new FormatException($"'{text}' is not a weights format: default, {string.Join(", ", WeightFormats())}.");
    }

    // "default" or "float32" is the library's own (null); a registered key/value layout otherwise.
    private static string? CacheFormat(string text)
    {
        string name = text.ToLowerInvariant() switch { "" or "default" or "float32" or "f32" => "", "bf16" => "bfloat16", var other => other };
        if (name.Length == 0)
        {
            return null;
        }

        return KeyValueLayouts.Names.Contains(name, StringComparer.OrdinalIgnoreCase) ? name
            : throw new FormatException($"'{text}' is not a key/value cache format: default, {string.Join(", ", KeyValueLayouts.Names.Order(StringComparer.Ordinal))}.");
    }

    private static string Text(long? value) => value?.ToString(CultureInfo.InvariantCulture) ?? "";

    private static string Text(int? value) => value?.ToString(CultureInfo.InvariantCulture) ?? "";

    private static string Text(double? value) => value?.ToString("R", CultureInfo.InvariantCulture) ?? "";

    private static string Text(bool? value) => value switch { true => "on", false => "off", null => "" };

    private static long? Whole(string text, long minimum, long maximum = long.MaxValue)
    {
        if (text.Length == 0)
        {
            return null;
        }

        return long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out long value) && value >= minimum && value <= maximum ? value
            : throw new FormatException(maximum == long.MaxValue ? $"'{text}' is not a whole number of {minimum} or more." : $"'{text}' is not a whole number from {minimum} to {maximum}.");
    }

    // A decimal number, with a comma read as the point (a phone's keyboard in many languages gives one).
    private static double? Real(string text, double? minimum = null, double? exclusiveMinimum = null)
    {
        if (text.Length == 0)
        {
            return null;
        }

        if (!double.TryParse(text.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out double value) || !double.IsFinite(value)
            || value < minimum || value <= exclusiveMinimum)
        {
            throw new FormatException(exclusiveMinimum is { } above ? $"'{text}' is not a number above {above.ToString(CultureInfo.InvariantCulture)}."
                : $"'{text}' is not a number of {(minimum ?? 0).ToString(CultureInfo.InvariantCulture)} or more.");
        }

        return value;
    }

    private static double? Fraction(string text)
    {
        double? share = Real(text, exclusiveMinimum: 0);
        return share > 1 ? throw new FormatException($"'{text}' is not in (0, 1].") : share;
    }

    private static bool? Switch(string text) => text.ToLowerInvariant() switch
    {
        "" or "default" => null,
        "on" or "true" or "1" => true,
        "off" or "false" or "0" => false,
        _ => throw new FormatException($"'{text}' is not on, off or default."),
    };

    // The setting's name of a vector path: "automatic", "native", "portable", "scalar".
    private static string VectorName(CpuVectorPaths paths) => paths.ToString().ToLowerInvariant();

    // "" is the library's own (null); otherwise a name ComputeResources.ParseCpuVectors takes (auto, automatic, native, portable, scalar).
    private static CpuVectorPaths? VectorPaths(string text)
    {
        if (text.Length == 0 || text.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        try
        {
            return ComputeResources.ParseCpuVectors(text);
        }
        catch (ArgumentException)
        {
            throw new FormatException($"'{text}' is not a vector path setting: automatic, native, portable or scalar.");
        }
    }

    private static string? Placement(string text)
    {
        if (text.Length == 0 || text.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        try
        {
            ComputeResources.ParseLookupTables(text);
            return text.ToLowerInvariant();
        }
        catch (ArgumentException e)
        {
            throw new FormatException(e.Message);
        }
    }

    // "auto" or empty is the first clock that ticks here (null); else a registered frame clock, as it is spelled there.
    private static string? FrameClockName(string text)
    {
        if (text.Length == 0 || text.Equals("auto", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        return FrameClocks.Names.FirstOrDefault(n => n.Equals(text, StringComparison.OrdinalIgnoreCase))
            ?? throw new FormatException($"'{text}' is not a frame clock: auto, {string.Join(", ", FrameClocks.Names)}.");
    }

    // "auto", "cpu" or kind:ordinal; whether the device exists is the start-up's to say (IdrakMauiStartup).
    private static string DeviceName(string text)
    {
        if (IsAuto(text))
        {
            return "auto";
        }

        int colon = text.IndexOf(':', StringComparison.Ordinal);
        string kind = colon < 0 ? text : text[..colon];
        bool ordinal = colon < 0 || int.TryParse(text.AsSpan(colon + 1), NumberStyles.None, CultureInfo.InvariantCulture, out _);
        return kind.Length > 0 && ordinal && kind.All(char.IsAsciiLetterOrDigit) ? text.ToLowerInvariant()
            : throw new FormatException($"'{text}' is not a device: auto, cpu, or a kind and its number such as vulkan:0.");
    }
}
