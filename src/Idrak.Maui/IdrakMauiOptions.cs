// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>Which theme the app shows.</summary>
public enum IdrakThemeChoice
{
    /// <summary>The phone's setting, followed as it changes.</summary>
    System,

    /// <summary>Light.</summary>
    Light,

    /// <summary>Dark.</summary>
    Dark,
}

/// <summary>
/// What <c>UseIdrak</c> sets up, and the settings page edits: the device, the theme, the cache folder, every
/// <see cref="ComputeResources"/> limit, the Hugging Face token, the storage a pull leaves free, the progress interval, the
/// models kept loaded and the features shown (and, set by the app only, the models the Models page suggests). A null limit means the library's own value: what its environment variable sets, else what it measures.
/// The values an app gives in <c>UseIdrak</c> are its defaults; what the user saved in the settings page is applied over
/// them (<see cref="KeepSettings"/>), at start-up, before the first device or download.
/// </summary>
public sealed class IdrakMauiOptions
{
    /// <summary>The device models load on: "auto" (the best found, <see cref="Device.Default"/>), "cpu", or "vulkan:0" and the like.</summary>
    public string Device { get; set; } = "auto";

    /// <summary>The theme: the phone's (the default), light or dark.</summary>
    public IdrakThemeChoice Theme { get; set; } = IdrakThemeChoice.System;

    /// <summary>
    /// The library's cache folder (<see cref="ComputeResources.CacheFolder"/>): models, prepared files, measured choices.
    /// Null: <c>IDRAK_CACHE</c> when set, else <c>idrak</c> in the app's own storage (MAUI's <c>FileSystem.AppDataDirectory</c>).
    /// A change takes effect at the next start.
    /// </summary>
    public string? CacheFolder { get; set; }

    /// <summary>The CPU threads the library may use (<see cref="ComputeResources.MaxCpuThreads"/>); null: every core, or IDRAK_CPU_THREADS.</summary>
    public int? MaxCpuThreads { get; set; }

    /// <summary>The most tensor memory on the CPU, in MiB (<see cref="ComputeResources.CpuMemoryLimit"/>); null: no limit, or IDRAK_CPU_MEMORY_LIMIT_MB.</summary>
    public long? CpuMemoryLimitMiB { get; set; }

    /// <summary>The most tensor memory on each GPU, in MiB (<see cref="ComputeResources.GpuMemoryLimit"/>); null: what the device has.</summary>
    public long? GpuMemoryLimitMiB { get; set; }

    /// <summary>GPU memory left free for the system, in MiB (<see cref="ComputeResources.GpuMemoryReserve"/>); null: measured from the device.</summary>
    public long? GpuMemoryReserveMiB { get; set; }

    /// <summary>Whether a full GPU places tensors in system memory (<see cref="ComputeResources.OffloadToHostMemory"/>); null: the library's.</summary>
    public bool? OffloadToHostMemory { get; set; }

    /// <summary>GPU memory kept free when offloaded tensors come back, in MiB (<see cref="ComputeResources.OffloadReturnHeadroom"/>); null: measured.</summary>
    public long? OffloadReturnHeadroomMiB { get; set; }

    /// <summary>The chunk a GPU places long-lived blocks in, in MiB (<see cref="ComputeResources.GpuChunkBytes"/>); null: from the device's pages. Next start.</summary>
    public long? GpuChunkMiB { get; set; }

    /// <summary>The stack each GPU thread gets, in bytes (<see cref="ComputeResources.GpuStackBytes"/>); null: what kernels need. Next start.</summary>
    public long? GpuStackBytes { get; set; }

    /// <summary>The share of available memory one operation's buffers may take, in (0, 1] (<see cref="ComputeResources.WorkspaceShare"/>); null: the library's.</summary>
    public double? WorkspaceShare { get; set; }

    /// <summary>How long a shape's first use may measure kernel choices, in ms (<see cref="ComputeResources.TuningBudget"/>); null: the library's.</summary>
    public double? TuningBudgetMs { get; set; }

    /// <summary>How far a measured choice reaches, in octaves (<see cref="ComputeResources.TuningNeighbourOctaves"/>); null: the library's.</summary>
    public double? TuningNeighbourOctaves { get; set; }

    /// <summary>Where frozen lookup tables live: "auto", "host" or "device" (<see cref="ComputeResources.LookupTables"/>); null: the library's.</summary>
    public string? LookupTables { get; set; }

    /// <summary>
    /// Which vector paths the CPU's kernels take (<see cref="ComputeResources.CpuVectors"/>): automatic, the platform's
    /// intrinsics only (native), the portable vectors only, or the plain loops (scalar). Every path gives the same results;
    /// they differ in speed. Null: the library's (IDRAK_CPU_VECTORS, else automatic).
    /// </summary>
    public CpuVectorPaths? CpuVectors { get; set; }

    /// <summary>The Hugging Face token for gated and private models (kept in the phone's secure storage); null: none, or HF_TOKEN.</summary>
    public string? HuggingFaceToken { get; set; }

    /// <summary>
    /// The free space a model pull leaves on the storage, in MiB: a pull whose files would leave less is refused before its
    /// first byte (the model store measures the storage); null: none kept (the files must only fit).
    /// </summary>
    public long? StorageHeadroomMiB { get; set; }

    /// <summary>
    /// Models the Models page offers as chips to tap (Hugging Face ids, or an id and a .gguf file in it); empty by default,
    /// set by the app in <c>UseIdrak</c>. Not a setting of the settings page: the app's own list.
    /// </summary>
    public IReadOnlyList<string> SuggestedModels { get; set; } = [];

    /// <summary>The least time between two updates of a progress view (the "maui" reporter).</summary>
    public TimeSpan ProgressInterval { get; set; } = TimeSpan.FromMilliseconds(250);

    /// <summary>How long a model stays loaded after its last use; null: until memory pressure or the app ends.</summary>
    public TimeSpan? ModelKeepAlive { get; set; } = TimeSpan.FromMinutes(10);

    /// <summary>The most models kept loaded at once in each host (loading another unloads the longest unused); null: no limit.</summary>
    public int? MaxLoadedModels { get; set; }

    /// <summary>The least memory-pressure level that unloads the models not in use (cached device memory is returned at every level).</summary>
    public MemoryPressureLevel UnloadModelsAt { get; set; } = MemoryPressureLevel.Low;

    /// <summary>The system prompt every chat of the Chat lab starts with (a system message, rendered by the model's own template); null: none.</summary>
    public string? ChatSystemPrompt { get; set; }

    /// <summary>The chat's sampling temperature (<see cref="GenerationOptions.Temperature"/>; 0 picks the likeliest token); null: the library's.</summary>
    public double? ChatTemperature { get; set; }

    /// <summary>The chat's nucleus share, in (0, 1] (<see cref="GenerationOptions.TopP"/>); null: the library's.</summary>
    public double? ChatTopP { get; set; }

    /// <summary>The chat's most likely tokens sampled among (<see cref="GenerationOptions.TopK"/>, 0: all); null: the library's.</summary>
    public int? ChatTopK { get; set; }

    /// <summary>The most tokens one answer has (<see cref="GenerationOptions.NumPredict"/>); null: the library's (until the turn ends or the context is full).</summary>
    public int? ChatMaxNewTokens { get; set; }

    /// <summary>
    /// The chat's context window in tokens (<see cref="GenerationOptions.NumCtx"/>; it also sizes the key/value cache and the
    /// memory a model is checked against, and is capped by the model's own); null: the library's.
    /// </summary>
    public int? ChatContext { get; set; }

    /// <summary>The chat's random seed (<see cref="GenerationOptions.Seed"/>), for answers that repeat; null: random.</summary>
    public int? ChatSeed { get; set; }

    /// <summary>Whether the chat asks for reasoning where the model's template has a switch for it (<see cref="ChatRequest.Think"/>); null: the model's own default.</summary>
    public bool? ChatThinking { get; set; }

    /// <summary>
    /// The format a chat model's projections are kept in once loaded: "bfloat16", "int8", "int4" or another registered packed
    /// format (<see cref="PackedWeightFormats.Names"/>); null: the library's (float32). Taken at the next load.
    /// </summary>
    public string? ChatWeights { get; set; }

    /// <summary>The chat's key/value cache format (a registered one, <see cref="KeyValueLayouts.Names"/>: "int8" halves it); null: the library's (float32). Taken at the next load.</summary>
    public string? ChatKeyValueFormat { get; set; }

    /// <summary>
    /// The window width, in device-independent units, from which the app shows its wide layout: the side rail and the
    /// pages in several columns or panes (the measured width of the window, so it follows rotation and split screen); below
    /// it, the bottom bar and one column. 840 by default.
    /// </summary>
    public double WideLayoutWidth { get; set; } = 840;

    /// <summary>How long Home's carousel shows a slide before the next; null or zero: it moves only when swiped. Six seconds by default.</summary>
    public TimeSpan? CarouselInterval { get; set; } = TimeSpan.FromSeconds(6);

    /// <summary>
    /// Whether the tabs' views are made ahead, one at a time while the app is idle after Home first shows (with the labs of a
    /// tab of several), so the first tap on a tab only shows it. True by default; false makes each when first opened.
    /// </summary>
    public bool PrebuildSections { get; set; } = true;

    /// <summary>How long a tab's view fades in when it is shown; zero (the default) shows it at once. The tap's command never waits for it.</summary>
    public TimeSpan SectionFade { get; set; } = TimeSpan.Zero;

    /// <summary>
    /// The frame clock live views (the waveform, the level meters) draw on, a <see cref="Abstractions.FrameClocks"/> name:
    /// "display" (Android's Choreographer) or "timer"; null or "auto": the first that can tick here.
    /// </summary>
    public string? FrameClock { get; set; }

    /// <summary>The most frames a second live views draw; null: every frame the display shows.</summary>
    public double? LiveFrameRate { get; set; }

    /// <summary>The sound one bar of the live waveform (and one step of a level meter) stands for; 50 ms by default.</summary>
    public TimeSpan LiveLevelWindow { get; set; } = TimeSpan.FromMilliseconds(50);

    /// <summary>The most lines the Terminal keeps (the oldest dropped); null: the app's (<see cref="ViewModels.TerminalLabOptions.Lines"/>).</summary>
    public int? TerminalLines { get; set; }

    /// <summary>The most lines the Terminal's history keeps; null: the app's.</summary>
    public int? TerminalHistory { get; set; }

    /// <summary>The Terminal history's latest lines shown as chips; null: the app's.</summary>
    public int? TerminalHistoryShown { get; set; }

    /// <summary>The most suggestions the Terminal shows at once; null: the app's.</summary>
    public int? TerminalSuggestions { get; set; }

    /// <summary>The folder the Terminal's commands run in; null: the app's storage.</summary>
    public string? TerminalFolder { get; set; }

    /// <summary>What Home says: its slides and its tiles' samples (the app's own words; the library's by default).</summary>
    public HomeOptions Home { get; set; } = new();

    /// <summary>Questions the chat offers to ask while a chat is empty; the library's few by default, empty for none.</summary>
    public IReadOnlyList<string> ChatSuggestions { get; set; } =
        ["Explain what runs on this phone", "Summarise this in one paragraph", "Write a haiku about GPUs"];

    /// <summary>The features turned off in the settings (by name); every other registered feature shows.</summary>
    public ISet<string> DisabledFeatures { get; set; } = new HashSet<string>(StringComparer.OrdinalIgnoreCase);

    /// <summary>Whether what the user saved in the settings page is applied over these values at start-up (true by default).</summary>
    public bool KeepSettings { get; set; } = true;

    /// <summary>
    /// Whether the library's progress goes to the "maui" reporter when nothing else names one (<c>IDRAK_PROGRESS</c> is set
    /// to it in the process unless it is set already). True by default.
    /// </summary>
    public bool UseMauiProgress { get; set; } = true;

    /// <summary>Where settings are kept; null: MAUI's Preferences.</summary>
    public ISettingsStore? SettingsStore { get; set; }

    /// <summary>Where the token is kept; null: MAUI's SecureStorage.</summary>
    public ISecretStore? SecretStore { get; set; }

    /// <summary>Whether the feature <paramref name="name"/> shows.</summary>
    public bool IsFeatureEnabled(string name) => !DisabledFeatures.Contains(name);

    /// <summary>A copy (its own set of disabled features).</summary>
    public IdrakMauiOptions Clone()
    {
        var copy = (IdrakMauiOptions)MemberwiseClone();
        copy.DisabledFeatures = new HashSet<string>(DisabledFeatures, StringComparer.OrdinalIgnoreCase);
        copy.SuggestedModels = [.. SuggestedModels];
        copy.ChatSuggestions = [.. ChatSuggestions];
        return copy;
    }

    /// <summary>Takes every value of <paramref name="other"/> (the settings page saving into the app's options).</summary>
    public void CopyFrom(IdrakMauiOptions other)
    {
        ArgumentNullException.ThrowIfNull(other);
        foreach (var setting in IdrakMauiSettings.Definitions)
        {
            setting.Set(this, setting.Get(other));
        }

        DisabledFeatures = new HashSet<string>(other.DisabledFeatures, StringComparer.OrdinalIgnoreCase);
        SuggestedModels = [.. other.SuggestedModels];
        ChatSuggestions = [.. other.ChatSuggestions];
        Home = other.Home;
        KeepSettings = other.KeepSettings;
        UseMauiProgress = other.UseMauiProgress;
        SettingsStore = other.SettingsStore;
        SecretStore = other.SecretStore;
    }
}
