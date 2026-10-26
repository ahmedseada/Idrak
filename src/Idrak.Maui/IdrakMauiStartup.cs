// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Maui;

/// <summary>What applying the options did (<see cref="IdrakMauiStartup.Apply"/>).</summary>
/// <param name="CacheFolder">The library's cache folder now.</param>
/// <param name="Device">The device asked for ("auto", "cpu", "vulkan:0").</param>
/// <param name="Problems">What could not be applied, each with why (a device that is not there: the library's default is kept).</param>
/// <param name="AppStorage">The app's own storage the default cache folder is made in.</param>
public sealed record IdrakStartupReport(string CacheFolder, string Device, IReadOnlyList<string> Problems, string AppStorage);

/// <summary>
/// Maps <see cref="IdrakMauiOptions"/> onto the library: <see cref="ComputeResources"/> (the cache folder first, then every
/// limit), <see cref="Device.Default"/>, the Hugging Face token (HF_TOKEN in the process, which every download reads). A
/// null limit keeps the library's own value: at start-up what its variable or its measurement gave, and later the value
/// the library had before the app first applied its options.
/// </summary>
public static class IdrakMauiStartup
{
    private const string TokenVariable = "HF_TOKEN";

    // The library's values before the app applied any (its variables or its defaults), read once at the first Apply.
    private static readonly Lazy<Snapshot> Library = new(Snapshot.Read);
    private static bool s_tokenSet;

    /// <summary>
    /// Applies <paramref name="options"/>. At start-up (<paramref name="previous"/> null) everything is applied, the cache
    /// folder first (an empty <see cref="IdrakMauiOptions.CacheFolder"/> is <c>idrak</c> in <paramref name="appStorage"/>,
    /// unless <c>IDRAK_CACHE</c> is set); later (the settings page saving, <paramref name="previous"/> the options before),
    /// what takes effect at once is applied and the settings that need a restart are left as they are
    /// (<see cref="IdrakMauiSettings.NeedRestart"/> names them).
    /// </summary>
    /// <param name="options">The options.</param>
    /// <param name="appStorage">The app's own storage (MAUI's <c>FileSystem.AppDataDirectory</c>).</param>
    /// <param name="previous">The options applied before, or null at start-up.</param>
    public static IdrakStartupReport Apply(IdrakMauiOptions options, string appStorage, IdrakMauiOptions? previous = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        ArgumentException.ThrowIfNullOrWhiteSpace(appStorage);
        var library = Library.Value;
        var problems = new List<string>();
        if (previous is null)
        {
            ComputeResources.CacheFolder = CacheFolder(options, appStorage);
            ComputeResources.GpuChunkBytes = options.GpuChunkMiB is { } chunk ? chunk << 20 : library.GpuChunkBytes;
            ComputeResources.GpuStackBytes = options.GpuStackBytes ?? library.GpuStackBytes;
        }

        ComputeResources.MaxCpuThreads = options.MaxCpuThreads ?? library.MaxCpuThreads;
        ComputeResources.CpuMemoryLimit = Bytes(options.CpuMemoryLimitMiB) ?? library.CpuMemoryLimit;
        ComputeResources.GpuMemoryLimit = Bytes(options.GpuMemoryLimitMiB) ?? library.GpuMemoryLimit;
        ComputeResources.GpuMemoryReserve = Bytes(options.GpuMemoryReserveMiB) ?? library.GpuMemoryReserve;
        ComputeResources.OffloadToHostMemory = options.OffloadToHostMemory ?? library.OffloadToHostMemory;
        ComputeResources.OffloadReturnHeadroom = Bytes(options.OffloadReturnHeadroomMiB) ?? library.OffloadReturnHeadroom;
        ComputeResources.WorkspaceShare = options.WorkspaceShare ?? library.WorkspaceShare;
        ComputeResources.TuningBudget = options.TuningBudgetMs is { } budget ? TimeSpan.FromMilliseconds(budget) : library.TuningBudget;
        ComputeResources.TuningNeighbourOctaves = options.TuningNeighbourOctaves ?? library.TuningNeighbourOctaves;
        ComputeResources.LookupTables = options.LookupTables is { } placement ? ComputeResources.ParseLookupTables(placement) : library.LookupTables;
        ComputeResources.CpuVectors = options.CpuVectors ?? library.CpuVectors;
        ApplyToken(options.HuggingFaceToken);
        ApplyDevice(options.Device, problems);
        return new IdrakStartupReport(ComputeResources.CacheFolder, options.Device, problems, appStorage);
    }

    /// <summary>The cache folder <paramref name="options"/> give: theirs, else null when <c>IDRAK_CACHE</c> is set (it decides), else <c>idrak</c> in <paramref name="appStorage"/>.</summary>
    public static string? CacheFolder(IdrakMauiOptions options, string appStorage)
    {
        ArgumentNullException.ThrowIfNull(options);
        return options.CacheFolder is { Length: > 0 } folder ? folder
            : Environment.GetEnvironmentVariable("IDRAK_CACHE") is { Length: > 0 } ? null
            : Path.Combine(appStorage, "idrak");
    }

    private static long? Bytes(long? mebibytes) => mebibytes is { } m ? m << 20 : null;

    private static void ApplyToken(string? token)
    {
        if (!string.IsNullOrWhiteSpace(token))
        {
            Environment.SetEnvironmentVariable(TokenVariable, token.Trim());
            s_tokenSet = true;
        }
        else if (s_tokenSet)
        {
            Environment.SetEnvironmentVariable(TokenVariable, null);      // the token the app had set is gone; one the process had stays
            s_tokenSet = false;
        }
    }

    // "auto" keeps the library's choice (the best device found); a device that is not there keeps it too, and says why.
    private static void ApplyDevice(string device, List<string> problems)
    {
        if (IdrakMauiSettings.IsAuto(device))
        {
            return;
        }

        try
        {
            Device.Default = Device.Parse(device);
        }
        catch (Exception e) when (e is InvalidOperationException or FormatException or ArgumentException or OverflowException)
        {
            problems.Add($"Device '{device}' is not available ({e.Message}); the library's default device is used.");
        }
    }

    private sealed record Snapshot(int MaxCpuThreads, long? CpuMemoryLimit, long? GpuMemoryLimit, long? GpuMemoryReserve, bool OffloadToHostMemory, long? OffloadReturnHeadroom,
        long? GpuChunkBytes, long? GpuStackBytes, double WorkspaceShare, TimeSpan TuningBudget, double TuningNeighbourOctaves, LookupTablePlacement LookupTables,
        CpuVectorPaths CpuVectors)
    {
        public static Snapshot Read() => new(ComputeResources.MaxCpuThreads, ComputeResources.CpuMemoryLimit, ComputeResources.GpuMemoryLimit, ComputeResources.GpuMemoryReserve,
            ComputeResources.OffloadToHostMemory, ComputeResources.OffloadReturnHeadroom, ComputeResources.GpuChunkBytes, ComputeResources.GpuStackBytes,
            ComputeResources.WorkspaceShare, ComputeResources.TuningBudget, ComputeResources.TuningNeighbourOctaves, ComputeResources.LookupTables,
            ComputeResources.CpuVectors);
    }
}
