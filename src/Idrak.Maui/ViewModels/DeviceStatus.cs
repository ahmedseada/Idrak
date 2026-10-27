// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.ViewModels;

/// <summary>What Home's tiles and cards say about the device, from one reading (<see cref="DeviceStatus.Summarize"/>).</summary>
/// <param name="CpuTitle">"8 cores".</param>
/// <param name="CpuDetail">"3 core groups", or the CPU's first instruction sets, or its threads.</param>
/// <param name="GpuTitle">The GPU's backend ("Vulkan"), or "CPU only".</param>
/// <param name="GpuDetail">"kernels 64 wide", "ready", "failed to start", or why none was found.</param>
/// <param name="GpuReady">Whether a GPU started.</param>
/// <param name="GpuFailed">Whether a GPU was found and could not start.</param>
/// <param name="MemoryFree">The memory free, as text ("4.2 GB"): the system's figure where the CPU lists it.</param>
/// <param name="MemoryTotal">The memory the process may use, as text ("11 GB"): the system's figure (held to a memory limit) where listed.</param>
/// <param name="MemoryUsed">The share in use, 0 to 1.</param>
/// <param name="DefaultDevice">The device models load on ("vulkan:0").</param>
public sealed record DeviceSummary(string CpuTitle, string CpuDetail, string GpuTitle, string GpuDetail, bool GpuReady, bool GpuFailed, string MemoryFree,
    string MemoryTotal, double MemoryUsed, string DefaultDevice)
{
    /// <summary>The status chip's text: "GPU ready", "GPU failed" or "CPU only".</summary>
    public string Chip => GpuReady ? "GPU ready" : GpuFailed ? "GPU failed" : "CPU only";

    /// <summary>A short line: "CPU · GPU ready", "CPU only".</summary>
    public string Short => GpuReady ? "CPU · GPU ready" : GpuFailed ? "CPU · GPU failed" : "CPU only";
}

/// <summary>A model in the store as Home and All labs list it.</summary>
/// <param name="Initial">The letter on its tile ("Q").</param>
/// <param name="Title">Its name without the owner ("Qwen3-0.6B").</param>
/// <param name="Detail">Its owner and whether it is complete.</param>
/// <param name="Size">Its size ("734 MB").</param>
/// <param name="Tone">Its tile's tone.</param>
public sealed record ModelRow(string Initial, string Title, string Detail, string Size, string Tone);

/// <summary>
/// The device and the store as the app's overview shows them (Home's chip, tiles and cards; All labs' Devices and Models
/// cards): read once on a worker thread (the first reading starts the backends) and again when asked. One for the app.
/// </summary>
public sealed class DeviceStatus : ViewModel
{
    private static readonly string[] Tones = ["Indigo", "Violet", "Green", "Amber", "Sky", "Rose"];
    private readonly IDeviceInformation _devices;
    private readonly IModelStore? _store;
    private Task? _reading;
    private DeviceSummary? _summary;
    private string _modelsText = "";
    private string _error = "";

    /// <summary>The status over <paramref name="devices"/> and <paramref name="store"/> (none: no models listed).</summary>
    public DeviceStatus(IDeviceInformation devices, IModelStore? store = null)
    {
        _devices = devices ?? throw new ArgumentNullException(nameof(devices));
        _store = store;
    }

    /// <summary>The last reading, or null before the first.</summary>
    public DeviceSummary? Summary
    {
        get => _summary;
        private set
        {
            if (SetProperty(ref _summary, value))
            {
                OnPropertyChanged(nameof(IsRead));
                OnPropertyChanged(nameof(Chip));
                OnPropertyChanged(nameof(DevicesText));
            }
        }
    }

    /// <summary>Whether the device was read.</summary>
    public bool IsRead => Summary is not null;

    /// <summary>The status chip's text ("GPU ready"; "Measuring…" before the first reading).</summary>
    public string Chip => Summary?.Chip ?? "Measuring…";

    /// <summary>"CPU · GPU ready", or "Measuring…".</summary>
    public string DevicesText => Summary?.Short ?? "Measuring…";

    /// <summary>The store's models, the largest first (at most six).</summary>
    public ObservableCollection<ModelRow> Models { get; } = [];

    /// <summary>"3 pulled · 2.1 GB", "None pulled yet".</summary>
    public string ModelsText
    {
        get => _modelsText;
        private set => SetProperty(ref _modelsText, value);
    }

    /// <summary>The ids of every model in the store (for search).</summary>
    public IReadOnlyList<string> ModelIds { get; private set; } = [];

    /// <summary>Why the last reading failed, or "".</summary>
    public string Error
    {
        get => _error;
        private set => SetProperty(ref _error, value);
    }

    /// <summary>Reads the device and the store once (later calls wait for the same reading), on a worker thread.</summary>
    public Task EnsureReadAsync() => _reading ??= RefreshAsync();

    /// <summary>Reads the device and the store again, on a worker thread.</summary>
    public async Task RefreshAsync()
    {
        try
        {
            var (summary, models) = await Task.Run(() => (Summarize(_devices.Read()), _store?.List() ?? [])).ConfigureAwait(true);
            Summary = summary;
            ShowModels(models);
            Error = "";
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = e.Message.Split('\n')[0];
        }
    }

    /// <summary>Lists the store's models again (on a worker thread).</summary>
    public async Task RefreshModelsAsync()
    {
        if (_store is null)
        {
            return;
        }

        try
        {
            ShowModels(await Task.Run(_store.List).ConfigureAwait(true));
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = e.Message;
        }
    }

    /// <summary>What a reading says, for the tiles: the CPU, the first GPU that started (or why none), the memory.</summary>
    public static DeviceSummary Summarize(DeviceReport report)
    {
        ArgumentNullException.ThrowIfNull(report);
        var p = report.Process;
        var cpu = report.Devices.FirstOrDefault(d => d.Info.Kind == "cpu");
        string? Fact(DeviceDetails? d, string part) => d?.Info.Facts?.FirstOrDefault(f => f.Key.Contains(part, StringComparison.OrdinalIgnoreCase)).Value;
        string cpuDetail = DeviceFacts.CoreGroupCount(DeviceFacts.Fact(cpu?.Info.Facts, "core groups")) is var groups and > 0
            ? string.Create(CultureInfo.InvariantCulture, $"{groups} core group{(groups == 1 ? "" : "s")}")
            : Fact(cpu, "instruction set") is { Length: > 0 } sets ? string.Join(" · ", sets.Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries).Take(2))
            : string.Create(CultureInfo.InvariantCulture, $"{p.CpuThreads} threads");

        var gpus = report.Devices.Where(d => d.Info.Kind != "cpu" && d.Info.Listed && d.Info.HardwareKind != "virtual").ToList();
        var ready = gpus.FirstOrDefault(d => d.Info.Error is null);
        var failed = gpus.FirstOrDefault(d => d.Info.Error is not null);
        string gpuTitle = ready?.Info.Backend ?? failed?.Info.Backend ?? "CPU only";
        string gpuDetail = ready is { } r ? r.Info.KernelWidth is { } width ? string.Create(CultureInfo.InvariantCulture, $"kernels {width} wide") : "ready"
            : failed is not null ? "failed to start"
            : report.Backends.FirstOrDefault(b => b.Count == 0)?.UnavailableReason is { Length: > 0 } why ? Short(why) : "no GPU found";

        // The memory from the system's figure (what is free on the phone, not the runtime's heap budget), where the CPU lists it;
        // else what the process may use less its working set.
        MemoryFigure memory;
        if (DeviceFacts.SystemMemory(cpu?.Info.Facts) is { } system)
        {
            memory = system;
        }
        else
        {
            long total = cpu?.Info.MemoryBytes is > 0 and var bytes ? bytes : p.AvailableMemory;
            long used = Math.Clamp(p.WorkingSet, 0, Math.Max(total, 0));
            memory = new MemoryFigure(Math.Max(0, total - used), total);
        }

        return new DeviceSummary(string.Create(CultureInfo.InvariantCulture, $"{p.Processors} cores"), cpuDetail, gpuTitle, gpuDetail, ready is not null,
            ready is null && failed is not null, ProgressText.Bytes(memory.Free), ProgressText.Bytes(memory.Total), memory.Used, report.DefaultDevice);
    }

    /// <summary>The rows of <paramref name="models"/> (the largest first, at most <paramref name="most"/>) and the line counting them.</summary>
    public static (IReadOnlyList<ModelRow> Rows, string Text) Describe(IReadOnlyList<StoredModel> models, int most = 6)
    {
        ArgumentNullException.ThrowIfNull(models);
        var rows = models.OrderByDescending(m => m.Bytes).Take(most).Select((m, i) =>
        {
            string name = m.Id.Contains('/', StringComparison.Ordinal) ? m.Id[(m.Id.IndexOf('/', StringComparison.Ordinal) + 1)..] : m.Id;
            string owner = m.Id.Contains('/', StringComparison.Ordinal) ? m.Id[..m.Id.IndexOf('/', StringComparison.Ordinal)] : "";
            return new ModelRow(name.Length > 0 ? char.ToUpperInvariant(name[0]).ToString() : "?", name,
                (owner.Length > 0 ? owner + " · " : "") + (m.Complete ? "ready" : "unfinished"), ProgressText.Bytes(m.Bytes), Tones[i % Tones.Length]);
        }).ToList();
        long bytes = models.Sum(m => m.Bytes);
        string text = models.Count == 0 ? "None pulled yet" : $"{models.Count} pulled · {ProgressText.Bytes(bytes)}";
        return (rows, text);
    }

    private void ShowModels(IReadOnlyList<StoredModel> models)
    {
        var (rows, text) = Describe(models);
        Lists.Replace(Models, rows, static (a, b) => a == b);                  // Home's rows made again only when the store changed
        ModelsText = text;
        ModelIds = [.. models.Select(m => m.Id).Distinct(StringComparer.OrdinalIgnoreCase)];
    }

    private static string Short(string text)
    {
        string first = text.Split('\n')[0].Trim();
        return first.Length <= 40 ? first : first[..39] + "…";
    }
}
