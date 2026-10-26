// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Windows.Input;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.ViewModels;

/// <summary>
/// The devices page: the process and the machine, then every device (the CPU with its features, each GPU of every backend
/// with its driver, kernel width, subgroup sizes, memory budget and memory in use), and each backend that found none with
/// why. Read on a worker thread, as the first reading starts the backends.
/// </summary>
public sealed class DevicesViewModel : ViewModel
{
    private readonly IDeviceInformation _devices;
    private bool _busy;
    private string _summary = "";
    private string _error = "";

    /// <summary>A view model over <paramref name="devices"/>.</summary>
    public DevicesViewModel(IDeviceInformation devices)
    {
        _devices = devices ?? throw new ArgumentNullException(nameof(devices));
        RefreshCommand = new Command(async () => await RefreshAsync(), () => !IsBusy);
    }

    /// <summary>The page's title.</summary>
    public string Title => "Devices";

    /// <summary>The cards: the process first, then each device and each backend that found none.</summary>
    public ObservableCollection<InfoCard> Cards { get; } = [];

    /// <summary>The process and the devices that compute, as summary cards (the process first).</summary>
    public ObservableCollection<DeviceCard> Devices { get; } = [];

    /// <summary>The virtual devices (reached by name only, or reporting themselves virtual: deferred, minimal, simulated), apart.</summary>
    public ObservableCollection<DeviceCard> VirtualDevices { get; } = [];

    /// <summary>Whether there is a virtual device to list.</summary>
    public bool HasVirtualDevices => VirtualDevices.Count > 0;

    /// <summary>
    /// A reading as summary cards: the process ("This phone"), then each device with its state (ready, failed, the default)
    /// and its few key facts, every other fact folded under it; virtual devices marked apart. <paramref name="cards"/> are
    /// the reading's full cards (<see cref="Describe"/>), made again when not given.
    /// </summary>
    public static IReadOnlyList<DeviceCard> Summarize(DeviceReport report, IReadOnlyList<InfoCard>? cards = null)
    {
        ArgumentNullException.ThrowIfNull(report);
        cards ??= Describe(report);
        var list = new List<DeviceCard>();
        if (cards.Count > 0)
        {
            var process = cards[0];
            list.Add(new DeviceCard(process.Title, process.Subtitle, "", "Process", DeviceState.Ready, isDefault: false, isVirtual: false, "spark",
                Pick(process.Rows, "runtime", "architecture", "memory the process may use", "working set"), process.Rows, ""));
        }

        foreach (var d in report.Devices)
        {
            var info = d.Info;
            var card = cards.FirstOrDefault(c => c.Title == info.Device);
            var rows = card?.Rows ?? [];
            var state = info.Error is not null ? DeviceState.Failed : DeviceState.Ready;
            bool isVirtual = !info.Listed || string.Equals(info.HardwareKind, "virtual", StringComparison.OrdinalIgnoreCase);
            var facts = info.Kind == "cpu" ? CpuFacts(info, rows) : Pick(rows, "memory", "kernel width", "lanes", "compute units", "instruction sets", "driver");
            list.Add(new DeviceCard(info.Backend, info.Name.Length > 0 ? info.Name : info.Device, info.Device, info.HardwareKind ?? info.Kind, state, info.IsDefault, isVirtual,
                info.Kind == "cpu" ? "cpu" : "zap", facts, rows, card?.Problem ?? "", DeviceFacts.RefusedKernels(DeviceFacts.Fact(info.Facts, "refused kernels"))));
        }

        foreach (var b in report.Backends.Where(b => b.Count == 0))
        {
            list.Add(new DeviceCard(b.Display, "none found", b.Kind, "", DeviceState.None, isDefault: false, isVirtual: false, "zap", [], [],
                b.UnavailableReason ?? "no device found"));
        }

        return list;
    }

    // The CPU's few key facts: its memory from the system's figure ("4.2 GB free of 11 GB"), its core groups in a line, the
    // vector paths that run, its instruction sets; each in the reading's own words where the facts do not read so.
    private static IReadOnlyList<InfoRow> CpuFacts(DeviceInfo info, IReadOnlyList<InfoRow> rows)
    {
        var facts = new List<InfoRow>(4);
        if (DeviceFacts.SystemMemory(info.Facts) is { } memory)
        {
            facts.Add(new("memory", memory.Text));
        }
        else if (rows.FirstOrDefault(r => r.Name == "memory") is { } row)
        {
            facts.Add(row);
        }

        if (DeviceFacts.CoreGroups(DeviceFacts.Fact(info.Facts, "core groups")) is { Length: > 0 } groups)
        {
            facts.Add(new("core groups", groups));
        }
        else if (rows.FirstOrDefault(r => r.Name == "compute units") is { } units)
        {
            facts.Add(units);
        }

        if (DeviceFacts.VectorPaths(DeviceFacts.Fact(info.Facts, "vector paths")) is { Length: > 0 } paths)
        {
            facts.Add(new("vector paths", paths));
        }

        if (rows.FirstOrDefault(r => r.Name == "instruction sets") is { } sets)
        {
            facts.Add(sets);
        }

        return [.. facts.Take(4)];
    }

    // The rows named, in that order, at most four.
    private static IReadOnlyList<InfoRow> Pick(IReadOnlyList<InfoRow> rows, params string[] names) =>
        [.. names.Select(n => rows.FirstOrDefault(r => r.Name == n)).OfType<InfoRow>().Take(4)];

    /// <summary>One line about the reading (how many devices, the default, how long it took).</summary>
    public string Summary
    {
        get => _summary;
        private set => SetProperty(ref _summary, value);
    }

    /// <summary>Why the reading failed, or "".</summary>
    public string Error
    {
        get => _error;
        private set
        {
            if (SetProperty(ref _error, value))
            {
                OnPropertyChanged(nameof(HasError));
            }
        }
    }

    /// <summary>Whether <see cref="Error"/> has something to show.</summary>
    public bool HasError => Error.Length > 0;

    /// <summary>Whether a reading is running.</summary>
    public bool IsBusy
    {
        get => _busy;
        set
        {
            if (SetProperty(ref _busy, value))
            {
                ((Command)RefreshCommand).ChangeCanExecute();
            }
        }
    }

    /// <summary>Reads the devices again.</summary>
    public ICommand RefreshCommand { get; }

    /// <summary>Reads the devices (on a worker thread) and shows them.</summary>
    public async Task RefreshAsync()
    {
        IsBusy = true;
        Error = "";
        try
        {
            var report = await Task.Run(() => _devices.Read()).ConfigureAwait(true);
            Cards.Clear();
            var cards = Describe(report);
            foreach (var card in cards)
            {
                Cards.Add(card);
            }

            Devices.Clear();
            VirtualDevices.Clear();
            foreach (var device in Summarize(report, cards))
            {
                (device.IsVirtual ? VirtualDevices : Devices).Add(device);
            }

            OnPropertyChanged(nameof(HasVirtualDevices));

            Summary = string.Create(CultureInfo.InvariantCulture,
                $"{report.Devices.Count} device{(report.Devices.Count == 1 ? "" : "s")}, default {report.DefaultDevice}, read in {report.Elapsed.TotalMilliseconds:0} ms");
        }
        catch (Exception e)
        {
            Error = e.Message;
        }
        finally
        {
            IsBusy = false;
        }
    }

    /// <summary>A reading as cards: the process, each device, each backend that found none.</summary>
    public static IReadOnlyList<InfoCard> Describe(DeviceReport report)
    {
        ArgumentNullException.ThrowIfNull(report);
        var p = report.Process;
        var cards = new List<InfoCard>
        {
            new("This phone", p.OperatingSystem,
            [
                new("architecture", p.Architecture),
                new("runtime", p.Runtime),
                new("processors", p.Processors.ToString(CultureInfo.InvariantCulture)),
                new("library threads", p.CpuThreads.ToString(CultureInfo.InvariantCulture)),
                new("memory the process may use", ProgressText.Bytes(p.AvailableMemory)),
                new("working set", ProgressText.Bytes(p.WorkingSet)),
                new("managed heap", ProgressText.Bytes(p.ManagedHeap)),
                new("cache folder", p.CacheFolder),
            ]),
        };

        foreach (var d in report.Devices)
        {
            var info = d.Info;
            var rows = new List<InfoRow> { new("backend", info.Backend) };
            Add(rows, "kind", info.HardwareKind);
            Add(rows, "driver", info.Driver);
            Add(rows, "memory", info.MemoryBytes is { } bytes ? ProgressText.Bytes(bytes) : null);
            Add(rows, "compute units", info.ComputeUnits?.ToString(CultureInfo.InvariantCulture));
            Add(rows, "lanes", info.SubgroupSize?.ToString(CultureInfo.InvariantCulture));
            Add(rows, "kernel width", info.KernelWidth?.ToString(CultureInfo.InvariantCulture));
            if (info.Error is null)
            {
                rows.Add(new("matrix units", info.MatrixUnits ? "yes" : "no"));
            }

            if (d.Memory is { } m)
            {
                rows.Add(new("memory in use", $"{ProgressText.Bytes(m.InUse)} in use, {ProgressText.Bytes(m.Cached)} cached, peak {ProgressText.Bytes(m.Peak)}"
                                               + (m.Limit is { } limit ? $", limit {ProgressText.Bytes(limit)}" : "")));
            }

            foreach (var (name, value) in info.Facts ?? [])
            {
                rows.Add(new(name == RuntimeMemoryFact ? RuntimeMemoryRow : name, value));
            }

            if (d.MeasuredChoices > 0)
            {
                rows.Add(new("measured choices kept", d.MeasuredChoices.ToString(CultureInfo.InvariantCulture)));
            }

            Add(rows, "kernels", info.KernelsUnavailable is { } off ? $"unavailable: {off}" : null);
            Add(rows, "note", info.IsDefault ? "the default device" : info.Note);
            Add(rows, "listed", info.Listed ? null : "by name only");
            cards.Add(new(info.Device, info.Name, rows, info.Error is { } error ? $"cannot start: {error}" : ""));
        }

        foreach (var b in report.Backends.Where(b => b.Count == 0))
        {
            cards.Add(new(b.Display, "none found", [], b.UnavailableReason ?? "no device found"));
        }

        return cards;
    }

    // The CPU's runtime figure (under Mono its heap budget, not the phone's memory), named as the runtime's in the details.
    private const string RuntimeMemoryFact = "runtime memory", RuntimeMemoryRow = "the runtime's memory figure";

    private static void Add(List<InfoRow> rows, string name, string? value)
    {
        if (!string.IsNullOrEmpty(value))
        {
            rows.Add(new(name, value));
        }
    }
}

/// <summary>Whether a device started.</summary>
public enum DeviceState
{
    /// <summary>It started.</summary>
    Ready,

    /// <summary>It was found and could not start.</summary>
    Failed,

    /// <summary>A backend found none.</summary>
    None,
}

/// <summary>A device as a summary card: its kind and name, its state as a badge, a few key facts, the rest folded.</summary>
public sealed class DeviceCard : ViewModel
{
    private bool _expanded;

    private bool _refusedExpanded;

    internal DeviceCard(string title, string name, string device, string kind, DeviceState state, bool isDefault, bool isVirtual, string icon,
        IReadOnlyList<InfoRow> facts, IReadOnlyList<InfoRow> details, string problem, IReadOnlyList<string>? refusedKernels = null)
    {
        RefusedKernels = refusedKernels ?? [];
        ToggleRefusedCommand = new Command(() => IsRefusedExpanded = !IsRefusedExpanded);
        Title = title;
        Name = name;
        Device = device;
        Kind = kind;
        State = state;
        IsDefault = isDefault;
        IsVirtual = isVirtual;
        Icon = icon;
        Facts = facts;
        Details = details;
        Problem = problem;
        ToggleCommand = new Command(() => IsExpanded = !IsExpanded);
    }

    /// <summary>The backend ("Vulkan", "CPU") or "This phone" for the process.</summary>
    public string Title { get; }

    /// <summary>The device's own name.</summary>
    public string Name { get; }

    /// <summary>Its name in options ("vulkan:0"), or "".</summary>
    public string Device { get; }

    /// <summary>What it reports itself to be ("discrete", "integrated", "cpu"), or "".</summary>
    public string Kind { get; }

    /// <summary>Whether it started.</summary>
    public DeviceState State { get; }

    /// <summary>Whether it started.</summary>
    public bool IsReady => State == DeviceState.Ready;

    /// <summary>Whether it could not start.</summary>
    public bool IsFailed => State == DeviceState.Failed;

    /// <summary>Whether a backend found none.</summary>
    public bool IsNone => State == DeviceState.None;

    /// <summary>The badge: "Default", "Ready", "Failed" or "None".</summary>
    public string Badge => IsFailed ? "Failed" : IsNone ? "None" : IsDefault ? "Default" : "Ready";

    /// <summary>Whether it is the device models load on.</summary>
    public bool IsDefault { get; }

    /// <summary>Whether it is virtual (deferred, minimal, simulated): listed apart.</summary>
    public bool IsVirtual { get; }

    /// <summary>Its icon's name.</summary>
    public string Icon { get; }

    /// <summary>Its few key facts.</summary>
    public IReadOnlyList<InfoRow> Facts { get; }

    /// <summary>Every fact it reported.</summary>
    public IReadOnlyList<InfoRow> Details { get; }

    /// <summary>Whether it has facts to unfold.</summary>
    public bool HasDetails => Details.Count > 0;

    /// <summary>What went wrong, or "".</summary>
    public string Problem { get; }

    /// <summary>Whether there is a problem to show.</summary>
    public bool HasProblem => Problem.Length > 0;

    /// <summary>Whether every fact shows.</summary>
    public bool IsExpanded
    {
        get => _expanded;
        set
        {
            if (SetProperty(ref _expanded, value))
            {
                OnPropertyChanged(nameof(ToggleText));
            }
        }
    }

    /// <summary>"Details" or "Less".</summary>
    public string ToggleText => IsExpanded ? "Less" : $"All {Details.Count} details";

    /// <summary>Folds or unfolds the facts.</summary>
    public ICommand ToggleCommand { get; }

    /// <summary>The kernels the driver refused, each with what was tried and what ran instead (its "refused kernels" fact).</summary>
    public IReadOnlyList<string> RefusedKernels { get; }

    /// <summary>Whether a kernel was refused (the card shows the badge).</summary>
    public bool HasRefusedKernels => RefusedKernels.Count > 0;

    /// <summary>The badge: "1 kernel on fallback", "3 kernels on fallback".</summary>
    public string RefusedText => RefusedKernels.Count == 1 ? "1 kernel on fallback" : $"{RefusedKernels.Count} kernels on fallback";

    /// <summary>Whether the refused kernels are listed.</summary>
    public bool IsRefusedExpanded
    {
        get => _refusedExpanded;
        set => SetProperty(ref _refusedExpanded, value && HasRefusedKernels);
    }

    /// <summary>Lists or folds the refused kernels.</summary>
    public ICommand ToggleRefusedCommand { get; }
}
