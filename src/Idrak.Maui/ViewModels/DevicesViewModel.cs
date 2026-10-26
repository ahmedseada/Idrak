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
            foreach (var card in Describe(report))
            {
                Cards.Add(card);
            }

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
                rows.Add(new(name, value));
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

    private static void Add(List<InfoRow> rows, string name, string? value)
    {
        if (!string.IsNullOrEmpty(value))
        {
            rows.Add(new(name, value));
        }
    }
}
