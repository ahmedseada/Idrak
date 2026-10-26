// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Reflection;
using Idrak.Abstraction.Devices;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Theme;
using Idrak.Maui.ViewModels;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Maui.Dispatching;

using Device = Idrak.Abstraction.Device;

namespace Idrak.Maui.Tests;

internal static partial class MauiTests
{
    public sealed class CheckException(string message) : Exception(message);

    public static readonly (string Name, Action Run)[] All =
    [
        ("maui options: applying them sets every ComputeResources limit, the device, the token and the cache folder (the app's storage by default)", OptionsApplied),
        ("maui options: applied again with empty limits, the library's own values come back; the cache folder waits for the next start; a missing device is named, not thrown", OptionsReapplied),
        ("maui settings: saved and read back, the token in the secret store only; an empty value is the library's own; a bad saved value is skipped and named; clear forgets them", SettingsSavedAndRead),
        ("maui settings: values are checked (a comma reads as the decimal point), and a change to the cache folder or the GPU chunk needs a restart", SettingsChecked),
        ("maui settings page: a refused value saves nothing and says why; a saved restart-only change says so and stays as it runs", SettingsPage),
        ("maui features: the library's devices feature is the default; an app adds, replaces and removes by name; turned-off ones are not enabled; an unknown name is not registered", FeaturesRegistry),
        ("maui features: the devices feature opens its page bound to its view model", FeaturesOpen),
        ("maui features: every planned feature is registered by its name with its title, order, needs and state (planned, with what is coming); the menu lists them after devices in the plan's order; each can be turned off", PlannedRegistered),
        ("maui features: the placeholder page and its view model open for every planned feature, with its title, glyph, summary, what it will need and what it will do", PlaceholderOpens),
        ("maui features: a ready feature registered over a planned one keeps its place in the menu; one that sets no order or state is ready and sorts after the ordered ones, by title", PlannedReplaced),
        ("maui theme: the planned mark has its style and colours in light and dark, readable on its own ground, the menu's and a card's; the menu and placeholder pages take sizes and colours from theme keys", PlannedMark),
        ("maui progress: 'maui' is in ProgressReporters and IDRAK_PROGRESS chooses it", ProgressRegistered),
        ("maui progress: thousands of reports between two showings make one UI update; the item shows the amounts, share, rate and time left; the detail is asked once a showing", ProgressThrottled),
        ("maui progress: stages, finishing with a message and clearing the item; nothing is shown after the end", ProgressStages),
        ("maui models: a memory-pressure signal from the unload level up unloads the idle models and keeps one in use; below it only cached memory is returned", ModelsReleased),
        ("maui models: a host is made once per name and refuses another model type under the same name", ModelHosts),
        ("maui devices: the CPU is read with its facts (instruction sets, caches), its memory and the process, and shown as cards", DevicesRead),
        ("maui model store: lists pulled models with their sizes and removes a revision or a whole model; refuses what is not a model id", ModelStoreListed),
        ("maui theme: the choice and the phone's theme give the theme in effect; light and dark carry the same colour keys; the styles load", Themes),
        ("maui model store: a pull through a local Hugging Face (no network) takes the library's files or one .gguf at a revision, sends the token from the secret store, resumes a partial file by a range", ModelStorePulled),
        ("maui model store: a pull cancelled between two files or inside one is listed as unfinished, never complete; pulling again resumes it to the same bytes", ModelStoreCancelled),
        ("maui model store: a pull that does not fit the free space (less the headroom kept) is refused before its first byte; the free space is measured", ModelStoreSpace),
        ("maui models page: a pull's progress reaches the view model (bytes, share, rate, time left), then the list with sizes; suggestions fill the id; removing asks first", ModelsPageProgress),
        ("maui models page: failures name their cause (no network, a token needed, refused, gated, not found, not enough space, a full storage, cancelled, not an id)", ModelsPageErrors),
        ("maui chat: a model of the store is chosen and loaded (device, formats, load time); an answer streams into its bubble with its count and tokens per second, cold then warm; the request carries the system prompt, the history, thinking and the settings' options; a new chat clears it; copying goes through the page's clipboard", ChatStreams),
        ("maui chat: a thousand tokens make a few UI updates (one a progress interval, the first and the last), the bubble every token", ChatBatched),
        ("maui chat: Stop cancels mid-answer, keeping what came as the answer", ChatStopped),
        ("maui chat: the settings map to GenerationOptions over the library's defaults, are checked, and give the load request (device, weights, cache format, context)", ChatSettings),
        ("maui chat: a model is measured before loading (weights by format, the cache by the context) and refused with the limit named when it does not fit; an unregistered family fails as not registered", ChatRefused),
        ("maui chat: a tiny generated Qwen3 decoder loads from a folder through the model service and streams its answers on the CPU, cold then warm, and cold again after memory pressure took it", ChatEndToEnd),
        ("maui chat: the chat feature is the library's (order 20, needs a model) and opens its page bound to its view model", ChatPageOpens),
        ("public API: api/Idrak.Maui.txt matches the public surface of Idrak.Maui (IDRAK_UPDATE_API=1 rewrites it)", ApiSurfaceCurrent),
    ];

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new CheckException(message);
        }
    }

    private static string Temp(string name)
    {
        string folder = Path.Combine(Path.GetTempPath(), $"idrak-maui-tests-{Environment.ProcessId}", name);
        if (Directory.Exists(folder))
        {
            Directory.Delete(folder, recursive: true);
        }

        Directory.CreateDirectory(folder);
        return folder;
    }

    // Puts the library's values back after a test that applied options (the defaults the first Apply read).
    private static void Restore(string storage)
    {
        IdrakMauiStartup.Apply(new IdrakMauiOptions(), storage);         // at start-up's: the GPU chunk and stack too
        ComputeResources.CacheFolder = null;
        Device.Default = Device.Cpu;
    }

    private static IdrakMauiOptions Limits() => new()
    {
        Device = "cpu",
        MaxCpuThreads = 1,
        CpuMemoryLimitMiB = 512,
        GpuMemoryLimitMiB = 1024,
        GpuMemoryReserveMiB = 64,
        OffloadToHostMemory = !ComputeResources.OffloadToHostMemory,
        OffloadReturnHeadroomMiB = 32,
        GpuChunkMiB = 8,
        GpuStackBytes = 4096,
        WorkspaceShare = 0.25,
        TuningBudgetMs = 200,
        TuningNeighbourOctaves = 2,
        LookupTables = "host",
        HuggingFaceToken = "hf_test_token",
    };

    private static void OptionsApplied()
    {
        string storage = Temp("apply");
        bool offload = ComputeResources.OffloadToHostMemory;
        try
        {
            var options = Limits();
            var report = IdrakMauiStartup.Apply(options, storage);
            string expected = Environment.GetEnvironmentVariable("IDRAK_CACHE") is { Length: > 0 } variable ? Path.GetFullPath(variable) : Path.Combine(storage, "idrak");
            Check(ComputeResources.CacheFolder == expected && report.CacheFolder == expected && report.AppStorage == storage, $"cache folder {ComputeResources.CacheFolder}, expected {expected}");
            Check(ComputeResources.MaxCpuThreads == 1, $"threads {ComputeResources.MaxCpuThreads}");
            Check(ComputeResources.CpuMemoryLimit == 512L << 20 && ComputeResources.GpuMemoryLimit == 1024L << 20 && ComputeResources.GpuMemoryReserve == 64L << 20
                  && ComputeResources.OffloadReturnHeadroom == 32L << 20, "memory limits in MiB to bytes");
            Check(ComputeResources.OffloadToHostMemory == !offload, "offload");
            Check(ComputeResources.GpuChunkBytes == 8L << 20 && ComputeResources.GpuStackBytes == 4096, "GPU chunk and stack");
            Check(ComputeResources.WorkspaceShare == 0.25 && ComputeResources.TuningBudget == TimeSpan.FromMilliseconds(200) && ComputeResources.TuningNeighbourOctaves == 2,
                "workspace and tuning");
            Check(ComputeResources.LookupTables == LookupTablePlacement.HostMemory, $"lookup tables {ComputeResources.LookupTables}");
            Check(Environment.GetEnvironmentVariable("HF_TOKEN") == "hf_test_token", "HF_TOKEN set in the process");
            Check(Device.Default == Device.Cpu && report.Problems.Count == 0, $"device {Device.Default}; problems: {string.Join("; ", report.Problems)}");

            var own = new IdrakMauiOptions { CacheFolder = Path.Combine(storage, "elsewhere") };
            IdrakMauiStartup.Apply(own, storage);
            Check(ComputeResources.CacheFolder == Path.Combine(storage, "elsewhere"), $"the app's own folder: {ComputeResources.CacheFolder}");
        }
        finally
        {
            Restore(storage);
        }
    }

    private static void OptionsReapplied()
    {
        string storage = Temp("reapply");
        int threads = ComputeResources.MaxCpuThreads;
        long? cpuLimit = ComputeResources.CpuMemoryLimit;
        double share = ComputeResources.WorkspaceShare;
        var lookup = ComputeResources.LookupTables;
        try
        {
            var first = Limits();
            IdrakMauiStartup.Apply(first, storage);
            string folder = ComputeResources.CacheFolder;
            var empty = new IdrakMauiOptions { CacheFolder = Path.Combine(storage, "next-start") };
            var report = IdrakMauiStartup.Apply(empty, storage, previous: first);
            Check(ComputeResources.MaxCpuThreads == threads && ComputeResources.CpuMemoryLimit == cpuLimit && ComputeResources.WorkspaceShare == share
                  && ComputeResources.LookupTables == lookup, $"the library's values back: threads {ComputeResources.MaxCpuThreads} (was {threads})");
            Check(ComputeResources.CacheFolder == folder && report.CacheFolder == folder, "the cache folder changed before a restart");
            Check(Environment.GetEnvironmentVariable("HF_TOKEN") is null, "the app's token is gone from the process");
            Check(IdrakMauiSettings.NeedRestart(first, empty).Contains("Cache and models folder"), "the restart is named");

            var missing = IdrakMauiStartup.Apply(new IdrakMauiOptions { Device = "nosuchkind:3" }, storage, previous: empty);
            Check(missing.Problems.Count == 1 && missing.Problems[0].Contains("nosuchkind:3", StringComparison.Ordinal) && Device.Default == Device.Cpu,
                $"problems: {string.Join("; ", missing.Problems)}");
        }
        finally
        {
            Restore(storage);
        }
    }

    private static void SettingsSavedAndRead()
    {
        var store = new MemorySettingsStore();
        var secrets = new MemorySecretStore();
        var settings = new IdrakMauiSettings(store, secrets);
        Check(!settings.HasSaved, "saved before saving");
        var options = Limits();
        options.Theme = IdrakThemeChoice.Dark;
        options.ProgressInterval = TimeSpan.FromMilliseconds(400);
        options.ModelKeepAlive = null;
        options.MaxLoadedModels = 2;
        options.UnloadModelsAt = MemoryPressureLevel.Moderate;
        options.DisabledFeatures = new HashSet<string>(["lab", "chat"], StringComparer.OrdinalIgnoreCase);
        IdrakMauiOptions? raised = null;
        settings.Saved += (_, o) => raised = o;
        settings.SaveAsync(options).GetAwaiter().GetResult();
        Check(settings.HasSaved && ReferenceEquals(raised, options), "Saved raised");
        Check(store.Keys.All(k => k.StartsWith(IdrakMauiSettings.Prefix, StringComparison.Ordinal)), $"keys: {string.Join(", ", store.Keys)}");
        Check(!store.Keys.Any(k => store.Get(k) == "hf_test_token") && secrets.GetAsync(IdrakMauiSettings.Prefix + "hf_token").GetAwaiter().GetResult() == "hf_test_token",
            "the token is kept in the secret store only");

        var read = new IdrakMauiOptions();
        var skipped = settings.LoadIntoAsync(read).GetAwaiter().GetResult();
        Check(skipped.Count == 0, string.Join("; ", skipped));
        foreach (var setting in IdrakMauiSettings.Definitions)
        {
            Check(setting.Get(read) == setting.Get(options), $"{setting.Key}: read {setting.Get(read)}, saved {setting.Get(options)}");
        }

        Check(read.DisabledFeatures.SetEquals(["chat", "lab"]) && read.ModelKeepAlive is null && read.ProgressInterval == TimeSpan.FromMilliseconds(400), "features, keep-alive, interval");

        // An empty saved value is the library's own: over the app's default 4 threads, the user's "" gives null.
        store.Set(IdrakMauiSettings.Prefix + "cpu_threads", "");
        store.Set(IdrakMauiSettings.Prefix + "workspace_share", "many");
        var app = new IdrakMauiOptions { MaxCpuThreads = 4 };
        skipped = settings.LoadIntoAsync(app).GetAwaiter().GetResult();
        Check(app.MaxCpuThreads is null, $"threads {app.MaxCpuThreads}");
        Check(skipped.Count == 1 && skipped[0].StartsWith("Workspace share", StringComparison.Ordinal), $"skipped: {string.Join("; ", skipped)}");

        settings.ClearAsync().GetAwaiter().GetResult();
        Check(!settings.HasSaved && store.Keys.Count == 0 && secrets.GetAsync(IdrakMauiSettings.Prefix + "hf_token").GetAwaiter().GetResult() is null, "cleared");
        var defaults = new IdrakMauiOptions { MaxCpuThreads = 3 };
        settings.LoadIntoAsync(defaults).GetAwaiter().GetResult();
        Check(defaults.MaxCpuThreads == 3, "nothing saved keeps the app's defaults");
    }

    private static void SettingsChecked()
    {
        IdrakSetting Find(string key) => IdrakMauiSettings.Definitions.Single(s => s.Key == key);
        var options = new IdrakMauiOptions();
        foreach (var (key, text) in new[] { ("cpu_threads", "0"), ("cpu_threads", "two"), ("workspace_share", "1.5"), ("cache_folder", "relative/folder"), ("device", "vulkan:x"),
                     ("theme", "blue"), ("offload", "maybe"), ("lookup_tables", "disk"), ("tuning_budget_ms", "0"), ("unload_models_at", "never"), ("storage_headroom_mb", "-1") })
        {
            bool refused = false;
            try
            {
                Find(key).Set(options, text);
            }
            catch (FormatException e)
            {
                refused = e.Message.Contains(text, StringComparison.Ordinal);
            }

            Check(refused, $"{key}={text} was taken, or refused without naming the value");
        }

        Find("workspace_share").Set(options, "0,5");
        Find("device").Set(options, "Vulkan:1");
        Find("theme").Set(options, "LIGHT");
        Find("offload").Set(options, "on");
        Check(options.WorkspaceShare == 0.5 && options.Device == "vulkan:1" && options.Theme == IdrakThemeChoice.Light && options.OffloadToHostMemory == true, "accepted values");
        Find("offload").Set(options, "default");
        Find("device").Set(options, "");
        Check(options.OffloadToHostMemory is null && options.Device == "auto", "defaults");

        var before = new IdrakMauiOptions();
        var after = before.Clone();
        after.GpuChunkMiB = 4;
        after.MaxCpuThreads = 1;
        Check(IdrakMauiSettings.NeedRestart(before, after).SequenceEqual(["GPU chunk (MiB)"]), string.Join(", ", IdrakMauiSettings.NeedRestart(before, after)));
        var chosen = new IdrakMauiOptions { Device = "cpu" };
        Check(IdrakMauiSettings.NeedRestart(chosen, new IdrakMauiOptions()).Count == 1, "back to auto needs a restart");
        Check(IdrakMauiSettings.Definitions.Select(s => s.Key).Distinct().Count() == IdrakMauiSettings.Definitions.Count, "a key twice");
    }

    private static void SettingsPage()
    {
        string storage = Temp("settings-page");
        try
        {
            var options = new IdrakMauiOptions();
            var startup = IdrakMauiStartup.Apply(options, storage);
            var settings = new IdrakMauiSettings(new MemorySettingsStore(), new MemorySecretStore());
            var progress = new ProgressCenter(TimeSpan.FromMilliseconds(250));
            var page = new SettingsViewModel(options, settings, startup, progress);
            Check(page.Entries.Count == IdrakMauiSettings.Definitions.Count && page.Features.Any(f => f.Name == MobileFeatures.Devices), "entries and features");
            Check(page.Entries.Single(e => e.Setting.Key == "device").Choices.Take(2).SequenceEqual(["auto", "cpu"]), "device choices");

            var threads = page.Entries.Single(e => e.Setting.Key == "cpu_threads");
            threads.Text = "lots";
            Check(!page.SaveAsync().GetAwaiter().GetResult() && threads.HasError && !settings.HasSaved, "a refused value saved");

            threads.Text = "";
            page.Entries.Single(e => e.Setting.Key == "cache_folder").Text = Path.Combine(storage, "later");
            page.Entries.Single(e => e.Setting.Key == "progress_interval_ms").Text = "500";
            page.Features.Single(f => f.Name == MobileFeatures.Devices).IsEnabled = false;
            Check(page.SaveAsync().GetAwaiter().GetResult() && settings.HasSaved, $"not saved: {page.Message}");
            Check(page.Message.Contains("Restart the app for: Cache and models folder", StringComparison.Ordinal), page.Message);
            Check(options.CacheFolder is null && progress.Interval == TimeSpan.FromMilliseconds(500) && !options.IsFeatureEnabled(MobileFeatures.Devices),
                "the restart-only change stays as it runs; the rest applies at once");
            var saved = new IdrakMauiOptions();
            settings.LoadIntoAsync(saved).GetAwaiter().GetResult();
            Check(saved.CacheFolder == Path.Combine(storage, "later"), "the restart-only change is saved for the next start");
        }
        finally
        {
            Restore(storage);
        }
    }

    private sealed class LabFeature(string name, string title, int order = 5) : IMobileFeature
    {
        public string Name => name;

        public string Summary => "a test feature";

        public string Title => title;

        public string Glyph => "*";

        public int Order => order;

        public PhoneNeeds Needs => PhoneNeeds.Microphone | PhoneNeeds.Model;

        public IReadOnlyList<string> Packages => ["Idrak.Audio"];

        public IReadOnlyList<string> ModelFamilies => [];

        public object CreateViewModel(IServiceProvider services) => new object();

        public Page CreatePage(IServiceProvider services, object viewModel) => new ContentPage();
    }

    private static void FeaturesRegistry()
    {
        Check(MobileFeatures.Origin(MobileFeatures.Devices) == Overrides.Library && MobileFeatures.Get("DEVICES") is DevicesFeature, "the library's devices feature");
        var lab = new LabFeature("lab", "Lab");
        var mine = new LabFeature(MobileFeatures.Devices, "My devices", order: 0);
        try
        {
            MobileFeatures.Register(lab);
            MobileFeatures.Register(mine);
            Check(MobileFeatures.Find("lab") == lab && MobileFeatures.Get(MobileFeatures.Devices) == mine, "registered");
            Check(MobileFeatures.Origin(MobileFeatures.Devices) == typeof(MauiTests).Assembly.GetName().Name && MobileFeatures.Default(MobileFeatures.Devices) is DevicesFeature,
                $"origin {MobileFeatures.Origin(MobileFeatures.Devices)}");
            Check(MobileFeatures.All.Select(f => f.Name).Take(3).SequenceEqual([MobileFeatures.Devices, "lab", MobileFeatures.Models]), $"order: {string.Join(", ", MobileFeatures.All.Select(f => f.Name))}");
            var options = new IdrakMauiOptions();
            options.DisabledFeatures.Add("LAB");
            Check(MobileFeatures.Enabled(options).Select(f => f.Name).Take(2).SequenceEqual([MobileFeatures.Devices, MobileFeatures.Models])
                  && !MobileFeatures.Enabled(options).Any(f => f.Name == "lab"), "a turned-off feature is enabled");
            Check(MobileFeatures.Unregister(MobileFeatures.Devices) && MobileFeatures.Get(MobileFeatures.Devices) is DevicesFeature, "the library's comes back");
            Check(!MobileFeatures.Unregister(MobileFeatures.Devices), "the library's default was removed");
        }
        finally
        {
            MobileFeatures.Unregister("lab");
            MobileFeatures.Unregister(MobileFeatures.Devices);
        }

        try
        {
            MobileFeatures.Get("nope");
            Check(false, "an unknown feature was found");
        }
        catch (NotRegisteredException e)
        {
            Check(e.Message.Contains("devices", StringComparison.Ordinal), e.Message);
        }
    }

    private static void FeaturesOpen()
    {
        using var services = new ServiceCollection().AddSingleton<IDeviceInformation, DeviceInformation>().BuildServiceProvider();
        var page = MobileFeatures.Open(MobileFeatures.Devices, services);
        Check(page is DevicesPage { BindingContext: DevicesViewModel } && page.Title == "Devices", $"page {page.GetType().Name}, title {page.Title}");
        var models = MobileFeatures.Open(MobileFeatures.Models, services);
        Check(models is ModelsPage { BindingContext: ModelsViewModel { Confirm: not null } } && models.Title == "Models", $"page {models.GetType().Name}, title {models.Title}");
    }

    // A dispatcher standing in for the UI thread: every dispatch is queued until the test runs them.
    private sealed class QueuedDispatcher : IDispatcher
    {
        public Queue<Action> Queue { get; } = new();

        public int Dispatched { get; private set; }

        public bool IsDispatchRequired => true;

        public bool Dispatch(Action action)
        {
            Dispatched++;
            Queue.Enqueue(action);
            return true;
        }

        public bool DispatchDelayed(TimeSpan delay, Action action) => Dispatch(action);

        public IDispatcherTimer CreateTimer() => throw new NotSupportedException();

        public void Run()
        {
            while (Queue.TryDequeue(out var action))
            {
                action();
            }
        }
    }

    // A clock the test moves by hand.
    private sealed class ManualTime : TimeProvider
    {
        private long _ticks = 1_000_000;

        public override long TimestampFrequency => TimeSpan.TicksPerSecond;

        public override long GetTimestamp() => _ticks;

        public override DateTimeOffset GetUtcNow() => new DateTimeOffset(2026, 10, 26, 0, 0, 0, TimeSpan.Zero).AddTicks(_ticks);

        public void Advance(TimeSpan time) => _ticks += time.Ticks;
    }

    private static void ProgressRegistered()
    {
        var center = new ProgressCenter(TimeSpan.FromMilliseconds(250));
        ProgressReporters.Register(new MauiProgressReporterKind(center));
        string? before = Environment.GetEnvironmentVariable(ProgressReporters.EnvironmentVariable);
        try
        {
            Check(ProgressReporters.Find("MAUI") is MauiProgressReporterKind kind && kind.Center == center, "registered");
            Environment.SetEnvironmentVariable(ProgressReporters.EnvironmentVariable, MauiProgressReporterKind.ReporterName);
            Check(ProgressReporters.Choose() == MauiProgressReporterKind.ReporterName, $"chose {ProgressReporters.Choose()}");
            using var reporter = ProgressReporters.Start("load", null, ProgressUnit.Items);
            reporter.Advance(3);
            Check(center.Items.Count == 1 && center.Items[0].Done == 3, "the reporter shows on the center (no app: the reporting thread itself)");
        }
        finally
        {
            Environment.SetEnvironmentVariable(ProgressReporters.EnvironmentVariable, before);
            ProgressReporters.Unregister(MauiProgressReporterKind.ReporterName);
        }
    }

    private static void ProgressThrottled()
    {
        var dispatcher = new QueuedDispatcher();
        var time = new ManualTime();
        var center = new ProgressCenter(TimeSpan.FromMilliseconds(250), dispatcher, time);
        int details = 0;
        var reporter = center.Start("pull model", 100_000, ProgressUnit.Bytes);
        reporter.Detail = () => $"detail {++details}";
        for (int i = 0; i < 10_000; i++)
        {
            reporter.Advance(1);
        }

        Check(dispatcher.Dispatched == 2 && details == 1, $"{dispatcher.Dispatched} dispatches (the item and one update), {details} details for 10,000 reports at one time");
        dispatcher.Run();                                                         // the UI thread takes the item and its first state
        Check(center.Items.Single().Done == 1 && center.Items[0].Updates == 1, "the first state is the first report's");
        time.Advance(TimeSpan.FromMilliseconds(100));
        reporter.Advance(1);
        Check(dispatcher.Dispatched == 2, "a showing within the interval");
        time.Advance(TimeSpan.FromMilliseconds(200));
        for (int i = 0; i < 1_000; i++)
        {
            reporter.Advance(1);
        }

        Check(dispatcher.Dispatched == 3 && details == 2, $"{dispatcher.Dispatched} dispatches, {details} details after the interval");
        dispatcher.Run();
        var item = center.Items.Single();
        Check(item.Done == 10_002 && item.Total == 100_000 && item.Unit == ProgressUnit.Bytes && !item.IsIndeterminate, $"done {item.Done}");
        Check(Math.Abs(item.Fraction - 0.10002) < 1e-9 && item.Elapsed == TimeSpan.FromMilliseconds(300), $"fraction {item.Fraction}, elapsed {item.Elapsed}");
        double rate = 10_002 / 0.3;
        Check(Math.Abs(item.Rate - rate) < 1e-6 && item.RateText == ProgressText.Rate(rate, ProgressUnit.Bytes), $"rate {item.Rate} ({item.RateText})");
        Check(item.TimeLeft is { } left && Math.Abs(left.TotalSeconds - (100_000 - 10_002) / rate) < 1e-3 && item.TimeLeftText.EndsWith(" left", StringComparison.Ordinal),
            $"left {item.TimeLeft} ({item.TimeLeftText})");
        Check(item.Detail == "detail 2" && item.Text == ProgressText.Format("pull model", 10_002, 100_000, TimeSpan.FromMilliseconds(300), ProgressUnit.Bytes) + "  detail 2",
            $"text {item.Text}");
        Check(item.AmountText == $"{ProgressText.Bytes(10_002)} / {ProgressText.Bytes(100_000)}" && item.Updates == 2, $"amount {item.AmountText}, {item.Updates} updates");

        // Updates that wait on the UI thread are not queued again: the later state replaces the earlier.
        time.Advance(TimeSpan.FromSeconds(1));
        reporter.Report(20_000);
        time.Advance(TimeSpan.FromSeconds(1));
        reporter.Report(30_000);
        Check(dispatcher.Dispatched == 4 && dispatcher.Queue.Count == 1, $"{dispatcher.Queue.Count} updates waiting");
        dispatcher.Run();
        Check(item.Done == 30_000 && item.Updates == 3, $"done {item.Done} after {item.Updates} updates");
    }

    private static void ProgressStages()
    {
        var dispatcher = new QueuedDispatcher();
        var time = new ManualTime();
        var center = new ProgressCenter(TimeSpan.FromMilliseconds(250), dispatcher, time) { KeepFinished = 1 };
        var reporter = center.Start("read", 10, ProgressUnit.Items);
        reporter.Report(10);
        time.Advance(TimeSpan.FromSeconds(1));
        reporter.NextStage("write", null, ProgressUnit.Bytes, "read 10 lines");
        reporter.Advance(500);
        dispatcher.Run();
        var item = center.Items.Single();
        Check(item.Label == "write" && item.Done == 0 && item.IsIndeterminate && item.PreviousStage == "read 10 lines" && item.Unit == ProgressUnit.Bytes,
            $"stage {item.Label}, done {item.Done}, previous '{item.PreviousStage}'");
        time.Advance(TimeSpan.FromSeconds(1));
        reporter.Advance(500);
        reporter.Finish("written");
        reporter.Advance(1);
        dispatcher.Run();
        Check(item.IsFinished && item.Message == "written" && item.Done == 1000, $"finished {item.IsFinished}, '{item.Message}', done {item.Done}");

        var second = center.Start("next");
        var third = center.Start("last");
        dispatcher.Run();
        Check(center.Items.Count == 3, $"{center.Items.Count} items: the finished one is kept while it is the only one");
        second.Finish();
        third.Clear();
        dispatcher.Run();
        Check(center.Items.Count == 2 && center.Items.All(i => i.IsFinished), $"{center.Items.Count} items after finishing one and clearing one");
        center.Start("more");
        dispatcher.Run();
        Check(center.Items.Count == 2 && center.Items[0].Label == "next", $"KeepFinished 1 keeps the latest finished: {string.Join(", ", center.Items.Select(i => i.Label))}");
    }

    private sealed class FakeModel(string name) : IDisposable
    {
        public string Name => name;

        public bool Disposed { get; private set; }

        public void Dispose() => Disposed = true;
    }

    private static void ModelsReleased()
    {
        var pressure = new ManualMemoryPressure();
        var options = new IdrakMauiOptions { ModelKeepAlive = null, UnloadModelsAt = MemoryPressureLevel.Low };
        var made = new List<FakeModel>();
        using var models = new ModelService(pressure, options);
        var host = models.Host("chat", name =>
        {
            var model = new FakeModel(name);
            made.Add(model);
            return model;
        });
        host.Acquire("idle").Dispose();
        using var busy = host.Acquire("busy");
        var releases = new List<MemoryRelease>();
        models.Released += (_, r) => releases.Add(r);
        Check(models.Loaded.Count == 2 && models.Loaded.All(m => m.Host == "chat"), "two loaded");

        pressure.Raise(MemoryPressureLevel.Moderate, "TRIM_MEMORY_RUNNING_MODERATE");
        Check(models.Loaded.Count == 2 && releases.Count == 1 && releases[0].Unloaded.Count == 0 && releases[0].CachedMemoryReleased, "below the level: cached memory only");

        pressure.Raise(MemoryPressureLevel.Low, "TRIM_MEMORY_RUNNING_LOW");
        Check(releases.Count == 2 && releases[1].Unloaded.SequenceEqual(["chat/idle"]) && releases[1].Signal.Source == "TRIM_MEMORY_RUNNING_LOW",
            $"unloaded: {string.Join(", ", releases[1].Unloaded)}");
        Check(made.Single(m => m.Name == "idle").Disposed && !made.Single(m => m.Name == "busy").Disposed, "the idle one disposed, the one in use kept");
        Check(models.Loaded.Single().Model.Name == "busy", "one left");

        busy.Dispose();
        var critical = models.Release(new MemoryPressureSignal(MemoryPressureLevel.Critical, "onLowMemory"));
        Check(critical.Unloaded.SequenceEqual(["chat/busy"]) && models.Loaded.Count == 0, "released once no longer in use");

        models.Dispose();
        pressure.Raise(MemoryPressureLevel.Critical);
        Check(releases.Count == 3, "a disposed service still listens");
    }

    private static void ModelHosts()
    {
        using var models = new ModelService(new ManualMemoryPressure(), new IdrakMauiOptions { MaxLoadedModels = 1 });
        var first = models.Host("speech", n => new FakeModel(n));
        Check(ReferenceEquals(first, models.Host("speech", n => new FakeModel(n + "!"))) && first.MaxLoaded == 1, "one host a name, with the options' limit");
        try
        {
            models.Host("speech", _ => new MemoryStream());
            Check(false, "another model type was taken under the same name");
        }
        catch (InvalidOperationException e)
        {
            Check(e.Message.Contains("FakeModel", StringComparison.Ordinal), e.Message);
        }
    }

    private static void DevicesRead()
    {
        var report = new DeviceInformation().Read(["cpu"]);
        var cpu = report.Devices.Single();
        Check(cpu.Info.Device == "cpu" && cpu.Info.Error is null && cpu.Memory is not null, $"cpu: {cpu.Info.Error}");
        var facts = cpu.Info.Facts?.ToDictionary(f => f.Key, f => f.Value) ?? [];
        Check(facts.ContainsKey("instruction sets") && facts.ContainsKey("caches") && facts.ContainsKey("threads"), $"facts: {string.Join(", ", facts.Keys)}");
        Check(report.Process.Processors == Environment.ProcessorCount && report.Process.AvailableMemory > 0 && report.Process.CacheFolder == ComputeResources.CacheFolder,
            "the process");
        var cards = DevicesViewModel.Describe(report);
        Check(cards.Count >= 2 && cards[0].Title == "This phone" && cards[1].Title == "cpu" && cards[1].Rows.Any(r => r.Name == "instruction sets" && r.Value.Length > 0)
              && cards[1].Rows.Any(r => r.Name == "memory in use"), $"cards: {string.Join(", ", cards.Select(c => c.Title))}");
    }

    private static void ModelStoreListed()
    {
        string cache = Temp("store");
        try
        {
            ComputeResources.CacheFolder = cache;
            var store = new ModelStore(new IdrakMauiOptions());
            Check(store.List().Count == 0, "an empty store");
            string root = Path.Combine(cache, "downloads", "huggingface", "models");
            Check(store.Folder == root, store.Folder);
            Write(Path.Combine(root, "owner", "name", "abc123", "config.json"), 100);
            Write(Path.Combine(root, "owner", "name", "abc123", "model.safetensors"), 4096);
            Write(Path.Combine(root, "owner", "name", "def456", "model.safetensors.part"), 50);
            Write(Path.Combine(root, "other", "gguf", "0123", "sub", "model.gguf"), 2048);
            var listed = store.List();
            Check(listed.Select(m => $"{m.Id}@{m.Revision}").Order(StringComparer.Ordinal).SequenceEqual(["other/gguf@0123", "owner/name@abc123", "owner/name@def456"]),
                string.Join(", ", listed.Select(m => $"{m.Id}@{m.Revision}")));
            var whole = listed.Single(m => m.Revision == "abc123");
            Check(whole.Bytes == 4196 && whole.Files == 2 && whole.Complete && !listed.Single(m => m.Revision == "def456").Complete
                  && listed.Single(m => m.Id == "other/gguf").Bytes == 2048, "sizes and completeness");
            Check(store.Remove("owner/name@def456") && store.List().Count == 2 && !store.Remove("owner/name@def456"), "a revision removed");
            Check(store.Remove("owner/name") && store.List().Single().Id == "other/gguf" && !Directory.Exists(Path.Combine(root, "owner")), "a model removed with its empty owner");
            foreach (string bad in (string[])["../escape", "owner/name@..", "owner", "/abs/path", "owner/name@a/b"])
            {
                bool refused = false;
                try
                {
                    store.Remove(bad);
                }
                catch (ArgumentException)
                {
                    refused = true;
                }

                Check(refused, $"'{bad}' was taken as a model id");
            }

            bool pullRefused = false;
            try
            {
                store.PullAsync("not a model").GetAwaiter().GetResult();
            }
            catch (ArgumentException)
            {
                pullRefused = true;
            }

            Check(pullRefused, "a pull of what is not an id");
        }
        finally
        {
            ComputeResources.CacheFolder = null;
        }

        static void Write(string path, int bytes)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            File.WriteAllBytes(path, new byte[bytes]);
        }
    }

    private static void Themes()
    {
        Check(IdrakThemes.Effective(IdrakThemeChoice.System, AppTheme.Dark) == AppTheme.Dark && IdrakThemes.Effective(IdrakThemeChoice.System, AppTheme.Unspecified) == AppTheme.Light
              && IdrakThemes.Effective(IdrakThemeChoice.Light, AppTheme.Dark) == AppTheme.Light && IdrakThemes.Effective(IdrakThemeChoice.Dark, AppTheme.Light) == AppTheme.Dark,
            "the theme in effect");
        var light = IdrakThemes.Colors(AppTheme.Light);
        var dark = IdrakThemes.Colors(AppTheme.Dark);
        Check(light is IdrakLightColors && dark is IdrakDarkColors, "the dictionaries");
        var lightKeys = light.Keys.ToHashSet();
        Check(lightKeys.SetEquals(dark.Keys) && lightKeys.Count >= 15, $"light: {string.Join(", ", lightKeys.Order())}; dark: {string.Join(", ", dark.Keys.Order())}");
        // The icon's own colours (assets/icon.svg), in both.
        string icon = File.ReadAllText(Path.Combine(RepositoryRoot(), "assets", "icon.svg"));
        foreach (var (key, hex) in new[] { ("IdrakGround", "#0b1020"), ("IdrakSky", "#38bdf8"), ("IdrakAmber", "#f59e0b") })
        {
            Check(icon.Contains(hex, StringComparison.OrdinalIgnoreCase), $"{hex} is not in the icon any more");
            Check(light[key] is Color l && dark[key] is Color d && l.ToArgbHex().EndsWith(hex[1..], StringComparison.OrdinalIgnoreCase) && l.Equals(d), $"{key}");
        }

        var styles = new IdrakStyles();
        Check(styles.ContainsKey("IdrakCard") && styles.ContainsKey("IdrakFontSizeBody") && styles.ContainsKey("IdrakCornerRadius"), "the styles and metrics");
        foreach (string key in (string[])["IdrakChip", "IdrakDangerButton", "IdrakSectionPadding", "IdrakChipPadding", "IdrakChipCornerRadius"])
        {
            Check(styles.ContainsKey(key), $"the Models page's {key}");
        }
    }

    private static void ApiSurfaceCurrent()
    {
        string path = Path.Combine(RepositoryRoot(), "api", "Idrak.Maui.txt");
        string actual = ApiSurface.Describe(typeof(IdrakMauiOptions).Assembly);
        if (Environment.GetEnvironmentVariable("IDRAK_UPDATE_API") == "1")
        {
            File.WriteAllText(path, actual);
            return;
        }

        string expected = File.Exists(path) ? File.ReadAllText(path).ReplaceLineEndings("\n") : "";
        Check(expected == actual, "the public API of Idrak.Maui changed: " + ApiSurface.FirstDifference(expected, actual)
            + "\nIf the change is intended: IDRAK_UPDATE_API=1 IDRAK_DEVICES=cpu IDRAK_FILTER=\"public API\" dotnet run -c Release --project tests/Idrak.Maui.Tests, then commit api/.");
    }

    // The folder holding Idrak.slnx, found from the build output folder.
    private static string RepositoryRoot()
    {
        for (var dir = new DirectoryInfo(AppContext.BaseDirectory); dir is not null; dir = dir.Parent)
        {
            if (File.Exists(Path.Combine(dir.FullName, "Idrak.slnx")))
            {
                return dir.FullName;
            }
        }

        throw new DirectoryNotFoundException($"Idrak.slnx not found above {AppContext.BaseDirectory}.");
    }
}
