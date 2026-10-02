// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using static Idrak.Backends.Vulkan.VulkanDriver;

namespace Idrak.Backends.Vulkan;

// Runtime policy no reported limit settles is measured on the device in use when the backend starts, the way the CUDA
// backend measures its kernel choices (CudaBackend.Tuning.cs): candidates take turns over several rounds and the median
// of each counts; the default stays unless another is clearly faster. Measured: whether dispatches push their
// descriptors, how many commands a batch takes before it is submitted, and how many submitted batches the host runs
// ahead of. A few tens of milliseconds, once: the results are kept on disk per device and driver (deviceUUID,
// driverUUID, driver version), so later processes read them, and a new driver measures again. Each keeps an
// environment override (IDRAK_VULKAN_PUSH_DESCRIPTORS, IDRAK_VULKAN_BATCH_COMMANDS, IDRAK_VULKAN_IN_FLIGHT).
//
// The cache is a tab-separated text file, one line per device, driver and choice (key, value, device name for people
// reading it), in IDRAK_CACHE or ~/.cache/idrak (the library's download cache) under vulkan/; IDRAK_VULKAN_TUNING_CACHE
// names another file, or 0 keeps nothing.
internal sealed unsafe partial class VulkanBackend
{
    /// <summary>A measured choice replaces the default only when this much faster (3%, as on CUDA).</summary>
    private const double TuneMargin = 0.97;

    /// <summary>
    /// The share of host time a batch's submission may take: a batch holds enough commands that recording them takes
    /// 1 / SubmitShare times as long as submitting it (the device waits at most that long for a burst's first batch).
    /// </summary>
    private const double SubmitShare = 0.05;

    /// <summary>The file measured runtime choices are kept in (null: not kept); tests point it elsewhere.</summary>
    internal static string? TuningCacheFile { get; set; } = DefaultTuningCacheFile();

    /// <summary>How the runtime policy was chosen: "measured", "cached" or "override" (descriptor mode).</summary>
    public string PushDescriptorsChoice { get; private set; } = "measured";

    /// <summary>Commands per batch before it is submitted on its own.</summary>
    internal int MaxBatchCommands => _maxBatchCommands;

    /// <summary>Submitted batches the host runs ahead of before it waits for the oldest.</summary>
    internal int MaxInFlight => _maxInFlight;

    /// <summary>What the start-up measurement found (zeros when the choices came from the cache or overrides).</summary>
    internal RuntimeMeasurement Measured { get; private set; }

    /// <summary>
    /// Medians of the start-up measurement, in µs: a dispatch with pushed descriptors and with sets (recording, submission
    /// and the device's run together; 0 for a mode the device lacks), recording one dispatch in the chosen mode, submitting
    /// a batch, and a one-command batch's round trip (submission to fence).
    /// </summary>
    internal readonly record struct RuntimeMeasurement(double Pushed, double Sets, double Record, double Submit, double RoundTrip);

    private static string? DefaultTuningCacheFile()
    {
        string? setting = Environment.GetEnvironmentVariable("IDRAK_VULKAN_TUNING_CACHE");
        if (setting is "0" or "false")
        {
            return null;
        }

        if (!string.IsNullOrEmpty(setting))
        {
            return setting;
        }

        string root = Environment.GetEnvironmentVariable("IDRAK_CACHE")
                      ?? Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.UserProfile), ".cache", "idrak");
        return Path.Combine(root, "vulkan", "tuning.tsv");
    }

    // The cache key prefix of this device and driver: by what the device reports, never its ordinal (vulkan:N), so a
    // choice stays with its GPU whatever order the devices come in.
    private string TuningKey() => $"{DeviceKey(_physical.Facts, _physical.DeviceName)}/{_physical.Facts.DriverUuid:N}/{_physical.Facts.DriverVersion:X}/{PowerSource.Current}";

    /// <summary>
    /// What names a device in the tuning cache: its deviceUUID, or (a device reporting none: all zeros) its PCI address
    /// when reported and its name, so two such devices never share cached choices.
    /// </summary>
    internal static string DeviceKey(VulkanDeviceFacts facts, string name)
    {
        if (facts.DeviceUuid != Guid.Empty)
        {
            return facts.DeviceUuid.ToString("N");
        }

        string pci = facts.PciAddress is { } a ? $"{a.Domain:x}.{a.Bus:x}.{a.Device:x}.{a.Function:x}-" : "";
        var text = new System.Text.StringBuilder("nouuid-").Append(pci);
        foreach (char c in name)
        {
            text.Append(char.IsLetterOrDigit(c) || c is '-' or '.' ? c : '_');
        }

        return text.ToString();
    }

    // The cached choices of this device and driver (empty when there are none).
    private Dictionary<string, string> CachedChoices()
    {
        var choices = new Dictionary<string, string>();
        string? file = TuningCacheFile;
        if (file is null)
        {
            return choices;
        }

        try
        {
            if (File.Exists(file))
            {
                string prefix = TuningKey() + "/";
                foreach (string line in File.ReadLines(file))
                {
                    string[] fields = line.Split('\t');
                    if (fields.Length >= 2 && fields[0].StartsWith(prefix, StringComparison.Ordinal))
                    {
                        choices[fields[0][prefix.Length..]] = fields[1];
                    }
                }
            }
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
        }

        return choices;
    }

    // Keeps `choices` for this device and driver, replacing those of the same names (best effort: a read-only or busy cache only means measuring again).
    private void CacheChoices(IReadOnlyDictionary<string, string> choices)
    {
        string? file = TuningCacheFile;
        if (file is null)
        {
            return;
        }

        try
        {
            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(file))!);
            string prefix = TuningKey() + "/";
            // Other choices of this device and driver stay (the runtime policy and the storage memory are kept apart).
            var replaced = choices.Keys.Select(k => $"{prefix}{k}\t").ToArray();
            var lines = File.Exists(file) ? File.ReadAllLines(file).Where(l => !replaced.Any(r => l.StartsWith(r, StringComparison.Ordinal))).ToList() : [];
            lines.AddRange(choices.Select(c => $"{prefix}{c.Key}\t{c.Value}\t{_physical.DeviceName} ({_physical.Driver})"));
            string temporary = $"{file}.{Environment.ProcessId}.{Environment.CurrentManagedThreadId}.tmp";
            File.WriteAllLines(temporary, lines);
            File.Move(temporary, file, overwrite: true);
        }
        catch (Exception ex) when (ex is IOException or UnauthorizedAccessException)
        {
        }
    }

    // Decides the runtime policy: overrides first (`requestedPush` from a test or IDRAK_VULKAN_PUSH_DESCRIPTORS), then the
    // cached measurement of this device and driver, else a measurement now. Pushing descriptors writes the storages into
    // the command buffer instead of allocating and updating a set (less host work), but how a driver runs either is its
    // own matter (on a CPU driver, measured with lavapipe, sets ran about 15 µs faster per dispatch); so it is measured.
    private void TuneRuntime(bool? requestedPush)
    {
        bool canPush = _pushDescriptorSet != null;
        var cached = CachedChoices();
        bool complete = cached.ContainsKey("push-descriptors") && cached.ContainsKey("batch-commands/sets") && cached.ContainsKey("in-flight/sets")
                        && (!canPush || (cached.ContainsKey("batch-commands/pushed") && cached.ContainsKey("in-flight/pushed")));
        if (!complete)
        {
            var (measurement, choices) = MeasureRuntime(canPush);
            Measured = measurement;
            CacheChoices(choices);
            cached = choices;
            PushDescriptorsChoice = "measured";
        }
        else
        {
            PushDescriptorsChoice = "cached";
        }

        _usePush = canPush && (requestedPush ?? cached["push-descriptors"] == "1");
        if (canPush && requestedPush is not null)
        {
            PushDescriptorsChoice = "override";
        }

        string mode = _usePush ? "pushed" : "sets";
        _maxBatchCommands = PositiveSetting("IDRAK_VULKAN_BATCH_COMMANDS") ?? Parse(cached[$"batch-commands/{mode}"]);
        _maxInFlight = PositiveSetting("IDRAK_VULKAN_IN_FLIGHT") ?? Parse(cached[$"in-flight/{mode}"]);
        PrepareBatches();

        static int Parse(string text) => int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value > 0 ? value : 1;
    }

    // Measures the runtime policy: an empty kernel binding three storages, dispatched in runs of a few hundred and as
    // single-command batches, each descriptor mode in turn over several rounds (medians count). Returns the medians and
    // the choices for both modes:
    // - descriptors pushed unless sets are clearly (3%) faster per dispatch;
    // - batch commands: as many as take 1 / SubmitShare times a submission's cost to record (submission costs at most
    //   SubmitShare of the host's time);
    // - batches in flight: enough that the batches the host records while one batch makes its round trip through the
    //   device fit, plus the one being waited for (at least two: one running while the next is recorded).
    // Leaves no pipeline or storage behind, and resets the dispatch counters.
    private (RuntimeMeasurement, Dictionary<string, string>) MeasureRuntime(bool canPush)
    {
        const int Count = 200, Rounds = 7, Bindings = 3;
        int[] modes = canPush ? [0, 1] : [1];                              // 0: pushed, 1: sets
        var kernels = new VulkanKernel[2];
        for (int mode = 0; mode < 2; mode++)
        {
            var builder = new KernelBuilder("runtime-probe", 1);
            for (int i = 0; i < Bindings; i++)
            {
                builder.Buffer($"b{i}");
            }

            var built = builder.Build();
            kernels[mode] = new VulkanKernel(built.Words, built.Bindings, built.PushBytes, built.Name, built.Writes);
        }

        var storages = new Storage[Bindings];
        for (int i = 0; i < Bindings; i++)
        {
            storages[i] = Allocate(1, zeroed: false);
        }

        var dispatch = new[] { new List<double>(), new List<double>() };
        var record = new[] { new List<double>(), new List<double>() };
        var submit = new[] { new List<double>(), new List<double>() };
        var roundTrip = new[] { new List<double>(), new List<double>() };
        (_maxBatchCommands, _maxInFlight) = (Count + 1, 1);                // the measurement submits by itself
        try
        {
            // One run: `count` dispatches recorded, then submitted, then waited for; the three times in µs.
            (double Record, double Submit, double Wait) Run(int mode, int count)
            {
                _usePush = mode == 0;
                long start = Stopwatch.GetTimestamp();
                lock (_gate)
                {
                    for (int i = 0; i < count; i++)
                    {
                        Dispatch(kernels[mode], 1, 1, 1, storages, []);
                    }

                    long recorded = Stopwatch.GetTimestamp();
                    Submit();
                    long submitted = Stopwatch.GetTimestamp();
                    WaitUntil(ulong.MaxValue);
                    long done = Stopwatch.GetTimestamp();
                    return (Stopwatch.GetElapsedTime(start, recorded).TotalMicroseconds, Stopwatch.GetElapsedTime(recorded, submitted).TotalMicroseconds,
                        Stopwatch.GetElapsedTime(submitted, done).TotalMicroseconds);
                }
            }

            foreach (int mode in modes)
            {
                Run(mode, Count);                                          // builds the pipeline and warms up, untimed
            }

            for (int round = 0; round < Rounds; round++)
            {
                foreach (int mode in modes)
                {
                    var (r, s, w) = Run(mode, Count);
                    dispatch[mode].Add((r + s + w) / Count);
                    record[mode].Add(r / Count);
                    var (_, s1, w1) = Run(mode, 1);
                    submit[mode].Add(s1);
                    roundTrip[mode].Add(s1 + w1);
                }
            }
        }
        finally
        {
            lock (_gate)
            {
                SubmitAndWait();
                foreach (var kernel in kernels)
                {
                    if (_pipelines.Remove(kernel, out var pipeline))
                    {
                        vkDestroyPipeline(_device, pipeline.Handle, null);
                    }
                }

                // Descriptor pools were sized for the measurement's batches: new ones follow the chosen batch size.
                foreach (var batch in _free.Append(_batch))
                {
                    foreach (var pool in batch.Pools)
                    {
                        vkDestroyDescriptorPool(_device, pool.Handle, null);
                    }

                    batch.Pools.Clear();
                    batch.Pool = 0;
                }

                (Dispatches, Submissions, Barriers) = (0, 0, 0);
            }

            foreach (var storage in storages)
            {
                storage.Release();
            }
        }

        double pushed = canPush ? Median(dispatch[0]) : 0, sets = Median(dispatch[1]);
        bool push = canPush && !(sets < pushed * TuneMargin);              // pushing is the default: less host work
        var choices = new Dictionary<string, string> { ["push-descriptors"] = push ? "1" : "0" };
        foreach (int mode in modes)
        {
            double recordUs = Math.Max(Median(record[mode]), 1e-3), submitUs = Median(submit[mode]), tripUs = Median(roundTrip[mode]);
            int batch = (int)Math.Clamp(Math.Ceiling(submitUs / (SubmitShare * recordUs)), 1, int.MaxValue);
            int inFlight = (int)Math.Clamp(Math.Ceiling(tripUs / (batch * recordUs)) + 1, 2, int.MaxValue);
            string name = mode == 0 ? "pushed" : "sets";
            choices[$"batch-commands/{name}"] = batch.ToString(CultureInfo.InvariantCulture);
            choices[$"in-flight/{name}"] = inFlight.ToString(CultureInfo.InvariantCulture);
        }

        int chosen = push ? 0 : 1;
        var measurement = new RuntimeMeasurement(Math.Round(pushed, 2), Math.Round(sets, 2), Math.Round(Median(record[chosen]), 3),
            Math.Round(Median(submit[chosen]), 1), Math.Round(Median(roundTrip[chosen]), 1));
        return (measurement, choices);
    }

    private static double Median(List<double> values)
    {
        if (values.Count == 0)
        {
            return 0;
        }

        values.Sort();
        int middle = values.Count / 2;
        return values.Count % 2 == 1 ? values[middle] : (values[middle - 1] + values[middle]) / 2;
    }

    // A positive count from the environment variable `name`, or null.
    private static int? PositiveSetting(string name) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value > 0 ? value : null;

    // A positive byte count from the environment variable `name`, or null.
    private static long? BytesSetting(string name) =>
        long.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out long value) && value > 0 ? value : null;
}
