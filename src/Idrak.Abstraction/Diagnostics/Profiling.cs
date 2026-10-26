// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using Idrak.Abstraction.Devices;

namespace Idrak.Abstraction.Diagnostics;

/// <summary>One measured value of a <see cref="ProfileRun"/>.</summary>
/// <param name="Name">What it is ("prefill", "allocated", "operations").</param>
/// <param name="Value">The value, in <paramref name="Unit"/>.</param>
/// <param name="Unit">"ms", "MB", "tok/s", "calls", … (empty for a plain number).</param>
/// <param name="Source">The probe that measured it (<see cref="IProfileProbe.Name"/>), or null for the workload and the runner.</param>
public readonly record struct ProfileValue(string Name, double Value, string Unit, string? Source);

/// <summary>A table a probe or a workload adds to a run (the kernels of a pass, the operations by count), its cells as text.</summary>
/// <param name="Name">The table's title.</param>
/// <param name="Columns">The column headings.</param>
/// <param name="Rows">The rows, each as many cells as there are columns.</param>
public sealed record ProfileTable(string Name, IReadOnlyList<string> Columns, IReadOnlyList<IReadOnlyList<string>> Rows);

/// <summary>What a <see cref="ProfileRun"/> was for.</summary>
public enum ProfileRunKind
{
    /// <summary>The first run, reported apart (one-time work: building, loading kernels, measuring choices).</summary>
    Cold,

    /// <summary>A timed run after the first; the summary's statistic is over these.</summary>
    Warm,

    /// <summary>A run for one intrusive probe (<see cref="IProfileProbe.Intrusive"/>), not timed: only that probe's values count.</summary>
    Probe,

    /// <summary>
    /// The one run of work that cannot be repeated (a command, an app's session), watched as it ran (<see cref="Profiler.Watch"/>):
    /// neither cold nor warm, its values reported as they are.
    /// </summary>
    Once,
}

/// <summary>How the warm runs of a profile are summed up into one value.</summary>
public enum ProfileStatistic
{
    /// <summary>The middle value (the lower of the two middle ones for an even count).</summary>
    Median,

    /// <summary>The mean.</summary>
    Mean,

    /// <summary>The smallest.</summary>
    Min,

    /// <summary>The largest.</summary>
    Max,
}

/// <summary>
/// One run of a profiled workload: the workload marks its stages and records what it knows (tokens, items), the probes add
/// what they measured around it. Not thread-safe: record from the thread running the workload.
/// </summary>
public sealed class ProfileRun
{
    private readonly List<ProfileValue> _values = [];
    private readonly List<ProfileTable> _tables = [];

    internal ProfileRun(int index, ProfileRunKind kind, Device device, string? probe) => (Index, Kind, Device, Probe) = (index, kind, device, probe);

    /// <summary>The run's place in the profile, from 0.</summary>
    public int Index { get; }

    /// <summary>Cold, warm, the pass of one intrusive probe, or the one run watched.</summary>
    public ProfileRunKind Kind { get; }

    /// <summary>The device profiled.</summary>
    public Device Device { get; }

    /// <summary>The intrusive probe this run was for (<see cref="ProfileRunKind.Probe"/>), else null.</summary>
    public string? Probe { get; }

    /// <summary>The values recorded so far, in order.</summary>
    public IReadOnlyList<ProfileValue> Values => _values;

    /// <summary>The tables added so far, in order.</summary>
    public IReadOnlyList<ProfileTable> Tables => _tables;

    // The probe whose end is recording (its values carry it as their Source), or null.
    internal string? Source { get; set; }

    /// <summary>The value named <paramref name="name"/>, or null.</summary>
    public double? Value(string name)
    {
        foreach (var value in _values)
        {
            if (value.Name == name)
            {
                return value.Value;
            }
        }

        return null;
    }

    /// <summary>Records <paramref name="value"/> as <paramref name="name"/> (replacing an earlier value of that name).</summary>
    public void Record(string name, double value, string unit = "")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(unit);
        int at = _values.FindIndex(v => v.Name == name);
        var entry = new ProfileValue(name, value, unit, Source);
        if (at >= 0)
        {
            _values[at] = entry;
        }
        else
        {
            _values.Add(entry);
        }
    }

    /// <summary>Records a stage's time as <paramref name="name"/>, in ms (a stage the workload timed itself, or one a library reports).</summary>
    public void Stage(string name, TimeSpan elapsed) => Record(name, elapsed.TotalMilliseconds, "ms");

    /// <summary>Runs <paramref name="body"/> as the stage <paramref name="name"/>: its time, the device's queued work included, in ms.</summary>
    public T Stage<T>(string name, Func<T> body)
    {
        ArgumentNullException.ThrowIfNull(body);
        long start = Stopwatch.GetTimestamp();
        var result = body();
        Device.Synchronize();
        Stage(name, Stopwatch.GetElapsedTime(start));
        return result;
    }

    /// <summary>Runs <paramref name="body"/> as the stage <paramref name="name"/>: its time, the device's queued work included, in ms.</summary>
    public void Stage(string name, Action body)
    {
        ArgumentNullException.ThrowIfNull(body);
        Stage(name, () =>
        {
            body();
            return 0;
        });
    }

    /// <summary>Adds <paramref name="table"/> to the run.</summary>
    public void Add(ProfileTable table)
    {
        ArgumentNullException.ThrowIfNull(table);
        _tables.Add(table);
    }
}

/// <summary>A value of a profile summed up: the cold run's, the statistic over the warm runs, their least and most.</summary>
/// <param name="Name">The value's name.</param>
/// <param name="Unit">Its unit.</param>
/// <param name="Source">The probe that measured it, or null for the workload and the runner.</param>
/// <param name="Cold">The cold run's value, or null (no cold run, or not measured in it).</param>
/// <param name="Value">The statistic over the warm runs; for an intrusive probe's value, its pass's value; for a watched run (<see cref="ProfileRunKind.Once"/>), its value.</param>
/// <param name="Min">The least of the warm runs (the pass's or the watched run's value).</param>
/// <param name="Max">The most of the warm runs (the pass's or the watched run's value).</param>
/// <param name="Count">The warm runs that measured it (1 for an intrusive probe's pass and for a watched run).</param>
public sealed record ProfileSummary(string Name, string Unit, string? Source, double? Cold, double? Value, double? Min, double? Max, int Count);

/// <summary>
/// What to profile and how: the runs, the probes, the statistic, and how the report is written. Every field can come from
/// the environment (<see cref="FromEnvironment"/>), so whoever runs a profile picks the criteria without code.
/// </summary>
public sealed record ProfileOptions
{
    /// <summary>Whether a first run is made and reported apart (true by default).</summary>
    public bool Cold { get; init; } = true;

    /// <summary>The timed runs after the cold one (3 by default; at least 1).</summary>
    public int WarmRuns { get; init; } = 3;

    /// <summary>
    /// The probes to measure with, by name (<see cref="ProfileProbes"/>); null for every registered probe the device
    /// supports. A named probe the device does not support is skipped and listed in <see cref="ProfileReport.Skipped"/>.
    /// </summary>
    public IReadOnlyList<string>? Probes { get; init; }

    /// <summary>How the warm runs are summed up (the median by default).</summary>
    public ProfileStatistic Statistic { get; init; } = ProfileStatistic.Median;

    /// <summary>The writer <see cref="ProfileReport.Write()"/> uses, by name (<see cref="ProfileWriters"/>; "table" by default).</summary>
    public string Format { get; init; } = ProfileWriters.Table;

    /// <summary>A file <see cref="ProfileReport.Write()"/> appends to, or null for the console.</summary>
    public string? Output { get; init; }

    /// <summary>
    /// <paramref name="defaults"/> (or the defaults) with what the environment sets: <c>IDRAK_PROFILE_COLD</c> (1 or 0),
    /// <c>IDRAK_PROFILE_WARM</c> (the warm runs), <c>IDRAK_PROFILE_PROBES</c> (names separated by commas, "all" for every
    /// probe, "none" for none), <c>IDRAK_PROFILE_STATISTIC</c> (median, mean, min or max), <c>IDRAK_PROFILE_FORMAT</c>
    /// (a writer's name) and <c>IDRAK_PROFILE_OUT</c> (a file).
    /// </summary>
    /// <exception cref="ArgumentException">A value that does not parse.</exception>
    public static ProfileOptions FromEnvironment(ProfileOptions? defaults = null) => From(Environment.GetEnvironmentVariable, defaults);

    internal static ProfileOptions From(Func<string, string?> variable, ProfileOptions? defaults)
    {
        var options = defaults ?? new ProfileOptions();
        if (variable("IDRAK_PROFILE_COLD") is { Length: > 0 } cold)
        {
            options = options with { Cold = cold.Trim() is "1" or "true" or "True" ? true : cold.Trim() is "0" or "false" or "False" ? false : throw Bad("IDRAK_PROFILE_COLD", cold, "1 or 0") };
        }

        if (variable("IDRAK_PROFILE_WARM") is { Length: > 0 } warm)
        {
            options = options with { WarmRuns = int.TryParse(warm, NumberStyles.Integer, CultureInfo.InvariantCulture, out int runs) && runs >= 1 ? runs : throw Bad("IDRAK_PROFILE_WARM", warm, "a count of 1 or more") };
        }

        if (variable("IDRAK_PROFILE_PROBES") is { Length: > 0 } probes)
        {
            options = options with
            {
                Probes = probes.Trim().ToLowerInvariant() switch
                {
                    "all" => null,
                    "none" => [],
                    _ => [.. probes.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)],
                },
            };
        }

        if (variable("IDRAK_PROFILE_STATISTIC") is { Length: > 0 } statistic)
        {
            options = options with { Statistic = Enum.TryParse<ProfileStatistic>(statistic.Trim(), ignoreCase: true, out var s) && Enum.IsDefined(s) ? s : throw Bad("IDRAK_PROFILE_STATISTIC", statistic, "median, mean, min or max") };
        }

        if (variable("IDRAK_PROFILE_FORMAT") is { Length: > 0 } format)
        {
            options = options with { Format = format.Trim() };
        }

        if (variable("IDRAK_PROFILE_OUT") is { Length: > 0 } output)
        {
            options = options with { Output = output.Trim() };
        }

        return options;

        static ArgumentException Bad(string name, string value, string expected) => new($"{name}={value} is not {expected}.");
    }
}

/// <summary>
/// A profile: the workload's runs (cold, warm, and one pass for each intrusive probe) with what was measured, summed up by
/// <see cref="ProfileOptions.Statistic"/>. Written by any registered writer (<see cref="Write(TextWriter, string?)"/>).
/// </summary>
public sealed class ProfileReport
{
    internal ProfileReport(string name, Device device, ProfileOptions options, IReadOnlyList<ProfileRun> runs, IReadOnlyList<string> probes, IReadOnlyList<string> skipped,
        IReadOnlyList<string>? notes = null)
    {
        (Name, Device, Options, Runs, Probes, Skipped, Notes) = (name, device, options, runs, probes, skipped, notes ?? []);
        Summary = Summarize();
    }

    /// <summary>What was profiled.</summary>
    public string Name { get; }

    /// <summary>The device it ran on.</summary>
    public Device Device { get; }

    /// <summary>The criteria it ran with.</summary>
    public ProfileOptions Options { get; }

    /// <summary>Every run in order: the cold one, the warm ones, then the intrusive probes' passes.</summary>
    public IReadOnlyList<ProfileRun> Runs { get; }

    /// <summary>The probes measured with.</summary>
    public IReadOnlyList<string> Probes { get; }

    /// <summary>The probes asked for that the device does not support.</summary>
    public IReadOnlyList<string> Skipped { get; }

    /// <summary>
    /// Lines the writers print under the title: what a watched run was and how it ended, the probes it left out
    /// (<see cref="ProfileSession.Note"/>, <see cref="Profiler.Watch"/>); empty for <see cref="Profiler.Run"/>.
    /// </summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>Whether the report is of one run watched as it ran (<see cref="Profiler.Watch"/>), not of runs repeated.</summary>
    public bool Watched => Runs.Count > 0 && Runs.All(r => r.Kind == ProfileRunKind.Once);

    /// <summary>Every value summed up: the timed runs' values first (in the order first recorded), then the intrusive probes' own.</summary>
    public IReadOnlyList<ProfileSummary> Summary { get; }

    /// <summary>The summary of the value <paramref name="name"/>, or null.</summary>
    public ProfileSummary? this[string name] => Summary.FirstOrDefault(s => s.Name == name);

    /// <summary>Writes the report with the writer <paramref name="format"/> (<see cref="ProfileOptions.Format"/> when null).</summary>
    /// <exception cref="NotSupportedException">No writer has that name.</exception>
    public void Write(TextWriter output, string? format = null)
    {
        ArgumentNullException.ThrowIfNull(output);
        ProfileWriters.Get(format ?? Options.Format).Write(this, output);
    }

    /// <summary>Writes the report with <see cref="ProfileOptions.Format"/> to <see cref="ProfileOptions.Output"/> (appended), or to the console.</summary>
    public void Write()
    {
        if (Options.Output is { } path)
        {
            using var file = new StreamWriter(path, append: true);
            Write(file);
        }
        else
        {
            Write(Console.Out);
        }
    }

    // Each value of the timed runs over the warm ones (and the cold one's apart); each intrusive probe's own values from its pass.
    private List<ProfileSummary> Summarize()
    {
        var cold = Runs.FirstOrDefault(r => r.Kind == ProfileRunKind.Cold);
        var warm = Runs.Where(r => r.Kind is ProfileRunKind.Warm or ProfileRunKind.Once).ToList();   // a watched run stands for the warm ones
        var summary = new List<ProfileSummary>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (var value in Runs.Where(r => r.Kind != ProfileRunKind.Probe).SelectMany(r => r.Values))
        {
            if (!seen.Add(value.Name))
            {
                continue;
            }

            var values = warm.Select(r => r.Value(value.Name)).OfType<double>().Order().ToArray();
            double? statistic = values.Length == 0 ? null : Options.Statistic switch
            {
                ProfileStatistic.Mean => values.Average(),
                ProfileStatistic.Min => values[0],
                ProfileStatistic.Max => values[^1],
                _ => values[(values.Length - 1) / 2],
            };
            summary.Add(new ProfileSummary(value.Name, value.Unit, value.Source, cold?.Value(value.Name), statistic,
                values.Length == 0 ? null : values[0], values.Length == 0 ? null : values[^1], values.Length));
        }

        foreach (var pass in Runs.Where(r => r.Kind == ProfileRunKind.Probe))
        {
            foreach (var value in pass.Values.Where(v => v.Source == pass.Probe && seen.Add(v.Name)))
            {
                summary.Add(new ProfileSummary(value.Name, value.Unit, value.Source, null, value.Value, value.Value, value.Value, 1));
            }
        }

        return summary;
    }
}

/// <summary>
/// Measures something around each run of a profiled workload: time on the processor, the managed heap, device memory, the
/// operations run, the device's kernels, or anything of one's own. Registered in <see cref="ProfileProbes"/> under
/// <see cref="Name"/>.
/// </summary>
public interface IProfileProbe
{
    /// <summary>The probe's name in <see cref="ProfileProbes"/> ("cpu", "heap").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>
    /// Whether measuring changes the run's time (a trace of every call, kernels waited for one by one): such a probe gets a
    /// run of its own, kept out of the timed runs.
    /// </summary>
    bool Intrusive { get; }

    /// <summary>Whether the probe can measure on <paramref name="device"/> (kernel timing only where the device has it).</summary>
    bool Supports(Device device);

    /// <summary>
    /// Whether <see cref="Profiler.Watch"/> measures with the probe when the options name none: by default every probe
    /// that is not <see cref="Intrusive"/>. A watched run happens once, so a probe that changes it (an intrusive one) or
    /// does work of its own on the device before it (the bandwidth's copies through device memory) says false; named in
    /// <see cref="ProfileOptions.Probes"/>, it is measured with all the same.
    /// </summary>
    bool Watched => !Intrusive;

    /// <summary>
    /// Starts measuring before the workload runs; returns what ends it after the run, recording into
    /// <paramref name="run"/> (<see cref="ProfileRun.Record"/>, <see cref="ProfileRun.Add"/>). Ended even when the
    /// workload throws.
    /// </summary>
    Action Begin(ProfileRun run);
}

/// <summary>
/// Writes a <see cref="ProfileReport"/>: a table for people, JSON, CSV or Markdown for tools and documents, or a format
/// of one's own. Registered in <see cref="ProfileWriters"/> under <see cref="Name"/>.
/// </summary>
public interface IProfileWriter
{
    /// <summary>The writer's name in <see cref="ProfileWriters"/> ("table", "json").</summary>
    string Name { get; }

    /// <summary>One line for help and listings.</summary>
    string Summary { get; }

    /// <summary>Writes <paramref name="report"/> to <paramref name="output"/>.</summary>
    void Write(ProfileReport report, TextWriter output);
}

/// <summary>
/// Profiles a workload: runs it cold once (unless <see cref="ProfileOptions.Cold"/> is off), then
/// <see cref="ProfileOptions.WarmRuns"/> times, each between the device's queued work finished, with every non-intrusive
/// probe around it and its wall time ("wall", ms); then once more for each intrusive probe alone. Whatever the workload
/// is (a page read, a training step, a transform), whoever runs it chooses what is measured and how it is written.
/// </summary>
/// <example>
/// <code>
/// var report = Profiler.Run("read a page", device, run =>
/// {
///     var image = run.Stage("prepare", () => reader.Prepared(request));
///     var reading = run.Stage("read", () => reader.Read(request));
///     run.Record("answer tokens", reading.Stats!.GeneratedTokens, "tok");
/// }, ProfileOptions.FromEnvironment());
/// report.Write();
/// </code>
/// </example>
public static class Profiler
{
    /// <summary>Profiles <paramref name="workload"/> on <paramref name="device"/> by <paramref name="options"/>.</summary>
    /// <exception cref="NotSupportedException">A probe named in the options is not registered (the message names the registered ones).</exception>
    public static ProfileReport Run(string name, Device device, Action<ProfileRun> workload, ProfileOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(workload);
        options ??= new ProfileOptions();
        ArgumentOutOfRangeException.ThrowIfLessThan(options.WarmRuns, 1, nameof(options));
        var named = options.Probes is { } names ? names.Select(ProfileProbes.Get).ToList() : [.. ProfileProbes.Names.Select(ProfileProbes.Get)];
        var probes = named.Where(p => p.Supports(device)).ToList();
        var skipped = named.Where(p => !p.Supports(device)).Select(p => p.Name).ToList();
        var runs = new List<ProfileRun>();
        if (options.Cold)
        {
            runs.Add(Once(runs.Count, ProfileRunKind.Cold, null, probes.Where(p => !p.Intrusive)));
        }

        for (int i = 0; i < options.WarmRuns; i++)
        {
            runs.Add(Once(runs.Count, ProfileRunKind.Warm, null, probes.Where(p => !p.Intrusive)));
        }

        foreach (var probe in probes.Where(p => p.Intrusive))
        {
            runs.Add(Once(runs.Count, ProfileRunKind.Probe, probe.Name, [probe]));
        }

        return new ProfileReport(name, device, options, runs, [.. probes.Select(p => p.Name)], skipped);

        ProfileRun Once(int index, ProfileRunKind kind, string? pass, IEnumerable<IProfileProbe> around)
        {
            var run = new ProfileRun(index, kind, device, pass);
            device.Synchronize();
            var ends = new List<(string Name, Action End)>();
            try
            {
                foreach (var probe in around)
                {
                    ends.Add((probe.Name, probe.Begin(run)));
                }

                long start = Stopwatch.GetTimestamp();
                workload(run);
                device.Synchronize();
                run.Stage("wall", Stopwatch.GetElapsedTime(start));
            }
            finally
            {
                for (int i = ends.Count - 1; i >= 0; i--)
                {
                    run.Source = ends[i].Name;
                    try
                    {
                        ends[i].End();
                    }
                    finally
                    {
                        run.Source = null;
                    }
                }
            }

            return run;
        }
    }

    /// <summary>
    /// Starts watching one run of work that cannot be repeated (a command line, an app's session): every selected probe
    /// begins now and ends at <see cref="ProfileSession.End"/>, which returns a report of that one run
    /// (<see cref="ProfileRunKind.Once"/>) with its wall time ("wall", ms: host time; the device's queued work is not waited
    /// for, so a watch can end from another thread, at Ctrl+C). <see cref="ProfileOptions.Cold"/>,
    /// <see cref="ProfileOptions.WarmRuns"/> and <see cref="ProfileOptions.Statistic"/> do not apply. With no probes named,
    /// the registered probes that are <see cref="IProfileProbe.Watched"/> are measured; the others are listed in a note.
    /// An intrusive probe cannot have a run of its own here: named in <see cref="ProfileOptions.Probes"/>, it measures the
    /// same run, and a note says the times include its cost.
    /// </summary>
    /// <exception cref="NotSupportedException">A probe named in the options is not registered (the message names the registered ones).</exception>
    /// <example>
    /// <code>
    /// using var watch = Profiler.Watch("idrak reinforce …", device, ProfileOptions.FromEnvironment());
    /// RunTheCommand();
    /// watch.End().Write();
    /// </code>
    /// </example>
    public static ProfileSession Watch(string name, Device device, ProfileOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(device);
        options ??= new ProfileOptions();
        var named = options.Probes is { } names ? names.Select(ProfileProbes.Get).ToList() : [.. ProfileProbes.Names.Select(ProfileProbes.Get).Where(p => p.Watched)];
        var probes = named.Where(p => p.Supports(device)).ToList();
        var skipped = named.Where(p => !p.Supports(device)).Select(p => p.Name).ToList();
        var notes = new List<string>();
        if (options.Probes is null && ProfileProbes.Names.Where(n => !ProfileProbes.Get(n).Watched).ToList() is { Count: > 0 } left)
        {
            notes.Add($"not measured in a watched run (they would change it): {string.Join(", ", left)}; name them in IDRAK_PROFILE_PROBES to add them");
        }

        if (probes.Where(p => p.Intrusive).Select(p => p.Name).ToList() is { Count: > 0 } intrusive)
        {
            notes.Add($"intrusive probes measured in the run itself: its times include their cost ({string.Join(", ", intrusive)})");
        }

        return new ProfileSession(name, device, options, probes, skipped, notes);
    }
}

/// <summary>
/// One run of work watched as it runs (<see cref="Profiler.Watch"/>): the probes began when it was made and end at
/// <see cref="End"/> (or <see cref="Dispose"/>), once, from any thread. A probe whose end fails leaves a note instead of
/// its values, so the rest of the report still comes.
/// </summary>
public sealed class ProfileSession : IDisposable
{
    private readonly Lock _lock = new();
    private readonly string _name;
    private readonly ProfileOptions _options;
    private readonly List<string> _probes;
    private readonly List<string> _skipped;
    private readonly List<string> _notes;
    private readonly List<(string Name, Action End)> _ends = [];
    private readonly long _start;
    private ProfileReport? _report;

    internal ProfileSession(string name, Device device, ProfileOptions options, List<IProfileProbe> probes, List<string> skipped, List<string> notes)
    {
        (_name, _options, _skipped, _notes) = (name, options, skipped, notes);
        Run = new ProfileRun(0, ProfileRunKind.Once, device, null);
        _probes = [.. probes.Select(p => p.Name)];
        try
        {
            foreach (var probe in probes)
            {
                _ends.Add((probe.Name, probe.Begin(Run)));
            }
        }
        catch
        {
            EndProbes();
            throw;
        }

        _start = Stopwatch.GetTimestamp();
    }

    /// <summary>The run being watched: record the work's own values and stages into it (from the thread doing the work).</summary>
    public ProfileRun Run { get; }

    /// <summary>Whether <see cref="End"/> was called.</summary>
    public bool Ended
    {
        get
        {
            lock (_lock)
            {
                return _report is not null;
            }
        }
    }

    /// <summary>Adds a line to the report's <see cref="ProfileReport.Notes"/> (how the work ended, what it was); ignored once ended.</summary>
    public void Note(string line)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(line);
        lock (_lock)
        {
            if (_report is null)
            {
                _notes.Add(line);
            }
        }
    }

    /// <summary>Ends every probe (the last begun first) and returns the report; again, the same report.</summary>
    public ProfileReport End()
    {
        lock (_lock)
        {
            if (_report is null)
            {
                Run.Stage("wall", Stopwatch.GetElapsedTime(_start));
                EndProbes();
                _report = new ProfileReport(_name, Run.Device, _options, [Run], _probes, _skipped, _notes);
            }

            return _report;
        }
    }

    /// <summary>Ends the watch (<see cref="End"/>) if it was not ended.</summary>
    public void Dispose() => End();

    private void EndProbes()
    {
        for (int i = _ends.Count - 1; i >= 0; i--)
        {
            Run.Source = _ends[i].Name;
            try
            {
                _ends[i].End();
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                _notes.Add($"probe {_ends[i].Name} failed at its end: {e.Message.Split('\n')[0]}");
            }
            finally
            {
                Run.Source = null;
            }
        }

        _ends.Clear();
    }
}

/// <summary>
/// The probes a <see cref="Profiler"/> measures with, by name (ignoring case). The library's, in this order:
/// <list type="bullet">
/// <item>"bandwidth": the device's memory bandwidth ("device bandwidth", GB/s): copies through device memory, measured once
/// per device before its first run, so a run's rate of bytes read (a decoding step's weights) has its ceiling beside it.
/// Not <see cref="IProfileProbe.Watched"/>: its copies would run inside a watched command.</item>
/// <item>"cpu": the process's processor time ("cpu", ms) and the cores it kept busy ("cores busy": processor time over wall time).</item>
/// <item>"heap": the managed heap: bytes allocated on every thread ("allocated", MB), collections of each generation
/// ("gc0", "gc1", "gc2") and the time paused for them ("gc pause", ms).</item>
/// <item>"memory": the device's memory after the run ("device in use", "device cached", MB), its peak during the run
/// ("device peak", MB) and the largest single block taken ("device largest block", MB), on devices that report it what
/// their driver has free ("device free", MB) and what neither the library nor that holds ("device other", MB: the
/// driver's context and kernels, other programs), and the process's working set ("process", MB) and peak since it
/// started ("process peak", MB).</item>
/// <item>"tuning": the kernel choices the devices measured during the run (a shape class met for the first time, from the
/// telemetry's <see cref="TuningCompleted"/>): "tunings" (count), "tuning time" (ms, nested measurements counted within
/// their outer one), "tuned operations", and the table "tuning" by device and operation, slowest first.</item>
/// <item>"fallbacks": the operations the run's device ran through the host fallback (each call copies its operands to the
/// CPU and back): "fallback calls" (calls) and "fallback operations", with the table "host fallbacks" by operation (the
/// device's counts by operation while the probe listens at <see cref="TelemetryLevel.Devices"/>, and the first fallback of
/// an operation on another device from <see cref="HostFallback"/>).</item>
/// <item>"graphs": the graph recordings that failed during the run (<see cref="GraphRecordingFailed"/>: the work ran as
/// ordinary steps instead, and ran again what the recording did before it failed): "graph recordings failed" (count), with
/// the table "graph recordings failed" by device, work and reason.</item>
/// <item>"inference": the stages of reading an answer (<see cref="InferenceStage"/>) listened to at
/// <see cref="TelemetryLevel.Stages"/>, which waits for nothing: for each stage seen "{stage} time" (ms), "{stage} events"
/// and "{stage} count" (what it counts: tokens for the prompt's pass and decoding), "prefill pieces", "prompt rate" and
/// "decode rate" (tok/s), and the table "inference stages" (events, count, rows, pieces, piece size, ms, MB, share). The
/// prompt's pass and the decoding chunks end with their tokens read back, so their times are complete; the other stages'
/// times are the host's (queued device work counts in the next stage).</item>
/// <item>"operations" (intrusive): every operation called on the device ("operations", calls; "operation kinds") and every
/// host fallback ("host fallbacks", calls), with tables of both by operation.</item>
/// <item>"kernels" (intrusive; devices that time their kernels, such as CUDA): the device's time in kernels ("kernel time",
/// ms), the launches ("launches"), the matrix products' rate ("matrix TFLOPS"), and a table of the kernels by time; recorded
/// graphs run step by step meanwhile (<see cref="ComputeGraph.RunStepsDirectly"/>), so a decoding step's kernels show too.</item>
/// <item>"stages" (intrusive: listening at <see cref="TelemetryLevel.Inference"/> makes <c>Module.Predict</c> wait for the
/// device): the stages of reading an answer (<see cref="InferenceStage"/>) in the run: "{stage} time" (ms), "{stage} count"
/// and "{stage} size" (MB) for each stage seen (the caches allocated, the encoder's output, a recorded graph's memory), and
/// the device's peak above the run's start when the encoder ends ("encoder peak above start", MB: its activations and output).</item>
/// <item>"timings" (intrusive: the device is waited for at every operation, <see cref="Telemetry.SynchronizeForTiming"/>):
/// the time of every operation by name, direction and output shape, and of every layer by type and depth, from the
/// telemetry's operation and layer events ("operation time", ms; "operation calls"; "operation shapes"; "layer calls"),
/// with the tables "operation timings" and "layer timings", slowest first. Nothing is subscribed outside its run.</item>
/// </list>
/// Register another with <see cref="Register"/>; one under a library name shadows the library's, which
/// <see cref="Unregister"/> brings back. The registry has no guard: a probe holds what it measures through a run, so it
/// cannot be swapped half-way.
/// </summary>
public static class ProfileProbes
{
    private static readonly SlotTable<string, IProfileProbe> Table = BuiltIn();

    static ProfileProbes() => LibraryDefaults.Ensure(typeof(ProfileProbes));   // the packages' probes ("network", "cluster")

    private static SlotTable<string, IProfileProbe> BuiltIn()
    {
        var table = new SlotTable<string, IProfileProbe>(nameof(ProfileProbes), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a probe holds what it measures from before a run to after it, so it cannot be swapped half-way");
        // The bandwidth first: its one measurement (before the first run) is done before the other probes start theirs.
        foreach (var probe in new IProfileProbe[]
                 {
                     new BandwidthProbe(), new CpuProbe(), new HeapProbe(), new MemoryProbe(), new TuningProbe(), new FallbacksProbe(), new GraphsProbe(), new InferenceProbe(),
                     new OperationsProbe(), new KernelsProbe(), new StagesProbe(), new TimingsProbe(),
                 })
        {
            table.RegisterDefault(probe.Name, probe);
        }

        return table;
    }

    /// <summary>Registers <paramref name="probe"/> under its <see cref="IProfileProbe.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IProfileProbe probe)
    {
        ArgumentNullException.ThrowIfNull(probe);
        ArgumentException.ThrowIfNullOrWhiteSpace(probe.Name);
        Table.Register(probe.Name, probe, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's probe <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The probe registered as <paramref name="name"/>, or null.</summary>
    public static IProfileProbe? Find(string name) => Table.Find(name);

    /// <summary>The probe registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is: the message names the registered ones.</exception>
    public static IProfileProbe Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(ProfileProbes), name, Table.Keys,
            $"Profile probe '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with ProfileProbes.Register.");

    /// <summary>The library's probe <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IProfileProbe? Default(string name) => Table.Default(name);

    /// <summary>Who registered the probe <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered probe as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(p => $"{p.Name}{(p.Intrusive ? " (a run of its own)" : "")}: {p.Summary}"));

    private sealed class BandwidthProbe : IProfileProbe
    {
        public string Name => "bandwidth";

        public string Summary => "the device's memory bandwidth as copies through it measure it, once per device before its first run: the ceiling of a rate of bytes read";

        public bool Intrusive => false;

        public bool Watched => false;                                           // its copies would run inside a watched command

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            double rate = Bandwidth(run.Device.Backend);
            return () =>
            {
                if (rate > 0)
                {
                    run.Record("device bandwidth", rate, "GB/s");
                }
            };
        }
    }

    // The bandwidth measured on each device, kept with its backend.
    private static readonly ConditionalWeakTable<Backend, StrongBox<double>> Bandwidths = [];

    /// <summary>
    /// GB/s that <see cref="Backend.Copy"/> moves on <paramref name="backend"/> (bytes read and written), measured once per
    /// device: buffers from 16 MiB, doubled while the rate still falls by more than a tenth (the device's caches hold the
    /// small ones, whatever their size), up to an eighth of the memory it reports available (the host's for a device that
    /// reports none); the best of three copies at each size. 0 when not even the first size fits.
    /// </summary>
    internal static double Bandwidth(Backend backend)
    {
        var box = Bandwidths.GetValue(backend, _ => new StrongBox<double>(double.NaN));
        lock (box)
        {
            if (double.IsNaN(box.Value))
            {
                box.Value = MeasureBandwidth(backend);
            }

            return box.Value;
        }
    }

    private static double MeasureBandwidth(Backend backend)
    {
        long free = backend.AvailableMemory() ?? GC.GetGCMemoryInfo().TotalAvailableMemoryBytes;
        long limit = Math.Min(free / 8, (long)int.MaxValue * sizeof(float));
        double previous = 0, rate = 0;
        for (long bytes = 16L << 20; 2 * bytes <= limit; bytes *= 2)
        {
            int n = (int)(bytes / sizeof(float));
            var x = backend.Allocate(n, zeroed: true);
            var y = backend.Allocate(n, zeroed: false);
            try
            {
                backend.Copy(x, y, n);
                backend.Synchronize();
                double best = double.MaxValue;
                for (int i = 0; i < 3; i++)
                {
                    long start = Stopwatch.GetTimestamp();
                    backend.Copy(x, y, n);
                    backend.Synchronize();
                    best = Math.Min(best, Stopwatch.GetElapsedTime(start).TotalSeconds);
                }

                rate = 2.0 * bytes / Math.Max(best, 1e-9) / 1e9;
            }
            finally
            {
                x.Release();
                y.Release();
            }

            if (previous > 0 && rate >= 0.9 * previous)
            {
                break;                                                       // past the caches: the rate holds
            }

            previous = rate;
        }

        return rate;
    }

    // The kernel choices measured during a run (TuningCompleted, at TelemetryLevel.Devices): count, candidates and time by
    // device and operation. Not intrusive: the event is published once per shape class, when the device has measured.
    private sealed class TuningProbe : IProfileProbe
    {
        public string Name => "tuning";

        public string Summary => "the kernel choices the devices measured in the run (shape classes met for the first time): count and time by operation and device";

        public bool Intrusive => false;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            var hook = new TuningSums();
            var subscription = Telemetry.Subscribe(hook);
            return () =>
            {
                subscription.Dispose();
                lock (hook)
                {
                    var rows = hook.Sums.OrderByDescending(t => t.Value.Ms).ThenBy(t => t.Key.Operation, StringComparer.Ordinal).ToList();
                    run.Record("tunings", rows.Sum(t => t.Value.Count));
                    run.Record("tuning time", hook.OuterMs, "ms");
                    run.Record("tuned operations", rows.Count);
                    if (rows.Count > 0)
                    {
                        run.Add(new ProfileTable("tuning", ["device", "operation", "tunings", "nested", "candidates", "ms", "share"],
                            [.. rows.Select(t => (IReadOnlyList<string>)[t.Key.Device, t.Key.Operation, Text(t.Value.Count), Text(t.Value.Nested), Text(t.Value.Candidates),
                                Text(t.Value.Ms), Text(hook.OuterMs > 0 ? 100 * t.Value.Ms / hook.OuterMs : 0) + "%"])]));
                    }
                }
            };
        }

        private sealed class TuningSums : ITelemetryHook
        {
            public TelemetryLevel Levels => TelemetryLevel.Devices;

            public Dictionary<(string Device, string Operation), (long Count, long Nested, long Candidates, double Ms)> Sums { get; } = [];

            // The time of the measurements not inside another's (a nested one's time lies within its outer one's).
            public double OuterMs { get; private set; }

            public void OnTuning(in TuningCompleted e)
            {
                lock (this)
                {
                    ref var sum = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(Sums, (e.Device, e.Operation), out _);
                    double ms = e.Duration.TotalMilliseconds;
                    sum = (sum.Count + 1, sum.Nested + (e.Nested ? 1 : 0), sum.Candidates + e.Candidates, sum.Ms + ms);
                    OuterMs += e.Nested ? 0 : ms;
                }
            }
        }
    }

    // The host fallbacks of a run: the run's device's counts by operation (Kernels.HostCallsByOperation, counted while the
    // probe listens at TelemetryLevel.Devices) and the first fallback of an operation on any other device (HostFallback).
    // Not intrusive: a fallback is counted where it happens, nothing is traced.
    private sealed class FallbacksProbe : IProfileProbe
    {
        public string Name => "fallbacks";

        public string Summary => "the operations run through the host fallback (operands copied to the CPU and back): calls by operation";

        public bool Intrusive => false;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            var backend = run.Device.Backend;
            var hook = new FirstFallbacks();
            var subscription = Telemetry.Subscribe(hook);                      // counting by operation starts with it
            long calls = Operations.Kernels.HostCalls(backend);
            var before = Operations.Kernels.HostCallsByOperation(backend);
            return () =>
            {
                long total = Operations.Kernels.HostCalls(backend) - calls;
                var after = Operations.Kernels.HostCallsByOperation(backend);
                subscription.Dispose();
                // Why: the device's kernels unavailable as a whole (IKernelAvailability), else no kernel of its own for it.
                string own = total > 0 && (backend as IKernelAvailability)?.KernelsUnavailable() is { } off ? $"kernels unavailable: {off}" : "no kernel of its own";
                var rows = after.Select(c => (Device: backend.Kind, Operation: c.Key, Calls: c.Value - before.GetValueOrDefault(c.Key)))
                    .Where(c => c.Calls > 0).OrderByDescending(c => c.Calls).ThenBy(c => c.Operation, StringComparer.Ordinal)
                    .Select(c => (c.Device, c.Operation, Calls: Text(c.Calls), Why: own)).ToList();
                lock (hook)
                {
                    // Another device's first fallback of an operation: its count is that device's, not watched here.
                    rows.AddRange(hook.Seen.Where(f => f.Device != backend.Kind)
                        .Select(f => (f.Device, f.Operation, Calls: "first", Why: f.Why is { } why ? $"kernels unavailable: {why}" : "no kernel of its own")));
                }

                run.Record("fallback calls", total, "calls");
                run.Record("fallback operations", rows.Count);
                if (rows.Count > 0)
                {
                    run.Add(new ProfileTable("host fallbacks", ["device", "operation", "calls", "why"], [.. rows.Select(r => (IReadOnlyList<string>)[r.Device, r.Operation, r.Calls, r.Why])]));
                }
            };
        }

        private sealed class FirstFallbacks : ITelemetryHook
        {
            public TelemetryLevel Levels => TelemetryLevel.Devices;

            public List<(string Device, string Operation, string? Why)> Seen { get; } = [];

            public void OnHostFallback(in HostFallback e)
            {
                lock (this)
                {
                    Seen.Add((e.Device, e.Operation, e.KernelsUnavailable));
                }
            }
        }
    }

    // The graph recordings that failed in a run (GraphRecordingFailed, at TelemetryLevel.Devices): count by device, work and
    // reason. Not intrusive: the event is published where a recording fails, nothing is traced.
    private sealed class GraphsProbe : IProfileProbe
    {
        public string Name => "graphs";

        public string Summary => "the graph recordings that failed (the work ran as ordinary steps instead): count by device, work and reason";

        public bool Intrusive => false;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            var hook = new FailedRecordings();
            var subscription = Telemetry.Subscribe(hook);
            return () =>
            {
                subscription.Dispose();
                lock (hook)
                {
                    var rows = hook.Counts.OrderByDescending(f => f.Value).ThenBy(f => f.Key.Reason, StringComparer.Ordinal).ToList();
                    run.Record("graph recordings failed", rows.Sum(f => f.Value));
                    if (rows.Count > 0)
                    {
                        run.Add(new ProfileTable("graph recordings failed", ["device", "work", "times", "reason"],
                            [.. rows.Select(f => (IReadOnlyList<string>)[f.Key.Device, f.Key.Work, Text(f.Value), f.Key.Reason])]));
                    }
                }
            };
        }

        private sealed class FailedRecordings : ITelemetryHook
        {
            public TelemetryLevel Levels => TelemetryLevel.Devices;

            public Dictionary<(string Device, string Work, string Reason), long> Counts { get; } = [];

            public void OnGraphRecordingFailed(in GraphRecordingFailed e)
            {
                lock (this)
                {
                    Counts[(e.Device, e.Work, e.Reason)] = Counts.GetValueOrDefault((e.Device, e.Work, e.Reason)) + 1;
                }
            }
        }
    }

    // The stages of reading an answer (InferenceStage) at TelemetryLevel.Stages, which waits for nothing (unlike the
    // Inference level, whose Predict events wait for the device): each stage's events, counts, rows, pieces, time and bytes.
    private sealed class InferenceProbe : IProfileProbe
    {
        // Each stage's name in values and tables ("decode chunk"), made once.
        private static readonly string[] StageNames = [.. Enum.GetValues<InferenceStageKind>().Select(k =>
            string.Concat(k.ToString().Select((c, i) => char.IsUpper(c) && i > 0 ? " " + char.ToLowerInvariant(c) : char.ToLowerInvariant(c).ToString())))];

        public string Name => "inference";

        public string Summary => "the stages of reading an answer without waiting for the device: the prompt's passes and their pieces, decoding chunks, "
                                 + "encoders, caches and graphs: events, tokens, time and rates";

        public bool Intrusive => false;

        public bool Supports(Device device) => true;

        private static string StageName(InferenceStageKind stage) => (int)stage < StageNames.Length ? StageNames[(int)stage] : stage.ToString();

        public Action Begin(ProfileRun run)
        {
            var hook = new StageSums();
            var subscription = Telemetry.Subscribe(hook);
            return () =>
            {
                subscription.Dispose();
                lock (hook)
                {
                    double total = hook.Sums.Values.Sum(v => v.Ms);
                    foreach (var (stage, sum) in hook.Sums)
                    {
                        string name = StageName(stage);
                        run.Record(string.Create(CultureInfo.InvariantCulture, $"{name} time"), sum.Ms, "ms");
                        run.Record(string.Create(CultureInfo.InvariantCulture, $"{name} events"), sum.Events);
                        run.Record(string.Create(CultureInfo.InvariantCulture, $"{name} count"), sum.Count);
                    }

                    if (hook.Sums.TryGetValue(InferenceStageKind.Prefill, out var prefill))
                    {
                        run.Record("prefill pieces", prefill.Pieces);
                        run.Record("prompt rate", prefill.Ms > 0 ? prefill.Count / (prefill.Ms / 1000) : 0, "tok/s");
                    }

                    if (hook.Sums.TryGetValue(InferenceStageKind.DecodeChunk, out var decode))
                    {
                        run.Record("decode rate", decode.Ms > 0 ? decode.Count / (decode.Ms / 1000) : 0, "tok/s");
                    }

                    if (hook.Sums.Count > 0)
                    {
                        run.Add(new ProfileTable("inference stages", ["stage", "events", "count", "rows", "pieces", "piece size", "ms", "MB", "share"],
                            [.. hook.Sums.OrderByDescending(s => s.Value.Ms).Select(s => (IReadOnlyList<string>)[StageName(s.Key), Text(s.Value.Events), Text(s.Value.Count),
                                Text(s.Value.Rows), Text(s.Value.Pieces), s.Value.PieceSize > 0 ? Text(s.Value.PieceSize) : "", Text(s.Value.Ms),
                                s.Value.Bytes > 0 ? Text(s.Value.Bytes / 1048576.0) : "", Text(total > 0 ? 100 * s.Value.Ms / total : 0) + "%"])]));
                    }
                }
            };
        }

        private sealed class StageSums : ITelemetryHook
        {
            public TelemetryLevel Levels => TelemetryLevel.Stages;

            // Per stage: events, what they count, the most rows read together, the pieces, the largest piece, ms, bytes.
            public SortedDictionary<InferenceStageKind, (long Events, long Count, int Rows, long Pieces, int PieceSize, double Ms, long Bytes)> Sums { get; } = [];

            public void OnInferenceStage(in InferenceStage e)
            {
                lock (this)
                {
                    var s = Sums.GetValueOrDefault(e.Stage);
                    Sums[e.Stage] = (s.Events + 1, s.Count + e.Count, Math.Max(s.Rows, e.Rows), s.Pieces + e.Pieces, Math.Max(s.PieceSize, e.PieceSize),
                        s.Ms + e.Duration.TotalMilliseconds, s.Bytes + e.Bytes);
                }
            }
        }
    }

    // The stages of reading an answer (InferenceStage events) in a run: each stage's time, count and bytes summed, and the
    // device's peak above the run's start when the encoder ends (its activations and output). Intrusive: listening at the
    // Inference level makes Module.Predict wait for the device.
    private sealed class StagesProbe : IProfileProbe
    {
        public string Name => "stages";

        public string Summary => "the stages of reading an answer (image decode and transforms, encoder, cache allocations, prefill, decoding chunks, graph "
                                 + "captures): time, counts and MB each, and the device's peak above the run's start when the encoder ends";

        public bool Intrusive => true;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            var backend = run.Device.Backend;

            // Blocks of tensors already unreachable go back first: freed by the finalizer during the run, they would lower
            // what is in use below the start and the encoder's peak above it with it.
            GC.Collect();
            GC.WaitForPendingFinalizers();
            backend.ResetPeakMemoryUsage();
            var hook = new StageSums(backend, backend.GetMemoryUsage().InUse);
            var subscription = Telemetry.Subscribe(hook);
            return () =>
            {
                subscription.Dispose();
                lock (hook)
                {
                    foreach (var (stage, (ms, count, bytes)) in hook.Sums)
                    {
                        run.Record(string.Create(CultureInfo.InvariantCulture, $"{stage} time"), ms, "ms");
                        run.Record(string.Create(CultureInfo.InvariantCulture, $"{stage} count"), count);
                        if (bytes > 0)
                        {
                            run.Record(string.Create(CultureInfo.InvariantCulture, $"{stage} size"), bytes / 1048576.0, "MB");
                        }
                    }

                    if (hook.EncoderPeak is { } peak)
                    {
                        run.Record("encoder peak above start", peak / 1048576.0, "MB");
                    }
                }
            };
        }

        private sealed class StageSums(Backend backend, long startInUse) : ITelemetryHook
        {
            public TelemetryLevel Levels => TelemetryLevel.Inference;

            public SortedDictionary<InferenceStageKind, (double Ms, long Count, long Bytes)> Sums { get; } = [];

            public long? EncoderPeak { get; private set; }

            public void OnInferenceStage(in InferenceStage e)
            {
                lock (this)
                {
                    var (ms, count, bytes) = Sums.GetValueOrDefault(e.Stage);
                    Sums[e.Stage] = (ms + e.Duration.TotalMilliseconds, count + e.Count, bytes + e.Bytes);
                    if (e.Stage == InferenceStageKind.Encoder)
                    {
                        EncoderPeak = Math.Max(EncoderPeak ?? 0, backend.GetMemoryUsage().Peak - startInUse);
                    }
                }
            }
        }
    }

    // Time per operation (by name, direction and output shape) and per layer (by type and depth), from the telemetry's
    // operation and layer events, the device waited for before each measurement so the times are execution times.
    // Intrusive: that waiting changes the run's time. A key is stored once per new operation and shape (no string per call).
    private sealed class TimingsProbe : IProfileProbe
    {
        public string Name => "timings";

        public string Summary => "the time of every operation by name and output shape, and of every layer by type and depth, the device waited for at each";

        public bool Intrusive => true;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            bool synchronize = Telemetry.SynchronizeForTiming;
            Telemetry.SynchronizeForTiming = true;
            var hook = new TimingSums();
            var subscription = Telemetry.Subscribe(hook);
            return () =>
            {
                subscription.Dispose();
                Telemetry.SynchronizeForTiming = synchronize;
                lock (hook)
                {
                    var operations = hook.Operations.OrderByDescending(o => o.Value.Ms).ThenBy(o => o.Key.Operation, StringComparer.Ordinal).ToList();
                    var layers = hook.Layers.OrderByDescending(l => l.Value.Ms).ThenBy(l => l.Key.Type, StringComparer.Ordinal).ThenBy(l => l.Key.Depth).ToList();
                    double total = operations.Sum(o => o.Value.Ms);
                    run.Record("operation time", total, "ms");
                    run.Record("operation calls", operations.Sum(o => o.Value.Calls), "calls");
                    run.Record("operation shapes", operations.Count);
                    run.Record("layer calls", layers.Sum(l => l.Value.Calls), "calls");
                    run.Add(new ProfileTable("operation timings", ["operation", "shape", "calls", "ms", "share"],
                        [.. operations.Select(o => (IReadOnlyList<string>)[o.Key.Backward ? o.Key.Operation + " (backward)" : o.Key.Operation, Tensor.FormatShape(o.Key.Shape),
                            Text(o.Value.Calls), Text(o.Value.Ms), Text(total > 0 ? 100 * o.Value.Ms / total : 0) + "%"])]));
                    run.Add(new ProfileTable("layer timings", ["layer", "depth", "calls", "ms"],
                        [.. layers.Select(l => (IReadOnlyList<string>)[l.Key.Type, Text(l.Key.Depth), Text(l.Value.Calls), Text(l.Value.Ms)])]));
                }
            };
        }

        // An operation by name, direction and output shape (the shape compared by its values).
        private readonly record struct OperationKey(string Operation, bool Backward, int[] Shape);

        private sealed class OperationKeys : IEqualityComparer<OperationKey>
        {
            public bool Equals(OperationKey x, OperationKey y) =>
                x.Backward == y.Backward && string.Equals(x.Operation, y.Operation, StringComparison.Ordinal) && x.Shape.AsSpan().SequenceEqual(y.Shape);

            public int GetHashCode(OperationKey key)
            {
                var hash = new HashCode();
                hash.Add(key.Operation, StringComparer.Ordinal);
                hash.Add(key.Backward);
                foreach (int d in key.Shape)
                {
                    hash.Add(d);
                }

                return hash.ToHashCode();
            }
        }

        private sealed class TimingSums : ITelemetryHook
        {
            public TelemetryLevel Levels => TelemetryLevel.Operations | TelemetryLevel.Layers;

            public Dictionary<OperationKey, (long Calls, double Ms)> Operations { get; } = new(new OperationKeys());

            public Dictionary<(string Type, int Depth), (long Calls, double Ms)> Layers { get; } = [];

            public void OnOperation(in OperationCompleted e)
            {
                lock (this)
                {
                    var key = new OperationKey(e.Operation, e.Backward, e.Shape);
                    ref var sum = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrNullRef(Operations, key);
                    if (Unsafe.IsNullRef(ref sum))
                    {
                        // The first call of a shape keeps its own copy: the key must not change with a reused array.
                        Operations.Add(key with { Shape = [.. e.Shape] }, (1, e.Duration.TotalMilliseconds));
                        return;
                    }

                    sum = (sum.Calls + 1, sum.Ms + e.Duration.TotalMilliseconds);
                }
            }

            public void OnLayerForward(in LayerForward e)
            {
                lock (this)
                {
                    ref var sum = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(Layers, (e.LayerType, e.Depth), out _);
                    sum = (sum.Calls + 1, sum.Ms + e.Duration.TotalMilliseconds);
                }
            }
        }
    }

    private sealed class CpuProbe : IProfileProbe
    {
        public string Name => "cpu";

        public string Summary => "the process's processor time and the cores it kept busy";

        public bool Intrusive => false;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            var process = Process.GetCurrentProcess();
            var before = process.TotalProcessorTime;
            long start = Stopwatch.GetTimestamp();
            return () =>
            {
                double wall = Stopwatch.GetElapsedTime(start).TotalMilliseconds;
                process.Refresh();
                double cpu = (process.TotalProcessorTime - before).TotalMilliseconds;
                run.Record("cpu", cpu, "ms");
                run.Record("cores busy", wall > 0 ? cpu / wall : 0, "cores");
                process.Dispose();
            };
        }
    }

    private sealed class HeapProbe : IProfileProbe
    {
        public string Name => "heap";

        public string Summary => "the managed heap: bytes allocated, collections of each generation, time paused";

        public bool Intrusive => false;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            int gen0 = GC.CollectionCount(0), gen1 = GC.CollectionCount(1), gen2 = GC.CollectionCount(2);
            var paused = GC.GetTotalPauseDuration();
            long allocated = GC.GetTotalAllocatedBytes(precise: true);
            return () =>
            {
                run.Record("allocated", (GC.GetTotalAllocatedBytes(precise: true) - allocated) / 1048576.0, "MB");
                run.Record("gc0", GC.CollectionCount(0) - gen0);
                run.Record("gc1", GC.CollectionCount(1) - gen1);
                run.Record("gc2", GC.CollectionCount(2) - gen2);
                run.Record("gc pause", (GC.GetTotalPauseDuration() - paused).TotalMilliseconds, "ms");
            };
        }
    }

    private sealed class MemoryProbe : IProfileProbe
    {
        public string Name => "memory";

        public string Summary => "the device's memory in use, cached and at its peak, and the process's working set and peak";

        public bool Intrusive => false;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            run.Device.Backend.ResetPeakMemoryUsage();
            return () =>
            {
                var usage = run.Device.Backend.GetMemoryUsage();
                run.Record("device in use", usage.InUse / 1048576.0, "MB");
                run.Record("device cached", usage.Cached / 1048576.0, "MB");
                run.Record("device peak", usage.Peak / 1048576.0, "MB");
                run.Record("device largest block", usage.Largest / 1048576.0, "MB");
                if (run.Device.Backend.FreeMemory() is { } free && run.Device.Backend.Hardware.MemoryBytes is { } total)
                {
                    // What the library does not hold: the driver's context, its loaded kernels, other programs.
                    run.Record("device free", free / 1048576.0, "MB");
                    run.Record("device other", Math.Max(0, total - free - usage.InUse - usage.Cached) / 1048576.0, "MB");
                }

                using var process = Process.GetCurrentProcess();
                run.Record("process", process.WorkingSet64 / 1048576.0, "MB");
                run.Record("process peak", process.PeakWorkingSet64 / 1048576.0, "MB");
            };
        }
    }

    private sealed class OperationsProbe : IProfileProbe
    {
        public string Name => "operations";

        public string Summary => "every operation called on the device and every host fallback, by operation";

        public bool Intrusive => true;

        public bool Supports(Device device) => true;

        public Action Begin(ProfileRun run)
        {
            var trace = Operations.Kernels.Trace(run.Device.Backend);
            return () =>
            {
                trace.Dispose();
                var calls = Operations.Ops.All.Concat(Operations.PluginOperations.All).Select(op => (op.Name, Calls: trace.Calls(op)))
                    .Where(c => c.Calls > 0).OrderByDescending(c => c.Calls).ToList();
                var fallbacks = trace.HostCallsByOperation.OrderByDescending(f => f.Value).ToList();
                run.Record("operations", calls.Sum(c => c.Calls), "calls");
                run.Record("operation kinds", calls.Count);
                run.Record("host fallbacks", trace.HostCalls, "calls");
                run.Add(new ProfileTable("operations", ["operation", "calls"], [.. calls.Select(c => (IReadOnlyList<string>)[c.Name, Text(c.Calls)])]));
                if (fallbacks.Count > 0)
                {
                    run.Add(new ProfileTable("host fallbacks", ["operation", "calls"], [.. fallbacks.Select(f => (IReadOnlyList<string>)[f.Key, Text(f.Value)])]));
                }
            };
        }
    }

    private sealed class KernelsProbe : IProfileProbe
    {
        public string Name => "kernels";

        public string Summary => "each kernel's launches, time and matrix-product rate, recorded graphs run step by step (devices that time their kernels, such as CUDA)";

        public bool Intrusive => true;

        public bool Supports(Device device) => device.Backend is IKernelProfiler;

        public Action Begin(ProfileRun run)
        {
            var profiler = (IKernelProfiler)run.Device.Backend;                   // Supports said it times its kernels
            var direct = ComputeGraph.RunStepsDirectly();                         // a graph's kernels are timed one by one too
            run.Device.Synchronize();
            profiler.StartProfile();
            return () =>
            {
                run.Device.Synchronize();
                direct.Dispose();
                var kernels = profiler.StopProfile()
                    .Select(e => (e.Key, e.Value.Calls, Ms: e.Value.Ticks * 1000.0 / Stopwatch.Frequency, e.Value.Flops))
                    .OrderByDescending(k => k.Ms).ToList();
                double total = kernels.Sum(k => k.Ms), productMs = kernels.Where(k => k.Flops > 0).Sum(k => k.Ms), flops = kernels.Sum(k => k.Flops);
                run.Record("kernel time", total, "ms");
                run.Record("launches", kernels.Sum(k => k.Calls));
                run.Record("matrix TFLOPS", productMs > 0 ? flops / (productMs * 1e9) : 0, "TFLOPS");
                run.Add(new ProfileTable("kernels", ["kernel", "calls", "ms", "share", "TFLOPS"],
                    [.. kernels.Select(k => (IReadOnlyList<string>)[k.Key, Text(k.Calls), Text(k.Ms), Text(total > 0 ? 100 * k.Ms / total : 0) + "%", k.Flops > 0 && k.Ms > 0 ? Text(k.Flops / (k.Ms * 1e9)) : ""])]));
            };
        }
    }

    // A number as the tables show it, the same on every machine.
    internal static string Text(double value) => value.ToString(Math.Abs(value) >= 100 ? "0" : "0.###", CultureInfo.InvariantCulture);
}
