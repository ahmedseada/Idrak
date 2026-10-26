// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;
using System.Text.Json;

namespace Idrak.Abstraction.Diagnostics;

/// <summary>How a progress reporter shows amounts.</summary>
public enum ProgressUnit
{
    /// <summary>Counted items ("1,234").</summary>
    Items,

    /// <summary>Bytes ("12.3 MB").</summary>
    Bytes,

    /// <summary>No amounts: the time so far, shown while the work runs (loading a model).</summary>
    Elapsed,
}

/// <summary>
/// The progress of one piece of long work (a download, a load, a training run, a benchmark): what is being done
/// (<see cref="Label"/>), how far (<see cref="Done"/> of <see cref="Total"/> when known, counted in
/// <see cref="Unit"/>), and from those its speed and the time left, shown as the reporter shows them (a line redrawn in
/// place, a line at intervals, JSON records, or nothing). Made by <see cref="ProgressReporters.Start"/> with the reporter
/// an option or <c>IDRAK_PROGRESS</c> names. <see cref="IProgress{T}.Report"/> and <see cref="Advance"/> may be called as
/// often as convenient, from any thread: a reporter shows the state at most at its interval, and allocates nothing when
/// it does not show it.
/// </summary>
public interface IProgressReporter : IProgress<long>, IDisposable
{
    /// <summary>What is in progress ("pull model.gguf", "epoch 3/40").</summary>
    string Label { get; set; }

    /// <summary>The amount when known; null shows the amount and rate without a share or time left.</summary>
    long? Total { get; set; }

    /// <summary>What the amounts are (a stage may change it: bytes read, then items made).</summary>
    ProgressUnit Unit { get; set; }

    /// <summary>The amount done so far.</summary>
    long Done { get; }

    /// <summary>Whether the reporter shows anything (false for the quiet one).</summary>
    bool Enabled { get; }

    /// <summary>
    /// More text shown after the amounts (a loss, the memory in use), asked for only when the state is shown, so it costs
    /// nothing between showings; null for none.
    /// </summary>
    Func<string>? Detail { get; set; }

    /// <summary>Adds <paramref name="amount"/> to the amount done.</summary>
    void Advance(long amount = 1);

    /// <summary>Shows the current state now, whatever the time since it was last shown.</summary>
    void Show();

    /// <summary>
    /// Ends the current stage and starts the next, with its own label, amount, rate and time left (the amount and the clock
    /// start again). With <paramref name="keepLine"/>, the stage's last state (or <paramref name="message"/> in its place)
    /// stays shown; without, the next stage takes its place.
    /// </summary>
    void NextStage(string label, long? total, ProgressUnit unit, string? message = null, bool keepLine = true);

    /// <summary>
    /// Takes a state shown in place off the screen for a moment, so a line of other output can be written; the next report
    /// shows it again below that output.
    /// </summary>
    void Erase();

    /// <summary>Shows the final state (or <paramref name="message"/> in its place) and ends the reporting.</summary>
    void Finish(string? message = null);

    /// <summary>Ends the reporting leaving nothing shown in place, for work whose result is reported elsewhere.</summary>
    void Clear();
}

/// <summary>What a progress reporter is made for (see <see cref="ProgressReporters.Register"/>).</summary>
/// <param name="Label">What is in progress.</param>
/// <param name="Total">The amount when known.</param>
/// <param name="Unit">What the amounts are.</param>
/// <param name="Output">Where it writes.</param>
/// <param name="Interval">The least time between two showings, or null for the reporter's own.</param>
public sealed record ProgressRequest(string Label, long? Total, ProgressUnit Unit, TextWriter Output, TimeSpan? Interval);

/// <summary>Which progress reporter <see cref="ProgressReporters.Start"/> makes, where it writes and how often.</summary>
public sealed record ProgressOptions
{
    /// <summary>
    /// The reporter, by name (<see cref="ProgressReporters"/>); null for <c>IDRAK_PROGRESS</c> when set, else chosen by the
    /// output (see <see cref="ProgressReporters.Choose"/>).
    /// </summary>
    public string? Reporter { get; init; }

    /// <summary>Where the reporter writes; null for the error output (the standard output stays clean for pipes).</summary>
    public TextWriter? Output { get; init; }

    /// <summary>
    /// The least time between two showings; null for the reporter's own (a tenth of a second in place, a minute for plain
    /// lines, a second for JSON records).
    /// </summary>
    public TimeSpan? Interval { get; init; }
}

/// <summary>
/// A kind of progress reporter, registered by its <see cref="IKind.Name"/> in <see cref="ProgressReporters"/>: makes the
/// reporter of one piece of work from a <see cref="ProgressRequest"/> (<see cref="IKind{TRequest, T}.Create"/>).
/// </summary>
public interface IProgressReporterKind : IKind<ProgressRequest, IProgressReporter>;

/// <summary>
/// The progress reporters, by name (ignoring case). The library's:
/// <list type="bullet">
/// <item>"console": one line redrawn in place ("label  45%  12.3 MB / 27.0 MB  3.1 MB/s  4s left"), at most ten times a
/// second, cut to the terminal's width; the time so far redrawn twice a second for <see cref="ProgressUnit.Elapsed"/>.</item>
/// <item>"plain": the same text as a new line at intervals (a minute by default), for output that goes to a file.</item>
/// <item>"json": one JSON object a line at intervals (a second by default) and one when a stage ends, for tools.</item>
/// <item>"quiet": nothing; counts only, at the cost of a field write.</item>
/// </list>
/// <see cref="Start"/> picks one by <see cref="ProgressOptions.Reporter"/>, else <c>IDRAK_PROGRESS</c>, else by the output:
/// "console" for the console's own stream on a terminal, "plain" for it redirected, "quiet" for any other writer. Register
/// another with <see cref="Register"/>; one under a library name shadows the library's, which <see cref="Unregister"/>
/// brings back. The registry has no guard: a reporter holds what it has shown, so it cannot be swapped half-way.
/// </summary>
public static class ProgressReporters
{
    /// <summary>The line redrawn in place.</summary>
    public const string ConsoleReporter = "console";

    /// <summary>A plain line at intervals.</summary>
    public const string PlainReporter = "plain";

    /// <summary>JSON records at intervals.</summary>
    public const string JsonReporter = "json";

    /// <summary>Nothing shown.</summary>
    public const string QuietReporter = "quiet";

    /// <summary>The environment variable that names the reporter when the options do not.</summary>
    public const string EnvironmentVariable = "IDRAK_PROGRESS";

    private static readonly KindTable<IProgressReporterKind> Table = new(nameof(ProgressReporters), "Progress reporter", "add it with ProgressReporters.Register",
        comparer: StringComparer.OrdinalIgnoreCase, unguarded: "a reporter holds what it has shown from its start to its end, so it cannot be swapped half-way");

    static ProgressReporters() => Overrides.AsLibraryDefaults(() =>
    {
        Register(new BuiltInKind(ConsoleReporter, "one line redrawn in place on the output: what, how far, speed, time left (at most 10 times a second)",
            request => new LineProgressReporter(request, inPlace: true)));
        Register(new BuiltInKind(PlainReporter, "the same text as a new line at intervals (a minute by default), for redirected output",
            request => new LineProgressReporter(request, inPlace: false)));
        Register(new BuiltInKind(JsonReporter, "one JSON object a line at intervals (a second by default) and at each stage's end", request => new JsonProgressReporter(request)));
        Register(new BuiltInKind(QuietReporter, "nothing shown", request => new QuietProgressReporter(request)));
    });

    /// <summary>
    /// Registers <paramref name="kind"/> under its <see cref="IKind.Name"/> (as options and <c>IDRAK_PROGRESS</c> give it),
    /// made for each piece of work (over the library's of that name, which stays behind it).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IProgressReporterKind kind) => Table.Register(kind, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's reporter <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The reporter kind registered as <paramref name="name"/>, or null.</summary>
    public static IProgressReporterKind? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The reporter kind registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotRegisteredException">None is: the message names the registered ones.</exception>
    public static IProgressReporterKind Get(ReadOnlySpan<char> name) => Table.Get(name);

    /// <summary>The library's reporter kind <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IProgressReporterKind? Default(string name) => Table.Default(name);

    /// <summary>Who registered the reporter <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each registered reporter as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    // A library reporter.
    private sealed class BuiltInKind(string name, string summary, Func<ProgressRequest, IProgressReporter> create) : IProgressReporterKind
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public IProgressReporter Create(ProgressRequest request) => create(request);
    }

    /// <summary>
    /// The reporter named by <c>IDRAK_PROGRESS</c>, or null when it is not set.
    /// </summary>
    /// <exception cref="ArgumentException">It names no registered reporter (the message names the variable and the registered ones).</exception>
    public static string? FromEnvironment() => FromValue(Environment.GetEnvironmentVariable(EnvironmentVariable));

    internal static string? FromValue(string? value)
    {
        if (string.IsNullOrWhiteSpace(value))
        {
            return null;
        }

        string name = value.Trim();
        return Table.Contains(name) ? name
            : throw new ArgumentException($"{EnvironmentVariable}={value} is not a registered progress reporter (registered: {string.Join(", ", Table.Names)}).");
    }

    /// <summary>
    /// The reporter <see cref="Start"/> makes for <paramref name="options"/>: <see cref="ProgressOptions.Reporter"/>, else
    /// <c>IDRAK_PROGRESS</c>, else by the output: "console" when it is the console's error or standard output and that
    /// goes to a terminal, "plain" when it is redirected or the platform has no console window (a phone's app), "quiet"
    /// for any other writer.
    /// </summary>
    public static string Choose(ProgressOptions? options = null)
    {
        if (options?.Reporter is { Length: > 0 } named)
        {
            return named;
        }

        if (FromEnvironment() is { } chosen)
        {
            return chosen;
        }

        var output = options?.Output ?? Console.Error;
        bool error = ReferenceEquals(output, Console.Error), standard = ReferenceEquals(output, Console.Out);
        return !error && !standard ? QuietReporter
            : !ConsoleWidth.Terminal.Supported || (error ? Console.IsErrorRedirected : Console.IsOutputRedirected) ? PlainReporter : ConsoleReporter;
    }

    /// <summary>Starts reporting the progress of <paramref name="label"/> with the reporter <see cref="Choose"/> picks.</summary>
    /// <param name="label">What is in progress.</param>
    /// <param name="total">The amount when known.</param>
    /// <param name="unit">What the amounts are.</param>
    /// <param name="options">The reporter, the output and the interval (see <see cref="ProgressOptions"/>).</param>
    /// <exception cref="NotSupportedException">The reporter named is not registered.</exception>
    public static IProgressReporter Start(string label, long? total = null, ProgressUnit unit = ProgressUnit.Items, ProgressOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(label);
        return Get(Choose(options)).Create(new ProgressRequest(label, total, unit, options?.Output ?? Console.Error, options?.Interval));
    }

    /// <summary>
    /// Takes every progress line the library's "console" reporter shows on <paramref name="writer"/> off the screen for a
    /// moment (<see cref="IProgressReporter.Erase"/>), so a line another part of the program writes does not land inside one.
    /// </summary>
    public static void EraseOn(TextWriter writer)
    {
        ArgumentNullException.ThrowIfNull(writer);
        LineProgressReporter.EraseOn(writer);
    }
}

/// <summary>Progress amounts as the library's reporters show them, the same on every machine.</summary>
public static class ProgressText
{
    /// <summary>
    /// A state as one line: "label  45%  12.3 MB / 27.0 MB  3.1 MB/s  4s left" (without a total: "label  12.3 MB  3.1
    /// MB/s"); slow items read "88 / 8,000  3.6 s each  7h 48m left"; <see cref="ProgressUnit.Elapsed"/> reads "label  3s".
    /// </summary>
    public static string Format(string label, long done, long? total, TimeSpan elapsed, ProgressUnit unit)
    {
        ArgumentNullException.ThrowIfNull(label);
        if (unit == ProgressUnit.Elapsed)
        {
            return $"{label}  {Duration(elapsed)}";
        }

        double seconds = elapsed.TotalSeconds;
        double rate = seconds > 0 ? done / seconds : 0;
        var line = new StringBuilder(label);
        if (total is long t && t > 0)
        {
            line.Append(CultureInfo.InvariantCulture, $"  {Math.Min(100, done * 100 / t),3}%  {Amount(done, unit)} / {Amount(t, unit)}");
        }
        else
        {
            line.Append("  ").Append(Amount(done, unit));
        }

        if (seconds >= 0.2 && rate > 0)
        {
            line.Append("  ").Append(Rate(rate, unit));
            if (total is long all && all > done)
            {
                line.Append("  ").Append(Duration(TimeSpan.FromSeconds((all - done) / rate))).Append(" left");
            }
        }

        return line.ToString();
    }

    /// <summary>
    /// A rate as the unit shows it: bytes as "3.1 MB/s"; items as "1,234/s", "2.6/s" under ten a second, and "3.6 s each"
    /// under one (a slow step would otherwise read "0/s").
    /// </summary>
    public static string Rate(double perSecond, ProgressUnit unit) => unit switch
    {
        ProgressUnit.Bytes => $"{Bytes((long)perSecond)}/s",
        _ when perSecond < 1 => string.Create(CultureInfo.InvariantCulture, $"{1 / perSecond:0.0} s each"),
        _ when perSecond < 10 => string.Create(CultureInfo.InvariantCulture, $"{perSecond:0.0}/s"),
        _ => $"{Amount((long)perSecond, unit)}/s",
    };

    /// <summary>An amount as the unit shows it (bytes as "12.3 MB", items as "1,234").</summary>
    public static string Amount(long value, ProgressUnit unit) => unit switch
    {
        ProgressUnit.Bytes => Bytes(value),
        _ => value.ToString("N0", CultureInfo.InvariantCulture),
    };

    /// <summary>A byte count: "512 B", "3.4 KB", "12.3 MB", "1.20 GB" (powers of 1024).</summary>
    public static string Bytes(long bytes) => bytes switch
    {
        < 1024 => string.Create(CultureInfo.InvariantCulture, $"{bytes} B"),
        < 1 << 20 => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1024.0:0.#} KB"),
        < 1 << 30 => string.Create(CultureInfo.InvariantCulture, $"{bytes / (double)(1 << 20):0.#} MB"),
        _ => string.Create(CultureInfo.InvariantCulture, $"{bytes / (double)(1 << 30):0.00} GB"),
    };

    /// <summary>A duration: "4s", "3m 05s" or "1h 02m".</summary>
    public static string Duration(TimeSpan time) =>
        time.TotalHours >= 1 ? string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalHours}h {time.Minutes:D2}m")
        : time.TotalMinutes >= 1 ? string.Create(CultureInfo.InvariantCulture, $"{(int)time.TotalMinutes}m {time.Seconds:D2}s")
        : string.Create(CultureInfo.InvariantCulture, $"{Math.Max(0, (int)Math.Ceiling(time.TotalSeconds))}s");
}

// "console" (a line redrawn in place) and "plain" (a new line at intervals): the label, the share done, the amount, the
// rate and the time left. A report only stores the amount and compares the clock with the last showing; the text is made
// when it is shown.
internal sealed class LineProgressReporter : IProgressReporter
{
    private static readonly ConcurrentDictionary<LineProgressReporter, byte> Drawn = new();   // the in-place lines on a screen now
    private readonly TextWriter _writer;
    private readonly bool _inPlace;
    private readonly TimeSpan _interval;
    private readonly Stopwatch _watch = Stopwatch.StartNew();
    private TimeSpan _lastDraw;
    private bool _drawn;
    private int _lastWidth;
    private long _done;
    private volatile bool _finished;
    private readonly Timer? _clock;
    private readonly ConsoleWidth _width;

    public LineProgressReporter(ProgressRequest request, bool inPlace, ConsoleWidth? width = null)
    {
        ArgumentNullException.ThrowIfNull(request);
        _width = width ?? ConsoleWidth.Terminal;
        Label = request.Label;
        Total = request.Total;
        Unit = request.Unit;
        _writer = request.Output;
        _inPlace = inPlace;
        _interval = request.Interval ?? (inPlace ? TimeSpan.FromMilliseconds(100) : TimeSpan.FromMinutes(1));
        if (inPlace)
        {
            Drawn[this] = 0;
        }

        if (request.Unit == ProgressUnit.Elapsed)
        {
            // Work without amounts (loading a model): the time so far, redrawn twice a second (at each interval for lines).
            _clock = new Timer(_ => Draw(force: true), null, TimeSpan.Zero, inPlace ? TimeSpan.FromMilliseconds(500) : _interval);
        }
    }

    public string Label { get; set; }

    public long? Total { get; set; }

    public ProgressUnit Unit { get; set; }

    public bool Enabled => true;

    public long Done => Interlocked.Read(ref _done);

    public Func<string>? Detail { get; set; }

    public void Report(long value)
    {
        Interlocked.Exchange(ref _done, value);
        Draw(force: false);
    }

    public void Advance(long amount = 1)
    {
        Interlocked.Add(ref _done, amount);
        Draw(force: false);
    }

    public void Show() => Draw(force: true);

    public void NextStage(string label, long? total, ProgressUnit unit, string? message = null, bool keepLine = true)
    {
        ArgumentNullException.ThrowIfNull(label);
        if (!_finished)
        {
            lock (_writer)
            {
                if (keepLine)
                {
                    string text = message ?? Text(_watch.Elapsed);
                    _writer.Write(_inPlace ? "\r" + text.PadRight(_lastWidth) + "\n" : text + "\n");
                }
                else if (_inPlace)
                {
                    _writer.Write("\r" + new string(' ', _lastWidth) + "\r");      // the next stage takes the line's place
                }

                _writer.Flush();
                _lastWidth = 0;
                _drawn = false;
            }
        }

        Label = label;
        Total = total;
        Unit = unit;
        Interlocked.Exchange(ref _done, 0);
        _watch.Restart();
    }

    public void Finish(string? message = null)
    {
        if (!End())
        {
            return;
        }

        lock (_writer)
        {
            string text = message ?? Text(_watch.Elapsed);
            _writer.Write(_inPlace ? "\r" + text.PadRight(_lastWidth) + "\n" : text + "\n");
            _writer.Flush();
        }
    }

    public void Clear()
    {
        if (!End() || !_inPlace)
        {
            return;
        }

        lock (_writer)
        {
            _writer.Write("\r" + new string(' ', _lastWidth) + "\r");
            _writer.Flush();
        }
    }

    public void Erase()
    {
        if (_finished)
        {
            return;
        }

        lock (_writer)
        {
            if (_inPlace)
            {
                _writer.Write("\r" + new string(' ', _lastWidth) + "\r");
                _writer.Flush();
                _lastWidth = 0;
            }

            _drawn = false;
        }
    }

    public void Dispose() => Finish();

    internal static void EraseOn(TextWriter writer)
    {
        foreach (var line in Drawn.Keys)
        {
            if (ReferenceEquals(line._writer, writer))
            {
                line.Erase();
            }
        }
    }

    // Marks the reporting ended; false when it had ended already.
    private bool End()
    {
        if (_finished)
        {
            return false;
        }

        _finished = true;
        Drawn.TryRemove(this, out _);
        _clock?.Dispose();
        return true;
    }

    private string Text(TimeSpan elapsed)
    {
        string text = ProgressText.Format(Label, Done, Total, elapsed, Unit);
        return Detail?.Invoke() is { Length: > 0 } more ? text + "  " + more : text;
    }

    private void Draw(bool force)
    {
        if (_finished)
        {
            return;
        }

        var now = _watch.Elapsed;
        if (!force && _drawn && now - _lastDraw < _interval)
        {
            return;
        }

        lock (_writer)
        {
            if (_finished)
            {
                return;                                                          // finished while the clock was drawing
            }

            _lastDraw = now;
            _drawn = true;
            string text = Text(now);
            if (!_inPlace)
            {
                _writer.Write(text + "\n");
                _writer.Flush();
                return;
            }

            int width = _width.Read();
            if (width > 0 && text.Length >= width)
            {
                text = text[..(width - 1)];                                      // one line: a wrapped one cannot be redrawn in place
            }

            _writer.Write("\r" + text.PadRight(_lastWidth));
            _lastWidth = text.Length;
            _writer.Flush();
        }
    }
}

// "json": one object a line, {"label", "done", "total", "unit", "elapsed" (s), "rate" (a second), "left" (s), "detail"},
// at intervals and when a stage ends ("end": true, with "message" when given).
internal sealed class JsonProgressReporter : IProgressReporter
{
    private readonly TextWriter _writer;
    private readonly TimeSpan _interval;
    private readonly Stopwatch _watch = Stopwatch.StartNew();
    private TimeSpan _lastWrite;
    private bool _written;
    private long _done;
    private volatile bool _finished;
    private readonly Timer? _clock;

    public JsonProgressReporter(ProgressRequest request)
    {
        ArgumentNullException.ThrowIfNull(request);
        Label = request.Label;
        Total = request.Total;
        Unit = request.Unit;
        _writer = request.Output;
        _interval = request.Interval ?? TimeSpan.FromSeconds(1);
        if (request.Unit == ProgressUnit.Elapsed)
        {
            _clock = new Timer(_ => Write(force: true, end: false, null), null, TimeSpan.Zero, _interval);
        }
    }

    public string Label { get; set; }

    public long? Total { get; set; }

    public ProgressUnit Unit { get; set; }

    public bool Enabled => true;

    public long Done => Interlocked.Read(ref _done);

    public Func<string>? Detail { get; set; }

    public void Report(long value)
    {
        Interlocked.Exchange(ref _done, value);
        Write(force: false, end: false, null);
    }

    public void Advance(long amount = 1)
    {
        Interlocked.Add(ref _done, amount);
        Write(force: false, end: false, null);
    }

    public void Show() => Write(force: true, end: false, null);

    public void NextStage(string label, long? total, ProgressUnit unit, string? message = null, bool keepLine = true)
    {
        ArgumentNullException.ThrowIfNull(label);
        Write(force: true, end: true, message);
        Label = label;
        Total = total;
        Unit = unit;
        Interlocked.Exchange(ref _done, 0);
        lock (_writer)
        {
            _written = false;
            _watch.Restart();
        }
    }

    public void Erase()
    {
    }

    public void Finish(string? message = null)
    {
        Write(force: true, end: true, message, finish: true);
        _clock?.Dispose();
    }

    public void Clear() => Finish();

    public void Dispose() => Finish();

    private void Write(bool force, bool end, string? message, bool finish = false)
    {
        if (_finished)
        {
            return;
        }

        var now = _watch.Elapsed;
        if (!force && _written && now - _lastWrite < _interval)
        {
            return;
        }

        lock (_writer)
        {
            if (_finished)
            {
                return;
            }

            _lastWrite = now;
            _written = true;
            long done = Done;
            double seconds = now.TotalSeconds, rate = seconds > 0 ? done / seconds : 0;
            var buffer = new ArrayBufferWriter<byte>(256);
            using (var w = new Utf8JsonWriter(buffer))
            {
                w.WriteStartObject();
                w.WriteString("label", Label);
                w.WriteString("unit", Unit switch { ProgressUnit.Bytes => "bytes", ProgressUnit.Elapsed => "elapsed", _ => "items" });
                w.WriteNumber("done", done);
                if (Total is { } total)
                {
                    w.WriteNumber("total", total);
                }

                FiniteJsonNumbers.WriteNumber(w, "elapsed", Math.Round(seconds, 3));
                if (Unit != ProgressUnit.Elapsed)
                {
                    FiniteJsonNumbers.WriteNumber(w, "rate", rate);
                    if (Total is { } all && all > done && rate > 0)
                    {
                        FiniteJsonNumbers.WriteNumber(w, "left", Math.Round((all - done) / rate, 3));
                    }
                }

                if (Detail?.Invoke() is { Length: > 0 } more)
                {
                    w.WriteString("detail", more);
                }

                if (end)
                {
                    w.WriteBoolean("end", true);
                    if (message is not null)
                    {
                        w.WriteString("message", message);
                    }
                }

                w.WriteEndObject();
            }

            _writer.Write(Encoding.UTF8.GetString(buffer.WrittenSpan) + "\n");
            _writer.Flush();
            _finished |= finish;                                                 // under the lock: nothing is written after the end
        }
    }
}

// "quiet": counts only; nothing is shown and nothing is allocated.
internal sealed class QuietProgressReporter(ProgressRequest request) : IProgressReporter
{
    private long _done;

    public string Label { get; set; } = request.Label;

    public long? Total { get; set; } = request.Total;

    public ProgressUnit Unit { get; set; } = request.Unit;

    public bool Enabled => false;

    public long Done => Interlocked.Read(ref _done);

    public Func<string>? Detail { get; set; }

    public void Report(long value) => Interlocked.Exchange(ref _done, value);

    public void Advance(long amount = 1) => Interlocked.Add(ref _done, amount);

    public void Show()
    {
    }

    public void NextStage(string label, long? total, ProgressUnit unit, string? message = null, bool keepLine = true)
    {
        (Label, Total, Unit) = (label, total, unit);
        Interlocked.Exchange(ref _done, 0);
    }

    public void Erase()
    {
    }

    public void Finish(string? message = null)
    {
    }

    public void Clear()
    {
    }

    public void Dispose()
    {
    }
}

// The console window's width in columns, or 0 where there is none to ask: output to a file or a pipe, or a platform
// without a console window (a phone's or a browser's app, where reading it throws PlatformNotSupportedException). Never
// throws. A platform that said it has no window is not asked again, so a line drawn ten times a second costs no
// exception a draw; an I/O error (no console attached at that moment) is asked again at the next draw, as before.
internal sealed class ConsoleWidth(Func<int> read)
{
    /// <summary>The process's console: 0 when both its outputs are redirected, else its window's width.</summary>
    public static ConsoleWidth Terminal { get; } = new(() => Console.IsErrorRedirected && Console.IsOutputRedirected ? 0 : Console.WindowWidth);

    private volatile bool _unsupported, _asked;

    /// <summary>Whether the platform has a console window to ask (false once it said it has none).</summary>
    public bool Supported
    {
        get
        {
            if (!_asked)
            {
                _ = Read();                                                     // asks the first time only
                _asked = true;
            }

            return !_unsupported;
        }
    }

    /// <summary>The columns, or 0 where there is no window to ask.</summary>
    public int Read()
    {
        if (_unsupported)
        {
            return 0;
        }

        try
        {
            return Math.Max(0, read());
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException)   // PlatformNotSupportedException is one
        {
            _unsupported = true;
            return 0;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or System.Security.SecurityException)
        {
            return 0;
        }
    }
}
