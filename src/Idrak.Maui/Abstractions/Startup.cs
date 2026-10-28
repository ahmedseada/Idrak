// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Maui.Abstractions;

/// <summary>Where a start-up step runs.</summary>
public enum StartupThread
{
    /// <summary>On a worker thread, after the worker steps before it (the CPU read, then the GPU's start).</summary>
    Worker = 0,

    /// <summary>On the UI thread, after the UI steps before it, alongside the worker steps (the tabs made ahead).</summary>
    Ui = 1,
}

/// <summary>How far a start-up step is.</summary>
public enum StartupStepState
{
    /// <summary>Not started.</summary>
    Waiting = 0,

    /// <summary>Running now.</summary>
    Running = 1,

    /// <summary>Done.</summary>
    Done = 2,

    /// <summary>Ended with an error (<see cref="StartupRun.Error"/>); the steps after it still run.</summary>
    Failed = 3,
}

/// <summary>What a start-up step is given: the app's services and the page the window shows (the host page).</summary>
/// <param name="Services">The app's services.</param>
/// <param name="Host">The window's page that started the run (<c>AppHostPage</c>), or null.</param>
public sealed record StartupContext(IServiceProvider Services, Page? Host);

/// <summary>
/// One piece of the app's real start-up work, run once when the window first shows (<see cref="StartupRun.RunAsync"/>) and
/// shown by the splash while it runs: its <see cref="Title"/> is the splash's line. Registered by its
/// <see cref="IKind.Name"/> in <see cref="StartupSteps"/>; the library's read the CPU, start the GPU and make the tabs ahead,
/// and an app adds its own or replaces one by name.
/// </summary>
public interface IStartupStep : IKind
{
    /// <summary>The splash's line while it runs ("Reading the CPU").</summary>
    string Title { get; }

    /// <summary>Its place among the steps: lower first.</summary>
    int Order { get; }

    /// <summary>Where it runs.</summary>
    StartupThread Thread { get; }

    /// <summary>Whether it has work to do in <paramref name="context"/> (false leaves it out of the run: no line, no share of the bar).</summary>
    bool IsWanted(StartupContext context);

    /// <summary>Does the work; the run marks it done when the task ends (failed when it throws).</summary>
    Task RunAsync(StartupContext context, CancellationToken cancellationToken);
}

/// <summary>
/// The start-up steps (a <see cref="SlotTable{TKey, TValue}"/>), by name ignoring case: "cpu" (the CPU described), "gpu"
/// (every GPU backend started, and Home's device summary and model list read) and "sections" (the tabs' views made ahead,
/// when <see cref="IdrakMauiOptions.PrebuildSections"/> is on) are the library's. <see cref="All"/> lists them in order.
/// </summary>
public static class StartupSteps
{
    /// <summary>The CPU described (its facts, its memory).</summary>
    public const string Cpu = "cpu";

    /// <summary>Every GPU backend started and listed, with Home's summary of the device and the store's models.</summary>
    public const string Gpu = "gpu";

    /// <summary>The tabs' views made ahead, one a turn of the UI thread.</summary>
    public const string Sections = "sections";

    private static readonly SlotTable<string, IStartupStep> Table = BuiltIn();

    private static SlotTable<string, IStartupStep> BuiltIn()
    {
        var table = new SlotTable<string, IStartupStep>(nameof(StartupSteps), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a start-up step runs once, while the app starts, with no call to fall back at");
        foreach (var step in Idrak.Maui.LibraryStartupSteps.All())
        {
            table.RegisterDefault(step.Name, step);
        }

        return table;
    }

    /// <summary>Registers <paramref name="step"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IStartupStep step)
    {
        ArgumentNullException.ThrowIfNull(step);
        ArgumentException.ThrowIfNullOrWhiteSpace(step.Name);
        Table.Register(step.Name, step, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's step of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in registration order.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Every registered step in order (<see cref="IStartupStep.Order"/>, then the name).</summary>
    public static IReadOnlyList<IStartupStep> All => [.. Table.Values.OrderBy(s => s.Order).ThenBy(s => s.Name, StringComparer.OrdinalIgnoreCase)];

    /// <summary>The step of that name, or null.</summary>
    public static IStartupStep? Find(string name) => Table.Find(name);

    /// <summary>The step of that name.</summary>
    /// <exception cref="NotRegisteredException">None is registered under that name (the message names those that are).</exception>
    public static IStartupStep Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(StartupSteps), name, Table.Keys,
            $"Start-up step '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with StartupSteps.Register.");

    /// <summary>Who registered the step of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);
}

/// <summary>
/// The app's start-up as it runs, one for the app: the work done before the window (<see cref="Record"/>: the saved settings
/// read, the options applied), then the registered steps (<see cref="RunAsync"/>, once, when the window first shows), each
/// with its state and time. The splash reads it once a frame without a lock or an allocation: <see cref="Count"/> entries,
/// <see cref="Started"/> of them begun (in the order they began, <see cref="StartedEntry"/>), <see cref="Finished"/> ended,
/// and <see cref="IsDone"/> once nothing more comes. A host's own work can be reported by hand with <see cref="Begin"/> and
/// <see cref="End"/>, and <see cref="Complete"/> says nothing more comes where no <see cref="RunAsync"/> runs (its end says it).
/// </summary>
public sealed class StartupRun
{
    private readonly Lock _gate = new();
    private readonly TimeProvider _time;
    private Entry[] _entries = [];
    private int[] _order = [];
    private int _count;
    private int _started;
    private int _finished;
    private bool _closed;                                                       // nothing more comes (Complete, or the steps' run ended)
    private bool _done;
    private Task? _running;
    private readonly TaskCompletionSource _completion = new(TaskCreationOptions.RunContinuationsAsynchronously);

    /// <summary>A run timed by <paramref name="time"/> (the system's clock by default).</summary>
    public StartupRun(TimeProvider? time = null) => _time = time ?? TimeProvider.System;

    /// <summary>How many entries it has (done before the window, planned and begun by hand).</summary>
    public int Count => Volatile.Read(ref _count);

    /// <summary>How many entries have begun.</summary>
    public int Started => Volatile.Read(ref _started);

    /// <summary>How many entries have ended (done or failed).</summary>
    public int Finished => Volatile.Read(ref _finished);

    /// <summary>Whether the start-up is over: every entry ended and nothing more comes (<see cref="Complete"/>, or the steps' run ended).</summary>
    public bool IsDone => Volatile.Read(ref _done);

    /// <summary>Ends when <see cref="IsDone"/> turns true.</summary>
    public Task Completion => _completion.Task;

    /// <summary>The entry that began <paramref name="position"/>-th (0 first), an index for <see cref="Title"/> and the rest.</summary>
    public int StartedEntry(int position) => Volatile.Read(ref _order)[position];

    /// <summary>The entry's name ("cpu").</summary>
    public string Name(int entry) => Volatile.Read(ref _entries)[entry].Name;

    /// <summary>The entry's line ("Reading the CPU").</summary>
    public string Title(int entry) => Volatile.Read(ref _entries)[entry].Title;

    /// <summary>How far the entry is.</summary>
    public StartupStepState State(int entry) => (StartupStepState)Volatile.Read(ref Volatile.Read(ref _entries)[entry].State);

    /// <summary>How long the entry took (so far, while it runs; zero before it began).</summary>
    public TimeSpan Took(int entry)
    {
        var e = Volatile.Read(ref _entries)[entry];
        long began = Volatile.Read(ref e.Began), ended = Volatile.Read(ref e.Ended);
        return began == 0 ? TimeSpan.Zero : _time.GetElapsedTime(began, ended == 0 ? _time.GetTimestamp() : ended);
    }

    /// <summary>Why the entry failed, or null.</summary>
    public string? Error(int entry) => Volatile.Read(ref _entries)[entry].Error;

    /// <summary>Adds work that was done before the window (the saved settings read, the options applied): begun and ended, taking <paramref name="took"/>.</summary>
    public void Record(string name, string title, TimeSpan took)
    {
        long now = _time.GetTimestamp();
        int entry = Add(name, title);
        var e = _entries[entry];
        e.Began = Math.Max(1, now - (long)(took.TotalSeconds * _time.TimestampFrequency));
        lock (_gate)
        {
            Start(entry);
        }

        Finish(entry, null, now);
    }

    /// <summary>Begins an entry by hand (a host's own start-up work, before <see cref="Complete"/>); returns it for <see cref="End"/>.</summary>
    /// <exception cref="InvalidOperationException">The run is over already.</exception>
    public int Begin(string name, string title)
    {
        int entry = Add(name, title);
        _entries[entry].Began = _time.GetTimestamp();
        lock (_gate)
        {
            Start(entry);
        }

        return entry;
    }

    /// <summary>Ends an entry begun by <see cref="Begin"/> (failed, with <paramref name="error"/>'s first line, when given).</summary>
    public void End(int entry, Exception? error = null) => Finish(entry, error, _time.GetTimestamp());

    /// <summary>Says nothing more comes: <see cref="IsDone"/> turns true once every entry has ended.</summary>
    public void Complete()
    {
        lock (_gate)
        {
            _closed = true;
        }

        CheckDone();
    }

    /// <summary>
    /// Runs <paramref name="steps"/> (the registered ones, <see cref="StartupSteps.All"/>, by default) that are wanted in
    /// <paramref name="context"/>, once: they are all planned first (so the bar knows its length), then the worker steps run
    /// one after another on a worker thread while the UI steps run one after another on the calling (UI) thread; a step that
    /// throws is marked failed and the rest still run. The run is complete (<see cref="IsDone"/>) when both have ended (and
    /// whatever was begun by hand meanwhile). A second call returns the first call's task.
    /// </summary>
    public Task RunAsync(StartupContext context, IReadOnlyList<IStartupStep>? steps = null, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(context);
        lock (_gate)
        {
            if (_running is not null)
            {
                return _running;
            }

            _running = Task.CompletedTask;
        }

        var wanted = new List<(IStartupStep Step, int Entry)>();
        foreach (var step in steps ?? StartupSteps.All)
        {
            bool want;
            try
            {
                want = step.IsWanted(context);
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                want = false;                                                    // a step that cannot say has nothing to do
            }

            if (want)
            {
                wanted.Add((step, Add(step.Name, step.Title)));
            }
        }

        var worker = wanted.Where(w => w.Step.Thread == StartupThread.Worker).ToArray();
        var ui = wanted.Where(w => w.Step.Thread == StartupThread.Ui).ToArray();
        var onWorker = worker.Length == 0 ? Task.CompletedTask : Task.Run(() => RunAll(worker, context, cancellationToken, onUi: false));
        var onUi = ui.Length == 0 ? Task.CompletedTask : RunAll(ui, context, cancellationToken, onUi: true);
        var all = Ended(onWorker, onUi);
        lock (_gate)
        {
            _running = all;
        }

        return all;
    }

    private async Task Ended(Task worker, Task ui)
    {
        try
        {
            await Task.WhenAll(worker, ui).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Cancelled: what began has ended, and the run is over.
        }
        finally
        {
            Complete();
        }
    }

    private async Task RunAll((IStartupStep Step, int Entry)[] steps, StartupContext context, CancellationToken cancellationToken, bool onUi)
    {
        foreach (var (step, entry) in steps)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                Finish(entry, new OperationCanceledException(), _time.GetTimestamp());
                continue;
            }

            _entries[entry].Began = _time.GetTimestamp();
            lock (_gate)
            {
                Start(entry);
            }

            try
            {
                await step.RunAsync(context, cancellationToken).ConfigureAwait(onUi);
                Finish(entry, null, _time.GetTimestamp());
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                Finish(entry, e, _time.GetTimestamp());
            }
        }
    }

    private int Add(string name, string title)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        lock (_gate)
        {
            if (_done)
            {
                throw new InvalidOperationException($"The start-up run is over; '{name}' cannot be added to it.");
            }

            var entries = new Entry[_count + 1];
            Array.Copy(_entries, entries, _count);
            entries[_count] = new Entry(name, title ?? name);
            Volatile.Write(ref _entries, entries);                              // the entry, then its count: a reader sees both
            Volatile.Write(ref _count, _count + 1);
            return _count - 1;
        }
    }

    // Called under the gate: the entry is the next to begin.
    private void Start(int entry)
    {
        var e = _entries[entry];
        if (e.State != (int)StartupStepState.Waiting)
        {
            return;
        }

        var order = _order.Length > _started ? _order : Grow(_order, _entries.Length);
        order[_started] = entry;
        Volatile.Write(ref _order, order);
        Volatile.Write(ref e.State, (int)StartupStepState.Running);
        Volatile.Write(ref _started, _started + 1);
    }

    private static int[] Grow(int[] order, int least)
    {
        var grown = new int[Math.Max(least, Math.Max(4, order.Length * 2))];
        Array.Copy(order, grown, order.Length);
        return grown;
    }

    private void Finish(int entry, Exception? error, long now)
    {
        lock (_gate)
        {
            var e = _entries[entry];
            if (e.State is (int)StartupStepState.Done or (int)StartupStepState.Failed)
            {
                return;
            }

            if (e.State == (int)StartupStepState.Waiting)
            {
                e.Began = now;
                Start(entry);
            }

            e.Error = error is null ? null : error.Message.Split('\n')[0];
            Volatile.Write(ref e.Ended, Math.Max(1, now));
            Volatile.Write(ref e.State, (int)(error is null ? StartupStepState.Done : StartupStepState.Failed));
            Volatile.Write(ref _finished, _finished + 1);
        }

        CheckDone();
    }

    private void CheckDone()
    {
        bool now;
        lock (_gate)
        {
            now = !_done && _closed && _finished == _count;
            if (now)
            {
                Volatile.Write(ref _done, true);
            }
        }

        if (now)
        {
            _completion.TrySetResult();
        }
    }

    private sealed class Entry(string name, string title)
    {
        public readonly string Name = name;
        public readonly string Title = title;
        public int State;
        public long Began;
        public long Ended;
        public string? Error;
    }
}
