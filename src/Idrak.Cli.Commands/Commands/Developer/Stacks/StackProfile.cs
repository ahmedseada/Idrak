// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.RegularExpressions;

namespace Idrak.Cli.Commands.Developer.Stacks;

/// <summary>Where a thread's moment went.</summary>
internal enum TimeKind : byte
{
    /// <summary>Running .NET code.</summary>
    Running,

    /// <summary>Outside .NET code, not waiting: a native call (a driver, the system) or the runtime itself (compiling, loading types).</summary>
    Outside,

    /// <summary>Spinning: a spin-wait burning a core while it waits.</summary>
    Spinning,

    /// <summary>Stopped by the runtime: a garbage collection's suspension, or waiting at a GC poll for one to end.</summary>
    Stopped,

    /// <summary>Waiting: blocked on a lock, an event, a semaphore, a sleep, I/O, or an idle thread pool's wait for work.</summary>
    Waiting,
}

/// <summary>
/// A thread's moment: from its observation to its next, its kind and its stack's methods (leaf first: without a GC poll's
/// frames at the leaf, so the method that polled owns the time), and the stack's methods as the trace has them (<paramref name="Raw"/>).
/// </summary>
internal readonly record struct Moment(long Thread, long Time, long Length, TimeKind Kind, int[] Frames, int[] Raw);

/// <summary>
/// A trace's stacks as moments (<see cref="Moment"/>): each observation lasts until its thread's next one (the time weighting
/// Microsoft's TraceEvent gives thread time), classed by where it was: running, outside .NET code, spinning, stopped by the
/// runtime or waiting. The classes come from the stack's framework frames from the leaf (the wait, spin and GC-poll methods
/// of the .NET libraries) and the runtime's suspensions for a collection; the first frame of the traced program's own code
/// ends the search. The sample profiler stops the runtime for each of its rounds too, and a thread in a loop then waits at
/// its GC poll: outside a collection that is the polling method's running time (its poll frames dropped), not a stop.
/// </summary>
internal sealed partial class StackProfile
{
    // The .NET library methods a thread waits in (a prefix of the full name), spins in, or stops in for a collection.
    private static readonly string[] WaitMethods =
    [
        "System.Threading.LowLevelLifoSemaphore.", "System.Threading.LowLevelMonitor.", "System.Threading.LowLevelLock.", "System.Threading.WaitHandle.Wait",
        "System.Threading.WaitSubsystem.", "System.Threading.Monitor.Wait", "System.Threading.Monitor.Enter", "System.Threading.Monitor.ReliableEnter",
        "System.Threading.Lock.Enter", "System.Threading.Lock.TryEnterSlow", "System.Threading.Thread.Sleep", "System.Threading.Thread.Join",
        "System.Threading.SemaphoreSlim.Wait", "System.Threading.ManualResetEventSlim.Wait", "System.Threading.Tasks.Task.Wait",
        "System.Threading.Tasks.Task.InternalWait", "System.Threading.Tasks.Task.SpinThenBlockingWait", "System.Threading.PortableThreadPool+GateThread.",
        "System.Threading.PortableThreadPool+WorkerThread.WaitForWork", "System.Threading.TimerQueue.", "System.Threading.Thread.WaitOne",
        "Interop+Kernel32.GetQueuedCompletionStatus", "Interop+Kernel32.WaitFor", "Interop+Kernel32.SleepEx", "Interop+Sys.WaitFor", "Interop+Sys.LowLevelMonitor",
    ];

    private static readonly string[] SpinMethods = ["System.Threading.SpinWait.", "System.Threading.Thread.SpinWait", "System.Threading.Thread.Yield"];

    private static readonly string[] PollMethods = ["System.Threading.Thread.<PollGC>", "System.Threading.Thread.PollGC"];

    private readonly byte[] _role;                                                // per method: 0 the program's, 1 a library's, 2 wait, 3 spin, 4 GC poll

    private StackProfile(NetTrace trace)
    {
        Trace = trace;
        Names = [.. trace.Methods.Select(m => m.Type.Length == 0 ? m.Name : $"{m.Type}.{m.Name}"), ReturningName];
        Display = [.. trace.Methods.Select(m => m.Type.Length == 0 ? m.Name : DisplayName(m.Type, m.Name)), ReturningName];
        Returning = Names.Length - 1;
        _role = new byte[Names.Length];
        for (int i = 0; i < Names.Length; i++)
        {
            string n = Names[i];
            _role[i] = PollMethods.Any(p => n.StartsWith(p, StringComparison.Ordinal)) ? (byte)4
                : SpinMethods.Any(p => n.StartsWith(p, StringComparison.Ordinal)) ? (byte)3
                : WaitMethods.Any(p => n.StartsWith(p, StringComparison.Ordinal)) ? (byte)2
                : Library(n) || i == 0 || i == Returning ? (byte)1 : (byte)0;
        }
    }

    // A thread caught as a call returned (the runtime's return stub, then a GC poll, over the caller): the method that ran is
    // gone from the stack; its caller and theirs keep the total time.
    private const string ReturningName = "[a call returning: its method unknown]";

    /// <summary>The trace read.</summary>
    public NetTrace Trace { get; }

    /// <summary>The pseudo-method a moment caught as a call returned owns (its callers keep their total time).</summary>
    public int Returning { get; }

    /// <summary>Each method's full name (type with namespace, then name).</summary>
    public string[] Names { get; }

    /// <summary>Each method's name as the report prints it: no namespace, the compiler's lambdas, local functions and state machines named after their method.</summary>
    public string[] Display { get; }

    /// <summary>The moments in the window, each thread's in time order.</summary>
    public List<Moment> Moments { get; } = [];

    /// <summary>The allocation reports in the window, each with its stack's methods.</summary>
    public List<(AllocationReport Report, int[] Frames)> Allocations { get; } = [];

    /// <summary>The window's start and end, in ticks since the session started.</summary>
    public (long Start, long End) Window { get; private set; }

    /// <summary>The sample profiler's interval between rounds as the system gave it: the median and the slowest tenth's, in ticks.</summary>
    public (long Median, long Slow) Interval { get; private set; }

    /// <summary>The sample profiler's samples and the other events with stacks, in the window.</summary>
    public (int Samples, int Events) Counts { get; private set; }

    /// <summary>Whether a method is a .NET library's (System, Microsoft, the interop stubs) rather than the traced program's.</summary>
    public bool IsLibrary(int method) => _role[method] != 0;

    /// <summary>The profile of <paramref name="trace"/> between <paramref name="from"/> and <paramref name="to"/> seconds (null: the start, the end).</summary>
    public static StackProfile Of(NetTrace trace, double? from = null, double? to = null)
    {
        var profile = new StackProfile(trace);
        profile.Build(from, to);
        return profile;
    }

    private void Build(double? from, double? to)
    {
        var frames = new Dictionary<ulong[], int[]>(ReferenceEqualityComparer.Instance);
        var trimmed = new Dictionary<int[], int[]>(ReferenceEqualityComparer.Instance);
        int[] Trimmed(int[] raw)
        {
            if (raw.Length == 0 || _role[raw[0]] != 4)
            {
                return raw;
            }

            if (!trimmed.TryGetValue(raw, out var kept))
            {
                // The poll's frames, then the runtime's helper that called it (code with no name) above the polling method.
                int skip = 0;
                while (skip < raw.Length - 1 && _role[raw[skip]] == 4)
                {
                    skip++;
                }

                int polls = skip;
                while (skip < raw.Length - 1 && raw[skip] == 0)
                {
                    skip++;
                }

                // Through the return stub: the moment is the returning call's, whose method the stack no longer holds.
                trimmed[raw] = kept = skip > polls ? [Returning, .. raw[skip..]] : raw[skip..];
            }

            return kept;
        }

        var methods = new Dictionary<ulong, int>();
        int[] Frames(ulong[] stack)
        {
            if (!frames.TryGetValue(stack, out var resolved))
            {
                resolved = new int[stack.Length];
                for (int i = 0; i < stack.Length; i++)
                {
                    if (!methods.TryGetValue(stack[i], out int m))
                    {
                        methods[stack[i]] = m = Trace.MethodAt(stack[i]);
                    }

                    resolved[i] = m;
                }

                frames[stack] = resolved;
            }

            return resolved;
        }

        long first = long.MaxValue, last = long.MinValue;
        foreach (var o in Trace.Observations)
        {
            first = Math.Min(first, o.Time);
            last = Math.Max(last, o.Time);
        }

        if (first == long.MaxValue)
        {
            first = last = 0;
        }

        long start = from is { } f ? (long)(f * Trace.Frequency) : first, end = to is { } t ? (long)(t * Trace.Frequency) : last;
        Window = (start, Math.Max(start, end));
        var collections = Trace.Suspensions.Where(s => s.ForCollection).ToList();
        var gaps = new List<long>();
        int samples = 0, events = 0;
        foreach (var thread in Trace.Observations.GroupBy(o => o.Thread))
        {
            // A thread the sample profiler never saw (a runtime thread raising a few events) has no time to weigh: its
            // observations are far apart, and the time between them is not known to be its own (TraceEvent leaves it out too).
            var list = thread.ToList();
            if (list.TrueForAll(o => o.Kind == ObservationKind.Event))
            {
                continue;
            }

            list.Sort((a, b) => a.Time.CompareTo(b.Time));
            long lastSample = -1;
            for (int i = 0; i < list.Count; i++)
            {
                var o = list[i];
                if (o.Kind != ObservationKind.Event)
                {
                    if (lastSample >= 0)
                    {
                        gaps.Add(o.Time - lastSample);
                    }

                    lastSample = o.Time;
                }

                if (o.Time < start || o.Time > end)
                {
                    continue;
                }

                // Until the thread's next observation (its last lasts nothing), cut at the window's end.
                long next = i + 1 < list.Count ? list[i + 1].Time : o.Time;
                var resolved = Frames(o.Stack);
                Moments.Add(new Moment(o.Thread, o.Time, Math.Max(0, Math.Min(next, end) - o.Time), Classify(o, resolved, collections), Trimmed(resolved), resolved));
                if (o.Kind == ObservationKind.Event)
                {
                    events++;
                }
                else
                {
                    samples++;
                }
            }
        }

        gaps.Sort();
        Interval = gaps.Count == 0 ? (0, 0) : (gaps[gaps.Count / 2], gaps[(int)(gaps.Count * 0.9)]);
        Counts = (samples, events);
        foreach (var a in Trace.Allocations)
        {
            if (a.Time >= start && a.Time <= end)
            {
                Allocations.Add((a, Frames(a.Stack)));
            }
        }
    }

    private TimeKind Classify(Observation o, int[] frames, List<Suspension> collections)
    {
        // A collection stops the threads in .NET code (and those at a GC poll); a thread outside .NET code goes on.
        bool collecting = InSuspension(collections, o.Time);
        if (frames.Length > 0 && _role[frames[0]] == 4)
        {
            return collecting ? TimeKind.Stopped : TimeKind.Running;
        }

        if (collecting && o.Kind != ObservationKind.Outside)
        {
            return TimeKind.Stopped;
        }

        // From the leaf through the libraries' frames: the first wait or spin method says what the thread was doing.
        foreach (int m in frames)
        {
            switch (_role[m])
            {
                case 2 or 3:
                    return o.Kind == ObservationKind.Outside ? TimeKind.Waiting : TimeKind.Spinning;
                case 0:
                    return o.Kind == ObservationKind.Outside ? TimeKind.Outside : TimeKind.Running;
            }
        }

        return o.Kind == ObservationKind.Outside ? TimeKind.Outside : TimeKind.Running;
    }

    private static bool InSuspension(List<Suspension> suspensions, long time)
    {
        int lo = 0, hi = suspensions.Count - 1;
        while (lo <= hi)
        {
            int mid = (lo + hi) >>> 1;
            if (suspensions[mid].End < time)
            {
                lo = mid + 1;
            }
            else if (suspensions[mid].Start > time)
            {
                hi = mid - 1;
            }
            else
            {
                return true;
            }
        }

        return false;
    }

    // A namespace or an enclosing type before a type's name, inside generic arguments ("System.", "CpuBackend+").
    [GeneratedRegex(@"(?<=[\[,])(?:[A-Za-z_][\w`]*[.+])+(?=[A-Za-z_])")]
    private static partial Regex Namespaces();

    private static bool Library(string name) =>
        name.StartsWith("System.", StringComparison.Ordinal) || name.StartsWith("Microsoft.", StringComparison.Ordinal) || name.StartsWith("Interop", StringComparison.Ordinal)
        || name.StartsWith("ILStubClass", StringComparison.Ordinal) || name.StartsWith("Internal.", StringComparison.Ordinal);

    /// <summary>
    /// A method's printed name: the type without its namespace (the last dot outside brackets), the compiler's types dropped
    /// (+&lt;&gt;c, +&lt;&gt;c__DisplayClass: their generic arguments kept at the end, which tell one instantiation from
    /// another), a lambda &lt;M&gt;b__ as "M (lambda)", a local function &lt;M&gt;g__F| as "M.F", a state machine's MoveNext
    /// (+&lt;M&gt;d__) as "M (async)"; every generic argument without its namespace.
    /// </summary>
    internal static string DisplayName(string type, string method)
    {
        string arguments = "";
        int depth = 0, cut = -1;
        for (int i = 0; i < type.Length; i++)
        {
            char c = type[i];
            depth += c == '[' ? 1 : c == ']' ? -1 : 0;
            if (depth == 0 && c == '.')
            {
                cut = i;
            }
        }

        var parts = new List<string>();
        string suffix = "";
        foreach (string segment in Split(type[(cut + 1)..]))
        {
            if (segment.StartsWith("<>", StringComparison.Ordinal))
            {
                int open = segment.IndexOf('[', StringComparison.Ordinal);
                arguments = open > 0 ? " " + segment[open..] : arguments;          // the compiler's closure types: their generic arguments only
                continue;
            }

            int close = segment.IndexOf('>', StringComparison.Ordinal);
            if (segment.StartsWith('<') && close > 1 && segment.AsSpan(close + 1).StartsWith("d__", StringComparison.Ordinal) && method == "MoveNext")
            {
                method = segment[1..close];
                suffix = " (async)";
                continue;
            }

            parts.Add(segment);
        }

        if (method.StartsWith('<') && method.IndexOf('>', StringComparison.Ordinal) is var end and > 1)
        {
            string outer = method[1..end], rest = method[(end + 1)..];
            if (rest.StartsWith("b__", StringComparison.Ordinal))
            {
                method = outer;
                suffix = " (lambda)";
            }
            else if (rest.StartsWith("g__", StringComparison.Ordinal))
            {
                int bar = rest.IndexOf('|', StringComparison.Ordinal);
                method = $"{outer}.{(bar > 3 ? rest[3..bar] : rest[3..])}";
            }
        }

        var name = new StringBuilder();
        foreach (string p in parts)
        {
            name.Append(p).Append('.');
        }

        return Namespaces().Replace(name.Append(method).Append(suffix).Append(arguments).ToString(), "");

        // A type's nested parts (split at '+' outside brackets).
        static IEnumerable<string> Split(string type)
        {
            int depth = 0, from = 0;
            for (int i = 0; i < type.Length; i++)
            {
                depth += type[i] == '[' ? 1 : type[i] == ']' ? -1 : 0;
                if (depth == 0 && type[i] == '+')
                {
                    yield return type[from..i];
                    from = i + 1;
                }
            }

            yield return type[from..];
        }
    }
}
