// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.Cli.Commands.Developer.Stacks;

/// <summary>What <c>idrak trace --stacks</c> reports and how much of it.</summary>
/// <param name="Top">Rows a table shows.</param>
/// <param name="Focus">A method name (any part, any case) whose callers, callees and time inside are shown, or null.</param>
/// <param name="Waits">Whether the method tables count waiting and stopped time too (by default busy time only).</param>
/// <param name="AllFrames">Whether the total-time table lists the .NET libraries' methods too.</param>
/// <param name="Threads">Whether each thread's line is shown.</param>
/// <param name="Slices">The timeline's slices (0: no timeline).</param>
internal sealed record StackReportOptions(int Top, string? Focus, bool Waits, bool AllFrames, bool Threads, int Slices);

/// <summary>
/// The report of a <see cref="StackProfile"/>: where the threads' time went (running, outside .NET code, spinning, stopped
/// by the runtime, waiting), the methods by their own and their total time, the allocations by type and the code that made
/// them, and on request a method's callers and callees (--focus), each thread (--by-thread), a timeline (--timeline) and a
/// comparison with an earlier trace (--compare). Printed as tables, or as one JSON object.
/// </summary>
internal static class StackReport
{
    private static readonly CultureInfo Invariant = CultureInfo.InvariantCulture;

    private static readonly (TimeKind Kind, string Name, string Meaning)[] Kinds =
    [
        (TimeKind.Running, "running .NET code", "the CPU in .NET code"),
        (TimeKind.Outside, "outside .NET code", "native calls (drivers, the system) and the runtime (compiling, loading)"),
        (TimeKind.Spinning, "spinning", "spin-waits: a core burned while waiting"),
        (TimeKind.Stopped, "stopped by the runtime", "garbage collections: every thread stopped, or waiting at a GC poll"),
        (TimeKind.Waiting, "waiting", "blocked: locks, events, sleeps, I/O, idle thread pools"),
    ];

    private static bool Busy(TimeKind kind) => kind is TimeKind.Running or TimeKind.Outside or TimeKind.Spinning;

    /// <summary>Writes the report (text, or JSON into <paramref name="json"/>).</summary>
    public static void Write(CommandContext context, StackProfile p, StackReportOptions o, StackProfile? old, JsonObject? json)
    {
        var trace = p.Trace;
        double freq = trace.Frequency;
        string Ms(long ticks) => (ticks * 1000.0 / freq).ToString(ticks * 1000.0 / freq >= 100 ? "N0" : "N1", Invariant);
        string Pct(long part, long whole) => whole == 0 ? "-" : (100.0 * part / whole).ToString(part * 1000L >= whole ? "N1" : "N2", Invariant) + "%";
        bool Counted(TimeKind k) => o.Waits || Busy(k);

        // ------------------------------------------------------------ where the time went
        var byKind = new long[5];
        var threads = new HashSet<long>();
        foreach (var m in p.Moments)
        {
            byKind[(int)m.Kind] += m.Length;
            threads.Add(m.Thread);
        }

        long all = byKind.Sum(), busy = byKind[(int)TimeKind.Running] + byKind[(int)TimeKind.Outside] + byKind[(int)TimeKind.Spinning];
        double span = trace.Seconds(p.Window.End - p.Window.Start);
        var collections = trace.Suspensions.Where(s => s.ForCollection && s.Start >= p.Window.Start && s.Start <= p.Window.End).ToList();
        long stoppedTicks = collections.Sum(s => s.End - s.Start), longest = collections.Count == 0 ? 0 : collections.Max(s => s.End - s.Start);
        var (median, slow) = p.Interval;
        string header = string.Create(Invariant,
            $"stacks: {threads.Count} threads over {span:N2} s ({trace.Seconds(p.Window.Start):N2} to {trace.Seconds(p.Window.End):N2} s); {p.Counts.Samples:N0} samples and {p.Counts.Events:N0} other events with stacks; a sampling round every {median * 1000.0 / freq:N1} ms (the slowest tenth {slow * 1000.0 / freq:N1} ms)");
        string busyLine = string.Create(Invariant,
            $"busy: {(span > 0 ? trace.Seconds(busy) / span : 0):N1} threads on average of {trace.Processors} processors; garbage collections: {collections.Count:N0}, every thread stopped {trace.Seconds(stoppedTicks):N2} s in all (the longest {longest * 1000.0 / freq:N1} ms)");
        if (json is null)
        {
            context.Write(header);
            if (!trace.Listed)
            {
                context.Write("  (the trace has no end-of-run method listing: the run was stopped early, and methods compiled ahead of time show as [unknown code])");
            }

            context.Write(string.Create(Invariant, $"where the threads' time went ({trace.Seconds(all):N1} s of thread time):"));
            context.Table(["Kind", "Share", "Seconds", "What it is"],
                Kinds.Select(k => (IReadOnlyList<string>)[k.Name, Pct(byKind[(int)k.Kind], all), trace.Seconds(byKind[(int)k.Kind]).ToString("N2", Invariant), k.Meaning]));
            context.Write(busyLine);
        }
        else
        {
            json["threads"] = threads.Count;
            json["seconds"] = span;
            json["window"] = new JsonArray(trace.Seconds(p.Window.Start), trace.Seconds(p.Window.End));
            json["samples"] = p.Counts.Samples;
            json["eventsWithStacks"] = p.Counts.Events;
            json["samplingMs"] = new JsonObject { ["median"] = median * 1000.0 / freq, ["slowestTenth"] = slow * 1000.0 / freq };
            json["processors"] = trace.Processors;
            json["listed"] = trace.Listed;
            json["time"] = new JsonObject([.. Kinds.Select(k => KeyValuePair.Create(k.Kind.ToString().ToLowerInvariant(), (JsonNode?)trace.Seconds(byKind[(int)k.Kind])))]);
            json["busyThreads"] = span > 0 ? trace.Seconds(busy) / span : 0;
            json["collections"] = new JsonObject { ["count"] = collections.Count, ["stoppedSeconds"] = trace.Seconds(stoppedTicks), ["longestMs"] = longest * 1000.0 / freq };
        }

        // ------------------------------------------------------------ methods by own and total time
        int n = p.Names.Length;
        var own = new long[n];
        var outside = new long[n];
        var total = new long[n];
        var stamp = new int[n];
        int s = 0;
        long counted = 0;
        foreach (var m in p.Moments)
        {
            if (!Counted(m.Kind))
            {
                continue;
            }

            counted += m.Length;
            int leaf = m.Frames.Length > 0 ? m.Frames[0] : 0;
            own[leaf] += m.Length;
            if (m.Kind == TimeKind.Outside)
            {
                outside[leaf] += m.Length;
            }

            s++;
            foreach (int f in m.Frames)
            {
                if (stamp[f] != s)
                {
                    stamp[f] = s;
                    total[f] += m.Length;
                }
            }
        }

        var ownTop = Enumerable.Range(0, n).Where(i => own[i] > 0).OrderByDescending(i => own[i]).Take(o.Top).ToList();
        var totalTop = Enumerable.Range(0, n).Where(i => total[i] > 0 && (o.AllFrames || !p.IsLibrary(i))).OrderByDescending(i => total[i]).Take(o.Top).ToList();
        string what = o.Waits ? "all time" : "busy time: running, outside .NET code, spinning";
        if (json is null)
        {
            context.Write(string.Create(Invariant, $"methods by their own time ({what}; {trace.Seconds(counted):N2} s):"));
            context.Table(["Own ms", "Own", "Outside .NET", "Total ms", "Total", "Method"],
                ownTop.Select(i => (IReadOnlyList<string>)[Ms(own[i]), Pct(own[i], counted), Pct(outside[i], own[i]), Ms(total[i]), Pct(total[i], counted), p.Display[i]]));
            if (own[p.Returning] * 100 > counted)
            {
                context.Write($"  {p.Display[p.Returning]}, {Pct(own[p.Returning], counted)}: the runtime stops threads at a call's return for each sampling round, so the method that ran is gone from the stack; its callers' total time holds it (many short calls in a loop make it large).");
            }
            context.Write(string.Create(Invariant, $"methods by their total time (with what they call; {(o.AllFrames ? "every method" : "the program's own methods: --all-frames for the .NET libraries' too")}):"));
            context.Table(["Total ms", "Total", "Own ms", "Method"],
                totalTop.Select(i => (IReadOnlyList<string>)[Ms(total[i]), Pct(total[i], counted), Ms(own[i]), p.Display[i]]));
        }
        else
        {
            json["countedSeconds"] = trace.Seconds(counted);
            JsonObject Row(int i) => new() { ["method"] = p.Names[i], ["name"] = p.Display[i], ["ownMs"] = own[i] * 1000.0 / freq, ["totalMs"] = total[i] * 1000.0 / freq, ["outsideMs"] = outside[i] * 1000.0 / freq };
            json["byOwnTime"] = new JsonArray([.. ownTop.Select(i => (JsonNode)Row(i))]);
            json["byTotalTime"] = new JsonArray([.. totalTop.Select(i => (JsonNode)Row(i))]);
        }

        // ------------------------------------------------------------ allocations
        // By type, the program's frame that allocated and the program's frame that called it.
        var allocations = new Dictionary<(string Type, int Method, int Caller), long>();
        long allocated = 0;
        foreach (var (report, frames) in p.Allocations)
        {
            int first = Array.FindIndex(frames, f => !p.IsLibrary(f));
            int at = first >= 0 ? frames[first] : frames.Length > 0 ? frames[0] : 0;
            int next = first >= 0 ? Array.FindIndex(frames, first + 1, f => !p.IsLibrary(f) && f != at) : -1;
            var key = (report.Type, at, next >= 0 ? frames[next] : -1);
            allocations[key] = allocations.GetValueOrDefault(key) + report.Bytes;
            allocated += report.Bytes;
        }

        var allocTop = allocations.OrderByDescending(a => a.Value).Take(o.Top).ToList();
        if (json is null)
        {
            context.Write(string.Create(Invariant, $"allocations ({allocated / 1048576.0:N0} MiB reported, a report each ~100 KB; by type and the program's code that made them):"));
            context.Table(["MiB", "Share", "Type", "Allocated in", "Called from"],
                allocTop.Select(a => (IReadOnlyList<string>)[(a.Value / 1048576.0).ToString("N1", Invariant), Pct(a.Value, allocated), a.Key.Type, p.Display[a.Key.Method],
                    a.Key.Caller >= 0 ? p.Display[a.Key.Caller] : "-"]));
        }
        else
        {
            json["allocatedMiB"] = allocated / 1048576.0;
            json["allocations"] = new JsonArray([.. allocTop.Select(a => (JsonNode)new JsonObject
            {
                ["type"] = a.Key.Type, ["in"] = p.Names[a.Key.Method], ["calledFrom"] = a.Key.Caller >= 0 ? p.Names[a.Key.Caller] : null, ["mib"] = a.Value / 1048576.0,
            })]);
        }

        if (o.Focus is { } focus)
        {
            Focus(context, p, o, focus, counted, Counted, Ms, Pct, json);
        }

        if (o.Threads)
        {
            ThreadTable(context, p, o, Ms, Pct, json);
        }

        if (o.Slices > 0)
        {
            Timeline(context, p, o, Counted, json);
        }

        if (old is not null)
        {
            Compare(context, p, old, o, Counted, json);
        }
    }

    // The callers, the calls and the leaves under the outermost frame matching `focus`.
    private static void Focus(CommandContext context, StackProfile p, StackReportOptions o, string focus, long counted, Func<TimeKind, bool> isCounted,
        Func<long, string> ms, Func<long, long, string> pct, JsonObject? json)
    {
        int n = p.Names.Length;
        var matches = new bool[n];
        for (int i = 1; i < n; i++)
        {
            matches[i] = p.Names[i].Contains(focus, StringComparison.OrdinalIgnoreCase) || p.Display[i].Contains(focus, StringComparison.OrdinalIgnoreCase);
        }

        var callers = new Dictionary<int, long>();
        var callees = new Dictionary<int, long>();
        var inside = new Dictionary<int, long>();
        var matched = new Dictionary<int, long>();
        long through = 0;
        foreach (var m in p.Moments)
        {
            if (!isCounted(m.Kind))
            {
                continue;
            }

            int r = -1;
            for (int i = m.Frames.Length - 1; i >= 0; i--)
            {
                if (matches[m.Frames[i]])
                {
                    r = i;
                    break;
                }
            }

            if (r < 0)
            {
                continue;
            }

            through += m.Length;
            matched[m.Frames[r]] = matched.GetValueOrDefault(m.Frames[r]) + m.Length;
            int caller = r + 1 < m.Frames.Length ? m.Frames[r + 1] : -1, callee = r > 0 ? m.Frames[r - 1] : -2;
            callers[caller] = callers.GetValueOrDefault(caller) + m.Length;
            callees[callee] = callees.GetValueOrDefault(callee) + m.Length;
            inside[m.Frames[0]] = inside.GetValueOrDefault(m.Frames[0]) + m.Length;
        }

        string Name(int i) => i == -1 ? "(the thread's start)" : i == -2 ? "(its own code)" : p.Display[i];
        var top = (Dictionary<int, long> d) => d.OrderByDescending(x => x.Value).Take(o.Top).ToList();
        if (json is null)
        {
            context.Write(string.Create(Invariant, $"focus \"{focus}\": {ms(through)} ms, {pct(through, counted)} of the counted time, through {matched.Count} matching methods"));
            if (through == 0)
            {
                context.Write("  no stack holds a method whose name holds that text");
                return;
            }

            foreach (var (title, table) in new[] { ("matching methods", matched), ("called from", callers), ("calls (its callees, with what they call)", callees), ("time inside it by the method running (own time)", inside) })
            {
                context.Write($"  {title}:");
                context.Table(["ms", "Share", "Method"], top(table).Select(x => (IReadOnlyList<string>)[ms(x.Value), pct(x.Value, through), Name(x.Key)]));
            }
        }
        else
        {
            JsonArray Rows(Dictionary<int, long> d) => new([.. top(d).Select(x => (JsonNode)new JsonObject
            {
                ["method"] = x.Key < 0 ? Name(x.Key) : p.Names[x.Key], ["ms"] = x.Value * 1000.0 / p.Trace.Frequency,
            })]);
            json["focus"] = new JsonObject
            {
                ["text"] = focus, ["ms"] = through * 1000.0 / p.Trace.Frequency, ["matching"] = Rows(matched), ["calledFrom"] = Rows(callers),
                ["calls"] = Rows(callees), ["inside"] = Rows(inside),
            };
        }
    }

    private static void ThreadTable(CommandContext context, StackProfile p, StackReportOptions o, Func<long, string> ms, Func<long, long, string> pct, JsonObject? json)
    {
        var rows = p.Moments.GroupBy(m => m.Thread).Select(g =>
        {
            var kinds = new long[5];
            var leaves = new Dictionary<int, long>();
            foreach (var m in g)
            {
                kinds[(int)m.Kind] += m.Length;
                if (Busy(m.Kind) && m.Frames.Length > 0)
                {
                    leaves[m.Frames[0]] = leaves.GetValueOrDefault(m.Frames[0]) + m.Length;
                }
            }

            long busy = kinds[0] + kinds[1] + kinds[2];
            int topLeaf = leaves.Count == 0 ? 0 : leaves.MaxBy(x => x.Value).Key;
            return (Thread: g.Key, Kinds: kinds, Busy: busy, All: kinds.Sum(), Top: topLeaf);
        }).OrderByDescending(r => r.Busy).Take(o.Top).ToList();
        if (json is null)
        {
            context.Write("threads (the busiest first):");
            context.Table(["Thread", "Busy ms", "Running", "Outside", "Spinning", "Stopped", "Waiting", "Busiest in"],
                rows.Select(r => (IReadOnlyList<string>)[r.Thread.ToString(Invariant), ms(r.Busy), pct(r.Kinds[0], r.All), pct(r.Kinds[1], r.All), pct(r.Kinds[2], r.All),
                    pct(r.Kinds[3], r.All), pct(r.Kinds[4], r.All), r.Busy == 0 ? "-" : p.Display[r.Top]]));
        }
        else
        {
            json["byThread"] = new JsonArray([.. rows.Select(r => (JsonNode)new JsonObject
            {
                ["thread"] = r.Thread, ["busyMs"] = r.Busy * 1000.0 / p.Trace.Frequency,
                ["time"] = new JsonObject([.. Kinds.Select(k => KeyValuePair.Create(k.Kind.ToString().ToLowerInvariant(), (JsonNode?)(r.Kinds[(int)k.Kind] * 1000.0 / p.Trace.Frequency)))]),
                ["busiestIn"] = r.Busy == 0 ? null : p.Names[r.Top],
            })]);
        }
    }

    // The window in equal slices: the busy threads on average, the share stopped by the runtime, the method most often running.
    private static void Timeline(CommandContext context, StackProfile p, StackReportOptions o, Func<TimeKind, bool> isCounted, JsonObject? json)
    {
        var (start, end) = p.Window;
        long width = Math.Max(1, (end - start + o.Slices - 1) / o.Slices);
        var busy = new long[o.Slices];
        var stopped = new long[o.Slices];
        var leaves = new Dictionary<int, long>[o.Slices];
        foreach (var m in p.Moments)
        {
            int i = (int)Math.Clamp((m.Time - start) / width, 0, o.Slices - 1);
            if (Busy(m.Kind))
            {
                busy[i] += m.Length;
            }
            else if (m.Kind == TimeKind.Stopped)
            {
                stopped[i] += m.Length;
            }

            if (isCounted(m.Kind) && m.Frames.Length > 0)
            {
                (leaves[i] ??= [])[m.Frames[0]] = leaves[i].GetValueOrDefault(m.Frames[0]) + m.Length;
            }
        }

        double seconds = p.Trace.Seconds(width);
        var rows = Enumerable.Range(0, o.Slices).Select(i =>
        {
            int top = leaves[i] is { Count: > 0 } l ? l.MaxBy(x => x.Value).Key : 0;
            return (From: p.Trace.Seconds(start + i * width), Busy: p.Trace.Seconds(busy[i]) / seconds, Stopped: p.Trace.Seconds(stopped[i]) / seconds, Top: top, Any: leaves[i] is { Count: > 0 });
        }).ToList();
        if (json is null)
        {
            context.Write(string.Create(Invariant, $"timeline ({o.Slices} slices of {seconds * 1000:N0} ms; threads busy and stopped on average):"));
            context.Table(["From s", "Busy", "Stopped", "Running most", "Bar"],
                rows.Select(r => (IReadOnlyList<string>)[r.From.ToString("N2", Invariant), r.Busy.ToString("N1", Invariant), r.Stopped.ToString("N1", Invariant),
                    r.Any ? p.Display[r.Top] : "-", new string('#', (int)Math.Round(Math.Min(r.Busy, 64)))]));
        }
        else
        {
            json["timeline"] = new JsonArray([.. rows.Select(r => (JsonNode)new JsonObject
            {
                ["from"] = r.From, ["busyThreads"] = r.Busy, ["stoppedThreads"] = r.Stopped, ["runningMost"] = r.Any ? p.Names[r.Top] : null,
            })]);
        }
    }

    // The kinds of time, and the methods whose own time changed most, against an earlier trace.
    private static void Compare(CommandContext context, StackProfile p, StackProfile old, StackReportOptions o, Func<TimeKind, bool> isCounted, JsonObject? json)
    {
        static (long[] Kinds, Dictionary<string, long> Own) Sums(StackProfile q, Func<TimeKind, bool> counted)
        {
            var kinds = new long[5];
            var own = new Dictionary<string, long>(StringComparer.Ordinal);
            foreach (var m in q.Moments)
            {
                kinds[(int)m.Kind] += m.Length;
                if (counted(m.Kind) && m.Frames.Length > 0)
                {
                    string name = q.Names[m.Frames[0]];
                    own[name] = own.GetValueOrDefault(name) + m.Length;
                }
            }

            return (kinds, own);
        }

        var (nowKinds, nowOwn) = Sums(p, isCounted);
        var (oldKinds, oldOwn) = Sums(old, isCounted);
        double Sec(StackProfile q, long t) => q.Trace.Seconds(t);
        var display = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var q in new[] { old, p })
        {
            for (int i = 0; i < q.Names.Length; i++)
            {
                display[q.Names[i]] = q.Display[i];
            }
        }

        var changes = nowOwn.Keys.Union(oldOwn.Keys)
            .Select(k => (Name: k, Old: Sec(old, oldOwn.GetValueOrDefault(k)) * 1000, Now: Sec(p, nowOwn.GetValueOrDefault(k)) * 1000))
            .OrderByDescending(c => Math.Abs(c.Now - c.Old)).Take(o.Top).ToList();
        double oldSpan = old.Trace.Seconds(old.Window.End - old.Window.Start), nowSpan = p.Trace.Seconds(p.Window.End - p.Window.Start);
        if (json is null)
        {
            context.Write(string.Create(Invariant, $"against {old.Trace.Path} ({oldSpan:N2} s then, {nowSpan:N2} s now):"));
            context.Table(["Kind", "Then s", "Now s", "Change"], Kinds.Select(k =>
            {
                double a = Sec(old, oldKinds[(int)k.Kind]), b = Sec(p, nowKinds[(int)k.Kind]);
                return (IReadOnlyList<string>)[k.Name, a.ToString("N2", Invariant), b.ToString("N2", Invariant), (b - a).ToString("+0.00;-0.00;0", Invariant)];
            }));
            context.Write("  the methods whose own time changed most:");
            context.Table(["Then ms", "Now ms", "Change", "Method"], changes.Select(c => (IReadOnlyList<string>)[c.Old.ToString("N1", Invariant), c.Now.ToString("N1", Invariant),
                (c.Now - c.Old).ToString("+0.0;-0.0;0", Invariant), display.GetValueOrDefault(c.Name, c.Name)]));
        }
        else
        {
            json["compare"] = new JsonObject
            {
                ["against"] = old.Trace.Path, ["secondsThen"] = oldSpan, ["secondsNow"] = nowSpan,
                ["timeThen"] = new JsonObject([.. Kinds.Select(k => KeyValuePair.Create(k.Kind.ToString().ToLowerInvariant(), (JsonNode?)Sec(old, oldKinds[(int)k.Kind])))]),
                ["methods"] = new JsonArray([.. changes.Select(c => (JsonNode)new JsonObject { ["method"] = c.Name, ["thenMs"] = c.Old, ["nowMs"] = c.Now })]),
            };
        }
    }
}
