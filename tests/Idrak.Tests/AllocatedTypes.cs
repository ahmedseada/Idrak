// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics.Tracing;

// What a call that should allocate nothing allocates: the types the runtime's allocation ticks name (one tick about every
// 100 KB allocated), so an allocation test's failure names the cause, on whatever machine it fails.
internal static partial class Tests
{
    internal static string AllocatedTypes(Action call, long bytes = 4 << 20, int maxCalls = 200_000)
    {
        using var ticks = new AllocationTicks();
        long start = GC.GetAllocatedBytesForCurrentThread();
        int calls = 0;
        while (calls < maxCalls && GC.GetAllocatedBytesForCurrentThread() - start < bytes)
        {
            call();
            calls++;
        }

        double perCall = (GC.GetAllocatedBytesForCurrentThread() - start) / (double)Math.Max(1, calls);
        for (int wait = 0; wait < 40 && ticks.Types.IsEmpty; wait++)
        {
            Thread.Sleep(50);                                                    // the events arrive on their own thread
        }

        Thread.Sleep(200);
        string types = ticks.Types.IsEmpty ? "no allocation ticks arrived"
            : string.Join(", ", ticks.Types.OrderByDescending(t => t.Value).Select(t => $"{t.Key} ({t.Value} ticks)"));
        return $"{perCall:F1} bytes a call over {calls} calls; allocated: {types}";
    }

    private sealed class AllocationTicks : EventListener
    {
        public readonly ConcurrentDictionary<string, int> Types = new();

        protected override void OnEventSourceCreated(EventSource source)
        {
            if (source.Name == "Microsoft-Windows-DotNETRuntime")
            {
                EnableEvents(source, EventLevel.Verbose, (EventKeywords)0x1);  // GC keyword: GCAllocationTick
            }
        }

        protected override void OnEventWritten(EventWrittenEventArgs e)
        {
            if (e.EventName?.StartsWith("GCAllocationTick", StringComparison.Ordinal) != true || e.PayloadNames is null)
            {
                return;
            }

            int at = e.PayloadNames.IndexOf("TypeName");
            if (at >= 0 && e.Payload?[at] is string type)
            {
                Types.AddOrUpdate(type, 1, (_, n) => n + 1);
            }
        }
    }
}
