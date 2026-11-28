// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// The CPU kernels' parallel loop that allocates nothing (CpuRanges), private to the backend.
internal sealed partial class CpuBackend
{
    /// <summary>The work of one range of chunks [first, last) of a parallel loop (<see cref="CpuRanges"/>), as a struct.</summary>
    private interface IRangeBody
    {
        /// <summary>Runs the items [<paramref name="first"/>, <paramref name="last"/>).</summary>
        void Run(int first, int last);
    }

    /// <summary>
    /// A parallel loop over ranges that allocates nothing once warm (a decoding step's kernels run it on every call):
    /// <c>Parallel.For</c> allocates its tasks, replicas, queue segments and closures on every call (about 2 KB with four
    /// cores). The body is a struct read through a pointer; its helpers are the process's crew (<see cref="Crew"/>: threads
    /// of its own that watch for the next call, so the calls of a decoding step do not each wake the thread pool's workers),
    /// or, while another caller holds the crew, work items made once per calling thread and queued to the thread pool again
    /// on each call. The caller takes chunks too, so a call finishes even if no helper starts. Each chunk is
    /// <c>[c · size, min((c + 1) · size, count))</c>, the chunks handed out in order to whoever asks next: the same ranges
    /// <c>Parallel.For</c> over the chunks runs, so a body whose chunks write apart gives the same result whichever thread
    /// runs each.
    /// </summary>
    private static unsafe class CpuRanges
    {
        [ThreadStatic]
        private static Job? t_job;

        // Set on a crew member's thread, and on a caller's while its crew call runs: a loop inside a body runs in order there.
        [ThreadStatic]
        private static bool t_inside;

        // (This loop's methods are compiled optimized from their first calls, as the bodies they run are: run by every thread
        // of a call, their unoptimized and then instrumented first code was what each thread ran around every chunk.)

        /// <summary>
        /// Runs <paramref name="body"/> over the <paramref name="chunks"/> chunks of <paramref name="size"/> items of
        /// [0, <paramref name="count"/>), on up to <paramref name="threads"/> threads (the caller's among them). The body is
        /// read in place through its address, so it must be a local of the caller (on its stack) for the whole call.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public static void Run<TBody>(int chunks, int size, int count, int threads, ref TBody body) where TBody : struct, IRangeBody
        {
            int helpers = Math.Min(Math.Min(chunks, threads), Environment.ProcessorCount) - 1;
            if (helpers <= 0 || t_inside)
            {
                InOrder(chunks, size, count, ref body);
                return;
            }

            // The process's crew when no other caller holds it; else this thread's helpers from the thread pool.
            if (Crew.Shared.TryRun(&Invoke<TBody>, Unsafe.AsPointer(ref body), chunks, size, count, helpers))
            {
                return;
            }

            var job = t_job ??= new Job();
            if (job.Busy)
            {
                InOrder(chunks, size, count, ref body);
                return;
            }

            job.Run(&Invoke<TBody>, Unsafe.AsPointer(ref body), chunks, size, count, helpers);
        }

        // One thread, or a loop inside a body on this thread: the chunks in order here.
        private static void InOrder<TBody>(int chunks, int size, int count, ref TBody body) where TBody : struct, IRangeBody
        {
            for (int c = 0; c < chunks; c++)
            {
                int start = c * size;
                if (start < count)
                {
                    body.Run(start, Math.Min(start + size, count));
                }
            }
        }

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static void Invoke<TBody>(void* body, int first, int last) where TBody : struct, IRangeBody => Unsafe.AsRef<TBody>(body).Run(first, last);

        // One calling thread's loop: its helpers (work items queued again on each call) and the state of the call running.
        private sealed class Job
        {
            // Set in `_active` once the caller has finished its share: helpers arriving after it leave without working.
            private const int Closed = 1 << 30;

            private readonly Helper[] _helpers;
            private readonly ManualResetEventSlim _done = new(false);
            private delegate*<void*, int, int, void> _invoke;
            private void* _body;
            private int _chunks, _size, _count;
            private int _next;

            // Helpers inside the call's work, plus Closed once the caller is done with its own (Closed alone between calls).
            private int _active = Closed;
            private ExceptionDispatchInfo? _error;

            public Job()
            {
                _helpers = new Helper[Math.Max(1, Environment.ProcessorCount - 1)];
                for (int i = 0; i < _helpers.Length; i++)
                {
                    _helpers[i] = new Helper(this);
                }
            }

            public bool Busy { get; private set; }

            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public void Run(delegate*<void*, int, int, void> invoke, void* body, int chunks, int size, int count, int helpers)
            {
                Busy = true;
                try
                {
                    _invoke = invoke;
                    _body = body;
                    (_chunks, _size, _count, _next, _error) = (chunks, size, count, 0, null);
                    Volatile.Write(ref _active, 0);                                      // open: the fields above are seen by every helper that joins
                    for (int i = 0; i < helpers; i++)
                    {
                        ThreadPool.UnsafeQueueUserWorkItem(_helpers[i], preferLocal: false);
                    }

                    Work();
                    if (Interlocked.Add(ref _active, Closed) != Closed)
                    {
                        _done.Wait();                                                    // helpers still inside: the last one out sets it
                    }

                    _done.Reset();
                    _body = null;
                    if (_error is { } error)
                    {
                        _error = null;
                        error.Throw();
                    }
                }
                finally
                {
                    Busy = false;
                }
            }

            // A helper: joins the open call (one queued for an earlier call joins the one running now, or leaves when none is).
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public void Help()
            {
                int state = Volatile.Read(ref _active);
                while (true)
                {
                    if ((state & Closed) != 0)
                    {
                        return;
                    }

                    int seen = Interlocked.CompareExchange(ref _active, state + 1, state);
                    if (seen == state)
                    {
                        break;
                    }

                    state = seen;
                }

                Work();
                if (Interlocked.Decrement(ref _active) == Closed)
                {
                    _done.Set();                                                         // the caller is waiting for the last helper
                }
            }

            // Takes the next chunk until none is left.
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            private void Work()
            {
                try
                {
                    int c;
                    while ((c = Interlocked.Increment(ref _next) - 1) < _chunks)
                    {
                        int start = c * _size;
                        if (start < _count)
                        {
                            _invoke(_body, start, Math.Min(start + _size, _count));
                        }
                    }
                }
                catch (Exception e)
                {
                    Interlocked.CompareExchange(ref _error, ExceptionDispatchInfo.Capture(e), null);
                    Volatile.Write(ref _next, int.MaxValue / 2);                         // no further chunks
                }
            }
        }

        private sealed class Helper(Job job) : IThreadPoolWorkItem
        {
            public void Execute() => job.Help();
        }

        // A value alone on its cache lines (the members read the call's epoch while others take chunks).
        [StructLayout(LayoutKind.Explicit, Size = 192)]
        private struct Lone
        {
            [FieldOffset(64)]
            public long Value;
        }

        /// <summary>
        /// The process's own threads for the loops, one caller at a time. A member waits for the next call by watching its
        /// epoch, a few pauses a look; each time it has watched as long as giving its core up costs it gives the core up to
        /// any thread ready to run (Thread.Sleep(0)), so it never holds a core another thread waits for; once it has watched
        /// as long as waking the sleeping crew takes here it sleeps on its own event until a call wakes it (both measured
        /// once: <see cref="Calibrate"/>). It joins a call as one of its helpers and takes chunks as the caller does. So a run
        /// of calls with short serial work between them (a decoding step's) finds its helpers awake.
        /// </summary>
        private sealed class Crew
        {
            private const int Closed = 1 << 30;

            // Rounds the measuring of a wake and of a yield take (Calibrate): their medians.
            private const int WakeRounds = 9, YieldRounds = 63;

            private static readonly Lock s_gate = new();
            private static Crew? s_shared;

            private readonly Member?[] _members = new Member?[Math.Max(1, Environment.ProcessorCount - 1)];
            private readonly ManualResetEventSlim _done = new(false);
            private delegate*<void*, int, int, void> _invoke;
            private void* _body;
            private int _chunks, _size, _count, _helpers;
            private Lone _next;
            private Lone _epoch;
            private Lone _active = new() { Value = Closed };
            private int _owned;
            private ExceptionDispatchInfo? _error;

            // How long a member watches for the next call before it sleeps, and between its yields (stopwatch ticks):
            // Calibrate's (none before it, so the members it makes sleep at once).
            private long _spinTicks, _yieldTicks;

            private Crew()
            {
            }

            public static Crew Shared => s_shared ?? Make();

            [MethodImpl(MethodImplOptions.NoInlining)]
            private static Crew Make()
            {
                lock (s_gate)
                {
                    if (s_shared is null)
                    {
                        var crew = new Crew();
                        crew.Calibrate();
                        s_shared = crew;
                    }

                    return s_shared;
                }
            }

            // What a yield costs here (the median time of a Thread.Sleep(0) on this thread), then the members the thread
            // budget takes made and their wake measured: every member asleep, a call that takes all of them timed from its
            // start until the last member is inside it (each chunk waits there until every thread has come, so each thread
            // takes one); the median of the rounds. A member then watches for the next call that long: a gap between calls
            // shorter than a wake finds it awake.
            private void Calibrate()
            {
                Span<long> yields = stackalloc long[YieldRounds];
                for (int r = 0; r < YieldRounds; r++)
                {
                    long t0 = Stopwatch.GetTimestamp();
                    Thread.Sleep(0);
                    yields[r] = Stopwatch.GetTimestamp() - t0;
                }

                yields.Sort();
                int count = Math.Clamp(ComputeResources.MaxCpuThreads - 1, 1, _members.Length);
                for (int i = 0; i < count; i++)
                {
                    _members[i] = new Member(this, i);
                }

                Span<long> wakes = stackalloc long[WakeRounds];
                for (int r = 0; r < WakeRounds; r++)
                {
                    long until = Stopwatch.GetTimestamp() + Stopwatch.Frequency / 10;
                    while (!AllAsleep(count) && Stopwatch.GetTimestamp() < until)
                    {
                        Thread.Sleep(0);
                    }

                    var arrivals = new Arrivals(count + 1);
                    TryRun(&Arrivals.Invoke, &arrivals, count + 1, 1, count + 1, count);
                    wakes[r] = arrivals.Last - arrivals.Start;
                }

                wakes.Sort();
                _yieldTicks = Math.Max(1, yields[YieldRounds / 2]);
                _spinTicks = wakes[WakeRounds / 2];
            }

            private bool AllAsleep(int count)
            {
                for (int i = 0; i < count; i++)
                {
                    if (!_members[i]!.Asleep)
                    {
                        return false;
                    }
                }

                return true;
            }

            // Calibrate's call: each chunk counts its thread in and waits (2 ms at most) until all have come.
            private struct Arrivals(int target)
            {
                public readonly long Start = Stopwatch.GetTimestamp();
                public long Last;
                private readonly int _target = target;
                private int _arrived;

                public static void Invoke(void* at, int first, int last)
                {
                    ref var a = ref Unsafe.AsRef<Arrivals>(at);
                    if (Interlocked.Increment(ref a._arrived) == a._target)
                    {
                        a.Last = Stopwatch.GetTimestamp();
                    }

                    long until = Stopwatch.GetTimestamp() + Stopwatch.Frequency / 500;
                    while (Volatile.Read(ref a._arrived) < a._target && Stopwatch.GetTimestamp() < until)
                    {
                        Thread.SpinWait(4);
                    }
                }
            }

            /// <summary>Runs the call on the crew, or false at once when another caller holds it.</summary>
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            public bool TryRun(delegate*<void*, int, int, void> invoke, void* body, int chunks, int size, int count, int helpers)
            {
                if (Interlocked.CompareExchange(ref _owned, 1, 0) != 0)
                {
                    return false;
                }

                t_inside = true;
                try
                {
                    helpers = Math.Min(helpers, _members.Length);
                    for (int i = 0; i < helpers; i++)
                    {
                        _members[i] ??= new Member(this, i);
                    }

                    _invoke = invoke;
                    _body = body;
                    (_chunks, _size, _count, _helpers, _error) = (chunks, size, count, helpers, null);
                    _next.Value = 0;
                    Volatile.Write(ref _active.Value, 0);                                // open: the fields above are seen by every member that joins
                    Interlocked.Increment(ref _epoch.Value);                             // and a fence before the members' sleep flags are read
                    for (int i = 0; i < helpers; i++)
                    {
                        _members[i]!.Wake();
                    }

                    Work();
                    if (Interlocked.Add(ref _active.Value, Closed) != Closed)
                    {
                        _done.Wait();                                                    // members still inside: the last one out sets it
                    }

                    _done.Reset();
                    _body = null;
                    if (_error is { } error)
                    {
                        _error = null;
                        error.Throw();
                    }
                }
                finally
                {
                    t_inside = false;
                    Volatile.Write(ref _owned, 0);
                }

                return true;
            }

            // A member's part of the call open now: joins while it is open, takes chunks, leaves. False when the call does not
            // take this member (its helpers are fewer).
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            private bool Join(int index)
            {
                long state = Volatile.Read(ref _active.Value);
                while (true)
                {
                    if (index >= _helpers)
                    {
                        return false;
                    }

                    if ((state & Closed) != 0)
                    {
                        return true;
                    }

                    long seen = Interlocked.CompareExchange(ref _active.Value, state + 1, state);
                    if (seen == state)
                    {
                        break;
                    }

                    state = seen;
                }

                Work();
                if (Interlocked.Decrement(ref _active.Value) == Closed)
                {
                    _done.Set();                                                         // the caller is waiting for the last member
                }

                return true;
            }

            // Takes the next chunk until none is left.
            [MethodImpl(MethodImplOptions.AggressiveOptimization)]
            private void Work()
            {
                try
                {
                    long c;
                    while ((c = Interlocked.Increment(ref _next.Value) - 1) < _chunks)
                    {
                        int start = (int)c * _size;
                        if (start < _count)
                        {
                            _invoke(_body, start, Math.Min(start + _size, _count));
                        }
                    }
                }
                catch (Exception e)
                {
                    Interlocked.CompareExchange(ref _error, ExceptionDispatchInfo.Capture(e), null);
                    Volatile.Write(ref _next.Value, int.MaxValue / 2);                   // no further chunks
                }
            }

            private sealed class Member
            {
                private readonly Crew _crew;
                private readonly int _index;
                private readonly AutoResetEvent _wake = new(false);
                private Lone _sleeping;

                public Member(Crew crew, int index)
                {
                    (_crew, _index) = (crew, index);
                    new Thread(Loop) { IsBackground = true, Name = "Idrak CPU " + index.ToString(CultureInfo.InvariantCulture) }.Start();
                }

                public bool Asleep => Volatile.Read(ref _sleeping.Value) != 0;

                // The caller's, after the epoch moved: a member asleep is woken (one going to sleep saw the epoch or is woken here).
                [MethodImpl(MethodImplOptions.AggressiveOptimization)]
                public void Wake()
                {
                    if (Volatile.Read(ref _sleeping.Value) != 0 && Interlocked.Exchange(ref _sleeping.Value, 0) != 0)
                    {
                        _wake.Set();
                    }
                }

                [MethodImpl(MethodImplOptions.AggressiveOptimization)]
                private void Loop()
                {
                    t_inside = true;
                    var crew = _crew;
                    long seen = Volatile.Read(ref crew._epoch.Value), since = Stopwatch.GetTimestamp();
                    while (true)
                    {
                        long yieldAt = since + crew._yieldTicks;
                        while (Volatile.Read(ref crew._epoch.Value) == seen)
                        {
                            Thread.SpinWait(4);
                            long now = Stopwatch.GetTimestamp();
                            if (now - since > crew._spinTicks)
                            {
                                Interlocked.Exchange(ref _sleeping.Value, 1);            // a fence before the epoch is read again
                                if (Volatile.Read(ref crew._epoch.Value) == seen)
                                {
                                    _wake.WaitOne();
                                }

                                Volatile.Write(ref _sleeping.Value, 0);
                                since = Stopwatch.GetTimestamp();
                                yieldAt = since + crew._yieldTicks;
                            }
                            else if (now >= yieldAt)
                            {
                                Thread.Sleep(0);
                                yieldAt = Stopwatch.GetTimestamp() + crew._yieldTicks;
                            }
                        }

                        seen = Volatile.Read(ref crew._epoch.Value);
                        if (crew.Join(_index) || _index < ComputeResources.MaxCpuThreads - 1)
                        {
                            since = Stopwatch.GetTimestamp();                            // the watch counts from the last call it could have been taken for
                        }
                    }
                }
            }
        }
    }
}
