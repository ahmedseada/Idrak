// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Runtime.ExceptionServices;

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
    /// cores). The body is a struct read through a pointer, the helpers are work items made once per calling thread and
    /// queued to the thread pool again on each call, and the caller takes chunks too, so a call finishes even if no helper
    /// starts. Each chunk is <c>[c · size, min((c + 1) · size, count))</c>, the chunks handed out in order to whoever asks
    /// next: the same ranges <c>Parallel.For</c> over the chunks runs, so a body whose chunks write apart gives the same
    /// result whichever thread runs each.
    /// </summary>
    private static unsafe class CpuRanges
    {
        [ThreadStatic]
        private static Job? t_job;

        /// <summary>
        /// Runs <paramref name="body"/> over the <paramref name="chunks"/> chunks of <paramref name="size"/> items of
        /// [0, <paramref name="count"/>), on up to <paramref name="threads"/> threads (the caller's among them). The body is
        /// read in place through its address, so it must be a local of the caller (on its stack) for the whole call.
        /// </summary>
        public static void Run<TBody>(int chunks, int size, int count, int threads, ref TBody body) where TBody : struct, IRangeBody
        {
            int helpers = Math.Min(Math.Min(chunks, threads), Environment.ProcessorCount) - 1;
            var job = t_job ??= new Job();
            if (helpers <= 0 || job.Busy)
            {
                // One thread, or a loop inside a body on this thread: the chunks in order here.
                for (int c = 0; c < chunks; c++)
                {
                    int start = c * size;
                    if (start < count)
                    {
                        body.Run(start, Math.Min(start + size, count));
                    }
                }

                return;
            }

            job.Run(&Invoke<TBody>, Unsafe.AsPointer(ref body), chunks, size, count, helpers);
        }

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
    }
}
