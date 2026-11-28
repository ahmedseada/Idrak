// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace Idrak.Abstraction.Devices.Cpu;

// The drivers every CPU operation runs through: an element-wise kernel over [0, n) (Elements) or a kernel a row at a time
// (Rows), each written once over IFloats, run by the width measured fastest for its shape (ChosenLanes: every width that
// runs here timed alike on the real inputs, its outputs moved to scratch while measured, the choice kept for the machine),
// split across the cores where the work passes the parallel cut-over.
internal sealed partial class CpuBackend
{
    /// <summary>An operation's whole run by one width of the float lanes, and its outputs moved to scratch for measuring.</summary>
    internal interface ILanesRun
    {
        void Run(Lanes lanes);

        void Redirect(ref Scratch scratch);
    }

    /// <summary>Scratch outputs for measuring: zeroed, pinned arrays, kept until the measuring ends.</summary>
    internal unsafe struct Scratch
    {
        private List<float[]>? _arrays;

        /// <summary>A pinned, zeroed block of <paramref name="count"/> floats (at least one).</summary>
        public float* Take(long count)
        {
            var array = GC.AllocateArray<float>((int)Math.Clamp(count, 1, int.MaxValue), pinned: true);
            (_arrays ??= []).Add(array);
            return (float*)Unsafe.AsPointer(ref MemoryMarshal.GetArrayDataReference(array));
        }

        /// <summary>Keeps the blocks alive until here.</summary>
        public readonly void End() => GC.KeepAlive(_arrays);
    }

    /// <summary>A width every operation takes instead of its measured one, where it runs here (the lane-bits proof); null: measured.</summary>
    internal static Lanes? ForcedLanes;

    // A choice of width: by the operation, its shape's classes, the vector paths setting and the threads it runs on.
    private readonly record struct LanesKey(string Operation, int A, int B, CpuVectorPaths Vectors, int Threads);

    private static readonly ConcurrentDictionary<LanesKey, Lanes> s_lanesChosen = new();

    /// <summary>Forgets every measured choice of this process (the widths and the described lane's), so the next call measures or reads the machine's file again (tests).</summary>
    internal static void ForgetChoices()
    {
        s_lanesChosen.Clear();
        s_readChosen.Clear();
        Interlocked.Increment(ref s_lanesForgotten);           // every plan's kept lanes are of an earlier generation now
        CpuChoices.Forget();
    }

    // The class of a size: its power of two rounded up (0 for 0 and 1).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static int SizeClass(long n) => n <= 1 ? 0 : 64 - BitOperations.LeadingZeroCount((ulong)(n - 1));

    /// <summary>
    /// The width <paramref name="run"/> takes at its shape: kept from this process or this machine's file, or measured now
    /// (every width that runs here timed alike, CpuChoices.RoundTimes, on the real inputs with the outputs in scratch);
    /// without measuring (IDRAK_AUTOTUNE=0) the widest.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static Lanes ChosenLanes<TR>(string operation, int a, int b, ref TR run) where TR : struct, ILanesRun =>
        ForcedLanes is { } forced && LanesRun(forced) ? forced : s_lanesChosen.TryGetValue(new LanesKey(operation, a, b, ComputeResources.CpuVectors, ComputeResources.MaxCpuThreads), out var lanes) ? lanes
            : MeasureLanes(new LanesKey(operation, a, b, ComputeResources.CpuVectors, ComputeResources.MaxCpuThreads), ref run);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private static Lanes MeasureLanes<TR>(LanesKey key, ref TR run) where TR : struct, ILanesRun
    {
        Span<Lanes> candidates = stackalloc Lanes[4];
        int count = 0;
        foreach (var lanes in (ReadOnlySpan<Lanes>)[Lanes.Wide512, Lanes.Wide256, Lanes.Wide128, Lanes.One])
        {
            if (LanesRun(lanes))
            {
                candidates[count++] = lanes;
            }
        }

        string shape = string.Create(CultureInfo.InvariantCulture, $"{key.Operation} 2^{key.A}x2^{key.B}");
        string fileKey = string.Create(CultureInfo.InvariantCulture, $"lanes {shape} {key.Vectors} t{key.Threads}");
        Lanes best = WidestLanes;
        if (CpuChoices.TryGet(fileKey, out var kept) && Enum.TryParse<Lanes>(kept, out var known) && Listed(candidates[..count], known))
        {
            best = known;
        }
        else if (count == 1)
        {
            best = candidates[0];
            ProductRoutes.Chose("cpu", shape, $"{LanesName(best)} (the only candidate)");
        }
        else if (CpuTuning.Autotune)
        {
            FusedInHardware.Ensure();
            var measured = run;
            var scratch = new Scratch();
            measured.Redirect(ref scratch);
            var each = new LanesCandidates<TR>(measured, candidates[..count].ToArray());
            var times = CpuChoices.RoundTimes(ref each, count);
            scratch.End();
            int at = CpuChoices.Lowest(times);
            best = candidates[at];
            var listed = new System.Text.StringBuilder();
            for (int c = 0; c < count; c++)
            {
                listed.Append(c == 0 ? "" : ", ").Append(LanesName(candidates[c])).Append(' ').Append(times[c].ToString("0.####", CultureInfo.InvariantCulture)).Append(" ms");
            }

            ProductRoutes.Chose("cpu", shape, $"{LanesName(best)} ({listed})");
            CpuChoices.Keep(fileKey, best.ToString());
        }

        s_lanesChosen[key] = best;
        return best;
    }

    private static bool Listed(ReadOnlySpan<Lanes> candidates, Lanes lanes)
    {
        foreach (var c in candidates)
        {
            if (c == lanes)
            {
                return true;
            }
        }

        return false;
    }

    private struct LanesCandidates<TR>(TR run, Lanes[] lanes) : CpuChoices.ICandidates where TR : struct, ILanesRun
    {
        private TR _run = run;

        public void Run(int index) => _run.Run(lanes[index]);
    }

    // ------------------------------------------------------------------ element-wise kernels

    /// <summary>An element-wise kernel: elements [i, i + TF.Count), and its outputs moved to scratch for measuring.</summary>
    internal interface IMap
    {
        void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>;

        void Redirect(ref Scratch scratch);
    }

    // `kernel` over [first, n) by the width measured for (operation, the range's class); `work` element operations an
    // element, for the parallel cut-over. A range past 0 is a chunk of a larger run (an operation reading a typed input a
    // chunk at a time): the kernel indexes its elements as in the whole run.
    internal static void Elements<TK>(string operation, TK kernel, long n, int work = 1, long first = 0) where TK : struct, IMap
    {
        if (n <= first)
        {
            return;
        }

        var run = new MapRun<TK> { Kernel = kernel, First = first, N = n, Work = work };
        run.Run(ChosenLanes(operation, SizeClass(n - first), 0, ref run));
    }

    private struct MapRun<TK> : ILanesRun where TK : struct, IMap
    {
        public TK Kernel;
        public long First, N;
        public int Work;

        public void Run(Lanes lanes)
        {
            switch (lanes)
            {
                case Lanes.Wide512:
                    Spread<TK, Vector512<float>, F512>(ref Kernel, First, N, Work);
                    break;
                case Lanes.Wide256:
                    Spread<TK, Vector256<float>, F256>(ref Kernel, First, N, Work);
                    break;
                case Lanes.Wide128:
                    Spread<TK, Vector128<float>, F128>(ref Kernel, First, N, Work);
                    break;
                default:
                    Spread<TK, float, F1>(ref Kernel, First, N, Work);
                    break;
            }
        }

        public void Redirect(ref Scratch scratch) => Kernel.Redirect(ref scratch);
    }

    // The elements [first, n) in chunks over the threads where they pass the cut-over (any split gives the same bits: each
    // element its own). Compiled optimized from its first call, as its chunks' body is: a width's kernel is measured
    // against the others on its first calls (ChosenLanes), and unoptimized code there times the vector widths' operations
    // as calls.
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Spread<TK, TV, TF>(ref TK kernel, long first, long n, int work) where TK : struct, IMap where TV : unmanaged where TF : struct, IFloats<TV>
    {
        int threads = ComputeResources.MaxCpuThreads;
        long count = n - first, total = count * Math.Max(1, work);
        if (threads <= 1 || total < ParallelThreshold)
        {
            Steps<TK, TV, TF>(ref kernel, (nint)first, (nint)n);
            return;
        }

        long grain = Math.Max(64, ParallelThreshold / Math.Max(1, work) / 2);
        int chunks = (int)Math.Clamp(count / grain, 1, threads * 4L);
        int size = (int)((count + chunks - 1) / chunks + 63 & ~63L);
        var body = new MapBody<TK, TV, TF> { Kernel = kernel, First = (nint)first };
        CpuRanges.Run((int)((count + size - 1) / size), size, (int)count, threads, ref body);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Steps<TK, TV, TF>(ref TK kernel, nint first, nint last) where TK : struct, IMap where TV : unmanaged where TF : struct, IFloats<TV>
    {
        nint i = first;
        if (TF.Count > 1)
        {
            for (; i + TF.Count <= last; i += TF.Count)
            {
                kernel.Step<TV, TF>(i);
            }
        }

        for (; i < last; i++)
        {
            kernel.Step<float, F1>(i);
        }
    }

    private struct MapBody<TK, TV, TF> : IRangeBody where TK : struct, IMap where TV : unmanaged where TF : struct, IFloats<TV>
    {
        public TK Kernel;
        public nint First;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last) => Steps<TK, TV, TF>(ref Kernel, First + first, First + last);
    }

    // ------------------------------------------------------------------ kernels a row at a time

    /// <summary>A kernel a row at a time (a row's reductions and its outputs), and its outputs moved to scratch for measuring.</summary>
    internal interface IRowKernel
    {
        void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>;

        void Redirect(ref Scratch scratch);
    }

    // `kernel` over rows [first, rows) of `cols` by the width measured for (operation, the classes of the rows and cols);
    // `work` element operations an element of a row, for the parallel cut-over. A row is one thread's (its sums in order).
    // Rows past 0 are a chunk of a larger run (typed inputs read a chunk of rows at a time): the kernel indexes them as in
    // the whole run.
    internal static void Rows<TK>(string operation, TK kernel, int rows, int cols, int work = 4, int first = 0) where TK : struct, IRowKernel
    {
        if (rows <= first)
        {
            return;
        }

        var run = new RowRun<TK> { Kernel = kernel, First = first, Count = rows, Cost = (long)Math.Max(1, cols) * Math.Max(1, work) };
        run.Run(ChosenLanes(operation, SizeClass(rows - first), SizeClass(cols), ref run));
    }

    private struct RowRun<TK> : ILanesRun where TK : struct, IRowKernel
    {
        public TK Kernel;
        public int First, Count;
        public long Cost;

        public void Run(Lanes lanes)
        {
            switch (lanes)
            {
                case Lanes.Wide512:
                    SpreadRows<TK, Vector512<float>, F512>(ref Kernel, First, Count, Cost);
                    break;
                case Lanes.Wide256:
                    SpreadRows<TK, Vector256<float>, F256>(ref Kernel, First, Count, Cost);
                    break;
                case Lanes.Wide128:
                    SpreadRows<TK, Vector128<float>, F128>(ref Kernel, First, Count, Cost);
                    break;
                default:
                    SpreadRows<TK, float, F1>(ref Kernel, First, Count, Cost);
                    break;
            }
        }

        public void Redirect(ref Scratch scratch) => Kernel.Redirect(ref scratch);
    }

    // (Compiled optimized from its first call, as Spread is, and every row kernel's Row.)
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void SpreadRows<TK, TV, TF>(ref TK kernel, int first, int end, long cost) where TK : struct, IRowKernel where TV : unmanaged where TF : struct, IFloats<TV>
    {
        int threads = ComputeResources.MaxCpuThreads, rows = end - first;
        if (threads <= 1 || rows == 1 || rows * cost < ParallelThreshold)
        {
            for (int r = first; r < end; r++)
            {
                kernel.Row<TV, TF>(r);
            }

            return;
        }

        int perChunk = (int)Math.Clamp(ParallelThreshold / 2 / Math.Max(1, cost), 1, rows);
        int chunks = Math.Clamp((rows + perChunk - 1) / perChunk, 1, threads * 4);
        int size = (rows + chunks - 1) / chunks;
        var body = new RowBody<TK, TV, TF> { Kernel = kernel, First = first };
        CpuRanges.Run((rows + size - 1) / size, size, rows, threads, ref body);
    }

    private struct RowBody<TK, TV, TF> : IRangeBody where TK : struct, IRowKernel where TV : unmanaged where TF : struct, IFloats<TV>
    {
        public TK Kernel;
        public int First;

        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Run(int first, int last)
        {
            for (int r = First + first; r < First + last; r++)
            {
                Kernel.Row<TV, TF>(r);
            }
        }
    }

    // ------------------------------------------------------------------ maps over a row's elements

    /// <summary>A map over elements [i, i + TF.Count) of a row (inside a row kernel).</summary>
    internal interface IRowMap
    {
        void Step<TV, TF>(nint i) where TV : unmanaged where TF : struct, IFloats<TV>;
    }

    // `map` over [0, n) of a row by the row kernel's width, the tail a float at a time.
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static void RowMap<TV, TF, TM>(ref TM map, nint n) where TV : unmanaged where TF : struct, IFloats<TV> where TM : struct, IRowMap
    {
        nint i = 0;
        if (TF.Count > 1)
        {
            for (; i + TF.Count <= n; i += TF.Count)
            {
                map.Step<TV, TF>(i);
            }
        }

        for (; i < n; i++)
        {
            map.Step<float, F1>(i);
        }
    }

    // A host pointer to a storage's floats (its array pinned by the caller's fixed statement).
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    internal static ref float At(Storage s) => ref MemoryMarshal.GetArrayDataReference(D(s));
}
