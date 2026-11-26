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

    // A choice of width: by the operation, its shape's classes and the vector paths setting.
    private readonly record struct LanesKey(string Operation, int A, int B, CpuVectorPaths Vectors);

    private static readonly ConcurrentDictionary<LanesKey, Lanes> s_lanesChosen = new();

    /// <summary>Forgets every measured choice of this process (the widths and the described lane's), so the next call measures or reads the machine's file again (tests).</summary>
    internal static void ForgetChoices()
    {
        s_lanesChosen.Clear();
        StoredLanes.Clear();
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
        s_lanesChosen.TryGetValue(new LanesKey(operation, a, b, ComputeResources.CpuVectors), out var lanes) ? lanes
            : MeasureLanes(new LanesKey(operation, a, b, ComputeResources.CpuVectors), ref run);

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
        string fileKey = string.Create(CultureInfo.InvariantCulture, $"lanes {shape} {key.Vectors}");
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

    // `kernel` over [0, n) by the width measured for (operation, n's class); `work` element operations an element, for the
    // parallel cut-over.
    internal static void Elements<TK>(string operation, TK kernel, long n, int work = 1) where TK : struct, IMap
    {
        if (n <= 0)
        {
            return;
        }

        var run = new MapRun<TK> { Kernel = kernel, N = n, Work = work };
        run.Run(ChosenLanes(operation, SizeClass(n), 0, ref run));
    }

    private struct MapRun<TK> : ILanesRun where TK : struct, IMap
    {
        public TK Kernel;
        public long N;
        public int Work;

        public void Run(Lanes lanes)
        {
            switch (lanes)
            {
                case Lanes.Wide512:
                    Spread<TK, Vector512<float>, F512>(ref Kernel, N, Work);
                    break;
                case Lanes.Wide256:
                    Spread<TK, Vector256<float>, F256>(ref Kernel, N, Work);
                    break;
                case Lanes.Wide128:
                    Spread<TK, Vector128<float>, F128>(ref Kernel, N, Work);
                    break;
                default:
                    Spread<TK, float, F1>(ref Kernel, N, Work);
                    break;
            }
        }

        public void Redirect(ref Scratch scratch) => Kernel.Redirect(ref scratch);
    }

    // The elements in chunks over the threads where they pass the cut-over (any split gives the same bits: each element its own).
    private static void Spread<TK, TV, TF>(ref TK kernel, long n, int work) where TK : struct, IMap where TV : unmanaged where TF : struct, IFloats<TV>
    {
        int threads = ComputeResources.MaxCpuThreads;
        long total = n * Math.Max(1, work);
        if (threads <= 1 || total < ParallelThreshold)
        {
            Steps<TK, TV, TF>(ref kernel, 0, (nint)n);
            return;
        }

        long grain = Math.Max(64, ParallelThreshold / Math.Max(1, work) / 2);
        int chunks = (int)Math.Clamp(n / grain, 1, threads * 4L);
        int size = (int)((n + chunks - 1) / chunks + 63 & ~63L);
        var body = new MapBody<TK, TV, TF> { Kernel = kernel };
        CpuRanges.Run((int)((n + size - 1) / size), size, (int)n, threads, ref body);
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

        public void Run(int first, int last) => Steps<TK, TV, TF>(ref Kernel, first, last);
    }

    // ------------------------------------------------------------------ kernels a row at a time

    /// <summary>A kernel a row at a time (a row's reductions and its outputs), and its outputs moved to scratch for measuring.</summary>
    internal interface IRowKernel
    {
        void Row<TV, TF>(int r) where TV : unmanaged where TF : struct, IFloats<TV>;

        void Redirect(ref Scratch scratch);
    }

    // `kernel` over rows [0, rows) of `cols` by the width measured for (operation, the classes of rows and cols); `work`
    // element operations an element of a row, for the parallel cut-over. A row is one thread's (its sums in order).
    internal static void Rows<TK>(string operation, TK kernel, int rows, int cols, int work = 4) where TK : struct, IRowKernel
    {
        if (rows <= 0)
        {
            return;
        }

        var run = new RowRun<TK> { Kernel = kernel, Count = rows, Cost = (long)Math.Max(1, cols) * Math.Max(1, work) };
        run.Run(ChosenLanes(operation, SizeClass(rows), SizeClass(cols), ref run));
    }

    private struct RowRun<TK> : ILanesRun where TK : struct, IRowKernel
    {
        public TK Kernel;
        public int Count;
        public long Cost;

        public void Run(Lanes lanes)
        {
            switch (lanes)
            {
                case Lanes.Wide512:
                    SpreadRows<TK, Vector512<float>, F512>(ref Kernel, Count, Cost);
                    break;
                case Lanes.Wide256:
                    SpreadRows<TK, Vector256<float>, F256>(ref Kernel, Count, Cost);
                    break;
                case Lanes.Wide128:
                    SpreadRows<TK, Vector128<float>, F128>(ref Kernel, Count, Cost);
                    break;
                default:
                    SpreadRows<TK, float, F1>(ref Kernel, Count, Cost);
                    break;
            }
        }

        public void Redirect(ref Scratch scratch) => Kernel.Redirect(ref scratch);
    }

    private static void SpreadRows<TK, TV, TF>(ref TK kernel, int rows, long cost) where TK : struct, IRowKernel where TV : unmanaged where TF : struct, IFloats<TV>
    {
        int threads = ComputeResources.MaxCpuThreads;
        if (threads <= 1 || rows == 1 || rows * cost < ParallelThreshold)
        {
            for (int r = 0; r < rows; r++)
            {
                kernel.Row<TV, TF>(r);
            }

            return;
        }

        int perChunk = (int)Math.Clamp(ParallelThreshold / 2 / Math.Max(1, cost), 1, rows);
        int chunks = Math.Clamp((rows + perChunk - 1) / perChunk, 1, threads * 4);
        int size = (rows + chunks - 1) / chunks;
        var body = new RowBody<TK, TV, TF> { Kernel = kernel };
        CpuRanges.Run((rows + size - 1) / size, size, rows, threads, ref body);
    }

    private struct RowBody<TK, TV, TF> : IRangeBody where TK : struct, IRowKernel where TV : unmanaged where TF : struct, IFloats<TV>
    {
        public TK Kernel;

        public void Run(int first, int last)
        {
            for (int r = first; r < last; r++)
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
