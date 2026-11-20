// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Adad.Devices.Cuda;

namespace Idrak.Adad.Measure;

/// <summary>One measurement: what was measured, its candidates' median time ratios to the default's, the confirmation, the choice.</summary>
public sealed record Measured(string What, int[] Candidates, float[] Medians, int Fallback, float Confirmation, int Chosen, bool Cold);

/// <summary>
/// Measures candidates on a CUDA device as the proof's Tune did (copied): each candidate's first run outside the timing
/// (loads its kernels, settles the choices it makes itself), then all of them in turns until the GPU's speed has settled
/// (TuneTiming.Warm), then each timed in pairs with the default (TuneTiming.Choose). Warm timings run a candidate back to
/// back (as many runs as take about 0.2 ms, at most 16); cold timings run it alone after a read of twice the L2 cache
/// (the read not timed), at most 4 runs. Every measurement is kept in <see cref="Log"/>.
/// </summary>
public sealed class Tuner(CudaDevice device)
{
    private CudaKernel? _flush;
    private DeviceBuffer _flushBlock, _flushSums;
    private long _flushFloats;
    private bool _measuring;

    /// <summary>Every measurement made, in order.</summary>
    public List<Measured> Log { get; } = [];

    /// <summary>Whether a measurement may run now (not inside another one's timing).</summary>
    public bool CanMeasure => !_measuring;

    /// <summary>The candidate of <paramref name="candidates"/> to keep; <paramref name="fallback"/> is the default (the reference).</summary>
    public int Measure(string what, ReadOnlySpan<int> candidates, int fallback, Action<int> run, bool cold)
    {
        if (candidates.Length == 1)
        {
            return candidates[0];
        }

        var list = candidates.ToArray();
        foreach (int c in list)
        {
            run(c);                                                        // first runs: kernels loaded, inner choices settled
        }

        bool flush = cold && EnsureFlush();
        _measuring = true;
        try
        {
            TuneTiming.Warm(() =>
            {
                device.TimeStart();
                foreach (int c in list)
                {
                    run(c);
                }

                return device.TimeEnd();
            });

            var repeats = new int[list.Length];
            for (int i = 0; i < list.Length; i++)
            {
                repeats[i] = TuneTiming.TimingRepeats(TimeRuns(list[i], 1, run, flush), flush);
            }

            var medians = new float[list.Length];
            var confirmation = new float[1];
            int chosen = list[TuneTiming.Choose(list, fallback, i => TimeRuns(list[i], repeats[i], run, flush) / repeats[i], medians: medians, confirmation: confirmation)];
            Log.Add(new Measured(what, list, medians, fallback, confirmation[0], chosen, flush));
            return chosen;
        }
        finally
        {
            _measuring = false;
        }
    }

    /// <summary>Milliseconds of <paramref name="count"/> runs of one candidate: back to back, or each alone after the L2 flush (<paramref name="cold"/>).</summary>
    public float TimeRuns(int candidate, int count, Action<int> run, bool cold)
    {
        if (cold && EnsureFlush())
        {
            float total = 0f;
            for (int i = 0; i < count; i++)
            {
                Flush();
                device.TimeStart();
                run(candidate);
                total += device.TimeEnd();
            }

            return total;
        }

        device.TimeStart();
        for (int i = 0; i < count; i++)
        {
            run(candidate);
        }

        return device.TimeEnd();
    }

    /// <summary>Reads twice the L2 cache's bytes (its sums written apart), so the next run finds none of its operands there.</summary>
    public void Flush()
    {
        if (!EnsureFlush())
        {
            return;
        }

        uint blocks = (uint)Math.Min((_flushFloats / 4 + 255) / 256, Math.Max(1, device.Multiprocessors) * 8);
        device.Launch(_flush!, blocks, 1, 1, 256, [_flushBlock.Pointer, _flushSums.Pointer, (ulong)(_flushFloats / 4)]);
    }

    private bool EnsureFlush()
    {
        if (_flush is not null)
        {
            return true;
        }

        _flushFloats = TuneTiming.L2FlushFloats(device.L2Bytes);
        if (_flushFloats <= 0)
        {
            return false;
        }

        var kernels = device.LoadModule(FlushModule, ["flush_read"], static _ => 0, out string reason, out _)
            ?? throw new CudaException($"the L2 flush kernel: {reason}");
        _flush = kernels[0];
        _flushBlock = device.Allocate(_flushFloats * sizeof(float));
        _flushSums = device.Allocate((long)Math.Max(1, device.Multiprocessors) * 8 * 256 * sizeof(float));
        device.Fill(_flushBlock, 0, _flushFloats);
        return true;
    }

    // Each thread sums its float4s of the block (grid-stride) and writes its sum: the reads cannot be dropped.
    private const string FlushModule = """
        .version 7.0
        .target sm_80
        .address_size 64
        .visible .entry flush_read(.param .u64 p_x, .param .u64 p_out, .param .u32 p_count4)
        {
            .reg .pred %p<2>;
            .reg .b32 %r<8>;
            .reg .f32 %f<8>;
            .reg .b64 %rd<8>;
            ld.param.u64 %rd1, [p_x];
            ld.param.u64 %rd2, [p_out];
            ld.param.u32 %r1, [p_count4];
            mov.u32 %r2, %ctaid.x;
            mov.u32 %r3, %tid.x;
            mad.lo.u32 %r2, %r2, 256, %r3;
            mov.u32 %r4, %nctaid.x;
            shl.b32 %r4, %r4, 8;
            mov.u32 %r5, %r2;
            mov.f32 %f0, 0f00000000;
        LOOP:
            setp.ge.u32 %p0, %r5, %r1;
            @%p0 bra DONE;
            mul.wide.u32 %rd3, %r5, 16;
            add.u64 %rd3, %rd3, %rd1;
            ld.global.cg.v4.f32 {%f1, %f2, %f3, %f4}, [%rd3];
            add.f32 %f0, %f0, %f1;
            add.f32 %f0, %f0, %f2;
            add.f32 %f0, %f0, %f3;
            add.f32 %f0, %f0, %f4;
            add.u32 %r5, %r5, %r4;
            bra LOOP;
        DONE:
            mul.wide.u32 %rd4, %r2, 4;
            add.u64 %rd4, %rd4, %rd2;
            st.global.f32 [%rd4], %f0;
            ret;
        }
        """;

    /// <summary>A measurement's line for the choice log.</summary>
    public static string Describe(Measured m) => string.Create(CultureInfo.InvariantCulture,
        $"{m.What} ({(m.Cold ? "cold L2" : "warm")}): {string.Join(", ", m.Candidates.Select((c, i) => $"{c}={m.Medians[i]:F3}"))} (time / default {m.Fallback}'s){(float.IsNaN(m.Confirmation) ? "" : $", timed again {m.Confirmation:F3}")} -> {m.Chosen}");
}
