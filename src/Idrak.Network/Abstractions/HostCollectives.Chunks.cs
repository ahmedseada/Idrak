// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Globalization;
using System.Numerics;

namespace Idrak.Networking.Abstractions;

// The measured chunk size (plan 23, phase C8): with ClusterOptions.ChunkBytes unset, the first collective over a real
// transport first agrees with the other ranks on the size this machine kept for the transport (the tuning cache,
// ClusterCache, "chunk" by transport name), and when they do not all hold the same one, measures it: an all-reduce of
// ClusterOptions.ChunkProbeBytes timed with each candidate chunk (powers of 4 from 64 KiB, then the whole probe in one
// message), each candidate's median of three warm calls taken as its slowest rank's (a max over the ranks, so every rank
// chooses alike), the fastest kept. Measured once per transport on a machine; a forced ChunkBytes skips it, and a virtual
// clock keeps one message a piece so predictions and replays stay exact.
internal abstract partial class HostCollectives
{
    /// <summary>The kind of entry the chunk size is kept under in the tuning cache.</summary>
    internal const string ChunkKind = "chunk";

    private const long SmallestChunk = 64 * 1024;

    /// <summary>The chunk in use (bytes): set, measured, or one message a piece.</summary>
    internal long ChunkBytes => _chunkBytes;

    /// <summary>How many chunk measurements this process ran (tests: a kept size is not measured again).</summary>
    internal static int ChunkMeasurements;

    private long MeasureChunk()
    {
        var options = Context!.Options;
        string key = Transport.Name;
        long most = Array.MaxLength / 2;
        long kept = ClusterCache.FindTuning(ChunkKind, key, options.TuningFile) is { Value: > 0 and var value } && BitOperations.IsPow2(value) && value <= most ? value : 0;
        Span<float> agree = [kept, -kept];                          // powers of two: exact in float32
        AllReduceHost(agree, ReduceOp.Max, WireType.Float32);
        if (agree[0] > 0 && agree[0] == -agree[1])
        {
            return (long)agree[0];                                  // every rank kept the same size
        }

        Interlocked.Increment(ref ChunkMeasurements);
        long probe = Math.Clamp(options.ChunkProbeBytes / sizeof(float) * sizeof(float), sizeof(float), most);
        var candidates = new List<long>();
        for (long c = Math.Min(SmallestChunk, probe); c < probe; c *= 4)
        {
            candidates.Add(c);
        }

        candidates.Add(probe);
        int count = (int)(probe / sizeof(float));
        var buffer = ArrayPool<float>.Shared.Rent(count);
        var times = new float[candidates.Count];
        Span<double> runs = stackalloc double[3];
        try
        {
            var values = buffer.AsSpan(0, count);
            for (int i = 0; i < candidates.Count; i++)
            {
                _chunkBytes = candidates[i];
                values.Clear();
                AllReduceHost(values, ReduceOp.Sum, WireType.Float32);   // warm: buffers, the pipes' first use
                for (int r = 0; r < runs.Length; r++)
                {
                    long start = Stopwatch.GetTimestamp();
                    AllReduceHost(values, ReduceOp.Sum, WireType.Float32);
                    runs[r] = Stopwatch.GetElapsedTime(start).TotalSeconds;
                }

                runs.Sort();
                times[i] = (float)runs[1];
            }

            _chunkBytes = most;                                     // the agreement below in one message
            AllReduceHost(times, ReduceOp.Max, WireType.Float32);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(buffer);
        }

        int best = 0;
        for (int i = 1; i < times.Length; i++)
        {
            if (times[i] <= times[best])
            {
                best = i;                                           // ties to the larger chunk: fewer messages
            }
        }

        var inv = CultureInfo.InvariantCulture;
        string detail = $"all-reduce of {probe.ToString(inv)} B over {WorldSize.ToString(inv)} ranks, slowest rank's median ms: "
                        + string.Join(", ", candidates.Select((c, i) => $"{c.ToString(inv)} B {(times[i] * 1e3).ToString("F3", inv)}"));
        try
        {
            ClusterCache.Keep(new ClusterTuning(ChunkKind, key, candidates[best], DateTimeOffset.Now, detail), options.TuningFile);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            // A cache that cannot be written: the size is measured again by the next process.
        }

        return candidates[best];
    }
}
