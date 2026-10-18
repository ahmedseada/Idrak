// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using Idrak.Networking.Abstractions;
using Idrak.Networking.Training;

namespace Idrak.Networking.Compression;

/// <summary>
/// "top-k" in <see cref="GradientCompressions"/>: each rank sends only the largest share (<see cref="ClusterOptions.TopK"/>)
/// of each bucket's values by magnitude, as (index, value) pairs every rank gathers; with error feedback
/// (<see cref="ClusterOptions.ErrorFeedback"/>) what a rank left out is added to its next step's values, so nothing is lost,
/// only late.
/// </summary>
internal sealed class TopKCompression : IGradientCompression
{
    public string Name => "top-k";

    public string Summary => "the largest share of each bucket's values (IDRAK_CLUSTER_TOPK) as index and value pairs; with error feedback the rest goes with the next step";

    public IGradientCompressor Create(ICollectives collectives, IReadOnlyList<Tensor> parameters, ClusterOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!(options.TopK > 0 && options.TopK <= 1))
        {
            throw new ArgumentException($"top-k sends a share of each bucket's values: set ClusterOptions.TopK (IDRAK_CLUSTER_TOPK) above 0 and at most 1, not {options.TopK}.", nameof(options));
        }

        return new TopKCompressor(collectives, parameters, options.TopK, options.ErrorFeedback);
    }
}

/// <summary>
/// A rank's top-k compressor. Per bucket (cut in blocks of 2^24 values, so an index travels exactly as float32): the
/// residual is added to this rank's values, the k largest by magnitude are found in one pass (a selection, not a sort;
/// ties go to the lower index, so every run picks alike), sent as (index, value · weight) pairs and gathered from every
/// rank, and summed into the bucket in rank order (rule 86: an order fixed by the ranks, the same on every rank); the
/// rest becomes the residual. The residual is one float per trained value, allocated once per parameter on first use.
/// </summary>
internal sealed class TopKCompressor(ICollectives collectives, IReadOnlyList<Tensor> parameters, double share, bool feedback) : IGradientCompressor
{
    private const int Block = 1 << 24;                               // float32 holds every whole number below it exactly

    private readonly HostValues _values = new(collectives);
    private readonly float[]?[] _residual = new float[]?[parameters.Count];

    public string Name => "top-k";

    /// <summary>The values a block of <paramref name="length"/> sends: the share, at least one.</summary>
    public int Kept(int length) => (int)Math.Clamp(Math.Ceiling(share * length), 1, Math.Max(1, length));

    public void Reduce(Span<float> values, float weight, ReadOnlySpan<GradientPiece> pieces)
    {
        bool own = weight != 0f;
        if (own && feedback)
        {
            foreach (var piece in pieces)
            {
                if (_residual[piece.Parameter] is { } kept)
                {
                    Add(values.Slice(piece.At, piece.Count), kept.AsSpan(piece.Offset, piece.Count));
                }
            }
        }

        for (int start = 0; start < values.Length; start += Block)
        {
            int length = Math.Min(Block, values.Length - start);
            ReduceBlock(values.Slice(start, length), start, weight, own, pieces);
        }
    }

    public void Forget(ReadOnlySpan<GradientPiece> pieces)
    {
        foreach (var piece in pieces)
        {
            _residual[piece.Parameter]?.AsSpan(piece.Offset, piece.Count).Clear();
        }
    }

    public void Dispose() => _values.Dispose();

    private void ReduceBlock(Span<float> block, int start, float weight, bool own, ReadOnlySpan<GradientPiece> pieces)
    {
        int n = block.Length, k = Kept(n), world = collectives.WorldSize, rank = collectives.Rank;
        var magnitudes = ArrayPool<float>.Shared.Rent(n);
        var pairs = ArrayPool<float>.Shared.Rent(2 * k * world);
        try
        {
            var mine = pairs.AsSpan(2 * k * rank, 2 * k);
            mine.Clear();                                            // a rank with no rows sends k zero pairs
            if (own)
            {
                for (int i = 0; i < n; i++)
                {
                    magnitudes[i] = Magnitude(block[i]);
                }

                float threshold = KthLargest(magnitudes.AsSpan(0, n), k);
                int taken = 0;
                for (int pass = 0; pass < 2 && taken < k; pass++)
                {
                    for (int i = 0; i < n && taken < k; i++)
                    {
                        float m = Magnitude(block[i]);
                        if (pass == 0 ? m > threshold : m == threshold)
                        {
                            mine[2 * taken] = i;
                            mine[(2 * taken) + 1] = block[i] * weight;
                            taken++;
                        }
                    }
                }

                if (feedback)
                {
                    // What was left out waits for the next step: the residual is the values less what was sent.
                    foreach (var piece in pieces)
                    {
                        int from = Math.Max(piece.At, start), to = Math.Min(piece.At + piece.Count, start + n);
                        if (from >= to)
                        {
                            continue;
                        }

                        var residual = _residual[piece.Parameter] ??= new float[parameters[piece.Parameter].Size];
                        int offset = piece.Offset + (from - piece.At);
                        block.Slice(from - start, to - from).CopyTo(residual.AsSpan(offset, to - from));
                    }

                    for (int j = 0; j < taken; j++)
                    {
                        int at = start + (int)mine[2 * j];
                        var piece = PieceAt(pieces, at);
                        _residual[piece.Parameter]![piece.Offset + (at - piece.At)] = 0f;
                    }
                }
            }

            _values.AllGather(pairs.AsSpan(0, 2 * k * world), 2 * k);
            block.Clear();
            for (int r = 0; r < world; r++)                          // rank order: the same sum on every rank
            {
                var theirs = pairs.AsSpan(2 * k * r, 2 * k);
                for (int j = 0; j < k; j++)
                {
                    block[(int)theirs[2 * j]] += theirs[(2 * j) + 1];
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(magnitudes);
            ArrayPool<float>.Shared.Return(pairs);
        }
    }

    // A NaN counts as the largest, so it travels and the step sees it.
    private static float Magnitude(float value) => float.IsNaN(value) ? float.PositiveInfinity : MathF.Abs(value);

    private static GradientPiece PieceAt(ReadOnlySpan<GradientPiece> pieces, int at)
    {
        int lo = 0, hi = pieces.Length - 1;
        while (lo < hi)
        {
            int mid = (lo + hi + 1) >> 1;
            if (pieces[mid].At <= at)
            {
                lo = mid;
            }
            else
            {
                hi = mid - 1;
            }
        }

        return pieces[lo];
    }

    private static void Add(Span<float> values, ReadOnlySpan<float> other)
    {
        for (int i = 0; i < values.Length; i++)
        {
            values[i] += other[i];
        }
    }

    /// <summary>The <paramref name="k"/>-th largest of <paramref name="a"/> (from 1), in one selection pass on average; <paramref name="a"/> is reordered.</summary>
    internal static float KthLargest(Span<float> a, int k)
    {
        int target = k - 1, lo = 0, hi = a.Length - 1;
        while (lo < hi)
        {
            float x = a[lo], y = a[(lo + hi) >>> 1], z = a[hi];
            float pivot = Math.Max(Math.Min(x, y), Math.Min(Math.Max(x, y), z));   // the median of three
            int i = lo, j = hi;
            while (i <= j)
            {
                while (a[i] > pivot)
                {
                    i++;
                }

                while (a[j] < pivot)
                {
                    j--;
                }

                if (i <= j)
                {
                    (a[i], a[j]) = (a[j], a[i]);
                    i++;
                    j--;
                }
            }

            if (target <= j)
            {
                hi = j;
            }
            else if (target >= i)
            {
                lo = i;
            }
            else
            {
                return a[target];
            }
        }

        return a[target];
    }
}
