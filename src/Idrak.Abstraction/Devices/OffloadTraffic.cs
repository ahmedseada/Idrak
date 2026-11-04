// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Globalization;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// What crossed the bus between system memory and a device for a model's weights and activations (plan 37, W0), counted
/// since the process started: the offloaded weights a device copied in before a layer computed (staged, and prefetched in
/// the background), the offloaded weights its kernels read in place over the bus (mapped: no room to stage them), the
/// layers that staged or read mapped weights, and the activations that crossed between layers placed on different devices
/// (a weight placement's split). The devices and layers count as they go (an interlocked add each); read two snapshots
/// (<see cref="Read"/>) around a run and take the difference (<see cref="OffloadTrafficCounts.Since"/>).
/// </summary>
public static class OffloadTraffic
{
    private static long s_staged, s_prefetched, s_mapped, s_layersStaged, s_layersMapped, s_boundary, s_crossings;
    private static readonly ConcurrentDictionary<string, long> s_mappedLayers = new(StringComparer.Ordinal);

    /// <summary>The counts so far.</summary>
    public static OffloadTrafficCounts Read() => new(Interlocked.Read(ref s_staged), Interlocked.Read(ref s_prefetched), Interlocked.Read(ref s_mapped),
        Interlocked.Read(ref s_layersStaged), Interlocked.Read(ref s_layersMapped), Interlocked.Read(ref s_boundary), Interlocked.Read(ref s_crossings),
        new Dictionary<string, long>(s_mappedLayers, StringComparer.Ordinal));

    /// <summary>The bytes read in place so far (<see cref="OffloadTrafficCounts.MappedBytes"/>), without a snapshot: a caller tells whether a call read any.</summary>
    public static long MappedBytes => Interlocked.Read(ref s_mapped);

    /// <summary>A device copied <paramref name="bytes"/> of an offloaded weight to its memory before a layer computed (not in the background).</summary>
    public static void CountStaged(long bytes) => Interlocked.Add(ref s_staged, bytes);

    /// <summary>A device started copying <paramref name="bytes"/> of an offloaded weight to its memory in the background (the next layer's).</summary>
    public static void CountPrefetched(long bytes) => Interlocked.Add(ref s_prefetched, bytes);

    /// <summary>A layer's kernels read <paramref name="bytes"/> of an offloaded weight in place, over the bus (it could not be staged).</summary>
    public static void CountMapped(long bytes) => Interlocked.Add(ref s_mapped, bytes);

    /// <summary>
    /// A layer computed with offloaded weights: <paramref name="staged"/> when some were copied in (now or in the
    /// background), <paramref name="mapped"/> when some were read in place; <paramref name="layer"/> names it in
    /// <see cref="OffloadTrafficCounts.MappedLayers"/>.
    /// </summary>
    public static void CountLayer(string layer, bool staged, bool mapped)
    {
        if (staged)
        {
            Interlocked.Increment(ref s_layersStaged);
        }

        if (mapped)
        {
            Interlocked.Increment(ref s_layersMapped);
            s_mappedLayers.AddOrUpdate(layer, 1, static (_, n) => n + 1);
        }
    }

    /// <summary>An activation of <paramref name="bytes"/> crossed between layers placed on different devices.</summary>
    public static void CountBoundary(long bytes)
    {
        Interlocked.Add(ref s_boundary, bytes);
        Interlocked.Increment(ref s_crossings);
    }
}

/// <summary>A snapshot of <see cref="OffloadTraffic"/>, or the difference of two (<see cref="Since"/>).</summary>
/// <param name="StagedBytes">Offloaded weights copied to the device before a layer computed.</param>
/// <param name="PrefetchedBytes">Offloaded weights copied to the device in the background.</param>
/// <param name="MappedBytes">Offloaded weights the device's kernels read in place over the bus.</param>
/// <param name="LayersStaged">Layer passes that copied offloaded weights in.</param>
/// <param name="LayersMapped">Layer passes that read offloaded weights in place.</param>
/// <param name="BoundaryBytes">Activations that crossed between layers placed on different devices.</param>
/// <param name="BoundaryCrossings">Those crossings.</param>
/// <param name="MappedLayers">The layers that read offloaded weights in place, by name, with how many passes did.</param>
public sealed record OffloadTrafficCounts(long StagedBytes, long PrefetchedBytes, long MappedBytes, long LayersStaged, long LayersMapped, long BoundaryBytes,
    long BoundaryCrossings, IReadOnlyDictionary<string, long> MappedLayers)
{
    /// <summary>Every byte that crossed the bus: weights staged, prefetched and read in place, and activations at boundaries.</summary>
    public long BusBytes => StagedBytes + PrefetchedBytes + MappedBytes + BoundaryBytes;

    /// <summary>Whether nothing crossed.</summary>
    public bool IsEmpty => BusBytes == 0 && LayersStaged == 0 && LayersMapped == 0 && BoundaryCrossings == 0;

    /// <summary>What crossed after <paramref name="before"/> (an earlier snapshot).</summary>
    public OffloadTrafficCounts Since(OffloadTrafficCounts before)
    {
        ArgumentNullException.ThrowIfNull(before);
        var layers = new Dictionary<string, long>(StringComparer.Ordinal);
        foreach (var (name, count) in MappedLayers)
        {
            long delta = count - before.MappedLayers.GetValueOrDefault(name);
            if (delta > 0)
            {
                layers[name] = delta;
            }
        }

        return new OffloadTrafficCounts(StagedBytes - before.StagedBytes, PrefetchedBytes - before.PrefetchedBytes, MappedBytes - before.MappedBytes,
            LayersStaged - before.LayersStaged, LayersMapped - before.LayersMapped, BoundaryBytes - before.BoundaryBytes, BoundaryCrossings - before.BoundaryCrossings, layers);
    }

    /// <summary>
    /// One line for people over <paramref name="tokens"/> tokens (a run's passes: the prompt's and each generated token's):
    /// "bus: 12.4 GB, 1.03 GB a token (staged 0.9 GB, prefetched 0.1 GB, read in place 0.03 GB; activations 8.2 KB);
    /// 180 layer passes staged, 4 read in place (Linear(4096 -> 14336) ×4)".
    /// </summary>
    public string Describe(int tokens)
    {
        int n = Math.Max(1, tokens);
        string line = string.Create(CultureInfo.InvariantCulture,
            $"bus: {Bytes(BusBytes)}, {Bytes(BusBytes / n)} a token (weights staged {Bytes(StagedBytes)}, prefetched {Bytes(PrefetchedBytes)}, read in place {Bytes(MappedBytes)}; "
            + $"activations {Bytes(BoundaryBytes)} in {BoundaryCrossings} crossings); {LayersStaged / (double)n:0.#} layers staged a token, {LayersMapped / (double)n:0.#} read in place a token");
        if (MappedLayers.Count > 0)
        {
            line += " (" + string.Join(", ", MappedLayers.OrderByDescending(p => p.Value).Take(4).Select(p => $"{p.Key} ×{p.Value}")) + (MappedLayers.Count > 4 ? ", …" : "") + ")";
        }

        return line;
    }

    /// <summary>The counts as JSON (idrak run's "bus"), with the per-token figures over <paramref name="tokens"/> tokens.</summary>
    public JsonObject ToJson(int tokens)
    {
        int n = Math.Max(1, tokens);
        var layers = new JsonObject();
        foreach (var (name, count) in MappedLayers.OrderByDescending(p => p.Value))
        {
            layers[name] = count;
        }

        return new JsonObject
        {
            ["bytes"] = BusBytes,
            ["bytes_per_token"] = BusBytes / n,
            ["staged_bytes"] = StagedBytes,
            ["prefetched_bytes"] = PrefetchedBytes,
            ["mapped_bytes"] = MappedBytes,
            ["boundary_bytes"] = BoundaryBytes,
            ["boundary_crossings"] = BoundaryCrossings,
            ["layers_staged"] = LayersStaged,
            ["layers_staged_per_token"] = Math.Round(LayersStaged / (double)n, 2),
            ["layers_mapped"] = LayersMapped,
            ["mapped_layers"] = layers,
            ["tokens"] = tokens,
        };
    }

    private static string Bytes(long bytes) => bytes switch
    {
        >= 1_000_000_000 => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e9:0.##} GB"),
        >= 1_000_000 => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e6:0.#} MB"),
        >= 1_000 => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e3:0.#} KB"),
        _ => string.Create(CultureInfo.InvariantCulture, $"{bytes} B"),
    };
}
