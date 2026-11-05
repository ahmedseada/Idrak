// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Models.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// The experts of one <see cref="MixtureOfExperts"/> layer placed apart (plan 37, R1, <see cref="ExpertPlacements"/>). At
/// most <see cref="Capacity"/> of them live on <see cref="Device"/> (resident); the others live on <see cref="Host"/>
/// (system memory) or, with a lazy load, are not read yet.
/// <para>
/// When the router picks an expert that is not resident, it is copied in, read from the checkpoint the first time.
/// When the cache is full, the resident expert the routing has picked least so far (the least recently picked among
/// equals) goes back to the host first. So the experts that stay are learned from the routing's counts at run time.
/// With <see cref="ColdOnCpu"/>, a cold expert's rows are computed by the CPU from system memory instead, as llama.cpp's
/// <c>--cpu-moe</c> does. A cold expert is still copied in once it is picked more often than the least used resident
/// one, or while the cache has room.
/// </para>
/// <para>
/// Every expert keeps its own weights, moved whole between the two memories (<see cref="Module.To(Device)"/>), so their
/// values never change: a layer gives the same results bit for bit whatever is resident. A device routing step needs
/// every expert's weights where it runs, so the layer routes on the host while the cache does not hold every expert
/// (<see cref="HoldsAll"/>). Such a step is counted (<see cref="HostRoutings"/>) and not recorded in a graph.
/// </para>
/// </summary>
public sealed class ExpertCache
{
    private readonly FeedForward[] _experts;
    private readonly Func<int, Device, FeedForward>? _build;
    private readonly bool[] _resident, _built;
    private readonly long[] _uses, _lastUse;
    private long _clock;

    internal ExpertCache(FeedForward[] experts, Device device, Device host, int capacity, bool coldOnCpu, Func<int, Device, FeedForward>? build)
    {
        ArgumentNullException.ThrowIfNull(experts);
        ArgumentNullException.ThrowIfNull(device);
        ArgumentNullException.ThrowIfNull(host);
        (_experts, Device, Host, Capacity, ColdOnCpu, _build) = (experts, device, host, Math.Clamp(capacity, 1, Math.Max(1, experts.Length)), coldOnCpu, build);
        _resident = new bool[experts.Length];
        _built = new bool[experts.Length];
        _uses = new long[experts.Length];
        _lastUse = new long[experts.Length];
        for (int e = 0; e < experts.Length; e++)
        {
            _built[e] = build is null;
            _resident[e] = build is null && experts[e].WeightsDevice == device;
        }

        ExpertTraffic.Track(this);
    }

    /// <summary>Where resident experts live and compute.</summary>
    public Device Device { get; private set; }

    /// <summary>Where the other experts live (system memory).</summary>
    public Device Host { get; }

    /// <summary>The experts resident at most.</summary>
    public int Capacity { get; }

    /// <summary>Whether a cold expert's rows are computed by the CPU instead of copied in.</summary>
    public bool ColdOnCpu { get; }

    /// <summary>Whether experts are read from the checkpoint the first time the router picks them (a lazy load).</summary>
    public bool Lazy => _build is not null;

    /// <summary>The layer's experts.</summary>
    public int Experts => _experts.Length;

    /// <summary>The experts resident on <see cref="Device"/> now.</summary>
    public int Resident => _resident.Count(r => r);

    /// <summary>The experts read so far (every one, unless the load is lazy).</summary>
    public int Built => _built.Count(b => b);

    /// <summary>Whether every expert is resident, so the layer routes on the device and a step through it can be recorded.</summary>
    public bool HoldsAll => Resident == _experts.Length;

    /// <summary>Picks of a resident expert.</summary>
    public long Hits { get; private set; }

    /// <summary>Picks of a cold expert that copied it in (or read it, the first time).</summary>
    public long Misses { get; private set; }

    /// <summary>The bytes of experts copied to the device (or read there from the checkpoint).</summary>
    public long CopiedBytes { get; private set; }

    /// <summary>The bytes of experts moved back to the host to make room.</summary>
    public long EvictedBytes { get; private set; }

    /// <summary>Steps this layer routed on the host because not every expert was resident.</summary>
    public long HostRoutings { get; private set; }

    /// <summary>Rows of cold experts the CPU computed (<see cref="ColdOnCpu"/>).</summary>
    public long CpuRows { get; private set; }

    /// <summary>How many times the routing has picked expert <paramref name="e"/> so far.</summary>
    public long Uses(int e) => _uses[e];

    /// <summary>Whether expert <paramref name="e"/> is resident on <see cref="Device"/>.</summary>
    public bool IsResident(int e) => _resident[e];

    /// <summary>Whether expert <paramref name="e"/>'s weights have been read (always, unless the load is lazy).</summary>
    public bool IsBuilt(int e) => _built[e];

    // A step's choices counted (the routing's picks, which decide what stays).
    internal void Count(ReadOnlySpan<int> choices)
    {
        _clock++;
        foreach (int e in choices)
        {
            _uses[e]++;
            _lastUse[e] = _clock;
        }
    }

    // A step routed on the host because not every expert was resident.
    internal void CountHostRouting()
    {
        HostRoutings++;
        ExpertTraffic.CountHostRouting();
    }

    // Rows a cold expert computed on the CPU.
    internal void CountCpuRows(int rows)
    {
        CpuRows += rows;
        ExpertTraffic.CountCpuRows(rows);
    }

    /// <summary>
    /// Makes every expert of <paramref name="wave"/> resident (at most <see cref="Capacity"/> of them): each cold one is
    /// copied in (read the first time), after the least used resident expert outside the wave goes back to the host when
    /// the cache is full.
    /// </summary>
    internal void Ensure(IReadOnlyList<int> wave)
    {
        if (wave.Count > Capacity)
        {
            throw new InvalidOperationException($"{wave.Count} experts asked at once of a cache of {Capacity}.");
        }

        foreach (int e in wave)
        {
            if (_resident[e])
            {
                Hits++;
                ExpertTraffic.CountHit();
            }
        }

        foreach (int e in wave)
        {
            if (!_resident[e])
            {
                if (Resident >= Capacity)
                {
                    Evict(Victim(wave));
                }

                Load(e);
            }
        }
    }

    /// <summary>
    /// With <see cref="ColdOnCpu"/>: copies in the cold experts of <paramref name="needed"/> while the cache has room, and
    /// those picked more often than the least used resident expert outside <paramref name="needed"/> in its place.
    /// </summary>
    internal void Promote(IReadOnlyList<int> needed)
    {
        foreach (int e in needed.OrderByDescending(e => _uses[e]).ThenBy(e => e))
        {
            if (_resident[e])
            {
                continue;
            }

            if (Resident < Capacity)
            {
                Load(e);
                continue;
            }

            int victim = Victim(needed, allowNone: true);
            if (victim >= 0 && _uses[e] > _uses[victim])
            {
                Evict(victim);
                Load(e);
            }
        }
    }

    // Expert e on the host (read there the first time, for the CPU to compute it).
    internal FeedForward OnHost(int e)
    {
        if (!_built[e])
        {
            Replace(e, _build!(e, Host));
        }

        return _experts[e];
    }

    // The layer moved to `device` (a split's layer moving between the device and the CPU): the resident experts go with it.
    internal void Retarget(Device device)
    {
        for (int e = 0; e < _experts.Length; e++)
        {
            if (_resident[e])
            {
                _experts[e].To(device);
            }
        }

        Device = device;
    }

    // The resident expert to give back: the least picked outside `keep` (the least recently picked among equals; the lowest
    // index among those), or -1 with `allowNone` when every resident one is kept.
    private int Victim(IReadOnlyList<int> keep, bool allowNone = false)
    {
        int victim = -1;
        for (int e = 0; e < _experts.Length; e++)
        {
            if (!_resident[e] || keep.Contains(e))
            {
                continue;
            }

            if (victim < 0 || _uses[e] < _uses[victim] || _uses[e] == _uses[victim] && _lastUse[e] < _lastUse[victim])
            {
                victim = e;
            }
        }

        return victim >= 0 || allowNone ? victim : throw new InvalidOperationException("Every resident expert is in use: the cache cannot make room.");
    }

    private void Load(int e)
    {
        if (!_built[e])
        {
            Replace(e, _build!(e, Device));
        }
        else
        {
            _experts[e].To(Device);
        }

        long bytes = Bytes(_experts[e]);
        _resident[e] = true;
        Misses++;
        CopiedBytes += bytes;
        ExpertTraffic.CountMiss(bytes);
    }

    private void Evict(int e)
    {
        long bytes = Bytes(_experts[e]);
        _experts[e].To(Host);
        _resident[e] = false;
        EvictedBytes += bytes;
        ExpertTraffic.CountEvicted(bytes);
    }

    private void Replace(int e, FeedForward built)
    {
        var placeholder = _experts[e];
        built.Name = $"experts.{e}";
        _experts[e] = built;
        _built[e] = true;
        placeholder.Dispose();                                                           // its tensors hold no memory
    }

    // The bytes an expert's weights take (its parameters and packed buffers).
    internal static long Bytes(Module expert) => expert.Parameters().Concat(expert.Buffers()).Sum(t => (long)t.Storage.Length * sizeof(float));

    /// <inheritdoc />
    public override string ToString() => string.Create(CultureInfo.InvariantCulture,
        $"{Resident} of {Experts} experts on {Device} (at most {Capacity}), {Experts - Resident} on {Host}{(Lazy ? $", {Built} read" : "")}; {Hits} hits, {Misses} misses");
}

/// <summary>
/// What the experts placed apart did in this process (plan 37, R1; <see cref="ExpertCache"/>): picks of resident experts
/// (hits) and of cold ones copied in (misses), the bytes copied in and moved back, steps routed on the host because a
/// layer's experts were not all resident, rows the CPU computed for cold experts, and the experts resident and cold now.
/// </summary>
public static class ExpertTraffic
{
    private static long s_hits, s_misses, s_copied, s_evicted, s_hostRoutings, s_cpuRows;
    private static readonly List<WeakReference<ExpertCache>> s_caches = [];

    /// <summary>The counts so far, with the experts resident and cold now in the caches alive.</summary>
    public static ExpertTrafficCounts Read()
    {
        int resident = 0, cold = 0;
        lock (s_caches)
        {
            s_caches.RemoveAll(w => !w.TryGetTarget(out _));
            foreach (var weak in s_caches)
            {
                if (weak.TryGetTarget(out var cache))
                {
                    resident += cache.Resident;
                    cold += cache.Experts - cache.Resident;
                }
            }
        }

        return new ExpertTrafficCounts(Interlocked.Read(ref s_hits), Interlocked.Read(ref s_misses), Interlocked.Read(ref s_copied), Interlocked.Read(ref s_evicted),
            Interlocked.Read(ref s_hostRoutings), Interlocked.Read(ref s_cpuRows), resident, cold);
    }

    internal static void Track(ExpertCache cache)
    {
        lock (s_caches)
        {
            s_caches.Add(new WeakReference<ExpertCache>(cache));
        }
    }

    internal static void CountHit() => Interlocked.Increment(ref s_hits);

    internal static void CountMiss(long bytes)
    {
        Interlocked.Increment(ref s_misses);
        Interlocked.Add(ref s_copied, bytes);
    }

    internal static void CountEvicted(long bytes) => Interlocked.Add(ref s_evicted, bytes);

    internal static void CountHostRouting() => Interlocked.Increment(ref s_hostRoutings);

    internal static void CountCpuRows(int rows) => Interlocked.Add(ref s_cpuRows, rows);
}

/// <summary>A snapshot of <see cref="ExpertTraffic"/>, or the difference of two (<see cref="Since"/>: the experts resident and cold stay the later's).</summary>
/// <param name="Hits">Picks of a resident expert.</param>
/// <param name="Misses">Picks of a cold expert that copied it in (or read it the first time).</param>
/// <param name="CopiedBytes">Bytes of experts copied to the device.</param>
/// <param name="EvictedBytes">Bytes of experts moved back to the host.</param>
/// <param name="HostRoutings">Layer steps routed on the host because not every expert was resident.</param>
/// <param name="CpuRows">Rows of cold experts the CPU computed.</param>
/// <param name="Resident">Experts resident on their devices now.</param>
/// <param name="Cold">Experts in system memory (or not read yet) now.</param>
public sealed record ExpertTrafficCounts(long Hits, long Misses, long CopiedBytes, long EvictedBytes, long HostRoutings, long CpuRows, int Resident, int Cold)
{
    /// <summary>Whether no expert was picked through a cache.</summary>
    public bool IsEmpty => Hits == 0 && Misses == 0 && CpuRows == 0 && HostRoutings == 0;

    /// <summary>What happened after <paramref name="before"/> (an earlier snapshot); the experts resident and cold are this one's.</summary>
    public ExpertTrafficCounts Since(ExpertTrafficCounts before)
    {
        ArgumentNullException.ThrowIfNull(before);
        return new(Hits - before.Hits, Misses - before.Misses, CopiedBytes - before.CopiedBytes, EvictedBytes - before.EvictedBytes, HostRoutings - before.HostRoutings,
            CpuRows - before.CpuRows, Resident, Cold);
    }

    /// <summary>
    /// One line for people over <paramref name="tokens"/> tokens: "experts: 64 resident, 448 cold; 1,203 hits, 97 misses;
    /// 2.1 GB copied in (21 MB a token), 1.9 GB moved back; 40 layer steps routed on the host".
    /// </summary>
    public string Describe(int tokens)
    {
        int n = Math.Max(1, tokens);
        return string.Create(CultureInfo.InvariantCulture,
            $"experts: {Resident} resident, {Cold} cold; {Hits:N0} hits, {Misses:N0} misses; {Bytes(CopiedBytes)} copied in ({Bytes(CopiedBytes / n)} a token), "
            + $"{Bytes(EvictedBytes)} moved back; {HostRoutings:N0} layer steps routed on the host; {CpuRows:N0} rows computed by the CPU");
    }

    private static string Bytes(long bytes) => bytes switch
    {
        >= 1_000_000_000 => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e9:0.##} GB"),
        >= 1_000_000 => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e6:0.#} MB"),
        >= 1_000 => string.Create(CultureInfo.InvariantCulture, $"{bytes / 1e3:0.#} KB"),
        _ => string.Create(CultureInfo.InvariantCulture, $"{bytes} B"),
    };
}
