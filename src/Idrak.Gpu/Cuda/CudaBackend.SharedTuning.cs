// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// The ranks of a cluster run rank 0's measured choices (ClusterOptions.ShareTuning, IMeasuredChoicesProvider): every choice
// of CudaBackend.Tuning.cs (splits, tiles, kernel variants, the attention path and the span kernels), since several pick
// different algorithms whose results differ in the last bits. Only choices of the same GPU (its UUID), driver and library
// build are taken (ChoicesIdentity: the tuning cache's identity with the UUID); another rank keeps its own.
//
// In one process a follower shares rank 0's tables and its measuring lock (Follow): a choice neither knows yet is measured
// once, by whichever rank needs it first (the lock's re-check, as for two threads of one backend), and every rank runs it.
// A follower that measures saves the shared table, which is rank 0's for this very GPU, driver and build, to the cache file
// of that identity: the file rank 0 keeps. Across processes a rank takes rank 0's choices as the run starts (Import), and
// never writes them to its own cache file (SaveTuning leaves out what was imported and not measured again here).
internal sealed unsafe partial class CudaBackend
{
    private CudaBackend? _leader;

    // Choices taken from another process's rank 0 (Import) and their values: not written to this backend's cache file.
    private Dictionary<TuneKey, int>? _imported;

    /// <summary>What this backend's measured choices apply to: the GPU's UUID and the tuning cache's identity (device, driver, library build).</summary>
    internal string ChoicesIdentity
    {
        get
        {
            byte* bytes = stackalloc byte[16];
            string uuid;
            try
            {
                uuid = cuDeviceGetUuid(bytes, _deviceHandle) == 0 ? new Guid(new ReadOnlySpan<byte>(bytes, 16)).ToString("N") : "no-uuid";
            }
            catch (EntryPointNotFoundException)
            {
                uuid = "no-uuid";
            }

            var identity = TuningIdentity;
            return $"cuda/{uuid}/{identity.Device}/{identity.Driver}/{identity.Library}";
        }
    }

    /// <summary>The backend whose choices this one runs (a cluster rank following rank 0), or null.</summary>
    internal CudaBackend? Leader => _leader;

    // Every choice known now: measured or taken here ("t"), read from the cache file (read again; "p") and lent ("l").
    internal MeasuredChoices ExportChoices()
    {
        var entries = new Dictionary<string, string>(StringComparer.Ordinal);
        lock (_tuned)
        {
            _persisted ??= [];
            if (_leader is null && _imported is null && TuningFileLocked() is { } file)
            {
                foreach (var (key, value) in TuningCache.Load(file.Path, file.Identity))
                {
                    _persisted[key] = value;                               // what others stored since (into the table followers share)
                }
            }

            foreach (var (key, value) in _persisted ?? [])
            {
                entries["p " + Format(key)] = value.ToString(CultureInfo.InvariantCulture);
            }

            foreach (var (key, value) in _tuned)
            {
                entries["t " + Format(key)] = value.ToString(CultureInfo.InvariantCulture);
            }

            foreach (var (key, value) in _lent)
            {
                entries["l " + Format(key)] = value.ToString(CultureInfo.InvariantCulture);
            }
        }

        return new MeasuredChoices(Kind, ChoicesIdentity, Name, entries);
    }

    // Takes `choices` in place of this backend's own when measured on this GPU, driver and build (never written to its file).
    internal MeasuredChoicesResult ImportChoices(MeasuredChoices choices)
    {
        if (Refusal(choices.Kind, choices.Identity, choices.Device) is { } refused)
        {
            return refused;
        }

        var tables = new Dictionary<char, Dictionary<TuneKey, int>> { ['t'] = [], ['p'] = [], ['l'] = [] };
        foreach (var (name, value) in choices.Entries)
        {
            if (name.Length > 2 && name[1] == ' ' && tables.TryGetValue(name[0], out var table) && TryParse(name[2..], out var key)
                && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int choice))
            {
                table[key] = choice;
            }
        }

        LeaveChoices();
        lock (_tuned)
        {
            _tuned.Clear();
            _lent.Clear();
            foreach (var (key, value) in tables['t'])
            {
                _tuned[key] = value;
            }

            foreach (var (key, value) in tables['l'])
            {
                _lent[key] = value;
            }

            _persisted = tables['p'];
            _imported = new Dictionary<TuneKey, int>(tables['p']);
            foreach (var (key, value) in tables['t'])
            {
                _imported[key] = value;
            }
        }

        return new MeasuredChoicesResult(true, tables['t'].Count + tables['p'].Count, $"took rank 0's {tables['t'].Count + tables['p'].Count} choices ({choices.Device})");
    }

    // Shares `leader`'s tables and measuring lock from now on, when it runs on this GPU, driver and build.
    internal MeasuredChoicesResult FollowChoices(CudaBackend leader)
    {
        if (ReferenceEquals(leader, this))
        {
            return new MeasuredChoicesResult(true, TunedCount, "rank 0's own backend");
        }

        if (Refusal(leader.Kind, leader.ChoicesIdentity, leader.Name) is { } refused)
        {
            return refused;
        }

        var known = leader.ExportChoices();                                // loads (and reads again) the leader's file
        LeaveChoices();
        lock (leader._tuned)
        {
            _tuned = leader._tuned;
            _lent = leader._lent;
            _persisted = leader._persisted ??= [];
            _measuring = leader._measuring;
            _imported = null;
            _leader = leader;
        }

        return new MeasuredChoicesResult(true, known.Entries.Count, $"shares rank 0's choices ({leader.Name})");
    }

    // Back to this backend's own choices: read from its cache file when next needed.
    internal void LeaveChoices()
    {
        if (_leader is null && _imported is null)
        {
            return;
        }

        lock (_tuned)
        {
            _tuned = [];
            _lent = [];
            _persisted = null;
            _measuring = new Lock();
            _imported = null;
            _leader = null;
        }
    }

    // The measured choices SaveTuning writes: all but those taken from another process's rank 0 and not measured again here.
    // Called under the _tuned lock.
    private Dictionary<TuneKey, int> OwnChoicesLocked()
    {
        var own = new Dictionary<TuneKey, int>(_tuned);
        if (_imported is { } imported)
        {
            foreach (var (key, value) in imported)
            {
                if (own.TryGetValue(key, out int current) && current == value)
                {
                    own.Remove(key);
                }
            }
        }

        return own;
    }

    private MeasuredChoicesResult? Refusal(string kind, string identity, string device)
    {
        if (!string.Equals(kind, Kind, StringComparison.Ordinal))
        {
            return new MeasuredChoicesResult(false, 0, $"kept its own: rank 0 runs on another kind of device ({kind}: {device}) than {Name}");
        }

        return identity == ChoicesIdentity ? null
            : new MeasuredChoicesResult(false, 0, $"kept its own: rank 0's choices were measured on another GPU, driver or build ({device}: {identity}) than {Name}");
    }

    private static string Format(TuneKey key) =>
        string.Create(CultureInfo.InvariantCulture, $"{key.Op} {key.Variant} {key.A} {key.B} {key.C} {key.D} {key.E} {key.F}");

    private static bool TryParse(string text, out TuneKey key)
    {
        key = default;
        string[] parts = text.Split(' ');
        if (parts.Length != 8 || !Enum.TryParse(parts[0], out TuneOp op) || !Enum.IsDefined(op))
        {
            return false;
        }

        Span<int> n = stackalloc int[7];
        for (int i = 0; i < 7; i++)
        {
            if (!int.TryParse(parts[i + 1], NumberStyles.Integer, CultureInfo.InvariantCulture, out n[i]))
            {
                return false;
            }
        }

        key = new TuneKey(op, n[0], n[1], n[2], n[3], n[4], n[5], n[6]);
        return true;
    }
}

/// <summary>The CUDA backend's measured choices between the ranks of a cluster (CudaBackend.SharedTuning.cs).</summary>
internal sealed class CudaMeasuredChoices : IMeasuredChoicesProvider
{
    public string Name => "cuda";

    public string Summary => "the CUDA kernels' measured choices (splits, tiles, variants, attention paths), per GPU, driver and library build";

    public bool Handles(Backend backend) => backend is CudaBackend;

    public MeasuredChoices Export(Backend backend) => Of(backend).ExportChoices();

    public MeasuredChoicesResult Import(Backend backend, MeasuredChoices choices)
    {
        ArgumentNullException.ThrowIfNull(choices);
        return Of(backend).ImportChoices(choices);
    }

    public MeasuredChoicesResult Follow(Backend follower, Backend leader)
    {
        var mine = Of(follower);
        return leader is CudaBackend cuda ? mine.FollowChoices(cuda)
            : new MeasuredChoicesResult(false, 0, $"kept its own: rank 0 runs on another kind of device ({leader.Kind}: {leader.Name}) than {mine.Name}");
    }

    public void Leave(Backend follower) => Of(follower).LeaveChoices();

    private static CudaBackend Of(Backend backend) => backend as CudaBackend
        ?? throw new ArgumentException($"{backend?.Name ?? "null"} is not a CUDA backend.", nameof(backend));
}
