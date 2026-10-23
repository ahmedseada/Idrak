// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Gpu.Vulkan;

// The ranks of a cluster run rank 0's measured choices (ClusterOptions.ShareTuning, IMeasuredChoicesProvider): the kernel
// choices of VulkanBackend.KernelTuning.cs and the workgroup width of VulkanBackend.WidthProbe.cs, since both decide which
// kernel runs and so the last bits of a result (the attention path, the tiled against the decoding kernel and the span
// widths use online against two-pass softmax; a cooperative-matrix product sums in an order the hardware defines). The
// runtime policy and the storage memory (VulkanBackend.Tuning.cs, .StorageMemory.cs) decide how commands are submitted and
// where storages live, never what a kernel computes, so each rank keeps its own.
//
// Only choices of the same device and driver are taken (ChoicesIdentity: the tuning cache's key, by deviceUUID, driverUUID,
// driver version and power source, with the kernels' reductions, cooperative-matrix shape and version); another rank keeps
// its own. In one process a follower shares rank 0's table itself (Follow): a choice neither knows yet is measured once, by
// whichever rank needs it first (Keep: the first stored wins), so every rank runs the same kernel from then on. Across
// processes a rank takes rank 0's choices as they were when the run started (Import). Choices taken are never written to
// the taking backend's cache file; what it measures itself afterwards is, as always.
internal sealed partial class VulkanBackend
{
    // The backend whose table this one shares (Follow), and this one's own width to go back to (Leave).
    private VulkanBackend? _leader;
    private int _ownWidth;
    private bool _imported;

    /// <summary>What this backend's measured choices apply to: its device and driver, and its kernels' reductions, cooperative-matrix shape and version.</summary>
    internal string ChoicesIdentity =>
        $"vulkan/{TuningKey()}/{(Limits.SubgroupReductions ? "subgroups" : "plain")}/{(MatrixShape is { } shape ? $"coop{shape}" : "none")}/v{TuningVersion}";

    /// <summary>The backend whose choices this one runs (a cluster rank following rank 0), or null.</summary>
    internal VulkanBackend? Leader => _leader;

    /// <summary>Whether this backend runs rank 0's choices taken at the start of a run in another process (Import).</summary>
    internal bool ImportedChoices => _imported;

    // Stores `chosen` for `key` (measured now) and returns the choice to run: one of `candidates` another backend sharing this
    // table stored while this one measured, when there is one (every rank then runs the same kernel), else `chosen`, kept in
    // the cache file. A stored choice that is no candidate (one that no longer fits) is replaced.
    private int Keep(VulkanTuneKey key, int chosen, ReadOnlySpan<int> candidates)
    {
        lock (_tuned)
        {
            if (_tuned.TryGetValue(key, out int stored) && stored != chosen && candidates.Contains(stored))
            {
                return stored;
            }

            _tuned[key] = chosen;
        }

        SaveTuning(key, chosen);
        return chosen;
    }

    // Every choice known now, its cache read again (choices other backends and processes of this device stored since are
    // added; those known here stay): the width, the measured choices and those lent to neighbouring shapes.
    internal MeasuredChoices ExportChoices()
    {
        var entries = new Dictionary<string, string>(StringComparer.Ordinal) { ["width"] = Width.ToString(CultureInfo.InvariantCulture) };
        lock (_tuned)
        {
            if (_leader is null && !_imported)
            {
                LoadTuning(keepKnown: _tuningLoaded);
                _tuningLoaded = true;
            }

            foreach (var (key, value) in _tuned)
            {
                entries[Format(key)] = value.ToString(CultureInfo.InvariantCulture);
            }

            foreach (var (key, value) in _lent)
            {
                entries["lent " + Format(key)] = value.ToString(CultureInfo.InvariantCulture);
            }
        }

        return new MeasuredChoices(Kind, ChoicesIdentity, Name, entries);
    }

    // Takes `choices` in place of this backend's own when they were measured on its device and driver: the width, then the
    // table (nothing written to the cache file). Otherwise keeps its own and says why.
    internal MeasuredChoicesResult ImportChoices(MeasuredChoices choices)
    {
        if (Refusal(choices.Kind, choices.Identity, choices.Device) is { } refused)
        {
            return refused;
        }

        if (!choices.Entries.TryGetValue("width", out string? text) || !int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int width) || !WidthFits(width))
        {
            return new MeasuredChoicesResult(false, 0, $"kept its own: rank 0's width ({text ?? "none"}) does not fit {Name}");
        }

        var tuned = new Dictionary<VulkanTuneKey, int>();
        var lent = new Dictionary<VulkanTuneKey, int>();
        foreach (var (name, value) in choices.Entries)
        {
            bool isLent = name.StartsWith("lent ", StringComparison.Ordinal);
            if (name != "width" && TryParse(isLent ? name[5..] : name, out var key) && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int choice))
            {
                (isLent ? lent : tuned)[key] = choice;
            }
        }

        LeaveChoices();
        _ownWidth = Width;
        SetWidth(width);
        lock (_tuned)
        {
            _tuned.Clear();
            _lent.Clear();
            foreach (var (key, value) in tuned)
            {
                _tuned[key] = value;
            }

            foreach (var (key, value) in lent)
            {
                _lent[key] = value;
            }

            _tuningLoaded = true;                                          // rank 0's, not this cache's
        }

        _imported = true;
        return new MeasuredChoicesResult(true, tuned.Count, $"took rank 0's {tuned.Count} choices at width {width} ({choices.Device})");
    }

    // Shares `leader`'s table from now on when it runs on this device and driver: this backend's width becomes the leader's
    // and its choices are the leader's (read from the leader's cache first). Otherwise keeps its own and says why.
    internal MeasuredChoicesResult FollowChoices(VulkanBackend leader)
    {
        if (ReferenceEquals(leader, this))
        {
            return new MeasuredChoicesResult(true, TunedCount, "rank 0's own backend");
        }

        if (Refusal(leader.Kind, leader.ChoicesIdentity, leader.Name) is { } refused)
        {
            return refused;
        }

        var leaderChoices = leader.ExportChoices();                       // loads (and refreshes) the leader's table
        LeaveChoices();
        _ownWidth = Width;
        SetWidth(leader.Width);
        lock (leader._tuned)
        {
            _tuned = leader._tuned;
            _lent = leader._lent;
            _tuningLoaded = true;
        }

        _leader = leader;
        return new MeasuredChoicesResult(true, leaderChoices.Entries.Count - 1, $"shares rank 0's choices at width {leader.Width} ({leader.Name})");
    }

    // Back to this backend's own choices and width: its table read again from its cache when next needed.
    internal void LeaveChoices()
    {
        if (_leader is null && !_imported)
        {
            return;
        }

        lock (_tuned)
        {
            _tuned = [];
            _lent = [];
            _leader = null;
            _imported = false;
        }

        SetWidth(_ownWidth);
        lock (_tuned)
        {
            _tuned.Clear();
            _lent.Clear();
            _tuningLoaded = false;
        }
    }

    // Why choices of `kind` measured on `identity` do not apply here, or null when they do.
    private MeasuredChoicesResult? Refusal(string kind, string identity, string device)
    {
        if (!string.Equals(kind, Kind, StringComparison.Ordinal))
        {
            return new MeasuredChoicesResult(false, 0, $"kept its own: rank 0 runs on another kind of device ({kind}: {device}) than {Name}");
        }

        return identity == ChoicesIdentity ? null
            : new MeasuredChoicesResult(false, 0, $"kept its own: rank 0's choices were measured on another device or driver ({device}: {identity}) than {Name} ({ChoicesIdentity})");
    }

    // Whether `width` is one this device can run (as the width probe's candidates).
    private bool WidthFits(int width) =>
        width >= VulkanKernels.MinWidth && (width & (width - 1)) == 0 && width <= VulkanKernels.MaxWidth && width <= Limits.MaxInvocations && width <= Limits.MaxSizeX
        && VulkanKernels.SharedBytesBound(width) <= Limits.SharedBytes;

    // A key as the cache file writes it ("op variant a b c d e f").
    private static string Format(VulkanTuneKey key) =>
        string.Create(CultureInfo.InvariantCulture, $"{key.Op} {key.Variant} {key.A} {key.B} {key.C} {key.D} {key.E} {key.F}");

    private static bool TryParse(string text, out VulkanTuneKey key)
    {
        key = default;
        string[] parts = text.Split(' ');
        if (parts.Length != 8 || !Enum.TryParse(parts[0], out VulkanTuneOp op) || !Enum.IsDefined(op))
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

        key = new VulkanTuneKey(op, n[0], n[1], n[2], n[3], n[4], n[5], n[6]);
        return true;
    }

    /// <summary>Tests: sets the choice for <paramref name="key"/> in this backend's table (kept in memory only).</summary>
    internal void SetTunedForTests(VulkanTuneKey key, int value)
    {
        lock (_tuned)
        {
            if (!_tuningLoaded)
            {
                _tuningLoaded = true;
                LoadTuning();
            }

            _tuned[key] = value;
        }
    }

    /// <summary>Tests: forgets the choice for <paramref name="key"/> (in memory; the cache file is left as is).</summary>
    internal void ForgetTunedForTests(VulkanTuneKey key)
    {
        lock (_tuned)
        {
            _tuned.Remove(key);
            _lent.Remove(key);
        }
    }

    /// <summary>Tests: this backend's own choices again, read from the cache file when next needed.</summary>
    internal void ResetTuningForTests()
    {
        LeaveChoices();
        lock (_tuned)
        {
            _tuned.Clear();
            _lent.Clear();
            _tuningLoaded = false;
        }
    }

    /// <summary>Tests: the choice this backend runs for <paramref name="key"/>, if known.</summary>
    internal bool TryTunedForTests(VulkanTuneKey key, out int value) => TryTuned(key, out value);
}

/// <summary>The Vulkan backend's measured choices between the ranks of a cluster (VulkanBackend.SharedTuning.cs).</summary>
internal sealed class VulkanMeasuredChoices : IMeasuredChoicesProvider
{
    public string Name => "vulkan";

    public string Summary => "the Vulkan kernels' measured choices (variants, splits, attention paths) and the workgroup width, per device and driver";

    public bool Handles(Backend backend) => backend is VulkanBackend;

    public MeasuredChoices Export(Backend backend) => Of(backend).ExportChoices();

    public MeasuredChoicesResult Import(Backend backend, MeasuredChoices choices)
    {
        ArgumentNullException.ThrowIfNull(choices);
        return Of(backend).ImportChoices(choices);
    }

    public MeasuredChoicesResult Follow(Backend follower, Backend leader)
    {
        var mine = Of(follower);
        return leader is VulkanBackend vulkan ? mine.FollowChoices(vulkan)
            : new MeasuredChoicesResult(false, 0, $"kept its own: rank 0 runs on another kind of device ({leader.Kind}: {leader.Name}) than {mine.Name}");
    }

    public void Leave(Backend follower) => Of(follower).LeaveChoices();

    private static VulkanBackend Of(Backend backend) => backend as VulkanBackend
        ?? throw new ArgumentException($"{backend?.Name ?? "null"} is not a Vulkan backend.", nameof(backend));
}
