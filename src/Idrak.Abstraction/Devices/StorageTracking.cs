// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Devices;

/// <summary>One storage a device holds (<see cref="Backend.LiveStorages"/>): its bytes and who made it.</summary>
/// <param name="Bytes">Its floats' bytes.</param>
/// <param name="Owner">
/// Who holds it: the name a caller gave it (<see cref="StorageTracking.Name"/>, or the namer passed to the listing), else
/// the method that allocated it (the first frame outside the tensor and device plumbing), else "unknown".
/// </param>
/// <param name="Storage">The storage itself.</param>
public readonly record struct LiveStorage(long Bytes, string Owner, Storage Storage);

/// <summary>
/// A debug listing of the storages each device holds (plan 23, C8b): while tracking is on (<see cref="Track"/>, or
/// <c>IDRAK_TRACK_STORAGES=1</c> for the whole process) every storage made is recorded with the method that allocated it,
/// so a test or a tool can say what a device's memory in use is made of (<see cref="Backend.LiveStorages"/>). Free when off:
/// making a storage reads one static field.
/// </summary>
public static class StorageTracking
{
    private static readonly ConditionalWeakTable<Storage, Entry> Entries = new();
    private static readonly bool FromEnvironment = Environment.GetEnvironmentVariable("IDRAK_TRACK_STORAGES") is "1" or "true";
    private static int _scopes;

    // Read by every storage's constructor: one static read when tracking is off.
    internal static volatile bool On = FromEnvironment;

    /// <summary>Whether storages are recorded now.</summary>
    public static bool Enabled => On;

    /// <summary>Records every storage made until the result is disposed (scopes nest; the environment's setting stays).</summary>
    public static IDisposable Track()
    {
        Interlocked.Increment(ref _scopes);
        On = true;
        return new Scope();
    }

    /// <summary>Names <paramref name="storage"/>'s owner in the listings (recorded only while tracking is on).</summary>
    public static void Name(Storage storage, string owner)
    {
        ArgumentNullException.ThrowIfNull(storage);
        if (On && Entries.TryGetValue(storage, out var entry))
        {
            entry.Name = owner;
        }
    }

    /// <summary>
    /// The storages <paramref name="backend"/> holds that were made while tracking was on, largest first: alive, not given
    /// back (<see cref="Storage.Alive"/>, not <see cref="Storage.Evicted"/>). Storages dropped without a release are
    /// collected first (a full collection and the pending finalizers), so only what something still holds is listed.
    /// </summary>
    /// <param name="backend">The device.</param>
    /// <param name="name">Names a storage's owner where the caller knows it (a parameter's name), or null.</param>
    public static IReadOnlyList<LiveStorage> List(Backend backend, Func<Storage, string?>? name = null)
    {
        ArgumentNullException.ThrowIfNull(backend);
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        var live = new List<LiveStorage>();
        foreach (var (storage, entry) in Entries)
        {
            if (ReferenceEquals(storage.Backend, backend) && storage.Alive && !storage.Evicted && !entry.Moved)
            {
                live.Add(new LiveStorage((long)storage.Length * sizeof(float), name?.Invoke(storage) ?? entry.Name ?? entry.Site ?? "unknown", storage));
            }
        }

        live.Sort((a, b) => b.Bytes != a.Bytes ? b.Bytes.CompareTo(a.Bytes) : string.CompareOrdinal(a.Owner, b.Owner));
        return live;
    }

    /// <summary>
    /// "N storages, B bytes: 262,144 B GradientReduction.Scratch; 4 B Trainer.Fit; …" (the largest <paramref name="top"/>,
    /// storages of one owner and size counted together).
    /// </summary>
    public static string Describe(IReadOnlyList<LiveStorage> storages, int top = 24)
    {
        ArgumentNullException.ThrowIfNull(storages);
        var text = new StringBuilder();
        text.Append(CultureInfo.InvariantCulture, $"{storages.Count} storage(s), {storages.Sum(s => s.Bytes):N0} B");
        var groups = storages.GroupBy(s => (s.Owner, s.Bytes)).Select(g => (g.Key.Owner, g.Key.Bytes, Count: g.Count()))
            .OrderByDescending(g => g.Bytes * g.Count).ThenBy(g => g.Owner, StringComparer.Ordinal).ToList();
        foreach (var (owner, bytes, count) in groups.Take(top))
        {
            text.Append(CultureInfo.InvariantCulture, $"; {(count > 1 ? $"{count} x " : "")}{bytes:N0} B {owner}");
        }

        if (groups.Count > top)
        {
            text.Append(CultureInfo.InvariantCulture, $"; {groups.Count - top} more");
        }

        return text.ToString();
    }

    // A storage was made: its allocating method recorded.
    internal static void Created(Storage storage) => Entries.AddOrUpdate(storage, new Entry(Site()));

    // A storage object whose memory another took over (Backend.Restore's fresh block): not listed on its own.
    internal static void Moved(Storage storage)
    {
        if (Entries.TryGetValue(storage, out var entry))
        {
            entry.Moved = true;
        }
    }

    // The first frame outside the storage, device and tensor plumbing: "Type.Method" (names only: trimming-safe).
    private static string? Site()
    {
        var trace = new StackTrace(2, false);
        for (int i = 0; i < trace.FrameCount; i++)
        {
            if (trace.GetFrame(i) is not { } frame || DiagnosticMethodInfo.Create(frame) is not { } method || method.DeclaringTypeName is not { } type || Plumbing(type))
            {
                continue;
            }

            // A lambda's or iterator's class ("Outer+<>c__DisplayClass3_0"): its owner's name; the namespace left out.
            var parts = type.Split('+');
            int owner = Array.FindLastIndex(parts, p => !p.Contains('<', StringComparison.Ordinal));
            string name = owner >= 0 ? parts[owner] : parts[0];
            name = name[(name.LastIndexOf('.') + 1)..];
            string member = method.Name.StartsWith('<') && method.Name.IndexOf('>', StringComparison.Ordinal) is > 1 and var close ? method.Name[1..close] : method.Name;
            return $"{name}.{member}";
        }

        return null;
    }

    private static bool Plumbing(string type) =>
        type.StartsWith("System.", StringComparison.Ordinal) || type.StartsWith("Microsoft.", StringComparison.Ordinal)
        || type.Contains("Backend", StringComparison.Ordinal) || type.Contains("Storage", StringComparison.Ordinal)
        || type.StartsWith("Idrak.Abstraction.Tensor", StringComparison.Ordinal) || type.Contains("Kernels", StringComparison.Ordinal)
        || type.Contains("HostCall", StringComparison.Ordinal) || type == "Idrak.Abstraction.Module";

    private sealed class Entry(string? site)
    {
        public string? Site { get; } = site;

        public string? Name { get; set; }

        public bool Moved { get; set; }
    }

    private sealed class Scope : IDisposable
    {
        private int _disposed;

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _disposed, 1) == 0 && Interlocked.Decrement(ref _scopes) == 0)
            {
                On = FromEnvironment;
            }
        }
    }
}
