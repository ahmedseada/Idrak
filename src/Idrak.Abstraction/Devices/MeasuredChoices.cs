// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text;

namespace Idrak.Abstraction.Devices;

/// <summary>
/// The choices a backend measured on its device at run time (which kernel variant of a product, how many splits of a
/// reduction, the workgroup width, which attention path), as one rank of a cluster hands them to another
/// (<see cref="IMeasuredChoicesProvider"/>). Some choices pick different algorithms whose results differ in the last bits
/// (an online against a two-pass softmax, a cooperative-matrix product's order), so ranks that each measure their own can
/// run different kernels; a cluster run gives every rank rank 0's (<see cref="Cluster.ClusterOptions.ShareTuning"/>).
/// </summary>
/// <param name="Kind">The device kind they were measured on ("vulkan", "cuda").</param>
/// <param name="Identity">
/// What they apply to, as the backend names it: the device (its UUID), its driver and the version of its kernels. A
/// backend takes choices of its own identity only.
/// </param>
/// <param name="Device">The device's name, for messages.</param>
/// <param name="Entries">Each choice under its key, both as the backend writes them (no tab or line break in either).</param>
public sealed record MeasuredChoices(string Kind, string Identity, string Device, IReadOnlyDictionary<string, string> Entries)
{
    private const string Magic = "idrak-measured-choices 1";

    /// <summary>The choices as bytes to send to another rank (UTF-8 lines: a header, then a key and its value a line).</summary>
    /// <exception cref="ArgumentException">A field, key or value holds a tab or a line break.</exception>
    public byte[] ToBytes()
    {
        var text = new StringBuilder();
        text.Append(Magic).Append('\n');
        foreach (string field in (string[])[Kind, Identity, Device])
        {
            text.Append(Plain(field, "field")).Append('\n');
        }

        foreach (var (key, value) in Entries)
        {
            text.Append(Plain(key, "key")).Append('\t').Append(Plain(value, "value")).Append('\n');
        }

        return Encoding.UTF8.GetBytes(text.ToString());
    }

    /// <summary>The choices <see cref="ToBytes"/> wrote.</summary>
    /// <exception cref="InvalidDataException">The bytes are not measured choices.</exception>
    public static MeasuredChoices FromBytes(ReadOnlySpan<byte> bytes)
    {
        string[] lines = Encoding.UTF8.GetString(bytes).Split('\n');
        if (lines.Length < 4 || lines[0] != Magic)
        {
            throw new InvalidDataException("not a rank's measured choices (no header)");
        }

        var entries = new Dictionary<string, string>(StringComparer.Ordinal);
        for (int i = 4; i < lines.Length; i++)
        {
            if (lines[i].Length == 0)
            {
                continue;
            }

            int tab = lines[i].IndexOf('\t', StringComparison.Ordinal);
            if (tab <= 0)
            {
                throw new InvalidDataException($"line {i + 1} of a rank's measured choices has no key and value: {lines[i]}");
            }

            entries[lines[i][..tab]] = lines[i][(tab + 1)..];
        }

        return new MeasuredChoices(lines[1], lines[2], lines[3], entries);
    }

    private static string Plain(string text, string what) =>
        text.AsSpan().IndexOfAny('\t', '\n', '\r') < 0 ? text : throw new ArgumentException($"A measured choice's {what} holds a tab or a line break: {text}");
}

/// <summary>What a backend did with another rank's measured choices (<see cref="IMeasuredChoicesProvider"/>).</summary>
/// <param name="Taken">Whether it runs them now; false: it keeps its own (another device, driver or kind), and <paramref name="Detail"/> says why.</param>
/// <param name="Choices">How many choices it took (those known when it took them; a live share adds rank 0's later ones).</param>
/// <param name="Detail">What happened, or why not, for the run's report.</param>
public sealed record MeasuredChoicesResult(bool Taken, int Choices, string Detail);

/// <summary>
/// Hands one kind of backend's measured choices between the ranks of a cluster, so every rank runs rank 0's kernels
/// (<see cref="Cluster.ClusterOptions.ShareTuning"/>). Each backend that measures choices registers one
/// (<see cref="MeasuredChoiceProviders"/>); the cluster uses this contract only. A provider never takes choices measured on
/// another device or driver (another UUID, driver or kernel version than the backend's own: <see cref="MeasuredChoices.Identity"/>),
/// and taking them writes nothing to the taking backend's own cache: what it measures itself afterwards is kept as usual.
/// </summary>
public interface IMeasuredChoicesProvider
{
    /// <summary>Its name: the device kind it serves ("vulkan").</summary>
    string Name { get; }

    /// <summary>What it shares, for listings.</summary>
    string Summary { get; }

    /// <summary>Whether <paramref name="backend"/> is one of its kind.</summary>
    bool Handles(Backend backend);

    /// <summary>
    /// Every choice <paramref name="backend"/> knows now: measured in this process, and those its cache keeps for its device
    /// and driver (read again, so choices other processes or backends of the device stored since are included).
    /// </summary>
    MeasuredChoices Export(Backend backend);

    /// <summary>
    /// Makes <paramref name="choices"/> (another process's rank 0, <see cref="Export"/>) <paramref name="backend"/>'s in place of
    /// its own when they were measured on its device and driver; otherwise it keeps its own and the result says why. The
    /// choices taken are not written to its cache.
    /// </summary>
    MeasuredChoicesResult Import(Backend backend, MeasuredChoices choices);

    /// <summary>
    /// In one process: <paramref name="follower"/> uses <paramref name="leader"/>'s choices from now on, live (a choice neither
    /// knows yet is measured once, by whichever needs it first, into the leader's), when both run on the same device and
    /// driver; otherwise it keeps its own and the result says why. Until <see cref="Leave"/>.
    /// </summary>
    MeasuredChoicesResult Follow(Backend follower, Backend leader);

    /// <summary>Ends <see cref="Follow"/> or <see cref="Import"/>: <paramref name="follower"/> takes its own choices again (read from its cache).</summary>
    void Leave(Backend follower);
}

/// <summary>
/// The registry of <see cref="IMeasuredChoicesProvider"/>s by device kind. The library registers its GPU backends' (Vulkan,
/// CUDA); the CPU backend has none (its measured cut-overs decide how work is split across threads, never a result).
/// </summary>
public static class MeasuredChoiceProviders
{
    private static readonly SlotTable<string, IMeasuredChoicesProvider> Table = new(nameof(MeasuredChoiceProviders), comparer: StringComparer.OrdinalIgnoreCase,
        unguarded: "a provider hands a backend's own state over, which no other implementation could fall back to");

    static MeasuredChoiceProviders() => LibraryDefaults.Ensure(typeof(MeasuredChoiceProviders));

    /// <summary>The registered providers, in registration order.</summary>
    public static IReadOnlyList<IMeasuredChoicesProvider> All => Table.Values;

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Registers <paramref name="provider"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IMeasuredChoicesProvider provider)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentException.ThrowIfNullOrWhiteSpace(provider.Name);
        Table.Register(provider.Name, provider, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's provider of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The provider of that name, or null.</summary>
    public static IMeasuredChoicesProvider? Find(string name) => Table.Find(name);

    /// <summary>The library's provider of that name, or null.</summary>
    public static IMeasuredChoicesProvider? Default(string name) => Table.Default(name);

    /// <summary>Who registered the provider of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>The first provider that handles <paramref name="backend"/>, or null: its backend measures nothing that is shared.</summary>
    public static IMeasuredChoicesProvider? For(Backend backend)
    {
        ArgumentNullException.ThrowIfNull(backend);
        foreach (var provider in Table.Values)
        {
            if (provider.Handles(backend))
            {
                return provider;
            }
        }

        return null;
    }

    /// <summary>Each provider as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(p => $"{p.Name}: {p.Summary}"));
}
