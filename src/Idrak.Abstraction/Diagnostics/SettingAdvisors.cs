// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Diagnostics;

/// <summary>
/// What a run measured on its device, for <see cref="ISettingAdvisor"/>s: the items it handled together (pages, rows,
/// sequences), the positions each one's cache held, the device's peak above what it held before the run, and what the
/// device had left after it.
/// </summary>
/// <param name="Items">Items handled together (pages read in one batch, rows of a step).</param>
/// <param name="Positions">Positions each item's attention cache held (0 when the run kept none).</param>
/// <param name="PeakAbove">The device's peak during the run above what it held before (the items' caches, activations, workspaces).</param>
/// <param name="Available">Bytes new tensors could take on the device after the run (<see cref="Devices.Backend.AvailableMemory"/>).</param>
public sealed record AdviceMeasure(int Items, int Positions, long PeakAbove, long Available);

/// <summary>
/// What an advisor is asked about: the device, the subject being advised (an image reader, a generator, a trainer, an
/// app's own component: each advisor answers for the subjects it knows and passes over the others), the share of the
/// device's memory the suggestions leave free, and a run's figures (null before one was measured).
/// </summary>
/// <param name="Device">The device the subject runs on.</param>
/// <param name="Subject">What is advised.</param>
/// <param name="Measured">A run's figures, or null.</param>
public sealed record AdviceSituation(Device Device, object Subject, AdviceMeasure? Measured)
{
    /// <summary>The share of the device's memory the suggested settings leave free (0.1: a tenth), for what a run did not show.</summary>
    public double Headroom { get; init; } = 0.1;
}

/// <summary>
/// Suggests settings from what a device reports (its memory, its matrix units) and what a run measured
/// (<see cref="SettingAdvisors"/>; published at <see cref="TelemetryLevel.Suggestions"/>). Suggest only what the figures
/// decide; nothing where one is missing, and nothing for a subject it does not know.
/// </summary>
public interface ISettingAdvisor
{
    /// <summary>The advisor's name (its key in <see cref="SettingAdvisors"/>).</summary>
    string Name { get; }

    /// <summary>What it suggests, in a line.</summary>
    string Summary { get; }

    /// <summary>The settings worth changing in <paramref name="situation"/> (none when nothing is decided or the subject is not its own).</summary>
    IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation);
}

/// <summary>
/// The advisors asked for suggested settings, by name: every package adds its own for the subjects it ships (Idrak.Nlp:
/// "image reader"), and an app registers its own or replaces one. Each subject asks them all
/// (<see cref="Suggest"/>) and publishes what they suggest, once each (<see cref="Publish"/>).
/// </summary>
public static class SettingAdvisors
{
    private static readonly SlotTable<string, ISettingAdvisor> Table =
        new(nameof(SettingAdvisors), (slot, app, library) => new GuardedAdvisor(slot, app, library), StringComparer.OrdinalIgnoreCase);

    static SettingAdvisors() => LibraryDefaults.Ensure(typeof(SettingAdvisors));

    /// <summary>Registers <paramref name="advisor"/> under its <see cref="ISettingAdvisor.Name"/> (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISettingAdvisor advisor)
    {
        ArgumentNullException.ThrowIfNull(advisor);
        ArgumentException.ThrowIfNullOrWhiteSpace(advisor.Name);
        Table.Register(advisor.Name, advisor, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's advisor <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The advisor registered as <paramref name="name"/>, or null.</summary>
    public static ISettingAdvisor? Find(string name) => Table.Find(name);

    /// <summary>The library's advisor <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ISettingAdvisor? Default(string name) => Table.Default(name);

    /// <summary>Who registered the advisor <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>What happens when the app's advisor <paramref name="name"/> throws (<see cref="SlotPolicy.Throw"/> unless set).</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>Each registered advisor as "name: summary", one per line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(a => $"{a.Name}: {a.Summary}"));

    /// <summary>Every registered advisor's suggestions for <paramref name="situation"/>, each setting and value once.</summary>
    public static IReadOnlyList<SettingSuggestion> Suggest(AdviceSituation situation)
    {
        ArgumentNullException.ThrowIfNull(situation);
        var seen = new HashSet<(string, string)>();
        return [.. Table.Values.SelectMany(a => a.Suggest(situation)).Where(s => seen.Add((s.Setting, s.Value)))];
    }

    /// <summary>
    /// Publishes the suggestions for <paramref name="situation"/> at <see cref="TelemetryLevel.Suggestions"/>, those not in
    /// <paramref name="published"/> (the subject's own record, so each goes out once); nothing when nobody listens there.
    /// </summary>
    public static void Publish(AdviceSituation situation, ISet<(string Setting, string Value)> published)
    {
        ArgumentNullException.ThrowIfNull(published);
        if (!Telemetry.IsEnabled(TelemetryLevel.Suggestions))
        {
            return;
        }

        foreach (var suggestion in Suggest(situation))
        {
            if (published.Add((suggestion.Setting, suggestion.Value)))
            {
                Telemetry.Suggestion(in suggestion);
            }
        }
    }

    private sealed class GuardedAdvisor(Slot slot, ISettingAdvisor app, ISettingAdvisor library) : ISettingAdvisor
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation) =>
            slot.Call(() => app.Suggest(situation).ToList(), () => library.Suggest(situation).ToList(),
                (a, b) => a.SequenceEqual(b) ? null : $"{a.Count} suggestions against the library's {b.Count}");
    }
}
