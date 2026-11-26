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
/// <param name="PeakAbove">The device's peak during the run above what it held before and after (the items' caches, activations, workspaces; not weights loaded on the way).</param>
/// <param name="Available">Bytes new tensors could take on the device after the run (<see cref="Devices.Backend.AvailableMemory"/>).</param>
public sealed record AdviceMeasure(int Items, int Positions, long PeakAbove, long Available);

/// <summary>
/// What an advisor is asked about: the device, the subject being advised (an image reader, a generator, a trainer, an
/// app's own component: each advisor answers for the subjects it knows and passes over the others), the share of the
/// device's memory the suggestions leave free beyond its reserve, and a run's figures (null before one was measured).
/// </summary>
/// <param name="Device">The device the subject runs on.</param>
/// <param name="Subject">What is advised.</param>
/// <param name="Measured">A run's figures, or null.</param>
public sealed record AdviceSituation(Device Device, object Subject, AdviceMeasure? Measured)
{
    /// <summary>
    /// The share of the device's memory the suggested settings leave free beyond what the device itself keeps in reserve
    /// (<see cref="Devices.Backend.AvailableMemory"/> already leaves that out): 0 by default.
    /// </summary>
    public double Headroom { get; init; }
}

/// <summary>Suggest only what the figures decide; nothing where one is missing, and nothing for a subject it does not
/// know.
/// </summary>
public interface ISettingAdvisor
{
    /// <summary></summary>
    string Name { get; }

    /// <summary>What it suggests, in a line.</summary>
    string Summary { get; }

    /// <summary>The settings worth changing in <paramref name="situation"/> (none when nothing is decided or the subject is not its own).</summary>
    IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation);
}

/// <summary>
/// An advisor for subjects of one type (an image reader, a loaded model, a cluster's step measure): the advisor answers
/// only for them, and its <see cref="Suggest(AdviceSituation, TSubject)"/> receives the subject typed, so it never tests
/// <see cref="AdviceSituation.Subject"/> itself; any other subject gets no suggestion.
/// </summary>
/// <typeparam name="TSubject">What it advises (a class: the subject's test boxes nothing).</typeparam>
public interface ISettingAdvisor<in TSubject> : ISettingAdvisor where TSubject : class
{
    /// <summary>The settings worth changing for <paramref name="subject"/> in <paramref name="situation"/> (none when nothing is decided).</summary>
    IEnumerable<SettingSuggestion> Suggest(AdviceSituation situation, TSubject subject);

    /// <inheritdoc />
    IEnumerable<SettingSuggestion> ISettingAdvisor.Suggest(AdviceSituation situation)
    {
        ArgumentNullException.ThrowIfNull(situation);
        return situation.Subject is TSubject subject ? Suggest(situation, subject) : [];
    }
}

