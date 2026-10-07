// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;

namespace Idrak.Abstraction.Testing;

/// <summary>How one item of a report came out.</summary>
public enum CheckStatus
{
    /// <summary>Checked, and it agrees.</summary>
    Passed,

    /// <summary>Checked, and at least one case disagrees (see <see cref="ConformanceReport.Failures"/>).</summary>
    Failed,

    /// <summary>Not checked; the entry's detail says why (no kernel on the device, no case reaches it, no reference).</summary>
    Skipped,
}

/// <summary>One item of a report: an operation of a device, or one property of a contract.</summary>
/// <param name="Name">The operation or property.</param>
/// <param name="Status">How it came out.</param>
/// <param name="Detail">What was checked (the kernel that ran, the calls compared), or why it was skipped.</param>
public sealed record CheckEntry(string Name, CheckStatus Status, string Detail)
{
    /// <inheritdoc />
    public override string ToString() => $"{Status.ToString().ToLowerInvariant(),-7} {Name}: {Detail}";
}

/// <summary>One case on which the implementation disagrees with the library default or breaks the contract.</summary>
/// <param name="Check">The operation or property that failed.</param>
/// <param name="Case">The case it failed on (its name and parameters).</param>
/// <param name="Message">What differs: the element, the value, the expected value and the tolerance.</param>
/// <param name="Repro">
/// The case as data, to save with <see cref="Regression.Save(string, RegressionCase)"/> and replay later; null when
/// the case cannot be saved.
/// </param>
public sealed record ConformanceFailure(string Check, string Case, string Message, RegressionCase? Repro = null)
{
    /// <inheritdoc />
    public override string ToString() => $"{Check} [{Case}]: {Message}";
}

/// <summary>Thrown by <see cref="ConformanceReport.ThrowIfFailed"/> and <see cref="StressReport.ThrowIfFailed"/>; the message is the report.</summary>
/// <param name="message">The report as text.</param>
public sealed class ConformanceException(string message) : Exception(message);

/// <summary>
/// What a conformance check found: an entry per operation or property, and every failing case. It asserts nothing
/// itself, so any test framework can check <see cref="Passed"/> (or call <see cref="ThrowIfFailed"/>) and print
/// <see cref="ToString"/>.
/// </summary>
public sealed class ConformanceReport
{
    internal ConformanceReport(string subject, IReadOnlyList<CheckEntry> entries, IReadOnlyList<ConformanceFailure> failures, int cases)
    {
        Subject = subject;
        Entries = entries;
        Failures = failures;
        Cases = cases;
    }

    /// <summary>What was checked: a device ("vulkan:0, llvmpipe"), or an implementation and the contract.</summary>
    public string Subject { get; }

    /// <summary>One entry per operation or property, in check order.</summary>
    public IReadOnlyList<CheckEntry> Entries { get; }

    /// <summary>Every failing case (at most <c>MaxFailures</c> of the options per item).</summary>
    public IReadOnlyList<ConformanceFailure> Failures { get; }

    /// <summary>The cases run.</summary>
    public int Cases { get; }

    /// <summary>True when no case failed.</summary>
    public bool Passed => Failures.Count == 0;

    /// <summary>The entries that were skipped, with why.</summary>
    public IEnumerable<CheckEntry> Skipped => Entries.Where(e => e.Status == CheckStatus.Skipped);

    /// <summary>Throws a <see cref="ConformanceException"/> carrying the report when a case failed.</summary>
    /// <exception cref="ConformanceException">A case failed.</exception>
    public void ThrowIfFailed()
    {
        if (!Passed)
        {
            throw new ConformanceException(ToString());
        }
    }

    /// <summary>
    /// Saves every failure that carries a case (<see cref="ConformanceFailure.Repro"/>) into <paramref name="folder"/>
    /// (one JSON file each), to replay with <see cref="Regression.Replay(string, Device, DeviceCheckOptions?)"/> or the
    /// contract overloads; returns the files written.
    /// </summary>
    public IReadOnlyList<string> SaveFailures(string folder) =>
        [.. Failures.Where(f => f.Repro is not null).Select(f => Regression.Save(Path.Combine(folder, Regression.FileName(f.Repro!)), f.Repro!))];

    /// <summary>The summary line, the failures, then the skipped entries.</summary>
    public override string ToString()
    {
        int passed = Entries.Count(e => e.Status == CheckStatus.Passed), failed = Entries.Count(e => e.Status == CheckStatus.Failed);
        var text = new StringBuilder();
        text.Append($"{Subject}: {(Passed ? "passed" : "FAILED")}, {Cases} cases; {passed} passed, {failed} failed, {Entries.Count - passed - failed} skipped");
        foreach (var failure in Failures)
        {
            text.Append("\n  FAIL ").Append(failure);
        }

        foreach (var entry in Skipped)
        {
            text.Append("\n  skip ").Append(entry.Name).Append(": ").Append(entry.Detail);
        }

        return text.ToString();
    }
}
