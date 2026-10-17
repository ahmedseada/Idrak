// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction;

/// <summary>
/// Reports on the calling thread: <see cref="Progress{T}"/> posts each report to the thread pool (or a captured
/// synchronization context), so its reports can arrive after the call that made them returned; this one calls
/// <paramref name="report"/> at once, in order, where the work runs.
/// </summary>
public sealed class SyncProgress<T>(Action<T> report) : IProgress<T>
{
    private readonly Action<T> _report = report ?? throw new ArgumentNullException(nameof(report));

    /// <inheritdoc />
    public void Report(T value) => _report(value);
}
