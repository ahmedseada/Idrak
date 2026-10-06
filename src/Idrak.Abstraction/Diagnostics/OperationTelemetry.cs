// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Diagnostics;

/// <summary>
/// Timing of tensor operations for telemetry: Idrak's <c>Telemetry</c> sets <see cref="Completed"/> while a hook listens
/// for operations, and tensor operations report to it; otherwise they read one field and carry on.
/// </summary>
internal static class OperationTelemetry
{
    /// <summary>Receives each operation (name, output, start timestamp, whether it is a backward step); null when no one listens.</summary>
    public static volatile Action<string, Tensor, long, bool>? Completed;

    /// <summary>A start timestamp when someone listens for operations, otherwise 0.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Start() => Completed is null ? 0 : Stopwatch.GetTimestamp();

    /// <summary>Reports an operation that started at <paramref name="start"/> (from <see cref="Start"/>).</summary>
    public static void Operation(string name, Tensor output, long start, bool backward) => Completed?.Invoke(name, output, start, backward);
}
