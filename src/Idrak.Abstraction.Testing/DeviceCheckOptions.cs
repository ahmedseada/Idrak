// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Operations;

namespace Idrak.Abstraction.Testing;

/// <summary>How <see cref="Conformance.Check(Device, DeviceCheckOptions?)"/> checks a device.</summary>
public sealed record DeviceCheckOptions
{
    /// <summary>
    /// The tolerances by operation name where the default <see cref="Tolerance"/> is too tight: products of bfloat16, 8-bit
    /// and 4-bit weights, attention over many keys, and the packed (bfloat16) outputs.
    /// </summary>
    public static IReadOnlyDictionary<string, float> DefaultTolerances { get; } = BuildDefaults();

    /// <summary>The seed of the random cases' first run (each run and case gets its own from it).</summary>
    public int Seed { get; init; } = 1;

    /// <summary>How many times each random case runs, with a new seed each time.</summary>
    public int RandomRuns { get; init; } = 2;

    /// <summary>Also the large shapes (a million elements, products of a thousand rows): slower, and closer to real models.</summary>
    public bool Large { get; init; }

    /// <summary>
    /// The agreement wanted for an operation without an entry in <see cref="Tolerances"/>: |device - CPU| ≤ tolerance ·
    /// max(1, |CPU|) for every value it writes.
    /// </summary>
    public float Tolerance { get; init; } = 1e-4f;

    /// <summary>Tolerances by operation name (<see cref="Operation.Name"/>); <see cref="DefaultTolerances"/> by default.</summary>
    public IReadOnlyDictionary<string, float> Tolerances { get; init; } = DefaultTolerances;

    /// <summary>The cases to run: <see cref="DeviceCases.All"/> by default; add your own (<c>[.. DeviceCases.All, mine]</c>).</summary>
    public IReadOnlyList<DeviceCase> Cases { get; init; } = DeviceCases.All;

    /// <summary>Runs only the cases whose name this accepts (all when null).</summary>
    public Func<string, bool>? Filter { get; init; }

    /// <summary>The failing calls reported per operation; the others are counted.</summary>
    public int MaxFailures { get; init; } = 3;

    /// <summary>The tolerance for <paramref name="operation"/>.</summary>
    public float ToleranceFor(Operation operation)
    {
        ArgumentNullException.ThrowIfNull(operation);
        return Tolerances.TryGetValue(operation.Name, out float tolerance) ? tolerance : Tolerance;
    }

    private static Dictionary<string, float> BuildDefaults()
    {
        var tolerances = new Dictionary<string, float>(StringComparer.Ordinal);
        foreach (var operation in Ops.All)
        {
            string name = operation.Name;
            if (name.Contains("BFloat16", StringComparison.Ordinal) || name.Contains("Packed", StringComparison.Ordinal)
                || name.Contains("Int4", StringComparison.Ordinal) || name.Contains("Int8", StringComparison.Ordinal)
                || name.Contains("Float8", StringComparison.Ordinal) || name is "NormRopeHeads" or "GemmStrided")
            {
                tolerances[name] = 2e-2f;
            }
            else if (name.Contains("Attention", StringComparison.Ordinal) || name.Contains("MatMul", StringComparison.Ordinal)
                     || name is "SoftmaxCrossEntropyRows")
            {
                tolerances[name] = 1e-3f;
            }
        }

        return tolerances;
    }
}
