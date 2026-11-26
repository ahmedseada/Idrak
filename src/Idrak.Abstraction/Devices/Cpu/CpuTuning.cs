// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Abstraction.Devices.Cpu;

/// <summary>
/// The CPU's choices that follow from what the machine reports (<see cref="CpuInfo"/>: threads, vector width, cache
/// sizes): no processor is named; sizes are fractions of the caches or multiples of the vector width. Whether the
/// described lane measures its variants at each shape (<see cref="Autotune"/>) is set here too. None of these choices
/// changes a result: they decide how work is split, never what is computed.
/// </summary>
internal static class CpuTuning
{
    /// <summary>Measure the described lane's variants at each shape (default); off (IDRAK_AUTOTUNE=0): its unmeasured variant.</summary>
    internal static bool Autotune = !TuningFiles.IsOff(Environment.GetEnvironmentVariable("IDRAK_AUTOTUNE"));

    /// <summary>Overrides the element-wise parallel cut-over (IDRAK_CPU_PARALLEL_ELEMENTS; tests and benchmarks).</summary>
    internal static int? ParallelElementsOverride = EnvironmentInt("IDRAK_CPU_PARALLEL_ELEMENTS");

    /// <summary>The machine the choices are made for.</summary>
    internal static CpuInfo Machine => CpuInfo.Current;

    /// <summary>
    /// Element-wise work (≈ element operations) from which host loops split across threads: where the operands of a binary
    /// loop (two inputs, an output, about 16 bytes per element) fill one core's L2, since below that one core streams them
    /// from its own cache faster than a parallel loop starts. 1 MiB L2 → 65,536.
    /// </summary>
    internal static int ParallelElementsFormula(CpuInfo c) => Math.Max(4 * c.VectorFloats, c.L2 / 16);

    /// <summary>
    /// Side of the square tiles a transpose copies: source and destination tiles (2 · side² floats) within half the L1 data
    /// cache, a power of two. 32 KiB L1 → 32; 64 KiB → 32; 128 KiB → 64.
    /// </summary>
    internal static int TransposeTile(CpuInfo c)
    {
        int side = 4;
        while (2 * (side * 2) * (side * 2) * sizeof(float) <= c.L1 / 2)
        {
            side *= 2;
        }

        return side;
    }

    /// <summary>Rows the few-row products of a layer take (the CPU's capabilities; the described lane measures each shape).</summary>
    internal const int FewRows = 8;

    /// <summary>See <see cref="ParallelElementsFormula"/>, unless overridden.</summary>
    internal static int ParallelElements => ParallelElementsOverride ?? ParallelElementsFormula(Machine);

    /// <summary>See <see cref="TransposeTile"/>.</summary>
    internal static readonly int TransposeSide = TransposeTile(CpuInfo.Current);

    /// <summary>Whether the setting allows the platform's intrinsics (Automatic and Native).</summary>
    internal static bool IntrinsicsAllowed => ComputeResources.CpuVectors is CpuVectorPaths.Automatic or CpuVectorPaths.Native;

    /// <summary>One line for device listings: the vector paths setting.</summary>
    internal static string DescribeVectorPaths() =>
        $"{ComputeResources.CpuVectors.ToString().ToLowerInvariant()} (IDRAK_CPU_VECTORS): the described lane's variants by the portable instructions"
        + (IntrinsicsAllowed ? " and by the instruction sets the CPU has" : "");

    /// <summary>One line for benchmarks and diagnostics: the machine and the choices made from it.</summary>
    internal static string Describe() =>
        $"CPU tuning: {Machine}\n  host loops parallel from {ParallelElements:N0} element operations; transpose tile {TransposeSide}; "
        + $"described lane measured: {(Autotune ? "yes" : "no (IDRAK_AUTOTUNE=0)")}";

    private static int? EnvironmentInt(string name) =>
        int.TryParse(Environment.GetEnvironmentVariable(name), NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value > 0 ? value : null;
}
