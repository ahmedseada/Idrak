// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// What a device reports about its hardware, for device listings and diagnostics (<c>idrak devices</c>, <c>idrak
/// doctor</c>); nothing reads it to choose a path (that is <see cref="BackendCapabilities"/>). Values a backend does not
/// report are null.
/// </summary>
public sealed record BackendHardware
{
    /// <summary>Memory the device's tensors live in, in bytes (the CPU: what the process may use).</summary>
    public long? MemoryBytes { get; init; }

    /// <summary>Compute units (multiprocessors, compute units; the CPU: logical processors).</summary>
    public int? ComputeUnits { get; init; }

    /// <summary>Lanes that run in lockstep (subgroup, warp or wavefront; the CPU: float lanes of a SIMD vector).</summary>
    public int? SubgroupSize { get; init; }

    /// <summary>Threads per workgroup (block) of the device's kernels, as chosen from its limits.</summary>
    public int? KernelWidth { get; init; }

    /// <summary>What the device reports itself to be: "discrete", "integrated", "virtual" or "cpu" (a CPU or a software driver).</summary>
    public string? HardwareKind { get; init; }

    /// <summary>The driver or runtime version, when the backend reports it separately from <see cref="Backend.Name"/>.</summary>
    public string? Driver { get; init; }
}
