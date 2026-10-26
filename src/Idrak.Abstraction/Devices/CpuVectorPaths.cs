// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Devices;

/// <summary>
/// Which vector paths the CPU's kernels take (<see cref="ComputeResources.CpuVectors"/>). Every path gives the same
/// results, bit for bit: they differ only in speed, so the choice is for comparing them on a machine (a runtime that
/// reports no platform intrinsics, such as Mono on ARM64, against one that does).
/// </summary>
public enum CpuVectorPaths : byte
{
    /// <summary>
    /// The platform's intrinsics where the runtime reports them, else the portable 128-bit vectors where the runtime
    /// accelerates them on an architecture whose vector unit has fused multiply-add (ARM64 under Mono, which reports no
    /// AdvSimd); the default.
    /// </summary>
    Automatic = 0,

    /// <summary>Only what the runtime reports as the platform's intrinsics (AVX-512, AVX2 with FMA, AdvSimd on ARM64), as before portable paths were chosen.</summary>
    Native = 1,

    /// <summary>The portable vector kernels only, never a platform intrinsic (the 128-bit register-tiled product wherever 128-bit vectors are accelerated).</summary>
    Portable = 2,

    /// <summary>
    /// No register-tiled kernels and no platform intrinsics: the plain loops (the few-row products' streaming loops). The
    /// portable <see cref="System.Numerics.Vector{T}"/> loops still run where the runtime accelerates them;
    /// DOTNET_EnableHWIntrinsic=0 turns those off as well.
    /// </summary>
    Scalar = 3,
}
