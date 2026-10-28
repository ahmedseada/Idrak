// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Hip;

// HIP's kernels. The library depends on no part of the HIP SDK, as it depends on no CUDA Toolkit, so HIP has no
// compiler: its kernels are Idrak's own GCN code (plan 35), written in C# and assembled by Idrak, as CUDA's are PTX.
// Until they land, every kernel operation takes the host fallback, and KernelsUnavailableReason says so.
internal sealed partial class HipBackend
{
    /// <summary>Why this device runs no kernels of its own.</summary>
    public string KernelsUnavailableReason =>
        "HIP's GCN kernels are not built yet: every kernel operation on this device runs through the host fallback";

    // Why the kernels did not load, for listings, diagnostics and the host-fallback event.
    string? IKernelAvailability.KernelsUnavailable() => KernelsUnavailableReason;
}
