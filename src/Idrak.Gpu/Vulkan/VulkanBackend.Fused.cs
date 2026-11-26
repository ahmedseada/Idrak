// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;

namespace Idrak.Gpu.Vulkan;

// The fused decoding operations (Backend.FusedKernels) as generated kernels (VulkanKernels.Fused.cs): fewer
// dispatches per token, the same math as the steps they replace. Each returns false where its kernel does not fit
// (storages larger than a binding, kernels off), and the caller then runs the unfused steps.
internal sealed partial class VulkanBackend
{
    /// <summary>Tests: true makes the fused operations decline (callers run the unfused steps), to compare the two.</summary>
    internal static bool FusedOff { get; set; }

    // Operands of a fused dispatch, on the stack (no allocation per dispatch).
    [InlineArray(16)]
    private struct Operands
    {
        private Storage _first;
    }

    // ------------------------------------------------------------------ attention heads: norms, rotation, layout, cache

    public override bool NormRopeHeadsKernel(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride)
    {
        int rot = cos is null ? 0 : half;
        if (FusedOff || cols <= 0 || cols > VulkanKernels.NormRopeMaxCols || batch <= 0 || steps <= 0 || heads <= 0 || kvHeads <= 0
            || (gainQ is null) != (gainK is null) || (cos is null) != (sin is null) || cos is not null && positions is null
            || rot < 0 || 2 * rot > cols || (long)batch * steps * (heads + 2L * kvHeads) > int.MaxValue)
        {
            return false;
        }

        int rows1 = batch * steps * heads, rows2 = batch * steps * kvHeads;
        var io = new Operands();
        (io[0], io[1], io[2], io[3], io[4], io[5]) = (q, k, v, gainQ ?? q, gainK ?? q, cos ?? q);
        (io[6], io[7], io[8], io[9], io[10], io[11]) = (sin ?? q, positions ?? q, position ?? q, yq, yk, yv);
        ReadOnlySpan<Storage> operands = io[..12];
        if (!Fit(operands) || !Bindable("norm_rope_heads"))
        {
            return false;
        }

        int flags = (gainQ is null ? 0 : 1) | (position is null ? 0 : 4);
        Span<byte> b = stackalloc byte[60];
        Run("norm_rope_heads", RowGroups(rows1 + 2L * rows2), 1, 1, operands,
            new Push(b).I(rows1).I(rows2).I(heads).I(kvHeads).I(steps).I(cols).I(rot).B(interleaved).I(flags).I(capacity).I(stride)
                .F(epsQ).F(offsetQ).F(epsK).F(offsetK).Bytes);
        return true;
    }
}
