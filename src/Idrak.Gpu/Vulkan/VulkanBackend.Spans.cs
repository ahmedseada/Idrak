// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Attention over one range of keys per query row (attention_spans, with each row's log-sum-exp: attention_spans_lse), a
// workgroup per block of rows of a head as attention_tiled. Its gradient takes the host fallback for now.
internal sealed partial class VulkanBackend
{
    public override void AttentionSpansKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant = default)
    {
        if (dim <= 0 || dim > VulkanKernels.AttentionMaxDim || kvHeads <= 0 || headsPerTable <= 0 || !Fit(q, keys, values, starts, ends, y)
            || logSumExp is not null && !Fit(logSumExp))
        {
            base.AttentionSpansKernel(q, keys, values, starts, ends, y, logSumExp, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            return;
        }

        if (heads <= 0 || rows <= 0)
        {
            return;
        }

        int rowsPerBlock = VulkanKernels.TiledAttentionRows(Width);
        long blocks = (long)heads * ((rows + rowsPerBlock - 1) / rowsPerBlock);
        Span<byte> b = stackalloc byte[32];
        var push = new Push(b).I(heads).I(rows).I(keyRows).I(dim).I(heads / kvHeads).I(headsPerTable).F(scale).F(variant.Softcap).Bytes;
        if (logSumExp is null)
        {
            Run("attention_spans", RowGroups(blocks), 1, 1, [q, keys, values, starts, ends, y], push);
        }
        else
        {
            Run("attention_spans_lse", RowGroups(blocks), 1, 1, [q, keys, values, starts, ends, y, logSumExp], push);
        }
    }
}
