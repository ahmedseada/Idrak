// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Numerics;
using System.Runtime.CompilerServices;
using Idrak.Layers;

namespace Idrak.Backends.Vulkan;

// The fused decoding operations (BackendCapabilities.FusedKernels) as generated kernels (VulkanKernels.Fused.cs): fewer
// dispatches per token, the same math as the steps they replace. Each returns false where its kernel does not fit (more
// rows than the few-row products take, a format of one's own, more storages than the device binds per stage, storages
// larger than a binding, kernels off), and the caller then runs the unfused steps. The products' widths, words per row
// and splits of k come from the device's limits and are measured per shape on first use, as the plain packed products'.
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

    // Kernel names by ((mode · formats + format) · row blocks + row block) · word variants + word variant, built once.
    private static readonly string[] FusedGemvNames =
        [.. new[] { VulkanKernels.FusedMode.Many, VulkanKernels.FusedMode.Pair, VulkanKernels.FusedMode.Gated }
            .SelectMany(mode => new[] { VulkanKernels.PackedFormat.Int8, VulkanKernels.PackedFormat.Int4, VulkanKernels.PackedFormat.BFloat16 }
                .SelectMany(f => VulkanKernels.GemvRowBlocks.SelectMany(r => VulkanKernels.GemvWordCounts.Select(w => VulkanKernels.FusedGemvName(mode, f, r, w)))))];

    private static int FusedVariant(VulkanKernels.FusedMode mode, VulkanKernels.PackedFormat format, int block) =>
        ((int)mode * 3 + (int)format) * VulkanKernels.GemvRowBlocks.Length + block;

    private static string FusedGemvKernel(VulkanKernels.FusedMode mode, VulkanKernels.PackedFormat format, int block, int variant) =>
        FusedGemvNames[FusedVariant(mode, format, block) * VulkanKernels.GemvWordCounts.Length + variant];

    // The kernels' format for a built-in packed format (null for one they do not read).
    private static VulkanKernels.PackedFormat? KernelFormat(PackedFormat format) => format switch
    {
        PackedFormat.Int8 => VulkanKernels.PackedFormat.Int8,
        PackedFormat.Int4 => VulkanKernels.PackedFormat.Int4,
        PackedFormat.BFloat16 => VulkanKernels.PackedFormat.BFloat16,
        _ => null,
    };

    // Whether the device binds as many storages per stage as the kernel declares (maxPerStageDescriptorStorageBuffers).
    private bool Bindable(string kernel) => Kernel(kernel).Bindings <= _physical.Properties.MaxPerStageDescriptorStorageBuffers;

    // Splits of k and rows per split for a packed product's choice (splits · 8 + word variant in its rest): chunks a
    // multiple of 32 rows; one split where the splits exceed the device's workgroup count along y or their partial sums
    // (`partial` floats per split) exceed a binding.
    private (int Splits, int Chunk) GemvPlan(int choice, int k, long partial)
    {
        const int Align = VulkanKernels.GemvChunkAlign;
        int splits = Math.Max(1, (choice & Rest) >> 3);
        int chunk = Math.Max(Align, ((k + splits - 1) / splits + Align - 1) / Align * Align);
        splits = Math.Max(1, (k + chunk - 1) / chunk);
        if (splits > Limits.MaxGroupsY || splits * partial > int.MaxValue || BlockBytes((int)Math.Min(int.MaxValue, splits * partial)) > MaxStorageBytes)
        {
            splits = 1;
            chunk = Math.Max(Align, (k + Align - 1) / Align * Align);
        }

        return (splits, chunk);
    }

    // The choice of a fused product (the same candidates as a plain packed product of nmax columns): forced by the test
    // settings, stored, or the formula's (the device's width, 32 words, the most splits allowed). -1 when no variant fits.
    // `measure` is set when the shape should be measured now (autotuning on, nothing stored, not inside a measurement).
    private int FusedPlan(VulkanTuneKey key, VulkanKernels.PackedFormat format, int m, int k, int nmax, out int maxSplits, out bool measure)
    {
        measure = false;
        int perWord = VulkanKernels.ColumnsPerWord(format);
        double bytes = format switch { VulkanKernels.PackedFormat.Int8 => 1, VulkanKernels.PackedFormat.Int4 => 0.5, _ => 2 };
        maxSplits = (int)Math.Clamp(Math.Min((k + GemvMinChunk - 1) / GemvMinChunk, k * bytes / (4.0 * m)), 1, Limits.MaxGroupsY);
        int most = 1 << BitOperations.Log2((uint)maxSplits);
        int fallback = WithWidth(Width, most * 8);
        if (!GemvValid(fallback, perWord, nmax))
        {
            var all = GemvCandidates(perWord, nmax, maxSplits);
            if (all.Length == 0)
            {
                return -1;
            }

            fallback = all[^1];
        }

        if (GemvSplits is not null || GemvWords is not null)
        {
            var counts = VulkanKernels.GemvWordCounts;
            int variant = Math.Max(0, Array.IndexOf(counts, GemvWords ?? counts[0]));
            int forced = WithWidth(Width, Math.Max(1, GemvSplits ?? most) * 8 + variant);
            return GemvValid(forced, perWord, nmax) ? forced : fallback;
        }

        if (TryTuned(key, out int chosen) && GemvValid(chosen, perWord, nmax))
        {
            return chosen;
        }

        measure = Autotune && !t_timing;
        return fallback;
    }

    // Measures a fused product's candidates on scratch copies of the outputs (bit i of `writes`: operand i). Kept apart
    // from the operations so their usual calls make no closure.
    private int MeasureFused(VulkanTuneKey key, VulkanKernels.PackedFormat format, int nmax, int maxSplits, int fallback, ReadOnlySpan<Storage> operands,
        ulong writes, Action<int, Storage[]> run) =>
        TuneWithScratch(key, GemvCandidates(VulkanKernels.ColumnsPerWord(format), nmax, maxSplits), fallback, operands.ToArray(), writes, run);

    // ------------------------------------------------------------------ several products sharing an input

    public override bool PackedMatMulMany(PackedFormat format, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products)
    {
        if (FusedOff || KernelFormat(format) is not { } f || products.Length is 0 or > 3 || m <= 0 || m > Capabilities.FewRows || k <= 0)
        {
            return false;
        }

        // Operands: x, q0 … q2, scales0 … scales2, bias0 … bias2, y0 … y2 (absent products and storages: harmless stand-ins).
        int count = products.Length, flags = 0, nmax = 0;
        Span<int> n = stackalloc int[3];
        var io = new Operands();
        io[0] = x;
        for (int j = 0; j < 3; j++)
        {
            var product = products[j < count ? j : 0];
            if (product.Columns <= 0 || f != VulkanKernels.PackedFormat.BFloat16 && product.Scales is null)
            {
                return false;
            }

            n[j] = j < count ? product.Columns : 0;
            nmax = Math.Max(nmax, n[j]);
            io[1 + j] = product.Packed;
            io[4 + j] = product.Scales ?? product.Output;
            io[7 + j] = j < count && product.Bias is { } bias ? bias : x;
            io[10 + j] = product.Output;
            flags |= j < count && product.Bias is not null ? 1 << j : 0;
        }

        ReadOnlySpan<Storage> operands = io[..13];
        int block = GemvBlock(m), rows = VulkanKernels.GemvRowBlocks[block];
        if (!Fit(operands) || ((long)m + rows - 1) / rows * count > Limits.MaxGroupsZ
            || !Bindable(FusedGemvKernel(VulkanKernels.FusedMode.Many, f, block, 0)) || !Bindable("gemv_many_reduce"))
        {
            return false;
        }

        var key = new VulkanTuneKey(VulkanTuneOp.FusedGemv, FusedVariant(VulkanKernels.FusedMode.Many, f, block), m, k, n[0], n[1], n[2]);
        int choice = FusedPlan(key, f, m, k, nmax, out int maxSplits, out bool measure);
        if (choice < 0)
        {
            return false;
        }

        if (measure)
        {
            choice = MeasureMany(key, f, block, choice, maxSplits, operands, count, m, k, n[0], n[1], n[2], flags);
        }

        RunMany(f, block, choice, operands, count, m, k, n[0], n[1], n[2], flags);
        return true;
    }

    private int MeasureMany(VulkanTuneKey key, VulkanKernels.PackedFormat f, int block, int fallback, int maxSplits, ReadOnlySpan<Storage> operands,
        int count, int m, int k, int n0, int n1, int n2, int flags) =>
        MeasureFused(key, f, Math.Max(n0, Math.Max(n1, n2)), maxSplits, fallback, operands, 0b111UL << 10,
            (c, s) => RunMany(f, block, c, s, count, m, k, n0, n1, n2, flags));

    private void RunMany(VulkanKernels.PackedFormat f, int block, int choice, ReadOnlySpan<Storage> io, int count, int m, int k, int n0, int n1, int n2, int flags)
    {
        int width = WidthOf(choice), variant = choice & 7, rows = VulkanKernels.GemvRowBlocks[block], words = VulkanKernels.GemvWordCounts[variant];
        int perWord = VulkanKernels.ColumnsPerWord(f), nmax = Math.Max(n0, Math.Max(n1, n2));
        bool bf16 = f == VulkanKernels.PackedFormat.BFloat16;
        uint columnBlocks = (uint)(((long)(nmax + perWord - 1) / perWord + words - 1) / words), rowBlocks = (uint)((m + rows - 1) / rows);
        var (splits, chunk) = GemvPlan(choice, k, (long)count * m * nmax);
        var part = splits > 1 ? Allocate(splits * count * m * nmax, zeroed: false) : null;
        try
        {
            var bound = new Operands();
            int at = 0;
            bound[at++] = io[0];
            for (int i = 1; i <= 12; i++)
            {
                if (!bf16 || i is < 4 or > 6)
                {
                    bound[at++] = io[i];
                }
            }

            bound[at++] = part ?? io[10];
            Span<byte> b = stackalloc byte[32];
            RunAt(FusedGemvKernel(VulkanKernels.FusedMode.Many, f, block, variant), width, columnBlocks, (uint)splits, rowBlocks * (uint)count, bound[..at],
                new Push(b).I(m).I(k).I(chunk).I(n0).I(n1).I(n2).I(count).I(flags).Bytes);
            if (part is not null)
            {
                Span<byte> r = stackalloc byte[36];
                int scaled = f == VulkanKernels.PackedFormat.Int8 ? 8 : 0;
                Grid("gemv_many_reduce", (long)count * m * nmax,
                    [part, bf16 ? part : io[4], bf16 ? part : io[5], bf16 ? part : io[6], io[7], io[8], io[9], io[10], io[11], io[12], io[10]],
                    new Push(r).I(m).I(nmax).I(n0).I(n1).I(n2).I(count).I(splits).I(flags | scaled).I(0).Bytes);
            }
        }
        finally
        {
            part?.Release();                                                   // reused in queue order
        }
    }

    // ------------------------------------------------------------------ gate and up with the activation

    public override bool PackedMatMulGatedPair(PackedFormat format, int activation, Storage x, int m, int k,
        ReadOnlySpan<(Storage Packed, Storage? Scales, Storage? Bias, Storage Output, int Columns)> products, Storage hidden)
    {
        if (FusedOff || KernelFormat(format) is not { } f || products.Length != 2 || products[0].Columns != products[1].Columns || products[0].Columns <= 0
            || activation is < 0 or > 2 || m <= 0 || m > Capabilities.FewRows || k <= 0
            || f != VulkanKernels.PackedFormat.BFloat16 && (products[0].Scales is null || products[1].Scales is null))
        {
            return false;
        }

        // Operands: x, q0, q1, scales0, scales1, bias0, bias1, y0, y1, hidden.
        int n = products[0].Columns, flags = (products[0].Bias is null ? 0 : 1) | (products[1].Bias is null ? 0 : 2);
        var io = new Operands();
        io[0] = x;
        for (int j = 0; j < 2; j++)
        {
            io[1 + j] = products[j].Packed;
            io[3 + j] = products[j].Scales ?? products[j].Output;
            io[5 + j] = products[j].Bias ?? x;
            io[7 + j] = products[j].Output;
        }

        io[9] = hidden;
        ReadOnlySpan<Storage> operands = io[..10];
        int block = GemvBlock(m), rows = VulkanKernels.GemvRowBlocks[block];
        if (!Fit(operands) || ((long)m + rows - 1) / rows > Limits.MaxGroupsZ
            || !Bindable(FusedGemvKernel(VulkanKernels.FusedMode.Pair, f, block, 0)) || !Bindable("gemv_many_reduce"))
        {
            return false;
        }

        var key = new VulkanTuneKey(VulkanTuneOp.FusedGemv, FusedVariant(VulkanKernels.FusedMode.Pair, f, block), m, k, n, n, activation);
        int choice = FusedPlan(key, f, m, k, n, out int maxSplits, out bool measure);
        if (choice < 0)
        {
            return false;
        }

        if (measure)
        {
            choice = MeasurePair(key, f, block, choice, maxSplits, operands, m, k, n, flags, activation);
        }

        RunPair(f, block, choice, operands, m, k, n, flags, activation);
        return true;
    }

    private int MeasurePair(VulkanTuneKey key, VulkanKernels.PackedFormat f, int block, int fallback, int maxSplits, ReadOnlySpan<Storage> operands,
        int m, int k, int n, int flags, int activation) =>
        MeasureFused(key, f, n, maxSplits, fallback, operands, 0b111UL << 7, (c, s) => RunPair(f, block, c, s, m, k, n, flags, activation));

    private void RunPair(VulkanKernels.PackedFormat f, int block, int choice, ReadOnlySpan<Storage> io, int m, int k, int n, int flags, int activation)
    {
        int width = WidthOf(choice), variant = choice & 7, rows = VulkanKernels.GemvRowBlocks[block], words = VulkanKernels.GemvWordCounts[variant];
        int perWord = VulkanKernels.ColumnsPerWord(f);
        bool bf16 = f == VulkanKernels.PackedFormat.BFloat16;
        uint columnBlocks = (uint)(((long)(n + perWord - 1) / perWord + words - 1) / words), rowBlocks = (uint)((m + rows - 1) / rows);
        var (splits, chunk) = GemvPlan(choice, k, 2L * m * n);
        var part = splits > 1 ? Allocate(splits * 2 * m * n, zeroed: false) : null;
        try
        {
            var bound = new Operands();
            int at = 0;
            for (int i = 0; i <= 9; i++)
            {
                if (!bf16 || i is < 3 or > 4)
                {
                    bound[at++] = io[i];
                }
            }

            bound[at++] = part ?? io[9];
            Span<byte> b = stackalloc byte[24];
            RunAt(FusedGemvKernel(VulkanKernels.FusedMode.Pair, f, block, variant), width, columnBlocks, (uint)splits, rowBlocks, bound[..at],
                new Push(b).I(m).I(n).I(k).I(chunk).I(flags).I(activation).Bytes);
            if (part is not null)
            {
                Span<byte> r = stackalloc byte[36];
                int scaled = f == VulkanKernels.PackedFormat.Int8 ? 8 : 0;
                Grid("gemv_many_reduce", (long)m * n,
                    [part, bf16 ? part : io[3], bf16 ? part : io[4], part, io[5], io[6], io[0], io[7], io[8], io[7], io[9]],
                    new Push(r).I(m).I(n).I(n).I(n).I(0).I(2).I(splits).I(flags | scaled | 16).I(activation).Bytes);
            }
        }
        finally
        {
            part?.Release();                                                   // reused in queue order
        }
    }

    // ------------------------------------------------------------------ a down projection reading the activation

    public override bool PackedMatMulGated(PackedFormat format, int activation, Storage gate, Storage up, Storage packed, Storage? scales, Storage y,
        int m, int n, int k)
    {
        if (FusedOff || KernelFormat(format) is not { } f || activation is not (0 or 1) || m <= 0 || m > Capabilities.FewRows || n <= 0 || k <= 0
            || f != VulkanKernels.PackedFormat.BFloat16 && scales is null)
        {
            return false;
        }

        // Operands: gate, up, q, scales, y.
        var io = new Operands();
        (io[0], io[1], io[2], io[3], io[4]) = (gate, up, packed, scales ?? y, y);
        ReadOnlySpan<Storage> operands = io[..5];
        int block = GemvBlock(m), rows = VulkanKernels.GemvRowBlocks[block];
        if (!Fit(operands) || ((long)m + rows - 1) / rows > Limits.MaxGroupsZ || !Bindable(FusedGemvKernel(VulkanKernels.FusedMode.Gated, f, block, 0)))
        {
            return false;
        }

        var key = new VulkanTuneKey(VulkanTuneOp.FusedGemv, FusedVariant(VulkanKernels.FusedMode.Gated, f, block), m, k, n, 0, activation);
        int choice = FusedPlan(key, f, m, k, n, out int maxSplits, out bool measure);
        if (choice < 0)
        {
            return false;
        }

        if (measure)
        {
            choice = MeasureGated(key, f, block, choice, maxSplits, operands, m, n, k, activation);
        }

        RunGated(f, block, choice, operands, m, n, k, activation);
        return true;
    }

    private int MeasureGated(VulkanTuneKey key, VulkanKernels.PackedFormat f, int block, int fallback, int maxSplits, ReadOnlySpan<Storage> operands,
        int m, int n, int k, int activation) =>
        MeasureFused(key, f, n, maxSplits, fallback, operands, 1UL << 4, (c, s) => RunGated(f, block, c, s, m, n, k, activation));

    private void RunGated(VulkanKernels.PackedFormat f, int block, int choice, ReadOnlySpan<Storage> io, int m, int n, int k, int activation)
    {
        int width = WidthOf(choice), variant = choice & 7, rows = VulkanKernels.GemvRowBlocks[block], words = VulkanKernels.GemvWordCounts[variant];
        int perWord = VulkanKernels.ColumnsPerWord(f);
        uint columnBlocks = (uint)(((long)(n + perWord - 1) / perWord + words - 1) / words), rowBlocks = (uint)((m + rows - 1) / rows);
        var (splits, chunk) = GemvPlan(choice, k, (long)m * n);
        var y = io[4];
        var output = splits == 1 ? y : Allocate(splits * m * n, zeroed: false);
        try
        {
            string kernel = FusedGemvKernel(VulkanKernels.FusedMode.Gated, f, block, variant);
            Span<byte> b = stackalloc byte[20];
            var push = new Push(b).I(m).I(n).I(k).I(chunk).I(activation).Bytes;
            if (f == VulkanKernels.PackedFormat.BFloat16)
            {
                RunAt(kernel, width, columnBlocks, (uint)splits, rowBlocks, [io[0], io[1], io[2], output], push);
            }
            else
            {
                RunAt(kernel, width, columnBlocks, (uint)splits, rowBlocks, [io[0], io[1], io[2], io[3], output], push);
            }

            if (splits > 1)
            {
                Span<byte> r = stackalloc byte[12];
                var reducePush = new Push(r).I(m * n).I(n).I(splits).Bytes;
                if (f == VulkanKernels.PackedFormat.Int8)
                {
                    Grid("int8_gemv_reduce", (long)m * n, [output, io[3], y], reducePush);
                }
                else
                {
                    Grid("gemv_reduce", (long)m * n, [output, y], reducePush);
                }
            }
        }
        finally
        {
            if (splits > 1)
            {
                output.Release();                                              // reused in queue order
            }
        }
    }

    // ------------------------------------------------------------------ a projection, its residual and the next norm

    public override bool PackedMatMulAddRmsNorm(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k,
        Storage residual, Storage sum, Storage gain, Storage normalized, float eps, float offset)
    {
        if (FusedOff || KernelFormat(format) is not { } f || m <= 0 || m > Capabilities.FewRows || n <= 0 || k <= 0
            || f != VulkanKernels.PackedFormat.BFloat16 && scales is null || !Fit(x, packed, scales ?? x, y, residual, sum, gain, normalized))
        {
            return false;
        }

        int block = GemvBlock(m), rows = VulkanKernels.GemvRowBlocks[block];
        if (((long)m + rows - 1) / rows > Limits.MaxGroupsZ)
        {
            return false;
        }

        // The product as the plain packed product would run it (its measured choice), its partial sums left for the row pass.
        int choice = GemvChoice(f, block, x, packed, scales, y, m, n, k);
        if (choice < 0)
        {
            return false;
        }

        int width = WidthOf(choice), variant = choice & 7, words = VulkanKernels.GemvWordCounts[variant], perWord = VulkanKernels.ColumnsPerWord(f);
        uint columnBlocks = (uint)(((long)(n + perWord - 1) / perWord + words - 1) / words), rowBlocks = (uint)((m + rows - 1) / rows);
        var (splits, chunk) = GemvPlan(choice, k, (long)m * n);
        var output = splits == 1 ? y : Allocate(splits * m * n, zeroed: false);
        try
        {
            string kernel = GemvKernelNames[((int)f * VulkanKernels.GemvRowBlocks.Length + block) * VulkanKernels.GemvWordCounts.Length + variant];
            Span<byte> b = stackalloc byte[16];
            var push = new Push(b).I(m).I(n).I(k).I(chunk).Bytes;
            if (scales is null)
            {
                RunAt(kernel, width, columnBlocks, (uint)splits, rowBlocks, [x, packed, output], push);
            }
            else
            {
                RunAt(kernel, width, columnBlocks, (uint)splits, rowBlocks, [x, packed, scales, output], push);
            }

            Span<byte> r = stackalloc byte[24];
            bool scaled = f == VulkanKernels.PackedFormat.Int8 && splits > 1;
            Rows("gemv_add_rms_norm", m, n, [output, scales ?? output, residual, y, sum, gain, normalized],
                new Push(r).I(m).I(n).I(splits).B(scaled).F(eps).F(offset).Bytes);
        }
        finally
        {
            if (splits > 1)
            {
                output.Release();                                              // reused in queue order
            }
        }

        return true;
    }

    // ------------------------------------------------------------------ attention heads: norms, rotation, layout, cache

    public override bool NormRopeHeads(Storage q, Storage k, Storage v, int batch, int steps, int heads, int kvHeads, int cols,
        Storage? gainQ, float epsQ, float offsetQ, Storage? gainK, float epsK, float offsetK, Storage? cos, Storage? sin, Storage? positions,
        int half, bool interleaved, Storage yq, Storage yk, Storage yv, Storage? position, int capacity, int stride, bool bfloat16)
    {
        int rot = cos is null ? 0 : half;
        if (FusedOff || cols <= 0 || cols > VulkanKernels.NormRopeMaxCols || batch <= 0 || steps <= 0 || heads <= 0 || kvHeads <= 0
            || (gainQ is null) != (gainK is null) || (cos is null) != (sin is null) || cos is not null && positions is null
            || rot < 0 || 2 * rot > cols || bfloat16 && (stride & 1) != 0 || (long)batch * steps * (heads + 2L * kvHeads) > int.MaxValue)
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

        int flags = (gainQ is null ? 0 : 1) | (bfloat16 ? 2 : 0) | (position is null ? 0 : 4);
        Span<byte> b = stackalloc byte[60];
        Run("norm_rope_heads", RowGroups(rows1 + 2L * rows2), 1, 1, operands,
            new Push(b).I(rows1).I(rows2).I(heads).I(kvHeads).I(steps).I(cols).I(rot).B(interleaved).I(flags).I(capacity).I(stride)
                .F(epsQ).F(offsetQ).F(epsK).F(offsetK).Bytes);
        return true;
    }
}
