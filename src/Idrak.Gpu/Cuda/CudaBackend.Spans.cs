// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Attention over one range of keys per query row (PtxKernels.SpansTiled.cs): the tensor-core kernel when MixedPrecision
// asks for tensor cores and the GPU has them (no soft-cap): the single-stage one or the pipelined one (keys and values
// copied to bfloat16 once, then two tiles in flight: PtxKernels.SpansPipelined.cs; the same outputs bit for bit), and its
// warps per block (the same bits too: tiles on a fixed grid, each warp only its own; PtxKernels.SpansTiled.cs), measured
// per shape (the single-stage kernel until measured), else the float32
// tiled kernel (head sizes up to 256: 32 query rows a block over 128), else (a head size it does not take, or a GPU whose
// shared memory per block cannot hold its tiles) the first kernel, attention_spans_f32, up to 128; the host fallback
// beyond. Each head size's module is built and loaded the first time it is needed, and judged by what the device reports
// (shared memory per block) before it is loaded. The gradient runs on attn_spans_bwd_kv_f32 and attn_spans_bwd_q_f32
// (PtxKernels.SpansBackward.cs; float32, head sizes up to 128), and over 128 on their wide versions (8 dimensions a lane,
// dynamic shared memory) when the device holds them; the host fallback beyond. SpanHeadDim says which head sizes this
// device runs on its own kernels, from its reported limits (BackendCapabilities.SpanAttentionHeadDim). And the measured choice between this operation and the composed
// scores (Backend.PrefersComposedAttention, for Tensor.AttentionFastest; with its gradient when training), kept with
// the other measured choices.
internal sealed unsafe partial class CudaBackend
{
    // The span modules by kind and padded head size ("f32 72", "tc 80"): their kernels, or null when the module cannot
    // run here (the reason in SpanModuleErrors).
    private readonly Dictionary<string, Dictionary<string, IntPtr>?> _spanModules = [];
    private readonly Dictionary<string, string> _spanModuleErrors = [];

    /// <summary>Span modules that did not load, with the reason (tests and diagnostics).</summary>
    internal IReadOnlyDictionary<string, string> SpanModuleErrors
    {
        get
        {
            lock (_spanModules)
            {
                return new Dictionary<string, string>(_spanModuleErrors);
            }
        }
    }

    public override void AttentionSpansKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, Storage? logSumExp, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale, AttentionVariant variant = default)
    {
        if (dim <= 0 || dim > PtxKernels.SpanMaxDim || kvHeads <= 0 || headsPerTable <= 0)
        {
            base.AttentionSpansKernel(q, keys, values, starts, ends, y, logSumExp, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            return;
        }

        if (heads <= 0 || rows <= 0)
        {
            return;
        }

        ulong lse = logSumExp is null ? 0UL : P(logSumExp);
        int td = PtxKernels.SpanTensorDim(dim);
        Span<int> candidates = stackalloc int[2 * PtxKernels.SpanTensorWarps.Length];
        if (variant.Softcap <= 0f && MixedPrecision.UsesTensorCores && td > 0 && SpanTensorCandidates(dim, kvHeads, keyRows, candidates) is > 0 and var count)
        {
            var key = new TuneKey(TuneOp.SpanKernel, dim, heads, TuneSizes.Class(rows), TuneSizes.Class(keyRows), kvHeads, headsPerTable);
            int chosen = SpanTensorForced is int forced && candidates[..count].Contains(forced) ? forced
                : TryTuned(key, candidates[..count], candidates[0], out int known) ? known
                : MeasureSpanTensor(key, candidates[..count].ToArray(), q, keys, values, starts, ends, y, lse, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale);
            RunSpanTensor(chosen, q, keys, values, starts, ends, y, lse, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale);
            return;
        }

        int fd = PtxKernels.SpanFloatDim(dim);
        if (fd > 0 && SpanFunction(tensor: false, fd) is { } tiled)
        {
            int blockRows = PtxKernels.SpanFloatRows(fd);
            t_sharedBytes = (uint)PtxKernels.SpanFloatShared(fd);
            Launch(tiled, (uint)((rows + blockRows - 1) / blockRows), (uint)heads, 1, 128, 1,
                P(q), P(keys), P(values), P(starts), P(ends), P(y), lse, U(rows), U(keyRows), U(dim), F(scale), U(heads / kvHeads), U(headsPerTable),
                F(variant.Softcap));
            return;
        }

        if (dim > PtxKernels.FlashMaxDim)
        {
            base.AttentionSpansKernel(q, keys, values, starts, ends, y, logSumExp, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale, variant);
            return;
        }

        Launch(K(PtxKernels.SpanAttentionName), (uint)((rows + PtxKernels.FlashTile - 1) / PtxKernels.FlashTile), (uint)heads, 1, 128, 1,
            P(q), P(keys), P(values), P(starts), P(ends), P(y), lse, U(rows), U(keyRows), U(dim), F(scale), U(heads / kvHeads), U(headsPerTable),
            F(variant.Softcap));
    }

    // Δ = dO · O per row (attn_bwd_d_f32), then dK and dV (a block per 16 keys of a key/value head, through every query
    // head of its group), then dQ (a block per 32 rows of a head): every gradient element added by one thread, in order.
    // The same float32 kernels under MixedPrecision (the gradient's sums stay float32).
    public override void AttentionSpansBackwardKernel(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage output, Storage logSumExp,
        Storage dOutput, Storage dq, Storage dkeys, Storage dvalues, int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale,
        AttentionVariant variant = default)
    {
        bool wide = dim > PtxKernels.FlashMaxDim;
        if (dim <= 0 || dim > PtxKernels.SpanMaxDim || kvHeads <= 0 || headsPerTable <= 0 || (long)heads * rows > int.MaxValue
            || wide && SpanBackwardWide() is null)
        {
            base.AttentionSpansBackwardKernel(q, keys, values, starts, ends, output, logSumExp, dOutput, dq, dkeys, dvalues, heads, kvHeads, headsPerTable, rows,
                keyRows, dim, scale, variant);
            return;
        }

        if (heads <= 0 || rows <= 0 || keyRows <= 0)
        {
            return;
        }

        int total = heads * rows;
        var delta = Allocate(total, zeroed: false);
        try
        {
            Launch1D(K("attn_bwd_d_f32"), total, P(output), P(dOutput), P(delta), U(dim), U(total));
            ReadOnlySpan<ulong> args = [P(q), P(keys), P(values), P(starts), P(ends), P(dOutput), P(logSumExp), P(delta), P(dq), P(dkeys), P(dvalues),
                U(rows), U(keyRows), U(dim), F(scale), U(heads / kvHeads), U(headsPerTable), F(variant.Softcap)];
            var (keysKernel, queriesKernel) = wide ? SpanBackwardWide()!.Value : (K(PtxKernels.SpanBackwardKeysName), K(PtxKernels.SpanBackwardQueriesName));
            if (wide)
            {
                t_sharedBytes = (uint)PtxKernels.SpanBackwardWideShared(PtxKernels.SpanBackwardWideKeysName);
            }

            Launch(keysKernel, (uint)((keyRows + PtxKernels.SpanBackwardKeys - 1) / PtxKernels.SpanBackwardKeys), (uint)kvHeads, 1, 128, 1, args);
            if (wide)
            {
                t_sharedBytes = (uint)PtxKernels.SpanBackwardWideShared(PtxKernels.SpanBackwardWideQueriesName);
            }

            Launch(queriesKernel, (uint)((rows + PtxKernels.SpanBackwardRows - 1) / PtxKernels.SpanBackwardRows), (uint)heads, 1, 128, 1, args);
        }
        finally
        {
            delta.Release();                                                  // reused in stream order
        }
    }

    /// <summary>
    /// Tests and benchmarks only: the tensor-core span kernel every call takes where it can (warps per block, plus
    /// <see cref="SpanPipelined"/> for the pipelined kernel) instead of the measured choice.
    /// </summary>
    internal static int? SpanTensorForced { get; set; }

    /// <summary>Added to the warps per block of a tensor-core span candidate that names the pipelined kernel (PtxKernels.SpansPipelined.cs).</summary>
    internal const int SpanPipelined = 100;

    // The tensor-core span kernels that can run this head size here, as candidates for the measured choice: the
    // single-stage kernel per warp count (warps), then the pipelined one (SpanPipelined + warps) where its module loads and
    // its copy of the keys and values fits a single allocation. The single-stage kernel first: what runs before measuring.
    private int SpanTensorCandidates(int dim, int kvHeads, int keyRows, Span<int> candidates)
    {
        int count = 0, td = PtxKernels.SpanTensorDim(dim), dv = PtxKernels.SpanPipelinedDim(dim);
        foreach (int w in PtxKernels.SpanTensorWarps)
        {
            if (SpanFunction(tensor: true, td, w) is not null)
            {
                candidates[count++] = w;
            }
        }

        if (dv > 0 && keyRows > 0 && (long)kvHeads * keyRows * dv <= int.MaxValue && kvHeads <= ushort.MaxValue && SpanPipelinedFunction(dv, 0) is not null)
        {
            foreach (int w in PtxKernels.SpanTensorWarps)
            {
                if (SpanPipelinedFunction(dv, w) is not null)
                {
                    candidates[count++] = SpanPipelined + w;
                }
            }
        }

        return count;
    }

    // The tensor-core span kernel and warps per block, measured for this shape (each candidate writes y, which the call
    // writes again).
    private int MeasureSpanTensor(TuneKey key, int[] candidates, Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, ulong lse,
        int heads, int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale) =>
        Tune(key, candidates, candidates[0],
            c => RunSpanTensor(c, q, keys, values, starts, ends, y, lse, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale));

    private void RunSpanTensor(int candidate, Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, ulong lse, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale)
    {
        if (candidate < SpanPipelined)
        {
            int rowsPerBlock = 16 * candidate;
            Launch(SpanFunction(tensor: true, PtxKernels.SpanTensorDim(dim), candidate)!.Value, (uint)((rows + rowsPerBlock - 1) / rowsPerBlock), (uint)heads, 1,
                (uint)(32 * candidate), 1,
                P(q), P(keys), P(values), P(starts), P(ends), P(y), lse, U(rows), U(keyRows), U(dim), F(scale * Log2E), U(heads / kvHeads), U(headsPerTable));
            return;
        }

        RunSpanPipelined(candidate - SpanPipelined, P(q), P(keys), P(values), starts, ends, P(y), lse, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale,
            SpanLayout.Contiguous(rows, dim), SpanLayout.Contiguous(keyRows, dim), SpanLayout.Contiguous(rows, dim));
    }

    // Where head hh = b·PerBatch + h of a tensor's rows starts (b·Batch + h·Head floats) and how far apart its rows are (Row).
    private readonly record struct SpanLayout(int PerBatch, int Batch, int Head, int Row)
    {
        // [heads, rows, dim]: every head its own batch.
        public static SpanLayout Contiguous(int rows, int dim) => new(1, rows * dim, 0, dim);

        // [batch, rows, *] rows of `row` floats holding `perBatch` heads of `dim` side by side.
        public static SpanLayout Rows(int perBatch, int rows, int row, int dim) => new(perBatch, rows * row, dim, row);
    }

    // Keys and values as bfloat16 rows once (through their layout), then the pipelined kernel with `warps` warps over
    // them (queries and outputs through theirs); the copy goes back to the pool in stream order. Pointers as the kernels
    // take them (offsets added).
    private void RunSpanPipelined(int warps, ulong q, ulong keys, ulong values, Storage starts, Storage ends, ulong y, ulong lse, int heads, int kvHeads,
        int headsPerTable, int rows, int keyRows, int dim, float scale, SpanLayout queries, SpanLayout keyValues, SpanLayout outputs)
    {
        int dv = PtxKernels.SpanPipelinedDim(dim), quads = dv / 4, blockRows = 16 * warps;
        long count = (long)kvHeads * keyRows * dv;                        // bfloat16 values of each copy: both take `count` floats
        var packed = Allocate((int)count, zeroed: false);
        try
        {
            Launch(SpanPipelinedFunction(dv, 0)!.Value, (uint)((keyRows * quads + PtxKernels.SpanPackThreads - 1) / PtxKernels.SpanPackThreads), (uint)kvHeads, 2,
                PtxKernels.SpanPackThreads, 1,
                keys, values, P(packed), U(keyRows), U(dim), U(quads), U(keyValues.PerBatch), U(keyValues.Batch), U(keyValues.Head), U(keyValues.Row));
            ulong packedKeys = P(packed), packedValues = packedKeys + (ulong)count * 2;
            t_sharedBytes = (uint)PtxKernels.SpanPipelinedShared(dv);
            Launch(SpanPipelinedFunction(dv, warps)!.Value, (uint)((rows + blockRows - 1) / blockRows), (uint)heads, 1, (uint)(32 * warps), 1,
                q, packedKeys, packedValues, P(starts), P(ends), y, lse, U(rows), U(keyRows), U(dim), F(scale * Log2E), U(heads / kvHeads),
                U(headsPerTable), U(queries.PerBatch), U(queries.Batch), U(queries.Head), U(queries.Row), U(outputs.Batch),
                U(outputs.Head), U(outputs.Row));
        }
        finally
        {
            packed.Release();
        }
    }

    // AttentionSpans read in place from a layer's packed rows (Backend.AttentionSpansStrided), on the kernel AttentionSpans
    // would take for the head size and precision, so the same values: under tensor cores the pipelined kernel (its layouts
    // in place of the copies), measured against the copies and AttentionSpans (what runs until measured); in float32 the
    // tiled kernel's strided twin (the same arithmetic). Elsewhere, and where AttentionSpans would take another kernel (the
    // single-stage tensor-core one alone, the first float32 one, the host), the copies.
    public override void AttentionSpansStridedKernel(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage starts, Storage ends, Storage y, int batch, int heads, int kvHeads, int headsPerTable, int rows, int dim, float scale)
    {
        int allHeads = batch * heads, allKv = batch * kvHeads;
        bool fits = batch > 0 && heads > 0 && kvHeads > 0 && rows > 0 && heads % kvHeads == 0 && allHeads % headsPerTable == 0
                    && (long)batch * rows * Math.Max(qRow, Math.Max(kRow, heads * dim)) <= int.MaxValue && dim <= PtxKernels.SpanMaxDim
                    && ((qOffset | kOffset | vOffset | (long)qRow | (long)kRow) & 3) == 0;   // the kernels read four floats at a time
        if (!fits)
        {
            base.AttentionSpansStridedKernel(q, qOffset, k, kOffset, v, vOffset, qRow, kRow, starts, ends, y, batch, heads, kvHeads, headsPerTable, rows, dim, scale);
            return;
        }

        var queries = SpanLayout.Rows(heads, rows, qRow, dim);
        var keyValues = SpanLayout.Rows(kvHeads, rows, kRow, dim);
        var outputs = SpanLayout.Rows(heads, rows, heads * dim, dim);
        ulong qp = P(q) + (ulong)qOffset * 4, kp = P(k) + (ulong)kOffset * 4, vp = P(v) + (ulong)vOffset * 4;
        int td = PtxKernels.SpanTensorDim(dim);
        if (MixedPrecision.UsesTensorCores && td > 0 && SpanTensorCandidates(dim, allKv, rows, stackalloc int[2 * PtxKernels.SpanTensorWarps.Length]) > 0)
        {
            // Candidates: 0 = the copies and AttentionSpans (its own measured kernel), or SpanPipelined + warps.
            Span<int> candidates = stackalloc int[1 + PtxKernels.SpanTensorWarps.Length];
            int count = 0;
            candidates[count++] = 0;
            foreach (int w in PtxKernels.SpanTensorWarps)
            {
                if (PtxKernels.SpanPipelinedDim(dim) > 0 && (long)allKv * rows * PtxKernels.SpanPipelinedDim(dim) <= int.MaxValue && allKv <= ushort.MaxValue
                    && SpanPipelinedFunction(PtxKernels.SpanPipelinedDim(dim), 0) is not null && SpanPipelinedFunction(PtxKernels.SpanPipelinedDim(dim), w) is not null)
                {
                    candidates[count++] = SpanPipelined + w;
                }
            }

            void Run(int candidate)
            {
                if (candidate == 0)
                {
                    base.AttentionSpansStridedKernel(q, qOffset, k, kOffset, v, vOffset, qRow, kRow, starts, ends, y, batch, heads, kvHeads, headsPerTable, rows, dim,
                        scale);
                }
                else
                {
                    RunSpanPipelined(candidate - SpanPipelined, qp, kp, vp, starts, ends, P(y), 0UL, allHeads, allKv, headsPerTable, rows, rows, dim, scale,
                        queries, keyValues, outputs);
                }
            }

            var key = new TuneKey(TuneOp.SpanStrided, dim, allHeads, TuneSizes.Class(rows), allKv, headsPerTable);
            int chosen = SpanTensorForced is int forced && candidates[..count].Contains(forced) ? forced
                : TryTuned(key, candidates[..count], 0, out int known) ? known
                : Tune(key, candidates[..count].ToArray(), 0, Run);
            Run(chosen);
            return;
        }

        // Past the tensor cores AttentionSpans takes the float32 tiled kernel where its module loads: its strided twin here.
        int fd = PtxKernels.SpanFloatDim(dim);
        if (fd > 0 && SpanFunction(tensor: false, fd) is not null && SpanFloatStrided(fd) is { } strided)
        {
            t_sharedBytes = (uint)PtxKernels.SpanFloatShared(fd);
            Launch(strided, (uint)((rows + PtxKernels.SpanFloatRows(fd) - 1) / PtxKernels.SpanFloatRows(fd)), (uint)allHeads, 1, 128, 1,
                qp, kp, vp, P(starts), P(ends), P(y), 0UL, U(rows), U(rows), U(dim), F(scale), U(heads / kvHeads), U(headsPerTable), F(0f),
                U(queries.PerBatch), U(queries.Batch), U(queries.Head), U(queries.Row), U(outputs.Batch), U(outputs.Head), U(outputs.Row),
                U(keyValues.PerBatch), U(keyValues.Batch), U(keyValues.Head), U(keyValues.Row));
            return;
        }

        base.AttentionSpansStridedKernel(q, qOffset, k, kOffset, v, vOffset, qRow, kRow, starts, ends, y, batch, heads, kvHeads, headsPerTable, rows, dim, scale);
    }

    // The strided twin of the float32 span kernel of padded head size `d` (its module loaded by SpanFunction).
    private IntPtr? SpanFloatStrided(int d)
    {
        lock (_spanModules)
        {
            return _spanModules.TryGetValue($"f32 {d}", out var kernels) && kernels is not null
                   && kernels.TryGetValue(PtxKernels.SpanFloatStridedName(d), out var function) ? function : null;
        }
    }

    // The pipelined span kernels by head size and warps per block (0: the packing kernel), null when they cannot run here.
    private readonly Dictionary<(int V, int Warps), IntPtr?> _spanPipelinedFunctions = [];

    // A kernel of the pipelined span module for head size `v` (a multiple of 8): the span kernel with `warps` warps per
    // block, or the packing kernel (0). Loads the module on first use, judged by the shared memory the device reports;
    // null when it cannot run.
    private IntPtr? SpanPipelinedFunction(int v, int warps)
    {
        lock (_spanModules)
        {
            if (_spanPipelinedFunctions.TryGetValue((v, warps), out var resolved))
            {
                return resolved;
            }

            string key = $"tcp {v}";
            if (!_spanModules.TryGetValue(key, out var kernels))
            {
                string? reason = null;
                if (_computeMajor < 8)
                {
                    reason = $"compute capability {_computeMajor}.x has no bfloat16 tensor cores (8.0 or newer needed)";
                }
                else
                {
                    var (moduleName, names, source) = PtxKernels.SpanPipelinedModule(v);
                    kernels = LoadDynamicModule(moduleName, names, source, k => k == PtxKernels.SpanPackName ? 0 : PtxKernels.SpanPipelinedShared(v), out reason);
                }

                _spanModules[key] = kernels;
                if (reason is not null)
                {
                    _spanModuleErrors[key] = reason;
                }
            }

            string kernel = warps == 0 ? PtxKernels.SpanPackName : PtxKernels.SpanPipelinedName(v, warps);
            resolved = kernels is not null && kernels.TryGetValue(kernel, out var function) ? function : null;
            _spanPipelinedFunctions[(v, warps)] = resolved;
            return resolved;
        }
    }

    // Span kernels resolved so far by kind, padded head size and warps per block (0 for the float32 kernel), null when one
    // cannot run: a launch formats no name and looks up no module.
    private readonly Dictionary<(bool Tensor, int D, int Warps), IntPtr?> _spanFunctions = [];

    // A span kernel of the module for (kind, padded head size): the tensor-core one with `warps` warps per block, else the
    // float32 one. Loads the module on first use; null when it cannot run.
    private IntPtr? SpanFunction(bool tensor, int d, int warps = 0)
    {
        lock (_spanModules)
        {
            if (_spanFunctions.TryGetValue((tensor, d, warps), out var resolved))
            {
                return resolved;
            }

            string key = $"{(tensor ? "tc" : "f32")} {d}";
            if (!_spanModules.TryGetValue(key, out var kernels))
            {
                kernels = LoadSpanModule(tensor, d, out string? reason);
                _spanModules[key] = kernels;
                if (reason is not null)
                {
                    _spanModuleErrors[key] = reason;
                }
            }

            string kernel = tensor ? PtxKernels.SpanTensorName(d, warps) : PtxKernels.SpanFloatName(d);
            resolved = kernels is not null && kernels.TryGetValue(kernel, out var function) ? function : null;
            _spanFunctions[(tensor, d, warps)] = resolved;
            return resolved;
        }
    }

    private Dictionary<string, IntPtr>? LoadSpanModule(bool tensor, int d, out string? reason)
    {
        reason = null;
        if (tensor && _computeMajor < 8)
        {
            reason = $"compute capability {_computeMajor}.x has no bfloat16 tensor cores (8.0 or newer needed)";
            return null;
        }

        var (moduleName, names, source) = tensor ? PtxKernels.SpanTensorModule(d) : PtxKernels.SpanFloatModule(d);
        int dynamicBytes = tensor ? 0 : PtxKernels.SpanFloatShared(d);
        return LoadDynamicModule(moduleName, names, source, _ => dynamicBytes, out reason);
    }

    // The wide span gradient's kernels (head sizes over 128), loaded on first use; null when the device cannot hold them
    // (the reason in SpanModuleErrors under "bwd wide").
    private (IntPtr Keys, IntPtr Queries)? SpanBackwardWide()
    {
        lock (_spanModules)
        {
            if (!_spanModules.TryGetValue("bwd wide", out var kernels))
            {
                var (moduleName, names, source) = PtxKernels.SpanBackwardWideModule();
                kernels = LoadDynamicModule(moduleName, names, source, PtxKernels.SpanBackwardWideShared, out string? reason);
                _spanModules["bwd wide"] = kernels;
                if (reason is not null)
                {
                    _spanModuleErrors["bwd wide"] = reason;
                }
            }

            return kernels is not null ? (kernels[PtxKernels.SpanBackwardWideKeysName], kernels[PtxKernels.SpanBackwardWideQueriesName]) : null;
        }
    }

    /// <summary>
    /// The largest head size whose attention over key ranges runs on this device's own kernels, forward and gradient:
    /// 128, or up to <see cref="PtxKernels.SpanMaxDim"/> (in steps of 16) where the shared memory the device reports per
    /// block holds the float32 kernel's tiles and the wide gradient's.
    /// </summary>
    internal static int SpanHeadDim(CudaDeviceLimits limits)
    {
        int gradient = Math.Max(PtxKernels.SpanBackwardWideShared(PtxKernels.SpanBackwardWideKeysName),
            PtxKernels.SpanBackwardWideShared(PtxKernels.SpanBackwardWideQueriesName));
        if (gradient > limits.SharedPerBlockOptin)
        {
            return PtxKernels.FlashMaxDim;
        }

        for (int d = PtxKernels.SpanMaxDim; d > PtxKernels.FlashMaxDim; d -= 16)
        {
            if (PtxKernels.SpanFloatShared(d) <= limits.SharedPerBlockOptin)
            {
                return d;
            }
        }

        return PtxKernels.FlashMaxDim;
    }

    // Loads a module whose kernels take `dynamicBytes(kernel)` of dynamic shared memory, judged first by the device's
    // reported limits: static shared memory within what a block may declare, static and dynamic within what it may opt in to.
    private Dictionary<string, IntPtr>? LoadDynamicModule(string moduleName, string[] names, string source, Func<string, int> dynamicBytesOf, out string? reason)
    {
        reason = null;

        foreach (var (kernel, staticBytes) in PtxKernels.StaticSharedBytes(source))
        {
            int dynamicBytes = dynamicBytesOf(kernel);
            if (staticBytes > _limits.SharedPerBlock || staticBytes + dynamicBytes > _limits.SharedPerBlockOptin)
            {
                reason = $"{kernel} needs {staticBytes + dynamicBytes} bytes of shared memory per block ({staticBytes} static); the GPU allows "
                         + $"{_limits.SharedPerBlock} static, {_limits.SharedPerBlockOptin} in all";
                return null;
            }
        }

        MakeCurrent();
        using var quiet = DeviceException.Handled();
        try
        {
            IntPtr module = LoadModule(source, moduleName);
            var counts = PtxKernels.ParameterCountsOf(source);
            var kernels = new Dictionary<string, IntPtr>();
            foreach (var kernel in names)
            {
                byte[] bytes = Encoding.ASCII.GetBytes(kernel + "\0");
                fixed (byte* p = bytes)
                {
                    Check(cuModuleGetFunction(out IntPtr function, module, p), $"cuModuleGetFunction({kernel})");
                    if (dynamicBytesOf(kernel) is > 0 and var dynamicBytes)
                    {
                        Check(cuFuncSetAttribute(function, FunctionAttributeMaxDynamicSharedSizeBytes, dynamicBytes), $"cuFuncSetAttribute({kernel})");
                    }

                    lock (_signatures)
                    {
                        _signatures[function] = (kernel, counts[kernel]);
                    }

                    kernels[kernel] = function;
                }
            }

            return kernels;
        }
        catch (CudaException e)
        {
            reason = $"{moduleName}: {e.Message}";
            return null;
        }
    }

    // ------------------------------------------------------------------ the measured choice of path

    public override bool PrefersComposedAttention(Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage? mask, int heads, int rows,
        int keyRows, int dim, float scale, bool training = false)
    {
        // Candidates: 0 = AttentionSpans (the default, also while nothing can be measured), 1 = composed. Keyed by the
        // precision too (each path's kernels change with it) and by whether the gradient is timed with the forward pass.
        var key = new TuneKey(TuneOp.AttentionPath, (int)MixedPrecision.Current, heads, TuneSizes.Class(rows), TuneSizes.Class(keyRows), dim,
            (mask is null ? 0 : 1) | (training ? 2 : 0));
        ReadOnlySpan<int> candidates = [0, 1];
        if (KnownChoice(key, candidates) is { } known)
        {
            return known == 1;
        }

        long scores = (long)heads * rows * keyRows, outputs = (long)heads * rows * dim, keyFloats = (long)heads * keyRows * dim;
        if (!CanMeasure || scores > int.MaxValue || outputs > int.MaxValue || keyFloats > int.MaxValue)
        {
            return false;
        }

        return MeasureAttentionPath(key, q, keys, values, starts, ends, mask, heads, rows, keyRows, dim, scale, training, scores, outputs, keyFloats);
    }

    // Times both attention paths for `key` (PrefersComposedAttention); true when the composed one won.
    private bool MeasureAttentionPath(TuneKey key, Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage? mask, int heads, int rows,
        int keyRows, int dim, float scale, bool training, long scores, long outputs, long keyFloats)
    {
        // Scratch: the scores and weights (and with the gradient the softmax's gradient, the log-sum-exp and dq, dkeys,
        // dvalues); the output doubles as dOutput in the gradient.
        var scratch = new List<Storage>();
        Storage Scratch(long length)
        {
            var storage = Allocate((int)length, zeroed: true);
            scratch.Add(storage);
            return storage;
        }

        try
        {
            var (s, w, y) = (Scratch(scores), Scratch(scores), Scratch(outputs));
            Storage? g = null, lse = null, dq = null, dk = null, dv = null;
            if (training)
            {
                (g, lse, dq, dk, dv) = (Scratch(scores), Scratch((long)heads * rows), Scratch(outputs), Scratch(keyFloats), Scratch(keyFloats));
            }

            if (Offload is { } offload && scratch.Any(offload.IsOffloaded))
            {
                return false;                                            // in system memory the timing would mislead
            }

            return Tune(key, [0, 1], 0, c =>
            {
                if (!training)
                {
                    if (c == 0)
                    {
                        AttentionSpans(q, keys, values, starts, ends, y, null, heads, heads, heads, rows, keyRows, dim, scale);
                    }
                    else
                    {
                        ComposedAttention(q, keys, values, mask, s, w, y, heads, rows, keyRows, dim, scale);
                    }
                }
                else if (c == 0)
                {
                    AttentionSpans(q, keys, values, starts, ends, y, lse, heads, heads, heads, rows, keyRows, dim, scale);
                    AttentionSpansBackward(q, keys, values, starts, ends, y, lse!, y, dq!, dk!, dv!, heads, heads, heads, rows, keyRows, dim, scale);
                }
                else
                {
                    ComposedAttentionTraining(q, keys, values, mask, s, w, g!, y, y, dq!, dk!, dv!, heads, rows, keyRows, dim, scale);
                }
            }) == 1;
        }
        catch (ResourceLimitExceededException)
        {
            return false;
        }
        finally
        {
            foreach (var storage in scratch)
            {
                storage.Release();
            }
        }
    }

    // A choice measured in this process or kept in the cache file for `key` (when one of `candidates`); null otherwise,
    // or when measuring is off (IDRAK_AUTOTUNE=0: the formulas only).
    private int? KnownChoice(TuneKey key, ReadOnlySpan<int> candidates)
    {
        if (!Autotune)
        {
            return null;
        }

        lock (_tuned)
        {
            if (_tuned.TryGetValue(key, out int known))
            {
                return candidates.Contains(known) ? known : null;
            }

            if (TryPersistedLocked(key, candidates, out int kept))
            {
                _tuned[key] = kept;
                return kept;
            }
        }

        return null;
    }

    public override long? FreeMemory()
    {
        MakeCurrent();
        return cuMemGetInfo(out nuint free, out _) == 0 ? (long)free : null;
    }

    public override long? AvailableMemory()
    {
        MakeCurrent();
        if (cuMemGetInfo(out nuint free, out _) != 0)
        {
            return null;
        }

        return Math.Max(0, (long)free + _memory.Usage.Cached - MemoryReserve);
    }
}
