// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using Idrak.Abstraction.Operations;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Attention over one range of keys per query row (PtxKernels.SpansTiled.cs): the float32 tiled kernel (head sizes up to
// 256: 32 query rows a block over 128), else (a head size it does not take, or a GPU whose
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
    // The span modules by padded head size ("f32 72"): their kernels, or null when the module cannot
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
        int fd = PtxKernels.SpanFloatDim(dim);
        if (fd > 0 && SpanFunction(fd) is { } tiled)
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

    // Where head hh = b·PerBatch + h of a tensor's rows starts (b·Batch + h·Head floats) and how far apart its rows are (Row).
    private readonly record struct SpanLayout(int PerBatch, int Batch, int Head, int Row)
    {
        // [batch, rows, *] rows of `row` floats holding `perBatch` heads of `dim` side by side.
        public static SpanLayout Rows(int perBatch, int rows, int row, int dim) => new(perBatch, rows * row, dim, row);
    }

    // AttentionSpans read in place from a layer's packed rows (Backend.AttentionSpansStrided), on the kernel AttentionSpans
    // would take for the head size, so the same values: the tiled kernel's strided twin (the same arithmetic). Where
    // AttentionSpans would take another kernel (the first float32 one, the host), the copies.
    public override void AttentionSpansStridedKernel(Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
        Storage starts, Storage ends, Storage y, int batch, int heads, int kvHeads, int headsPerTable, int rows, int dim, float scale)
    {
        int allHeads = batch * heads;
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

        // AttentionSpans takes the float32 tiled kernel where its module loads: its strided twin here.
        int fd = PtxKernels.SpanFloatDim(dim);
        if (fd > 0 && SpanFunction(fd) is not null && SpanFloatStrided(fd) is { } strided)
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

    // Span kernels resolved so far by padded head size, null when one cannot run: a launch formats no name and looks up no
    // module.
    private readonly Dictionary<int, IntPtr?> _spanFunctions = [];

    // The float32 span kernel of the module for a padded head size. Loads the module on first use; null when it cannot run.
    private IntPtr? SpanFunction(int d)
    {
        lock (_spanModules)
        {
            if (_spanFunctions.TryGetValue(d, out var resolved))
            {
                return resolved;
            }

            string key = $"f32 {d}";
            if (!_spanModules.TryGetValue(key, out var kernels))
            {
                kernels = LoadSpanModule(d, out string? reason);
                _spanModules[key] = kernels;
                if (reason is not null)
                {
                    _spanModuleErrors[key] = reason;
                }
            }

            resolved = kernels is not null && kernels.TryGetValue(PtxKernels.SpanFloatName(d), out var function) ? function : null;
            _spanFunctions[d] = resolved;
            return resolved;
        }
    }

    private Dictionary<string, IntPtr>? LoadSpanModule(int d, out string? reason)
    {
        var (moduleName, names, source) = PtxKernels.SpanFloatModule(d);
        int dynamicBytes = PtxKernels.SpanFloatShared(d);
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
            var signatures = new List<KeyValuePair<IntPtr, (string Name, int Parameters)>>();
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

                    signatures.Add(new(function, (kernel, counts[kernel])));
                    kernels[kernel] = function;
                }
            }

            AddSignatures(signatures);                              // published before the caller can launch them
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
        using var operation = EnterOperation();                            // addresses read until the work is queued (the move guard)
        // Candidates: 0 = AttentionSpans (the default, also while nothing can be measured), 1 = composed. Keyed by whether
        // the gradient is timed with the forward pass.
        var key = new TuneKey(TuneOp.AttentionPath, 0, heads, TuneSizes.Class(rows), TuneSizes.Class(keyRows), dim,
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

        lock (_measuring)                                                  // one measurement at a time, its scratch made once
        {
            return KnownChoice(key, candidates) is { } measured ? measured == 1
                : MeasureAttentionPath(key, q, keys, values, starts, ends, mask, heads, rows, keyRows, dim, scale, training, scores, outputs, keyFloats);
        }
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
            // The scratch (the full scores: often the largest blocks a model ever takes) goes back to the driver, not to
            // the pool: its sizes come back only for the composed path itself, chosen rarely.
            var blocks = new HashSet<ulong>();
            foreach (var storage in scratch)
            {
                if (storage is CudaStorage block)
                {
                    blocks.Add(block.Pointer);
                }

                storage.Release();
            }

            FreeCachedBlocks(blocks);
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

    long? IMemoryReport.FreeMemory()
    {
        MakeCurrent();
        return cuMemGetInfo(out nuint free, out _) == 0 ? (long)free : null;
    }

    long? IMemoryReport.AvailableMemory()
    {
        MakeCurrent();
        if (cuMemGetInfo(out nuint free, out _) != 0)
        {
            return null;
        }

        return Math.Max(0, (long)free + _memory.Usage.Cached - MemoryReserve);
    }
}
