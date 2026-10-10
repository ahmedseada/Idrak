// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Attention over one range of keys per query row (PtxKernels.SpansTiled.cs): the tensor-core kernel when MixedPrecision
// asks for tensor cores and the GPU has them (no soft-cap; its warps per block measured per shape), else the float32
// tiled kernel, else (a head size it does not take, or a GPU whose shared memory per block cannot hold its tiles) the
// first kernel, attention_spans_f32; head sizes over 128 take the host fallback. Each head size's module is built and
// loaded the first time it is needed, and judged by what the device reports (shared memory per block) before it is
// loaded. The gradient runs on attn_spans_bwd_kv_f32 and attn_spans_bwd_q_f32 (PtxKernels.SpansBackward.cs; float32,
// head sizes up to 128, the host fallback beyond). And the measured choice between this operation and the composed
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
        if (dim <= 0 || dim > PtxKernels.FlashMaxDim || kvHeads <= 0 || headsPerTable <= 0)
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
        if (variant.Softcap <= 0f && MixedPrecision.UsesTensorCores && td > 0 && SpanFunction(tensor: true, td, PtxKernels.SpanTensorName(td, PtxKernels.SpanTensorWarps[0])) is not null)
        {
            var warps = PtxKernels.SpanTensorWarps.Where(w => SpanFunction(tensor: true, td, PtxKernels.SpanTensorName(td, w)) is not null).ToArray();
            int chosen = Tune(new TuneKey(TuneOp.SpanWarps, td, heads, TuneSizes.Class(rows), TuneSizes.Class(keyRows), kvHeads, headsPerTable), warps, PtxKernels.SpanTensorWarps[0],
                w => LaunchSpanTensor(td, w, q, keys, values, starts, ends, y, lse, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale));
            LaunchSpanTensor(td, chosen, q, keys, values, starts, ends, y, lse, heads, kvHeads, headsPerTable, rows, keyRows, dim, scale);
            return;
        }

        int fd = PtxKernels.SpanFloatDim(dim);
        if (fd > 0 && SpanFunction(tensor: false, fd, PtxKernels.SpanFloatName(fd)) is { } tiled)
        {
            t_sharedBytes = (uint)PtxKernels.SpanFloatShared(fd);
            Launch(tiled, (uint)((rows + PtxKernels.SpanFloatRows - 1) / PtxKernels.SpanFloatRows), (uint)heads, 1, 128, 1,
                P(q), P(keys), P(values), P(starts), P(ends), P(y), lse, U(rows), U(keyRows), U(dim), F(scale), U(heads / kvHeads), U(headsPerTable),
                F(variant.Softcap));
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
        if (dim <= 0 || dim > PtxKernels.FlashMaxDim || kvHeads <= 0 || headsPerTable <= 0 || (long)heads * rows > int.MaxValue)
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
            Launch(K(PtxKernels.SpanBackwardKeysName), (uint)((keyRows + PtxKernels.SpanBackwardKeys - 1) / PtxKernels.SpanBackwardKeys), (uint)kvHeads, 1,
                128, 1, args);
            Launch(K(PtxKernels.SpanBackwardQueriesName), (uint)((rows + PtxKernels.SpanBackwardRows - 1) / PtxKernels.SpanBackwardRows), (uint)heads, 1,
                128, 1, args);
        }
        finally
        {
            delta.Release();                                                  // reused in stream order
        }
    }

    private void LaunchSpanTensor(int td, int warps, Storage q, Storage keys, Storage values, Storage starts, Storage ends, Storage y, ulong lse, int heads,
        int kvHeads, int headsPerTable, int rows, int keyRows, int dim, float scale)
    {
        int rowsPerBlock = 16 * warps;
        Launch(SpanFunction(tensor: true, td, PtxKernels.SpanTensorName(td, warps))!.Value, (uint)((rows + rowsPerBlock - 1) / rowsPerBlock), (uint)heads, 1,
            (uint)(32 * warps), 1,
            P(q), P(keys), P(values), P(starts), P(ends), P(y), lse, U(rows), U(keyRows), U(dim), F(scale * Log2E), U(heads / kvHeads), U(headsPerTable));
    }

    // A span kernel of the module for (kind, padded head size), loading the module on first use; null when it cannot run.
    private IntPtr? SpanFunction(bool tensor, int d, string kernel)
    {
        string key = $"{(tensor ? "tc" : "f32")} {d}";
        lock (_spanModules)
        {
            if (!_spanModules.TryGetValue(key, out var kernels))
            {
                kernels = LoadSpanModule(tensor, d, out string? reason);
                _spanModules[key] = kernels;
                if (reason is not null)
                {
                    _spanModuleErrors[key] = reason;
                }
            }

            return kernels is not null && kernels.TryGetValue(kernel, out var function) ? function : null;
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

        // Judged by the device's reported limits: static shared memory within what a block may declare, static and
        // dynamic within what a block may opt in to.
        foreach (var (kernel, staticBytes) in PtxKernels.StaticSharedBytes(source))
        {
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
                    if (dynamicBytes > 0)
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

            return Tune(key, candidates, 0, c =>
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
