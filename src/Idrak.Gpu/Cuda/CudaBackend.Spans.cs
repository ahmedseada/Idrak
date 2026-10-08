// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using static Idrak.Gpu.Cuda.CudaDriver;

namespace Idrak.Gpu.Cuda;

// Attention over one range of keys per query row (PtxKernels.SpansTiled.cs): the tensor-core kernel when MixedPrecision
// asks for tensor cores and the GPU has them (no soft-cap; its warps per block measured per shape), else the float32
// tiled kernel, else (a head size it does not take, or a GPU whose shared memory per block cannot hold its tiles) the
// first kernel, attention_spans_f32; head sizes over 128 and every gradient take the host fallback. Each head size's
// module is built and loaded the first time it is needed, and judged by what the device reports (shared memory per
// block) before it is loaded. And the measured choice between this operation and the composed scores
// (Backend.PrefersComposedAttention, for Tensor.AttentionFastest), kept with the other measured choices.
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
            int chosen = Tune(new TuneKey(TuneOp.SpanWarps, td, heads, rows, keyRows, kvHeads, headsPerTable), warps, PtxKernels.SpanTensorWarps[0],
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
        int keyRows, int dim, float scale)
    {
        // Candidates: 0 = AttentionSpans (the default, also while nothing can be measured), 1 = composed. Keyed by the
        // precision too: each path's kernels change with it.
        var key = new TuneKey(TuneOp.AttentionPath, (int)MixedPrecision.Current, heads, rows, keyRows, dim, mask is null ? 0 : 1);
        ReadOnlySpan<int> candidates = [0, 1];
        if (KnownChoice(key, candidates) is { } known)
        {
            return known == 1;
        }

        long scores = (long)heads * rows * keyRows, outputs = (long)heads * rows * dim;
        if (!CanMeasure || scores > int.MaxValue || outputs > int.MaxValue)
        {
            return false;
        }

        Storage? s = null, w = null, y = null;
        try
        {
            s = Allocate((int)scores, zeroed: false);
            w = Allocate((int)scores, zeroed: false);
            y = Allocate((int)outputs, zeroed: false);
            if (Offload is { } offload && (offload.IsOffloaded(s) || offload.IsOffloaded(w)))
            {
                return false;                                            // in system memory the timing would mislead
            }

            var (sc, wc, yc) = (s, w, y);
            return Tune(key, candidates, 0, c =>
            {
                if (c == 0)
                {
                    AttentionSpans(q, keys, values, starts, ends, yc, null, heads, heads, heads, rows, keyRows, dim, scale);
                }
                else
                {
                    ComposedAttention(q, keys, values, mask, sc, wc, yc, heads, rows, keyRows, dim, scale);
                }
            }) == 1;
        }
        catch (ResourceLimitExceededException)
        {
            return false;
        }
        finally
        {
            s?.Release();
            w?.Release();
            y?.Release();
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
                return known;
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
