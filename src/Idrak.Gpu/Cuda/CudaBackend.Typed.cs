// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace Idrak.Gpu.Cuda;

// The product routes on CUDA (ProductRoutes): products of int8 and bfloat16 weights by their own kernels
// (ProductRoute.Name, today), by one generic product per weight description (ProductRoute.Generic: PtxKernels.Typed.cs,
// widening any type from its description inside the kernel), or by the faster of the two, measured once per shape
// (ProductRoute.Measured: Tune, the named kernel unless the generic one is clearly faster). Weights kept as stored (q8_0
// and any type with a description here) take the generic product under the generic and measured routes; today CUDA has
// no product of its own for them and a load expands them to float32.
internal sealed unsafe partial class CudaBackend
{
    // A description's generic kernels, looked up by row count without formatting a name (the per-token path).
    private sealed class TypedFunctions(Dictionary<string, IntPtr> kernels, TypedDecode d)
    {
        public IntPtr TileInputs { get; } = d.PerValue ? kernels[PtxKernels.TypedTileName(d, false)] : IntPtr.Zero;

        public IntPtr TileOutputs { get; } = kernels[PtxKernels.TypedTileName(d, true)];

        public IntPtr Sum { get; } = kernels[PtxKernels.TypedSumName];

        public IntPtr[] GemvInputs { get; } = d.PerValue ? [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, false, r)])] : [];

        public IntPtr[] GemvOutputs { get; } = [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, true, r)])];

        // Outputs-major few-row products reading 16 bytes a lane a pass (rows of whole 16-byte groups).
        public IntPtr[] GemvOutputsWide { get; } = d.PerValue ? [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, true, r, wide: true)])] : [];

        // The same, grouped: 2 or 3 weights sharing x in one launch (StoredMatMulManyKernel).
        public IntPtr[] GemvOutputsMany { get; } = [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, true, r) + "_many"])];

        public IntPtr[] GemvOutputsWideMany { get; } = d.PerValue ? [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, true, r, wide: true) + "_many"])] : [];

        // With the fused epilogues (PtxKernels.Epilogues.cs): the gate/up pair's act(gate) · up, a product's residual and norm.
        public IntPtr[] GemvGated { get; } = [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, true, r) + "_many_gated"])];

        public IntPtr[] GemvWideGated { get; } = d.PerValue ? [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, true, r, wide: true) + "_many_gated"])] : [];

        public IntPtr[] GemvAddNorm { get; } = [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, true, r) + "_addnorm"])];

        public IntPtr[] GemvWideAddNorm { get; } = d.PerValue ? [.. PtxKernels.TypedGemvRows.ToArray().Select(r => kernels[PtxKernels.TypedGemvName(d, true, r, wide: true) + "_addnorm"])] : [];
    }

    // Each type's description once (null: these kernels cannot widen it).
    private static readonly ConcurrentDictionary<ElementType, TypedDecode?> TypedDecodes = new();

    internal static TypedDecode? DecodeOf(ElementType type) => TypedDecodes.GetOrAdd(type, TypedDecode.Of);

    // The modules built and loaded on first use, by description (null: it did not load, TypedUnavailableReason says why).
    private readonly ConcurrentDictionary<TypedDecode, TypedFunctions?> _typed = new();
    private readonly Lock _typedLock = new();

    /// <summary>Why the last generic module did not load, or null (diagnostics).</summary>
    internal string? TypedUnavailableReason { get; private set; }

    private TypedFunctions? Typed(TypedDecode d) => _typed.TryGetValue(d, out var functions) ? functions : LoadTyped(d);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private TypedFunctions? LoadTyped(TypedDecode d)
    {
        lock (_typedLock)
        {
            if (_typed.TryGetValue(d, out var known))
            {
                return known;
            }

            var (names, source) = PtxKernels.TypedModule(d);
            var kernels = LoadDynamicModule($"typed_{d.Code}", names, source, static _ => 0, out string? reason);
            var functions = kernels is null ? null : new TypedFunctions(kernels, d);
            if (kernels is null)
            {
                TypedUnavailableReason = reason;
            }

            _typed[d] = functions;
            return functions;
        }
    }

    // The few-row kernel compiled for the fewest rows that hold m (PtxKernels.TypedGemvRows).
    private static int TypedRowVariant(int m)
    {
        var rows = PtxKernels.TypedGemvRows;
        for (int i = 0; i < rows.Length; i++)
        {
            if (m <= rows[i])
            {
                return i;
            }
        }

        return rows.Length - 1;
    }

    /// <summary>
    /// y [m, n] = x [m, k] · W for W of <paramref name="type"/> stored inputs-major ([k, ld], value (r, c) at r · ld + c,
    /// as the int8 and bfloat16 packed weights lie), each value times its column's scale when <paramref name="scales"/> is
    /// given: the generic product, widening the type from its description. False when the type has no description these
    /// kernels widen or the rows are not whole words.
    /// </summary>
    internal bool TypedInputsMajor(ElementType type, Storage x, Storage w, Storage? scales, Storage y, int m, int n, int k, int ld)
    {
        if (DecodeOf(type) is not { PerValue: true } d || ld % d.PerWord != 0 || Typed(d) is not { } functions)
        {
            return false;
        }

        if (m == 0 || n == 0)
        {
            return true;
        }

        if (k == 0)
        {
            Fill(y, m * n, 0f);
            return true;
        }

        ulong s = scales is null ? 0UL : P(scales);
        if (m > PtxKernels.TypedGemvRows[^1])
        {
            Launch(functions.TileInputs, (uint)((n + PtxKernels.TypedTileColumns - 1) / PtxKernels.TypedTileColumns),
                (uint)((m + PtxKernels.TypedTileRows - 1) / PtxKernels.TypedTileRows), PtxKernels.TypedTileThreads, 1,
                P(x), P(w), s, P(y), U(m), U(n), U(k), U(ld));
            return true;
        }

        // A word of columns a thread, the rows split so the launch has about 256 blocks (from the shape alone, so the
        // parts and their order follow the sizes, never the device), chunks of 64 rows at least.
        int words = ld / d.PerWord, blocks = (words + PtxKernels.TypedGemvThreads - 1) / PtxKernels.TypedGemvThreads;
        int wanted = Math.Clamp((256 + blocks - 1) / blocks, 1, Math.Max(1, k / 64));
        int chunk = ((k + wanted - 1) / wanted + 3) / 4 * 4;
        int splits = (k + chunk - 1) / chunk;
        var kernel = functions.GemvInputs[TypedRowVariant(m)];
        if (splits == 1)
        {
            Launch(kernel, (uint)blocks, 1, PtxKernels.TypedGemvThreads, 1, P(x), P(w), s, P(y), P(y), U(m), U(n), U(k), U(ld), U(chunk), U(1));
            return true;
        }

        var part = _splitPartials.Get(this, splits * m * n);
        Launch(kernel, (uint)blocks, (uint)splits, PtxKernels.TypedGemvThreads, 1, P(x), P(w), s, P(y), P(part), U(m), U(n), U(k), U(ld), U(chunk), U(splits));
        Launch(functions.Sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(part), P(y), U(m * n), U(splits));
        return true;
    }

    /// <summary>
    /// y [m, n] = x [m, k] · W for W of <paramref name="type"/> stored outputs-major (n rows of k values, as a file stores
    /// them), each value times its row's scale when <paramref name="rowScales"/> is given: the generic product. False when
    /// the type has no description these kernels widen, a row is not whole words or blocks, or a block format has scales.
    /// </summary>
    internal bool TypedOutputsMajor(ElementType type, Storage x, Storage w, Storage? rowScales, Storage y, int m, int n, int k)
    {
        if (DecodeOf(type) is not { } d || (d.PerValue ? (long)k * d.Bits % 32 != 0 : k % PtxKernels.TypedTileDepth != 0 || rowScales is not null)
            || Typed(d) is not { } functions)
        {
            return false;
        }

        if (m == 0 || n == 0)
        {
            return true;
        }

        if (k == 0)
        {
            Fill(y, m * n, 0f);
            return true;
        }

        ulong s = rowScales is null ? 0UL : P(rowScales);
        if (m > PtxKernels.TypedGemvRows[^1])
        {
            Launch(functions.TileOutputs, (uint)((n + PtxKernels.TypedTileColumns - 1) / PtxKernels.TypedTileColumns),
                (uint)((m + PtxKernels.TypedTileRows - 1) / PtxKernels.TypedTileRows), PtxKernels.TypedTileThreads, 1,
                P(x), P(w), s, P(y), U(m), U(n), U(k), U(k));
            return true;
        }

        bool wide = d.PerValue && (long)k * d.Bits / 8 % 16 == 0;
        Launch((wide ? functions.GemvOutputsWide : functions.GemvOutputs)[TypedRowVariant(m)], (uint)((n + PtxKernels.TypedWarpColumns - 1) / PtxKernels.TypedWarpColumns), 1,
            (uint)(PtxKernels.TypedWarpColumns * 32), 1, P(x), P(w), s, P(y), U(m), U(n), U(k));
        return true;
    }

    // ------------------------------------------------------------------ the matrix-unit lane (measured route)

    // Each description's tensor-core kernel, built and loaded on first use where the device reaches its target (null: it
    // cannot, TypedUnavailableReason says why).
    private readonly ConcurrentDictionary<TypedDecode, IntPtr?> _typedMma = new();

    // A number for each description, for the measured choices' keys (TuneKey holds numbers only).
    private static readonly ConcurrentDictionary<TypedDecode, int> TypedIds = new();

    private IntPtr? TypedMma(TypedDecode d) => _typedMma.TryGetValue(d, out var function) ? function : LoadTypedMma(d);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private IntPtr? LoadTypedMma(TypedDecode d)
    {
        lock (_typedLock)
        {
            if (_typedMma.TryGetValue(d, out var known))
            {
                return known;
            }

            IntPtr? function = null;
            if (BelowTarget(PtxKernels.TensorCoreTarget) is { } below)
            {
                TypedUnavailableReason = below;
            }
            else
            {
                var (names, source) = PtxKernels.TypedMmaModule(d);
                var kernels = LoadDynamicModule($"typed_mma_{d.Code}", names, source, static _ => 0, out string? reason);
                if (kernels is null)
                {
                    TypedUnavailableReason = reason;
                }
                else
                {
                    function = kernels[names[0]];
                }
            }

            _typedMma[d] = function;
            return function;
        }
    }

    // Whether the matrix-unit lane can take this product: the run computes in bfloat16 (MixedPrecision), the type's values
    // widen into bfloat16 tiles (TypedDecode.MatrixUnits), k is whole steps of 32, and the kernel loads on this device.
    private bool MatrixUnitLane(TypedDecode d, int m, int k) =>
        m > PtxKernels.TypedGemvRows[^1] && BFloat16Products && d.MatrixUnits && k % PtxKernels.TypedTileDepth == 0 && TypedMma(d) is not null;

    private void RunTypedMma(TypedDecode d, Storage x, Storage w, Storage? rowScales, Storage y, int m, int n, int k) =>
        Launch(TypedMma(d)!.Value, (uint)((n + 63) / 64), (uint)((m + 63) / 64), 128, 1, P(x), P(w), rowScales is null ? 0UL : P(rowScales), P(y), U(m), U(n), U(k));

    // The measured route's product of stored weights: of the lanes the device and the type's description allow (the device
    // accepts the instruction, the description fits it), whichever measures fastest at the shape on the real operands (Tune:
    // the f32 tile or the few-row lane unless another is clearly faster), once: the f32 tile or the few-row lane (0), the
    // simple matrix-unit kernel (1, TypedMma), the native bfloat16 kind (2, the operands packed into the kind's rows, then the
    // shape-measured kernel: NativeMma), the native kind decoding in the kernel (4), reading int8 codes raw (5) and the
    // tensor rows (6). Each shape's lane recorded (ProductRoutes.Chose).
    private bool StoredMeasured(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k)
    {
        if (DecodeOf(weight.ElementType) is not { } d)
        {
            return false;
        }

        bool few = m <= PtxKernels.TypedGemvRows[^1];
        Span<int> candidates = stackalloc int[7];
        int count = 0;
        candidates[count++] = 0;
        if (MatrixUnitLane(d, m, k) && (scales is null || d.PerValue))
        {
            candidates[count++] = 1;
            if (k % 8 == 0 && NativeKernel(PtxKernels.NativeKind.For(d)) is not null && NativePack(d, PtxKernels.NativeKind.For(d)) is not null)
            {
                candidates[count++] = 2;
            }
        }

        if (NativeDirectLane(weight.ElementType, scales, m, k))
        {
            candidates[count++] = 4;
        }

        if (MatrixUnitLane(d, m, k) && NativeRawLane(d, m, k))
        {
            candidates[count++] = 5;
        }

        if (TensorRowsSource(weight.ElementType, scales, m, k) is not null)
        {
            candidates[count++] = 6;
        }

        int lane = 0;
        if (count > 1)
        {
            var key = new TuneKey(TuneOp.ProductRoute, 1000 + TypedIds.GetOrAdd(d, static _ => TypedIds.Count), few ? m : TuneSizes.Class(m), n, k, count);
            if (!TryTuned(key, candidates[..count], 0, out lane))
            {
                lane = MeasureStoredRoute(key, candidates[..count].ToArray(), d, x, weight, scales, y, m, n, k);
            }
        }

        if (!RunStoredLane(lane, d, x, weight, scales, y, m, n, k))
        {
            return false;
        }

        if (_storedSaid.TryAdd((d, few ? m : TuneSizes.Class(m), n, k, BFloat16Products), 0))
        {
            SayStored(weight.ElementType, m, n, k, lane switch
            {
                6 => "tensor rows (today's tensor-core kernel over the rows as stored, or widened to bfloat16 rows; k split in the kernel)",
                5 => "native bfloat16 kind (int8 codes read raw, no packing pass)",
                4 => "native bfloat16 kind (decoded in the kernel)",
                2 => "native bfloat16 kind (packed)",
                1 => "matrix units (simple kernel)",
                _ when few => "few-row lane",
                _ => count > 1 ? "f32 tile" : "f32 tile (the only lane)",
            });
        }

        return true;
    }

    // One lane of a stored product (0 the f32 tile or the few-row lane, 1 the simple matrix-unit kernel, 2 the native kind,
    // 4 the native kind decoding in the kernel, 5 the native kind reading int8 codes raw, 6 the tensor
    // rows: today's tensor-core kernel over the rows as stored, CudaBackend.TensorRows.cs).
    private bool RunStoredLane(int lane, TypedDecode d, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k)
    {
        switch (lane)
        {
            case 1:
                RunTypedMma(d, x, weight, scales, y, m, n, k);
                return true;
            case 2 when RunNative(d, x, weight, scales, y, m, n, k):
                return true;
            case 4 when RunNativeDirect(x, weight, scales, y, m, n, k):
                return true;
            case 5 when RunNativeRaw(d, x, weight, scales, y, m, n, k):
                return true;
            case 6 when RunTensorRows(x, weight, scales, y, m, n, k):
                return true;
            default:
                return TypedOutputsMajor(weight.ElementType, x, weight, scales, y, m, n, k);
        }
    }

    private readonly ConcurrentDictionary<(TypedDecode, int, int, int, bool), byte> _storedSaid = new();

    [MethodImpl(MethodImplOptions.NoInlining)]
    // Timed warm: these lanes read contiguous rows, and warm repeats chose the lanes that were faster in the passes (cold
    // timings here cost bfloat16, int8 and q4_0 prompts 6%, measured).
    private int MeasureStoredRoute(TuneKey key, int[] candidates, TypedDecode d, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k) =>
        Tune(key, candidates, 0, c => RunStoredLane(c, d, x, weight, scales, y, m, n, k));

    // ------------------------------------------------------------------ the native bfloat16 kind (the packed tensor kernel)

    // A kind's chosen shape, whether it was measured, and every shape the driver accepted (the per-product-shape choice
    // measures among them and the k splits), each by the kind's place in NativeKind.All.
    private readonly (IntPtr Function, PtxKernels.NativeMmaConfig Config)?[] _native = new (IntPtr, PtxKernels.NativeMmaConfig)?[PtxKernels.NativeKind.All.Length];
    private readonly bool[] _nativeTried = new bool[PtxKernels.NativeKind.All.Length];
    private readonly List<(IntPtr Function, PtxKernels.NativeMmaConfig Config, int Index)>[] _nativeAccepted = [.. PtxKernels.NativeKind.All.Select(_ => new List<(IntPtr, PtxKernels.NativeMmaConfig, int)>())];

    // A kind's tuning keys: bfloat16's as before (variants 9000, 9100), each other kind 50 on.
    private static int NativeKey(PtxKernels.NativeKind kind, int variant) => variant + 50 * kind.Index;

    /// <summary>
    /// The device's native bfloat16 tensor kernel: of NativeMmaConfigs, those whose shared memory the
    /// device allows and whose PTX its driver accepts (the kind native here), the one that runs a 2048³ product fastest,
    /// measured once (Tune, kept in the tuning cache); null when the driver accepts none.
    /// </summary>
    internal (IntPtr Function, PtxKernels.NativeMmaConfig Config)? NativeKernel(PtxKernels.NativeKind? of = null)
    {
        var kind = of ?? PtxKernels.NativeKind.BF16;
        int at = kind.Index;
        if (Volatile.Read(ref _nativeTried[at]))
        {
            return _native[at];
        }

        lock (_typedLock)
        {
            if (!_nativeTried[at])
            {
                try
                {
                    _native[at] = MeasureNative(kind);
                }
                finally
                {
                    Volatile.Write(ref _nativeTried[at], true);
                }
            }

            return _native[at];
        }
    }

    private (IntPtr Function, PtxKernels.NativeMmaConfig Config)? MeasureNative(PtxKernels.NativeKind kind)
    {
        var accepted = new List<(IntPtr Function, PtxKernels.NativeMmaConfig Config, int Index)>();
        var configs = PtxKernels.NativeMmaConfigs;
        for (int i = 0; i < configs.Length; i++)
        {
            int shared = PtxKernels.NativeMmaShared(configs[i]);
            if (shared > _limits.SharedPerBlockOptin)
            {
                continue;
            }

            string source = PtxKernels.NativeMma(kind, configs[i], out string entry);
            int dynamic = shared > 48 * 1024 ? shared : 0;
            if (LoadDynamicModule(entry, [entry], source, _ => dynamic, out string? reason) is { } kernels)
            {
                accepted.Add((kernels[entry], configs[i], i));
            }
            else
            {
                TypedUnavailableReason = reason;
            }
        }

        _nativeAccepted[kind.Index] = accepted;
        if (accepted.Count == 0)
        {
            return null;
        }

        const int Size = 2048;
        var a = Allocate(Size * Size / 2, zeroed: true);
        var b = Allocate(Size * Size / 2, zeroed: true);
        var c = Allocate(Size * Size, zeroed: false);
        try
        {
            int[] candidates = [.. Enumerable.Range(0, accepted.Count)];
            int chosen = Tune(new TuneKey(TuneOp.ProductRoute, NativeKey(kind, 9000), Size, Size, Size, accepted.Count), candidates, 0,
                i => LaunchNative(accepted[i].Function, accepted[i].Config, a, b, c, Size, Size, Size, 1));
            var best = accepted[chosen];
            var cfg = best.Config;
            ProductRoutes.Chose(Kind, kind == PtxKernels.NativeKind.BF16 ? "native bfloat16 tensor kernel" : $"native {kind.Name} tensor kernel", string.Create(System.Globalization.CultureInfo.InvariantCulture,
                $"rows of {cfg.RowBytes} B, {cfg.Stages} stages, {cfg.Bn} columns{(cfg.Swizzle ? ", swizzled" : "")} ({accepted.Count} of {configs.Length} shapes accepted)"));
            return (best.Function, cfg);
        }
        finally
        {
            a.Release();
            b.Release();
            c.Release();
        }
    }

    // The native kernel over `splits` ranges of k's blocks (grid z): with one, C is y; with more, part z of `c` ([splits, m, n]).
    private void LaunchNative(IntPtr function, PtxKernels.NativeMmaConfig cfg, Storage a, Storage b, Storage c, int m, int n, int kp, int splits)
    {
        int shared = PtxKernels.NativeMmaShared(cfg);
        int blockValues = cfg.RowBytes * 8 / 16, blocks = kp / blockValues, per = (blocks + splits - 1) / splits;
        t_sharedBytes = shared > 48 * 1024 ? (uint)shared : 0;
        Launch(function, (uint)((n + cfg.Bn - 1) / cfg.Bn), (uint)((m + 127) / 128), (uint)splits, PtxKernels.NativeMmaThreads, 1,
            P(a), P(b), P(c), U(m), U(n), U(kp), U(kp), U(per));
    }

    // The k splits a native product may take: 1, 2, 4 and 8, while each split keeps a block of k at least.
    private static int[] NativeSplits(PtxKernels.NativeMmaConfig cfg, int kp)
    {
        int blocks = kp / (cfg.RowBytes * 8 / 16);
        return [.. new[] { 1, 2, 4, 8 }.Where(s => s == 1 || blocks >= 2 * s)];
    }

    // The native product's shape and k split for this product's shape: of every accepted shape and its splits, the fastest
    // measured on the real operands (Tune: the device's shape unsplit unless another is clearly faster), once.
    private (int Shape, int Splits) NativeChoice(PtxKernels.NativeKind kind, Storage px, Storage pw, Storage y, int m, int n, int kp)
    {
        var shapes = _nativeAccepted[kind.Index];
        var choices = new List<int>();
        int fallback = -1;
        for (int i = 0; i < shapes.Count; i++)
        {
            foreach (int s in NativeSplits(shapes[i].Config, kp))
            {
                int code = i * 16 + s;
                choices.Add(code);
                if (s == 1 && shapes[i].Function == _native[kind.Index]!.Value.Function)
                {
                    fallback = code;
                }
            }
        }

        var key = new TuneKey(TuneOp.ProductRoute, NativeKey(kind, 9100), TuneSizes.Class(m), n, kp, shapes.Count);
        if (!TryTuned(key, choices.ToArray(), fallback, out int chosen))
        {
            chosen = Tune(key, choices.ToArray(), fallback, code => RunNativeShape(kind, code / 16, code % 16, px, pw, y, m, n, kp));
        }

        return (chosen / 16, chosen % 16);
    }

    // The native kernel of accepted shape `shape` over `splits` ranges of k; splits added in order into y (typed_sum_parts).
    private void RunNativeShape(PtxKernels.NativeKind kind, int shape, int splits, Storage px, Storage pw, Storage y, int m, int n, int kp)
    {
        var (function, cfg, _) = _nativeAccepted[kind.Index][shape];
        if (splits == 1)
        {
            LaunchNative(function, cfg, px, pw, y, m, n, kp, 1);
            return;
        }

        var part = _splitPartials.Get(this, splits * m * n);
        LaunchNative(function, cfg, px, pw, part, m, n, kp, splits);
        var sum = Typed(DecodeOf(ElementType.Float32)!)!.Sum;
        Launch(sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(part), P(y), U(m * n), U(splits));
    }

    // Each description's packing kernels into a kind (x, then W into the kind's rows), loaded on first use.
    private readonly ConcurrentDictionary<(TypedDecode, PtxKernels.NativeKind), (IntPtr X, IntPtr W)?> _nativePack = new();

    private (IntPtr X, IntPtr W)? NativePack(TypedDecode d, PtxKernels.NativeKind? of = null)
    {
        var kind = of ?? PtxKernels.NativeKind.BF16;
        return _nativePack.TryGetValue((d, kind), out var pack) ? pack : LoadNativePack(d, kind);
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private (IntPtr X, IntPtr W)? LoadNativePack(TypedDecode d, PtxKernels.NativeKind kind)
    {
        lock (_typedLock)
        {
            if (_nativePack.TryGetValue((d, kind), out var known))
            {
                return known;
            }

            var (names, source) = PtxKernels.NativePackModule(d, kind);
            var kernels = LoadDynamicModule(kind == PtxKernels.NativeKind.BF16 ? $"native_pack_{d.Code}" : $"native_pack_{d.Code}_{kind.Name}", names, source, static _ => 0, out string? reason);
            (IntPtr, IntPtr)? pack = kernels is null ? null : (kernels[names[0]], kernels[names[1]]);
            if (kernels is null)
            {
                TypedUnavailableReason = reason;
            }

            _nativePack[(d, kind)] = pack;
            return pack;
        }
    }

    /// <summary>Tests: the stored products' native bfloat16 lane for <paramref name="type"/> run directly (false when it cannot take the product).</summary>
    internal bool StoredNative(ElementType type, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k) =>
        DecodeOf(type) is { MatrixUnits: true } d && (scales is null || d.PerValue) && RunNative(d, x, weight, scales, y, m, n, k);

    // The native lane: x and W packed into the kind's rows of kp (the kind the weights' description matches: a float weight
    // of a kind's bits is read in place, as it is; any other widened into bfloat16), then the native kernel. False when the
    // device has no native kernel of the kind or the packing does not load.
    private bool RunNative(TypedDecode d, Storage x, Storage w, Storage? scales, Storage y, int m, int n, int k)
    {
        var kind = PtxKernels.NativeKind.For(d);
        if (NativeKernel(kind) is null || NativePack(d, kind) is not { } pack || k % 8 != 0)
        {
            return false;
        }

        int kp = PtxKernels.NativePadded(k);
        var shared = SharedPack(x, m, k, kind);
        var px = shared ?? Allocate(m * kp / 2, zeroed: false);
        bool asIs = kind.Holds(d) && kp == k && scales is null;
        var pw = asIs ? w : Allocate(n * kp / 2, zeroed: false);
        try
        {
            if (shared is null)
            {
                LaunchPack(pack.X, m, kp, P(x), P(px), U(m), U(k), U(kp));
            }

            if (!asIs)
            {
                LaunchPack(pack.W, n, kp, P(w), scales is null ? 0UL : P(scales), P(pw), U(n), U(k), U(kp));
            }

            var (shape, splits) = NativeChoice(kind, px, pw, y, m, n, kp);
            RunNativeShape(kind, shape, splits, px, pw, y, m, n, kp);
            return true;
        }
        finally
        {
            if (shared is null)
            {
                px.Release();
            }

            if (!asIs)
            {
                pw.Release();
            }
        }
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    // `part`: what of the shape's work the lane does, where it is not the product (a fused pass, a group's launches).
    private void SayStored(ElementType type, int m, int n, int k, string lane, string? part = null) =>
        ProductRoutes.Chose(Kind, string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"{type.Name} {m}x{k}->{n}{(BFloat16Products ? " mixed bfloat16" : "")}{(part is null ? "" : ", " + part)}"), lane);

    /// <summary>Tests: the stored products' tensor-core lane for <paramref name="type"/> run directly (false when it cannot take the product).</summary>
    internal bool StoredMatrixUnits(ElementType type, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k)
    {
        if (DecodeOf(type) is not { } d || !MatrixUnitLane(d, m, k))
        {
            return false;
        }

        RunTypedMma(d, x, weight, scales, y, m, n, k);
        return true;
    }

    // Stored weights (StoredWeight): under the measured route only lanes that never widen the weight (StoredRouted: the
    // described lane, the written kernels that read the type as stored), where one takes it; the generic product under the
    // generic route (the described lane for a type no generic kernel widens), and under the measured route for a type
    // neither reads, for a description these kernels widen with a scale per output row or none; the host fallback otherwise.
    public override void StoredMatMulKernel(Storage x, Storage weight, Storage? scales, Storage? zeros, Storage y, int m, int n, int k, StoredScales layout)
    {
        bool perRow = !layout.HasScales || layout is { BlockRows: 1, ScaleColumns: 1 } && layout.BlockColumns >= k;
        if (ProductRoutes.Current == ProductRoute.Measured && zeros is null && perRow && StoredRouted(x, weight, scales, y, m, n, k))
        {
            return;
        }

        if (ProductRoutes.Current != ProductRoute.Name && zeros is null && scales is null && (weight.ElementType.Super is not null || weight.ElementType.Grid is not null)
            && StoredSuper(x, weight, y, m, n, k))
        {
            return;
        }

        if (ProductRoutes.Current != ProductRoute.Name && zeros is null && perRow
            && (ProductRoutes.Current == ProductRoute.Measured ? StoredMeasured(x, weight, scales, y, m, n, k) : TypedOutputsMajor(weight.ElementType, x, weight, scales, y, m, n, k)))
        {
            return;
        }

        if (ProductRoutes.Current == ProductRoute.Generic && zeros is null && perRow && DescribedOf(weight.ElementType) is { } plan
            && RunDescribedRows(plan, x, weight, scales, y, m, n, k))
        {
            return;
        }

        base.StoredMatMulKernel(x, weight, scales, zeros, y, m, n, k, layout);
    }

    // The types a load keeps as stored for this device (BackendCapabilities.StoredFormats): under the generic and measured
    // routes those the generic kernels widen that carry no scales of their own (block formats, floats), and every type the
    // described lane reads where its kernels load (the device reaches their target); none under the named route (today's).
    // Read when asked, so types registered after the device started (ggml's) count.
    private sealed class DescribedStoredFormats(bool described) : IReadOnlySet<string>
    {
        public bool Contains(string item) => ProductRoutes.Current != ProductRoute.Name && ElementTypes.Find(item) is { } type
            && ((type.IsBlock || type.Float is not null) && DecodeOf(type) is not null || type.Super is { } super && PtxKernels.SuperFits(super)
                || type.Grid is { } grid && PtxKernels.GridFits(grid) || described && DescribedOf(type) is not null);

        private HashSet<string> Now() =>
            new(ElementTypes.All.Where(t => Contains(t.Name)).Select(t => t.Name), StringComparer.OrdinalIgnoreCase);

        public int Count => Now().Count;

        public IEnumerator<string> GetEnumerator() => Now().GetEnumerator();

        System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

        public bool IsProperSubsetOf(IEnumerable<string> other) => Now().IsProperSubsetOf(other);

        public bool IsProperSupersetOf(IEnumerable<string> other) => Now().IsProperSupersetOf(other);

        public bool IsSubsetOf(IEnumerable<string> other) => Now().IsSubsetOf(other);

        public bool IsSupersetOf(IEnumerable<string> other) => Now().IsSupersetOf(other);

        public bool Overlaps(IEnumerable<string> other) => Now().Overlaps(other);

        public bool SetEquals(IEnumerable<string> other) => Now().SetEquals(other);
    }

    // ------------------------------------------------------------------ the described lane (any type read from its description)

    // Each type's plan once (null: the described lane does not read it; DescribedReason says why).
    private static readonly ConcurrentDictionary<ElementType, DescribedPlan?> DescribedPlans = new();

    /// <summary>The described lane's plan of <paramref name="type"/> (its description compiled once), or null where it does not read the type.</summary>
    internal static DescribedPlan? DescribedOf(ElementType type) =>
        DescribedPlans.GetOrAdd(type, static t => t.Description is { } d ? DescribedPlan.Of(d, out _) : null);

    /// <summary>Why the described lane does not read <paramref name="type"/>, or null where it does (diagnostics, tests).</summary>
    internal static string? DescribedReason(ElementType type) =>
        type.Description is not { } d ? "the type has no description" : DescribedPlan.Of(d, out string? reason) is null ? reason : null;

    // A plan's kernels for rows of one alignment: the rows products (by DescribedRowCounts), the tensor-core product (zero
    // where the module has none), the widening of x (a wide plan's).
    private sealed class DescribedFunctions(Dictionary<string, IntPtr> kernels, DescribedPlan plan, int align, bool pairs)
    {
        public IntPtr[] Rows { get; } = [.. PtxKernels.DescribedRowCounts.ToArray().Select(r => kernels[PtxKernels.DescribedRowsName(plan, align, pairs, r)])];

        // The same over a group of 2 or 3 weights sharing x in one launch (few rows: DescribedRowCounts up to 8).
        public IntPtr[] RowsMany { get; } = [.. PtxKernels.DescribedRowCounts.ToArray().Where(r => r <= PtxKernels.TypedGemvRows[^1])
            .Select(r => kernels[PtxKernels.DescribedRowsName(plan, align, pairs, r) + "_many"])];

        // The fused few-row variants (DescribedRowCounts up to 8; zero where the module has none: its shared memory past the
        // device's): the gate/up pair's act(gate) · up, a product's residual + y and its RMS norm.
        public IntPtr[] RowsGated { get; } = [.. PtxKernels.DescribedRowCounts.ToArray().Where(r => r <= PtxKernels.TypedGemvRows[^1])
            .Select(r => kernels.GetValueOrDefault(PtxKernels.DescribedFusedName(plan, align, pairs, r, PtxKernels.Epilogue.Gated)))];

        public IntPtr[] RowsAddNorm { get; } = [.. PtxKernels.DescribedRowCounts.ToArray().Where(r => r <= PtxKernels.TypedGemvRows[^1])
            .Select(r => kernels.GetValueOrDefault(PtxKernels.DescribedFusedName(plan, align, pairs, r, PtxKernels.Epilogue.AddNorm)))];

        public IntPtr Mma { get; } = kernels.GetValueOrDefault(PtxKernels.DescribedMmaName(plan, align));

        public IntPtr Widen { get; } = kernels.GetValueOrDefault(PtxKernels.DescribedWidenName(plan, align, pairs));

        // A whole-byte integer code's integer product and x's rounding to int8 (zero where the module has none).
        public IntPtr Quant { get; } = kernels.GetValueOrDefault(PtxKernels.DescribedQuantName(plan));

        public IntPtr Int { get; } = kernels.GetValueOrDefault(PtxKernels.DescribedIntName(plan));

        // The integer product over narrow codes (zero where the module has none).
        public IntPtr Narrow { get; } = kernels.GetValueOrDefault(PtxKernels.DescribedNarrowName(plan, align));
    }

    private readonly ConcurrentDictionary<(DescribedPlan, int, bool), DescribedFunctions?> _described = new();

    // A wide plan's x widened to float64 (a call's, in queue order).
    private readonly KeptScratch _describedWide = new();

    private DescribedFunctions? Described(DescribedPlan plan, int align, bool pairs) =>
        _described.TryGetValue((plan, align, pairs), out var functions) ? functions : LoadDescribed(plan, align, pairs);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private DescribedFunctions? LoadDescribed(DescribedPlan plan, int align, bool pairs)
    {
        lock (_typedLock)
        {
            if (_described.TryGetValue((plan, align, pairs), out var known))
            {
                return known;
            }

            DescribedFunctions? functions = null;
            if (BelowTarget(PtxKernels.TensorCoreTarget) is { } below)
            {
                TypedUnavailableReason = below;
            }
            else
            {
                var (names, source) = PtxKernels.DescribedModule(plan, align, pairs, _limits.SharedPerBlockOptin);
                var kernels = LoadDynamicModule(PtxKernels.DescribedModuleName(plan, align, pairs), names, source, name => DescribedShared(plan, align, pairs, name), out string? reason);
                if (kernels is null)
                {
                    TypedUnavailableReason = reason;
                }
                else
                {
                    functions = new DescribedFunctions(kernels, plan, align, pairs);
                }
            }

            _described[(plan, align, pairs)] = functions;
            return functions;
        }
    }

    // A described kernel's dynamic shared memory, by its name.
    private static int DescribedShared(DescribedPlan plan, int align, bool pairs, string name)
    {
        if (name == PtxKernels.DescribedMmaName(plan, align))
        {
            return PtxKernels.DescribedMmaShared(plan, align);
        }

        if (name == PtxKernels.DescribedIntName(plan))
        {
            return PtxKernels.DescribedIntShared(plan);
        }

        if (name == PtxKernels.DescribedNarrowName(plan, align))
        {
            return PtxKernels.DescribedNarrowShared(plan, align);
        }

        foreach (int rows in PtxKernels.DescribedRowCounts)
        {
            string own = PtxKernels.DescribedRowsName(plan, align, pairs, rows);
            if (name == own || name == own + "_many")
            {
                return PtxKernels.DescribedRowsShared(plan, align, rows);
            }

            foreach (var epilogue in new[] { PtxKernels.Epilogue.Gated, PtxKernels.Epilogue.AddNorm })
            {
                if (name == PtxKernels.DescribedFusedName(plan, align, pairs, rows, epilogue))
                {
                    return PtxKernels.DescribedRowsShared(plan, align, rows, epilogue);
                }
            }
        }

        return 0;
    }

    // The bytes of a row of k values and the alignment its pattern blocks start at (16 at most): what a plan's kernels are
    // compiled for. False where a row passes 2 GB.
    private static bool DescribedRow(DescribedPlan plan, ElementType type, int k, out int rowBytes, out int align)
    {
        long bytes = type.ByteCount(k);
        rowBytes = (int)Math.Min(bytes, int.MaxValue);
        align = DescribedPlan.Gcd(16, DescribedPlan.Gcd(rowBytes, plan.PatternBytes));
        return bytes <= int.MaxValue;
    }

    // The described lane's rows product: y [m, n] = x [m, k] · Wᵀ (× a scale an output row), any row count: the variant of
    // the fewest rows that hold m, past 16 the grid's y over chunks of 16; a wide plan's x widened to float64 first. `split`:
    // log2 of the warps a group of 8 weight rows takes (0 - 3: 1, 2, 4 or 8 warps; a block 64 >> split rows).
    private bool RunDescribedRows(DescribedPlan plan, Storage x, Storage w, Storage? rowScales, Storage y, int m, int n, int k, int split = 3)
    {
        if (!DescribedRow(plan, w.ElementType, k, out int rowBytes, out int align) || (long)m * k > int.MaxValue / 2
            || Described(plan, align, k % 2 == 0) is not { } functions)
        {
            return false;
        }

        if (m == 0 || n == 0)
        {
            return true;
        }

        if (k == 0)
        {
            Fill(y, m * n, 0f);
            return true;
        }

        var counts = PtxKernels.DescribedRowCounts;
        int variant = counts.Length - 1;
        for (int i = 0; i < counts.Length; i++)
        {
            if (m <= counts[i])
            {
                variant = i;
                break;
            }
        }

        int rows = counts[variant];
        var input = x;
        if (plan.Wide)
        {
            input = _describedWide.Get(this, 2 * m * k);
            Launch(functions.Widen, (uint)((m * k + 255) / 256), 1, 256, 1, P(x), P(input), U(m * k));
        }

        int perBlock = 64 >> split;
        t_sharedBytes = (uint)PtxKernels.DescribedRowsShared(plan, align, rows);
        Launch(functions.Rows[variant], (uint)((n + perBlock - 1) / perBlock), (uint)((m + rows - 1) / rows),
            PtxKernels.DescribedRowsThreads, 1, P(input), P(w), rowScales is null ? 0UL : P(rowScales), P(y), U(m), U(n), U(k), U(rowBytes), U(split));
        return true;
    }

    // The described rows product over a group of 2 or 3 weights of one type sharing x (few rows; all with a scale an output
    // row or none) in one launch, each block one weight's 64 >> split rows: each output its own product's bits.
    private bool RunDescribedMany(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products, int split)
    {
        var type = products[0].Values.ElementType;
        if (products.Length is < 2 or > 3 || m is <= 0 or > 8 || k <= 0 || DescribedOf(type) is not { } plan || !DescribedRow(plan, type, k, out int rowBytes, out int align)
            || (long)m * k > int.MaxValue / 2 || Described(plan, align, k % 2 == 0) is not { } functions)
        {
            return false;
        }

        bool scaled = products[0].Scales is not null;
        long blocks = 0;
        int perBlock = 64 >> split;
        Span<ulong> args = stackalloc ulong[17];
        args.Clear();
        for (int i = 0; i < products.Length; i++)
        {
            var (values, scales, output, columns) = products[i];
            if (values.ElementType != type || scales is not null != scaled || columns <= 0)
            {
                return false;
            }

            (args[1 + 3 * i], args[2 + 3 * i], args[3 + 3 * i], args[13 + i]) = (P(values), scales is null ? 0UL : P(scales), P(output), U(columns));
            blocks += (columns + perBlock - 1) / perBlock;
        }

        args[16] = U(split);

        int variant = m <= 1 ? 0 : m <= 2 ? 1 : m <= 4 ? 2 : 3, rows = PtxKernels.DescribedRowCounts[variant];
        var input = x;
        if (plan.Wide)
        {
            input = _describedWide.Get(this, 2 * m * k);
            Launch(functions.Widen, (uint)((m * k + 255) / 256), 1, 256, 1, P(x), P(input), U(m * k));
        }

        (args[0], args[10], args[11], args[12]) = (P(input), U(m), U(k), U(rowBytes));
        t_sharedBytes = (uint)PtxKernels.DescribedRowsShared(plan, align, rows);
        Launch(functions.RowsMany[variant], (uint)blocks, 1, PtxKernels.DescribedRowsThreads, 1, args);
        return true;
    }

    // The described rows product's variant of the fewest rows that hold m (few rows: up to 8).
    private static int DescribedFewVariant(int m) => m <= 1 ? 0 : m <= 2 ? 1 : m <= 4 ? 2 : 3;

    // The described lane's fused gate/up pair (few rows): both weights' same rows in one launch, each lane decoding the
    // gate's and the up's values against one load of x, their sums as their own launches add them (split: log2 of the
    // warps a group of 8 rows takes), then hidden = act(gate) · up as gated_act_f32 computes it; neither product stored.
    // False where the plan, the shapes or the device do not take it.
    private bool RunDescribedGated(int activation, Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products,
        Storage hidden, int split)
    {
        if (products.Length != 2 || m is <= 0 or > 8 || k <= 0 || activation is < 0 or > 2)
        {
            return false;
        }

        var (gate, up) = (products[0], products[1]);
        var type = gate.Values.ElementType;
        if (up.Values.ElementType != type || gate.Columns != up.Columns || gate.Columns <= 0 || gate.Scales is null != up.Scales is null
            || DescribedOf(type) is not { } plan || !DescribedRow(plan, type, k, out int rowBytes, out int align) || (long)m * k > int.MaxValue / 2
            || Described(plan, align, k % 2 == 0) is not { } functions || functions.RowsGated[DescribedFewVariant(m)] == 0)
        {
            return false;
        }

        int n = gate.Columns, variant = DescribedFewVariant(m), rows = PtxKernels.DescribedRowCounts[variant], perBlock = 64 >> split;
        var kernel = functions.RowsGated[variant];
        var input = x;
        if (plan.Wide)
        {
            input = _describedWide.Get(this, 2 * m * k);
            Launch(functions.Widen, (uint)((m * k + 255) / 256), 1, 256, 1, P(x), P(input), U(m * k));
        }

        t_sharedBytes = (uint)PtxKernels.DescribedRowsShared(plan, align, rows, PtxKernels.Epilogue.Gated);
        Launch(kernel, (uint)((n + perBlock - 1) / perBlock), 1, PtxKernels.DescribedRowsThreads, 1, P(input), P(gate.Values), gate.Scales is null ? 0UL : P(gate.Scales),
            P(hidden), P(up.Values), up.Scales is null ? 0UL : P(up.Scales), U(m), U(n), U(k), U(rowBytes), U(split), U(activation));
        return true;
    }

    // Each block's sums of squares of the fused residual and norm (a call's, in queue order): [blocks, m].
    private readonly KeptScratch _describedNormParts = new();

    // The described lane's fused residual and norm (few rows): y = x · Wᵀ (× a scale an output row) by the rows product
    // (split as above), sum = residual + y in registers and stored, normalized = sum · 1 / sqrt(mean of the row's squares +
    // eps) · (gain + offset) by the last block to finish, as add_rms_norm_affine_f32 computes it; y not stored. False where
    // the plan, the shapes or the device do not take it.
    private bool RunDescribedAddNorm(Storage x, Storage weight, Storage? scales, int m, int n, int k, Storage residual, Storage sum, Storage gain, Storage normalized,
        float eps, float offset, int split)
    {
        var type = weight.ElementType;
        if (m is <= 0 or > 8 || n <= 0 || k <= 0 || DescribedOf(type) is not { } plan || !DescribedRow(plan, type, k, out int rowBytes, out int align)
            || (long)m * k > int.MaxValue / 2 || Described(plan, align, k % 2 == 0) is not { } functions || functions.RowsAddNorm[DescribedFewVariant(m)] == 0)
        {
            return false;
        }

        int variant = DescribedFewVariant(m), rows = PtxKernels.DescribedRowCounts[variant], perBlock = 64 >> split, blocks = (n + perBlock - 1) / perBlock;
        var kernel = functions.RowsAddNorm[variant];
        var input = x;
        if (plan.Wide)
        {
            input = _describedWide.Get(this, 2 * m * k);
            Launch(functions.Widen, (uint)((m * k + 255) / 256), 1, 256, 1, P(x), P(input), U(m * k));
        }

        var parts = _describedNormParts.Get(this, blocks * m);
        t_sharedBytes = (uint)PtxKernels.DescribedRowsShared(plan, align, rows, PtxKernels.Epilogue.AddNorm);
        Launch(kernel, (uint)blocks, 1, PtxKernels.DescribedRowsThreads, 1, P(input), P(weight), scales is null ? 0UL : P(scales), P(residual), P(sum), P(gain), P(normalized),
            P(parts), EpilogueCount(), U(m), U(n), U(k), U(rowBytes), U(split), F(eps), F(offset));
        return true;
    }

    // The k splits the described tensor-core product is measured with.
    private static readonly int[] DescribedSplits = [1, 2, 4, 8];

    // The values x's bfloat16 rows are padded to for the tensor-core product: whole spans of W and the native kind's 128.
    private static int DescribedPadded(DescribedPlan plan, int k)
    {
        int span = PtxKernels.DescribedMmaSpan(plan), unit = span / DescribedPlan.Gcd(span, 128) * 128;
        return (k + unit - 1) / unit * unit;
    }

    // Whether the described tensor-core product takes the product: a bfloat16 run, prompt-sized, a float32 plan, and the
    // kernels and x's packing load here.
    private bool DescribedMmaTakes(DescribedPlan plan, ElementType type, int m, int k) =>
        m > PtxKernels.TypedGemvRows[^1] && BFloat16Products && !plan.Wide && (long)m * DescribedPadded(plan, k) <= int.MaxValue
        && DescribedRow(plan, type, k, out _, out int align) && Described(plan, align, true) is { Mma: not 0 }
        && NativePack(DecodeOf(ElementType.Float32)!, PtxKernels.NativeKind.BF16) is not null;

    // The described tensor-core product, k split in up to `splits` parts (each whole spans of W; their sums added in order):
    // x packed to bfloat16 rows once (shared by a group's products, SharedPack), W read as stored.
    private bool RunDescribedMma(DescribedPlan plan, int splits, Storage x, Storage w, Storage? rowScales, Storage y, int m, int n, int k)
    {
        if (!DescribedMmaTakes(plan, w.ElementType, m, k) || !DescribedRow(plan, w.ElementType, k, out int rowBytes, out int align)
            || Described(plan, align, true) is not { } functions || NativePack(DecodeOf(ElementType.Float32)!, PtxKernels.NativeKind.BF16) is not { } pack)
        {
            return false;
        }

        if (n == 0)
        {
            return true;
        }

        int kp = DescribedPadded(plan, k), spans = kp / PtxKernels.DescribedMmaSpan(plan);
        int per = (spans + splits - 1) / splits, used = (spans + per - 1) / per;
        var shared = kp == PtxKernels.NativePadded(k) ? SharedPack(x, m, k, PtxKernels.NativeKind.BF16) : null;
        var px = shared ?? Allocate(m * kp / 2, zeroed: false);
        try
        {
            if (shared is null)
            {
                LaunchPack(pack.X, m, kp, P(x), P(px), U(m), U(k), U(kp));
            }

            var target = used == 1 ? y : _splitPartials.Get(this, used * m * n);
            t_sharedBytes = (uint)PtxKernels.DescribedMmaShared(plan, align);
            Launch(functions.Mma, (uint)((n + PtxKernels.DescribedMmaTile - 1) / PtxKernels.DescribedMmaTile), (uint)((m + PtxKernels.DescribedMmaTile - 1) / PtxKernels.DescribedMmaTile),
                (uint)used, PtxKernels.DescribedMmaThreads, 1, P(px), P(w), rowScales is null ? 0UL : P(rowScales), P(target), U(m), U(n), U(k), U(kp), U(rowBytes), U(per));
            if (used > 1)
            {
                var sum = Typed(DecodeOf(ElementType.Float32)!)!.Sum;
                Launch(sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(target), P(y), U(m * n), U(used));
            }

            return true;
        }
        finally
        {
            if (shared is null)
            {
                px.Release();
            }
        }
    }

    // x rounded to int8 for the integer product (a call's, in queue order): the codes, then the blocks' scales.
    private readonly KeptScratch _describedQuant = new();

    // Whether the described integer product takes the product: a whole-byte integer code (DescribedPlan.IntegerBytes), a
    // bfloat16 run's prompts (x rounded to int8 a block of 32 values, as the run rounds x for its tensor cores), rows of whole
    // 16-byte chunks, and the kernels load here.
    private bool DescribedIntTakes(DescribedPlan plan, ElementType type, int m, int k) =>
        plan.IntegerBytes > 0 && m > PtxKernels.TypedGemvRows[^1] && BFloat16Products && DescribedRow(plan, type, k, out _, out int align) && align == 16
        && (long)m * ((k + 31) / 32 * 32) <= int.MaxValue / 2 && Described(plan, 16, true) is { Int: not 0, Quant: not 0 };

    // The described integer product, k split in up to `splits` parts (each whole steps of 32; their sums added in order):
    // x rounded to int8 a block of 32 values once (DescribedQuant), then the codes' byte planes by int8 tensor cores.
    private bool RunDescribedInt(DescribedPlan plan, int splits, Storage x, Storage w, Storage? rowScales, Storage y, int m, int n, int k)
    {
        if (!DescribedIntTakes(plan, w.ElementType, m, k) || !DescribedRow(plan, w.ElementType, k, out int rowBytes, out _) || Described(plan, 16, true) is not { } functions)
        {
            return false;
        }

        if (n == 0)
        {
            return true;
        }

        int kp = (k + 31) / 32 * 32, steps = kp / 32, codes = m * kp / 4;
        int per = (steps + splits - 1) / splits, used = (steps + per - 1) / per;
        var quant = _describedQuant.Get(this, codes + m * steps);
        ulong q = P(quant), s = q + (ulong)codes * sizeof(float);
        Launch(functions.Quant, (uint)((m * steps + 255) / 256), 1, 256, 1, P(x), q, s, 0UL, U(m), U(k), U(kp));
        var target = used == 1 ? y : _splitPartials.Get(this, used * m * n);
        t_sharedBytes = (uint)PtxKernels.DescribedIntShared(plan);
        Launch(functions.Int, (uint)((n + PtxKernels.DescribedMmaTile - 1) / PtxKernels.DescribedMmaTile), (uint)((m + PtxKernels.DescribedMmaTile - 1) / PtxKernels.DescribedMmaTile),
            (uint)used, PtxKernels.DescribedMmaThreads, 1, q, s, rowScales is null ? 0UL : P(rowScales), P(w), P(target), U(m), U(n), U(k), U(kp), U(rowBytes), U(per));
        if (used > 1)
        {
            var sum = Typed(DecodeOf(ElementType.Float32)!)!.Sum;
            Launch(sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(target), P(y), U(m * n), U(used));
        }

        return true;
    }

    // Whether the integer product over narrow codes takes the product (DescribedPlan.NarrowChunk: codes of 8 bits or fewer
    // whose numbers are whole within int8 at a power of two): a bfloat16 run's prompts (x rounded to int8 a block of 32, as
    // the run rounds x for its tensor cores), x in pairs, and the kernels load here (the device has int8 tensor cores: the
    // module's target, and the kernel's shared memory within its limit).
    private bool DescribedNarrowTakes(DescribedPlan plan, ElementType type, int m, int k) =>
        plan.NarrowChunk > 0 && m > PtxKernels.TypedGemvRows[^1] && BFloat16Products && k % 2 == 0 && DescribedRow(plan, type, k, out _, out int align)
        && (long)m * DescribedPadded(plan, k) <= int.MaxValue / 2 && Described(plan, align, true) is { Narrow: not 0, Quant: not 0 };

    // The integer product over narrow codes, k split in up to `splits` parts (each whole spans of W; their sums added in
    // order): x rounded to int8 a block of 32 values once (DescribedQuant, with its codes' sums of 16 where the plan has
    // minimums), then the codes made int8 in registers by int8 tensor cores, the factors a chunk at a time.
    private bool RunDescribedNarrow(DescribedPlan plan, int splits, Storage x, Storage w, Storage? rowScales, Storage y, int m, int n, int k)
    {
        if (!DescribedNarrowTakes(plan, w.ElementType, m, k) || !DescribedRow(plan, w.ElementType, k, out int rowBytes, out int align)
            || Described(plan, align, true) is not { } functions)
        {
            return false;
        }

        if (n == 0)
        {
            return true;
        }

        bool mins = plan.Factors.Any(f => f.Min);
        int kp = DescribedPadded(plan, k), spans = kp / PtxKernels.DescribedMmaSpan(plan), blocks = kp / 32, codes = m * kp / 4;
        int per = (spans + splits - 1) / splits, used = (spans + per - 1) / per;
        var quant = _describedQuant.Get(this, codes + m * blocks + (mins ? m * kp / 16 : 0));
        ulong q = P(quant), s = q + (ulong)codes * sizeof(float), sums = mins ? s + (ulong)(m * blocks) * sizeof(float) : 0UL;
        Launch(functions.Quant, (uint)((m * blocks + 255) / 256), 1, 256, 1, P(x), q, s, sums, U(m), U(k), U(kp));
        var target = used == 1 ? y : _splitPartials.Get(this, used * m * n);
        t_sharedBytes = (uint)PtxKernels.DescribedNarrowShared(plan, align);
        Launch(functions.Narrow, (uint)((n + PtxKernels.DescribedMmaTile - 1) / PtxKernels.DescribedMmaTile), (uint)((m + PtxKernels.DescribedMmaTile - 1) / PtxKernels.DescribedMmaTile),
            (uint)used, PtxKernels.DescribedMmaThreads, 1, q, s, rowScales is null ? 0UL : P(rowScales), P(w), P(target), sums, U(m), U(n), U(k), U(kp), U(rowBytes), U(per));
        if (used > 1)
        {
            var sum = Typed(DecodeOf(ElementType.Float32)!)!.Sum;
            Launch(sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(target), P(y), U(m * n), U(used));
        }

        return true;
    }

    /// <summary>
    /// Tests: the described lane run directly: the rows product (<paramref name="mma"/> and <paramref name="integer"/> false;
    /// any row count), the tensor-core product (<paramref name="mma"/>), or the integer product (<paramref name="integer"/>:
    /// a whole-byte integer code's byte planes, x rounded to int8 a block of 32), k split in <paramref name="splits"/> (a
    /// bfloat16 run's prompts). False where it does not take the product (the type's description it does not read: DescribedReason).
    /// </summary>
    internal bool StoredDescribed(ElementType type, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, bool mma = false, int splits = 1,
        bool integer = false, int rowSplit = 3) =>
        DescribedOf(type) is { } plan && (integer ? plan.IntegerBytes > 0 ? RunDescribedInt(plan, splits, x, weight, scales, y, m, n, k) : RunDescribedNarrow(plan, splits, x, weight, scales, y, m, n, k)
            : mma ? RunDescribedMma(plan, splits, x, weight, scales, y, m, n, k) : RunDescribedRows(plan, x, weight, scales, y, m, n, k, rowSplit));

    // The measured route's lanes beyond the written ones (0 - 6 above): the written super-block or grid few-row kernel, the
    // described rows product, the described tensor-core product (k split in 1, 2, 4 or 8).
    // The rows product's lanes 30 + split (log2 of the warps a group of 8 weight rows takes: 0 - 3), the tensor-core
    // product's 34 + its k split's place in DescribedSplits, the integer product's 40 + it, the integer product over narrow
    // codes' 44 + it.
    private const int SuperFewLane = 20, DescribedRowsLane = 30, DescribedMmaLane = 34, DescribedIntLane = 40, DescribedNarrowLane = 44;

    private static readonly ConcurrentDictionary<string, int> RoutedIds = new(StringComparer.Ordinal);

    private readonly ConcurrentDictionary<(string, int, int, int, int, bool), byte> _routedSaid = new();

    /// <summary>
    /// The measured route's product of a stored weight (IDRAK_ROUTES=measured): only lanes that never widen the weight (no
    /// copy of its values in float32 or bfloat16, at load or a call, in memory or in shared memory): the described lane
    /// where it reads the type and ProductRoutes.Lanes takes it, the written kernels that read the type as stored where the
    /// setting takes them (the few-row kernels, decoding in registers; a 16-bit float's tensor-core kernels, its values as
    /// they are the operands); the fastest at the shape measured once on the real operands (Tune). With the written kernels
    /// alone and none for the shape, the described lane, never a widening one. False where no lane takes the product (a type
    /// the described lane does not read and no written kernel reads as stored at the shape: today's lanes then).
    /// </summary>
    private bool StoredRouted(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k)
    {
        if (RoutedLane(x, weight, scales, y, m, n, k) is not { } lane)
        {
            return false;
        }

        return m == 0 || n == 0 || RunRouted(lane, DescribedOf(weight.ElementType), x, weight, scales, y, m, n, k);
    }

    // The measured route's lane of a stored weight's product at the shape (StoredRouted), measured once on the real operands
    // (writing y, which the caller writes again), or null where no lane takes it. Recorded (ProductRoutes.Chose) once a shape.
    private int? RoutedLane(Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k)
    {
        var type = weight.ElementType;
        var lanes = ProductRoutes.Lanes;
        var plan = DescribedOf(type);
        bool few = m <= PtxKernels.TypedGemvRows[^1];
        Span<int> candidates = stackalloc int[24];
        int count = 0;
        // The first candidate is kept unless another is clearly faster (Tune's margin): the described lane, but for a prompt
        // of a type whose stored values are the tensor cores' operands as they are (a native kind holds it: bfloat16 in a
        // bfloat16 run), the written kernels that read them so.
        bool asOperands = !few && DecodeOf(type) is { } typed && PtxKernels.NativeKind.For(typed).Holds(typed);
        bool takesDescribed = plan is not null && lanes != RouteLanes.Written, takesWritten = !(plan is not null && lanes == RouteLanes.Described);
        if (takesWritten && asOperands)
        {
            count = AddWritten(type, scales, m, k, candidates);
        }

        bool describedFirst = takesDescribed;
        if (describedFirst)
        {
            count += AddDescribed(plan!, type, m, k, candidates[count..]);
        }

        if (takesWritten && !asOperands)
        {
            count += AddWritten(type, scales, m, k, candidates[count..]);
        }

        if (count == 0 && plan is not null && !describedFirst)
        {
            count = AddDescribed(plan, type, m, k, candidates);
        }

        if (count == 0)
        {
            return null;
        }

        int lane = candidates[0];
        if (count > 1 && m > 0 && n > 0)
        {
            var key = new TuneKey(TuneOp.ProductRoute, 6000 + RoutedIds.GetOrAdd(type.Name, static _ => RoutedIds.Count), few ? m : TuneSizes.Class(m), n, k, count, (int)lanes,
                BFloat16Products ? 1 : 0);
            if (!TryTuned(key, candidates[..count], candidates[0], out lane))
            {
                lane = MeasureRouted(key, candidates[..count].ToArray(), x, weight, scales, y, m, n, k, few);
            }
        }

        if (_routedSaid.TryAdd((type.Name, few ? m : TuneSizes.Class(m), n, k, (int)lanes, BFloat16Products), 0))
        {
            SayStored(type, m, n, k, $"{RoutedName(lane)}{(count > 1 ? $" (of {count} lanes)" : " (the only lane)")}");
        }

        return lane;
    }

    // Whether every product of a group takes the lane `lane` alone under the measured route (RoutedLane): then the written
    // kernels' one launch over the group gives each product's own bits.
    private bool EachTakes(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage? Scales, Storage Output, int Columns)> products, int lane)
    {
        foreach (var (values, scales, output, columns) in products)
        {
            if (RoutedLane(x, values, scales, output, m, columns, k) != lane)
            {
                return false;
            }
        }

        return true;
    }

    // The described lane's candidates: the rows product; for a bfloat16 run's prompts, the tensor-core product first, by its
    // k splits (each at least a span of W a split), and a whole-byte integer code's integer product by its k splits.
    private int AddDescribed(DescribedPlan plan, ElementType type, int m, int k, Span<int> into)
    {
        int count = 0;
        if (DescribedMmaTakes(plan, type, m, k))
        {
            int spans = DescribedPadded(plan, k) / PtxKernels.DescribedMmaSpan(plan);
            for (int s = 0; s < DescribedSplits.Length; s++)
            {
                if (DescribedSplits[s] == 1 || spans >= 2 * DescribedSplits[s])
                {
                    into[count++] = DescribedMmaLane + s;
                }
            }
        }

        if (DescribedIntTakes(plan, type, m, k))
        {
            int steps = (k + 31) / 32;
            for (int s = 0; s < DescribedSplits.Length; s++)
            {
                if (DescribedSplits[s] == 1 || steps >= 2 * DescribedSplits[s])
                {
                    into[count++] = DescribedIntLane + s;
                }
            }
        }

        if (DescribedNarrowTakes(plan, type, m, k))
        {
            int spans = DescribedPadded(plan, k) / PtxKernels.DescribedMmaSpan(plan);
            for (int s = 0; s < DescribedSplits.Length; s++)
            {
                if (DescribedSplits[s] == 1 || spans >= 2 * DescribedSplits[s])
                {
                    into[count++] = DescribedNarrowLane + s;
                }
            }
        }

        if (DescribedRow(plan, type, k, out _, out _))
        {
            for (int split = 3; split >= 0; split--)
            {
                into[count++] = DescribedRowsLane + split;                          // 8, 4, 2 or 1 warps a group of 8 rows
            }
        }

        return count;
    }

    // The written kernels' candidates that read the type as stored: few rows, the few-row kernels (a super-block or grid
    // description's; a typed description's); prompts, a 16-bit float's tensor-core kernels over its values as they are (the
    // matrix units, the native kind read in place, the native kind staging them as they are, the tensor rows as stored) and
    // the native kind over int8 codes read raw (widened in registers).
    private int AddWritten(ElementType type, Storage? scales, int m, int k, Span<int> into)
    {
        int count = 0;
        bool few = m <= PtxKernels.TypedGemvRows[^1];
        if (type.Super is not null || type.Grid is not null)
        {
            if (few && scales is null && Decoded(type) is var (_, values) && k % values == 0)
            {
                into[count++] = SuperFewLane;
            }

            return count;
        }

        if (DecodeOf(type) is not { } d)
        {
            return 0;
        }

        if (few)
        {
            if ((d.PerValue ? (long)k * d.Bits % 32 == 0 : k % PtxKernels.TypedTileDepth == 0 && scales is null) && Typed(d) is not null)
            {
                into[count++] = 0;
            }

            return count;
        }

        bool asStored = PtxKernels.NativeKind.BF16.Holds(d);
        if (asStored && MatrixUnitLane(d, m, k) && scales is null)
        {
            into[count++] = 1;
        }

        var kind = PtxKernels.NativeKind.For(d);
        if (kind.Holds(d) && MatrixUnitLane(d, m, k) && scales is null && k % 8 == 0 && PtxKernels.NativePadded(k) == k && NativeKernel(kind) is not null && NativePack(d, kind) is not null)
        {
            into[count++] = 2;
        }

        if (asStored && NativeDirectLane(type, scales, m, k))
        {
            into[count++] = 4;
        }

        if (MatrixUnitLane(d, m, k) && NativeRawLane(d, m, k))
        {
            into[count++] = 5;
        }

        if (TensorRowsSource(type, scales, m, k) is { Widen: 0, BlockBytes: 0, Int8: false })
        {
            into[count++] = 6;
        }

        return count;
    }

    // Timed cold for few rows (a decoding step reads each weight from memory), warm for prompts (as the lanes over stored
    // rows were measured before).
    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureRouted(TuneKey key, int[] candidates, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k, bool few)
    {
        var plan = DescribedOf(weight.ElementType);
        return Tune(key, candidates, candidates[0], c => RunRouted(c, plan, x, weight, scales, y, m, n, k), cold: few);
    }

    // One lane of the measured route's product (StoredRouted's candidates).
    private bool RunRouted(int lane, DescribedPlan? plan, Storage x, Storage weight, Storage? scales, Storage y, int m, int n, int k)
    {
        switch (lane)
        {
            case >= DescribedRowsLane and < DescribedRowsLane + 4:
                return RunDescribedRows(plan!, x, weight, scales, y, m, n, k, lane - DescribedRowsLane);
            case >= DescribedMmaLane and < DescribedMmaLane + 4:
                return RunDescribedMma(plan!, DescribedSplits[lane - DescribedMmaLane], x, weight, scales, y, m, n, k);
            case >= DescribedIntLane and < DescribedIntLane + 4:
                return RunDescribedInt(plan!, DescribedSplits[lane - DescribedIntLane], x, weight, scales, y, m, n, k);
            case >= DescribedNarrowLane and < DescribedNarrowLane + 4:
                return RunDescribedNarrow(plan!, DescribedSplits[lane - DescribedNarrowLane], x, weight, scales, y, m, n, k);
            case SuperFewLane:
                if (Decoded(weight.ElementType) is not var (f, _))
                {
                    return false;
                }

                Launch(f.Gemv[TypedRowVariant(m)], (uint)((n + 7) / 8), 1, 256, 1, P(x), P(weight), P(y), U(m), U(n), U(k));
                return true;
            default:
                return DecodeOf(weight.ElementType) is { } d && RunStoredLane(lane, d, x, weight, scales, y, m, n, k);
        }
    }

    private static string RoutedName(int lane) => lane switch
    {
        >= DescribedRowsLane and < DescribedRowsLane + 4 => string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described lane, rows product (values decoded in registers from the stored bytes; {1 << (lane - DescribedRowsLane)} warps a group of 8 rows)"),
        >= DescribedMmaLane and < DescribedMmaLane + 4 => string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described lane, tensor cores (values decoded in registers from the stored bytes){(lane == DescribedMmaLane ? "" : $", k split in {DescribedSplits[lane - DescribedMmaLane]}")}"),
        >= DescribedIntLane and < DescribedIntLane + 4 => string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described lane, int8 tensor cores (the codes' byte planes exact, x rounded to int8 a block of 32){(lane == DescribedIntLane ? "" : $", k split in {DescribedSplits[lane - DescribedIntLane]}")}"),
        >= DescribedNarrowLane and < DescribedNarrowLane + 4 => string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described lane, int8 tensor cores over narrow codes (made int8 in registers, x rounded to int8 a block of 32){(lane == DescribedNarrowLane ? "" : $", k split in {DescribedSplits[lane - DescribedNarrowLane]}")}"),
        SuperFewLane => "written few-row kernel (from the super-block description)",
        0 => "written few-row kernel",
        1 => "written matrix units (values as stored)",
        2 => "written native kind (values read in place)",
        4 => "written native kind (values staged as stored)",
        5 => "written native kind (int8 codes read raw)",
        6 => "written tensor rows (rows as stored)",
        _ => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"lane {lane}"),
    };

    // ------------------------------------------------------------------ the routes

    // Whether the format's own kernels run this product (fused ones included): the named route yes, the generic route no;
    // the measured route the faster at the shape, measured once on the real operands (Tune: the named lane unless the
    // generic product is clearly faster). Formats outside the experiment (int4) always take their own.
    private bool RoutedNamed(PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        if (format is not (PackedFormat.Int8 or PackedFormat.BFloat16))
        {
            return true;
        }

        switch (ProductRoutes.Current)
        {
            case ProductRoute.Name:
                return true;
            case ProductRoute.Generic:
                return false;
        }

        // Keyed by the mixed precision too: the format's own prompt-sized products run on tensor cores under bfloat16.
        var key = new TuneKey(TuneOp.ProductRoute, (int)format * 2 + (BFloat16Products ? 1 : 0), m <= PtxKernels.TypedGemvRows[^1] ? m : TuneSizes.Class(m), n, k);
        if (!TryTuned(key, [0, 1], 0, out int lane))
        {
            lane = MeasureRoute(key, format, x, packed, scales, y, m, n, k);
        }

        if (_routesSaid.TryAdd(key, 0))
        {
            SayRoute(format, m, n, k, lane);
        }

        return lane == 0;
    }

    // The shapes whose route this process has recorded (ProductRoutes.Chose), measured here or read from the tuning cache.
    private readonly ConcurrentDictionary<TuneKey, byte> _routesSaid = new();

    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureRoute(TuneKey key, PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k) =>
        Tune(key, [0, 1], 0, c => RunRoute(c, format, x, packed, scales, y, m, n, k), cold: m <= PtxKernels.TypedGemvRows[^1]);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private void SayRoute(PackedFormat format, int m, int n, int k, int lane) =>
        ProductRoutes.Chose(Kind, string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"{format.KernelName()} {m}x{k}->{n}{(BFloat16Products ? " mixed bfloat16" : "")}"), lane == 0 ? "named" : "generic");

    // One lane of a product, writing all of y: 0 the format's own kernels (as the named route runs them), 1 the generic product.
    private void RunRoute(int lane, PackedFormat format, Storage x, Storage packed, Storage? scales, Storage y, int m, int n, int k)
    {
        bool int8 = format == PackedFormat.Int8;
        if (lane == 1 && (int8
                ? TypedInputsMajor(ElementType.Int8, x, packed, scales, y, m, n, k, (n + 3) / 4 * 4)
                : TypedInputsMajor(ElementType.BFloat16, x, packed, null, y, m, n, k, (n + 1) / 2 * 2)))
        {
            return;
        }

        if (int8)
        {
            Int8Named(x, packed, scales ?? packed, y, m, n, k);
        }
        else
        {
            BFloat16Plain(x, packed, y, m, n, k);
        }
    }

    // Today's route of an int8 product for any row count, as the tensor's product runs it (Tensor.MatMulInt8): few rows by
    // the packed product where the device prefers it, else the direct kernel; more by the packed product, else the weights
    // expanded and the float32 product.
    private void Int8Named(Storage x, Storage q, Storage scales, Storage y, int m, int n, int k)
    {
        if (m <= 8)
        {
            if (!(PrefersPackedNamed(PackedFormat.Int8, x, q, scales, y, m, n, k) && PackedMatMulLargeNamed(PackedFormat.Int8, x, q, scales, y, m, n, k)))
            {
                Int8Plain(x, q, scales, y, m, n, k);
            }

            return;
        }

        if (PackedMatMulLargeNamed(PackedFormat.Int8, x, q, scales, y, m, n, k))
        {
            return;
        }

        var w = Allocate(k * n, zeroed: false);
        try
        {
            Int8Dequantize(q, scales, w, k, n);
            MatMul(x, w, y, m, n, k, false, false, 0f);
        }
        finally
        {
            w.Release();
        }
    }
}
