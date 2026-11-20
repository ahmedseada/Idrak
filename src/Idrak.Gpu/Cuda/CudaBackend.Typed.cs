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
    private void SayStored(ElementType type, int m, int n, int k, string lane) =>
        ProductRoutes.Chose(Kind, string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"{type.Name} {m}x{k}->{n}{(BFloat16Products ? " mixed bfloat16" : "")}"), lane);

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

    // Stored weights (StoredWeight): the generic product under the generic and measured routes, for a description these
    // kernels widen with a scale per output row or none; the host fallback otherwise, as today.
    public override void StoredMatMulKernel(Storage x, Storage weight, Storage? scales, Storage? zeros, Storage y, int m, int n, int k, StoredScales layout)
    {
        bool perRow = !layout.HasScales || layout is { BlockRows: 1, ScaleColumns: 1 } && layout.BlockColumns >= k;
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

        base.StoredMatMulKernel(x, weight, scales, zeros, y, m, n, k, layout);
    }

    // The types a load keeps as stored for this device (BackendCapabilities.StoredFormats): under the generic and measured
    // routes those the generic kernels widen that carry no scales of their own (block formats, floats); none under the
    // named route (today's). Read when asked, so types registered after the device started (ggml's) count.
    private sealed class DescribedStoredFormats : IReadOnlySet<string>
    {
        public static readonly DescribedStoredFormats Instance = new();

        public bool Contains(string item) => ProductRoutes.Current != ProductRoute.Name && ElementTypes.Find(item) is { } type
            && ((type.IsBlock || type.Float is not null) && DecodeOf(type) is not null || type.Super is { } super && PtxKernels.SuperFits(super)
                || type.Grid is { } grid && PtxKernels.GridFits(grid));

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
