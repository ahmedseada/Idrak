// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace Idrak.Gpu.Cuda;

// The experiment's products of super-block weights on CUDA (ElementType.Super: ggml's k-quants, under the generic, measured
// and adad routes), every kernel generated from the description (PtxKernels.Super.cs): up to 8 rows the few-row kernel;
// more, the weights widened to float32 and the float32 product, or in a bfloat16 run widened to bfloat16 rows and the
// native kind's kernel (NativeMma), whichever measures faster at the shape (Tune: the float32 one unless the native kind is
// clearly faster), once; the generic route takes the float32 one.
internal sealed unsafe partial class CudaBackend
{
    private sealed class SuperFunctions(Dictionary<string, IntPtr> kernels, string[] names)
    {
        public IntPtr WidenF32 { get; } = kernels[names[0]];

        public IntPtr WidenBf16 { get; } = kernels[names[1]];

        public IntPtr[] Gemv { get; } = [.. names.Skip(2).Take(PtxKernels.TypedGemvRows.Length).Select(n => kernels[n])];

        // The few-row products grouped: 2 or 3 weights sharing x in one launch (StoredMatMulManyKernel).
        public IntPtr[] GemvMany { get; } = [.. names.Skip(2 + PtxKernels.TypedGemvRows.Length).Take(PtxKernels.TypedGemvRows.Length).Select(n => kernels[n])];

        public IntPtr[] GemvGated { get; } = [.. names.Skip(2 + 2 * PtxKernels.TypedGemvRows.Length).Take(PtxKernels.TypedGemvRows.Length).Select(n => kernels[n])];

        public IntPtr[] GemvAddNorm { get; } = [.. names.Skip(2 + 3 * PtxKernels.TypedGemvRows.Length).Select(n => kernels[n])];
    }

    private readonly ConcurrentDictionary<SuperBlock, SuperFunctions?> _super = new();

    // A number for each description, for the measured choices' keys.
    private static readonly ConcurrentDictionary<string, int> SuperIds = new(StringComparer.Ordinal);

    private readonly ConcurrentDictionary<(string, int, int, int, bool), byte> _superSaid = new();

    private SuperFunctions? Super(SuperBlock s) => _super.TryGetValue(s, out var functions) ? functions : LoadSuper(s);

    // The grid-codebook formats' kernels (PtxKernels.GridModule: the same kernels, the grid's decoder), loaded on first use.
    private readonly ConcurrentDictionary<GridBlock, SuperFunctions?> _grid = new();

    private SuperFunctions? Grid(GridBlock g) => _grid.TryGetValue(g, out var functions) ? functions : LoadGrid(g);

    [MethodImpl(MethodImplOptions.NoInlining)]
    private SuperFunctions? LoadGrid(GridBlock g)
    {
        lock (_typedLock)
        {
            if (_grid.TryGetValue(g, out var known))
            {
                return known;
            }

            SuperFunctions? functions = null;
            if (!PtxKernels.GridFits(g))
            {
                TypedUnavailableReason = "the grid description has a shape these kernels do not widen (PtxKernels.GridFits)";
            }
            else if (BelowTarget(PtxKernels.TensorCoreTarget) is { } below)
            {
                TypedUnavailableReason = below;
            }
            else
            {
                var names = PtxKernels.GridNames(g);
                var kernels = LoadDynamicModule($"grid_{PtxKernels.GridCode(g)}", names, PtxKernels.GridModule(g), static _ => 0, out string? reason);
                if (kernels is null)
                {
                    TypedUnavailableReason = reason;
                }
                else
                {
                    functions = new SuperFunctions(kernels, names);
                }
            }

            _grid[g] = functions;
            return functions;
        }
    }

    // A block format the generated kernels widen (a super-block's or a grid's description): its kernels and values a block.
    private (SuperFunctions Functions, int Values)? Decoded(ElementType type) =>
        type.Super is { } s ? Super(s) is { } f ? (f, s.Values) : null
        : type.Grid is { } g ? Grid(g) is { } gf ? (gf, g.Values) : null
        : null;

    [MethodImpl(MethodImplOptions.NoInlining)]
    private SuperFunctions? LoadSuper(SuperBlock s)
    {
        lock (_typedLock)
        {
            if (_super.TryGetValue(s, out var known))
            {
                return known;
            }

            SuperFunctions? functions = null;
            if (!PtxKernels.SuperFits(s))
            {
                TypedUnavailableReason = "the super-block description has a shape these kernels do not widen (PtxKernels.SuperFits)";
            }
            else if (BelowTarget(PtxKernels.TensorCoreTarget) is { } below)
            {
                TypedUnavailableReason = below;
            }
            else
            {
                var names = PtxKernels.SuperNames(s);
                var kernels = LoadDynamicModule($"super_{PtxKernels.SuperCode(s)}", names, PtxKernels.SuperModule(s), static _ => 0, out string? reason);
                if (kernels is null)
                {
                    TypedUnavailableReason = reason;
                }
                else
                {
                    functions = new SuperFunctions(kernels, names);
                }
            }

            _super[s] = functions;
            return functions;
        }
    }

    // y [m, n] = x [m, k] · Wᵀ for W of a super-block description stored as the file stores it (n rows of k values).
    private bool StoredSuper(Storage x, Storage w, Storage y, int m, int n, int k)
    {
        if (Decoded(w.ElementType) is not var (f, values) || k % values != 0)
        {
            return false;
        }

        if (m == 0 || n == 0)
        {
            return true;
        }

        bool few = m <= PtxKernels.TypedGemvRows[^1];
        int lane = 0;
        if (few)
        {
            Launch(f.Gemv[TypedRowVariant(m)], (uint)((n + 7) / 8), 1, 256, 1, P(x), P(w), P(y), U(m), U(n), U(k));
        }
        else
        {
            if (ProductRoutes.Current != ProductRoute.Generic && BFloat16Products && NativeKernel() is not null && NativePack(DecodeOf(ElementType.Float32)!) is not null)
            {
                bool rows = TensorRowsSource(w.ElementType, null, m, k) is not null;
                int[] lanes = NativeDirectLane(w.ElementType, null, m, k) ? rows ? [0, 1, 2, 3] : [0, 1, 2] : rows ? [0, 1, 3] : [0, 1];
                var key = new TuneKey(TuneOp.ProductRoute, 3000 + SuperIds.GetOrAdd(w.ElementType.Name, static _ => SuperIds.Count), TuneSizes.Class(m), n, k, lanes.Length);
                if (!TryTuned(key, lanes, 0, out lane))
                {
                    lane = MeasureSuper(key, lanes, f, x, w, y, m, n, k);
                }
            }

            RunSuperLane(lane, f, x, w, y, m, n, k);
        }

        if (_superSaid.TryAdd((w.ElementType.Name, few ? m : TuneSizes.Class(m), n, k, BFloat16Products), 0))
        {
            SayStored(w.ElementType, m, n, k, few ? "few-row lane (from the super-block description)" : lane switch
            {
                3 => "tensor rows (widened to bfloat16 rows from the description; k split in the kernel)",
                2 => "native bfloat16 kind (decoded in the kernel)",
                1 => "native bfloat16 kind (rows widened from the description)",
                _ => "widened to float32 from the description, the float32 product",
            });
        }

        return true;
    }

    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureSuper(TuneKey key, int[] lanes, SuperFunctions f, Storage x, Storage w, Storage y, int m, int n, int k) =>
        Tune(key, lanes, 0, c => RunSuperLane(c, f, x, w, y, m, n, k));

    // A prompt-sized lane: 1 the native kind, 2 the native kind decoding in the kernel, 3 the tensor rows (where they run),
    // else the weights widened to float32 and the float32 product.
    private void RunSuperLane(int lane, SuperFunctions f, Storage x, Storage w, Storage y, int m, int n, int k)
    {
        if (lane == 1 && SuperNative(f, x, w, y, m, n, k) || lane == 2 && RunNativeDirect(x, w, null, y, m, n, k) || lane == 3 && RunTensorRows(x, w, null, y, m, n, k))
        {
            return;
        }

        var wide = Allocate(n * k, zeroed: false);
        try
        {
            Launch(f.WidenF32, (uint)(((long)n * k / 8 + 255) / 256), 1, 256, 1, P(w), P(wide), U(n), U(k), U(k));
            MatMul(x, wide, y, m, n, k, false, true, 0f);
        }
        finally
        {
            wide.Release();
        }
    }

    // x packed into bfloat16 rows of kp, W widened into bfloat16 rows of kp (its padding zero), then the native kernel.
    private bool SuperNative(SuperFunctions f, Storage x, Storage w, Storage y, int m, int n, int k)
    {
        if (NativeKernel() is null || NativePack(DecodeOf(ElementType.Float32)!) is not { } pack)
        {
            return false;
        }

        int kp = PtxKernels.NativePadded(k);
        var shared = SharedPack(x, m, k, PtxKernels.NativeKind.BF16);
        var px = shared ?? Allocate(m * kp / 2, zeroed: false);
        var pw = Allocate(n * kp / 2, zeroed: false);
        try
        {
            if (shared is null)
            {
                LaunchPack(pack.X, m, kp, P(x), P(px), U(m), U(k), U(kp));
            }

            Launch(f.WidenBf16, (uint)(((long)n * kp / 8 + 255) / 256), 1, 256, 1, P(w), P(pw), U(n), U(k), U(kp));
            var (shape, splits) = NativeChoice(PtxKernels.NativeKind.BF16, px, pw, y, m, n, kp);
            RunNativeShape(PtxKernels.NativeKind.BF16, shape, splits, px, pw, y, m, n, kp);
            return true;
        }
        finally
        {
            if (shared is null)
            {
                px.Release();
            }

            pw.Release();
        }
    }

    /// <summary>Tests: a super-block weight's product by one lane (few rows: the few-row kernel; more: float32 or the native kind).</summary>
    internal bool StoredSuperLane(ElementType type, Storage x, Storage weight, Storage y, int m, int n, int k, bool native)
    {
        if (Decoded(type) is not var (f, values) || k % values != 0)
        {
            return false;
        }

        if (m <= PtxKernels.TypedGemvRows[^1])
        {
            Launch(f.Gemv[TypedRowVariant(m)], (uint)((n + 7) / 8), 1, 256, 1, P(x), P(weight), P(y), U(m), U(n), U(k));
            return true;
        }

        if (native)
        {
            return SuperNative(f, x, weight, y, m, n, k);
        }

        RunSuperLane(0, f, x, weight, y, m, n, k);
        return true;
    }
}
