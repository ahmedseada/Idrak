// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Runtime.CompilerServices;

namespace Idrak.Gpu.Cuda;

// The products of stored weights on CUDA: the described lane (PtxKernels.Typed.cs), which reads any type from its
// description alone (TypeDescription), its variants measured once per shape (Tune).
internal sealed unsafe partial class CudaBackend
{
    // The modules' loading lock, and why the last one did not load (diagnostics).
    private readonly Lock _typedLock = new();

    /// <summary>Why the last described module did not load, or null (diagnostics).</summary>
    internal string? TypedUnavailableReason { get; private set; }

    // Stored weights: the described lane's product.
    public override void StoredMatMulKernel(Storage x, Storage weight, Storage y, int m, int n, int k)
    {
        if (StoredRouted(x, weight, y, m, n, k))
        {
            return;
        }

        base.StoredMatMulKernel(x, weight, y, m, n, k);
    }

    // The kernel adding a product's k splits in split order (PtxKernels.SplitSum), loaded on first use.
    private IntPtr _splitSum;

    private IntPtr SplitSum()
    {
        if (_splitSum != IntPtr.Zero)
        {
            return _splitSum;
        }

        lock (_typedLock)
        {
            if (_splitSum == IntPtr.Zero)
            {
                var (name, source) = PtxKernels.SplitSum();
                _splitSum = LoadDynamicModule(name, [name], source, static _ => 0, out string? reason) is { } kernels ? kernels[name]
                    : throw new CudaException($"The described lane's split sum did not load: {reason}");
            }

            return _splitSum;
        }
    }

    // ------------------------------------------------------------------ the described lane (any type read from its description)

    // Each description's plan once (null: the described lane does not read it; DescribedReason says why).
    private static readonly ConcurrentDictionary<TypeDescription, DescribedPlan?> DescribedPlans = new();

    /// <summary>The described lane's plan of <paramref name="description"/> (compiled once), or null where it does not read it.</summary>
    internal static DescribedPlan? DescribedOf(TypeDescription? description) =>
        description is null ? null : DescribedPlans.GetOrAdd(description, static d => DescribedPlan.Of(d, out _));

    /// <summary>Why the described lane does not read <paramref name="description"/>, or null where it does (diagnostics, tests).</summary>
    internal static string? DescribedReason(TypeDescription? description) =>
        description is null ? "the weight has no description" : DescribedPlan.Of(description, out string? reason) is null ? reason : null;

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
    private static bool DescribedRow(DescribedPlan plan, TypeDescription description, int k, out int rowBytes, out int align)
    {
        long bytes = k % description.UnitValues == 0 ? (long)k / description.UnitValues * description.UnitBytes : long.MaxValue;
        rowBytes = (int)Math.Min(bytes, int.MaxValue);
        align = DescribedPlan.Gcd(16, DescribedPlan.Gcd(rowBytes, plan.PatternBytes));
        return bytes <= int.MaxValue;
    }

    // The described lane's rows product: y [m, n] = x [m, k] · Wᵀ, any row count: the variant of
    // the fewest rows that hold m, past 16 the grid's y over chunks of 16; a wide plan's x widened to float64 first. `split`:
    // log2 of the warps a group of 8 weight rows takes (0 - 3: 1, 2, 4 or 8 warps; a block 64 >> split rows).
    private bool RunDescribedRows(DescribedPlan plan, Storage x, Storage w, Storage y, int m, int n, int k, int split = 3)
    {
        if (!DescribedRow(plan, w.Description!, k, out int rowBytes, out int align) || (long)m * k > int.MaxValue / 2
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
            PtxKernels.DescribedRowsThreads, 1, P(input), P(w), P(y), U(m), U(n), U(k), U(rowBytes), U(split));
        return true;
    }

    // The described rows product over a group of 2 or 3 weights of one type sharing x (few rows) in one launch, each block
    // one weight's 64 >> split rows: each output its own product's bits.
    private bool RunDescribedMany(Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage Output, int Columns)> products, int split)
    {
        var description = products[0].Values.Description!;
        if (products.Length is < 2 or > 3 || m is <= 0 or > 8 || k <= 0 || DescribedOf(description) is not { } plan || !DescribedRow(plan, description, k, out int rowBytes, out int align)
            || (long)m * k > int.MaxValue / 2 || Described(plan, align, k % 2 == 0) is not { } functions)
        {
            return false;
        }

        long blocks = 0;
        int perBlock = 64 >> split;
        Span<ulong> args = stackalloc ulong[14];
        args.Clear();
        for (int i = 0; i < products.Length; i++)
        {
            var (values, output, columns) = products[i];
            if (!Equals(values.Description, description) || columns <= 0)
            {
                return false;
            }

            (args[1 + 2 * i], args[2 + 2 * i], args[10 + i]) = (P(values), P(output), U(columns));
            blocks += (columns + perBlock - 1) / perBlock;
        }

        args[13] = U(split);

        int variant = m <= 1 ? 0 : m <= 2 ? 1 : m <= 4 ? 2 : 3, rows = PtxKernels.DescribedRowCounts[variant];
        var input = x;
        if (plan.Wide)
        {
            input = _describedWide.Get(this, 2 * m * k);
            Launch(functions.Widen, (uint)((m * k + 255) / 256), 1, 256, 1, P(x), P(input), U(m * k));
        }

        (args[0], args[7], args[8], args[9]) = (P(input), U(m), U(k), U(rowBytes));
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
    private bool RunDescribedGated(int activation, Storage x, int m, int k, ReadOnlySpan<(Storage Values, Storage Output, int Columns)> products,
        Storage hidden, int split)
    {
        if (products.Length != 2 || m is <= 0 or > 8 || k <= 0 || activation is < 0 or > 2)
        {
            return false;
        }

        var (gate, up) = (products[0], products[1]);
        var description = gate.Values.Description;
        if (!Equals(up.Values.Description, description) || gate.Columns != up.Columns || gate.Columns <= 0
            || DescribedOf(description) is not { } plan || !DescribedRow(plan, description!, k, out int rowBytes, out int align) || (long)m * k > int.MaxValue / 2
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
        Launch(kernel, (uint)((n + perBlock - 1) / perBlock), 1, PtxKernels.DescribedRowsThreads, 1, P(input), P(gate.Values),
            P(hidden), P(up.Values), U(m), U(n), U(k), U(rowBytes), U(split), U(activation));
        return true;
    }

    // Each block's sums of squares of the fused residual and norm (a call's, in queue order): [blocks, m].
    private readonly KeptScratch _describedNormParts = new();

    // The described lane's fused residual and norm (few rows): y = x · Wᵀ by the rows product
    // (split as above), sum = residual + y in registers and stored, normalized = sum · 1 / sqrt(mean of the row's squares +
    // eps) · (gain + offset) by the last block to finish, as add_rms_norm_affine_f32 computes it; y not stored. False where
    // the plan, the shapes or the device do not take it.
    private bool RunDescribedAddNorm(Storage x, Storage weight, int m, int n, int k, Storage residual, Storage sum, Storage gain, Storage normalized,
        float eps, float offset, int split)
    {
        var description = weight.Description!;
        if (m is <= 0 or > 8 || n <= 0 || k <= 0 || DescribedOf(description) is not { } plan || !DescribedRow(plan, description, k, out int rowBytes, out int align)
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
        Launch(kernel, (uint)blocks, 1, PtxKernels.DescribedRowsThreads, 1, P(input), P(weight), P(residual), P(sum), P(gain), P(normalized),
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

    // x rounded to int8 for the integer product (a call's, in queue order): the codes, then the blocks' scales.
    private readonly KeptScratch _describedQuant = new();

    // Whether the described integer product takes the product: a whole-byte integer code (DescribedPlan.IntegerBytes),
    // prompts (x rounded to int8 a block of 32 values), rows of whole 16-byte chunks, and the kernels load here.
    private bool DescribedIntTakes(DescribedPlan plan, TypeDescription description, int m, int k) =>
        plan.IntegerBytes > 0 && m > PtxKernels.TypedGemvRows[^1] && DescribedRow(plan, description, k, out _, out int align) && align == 16
        && (long)m * ((k + 31) / 32 * 32) <= int.MaxValue / 2 && Described(plan, 16, true) is { Int: not 0, Quant: not 0 };

    // The described integer product, k split in up to `splits` parts (each whole steps of 32; their sums added in order):
    // x rounded to int8 a block of 32 values once (DescribedQuant), then the codes' byte planes by int8 tensor cores.
    private bool RunDescribedInt(DescribedPlan plan, int splits, Storage x, Storage w, Storage y, int m, int n, int k)
    {
        if (!DescribedIntTakes(plan, w.Description!, m, k) || !DescribedRow(plan, w.Description!, k, out int rowBytes, out _) || Described(plan, 16, true) is not { } functions)
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
            (uint)used, PtxKernels.DescribedMmaThreads, 1, q, s, P(w), P(target), U(m), U(n), U(k), U(kp), U(rowBytes), U(per));
        if (used > 1)
        {
            var sum = SplitSum();
            Launch(sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(target), P(y), U(m * n), U(used));
        }

        return true;
    }

    // Whether the integer product over narrow codes takes the product (DescribedPlan.NarrowChunk: codes of 8 bits or fewer
    // whose numbers are whole within int8 at a power of two): prompts (x rounded to int8 a block of 32), x in pairs, and the
    // kernels load here (the device has int8 tensor cores: the
    // module's target, and the kernel's shared memory within its limit).
    private bool DescribedNarrowTakes(DescribedPlan plan, TypeDescription description, int m, int k) =>
        plan.NarrowChunk > 0 && m > PtxKernels.TypedGemvRows[^1] && k % 2 == 0 && DescribedRow(plan, description, k, out _, out int align)
        && (long)m * DescribedPadded(plan, k) <= int.MaxValue / 2 && Described(plan, align, true) is { Narrow: not 0, Quant: not 0 };

    // The integer product over narrow codes, k split in up to `splits` parts (each whole spans of W; their sums added in
    // order): x rounded to int8 a block of 32 values once (DescribedQuant, with its codes' sums of 16 where the plan has
    // minimums), then the codes made int8 in registers by int8 tensor cores, the factors a chunk at a time.
    private bool RunDescribedNarrow(DescribedPlan plan, int splits, Storage x, Storage w, Storage y, int m, int n, int k)
    {
        if (!DescribedNarrowTakes(plan, w.Description!, m, k) || !DescribedRow(plan, w.Description!, k, out int rowBytes, out int align)
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
            (uint)used, PtxKernels.DescribedMmaThreads, 1, q, s, P(w), P(target), sums, U(m), U(n), U(k), U(kp), U(rowBytes), U(per));
        if (used > 1)
        {
            var sum = SplitSum();
            Launch(sum, (uint)((m * n + 255) / 256), 1, 256, 1, P(target), P(y), U(m * n), U(used));
        }

        return true;
    }

    // The described lane's variants: the rows product's lanes 30 + split (log2 of the warps a group of 8 weight rows takes:
    // 0 - 3), the integer product's 40 + its k split's place in DescribedSplits, the integer product over narrow codes'
    // 44 + it.
    private const int DescribedRowsLane = 30, DescribedIntLane = 40, DescribedNarrowLane = 44;

    // A number for each description, for the measured choices' keys (TuneKey holds numbers only).
    private static readonly ConcurrentDictionary<TypeDescription, int> RoutedIds = new();

    private readonly ConcurrentDictionary<(TypeDescription, int, int, int), byte> _routedSaid = new();

    /// <summary>
    /// The product of a stored weight by the described lane: its variants that take the shape, the fastest measured once
    /// on the real operands (Tune). False where the described lane does not read the weight's description.
    /// </summary>
    private bool StoredRouted(Storage x, Storage weight, Storage y, int m, int n, int k)
    {
        if (RoutedLane(x, weight, y, m, n, k) is not { } lane)
        {
            return false;
        }

        return m == 0 || n == 0 || RunRouted(lane, DescribedOf(weight.Description), x, weight, y, m, n, k);
    }

    // The described lane's variant of a stored weight's product at the shape (StoredRouted), measured once on the real
    // operands (writing y, which the caller writes again), or null where none takes it. Recorded (ProductRoutes.Chose) once
    // a shape.
    private int? RoutedLane(Storage x, Storage weight, Storage y, int m, int n, int k)
    {
        if (weight.Description is not { } description || DescribedOf(description) is not { } plan)
        {
            return null;
        }

        bool few = m <= PtxKernels.TypedGemvRows[^1];
        Span<int> candidates = stackalloc int[24];
        int count = AddDescribed(plan, description, m, k, candidates);
        if (count == 0)
        {
            return null;
        }

        int lane = candidates[0];
        if (count > 1 && m > 0 && n > 0)
        {
            var key = new TuneKey(TuneOp.ProductRoute, 6000 + RoutedIds.GetOrAdd(description, static _ => RoutedIds.Count), few ? m : TuneSizes.Class(m), n, k, count, 0, 0);
            if (!TryTuned(key, candidates[..count], candidates[0], out lane))
            {
                lane = MeasureRouted(key, candidates[..count].ToArray(), x, weight, y, m, n, k, few);
            }
        }

        if (_routedSaid.TryAdd((description, few ? m : TuneSizes.Class(m), n, k), 0))
        {
            SayStored(description, m, n, k, $"{RoutedName(lane)}{(count > 1 ? $" (of {count} lanes)" : " (the only lane)")}");
        }

        return lane;
    }

    // The described lane's candidates: for prompts a whole-byte integer code's integer product by its k
    // splits and the integer product over narrow codes by its k splits, then the rows product.
    private int AddDescribed(DescribedPlan plan, TypeDescription description, int m, int k, Span<int> into)
    {
        int count = 0;
        if (DescribedIntTakes(plan, description, m, k))
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

        if (DescribedNarrowTakes(plan, description, m, k))
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

        if (DescribedRow(plan, description, k, out _, out _))
        {
            for (int split = 3; split >= 0; split--)
            {
                into[count++] = DescribedRowsLane + split;                          // 8, 4, 2 or 1 warps a group of 8 rows
            }
        }

        return count;
    }

    // Timed cold for few rows (a decoding step reads each weight from memory), warm for prompts.
    [MethodImpl(MethodImplOptions.NoInlining)]
    private int MeasureRouted(TuneKey key, int[] candidates, Storage x, Storage weight, Storage y, int m, int n, int k, bool few)
    {
        var plan = DescribedOf(weight.Description);
        return Tune(key, candidates, candidates[0], c => RunRouted(c, plan, x, weight, y, m, n, k), cold: few);
    }

    // One variant of the described lane's product (StoredRouted's candidates).
    private bool RunRouted(int lane, DescribedPlan? plan, Storage x, Storage weight, Storage y, int m, int n, int k) => lane switch
    {
        >= DescribedRowsLane and < DescribedRowsLane + 4 => RunDescribedRows(plan!, x, weight, y, m, n, k, lane - DescribedRowsLane),
        >= DescribedIntLane and < DescribedIntLane + 4 => RunDescribedInt(plan!, DescribedSplits[lane - DescribedIntLane], x, weight, y, m, n, k),
        >= DescribedNarrowLane and < DescribedNarrowLane + 4 => RunDescribedNarrow(plan!, DescribedSplits[lane - DescribedNarrowLane], x, weight, y, m, n, k),
        _ => false,
    };

    private static string RoutedName(int lane) => lane switch
    {
        >= DescribedRowsLane and < DescribedRowsLane + 4 => string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described lane, rows product (values decoded in registers from the stored bytes; {1 << (lane - DescribedRowsLane)} warps a group of 8 rows)"),
        >= DescribedIntLane and < DescribedIntLane + 4 => string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described lane, int8 tensor cores (the codes' byte planes exact, x rounded to int8 a block of 32){(lane == DescribedIntLane ? "" : $", k split in {DescribedSplits[lane - DescribedIntLane]}")}"),
        >= DescribedNarrowLane and < DescribedNarrowLane + 4 => string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described lane, int8 tensor cores over narrow codes (made int8 in registers, x rounded to int8 a block of 32){(lane == DescribedNarrowLane ? "" : $", k split in {DescribedSplits[lane - DescribedNarrowLane]}")}"),
        _ => string.Create(System.Globalization.CultureInfo.InvariantCulture, $"lane {lane}"),
    };

    // `part`: what of the shape's work the lane does, where it is not the product (a fused pass, a group's launches).
    private void SayStored(TypeDescription description, int m, int n, int k, string lane, string? part = null) =>
        ProductRoutes.Chose(Kind, string.Create(System.Globalization.CultureInfo.InvariantCulture,
            $"described ({description.Values} values in {description.Bytes} bytes) {m}x{k}->{n}{(part is null ? "" : ", " + part)}"), lane);
}
