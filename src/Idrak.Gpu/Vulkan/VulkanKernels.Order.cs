// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// One summation order per operation, whatever the tuned choice (docs/optimization.md rule 86): the choices the backend
// measures per device (VulkanBackend.KernelTuning.cs: how many splits, which width, which word variant, which kernel)
// decide only how fast a result comes, never its bits, so two processes that measured different choices (two cluster
// ranks, two drivers of one card) agree. The orders, fixed by the shapes alone:
//
//   - a packed product's dot products over k (VulkanTuneOp.Gemv, FusedGemv, and the fused epilogues' second passes):
//     k in leaves of GemvChunkAlign (32) rows, each leaf's terms added in turn from +0 (one fused multiply-add each),
//     the leaves combined in the pairwise tree of consecutive leaves (2i with 2i + 1, then aligned pairs of those, the
//     last incomplete subtrees as if padded with -0). A split of k takes an aligned power of two of the leaves, a slice
//     of its workgroup an aligned power of two of the split's, so each partial sum is a subtree's and every pass after
//     (the slices in workgroup memory, gemv_reduce over the splits) finishes the same tree (PairwiseStack, Pairwise).
//   - the prompt-sized products (PackedPrompt) and the float32 products (MatMul): every term added in turn along k
//     from +0, one fused multiply-add each, as the tiled and register-blocked kernels do; the few-rows kernels take that
//     order too (one leaf of all of k, their slices on rows of x), int8's column scales multiply the finished sums
//     (expanding the weights keeps them apart until the product is done).
//   - a two-pass reduction (Splits): rows (columns) or elements (groups) in leaves of SplitLeaf, each leaf's sums in
//     turn (moments: its mean, then the squares about it), the leaves in the pairwise tree, moments merged as Chan et
//     al. with the left operand first; a split takes an aligned power of two of the leaves, the workgroup's invocations
//     aligned parts of a split's (KernelBuilder.ReducePairwise), group_finish the splits.
//   - decoding attention (Attention): the cached positions a row sees in blocks of the head size rounded up to a power
//     of two; a block's max, weights exp(score - max), their total and weighted values each in turn; the blocks merged
//     in the tree of their indices' bits from the lowest (a split takes the blocks of one residue, a part of the
//     workgroup the next bits, so splits and parts stay balanced), the left operand first (AttentionMerge).
//
// Choices whose order the hardware defines are the exception: the cooperative-matrix products (VulkanTuneOp.MatMul's
// and PackedPrompt's cooperative-matrix candidates, MixedMatMul, MixedPackedPrompt) accumulate as the device's matrix
// units do. The attention paths (AttentionPath, TiledAttention, SpanAttention) are different algorithms, not orders of
// one; cluster ranks agree on them by sharing the measured choices.
internal static partial class VulkanKernels
{
    /// <summary>Rows (columns) or elements (groups) per leaf of the two-pass reductions' tree.</summary>
    public const int SplitLeaf = 16;

    // Entries of a pairwise stack over the splits of a dispatch (at most 2^17: more than any workgroup count along y).
    private const int SplitStackEntries = 18;

    // A pairwise tree over leaves taken in turn (0, 1, 2, … since Reset): a stack of the sums of complete aligned
    // subtrees (leaf i merges the entries its trailing one bits complete), folded from the top at the end, so the result
    // is the tree of consecutive leaves padded with empty ones to any power of two. The top entry stays in registers and
    // the rest in function memory (`entries` deep, the top included: a walk of up to 2^entries leaves, whose deepest point
    // holds one entry per level but the last), so a walk of one or two leaves
    // touches no memory and longer ones write an entry every other leaf; each entry holds `width` values.
    private sealed class PairwiseStack
    {
        private readonly KernelBuilder _k;
        private readonly int _width;
        private readonly SharedArray? _stack;
        private readonly Var _depth;
        private readonly Var[] _top;
        private readonly Func<Val[], Val[], Val[]> _merge;

        public PairwiseStack(KernelBuilder k, int width, int entries, Func<Val[], Val[], Val[]> merge)
        {
            (_k, _width, _merge) = (k, width, merge);
            _stack = entries > 1 ? k.LocalArray((entries - 1) * width) : null;
            _depth = k.Local(0);
            _top = [.. Enumerable.Range(0, width).Select(_ => k.Local(0f))];
        }

        public void Reset() => _depth.V = _k.Int(0);

        // Entry `at` below the top (in function memory).
        private Val[] Entry(Val at) => [.. Enumerable.Range(0, _width).Select(i => _stack![at * _width + i])];

        private void SetTop(Val[] values)
        {
            for (int i = 0; i < _width; i++)
            {
                _top[i].V = values[i];
            }
        }

        // Adds leaf `index`, the next in turn.
        public void Push(Val index, Val[] leaf)
        {
            var k = _k;
            var carry = leaf.Select(v => k.Local(v)).ToArray();
            k.If((index & 1).Eq(1), () =>
            {
                // The top is the left half of this leaf's pair; then the earlier subtrees this one completes.
                var merged = _merge([.. _top.Select(t => t.V)], [.. carry.Select(c => c.V)]);
                for (int i = 0; i < _width; i++)
                {
                    carry[i].V = merged[i];
                }

                _depth.V = _depth.V - 1;
                if (_stack is not null)
                {
                    var bits = k.Local(index >> 1);
                    k.While(() => (bits.V & 1).Eq(1), () =>
                    {
                        _depth.V = _depth.V - 1;
                        var deeper = _merge(Entry(_depth.V), [.. carry.Select(c => c.V)]);   // the earlier subtree on the left
                        for (int i = 0; i < _width; i++)
                        {
                            carry[i].V = deeper[i];
                        }

                        bits.V = bits.V >> 1;
                    });
                }
            }, () =>
            {
                if (_stack is not null)
                {
                    k.If(_depth.V > 0, () =>
                    {
                        for (int i = 0; i < _width; i++)
                        {
                            _stack[(_depth.V - 1) * _width + i] = _top[i].V;
                        }
                    });
                }
            });
            SetTop([.. carry.Select(c => c.V)]);
            _depth.V = _depth.V + 1;
        }

        // The tree's result (`empty` when no leaf was added).
        public Val[] Total(Val[] empty)
        {
            var k = _k;
            var total = empty.Select(v => k.Local(v)).ToArray();
            k.If(_depth.V > 0, () =>
            {
                for (int i = 0; i < _width; i++)
                {
                    total[i].V = _top[i].V;
                }

                if (_stack is not null)
                {
                    _depth.V = _depth.V - 1;
                    k.While(() => _depth.V > 0, () =>
                    {
                        _depth.V = _depth.V - 1;
                        var merged = _merge(Entry(_depth.V), [.. total.Select(t => t.V)]);
                        for (int i = 0; i < _width; i++)
                        {
                            total[i].V = merged[i];
                        }
                    });
                }
            });
            return [.. total.Select(t => t.V)];
        }
    }

    // Sums merged in a tree: each value's own sum, rounded on its own.
    private static Func<Val[], Val[], Val[]> SumsMerge(KernelBuilder k) => (left, right) => [.. left.Zip(right, k.AddRounded)];

    // Σ term(i) for i < count in the pairwise tree of consecutive terms (-0 for none), through `stack` (up to 2^entries
    // terms).
    private static Val Pairwise(KernelBuilder k, PairwiseStack stack, Val count, Func<Val, Val> term)
    {
        stack.Reset();
        k.For(k.Int(0), count, 1, i => stack.Push(i, [term(i)]));
        return stack.Total([k.Float(-0f)])[0];
    }

    // The moments (count, mean, M2: the sum of squares about the mean) of two parts merged (Chan et al.), the left part
    // first: an empty part (count 0) gives the other exactly.
    private static Val[] MomentsMerge(KernelBuilder k, Val[] left, Val[] right)
    {
        var (na, ma, sa) = (left[0], left[1], left[2]);
        var (nb, mb, sb) = (right[0], right[1], right[2]);
        var n = k.AddRounded(na, nb);
        var safe = k.Max(n, k.Float(1f));
        var delta = mb - ma;
        var mean = k.Fma(delta, nb / safe, ma);
        var m2 = k.Fma(k.MulRounded(delta, delta), k.MulRounded(na, nb) / safe, k.AddRounded(sa, sb));
        var leftEmpty = !(na > 0f);
        var rightEmpty = !(nb > 0f);
        return
        [
            k.Select(rightEmpty, na, k.Select(leftEmpty, nb, n)),
            k.Select(rightEmpty, ma, k.Select(leftEmpty, mb, mean)),
            k.Select(rightEmpty, sa, k.Select(leftEmpty, sb, m2)),
        ];
    }
}
