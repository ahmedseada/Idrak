// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Gpu.Vulkan;

// Convolutions without unfolded patches, average pooling, and image resampling.
//
// The convolution and its two gradients are each a product whose operands are read where they lie (implicit products):
// the patches of the input are gathered as the tile of the product needs them, never written out. Every product kernel
// has the tiling of batched_matmul (register-blocked: 4 × 4 outputs per invocation, blocks of 4S × 4S, steps of 16 along
// the sum) and of batched_matmul_tile (one output per invocation, S × S blocks), S = MatSide(width); the backend measures
// which (or the composed path, or the depthwise kernels) is fastest per shape.
//
//   forward          batch (image n, group q); rows: the group's filters f; columns: output positions (oh, ow); sum over
//                    (c, kh, kw) of weight[q·Fg + f, (c, kh, kw)] · x[n, q·Cg + c, oh·SH - PH + kh·DH, ow·SW - PW + kw·DW],
//                    then the bias and the activation.
//   backward input   batch (n, q); rows: the group's channels c; columns: input positions (ih, iw); sum over (kh, kw, f) of
//                    weight[q·Fg + f, (c, kh, kw)] · dy[n, q·Fg + f, oh, ow] where ih = oh·SH - PH + kh·DH (and so for
//                    the width) has a whole oh within [0, OH); added to dx.
//   backward weight  batch (q, split s); rows: the group's filters f; columns: (c, kh, kw); sum over the positions (n, oh,
//                    ow) of split s of dy[n, q·Fg + f, oh, ow] · x[n, q·Cg + c, ih, iw]: added to dweight with one
//                    split, else written to part[s] and finished by conv_split_reduce.
//
// One order per pass, whatever the choice (VulkanKernels.Order.cs; docs/optimization.md rule 86):
//
//   forward          every term of the sum in turn from +0, one fused multiply-add each (as the float32 products do), then
//                    the bias and the activation: the composed path's order too.
//   backward input   the composed path's order (its products, then col2im): dx, then each window (kh, kw) that covers
//                    the element, kh then kw descending, each window's sum over the group's filters in turn from +0 by
//                    fused multiply-adds, added in turn (InputPass, ConvWindowEnd).
//   backward weight  the positions in leaves of ConvLeaf, each leaf's terms in turn from +0 by fused multiply-adds, the
//                    leaves in the pairwise tree of consecutive leaves (PairwiseStack), a split an aligned power of two of
//                    them (chunk, a push constant), conv_split_reduce the splits' tree, then dweight + the total; unfolded
//                    (conv_unfolded_weight) the same product over im2col's patches.
//
// The depthwise kernels take the same orders (terms outside the image are skipped: adding 0 · v to a sum that started
// from +0 changes nothing). Push constants of every convolution kernel: N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, DH, DW,
// F, G, flags, splits, chunk (flags: 1 bias, activation · 2 for the forward kernels; chunk: positions per split of the
// weight gradient).
internal static partial class VulkanKernels
{
    /// <summary>The variants of the implicit convolution products (the backend's tile choice).</summary>
    public static readonly string[] ConvolutionPasses = ["conv_forward", "conv_backward_input", "conv_backward_weight", "conv_unfolded_weight"];

    /// <summary>
    /// Positions per leaf of the weight gradient's tree: a multiple of every step of the implicit products along the sum
    /// (16 for the register-blocked kernel, MatSide(width) up to 32 for the tiled one), so a step never straddles leaves.
    /// 128 rather than 32: the register-blocked kernel's 16 sums then go to its stack every 8 steps, not every 2 (on
    /// lavapipe its weight gradient took 1.3 to 1.5 times the old time with leaves of 32, 1.02 to 1.07 with 128).
    /// </summary>
    public const int ConvLeaf = 128;

    private const int ConvLeafShift = 7;

    /// <summary>
    /// Entries of the register-blocked weight gradient's pairwise stack (16 sums an entry, the top in registers): 128 floats
    /// of function memory per invocation, a split walks at most 2^9 leaves (65,536 positions; the backend raises the
    /// splits to fit).
    /// </summary>
    public const int ConvBlockedStackEntries = 9;

    /// <summary>Entries of the tiled weight gradient's pairwise stack (one sum an entry): a split walks at most 2^18 leaves.</summary>
    public const int ConvTileStackEntries = 18;

    // Entries of the depthwise weight gradient's stack: an invocation walks at most 2^20 leaves (2^30 positions over 64 invocations need 2^17).
    private const int ConvDepthwiseStackEntries = 20;

    private static IEnumerable<(string, Func<SpirvKernel>)> ConvolutionKernels()
    {
        foreach (var (name, spec) in new (string, Func<ConvPass>)[]
        {
            ("conv_forward", () => new ForwardPass()),
            ("conv_backward_input", () => new InputPass()),
            ("conv_backward_weight", () => new WeightPass(unfolded: false)),
            ("conv_unfolded_weight", () => new WeightPass(unfolded: true)),
        })
        {
            yield return (name, () => ConvGemmBlocked(name, spec()));
            yield return (name + "_tile", () => ConvGemmTile(name + "_tile", spec()));
        }

        // dweight[i] += Σ_s part[s · count + i], the splits (aligned subtrees of the leaves) in their pairwise tree.
        yield return ("conv_split_reduce", () =>
        {
            var k = new KernelBuilder("conv_split_reduce", Block);
            var (part, dw) = (k.Buffer("part"), k.Buffer("dw"));
            var (count, splits) = (k.PushInt("count"), k.PushInt("splits"));
            var stack = new PairwiseStack(k, 1, SplitStackEntries, SumsMerge(k));
            Grid(k, count, i => dw[i] = dw[i] + Pairwise(k, stack, splits, s => part[s * count + i]));
            return k.Build();
        });

        // Depthwise (one input channel per group, M = F / C filters a channel): an invocation per output sums its window,
        // rows then columns, from +0 by fused multiply-adds (the forward product's order), adds the bias and applies the
        // activation.
        yield return ("conv_depthwise", () =>
        {
            var k = new KernelBuilder("conv_depthwise", Block);
            var (x, w, bias, y) = (k.Buffer("x"), k.Buffer("w"), k.Buffer("bias"), k.Buffer("y"));
            var g = ConvPushes(k);
            Grid(k, g.N * g.F * g.OH * g.OW, o =>
            {
                var ow = o % g.OW;
                var t = o / g.OW;
                var oh = t % g.OH;
                var nf = t / g.OH;
                var f = nf % g.F;
                var n = nf / g.F;
                var c = f / (g.F / g.C);
                var plane = (n * g.C + c) * g.H;
                var acc = k.Local(0f);
                k.For(k.Int(0), g.KH, 1, i =>
                {
                    var ih = oh * g.SH - g.PH + i * g.DH;
                    k.If((ih >= 0) & (ih < g.H), () =>
                    {
                        var row = (plane + ih) * g.W;
                        k.For(k.Int(0), g.KW, 1, j =>
                        {
                            var iw = ow * g.SW - g.PW + j * g.DW;
                            k.If((iw >= 0) & (iw < g.W), () => acc.V = k.Fma(w[(f * g.KH + i) * g.KW + j], x[row + iw], acc.V));
                        });
                    });
                });
                var v = acc.V + k.Select((g.Flags & 1).Ne(0), bias[f], k.Float(0f));
                y[o] = Activated(k, v, g.Flags >> 1);
            });
            return k.Build();
        });

        // dx[n, c, ih, iw] += the windows covering the element, kh then kw descending (col2im's order), each window's sum
        // over the channel's filters in order from +0 by fused multiply-adds, added to dx in turn (the input gradient's
        // order): an invocation per input element.
        yield return ("conv_depthwise_backward_input", () =>
        {
            var k = new KernelBuilder("conv_depthwise_backward_input", Block);
            var (dy, w, dx) = (k.Buffer("dy"), k.Buffer("w"), k.Buffer("dx"));
            var g = ConvPushes(k);
            Grid(k, g.N * g.C * g.H * g.W, idx =>
            {
                var iw = idx % g.W;
                var t = idx / g.W;
                var ih = t % g.H;
                var nc = t / g.H;
                var c = nc % g.C;
                var n = nc / g.C;
                var perChannel = g.F / g.C;
                var acc = k.Local(dx[idx]);
                k.For(k.Int(0), g.KH, 1, u =>
                {
                    var i = g.KH - 1 - u;
                    var th = ih + g.PH - i * g.DH;
                    var oh = th / g.SH;
                    k.If((th >= 0) & (th % g.SH).Eq(0) & (oh < g.OH), () =>
                    {
                        k.For(k.Int(0), g.KW, 1, v =>
                        {
                            var j = g.KW - 1 - v;
                            var tw = iw + g.PW - j * g.DW;
                            var ow = tw / g.SW;
                            k.If((tw >= 0) & (tw % g.SW).Eq(0) & (ow < g.OW), () =>
                            {
                                var window = k.Local(0f);
                                k.For(k.Int(0), perChannel, 1, m =>
                                {
                                    var f = c * perChannel + m;
                                    window.V = k.Fma(w[(f * g.KH + i) * g.KW + j], dy[((n * g.F + f) * g.OH + oh) * g.OW + ow], window.V);
                                });
                                acc.V = acc.V + window.V;
                            });
                        });
                    });
                });
                dx[idx] = acc.V;
            });
            return k.Build();
        });

        // dweight[f, i, j] += Σ_{n, oh, ow} dy[n, f, oh, ow] · x[n, f / M, ih, iw] in the weight gradient's order: a
        // workgroup per weight; the positions in leaves of ConvLeaf, invocation t walking leaves [t · per, (t + 1) · per)
        // (per the power of two that lets the workgroup cover them all) through a PairwiseStack, each leaf's terms in turn
        // from +0; the invocations' subtrees then merged in the pairwise tree of consecutive invocations
        // (KernelBuilder.ReducePairwise), so any width gives the bits of the implicit products at any split count.
        yield return ("conv_depthwise_backward_weight", () =>
        {
            var k = new KernelBuilder("conv_depthwise_backward_weight", Block);
            var (x, dy, dw) = (k.Buffer("x"), k.Buffer("dy"), k.Buffer("dw"));
            var g = ConvPushes(k);
            var scratch = new[] { k.Shared("scratch", Block) };
            var merge = SumsMerge(k);
            var stack = new PairwiseStack(k, 1, ConvDepthwiseStackEntries, merge);
            var positions = g.N * g.OH * g.OW;
            var leaves = (positions + (ConvLeaf - 1)) >> ConvLeafShift;
            var per = k.Local(1);
            k.While(() => per.V * Block < leaves, () => per.V = per.V * 2);
            var first = k.LocalX * per.V;
            var last = k.Min(leaves, first + per.V);
            EachRow(k, g.F * g.KH * g.KW, e =>
            {
                var j = e % g.KW;
                var t = e / g.KW;
                var i = t % g.KH;
                var f = t / g.KH;
                var c = f / (g.F / g.C);
                stack.Reset();
                k.For(first, last, 1, leaf =>
                {
                    var acc = k.Local(0f);
                    var start = leaf << ConvLeafShift;
                    k.For(start, k.Min(positions, start + ConvLeaf), 1, p =>
                    {
                        var ow = p % g.OW;
                        var u = p / g.OW;
                        var oh = u % g.OH;
                        var n = u / g.OH;
                        var ih = oh * g.SH - g.PH + i * g.DH;
                        var iw = ow * g.SW - g.PW + j * g.DW;
                        k.If((ih >= 0) & (ih < g.H) & (iw >= 0) & (iw < g.W), () =>
                            acc.V = k.Fma(dy[((n * g.F + f) * g.OH + oh) * g.OW + ow], x[((n * g.C + c) * g.H + ih) * g.W + iw], acc.V));
                    });
                    stack.Push(leaf - first, [acc.V]);
                });
                var sum = k.ReducePairwise(scratch, stack.Total([k.Float(-0f)]), merge)[0];
                k.If(k.LocalX.Eq(0), () => dw[e] = dw[e] + sum);
            });
            return k.Build();
        });

        // Average pooling: an invocation per output sums its window in row order and divides (KH·KW, or the positions it
        // covers); the gradient gathers dy / divisor of the windows covering each input element, in window order.
        yield return ("avg_pool", () =>
        {
            var k = new KernelBuilder("avg_pool", Block);
            var (x, y) = (k.Buffer("x"), k.Buffer("y"));
            var g = Geometry(k);
            var (countPad, bottom, right) = (k.PushInt("countPad"), k.PushInt("bottom"), k.PushInt("right"));
            Grid(k, g.N * g.C * g.OH * g.OW, idx =>
            {
                var ow = idx % g.OW;
                var t = idx / g.OW;
                var (oh, nc) = (t % g.OH, t / g.OH);
                var plane = nc * g.H * g.W;
                var (r0, c0) = (oh * g.SH - g.PH, ow * g.SW - g.PW);
                var (rStart, rEnd) = (k.Max(r0, k.Int(0)), k.Min(r0 + g.KH, g.H));
                var (cStart, cEnd) = (k.Max(c0, k.Int(0)), k.Min(c0 + g.KW, g.W));
                var sum = k.Local(0f);
                k.For(rStart, rEnd, 1, r => k.For(cStart, cEnd, 1, col => sum.V = sum.V + x[plane + r * g.W + col]));
                y[idx] = sum.V / PoolDivisor(k, g, oh, ow, countPad, bottom, right);
            });
            return k.Build();
        });

        yield return ("avg_pool_backward", () =>
        {
            var k = new KernelBuilder("avg_pool_backward", Block);
            var (dy, dx) = (k.Buffer("dy"), k.Buffer("dx"));
            var g = Geometry(k);
            var (countPad, bottom, right) = (k.PushInt("countPad"), k.PushInt("bottom"), k.PushInt("right"));
            Grid(k, g.N * g.C * g.H * g.W, idx =>
            {
                var iw = idx % g.W;
                var t = idx / g.W;
                var (ih, nc) = (t % g.H, t / g.H);
                var (ohFirst, ohLast) = Covering(k, ih, g.PH, g.KH, g.SH, g.OH);
                var (owFirst, owLast) = Covering(k, iw, g.PW, g.KW, g.SW, g.OW);
                var acc = k.Local(dx[idx]);
                k.For(ohFirst, ohLast, 1, oh =>
                {
                    var rowBase = (nc * g.OH + oh) * g.OW;
                    k.For(owFirst, owLast, 1, ow => acc.V = acc.V + dy[rowBase + ow] / PoolDivisor(k, g, oh, ow, countPad, bottom, right));
                });
                dx[idx] = acc.V;
            });
            return k.Build();
        });

        // Resampling and per-channel normalization (Backend.ResizeNormalize): an invocation per output takes the vertical
        // taps in order, each the horizontal pass of its row (the taps in order) computed in place, so the result is the
        // two-pass one without the intermediate image; floats, or Pillow's 8-bit passes in integers.
        yield return ("resize_normalize", () =>
        {
            var k = new KernelBuilder("resize_normalize", Block);
            var (x, coefficients, values, y) = (k.Buffer("x"), k.Buffer("coefficients"), k.Buffer("values"), k.Buffer("y"));
            var (planes, channels, height, width) = (k.PushInt("planes"), k.PushInt("channels"), k.PushInt("height"), k.PushInt("width"));
            var (outHeight, outWidth, xTaps, yTaps, bytes) = (k.PushInt("outHeight"), k.PushInt("outWidth"), k.PushInt("xTaps"), k.PushInt("yTaps"), k.PushInt("bytes"));
            var isBytes = bytes.Ne(0);
            var yBase = k.Select(xTaps > 0, outWidth * (xTaps + 2), k.Int(0));
            Grid(k, planes * outHeight * outWidth, o =>
            {
                var ox = o % outWidth;
                var t = o / outWidth;
                var (oy, p) = (t % outHeight, t / outHeight);
                var plane = p * height * width;
                var xAt = ox * (xTaps + 2);

                // One row's horizontal value: a float, or the 8-bit pass's byte (as a float); without horizontal taps, the
                // input itself. Every read is under the branch that needs it (the buffers' lengths depend on the mode).
                Val Across(Val row)
                {
                    var f = k.Local(0f);
                    k.If(xTaps > 0, () =>
                    {
                        var start = plane + row * width + coefficients.Int(xAt);
                        var count = coefficients.Int(xAt + 1);
                        k.If(isBytes, () =>
                        {
                            var s = k.Local(1 << 21);
                            k.For(k.Int(0), count, 1, i => s.V = s.V + Byte(k, x, start + i) * coefficients.Int(xAt + 2 + i));
                            f.V = Clip8(k, s.V);
                        }, () => k.For(k.Int(0), count, 1, i => f.V = f.V + x[start + i] * coefficients[xAt + 2 + i]));
                    }, () =>
                    {
                        var at = plane + row * width + ox;
                        k.If(isBytes, () => f.V = Byte(k, x, at).ToFloat(), () => f.V = x[at]);
                    });
                    return f.V;
                }

                var value = k.Local(0f);
                k.If(yTaps > 0, () =>
                {
                    var yAt = yBase + oy * (yTaps + 2);
                    var (first, count) = (coefficients.Int(yAt), coefficients.Int(yAt + 1));
                    k.If(isBytes, () =>
                    {
                        var s = k.Local(1 << 21);
                        k.For(k.Int(0), count, 1, i => s.V = s.V + Across(first + i).ToInt() * coefficients.Int(yAt + 2 + i));
                        value.V = Clip8(k, s.V);
                    }, () => k.For(k.Int(0), count, 1, i => value.V = value.V + Across(first + i) * coefficients[yAt + 2 + i]));
                }, () => value.V = Across(oy));
                var c = p % channels;
                k.If(isBytes, () => y[o] = values[256 * c + value.V.ToInt()], () => y[o] = value.V * values[2 * c] + values[2 * c + 1]);
            });
            return k.Build();
        });
    }

    // Byte i of the packed words (bits 8·(i % 4) of word i / 4) as an int.
    private static Val Byte(KernelBuilder k, Buf x, Val i) => (x.UInt(i >> 2).ShiftRight(((i & 3) * 8).AsUInt()) & 0xFF).AsInt();

    // Pillow's clip8 of a 22-bit fixed-point sum, as a float: 255 from 2^30 up, 0 at or below 0, else the sum >> 22.
    private static Val Clip8(KernelBuilder k, Val sum) =>
        k.Select(sum >= (1 << 30), k.Float(255f), k.Select(sum <= 0, k.Float(0f), (sum >> 22).ToFloat()));

    // The divisor of an average-pooling window: its rows and columns up to the padded end (H + bottom, W + right), or
    // (countPad 0) the input positions it covers, at least 1.
    private static Val PoolDivisor(KernelBuilder k, WindowGeometry g, Val oh, Val ow, Val countPad, Val bottom, Val right)
    {
        var (r0, c0) = (oh * g.SH - g.PH, ow * g.SW - g.PW);
        var counted = (k.Min(r0 + g.KH, g.H + bottom) - r0) * (k.Min(c0 + g.KW, g.W + right) - c0);
        var rows = k.Min(r0 + g.KH, g.H) - k.Max(r0, k.Int(0));
        var cols = k.Min(c0 + g.KW, g.W) - k.Max(c0, k.Int(0));
        return k.Select(countPad.Ne(0), counted, k.Max(rows * cols, k.Int(1))).ToFloat();
    }

    // The activation code of the forward kernels (ConvActivation: 0 none, 1 ReLU, 2 sigmoid, 3 tanh, 4 GELU, 5 SiLU), as
    // the element-wise kernels compute each.
    private static Val Activated(KernelBuilder k, Val v, Val code)
    {
        var r = k.Local(v);
        k.If(code.Ne(0), () =>
        {
            k.If(code.Eq(1), () => r.V = k.Max(v, k.Float(0f)));
            k.If(code.Eq(2), () => r.V = Sigmoid(k, v));
            k.If(code.Eq(3), () => r.V = Tanh(k, v));
            k.If(code.Eq(4), () => r.V = Gelu(k, v));
            k.If(code.Eq(5), () => r.V = v * Sigmoid(k, v));
        });
        return r.V;
    }

    // The push constants of the convolution kernels, in the order they declare them.
    private readonly record struct ConvGeometryPush(Val N, Val C, Val H, Val W, Val KH, Val KW, Val SH, Val SW, Val PH, Val PW, Val OH, Val OW,
        Val DH, Val DW, Val F, Val G, Val Flags, Val Splits, Val Chunk);

    private static ConvGeometryPush ConvPushes(KernelBuilder k) => new(
        k.PushInt("N"), k.PushInt("C"), k.PushInt("H"), k.PushInt("W"), k.PushInt("KH"), k.PushInt("KW"), k.PushInt("SH"), k.PushInt("SW"),
        k.PushInt("PH"), k.PushInt("PW"), k.PushInt("OH"), k.PushInt("OW"), k.PushInt("DH"), k.PushInt("DW"), k.PushInt("F"), k.PushInt("G"),
        k.PushInt("flags"), k.PushInt("splits"), k.PushInt("chunk"));

    // ------------------------------------------------------------------ the implicit products

    // One pass of the convolution as a product: its buffers, its batch count, and per batch entry the product's rows,
    // columns, range of the sum, its operands and where each output goes.
    private abstract class ConvPass
    {
        protected KernelBuilder K = null!;
        protected ConvGeometryPush G;

        public void Declare(KernelBuilder k)
        {
            K = k;
            Buffers(k);
            G = ConvPushes(k);
        }

        protected abstract void Buffers(KernelBuilder k);

        // Whether the sum runs in the weight gradient's order (leaves in a pairwise tree, see the file's comment) rather
        // than every term in turn.
        public virtual bool Tree => false;

        // Whether the sum runs in the input gradient's order: segments of SegmentLength terms, each summed from +0, then
        // added in turn to Initial where Covers holds (ConvSegmentEnd).
        public virtual bool Windows => false;

        public virtual Val SegmentLength => throw new NotSupportedException();

        public virtual Val Initial(Val row, Val column) => throw new NotSupportedException();

        public virtual Val Covers(Val segment, Val column) => throw new NotSupportedException();

        public abstract Val Batch { get; }

        // Sets the batch entry's values (called inside the loop over batch entries).
        public abstract void Begin(Val bi);

        public abstract Val Rows { get; }

        public abstract Val Columns { get; }

        public abstract Val SumStart { get; }

        public abstract Val SumEnd { get; }

        // The left operand at (row, kk); called where row and kk are in range.
        public abstract Val A(Val row, Val kk);

        // The right operand at (kk, column): whether it is inside the data, and its load (called only then).
        public abstract (Val Inside, Func<Val> Load) B(Val kk, Val column);

        public abstract void Store(Val row, Val column, Val value);
    }

    // y = conv(x, w) + bias, activated.
    private sealed class ForwardPass : ConvPass
    {
        private Buf _x = null!, _w = null!, _bias = null!, _y = null!;
        private Val _n, _q, _fg, _cg, _patch;

        protected override void Buffers(KernelBuilder k) => (_x, _w, _bias, _y) = (k.Buffer("x"), k.Buffer("w"), k.Buffer("bias"), k.Buffer("y"));

        public override Val Batch => G.N * G.G;

        public override void Begin(Val bi)
        {
            (_n, _q) = (bi / G.G, bi % G.G);
            (_fg, _cg) = (G.F / G.G, G.C / G.G);
            _patch = _cg * G.KH * G.KW;
        }

        public override Val Rows => _fg;

        public override Val Columns => G.OH * G.OW;

        public override Val SumStart => K.Int(0);

        public override Val SumEnd => _patch;

        public override Val A(Val row, Val kk) => _w[(_q * _fg + row) * _patch + kk];

        public override (Val, Func<Val>) B(Val kk, Val column)
        {
            var area = G.KH * G.KW;
            var (c, r) = (kk / area, kk % area);
            var (kh, kw) = (r / G.KW, r % G.KW);
            var (oh, ow) = (column / G.OW, column % G.OW);
            var ih = oh * G.SH - G.PH + kh * G.DH;
            var iw = ow * G.SW - G.PW + kw * G.DW;
            var inside = (ih >= 0) & (ih < G.H) & (iw >= 0) & (iw < G.W);
            return (inside, () => _x[((_n * G.C + _q * _cg + c) * G.H + ih) * G.W + iw]);
        }

        public override void Store(Val row, Val column, Val value)
        {
            var f = _q * _fg + row;
            var v = value + K.Select((G.Flags & 1).Ne(0), _bias[f], K.Float(0f));
            _y[(_n * G.F + f) * (G.OH * G.OW) + column] = Activated(K, v, G.Flags >> 1);
        }
    }

    // dx += the transposed convolution of dy, in the input gradient's order (the composed path's: the products' sums over
    // the filters, then the folded windows, Backend.Convolution.cs and col2im): segment s is window r = KH·KW - 1 - s
    // (kh, kw descending, as col2im walks oh and ow ascending), its terms the group's filters f, dx then each covering
    // window's sum in turn.
    private sealed class InputPass : ConvPass
    {
        private Buf _dy = null!, _w = null!, _dx = null!;
        private Val _n, _q, _fg, _cg, _patch;

        protected override void Buffers(KernelBuilder k) => (_dy, _w, _dx) = (k.Buffer("dy"), k.Buffer("w"), k.Buffer("dx"));

        public override bool Windows => true;

        public override Val Batch => G.N * G.G;

        public override void Begin(Val bi)
        {
            (_n, _q) = (bi / G.G, bi % G.G);
            (_fg, _cg) = (G.F / G.G, G.C / G.G);
            _patch = _cg * G.KH * G.KW;
        }

        public override Val Rows => _cg;

        public override Val Columns => G.H * G.W;

        public override Val SumStart => K.Int(0);

        public override Val SumEnd => G.KH * G.KW * _fg;

        public override Val SegmentLength => _fg;

        // A segment's window (its index in the patch) and a term's filter.
        private (Val R, Val F) Term(Val kk) => (G.KH * G.KW - 1 - kk / _fg, kk % _fg);

        // Window r's output position for input position `column`, and whether it reads that position.
        private (Val Inside, Val Oh, Val Ow) Window(Val r, Val column)
        {
            var (kh, kw) = (r / G.KW, r % G.KW);
            var (ih, iw) = (column / G.W, column % G.W);
            var th = ih + G.PH - kh * G.DH;
            var tw = iw + G.PW - kw * G.DW;
            var (oh, ow) = (th / G.SH, tw / G.SW);
            return ((th >= 0) & (tw >= 0) & (th % G.SH).Eq(0) & (tw % G.SW).Eq(0) & (oh < G.OH) & (ow < G.OW), oh, ow);
        }

        public override Val A(Val row, Val kk)
        {
            var (r, f) = Term(kk);
            return _w[(_q * _fg + f) * _patch + row * (G.KH * G.KW) + r];
        }

        public override (Val, Func<Val>) B(Val kk, Val column)
        {
            var (r, f) = Term(kk);
            var (inside, oh, ow) = Window(r, column);
            return (inside, () => _dy[((_n * G.F + _q * _fg + f) * G.OH + oh) * G.OW + ow]);
        }

        public override Val Initial(Val row, Val column) => _dx[(_n * G.C + _q * _cg + row) * (G.H * G.W) + column];

        public override Val Covers(Val segment, Val column) => Window(G.KH * G.KW - 1 - segment, column).Inside;

        public override void Store(Val row, Val column, Val value) => _dx[(_n * G.C + _q * _cg + row) * (G.H * G.W) + column] = value;
    }

    // dweight += dyᵀ · patches, the positions in splits of `chunk` (an aligned power of two of the leaves). Unfolded, the
    // patches are read from im2col's cols [(n, oh, ow), (c, kh, kw)] (zeros outside the image) instead of gathered from x.
    private sealed class WeightPass(bool unfolded) : ConvPass
    {
        private Buf _x = null!, _dy = null!, _dw = null!, _part = null!;
        private Val _q, _s, _fg, _cg, _patch, _start, _end;

        protected override void Buffers(KernelBuilder k) =>
            (_x, _dy, _dw, _part) = (k.Buffer(unfolded ? "cols" : "x"), k.Buffer("dy"), k.Buffer("dw"), k.Buffer("part"));

        public override bool Tree => true;

        public override Val Batch => G.G * G.Splits;

        public override void Begin(Val bi)
        {
            (_q, _s) = (bi / G.Splits, bi % G.Splits);
            (_fg, _cg) = (G.F / G.G, G.C / G.G);
            _patch = _cg * G.KH * G.KW;
            var total = G.N * G.OH * G.OW;
            _start = K.Min(_s * G.Chunk, total);
            _end = K.Min(_start + G.Chunk, total);
        }

        public override Val Rows => _fg;

        public override Val Columns => _patch;

        public override Val SumStart => _start;

        public override Val SumEnd => _end;

        public override Val A(Val row, Val kk)
        {
            var outputs = G.OH * G.OW;
            return _dy[((kk / outputs) * G.F + _q * _fg + row) * outputs + kk % outputs];
        }

        public override (Val, Func<Val>) B(Val kk, Val column)
        {
            if (unfolded)
            {
                return (K.Bool(true), () => _x[kk * (G.C * G.KH * G.KW) + _q * _patch + column]);
            }

            var outputs = G.OH * G.OW;
            var (n, p) = (kk / outputs, kk % outputs);
            var (oh, ow) = (p / G.OW, p % G.OW);
            var area = G.KH * G.KW;
            var (c, r) = (column / area, column % area);
            var (kh, kw) = (r / G.KW, r % G.KW);
            var ih = oh * G.SH - G.PH + kh * G.DH;
            var iw = ow * G.SW - G.PW + kw * G.DW;
            var inside = (ih >= 0) & (ih < G.H) & (iw >= 0) & (iw < G.W);
            return (inside, () => _x[((n * G.C + _q * _cg + c) * G.H + ih) * G.W + iw]);
        }

        public override void Store(Val row, Val column, Val value)
        {
            var o = (_q * _fg + row) * _patch + column;
            K.If(G.Splits.Eq(1), () => _dw[o] = _dw[o] + value, () => _part[_s * (G.F * _patch) + o] = value);
        }
    }

    // The register-blocked tiling of batched_matmul over a convolution pass (see the file's comment).
    //
    // Workgroup memory: tileA[kk · (T + 1) + r] and tileB[kk · (T + 1) + col] are written only between the barrier ending
    // the previous step's reads and the barrier before this step's reads, each element by exactly one invocation.
    private static SpirvKernel ConvGemmBlocked(string name, ConvPass pass)
    {
        int side = MatSide(Block), threads = side * side, Per = MatPer, tileEdge = Per * side, stride = tileEdge + 1;
        int sideShift = System.Numerics.BitOperations.Log2((uint)side), edgeShift = System.Numerics.BitOperations.Log2((uint)tileEdge);
        var k = new KernelBuilder(name, threads);
        pass.Declare(k);
        var tileA = k.Shared("tileA", MatDepth * stride);
        var tileB = k.Shared("tileB", MatDepth * stride);
        var tid = k.LocalX;
        var (tx, ty) = (tid & (side - 1), tid >> sideShift);
        var rowBase = k.GroupY * tileEdge;
        var colBase = k.GroupX * tileEdge;
        var acc = new Var[Per * Per];
        for (int i = 0; i < acc.Length; i++)
        {
            acc[i] = k.Local(ScalarKind.Float);
        }

        var stack = pass.Tree ? new PairwiseStack(k, acc.Length, ConvBlockedStackEntries, SumsMerge(k)) : null;
        var columnOf = Enumerable.Range(0, Per).Select(j => colBase + tx + side * j).ToArray();
        var rowOf = Enumerable.Range(0, Per).Select(i => rowBase + ty + side * i).ToArray();
        var segments = pass.Windows ? new ConvSegments(k, pass, acc.Length, segment =>
        {
            var covers = columnOf.Select(c => pass.Covers(segment, c)).ToArray();
            return [.. Enumerable.Range(0, acc.Length).Select(o => covers[o % Per])];
        }) : null;

        // Invocation tid's p-th element of each staged tile: (r, ka) of the left operand along the sum (16 consecutive
        // invocations a row), (kb, col) of the right one along the columns.
        var loads = new (Val R, Val KA, Val Col, Val KB)[tileEdge * MatDepth / threads];
        for (int p = 0; p < loads.Length; p++)
        {
            var l = tid + p * threads;
            loads[p] = (l >> 4, l & (MatDepth - 1), l & (tileEdge - 1), l >> edgeShift);
        }

        k.For(k.GroupZ, pass.Batch, bi =>
        {
            pass.Begin(bi);
            var (rows, columns, start, end) = (pass.Rows, pass.Columns, pass.SumStart, pass.SumEnd);
            foreach (var v in acc)
            {
                v.V = k.Float(0f);
            }

            stack?.Reset();
            segments?.Begin(MatDepth, o => (rowOf[o / Per], columnOf[o % Per]), rows, columns);
            k.For(start, end, MatDepth, t =>
            {
                foreach (var (r, ka, col, kb) in loads)
                {
                    var (row, kka) = (rowBase + r, t + ka);
                    var va = k.Local(0f);
                    k.If((row < rows) & (kka < end), () => va.V = pass.A(row, kka));
                    tileA[ka * stride + r] = va.V;
                    var (column, kkb) = (colBase + col, t + kb);
                    var vb = k.Local(0f);
                    var (inside, load) = pass.B(kkb, column);
                    k.If((kkb < end) & (column < columns) & inside, () => vb.V = load());
                    tileB[kb * stride + col] = vb.V;
                }

                k.Barrier();

                // The step's terms; `each`: a segment may end after any of them (segments not a multiple of the step).
                void Multiply(bool each)
                {
                    var sums = acc.Select(v => v.V).ToArray();
                    for (int e = 0; e < MatDepth; e++)
                    {
                        var av = new Val[Per];
                        var bv = new Val[Per];
                        for (int i = 0; i < Per; i++)
                        {
                            av[i] = tileA[k.Int(e * stride + side * i) + ty];
                            bv[i] = tileB[k.Int(e * stride + side * i) + tx];
                        }

                        for (int i = 0; i < Per; i++)
                        {
                            for (int j = 0; j < Per; j++)
                            {
                                sums[i * Per + j] = k.Fma(av[i], bv[j], sums[i * Per + j]);
                            }
                        }

                        if (each)
                        {
                            for (int i = 0; i < acc.Length; i++)
                            {
                                acc[i].V = sums[i];
                            }

                            segments!.End(t + (e + 1), acc);
                            sums = acc.Select(v => v.V).ToArray();
                        }
                    }

                    for (int i = 0; i < acc.Length; i++)
                    {
                        acc[i].V = sums[i];
                    }
                }

                if (segments is null)
                {
                    Multiply(false);
                }
                else
                {
                    k.If(segments.Aligned, () => Multiply(false), () => Multiply(true));
                }

                k.Barrier();
                if (stack is not null)
                {
                    ConvLeafEnd(k, stack, acc, t, MatDepth, start, end);
                }

                if (segments is not null)
                {
                    k.If(segments.Aligned, () => segments.End(t + MatDepth, acc));
                }
            });

            Val[] totals = segments is not null ? segments.Totals
                : stack is null ? [.. acc.Select(v => v.V)] : stack.Total([.. acc.Select(_ => k.Float(-0f))]);
            for (int i = 0; i < Per; i++)
            {
                for (int j = 0; j < Per; j++)
                {
                    var (row, col, value) = (rowOf[i], columnOf[j], totals[i * Per + j]);
                    k.If((row < rows) & (col < columns), () => pass.Store(row, col, value));
                }
            }
        }, k.GroupsZ);
        return k.Build();
    }

    // The one-output-per-invocation tiling of batched_matmul_tile over a convolution pass: S × S workgroups, S × S tiles of
    // both operands staged per step of S along the sum.
    //
    // Workgroup memory: tileA[ty · S + tx], tileB[ty · S + tx] written by invocation (tx, ty) only, between the barrier
    // ending the previous step's reads and the barrier before this step's reads.
    private static SpirvKernel ConvGemmTile(string name, ConvPass pass)
    {
        int side = MatSide(Block);
        var k = new KernelBuilder(name, side, side);
        pass.Declare(k);
        var tileA = k.Shared("tileA", side * side);
        var tileB = k.Shared("tileB", side * side);
        var (tx, ty) = (k.LocalX, k.LocalY);
        var row = k.GroupY * side + ty;
        var col = k.GroupX * side + tx;
        var at = ty * side + tx;
        var stack = pass.Tree ? new PairwiseStack(k, 1, ConvTileStackEntries, SumsMerge(k)) : null;
        var segments = pass.Windows ? new ConvSegments(k, pass, 1, segment => [pass.Covers(segment, col)]) : null;
        k.For(k.GroupZ, pass.Batch, bi =>
        {
            pass.Begin(bi);
            var (rows, columns, start, end) = (pass.Rows, pass.Columns, pass.SumStart, pass.SumEnd);
            var acc = k.Local(0f);
            stack?.Reset();
            segments?.Begin(side, _ => (row, col), rows, columns);

            k.For(start, end, side, t =>
            {
                var (ka, kb) = (t + tx, t + ty);
                var va = k.Local(0f);
                k.If((row < rows) & (ka < end), () => va.V = pass.A(row, ka));
                tileA[at] = va.V;
                var vb = k.Local(0f);
                var (inside, load) = pass.B(kb, col);
                k.If((kb < end) & (col < columns) & inside, () => vb.V = load());
                tileB[at] = vb.V;
                k.Barrier();

                // The step's terms; `each`: a segment may end after any of them (segments not a multiple of the step).
                void Multiply(bool each)
                {
                    var sum = acc.V;
                    for (int e = 0; e < side; e++)
                    {
                        sum = k.Fma(tileA[ty * side + e], tileB[k.Int(e * side) + tx], sum);
                        if (each)
                        {
                            acc.V = sum;
                            segments!.End(t + (e + 1), [acc]);
                            sum = acc.V;
                        }
                    }

                    acc.V = sum;
                }

                if (segments is null)
                {
                    Multiply(false);
                }
                else
                {
                    k.If(segments.Aligned, () => Multiply(false), () => Multiply(true));
                }

                k.Barrier();
                if (stack is not null)
                {
                    ConvLeafEnd(k, stack, [acc], t, side, start, end);
                }

                if (segments is not null)
                {
                    k.If(segments.Aligned, () => segments.End(t + side, [acc]));
                }
            });
            var total = segments is not null ? segments.Totals[0] : stack is null ? acc.V : stack.Total([k.Float(-0f)])[0];
            k.If((row < rows) & (col < columns), () => pass.Store(row, col, total));
        }, k.GroupsZ);
        return k.Build();
    }

    // The input gradient's windows in an implicit product (ConvPass.Windows): the sum runs from 0 in segments of
    // pass.SegmentLength terms, each output's sum from +0; when a segment ends (End: after the term before `next`), each
    // output adds its sum to its total where the segment's window covers it (`covers`: per output) and its sum starts again
    // from +0. Aligned: the segments are a multiple of the step, so they end only with a step (checked once a step);
    // otherwise after any term. The segment's end and index are the workgroup's own.
    private sealed class ConvSegments(KernelBuilder k, ConvPass pass, int outputs, Func<Val, Val[]> covers)
    {
        private readonly Var[] _totals = [.. Enumerable.Range(0, outputs).Select(_ => k.Local(ScalarKind.Float))];
        private readonly Var _end = k.Local(ScalarKind.Int), _index = k.Local(ScalarKind.Int);
        private Val _length;

        public Val Aligned { get; private set; }

        public Val[] Totals => [.. _totals.Select(t => t.V)];

        // Starts a batch entry: each output's total from pass.Initial (output o at place(o), within rows and columns).
        public void Begin(int step, Func<int, (Val Row, Val Column)> place, Val rows, Val columns)
        {
            _length = pass.SegmentLength;
            Aligned = (_length % step).Eq(0);
            _end.V = _length;
            _index.V = k.Int(0);
            for (int o = 0; o < _totals.Length; o++)
            {
                var (row, column, total) = (place(o).Row, place(o).Column, _totals[o]);
                total.V = k.Float(0f);
                k.If((row < rows) & (column < columns), () => total.V = pass.Initial(row, column));
            }
        }

        public void End(Val next, Var[] sums) => k.If(next.Eq(_end.V), () =>
        {
            var covered = covers(_index.V);
            for (int o = 0; o < sums.Length; o++)
            {
                _totals[o].V = k.Select(covered[o], _totals[o].V + sums[o].V, _totals[o].V);
                sums[o].V = k.Float(0f);
            }

            _end.V = _end.V + _length;
            _index.V = _index.V + 1;
        });
    }

    // The weight gradient's leaves in an implicit product: after the step of `step` terms at t (the split's sum running
    // from `start`, a multiple of ConvLeaf, to `end`), when the step ends a leaf or the split, the leaf's sums go to the
    // pairwise stack (its index in the split) and the sums start again from +0. The condition is the workgroup's own.
    private static void ConvLeafEnd(KernelBuilder k, PairwiseStack stack, Var[] sums, Val t, int step, Val start, Val end)
    {
        if (ConvLeaf % step != 0)
        {
            throw new InvalidOperationException($"A step of {step} along the sum straddles the weight gradient's leaves of {ConvLeaf}.");
        }

        var next = t + step;
        k.If(((next - start) & (ConvLeaf - 1)).Eq(0) | (next >= end), () =>
        {
            stack.Push((t - start) >> ConvLeafShift, [.. sums.Select(v => v.V)]);
            foreach (var v in sums)
            {
                v.V = k.Float(0f);
            }
        });
    }
}
