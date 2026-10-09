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
//   backward input   batch (n, q); rows: the group's channels c; columns: input positions (ih, iw); sum over (f, kh, kw) of
//                    weight[q·Fg + f, (c, kh, kw)] · dy[n, q·Fg + f, oh, ow] where ih = oh·SH - PH + kh·DH (and so for
//                    the width) has a whole oh within [0, OH); added to dx.
//   backward weight  batch (q, split s); rows: the group's filters f; columns: (c, kh, kw); sum over the positions (n, oh,
//                    ow) of split s of dy[n, q·Fg + f, oh, ow] · x[n, q·Cg + c, ih, iw]: added to dweight with one
//                    split, else written to part[s] and added split by split, in order (conv_split_reduce).
//
// Every output adds its terms in the order of the sum, so a shape gives the same bits run after run. Push constants of
// every convolution kernel: N, C, H, W, KH, KW, SH, SW, PH, PW, OH, OW, DH, DW, F, G, flags, splits (flags: 1 bias,
// activation · 2 for the forward kernels).
internal static partial class VulkanKernels
{
    /// <summary>The variants of the implicit convolution products (the backend's tile choice).</summary>
    public static readonly string[] ConvolutionPasses = ["conv_forward", "conv_backward_input", "conv_backward_weight"];

    private static IEnumerable<(string, Func<SpirvKernel>)> ConvolutionKernels()
    {
        foreach (var (name, spec) in new (string, Func<ConvPass>)[]
        {
            ("conv_forward", () => new ForwardPass()),
            ("conv_backward_input", () => new InputPass()),
            ("conv_backward_weight", () => new WeightPass()),
        })
        {
            yield return (name, () => ConvGemmBlocked(name, spec()));
            yield return (name + "_tile", () => ConvGemmTile(name + "_tile", spec()));
        }

        // dweight[i] += Σ_s part[s · count + i], splits in order.
        yield return ("conv_split_reduce", () =>
        {
            var k = new KernelBuilder("conv_split_reduce", Block);
            var (part, dw) = (k.Buffer("part"), k.Buffer("dw"));
            var (count, splits) = (k.PushInt("count"), k.PushInt("splits"));
            Grid(k, count, i =>
            {
                var acc = k.Local(dw[i]);
                k.For(k.Int(0), splits, 1, s => acc.V = acc.V + part[s * count + i]);
                dw[i] = acc.V;
            });
            return k.Build();
        });

        // Depthwise (one input channel per group, M = F / C filters a channel): an invocation per output adds its window,
        // rows then columns, to the bias and applies the activation.
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
                            k.If((iw >= 0) & (iw < g.W), () => acc.V = acc.V + w[(f * g.KH + i) * g.KW + j] * x[row + iw]);
                        });
                    });
                });
                var v = acc.V + k.Select((g.Flags & 1).Ne(0), bias[f], k.Float(0f));
                y[o] = Activated(k, v, g.Flags >> 1);
            });
            return k.Build();
        });

        // dx[n, c, ih, iw] += Σ over the channel's filters, in order, and the windows covering the element, in window
        // order, of weight · dy: an invocation per input element.
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
                k.For(k.Int(0), perChannel, 1, m =>
                {
                    var f = c * perChannel + m;
                    var plane = (n * g.F + f) * g.OH;
                    k.For(k.Int(0), g.KH, 1, i =>
                    {
                        var th = ih + g.PH - i * g.DH;
                        var oh = th / g.SH;
                        k.If((th >= 0) & (th % g.SH).Eq(0) & (oh < g.OH), () =>
                        {
                            k.For(k.Int(0), g.KW, 1, j =>
                            {
                                var tw = iw + g.PW - j * g.DW;
                                var ow = tw / g.SW;
                                k.If((tw >= 0) & (tw % g.SW).Eq(0) & (ow < g.OW), () =>
                                    acc.V = acc.V + w[(f * g.KH + i) * g.KW + j] * dy[(plane + oh) * g.OW + ow]);
                            });
                        });
                    });
                });
                dx[idx] = acc.V;
            });
            return k.Build();
        });

        // dweight[f, i, j] += Σ_{n, oh, ow} dy[n, f, oh, ow] · x[n, f / M, ih, iw]: a workgroup per weight, its invocations
        // over the positions (a fixed stride), then the workgroup's sum (a fixed tree), so the bits do not change.
        yield return ("conv_depthwise_backward_weight", () =>
        {
            var k = new KernelBuilder("conv_depthwise_backward_weight", Block);
            var (x, dy, dw) = (k.Buffer("x"), k.Buffer("dy"), k.Buffer("dw"));
            var g = ConvPushes(k);
            var scratch = k.Shared("scratch", Block);
            EachRow(k, g.F * g.KH * g.KW, e =>
            {
                var j = e % g.KW;
                var t = e / g.KW;
                var i = t % g.KH;
                var f = t / g.KH;
                var c = f / (g.F / g.C);
                var positions = g.N * g.OH * g.OW;
                var acc = k.Local(0f);
                k.For(k.LocalX, positions, Block, p =>
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
                var sum = k.ReduceSum(scratch, acc.V);
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
        Val DH, Val DW, Val F, Val G, Val Flags, Val Splits);

    private static ConvGeometryPush ConvPushes(KernelBuilder k) => new(
        k.PushInt("N"), k.PushInt("C"), k.PushInt("H"), k.PushInt("W"), k.PushInt("KH"), k.PushInt("KW"), k.PushInt("SH"), k.PushInt("SW"),
        k.PushInt("PH"), k.PushInt("PW"), k.PushInt("OH"), k.PushInt("OW"), k.PushInt("DH"), k.PushInt("DW"), k.PushInt("F"), k.PushInt("G"),
        k.PushInt("flags"), k.PushInt("splits"));

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

    // dx += the transposed convolution of dy.
    private sealed class InputPass : ConvPass
    {
        private Buf _dy = null!, _w = null!, _dx = null!;
        private Val _n, _q, _fg, _cg, _patch;

        protected override void Buffers(KernelBuilder k) => (_dy, _w, _dx) = (k.Buffer("dy"), k.Buffer("w"), k.Buffer("dx"));

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

        public override Val SumEnd => _fg * G.KH * G.KW;

        public override Val A(Val row, Val kk)
        {
            var area = G.KH * G.KW;
            var (f, r) = (kk / area, kk % area);
            return _w[(_q * _fg + f) * _patch + row * area + r];
        }

        public override (Val, Func<Val>) B(Val kk, Val column)
        {
            var area = G.KH * G.KW;
            var (f, r) = (kk / area, kk % area);
            var (kh, kw) = (r / G.KW, r % G.KW);
            var (ih, iw) = (column / G.W, column % G.W);
            var th = ih + G.PH - kh * G.DH;
            var tw = iw + G.PW - kw * G.DW;
            var (oh, ow) = (th / G.SH, tw / G.SW);
            var inside = (th >= 0) & (tw >= 0) & (th % G.SH).Eq(0) & (tw % G.SW).Eq(0) & (oh < G.OH) & (ow < G.OW);
            return (inside, () => _dy[((_n * G.F + _q * _fg + f) * G.OH + oh) * G.OW + ow]);
        }

        public override void Store(Val row, Val column, Val value)
        {
            var o = (_n * G.C + _q * _cg + row) * (G.H * G.W) + column;
            _dx[o] = _dx[o] + value;
        }
    }

    // dweight += dyᵀ · patches, the positions in splits.
    private sealed class WeightPass : ConvPass
    {
        private Buf _x = null!, _dy = null!, _dw = null!, _part = null!;
        private Val _q, _s, _fg, _cg, _patch, _start, _end;

        protected override void Buffers(KernelBuilder k) => (_x, _dy, _dw, _part) = (k.Buffer("x"), k.Buffer("dy"), k.Buffer("dw"), k.Buffer("part"));

        public override Val Batch => G.G * G.Splits;

        public override void Begin(Val bi)
        {
            (_q, _s) = (bi / G.Splits, bi % G.Splits);
            (_fg, _cg) = (G.F / G.G, G.C / G.G);
            _patch = _cg * G.KH * G.KW;
            var total = G.N * G.OH * G.OW;
            var chunk = (total + G.Splits - 1) / G.Splits;
            chunk = (chunk + (MatDepth - 1)) / MatDepth * MatDepth;               // whole steps of the sum
            _start = K.Min(_s * chunk, total);
            _end = K.Min(_start + chunk, total);
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
            var (rows, columns, end) = (pass.Rows, pass.Columns, pass.SumEnd);
            foreach (var v in acc)
            {
                v.V = k.Float(0f);
            }

            k.For(pass.SumStart, end, MatDepth, t =>
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
                }

                for (int i = 0; i < acc.Length; i++)
                {
                    acc[i].V = sums[i];
                }

                k.Barrier();
            });

            for (int i = 0; i < Per; i++)
            {
                for (int j = 0; j < Per; j++)
                {
                    var (row, col, value) = (rowBase + ty + side * i, colBase + tx + side * j, acc[i * Per + j]);
                    k.If((row < rows) & (col < columns), () => pass.Store(row, col, value.V));
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
        k.For(k.GroupZ, pass.Batch, bi =>
        {
            pass.Begin(bi);
            var (rows, columns, end) = (pass.Rows, pass.Columns, pass.SumEnd);
            var acc = k.Local(0f);
            k.For(pass.SumStart, end, side, t =>
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
                var sum = acc.V;
                for (int e = 0; e < side; e++)
                {
                    sum = k.Fma(tileA[ty * side + e], tileB[k.Int(e * side) + tx], sum);
                }

                acc.V = sum;
                k.Barrier();
            });
            k.If((row < rows) & (col < columns), () => pass.Store(row, col, acc.V));
        }, k.GroupsZ);
        return k.Build();
    }
}
