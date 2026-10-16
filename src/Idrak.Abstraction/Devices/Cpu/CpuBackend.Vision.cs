// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace Idrak.Abstraction.Devices.Cpu;

// Image operations on the CPU: depthwise convolutions (one input channel per group) and 1-D convolutions (one row, stride
// 1, one group: sound) as direct loops, average pooling and its gradient, and image resampling with per-channel
// normalization. Other convolutions take the composed path (patches and the tiled products, Backend.Convolution.cs).
internal sealed partial class CpuBackend
{
    // A depthwise convolution (one input channel per group, any number of filters per channel) runs as direct loops: the
    // patches of the composed path would be KH·KW times the input for products of KH·KW terms.
    private static bool Depthwise(in ConvGeometry g, int groups) => groups > 1 && g.C == groups;

    public override void ConvolutionKernel(Storage x, Storage weight, Storage? bias, Storage y, in ConvGeometry g, int filters, int groups, ConvActivation activation)
    {
        if (Direct1D(in g, groups))
        {
            DirectConvolution1D(D(x), D(weight), bias is null ? null : D(bias), D(y), in g, filters);
            if (ActivationOp(activation) is { } act)
            {
                Unary(act, y, y, g.N * filters * g.OW);
            }

            return;
        }

        if (!Depthwise(in g, groups))
        {
            base.ConvolutionKernel(x, weight, bias, y, in g, filters, groups, activation);
            return;
        }

        float[] xv = D(x), wv = D(weight), yv = D(y);
        float[]? bv = bias is null ? null : D(bias);
        var geo = g;
        int outputs = g.OH * g.OW;
        For(g.N * filters, (long)g.N * filters * outputs * g.KH * g.KW, (start, end) =>
        {
            for (int nf = start; nf < end; nf++)
            {
                int n = nf / filters, f = nf % filters, c = f / (filters / geo.C);
                var plane = yv.AsSpan(nf * outputs, outputs);
                DepthwisePlane(xv.AsSpan((n * geo.C + c) * geo.H * geo.W, geo.H * geo.W), wv.AsSpan(f * geo.KH * geo.KW, geo.KH * geo.KW), plane, geo,
                    bv is null ? 0f : bv[f]);
            }
        });
        if (ActivationOp(activation) is { } op)
        {
            Unary(op, y, y, g.N * filters * outputs);
        }
    }

    // A 1-D convolution (one row, stride 1, one group, any window, dilation and padding: a sound's channels over time)
    // runs directly: each task copies the input under one stretch of outputs (every channel, zeros outside the sound) and
    // sums its filters' taps over it, four filters and two vectors of outputs at a time. The composed path's patches are
    // the window's length times the input (11 taps over 32 channels of 57,344 samples: 80 MB a call, written and read
    // again) and its product has only as many columns as filters.
    private static bool Direct1D(in ConvGeometry g, int groups) =>
        groups == 1 && g.H == 1 && g.KH == 1 && g.PH == 0 && g.PadBottom == 0 && g.SW == 1 && g.OW > 0 && g.N > 0;

    // The floats of the input copy a task sums over (its channels' stretch), sized to stay in a core's cache.
    private const int DirectTileFloats = 1 << 16;

    // The register block: four filters by two vectors of outputs (8 accumulators; measured faster than 4 x 3 and than
    // 512-bit vectors on a 4-core AVX-512 x64).
    private const int DirectFilters = 4, DirectVectors = 2;

    private static void DirectConvolution1D(float[] x, float[] w, float[]? bias, float[] y, in ConvGeometry g, int filters)
    {
        int channels = g.C, taps = g.KW, outputs = g.OW, span = (taps - 1) * g.DW, step = DirectVectors * Vector<float>.Count;
        int tile = Math.Max(step, (DirectTileFloats / channels - span) / step * step);
        tile = Math.Min(tile, (outputs + step - 1) / step * step);
        int tiles = (outputs + tile - 1) / tile, blocks = (filters + DirectFilters - 1) / DirectFilters;

        // Split the filters too when the stretches alone do not give every core several tasks.
        int chunks = Math.Clamp((ComputeResources.MaxCpuThreads * 4 + g.N * tiles - 1) / (g.N * tiles), 1, blocks);
        int perChunk = (blocks + chunks - 1) / chunks;
        int row = tile + span;
        var geo = g;
        For(g.N * tiles * chunks, (long)g.N * filters * outputs * channels * taps, (start, end) =>
        {
            float[] input = ArrayPool<float>.Shared.Rent(channels * row);
            try
            {
                int filled = -1;
                for (int task = start; task < end; task++)
                {
                    int chunk = task % chunks, stretch = task / chunks, n = stretch / tiles, first = stretch % tiles * tile;
                    if (stretch != filled)
                    {
                        DirectInput(x, input, in geo, n, first - geo.PW, row);
                        filled = stretch;
                    }

                    int count = Math.Min(tile, outputs - first);
                    for (int b = chunk * perChunk, last = Math.Min(blocks, (chunk + 1) * perChunk); b < last; b++)
                    {
                        DirectFilterBlock(input, w, bias, y, in geo, filters, n, b * DirectFilters, first, count, row);
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(input);
            }
        });
    }

    // input[c, j] = x[n, c, from + j] for j in [0, row), 0 outside the sound.
    private static void DirectInput(float[] x, float[] input, in ConvGeometry g, int n, int from, int row)
    {
        int width = g.W;
        int lo = Math.Clamp(-from, 0, row), hi = Math.Clamp(width - from, lo, row);
        for (int c = 0; c < g.C; c++)
        {
            var target = input.AsSpan(c * row, row);
            target[..lo].Clear();
            if (hi > lo)
            {
                x.AsSpan((n * g.C + c) * width + from + lo, hi - lo).CopyTo(target[lo..hi]);
            }

            target[hi..].Clear();
        }
    }

    // Outputs [first, first + count) of up to four filters from f: bias + Σ over the channels, then the taps, of
    // weight · input, two vectors of outputs at a time (the last ones written as far as count).
    private static void DirectFilterBlock(float[] input, float[] w, float[]? bias, float[] y, in ConvGeometry g, int filters, int n, int f, int first, int count,
        int row)
    {
        int lanes = Vector<float>.Count, step = DirectVectors * lanes, channels = g.C, taps = g.KW, dilation = g.DW, outputs = g.OW, patch = channels * taps;
        int rows = Math.Min(DirectFilters, filters - f);
        ref float wr0 = ref w[f * patch];
        ref float wr1 = ref w[(f + Math.Min(1, rows - 1)) * patch];
        ref float wr2 = ref w[(f + Math.Min(2, rows - 1)) * patch];
        ref float wr3 = ref w[(f + Math.Min(3, rows - 1)) * patch];
        float b0 = bias?[f] ?? 0f, b1 = rows > 1 ? bias?[f + 1] ?? 0f : 0f, b2 = rows > 2 ? bias?[f + 2] ?? 0f : 0f, b3 = rows > 3 ? bias?[f + 3] ?? 0f : 0f;
        ref float source = ref MemoryMarshal.GetArrayDataReference(input);
        Span<float> spill = stackalloc float[step];
        for (int o = 0; o < count; o += step)
        {
            Vector<float> a00 = new(b0), a01 = a00, a10 = new(b1), a11 = a10, a20 = new(b2), a21 = a20, a30 = new(b3), a31 = a30;
            for (int c = 0; c < channels; c++)
            {
                int at = c * row + o, wc = c * taps;
                for (int k = 0; k < taps; k++, at += dilation)
                {
                    var x0 = Vector.LoadUnsafe(ref source, (nuint)at);
                    var x1 = Vector.LoadUnsafe(ref source, (nuint)(at + lanes));
                    var w0 = new Vector<float>(Unsafe.Add(ref wr0, wc + k));
                    var w1 = new Vector<float>(Unsafe.Add(ref wr1, wc + k));
                    var w2 = new Vector<float>(Unsafe.Add(ref wr2, wc + k));
                    var w3 = new Vector<float>(Unsafe.Add(ref wr3, wc + k));
                    a00 = Vector.FusedMultiplyAdd(x0, w0, a00);
                    a01 = Vector.FusedMultiplyAdd(x1, w0, a01);
                    a10 = Vector.FusedMultiplyAdd(x0, w1, a10);
                    a11 = Vector.FusedMultiplyAdd(x1, w1, a11);
                    a20 = Vector.FusedMultiplyAdd(x0, w2, a20);
                    a21 = Vector.FusedMultiplyAdd(x1, w2, a21);
                    a30 = Vector.FusedMultiplyAdd(x0, w3, a30);
                    a31 = Vector.FusedMultiplyAdd(x1, w3, a31);
                }
            }

            int length = Math.Min(step, count - o), outAt = (n * filters + f) * outputs + first + o;
            DirectStore(a00, a01, y.AsSpan(outAt, length), spill);
            if (rows > 1)
            {
                DirectStore(a10, a11, y.AsSpan(outAt + outputs, length), spill);
            }

            if (rows > 2)
            {
                DirectStore(a20, a21, y.AsSpan(outAt + 2 * outputs, length), spill);
            }

            if (rows > 3)
            {
                DirectStore(a30, a31, y.AsSpan(outAt + 3 * outputs, length), spill);
            }
        }
    }

    // Two vectors of outputs into target (as many as it holds).
    private static void DirectStore(Vector<float> v0, Vector<float> v1, Span<float> target, Span<float> spill)
    {
        int lanes = Vector<float>.Count;
        var into = target.Length == DirectVectors * lanes ? target : spill;
        v0.CopyTo(into);
        v1.CopyTo(into[lanes..]);
        if (target.Length != DirectVectors * lanes)
        {
            spill[..target.Length].CopyTo(target);
        }
    }

    // One output plane: bias + Σ over the window (rows, then columns) of weight · input.
    private static void DepthwisePlane(ReadOnlySpan<float> x, ReadOnlySpan<float> w, Span<float> y, in ConvGeometry g, float bias)
    {
        int h = g.H, wd = g.W, kh = g.KH, kw = g.KW, dh = g.DH, dw = g.DW;
        for (int oh = 0, o = 0; oh < g.OH; oh++)
        {
            int ih0 = oh * g.SH - g.PH;
            for (int ow = 0; ow < g.OW; ow++, o++)
            {
                int iw0 = ow * g.SW - g.PW;
                float sum = 0f;
                for (int i = 0; i < kh; i++)
                {
                    int ih = ih0 + i * dh;
                    if ((uint)ih >= (uint)h)
                    {
                        continue;
                    }

                    var row = x.Slice(ih * wd, wd);
                    for (int j = 0; j < kw; j++)
                    {
                        int iw = iw0 + j * dw;
                        if ((uint)iw < (uint)wd)
                        {
                            sum += w[i * kw + j] * row[iw];
                        }
                    }
                }

                y[o] = sum + bias;
            }
        }
    }

    public override void ConvolutionBackwardInputKernel(Storage dy, Storage weight, Storage dx, in ConvGeometry g, int filters, int groups)
    {
        if (!Depthwise(in g, groups))
        {
            base.ConvolutionBackwardInputKernel(dy, weight, dx, in g, filters, groups);
            return;
        }

        // Each input plane gathers from the planes of its channel's filters, in filter order, then window order: planes
        // write disjoint parts of dx.
        float[] gv = D(dy), wv = D(weight), dv = D(dx);
        var geo = g;
        int perChannel = filters / g.C, outputs = g.OH * g.OW, area = g.H * g.W;
        For(g.N * g.C, (long)g.N * filters * outputs * g.KH * g.KW, (start, end) =>
        {
            for (int nc = start; nc < end; nc++)
            {
                int n = nc / geo.C, c = nc % geo.C;
                var target = dv.AsSpan(nc * area, area);
                for (int m = 0; m < perChannel; m++)
                {
                    int f = c * perChannel + m;
                    var source = gv.AsSpan((n * filters + f) * outputs, outputs);
                    var w = wv.AsSpan(f * geo.KH * geo.KW, geo.KH * geo.KW);
                    for (int oh = 0, o = 0; oh < geo.OH; oh++)
                    {
                        int ih0 = oh * geo.SH - geo.PH;
                        for (int ow = 0; ow < geo.OW; ow++, o++)
                        {
                            int iw0 = ow * geo.SW - geo.PW;
                            float d = source[o];
                            for (int i = 0; i < geo.KH; i++)
                            {
                                int ih = ih0 + i * geo.DH;
                                if ((uint)ih >= (uint)geo.H)
                                {
                                    continue;
                                }

                                for (int j = 0; j < geo.KW; j++)
                                {
                                    int iw = iw0 + j * geo.DW;
                                    if ((uint)iw < (uint)geo.W)
                                    {
                                        target[ih * geo.W + iw] += w[i * geo.KW + j] * d;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        });
    }

    public override void ConvolutionBackwardWeightKernel(Storage x, Storage dy, Storage dweight, in ConvGeometry g, int filters, int groups)
    {
        if (!Depthwise(in g, groups))
        {
            base.ConvolutionBackwardWeightKernel(x, dy, dweight, in g, filters, groups);
            return;
        }

        // Each filter's KH·KW sums over the images and positions, in that order: filters write disjoint weights.
        float[] xv = D(x), gv = D(dy), wv = D(dweight);
        var geo = g;
        int perChannel = filters / g.C, outputs = g.OH * g.OW, area = g.H * g.W, taps = g.KH * g.KW;
        For(filters, (long)g.N * filters * outputs * taps, (start, end) =>
        {
            var buffer = ArrayPool<float>.Shared.Rent(taps);
            try
            {
                for (int f = start; f < end; f++)
                {
                    int c = f / perChannel;
                    var acc = buffer.AsSpan(0, taps);
                    acc.Clear();
                    for (int n = 0; n < geo.N; n++)
                    {
                        var plane = xv.AsSpan((n * geo.C + c) * area, area);
                        var source = gv.AsSpan((n * filters + f) * outputs, outputs);
                        for (int oh = 0, o = 0; oh < geo.OH; oh++)
                        {
                            int ih0 = oh * geo.SH - geo.PH;
                            for (int ow = 0; ow < geo.OW; ow++, o++)
                            {
                                int iw0 = ow * geo.SW - geo.PW;
                                float d = source[o];
                                for (int i = 0; i < geo.KH; i++)
                                {
                                    int ih = ih0 + i * geo.DH;
                                    if ((uint)ih >= (uint)geo.H)
                                    {
                                        continue;
                                    }

                                    for (int j = 0; j < geo.KW; j++)
                                    {
                                        int iw = iw0 + j * geo.DW;
                                        if ((uint)iw < (uint)geo.W)
                                        {
                                            acc[i * geo.KW + j] += d * plane[ih * geo.W + iw];
                                        }
                                    }
                                }
                            }
                        }
                    }

                    var target = wv.AsSpan(f * taps, taps);
                    for (int t = 0; t < taps; t++)
                    {
                        target[t] += acc[t];
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(buffer);
            }
        });
    }

    // ------------------------------------------------------------------ average pooling

    // The divisor of the window at (oh, ow): its rows and columns up to the padded end (H + padBottom, W + padRight), or the
    // input positions it covers (at least 1).
    private static float PoolDivisor(in ConvGeometry g, int oh, int ow, bool countIncludePad, int padBottom, int padRight)
    {
        int r0 = oh * g.SH - g.PH, c0 = ow * g.SW - g.PW;
        if (countIncludePad)
        {
            return (Math.Min(r0 + g.KH, g.H + padBottom) - r0) * (Math.Min(c0 + g.KW, g.W + padRight) - c0);
        }

        int rows = Math.Min(r0 + g.KH, g.H) - Math.Max(r0, 0), cols = Math.Min(c0 + g.KW, g.W) - Math.Max(c0, 0);
        return Math.Max(rows * cols, 1);
    }

    public override void AvgPoolKernel(Storage x, Storage y, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        float[] xv = D(x), yv = D(y);
        var geo = g;
        For(g.N * g.C, (long)g.N * g.C * g.OH * g.OW * g.KH * g.KW, (start, end) =>
        {
            for (int nc = start; nc < end; nc++)
            {
                var plane = xv.AsSpan(nc * geo.H * geo.W, geo.H * geo.W);
                int o = nc * geo.OH * geo.OW;
                for (int oh = 0; oh < geo.OH; oh++)
                {
                    int r0 = oh * geo.SH - geo.PH;
                    int rStart = Math.Max(r0, 0), rEnd = Math.Min(r0 + geo.KH, geo.H);
                    for (int ow = 0; ow < geo.OW; ow++, o++)
                    {
                        int c0 = ow * geo.SW - geo.PW;
                        int cStart = Math.Max(c0, 0), cEnd = Math.Min(c0 + geo.KW, geo.W);
                        float sum = 0f;
                        for (int r = rStart; r < rEnd; r++)
                        {
                            for (int c = cStart; c < cEnd; c++)
                            {
                                sum += plane[r * geo.W + c];
                            }
                        }

                        yv[o] = sum / PoolDivisor(geo, oh, ow, countIncludePad, padBottom, padRight);
                    }
                }
            }
        });
    }

    public override void AvgPoolBackwardKernel(Storage dy, Storage dx, in ConvGeometry g, bool countIncludePad, int padBottom, int padRight)
    {
        float[] gv = D(dy), dv = D(dx);
        var geo = g;
        For(g.N * g.C, (long)g.N * g.C * g.OH * g.OW * g.KH * g.KW, (start, end) =>
        {
            for (int nc = start; nc < end; nc++)
            {
                var plane = dv.AsSpan(nc * geo.H * geo.W, geo.H * geo.W);
                var source = gv.AsSpan(nc * geo.OH * geo.OW, geo.OH * geo.OW);
                for (int ih = 0; ih < geo.H; ih++)
                {
                    var (ohFirst, ohLast) = Covering(ih, geo.PH, geo.KH, geo.SH, geo.OH);
                    for (int iw = 0; iw < geo.W; iw++)
                    {
                        var (owFirst, owLast) = Covering(iw, geo.PW, geo.KW, geo.SW, geo.OW);
                        float sum = plane[ih * geo.W + iw];
                        for (int oh = ohFirst; oh < ohLast; oh++)
                        {
                            for (int ow = owFirst; ow < owLast; ow++)
                            {
                                sum += source[oh * geo.OW + ow] / PoolDivisor(geo, oh, ow, countIncludePad, padBottom, padRight);
                            }
                        }

                        plane[ih * geo.W + iw] = sum;
                    }
                }
            }
        });
    }

    // The windows [first, last) along one axis whose span covers input coordinate i (o·stride - pad ≤ i < o·stride - pad + size).
    private static (int First, int Last) Covering(int i, int pad, int size, int stride, int outputs)
    {
        int low = i + pad - size + 1;
        int first = low > 0 ? (low + stride - 1) / stride : 0;
        int last = Math.Min((i + pad) / stride + 1, outputs);
        return (first, last);
    }

    // ------------------------------------------------------------------ resampling and normalization

    public override void ResizeNormalizeKernel(Storage x, Storage coefficients, Storage values, Storage y, int planes, int channels, int height, int width,
        int outHeight, int outWidth, int xTaps, int yTaps, bool bytes)
    {
        float[] xv = D(x), cv = D(coefficients), vv = D(values), yv = D(y);
        int across = xTaps > 0 ? outWidth : width, rows = height;
        int yBase = xTaps > 0 ? outWidth * (2 + xTaps) : 0;
        For(planes, (long)planes * outHeight * outWidth * Math.Max(1, xTaps + yTaps), (start, end) =>
        {
            // The horizontal pass of one plane ([height, across]), then the vertical one into y, each output's taps in order.
            var middle = ArrayPool<float>.Shared.Rent(rows * across);
            try
            {
                for (int p = start; p < end; p++)
                {
                    int c = p % Math.Max(channels, 1);
                    long plane = (long)p * height * width;
                    for (int r = 0; r < rows; r++)
                    {
                        for (int ox = 0; ox < across; ox++)
                        {
                            float v;
                            if (xTaps == 0)
                            {
                                v = Pixel(xv, plane + (long)r * width + ox, bytes);
                            }
                            else
                            {
                                int at = ox * (2 + xTaps), first = Bits(cv[at]), count = Bits(cv[at + 1]);
                                v = bytes ? ByteSum(xv, plane + (long)r * width + first, 1, cv.AsSpan(at + 2, count))
                                          : FloatSum(xv, plane + (long)r * width + first, 1, cv.AsSpan(at + 2, count));
                            }

                            middle[r * across + ox] = v;
                        }
                    }

                    long o = (long)p * outHeight * outWidth;
                    for (int oy = 0; oy < outHeight; oy++)
                    {
                        for (int ox = 0; ox < outWidth; ox++, o++)
                        {
                            float v;
                            if (yTaps == 0)
                            {
                                v = middle[oy * across + ox];
                            }
                            else
                            {
                                int at = yBase + oy * (2 + yTaps), first = Bits(cv[at]), count = Bits(cv[at + 1]);
                                v = bytes ? ByteSum(middle, first * across + ox, across, cv.AsSpan(at + 2, count), stored: true)
                                          : FloatSum(middle, first * across + ox, across, cv.AsSpan(at + 2, count));
                            }

                            yv[o] = bytes ? vv[256 * c + (int)v] : v * vv[2 * c] + vv[2 * c + 1];
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(middle);
            }
        });

        static int Bits(float v) => BitConverter.SingleToInt32Bits(v);

        // An input value: a float, or byte i of the packed words (as a float holding 0 ... 255).
        static float Pixel(float[] x, long i, bool bytes) =>
            bytes ? (BitConverter.SingleToUInt32Bits(x[i >> 2]) >> (int)(8 * (i & 3))) & 0xFF : x[i];

        // Σ value · weight over the taps, in order.
        static float FloatSum(float[] x, long first, int step, ReadOnlySpan<float> weights)
        {
            float sum = 0f;
            for (int t = 0; t < weights.Length; t++)
            {
                sum += x[first + (long)t * step] * weights[t];
            }

            return sum;
        }

        // Pillow's 8-bit pass: 2^21 + Σ byte · weight (22 fractional bits) as integers, shifted right by 22, clipped to
        // [0, 255]. `stored`: the bytes are the first pass's results (floats holding 0 ... 255), else packed input bytes.
        static float ByteSum(float[] x, long first, int step, ReadOnlySpan<float> weights, bool stored = false)
        {
            int sum = 1 << 21;
            for (int t = 0; t < weights.Length; t++)
            {
                long i = first + (long)t * step;
                int value = stored ? (int)x[i] : (int)((BitConverter.SingleToUInt32Bits(x[i >> 2]) >> (int)(8 * (i & 3))) & 0xFF);
                sum += value * BitConverter.SingleToInt32Bits(weights[t]);
            }

            return sum >= 1 << 30 ? 255 : sum <= 0 ? 0 : sum >> 22;
        }
    }
}
