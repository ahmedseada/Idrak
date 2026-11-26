// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;

namespace Idrak.Abstraction.Devices.Cpu;

// Image resampling with each channel's normalization in one pass (Backend.ResizeNormalizeKernel): a plane a row of the
// row driver (one thread's), its horizontal pass a float at a time (each output column its own taps), its vertical pass
// over the lanes along the row (every column of an output row takes the same taps), each output's taps in order, fused
// from zero, so every width gives the same bits. Bytes: Pillow's integer passes, a value at a time.
internal sealed unsafe partial class CpuBackend
{
    public override void ResizeNormalizeKernel(Storage x, Storage coefficients, Storage values, Storage y, int planes, int channels, int height, int width,
        int outHeight, int outWidth, int xTaps, int yTaps, bool bytes)
    {
        if (planes <= 0 || outHeight <= 0 || outWidth <= 0)
        {
            return;
        }

        fixed (float* px = V(x), pc = V(coefficients), pv = V(values), py = V(y))
        {
            var kernel = new ResizeRows(px, pc, pv, py, Math.Max(1, channels), height, width, outHeight, outWidth, xTaps, yTaps, bytes, planes);
            Rows(bytes ? "ResizeNormalize bytes" : "ResizeNormalize", kernel, planes, outHeight * outWidth, Math.Max(1, xTaps + yTaps));
        }
    }

    private struct ResizeRows(float* x, float* coefficients, float* values, float* y, int channels, int height, int width, int outHeight, int outWidth, int xTaps,
        int yTaps, bool bytes, int planes) : IRowKernel
    {
        private float* _y = y;

        public readonly void Row<TV, TF>(int p) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            int c = p % channels, across = xTaps > 0 ? outWidth : width;
            long plane = (long)p * height * width;
            float* target = _y + (long)p * outHeight * outWidth;
            // Floats without a horizontal pass: the plane itself is the vertical pass's input.
            bool inPlace = xTaps == 0 && !bytes;
            var rented = ArrayPool<float>.Shared.Rent(inPlace ? 1 : Math.Max(1, height * across));
            try
            {
                fixed (float* buffer = rented)
                {
                    float* middle = inPlace ? x + plane : buffer;
                    if (!inPlace)
                    {
                        Across(plane, across, middle);
                    }

                    float* rowsAt = coefficients + (xTaps > 0 ? (long)outWidth * (2 + xTaps) : 0);
                    for (int oy = 0; oy < outHeight; oy++)
                    {
                        float* output = target + (long)oy * outWidth;
                        int first = oy, taps = 1;
                        float* weights = null;
                        if (yTaps > 0)
                        {
                            float* at = rowsAt + (long)oy * (2 + yTaps);
                            first = Whole(at[0]);
                            taps = Whole(at[1]);
                            weights = at + 2;
                        }

                        if (bytes)
                        {
                            for (int ox = 0; ox < outWidth; ox++)
                            {
                                int v = weights == null ? (int)middle[(long)oy * across + ox] : ByteSum(middle, (long)first * across + ox, across, weights, taps);
                                output[ox] = values[256 * c + v];
                            }
                        }
                        else
                        {
                            var map = new ResizeDownMap(middle + (long)first * across, across, weights, taps, output, values[2 * c], values[2 * c + 1]);
                            RowMap<TV, TF, ResizeDownMap>(ref map, outWidth);
                        }
                    }
                }
            }
            finally
            {
                ArrayPool<float>.Shared.Return(rented);
            }
        }

        // The horizontal pass of the plane at `plane` into middle [height, across] (no taps: the values as they are).
        private readonly void Across(long plane, int across, float* middle)
        {
            for (int r = 0; r < height; r++)
            {
                long row = plane + (long)r * width;
                float* into = middle + (long)r * across;
                for (int ox = 0; ox < across; ox++)
                {
                    if (xTaps == 0)
                    {
                        into[ox] = bytes ? Byte(row + ox) : x[row + ox];
                        continue;
                    }

                    float* at = coefficients + (long)ox * (2 + xTaps);
                    int first = Whole(at[0]), taps = Whole(at[1]);
                    if (bytes)
                    {
                        int sum = 1 << 21;
                        for (int t = 0; t < taps; t++)
                        {
                            sum += Byte(row + first + t) * Whole(at[2 + t]);
                        }

                        into[ox] = Clip(sum);
                    }
                    else
                    {
                        float sum = 0f;
                        for (int t = 0; t < taps; t++)
                        {
                            sum = MathF.FusedMultiplyAdd(x[row + first + t], at[2 + t], sum);
                        }

                        into[ox] = sum;
                    }
                }
            }
        }

        // Byte i of the packed words (value i in bits 8·(i % 4) of word i / 4).
        private readonly int Byte(long i) => (int)((BitConverter.SingleToUInt32Bits(x[i >> 2]) >> (int)(8 * (i & 3))) & 0xFF);

        public void Redirect(ref Scratch scratch) => _y = scratch.Take((long)planes * outHeight * outWidth);
    }

    // An int held in a float's bits.
    private static int Whole(float v) => BitConverter.SingleToInt32Bits(v);

    // Pillow's rounding of a pass: the sum (2^21 added) shifted right by 22, clipped to [0, 255].
    private static int Clip(int sum) => sum >= 1 << 30 ? 255 : sum <= 0 ? 0 : sum >> 22;

    // A vertical byte pass over the first pass's results (whole values 0 ... 255 held as floats).
    private static int ByteSum(float* middle, long first, int step, float* weights, int taps)
    {
        int sum = 1 << 21;
        for (int t = 0; t < taps; t++)
        {
            sum += (int)middle[first + (long)t * step] * Whole(weights[t]);
        }

        return Clip(sum);
    }

    // An output row of the vertical pass: Σ_t w_t · middle[first + t, ox] in tap order (fused from zero; no taps: the row
    // as it is), then value · scale + shift (fused).
    private struct ResizeDownMap(float* rows, int across, float* weights, int taps, float* output, float scale, float shift) : IRowMap
    {
        public readonly void Step<TV, TF>(nint ox) where TV : unmanaged where TF : struct, IFloats<TV>
        {
            TV v;
            if (weights == null)
            {
                v = TF.Load(rows + ox);
            }
            else
            {
                v = TF.All(0f);
                for (int t = 0; t < taps; t++)
                {
                    v = TF.Fma(TF.Load(rows + (long)t * across + ox), TF.All(weights[t]), v);
                }
            }

            TF.Store(output + ox, TF.Fma(v, TF.All(scale), TF.All(shift)));
        }
    }
}
