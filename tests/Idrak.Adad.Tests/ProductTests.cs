// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Adad.Abstractions;
using Idrak.Adad.Devices.Cuda;
using Idrak.Adad.Types;
using static Idrak.Adad.Devices.Cuda.NativeKernels;

namespace Idrak.Adad.Tests;

internal static class ProductTests
{
    private const int Control = 0, A = 1, B = 2;

    // A shape's operands on the device: x, W as stored, y.
    private sealed class Operands : IDisposable
    {
        private readonly CudaDevice _device;

        public Operands(CudaDevice device, float[] x, byte[] w, int m, int n, int k)
        {
            _device = device;
            X = device.Allocate(x.LongLength * 4);
            W = device.Allocate(w.LongLength);
            Y = device.Allocate((long)m * n * 4);
            device.Upload(X, x.AsSpan());
            device.Upload(W, w.AsSpan());
            Op = new ProductOperands(X.Pointer, W.Pointer, Y.Pointer, m, n, k);
        }

        public DeviceBuffer X { get; }

        public DeviceBuffer W { get; }

        public DeviceBuffer Y { get; }

        public ProductOperands Op { get; }

        public float[] Run(SuperLane lane, int variant)
        {
            _device.Fill(Y, 0x7FC00000, (long)Op.M * Op.N);                // NaN: a value not written shows
            lane.Run(variant, Op);
            var y = new float[Op.M * Op.N];
            _device.Download(Y, y.AsSpan());
            return y;
        }

        public void Dispose()
        {
            _device.Release(X);
            _device.Release(W);
            _device.Release(Y);
        }
    }

    private static SuperLane? LaneOf(Checks t, int lane, out Gpu? gpu)
    {
        gpu = Gpu.Get(out string reason);
        if (gpu is null)
        {
            t.Check(false, $"no CUDA device: {reason}");
            return null;
        }

        var l = gpu.Product.Lane(SuperBlocks.Q4K, lane, out string why);
        t.Check(l is not null, $"lane {SuperProduct.LaneNames[lane]} is no candidate: {why}");
        return l;
    }

    // The lane's variant of shape `config` (by value) and `splits`, or −1 when the lane does not offer it at this shape.
    private static int VariantOf(SuperLane lane, NativeMmaConfig config, int splits, int m, int n, int k) =>
        lane.Variants(m, n, k, out _).Where(v => lane.Describe(v) == (config, splits)).DefaultIfEmpty(-1).First();

    private static (float[] X, byte[] W) ShapeData(int m, int n, int k) => (Data.RandomRows(m, k, 1000 + k + m), Data.RandomBlocks(SuperBlocks.Q4K, n, k, n * 31 + k));

    private static int FirstDifference(float[] a, float[] b)
    {
        for (int i = 0; i < a.Length; i++)
        {
            if (BitConverter.SingleToInt32Bits(a[i]) != BitConverter.SingleToInt32Bits(b[i]))
            {
                return i;
            }
        }

        return -1;
    }

    public static void ControlMatchesTheSource(Checks t)
    {
        var control = LaneOf(t, Control, out var gpu);
        if (control is null)
        {
            return;
        }

        foreach (var (m, n, k) in Data.Shapes)
        {
            var (x, w) = ShapeData(m, n, k);
            using var op = new Operands(gpu!.Device, x, w, m, n, k);
            var first = Recorded.ControlFirstShape.Single(r => (r.M, r.N, r.K) == (m, n, k));
            int v = VariantOf(control, NativeMmaConfigs[0], 1, m, n, k);
            t.Check(v >= 0, $"{m}x{n}x{k}: the control offers no {NativeMmaConfigs[0]} unsplit");
            if (v >= 0)
            {
                string sha = Data.Sha(op.Run(control, v));
                t.Check(sha == first.Sha, $"{m}x{n}x{k}, {NativeMmaConfigs[0]} unsplit: y {sha}, the source's {first.Sha}");
            }

            var measured = Recorded.ControlMeasured.Single(r => (r.M, r.N, r.K) == (m, n, k));
            v = VariantOf(control, NativeMmaConfigs[measured.Shape], measured.Splits, m, n, k);
            t.Check(v >= 0, $"{m}x{n}x{k}: the control offers no {NativeMmaConfigs[measured.Shape]} in {measured.Splits} splits (the source's measured choice)");
            if (v >= 0)
            {
                string sha = Data.Sha(op.Run(control, v));
                t.Check(sha == measured.Sha, $"{m}x{n}x{k}, {NativeMmaConfigs[measured.Shape]} in {measured.Splits} splits: y {sha}, the source's {measured.Sha}");
            }
        }
    }

    public static void DecodeMatchesControl(Checks t)
    {
        var control = LaneOf(t, Control, out var gpu);
        var decode = LaneOf(t, A, out _);
        if (control is null || decode is null)
        {
            return;
        }

        int compared = 0;
        foreach (var (m, n, k) in Data.Shapes)
        {
            var (x, w) = ShapeData(m, n, k);
            using var op = new Operands(gpu!.Device, x, w, m, n, k);
            var reference = new Dictionary<(int RowBytes, int Splits), float[]>();
            foreach (int v in decode.Variants(m, n, k, out _))
            {
                var (cfg, splits) = decode.Describe(v);
                if (!reference.TryGetValue((cfg.RowBytes, splits), out var expected))
                {
                    // The control with the same k blocks (row bytes) and splits: the same split boundaries.
                    var same = NativeMmaConfigs.First(c => c.RowBytes == cfg.RowBytes);
                    int cv = VariantOf(control, same, splits, m, n, k);
                    t.Check(cv >= 0, $"{m}x{n}x{k}: the control offers no {same} in {splits} splits");
                    if (cv < 0)
                    {
                        continue;
                    }

                    expected = op.Run(control, cv);
                    reference[(cfg.RowBytes, splits)] = expected;
                }

                var y = op.Run(decode, v);
                int at = FirstDifference(y, expected);
                compared++;
                t.Check(at < 0, at < 0 ? "" : string.Create(CultureInfo.InvariantCulture,
                    $"{m}x{n}x{k}, A {cfg} in {splits} splits: y[{at / n}, {at % n}] = {y[at]:R}, the control's {expected[at]:R}"));
            }
        }

        t.Check(compared > 0, "no A variant ran");
        Console.Write($"({compared} A variants against the control) ");
    }

    // ------------------------------------------------------------------ B's CPU model

    // x rounded as B's kernel rounds it: a 32-value block's d = max |x| / 127, q = round-to-even(x · (1 / d)) within ±127
    // (0 where d is 0), the block's pair (d, d · Σq).
    private static (sbyte[] Q, float[] D, float[] Sum) QuantizeModel(float[] x, int m, int k)
    {
        var q = new sbyte[x.Length];
        var d = new float[m * (k / 32)];
        var sum = new float[d.Length];
        for (int b = 0; b < d.Length; b++)
        {
            float amax = 0f;
            for (int i = 0; i < 32; i++)
            {
                amax = MathF.Max(amax, MathF.Abs(x[b * 32 + i]));
            }

            float scale = amax / 127f, inv = scale == 0f ? 0f : 1f / scale;
            int total = 0;
            for (int i = 0; i < 32; i++)
            {
                int v = Math.Clamp((int)MathF.Round(x[b * 32 + i] * inv), -127, 127);
                q[b * 32 + i] = (sbyte)v;
                total += v;
            }

            (d[b], sum[b]) = (scale, scale * total);
        }

        return (q, d, sum);
    }

    // W's codes, and each row's sub-blocks' scale and minimum, by the CPU decoder's per-value form.
    private static (byte[] Codes, float[] Scale, float[] Min) WeightModel(byte[] w, int n, int k)
    {
        var s = SuperBlocks.Q4K;
        var codes = new byte[n * k];
        var scale = new float[n * (k / 32)];
        var min = new float[scale.Length];
        Parallel.For(0, n, j =>
        {
            for (int blk = 0; blk < k / s.Values; blk++)
            {
                var block = w.AsSpan((j * (k / s.Values) + blk) * s.Bytes, s.Bytes);
                for (int sub = 0; sub < s.Values / 32; sub++)
                {
                    (scale[j * (k / 32) + blk * 8 + sub], min[j * (k / 32) + blk * 8 + sub]) = s.SubScale(block, sub);
                }

                for (int v = 0; v < s.Values; v++)
                {
                    codes[j * k + blk * s.Values + v] = (byte)s.CodeAt(block, v);
                }
            }
        });
        return (codes, scale, min);
    }

    // B's arithmetic on the CPU, rows `rows`: per split (its sub-blocks of k in order) C = 0, then per sub-block an int32
    // sum and C = fma(dsc · xs, isum, C), C = fma(−dm, xsum, C); the splits added in order.
    private static float[] Int8Model((sbyte[] Q, float[] D, float[] Sum) xq, (byte[] Codes, float[] Scale, float[] Min) wm, int[] rows, int n, int k, int stageValues, int splits)
    {
        int subs = k / 32, blocks = k / stageValues, per = (blocks + splits - 1) / splits, perSubs = per * stageValues / 32;
        var y = new float[rows.Length * n];
        Parallel.For(0, n, j =>
        {
            for (int r = 0; r < rows.Length; r++)
            {
                int i = rows[r];
                float total = 0f;
                for (int z = 0; z < splits; z++)
                {
                    float acc = 0f;
                    for (int b = z * perSubs; b < Math.Min((z + 1) * perSubs, subs); b++)
                    {
                        int isum = 0;
                        for (int e = 0; e < 32; e++)
                        {
                            isum += xq.Q[i * k + b * 32 + e] * wm.Codes[j * k + b * 32 + e];
                        }

                        float factor = wm.Scale[j * subs + b] * xq.D[i * subs + b];
                        acc = MathF.FusedMultiplyAdd(factor, isum, acc);
                        acc = MathF.FusedMultiplyAdd(-wm.Min[j * subs + b], xq.Sum[i * subs + b], acc);
                    }

                    total = z == 0 ? acc : total + acc;
                }

                y[r * n + j] = total;
            }
        });
        return y;
    }

    // The rows a model is checked on: all of them for a small product, else the first and last of each 64 and the middle.
    private static int[] CheckedRows(int m, int n, int k) =>
        (long)m * n * k <= 1L << 28 ? [.. Enumerable.Range(0, m)] : [.. new[] { 0, 1, 63, 64, 127, m / 2, m - 2, m - 1 }.Where(r => r < m).Distinct()];

    public static void QuantizeMatchesModel(Checks t)
    {
        var lane = LaneOf(t, B, out var gpu);
        if (lane is null)
        {
            return;
        }

        foreach (var (m, n, k) in Data.Shapes)
        {
            var (x, w) = ShapeData(m, n, k);
            Array.Clear(x, 0, Math.Min(64, x.Length));                    // a block of zeros: d = 0
            x[100 % x.Length] = 1e-30f;
            using var op = new Operands(gpu!.Device, x, w, m, n, k);
            int v = lane.Variants(m, n, k, out _)[0];
            var scratch = lane.Rent(v, op.Op);
            try
            {
                lane.Launch(0, v, op.Op, scratch);
                var q = new sbyte[m * k];
                var pairs = new float[m * (k / 32) * 2];
                gpu.Device.Download(scratch.A, q.AsSpan());
                gpu.Device.Download(scratch.B, pairs.AsSpan());
                var model = QuantizeModel(x, m, k);
                int codes = Enumerable.Range(0, q.Length).FirstOrDefault(i => q[i] != model.Q[i], -1);
                t.Check(codes < 0, codes < 0 ? "" : $"{m}x{k}: x[{codes / k}, {codes % k}] rounds to {q[codes]}, the model's {model.Q[codes]}");
                int pair = Enumerable.Range(0, model.D.Length).FirstOrDefault(b => BitConverter.SingleToInt32Bits(pairs[2 * b]) != BitConverter.SingleToInt32Bits(model.D[b])
                    || BitConverter.SingleToInt32Bits(pairs[2 * b + 1]) != BitConverter.SingleToInt32Bits(model.Sum[b]), -1);
                t.Check(pair < 0, pair < 0 ? "" : $"{m}x{k}: block {pair}'s pair ({pairs[2 * pair]:R}, {pairs[2 * pair + 1]:R}), the model's ({model.D[pair]:R}, {model.Sum[pair]:R})");
            }
            finally
            {
                gpu.Device.Synchronize();
                lane.Return(scratch);
            }
        }
    }

    public static void Int8MatchesModel(Checks t)
    {
        var lane = LaneOf(t, B, out var gpu);
        if (lane is null)
        {
            return;
        }

        int compared = 0;
        double worstOfBound = 0, worstRelative = 0;
        foreach (var (m, n, k) in Data.Shapes)
        {
            var (x, w) = ShapeData(m, n, k);
            using var op = new Operands(gpu!.Device, x, w, m, n, k);
            var xq = QuantizeModel(x, m, k);
            var wm = WeightModel(w, n, k);
            int[] rows = CheckedRows(m, n, k);
            var models = new Dictionary<(int StageValues, int Splits), float[]>();

            // The float reference (double sums over the decoded weight) and the stated bound, a value of y each.
            var decoded = new float[n * k];
            SuperBlocks.Q4K.Decode(w, decoded);
            var reference = new double[rows.Length * n];
            var bound = new double[rows.Length * n];
            var magnitude = new double[rows.Length * n];
            Parallel.For(0, n, j =>
            {
                int subs = k / 32;
                for (int r = 0; r < rows.Length; r++)
                {
                    int i = rows[r];
                    double sum = 0, rounding = 0, widening = 0, steps = 0, size = 0;
                    for (int b = 0; b < subs; b++)
                    {
                        int isum = 0;
                        for (int e = 0; e < 32; e++)
                        {
                            int at = b * 32 + e;
                            double xv = x[i * k + at], wv = decoded[j * k + at], xr = xq.D[i * subs + b] * (double)xq.Q[i * k + at];
                            sum += xv * wv;
                            size += Math.Abs(xv * wv);
                            rounding += Math.Abs(wv) * Math.Abs(xv - xr);
                            widening += Math.Abs(xr) * Math.Pow(2, -23) * (Math.Abs(wm.Scale[j * subs + b] * (double)wm.Codes[j * k + at]) + Math.Abs(wm.Min[j * subs + b]));
                            isum += xq.Q[i * k + at] * wm.Codes[j * k + at];
                        }

                        steps += Math.Abs((double)wm.Scale[j * subs + b] * xq.D[i * subs + b] * isum) + Math.Abs((double)wm.Min[j * subs + b] * xq.Sum[i * subs + b]);
                    }

                    reference[r * n + j] = sum;
                    magnitude[r * n + j] = size;
                    bound[r * n + j] = rounding + widening + (subs + 4) * Math.Pow(2, -23) * steps + 1e-30;
                }
            });

            foreach (int v in lane.Variants(m, n, k, out _))
            {
                var (cfg, splits) = lane.Describe(v);
                int stageValues = SuperMmaKernels.Int8StageValues(cfg);
                if (!models.TryGetValue((stageValues, splits), out var model))
                {
                    model = Int8Model(xq, wm, rows, n, k, stageValues, splits);
                    models[(stageValues, splits)] = model;
                }

                var y = op.Run(lane, v);
                compared++;
                int bad = -1, outside = -1;
                for (int r = 0; r < rows.Length && (bad < 0 || outside < 0); r++)
                {
                    for (int j = 0; j < n; j++)
                    {
                        float got = y[rows[r] * n + j];
                        if (bad < 0 && BitConverter.SingleToInt32Bits(got) != BitConverter.SingleToInt32Bits(model[r * n + j]))
                        {
                            bad = r * n + j;
                        }

                        double error = Math.Abs(got - reference[r * n + j]);
                        worstOfBound = Math.Max(worstOfBound, error / bound[r * n + j]);
                        worstRelative = Math.Max(worstRelative, error / Math.Max(magnitude[r * n + j], 1e-30));
                        if (outside < 0 && !(error <= bound[r * n + j]))
                        {
                            outside = r * n + j;
                        }
                    }
                }

                t.Check(bad < 0, bad < 0 ? "" : string.Create(CultureInfo.InvariantCulture,
                    $"{m}x{n}x{k}, B {cfg} in {splits} splits: y[{rows[bad / n]}, {bad % n}] = {y[rows[bad / n] * n + bad % n]:R}, its CPU model's {model[bad]:R}"));
                t.Check(outside < 0, outside < 0 ? "" : string.Create(CultureInfo.InvariantCulture,
                    $"{m}x{n}x{k}, B {cfg} in {splits} splits: y[{rows[outside / n]}, {outside % n}] = {y[rows[outside / n] * n + outside % n]:R}, the reference {reference[outside]:R}, past the bound {bound[outside]:E3}"));
            }
        }

        Console.Write(string.Create(CultureInfo.InvariantCulture, $"({compared} B variants; the largest error {worstOfBound:F3} of the bound, {worstRelative:E2} of the sum of |x·w|) "));
    }

    public static void MeasuredChoiceIsStored(Checks t)
    {
        var gpu = Gpu.Get(out string reason);
        if (gpu is null)
        {
            t.Check(false, $"no CUDA device: {reason}");
            return;
        }

        var s = SuperBlocks.Q4K;
        var product = new SuperProduct(gpu.Device, gpu.Tuner);
        var (m, n, k) = (200, 300, 768);
        var (x, w) = ShapeData(m, n, k);
        using var op = new Operands(gpu.Device, x, w, m, n, k);
        product.Run(s, op.Op);
        int lane = product.ChosenLane(s, m, n, k);
        t.Check(lane is >= 0 and < 3, $"the chosen lane {lane}");
        int measured = gpu.Tuner.Log.Count;
        product.Run(s, op.Op);
        t.Check(gpu.Tuner.Log.Count == measured, "a second call measured again");
        string saved = product.SaveChoices();
        t.Check(saved.Contains(string.Create(CultureInfo.InvariantCulture, $"lane {s.Id} {Idrak.Adad.Measure.SizeClass.Of(m)} {n} {k} = {lane}"), StringComparison.Ordinal),
            $"the store's lines carry the description's id and m's class: {saved}");
        var again = new SuperProduct(gpu.Device, gpu.Tuner) { Measuring = false };
        t.Check(again.LoadChoices(saved) > 0 && again.ChosenLane(s, m, n, k) == lane, "the store's lines read back");
    }
}
