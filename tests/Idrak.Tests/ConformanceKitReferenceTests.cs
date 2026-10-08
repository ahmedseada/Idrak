// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.InteropServices;
using Idrak.Abstraction.Operations;
using Idrak.Abstraction.Testing;

// Plan 10, W4.5: the kit's references for the operations the CPU has no kernel for (only devices with matrix units run
// them), proved without such a device: kernels written here from the contracts, independently of the kit, registered
// on the minimal device pass the check, and each made slightly wrong fails it under its own name.
internal static partial class Tests
{
    // Kernels from the contracts in Backend.cs, written apart from the kit's references (another e4m3 encoder, another
    // loop order); `wrong` makes one of them slightly wrong.
    private sealed class MatrixUnitKernels(string wrong = "")
    {
        // Every non-negative finite e4m3 value by code (0x00-0x7E): 0.mmm · 2⁻⁶ for exponent field 0, else 1.mmm · 2^(e-7).
        private static readonly double[] Codes = [.. Enumerable.Range(0, 0x7F).Select(code =>
            (code >> 3) == 0 ? (code & 7) / 8.0 * Math.Pow(2, -6) : (1 + (code & 7) / 8.0) * Math.Pow(2, (code >> 3) - 7))];

        // The nearest code by search (ties to the even code, past 448 the largest), the sign kept; NaN 0x7F.
        // Truncating: the largest code not above |x| (a wrong rounding).
        public static byte Encode(float x, bool truncate = false)
        {
            if (float.IsNaN(x))
            {
                return 0x7F;
            }

            double a = Math.Abs((double)x);
            int best = 0;
            for (int code = 1; code < Codes.Length; code++)
            {
                double distance = Math.Abs(Codes[code] - a), bestDistance = Math.Abs(Codes[best] - a);
                if (truncate ? Codes[code] <= a : distance < bestDistance || (distance == bestDistance && (code & 1) == 0))
                {
                    best = code;
                }
            }

            return (byte)(best | (float.IsNegative(x) ? 0x80 : 0));
        }

        public static double Decode(byte code) => (code & 0x7F) == 0x7F ? double.NaN : (code & 0x80) != 0 ? -Codes[code & 0x7F] : Codes[code];

        private static float[] Get(Backend backend, Storage s)
        {
            var values = new float[s.Length];
            backend.Download(s, values);
            return values;
        }

        // The codes of `values` scaled by the reciprocal of their scale, max |·| / 448 (1 for zeros).
        private static byte[] Quantize(IReadOnlyList<float> values, out float scale, bool truncate = false)
        {
            float max = values.Count == 0 ? 0f : values.Max(MathF.Abs);
            scale = max > 0f ? max / 448f : 1f;
            float inverse = 1f / scale;
            return [.. values.Select(value => Encode(value * inverse, truncate))];
        }

        public bool Float8QuantizeWeight(Backend backend, Storage w, int k, int n, Storage values, Storage scales)
        {
            float[] wv = Get(backend, w);
            int padded = (k + 63) / 64 * 64;
            var bytes = new byte[n * padded];
            var columnScales = new float[n];
            for (int j = 0; j < n; j++)
            {
                Quantize([.. Enumerable.Range(0, k).Select(r => wv[r * n + j])], out columnScales[j], wrong == "Float8QuantizeWeight").CopyTo(bytes, j * padded);
            }

            backend.Upload(MemoryMarshal.Cast<byte, float>(bytes.AsSpan()), values);
            backend.Upload(columnScales, scales);
            return true;
        }

        public bool Float8MatMul(Backend backend, Storage x, int m, int k, Storage values, Storage scales, int n, Storage y, float beta)
        {
            float[] xv = Get(backend, x), sv = Get(backend, scales), yv = Get(backend, y);
            var w8 = MemoryMarshal.Cast<float, byte>(Get(backend, values).AsSpan()).ToArray();
            int padded = w8.Length / Math.Max(n, 1);
            for (int i = 0; i < m; i++)
            {
                var x8 = Quantize(xv.AsSpan(i * k, k).ToArray(), out float sx);
                for (int j = 0; j < n; j++)
                {
                    double sum = 0;
                    for (int p = 0; p < k; p++)
                    {
                        sum += Decode(x8[p]) * Decode(w8[j * padded + p]);
                    }

                    float product = (float)(sx * sv[j] * sum) * (wrong == "Float8MatMul" ? 1.05f : 1f);
                    yv[i * n + j] = beta == 0f ? product : product + beta * yv[i * n + j];
                }
            }

            backend.Upload(yv, y);
            return true;
        }

        // Scores of query head `head` (batch b, step t) against key positions 0..t, scaled: the visible keys only
        // (one more, a future key, when the forward pass is to be wrong).
        private static double[] Scores(float[] q, long qAt, float[] k, long kOffset, int kRow, int kvHead, int b, int steps, int t, int dim, float scale, bool peek)
        {
            int visible = Math.Min(steps, t + 1 + (peek ? 1 : 0));
            var s = new double[visible];
            for (int c = 0; c < visible; c++)
            {
                for (int d = 0; d < dim; d++)
                {
                    s[c] += (double)q[qAt + d] * k[kOffset + (long)(b * steps + c) * kRow + kvHead * dim + d];
                }

                s[c] *= scale;
            }

            return s;
        }

        public bool AttentionStrided(Backend backend, Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
            Storage y, Storage? logSumExp, int batch, int kvHeads, int group, int steps, int dim, float scale)
        {
            float[] qv = Get(backend, q), kv = Get(backend, k), vv = Get(backend, v), yv = Get(backend, y);
            float[]? lse = logSumExp is null ? null : Get(backend, logSumExp);
            int heads = kvHeads * group;
            for (int b = 0; b < batch; b++)
            {
                for (int t = 0; t < steps; t++)
                {
                    for (int head = 0; head < heads; head++)
                    {
                        int kvHead = head / group, g = head % group;
                        var s = Scores(qv, qOffset + (long)(b * steps + t) * qRow + head * dim, kv, kOffset, kRow, kvHead, b, steps, t, dim, scale,
                            wrong == "AttentionStrided");
                        double total = s.Sum(e => Math.Exp(e));
                        for (int d = 0; d < dim; d++)
                        {
                            double acc = 0;
                            for (int c = 0; c < s.Length; c++)
                            {
                                acc += Math.Exp(s[c]) / total * vv[vOffset + (long)(b * steps + c) * kRow + kvHead * dim + d];
                            }

                            yv[((long)(b * steps + t) * heads + head) * dim + d] = (float)acc;
                        }

                        if (lse is not null)
                        {
                            lse[((b * kvHeads + kvHead) * group + g) * steps + t] = (float)Math.Log(total);
                        }
                    }
                }
            }

            backend.Upload(yv, y);
            if (lse is not null)
            {
                backend.Upload(lse, logSumExp!);
            }

            return true;
        }

        public bool AttentionStridedBackward(Backend backend, Storage q, long qOffset, Storage k, long kOffset, Storage v, long vOffset, int qRow, int kRow,
            Storage y, Storage logSumExp, Storage dOutput, Storage dq, long dqOffset, Storage dk, long dkOffset, Storage dv, long dvOffset,
            int batch, int kvHeads, int group, int steps, int dim, float scale)
        {
            float[] qv = Get(backend, q), kv = Get(backend, k), vv = Get(backend, v), yv = Get(backend, y), lse = Get(backend, logSumExp), dov = Get(backend, dOutput);

            // dq, dk and dv may be one storage: one host copy per distinct storage.
            var copies = new List<(Storage Storage, float[] Values)>();
            float[] Copy(Storage s)
            {
                foreach (var (storage, values) in copies)
                {
                    if (ReferenceEquals(storage, s))
                    {
                        return values;
                    }
                }

                copies.Add((s, Get(backend, s)));
                return copies[^1].Values;
            }

            float[] dqv = Copy(dq), dkv = Copy(dk), dvv = Copy(dv);
            int heads = kvHeads * group;
            float keyScale = wrong == "AttentionStridedBackward" ? 1f : scale;
            for (int b = 0; b < batch; b++)
            {
                for (int head = 0; head < heads; head++)
                {
                    int kvHead = head / group, g = head % group;
                    for (int t = 0; t < steps; t++)
                    {
                        long qAt = (long)(b * steps + t) * qRow + head * dim, yAt = ((long)(b * steps + t) * heads + head) * dim;
                        var s = Scores(qv, qOffset + qAt, kv, kOffset, kRow, kvHead, b, steps, t, dim, scale, peek: false);
                        double l = lse[((b * kvHeads + kvHead) * group + g) * steps + t], delta = 0;
                        for (int d = 0; d < dim; d++)
                        {
                            delta += (double)yv[yAt + d] * dov[yAt + d];
                        }

                        for (int c = 0; c < s.Length; c++)
                        {
                            long kAt = (long)(b * steps + c) * kRow + kvHead * dim;
                            double p = Math.Exp(s[c] - l), dp = 0;
                            for (int d = 0; d < dim; d++)
                            {
                                dp += (double)dov[yAt + d] * vv[vOffset + kAt + d];
                            }

                            double ds = p * (dp - delta);
                            for (int d = 0; d < dim; d++)
                            {
                                dqv[dqOffset + qAt + d] += (float)(scale * ds * kv[kOffset + kAt + d]);
                                dkv[dkOffset + kAt + d] += (float)(keyScale * ds * qv[qOffset + qAt + d]);
                                dvv[dvOffset + kAt + d] += (float)(p * dov[yAt + d]);
                            }
                        }
                    }
                }
            }

            foreach (var (storage, values) in copies)
            {
                backend.Upload(values, storage);
            }

            return true;
        }

        // Registers the four kernels for a device kind until disposed.
        public IDisposable Register(string kind)
        {
            IDisposable[] registrations =
            [
                Kernels.Register(Ops.Float8QuantizeWeight, kind, (OperationKernels.Float8QuantizeWeight)Float8QuantizeWeight),
                Kernels.Register(Ops.Float8MatMul, kind, (OperationKernels.Float8MatMul)Float8MatMul),
                Kernels.Register(Ops.AttentionStrided, kind, (OperationKernels.AttentionStrided)AttentionStrided),
                Kernels.Register(Ops.AttentionStridedBackward, kind, (OperationKernels.AttentionStridedBackward)AttentionStridedBackward),
            ];
            return new Registrations(registrations);
        }

        private sealed class Registrations(IDisposable[] registrations) : IDisposable
        {
            public void Dispose()
            {
                foreach (var registration in registrations)
                {
                    registration.Dispose();
                }
            }
        }
    }

    // e4m3 codes from the format's definition: the largest value, saturation, the subnormals, ties to even, NaN, signs.
    private static void Float8CodesKnown()
    {
        (float Value, byte Code)[] known =
        [
            (448f, 0x7E), (-448f, 0xFE), (450f, 0x7E), (464f, 0x7E), (1e9f, 0x7E), (float.PositiveInfinity, 0x7E), (float.NegativeInfinity, 0xFE),
            (416f, 0x7D), (432f, 0x7E), (1f, 0x38), (-1f, 0xB8), (1.0625f, 0x38), (1.1875f, 0x3A), (17f, 0x58), (19f, 0x5A), (0.3f, 0x2A),
            (240f, 0x77), (0f, 0x00), (-0f, 0x80), (MathF.ScaleB(1f, -6), 0x08), (MathF.ScaleB(1f, -9), 0x01), (MathF.ScaleB(7f, -9), 0x07),
            (MathF.ScaleB(1f, -10), 0x00), (MathF.ScaleB(3f, -10), 0x02), (MathF.ScaleB(15f, -10), 0x08), (MathF.ScaleB(1f, -11), 0x00),
            (-MathF.ScaleB(5f, -10), 0x82), (float.NaN, 0x7F),
        ];
        foreach (var (value, code) in known)
        {
            byte encoded = MatrixUnitKernels.Encode(value);
            Check(encoded == code, $"e4m3 of {value:R}: 0x{encoded:X2}, expected 0x{code:X2}");
        }

        Check(MatrixUnitKernels.Decode(0x7E) == 448 && MatrixUnitKernels.Decode(0xFE) == -448 && MatrixUnitKernels.Decode(0x01) == Math.Pow(2, -9)
              && MatrixUnitKernels.Decode(0x08) == Math.Pow(2, -6) && MatrixUnitKernels.Decode(0x38) == 1 && MatrixUnitKernels.Decode(0x77) == 240
              && double.IsNaN(MatrixUnitKernels.Decode(0x7F)) && double.IsNaN(MatrixUnitKernels.Decode(0xFF)), "e4m3 values of known codes");
        for (int code = 0; code < 256; code++)
        {
            if ((code & 0x7F) != 0x7F)
            {
                Check(MatrixUnitKernels.Encode((float)MatrixUnitKernels.Decode((byte)code)) == code, $"e4m3 code 0x{code:X2} round trip");
            }
        }
    }

    private static void KitChecksMatrixUnitOperations(Device device)
    {
        _ = device;
        if (!FirstRun(nameof(KitChecksMatrixUnitOperations)))
        {
            return;
        }

        Float8CodesKnown();
        string[] names = ["Float8QuantizeWeight", "Float8MatMul", "AttentionStrided", "AttentionStridedBackward"];
        var minimal = MinimalBackend.Instance;
        var options = new DeviceCheckOptions
        {
            Filter = name => name.StartsWith("packed weights: fused", StringComparison.Ordinal) || name.StartsWith("attention:", StringComparison.Ordinal),
        };

        // Without kernels of their own: not run, no failure (the CPU too).
        foreach (var report in new[] { Conformance.Check(minimal, options), Conformance.Check(Device.Cpu, options) })
        {
            report.ThrowIfFailed();
            foreach (string name in names)
            {
                var entry = report.Entries.Single(e => e.Name == name);
                Check(entry.Status == CheckStatus.Skipped && entry.Detail.StartsWith("unsupported", StringComparison.Ordinal), $"{name} without a kernel: {entry}");
            }
        }

        // Correct kernels pass, every call compared.
        using (new MatrixUnitKernels().Register("minimal"))
        {
            var report = Conformance.Check(minimal, options);
            report.ThrowIfFailed();
            foreach (string name in names)
            {
                var entry = report.Entries.Single(e => e.Name == name);
                Check(entry.Status == CheckStatus.Passed && entry.Detail.StartsWith("a registered kernel", StringComparison.Ordinal)
                      && !entry.Detail.Contains("no kernel", StringComparison.Ordinal), $"{name} with a correct kernel: {entry}");
            }
        }

        // Each one slightly wrong (truncated codes, a product 5% off, one future key seen, dk without the scale): found under its name.
        foreach (string wrong in names)
        {
            using (new MatrixUnitKernels(wrong).Register("minimal"))
            {
                var report = Conformance.Check(minimal, options);
                Check(!report.Passed && report.Failures.Count > 0 && report.Failures.All(f => f.Check == wrong)
                      && report.Entries.Single(e => e.Name == wrong).Status == CheckStatus.Failed
                      && report.Entries.Where(e => e.Name != wrong && names.Contains(e.Name)).All(e => e.Status == CheckStatus.Passed),
                    $"a wrong {wrong} is found:\n{report}");
            }
        }
    }
}
