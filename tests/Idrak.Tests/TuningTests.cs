// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;
using Idrak.Backends.Cuda;
using Idrak.Layers;

// Card-dependent choices (k splits, kernel, tile) are measured on the device in use, not taken from one card's benchmarks.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] TuningGroup =
    [
        ("tuning: card-dependent choices (few-row kernel and splits, prompt splits, tensor-core splits, tiles) are measured on the device and give the results of the formulas' choices", TunedChoicesMatch),
    ];

    private static void TunedChoicesMatch(Device device)
    {
        if (device.Backend is not CudaBackend cuda)
        {
            return;                                                              // the CPU has nothing card-dependent to measure
        }

        var r = new Random(21);
        float[] Values(int n) => [.. Enumerable.Range(0, n).Select(_ => r.NextSingle() * 2 - 1)];
        var (w, xs, a, b) = (Values(512 * 1536), Values(64 * 512), Values(256 * 4096), Values(4096 * 256));
        float[][] Run()
        {
            using var weights = Tensor.From(w, [512, 1536], device);
            using var int8 = Int8Weight.Quantize(weights);
            var results = new List<float[]>();
            using (MixedPrecision.BFloat16())
            {
                foreach (int rows in new[] { 1, 4, 8, 40 })                     // GEMV splits, few-row kernel, prompt splits
                {
                    using var x = Tensor.From(xs.AsSpan(0, rows * 512), [rows, 512], device);
                    using var y = x.MatMulInt8(int8);
                    results.Add(y.ToArray());
                }

                using var ta = Tensor.From(a, [256, 4096], device);              // a long k over few output tiles: k splits
                using var tb = Tensor.From(b, [4096, 256], device);
                using var product = ta.MatMul(tb);
                results.Add(product.ToArray());
            }

            using var fa = Tensor.From(a, [256, 4096], device);                  // float tiles (no tensor cores)
            using var fb = Tensor.From(b, [4096, 256], device);
            using var plain = fa.MatMul(fb);
            results.Add(plain.ToArray());
            return [.. results];
        }

        bool autotune = CudaBackend.Autotune;
        try
        {
            CudaBackend.Autotune = false;
            cuda.ForgetTuning();
            var formulas = Run();
            Check(cuda.TunedCount == 0, "nothing measured with autotuning off");

            CudaBackend.Autotune = true;
            var measured = Run();
            Check(cuda.TunedCount > 0, $"choices measured on the device ({cuda.TunedCount})");
            int known = cuda.TunedCount;
            var again = Run();
            Check(cuda.TunedCount == known, "each shape is measured once");
            for (int i = 0; i < formulas.Length; i++)
            {
                AssertClose(formulas[i], measured[i], 2e-3f * Math.Max(1f, formulas[i].Max(Math.Abs)), $"product {i}: measured choice against the formula's");
                AssertClose(measured[i], again[i], 2e-3f * Math.Max(1f, measured[i].Max(Math.Abs)), $"product {i}: the kept choice");
            }
        }
        finally
        {
            CudaBackend.Autotune = autotune;
        }
    }
}
