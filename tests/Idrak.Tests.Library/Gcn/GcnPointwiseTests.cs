// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak;
using Idrak.Abstraction.Devices.Cpu;
using Idrak.Gpu.Abstractions;
using Idrak.Gpu.Hip;
using Idrak.Tests.Gcn;

// HIP's pointwise kernels in GCN (GcnKernels.Pointwise), each run by the GCN interpreter and proven against the CPU (plan 35, G3a).
// Every kernel is assembled for two gfx9 processors and run over counts that are not a multiple of the block (1, a block
// less one, a few chunks with waves left idle, many chunks under a grid of five blocks, so each wave loops), and once
// with its buffers 4 bytes off 16-byte alignment (the tail alone, grid-stride). The inputs mix special values (NaN,
// ±∞, ±0, denormals, large magnitudes, the functions' edges) into random ones; the CPU backend computes the same
// operation on the same inputs, and every element is compared: NaN to NaN, ±∞ exactly, the rest within the tolerance
// the device-against-CPU test uses (HipTests: 1e-4 of max(1, |expected|) for the activations), bit for bit where the
// CPU's arithmetic is exact. The interpreter fails the run on any missing wait, race or access out of range; the
// floats before and past the count are checked untouched.
internal abstract partial class Tests
{
    internal static readonly (string Name, Action<Device> Run)[] GcnPointwiseGroup = PointwiseGroup();

    private enum PointwiseKind
    {
        Unary,
        Binary,
        MulAdd,
        Affine,
        Axpy,
        RowVector,
    }

    // One kernel's case: what the CPU runs, the inputs' random range, the largest finite magnitude among its special
    // values, and how close it must come (Exact: the bits, NaN's payload aside).
    private sealed record PointwiseCase(string Kernel, PointwiseKind Kind, string What, UnaryOp Unary = UnaryOp.Relu, BinaryOp Binary = BinaryOp.Add,
        float Low = -8f, float High = 8f, float Largest = float.MaxValue, bool Exact = false, float Tolerance = 1e-4f);

    private static (string Name, Action<Device> Run)[] PointwiseGroup()
    {
        PointwiseCase[] cases =
        [
            new("affine_f32", PointwiseKind.Affine, "y = fma(x, alpha, beta)", Exact: true),
            new("axpy_f32", PointwiseKind.Axpy, "y = fma(x, alpha, y)", Exact: true),
            new("muladd_f32", PointwiseKind.MulAdd, "c = fma(a, b, c)", Exact: true),
            new("add_f32", PointwiseKind.Binary, "c = a + b", Binary: BinaryOp.Add, Exact: true),
            new("sub_f32", PointwiseKind.Binary, "c = a - b", Binary: BinaryOp.Sub, Exact: true),
            new("mul_f32", PointwiseKind.Binary, "c = a · b", Binary: BinaryOp.Mul, Exact: true),
            new("max_f32", PointwiseKind.Binary, "c = MathF.Max(a, b) (NaN either side, +0 above -0)", Binary: BinaryOp.Maximum, Exact: true),
            new("min_f32", PointwiseKind.Binary, "c = MathF.Min(a, b) (NaN either side, -0 below +0)", Binary: BinaryOp.Minimum, Exact: true),
            new("add_rowvec_f32", PointwiseKind.RowVector, "c = a + v[i mod cols] (cols 1, 3, 13, 17, 1000)", Exact: true),
            new("relu_f32", PointwiseKind.Unary, "MathF.Max(x, 0) (NaN kept, -0 to +0)", Unary: UnaryOp.Relu, Exact: true),
            new("sigmoid_f32", PointwiseKind.Unary, "1 / (1 + e^-x)", Unary: UnaryOp.Sigmoid),
            new("tanh_f32", PointwiseKind.Unary, "the CPU's rational", Unary: UnaryOp.Tanh, Low: -10f, High: 10f),
            new("gelu_f32", PointwiseKind.Unary, "GELU, tanh form", Unary: UnaryOp.Gelu),
            new("gelu_erf_f32", PointwiseKind.Unary, "GELU, erf form", Unary: UnaryOp.GeluErf),
            new("silu_f32", PointwiseKind.Unary, "x · sigmoid(x)", Unary: UnaryOp.Silu),
            new("square_f32", PointwiseKind.Unary, "x · x", Unary: UnaryOp.Square, Exact: true),
            new("abs_f32", PointwiseKind.Unary, "|x|", Unary: UnaryOp.Abs, Exact: true),
            new("exp_f32", PointwiseKind.Unary, "e^x (to 0 below -103, ∞ above 88.72)", Unary: UnaryOp.Exp, Low: -90f, High: 88f),
            new("log_f32", PointwiseKind.Unary, "ln x (denormals, 0, negatives)", Unary: UnaryOp.Log, Low: 1e-6f, High: 1e6f),
            new("sqrt_f32", PointwiseKind.Unary, "√x correctly rounded", Unary: UnaryOp.Sqrt, Low: 0f, High: 1e4f, Exact: true),
            new("sin_f32", PointwiseKind.Unary, "sin x (magnitudes to 1e5)", Unary: UnaryOp.Sin, Low: -100f, High: 100f, Largest: 1e5f),
            new("cos_f32", PointwiseKind.Unary, "cos x (magnitudes to 1e5)", Unary: UnaryOp.Cos, Low: -100f, High: 100f, Largest: 1e5f),
            // ±0 by value: the CPU's vector lanes give +0 for -0, its scalar tail -0 (x itself, as CUDA's sign_f32 and this kernel).
            new("sign_f32", PointwiseKind.Unary, "±1, ±0 (by value) and NaN kept", Unary: UnaryOp.Sign, Tolerance: 0f),
        ];
        return
        [
            ("gcn kernels: element-wise: GcnKernels.All (fill_f32, the element-wise and the row kernels) assembles into one code object for gfx900, gfx906, gfx908 and gfx90c, every kernel found in it by name; each element-wise kernel declares its argument list, its block of 256 and the dispatch packet, and its printed text reads back to the same bytes", PointwiseAll),
            .. cases.Select(c => ($"gcn kernels: element-wise {c.Kernel}, {c.What}: in the interpreter for gfx90c and gfx906 over 1, 255, 1,000 and about 70,000 elements (grids of 1 to 5 blocks) and about 4,100 misaligned ones, special values included, against the CPU backend {(c.Exact ? "bit for bit" : string.Create(CultureInfo.InvariantCulture, $"within {c.Tolerance:G}"))}; no missing wait, race or access out of range, nothing outside the count written",
                (Action<Device>)(device => PointwiseCheck(device, c)))),
        ];
    }

    // The processors the kernels run on in the interpreter: the one this plan is checked against and another of the family.
    private static readonly string[] PointwiseIsas = ["gfx90c", "gfx906"];

    // Special values: NaN, ±∞, ±0, denormals, the smallest normal, large magnitudes, and the functions' edges (e^x's
    // overflow and underflow, tanh's saturation, the trig functions' larger turns).
    private static readonly float[] PointwiseSpecials =
    [
        float.NaN, float.PositiveInfinity, float.NegativeInfinity, 0f, -0f, 1e-40f, -1e-40f, float.Epsilon, -2.5e-39f, 1.17549435e-38f,
        3e38f, -3e38f, 1e30f, -1e30f, 1e10f, -1e10f, 88.5f, -88.5f, 100f, -100f, -104f, 1e5f, -1e5f, 9.5f, -9.5f, 7.9f, 1e-30f, 0.5f, -1f, 1f,
        -12345.678f, 31415.926f,
    ];

    internal static void PointwiseAll(Device device)
    {
        _ = device;
        var all = GcnKernels.All;
        Check(all.Select(k => k.Name).Distinct().Count() == all.Count, $"kernel names are unique: {string.Join(", ", all.Select(k => k.Name))}");
        Check(GcnKernels.Pointwise.Count == 23 && all.Skip(1).Take(GcnKernels.Pointwise.Count).SequenceEqual(GcnKernels.Pointwise),
            $"the 23 element-wise kernels follow fill_f32: {string.Join(", ", GcnKernels.Pointwise.Select(k => k.Name))}");
        foreach (string processor in new[] { "gfx900", "gfx906", "gfx908", "gfx90c" })
        {
            var isa = GcnIsaVersion.Parse(processor);
            var target = GcnTargets.For(isa);
            var codes = new List<GcnKernelCode>();
            foreach (var source in all)
            {
                using var asm = new GcnAssembler(target, isa);
                var code = asm.AssembleKernel(source.Text);
                codes.Add(code);
                if (GcnKernels.Pointwise.Contains(source))
                {
                    var expected = PointwiseArgumentsOf(source.Name);
                    Check(code.Arguments.SequenceEqual(expected) && code.RequiredWorkgroupSize == (GcnKernels.PointwiseBlock, 1, 1)
                          && code.UserSgprs == (GcnUserSgprs.DispatchPtr | GcnUserSgprs.KernargSegmentPtr) && code.WorkgroupIds == 1,
                        $"{processor} {source.Name}: arguments {string.Join(", ", code.Arguments)}, block {code.RequiredWorkgroupSize}, user SGPRs {code.UserSgprs}");
                    byte[] again = AssembleGcn(target, isa, asm.PrintedText);
                    Check(again.AsSpan().SequenceEqual(code.Code.Span), $"{processor} {source.Name}: its printed text reads back to other bytes:\n{asm.PrintedText}");
                }
            }

            byte[] codeObject = GcnCodeObject.Write(target, isa, 5, [.. codes]);
            foreach (var code in codes)
            {
                var image = GcnCodeObjectReader.Read(codeObject, code.Name);
                Check(image.Name == code.Name && image.Code.Length > 0, $"{processor}: {code.Name} found in the code object");
            }

            if (processor == "gfx90c")
            {
                Console.WriteLine($"    gfx90c registers (VGPRs/SGPRs): {string.Join(", ", codes.Where(c => GcnKernels.Pointwise.Any(p => p.Name == c.Name)).Select(c => $"{c.Name} {c.Vgprs}/{c.Sgprs}"))}");
            }
        }
    }

    private static IReadOnlyList<GcnArgument> PointwiseArgumentsOf(string kernel) => kernel switch
    {
        "affine_f32" => GcnKernels.AffineArguments,
        "axpy_f32" => GcnKernels.AxpyArguments,
        "add_f32" or "sub_f32" or "mul_f32" or "max_f32" or "min_f32" or "muladd_f32" => GcnKernels.BinaryArguments,
        "add_rowvec_f32" => GcnKernels.RowVectorArguments,
        _ => GcnKernels.UnaryArguments,
    };

    private static void PointwiseCheck(Device device, PointwiseCase c)
    {
        _ = device;
        var source = GcnKernels.Pointwise.Single(k => k.Name == c.Kernel);
        double worst = 0;
        long worstUlps = 0;
        int compared = 0;
        foreach (string processor in PointwiseIsas)
        {
            var isa = GcnIsaVersion.Parse(processor);
            var target = GcnTargets.For(isa);
            GcnKernelCode code;
            using (var asm = new GcnAssembler(target, isa))
            {
                code = asm.AssembleKernel(source.Text);
            }

            var image = GcnCodeObjectReader.Read(GcnCodeObject.Write(target, isa, 5, [code]), c.Kernel);
            (int N, int Blocks, bool Misaligned, int Cols)[] runs = c.Kind == PointwiseKind.RowVector
                ? [(1, 1, false, 1), (255, 1, false, 17), (1000, 3, false, 1000), (70023, 5, false, 17), (70002, 5, false, 3), (4095, 2, true, 13)]
                : [(1, 1, false, 0), (255, 1, false, 0), (1000, 3, false, 0), (70001, 5, false, 0), (4099, 2, true, 0)];
            foreach (var run in runs)
            {
                var (gap, ulps, count) = PointwiseRun(c, image, processor, run.N, run.Blocks, run.Misaligned, run.Cols);
                worst = Math.Max(worst, gap);
                worstUlps = Math.Max(worstUlps, ulps);
                compared += count;
            }
        }

        Console.WriteLine(string.Create(CultureInfo.InvariantCulture,
            $"    {c.Kernel}: {compared:N0} elements on two processors; largest gap to the CPU {worst:G3} (of max(1, |expected|)), {worstUlps} ulp"));
    }

    // One run: the inputs, the interpreter's output and the CPU's, compared. The largest gap, the largest distance in
    // ulps (finite values of one sign) and the elements compared.
    private static (double Gap, long Ulps, int Count) PointwiseRun(PointwiseCase c, GcnKernelImage image, string processor, int n, int blocks, bool misaligned, int cols)
    {
        const float Alpha = -1.25f, Beta = 0.5f, AxpyAlpha = 0.75f, Sentinel = -98765.5f;
        const int Pad = 67;
        var random = new Random(n * 31 + blocks);
        float Value(int i, int salt)
        {
            if (i % 5 == 2 + salt % 3)
            {
                float special = PointwiseSpecials[(i / 5 + salt) % PointwiseSpecials.Length];
                if (!float.IsFinite(special) || MathF.Abs(special) <= c.Largest)
                {
                    return special;
                }
            }

            return c.Low + random.NextSingle() * (c.High - c.Low);
        }

        float[] Values(int count, int salt) => [.. Enumerable.Range(0, count).Select(i => Value(i, salt))];

        // The buffers in argument order, and which one the kernel writes.
        float[][] buffers;
        int written;
        switch (c.Kind)
        {
            case PointwiseKind.Unary or PointwiseKind.Affine or PointwiseKind.Axpy:
                buffers = [Values(n, 0), Values(n, 1)];
                written = 1;
                break;
            case PointwiseKind.RowVector:
                buffers = [Values(n, 0), Values(cols, 1), Values(n, 2)];
                written = 2;
                break;
            default:
            {
                float[] a = Values(n, 0), b = Values(n, 1);
                for (int i = 0; i < n; i++)
                {
                    // Ties, signed zeros against each other and NaN against NaN, for the extremes.
                    (a[i], b[i]) = (i % 13) switch
                    {
                        4 => (0f, -0f),
                        9 => (-0f, 0f),
                        6 => (a[i], a[i]),
                        11 => (float.NaN, float.NaN),
                        _ => (a[i], b[i]),
                    };
                }

                buffers = [a, b, Values(n, 2)];
                written = 2;
                break;
            }
        }

        // The interpreter: each buffer mapped (4 bytes in when misaligned) between sentinels.
        int first = misaligned ? 1 : 0;
        var memory = new GcnMemory();
        var addresses = new ulong[buffers.Length];
        for (int b = 0; b < buffers.Length; b++)
        {
            var mapped = new float[first + buffers[b].Length + (b == written ? Pad : 0)];
            Array.Fill(mapped, Sentinel);
            buffers[b].CopyTo(mapped, first);
            addresses[b] = memory.Map<float>($"buffer {b}", mapped) + (ulong)(4 * first);
        }

        byte[] arguments;
        switch (c.Kind)
        {
            case PointwiseKind.Unary:
                arguments = new byte[GcnKernels.UnaryArgumentBytes];
                GcnKernels.PackUnary(arguments, addresses[0], addresses[1], (uint)n);
                break;
            case PointwiseKind.Affine:
                arguments = new byte[GcnKernels.AffineArgumentBytes];
                GcnKernels.PackAffine(arguments, addresses[0], addresses[1], Alpha, Beta, (uint)n);
                break;
            case PointwiseKind.Axpy:
                arguments = new byte[GcnKernels.AxpyArgumentBytes];
                GcnKernels.PackAxpy(arguments, addresses[0], addresses[1], AxpyAlpha, (uint)n);
                break;
            case PointwiseKind.RowVector:
                arguments = new byte[GcnKernels.RowVectorArgumentBytes];
                GcnKernels.PackAddRowVector(arguments, addresses[0], addresses[1], addresses[2], (uint)cols, (uint)n);
                break;
            default:
                arguments = new byte[GcnKernels.BinaryArgumentBytes];
                GcnKernels.PackBinary(arguments, addresses[0], addresses[1], addresses[2], (uint)n);
                break;
        }

        string what = string.Create(CultureInfo.InvariantCulture,
            $"{c.Kernel} on {processor}, {n} elements{(cols > 0 ? $" ({n / cols} rows of {cols})" : "")} in {blocks} blocks{(misaligned ? ", misaligned" : "")}");
        try
        {
            GcnInterpreter.Run(image, memory, new GcnDispatch
            {
                Workgroups = (blocks, 1, 1), WorkgroupSize = (GcnKernels.PointwiseBlock, 1, 1), Arguments = arguments, Seed = (ulong)(n + blocks),
            });
        }
        catch (GcnInterpreterException e)
        {
            throw new Exception($"{what}: the interpreter reports {e.Kind}: {e.Message}", e);
        }

        float[] device = memory.Read<float>(addresses[written] - (ulong)(4 * first), first + n + Pad);
        Check(device.AsSpan(0, first).IndexOfAnyExcept(Sentinel) < 0 && device.AsSpan(first + n).IndexOfAnyExcept(Sentinel) < 0,
            $"{what}: a float before or past the count was written");

        // The CPU backend on the same inputs.
        var cpu = CpuBackend.Instance;
        var storages = new Storage?[buffers.Length];
        try
        {
            for (int b = 0; b < buffers.Length; b++)
            {
                storages[b] = cpu.Allocate(buffers[b].Length, zeroed: false);
                cpu.Upload(buffers[b], storages[b]!);
            }

            var s = storages.Select(x => x!).ToArray();
            switch (c.Kind)
            {
                case PointwiseKind.Unary: cpu.Unary(c.Unary, s[0], s[1], n); break;
                case PointwiseKind.Affine: cpu.Affine(s[0], s[1], n, Alpha, Beta); break;
                case PointwiseKind.Axpy: cpu.Axpy(s[0], s[1], n, AxpyAlpha); break;
                case PointwiseKind.MulAdd: cpu.MulAdd(s[0], s[1], s[2], n); break;
                case PointwiseKind.RowVector: cpu.AddRowVector(s[0], s[1], s[2], n / cols, cols); break;
                default: cpu.Binary(c.Binary, s[0], s[1], s[2], n); break;
            }

            var expected = new float[n];
            cpu.Download(s[written], expected);
            double worst = 0;
            long worstUlps = 0;
            for (int i = 0; i < n; i++)
            {
                float e = expected[i], a = device[first + i];
                bool same = float.IsNaN(e) ? float.IsNaN(a) : BitConverter.SingleToInt32Bits(e) == BitConverter.SingleToInt32Bits(a);
                double gap = same ? 0
                    : float.IsNaN(e) || float.IsNaN(a) || float.IsInfinity(e) || float.IsInfinity(a) ? double.PositiveInfinity
                    : Math.Abs((double)e - a) / Math.Max(1.0, Math.Abs((double)e));
                if (c.Exact ? !same : !(gap <= c.Tolerance))
                {
                    int at = i;
                    string inputs = string.Join(", ", buffers.Select((values, b) => b == written && c.Kind is not (PointwiseKind.Axpy or PointwiseKind.MulAdd) ? null
                        : values[c.Kind == PointwiseKind.RowVector && b == 1 ? at % cols : at].ToString("R", CultureInfo.InvariantCulture)).OfType<string>());
                    throw new Exception(string.Create(CultureInfo.InvariantCulture,
                        $"{what}: element {i} (inputs {inputs}) is {a:R}, the CPU's {e:R}{(c.Exact ? " (bit for bit)" : $" (gap {gap:G3}, allowed {c.Tolerance:G})")}"));
                }

                worst = Math.Max(worst, gap);
                if (!same && float.IsFinite(e) && float.IsFinite(a) && float.IsNegative(e) == float.IsNegative(a))
                {
                    worstUlps = Math.Max(worstUlps, Math.Abs((long)BitConverter.SingleToInt32Bits(e) - BitConverter.SingleToInt32Bits(a)));
                }
            }

            return (worst, worstUlps, n);
        }
        finally
        {
            foreach (var storage in storages)
            {
                storage?.Release();
            }
        }
    }
}
