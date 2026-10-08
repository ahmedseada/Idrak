// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Devices;
using Idrak.Abstraction.Operations;
using Idrak.Abstraction.Testing;
using Idrak.Generation;
using Idrak.Gpu.Cuda;
using Idrak.Gpu.Hip;
using Idrak.Layers;
using Idrak.Layers.Abstractions;

namespace Idrak.PluginTests;

/// <summary>
/// The plug-ins' own device kernels (<see cref="PluginKernels"/>): the packed format, the cache layout, an
/// <see cref="Autograd.Function"/> and a graph operation each run their plug-in operation, through its default kernel or
/// through the kernel <see cref="PluginKernels.Install"/> registers for the device's kind (the CPU; Vulkan, CUDA and HIP
/// for the packed format and the cache layout), seen in <see cref="Kernels.Chain"/> and counted by <see cref="Kernels.Trace"/>. The test
/// runner in tests/Idrak.Tests runs <see cref="All"/> on every device; a failed check throws.
/// </summary>
public static class KernelPluginTests
{
    /// <summary>The tests, by name.</summary>
    public static IReadOnlyList<(string Name, Action<Device> Run)> All { get; } =
    [
        ("outside plug-in kernels: the packed format unpacks through its plug-in operation, by default and through the kernel registered for the device (CPU, Vulkan SPIR-V, CUDA PTX, HIP C++), to the same weights and products", PackedFormatKernel),
        ("outside plug-in kernels: the key/value cache layout expands its rows through its plug-in operation, by default and through the kernel registered for the device (CPU, Vulkan SPIR-V, CUDA PTX, HIP C++), and generates the float32 cache's text", CacheLayoutKernel),
        ("outside plug-in kernels: an Autograd.Function and a graph operation run their plug-in operations, by default and through the CPU kernel, with their gradients", FunctionAndGraphOpKernels),
        ("outside plug-in kernels: the testing kit checks each plug-in kernel on the device against its default kernel on the CPU (Conformance.Check)", KitChecksKernels),
        ("outside plug-in kernels: CUDA and HIP kernels read their parameters from the PTX and the HIP C++ source and refuse other devices and mismatched arguments", GpuKernelShapes),
        ("outside plug-in kernels: the plug-in's PTX assembles with ptxas for sm_50, sm_75 and sm_120 (skipped without ptxas: IDRAK_PTXAS, PATH or CUDA_PATH)", PtxAssembles),
    ];

    private static void KitChecksKernels(Device device)
    {
        using var kernels = PluginKernels.Install();

        // Random values from the seed, uploaded to the backend; a fresh output; the kernel; the output read back.
        static (Storage Storage, float[] Values) Input(Backend backend, Random random, int n)
        {
            var values = Enumerable.Range(0, n).Select(_ => random.NextSingle() * 4f - 2f).ToArray();
            var storage = backend.Allocate(n, zeroed: false);
            backend.Upload(values, storage);
            return (storage, values);
        }

        static float[] Output(Backend backend, Storage output, int n, params Storage[] free)
        {
            var values = new float[n];
            backend.Download(output, values);
            foreach (var storage in free.Append(output))
            {
                backend.Return(storage);
            }

            return values;
        }

        var reports = new[]
        {
            Conformance.Check(PluginKernels.Unpack, device, (backend, kernel, seed) =>
            {
                var random = new Random(seed);
                int n = 1 + random.Next(300);
                var words = Enumerable.Range(0, (n + 1) / 2).Select(_ => BitConverter.UInt32BitsToSingle((uint)random.Next() & 0x7F7F7F7Fu)).ToArray();
                var packed = backend.Allocate(words.Length, zeroed: false);
                backend.Upload(words, packed);
                var values = backend.Allocate(n, zeroed: true);
                kernel(backend, packed, values, n);
                return Output(backend, values, n, packed);
            }, tolerance: 0f),
            Conformance.Check(PluginKernels.Scale, device, (backend, kernel, seed) =>
            {
                var random = new Random(seed);
                int width = 1 + random.Next(40), count = 1 + random.Next(20);
                var rows = Input(backend, random, count * width);
                var scales = Input(backend, random, count);
                var output = backend.Allocate(count * width, zeroed: true);
                kernel(backend, rows.Storage, scales.Storage, output, count * width, width);
                return Output(backend, output, count * width, rows.Storage, scales.Storage);
            }),
            Conformance.Check(PluginKernels.SoftplusOp, device, (backend, kernel, seed) =>
            {
                int n = 1 + new Random(seed).Next(500);
                var x = Input(backend, new Random(seed + 1), n);
                var y = backend.Allocate(n, zeroed: true);
                kernel(backend, x.Storage, y, n);
                return Output(backend, y, n, x.Storage);
            }),
        };

        foreach (var report in reports)
        {
            Check(report.Passed && report.Cases == 8, report.ToString());
        }

        bool registers = RegistersFormatKernels(device.Backend);
        Check(reports[0].Entries[0].Detail.StartsWith(registers ? "the registered kernel" : "the host kernel", StringComparison.Ordinal), reports[0].Entries[0].Detail);
    }

    private static void PackedFormatKernel(Device device)
    {
        var random = new Random(11);
        const int Rows = 17, Columns = 9, Batch = 3;
        var values = Enumerable.Range(0, Rows * Columns).Select(_ => random.NextSingle() * 0.4f - 0.2f).ToArray();
        var inputs = Enumerable.Range(0, Batch * Rows).Select(_ => random.NextSingle() * 2f - 1f).ToArray();
        var rounded = values.Select(v => BitConverter.UInt32BitsToSingle(((BitConverter.SingleToUInt32Bits(v) + 0x7FFFu + ((BitConverter.SingleToUInt32Bits(v) >> 16) & 1u)) >> 16) << 16)).ToArray();
        using var weight = RoundedWeight.Pack(values, Rows, Columns, device);
        Check(weight.Bytes == 4L * ((Rows * Columns + 1) / 2), $"two values a word: {weight.Bytes} bytes");

        var backend = device.Backend;
        var unpack = PluginKernels.Unpack;
        bool registers = RegistersFormatKernels(backend);
        float[]? product = null;
        foreach (bool installed in new[] { false, true })
        {
            using var kernels = installed ? PluginKernels.Install() : null;
            var expected = !installed ? (device.Type == DeviceType.Cpu ? KernelSource.Device : KernelSource.Host)
                : registers ? KernelSource.Registered : KernelSource.Host;
            Check(Kernels.Chain(backend)[unpack.Index].Source == expected, $"installed {installed}: {Kernels.Chain(backend)[unpack.Index].Source}, expected {expected}");

            using var scope = new TensorScope();
            float[] expanded;
            IReadOnlyDictionary<string, long> host;
            long calls;
            using (var trace = Kernels.Trace(backend))
            {
                expanded = weight.Dequantize().ToArray();
                host = trace.HostCallsByOperation;
                calls = trace.Calls(unpack);
            }

            CheckClose(rounded, expanded, 0f, $"installed {installed}: the unpacked weights");
            Check(calls == 1 && host.ContainsKey(unpack.Name) == (expected == KernelSource.Host), $"installed {installed}: {calls} calls, host fallbacks {string.Join(", ", host.Keys)}");

            // Through a Linear layer: the same product either way, with the input's gradient.
            using var linear = Linear.FromPacked(PackedWeight.Pack((v, r, c, d) => RoundedWeight.Pack(v, r, c, d), RoundedWeight.FormatName, values, Rows, Columns, device), null);
            var x = Tensor.From(inputs, [Batch, Rows], device, requiresGrad: true);
            var y = linear.Forward(x);
            var reference = Tensor.From(inputs, [Batch, Rows], device).MatMul(Tensor.From(rounded, [Rows, Columns], device)).ToArray();
            CheckClose(reference, y.ToArray(), 1e-5f, $"installed {installed}: the product");
            y.Sum().Backward();
            Check(x.Grad is not null, "the input's gradient");
            product ??= y.ToArray();
            CheckClose(product, y.ToArray(), 0f, "the same product with the registered kernel");
        }
    }

    private static void CacheLayoutKernel(Device device)
    {
        var layout = new ScaledLayout();
        var tokenizer = new CharTokenizer("abcdefghijklmnopqrstuvwxyz .,");
        var random = new Random(8);
        using var model = new Sequential
        {
            new Embedding(tokenizer.VocabularySize, 16, device, random),
            new PositionalEncoding(32, 16, device),
            new TransformerEncoderLayer(16, 2, dropout: 0f, causal: true, device: device, random: random),
            new LayerNorm(16, device: device),
            new Linear(16, tokenizer.VocabularySize, device: device, random: random),
        };
        var options = new GenerationOptions { Temperature = 0f, TopK = 1, RepeatPenalty = 1f, NumPredict = 8 };
        string expected = new TextGenerator(model, tokenizer, 32) { KeepCache = false }.Generate("abc", options).Text;

        var backend = device.Backend;
        var scale = PluginKernels.Scale;
        foreach (bool installed in new[] { false, true })
        {
            using var kernels = installed ? PluginKernels.Install() : null;
            var source = installed && RegistersFormatKernels(backend) ? KernelSource.Registered : KernelSource.Composed;
            Check(Kernels.Chain(backend)[scale.Index].Source == source, $"installed {installed}: {Kernels.Chain(backend)[scale.Index].Source}, expected {source}");
            string actual;
            long calls;
            using (var trace = Kernels.Trace(backend))
            {
                actual = new TextGenerator(model, tokenizer, 32) { KeepCache = false, CacheLayout = layout }.Generate("abc", options).Text;
                calls = trace.Calls(scale);
            }

            Check(actual == expected && calls > 0, $"installed {installed}: '{actual}' through {calls} expansions, float32 cache '{expected}'");
        }
    }

    private static void FunctionAndGraphOpKernels(Device device)
    {
        float[] values = [-3.5f, -1.2f, -0.4f, 0f, 0.3f, 1.1f, 2.6f, 4f];
        var backend = device.Backend;
        bool cpu = device.Type == DeviceType.Cpu;
        GraphOps.Register(PluginKernels.ScaledTanhName, PluginKernels.ScaledTanhGraphOp);
        var graph = new GraphModule("x", "y", [new GraphNode(PluginKernels.ScaledTanhName, ["x"], "y", new JsonObject { ["scale"] = 1.5f })]);
        try
        {
            foreach (bool installed in new[] { false, true })
            {
                using var kernels = installed ? PluginKernels.Install() : null;
                var expected = installed && cpu ? KernelSource.Registered : KernelSource.Composed;
                var chain = Kernels.Chain(backend);
                Check(chain[PluginKernels.SoftplusOp.Index].Source == expected && chain[PluginKernels.ScaledTanhOp.Index].Source == expected,
                    $"installed {installed}: softplus {chain[PluginKernels.SoftplusOp.Index].Source}, scaled tanh {chain[PluginKernels.ScaledTanhOp.Index].Source}");

                using var scope = new TensorScope();
                using var trace = Kernels.Trace(backend);

                // The Autograd.Function: softplus and its gradient sigmoid.
                var x = Tensor.From(values, [values.Length], device, requiresGrad: true);
                var y = PluginTests.Softplus.Apply(x);
                y.Sum().Backward();
                CheckClose([.. values.Select(v => MathF.Log(1f + MathF.Exp(v)))], y.ToArray(), 1e-5f, $"installed {installed}: softplus");
                CheckClose([.. values.Select(v => 1f / (1f + MathF.Exp(-v)))], x.Grad!.ToArray(), 1e-5f, $"installed {installed}: softplus gradient");

                // The graph operation: 1.5 · tanh and its gradient.
                var t = Tensor.From(values, [values.Length], device, requiresGrad: true);
                var z = graph.Forward(t);
                z.Sum().Backward();
                CheckClose([.. values.Select(v => 1.5f * MathF.Tanh(v))], z.ToArray(), 1e-5f, $"installed {installed}: the graph operation");
                CheckClose([.. values.Select(v => 1.5f * (1f - MathF.Tanh(v) * MathF.Tanh(v)))], t.Grad!.ToArray(), 1e-5f, $"installed {installed}: its gradient");
                Check(trace.Calls(PluginKernels.SoftplusOp) == 1 && trace.Calls(PluginKernels.ScaledTanhOp) == 1,
                    $"calls: softplus {trace.Calls(PluginKernels.SoftplusOp)}, scaled tanh {trace.Calls(PluginKernels.ScaledTanhOp)}");
            }
        }
        finally
        {
            GraphOps.Unregister(PluginKernels.ScaledTanhName);
            graph.Dispose();
        }
    }

    // Whether Install registers the packed format's and the cache layout's kernels for this device: every kind but HIP,
    // and HIP where hipRTC can compile them.
    private static bool RegistersFormatKernels(Backend backend) =>
        backend.Kind is "cpu" or "vulkan" or "cuda" || (backend.Kind == "hip" && HipKernel.UnavailableReason(backend) is null);

    private static void GpuKernelShapes(Device device)
    {
        var cuda = PluginKernels.CudaKernels;
        var hip = PluginKernels.HipKernels;
        Check(cuda[0].Parameters == "ppi" && cuda[1].Parameters == "pppii", $"PTX parameters: {cuda[0].Parameters}, {cuda[1].Parameters}");
        Check(hip[0].Parameters == "ppi" && hip[1].Parameters == "pppii", $"HIP parameters: {hip[0].Parameters}, {hip[1].Parameters}");
        Check(new CudaKernel(".visible .entry k(.param .u64 a, .param .b32 b, .param .f32 c, .param .s32 d, .param .align 8 .b8 e[16]) { ret; }", "k").Parameters == "xwfi?",
            "PTX parameter types");
        Check(new HipKernel("""extern "C" __global__ void __launch_bounds__(256) k(float* __restrict__ a, unsigned int b, float c, long long d, double e) {}""", "k").Parameters == "pifl?",
            "HIP parameter types");
        Throws<ArgumentException>(() => new CudaKernel(PluginGpuSources.Ptx, "outside_missing"), "an entry the PTX does not declare");
        Throws<ArgumentException>(() => new HipKernel(PluginGpuSources.Hip, "outside_missing"), "a function the source does not declare");

        var backend = device.Backend;
        var x = backend.Allocate(4, zeroed: true);
        try
        {
            if (backend.Kind != "cuda")
            {
                Throws<ArgumentException>(() => cuda[0].Launch(backend, 1, 1, 1, 64, 1, 1, [x, x, 4]), "a CUDA kernel on another device");
            }
            else
            {
                Throws<ArgumentException>(() => cuda[0].Launch(backend, 1, 1, 1, 64, 1, 1, [x, x, 4f]), "a float for an integer parameter");
                Throws<ArgumentException>(() => cuda[0].Launch(backend, 1, 1, 1, 64, 1, 1, [x, x]), "one argument short");
            }

            if (backend.Kind != "hip")
            {
                Throws<ArgumentException>(() => hip[0].Launch(backend, 1, 1, 1, 64, 1, 1, [x, x, 4]), "a HIP kernel on another device");
                Check(HipKernel.UnavailableReason(backend) is { Length: > 0 }, "no HIP kernels on another device");
            }
            else
            {
                Throws<ArgumentException>(() => hip[0].Launch(backend, 1, 1, 1, 64, 1, 1, [x, 4, x]), "an integer for a pointer parameter");
            }
        }
        finally
        {
            backend.Return(x);
        }
    }

    private static void PtxAssembles(Device device)
    {
        if (FindPtxas() is not { } ptxas)
        {
            Console.WriteLine("    (ptxas not found: set IDRAK_PTXAS, put it on PATH or set CUDA_PATH; skipped)");
            return;
        }

        string file = Path.Combine(Path.GetTempPath(), $"idrak-plugin-{Environment.ProcessId}.ptx");
        File.WriteAllText(file, PluginGpuSources.Ptx);
        try
        {
            foreach (string architecture in new[] { "sm_50", "sm_75", "sm_120" })
            {
                var start = new ProcessStartInfo(ptxas) { RedirectStandardError = true, RedirectStandardOutput = true };
                foreach (string argument in new[] { $"-arch={architecture}", file, "-o", OperatingSystem.IsWindows() ? "NUL" : "/dev/null" })
                {
                    start.ArgumentList.Add(argument);
                }

                using var process = Process.Start(start)!;
                var output = process.StandardOutput.ReadToEndAsync();
                string errors = process.StandardError.ReadToEnd() + output.Result;
                process.WaitForExit();
                Check(process.ExitCode == 0, $"ptxas -arch={architecture}: exit {process.ExitCode}: {errors}");
            }
        }
        finally
        {
            File.Delete(file);
        }
    }

    // ptxas from IDRAK_PTXAS, else PATH, else CUDA_PATH/bin; null when none is there.
    private static string? FindPtxas()
    {
        string name = OperatingSystem.IsWindows() ? "ptxas.exe" : "ptxas";
        if (Environment.GetEnvironmentVariable("IDRAK_PTXAS") is { Length: > 0 } chosen)
        {
            return File.Exists(chosen) ? chosen : throw new InvalidOperationException($"IDRAK_PTXAS names {chosen}, which does not exist");
        }

        var folders = (Environment.GetEnvironmentVariable("PATH") ?? "").Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries).ToList();
        if (Environment.GetEnvironmentVariable("CUDA_PATH") is { Length: > 0 } cudaPath)
        {
            folders.Add(Path.Combine(cudaPath, "bin"));
        }

        return folders.Select(f => Path.Combine(f, name)).FirstOrDefault(File.Exists);
    }

    private static void Throws<TException>(Action action, string what) where TException : Exception
    {
        try
        {
            action();
        }
        catch (TException)
        {
            return;
        }

        throw new InvalidOperationException($"{what}: no {typeof(TException).Name}");
    }

    private static void Check(bool condition, string message)
    {
        if (!condition)
        {
            throw new InvalidOperationException(message);
        }
    }

    private static void CheckClose(float[] expected, float[] actual, float tolerance, string what)
    {
        Check(expected.Length == actual.Length, $"{what}: {actual.Length} values, expected {expected.Length}");
        for (int i = 0; i < expected.Length; i++)
        {
            Check(MathF.Abs(expected[i] - actual[i]) <= tolerance * MathF.Max(1f, MathF.Abs(expected[i])), $"{what}: element {i} is {actual[i]}, expected {expected[i]}");
        }
    }
}
