// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Abstraction.Operations;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Layers.Abstractions;

namespace Idrak.PluginTests;

/// <summary>
/// The plug-ins' own device kernels (<see cref="PluginKernels"/>): the packed format, the cache layout, an
/// <see cref="Autograd.Function"/> and a graph operation each run their plug-in operation, through its default kernel or
/// through the kernel <see cref="PluginKernels.Install"/> registers for the device's kind (the CPU; Vulkan for the packed
/// format and the cache layout), seen in <see cref="Kernels.Chain"/> and counted by <see cref="Kernels.Trace"/>. The test
/// runner in tests/Idrak.Tests runs <see cref="All"/> on every device; a failed check throws.
/// </summary>
public static class KernelPluginTests
{
    /// <summary>The tests, by name.</summary>
    public static IReadOnlyList<(string Name, Action<Device> Run)> All { get; } =
    [
        ("outside plug-in kernels: the packed format unpacks through its plug-in operation, by default and through the kernel registered for the device (CPU, Vulkan SPIR-V), to the same weights and products", PackedFormatKernel),
        ("outside plug-in kernels: the key/value cache layout expands its rows through its plug-in operation, by default and through the kernel registered for the device (CPU, Vulkan SPIR-V), and generates the float32 cache's text", CacheLayoutKernel),
        ("outside plug-in kernels: an Autograd.Function and a graph operation run their plug-in operations, by default and through the CPU kernel, with their gradients", FunctionAndGraphOpKernels),
    ];

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
        bool registers = backend.Kind is "cpu" or "vulkan";
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
            var source = installed && backend.Kind is "cpu" or "vulkan" ? KernelSource.Registered : KernelSource.Composed;
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
