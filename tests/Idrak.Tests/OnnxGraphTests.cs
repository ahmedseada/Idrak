// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Onnx;
using Idrak.Onnx.Abstractions;
using Idrak.Onnx.Runtime;

// ONNX graph export (import → export → import) and graph operations.
internal abstract partial class DesktopTests : Tests
{
    // y = x · factor, a custom layer exported through the exporter's translator.
    private sealed class Times(float factor) : Module
    {
        public float Factor { get; } = factor;

        protected override Tensor ForwardCore(Tensor input) => input * Factor;
    }

    private static void OnnxGraphRoundTrip(Device device)
    {
        var r = new Random(51);
        using var model = new Sequential
        {
            new Conv2d(3, 8, 3, 1, 1, device: device, random: r), new BatchNorm(8, device: device), new ReLU(),
            new Residual(8, 8, 1, device, r),
            new Residual(8, 12, 2, device, r),
            new Branches(new Conv2d(12, 4, 1, device: device, random: r), new Conv2d(12, 4, 3, 1, 1, device: device, random: r)),
            new MaxPool2d(2), new Flatten(), new Linear(8 * 2 * 2, 5, device: device, random: r), new Softmax(),
        };
        WarmBatchNorm(model, RandomInput(device, r, 16, 3, 8, 8));
        using var imported = OnnxImport.Load(PyTorchStyle(OnnxExport.For(model).Input(3, 8, 8)).ToBytes(), device);
        var graph = (GraphModule)imported.Model;

        // The imported graph exported again: ONNX Runtime and a second import compute what the original does.
        var bytes = OnnxExport.For(graph).Input(3, 8, 8).ToBytes();
        using var onnx = OnnxModule.Load(bytes);
        using var again = OnnxImport.Load(bytes, device);
        Check(again.IsGraph, "the exported graph imports as a graph again");
        // The graph output is an Identity node in every export, so only those differ.
        var ops = graph.Nodes.Where(n => n.Op != "identity").Select(n => n.Layer?.GetType().Name ?? n.Op).Order().ToList();
        var opsAgain = ((GraphModule)again.Model).Nodes.Where(n => n.Op != "identity").Select(n => n.Layer?.GetType().Name ?? n.Op).Order().ToList();
        Check(ops.SequenceEqual(opsAgain), $"the same nodes: {string.Join(", ", ops)} vs {string.Join(", ", opsAgain)}");
        foreach (int batch in new[] { 1, 3 })
        {
            var x = RandomInput(device, r, batch, 3, 8, 8);
            using var expected = model.Predict(x);
            using var runtime = onnx.Predict(x);
            using var reimported = again.Model.Predict(x);
            AssertClose(expected.ToArray(), runtime.ToArray(), 1e-4f, $"batch {batch}: ONNX Runtime on the exported graph");
            AssertClose(expected.ToArray(), reimported.ToArray(), 1e-4f, $"batch {batch}: the graph imported again");
        }

        // Inside a Sequential too, and from its JSON description with the weights.
        var json = JsonNode.Parse(graph.ToJson().ToJsonString())!.AsObject();
        using var rebuilt = GraphModule.FromJson(json, device);
        using (var weights = new MemoryStream())
        {
            graph.Save(weights);
            weights.Position = 0;
            rebuilt.Load(weights);
        }

        using var wrapped = new Sequential { rebuilt, new Times(1f) };
        wrapped.Eval();
        var input = RandomInput(device, r, 2, 3, 8, 8);
        using var reference = model.Predict(input);
        using var fromJson = OnnxModule.Load(OnnxExport.For(wrapped).Input(3, 8, 8).Module<Times>((g, _, v, shape) => g.Node("Identity", [v], shape)).ToBytes());
        using var output = fromJson.Predict(input);
        AssertClose(reference.ToArray(), output.ToArray(), 1e-4f, "a graph rebuilt from JSON, inside a Sequential, exported");
    }

    private static void OnnxGraphCommonOps(Device device)
    {
        var r = new Random(53);
        using var model = new Sequential { new Linear(6, 6, device: device, random: r), new Lambda(v => v, "Ops") };
        model.Eval();
        // Clip, Pow, Sqrt, Neg, LeakyRelu, Elu, HardSigmoid, HardSwish, Max and Min, checked against ONNX Runtime.
        var bytes = OnnxExport.For(model).Input(6).Lambda("Ops", (g, _, x, shape) =>
        {
            var clipped = g.Node("Clip", [x, g.Scalar("lo", -0.5f), g.Scalar("hi", 0.4f)]);
            var squared = g.Node("Pow", [x, g.Scalar("two", 2f)]);
            var root = g.Node("Sqrt", [g.Node("Add", [squared, g.Scalar("one", 1f)])]);
            var cube = g.Node("Pow", [g.Node("Neg", [x]), g.Scalar("three", 3f)]);
            var leaky = g.Node("LeakyRelu", [x], null, OnnxAttribute.Of("alpha", 0.1f));
            var elu = g.Node("Elu", [x], null, OnnxAttribute.Of("alpha", 0.7f));
            var hard = g.Node("HardSigmoid", [x], null, OnnxAttribute.Of("alpha", 0.3f), OnnxAttribute.Of("beta", 0.4f));
            var swish = g.Node("HardSwish", [g.Node("Mul", [x, g.Scalar("four", 4f)])]);
            var max = g.Node("Max", [clipped, leaky]);
            var min = g.Node("Min", [elu, hard]);
            var sum = new[] { root, cube, swish, max }.Aggregate(min, (a, b) => g.Node("Add", [a, b]));
            return g.Node("Identity", [sum], shape);
        }).ToBytes();
        using var onnx = OnnxModule.Load(bytes);
        using var imported = OnnxImport.Load(bytes, device);
        Check(imported.IsGraph, "the operators make a graph");
        var x = RandomInput(device, r, 3, 6) * 3f;
        using var expected = onnx.Predict(x);
        using var actual = imported.Model.Predict(x);
        AssertClose(expected.ToArray(), actual.ToArray(), 1e-4f, "common operators match ONNX Runtime");
        using var exported = OnnxModule.Load(OnnxExport.For(imported.Model).Input(6).ToBytes());
        using var again = exported.Predict(x);
        AssertClose(expected.ToArray(), again.ToArray(), 1e-4f, "and are exported again");
    }
}
