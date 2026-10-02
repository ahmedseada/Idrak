// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Onnx;
using Idrak.Onnx.Runtime;

// ONNX export registry, graph export (import → export → import), graph operations and layer types of your own.
internal static partial class Tests
{
    // y = x · factor, a custom layer exported through a registered translator.
    private sealed class Times(float factor) : Module
    {
        public float Factor { get; } = factor;

        protected override Tensor ForwardCore(Tensor input) => input * Factor;
    }

    // y = x + shift (a learned vector), a custom layer with weights for the layer type registry.
    private sealed class Shift : Module
    {
        public Shift(int features, Device device)
        {
            Features = features;
            Offset = CreateParameter([.. Enumerable.Range(0, features).Select(i => 0.1f * i - 0.2f)], [features], device);
        }

        public int Features { get; }

        public Tensor Offset { get; }

        public override IEnumerable<Tensor> Parameters() => [Offset];

        protected override Tensor ForwardCore(Tensor input) => input + Offset;
    }

    // relu(x) + 1.5 · tanh(linear(x)) · 2 + x: a skip connection around a custom ONNX operator (ScaledTanh) and a
    // custom builder step (Scale), as an outside exporter would write it.
    private sealed class CustomSkip(Linear linear) : Module
    {
        public Linear Linear { get; } = linear;

        public override IEnumerable<Module> Children() => [Linear];

        protected override Tensor ForwardCore(Tensor input) => input.Relu() + Linear.Forward(input).Tanh() * 1.5f * 2f + input;
    }

    private static void OnnxExportRegistry(Device device)
    {
        string[] types = ["Linear", "Conv2d", "BatchNorm", "LSTM", "GRU", "Sequential", "GraphModule", "Lambda", "TransformerEncoderLayer"];
        Check(types.All(t => OnnxExportOps.Types.Any(r => r.Name == t)), $"export types: {string.Join(", ", OnnxExportOps.Types.Select(t => t.Name))}");
        Check(new[] { "MeanOverTime", "FirstStep", "LastStep", "Reshape" }.All(OnnxExportOps.LambdaNames.Contains), "builder lambdas registered");
        Check(new[] { "add", "relu", "concat", "shape", "reshape", "clip", "pow" }.All(OnnxExportOps.GraphOpNames.Contains), "graph operations registered");

        var r = new Random(50);
        using var model = new Sequential { new Linear(4, 3, device: device, random: r), new Times(2f) };
        model.Eval();
        var x = RandomInput(device, r, 2, 4);
        try
        {
            OnnxExport.For(model).Input(4).ToBytes();
            Check(false, "an unregistered module must be rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("Times") && ex.Message.Contains("OnnxExportOps.Register") && ex.Message.Contains("Linear"), ex.Message);
        }

        var linear = OnnxExportOps.Get<Linear>();
        try
        {
            OnnxExportOps.Register<Times>((g, m, v, shape) => g.Node("Mul", [v, g.Scalar("factor", m.Factor)], shape));
            CheckOnnx(model, [4], x, "a globally registered translator");

            // The exporter's own translator takes precedence (here a wrong one, to see it is used).
            using var onnx = OnnxModule.Load(OnnxExport.For(model).Input(4).Module<Times>((g, _, v, shape) => g.Node("Mul", [v, g.Scalar("f", 3f)], shape)).ToBytes());
            using var expected = model.Predict(x);
            using var actual = onnx.Predict(x);
            AssertClose([.. expected.ToArray().Select(v => v * 1.5f)], actual.ToArray(), 1e-4f, "the exporter's translator takes precedence");

            // A built-in can be replaced, and restored.
            OnnxExportOps.Register<Linear>((g, l, v, shape) => g.Node("Mul", [linear(g, l, v, shape), g.Scalar("twice", 2f)], shape));
            using var replaced = OnnxModule.Load(OnnxExport.For(model).Input(4).ToBytes());
            using var doubled = replaced.Predict(x);
            AssertClose([.. expected.ToArray().Select(v => v * 2f)], doubled.ToArray(), 1e-4f, "the replaced Linear is used");
        }
        finally
        {
            OnnxExportOps.Register(linear);
            OnnxExportOps.Unregister<Times>();
        }

        Check(OnnxExportOps.Get<Linear>() == linear && !OnnxExportOps.Types.Contains(typeof(Times)), "registrations restored");
        CheckOnnx(new Sequential { new Linear(4, 3, device: device, random: r) }, [4], x, "the restored Linear");
        try
        {
            OnnxExportOps.GetGraphOp("nope");
            Check(false, "an unknown graph operation must be rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("'nope'") && ex.Message.Contains("relu") && ex.Message.Contains("RegisterGraphOp"), ex.Message);
        }
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

    private static void OnnxGraphOpPlugin(Device device)
    {
        var r = new Random(52);
        using var model = new Sequential { new Linear(5, 6, device: device, random: r), new CustomSkip(new Linear(6, 6, device: device, random: r)), new Linear(6, 3, device: device, random: r) };
        model.Eval();
        // The custom block written as an outside exporter would: ScaledTanh (a custom operator), then a Scale node
        // (another custom operator imported as a builder step), around a skip connection.
        var bytes = OnnxExport.For(model).Input(5).Module<CustomSkip>((g, block, x, shape) =>
        {
            var hidden = g.Module(block.Linear, x);
            var scaled = g.Node("Scale", [g.Node("ScaledTanh", [hidden], hidden.Shape, OnnxAttribute.Of("scale", 1.5f))], hidden.Shape, OnnxAttribute.Of("factor", 2f));
            return g.Node("Add", [g.Node("Add", [g.Node("Relu", [x]), scaled]), x], shape);
        }).ToBytes();

        try
        {
            OnnxImport.Load(bytes, device).Dispose();
            Check(false, "an unregistered operator in a graph must be rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("ScaledTanh") && ex.Message.Contains("OnnxImportOps.Register"), ex.Message);
        }

        try
        {
            GraphOps.Register("scaled_tanh", c => c.Input(0).Tanh() * c.Float("scale", 1f));
            OnnxImportOps.Register("ScaledTanh", c => c.AddGraphOp("scaled_tanh"));
            NetworkOps.Register("scale", (b, a) =>
            {
                float factor = a.Float("factor");
                return b.Lambda(v => v * factor, "Scale", [.. b.CurrentShape]);
            });
            OnnxImportOps.Register("Scale", c => c.Add(b => b.Op("scale", new JsonObject { ["factor"] = c.Float("factor", 1f) })));

            using var imported = OnnxImport.Load(bytes, device);
            var graph = (GraphModule)imported.Model;
            Check(imported.IsGraph && graph.Nodes.Any(n => n.Op == "scaled_tanh") && graph.Nodes.Any(n => n.Layer is Sequential),
                string.Join(", ", graph.Nodes.Select(n => n.Layer?.ToString() ?? n.Op)));
            var x = RandomInput(device, r, 4, 5);
            using var expected = model.Predict(x);
            using (var actual = graph.Predict(x))
            {
                AssertClose(expected.ToArray(), actual.ToArray(), 1e-4f, "the custom operators compute what the reference does");
            }

            // The graph's JSON names the operation and the builder step; it rebuilds and computes the same.
            var json = JsonNode.Parse(graph.ToJson().ToJsonString())!.AsObject();
            using var rebuilt = GraphModule.FromJson(json, device);
            using (var weights = new MemoryStream())
            {
                graph.Save(weights);
                weights.Position = 0;
                rebuilt.Load(weights);
            }

            using (var actual = rebuilt.Predict(x))
            {
                AssertClose(expected.ToArray(), actual.ToArray(), 1e-4f, "rebuilt from JSON");
            }

            // Exported again with translators for the custom operation and lambda, then imported once more.
            OnnxExportOps.RegisterGraphOp("scaled_tanh", c => c.Graph.Node("ScaledTanh", c.Inputs, c.OutputShape, OnnxAttribute.Of("scale", c.Float("scale") ?? 1f)));
            OnnxExportOps.RegisterLambda("Scale", (g, _, v, shape) => g.Node("Scale", [v], shape, OnnxAttribute.Of("factor", 2f)));
            using var again = OnnxImport.Load(OnnxExport.For(graph).Input(5).ToBytes(), device);
            using (var actual = again.Model.Predict(x))
            {
                AssertClose(expected.ToArray(), actual.ToArray(), 1e-4f, "import → export → import with custom operators");
            }
        }
        finally
        {
            GraphOps.Unregister("scaled_tanh");
            OnnxImportOps.Unregister("ScaledTanh");
            OnnxImportOps.Unregister("Scale");
            NetworkOps.Unregister("scale");
            OnnxExportOps.UnregisterGraphOp("scaled_tanh");
            OnnxExportOps.UnregisterLambda("Scale");
        }

        try
        {
            GraphOps.Register("add", c => c.Input(0));
            Check(false, "a structural operation cannot be replaced");
        }
        catch (ArgumentException ex)
        {
            Check(ex.Message.Contains("'add'"), ex.Message);
        }
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

    private static void GraphLayerTypes(Device device)
    {
        Check(new[] { "linear", "conv2d", "batchnorm", "sequential", "lstm" }.All(LayerTypes.Names.Contains), string.Join(", ", LayerTypes.Names));
        var r = new Random(54);
        var linear = new Linear(4, 4, device: device, random: r);
        var shift = new Shift(4, device);
        using var graph = new GraphModule("x", "y",
        [
            new GraphNode("layer", ["x"], "h", Layer: linear),
            new GraphNode("layer", ["h"], "s", Layer: shift),
            new GraphNode("add", ["s", "x"], "y"),
        ]);
        try
        {
            graph.ToJson();
            Check(false, "an unregistered layer type cannot be described");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("Shift") && ex.Message.Contains("LayerTypes.Register"), ex.Message);
        }

        string path = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}.ikm");
        try
        {
            LayerTypes.Register<Shift>("shift", s => new() { ["features"] = s.Features }, (d, dev) => new Shift((int)d["features"]!, dev));
            var json = JsonNode.Parse(graph.ToJson().ToJsonString())!.AsObject();
            Check(json.ToJsonString().Contains("\"type\":\"shift\""), json.ToJsonString());
            using var rebuilt = GraphModule.FromJson(json, device);
            using (var weights = new MemoryStream())
            {
                graph.Save(weights);
                weights.Position = 0;
                rebuilt.Load(weights);
            }

            var x = RandomInput(device, r, 3, 4);
            using var expected = graph.Predict(x);
            using (var actual = rebuilt.Predict(x))
            {
                AssertClose(expected.ToArray(), actual.ToArray(), 0f, "a custom layer survives the graph's JSON");
            }

            ModelPackage.Create(path).Architecture(ModelPackage.DefaultModelName, graph.ToJson()).Weights(ModelPackage.DefaultModelName, graph).Save();
            using var package = ModelPackage.Open(path);
            using var loaded = package.BuildModel(device: device);
            using (var actual = loaded.Predict(x))
            {
                AssertClose(expected.ToArray(), actual.ToArray(), 0f, "and a model package");
            }
        }
        finally
        {
            LayerTypes.Unregister("shift");
            File.Delete(path);
        }

        Check(!LayerTypes.Names.Contains("shift"), "unregistered");
    }
}
