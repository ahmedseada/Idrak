// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Onnx;
using Idrak.Onnx.Abstractions;

// Registries instead of closed switches: the built-in ONNX import operators and network-builder steps go through the
// same registration as steps and operators of your own.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] OpPluginGroup =
    [
        ("op plugins: every built-in network step and ONNX import operator is registered, and a varied network replays to the same JSON and outputs", BuiltInOpsRegistered),
        ("op plugins: a custom network step (a lambda, a layer factory, a composite of registered steps) is written to JSON, replayed and computes like the built-in layers", CustomNetworkOp),
        ("op plugins: a custom ONNX operator imports through a translator onto a custom step, computes, saves and reloads; a built-in operator can be replaced", CustomOnnxImportOp),
        ("op plugins: unknown steps and operators are rejected with the registered names and how to register", UnknownOpErrors),
    ];

    private static void BuiltInOpsRegistered(Device device)
    {
        string[] steps = ["linear", "relu", "tanh", "sigmoid", "gelu", "softmax", "dropout", "batchnorm", "layernorm", "conv2d", "maxpool2d",
            "globalavgpool2d", "flatten", "embedding", "positional", "transformer", "attention", "lstm", "gru", "meanOverTime", "lastStep", "firstStep", "reshape"];
        Check(steps.All(NetworkOps.Names.Contains), $"network steps: {string.Join(", ", NetworkOps.Names)}");
        string[] operators = ["Identity", "Dropout", "Relu", "Tanh", "Sigmoid", "Softmax", "BatchNormalization", "LayerNormalization", "Conv", "MaxPool",
            "GlobalAveragePool", "Flatten", "Cast", "Gather", "Add", "ReduceMean", "Reshape"];
        Check(operators.All(OnnxImportOps.Names.Contains), $"ONNX operators: {string.Join(", ", OnnxImportOps.Names)}");

        var r = new Random(41);
        var builders = new[]
        {
            Network.Input(6).Seed(42).Linear(8, bias: false).BatchNorm().GELU().Dropout(0.2f).Linear(5).LayerNorm().Tanh().Linear(3).Sigmoid().Softmax(),
            Network.Image(2, 9, 9).Seed(43).Conv2d(4, 3, stride: 2, padding: 1).ReLU().MaxPool2d(2, padding: 1).Conv2d(3, 1).GlobalAveragePool2d(),
            Network.Image(1, 6, 6).Seed(44).MaxPool2d(2, stride: 2).Flatten().Reshape(3, 3).LSTM(4, returnSequences: true).GRU(5).Linear(2),
            Network.Tokens(5).Seed(45).Embedding(11, 8).PositionalEncoding(maxLength: 7).TransformerEncoderLayer(2, ffDim: 12, dropout: 0f, causal: true)
                .MultiHeadAttention(2, causal: false).FirstStep().Reshape(2, 4).MeanOverTime(),
            Network.Sequence(4, 3).Seed(46).LSTM(6, returnSequences: true).LastStep().Linear(2),
        };
        foreach (var builder in builders)
        {
            var json = builder.ToJson();
            var replayed = Network.FromJson(json);
            Check(JsonNode.DeepEquals(replayed.ToJson(), json), $"{builder}: replayed JSON {replayed.ToJson().ToJsonString()} vs {json.ToJsonString()}");
            using var original = builder.OnDevice(device).Build();
            using var again = replayed.OnDevice(device).Build();
            original.Eval();
            again.Eval();
            int[] shape = [2, .. builder.InputShape];
            using var input = builder.InputKind == InputKind.Tokens ? RandomIds(device, r, 11, shape) : RandomInput(device, r, shape);
            using var expected = original.Predict(input);
            using var actual = again.Predict(input);
            AssertClose(expected.ToArray(), actual.ToArray(), 0f, $"{builder}: replayed outputs");
        }
    }

    private static void CustomNetworkOp(Device device)
    {
        try
        {
            NetworkOps.Register("scale", (b, a) =>
            {
                float factor = a.Float("factor");
                return b.Lambda(x => x * factor, "Scale", [.. b.CurrentShape]);
            });
            NetworkOps.Register("dense", (b, a) =>
            {
                int inputs = b.CurrentShape[^1], outputs = a.Int("out");
                return b.Add((d, random) => new Linear(inputs, outputs, a.Has("bias") && a.Bool("bias"), d, random), [.. b.CurrentShape.SkipLast(1), outputs]);
            });

            var builder = Network.Input(4).Seed(7).Linear(3).Op("scale", new JsonObject { ["factor"] = 2.5f }).ReLU().Op("dense", new JsonObject { ["out"] = 2 });
            Check(builder.IsDescribable && builder.Count == 4, $"{builder} describable {builder.IsDescribable}");
            var json = builder.ToJson();
            var ops = json["steps"]!.AsArray().Select(s => (string)s!["op"]!).ToArray();
            Check(ops.SequenceEqual(["linear", "scale", "relu", "dense"]), string.Join(", ", ops));
            Check((float)json["steps"]![1]!["factor"]! == 2.5f, json.ToJsonString());

            var replayed = Network.FromJson(json);
            Check(JsonNode.DeepEquals(replayed.ToJson(), json), $"replayed JSON {replayed.ToJson().ToJsonString()}");
            using var model = builder.OnDevice(device).Build();
            using var again = replayed.OnDevice(device).Build();
            Check(Network.ArchitectureOf(model) is not null, "the built model keeps its architecture");

            // The same network written with built-in layers: the factory gets the builder's device and random, so the
            // weights are the same.
            using var reference = Network.Input(4).Seed(7).OnDevice(device).Linear(3).Lambda(x => x * 2.5f, "Scale", [3]).ReLU().Linear(2, bias: false).Build();
            Check(model.Last().Parameters().All(p => p.Device == device), "the factory's layer is on the builder's device");
            var input = RandomInput(device, new Random(8), 5, 4);
            using var expected = reference.Predict(input);
            using var actual = model.Predict(input);
            using var replayedOutput = again.Predict(input);
            AssertClose(expected.ToArray(), actual.ToArray(), 1e-5f, "custom steps compute like the built-in layers");
            AssertClose(actual.ToArray(), replayedOutput.ToArray(), 0f, "the replayed network computes the same");

            // A step made of registered steps, through the builder contract alone: the built-in steps describe their
            // layers, so the JSON lists them and replays without the composite step.
            NetworkOps.Register("block", (b, a) => b.Op("linear", new JsonObject { ["out"] = a.Int("width"), ["bias"] = true }).Op("relu"));
            var composite = Network.Input(4).Seed(9).Op("block", new JsonObject { ["width"] = 3 });
            var compositeJson = composite.ToJson();
            Check(composite.CurrentShape.SequenceEqual([3]) && composite.Count == 2, composite.ToString());
            Check(compositeJson["steps"]!.AsArray().Select(s => (string)s!["op"]!).SequenceEqual(["linear", "relu"]), compositeJson.ToJsonString());
            Check(JsonNode.DeepEquals(Network.Input(4).Seed(9).Linear(3).ReLU().ToJson(), compositeJson), "the composite step writes the built-in steps");
            Check(NetworkOps.Contains("block") && !NetworkOps.Contains("nope"), "Contains");
        }
        finally
        {
            NetworkOps.Unregister("scale");
            NetworkOps.Unregister("dense");
            NetworkOps.Unregister("block");
        }
    }

    private static void CustomOnnxImportOp(Device device)
    {
        var relu = OnnxImportOps.Get("Relu");
        try
        {
            using var model = new Sequential { new Linear(4, 3, device: device, random: new Random(31)), new Lambda(x => x * 3f, "Triple") };
            model.Eval();
            // Exported as a made-up "Scale" operator with a factor attribute (no Idrak layer imports it until one is registered).
            var bytes = OnnxExport.For(model).Input(4)
                .Lambda("Triple", (g, _, x, shape) => g.Node("Scale", [x], shape, OnnxAttribute.Of("factor", 3f))).ToBytes();
            try
            {
                OnnxImport.Load(bytes, device).Dispose();
                Check(false, "the unregistered Scale operator must be rejected");
            }
            catch (NotSupportedException ex)
            {
                Check(ex.Message.Contains("Scale") && ex.Message.Contains("OnnxImportOps.Register"), ex.Message);
            }

            NetworkOps.Register("scale", (b, a) =>
            {
                float factor = a.Float("factor");
                return b.Lambda(x => x * factor, "Scale", [.. b.CurrentShape]);
            });
            OnnxImportOps.Register("Scale", c => c.Add("scale", new JsonObject { ["factor"] = c.Float("factor", 1f) }));
            var input = RandomInput(device, new Random(32), 6, 4);
            using var expected = model.Predict(input);
            string path = Path.Combine(Path.GetTempPath(), $"ns-{Guid.NewGuid():N}.ikm");
            try
            {
                using (var imported = OnnxImport.Load(bytes, device))
                {
                    Check(!imported.IsGraph && imported.Network is not null, "a chain of layers");
                    var layers = ((Sequential)imported.Model).Select(m => m.ToString()).ToArray();
                    Check(layers.Length == 2 && layers[1]!.Contains("Scale"), string.Join(", ", layers));
                    using var actual = imported.Model.Predict(input);
                    AssertClose(expected.ToArray(), actual.ToArray(), 1e-5f, "imported custom operator");
                    imported.SavePackage(path);                                 // the custom step is described, so it is saved
                }

                using var predictor = Predictor.Load(path, device).Build();
                var p = predictor.Predict([0.1f, -0.2f, 0.3f, 0.4f]);
                using var single = Tensor.From([0.1f, -0.2f, 0.3f, 0.4f], [1, 4], device);
                using var reference = model.Predict(single);
                AssertClose(reference.ToArray(), p, 1e-5f, "the package replays the custom step");
            }
            finally
            {
                File.Delete(path);
            }

            // Replacing a built-in operator: Relu imported as Tanh.
            OnnxImportOps.Register("Relu", c => c.Add("tanh"));
            using var small = Network.Input(3).OnDevice(device).Seed(33).Linear(2).ReLU().Build();
            using (var replaced = OnnxImport.Load(OnnxExport.For(small).Input(3).ToBytes(), device))
            {
                var names = ((Sequential)replaced.Model).Select(m => m.GetType().Name).ToArray();
                Check(names.SequenceEqual(["Linear", "Tanh"]), string.Join(", ", names));
            }
        }
        finally
        {
            OnnxImportOps.Unregister("Relu");
            OnnxImportOps.Unregister("Scale");
            NetworkOps.Unregister("scale");
        }

        Check(OnnxImportOps.Get("Relu") == relu && !OnnxImportOps.Names.Contains("Scale"), "registrations restored");
    }

    private static void UnknownOpErrors(Device device)
    {
        var json = Network.Input(3).Linear(2).ToJson();
        json["steps"]!.AsArray().Add(new JsonObject { ["op"] = "nope" });
        try
        {
            Network.FromJson(json);
            Check(false, "an unknown step must be rejected");
        }
        catch (InvalidDataException ex)
        {
            Check(ex.Message.Contains("'nope'") && ex.Message.Contains("linear") && ex.Message.Contains("NetworkOps.Register"), ex.Message);
        }

        try
        {
            Network.Input(3).OnDevice(device).Op("nope");
            Check(false, "an unknown step must be rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("'nope'") && ex.Message.Contains("relu") && ex.Message.Contains("NetworkOps.Register"), ex.Message);
        }

        try
        {
            var missing = Network.Input(3).Linear(2).ToJson();
            missing["steps"]![0]!.AsObject().Remove("out");
            Network.FromJson(missing);
            Check(false, "a step without a required argument must be rejected");
        }
        catch (InvalidDataException ex)
        {
            Check(ex.Message.Contains("'linear'") && ex.Message.Contains("'out'"), ex.Message);
        }

        try
        {
            OnnxImportOps.Get("Nope");
            Check(false, "an unknown operator must be rejected");
        }
        catch (NotSupportedException ex)
        {
            Check(ex.Message.Contains("'Nope'") && ex.Message.Contains("Relu") && ex.Message.Contains("OnnxImportOps.Register"), ex.Message);
        }
    }
}
