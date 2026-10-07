// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Onnx.Abstractions;

namespace Idrak.Onnx;

/// <summary>
/// The built-in translators, registered like translators of your own by <see cref="LibraryRegistrations"/>: the export
/// translators in <see cref="OnnxExportOps"/> (layers, the builder's lambdas, graph operations) and the import
/// translators in <see cref="OnnxImportOps"/> (<see cref="Importer.BuiltInOps"/>).
/// </summary>
internal static class OnnxBuiltIns
{
    public static void RegisterImports()
    {
        foreach (var (op, translate) in Importer.BuiltInOps)
        {
            OnnxImportOps.Register(op, translate);
        }
    }

    public static void RegisterExports()
    {
        OnnxExportOps.Register<Layers.Linear>((g, linear, x, shape) => Linear(g, linear, x, shape));
        OnnxExportOps.Register<ReLU>((g, _, x, shape) => g.Node("Relu", [x], shape));
        OnnxExportOps.Register<Tanh>((g, _, x, shape) => g.Node("Tanh", [x], shape));
        OnnxExportOps.Register<Sigmoid>((g, _, x, shape) => g.Node("Sigmoid", [x], shape));
        OnnxExportOps.Register<GELU>((g, _, x, shape) => Gelu(g, x, shape));
        OnnxExportOps.Register<Softmax>((g, _, x, shape) => g.Node("Softmax", [x], shape, OnnxAttribute.Of("axis", -1L)));
        OnnxExportOps.Register<Dropout>((_, _, x, _) => x);
        OnnxExportOps.Register<BatchNorm>((g, bn, x, shape) => g.Node("BatchNormalization",
            [x, Weights(g, "gamma", bn.Gamma), Weights(g, "beta", bn.Beta), Weights(g, "mean", bn.RunningMean), Weights(g, "var", bn.RunningVariance)],
            shape, OnnxAttribute.Of("epsilon", bn.Epsilon)));
        OnnxExportOps.Register<LayerNorm>((g, ln, x, shape) => LayerNorm(g, ln, x, shape));
        OnnxExportOps.Register<Conv2d>((g, conv, x, shape) => g.Node("Conv",
            [x, g.Constant("conv_w", conv.Weight.ToArray(), conv.OutChannels, conv.InChannels, conv.KernelSize, conv.KernelSize),
                conv.Bias is null ? null : Weights(g, "conv_b", conv.Bias)],
            shape, OnnxAttribute.Of("kernel_shape", [conv.KernelSize, conv.KernelSize]), OnnxAttribute.Of("strides", [conv.Stride, conv.Stride]),
            OnnxAttribute.Of("pads", [conv.Padding, conv.Padding, conv.Padding, conv.Padding])));
        OnnxExportOps.Register<MaxPool2d>((g, pool, x, shape) => g.Node("MaxPool", [x], shape,
            OnnxAttribute.Of("kernel_shape", [pool.KernelSize, pool.KernelSize]), OnnxAttribute.Of("strides", [pool.Stride, pool.Stride]),
            OnnxAttribute.Of("pads", [pool.Padding, pool.Padding, pool.Padding, pool.Padding])));
        OnnxExportOps.Register<GlobalAveragePool2d>((g, _, x, shape) => g.Node("Flatten", [g.Node("GlobalAveragePool", [x])], shape, OnnxAttribute.Of("axis", 1L)));
        OnnxExportOps.Register<Layers.Flatten>((g, _, x, shape) => g.Node("Flatten", [x], shape, OnnxAttribute.Of("axis", 1L)));
        OnnxExportOps.Register<Embedding>((g, e, x, shape) => g.Node("Gather",
            [g.Constant("embedding", e.WeightValues(), e.Vocabulary, e.Dim), g.Node("Cast", [x], null, OnnxAttribute.Of("to", 7L))], shape, OnnxAttribute.Of("axis", 0L)));
        OnnxExportOps.Register<PositionalEncoding>((g, pe, x, shape) =>
            g.Node("Add", [x, g.Constant("positions", PositionTable(x.Shape![^2], pe.Dim), x.Shape[^2], pe.Dim)], shape));
        OnnxExportOps.Register<MultiHeadAttention>((g, mha, x, shape) => Attention(g, mha, x, shape));
        OnnxExportOps.Register<TransformerEncoderLayer>((g, layer, x, shape) => EncoderLayer(g, layer, x, shape));
        OnnxExportOps.Register<LSTM>((g, lstm, x, shape) => Recurrent(g, "LSTM", lstm, [0, 3, 1, 2], x, shape));    // ONNX gate order i, o, f, c; ours i, f, c, o
        OnnxExportOps.Register<GRU>((g, gru, x, shape) => Recurrent(g, "GRU", gru, [1, 0, 2], x, shape));            // ONNX z, r, h; ours r, z, h
        OnnxExportOps.Register<Sequential>((g, sequential, x, _) => sequential.Aggregate(x, (value, child) => g.Module(child, value)));
        OnnxExportOps.Register<GraphModule>(Graph);
        OnnxExportOps.Register<Layers.Lambda>((g, lambda, x, shape) => (OnnxExportOps.FindLambda(lambda.ToString())
            ?? throw new NotSupportedException($"The lambda '{lambda}' cannot be exported (registered: {string.Join(", ", OnnxExportOps.LambdaNames)}): "
                + $"register a translator with OnnxExportOps.RegisterLambda(\"{lambda}\", ...) or the exporter's Lambda(\"{lambda}\", ...)."))(g, lambda, x, shape));

        OnnxExportOps.RegisterLambda("MeanOverTime", OnnxExport.MeanOverTime);
        OnnxExportOps.RegisterLambda("FirstStep", OnnxExport.FirstStep);
        OnnxExportOps.RegisterLambda("LastStep", OnnxExport.LastStep);
        OnnxExportOps.RegisterLambda("Reshape", OnnxExport.Reshape);

        RegisterGraphOps();
    }

    // ------------------------------------------------------------------ graphs

    // A GraphModule node by node: its constants become initializers, its layers are translated like any module and its
    // operations by their registered translators. Shapes come from running the graph on a zero sample.
    private static OnnxValue Graph(OnnxGraph g, GraphModule graph, OnnxValue x, IReadOnlyList<int> outputShape)
    {
        var exporter = (g as OnnxGraphWriter)?.Exporter ?? throw new InvalidOperationException("A graph is exported through an OnnxExporter.");
        var device = graph.Parameters().Concat(graph.Buffers()).FirstOrDefault()?.Device ?? Device.Default;
        var trace = graph.Trace(Tensor.Zeros([1, .. x.Shape!.Skip(1)], device));
        var values = new Dictionary<string, OnnxValue>(StringComparer.Ordinal) { [graph.Input] = x };
        var floats = graph.Constants.ToDictionary(c => c.Name, c => c.Value);
        var integers = graph.IntegerConstants.ToDictionary(c => c.Name, c => (c.Values, c.Dims));
        OnnxValue? Value(string name)
        {
            if (name.Length == 0)
            {
                return null;
            }

            if (!values.TryGetValue(name, out var value))
            {
                value = floats.TryGetValue(name, out var tensor) ? g.Constant(name, tensor.ToArray(), tensor.Shape.ToArray())
                    : integers.TryGetValue(name, out var ints) ? g.Constant(name, ints.Values, ints.Dims)
                    : throw new InvalidOperationException($"Value '{name}' is not available.");
                values[name] = value;
            }

            return value;
        }

        long[]? Integers(string name) =>
            integers.TryGetValue(name, out var ints) ? ints.Values
            : trace.TryGetValue(name, out var traced) && traced is not Tensor ? GraphModule.AsIntegers(traced)
            : floats.TryGetValue(name, out var constant) ? GraphModule.AsIntegers(constant)
            : null;

        foreach (var node in graph.Nodes)
        {
            var inputs = node.Inputs.Select(Value).ToList();
            OnnxValue output;
            if (node.Op == "layer")
            {
                output = g.Module(node.Layer!, inputs[0]!);
            }
            else
            {
                var shape = trace[node.Output] is Tensor t ? (IReadOnlyList<int>)[-1, .. t.Shape[1..].ToArray()] : null;
                var translate = exporter.FindGraphOp(node.Op) ?? throw new NotSupportedException(
                    $"The graph operation '{node.Op}' cannot be exported (registered: {string.Join(", ", OnnxExportOps.GraphOpNames)}): "
                    + $"register a translator with OnnxExportOps.RegisterGraphOp(\"{node.Op}\", ...) or the exporter's GraphOp(\"{node.Op}\", ...).");
                output = translate(new OnnxGraphOpContext(g, node, inputs, shape, i => i < node.Inputs.Count && node.Inputs[i].Length > 0 ? Integers(node.Inputs[i]) : null));
            }

            values[node.Output] = output;
        }

        return values[graph.Output];
    }

    // The graph operations as the ONNX operators of the same names (opset 17).
    private static void RegisterGraphOps()
    {
        foreach (var (op, onnx) in new[]
        {
            ("add", "Add"), ("sub", "Sub"), ("mul", "Mul"), ("div", "Div"), ("matmul", "MatMul"), ("relu", "Relu"), ("tanh", "Tanh"),
            ("sigmoid", "Sigmoid"), ("exp", "Exp"), ("log", "Log"), ("abs", "Abs"), ("neg", "Neg"), ("sqrt", "Sqrt"), ("pow", "Pow"),
            ("max", "Max"), ("min", "Min"), ("slice", "Slice"), ("hard_swish", "HardSwish"), ("global_average_pool", "GlobalAveragePool"),
            ("identity", "Identity"),
        })
        {
            OnnxExportOps.RegisterGraphOp(op, c => c.Operator(onnx));
        }

        OnnxExportOps.RegisterGraphOp("cast", c => c.Int("to") is { } to ? c.Operator("Cast", OnnxAttribute.Of("to", to)) : c.Operator("Identity"));
        OnnxExportOps.RegisterGraphOp("softmax", c => c.Operator("Softmax", OnnxAttribute.Of("axis", c.Int("axis") ?? -1)));
        OnnxExportOps.RegisterGraphOp("flatten", c => c.Operator("Flatten", OnnxAttribute.Of("axis", c.Int("axis") ?? 1)));
        OnnxExportOps.RegisterGraphOp("concat", c => c.Operator("Concat", OnnxAttribute.Of("axis", c.Int("axis") ?? 0)));
        OnnxExportOps.RegisterGraphOp("gather", c => c.Operator("Gather", OnnxAttribute.Of("axis", c.Int("axis") ?? 0)));
        OnnxExportOps.RegisterGraphOp("reshape", c => c.Operator("Reshape", OnnxAttribute.Of("allowzero", c.Int("allowzero") ?? 0)));
        OnnxExportOps.RegisterGraphOp("transpose", c => c.Ints("perm") is { } perm ? c.Operator("Transpose", OnnxAttribute.Of("perm", perm)) : c.Operator("Transpose"));
        OnnxExportOps.RegisterGraphOp("shape", c =>
        {
            List<OnnxAttribute> attributes = [];
            if (c.Int("start") is { } start)
            {
                attributes.Add(OnnxAttribute.Of("start", start));
            }

            if (c.Int("end") is { } end)
            {
                attributes.Add(OnnxAttribute.Of("end", end));
            }

            return c.Operator("Shape", [.. attributes]);
        });
        OnnxExportOps.RegisterGraphOp("unsqueeze", c => Axes(c, "Unsqueeze"));
        OnnxExportOps.RegisterGraphOp("squeeze", c => Axes(c, "Squeeze"));
        OnnxExportOps.RegisterGraphOp("reduce_mean", c =>
        {
            var axes = c.Ints("axes") ?? c.Integers(1);
            List<OnnxAttribute> attributes = [OnnxAttribute.Of("keepdims", c.Int("keepdims") ?? 1)];
            if (axes is not null)
            {
                attributes.Add(OnnxAttribute.Of("axes", axes));
            }

            return c.Graph.Node("ReduceMean", [c.Inputs[0]], c.OutputShape, [.. attributes]);         // opset 17: axes is an attribute
        });
        OnnxExportOps.RegisterGraphOp("clip", c =>
        {
            // Opset 11 and later: the bounds are inputs (older files had them as attributes).
            var g = c.Graph;
            OnnxValue? Bound(int index, string attribute) =>
                index < c.Inputs.Count && c.Inputs[index] is { } given ? given : c.Float(attribute) is { } value ? g.Scalar(attribute, value) : null;
            return g.Node("Clip", [c.Inputs[0], Bound(1, "min"), Bound(2, "max")], c.OutputShape);
        });
        OnnxExportOps.RegisterGraphOp("leaky_relu", c => c.Graph.Node("LeakyRelu", [c.Inputs[0]], c.OutputShape, OnnxAttribute.Of("alpha", c.Float("alpha") ?? 0.01f)));
        OnnxExportOps.RegisterGraphOp("elu", c => c.Graph.Node("Elu", [c.Inputs[0]], c.OutputShape, OnnxAttribute.Of("alpha", c.Float("alpha") ?? 1f)));
        OnnxExportOps.RegisterGraphOp("hard_sigmoid", c => c.Graph.Node("HardSigmoid", [c.Inputs[0]], c.OutputShape,
            OnnxAttribute.Of("alpha", c.Float("alpha") ?? 0.2f), OnnxAttribute.Of("beta", c.Float("beta") ?? 0.5f)));
    }

    // Squeeze and Unsqueeze take their axes as an input since opset 13.
    private static OnnxValue Axes(OnnxGraphOpContext c, string op) => c.Ints("axes") is { } axes
        ? c.Graph.Node(op, [c.Inputs[0], c.Graph.Ints("axes", axes)], c.OutputShape)
        : c.Operator(op);

    // ------------------------------------------------------------------ layers

    private static OnnxValue Weights(OnnxGraph g, string hint, Tensor tensor) => g.Constant(hint, tensor.ToArray(), tensor.Shape.ToArray());

    private static OnnxValue Linear(OnnxGraph g, Layers.Linear linear, OnnxValue x, IReadOnlyList<int>? shape)
    {
        var weight = linear.WeightValues();                                      // int8 weights are exported dequantized
        if (linear.Adapter is { } a)
        {
            // The adapter folded in, as MergeLora does.
            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            weight = a.Merge(linear, Tensor.From(weight, [linear.InFeatures, linear.OutFeatures], a.Parameters[0].Device)).ToArray();
        }

        var product = g.Node("MatMul", [x, g.Constant("weight", weight, linear.InFeatures, linear.OutFeatures)]);
        return linear.Bias is null ? product with { Shape = shape } : g.Node("Add", [product, Weights(g, "bias", linear.Bias)], shape);
    }

    // GELU, tanh approximation (as Idrak computes it): 0.5 x (1 + tanh(√(2/π) (x + 0.044715 x³))).
    private static OnnxValue Gelu(OnnxGraph g, OnnxValue x, IReadOnlyList<int>? shape)
    {
        var cube = g.Node("Mul", [g.Node("Mul", [x, x]), x]);
        var inner = g.Node("Mul", [g.Node("Add", [x, g.Node("Mul", [cube, g.Scalar("gelu_c", 0.044715f)])]), g.Scalar("gelu_k", 0.7978845608f)]);
        var gate = g.Node("Add", [g.Node("Tanh", [inner]), g.Scalar("one", 1f)]);
        return g.Node("Mul", [g.Node("Mul", [x, gate]), g.Scalar("half", 0.5f)], shape);
    }

    private static OnnxValue LayerNorm(OnnxGraph g, LayerNorm ln, OnnxValue x, IReadOnlyList<int>? shape) =>
        g.Node("LayerNormalization", [x, Weights(g, "ln_gamma", ln.Gamma), Weights(g, "ln_beta", ln.Beta)], shape,
            OnnxAttribute.Of("axis", -1L), OnnxAttribute.Of("epsilon", ln.Epsilon));

    private static float[] PositionTable(int length, int dim)
    {
        var values = new float[length * dim];
        for (int pos = 0; pos < length; pos++)
        {
            for (int i = 0; i < dim; i++)
            {
                double angle = pos / Math.Pow(10000, 2 * (i / 2) / (double)dim);
                values[pos * dim + i] = (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle));
            }
        }

        return values;
    }

    private static OnnxValue Attention(OnnxGraph g, MultiHeadAttention mha, OnnxValue x, IReadOnlyList<int>? shape)
    {
        var children = mha.Children().ToList();
        var (qkvProjection, outputProjection) = ((Layers.Linear)children[0], (Layers.Linear)children[1]);
        int t = x.Shape![1], d = mha.Dim, h = mha.Heads, dh = d / h;
        var qkv = Linear(g, qkvProjection, x, null);                                            // [N, T, 3D]
        var split = g.Node("Transpose", [g.Node("Reshape", [qkv, g.Ints("shape", -1, t, 3, h, dh)])], null,
            OnnxAttribute.Of("perm", [2L, 0, 3, 1, 4]));                                        // [3, N, H, T, dh]
        OnnxValue Part(long i) => g.Node("Gather", [split, g.Constant("part", [i])], null, OnnxAttribute.Of("axis", 0L));
        var (q, k, v) = (Part(0), Part(1), Part(2));
        var scores = g.Node("Mul", [g.Node("MatMul", [q, g.Node("Transpose", [k], null, OnnxAttribute.Of("perm", [0L, 1, 3, 2]))]),
            g.Scalar("scale", 1f / MathF.Sqrt(dh))]);
        if (mha.Causal)
        {
            var mask = new float[t * t];
            for (int i = 0; i < t; i++)
            {
                for (int j = i + 1; j < t; j++)
                {
                    mask[i * t + j] = -1e9f;
                }
            }

            scores = g.Node("Add", [scores, g.Constant("causal_mask", mask, t, t)]);
        }

        var weights = g.Node("Softmax", [scores], null, OnnxAttribute.Of("axis", -1L));
        var context = g.Node("Reshape", [g.Node("Transpose", [g.Node("MatMul", [weights, v])], null, OnnxAttribute.Of("perm", [0L, 2, 1, 3])),
            g.Ints("shape", -1, t, d)]);                                                        // [N, T, D]
        return Linear(g, outputProjection, context, shape);
    }

    private static OnnxValue EncoderLayer(OnnxGraph g, TransformerEncoderLayer layer, OnnxValue x, IReadOnlyList<int>? shape)
    {
        var c = layer.Children().ToList();
        var (norm1, attention, norm2, ff1, ff2) = ((LayerNorm)c[0], (MultiHeadAttention)c[1], (LayerNorm)c[2], (Layers.Linear)c[3], (Layers.Linear)c[4]);
        var attended = g.Node("Add", [x, Attention(g, attention, LayerNorm(g, norm1, x, x.Shape), x.Shape)], x.Shape);
        var hidden = Linear(g, ff2, Gelu(g, Linear(g, ff1, LayerNorm(g, norm2, attended, x.Shape), null), null), null);
        return g.Node("Add", [attended, hidden], shape);
    }

    // ONNX LSTM/GRU with the time-major layout: X [T, N, I] → Y [T, 1, N, H], Y_h [1, N, H]. Weights are regrouped from
    // Idrak's [I, G·H] (gate blocks in our order) to ONNX's [1, G·H, I] (gate blocks in ONNX's order); the whole
    // bias goes into the input bias (the recurrent bias is zero), which for GRU needs linear_before_reset = 1.
    private static OnnxValue Recurrent(OnnxGraph g, string op, RecurrentModule rnn, int[] order, OnnxValue x, IReadOnlyList<int>? shape)
    {
        int h = rnn.HiddenSize, gates = order.Length;
        float[] Regroup(Tensor weight, int rows)
        {
            var source = weight.ToArray();                                                      // [rows, G·H]
            var result = new float[gates * h * rows];                                           // [G·H, rows]
            for (int gate = 0; gate < gates; gate++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (int r = 0; r < rows; r++)
                    {
                        result[(gate * h + j) * rows + r] = source[r * gates * h + order[gate] * h + j];
                    }
                }
            }

            return result;
        }

        var bias = rnn.Bias.ToArray();
        var b = new float[2 * gates * h];
        for (int gate = 0; gate < gates; gate++)
        {
            Array.Copy(bias, order[gate] * h, b, gate * h, h);
        }

        var timeMajor = g.Node("Transpose", [x], null, OnnxAttribute.Of("perm", [1L, 0, 2]));
        List<OnnxAttribute> attributes = [OnnxAttribute.Of("hidden_size", (long)h)];
        if (op == "GRU")
        {
            attributes.Add(OnnxAttribute.Of("linear_before_reset", 1L));
        }

        var outputs = g.Nodes(op, [timeMajor, g.Constant("rnn_w", Regroup(rnn.InputWeight, rnn.InputSize), 1, gates * h, rnn.InputSize),
            g.Constant("rnn_r", Regroup(rnn.HiddenWeight, h), 1, gates * h, h), g.Constant("rnn_b", b, 1, 2 * gates * h)], 2, [.. attributes]);
        return rnn.ReturnSequences
            ? g.Node("Transpose", [g.Node("Squeeze", [outputs[0], g.Ints("axes", 1)])], shape, OnnxAttribute.Of("perm", [1L, 0, 2]))
            : g.Node("Squeeze", [outputs[1], g.Ints("axes", 0)], shape);
    }
}
