// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Onnx;

/// <summary>
/// Translates one module into ONNX nodes: <paramref name="input"/> is the module's input value, <paramref name="outputShape"/>
/// the shape it produces (measured by running the module, -1 for the batch dimension). Returns the output value.
/// </summary>
public delegate OnnxValue OnnxTranslator<in T>(OnnxGraph graph, T module, OnnxValue input, IReadOnlyList<int> outputShape) where T : Module;

/// <summary>Exports Idrak networks to ONNX. Start with <see cref="For"/>, or use <see cref="ExportOnnx"/>.</summary>
public static class OnnxExport
{
    /// <summary>Starts an export of <paramref name="model"/>; <see cref="OnnxExporter.Input"/> is required.</summary>
    public static OnnxExporter For(Module model) => new(model);

    /// <summary>Exports <paramref name="model"/> for inputs of one sample shaped <paramref name="sampleShape"/> (the batch dimension stays dynamic).</summary>
    public static void ExportOnnx(this Module model, string path, params int[] sampleShape) => For(model).Input(sampleShape).Save(path);

    /// <summary>
    /// The builder's shape helpers as translators, for your own lambdas that do the same thing (for example a
    /// "ClsToken" lambda that keeps the first time step: <c>.Lambda("ClsToken", OnnxExport.FirstStep)</c>).
    /// </summary>
    public static OnnxValue MeanOverTime(OnnxGraph graph, Module module, OnnxValue input, IReadOnlyList<int> outputShape) =>
        graph.Node("ReduceMean", [input], outputShape, OnnxAttribute.Of("axes", [1L]), OnnxAttribute.Of("keepdims", 0L));

    /// <summary>Keeps time step 0: [N, T, F] → [N, F].</summary>
    public static OnnxValue FirstStep(OnnxGraph graph, Module module, OnnxValue input, IReadOnlyList<int> outputShape) =>
        graph.Node("Gather", [input, graph.Constant("step", [0L])], outputShape, OnnxAttribute.Of("axis", 1L));

    /// <summary>Keeps the last time step: [N, T, F] → [N, F].</summary>
    public static OnnxValue LastStep(OnnxGraph graph, Module module, OnnxValue input, IReadOnlyList<int> outputShape) =>
        graph.Node("Gather", [input, graph.Constant("step", [(long)input.Shape![1] - 1])], outputShape, OnnxAttribute.Of("axis", 1L));

    /// <summary>Reshapes each sample to the measured output shape.</summary>
    public static OnnxValue Reshape(OnnxGraph graph, Module module, OnnxValue input, IReadOnlyList<int> outputShape) =>
        graph.Node("Reshape", [input, graph.Ints("shape", [-1, .. outputShape.Skip(1).Select(d => (long)d)])], outputShape);
}

/// <summary>
/// Configures an ONNX export. Supported layers: Linear (LoRA adapters are merged), activations, Softmax, Dropout
/// (removed), BatchNorm, LayerNorm, Conv2d, MaxPool2d, GlobalAveragePool2d, Flatten, Embedding, PositionalEncoding,
/// MultiHeadAttention, TransformerEncoderLayer, LSTM, GRU, Sequential, and the builder's MeanOverTime, FirstStep,
/// LastStep and Reshape lambdas. Other lambdas and custom modules need a translator (<see cref="Lambda"/>,
/// <see cref="Module{T}"/>). Inputs are float32 (token ids too, as in Idrak); the batch dimension is dynamic.
/// </summary>
public sealed class OnnxExporter
{
    private readonly Module _model;
    private readonly Dictionary<string, OnnxTranslator<Module>> _lambdas = new()
    {
        ["MeanOverTime"] = OnnxExport.MeanOverTime,
        ["FirstStep"] = OnnxExport.FirstStep,
        ["LastStep"] = OnnxExport.LastStep,
        ["Reshape"] = OnnxExport.Reshape,
    };

    private readonly List<(Type Type, OnnxTranslator<Module> Translate)> _modules = [];
    private readonly Dictionary<string, string> _metadata = [];
    private int[]? _sampleShape;
    private string _inputName = "input";
    private string _outputName = "output";

    internal OnnxExporter(Module model) => _model = model;

    /// <summary>The shape of one input sample, without the batch dimension (for example [9], [1, 16, 16] or [sequenceLength]).</summary>
    public OnnxExporter Input(params int[] sampleShape)
    {
        if (sampleShape.Length == 0 || sampleShape.Any(d => d <= 0))
        {
            throw new ArgumentException("The sample shape needs at least one positive dimension.", nameof(sampleShape));
        }

        _sampleShape = sampleShape;
        return this;
    }

    /// <summary>Names of the graph's input and output ("input" and "output" unless set).</summary>
    public OnnxExporter Names(string input, string output)
    {
        _inputName = input;
        _outputName = output;
        return this;
    }

    /// <summary>A translator for the lambdas named <paramref name="name"/> (their <c>ToString()</c>).</summary>
    public OnnxExporter Lambda(string name, OnnxTranslator<Module> translate)
    {
        _lambdas[name] = translate;
        return this;
    }

    /// <summary>A translator for modules of type <typeparamref name="T"/> (a custom layer); it takes precedence over the built-ins.</summary>
    public OnnxExporter Module<T>(OnnxTranslator<T> translate) where T : Module
    {
        _modules.Add((typeof(T), (g, m, x, s) => translate(g, (T)m, x, s)));
        return this;
    }

    /// <summary>A key/value pair stored in the model's metadata (for example the class names or the tokenizer).</summary>
    public OnnxExporter Metadata(string key, string value)
    {
        _metadata[key] = value;
        return this;
    }

    /// <summary>Writes the .onnx file.</summary>
    public void Save(string path) => File.WriteAllBytes(path, ToBytes());

    /// <summary>Writes the model to <paramref name="stream"/>.</summary>
    public void Save(Stream stream) => stream.Write(ToBytes());

    /// <summary>The model as the bytes of an .onnx file.</summary>
    public byte[] ToBytes()
    {
        var sampleShape = _sampleShape ?? throw new InvalidOperationException("Call Input(sampleShape) with the shape of one input sample.");
        var graph = new OnnxGraph();
        bool wasTraining = _model.IsTraining;
        _model.Eval();
        try
        {
            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            var device = _model.Parameters().FirstOrDefault()?.Device ?? Device.Default;
            int[] inputShape = [-1, .. sampleShape];
            var input = new OnnxValue(graph.Unique(_inputName), inputShape);
            var (value, sample) = Emit(graph, _model, input, Tensor.Zeros([1, .. sampleShape], device));
            int[] outputShape = [-1, .. sample.Shape[1..].ToArray()];
            var output = graph.Output(value, _outputName, outputShape);
            return graph.ToModel(input, inputShape, output, outputShape, _model.DisplayName, "Idrak",
                typeof(Module).Assembly.GetName().Version?.ToString() ?? "", _metadata);
        }
        finally
        {
            _model.Train(wasTraining);
        }
    }

    private (OnnxValue Value, Tensor Sample) Emit(OnnxGraph graph, Module module, OnnxValue x, Tensor sample)
    {
        if (module is Sequential sequential && !_modules.Any(m => m.Type.IsInstanceOfType(module)))
        {
            foreach (var child in sequential)
            {
                (x, sample) = Emit(graph, child, x, sample);
            }

            return (x, sample);
        }

        var output = module.Forward(sample);
        int[] shape = [-1, .. output.Shape[1..].ToArray()];
        return (Translate(graph, module, x with { Shape = x.Shape ?? [-1, .. sample.Shape[1..].ToArray()] }, shape), output);
    }

    private OnnxValue Translate(OnnxGraph g, Module module, OnnxValue x, int[] shape)
    {
        foreach (var (type, translate) in _modules)
        {
            if (type.IsInstanceOfType(module))
            {
                return translate(g, module, x, shape);
            }
        }

        return module switch
        {
            Layers.Linear linear => Linear(g, linear, x, shape),
            ReLU => g.Node("Relu", [x], shape),
            Tanh => g.Node("Tanh", [x], shape),
            Sigmoid => g.Node("Sigmoid", [x], shape),
            GELU => Gelu(g, x, shape),
            Softmax => g.Node("Softmax", [x], shape, OnnxAttribute.Of("axis", -1L)),
            Dropout => x,
            BatchNorm bn => g.Node("BatchNormalization",
                [x, Weights(g, "gamma", bn.Gamma), Weights(g, "beta", bn.Beta), Weights(g, "mean", bn.RunningMean), Weights(g, "var", bn.RunningVariance)],
                shape, OnnxAttribute.Of("epsilon", bn.Epsilon)),
            LayerNorm ln => LayerNorm(g, ln, x, shape),
            Conv2d conv => g.Node("Conv",
                [x, g.Constant("conv_w", conv.Weight.ToArray(), conv.OutChannels, conv.InChannels, conv.KernelSize, conv.KernelSize),
                    conv.Bias is null ? null : Weights(g, "conv_b", conv.Bias)],
                shape, OnnxAttribute.Of("kernel_shape", [conv.KernelSize, conv.KernelSize]), OnnxAttribute.Of("strides", [conv.Stride, conv.Stride]),
                OnnxAttribute.Of("pads", [conv.Padding, conv.Padding, conv.Padding, conv.Padding])),
            MaxPool2d pool => g.Node("MaxPool", [x], shape,
                OnnxAttribute.Of("kernel_shape", [pool.KernelSize, pool.KernelSize]), OnnxAttribute.Of("strides", [pool.Stride, pool.Stride]),
                OnnxAttribute.Of("pads", [pool.Padding, pool.Padding, pool.Padding, pool.Padding])),
            GlobalAveragePool2d => g.Node("Flatten", [g.Node("GlobalAveragePool", [x])], shape, OnnxAttribute.Of("axis", 1L)),
            Layers.Flatten => g.Node("Flatten", [x], shape, OnnxAttribute.Of("axis", 1L)),
            Embedding e => g.Node("Gather", [g.Constant("embedding", e.WeightValues(), e.Vocabulary, e.Dim), g.Node("Cast", [x], null, OnnxAttribute.Of("to", 7L))], shape,
                OnnxAttribute.Of("axis", 0L)),
            PositionalEncoding pe => g.Node("Add", [x, g.Constant("positions", PositionTable(x.Shape![^2], pe.Dim), x.Shape[^2], pe.Dim)], shape),
            MultiHeadAttention mha => Attention(g, mha, x, shape),
            TransformerEncoderLayer layer => EncoderLayer(g, layer, x, shape),
            LSTM lstm => Recurrent(g, "LSTM", lstm, [0, 3, 1, 2], x, shape),       // ONNX gate order i, o, f, c; ours i, f, c, o
            GRU gru => Recurrent(g, "GRU", gru, [1, 0, 2], x, shape),              // ONNX z, r, h; ours r, z, h
            Layers.Lambda lambda => _lambdas.TryGetValue(lambda.ToString(), out var translate)
                ? translate(g, lambda, x, shape)
                : throw new NotSupportedException($"The lambda '{lambda}' cannot be exported: register a translator with Lambda(\"{lambda}\", ...)."),
            _ => throw new NotSupportedException($"{module.GetType().Name} ({module.DisplayName}) cannot be exported: register a translator with Module<{module.GetType().Name}>(...)."),
        };
    }

    private static OnnxValue Weights(OnnxGraph g, string hint, Tensor tensor) => g.Constant(hint, tensor.ToArray(), tensor.Shape.ToArray());

    private static OnnxValue Linear(OnnxGraph g, Layers.Linear linear, OnnxValue x, IReadOnlyList<int>? shape)
    {
        var weight = linear.WeightValues();                                      // int8 weights are exported dequantized
        if (linear.Adapter is { } a)
        {
            using var update = a.A.MatMul(a.B) * a.Scale;                        // the LoRA update folded in, as MergeLora does
            var delta = update.ToArray();
            for (int i = 0; i < weight.Length; i++)
            {
                weight[i] += delta[i];
            }
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
