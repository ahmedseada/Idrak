// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Onnx;

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
/// Configures an ONNX export. The layers registered in <see cref="OnnxExportOps"/> are supported: Linear (LoRA adapters
/// are merged), activations, Softmax, Dropout (removed), BatchNorm, LayerNorm, Conv2d, MaxPool2d, GlobalAveragePool2d,
/// Flatten, Embedding, PositionalEncoding, MultiHeadAttention, TransformerEncoderLayer, LSTM, GRU, Sequential,
/// GraphModule (skip connections, branches and shape arithmetic, so an imported graph can be exported again), and the
/// builder's MeanOverTime, FirstStep, LastStep and Reshape lambdas. Other lambdas and custom modules need a translator,
/// registered for every export (<see cref="OnnxExportOps"/>) or given to this one (<see cref="Lambda"/>,
/// <see cref="Module{T}"/>, <see cref="GraphOp"/>), which takes precedence. Inputs are float32 (token ids too, as in
/// Idrak); the batch dimension is dynamic. The graph has one input and one output, as an Idrak module does.
/// </summary>
public sealed class OnnxExporter
{
    private readonly Module _model;
    private readonly Dictionary<string, OnnxTranslator<Module>> _lambdas = new(StringComparer.Ordinal);
    private readonly Dictionary<string, OnnxGraphOpTranslator> _graphOps = new(StringComparer.Ordinal);
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

    /// <summary>A translator for the lambdas named <paramref name="name"/> (their <c>ToString()</c>) in this export; it takes precedence over <see cref="OnnxExportOps"/>.</summary>
    public OnnxExporter Lambda(string name, OnnxTranslator<Module> translate)
    {
        _lambdas[name] = translate;
        return this;
    }

    /// <summary>A translator for modules of type <typeparamref name="T"/> (a custom layer) in this export; it takes precedence over <see cref="OnnxExportOps"/>.</summary>
    public OnnxExporter Module<T>(OnnxTranslator<T> translate) where T : Module
    {
        _modules.Add((typeof(T), (g, m, x, s) => translate(g, (T)m, x, s)));
        return this;
    }

    /// <summary>A translator for <see cref="GraphModule"/> nodes running the operation <paramref name="op"/> in this export; it takes precedence over <see cref="OnnxExportOps"/>.</summary>
    public OnnxExporter GraphOp(string op, OnnxGraphOpTranslator translate)
    {
        _graphOps[op] = translate;
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
        var graph = new OnnxGraphWriter(this);
        bool wasTraining = _model.IsTraining;
        _model.Eval();
        try
        {
            using var noGrad = Autograd.NoGrad();
            using var scope = new TensorScope();
            int[] inputShape = [-1, .. sampleShape];
            var input = new OnnxValue(graph.Unique(_inputName), inputShape);
            var value = Emit(graph, _model, input);
            var output = graph.Output(value, _outputName, value.Shape!);
            return graph.ToModel(input, inputShape, output, value.Shape!, _model.DisplayName, "Idrak",
                typeof(Module).Assembly.GetName().Version?.ToString() ?? "", _metadata);
        }
        finally
        {
            _model.Train(wasTraining);
        }
    }

    // Translates `module` applied to `x` (whose shape is known); the output shape is measured by running the module on
    // a zero sample of that shape.
    internal OnnxValue Emit(OnnxGraph graph, Module module, OnnxValue x)
    {
        var device = module.WeightsDevice ?? _model.WeightsDevice ?? Device.Default;
        var inputShape = x.Shape ?? throw new InvalidOperationException($"The shape of the input of {module.DisplayName} is unknown.");
        var sample = Tensor.Zeros([1, .. inputShape.Skip(1)], device);
        var output = module.Forward(sample);                                   // both freed by the export's tensor scope
        int[] shape = [-1, .. output.Shape[1..].ToArray()];
        return Translate(graph, module, x, shape) with { Shape = shape };
    }

    internal OnnxGraphOpTranslator? FindGraphOp(string op) => _graphOps.TryGetValue(op, out var translate) ? translate : OnnxExportOps.FindGraphOp(op);

    private OnnxValue Translate(OnnxGraph g, Module module, OnnxValue x, int[] shape)
    {
        foreach (var (type, translate) in _modules)
        {
            if (type.IsInstanceOfType(module))
            {
                return translate(g, module, x, shape);
            }
        }

        if (module is Layers.Lambda lambda && _lambdas.TryGetValue(lambda.ToString(), out var forLambda))
        {
            return forLambda(g, lambda, x, shape);
        }

        var registered = OnnxExportOps.Find(module.GetType()) ?? throw new NotSupportedException(
            $"{module.GetType().Name} ({module.DisplayName}) cannot be exported (registered: {string.Join(", ", OnnxExportOps.Types.Select(t => t.Name).Order())}): "
            + $"register a translator with OnnxExportOps.Register<{module.GetType().Name}>(...) or the exporter's Module<{module.GetType().Name}>(...).");
        return registered(g, module, x, shape);
    }
}
