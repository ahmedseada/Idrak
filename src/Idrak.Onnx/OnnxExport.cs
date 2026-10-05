// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.Onnx;

/// <summary>
/// Translates one module into ONNX nodes: <paramref name="input"/> is the module's input value, <paramref name="outputShape"/>
/// the shape it produces (measured by running the module, -1 for the batch dimension). Returns the output value.
/// Register it for every export with <see cref="OnnxExportOps.Register{T}"/>, or for one with <see cref="OnnxExporter.Module{T}"/>.
/// </summary>
public delegate OnnxValue OnnxTranslator<in T>(OnnxGraph graph, T module, OnnxValue input, IReadOnlyList<int> outputShape) where T : Module;

/// <summary>
/// Translates one operation node of a <see cref="GraphModule"/> into ONNX nodes and returns the output value. Register it
/// with <see cref="OnnxExportOps.RegisterGraphOp"/> (or <see cref="OnnxExporter.GraphOp"/>) under the operation's name.
/// </summary>
public delegate OnnxValue OnnxGraphOpTranslator(OnnxGraphOpContext context);

/// <summary>The graph node being exported, its input values and its measured output shape, for an <see cref="OnnxGraphOpTranslator"/>.</summary>
public sealed class OnnxGraphOpContext
{
    private readonly Func<int, long[]?> _integers;

    internal OnnxGraphOpContext(OnnxGraph graph, GraphNode node, IReadOnlyList<OnnxValue?> inputs, IReadOnlyList<int>? outputShape, Func<int, long[]?> integers)
    {
        Graph = graph;
        Node = node;
        Inputs = inputs;
        OutputShape = outputShape;
        _integers = integers;
    }

    /// <summary>The ONNX graph to add nodes to.</summary>
    public OnnxGraph Graph { get; }

    /// <summary>The graph node: its operation, inputs, output and attributes.</summary>
    public GraphNode Node { get; }

    /// <summary>The node's inputs as ONNX values (null for an omitted optional input).</summary>
    public IReadOnlyList<OnnxValue?> Inputs { get; }

    /// <summary>The shape of the node's output (-1 for the batch dimension), or null when it is an integer value (a shape, axes or indices).</summary>
    public IReadOnlyList<int>? OutputShape { get; }

    /// <summary>The integer values of input <paramref name="index"/> (a constant, or a shape computed for the export's sample), or null for a tensor.</summary>
    public long[]? Integers(int index) => _integers(index);

    /// <summary>The integer attribute <paramref name="name"/>, or null when the node does not set it.</summary>
    public long? Int(string name) => Node.Attributes?[name] is JsonValue v ? (long)v : null;

    /// <summary>The number attribute <paramref name="name"/>, or null when the node does not set it.</summary>
    public float? Float(string name) => Node.Attributes?[name] is JsonValue v ? (float)v : null;

    /// <summary>The integer list attribute <paramref name="name"/>, or null.</summary>
    public long[]? Ints(string name) => Node.Attributes?[name] is JsonArray a ? [.. a.Select(v => (long)v!)] : null;

    /// <summary>Adds the ONNX node <paramref name="op"/> on <see cref="Inputs"/> with <see cref="OutputShape"/>.</summary>
    public OnnxValue Operator(string op, params OnnxAttribute[] attributes) => Graph.Node(op, Inputs, OutputShape, attributes);
}

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
/// How modules, lambdas and graph operations are written to ONNX, for every export. The built-in layers are registered
/// here (Linear, Conv2d, BatchNorm, LSTM, Sequential, GraphModule, ...), as are the builder's lambdas (MeanOverTime,
/// FirstStep, LastStep, Reshape) and every <see cref="GraphModule"/> operation; add or replace one with
/// <see cref="Register{T}"/>, <see cref="RegisterLambda"/> or <see cref="RegisterGraphOp"/>. A module uses the translator
/// registered for its own type or its nearest registered base type. Translators given to one exporter
/// (<see cref="OnnxExporter.Module{T}"/>, <see cref="OnnxExporter.Lambda"/>, <see cref="OnnxExporter.GraphOp"/>) take
/// precedence over these.
/// </summary>
public static class OnnxExportOps
{
    private static readonly Dictionary<Type, (Delegate Original, OnnxTranslator<Module> Translate)> Modules = [];
    private static readonly Dictionary<string, OnnxTranslator<Module>> Lambdas = new(StringComparer.Ordinal);
    private static readonly Dictionary<string, OnnxGraphOpTranslator> GraphOps = new(StringComparer.Ordinal);

    static OnnxExportOps() => OnnxBuiltIns.Register();

    /// <summary>Registers (or replaces) how modules of type <typeparamref name="T"/> (and types derived from it without their own translator) are exported.</summary>
    public static void Register<T>(OnnxTranslator<T> translate) where T : Module
    {
        ArgumentNullException.ThrowIfNull(translate);
        lock (Modules)
        {
            Modules[typeof(T)] = (translate, (g, m, x, s) => translate(g, (T)m, x, s));
        }
    }

    /// <summary>Removes the translator of modules of type <typeparamref name="T"/>; returns whether one was registered.</summary>
    public static bool Unregister<T>() where T : Module
    {
        lock (Modules)
        {
            return Modules.Remove(typeof(T));
        }
    }

    /// <summary>The module types with a registered translator.</summary>
    public static IReadOnlyCollection<Type> Types
    {
        get
        {
            lock (Modules)
            {
                return [.. Modules.Keys];
            }
        }
    }

    /// <summary>The translator registered for modules of exactly the type <typeparamref name="T"/>.</summary>
    public static OnnxTranslator<T> Get<T>() where T : Module
    {
        lock (Modules)
        {
            return Modules.TryGetValue(typeof(T), out var entry) ? (OnnxTranslator<T>)entry.Original
                : throw new NotSupportedException($"No ONNX export translator for {typeof(T).Name} is registered ({TypeNames()}); add it with OnnxExportOps.Register<{typeof(T).Name}>.");
        }
    }

    /// <summary>Registers (or replaces) how the lambdas named <paramref name="name"/> (their <c>ToString()</c>) are exported.</summary>
    public static void RegisterLambda(string name, OnnxTranslator<Module> translate)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(translate);
        lock (Lambdas)
        {
            Lambdas[name] = translate;
        }
    }

    /// <summary>Removes the lambda translator <paramref name="name"/>; returns whether it was registered.</summary>
    public static bool UnregisterLambda(string name)
    {
        lock (Lambdas)
        {
            return Lambdas.Remove(name);
        }
    }

    /// <summary>The lambda names with a registered translator.</summary>
    public static IReadOnlyCollection<string> LambdaNames
    {
        get
        {
            lock (Lambdas)
            {
                return [.. Lambdas.Keys];
            }
        }
    }

    /// <summary>Registers (or replaces) how <see cref="GraphModule"/> nodes running the operation <paramref name="op"/> are exported.</summary>
    public static void RegisterGraphOp(string op, OnnxGraphOpTranslator translate)
    {
        ArgumentException.ThrowIfNullOrEmpty(op);
        ArgumentNullException.ThrowIfNull(translate);
        lock (GraphOps)
        {
            GraphOps[op] = translate;
        }
    }

    /// <summary>Removes the graph operation translator <paramref name="op"/>; returns whether it was registered.</summary>
    public static bool UnregisterGraphOp(string op)
    {
        lock (GraphOps)
        {
            return GraphOps.Remove(op);
        }
    }

    /// <summary>The graph operations with a registered translator.</summary>
    public static IReadOnlyCollection<string> GraphOpNames
    {
        get
        {
            lock (GraphOps)
            {
                return [.. GraphOps.Keys];
            }
        }
    }

    /// <summary>The translator registered for the graph operation <paramref name="op"/>.</summary>
    public static OnnxGraphOpTranslator GetGraphOp(string op) => TryGetGraphOp(op)
        ?? throw new NotSupportedException($"No ONNX export translator for the graph operation '{op}' is registered ({string.Join(", ", GraphOpNames)}); add it with OnnxExportOps.RegisterGraphOp.");

    internal static string TypeNames() => string.Join(", ", Types.Select(t => t.Name).Order());

    // The translator of the module's type or its nearest registered base type.
    internal static OnnxTranslator<Module>? TryGet(Type type)
    {
        lock (Modules)
        {
            for (Type? t = type; t is not null && t != typeof(object); t = t.BaseType)
            {
                if (Modules.TryGetValue(t, out var entry))
                {
                    return entry.Translate;
                }
            }

            return null;
        }
    }

    internal static OnnxTranslator<Module>? TryGetLambda(string name)
    {
        lock (Lambdas)
        {
            return Lambdas.TryGetValue(name, out var translate) ? translate : null;
        }
    }

    internal static OnnxGraphOpTranslator? TryGetGraphOp(string op)
    {
        lock (GraphOps)
        {
            return GraphOps.TryGetValue(op, out var translate) ? translate : null;
        }
    }
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
        var graph = new OnnxGraph { Exporter = this };
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

    internal OnnxGraphOpTranslator? FindGraphOp(string op) => _graphOps.TryGetValue(op, out var translate) ? translate : OnnxExportOps.TryGetGraphOp(op);

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

        var registered = OnnxExportOps.TryGet(module.GetType()) ?? throw new NotSupportedException(
            $"{module.GetType().Name} ({module.DisplayName}) cannot be exported (registered: {OnnxExportOps.TypeNames()}): "
            + $"register a translator with OnnxExportOps.Register<{module.GetType().Name}>(...) or the exporter's Module<{module.GetType().Name}>(...).");
        return registered(g, module, x, shape);
    }
}
