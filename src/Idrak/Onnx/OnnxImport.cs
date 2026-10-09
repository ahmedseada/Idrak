// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Layers.Abstractions;
using Idrak.Onnx.Abstractions;

namespace Idrak.Onnx;

/// <summary>
/// An ONNX model rebuilt from Idrak layers, with the file's weights, on the device chosen at import (so it runs
/// on Idrak's own CPU or CUDA kernels). A chain of layers becomes a <see cref="Sequential"/> with its
/// <see cref="Network"/> description; a graph with skip connections or branches becomes a <see cref="GraphModule"/>.
/// Dispose it to release the model.
/// </summary>
public sealed class ImportedNetwork : IDisposable
{
    internal ImportedNetwork(NetworkBuilder? network, Module model, IReadOnlyDictionary<string, string> metadata, IReadOnlyList<string> notes, int[]? inputShape)
    {
        InputShape = inputShape;
        Network = network;
        Model = model;
        Metadata = metadata;
        Notes = notes;
    }

    /// <summary>The layers as builder steps when the model is a chain (a <see cref="Sequential"/>); null for a graph.</summary>
    public NetworkBuilder? Network { get; }

    /// <summary>The network with the file's weights: a <see cref="Sequential"/> for a chain, a <see cref="GraphModule"/> for a graph.</summary>
    public Module Model { get; }

    /// <summary>The shape of one input sample from the file (without the batch dimension), or null when it has dynamic dimensions.</summary>
    public IReadOnlyList<int>? InputShape { get; }

    /// <summary>True when the model is a <see cref="GraphModule"/> (it has skip connections, branches or shape arithmetic).</summary>
    public bool IsGraph => Model is GraphModule;

    /// <summary>The ONNX file's metadata (custom key/value pairs).</summary>
    public IReadOnlyDictionary<string, string> Metadata { get; }

    /// <summary>Approximations made during import (for example exact GELU replaced by the tanh approximation).</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>
    /// Saves a Idrak package (.ikm): the architecture, the weights, the ONNX metadata (as the JSON entry
    /// "onnx-metadata") and the input shape. Load it with <c>ModelPackage.Open(path).BuildNetwork(device)</c> or <c>Predictor.Load(path, device)</c>.
    /// </summary>
    public void SavePackage(string path)
    {
        var metadata = new JsonObject();
        foreach (var (key, value) in Metadata)
        {
            metadata[key] = value;
        }

        var writer = ModelPackage.Create(path);
        writer = Network is not null ? writer.Architecture(Network) : writer.Architecture(ModelPackage.DefaultModelName, ((GraphModule)Model).ToJson());
        writer.Weights(ModelPackage.DefaultModelName, Model).Json("onnx-metadata", metadata);
        if (InputShape is { } shape)
        {
            // Read by Predictor.Load, so a flat float[] sample is reshaped to the model's input (for example [3, 224, 224]).
            writer.Json("predictor", new JsonObject { ["inputShape"] = new JsonArray([.. shape.Select(d => (JsonNode)d)]) });
        }

        writer.Save();
    }

    /// <inheritdoc />
    public void Dispose() => Model.Dispose();
}

/// <summary>
/// Imports ONNX models into Idrak layers. The graph must be a chain of supported layers from one input to one
/// output: MatMul/Gemm (+ Add) → Linear; Relu, Tanh, Sigmoid, Softmax; GELU (the Gelu op, the tanh form, or the erf
/// form, which becomes the tanh approximation); BatchNormalization; LayerNormalization; Conv (rectangular kernels,
/// strides, symmetric padding, dilation, groups); ConvTranspose (the same, with output_padding); MaxPool and AveragePool
/// (padding below and right may differ, ceil_mode); GlobalAveragePool and GlobalMaxPool (adaptive pooling to 1x1, or
/// global average pooling + Flatten); Resize (nearest with asymmetric coordinates and floor, linear with half_pixel or
/// align_corners; constant scales or sizes, or sizes computed from the input's shape as PyTorch exports them);
/// GroupNormalization, InstanceNormalization, and PyTorch's group norm (Reshape, InstanceNormalization, Reshape, Mul, Add);
/// Flatten; Gather
/// on a weight table → Embedding; sinusoidal position tables → PositionalEncoding; LSTM and GRU (forward or
/// bidirectional, stacked, PyTorch-style GRU, zero initial states); an image's columns as a sequence; ReduceMean over time;
/// first/last time step; Reshape; Dropout and Identity (skipped); and the attention and transformer-layer blocks
/// Idrak exports. Anything else is reported with the node that could not be imported; operators of your own are
/// registered in <see cref="OnnxImportOps"/>.
/// </summary>
public static class OnnxImport
{
    /// <summary>Imports an .onnx file onto <paramref name="device"/> (the default device when null).</summary>
    /// <param name="path">The .onnx file.</param>
    /// <param name="device">Where the layers are created.</param>
    /// <param name="sampleShape">The shape of one input sample, when the file leaves dimensions other than the batch dynamic.</param>
    public static ImportedNetwork Load(string path, Device? device = null, int[]? sampleShape = null) =>
        Load(File.ReadAllBytes(path), device, sampleShape);

    /// <summary>Imports a model from the bytes of an .onnx file.</summary>
    public static ImportedNetwork Load(byte[] model, Device? device = null, int[]? sampleShape = null)
    {
        var onnx = OnnxModel.Read(model);
        try
        {
            return new Importer(onnx, device ?? Device.Default, sampleShape).Run();
        }
        catch (NotSupportedException)
        {
            // Not a chain of layers (skip connections, branches, shape arithmetic): import it as a graph instead.
            return new Importer(onnx, device ?? Device.Default, sampleShape).RunGraph();
        }
    }
}

internal sealed class Importer(OnnxModel model, Device device, int[]? sampleShape)
{
    private static readonly int[] LstmOrder = [0, 3, 1, 2];   // ONNX gate g holds our gate LstmOrder[g] (ONNX i, o, f, c; ours i, f, c, o)
    private static readonly int[] GruOrder = [1, 0, 2];       // ONNX z, r, h; ours r, z, h

    private readonly Dictionary<string, List<OnnxNode>> _users = model.Nodes
        .SelectMany(n => n.Inputs.Where(i => i.Length > 0).Distinct().Select(i => (i, n))).GroupBy(p => p.i).ToDictionary(g => g.Key, g => g.Select(p => p.n).ToList());

    private readonly HashSet<OnnxNode> _consumed = [];
    private readonly List<Action<Module>?> _loaders = [];
    private readonly List<string> _notes = [];
    private NetworkBuilder _network = null!;

    public ImportedNetwork Run()
    {
        if (model.Inputs.Count != 1 || model.Outputs.Count != 1)
        {
            throw new NotSupportedException($"The importer needs one input and one output; the model has {model.Inputs.Count} and {model.Outputs.Count}.");
        }

        var input = model.Inputs[0];
        int[] shape = sampleShape ?? (input.Shape is { Length: > 1 } s ? s[1..] : throw new NotSupportedException("The input's shape is unknown; pass sampleShape."));
        if (shape.Any(d => d <= 0))
        {
            throw new NotSupportedException($"The input's sample shape [{string.Join(", ", shape)}] has dynamic dimensions; pass sampleShape.");
        }

        bool tokens = Users(input.Name).Any(n => n.Op == "Cast" || (n.Op == "Gather" && n.Inputs[1] == input.Name && Const(n.Inputs[0]) is not null));
        _network = (tokens, shape.Length) switch
        {
            (true, 1) => Layers.Network.Tokens(shape[0]),
            (false, 1) => Layers.Network.Input(shape[0]),
            (false, 2) => Layers.Network.Sequence(shape[0], shape[1]),
            (false, 3) => Layers.Network.Image(shape[0], shape[1], shape[2]),
            _ => throw new NotSupportedException($"Unsupported input shape [{string.Join(", ", shape)}]."),
        };
        _network.OnDevice(device).Seed(0).Named(string.IsNullOrEmpty(model.GraphName) ? "onnx" : model.GraphName);

        string current = input.Name, output = model.Outputs[0].Name;
        while (current != output)
        {
            current = Step(current);
        }

        var built = _network.Build();
        try
        {
            var modules = built.ToList();
            if (modules.Count != _loaders.Count)
            {
                throw new InvalidOperationException($"Internal error: {modules.Count} layers for {_loaders.Count} imported steps.");
            }

            using (Autograd.NoGrad())
            {
                for (int i = 0; i < modules.Count; i++)
                {
                    _loaders[i]?.Invoke(modules[i]);
                }
            }
        }
        catch
        {
            built.Dispose();
            throw;
        }

        return new ImportedNetwork(_network, built, model.Metadata, _notes, shape);
    }

    // ------------------------------------------------------------------ the chain

    private string Step(string x)
    {
        var users = Users(x);
        if (users.Count == 0)
        {
            throw new NotSupportedException($"The value '{x}' is not used by any node and is not the graph output.");
        }

        if ((MatchTransformer(x) ?? MatchRecurrent(x) ?? MatchColumnsToSequence(x) ?? MatchGroupNorm(x) ?? MatchResize(x) ?? AttentionLayer(ParseAttention(x))
             ?? GeluLayer(ParseGelu(x)) ?? LinearLayer(ParseLinear(x))) is { } match)
        {
            Consume(match.Nodes);
            return Push(match.Step, match.Load, match.Output);
        }

        if (users.Count != 1)
        {
            throw new NotSupportedException($"'{x}' feeds {users.Count} nodes ({string.Join(", ", users)}); only chains of layers can be imported.");
        }

        var node = users[0];
        Consume(node);
        var translate = OnnxImportOps.Find(node.Op) ?? throw UnknownOperator(node);
        return translate(new ImportContext(this, node, x));
    }

    // The single-node operators, registered in OnnxImportOps (a translator returns the value the chain continues from).
    // A graph uses its own primitives for these (they build chains), unless a translator of your own replaces one.
    internal static readonly IReadOnlyDictionary<string, OnnxImportTranslator> BuiltInOps = new Dictionary<string, Func<ImportContext, string>>
    {
        ["Identity"] = c => c.Output,
        ["Dropout"] = c => c.Output,
        ["Relu"] = c => c.Add(b => b.ReLU()),
        ["Tanh"] = c => c.Add(b => b.Tanh()),
        ["Sigmoid"] = c => c.Add(b => b.Sigmoid()),
        ["Softmax"] = c =>
        {
            long axis = c.Int("axis", c.Opset >= 13 ? -1 : 1);
            if (axis != -1 && axis != c.CurrentShape.Count)
            {
                throw c.Unsupported($"softmax over axis {axis} (only the last axis is supported)");
            }

            return c.Add(b => b.Softmax());
        },
        ["BatchNormalization"] = c => c.Importer.PushMatch(c.Importer.BatchNormLayer(c.Node)),
        ["LayerNormalization"] = c =>
        {
            if (c.Int("axis", -1) is not (-1) && c.Int("axis", -1) != c.CurrentShape.Count)
            {
                throw c.Unsupported("layer normalization over more than the last axis");
            }

            return c.Importer.PushMatch(c.Importer.LayerNormLayer(c.Node));
        },
        ["Conv"] = c => c.Importer.PushMatch(c.Importer.ConvLayer(c.Node)),
        ["ConvTranspose"] = c => c.Importer.PushMatch(c.Importer.ConvTransposeLayer(c.Node)),
        ["MaxPool"] = c => c.Importer.PushMatch(c.Importer.MaxPoolLayer(c.Node)),
        ["AveragePool"] = c => c.Importer.PushMatch(c.Importer.AveragePoolLayer(c.Node)),
        ["GlobalAveragePool"] = c => c.Importer.GlobalAveragePool(c.Node),
        ["GlobalMaxPool"] = c => c.Importer.PushMatch(c.Importer.GlobalPoolLayer(c.Node, max: true)),
        ["Resize"] = c => c.Importer.PushMatch(c.Importer.ResizeLayer(c.Node, null)),
        ["GroupNormalization"] = c => c.Importer.PushMatch(c.Importer.GroupNormalizationLayer(c.Node, c.CurrentShape[0])),
        ["InstanceNormalization"] = c => c.Importer.PushMatch(c.Importer.InstanceNormLayer(c.Node)),
        ["Flatten"] = c => c.Int("axis", 1) == 1 ? c.Add(b => b.Flatten()) : throw c.Unsupported("flatten from an axis other than 1"),
        ["Cast"] = c => c.Importer.Cast(c.Node),
        ["Gather"] = c => c.Importer.Gather(c.Node, c.Input),
        ["Add"] = c => c.Importer.PositionTable(c.Node, c.Input),
        ["ReduceMean"] = c =>
        {
            var axes = c.Ints("axes") ?? (c.Inputs.Count > 1 ? c.Importer.Constant(c.Inputs[1])?.AsLongs() : null);
            if (axes is [1] && c.Int("keepdims", 1) == 0)
            {
                return c.Add(b => b.MeanOverTime());
            }

            throw c.Unsupported("a mean over axes other than time");
        },
        ["Reshape"] = c =>
        {
            var shape = (c.Inputs.Count > 1 ? c.Importer.Constant(c.Inputs[1])?.AsLongs() : null) ?? throw c.Unsupported("a computed shape");
            if (shape.Length < 2 || shape[0] is not (-1 or 0) || shape[1..].Any(d => d <= 0))
            {
                throw c.Unsupported($"reshape to [{string.Join(", ", shape)}] (the batch must stay first, other sizes fixed)");
            }

            return c.Add(b => b.Reshape([.. shape[1..].Select(d => (int)d)]));
        },
    }.ToDictionary(p => p.Key, p => (OnnxImportTranslator)(c => p.Value((ImportContext)c)), StringComparer.Ordinal);

    private string GlobalAveragePool(OnnxNode node)
    {
        // Followed by a flatten: the global average pool that gives [C]; otherwise adaptive pooling to [C, 1, 1].
        if (Users(node.Outputs[0]) is not [var flatten] || !(flatten.Op == "Flatten" && flatten.Int("axis", 1) == 1
            || flatten.Op == "Reshape" && Const(flatten.Inputs[1]) is { } target && target.AsLongs() is [-1 or 0, _]
            || flatten.Op == "Squeeze"))
        {
            return PushMatch(GlobalPoolLayer(node, max: false));
        }

        Consume(flatten);
        return Push(b => b.GlobalAveragePool2d(), null, flatten.Outputs[0]);
    }

    private string Cast(OnnxNode node)
    {
        if (Users(node.Outputs[0]) is [{ Op: "Gather" } gather] && EmbeddingLayer(gather, node.Outputs[0]) is { } embedding)
        {
            Consume(gather);
            return PushMatch(embedding);
        }

        throw Unsupported(node, "a cast that does not feed an embedding lookup");
    }

    internal long Opset => model.Opset;

    internal IReadOnlyList<int> CurrentShape => _network.CurrentShape;

    internal OnnxTensor? Constant(string name) => Const(name);

    internal void Note(string note) => _notes.Add(note);

    private string PushMatch(LayerMatch match) => Push(match.Step, match.Load, match.Output);

    // Adds the step's layers; `load` fills the first of them (the others keep their initial weights).
    internal string Push(Func<NetworkBuilder, NetworkBuilder> step, Action<Module>? load, string output)
    {
        int count = _network.Count;
        step(_network);
        for (int i = count; i < _network.Count; i++)
        {
            _loaders.Add(i == count ? load : null);
        }

        return output;
    }

    private List<OnnxNode> Users(string value) => _users.TryGetValue(value, out var users) ? [.. users.Where(u => !_consumed.Contains(u))] : [];

    private OnnxNode? Only(string value, string op) => Users(value) is [var node] && node.Op == op ? node : null;

    private OnnxTensor? Const(string name) => name.Length > 0 && model.Constants.TryGetValue(name, out var t) ? t : null;

    private float? Scalar(string name) => Const(name) is { Size: 1 } t ? t.AsFloats()[0] : null;

    private void Consume(params IEnumerable<OnnxNode> nodes)
    {
        foreach (var node in nodes)
        {
            _consumed.Add(node);
        }
    }

    internal static NotSupportedException Unsupported(OnnxNode node, string what) =>
        new($"Cannot import {node}: {what} has no Idrak layer.");

    private static NotSupportedException UnknownOperator(OnnxNode node) =>
        new($"Cannot import {node}: the {node.Op} operator has no Idrak layer (registered: {string.Join(", ", OnnxImportOps.Names)}); "
            + $"add a translator with OnnxImportOps.Register(\"{node.Op}\", ...).");

    // The other input of a binary node whose one input is `x`.
    private static string? Other(OnnxNode node, string x) =>
        node.Inputs.Count == 2 && node.Inputs[0] == x ? node.Inputs[1] : node.Inputs.Count == 2 && node.Inputs[1] == x ? node.Inputs[0] : null;

    // ------------------------------------------------------------------ graphs

    // ONNX operators and the GraphModule operations they become.
    private static readonly Dictionary<string, string> Primitives = new()
    {
        ["Add"] = "add", ["Sub"] = "sub", ["Mul"] = "mul", ["Div"] = "div", ["MatMul"] = "matmul",
        ["Relu"] = "relu", ["Tanh"] = "tanh", ["Sigmoid"] = "sigmoid", ["Exp"] = "exp", ["Log"] = "log", ["Abs"] = "abs",
        ["Neg"] = "neg", ["Sqrt"] = "sqrt", ["Pow"] = "pow", ["Clip"] = "clip", ["LeakyRelu"] = "leaky_relu", ["Elu"] = "elu",
        ["HardSigmoid"] = "hard_sigmoid", ["HardSwish"] = "hard_swish", ["Max"] = "max", ["Min"] = "min",
        ["Softmax"] = "softmax", ["Flatten"] = "flatten", ["Reshape"] = "reshape", ["Transpose"] = "transpose", ["Concat"] = "concat",
        ["Shape"] = "shape", ["Gather"] = "gather", ["Slice"] = "slice", ["Unsqueeze"] = "unsqueeze", ["Squeeze"] = "squeeze",
        ["Identity"] = "identity", ["Cast"] = "cast", ["Dropout"] = "identity", ["ReduceMean"] = "reduce_mean",
        ["GlobalAveragePool"] = "global_average_pool",
    };

    private readonly List<GraphNode> _graphNodes = [];
    private readonly List<(string Name, Tensor Value)> _graphConstants = [];
    private readonly List<(string Name, long[] Values, int[] Dims)> _graphIntegers = [];
    private readonly HashSet<string> _registered = [];
    private readonly List<Module> _graphLayers = [];

    // Values of one zero sample computed so far, for the shapes translators ask for (filled only when asked).
    private Dictionary<string, object>? _trace;
    private TensorScope? _traceScope;
    private int _traced;
    private int[]? _graphSample;

    public ImportedNetwork RunGraph()
    {
        if (model.Inputs.Count != 1 || model.Outputs.Count != 1)
        {
            throw new NotSupportedException($"The importer needs one input and one output; the model has {model.Inputs.Count} and {model.Outputs.Count}.");
        }

        _graphSample = sampleShape ?? (model.Inputs[0].Shape is { Length: > 1 } s && s[1..].All(d => d > 0) ? s[1..] : null);
        try
        {
            foreach (var node in model.Nodes)
            {
                if (_consumed.Contains(node))
                {
                    continue;
                }

                string? x = node.Inputs.FirstOrDefault(i => i.Length > 0 && Const(i) is null);
                if (x is not null && MatchLayer(node, x) is { } match)
                {
                    Consume(match.Nodes);
                    var layer = match.Create();
                    _graphLayers.Add(layer);
                    using (Autograd.NoGrad())
                    {
                        match.Load?.Invoke(layer);
                    }

                    _graphNodes.Add(new GraphNode("layer", [x], match.Output, Layer: layer));
                    continue;
                }

                Consume(node);
                var translate = OnnxImportOps.Find(node.Op);
                bool primitive = Primitives.ContainsKey(node.Op) && node.Domain is "" or "ai.onnx";
                bool replaced = translate is not null && !(BuiltInOps.TryGetValue(node.Op, out var builtIn) && builtIn == translate);
                if (replaced && x is not null)
                {
                    Translate(node, x, translate!);
                }
                else if (primitive)
                {
                    _graphNodes.Add(Primitive(node));
                }
                else
                {
                    // The built-in translators build chains of layers; in a graph their operators are the primitives above.
                    throw translate is null ? UnknownOperator(node)
                        : Unsupported(node, $"the {node.Op} operator in a graph (its built-in translator only imports chains of layers)");
                }
            }

            var graph = new GraphModule(model.Inputs[0].Name, model.Outputs[0].Name, _graphNodes, _graphConstants, _graphIntegers)
            {
                Name = string.IsNullOrEmpty(model.GraphName) ? "onnx" : model.GraphName,
            };
            return new ImportedNetwork(null, graph, model.Metadata, _notes, _graphSample);
        }
        catch
        {
            _graphLayers.ForEach(l => l.Dispose());
            _graphConstants.ForEach(c => c.Value.Dispose());
            throw;
        }
        finally
        {
            _traceScope?.Dispose();
        }
    }

    // The layer pattern that starts at `node` (whose data input is `x`), if any.
    private LayerMatch? MatchLayer(OnnxNode node, string x)
    {
        var match = node.Op switch
        {
            "LayerNormalization" => MatchTransformer(x) ?? (node.Int("axis", -1) == -1 && node.Inputs[0] == x ? LayerNormLayer(node) : null),
            "Transpose" => MatchRecurrent(x),
            "MatMul" or "Gemm" => AttentionLayer(ParseAttention(x)) ?? LinearLayer(ParseLinear(x)),
            "Mul" or "Div" or "Gelu" => GeluLayer(ParseGelu(x)),
            "BatchNormalization" when node.Inputs[0] == x => BatchNormLayer(node),
            "Conv" when node.Inputs[0] == x => ConvLayer(node),
            "ConvTranspose" when node.Inputs[0] == x => ConvTransposeLayer(node),
            "MaxPool" => MaxPoolLayer(node),
            "AveragePool" => AveragePoolLayer(node),
            "GlobalMaxPool" => GlobalPoolLayer(node, max: true),
            "Shape" => MatchResize(x),
            "Resize" when node.Inputs[0] == x => ResizeLayer(node, null),
            "Reshape" => MatchGroupNorm(x),
            "GroupNormalization" when node.Inputs[0] == x => GroupNormalizationLayer(node, SampleShape(node, x)[0]),
            "InstanceNormalization" when node.Inputs[0] == x => InstanceNormLayer(node),
            "Gather" => EmbeddingLayer(node, x),
            _ => null,
        };
        return match is not null && match.Nodes.Contains(node) ? match : null;
    }

    // A node imported through a translator of your own: its builder steps become layer nodes, its graph operations
    // (OnnxImportContext.AddGraphOp) operation nodes; a node that adds nothing passes its input through.
    private void Translate(OnnxNode node, string x, OnnxImportTranslator translate)
    {
        CheckExtraOutputs(node);
        var context = new ImportContext(this, node, x, graph: true);
        translate(context);
        if (!_graphNodes.Any(n => n.Output == node.Outputs[0]))
        {
            _graphNodes.Add(new GraphNode("identity", [context.Current], node.Outputs[0]));
        }
    }

    // Graph mode of OnnxImportContext.Add: the steps, built on the shape of `input`, become one layer node.
    internal string PushGraph(OnnxNode node, string input, Func<NetworkBuilder, NetworkBuilder> step, Action<Module>? load, string output)
    {
        var shape = SampleShape(node, input);
        var builder = shape.Length switch
        {
            1 => Layers.Network.Input(shape[0]),
            2 => Layers.Network.Sequence(shape[0], shape[1]),
            3 => Layers.Network.Image(shape[0], shape[1], shape[2]),
            _ => throw Unsupported(node, $"builder steps on a value shaped [{string.Join(", ", shape)}] (one to three dimensions per sample)"),
        };
        step(builder.OnDevice(device).Seed(0));
        if (builder.Count == 0)
        {
            _graphNodes.Add(new GraphNode("identity", [input], output));
            return output;
        }

        var built = builder.Build();
        var first = built[0];
        Module layer = built.Count == 1 && LayerTypes.CanDescribe(first) ? first : built;    // a block keeps the builder's description
        _graphLayers.Add(layer);
        using (Autograd.NoGrad())
        {
            load?.Invoke(first);
        }

        _graphNodes.Add(new GraphNode("layer", [input], output, Layer: layer));
        return output;
    }

    // Graph mode of OnnxImportContext.AddGraphOp.
    internal string AddGraphOp(OnnxNode node, string op, IReadOnlyList<string> inputs, JsonObject? attributes, string output)
    {
        if (!GraphOps.Contains(op))
        {
            throw Unsupported(node, $"the graph operation '{op}', which is not registered (registered: {string.Join(", ", GraphOps.Names)}; add it with GraphOps.Register)");
        }

        RegisterConstants(inputs);
        _graphNodes.Add(new GraphNode(op, [.. inputs], output, attributes));
        return output;
    }

    // The shape of one sample at `value`, computed by running the graph so far on a zero input.
    internal int[] SampleShape(OnnxNode node, string value)
    {
        if (_graphSample is not { } sample)
        {
            throw Unsupported(node, "a translator that needs the shape of its input while the model's input has dynamic dimensions (pass sampleShape)");
        }

        if (_trace is null)
        {
            _traceScope = new TensorScope();
            _trace = new() { [model.Inputs[0].Name] = Tensor.Zeros([1, .. sample], device) };
        }

        using (Autograd.NoGrad())
        {
            for (; _traced < _graphNodes.Count; _traced++)
            {
                var graphNode = _graphNodes[_traced];
                bool training = graphNode.Layer?.IsTraining ?? false;
                graphNode.Layer?.Eval();                                   // batch normalization must not update its statistics
                try
                {
                    _trace[graphNode.Output] = GraphModule.RunNode(graphNode, Lookup);
                }
                finally
                {
                    graphNode.Layer?.Train(training);
                }
            }
        }

        return Lookup(value) is Tensor t ? t.Shape[1..].ToArray()
            : throw Unsupported(node, $"builder steps on '{value}', which is an integer value or not computed yet");
    }

    private object? Lookup(string name) =>
        _trace!.TryGetValue(name, out var v) ? v
        : _graphConstants.FirstOrDefault(c => c.Name == name).Value is { } t ? t
        : _graphIntegers.FirstOrDefault(c => c.Name == name) is { Values: not null } i ? GraphModule.IntegerValue(i.Values, i.Dims)
        : null;

    private void CheckExtraOutputs(OnnxNode node)
    {
        if (node.Outputs.Skip(1).Any(o => o.Length > 0 && Users(o).Count > 0))
        {
            throw Unsupported(node, "a node whose extra outputs are used");
        }
    }

    // The node's constant inputs become the graph's constants (floats as buffers, integers on the host).
    private void RegisterConstants(IEnumerable<string> inputs)
    {
        foreach (var input in inputs.Where(i => i.Length > 0 && !_registered.Contains(i)))
        {
            if (Const(input) is not { } constant)
            {
                continue;
            }

            _registered.Add(input);
            if (constant.Floats is { } floats)
            {
                _graphConstants.Add((input, Tensor.Persistent(floats, constant.Dims, device, requiresGrad: false)));
            }
            else
            {
                _graphIntegers.Add((input, constant.Longs!, constant.Dims));
            }
        }
    }

    // The node's attributes as JSON (integers, numbers, strings and their lists).
    internal static JsonObject? Attributes(OnnxNode node)
    {
        var attributes = new JsonObject();
        foreach (var (name, value) in node.Attributes)
        {
            if (value.Int is { } i)
            {
                attributes[name] = i;
            }
            else if (value.Float is { } f)
            {
                attributes[name] = f;
            }
            else if (value.String is { } text)
            {
                attributes[name] = text;
            }
            else if (value.Ints is { } ints)
            {
                attributes[name] = new JsonArray([.. ints.Select(v => (JsonNode)v)]);
            }
            else if (value.Floats is { } floats)
            {
                attributes[name] = new JsonArray([.. floats.Select(v => (JsonNode)v)]);
            }
        }

        return attributes.Count > 0 ? attributes : null;
    }

    private GraphNode Primitive(OnnxNode node)
    {
        var op = Primitives[node.Op];
        CheckExtraOutputs(node);
        var attributes = Attributes(node) ?? [];
        List<string> inputs = node.Op == "Dropout" ? [node.Inputs[0]] : [.. node.Inputs];
        if (node.Op == "Softmax" && !node.Attributes.ContainsKey("axis") && model.Opset < 13)
        {
            attributes["axis"] = 1;
        }

        if (node.Op == "Slice" && node.Ints("starts") is { } starts)
        {
            // Opset < 10: starts, ends and axes are attributes; make them inputs like later opsets.
            string Add(string what, long[] values)
            {
                string name = $"{node.Outputs[0]}_{what}";
                _graphIntegers.Add((name, values, [values.Length]));
                _registered.Add(name);
                return name;
            }

            inputs = [node.Inputs[0], Add("starts", starts), Add("ends", node.Ints("ends")!), .. node.Ints("axes") is { } axes ? [Add("axes", axes)] : new List<string>()];
            foreach (var key in new[] { "starts", "ends", "axes" })
            {
                attributes.Remove(key);
            }
        }

        RegisterConstants(inputs);
        return new GraphNode(op, inputs, node.Outputs[0], attributes.Count > 0 ? attributes : null);
    }

    // ------------------------------------------------------------------ Linear: MatMul (+ Add) or Gemm

    private sealed record LinearMatch(OnnxNode[] Nodes, float[] Weight, float[]? Bias, int In, int Out, string Output);

    private LinearMatch? ParseLinear(string x)
    {
        foreach (var node in Users(x))
        {
            if (node.Op == "MatMul" && node.Inputs[0] == x && Const(node.Inputs[1]) is { Dims.Length: 2 } w)
            {
                var (inF, outF) = (w.Dims[0], w.Dims[1]);
                if (Users(node.Outputs[0]) is [{ Op: "Add" } add] && Other(add, node.Outputs[0]) is { } b && Const(b) is { } bias && bias.Size == outF)
                {
                    return new([node, add], w.AsFloats(), bias.AsFloats(), inF, outF, add.Outputs[0]);
                }

                return new([node], w.AsFloats(), null, inF, outF, node.Outputs[0]);
            }

            if (node.Op == "Gemm" && node.Inputs[0] == x && Const(node.Inputs[1]) is { Dims.Length: 2 } g
                && node.Int("transA", 0) == 0 && node.Float("alpha", 1f) == 1f && node.Float("beta", 1f) == 1f)
            {
                bool transposed = node.Int("transB", 0) == 1;
                var (inF, outF) = transposed ? (g.Dims[1], g.Dims[0]) : (g.Dims[0], g.Dims[1]);
                var values = g.AsFloats();
                var weight = transposed ? Transpose(values, outF, inF) : values;
                var bias = node.Inputs.Count > 2 ? Const(node.Inputs[2]) : null;
                if (bias is not null && bias.Size != outF)
                {
                    continue;
                }

                return new([node], weight, bias?.AsFloats(), inF, outF, node.Outputs[0]);
            }
        }

        return null;
    }

    private static float[] Transpose(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                result[c * rows + r] = values[r * columns + c];
            }
        }

        return result;
    }

    private static void LoadLinear(Linear linear, LinearMatch match)
    {
        linear.Weight.Load(match.Weight);
        if (match.Bias is { } bias)
        {
            linear.Bias!.Load(bias);
        }
    }

    // ------------------------------------------------------------------ GELU

    private sealed record GeluMatch(OnnxNode[] Nodes, string Output);

    private GeluMatch? ParseGelu(string x)
    {
        var users = Users(x);
        if (users is [{ Op: "Gelu" } op])
        {
            if (op.String("approximate") != "tanh")
            {
                _notes.Add($"{op}: exact GELU imported as the tanh approximation (differences below 0.001).");
            }

            return new([op], op.Outputs[0]);
        }

        // Tanh form (Idrak's export): 0.5 · x · (1 + tanh(k (x + c x³))).
        if (users.FirstOrDefault(n => n.Op == "Mul" && n.Inputs.Count == 2 && n.Inputs[0] == x && n.Inputs[1] == x) is { } square
            && Only(square.Outputs[0], "Mul") is { } cube
            && Only(cube.Outputs[0], "Mul") is { } scaled && Near(Scalar(Other(scaled, cube.Outputs[0]) ?? ""), 0.044715f)
            && Only(scaled.Outputs[0], "Add") is { } inner && Other(inner, scaled.Outputs[0]) == x
            && Only(inner.Outputs[0], "Mul") is { } k && Near(Scalar(Other(k, inner.Outputs[0]) ?? ""), 0.7978845608f)
            && Only(k.Outputs[0], "Tanh") is { } tanh
            && Only(tanh.Outputs[0], "Add") is { } plusOne && Near(Scalar(Other(plusOne, tanh.Outputs[0]) ?? ""), 1f)
            && Only(plusOne.Outputs[0], "Mul") is { } gate && Other(gate, plusOne.Outputs[0]) == x
            && Only(gate.Outputs[0], "Mul") is { } half && Near(Scalar(Other(half, gate.Outputs[0]) ?? ""), 0.5f))
        {
            return new([square, cube, scaled, inner, k, tanh, plusOne, gate, half], half.Outputs[0]);
        }

        // Erf form (PyTorch): x · 0.5 · (1 + erf(x / √2)), with the 0.5 applied first or last.
        if (users.FirstOrDefault(n => n.Op == "Div" && n.Inputs[0] == x && Near(Scalar(n.Inputs[1]), MathF.Sqrt(2f))) is { } div
            && Only(div.Outputs[0], "Erf") is { } erf
            && Only(erf.Outputs[0], "Add") is { } add && Near(Scalar(Other(add, erf.Outputs[0]) ?? ""), 1f)
            && Only(add.Outputs[0], "Mul") is { } product)
        {
            string? factor = Other(product, add.Outputs[0]);
            var nodes = new List<OnnxNode> { div, erf, add, product };
            string output = product.Outputs[0];
            bool halfFirst = factor is not null && users.Any(n => n.Op == "Mul" && n.Outputs[0] == factor && Near(Scalar(Other(n, x) ?? ""), 0.5f));
            if (halfFirst)
            {
                nodes.Add(users.First(n => n.Outputs[0] == factor));
            }
            else if (factor == x && Only(output, "Mul") is { } last && Near(Scalar(Other(last, output) ?? ""), 0.5f))
            {
                nodes.Add(last);
                output = last.Outputs[0];
            }
            else
            {
                return null;
            }

            _notes.Add($"{div}: exact (erf) GELU imported as the tanh approximation (differences below 0.001).");
            return new([.. nodes], output);
        }

        return null;
    }

    private static bool Near(float? value, float expected) => value is { } v && MathF.Abs(v - expected) <= 1e-4f * MathF.Max(1f, MathF.Abs(expected));

    // ------------------------------------------------------------------ attention and transformer layers (Idrak's export)

    private sealed record AttentionMatch(OnnxNode[] Nodes, int Heads, bool Causal, LinearMatch Qkv, LinearMatch Projection, string Output);

    private AttentionMatch? ParseAttention(string x)
    {
        if (ParseLinear(x) is not { } qkv || qkv.Out != 3 * qkv.In
            || Only(qkv.Output, "Reshape") is not { } split || Const(split.Inputs[1])?.AsLongs() is not [_, var t, 3, var h, var dh]
            || Only(split.Outputs[0], "Transpose") is not { } heads
            || Users(heads.Outputs[0]) is not { Count: 3 } parts || parts.Any(p => p.Op != "Gather"))
        {
            return null;
        }

        OnnxNode? Part(long i) => parts.FirstOrDefault(p => Const(p.Inputs[1])?.AsLongs() is [var v] && v == i);
        if (Part(0) is not { } q || Part(1) is not { } k || Part(2) is not { } v
            || Only(k.Outputs[0], "Transpose") is not { } kT
            || Only(q.Outputs[0], "MatMul") is not { } scores || scores.Inputs[1] != kT.Outputs[0]
            || Only(scores.Outputs[0], "Mul") is not { } scale || !Near(Scalar(Other(scale, scores.Outputs[0]) ?? ""), 1f / MathF.Sqrt(dh)))
        {
            return null;
        }

        var nodes = new List<OnnxNode>(qkv.Nodes) { split, heads, q, k, v, kT, scores, scale };
        string logits = scale.Outputs[0];
        bool causal = false;
        if (Only(logits, "Add") is { } mask && Const(Other(mask, logits) ?? "") is { Dims: [var rows, var columns] } && rows == t && columns == t)
        {
            nodes.Add(mask);
            logits = mask.Outputs[0];
            causal = true;
        }

        if (Only(logits, "Softmax") is not { } softmax
            || Only(softmax.Outputs[0], "MatMul") is not { } context || context.Inputs[1] != v.Outputs[0]
            || Only(context.Outputs[0], "Transpose") is not { } merge
            || Only(merge.Outputs[0], "Reshape") is not { } join
            || ParseLinear(join.Outputs[0]) is not { } projection)
        {
            return null;
        }

        nodes.AddRange([softmax, context, merge, join, .. projection.Nodes]);
        return new([.. nodes], (int)h, causal, qkv, projection, projection.Output);
    }

    private static void LoadAttention(Module module, AttentionMatch match)
    {
        var children = module.Children().ToList();
        LoadLinear((Linear)children[0], match.Qkv);
        LoadLinear((Linear)children[1], match.Projection);
    }

    private LayerMatch? MatchTransformer(string x)
    {
        var users = Users(x);
        if (users.Count != 2 || users.FirstOrDefault(n => n.Op == "LayerNormalization" && n.Inputs[0] == x) is not { } norm1
            || users.FirstOrDefault(n => n.Op == "Add") is not { } residual1
            || ParseAttention(norm1.Outputs[0]) is not { } attention || Other(residual1, x) != attention.Output)
        {
            return null;
        }

        string a = residual1.Outputs[0];
        var after = Users(a);
        if (after.Count != 2 || after.FirstOrDefault(n => n.Op == "LayerNormalization" && n.Inputs[0] == a) is not { } norm2
            || after.FirstOrDefault(n => n.Op == "Add") is not { } residual2
            || ParseLinear(norm2.Outputs[0]) is not { } ff1
            || ParseGelu(ff1.Output) is not { } gelu || ParseLinear(gelu.Output) is not { } ff2 || Other(residual2, a) != ff2.Output)
        {
            return null;
        }

        int dim = attention.Qkv.In;
        return new([norm1, residual1, norm2, residual2, .. attention.Nodes, .. ff1.Nodes, .. gelu.Nodes, .. ff2.Nodes], residual2.Outputs[0],
            b => b.TransformerEncoderLayer(attention.Heads, ff1.Out, dropout: 0f, attention.Causal),
            () => new TransformerEncoderLayer(dim, attention.Heads, ff1.Out, 0f, attention.Causal, device),
            m =>
            {
                var c = m.Children().ToList();
                LoadNorm((LayerNorm)c[0], norm1);
                LoadAttention(c[1], attention);
                LoadNorm((LayerNorm)c[2], norm2);
                LoadLinear((Linear)c[3], ff1);
                LoadLinear((Linear)c[4], ff2);
            });
    }

    private void LoadNorm(LayerNorm norm, OnnxNode node)
    {
        norm.Gamma.Load(Const(node.Inputs[1])!.AsFloats());
        norm.Beta.Load(node.Inputs.Count > 2 && Const(node.Inputs[2]) is { } beta ? beta.AsFloats() : new float[norm.Features]);
    }

    // ------------------------------------------------------------------ recurrent layers

    // A recurrent layer as Idrak and PyTorch export it: [N, T, F] transposed to time-major, then one LSTM or GRU node per
    // layer (one or two directions); between layers and after the last, Y [T, D, N, H] squeezed (one direction) or
    // transposed and reshaped to [T, N, D·H] (two); then transposed back to [N, T, ·] for every step, or the last states
    // Y_h [D, N, H] squeezed or transposed and reshaped to [N, D·H]. Initial states are zero: constants, or zeros expanded
    // to a shape computed from the input (PyTorch's exporter writes these).
    private LayerMatch? MatchRecurrent(string x)
    {
        if (Only(x, "Transpose") is not { } timeMajor || timeMajor.Ints("perm") is not [1, 0, 2] || RecurrentNode(timeMajor.Outputs[0]) is null)
        {
            return null;
        }

        var nodes = new List<OnnxNode> { timeMajor };
        var layers = new List<RecurrentLayerWeights[]>();
        string value = timeMajor.Outputs[0], op = "", output;
        int h = 0, directions = 0, inputs = 0;
        bool sequences;
        while (true)
        {
            var (rnn, state) = RecurrentNode(value) ?? throw new NotSupportedException($"'{value}' feeds no recurrent layer with zero initial states alone.");
            bool lstm = rnn.Op == "LSTM";
            int d = rnn.String("direction") switch { null or "forward" => 1, "bidirectional" => 2, _ => 0 };
            if (d == 0 || rnn.Int("layout", 0) != 0 || rnn.Attributes.ContainsKey("activations") || rnn.Attributes.ContainsKey("clip")
                || !lstm && rnn.Int("linear_before_reset", 0) != 1 || lstm && rnn.Int("input_forget", 0) != 0
                || layers.Count > 0 && (rnn.Op != op || (int)rnn.Int("hidden_size", 0) != h || d != directions))
            {
                throw Unsupported(rnn, "this recurrent configuration (forward or bidirectional, default activations, no peepholes, zero initial states, the same "
                    + "cell, size and directions in every stacked layer and, for GRU, linear_before_reset = 1)");
            }

            (op, h, directions) = (rnn.Op, (int)rnn.Int("hidden_size", 0), d);
            var weights = RecurrentWeightsOf(rnn, lstm, h, directions);
            inputs = layers.Count == 0 ? weights[0].Inputs : inputs;
            layers.Add(weights);
            nodes.Add(rnn);
            nodes.AddRange(state);

            // Every step, as the next layer's input or the output.
            string? next = null;
            if (directions == 1 && Users(rnn.Outputs[0]) is [{ Op: "Squeeze" } squeeze] && SqueezeAxes(squeeze) is [1])
            {
                nodes.Add(squeeze);
                next = squeeze.Outputs[0];
            }
            else if (directions == 2 && Users(rnn.Outputs[0]) is [{ Op: "Transpose" } split] && split.Ints("perm") is [0, 2, 1, 3]
                     && Only(split.Outputs[0], "Reshape") is { } joined && Const(joined.Inputs[1])?.AsLongs() is [0, 0, var features] && (features == -1 || features == 2 * h))
            {
                nodes.AddRange([split, joined]);
                next = joined.Outputs[0];
            }

            if (next is not null)
            {
                if (Only(next, "Transpose") is { } back && back.Ints("perm") is [1, 0, 2])
                {
                    nodes.Add(back);
                    (sequences, output) = (true, back.Outputs[0]);
                    break;
                }

                value = next;
                continue;
            }

            // The last states.
            if (Users(rnn.Outputs[0]).Count == 0 && rnn.Outputs.Count > 1)
            {
                if (directions == 1 && Users(rnn.Outputs[1]) is [{ Op: "Squeeze" } last] && SqueezeAxes(last) is [0])
                {
                    nodes.Add(last);
                    (sequences, output) = (false, last.Outputs[0]);
                    break;
                }

                if (directions == 2 && Users(rnn.Outputs[1]) is [{ Op: "Transpose" } batchFirst] && batchFirst.Ints("perm") is [1, 0, 2]
                    && Only(batchFirst.Outputs[0], "Reshape") is { } joined && Const(joined.Inputs[1])?.AsLongs() is [0 or -1, var features] && (features == -1 || features == 2 * h))
                {
                    nodes.AddRange([batchFirst, joined]);
                    (sequences, output) = (false, joined.Outputs[0]);
                    break;
                }
            }

            throw Unsupported(rnn, "a recurrent layer whose outputs are not used as [batch, time, hidden] or the last state");
        }

        bool lstmLayer = op == "LSTM", bidirectional = directions == 2;
        bool candidateBias = layers.Any(l => l.Any(w => w.HiddenBias is not null));
        bool plain = !bidirectional && layers.Count == 1 && !candidateBias;
        int count = layers.Count;
        return new([.. nodes], output,
            builder => plain ? (lstmLayer ? builder.LSTM(h, sequences) : builder.GRU(h, sequences))
                : lstmLayer ? builder.LSTM(h, sequences, bidirectional, count) : builder.GRU(h, sequences, bidirectional, count, candidateBias),
            () => plain ? (lstmLayer ? new LSTM(inputs, h, sequences, device) : new GRU(inputs, h, sequences, device))
                : lstmLayer ? new LSTM(inputs, h, sequences, bidirectional, count, device) : new GRU(inputs, h, sequences, bidirectional, count, candidateBias, device),
            m =>
            {
                var module = (RecurrentModule)m;
                var all = layers.SelectMany(l => l).ToList();
                for (int i = 0; i < all.Count; i++)
                {
                    var target = module.Weights[i];
                    target.InputWeight.Load(all[i].InputWeight);
                    target.HiddenWeight.Load(all[i].HiddenWeight);
                    target.Bias.Load(all[i].Bias);
                    target.HiddenBias?.Load(all[i].HiddenBias ?? new float[h]);
                }
            });
    }

    // One direction of one layer in Idrak's layout: [inputs, G·H], [H, G·H], [G·H] and a GRU's candidate bias kept apart.
    private sealed record RecurrentLayerWeights(int Inputs, float[] InputWeight, float[] HiddenWeight, float[] Bias, float[]? HiddenBias);

    private RecurrentLayerWeights[] RecurrentWeightsOf(OnnxNode rnn, bool lstm, int h, int directions)
    {
        int gates = lstm ? 4 : 3;
        var order = lstm ? LstmOrder : GruOrder;
        var w = Const(rnn.Inputs[1]) ?? throw Unsupported(rnn, "computed weights");
        var rw = Const(rnn.Inputs[2]) ?? throw Unsupported(rnn, "computed weights");
        var b = rnn.Inputs.Count > 3 && rnn.Inputs[3].Length > 0 ? Const(rnn.Inputs[3]) ?? throw Unsupported(rnn, "a computed bias") : null;
        int inputs = w.Dims[2];
        var (wv, rv, bv) = (w.AsFloats(), rw.AsFloats(), b?.AsFloats());
        float[] Regroup(float[] source, int direction, int rows)
        {
            var result = new float[rows * gates * h];                                  // ours: [rows, G·H]
            int offset = direction * gates * h * rows;
            for (int gate = 0; gate < gates; gate++)
            {
                for (int j = 0; j < h; j++)
                {
                    for (int r = 0; r < rows; r++)
                    {
                        result[r * gates * h + order[gate] * h + j] = source[offset + (gate * h + j) * rows + r];
                    }
                }
            }

            return result;
        }

        var result = new RecurrentLayerWeights[directions];
        for (int d = 0; d < directions; d++)
        {
            var bias = new float[gates * h];
            float[]? apart = null;
            if (bv is not null)
            {
                int offset = d * 2 * gates * h;
                for (int gate = 0; gate < gates; gate++)
                {
                    for (int j = 0; j < h; j++)
                    {
                        float recurrent = bv[offset + (gates + gate) * h + j];
                        bias[order[gate] * h + j] = bv[offset + gate * h + j];
                        if (!lstm && order[gate] == gates - 1)
                        {
                            if (recurrent != 0f)
                            {
                                (apart ??= new float[h])[j] = recurrent;      // inside the reset gate's product: kept apart
                            }
                        }
                        else
                        {
                            bias[order[gate] * h + j] += recurrent;
                        }
                    }
                }
            }

            result[d] = new RecurrentLayerWeights(inputs, Regroup(wv, d, inputs), Regroup(rv, d, h), bias, apart);
        }

        return result;
    }

    // The recurrent node `value` feeds (as its input sequence) and the nodes computing its zero initial states, when every
    // other user of `value` belongs to those; null otherwise.
    private (OnnxNode Node, List<OnnxNode> State)? RecurrentNode(string value)
    {
        var users = Users(value);
        if (users.Where(u => u.Op is "LSTM" or "GRU" && u.Inputs[0] == value).ToList() is not [var rnn])
        {
            return null;
        }

        var state = new List<OnnxNode>();
        for (int i = 4; i < rnn.Inputs.Count; i++)
        {
            string input = rnn.Inputs[i];
            if (input.Length == 0 || i != 4 && Const(input) is { } zeros && zeros.AsFloats().All(v => v == 0f))
            {
                continue;
            }

            // Zeros expanded to a shape computed from `value` (or from constants).
            if (i is 5 or 6 && Producer(input) is { Op: "Expand" } expand && Const(expand.Inputs[0]) is { } fill && fill.AsFloats().All(v => v == 0f)
                && ShapeNodes(expand.Inputs[1], value, state))
            {
                state.Add(expand);
                continue;
            }

            return null;
        }

        return users.All(u => u == rnn || state.Contains(u)) ? (rnn, state) : null;
    }

    // Collects the nodes computing `name` from `value` and constants alone; false when anything else feeds them.
    private bool ShapeNodes(string name, string value, List<OnnxNode> nodes)
    {
        if (name == value || Const(name) is not null)
        {
            return true;
        }

        if (Producer(name) is not { } node || node.Op is "LSTM" or "GRU")
        {
            return false;
        }

        if (!nodes.Contains(node))
        {
            nodes.Add(node);
        }

        return node.Inputs.Where(i => i.Length > 0).All(i => ShapeNodes(i, value, nodes));
    }

    private Dictionary<string, OnnxNode>? _producers;

    private OnnxNode? Producer(string value)
    {
        _producers ??= model.Nodes.SelectMany(n => n.Outputs.Where(o => o.Length > 0).Select(o => (o, n))).GroupBy(p => p.o).ToDictionary(g => g.Key, g => g.First().n);
        return _producers.TryGetValue(value, out var node) ? node : null;
    }

    // An image's columns read as a sequence (NetworkBuilder.ColumnsToSequence): [N, C, H, W] transposed to [N, W, C, H] and
    // reshaped to [N, W, C·H].
    private LayerMatch? MatchColumnsToSequence(string x)
    {
        if (Only(x, "Transpose") is not { } transpose || transpose.Ints("perm") is not [0, 3, 1, 2] || Only(transpose.Outputs[0], "Reshape") is not { } reshape
            || _network.CurrentShape is not [var c, var h, var w] || Const(reshape.Inputs[1])?.AsLongs() is not [-1 or 0, var steps, var features]
            || steps != w || features != c * h)
        {
            return null;
        }

        return new([transpose, reshape], reshape.Outputs[0], b => b.ColumnsToSequence(), () => throw new NotSupportedException("ColumnsToSequence is a builder step."), null);
    }

    private long[]? SqueezeAxes(OnnxNode squeeze) => squeeze.Ints("axes") ?? (squeeze.Inputs.Count > 1 ? Const(squeeze.Inputs[1])?.AsLongs() : null);

    // ------------------------------------------------------------------ the rest

    private sealed record LayerMatch(OnnxNode[] Nodes, string Output, Func<NetworkBuilder, NetworkBuilder> Step, Func<Module> Create, Action<Module>? Load);

    private LayerMatch? LinearLayer(LinearMatch? l) => l is null ? null
        : new(l.Nodes, l.Output, b => b.Linear(l.Out, bias: l.Bias is not null), () => new Linear(l.In, l.Out, l.Bias is not null, device), m => LoadLinear((Linear)m, l));

    private LayerMatch? GeluLayer(GeluMatch? g) => g is null ? null : new(g.Nodes, g.Output, b => b.GELU(), () => new GELU(), null);

    private LayerMatch? AttentionLayer(AttentionMatch? a) => a is null ? null
        : new(a.Nodes, a.Output, b => b.MultiHeadAttention(a.Heads, a.Causal), () => new MultiHeadAttention(a.Qkv.In, a.Heads, a.Causal, 0f, device), m => LoadAttention(m, a));

    private OnnxTensor Required(OnnxNode node, int index) =>
        index < node.Inputs.Count && Const(node.Inputs[index]) is { } t ? t : throw Unsupported(node, $"a computed input {index} (weights must be constants)");

    private LayerMatch BatchNormLayer(OnnxNode node)
    {
        var (gamma, beta, mean, variance) = (Required(node, 1), Required(node, 2), Required(node, 3), Required(node, 4));
        float momentum = 1f - node.Float("momentum", 0.9f), epsilon = node.Float("epsilon", 1e-5f);
        return new([node], node.Outputs[0], b => b.BatchNorm(momentum, epsilon), () => new BatchNorm(gamma.Size, momentum, epsilon, device), m =>
        {
            var bn = (BatchNorm)m;
            bn.Gamma.Load(gamma.AsFloats());
            bn.Beta.Load(beta.AsFloats());
            bn.RunningMean.Load(mean.AsFloats());
            bn.RunningVariance.Load(variance.AsFloats());
        });
    }

    private LayerMatch LayerNormLayer(OnnxNode node)
    {
        var scale = Required(node, 1);
        float epsilon = node.Float("epsilon", 1e-5f);
        return new([node], node.Outputs[0], b => b.LayerNorm(epsilon), () => new LayerNorm(scale.Size, epsilon, device), m => LoadNorm((LayerNorm)m, node));
    }

    private LayerMatch ConvLayer(OnnxNode node)
    {
        var weight = Required(node, 1);
        var bias = node.Inputs.Count > 2 && node.Inputs[2].Length > 0 ? Required(node, 2) : null;
        if (weight.Dims is not [var outC, var perGroup, var kh, var kw])
        {
            throw Unsupported(node, "a convolution that is not 2-D");
        }

        var w = Window(node, ((int)kh, (int)kw), dilation: true);
        int groups = (int)node.Int("group", 1);
        if (w.Kernel != ((int)kh, (int)kw) || groups <= 0 || outC % groups != 0)
        {
            throw Unsupported(node, "a convolution whose kernel_shape or group does not match its weights");
        }

        int inC = (int)perGroup * groups, outputs = (int)outC;
        bool square = w.Square && w.Dilation == (1, 1) && groups == 1;
        return new([node], node.Outputs[0],
            b => square ? b.Conv2d(outputs, w.Kernel.Height, w.Stride.Height, w.Padding.Height, bias is not null)
                : b.Conv2d(outputs, w.Kernel, w.Stride, w.Padding, w.Dilation == (1, 1) ? null : w.Dilation, groups, bias is not null),
            () => new Conv2d(inC, outputs, w.Kernel, w.Stride, w.Padding, w.Dilation, groups, bias is not null, device), m =>
            {
                var conv = (Conv2d)m;
                conv.Weight.Load(weight.AsFloats());                                   // [out, in / groups, kh, kw] = [out, in / groups · kh · kw]
                if (bias is not null)
                {
                    conv.Bias!.Load(bias.AsFloats());
                }
            });
    }

    private LayerMatch MaxPoolLayer(OnnxNode node)
    {
        var w = Window(node, null, dilation: false, uneven: true);
        bool ceil = node.Int("ceil_mode", 0) != 0;
        if (node.Outputs.Count > 1 && node.Outputs[1].Length > 0 && Users(node.Outputs[1]).Count > 0)
        {
            throw Unsupported(node, "max pooling whose indices output is used");
        }

        bool square = w.Square && !ceil;
        var end = w.Uneven ? w.PaddingEnd : ((int, int)?)null;
        return new([node], node.Outputs[0],
            b => square ? b.MaxPool2d(w.Kernel.Height, w.Stride.Height, w.Padding.Height) : b.MaxPool2d(w.Kernel, w.Stride, w.Padding, ceil, end),
            () => new MaxPool2d(w.Kernel, w.Stride, w.Padding, ceil, end), null);
    }

    private LayerMatch AveragePoolLayer(OnnxNode node)
    {
        var w = Window(node, null, dilation: false, uneven: true);
        bool ceil = node.Int("ceil_mode", 0) != 0;
        bool countIncludePad = node.Int("count_include_pad", 0) != 0;                   // ONNX's default leaves padding out
        var end = w.Uneven ? w.PaddingEnd : ((int, int)?)null;
        return new([node], node.Outputs[0], b => b.AvgPool2d(w.Kernel, w.Stride, w.Padding, countIncludePad, ceil, end),
            () => new AvgPool2d(w.Kernel, w.Stride, w.Padding, countIncludePad, ceil, end), null);
    }

    // GlobalAveragePool / GlobalMaxPool kept as [C, 1, 1]: adaptive pooling to one position.
    private LayerMatch GlobalPoolLayer(OnnxNode node, bool max) => new([node], node.Outputs[0],
        b => max ? b.AdaptiveMaxPool2d(1) : b.AdaptiveAvgPool2d(1), () => max ? new AdaptiveMaxPool2d(1) : new AdaptiveAvgPool2d(1), null);

    private LayerMatch ConvTransposeLayer(OnnxNode node)
    {
        var weight = Required(node, 1);
        var bias = node.Inputs.Count > 2 && node.Inputs[2].Length > 0 ? Required(node, 2) : null;
        if (weight.Dims is not [var inC, var perGroup, var kh, var kw])
        {
            throw Unsupported(node, "a transposed convolution that is not 2-D");
        }

        if (node.Ints("output_shape") is not null)
        {
            throw Unsupported(node, "a transposed convolution given an output_shape (pads and output_padding are)");
        }

        var w = Window(node, ((int)kh, (int)kw), dilation: true);
        int groups = (int)node.Int("group", 1);
        if (w.Kernel != ((int)kh, (int)kw) || groups <= 0 || inC % groups != 0 || node.Ints("output_padding") is { } given && given.Length != 2)
        {
            throw Unsupported(node, "a transposed convolution whose kernel_shape, group or output_padding does not match its weights");
        }

        var outputPadding = node.Ints("output_padding") is [var oph, var opw] ? ((int)oph, (int)opw) : (0, 0);
        int inputs = (int)inC, outputs = (int)perGroup * groups;
        return new([node], node.Outputs[0],
            b => b.ConvTranspose2d(outputs, w.Kernel, w.Stride, w.Padding, outputPadding, w.Dilation == (1, 1) ? null : w.Dilation, groups, bias is not null),
            () => new ConvTranspose2d(inputs, outputs, w.Kernel, w.Stride, w.Padding, outputPadding, w.Dilation, groups, bias is not null, device), m =>
            {
                var conv = (ConvTranspose2d)m;
                conv.Weight.Load(weight.AsFloats());                                   // [in, out / groups, kh, kw] = [in, out / groups · kh · kw]
                if (bias is not null)
                {
                    conv.Bias!.Load(bias.AsFloats());
                }
            });
    }

    // Resize as Upsample: nearest with asymmetric coordinates and floor (PyTorch's nearest), linear with half_pixel (or
    // pytorch_half_pixel, the same but for one output position) or align_corners; scales [1, 1, h, w] or sizes [N, C, h, w]
    // as constants, or the (height, width) a matched shape computation gives (`size`).
    private LayerMatch ResizeLayer(OnnxNode node, (int Height, int Width)? size, params OnnxNode[] more)
    {
        string mode = node.String("mode") ?? "nearest", transform = node.String("coordinate_transformation_mode") ?? "half_pixel";
        string nearest = node.String("nearest_mode") ?? "round_prefer_floor";
        (InterpolationMode Mode, bool Align) how = (mode, transform) switch
        {
            ("nearest", "asymmetric") when nearest == "floor" => (InterpolationMode.Nearest, false),
            ("linear", "half_pixel" or "pytorch_half_pixel") => (InterpolationMode.Bilinear, false),
            ("linear", "align_corners") => (InterpolationMode.Bilinear, true),
            _ => throw Unsupported(node, $"a {mode} resize with {transform} coordinates{(mode == "nearest" ? $" and {nearest} rounding" : "")} (nearest is imported with asymmetric "
                + "coordinates and floor rounding, linear with half_pixel or align_corners)"),
        };
        if (node.Int("antialias", 0) != 0 || node.Ints("axes") is not null || node.Int("exclude_outside", 0) != 0)
        {
            throw Unsupported(node, "a resize with antialias, axes or exclude_outside");
        }

        // Opset 10 takes (X, scales); later opsets (X, roi, scales, sizes).
        string Input(int index) => index < node.Inputs.Count ? node.Inputs[index] : "";
        string scalesName = Opset < 11 ? Input(1) : Input(2), sizesName = Opset < 11 ? "" : Input(3);
        (float, float)? scale = null;
        if (size is null && sizesName.Length > 0)
        {
            size = Const(sizesName)?.AsLongs() is [_, _, var h, var w] ? ((int)h, (int)w) : throw Unsupported(node, "a resize to sizes computed in the graph");
        }
        else if (size is null)
        {
            scale = (scalesName.Length > 0 ? Const(scalesName)?.AsFloats() : null) is [1f, 1f, var sh, var sw] ? (sh, sw)
                : throw Unsupported(node, "a resize whose scales are computed or scale the batch or the channels");
        }

        return new([node, .. more], node.Outputs[0],
            b => scale is { } f ? b.Upsample(f, how.Mode, how.Align) : b.UpsampleToSize(size!.Value, how.Mode, how.Align),
            () => scale is { } f ? new Upsample(f, how.Mode, how.Align) : Upsample.ToSize(size!.Value, how.Mode, how.Align), null);
    }

    // PyTorch's resize to a size: Resize(x, sizes = Concat(Slice(Shape(x), 0, 2), [h, w])).
    private LayerMatch? MatchResize(string x)
    {
        foreach (var shape in Users(x).Where(n => n.Op == "Shape"))
        {
            if (Only(shape.Outputs[0], "Slice") is not { } slice || Const(slice.Inputs.ElementAtOrDefault(1) ?? "")?.AsLongs() is not [0]
                || Const(slice.Inputs.ElementAtOrDefault(2) ?? "")?.AsLongs() is not [2] || Only(slice.Outputs[0], "Concat") is not { Inputs.Count: 2 } concat
                || concat.Inputs[0] != slice.Outputs[0] || Const(concat.Inputs[1])?.AsLongs() is not [var h, var w]
                || Only(concat.Outputs[0], "Resize") is not { } resize || resize.Inputs[0] != x || resize.Inputs.ElementAtOrDefault(3) != concat.Outputs[0])
            {
                continue;
            }

            return ResizeLayer(resize, ((int)h, (int)w), shape, slice, concat);
        }

        return null;
    }

    // PyTorch's group norm before opset 18: Reshape(x, [0, groups, -1]), InstanceNormalization (unit scale, zero shift),
    // Reshape back (to a constant shape or Shape(x)), then Mul by gamma and Add beta per channel (left out without affine).
    private LayerMatch? MatchGroupNorm(string x)
    {
        foreach (var reshape in Users(x).Where(n => n.Op == "Reshape" && n.Inputs[0] == x))
        {
            if (Const(reshape.Inputs.ElementAtOrDefault(1) ?? "")?.AsLongs() is not [0, var g, -1] || Only(reshape.Outputs[0], "InstanceNormalization") is not { } norm
                || Const(norm.Inputs.ElementAtOrDefault(1) ?? "")?.AsFloats() is not { } ones || ones.Length != g || ones.Any(v => v != 1f)
                || Const(norm.Inputs.ElementAtOrDefault(2) ?? "")?.AsFloats() is not { } zeros || zeros.Length != g || zeros.Any(v => v != 0f)
                || Only(norm.Outputs[0], "Reshape") is not { } back)
            {
                continue;
            }

            List<OnnxNode> nodes = [reshape, norm, back];
            string target = back.Inputs.ElementAtOrDefault(1) ?? "";
            if (Const(target) is null)
            {
                if (Users(x).FirstOrDefault(n => n.Op == "Shape" && n.Outputs[0] == target) is not { } shape)
                {
                    continue;
                }

                nodes.Add(shape);
            }

            float[]? gamma = null, beta = null;
            string output = back.Outputs[0];
            if (Only(output, "Mul") is { } mul && Other(mul, output) is { } scaleName && Const(scaleName)?.AsFloats() is { } scale
                && Only(mul.Outputs[0], "Add") is { } add && Other(add, mul.Outputs[0]) is { } shiftName && Const(shiftName)?.AsFloats() is { } shift && shift.Length == scale.Length)
            {
                (gamma, beta, output) = (scale, shift, add.Outputs[0]);
                nodes.AddRange([mul, add]);
            }

            int groups = (int)g;
            float epsilon = norm.Float("epsilon", 1e-5f);
            int Channels() => gamma?.Length ?? (Const(target)?.AsLongs() is [_, var c, ..] && c > 0 ? (int)c : SampleShape(reshape, x)[0]);
            return new([.. nodes], output, b => b.GroupNorm(groups, epsilon, gamma is not null), () => new GroupNorm(groups, Channels(), epsilon, gamma is not null, device), m =>
            {
                var layer = (GroupNorm)m;
                if (gamma is not null)
                {
                    layer.Gamma!.Load(gamma);
                    layer.Beta!.Load(beta!);
                }
            });
        }

        return null;
    }

    // GroupNormalization: opset 18's scale and bias per group, opset 21's per channel.
    private LayerMatch GroupNormalizationLayer(OnnxNode node, int channels)
    {
        int groups = (int)node.Int("num_groups", 0);
        float[] scale = Required(node, 1).AsFloats(), bias = Required(node, 2).AsFloats();
        if (groups <= 0 || channels % groups != 0 || scale.Length != bias.Length || scale.Length != channels && scale.Length != groups || node.Int("stash_type", 1) != 1)
        {
            throw Unsupported(node, $"group normalization with {groups} groups, {scale.Length} scales and {bias.Length} biases over {channels} channels");
        }

        float[] PerChannel(float[] values) => values.Length == channels ? values : [.. Enumerable.Range(0, channels).Select(c => values[c / (channels / groups)])];
        float epsilon = node.Float("epsilon", 1e-5f);
        return new([node], node.Outputs[0], b => b.GroupNorm(groups, epsilon), () => new GroupNorm(groups, channels, epsilon, true, device), m =>
        {
            var layer = (GroupNorm)m;
            layer.Gamma!.Load(PerChannel(scale));
            layer.Beta!.Load(PerChannel(bias));
        });
    }

    // InstanceNormalization: group norm with one channel a group.
    private LayerMatch InstanceNormLayer(OnnxNode node)
    {
        float[] scale = Required(node, 1).AsFloats(), bias = Required(node, 2).AsFloats();
        int channels = scale.Length;
        float epsilon = node.Float("epsilon", 1e-5f);
        return new([node], node.Outputs[0], b => b.GroupNorm(channels, epsilon), () => new GroupNorm(channels, channels, epsilon, true, device), m =>
        {
            var layer = (GroupNorm)m;
            layer.Gamma!.Load(scale);
            layer.Beta!.Load(bias);
        });
    }

    private LayerMatch? EmbeddingLayer(OnnxNode gather, string ids) =>
        gather.Op == "Gather" && gather.Inputs.Count == 2 && gather.Inputs[1] == ids && gather.Int("axis", 0) == 0 && Const(gather.Inputs[0]) is { Dims.Length: 2 } table
            ? new([gather], gather.Outputs[0], b => b.Embedding(table.Dims[0], table.Dims[1]), () => new Embedding(table.Dims[0], table.Dims[1], device),
                m => ((Embedding)m).Weight.Load(table.AsFloats()))
            : null;

    private sealed record WindowShape((int Height, int Width) Kernel, (int Height, int Width) Stride, (int Height, int Width) Padding, (int Height, int Width) Dilation,
        (int Height, int Width) PaddingEnd)
    {
        public bool Uneven => PaddingEnd != Padding;

        public bool Square => Kernel.Height == Kernel.Width && Stride.Height == Stride.Width && Padding.Height == Padding.Width && !Uneven;
    }

    // A 2-D window: kernel_shape (or the weights' kernel), strides, explicit padding (equal above and below and left and
    // right, unless the layer takes more below and right: `uneven`), and dilations where the layer takes them.
    private static WindowShape Window(OnnxNode node, (int, int)? weights, bool dilation, bool uneven = false)
    {
        var kernel = node.Ints("kernel_shape") ?? (weights is var (wh, ww) ? [wh, ww] : null);
        var strides = node.Ints("strides") ?? [1, 1];
        var pads = node.Ints("pads") ?? [0, 0, 0, 0];
        var dilations = node.Ints("dilations") ?? [1, 1];
        if (node.String("auto_pad") is { } autoPad && autoPad != "NOTSET" || kernel is not [var kh, var kw] || strides is not [var sh, var sw]
            || pads is not [var top, var left, var bottom, var right] || !uneven && (top != bottom || left != right) || dilations is not [var dh, var dw]
            || !dilation && (dh != 1 || dw != 1))
        {
            throw Unsupported(node, "a 2-D window with explicit padding" + (uneven ? "" : ", equal above and below and left and right") + (dilation ? "" : ", and no dilation"));
        }

        return new(((int)kh, (int)kw), ((int)sh, (int)sw), ((int)top, (int)left), ((int)dh, (int)dw), ((int)bottom, (int)right));
    }

    private string Gather(OnnxNode node, string x)
    {
        if (EmbeddingLayer(node, x) is { } embedding)
        {
            return PushMatch(embedding);
        }

        if (node.Inputs[0] == x && node.Int("axis", 0) == 1 && Const(node.Inputs[1]) is { Dims.Length: 0 } index && _network.CurrentShape.Count == 2)
        {
            long i = index.AsLongs()[0], t = _network.CurrentShape[0];
            if (i == 0)
            {
                return Push(b => b.FirstStep(), null, node.Outputs[0]);
            }

            if (i == t - 1 || i == -1)
            {
                return Push(b => b.LastStep(), null, node.Outputs[0]);
            }
        }

        throw Unsupported(node, "a gather that is not an embedding lookup or the first/last time step");
    }

    private string PositionTable(OnnxNode node, string x)
    {
        if (Const(Other(node, x) ?? "") is { Dims: [var t, var d] } table && _network.CurrentShape.SequenceEqual([t, d]))
        {
            var values = table.AsFloats();
            for (int pos = 0; pos < t; pos++)
            {
                for (int i = 0; i < d; i++)
                {
                    double angle = pos / Math.Pow(10000, 2 * (i / 2) / (double)d);
                    if (MathF.Abs(values[pos * d + i] - (float)(i % 2 == 0 ? Math.Sin(angle) : Math.Cos(angle))) > 1e-4f)
                    {
                        throw Unsupported(node, "a learned (non-sinusoidal) position table");
                    }
                }
            }

            return Push(b => b.PositionalEncoding(), null, node.Outputs[0]);
        }

        throw Unsupported(node, "adding a value that is not a position table");
    }
}
