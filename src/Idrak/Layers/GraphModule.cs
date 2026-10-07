// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Layers;

/// <summary>
/// A network whose layers form a graph rather than a chain: values can be used several times (skip connections,
/// branches) and combined. Nodes run in order; each is a layer (any type registered in <see cref="LayerTypes"/>:
/// <see cref="Linear"/>, <see cref="Conv2d"/>, <see cref="BatchNorm"/>, … or your own) or an operation: the structural
/// ones (identity, cast, add, sub, mul, div, concat, shape, gather, slice, unsqueeze, squeeze, reshape) and those
/// registered in <see cref="GraphOps"/> (matmul, relu, tanh, sigmoid, exp, log, abs, neg, sqrt, pow, clip, leaky_relu,
/// elu, hard_sigmoid, hard_swish, max, min, softmax, flatten, transpose, reduce_mean, global_average_pool, or your own),
/// following the ONNX operators of the same names. Integer values such as shapes are computed on the host, so shape
/// arithmetic works for any batch size. Constants are buffers: they move with the model and are saved with its weights.
/// <see cref="ToJson"/> and <see cref="FromJson"/> describe the graph so packages can rebuild it. The ONNX importer
/// creates these for models with skip connections, and the ONNX exporter writes them back.
/// </summary>
public sealed class GraphModule : Module
{
    private const string Format = "idrak-graph/1";
    private readonly List<GraphNode> _nodes;
    private readonly List<string> _constantNames;
    private readonly Dictionary<string, Tensor> _constants;
    private readonly Dictionary<string, GraphValues.HostValue> _integers;

    /// <summary>Creates the graph.</summary>
    /// <param name="input">Name of the input value.</param>
    /// <param name="output">Name of the value returned.</param>
    /// <param name="nodes">The operations, in an order where every input is available when its node runs.</param>
    /// <param name="constants">Float constants (the module takes ownership).</param>
    /// <param name="integers">Integer constants (shapes, axes, indices) with their dimensions.</param>
    public GraphModule(string input, string output, IEnumerable<GraphNode> nodes, IEnumerable<(string Name, Tensor Value)>? constants = null,
        IEnumerable<(string Name, long[] Values, int[] Dims)>? integers = null)
    {
        Input = input;
        Output = output;
        _nodes = [.. nodes];
        var floats = constants?.ToList() ?? [];
        _constantNames = [.. floats.Select(c => c.Name)];
        _constants = floats.ToDictionary(c => c.Name, c => c.Value);
        _integers = (integers ?? []).ToDictionary(c => c.Name, c => new GraphValues.HostValue(c.Values, c.Dims));
        var known = new HashSet<string>([input, .. _constants.Keys, .. _integers.Keys]);
        foreach (var node in _nodes)
        {
            if (node.Inputs.FirstOrDefault(i => i.Length > 0 && !known.Contains(i)) is { } missing)
            {
                throw new ArgumentException($"Node '{node.Output}' ({node.Op}) uses '{missing}' before it is computed.");
            }

            if (node.Op == "layer" && node.Layer is null)
            {
                throw new ArgumentException($"Node '{node.Output}' is a layer node without a layer.");
            }

            if (!GraphOps.Contains(node.Op))
            {
                throw new ArgumentException($"Node '{node.Output}' uses the unknown operation '{node.Op}' (known: {string.Join(", ", GraphOps.Names)}); add it with GraphOps.Register.");
            }

            known.Add(node.Output);
        }

        if (!known.Contains(output))
        {
            throw new ArgumentException($"The output '{output}' is never computed.");
        }
    }

    /// <summary>Name of the input value.</summary>
    public string Input { get; }

    /// <summary>Name of the output value.</summary>
    public string Output { get; }

    /// <summary>The operations in execution order.</summary>
    public IReadOnlyList<GraphNode> Nodes => _nodes;

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => _nodes.Where(n => n.Layer is not null).Select(n => n.Layer!);

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => base.Buffers().Concat(_constantNames.Select(n => _constants[n]));

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        base.MoveTo(device);
        foreach (var name in _constantNames)
        {
            _constants[name] = MoveTensor(_constants[name], device);
        }
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var constant in _constants.Values)
        {
            constant.Dispose();
        }

        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"GraphModule({_nodes.Count} nodes, {Children().Count()} layers)";

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) =>
        Trace(input)[Output] as Tensor ?? throw new InvalidOperationException($"The output '{Output}' is an integer value, not a tensor.");

    // Every value the graph computes for `input`, by name: tensors, and integers computed on the host (GraphValues.HostValue).
    internal Dictionary<string, object> Trace(Tensor input)
    {
        var values = new Dictionary<string, object> { [Input] = input };
        foreach (var node in _nodes)
        {
            values[node.Output] = RunNode(node, name => values.TryGetValue(name, out var v) ? v
                : _constants.TryGetValue(name, out var c) ? c
                : _integers.TryGetValue(name, out var h) ? h
                : null);
        }

        return values;
    }

    // The float constants in order, and the integer constants.
    internal IReadOnlyList<(string Name, Tensor Value)> Constants => [.. _constantNames.Select(n => (n, _constants[n]))];

    internal IReadOnlyList<(string Name, long[] Values, int[] Dims)> IntegerConstants => [.. _integers.Select(p => (p.Key, p.Value.Values, p.Value.Dims))];

    // Runs one node, reading its inputs through `lookup` (null for a value that is not available).
    internal static object RunNode(GraphNode node, Func<string, object?> lookup)
    {
        object Arg(int i)
        {
            string name = node.Inputs[i];
            return lookup(name) ?? throw new InvalidOperationException($"Value '{name}' is not available.");
        }

        try
        {
            return Run(node, Arg, node.Inputs.Count);
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or NotSupportedException)
        {
            throw new InvalidOperationException($"Graph node '{node.Output}' ({node.Op}) failed: {ex.Message}", ex);
        }
    }

    // An integer value computed on the host, as the graph holds it.
    internal static object IntegerValue(long[] values, int[] dims) => new GraphValues.HostValue(values, dims);

    // A value's integers: a host value's own, or a tensor's values truncated.
    internal static long[] AsIntegers(object value) => GraphValues.AsIntegers(value);

    // ------------------------------------------------------------------ operations


    private static object Run(GraphNode node, Func<int, object> arg, int count)
    {
        var a = node.Attributes;
        Tensor T(int i) => arg(i) as Tensor ?? throw new InvalidOperationException($"input {i} must be a tensor");
        long[] Ints(int i) => AsIntegers(arg(i));
        long[]? Axes(int inputIndex) => a?["axes"] is JsonArray axes ? [.. axes.Select(v => (long)v!)] : count > inputIndex && node.Inputs[inputIndex].Length > 0 ? Ints(inputIndex) : null;
        long Int(string name, long fallback) => a?[name] is JsonValue v ? (long)v : fallback;

        switch (node.Op)
        {
            case "layer":
                return node.Layer!.Forward(T(0));
            case "identity" or "cast":
                return arg(0);
            case "add" or "sub" or "mul" or "div":
                return Binary(node.Op, arg(0), arg(1));
            case "reshape":
                return Reshape(T(0), Ints(1), Int("allowzero", 0) == 1);
            case "concat":
                int concatAxis = (int)Int("axis", 0);
                var parts = Enumerable.Range(0, count).Select(arg).ToList();
                if (parts.All(v => v is GraphValues.HostValue))
                {
                    return new GraphValues.HostValue([.. parts.SelectMany(v => ((GraphValues.HostValue)v).Values)], [parts.Sum(v => ((GraphValues.HostValue)v).Values.Length)]);
                }

                var tensors = parts.Select(v => v as Tensor ?? throw new NotSupportedException("concatenating tensors with integer values")).ToList();
                return Tensor.Concat(tensors, Normalize(concatAxis, tensors[0].Rank));
            case "shape":
                var shape = arg(0) is Tensor st ? st.Shape.ToArray() : ((GraphValues.HostValue)arg(0)).Dims;
                int start = Normalize((int)Int("start", 0), shape.Length + 1), end = a?["end"] is JsonValue e ? Normalize((int)e, shape.Length + 1) : shape.Length;
                return new GraphValues.HostValue([.. shape[start..end].Select(d => (long)d)], [end - start]);
            case "gather":
                return Gather(arg(0), arg(1), (int)Int("axis", 0));
            case "slice":
                return Slice(arg(0), Ints(1), Ints(2), count > 3 && node.Inputs[3].Length > 0 ? Ints(3) : null, count > 4 && node.Inputs[4].Length > 0 ? Ints(4) : null);
            case "unsqueeze":
                return Unsqueeze(arg(0), Axes(1) ?? throw new InvalidOperationException("unsqueeze needs axes"));
            case "squeeze":
                return Squeeze(arg(0), Axes(1));
            default:
                var op = GraphOps.TryGet(node.Op) ?? throw new NotSupportedException($"unknown operation '{node.Op}'; register it with GraphOps.Register");
                return op(new GraphOpContext(node, arg));
        }
    }

    internal static int Normalize(int axis, int rank) => axis < 0 ? axis + rank : axis;

    internal static object Binary(string op, object left, object right)
    {
        if (left is GraphValues.HostValue hl && right is GraphValues.HostValue hr)
        {
            long Apply(long x, long y) => op switch { "add" => x + y, "sub" => x - y, "mul" => x * y, _ => x / y };
            return hl.Values.Length == hr.Values.Length ? new GraphValues.HostValue([.. hl.Values.Zip(hr.Values, Apply)], hl.Dims)
                : hr.Values.Length == 1 ? new GraphValues.HostValue([.. hl.Values.Select(v => Apply(v, hr.Values[0]))], hl.Dims)
                : new GraphValues.HostValue([.. hr.Values.Select(v => Apply(hl.Values[0], v))], hr.Dims);
        }

        var a = left as Tensor ?? throw new NotSupportedException("integer values with tensors");
        var b = right as Tensor ?? throw new NotSupportedException("integer values with tensors");
        if (b.Size == 1 && a.Size != 1 || b.Size == 1 && a.Size == 1 && b.Rank <= a.Rank)
        {
            float v = b.ToArray()[0];
            return op switch { "add" => a + v, "sub" => a - v, "mul" => a * v, _ => a / v };
        }

        if (a.Size == 1)
        {
            float v = a.ToArray()[0];
            return op switch
            {
                "add" => b + v,
                "sub" => v - b,
                "mul" => b * v,
                _ => throw new NotSupportedException("dividing a scalar by a tensor"),
            };
        }

        if (a.Shape.SequenceEqual(b.Shape))
        {
            return op switch { "add" => a + b, "sub" => a - b, "mul" => a * b, _ => throw new NotSupportedException("dividing by a tensor") };
        }

        // Broadcasting: the smaller operand (leading 1-sized dimensions dropped) must equal the other's trailing dimensions.
        var small = Trim(b);
        if (op is "add" or "sub" && small.Rank < a.Rank && a.Shape[(a.Rank - small.Rank)..].SequenceEqual(small.Shape))
        {
            return op == "add" ? a + small : a + small * -1f;
        }

        small = Trim(a);
        if (op == "add" && small.Rank < b.Rank && b.Shape[(b.Rank - small.Rank)..].SequenceEqual(small.Shape))
        {
            return b + small;
        }

        throw new NotSupportedException($"{op} of shapes {Tensor.FormatShape(a.Shape)} and {Tensor.FormatShape(b.Shape)} (only equal shapes, scalars and trailing broadcasts)");
    }

    // Drops leading 1-sized dimensions of a constant-like operand ([1, C] → [C]) so trailing broadcasts line up.
    private static Tensor Trim(Tensor t)
    {
        int lead = 0;
        while (lead < t.Rank - 1 && t.Shape[lead] == 1)
        {
            lead++;
        }

        return lead == 0 ? t : t.Reshape(t.Shape[lead..]);
    }

    private static Tensor Reshape(Tensor x, long[] target, bool allowZero)
    {
        var shape = new int[target.Length];
        for (int i = 0; i < target.Length; i++)
        {
            shape[i] = target[i] == 0 && !allowZero ? x.Shape[i] : (int)target[i];
        }

        return x.Reshape(shape);
    }

    private static object Gather(object data, object indices, int axis)
    {
        var index = indices switch
        {
            GraphValues.HostValue h => h,
            Tensor t => new GraphValues.HostValue([.. t.ToArray().Select(v => (long)v)], t.Shape.ToArray()),
            _ => throw new InvalidOperationException("bad indices"),
        };
        if (data is GraphValues.HostValue host)
        {
            long Pick(long i) => host.Values[i < 0 ? i + host.Values.Length : i];
            return new GraphValues.HostValue([.. index.Values.Select(Pick)], index.Dims);
        }

        var x = (Tensor)data;
        axis = Normalize(axis, x.Rank);
        var pieces = index.Values.Select(i => x.Narrow(axis, (int)(i < 0 ? i + x.Shape[axis] : i), 1)).ToList();
        var gathered = pieces.Count == 1 ? pieces[0] : Tensor.Concat(pieces, axis);
        int[] shape = [.. x.Shape[..axis].ToArray(), .. index.Dims, .. x.Shape[(axis + 1)..].ToArray()];
        return gathered.Reshape(shape);
    }

    private static object Slice(object data, long[] starts, long[] ends, long[]? axes, long[]? steps)
    {
        if (steps is not null && steps.Any(s => s != 1))
        {
            throw new NotSupportedException("slices with a step other than 1");
        }

        if (data is GraphValues.HostValue host)
        {
            int n = host.Values.Length;
            int from = (int)Math.Clamp(starts[0] < 0 ? starts[0] + n : starts[0], 0, n), to = (int)Math.Clamp(ends[0] < 0 ? ends[0] + n : ends[0], 0, n);
            return new GraphValues.HostValue(host.Values[from..Math.Max(from, to)], [Math.Max(0, to - from)]);
        }

        var x = (Tensor)data;
        for (int i = 0; i < starts.Length; i++)
        {
            int axis = Normalize((int)(axes?[i] ?? i), x.Rank), size = x.Shape[axis];
            int from = (int)Math.Clamp(starts[i] < 0 ? starts[i] + size : starts[i], 0, size);
            int to = (int)Math.Clamp(ends[i] < 0 ? ends[i] + size : ends[i], 0, size);
            if (from != 0 || to != size)
            {
                x = x.Narrow(axis, from, Math.Max(0, to - from));
            }
        }

        return x;
    }

    private static object Unsqueeze(object data, long[] axes)
    {
        int[] dims = data is Tensor t ? t.Shape.ToArray() : ((GraphValues.HostValue)data).Dims;
        var result = dims.ToList();
        foreach (int axis in axes.Select(v => Normalize((int)v, dims.Length + axes.Length)).Order())
        {
            result.Insert(axis, 1);
        }

        return data is Tensor tensor ? tensor.Reshape([.. result]) : new GraphValues.HostValue(((GraphValues.HostValue)data).Values, [.. result]);
    }

    private static object Squeeze(object data, long[]? axes)
    {
        int[] dims = data is Tensor t ? t.Shape.ToArray() : ((GraphValues.HostValue)data).Dims;
        var drop = axes?.Select(v => Normalize((int)v, dims.Length)).ToHashSet() ?? [.. Enumerable.Range(0, dims.Length).Where(i => dims[i] == 1)];
        int[] result = [.. dims.Where((d, i) => !drop.Contains(i))];
        return data is Tensor tensor ? tensor.Reshape(result) : new GraphValues.HostValue(((GraphValues.HostValue)data).Values, result);
    }

    // ------------------------------------------------------------------ description

    /// <summary>
    /// The graph as JSON (format "idrak-graph/1"): nodes with their layers' settings, constants' shapes and the
    /// integer constants. Weights and float constants are not included; save them with <see cref="Idrak.ModuleFiles.Save(Module, string)"/>.
    /// </summary>
    public JsonObject ToJson() => new()
    {
        ["format"] = Format,
        ["input"] = Input,
        ["output"] = Output,
        ["constants"] = new JsonArray([.. _constantNames.Select(n => (JsonNode)new JsonObject
        {
            ["name"] = n,
            ["shape"] = new JsonArray([.. _constants[n].Shape.ToArray().Select(d => (JsonNode)d)]),
        })]),
        ["integers"] = new JsonArray([.. _integers.Select(p => (JsonNode)new JsonObject
        {
            ["name"] = p.Key,
            ["values"] = new JsonArray([.. p.Value.Values.Select(v => (JsonNode)v)]),
            ["dims"] = new JsonArray([.. p.Value.Dims.Select(d => (JsonNode)d)]),
        })]),
        ["nodes"] = new JsonArray([.. _nodes.Select(n =>
        {
            var node = new JsonObject
            {
                ["op"] = n.Op,
                ["inputs"] = new JsonArray([.. n.Inputs.Select(i => (JsonNode)i)]),
                ["output"] = n.Output,
            };
            if (n.Attributes is not null)
            {
                node["attributes"] = n.Attributes.DeepClone();
            }

            if (n.Layer is not null)
            {
                node["layer"] = LayerTypes.Describe(n.Layer);
            }

            return (JsonNode)node;
        })]),
    };

    /// <summary>Rebuilds a graph described by <see cref="ToJson"/> on <paramref name="device"/>, with zero constants and fresh layers (load the weights next).</summary>
    public static GraphModule FromJson(JsonObject description, Device? device = null)
    {
        if ((string?)description["format"] != Format)
        {
            throw new InvalidDataException($"Not a Idrak graph description (format '{description["format"]}').");
        }

        device ??= Device.Default;
        var constants = description["constants"]!.AsArray().Select(c =>
        {
            int[] shape = [.. c!["shape"]!.AsArray().Select(d => (int)d!)];
            return ((string)c["name"]!, Tensor.Persistent(new float[shape.Aggregate(1, (x, y) => x * y)], shape, device, requiresGrad: false));
        }).ToList();
        var integers = description["integers"]!.AsArray().Select(c =>
            ((string)c!["name"]!, c["values"]!.AsArray().Select(v => (long)v!).ToArray(), c["dims"]!.AsArray().Select(v => (int)v!).ToArray()));
        var nodes = description["nodes"]!.AsArray().Select(n => new GraphNode(
            (string)n!["op"]!,
            [.. n["inputs"]!.AsArray().Select(i => (string)i!)],
            (string)n["output"]!,
            n["attributes"]?.DeepClone().AsObject(),
            n["layer"] is JsonObject layer ? LayerTypes.Create(layer, device) : null));
        return new GraphModule((string)description["input"]!, (string)description["output"]!, nodes, constants, integers);
    }

    /// <summary>Whether <paramref name="description"/> is a graph description (rather than a network builder's).</summary>
    public static bool IsDescription(JsonObject description) => (string?)description["format"] == Format;
}
