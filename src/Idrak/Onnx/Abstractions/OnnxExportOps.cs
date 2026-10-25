// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

namespace Idrak.Onnx.Abstractions;

/// <summary>
/// How modules of one type are written to ONNX, registered by that type in <see cref="OnnxExportOps"/> (it serves types
/// derived from it without a translator of their own too). Implement <see cref="IOnnxExportOp{T}"/>, typed by the module.
/// </summary>
public interface IOnnxExportOp
{
    /// <summary>The module type it translates.</summary>
    Type ModuleType { get; }

    /// <summary>
    /// Translates <paramref name="module"/> into ONNX nodes of <paramref name="graph"/>: <paramref name="input"/> is the
    /// module's input value, <paramref name="outputShape"/> the shape it produces (measured by running the module, -1 for
    /// the batch dimension). Returns the output value.
    /// </summary>
    OnnxValue Export(OnnxGraph graph, Module module, OnnxValue input, IReadOnlyList<int> outputShape);
}

/// <summary>How modules of type <typeparamref name="T"/> are written to ONNX (<see cref="IOnnxExportOp"/>).</summary>
/// <typeparam name="T">The module type.</typeparam>
public interface IOnnxExportOp<T> : IOnnxExportOp where T : Module
{
    /// <summary>Translates <paramref name="module"/> into ONNX nodes (see <see cref="IOnnxExportOp.Export"/>).</summary>
    OnnxValue Export(OnnxGraph graph, T module, OnnxValue input, IReadOnlyList<int> outputShape);

    Type IOnnxExportOp.ModuleType => typeof(T);

    OnnxValue IOnnxExportOp.Export(OnnxGraph graph, Module module, OnnxValue input, IReadOnlyList<int> outputShape) => Export(graph, (T)module, input, outputShape);
}

/// <summary>
/// How lambda layers of one name (<see cref="INamedLambda.Name"/>) are written to ONNX, registered by its
/// <see cref="IKind.Name"/> in <see cref="OnnxExportOps"/>.
/// </summary>
public interface IOnnxLambdaExport : IKind
{
    /// <summary>Translates the lambda <paramref name="module"/> into ONNX nodes (see <see cref="IOnnxExportOp.Export"/>).</summary>
    OnnxValue Export(OnnxGraph graph, Module module, OnnxValue input, IReadOnlyList<int> outputShape);
}

/// <summary>
/// How graph module nodes (<see cref="GraphNode"/>) running one operation are written to ONNX, registered by its
/// <see cref="IKind.Name"/> (the operation's, as <see cref="GraphOps"/> names it) in <see cref="OnnxExportOps"/>.
/// </summary>
public interface IOnnxGraphOpExport : IKind
{
    /// <summary>Translates the node into ONNX nodes and returns the output value.</summary>
    OnnxValue Export(OnnxGraphOpContext context);
}

/// <summary>
/// Translates one module into ONNX nodes: <paramref name="input"/> is the module's input value, <paramref name="outputShape"/>
/// the shape it produces (measured by running the module, -1 for the batch dimension). Returns the output value. A
/// translator as a function (see <see cref="OnnxExportOps.Register{T}(OnnxTranslator{T})"/>, which makes it an
/// <see cref="IOnnxExportOp{T}"/>), or for one export (<c>OnnxExporter.Module</c> in Idrak).
/// </summary>
public delegate OnnxValue OnnxTranslator<in T>(OnnxGraph graph, T module, OnnxValue input, IReadOnlyList<int> outputShape) where T : Module;

/// <summary>
/// Translates one operation node of a graph module (<see cref="GraphNode"/>, <see cref="GraphOps"/>) into ONNX nodes and
/// returns the output value: a translator as a function (see <see cref="OnnxExportOps.RegisterGraphOp(string, OnnxGraphOpTranslator)"/>,
/// which makes it an <see cref="IOnnxGraphOpExport"/>, or for one export, <c>OnnxExporter.GraphOp</c> in Idrak).
/// </summary>
public delegate OnnxValue OnnxGraphOpTranslator(OnnxGraphOpContext context);

/// <summary>The graph node being exported, its input values and its measured output shape, for an <see cref="IOnnxGraphOpExport"/>.</summary>
public sealed class OnnxGraphOpContext
{
    private readonly Func<int, long[]?> _integers;

    /// <summary>
    /// The context of exporting <paramref name="node"/> into <paramref name="graph"/>: its inputs as ONNX values, its
    /// measured output shape, and <paramref name="integers"/>, which gives the integer values of an input by its index
    /// (null for a tensor). Exporters create it; translators read it.
    /// </summary>
    public OnnxGraphOpContext(OnnxGraph graph, GraphNode node, IReadOnlyList<OnnxValue?> inputs, IReadOnlyList<int>? outputShape, Func<int, long[]?> integers)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(node);
        ArgumentNullException.ThrowIfNull(inputs);
        ArgumentNullException.ThrowIfNull(integers);
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

/// <summary>
/// How modules, lambdas and graph operations are written to ONNX, for every export. Idrak registers the built-in
/// layers (Linear, Conv2d, BatchNorm, LSTM, Sequential, GraphModule, ...), the builder's lambdas (MeanOverTime,
/// FirstStep, LastStep, Reshape: <see cref="INamedLambda"/>) and every built-in graph operation (<see cref="GraphOps"/>);
/// add or replace one with <see cref="Register(IOnnxExportOp)"/>, <see cref="RegisterLambda(IOnnxLambdaExport)"/> or
/// <see cref="RegisterGraphOp(IOnnxGraphOpExport)"/>. A module uses the translator registered for its own type or its
/// nearest registered base type (<see cref="Find"/>). Translators given to one exporter (<c>OnnxExporter.Module</c>,
/// <c>Lambda</c> and <c>GraphOp</c> in Idrak) take precedence over these.
/// </summary>
public static class OnnxExportOps
{
    private const string Unguarded = "a translator writes nodes into the ONNX graph as it runs, and a half-run translator cannot be undone";

    private static readonly SlotTable<Type, IOnnxExportOp> Modules = new("OnnxExportOps.Modules", unguarded: Unguarded);
    private static readonly KindTable<IOnnxLambdaExport> Lambdas = new("OnnxExportOps.Lambdas", "ONNX lambda translator", "add it with OnnxExportOps.RegisterLambda",
        comparer: StringComparer.Ordinal, unguarded: Unguarded);
    private static readonly KindTable<IOnnxGraphOpExport> GraphOpTranslators = new("OnnxExportOps.GraphOps", "ONNX graph operation translator",
        "add it with OnnxExportOps.RegisterGraphOp", comparer: StringComparer.Ordinal, unguarded: Unguarded);

    static OnnxExportOps() => Overrides.AsLibraryDefaults(OnnxBuiltIns.RegisterExports);   // the built-in translators, on first use

    /// <summary>
    /// Registers <paramref name="op"/> for modules of its <see cref="IOnnxExportOp.ModuleType"/> (and types derived from it
    /// without their own translator); for a type the library translates it overrides the library's until <see cref="Unregister{T}"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IOnnxExportOp op)
    {
        ArgumentNullException.ThrowIfNull(op);
        Modules.Register(op.ModuleType, op, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>
    /// Registers how modules of type <typeparamref name="T"/> are exported, by a function (made into an
    /// <see cref="IOnnxExportOp{T}"/>; see <see cref="Register(IOnnxExportOp)"/>).
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register<T>(OnnxTranslator<T> translate) where T : Module
    {
        ArgumentNullException.ThrowIfNull(translate);
        Modules.Register(typeof(T), new FunctionOp<T>(translate), System.Reflection.Assembly.GetCallingAssembly(), translate);
    }

    /// <summary>Removes the app's translator of modules of type <typeparamref name="T"/> (the library's comes back); false when the app registered none.</summary>
    public static bool Unregister<T>() where T : Module => Modules.Unregister(typeof(T));

    /// <summary>The module types with a registered translator.</summary>
    public static IReadOnlyCollection<Type> Types => Modules.Keys;

    /// <summary>The translator registered for modules of exactly the type <typeparamref name="T"/>.</summary>
    public static IOnnxExportOp Get<T>() where T : Module =>
        Modules.TryGet(typeof(T), out var op) ? op
            : throw new NotSupportedException($"No ONNX export translator for {typeof(T).Name} is registered ({string.Join(", ", Types.Select(t => t.Name).Order())}); add it with OnnxExportOps.Register<{typeof(T).Name}>.");

    /// <summary>The library's translator of modules of type <typeparamref name="T"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IOnnxExportOp? Default<T>() where T : Module => Modules.HasDefault(typeof(T)) ? Modules.Default(typeof(T)) : null;

    /// <summary>Who registered the translator of modules of type <typeparamref name="T"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin<T>() where T : Module => Modules.Origin(typeof(T));

    /// <summary>Registers <paramref name="translate"/> for the lambdas of its <see cref="IKind.Name"/>; a built-in name overrides the library's until <see cref="UnregisterLambda"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void RegisterLambda(IOnnxLambdaExport translate) => Lambdas.Register(translate, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Registers how the lambdas named <paramref name="name"/> (<see cref="INamedLambda.Name"/>) are exported, by a function (made into an <see cref="IOnnxLambdaExport"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void RegisterLambda(string name, OnnxTranslator<Module> translate)
    {
        ArgumentException.ThrowIfNullOrEmpty(name);
        ArgumentNullException.ThrowIfNull(translate);
        Lambdas.Slots.Register(name, new FunctionLambda(name, translate), System.Reflection.Assembly.GetCallingAssembly(), translate);
    }

    /// <summary>Removes the app's lambda translator <paramref name="name"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool UnregisterLambda(string name) => Lambdas.Unregister(name);

    /// <summary>The lambda names with a registered translator.</summary>
    public static IReadOnlyCollection<string> LambdaNames => Lambdas.Names;

    /// <summary>The library's translator of the lambdas named <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IOnnxLambdaExport? DefaultLambda(string name) => Lambdas.Default(name);

    /// <summary>Who registered the lambda translator <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? LambdaOrigin(string name) => Lambdas.Origin(name);

    /// <summary>Registers <paramref name="translate"/> for the graph module nodes running the operation of its <see cref="IKind.Name"/>; a built-in operation overrides the library's until <see cref="UnregisterGraphOp"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void RegisterGraphOp(IOnnxGraphOpExport translate) => GraphOpTranslators.Register(translate, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Registers how graph module nodes (<see cref="GraphNode"/>) running the operation <paramref name="op"/> are exported, by a function (made into an <see cref="IOnnxGraphOpExport"/>).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void RegisterGraphOp(string op, OnnxGraphOpTranslator translate)
    {
        ArgumentException.ThrowIfNullOrEmpty(op);
        ArgumentNullException.ThrowIfNull(translate);
        GraphOpTranslators.Slots.Register(op, new FunctionGraphOp(op, translate), System.Reflection.Assembly.GetCallingAssembly(), translate);
    }

    /// <summary>Removes the app's graph operation translator <paramref name="op"/> (a built-in operation gets the library's back); false when the app registered none.</summary>
    public static bool UnregisterGraphOp(string op) => GraphOpTranslators.Unregister(op);

    /// <summary>The graph operations with a registered translator.</summary>
    public static IReadOnlyCollection<string> GraphOpNames => GraphOpTranslators.Names;

    /// <summary>The translator registered for the graph operation <paramref name="op"/>.</summary>
    public static IOnnxGraphOpExport GetGraphOp(ReadOnlySpan<char> op) => FindGraphOp(op)
        ?? throw new NotSupportedException($"No ONNX export translator for the graph operation '{op}' is registered ({string.Join(", ", GraphOpNames)}); add it with OnnxExportOps.RegisterGraphOp.");

    /// <summary>The library's translator of the graph operation <paramref name="op"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IOnnxGraphOpExport? DefaultGraphOp(string op) => GraphOpTranslators.Default(op);

    /// <summary>Who registered the graph operation translator <paramref name="op"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? GraphOpOrigin(string op) => GraphOpTranslators.Origin(op);

    /// <summary>The translator of modules of type <paramref name="type"/>: the one registered for it or for its nearest registered base type, or null.</summary>
    public static IOnnxExportOp? Find(Type type)
    {
        for (Type? t = type; t is not null && t != typeof(object); t = t.BaseType)
        {
            if (Modules.TryGet(t, out var op))
            {
                return op;
            }
        }

        return null;
    }

    /// <summary>The translator registered for the lambdas named <paramref name="name"/>, or null.</summary>
    public static IOnnxLambdaExport? FindLambda(ReadOnlySpan<char> name) => Lambdas.Find(name);

    /// <summary>The translator registered for the graph operation <paramref name="op"/>, or null.</summary>
    public static IOnnxGraphOpExport? FindGraphOp(ReadOnlySpan<char> op) => GraphOpTranslators.Find(op);

    // Translators from functions.
    private sealed class FunctionOp<T>(OnnxTranslator<T> translate) : IOnnxExportOp<T> where T : Module
    {
        public OnnxValue Export(OnnxGraph graph, T module, OnnxValue input, IReadOnlyList<int> outputShape) => translate(graph, module, input, outputShape);
    }

    private sealed class FunctionLambda(string name, OnnxTranslator<Module> translate) : IOnnxLambdaExport
    {
        public string Name => name;

        public string Summary => name;

        public OnnxValue Export(OnnxGraph graph, Module module, OnnxValue input, IReadOnlyList<int> outputShape) => translate(graph, module, input, outputShape);
    }

    private sealed class FunctionGraphOp(string name, OnnxGraphOpTranslator translate) : IOnnxGraphOpExport
    {
        public string Name => name;

        public string Summary => name;

        public OnnxValue Export(OnnxGraphOpContext context) => translate(context);
    }
}
