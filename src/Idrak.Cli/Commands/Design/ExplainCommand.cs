// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Design;

/// <summary>
/// <c>idrak explain network.json</c> (<c>x</c>): the layers of a network description with their output shapes,
/// parameters and FLOPs per sample, and the memory to run and to train it, against the chosen device's memory.
/// </summary>
internal sealed class ExplainCommand : Command
{
    public override string Name => "explain";

    public override IReadOnlyCollection<string> Aliases => ["x"];

    public override string Summary => "Layers, shapes, parameters, FLOPs and memory of a network.json";

    public override string Usage =>
        "NETWORK.json [--batch N] [options]\n\n" +
        "NETWORK.json is the network builder's JSON (written by idrak suggest, NetworkBuilder.ToJson or a model package's\n" +
        "architecture), or a folder holding network.json. The batch size for memory is --batch, else a train.json next to\n" +
        "it, else 32. FLOPs count a multiply-add as two, forward only; a training step costs about three times as much.\n" +
        "Memory: float32 weights; inference keeps a layer's input and output; training with AdamW keeps weights, gradients\n" +
        "and two moments (16 bytes per parameter) and every activation twice (for the backward pass).\n\n" +
        "Options:\n" +
        "      --batch N   the batch size the memory is computed for\n\n" +
        "Examples:\n" +
        "  idrak x ./run/network.json                 parameters, FLOPs, memory\n" +
        "  idrak explain network.json -d vulkan:0 -j  against the GPU, as JSON\n" +
        "  idrak x network.json --batch 256           memory at a larger batch\n\n" +
        "Limits: a GPU's memory is known only through a configured limit (the library does not report it).\n" +
        "Environment: IDRAK_CONFIG, IDRAK_TRACE.";

    public override IReadOnlyCollection<string> ValueOptions => ["--batch"];

    public override int Run(CommandContext context)
    {
        var (path, description) = Read(context.Argument(0, "NETWORK.json"));
        var analysis = NetworkAnalysis.Of(description);
        int batch = context.Option("--batch") is not null ? context.IntOption("--batch", 32) : TrainBatch(path) ?? 32;
        if (batch < 1)
        {
            throw new UsageException("--batch needs 1 or more.");
        }

        long inferenceOne = analysis.InferenceBytes(1), inferenceBatch = analysis.InferenceBytes(batch), training = analysis.TrainingBytes(batch);
        long? memory = DeviceMemory.Total(context.Device);

        context.Write($"{(string?)description["name"] ?? Path.GetFileName(path)}: {analysis.Kind} input {NetworkAnalysis.Shape(analysis.Input)} -> {NetworkAnalysis.Shape(analysis.Output)}, {analysis.Layers.Count} layers");
        context.Write("");
        context.Table(["#", "Layer", "Output", "Parameters", "FLOPs"],
            analysis.Layers.Select(l => (IReadOnlyList<string>)[l.Index.ToString(), l.Description, NetworkAnalysis.Shape(l.Output), l.Parameters.ToString("N0"), DeviceMemory.Count(l.Flops)]));
        context.Write("");
        context.Write($"Parameters    {analysis.Parameters:N0} ({DeviceMemory.Format(analysis.Parameters * 4.0)} as float32)");
        context.Write($"FLOPs         {DeviceMemory.Count(analysis.Flops)} per sample forward, about {DeviceMemory.Count(3.0 * analysis.Flops)} per sample to train");
        context.Write($"Inference     {DeviceMemory.Format(inferenceOne)} at batch 1, {DeviceMemory.Format(inferenceBatch)} at batch {batch}");
        context.Write($"Training      {DeviceMemory.Format(training)} at batch {batch} (AdamW)");
        context.Write(memory is { } m
            ? $"Device        {context.Device} offers {DeviceMemory.Format(m)}: training {(training <= m * 0.8 ? "fits" : "does not fit (lower the batch or the network)")}, inference {(inferenceBatch <= m * 0.8 ? "fits" : "does not fit")}"
            : $"Device        {context.Device}: its memory is not reported (set a GPU memory limit to compare)");
        foreach (string op in analysis.Unknown.Distinct())
        {
            context.Write($"Note          '{op}' is a registered step this analysis does not know: its parameters are counted, its FLOPs are not");
        }

        context.WriteJson(new JsonObject
        {
            ["path"] = path,
            ["name"] = (string?)description["name"],
            ["input"] = new JsonObject { ["kind"] = analysis.Kind.ToString(), ["shape"] = Ints(analysis.Input) },
            ["output"] = Ints(analysis.Output),
            ["layers"] = new JsonArray([.. analysis.Layers.Select(l => (JsonNode)new JsonObject
            {
                ["index"] = l.Index, ["op"] = l.Op, ["description"] = l.Description, ["output"] = Ints(l.Output), ["parameters"] = l.Parameters,
                ["flops"] = l.Flops, ["activations"] = l.Activations,
            })]),
            ["parameters"] = analysis.Parameters,
            ["flopsPerSample"] = analysis.Flops,
            ["trainingFlopsPerSample"] = 3 * analysis.Flops,
            ["batch"] = batch,
            ["memory"] = new JsonObject
            {
                ["weights"] = analysis.Parameters * 4, ["inference1"] = inferenceOne, ["inferenceBatch"] = inferenceBatch, ["training"] = training,
                ["device"] = context.Device.ToString(), ["deviceMemory"] = memory,
                ["trainingFits"] = memory is { } dm ? training <= dm * 0.8 : null,
            },
            ["unknownSteps"] = new JsonArray([.. analysis.Unknown.Distinct().Select(u => (JsonNode)u)]),
        });
        return ExitCodes.Ok;
    }

    /// <summary>Reads a network description from a file or a folder holding network.json.</summary>
    internal static (string Path, JsonObject Description) Read(string argument)
    {
        string path = Path.GetFullPath(argument);
        if (Directory.Exists(path))
        {
            path = Path.Combine(path, "network.json");
        }

        if (!File.Exists(path))
        {
            throw new UsageException($"No network description at {path}: give a network.json (idrak suggest writes one).");
        }

        var node = System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(path)) as JsonObject
                   ?? throw new InvalidDataException($"{path} is not a JSON object.");
        if ((string?)node["format"] != "idrak-network/1" && node["architecture"] is JsonObject inner)
        {
            node = inner;                                                               // a package manifest holding the architecture
        }

        return (path, node);
    }

    private static int? TrainBatch(string path)
    {
        string train = Path.Combine(Path.GetDirectoryName(path)!, "train.json");
        try
        {
            return File.Exists(train) ? (int?)JsonNode.Parse(File.ReadAllText(train))?["batchSize"] : null;
        }
        catch (Exception e) when (e is System.Text.Json.JsonException or InvalidOperationException or FormatException)
        {
            return null;
        }
    }

    private static JsonArray Ints(IEnumerable<int> values) => new([.. values.Select(v => (JsonNode)v)]);
}
