// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.Cli.Shared;

/// <summary>One step of a network description: what it is, its output shape, parameters, FLOPs and activations per sample.</summary>
internal sealed record LayerInfo(int Index, string Op, string Description, int[] Output, long Parameters, long Flops, long Activations);

/// <summary>
/// What a network description (the builder's JSON, <c>idrak-network/1</c>) costs, for <c>idrak explain</c>,
/// <c>idrak viz</c> and the size rules of <c>idrak suggest</c>. Each step is replayed through the builder
/// (<see cref="NetworkBuilder.Op"/>), so output shapes are the builder's own; parameters are counted from the layer
/// sizes exactly as the layers create them (a step registered by a plug-in is built on the CPU and counted instead).
/// FLOPs count a multiply-add as two and an element-wise operation as one per value, forward only, per sample.
/// </summary>
internal sealed class NetworkAnalysis
{
    private NetworkAnalysis(JsonObject description, InputKind kind, int[] input)
    {
        Description = description;
        Kind = kind;
        Input = input;
    }

    public JsonObject Description { get; }

    public InputKind Kind { get; }

    public int[] Input { get; }

    public List<LayerInfo> Layers { get; } = [];

    /// <summary>Steps the analysis does not know (registered by a plug-in): their FLOPs are not counted.</summary>
    public List<string> Unknown { get; } = [];

    public long Parameters => Layers.Sum(l => l.Parameters);

    /// <summary>Forward FLOPs per sample; a training step costs about three times as much (forward and backward).</summary>
    public long Flops => Layers.Sum(l => l.Flops);

    public int[] Output => Layers.Count == 0 ? Input : Layers[^1].Output;

    /// <summary>
    /// Memory to run the network on a batch: the float32 weights plus the two largest consecutive activations
    /// (a layer's input and output live together; earlier ones are freed).
    /// </summary>
    public long InferenceBytes(int batch)
    {
        long largest = Elements(Input);
        long previous = largest;
        foreach (var layer in Layers)
        {
            largest = Math.Max(largest, previous + layer.Activations);
            previous = layer.Activations;
        }

        return Parameters * 4 + batch * largest * 4;
    }

    /// <summary>
    /// Memory to train on a batch with AdamW: the weights, their gradients and the two moments (4 × 4 bytes per
    /// parameter), plus every layer's output kept for the backward pass and as much again for the gradients flowing
    /// back (attention keeps its score matrices too, counted in the layer's activations).
    /// </summary>
    public long TrainingBytes(int batch)
    {
        long activations = Elements(Input) + Layers.Sum(l => l.Activations);
        return Parameters * 16 + batch * activations * 4 * 2;
    }

    /// <summary>Reads a network description (the JSON <see cref="NetworkBuilder.ToJson"/> writes).</summary>
    public static NetworkAnalysis Of(JsonNode description)
    {
        var json = description as JsonObject ?? throw new InvalidDataException("A network description is a JSON object.");
        var empty = (JsonObject)json.DeepClone();
        empty["steps"] = new JsonArray();
        var builder = Network.FromJson(empty);
        var analysis = new NetworkAnalysis(json, builder.InputKind, [.. builder.InputShape]);
        Sequential? built = null;
        try
        {
            int index = 0;
            foreach (var node in json["steps"]?.AsArray() ?? [])
            {
                var step = node as JsonObject ?? throw new InvalidDataException("Each network step is a JSON object.");
                string op = (string?)step["op"] ?? throw new InvalidDataException("A network step has no \"op\".");
                if (!NetworkOps.Names.Contains(op))
                {
                    throw new InvalidDataException($"Unknown network step '{op}' (registered: {string.Join(", ", NetworkOps.Names)}); load the plug-in that adds it with --plugin.");
                }

                int[] before = [.. builder.CurrentShape];
                int modules = builder.Count;
                builder.Op(op, step);
                int[] after = [.. builder.CurrentShape];
                long? parameters = CountParameters(op, step, before, after);
                if (parameters is null)
                {
                    // A step this analysis does not know: build the network once (on the CPU) and count its layers.
                    built ??= Network.FromJson(json).OnDevice(Device.Cpu).Build();
                    parameters = Enumerable.Range(modules, builder.Count - modules).Sum(i => i < built.Count ? built[i].ParameterCount : 0);
                    analysis.Unknown.Add(op);
                }

                analysis.Layers.Add(new LayerInfo(++index, op, Describe(op, step), after, parameters.Value, StepFlops(op, step, before, after),
                    Elements(after) + AttentionScores(op, step, before)));
            }
        }
        finally
        {
            built?.Dispose();
        }

        return analysis;
    }

    /// <summary>A shape as text, "[16, 28, 28]".</summary>
    public static string Shape(IReadOnlyList<int> shape) => "[" + string.Join(", ", shape) + "]";

    /// <summary>A step in words: "linear 64", "conv2d 16, 3×3, padding 1", "dropout 0.1".</summary>
    public static string Describe(string op, JsonObject step)
    {
        int I(string key, int fallback = 0) => (int?)step[key] ?? fallback;
        float F(string key, float fallback = 0) => (float?)step[key] ?? fallback;
        return op switch
        {
            "linear" => $"linear {I("out")}" + ((bool?)step["bias"] == false ? ", no bias" : ""),
            "conv2d" => $"conv2d {I("out")}, {I("kernel")}x{I("kernel")}" + (I("stride", 1) != 1 ? $", stride {I("stride")}" : "") + (I("padding") != 0 ? $", padding {I("padding")}" : ""),
            "maxpool2d" => $"max pool {I("kernel")}x{I("kernel")}",
            "globalavgpool2d" => "global average pool",
            "batchnorm" => "batch norm",
            "layernorm" => "layer norm",
            "dropout" => $"dropout {F("p", 0.5f):0.##}",
            "embedding" => $"embedding {I("vocabulary")} x {I("dim")}",
            "positional" => "positional encoding",
            "transformer" => $"transformer, {I("heads")} heads" + (step["ffDim"] is not null ? $", feed-forward {I("ffDim")}" : "") + ((bool?)step["causal"] == true ? ", causal" : ""),
            "attention" => $"attention, {I("heads")} heads" + ((bool?)step["causal"] == true ? ", causal" : ""),
            "lstm" or "gru" => $"{op} {I("hidden")}" + ((bool?)step["returnSequences"] == true ? ", every step" : ""),
            "meanOverTime" => "mean over time",
            "lastStep" => "last step",
            "firstStep" => "first step",
            "relu" => "ReLU",
            "gelu" => "GELU",
            "tanh" => "tanh",
            "sigmoid" => "sigmoid",
            "softmax" => "softmax",
            _ => op,
        };
    }

    private static long Elements(IReadOnlyList<int> shape) => shape.Aggregate(1L, (a, b) => a * b);

    // Parameters as the layers create them (see Linear, Conv2d, BatchNorm, LayerNorm, Embedding, MultiHeadAttention,
    // TransformerEncoderLayer, LSTM and GRU); null for a step this analysis does not know.
    private static long? CountParameters(string op, JsonObject step, int[] input, int[] output)
    {
        long d = input.Length > 0 ? input[^1] : 0;
        switch (op)
        {
            case "linear":
                long outFeatures = (int)step["out"]!;
                return d * outFeatures + ((bool?)step["bias"] ?? true ? outFeatures : 0);
            case "conv2d":
                long outChannels = (int)step["out"]!, kernel = (int)step["kernel"]!;
                return input[0] * outChannels * kernel * kernel + ((bool?)step["bias"] ?? true ? outChannels : 0);
            case "batchnorm":
                return 2L * input[0];                                           // gamma and beta (running statistics are buffers)
            case "layernorm":
                return 2 * d;
            case "embedding":
                return (long)(int)step["vocabulary"]! * (int)step["dim"]!;
            case "attention":
                return Attention(d);
            case "transformer":
                long ff = (int?)step["ffDim"] ?? 4 * (int)d;
                return 2 * (2 * d) + Attention(d) + (d * ff + ff) + (ff * d + d);
            case "lstm" or "gru":
                long gates = op == "lstm" ? 4 : 3, hidden = (int)step["hidden"]!;
                return d * gates * hidden + hidden * gates * hidden + gates * hidden;
            case "relu" or "tanh" or "sigmoid" or "gelu" or "softmax" or "dropout" or "maxpool2d" or "globalavgpool2d" or "flatten"
                or "positional" or "meanOverTime" or "lastStep" or "firstStep" or "reshape":
                return 0;
            default:
                return null;
        }

        static long Attention(long d) => d * 3 * d + 3 * d + d * d + d;     // the fused q/k/v projection and the output projection
    }

    private static long StepFlops(string op, JsonObject step, int[] input, int[] output)
    {
        long inElements = Elements(input), outElements = Elements(output);
        long d = input.Length > 0 ? input[^1] : 0;
        long positions = input.Length > 1 ? inElements / Math.Max(1, d) : 1;
        long t = input.Length == 2 ? input[0] : 1;
        switch (op)
        {
            case "linear":
                return 2 * positions * d * (int)step["out"]!;
            case "conv2d":
                long kernel = (int)step["kernel"]!;
                return 2 * input[0] * kernel * kernel * outElements;
            case "maxpool2d":
                long k = (int)step["kernel"]!;
                return outElements * k * k;
            case "batchnorm" or "layernorm":
                return 4 * inElements;
            case "embedding":
                return 0;
            case "attention":
                return AttentionFlops(t, d);
            case "transformer":
                long ff = (int?)step["ffDim"] ?? 4 * (int)d;
                return AttentionFlops(t, d) + 4 * t * d * ff + 8 * inElements + 2 * inElements;
            case "lstm" or "gru":
                long gates = op == "lstm" ? 4 : 3, hidden = (int)step["hidden"]!;
                return t * (2 * (d + hidden) * gates * hidden + 10 * hidden);
            case "globalavgpool2d" or "meanOverTime":
                return inElements;
            case "flatten" or "reshape" or "lastStep" or "firstStep" or "dropout":
                return 0;
            case "relu" or "tanh" or "sigmoid" or "gelu" or "softmax" or "positional":
                return outElements;
            default:
                return 0;
        }

        // q/k/v and output projections (8·T·d²) plus the scores and the weighted sum (4·T²·d).
        static long AttentionFlops(long t, long d) => 8 * t * d * d + 4 * t * t * d;
    }

    // Attention keeps a T×T score matrix per head for the backward pass.
    private static long AttentionScores(string op, JsonObject step, int[] input) =>
        op is "attention" or "transformer" && input.Length == 2 ? (long)input[0] * input[0] * ((int?)step["heads"] ?? 1) : 0;
}
