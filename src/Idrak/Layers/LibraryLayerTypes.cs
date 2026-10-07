// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// The layer types this assembly ships, registered in <see cref="LayerTypes"/> (Idrak.Abstraction) by
/// <see cref="LibraryRegistrations"/>, so modules built from them can be described as JSON and created again.
/// </summary>
internal static class LibraryLayerTypes
{
    public static void RegisterAll()
    {
        LayerTypes.Register<Linear>("linear", l => new() { ["in"] = l.InFeatures, ["out"] = l.OutFeatures, ["bias"] = l.Bias is not null },
            (d, device) => new Linear(I(d, "in"), I(d, "out"), B(d, "bias"), device));
        LayerTypes.Register<Conv2d>("conv2d", c => new()
        {
            ["in"] = c.InChannels, ["out"] = c.OutChannels, ["kernel"] = c.KernelSize, ["stride"] = c.Stride, ["padding"] = c.Padding, ["bias"] = c.Bias is not null,
        }, (d, device) => new Conv2d(I(d, "in"), I(d, "out"), I(d, "kernel"), I(d, "stride"), I(d, "padding"), B(d, "bias"), device));
        LayerTypes.Register<BatchNorm>("batchnorm", b => new() { ["channels"] = b.Channels, ["momentum"] = b.Momentum, ["epsilon"] = b.Epsilon },
            (d, device) => new BatchNorm(I(d, "channels"), F(d, "momentum"), F(d, "epsilon"), device));
        LayerTypes.Register<LayerNorm>("layernorm", n => new() { ["features"] = n.Features, ["epsilon"] = n.Epsilon },
            (d, device) => new LayerNorm(I(d, "features"), F(d, "epsilon"), device));
        LayerTypes.Register<Embedding>("embedding", e => new() { ["vocabulary"] = e.Vocabulary, ["dim"] = e.Dim },
            (d, device) => new Embedding(I(d, "vocabulary"), I(d, "dim"), device));
        LayerTypes.Register<PositionalEncoding>("positional", p => new() { ["maxLength"] = p.MaxLength, ["dim"] = p.Dim },
            (d, device) => new PositionalEncoding(I(d, "maxLength"), I(d, "dim"), device));
        LayerTypes.Register<MultiHeadAttention>("attention", a => new() { ["dim"] = a.Dim, ["heads"] = a.Heads, ["causal"] = a.Causal },
            (d, device) => new MultiHeadAttention(I(d, "dim"), I(d, "heads"), B(d, "causal"), 0f, device));
        LayerTypes.Register<TransformerEncoderLayer>("transformer", t => t.Children().ToList() is [_, MultiHeadAttention a, _, Linear ff, ..]
            ? new() { ["dim"] = t.Dim, ["heads"] = a.Heads, ["ffDim"] = ff.OutFeatures, ["causal"] = a.Causal }
            : throw new NotSupportedException("a transformer layer without its standard sublayers cannot be described"),
            (d, device) => new TransformerEncoderLayer(I(d, "dim"), I(d, "heads"), I(d, "ffDim"), 0f, B(d, "causal"), device));
        LayerTypes.Register<LSTM>("lstm", r => new() { ["in"] = r.InputSize, ["hidden"] = r.HiddenSize, ["sequences"] = r.ReturnSequences },
            (d, device) => new LSTM(I(d, "in"), I(d, "hidden"), B(d, "sequences"), device));
        LayerTypes.Register<GRU>("gru", r => new() { ["in"] = r.InputSize, ["hidden"] = r.HiddenSize, ["sequences"] = r.ReturnSequences },
            (d, device) => new GRU(I(d, "in"), I(d, "hidden"), B(d, "sequences"), device));
        LayerTypes.Register<MaxPool2d>("maxpool2d", m => new() { ["kernel"] = m.KernelSize, ["stride"] = m.Stride, ["padding"] = m.Padding },
            (d, _) => new MaxPool2d(I(d, "kernel"), I(d, "stride"), I(d, "padding")));
        LayerTypes.Register<GlobalAveragePool2d>("globalavgpool2d", _ => [], (_, _) => new GlobalAveragePool2d());
        LayerTypes.Register<Flatten>("flatten", _ => [], (_, _) => new Flatten());
        LayerTypes.Register<ReLU>("relu", _ => [], (_, _) => new ReLU());
        LayerTypes.Register<Tanh>("tanh", _ => [], (_, _) => new Tanh());
        LayerTypes.Register<Sigmoid>("sigmoid", _ => [], (_, _) => new Sigmoid());
        LayerTypes.Register<GELU>("gelu", _ => [], (_, _) => new GELU());
        LayerTypes.Register<Softmax>("softmax", _ => [], (_, _) => new Softmax());
        LayerTypes.Register<Dropout>("dropout", d => new() { ["p"] = d.Probability }, (d, _) => new Dropout(F(d, "p")));

        // A block: the network builder's description when it built the block (so registered builder steps, lambdas
        // included, survive), otherwise each layer's own description.
        LayerTypes.Register<Sequential>("sequential", s => Network.ArchitectureOf(s) is { } network
            ? new() { ["network"] = network.DeepClone() }
            : new() { ["layers"] = new JsonArray([.. s.Select(m => (JsonNode)LayerTypes.Describe(m))]) },
            (d, device) => d["network"] is JsonObject network
                ? Network.FromJson(network).OnDevice(device).Build()
                : new Sequential([.. d["layers"]!.AsArray().Select(l => LayerTypes.Create(l!.AsObject(), device))]));
    }

    private static int I(JsonObject d, string key) => (int)d[key]!;

    private static float F(JsonObject d, string key) => (float)d[key]!;

    private static bool B(JsonObject d, string key) => (bool)d[key]!;
}
