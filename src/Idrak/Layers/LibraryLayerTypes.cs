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
        LayerTypes.Register<Conv2d>("conv2d", c =>
        {
            // A square layer is described as before (one number each); any other with (height, width) pairs, dilation and groups.
            var d = new JsonObject
            {
                ["in"] = c.InChannels, ["out"] = c.OutChannels, ["kernel"] = Pair(c.KernelHeight, c.KernelWidth), ["stride"] = Pair(c.StrideHeight, c.StrideWidth),
                ["padding"] = Pair(c.PaddingHeight, c.PaddingWidth), ["bias"] = c.Bias is not null,
            };
            if (c.DilationHeight != 1 || c.DilationWidth != 1)
            {
                d["dilation"] = Pair(c.DilationHeight, c.DilationWidth);
            }

            if (c.Groups != 1)
            {
                d["groups"] = c.Groups;
            }

            return d;
        }, (d, device) => new Conv2d(I(d, "in"), I(d, "out"), P(d, "kernel"), P(d, "stride"), P(d, "padding"), d["dilation"] is null ? null : P(d, "dilation"),
            d["groups"] is null ? 1 : I(d, "groups"), B(d, "bias"), device));
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
        LayerTypes.Register<LSTM>("lstm", r => Recurrent(r, false),
            (d, device) => new LSTM(I(d, "in"), I(d, "hidden"), B(d, "sequences"), d["bidirectional"] is not null && B(d, "bidirectional"), d["layers"] is null ? 1 : I(d, "layers"), device));
        LayerTypes.Register<GRU>("gru", r => Recurrent(r, r.CandidateBias),
            (d, device) => new GRU(I(d, "in"), I(d, "hidden"), B(d, "sequences"), d["bidirectional"] is not null && B(d, "bidirectional"), d["layers"] is null ? 1 : I(d, "layers"),
                d["candidateBias"] is not null && B(d, "candidateBias"), device));
        LayerTypes.Register<MaxPool2d>("maxpool2d", m => Pooling(new()
        {
            ["kernel"] = Pair(m.KernelHeight, m.KernelWidth), ["stride"] = Pair(m.StrideHeight, m.StrideWidth), ["padding"] = Pair(m.PaddingHeight, m.PaddingWidth),
        }, m.PaddingHeight, m.PaddingWidth, m.PaddingBottom, m.PaddingRight, m.CeilMode),
            (d, _) => new MaxPool2d(P(d, "kernel"), P(d, "stride"), P(d, "padding"), d["ceilMode"] is not null && B(d, "ceilMode"), d["paddingEnd"] is null ? null : P(d, "paddingEnd")));
        LayerTypes.Register<AvgPool2d>("avgpool2d", m => Pooling(new()
        {
            ["kernel"] = Pair(m.KernelHeight, m.KernelWidth), ["stride"] = Pair(m.StrideHeight, m.StrideWidth), ["padding"] = Pair(m.PaddingHeight, m.PaddingWidth),
            ["countIncludePad"] = m.CountIncludePad,
        }, m.PaddingHeight, m.PaddingWidth, m.PaddingBottom, m.PaddingRight, m.CeilMode),
            (d, _) => new AvgPool2d(P(d, "kernel"), P(d, "stride"), P(d, "padding"), B(d, "countIncludePad"), d["ceilMode"] is not null && B(d, "ceilMode"),
                d["paddingEnd"] is null ? null : P(d, "paddingEnd")));
        LayerTypes.Register<AdaptiveAvgPool2d>("adaptiveavgpool2d", m => new() { ["size"] = Pair(m.OutputHeight, m.OutputWidth) },
            (d, _) => new AdaptiveAvgPool2d(P(d, "size")));
        LayerTypes.Register<AdaptiveMaxPool2d>("adaptivemaxpool2d", m => new() { ["size"] = Pair(m.OutputHeight, m.OutputWidth) },
            (d, _) => new AdaptiveMaxPool2d(P(d, "size")));
        LayerTypes.Register<ConvTranspose2d>("convtranspose2d", c => new()
        {
            ["in"] = c.InChannels, ["out"] = c.OutChannels, ["kernel"] = Pair(c.KernelHeight, c.KernelWidth), ["stride"] = Pair(c.StrideHeight, c.StrideWidth),
            ["padding"] = Pair(c.PaddingHeight, c.PaddingWidth), ["outputPadding"] = Pair(c.OutputPaddingHeight, c.OutputPaddingWidth),
            ["dilation"] = Pair(c.DilationHeight, c.DilationWidth), ["groups"] = c.Groups, ["bias"] = c.Bias is not null,
        }, (d, device) => new ConvTranspose2d(I(d, "in"), I(d, "out"), P(d, "kernel"), P(d, "stride"), P(d, "padding"), P(d, "outputPadding"), P(d, "dilation"), I(d, "groups"),
            B(d, "bias"), device));
        LayerTypes.Register<Upsample>("upsample", u =>
        {
            var d = u.Size is { } size ? new JsonObject { ["size"] = Pair(size.Height, size.Width) }
                : new JsonObject { ["scale"] = new JsonArray(u.ScaleFactor!.Value.Height, u.ScaleFactor.Value.Width) };
            d["mode"] = u.Mode == InterpolationMode.Nearest ? "nearest" : "bilinear";
            d["alignCorners"] = u.AlignCorners;
            return d;
        }, (d, _) =>
        {
            var mode = (string)d["mode"]! == "bilinear" ? InterpolationMode.Bilinear : InterpolationMode.Nearest;
            return d["size"] is not null ? Upsample.ToSize(P(d, "size"), mode, B(d, "alignCorners"))
                : new Upsample(((float)d["scale"]![0]!, (float)d["scale"]![1]!), mode, B(d, "alignCorners"));
        });
        LayerTypes.Register<GroupNorm>("groupnorm", g => new() { ["groups"] = g.Groups, ["channels"] = g.Channels, ["epsilon"] = g.Epsilon, ["affine"] = g.Affine },
            (d, device) => new GroupNorm(I(d, "groups"), I(d, "channels"), F(d, "epsilon"), B(d, "affine"), device));
        LayerTypes.Register<GlobalAveragePool2d>("globalavgpool2d", _ => [], (_, _) => new GlobalAveragePool2d());
        LayerTypes.Register<Flatten>("flatten", _ => [], (_, _) => new Flatten());
        LayerTypes.Register<ReLU>("relu", _ => [], (_, _) => new ReLU());
        LayerTypes.Register<Tanh>("tanh", _ => [], (_, _) => new Tanh());
        LayerTypes.Register<Sigmoid>("sigmoid", _ => [], (_, _) => new Sigmoid());
        LayerTypes.Register<GELU>("gelu", _ => [], (_, _) => new GELU());
        LayerTypes.Register<ExactGELU>("gelu_exact", _ => [], (_, _) => new ExactGELU());
        LayerTypes.Register<QuickGELU>("quick_gelu", _ => [], (_, _) => new QuickGELU());
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

    // A (height, width) pair: one number when both are equal (as square layers were always described), else [height, width].
    private static JsonNode Pair(int height, int width) => height == width ? JsonValue.Create(height) : new JsonArray(height, width);

    // A pooling layer's ceil mode and padding below and right, described only when they are not the defaults (so every
    // other pooling layer is described as before).
    private static JsonObject Pooling(JsonObject d, int top, int left, int bottom, int right, bool ceilMode)
    {
        if (bottom != top || right != left)
        {
            d["paddingEnd"] = Pair(bottom, right);
        }

        if (ceilMode)
        {
            d["ceilMode"] = true;
        }

        return d;
    }

    private static (int Height, int Width) P(JsonObject d, string key) => d[key] is JsonArray pair ? ((int)pair[0]!, (int)pair[1]!) : (I(d, key), I(d, key));

    // One direction and one layer are described as before; more with "bidirectional" and "layers".
    private static JsonObject Recurrent(RecurrentModule r, bool candidateBias)
    {
        var d = new JsonObject { ["in"] = r.InputSize, ["hidden"] = r.HiddenSize, ["sequences"] = r.ReturnSequences };
        if (r.Bidirectional || r.Layers > 1 || candidateBias)
        {
            d["bidirectional"] = r.Bidirectional;
            d["layers"] = r.Layers;
        }

        if (candidateBias)
        {
            d["candidateBias"] = true;
        }

        return d;
    }

    private static float F(JsonObject d, string key) => (float)d[key]!;

    private static bool B(JsonObject d, string key) => (bool)d[key]!;
}
