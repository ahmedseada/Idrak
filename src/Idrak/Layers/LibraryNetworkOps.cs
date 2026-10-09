// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// The network steps this assembly ships, one per <see cref="NetworkBuilder"/> layer method, registered in
/// <see cref="NetworkOps"/> (Idrak.Abstraction) by <see cref="LibraryRegistrations"/>, so <see cref="Network.FromJson"/>
/// and <see cref="NetworkBuilder.Op"/> replay what <see cref="NetworkBuilder.ToJson"/> writes.
/// </summary>
internal static class LibraryNetworkOps
{
    public static void RegisterAll()
    {
        NetworkOps.Register("linear", (b, a) => Of(b).Linear(a.Int("out"), a.Bool("bias")));
        NetworkOps.Register("relu", (b, _) => Of(b).ReLU());
        NetworkOps.Register("tanh", (b, _) => Of(b).Tanh());
        NetworkOps.Register("sigmoid", (b, _) => Of(b).Sigmoid());
        NetworkOps.Register("gelu", (b, _) => Of(b).GELU());
        NetworkOps.Register("softmax", (b, _) => Of(b).Softmax());
        NetworkOps.Register("dropout", (b, a) => Of(b).Dropout(a.Float("p")));
        NetworkOps.Register("batchnorm", (b, a) => Of(b).BatchNorm(a.Float("momentum"), a.Float("epsilon")));
        NetworkOps.Register("layernorm", (b, a) => Of(b).LayerNorm(a.Float("epsilon")));
        NetworkOps.Register("normalize", (b, a) => Of(b).Normalize(a.Floats("mean"), a.Floats("std")));
        NetworkOps.Register("conv2d", (b, a) => Pairs(a, "kernel", "stride", "padding") || a.Has("dilation") || a.Has("groups")
            ? Of(b).Conv2d(a.Int("out"), Pair(a, "kernel"), Pair(a, "stride"), Pair(a, "padding"), OptionalPair(a, "dilation"), a.OptionalInt("groups") ?? 1, a.Bool("bias"))
            : Of(b).Conv2d(a.Int("out"), a.Int("kernel"), a.Int("stride"), a.Int("padding"), a.Bool("bias")));
        NetworkOps.Register("maxpool2d", (b, a) => Pairs(a, "kernel", "stride", "padding") || a.Has("ceilMode") || a.Has("paddingEnd")
            ? Of(b).MaxPool2d(Pair(a, "kernel"), OptionalPair(a, "stride"), Pair(a, "padding"), CeilMode(a), OptionalPair(a, "paddingEnd"))
            : Of(b).MaxPool2d(a.Int("kernel"), a.OptionalInt("stride"), a.Int("padding")));
        NetworkOps.Register("avgpool2d", (b, a) => Of(b).AvgPool2d(Pair(a, "kernel"), OptionalPair(a, "stride"), Pair(a, "padding"), a.Bool("countIncludePad"), CeilMode(a),
            OptionalPair(a, "paddingEnd")));
        NetworkOps.Register("adaptiveavgpool2d", (b, a) => Of(b).AdaptiveAvgPool2d(Pair(a, "size")));
        NetworkOps.Register("adaptivemaxpool2d", (b, a) => Of(b).AdaptiveMaxPool2d(Pair(a, "size")));
        NetworkOps.Register("convtranspose2d", (b, a) => Of(b).ConvTranspose2d(a.Int("out"), Pair(a, "kernel"), Pair(a, "stride"), Pair(a, "padding"), OptionalPair(a, "outputPadding"),
            OptionalPair(a, "dilation"), a.OptionalInt("groups") ?? 1, a.Bool("bias")));
        NetworkOps.Register("upsample", (b, a) =>
        {
            var mode = Interpolation(a);
            bool align = a.Has("alignCorners") && a.Bool("alignCorners");
            return a.Has("size") ? Of(b).UpsampleToSize(Pair(a, "size"), mode, align) : Of(b).Upsample(FloatPair(a, "scale"), mode, align);
        });
        NetworkOps.Register("groupnorm", (b, a) => Of(b).GroupNorm(a.Int("groups"), a.Float("epsilon"), !a.Has("affine") || a.Bool("affine")));
        NetworkOps.Register("columnsToSequence", (b, _) => Of(b).ColumnsToSequence());
        NetworkOps.Register("globalavgpool2d", (b, _) => Of(b).GlobalAveragePool2d());
        NetworkOps.Register("flatten", (b, _) => Of(b).Flatten());
        NetworkOps.Register("embedding", (b, a) => Of(b).Embedding(a.Int("vocabulary"), a.Int("dim")));
        NetworkOps.Register("positional", (b, a) => Of(b).PositionalEncoding(a.OptionalInt("maxLength")));
        NetworkOps.Register("transformer", (b, a) => Of(b).TransformerEncoderLayer(a.Int("heads"), a.OptionalInt("ffDim"), a.Float("dropout"), a.Bool("causal")));
        NetworkOps.Register("attention", (b, a) => Of(b).MultiHeadAttention(a.Int("heads"), a.Bool("causal"), a.Float("dropout")));
        NetworkOps.Register("lstm", (b, a) => a.Has("bidirectional")
            ? Of(b).LSTM(a.Int("hidden"), a.Bool("returnSequences"), a.Bool("bidirectional"), a.OptionalInt("layers") ?? 1)
            : Of(b).LSTM(a.Int("hidden"), a.Bool("returnSequences")));
        NetworkOps.Register("gru", (b, a) => a.Has("bidirectional")
            ? Of(b).GRU(a.Int("hidden"), a.Bool("returnSequences"), a.Bool("bidirectional"), a.OptionalInt("layers") ?? 1, a.Has("candidateBias") && a.Bool("candidateBias"))
            : Of(b).GRU(a.Int("hidden"), a.Bool("returnSequences")));
        NetworkOps.Register("meanOverTime", (b, _) => Of(b).MeanOverTime());
        NetworkOps.Register("lastStep", (b, _) => Of(b).LastStep());
        NetworkOps.Register("firstStep", (b, _) => Of(b).FirstStep());
        NetworkOps.Register("reshape", (b, a) => Of(b).Reshape(a.Ints("shape")));
    }

    // A (height, width) pair: one number for both, or [height, width].
    private static (int Height, int Width) Pair(NetworkOpArguments a, string key) =>
        a.Json[key] is JsonArray pair ? ((int)pair[0]!, (int)pair[1]!) : (a.Int(key), a.Int(key));

    private static (int Height, int Width)? OptionalPair(NetworkOpArguments a, string key) => a.Has(key) ? Pair(a, key) : null;

    // A (height, width) pair of numbers: one for both, or [height, width].
    private static (float Height, float Width) FloatPair(NetworkOpArguments a, string key) =>
        a.Json[key] is JsonArray pair ? ((float)pair[0]!, (float)pair[1]!) : (a.Float(key), a.Float(key));

    private static bool CeilMode(NetworkOpArguments a) => a.Has("ceilMode") && a.Bool("ceilMode");

    // "nearest" (the default) or "bilinear".
    private static InterpolationMode Interpolation(NetworkOpArguments a) => (string?)a.Json["mode"] switch
    {
        null or "nearest" => InterpolationMode.Nearest,
        "bilinear" => InterpolationMode.Bilinear,
        var other => throw new InvalidDataException($"Upsampling is \"nearest\" or \"bilinear\", not \"{other}\"."),
    };

    // Whether any of the keys holds a [height, width] pair (a step the pair overloads wrote).
    private static bool Pairs(NetworkOpArguments a, params string[] keys) => keys.Any(k => a.Json[k] is JsonArray);

    // The built-in steps are the builder's own layer methods, so they need Idrak's builder, the one implementation.
    private static NetworkBuilder Of(INetworkBuilder builder) => builder as NetworkBuilder
        ?? throw new NotSupportedException($"The built-in network steps run on Idrak's NetworkBuilder, not on {builder.GetType().FullName}.");
}
