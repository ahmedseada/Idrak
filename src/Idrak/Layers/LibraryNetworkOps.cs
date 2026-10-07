// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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
        NetworkOps.Register("conv2d", (b, a) => Of(b).Conv2d(a.Int("out"), a.Int("kernel"), a.Int("stride"), a.Int("padding"), a.Bool("bias")));
        NetworkOps.Register("maxpool2d", (b, a) => Of(b).MaxPool2d(a.Int("kernel"), a.OptionalInt("stride"), a.Int("padding")));
        NetworkOps.Register("globalavgpool2d", (b, _) => Of(b).GlobalAveragePool2d());
        NetworkOps.Register("flatten", (b, _) => Of(b).Flatten());
        NetworkOps.Register("embedding", (b, a) => Of(b).Embedding(a.Int("vocabulary"), a.Int("dim")));
        NetworkOps.Register("positional", (b, a) => Of(b).PositionalEncoding(a.OptionalInt("maxLength")));
        NetworkOps.Register("transformer", (b, a) => Of(b).TransformerEncoderLayer(a.Int("heads"), a.OptionalInt("ffDim"), a.Float("dropout"), a.Bool("causal")));
        NetworkOps.Register("attention", (b, a) => Of(b).MultiHeadAttention(a.Int("heads"), a.Bool("causal"), a.Float("dropout")));
        NetworkOps.Register("lstm", (b, a) => Of(b).LSTM(a.Int("hidden"), a.Bool("returnSequences")));
        NetworkOps.Register("gru", (b, a) => Of(b).GRU(a.Int("hidden"), a.Bool("returnSequences")));
        NetworkOps.Register("meanOverTime", (b, _) => Of(b).MeanOverTime());
        NetworkOps.Register("lastStep", (b, _) => Of(b).LastStep());
        NetworkOps.Register("firstStep", (b, _) => Of(b).FirstStep());
        NetworkOps.Register("reshape", (b, a) => Of(b).Reshape(a.Ints("shape")));
    }

    // The built-in steps are the builder's own layer methods, so they need Idrak's builder, the one implementation.
    private static NetworkBuilder Of(INetworkBuilder builder) => builder as NetworkBuilder
        ?? throw new NotSupportedException($"The built-in network steps run on Idrak's NetworkBuilder, not on {builder.GetType().FullName}.");
}
