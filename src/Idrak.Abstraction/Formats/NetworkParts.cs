// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Formats;

/// <summary>
/// The parts of a pretrained network as training speaks of them: which part each module belongs to (the
/// <see cref="Embeddings"/>, the <see cref="Decoder"/> or the output <see cref="Head"/>, which a pretraining stage trains
/// or freezes by name), and the projections adapters go on by default. An architecture whose network names its modules
/// otherwise than the library's decoders implements it beside the model's architecture (core's <c>IPretrainedArchitecture</c>); any other uses
/// <see cref="NetworkParts.Decoder"/> (<c>PretrainedModel.NetworkParts</c>).
/// </summary>
public interface INetworkParts
{
    /// <summary>The token and position embeddings: "embeddings".</summary>
    const string Embeddings = "embeddings";

    /// <summary>The blocks and the final norm: "decoder".</summary>
    const string Decoder = "decoder";

    /// <summary>The output head: "head".</summary>
    const string Head = "head";

    /// <summary>The part (<see cref="Embeddings"/>, <see cref="Decoder"/> or <see cref="Head"/>) of the module at <paramref name="path"/> (<c>PretrainedModel.NamedModules</c>).</summary>
    string PartOf(ReadOnlySpan<char> path);

    /// <summary>The projections adapters go on when none are named (<c>FineTuningOptions.Targets</c>), by layer name.</summary>
    IReadOnlyList<string> AdapterTargets { get; }
}

/// <summary>The parts of the library's decoders (<see cref="INetworkParts"/>).</summary>
public static class NetworkParts
{
    /// <summary>
    /// The library's decoders: a module under "embed" or "pos" is the embeddings, one under "head" the head, any other the
    /// decoder; adapters go on the attention's q, k, v and o and the feed-forward's gate, up and down.
    /// </summary>
    public static INetworkParts Decoder { get; } = new DecoderParts();

    private sealed class DecoderParts : INetworkParts
    {
        public IReadOnlyList<string> AdapterTargets { get; } = ["q", "k", "v", "o", "gate", "up", "down"];

        public string PartOf(ReadOnlySpan<char> path)
        {
            int dot = path.IndexOf('.');
            var top = dot < 0 ? path : path[..dot];
            return top.SequenceEqual("embed") || top.SequenceEqual("pos") ? INetworkParts.Embeddings
                : top.SequenceEqual("head") ? INetworkParts.Head : INetworkParts.Decoder;
        }
    }
}
