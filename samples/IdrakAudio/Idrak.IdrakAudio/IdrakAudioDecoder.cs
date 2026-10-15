// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.IdrakAudio;

/// <summary>
/// The text decoder of an IdrakAudio "seq2seq" model, built from the library's decoder layers so it decodes through the
/// key/value cache and <see cref="Idrak.Generation.BeamSearch"/> like any decoder: the token embedding
/// ("decoder.embed.weight"), learned positions ("decoder.positions.weight"), pre-norm blocks (layer norm, causal
/// self-attention "self_attn.{q,k,v,out}_proj"; layer norm, cross-attention over the encoder's output
/// "cross_attn.{q,k,v,out}_proj"; layer norm, feed-forward "linear1", the activation, "linear2"; norm1 to norm3 in that
/// order), a final layer norm ("decoder.norm") and the head, tied to the embedding or "decoder.head.weight".
/// </summary>
internal static class IdrakAudioDecoder
{
    /// <summary>The decoder (embedding, positions, blocks, norm, head) and its head; the weights in the order its parameters list them.</summary>
    public static (Sequential Decoder, Linear Head) Build(IdrakAudioConfig config, IdrakAudioWeights weights)
    {
        int d = config.DecoderDim, heads = config.DecoderHeads, dh = d / heads, memory = config.Dim;
        var decoder = new Sequential();
        try
        {
            var embedding = Embedding.FromWeights(weights.Table("decoder.embed.weight", config.Classes, d));
            embedding.Name = "embed";
            decoder.Add(embedding);
            decoder.Add(new PositionEmbedding(weights.Table("decoder.positions.weight", config.MaxTargetPositions, d)) { Name = "positions" });
            for (int i = 0; i < config.DecoderLayers; i++)
            {
                string layer = $"decoder.layers.{i}.";
                var attentionNorm = weights.Norm(layer + "norm1", d, config.NormEpsilon);
                var attention = new CausalSelfAttention(
                    weights.Linear(layer + "self_attn.q_proj", d, d), weights.Linear(layer + "self_attn.k_proj", d, d),
                    weights.Linear(layer + "self_attn.v_proj", d, d), weights.Linear(layer + "self_attn.out_proj", d, d),
                    null, null, heads, heads, dh, rope: null, config.MaxTargetPositions);
                var crossNorm = weights.Norm(layer + "norm2", d, config.NormEpsilon);
                var cross = new CrossAttention(
                    weights.Linear(layer + "cross_attn.q_proj", d, d), weights.Linear(layer + "cross_attn.k_proj", memory, d),
                    weights.Linear(layer + "cross_attn.v_proj", memory, d), weights.Linear(layer + "cross_attn.out_proj", d, d), heads);
                var feedForwardNorm = weights.Norm(layer + "norm3", d, config.NormEpsilon);
                var feedForward = new Sequential
                {
                    weights.Linear(layer + "linear1", d, config.DecoderFfnDim),
                    config.Activation == "gelu" ? new ExactGELU() : new GELU(),
                    weights.Linear(layer + "linear2", config.DecoderFfnDim, d),
                };
                decoder.Add(new DecoderBlock(attentionNorm, attention, crossNorm, cross, feedForwardNorm, feedForward,
                    config.Dropout > 0 ? new Dropout(config.Dropout) : null) { Name = $"layers.{i}" });
            }

            var norm = weights.Norm("decoder.norm", d, config.NormEpsilon);
            norm.Name = "norm";
            decoder.Add(norm);
            var head = config.TieEmbeddings ? Linear.Tied(embedding) : weights.Linear("decoder.head", d, config.Classes, bias: false);
            head.Name = "head";
            decoder.Add(head);
            return (decoder, head);
        }
        catch
        {
            decoder.Dispose();
            throw;
        }
    }
}
