// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Audio.Plugins.Whisper;

/// <summary>
/// Builds a Whisper-style text decoder from the checkpoint's <c>model.decoder.*</c> tensors with the library's decoder
/// layers: token embedding, learned positions, decoder blocks (layer norms, causal self-attention, cross-attention over
/// the encoder's output, a feed-forward with the configured activation), a final layer norm and the head tied to the
/// embeddings (or <c>proj_out</c> when stored apart). The result decodes through the key/value cache like any decoder.
/// </summary>
public static class WhisperDecoder
{
    /// <summary>Reads the decoder as a <see cref="Sequential"/> (embedding, positions, blocks, norm, head).</summary>
    public static Sequential Read(ITensorStore store, WhisperConfig config, EncoderWeights weights, Device device, string prefix = "model.decoder.") =>
        Read(store, config, weights, device, prefix, null);

    // The same, each tensor's layer recorded in `layout` (for writing the model back under the checkpoint's names).
    internal static Sequential Read(ITensorStore store, WhisperConfig config, EncoderWeights weights, Device device, string prefix, WhisperLayout? layout)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(config);
        int d = config.Dim, heads = config.DecoderHeads, dh = config.DecoderHeadDim;
        var model = new Sequential();
        try
        {
            var embedding = Embedding.FromWeights(WhisperWeights.Table(store, prefix + "embed_tokens.weight", config.Vocabulary, d, device));
            embedding.Name = "embed";
            layout?.Add(prefix + "embed_tokens.weight", () => embedding.Weight, [config.Vocabulary, d]);
            model.Add(embedding);
            if (config.ScaleEmbedding)
            {
                model.Add(new Scale(MathF.Sqrt(d)));
            }

            var positions = new PositionEmbedding(WhisperWeights.Table(store, prefix + "embed_positions.weight", config.MaxTargetPositions, d, device)) { Name = "pos" };
            layout?.Add(prefix + "embed_positions.weight", () => positions.Weight, [config.MaxTargetPositions, d]);
            model.Add(positions);
            for (int i = 0; i < config.DecoderLayers; i++)
            {
                string layer = $"{prefix}layers.{i}.";
                var attention = new CausalSelfAttention(
                    WhisperWeights.Projection(store, layer + "self_attn.q_proj", d, d, weights, device, layout),
                    WhisperWeights.Projection(store, layer + "self_attn.k_proj", d, d, weights, device, layout),
                    WhisperWeights.Projection(store, layer + "self_attn.v_proj", d, d, weights, device, layout),
                    WhisperWeights.Projection(store, layer + "self_attn.out_proj", d, d, weights, device, layout),
                    null, null, heads, heads, dh, rope: null, config.MaxTargetPositions);
                var cross = new CrossAttention(
                    WhisperWeights.Projection(store, layer + "encoder_attn.q_proj", d, d, weights, device, layout),
                    WhisperWeights.Projection(store, layer + "encoder_attn.k_proj", d, d, weights, device, layout),
                    WhisperWeights.Projection(store, layer + "encoder_attn.v_proj", d, d, weights, device, layout),
                    WhisperWeights.Projection(store, layer + "encoder_attn.out_proj", d, d, weights, device, layout),
                    heads);
                var feedForward = new Sequential
                {
                    WhisperWeights.Projection(store, layer + "fc1", d, config.DecoderFfDim, weights, device, layout),
                    WhisperWeights.Activation(config.Activation),
                    WhisperWeights.Projection(store, layer + "fc2", config.DecoderFfDim, d, weights, device, layout),
                };
                model.Add(new DecoderBlock(WhisperWeights.Norm(store, layer + "self_attn_layer_norm", d, config.NormEpsilon, device, layout), attention,
                    WhisperWeights.Norm(store, layer + "encoder_attn_layer_norm", d, config.NormEpsilon, device, layout), cross,
                    WhisperWeights.Norm(store, layer + "final_layer_norm", d, config.NormEpsilon, device, layout), feedForward) { Name = $"layers.{i}" });
            }

            model.Add(WhisperWeights.Norm(store, prefix + "layer_norm", d, config.NormEpsilon, device, layout));
            var head = store.Contains("proj_out.weight") ? WhisperWeights.Projection(store, "proj_out", d, config.Vocabulary, weights, device, layout) : Linear.Tied(embedding);
            head.Name = "head";
            model.Add(head);
            return model;
        }
        catch
        {
            model.Dispose();
            throw;
        }
    }
}
