// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.T5;

/// <summary>
/// How a T5 checkpoint (Hugging Face's names: <c>shared</c>, <c>encoder.block.N.layer.M</c>, <c>decoder.block.N.layer.M</c>,
/// <c>lm_head</c>) becomes the library's layers: a <see cref="SequenceEncoder"/> of pre-norm
/// <see cref="TransformerEncoderLayer"/>s with RMS norms, and a decoder of <see cref="DecoderBlock"/>s with
/// cross-attention, every attention unscaled and the "bucketed" position bias of each stack's first layer shared by the
/// rest. Shared by the encoder-decoder family and the encoder family. Public API only.
/// </summary>
internal static class T5Layers
{
    /// <summary>The sizes and settings config.json gives.</summary>
    internal sealed record Sizes(int Vocabulary, int Width, int HeadWidth, int FeedForward, int EncoderLayers, int DecoderLayers, int Heads, int Buckets,
        int MaxDistance, float Epsilon, bool Gated, FeedForwardActivation Activation, bool ScaleOutput, int StartToken, int EndToken, int MaxTargetPositions)
    {
        public static Sizes Read(JsonObject config)
        {
            ArgumentNullException.ThrowIfNull(config);
            string projection = config["feed_forward_proj"] is { } proj ? proj.GetValueKind() == JsonValueKind.String ? proj.GetValue<string>()
                : throw new InvalidDataException($"T5's feed_forward_proj is a name; got {proj.ToJsonString()}.") : "relu";
            var (gated, activation) = projection switch
            {
                "relu" => (false, FeedForwardActivation.Relu),
                "gated-gelu" or "gated-gelu_new" or "gated-gelu_pytorch_tanh" => (true, FeedForwardActivation.Gelu),
                "gated-silu" => (true, FeedForwardActivation.Silu),
                _ => throw new NotSupportedException($"T5's feed_forward_proj '{projection}' is not one the family reads (relu, gated-gelu, gated-silu)."),
            };

            int layers = Int(config, "num_layers"), heads = Int(config, "num_heads");
            int buckets = config["relative_attention_num_buckets"] is null ? 32 : Int(config, "relative_attention_num_buckets");
            int distance = config["relative_attention_max_distance"] is null ? 128 : Int(config, "relative_attention_max_distance");
            if (buckets < 4 || buckets % 2 != 0)
            {
                throw new NotSupportedException($"T5's relative_attention_num_buckets {buckets} is not one the family reads (an even number of 4 or more: half for each side in the encoder).");
            }

            if (distance <= buckets / 2)
            {
                throw new NotSupportedException($"T5's relative_attention_max_distance {distance} is not above relative_attention_num_buckets / 2 ({buckets / 2}), the distances its exact buckets cover.");
            }

            float epsilon = config["layer_norm_epsilon"] is { } e ? (float)e.GetValue<double>() : 1e-6f;
            if (!(epsilon > 0f) || !float.IsFinite(epsilon))
            {
                throw new InvalidDataException($"T5's layer_norm_epsilon is a positive number; got {epsilon.ToString(CultureInfo.InvariantCulture)}.");
            }

            // The decoder's output is scaled by width^-0.5 before a head tied to the shared embedding (T5 v1.0); T5 v1.1
            // and mT5 say tie_word_embeddings false and scale nothing. transformers 5 writes the choice itself as
            // scale_decoder_outputs when it saves a configuration.
            bool scale = config["scale_decoder_outputs"] is { } scaled ? scaled.GetValueKind() == JsonValueKind.True
                : config["tie_word_embeddings"] is not { } tie || tie.GetValueKind() != JsonValueKind.False;
            int end = config["eos_token_id"] switch
            {
                null => 1,
                JsonArray { Count: > 0 } list => (int)list[0]!,
                JsonValue value when value.GetValueKind() == JsonValueKind.Number => (int)value,
                var other => throw new InvalidDataException($"T5's eos_token_id is a token id; got {other.ToJsonString()}."),
            };
            int pad = config["pad_token_id"] is { } padNode && padNode.GetValueKind() == JsonValueKind.Number ? (int)padNode : 0;
            int start = config["decoder_start_token_id"] is { } startNode && startNode.GetValueKind() == JsonValueKind.Number ? (int)startNode : pad;
            int decoderLayers = config["num_decoder_layers"] is { } count && count.GetValueKind() == JsonValueKind.Number ? Int(config, "num_decoder_layers") : layers;
            return new Sizes(Int(config, "vocab_size"), Int(config, "d_model"), Int(config, "d_kv"), Int(config, "d_ff"), layers, decoderLayers,
                heads, buckets, distance, epsilon, gated, activation, scale, start, end,
                config["n_positions"] is { } positions && positions.GetValueKind() == JsonValueKind.Number ? (int)positions : DefaultTargetPositions);
        }

        // The longest target when config.json gives no n_positions (the length T5 was trained with; its relative
        // positions take longer ones, and the caller's NumPredict caps an answer below it).
        private const int DefaultTargetPositions = 512;

        private static int Int(JsonObject config, string key)
        {
            if (config[key] is not { } node || node.GetValueKind() != JsonValueKind.Number)
            {
                throw new InvalidDataException($"config.json has no {key} (a number), which a T5 model needs.");
            }

            int value = node.GetValue<int>();
            return value > 0 ? value : throw new InvalidDataException($"T5's {key} is positive; got {value.ToString(CultureInfo.InvariantCulture)}.");
        }
    }

    /// <summary>The shared token embedding: shared.weight, else a stack's embed_tokens.weight.</summary>
    public static Embedding Shared(ITensorStore store, Sizes sizes, Device device) =>
        Embedding.FromWeights(Tensor.Persistent(Checked(store, SharedName(store), [sizes.Vocabulary, sizes.Width], transposed: false), [sizes.Vocabulary, sizes.Width], device,
            requiresGrad: true));

    private static string SharedName(ITensorStore store) =>
        new[] { "shared.weight", "encoder.embed_tokens.weight", "decoder.embed_tokens.weight" }.FirstOrDefault(store.Contains)
        ?? throw new InvalidDataException("The T5 checkpoint has no shared.weight (nor encoder.embed_tokens.weight): not a T5 checkpoint.");

    /// <summary>
    /// The encoder: the shared embedding, pre-norm blocks of RMS norms, unscaled attention with the first block's
    /// bidirectional "bucketed" bias shared by the others, the feed-forward, and the final RMS norm. The encoder takes
    /// <paramref name="words"/> (it disposes it with itself).
    /// </summary>
    public static SequenceEncoder Encoder(ITensorStore store, Sizes sizes, Embedding words, Device device)
    {
        var built = new List<IDisposable>();
        try
        {
            var bias = Bias(store, "encoder", sizes, bidirectional: true, device);
            built.Add((IDisposable)bias);
            var blocks = new List<Module>(sizes.EncoderLayers);
            int inner = sizes.Heads * sizes.HeadWidth;
            for (int i = 0; i < sizes.EncoderLayers; i++)
            {
                string l = $"encoder.block.{i}.layer.";
                var qkv = Keep(built, Fused(store, $"{l}0.SelfAttention", sizes, device));
                var output = Keep(built, Dense(store, $"{l}0.SelfAttention.o.weight", inner, sizes.Width, device));
                var attention = new MultiHeadAttention(qkv, output, sizes.Heads) { ScoreScale = 1f, PositionBias = bias, SharesPositionBias = i > 0 };
                built.Remove(qkv);
                built.Remove(output);
                Keep(built, attention);
                var block = TransformerEncoderLayer.FromLayers(Keep(built, Norm(store, $"{l}0.layer_norm.weight", sizes, device)), attention,
                    Keep(built, Norm(store, $"{l}1.layer_norm.weight", sizes, device)), Keep(built, FeedForwardOf(store, $"{l}1.DenseReluDense", sizes, device)));
                blocks.Add(Keep(built, block));
                if (i == 0)
                {
                    built.Remove((IDisposable)bias);                       // owned by the first attention from here on
                }
            }

            var final = Keep(built, Norm(store, "encoder.final_layer_norm.weight", sizes, device));
            return new SequenceEncoder(words, blocks, positions: null, tokenTypes: null, embeddingNorm: null, finalNorm: final);
        }
        catch
        {
            foreach (var part in Enumerable.Reverse(built))
            {
                part.Dispose();
            }

            throw;
        }
    }

    /// <summary>
    /// The decoder, ids to logits: the shared embedding read in place, blocks of self-attention (unscaled, causal, the
    /// first block's one-way "bucketed" bias shared), cross-attention (unscaled) and the feed-forward, each after an RMS
    /// norm, then the final RMS norm, the width^-0.5 scale when the configuration asks for it, and the head: the shared
    /// embedding read in place (tied) unless the checkpoint holds an lm_head.weight of other values.
    /// </summary>
    public static Sequential Decoder(ITensorStore store, Sizes sizes, Embedding words, Device device, bool tie)
    {
        var built = new List<IDisposable>();
        try
        {
            var modules = new List<Module> { new SharedEmbedding(words) };
            var bias = Bias(store, "decoder", sizes, bidirectional: false, device);
            built.Add((IDisposable)bias);
            int inner = sizes.Heads * sizes.HeadWidth;
            for (int i = 0; i < sizes.DecoderLayers; i++)
            {
                string l = $"decoder.block.{i}.layer.";
                var self = new CausalSelfAttention(Keep(built, Dense(store, $"{l}0.SelfAttention.q.weight", sizes.Width, inner, device)),
                    Keep(built, Dense(store, $"{l}0.SelfAttention.k.weight", sizes.Width, inner, device)),
                    Keep(built, Dense(store, $"{l}0.SelfAttention.v.weight", sizes.Width, inner, device)),
                    Keep(built, Dense(store, $"{l}0.SelfAttention.o.weight", inner, sizes.Width, device)), null, null, sizes.Heads, sizes.Heads, sizes.HeadWidth, null,
                    sizes.MaxTargetPositions)
                {
                    ScoreScale = 1f,
                    PositionBias = bias,
                    SharesPositionBias = i > 0,
                };
                var cross = new CrossAttention(Keep(built, Dense(store, $"{l}1.EncDecAttention.q.weight", sizes.Width, inner, device)),
                    Keep(built, Dense(store, $"{l}1.EncDecAttention.k.weight", sizes.Width, inner, device)),
                    Keep(built, Dense(store, $"{l}1.EncDecAttention.v.weight", sizes.Width, inner, device)),
                    Keep(built, Dense(store, $"{l}1.EncDecAttention.o.weight", inner, sizes.Width, device)), sizes.Heads) { ScoreScale = 1f };
                var block = new DecoderBlock(Keep(built, Norm(store, $"{l}0.layer_norm.weight", sizes, device)), Keep(built, self),
                    Keep(built, Norm(store, $"{l}1.layer_norm.weight", sizes, device)), Keep(built, cross),
                    Keep(built, Norm(store, $"{l}2.layer_norm.weight", sizes, device)), Keep(built, FeedForwardOf(store, $"{l}2.DenseReluDense", sizes, device)));
                modules.Add(Keep(built, block));
                if (i == 0)
                {
                    built.Remove((IDisposable)bias);                       // owned by the first self-attention from here on
                }
            }

            modules.Add(Keep(built, Norm(store, "decoder.final_layer_norm.weight", sizes, device)));
            if (sizes.ScaleOutput)
            {
                modules.Add(new Scale(1f / MathF.Sqrt(sizes.Width)));
            }

            modules.Add(Keep(built, Head(store, sizes, words, device, tie)));
            return new Sequential(modules);
        }
        catch
        {
            foreach (var part in Enumerable.Reverse(built))
            {
                part.Dispose();
            }

            throw;
        }
    }

    // The head: lm_head.weight [vocabulary, width] when the checkpoint holds one of other values than the shared embedding
    // (or the configuration unties it), else the shared embedding read in place (one table).
    private static Linear Head(ITensorStore store, Sizes sizes, Embedding words, Device device, bool tie)
    {
        const string name = "lm_head.weight";
        if (!store.Contains(name))
        {
            return tie ? Linear.Tied(words)
                : throw new InvalidDataException($"The T5 checkpoint has no {name}, and config.json's tie_word_embeddings false gives the decoder a head of its own.");
        }

        if (tie && store.Read(name).AsSpan().SequenceEqual(store.Read(SharedName(store))))
        {
            return Linear.Tied(words);                                          // the same values: one table, read in place
        }

        return Dense(store, name, sizes.Width, sizes.Vocabulary, device);      // [vocabulary, width] stored: read transposed
    }

    // A stack's position bias: its first block's relative_attention_bias.weight [buckets, heads] as the library's "bucketed" kind.
    private static IAttentionPositionBias Bias(ITensorStore store, string stack, Sizes sizes, bool bidirectional, Device device)
    {
        string name = $"{stack}.block.0.layer.0.SelfAttention.relative_attention_bias.weight";
        var table = Checked(store, name, [sizes.Buckets, sizes.Heads], transposed: false);
        var settings = new JsonObject { ["num_buckets"] = sizes.Buckets, ["max_distance"] = sizes.MaxDistance, ["bidirectional"] = bidirectional };
        return AttentionPositionBiases.Create(BucketedPositionBias.KindName, new PositionBiasRequest(sizes.Heads, settings, device, 0,
            (part, _) => part == "table" ? table : null));
    }

    // q, k and v side by side, [width, 3 · inner] (each read transposed from Hugging Face's [inner, width]), no biases.
    private static Linear Fused(ITensorStore store, string attention, Sizes sizes, Device device)
    {
        int w = sizes.Width, inner = sizes.Heads * sizes.HeadWidth;
        var weight = new float[w * 3 * inner];
        string[] parts = ["q", "k", "v"];
        for (int part = 0; part < 3; part++)
        {
            var values = Checked(store, $"{attention}.{parts[part]}.weight", [w, inner], transposed: true);     // [in, out]
            for (int row = 0; row < w; row++)
            {
                values.AsSpan(row * inner, inner).CopyTo(weight.AsSpan(row * 3 * inner + part * inner, inner));
            }
        }

        return Linear.FromWeights(Tensor.Persistent(weight, [w, 3 * inner], device, requiresGrad: true));
    }

    // The feed-forward: wi (ReLU) and wo, or the gate wi_0, the up projection wi_1 and wo (gated), no biases.
    private static FeedForward FeedForwardOf(ITensorStore store, string prefix, Sizes sizes, Device device)
    {
        int w = sizes.Width, f = sizes.FeedForward;
        var down = Dense(store, $"{prefix}.wo.weight", f, w, device);
        try
        {
            return sizes.Gated
                ? new FeedForward(Dense(store, $"{prefix}.wi_0.weight", w, f, device), Dense(store, $"{prefix}.wi_1.weight", w, f, device), down, sizes.Activation)
                : new FeedForward(null, Dense(store, $"{prefix}.wi.weight", w, f, device), down, sizes.Activation);
        }
        catch
        {
            down.Dispose();
            throw;
        }
    }

    // A dense layer [inputs, outputs] from Hugging Face's [outputs, inputs] weight, no bias.
    private static Linear Dense(ITensorStore store, string name, int inputs, int outputs, Device device) =>
        Linear.FromWeights(Tensor.Persistent(Checked(store, name, [inputs, outputs], transposed: true), [inputs, outputs], device, requiresGrad: true));

    private static RMSNorm Norm(ITensorStore store, string name, Sizes sizes, Device device) =>
        RMSNorm.FromWeights(Tensor.Persistent(Checked(store, name, [sizes.Width], transposed: false), [sizes.Width], device, requiresGrad: true), sizes.Epsilon);

    // The tensor's values in `shape` (Idrak's layout): read as stored, or transposed from [shape[1], shape[0]].
    private static float[] Checked(ITensorStore store, string name, int[] shape, bool transposed)
    {
        if (!store.Contains(name))
        {
            throw new InvalidDataException($"The T5 checkpoint has no tensor {name}.");
        }

        int[] stored = store.ShapeOf(name);
        int[] expected = transposed ? [shape[1], shape[0]] : shape;
        if (!stored.SequenceEqual(expected))
        {
            throw new InvalidDataException($"The T5 tensor {name} is [{string.Join(", ", stored)}]; the configuration makes it [{string.Join(", ", expected)}].");
        }

        return transposed ? store.ReadTransposed(name) : store.Read(name);
    }

    private static T Keep<T>(List<IDisposable> built, T part)
        where T : IDisposable
    {
        built.Add(part);
        return part;
    }

    /// <summary>
    /// The decoder's input embedding: the encoder's token embedding read in place (T5's shared table), so the table is held
    /// and trained once; its parameters are the encoder's, not this module's.
    /// </summary>
    private sealed class SharedEmbedding(Embedding words) : Module
    {
        protected override Tensor ForwardCore(Tensor input) => words.Forward(input);

        public override IEnumerable<Tensor> Parameters() => [];

        public override string ToString() => $"SharedEmbedding({words})";
    }
}

/// <summary>The T5 encoder-decoder under one architecture name (T5ForConditionalGeneration, MT5ForConditionalGeneration).</summary>
internal sealed class T5Seq2SeqFamily(string architecture) : IPretrainedSeq2SeqFamily
{
    public string Name { get; } = architecture;

    public ISeq2SeqModel Build(PretrainedCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var sizes = T5Layers.Sizes.Read(checkpoint.Config);
        var store = checkpoint.Tensors;
        var device = checkpoint.Device;
        bool tie = checkpoint.Config["tie_word_embeddings"] is not { } tie0 || tie0.GetValueKind() != JsonValueKind.False;
        if ((uint)sizes.StartToken >= (uint)sizes.Vocabulary || (uint)sizes.EndToken >= (uint)sizes.Vocabulary)
        {
            throw new InvalidDataException($"T5's decoder_start_token_id {sizes.StartToken} and eos_token_id {sizes.EndToken} are tokens of its vocab_size {sizes.Vocabulary}.");
        }

        var words = T5Layers.Shared(store, sizes, device);
        SequenceEncoder? encoder = null;
        Sequential? decoder = null;
        try
        {
            encoder = T5Layers.Encoder(store, sizes, words, device);
            decoder = T5Layers.Decoder(store, sizes, words, device, tie);
            return new Seq2SeqNetwork(encoder, decoder, sizes.StartToken, sizes.EndToken, sizes.MaxTargetPositions);
        }
        catch
        {
            decoder?.Dispose();
            if (encoder is not null)
            {
                encoder.Dispose();                                             // the shared embedding with it
            }
            else
            {
                words.Dispose();
            }

            throw;
        }
    }

    // T5 was trained with </s> closing every source: appended when a tokenizer without a post-processor leaves it out.
    public bool EndsSource(JsonObject config) => true;
}

/// <summary>The T5 encoder alone under one architecture name (T5EncoderModel, MT5EncoderModel): sentence vectors by the mean of its states.</summary>
internal sealed class T5EncoderFamily(string architecture) : IPretrainedEncoderFamily
{
    public string Name { get; } = architecture;

    public ISequenceEncoder Build(PretrainedCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var sizes = T5Layers.Sizes.Read(checkpoint.Config);
        var words = T5Layers.Shared(checkpoint.Tensors, sizes, checkpoint.Device);
        try
        {
            return T5Layers.Encoder(checkpoint.Tensors, sizes, words, checkpoint.Device);
        }
        catch
        {
            words.Dispose();
            throw;
        }
    }

    // T5's encoder has no first-token summary: its sentence vectors are the mean of the states (as sentence-T5's).
    public string Pooler(JsonObject config) => SequencePoolers.Mean;
}
