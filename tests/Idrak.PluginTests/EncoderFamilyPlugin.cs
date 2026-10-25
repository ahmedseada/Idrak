// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Generation;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.PluginTests;

/// <summary>The sizes of a toy checkpoint written by <see cref="ToyEncoderPlugin"/>.</summary>
/// <param name="Vocabulary">Source tokens.</param>
/// <param name="Width">The model's width.</param>
/// <param name="Heads">Attention heads.</param>
/// <param name="Layers">Encoder blocks.</param>
/// <param name="FeedForward">The feed-forward blocks' hidden width.</param>
/// <param name="MaxPositions">The encoder's longest sequence.</param>
/// <param name="Types">Token types (0: none).</param>
/// <param name="Labels">The classifier's labels.</param>
/// <param name="PositionBias">The encoder's position bias kind ("none", "relative" or <see cref="ToyEncoderPlugin.DistanceBias"/>), shared by its blocks.</param>
/// <param name="DecoderVocabulary">Target tokens (encoder-decoders).</param>
/// <param name="DecoderLayers">Decoder blocks (encoder-decoders).</param>
/// <param name="DecoderMaxPositions">The decoder's longest target (encoder-decoders).</param>
/// <param name="Seed">The seed of the random tensors.</param>
public sealed record ToySizes(int Vocabulary = 40, int Width = 16, int Heads = 2, int Layers = 2, int FeedForward = 32, int MaxPositions = 32, int Types = 2,
    int Labels = 3, string PositionBias = "none", int DecoderVocabulary = 24, int DecoderLayers = 2, int DecoderMaxPositions = 16, int Seed = 1);

/// <summary>
/// An encoder family and a text encoder-decoder family of their own, registered from outside the library with its public
/// API alone (plan 26, P8): a toy encoder (token, type and learned position embeddings, pre-norm blocks of
/// <see cref="MultiHeadAttention"/> with an optional position bias shared by the blocks, a final norm) with a classifier and
/// a masked-language-model head from <see cref="EncoderHeads"/>, read by <see cref="PretrainedEncoderModel.Load"/>; and a
/// toy encoder-decoder (that encoder and a decoder of <see cref="DecoderBlock"/>s with <see cref="CrossAttention"/>, the
/// self-attention with an optional one-way position bias) read by <see cref="PretrainedSeq2SeqModel.Load"/>. The
/// checkpoints are safetensors files of random tensors under the family's own names (in Idrak's layout), written by
/// <see cref="WriteEncoder"/> and <see cref="WriteSeq2Seq"/>. A position bias kind of its own,
/// <see cref="DistanceBias"/>, derives from the library's <see cref="AttentionPositionBias"/>.
/// </summary>
public static class ToyEncoderPlugin
{
    /// <summary>The encoder family's architecture name.</summary>
    public const string Architecture = "OutsideToyEncoder";

    /// <summary>The encoder-decoder family's architecture name.</summary>
    public const string Seq2SeqArchitecture = "OutsideToySeq2Seq";

    /// <summary>The position bias kind of its own: a slope per head times the distance, no parameters.</summary>
    public const string DistanceBias = "outside-distance";

    private static readonly EncoderFamily Encoders = new();
    private static readonly Seq2SeqFamily Seq2Seq = new();

    /// <summary>Registers both families and the bias kind (registering twice registers once).</summary>
    public static void Register()
    {
        PretrainedEncoders.Register(Encoders);
        PretrainedSeq2Seq.Register(Seq2Seq);
        AttentionPositionBiases.Register(DistanceKind.Instance);
    }

    /// <summary>Removes both families and the bias kind.</summary>
    public static void Unregister()
    {
        PretrainedEncoders.Unregister(Architecture);
        PretrainedSeq2Seq.Unregister(Seq2SeqArchitecture);
        AttentionPositionBiases.Unregister(DistanceBias);
    }

    /// <summary>Writes a toy encoder checkpoint (config.json and model.safetensors) into <paramref name="folder"/>.</summary>
    public static string WriteEncoder(string folder, ToySizes sizes)
    {
        ArgumentNullException.ThrowIfNull(sizes);
        var random = new Random(sizes.Seed);
        var tensors = new List<(string, int[], float[])>();
        AddEncoder(tensors, sizes, random);
        int w = sizes.Width;
        Add(tensors, "cls.dense.weight", [w, w], random);
        Add(tensors, "cls.dense.bias", [w], random);
        Add(tensors, "cls.out.weight", [w, sizes.Labels], random);
        Add(tensors, "cls.out.bias", [sizes.Labels], random);
        Add(tensors, "mlm.dense.weight", [w, w], random);
        Add(tensors, "mlm.dense.bias", [w], random);
        Norm(tensors, "mlm.norm", w, random);
        Add(tensors, "mlm.out.weight", [w, sizes.Vocabulary], random);
        Add(tensors, "mlm.out.bias", [sizes.Vocabulary], random);
        return Write(folder, Architecture, sizes, tensors);
    }

    /// <summary>Writes a toy encoder-decoder checkpoint into <paramref name="folder"/>.</summary>
    public static string WriteSeq2Seq(string folder, ToySizes sizes)
    {
        ArgumentNullException.ThrowIfNull(sizes);
        var random = new Random(sizes.Seed);
        var tensors = new List<(string, int[], float[])>();
        AddEncoder(tensors, sizes, random);
        int w = sizes.Width;
        Add(tensors, "dec.words", [sizes.DecoderVocabulary, w], random);
        Add(tensors, "dec.positions", [sizes.DecoderMaxPositions, w], random);
        for (int i = 0; i < sizes.DecoderLayers; i++)
        {
            string p = $"dec.{i}";
            Norm(tensors, $"{p}.attn_norm", w, random);
            foreach (string name in new[] { "q", "k", "v", "o", "cq", "ck", "cv", "co" })
            {
                Add(tensors, $"{p}.{name}.weight", [w, w], random);
            }

            foreach (string name in new[] { "cq", "ck", "cv", "co" })
            {
                Add(tensors, $"{p}.{name}.bias", [w], random);
            }

            Norm(tensors, $"{p}.cross_norm", w, random);
            Norm(tensors, $"{p}.mlp_norm", w, random);
            Add(tensors, $"{p}.up.weight", [w, sizes.FeedForward], random);
            Add(tensors, $"{p}.up.bias", [sizes.FeedForward], random);
            Add(tensors, $"{p}.down.weight", [sizes.FeedForward, w], random);
            Add(tensors, $"{p}.down.bias", [w], random);
        }

        if (sizes.PositionBias == "relative")
        {
            Add(tensors, "dec.bias.table", [9, sizes.Heads], random);           // one-way, max distance 8
        }

        Norm(tensors, "dec.final_norm", w, random);
        Add(tensors, "dec.head.weight", [w, sizes.DecoderVocabulary], random);
        Add(tensors, "dec.head.bias", [sizes.DecoderVocabulary], random);
        return Write(folder, Seq2SeqArchitecture, sizes, tensors);
    }

    // The encoder's tensors (prefix "enc.").
    private static void AddEncoder(List<(string, int[], float[])> tensors, ToySizes sizes, Random random)
    {
        int w = sizes.Width;
        Add(tensors, "enc.words", [sizes.Vocabulary, w], random);
        if (sizes.Types > 0)
        {
            Add(tensors, "enc.types", [sizes.Types, w], random);
        }

        Add(tensors, "enc.positions", [sizes.MaxPositions, w], random);
        Norm(tensors, "enc.embedding_norm", w, random);
        for (int i = 0; i < sizes.Layers; i++)
        {
            string p = $"enc.{i}";
            Norm(tensors, $"{p}.norm1", w, random);
            Add(tensors, $"{p}.qkv.weight", [w, 3 * w], random);
            Add(tensors, $"{p}.qkv.bias", [3 * w], random);
            Add(tensors, $"{p}.out.weight", [w, w], random);
            Add(tensors, $"{p}.out.bias", [w], random);
            Norm(tensors, $"{p}.norm2", w, random);
            Add(tensors, $"{p}.ff1.weight", [w, sizes.FeedForward], random);
            Add(tensors, $"{p}.ff1.bias", [sizes.FeedForward], random);
            Add(tensors, $"{p}.ff2.weight", [sizes.FeedForward, w], random);
            Add(tensors, $"{p}.ff2.bias", [w], random);
        }

        if (sizes.PositionBias == "relative")
        {
            Add(tensors, "enc.bias.table", [17, sizes.Heads], random);          // both ways, max distance 8
        }

        Norm(tensors, "enc.final_norm", w, random);
    }

    private static void Add(List<(string, int[], float[])> tensors, string name, int[] shape, Random random) =>
        tensors.Add((name, shape, [.. Enumerable.Range(0, shape.Aggregate(1, (a, b) => a * b)).Select(_ => (random.NextSingle() * 2f - 1f) * 0.4f)]));

    private static void Norm(List<(string, int[], float[])> tensors, string name, int width, Random random)
    {
        tensors.Add(($"{name}.weight", [width], [.. Enumerable.Range(0, width).Select(_ => 1f + (random.NextSingle() - 0.5f) * 0.2f)]));
        tensors.Add(($"{name}.bias", [width], [.. Enumerable.Range(0, width).Select(_ => (random.NextSingle() - 0.5f) * 0.2f)]));
    }

    private static string Write(string folder, string architecture, ToySizes sizes, List<(string, int[], float[])> tensors)
    {
        Directory.CreateDirectory(folder);
        var config = new JsonObject
        {
            ["architectures"] = new JsonArray(architecture),
            ["vocab_size"] = sizes.Vocabulary, ["hidden"] = sizes.Width, ["heads"] = sizes.Heads, ["layers"] = sizes.Layers, ["ffn"] = sizes.FeedForward,
            ["max_positions"] = sizes.MaxPositions, ["type_vocab"] = sizes.Types, ["labels"] = sizes.Labels, ["position_bias"] = sizes.PositionBias,
            ["decoder_vocab"] = sizes.DecoderVocabulary, ["decoder_layers"] = sizes.DecoderLayers, ["decoder_max_positions"] = sizes.DecoderMaxPositions,
            ["start_token"] = 0, ["end_token"] = 1,
        };
        File.WriteAllText(Path.Combine(folder, "config.json"), config.ToJsonString());
        SafeTensorsWriter.Write(Path.Combine(folder, "model.safetensors"), tensors, SafeTensorType.F32);
        return folder;
    }

    // ---------------------------------------------------------------- reading

    private static int Int(JsonObject config, string key) => (int)config[key]!;

    private static Tensor Parameter(ITensorStore store, string name, Device device) =>
        store.Contains(name) ? Tensor.Persistent(store.Read(name), store.ShapeOf(name), device, requiresGrad: true)
            : throw new InvalidDataException($"The toy checkpoint has no tensor {name}.");

    private static Linear Dense(ITensorStore store, string name, Device device) =>
        Linear.FromWeights(Parameter(store, $"{name}.weight", device), store.Contains($"{name}.bias") ? Parameter(store, $"{name}.bias", device) : null);

    private static LayerNorm LayerNormOf(ITensorStore store, string name, Device device) =>
        LayerNorm.FromWeights(Parameter(store, $"{name}.weight", device), Parameter(store, $"{name}.bias", device), 1e-5f);

    // The family's position bias for `heads` heads (null for "none"), its tensors read from the store under `prefix`.
    private static IAttentionPositionBias? Bias(JsonObject config, ITensorStore store, string prefix, bool bidirectional, Device device)
    {
        string kind = (string?)config["position_bias"] ?? "none";
        if (kind == "none")
        {
            return null;
        }

        var settings = new JsonObject { ["max_distance"] = 8, ["bidirectional"] = bidirectional };
        return AttentionPositionBiases.Create(kind, new PositionBiasRequest(Int(config, "heads"), settings, device, 0,
            (name, shape) => store.Contains($"{prefix}.{name}") ? store.Read($"{prefix}.{name}") : null));
    }

    // The encoder: words, types, positions, embedding norm, pre-norm blocks over a shared bias, final norm.
    private static SequenceEncoder BuildEncoder(JsonObject config, ITensorStore store, Device device)
    {
        int heads = Int(config, "heads"), layers = Int(config, "layers");
        var bias = Bias(config, store, "enc.bias", bidirectional: true, device);
        var blocks = new List<Module>();
        for (int i = 0; i < layers; i++)
        {
            string p = $"enc.{i}";
            var attention = new MultiHeadAttention(Dense(store, $"{p}.qkv", device), Dense(store, $"{p}.out", device), heads)
            {
                PositionBias = bias,
                SharesPositionBias = i > 0,                                        // the first block owns it
            };
            blocks.Add(TransformerEncoderLayer.FromLayers(LayerNormOf(store, $"{p}.norm1", device), attention, LayerNormOf(store, $"{p}.norm2", device),
                Dense(store, $"{p}.ff1", device), Dense(store, $"{p}.ff2", device), new ExactGELU()));
        }

        return new SequenceEncoder(Embedding.FromWeights(Parameter(store, "enc.words", device)), blocks,
            new PositionEmbedding(Parameter(store, "enc.positions", device)),
            Int(config, "type_vocab") > 0 ? Embedding.FromWeights(Parameter(store, "enc.types", device)) : null,
            LayerNormOf(store, "enc.embedding_norm", device), LayerNormOf(store, "enc.final_norm", device));
    }

    // The head tensors under `prefix` ("cls", "mlm"), by the names EncoderHeads' kinds read.
    private static Func<string, IReadOnlyList<int>, float[]?> Reader(ITensorStore store, string prefix) =>
        (name, _) => store.Contains($"{prefix}.{name}") ? store.Read($"{prefix}.{name}") : null;

    private sealed class EncoderFamily : IPretrainedEncoderFamily
    {
        public string Name => Architecture;

        public ISequenceEncoder Build(PretrainedCheckpoint checkpoint) => BuildEncoder(checkpoint.Config, checkpoint.Tensors, checkpoint.Device);

        public IReadOnlyList<EncoderHead> Heads(PretrainedCheckpoint checkpoint, ISequenceEncoder encoder)
        {
            var config = checkpoint.Config;
            return
            [
                EncoderHeads.Create(EncoderHeads.SequenceClassification, new EncoderHeadRequest(encoder.Width, Int(config, "labels"),
                    new JsonObject { ["dense"] = true }, checkpoint.Device, 0, Reader(checkpoint.Tensors, "cls"))),
                EncoderHeads.Create(EncoderHeads.MaskedLanguageModel, new EncoderHeadRequest(encoder.Width, encoder.Vocabulary,
                    new JsonObject { ["transform"] = true, ["norm_epsilon"] = 1e-5 }, checkpoint.Device, 0, Reader(checkpoint.Tensors, "mlm"))),
            ];
        }

        public string Pooler(JsonObject config) => (string?)config["pooler"] ?? SequencePoolers.First;
    }

    private sealed class Seq2SeqFamily : IPretrainedSeq2SeqFamily
    {
        public string Name => Seq2SeqArchitecture;

        public ISeq2SeqModel Build(PretrainedCheckpoint checkpoint)
        {
            var (config, store, device) = (checkpoint.Config, checkpoint.Tensors, checkpoint.Device);
            var encoder = BuildEncoder(config, store, device);
            int heads = Int(config, "heads"), width = Int(config, "hidden"), positions = Int(config, "decoder_max_positions");
            var bias = Bias(config, store, "dec.bias", bidirectional: false, device);
            var modules = new List<Module>
            {
                Embedding.FromWeights(Parameter(store, "dec.words", device)),
                new PositionEmbedding(Parameter(store, "dec.positions", device)),
            };
            for (int i = 0; i < Int(config, "decoder_layers"); i++)
            {
                string p = $"dec.{i}";
                var self = new CausalSelfAttention(Dense(store, $"{p}.q", device), Dense(store, $"{p}.k", device), Dense(store, $"{p}.v", device),
                    Dense(store, $"{p}.o", device), null, null, heads, heads, width / heads, null, positions)
                {
                    PositionBias = bias,
                    SharesPositionBias = i > 0,
                };
                var cross = new CrossAttention(Dense(store, $"{p}.cq", device), Dense(store, $"{p}.ck", device), Dense(store, $"{p}.cv", device),
                    Dense(store, $"{p}.co", device), heads);
                modules.Add(new DecoderBlock(LayerNormOf(store, $"{p}.attn_norm", device), self, LayerNormOf(store, $"{p}.cross_norm", device), cross,
                    LayerNormOf(store, $"{p}.mlp_norm", device), new FeedForward(null, Dense(store, $"{p}.up", device), Dense(store, $"{p}.down", device), FeedForwardActivation.Gelu)));
            }

            modules.Add(LayerNormOf(store, "dec.final_norm", device));
            modules.Add(Dense(store, "dec.head", device));
            return new Seq2SeqNetwork(encoder, new Sequential(modules), Int(config, "start_token"), Int(config, "end_token"));
        }
    }

    // The bias kind of its own: no settings, no trained tensors.
    private sealed class DistanceKind : IPositionBiasKind
    {
        public static readonly DistanceKind Instance = new();

        public string Name => DistanceBias;

        public string Summary => "a slope per head times the distance";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public IReadOnlyList<string> Weights => [];

        public IAttentionPositionBias Create(PositionBiasRequest request) => new Distance(request.Heads);
    }

    // -slope(h) · |key - query|, slope(h) = 0.5 / (h + 1): computed by position, no parameters (a kind of its own on the
    // library's base, which keeps it per shape).
    private sealed class Distance(int heads) : AttentionPositionBias(heads)
    {
        protected override Tensor Compute(int queries, int keys, int queryStart, Device device)
        {
            var values = new float[Heads * queries * keys];
            for (int h = 0; h < Heads; h++)
            {
                float slope = 0.5f / (h + 1);
                for (int i = 0; i < queries; i++)
                {
                    for (int j = 0; j < keys; j++)
                    {
                        values[(h * queries + i) * keys + j] = -slope * Math.Abs(j - (queryStart + i));
                    }
                }
            }

            return Tensor.From(values, [Heads, queries, keys], device);
        }

        public override string ToString() => $"DistanceBias({Heads} heads)";
    }
}
