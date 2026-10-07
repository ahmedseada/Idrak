// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>The normalization of a decoder model.</summary>
public enum DecoderNorm
{
    /// <summary>RMS normalization (Idrak's RMSNorm layer).</summary>
    Rms,

    /// <summary>Layer normalization with a bias (Idrak's LayerNorm layer).</summary>
    Layer,
}

/// <summary>The activation of a decoder's feed-forward blocks.</summary>
public enum FeedForwardActivation
{
    /// <summary>x · sigmoid(x) (also called swish; with a gate: SwiGLU).</summary>
    Silu,

    /// <summary>GELU, tanh approximation (with a gate: GeGLU).</summary>
    Gelu,

    /// <summary>max(0, x).</summary>
    Relu,
}

/// <summary>
/// A decoder-only language model described by its settings, so different model families are data rather than code:
/// token embedding (optionally scaled) → <see cref="Layers"/> × decoder block (normalization, causal
/// self-attention with grouped-query heads, rotary embeddings, feed-forward block) → final normalization → output head.
/// Idrak's <c>DecoderBuilder.Build</c> (<c>spec.Build(weights, options)</c>) creates it with random weights or from an
/// <c>IWeightSource</c>, as a network that works with text generation, the KV cache, int8 quantization and LoRA.
/// A spec is data: model families (<c>PretrainedArchitectures</c> in Idrak) read it from a checkpoint's configuration, and
/// model packages store it as JSON (<see cref="ToJson"/>).
/// <para>
/// Weight names (Idrak layout): <c>embed</c> [vocabulary, dim]; for each layer i, <c>layers.i.attn_norm</c>,
/// <c>layers.i.attn.q</c> [dim, heads·headDim], <c>layers.i.attn.k</c> and <c>layers.i.attn.v</c> [dim, kvHeads·headDim],
/// <c>layers.i.attn.o</c> [heads·headDim, dim], <c>layers.i.attn.q_norm</c>/<c>k_norm</c> [headDim],
/// <c>layers.i.mlp_norm</c>, <c>layers.i.post_attn_norm</c>, <c>layers.i.post_mlp_norm</c>,
/// <c>layers.i.mlp.gate</c>/<c>up</c> [dim, ffDim], <c>layers.i.mlp.down</c> [ffDim, dim]; <c>norm</c>; <c>head</c> [dim, vocabulary]
/// (not read when the embeddings are tied). Each name is followed by ".weight" or ".bias"; normalizations have
/// ".weight" (and ".bias" for layer normalization).
/// </para>
/// <para>
/// Mixture-of-experts layers (<see cref="Experts"/> above 0) have instead
/// <c>layers.i.mlp.router</c> [dim, experts], <c>layers.i.mlp.experts.j.gate</c>/<c>up</c> [dim, expertFfDim] and
/// <c>layers.i.mlp.experts.j.down</c> [expertFfDim, dim] for each expert j, and with a shared expert
/// <c>layers.i.mlp.shared.gate</c>/<c>up</c>/<c>down</c> (its hidden size <see cref="SharedExpertFfDim"/>) and
/// <c>layers.i.mlp.shared_gate</c> [dim, 1].
/// </para>
/// </summary>
public sealed record DecoderSpec
{
    private const string Format = "idrak-decoder/1";

    /// <summary>Token ids.</summary>
    public required int Vocabulary { get; init; }

    /// <summary>Model width.</summary>
    public required int Dim { get; init; }

    /// <summary>Decoder blocks.</summary>
    public required int Layers { get; init; }

    /// <summary>Query heads.</summary>
    public required int Heads { get; init; }

    /// <summary>Key/value heads (equal to <see cref="Heads"/> for standard attention; fewer for grouped-query attention).</summary>
    public required int KvHeads { get; init; }

    /// <summary>Size of each attention head (often Dim / Heads, but not always).</summary>
    public required int HeadDim { get; init; }

    /// <summary>Hidden size of the feed-forward blocks.</summary>
    public required int FfDim { get; init; }

    /// <summary>Longest sequence the model supports.</summary>
    public required int MaxPositions { get; init; }

    /// <summary>The normalization.</summary>
    public DecoderNorm Norm { get; init; } = DecoderNorm.Rms;

    /// <summary>Normalization epsilon.</summary>
    public float NormEpsilon { get; init; } = 1e-6f;

    /// <summary>RMS normalization gain offset (the effective gain is weight + offset).</summary>
    public float NormOffset { get; init; }

    /// <summary>Gated feed-forward blocks (SwiGLU/GeGLU) instead of plain ones.</summary>
    public bool Gated { get; init; } = true;

    /// <summary>The feed-forward activation.</summary>
    public FeedForwardActivation Activation { get; init; } = FeedForwardActivation.Silu;

    /// <summary>Rotary embedding settings (null for none).</summary>
    public RopeSettings? Rope { get; init; } = new(10000f);

    /// <summary>Biases on the query, key and value projections.</summary>
    public bool QkvBias { get; init; }

    /// <summary>A bias on the attention output projection.</summary>
    public bool OutputBias { get; init; }

    /// <summary>Biases in the feed-forward blocks.</summary>
    public bool FeedForwardBias { get; init; }

    /// <summary>RMS-normalize each head's queries and keys.</summary>
    public bool QkNorm { get; init; }

    /// <summary>Normalize the attention and feed-forward outputs before the residual additions too.</summary>
    public bool PostNorms { get; init; }

    /// <summary>Attention and feed-forward blocks read the same normalized input and are added together.</summary>
    public bool ParallelBlocks { get; init; }

    /// <summary>The output head reuses the (transposed) token embedding.</summary>
    public bool TieEmbeddings { get; init; }

    /// <summary>Multiply the token embeddings by this (null for none).</summary>
    public float? EmbeddingScale { get; init; }

    /// <summary>A bias on the output head.</summary>
    public bool HeadBias { get; init; }

    /// <summary>
    /// Learned absolute positions ("pos.weight" [<see cref="MaxPositions"/>, dim], added to the token embeddings, as in
    /// GPT-2); usually with <see cref="Rope"/> null.
    /// </summary>
    public bool LearnedPositions { get; init; }

    /// <summary>
    /// Dropout probability during training: on the embeddings and on each block's attention and feed-forward outputs
    /// before the residual additions (GPT-2's embedding and residual dropout). Attention weights are not dropped.
    /// </summary>
    public float Dropout { get; init; }

    /// <summary>
    /// How many positions each query of a windowed layer attends to, itself included (sliding-window attention, as in
    /// Mistral, Qwen2, Gemma 2 and 3), or null for full causal attention in every layer.
    /// </summary>
    public int? SlidingWindow { get; init; }

    /// <summary>
    /// Which layers use <see cref="SlidingWindow"/>, one entry per layer (Gemma 2 alternates; Qwen2 windows the layers from
    /// max_window_layers on); null: every layer, when a window is set.
    /// </summary>
    public IReadOnlyList<bool>? SlidingWindowLayers { get; init; }

    /// <summary>The rotary settings of the windowed layers (Gemma 3's local base), or null to use <see cref="Rope"/> in every layer.</summary>
    public RopeSettings? SlidingWindowRope { get; init; }

    /// <summary>The factor on the attention scores (1 / √<see cref="HeadDim"/> when null; Gemma 2 and 3: query_pre_attn_scalar^-1/2).</summary>
    public float? AttentionScale { get; init; }

    /// <summary>Soft-capping of the attention scores, cap · tanh(score / cap) before the mask (Gemma 2: 50), or null.</summary>
    public float? AttentionSoftcap { get; init; }

    /// <summary>Soft-capping of the output logits, cap · tanh(logit / cap) (Gemma 2: 30), or null.</summary>
    public float? LogitSoftcap { get; init; }

    /// <summary>
    /// Experts in each mixture-of-experts layer (Mixtral's num_local_experts, Qwen's num_experts), or 0 for dense
    /// feed-forward blocks everywhere.
    /// </summary>
    public int Experts { get; init; }

    /// <summary>Experts each token is routed to (num_experts_per_tok), between 1 and <see cref="Experts"/>.</summary>
    public int ExpertsPerToken { get; init; }

    /// <summary>Hidden size of each expert (Qwen's moe_intermediate_size), or 0 for <see cref="FfDim"/>.</summary>
    public int ExpertFfDim { get; init; }

    /// <summary>Hidden size of the shared expert every token also goes through, scaled by a sigmoid gate (Qwen2-MoE), or 0 for none.</summary>
    public int SharedExpertFfDim { get; init; }

    /// <summary>Renormalize the chosen experts' probabilities to sum to one (Mixtral always; Qwen per norm_topk_prob).</summary>
    public bool NormalizeTopK { get; init; } = true;

    /// <summary>
    /// Which layers are mixture-of-experts layers, one entry per layer (the others are dense, of width <see cref="FfDim"/>;
    /// Qwen's decoder_sparse_step and mlp_only_layers); null: every layer, when <see cref="Experts"/> is set.
    /// </summary>
    public IReadOnlyList<bool>? ExpertLayers { get; init; }

    /// <summary>Whether layer <paramref name="layer"/> is a mixture-of-experts layer.</summary>
    public bool IsExpertLayer(int layer) => Experts > 0 && (ExpertLayers is not { } layers || layers[layer]);

    /// <summary>The hidden size of each expert: <see cref="ExpertFfDim"/>, or <see cref="FfDim"/> when that is 0.</summary>
    public int ExpertHidden => ExpertFfDim > 0 ? ExpertFfDim : FfDim;

    /// <summary>Whether layer <paramref name="layer"/> attends through the sliding window.</summary>
    public bool IsWindowed(int layer) => SlidingWindow is not null && (SlidingWindowLayers is not { } layers || layers[layer]);

    /// <summary>Parameters of the model (weights and biases).</summary>
    public long ParameterCount => Count(Experts);

    /// <summary>
    /// Parameters each token goes through: <see cref="ParameterCount"/> for a dense model; with experts, only
    /// <see cref="ExpertsPerToken"/> experts of each mixture-of-experts layer count (the router and a shared expert do).
    /// </summary>
    public long ActiveParameterCount => Count(ExpertsPerToken);

    // The parameters with `experts` experts counted in each mixture-of-experts layer.
    private long Count(int experts)
    {
        long attention = (long)Dim * HeadDim * (Heads + 2 * KvHeads) + (long)Heads * HeadDim * Dim;
        int projections = Gated ? 3 : 2;
        long dense = (long)Dim * FfDim * projections;
        long sparse = (long)Dim * Experts + (long)experts * Dim * ExpertHidden * projections
            + (SharedExpertFfDim > 0 ? (long)Dim * SharedExpertFfDim * projections + Dim : 0);
        long norms = Dim * (ParallelBlocks ? 1 : 2) * (PostNorms ? 2 : 1) + (QkNorm ? 2 * HeadDim : 0);
        long layers = 0;
        for (int i = 0; i < Layers; i++)
        {
            layers += attention + (IsExpertLayer(i) ? sparse : dense) + norms;
        }

        return (long)Vocabulary * Dim * (TieEmbeddings ? 1 : 2) + (LearnedPositions ? (long)MaxPositions * Dim : 0) + layers + Dim;
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Module, DecoderSpec> Specs = [];

    /// <summary>The spec <paramref name="model"/> was built from (see <see cref="Describe"/>), or null.</summary>
    public static DecoderSpec? Of(Module model) => Specs.TryGetValue(model, out var spec) ? spec : null;

    /// <summary>
    /// Records that <paramref name="model"/> is the network this spec describes, so <see cref="Of"/> finds it (the inference
    /// engine reads the model's sizes from it). Building a spec records it; a family that builds its own network may too.
    /// </summary>
    public void Describe(Module model)
    {
        ArgumentNullException.ThrowIfNull(model);
        Specs.AddOrUpdate(model, this);
    }

    /// <summary>The spec as JSON (format "idrak-decoder/1"), for packages.</summary>
    public JsonObject ToJson()
    {
        var json = new JsonObject
        {
            ["format"] = Format,
            ["vocabulary"] = Vocabulary, ["dim"] = Dim, ["layers"] = Layers, ["heads"] = Heads, ["kvHeads"] = KvHeads,
            ["headDim"] = HeadDim, ["ffDim"] = FfDim, ["maxPositions"] = MaxPositions,
            ["norm"] = Norm.ToString(), ["normEpsilon"] = NormEpsilon, ["normOffset"] = NormOffset,
            ["gated"] = Gated, ["activation"] = Activation.ToString(),
            ["qkvBias"] = QkvBias, ["outputBias"] = OutputBias, ["feedForwardBias"] = FeedForwardBias, ["qkNorm"] = QkNorm,
            ["postNorms"] = PostNorms, ["parallelBlocks"] = ParallelBlocks, ["tieEmbeddings"] = TieEmbeddings, ["headBias"] = HeadBias,
            ["learnedPositions"] = LearnedPositions, ["dropout"] = Dropout,
        };
        if (EmbeddingScale is { } scale)
        {
            json["embeddingScale"] = scale;
        }

        if (Rope is { } rope)
        {
            json["rope"] = RopeJson(rope);
        }

        if (SlidingWindow is { } window)
        {
            json["slidingWindow"] = window;
        }

        if (SlidingWindowLayers is { } layers)
        {
            json["slidingWindowLayers"] = new JsonArray([.. layers.Select(l => (JsonNode)l)]);
        }

        if (SlidingWindowRope is { } local)
        {
            json["slidingWindowRope"] = RopeJson(local);
        }

        if (AttentionScale is { } attentionScale)
        {
            json["attentionScale"] = attentionScale;
        }

        if (AttentionSoftcap is { } attentionSoftcap)
        {
            json["attentionSoftcap"] = attentionSoftcap;
        }

        if (LogitSoftcap is { } logitSoftcap)
        {
            json["logitSoftcap"] = logitSoftcap;
        }

        if (Experts > 0)
        {
            json["experts"] = Experts;
            json["expertsPerToken"] = ExpertsPerToken;
            json["expertFfDim"] = ExpertFfDim;
            json["sharedExpertFfDim"] = SharedExpertFfDim;
            json["normalizeTopK"] = NormalizeTopK;
            if (ExpertLayers is { } expertLayers)
            {
                json["expertLayers"] = new JsonArray([.. expertLayers.Select(l => (JsonNode)l)]);
            }
        }

        return json;
    }

    private static JsonObject RopeJson(RopeSettings rope)
    {
        var r = new JsonObject { ["theta"] = rope.Theta, ["interleaved"] = rope.Interleaved };
        if (rope.RotaryDim is { } rotary)
        {
            r["rotaryDim"] = rotary;
        }

        if (rope.Scaling is { } s)
        {
            r["scaling"] = new JsonObject { ["type"] = s.Type, ["parameters"] = s.Parameters };
        }

        return r;
    }

    private static RopeSettings RopeFromJson(JsonObject r)
    {
        RopeScaling? scaling = r["scaling"] is not JsonObject s ? null
            : s["parameters"] is JsonObject parameters ? new((string)s["type"]!, parameters)
            : (string)s["type"]! == "llama3"                                  // written before scalings were registered
                ? RopeScaling.Llama3((double)s["factor"]!, (double)s["lowFrequencyFactor"]!, (double)s["highFrequencyFactor"]!, (int)s["originalMaxPositions"]!)
                : new((string)s["type"]!, new JsonObject { ["factor"] = (double)s["factor"]! });
        return new RopeSettings((float)r["theta"]!, (int?)r["rotaryDim"], (bool)r["interleaved"]!, scaling);
    }

    /// <summary>Reads a spec written by <see cref="ToJson"/>.</summary>
    public static DecoderSpec FromJson(JsonObject json)
    {
        if (!IsDescription(json))
        {
            throw new InvalidDataException($"Not a decoder description (format '{json["format"]}').");
        }

        return new DecoderSpec
        {
            Vocabulary = (int)json["vocabulary"]!, Dim = (int)json["dim"]!, Layers = (int)json["layers"]!, Heads = (int)json["heads"]!,
            KvHeads = (int)json["kvHeads"]!, HeadDim = (int)json["headDim"]!, FfDim = (int)json["ffDim"]!, MaxPositions = (int)json["maxPositions"]!,
            Norm = Enum.Parse<DecoderNorm>((string)json["norm"]!), NormEpsilon = (float)json["normEpsilon"]!, NormOffset = (float)json["normOffset"]!,
            Gated = (bool)json["gated"]!, Activation = Enum.Parse<FeedForwardActivation>((string)json["activation"]!),
            QkvBias = (bool)json["qkvBias"]!, OutputBias = (bool)json["outputBias"]!, FeedForwardBias = (bool)json["feedForwardBias"]!,
            QkNorm = (bool)json["qkNorm"]!, PostNorms = (bool)json["postNorms"]!, ParallelBlocks = (bool)json["parallelBlocks"]!,
            TieEmbeddings = (bool)json["tieEmbeddings"]!, HeadBias = (bool)json["headBias"]!, EmbeddingScale = (float?)json["embeddingScale"],
            Rope = json["rope"] is JsonObject r ? RopeFromJson(r) : null, LearnedPositions = (bool?)json["learnedPositions"] ?? false, Dropout = (float?)json["dropout"] ?? 0f,
            SlidingWindow = (int?)json["slidingWindow"],
            SlidingWindowLayers = json["slidingWindowLayers"] is JsonArray layers ? [.. layers.Select(l => (bool)l!)] : null,
            SlidingWindowRope = json["slidingWindowRope"] is JsonObject local ? RopeFromJson(local) : null,
            AttentionScale = (float?)json["attentionScale"], AttentionSoftcap = (float?)json["attentionSoftcap"], LogitSoftcap = (float?)json["logitSoftcap"],
            Experts = (int?)json["experts"] ?? 0, ExpertsPerToken = (int?)json["expertsPerToken"] ?? 0, ExpertFfDim = (int?)json["expertFfDim"] ?? 0,
            SharedExpertFfDim = (int?)json["sharedExpertFfDim"] ?? 0, NormalizeTopK = (bool?)json["normalizeTopK"] ?? true,
            ExpertLayers = json["expertLayers"] is JsonArray expertLayers ? [.. expertLayers.Select(l => (bool)l!)] : null,
        };
    }

    /// <summary>Whether <paramref name="json"/> is a decoder description.</summary>
    public static bool IsDescription(JsonObject json) => (string?)json["format"] == Format;
}
