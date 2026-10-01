// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Layers;

/// <summary>The normalization of a decoder model.</summary>
public enum DecoderNorm
{
    /// <summary>RMS normalization (<see cref="RMSNorm"/>).</summary>
    Rms,

    /// <summary>Layer normalization with a bias (<see cref="LayerNorm"/>).</summary>
    Layer,
}

/// <summary>
/// Provides weights by name for <see cref="DecoderSpec.Build"/>, in Idrak's layout (Linear weights [in, out]). The
/// names are listed in <see cref="DecoderSpec"/>. Implementations read files (for example a pretrained-model package
/// translating another framework's names and layouts) or anything else.
/// </summary>
public interface IWeightSource
{
    /// <summary>The values of the tensor <paramref name="name"/> with <paramref name="shape"/>, or null when the source does not have it.</summary>
    float[]? Read(string name, IReadOnlyList<int> shape);
}

/// <summary>Settings for <see cref="DecoderSpec.Build"/>.</summary>
public sealed record DecoderBuildOptions
{
    /// <summary>Where the layers are created (the default device when null).</summary>
    public Device? Device { get; init; }

    /// <summary>
    /// Store every Linear layer (attention, feed-forward and output head) as int8: each weight is quantized on the host as
    /// it is read, so a large model never exists as float32 on the device.
    /// </summary>
    public bool Int8 { get; init; }

    /// <summary>
    /// Store the projections as bfloat16 (half the float32 memory, values to about 3 significant digits; exact for
    /// checkpoints stored in bfloat16). <see cref="Int8"/> takes precedence.
    /// </summary>
    public bool BFloat16 { get; init; }

    /// <summary>
    /// Store the projections as 4-bit weights (see <see cref="Int4Weight"/>): about 5 bits per weight, quantized on the host
    /// as they are read. <see cref="Int8"/> takes precedence; this over <see cref="BFloat16"/>.
    /// </summary>
    public bool Int4 { get; init; }

    /// <summary>
    /// Store the projections in the packed format registered under this name (see <see cref="PackedWeight.FormatNames"/>:
    /// "int8", "int4", "bfloat16", or a format of your own added with <see cref="PackedWeight.Register"/>), packed on the
    /// host as they are read. Takes precedence over <see cref="Int8"/>, <see cref="Int4"/> and <see cref="BFloat16"/>.
    /// </summary>
    public string? PackedFormatName { get; init; }

    /// <summary>
    /// Standard deviation of a normal initialization of every weight matrix and embedding table (biases zero, norm gains
    /// one), as GPT-2 and nanoGPT use (0.02). Null keeps the default (uniform Xavier weights, ±0.02 embeddings).
    /// Ignored when building from weights.
    /// </summary>
    public float? InitStd { get; init; }

    /// <summary>Longest sequence the model will see (the rotary tables' size); the spec's <see cref="DecoderSpec.MaxPositions"/> when null.</summary>
    public int? MaxPositions { get; init; }

    /// <summary>Seed for random initialization (when no weights are given).</summary>
    public int Seed { get; init; }
}

/// <summary>
/// A decoder-only language model described by its settings, so different model families are data rather than code:
/// token embedding (optionally scaled) → <see cref="Layers"/> × <see cref="DecoderBlock"/> (normalization, causal
/// self-attention with grouped-query heads, rotary embeddings, feed-forward block) → final normalization → output head.
/// <see cref="Build"/> creates it with random weights or from an <see cref="IWeightSource"/>, as a <see cref="Sequential"/>
/// that works with <see cref="Generation.TextGenerator"/>, the KV cache, int8 quantization and LoRA.
/// <para>
/// Weight names (Idrak layout): <c>embed</c> [vocabulary, dim]; for each layer i, <c>layers.i.attn_norm</c>,
/// <c>layers.i.attn.q</c> [dim, heads·headDim], <c>layers.i.attn.k</c> and <c>layers.i.attn.v</c> [dim, kvHeads·headDim],
/// <c>layers.i.attn.o</c> [heads·headDim, dim], <c>layers.i.attn.q_norm</c>/<c>k_norm</c> [headDim],
/// <c>layers.i.mlp_norm</c>, <c>layers.i.post_attn_norm</c>, <c>layers.i.post_mlp_norm</c>,
/// <c>layers.i.mlp.gate</c>/<c>up</c> [dim, ffDim], <c>layers.i.mlp.down</c> [ffDim, dim]; <c>norm</c>; <c>head</c> [dim, vocabulary]
/// (not read when the embeddings are tied). Each name is followed by ".weight" or ".bias"; normalizations have
/// ".weight" (and ".bias" for layer normalization).
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

    /// <summary>Parameters of the model (weights and biases).</summary>
    public long ParameterCount
    {
        get
        {
            long attention = (long)Dim * HeadDim * (Heads + 2 * KvHeads) + (long)Heads * HeadDim * Dim;
            long feedForward = (long)Dim * FfDim * (Gated ? 3 : 2);
            long norms = Dim * (ParallelBlocks ? 1 : 2) * (PostNorms ? 2 : 1) + (QkNorm ? 2 * HeadDim : 0);
            return (long)Vocabulary * Dim * (TieEmbeddings ? 1 : 2) + (LearnedPositions ? (long)MaxPositions * Dim : 0)
                + Layers * (attention + feedForward + norms) + Dim;
        }
    }

    /// <summary>
    /// Creates the model: with <paramref name="weights"/>, every layer is made from the named weights (and quantized to
    /// int8 on the host when <see cref="DecoderBuildOptions.Int8"/>); without, from a seeded random initialization.
    /// </summary>
    public Sequential Build(IWeightSource? weights = null, DecoderBuildOptions? options = null)
    {
        options ??= new DecoderBuildOptions();
        var device = options.Device ?? Device.Default;
        int maxPositions = options.MaxPositions ?? MaxPositions;
        var random = new Random(options.Seed);
        var created = new List<Module>();
        Linear? head = null;
        try
        {
            Tensor Tensor(string name, int[] shape, Func<float[]> fallback)
            {
                var values = weights is null ? fallback() : weights.Read(name, shape) ?? throw new InvalidDataException($"The weights have no '{name}' {Idrak.Tensor.FormatShape(shape)}.");
                return Idrak.Tensor.Persistent(values, shape, device, requiresGrad: true);
            }

            float[] Uniform(int count, float bound) => [.. Enumerable.Range(0, count).Select(_ => (random.NextSingle() * 2f - 1f) * bound)];

            float[] Normal(int count, float std)
            {
                var values = new float[count];
                for (int i = 0; i < count; i++)
                {
                    double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
                    values[i] = (float)(std * Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2));
                }

                return values;
            }

            float[] Initial(int count, float uniformBound) => options.InitStd is { } std ? Normal(count, std) : Uniform(count, uniformBound);

            // The format the projections are packed in (frozen weights), or null for trainable float32 ones; a format named
            // in the options is looked up before anything is read.
            PackedFormat? packed = options.Int8 ? PackedFormat.Int8 : options.Int4 ? PackedFormat.Int4 : options.BFloat16 ? PackedFormat.BFloat16 : null;
            string? packedName = options.PackedFormatName;
            var factory = packedName is null ? null : PackedWeight.Factory(packedName);

            Linear Projection(string name, int inputs, int outputs, bool bias, float[]? transposedFrom = null)
            {
                int[] shape = [inputs, outputs];
                float[] Values() => transposedFrom is not null ? Transpose(transposedFrom, outputs, inputs)
                    : weights?.Read($"{name}.weight", shape) ?? (weights is null
                        ? Initial(inputs * outputs, MathF.Sqrt(6f / (inputs + outputs)))
                        : throw new InvalidDataException($"The weights have no '{name}.weight' [{inputs}, {outputs}]."));
                var b = bias ? Tensor($"{name}.bias", [outputs], () => new float[outputs]) : null;
                return factory is not null ? Linear.FromPacked(PackedWeight.Pack(factory, packedName!, Values(), inputs, outputs, device), b)
                    : packed is { } format ? Linear.FromPacked(PackedWeight.FromValues(format, Values(), inputs, outputs, device), b)
                    : Linear.FromWeights(Idrak.Tensor.Persistent(Values(), shape, device, requiresGrad: true), b);
            }

            Module Normalization(string name, int features)
            {
                var gain = Tensor($"{name}.weight", [features], () => Enumerable.Repeat(Norm == DecoderNorm.Rms ? 1f - NormOffset : 1f, features).ToArray());
                return Norm == DecoderNorm.Rms
                    ? RMSNorm.FromWeights(gain, NormEpsilon, NormOffset)
                    : LayerNorm.FromWeights(gain, Tensor($"{name}.bias", [features], () => new float[features]), NormEpsilon);
            }

            float[]? embeddingValues = weights is null
                ? (options.InitStd is { } embedStd ? Normal(Vocabulary * Dim, embedStd) : [.. Enumerable.Range(0, Vocabulary * Dim).Select(_ => (float)(random.NextDouble() * 2 - 1) * 0.02f)])
                : weights.Read("embed.weight", [Vocabulary, Dim]) ?? throw new InvalidDataException($"The weights have no 'embed.weight' [{Vocabulary}, {Dim}].");
            // Frozen-weight builds keep the table as bfloat16 (half the memory; lossless for bfloat16 checkpoints).
            bool frozen = packed is not null || factory is not null;
            var embedding = frozen
                ? Embedding.FromBFloat16(BFloat16Weight.FromValues(embeddingValues, Vocabulary, Dim, device))
                : Embedding.FromWeights(Idrak.Tensor.Persistent(embeddingValues, [Vocabulary, Dim], device, requiresGrad: true));
            embedding.Name = "embed";
            created.Add(embedding);
            // A packed tied head gets its own transposed copy of the table: made now, so the table's float values are not
            // kept alive while every layer is read (a float tied head reads the embedding in place; see below).
            if (TieEmbeddings && frozen)
            {
                head = Projection("head", Dim, Vocabulary, HeadBias, embeddingValues);
            }

            embeddingValues = null;
            if (EmbeddingScale is { } scale)
            {
                created.Add(new Scale(scale) { Name = "embed_scale" });
            }

            if (LearnedPositions)
            {
                created.Add(new PositionEmbedding(Tensor("pos.weight", [maxPositions, Dim],
                    () => options.InitStd is { } std ? Normal(maxPositions * Dim, std) : Uniform(maxPositions * Dim, 0.02f))) { Name = "pos" });
            }

            if (Dropout > 0f)
            {
                created.Add(new Dropout(Dropout, new Random(random.Next())) { Name = "embed_drop" });
            }

            for (int i = 0; i < Layers; i++)
            {
                string p = $"layers.{i}";
                var attentionNorm = Normalization($"{p}.attn_norm", Dim);
                var attention = new CausalSelfAttention(
                    Projection($"{p}.attn.q", Dim, Heads * HeadDim, QkvBias),
                    Projection($"{p}.attn.k", Dim, KvHeads * HeadDim, QkvBias),
                    Projection($"{p}.attn.v", Dim, KvHeads * HeadDim, QkvBias),
                    Projection($"{p}.attn.o", Heads * HeadDim, Dim, OutputBias),
                    QkNorm ? RMSNorm.FromWeights(Tensor($"{p}.attn.q_norm.weight", [HeadDim], () => Enumerable.Repeat(1f - NormOffset, HeadDim).ToArray()), NormEpsilon, NormOffset) : null,
                    QkNorm ? RMSNorm.FromWeights(Tensor($"{p}.attn.k_norm.weight", [HeadDim], () => Enumerable.Repeat(1f - NormOffset, HeadDim).ToArray()), NormEpsilon, NormOffset) : null,
                    Heads, KvHeads, HeadDim, Rope, maxPositions);
                var feedForwardNorm = ParallelBlocks ? null : Normalization($"{p}.mlp_norm", Dim);
                var feedForward = new FeedForward(
                    Gated ? Projection($"{p}.mlp.gate", Dim, FfDim, FeedForwardBias) : null,
                    Projection($"{p}.mlp.up", Dim, FfDim, FeedForwardBias),
                    Projection($"{p}.mlp.down", FfDim, Dim, FeedForwardBias),
                    Activation);
                created.Add(new DecoderBlock(attentionNorm, attention, feedForwardNorm, feedForward,
                    PostNorms ? Normalization($"{p}.post_attn_norm", Dim) : null,
                    PostNorms ? Normalization($"{p}.post_mlp_norm", Dim) : null,
                    Dropout > 0f ? new Dropout(Dropout, new Random(random.Next())) : null) { Name = p });
            }

            var norm = Normalization("norm", Dim);
            norm.Name = "norm";
            created.Add(norm);
            // A float tied head reads the embedding table in place; packed tied heads were made with the embedding.
            head ??= TieEmbeddings
                ? Linear.Tied(embedding, HeadBias ? Tensor("head.bias", [Vocabulary], () => new float[Vocabulary]) : null)
                : Projection("head", Dim, Vocabulary, HeadBias);
            head.Name = "head";
            created.Add(head);
            var blocks = created.OfType<DecoderBlock>().ToList();
            for (int i = 0; i < blocks.Count; i++)
            {
                blocks[i].NextNorm = (i + 1 < blocks.Count ? blocks[i + 1].AttentionNorm : norm) as RMSNorm;
            }

            var model = new Sequential(created) { Name = "decoder" };
            Specs.AddOrUpdate(model, this);
            return model;
        }
        catch
        {
            created.ForEach(m => m.Dispose());
            if (head is not null && !created.Contains(head))
            {
                head.Dispose();
            }

            throw;
        }
    }

    private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<Module, DecoderSpec> Specs = [];

    /// <summary>The spec a model was built from with <see cref="Build"/>, or null.</summary>
    public static DecoderSpec? Of(Module model) => Specs.TryGetValue(model, out var spec) ? spec : null;

    private static float[] Transpose(float[] values, int rows, int columns) => HostParallel.Transpose(values, rows, columns);

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
            var r = new JsonObject { ["theta"] = rope.Theta, ["interleaved"] = rope.Interleaved };
            if (rope.RotaryDim is { } rotary)
            {
                r["rotaryDim"] = rotary;
            }

            if (rope.Scaling is { } s)
            {
                r["scaling"] = new JsonObject
                {
                    ["type"] = s.Type, ["factor"] = s.Factor, ["lowFrequencyFactor"] = s.LowFrequencyFactor,
                    ["highFrequencyFactor"] = s.HighFrequencyFactor, ["originalMaxPositions"] = s.OriginalMaxPositions,
                };
            }

            json["rope"] = r;
        }

        return json;
    }

    /// <summary>Reads a spec written by <see cref="ToJson"/>.</summary>
    public static DecoderSpec FromJson(JsonObject json)
    {
        if (!IsDescription(json))
        {
            throw new InvalidDataException($"Not a decoder description (format '{json["format"]}').");
        }

        RopeSettings? rope = null;
        if (json["rope"] is JsonObject r)
        {
            RopeScaling? scaling = r["scaling"] is JsonObject s
                ? new((string)s["type"]!, (float)s["factor"]!, (float)s["lowFrequencyFactor"]!, (float)s["highFrequencyFactor"]!, (int)s["originalMaxPositions"]!)
                : null;
            rope = new RopeSettings((float)r["theta"]!, (int?)r["rotaryDim"], (bool)r["interleaved"]!, scaling);
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
            Rope = rope, LearnedPositions = (bool?)json["learnedPositions"] ?? false, Dropout = (float?)json["dropout"] ?? 0f,
        };
    }

    /// <summary>Whether <paramref name="json"/> is a decoder description.</summary>
    public static bool IsDescription(JsonObject json) => (string?)json["format"] == Format;
}
