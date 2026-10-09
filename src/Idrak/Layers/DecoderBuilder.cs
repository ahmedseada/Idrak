// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// Builds the network a <see cref="DecoderSpec"/> (Idrak.Abstraction) describes from Idrak's layers: token embedding,
/// <see cref="DecoderBlock"/>s (normalization, <see cref="CausalSelfAttention"/>, <see cref="FeedForward"/> or
/// <see cref="MixtureOfExperts"/>), final normalization and output head, in a <see cref="Sequential"/>.
/// </summary>
public static class DecoderBuilder
{
    /// <summary>
    /// Creates the model: with <paramref name="weights"/>, every layer is made from the named weights (and quantized to
    /// int8 on the host when <see cref="DecoderBuildOptions.Int8"/>); without, from a seeded random initialization.
    /// </summary>
    /// <param name="spec">The model to create.</param>
    /// <param name="weights">The named weights to read, or null for a random initialization.</param>
    /// <param name="options">Device, packed weight format, context length and initialization.</param>
    public static Sequential Build(this DecoderSpec spec, IWeightSource? weights = null, DecoderBuildOptions? options = null)
    {
        options ??= new DecoderBuildOptions();
        if (spec.SlidingWindowLayers is { } windowed && windowed.Count != spec.Layers)
        {
            throw new ArgumentException($"SlidingWindowLayers has {windowed.Count} entries for {spec.Layers} layers.");
        }

        if (spec.ExpertLayers is { } expertLayers && expertLayers.Count != spec.Layers)
        {
            throw new ArgumentException($"ExpertLayers has {expertLayers.Count} entries for {spec.Layers} layers.");
        }

        if (spec.Experts < 0 || spec.Experts > 0 && (spec.ExpertsPerToken < 1 || spec.ExpertsPerToken > spec.Experts))
        {
            throw new ArgumentException($"{spec.ExpertsPerToken} experts per token of {spec.Experts}: each token goes to between 1 and {spec.Experts}.");
        }

        var device = options.Device ?? Device.Default;
        int maxPositions = options.MaxPositions ?? spec.MaxPositions;
        var random = new Random(options.Seed);
        var created = new List<Module>();
        Linear? head = null;
        try
        {
            Tensor Tensor(string name, int[] shape, Func<float[]> fallback)
            {
                var values = weights is null ? fallback() : weights.Read(name, shape) ?? throw new InvalidDataException($"The weights have no '{name}' {Idrak.Abstraction.Tensor.FormatShape(shape)}.");
                return Idrak.Abstraction.Tensor.Persistent(values, shape, device, requiresGrad: true);
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
                    : Linear.FromWeights(Idrak.Abstraction.Tensor.Persistent(Values(), shape, device, requiresGrad: true), b);
            }

            Module Normalization(string name, int features)
            {
                var gain = Tensor($"{name}.weight", [features], () => Enumerable.Repeat(spec.Norm == DecoderNorm.Rms ? 1f - spec.NormOffset : 1f, features).ToArray());
                return spec.Norm == DecoderNorm.Rms
                    ? RMSNorm.FromWeights(gain, spec.NormEpsilon, spec.NormOffset)
                    : LayerNorm.FromWeights(gain, Tensor($"{name}.bias", [features], () => new float[features]), spec.NormEpsilon);
            }

            float[]? embeddingValues = weights is null
                ? (options.InitStd is { } embedStd ? Normal(spec.Vocabulary * spec.Dim, embedStd) : [.. Enumerable.Range(0, spec.Vocabulary * spec.Dim).Select(_ => (float)(random.NextDouble() * 2 - 1) * 0.02f)])
                : weights.Read("embed.weight", [spec.Vocabulary, spec.Dim]) ?? throw new InvalidDataException($"The weights have no 'embed.weight' [{spec.Vocabulary}, {spec.Dim}].");
            // Frozen-weight builds keep the table as bfloat16 (half the memory; lossless for bfloat16 checkpoints).
            bool frozen = packed is not null || factory is not null;
            // A packed tied head gets its own transposed copy of the table: made now, so the table's float values are not
            // kept alive while every layer is read (a float tied head reads the embedding in place; see below). When that
            // copy is bfloat16, it is the only one: the embedding reads its columns (the same bfloat16 values the table
            // would hold, so lookups are unchanged bit for bit), which saves the table's memory once more. Heads in
            // other formats (int8, int4, one's own) keep a separate bfloat16 table, since the head's values differ from it.
            if (spec.TieEmbeddings && frozen)
            {
                head = Projection("head", spec.Dim, spec.Vocabulary, spec.HeadBias, embeddingValues);
            }

            var embedding = head is { BFloat16: not null } ? Embedding.FromHead(head)
                : frozen ? Embedding.FromBFloat16(BFloat16Weight.FromValues(embeddingValues, spec.Vocabulary, spec.Dim, device))
                : Embedding.FromWeights(Idrak.Abstraction.Tensor.Persistent(embeddingValues, [spec.Vocabulary, spec.Dim], device, requiresGrad: true));
            embedding.Name = "embed";
            created.Add(embedding);

            embeddingValues = null;
            if (spec.EmbeddingScale is { } scale)
            {
                created.Add(new Scale(scale) { Name = "embed_scale" });
            }

            if (spec.LearnedPositions)
            {
                created.Add(new PositionEmbedding(Tensor("pos.weight", [maxPositions, spec.Dim],
                    () => options.InitStd is { } std ? Normal(maxPositions * spec.Dim, std) : Uniform(maxPositions * spec.Dim, 0.02f))) { Name = "pos" });
            }

            if (spec.Dropout > 0f)
            {
                created.Add(new Dropout(spec.Dropout, new Random(random.Next())) { Name = "embed_drop" });
            }

            for (int i = 0; i < spec.Layers; i++)
            {
                string p = $"layers.{i}";
                var attentionNorm = Normalization($"{p}.attn_norm", spec.Dim);
                var attention = new CausalSelfAttention(
                    Projection($"{p}.attn.q", spec.Dim, spec.Heads * spec.HeadDim, spec.QkvBias),
                    Projection($"{p}.attn.k", spec.Dim, spec.KvHeads * spec.HeadDim, spec.QkvBias),
                    Projection($"{p}.attn.v", spec.Dim, spec.KvHeads * spec.HeadDim, spec.QkvBias),
                    Projection($"{p}.attn.o", spec.Heads * spec.HeadDim, spec.Dim, spec.OutputBias),
                    spec.QkNorm ? RMSNorm.FromWeights(Tensor($"{p}.attn.q_norm.weight", [spec.HeadDim], () => Enumerable.Repeat(1f - spec.NormOffset, spec.HeadDim).ToArray()), spec.NormEpsilon, spec.NormOffset) : null,
                    spec.QkNorm ? RMSNorm.FromWeights(Tensor($"{p}.attn.k_norm.weight", [spec.HeadDim], () => Enumerable.Repeat(1f - spec.NormOffset, spec.HeadDim).ToArray()), spec.NormEpsilon, spec.NormOffset) : null,
                    spec.Heads, spec.KvHeads, spec.HeadDim, spec.IsWindowed(i) ? spec.SlidingWindowRope ?? spec.Rope : spec.Rope, maxPositions)
                {
                    SlidingWindow = spec.IsWindowed(i) ? spec.SlidingWindow : null,
                    ScoreScale = spec.AttentionScale ?? 1f / MathF.Sqrt(spec.HeadDim),
                    ScoreSoftcap = spec.AttentionSoftcap,
                };
                var feedForwardNorm = spec.ParallelBlocks ? null : Normalization($"{p}.mlp_norm", spec.Dim);
                FeedForward Dense(string name, int hidden) => new(
                    spec.Gated ? Projection($"{name}.gate", spec.Dim, hidden, spec.FeedForwardBias) : null,
                    Projection($"{name}.up", spec.Dim, hidden, spec.FeedForwardBias),
                    Projection($"{name}.down", hidden, spec.Dim, spec.FeedForwardBias),
                    spec.Activation);

                // The router and the shared expert's gate stay float32 in every build: they are small, and a packed
                // router's rounding could change which experts a token goes to.
                Linear Float(string name, int outputs) => Linear.FromWeights(Tensor($"{name}.weight", [spec.Dim, outputs],
                    () => Initial(spec.Dim * outputs, MathF.Sqrt(6f / (spec.Dim + outputs)))));

                Module feedForward = spec.IsExpertLayer(i)
                    ? new MixtureOfExperts(Float($"{p}.mlp.router", spec.Experts),
                        [.. Enumerable.Range(0, spec.Experts).Select(j => Dense($"{p}.mlp.experts.{j}", spec.ExpertHidden))], spec.ExpertsPerToken, spec.NormalizeTopK,
                        spec.SharedExpertFfDim > 0 ? Dense($"{p}.mlp.shared", spec.SharedExpertFfDim) : null,
                        spec.SharedExpertFfDim > 0 ? Float($"{p}.mlp.shared_gate", 1) : null)
                    : Dense($"{p}.mlp", spec.FfDim);
                created.Add(new DecoderBlock(attentionNorm, attention, feedForwardNorm, feedForward,
                    spec.PostNorms ? Normalization($"{p}.post_attn_norm", spec.Dim) : null,
                    spec.PostNorms ? Normalization($"{p}.post_mlp_norm", spec.Dim) : null,
                    spec.Dropout > 0f ? new Dropout(spec.Dropout, new Random(random.Next())) : null) { Name = p });
            }

            var norm = Normalization("norm", spec.Dim);
            norm.Name = "norm";
            created.Add(norm);
            // A float tied head reads the embedding table in place; packed tied heads were made with the embedding.
            head ??= spec.TieEmbeddings
                ? Linear.Tied(embedding, spec.HeadBias ? Tensor("head.bias", [spec.Vocabulary], () => new float[spec.Vocabulary]) : null)
                : Projection("head", spec.Dim, spec.Vocabulary, spec.HeadBias);
            head.Name = "head";
            head.OutputSoftcap = spec.LogitSoftcap;
            created.Add(head);
            var blocks = created.OfType<DecoderBlock>().ToList();
            for (int i = 0; i < blocks.Count; i++)
            {
                blocks[i].NextNorm = (i + 1 < blocks.Count ? blocks[i + 1].AttentionNorm : norm) as RMSNorm;
            }

            var model = new Sequential(created) { Name = "decoder" };
            spec.Describe(model);
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

    private static float[] Transpose(float[] values, int rows, int columns) => HostParallel.Transpose(values, rows, columns);
}
