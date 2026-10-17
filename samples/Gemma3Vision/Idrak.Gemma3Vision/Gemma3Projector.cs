// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Vision;

namespace Idrak.Gemma3Vision;

/// <summary>
/// Gemma 3's multimodal projector: the encoder's [images, side², visionDim] patch outputs (row-major) are average-pooled
/// over the side x side grid with kernel = stride = <see cref="PoolSize"/>, giving (side / PoolSize)² tokens per image
/// (row-major), each RMS-normalized with a gain of 1 + w (Gemma's RMSNorm, in float32), then projected to the text
/// decoder's width as x · W with W stored [visionDim, textDim] (read as stored, no transposition). The output is the
/// soft tokens' embeddings, not scaled by √textDim.
/// </summary>
public sealed class Gemma3Projector : Module
{
    private readonly RMSNorm _norm;
    private readonly Linear _projection;

    private Gemma3Projector(RMSNorm norm, Linear projection, int poolSize)
    {
        _norm = norm;
        _projection = projection;
        PoolSize = poolSize;
    }

    /// <summary>The pooling's kernel and stride over the grid of patches.</summary>
    public int PoolSize { get; }

    /// <summary>The encoder's width (the input's last dimension).</summary>
    public int VisionDim => _projection.InFeatures;

    /// <summary>The text decoder's width (the output's last dimension).</summary>
    public int TextDim => _projection.OutFeatures;

    /// <summary>
    /// Builds the projector from Gemma 3's tensors (<c>mm_soft_emb_norm.weight</c> [visionDim] and
    /// <c>mm_input_projection_weight</c> [visionDim, textDim]) under <paramref name="prefix"/>, as
    /// <see cref="Gemma3Vision.OpenTensors"/> gives them with the prefix <c>projector.</c>. The projection keeps the
    /// precision <paramref name="weights"/> asks for (by default as stored, <see cref="StoredWeights"/>); the norm is float32.
    /// </summary>
    public static Gemma3Projector FromTensors(ITensorStore tensors, int visionDim, int textDim, int poolSize, float normEpsilon, string prefix = "projector.", Device? device = null,
        EncoderWeights weights = EncoderWeights.AsStored)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(poolSize);
        device ??= Device.Default;
        string normName = $"{prefix}mm_soft_emb_norm.weight", projectionName = $"{prefix}mm_input_projection_weight";
        if (!tensors.ShapeOf(normName).SequenceEqual([visionDim]) || !tensors.ShapeOf(projectionName).SequenceEqual([visionDim, textDim]))
        {
            throw new InvalidDataException($"{normName} {Tensor.FormatShape(tensors.ShapeOf(normName))} and {projectionName} {Tensor.FormatShape(tensors.ShapeOf(projectionName))}: "
                + $"expected [{visionDim}] and [{visionDim}, {textDim}].");
        }

        var gain = Tensor.Persistent(tensors.Read(normName), [visionDim], device, requiresGrad: true);
        var projection = StoredWeights.Linear(tensors.Read(projectionName), visionDim, textDim, null,              // x · W as stored
            StoredWeights.KeepsBFloat16(tensors, weights, projectionName), device);
        return new Gemma3Projector(RMSNorm.FromWeights(gain, normEpsilon, offset: 1f), projection, poolSize);
    }

    /// <summary>[images, side², visionDim] to [images, (side / PoolSize)², textDim].</summary>
    protected override Tensor ForwardCore(Tensor input)
    {
        int side = (int)Math.Round(Math.Sqrt(input.Rank == 3 ? input.Shape[1] : 0));
        if (input.Rank != 3 || input.Shape[2] != VisionDim || side * side != input.Shape[1] || side % PoolSize != 0)
        {
            throw new ArgumentException($"The projector takes [images, side², {VisionDim}] with a side divisible by {PoolSize}, got {Tensor.FormatShape(input.Shape)}.");
        }

        return _projection.Forward(_norm.Forward(PatchGrid.AveragePool(input, PoolSize)));
    }

    // The norm's gain (w of 1 + w) and the projection's weight [visionDim, textDim], as stored.
    internal Tensor Gain => _norm.Gain;

    internal Tensor Weight => _projection.Weight;

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_norm, _projection];

    /// <inheritdoc />
    public override string ToString() => $"Gemma3Projector(pool {PoolSize}, {VisionDim} -> {TextDim})";
}
