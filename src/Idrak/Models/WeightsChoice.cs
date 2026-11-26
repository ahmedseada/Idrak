// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// What a weight placement and an expert placement of <see cref="PretrainedModel.Load"/> are asked, counted from the
/// model's spec before any weight is read: the float weights a part takes, the working memory beside them (the key/value
/// caches, a step's scores, the headroom). Card-agnostic: every figure is measured or read from the model.
/// </summary>
public static class AutoWeights
{
    /// <summary>The context the caches are counted for when neither the options nor the load give one: 4096 tokens.</summary>
    public const int DefaultContext = 4096;

    /// <summary>The share of the weights' bytes kept free beside the working memory when the options give none: 0.1.</summary>
    public const double DefaultHeadroom = 0.1;

    /// <summary>The bytes <paramref name="spec"/>'s float weights take.</summary>
    public static long WeightBytes(DecoderSpec spec)
    {
        ArgumentNullException.ThrowIfNull(spec);
        return spec.ParameterCount * sizeof(float);
    }

    /// <summary>
    /// The working memory beside the weights: the key/value caches of <paramref name="context"/> tokens (every layer,
    /// float32), a step's scores over the vocabulary, and <paramref name="headroom"/> of <paramref name="weightBytes"/>.
    /// </summary>
    public static long WorkingBytes(DecoderSpec spec, int context, long weightBytes, double headroom)
    {
        ArgumentNullException.ThrowIfNull(spec);
        long caches = 2L * spec.Layers * spec.KvHeads * spec.HeadDim * Math.Max(1, context) * sizeof(float);
        long scores = 2L * spec.Vocabulary * sizeof(float);
        return caches + scores + (long)(Math.Max(0, headroom) * weightBytes);
    }

    /// <summary>
    /// What a weight placement is asked for <paramref name="spec"/> on <paramref name="device"/>
    /// (<see cref="WeightPlacementRequest"/>), counted from the spec before any weight is read: one decoder layer's float
    /// weights, the weights that stay on the device (token table, head, final norm), one layer's float32 key/value cache for
    /// <paramref name="context"/> tokens, and a step's scores with <paramref name="headroom"/> of the weights;
    /// <paramref name="available"/> is the device's free memory (or a budget) and <paramref name="deviceLayers"/> the user's
    /// count, if any.
    /// </summary>
    public static WeightPlacementRequest PlacementRequest(DecoderSpec spec, int context, Device device, long? available, int? deviceLayers,
        double headroom = DefaultHeadroom)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(device);
        long table = (long)spec.Vocabulary * spec.Dim;
        long layerParameters = Math.Max(0, spec.ParameterCount - (spec.TieEmbeddings ? table : 2 * table) - spec.Dim) / Math.Max(1, spec.Layers);
        long layerBytes = layerParameters * sizeof(float);
        long weights = WeightBytes(spec);
        long fixedBytes = Math.Max(0, weights - layerBytes * spec.Layers);
        long layerCache = 2L * spec.KvHeads * spec.HeadDim * Math.Max(1, context) * sizeof(float);
        long working = 2L * spec.Vocabulary * sizeof(float) + (long)(Math.Max(0, headroom) * weights);
        return new WeightPlacementRequest(device, spec.Layers, layerBytes, fixedBytes, layerCache, working, available, deviceLayers, device.Backend.Offload is not null);
    }

    /// <summary>
    /// What an expert placement is asked for <paramref name="spec"/>'s experts on <paramref name="device"/>
    /// (<see cref="ExpertPlacementRequest"/>, plan 37 R1), counted from the spec before any weight is read:
    /// - one expert's float projections;
    /// - everything else the run holds on the device: the other weights, every layer's float32 key/value cache for
    ///   <paramref name="context"/> tokens, a step's scores and <paramref name="headroom"/> of the weights.
    /// <paramref name="available"/> is the device's free memory (or a budget), <paramref name="cacheBytes"/> the user's
    /// limit on the experts kept there, if any.
    /// </summary>
    public static ExpertPlacementRequest ExpertRequest(DecoderSpec spec, int context, Device device, long? available, long? cacheBytes, bool coldOnCpu,
        double headroom = DefaultHeadroom)
    {
        ArgumentNullException.ThrowIfNull(spec);
        ArgumentNullException.ThrowIfNull(device);
        int expertLayers = Enumerable.Range(0, spec.Layers).Count(spec.IsExpertLayer);
        long expertBytes = (long)spec.Dim * spec.ExpertHidden * (spec.Gated ? 3 : 2) * sizeof(float);
        long weights = WeightBytes(spec);
        long other = Math.Max(0, weights - (long)expertLayers * spec.Experts * expertBytes) + WorkingBytes(spec, context, weights, headroom);
        return new ExpertPlacementRequest(device, expertLayers, spec.Experts, spec.ExpertsPerToken, expertBytes, other, available, cacheBytes, coldOnCpu);
    }

    /// <summary>Bytes as gigabytes with one decimal ("8.1 GB", invariant culture).</summary>
    public static string Gigabytes(long bytes) => (bytes / 1e9).ToString("F1", CultureInfo.InvariantCulture) + " GB";
}
