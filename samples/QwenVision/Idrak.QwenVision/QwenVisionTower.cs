// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.QwenVision;

/// <summary>
/// The Qwen-VL vision transformer (Qwen2_5_VisionTransformer / Qwen2VisionTransformer): pixel patches through the
/// patch embedding (a 3D convolution over channels × temporal × patch × patch, no bias), 2D rotary positions from each
/// patch's row and column, then <see cref="QwenVisionConfig.Depth"/> blocks (attention with a q/k/v projection and an
/// output projection, a gated feed-forward), the sequence of several images of different sizes in one pass with each
/// image attending within its own span (<c>AttentionSpans</c>: the library's span-masked attention). Qwen2.5-VL attends
/// within windows of <see cref="QwenVisionConfig.WindowSize"/> pixels in every block but
/// <see cref="QwenVisionConfig.FullAttentionBlocks"/>, with RMSNorm and SwiGLU; Qwen2-VL attends over the whole image
/// in every block, with LayerNorm and QuickGELU. No block stores a score matrix: the attention runs tiled on the device
/// (plan 17, lane A). Weights as the checkpoint stores them, or float32 for training (<see cref="EncoderWeights"/>).
/// Built in plan 16, step 5 (Qwen2.5-VL) and step 6 (Qwen2-VL); the fixture of step 0 gives each stage's output.
/// </summary>
public sealed class QwenVisionTower : Module
{
    internal QwenVisionTower(QwenVisionConfig config, QwenVlVariant variant) => (Config, Variant) = (config, variant);

    /// <summary>The tower's configuration.</summary>
    public QwenVisionConfig Config { get; }

    /// <summary>Which family's tower: the blocks differ (see the class remarks).</summary>
    public QwenVlVariant Variant { get; }

    /// <summary>The tower from the checkpoint's tensors under <paramref name="prefix"/> (the <c>visual.</c> or <c>model.visual.</c> of the layout read), on <paramref name="device"/>.</summary>
    public static QwenVisionTower FromTensors(QwenVisionConfig config, QwenVlVariant variant, ITensorStore tensors, string prefix, Device device, EncoderWeights weights) =>
        throw QwenVisionPlugin.NotBuilt("the vision tower", variant == QwenVlVariant.Qwen25Vl ? 5 : 6);

    /// <summary>The patch outputs [patches, tower width] of one or several images' patches with their grids (each image attends within itself).</summary>
    public Tensor Forward(Tensor patches, IReadOnlyList<QwenImageGrid> grids) => throw QwenVisionPlugin.NotBuilt("the tower's forward pass", 5);

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => throw QwenVisionPlugin.NotBuilt("the tower's forward pass (one image)", 5);

    /// <inheritdoc />
    public override string ToString() => $"QwenVisionTower({Variant}, {Config})";
}
