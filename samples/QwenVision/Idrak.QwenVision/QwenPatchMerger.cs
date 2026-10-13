// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.QwenVision;

/// <summary>
/// Qwen-VL's patch merger (Qwen2_5_VLPatchMerger / PatchMerger): the tower's patch outputs normalized (RMSNorm in
/// Qwen2.5-VL, LayerNorm in Qwen2-VL), each <see cref="QwenVisionConfig.MergeSize"/>² group of neighbouring patches
/// concatenated into one row of merge² × tower width, then a two-layer MLP with GELU to the text decoder's width: one
/// embedding per image token, [tokens, text width]. The patches arrive in the tower's order, which already groups each
/// merge window's patches together, so the concatenation is a reshape. It trains as the family's
/// <see cref="VisionTuningParts.Projector"/>. Built in plan 16, step 5 (and step 6 for Qwen2-VL's norm); the fixture
/// of step 0 gives its output.
/// </summary>
public sealed class QwenPatchMerger : Module
{
    internal QwenPatchMerger(QwenVisionConfig config) => Config = config;

    /// <summary>The configuration (merge size, tower and output widths).</summary>
    public QwenVisionConfig Config { get; }

    /// <summary>The merger from the checkpoint's tensors under <paramref name="prefix"/> (the tower's prefix plus <c>merger.</c>), on <paramref name="device"/>.</summary>
    public static QwenPatchMerger FromTensors(QwenVisionConfig config, QwenVlVariant variant, ITensorStore tensors, string prefix, Device device, EncoderWeights weights) =>
        throw QwenVisionPlugin.NotBuilt("the patch merger", variant == QwenVlVariant.Qwen25Vl ? 5 : 6);

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => throw QwenVisionPlugin.NotBuilt("the merger's forward pass", 5);

    /// <inheritdoc />
    public override string ToString() => $"QwenPatchMerger({Config.MergeSize}² × {Config.Dim} → {Config.OutDim})";
}
