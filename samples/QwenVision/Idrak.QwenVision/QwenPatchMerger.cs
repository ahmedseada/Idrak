// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.QwenVision;

/// <summary>
/// Qwen-VL's patch merger (Qwen2_5_VLPatchMerger / PatchMerger): the tower's patch outputs normalized (<c>ln_q</c>:
/// RMSNorm in Qwen2.5-VL, LayerNorm in Qwen2-VL), each merge² group of neighbouring patches (consecutive in the
/// processor's order) joined into one row of merge² × tower width, then a two-layer MLP with exact GELU to the text
/// decoder's width: one embedding per image token. Trains as the family's <see cref="VisionTuningParts.Projector"/>.
/// As a module it takes [patches, width] or [images, patches, width] and gives [tokens, out] or [images, tokens, out].
/// </summary>
public sealed class QwenPatchMerger : Module
{
    private readonly Module _norm;
    private readonly Linear _fc1, _fc2;

    private QwenPatchMerger(QwenVisionConfig config, Module norm, Linear fc1, Linear fc2) => (Config, _norm, _fc1, _fc2) = (config, norm, fc1, fc2);

    /// <summary>The configuration (merge size, tower and output widths).</summary>
    public QwenVisionConfig Config { get; }

    /// <summary>The merger from the vision part's tensors (<c>merger.ln_q</c>, <c>merger.mlp.0</c>, <c>merger.mlp.2</c>), on <paramref name="device"/>.</summary>
    public static QwenPatchMerger FromTensors(QwenVisionConfig config, QwenVlVariant variant, ITensorStore tensors, Device device, EncoderWeights weights)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
        var norm = QwenNorm.Read(tensors, "merger.ln_q", variant, config.Dim, device);
        Linear? fc1 = null;
        try
        {
            fc1 = StoredWeights.Linear(tensors, "merger.mlp.0", weights, device);
            return new QwenPatchMerger(config, norm, fc1, StoredWeights.Linear(tensors, "merger.mlp.2", weights, device));
        }
        catch
        {
            norm.Dispose();
            fc1?.Dispose();
            throw;
        }
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        int unit = Config.MergeSize * Config.MergeSize, dim = Config.Dim;
        if (input.Rank is not (2 or 3) || input.Shape[^1] != dim || input.Shape[^2] % unit != 0)
        {
            throw new ArgumentException($"The merger takes [patches, {dim}] or [images, patches, {dim}] with patches a multiple of {unit}, got {Tensor.FormatShape(input.Shape)}.", nameof(input));
        }

        int rows = input.Size / dim;
        var joined = _norm.Forward(input.Reshape(rows, dim)).Reshape(rows / unit, unit * dim);
        var output = _fc2.Forward(ExactGELU.Apply(_fc1.Forward(joined)));
        return input.Rank == 2 ? output : output.Reshape(input.Shape[0], input.Shape[1] / unit, Config.OutDim);
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_norm, _fc1, _fc2];

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var child in Children())
        {
            child.Dispose();
        }

        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public override string ToString() => $"QwenPatchMerger({Config.MergeSize}² x {Config.Dim} -> {Config.OutDim})";

    // Every trainable tensor under its name in the vision part, and whether it is held transposed.
    internal Dictionary<string, (Tensor Tensor, bool Transposed)> Values()
    {
        var values = QwenNorm.Values(_norm, "merger.ln_q").ToDictionary(v => v.Name, v => v.Value, StringComparer.Ordinal);
        values["merger.mlp.0.weight"] = (_fc1.Weight, true);
        values["merger.mlp.0.bias"] = (_fc1.Bias!, false);
        values["merger.mlp.2.weight"] = (_fc2.Weight, true);
        values["merger.mlp.2.bias"] = (_fc2.Bias!, false);
        return values;
    }
}
