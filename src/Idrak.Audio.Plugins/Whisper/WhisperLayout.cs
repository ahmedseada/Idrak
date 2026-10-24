// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Audio.Plugins.Whisper;

/// <summary>
/// Where each tensor of a Whisper checkpoint went when it was read (transformers' name → the layer holding it now), so a
/// trained model is written back under the same names and layouts: projections as [out, in] (the library holds
/// [in, out]), the packed query, key and value split again, the key's bias left out where the checkpoint had none (the
/// zeros the packing adds get no gradient: softmax ignores a constant added to every key). Values are read one tensor
/// at a time when written, through the layer (a layer moved to another device holds another tensor).
/// </summary>
internal sealed class WhisperLayout
{
    private readonly List<Entry> _entries = [];

    // One stored tensor: its name and shape, and how its values and its gradient are read from the layer now.
    private sealed record Entry(string Name, int[] Shape, Func<float[]> Values, Func<float[]?> Gradient);

    /// <summary>The tensors in the order they were read, each read from its layer when asked.</summary>
    public IReadOnlyList<(string Name, int[] Shape, Func<float[]> Values)> Tensors => [.. _entries.Select(e => (e.Name, e.Shape, e.Values))];

    /// <summary>Each stored tensor's gradient under its name and layout (null where the layer has none: a fixed table).</summary>
    public IEnumerable<(string Name, int[] Shape, float[]? Gradient)> Gradients() => _entries.Select(e => (e.Name, e.Shape, e.Gradient()));

    /// <summary>A tensor kept as stored (a table, a vector, a convolution's kernel as [out, in, k]).</summary>
    public void Add(string name, Func<Tensor> tensor, int[] shape) =>
        _entries.Add(new Entry(name, shape, () => tensor().ToArray(), () => tensor().Grad?.ToArray()));

    /// <summary>A projection: <c>{name}.weight</c> [out, in] and its bias when it has one.</summary>
    public void Add(string name, Linear layer)
    {
        _entries.Add(new Entry(name + ".weight", [layer.OutFeatures, layer.InFeatures],
            () => HostParallel.Transpose(((ILinearLayer)layer).WeightValues(), layer.InFeatures, layer.OutFeatures),
            () => layer.PackedWeight is null && layer.TiedTo is null && layer.Weight.Grad is { } g ? HostParallel.Transpose(g.ToArray(), layer.InFeatures, layer.OutFeatures) : null));
        if (layer.Bias is not null)
        {
            _entries.Add(new Entry(name + ".bias", [layer.OutFeatures], () => layer.Bias!.ToArray(), () => layer.Bias!.Grad?.ToArray()));
        }
    }

    /// <summary>A layer normalization: <c>{name}.weight</c> and <c>{name}.bias</c>.</summary>
    public void Add(string name, LayerNorm norm)
    {
        _entries.Add(new Entry(name + ".weight", [norm.Features], () => norm.Gamma.ToArray(), () => norm.Gamma.Grad?.ToArray()));
        _entries.Add(new Entry(name + ".bias", [norm.Features], () => norm.Beta.ToArray(), () => norm.Beta.Grad?.ToArray()));
    }

    /// <summary>
    /// The query, key and value projections packed side by side in <paramref name="packed"/> ([dim, 3 · dim]): each part's
    /// weight [dim, dim] as [out, in], and its bias where <paramref name="biases"/> says the checkpoint stored one.
    /// </summary>
    public void AddPacked(string prefix, Linear packed, int dim, ReadOnlySpan<bool> biases)
    {
        string[] parts = ["q_proj", "k_proj", "v_proj"];
        float[]? values = null;                                                // read once for the three parts, dropped after the last
        for (int p = 0; p < 3; p++)
        {
            int part = p;
            _entries.Add(new Entry(prefix + parts[p] + ".weight", [dim, dim], () =>
            {
                var weight = Part(values ??= ((ILinearLayer)packed).WeightValues(), dim, part);
                if (part == 2)
                {
                    values = null;
                }

                return weight;
            }, () => packed.PackedWeight is null && packed.Weight.Grad is { } g ? Part(g.ToArray(), dim, part) : null));
            if (biases[p])
            {
                _entries.Add(new Entry(prefix + parts[p] + ".bias", [dim], () => packed.Bias!.ToArray().AsSpan(part * dim, dim).ToArray(),
                    () => packed.Bias!.Grad?.ToArray().AsSpan(part * dim, dim).ToArray()));
            }
        }
    }

    // Part p of a packed [dim, 3 · dim] weight ([in, out]) as [out, in]: its rows' slices copied in order, then transposed.
    private static float[] Part(float[] packed, int dim, int p)
    {
        var part = GC.AllocateUninitializedArray<float>(dim * dim);
        int width = 3 * dim, offset = p * dim;
        for (int i = 0; i < dim; i++)
        {
            packed.AsSpan(i * width + offset, dim).CopyTo(part.AsSpan(i * dim, dim));
        }

        return HostParallel.Transpose(part, dim, dim);
    }
}
