// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.Audio.Plugins.Whisper;

/// <summary>The layers of a Whisper-style checkpoint read from its tensors (transformers' names), one tensor at a time.</summary>
internal static class WhisperWeights
{
    /// <summary>A 1-D tensor as a parameter on <paramref name="device"/>.</summary>
    public static Tensor Vector(ITensorStore store, string name, int length, Device device)
    {
        var shape = Shape(store, name);
        if (shape.Length != 1 || shape[0] != length)
        {
            throw new InvalidDataException($"The checkpoint's '{name}' is {Tensor.FormatShape(shape)}; [{length}] was expected.");
        }

        return Tensor.Persistent(store.Read(name), [length], device, requiresGrad: true);
    }

    /// <summary>A 2-D tensor [rows, columns] as a parameter (an embedding table, a position table).</summary>
    public static Tensor Table(ITensorStore store, string name, int rows, int columns, Device device)
    {
        var shape = Shape(store, name);
        if (shape.Length != 2 || shape[0] != rows || shape[1] != columns)
        {
            throw new InvalidDataException($"The checkpoint's '{name}' is {Tensor.FormatShape(shape)}; [{rows}, {columns}] was expected.");
        }

        return Tensor.Persistent(store.Read(name), [rows, columns], device, requiresGrad: true);
    }

    /// <summary>A layer normalization from <c>{name}.weight</c> and <c>{name}.bias</c>.</summary>
    public static LayerNorm Norm(ITensorStore store, string name, int features, float epsilon, Device device) =>
        LayerNorm.FromWeights(Vector(store, name + ".weight", features, device), Vector(store, name + ".bias", features, device), epsilon);

    /// <summary>A projection from <c>{name}.weight</c> [out, in] (and <c>{name}.bias</c> when stored), checked to be [inputs, outputs].</summary>
    public static Linear Projection(ITensorStore store, string name, int inputs, int outputs, EncoderWeights weights, Device device)
    {
        var shape = Shape(store, name + ".weight");
        if (shape.Length != 2 || shape[0] != outputs || shape[1] != inputs)
        {
            throw new InvalidDataException($"The checkpoint's '{name}.weight' is {Tensor.FormatShape(shape)}; [{outputs}, {inputs}] was expected.");
        }

        return StoredWeights.Linear(store, name, weights, device);
    }

    /// <summary>
    /// The query, key and value projections of <paramref name="prefix"/> (<c>q_proj</c>, <c>k_proj</c>, <c>v_proj</c>) side
    /// by side as one [dim, 3 · dim] layer for <see cref="MultiHeadAttention.FromWeights"/> (a key projection without a bias
    /// gets zeros).
    /// </summary>
    public static Linear PackedQkv(ITensorStore store, string prefix, int dim, EncoderWeights weights, Device device)
    {
        var packed = new float[dim * 3 * dim];
        var bias = new float[3 * dim];
        bool anyBias = false;
        string[] parts = ["q_proj", "k_proj", "v_proj"];
        for (int p = 0; p < 3; p++)
        {
            string name = prefix + parts[p] + ".weight";
            var shape = Shape(store, name);
            if (shape.Length != 2 || shape[0] != dim || shape[1] != dim)
            {
                throw new InvalidDataException($"The checkpoint's '{name}' is {Tensor.FormatShape(shape)}; [{dim}, {dim}] was expected.");
            }

            var values = store.ReadTransposed(name);                          // [in, out]
            for (int i = 0; i < dim; i++)
            {
                values.AsSpan(i * dim, dim).CopyTo(packed.AsSpan(i * 3 * dim + p * dim, dim));
            }

            string biasName = prefix + parts[p] + ".bias";
            if (store.Contains(biasName))
            {
                store.Read(biasName).AsSpan(0, dim).CopyTo(bias.AsSpan(p * dim, dim));
                anyBias = true;
            }
        }

        // The three keep one precision: the one each stored tensor allows (a bfloat16 or float16 checkpoint's own, as stored).
        var kept = StoredWeights.Kept(store, weights, parts.Select(part => prefix + part + ".weight"));
        return StoredWeights.Linear(packed, dim, 3 * dim, anyBias ? Tensor.Persistent(bias, [3 * dim], device, requiresGrad: true) : null, kept, device);
    }

    /// <summary>The feed-forward activation named by the configuration.</summary>
    public static Module Activation(string name) => name switch
    {
        "gelu" => new ExactGELU(),
        "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => new GELU(),
        "relu" => new ReLU(),
        _ => throw new NotSupportedException($"The activation '{name}' is not one this family runs (gelu, gelu_new, gelu_pytorch_tanh, relu)."),
    };

    private static int[] Shape(ITensorStore store, string name) =>
        store.Contains(name) ? store.ShapeOf(name) : throw new InvalidDataException($"The checkpoint has no tensor '{name}'.");
}
