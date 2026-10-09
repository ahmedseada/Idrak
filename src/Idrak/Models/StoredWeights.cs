// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// Building layers from a checkpoint's tensors in the precision they are stored in (family-neutral: any encoder a vision
/// family builds from an <see cref="ITensorStore"/>). A bfloat16 checkpoint's projection weights can stay bfloat16: the
/// values are the same as float32 copies of them, and the products read them into float32 sums, so the layer computes
/// what a float32 layer holding those values computes, in half the memory.
/// </summary>
public static class StoredWeights
{
    /// <summary>
    /// Whether projection weights read from <paramref name="names"/> are kept as bfloat16 under
    /// <paramref name="weights"/>: <see cref="EncoderWeights.BFloat16"/> always, <see cref="EncoderWeights.AsStored"/>
    /// when every one of them is stored as bfloat16 (<see cref="ITensorStore.FormatOf"/>), <see cref="EncoderWeights.Float32"/> never.
    /// </summary>
    /// <param name="store">The checkpoint's tensors.</param>
    /// <param name="weights">What the caller asked for.</param>
    /// <param name="names">The stored tensors the layer's weight is made of (several when they are joined, as q, k and v).</param>
    public static bool KeepsBFloat16(ITensorStore store, EncoderWeights weights, params IEnumerable<string> names)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(names);
        return weights switch
        {
            EncoderWeights.BFloat16 => true,
            EncoderWeights.AsStored => names.All(n => store.FormatOf(n) == WeightFormat.BFloat16),
            _ => false,
        };
    }

    /// <summary>
    /// A <see cref="Linear"/> layer y = x · W (+ b) from host values W [<paramref name="inputs"/>, <paramref name="outputs"/>]:
    /// bfloat16 weights (fixed) when <paramref name="bfloat16"/>, else float32 weights that train. The bias stays as given.
    /// </summary>
    /// <param name="values">W, row-major [inputs, outputs].</param>
    /// <param name="inputs">Input features.</param>
    /// <param name="outputs">Output features.</param>
    /// <param name="bias">The bias [outputs], or null; the layer takes ownership.</param>
    /// <param name="bfloat16">Keep W as bfloat16 (rounded to nearest; exact for values a bfloat16 checkpoint stores).</param>
    /// <param name="device">Where the layer lives.</param>
    public static Linear Linear(float[] values, int inputs, int outputs, Tensor? bias, bool bfloat16, Device device)
    {
        ArgumentNullException.ThrowIfNull(values);
        ArgumentNullException.ThrowIfNull(device);
        if (values.Length != inputs * outputs)
        {
            throw new ArgumentException($"{values.Length} values do not fill [{inputs}, {outputs}].", nameof(values));
        }

        return bfloat16
            ? Layers.Linear.FromBFloat16(BFloat16Weight.FromValues(values, inputs, outputs, device), bias)
            : Layers.Linear.FromWeights(Tensor.Persistent(values, [inputs, outputs], device, requiresGrad: true), bias);
    }
}
