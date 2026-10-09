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
    /// A <see cref="Layers.Linear"/> layer y = x · W (+ b) from host values W [<paramref name="inputs"/>, <paramref name="outputs"/>]:
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

    /// <summary>
    /// A <see cref="Layers.Conv2d"/> from PyTorch's tensors <c>{name}.weight</c> [out, in / groups, kh, kw] and, when the
    /// checkpoint has it, <c>{name}.bias</c> [out]; the kernel size comes from the weight. The weight is read once, straight
    /// into the layer's [out, in / groups · kh · kw] tensor (float32: a convolution has no narrower form; a bfloat16
    /// checkpoint's values are kept exactly).
    /// </summary>
    /// <exception cref="InvalidDataException">The weight is missing or not 4-D, or the bias does not match it.</exception>
    public static Conv2d Conv2d(ITensorStore store, string name, (int Height, int Width)? stride = null, (int Height, int Width)? padding = null,
        (int Height, int Width)? dilation = null, int groups = 1, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentException.ThrowIfNullOrEmpty(name);
        var shape = ShapeOf(store, name + ".weight", 4);
        var weight = Tensor.Persistent(store.Read(name + ".weight"), [shape[0], shape[1] * shape[2] * shape[3]], device, requiresGrad: true);
        Tensor? bias = null;
        try
        {
            bias = Optional(store, name + ".bias", shape[0], device);
            return Layers.Conv2d.FromWeights(weight, bias, (shape[2], shape[3]), stride, padding, dilation, groups);
        }
        catch
        {
            weight.Dispose();
            bias?.Dispose();
            throw;
        }
    }

    /// <summary>
    /// A <see cref="Layers.BatchNorm"/> from PyTorch's <c>{name}.weight</c>, <c>.bias</c>, <c>.running_mean</c> and
    /// <c>.running_var</c> (a norm without affine parameters keeps scale 1 and shift 0; <c>num_batches_tracked</c> is not read).
    /// </summary>
    /// <exception cref="InvalidDataException">The running statistics are missing or their sizes disagree.</exception>
    public static BatchNorm BatchNorm(ITensorStore store, string name, float epsilon = 1e-5f, float momentum = 0.1f, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentException.ThrowIfNullOrEmpty(name);
        int channels = ShapeOf(store, name + ".running_mean", 1)[0];
        var norm = new BatchNorm(channels, momentum, epsilon, device);
        try
        {
            norm.RunningMean.Load(Values(store, name + ".running_mean", channels));
            norm.RunningVariance.Load(Values(store, name + ".running_var", channels));
            if (store.Contains(name + ".weight"))
            {
                norm.Gamma.Load(Values(store, name + ".weight", channels));
            }

            if (store.Contains(name + ".bias"))
            {
                norm.Beta.Load(Values(store, name + ".bias", channels));
            }

            return norm;
        }
        catch
        {
            norm.Dispose();
            throw;
        }
    }

    /// <summary>
    /// A <see cref="Layers.Linear"/> from PyTorch's <c>{name}.weight</c> [out, in] (transposed while it is read) and, when
    /// the checkpoint has it, <c>{name}.bias</c> [out]; the weight kept as bfloat16 when <see cref="KeepsBFloat16"/> says so
    /// for <paramref name="weights"/>.
    /// </summary>
    /// <exception cref="InvalidDataException">The weight is missing or not 2-D, or the bias does not match it.</exception>
    public static Linear Linear(ITensorStore store, string name, EncoderWeights weights = EncoderWeights.AsStored, Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentException.ThrowIfNullOrEmpty(name);
        var shape = ShapeOf(store, name + ".weight", 2);
        device ??= Device.Default;
        var bias = Optional(store, name + ".bias", shape[0], device);
        try
        {
            return Linear(store.ReadTransposed(name + ".weight"), shape[1], shape[0], bias, KeepsBFloat16(store, weights, name + ".weight"), device);
        }
        catch
        {
            bias?.Dispose();
            throw;
        }
    }

    // The shape of a tensor a layer needs, of the given rank.
    private static int[] ShapeOf(ITensorStore store, string tensor, int rank)
    {
        if (!store.Contains(tensor))
        {
            throw new InvalidDataException($"The checkpoint has no tensor '{tensor}'.");
        }

        var shape = store.ShapeOf(tensor);
        return shape.Length == rank ? shape
            : throw new InvalidDataException($"The checkpoint's '{tensor}' is {Tensor.FormatShape(shape)}; a {rank}-D tensor was expected.");
    }

    private static float[] Values(ITensorStore store, string tensor, int count)
    {
        var shape = ShapeOf(store, tensor, 1);
        return shape[0] == count ? store.Read(tensor)
            : throw new InvalidDataException($"The checkpoint's '{tensor}' has {shape[0]} values; {count} were expected.");
    }

    // A [count] tensor the checkpoint may leave out (a bias), trainable as the layer's own.
    private static Tensor? Optional(ITensorStore store, string tensor, int count, Device? device) =>
        store.Contains(tensor) ? Tensor.Persistent(Values(store, tensor, count), [count], device, requiresGrad: true) : null;
}
