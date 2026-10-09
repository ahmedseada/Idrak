// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;

namespace Idrak.Layers;

// Inference fusion: a convolution followed by a batch norm in evaluation mode and/or an activation the convolution applies
// in its own pass (ReLU, sigmoid, tanh, GELU) runs as one convolution, the norm folded into its weights and bias:
// scale = γ / √(running variance + ε), weight' = weight · scale (per filter), bias' = (bias - running mean) · scale + β.
// Sequential runs its layers this way when nothing is recorded for a gradient; the result equals the layers run one by one
// up to float rounding.
public sealed partial class Conv2d
{
    // The activation a module applies, when the convolution can apply it in its pass; null otherwise.
    internal static ConvActivation? FusedActivation(Module module) => module switch
    {
        ReLU => ConvActivation.Relu,
        Sigmoid => ConvActivation.Sigmoid,
        Tanh => ConvActivation.Tanh,
        GELU => ConvActivation.Gelu,
        _ => null,
    };

    /// <summary>
    /// The steps of <paramref name="modules"/>' first <paramref name="count"/> layers with each convolution fused with a
    /// following batch norm in evaluation mode and activation (each step runs from its first layer and covers its count),
    /// or null when no convolution has anything to fuse.
    /// </summary>
    internal static (int First, int Count)[]? FusedSteps(IReadOnlyList<Module> modules, int count)
    {
        List<(int, int)>? steps = null;
        for (int i = 0; i < count; i++)
        {
            int span = 1;
            if (modules[i] is Conv2d conv)
            {
                if (i + span < count && modules[i + span] is BatchNorm { IsTraining: false } norm && norm.Channels == conv.OutChannels)
                {
                    span++;
                }

                if (i + span < count && FusedActivation(modules[i + span]) is not null)
                {
                    span++;
                }
            }

            if (span > 1 && steps is null)
            {
                steps = [];
                for (int j = 0; j < i; j++)
                {
                    steps.Add((j, 1));
                }
            }

            steps?.Add((i, span));
            i += span - 1;
        }

        return steps?.ToArray();
    }

    /// <summary>
    /// This convolution with <paramref name="norm"/> (evaluation mode; null for none) folded into its weights and bias and
    /// <paramref name="activation"/> applied in the same pass. Records nothing for a gradient.
    /// </summary>
    internal Tensor ForwardFused(Tensor input, BatchNorm? norm, ConvActivation activation)
    {
        if (input.Rank != 4 || input.Shape[1] != InChannels)
        {
            throw new ArgumentException($"Conv2d expects [N, {InChannels}, H, W], got {Tensor.FormatShape(input.Shape)}.");
        }

        var g = Geometry(input.Shape[0], input.Shape[2], input.Shape[3]);
        if (g.OH <= 0 || g.OW <= 0)
        {
            throw new ArgumentException($"A {KernelHeight}x{KernelWidth} kernel (dilation {DilationHeight}x{DilationWidth}) does not fit a {g.H}x{g.W} input with padding {PaddingHeight}x{PaddingWidth}.");
        }

        using var offload = input.Device.Backend.Offload is not null ? TensorOffloading.EnterForward?.Invoke(this, input) : null;
        using var normOffload = norm is not null && input.Device.Backend.Offload is not null ? TensorOffloading.EnterForward?.Invoke(norm, input) : null;
        if (norm is null)
        {
            return input.Convolution(Weight, Bias, g, Groups, activation);
        }

        int filters = OutChannels;
        var backend = Weight.Backend;
        using var scale = Tensor.Empty([filters], input.Device, track: false);
        using var weight = Tensor.Empty(Weight.Shape, input.Device, track: false);
        using var bias = Tensor.Empty([filters], input.Device, track: false);
        backend.InvSqrt(norm.RunningVariance.Storage, scale.Storage, filters, norm.Epsilon);
        backend.Binary(BinaryOp.Mul, norm.Gamma.Storage, scale.Storage, scale.Storage, filters);
        backend.GroupScaleShift(Weight.Storage, scale.Storage, null, weight.Storage, Weight.Size, filters, Weight.Size / filters, accumulate: false);
        if (Bias is null)
        {
            backend.Affine(norm.RunningMean.Storage, bias.Storage, filters, -1f, 0f);
        }
        else
        {
            backend.Binary(BinaryOp.Sub, Bias.Storage, norm.RunningMean.Storage, bias.Storage, filters);
        }

        backend.Binary(BinaryOp.Mul, bias.Storage, scale.Storage, bias.Storage, filters);
        backend.Binary(BinaryOp.Add, bias.Storage, norm.Beta.Storage, bias.Storage, filters);
        return input.Convolution(weight, bias, g, Groups, activation);
    }
}
