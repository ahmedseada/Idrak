// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models;

namespace Idrak.Layers;

/// <summary>
/// A transformer feed-forward of two convolutions over time (transformers' <c>VitsFeedForward</c>, the Glow-TTS and VITS
/// text encoders'): rows [B · T, C] of a batch of padded sequences become [B, C, T], go through a convolution of
/// <see cref="Kernel"/> taps to the feed-forward width, the activation, and a second convolution back, and return as
/// rows. An even kernel pads asymmetrically (k / 2 before, (k − 1) / 2 after) as transformers does; the convolutions read a
/// batch's padding as zeros (the mask applied before each).
/// </summary>
public sealed class ConvolutionalFeedForward : Module
{
    private readonly Conv1d _conv1, _conv2;
    private readonly Func<Tensor, Tensor> _activation;

    /// <summary>Over <paramref name="conv1"/> (C → F) and <paramref name="conv2"/> (F → C) of <paramref name="kernel"/> taps each, with <paramref name="activation"/> between them.</summary>
    public ConvolutionalFeedForward(Conv1d conv1, Conv1d conv2, Func<Tensor, Tensor> activation, int kernel)
    {
        _conv1 = conv1 ?? throw new ArgumentNullException(nameof(conv1));
        _conv2 = conv2 ?? throw new ArgumentNullException(nameof(conv2));
        _activation = activation ?? throw new ArgumentNullException(nameof(activation));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kernel);
        Kernel = kernel;
    }

    /// <summary>The convolutions' taps.</summary>
    public int Kernel { get; }

    /// <summary>The names <see cref="Activation"/> takes.</summary>
    public static IReadOnlyList<string> Activations { get; } = ["relu", "gelu", "gelu_new", "gelu_pytorch_tanh", "gelu_fast", "silu", "swish", "quick_gelu"];

    /// <summary>The activation a configuration names (<c>hidden_act</c>): relu, gelu (exact), gelu_new / gelu_pytorch_tanh / gelu_fast (the tanh form), silu / swish, quick_gelu.</summary>
    /// <exception cref="NotSupportedException">The name is not one of <see cref="Activations"/>.</exception>
    public static Func<Tensor, Tensor> Activation(string name) => name switch
    {
        "relu" => x => x.Relu(),
        "gelu" => x => x.GeluErf(),
        "gelu_new" or "gelu_pytorch_tanh" or "gelu_fast" => x => x.Gelu(),
        "silu" or "swish" => x => x.Silu(),
        "quick_gelu" => x => x * (x * 1.702f).Sigmoid(),
        _ => throw new NotSupportedException($"hidden_act '{name}' is not supported ({string.Join(", ", Activations)})."),
    };

    /// <summary>
    /// From stored weights under <paramref name="prefix"/>: <c>conv_1</c> (<paramref name="hidden"/> → <paramref name="ffDim"/>) and
    /// <c>conv_2</c> (back), transformers' names, of <paramref name="kernel"/> taps (an odd kernel padded symmetrically by the
    /// convolution, an even one by the pass), with the activation <paramref name="activation"/> names.
    /// </summary>
    public static ConvolutionalFeedForward FromStored(StoredLayers stored, string prefix, int hidden, int ffDim, int kernel, string activation)
    {
        ArgumentNullException.ThrowIfNull(stored);
        int symmetric = kernel % 2 == 1 ? kernel / 2 : 0;
        var conv1 = stored.Conv(StoredLayers.Join(prefix, "conv_1"), hidden, ffDim, kernel, symmetric);
        var conv2 = stored.Conv(StoredLayers.Join(prefix, "conv_2"), ffDim, hidden, kernel, symmetric);
        return new ConvolutionalFeedForward(conv1, conv2, Activation(activation), kernel);
    }

    /// <summary>
    /// The rows x [B · T, C] of <paramref name="batch"/> sequences padded to <paramref name="t"/> positions (the mask null for
    /// one, unpadded) through the feed-forward: [B · T, C], the convolutions reading the padding as zeros.
    /// </summary>
    public Tensor Forward(Tensor x, int batch, int t, PaddedSequences? mask)
    {
        ArgumentNullException.ThrowIfNull(x);
        int c = x.Shape[^1], k = Kernel, left = k % 2 == 1 ? 0 : k / 2, right = k % 2 == 1 ? 0 : (k - 1) / 2;   // even kernels pad asymmetrically
        var y = x.Reshape(batch, t, c).Permute(0, 2, 1);                       // [B, C, T]
        if (mask is not null)
        {
            y = mask.Apply(y);
        }

        var h = _activation(_conv1.Forward(left + right == 0 ? y : y.Pad(-1, left, right)));
        if (mask is not null)
        {
            h = mask.Apply(h);
        }

        return _conv2.Forward(left + right == 0 ? h : h.Pad(-1, left, right)).Permute(0, 2, 1).Reshape(batch * t, c);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Forward(input, 1, input.Shape[0], null);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_conv1, _conv2];

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        _conv1.Dispose();
        _conv2.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"ConvolutionalFeedForward(k{Kernel})";
}
