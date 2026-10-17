// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.Audio.Layers;

/// <summary>
/// One mean-only affine coupling layer (VITS's <c>ResidualCouplingLayer</c>): the first half of the channels, through a
/// pointwise convolution, a <see cref="WaveNet"/> conditioned on the speaker and a pointwise projection, gives the shift
/// added to the second half (the log-stddev is zero, so nothing scales). <see cref="Forward"/> is the flow's direction
/// (training, and voice conversion to the prior), <see cref="Reverse"/> its inverse (speaking).
/// </summary>
public sealed class ResidualCouplingLayer : Module
{
    private readonly int _half;
    private readonly Conv1d _pre, _post;
    private readonly WaveNet _waveNet;

    /// <summary>Over <paramref name="pre"/> (half → hidden), the <paramref name="waveNet"/> (hidden) and <paramref name="post"/> (hidden → half).</summary>
    public ResidualCouplingLayer(Conv1d pre, WaveNet waveNet, Conv1d post, int half)
    {
        _pre = pre ?? throw new ArgumentNullException(nameof(pre));
        _waveNet = waveNet ?? throw new ArgumentNullException(nameof(waveNet));
        _post = post ?? throw new ArgumentNullException(nameof(post));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(half);
        _half = half;
    }

    /// <summary>The channels each half holds.</summary>
    public int Half => _half;

    /// <summary>From stored weights under <paramref name="prefix"/>: <c>conv_pre</c>, <c>wavenet</c> (<see cref="WaveNet.FromStored"/>) and <c>conv_post</c>.</summary>
    public static ResidualCouplingLayer FromStored(StoredLayers stored, string prefix, int channels, int hidden, int waveNetLayers, int kernel, int dilationRate, int conditioningSize)
    {
        ArgumentNullException.ThrowIfNull(stored);
        int half = channels / 2;
        var pre = stored.Conv($"{prefix}.conv_pre", half, hidden, 1, 0);
        var waveNet = WaveNet.FromStored(stored, $"{prefix}.wavenet", hidden, waveNetLayers, kernel, dilationRate, conditioningSize);
        var post = stored.Conv($"{prefix}.conv_post", hidden, half, 1, 0);
        return new ResidualCouplingLayer(pre, waveNet, post, half);
    }

    /// <summary>The inverse on latents [2 · half, T] with the speaker's embedding [E, 1] (null for none): the shift subtracted from the second half.</summary>
    public Tensor Reverse(Tensor z, Tensor? speaker)
    {
        int t = z.Shape[1];
        var first = z.Narrow(0, 0, _half);
        var second = z.Narrow(0, _half, _half);
        return Tensor.Concat([first, second - Mean(first.Reshape(1, _half, t), speaker, null).Reshape(_half, t)], 0);
    }

    /// <summary>
    /// The flow's direction on a batch [B, 2 · half, T] (zero over the padding) with the speaker's embedding (one sequence
    /// only) and the batch's masks: the shift added to the second half, zeroed over the padding. A block of the model's
    /// own checkpoints (<see cref="Checkpointing.RunBlock"/>).
    /// </summary>
    public Tensor Forward(Tensor z, Tensor? speaker, PaddedSequences? mask)
    {
        var first = z.Narrow(1, 0, _half);
        var second = z.Narrow(1, _half, _half);
        var moved = second + Checkpointing.RunBlock(x => Mean(x, speaker, mask), first);
        return Tensor.Concat([first, mask is null ? moved : mask.Apply(moved)], 1);
    }

    // The shift [B, half, T] the first half conditions, zero over the padding.
    private Tensor Mean(Tensor first, Tensor? speaker, PaddedSequences? mask)
    {
        var h = _pre.Forward(first);
        if (mask is not null)
        {
            h = mask.Apply(h);
        }

        var shift = _post.Forward(_waveNet.Forward(h, speaker, mask));
        return mask is null ? shift : mask.Apply(shift);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Forward(input, null, null);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_pre, _waveNet, _post];

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        _pre.Dispose();
        _post.Dispose();
        _waveNet.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"ResidualCouplingLayer({_half} + {_half} channels)";
}

/// <summary>
/// A stack of <see cref="ResidualCouplingLayer"/>s with the channels flipped between them (VITS's
/// <c>ResidualCouplingBlock</c>): <see cref="Forward(Tensor, Tensor?, PaddedSequences?)"/> takes acoustic latents to the prior's (training, voice
/// conversion), <see cref="Reverse"/> the prior's to the acoustic ones (speaking).
/// </summary>
public sealed class ResidualCouplingBlock : Module
{
    private readonly ResidualCouplingLayer[] _layers;

    /// <summary>Over the layers, in the flow's order.</summary>
    public ResidualCouplingBlock(IReadOnlyList<ResidualCouplingLayer> layers)
    {
        ArgumentNullException.ThrowIfNull(layers);
        _layers = [.. layers];
    }

    /// <summary>The coupling layers.</summary>
    public IReadOnlyList<ResidualCouplingLayer> Layers => _layers;

    /// <summary>From stored weights under <paramref name="prefix"/>: <c>flows.{i}</c> for <paramref name="flows"/> layers (<see cref="ResidualCouplingLayer.FromStored"/>).</summary>
    public static ResidualCouplingBlock FromStored(StoredLayers stored, string prefix, int flows, int channels, int hidden, int waveNetLayers, int kernel, int dilationRate, int conditioningSize)
    {
        var layers = new ResidualCouplingLayer[flows];
        for (int i = 0; i < flows; i++)
        {
            layers[i] = ResidualCouplingLayer.FromStored(stored, $"{prefix}.flows.{i}", channels, hidden, waveNetLayers, kernel, dilationRate, conditioningSize);
        }

        return new ResidualCouplingBlock(layers);
    }

    /// <summary>The acoustic latents [F, T] of the prior latents [F, T], with the speaker's embedding [E, 1] (null for none).</summary>
    public Tensor Reverse(Tensor z, Tensor? speaker)
    {
        for (int i = _layers.Length - 1; i >= 0; i--)
        {
            z = _layers[i].Reverse(z.Flip(z.Shape.Length == 2 ? 0 : 1), speaker);
        }

        return z;
    }

    /// <summary>The prior latents [F, T] of the acoustic latents [F, T] (the inverse of <see cref="Reverse"/>), with the speaker's embedding [E, 1].</summary>
    public Tensor Forward(Tensor z, Tensor? speaker) => Forward(z.Reshape(1, z.Shape[0], z.Shape[1]), speaker, null).Reshape(z.Shape[0], z.Shape[1]);

    /// <summary>
    /// A batch of acoustic latents [B, F, T] (zero over the padding) to the prior's, with the speaker's embedding (one
    /// sequence only) and the batch's masks: each layer's shift zeroed over the padding.
    /// </summary>
    public Tensor Forward(Tensor z, Tensor? speaker, PaddedSequences? mask)
    {
        for (int i = 0; i < _layers.Length; i++)
        {
            z = _layers[i].Forward(z, speaker, mask).Flip(1);
        }

        return z;
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => input.Shape.Length == 2 ? Forward(input, null) : Forward(input, null, null);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => _layers;

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var layer in _layers) layer.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"ResidualCouplingBlock({_layers.Length} layers)";
}
