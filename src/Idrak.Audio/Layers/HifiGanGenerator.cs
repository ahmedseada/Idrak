// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.Audio.Layers;

/// <summary>
/// One of HiFi-GAN's residual blocks: pairs of a dilated and a plain convolution, each after a leaky ReLU, added to the
/// input. With gradients off, each pair runs in a scope that keeps only its result.
/// </summary>
public sealed class HifiGanResidualBlock : Module
{
    private readonly float _slope;
    private readonly Conv1d[] _convs1, _convs2;

    /// <summary>Over the dilated convolutions <paramref name="convs1"/> and the plain ones <paramref name="convs2"/>, paired, with the leaky ReLU's <paramref name="slope"/>.</summary>
    public HifiGanResidualBlock(IReadOnlyList<Conv1d> convs1, IReadOnlyList<Conv1d> convs2, float slope)
    {
        ArgumentNullException.ThrowIfNull(convs1);
        ArgumentNullException.ThrowIfNull(convs2);
        if (convs1.Count == 0 || convs2.Count != convs1.Count)
        {
            throw new ArgumentException($"A residual block's convolutions pair up: {convs1.Count} dilated, {convs2.Count} plain.", nameof(convs2));
        }

        _convs1 = [.. convs1];
        _convs2 = [.. convs2];
        _slope = slope;
    }

    /// <summary>From stored weights under <paramref name="prefix"/>: <c>convs1.{i}</c> dilated by <paramref name="dilations"/>[i] and <c>convs2.{i}</c>, of <paramref name="kernel"/> taps over <paramref name="channels"/>.</summary>
    public static HifiGanResidualBlock FromStored(StoredLayers stored, string prefix, int channels, int kernel, IReadOnlyList<int> dilations, float slope)
    {
        ArgumentNullException.ThrowIfNull(stored);
        var convs1 = new Conv1d[dilations.Count];
        var convs2 = new Conv1d[dilations.Count];
        for (int i = 0; i < dilations.Count; i++)
        {
            int dilation = dilations[i];
            convs1[i] = stored.Conv(StoredLayers.Join(prefix, $"convs1.{i}"), channels, channels, kernel, (kernel * dilation - dilation) / 2, dilation);
            convs2[i] = stored.Conv(StoredLayers.Join(prefix, $"convs2.{i}"), channels, channels, kernel, (kernel - 1) / 2);
        }

        return new HifiGanResidualBlock(convs1, convs2, slope);
    }

    /// <summary>x [B, C, T] through the pairs. With <paramref name="release"/>, each pair in its own scope keeping only the new x, and an x a previous pair made released once replaced.</summary>
    public Tensor Forward(Tensor x, bool release)
    {
        var input = x;
        for (int i = 0; i < _convs1.Length; i++)
        {
            Tensor next;
            using (var step = release ? new TensorScope() : null)
            {
                var h = _convs1[i].Forward(AudioLayerOps.LeakyRelu(x, _slope));
                h = _convs2[i].Forward(AudioLayerOps.LeakyRelu(h, _slope));
                next = h + x;
                step?.Keep(next);
            }

            if (release && !ReferenceEquals(x, input))
            {
                TensorScope.Untrack(x).Dispose();
            }

            x = next;
        }

        return x;
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Forward(input, release: !Autograd.IsEnabled);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [.. _convs1, .. _convs2];

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var c in _convs1) c.Dispose();
        foreach (var c in _convs2) c.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"HifiGanResidualBlock({_convs1.Length} pairs)";
}

/// <summary>
/// HiFi-GAN's generator (VITS's decoder): a convolution into the first channel count, then per stage a leaky ReLU, a
/// transposed convolution (the rate's upsampling) and the stage's residual blocks averaged; a last leaky ReLU, the output
/// convolution and tanh. One latents [F, T] (with a speaker's embedding) or a batch [B, F, T] (training: every sequence's
/// segment through each layer in one launch). <see cref="Samples"/> decodes long latents in pieces with context, so the
/// device holds one piece's tensors whatever the length.
/// </summary>
public sealed class HifiGanGenerator : Module
{
    private readonly Conv1d _pre, _post;
    private readonly Conv1d? _cond;
    private readonly ConvTranspose1d[] _upsamplers;
    private readonly HifiGanResidualBlock[] _blocks;
    private readonly int _kernels;
    private readonly float _slope, _finalSlope;

    /// <summary>
    /// Over the layers: <paramref name="pre"/>, the speaker <paramref name="conditioning"/> (or null), the stages'
    /// <paramref name="upsamplers"/>, their residual <paramref name="blocks"/> (<paramref name="kernelsPerStage"/> a stage, in
    /// stage order), <paramref name="post"/>; the leaky ReLU slopes of the stages and the last one; the <paramref name="hop"/>
    /// (samples a latent frame, the upsampling rates' product); the <paramref name="contextFrames"/> a piece needs on each
    /// side (<see cref="ContextFramesFor"/>).
    /// </summary>
    public HifiGanGenerator(Conv1d pre, Conv1d? conditioning, IReadOnlyList<ConvTranspose1d> upsamplers, IReadOnlyList<HifiGanResidualBlock> blocks, Conv1d post,
        int kernelsPerStage, float slope, float finalSlope, int hop, int contextFrames)
    {
        _pre = pre ?? throw new ArgumentNullException(nameof(pre));
        _post = post ?? throw new ArgumentNullException(nameof(post));
        ArgumentNullException.ThrowIfNull(upsamplers);
        ArgumentNullException.ThrowIfNull(blocks);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(kernelsPerStage);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hop);
        if (blocks.Count != upsamplers.Count * kernelsPerStage)
        {
            throw new ArgumentException($"{upsamplers.Count} stages of {kernelsPerStage} residual blocks need {upsamplers.Count * kernelsPerStage} blocks, not {blocks.Count}.", nameof(blocks));
        }

        _cond = conditioning;
        _upsamplers = [.. upsamplers];
        _blocks = [.. blocks];
        _kernels = kernelsPerStage;
        _slope = slope;
        _finalSlope = finalSlope;
        Hop = hop;
        ContextFrames = contextFrames;
    }

    /// <summary>The samples a latent frame makes.</summary>
    public int Hop { get; }

    /// <summary>
    /// The latent frames on each side that an output sample depends on (the receptive field of the convolutions and the
    /// transposed convolutions, in frames, rounded up, plus two): a piece decoded with this much context on both sides
    /// gives the samples a whole pass gives.
    /// </summary>
    public int ContextFrames { get; }

    /// <summary>
    /// From stored weights under <paramref name="prefix"/> (transformers' VITS names): <c>conv_pre</c> (<paramref name="flowSize"/>
    /// → <paramref name="initialChannels"/>), <c>cond</c> when <paramref name="conditioningSize"/> is above 0, <c>upsampler.{i}</c>
    /// (the channels halving a stage, kernel <paramref name="upsampleKernels"/>[i] at stride <paramref name="upsampleRates"/>[i]),
    /// <c>resblocks.{i · kernels + j}</c> (<paramref name="resblockKernels"/>[j] taps, <paramref name="resblockDilations"/>[j]) and <c>conv_post</c>.
    /// </summary>
    public static HifiGanGenerator FromStored(StoredLayers stored, string prefix, int flowSize, int initialChannels, IReadOnlyList<int> upsampleRates, IReadOnlyList<int> upsampleKernels,
        IReadOnlyList<int> resblockKernels, IReadOnlyList<int[]> resblockDilations, float slope, float finalSlope, int conditioningSize)
    {
        ArgumentNullException.ThrowIfNull(stored);
        ArgumentNullException.ThrowIfNull(upsampleRates);
        ArgumentNullException.ThrowIfNull(resblockKernels);
        if (upsampleKernels.Count != upsampleRates.Count || resblockDilations.Count != resblockKernels.Count)
        {
            throw new ArgumentException($"The upsampling rates and kernels pair up ({upsampleRates.Count}, {upsampleKernels.Count}), as do the residual blocks' kernels and dilations ({resblockKernels.Count}, {resblockDilations.Count}).", nameof(upsampleKernels));
        }

        var pre = stored.Conv(StoredLayers.Join(prefix, "conv_pre"), flowSize, initialChannels, 7, 3);
        var cond = conditioningSize > 0 ? stored.Conv(StoredLayers.Join(prefix, "cond"), conditioningSize, initialChannels, 1, 0) : null;
        int stages = upsampleRates.Count, kernels = resblockKernels.Count;
        var upsamplers = new ConvTranspose1d[stages];
        var blocks = new HifiGanResidualBlock[stages * kernels];
        int hop = 1;
        for (int i = 0; i < stages; i++)
        {
            int rate = upsampleRates[i], kernel = upsampleKernels[i];
            int inChannels = initialChannels >> i, outChannels = initialChannels >> (i + 1);
            upsamplers[i] = stored.ConvTranspose(StoredLayers.Join(prefix, $"upsampler.{i}"), inChannels, outChannels, kernel, rate, (kernel - rate) / 2);
            hop *= rate;
            for (int j = 0; j < kernels; j++)
            {
                blocks[i * kernels + j] = HifiGanResidualBlock.FromStored(stored, StoredLayers.Join(prefix, $"resblocks.{i * kernels + j}"), outChannels, resblockKernels[j], resblockDilations[j], slope);
            }
        }

        var post = stored.Conv(StoredLayers.Join(prefix, "conv_post"), initialChannels >> stages, 1, 7, 3);
        return new HifiGanGenerator(pre, cond, upsamplers, blocks, post, kernels, slope, finalSlope, hop, ContextFramesFor(upsampleRates, upsampleKernels, resblockKernels, resblockDilations));
    }

    /// <summary>
    /// One side of the receptive field, in latent frames: conv_pre's half kernel; per stage the transposed convolution's
    /// reach (its kernel over its stride, in the stage's input frames) and the widest residual block's (each dilated pair's
    /// half kernels, in the stage's samples); conv_post's half kernel at the output rate; plus two.
    /// </summary>
    public static int ContextFramesFor(IReadOnlyList<int> upsampleRates, IReadOnlyList<int> upsampleKernels, IReadOnlyList<int> resblockKernels, IReadOnlyList<int[]> resblockDilations)
    {
        ArgumentNullException.ThrowIfNull(upsampleRates);
        double frames = 3, rate = 1;
        for (int i = 0; i < upsampleRates.Count; i++)
        {
            frames += Math.Ceiling((double)upsampleKernels[i] / upsampleRates[i]) / rate;
            rate *= upsampleRates[i];
            double widest = 0;
            for (int j = 0; j < resblockKernels.Count; j++)
            {
                int half = (resblockKernels[j] - 1) / 2;
                widest = Math.Max(widest, resblockDilations[j].Sum(d => half * d + half));
            }

            frames += widest / rate;
        }

        return (int)Math.Ceiling(frames + 3 / rate) + 2;
    }

    /// <summary>
    /// The samples of the latents [F, T] on the host. With gradients off and <paramref name="chunkFrames"/> above 0 and
    /// below T, the latents are decoded in pieces of that many frames, each with <see cref="ContextFrames"/> of context
    /// on both sides that is cut away again: the device holds one piece's tensors, whatever the length. A piece the
    /// device has no memory for (<see cref="ResourceLimitExceededException"/>) is decoded again in halves.
    /// </summary>
    public float[] Samples(Tensor latents, Tensor? speaker, int chunkFrames)
    {
        ArgumentNullException.ThrowIfNull(latents);
        int frames = latents.Shape[1], hop = Hop;
        if (chunkFrames <= 0 || frames <= chunkFrames || Autograd.IsEnabled)
        {
            return Forward(latents, speaker).ToArray();
        }

        var output = new float[(long)frames * hop];
        int chunk = chunkFrames;
        for (int start = 0; start < frames;)
        {
            int count = Math.Min(chunk, frames - start);
            int from = Math.Max(0, start - ContextFrames), to = Math.Min(frames, start + count + ContextFrames);
            try
            {
                using var scope = new TensorScope();
                var piece = Forward(latents.Narrow(1, from, to - from), speaker).ToArray();
                piece.AsSpan((start - from) * hop, count * hop).CopyTo(output.AsSpan(start * hop));
                start += count;
            }
            catch (ResourceLimitExceededException) when (chunk > 2 * ContextFrames)
            {
                chunk /= 2;                                                      // short of device memory: smaller pieces, the same samples
            }
        }

        return output;
    }

    /// <summary>
    /// The samples [1, T · hop] of the latents [F, T], or [B, T · hop] of a batch [B, F, T], with the speaker's embedding
    /// [E, 1] (null for none; a batch takes none). With gradients off, each stage and each residual block's step runs in a
    /// scope of its own that keeps only its result, so the device holds a few tensors of the widest stage at a time. With
    /// gradients on, each residual block is a block of the model's own checkpoints (<see cref="Checkpointing.RunBlock"/>).
    /// </summary>
    public Tensor Forward(Tensor latents, Tensor? speaker)
    {
        bool release = !Autograd.IsEnabled;
        int batch = latents.Shape.Length == 3 ? latents.Shape[0] : 1;
        Tensor x;
        using (var first = release ? new TensorScope() : null)
        {
            x = _pre.Forward(latents.Shape.Length == 3 ? latents : latents.Reshape(1, latents.Shape[0], latents.Shape[1]));
            if (speaker is not null && _cond is not null)
            {
                if (batch != 1)
                {
                    throw new NotSupportedException("The generator takes a speaker's embedding for one sequence's latents at a time, not for a batch.");
                }

                x = x + AudioLayerOps.ExpandColumns(_cond.Forward(speaker), x.Shape[2]).Reshape(1, x.Shape[1], x.Shape[2]);
            }

            first?.Keep(x);
        }

        for (int i = 0; i < _upsamplers.Length; i++)
        {
            var previous = x;
            using (var stage = release ? new TensorScope() : null)
            {
                var up = _upsamplers[i].Forward(AudioLayerOps.LeakyRelu(x, _slope));
                Tensor? sum = null;
                for (int j = 0; j < _kernels; j++)
                {
                    using var block = release ? new TensorScope() : null;
                    var residual = _blocks[i * _kernels + j];
                    var y = Checkpointing.RunBlock(input => residual.Forward(input, release), up);
                    var total = sum is null ? y : sum + y;
                    block?.Keep(total);
                    if (release && sum is not null && stage!.Owns(sum))
                    {
                        TensorScope.Untrack(sum).Dispose();                     // the previous partial sum is not needed again
                    }

                    sum = total;
                }

                x = _kernels == 1 ? sum! : sum! * (1f / _kernels);
                stage?.Keep(x);
            }

            if (release && !ReferenceEquals(previous, latents))
            {
                TensorScope.Untrack(previous).Dispose();                         // the stage before is not needed again
            }
        }

        using var last = release ? new TensorScope() : null;
        var made = _post.Forward(AudioLayerOps.LeakyRelu(x, _finalSlope)).Tanh();   // [B, 1, L]
        var samples = made.Reshape(batch, made.Shape[2]);
        last?.Keep(samples);
        if (release)
        {
            TensorScope.Untrack(x).Dispose();
        }

        return samples;
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => Forward(input, null);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_pre, .. _upsamplers, .. _blocks, _post, .. (_cond is null ? Array.Empty<Module>() : [_cond])];

    /// <inheritdoc />
    protected override void MoveTo(Device device) => throw new NotSupportedException("The layers are built on their device.");

    /// <inheritdoc />
    public override void Dispose()
    {
        _pre.Dispose();
        _post.Dispose();
        _cond?.Dispose();
        foreach (var u in _upsamplers) u.Dispose();
        foreach (var b in _blocks) b.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"HifiGanGenerator({_upsamplers.Length} stages, hop {Hop})";
}
