// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.VitsSpeech;

/// <summary>
/// transformers' <c>VitsHifiGan</c>: a convolution into the first channel count, then per stage a leaky ReLU, a
/// transposed convolution (the rate's upsampling), the residual blocks averaged; a last leaky ReLU, the output
/// convolution and tanh. <see cref="VitsSynthesizer.Vocoder"/> exposes it as an <see cref="IVocoder"/> over the flow's latents.
/// </summary>
internal sealed class HifiGanDecoder : IDisposable
{
    private readonly VitsConfig _config;
    private readonly Conv1d _pre, _post;
    private readonly Conv1d? _cond;
    private readonly ConvTranspose1d[] _upsamplers;
    private readonly ResidualBlock[] _blocks;

    public HifiGanDecoder(VitsWeights weights, VitsConfig config)
    {
        _config = config;
        int channels = config.UpsampleInitialChannel;
        _pre = weights.Conv("decoder.conv_pre", config.FlowSize, channels, 7, 3);
        if (config.SpeakerEmbeddingSize > 0)
        {
            _cond = weights.Conv("decoder.cond", config.SpeakerEmbeddingSize, channels, 1, 0);
        }

        int stages = config.UpsampleRates.Length, kernels = config.ResblockKernelSizes.Length;
        _upsamplers = new ConvTranspose1d[stages];
        _blocks = new ResidualBlock[stages * kernels];
        for (int i = 0; i < stages; i++)
        {
            int rate = config.UpsampleRates[i], kernel = config.UpsampleKernelSizes[i];
            int inChannels = channels >> i, outChannels = channels >> (i + 1);
            _upsamplers[i] = weights.ConvTranspose($"decoder.upsampler.{i}", inChannels, outChannels, kernel, rate, (kernel - rate) / 2);
            for (int j = 0; j < kernels; j++)
            {
                _blocks[i * kernels + j] = new ResidualBlock(weights, $"decoder.resblocks.{i * kernels + j}", outChannels, config.ResblockKernelSizes[j],
                    config.ResblockDilationSizes[j], config.LeakyReluSlope);
            }
        }

        _post = weights.Conv("decoder.conv_post", channels >> stages, 1, 7, 3);
        ContextFrames = Context(config);
    }

    /// <summary>
    /// The latent frames on each side that an output sample depends on (the receptive field of the convolutions and the
    /// transposed convolutions, in frames, rounded up, plus two): a piece decoded with this much context on both sides
    /// gives the samples a whole pass gives.
    /// </summary>
    public int ContextFrames { get; }

    /// <summary>
    /// The samples of the latents [F, T] on the host. With gradients off and <paramref name="chunkFrames"/> above 0 and
    /// below T, the latents are decoded in pieces of that many frames, each with <see cref="ContextFrames"/> of context
    /// on both sides that is cut away again: the device holds one piece's tensors, whatever the length (rule 76). A
    /// piece the device has no memory for (<see cref="ResourceLimitExceededException"/>) is decoded again in halves.
    /// </summary>
    public float[] Samples(Tensor latents, Tensor? speaker, int chunkFrames)
    {
        int frames = latents.Shape[1], hop = _config.HopLength;
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

    // One side of the receptive field, in latent frames: conv_pre's half kernel; per stage the transposed convolution's
    // reach (its kernel over its stride, in the stage's input frames) and the widest residual block's (each dilated pair's
    // half kernels, in the stage's samples); conv_post's half kernel at the output rate.
    private static int Context(VitsConfig config)
    {
        double frames = 3, rate = 1;
        for (int i = 0; i < config.UpsampleRates.Length; i++)
        {
            frames += Math.Ceiling((double)config.UpsampleKernelSizes[i] / config.UpsampleRates[i]) / rate;
            rate *= config.UpsampleRates[i];
            double widest = 0;
            for (int j = 0; j < config.ResblockKernelSizes.Length; j++)
            {
                int half = (config.ResblockKernelSizes[j] - 1) / 2;
                widest = Math.Max(widest, config.ResblockDilationSizes[j].Sum(d => half * d + half));
            }

            frames += widest / rate;
        }

        return (int)Math.Ceiling(frames + 3 / rate) + 2;
    }

    /// <summary>
    /// The samples [1, T · hop] of the latents [F, T], or [B, T · hop] of a batch [B, F, T] (training: every utterance's
    /// segment through each layer in one launch), with the speaker's embedding [E, 1] (null for one speaker; a batch
    /// takes none). With gradients off, each stage and each residual block's step runs in a scope of its own that keeps
    /// only its result, so the device holds a few tensors of the widest stage at a time, not every intermediate of the
    /// pass (rule 76); with gradients on, every intermediate stays for the backward pass.
    /// </summary>
    public Tensor Forward(Tensor latents, Tensor? speaker)
    {
        bool release = !Autograd.IsEnabled;
        int batch = latents.Shape.Length == 3 ? latents.Shape[0] : 1;
        Tensor x;
        using (var first = release ? new TensorScope() : null)
        {
            x = _pre.Forward3(latents.Shape.Length == 3 ? latents : latents.Reshape(1, latents.Shape[0], latents.Shape[1]));
            if (speaker is not null && _cond is not null)
            {
                if (batch != 1)
                {
                    throw new NotSupportedException("The decoder takes a speaker's embedding for one utterance's latents at a time, not for a batch.");
                }

                x = x + VitsOps.ExpandColumns(_cond.Forward(speaker), x.Shape[2]).Reshape(1, x.Shape[1], x.Shape[2]);
            }

            first?.Keep(x);
        }

        int kernels = _config.ResblockKernelSizes.Length;
        for (int i = 0; i < _upsamplers.Length; i++)
        {
            var previous = x;
            using (var stage = release ? new TensorScope() : null)
            {
                var up = _upsamplers[i].Forward3(VitsOps.LeakyRelu(x, _config.LeakyReluSlope));
                Tensor? sum = null;
                for (int j = 0; j < kernels; j++)
                {
                    using var block = release ? new TensorScope() : null;
                    var residual = _blocks[i * kernels + j];
                    var y = VitsCheckpoints.Run(x => residual.Forward(x, release), up);
                    var total = sum is null ? y : sum + y;
                    block?.Keep(total);
                    if (release && sum is not null && stage!.Owns(sum))
                    {
                        TensorScope.Untrack(sum).Dispose();                     // the previous partial sum is not needed again
                    }

                    sum = total;
                }

                x = kernels == 1 ? sum! : sum! * (1f / kernels);
                stage?.Keep(x);
            }

            if (release && !ReferenceEquals(previous, latents))
            {
                TensorScope.Untrack(previous).Dispose();                         // the stage before is not needed again
            }
        }

        using var last = release ? new TensorScope() : null;
        var made = _post.Forward3(VitsOps.LeakyRelu(x, _config.FinalLeakyReluSlope)).Tanh();   // [B, 1, L]
        var samples = made.Reshape(batch, made.Shape[2]);
        last?.Keep(samples);
        if (release)
        {
            TensorScope.Untrack(x).Dispose();
        }

        return samples;
    }

    public void Dispose()
    {
        _pre.Dispose();
        _post.Dispose();
        _cond?.Dispose();
        foreach (var u in _upsamplers) u.Dispose();
        foreach (var b in _blocks) b.Dispose();
    }

    private sealed class ResidualBlock : IDisposable
    {
        private readonly float _slope;
        private readonly Conv1d[] _convs1, _convs2;

        public ResidualBlock(VitsWeights weights, string prefix, int channels, int kernel, int[] dilations, float slope)
        {
            _slope = slope;
            _convs1 = new Conv1d[dilations.Length];
            _convs2 = new Conv1d[dilations.Length];
            for (int i = 0; i < dilations.Length; i++)
            {
                int dilation = dilations[i];
                _convs1[i] = weights.Conv($"{prefix}.convs1.{i}", channels, channels, kernel, (kernel * dilation - dilation) / 2, dilation);
                _convs2[i] = weights.Conv($"{prefix}.convs2.{i}", channels, channels, kernel, (kernel - 1) / 2);
            }
        }

        // With `release`, each step in its own scope keeping only the new x, and an x a previous step made released once replaced.
        public Tensor Forward(Tensor x, bool release)
        {
            var input = x;
            for (int i = 0; i < _convs1.Length; i++)
            {
                Tensor next;
                using (var step = release ? new TensorScope() : null)
                {
                    var h = _convs1[i].Forward3(VitsOps.LeakyRelu(x, _slope));
                    h = _convs2[i].Forward3(VitsOps.LeakyRelu(h, _slope));
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

        public void Dispose()
        {
            foreach (var c in _convs1) c.Dispose();
            foreach (var c in _convs2) c.Dispose();
        }
    }
}
