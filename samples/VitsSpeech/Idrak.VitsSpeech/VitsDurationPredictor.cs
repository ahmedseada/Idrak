// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models;

namespace Idrak.VitsSpeech;

/// <summary>How many frames each token lasts: the log-durations of the hidden states [C, T] (speaking: one utterance, no mask).</summary>
internal interface IVitsDurationPredictor : IDisposable
{
    /// <summary>The log-durations [T] of <paramref name="hidden"/> [C, T], with the speaker's embedding [E, 1] (null for one speaker) and the noise scale (the stochastic predictor draws from <paramref name="random"/>).</summary>
    float[] LogDurations(Tensor hidden, Tensor? speaker, float noiseScale, Random random);

    /// <summary>
    /// The training loss of the tokens' frames <paramref name="durations"/> [B, 1, T] (1 over the padding) given a batch of
    /// hidden states [B, C, T] (detached by the caller, zero over the padding), the noise [B, 2, T] a stochastic predictor
    /// draws (zero over the padding) and the batch's mask (null: one utterance, unpadded): a scalar summed over the tokens.
    /// </summary>
    Tensor Loss(Tensor hidden, Tensor durations, Tensor noise, PaddedSequences? mask);
}

/// <summary>
/// transformers' <c>VitsStochasticDurationPredictor</c> run in reverse (inference): the conditioning from the hidden
/// states (a pointwise convolution, the depth-separable convolution, a pointwise projection), then noise [2, T] scaled
/// by the noise scale through the flows in reverse order (each after flipping the two channels): the spline flows but
/// the first, then the element-wise affine. The first channel is the log-duration. The splines run on the host in
/// double (their inputs are T values a token each; the durations are needed on the host anyway).
/// </summary>
internal sealed class VitsStochasticDurationPredictor : IVitsDurationPredictor
{
    private readonly VitsConfig _config;
    private readonly Conv1d _pre, _proj;
    private readonly Conv1d? _cond;
    private readonly DilatedDepthSeparableConv _dds;
    private readonly ElementwiseAffineFlow _affine;                              // the element-wise affine flow over the channels
    private readonly bool _trainable;
    private readonly ConvolutionalSplineFlow[] _flows;
    private readonly Posterior? _posterior;                                      // training: the posterior flows over the durations

    public VitsStochasticDurationPredictor(StoredLayers weights, VitsConfig config)
    {
        _config = config;
        int c = config.HiddenSize;
        _pre = weights.Conv("duration_predictor.conv_pre", c, c, 1, 0);
        _proj = weights.Conv("duration_predictor.conv_proj", c, c, 1, 0);
        _dds = Dds(weights, "duration_predictor.conv_dds", config);
        if (config.SpeakerEmbeddingSize > 0)
        {
            _cond = weights.Conv("duration_predictor.cond", config.SpeakerEmbeddingSize, c, 1, 0);
        }

        int channels = config.DepthSeparableChannels;
        _affine = ElementwiseAffineFlow.FromStored(weights, "duration_predictor.flows.0", channels);
        _trainable = weights.Trainable;

        _flows = new ConvolutionalSplineFlow[config.DurationFlows];
        for (int i = 0; i < config.DurationFlows; i++)
        {
            _flows[i] = SplineFlow(weights, $"duration_predictor.flows.{i + 1}", config);
        }

        if (weights.Trainable)
        {
            if (channels != 2)
            {
                throw new NotSupportedException($"Training reads the stochastic duration predictor's two channels (the duration and its pair); depth_separable_channels is {channels}.");
            }

            if (!weights.Contains("duration_predictor.post_conv_pre.weight"))
            {
                throw new InvalidDataException("The checkpoint has no duration_predictor.post_* tensors (the posterior flows over the durations, which training needs and "
                    + "transformers' VitsModel holds); train from a checkpoint that keeps them.");
            }

            _posterior = new Posterior(weights, config);
        }
    }

    // The library's dilated depth-separable convolution and spline flow at the configuration's shapes.
    private static DilatedDepthSeparableConv Dds(StoredLayers weights, string prefix, VitsConfig config) =>
        DilatedDepthSeparableConv.FromStored(weights, prefix, config.HiddenSize, config.DurationKernelSize, config.DepthSeparableLayers, config.LayerNormEps);

    private static ConvolutionalSplineFlow SplineFlow(StoredLayers weights, string prefix, VitsConfig config) =>
        ConvolutionalSplineFlow.FromStored(weights, prefix, config.DepthSeparableChannels / 2, config.HiddenSize, config.DurationFlowBins, config.DurationTailBound,
            config.DurationKernelSize, config.DepthSeparableLayers, config.LayerNormEps);

    /// <summary>
    /// The training loss of the durations (the forward direction, reverse false): the negative log-likelihood of each
    /// token's frames <paramref name="durations"/> [1, T] under the flows conditioned on <paramref name="hidden"/> [C, T]
    /// (detached: the text encoder gets no gradient from it, as in VITS), dequantized by the posterior flows over
    /// <paramref name="noise"/> [2, T], plus the posterior's log-density: a scalar, summed over the tokens. The flows run
    /// in VITS's order, no flip after the element-wise affine ([affine, conv flow, flip, conv flow, flip, ...]): the order
    /// whose inverse is the reverse <see cref="LogDurations"/> speaks with. transformers' forward flips after the affine
    /// too, which trains flows its reverse does not undo: a model fine-tuned that way speaks with durations it was not
    /// trained on (far shorter ones on a real checkpoint).
    /// </summary>
    public Tensor Loss(Tensor hidden, Tensor durations, Tensor noise, PaddedSequences? mask) => Nll(hidden, durations, noise, mask);

    public Tensor Nll(Tensor hidden, Tensor durations, Tensor noise, PaddedSequences? mask)
    {
        var posterior = _posterior ?? throw new InvalidOperationException("The duration predictor was loaded to speak; load the model with SpeechSynthesisLoadOptions.Training to train it.");
        int t = hidden.Shape[2], positions = mask?.Positions ?? t;
        var x = Conditioning(hidden, null, mask);
        var h = posterior.Proj.Forward(posterior.Dds.Forward(posterior.Pre.Forward(durations), null, mask));
        if (mask is not null)
        {
            h = mask.Apply(h);
        }

        // The posterior: the noise through the affine, then its flows (each after which the channels flip), conditioned on both.
        var conditioning = x + h;
        var latents = posterior.Affine.Forward(noise, mask, positions, out var logDetQ);
        foreach (var flow in posterior.Flows)
        {
            latents = flow.Forward(latents, conditioning, mask, out var logDet);
            latents = Flip(latents);
            logDetQ = logDetQ + logDet;
        }

        var first = latents.Narrow(1, 0, 1);                                     // [B, 1, T]
        var second = latents.Narrow(1, 1, 1);
        var magnitude = first.Abs();
        var sigmoids = magnitude + ((magnitude * -1f).Exp() + 1f).Log() * 2f;   // -(logsigmoid(u) + logsigmoid(-u))
        logDetQ = logDetQ - (mask is null ? sigmoids : mask.Apply(sigmoids)).Sum();
        var noiseValues = noise.ToArray();                                       // [B, 2, T]: the padding's noise is zero
        double noiseSquares = 0;
        foreach (float v in noiseValues)
        {
            noiseSquares += (double)v * v;
        }

        float logQ0 = (float)(-0.5 * (2 * positions * Math.Log(2 * Math.PI) + noiseSquares));
        var logQ = logDetQ * -1f + logQ0;

        // The durations dequantized by the posterior's first channel, through the flows to the prior.
        var dequantized = (durations - first.Sigmoid()).Clamp(1e-5f, float.PositiveInfinity).Log();
        latents = Prior(Tensor.Concat([dequantized, second], 1), x, mask, out var flowsDet);
        var logDetSum = (mask is null ? dequantized : mask.Apply(dequantized)).Sum() * -1f + flowsDet;
        var squares = latents.Square() * 0.5f;
        var nll = (mask is null ? squares : mask.Apply(squares)).Sum() + (float)(0.5 * 2 * positions * Math.Log(2 * Math.PI)) - logDetSum;
        return nll + logQ;
    }

    /// <summary>The conditioning of every flow [C, T] from the hidden states [C, T] and the speaker's embedding [E, 1] (null for one speaker).</summary>
    internal Tensor Conditioning(Tensor hidden, Tensor? speaker)
    {
        int c = hidden.Shape[0], t = hidden.Shape[1];
        return Conditioning(hidden.Reshape(1, c, t), speaker, null).Reshape(c, t);
    }

    /// <summary>The conditioning of every flow [B, C, T] from a batch of hidden states [B, C, T] with the speaker's embedding (one utterance only) and the batch's mask, zero over the padding.</summary>
    internal Tensor Conditioning(Tensor hidden, Tensor? speaker, PaddedSequences? mask)
    {
        var x = _pre.Forward(hidden);
        if (speaker is not null && _cond is not null)
        {
            if (hidden.Shape[0] != 1)
            {
                throw new NotSupportedException("The duration predictor takes a speaker's embedding for one utterance at a time, not for a batch.");
            }

            x = x + VitsOps.ExpandColumns(_cond.Forward(speaker), hidden.Shape[2]).Reshape(1, x.Shape[1], hidden.Shape[2]);
        }

        var conditioning = _proj.Forward(_dds.Forward(x, null, mask));
        return mask is null ? conditioning : mask.Apply(conditioning);
    }

    /// <summary>
    /// The flows forward (training): latents [B, 2, T] (the log-durations and their pair) through the element-wise affine,
    /// then each spline flow followed by a flip, conditioned on <paramref name="x"/> [B, C, T]; the log-determinant summed
    /// over the tokens within the utterances. <see cref="Reverse"/> takes the result back to the log-durations.
    /// </summary>
    internal Tensor Prior(Tensor latents, Tensor x, PaddedSequences? mask, out Tensor logDet)
    {
        if (!_trainable)
        {
            throw new InvalidOperationException("The duration predictor was loaded to speak; load the model with SpeechSynthesisLoadOptions.Training to train it.");
        }

        latents = _affine.Forward(latents, mask, mask?.Positions ?? latents.Shape[2], out logDet);
        foreach (var flow in _flows)
        {
            latents = flow.Forward(latents, x, mask, out var flowDet);
            latents = Flip(latents);
            logDet = logDet + flowDet;
        }

        return latents;
    }

    // torch.flip over the two channels of [B, 2, T].
    private static Tensor Flip(Tensor x) => Tensor.Concat([x.Narrow(1, 1, 1), x.Narrow(1, 0, 1)], 1);

    public float[] LogDurations(Tensor hidden, Tensor? speaker, float noiseScale, Random random)
    {
        int t = hidden.Shape[1], channels = _config.DepthSeparableChannels;
        var x = Conditioning(hidden, speaker);                                   // the conditioning of every flow, [C, T]

        // The latents [channels, T]: noise times the scale (none at scale 0).
        var latents = new float[channels * t];
        if (noiseScale != 0)
        {
            for (int i = 0; i < latents.Length; i++)
            {
                latents[i] = (float)(random.NextGaussian() * noiseScale);
            }
        }

        return Reverse(latents, x);
    }

    /// <summary>
    /// The flows in reverse (speaking) on latents [channels, T] on the host (overwritten), conditioned on <paramref name="x"/>
    /// [C, T]: the log-durations [T]. The first spline flow is left out, as transformers and VITS leave it: forward, it
    /// changes only the second channel, which the affine keeps apart from the log-durations.
    /// </summary>
    internal float[] Reverse(float[] latents, Tensor x)
    {
        int t = x.Shape[1], channels = _config.DepthSeparableChannels;
        // reversed(flows)[:-2] + [reversed(flows)[-1]]: the spline flows from the last down to the second, then the affine.
        for (int i = _flows.Length - 1; i >= 1; i--)
        {
            FlipChannels(latents, channels, t);
            _flows[i].Reverse(latents, x, channels, t);
        }

        FlipChannels(latents, channels, t);
        _affine.Reverse(latents, t);                                             // the model's weights as they are now; the first channel is the log-duration
        return latents[..t];
    }

    // torch.flip over the channel axis of [channels, T].
    private static void FlipChannels(float[] values, int channels, int t)
    {
        for (int c = 0; c < channels / 2; c++)
        {
            var a = values.AsSpan(c * t, t);
            var b = values.AsSpan((channels - 1 - c) * t, t);
            for (int n = 0; n < t; n++)
            {
                (a[n], b[n]) = (b[n], a[n]);
            }
        }
    }

    public void Dispose()
    {
        _pre.Dispose();
        _proj.Dispose();
        _cond?.Dispose();
        _dds.Dispose();
        foreach (var flow in _flows) flow.Dispose();
        _affine.Dispose();
        _posterior?.Dispose();
    }

    // The posterior flows over the durations (transformers' post_conv_pre, post_conv_dds, post_conv_proj, post_flows): training only.
    private sealed class Posterior : IDisposable
    {
        public Posterior(StoredLayers weights, VitsConfig config)
        {
            int c = config.HiddenSize, channels = config.DepthSeparableChannels;
            Pre = weights.Conv("duration_predictor.post_conv_pre", 1, c, 1, 0);
            Proj = weights.Conv("duration_predictor.post_conv_proj", c, c, 1, 0);
            Dds = VitsStochasticDurationPredictor.Dds(weights, "duration_predictor.post_conv_dds", config);
            Affine = ElementwiseAffineFlow.FromStored(weights, "duration_predictor.post_flows.0", channels);
            Flows = new ConvolutionalSplineFlow[config.DurationFlows];
            for (int i = 0; i < Flows.Length; i++)
            {
                Flows[i] = SplineFlow(weights, $"duration_predictor.post_flows.{i + 1}", config);
            }
        }

        public Conv1d Pre { get; }

        public Conv1d Proj { get; }

        public DilatedDepthSeparableConv Dds { get; }

        public ElementwiseAffineFlow Affine { get; }

        public ConvolutionalSplineFlow[] Flows { get; }

        public void Dispose()
        {
            Pre.Dispose();
            Proj.Dispose();
            Dds.Dispose();
            Affine.Dispose();
            foreach (var flow in Flows) flow.Dispose();
        }
    }
}

/// <summary>
/// transformers' <c>VitsDurationPredictor</c> (the deterministic one, when <c>use_stochastic_duration_prediction</c> is
/// false): two convolutions with ReLU and LayerNorm, then a pointwise projection to one channel.
/// </summary>
internal sealed class VitsDeterministicDurationPredictor : IVitsDurationPredictor
{
    private readonly Conv1d _conv1, _conv2, _proj;
    private readonly Conv1d? _cond;
    private readonly LayerNorm _norm1, _norm2;

    public VitsDeterministicDurationPredictor(StoredLayers weights, VitsConfig config)
    {
        int k = config.DurationKernelSize, filters = config.DurationFilterChannels;
        _conv1 = weights.Conv("duration_predictor.conv_1", config.HiddenSize, filters, k, k / 2);
        _norm1 = weights.Norm("duration_predictor.norm_1", filters, config.LayerNormEps);
        _conv2 = weights.Conv("duration_predictor.conv_2", filters, filters, k, k / 2);
        _norm2 = weights.Norm("duration_predictor.norm_2", filters, config.LayerNormEps);
        _proj = weights.Conv("duration_predictor.proj", filters, 1, 1, 0);
        if (config.SpeakerEmbeddingSize > 0)
        {
            _cond = weights.Conv("duration_predictor.cond", config.SpeakerEmbeddingSize, config.HiddenSize, 1, 0);
        }
    }

    public float[] LogDurations(Tensor hidden, Tensor? speaker, float noiseScale, Random random)
    {
        var x = hidden;
        if (speaker is not null && _cond is not null)
        {
            x = x + VitsOps.ExpandColumns(_cond.Forward(speaker), hidden.Shape[1]);
        }

        return Predict(x).ToArray();                                             // [1, T]
    }

    /// <summary>VITS's deterministic duration loss: Σ (log-duration predicted - log(frames + 1e-6))² over the tokens within the utterances.</summary>
    public Tensor Loss(Tensor hidden, Tensor durations, Tensor noise, PaddedSequences? mask)
    {
        _ = noise;
        var errors = (Predict3(hidden, mask) - (durations + 1e-6f).Log()).Square();
        return (mask is null ? errors : mask.Apply(errors)).Sum();
    }

    private Tensor Predict(Tensor x) => Predict3(x.Reshape(1, x.Shape[0], x.Shape[1]), null).Reshape(1, x.Shape[1]);

    // [B, C, T] to the log-durations [B, 1, T]; the convolutions read the padding as zeros.
    private Tensor Predict3(Tensor x, PaddedSequences? mask)
    {
        x = Norm(_norm1, _conv1.Forward(mask is null ? x : mask.Apply(x)).Relu());
        x = Norm(_norm2, _conv2.Forward(mask is null ? x : mask.Apply(x)).Relu());
        return _proj.Forward(mask is null ? x : mask.Apply(x));
    }

    private static Tensor Norm(LayerNorm norm, Tensor h) => norm.Forward(h.Permute(0, 2, 1)).Permute(0, 2, 1);

    public void Dispose()
    {
        _conv1.Dispose();
        _conv2.Dispose();
        _proj.Dispose();
        _cond?.Dispose();
        _norm1.Dispose();
        _norm2.Dispose();
    }
}
