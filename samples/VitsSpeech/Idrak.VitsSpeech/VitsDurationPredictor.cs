// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

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
    private readonly VitsDilatedDepthSeparableConv _dds;
    private readonly float[] _logScale, _translate;                              // the element-wise affine flow, [channels]
    private readonly Tensor? _logScaleParameter, _translateParameter;            // training: the same as parameters [channels, 1]
    private readonly ConvFlow[] _flows;
    private readonly Posterior? _posterior;                                      // training: the posterior flows over the durations

    public VitsStochasticDurationPredictor(VitsWeights weights, VitsConfig config)
    {
        _config = config;
        int c = config.HiddenSize;
        _pre = weights.Conv("duration_predictor.conv_pre", c, c, 1, 0);
        _proj = weights.Conv("duration_predictor.conv_proj", c, c, 1, 0);
        _dds = new VitsDilatedDepthSeparableConv(weights, "duration_predictor.conv_dds", config);
        if (config.SpeakerEmbeddingSize > 0)
        {
            _cond = weights.Conv("duration_predictor.cond", config.SpeakerEmbeddingSize, c, 1, 0);
        }

        int channels = config.DepthSeparableChannels;
        _logScale = weights.Read("duration_predictor.flows.0.log_scale");
        _translate = weights.Read("duration_predictor.flows.0.translate");
        if (_logScale.Length != channels || _translate.Length != channels)
        {
            throw new InvalidDataException($"duration_predictor.flows.0 has {_logScale.Length} channels, the configuration says {channels}.");
        }

        _flows = new ConvFlow[config.DurationFlows];
        for (int i = 0; i < config.DurationFlows; i++)
        {
            _flows[i] = new ConvFlow(weights, $"duration_predictor.flows.{i + 1}", config);
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

            _logScaleParameter = weights.Parameter("duration_predictor.flows.0.log_scale", _logScale, [channels, 1], weights.Shape("duration_predictor.flows.0.log_scale"));
            _translateParameter = weights.Parameter("duration_predictor.flows.0.translate", _translate, [channels, 1], weights.Shape("duration_predictor.flows.0.translate"));
            _posterior = new Posterior(weights, config);
        }
    }

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
        var latents = Affine(noise, posterior.LogScale, posterior.Translate, mask, positions, out var logDetQ);
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
        var logScale = _logScaleParameter ?? throw new InvalidOperationException("The duration predictor was loaded to speak; load the model with SpeechSynthesisLoadOptions.Training to train it.");
        latents = Affine(latents, logScale, _translateParameter!, mask, mask?.Positions ?? latents.Shape[2], out logDet);
        foreach (var flow in _flows)
        {
            latents = flow.Forward(latents, x, mask, out var flowDet);
            latents = Flip(latents);
            logDet = logDet + flowDet;
        }

        return latents;
    }

    // transformers' VitsElementwiseAffine forward on [B, 2, T]: translate + exp(log_scale) · x per channel (one row-wise
    // affine over the channels), zero over the padding; its log-determinant is Σ log_scale over the tokens within the utterances.
    private static Tensor Affine(Tensor x, Tensor logScale, Tensor translate, PaddedSequences? mask, int positions, out Tensor logDet)
    {
        int channels = x.Shape[1], t = x.Shape[2];
        logDet = logScale.Sum() * positions;
        var y = x.GroupAffine(logScale.Reshape(channels).Exp(), translate.Reshape(channels), channels, t);
        return mask is null ? y : mask.Apply(y);
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
                latents[i] = (float)(VitsNoise.Normal(random) * noiseScale);
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
        var logScale = _logScaleParameter?.ToArray() ?? _logScale;               // a model in training speaks with its weights as they are now
        var translate = _translateParameter?.ToArray() ?? _translate;

        // reversed(flows)[:-2] + [reversed(flows)[-1]]: the spline flows from the last down to the second, then the affine.
        for (int i = _flows.Length - 1; i >= 1; i--)
        {
            FlipChannels(latents, channels, t);
            _flows[i].Reverse(latents, x, channels, t);
        }

        FlipChannels(latents, channels, t);
        var logDurations = new float[t];
        for (int n = 0; n < t; n++)
        {
            logDurations[n] = (float)((latents[n] - translate[0]) * Math.Exp(-logScale[0]));
        }

        return logDurations;
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
        _logScaleParameter?.Dispose();
        _translateParameter?.Dispose();
        _posterior?.Dispose();
    }

    // The posterior flows over the durations (transformers' post_conv_pre, post_conv_dds, post_conv_proj, post_flows): training only.
    private sealed class Posterior : IDisposable
    {
        public Posterior(VitsWeights weights, VitsConfig config)
        {
            int c = config.HiddenSize, channels = config.DepthSeparableChannels;
            Pre = weights.Conv("duration_predictor.post_conv_pre", 1, c, 1, 0);
            Proj = weights.Conv("duration_predictor.post_conv_proj", c, c, 1, 0);
            Dds = new VitsDilatedDepthSeparableConv(weights, "duration_predictor.post_conv_dds", config);
            LogScale = weights.Tensor("duration_predictor.post_flows.0.log_scale", channels, 1);
            Translate = weights.Tensor("duration_predictor.post_flows.0.translate", channels, 1);
            Flows = new ConvFlow[config.DurationFlows];
            for (int i = 0; i < Flows.Length; i++)
            {
                Flows[i] = new ConvFlow(weights, $"duration_predictor.post_flows.{i + 1}", config);
            }
        }

        public Conv1d Pre { get; }

        public Conv1d Proj { get; }

        public VitsDilatedDepthSeparableConv Dds { get; }

        public Tensor LogScale { get; }

        public Tensor Translate { get; }

        public ConvFlow[] Flows { get; }

        public void Dispose()
        {
            Pre.Dispose();
            Proj.Dispose();
            Dds.Dispose();
            LogScale.Dispose();
            Translate.Dispose();
            foreach (var flow in Flows) flow.Dispose();
        }
    }

    // transformers' VitsConvFlow in reverse: the first half of the channels conditions a rational-quadratic spline that
    // the second half goes through backwards.
    private sealed class ConvFlow : IDisposable
    {
        private readonly VitsConfig _config;
        private readonly int _half;
        private readonly Conv1d _pre, _proj;
        private readonly VitsDilatedDepthSeparableConv _dds;

        public ConvFlow(VitsWeights weights, string prefix, VitsConfig config)
        {
            _config = config;
            _half = config.DepthSeparableChannels / 2;
            int c = config.HiddenSize, bins = config.DurationFlowBins;
            _pre = weights.Conv($"{prefix}.conv_pre", _half, c, 1, 0);
            _dds = new VitsDilatedDepthSeparableConv(weights, $"{prefix}.conv_dds", config);
            _proj = weights.Conv($"{prefix}.conv_proj", c, _half * (3 * bins - 1), 1, 0);
        }

        /// <summary>
        /// The flow forward (training) on latents [B, 2, T]: the second channel through the spline the first one conditions
        /// (with <paramref name="conditioning"/> [B, C, T]), composed from tensor operations so its gradients reach every
        /// input; the log-determinant summed over the tokens within the utterances (the batch's mask; null: unpadded).
        /// </summary>
        public Tensor Forward(Tensor latents, Tensor conditioning, PaddedSequences? mask, out Tensor logDet)
        {
            int batch = latents.Shape[0], t = latents.Shape[2], bins = _config.DurationFlowBins;
            var first = latents.Narrow(1, 0, _half);                                 // [B, 1, T]
            var second = latents.Narrow(1, _half, _half);
            var h = _proj.Forward(_dds.Forward(_pre.Forward(first), conditioning, mask));   // [B, 3 bins - 1, T] (one channel)
            if (mask is not null)
            {
                h = mask.Apply(h);
            }

            var rows = h.Permute(0, 2, 1).Reshape(batch * t, 3 * bins - 1);        // [B·T, 3 bins - 1]
            float scale = 1f / MathF.Sqrt(_config.HiddenSize);
            var (outputs, logAbsDet) = VitsSpline.Forward(second.Reshape(batch * t, 1), rows.Narrow(1, 0, bins) * scale, rows.Narrow(1, bins, bins) * scale,
                rows.Narrow(1, 2 * bins, bins - 1), _config.DurationTailBound);
            var moved = outputs.Reshape(batch, 1, t);
            var determinant = logAbsDet.Reshape(batch, 1, t);
            if (mask is not null)
            {
                moved = mask.Apply(moved);
                determinant = mask.Apply(determinant);
            }

            logDet = determinant.Sum();
            return Tensor.Concat([first, moved], 1);
        }

        /// <summary>latents [channels, T] on the host: the second half replaced by the spline's inverse, conditioned on the first half and <paramref name="conditioning"/> [C, T].</summary>
        public void Reverse(float[] latents, Tensor conditioning, int channels, int t)
        {
            int bins = _config.DurationFlowBins, perChannel = 3 * bins - 1;
            float[] parameters;
            using (new TensorScope())
            {
                var first = Tensor.From(latents.AsSpan(0, _half * t), [_half, t], conditioning.Device);
                var h = _proj.Forward(_dds.Forward(_pre.Forward(first), conditioning));   // [half · (3 bins - 1), T]
                parameters = h.ToArray();
            }

            double scale = 1 / Math.Sqrt(_config.HiddenSize);
            var widths = new double[bins];
            var heights = new double[bins];
            var derivatives = new double[bins + 1];
            for (int c = 0; c < _half; c++)
            {
                for (int n = 0; n < t; n++)
                {
                    // hidden.reshape(half, 3 bins - 1, T).permute(0, 2, 1): parameter k of channel c at time n.
                    int at = c * perChannel * t + n;
                    for (int k = 0; k < bins; k++)
                    {
                        widths[k] = parameters[at + k * t] * scale;
                        heights[k] = parameters[at + (bins + k) * t] * scale;
                    }

                    for (int k = 0; k < bins - 1; k++)
                    {
                        derivatives[k + 1] = parameters[at + (2 * bins + k) * t];
                    }

                    int index = (_half + c) * t + n;
                    latents[index] = (float)VitsSpline.Inverse(latents[index], widths, heights, derivatives, _config.DurationTailBound);
                }
            }
        }

        public void Dispose()
        {
            _pre.Dispose();
            _proj.Dispose();
            _dds.Dispose();
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

    public VitsDeterministicDurationPredictor(VitsWeights weights, VitsConfig config)
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

/// <summary>
/// The unconstrained rational-quadratic spline of transformers' <c>modeling_vits</c> (Durkan et al.'s neural spline
/// flows), inverted: the identity outside ±tail, inside it the bin found by the cumulative heights and the quadratic
/// solved for the position within it. In double, so the inverse matches the float reference within its tolerance.
/// </summary>
internal static class VitsSpline
{
    private const double MinBinWidth = 1e-3, MinBinHeight = 1e-3, MinDerivative = 1e-3;

    /// <summary>
    /// The spline forward (training, transformers' <c>_unconstrained_rational_quadratic_spline</c> with reverse false):
    /// the inputs <paramref name="x"/> [N, 1] through the splines of the unnormalized widths and heights [N, bins] and
    /// inner derivatives [N, bins - 1], and the log of each one's derivative, from tensor operations so their gradients
    /// reach every input: the widths' softmax cumulated by a product with a triangle of ones, each input's bin found on
    /// the host (from the cumulative widths) and its values picked by one-hot rows. Inputs outside ±tail pass unchanged
    /// (their log-derivative 0), as transformers', and the spline is evaluated at 0 for them so nothing there is not finite.
    /// </summary>
    public static (Tensor Outputs, Tensor LogAbsDet) Forward(Tensor x, Tensor widths, Tensor heights, Tensor derivatives, float tail)
    {
        int n = widths.Shape[0], bins = widths.Shape[1];
        var device = x.Device;
        var inputs = x.ToArray();
        var inside = new float[n];
        var outside = new float[n];
        for (int i = 0; i < n; i++)
        {
            bool within = inputs[i] >= -tail && inputs[i] <= tail;
            (inside[i], outside[i]) = within ? (1f, 0f) : (0f, 1f);
        }

        var keep = Tensor.From(inside, [n, 1], device);
        var pass = Tensor.From(outside, [n, 1], device);
        var at = x * keep;                                                        // outside inputs read at 0, a point inside
        var (cumWidths, binWidths) = Cumulative(widths, (float)MinBinWidth, tail);
        var (cumHeights, binHeights) = Cumulative(heights, (float)MinBinHeight, tail);
        var ones = Tensor.Ones([n, 1], device);
        var inner = derivatives.Relu() + ((derivatives.Abs() * -1f).Exp() + 1f).Log() + (float)MinDerivative;   // min + softplus
        var slopes = Tensor.Concat([ones, inner, ones], 1);                       // [N, bins + 1]; the edge derivatives are 1 (softplus of the constant + min)

        // Each input's bin: the cumulative widths it reaches (the top one nudged up), counted on the host.
        var cum = cumWidths.ToArray();
        var pickBin = new float[n * bins];
        var pickEdge = new float[n * (bins + 1)];
        var pickNext = new float[n * (bins + 1)];
        for (int i = 0; i < n; i++)
        {
            float v = inside[i] == 1f ? inputs[i] : 0f;
            int bin = -1;
            for (int k = 0; k <= bins; k++)
            {
                float edge = cum[i * (bins + 1) + k] + (k == bins ? 1e-6f : 0f);
                if (v >= edge)
                {
                    bin++;
                }
            }

            bin = Math.Clamp(bin, 0, bins - 1);
            pickBin[i * bins + bin] = 1f;
            pickEdge[i * (bins + 1) + bin] = 1f;
            pickNext[i * (bins + 1) + bin + 1] = 1f;
        }

        var inBin = Tensor.From(pickBin, [n, bins], device);
        var atEdge = Tensor.From(pickEdge, [n, bins + 1], device);
        var atNext = Tensor.From(pickNext, [n, bins + 1], device);
        Tensor Pick(Tensor values, Tensor rows) => (values * rows).Sum(1, keepDim: true);   // [N, 1]

        var inputCumWidth = Pick(cumWidths, atEdge);
        var inputBinWidth = Pick(binWidths, inBin);
        var inputCumHeight = Pick(cumHeights, atEdge);
        var inputHeight = Pick(binHeights, inBin);
        var inputDelta = Pick(binHeights * binWidths.Pow(-1f), inBin);
        var d0 = Pick(slopes, atEdge);
        var d1 = Pick(slopes, atNext);

        var theta = (at - inputCumWidth) * inputBinWidth.Pow(-1f);
        var rest = theta * -1f + 1f;
        var thetaRest = theta * rest;
        var numerator = inputHeight * (inputDelta * theta.Square() + d0 * thetaRest);
        var denominator = inputDelta + (d0 + d1 - inputDelta * 2f) * thetaRest;
        var outputs = inputCumHeight + numerator * denominator.Pow(-1f);
        var derivativeNumerator = inputDelta.Square() * (d1 * theta.Square() + inputDelta * 2f * thetaRest + d0 * rest.Square());
        var logAbsDet = derivativeNumerator.Log() - denominator.Log() * 2f;
        return (outputs * keep + x * pass, logAbsDet * keep);
    }

    // softmax, the minimum added, cumulated (a product with the upper triangle of ones), stretched to [-tail, tail] with
    // exact ends: the cumulative edges [N, bins + 1] and the bins' sizes [N, bins].
    private static (Tensor Cumulative, Tensor Sizes) Cumulative(Tensor unnormalized, float minimum, float tail)
    {
        int n = unnormalized.Shape[0], bins = unnormalized.Shape[1];
        var device = unnormalized.Device;
        var sizes = unnormalized.Softmax() * (1f - minimum * bins) + minimum;
        var triangle = new float[bins * (bins - 1)];
        for (int i = 0; i < bins; i++)
        {
            for (int j = i; j < bins - 1; j++)
            {
                triangle[i * (bins - 1) + j] = 1f;
            }
        }

        var running = sizes.MatMul(Tensor.From(triangle, [bins, bins - 1], device));   // [N, bins - 1]: the sums up to each inner edge
        var lower = Tensor.From(Enumerable.Repeat(-tail, n).ToArray(), [n, 1], device);
        var upper = Tensor.From(Enumerable.Repeat(tail, n).ToArray(), [n, 1], device);
        var cumulative = Tensor.Concat([lower, running * (2f * tail) - tail, upper], 1);
        return (cumulative, cumulative.Narrow(1, 1, bins) - cumulative.Narrow(1, 0, bins));
    }

    /// <summary>
    /// The input whose spline value is <paramref name="y"/>. <paramref name="derivatives"/> holds bins + 1 values whose
    /// first and last are overwritten with the constant that makes the edge derivatives 1.
    /// </summary>
    public static double Inverse(double y, ReadOnlySpan<double> unnormalizedWidths, ReadOnlySpan<double> unnormalizedHeights, Span<double> derivatives, double tail)
    {
        if (y < -tail || y > tail)
        {
            return y;
        }

        int bins = unnormalizedWidths.Length;
        double constant = Math.Log(Math.Exp(1 - MinDerivative) - 1);
        derivatives[0] = constant;
        derivatives[bins] = constant;

        Span<double> cumWidths = stackalloc double[bins + 1];
        Span<double> cumHeights = stackalloc double[bins + 1];
        Cumulative(unnormalizedWidths, MinBinWidth, tail, cumWidths);
        Cumulative(unnormalizedHeights, MinBinHeight, tail, cumHeights);

        // The bin: the last cumulative height the input reaches (the top edge nudged so the bound falls inside the last bin).
        cumHeights[bins] += 1e-6;
        int bin = -1;
        for (int k = 0; k <= bins; k++)
        {
            if (y >= cumHeights[k])
            {
                bin++;
            }
        }

        bin = Math.Clamp(bin, 0, bins - 1);
        cumHeights[bins] -= 1e-6;
        double width = cumWidths[bin + 1] - cumWidths[bin], height = cumHeights[bin + 1] - cumHeights[bin];
        double delta = height / width;
        double d0 = MinDerivative + Softplus(derivatives[bin]), d1 = MinDerivative + Softplus(derivatives[bin + 1]);
        double intermediate1 = d0 + d1 - 2 * delta;
        double intermediate2 = y - cumHeights[bin];
        double intermediate3 = intermediate2 * intermediate1;
        double a = height * (delta - d0) + intermediate3;
        double b = height * d0 - intermediate3;
        double c = -delta * intermediate2;
        double discriminant = b * b - 4 * a * c;
        if (discriminant < 0)
        {
            throw new InvalidDataException($"The duration spline has no inverse at {y} (discriminant {discriminant}).");
        }

        double root = 2 * c / (-b - Math.Sqrt(discriminant));
        return root * width + cumWidths[bin];
    }

    // softmax, the minimum width added, cumulated from 0, stretched to [-tail, tail] with exact ends.
    private static void Cumulative(ReadOnlySpan<double> unnormalized, double minimum, double tail, Span<double> cumulative)
    {
        int bins = unnormalized.Length;
        double max = double.NegativeInfinity;
        foreach (double v in unnormalized)
        {
            max = Math.Max(max, v);
        }

        double sum = 0;
        for (int k = 0; k < bins; k++)
        {
            sum += Math.Exp(unnormalized[k] - max);
        }

        cumulative[0] = -tail;
        double running = 0;
        for (int k = 0; k < bins; k++)
        {
            running += minimum + (1 - minimum * bins) * (Math.Exp(unnormalized[k] - max) / sum);
            cumulative[k + 1] = 2 * tail * running - tail;
        }

        cumulative[bins] = tail;
    }

    private static double Softplus(double x) => x > 20 ? x : Math.Log(1 + Math.Exp(x));
}

/// <summary>Standard normal draws from a seeded <see cref="Random"/> (Box-Muller), so a seed gives the same sound.</summary>
internal static class VitsNoise
{
    public static double Normal(Random random)
    {
        double u1 = 1.0 - random.NextDouble(), u2 = random.NextDouble();
        return Math.Sqrt(-2 * Math.Log(u1)) * Math.Cos(2 * Math.PI * u2);
    }
}
