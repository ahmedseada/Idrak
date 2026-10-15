// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.VitsSpeech;

/// <summary>How many frames each token lasts: the log-durations of the hidden states [C, T] (one utterance, no mask).</summary>
internal interface IVitsDurationPredictor : IDisposable
{
    /// <summary>The log-durations [T] of <paramref name="hidden"/> [C, T], with the speaker's embedding [E, 1] (null for one speaker) and the noise scale (the stochastic predictor draws from <paramref name="random"/>).</summary>
    float[] LogDurations(Tensor hidden, Tensor? speaker, float noiseScale, Random random);
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
    private readonly ConvFlow[] _flows;

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
    }

    public float[] LogDurations(Tensor hidden, Tensor? speaker, float noiseScale, Random random)
    {
        int t = hidden.Shape[1], channels = _config.DepthSeparableChannels;
        var x = _pre.Forward(hidden);
        if (speaker is not null && _cond is not null)
        {
            x = x + VitsOps.ExpandColumns(_cond.Forward(speaker), t);
        }

        x = _proj.Forward(_dds.Forward(x, null));                                // the conditioning of every flow, [C, T]

        // The latents [channels, T]: noise times the scale (none at scale 0).
        var latents = new float[channels * t];
        if (noiseScale != 0)
        {
            for (int i = 0; i < latents.Length; i++)
            {
                latents[i] = (float)(VitsNoise.Normal(random) * noiseScale);
            }
        }

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
            logDurations[n] = (float)((latents[n] - _translate[0]) * Math.Exp(-_logScale[0]));
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

        x = _norm1.Forward(_conv1.Forward(x).Relu().Transpose()).Transpose();
        x = _norm2.Forward(_conv2.Forward(x).Relu().Transpose()).Transpose();
        return _proj.Forward(x).ToArray();                                       // [1, T]
    }

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
