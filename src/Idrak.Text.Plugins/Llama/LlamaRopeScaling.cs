// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Text.Plugins.Llama;

/// <summary>
/// Llama 3's RoPE scaling ("llama3" in config.json's rope_scaling.rope_type, as transformers' <c>modeling_rope_utils.py</c>
/// has it): frequencies whose wavelength exceeds original / <c>low_freq_factor</c> are divided by the factor, those under
/// original / <c>high_freq_factor</c> are kept, and those between are blended.
/// </summary>
public static class LlamaRopeScaling
{
    /// <summary>config.json's rope_scaling.rope_type of the scaling: "llama3".</summary>
    public const string Name = "llama3";

    /// <summary>
    /// Llama 3's scaling of <paramref name="frequencies"/>, in place: wavelengths longer than original / low stretched by the
    /// factor, shorter than original / high kept, a smooth blend between.
    /// </summary>
    // transformers _compute_llama3_parameters: wavelength = 2π / f; f / factor above original / low_freq_factor, f below
    // original / high_freq_factor, else (1 - s) · f / factor + s · f with s = (original / wavelength - low) / (high - low).
    public static RopeScalingResult Scale(double[] frequencies, double factor, double lowFrequencyFactor = 1, double highFrequencyFactor = 4,
        double originalMaxPositions = 8192)
    {
        ArgumentNullException.ThrowIfNull(frequencies);
        double low = lowFrequencyFactor, high = highFrequencyFactor, original = originalMaxPositions;
        double lowWavelength = original / low, highWavelength = original / high;
        for (int i = 0; i < frequencies.Length; i++)
        {
            double wavelength = 2 * Math.PI / frequencies[i];
            if (wavelength > lowWavelength)
            {
                frequencies[i] /= factor;
            }
            else if (wavelength >= highWavelength)
            {
                double smooth = (original / wavelength - low) / (high - low);
                frequencies[i] = (1 - smooth) * frequencies[i] / factor + smooth * frequencies[i];
            }
        }

        return new(frequencies);
    }
}
