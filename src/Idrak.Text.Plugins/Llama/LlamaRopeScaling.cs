// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Text.Plugins.Llama;

/// <summary>
/// Llama 3's RoPE scaling ("llama3" in <see cref="RopeScalings"/>, as transformers' <c>modeling_rope_utils.py</c> has it):
/// frequencies whose wavelength exceeds original / <c>low_freq_factor</c> are divided by the factor, those under
/// original / <c>high_freq_factor</c> are kept, and those between are blended.
/// </summary>
public static class LlamaRopeScaling
{
    /// <summary>The scaling's name in <see cref="RopeScalings"/> and config.json's rope_scaling.rope_type: "llama3".</summary>
    public const string Name = "llama3";

    /// <summary>Llama 3's scaling: wavelengths longer than original / low stretched by the factor, shorter than original / high kept, a smooth blend between.</summary>
    public static RopeScaling Create(double factor, double lowFrequencyFactor = 1, double highFrequencyFactor = 4, int originalMaxPositions = 8192) =>
        new(Name, new JsonObject
        {
            ["factor"] = factor, ["low_freq_factor"] = lowFrequencyFactor, ["high_freq_factor"] = highFrequencyFactor,
            ["original_max_position_embeddings"] = originalMaxPositions,
        });

    /// <summary>The method (registered as <see cref="Name"/> by <see cref="LlamaPlugin"/>).</summary>
    // transformers _compute_llama3_parameters: wavelength = 2π / f; f / factor above original / low_freq_factor, f below
    // original / high_freq_factor, else (1 - s) · f / factor + s · f with s = (original / wavelength - low) / (high - low).
    public static RopeScalingResult Scale(RopeScalingInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var p = input.Parameters;
        double factor = RopeScaling.Number(p, "factor"), low = RopeScaling.Number(p, "low_freq_factor", 1), high = RopeScaling.Number(p, "high_freq_factor", 4);
        double original = RopeScaling.Number(p, "original_max_position_embeddings", 8192);
        double lowWavelength = original / low, highWavelength = original / high;
        var frequencies = input.Frequencies;
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
