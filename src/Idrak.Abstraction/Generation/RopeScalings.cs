// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>What a RoPE scaling returns.</summary>
/// <param name="Frequencies">The scaled frequency of each pair.</param>
/// <param name="AttentionFactor">
/// A factor on the rotary cos and sin tables (YaRN's attention scaling, mscale), so every query·key product of the rotated
/// dimensions is scaled by its square; 1 for none.
/// </param>
public sealed record RopeScalingResult(double[] Frequencies, double AttentionFactor = 1.0)
{
    /// <summary>
    /// The frequencies used for the token at a position, when they depend on it (dynamic NTK scaling); null when every
    /// position uses <see cref="Frequencies"/>. Asked once per position, possibly from several threads at once, when the rotary tables are made.
    /// </summary>
    public IPositionFrequencies? FrequenciesAt { get; init; }
}

/// <summary>Rotary frequencies that depend on the token's position (<see cref="RopeScalingResult.FrequenciesAt"/>).</summary>
public interface IPositionFrequencies
{
    /// <summary>The frequency of each pair for the token at <paramref name="position"/>; safe to call from several threads at once.</summary>
    double[] At(int position);
}

/// <summary>
/// The library's RoPE scaling methods, as Hugging Face's <c>rope_scaling.rope_type</c> names them: "linear", "yarn" and
/// "dynamic", each taking its parameters as arguments.
/// <para>
/// They follow transformers' <c>modeling_rope_utils.py</c>: "linear" divides every frequency by <c>factor</c>;
/// "yarn" and "dynamic" are described at their methods below. "dynamic" depends on the sequence length: transformers computes one set of frequencies per forward pass from the
/// longest position in it, so a prompt longer than <c>max_position_embeddings</c> run in one pass rotates all its tokens
/// with the frequencies of its last position. Idrak rotates each token with the frequencies of its own position (what
/// transformers does when decoding one token at a time), so the two agree within <c>max_position_embeddings</c> and when
/// decoding token by token, and differ for the earlier tokens of a longer prompt given in one pass.
/// </para>
/// </summary>
public static class RopeScalings
{
    /// <summary>"linear" (transformers' <c>_compute_linear_scaling_rope_parameters</c>): every frequency divided by <paramref name="factor"/>, in place.</summary>
    public static RopeScalingResult Linear(double[] frequencies, double factor)
    {
        ArgumentNullException.ThrowIfNull(frequencies);
        for (int i = 0; i < frequencies.Length; i++)
        {
            frequencies[i] /= factor;
        }

        return new(frequencies);
    }

    // transformers _compute_yarn_parameters (YaRN, arXiv 2309.00071), with d the rotary dimensions, b the base θ, L the
    // original context (original_max_position_embeddings, else max_position_embeddings):
    //   correction(r) = d · ln(L / (2π r)) / (2 ln b)
    //   low = max(floor(correction(beta_fast)), 0), high = min(ceil(correction(beta_slow)), d - 1)   (no floor/ceil when truncate is false)
    //   ramp_i = clamp((i - low) / (high - low), 0, 1)   (high += 0.001 when low == high)
    //   f_i = (f_i / factor) · ramp_i + f_i · (1 - ramp_i)
    //   attention factor = attention_factor, else get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim) when both
    //   are given, else get_mscale(factor), where get_mscale(s, m) = 0.1 · m · ln(s) + 1 (1 for s ≤ 1).
    // beta_fast and beta_slow default to 32 and 1 (also when 0, as transformers' `or` does).
    /// <summary>"yarn" (YaRN): the slow frequencies divided by <paramref name="factor"/>, the fast ones kept, a ramp between, and an attention factor; in place.</summary>
    public static RopeScalingResult Yarn(double[] frequencies, double theta, int rotaryDim, double factor, double originalMaxPositions, double betaFast = 32,
        double betaSlow = 1, bool truncate = true, double? attentionFactor = null, double mscale = 0, double mscaleAllDim = 0)
    {
        ArgumentNullException.ThrowIfNull(frequencies);
        double original = originalMaxPositions;
        betaFast = betaFast != 0 ? betaFast : 32;
        betaSlow = betaSlow != 0 ? betaSlow : 1;

        static double MScale(double scale, double m = 1) => scale <= 1 ? 1.0 : 0.1 * m * PortableMath.Log(scale) + 1.0;
        double attention;
        if (attentionFactor is { } given)
        {
            attention = given;
        }
        else
        {
            attention = mscale != 0 && mscaleAllDim != 0 ? MScale(factor, mscale) / MScale(factor, mscaleAllDim) : MScale(factor);
        }

        int dim = rotaryDim;
        double Correction(double rotations) => dim * PortableMath.Log(original / (rotations * 2 * Math.PI)) / (2 * PortableMath.Log(theta));
        double low = Correction(betaFast), high = Correction(betaSlow);
        if (truncate)
        {
            (low, high) = (Math.Floor(low), Math.Ceiling(high));
        }

        (low, high) = (Math.Max(low, 0), Math.Min(high, dim - 1));
        if (low == high)
        {
            high += 0.001;                                                   // as transformers: no division by zero
        }

        for (int i = 0; i < frequencies.Length; i++)
        {
            double ramp = Math.Clamp((i - low) / (high - low), 0.0, 1.0);
            double extrapolation = 1 - ramp;
            frequencies[i] = frequencies[i] / factor * (1 - extrapolation) + frequencies[i] * extrapolation;
        }

        return new(frequencies, attention);
    }

    // transformers _compute_dynamic_ntk_parameters, with d the rotary dimensions, M = max_position_embeddings and the
    // sequence length s = max(position + 1, M):
    //   b' = b · (factor · s / M - (factor - 1)) ^ (d / (d - 2)),   f_i = b' ^ (-2i / d)
    // Within M the base is unchanged. Each position uses its own s (see the class remarks).
    /// <summary>"dynamic" (dynamic NTK): the base grows with the position beyond <paramref name="maxPositions"/>.</summary>
    public static RopeScalingResult Dynamic(double[] frequencies, int rotaryDim, double factor, double maxPositions)
    {
        ArgumentNullException.ThrowIfNull(frequencies);
        return new(frequencies) { FrequenciesAt = new DynamicFrequencies(frequencies, factor, maxPositions, rotaryDim) };
    }

    // Dynamic NTK's frequencies at a position: the unscaled ones within max_position_embeddings, the grown base beyond.
    private sealed class DynamicFrequencies(double[] unscaled, double factor, double max, int dim) : IPositionFrequencies
    {
        public double[] At(int position)
        {
            double length = position + 1;
            if (length <= max)
            {
                return unscaled;
            }

            // f_i · (b' / b)^(-2i / d): the unscaled frequencies with the grown base.
            double growth = PortableMath.Pow(factor * length / max - (factor - 1), (double)dim / (dim - 2));
            var frequencies = new double[unscaled.Length];
            for (int i = 0; i < frequencies.Length; i++)
            {
                frequencies[i] = unscaled[i] * PortableMath.Pow(growth, -2.0 * i / dim);
            }

            return frequencies;
        }
    }
}

/// <summary>Rotary position embedding settings.</summary>
/// <param name="Theta">The base of the frequencies (10000 in the original paper; larger for long-context models).</param>
/// <param name="RotaryDim">Dimensions of each head that rotate (all when null).</param>
/// <param name="Interleaved">Pairs are (2i, 2i+1) instead of (i, i + rotaryDim / 2).</param>
public sealed record RopeSettings(float Theta, int? RotaryDim = null, bool Interleaved = false)
{
    /// <summary>
    /// Several position axes (M-RoPE, as Qwen2-VL and Qwen2.5-VL rotate), or null for one: each frequency pair rotates by
    /// the position on the axis <see cref="RotarySections"/> gives it. Text tokens hold the same position on every axis,
    /// so a model with sections reads text exactly as without them; a prompt with images whose tokens carry axis positions
    /// (<see cref="ImageFeatures.Positions"/>) rotates each pair by its own axis (<see cref="ImagePositionRules"/>).
    /// </summary>
    public RotarySections? Sections { get; init; }

    /// <summary>The rotation frequency of each pair (after scaling; for a scaling whose frequencies depend on the position, those of the first positions).</summary>
    public double[] Frequencies(int headDim) => Scaled(headDim).Frequencies;

    /// <summary>
    /// The frequencies of each pair and the factor on the cos and sin tables.
    /// </summary>
    public RopeScalingResult Scaled(int headDim)
    {
        int rotary = RotaryDim ?? headDim, half = rotary / 2;
        var frequencies = new double[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = 1.0 / PortableMath.Pow(Theta, 2.0 * i / rotary);
        }

        return new RopeScalingResult(frequencies);
    }
}

/// <summary>
/// How the frequency pairs of a rotary embedding split over several position axes (M-RoPE): <see cref="Pairs"/>[a]
/// consecutive pairs take their position from axis a, in order (Qwen2-VL's <c>mrope_section</c>, for example
/// [16, 24, 24]: time, height, width over a head of 128 dimensions). Compared by value.
/// </summary>
public sealed class RotarySections : IEquatable<RotarySections>
{
    private readonly int[] _pairs;

    /// <summary>The pairs per axis, in axis order.</summary>
    /// <exception cref="ArgumentException">No axis, or an axis without pairs.</exception>
    public RotarySections(params int[] pairs)
    {
        ArgumentNullException.ThrowIfNull(pairs);
        if (pairs.Length == 0 || pairs.Any(p => p <= 0))
        {
            throw new ArgumentException($"Rotary sections are one positive count of frequency pairs per axis, got [{string.Join(", ", pairs)}].", nameof(pairs));
        }

        _pairs = [.. pairs];
    }

    /// <summary>The frequency pairs of each axis.</summary>
    public IReadOnlyList<int> Pairs => _pairs;

    /// <summary>The number of position axes.</summary>
    public int Axes => _pairs.Length;

    /// <summary>All the pairs: half the rotated dimensions of a head.</summary>
    public int Total => _pairs.Sum();

    /// <summary>The axis of each frequency pair, [<see cref="Total"/>].</summary>
    public int[] AxisOfPairs()
    {
        var axes = new int[Total];
        int at = 0;
        for (int a = 0; a < _pairs.Length; a++)
        {
            axes.AsSpan(at, _pairs[a]).Fill(a);
            at += _pairs[a];
        }

        return axes;
    }

    /// <inheritdoc />
    public bool Equals(RotarySections? other) => other is not null && _pairs.AsSpan().SequenceEqual(other._pairs);

    /// <inheritdoc />
    public override bool Equals(object? obj) => Equals(obj as RotarySections);

    /// <inheritdoc />
    public override int GetHashCode()
    {
        var hash = new HashCode();
        foreach (int p in _pairs)
        {
            hash.Add(p);
        }

        return hash.ToHashCode();
    }

    /// <inheritdoc />
    public override string ToString() => $"[{string.Join(", ", _pairs)}]";
}
