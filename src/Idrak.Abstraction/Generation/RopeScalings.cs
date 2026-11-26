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

/// <summary>What a RoPE scaling method receives (<see cref="IRopeScalingMethod.Scale"/>).</summary>
/// <param name="Frequencies">The unscaled rotation frequency of each pair, θ^(-2i / rotaryDim) (a copy the method may change and return).</param>
/// <param name="Theta">The base θ of the frequencies.</param>
/// <param name="RotaryDim">The dimensions of each head that rotate (twice the number of frequencies).</param>
/// <param name="Parameters">The scaling's parameters (<see cref="RopeScaling.Parameters"/>).</param>
public sealed record RopeScalingInput(double[] Frequencies, double Theta, int RotaryDim, JsonObject Parameters);

/// <summary>
/// A RoPE scaling method, defined by its own class: the names a configuration calls it by (Hugging Face's
/// <c>rope_scaling.rope_type</c>, <see cref="Names"/>) and the scaled frequencies (and attention factor) it makes from the
/// unscaled ones and its parameters (<see cref="Scale"/>). A configuration's name is matched against the methods' own
/// names only where the configuration comes in (<see cref="RopeScalings.Read"/>), by walking a fixed list
/// (<see cref="RopeScalings.All"/>, the library's; a family adds its own, as Llama 3's); a loaded model holds the method
/// itself (<see cref="RopeScaling.Method"/>), never its name.
/// </summary>
public interface IRopeScalingMethod
{
    /// <summary>The names a configuration's rope_scaling.rope_type calls the method by, the first its own ("linear").</summary>
    IReadOnlyList<string> Names { get; }

    /// <summary>The scaled frequencies of <paramref name="input"/> (it may scale its array in place and return it).</summary>
    RopeScalingResult Scale(RopeScalingInput input);
}

/// <summary>
/// How rotary position embeddings stretch their frequencies for contexts longer than the model was trained on: the method
/// itself (<see cref="Method"/>, an <see cref="IRopeScalingMethod"/>) and its parameters, as Hugging Face's
/// <c>rope_scaling</c> holds them (for example <c>{"factor": 4, "original_max_position_embeddings": 32768}</c>). Two
/// scalings are equal when their methods are the same object and their parameters are equal.
/// </summary>
public sealed record RopeScaling
{
    private readonly JsonObject _parameters;

    /// <summary>A scaling by <paramref name="method"/> with <paramref name="parameters"/> (copied; none when null).</summary>
    public RopeScaling(IRopeScalingMethod method, JsonObject? parameters = null)
    {
        ArgumentNullException.ThrowIfNull(method);
        Method = method;
        _parameters = parameters is null ? [] : (JsonObject)parameters.DeepClone();
    }

    /// <summary>The method.</summary>
    public IRopeScalingMethod Method { get; }

    /// <summary>The method's own name (its first of <see cref="IRopeScalingMethod.Names"/>), as a configuration writes it.</summary>
    public string Name => Method.Names[0];

    /// <summary>The method's parameters (a copy: changing it does not change the scaling).</summary>
    public JsonObject Parameters => (JsonObject)_parameters.DeepClone();

    /// <summary>Positions divided by <paramref name="factor"/> (every frequency divided by it).</summary>
    public static RopeScaling Linear(double factor) => new(LinearRopeScaling.Method, new JsonObject { ["factor"] = factor });

    /// <summary>
    /// YaRN: frequencies that turn fewer than <paramref name="betaSlow"/> times over the original context are divided by the
    /// factor, those turning more than <paramref name="betaFast"/> times are kept, with a linear ramp between; the rotary
    /// tables are multiplied by the attention factor (0.1 · ln(factor) + 1 when null).
    /// </summary>
    public static RopeScaling Yarn(double factor, int originalMaxPositions, double? attentionFactor = null, double betaFast = 32, double betaSlow = 1)
    {
        var parameters = new JsonObject
        {
            ["factor"] = factor, ["original_max_position_embeddings"] = originalMaxPositions, ["beta_fast"] = betaFast, ["beta_slow"] = betaSlow,
        };
        if (attentionFactor is { } attention)
        {
            parameters["attention_factor"] = attention;
        }

        return new(YarnRopeScaling.Method, parameters);
    }

    /// <summary>
    /// Dynamic NTK scaling: within <paramref name="maxPositions"/> the frequencies are unchanged; beyond, the base grows with
    /// the position (see <see cref="RopeScalings"/> for how Idrak applies it).
    /// </summary>
    public static RopeScaling Dynamic(double factor, int maxPositions) =>
        new(DynamicRopeScaling.Method, new JsonObject { ["factor"] = factor, ["max_position_embeddings"] = maxPositions });

    /// <inheritdoc />
    public bool Equals(RopeScaling? other) => other is not null && ReferenceEquals(Method, other.Method) && JsonNode.DeepEquals(_parameters, other._parameters);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(RuntimeHelpers.GetHashCode(Method), _parameters.Count);

    /// <inheritdoc />
    public override string ToString() => $"RopeScaling({Name}, {_parameters.ToJsonString()})";

    /// <summary>
    /// A number parameter of <paramref name="parameters"/> (whatever number type it holds), or <paramref name="fallback"/>
    /// when it is missing or null; for scaling methods reading their parameters. Without a fallback a missing parameter
    /// throws <see cref="ArgumentException"/> naming it.
    /// </summary>
    public static double Number(JsonObject parameters, string key, double? fallback = null)
    {
        ArgumentNullException.ThrowIfNull(parameters);
        return parameters[key] is JsonValue value && value.GetValueKind() == System.Text.Json.JsonValueKind.Number
            ? double.Parse(value.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture)   // whatever number type it holds
            : fallback ?? throw new ArgumentException($"RoPE scaling needs the parameter '{key}'.");
    }
}

/// <summary>"linear" (transformers' <c>_compute_linear_scaling_rope_parameters</c>): every frequency divided by its factor.</summary>
public sealed class LinearRopeScaling : IRopeScalingMethod
{
    private LinearRopeScaling()
    {
    }

    /// <summary>The method.</summary>
    public static IRopeScalingMethod Method { get; } = new LinearRopeScaling();

    /// <inheritdoc />
    public IReadOnlyList<string> Names { get; } = ["linear"];

    /// <inheritdoc />
    public RopeScalingResult Scale(RopeScalingInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        return RopeScalings.Linear(input.Frequencies, RopeScaling.Number(input.Parameters, "factor"));
    }
}

/// <summary>
/// "yarn" (YaRN, transformers' <c>_compute_yarn_parameters</c>): its parameters factor, original_max_position_embeddings
/// (else max_position_embeddings), beta_fast, beta_slow, truncate, attention_factor, mscale and mscale_all_dim
/// (<see cref="RopeScalings.Yarn"/>).
/// </summary>
public sealed class YarnRopeScaling : IRopeScalingMethod
{
    private YarnRopeScaling()
    {
    }

    /// <summary>The method.</summary>
    public static IRopeScalingMethod Method { get; } = new YarnRopeScaling();

    /// <inheritdoc />
    public IReadOnlyList<string> Names { get; } = ["yarn"];

    /// <inheritdoc />
    public RopeScalingResult Scale(RopeScalingInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var p = input.Parameters;
        double original = p["original_max_position_embeddings"] is not null ? RopeScaling.Number(p, "original_max_position_embeddings")
            : RopeScaling.Number(p, "max_position_embeddings");
        bool truncate = p["truncate"] is not JsonValue t || !t.TryGetValue<bool>(out var flag) || flag;
        return RopeScalings.Yarn(input.Frequencies, input.Theta, input.RotaryDim, RopeScaling.Number(p, "factor"), original, RopeScaling.Number(p, "beta_fast", 0),
            RopeScaling.Number(p, "beta_slow", 0), truncate, p["attention_factor"] is not null ? RopeScaling.Number(p, "attention_factor") : null,
            RopeScaling.Number(p, "mscale", 0), RopeScaling.Number(p, "mscale_all_dim", 0));
    }
}

/// <summary>"dynamic" (dynamic NTK, transformers' <c>_compute_dynamic_ntk_parameters</c>): its parameters factor and max_position_embeddings.</summary>
public sealed class DynamicRopeScaling : IRopeScalingMethod
{
    private DynamicRopeScaling()
    {
    }

    /// <summary>The method.</summary>
    public static IRopeScalingMethod Method { get; } = new DynamicRopeScaling();

    /// <inheritdoc />
    public IReadOnlyList<string> Names { get; } = ["dynamic"];

    /// <inheritdoc />
    public RopeScalingResult Scale(RopeScalingInput input)
    {
        ArgumentNullException.ThrowIfNull(input);
        var p = input.Parameters;
        return RopeScalings.Dynamic(input.Frequencies, input.RotaryDim, RopeScaling.Number(p, "factor"), RopeScaling.Number(p, "max_position_embeddings"));
    }
}

/// <summary>
/// The library's RoPE scaling methods, as Hugging Face's <c>rope_scaling.rope_type</c> names them: "linear", "yarn" and
/// "dynamic" (<see cref="All"/>, each its own class), their computations taking their parameters as arguments, and the
/// reading of a configuration's rope_scaling by the methods' own names where it comes in (<see cref="Read"/>).
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
    /// <summary>The library's methods, one immutable list in source (no lookup in it, nothing added): linear, yarn, dynamic.</summary>
    public static IReadOnlyList<IRopeScalingMethod> All { get; } = Array.AsReadOnly<IRopeScalingMethod>([LinearRopeScaling.Method, YarnRopeScaling.Method, DynamicRopeScaling.Method]);

    /// <summary>
    /// A configuration's <c>rope_scaling</c> (or <c>rope_parameters</c>) as a scaling, where it comes in: its
    /// <c>rope_type</c> (or <c>type</c>) compared with each of <paramref name="methods"/>' own names in turn (ignoring case),
    /// its other keys the parameters, with <paramref name="maxPositions"/> (the model's context length, which several
    /// methods fall back on) as max_position_embeddings unless given. Null for none or "default".
    /// </summary>
    /// <param name="ropeScaling">The configuration's rope_scaling, or null.</param>
    /// <param name="maxPositions">The model's context length (max_position_embeddings).</param>
    /// <param name="methods">The methods to walk (<see cref="All"/>, or a family's list that adds its own).</param>
    /// <exception cref="NotSupportedException">No method states the configuration's name.</exception>
    public static RopeScaling? Read(JsonObject? ropeScaling, int maxPositions, IReadOnlyList<IRopeScalingMethod> methods)
    {
        ArgumentNullException.ThrowIfNull(methods);
        if (ropeScaling is null || ((string?)ropeScaling["rope_type"] ?? (string?)ropeScaling["type"] ?? "default") is not { } type
            || type.Equals("default", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        var parameters = (JsonObject)ropeScaling.DeepClone();
        parameters.Remove("type");
        parameters.Remove("rope_type");
        parameters["max_position_embeddings"] ??= maxPositions;
        return new RopeScaling(Named(type, methods), parameters);
    }

    /// <summary>
    /// The method of <paramref name="methods"/> that states <paramref name="name"/> (a configuration's or a saved spec's
    /// text) among its own names, ignoring case: the list walked in turn where the text comes in.
    /// </summary>
    /// <exception cref="NotSupportedException">No method states the name.</exception>
    public static IRopeScalingMethod Named(string name, IReadOnlyList<IRopeScalingMethod> methods)
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(methods);
        foreach (var method in methods)
        {
            foreach (string own in method.Names)
            {
                if (own.Equals(name, StringComparison.OrdinalIgnoreCase))
                {
                    return method;
                }
            }
        }

        throw new NotSupportedException($"The RoPE scaling '{name}' is none of the methods handed over ({string.Join(", ", methods.SelectMany(m => m.Names))}); "
            + "a family whose models scale their own way hands over its method (Idrak.Text.Plugins' Llama gives \"llama3\"), or remove rope_scaling to use the model within its original context.");
    }

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

    /// <summary>Frequency scaling for long contexts (the method itself and its parameters, <see cref="RopeScaling"/>), or null.</summary>
    public RopeScaling? Scaling { get; init; }

    /// <summary>The rotation frequency of each pair (after scaling; for a scaling whose frequencies depend on the position, those of the first positions).</summary>
    public double[] Frequencies(int headDim) => Scaled(headDim).Frequencies;

    /// <summary>
    /// The frequencies of each pair and the factor on the cos and sin tables, after <see cref="Scaling"/> (computed by
    /// its method, <see cref="IRopeScalingMethod.Scale"/>).
    /// </summary>
    public RopeScalingResult Scaled(int headDim)
    {
        int rotary = RotaryDim ?? headDim, half = rotary / 2;
        var frequencies = new double[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = 1.0 / PortableMath.Pow(Theta, 2.0 * i / rotary);
        }

        if (Scaling is not { } scaling)
        {
            return new RopeScalingResult(frequencies);
        }

        var result = scaling.Method.Scale(new RopeScalingInput(frequencies, Theta, rotary, scaling.Parameters));
        if (result.Frequencies.Length != half)
        {
            throw new InvalidOperationException($"RoPE scaling '{scaling.Name}' returned {result.Frequencies.Length} frequencies; {half} expected.");
        }

        return result;
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
