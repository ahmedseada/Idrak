// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// How rotary position embeddings stretch their frequencies for contexts longer than the model was trained on: the name
/// of a method registered in <see cref="RopeScalings"/> ("linear", "llama3", "yarn", "dynamic", or one of your own) and
/// its parameters, as Hugging Face's <c>rope_scaling</c> holds them (for example <c>{"factor": 4,
/// "original_max_position_embeddings": 32768}</c>). Two scalings are equal when their types (ignoring case) and parameters are.
/// </summary>
public sealed record RopeScaling
{
    private readonly JsonObject _parameters;

    /// <summary>A scaling of the registered <paramref name="type"/> with <paramref name="parameters"/> (copied; none when null).</summary>
    public RopeScaling(string type, JsonObject? parameters = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(type);
        Type = type;
        _parameters = parameters is null ? [] : (JsonObject)parameters.DeepClone();
    }

    /// <summary>The method's name in <see cref="RopeScalings"/>.</summary>
    public string Type { get; }

    /// <summary>The method's parameters (a copy: changing it does not change the scaling).</summary>
    public JsonObject Parameters => (JsonObject)_parameters.DeepClone();

    /// <summary>Positions divided by <paramref name="factor"/> (every frequency divided by it).</summary>
    public static RopeScaling Linear(double factor) => new("linear", new JsonObject { ["factor"] = factor });

    /// <summary>Llama 3's scaling: wavelengths longer than original / low stretched by the factor, shorter than original / high kept, a smooth blend between.</summary>
    public static RopeScaling Llama3(double factor, double lowFrequencyFactor = 1, double highFrequencyFactor = 4, int originalMaxPositions = 8192) =>
        new("llama3", new JsonObject
        {
            ["factor"] = factor, ["low_freq_factor"] = lowFrequencyFactor, ["high_freq_factor"] = highFrequencyFactor,
            ["original_max_position_embeddings"] = originalMaxPositions,
        });

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

        return new("yarn", parameters);
    }

    /// <summary>
    /// Dynamic NTK scaling: within <paramref name="maxPositions"/> the frequencies are unchanged; beyond, the base grows with
    /// the position (see <see cref="RopeScalings"/> for how Idrak applies it).
    /// </summary>
    public static RopeScaling Dynamic(double factor, int maxPositions) =>
        new("dynamic", new JsonObject { ["factor"] = factor, ["max_position_embeddings"] = maxPositions });

    /// <inheritdoc />
    public bool Equals(RopeScaling? other) =>
        other is not null && string.Equals(Type, other.Type, StringComparison.OrdinalIgnoreCase) && JsonNode.DeepEquals(_parameters, other._parameters);

    /// <inheritdoc />
    public override int GetHashCode() => HashCode.Combine(StringComparer.OrdinalIgnoreCase.GetHashCode(Type), _parameters.Count);

    /// <inheritdoc />
    public override string ToString() => $"RopeScaling({Type}, {_parameters.ToJsonString()})";

    /// <summary>A number parameter, or <paramref name="fallback"/> when it is missing or null.</summary>
    internal static double Number(JsonObject parameters, string key, double? fallback = null) =>
        parameters[key] is JsonValue value && value.GetValueKind() == System.Text.Json.JsonValueKind.Number
            ? double.Parse(value.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture)   // whatever number type it holds
            : fallback ?? throw new ArgumentException($"RoPE scaling needs the parameter '{key}'.");
}

/// <summary>What a RoPE scaling method receives (see <see cref="RopeScalings.Register"/>).</summary>
/// <param name="Frequencies">The unscaled rotation frequency of each pair, θ^(-2i / rotaryDim) (a copy the method may change and return).</param>
/// <param name="Theta">The base θ of the frequencies.</param>
/// <param name="RotaryDim">The dimensions of each head that rotate (twice the number of frequencies).</param>
/// <param name="Parameters">The scaling's parameters (<see cref="RopeScaling.Parameters"/>).</param>
public sealed record RopeScalingInput(double[] Frequencies, double Theta, int RotaryDim, JsonObject Parameters);

/// <summary>What a RoPE scaling method returns.</summary>
/// <param name="Frequencies">The scaled frequency of each pair.</param>
/// <param name="AttentionFactor">
/// A factor on the rotary cos and sin tables (YaRN's attention scaling, mscale), so every query·key product of the rotated
/// dimensions is scaled by its square; 1 for none.
/// </param>
public sealed record RopeScalingResult(double[] Frequencies, double AttentionFactor = 1.0)
{
    /// <summary>
    /// The frequencies used for the token at a position, when they depend on it (dynamic NTK scaling); null when every
    /// position uses <see cref="Frequencies"/>. Called once per position, possibly from several threads at once, when the rotary tables are made.
    /// </summary>
    public Func<int, double[]>? FrequenciesAt { get; init; }
}

/// <summary>Scales rotary frequencies; register it with <see cref="RopeScalings.Register"/>.</summary>
public delegate RopeScalingResult RopeScalingMethod(RopeScalingInput input);

/// <summary>
/// The RoPE scaling methods by name (<see cref="RopeScaling.Type"/>, ignoring case), as Hugging Face's
/// <c>rope_scaling.rope_type</c> names them: "linear", "llama3", "yarn" and "dynamic" are registered; add your own with
/// <see cref="Register"/>, and configurations or GGUF files naming it are read with it. A method registered under a
/// built-in name overrides the library's, which stays behind it as the fallback (<see cref="SlotPolicy"/>).
/// <para>
/// The built-ins follow transformers' <c>modeling_rope_utils.py</c>: "linear" divides every frequency by <c>factor</c>;
/// "llama3" divides frequencies whose wavelength exceeds original / <c>low_freq_factor</c> by the factor, keeps those
/// under original / <c>high_freq_factor</c>, and blends between; "yarn" and "dynamic" are described at their methods
/// below. "dynamic" depends on the sequence length: transformers computes one set of frequencies per forward pass from the
/// longest position in it, so a prompt longer than <c>max_position_embeddings</c> run in one pass rotates all its tokens
/// with the frequencies of its last position. Idrak rotates each token with the frequencies of its own position (what
/// transformers does when decoding one token at a time), so the two agree within <c>max_position_embeddings</c> and when
/// decoding token by token, and differ for the earlier tokens of a longer prompt given in one pass.
/// </para>
/// </summary>
public static class RopeScalings
{
    private static readonly SlotTable<string, RopeScalingMethod> Table = BuiltIn();

    private static SlotTable<string, RopeScalingMethod> BuiltIn()
    {
        var table = new SlotTable<string, RopeScalingMethod>(nameof(RopeScalings), Guard, StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault("linear", Linear);
        table.RegisterDefault("llama3", Llama3);
        table.RegisterDefault("yarn", Yarn);
        table.RegisterDefault("dynamic", Dynamic);
        return table;
    }

    /// <summary>
    /// Registers the scaling method <paramref name="type"/> (names ignore case). Under a built-in name it shadows the
    /// library's method, which stays as its fallback (see <see cref="SetPolicy"/>) until <see cref="Unregister"/>.
    /// </summary>
    /// <param name="type">The name configurations use (<c>rope_scaling.rope_type</c>).</param>
    /// <param name="method">The frequencies (and attention factor) from the unscaled ones and the parameters.</param>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(string type, RopeScalingMethod method)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(type);
        ArgumentNullException.ThrowIfNull(method);
        Table.Register(type, method, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's method <paramref name="type"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string type) => Table.Unregister(type);

    /// <summary>The registered method names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>Whether a method is registered as <paramref name="type"/>.</summary>
    public static bool Contains(string type) => Table.Contains(type);

    /// <summary>The method registered as <paramref name="type"/> (any case); an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    public static RopeScalingMethod Get(string type) =>
        Table.TryGet(type, out var method) ? method
            : throw new NotSupportedException($"No RoPE scaling '{type}' is registered ({string.Join(", ", Table.Keys)}); add it with RopeScalings.Register.");

    /// <summary>The library's method <paramref name="type"/>, whatever an app registered over it (for an app's method to delegate to); null when the library has none.</summary>
    public static RopeScalingMethod? Default(string type) => Table.Default(type);

    /// <summary>Who registered the method <paramref name="type"/> resolves to: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string type) => Table.Origin(type);

    /// <summary>
    /// What happens when the app's method <paramref name="type"/> fails (<see cref="SlotPolicy.FallBack"/> to the library's
    /// unless set), or whether it only runs beside it (<see cref="SlotPolicy.Shadow"/>, on <paramref name="shadowRate"/> of the calls).
    /// </summary>
    public static void SetPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(type, policy, shadowRate);

    // The app's method gets a copy of the frequencies (a method may change them in place), so the library's can still run
    // on the originals; outputs agree when frequencies, attention factor and per-position frequencies do.
    private static RopeScalingMethod Guard(Slot slot, RopeScalingMethod app, RopeScalingMethod library) => input =>
    {
        var copy = input with { Frequencies = (double[])input.Frequencies.Clone() };
        return slot.Call(() => app(copy), () => library(input), Compare);
    };

    private static string? Compare(RopeScalingResult library, RopeScalingResult other)
    {
        if (Comparisons.Difference(library.Frequencies, other.Frequencies, 1e-9) is { } frequencies)
        {
            return "frequencies: " + frequencies;
        }

        if (Comparisons.Difference([library.AttentionFactor], [other.AttentionFactor], 1e-9) is not null)
        {
            return $"attention factor {library.AttentionFactor} != {other.AttentionFactor}";
        }

        if ((library.FrequenciesAt is null) != (other.FrequenciesAt is null))
        {
            return library.FrequenciesAt is null ? "the override's frequencies depend on the position" : "the override's frequencies do not depend on the position";
        }

        foreach (int position in (int[])[0, 1 << 16])
        {
            if (library.FrequenciesAt is { } at && Comparisons.Difference(at(position), other.FrequenciesAt!(position), 1e-9) is { } moved)
            {
                return $"frequencies at position {position}: {moved}";
            }
        }

        return null;
    }

    // transformers _compute_linear_scaling_rope_parameters: inv_freq /= factor.
    private static RopeScalingResult Linear(RopeScalingInput input)
    {
        double factor = RopeScaling.Number(input.Parameters, "factor");
        var frequencies = input.Frequencies;
        for (int i = 0; i < frequencies.Length; i++)
        {
            frequencies[i] /= factor;
        }

        return new(frequencies);
    }

    // transformers _compute_llama3_parameters: wavelength = 2π / f; f / factor above original / low_freq_factor, f below
    // original / high_freq_factor, else (1 - s) · f / factor + s · f with s = (original / wavelength - low) / (high - low).
    private static RopeScalingResult Llama3(RopeScalingInput input)
    {
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

    // transformers _compute_yarn_parameters (YaRN, arXiv 2309.00071), with d the rotary dimensions, b the base θ, L the
    // original context (original_max_position_embeddings, else max_position_embeddings):
    //   correction(r) = d · ln(L / (2π r)) / (2 ln b)
    //   low = max(floor(correction(beta_fast)), 0), high = min(ceil(correction(beta_slow)), d - 1)   (no floor/ceil when truncate is false)
    //   ramp_i = clamp((i - low) / (high - low), 0, 1)   (high += 0.001 when low == high)
    //   f_i = (f_i / factor) · ramp_i + f_i · (1 - ramp_i)
    //   attention factor = attention_factor, else get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim) when both
    //   are given, else get_mscale(factor), where get_mscale(s, m) = 0.1 · m · ln(s) + 1 (1 for s ≤ 1).
    // beta_fast and beta_slow default to 32 and 1 (also when 0, as transformers' `or` does).
    private static RopeScalingResult Yarn(RopeScalingInput input)
    {
        var p = input.Parameters;
        double factor = RopeScaling.Number(p, "factor");
        double original = p["original_max_position_embeddings"] is not null ? RopeScaling.Number(p, "original_max_position_embeddings")
            : RopeScaling.Number(p, "max_position_embeddings");
        double betaFast = RopeScaling.Number(p, "beta_fast", 0) is var fast && fast != 0 ? fast : 32;
        double betaSlow = RopeScaling.Number(p, "beta_slow", 0) is var slow && slow != 0 ? slow : 1;
        bool truncate = p["truncate"] is not JsonValue t || !t.TryGetValue<bool>(out var flag) || flag;

        static double MScale(double scale, double m = 1) => scale <= 1 ? 1.0 : 0.1 * m * Math.Log(scale) + 1.0;
        double attention;
        if (p["attention_factor"] is not null)
        {
            attention = RopeScaling.Number(p, "attention_factor");
        }
        else
        {
            double mscale = RopeScaling.Number(p, "mscale", 0), mscaleAllDim = RopeScaling.Number(p, "mscale_all_dim", 0);
            attention = mscale != 0 && mscaleAllDim != 0 ? MScale(factor, mscale) / MScale(factor, mscaleAllDim) : MScale(factor);
        }

        int dim = input.RotaryDim;
        double Correction(double rotations) => dim * Math.Log(original / (rotations * 2 * Math.PI)) / (2 * Math.Log(input.Theta));
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

        var frequencies = input.Frequencies;
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
    private static RopeScalingResult Dynamic(RopeScalingInput input)
    {
        var p = input.Parameters;
        double factor = RopeScaling.Number(p, "factor"), max = RopeScaling.Number(p, "max_position_embeddings");
        int dim = input.RotaryDim;
        var unscaled = input.Frequencies;
        return new(unscaled)
        {
            FrequenciesAt = position =>
            {
                double length = position + 1;
                if (length <= max)
                {
                    return unscaled;
                }

                // f_i · (b' / b)^(-2i / d): the unscaled frequencies with the grown base.
                double growth = Math.Pow(factor * length / max - (factor - 1), (double)dim / (dim - 2));
                var frequencies = new double[unscaled.Length];
                for (int i = 0; i < frequencies.Length; i++)
                {
                    frequencies[i] = unscaled[i] * Math.Pow(growth, -2.0 * i / dim);
                }

                return frequencies;
            },
        };
    }
}

/// <summary>Rotary position embedding settings.</summary>
/// <param name="Theta">The base of the frequencies (10000 in the original paper; larger for long-context models).</param>
/// <param name="RotaryDim">Dimensions of each head that rotate (all when null).</param>
/// <param name="Interleaved">Pairs are (2i, 2i+1) instead of (i, i + rotaryDim / 2).</param>
/// <param name="Scaling">Frequency scaling for long contexts, or null.</param>
public sealed record RopeSettings(float Theta, int? RotaryDim = null, bool Interleaved = false, RopeScaling? Scaling = null)
{
    /// <summary>The rotation frequency of each pair (after scaling; for a scaling whose frequencies depend on the position, those of the first positions).</summary>
    public double[] Frequencies(int headDim) => Scaled(headDim).Frequencies;

    /// <summary>
    /// The frequencies of each pair and the factor on the cos and sin tables, after <see cref="Scaling"/> (computed by the
    /// method registered for its type in <see cref="RopeScalings"/>).
    /// </summary>
    public RopeScalingResult Scaled(int headDim)
    {
        int rotary = RotaryDim ?? headDim, half = rotary / 2;
        var frequencies = new double[half];
        for (int i = 0; i < half; i++)
        {
            frequencies[i] = 1.0 / Math.Pow(Theta, 2.0 * i / rotary);
        }

        if (Scaling is null)
        {
            return new RopeScalingResult(frequencies);
        }

        var result = RopeScalings.Get(Scaling.Type)(new RopeScalingInput(frequencies, Theta, rotary, Scaling.Parameters));
        if (result.Frequencies.Length != half)
        {
            throw new InvalidOperationException($"RoPE scaling '{Scaling.Type}' returned {result.Frequencies.Length} frequencies; {half} expected.");
        }

        return result;
    }
}
