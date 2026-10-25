// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

/// <summary>
/// How rotary position embeddings stretch their frequencies for contexts longer than the model was trained on: the name
/// of a method registered in <see cref="RopeScalings"/> ("linear", "yarn", "dynamic", a plug-in's, or one of your own) and
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

    /// <summary>
    /// A number parameter of <paramref name="parameters"/> (whatever number type it holds), or <paramref name="fallback"/>
    /// when it is missing or null; for scaling methods reading their parameters. Without a fallback a missing parameter
    /// throws <see cref="ArgumentException"/> naming it.
    /// </summary>
    public static double Number(JsonObject parameters, string key, double? fallback = null) =>
        parameters[key] is JsonValue value && value.GetValueKind() == System.Text.Json.JsonValueKind.Number
            ? double.Parse(value.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture)   // whatever number type it holds
            : fallback ?? throw new ArgumentException($"RoPE scaling needs the parameter '{key}'.");
}

/// <summary>What a RoPE scaling receives (<see cref="IRopeScaling"/>).</summary>
/// <param name="Frequencies">The unscaled rotation frequency of each pair, θ^(-2i / rotaryDim) (a copy the method may change and return).</param>
/// <param name="Theta">The base θ of the frequencies.</param>
/// <param name="RotaryDim">The dimensions of each head that rotate (twice the number of frequencies).</param>
/// <param name="Parameters">The scaling's parameters (<see cref="RopeScaling.Parameters"/>).</param>
public sealed record RopeScalingInput(double[] Frequencies, double Theta, int RotaryDim, JsonObject Parameters);

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
/// A RoPE scaling method, registered by its <see cref="IKind.Name"/> in <see cref="RopeScalings"/> (as Hugging Face's
/// <c>rope_scaling.rope_type</c> names it): the parameters it reads (<see cref="IConfigurableKind.Keys"/>) and the scaled
/// frequencies (and attention factor) it makes from the unscaled ones (<see cref="IKind{TRequest, T}.Create"/>).
/// </summary>
public interface IRopeScaling : IKind<RopeScalingInput, RopeScalingResult>;

/// <summary>
/// The RoPE scaling methods by name (<see cref="RopeScaling.Type"/>, ignoring case), as Hugging Face's
/// <c>rope_scaling.rope_type</c> names them: "linear", "yarn" and "dynamic" are the library's; a family's own scaling is
/// its plug-in's (Idrak.Text.Plugins registers "llama3"); add your own with <see cref="Register"/>, and configurations or
/// GGUF files naming it are read with it. A method registered under a built-in name overrides the library's, which stays
/// behind it as the fallback (<see cref="SlotPolicy"/>).
/// <para>
/// The built-ins follow transformers' <c>modeling_rope_utils.py</c>: "linear" divides every frequency by <c>factor</c>;
/// "yarn" and "dynamic" are described at their methods below. "dynamic" depends on the sequence length: transformers computes one set of frequencies per forward pass from the
/// longest position in it, so a prompt longer than <c>max_position_embeddings</c> run in one pass rotates all its tokens
/// with the frequencies of its last position. Idrak rotates each token with the frequencies of its own position (what
/// transformers does when decoding one token at a time), so the two agree within <c>max_position_embeddings</c> and when
/// decoding token by token, and differ for the earlier tokens of a longer prompt given in one pass.
/// </para>
/// </summary>
public static class RopeScalings
{
    private static readonly KindTable<IRopeScaling> Table = BuiltIn();

    private static KindTable<IRopeScaling> BuiltIn()
    {
        var table = new KindTable<IRopeScaling>(nameof(RopeScalings), "RoPE scaling", "add it with RopeScalings.Register", Guard, StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(new BuiltInScaling("linear", "every frequency divided by factor", new OptionKeys("factor"), Linear));
        table.RegisterDefault(new BuiltInScaling("yarn", "YaRN: slow frequencies divided by factor, fast ones kept, a ramp between, and an attention factor",
            new OptionKeys("factor", "original_max_position_embeddings", "max_position_embeddings", "beta_fast", "beta_slow", "truncate", "attention_factor", "mscale", "mscale_all_dim"), Yarn));
        table.RegisterDefault(new BuiltInScaling("dynamic", "dynamic NTK: the base grows with the position beyond max_position_embeddings",
            new OptionKeys("factor", "max_position_embeddings"), Dynamic));
        return table;
    }

    /// <summary>
    /// Registers <paramref name="scaling"/> under its <see cref="IKind.Name"/> (names ignore case: the name configurations
    /// use, <c>rope_scaling.rope_type</c>). Under a built-in name it shadows the library's method, which stays behind it
    /// (see <see cref="SetPolicy"/>) until <see cref="Unregister"/>.
    /// </summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IRopeScaling scaling) => Table.Register(scaling, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's method <paramref name="type"/> (a built-in name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string type) => Table.Unregister(type);

    /// <summary>The registered method names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>Whether a method is registered as <paramref name="type"/>.</summary>
    public static bool Contains(ReadOnlySpan<char> type) => Table.Contains(type);

    /// <summary>The method registered as <paramref name="type"/> (any case); an app's, guarded by its <see cref="SetPolicy">policy</see>.</summary>
    public static IRopeScaling Get(ReadOnlySpan<char> type) =>
        Table.TryGet(type, out var method) ? method
            : throw new NotSupportedException($"No RoPE scaling '{type}' is registered ({string.Join(", ", Table.Names)}); add it with RopeScalings.Register, "
                + "or reference Idrak.Text.Plugins (a family's own scaling) and call TextPlugins.RegisterAll().");

    /// <summary>The library's method <paramref name="type"/>, whatever an app registered over it (for an app's method to delegate to); null when the library has none.</summary>
    public static IRopeScaling? Default(string type) => Table.Default(type);

    /// <summary>Who registered the method <paramref name="type"/> resolves to: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string type) => Table.Origin(type);

    /// <summary>
    /// What happens when the app's method <paramref name="type"/> fails (<see cref="SlotPolicy.Throw"/> unless set: the error reaches the caller;
    /// <see cref="SlotPolicy.FallBack"/> retries on the library's), or whether it only runs beside it (<see cref="SlotPolicy.Shadow"/>, on <paramref name="shadowRate"/> of the calls).
    /// </summary>
    public static void SetPolicy(string type, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(type, policy, shadowRate);

    // The app's method gets a copy of the frequencies (a method may change them in place), so the library's can still run
    // on the originals; outputs agree when frequencies, attention factor and per-position frequencies do.
    private static IRopeScaling Guard(Slot slot, IRopeScaling app, IRopeScaling library) => new GuardedScaling(slot, app, library);

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
            if (library.FrequenciesAt is { } at && Comparisons.Difference(at.At(position), other.FrequenciesAt!.At(position), 1e-9) is { } moved)
            {
                return $"frequencies at position {position}: {moved}";
            }
        }

        return null;
    }

    // A library method: its parameters' keys and its computation.
    private sealed class BuiltInScaling(string name, string summary, OptionKeys keys, Func<RopeScalingInput, RopeScalingResult> scale) : IRopeScaling
    {
        public string Name => name;

        public string Summary => summary;

        public IReadOnlySet<string> Keys => keys;

        public RopeScalingResult Create(RopeScalingInput request) => scale(request);
    }

    private sealed class GuardedScaling(Slot slot, IRopeScaling app, IRopeScaling library) : IRopeScaling
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public IReadOnlySet<string> Keys => app.Keys;

        public RopeScalingResult Create(RopeScalingInput request)
        {
            var copy = request with { Frequencies = (double[])request.Frequencies.Clone() };
            return slot.Call(() => app.Create(copy), () => library.Create(request), Compare);
        }
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
        return new(unscaled) { FrequenciesAt = new DynamicFrequencies(unscaled, factor, max, dim) };
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
            double growth = Math.Pow(factor * length / max - (factor - 1), (double)dim / (dim - 2));
            var frequencies = new double[unscaled.Length];
            for (int i = 0; i < frequencies.Length; i++)
            {
                frequencies[i] = unscaled[i] * Math.Pow(growth, -2.0 * i / dim);
            }

            return frequencies;
        }
    }
}

/// <summary>Rotary position embedding settings.</summary>
/// <param name="Theta">The base of the frequencies (10000 in the original paper; larger for long-context models).</param>
/// <param name="RotaryDim">Dimensions of each head that rotate (all when null).</param>
/// <param name="Interleaved">Pairs are (2i, 2i+1) instead of (i, i + rotaryDim / 2).</param>
/// <param name="Scaling">Frequency scaling for long contexts, or null.</param>
public sealed record RopeSettings(float Theta, int? RotaryDim = null, bool Interleaved = false, RopeScaling? Scaling = null)
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

        var result = RopeScalings.Get(Scaling.Type).Create(new RopeScalingInput(frequencies, Theta, rotary, Scaling.Parameters));
        if (result.Frequencies.Length != half)
        {
            throw new InvalidOperationException($"RoPE scaling '{Scaling.Type}' returned {result.Frequencies.Length} frequencies; {half} expected.");
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
