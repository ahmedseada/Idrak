// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a RoPE scaling method (<see cref="RopeScalingMethod"/>, registered in <see cref="RopeScalings"/>) for one
/// scaling type: given the unscaled frequencies of a head (a base θ and the rotating dimensions) and the scaling's
/// parameters, it returns as many frequencies, finite and positive, with a finite positive attention factor; it agrees
/// with the library's method of the same type (<see cref="RopeScalings.Default"/>) within <see cref="Tolerance"/>,
/// position by position when the frequencies depend on the position; and its per-position frequencies may be asked for
/// from several threads at once.
/// </summary>
/// <param name="type">The scaling type ("yarn", or a type of your own).</param>
/// <param name="parameterSets">
/// The parameters to try (as <see cref="RopeScaling.Parameters"/>); the built-in types have their own, a type of your own
/// needs some.
/// </param>
public sealed class RopeScalingSuite(string type, IReadOnlyList<JsonObject>? parameterSets = null) : ContractSuite<RopeScalingMethod>
{
    private readonly IReadOnlyList<JsonObject> _parameterSets = parameterSets ?? BuiltInParameters(type)
        ?? throw new ArgumentException($"RoPE scaling '{type}' is not built in: give the parameter sets to check it with.", nameof(parameterSets));

    /// <summary>The scaling type checked.</summary>
    public string Type { get; } = type;

    /// <summary>The relative agreement wanted with the library's method: |f - f_default| ≤ tolerance · f_default.</summary>
    public double Tolerance { get; init; } = 1e-6;

    /// <inheritdoc />
    public override string Name => "rope-scaling";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        foreach (var (theta, rotary) in new[] { (10_000.0, 64), (500_000.0, 128), (1_000_000.0, 80), (10_000.0, 2) })
        {
            foreach (var parameters in _parameterSets)
            {
                yield return Case(theta, rotary, parameters, [0, 1, 2047, 4096, 8191, 8192, 65_535, 131_071]);
            }
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        double theta = Math.Round(Math.Pow(10, 3 + random.NextDouble() * 4));
        int rotary = 2 * random.Next(1, large ? 257 : 65);
        var parameters = (JsonObject)_parameterSets[random.Next(_parameterSets.Count)].DeepClone();
        if (parameters["factor"] is not null)
        {
            parameters["factor"] = Math.Round(1 + random.NextDouble() * 31, 3);
        }

        int[] positions = [.. Enumerable.Range(0, 6).Select(_ => random.Next(0, large ? 1 << 20 : 1 << 17))];
        return Case(theta, rotary, parameters, positions);
    }

    /// <inheritdoc />
    public override void Run(RopeScalingMethod implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        double theta = (double)data["theta"]!;
        int rotary = (int)data["rotaryDim"]!;
        var parameters = data["parameters"]!.AsObject();
        int[] positions = [.. data["positions"]!.AsArray().Select(p => (int)p!)];

        var result = implementation(Input(theta, rotary, parameters));
        int half = rotary / 2;
        checks.Check("as many frequencies as pairs", result.Frequencies.Length == half, $"{result.Frequencies.Length} frequencies, {half} expected");
        checks.Check("frequencies finite and positive", result.Frequencies.All(f => double.IsFinite(f) && f > 0),
            $"frequencies {string.Join(", ", result.Frequencies.Take(8))}…");
        checks.Check("attention factor finite and positive", double.IsFinite(result.AttentionFactor) && result.AttentionFactor > 0, $"attention factor {result.AttentionFactor}");

        var reference = RopeScalings.Default(Type)?.Invoke(Input(theta, rotary, parameters));
        if (reference is null)
        {
            checks.Skip("agrees with the library's method", $"the library has no '{Type}' scaling");
        }
        else
        {
            string? difference = Relative(reference.Frequencies, result.Frequencies) is { } f ? $"frequencies: {f}"
                : Relative([reference.AttentionFactor], [result.AttentionFactor]) is { } a ? $"attention factor: {a}" : null;
            checks.Compare("agrees with the library's method", difference);
        }

        if (result.FrequenciesAt is null && reference?.FrequenciesAt is null)
        {
            checks.Skip("per-position frequencies", "the same frequencies at every position");
            return;
        }

        foreach (int position in positions)
        {
            var at = result.FrequenciesAt?.Invoke(position) ?? result.Frequencies;
            if (reference is not null)
            {
                var expected = reference.FrequenciesAt?.Invoke(position) ?? reference.Frequencies;
                checks.Compare("per-position frequencies agree with the library's method", Relative(expected, at) is { } d ? $"position {position}: {d}" : null);
            }
            else
            {
                checks.Check("per-position frequencies finite and positive", at.Length == half && at.All(f => double.IsFinite(f) && f > 0), $"position {position}");
            }
        }

        // Called once per position, possibly from several threads at once, when the rotary tables are made.
        var sequential = positions.Select(p => result.FrequenciesAt?.Invoke(p) ?? result.Frequencies).ToArray();
        var parallel = new double[positions.Length][];
        Parallel.For(0, positions.Length * 8, i => parallel[i % positions.Length] = result.FrequenciesAt?.Invoke(positions[i % positions.Length]) ?? result.Frequencies);
        checks.Compare("per-position frequencies from several threads", Enumerable.Range(0, positions.Length)
            .Select(i => Comparisons.Difference(sequential[i], parallel[i], 0) is { } d ? $"position {positions[i]}: {d}" : null).FirstOrDefault(d => d is not null));
    }

    private static RopeScalingInput Input(double theta, int rotary, JsonObject parameters)
    {
        var frequencies = new double[rotary / 2];
        for (int i = 0; i < frequencies.Length; i++)
        {
            frequencies[i] = 1.0 / Math.Pow(theta, 2.0 * i / rotary);
        }

        return new RopeScalingInput(frequencies, theta, rotary, (JsonObject)parameters.DeepClone());
    }

    private string? Relative(ReadOnlySpan<double> expected, ReadOnlySpan<double> actual)
    {
        if (expected.Length != actual.Length)
        {
            return $"{actual.Length} values, {expected.Length} expected";
        }

        for (int i = 0; i < expected.Length; i++)
        {
            if (!(Math.Abs(expected[i] - actual[i]) <= Tolerance * Math.Abs(expected[i])) && !expected[i].Equals(actual[i]))
            {
                return $"element {i} is {actual[i]:R}, expected {expected[i]:R} (relative tolerance {Tolerance:G3})";
            }
        }

        return null;
    }

    private static ContractCase Case(double theta, int rotary, JsonObject parameters, int[] positions) =>
        new($"θ {theta}, {rotary} dims, {parameters.ToJsonString()}", new JsonObject
        {
            ["theta"] = theta,
            ["rotaryDim"] = rotary,
            ["parameters"] = parameters.DeepClone(),
            ["positions"] = new JsonArray([.. positions.Select(p => (JsonNode)p)]),
        });

    private static List<JsonObject>? BuiltInParameters(string type) => type.ToLowerInvariant() switch
    {
        "linear" => [RopeScaling.Linear(2).Parameters, RopeScaling.Linear(8).Parameters, RopeScaling.Linear(0.5).Parameters],
        "llama3" => [RopeScaling.Llama3(8).Parameters, RopeScaling.Llama3(32, 1, 4, 8192).Parameters, RopeScaling.Llama3(4, 2, 8, 4096).Parameters],
        "yarn" =>
        [
            RopeScaling.Yarn(4, 32_768).Parameters, RopeScaling.Yarn(40, 4096, betaFast: 32, betaSlow: 1).Parameters,
            RopeScaling.Yarn(8, 8192, attentionFactor: 1.2).Parameters,
            new JsonObject { ["factor"] = 16, ["original_max_position_embeddings"] = 4096, ["mscale"] = 1, ["mscale_all_dim"] = 0.707 },
            new JsonObject { ["factor"] = 4, ["max_position_embeddings"] = 8192, ["truncate"] = false },
        ],
        "dynamic" => [RopeScaling.Dynamic(2, 4096).Parameters, RopeScaling.Dynamic(8, 2048).Parameters],
        _ => null,
    };
}
