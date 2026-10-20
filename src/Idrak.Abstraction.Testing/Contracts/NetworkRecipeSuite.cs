// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// A network recipe under test (Idrak's <c>INetworkRecipe</c> of <c>NetworkRecipes</c>, given as a delegate since this kit
/// depends on Idrak.Abstraction alone): builds the network of the recipe's arguments, every layer's weights drawn in order
/// from the random given.
/// </summary>
/// <param name="Build">The network for the arguments, its weights drawn from the random.</param>
public sealed record NetworkRecipeUnderTest(Func<JsonObject, Random, Module> Build);

/// <summary>A recipe's arguments the suite builds it with, and an input batch the network takes.</summary>
/// <param name="Arguments">The recipe's arguments.</param>
/// <param name="InputShape">The shape of an input batch, the batch first.</param>
/// <param name="Vocabulary">For token ids: the inputs are ids below this; null for numbers.</param>
public sealed record NetworkRecipeSample(JsonObject Arguments, int[] InputShape, int? Vocabulary = null);

/// <summary>
/// The checks of a network recipe on its samples, each with a seed: the same seed gives the same weights (parameters of
/// the same count, shapes and values) and the same predictions, the predictions are finite and keep the batch, and, given
/// a reference (the library's recipe of that name), the same parameter count, shapes, weights and predictions within
/// <see cref="Tolerance"/>. The library's recipes ("mlp", "cnn", "rnn", "transformer-classifier", "gpt") have samples of
/// their own; a recipe of your own needs some.
/// </summary>
/// <param name="recipe">The recipe's name.</param>
/// <param name="samples">The arguments and inputs to try; null for a library recipe's own.</param>
/// <param name="reference">The recipe to agree with (the library's of that name), or null.</param>
public sealed class NetworkRecipeSuite(string recipe, IReadOnlyList<NetworkRecipeSample>? samples = null, NetworkRecipeUnderTest? reference = null)
    : ContractSuite<NetworkRecipeUnderTest>
{
    private readonly IReadOnlyList<NetworkRecipeSample> _samples = samples ?? BuiltInSamples(recipe)
        ?? throw new ArgumentException($"Network recipe '{recipe}' is not built in: give the samples to check it with.", nameof(samples));

    /// <summary>The recipe checked.</summary>
    public string Recipe { get; } = recipe;

    /// <summary>The recipe to agree with, or null.</summary>
    public NetworkRecipeUnderTest? Reference { get; } = reference;

    /// <summary>The relative agreement wanted with the reference's predictions.</summary>
    public float Tolerance { get; init; } = 1e-5f;

    /// <inheritdoc />
    public override string Name => "network-recipe";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases() => _samples.Select((_, i) => Case(i, seed: 2));

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case(random.Next(_samples.Count), random.Next());
    }

    /// <inheritdoc />
    public override void Run(NetworkRecipeUnderTest implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var sample = _samples[(int)@case.Data["sample"]!];
        int seed = (int)@case.Data["seed"]!;
        using var first = implementation.Build((JsonObject)sample.Arguments.DeepClone(), new Random(seed));
        using var second = implementation.Build((JsonObject)sample.Arguments.DeepClone(), new Random(seed));
        checks.Compare("the same seed gives the same weights", Weights(first, second, 0f));

        using var scope = new TensorScope();
        var input = Input(sample, seed);
        var output = first.Predict(input);
        float[] values = output.ToArray();
        checks.Check("predictions finite", values.All(float.IsFinite), $"{values.Count(v => !float.IsFinite(v))} of {values.Length} values not finite");
        checks.Check("predictions keep the batch", output.Shape.Length > 0 && output.Shape[0] == sample.InputShape[0],
            $"output {Tensor.FormatShape(output.Shape.ToArray())} for input {Tensor.FormatShape(sample.InputShape)}");
        checks.Compare("the same weights give the same predictions", Comparisons.Difference(values, second.Predict(input).ToArray(), 0f));
        if (Reference is null)
        {
            checks.Skip("agrees with the reference", "no reference recipe given");
            return;
        }

        using var expected = Reference.Build((JsonObject)sample.Arguments.DeepClone(), new Random(seed));
        checks.Compare("agrees with the reference", Weights(expected, first, 0f) is { } w ? $"weights: {w}"
            : Comparisons.Difference(expected.Predict(input).ToArray(), values, Tolerance) is { } p ? $"predictions: {p}" : null);
    }

    // How two networks' parameters differ (count, shapes, values), or null.
    private static string? Weights(Module expected, Module actual, float tolerance)
    {
        var a = expected.Parameters().ToList();
        var b = actual.Parameters().ToList();
        if (a.Count != b.Count || expected.ParameterCount != actual.ParameterCount)
        {
            return $"{b.Count} parameters ({actual.ParameterCount:N0} values), expected {a.Count} ({expected.ParameterCount:N0})";
        }

        for (int i = 0; i < a.Count; i++)
        {
            if (!a[i].Shape.SequenceEqual(b[i].Shape))
            {
                return $"parameter {i} is {Tensor.FormatShape(b[i].Shape.ToArray())}, expected {Tensor.FormatShape(a[i].Shape.ToArray())}";
            }

            if (Comparisons.Difference(a[i].ToArray(), b[i].ToArray(), tolerance) is { } d)
            {
                return $"parameter {i}: {d}";
            }
        }

        return null;
    }

    // A deterministic input batch: token ids below the vocabulary, or numbers in [-1, 1).
    private static Tensor Input(NetworkRecipeSample sample, int seed)
    {
        var random = new Random(seed ^ 0x5bd1e995);
        int size = sample.InputShape.Aggregate(1, (x, y) => x * y);
        float[] values = sample.Vocabulary is { } vocabulary
            ? [.. Enumerable.Range(0, size).Select(_ => (float)random.Next(vocabulary))]
            : [.. Enumerable.Range(0, size).Select(_ => random.NextSingle() * 2 - 1)];
        return Tensor.From(values, sample.InputShape, Device.Cpu);
    }

    private ContractCase Case(int sample, int seed) =>
        new($"{Recipe} sample {sample}, seed {seed}", new JsonObject { ["sample"] = sample, ["seed"] = seed, ["arguments"] = _samples[sample].Arguments.DeepClone() });

    // The library recipes' samples: small networks of each kind.
    private static IReadOnlyList<NetworkRecipeSample>? BuiltInSamples(string recipe) => recipe.ToLowerInvariant() switch
    {
        "mlp" =>
        [
            new(new JsonObject { ["inputs"] = 5, ["hidden"] = new JsonArray(8, 4), ["outputs"] = 3 }, [4, 5]),
            new(new JsonObject { ["inputs"] = 3, ["hidden"] = new JsonArray(6), ["outputs"] = 2, ["activation"] = "tanh", ["dropout"] = 0.25f }, [2, 3]),
        ],
        "cnn" => [new(new JsonObject { ["channels"] = 1, ["height"] = 8, ["width"] = 8, ["filters"] = new JsonArray(4, 6), ["kernelSize"] = 3, ["outputs"] = 3 }, [2, 1, 8, 8])],
        "rnn" =>
        [
            new(new JsonObject { ["vocabulary"] = 9, ["length"] = 6, ["embed"] = 4, ["hidden"] = 5, ["outputs"] = 2 }, [3, 6], 9),
            new(new JsonObject { ["cell"] = "gru", ["vocabulary"] = 9, ["length"] = 6, ["embed"] = 4, ["hidden"] = 5, ["outputs"] = 2 }, [3, 6], 9),
        ],
        "transformer-classifier" =>
            [new(new JsonObject { ["vocabulary"] = 11, ["length"] = 6, ["dim"] = 8, ["heads"] = 2, ["layers"] = 1, ["outputs"] = 3 }, [2, 6], 11)],
        "gpt" =>
        [
            new(new JsonObject { ["vocabulary"] = 13, ["context"] = 8, ["dim"] = 8, ["heads"] = 2, ["layers"] = 2 }, [2, 8], 13),
            new(new JsonObject { ["vocabulary"] = 7, ["context"] = 5, ["dim"] = 12, ["heads"] = 3, ["layers"] = 1, ["ffDim"] = 20, ["dropout"] = 0f }, [1, 5], 7),
        ],
        _ => null,
    };
}
