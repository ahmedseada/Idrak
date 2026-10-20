// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of an attention position bias (<see cref="IAttentionPositionBias"/>, a kind of
/// <see cref="AttentionPositionBiases"/>) for queries, keys and a first query position:
/// <list type="bullet">
/// <item>the bias is [heads, queries, keys] on the asked device, every value finite;</item>
/// <item>asked again, it gives the same values (a kept bias is the one computed);</item>
/// <item>a decoding step's bias is the whole sequence's rows for its positions: the bias of queries [start, start + q)
/// over k keys equals rows start to start + q of the bias of k queries over k keys, asked in either order (so cached
/// decoding equals the pass over the whole sequence);</item>
/// <item>given a <c>reference</c> (the same kind made by another implementation, or the library's), the same values
/// within <see cref="Tolerance"/>.</item>
/// </list>
/// </summary>
/// <param name="reference">The bias to compare with, or null.</param>
/// <param name="device">Where the bias is asked for (the CPU when null).</param>
public sealed class PositionBiasSuite(IAttentionPositionBias? reference = null, Device? device = null) : ContractSuite<IAttentionPositionBias>
{
    /// <inheritdoc />
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <summary>The agreement wanted with the reference (see <see cref="Comparisons.Close"/>).</summary>
    public float Tolerance { get; init; } = 1e-6f;

    /// <inheritdoc />
    public override string Name => "attention-position-bias";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        foreach (var (queries, keys, start) in new[] { (1, 1, 0), (5, 5, 0), (1, 8, 3), (1, 8, 7), (3, 10, 4), (7, 7, 0), (2, 16, 14), (1, 300, 299) })
        {
            yield return Case(queries, keys, start);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int keys = random.Next(1, large ? 513 : 65), start = random.Next(0, keys), queries = random.Next(1, keys - start + 1);
        return Case(queries, keys, start);
    }

    /// <inheritdoc />
    public override void Run(IAttentionPositionBias implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        int queries = (int)@case.Data["queries"]!, keys = (int)@case.Data["keys"]!, start = (int)@case.Data["start"]!;
        int heads = implementation.Heads;
        using var scope = new TensorScope();
        var step = implementation.For(queries, keys, start, Device);
        bool shaped = step.Rank == 3 && step.Shape[0] == heads && step.Shape[1] == queries && step.Shape[2] == keys;
        checks.Check("the bias is [heads, queries, keys]", shaped, $"{Tensor.FormatShape(step.Shape)}, [{heads}, {queries}, {keys}] expected");
        checks.Check("on the asked device", step.Device == Device, $"on {step.Device}, {Device} asked for");
        if (!shaped || step.Device != Device)
        {
            return;
        }

        float[] values = step.ToArray();
        checks.Check("every value finite", values.All(float.IsFinite), $"a value of {values.FirstOrDefault(v => !float.IsFinite(v))}");
        checks.Compare("the same values asked again", Comparisons.Difference(values, implementation.For(queries, keys, start, Device).ToArray(), 0f));

        // The whole sequence's rows for the step's positions, asked after the step and, from a fresh order, before it.
        float[] whole = implementation.For(keys, keys, 0, Device).ToArray();
        checks.Compare("a step's bias is the whole sequence's rows", Comparisons.Difference(Rows(whole, heads, keys, start, queries), values, 0f));
        float[] again = implementation.For(queries, keys, start, Device).ToArray();
        checks.Compare("asked after the whole sequence, the same step", Comparisons.Difference(values, again, 0f));
        if (reference is not null)
        {
            checks.Compare("the reference's values", Comparisons.Difference(reference.For(queries, keys, start, Device).ToArray(), values, Tolerance));
        }
    }

    // Rows [start, start + queries) of each head's [keys, keys] bias.
    private static float[] Rows(float[] whole, int heads, int keys, int start, int queries)
    {
        var rows = new float[heads * queries * keys];
        for (int h = 0; h < heads; h++)
        {
            whole.AsSpan((h * keys + start) * keys, queries * keys).CopyTo(rows.AsSpan(h * queries * keys));
        }

        return rows;
    }

    private static ContractCase Case(int queries, int keys, int start) =>
        new($"{queries} queries from position {start} over {keys} keys", new JsonObject { ["queries"] = queries, ["keys"] = keys, ["start"] = start });
}

/// <summary>
/// The checks of a sequence pooler (<see cref="ISequencePooler"/>, registered in <see cref="SequencePoolers"/>) on padded
/// batches of random states:
/// <list type="bullet">
/// <item>one vector per sequence, [batch, width], every value finite;</item>
/// <item>the padding does not matter: other values at the padded positions give the same vectors;</item>
/// <item>a sequence pools the same alone ([1, length, width]; an empty one, its first position) as in the batch;</item>
/// <item>given a <c>reference</c> (the library's pooler of that name, say), the same vectors within
/// <see cref="Tolerance"/>.</item>
/// </list>
/// </summary>
/// <param name="reference">The pooler to compare with, or null.</param>
/// <param name="device">Where the states live (the CPU when null).</param>
public sealed class SequencePoolerSuite(ISequencePooler? reference = null, Device? device = null) : ContractSuite<ISequencePooler>
{
    /// <inheritdoc />
    public override Device Device { get; } = device ?? Device.Cpu;

    /// <summary>The agreement wanted between the batch and one by one, and with the reference.</summary>
    public float Tolerance { get; init; } = 1e-5f;

    /// <inheritdoc />
    public override string Name => "sequence-pooler";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("one full sequence", 1, [4], 4, 3);
        yield return Case("padded batch", 3, [5, 2, 4], 5, 8);
        yield return Case("an empty sequence", 4, [3, 0, 1, 3], 3, 2);
        yield return Case("wide states", 5, [7, 1], 7, 130);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int n = random.Next(1, large ? 17 : 6), t = random.Next(1, large ? 129 : 17);
        int[] lengths = [.. Enumerable.Range(0, n).Select(_ => random.Next(0, t + 1))];
        return Case("random", random.Next(), lengths, t, random.Next(1, large ? 257 : 33));
    }

    /// <inheritdoc />
    public override void Run(ISequencePooler implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        int seed = (int)d["seed"]!, t = (int)d["time"]!, width = (int)d["width"]!;
        int[] lengths = [.. d["lengths"]!.AsArray().Select(l => (int)l!)];
        int n = lengths.Length;
        var random = new Random(seed);
        float[] states = [.. Enumerable.Range(0, n * t * width).Select(_ => random.NextSingle() * 4f - 2f)];
        using var scope = new TensorScope();
        using var noGrad = Autograd.NoGrad();
        var pooled = implementation.Pool(Tensor.From(states, [n, t, width], Device), lengths);
        bool shaped = pooled.Rank == 2 && pooled.Shape[0] == n && pooled.Shape[1] == width;
        checks.Check("one vector per sequence, [batch, width]", shaped, $"{Tensor.FormatShape(pooled.Shape)}, [{n}, {width}] expected");
        if (!shaped)
        {
            return;
        }

        float[] vectors = pooled.ToArray();
        checks.Check("every value finite", vectors.All(float.IsFinite), $"a value of {vectors.FirstOrDefault(v => !float.IsFinite(v))}");

        // Other values at the padded positions (an empty sequence keeps its first position, which it pools).
        float[] other = [.. states];
        for (int b = 0; b < n; b++)
        {
            for (int p = Math.Max(lengths[b], 1); p < t; p++)
            {
                for (int c = 0; c < width; c++)
                {
                    other[(b * t + p) * width + c] = random.NextSingle() * 100f - 50f;
                }
            }
        }

        checks.Compare("the padding does not matter", Comparisons.Difference(vectors, implementation.Pool(Tensor.From(other, [n, t, width], Device), lengths).ToArray(), 0f));

        string? alone = null;
        for (int b = 0; b < n && alone is null; b++)
        {
            int length = lengths[b], kept = Math.Max(length, 1);
            var own = Tensor.From(states.AsSpan(b * t * width, kept * width), [1, kept, width], Device);
            var single = implementation.Pool(own, length == 0 ? new[] { 0 } : null).ToArray();
            alone = Comparisons.Difference(single, vectors.AsSpan(b * width, width), Tolerance) is { } difference ? $"sequence {b} (length {length}): {difference}" : null;
        }

        checks.Compare("a sequence alone pools as in the batch", alone);
        if (reference is not null)
        {
            checks.Compare("the reference's vectors", Comparisons.Difference(reference.Pool(Tensor.From(states, [n, t, width], Device), lengths).ToArray(), vectors, Tolerance));
        }
    }

    private static ContractCase Case(string name, int seed, int[] lengths, int time, int width) =>
        new($"{name}: lengths {string.Join(", ", lengths)} of {time}, width {width}", new JsonObject
        {
            ["seed"] = seed, ["time"] = time, ["width"] = width, ["lengths"] = new JsonArray([.. lengths.Select(l => (JsonNode)l)]),
        });
}
