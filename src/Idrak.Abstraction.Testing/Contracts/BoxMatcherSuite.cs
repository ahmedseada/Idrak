// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>What a box matcher is expected to do (<see cref="BoxMatcherSuite"/>).</summary>
/// <param name="OneToOne">One truth per prediction at most and the largest total quality (set prediction); else threshold matching.</param>
/// <param name="Positive">Threshold matching: the quality a match needs.</param>
/// <param name="Negative">Threshold matching: below it a prediction is a negative (between the two, ignored).</param>
/// <param name="AllowLowQuality">Threshold matching: each truth also keeps the predictions of its best quality.</param>
public sealed record BoxMatcherExpectation(bool OneToOne, float Positive = 0.5f, float Negative = 0.4f, bool AllowLowQuality = false)
{
    /// <summary>One-to-one matching of the largest total quality.</summary>
    public static BoxMatcherExpectation Optimal { get; } = new(true);
}

/// <summary>
/// The checks of a box matcher (Idrak.Vision's <c>BoxMatcher</c>, given as a delegate with its options bound: a quality
/// matrix [predictions, truths] in, each prediction's truth (or -1 for a negative, -2 for ignored) out) on quality
/// matrices of random boxes' IoU, ties, forbidden pairs (-∞) and empty sides:
/// <list type="bullet">
/// <item>one answer per prediction, each a truth, -1 or -2; the same every time;</item>
/// <item>threshold matching: a matched prediction has its best truth and passes the positive threshold (or is a truth's best,
/// with low-quality matches); below the negative threshold, a negative; between, ignored;</item>
/// <item>one-to-one matching: no truth twice, min(predictions, truths) pairs where no pair is forbidden, and a total quality no
/// assignment beats (every assignment listed for small matrices).</item>
/// </list>
/// </summary>
/// <param name="expectation">What the matcher is expected to do.</param>
public sealed class BoxMatcherSuite(BoxMatcherExpectation expectation) : ContractSuite<Func<ReadOnlyMemory<float>, int, int, int[]>>
{
    /// <inheritdoc />
    public override string Name => "box-matcher";

    /// <summary>What the matcher is expected to do.</summary>
    public BoxMatcherExpectation Expectation { get; } = expectation ?? throw new ArgumentNullException(nameof(expectation));

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("no truths", 1, 3, 0, ties: false, forbidden: false);
        yield return Case("no predictions", 2, 0, 3, ties: false, forbidden: false);
        yield return Case("square", 3, 4, 4, ties: false, forbidden: false);
        yield return Case("more predictions than truths", 4, 7, 3, ties: false, forbidden: false);
        yield return Case("more truths than predictions", 5, 2, 5, ties: false, forbidden: false);
        yield return Case("ties", 6, 5, 4, ties: true, forbidden: false);
        yield return Case("forbidden pairs", 7, 5, 5, ties: false, forbidden: true);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int max = large ? 120 : 7;
        return Case("random", random.Next(), random.Next(0, max), random.Next(0, max), random.Next(4) == 0, random.Next(4) == 0);
    }

    /// <inheritdoc />
    public override void Run(Func<ReadOnlyMemory<float>, int, int, int[]> implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var d = @case.Data;
        int p = (int)d["predictions"]!, g = (int)d["truths"]!;
        var q = Quality(new Random((int)d["seed"]!), p, g, (bool)d["ties"]!, (bool)d["forbidden"]! && Expectation.OneToOne);
        var matches = implementation(q, p, g);
        checks.Check("one answer per prediction, a truth, -1 or -2", matches.Length == p && matches.All(m => m >= -2 && m < g),
            $"[{string.Join(", ", matches)}] for {p} predictions and {g} truths");
        checks.Compare("the same answer every time", Comparisons.Difference(matches, implementation(q, p, g)));
        if (matches.Length != p)
        {
            return;
        }

        if (Expectation.OneToOne)
        {
            OneToOne(q, p, g, matches, checks);
        }
        else
        {
            Thresholds(q, p, g, matches, checks);
        }
    }

    private static void OneToOne(float[] q, int p, int g, int[] matches, CaseChecks checks)
    {
        var used = matches.Where(m => m >= 0).ToList();
        checks.Check("no truth matched twice", used.Count == used.Distinct().Count(), $"[{string.Join(", ", matches)}]");
        checks.Check("no forbidden pair matched", Enumerable.Range(0, p).All(i => matches[i] < 0 || !float.IsNegativeInfinity(q[i * g + matches[i]])), $"[{string.Join(", ", matches)}]");
        checks.Check("nothing ignored", matches.All(m => m != -2), $"[{string.Join(", ", matches)}]");
        if (p > 7 && g > 7)
        {
            checks.Skip("the largest total quality", "too large to list every assignment");
            return;
        }

        var (best, pairs) = Best(q, p, g);
        double total = Enumerable.Range(0, p).Where(i => matches[i] >= 0).Sum(i => (double)q[i * g + matches[i]]);
        checks.Check("as many pairs as can be made", used.Count == pairs, $"{used.Count} pairs, {pairs} possible");
        checks.Check("the largest total quality", total >= best - 1e-4 * Math.Max(1, Math.Abs(best)), $"total {total}, best {best}");
    }

    // The largest total over assignments with the most pairs that avoid forbidden ones (every assignment listed).
    private static (double Best, int Pairs) Best(float[] q, int p, int g)
    {
        double best = double.NegativeInfinity;
        int bestPairs = -1;
        var taken = new bool[g];
        void Visit(int i, double total, int pairs)
        {
            if (i == p)
            {
                if (pairs > bestPairs || pairs == bestPairs && total > best)
                {
                    (best, bestPairs) = (total, pairs);
                }

                return;
            }

            Visit(i + 1, total, pairs);
            for (int j = 0; j < g; j++)
            {
                if (!taken[j] && !float.IsNegativeInfinity(q[i * g + j]))
                {
                    taken[j] = true;
                    Visit(i + 1, total + q[i * g + j], pairs + 1);
                    taken[j] = false;
                }
            }
        }

        Visit(0, 0, 0);
        return (bestPairs <= 0 ? 0 : best, Math.Max(0, bestPairs));
    }

    private void Thresholds(float[] q, int p, int g, int[] matches, CaseChecks checks)
    {
        var e = Expectation;
        var bestOfTruth = Enumerable.Range(0, g).Select(j => Enumerable.Range(0, p).Select(i => q[i * g + j]).DefaultIfEmpty(float.NegativeInfinity).Max()).ToArray();
        string? wrong = null;
        for (int i = 0; i < p && wrong is null; i++)
        {
            float best = Enumerable.Range(0, g).Select(j => q[i * g + j]).DefaultIfEmpty(float.NegativeInfinity).Max();
            bool lowQuality = e.AllowLowQuality && Enumerable.Range(0, g).Any(j => q[i * g + j] == bestOfTruth[j] && !float.IsNegativeInfinity(bestOfTruth[j]));
            int m = matches[i];
            wrong = m >= 0 && q[i * g + m] != best ? $"prediction {i} matched truth {m} (quality {q[i * g + m]}), not its best ({best})"
                : m >= 0 && best < e.Positive && !lowQuality ? $"prediction {i} matched with quality {best}, below {e.Positive}"
                : m < 0 && g > 0 && (best >= e.Positive || lowQuality) ? $"prediction {i} (best quality {best}) is not matched"
                : m == -1 && best >= e.Negative && !(g == 0) ? $"prediction {i} (best quality {best}) is a negative, not ignored"
                : m == -2 && best < e.Negative ? $"prediction {i} (best quality {best}) is ignored, not a negative" : null;
        }

        checks.Compare("matched, ignored and negative by the thresholds", wrong);
    }

    // IoU of random boxes; with ties, quantized to a few values; forbidden pairs as -∞.
    private static float[] Quality(Random random, int p, int g, bool ties, bool forbidden)
    {
        BoundingBox Box() => new(random.NextSingle() * 40, random.NextSingle() * 40, 4 + random.NextSingle() * 20, 4 + random.NextSingle() * 20);
        var truths = Enumerable.Range(0, g).Select(_ => Box()).ToArray();
        var q = new float[p * g];
        for (int i = 0; i < p; i++)
        {
            var box = g > 0 && random.Next(2) == 0 ? Jitter(truths[random.Next(g)]) : Box();
            for (int j = 0; j < g; j++)
            {
                float v = box.IntersectionOverUnion(truths[j]);
                q[i * g + j] = ties ? MathF.Round(v * 4) / 4 : v;
                if (forbidden && random.Next(5) == 0)
                {
                    q[i * g + j] = float.NegativeInfinity;
                }
            }
        }

        return q;

        BoundingBox Jitter(BoundingBox b) => new(b.X + (random.NextSingle() - 0.5f) * 6, b.Y + (random.NextSingle() - 0.5f) * 6, b.Width, b.Height);
    }

    private static ContractCase Case(string name, int seed, int predictions, int truths, bool ties, bool forbidden) =>
        new($"{name} ({predictions} predictions, {truths} truths{(ties ? ", ties" : "")}{(forbidden ? ", forbidden pairs" : "")})", new JsonObject
        {
            ["seed"] = seed,
            ["predictions"] = predictions,
            ["truths"] = truths,
            ["ties"] = ties,
            ["forbidden"] = forbidden,
        });
}
