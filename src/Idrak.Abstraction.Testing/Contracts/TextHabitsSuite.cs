// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Text;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of text habits (<see cref="ITextHabits"/>, registered in <see cref="TextHabits"/>) on sample texts:
/// <list type="bullet">
/// <item>at chance 0 every text comes back as it was;</item>
/// <item>the same text and seed give the same result, at every chance;</item>
/// <item>a habit at chance 1 changes the texts it holds for, and a text no habit holds for comes back as it was;</item>
/// <item>the same results from several threads (each with its own generator);</item>
/// <item>given a <paramref name="reference"/> (the library's set it replaces), the same results.</item>
/// </list>
/// </summary>
/// <param name="texts">Texts in the habits' language, some each habit holds for.</param>
/// <param name="untouched">A text none of the habits holds for.</param>
/// <param name="reference">The library's habits to agree with, or null.</param>
public sealed class TextHabitsSuite(IReadOnlyList<string> texts, string untouched, ITextHabits? reference = null) : ContractSuite<ITextHabits>
{
    /// <inheritdoc />
    public override string Name => "text-habits";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        foreach (double chance in (double[])[0, 1, 0.5])
        {
            yield return new ContractCase($"every text at chance {chance}", new JsonObject { ["chance"] = chance, ["seed"] = 7 });
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return new ContractCase("every text at a random chance and seed", new JsonObject { ["chance"] = Math.Round(random.NextDouble(), 3), ["seed"] = random.Next() });
    }

    /// <inheritdoc />
    public override void Run(ITextHabits implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        double chance = (double)@case.Data["chance"]!;
        int seed = (int)@case.Data["seed"]!;
        string[] Apply(ITextHabits habits)
        {
            var random = new Random(seed);
            return [.. texts.Select(t => habits.Apply(t, random, _ => chance))];
        }

        var once = Apply(implementation);
        checks.Check("the same seed, the same texts", once.SequenceEqual(Apply(implementation)), string.Join(" | ", once));
        if (chance == 0)
        {
            checks.Check("at chance 0 nothing changes", once.SequenceEqual(texts), string.Join(" | ", once));
        }
        else if (chance == 1)
        {
            checks.Check("at chance 1 the habits change the texts they hold for", once.Zip(texts).Any(p => p.First != p.Second), string.Join(" | ", once));
            checks.Check("a text no habit holds for comes back as it was", implementation.Apply(untouched, new Random(seed), _ => 1) == untouched,
                implementation.Apply(untouched, new Random(seed), _ => 1));
        }

        var parallel = new string[4][];
        Parallel.For(0, parallel.Length, i => parallel[i] = Apply(implementation));
        checks.Check("the same texts from several threads", parallel.All(p => p.SequenceEqual(once)), "the threads differ");
        if (reference is not null)
        {
            checks.Check("agrees with the reference", Apply(reference).SequenceEqual(once), string.Join(" | ", Apply(reference)));
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference given");
        }
    }
}
