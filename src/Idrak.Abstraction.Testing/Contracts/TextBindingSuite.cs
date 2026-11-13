// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a text binding (<see cref="ITextBinding"/>, registered in <see cref="TextBindings"/>) on lines of words
/// and numbers put into rows of regions (a grid's), single regions (a box's) and groups out of order, with its default
/// settings and the settings given:
/// <list type="bullet">
/// <item>one text per region, the same for the same input and seed;</item>
/// <item>the texts, group after group, are units of the lines in their order (some may be left out, none added or moved);</item>
/// <item>the label is the drawn text: one line per group that has text, in the order of the groups' numbers, each the
/// group's non-empty texts in the order of their indexes, so a region left empty is absent from it.</item>
/// </list>
/// </summary>
/// <param name="settings">Option sets to read the binding's settings from besides its defaults (its own keys only).</param>
public sealed class TextBindingSuite(IReadOnlyList<IReadOnlyDictionary<string, string>>? settings = null) : ContractSuite<ITextBinding>
{
    private static readonly string[] Words = ["باب", "مدينة", "river", "Idrak", "12", "٣٤", "صفر", "x", "north", "7.5"];

    /// <summary>The option sets the binding is checked with, its defaults first.</summary>
    public IReadOnlyList<IReadOnlyDictionary<string, string>> Settings { get; } = [PageOptions.None, .. settings ?? []];

    /// <inheritdoc />
    public override string Name => "text-binding";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int s = 0; s < Settings.Count; s++)
        {
            yield return Case("no lines, no regions", s, [], [], seed: 1);
            yield return Case("a grid of two rows of four", s, ["1 باب مدينة 12 7", "2 river north 3"], Grid(2, 4), seed: 2);
            yield return Case("more lines than rows", s, ["a b", "c d", "e f"], Grid(2, 2), seed: 3);
            yield return Case("more rows than lines", s, ["a b c"], Grid(3, 3), seed: 4);
            yield return Case("a box takes its whole line", s, ["one two three"], [[0, 0]], seed: 5);
            yield return Case("groups given out of order", s, ["first line here", "second line"], [[1, 1], [1, 0], [0, 2], [0, 0], [0, 1]], seed: 6);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int lines = random.Next(0, large ? 80 : 7), rows = random.Next(0, large ? 80 : 7), cols = random.Next(1, large ? 30 : 9);
        var text = Enumerable.Range(0, lines).Select(_ => string.Join(' ', Enumerable.Range(0, random.Next(1, large ? 40 : 12)).Select(_ => Words[random.Next(Words.Length)]))).ToArray();
        var regions = random.Next(4) == 0 ? Enumerable.Range(0, rows).Select(r => new[] { r, 0 }).ToArray() : Grid(rows, cols);
        return Case($"{lines} lines into {regions.Length} regions", random.Next(Settings.Count), text, regions, random.Next());
    }

    /// <inheritdoc />
    public override void Run(ITextBinding implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        var lines = data["lines"]!.AsArray().Select(l => (string)l!).ToArray();
        var regions = data["regions"]!.AsArray().Select((r, i) => new ShapeRegion((int)r![0]!, (int)r[1]!, i * 10, 0, 10, 10)).ToArray();
        int seed = (int)data["seed"]!;
        var settings = implementation.Read(Settings[(int)data["settings"]!]);
        var bound = implementation.Bind(lines, regions, settings, seed);
        var again = implementation.Bind(lines, regions, settings, seed);
        checks.Check("a text per region, the same for the same seed", bound.Texts.Count == regions.Length && bound.Texts.SequenceEqual(again.Texts) && bound.Label.SequenceEqual(again.Label),
            $"{bound.Texts.Count} texts for {regions.Length} regions");
        if (bound.Texts.Count != regions.Length)
        {
            return;
        }

        var groups = regions.Select((r, i) => (Region: r, Text: bound.Texts[i])).GroupBy(p => p.Region.Group).OrderBy(g => g.Key).ToList();
        var expected = new List<string>();
        var units = new List<string>();
        foreach (var group in groups)
        {
            var texts = group.OrderBy(p => p.Region.Index).Select(p => p.Text).Where(t => t.Length > 0).ToList();
            units.AddRange(texts.SelectMany(t => t.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)));
            if (texts.Count > 0)
            {
                expected.Add(string.Join(" ", texts));
            }
        }

        var source = lines.SelectMany(l => l.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries)).ToArray();
        checks.Check("the texts are the lines' units in order", IsSubsequence(units, source), $"'{string.Join(" ", units)}' are not units of the lines in order");
        checks.Check("the label is the drawn text, a line a group in reading order", expected.SequenceEqual(bound.Label),
            $"label '{string.Join(" / ", bound.Label)}', drawn '{string.Join(" / ", expected)}'");
    }

    /// <inheritdoc />
    protected override string Describe(ITextBinding implementation) => $"{implementation.Name} ({implementation.GetType().Name})";

    private static bool IsSubsequence(List<string> units, string[] source)
    {
        int at = 0;
        foreach (string unit in units)
        {
            while (at < source.Length && source[at] != unit)
            {
                at++;
            }

            if (at == source.Length)
            {
                return false;
            }

            at++;
        }

        return true;
    }

    private static int[][] Grid(int rows, int cols) => [.. Enumerable.Range(0, rows).SelectMany(r => Enumerable.Range(0, cols).Select(c => new[] { r, c }))];

    private static ContractCase Case(string name, int settings, string[] lines, int[][] regions, int seed) =>
        new(name, new JsonObject
        {
            ["settings"] = settings, ["lines"] = new JsonArray([.. lines.Select(l => (JsonNode)l)]),
            ["regions"] = new JsonArray([.. regions.Select(r => (JsonNode)new JsonArray(r[0], r[1]))]), ["seed"] = seed,
        });
}
