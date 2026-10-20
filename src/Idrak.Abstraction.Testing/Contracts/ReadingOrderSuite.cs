// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a reading order (<see cref="IReadingOrder"/>, registered in <see cref="ReadingOrders"/>) on lines of
/// Arabic and Hebrew words, Latin words, European and Arabic-Indic numbers and separators, in both directions:
/// <list type="bullet">
/// <item>the columns hold the line's characters, each once (a reordering, nothing added or lost);</item>
/// <item>a line of the line's own direction only (Arabic in a right-to-left line, Latin in a left-to-right one) keeps its
/// order;</item>
/// <item>a reading turned back to logical order gives the same columns again (and, with <see cref="Invertible"/>, the
/// text itself: an order that maps no two texts to the same columns);</item>
/// <item>the same text again and from several threads;</item>
/// <item>given a <see cref="Reference"/> (the library's order it replaces), the same columns and the same text back.</item>
/// </list>
/// </summary>
/// <param name="reference">The library's order to agree with, or null.</param>
/// <param name="invertible">Whether every text comes back from its columns exactly.</param>
public sealed class ReadingOrderSuite(IReadingOrder? reference = null, bool invertible = false) : ContractSuite<IReadingOrder>
{
    private static readonly string[] Arabic = ["باب", "سم", "النهر", "مدينة", "كتب"];
    private static readonly string[] Hebrew = ["שלום", "עיר"];
    private static readonly string[] Latin = ["abc", "de", "river", "Idrak", "x"];
    private static readonly string[] Numbers = ["123", "2026", "3.14", "12-3", "١٢", "١٢٫٥"];

    /// <summary>The library's order to agree with, or null.</summary>
    public IReadingOrder? Reference { get; } = reference;

    /// <summary>Whether every text is expected back from its columns exactly.</summary>
    public bool Invertible { get; } = invertible;

    /// <inheritdoc />
    public override string Name => "reading-order";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("empty", "", rightToLeft: true);
        yield return Case("arabic only", "باب سم النهر", rightToLeft: true);
        yield return Case("latin only, left to right", "the river runs north", rightToLeft: false);
        yield return Case("numbers and latin in arabic", "با 123 س abc de م ١٢", rightToLeft: true);
        yield return Case("arabic in latin", "the word باب سم here", rightToLeft: false);
        yield return Case("a decimal number", "سم 3.14 باب", rightToLeft: true);
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        bool rightToLeft = random.Next(2) == 0;
        int words = random.Next(1, large ? 60 : 9);
        string Word() => random.Next(10) switch
        {
            < 4 => rightToLeft ? Arabic[random.Next(Arabic.Length)] : Latin[random.Next(Latin.Length)],
            < 6 => rightToLeft ? Latin[random.Next(Latin.Length)] : Arabic[random.Next(Arabic.Length)],
            < 7 => Hebrew[random.Next(Hebrew.Length)],
            _ => Numbers[random.Next(Numbers.Length)],
        };
        return Case($"random line of {words} words, {(rightToLeft ? "right to left" : "left to right")}", string.Join(' ', Enumerable.Range(0, words).Select(_ => Word())), rightToLeft);
    }

    /// <inheritdoc />
    public override void Run(IReadingOrder implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        string logical = (string)@case.Data["text"]!;
        bool rightToLeft = (bool)@case.Data["rtl"]!;
        string columns = implementation.ToColumns(logical, rightToLeft);
        checks.Check("the line's characters, each once", Sorted(columns) == Sorted(logical), $"'{columns}' for '{logical}'");
        bool own = logical.EnumerateRunes().All(r => !char.IsLetterOrDigit(r.ToString(), 0) || IsRightToLeft(r) == rightToLeft && !System.Text.Rune.IsDigit(r));
        if (own)
        {
            checks.Check("a line of its own direction keeps its order", columns == logical, $"'{columns}' for '{logical}'");
        }

        string back = implementation.FromColumns(columns, rightToLeft);
        checks.Check("read back, the same columns", implementation.ToColumns(back, rightToLeft) == columns, $"'{logical}' → '{columns}' → '{back}'");
        if (Invertible)
        {
            checks.Check("read back, the same text", back == logical, $"'{logical}' → '{columns}' → '{back}'");
        }
        else
        {
            checks.Skip("read back, the same text", "the order maps some texts to the same columns");
        }

        checks.Check("the same columns again", implementation.ToColumns(logical, rightToLeft) == columns, "other columns");
        var parallel = new string[4];
        Parallel.For(0, parallel.Length, i => parallel[i] = implementation.ToColumns(logical, rightToLeft));
        checks.Check("the same columns from several threads", parallel.All(p => p == columns), string.Join(" | ", parallel));
        if (Reference is not null)
        {
            string expected = Reference.ToColumns(logical, rightToLeft);
            checks.Check("agrees with the reference", expected == columns && Reference.FromColumns(columns, rightToLeft) == back,
                $"'{columns}' and back '{back}', the reference '{expected}' and back '{Reference.FromColumns(columns, rightToLeft)}'");
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference order given");
        }
    }

    private static bool IsRightToLeft(System.Text.Rune rune) => rune.Value is >= 0x0590 and <= 0x08FF;

    private static string Sorted(string text) => string.Concat(text.EnumerateRunes().Select(r => r.Value).Order().Select(v => char.ConvertFromUtf32(v)));

    private static ContractCase Case(string name, string text, bool rightToLeft) => new(name, new JsonObject { ["text"] = text, ["rtl"] = rightToLeft });
}
