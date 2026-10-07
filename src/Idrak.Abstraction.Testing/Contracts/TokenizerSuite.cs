// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a tokenizer (<see cref="ITokenizer"/>) on texts (empty, whitespace, punctuation, accents, Arabic,
/// Chinese, emoji, combining marks, control characters, long runs, random Unicode) and on id sequences:
/// <list type="bullet">
/// <item>every id is in [0, <see cref="ITokenizer.VocabularySize"/>), and encoding the same text gives the same ids;</item>
/// <item>the round trip: decoding the ids gives back the text (<see cref="ExactRoundTrip"/>), or at least a text that a
/// second round trip keeps unchanged (characters outside the vocabulary may be dropped, case and spaces normalized);</item>
/// <item>decoding a span gives the text decoding a list gives, any in-range id decodes, and <see cref="ITokenizer.TokenOf"/>
/// is "" past the end;</item>
/// <item>with a <c>reference</c> (the library tokenizer an app's one replaces or wraps): the same ids and text.</item>
/// </list>
/// </summary>
/// <param name="reference">The tokenizer to agree with, or null.</param>
public sealed class TokenizerSuite(ITokenizer? reference = null) : ContractSuite<ITokenizer>
{
    private static readonly string[] Texts =
    [
        "", " ", "a", "hello world", "Hello, World! How are you?", "  leading and trailing  ", "line\nbreak\ttab\r\nwindows",
        "Ünïcödé ß café naïve façade", "مرحبا بالعالم، كيف حالك؟", "中文字符和标点。", "emoji 😀👍🏽 and 🇸🇦 flags", "é combining",
        "‏‎ marks ﻿", "control \u0001\u001f chars", "digits 12345 3.14159 -0.5e10", "if (x) { y(); } // code",
        "<sum> special <unk> tokens </s>", new string('a', 1000), string.Concat(Enumerable.Repeat("word ", 300)),
    ];

    /// <summary>The library tokenizer this one must agree with, or null.</summary>
    public ITokenizer? Reference { get; } = reference;

    /// <summary>Whether decoding a text's ids must give the text exactly (byte-level tokenizers); else it must be stable.</summary>
    public bool ExactRoundTrip { get; init; }

    /// <summary>Texts of your own, checked after the built-in ones (the inputs your app sends).</summary>
    public IReadOnlyList<string> ExtraTexts { get; init; } = [];

    /// <inheritdoc />
    public override string Name => "tokenizer";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int i = 0; i < Texts.Length; i++)
        {
            yield return TextCase($"text {i}", Texts[i]);
        }

        for (int i = 0; i < ExtraTexts.Count; i++)
        {
            yield return TextCase($"your text {i}", ExtraTexts[i]);
        }

        yield return new ContractCase("ids: the first and last of the vocabulary", new JsonObject { ["ids"] = new JsonArray(0, 1, -1) });
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        if (random.Next(4) == 0)
        {
            // Ids are drawn when the case runs, as fractions of the vocabulary (the suite does not know its size).
            int count = random.Next(0, large ? 2000 : 40);
            return new ContractCase($"random ids ({count})", new JsonObject
            {
                ["ids"] = new JsonArray([.. Enumerable.Range(0, count).Select(_ => (JsonNode)random.NextDouble())]),
            });
        }

        var text = new StringBuilder();
        int pieces = random.Next(1, large ? 400 : 24);
        for (int i = 0; i < pieces; i++)
        {
            text.Append(random.Next(10) switch
            {
                < 5 => Words[random.Next(Words.Length)],
                5 => random.Next(-100_000, 100_000).ToString(System.Globalization.CultureInfo.InvariantCulture),
                6 => Texts[random.Next(Texts.Length)],
                7 => char.ConvertFromUtf32(Scalar(random)),
                _ => " .,;:!?'\"()[]{}<>/\\-_\n\t"[random.Next(23)].ToString(),
            });
            if (random.Next(3) > 0)
            {
                text.Append(' ');
            }
        }

        return TextCase($"random text ({text.Length} chars)", text.ToString());
    }

    /// <inheritdoc />
    public override void Run(ITokenizer implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        int size = implementation.VocabularySize;
        checks.Check("a vocabulary", size > 0, $"VocabularySize is {size}");
        if (@case.Data["ids"] is JsonArray idsJson)
        {
            RunIds(implementation, idsJson, size, checks);
            return;
        }

        string text = (string)@case.Data["text"]!;
        var ids = implementation.Encode(text);
        int outside = ids.ToList().FindIndex(id => id < 0 || id >= size);
        checks.Check("ids in range", outside < 0, outside < 0 ? "" : $"id {outside} is {ids[outside]}, outside [0, {size})");
        checks.Compare("the same ids every time", Comparisons.Difference(ids, implementation.Encode(text)));

        string decoded = implementation.Decode(ids);
        checks.Compare("a span decodes as a list does", Comparisons.Difference(decoded, implementation.Decode(ids.ToArray().AsSpan())));
        if (ExactRoundTrip)
        {
            checks.Compare("round trip gives the text", Comparisons.Difference(text, decoded));
        }
        else
        {
            checks.Compare("round trip is stable", Comparisons.Difference(decoded, implementation.Decode(implementation.Encode(decoded))));
        }

        if (Reference is null)
        {
            checks.Skip("the reference's ids", "no reference tokenizer given");
            return;
        }

        var expected = Reference.Encode(text);
        checks.Compare("the reference's ids", Comparisons.Difference(expected, ids));
        checks.Compare("the reference's text", Comparisons.Difference(Reference.Decode(expected), decoded));
    }

    private void RunIds(ITokenizer implementation, JsonArray idsJson, int size, CaseChecks checks)
    {
        // Integers are ids (negative from the end); fractions are drawn ids, as a share of the vocabulary.
        var ids = idsJson.Select(n => double.Parse(n!.ToJsonString(), System.Globalization.CultureInfo.InvariantCulture) is var v
            && v == Math.Floor(v) && Math.Abs(v) < size ? (int)(v < 0 ? size + v : v) : (int)Math.Clamp(v * size, 0, size - 1)).ToArray();
        string decoded = implementation.Decode(ids);
        checks.Compare("a span decodes as a list does", Comparisons.Difference(decoded, implementation.Decode(ids.AsSpan())));
        checks.Check("TokenOf is empty past the end", implementation.TokenOf(size) == "" && implementation.TokenOf(-1) == "",
            $"TokenOf({size}) is \"{implementation.TokenOf(size)}\", TokenOf(-1) \"{implementation.TokenOf(-1)}\"");
        if (Reference is not null)
        {
            checks.Compare("the reference's text", Comparisons.Difference(Reference.Decode(ids), decoded));
        }
    }

    private static ContractCase TextCase(string name, string text) => new(name, new JsonObject { ["text"] = text });

    // A Unicode scalar value from the ranges texts use (ASCII, Latin, Arabic, CJK, emoji), never a lone surrogate.
    private static int Scalar(Random random) => random.Next(6) switch
    {
        0 => random.Next(0x20, 0x7F),
        1 => random.Next(0xA0, 0x250),
        2 => random.Next(0x600, 0x700),
        3 => random.Next(0x4E00, 0x9FFF),
        4 => random.Next(0x1F300, 0x1FAFF),
        _ => random.Next(0x0, 0x20),
    };

    private static readonly string[] Words =
    [
        "the", "a", "model", "token", "Idrak", "learning", "is", "of", "and", "THE", "Hello", "world", "data", "x", "y",
        "naïve", "مرحبا", "数据", "😀", "don't", "e-mail", "C#", "3.5", "GPU", "<s>", "</s>",
    ];
}
