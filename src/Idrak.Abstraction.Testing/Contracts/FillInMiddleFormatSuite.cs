// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a fill-in-the-middle format (<see cref="IFillInMiddleFormat"/>, registered in
/// <see cref="FillInMiddleFormats"/>) with a tokenizer that has its tokens, on code-like prefixes and suffixes (empty ones,
/// several lines, Arabic and emoji, text that looks like markup) with and without repository snippets:
/// <list type="bullet">
/// <item>it detects the tokenizer;</item>
/// <item>the prompt holds the prefix and the suffix, and its text up to <see cref="IFillInMiddlePrompt.PrefixEnd"/> ends with the prefix;</item>
/// <item>its ids are the ids the tokenizer reads from its text;</item>
/// <item>every stop is a token of the tokenizer, its id beside it;</item>
/// <item>a longer prefix keeps the text up to the old prefix's end (what a kept cache reuses while the caret moves on);</item>
/// <item>the snippets it says it holds are in the text, path and text, and it holds no more than it was given;</item>
/// <item>the same prompt again and from several threads;</item>
/// <item>given a <see cref="Reference"/> (the format it replaces), the same text, ids and stops.</item>
/// </list>
/// </summary>
/// <param name="tokenizer">A tokenizer that has the format's tokens (the model's, or one made for the test).</param>
/// <param name="reference">The format to agree with, or null.</param>
public sealed class FillInMiddleFormatSuite(IPretrainedTokenizer tokenizer, IFillInMiddleFormat? reference = null) : ContractSuite<IFillInMiddleFormat>
{
    private static readonly string[] Pieces = ["def ", "return ", "x", " = ", "(", ")", ":", "\n", "    ", "if ", "for i in range(n)", "{", "}", ";", "// ",
        "النهر", "😊", "<div>", "\"text\"", "0.5", "\t", "a_b", "Foo.Bar()"];

    /// <summary>The tokenizer the prompts are made with.</summary>
    public IPretrainedTokenizer Tokenizer { get; } = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

    /// <summary>The format to agree with, or null.</summary>
    public IFillInMiddleFormat? Reference { get; } = reference;

    /// <inheritdoc />
    public override string Name => "fill-in-middle-format";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        yield return Case("both empty", "", "", []);
        yield return Case("an empty prefix", "", "\n    return x\n", []);
        yield return Case("an empty suffix", "def f(x):\n    ", "", []);
        yield return Case("a function's body", "def f(x):\n    y = ", "\n    return y\n", []);
        yield return Case("Arabic and emoji", "s = \"النهر ", " 😊\"\n", []);
        yield return Case("markup-like text", "<div class=\"a\">", "</div>\n", []);
        yield return Case("with snippets and a path", "import util\n\nutil.", "\n", [("src/util.py", "def go():\n    pass\n"), ("README.md", "# util\n")], "src/main.py", "demo");
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        string Code(int pieces)
        {
            var text = new StringBuilder();
            for (int i = 0; i < pieces; i++)
            {
                text.Append(Pieces[random.Next(Pieces.Length)]);
            }

            return text.ToString();
        }

        int size = large ? 400 : 24;
        var snippets = new List<(string, string)>();
        for (int s = random.Next(0, 3); s > 0; s--)
        {
            snippets.Add(($"src/f{random.Next(100)}.py", Code(random.Next(1, size))));
        }

        string? path = random.Next(2) == 0 ? "src/main.py" : null;
        return Case($"random code ({snippets.Count} snippets)", Code(random.Next(0, size)), Code(random.Next(0, size)), snippets, path, random.Next(2) == 0 ? "repo" : null);
    }

    /// <inheritdoc />
    public override void Run(IFillInMiddleFormat implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var context = Context(@case.Data);
        checks.Check("detects the tokenizer", implementation.Detect(Tokenizer), $"'{implementation.Name}' does not detect a tokenizer that has its tokens");
        var prompt = implementation.Create(context);
        string text = prompt.Text;
        checks.Check("the prompt holds the prefix and the suffix", text.Contains(context.Prefix, StringComparison.Ordinal) && text.Contains(context.Suffix, StringComparison.Ordinal),
            $"'{text}'");
        bool prefixEnd = prompt.PrefixEnd >= context.Prefix.Length && prompt.PrefixEnd <= text.Length
                         && text.AsSpan(0, prompt.PrefixEnd).EndsWith(context.Prefix, StringComparison.Ordinal);
        checks.Check("the text up to PrefixEnd ends with the prefix", prefixEnd, $"PrefixEnd {prompt.PrefixEnd} of {text.Length} in '{text}'");
        var read = Tokenizer.Encode(text);
        checks.Check("the ids are the text's", read.SequenceEqual(prompt.Ids), $"ids [{string.Join(", ", prompt.Ids)}], the tokenizer reads [{string.Join(", ", read)}]");
        bool stops = prompt.Stop.Count == prompt.StopIds.Count && prompt.Stop.Select(Tokenizer.IdOf).SequenceEqual(prompt.StopIds.Select(i => (int?)i));
        checks.Check("every stop is a token with its id", stops, $"stops [{string.Join(", ", prompt.Stop)}], ids [{string.Join(", ", prompt.StopIds)}]");
        if (prefixEnd)
        {
            var longer = implementation.Create(context with { Prefix = context.Prefix + "x1" });
            checks.Check("a longer prefix keeps the text up to the old prefix's end", longer.Text.AsSpan().StartsWith(text.AsSpan(0, prompt.PrefixEnd), StringComparison.Ordinal),
                $"'{text[..prompt.PrefixEnd]}' then '{longer.Text}'");
        }

        bool snippets = prompt.Snippets >= 0 && prompt.Snippets <= context.Snippets.Count
                        && context.Snippets.Take(prompt.Snippets).All(s => text.Contains(s.Path, StringComparison.Ordinal) && text.Contains(s.Text, StringComparison.Ordinal));
        checks.Check("the snippets it holds are in the text", snippets, $"{prompt.Snippets} of {context.Snippets.Count} snippets in '{text}'");
        var again = implementation.Create(context);
        checks.Check("the same prompt again", Same(prompt, again), $"'{again.Text}'");
        var parallel = new IFillInMiddlePrompt[4];
        Parallel.For(0, parallel.Length, i => parallel[i] = implementation.Create(context));
        checks.Check("the same prompt from several threads", parallel.All(p => Same(prompt, p)), "another prompt");
        if (Reference is not null)
        {
            var expected = Reference.Create(context);
            checks.Check("agrees with the reference", Same(expected, prompt), $"'{prompt.Text}', the reference '{expected.Text}'");
        }
        else
        {
            checks.Skip("agrees with the reference", "no reference format given");
        }
    }

    private static bool Same(IFillInMiddlePrompt a, IFillInMiddlePrompt b) =>
        a.Text == b.Text && a.Ids.SequenceEqual(b.Ids) && a.Stop.SequenceEqual(b.Stop) && a.StopIds.SequenceEqual(b.StopIds) && a.PrefixEnd == b.PrefixEnd && a.Snippets == b.Snippets;

    private FillInMiddleContext Context(JsonObject data)
    {
        var snippets = ((JsonArray)data["snippets"]!).Select(s => new FillInMiddleSnippet((string)s!["path"]!, (string)s["text"]!)).ToList();
        return new FillInMiddleContext(Tokenizer, (string)data["prefix"]!, (string)data["suffix"]!)
        {
            Snippets = snippets, Path = (string?)data["path"], Repository = (string?)data["repository"],
        };
    }

    private static ContractCase Case(string name, string prefix, string suffix, IReadOnlyList<(string Path, string Text)> snippets, string? path = null, string? repository = null) =>
        new(name, new JsonObject
        {
            ["prefix"] = prefix, ["suffix"] = suffix, ["path"] = path, ["repository"] = repository,
            ["snippets"] = new JsonArray([.. snippets.Select(s => (JsonNode)new JsonObject { ["path"] = s.Path, ["text"] = s.Text })]),
        });
}
