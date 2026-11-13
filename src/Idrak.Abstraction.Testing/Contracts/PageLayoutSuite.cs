// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a page layout (<see cref="IPageLayout"/>, registered in <see cref="PageLayouts"/>) drawing the contents
/// given in the style given (its fonts are the caller's: the kit has none) from several seeds:
/// <list type="bullet">
/// <item>the same seed draws the same page: the same pixels, text and parts;</item>
/// <item>the text is exactly the drawn text: its words are the words of the parts that hold text;</item>
/// <item>every part with text is inside the page and inked (its darkest pixel under half the paper's lightness).</item>
/// </list>
/// </summary>
/// <param name="style">The style the pages are drawn in (its fonts must have the contents' characters).</param>
/// <param name="contents">The contents drawn.</param>
public sealed class PageLayoutSuite(PageStyle style, IReadOnlyList<PageContent> contents) : ContractSuite<IPageLayout>
{
    /// <summary>The style the pages are drawn in.</summary>
    public PageStyle Style { get; } = style ?? throw new ArgumentNullException(nameof(style));

    /// <summary>The contents drawn.</summary>
    public IReadOnlyList<PageContent> Contents { get; } = contents is { Count: > 0 } ? contents : throw new ArgumentException("A layout is checked on at least one content.", nameof(contents));

    /// <inheritdoc />
    public override string Name => "page-layout";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int c = 0; c < Contents.Count; c++)
        {
            yield return Case($"content {c}, seed 1", c, 1);
            yield return Case($"content {c}, seed 2", c, 2);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int content = random.Next(Contents.Count), seed = random.Next();
        return Case($"content {content}, seed {seed}", content, seed);
    }

    /// <inheritdoc />
    public override void Run(IPageLayout implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var content = Contents[(int)@case.Data["content"]!];
        int seed = (int)@case.Data["seed"]!;
        var page = implementation.Render(content, Style, seed);
        var again = implementation.Render(content, Style, seed);
        var image = page.Image;
        checks.Check("the same seed, the same page",
            image.Channels == again.Image.Channels && image.Width == again.Image.Width && image.Height == again.Image.Height
            && image.Pixels.AsSpan().SequenceEqual(again.Image.Pixels) && page.Text == again.Text && page.Blocks.SequenceEqual(again.Blocks) && page.Description == again.Description,
            "another page from the same seed");

        var drawn = page.Blocks.Where(b => b.Text.Length > 0).ToList();
        var words = Words(page.Text);
        var blockWords = Words(string.Join("\n", drawn.Select(b => b.Text)));
        checks.Check("the text is the drawn text", words.SequenceEqual(blockWords),
            $"the text has {words.Count} words, its parts {blockWords.Count}: '{page.Text}' against '{string.Join(" / ", drawn.Select(b => b.Text))}'");

        var pixels = image.Pixels;
        int plane = image.Width * image.Height;
        float paper = 0;
        for (int c = 0; c < image.Channels; c++)
        {
            paper += pixels[c * plane] / image.Channels;
        }

        string? wrong = null;
        foreach (var block in drawn)
        {
            if (block.X < -0.5f || block.Y < -0.5f || block.X + block.Width > image.Width + 0.5f || block.Y + block.Height > image.Height + 0.5f)
            {
                wrong = string.Create(CultureInfo.InvariantCulture, $"{block.Kind} at {block.X:F0},{block.Y:F0} {block.Width:F0}x{block.Height:F0} outside the page {image.Width}x{image.Height}");
                break;
            }

            float darkest = 1;
            for (int y = Math.Max(0, (int)block.Y); y < Math.Min(image.Height, (int)MathF.Ceiling(block.Y + block.Height)); y++)
            {
                for (int x = Math.Max(0, (int)block.X); x < Math.Min(image.Width, (int)MathF.Ceiling(block.X + block.Width)); x++)
                {
                    float v = 0;
                    for (int c = 0; c < image.Channels; c++)
                    {
                        v += pixels[c * plane + y * image.Width + x] / image.Channels;
                    }

                    darkest = Math.Min(darkest, v);
                }
            }

            if (!(darkest < paper * 0.5f))
            {
                wrong = string.Create(CultureInfo.InvariantCulture, $"{block.Kind} '{block.Text}' is not inked: its darkest is {darkest:F2} on paper {paper:F2}");
                break;
            }
        }

        checks.Check("every part with text inside the page and inked", wrong is null, wrong ?? "");
    }

    /// <inheritdoc />
    protected override string Describe(IPageLayout implementation) => $"{implementation.Name} ({implementation.GetType().Name})";

    // The words of a text, sorted (the parts may hold them in another order than the text's lines).
    private static List<string> Words(string text) =>
        [.. text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Order(StringComparer.Ordinal)];

    private static ContractCase Case(string name, int content, int seed) => new(name, new JsonObject { ["content"] = content, ["seed"] = seed });
}
