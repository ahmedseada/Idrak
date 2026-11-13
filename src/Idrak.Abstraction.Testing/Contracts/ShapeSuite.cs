// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a shape (<see cref="IShape"/>, registered in <see cref="Shapes"/>) on its steps (the ones given, its
/// defaults at least) in boxes of several sizes, both directions, with the common options moved about:
/// <list type="bullet">
/// <item>the same step and context give the same plan;</item>
/// <item>its regions are inside the box asked for (below its top and within its width when the box may grow down);</item>
/// <item>no two regions overlap;</item>
/// <item>widths given (<c>widths=</c>) are honoured: a wider share gives a wider region of the group, the first in reading order at the start side;</item>
/// <item>its paths stay inside the box, give or take half their stroke (and below it when the box may grow down);</item>
/// <item>its keys take neither a common option's name nor a registered shape's, and an option it does not take is refused.</item>
/// </list>
/// </summary>
/// <param name="steps">The steps to check it on (each naming the shape); its default step (its name alone) when none.</param>
public sealed class ShapeSuite(IReadOnlyList<ShapeStep> steps) : ContractSuite<IShape>
{
    // Common options a random case adds to a step, as key, value pairs.
    private static readonly string[][] Common =
    [
        [], ["at", "0.1:0.2", "size", "0.8:0.7"], ["size", "0.5:0.6"], ["margin", "0.05"], ["width", "2.5"], ["dash", "dashed"], ["dash", "dotted"], ["tone", "0.6"],
    ];

    /// <summary>The steps it is checked on.</summary>
    public IReadOnlyList<ShapeStep> Steps { get; } = steps is { Count: > 0 } ? steps : throw new ArgumentException("A shape is checked on at least one step.", nameof(steps));

    /// <inheritdoc />
    public override string Name => "shape";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int i = 0; i < Steps.Count; i++)
        {
            yield return Case($"{Steps[i]} in a page, left to right", i, [], 60, 80, 1000, 700, rtl: false, grows: true, lines: 4, units: 5, seed: 1);
            yield return Case($"{Steps[i]} in a page, right to left", i, [], 60, 80, 1000, 700, rtl: true, grows: true, lines: 4, units: 5, seed: 2);
            yield return Case($"{Steps[i]} in a small box", i, [], 300, 400, 180, 90, rtl: false, grows: false, lines: 2, units: 3, seed: 3);
            yield return Case($"{Steps[i]} in a tall narrow box, right to left", i, [], 10, 10, 40, 400, rtl: true, grows: false, lines: 1, units: 1, seed: 4);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int step = random.Next(Steps.Count);
        var common = Common[random.Next(Common.Length)];
        float width = 20 + (float)random.NextDouble() * (large ? 3000 : 900), height = 20 + (float)random.NextDouble() * (large ? 3000 : 700);
        return Case($"{Steps[step]} {string.Join("=", common)} in {width:F0}x{height:F0}", step, common, (float)random.NextDouble() * 200, (float)random.NextDouble() * 200,
            width, height, rtl: random.Next(2) == 0, grows: random.Next(2) == 0, lines: random.Next(1, large ? 60 : 8), units: random.Next(1, large ? 40 : 10), seed: random.Next());
    }

    /// <inheritdoc />
    public override void Run(IShape implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        var step = Steps[(int)data["step"]!];
        if (data["common"] is JsonArray common)
        {
            for (int i = 0; i + 1 < common.Count; i += 2)
            {
                step = step.With((string)common[i]!, (string)common[i + 1]!);
            }
        }

        var outer = new ShapeContext
        {
            X = (float)data["x"]!, Y = (float)data["y"]!, Width = (float)data["width"]!, Height = (float)data["height"]!, RightToLeft = (bool)data["rtl"]!,
            GrowsDown = (bool)data["grows"]!, Lines = (int)data["lines"]!, MostUnits = (int)data["units"]!, Seed = (int)data["seed"]!,
            TextSize = 20, LineHeight = 26, StrokeWidth = 2,
        };

        ShapePipeline.Check(implementation, step);
        var context = outer.Place(step);
        var plan = implementation.Plan(step, context);
        checks.Check("the same plan again", implementation.Plan(step, context).ToString() == plan.ToString(), "another plan for the same step and context");

        const float Slack = 0.01f;
        float bottom = context.GrowsDown ? float.PositiveInfinity : context.Bottom;
        var outside = plan.Regions.Where(r => r.X < context.X - Slack || r.Y < context.Y - Slack || r.X + r.Width > context.Right + Slack || r.Y + r.Height > bottom + Slack
            || !(r.Width >= 0 && r.Height >= 0)).ToList();
        checks.Check("its regions inside the box asked for", outside.Count == 0,
            $"{outside.Count} regions outside {context.X:F1},{context.Y:F1} {context.Width:F1}x{context.Height:F1}, the first {outside.FirstOrDefault()}");

        string? overlap = null;
        for (int i = 0; i < plan.Regions.Count && overlap is null; i++)
        {
            for (int j = i + 1; j < plan.Regions.Count && overlap is null; j++)
            {
                var (a, b) = (plan.Regions[i], plan.Regions[j]);
                float w = Math.Min(a.X + a.Width, b.X + b.Width) - Math.Max(a.X, b.X), h = Math.Min(a.Y + a.Height, b.Y + b.Height) - Math.Max(a.Y, b.Y);
                if (w > 0.01f && h > 0.01f)
                {
                    overlap = $"{a} and {b}";
                }
            }
        }

        checks.Check("no two regions overlap", overlap is null, overlap ?? "");

        if (step.Option("widths") is { } widthsText && plan.Regions.Count > 1)
        {
            var widths = PageOptions.Numbers("widths", widthsText, 0, float.MaxValue);
            string? wrong = null;
            foreach (var group in plan.Regions.GroupBy(r => r.Group))
            {
                var row = group.OrderBy(r => r.Index).ToList();
                for (int i = 0; i + 1 < row.Count && wrong is null; i++)
                {
                    float wi = i < widths.Length ? widths[i] : 1, wj = i + 1 < widths.Length ? widths[i + 1] : 1;
                    bool startFirst = context.RightToLeft ? row[i].X > row[i + 1].X : row[i].X < row[i + 1].X;
                    if (!startFirst || (wi > wj * 1.01f && row[i].Width < row[i + 1].Width - 0.01f) || (wj > wi * 1.01f && row[i + 1].Width < row[i].Width - 0.01f))
                    {
                        wrong = $"row {group.Key}: region {i} {row[i]} and {i + 1} {row[i + 1]} for widths {wi} and {wj}";
                    }
                }
            }

            checks.Check("widths honoured, from the start side", wrong is null, wrong ?? "");
        }
        else
        {
            checks.Skip("widths honoured, from the start side", "the step gives no widths");
        }

        string? stray = null;
        foreach (var path in plan.Paths)
        {
            float reach = path.Width * outer.StrokeWidth / 2 + 0.5f;
            var p = path.Points.Span;
            for (int i = 0; i < p.Length && stray is null; i += 2)
            {
                if (p[i] < context.X - reach || p[i] > context.Right + reach || p[i + 1] < context.Y - reach || p[i + 1] > bottom + reach || !float.IsFinite(p[i]) || !float.IsFinite(p[i + 1]))
                {
                    stray = string.Create(CultureInfo.InvariantCulture, $"a point at {p[i]:F1},{p[i + 1]:F1} outside {context.X:F1},{context.Y:F1} {context.Width:F1}x{context.Height:F1}");
                }
            }
        }

        foreach (var area in plan.Areas)
        {
            var (x0, y0, x1, y1) = area.Bounds();
            if (stray is null && (x0 < context.X - 0.5f || y0 < context.Y - 0.5f || x1 > context.Right + 0.5f || y1 > bottom + 0.5f))
            {
                stray = string.Create(CultureInfo.InvariantCulture, $"an area {x0:F1},{y0:F1} to {x1:F1},{y1:F1} outside the box");
            }
        }

        checks.Check("its paths inside the box, give or take half their stroke", stray is null, stray ?? "");

        string? clash = implementation.Keys.FirstOrDefault(k => Shapes.CommonKeys.Contains(k) || Shapes.Find(k) is not null);
        checks.Check("its keys are neither common options nor shapes", clash is null, $"its key '{clash}' is also a common option or a shape's name");
        try
        {
            ShapePipeline.Check(implementation, step.With("no-such-option", "1"));
            checks.Fail("an option it does not take is refused", "no-such-option=1 was taken");
        }
        catch (ArgumentException e)
        {
            checks.Check("an option it does not take is refused", e.Message.Contains("no-such-option", StringComparison.Ordinal), $"the refusal does not name the option: {e.Message}");
        }
    }

    /// <inheritdoc />
    protected override string Describe(IShape implementation) => $"{implementation.Name} ({implementation.GetType().Name})";

    private static ContractCase Case(string name, int step, string[] common, float x, float y, float width, float height, bool rtl, bool grows, int lines, int units, int seed) =>
        new(name, new JsonObject
        {
            ["step"] = step, ["common"] = new JsonArray([.. common.Select(c => (JsonNode)c)]), ["x"] = x, ["y"] = y, ["width"] = width, ["height"] = height, ["rtl"] = rtl,
            ["grows"] = grows, ["lines"] = lines, ["units"] = units, ["seed"] = seed,
        });
}
