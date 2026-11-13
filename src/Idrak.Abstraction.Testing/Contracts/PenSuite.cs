// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a pen (<see cref="IPen"/>, registered in <see cref="Pens"/>) on straight, bent and closed paths and a
/// single point, at several widths, with its default settings and the settings given:
/// <list type="bullet">
/// <item>the same path, settings and seed give the same stroke;</item>
/// <item>every point is within what the pen declares it strays (<see cref="IPen.Stray"/>) of the ideal path;</item>
/// <item>every width is above 0 and finite, every ink from 0 to 1, and the stroke is closed when its path is;</item>
/// <item>with <see cref="Exact"/>, the stroke is the path itself.</item>
/// </list>
/// </summary>
/// <param name="settings">Option sets to read the pen's settings from besides its defaults (its own keys only).</param>
/// <param name="exact">Whether the pen draws its path exactly (an ideal pen).</param>
public sealed class PenSuite(IReadOnlyList<IReadOnlyDictionary<string, string>>? settings = null, bool exact = false) : ContractSuite<IPen>
{
    /// <summary>The option sets the pen is checked with, its defaults first.</summary>
    public IReadOnlyList<IReadOnlyDictionary<string, string>> Settings { get; } = [PageOptions.None, .. settings ?? []];

    /// <summary>Whether the pen is expected to draw its path exactly.</summary>
    public bool Exact { get; } = exact;

    /// <inheritdoc />
    public override string Name => "pen";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int s = 0; s < Settings.Count; s++)
        {
            yield return Case("a straight line", s, [10, 50, 610, 50], closed: false, width: 2.5f, seed: 1);
            yield return Case("a bent line", s, [10, 10, 300, 40, 320, 400, 20, 380], closed: false, width: 3, seed: 2);
            yield return Case("a closed square", s, [100, 100, 300, 100, 300, 300, 100, 300], closed: true, width: 2, seed: 3);
            yield return Case("a short line", s, [5, 5, 8, 6], closed: false, width: 1.5f, seed: 4);
            yield return Case("a single point", s, [40, 40], closed: false, width: 4, seed: 5);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        int count = random.Next(2, large ? 200 : 12);
        var points = new float[2 * count];
        for (int i = 0; i < points.Length; i++)
        {
            points[i] = (float)random.NextDouble() * (large ? 4000 : 800);
        }

        return Case($"a random path of {count} points", random.Next(Settings.Count), points, closed: random.Next(3) == 0, width: 0.5f + (float)random.NextDouble() * 6, seed: random.Next());
    }

    /// <inheritdoc />
    public override void Run(IPen implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        var points = data["points"]!.AsArray().Select(p => (float)p!).ToArray();
        bool closed = (bool)data["closed"]!;
        float width = (float)data["width"]!;
        int seed = (int)data["seed"]!;
        var path = new PenPoint[points.Length / 2];
        for (int i = 0; i < path.Length; i++)
        {
            path[i] = new PenPoint(points[2 * i], points[2 * i + 1], width);
        }

        var settings = implementation.Read(Settings[(int)data["settings"]!]);
        using var stroke = new PenStroke();
        using var again = new PenStroke();
        implementation.Stroke(path, closed, settings, seed, stroke);
        implementation.Stroke(path, closed, settings, seed, again);
        checks.Check("the same stroke for the same seed", stroke.Points.SequenceEqual(again.Points) && stroke.Closed == again.Closed,
            $"{stroke.Count} points, then {again.Count}");

        float bound = implementation.Stray(settings, width) + 1e-3f * Math.Max(1, width);
        string? far = null;
        foreach (var p in stroke.Points)
        {
            float distance = Distance(p.X, p.Y, path, closed);
            if (distance > bound + 1e-3f * Math.Max(1, Math.Abs(p.X) + Math.Abs(p.Y)))
            {
                far = string.Create(CultureInfo.InvariantCulture, $"a point at {p.X:F2},{p.Y:F2} is {distance:F3} from the path; the pen declares {bound:F3}");
                break;
            }
        }

        checks.Check("every point within the declared stray", far is null, far ?? "");
        var bad = stroke.Points.ToArray().FirstOrDefault(p => !(p.Width > 0) || !float.IsFinite(p.Width) || !(p.Ink is >= 0 and <= 1) || !float.IsFinite(p.X) || !float.IsFinite(p.Y));
        checks.Check("widths above 0, inks from 0 to 1", stroke.Count > 0 && bad == default, stroke.Count == 0 ? "no point drawn" : $"{bad}");
        checks.Check("closed when its path is", stroke.Closed == closed || path.Length < 3, $"the stroke is {(stroke.Closed ? "" : "not ")}closed");
        if (Exact)
        {
            checks.Check("the stroke is the path", stroke.Points.SequenceEqual(path), $"{stroke.Count} points for a path of {path.Length}");
        }
        else
        {
            checks.Skip("the stroke is the path", "the pen is not exact");
        }
    }

    /// <inheritdoc />
    protected override string Describe(IPen implementation) => $"{implementation.Name} ({implementation.GetType().Name})";

    // The distance from (x, y) to the path's polyline (through its first point again when closed).
    private static float Distance(float x, float y, PenPoint[] path, bool closed)
    {
        double best = Math.Sqrt((x - path[0].X) * (double)(x - path[0].X) + (y - path[0].Y) * (double)(y - path[0].Y));
        int segments = closed ? path.Length : path.Length - 1;
        for (int i = 0; i < segments; i++)
        {
            var a = path[i];
            var b = path[(i + 1) % path.Length];
            double dx = b.X - a.X, dy = b.Y - a.Y, length = dx * dx + dy * dy;
            double t = length > 0 ? Math.Clamp(((x - a.X) * dx + (y - a.Y) * dy) / length, 0, 1) : 0;
            double px = a.X + t * dx - x, py = a.Y + t * dy - y;
            best = Math.Min(best, Math.Sqrt(px * px + py * py));
        }

        return (float)best;
    }

    private static ContractCase Case(string name, int settings, float[] points, bool closed, float width, int seed) =>
        new(name, new JsonObject
        {
            ["settings"] = settings, ["points"] = new JsonArray([.. points.Select(p => (JsonNode)p)]), ["closed"] = closed, ["width"] = width, ["seed"] = seed,
        });
}
