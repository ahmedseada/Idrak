// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Data;

namespace Idrak.Abstraction.Testing;

/// <summary>
/// The checks of a glyph pose (<see cref="IGlyphPose"/>, registered in <see cref="GlyphPoses"/>) along lines of several
/// text sizes, with its default settings and the settings given:
/// <list type="bullet">
/// <item>the same input gives the same pose, so every glyph of a cluster (asking with its cluster's place) gets one pose;</item>
/// <item>every pose is within the bounds its settings declare (move, scale, slant, turn, ink), and finite;</item>
/// <item>neighbouring clusters a twentieth of an em apart differ in their move down, slant and turn by at most a tenth of
/// those bounds (joined letters stay joined);</item>
/// <item>with <see cref="Identity"/>, every pose is the identity.</item>
/// </list>
/// </summary>
/// <param name="settings">Option sets to read the pose's settings from besides its defaults (its own keys only).</param>
/// <param name="identity">Whether the pose draws every glyph as set (the "none" pose).</param>
public sealed class GlyphPoseSuite(IReadOnlyList<IReadOnlyDictionary<string, string>>? settings = null, bool identity = false) : ContractSuite<IGlyphPose>
{
    /// <summary>The option sets the pose is checked with, its defaults first.</summary>
    public IReadOnlyList<IReadOnlyDictionary<string, string>> Settings { get; } = [PageOptions.None, .. settings ?? []];

    /// <summary>Whether every pose is expected to be the identity.</summary>
    public bool Identity { get; } = identity;

    /// <inheritdoc />
    public override string Name => "glyph-pose";

    /// <inheritdoc />
    public override IEnumerable<ContractCase> FixedCases()
    {
        for (int s = 0; s < Settings.Count; s++)
        {
            yield return Case("a line of body text", s, line: 0, size: 30, seed: 1, length: 40);
            yield return Case("a later line, small text", s, line: 17, size: 9, seed: 2, length: 60);
            yield return Case("large text", s, line: 3, size: 120, seed: 3, length: 12);
        }
    }

    /// <inheritdoc />
    public override ContractCase RandomCase(Random random, bool large)
    {
        ArgumentNullException.ThrowIfNull(random);
        return Case("a random line", random.Next(Settings.Count), random.Next(0, 500), 4 + (float)random.NextDouble() * 200, random.Next(), random.Next(1, large ? 400 : 60));
    }

    /// <inheritdoc />
    public override void Run(IGlyphPose implementation, ContractCase @case, CaseChecks checks)
    {
        ArgumentNullException.ThrowIfNull(implementation);
        ArgumentNullException.ThrowIfNull(@case);
        ArgumentNullException.ThrowIfNull(checks);
        var data = @case.Data;
        var settings = implementation.Read(Settings[(int)data["settings"]!]);
        var bounds = implementation.Bounds(settings);
        int line = (int)data["line"]!, seed = (int)data["seed"]!, length = (int)data["length"]!;
        float size = (float)data["size"]!;
        string? again = null, outside = null, jump = null, moved = null;
        GlyphPose? before = null;
        for (int k = 0; k < length; k++)
        {
            // A cluster every twentieth of an em along the line.
            var input = new GlyphPoseInput(line, k * size / 20, k, size, seed);
            var pose = implementation.Pose(input, settings);
            if (implementation.Pose(input, settings) != pose)
            {
                again ??= $"cluster {k}: another pose asked again";
            }

            const float Slack = 1e-4f;
            if (!(Math.Abs(pose.Dx) <= bounds.Move * size * (1 + Slack) + Slack && Math.Abs(pose.Dy) <= bounds.Move * size * (1 + Slack) + Slack
                  && Math.Abs(pose.Scale - 1) <= bounds.Scale + Slack && Math.Abs(pose.Shear) <= bounds.Shear + Slack && Math.Abs(pose.Rotation) <= bounds.Rotation + Slack
                  && pose.Ink >= bounds.Ink - Slack && pose.Ink <= 1 + Slack && pose.Scale > 0))
            {
                outside ??= string.Create(CultureInfo.InvariantCulture, $"cluster {k}: {pose} outside {bounds} at {size}px");
            }

            if (before is { } last && (Math.Abs(pose.Dy - last.Dy) > 0.1f * bounds.Move * size + Slack * size || Math.Abs(pose.Shear - last.Shear) > 0.1f * bounds.Shear + Slack
                || Math.Abs(pose.Rotation - last.Rotation) > 0.1f * bounds.Rotation + Slack))
            {
                jump ??= $"clusters {k - 1} and {k}, a twentieth of an em apart: {last} then {pose}";
            }

            if (!pose.IsIdentity)
            {
                moved ??= $"cluster {k}: {pose}";
            }

            before = pose;
        }

        checks.Check("the same pose asked again (one pose a cluster)", again is null, again ?? "");
        checks.Check("every pose within its declared bounds", outside is null, outside ?? "");
        checks.Check("neighbouring clusters change only smoothly", jump is null, jump ?? "");
        if (Identity)
        {
            checks.Check("every pose the identity", moved is null && bounds == GlyphPoseBounds.None, moved ?? $"bounds {bounds}");
        }
        else
        {
            checks.Skip("every pose the identity", "the pose is not the identity");
        }
    }

    /// <inheritdoc />
    protected override string Describe(IGlyphPose implementation) => $"{implementation.Name} ({implementation.GetType().Name})";

    private static ContractCase Case(string name, int settings, int line, float size, int seed, int length) =>
        new(name, new JsonObject { ["settings"] = settings, ["line"] = line, ["size"] = size, ["seed"] = seed, ["length"] = length });
}
