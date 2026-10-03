// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Cli.Shared;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands;

/// <summary>
/// <c>idrak diff A B</c>: which tensors differ between two checkpoints and by how much (largest absolute difference and
/// relative L2 difference), and the tensors only one of them has. GGUF and safetensors checkpoints compare by their
/// Hugging Face tensor names, so a GGUF file compares with the folder it was made from.
/// </summary>
internal sealed class DiffCommand : Command
{
    public override string Name => "diff";

    public override string Summary => "Which tensors differ between two checkpoints, and by how much";

    public override string Usage => """
        A B [--limit N] [--tolerance X] [--filter TEXT]

          A, B              model folders, .gguf or .safetensors files, cached Hugging Face ids or aliases
              --limit N     show the N tensors that differ most (default 20; JSON has all)
              --tolerance X a relative L2 difference at most X counts as equal (default 0)
              --filter TEXT only tensors whose name contains this text

        Relative difference: ||a - b|| / ||a||. Every tensor is read in float32, one pair at a time.

        Examples:
          idrak diff ./base ./merged
          idrak diff ./model.gguf ./model-hf --tolerance 1e-3 --json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--limit", "--tolerance", "--filter"];

    public override int Run(CommandContext context)
    {
        string a = context.Argument(0, "A (the first checkpoint)"), b = context.Argument(1, "B (the second checkpoint)");
        int limit = context.IntOption("--limit", 20);
        double tolerance = context.Option("--tolerance") is { } t
            ? double.TryParse(t, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double v) && v >= 0 ? v
                : throw new UsageException($"--tolerance needs a number at least 0, not '{t}'.")
            : 0;
        string? filter = context.Option("--filter");
        using var left = Open(context, a);
        using var right = Open(context, b);
        var namesA = left.Names.Where(n => filter is null || n.Contains(filter, StringComparison.OrdinalIgnoreCase)).ToHashSet(StringComparer.Ordinal);
        var namesB = right.Names.Where(n => filter is null || n.Contains(filter, StringComparison.OrdinalIgnoreCase)).ToHashSet(StringComparer.Ordinal);
        var onlyA = namesA.Except(namesB).Order(StringComparer.Ordinal).ToList();
        var onlyB = namesB.Except(namesA).Order(StringComparer.Ordinal).ToList();
        var rows = new List<(string Name, string Shape, double MaxAbs, double Relative, bool ShapeMismatch)>();
        foreach (string name in namesA.Intersect(namesB).Order(StringComparer.Ordinal))
        {
            int[] shapeA = left.ShapeOf(name), shapeB = right.ShapeOf(name);
            string shape = "[" + string.Join(", ", shapeA) + "]";
            if (!shapeA.SequenceEqual(shapeB))
            {
                rows.Add((name, $"{shape} vs [{string.Join(", ", shapeB)}]", double.NaN, double.NaN, true));
                continue;
            }

            float[] x = left.Read(name), y = right.Read(name);
            double maxAbs = 0, diff = 0, norm = 0;
            for (int i = 0; i < x.Length; i++)
            {
                double d = (double)x[i] - y[i];
                maxAbs = Math.Max(maxAbs, Math.Abs(d));
                diff += d * d;
                norm += (double)x[i] * x[i];
            }

            rows.Add((name, shape, maxAbs, norm > 0 ? Math.Sqrt(diff / norm) : Math.Sqrt(diff), false));
        }

        var differing = rows.Where(r => r.ShapeMismatch || r.Relative > tolerance).OrderByDescending(r => r.ShapeMismatch ? double.MaxValue : r.Relative).ToList();
        bool same = differing.Count == 0 && onlyA.Count == 0 && onlyB.Count == 0;
        context.Write(same
            ? $"{a} and {b}: all {rows.Count} tensors equal{(tolerance > 0 ? $" within {tolerance:G3}" : "")}."
            : $"{a} vs {b}: {rows.Count} shared tensors, {differing.Count} differ; {onlyA.Count} only in A, {onlyB.Count} only in B.");
        if (differing.Count > 0)
        {
            context.Table(["Tensor", "Shape", "Max |a-b|", "Relative"], differing.Take(limit).Select(r => (IReadOnlyList<string>)[r.Name, r.Shape,
                r.ShapeMismatch ? "shapes differ" : r.MaxAbs.ToString("G4", System.Globalization.CultureInfo.InvariantCulture),
                r.ShapeMismatch ? "-" : r.Relative.ToString("G4", System.Globalization.CultureInfo.InvariantCulture)]));
            if (differing.Count > limit)
            {
                context.Write($"... {differing.Count - limit} more (--limit)");
            }
        }

        foreach (var (side, names) in new[] { ("A", onlyA), ("B", onlyB) }.Where(s => s.Item2.Count > 0))
        {
            context.Write($"Only in {side}: {string.Join(", ", names.Take(10))}{(names.Count > 10 ? $" … ({names.Count})" : "")}");
        }

        static JsonNode? Number(double value) => double.IsFinite(value) ? JsonValue.Create(value) : null;
        context.WriteJson(new JsonObject
        {
            ["a"] = a,
            ["b"] = b,
            ["same"] = same,
            ["shared"] = rows.Count,
            ["tolerance"] = tolerance,
            ["differing"] = new JsonArray([.. differing.Select(r => (JsonNode)new JsonObject
            {
                ["name"] = r.Name,
                ["shape"] = r.Shape,
                ["shapeMismatch"] = r.ShapeMismatch,
                ["maxAbs"] = Number(r.MaxAbs),
                ["relative"] = Number(r.Relative),
            })]),
            ["onlyA"] = new JsonArray([.. onlyA.Select(n => (JsonNode?)JsonValue.Create(n))]),
            ["onlyB"] = new JsonArray([.. onlyB.Select(n => (JsonNode?)JsonValue.Create(n))]),
        });
        return ExitCodes.Ok;
    }

    private static ITensorStore Open(CommandContext context, string name) =>
        File.Exists(name) && name.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase)
            ? SafeTensorsReader.Open(name)
            : ModelCache.OpenTensors(ModelCache.Locate(context, name).Folder);
}
