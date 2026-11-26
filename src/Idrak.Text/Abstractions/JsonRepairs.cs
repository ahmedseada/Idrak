// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;

namespace Idrak.Text.Abstractions;

/// <summary>The JSON a repair found in a model's answer, and whether it had to be repaired to parse.</summary>
/// <param name="Json">The value (an object or an array; null only for a JSON null).</param>
/// <param name="Repaired">False when the JSON parsed as it was written.</param>
public sealed record JsonRepairResult(JsonNode? Json, bool Repaired);

/// <summary>
/// A way of reading the JSON of a model's answer when the model did not write it exactly (an answer cut at its token
/// limit, a code fence around it, commas or quotes wrong).
/// </summary>
public interface IJsonRepair
{
    /// <summary>Its name.</summary>
    string Name { get; }

    /// <summary>The first JSON object or array of <paramref name="text"/>, repaired; null when the text has none.</summary>
    JsonRepairResult? Repair(string text);
}

/// <summary>
/// The JSON repairs. The library's: <see cref="Lenient"/> (as Python's json_repair repairs a model's JSON), which
/// <see cref="Parse"/> reaches directly.
/// </summary>
public static class JsonRepairs
{
    private static readonly LenientJsonRepair LenientRepair = new();
    /// <summary>The library's repair: open strings and brackets closed, commas, quotes, bare keys and Python literals fixed, code fences dropped.</summary>
    public const string Lenient = "lenient";

    /// <summary>The JSON in <paramref name="text"/>, repaired leniently (<see cref="Lenient"/>); null when it holds none.</summary>
    public static JsonRepairResult? Parse(string text) => LenientRepair.Repair(text);
}
