// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Nlp.Abstractions;

/// <summary>The JSON a repair found in a model's answer, and whether it had to be repaired to parse.</summary>
/// <param name="Json">The value (an object or an array; null only for a JSON null).</param>
/// <param name="Repaired">False when the JSON parsed as it was written.</param>
public sealed record JsonRepairResult(JsonNode? Json, bool Repaired);

/// <summary>
/// A way of reading the JSON of a model's answer when the model did not write it exactly (an answer cut at its token
/// limit, a code fence around it, commas or quotes wrong). Register one with <see cref="JsonRepairs.Register"/>.
/// </summary>
public interface IJsonRepair
{
    /// <summary>The name it is registered under.</summary>
    string Name { get; }

    /// <summary>One line on what it repairs.</summary>
    string Summary { get; }

    /// <summary>The first JSON object or array of <paramref name="text"/>, repaired; null when the text has none.</summary>
    JsonRepairResult? Repair(string text);
}

/// <summary>
/// The JSON repairs, by name (ignoring case). The library's: <see cref="Lenient"/> (as Python's json_repair repairs a
/// model's JSON). Register another with <see cref="Register"/>; one under the library's name shadows it, which
/// <see cref="Unregister"/> brings back.
/// </summary>
public static class JsonRepairs
{
    /// <summary>The library's repair: open strings and brackets closed, commas, quotes, bare keys and Python literals fixed, code fences dropped.</summary>
    public const string Lenient = "lenient";

    private static readonly SlotTable<string, IJsonRepair> Table = BuiltIn();

    private static SlotTable<string, IJsonRepair> BuiltIn()
    {
        var table = new SlotTable<string, IJsonRepair>(nameof(JsonRepairs), (slot, app, library) => new GuardedRepair(slot, app, library), StringComparer.OrdinalIgnoreCase);
        table.RegisterDefault(Lenient, new LenientJsonRepair());
        return table;
    }

    /// <summary>Registers <paramref name="repair"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IJsonRepair repair)
    {
        ArgumentNullException.ThrowIfNull(repair);
        ArgumentException.ThrowIfNullOrWhiteSpace(repair.Name);
        Table.Register(repair.Name, repair, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's repair of that name (the library's, if any, answers again).</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The repair of that name, or null.</summary>
    public static IJsonRepair? Find(string name) => Table.Find(name);

    /// <summary>The repair of that name; throws when none is registered.</summary>
    public static IJsonRepair Get(string name) => Table.Find(name)
        ?? throw new NotSupportedException($"JSON repair '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with JsonRepairs.Register.");

    /// <summary>The library's repair of that name, or null.</summary>
    public static IJsonRepair? Default(string name) => Table.Default(name);

    /// <summary>The assembly that registered the repair of that name.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>How an app's repair of that name is run against the library's.</summary>
    public static void SetPolicy(string name, SlotPolicy policy, double shadowRate = Slot.DefaultShadowRate) => Table.SetPolicy(name, policy, shadowRate);

    /// <summary>One line a repair.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(r => $"{r.Name}: {r.Summary}"));

    /// <summary>The JSON of <paramref name="text"/> by the repair <paramref name="name"/> (the library's lenient one by default); null when it has none.</summary>
    public static JsonRepairResult? Parse(string text, string name = Lenient) => Get(name).Repair(text);

    private sealed class GuardedRepair(Slot slot, IJsonRepair app, IJsonRepair library) : IJsonRepair
    {
        public string Name => app.Name;

        public string Summary => app.Summary;

        public JsonRepairResult? Repair(string text) =>
            slot.Call(() => app.Repair(text), () => library.Repair(text),
                (a, b) => Same(a, b) ? null : $"{a?.Json?.ToJsonString() ?? "none"} against the library's {b?.Json?.ToJsonString() ?? "none"}");

        private static bool Same(JsonRepairResult? a, JsonRepairResult? b) =>
            a is null ? b is null : b is not null && JsonNode.DeepEquals(a.Json, b.Json);
    }
}
