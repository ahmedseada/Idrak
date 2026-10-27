// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text.Json.Nodes;
using Idrak.Generation;

namespace Idrak.Cli.Shared;

/// <summary>
/// Tools from assemblies (<c>--tools FILE.dll</c> of run, chat, agent and serve; <c>idrak tools list/test</c>;
/// <c>idrak mcp serve</c>): the tools each assembly declares with <see cref="ToolProviderAttribute"/>
/// (<see cref="ToolProviders.Of"/>, the attributes read once per assembly).
/// </summary>
internal static class ToolAssemblies
{
    /// <summary>
    /// The feature switch saying whether the commands may load assemblies from files (<c>-P</c> plug-ins, <c>--tools</c>
    /// DLLs): on unless a host turns it off. A trimmed host turns it off in its project
    /// (<c>&lt;RuntimeHostConfigurationOption Include="Idrak.Cli.LoadAssemblies" Value="false" Trim="true" /&gt;</c>): the
    /// trimmer then drops the loading code, and those options are refused, named.
    /// </summary>
    public const string LoadAssembliesSwitch = "Idrak.Cli.LoadAssemblies";

    /// <summary>Whether assemblies may be loaded from files (<see cref="LoadAssembliesSwitch"/>, on by default).</summary>
    [FeatureSwitchDefinition(LoadAssembliesSwitch)]
    [FeatureGuard(typeof(RequiresUnreferencedCodeAttribute))]
    [FeatureGuard(typeof(RequiresDynamicCodeAttribute))]
    [UnconditionalSuppressMessage("Trimming", "IL4000", Justification = "A feature switch (FeatureSwitchDefinition): a trimmed host sets it off and the trimmer "
        + "substitutes false, removing the loading code; where it is on (the tool, not trimmed) loading runs as it always did.")]
    public static bool CanLoad => !AppContext.TryGetSwitch(LoadAssembliesSwitch, out bool on) || on;

    /// <summary>The refusal of <paramref name="what"/> (an option loading an assembly) where <see cref="CanLoad"/> is off.</summary>
    public static CommandUnavailableException Refusal(string what) =>
        new($"{what} loads an assembly from a file, which this build cannot: its code was trimmed to what the build itself uses ({LoadAssembliesSwitch} is off).");

    /// <summary>The tools of the assemblies at <paramref name="paths"/>, in the order given (none: an empty list).</summary>
    /// <exception cref="InvalidOperationException">An assembly declares no tools, or two tools share a name.</exception>
    /// <exception cref="CommandUnavailableException">Assemblies cannot be loaded here (<see cref="CanLoad"/>).</exception>
    public static IReadOnlyList<ITool> Load(IEnumerable<string> paths)
    {
        var list = paths.ToList();
        if (list.Count == 0)
        {
            return [];
        }

        if (CanLoad)
        {
            var tools = new List<ITool>();
            foreach (string path in list)
            {
                tools.AddRange(FromFile(path));
            }

            var duplicate = tools.GroupBy(t => t.Definition.Name).FirstOrDefault(g => g.Count() > 1);
            return duplicate is null ? tools : throw new InvalidOperationException($"Two tools are named '{duplicate.Key}'.");
        }

        throw Refusal("--tools");
    }

    /// <summary>The tools of the assembly at <paramref name="path"/>, in the order it declares them.</summary>
    /// <exception cref="UsageException">The file does not exist.</exception>
    /// <exception cref="InvalidOperationException">The assembly declares no tools.</exception>
    /// <exception cref="CommandUnavailableException">Assemblies cannot be loaded here (<see cref="CanLoad"/>).</exception>
    public static IReadOnlyList<ITool> Load(string path)
    {
        if (CanLoad)
        {
            return FromFile(path);
        }

        throw Refusal(path);
    }

    [RequiresUnreferencedCode("Loads an assembly from a file.")]
    [RequiresDynamicCode("Loads an assembly from a file.")]
    private static IReadOnlyList<ITool> FromFile(string path)
    {
        string full = Path.GetFullPath(path);
        if (!File.Exists(full))
        {
            throw new UsageException($"Tool assembly not found: {full}");
        }

        var tools = ToolProviders.Of(Assembly.LoadFrom(full));
        return tools.Count > 0 ? tools
            : throw new InvalidOperationException($"{full} declares no tools: add [assembly: ToolProvider(typeof(YourTools))] with YourTools : IToolProvider "
                                                  + "(Idrak.Abstraction.Generation), whose Tools (ITool) can be ToolProviders.FromMethods(this) for its [Tool(name, description)] methods.");
    }

    /// <summary>A registry of <paramref name="tools"/>, or null when there are none.</summary>
    public static ToolRegistry? Registry(IReadOnlyList<ITool> tools) => tools.Count == 0 ? null : ToolRegistry.Create().Add(tools).Build();

    /// <summary>Sample arguments for a tool's JSON schema (for <c>idrak tools test</c>): its examples, defaults or a value of each type.</summary>
    public static JsonObject SampleArguments(JsonNode? schema)
    {
        var arguments = new JsonObject();
        if (schema?["properties"] is JsonObject properties)
        {
            foreach (var (name, property) in properties)
            {
                arguments[name] = Sample(property);
            }
        }

        return arguments;
    }

    private static JsonNode? Sample(JsonNode? property)
    {
        if (property?["default"] is { } d)
        {
            return d.DeepClone();
        }

        if (property?["examples"] is JsonArray { Count: > 0 } examples)
        {
            return examples[0]?.DeepClone();
        }

        if (property?["enum"] is JsonArray { Count: > 0 } options)
        {
            return options[0]?.DeepClone();
        }

        string? type = property?["type"] is JsonValue v && v.TryGetValue(out string? t) ? t : null;
        return type switch
        {
            "string" => "test",
            "integer" => 1,
            "number" => 1.5,
            "boolean" => true,
            "array" => new JsonArray(),
            "object" => SampleArguments(property),
            _ => null,
        };
    }
}
