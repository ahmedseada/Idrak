// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Linq.Expressions;
using System.Reflection;
using System.Text.Json.Nodes;
using Idrak.Generation;

namespace Idrak.Cli.Shared;

/// <summary>
/// Tools from assemblies (<c>--tools FILE.dll</c>, <c>idrak tools list/test</c>; serve may reuse it): every public
/// method marked <see cref="ToolAttribute"/> on the assembly's public types (static methods, and instance methods of a
/// type with a public parameterless constructor), and every public static property or field of type <see cref="Tool"/>
/// or <c>IEnumerable&lt;Tool&gt;</c>.
/// </summary>
internal static class ToolAssemblies
{
    /// <summary>The tools of the assemblies at <paramref name="paths"/>, in the order found.</summary>
    [RequiresUnreferencedCode("Reads tool methods with reflection.")]
    [RequiresDynamicCode("Reads tool methods with reflection.")]
    public static IReadOnlyList<Tool> Load(IEnumerable<string> paths)
    {
        var tools = new List<Tool>();
        foreach (string path in paths)
        {
            string full = Path.GetFullPath(path);
            if (!File.Exists(full))
            {
                throw new UsageException($"Tool assembly not found: {full}");
            }

            var assembly = Assembly.LoadFrom(full);
            int before = tools.Count;
            foreach (var type in assembly.GetExportedTypes())
            {
                tools.AddRange(FromType(type));
            }

            if (tools.Count == before)
            {
                throw new InvalidOperationException($"{full} has no tools: mark public methods with [Tool(name, description)] (Idrak.Abstraction.Generation).");
            }
        }

        var duplicate = tools.GroupBy(t => t.Definition.Name).FirstOrDefault(g => g.Count() > 1);
        return duplicate is null ? tools : throw new InvalidOperationException($"Two tools are named '{duplicate.Key}'.");
    }

    /// <summary>A registry of <paramref name="tools"/>, or null when there are none.</summary>
    public static ToolRegistry? Registry(IReadOnlyList<Tool> tools) => tools.Count == 0 ? null : ToolRegistry.Create().Add(tools).Build();

    [RequiresUnreferencedCode("Reads tool methods with reflection.")]
    [RequiresDynamicCode("Reads tool methods with reflection.")]
    private static IEnumerable<Tool> FromType(Type type)
    {
        const BindingFlags Public = BindingFlags.Public | BindingFlags.Static | BindingFlags.Instance | BindingFlags.DeclaredOnly;
        object? instance = null;
        foreach (var method in type.GetMethods(Public))
        {
            if (method.GetCustomAttribute<ToolAttribute>() is not { } attribute || method.IsGenericMethodDefinition)
            {
                continue;
            }

            object? target = null;
            if (!method.IsStatic)
            {
                if (type.IsAbstract || type.GetConstructor(Type.EmptyTypes) is null)
                {
                    throw new InvalidOperationException($"{type.FullName}.{method.Name} is an instance tool, but {type.Name} has no public parameterless constructor.");
                }

                target = instance ??= Activator.CreateInstance(type);
            }

            var delegateType = Expression.GetDelegateType([.. method.GetParameters().Select(p => p.ParameterType), method.ReturnType]);
            yield return Tool.FromDelegate(attribute.Name, attribute.Description, method.CreateDelegate(delegateType, target));
        }

        foreach (var member in type.GetProperties(BindingFlags.Public | BindingFlags.Static).Select(p => (p.PropertyType, Get: (Func<object?>)(() => p.GetValue(null))))
                     .Concat(type.GetFields(BindingFlags.Public | BindingFlags.Static).Select(f => (Type: f.FieldType, Get: (Func<object?>)(() => f.GetValue(null))))))
        {
            if (member.Item1 == typeof(Tool) && member.Get() is Tool tool)
            {
                yield return tool;
            }
            else if (typeof(IEnumerable<Tool>).IsAssignableFrom(member.Item1) && member.Get() is IEnumerable<Tool> many)
            {
                foreach (var each in many)
                {
                    yield return each;
                }
            }
        }
    }

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
