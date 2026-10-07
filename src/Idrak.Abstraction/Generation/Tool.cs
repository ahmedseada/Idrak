// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.ComponentModel;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;

namespace Idrak.Abstraction.Generation;

/// <summary>Marks a method as a tool a chat model may call (see <see cref="ToolRegistryBuilder.Add(object)"/>).</summary>
/// <param name="name">The name the model uses to call it.</param>
/// <param name="description">What the tool does, for the model.</param>
[AttributeUsage(AttributeTargets.Method)]
public sealed class ToolAttribute(string name, string description) : Attribute
{
    /// <summary>The name the model uses to call the tool.</summary>
    public string Name { get; } = name;

    /// <summary>What the tool does, for the model.</summary>
    public string Description { get; } = description;
}

/// <summary>
/// A tool: the definition shown to the model and the function that runs it. Build one with <see cref="Create"/>
/// (explicit JSON schema, trimming and AOT safe) or <see cref="FromDelegate"/> (schema read from the parameters).
/// </summary>
/// <param name="Definition">Name, description and JSON-schema parameters.</param>
/// <param name="Invoke">Runs the tool with validated arguments; returns the text sent back to the model.</param>
public sealed record Tool(ToolDefinition Definition, Func<JsonObject, CancellationToken, Task<string>> Invoke)
{
    /// <summary>A tool with an explicit JSON schema (an object schema with "properties" and "required").</summary>
    public static Tool Create(string name, string description, JsonNode parameters, Func<JsonObject, CancellationToken, Task<string>> invoke) =>
        new(new ToolDefinition(name, description, parameters), invoke);

    /// <summary>
    /// A tool from a delegate. The JSON schema is a fixed translation of the parameters: string → "string", bool →
    /// "boolean", integers → "integer", other numbers → "number", enums → "string" with their names, arrays and lists →
    /// "array", other types → their JSON schema; <see cref="DescriptionAttribute"/> on a parameter becomes its description;
    /// parameters without a default value are required; a <see cref="CancellationToken"/> parameter receives the call's token.
    /// The result is sent to the model as is when it is a string, otherwise as JSON (tasks are awaited).
    /// </summary>
    [RequiresUnreferencedCode("Reads parameter types and serializes values with reflection; use Tool.Create with an explicit schema for trimmed or AOT apps.")]
    [RequiresDynamicCode("Reads parameter types and serializes values with reflection; use Tool.Create with an explicit schema for trimmed or AOT apps.")]
    public static Tool FromDelegate(string name, string description, Delegate function) =>
        FromMethod(name, description, function.Method, function.Target);

    [RequiresUnreferencedCode("Reflection.")]
    [RequiresDynamicCode("Reflection.")]
    internal static Tool FromMethod(string name, string description, MethodInfo method, object? target)
    {
        var parameters = method.GetParameters();
        var properties = new JsonObject();
        var required = new JsonArray();
        foreach (var p in parameters.Where(p => p.ParameterType != typeof(CancellationToken)))
        {
            var schema = ToolSchema.For(p.ParameterType);
            if (p.GetCustomAttribute<DescriptionAttribute>() is { } d)
            {
                schema["description"] = d.Description;
            }

            properties[p.Name!] = schema;
            if (!p.HasDefaultValue)
            {
                required.Add(p.Name);
            }
        }

        var parametersSchema = new JsonObject { ["type"] = "object", ["properties"] = properties, ["required"] = required };
        return Create(name, description, parametersSchema, async (args, token) =>
        {
            var values = parameters.Select(p =>
                p.ParameterType == typeof(CancellationToken) ? token
                : args[p.Name!] is { } node ? node.Deserialize(p.ParameterType, ToolSchema.Json)
                : p.HasDefaultValue ? p.DefaultValue : null).ToArray();
            object? result;
            try
            {
                result = method.Invoke(target, values);
            }
            catch (TargetInvocationException ex) when (ex.InnerException is not null)
            {
                throw ex.InnerException;
            }

            result = await ToolSchema.AwaitResult(result).ConfigureAwait(false);
            return result switch
            {
                null => "",
                string text => text,
                JsonNode json => json.ToJsonString(),
                _ => JsonSerializer.Serialize(result, result.GetType(), ToolSchema.Json),
            };
        });
    }
}


/// <summary>JSON-schema helpers for tools.</summary>
internal static class ToolSchema
{
    /// <summary>Web defaults (camelCase) plus enums as names, matching the enum schemas.</summary>
    [field: MaybeNull]
    public static JsonSerializerOptions Json
    {
        [RequiresUnreferencedCode("Reflection.")]
        [RequiresDynamicCode("Reflection.")]
        get => field ??= new JsonSerializerOptions(JsonSerializerOptions.Web) { Converters = { new System.Text.Json.Serialization.JsonStringEnumConverter() } };
    }

    [RequiresUnreferencedCode("Reflection.")]
    [RequiresDynamicCode("Reflection.")]
    public static JsonObject For(Type type)
    {
        type = Nullable.GetUnderlyingType(type) ?? type;
        if (type == typeof(string) || type == typeof(Uri) || type == typeof(char))
        {
            return new JsonObject { ["type"] = "string" };
        }

        if (type == typeof(bool))
        {
            return new JsonObject { ["type"] = "boolean" };
        }

        if (type == typeof(int) || type == typeof(long) || type == typeof(short) || type == typeof(byte) || type == typeof(uint) || type == typeof(ulong))
        {
            return new JsonObject { ["type"] = "integer" };
        }

        if (type == typeof(float) || type == typeof(double) || type == typeof(decimal))
        {
            return new JsonObject { ["type"] = "number" };
        }

        if (type.IsEnum)
        {
            return new JsonObject { ["type"] = "string", ["enum"] = new JsonArray([.. Enum.GetNames(type).Select(n => (JsonNode)n)]) };
        }

        var element = type.IsArray ? type.GetElementType()
            : type.IsGenericType && type.GetGenericArguments() is [var arg] && typeof(IEnumerable<>).MakeGenericType(arg).IsAssignableFrom(type) ? arg
            : null;
        if (element is not null)
        {
            return new JsonObject { ["type"] = "array", ["items"] = For(element) };
        }

        return Json.GetJsonSchemaAsNode(type) as JsonObject ?? new JsonObject { ["type"] = "object" };
    }

    [RequiresUnreferencedCode("Reflection.")]
    [RequiresDynamicCode("Reflection.")]
    public static async Task<object?> AwaitResult(object? result)
    {
        if (result is null)
        {
            return null;
        }

        var type = result.GetType();
        if (type.IsGenericType && type.GetGenericTypeDefinition() == typeof(ValueTask<>))
        {
            result = type.GetMethod("AsTask")!.Invoke(result, null);
        }
        else if (result is ValueTask valueTask)
        {
            await valueTask.ConfigureAwait(false);
            return null;
        }

        if (result is Task task)
        {
            await task.ConfigureAwait(false);
            var taskType = task.GetType();
            return taskType.IsGenericType && taskType.GetGenericArguments()[0].Name != "VoidTaskResult"
                ? taskType.GetProperty("Result")!.GetValue(task) : null;
        }

        return result;
    }

    /// <summary>Checks required properties and basic types; returns an error message, or null when the arguments are valid.</summary>
    public static string? Validate(JsonNode? schema, JsonObject arguments)
    {
        if (schema is not JsonObject s || s["properties"] is not JsonObject properties)
        {
            return null;
        }

        foreach (var name in (s["required"] as JsonArray ?? []).Select(n => (string)n!))
        {
            if (arguments[name] is null)
            {
                return $"argument '{name}' is required ({Describe(properties[name])})";
            }
        }

        foreach (var (name, value) in arguments)
        {
            if (properties[name] is JsonObject property && value is not null && Mismatch(property, value) is { } expected)
            {
                return $"argument '{name}' must be {expected}";
            }
        }

        return null;
    }

    // Works for parsed numbers and for values created in code (a JsonValue holding an int does not convert to long).
    private static bool IsWhole(JsonNode value) =>
        decimal.TryParse(value.ToJsonString(), System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out var d)
        && d == decimal.Truncate(d);

    private static string Describe(JsonNode? property) => TypeOf(property) ?? "any type";

    // "type" may also be an array (["string", "null"]) in schemas from other sources; only a single type is checked.
    private static string? TypeOf(JsonNode? property) => property?["type"] is JsonValue v && v.TryGetValue(out string? type) ? type : null;

    private static string? Mismatch(JsonObject property, JsonNode value)
    {
        var kind = value.GetValueKind();
        string? type = TypeOf(property);
        bool ok = type switch
        {
            "string" => kind == JsonValueKind.String,
            "boolean" => kind is JsonValueKind.True or JsonValueKind.False,
            "integer" => kind == JsonValueKind.Number && IsWhole(value),
            "number" => kind == JsonValueKind.Number,
            "array" => kind == JsonValueKind.Array,
            "object" => kind == JsonValueKind.Object,
            _ => true,
        };
        if (!ok)
        {
            return $"a{(type is "integer" or "array" or "object" ? "n" : "")} {type}";
        }

        if (property["enum"] is JsonArray options && !options.Any(o => JsonNode.DeepEquals(o, value)))
        {
            return $"one of {string.Join(", ", options.Select(o => o?.ToString() ?? "null"))}";
        }

        return null;
    }
}
