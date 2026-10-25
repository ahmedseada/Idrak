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
/// A tool a chat model may call: the definition shown to it and what runs a call. <see cref="ToolRegistry"/> holds tools
/// by their definition's name, validates each call's arguments against the definition's schema, applies its rules and
/// then invokes the tool. <see cref="Tool"/> makes one from a function or a delegate.
/// </summary>
public interface ITool
{
    /// <summary>Name, description and JSON-schema parameters.</summary>
    ToolDefinition Definition { get; }

    /// <summary>
    /// Runs the tool with validated <paramref name="arguments"/>; returns the text sent back to the model. Only the waiting
    /// is asynchronous: an implementation reads its arguments in a synchronous step before it awaits anything.
    /// </summary>
    Task<string> InvokeAsync(JsonObject arguments, CancellationToken cancellationToken);
}

/// <summary>
/// A tool (<see cref="ITool"/>) from a function: the definition shown to the model and the function that runs it. Build
/// one with <see cref="Create"/> (explicit JSON schema, trimming and AOT safe) or <see cref="FromDelegate"/> (schema read
/// from the parameters).
/// </summary>
/// <param name="Definition">Name, description and JSON-schema parameters.</param>
/// <param name="Invoke">Runs the tool with validated arguments; returns the text sent back to the model.</param>
public sealed record Tool(ToolDefinition Definition, Func<JsonObject, CancellationToken, Task<string>> Invoke) : ITool
{
    /// <inheritdoc />
    public Task<string> InvokeAsync(JsonObject arguments, CancellationToken cancellationToken) => Invoke(arguments, cancellationToken);

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
        var awaitResult = ToolSchema.Awaiter(method.ReturnType);           // how the method's result is awaited, found once
        return Create(name, description, parametersSchema, async (args, token) =>
        {
            object? result = Call(method, target, parameters, args, token);
            if (awaitResult is not null)
            {
                result = await awaitResult(result).ConfigureAwait(false);
            }

            return ToolSchema.Text(result);
        });
    }

    // The synchronous core of a call: the arguments read from their JSON, the method called (a task it returns is awaited by the caller).
    [RequiresUnreferencedCode("Reflection.")]
    [RequiresDynamicCode("Reflection.")]
    private static object? Call(MethodInfo method, object? target, ParameterInfo[] parameters, JsonObject args, CancellationToken token)
    {
        var values = new object?[parameters.Length];
        for (int i = 0; i < parameters.Length; i++)
        {
            var p = parameters[i];
            values[i] = p.ParameterType == typeof(CancellationToken) ? token
                : args[p.Name!] is { } node ? node.Deserialize(p.ParameterType, ToolSchema.Json)
                : p.HasDefaultValue ? p.DefaultValue : null;
        }

        try
        {
            return method.Invoke(target, values);
        }
        catch (TargetInvocationException ex) when (ex.InnerException is not null)
        {
            System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(ex.InnerException).Throw();
            throw;
        }
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

    /// <summary>
    /// How a method returning <paramref name="returnType"/> is awaited: a function from its result to the awaited value
    /// (null for a task without one), made once from the type; null when the result is not awaitable (it is the value).
    /// </summary>
    [RequiresUnreferencedCode("Reflection.")]
    [RequiresDynamicCode("Reflection.")]
    public static Func<object?, Task<object?>>? Awaiter(Type returnType)
    {
        if (returnType == typeof(Task) || returnType == typeof(ValueTask))
        {
            return returnType == typeof(Task) ? AwaitTask : AwaitValueTask;
        }

        if (returnType.IsGenericType && (returnType.GetGenericTypeDefinition() == typeof(Task<>) || returnType.GetGenericTypeDefinition() == typeof(ValueTask<>)))
        {
            var helper = returnType.GetGenericTypeDefinition() == typeof(Task<>) ? nameof(AwaitTaskOf) : nameof(AwaitValueTaskOf);
            return typeof(ToolSchema).GetMethod(helper, BindingFlags.NonPublic | BindingFlags.Static)!.MakeGenericMethod(returnType.GetGenericArguments()[0])
                .CreateDelegate<Func<object?, Task<object?>>>();
        }

        // A method declared to return object (or a base type) may still hand back a task: awaited by what it is.
        return returnType.IsAssignableFrom(typeof(Task)) || returnType.IsAssignableFrom(typeof(ValueTask)) ? AwaitAny : null;
    }

    private static async Task<object?> AwaitTask(object? result)
    {
        await ((Task)result!).ConfigureAwait(false);
        return null;
    }

    private static async Task<object?> AwaitValueTask(object? result)
    {
        await ((ValueTask)result!).ConfigureAwait(false);
        return null;
    }

    private static async Task<object?> AwaitTaskOf<T>(object? result) => await ((Task<T>)result!).ConfigureAwait(false);

    private static async Task<object?> AwaitValueTaskOf<T>(object? result) => await ((ValueTask<T>)result!).ConfigureAwait(false);

    // A result whose declared type does not say whether it is a task (a method returning object): awaited by what it is.
    [RequiresUnreferencedCode("Reflection.")]
    [RequiresDynamicCode("Reflection.")]
    private static async Task<object?> AwaitAny(object? result)
    {
        if (result is ValueTask valueTask)
        {
            await valueTask.ConfigureAwait(false);
            return null;
        }

        if (result?.GetType() is { IsGenericType: true } type && type.GetGenericTypeDefinition() == typeof(ValueTask<>))
        {
            return await Awaiter(type)!(result).ConfigureAwait(false);
        }

        if (result is not Task task)
        {
            return result;
        }

        await task.ConfigureAwait(false);
        for (var t = task.GetType(); t is not null && t != typeof(Task); t = t.BaseType)
        {
            if (t.IsGenericType && t.GetGenericTypeDefinition() == typeof(Task<>))
            {
                return t.GetGenericArguments()[0].Name == "VoidTaskResult" ? null : await Awaiter(t)!(task).ConfigureAwait(false);   // an async method's Task
            }
        }

        return null;
    }

    /// <summary>The text sent to the model for a tool's result: a string as is, JSON otherwise ("" for none).</summary>
    [RequiresUnreferencedCode("Reflection.")]
    [RequiresDynamicCode("Reflection.")]
    public static string Text(object? result) => result switch
    {
        null => "",
        string text => text,
        JsonNode json => json.ToJsonString(),
        _ => JsonSerializer.Serialize(result, result.GetType(), ToolSchema.Json),
    };

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
