using System.ComponentModel;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Net;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Schema;
using System.Text.RegularExpressions;
using Idrak.Diagnostics;

namespace Idrak.Generation;

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

/// <summary>The outcome of one tool call.</summary>
/// <param name="Call">The call the model made.</param>
/// <param name="Content">The text returned to the model (the tool's result, or a JSON error object).</param>
/// <param name="Succeeded">False when the tool was unknown, not allowed, denied, given invalid arguments, timed out or threw.</param>
/// <param name="Error">What went wrong, or null.</param>
/// <param name="Duration">Time spent.</param>
public sealed record ToolResult(ToolCall Call, string Content, bool Succeeded, string? Error, TimeSpan Duration)
{
    /// <summary>The <c>tool</c> message that sends this result back to the model.</summary>
    public ChatMessage ToMessage() => new("tool", Content, ToolName: Call.Name);
}

/// <summary>
/// The tools a chat model may use, with their safety rules. Build with <see cref="Create"/>; pass to a
/// <see cref="Conversation"/>, the inference engine or the ASP.NET Core endpoints. Arguments are validated against each
/// tool's schema before it runs; problems go back to the model as an error it can correct.
/// </summary>
public sealed class ToolRegistry
{
    private readonly Dictionary<string, Tool> _tools;
    private readonly Dictionary<string, Func<JsonObject, bool>> _allow;
    private readonly Dictionary<string, Func<ToolCall, CancellationToken, ValueTask<bool>>> _approval;
    private readonly TimeSpan? _timeout;

    internal ToolRegistry(Dictionary<string, Tool> tools, Dictionary<string, Func<JsonObject, bool>> allow,
        Dictionary<string, Func<ToolCall, CancellationToken, ValueTask<bool>>> approval, TimeSpan? timeout, bool parallel)
    {
        _tools = tools;
        _allow = allow;
        _approval = approval;
        _timeout = timeout;
        Parallel = parallel;
        Definitions = [.. tools.Values.Select(t => t.Definition)];
    }

    /// <summary>Starts a registry.</summary>
    public static ToolRegistryBuilder Create() => new();

    /// <summary>The definitions shown to the model, in the order the tools were added.</summary>
    public IReadOnlyList<ToolDefinition> Definitions { get; }

    /// <summary>Whether the calls of one reply run concurrently.</summary>
    public bool Parallel { get; }

    /// <summary>Validates, checks the rules for and runs one call.</summary>
    public async Task<ToolResult> InvokeAsync(ToolCall call, CancellationToken cancellationToken = default)
    {
        var clock = Stopwatch.StartNew();
        string? error = null;
        string content;
        try
        {
            if (!_tools.TryGetValue(call.Name, out var tool))
            {
                error = $"unknown tool '{call.Name}'; available: {string.Join(", ", _tools.Keys)}";
            }
            else if (ToolSchema.Validate(tool.Definition.Parameters, call.Arguments) is { } invalid)
            {
                error = invalid;
            }
            else if (_allow.TryGetValue(call.Name, out var allowed) && !allowed(call.Arguments))
            {
                error = $"these arguments are not allowed for '{call.Name}'";
            }
            else if (_approval.TryGetValue(call.Name, out var approve) && !await approve(call, cancellationToken).ConfigureAwait(false))
            {
                error = $"the call to '{call.Name}' was not approved";
            }

            if (error is null)
            {
                using var limit = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
                if (_timeout is { } timeout)
                {
                    limit.CancelAfter(timeout);
                }

                try
                {
                    content = await tool!.Invoke(call.Arguments, limit.Token).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (!cancellationToken.IsCancellationRequested)
                {
                    error = $"'{call.Name}' timed out after {_timeout!.Value.TotalSeconds:0.###} s";
                    content = "";
                }
            }
            else
            {
                content = "";
            }
        }
        catch (Exception ex) when (ex is not OperationCanceledException || !cancellationToken.IsCancellationRequested)
        {
            error = $"'{call.Name}' failed: {ex.Message}";
            content = "";
        }

        if (error is not null)
        {
            content = new JsonObject { ["error"] = error }.ToJsonString();
        }

        var result = new ToolResult(call, content, error is null, error, clock.Elapsed);
        if (Telemetry.IsEnabled(TelemetryLevel.Tools))
        {
            Telemetry.ToolCall(new ToolCallCompleted(call.Name, call.Arguments.ToJsonString(), result.Duration, result.Succeeded, error));
        }

        return result;
    }

    /// <summary>Runs several calls (concurrently when <see cref="Parallel"/>), returning results in call order.</summary>
    public async Task<IReadOnlyList<ToolResult>> InvokeAsync(IReadOnlyList<ToolCall> calls, CancellationToken cancellationToken = default)
    {
        if (Parallel)
        {
            return await Task.WhenAll(calls.Select(c => InvokeAsync(c, cancellationToken))).ConfigureAwait(false);
        }

        var results = new List<ToolResult>(calls.Count);
        foreach (var call in calls)
        {
            results.Add(await InvokeAsync(call, cancellationToken).ConfigureAwait(false));
        }

        return results;
    }
}

/// <summary>Collects tools and rules for a <see cref="ToolRegistry"/>. Nothing is restricted unless a rule is added.</summary>
public sealed class ToolRegistryBuilder
{
    private readonly Dictionary<string, Tool> _tools = [];
    private readonly Dictionary<string, Func<JsonObject, bool>> _allow = [];
    private readonly Dictionary<string, Func<ToolCall, CancellationToken, ValueTask<bool>>> _approval = [];
    private TimeSpan? _timeout;
    private bool _parallel;

    internal ToolRegistryBuilder()
    {
    }

    /// <summary>Adds a tool.</summary>
    public ToolRegistryBuilder Add(Tool tool)
    {
        if (!_tools.TryAdd(tool.Definition.Name, tool))
        {
            throw new ArgumentException($"A tool named '{tool.Definition.Name}' was already added.", nameof(tool));
        }

        return this;
    }

    /// <summary>Adds several tools (for example those of an MCP server).</summary>
    public ToolRegistryBuilder Add(IEnumerable<Tool> tools)
    {
        foreach (var tool in tools)
        {
            Add(tool);
        }

        return this;
    }

    /// <summary>Adds a tool with an explicit JSON schema (<see cref="Tool.Create"/>).</summary>
    public ToolRegistryBuilder Add(string name, string description, JsonNode parameters, Func<JsonObject, CancellationToken, Task<string>> invoke) =>
        Add(Tool.Create(name, description, parameters, invoke));

    /// <summary>Adds a tool from a delegate (<see cref="Tool.FromDelegate"/>).</summary>
    [RequiresUnreferencedCode("See Tool.FromDelegate.")]
    [RequiresDynamicCode("See Tool.FromDelegate.")]
    public ToolRegistryBuilder Add(string name, string description, Delegate function) => Add(Tool.FromDelegate(name, description, function));

    /// <summary>Adds every method of <paramref name="instance"/> marked with <see cref="ToolAttribute"/> (static and instance methods).</summary>
    [RequiresUnreferencedCode("See Tool.FromDelegate.")]
    [RequiresDynamicCode("See Tool.FromDelegate.")]
    public ToolRegistryBuilder Add(object instance)
    {
        var methods = instance.GetType().GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance | BindingFlags.Static)
            .Select(m => (Method: m, Attribute: m.GetCustomAttribute<ToolAttribute>())).Where(m => m.Attribute is not null).ToList();
        if (methods.Count == 0)
        {
            throw new ArgumentException($"{instance.GetType().Name} has no methods marked [Tool].", nameof(instance));
        }

        foreach (var (method, attribute) in methods)
        {
            Add(Tool.FromMethod(attribute!.Name, attribute.Description, method, method.IsStatic ? null : instance));
        }

        return this;
    }

    /// <summary>Adds the [Tool] methods of a new <typeparamref name="T"/>.</summary>
    [RequiresUnreferencedCode("See Tool.FromDelegate.")]
    [RequiresDynamicCode("See Tool.FromDelegate.")]
    public ToolRegistryBuilder Add<T>() where T : new() => Add(new T());

    /// <summary>Runs <paramref name="name"/> only when <paramref name="allowed"/> accepts the arguments (for example an allowlist of URLs).</summary>
    public ToolRegistryBuilder Allow(string name, Func<JsonObject, bool> allowed)
    {
        _allow[name] = allowed;
        return this;
    }

    /// <summary>Asks <paramref name="approve"/> before every call of <paramref name="name"/>; a false answer sends an error back to the model.</summary>
    public ToolRegistryBuilder RequireApproval(string name, Func<ToolCall, CancellationToken, ValueTask<bool>> approve)
    {
        _approval[name] = approve;
        return this;
    }

    /// <summary>Cancels any tool call that runs longer than <paramref name="limit"/> (it becomes an error for the model).</summary>
    public ToolRegistryBuilder Timeout(TimeSpan limit)
    {
        _timeout = limit;
        return this;
    }

    /// <summary>Runs the calls of one reply concurrently instead of one after another.</summary>
    public ToolRegistryBuilder Parallel(bool parallel = true)
    {
        _parallel = parallel;
        return this;
    }

    /// <summary>Creates the registry; rules must name tools that were added.</summary>
    public ToolRegistry Build()
    {
        foreach (var name in _allow.Keys.Concat(_approval.Keys).Where(n => !_tools.ContainsKey(n)))
        {
            throw new InvalidOperationException($"A rule names '{name}', but no tool with that name was added.");
        }

        return new ToolRegistry(new(_tools), new(_allow), new(_approval), _timeout, _parallel);
    }
}

/// <summary>Built-in tools.</summary>
public static partial class WebTools
{
    /// <summary>
    /// <c>web_fetch</c>: downloads a page and returns its text (tags, scripts and styles removed; whitespace collapsed),
    /// cut to <paramref name="maxCharacters"/>. Only absolute http(s) URLs accepted by <paramref name="allow"/> are fetched.
    /// </summary>
    public static Tool Fetch(HttpClient http, Func<Uri, bool> allow, int maxCharacters)
    {
        ArgumentNullException.ThrowIfNull(allow);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxCharacters);
        var schema = new JsonObject
        {
            ["type"] = "object",
            ["properties"] = new JsonObject { ["url"] = new JsonObject { ["type"] = "string", ["description"] = "Absolute http(s) URL." } },
            ["required"] = new JsonArray("url"),
        };
        return Tool.Create("web_fetch", "Fetch a web page and return its text.", schema, async (args, token) =>
        {
            if (!Uri.TryCreate((string?)args["url"], UriKind.Absolute, out var url) || url.Scheme is not ("http" or "https"))
            {
                throw new ArgumentException("url must be an absolute http or https URL");
            }

            if (!allow(url))
            {
                throw new UnauthorizedAccessException($"{url} is not on the allowlist");
            }

            string html = await http.GetStringAsync(url, token).ConfigureAwait(false);
            string text = WebUtility.HtmlDecode(Tags().Replace(Blocks().Replace(html, " "), " "));
            return CollapseSpaces(text, maxCharacters);
        });
    }

    // Regex.Replace(text, @"\s+", " ").Trim() cut to max characters, in one pass that stops at the cut: \s is
    // char.IsWhiteSpace (as for Trim), so each run of white space becomes one space and none is kept at either end.
    internal static string CollapseSpaces(string text, int max)
    {
        char[] buffer = System.Buffers.ArrayPool<char>.Shared.Rent(Math.Min(text.Length, max));
        int length = 0;
        bool space = false;
        foreach (char c in text)
        {
            if (char.IsWhiteSpace(c))
            {
                space = length > 0;
                continue;
            }

            if (space && length < max)
            {
                buffer[length++] = ' ';
            }

            if (length == max)
            {
                break;
            }

            buffer[length++] = c;
            space = false;
        }

        string result = new(buffer, 0, length);
        System.Buffers.ArrayPool<char>.Shared.Return(buffer);
        return result;
    }

    [GeneratedRegex(@"<(script|style)[^>]*>.*?</\1>", RegexOptions.Singleline | RegexOptions.IgnoreCase)]
    private static partial Regex Blocks();

    [GeneratedRegex("<[^>]+>")]
    private static partial Regex Tags();
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
