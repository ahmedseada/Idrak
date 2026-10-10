// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Reflection;
using System.Text.Json.Nodes;

namespace Idrak.Abstraction.Generation;

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
/// The tools a chat model may use: their definitions, shown to the model, and a way to run the calls the model makes.
/// Conversations, the inference engine's chat models, the ASP.NET Core chat endpoint, the MCP bridge and
/// <see cref="ChatTools.WithTools"/> all take one. <see cref="ToolRegistry"/> is the default (validation, allow rules,
/// approvals, a timeout); an application may supply its own.
/// </summary>
public interface IToolRegistry
{
    /// <summary>The definitions shown to the model.</summary>
    IReadOnlyList<ToolDefinition> Definitions { get; }

    /// <summary>
    /// Runs one call. A problem (an unknown tool, invalid arguments, a refusal, a timeout, an exception) gives a failed
    /// <see cref="ToolResult"/> whose content tells the model what went wrong, not an exception.
    /// </summary>
    Task<ToolResult> InvokeAsync(ToolCall call, CancellationToken cancellationToken = default);

    /// <summary>Runs the calls of one reply, returning results in call order.</summary>
    Task<IReadOnlyList<ToolResult>> InvokeAsync(IReadOnlyList<ToolCall> calls, CancellationToken cancellationToken = default);
}

/// <summary>
/// The default <see cref="IToolRegistry"/>: tools with their safety rules. Build with <see cref="Create"/>. Arguments
/// are validated against each tool's schema before it runs; problems go back to the model as an error it can correct.
/// </summary>
public sealed class ToolRegistry : IToolRegistry
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
                    error = string.Create(System.Globalization.CultureInfo.InvariantCulture, $"'{call.Name}' timed out after {_timeout!.Value.TotalSeconds:0.###} s");   // read by the model
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
