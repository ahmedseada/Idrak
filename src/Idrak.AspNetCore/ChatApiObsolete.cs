// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Idrak.Generation;

namespace Idrak.AspNetCore;

// The chat API's former names, kept for one release so applications move without breaking. Each type converts to and
// from its new name implicitly, so code that builds or reads the old types keeps compiling (with an obsolete warning).

#pragma warning disable CS0618 // the forwarders refer to each other

/// <summary>Former name of <see cref="ChatApiOptions"/>.</summary>
[Obsolete("Use ChatApiOptions (and MapChatApi); this name is removed in the next release.")]
public sealed class OllamaApiOptions
{
    internal OllamaApiOptions(ChatApiOptions options) => Options = options;

    internal ChatApiOptions Options { get; }

    /// <summary>See <see cref="ChatApiOptions.Tools"/>.</summary>
    public OllamaApiOptions Tools(ToolExecution execution, int? maxRounds = null)
    {
        Options.Tools(execution, maxRounds);
        return this;
    }

    /// <summary>See <see cref="ChatApiOptions.ModelName"/>.</summary>
    public OllamaApiOptions ModelName(string name)
    {
        Options.ModelName(name);
        return this;
    }

    /// <summary>See <see cref="ChatApiOptions.Version"/>.</summary>
    public OllamaApiOptions Version(string version)
    {
        Options.Version(version);
        return this;
    }
}

/// <summary>Former name of <see cref="ChatApiRequest"/>.</summary>
[Obsolete("Use ChatApiRequest; this name is removed in the next release.")]
public sealed record OllamaChatRequest(
    [property: JsonPropertyName("model")] string? Model,
    [property: JsonPropertyName("messages")] List<OllamaMessage> Messages,
    [property: JsonPropertyName("stream")] bool? Stream = null,
    [property: JsonPropertyName("think")] JsonElement? Think = null,
    [property: JsonPropertyName("keep_alive")] JsonElement? KeepAlive = null,
    [property: JsonPropertyName("options")] Dictionary<string, JsonElement>? Options = null,
    [property: JsonPropertyName("tools")] List<OllamaTool>? Tools = null)
{
    /// <summary>The request under its new name.</summary>
    public static implicit operator ChatApiRequest(OllamaChatRequest r) =>
        new(r.Model, r.Messages?.Select(m => (ChatApiMessage)m).ToList()!, r.Stream, r.Think, r.KeepAlive, r.Options, r.Tools?.Select(t => (ChatApiTool)t).ToList());

    /// <summary>The request under its former name.</summary>
    public static implicit operator OllamaChatRequest(ChatApiRequest r) =>
        new(r.Model, r.Messages?.Select(m => (OllamaMessage)m).ToList()!, r.Stream, r.Think, r.KeepAlive, r.Options, r.Tools?.Select(t => (OllamaTool)t).ToList());
}

/// <summary>Former name of <see cref="ChatApiMessage"/>.</summary>
[Obsolete("Use ChatApiMessage; this name is removed in the next release.")]
public sealed record OllamaMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string? Content = null,
    [property: JsonPropertyName("thinking"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Thinking = null,
    [property: JsonPropertyName("tool_calls"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] List<OllamaToolCall>? ToolCalls = null,
    [property: JsonPropertyName("tool_name"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? ToolName = null)
{
    /// <summary>The message under its new name.</summary>
    public static implicit operator ChatApiMessage(OllamaMessage m) =>
        new(m.Role, m.Content, m.Thinking, m.ToolCalls?.Select(c => (ChatApiToolCall)c).ToList(), m.ToolName);

    /// <summary>The message under its former name.</summary>
    public static implicit operator OllamaMessage(ChatApiMessage m) =>
        new(m.Role, m.Content, m.Thinking, m.ToolCalls?.Select(c => (OllamaToolCall)c).ToList(), m.ToolName);
}

/// <summary>Former name of <see cref="ChatApiTool"/>.</summary>
[Obsolete("Use ChatApiTool; this name is removed in the next release.")]
public sealed record OllamaTool(
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("function")] OllamaFunction Function)
{
    /// <summary>The tool under its new name.</summary>
    public static implicit operator ChatApiTool(OllamaTool t) => new(t.Type, t.Function);

    /// <summary>The tool under its former name.</summary>
    public static implicit operator OllamaTool(ChatApiTool t) => new(t.Type, t.Function);
}

/// <summary>Former name of <see cref="ChatApiFunction"/>.</summary>
[Obsolete("Use ChatApiFunction; this name is removed in the next release.")]
public sealed record OllamaFunction(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("description")] string? Description = null,
    [property: JsonPropertyName("parameters")] JsonNode? Parameters = null)
{
    /// <summary>The function under its new name.</summary>
    public static implicit operator ChatApiFunction(OllamaFunction f) => new(f.Name, f.Description, f.Parameters);

    /// <summary>The function under its former name.</summary>
    public static implicit operator OllamaFunction(ChatApiFunction f) => new(f.Name, f.Description, f.Parameters);
}

/// <summary>Former name of <see cref="ChatApiToolCall"/>.</summary>
[Obsolete("Use ChatApiToolCall; this name is removed in the next release.")]
public sealed record OllamaToolCall([property: JsonPropertyName("function")] OllamaCalledFunction Function)
{
    /// <summary>The call under its new name.</summary>
    public static implicit operator ChatApiToolCall(OllamaToolCall c) => new(c.Function);

    /// <summary>The call under its former name.</summary>
    public static implicit operator OllamaToolCall(ChatApiToolCall c) => new(c.Function);
}

/// <summary>Former name of <see cref="ChatApiCalledFunction"/>.</summary>
[Obsolete("Use ChatApiCalledFunction; this name is removed in the next release.")]
public sealed record OllamaCalledFunction(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("arguments")] JsonObject Arguments,
    [property: JsonPropertyName("index"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? Index = null)
{
    /// <summary>The function under its new name.</summary>
    public static implicit operator ChatApiCalledFunction(OllamaCalledFunction f) => new(f.Name, f.Arguments, f.Index);

    /// <summary>The function under its former name.</summary>
    public static implicit operator OllamaCalledFunction(ChatApiCalledFunction f) => new(f.Name, f.Arguments, f.Index);
}

/// <summary>Former name of <see cref="ChatApiResponse"/>.</summary>
[Obsolete("Use ChatApiResponse; this name is removed in the next release.")]
public sealed record OllamaChatResponse(
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("created_at")] DateTimeOffset CreatedAt,
    [property: JsonPropertyName("message")] OllamaMessage Message,
    [property: JsonPropertyName("done")] bool Done,
    [property: JsonPropertyName("done_reason"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? DoneReason = null,
    [property: JsonPropertyName("total_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? TotalDuration = null,
    [property: JsonPropertyName("load_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? LoadDuration = null,
    [property: JsonPropertyName("prompt_eval_count"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? PromptEvalCount = null,
    [property: JsonPropertyName("prompt_eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? PromptEvalDuration = null,
    [property: JsonPropertyName("eval_count"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? EvalCount = null,
    [property: JsonPropertyName("eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? EvalDuration = null)
{
    /// <summary>The response under its new name.</summary>
    public static implicit operator ChatApiResponse(OllamaChatResponse r) =>
        new(r.Model, r.CreatedAt, r.Message, r.Done, r.DoneReason, r.TotalDuration, r.LoadDuration, r.PromptEvalCount, r.PromptEvalDuration, r.EvalCount, r.EvalDuration);

    /// <summary>The response under its former name.</summary>
    public static implicit operator OllamaChatResponse(ChatApiResponse r) =>
        new(r.Model, r.CreatedAt, r.Message, r.Done, r.DoneReason, r.TotalDuration, r.LoadDuration, r.PromptEvalCount, r.PromptEvalDuration, r.EvalCount, r.EvalDuration);
}

/// <summary>Former name of <see cref="ChatApiModelTag"/>.</summary>
[Obsolete("Use ChatApiModelTag; this name is removed in the next release.")]
public sealed record OllamaModelTag(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("modified_at")] DateTimeOffset ModifiedAt,
    [property: JsonPropertyName("size")] long Size,
    [property: JsonPropertyName("details")] OllamaModelDetails Details)
{
    /// <summary>The entry under its new name.</summary>
    public static implicit operator ChatApiModelTag(OllamaModelTag t) => new(t.Name, t.Model, t.ModifiedAt, t.Size, t.Details);

    /// <summary>The entry under its former name.</summary>
    public static implicit operator OllamaModelTag(ChatApiModelTag t) => new(t.Name, t.Model, t.ModifiedAt, t.Size, t.Details);
}

/// <summary>Former name of <see cref="ChatApiModelDetails"/>.</summary>
[Obsolete("Use ChatApiModelDetails; this name is removed in the next release.")]
public sealed record OllamaModelDetails(
    [property: JsonPropertyName("format")] string Format,
    [property: JsonPropertyName("family")] string Family,
    [property: JsonPropertyName("parameter_size")] string ParameterSize,
    [property: JsonPropertyName("context_length")] int ContextLength)
{
    /// <summary>The details under their new name.</summary>
    public static implicit operator ChatApiModelDetails(OllamaModelDetails d) => new(d.Format, d.Family, d.ParameterSize, d.ContextLength);

    /// <summary>The details under their former name.</summary>
    public static implicit operator OllamaModelDetails(ChatApiModelDetails d) => new(d.Format, d.Family, d.ParameterSize, d.ContextLength);
}

/// <summary>Former name of <see cref="ChatApiRunningModel"/>.</summary>
[Obsolete("Use ChatApiRunningModel; this name is removed in the next release.")]
public sealed record OllamaRunningModel(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("size")] long Size,
    [property: JsonPropertyName("expires_at")] DateTimeOffset? ExpiresAt,
    [property: JsonPropertyName("size_vram")] long SizeVram,
    [property: JsonPropertyName("context_length")] int ContextLength)
{
    /// <summary>The model under its new name.</summary>
    public static implicit operator ChatApiRunningModel(OllamaRunningModel m) => new(m.Name, m.Model, m.Size, m.ExpiresAt, m.SizeVram, m.ContextLength);

    /// <summary>The model under its former name.</summary>
    public static implicit operator OllamaRunningModel(ChatApiRunningModel m) => new(m.Name, m.Model, m.Size, m.ExpiresAt, m.SizeVram, m.ContextLength);
}

/// <summary>Former name of <see cref="ChatApiTranslation"/>.</summary>
[Obsolete("Use ChatApiTranslation; this name is removed in the next release.")]
public static class OllamaTranslation
{
    /// <summary>See <see cref="ChatApiTranslation.Translate"/>.</summary>
    public static (ChatRequest Request, TimeSpan? KeepAlive, bool KeepAliveGiven) Translate(OllamaChatRequest request) => ChatApiTranslation.Translate(request);

    /// <summary>See <see cref="ChatApiTranslation.Options"/>.</summary>
    public static GenerationOptions Options(Dictionary<string, JsonElement>? values) => ChatApiTranslation.Options(values);

    /// <summary>See <see cref="ChatApiTranslation.KeepAliveOf"/>.</summary>
    public static (TimeSpan? Value, bool Given) KeepAliveOf(JsonElement? value) => ChatApiTranslation.KeepAliveOf(value);
}

#pragma warning restore CS0618
