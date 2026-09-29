using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.Json.Serialization;
using Idrak.Generation;

namespace Idrak.AspNetCore;

// ------------------------------------------------------------------ wire format (Ollama-compatible JSON)

/// <summary>An /api/chat request in the Ollama format. Unknown fields are ignored; any model name selects the served model.</summary>
public sealed record OllamaChatRequest(
    [property: JsonPropertyName("model")] string? Model,
    [property: JsonPropertyName("messages")] List<OllamaMessage> Messages,
    [property: JsonPropertyName("stream")] bool? Stream = null,
    [property: JsonPropertyName("think")] JsonElement? Think = null,
    [property: JsonPropertyName("keep_alive")] JsonElement? KeepAlive = null,
    [property: JsonPropertyName("options")] Dictionary<string, JsonElement>? Options = null,
    [property: JsonPropertyName("tools")] List<OllamaTool>? Tools = null);

/// <summary>A chat message: role (system, user, assistant, tool), content, and optionally reasoning, tool calls or the tool name.</summary>
public sealed record OllamaMessage(
    [property: JsonPropertyName("role")] string Role,
    [property: JsonPropertyName("content")] string? Content = null,
    [property: JsonPropertyName("thinking"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? Thinking = null,
    [property: JsonPropertyName("tool_calls"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] List<OllamaToolCall>? ToolCalls = null,
    [property: JsonPropertyName("tool_name"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] string? ToolName = null);

/// <summary>A tool definition: {"type": "function", "function": {name, description, parameters}}.</summary>
public sealed record OllamaTool(
    [property: JsonPropertyName("type")] string Type,
    [property: JsonPropertyName("function")] OllamaFunction Function);

/// <summary>A function's name, description and JSON-schema parameters.</summary>
public sealed record OllamaFunction(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("description")] string? Description = null,
    [property: JsonPropertyName("parameters")] JsonNode? Parameters = null);

/// <summary>A tool call returned by the model.</summary>
public sealed record OllamaToolCall([property: JsonPropertyName("function")] OllamaCalledFunction Function);

/// <summary>The called function and its arguments.</summary>
public sealed record OllamaCalledFunction(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("arguments")] JsonObject Arguments,
    [property: JsonPropertyName("index"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] int? Index = null);

/// <summary>One streamed line (or the whole non-streamed reply) of /api/chat. Durations are in nanoseconds.</summary>
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
    [property: JsonPropertyName("eval_duration"), JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)] long? EvalDuration = null);

/// <summary>A model entry for /api/tags.</summary>
public sealed record OllamaModelTag(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("modified_at")] DateTimeOffset ModifiedAt,
    [property: JsonPropertyName("size")] long Size,
    [property: JsonPropertyName("details")] OllamaModelDetails Details);

/// <summary>Model details.</summary>
public sealed record OllamaModelDetails(
    [property: JsonPropertyName("format")] string Format,
    [property: JsonPropertyName("family")] string Family,
    [property: JsonPropertyName("parameter_size")] string ParameterSize,
    [property: JsonPropertyName("context_length")] int ContextLength);

/// <summary>A loaded model for /api/ps.</summary>
public sealed record OllamaRunningModel(
    [property: JsonPropertyName("name")] string Name,
    [property: JsonPropertyName("model")] string Model,
    [property: JsonPropertyName("size")] long Size,
    [property: JsonPropertyName("expires_at")] DateTimeOffset? ExpiresAt,
    [property: JsonPropertyName("size_vram")] long SizeVram,
    [property: JsonPropertyName("context_length")] int ContextLength);

/// <summary>Converts Ollama requests to the library's chat types.</summary>
public static class OllamaTranslation
{
    /// <summary>
    /// Validates the request and turns it into a <see cref="ChatRequest"/>; throws <see cref="ArgumentException"/> on bad input.
    /// Known options map to <see cref="GenerationOptions"/>; other options are accepted and ignored, as Ollama does.
    /// </summary>
    public static (ChatRequest Request, TimeSpan? KeepAlive, bool KeepAliveGiven) Translate(OllamaChatRequest request)
    {
        if (request.Messages is null)
        {
            throw new ArgumentException("messages is required.");
        }

        var options = Options(request.Options);
        bool? think = request.Think?.ValueKind switch
        {
            JsonValueKind.True => true,
            JsonValueKind.False => false,
            JsonValueKind.String => request.Think.Value.GetString() is not ("" or "none" or "false"),   // "low" | "medium" | "high"
            _ => null,
        };

        var messages = request.Messages.Select(m => new ChatMessage(m.Role, m.Content ?? "", m.Thinking,
            m.ToolCalls?.Select(c => new ToolCall(c.Function.Name, c.Function.Arguments)).ToList(), m.ToolName)).ToList();
        var tools = (request.Tools ?? []).Where(t => t.Type == "function")
            .Select(t => new ToolDefinition(t.Function.Name, t.Function.Description, t.Function.Parameters)).ToList();
        var (keepAlive, given) = KeepAliveOf(request.KeepAlive);
        return (new ChatRequest(messages, tools, think, options), keepAlive, given);
    }

    /// <summary>Maps Ollama's option names to <see cref="GenerationOptions"/> (unknown names are ignored).</summary>
    public static GenerationOptions Options(Dictionary<string, JsonElement>? values)
    {
        var options = new GenerationOptions();
        foreach (var (key, value) in values ?? [])
        {
            try
            {
                options = key switch
                {
                    "temperature" => options with { Temperature = value.GetSingle() },
                    "top_k" => options with { TopK = value.GetInt32() },
                    "top_p" => options with { TopP = value.GetSingle() },
                    "min_p" => options with { MinP = value.GetSingle() },
                    "repeat_penalty" => options with { RepeatPenalty = value.GetSingle() },
                    "repeat_last_n" => options with { RepeatLastN = value.GetInt32() },
                    "presence_penalty" => options with { PresencePenalty = value.GetSingle() },
                    "frequency_penalty" => options with { FrequencyPenalty = value.GetSingle() },
                    "seed" => options with { Seed = value.GetInt32() },
                    "num_ctx" => options with { NumCtx = value.GetInt32() },
                    "num_predict" => options with { NumPredict = value.GetInt32() },
                    "stop" => options with { Stop = [.. value.EnumerateArray().Select(s => s.GetString()!)] },
                    _ => options,
                };
            }
            catch (Exception ex) when (ex is InvalidOperationException or FormatException)
            {
                throw new ArgumentException($"options.{key}: {ex.Message}");
            }
        }

        return options;
    }

    /// <summary>Parses keep_alive ("30m", 90, 0, -1); absent or null means "not given".</summary>
    public static (TimeSpan? Value, bool Given) KeepAliveOf(JsonElement? value)
    {
        if (value is null || value.Value.ValueKind is JsonValueKind.Null or JsonValueKind.Undefined)
        {
            return (null, false);
        }

        try
        {
            return (KeepAlive.Parse(value, null), true);
        }
        catch (FormatException ex)
        {
            throw new ArgumentException($"keep_alive: {ex.Message}");
        }
    }

    internal static List<OllamaToolCall>? Calls(IReadOnlyList<ToolCall> calls, ref int index)
    {
        if (calls.Count == 0)
        {
            return null;
        }

        var list = new List<OllamaToolCall>();
        foreach (var c in calls)
        {
            list.Add(new OllamaToolCall(new OllamaCalledFunction(c.Name, c.Arguments, index++)));
        }

        return list;
    }
}
