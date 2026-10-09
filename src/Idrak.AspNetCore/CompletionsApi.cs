// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Idrak.Inference;

namespace Idrak.AspNetCore;

/// <summary>
/// Settings of <see cref="CompletionsApiEndpoints.MapCompletionsApi"/>: chat models served besides the engine's, and
/// embedding models for <c>/embeddings</c>.
/// </summary>
public sealed class CompletionsApiOptions
{
    internal Dictionary<string, IChatModel> ChatModels { get; } = new(StringComparer.Ordinal);
    internal Dictionary<string, IEmbedder> Embedders { get; } = new(StringComparer.Ordinal);
    internal ImageInputOptions ImageSettings { get; } = new();

    /// <summary>
    /// How <c>/chat/completions</c> and <c>/chat/upload</c> take images (<see cref="ImageInputOptions"/>: the most per
    /// request, and whether http(s) image URLs are downloaded; data URLs only unless allowed).
    /// </summary>
    public CompletionsApiOptions Images(Action<ImageInputOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        configure(ImageSettings);
        return this;
    }

    /// <summary>Serves <paramref name="model"/> as <paramref name="name"/> on <c>/chat/completions</c> (in addition to the engine's chat models).</summary>
    public CompletionsApiOptions ChatModel(string name, IChatModel model)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ChatModels[name] = model ?? throw new ArgumentNullException(nameof(model));
        return this;
    }

    /// <summary>Serves <paramref name="embedder"/> as <paramref name="name"/> on <c>/embeddings</c>.</summary>
    public CompletionsApiOptions Embeddings(string name, IEmbedder embedder)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        Embedders[name] = embedder ?? throw new ArgumentNullException(nameof(embedder));
        return this;
    }
}

/// <summary>
/// The OpenAI-style API (named by its wire format): <c>GET /models</c>, <c>POST /chat/completions</c> (streaming as
/// server-sent events, tool calls, images as <c>image_url</c> content parts), <c>POST /chat/upload</c> (the same answer
/// for a <c>multipart/form-data</c> form with image files and a prompt), <c>POST /completions</c> (raw text) and
/// <c>POST /embeddings</c>, over the text and chat models of the <see cref="InferenceEngine"/> registered by
/// <c>AddIdrak()</c> and the models given in <see cref="CompletionsApiOptions"/>. The request's <c>model</c> selects the
/// model by name; when only one chat model is served, any name selects it. Errors are
/// <c>{"error": {"message", "type", "code"}}</c> with 400 (bad input, or a part the model does not take: an image for a
/// text-only model), 404 (unknown model), 413 (a request larger than the host allows), 415 (an upload that is not a
/// form), 503 (queue full) or 504 (timeout).
/// </summary>
public static class CompletionsApiEndpoints
{
    /// <summary>Maps the endpoints under <paramref name="route"/> (usually "/v1").</summary>
    public static RouteGroupBuilder MapCompletionsApi(this IEndpointRouteBuilder app, string route = "/v1", Action<CompletionsApiOptions>? configure = null)
    {
        var settings = new CompletionsApiOptions();
        configure?.Invoke(settings);
        var group = app.MapGroup(route);
        group.MapGet("/models", (HttpContext http) => Models(http, settings)).WithName("CompletionsModels");
        group.MapPost("/chat/completions", (HttpContext http, CancellationToken token) => ChatCompletions(http, settings, token)).WithName("ChatCompletions");
        group.MapPost("/chat/upload", (HttpContext http, CancellationToken token) => ChatUpload(http, settings, token)).WithName("ChatUpload");
        group.MapPost("/completions", (HttpContext http, CancellationToken token) => Completions(http, settings, token)).WithName("Completions");
        group.MapPost("/embeddings", (HttpContext http, CancellationToken token) => Embeddings(http, settings, token)).WithName("Embeddings");
        return group;
    }

    // ------------------------------------------------------------------ /models

    private static IResult Models(HttpContext http, CompletionsApiOptions settings)
    {
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var names = (Engine(http)?.Names ?? []).Concat(settings.ChatModels.Keys).Concat(settings.Embedders.Keys).Distinct(StringComparer.Ordinal);
        var data = new JsonArray();
        foreach (string name in names)
        {
            data.Add(new JsonObject { ["id"] = name, ["object"] = "model", ["created"] = created, ["owned_by"] = "idrak" });
        }

        return JsonResult(new JsonObject { ["object"] = "list", ["data"] = data });
    }

    // ------------------------------------------------------------------ /chat/completions

    private static async Task<IResult> ChatCompletions(HttpContext http, CompletionsApiOptions settings, CancellationToken token)
    {
        JsonObject body;
        ChatRequest request;
        try
        {
            body = await ReadBody(http, token);
            await ImageRequests.ResolveUrls(body, settings.ImageSettings, token);
            request = CompletionsTranslation.Chat(body);
            request = ImageRequests.Check(request, settings.ImageSettings, (bool?)body["grayscale"] == true);
        }
        catch (Exception ex) when (ex is ArgumentException or JsonException or InvalidOperationException or FormatException)
        {
            return Error(400, ex.Message);
        }
        catch (BadHttpRequestException ex)
        {
            return Error(ex.StatusCode, ex.Message, ex.StatusCode == 413 ? "request_too_large" : null);
        }

        return await Answer(http, settings, (string?)body["model"], request, (bool?)body["stream"] == true,
            (bool?)body["stream_options"]?["include_usage"] == true, token);
    }

    // POST /chat/upload: a multipart/form-data form (image files, prompt, system, stream, max_tokens, temperature,
    // grayscale, model, and the other options of /chat/completions by the same names) answered as /chat/completions.
    private static async Task<IResult> ChatUpload(HttpContext http, CompletionsApiOptions settings, CancellationToken token)
    {
        if (!http.Request.HasFormContentType)
        {
            return Error(415, "POST a multipart/form-data form: image (one or more files), prompt (text), and optionally system, stream, "
                + "max_tokens, temperature, grayscale and model. JSON requests go to /chat/completions.", "unsupported_media_type");
        }

        IFormCollection form;
        ChatRequest request;
        bool stream, usage;
        try
        {
            form = await http.Request.ReadFormAsync(token);
            (request, stream, usage) = await CompletionsTranslation.Upload(form, token);
            request = ImageRequests.Check(request, settings.ImageSettings, Flag(form, "grayscale"));
        }
        catch (Exception ex) when (ex is ArgumentException or InvalidOperationException or FormatException)
        {
            return Error(400, ex.Message);
        }
        catch (BadHttpRequestException ex)
        {
            return Error(ex.StatusCode, ex.Message, ex.StatusCode == 413 ? "request_too_large" : null);
        }
        catch (InvalidDataException ex)
        {
            // The form reader's limits (FormOptions) are this exception too; anything else is a malformed form.
            return ex.Message.Contains("limit", StringComparison.OrdinalIgnoreCase)
                ? Error(413, $"The form is larger than this server takes: {ex.Message}", "request_too_large")
                : Error(400, $"The form is not valid multipart/form-data: {ex.Message}");
        }
        catch (IOException ex)
        {
            return Error(400, $"The form is not valid multipart/form-data: {ex.Message}");
        }

        return await Answer(http, settings, form["model"].FirstOrDefault(), request, stream, usage, token);
    }

    // A form field as true or false (absent or empty: false).
    internal static bool Flag(IFormCollection form, string key) => form[key].FirstOrDefault()?.Trim().ToLowerInvariant() switch
    {
        null or "" or "false" or "0" or "no" or "off" => false,
        "true" or "1" or "yes" or "on" => true,
        var other => throw new ArgumentException($"{key}: true or false, not '{other}'."),
    };

    // A chat request answered by the model named (or the only one): one chat.completion object, or chat.completion.chunk
    // events then [DONE] when streamed.
    private static async Task<IResult> Answer(HttpContext http, CompletionsApiOptions settings, string? asked, ChatRequest request, bool stream, bool usage,
        CancellationToken token)
    {
        var engine = Engine(http);
        string? name = Pick(asked, settings.ChatModels.Keys.Concat(ChatNames(engine)));
        if (name is null)
        {
            return Error(404, $"The model '{asked}' is not served; GET /models lists the models.", "model_not_found");
        }

        IChatModel model = settings.ChatModels.TryGetValue(name, out var extra) ? extra : engine!.Model<IChatModel>(name);
        try
        {
            ChatParts.ThrowIfUnsupported(model, request, name);         // an image for a text-only model: before anything loads
        }
        catch (NotSupportedException ex)
        {
            return Error(400, ex.Message, "unsupported_content");
        }

        string id = "chatcmpl-" + RandomId();
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var chunks = model.StreamAsync(request, token).GetAsyncEnumerator(token);
        var first = await Guard(async () => await chunks.MoveNextAsync() ? null : Error(500, "no output"));
        if (first is not null)
        {
            await chunks.DisposeAsync();
            return first;
        }

        if (!stream)
        {
            var last = chunks.Current;
            while (!last.Done && await chunks.MoveNextAsync())
            {
                last = chunks.Current;
            }

            await chunks.DisposeAsync();
            var message = last.Message ?? new ChatMessage("assistant", last.Delta.Content, last.Delta.Thinking, last.Delta.ToolCalls);
            bool calls = message.ToolCalls is { Count: > 0 };
            return JsonResult(new JsonObject
            {
                ["id"] = id,
                ["object"] = "chat.completion",
                ["created"] = created,
                ["model"] = name,
                ["choices"] = new JsonArray(new JsonObject
                {
                    ["index"] = 0,
                    ["message"] = CompletionsTranslation.Message(message),
                    ["finish_reason"] = calls ? "tool_calls" : last.DoneReason ?? "stop",
                }),
                ["usage"] = Usage(last.Stats),
            });
        }

        return Results.Stream(async output =>
        {
            await using var _ = chunks;
            bool sawCalls = false, opening = true;
            int index = 0;
            do
            {
                var chunk = chunks.Current;
                var delta = new JsonObject();
                if (opening)
                {
                    delta["role"] = "assistant";
                    opening = false;
                }

                if (chunk.Delta.Content.Length > 0)
                {
                    delta["content"] = chunk.Delta.Content;
                }

                if (chunk.Delta.Thinking.Length > 0)
                {
                    delta["reasoning_content"] = chunk.Delta.Thinking;
                }

                if (chunk.Delta.ToolCalls.Count > 0)
                {
                    var calls = new JsonArray();
                    foreach (var call in chunk.Delta.ToolCalls)
                    {
                        var json = CompletionsTranslation.Call(call);
                        json["index"] = index++;
                        calls.Add(json);
                    }

                    delta["tool_calls"] = calls;
                    sawCalls = true;
                }

                if (delta.Count > 0 || chunk.Done)
                {
                    string? reason = chunk.Done ? (sawCalls ? "tool_calls" : chunk.DoneReason ?? "stop") : null;
                    if (delta.Count > 0 && chunk.Done)
                    {
                        // The last piece's text first, then the finish reason on its own (as the wire format's clients expect).
                        await Event(output, ChatChunkJson(id, created, name, delta, null), token);
                        delta = [];
                    }

                    await Event(output, ChatChunkJson(id, created, name, delta, reason), token);
                }

                if (chunk.Done && usage)
                {
                    var json = ChatChunkJson(id, created, name, null, null);
                    json["usage"] = Usage(chunk.Stats);
                    await Event(output, json, token);
                }
            }
            while (await chunks.MoveNextAsync());

            await output.WriteAsync("data: [DONE]\n\n"u8.ToArray(), token);
            await output.FlushAsync(token);
        }, "text/event-stream");
    }

    private static JsonObject ChatChunkJson(string id, long created, string model, JsonObject? delta, string? reason) => new()
    {
        ["id"] = id,
        ["object"] = "chat.completion.chunk",
        ["created"] = created,
        ["model"] = model,
        ["choices"] = delta is null ? new JsonArray() : new JsonArray(new JsonObject { ["index"] = 0, ["delta"] = delta, ["finish_reason"] = reason }),
    };

    // ------------------------------------------------------------------ /completions

    private static async Task<IResult> Completions(HttpContext http, CompletionsApiOptions settings, CancellationToken token)
    {
        JsonObject body;
        string prompt;
        GenerationOptions options;
        try
        {
            body = await ReadBody(http, token);
            prompt = body["prompt"] switch
            {
                JsonValue v when v.TryGetValue(out string? s) => s,
                JsonArray { Count: 1 } a when a[0] is JsonValue v && v.TryGetValue(out string? s) => s,
                JsonArray => throw new ArgumentException("prompt: one prompt per request (a string or an array of one string)."),
                _ => throw new ArgumentException("prompt is required."),
            };
            options = CompletionsTranslation.Options(body);
        }
        catch (Exception ex) when (ex is ArgumentException or JsonException or InvalidOperationException or FormatException)
        {
            return Error(400, ex.Message);
        }

        var engine = Engine(http);
        string? asked = (string?)body["model"];
        string? name = Pick(asked, engine?.Names.Where(n => engine.TryGetModel<ITextModel>(n, out _)) ?? []);
        if (name is null)
        {
            return Error(404, settings.ChatModels.ContainsKey(asked ?? "")
                ? $"'{asked}' answers chat requests only; use /chat/completions."
                : $"The model '{asked}' is not served; GET /models lists the models.", "model_not_found");
        }

        bool stream = (bool?)body["stream"] == true;
        bool usage = (bool?)body["stream_options"]?["include_usage"] == true;
        string id = "cmpl-" + RandomId();
        long created = DateTimeOffset.UtcNow.ToUnixTimeSeconds();
        var chunks = engine!.Model<ITextModel>(name).StreamAsync(prompt, options, token).GetAsyncEnumerator(token);
        var first = await Guard(async () => await chunks.MoveNextAsync() ? null : Error(500, "no output"));
        if (first is not null)
        {
            await chunks.DisposeAsync();
            return first;
        }

        if (!stream)
        {
            var text = new StringBuilder();
            var last = chunks.Current;
            text.Append(last.Text);
            while (!last.Done && await chunks.MoveNextAsync())
            {
                last = chunks.Current;
                text.Append(last.Text);
            }

            await chunks.DisposeAsync();
            return JsonResult(new JsonObject
            {
                ["id"] = id,
                ["object"] = "text_completion",
                ["created"] = created,
                ["model"] = name,
                ["choices"] = new JsonArray(new JsonObject { ["text"] = text.ToString(), ["index"] = 0, ["logprobs"] = null, ["finish_reason"] = last.DoneReason ?? "stop" }),
                ["usage"] = Usage(last.Stats),
            });
        }

        return Results.Stream(async output =>
        {
            await using var _ = chunks;
            do
            {
                var chunk = chunks.Current;
                if (chunk.Text.Length > 0 || chunk.Done)
                {
                    await Event(output, new JsonObject
                    {
                        ["id"] = id,
                        ["object"] = "text_completion",
                        ["created"] = created,
                        ["model"] = name,
                        ["choices"] = new JsonArray(new JsonObject
                        {
                            ["text"] = chunk.Text, ["index"] = 0, ["logprobs"] = null, ["finish_reason"] = chunk.Done ? chunk.DoneReason ?? "stop" : null,
                        }),
                    }, token);
                }

                if (chunk.Done && usage)
                {
                    await Event(output, new JsonObject
                    {
                        ["id"] = id, ["object"] = "text_completion", ["created"] = created, ["model"] = name, ["choices"] = new JsonArray(), ["usage"] = Usage(chunk.Stats),
                    }, token);
                }
            }
            while (await chunks.MoveNextAsync());

            await output.WriteAsync("data: [DONE]\n\n"u8.ToArray(), token);
            await output.FlushAsync(token);
        }, "text/event-stream");
    }

    // ------------------------------------------------------------------ /embeddings

    private static async Task<IResult> Embeddings(HttpContext http, CompletionsApiOptions settings, CancellationToken token)
    {
        JsonObject body;
        List<string> inputs;
        try
        {
            body = await ReadBody(http, token);
            inputs = body["input"] switch
            {
                JsonValue v when v.TryGetValue(out string? s) => [s],
                JsonArray a when a.All(n => n is JsonValue v && v.TryGetValue(out string? _)) && a.Count > 0 => [.. a.Select(n => (string)n!)],
                _ => throw new ArgumentException("input is required: a string or an array of strings."),
            };
        }
        catch (Exception ex) when (ex is ArgumentException or JsonException or InvalidOperationException or FormatException)
        {
            return Error(400, ex.Message);
        }

        string? asked = (string?)body["model"];
        if (settings.Embedders.Count == 0)
        {
            return Error(404, "No embedding model is served.", "model_not_found");
        }

        string? name = Pick(asked, settings.Embedders.Keys);
        if (name is null)
        {
            return Error(404, $"The embedding model '{asked}' is not served; GET /models lists the models.", "model_not_found");
        }

        float[][] vectors;
        try
        {
            vectors = await settings.Embedders[name].EmbedAsync(inputs, token);
        }
        catch (ArgumentException ex)
        {
            return Error(400, ex.Message);
        }

        var data = new JsonArray();
        for (int i = 0; i < vectors.Length; i++)
        {
            data.Add(new JsonObject { ["object"] = "embedding", ["index"] = i, ["embedding"] = new JsonArray([.. vectors[i].Select(v => (JsonNode)v)]) });
        }

        int tokens = inputs.Sum(s => s.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries).Length);     // words: the embedder reports no token counts
        return JsonResult(new JsonObject
        {
            ["object"] = "list",
            ["data"] = data,
            ["model"] = name,
            ["usage"] = new JsonObject { ["prompt_tokens"] = tokens, ["total_tokens"] = tokens },
        });
    }

    // ------------------------------------------------------------------ helpers

    private static InferenceEngine? Engine(HttpContext http) => http.RequestServices.GetService<InferenceEngine>();

    private static IEnumerable<string> ChatNames(InferenceEngine? engine) => engine?.Names.Where(n => engine.TryGetModel<IChatModel>(n, out _)) ?? [];

    // The served name for a request: an exact match, the name without a ":latest" tag, or the only model served.
    private static string? Pick(string? asked, IEnumerable<string> names)
    {
        var all = names.Distinct(StringComparer.Ordinal).ToList();
        if (asked is not null)
        {
            if (all.Contains(asked))
            {
                return asked;
            }

            if (asked.EndsWith(":latest", StringComparison.Ordinal) && all.Contains(asked[..^":latest".Length]))
            {
                return asked[..^":latest".Length];
            }
        }

        return all.Count == 1 ? all[0] : null;
    }

    private static async Task<JsonObject> ReadBody(HttpContext http, CancellationToken token)
    {
        var node = await JsonNode.ParseAsync(http.Request.Body, cancellationToken: token);
        return node as JsonObject ?? throw new ArgumentException("The request body must be a JSON object.");
    }

    private static JsonObject Usage(GenerationStats? stats) => new()
    {
        ["prompt_tokens"] = stats?.PromptTokens ?? 0,
        ["completion_tokens"] = stats?.GeneratedTokens ?? 0,
        ["total_tokens"] = (stats?.PromptTokens ?? 0) + (stats?.GeneratedTokens ?? 0),
    };

    private static async Task Event(Stream output, JsonObject json, CancellationToken token)
    {
        await output.WriteAsync(Encoding.UTF8.GetBytes("data: " + json.ToJsonString() + "\n\n"), token);
        await output.FlushAsync(token);
    }

    private static async Task<IResult?> Guard(Func<Task<IResult?>> action)
    {
        try
        {
            return await action();
        }
        catch (InferenceQueueFullException ex)
        {
            return Error(503, ex.Message, "queue_full");
        }
        catch (TimeoutException ex)
        {
            return Error(504, ex.Message, "timeout");
        }
        catch (KeyNotFoundException ex)
        {
            return Error(404, ex.Message, "model_not_found");
        }
        catch (FileNotFoundException ex)
        {
            return Error(404, ex.Message, "model_not_found");
        }
        catch (ArgumentException ex)
        {
            return Error(400, ex.Message);
        }
        catch (NotSupportedException ex)
        {
            return Error(400, ex.Message, "unsupported_content");
        }
        catch (InvalidDataException ex)
        {
            return Error(400, $"An image does not decode: {ex.Message}", "invalid_image");
        }
    }

    private static IResult JsonResult(JsonNode node) => Results.Text(node.ToJsonString(), "application/json", Encoding.UTF8);

    private static IResult Error(int status, string message, string? code = null) => Results.Text(new JsonObject
    {
        ["error"] = new JsonObject
        {
            ["message"] = message,
            ["type"] = status >= 500 ? "server_error" : status == 404 ? "not_found_error" : status == 413 ? "request_too_large" : "invalid_request_error",
            ["code"] = code,
        },
    }.ToJsonString(), "application/json", Encoding.UTF8, status);

    internal static string RandomId() => Convert.ToHexStringLower(RandomNumberGenerator.GetBytes(12));
}

/// <summary>Converts OpenAI-style requests and messages to and from the library's chat types.</summary>
public static class CompletionsTranslation
{
    /// <summary>
    /// The chat request in an OpenAI-style body: messages (content as a string or text parts; assistant tool calls with
    /// arguments as a JSON string; tool results tied to their call by <c>tool_call_id</c>), tools, the reasoning effort
    /// and the generation options (<see cref="Options"/>). Throws <see cref="ArgumentException"/> on bad input.
    /// </summary>
    public static ChatRequest Chat(JsonObject body)
    {
        if (body["messages"] is not JsonArray messages || messages.Count == 0)
        {
            throw new ArgumentException("messages is required: a non-empty array.");
        }

        var callNames = new Dictionary<string, string>(StringComparer.Ordinal);
        var list = new List<ChatMessage>();
        foreach (var node in messages)
        {
            if (node is not JsonObject m || (string?)m["role"] is not { } role)
            {
                throw new ArgumentException("Each message needs a role.");
            }

            role = role == "developer" ? "system" : role;
            var content = Parts(m["content"]);
            List<ToolCall>? calls = null;
            if (m["tool_calls"] is JsonArray toolCalls && toolCalls.Count > 0)
            {
                calls = [];
                foreach (var c in toolCalls)
                {
                    var function = c?["function"] ?? throw new ArgumentException("A tool call needs a function.");
                    string name = (string?)function["name"] ?? throw new ArgumentException("A tool call needs a function name.");
                    calls.Add(new ToolCall(name, Arguments(function["arguments"])));
                    if ((string?)c!["id"] is { } id)
                    {
                        callNames[id] = name;
                    }
                }
            }

            string? toolName = null;
            if (role == "tool")
            {
                toolName = (string?)m["name"] ?? ((string?)m["tool_call_id"] is { } id && callNames.TryGetValue(id, out var called) ? called : null);
            }

            list.Add(new ChatMessage(role, content, (string?)m["reasoning_content"], calls, toolName));
        }

        var tools = new List<ToolDefinition>();
        if ((string?)body["tool_choice"] != "none" && body["tools"] is JsonArray toolList)
        {
            foreach (var t in toolList)
            {
                if ((string?)t?["type"] == "function" && t["function"] is JsonObject f)
                {
                    tools.Add(new ToolDefinition((string?)f["name"] ?? throw new ArgumentException("A tool needs a function name."),
                        (string?)f["description"], f["parameters"]?.DeepClone()));
                }
            }
        }

        bool? think = (string?)body["reasoning_effort"] switch
        {
            null => null,
            "none" => false,
            _ => true,
        };
        return new ChatRequest(list, tools, think, Options(body));
    }

    /// <summary>
    /// The generation options of an OpenAI-style body: temperature, top_p, max_tokens (or max_completion_tokens), seed,
    /// stop (a string or an array), presence_penalty, frequency_penalty, and the common extras top_k, min_p and
    /// repeat_penalty; anything else is ignored. The library's defaults apply to what is not given, except
    /// repeat_penalty: 1 (off) unless given, since the wire format has no repetition penalty (so an answer is the one
    /// <c>idrak run</c> gives, and an OCR model may repeat characters a page repeats).
    /// </summary>
    public static GenerationOptions Options(JsonObject body)
    {
        var options = new GenerationOptions { RepeatPenalty = 1f };
        if ((int?)body["n"] is > 1)
        {
            throw new ArgumentException("n: one choice per request.");
        }

        foreach (var (key, value) in body)
        {
            if (value is null)
            {
                continue;
            }

            try
            {
                options = key switch
                {
                    "temperature" => options with { Temperature = (float)value },
                    "top_p" => options with { TopP = (float)value },
                    "top_k" => options with { TopK = (int)value },
                    "min_p" => options with { MinP = (float)value },
                    "repeat_penalty" => options with { RepeatPenalty = (float)value },
                    "presence_penalty" => options with { PresencePenalty = (float)value },
                    "frequency_penalty" => options with { FrequencyPenalty = (float)value },
                    "seed" => options with { Seed = (int)value },
                    "max_tokens" or "max_completion_tokens" => options with { NumPredict = (int)value },
                    "stop" => options with { Stop = value is JsonArray a ? [.. a.Select(s => (string)s!)] : [(string)value!] },
                    _ => options,
                };
            }
            catch (Exception ex) when (ex is InvalidOperationException or FormatException)
            {
                throw new ArgumentException($"{key}: {ex.Message}");
            }
        }

        return options;
    }

    /// <summary>An assistant message as OpenAI-style JSON (content, reasoning_content, tool_calls with arguments as a JSON string).</summary>
    public static JsonObject Message(ChatMessage message)
    {
        var json = new JsonObject { ["role"] = message.Role, ["content"] = message.Content };
        if (!string.IsNullOrEmpty(message.Thinking))
        {
            json["reasoning_content"] = message.Thinking;
        }

        if (message.ToolCalls is { Count: > 0 } calls)
        {
            json["tool_calls"] = new JsonArray([.. calls.Select(c => (JsonNode)Call(c))]);
        }

        return json;
    }

    /// <summary>A tool call as OpenAI-style JSON: <c>{"id", "type": "function", "function": {"name", "arguments": "{...}"}}</c>.</summary>
    public static JsonObject Call(ToolCall call) => new()
    {
        ["id"] = "call_" + CompletionsApiEndpoints.RandomId(),
        ["type"] = "function",
        ["function"] = new JsonObject { ["name"] = call.Name, ["arguments"] = call.Arguments.ToJsonString() },
    };

    // Content as a string, or an array of parts: {"type": "text", "text"} and {"type": "image_url", "image_url": {"url"}}
    // with a data URL (http(s) URLs are downloaded by the endpoint first, when its options allow). Other parts are
    // refused, never dropped.
    private static List<ChatPart> Parts(JsonNode? content)
    {
        switch (content)
        {
            case null:
                return [];
            case JsonValue v when v.TryGetValue(out string? s):
                return s.Length == 0 ? [] : [new ChatText(s)];
            case JsonArray parts:
                var list = new List<ChatPart>(parts.Count);
                foreach (var p in parts)
                {
                    switch ((string?)p?["type"])
                    {
                        case "text":
                            list.Add(new ChatText((string?)p!["text"] ?? throw new ArgumentException("A text part needs \"text\".")));
                            break;
                        case "image_url":
                            list.Add(Image(p!.AsObject()));
                            break;
                        case var type:
                            throw new ArgumentException($"Content parts of type '{type}' are not supported; send text and image_url parts.");
                    }
                }

                return list;
            default:
                throw new ArgumentException("A message's content must be a string or an array of parts.");
        }
    }

    // An image_url part with a data URL (a 400 for anything else: an address the endpoint did not download, bad base64).
    private static ChatImage Image(JsonObject part)
    {
        string url = ImageUrl(part) ?? throw new ArgumentException("An image_url part needs \"image_url\": {\"url\": \"data:image/png;base64,...\"}.");
        if (!url.StartsWith("data:", StringComparison.OrdinalIgnoreCase))
        {
            throw new ArgumentException("image_url: give the image as a data URL (data:image/png;base64,...)"
                + (url.StartsWith("http", StringComparison.OrdinalIgnoreCase) ? "; this server does not download http(s) URLs." : "."));
        }

        try
        {
            return ChatImage.FromDataUrl(url);
        }
        catch (FormatException ex)
        {
            throw new ArgumentException($"image_url: {ex.Message}");
        }
    }

    /// <summary>The address of an <c>image_url</c> part: <c>{"image_url": {"url": "..."}}</c>, or leniently <c>{"image_url": "..."}</c>.</summary>
    internal static string? ImageUrl(JsonObject part) => part["image_url"] switch
    {
        JsonObject o => (string?)o["url"],
        JsonValue v when v.TryGetValue(out string? url) => url,
        _ => null,
    };

    /// <summary>
    /// The chat request of a <c>multipart/form-data</c> form (<c>/chat/upload</c>): the files of <c>image</c> (or
    /// <c>images</c>), then the text of <c>prompt</c>, as one user message after an optional <c>system</c> message; the
    /// generation options by the names of <see cref="Options"/> (<c>max_tokens</c>, <c>temperature</c>, <c>top_p</c>,
    /// <c>top_k</c>, <c>min_p</c>, <c>seed</c>, <c>stop</c>, ...); <c>stream</c> and <c>include_usage</c> as true or
    /// false. Throws <see cref="ArgumentException"/> on bad input.
    /// </summary>
    public static async Task<(ChatRequest Request, bool Stream, bool Usage)> Upload(IFormCollection form, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(form);
        if (form.Files.FirstOrDefault(f => !IsImageField(f.Name)) is { } other)
        {
            throw new ArgumentException($"The file field '{other.Name}' is not one this endpoint reads; send image files as 'image'.");
        }

        var parts = new List<ChatPart>();
        foreach (var file in form.Files)
        {
            if (file.Length == 0)
            {
                throw new ArgumentException($"image: the file '{file.FileName}' is empty.");
            }

            using var data = new MemoryStream((int)Math.Min(file.Length, int.MaxValue));
            await file.CopyToAsync(data, cancellationToken);
            string? type = file.ContentType is { } t && t.StartsWith("image/", StringComparison.OrdinalIgnoreCase) ? t : null;
            parts.Add(ChatImage.FromBytes(data.ToArray(), type));
        }

        string prompt = form["prompt"].FirstOrDefault() ?? "";
        if (prompt.Length == 0 && parts.Count == 0)
        {
            throw new ArgumentException("prompt is required (and image: one or more files).");
        }

        if (prompt.Length > 0)
        {
            parts.Add(new ChatText(prompt));
        }

        var messages = new List<ChatMessage>();
        if (form["system"].FirstOrDefault() is { Length: > 0 } system)
        {
            messages.Add(new ChatMessage("system", system));
        }

        messages.Add(new ChatMessage("user", parts));
        var options = new JsonObject();
        foreach (string key in UploadNumbers)
        {
            if (form[key].FirstOrDefault() is { Length: > 0 } text)
            {
                if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double number))
                {
                    throw new ArgumentException($"{key}: a number, not '{text}'.");
                }

                if (key is "top_k" or "seed" or "max_tokens" or "max_completion_tokens" && number % 1 != 0)
                {
                    throw new ArgumentException($"{key}: a whole number, not '{text}'.");
                }

                options[key] = JsonNode.Parse(number.ToString("R", CultureInfo.InvariantCulture));      // read as JSON numbers are
            }
        }

        if (form["stop"].Where(s => !string.IsNullOrEmpty(s)).ToList() is { Count: > 0 } stops)
        {
            options["stop"] = new JsonArray([.. stops.Select(s => (JsonNode)s!)]);
        }

        bool? think = form["reasoning_effort"].FirstOrDefault() switch
        {
            null or "" => null,
            "none" => false,
            _ => true,
        };
        return (new ChatRequest(messages, [], think, Options(options)),
            CompletionsApiEndpoints.Flag(form, "stream"), CompletionsApiEndpoints.Flag(form, "include_usage"));
    }

    private static bool IsImageField(string name) => name is "image" or "images" or "image[]";

    private static readonly string[] UploadNumbers =
        ["temperature", "top_p", "top_k", "min_p", "repeat_penalty", "presence_penalty", "frequency_penalty", "seed", "max_tokens", "max_completion_tokens"];

    // Arguments as a JSON string (the wire format) or, leniently, as an object.
    private static JsonObject Arguments(JsonNode? arguments) => arguments switch
    {
        null => [],
        JsonObject o => (JsonObject)o.DeepClone(),
        JsonValue v when v.TryGetValue(out string? s) => string.IsNullOrWhiteSpace(s) ? []
            : JsonNode.Parse(s) as JsonObject ?? throw new ArgumentException("A tool call's arguments must be a JSON object."),
        _ => throw new ArgumentException("A tool call's arguments must be a JSON object."),
    };
}
