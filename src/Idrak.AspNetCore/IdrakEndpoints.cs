// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Diagnostics;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Idrak.Inference;

namespace Idrak.AspNetCore;

/// <summary>Who runs the tools a chat model asks for.</summary>
public enum ToolExecution
{
    /// <summary>The model's tool calls are returned in <c>message.tool_calls</c> and the client runs them (the API's usual behaviour).</summary>
    Client,

    /// <summary>
    /// The server runs the tools (those given to <see cref="ChatApiOptions.Tools"/>, else the chat model's own,
    /// <see cref="IToolChatModel.Tools"/>) and returns the final answer.
    /// </summary>
    Server,
}

/// <summary>Settings of <see cref="IdrakEndpointExtensions.MapChatApi"/>. <see cref="Tools"/> must be called.</summary>
public sealed class ChatApiOptions
{
    internal ToolExecution? Execution { get; private set; }
    internal int? Rounds { get; private set; }
    internal IToolRegistry? ServerTools { get; private set; }
    internal string? Served { get; private set; }
    internal ImageInputOptions ImageSettings { get; } = new();
    internal string VersionText { get; private set; } =
        typeof(InferenceEngine).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "0.0.0";

    /// <summary>
    /// Who runs tool calls. With <see cref="ToolExecution.Server"/>, <paramref name="maxRounds"/> (required) bounds the
    /// call→result rounds per request, and <paramref name="tools"/> are the tools to run (the chat model's own,
    /// <see cref="IToolChatModel.Tools"/>, when not given).
    /// </summary>
    public ChatApiOptions Tools(ToolExecution execution, int? maxRounds = null, IToolRegistry? tools = null)
    {
        if (execution == ToolExecution.Server && maxRounds is null)
        {
            throw new ArgumentException("Server-side tool execution needs maxRounds.", nameof(maxRounds));
        }

        if (execution == ToolExecution.Client && tools is not null)
        {
            throw new ArgumentException("Tools run on the server only; the client runs its own.", nameof(tools));
        }

        Execution = execution;
        Rounds = maxRounds;
        ServerTools = tools;
        return this;
    }

    /// <summary>How /chat takes images (<see cref="ImageInputOptions.MaxImages"/>; the chat API has no image URLs).</summary>
    public ChatApiOptions Images(Action<ImageInputOptions> configure)
    {
        ArgumentNullException.ThrowIfNull(configure);
        configure(ImageSettings);
        return this;
    }

    /// <summary>The name reported by /tags and /ps (the engine's model name unless set).</summary>
    public ChatApiOptions ModelName(string name)
    {
        Served = name;
        return this;
    }

    /// <summary>The version reported by /version (the Idrak assembly version unless set).</summary>
    public ChatApiOptions Version(string version)
    {
        VersionText = version;
        return this;
    }
}

/// <summary>A request to <see cref="IdrakEndpointExtensions.MapGenerate"/>: the prompt, the chat API's options (temperature, top_k, num_predict, ...) and whether to stream (server-sent events).</summary>
public sealed record GenerationRequest(
    [property: JsonPropertyName("prompt")] string Prompt,
    [property: JsonPropertyName("options")] Dictionary<string, JsonElement>? Options = null,
    [property: JsonPropertyName("stream")] bool? Stream = null);

/// <summary>ASP.NET Core endpoints over the <see cref="InferenceEngine"/> registered by <c>AddIdrak()</c>.</summary>
public static class IdrakEndpointExtensions
{
    private static readonly JsonSerializerOptions Json = new(JsonSerializerDefaults.Web);

    /// <summary>
    /// POST <paramref name="route"/> with one <typeparamref name="TIn"/> as JSON returns the <typeparamref name="TOut"/>;
    /// POST <paramref name="route"/>/batch with an array returns an array. Errors: 400 bad input, 503 queue full, 504 timeout.
    /// </summary>
    public static RouteGroupBuilder MapPredictor<TIn, TOut>(this IEndpointRouteBuilder app, string route, string name)
    {
        var group = app.MapGroup(route);
        Delegate one = async (TIn input, InferenceEngine engine, CancellationToken token) =>
            await Respond(async () => Results.Ok(await engine.PredictAsync<TIn, TOut>(name, input, token)));
        Delegate many = async (TIn[] inputs, InferenceEngine engine, CancellationToken token) =>
            await Respond(async () => Results.Ok(await engine.PredictAsync<TIn, TOut>(name, inputs, token)));
        group.MapPost("", one).WithName($"Predict-{name}");
        group.MapPost("/batch", many).WithName($"PredictBatch-{name}");
        return group;
    }

    /// <summary>
    /// POST <paramref name="route"/> with {prompt, options, stream}: without streaming, one JSON object with the text,
    /// done_reason and timings; with <c>"stream": true</c>, server-sent events: <c>chunk</c> events with text, then one <c>done</c> event.
    /// </summary>
    public static RouteHandlerBuilder MapGenerate(this IEndpointRouteBuilder app, string route, string name) =>
        app.MapPost(route, async (GenerationRequest request, InferenceEngine engine, HttpContext http, CancellationToken token) =>
        {
            if (string.IsNullOrEmpty(request.Prompt))
            {
                return Error(400, "prompt is required.");
            }

            GenerationOptions options;
            try
            {
                options = ChatApiTranslation.Options(request.Options);
            }
            catch (ArgumentException ex)
            {
                return Error(400, ex.Message);
            }

            if (request.Stream != true)
            {
                return await Respond(async () =>
                {
                    var (text, reason, stats) = await engine.Model<ITextModel>(name).GenerateAsync(request.Prompt, options, token);
                    return Results.Ok(new { text, done_reason = reason, stats = Stats(stats) });
                });
            }

            var stream = engine.Model<ITextModel>(name).StreamAsync(request.Prompt, options, token).GetAsyncEnumerator(token);
            var first = await Guard(async () => await stream.MoveNextAsync() ? null : Error(500, "no output"));
            if (first is not null)
            {
                await stream.DisposeAsync();
                return first;
            }

            return Results.Stream(async body =>
            {
                await using var _ = stream;
                var (buffer, writer) = EventWriter();
                using var __ = writer;
                do
                {
                    // Each event is built as UTF-8 in one reused buffer: "event: …\ndata: " + the JSON + "\n\n".
                    var chunk = stream.Current;
                    buffer.Write(chunk.Done ? "event: done\ndata: "u8 : "event: chunk\ndata: "u8);
                    writer.Reset(buffer);
                    if (chunk.Done)
                    {
                        JsonSerializer.Serialize(writer, new { done_reason = chunk.DoneReason, stats = Stats(chunk.Stats!) }, Json);
                    }
                    else
                    {
                        JsonSerializer.Serialize(writer, new { text = chunk.Text }, Json);
                    }

                    writer.Flush();
                    buffer.Write("\n\n"u8);
                    await body.WriteAsync(buffer.WrittenMemory, token);
                    await body.FlushAsync(token);
                    buffer.ResetWrittenCount();
                }
                while (await stream.MoveNextAsync());
            }, "text/event-stream");
        }).WithName($"Generate-{name}");

    /// <summary>
    /// The chat API under <paramref name="route"/>, on the routes common local-model clients call: POST /chat (NDJSON
    /// streaming, think, tools, options, keep_alive; images as a message's <c>images</c> or as image parts of its
    /// content, see <see cref="ChatApiMessage"/>, <c>"grayscale": true</c> to read them grey, <c>"vision_options"</c> for the vision family, <c>"image_transforms"</c> run on the images first), GET /tags, GET /ps and GET /version, all serving the chat model
    /// <paramref name="name"/>. The request body is read as JSON whatever its Content-Type (clients often send none).
    /// <see cref="ChatApiOptions.Tools"/> must be set. The OpenAI-style <c>/v1</c> API is
    /// <see cref="CompletionsApiEndpoints.MapCompletionsApi"/>.
    /// </summary>
    public static RouteGroupBuilder MapChatApi(this IEndpointRouteBuilder app, string route, string name, Action<ChatApiOptions> configure)
    {
        var settings = new ChatApiOptions();
        configure(settings);
        if (settings.Execution is null)
        {
            throw new InvalidOperationException("MapChatApi needs options.Tools(ToolExecution.Client) or options.Tools(ToolExecution.Server, maxRounds).");
        }

        var group = app.MapGroup(route);
        group.MapPost("/chat", (HttpRequest http, InferenceEngine engine, CancellationToken token) => Chat(http, engine, name, settings, token))
            .WithName($"Chat-{name}");
        group.MapGet("/tags", async (InferenceEngine engine, CancellationToken token) =>
        {
            ModelDescription d;
            try
            {
                d = await engine.DescribeAsync(name, token);
            }
            catch (FileNotFoundException)
            {
                return Results.Ok(new { models = Array.Empty<ChatApiModelTag>() });     // nothing to serve yet
            }

            string served = settings.Served ?? name;
            return Results.Ok(new
            {
                models = new[]
                {
                    new ChatApiModelTag(served, served, DateTimeOffset.UtcNow, d.Parameters * sizeof(float),
                        new ChatApiModelDetails("idrak", d.Kind, FormatCount(d.Parameters), d.ContextLength ?? 0)),
                },
            });
        }).WithName($"ChatTags-{name}");
        group.MapGet("/ps", (InferenceEngine engine) =>
        {
            var status = engine.Models.First(m => m.Name == name);
            string served = settings.Served ?? name;
            if (!status.Loaded)
            {
                return Results.Ok(new { models = Array.Empty<ChatApiRunningModel>() });
            }

            var d = engine.DescribeAsync(name).GetAwaiter().GetResult();
            long size = d.Parameters * sizeof(float);
            return Results.Ok(new
            {
                models = new[] { new ChatApiRunningModel(served, served, size, status.ExpiresAt, d.Device.IsGpu ? size : 0, d.ContextLength ?? 0) },
            });
        }).WithName($"ChatPs-{name}");
        group.MapGet("/version", () => Results.Ok(new { version = settings.VersionText })).WithName($"ChatVersion-{name}");
        return group;
    }

    /// <summary>GET <paramref name="route"/>: every engine model with its state and statistics, and every device with its memory use.</summary>
    public static RouteHandlerBuilder MapIdrakStatus(this IEndpointRouteBuilder app, string route) =>
        app.MapGet(route, (InferenceEngine engine) =>
        {
            var devices = Device.Available;

            return Results.Ok(new
            {
                models = engine.Models.Select(m => new
                {
                    name = m.Name, kind = m.Kind, loaded = m.Loaded, instances = m.Instances, running = m.Running,
                    queued = m.Queued, expires_at = m.ExpiresAt, stats = engine.Stats(m.Name),
                }),
                devices = devices.Select(d =>
                {
                    var memory = ComputeResources.GetMemoryUsage(d);
                    return new { device = d.ToString(), name = d.Name, memory_in_use = memory.InUse, memory_cached = memory.Cached, memory_limit = memory.Limit };
                }),
            });
        }).WithName("IdrakStatus");

    // ------------------------------------------------------------------ /chat

    private static async Task<IResult> Chat(HttpRequest http, InferenceEngine engine, string name, ChatApiOptions settings, CancellationToken token)
    {
        ChatApiRequest? request;
        ChatRequest chat;
        try
        {
            request = await JsonSerializer.DeserializeAsync<ChatApiRequest>(http.Body, Json, token)
                ?? throw new ArgumentException("empty request body");
            (chat, var keepAlive, bool given) = ChatApiTranslation.Translate(request);
            chat = ImageRequests.WithVisionOptions(ImageRequests.Check(chat, settings.ImageSettings, request.Grayscale == true), request.VisionOptions);
            chat = ImageRequests.WithImageTransforms(chat, request.ImageTransforms);
            if (given)
            {
                engine.KeepAlive(name, keepAlive);
            }
        }
        catch (Exception ex) when (ex is ArgumentException or JsonException or InvalidOperationException)
        {
            return Error(400, ex.Message);
        }
        catch (BadHttpRequestException ex)
        {
            return Error(ex.StatusCode, ex.Message);
        }

        string served = request.Model ?? settings.Served ?? name;
        bool stream = request.Stream != false;
        var clock = Stopwatch.StartNew();
        var model = engine.Model<IChatModel>(name);
        if (settings.Execution == ToolExecution.Server)
        {
            var tools = settings.ServerTools ?? (model as IToolChatModel)?.Tools;
            if (tools is null)
            {
                return Error(500, $"Server-side tool execution is on, but the chat model '{name}' has no tools (give them to options.Tools or to the model).");
            }

            model = model.WithTools(tools, settings.Rounds!.Value);
        }

        try
        {
            ChatParts.ThrowIfUnsupported(model, chat, name);            // an image for a text-only model: before anything loads
        }
        catch (NotSupportedException ex)
        {
            return Error(400, ex.Message);
        }

        var lines = ChatLines(model.StreamAsync(chat, token), served, stream, clock).GetAsyncEnumerator(token);

        // Read the first line before answering, so a full queue, a timeout or a load failure gets a proper status code.
        var first = await Guard(async () => await lines.MoveNextAsync() ? null : Error(500, "no output"));
        if (first is not null)
        {
            await lines.DisposeAsync();
            return first;
        }

        if (!stream)
        {
            ChatApiResponse last = lines.Current;
            while (await lines.MoveNextAsync())
            {
                last = lines.Current;
            }

            await lines.DisposeAsync();
            return Results.Json(last, Json);
        }

        return Results.Stream(async body =>
        {
            await using var _ = lines;
            var (buffer, writer) = EventWriter();
            using var __ = writer;
            do
            {
                // The JSON line and its newline as UTF-8 in one reused buffer, written at once.
                writer.Reset(buffer);
                JsonSerializer.Serialize(writer, lines.Current, Json);
                writer.Flush();
                buffer.Write("\n"u8);
                await body.WriteAsync(buffer.WrittenMemory, token);
                await body.FlushAsync(token);
                buffer.ResetWrittenCount();
            }
            while (await lines.MoveNextAsync());
        }, "application/x-ndjson");
    }

    private static async IAsyncEnumerable<ChatApiResponse> ChatLines(IAsyncEnumerable<ChatChunk> chunks, string served, bool stream, Stopwatch clock)
    {
        int toolIndex = 0;
        await foreach (var chunk in chunks)
        {
            if (!chunk.Done)
            {
                if (stream)
                {
                    yield return Line(served, chunk.Delta.Content, chunk.Delta.Thinking, ChatApiTranslation.Calls(chunk.Delta.ToolCalls, ref toolIndex));
                }

                continue;
            }

            var message = stream
                ? new ChatApiMessage("assistant", chunk.Delta.Content, NullIfEmpty(chunk.Delta.Thinking), ChatApiTranslation.Calls(chunk.Delta.ToolCalls, ref toolIndex))
                : new ChatApiMessage("assistant", chunk.Message!.Content, chunk.Message.Thinking, ChatApiTranslation.Calls(chunk.Message.ToolCalls ?? [], ref toolIndex));
            yield return Final(served, message, chunk.DoneReason, chunk.Stats, clock);
        }
    }

    private static ChatApiResponse Line(string served, string content, string thinking, List<ChatApiToolCall>? calls) =>
        new(served, DateTimeOffset.UtcNow, new ChatApiMessage("assistant", content, NullIfEmpty(thinking), calls), false);

    private static ChatApiResponse Final(string served, ChatApiMessage message, string? reason, GenerationStats? s, Stopwatch clock) =>
        new(served, DateTimeOffset.UtcNow, message, true, reason, Nanoseconds(clock.Elapsed), 0,
            s?.PromptTokens, s is null ? null : Nanoseconds(s.PromptDuration), s?.GeneratedTokens, s is null ? null : Nanoseconds(s.GenerationDuration));

    // ------------------------------------------------------------------ helpers

    private static async Task<IResult?> Guard(Func<Task<IResult?>> action)
    {
        try
        {
            return await action();
        }
        catch (InferenceQueueFullException ex)
        {
            return Error(503, ex.Message);
        }
        catch (TimeoutException ex)
        {
            return Error(504, ex.Message);
        }
        catch (KeyNotFoundException ex)
        {
            return Error(404, ex.Message);
        }
        catch (FileNotFoundException ex)
        {
            return Error(404, ex.Message);
        }
        catch (ArgumentException ex)
        {
            return Error(400, ex.Message);
        }
        catch (NotSupportedException ex)
        {
            return Error(400, ex.Message);
        }
        catch (InvalidDataException ex)
        {
            return Error(400, $"An image does not decode: {ex.Message}");
        }
    }

    private static async Task<IResult> Respond(Func<Task<IResult>> action) => (await Guard(async () => (IResult?)await action()))!;

    // A buffer reused for each streamed event or line, and a JSON writer over it with the settings the serializer would
    // use for a string or a stream (same encoder and indentation, so the same bytes).
    private static (ArrayBufferWriter<byte> Buffer, Utf8JsonWriter Writer) EventWriter()
    {
        var buffer = new ArrayBufferWriter<byte>();
        return (buffer, new Utf8JsonWriter(buffer, new JsonWriterOptions { Encoder = Json.Encoder, Indented = Json.WriteIndented }));
    }

    private static IResult Error(int status, string message) => Results.Json(new { error = message }, Json, statusCode: status);

    private static object Stats(GenerationStats s) => new
    {
        prompt_tokens = s.PromptTokens,
        generated_tokens = s.GeneratedTokens,
        prompt_ms = s.PromptDuration.TotalMilliseconds,
        generation_ms = s.GenerationDuration.TotalMilliseconds,
        total_ms = s.TotalDuration.TotalMilliseconds,
        tokens_per_second = s.TokensPerSecond,
        context_resets = s.ContextResets,
    };

    private static string FormatCount(long n) => n >= 1_000_000_000 ? $"{n / 1e9:0.#}B" : n >= 1_000_000 ? $"{n / 1e6:0.#}M" : $"{n / 1e3:0}K";

    private static string? NullIfEmpty(string text) => text.Length == 0 ? null : text;

    private static long Nanoseconds(TimeSpan t) => t.Ticks * 100;
}
