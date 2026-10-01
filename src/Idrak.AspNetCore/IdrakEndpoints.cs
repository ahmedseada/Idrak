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
using Idrak.Generation;
using Idrak.Inference;

namespace Idrak.AspNetCore;

/// <summary>Who runs the tools a chat model asks for.</summary>
public enum ToolExecution
{
    /// <summary>The model's tool calls are returned in <c>message.tool_calls</c> and the client runs them (Ollama's behaviour).</summary>
    Client,

    /// <summary>The server runs the tools registered with the chat model (<see cref="GenerativeModelBuilder.Tools"/>) and returns the final answer.</summary>
    Server,
}

/// <summary>Settings of <see cref="IdrakEndpointExtensions.MapOllamaApi"/>. <see cref="Tools"/> must be called.</summary>
public sealed class OllamaApiOptions
{
    internal ToolExecution? Execution { get; private set; }
    internal int? Rounds { get; private set; }
    internal string? Served { get; private set; }
    internal string VersionText { get; private set; } =
        typeof(InferenceEngine).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "0.0.0";

    /// <summary>Who runs tool calls; with <see cref="ToolExecution.Server"/>, <paramref name="maxRounds"/> (required) bounds the call→result rounds per request.</summary>
    public OllamaApiOptions Tools(ToolExecution execution, int? maxRounds = null)
    {
        if (execution == ToolExecution.Server && maxRounds is null)
        {
            throw new ArgumentException("Server-side tool execution needs maxRounds.", nameof(maxRounds));
        }

        Execution = execution;
        Rounds = maxRounds;
        return this;
    }

    /// <summary>The name reported by /tags and /ps (the engine's model name unless set).</summary>
    public OllamaApiOptions ModelName(string name)
    {
        Served = name;
        return this;
    }

    /// <summary>The version reported by /version (the Idrak assembly version unless set).</summary>
    public OllamaApiOptions Version(string version)
    {
        VersionText = version;
        return this;
    }
}

/// <summary>A request to <see cref="IdrakEndpointExtensions.MapGenerate"/>: the prompt, Ollama-style options and whether to stream (server-sent events).</summary>
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
                options = OllamaTranslation.Options(request.Options);
            }
            catch (ArgumentException ex)
            {
                return Error(400, ex.Message);
            }

            if (request.Stream != true)
            {
                return await Respond(async () =>
                {
                    var (text, reason, stats) = await engine.GenerateAsync(name, request.Prompt, options, token);
                    return Results.Ok(new { text, done_reason = reason, stats = Stats(stats) });
                });
            }

            var stream = engine.StreamAsync(name, request.Prompt, options, token).GetAsyncEnumerator(token);
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
    /// Ollama-compatible endpoints under <paramref name="route"/>: POST /chat (NDJSON streaming, think, tools, options,
    /// keep_alive), GET /tags, GET /ps and GET /version, all serving the chat model <paramref name="name"/>. The request
    /// body is read as JSON whatever its Content-Type (as Ollama does). <see cref="OllamaApiOptions.Tools"/> must be set.
    /// </summary>
    public static RouteGroupBuilder MapOllamaApi(this IEndpointRouteBuilder app, string route, string name, Action<OllamaApiOptions> configure)
    {
        var settings = new OllamaApiOptions();
        configure(settings);
        if (settings.Execution is null)
        {
            throw new InvalidOperationException("MapOllamaApi needs options.Tools(ToolExecution.Client) or options.Tools(ToolExecution.Server, maxRounds).");
        }

        var group = app.MapGroup(route);
        group.MapPost("/chat", (HttpRequest http, InferenceEngine engine, CancellationToken token) => Chat(http, engine, name, settings, token))
            .WithName($"OllamaChat-{name}");
        group.MapGet("/tags", async (InferenceEngine engine, CancellationToken token) =>
        {
            ModelDescription d;
            try
            {
                d = await engine.DescribeAsync(name, token);
            }
            catch (FileNotFoundException)
            {
                return Results.Ok(new { models = Array.Empty<OllamaModelTag>() });     // nothing to serve yet
            }

            string served = settings.Served ?? name;
            return Results.Ok(new
            {
                models = new[]
                {
                    new OllamaModelTag(served, served, DateTimeOffset.UtcNow, d.Parameters * sizeof(float),
                        new OllamaModelDetails("idrak", d.Kind.ToString().ToLowerInvariant(), FormatCount(d.Parameters), d.ContextLength ?? 0)),
                },
            });
        }).WithName($"OllamaTags-{name}");
        group.MapGet("/ps", (InferenceEngine engine) =>
        {
            var status = engine.Models.First(m => m.Name == name);
            string served = settings.Served ?? name;
            if (!status.Loaded)
            {
                return Results.Ok(new { models = Array.Empty<OllamaRunningModel>() });
            }

            var d = engine.DescribeAsync(name).GetAwaiter().GetResult();
            long size = d.Parameters * sizeof(float);
            return Results.Ok(new
            {
                models = new[] { new OllamaRunningModel(served, served, size, status.ExpiresAt, d.Device.IsGpu ? size : 0, d.ContextLength ?? 0) },
            });
        }).WithName($"OllamaPs-{name}");
        group.MapGet("/version", () => Results.Ok(new { version = settings.VersionText })).WithName($"OllamaVersion-{name}");
        return group;
    }

    /// <summary>GET <paramref name="route"/>: every engine model with its state and statistics, and every device with its memory use.</summary>
    public static RouteHandlerBuilder MapIdrakStatus(this IEndpointRouteBuilder app, string route) =>
        app.MapGet(route, (InferenceEngine engine) =>
        {
            var devices = new List<Device> { Device.Cpu };
            for (int i = 0; i < Device.CudaDeviceCount; i++)
            {
                devices.Add(Device.Cuda(i));
            }

            return Results.Ok(new
            {
                models = engine.Models.Select(m => new
                {
                    name = m.Name, kind = m.Kind.ToString(), loaded = m.Loaded, instances = m.Instances, running = m.Running,
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

    private static async Task<IResult> Chat(HttpRequest http, InferenceEngine engine, string name, OllamaApiOptions settings, CancellationToken token)
    {
        OllamaChatRequest? request;
        ChatRequest chat;
        try
        {
            request = await JsonSerializer.DeserializeAsync<OllamaChatRequest>(http.Body, Json, token)
                ?? throw new ArgumentException("empty request body");
            (chat, var keepAlive, bool given) = OllamaTranslation.Translate(request);
            if (given)
            {
                engine.KeepAlive(name, keepAlive);
            }
        }
        catch (Exception ex) when (ex is ArgumentException or JsonException or InvalidOperationException)
        {
            return Error(400, ex.Message);
        }

        string served = request.Model ?? settings.Served ?? name;
        bool stream = request.Stream != false;
        var clock = Stopwatch.StartNew();
        IAsyncEnumerator<OllamaChatResponse> lines;
        if (settings.Execution == ToolExecution.Server)
        {
            var tools = engine.ToolsOf(name);
            if (tools is null)
            {
                return Error(500, $"Server-side tool execution is on, but the chat model '{name}' has no tools (GenerativeModelBuilder.Tools).");
            }

            var conversation = Conversation.For(engine.ChatModel(name)).Messages(chat.Messages).Think(chat.Think)
                .Tools(tools).MaxToolRounds(settings.Rounds!.Value);
            if (chat.Options is { } options)
            {
                conversation.Options(options);
            }

            lines = ServerLines(conversation.Build(), served, stream, clock, token).GetAsyncEnumerator(token);
        }
        else
        {
            lines = ClientLines(engine.StreamChatAsync(name, chat, token), served, stream, clock).GetAsyncEnumerator(token);
        }

        // Read the first line before answering, so a full queue, a timeout or a load failure gets a proper status code.
        var first = await Guard(async () => await lines.MoveNextAsync() ? null : Error(500, "no output"));
        if (first is not null)
        {
            await lines.DisposeAsync();
            return first;
        }

        if (!stream)
        {
            OllamaChatResponse last = lines.Current;
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

    private static async IAsyncEnumerable<OllamaChatResponse> ClientLines(IAsyncEnumerable<ChatChunk> chunks, string served, bool stream, Stopwatch clock)
    {
        int toolIndex = 0;
        await foreach (var chunk in chunks)
        {
            if (!chunk.Done)
            {
                if (stream)
                {
                    yield return Line(served, chunk.Delta.Content, chunk.Delta.Thinking, OllamaTranslation.Calls(chunk.Delta.ToolCalls, ref toolIndex));
                }

                continue;
            }

            var message = stream
                ? new OllamaMessage("assistant", chunk.Delta.Content, NullIfEmpty(chunk.Delta.Thinking), OllamaTranslation.Calls(chunk.Delta.ToolCalls, ref toolIndex))
                : new OllamaMessage("assistant", chunk.Message!.Content, chunk.Message.Thinking, OllamaTranslation.Calls(chunk.Message.ToolCalls ?? [], ref toolIndex));
            yield return Final(served, message, chunk.DoneReason, chunk.Stats, clock);
        }
    }

    private static async IAsyncEnumerable<OllamaChatResponse> ServerLines(Conversation conversation, string served, bool stream, Stopwatch clock,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken token)
    {
        await foreach (var delta in conversation.StreamContinueAsync(token))
        {
            if (stream && (delta.Content.Length > 0 || delta.Thinking.Length > 0))
            {
                yield return Line(served, delta.Content, delta.Thinking, null);
            }
        }

        var reply = conversation.LastReply!;
        int index = 0;
        var message = stream
            ? new OllamaMessage("assistant", "", null, reply.ToolLimitReached ? OllamaTranslation.Calls(reply.Message.ToolCalls ?? [], ref index) : null)
            : new OllamaMessage("assistant", reply.Message.Content, reply.Message.Thinking,
                reply.ToolLimitReached ? OllamaTranslation.Calls(reply.Message.ToolCalls ?? [], ref index) : null);
        yield return Final(served, message, reply.DoneReason, reply.Stats, clock);
    }

    private static OllamaChatResponse Line(string served, string content, string thinking, List<OllamaToolCall>? calls) =>
        new(served, DateTimeOffset.UtcNow, new OllamaMessage("assistant", content, NullIfEmpty(thinking), calls), false);

    private static OllamaChatResponse Final(string served, OllamaMessage message, string? reason, GenerationStats? s, Stopwatch clock) =>
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
