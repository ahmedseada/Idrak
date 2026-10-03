// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using Idrak.AspNetCore;
using Idrak.Cli.Shared;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.LanguageModels;

namespace Idrak.Cli.Commands.Serve;

/// <summary>One model of a server: the name clients use, what was asked for, and what is read at startup (the template).</summary>
internal sealed record ServedModel(string Name, string Source, Models.ModelChoice Choice, string Path, string Folder, ChatTemplate? Template, JsonObject? Config);

/// <summary>The server's settings from the command line.</summary>
internal sealed record ServeSettings(string Host, int Port, string? ApiKey, IReadOnlyList<string> Cors, int MaxConcurrency, TimeSpan? KeepAlive, string KeepAliveText)
{
    /// <summary>SHA-256 hashes (hex) of the keys added with <c>idrak server keys add</c>; any of them, or <see cref="ApiKey"/>, is accepted.</summary>
    public IReadOnlyList<string> KeyHashes { get; init; } = [];

    /// <summary>Whether GET /metrics answers (--metrics).</summary>
    public bool Metrics { get; init; }

    /// <summary>The request log file (--log-requests), or null.</summary>
    public string? RequestLog { get; init; }

    /// <summary>Whether the request log holds the request bodies (--log-content).</summary>
    public bool LogContent { get; init; }

    /// <summary>Whether requests need a key.</summary>
    public bool NeedsKey => ApiKey is not null || KeyHashes.Count > 0;
}

/// <summary>
/// The server behind <c>idrak serve</c> and <c>idrak ui</c>: an inference engine with every model loaded on its first
/// request and unloaded after the keep-alive time, the chat API (<c>/api/chat</c>, <c>/api/tags</c>, <c>/api/ps</c>,
/// <c>/api/version</c>), the OpenAI-style API (<c>/v1</c>), a web chat page (<c>/ui</c>) and the control endpoints the
/// <c>server</c> commands use (<c>/idrak/ps</c>, <c>/idrak/load</c>, <c>/idrak/unload</c>, <c>/idrak/stop</c>,
/// <c>/idrak/status</c>).
/// </summary>
internal sealed class ServeHost
{
    /// <summary>The port when none is given: the one common local-model clients connect to.</summary>
    public const int DefaultPort = 11434;

    /// <summary>The options of serve and ui (besides the model options).</summary>
    public static readonly string[] ValueOptions = ["--host", "--port", "--api-key", "--cors", "--max-concurrency", "--keep-alive", "--log-requests"];

    /// <summary>The flags of serve and ui.</summary>
    public static readonly string[] Flags = ["--metrics", "--log-content"];

    /// <summary>Their short forms.</summary>
    public static readonly Dictionary<string, string> ShortForms = new() { ["-p"] = "--port", ["-H"] = "--host" };

    private const string Internal = "/idrak/models";           // where each model's chat API is mapped; /api/chat dispatches to it

    private static readonly string[] ModelRoutes = ["/api/chat", "/v1/chat/completions", "/v1/completions", "/v1/embeddings"];

    private readonly CommandContext _context;
    private readonly object _print = new();
    private readonly ConcurrentDictionary<string, DateTimeOffset> _lastUsed = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, long> _memory = new(StringComparer.Ordinal);
    private readonly ConcurrentDictionary<string, long> _requests = new(StringComparer.Ordinal);
    private StreamWriter? _requestLog;

    public ServeHost(CommandContext context, ServeSettings settings, IReadOnlyList<ServedModel> models)
    {
        _context = context;
        Settings = settings;
        Served = models;
    }

    public ServeSettings Settings { get; }

    public IReadOnlyList<ServedModel> Served { get; }

    /// <summary>The settings from the command's options (usage errors for bad values).</summary>
    public static ServeSettings ReadSettings(CommandContext context)
    {
        string host = context.Option("--host") ?? "127.0.0.1";
        int port = context.IntOption("--port", DefaultPort);
        if (port is < 0 or > 65535)
        {
            throw new UsageException($"--port must be between 0 and 65535, not {port}.");
        }

        int concurrency = context.IntOption("--max-concurrency", 0);
        if (concurrency < 0)
        {
            throw new UsageException("--max-concurrency must be 0 (no limit) or more.");
        }

        string keepText = context.Option("--keep-alive") ?? "5m";
        TimeSpan? keep;
        try
        {
            keep = KeepAlive.Parse(keepText);
        }
        catch (FormatException e)
        {
            throw new UsageException($"--keep-alive: {e.Message}");
        }

        return new ServeSettings(host, port, context.Option("--api-key") ?? Environment.GetEnvironmentVariable("IDRAK_API_KEY"),
            [.. context.Options("--cors").SelectMany(o => o.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))],
            concurrency, keep, keepText)
        {
            KeyHashes = [.. ServerKeys.Hashes(context)],
            Metrics = context.Flag("--metrics"),
            RequestLog = context.Option("--log-requests"),
            LogContent = context.Flag("--log-content") && (context.Option("--log-requests") is not null
                ? true : throw new UsageException("--log-content adds the request bodies to the request log; give the log with --log-requests FILE.")),
        };
    }

    /// <summary>
    /// The models named on the command line: NAME=MODEL serves MODEL as NAME; otherwise an alias keeps its name, a file
    /// or folder is named after it, and a Hugging Face id is used as is. Each is found (or downloaded) and its chat
    /// template read now; the weights load on the first request.
    /// </summary>
    public static List<ServedModel> ReadModels(CommandContext context, IReadOnlyList<string> arguments)
    {
        if (arguments.Count == 0)
        {
            throw new UsageException("Name at least one model to serve (a Hugging Face id, a folder, a GGUF file or an alias).");
        }

        var models = new List<ServedModel>();
        foreach (string argument in arguments)
        {
            string name, source = argument;
            int equals = argument.IndexOf('=');
            if (equals > 0 && !argument[..equals].Contains('/') && !argument[..equals].Contains('\\'))
            {
                (name, source) = (argument[..equals], argument[(equals + 1)..]);
            }
            else
            {
                name = context.Config.Object("aliases")?[argument] is not null ? argument
                    : File.Exists(argument) ? System.IO.Path.GetFileNameWithoutExtension(argument)
                    : Directory.Exists(argument) ? System.IO.Path.GetFileName(System.IO.Path.TrimEndingDirectorySeparator(System.IO.Path.GetFullPath(argument)))
                    : argument;
            }

            if (models.Any(m => m.Name == name))
            {
                throw new UsageException($"Two models are served as '{name}'; name one with NAME=MODEL.");
            }

            var choice = Models.Choose(context, source);
            if (choice.Kv is { } kv)
            {
                _ = KeyValueLayouts.Get(kv);                    // an unknown format fails now, not on the first request
            }

            string path = Models.Resolve(context, choice.Model);
            string folder = CheckpointFormats.For(path).Prepare(path);
            var tokenizer = File.Exists(System.IO.Path.Combine(folder, "tokenizer.json")) ? BpeTokenizer.Load(folder) : null;
            var template = JinjaChatTemplate.Load(folder, tokenizer);
            string configFile = System.IO.Path.Combine(folder, "config.json");
            var config = File.Exists(configFile) ? JsonNode.Parse(File.ReadAllText(configFile)) as JsonObject : null;
            models.Add(new ServedModel(name, source, choice, path, folder, template, config));
        }

        return models;
    }

    /// <summary>Starts the server, prints where it listens, waits for Ctrl+C or <c>idrak server stop</c>, then drains open requests.</summary>
    public int Run(Action<string>? started = null)
    {
        if (Settings.RequestLog is { } log)
        {
            Directory.CreateDirectory(System.IO.Path.GetDirectoryName(System.IO.Path.GetFullPath(log))!);
            _requestLog = new StreamWriter(log, append: true);
        }

        var app = Build();
        try
        {
            app.StartAsync().GetAwaiter().GetResult();
        }
        catch (IOException e)
        {
            ((IAsyncDisposable)app).DisposeAsync().AsTask().GetAwaiter().GetResult();
            _requestLog?.Dispose();
            _context.Error($"idrak serve: cannot listen on {Settings.Host}:{Settings.Port} ({e.Message}). Choose another port with --port, or stop the server using it.");
            return ExitCodes.Failed;
        }

        string url = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        int port = new Uri(url).Port;
        string local = Settings.Host is "0.0.0.0" or "*" or "+" or "::" or "[::]" ? $"http://127.0.0.1:{port}" : url;
        string state = ServerState.Write(_context, local, port, Served.Select(m => m.Name));
        try
        {
            Announce(local);
            started?.Invoke(local);
            app.WaitForShutdownAsync().GetAwaiter().GetResult();
        }
        finally
        {
            ServerState.Remove(state);
            ((IAsyncDisposable)app).DisposeAsync().AsTask().GetAwaiter().GetResult();
            _requestLog?.Dispose();
        }

        Print("server stopped");
        return ExitCodes.Ok;
    }

    private void Announce(string url)
    {
        string keep = Settings.KeepAlive is null ? "stays loaded" : Settings.KeepAlive == TimeSpan.Zero ? "unloads after each request" : $"unloads after {Settings.KeepAliveText} idle";
        _context.Write($"Serving {Served.Count} model{(Served.Count == 1 ? "" : "s")} on {url} (device {_context.Device}; each loads on its first request and {keep})");
        _context.Table(["Model", "Source"], Served.Select(m => (IReadOnlyList<string>)[m.Name, m.Source]));
        _context.Write($"Chat API          {url}/api/chat");
        _context.Write($"OpenAI-style API  {url}/v1");
        _context.Write($"Web chat          {url}/ui");
        if (Settings.Metrics)
        {
            _context.Write($"Metrics           {url}/metrics");
        }

        if (Settings.RequestLog is { } log)
        {
            _context.Write($"Request log       {System.IO.Path.GetFullPath(log)}{(Settings.LogContent ? " (with the request bodies)" : "")}");
        }

        _context.Write($"Stop with Ctrl+C or: idrak server stop -p {new Uri(url).Port}");
        _context.WriteJson(new JsonObject
        {
            ["url"] = url,
            ["chat_api"] = url + "/api",
            ["openai_style_api"] = url + "/v1",
            ["ui"] = url + "/ui",
            ["device"] = _context.Device.ToString(),
            ["keep_alive"] = Settings.KeepAliveText,
            ["max_concurrency"] = Settings.MaxConcurrency,
            ["api_key"] = Settings.NeedsKey,
            ["metrics"] = Settings.Metrics ? url + "/metrics" : null,
            ["request_log"] = Settings.RequestLog is { } file ? System.IO.Path.GetFullPath(file) : null,
            ["models"] = new JsonArray([.. Served.Select(m => (JsonNode)new JsonObject { ["name"] = m.Name, ["source"] = m.Source, ["path"] = m.Path })]),
        });
        _context.Output.Flush();
    }

    private WebApplication Build()
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.Services.Configure<ConsoleLifetimeOptions>(o => o.SuppressStatusMessages = true);
        builder.Services.Configure<HostOptions>(o => o.ShutdownTimeout = TimeSpan.FromSeconds(30));      // open requests drain
        builder.WebHost.UseUrls($"http://{(Settings.Host is "0.0.0.0" or "*" ? "0.0.0.0" : Settings.Host)}:{Settings.Port}");
        if (Settings.Cors.Count > 0)
        {
            builder.Services.AddCors(o => o.AddDefaultPolicy(p =>
            {
                p.AllowAnyHeader().AllowAnyMethod();
                if (Settings.Cors.Contains("*"))
                {
                    p.AllowAnyOrigin();
                }
                else
                {
                    p.WithOrigins([.. Settings.Cors]);
                }
            }));
        }

        var idrak = builder.Services.AddIdrak().LoadOnFirstUse();
        foreach (var model in Served)
        {
            var served = model;
            idrak.AddChatModel(served.Name, _ => Load(served), (_, b) =>
            {
                b = b.KeepAlive(Settings.KeepAlive);
                return served.Template is null ? b : b.Template(served.Template);
            });
        }

        var app = builder.Build();
        if (Settings.Cors.Count > 0)
        {
            app.UseCors();
        }

        app.Use(Log);
        app.Use(Authorize);
        var gate = Settings.MaxConcurrency > 0 ? new SemaphoreSlim(Settings.MaxConcurrency) : null;
        app.Use((http, next) => Dispatch(http, next, gate));
        app.UseRouting();

        for (int i = 0; i < Served.Count; i++)
        {
            string name = Served[i].Name;
            app.MapOllamaApi($"{Internal}/{i}/api", name, o => o.Tools(ToolExecution.Client).ModelName(name).Version(Version));
        }

        app.MapGet("/", () => Results.Text("Idrak is running\n"));
        app.MapGet("/api/version", () => Results.Json(new { version = Version }));
        app.MapGet("/api/tags", Tags);
        app.MapGet("/api/ps", (InferenceEngine engine) => Ps(engine));
        app.MapCompletionsApi("/v1");
        app.MapGet("/ui", () => Results.Content(ChatPage.Html, "text/html; charset=utf-8"));
        app.MapIdrakStatus("/idrak/status");
        if (Settings.Metrics)
        {
            app.MapGet("/metrics", (InferenceEngine engine) => Results.Text(Metrics(engine), "text/plain; version=0.0.4; charset=utf-8"));
        }

        app.MapGet("/idrak/ps", (InferenceEngine engine) => Results.Text(Processes(engine).ToJsonString(), "application/json"));
        app.MapPost("/idrak/load", (HttpRequest request, InferenceEngine engine, CancellationToken token) => Control(request, engine, load: true, token));
        app.MapPost("/idrak/unload", (HttpRequest request, InferenceEngine engine, CancellationToken token) => Control(request, engine, load: false, token));
        app.MapPost("/idrak/stop", (IHostApplicationLifetime lifetime) =>
        {
            Print("stopping (asked through /idrak/stop)");
            _ = Task.Run(async () =>
            {
                await Task.Delay(100);                          // let this answer go out first
                lifetime.StopApplication();
            });
            return Results.Json(new { stopping = true });
        });
        return app;
    }

    // ------------------------------------------------------------------ models

    private TextGenerator Load(ServedModel model)
    {
        var device = _context.Device;
        long before = ComputeResources.GetMemoryUsage(device).InUse;
        var watch = Stopwatch.StartNew();
        var pretrained = Models.Load(_context, model.Choice with { Model = model.Path });
        var generator = model.Choice.Kv is { } kv ? pretrained.CreateGenerator(KeyValueLayouts.Get(kv), model.Choice.Context)
            : pretrained.CreateGenerator(contextLength: model.Choice.Context);
        _memory[model.Name] = Math.Max(0, ComputeResources.GetMemoryUsage(device).InUse - before);
        Print($"loaded {model.Name} in {watch.Elapsed.TotalSeconds:F1} s on {device}");
        return generator;
    }

    // The served model a request names; with anyName (client requests), any name selects the only model.
    private ServedModel? Find(string? name, bool anyName = true)
    {
        if (name is not null)
        {
            var exact = Served.FirstOrDefault(m => m.Name == name)
                ?? (name.EndsWith(":latest", StringComparison.Ordinal) ? Served.FirstOrDefault(m => m.Name == name[..^":latest".Length]) : null);
            if (exact is not null)
            {
                return exact;
            }
        }

        return anyName && Served.Count == 1 ? Served[0] : null;
    }

    // ------------------------------------------------------------------ middleware

    private async Task Log(HttpContext http, Func<Task> next)
    {
        var watch = Stopwatch.StartNew();
        string path = http.Request.Path;                    // before /api/chat is sent to its model's route
        await next();
        if (_context.Verbose)
        {
            Print($"{http.Request.Method} {path} {http.Response.StatusCode} {watch.Elapsed.TotalMilliseconds:F0} ms");
        }
    }

    private async Task Authorize(HttpContext http, Func<Task> next)
    {
        string path = http.Request.Path.Value ?? "";
        if (!Settings.NeedsKey || HttpMethods.IsOptions(http.Request.Method) || path is "/" or "/ui")
        {
            await next();
            return;
        }

        string given = http.Request.Headers.Authorization.ToString() is { } auth && auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase)
            ? auth["Bearer ".Length..].Trim() : http.Request.Headers["X-API-Key"].ToString();
        bool valid = Settings.ApiKey is { } key && System.Security.Cryptography.CryptographicOperations.FixedTimeEquals(
            System.Text.Encoding.UTF8.GetBytes(given), System.Text.Encoding.UTF8.GetBytes(key));
        valid |= given.Length > 0 && ServerKeys.Matches(given, Settings.KeyHashes);
        if (!valid)
        {
            http.Response.StatusCode = StatusCodes.Status401Unauthorized;
            await http.Response.WriteAsJsonAsync(new { error = "A valid API key is required (Authorization: Bearer KEY)." });
            return;
        }

        await next();
    }

    // Model requests: waits for a free slot (--max-concurrency), sends /api/chat to the chat API of the model named in
    // the body, records when each model was last used and writes the request log line (--log-requests).
    private async Task Dispatch(HttpContext http, Func<Task> next, SemaphoreSlim? gate)
    {
        string path = http.Request.Path.Value ?? "";
        if (!HttpMethods.IsPost(http.Request.Method) || !ModelRoutes.Contains(path))
        {
            await next();
            return;
        }

        var started = DateTimeOffset.UtcNow;
        var watch = Stopwatch.StartNew();
        http.Request.EnableBuffering();
        string body = await new StreamReader(http.Request.Body, leaveOpen: true).ReadToEndAsync(http.RequestAborted);
        http.Request.Body.Position = 0;
        string? asked = null;
        try
        {
            using var document = JsonDocument.Parse(body);
            if (document.RootElement.ValueKind == JsonValueKind.Object && document.RootElement.TryGetProperty("model", out var model) && model.ValueKind == JsonValueKind.String)
            {
                asked = model.GetString();
            }
        }
        catch (JsonException)
        {
            // The endpoint answers bad JSON with 400.
        }

        var served = Find(asked);
        if (path == "/api/chat")
        {
            if (served is null)
            {
                http.Response.StatusCode = StatusCodes.Status404NotFound;
                await http.Response.WriteAsJsonAsync(new { error = $"model '{asked}' not found; served: {string.Join(", ", Served.Select(m => m.Name))}" });
                return;
            }

            http.Request.Path = $"{Internal}/{IndexOf(served)}/api/chat";
        }

        // The last part of the answer is kept to read its token counts (the usage, or the final line's counts).
        var tail = _requestLog is null ? null : new TailStream(http.Response.Body);
        if (tail is not null)
        {
            http.Response.Body = tail;
        }

        if (gate is not null)
        {
            await gate.WaitAsync(http.RequestAborted);
        }

        try
        {
            await next();
        }
        finally
        {
            gate?.Release();
            if (served is not null)
            {
                _lastUsed[served.Name] = DateTimeOffset.UtcNow;
                _requests.AddOrUpdate(served.Name, 1, (_, n) => n + 1);
            }

            if (tail is not null)
            {
                http.Response.Body = tail.Inner;
                WriteLogLine(http, path, served?.Name ?? asked, body, tail, started, watch.Elapsed);
            }
        }
    }

    private void WriteLogLine(HttpContext http, string path, string? model, string body, TailStream tail, DateTimeOffset started, TimeSpan elapsed)
    {
        var (prompt, generated) = TailStream.Tokens(tail.Text);
        double seconds = elapsed.TotalSeconds;
        var line = new JsonObject
        {
            ["time"] = started.ToString("O", CultureInfo.InvariantCulture),
            ["method"] = http.Request.Method,
            ["path"] = path,
            ["model"] = model,
            ["status"] = http.Response.StatusCode,
            ["ms"] = Math.Round(seconds * 1000, 1),
            ["prompt_tokens"] = prompt,
            ["completion_tokens"] = generated,
            ["tokens_per_second"] = generated is > 0 && seconds > 0 ? Math.Round(generated.Value / seconds, 2) : null,
        };
        if (Settings.LogContent)
        {
            line["request"] = ServerClient.TryParse(body) ?? JsonValue.Create(body);
        }

        lock (_requestLog!)
        {
            _requestLog.WriteLine(line.ToJsonString());
            _requestLog.Flush();
        }
    }

    private int IndexOf(ServedModel model)
    {
        for (int i = 0; i < Served.Count; i++)
        {
            if (ReferenceEquals(Served[i], model))
            {
                return i;
            }
        }

        return -1;
    }

    // ------------------------------------------------------------------ chat API: tags and ps over every model

    private IResult Tags()
    {
        var models = new JsonArray();
        foreach (var m in Served)
        {
            var written = File.Exists(m.Path) ? File.GetLastWriteTimeUtc(m.Path) : Directory.Exists(m.Path) ? Directory.GetLastWriteTimeUtc(m.Path) : DateTime.UtcNow;
            models.Add(new JsonObject
            {
                ["name"] = m.Name,
                ["model"] = m.Name,
                ["modified_at"] = new DateTimeOffset(written, TimeSpan.Zero).ToString("O", CultureInfo.InvariantCulture),
                ["size"] = SizeOnDisk(m.Path),
                ["details"] = new JsonObject
                {
                    ["format"] = m.Path.EndsWith(".gguf", StringComparison.OrdinalIgnoreCase) ? "gguf" : "safetensors",
                    ["family"] = (string?)m.Config?["model_type"] ?? "",
                    ["parameter_size"] = "",
                    ["context_length"] = (int?)m.Config?["max_position_embeddings"] ?? 0,
                },
            });
        }

        return Results.Text(new JsonObject { ["models"] = models }.ToJsonString(), "application/json");
    }

    private IResult Ps(InferenceEngine engine)
    {
        var models = new JsonArray();
        foreach (var (model, status, description) in Loaded(engine))
        {
            long memory = _memory.GetValueOrDefault(model.Name);
            models.Add(new JsonObject
            {
                ["name"] = model.Name,
                ["model"] = model.Name,
                ["size"] = memory,
                ["expires_at"] = status.ExpiresAt?.ToString("O", CultureInfo.InvariantCulture),
                ["size_vram"] = description?.Device.IsGpu == true ? memory : 0,
                ["context_length"] = description?.ContextLength ?? 0,
            });
        }

        return Results.Text(new JsonObject { ["models"] = models }.ToJsonString(), "application/json");
    }

    private IEnumerable<(ServedModel Model, ModelStatus Status, ModelDescription? Description)> Loaded(InferenceEngine engine)
    {
        foreach (var model in Served)
        {
            var status = engine.Models.First(s => s.Name == model.Name);
            if (status.Loaded)
            {
                yield return (model, status, Describe(engine, model.Name));
            }
        }
    }

    private static ModelDescription? Describe(InferenceEngine engine, string name)
    {
        try
        {
            var task = engine.DescribeAsync(name);
            return task.IsCompleted ? task.GetAwaiter().GetResult() : null;          // only a loaded model's description, never a load
        }
        catch (InvalidOperationException)
        {
            return null;
        }
    }

    // ------------------------------------------------------------------ control endpoints

    private JsonObject Processes(InferenceEngine engine)
    {
        var models = new JsonArray();
        foreach (var model in Served)
        {
            var status = engine.Models.First(s => s.Name == model.Name);
            var description = status.Loaded ? Describe(engine, model.Name) : null;
            models.Add(new JsonObject
            {
                ["name"] = model.Name,
                ["source"] = model.Source,
                ["loaded"] = status.Loaded,
                ["device"] = description?.Device.ToString() ?? _context.Device.ToString(),
                ["parameters"] = description?.Parameters,
                ["memory_bytes"] = status.Loaded ? _memory.GetValueOrDefault(model.Name) : 0,
                ["context_length"] = description?.ContextLength,
                ["running"] = status.Running,
                ["queued"] = status.Queued,
                ["expires_at"] = status.ExpiresAt?.ToString("O", CultureInfo.InvariantCulture),
                ["last_used"] = _lastUsed.TryGetValue(model.Name, out var used) ? used.ToString("O", CultureInfo.InvariantCulture) : null,
                ["requests"] = _requests.GetValueOrDefault(model.Name),
            });
        }

        var devices = new JsonArray();
        foreach (var device in new[] { _context.Device })
        {
            var memory = ComputeResources.GetMemoryUsage(device);
            devices.Add(new JsonObject { ["device"] = device.ToString(), ["memory_in_use"] = memory.InUse, ["memory_limit"] = memory.Limit });
        }

        return new JsonObject { ["models"] = models, ["devices"] = devices, ["keep_alive"] = Settings.KeepAliveText };
    }

    private async Task<IResult> Control(HttpRequest request, InferenceEngine engine, bool load, CancellationToken token)
    {
        string? asked;
        try
        {
            asked = (string?)(await JsonNode.ParseAsync(request.Body, cancellationToken: token))?["model"];
        }
        catch (JsonException e)
        {
            return Results.Json(new { error = e.Message }, statusCode: StatusCodes.Status400BadRequest);
        }

        var model = asked is null ? null : Find(asked, anyName: false);
        if (model is null)
        {
            return Results.Json(new { error = $"model '{asked}' not found; served: {string.Join(", ", Served.Select(m => m.Name))}" }, statusCode: StatusCodes.Status404NotFound);
        }

        var watch = Stopwatch.StartNew();
        if (load)
        {
            await engine.LoadAsync(model.Name, token);
            _lastUsed[model.Name] = DateTimeOffset.UtcNow;
        }
        else
        {
            await engine.UnloadAsync(model.Name);
            _memory.TryRemove(model.Name, out _);
            Print($"unloaded {model.Name}");
        }

        return Results.Json(new { model = model.Name, loaded = load, seconds = Math.Round(watch.Elapsed.TotalSeconds, 2) });
    }

    // ------------------------------------------------------------------ metrics

    // The plain text exposition format monitoring tools scrape: HELP and TYPE lines, then one sample per line.
    private string Metrics(InferenceEngine engine)
    {
        var text = new System.Text.StringBuilder();
        void Family(string name, string type, string help, IEnumerable<(string Labels, double Value)> samples)
        {
            text.Append("# HELP ").Append(name).Append(' ').Append(help).Append('\n');
            text.Append("# TYPE ").Append(name).Append(' ').Append(type).Append('\n');
            foreach (var (labels, value) in samples)
            {
                text.Append(name).Append(labels).Append(' ').Append(value.ToString("R", CultureInfo.InvariantCulture)).Append('\n');
            }
        }

        static string Label(string key, string value) => $"{{{key}=\"{value.Replace("\\", "\\\\").Replace("\"", "\\\"")}\"}}";
        var rows = Served.Select(m => (Name: Label("model", m.Name), Model: m, Stats: engine.Stats(m.Name), Status: engine.Models.First(s => s.Name == m.Name))).ToList();
        Family("idrak_requests_total", "counter", "Requests answered by each model.", rows.Select(r => (r.Name, (double)r.Stats.Requests)));
        Family("idrak_requests_rejected_total", "counter", "Requests rejected because the queue was full.", rows.Select(r => (r.Name, (double)r.Stats.Rejected)));
        Family("idrak_requests_failed_total", "counter", "Requests that failed.", rows.Select(r => (r.Name, (double)r.Stats.Failed)));
        Family("idrak_generated_tokens_total", "counter", "Tokens generated by each model.", rows.Select(r => (r.Name, (double)r.Stats.Rows)));
        Family("idrak_tokens_per_second", "gauge", "Tokens generated per second of model time.", rows.Select(r => (r.Name, r.Stats.RowsPerSecond)));
        Family("idrak_request_latency_seconds", "gauge", "Mean time of the last 1,024 requests, queue wait included.", rows.Select(r => (r.Name, r.Stats.AverageLatency.TotalSeconds)));
        Family("idrak_request_latency_p95_seconds", "gauge", "95th percentile of the same.", rows.Select(r => (r.Name, r.Stats.P95Latency.TotalSeconds)));
        Family("idrak_queue_wait_seconds", "gauge", "Mean time the last 1,024 requests waited for the model.", rows.Select(r => (r.Name, r.Stats.AverageQueueWait.TotalSeconds)));
        Family("idrak_requests_running", "gauge", "Requests generating now.", rows.Select(r => (r.Name, (double)r.Status.Running)));
        Family("idrak_requests_queued", "gauge", "Requests waiting for the model.", rows.Select(r => (r.Name, (double)r.Status.Queued)));
        Family("idrak_model_loaded", "gauge", "1 when the model is loaded.", rows.Select(r => (r.Name, r.Status.Loaded ? 1.0 : 0.0)));
        Family("idrak_model_memory_bytes", "gauge", "Device memory the model took when it loaded.",
            rows.Select(r => (r.Name, r.Status.Loaded ? (double)_memory.GetValueOrDefault(r.Model.Name) : 0)));
        var memory = ComputeResources.GetMemoryUsage(_context.Device);
        string device = Label("device", _context.Device.ToString());
        Family("idrak_device_memory_in_use_bytes", "gauge", "Memory in use on the serving device.", [(device, memory.InUse)]);
        Family("idrak_device_memory_cached_bytes", "gauge", "Memory cached for reuse on the serving device.", [(device, memory.Cached)]);
        return text.ToString();
    }


    // ------------------------------------------------------------------ helpers

    private static string Version => typeof(InferenceEngine).Assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()?.InformationalVersion ?? "0.0.0";

    private static long SizeOnDisk(string path) =>
        File.Exists(path) ? new FileInfo(path).Length
        : Directory.Exists(path) ? Directory.EnumerateFiles(path, "*", SearchOption.AllDirectories).Sum(f => new FileInfo(f).Length) : 0;

    // A line from a request thread, whole (not with --quiet or --json).
    private void Print(string line)
    {
        lock (_print)
        {
            _context.Write(line);
            _context.Output.Flush();
        }
    }
}

/// <summary>
/// A small file per running server (<c>CACHE/servers/PORT.json</c>: its URL, process and models), so the server
/// commands find it without --port.
/// </summary>
internal static class ServerState
{
    public static string Folder(CommandContext context) => Path.Combine(context.CacheFolder, "servers");

    public static string Write(CommandContext context, string url, int port, IEnumerable<string> models)
    {
        string file = Path.Combine(Folder(context), $"{port}.json");
        try
        {
            Directory.CreateDirectory(Folder(context));
            File.WriteAllText(file, new JsonObject
            {
                ["url"] = url,
                ["pid"] = Environment.ProcessId,
                ["started"] = DateTimeOffset.UtcNow.ToString("O", CultureInfo.InvariantCulture),
                ["models"] = new JsonArray([.. models.Select(m => (JsonNode)m)]),
            }.ToJsonString());
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            context.Detail($"could not write {file}: {e.Message}");
        }

        return file;
    }

    public static void Remove(string file)
    {
        try
        {
            File.Delete(file);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
        }
    }

    /// <summary>The URL of the most recently started server whose process still runs, or null.</summary>
    public static string? Find(CommandContext context)
    {
        if (!Directory.Exists(Folder(context)))
        {
            return null;
        }

        var running = new List<(DateTimeOffset Started, string Url)>();
        foreach (string file in Directory.EnumerateFiles(Folder(context), "*.json"))
        {
            try
            {
                var state = JsonNode.Parse(File.ReadAllText(file))!;
                if (Alive((int)state["pid"]!))
                {
                    running.Add((DateTimeOffset.Parse((string)state["started"]!, CultureInfo.InvariantCulture), (string)state["url"]!));
                }
            }
            catch (Exception e) when (e is JsonException or IOException or FormatException or InvalidOperationException or NullReferenceException)
            {
                // a file being written or left over: skip it
            }
        }

        return running.Count == 0 ? null : running.MaxBy(r => r.Started).Url;
    }

    private static bool Alive(int pid)
    {
        try
        {
            using var process = Process.GetProcessById(pid);
            return !process.HasExited;
        }
        catch (ArgumentException)
        {
            return false;
        }
    }
}
