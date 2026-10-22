// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Globalization;
using System.Net;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Commands.Serve;
using Idrak.Cli.Shared;
using Idrak.Cluster;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>The dashboard's settings (plan 23, phase C9): where it listens, who may call it, and what the page starts with.</summary>
/// <param name="Host">The address it listens on (<c>--host</c>, <c>IDRAK_CLUSTER_UI_HOST</c>, 127.0.0.1).</param>
/// <param name="Port">The port (<c>--port</c>, <c>IDRAK_CLUSTER_UI_PORT</c>; 0: one the system picks).</param>
/// <param name="ApiKey">A key requests from other machines may give (<c>IDRAK_API_KEY</c>), or null.</param>
/// <param name="KeyHashes">The keys <c>idrak server keys add</c> stored (their hashes).</param>
/// <param name="Topology">The topology file the page opens and saves (<c>--topology</c>), or null.</param>
internal sealed record DashboardSettings(string Host, int Port, string? ApiKey, IReadOnlyList<string> KeyHashes, string? Topology)
{
    /// <summary>The variable choosing the port.</summary>
    public const string PortVariable = "IDRAK_CLUSTER_UI_PORT";

    /// <summary>The variable choosing the address.</summary>
    public const string HostVariable = "IDRAK_CLUSTER_UI_HOST";

    /// <summary>The records kept per run for a page that connects late (<c>IDRAK_CLUSTER_UI_HISTORY</c>).</summary>
    public int History { get; init; } = 20_000;

    /// <summary>How often a listener reads a run folder's new lines (<c>IDRAK_CLUSTER_UI_POLL</c>, seconds).</summary>
    public TimeSpan Poll { get; init; } = TimeSpan.FromMilliseconds(500);

    /// <summary>The unit the page shows sizes in (B, KB, MB, GB, TB, powers of 1,024), or null: the largest each fills.</summary>
    public string? Unit { get; init; }

    /// <summary>The page's language: the tool's ("en" or "ar", right to left).</summary>
    public string Language { get; init; } = "en";

    /// <summary>A run folder to follow from the start (<c>--follow</c>), or null.</summary>
    public string? Follow { get; init; }

    /// <summary>Whether a key is configured for requests from other machines.</summary>
    public bool NeedsKey => ApiKey is not null || KeyHashes.Count > 0;

    /// <summary>The settings from a command's options and the environment.</summary>
    /// <exception cref="UsageException">A bad port, unit or setting; the message names its source.</exception>
    public static DashboardSettings Read(CommandContext context, string? topology)
    {
        string host = context.Option("--host") ?? (Environment.GetEnvironmentVariable(HostVariable) is { Length: > 0 } named ? named : "127.0.0.1");
        int port = Whole(context.Option("--port"), "--port", 0, 65535) ?? Whole(Environment.GetEnvironmentVariable(PortVariable), PortVariable, 0, 65535) ?? 0;
        string? unit = context.Option("--unit");
        if (unit is not null && unit.Replace("i", "", StringComparison.OrdinalIgnoreCase).ToUpperInvariant() is not ("B" or "KB" or "MB" or "GB" or "TB"))
        {
            throw new UsageException($"--unit: '{unit}' is not a unit; use B, KB, MB, GB or TB (powers of 1,024).");
        }

        double poll = Environment.GetEnvironmentVariable("IDRAK_CLUSTER_UI_POLL") is { Length: > 0 } pollText
            ? double.TryParse(pollText, NumberStyles.Float, CultureInfo.InvariantCulture, out double seconds) && seconds > 0 && double.IsFinite(seconds)
                ? seconds : throw new UsageException($"IDRAK_CLUSTER_UI_POLL={pollText} is not a number of seconds above 0.")
            : 0.5;
        return new DashboardSettings(host, port, Environment.GetEnvironmentVariable("IDRAK_API_KEY") is { Length: > 0 } key ? key : null, [.. ServerKeys.Hashes(context)], topology)
        {
            History = Whole(Environment.GetEnvironmentVariable("IDRAK_CLUSTER_UI_HISTORY"), "IDRAK_CLUSTER_UI_HISTORY", 1, int.MaxValue) ?? 20_000,
            Poll = TimeSpan.FromSeconds(poll),
            Unit = unit?.Replace("i", "", StringComparison.OrdinalIgnoreCase).ToUpperInvariant(),
            Language = Messages.Language,
            Follow = context.Option("--follow"),
        };
    }

    private static int? Whole(string? text, string source, int min, int max) => text is not { Length: > 0 } ? null
        : int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= min && value <= max
            ? value : throw new UsageException($"{source} must be a whole number from {min} to {max}, not '{text}'.");
}

/// <summary>
/// The cluster dashboard (plan 23, phase C9, decision 8): one self-contained page and the API it calls, served by ASP.NET
/// Core as <c>idrak serve</c> is. It sets up topologies through the same <see cref="TopologyBuilder"/> as
/// <c>cluster init</c>, inspects them (<see cref="TopologyInspection"/>), starts simulated runs in this process or as
/// processes of this machine, steers them through the library (<see cref="ClusterRunControl"/>, the launcher's
/// <see cref="LocalhostControl"/>), follows a real run's folder read-only, and streams every run's records as server-sent
/// events. It listens on 127.0.0.1 unless told otherwise; a request from another machine needs an API key (the
/// <c>idrak server keys</c> store or <c>IDRAK_API_KEY</c>, as Bearer or X-API-Key), and a request another site's page
/// makes is refused (no CORS, the Origin and Host checked).
/// </summary>
internal sealed class DashboardServer : IDisposable
{
    /// <summary>Every route the server answers, as "METHOD /path" with {id} for a run's id: the page calls these and no other.</summary>
    public static readonly string[] Routes =
    [
        "GET /api/machine", "POST /api/topology/questions", "GET /api/topology", "POST /api/topology", "GET /api/topology/inspect",
        "POST /api/topology/inspect", "GET /api/runs", "POST /api/runs", "POST /api/runs/{id}/pause", "POST /api/runs/{id}/resume",
        "POST /api/runs/{id}/stop", "POST /api/runs/{id}/faults", "GET /api/runs/{id}/events",
    ];

    private readonly Lazy<MachineFacts> _facts;
    private WebApplication? _app;

    public DashboardServer(DashboardSettings settings, Func<MachineFacts>? facts = null)
    {
        Settings = settings;
        Runs = new DashboardRuns(settings);
        _facts = new Lazy<MachineFacts>(facts ?? MachineFacts.Gather, LazyThreadSafetyMode.ExecutionAndPublication);
    }

    public DashboardSettings Settings { get; }

    public DashboardRuns Runs { get; }

    /// <summary>Where it listens (once started).</summary>
    public string? Url { get; private set; }

    /// <summary>The page's address.</summary>
    public string? PageUrl => Url is null ? null : Url + "/";

    /// <summary>Whether the address it listens on is this machine's alone.</summary>
    public bool Loopback => IsLoopbackHost(Settings.Host);

    /// <summary>
    /// Why a request from <paramref name="remote"/> giving <paramref name="key"/> is refused, or null when it is taken: this
    /// machine's requests always are; another machine's need a valid key, and none is taken when no key is configured.
    /// </summary>
    public static string? Refusal(IPAddress? remote, string? key, DashboardSettings settings)
    {
        ArgumentNullException.ThrowIfNull(settings);
        if (remote is null || IPAddress.IsLoopback(remote))
        {
            return null;
        }

        if (!settings.NeedsKey)
        {
            return "This dashboard answers other machines only with an API key: add one with idrak server keys add NAME (or set IDRAK_API_KEY) and start it again.";
        }

        string given = key ?? "";
        bool valid = settings.ApiKey is { } own && CryptographicOperations.FixedTimeEquals(Encoding.UTF8.GetBytes(given), Encoding.UTF8.GetBytes(own));
        valid |= given.Length > 0 && ServerKeys.Matches(given, settings.KeyHashes);
        return valid ? null : "A valid API key is required from another machine (Authorization: Bearer KEY, or X-API-Key).";
    }

    /// <summary>Starts listening; returns the address (the port the system picked, when 0 was asked).</summary>
    /// <exception cref="IOException">The address or port cannot be listened on.</exception>
    public string Start()
    {
        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.Services.Configure<ConsoleLifetimeOptions>(o => o.SuppressStatusMessages = true);
        builder.Services.Configure<HostOptions>(o => o.ShutdownTimeout = TimeSpan.FromSeconds(5));
        string host = Settings.Host is "0.0.0.0" or "*" ? "0.0.0.0" : Settings.Host.Contains(':', StringComparison.Ordinal) && !Settings.Host.StartsWith('[') ? $"[{Settings.Host}]" : Settings.Host;
        builder.WebHost.UseUrls(string.Create(CultureInfo.InvariantCulture, $"http://{host}:{Settings.Port}"));
        var app = builder.Build();
        app.Use(Guard);
        Map(app);
        app.StartAsync().GetAwaiter().GetResult();
        _app = app;
        string url = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
        int port = new Uri(url).Port;
        Url = Settings.Host is "0.0.0.0" or "*" or "::" or "[::]" ? string.Create(CultureInfo.InvariantCulture, $"http://127.0.0.1:{port}") : url.TrimEnd('/');
        if (Settings.Follow is { } follow)
        {
            Runs.Follow(follow);
        }

        return Url;
    }

    public void Dispose()
    {
        Runs.Dispose();
        if (_app is { } app)
        {
            _app = null;
            app.StopAsync().GetAwaiter().GetResult();
            ((IAsyncDisposable)app).DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }

    private static bool IsLoopbackHost(string host) =>
        host is "localhost" or "127.0.0.1" or "::1" or "[::1]" || (IPAddress.TryParse(host.Trim('[', ']'), out var address) && IPAddress.IsLoopback(address));

    // Every request: from another site's page refused (its Origin is not this server, or a name that is not this machine's
    // reached a loopback dashboard), from another machine without a valid key refused.
    private async Task Guard(HttpContext http, Func<Task> next)
    {
        string? origin = http.Request.Headers.Origin.FirstOrDefault();
        bool foreignOrigin = origin is { Length: > 0 } && (!Uri.TryCreate(origin, UriKind.Absolute, out var from) || !string.Equals(from.Authority, http.Request.Host.Value, StringComparison.OrdinalIgnoreCase));
        bool foreignHost = Loopback && !IsLoopbackHost(http.Request.Host.Host);
        if (foreignOrigin || foreignHost)
        {
            await Error(http, StatusCodes.Status403Forbidden, foreignOrigin ? "Requests from another site's page are refused." : $"This dashboard answers to this machine's names only, not {http.Request.Host.Host}.");
            return;
        }

        if ((http.Request.Path.Value ?? "/") != "/")
        {
            string? key = http.Request.Headers.Authorization.ToString() is { } auth && auth.StartsWith("Bearer ", StringComparison.OrdinalIgnoreCase)
                ? auth["Bearer ".Length..].Trim() : http.Request.Headers["X-API-Key"].FirstOrDefault();
            if (Refusal(http.Connection.RemoteIpAddress, key, Settings) is { } refusal)
            {
                await Error(http, StatusCodes.Status401Unauthorized, refusal);
                return;
            }
        }

        await next();
    }

    private void Map(WebApplication app)
    {
        app.MapGet("/", () => Results.Content(DashboardPage.For(Settings.Language), "text/html; charset=utf-8"));
        app.MapGet("/api/machine", () => Json(DashboardTopology.Machine(_facts.Value, Settings)));
        app.MapPost("/api/topology/questions", (HttpRequest request) => Body(request, body => Json(DashboardTopology.Ask(_facts.Value, body))));
        app.MapGet("/api/topology", (string? path) => Saved(path));
        app.MapPost("/api/topology", (HttpRequest request) => Body(request, Save));
        app.MapGet("/api/topology/inspect", (string? path, string? bytes) => Inspect(path, null, bytes));
        app.MapPost("/api/topology/inspect", (HttpRequest request) => Body(request, body => Inspect((string?)body["path"], body["answers"], (string?)body["bytes"])));
        app.MapGet("/api/runs", () => Json(new JsonObject { ["runs"] = new JsonArray([.. Runs.All.Select(r => (JsonNode)r.Summary())]), ["active"] = Runs.Active?.Id }));
        app.MapPost("/api/runs", (HttpRequest request) => Body(request, StartRun));
        app.MapPost("/api/runs/{id}/pause", (string id) => Steer(id, "pause"));
        app.MapPost("/api/runs/{id}/resume", (string id) => Steer(id, "resume"));
        app.MapPost("/api/runs/{id}/stop", (string id) => Steer(id, "stop"));
        app.MapPost("/api/runs/{id}/faults", (string id, HttpRequest request) => Body(request, body => Fault(id, body)));
        app.MapGet("/api/runs/{id}/events", (string id, HttpContext http) => Events(id, http));
    }

    // The saved topology file (the server's, or ?path=) as the builder's answers and its text.
    private IResult Saved(string? path)
    {
        string? file = path ?? Settings.Topology;
        if (file is null)
        {
            return Json(new JsonObject { ["path"] = null, ["answers"] = null, ["json"] = null });
        }

        if (!File.Exists(file))
        {
            return Json(new JsonObject { ["path"] = Path.GetFullPath(file), ["answers"] = null, ["json"] = null, ["missing"] = true });
        }

        try
        {
            var topology = ClusterTopology.Load(file);
            return Json(new JsonObject { ["path"] = Path.GetFullPath(file), ["answers"] = DashboardTopology.AnswersOf(topology), ["json"] = File.ReadAllText(file) });
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException)
        {
            return Error(StatusCodes.Status400BadRequest, e.Message);
        }
    }

    // POST /api/topology: the answers checked and the file written (when "save" is true) where the body or the server says.
    private IResult Save(JsonObject body)
    {
        var (topology, json) = DashboardTopology.Build(_facts.Value, body["answers"]);
        string? path = (string?)body["path"] ?? Settings.Topology;
        bool save = body["save"]?.GetValue<bool>() ?? false;
        if (save)
        {
            if (path is null)
            {
                return Error(StatusCodes.Status400BadRequest, "Give the file to save to (\"path\"), or start the dashboard with --topology FILE.");
            }

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
            File.WriteAllText(path, json);
        }

        return Json(new JsonObject
        {
            ["path"] = path is null ? null : Path.GetFullPath(path), ["saved"] = save, ["nodes"] = topology.Nodes.Count, ["ranks"] = topology.WorldSize, ["json"] = json,
        });
    }

    private IResult Inspect(string? path, JsonNode? answers, string? bytesText)
    {
        long bytes;
        try
        {
            bytes = ClusterTopology.ParseMemory(bytesText ?? DashboardTopology.DefaultPredictBytes);
        }
        catch (FormatException e)
        {
            return Error(StatusCodes.Status400BadRequest, $"bytes: {e.Message}");
        }

        try
        {
            if (answers is not null)
            {
                return Json(DashboardTopology.Inspect(DashboardTopology.Build(_facts.Value, answers).Topology, _facts.Value, bytes, null));
            }

            string file = path ?? Settings.Topology ?? throw new ArgumentException("No topology yet: give its answers, a path, or start the dashboard with --topology FILE.");
            return Json(DashboardTopology.Inspect(ClusterTopology.Load(file), _facts.Value, bytes, file));
        }
        catch (Exception e) when (e is ArgumentException or InvalidDataException or NotSupportedException or FileNotFoundException)
        {
            return Error(StatusCodes.Status400BadRequest, e.Message);
        }
    }

    // POST /api/runs: a simulated run of a training command (in this process, or as processes here), or a folder followed.
    private IResult StartRun(JsonObject body)
    {
        if ((string?)body["follow"] is { } folder)
        {
            return Json(Runs.Follow(folder).Summary());
        }

        string path = (string?)body["topology"] ?? Settings.Topology ?? throw new ArgumentException("Give the topology (\"topology\"), or save one first.");
        var command = body["command"] switch
        {
            JsonArray list => list.Select(a => (string?)a ?? "").ToList(),
            JsonValue text => Split((string?)text ?? ""),
            _ => throw new ArgumentException("Give the training command (\"command\": [\"pretrain\", MODEL, …, \"-o\", RUN])."),
        };
        if (command.Count > 0 && command[0] == "idrak")
        {
            command.RemoveAt(0);
        }

        ClusterSimCommand.CheckTrainer(command);
        var topology = ClusterTopology.Load(path);
        var options = ClusterOptions.FromEnvironment() with { Topology = path };
        if ((string?)body["collective"] is { Length: > 0 } collective)
        {
            options = options with { Collective = collective };
        }

        if ((string?)body["wire"] is { Length: > 0 } wire)
        {
            options = options with { Wire = wire is "bf16" or "bfloat16" ? WireType.BFloat16 : wire is "f32" or "float32" ? WireType.Float32 : throw new ArgumentException($"wire is f32 or bf16, not '{wire}'.") };
        }

        if (body["seed"] is { } seed)
        {
            options = options with { Simulation = options.Simulation with { Seed = seed.GetValueKind() == System.Text.Json.JsonValueKind.Number ? seed.GetValue<int>() : int.Parse((string)seed!, CultureInfo.InvariantCulture) } };
        }

        if ((string?)body["failures"] is { Length: > 0 } failures)
        {
            options = options with { Simulation = options.Simulation with { Failures = failures } };
        }

        string mode = (string?)body["mode"] ?? "in-process";
        _ = Collectives.Get(options.CollectiveFor(topology));
        string model = options.Simulation.Failures ?? topology.Simulation.Failures;
        _ = FailureModels.Get(model);
        bool paused = body["paused"]?.GetValue<bool>() ?? false;
        DashboardRun run;
        switch (mode)
        {
            case "in-process":
                run = Runs.Attach(RunKind.InProcess, path, command, options, topology);
                options = options with { Simulation = options.Simulation with { Mode = SimulationMode.InProcess, EventLog = options.Simulation.EventLog ?? run.EventLogPath } };
                var control = new ClusterRunControl();
                run.Bind(control);
                if (paused)
                {
                    control.Pause();
                }

                run.Worker = new Thread(() =>
                {
                    try
                    {
                        var (result, _, _) = ClusterSimCommand.RunRanks(topology, options, command, run.Output(null), run.Errors(null), control);
                        run.Finish(result, control);
                    }
                    catch (Exception e)
                    {
                        run.Finish("failed", e.Message);
                    }
                }) { IsBackground = true, Name = $"dashboard {run.Id}" };
                break;
            case "localhost":
                if (!string.Equals(model, "none", StringComparison.OrdinalIgnoreCase))
                {
                    throw new ArgumentException($"failures {model} needs the simulated transport (mode in-process): a localhost run joins its processes by the real transports, which inject no faults.");
                }

                run = Runs.Attach(RunKind.Localhost, path, command, options, topology);
                var steer = new LocalhostControl(run.ControlPath);
                run.Bind(steer);
                if (paused)
                {
                    run.Pause();
                }

                var launch = ClusterLaunch.Program(["idrak", .. command], ClusterSimCommand.LocalhostEnvironment(options)) with { Control = steer };
                var localOptions = options with { Simulation = options.Simulation with { Mode = SimulationMode.Localhost } };
                run.Worker = new Thread(() =>
                {
                    try
                    {
                        run.Finish(LocalhostCluster.Run(path, localOptions, launch, null, run.Line, run.Cancel.Token));
                    }
                    catch (Exception e)
                    {
                        run.Finish("failed", e.Message);
                    }
                }) { IsBackground = true, Name = $"dashboard {run.Id}" };
                break;
            default:
                throw new ArgumentException($"mode is in-process or localhost, not '{mode}'.");
        }

        run.Worker!.Start();
        return Json(run.Summary());
    }

    private IResult Steer(string id, string action)
    {
        if (Runs.Find(id) is not { } run)
        {
            return Error(StatusCodes.Status404NotFound, $"No run {id}.");
        }

        if (run.Refusal(action) is { } refusal)
        {
            return Error(StatusCodes.Status409Conflict, refusal);
        }

        long step = action switch
        {
            "pause" => run.Pause(),
            "stop" => run.Stop(),
            _ => Resumed(run),
        };
        var answer = run.Summary();
        answer["at"] = step;
        return Json(answer);

        static long Resumed(DashboardRun run)
        {
            run.Resume();
            return run.Step;
        }
    }

    // POST /api/runs/{id}/faults: {kind, ranks (or rank), duration_ms, factor} queued for the next step.
    private IResult Fault(string id, JsonObject body)
    {
        if (Runs.Find(id) is not { } run)
        {
            return Error(StatusCodes.Status404NotFound, $"No run {id}.");
        }

        if (run.Refusal("faults") is { } refusal)
        {
            return Error(StatusCodes.Status409Conflict, refusal);
        }

        string kindText = (string?)body["kind"] ?? throw new ArgumentException($"Give the fault's kind ({string.Join(", ", Enum.GetNames<FaultKind>())}).");
        string plain = kindText.Replace("-", "", StringComparison.Ordinal).Replace("_", "", StringComparison.Ordinal);
        var kind = Enum.TryParse<FaultKind>(plain, ignoreCase: true, out var parsed) && Enum.IsDefined(parsed)
            ? parsed : throw new ArgumentException($"'{kindText}' is not a fault kind ({string.Join(", ", Enum.GetNames<FaultKind>())}).");
        int[] ranks = body["ranks"] is JsonArray list ? [.. list.Select(r => r!.GetValue<int>())] : body["rank"] is { } one ? [one.GetValue<int>()] : throw new ArgumentException("Give the rank(s) the fault touches (\"ranks\").");
        var duration = TimeSpan.FromMilliseconds(body["duration_ms"]?.GetValue<double>() ?? 0);
        double factor = body["factor"]?.GetValue<double>() ?? kind switch
        {
            FaultKind.SlowWorker => 4,
            FaultKind.LinkLoss => 0.2,
            FaultKind.LateJoin => 1,
            _ => 0,
        };
        var fault = run.Control!.Inject(kind, ranks, duration, factor);
        return Json(new JsonObject
        {
            ["run"] = run.Id, ["kind"] = fault.Kind.ToString(), ["ranks"] = new JsonArray([.. fault.Ranks.Select(r => (JsonNode)r)]), ["duration_ms"] = duration.TotalMilliseconds,
            ["factor"] = factor, ["step"] = run.Control.LastStep + 1, ["seed"] = run.Seed,
            ["detail"] = "queued for the next step the run begins; written to the event log with its step, so idrak cluster replay --from the run's event log replays the run as steered",
        });
    }

    // GET /api/runs/{id}/events: server-sent events, every record kept and then each new one, until the page goes.
    private async Task Events(string id, HttpContext http)
    {
        if (Runs.Find(id) is not { } run)
        {
            await Error(http, StatusCodes.Status404NotFound, $"No run {id}.");
            return;
        }

        http.Response.ContentType = "text/event-stream";
        http.Response.Headers.CacheControl = "no-cache";
        var token = http.RequestAborted;
        var (kept, channel, leave) = Runs.Listen(run);
        using (leave)
        {
            // A localhost or followed run's folder; a reinforce run's JSON log (its iterations, rewards and pass rate, Plan 27 R11).
            var tails = new List<DashboardTail>(2);
            if (run.Kind != RunKind.InProcess && run.Folder is { } folder)
            {
                tails.Add(new DashboardTail(folder));
            }

            if (run.Figures is { } figures)
            {
                tails.Add(figures.Tail());
            }

            try
            {
                foreach (string record in kept)
                {
                    await Send(http, record, token);
                }

                while (!token.IsCancellationRequested)
                {
                    while (channel.Reader.TryRead(out string? record))
                    {
                        await Send(http, record, token);
                    }

                    foreach (var tail in tails)
                    {
                        foreach (string line in tail.Read())
                        {
                            await Send(http, line, token);
                        }
                    }

                    if (tails.Count > 0)
                    {
                        run.PollState();
                    }

                    await http.Response.Body.FlushAsync(token);
                    using var wait = CancellationTokenSource.CreateLinkedTokenSource(token);
                    wait.CancelAfter(Settings.Poll);
                    try
                    {
                        await channel.Reader.WaitToReadAsync(wait.Token);
                    }
                    catch (OperationCanceledException) when (!token.IsCancellationRequested)
                    {
                        // The poll passed: read the folder again.
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // The page went away.
            }
            catch (IOException)
            {
                // The connection closed.
            }
        }
    }

    private static Task Send(HttpContext http, string record, CancellationToken token) => http.Response.WriteAsync("data: " + record + "\n\n", token);

    // A JSON body handed to `answer`; a bad body or an ArgumentException is a 400 naming why.
    private static async Task<IResult> Body(HttpRequest request, Func<JsonObject, IResult> answer)
    {
        JsonObject body;
        try
        {
            body = await JsonNode.ParseAsync(request.Body, cancellationToken: request.HttpContext.RequestAborted) as JsonObject ?? new JsonObject();
        }
        catch (System.Text.Json.JsonException e)
        {
            return Error(StatusCodes.Status400BadRequest, $"The body is not a JSON object: {e.Message}");
        }

        try
        {
            return answer(body);
        }
        catch (Exception e) when (e is ArgumentException or UsageException or InvalidDataException or NotSupportedException or FileNotFoundException or FormatException
                                      or InvalidOperationException)
        {
            return Error(e is InvalidOperationException ? StatusCodes.Status409Conflict : StatusCodes.Status400BadRequest, e.Message);
        }
    }

    private static IResult Json(JsonNode node) => Results.Text(node.ToJsonString(), "application/json; charset=utf-8");

    private static IResult Error(int status, string message) => Results.Text(new JsonObject { ["error"] = message }.ToJsonString(), "application/json; charset=utf-8", statusCode: status);

    private static Task Error(HttpContext http, int status, string message)
    {
        http.Response.StatusCode = status;
        http.Response.ContentType = "application/json; charset=utf-8";
        return http.Response.WriteAsync(new JsonObject { ["error"] = message }.ToJsonString());
    }

    // A command line as words: spaces part them, double quotes keep a word's spaces.
    private static List<string> Split(string text)
    {
        var words = new List<string>();
        var word = new StringBuilder();
        bool quoted = false, any = false;
        foreach (char c in text)
        {
            if (c == '"')
            {
                quoted = !quoted;
                any = true;
            }
            else if (char.IsWhiteSpace(c) && !quoted)
            {
                if (any)
                {
                    words.Add(word.ToString());
                    word.Clear();
                    any = false;
                }
            }
            else
            {
                word.Append(c);
                any = true;
            }
        }

        if (any)
        {
            words.Add(word.ToString());
        }

        return words;
    }
}

/// <summary>Opens the dashboard for a command: <c>cluster ui</c>, and <c>cluster sim --dashboard</c> on its own run.</summary>
internal static class Dashboard
{
    /// <summary>Starts the dashboard with the command's settings, prints its address and opens it unless <c>--no-browser</c>.</summary>
    public static DashboardServer Open(CommandContext context, string? topology)
    {
        var settings = DashboardSettings.Read(context, topology);
        var server = new DashboardServer(settings);
        try
        {
            server.Start();
        }
        catch (IOException e)
        {
            server.Dispose();
            throw new UsageException($"The dashboard cannot listen on {settings.Host}:{settings.Port} ({e.Message}): choose another --port.");
        }

        context.Write($"dashboard {server.PageUrl}" + (server.Loopback ? " (this machine only)"
            : settings.NeedsKey ? " (other machines give an API key)" : " (other machines are refused: no API key is set; idrak server keys add NAME)"));
        if (context.Flag("--no-browser") || !OpenBrowser(server.PageUrl!))
        {
            context.Write($"Open {server.PageUrl} in a browser.");
        }

        return server;
    }

    private static bool OpenBrowser(string url)
    {
        try
        {
            using var _ = Process.Start(new ProcessStartInfo(url) { UseShellExecute = true });
            return true;
        }
        catch (Exception e) when (e is System.ComponentModel.Win32Exception or InvalidOperationException or PlatformNotSupportedException)
        {
            return false;
        }
    }
}

/// <summary>The dashboard's page: one self-contained file (no external address), embedded in the tool.</summary>
internal static class DashboardPage
{
    private static readonly Lazy<string> Page = new(() =>
    {
        using var stream = typeof(DashboardPage).Assembly.GetManifestResourceStream("Dashboard/dashboard.html")
            ?? throw new InvalidOperationException("The dashboard page is not embedded in the tool (Dashboard/dashboard.html).");
        using var reader = new StreamReader(stream, Encoding.UTF8);
        return reader.ReadToEnd();
    });

    /// <summary>The page as written.</summary>
    public static string Html => Page.Value;

    /// <summary>The page in <paramref name="language"/> ("ar": Arabic, right to left).</summary>
    public static string For(string language) => language == "ar"
        ? Html.Replace("<html lang=\"en\" dir=\"ltr\">", "<html lang=\"ar\" dir=\"rtl\">", StringComparison.Ordinal)
        : Html;
}
