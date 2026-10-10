// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Serve;

/// <summary>
/// How the server commands reach a running server: <c>--host</c>/<c>--port</c> (or IDRAK_PORT) when given, else the
/// most recently started server on this machine (its state file in the cache folder), else 127.0.0.1 on the config's
/// serve.port or the default port.
/// </summary>
internal static class ServerClient
{
    public static readonly string[] ValueOptions = ["--host", "--port", "--api-key"];

    public static readonly Dictionary<string, string> ShortForms = new() { ["-H"] = "--host", ["-p"] = "--port" };

    public const string OptionsHelp = """
          -H, --host NAME    the server's address (default: the server started last on this machine, else 127.0.0.1)
          -p, --port N       the server's port (default: IDRAK_PORT, else as above, else the config's serve.port,
                             else 7317)
              --api-key KEY  the server's API key (default IDRAK_API_KEY)
        """;

    public static string BaseUrl(CommandContext context)
    {
        string? host = context.Option("--host");
        int? port = ServeHost.ExplicitPort(context);
        if (host is null && port is null && ServerState.Find(context) is { } running)
        {
            return running;
        }

        int number = port ?? ServeHost.ConfiguredPort(context) ?? ServeHost.DefaultPort;
        host ??= "127.0.0.1";
        host = host is "0.0.0.0" or "*" ? "127.0.0.1" : host;
        return host.StartsWith("http://", StringComparison.OrdinalIgnoreCase) || host.StartsWith("https://", StringComparison.OrdinalIgnoreCase)
            ? $"{host.TrimEnd('/')}{(port is null ? "" : $":{number}")}" : $"http://{host}:{number}";
    }

    /// <summary>Sends a request; a server that does not answer is a failure that says how to start one.</summary>
    public static HttpResponseMessage Send(CommandContext context, HttpMethod method, string path, string? json, bool stream = false)
    {
        string url = BaseUrl(context);
        // Not disposed here: a streamed answer is read after this returns (the client goes with the process; --timeout closes it).
        var http = Http.Client(context);
        var request = new HttpRequestMessage(method, url + (path.StartsWith('/') ? path : "/" + path));
        if (json is not null)
        {
            request.Content = new StringContent(json, Encoding.UTF8, "application/json");
        }

        if ((context.Option("--api-key") ?? System.Environment.GetEnvironmentVariable("IDRAK_API_KEY")) is { Length: > 0 } key)
        {
            request.Headers.Authorization = new AuthenticationHeaderValue("Bearer", key);
        }

        context.Detail($"{method} {request.RequestUri}");
        try
        {
            return http.SendAsync(request, stream ? HttpCompletionOption.ResponseHeadersRead : HttpCompletionOption.ResponseContentRead).GetAwaiter().GetResult();
        }
        catch (HttpRequestException e)
        {
            throw new InvalidOperationException($"no server answers at {url} ({e.Message}). Start one with 'idrak serve MODEL', or name it with --host and --port.");
        }
    }

    /// <summary>A control call that must succeed; returns its JSON answer.</summary>
    public static JsonNode Call(CommandContext context, HttpMethod method, string path, JsonObject? body = null)
    {
        using var response = Send(context, method, path, body?.ToJsonString());
        string text = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
        if (!response.IsSuccessStatusCode)
        {
            string message = TryParse(text)?["error"] is JsonValue error ? error.ToString() : text;
            throw new InvalidOperationException(response.StatusCode == System.Net.HttpStatusCode.Unauthorized
                ? $"the server needs an API key: {message} Give it with --api-key or IDRAK_API_KEY."
                : $"{(int)response.StatusCode} {response.ReasonPhrase}: {message}");
        }

        return TryParse(text) ?? throw new InvalidOperationException($"the server at {BaseUrl(context)} answered something other than JSON; is it an Idrak server?");
    }

    public static JsonNode? TryParse(string text)
    {
        try
        {
            return JsonNode.Parse(text);
        }
        catch (JsonException)
        {
            return null;
        }
    }
}

/// <summary><c>idrak server ps</c> (<c>idrak ps</c>): the models of a running server.</summary>
internal sealed class ServerPsCommand : Command
{
    public override string Name => "server ps";

    public override IReadOnlyCollection<string> Aliases => ["ps"];

    public override string Summary => "Models of a running server: loaded or not, memory, context, last use";

    public override string Usage => $"""
        [options]

        Options:
        {ServerClient.OptionsHelp}

        Examples:
          idrak ps
          idrak server ps -p 8080 --json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ServerClient.ValueOptions;

    public override IReadOnlyDictionary<string, string> ShortForms => ServerClient.ShortForms;

    public override int Run(CommandContext context)
    {
        var answer = ServerClient.Call(context, HttpMethod.Get, "/idrak/ps");
        context.WriteJson(answer);
        var models = answer["models"]?.AsArray() ?? [];
        context.Write($"Server {ServerClient.BaseUrl(context)} (keep-alive {answer["keep_alive"]})");
        context.Table(["Model", "State", "Device", "Parameters", "Memory", "Context", "Unloads", "Last used", "Requests"], models.Select(m => (IReadOnlyList<string>)
        [
            (string)m!["name"]!,
            (bool)m["loaded"]! ? ((int)m["running"]! > 0 ? "running" : "loaded") : "not loaded",
            (string?)m["device"] ?? "",
            m["parameters"] is { } p ? Count((long)p) : "-",
            (long)m["memory_bytes"]! > 0 ? Units.Bytes((long)m["memory_bytes"]!) : "-",
            m["context_length"]?.ToString() ?? "-",
            (bool)m["loaded"]! ? When((string?)m["expires_at"], future: true) ?? "never" : "-",
            When((string?)m["last_used"], future: false) ?? "never",
            m["requests"]?.ToString() ?? "0",
        ]));
        foreach (var d in answer["devices"]?.AsArray() ?? [])
        {
            context.Write($"{d!["device"]}: {Units.Bytes((long)d["memory_in_use"]!)} in use" + (d["memory_limit"] is { } limit ? $" of {Units.Bytes((long)limit)}" : ""));
        }

        return ExitCodes.Ok;
    }


    private static string Count(long n) => n >= 1_000_000_000 ? string.Create(CultureInfo.InvariantCulture, $"{n / 1e9:0.#}B")
        : n >= 1_000_000 ? string.Create(CultureInfo.InvariantCulture, $"{n / 1e6:0.#}M")
        : n >= 1_000 ? string.Create(CultureInfo.InvariantCulture, $"{n / 1e3:0.#}K") : n.ToString(CultureInfo.InvariantCulture);

    private static string? When(string? iso, bool future)
    {
        if (iso is null)
        {
            return null;
        }

        var span = DateTimeOffset.Parse(iso, CultureInfo.InvariantCulture) - DateTimeOffset.UtcNow;
        span = future ? span : -span;
        string text = span.TotalSeconds < 60 ? string.Create(CultureInfo.InvariantCulture, $"{Math.Max(0, span.TotalSeconds):0} s")
            : span.TotalMinutes < 60 ? string.Create(CultureInfo.InvariantCulture, $"{span.TotalMinutes:0} min")
            : string.Create(CultureInfo.InvariantCulture, $"{span.TotalHours:0.#} h");
        return future ? "in " + text : text + " ago";
    }
}

/// <summary><c>idrak server stop</c>: stops a running server (open requests finish first).</summary>
internal sealed class ServerStopCommand : Command
{
    public override string Name => "server stop";

    public override string Summary => "Stop a running server; open requests finish first";

    public override string Usage => $"""
        [options]

        Open requests finish first; --timeout bounds how long the call waits.

        Options:
        {ServerClient.OptionsHelp}

        Examples:
          idrak server stop
          idrak server stop -p 8080
        """;

    public override IReadOnlyCollection<string> ValueOptions => ServerClient.ValueOptions;

    public override IReadOnlyDictionary<string, string> ShortForms => ServerClient.ShortForms;

    public override int Run(CommandContext context)
    {
        string url = ServerClient.BaseUrl(context);
        var answer = ServerClient.Call(context, HttpMethod.Post, "/idrak/stop", []);
        context.Write($"Stopping the server at {url}");
        context.WriteJson(new JsonObject { ["url"] = url, ["stopping"] = answer["stopping"]?.DeepClone() });
        return ExitCodes.Ok;
    }
}

/// <summary>Shared by server load and server unload.</summary>
internal abstract class ServerModelCommand(bool load) : Command
{
    public override string Usage => $"""
        MODEL [options]

        MODEL is a name the server serves (see 'idrak ps').

        Options:
        {ServerClient.OptionsHelp}

        Examples:
          idrak server {(load ? "load" : "unload")} qwen
          idrak server {(load ? "load" : "unload")} qwen -p 8080 --json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ServerClient.ValueOptions;

    public override IReadOnlyDictionary<string, string> ShortForms => ServerClient.ShortForms;

    public override int Run(CommandContext context)
    {
        string model = context.Argument(0, "MODEL (a name the server serves; see 'idrak ps')");
        var answer = ServerClient.Call(context, HttpMethod.Post, load ? "/idrak/load" : "/idrak/unload", new JsonObject { ["model"] = model });
        context.Write(load ? $"{answer["model"]} is loaded ({answer["seconds"]} s)" : $"{answer["model"]} is unloaded");
        context.WriteJson(answer);
        return ExitCodes.Ok;
    }
}

/// <summary><c>idrak server load MODEL</c>.</summary>
internal sealed class ServerLoadCommand() : ServerModelCommand(load: true)
{
    public override string Name => "server load";

    public override string Summary => "Load a model in a running server now, instead of on its first request";
}

/// <summary><c>idrak server unload MODEL</c>.</summary>
internal sealed class ServerUnloadCommand() : ServerModelCommand(load: false)
{
    public override string Name => "server unload";

    public override string Summary => "Unload a model from a running server, freeing its memory";
}

/// <summary><c>idrak api PATH [JSON]</c>: calls an endpoint of a running server.</summary>
internal sealed class ApiCommand : Command
{
    public override string Name => "api";

    public override string Summary => "Call an endpoint of a running server (for scripts and checks)";

    public override string Usage => $$"""
        PATH [JSON] [options]

        GET PATH, or POST JSON to it ('-' reads the JSON from standard input). The answer is printed as it arrives
        (streamed answers line by line); an error status exits with 1. --timeout bounds the whole call, a streamed
        answer included.

        Options:
              --method NAME  GET, POST, DELETE, ... (default GET, or POST with JSON)
        {{ServerClient.OptionsHelp}}

        Examples:
          idrak api /api/tags
          idrak api /v1/chat/completions '{"model":"qwen","messages":[{"role":"user","content":"Hi"}]}'
          idrak api /api/chat '{"model":"qwen","messages":[{"role":"user","content":"Hi"}]}' -p 8080
        """;

    public override IReadOnlyCollection<string> ValueOptions => [.. ServerClient.ValueOptions, "--method"];

    public override IReadOnlyDictionary<string, string> ShortForms => ServerClient.ShortForms;

    public override int Run(CommandContext context)
    {
        string path = context.Argument(0, "PATH (for example /api/tags)");
        string? json = context.Positional.Count > 1 ? context.Positional[1] : null;
        if (context.Positional.Count > 2)
        {
            throw new UsageException("Give the JSON body as one argument (quote it).");
        }

        if (json == "-")
        {
            json = Console.In.ReadToEnd();
        }

        if (json is not null && ServerClient.TryParse(json) is null)
        {
            throw new UsageException("The body is not valid JSON.");
        }

        var method = new HttpMethod((context.Option("--method") ?? (json is null ? "GET" : "POST")).ToUpperInvariant());
        using var response = ServerClient.Send(context, method, path, json, stream: true);
        using var reader = new StreamReader(response.Content.ReadAsStream());
        var target = response.IsSuccessStatusCode ? context.Output : context.ErrorOutput;
        if (!response.IsSuccessStatusCode)
        {
            context.Error($"{(int)response.StatusCode} {response.ReasonPhrase}");
        }

        if (context.Quiet && response.IsSuccessStatusCode)
        {
            reader.ReadToEnd();
            return ExitCodes.Ok;
        }

        while (reader.ReadLine() is { } line)
        {
            target.WriteLine(line);
            target.Flush();
        }

        return response.IsSuccessStatusCode ? ExitCodes.Ok : ExitCodes.Failed;
    }
}
