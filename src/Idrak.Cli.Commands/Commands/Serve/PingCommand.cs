// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Net.Http.Headers;
using System.Text.Json.Nodes;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Serve;

/// <summary><c>idrak ping [URL]</c>: checks any server that speaks the chat API or the OpenAI-style API.</summary>
internal sealed class PingCommand : Command
{
    public override string Name => "ping";

    public override string Summary => "Check a running server (Idrak or any compatible one): reachable, APIs, models, latency";

    public override string Usage => """
        [URL] [options]

        Asks URL (default: the server started last on this machine, else http://127.0.0.1:7317) for its models over
        the OpenAI-style API (GET /v1/models) and the chat API (GET /api/tags, /api/version), and reports which answer,
        the models and the time each answer took. Exits with 1 when neither API answers.

        Options:
              --api-key KEY  the server's API key (default IDRAK_API_KEY)

        Examples:
          idrak ping
          idrak ping http://192.168.1.20:8080 --json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--api-key"];

    public override int Run(CommandContext context)
    {
        string url = context.Positional.Count > 0 ? context.Positional[0] : ServerClient.BaseUrl(context);
        url = url.Contains("://", StringComparison.Ordinal) ? url.TrimEnd('/') : "http://" + url.TrimEnd('/');
        if (!Uri.TryCreate(url, UriKind.Absolute, out _))
        {
            throw new UsageException($"'{url}' is not a URL (for example http://127.0.0.1:7317).");
        }

        var limit = context.Timeout ?? TimeSpan.FromSeconds(10);                      // per call: --timeout, else 10 s
        using var http = Http.Client(context, limit);
        if ((context.Option("--api-key") ?? Environment.GetEnvironmentVariable("IDRAK_API_KEY")) is { Length: > 0 } key)
        {
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", key);
        }

        (int Status, JsonNode? Body, double Ms, string? Error) Get(string path)
        {
            var watch = Stopwatch.StartNew();
            try
            {
                using var response = http.GetAsync(url + path).GetAwaiter().GetResult();
                return ((int)response.StatusCode, ServerClient.TryParse(response.Content.ReadAsStringAsync().GetAwaiter().GetResult()), watch.Elapsed.TotalMilliseconds, null);
            }
            catch (Exception e) when (e is HttpRequestException or TaskCanceledException)
            {
                return (0, null, watch.Elapsed.TotalMilliseconds, e is TaskCanceledException ? $"no answer within {Units.Duration(limit)}" : e.Message);
            }
        }

        var openai = Get("/v1/models");
        var tags = Get("/api/tags");
        var version = Get("/api/version");
        bool openaiOk = openai.Status == 200 && openai.Body?["data"] is JsonArray;
        bool chatOk = tags.Status == 200 && tags.Body?["models"] is JsonArray;
        string[] openaiModels = openai.Body?["data"] is JsonArray d ? [.. d.Select(m => (string?)m?["id"]).OfType<string>()] : [];
        string[] chatModels = tags.Body?["models"] is JsonArray t ? [.. t.Select(m => (string?)m?["name"]).OfType<string>()] : [];
        bool reachable = openai.Status != 0 || tags.Status != 0;
        string? serverVersion = version.Status == 200 && version.Body?["version"] is JsonValue v ? v.ToString() : null;

        JsonObject Api(bool ok, (int Status, JsonNode? Body, double Ms, string? Error) answer, string[] models) => new()
        {
            ["ok"] = ok, ["status"] = answer.Status, ["ms"] = Math.Round(answer.Ms, 1), ["models"] = new JsonArray([.. models.Select(m => (JsonNode)m)]),
        };
        context.WriteJson(new JsonObject
        {
            ["url"] = url,
            ["reachable"] = reachable,
            ["version"] = serverVersion,
            ["openai_style_api"] = Api(openaiOk, openai, openaiModels),
            ["chat_api"] = Api(chatOk, tags, chatModels),
        });
        if (!reachable)
        {
            context.Error($"idrak ping: no answer from {url} ({openai.Error ?? tags.Error}). Is the server running? Start one with 'idrak serve MODEL'.");
            return ExitCodes.Failed;
        }

        static string Line(bool ok, int status, double ms, string[] models) =>
            ok ? $"ok, {ms:F0} ms, {models.Length} model{(models.Length == 1 ? "" : "s")}{(models.Length > 0 ? ": " + string.Join(", ", models) : "")}"
            : status == 401 ? "needs an API key (--api-key)" : $"not available ({status})";
        context.Write($"{url}{(serverVersion is null ? "" : $" (version {serverVersion})")}");
        context.Write($"  OpenAI-style API  {Line(openaiOk, openai.Status, openai.Ms, openaiModels)}");
        context.Write($"  chat API          {Line(chatOk, tags.Status, tags.Ms, chatModels)}");
        return openaiOk || chatOk ? ExitCodes.Ok : ExitCodes.Failed;
    }
}
