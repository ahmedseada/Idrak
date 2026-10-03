// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Pipelines;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Protocol;
using Idrak;
using Idrak.AspNetCore;
using Idrak.Cli;
using Idrak.Cli.Commands.Serve;
using Idrak.Generation;
using Idrak.Mcp;
using Idrak.Retrieval;

/// <summary>Tools served by the 'idrak mcp serve' test (the test assembly is the tools assembly).</summary>
public sealed class CliServeTestTools
{
    /// <summary>Adds two integers.</summary>
    [Tool("serve_add", "Adds two integers.")]
    public int Add(int a, int b) => a + b;

    /// <summary>Repeats a text.</summary>
    [Tool("serve_echo", "Repeats the text.")]
    public static string Echo(string text) => text;
}

// The Serve group of the idrak CLI: serve (both APIs, streaming and not), ps, api, ping, server load/unload/stop/keys,
// metrics and the request log against a server started in-process on a free port with a tiny GGUF model; mcp serve
// over pipes; and the OpenAI-style API of Idrak.AspNetCore with a scripted chat model (tool calls).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliServeGroup =
    [
        ("cli serve: help, short forms, usage errors, server keys", d => { if (d == Device.Cpu) CliServeHelpAndKeys(); }),
        ("cli serve: the port (--port, IDRAK_PORT, config serve.port, default 7317) for serve and the client commands", d => { if (d == Device.Cpu) CliServePort(); }),
        ("cli serve: serve a tiny model, both APIs (streaming and not), ps, api, ping, load/unload, metrics, request log, stop", CliServeEndToEnd),
        ("cli serve: mcp serve over pipes, --list", d => { if (d == Device.Cpu) CliMcpServe(); }),
        ("cli serve: OpenAI-style API in Idrak.AspNetCore (tool calls streamed and not, translation, embeddings)", d => { if (d == Device.Cpu) CompletionsApiScripted(); }),
    ];

    private static (int Code, string Output, string Error) ServeCli(params string[] args)
    {
        var output = new StringWriter();
        var error = new StringWriter();
        int code = CommandLine.Run(args, output, error);
        return (code, output.ToString(), error.ToString());
    }

    private static void CliServeHelpAndKeys()
    {
        string[] names = ["serve", "s", "ui", "server ps", "ps", "server stop", "server load", "server unload", "server keys add", "server keys list",
            "server keys rm", "api", "ping", "mcp serve"];
        foreach (string name in names)
        {
            var (code, text, _) = ServeCli([.. name.Split(' '), "--help"]);
            Check(code == 0 && text.Contains("Examples:") && text.Contains("Environment (idrak help env for all):"), $"help of {name}: {text}");
            Check(!System.Text.RegularExpressions.Regex.Replace(text, @"\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b", "").Contains("ollama", StringComparison.OrdinalIgnoreCase), $"no provider name in the help of {name}");
        }

        foreach (var command in Idrak.Cli.Commands.ServeCommands.All)
        {
            var shorts = command.ShortForms.Keys.ToList();
            Check(shorts.Distinct().Count() == shorts.Count && !shorts.Any(CommandContext.CommonShortForms.ContainsKey), $"{command.Name}: short forms unique and not common ones");
            Check(command.ShortForms.Values.All(v => command.ValueOptions.Contains(v) || command.Flags.Contains(v)), $"{command.Name}: every short form names an option");
        }

        Check(ServeCli("serve").Code == 2, "serve without a model is a usage error");
        Check(ServeCli("serve", TestData("gguf/tiny-qwen3-q8.gguf"), "--keep-alive", "soon").Code == 2, "a bad --keep-alive is a usage error");
        Check(ServeCli("serve", TestData("gguf/tiny-qwen3-q8.gguf"), "--log-content").Code == 2, "--log-content needs --log-requests");
        Check(ServeCli("serve", "a=" + TestData("gguf/tiny-qwen3-q8.gguf"), "a=" + TestData("gguf/tiny-llama.gguf"), "--cache", TempFolder()).Code == 2, "two models with one name");
        Check(ServeCli("api").Code == 2 && ServeCli("api", "/x", "{not json").Code == 2, "api needs a path and JSON");

        string config = Path.Combine(TempFolder(), "config.json");
        var added = ServeCli("server", "keys", "add", "laptop", "-C", config, "--json");
        var key = JsonNode.Parse(added.Output)!;
        Check(added.Code == 0 && ((string)key["key"]!).StartsWith("idrak-", StringComparison.Ordinal), $"keys add: {added.Output}{added.Error}");
        Check(!File.ReadAllText(config).Contains((string)key["key"]!) && File.ReadAllText(config).Contains(ServerKeys.Hash((string)key["key"]!)), "only the hash is stored");
        Check(ServeCli("server", "keys", "add", "laptop", "-C", config).Code == 2, "a key name is used once");
        var list = ServeCli("server", "keys", "list", "-C", config, "-j");
        Check(list.Code == 0 && (string?)JsonNode.Parse(list.Output)!["keys"]![0]!["name"] == "laptop" && !list.Output.Contains("sha256"), $"keys list: {list.Output}");
        Check(ServeCli("server", "keys", "rm", "laptop", "-C", config).Code == 0 && ServeCli("server", "keys", "rm", "laptop", "-C", config).Code == 1, "keys rm");
    }

    private static void CliServeEndToEnd(Device device)
    {
        string cache = TempFolder(), folder = TempFolder();
        string config = Path.Combine(folder, "config.json"), log = Path.Combine(folder, "requests.jsonl");
        Check(ServeCli("server", "keys", "add", "ci", "--key", "ci-key-12345", "-C", config).Code == 0, "keys add --key");
        int port;
        using (var probe = new TcpListener(IPAddress.Loopback, 0))
        {
            probe.Start();
            port = ((IPEndPoint)probe.LocalEndpoint).Port;
        }

        string url = $"http://127.0.0.1:{port}";
        var serveOutput = new StringWriter();
        var serveError = new StringWriter();
        var server = Task.Run(() => CommandLine.Run(["s", "tiny=" + TestData("gguf/tiny-qwen3-q8.gguf"), "-p", port.ToString(), "-d", device.ToString(),
            "--cache", cache, "-C", config, "--api-key", "secret", "--metrics", "--log-requests", log, "--max-concurrency", "2", "--keep-alive", "10m"],
            TextWriter.Synchronized(serveOutput), TextWriter.Synchronized(serveError)));
        using var http = new HttpClient { BaseAddress = new Uri(url), Timeout = TimeSpan.FromMinutes(2) };
        var deadline = DateTime.UtcNow.AddSeconds(60);
        while (true)
        {
            Check(!server.IsCompleted, $"the server stopped early: {serveOutput}{serveError}");
            try
            {
                if (http.GetAsync("/").Result.IsSuccessStatusCode)
                {
                    break;
                }
            }
            catch (AggregateException)
            {
            }

            Check(DateTime.UtcNow < deadline, "the server answers within a minute");
            Thread.Sleep(100);
        }

        try
        {
            Check(serveOutput.ToString().Contains($"{url}/v1") && !serveOutput.ToString().Contains("ollama", StringComparison.OrdinalIgnoreCase), $"announcement: {serveOutput}");
            Check(http.GetAsync("/v1/models").Result.StatusCode == HttpStatusCode.Unauthorized, "a key is required");
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "ci-key-12345");
            Check(http.GetStringAsync("/v1/models").Result.Contains("\"tiny\""), "a stored key works; /v1/models");
            http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", "secret");

            JsonNode Post(string path, string body) =>
                JsonNode.Parse(http.PostAsync(path, new StringContent(body, Encoding.UTF8, "application/json")).Result.Content.ReadAsStringAsync().Result)!;
            string PostText(string path, string body) => http.PostAsync(path, new StringContent(body, Encoding.UTF8, "application/json")).Result.Content.ReadAsStringAsync().Result;

            // OpenAI-style API
            var chat = Post("/v1/chat/completions", """{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"seed":1}""");
            Check((string?)chat["object"] == "chat.completion" && (string?)chat["choices"]![0]!["message"]!["role"] == "assistant"
                && (int)chat["usage"]!["completion_tokens"]! is > 0 and <= 4, $"chat completion: {chat}");
            string sse = PostText("/v1/chat/completions", """{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":4,"seed":1,"stream":true,"stream_options":{"include_usage":true}}""");
            var events = sse.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
            Check(events.All(e => e.StartsWith("data: ", StringComparison.Ordinal)) && events[^1] == "data: [DONE]", $"SSE framing: {sse}");
            var parsed = events[..^1].Select(e => JsonNode.Parse(e["data: ".Length..])!).ToList();
            Check((string?)parsed[0]["choices"]![0]!["delta"]!["role"] == "assistant" && parsed.Any(p => p["choices"]!.AsArray().Count > 0 && p["choices"]![0]!["finish_reason"] is not null)
                && parsed[^1]["usage"] is not null, $"streamed chunks: {sse}");
            var withTools = Post("/v1/chat/completions", """
                {"model":"tiny","max_tokens":3,"messages":[{"role":"user","content":"weather?"},
                 {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Cairo\"}"}}]},
                 {"role":"tool","tool_call_id":"call_1","content":"sunny"}],
                 "tools":[{"type":"function","function":{"name":"weather","description":"Weather of a city","parameters":{"type":"object","properties":{"city":{"type":"string"}}}}}]}
                """);
            Check(withTools["choices"] is JsonArray, $"a request with tools and tool results: {withTools}");
            var completion = Post("/v1/completions", """{"model":"tiny","prompt":"ab","max_tokens":3}""");
            Check((string?)completion["object"] == "text_completion" && completion["choices"]![0]!["text"] is not null, $"completion: {completion}");
            string completionStream = PostText("/v1/completions", """{"model":"tiny","prompt":"ab","max_tokens":3,"stream":true}""");
            Check(completionStream.EndsWith("data: [DONE]\n\n", StringComparison.Ordinal) && completionStream.Contains("text_completion"), "streamed completion");
            Check(http.PostAsync("/v1/embeddings", new StringContent("""{"input":"x"}""")).Result.StatusCode == HttpStatusCode.NotFound, "no embedding model: 404");
            Check(http.PostAsync("/v1/chat/completions", new StringContent("{\"messages\":")).Result.StatusCode == HttpStatusCode.BadRequest, "bad JSON: 400");

            // chat API
            var single = Post("/api/chat", """{"model":"tiny:latest","messages":[{"role":"user","content":"hi"}],"stream":false,"options":{"num_predict":4}}""");
            Check((bool)single["done"]! && (string?)single["message"]!["role"] == "assistant" && (string?)single["model"] == "tiny:latest", $"chat API, not streamed: {single}");
            var lines = PostText("/api/chat", """{"model":"tiny","messages":[{"role":"user","content":"hi"}],"options":{"num_predict":4}}""")
                .Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => JsonNode.Parse(l)!).ToList();
            Check(lines.Count >= 1 && (bool)lines[^1]["done"]! && (int)lines[^1]["eval_count"]! > 0, "chat API, streamed lines");
            Check((string?)JsonNode.Parse(http.GetStringAsync("/api/tags").Result)!["models"]![0]!["name"] == "tiny", "tags");
            Check((string?)JsonNode.Parse(http.GetStringAsync("/api/ps").Result)!["models"]![0]!["name"] == "tiny", "ps: loaded");
            Check(JsonNode.Parse(http.GetStringAsync("/api/version").Result)!["version"] is not null, "version");
            Check(http.GetStringAsync("/ui").Result.Contains("Idrak chat"), "the web chat page");
            string metrics = http.GetStringAsync("/metrics").Result;
            Check(metrics.Contains("# TYPE idrak_requests_total counter") && metrics.Contains("idrak_model_loaded{model=\"tiny\"} 1")
                && metrics.Contains("idrak_generated_tokens_total{model=\"tiny\"}"), $"metrics: {metrics}");

            // the commands against the running server (found through its state file in the cache)
            var ps = ServeCli("ps", "--cache", cache, "--api-key", "secret", "--json");
            var model = JsonNode.Parse(ps.Output)!["models"]![0]!;
            Check(ps.Code == 0 && (string?)model["name"] == "tiny" && (bool)model["loaded"]! && (long)model["requests"]! >= 6 && model["last_used"] is not null, $"ps: {ps.Output}{ps.Error}");
            var psText = ServeCli("server", "ps", "-p", port.ToString(), "--api-key", "secret");
            Check(psText.Code == 0 && psText.Output.Contains("tiny") && psText.Output.Contains("loaded"), $"ps text: {psText.Output}{psText.Error}");
            Check(ServeCli("ps", "-p", port.ToString()).Code == 1, "ps without the key fails");
            var api = ServeCli("api", "/api/tags", "--cache", cache, "--api-key", "secret");
            Check(api.Code == 0 && api.Output.Contains("\"tiny\""), $"api GET: {api.Output}{api.Error}");
            var apiPost = ServeCli("api", "/v1/chat/completions", """{"model":"tiny","messages":[{"role":"user","content":"hi"}],"max_tokens":2}""", "-p", port.ToString(), "--api-key", "secret");
            Check(apiPost.Code == 0 && apiPost.Output.Contains("chat.completion"), $"api POST: {apiPost.Output}{apiPost.Error}");
            Check(ServeCli("api", "/nothing-here", "-p", port.ToString(), "--api-key", "secret").Code == 1, "api: an error status exits with 1");
            var ping = ServeCli("ping", url, "--api-key", "secret", "--json");
            var pinged = JsonNode.Parse(ping.Output)!;
            Check(ping.Code == 0 && (bool)pinged["openai_style_api"]!["ok"]! && (bool)pinged["chat_api"]!["ok"]! && (string?)pinged["chat_api"]!["models"]![0] == "tiny", $"ping: {ping.Output}");
            var unload = ServeCli("server", "unload", "tiny", "-p", port.ToString(), "--api-key", "secret", "-j");
            Check(unload.Code == 0 && !(bool)JsonNode.Parse(ServeCli("ps", "-p", port.ToString(), "--api-key", "secret", "-j").Output)!["models"]![0]!["loaded"]!, $"unload: {unload.Output}{unload.Error}");
            var load = ServeCli("server", "load", "tiny", "-p", port.ToString(), "--api-key", "secret");
            Check(load.Code == 0 && (bool)JsonNode.Parse(ServeCli("ps", "-p", port.ToString(), "--api-key", "secret", "-j").Output)!["models"]![0]!["loaded"]!, $"load: {load.Output}{load.Error}");
            Check(ServeCli("server", "load", "nope", "-p", port.ToString(), "--api-key", "secret").Code == 1, "loading an unknown model fails");

            var logged = File.ReadAllLines(log).Select(l => JsonNode.Parse(l)!).ToList();
            Check(logged.Count >= 7 && logged.All(l => l["request"] is null) && logged.Any(l => (string?)l["path"] == "/api/chat" && (int?)l["completion_tokens"] > 0)
                && logged.Any(l => (string?)l["path"] == "/v1/chat/completions" && (string?)l["model"] == "tiny" && (int?)l["completion_tokens"] > 0), $"request log: {string.Join('\n', logged)}");
        }
        finally
        {
            var stop = ServeCli("server", "stop", "--cache", cache, "--api-key", "secret");
            Check(stop.Code == 0, $"server stop: {stop.Output}{stop.Error}");
            Check(server.Wait(TimeSpan.FromSeconds(60)) && server.Result == 0, $"serve exits with 0 after stop: {serveOutput}{serveError}");
        }

        Check(!Directory.EnumerateFiles(Path.Combine(cache, "servers")).Any(), "the state file is removed");
        Check(ServeCli("ping", url).Code == 1, "ping fails once the server is gone");
    }

    // The port: --port, else IDRAK_PORT, else the config's serve.port, else Idrak's own default; the client commands put
    // the server started last on this machine between the environment and the config.
    private static void CliServePort()
    {
        string folder = TempFolder(), cache = Path.Combine(folder, "cache"), config = Path.Combine(folder, "config.json");
        var serve = CommandTable.All.First(c => c.Name == "serve");
        var ps = CommandTable.All.First(c => c.Name == "server ps");
        string? old = Environment.GetEnvironmentVariable(ServeHost.PortVariable);

        (int Port, string Url) Chosen(params string[] port)
        {
            var options = new Dictionary<string, List<string>> { ["--config"] = [config], ["--cache"] = [cache] };
            if (port.Length > 0)
            {
                options["--port"] = [.. port];
            }

            using var forServe = new CommandContext(serve, [], options, [], new StringWriter(), new StringWriter());
            using var forClient = new CommandContext(ps, [], options, [], new StringWriter(), new StringWriter());
            return (ServeHost.ReadSettings(forServe).Port, ServerClient.BaseUrl(forClient));
        }

        bool Refused(Action action)
        {
            try
            {
                action();
                return false;
            }
            catch (UsageException)
            {
                return true;
            }
        }

        try
        {
            Environment.SetEnvironmentVariable(ServeHost.PortVariable, null);
            Check(ServeHost.DefaultPort == 7317 && Chosen() == (7317, "http://127.0.0.1:7317"), $"the default port: {Chosen()}");
            Check(ServeCli("config", "set", "serve.port", "8123", "-C", config).Code == 0 && Chosen() == (8123, "http://127.0.0.1:8123"), $"config serve.port: {Chosen()}");
            Environment.SetEnvironmentVariable(ServeHost.PortVariable, "8124");
            Check(Chosen() == (8124, "http://127.0.0.1:8124"), $"IDRAK_PORT before the config: {Chosen()}");
            Check(Chosen("8125") == (8125, "http://127.0.0.1:8125") && Chosen("0").Port == 0, $"--port before IDRAK_PORT: {Chosen("8125")}");

            // A running server's state file (this process stands in for it): found by the clients unless the port is given.
            Directory.CreateDirectory(Path.Combine(cache, "servers"));
            File.WriteAllText(Path.Combine(cache, "servers", "9301.json"), new JsonObject
            {
                ["url"] = "http://127.0.0.1:9301", ["pid"] = Environment.ProcessId, ["started"] = DateTimeOffset.UtcNow.ToString("O"), ["models"] = new JsonArray(),
            }.ToJsonString());
            Check(Chosen().Url == "http://127.0.0.1:8124", "IDRAK_PORT before the running server");
            Environment.SetEnvironmentVariable(ServeHost.PortVariable, null);
            Check(Chosen() == (8123, "http://127.0.0.1:9301"), $"the running server before the config for clients, the config for serve: {Chosen()}");

            Environment.SetEnvironmentVariable(ServeHost.PortVariable, "port");
            Check(Refused(() => Chosen()), "IDRAK_PORT that is not a number is a usage error");
            Environment.SetEnvironmentVariable(ServeHost.PortVariable, "70000");
            Check(Refused(() => Chosen()), "IDRAK_PORT out of range is a usage error");
            Environment.SetEnvironmentVariable(ServeHost.PortVariable, null);
            Check(ServeCli("config", "set", "serve.port", "\"8126\"", "-C", config).Code == 0 && Chosen().Port == 8126, "serve.port as text");
            Check(ServeCli("config", "set", "serve.port", "70000", "-C", config).Code == 0 && Refused(() => Chosen()), "serve.port out of range is a usage error");
            Check(ServeCli("serve", TestData("gguf/tiny-qwen3-q8.gguf"), "-p", "65536").Code == 2, "--port out of range is a usage error");
        }
        finally
        {
            Environment.SetEnvironmentVariable(ServeHost.PortVariable, old);
            Directory.Delete(folder, recursive: true);
        }

        string help = ServeCli("serve", "--help").Output;
        Check(help.Contains("7317", StringComparison.Ordinal) && help.Contains("-p 11434", StringComparison.Ordinal) && help.Contains(ServeHost.PortVariable, StringComparison.Ordinal)
              && help.Contains("serve.port", StringComparison.Ordinal), $"serve's help names the default, the variable, the config key and -p 11434: {help}");
        Check(Idrak.Cli.Shared.EnvironmentVariables.Find(ServeHost.PortVariable) is { Group: "The tool" } variable && variable.Default.Contains("7317", StringComparison.Ordinal)
              && variable.Commands.Contains("serve") && variable.Commands.Contains("ping"), "IDRAK_PORT in the environment table, with its default");
        foreach (string name in new[] { "ui", "api", "ping", "server ps", "server stop" })
        {
            Check(Help.For(CommandTable.All.First(c => c.Name == name)).Contains(ServeHost.PortVariable, StringComparison.Ordinal), $"{name}'s help names {ServeHost.PortVariable}");
        }
    }

    private static void CliMcpServe()
    {
        string tools = typeof(CliServeTestTools).Assembly.Location;
        var list = ServeCli("mcp", "serve", tools, "--list", "--json");
        var names = JsonNode.Parse(list.Output)!["tools"]!.AsArray().Select(t => (string)t!["name"]!)
            .Where(n => n.StartsWith("serve_", StringComparison.Ordinal)).Order().ToList();          // the test assembly holds other groups' tools too
        Check(list.Code == 0 && names.SequenceEqual(["serve_add", "serve_echo"]), $"mcp serve --list: {list.Output}{list.Error}");
        Check(ServeCli("mcp", "serve").Code == 2 && ServeCli("mcp", "serve", "missing.dll").Code == 2, "mcp serve usage errors");

        var toServer = new Pipe();
        var toClient = new Pipe();
        var previous = McpServeCommand.Streams;
        McpServeCommand.Streams = () => (toServer.Reader.AsStream(), toClient.Writer.AsStream());
        try
        {
            var error = new StringWriter();
            var serving = Task.Run(() => CommandLine.Run(["mcp", "serve", tools], new StringWriter(), error));
            Task.Run(async () =>
            {
                await using var source = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream()));
                Check(source.ServerName == "idrak", $"server name {source.ServerName}");
                var listed = await source.ListToolsAsync();
                Check(listed.Select(t => t.Definition.Name).Where(n => n.StartsWith("serve_", StringComparison.Ordinal)).Order().SequenceEqual(["serve_add", "serve_echo"]), "tools over MCP");
                Check(await source.CallAsync("serve_add", new JsonObject { ["a"] = 2, ["b"] = 40 }) == "42", "a call over MCP");
                Check(await source.CallAsync("serve_echo", new JsonObject { ["text"] = "hi" }) == "hi", "a static tool over MCP");
            }).GetAwaiter().GetResult();
            toServer.Writer.Complete();
            Check(serving.Wait(TimeSpan.FromSeconds(30)) && serving.Result == 0 && error.ToString().Contains("tools over MCP") && error.ToString().Contains("serve_add"), $"mcp serve ends with its input: {error}");
        }
        finally
        {
            McpServeCommand.Streams = previous;
        }
    }

    private sealed class FixedEmbedder : IEmbedder
    {
        public ValueTask<float[][]> EmbedAsync(IReadOnlyList<string> texts, CancellationToken cancellationToken = default) =>
            ValueTask.FromResult(texts.Select(t => new[] { t.Length, 1f }).ToArray());
    }

    private static void CompletionsApiScripted()
    {
        var translated = CompletionsTranslation.Chat(JsonNode.Parse("""
            {"messages":[{"role":"developer","content":[{"type":"text","text":"Be brief."}]},{"role":"user","content":"weather?"},
             {"role":"assistant","content":null,"tool_calls":[{"id":"call_9","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Cairo\"}"}}]},
             {"role":"tool","tool_call_id":"call_9","content":"sunny"}],
             "tools":[{"type":"function","function":{"name":"weather","parameters":{"type":"object"}}}],
             "temperature":0.2,"max_tokens":7,"stop":"END","seed":3,"reasoning_effort":"none"}
            """)!.AsObject());
        Check(translated.Messages[0] is { Role: "system", Content: "Be brief." } && translated.Messages[2].ToolCalls![0].Arguments["city"]!.ToString() == "Cairo"
            && translated.Messages[3].ToolName == "weather" && translated.Tools!.Single().Name == "weather" && translated.Think == false
            && translated.Options!.NumPredict == 7 && translated.Options.Stop.SequenceEqual(["END"]) && translated.Options.Seed == 3 && Math.Abs(translated.Options.Temperature - 0.2f) < 1e-6,
            "translation of an OpenAI-style request");
        bool threw = false;
        try { CompletionsTranslation.Chat(new JsonObject { ["messages"] = new JsonArray() }); } catch (ArgumentException) { threw = true; }
        Check(threw, "empty messages are rejected");

        var call = FakeChatModel.ToolCall("weather", new JsonObject { ["city"] = "Cairo" });
        var scripted = FakeChatModel.Script(call, call, FakeChatModel.Answer("It is sunny.", "checked"));
        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        var app = builder.Build();
        app.MapCompletionsApi("/v1", o => o.ChatModel("scripted", scripted).Embeddings("words", new FixedEmbedder()));
        app.StartAsync().GetAwaiter().GetResult();
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri(app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First()) };
            string Post(string path, string body) => http.PostAsync(path, new StringContent(body)).Result.Content.ReadAsStringAsync().Result;
            const string Ask = """{"model":"any","messages":[{"role":"user","content":"weather?"}],"tools":[{"type":"function","function":{"name":"weather"}}]""";
            var reply = JsonNode.Parse(Post("/v1/chat/completions", Ask + "}"))!["choices"]![0]!;
            var toolCall = reply["message"]!["tool_calls"]![0]!;
            Check((string?)reply["finish_reason"] == "tool_calls" && (string?)toolCall["function"]!["name"] == "weather"
                && JsonNode.Parse((string)toolCall["function"]!["arguments"]!)!["city"]!.ToString() == "Cairo" && ((string)toolCall["id"]!).StartsWith("call_", StringComparison.Ordinal),
                $"tool call, not streamed: {reply}");
            string sse = Post("/v1/chat/completions", Ask + ""","stream":true}""");
            var chunks = sse.Split("\n\n", StringSplitOptions.RemoveEmptyEntries).Where(e => e != "data: [DONE]").Select(e => JsonNode.Parse(e["data: ".Length..])!["choices"]![0]!).ToList();
            Check(chunks.Any(c => (string?)c["delta"]!["tool_calls"]?[0]?["function"]?["name"] == "weather" && (int?)c["delta"]!["tool_calls"]![0]!["index"] == 0)
                && (string?)chunks[^1]["finish_reason"] == "tool_calls" && sse.EndsWith("data: [DONE]\n\n", StringComparison.Ordinal), $"tool call, streamed: {sse}");
            var answer = JsonNode.Parse(Post("/v1/chat/completions", Ask + "}"))!["choices"]![0]!;
            Check((string?)answer["message"]!["content"] == "It is sunny." && (string?)answer["message"]!["reasoning_content"] == "checked" && (string?)answer["finish_reason"] == "stop",
                $"answer with reasoning: {answer}");
            Check(scripted.Requests[^1].Tools!.Single().Name == "weather", "the tools reach the model");
            var models = JsonNode.Parse(http.GetStringAsync("/v1/models").Result)!["data"]!.AsArray().Select(m => (string)m!["id"]!).Order().ToList();
            Check(models.SequenceEqual(["scripted", "words"]), $"models {string.Join(",", models)}");
            var embedded = JsonNode.Parse(Post("/v1/embeddings", """{"model":"words","input":["abc","de"]}"""))!;
            Check((float)embedded["data"]![1]!["embedding"]![0]! == 2f && (int)embedded["data"]![1]!["index"]! == 1, $"embeddings: {embedded}");
            Check(http.PostAsync("/v1/completions", new StringContent("""{"model":"scripted","prompt":"x"}""")).Result.StatusCode == HttpStatusCode.NotFound, "a chat-only model has no raw completions");
            Check(JsonNode.Parse(Post("/v1/chat/completions", """{"messages":[]}"""))!["error"]!["type"]!.ToString() == "invalid_request_error", "error shape");
        }
        finally
        {
            app.StopAsync().GetAwaiter().GetResult();
            ((IAsyncDisposable)app).DisposeAsync().AsTask().GetAwaiter().GetResult();
        }
    }
}
