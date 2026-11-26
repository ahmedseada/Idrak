// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.IO.Pipelines;
using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.AspNetCore;
using Idrak.Cli;
using Idrak.Cli.Commands.Serve;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Mcp;
using Idrak.Models;
using Idrak.Text.Inference;
using Idrak.Text.Plugins.T5;
using Idrak.Text.Generation;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using ModelContextProtocol.Protocol;

[assembly: ToolProvider(typeof(CliServeTestTools))]

/// <summary>
/// Tools served by the 'idrak mcp serve' test (the test assembly is the tools assembly, declaring them with ToolProvider):
/// the [Tool] methods and a tool of its own type (any ITool).
/// </summary>
public sealed class CliServeTestTools : IToolProvider
{
    /// <inheritdoc/>
    public IReadOnlyList<Idrak.Abstraction.Generation.ITool> Tools => [.. ToolProviders.FromMethods(this), new UpperTool()];

    // A tool that is not a Tool record: what a provider may offer beside its methods.
    private sealed class UpperTool : Idrak.Abstraction.Generation.ITool
    {
        public ToolDefinition Definition { get; } = new("serve_upper", "Upper-cases the text.",
            new JsonObject { ["type"] = "object", ["properties"] = new JsonObject { ["text"] = new JsonObject { ["type"] = "string" } }, ["required"] = new JsonArray("text") });

        public Task<string> InvokeAsync(JsonObject arguments, CancellationToken cancellationToken) =>
            Task.FromResult(((string?)arguments["text"] ?? "").ToUpperInvariant());
    }

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
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] CliServeGroup =
    [
        ("cli serve: help, short forms, usage errors, server keys", d => { CpuOnly(d); CliServeHelpAndKeys(); }),
        ("cli serve: the port (--port, IDRAK_PORT, config serve.port, default 7317) for serve and the client commands", d => { CpuOnly(d); CliServePort(); }),
        ("cli serve: mcp serve over pipes, --list", d => { CpuOnly(d); CliMcpServe(); }),
        ("cli serve: OpenAI-style API in Idrak.AspNetCore (tool calls streamed and not, translation, embeddings)", d => { CpuOnly(d); CompletionsApiScripted(); }),
        ("cli serve: the doors: an empty --api-key or IDRAK_API_KEY refused at startup; a public bind without a key warns first (refused with --require-key); a cross-origin POST to /idrak/stop is 403 and the server stays up, allowed origins pass; a JSON route answers another Content-Type with 415; response_format, logprobs, an object tool_choice and the other unimplemented fields answer 400 naming the field", d => { CpuOnly(d); CliServeDoors(); }),
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

    // The doors of idrak serve (plan 36, G-1 1b): an empty key, a public bind without a key, browser pages calling the
    // control endpoints, bodies not declared JSON, and request fields the OpenAI-style API does not implement.
    private static void CliServeDoors()
    {
        string cache = TempFolder(), folder = TempFolder(), config = Path.Combine(folder, "config.json");
        string model = "tiny=" + TestData("gguf/tiny-qwen3-q8.gguf");
        string?[] saved = [.. new[] { ServeOptions.ApiKeyVariable, ServeOptions.RequireKeyVariable, ServeOptions.ControlOriginsVariable, ServeOptions.AnyContentTypeVariable }
            .Select(Environment.GetEnvironmentVariable)];
        try
        {
            Environment.SetEnvironmentVariable(ServeOptions.ApiKeyVariable, null);
            Environment.SetEnvironmentVariable(ServeOptions.RequireKeyVariable, null);
            Environment.SetEnvironmentVariable(ServeOptions.ControlOriginsVariable, null);
            Environment.SetEnvironmentVariable(ServeOptions.AnyContentTypeVariable, null);

            // An empty key is a configuration error at startup, naming the flag and the variable (an empty key matched an empty bearer).
            foreach (string[] empty in new[] { new[] { "--api-key", "" }, new[] { "--api-key=   " } })
            {
                var refused = ServeCli([.. new[] { "serve", model, "-p", "0", "--cache", cache, "-C", config }, .. empty]);
                Check(refused.Code == ExitCodes.Usage && refused.Error.Contains("--api-key is empty", StringComparison.Ordinal)
                      && refused.Error.Contains(ServeOptions.ApiKeyVariable, StringComparison.Ordinal),
                    $"serve {string.Join(' ', empty)}: refused at startup naming --api-key and {ServeOptions.ApiKeyVariable} (exit {refused.Code}): {refused.Error}");
            }

            Environment.SetEnvironmentVariable(ServeOptions.ApiKeyVariable, " ");                  // set but blank (an empty value unsets it on Windows)
            var blank = ServeCli("serve", model, "-p", "0", "--cache", cache, "-C", config);
            Check(blank.Code == ExitCodes.Usage && blank.Error.Contains($"{ServeOptions.ApiKeyVariable} is empty", StringComparison.Ordinal)
                  && blank.Error.Contains("--api-key", StringComparison.Ordinal), $"a blank {ServeOptions.ApiKeyVariable} is refused at startup naming it and --api-key (exit {blank.Code}): {blank.Error}");
            Environment.SetEnvironmentVariable(ServeOptions.ApiKeyVariable, null);

            // --require-key (or its variable): a public bind without a key refused; with a key, or on loopback, it starts as before.
            var required = ServeCli("serve", model, "-H", "0.0.0.0", "-p", "0", "--require-key", "--cache", cache, "-C", config);
            Check(required.Code == ExitCodes.Usage && required.Error.Contains("--require-key", StringComparison.Ordinal) && required.Error.Contains(ServeOptions.RequireKeyVariable, StringComparison.Ordinal)
                  && required.Error.Contains("--api-key", StringComparison.Ordinal), $"--require-key refuses -H 0.0.0.0 without a key, naming the flag, the variable and --api-key (exit {required.Code}): {required.Error}");
            Environment.SetEnvironmentVariable(ServeOptions.RequireKeyVariable, "1");
            var requiredByVariable = ServeCli("serve", model, "-H", "::", "-p", "0", "--cache", cache, "-C", config);
            Check(requiredByVariable.Code == ExitCodes.Usage && requiredByVariable.Error.Contains(ServeOptions.RequireKeyVariable, StringComparison.Ordinal),
                $"{ServeOptions.RequireKeyVariable}=1 refuses -H :: without a key (exit {requiredByVariable.Code}): {requiredByVariable.Error}");
            Environment.SetEnvironmentVariable(ServeOptions.RequireKeyVariable, "maybe");
            var badSwitch = ServeCli("serve", model, "-p", "0", "--cache", cache, "-C", config);
            Check(badSwitch.Code == ExitCodes.Usage && badSwitch.Error.Contains(ServeOptions.RequireKeyVariable, StringComparison.Ordinal), $"{ServeOptions.RequireKeyVariable}=maybe is a usage error naming it: {badSwitch.Error}");
            Environment.SetEnvironmentVariable(ServeOptions.RequireKeyVariable, null);
            var badOrigin = ServeCli("serve", model, "-p", "0", "--control-origins", "not an origin", "--cache", cache, "-C", config);
            Check(badOrigin.Code == ExitCodes.Usage && badOrigin.Error.Contains("--control-origins", StringComparison.Ordinal), $"a bad --control-origins is a usage error naming it: {badOrigin.Error}");
            Check(ServeOptions.IsPublicHost("0.0.0.0") && ServeOptions.IsPublicHost("*") && ServeOptions.IsPublicHost("+") && ServeOptions.IsPublicHost("::") && ServeOptions.IsPublicHost("[::]")
                  && ServeOptions.IsPublicHost("192.168.1.5") && !ServeOptions.IsPublicHost("127.0.0.1") && !ServeOptions.IsPublicHost("localhost") && !ServeOptions.IsPublicHost("[::1]"),
                "public hosts: the wildcards and a LAN address; loopback and localhost are not");

            // The unimplemented fields, as the translation reads them: the defaults answered, the rest refused naming the field.
            static JsonObject Ask(params (string Field, string Value)[] fields)
            {
                var body = new JsonObject { ["model"] = "tiny", ["messages"] = new JsonArray(new JsonObject { ["role"] = "user", ["content"] = "hi" }) };
                foreach (var (field, value) in fields)
                {
                    body[field] = JsonNode.Parse(value);
                }

                return body;
            }

            static string? Refusal(JsonObject body)
            {
                try
                {
                    CompletionsTranslation.Chat(body);
                    return null;
                }
                catch (ArgumentException e)
                {
                    return e.Message;
                }
            }

            Check(Refusal(Ask(("response_format", """{"type":"text"}"""), ("logprobs", "false"), ("top_logprobs", "0"), ("tool_choice", "\"auto\""), ("n", "1"),
                    ("logit_bias", "{}"), ("user", "\"u\""), ("modalities", """["text"]"""))) is null,
                "the defaults of the unimplemented fields (response_format text, logprobs false, top_logprobs 0, tool_choice auto, n 1, an empty logit_bias, modalities text) are answered");
            foreach (var (field, value) in new[]
                     {
                         ("response_format", """{"type":"json_schema","json_schema":{"name":"x","schema":{"type":"object"}}}"""), ("response_format", """{"type":"json_object"}"""),
                         ("json_schema", """{"type":"object"}"""), ("logprobs", "true"), ("top_logprobs", "3"), ("logit_bias", """{"50256":-100}"""),
                         ("tool_choice", """{"type":"function","function":{"name":"weather"}}"""), ("tool_choice", "\"required\""), ("functions", """[{"name":"weather"}]"""),
                         ("modalities", """["text","audio"]"""), ("audio", """{"voice":"alloy","format":"wav"}"""), ("n", "2"),
                     })
            {
                string? message = Refusal(Ask((field, value)));
                Check(message is not null && message.StartsWith(field + ":", StringComparison.Ordinal), $"{field}: {value} refused naming {field}, not ignored: {message ?? "accepted"}");
            }

            int port;
            using (var probe = new TcpListener(IPAddress.Loopback, 0))
            {
                probe.Start();
                port = ((IPEndPoint)probe.LocalEndpoint).Port;
            }

            // Listening on every interface without a key: the warning comes first; the doors are guarded all the same.
            string url = $"http://127.0.0.1:{port}";
            var serveOutput = new StringWriter();
            var serveError = new StringWriter();
            var server = Task.Run(() => CommandLine.Run(["serve", model, "-H", "0.0.0.0", "-p", port.ToString(), "--cache", cache, "-C", config,
                "--cors", "http://cors.example:3000", "--control-origins", "http://ok.example"], TextWriter.Synchronized(serveOutput), TextWriter.Synchronized(serveError)));
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
                string firstLine = serveError.ToString().Split('\n')[0];
                Check(firstLine.Contains("warning", StringComparison.Ordinal) && firstLine.Contains("0.0.0.0", StringComparison.Ordinal) && firstLine.Contains("no API key", StringComparison.Ordinal)
                      && firstLine.Contains("--require-key", StringComparison.Ordinal), $"a public bind without a key prints the warning first: {serveError}");

                HttpResponseMessage Send(string path, string body, string type = "application/json", (string Name, string Value)[]? headers = null)
                {
                    using var request = new HttpRequestMessage(HttpMethod.Post, path) { Content = new StringContent(body, Encoding.UTF8, type) };
                    foreach (var (name, value) in headers ?? [])
                    {
                        request.Headers.TryAddWithoutValidation(name, value);
                    }

                    return http.SendAsync(request).Result;
                }

                string Body(HttpResponseMessage response) => response.Content.ReadAsStringAsync().Result;

                // A page from another origin cannot stop (or load, or unload) the server: 403 naming the setting, and it stays up.
                var stop = Send("/idrak/stop", "[]", headers: [("Origin", "http://evil.example")]);
                Check(stop.StatusCode == HttpStatusCode.Forbidden && Body(stop).Contains("--control-origins", StringComparison.Ordinal),
                    $"a cross-origin POST to /idrak/stop is 403 naming --control-origins, not {(int)stop.StatusCode}: {Body(stop)}");
                var rebound = Send("/idrak/stop", "[]", headers: [("Origin", $"http://evil.example:{port}")]);
                Check(rebound.StatusCode == HttpStatusCode.Forbidden, $"a page of another host at the server's port (DNS rebinding) is 403, not {(int)rebound.StatusCode}");
                var sandboxed = Send("/idrak/stop", "[]", headers: [("Origin", "null")]);
                Check(sandboxed.StatusCode == HttpStatusCode.Forbidden, $"Origin: null is 403, not {(int)sandboxed.StatusCode}");
                var crossSite = Send("/idrak/stop", "[]", headers: [("Sec-Fetch-Site", "cross-site")]);
                Check(crossSite.StatusCode == HttpStatusCode.Forbidden && Body(crossSite).Contains("Sec-Fetch-Site", StringComparison.Ordinal),
                    $"Sec-Fetch-Site: cross-site without an Origin is 403 naming it, not {(int)crossSite.StatusCode}: {Body(crossSite)}");
                var load = Send("/idrak/load", """{"model":"tiny"}""", headers: [("Origin", "http://evil.example")]);
                Check(load.StatusCode == HttpStatusCode.Forbidden, $"a cross-origin POST to /idrak/load is 403, not {(int)load.StatusCode}");
                Thread.Sleep(500);                                  // a stop that got through would have ended the server by now
                Check(!server.IsCompleted && http.GetAsync("/").Result.IsSuccessStatusCode, $"the server stays up after the refused stops: {serveOutput}{serveError}");
                Check(serveOutput.ToString().Contains("/idrak/stop: 403", StringComparison.Ordinal), $"the server prints the refusals: {serveOutput}");

                // Allowed: --control-origins, an origin --cors names, the server's own page (loopback at its port), and no Origin at all.
                foreach (var origin in new[] { "http://ok.example", "http://cors.example:3000", $"http://127.0.0.1:{port}", $"http://localhost:{port}", "" })
                {
                    var unload = Send("/idrak/unload", """{"model":"tiny"}""", headers: origin.Length == 0 ? null : [("Origin", origin)]);
                    Check(unload.StatusCode == HttpStatusCode.OK, $"/idrak/unload from '{origin}' is allowed, not {(int)unload.StatusCode}: {Body(unload)}");
                }

                // A JSON route answers a body declared as another type with 415 (a cross-site form or text/plain post).
                string any = Ask(("prompt", "\"x\"")).ToJsonString();
                foreach (var (path, type) in new[] { ("/idrak/unload", "text/plain"), ("/v1/chat/completions", "text/plain"), ("/api/chat", "application/x-www-form-urlencoded"), ("/v1/completions", "multipart/form-data") })
                {
                    var wrong = Send(path, any, type);
                    Check(wrong.StatusCode == HttpStatusCode.UnsupportedMediaType && Body(wrong).Contains("application/json", StringComparison.Ordinal)
                          && Body(wrong).Contains("--any-content-type", StringComparison.Ordinal), $"{path} with {type} is 415 naming the type to send and --any-content-type, not {(int)wrong.StatusCode}: {Body(wrong)}");
                }

                Check((string?)JsonNode.Parse(Body(Send("/v1/chat/completions", "{}", "text/plain")))?["error"]?["code"] == "unsupported_media_type", "the /v1 415 has the OpenAI error shape");

                // The OpenAI-style API refuses what it does not implement, naming the field (before any model loads).
                foreach (var (path, field, json) in new[]
                         {
                             ("/v1/chat/completions", "response_format", Ask(("response_format", """{"type":"json_object"}"""))),
                             ("/v1/chat/completions", "logprobs", Ask(("logprobs", "true"))),
                             ("/v1/chat/completions", "tool_choice", Ask(("tools", """[{"type":"function","function":{"name":"f"}}]"""), ("tool_choice", """{"type":"function","function":{"name":"f"}}"""))),
                             ("/v1/completions", "logprobs", new JsonObject { ["model"] = "tiny", ["prompt"] = "x", ["logprobs"] = 2 }),
                             ("/v1/completions", "echo", new JsonObject { ["model"] = "tiny", ["prompt"] = "x", ["echo"] = true }),
                         })
                {
                    var refused = Send(path, json.ToJsonString());
                    string message = (string?)JsonNode.Parse(Body(refused))?["error"]?["message"] ?? "";
                    Check(refused.StatusCode == HttpStatusCode.BadRequest && message.StartsWith(field + ":", StringComparison.Ordinal),
                        $"{path} with {field} answers 400 naming {field}, not {(int)refused.StatusCode}: {Body(refused)}");
                }

                var ps = JsonNode.Parse(http.GetStringAsync("/idrak/ps").Result)!["models"]![0]!;
                Check(!(bool)ps["loaded"]!, $"none of the refused requests loaded the model: {ps}");
            }
            finally
            {
                var stop = ServeCli("server", "stop", "-p", port.ToString());
                Check(stop.Code == 0, $"server stop from the command line (no Origin) still works: {stop.Output}{stop.Error}");
                Check(server.Wait(TimeSpan.FromSeconds(60)) && server.Result == 0, $"serve exits with 0 after stop: {serveOutput}{serveError}");
            }
        }
        finally
        {
            Environment.SetEnvironmentVariable(ServeOptions.ApiKeyVariable, saved[0]);
            Environment.SetEnvironmentVariable(ServeOptions.RequireKeyVariable, saved[1]);
            Environment.SetEnvironmentVariable(ServeOptions.ControlOriginsVariable, saved[2]);
            Environment.SetEnvironmentVariable(ServeOptions.AnyContentTypeVariable, saved[3]);
        }
    }

    // The port: --port, else IDRAK_PORT, else the config's serve.port, else Idrak's own default; the client commands put
    // the server started last on this machine between the environment and the config.
    private static void CliServePort()
    {
        string folder = TempFolder(), cache = Path.Combine(folder, "cache"), config = Path.Combine(folder, "config.json");
        var serve = CommandTable.All.First(c => c.Name == "serve");
        var ps = CommandTable.All.First(c => c.Name == "server ps");
        string? old = Environment.GetEnvironmentVariable(ServeOptions.PortVariable);

        (int Port, string Url) Chosen(params string[] port)
        {
            var options = new Dictionary<string, List<string>> { ["--config"] = [config], ["--cache"] = [cache] };
            if (port.Length > 0)
            {
                options["--port"] = [.. port];
            }

            using var forServe = new CommandContext(serve, [], options, [], new StringWriter(), new StringWriter());
            using var forClient = new CommandContext(ps, [], options, [], new StringWriter(), new StringWriter());
            return (ServeOptions.ReadSettings(forServe).Port, ServerClient.BaseUrl(forClient));
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
            Environment.SetEnvironmentVariable(ServeOptions.PortVariable, null);
            Check(ServeOptions.DefaultPort == 7317 && Chosen() == (7317, "http://127.0.0.1:7317"), $"the default port: {Chosen()}");
            Check(ServeCli("config", "set", "serve.port", "8123", "-C", config).Code == 0 && Chosen() == (8123, "http://127.0.0.1:8123"), $"config serve.port: {Chosen()}");
            Environment.SetEnvironmentVariable(ServeOptions.PortVariable, "8124");
            Check(Chosen() == (8124, "http://127.0.0.1:8124"), $"IDRAK_PORT before the config: {Chosen()}");
            Check(Chosen("8125") == (8125, "http://127.0.0.1:8125") && Chosen("0").Port == 0, $"--port before IDRAK_PORT: {Chosen("8125")}");

            // A running server's state file (this process stands in for it): found by the clients unless the port is given.
            Directory.CreateDirectory(Path.Combine(cache, "servers"));
            File.WriteAllText(Path.Combine(cache, "servers", "9301.json"), new JsonObject
            {
                ["url"] = "http://127.0.0.1:9301", ["pid"] = Environment.ProcessId, ["started"] = DateTimeOffset.UtcNow.ToString("O"), ["models"] = new JsonArray(),
            }.ToJsonString());
            Check(Chosen().Url == "http://127.0.0.1:8124", "IDRAK_PORT before the running server");
            Environment.SetEnvironmentVariable(ServeOptions.PortVariable, null);
            Check(Chosen() == (8123, "http://127.0.0.1:9301"), $"the running server before the config for clients, the config for serve: {Chosen()}");

            Environment.SetEnvironmentVariable(ServeOptions.PortVariable, "port");
            Check(Refused(() => Chosen()), "IDRAK_PORT that is not a number is a usage error");
            Environment.SetEnvironmentVariable(ServeOptions.PortVariable, "70000");
            Check(Refused(() => Chosen()), "IDRAK_PORT out of range is a usage error");
            Environment.SetEnvironmentVariable(ServeOptions.PortVariable, null);
            Check(ServeCli("config", "set", "serve.port", "\"8126\"", "-C", config).Code == 0 && Chosen().Port == 8126, "serve.port as text");
            Check(ServeCli("config", "set", "serve.port", "70000", "-C", config).Code == 0 && Refused(() => Chosen()), "serve.port out of range is a usage error");
            Check(ServeCli("serve", TestData("gguf/tiny-qwen3-q8.gguf"), "-p", "65536").Code == 2, "--port out of range is a usage error");
        }
        finally
        {
            Environment.SetEnvironmentVariable(ServeOptions.PortVariable, old);
            Directory.Delete(folder, recursive: true);
        }

        string help = ServeCli("serve", "--help").Output;
        Check(help.Contains("7317", StringComparison.Ordinal) && help.Contains("-p 11434", StringComparison.Ordinal) && help.Contains(ServeOptions.PortVariable, StringComparison.Ordinal)
              && help.Contains("serve.port", StringComparison.Ordinal), $"serve's help names the default, the variable, the config key and -p 11434: {help}");
        Check(Idrak.Cli.Shared.EnvironmentVariables.Find(ServeOptions.PortVariable) is { Group: "The tool" } variable && variable.Default.Contains("7317", StringComparison.Ordinal)
              && variable.Commands.Contains("serve") && variable.Commands.Contains("ping"), "IDRAK_PORT in the environment table, with its default");
        foreach (string name in new[] { "ui", "api", "ping", "server ps", "server stop" })
        {
            Check(Help.For(CommandTable.All.First(c => c.Name == name)).Contains(ServeOptions.PortVariable, StringComparison.Ordinal), $"{name}'s help names {ServeOptions.PortVariable}");
        }
    }

    private static void CliMcpServe()
    {
        string tools = typeof(CliServeTestTools).Assembly.Location;
        var list = ServeCli("mcp", "serve", tools, "--list", "--json");
        var names = JsonNode.Parse(list.Output)!["tools"]!.AsArray().Select(t => (string)t!["name"]!)
            .Where(n => n.StartsWith("serve_", StringComparison.Ordinal)).Order().ToList();          // the test assembly holds other groups' tools too
        Check(list.Code == 0 && names.SequenceEqual(["serve_add", "serve_echo", "serve_upper"]), $"mcp serve --list: {list.Output}{list.Error}");
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
                Check(listed.Select(t => t.Definition.Name).Where(n => n.StartsWith("serve_", StringComparison.Ordinal)).Order().SequenceEqual(["serve_add", "serve_echo", "serve_upper"]), "tools over MCP");
                Check(await source.CallAsync("serve_add", new JsonObject { ["a"] = 2, ["b"] = 40 }) == "42", "a call over MCP");
                Check(await source.CallAsync("serve_echo", new JsonObject { ["text"] = "hi" }) == "hi", "a static tool over MCP");
                Check(await source.CallAsync("serve_upper", new JsonObject { ["text"] = "hi" }) == "HI", "a provider's own ITool over MCP");
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
