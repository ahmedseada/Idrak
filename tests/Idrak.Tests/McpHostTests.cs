// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.IO.Pipelines;
using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Cli;
using Idrak.Cli.Commands.Serve;
using Idrak.Mcp;
using Idrak.Mcp.Abstractions;
using StreamClientTransport = ModelContextProtocol.Protocol.StreamClientTransport;
using Tool = Idrak.Abstraction.Generation.Tool;

// MCP served from any process (McpHost over the McpServerTransports registry): stdio over pipes and http over a real
// loopback listener each list tools, call one, cancel a long call and stop; http's token, loopback-only bind, sessions
// and origin check; the "not registered" refusal; mcp serve --transport http end to end; the per-call cost of each.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] McpHostGroup =
    [
        ("mcp host: stdio over pipes lists the tools, calls one, cancels a long call (Idrak's client sends notifications/cancelled: the server's tool stops) and stops", d => { if (d == Device.Cpu) Run(McpHostStdioAsync); }),
        ("mcp host: http on loopback: McpTools.ConnectHttpAsync lists, calls, cancels a long call (client cancel and disconnect), ends its session, stops", d => { if (d == Device.Cpu) Run(McpHostHttpAsync); }),
        ("mcp host: http refuses a missing or wrong token (401), a page off this machine (403), an unknown session (404), a non-loopback bind unless allowed; the token can be turned off", d => { if (d == Device.Cpu) Run(McpHostHttpRefusalsAsync); }),
        ("mcp host: an unknown transport fails with \"not registered\" naming stdio and http; option keys a transport does not take are refused", d => { if (d == Device.Cpu) Run(McpHostRegistryAsync); }),
        ("mcp host: a client of the 2026-07-28 revision connects over http in one request (server/discover, no 400, no initialize) and over stdio; sessionless GET and DELETE are 405; a sessionless notifications/cancelled stops its call", d => { if (d == Device.Cpu) Run(McpHostSessionlessAsync); }),
        ("mcp host: http MaxSessions: a new session over the limit is 503 with Retry-After; ReplaceIdleSessions closes the idle one unused the longest for it; IDRAK_MCP_MAX_SESSIONS and IDRAK_MCP_REPLACE_IDLE", d => { if (d == Device.Cpu) Run(McpHostSessionLimitAsync); }),
        ("mcp host: telemetry (TelemetryLevel.Mcp): sessions opened and closed, requests received and answered (method, tool, status, bytes, time), tool calls finished, failed and cancelled, refusals with status and reason, over http and stdio", d => { if (d == Device.Cpu) Run(McpHostTelemetryAsync); }),
        ("mcp host: per-call cost of a trivial tool over http and over stdio, cold and warm apart, telemetry off and with a listener (printed)", d => { if (d == Device.Cpu) Run(McpHostOverheadAsync); }),
        ("cli serve: mcp serve --transport http prints its address and token; a client with the token lists and calls; --verbose prints its requests, --profile its report at exit; Ctrl+C stops it", d => { if (d == Device.Cpu) CliMcpServeHttp(); }),
        ("cli serve: mcp serve over stdio with --verbose and --profile: the events and the report go to the error output, the protocol's output holds JSON-RPC lines only; --quiet drops the status line, not the events", d => { if (d == Device.Cpu) CliMcpServeStdioVerbose(); }),
    ];

    private static void Run(Func<Task> test) => test().GetAwaiter().GetResult();

    // echo (instant) and slow (waits until cancelled, then says so).
    private static (ToolRegistry Registry, Func<TaskCompletionSource> Cancelled) McpTestTools()
    {
        var cancelled = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var registry = ToolRegistry.Create()
            .Add(Tool.Create("echo", "Says the text back", JsonNode.Parse("""{"type":"object","properties":{"text":{"type":"string"}}}""")!,
                (arguments, _) => Task.FromResult((string?)arguments["text"] ?? "")))
            .Add(Tool.Create("slow", "Waits until it is cancelled", JsonNode.Parse("""{"type":"object","properties":{}}""")!, async (_, ct) =>
            {
                try
                {
                    await Task.Delay(TimeSpan.FromMinutes(2), ct);
                }
                catch (OperationCanceledException)
                {
                    Volatile.Read(ref cancelled).TrySetResult();
                    throw;
                }

                return "done";
            }))
            .Add(Tool.Create("fail", "Always fails", JsonNode.Parse("""{"type":"object","properties":{}}""")!,
                (_, _) => Task.FromException<string>(new InvalidOperationException("the fail tool fails"))))
            .Build();
        return (registry, () =>
        {
            var next = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
            Volatile.Write(ref cancelled, next);
            return next;
        });
    }

    private static async Task CheckToolsAndCallAsync(McpToolSource source, string where)
    {
        var tools = await source.ListToolsAsync();
        Check(tools.Select(t => t.Definition.Name).Order().SequenceEqual(["echo", "fail", "slow"]), $"{where}: tools {string.Join(", ", tools.Select(t => t.Definition.Name))}");
        Check(await source.CallAsync("echo", new JsonObject { ["text"] = "سلام" }) == "سلام", $"{where}: a call answers");
    }

    // A long call the client cancels: the server's tool sees its token cancelled (Idrak's client sends notifications/cancelled
    // over any transport; over http the SDK's client also closes the request's connection, which cancels the call too).
    private static async Task CheckCancelAsync(McpToolSource source, Func<TaskCompletionSource> armed, string where)
    {
        var cancelled = armed();
        using var cancel = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
        bool threw = false;
        try
        {
            await source.CallAsync("slow", new JsonObject(), cancel.Token);
        }
        catch (OperationCanceledException)
        {
            threw = true;
        }

        Check(threw, $"{where}: the cancelled call throws OperationCanceledException");
        Check(await Task.WhenAny(cancelled.Task, Task.Delay(TimeSpan.FromSeconds(15))) == cancelled.Task, $"{where}: the server's tool was not cancelled when the client cancelled");
    }

    private static async Task McpHostStdioAsync()
    {
        var (registry, armed) = McpTestTools();
        var toServer = new Pipe();
        var toClient = new Pipe();
        var host = await McpHost.StartAsync(McpServerTransports.Stdio, registry, new McpHostOptions { Input = toServer.Reader.AsStream(), Output = toClient.Writer.AsStream() });
        Check(host.Address == "stdio" && host.Token is null && host.Transport == "stdio" && host.ToolCount == 3, $"stdio host: {host.Address}, {host.Token}, {host.ToolCount} tools");
        await using (var source = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream())))
        {
            Check(source.ServerName == "idrak", $"stdio: server name {source.ServerName}");
            await CheckToolsAndCallAsync(source, "stdio");
            using (var recorder = new McpRecorder())
            {
                await CheckCancelAsync(source, armed, "stdio");
                Check(await recorder.WaitAsync(e => e.Kind == McpEventKind.CancelSent && e.Tool == "slow"), $"stdio: the client reports the cancellation it sent: {recorder}");
            }

            Check(await source.CallAsync("echo", new JsonObject { ["text"] = "after" }) == "after", "stdio: the session goes on after a cancelled call");
            await host.StopAsync();
            Check(host.Completion.IsCompleted, "stdio: StopAsync completes the host");
        }

        await host.DisposeAsync();                     // a second stop is a no-op

        // A long call the client cancels (notifications/cancelled; the SDK's client sends none, so the lines are written
        // here as a client that sends it would): the server's tool sees its token cancelled, and the session goes on.
        toServer = new Pipe();
        toClient = new Pipe();
        await using (var raw = await McpHost.StartAsync("stdio", registry, new McpHostOptions { Input = toServer.Reader.AsStream(), Output = toClient.Writer.AsStream() }))
        {
            var input = toServer.Writer.AsStream();
            using var output = new StreamReader(toClient.Reader.AsStream());
            async Task SendAsync(string line) => await input.WriteAsync(Encoding.UTF8.GetBytes(line + "\n"));
            await SendAsync("""{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"raw","version":"1"}}}""");
            Check((await output.ReadLineAsync())?.Contains("\"serverInfo\"", StringComparison.Ordinal) == true, "stdio: initialize answers");
            await SendAsync("""{"jsonrpc":"2.0","method":"notifications/initialized"}""");
            var cancelled = armed();
            await SendAsync("""{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"slow","arguments":{}}}""");
            await Task.Delay(200);
            await SendAsync("""{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":2,"reason":"test"}}""");
            Check(await Task.WhenAny(cancelled.Task, Task.Delay(TimeSpan.FromSeconds(15))) == cancelled.Task, "stdio: the server's tool was not cancelled by notifications/cancelled");
            await SendAsync("""{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"echo","arguments":{"text":"on"}}}""");
            string? answer;
            while ((answer = await output.ReadLineAsync()) is not null && !answer.Contains("\"id\":3", StringComparison.Ordinal))
            {
                // the cancelled call may still answer (or not) first
            }

            Check(answer?.Contains("\"on\"", StringComparison.Ordinal) == true, $"stdio: the session goes on after a cancelled call: {answer}");
        }

        // A host also ends when its client closes its end.
        toServer = new Pipe();
        toClient = new Pipe();
        await using var second = await McpHost.StartAsync("STDIO", registry, new McpHostOptions { Input = toServer.Reader.AsStream(), Output = toClient.Writer.AsStream() });
        await toServer.Writer.CompleteAsync();
        Check(await Task.WhenAny(second.Completion, Task.Delay(TimeSpan.FromSeconds(15))) == second.Completion, "stdio: the host ends when its input ends (names ignore case)");
    }

    private static async Task McpHostHttpAsync()
    {
        var (registry, armed) = McpTestTools();
        var host = await McpHost.StartAsync(McpServerTransports.Http, registry);
        var address = new Uri(host.Address);
        Check(address.Scheme == "http" && address.Host == "127.0.0.1" && address.Port > 0 && address.AbsolutePath == "/mcp" && host.Token is { Length: >= 32 },
            $"http host: {host.Address}, token {host.Token}");
        await using (var source = await McpTools.ConnectHttpAsync(address, host.Token))
        {
            Check(source.ServerName == "idrak", $"http: server name {source.ServerName}");
            await CheckToolsAndCallAsync(source, "http");
            await CheckCancelAsync(source, armed, "http");
            Check(await source.CallAsync("echo", new JsonObject { ["text"] = "after" }) == "after", "http: the session goes on after a cancelled call");

            // Two clients at once: each its own session.
            await using var other = await McpTools.ConnectHttpAsync(address, host.Token);
            var both = await Task.WhenAll(source.CallAsync("echo", new JsonObject { ["text"] = "a" }), other.CallAsync("echo", new JsonObject { ["text"] = "b" }));
            Check(both[0] == "a" && both[1] == "b", "http: two clients at once");
        }

        // A client that disconnects in the middle of a call: the call is cancelled.
        using (var http = new HttpClient())
        {
            string session = await RawInitializeAsync(http, host.Address, host.Token);
            var cancelled = armed();
            using var gone = new CancellationTokenSource(TimeSpan.FromMilliseconds(300));
            try
            {
                await RawPostAsync(http, host.Address, host.Token, session, """{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"slow","arguments":{}}}""", gone.Token);
                Check(false, "http: the slow call answered");
            }
            catch (OperationCanceledException)
            {
                // the client gave up and closed its connection
            }

            Check(await Task.WhenAny(cancelled.Task, Task.Delay(TimeSpan.FromSeconds(15))) == cancelled.Task, "http: a call whose client disconnected was not cancelled");

            // DELETE ends the session: its id is unknown after it.
            var delete = new HttpRequestMessage(HttpMethod.Delete, host.Address);
            delete.Headers.TryAddWithoutValidation("Authorization", "Bearer " + host.Token);
            delete.Headers.TryAddWithoutValidation("Mcp-Session-Id", session);
            Check((await http.SendAsync(delete)).StatusCode == HttpStatusCode.OK, "http: DELETE ends the session");
            var after = await RawPostAsync(http, host.Address, host.Token, session, """{"jsonrpc":"2.0","id":8,"method":"tools/list"}""", default);
            Check(after.StatusCode == HttpStatusCode.NotFound, $"http: a deleted session answers 404, not {(int)after.StatusCode}");
        }

        await host.StopAsync();
        Check(host.Completion.IsCompletedSuccessfully, "http: StopAsync completes the host");
        bool refused = false;
        try
        {
            using var http = new HttpClient();
            await RawInitializeAsync(http, host.Address, host.Token);
        }
        catch (HttpRequestException)
        {
            refused = true;
        }

        Check(refused, "http: a stopped host releases its port (connections refused)");
        await host.DisposeAsync();
    }

    private static async Task McpHostHttpRefusalsAsync()
    {
        var (registry, _) = McpTestTools();
        await using (var host = await McpHost.StartAsync("http", registry, new McpHostOptions { Token = "the-token" }))
        {
            Check(host.Token == "the-token", "http: the token given is the one required");
            using var http = new HttpClient();
            Check((await RawPostAsync(http, host.Address, null, null, InitializeBody, default)).StatusCode == HttpStatusCode.Unauthorized, "http: no token: 401");
            var wrong = await RawPostAsync(http, host.Address, "the-tokeN", null, InitializeBody, default);
            Check(wrong.StatusCode == HttpStatusCode.Unauthorized && wrong.Headers.WwwAuthenticate.ToString().Contains("Bearer", StringComparison.Ordinal), "http: a wrong token: 401 with WWW-Authenticate: Bearer");
            Check((await RawPostAsync(http, host.Address, "the-token", "no-such-session", """{"jsonrpc":"2.0","id":1,"method":"tools/list"}""", default)).StatusCode == HttpStatusCode.NotFound,
                "http: an unknown session: 404");
            Check((await RawPostAsync(http, host.Address, "the-token", null, """{"jsonrpc":"2.0","id":1,"method":"tools/list"}""", default)).StatusCode == HttpStatusCode.BadRequest,
                "http: a request before initialize, without a session: 400");
            Check((await RawPostAsync(http, host.Address, "the-token", null, "{not json", default)).StatusCode == HttpStatusCode.BadRequest, "http: a body that is not JSON-RPC: 400");
            var page = await RawPostAsync(http, host.Address, "the-token", null, InitializeBody, default, origin: "http://evil.example");
            Check(page.StatusCode == HttpStatusCode.Forbidden, $"http: a page off this machine: 403, not {(int)page.StatusCode}");
            var local = await RawPostAsync(http, host.Address, "the-token", null, InitializeBody, default, origin: "http://localhost:3000");
            Check(local.StatusCode == HttpStatusCode.OK, $"http: a page on this machine: 200, not {(int)local.StatusCode}");
            Check((await http.GetAsync(host.Address.Replace("/mcp", "/other", StringComparison.Ordinal))).StatusCode == HttpStatusCode.NotFound, "http: another path: 404");
            bool failed = false;
            try
            {
                await using var client = await McpTools.ConnectHttpAsync(new Uri(host.Address), "wrong");
            }
            catch (Exception)
            {
                failed = true;
            }

            Check(failed, "http: McpTools.ConnectHttpAsync with a wrong token fails");
        }

        // Off this machine only when asked.
        string? refusal = null;
        try
        {
            await using var open = await McpHost.StartAsync("http", registry, new McpHostOptions { Address = "0.0.0.0:0" });
        }
        catch (ArgumentException e)
        {
            refusal = e.Message;
        }

        Check(refusal is not null && refusal.Contains("AllowRemote", StringComparison.Ordinal), $"http: a non-loopback bind is refused unless allowed: {refusal}");
        await using (var allowed = await McpHost.StartAsync("http", registry, new McpHostOptions { Address = "0.0.0.0:0", AllowRemote = true }))
        {
            await using var client = await McpTools.ConnectHttpAsync(new Uri(allowed.Address), allowed.Token);
            Check(await client.CallAsync("echo", new JsonObject { ["text"] = "x" }) == "x", $"http: with AllowRemote it listens on every interface ({allowed.Address})");
        }

        // The token turned off: any request on the address is answered.
        await using (var open = await McpHost.StartAsync("http", registry, new McpHostOptions().WithToken(McpHostOptions.NoToken)))
        {
            Check(open.Token is null, "http: WithToken(\"none\") requires no token");
            using var http = new HttpClient();
            Check((await RawPostAsync(http, open.Address, null, null, InitializeBody, default)).StatusCode == HttpStatusCode.OK, "http: no token required: 200");
            await using var client = await McpTools.ConnectHttpAsync(new Uri(open.Address));
            Check(await client.CallAsync("echo", new JsonObject { ["text"] = "y" }) == "y", "http: a client without a token when none is required");
        }

        // A fixed port (one the system freed just now) is the one listened on.
        int port;
        using (var probe = new System.Net.Sockets.TcpListener(IPAddress.Loopback, 0))
        {
            probe.Start();
            port = ((IPEndPoint)probe.LocalEndpoint).Port;
        }

        await using (var fixedPort = await McpHost.StartAsync("http", registry, new McpHostOptions { Address = $"localhost:{port}" }))
        {
            Check(fixedPort.Address == $"http://127.0.0.1:{port}/mcp", $"http: --mcp-address localhost:{port} gives {fixedPort.Address}");
        }
    }

    private static async Task McpHostRegistryAsync()
    {
        var (registry, _) = McpTestTools();
        Check(McpServerTransports.Names.Contains("stdio") && McpServerTransports.Names.Contains("http") && McpServerTransports.Find("HTTP") is not null
              && McpServerTransports.Origin("http") == Overrides.Library && McpServerTransports.Describe().Contains("http: ", StringComparison.Ordinal),
            $"the built-in transports: {string.Join(", ", McpServerTransports.Names)}");
        NotRegisteredException? error = null;
        try
        {
            await McpHost.StartAsync("pigeon", registry);
        }
        catch (NotRegisteredException e)
        {
            error = e;
        }

        Check(error is not null && error.Message.Contains("'pigeon' is not registered", StringComparison.Ordinal) && error.Message.Contains("stdio, http", StringComparison.Ordinal)
              && error.Message.Contains("McpServerTransports.Register", StringComparison.Ordinal), $"an unknown transport: {error?.Message}");
        ArgumentException? keys = null;
        try
        {
            await McpHost.StartAsync("http", registry, new McpHostOptions { TransportOptions = new Dictionary<string, string> { ["backlog"] = "5" } });
        }
        catch (ArgumentException e)
        {
            keys = e;
        }

        Check(keys is not null && keys.Message.Contains("'backlog'", StringComparison.Ordinal), $"an option key http does not take: {keys?.Message}");

        // An app's transport over a library name takes its place; unregistering brings the library's back.
        var mine = new RecordingTransport();
        McpServerTransports.Register(mine);
        try
        {
            Check(McpServerTransports.Get("stdio") == mine && McpServerTransports.Origin("stdio") != Overrides.Library, "an app's stdio over the library's");
        }
        finally
        {
            McpServerTransports.Unregister("stdio");
        }

        Check(McpServerTransports.Get("stdio") != mine && McpServerTransports.Default("stdio") is not null, "unregistering brings the library's stdio back");
    }

    private sealed class NullMcpHook : ITelemetryHook
    {
        public TelemetryLevel Levels => TelemetryLevel.Mcp;

        public void OnMcp(in McpEvent e)
        {
        }
    }

    private sealed class RecordingTransport : IMcpServerTransport
    {
        public string Name => "stdio";

        public string Summary => "a test's";

        public IReadOnlySet<string> Keys => OptionKeys.None;

        public ValueTask<IMcpListener> StartAsync(McpTransportRequest request, CancellationToken cancellationToken = default) => throw new NotSupportedException();
    }

    // Per-call cost of echo, client and server in this process. Cold: the first call on a fresh connection (the first
    // transport also pays for compiling the shared call path). Warm: both connections warmed first (the tiered JIT
    // finishes), then timed in alternating blocks so neither runs on a machine the other warmed differently.
    private static async Task McpHostOverheadAsync()
    {
        var (registry, _) = McpTestTools();
        const int warmup = 1500, blocks = 10, block = 300;
        var arguments = new JsonObject { ["text"] = "hi" };
        await using var httpHost = await McpHost.StartAsync("http", registry);
        await using var http = await McpTools.ConnectHttpAsync(new Uri(httpHost.Address), httpHost.Token);
        await using var session = await McpTools.ConnectAsync(new ModelContextProtocol.Client.HttpClientTransport(new ModelContextProtocol.Client.HttpClientTransportOptions
        {
            Endpoint = new Uri(httpHost.Address),
            AdditionalHeaders = new Dictionary<string, string> { ["Authorization"] = "Bearer " + httpHost.Token },
        }), new ModelContextProtocol.Client.McpClientOptions { ProtocolVersion = "2025-11-25" });
        var toServer = new Pipe();
        var toClient = new Pipe();
        await using var stdioHost = await McpHost.StartAsync("stdio", registry, new McpHostOptions { Input = toServer.Reader.AsStream(), Output = toClient.Writer.AsStream() });
        await using var stdio = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream()));
        async Task<double> TimeAsync(McpToolSource source, int count)
        {
            long start = Stopwatch.GetTimestamp();
            for (int i = 0; i < count; i++)
            {
                await source.CallAsync("echo", arguments);
            }

            return Stopwatch.GetElapsedTime(start).TotalMicroseconds;
        }

        double httpCold = await TimeAsync(http, 1), sessionCold = await TimeAsync(session, 1), stdioCold = await TimeAsync(stdio, 1);
        await TimeAsync(http, warmup);
        await TimeAsync(session, warmup);
        await TimeAsync(stdio, warmup);
        double httpWarm = 0, sessionWarm = 0, stdioWarm = 0;
        long httpBytes = 0, sessionBytes = 0, stdioBytes = 0;
        for (int b = 0; b < blocks; b++)
        {
            long before = GC.GetTotalAllocatedBytes(precise: true);
            httpWarm += await TimeAsync(http, block);
            long middle = GC.GetTotalAllocatedBytes(precise: true);
            sessionWarm += await TimeAsync(session, block);
            long after = GC.GetTotalAllocatedBytes(precise: true);
            stdioWarm += await TimeAsync(stdio, block);
            stdioBytes += GC.GetTotalAllocatedBytes(precise: true) - after;
            sessionBytes += after - middle;
            httpBytes += middle - before;
        }

        int calls = blocks * block;
        Console.WriteLine($"    mcp per call (echo, client and server in process): http sessionless (2026-07-28) cold {httpCold:F0} us, warm {httpWarm / calls:F1} us, {httpBytes / calls} B; "
                          + $"http session (2025-11-25) cold {sessionCold:F0} us, warm {sessionWarm / calls:F1} us, {sessionBytes / calls} B; "
                          + $"stdio (pipes) cold {stdioCold:F0} us, warm {stdioWarm / calls:F1} us, {stdioBytes / calls} B");

        // Telemetry off and a listener at TelemetryLevel.Mcp (it keeps nothing), in alternating blocks on the warm
        // connections: what the events cost when somebody listens, against the one static read when nobody does.
        double httpOff = 0, stdioOff = 0, httpHeard = 0, stdioHeard = 0;
        for (int b = 0; b < blocks; b++)
        {
            httpOff += await TimeAsync(http, block);
            stdioOff += await TimeAsync(stdio, block);
            using (Telemetry.Subscribe(new NullMcpHook()))
            {
                httpHeard += await TimeAsync(http, block);
                stdioHeard += await TimeAsync(stdio, block);
            }
        }

        Console.WriteLine($"    mcp per call, telemetry off / with a listener (alternating blocks): http {httpOff / calls:F1} / {httpHeard / calls:F1} us, "
                          + $"stdio {stdioOff / calls:F1} / {stdioHeard / calls:F1} us");
        Check(httpWarm > 0 && stdioWarm > 0 && httpHeard > 0 && stdioHeard > 0, "the calls were timed");
    }

    private static void CliMcpServeHttp()
    {
        string tools = typeof(CliServeTestTools).Assembly.Location;
        var error = new McpErrorWriter();
        using var stop = new CancellationTokenSource();
        var output = new StringWriter();
        var serving = Task.Run(() => CommandLine.Run(["mcp", "serve", tools, "--transport", "http", "--mcp-token", "cli-token", "--quiet", "--verbose", "--profile"], output, error,
            CommandHost.Console, stop.Token));
        try
        {
            Check(error.Address.Wait(TimeSpan.FromSeconds(30)), $"mcp serve --transport http prints its address: {error}");
            string text = error.ToString();
            var url = Regex.Match(text, @"at (http://127\.0\.0\.1:\d+/mcp)").Groups[1].Value;
            Check(url.Length > 0 && text.Contains("Bearer cli-token", StringComparison.Ordinal) && text.Contains("--mcp-token cli-token", StringComparison.Ordinal),
                $"the address and token are printed (also with --quiet): {text}");
            Task.Run(async () =>
            {
                await using var source = await McpTools.ConnectHttpAsync(new Uri(url), "cli-token");
                Check((await source.ListToolsAsync()).Any(t => t.Definition.Name == "serve_add"), "mcp serve over http lists the assembly's tools");
                Check(await source.CallAsync("serve_add", new JsonObject { ["a"] = 2, ["b"] = 40 }) == "42", "mcp serve over http calls one");
            }).GetAwaiter().GetResult();
        }
        finally
        {
            stop.Cancel();
        }

        Check(serving.Wait(TimeSpan.FromSeconds(30)) && serving.Result == ExitCodes.Stopped, $"mcp serve over http stops with the host's stop (exit {(serving.IsCompleted ? serving.Result : -1)}): {error}");
        string all = error.ToString();
        Check(all.Contains("verbose: mcp: http tools/call serve_add: 200 in", StringComparison.Ordinal) && all.Contains("verbose: mcp: tool serve_add finished", StringComparison.Ordinal),
            $"mcp serve --verbose prints each request and tool call: {all}");
        Check(all.Contains("mcp requests", StringComparison.Ordinal) && all.Contains("mcp tool calls", StringComparison.Ordinal) && all.Contains("server/discover", StringComparison.Ordinal),
            $"mcp serve --profile writes the session's requests and tool calls at exit: {all}");
        Check(output.ToString().Length == 0, $"nothing on the output: {output}");
        var bad = ServeCli("mcp", "serve", tools, "--transport", "pigeon");
        Check(bad.Code == 2 && bad.Error.Contains("not registered", StringComparison.Ordinal), $"mcp serve --transport pigeon: {bad.Error}");
        var remote = ServeCli("mcp", "serve", tools, "--transport", "http", "--mcp-address", "0.0.0.0:0");
        Check(remote.Code == 2 && remote.Error.Contains("--mcp-allow-remote", StringComparison.Ordinal), $"mcp serve on 0.0.0.0 without --mcp-allow-remote: {remote.Error}");
    }

    // mcp serve over stdio: whatever --verbose and --profile print goes to the error output; the protocol's output carries
    // JSON-RPC lines only, and the command's output nothing.
    private static void CliMcpServeStdioVerbose()
    {
        string tools = typeof(CliServeTestTools).Assembly.Location;
        foreach (string[] flags in (string[][])[["--verbose", "--profile"], ["--quiet", "--verbose"]])
        {
            var toServer = new Pipe();
            var toClient = new Pipe();
            var previous = McpServeCommand.Streams;
            McpServeCommand.Streams = () => (toServer.Reader.AsStream(), toClient.Writer.AsStream());
            var error = new McpErrorWriter();
            var output = new StringWriter();
            string where = $"mcp serve {string.Join(' ', flags)}";
            try
            {
                var serving = Task.Run(() => CommandLine.Run(["mcp", "serve", tools, .. flags], output, error));
                var lines = new List<string>();
                Task.Run(async () =>
                {
                    var input = toServer.Writer.AsStream();
                    using var reader = new StreamReader(toClient.Reader.AsStream());
                    async Task<string> AskAsync(string line, string id)
                    {
                        await input.WriteAsync(Encoding.UTF8.GetBytes(line + "\n"));
                        string? answer;
                        while ((answer = await reader.ReadLineAsync()) is not null)
                        {
                            lines.Add(answer);
                            if (answer.Contains($"\"id\":{id}", StringComparison.Ordinal))
                            {
                                return answer;
                            }
                        }

                        return "";
                    }

                    await AskAsync(InitializeBody, "1");
                    await input.WriteAsync("""{"jsonrpc":"2.0","method":"notifications/initialized"}"""u8.ToArray().Append((byte)'\n').ToArray());
                    string sum = await AskAsync("""{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"serve_add","arguments":{"a":2,"b":40}}}""", "2");
                    Check(sum.Contains("42", StringComparison.Ordinal), $"{where}: the call answers: {sum}");
                    await toServer.Writer.CompleteAsync();
                }).GetAwaiter().GetResult();
                Check(serving.Wait(TimeSpan.FromSeconds(30)) && serving.Result == 0, $"{where} ends with its input: {error}");
                bool json = lines.All(l =>
                {
                    try
                    {
                        return JsonNode.Parse(l) is JsonObject o && (string?)o["jsonrpc"] == "2.0";
                    }
                    catch (System.Text.Json.JsonException)
                    {
                        return false;
                    }
                });
                Check(lines.Count >= 2 && json && output.ToString().Length == 0, $"{where}: the protocol's output holds JSON-RPC only ({lines.Count} lines), the output nothing: {string.Join("\n", lines)}{output}");
                string text = error.ToString();
                Check(text.Contains("verbose: mcp: stdio tools/call serve_add: in", StringComparison.Ordinal) && text.Contains("verbose: mcp: stdio session opened", StringComparison.Ordinal)
                      && text.Contains("verbose: mcp: stdio session closed", StringComparison.Ordinal), $"{where}: the events on the error output: {text}");
                Check(flags.Contains("--quiet") != text.Contains("Serving ", StringComparison.Ordinal), $"{where}: the status line {(flags.Contains("--quiet") ? "dropped" : "printed")}: {text}");
                Check(flags.Contains("--profile") == text.Contains("mcp requests", StringComparison.Ordinal), $"{where}: the profile report {(flags.Contains("--profile") ? "written" : "not asked")}: {text}");
            }
            finally
            {
                McpServeCommand.Streams = previous;
            }
        }
    }

    // The error output of a command running on another thread: read whole at any time, and a signal once the address is out.
    private sealed class McpErrorWriter : TextWriter
    {
        private readonly StringBuilder _text = new();

        public ManualResetEventSlim Address { get; } = new();

        public override Encoding Encoding => Encoding.UTF8;

        public override void Write(char value)
        {
            lock (_text)
            {
                _text.Append(value);
            }
        }

        public override void Write(string? value)
        {
            lock (_text)
            {
                _text.Append(value);
                if (_text.ToString().Contains("Bearer ", StringComparison.Ordinal))
                {
                    Address.Set();
                }
            }
        }

        public override void WriteLine(string? value) => Write(value + "\n");

        public override string ToString()
        {
            lock (_text)
            {
                return _text.ToString();
            }
        }
    }

    // Records the MCP events while it lives.
    private sealed class McpRecorder : ITelemetryHook, IDisposable
    {
        private readonly List<McpEvent> _events = [];
        private readonly IDisposable _subscription;

        public McpRecorder() => _subscription = Telemetry.Subscribe(this);

        public TelemetryLevel Levels => TelemetryLevel.Mcp;

        public void OnMcp(in McpEvent e)
        {
            lock (_events)
            {
                _events.Add(e);
            }
        }

        public McpEvent[] Events
        {
            get
            {
                lock (_events)
                {
                    return [.. _events];
                }
            }
        }

        public void Clear()
        {
            lock (_events)
            {
                _events.Clear();
            }
        }

        // Whether an event matching arrives within 15 s.
        public async Task<bool> WaitAsync(Func<McpEvent, bool> match)
        {
            long until = Environment.TickCount64 + 15_000;
            while (!Events.Any(match))
            {
                if (Environment.TickCount64 > until)
                {
                    return false;
                }

                await Task.Delay(10);
            }

            return true;
        }

        public void Dispose() => _subscription.Dispose();

        public override string ToString() => string.Join("\n", Events.Select(e => e.Describe()));
    }

    // Sessionless (2026-07-28): the SDK's client probes with server/discover; served as the revision defines, it connects in
    // that one request. Before, the probe was answered 400 and the client fell back to initialize and initialized (3 requests).
    private static async Task McpHostSessionlessAsync()
    {
        var (registry, armed) = McpTestTools();
        await using var host = await McpHost.StartAsync("http", registry);
        using var recorder = new McpRecorder();
        int connected;
        await using (var source = await McpTools.ConnectHttpAsync(new Uri(host.Address), host.Token))
        {
            // The server records an answered request after writing it, so the client may be connected a moment before
            // the event lands: wait for it (bounded) before counting.
            var waited = System.Diagnostics.Stopwatch.StartNew();
            while (!recorder.Events.Any(e => e.Transport == "http" && e.Kind is McpEventKind.RequestAnswered or McpEventKind.Refused) && waited.Elapsed < TimeSpan.FromSeconds(5))
            {
                await Task.Delay(5);
            }

            var connect = recorder.Events.Where(e => e.Transport == "http").ToList();
            int requests = connect.Count(e => e.Kind is McpEventKind.RequestAnswered or McpEventKind.Refused);
            Check(source.ServerName == "idrak" && connect.All(e => e.Kind != McpEventKind.Refused && e.Status < 400)
                  && connect.Any(e => e.Kind == McpEventKind.RequestAnswered && e.Method == "server/discover" && e.Status == 200 && e.Session is null)
                  && !connect.Any(e => e.Method is "initialize" or "notifications/initialized") && requests == 1,
                $"http: the client connects without a session in one request (no 400, no initialize): {requests} requests\n{recorder}");
            connected = requests;
            await CheckToolsAndCallAsync(source, "http sessionless");
            Check(recorder.Events.Where(e => e.Kind == McpEventKind.RequestAnswered && e.Transport == "http").All(e => e.Session is null && e.Status == 200),
                $"http: every request of the client went without a session: {recorder}");
            await CheckCancelAsync(source, armed, "http sessionless");
            Check(await source.CallAsync("echo", new JsonObject { ["text"] = "after" }) == "after", "http sessionless: calls go on after a cancelled one");
        }

        // A client pinned to a revision with sessions still gets one (initialize, then its Mcp-Session-Id).
        recorder.Clear();
        var pinned = new ModelContextProtocol.Client.HttpClientTransport(new ModelContextProtocol.Client.HttpClientTransportOptions
        {
            Endpoint = new Uri(host.Address),
            AdditionalHeaders = new Dictionary<string, string> { ["Authorization"] = "Bearer " + host.Token },
        });
        await using (var old = await McpTools.ConnectAsync(pinned, new ModelContextProtocol.Client.McpClientOptions { ProtocolVersion = "2025-11-25" }))
        {
            // What a connection cost before sessionless requests were served: the probe answered 400, then this handshake.
            int handshake = recorder.Events.Count(e => e.Transport == "http" && e.Kind is McpEventKind.RequestAnswered or McpEventKind.Refused);
            Console.WriteLine($"    mcp connect over http: {connected} request(s) now (server/discover); before: {1 + handshake} (server/discover answered 400, then {handshake} of the handshake)");

            Check(await old.CallAsync("echo", new JsonObject { ["text"] = "old" }) == "old"
                  && recorder.Events.Any(e => e.Kind == McpEventKind.SessionOpened && e.Session is { Length: > 0 }), $"http: a 2025-11-25 client gets a session: {recorder}");
        }

        // Sessionless GET and DELETE: 405 (Allow: POST); a sessionless cancellation finds its call by id.
        using var http = new HttpClient();
        var get = new HttpRequestMessage(HttpMethod.Get, host.Address);
        get.Headers.TryAddWithoutValidation("Authorization", "Bearer " + host.Token);
        get.Headers.TryAddWithoutValidation("MCP-Protocol-Version", "2026-07-28");
        var got = await http.SendAsync(get);
        Check(got.StatusCode == HttpStatusCode.MethodNotAllowed && got.Content.Headers.Allow.Contains("POST"), $"http: a sessionless GET is 405 with Allow: POST, not {(int)got.StatusCode}");
        var cancelled = armed();
        const string meta = """
            "_meta":{"io.modelcontextprotocol/protocolVersion":"2026-07-28","io.modelcontextprotocol/clientCapabilities":{},"io.modelcontextprotocol/clientInfo":{"name":"raw","version":"1"}}
            """;
        using var other = new HttpClient();
        var call = RawPostAsync(http, host.Address, host.Token, null, """{"jsonrpc":"2.0","id":"raw-1","method":"tools/call","params":{"name":"slow","arguments":{},""" + meta + "}}", default,
            protocol: "2026-07-28");
        Check(await recorder.WaitAsync(e => e.Kind == McpEventKind.ToolCallStarted && e.Tool == "slow"), "http: the sessionless slow call started");
        var cancel = await RawPostAsync(other, host.Address, host.Token, null, """{"jsonrpc":"2.0","method":"notifications/cancelled","params":{"requestId":"raw-1","reason":"test"}}""", default,
            protocol: "2026-07-28");
        Check(cancel.StatusCode == HttpStatusCode.Accepted, $"http: a sessionless notification is 202, not {(int)cancel.StatusCode}");
        Check(await Task.WhenAny(cancelled.Task, Task.Delay(TimeSpan.FromSeconds(15))) == cancelled.Task, "http: a sessionless notifications/cancelled did not stop its call");
        try
        {
            _ = await call;                                                    // a cancelled call is answered with nothing: its connection closes
        }
        catch (Exception x) when (x is HttpRequestException or IOException or TaskCanceledException)
        {
        }
        var unknown = await RawPostAsync(http, host.Address, host.Token, null, """{"jsonrpc":"2.0","id":5,"method":"initialize","params":{"protocolVersion":"2026-07-28","capabilities":{},"clientInfo":{"name":"raw","version":"1"}}}""",
            default, protocol: "2026-07-28");
        Check(unknown.StatusCode == HttpStatusCode.NotFound, $"http: initialize on 2026-07-28 is method-not-found, 404, not {(int)unknown.StatusCode}");

        // Over stdio the same client connects natively too: server/discover, no initialize.
        recorder.Clear();
        var (toServer, toClient) = (new Pipe(), new Pipe());
        await using var stdioHost = await McpHost.StartAsync("stdio", registry, new McpHostOptions { Input = toServer.Reader.AsStream(), Output = toClient.Writer.AsStream() });
        await using (var stdio = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream())))
        {
            Check(await stdio.CallAsync("echo", new JsonObject { ["text"] = "s" }) == "s", "stdio: a call");
            var methods = recorder.Events.Where(e => e.Transport == "stdio" && e.Kind == McpEventKind.RequestReceived).Select(e => e.Method).ToList();
            Check(methods.FirstOrDefault() == "server/discover" && !methods.Contains("initialize"), $"stdio: the client connects with server/discover alone: {string.Join(", ", methods)}");
        }
    }

    private static async Task McpHostSessionLimitAsync()
    {
        var (registry, _) = McpTestTools();
        using var http = new HttpClient();
        await using (var host = await McpHost.StartAsync("http", registry, new McpHostOptions { Token = "t", MaxSessions = 2 }))
        {
            string first = await RawInitializeAsync(http, host.Address, "t");
            await RawInitializeAsync(http, host.Address, "t");
            using var recorder = new McpRecorder();
            var refused = await RawPostAsync(http, host.Address, "t", null, InitializeBody, default);
            int retry = refused.Headers.RetryAfter?.Delta is { } delta ? (int)delta.TotalSeconds : -1;
            Check(refused.StatusCode == HttpStatusCode.ServiceUnavailable && retry >= 1, $"http: a third session over 2 is 503 with Retry-After ({retry} s), not {(int)refused.StatusCode}");
            Check(recorder.Events.Any(e => e.Kind == McpEventKind.Refused && e.Status == 503 && e.Reason?.Contains("most sessions", StringComparison.Ordinal) == true), $"http: the 503 is reported: {recorder}");
            Check((await RawPostAsync(http, host.Address, "t", first, """{"jsonrpc":"2.0","id":2,"method":"tools/list"}""", default)).StatusCode == HttpStatusCode.OK,
                "http: the open sessions go on at the limit");
            var delete = new HttpRequestMessage(HttpMethod.Delete, host.Address);
            delete.Headers.TryAddWithoutValidation("Authorization", "Bearer t");
            delete.Headers.TryAddWithoutValidation("Mcp-Session-Id", first);
            await http.SendAsync(delete);
            await RawInitializeAsync(http, host.Address, "t");                  // a session deleted frees its place
        }

        await using (var host = await McpHost.StartAsync("http", registry, new McpHostOptions { Token = "t", MaxSessions = 1, ReplaceIdleSessions = true }))
        {
            string first = await RawInitializeAsync(http, host.Address, "t");
            string second = await RawInitializeAsync(http, host.Address, "t");
            Check((await RawPostAsync(http, host.Address, "t", first, """{"jsonrpc":"2.0","id":2,"method":"tools/list"}""", default)).StatusCode == HttpStatusCode.NotFound
                  && (await RawPostAsync(http, host.Address, "t", second, """{"jsonrpc":"2.0","id":2,"method":"tools/list"}""", default)).StatusCode == HttpStatusCode.OK,
                "http: with ReplaceIdleSessions the idle session unused the longest made room for the new one");
        }

        await using (var unlimited = await McpHost.StartAsync("http", registry, new McpHostOptions { Token = "t", MaxSessions = 0 }))
        {
            for (int i = 0; i < 5; i++)
            {
                await RawInitializeAsync(http, unlimited.Address, "t");
            }
        }

        string? max = Environment.GetEnvironmentVariable(McpHostOptions.MaxSessionsVariable), replace = Environment.GetEnvironmentVariable(McpHostOptions.ReplaceIdleVariable);
        try
        {
            Environment.SetEnvironmentVariable(McpHostOptions.MaxSessionsVariable, "3");
            Environment.SetEnvironmentVariable(McpHostOptions.ReplaceIdleVariable, "1");
            var options = McpHostOptions.FromEnvironment();
            Check(options.MaxSessions == 3 && options.ReplaceIdleSessions, $"IDRAK_MCP_MAX_SESSIONS=3, IDRAK_MCP_REPLACE_IDLE=1: {options.MaxSessions}, {options.ReplaceIdleSessions}");
            Environment.SetEnvironmentVariable(McpHostOptions.MaxSessionsVariable, "many");
            string? error = null;
            try
            {
                McpHostOptions.FromEnvironment();
            }
            catch (ArgumentException e)
            {
                error = e.Message;
            }

            Check(error?.Contains("IDRAK_MCP_MAX_SESSIONS=many", StringComparison.Ordinal) == true, $"a bad IDRAK_MCP_MAX_SESSIONS is named: {error}");
            Check(new McpHostOptions().MaxSessions == McpTransportRequest.DefaultMaxSessions && !new McpHostOptions().ReplaceIdleSessions, "the defaults: 1024, no replacing");
        }
        finally
        {
            Environment.SetEnvironmentVariable(McpHostOptions.MaxSessionsVariable, max);
            Environment.SetEnvironmentVariable(McpHostOptions.ReplaceIdleVariable, replace);
        }
    }

    private static async Task McpHostTelemetryAsync()
    {
        var (registry, armed) = McpTestTools();
        using var recorder = new McpRecorder();
        await using (var host = await McpHost.StartAsync("http", registry, new McpHostOptions { Token = "t" }))
        {
            using var http = new HttpClient();
            string session = await RawInitializeAsync(http, host.Address, "t");
            await RawPostAsync(http, host.Address, "t", session, """{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"echo","arguments":{"text":"hi"}}}""", default);
            await RawPostAsync(http, host.Address, "t", session, """{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"fail","arguments":{}}}""", default);
            await RawPostAsync(http, host.Address, "wrong", null, InitializeBody, default);
            var delete = new HttpRequestMessage(HttpMethod.Delete, host.Address);
            delete.Headers.TryAddWithoutValidation("Authorization", "Bearer t");
            delete.Headers.TryAddWithoutValidation("Mcp-Session-Id", session);
            await http.SendAsync(delete);
            var e = recorder.Events;
            Check(e.Any(x => x.Kind == McpEventKind.SessionOpened && x.Session == session && x.Client == "raw 1" && x.Address?.StartsWith("127.0.0.1:", StringComparison.Ordinal) == true),
                $"http: the session opened, with its client and address: {recorder}");
            Check(e.Any(x => x.Kind == McpEventKind.RequestReceived && x.Method == "tools/call" && x.Tool == "echo" && x.BytesIn > 0), $"http: the call received: {recorder}");
            Check(e.Any(x => x.Kind == McpEventKind.RequestAnswered && x.Session == session && x.Method == "tools/call" && x.Tool == "echo" && x.Status == 200 && x.BytesIn > 0 && x.BytesOut > 0
                             && x.Duration > TimeSpan.Zero), $"http: the call answered with status, bytes and time: {recorder}");
            Check(e.Any(x => x.Kind == McpEventKind.ToolCallStarted && x.Tool == "echo") && e.Any(x => x.Kind == McpEventKind.ToolCallFinished && x.Tool == "echo" && x.Session == session),
                $"http: the tool call started and finished: {recorder}");
            Check(e.Any(x => x.Kind == McpEventKind.ToolCallFailed && x.Tool == "fail" && x.Reason?.Contains("fails", StringComparison.Ordinal) == true), $"http: the failed call, with why: {recorder}");
            Check(e.Any(x => x.Kind == McpEventKind.Refused && x.Status == 401 && x.Reason?.Contains("Bearer", StringComparison.Ordinal) == true), $"http: the 401, with why: {recorder}");
            Check(e.Any(x => x.Kind == McpEventKind.SessionClosed && x.Session == session && x.Reason == "deleted by its client"), $"http: the session closed, with why: {recorder}");

            // A cancelled tool call.
            await using var source = await McpTools.ConnectHttpAsync(new Uri(host.Address), "t");
            await CheckCancelAsync(source, armed, "http telemetry");
            Check(await recorder.WaitAsync(x => x.Kind == McpEventKind.ToolCallCancelled && x.Tool == "slow"), $"http: the cancelled tool call: {recorder}");
            Check(e.All(x => x.Describe().Length > 0), "every event describes itself");
        }

        recorder.Clear();
        var (toServer, toClient) = (new Pipe(), new Pipe());
        var stdioHost = await McpHost.StartAsync("stdio", registry, new McpHostOptions { Input = toServer.Reader.AsStream(), Output = toClient.Writer.AsStream() });
        await using (var stdio = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream())))
        {
            await stdio.CallAsync("echo", new JsonObject { ["text"] = "s" });
            await stdioHost.StopAsync();
        }

        await stdioHost.DisposeAsync();
        var s = recorder.Events.Where(x => x.Transport == "stdio").ToList();
        Check(s.Any(x => x.Kind == McpEventKind.SessionOpened && x.Client?.Length > 0) && s.Any(x => x.Kind == McpEventKind.RequestAnswered && x.Method == "tools/call" && x.Tool == "echo"
              && x.Duration > TimeSpan.Zero) && s.Any(x => x.Kind == McpEventKind.SessionClosed && x.BytesIn > 0 && x.BytesOut > 0),
            $"stdio: the session (opened with its client, closed with its bytes) and the answered call: {recorder}");
    }

    private const string InitializeBody = """{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"raw","version":"1"}}}""";

    private static async Task<HttpResponseMessage> RawPostAsync(HttpClient http, string address, string? token, string? session, string body, CancellationToken cancellationToken,
        string? origin = null, string? protocol = null)
    {
        var request = new HttpRequestMessage(HttpMethod.Post, address) { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        request.Headers.TryAddWithoutValidation("Accept", "application/json, text/event-stream");
        if (token is not null)
        {
            request.Headers.TryAddWithoutValidation("Authorization", "Bearer " + token);
        }

        if (session is not null)
        {
            request.Headers.TryAddWithoutValidation("Mcp-Session-Id", session);
        }

        if (origin is not null)
        {
            request.Headers.TryAddWithoutValidation("Origin", origin);
        }

        if (protocol is not null)
        {
            request.Headers.TryAddWithoutValidation("MCP-Protocol-Version", protocol);
        }

        var response = await http.SendAsync(request, HttpCompletionOption.ResponseContentRead, cancellationToken);
        return response;
    }

    // initialize and initialized over raw HTTP; the session id.
    private static async Task<string> RawInitializeAsync(HttpClient http, string address, string? token)
    {
        var response = await RawPostAsync(http, address, token, null, InitializeBody, default);
        Check(response.StatusCode == HttpStatusCode.OK && response.Content.Headers.ContentType?.MediaType == "text/event-stream", $"http: initialize answers {(int)response.StatusCode} {response.Content.Headers.ContentType}");
        string session = response.Headers.GetValues("Mcp-Session-Id").Single();
        string events = await response.Content.ReadAsStringAsync();
        Check(events.StartsWith("event: message", StringComparison.Ordinal) && events.Contains("\"serverInfo\"", StringComparison.Ordinal), $"http: initialize's event: {events}");
        var initialized = await RawPostAsync(http, address, token, session, """{"jsonrpc":"2.0","method":"notifications/initialized"}""", default);
        Check(initialized.StatusCode == HttpStatusCode.Accepted, $"http: a notification answers 202, not {(int)initialized.StatusCode}");
        return session;
    }
}
