// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.IO.Pipelines;
using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Idrak.Cli;
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
        ("mcp host: stdio over pipes lists the tools, calls one, cancels a long call and stops", d => { if (d == Device.Cpu) Run(McpHostStdioAsync); }),
        ("mcp host: http on loopback: McpTools.ConnectHttpAsync lists, calls, cancels a long call (client cancel and disconnect), ends its session, stops", d => { if (d == Device.Cpu) Run(McpHostHttpAsync); }),
        ("mcp host: http refuses a missing or wrong token (401), a page off this machine (403), an unknown session (404), a non-loopback bind unless allowed; the token can be turned off", d => { if (d == Device.Cpu) Run(McpHostHttpRefusalsAsync); }),
        ("mcp host: an unknown transport fails with \"not registered\" naming stdio and http; option keys a transport does not take are refused", d => { if (d == Device.Cpu) Run(McpHostRegistryAsync); }),
        ("mcp host: per-call cost of a trivial tool over http and over stdio, cold and warm apart (printed)", d => { if (d == Device.Cpu) Run(McpHostOverheadAsync); }),
        ("cli serve: mcp serve --transport http prints its address and token; a client with the token lists and calls; Ctrl+C stops it", d => { if (d == Device.Cpu) CliMcpServeHttp(); }),
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
        Check(tools.Select(t => t.Definition.Name).Order().SequenceEqual(["echo", "slow"]), $"{where}: tools {string.Join(", ", tools.Select(t => t.Definition.Name))}");
        Check(await source.CallAsync("echo", new JsonObject { ["text"] = "سلام" }) == "سلام", $"{where}: a call answers");
    }

    // A long call the client cancels: the server's tool sees its token cancelled (over http the SDK's client closes the
    // request's connection, which the server takes as the call's cancellation).
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
        Check(host.Address == "stdio" && host.Token is null && host.Transport == "stdio" && host.ToolCount == 2, $"stdio host: {host.Address}, {host.Token}, {host.ToolCount} tools");
        await using (var source = await McpTools.ConnectAsync(new StreamClientTransport(toServer.Writer.AsStream(), toClient.Reader.AsStream())))
        {
            Check(source.ServerName == "idrak", $"stdio: server name {source.ServerName}");
            await CheckToolsAndCallAsync(source, "stdio");
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

        double httpCold = await TimeAsync(http, 1), stdioCold = await TimeAsync(stdio, 1);
        await TimeAsync(http, warmup);
        await TimeAsync(stdio, warmup);
        double httpWarm = 0, stdioWarm = 0;
        long httpBytes = 0, stdioBytes = 0;
        for (int b = 0; b < blocks; b++)
        {
            long before = GC.GetTotalAllocatedBytes(precise: true);
            httpWarm += await TimeAsync(http, block);
            long middle = GC.GetTotalAllocatedBytes(precise: true);
            stdioWarm += await TimeAsync(stdio, block);
            stdioBytes += GC.GetTotalAllocatedBytes(precise: true) - middle;
            httpBytes += middle - before;
        }

        int calls = blocks * block;
        Console.WriteLine($"    mcp per call (echo, client and server in process): http cold {httpCold:F0} us, warm {httpWarm / calls:F1} us, {httpBytes / calls} B; "
                          + $"stdio (pipes) cold {stdioCold:F0} us, warm {stdioWarm / calls:F1} us, {stdioBytes / calls} B");
        Check(httpWarm > 0 && stdioWarm > 0, "the calls were timed");
    }

    private static void CliMcpServeHttp()
    {
        string tools = typeof(CliServeTestTools).Assembly.Location;
        var error = new McpErrorWriter();
        using var stop = new CancellationTokenSource();
        var serving = Task.Run(() => CommandLine.Run(["mcp", "serve", tools, "--transport", "http", "--mcp-token", "cli-token", "--quiet"], new StringWriter(), error,
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
        var bad = ServeCli("mcp", "serve", tools, "--transport", "pigeon");
        Check(bad.Code == 2 && bad.Error.Contains("not registered", StringComparison.Ordinal), $"mcp serve --transport pigeon: {bad.Error}");
        var remote = ServeCli("mcp", "serve", tools, "--transport", "http", "--mcp-address", "0.0.0.0:0");
        Check(remote.Code == 2 && remote.Error.Contains("--mcp-allow-remote", StringComparison.Ordinal), $"mcp serve on 0.0.0.0 without --mcp-allow-remote: {remote.Error}");
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

    private const string InitializeBody = """{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"raw","version":"1"}}}""";

    private static async Task<HttpResponseMessage> RawPostAsync(HttpClient http, string address, string? token, string? session, string body, CancellationToken cancellationToken,
        string? origin = null)
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
