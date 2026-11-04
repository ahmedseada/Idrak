// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.AspNetCore;
using Idrak.Cli;
using Idrak.Cli.Commands.Serve;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

// Image URLs through a proxy (plan 36, after G-1): the download is held to the address checked by a tunnel the server opens.
internal abstract partial class DesktopTests : Tests
{
    // A proxy in this process: records each request's first line (the CONNECT target) and answers with `Answer`, then closes.
    private sealed class RecordingProxy : IDisposable
    {
        private readonly TcpListener _listener = new(IPAddress.Loopback, 0);
        private readonly Task _loop;

        public RecordingProxy()
        {
            _listener.Start();
            _loop = Task.Run(Serve);
        }

        public int Port => ((IPEndPoint)_listener.LocalEndpoint).Port;

        public System.Collections.Concurrent.ConcurrentQueue<string> Requests { get; } = new();

        public volatile string Answer = "HTTP/1.1 200 Connection established\r\n\r\n";

        private async Task Serve()
        {
            while (true)
            {
                TcpClient client;
                try
                {
                    client = await _listener.AcceptTcpClientAsync();
                }
                catch (Exception e) when (e is SocketException or ObjectDisposedException)
                {
                    return;
                }

                using (client)
                {
                    try
                    {
                        var stream = client.GetStream();
                        var head = new StringBuilder();
                        var one = new byte[1];
                        while (!head.ToString().EndsWith("\r\n\r\n", StringComparison.Ordinal) && await stream.ReadAsync(one) > 0)
                        {
                            head.Append((char)one[0]);
                        }

                        Requests.Enqueue(head.ToString().Split("\r\n")[0]);
                        await stream.WriteAsync(Encoding.ASCII.GetBytes(Answer));
                    }
                    catch (IOException)
                    {
                    }
                }
            }
        }

        public void Dispose()
        {
            _listener.Stop();
            _loop.Wait(TimeSpan.FromSeconds(5));
        }
    }

    private static void ImageUrlProxyTunnel()
    {
        using var proxy = new RecordingProxy();
        var lookups = new System.Collections.Concurrent.ConcurrentDictionary<string, int>(StringComparer.Ordinal);

        // A name that is public the first time it is looked up and loopback after (DNS rebinding).
        Task<IPAddress[]> Rebinding(string host, CancellationToken token) =>
            Task.FromResult<IPAddress[]>(lookups.AddOrUpdate(host, 1, (_, n) => n + 1) == 1 ? [IPAddress.Parse("93.184.216.34")] : [IPAddress.Loopback]);

        (WebApplication App, string Root) Start(Action<ImageInputOptions> images)
        {
            var builder = WebApplication.CreateSlimBuilder();
            builder.Logging.ClearProviders();
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            var app = builder.Build();
            app.MapCompletionsApi("/v1", o => o.Images(images));
            app.StartAsync().GetAwaiter().GetResult();
            return (app, app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First());
        }

        var webProxy = new WebProxy(new Uri($"http://127.0.0.1:{proxy.Port}"));
        var (tunnelling, tunnelRoot) = Start(i => { i.AllowUrls = true; i.Proxy = webProxy; i.Resolver = Rebinding; });
        var (refusing, refusingRoot) = Start(i => { i.AllowUrls = true; i.Proxy = webProxy; i.RefuseThroughProxy = true; });
        using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(1) };
        string Ask(string root, string url)
        {
            var body = new JsonObject
            {
                ["model"] = "m",
                ["messages"] = new JsonArray(new JsonObject
                {
                    ["role"] = "user",
                    ["content"] = new JsonArray(new JsonObject { ["type"] = "image_url", ["image_url"] = new JsonObject { ["url"] = url } }, new JsonObject { ["type"] = "text", ["text"] = "x" }),
                }),
            };
            var response = http.PostAsync($"{root}/v1/chat/completions", new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json")).Result;
            string text = response.Content.ReadAsStringAsync().Result;
            string message = text.StartsWith('{') ? (string?)JsonNode.Parse(text)?["error"]?["message"] ?? text : text;
            return $"{(int)response.StatusCode} {message}";
        }

        try
        {
            // The tunnel goes to the address checked on the one lookup, never to what a second lookup would give.
            string tunnelled = Ask(tunnelRoot, "https://images.example/scan.png");
            Check(proxy.Requests.TryDequeue(out string? first) && first == "CONNECT 93.184.216.34:443 HTTP/1.1",
                $"the proxy is asked for a tunnel to the checked public address, not the name: '{first}' ({tunnelled})");
            Check(lookups.GetValueOrDefault("images.example") == 1, $"the host is looked up once ({lookups.GetValueOrDefault("images.example")}), so a rebinding name cannot change the address");
            Check(tunnelled.StartsWith("400 ", StringComparison.Ordinal) && tunnelled.Contains("https://images.example/scan.png", StringComparison.Ordinal),
                $"the fake proxy's empty tunnel fails the TLS handshake: a 400 naming the URL: {tunnelled}");

            // A proxy that refuses a tunnel to an address: the error names the proxy and its status, nothing else it sent.
            proxy.Answer = "HTTP/1.1 403 Forbidden\r\nX-Detail: SECRET-PROXY-TEXT\r\n\r\n";
            lookups.Clear();
            string refused = Ask(tunnelRoot, "https://images.example/scan.png");
            Check(refused.StartsWith("400 ", StringComparison.Ordinal) && refused.Contains("answered HTTP 403", StringComparison.Ordinal)
                  && refused.Contains($"127.0.0.1:{proxy.Port}", StringComparison.Ordinal) && !refused.Contains("SECRET-PROXY-TEXT", StringComparison.Ordinal),
                $"a proxy that refuses the tunnel: a 400 naming the proxy and its status: {refused}");

            // Plain http cannot be held to the checked address through a proxy; RefuseThroughProxy refuses every URL it would carry.
            lookups.Clear();
            string plain = Ask(tunnelRoot, "http://images.example/scan.png");
            Check(plain.StartsWith("400 ", StringComparison.Ordinal) && plain.Contains("plain http", StringComparison.Ordinal) && plain.Contains("https", StringComparison.Ordinal),
                $"a plain http URL through a proxy is refused, asking for https: {plain}");
            string never = Ask(refusingRoot, "https://images.example/scan.png");
            Check(never.StartsWith("400 ", StringComparison.Ordinal) && never.Contains("RefuseThroughProxy", StringComparison.Ordinal) && never.Contains("--refuse-proxied-image-urls", StringComparison.Ordinal),
                $"RefuseThroughProxy refuses a URL the proxy would carry, naming the setting: {never}");
            Check(proxy.Requests.All(r => r == "CONNECT 93.184.216.34:443 HTTP/1.1"),
                $"every tunnel the proxy was asked for went to the checked address: {string.Join(" | ", proxy.Requests)}");
        }
        finally
        {
            foreach (var app in new[] { tunnelling, refusing })
            {
                app.StopAsync().GetAwaiter().GetResult();
                ((IAsyncDisposable)app).DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
        }

        // serve: the flag or its variable sets it.
        string folder = TempFolder();
        var serve = CommandTable.All.First(c => c.Name == "serve");
        string? saved = Environment.GetEnvironmentVariable(ServeOptions.RefuseProxiedImageUrlsVariable);
        ServeSettings Read(params string[] flags)
        {
            var options = new Dictionary<string, List<string>> { ["--config"] = [Path.Combine(folder, "config.json")], ["--cache"] = [folder] };
            using var context = new CommandContext(serve, [], options, [.. flags], new StringWriter(), new StringWriter());
            return ServeOptions.ReadSettings(context);
        }

        try
        {
            Environment.SetEnvironmentVariable(ServeOptions.RefuseProxiedImageUrlsVariable, null);
            Check(!Read("--allow-image-urls").RefuseProxiedImageUrls && Read("--allow-image-urls", "--refuse-proxied-image-urls").RefuseProxiedImageUrls,
                "--refuse-proxied-image-urls turns it on, off by default");
            Environment.SetEnvironmentVariable(ServeOptions.RefuseProxiedImageUrlsVariable, "1");
            Check(Read("--allow-image-urls").RefuseProxiedImageUrls, $"{ServeOptions.RefuseProxiedImageUrlsVariable}=1 turns it on");
        }
        finally
        {
            Environment.SetEnvironmentVariable(ServeOptions.RefuseProxiedImageUrlsVariable, saved);
            Directory.Delete(folder, recursive: true);
        }

        Check(Idrak.Cli.Shared.EnvironmentVariables.Find(ServeOptions.RefuseProxiedImageUrlsVariable) is { } variable && variable.Commands.Contains("serve"),
            $"{ServeOptions.RefuseProxiedImageUrlsVariable} is in the tool's table");
    }
}
