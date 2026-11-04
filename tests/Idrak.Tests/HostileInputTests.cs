// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Text;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.AspNetCore;
using Idrak.Cli;
using Idrak.Cli.Commands.Serve;
using Idrak.Gpu.Hip;
using Idrak.Models;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

// Inputs that can be hostile (plan 36, G-1 3a): image URLs that reach the server's own network, GGUF headers that size
// allocations from untrusted counts, cached HIP code objects changed on disk, and a MatMulBias weight of the wrong rank.
internal abstract partial class DesktopTests : Tests
{
    private static readonly (string Name, Action<Device> Run)[] HostileInputGroup =
    [
        ("cli images: an image URL that reaches a loopback, private, link-local, multicast or reserved address (a literal or a name) is refused naming the URL and the address; AllowPrivateUrls downloads it and follows redirects; a failed download names the status, never the remote body; serve's --allow-private-image-urls and IDRAK_SERVE_PRIVATE_IMAGE_URLS set it", d => { CpuOnly(d); ImageUrlAddresses(); }),
        ("cli images: through a proxy an https image URL goes by a tunnel to the address checked (CONNECT to it, never a second lookup); a proxy's refusal names its status; a plain http URL through a proxy and any URL under RefuseThroughProxy are refused; serve's --refuse-proxied-image-urls and IDRAK_SERVE_REFUSE_PROXIED_IMAGE_URLS set it", d => { CpuOnly(d); ImageUrlProxyTunnel(); }),
        ("gguf: a hostile header fails naming the cause before anything is sized by it: too many dimensions, a duplicate tensor name, a tensor count or a string longer than the file", d => { CpuOnly(d); GgufHostileHeaders(); }),
        ("hip cache: a kept code object carries its hash; one changed on disk, cut short or without a hash is refused, deleted and named, never loaded", d => { CpuOnly(d); HipCacheTampered(); }),
        ("matmul bias: a weight that is not a matrix (rank 1) takes MatMul's path and its shape message, not an index error; a matrix weight gives x·w + bias", MatMulBiasShapes),
    ];

    private static void ImageUrlAddresses()
    {
        int hits = 0;
        byte[] png = File.ReadAllBytes(TestData("vlm/image.png"));
        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        var app = builder.Build();
        app.MapGet("/i.png", () =>
        {
            Interlocked.Increment(ref hits);
            return Results.File(png, "image/png");
        });
        app.MapGet("/r", () => Results.Redirect("/i.png"));
        app.MapGet("/gone", () => Results.Text("SECRET-REMOTE-BODY", "text/plain", statusCode: 404));
        app.MapCompletionsApi("/public", o => o.Images(i => i.AllowUrls = true));
        app.StartAsync().GetAwaiter().GetResult();

        // The API that allows private addresses on a server of its own (an app maps the API once: its endpoints are named).
        var anyBuilder = WebApplication.CreateSlimBuilder();
        anyBuilder.Logging.ClearProviders();
        anyBuilder.WebHost.UseUrls("http://127.0.0.1:0");
        var anyApp = anyBuilder.Build();
        anyApp.MapCompletionsApi("/any", o => o.Images(i => { i.AllowUrls = true; i.AllowPrivateUrls = true; }));
        anyApp.StartAsync().GetAwaiter().GetResult();
        try
        {
            string root = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            string anyRoot = anyApp.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            int port = new Uri(root).Port;
            using var http = new HttpClient { Timeout = TimeSpan.FromMinutes(1) };
            (HttpStatusCode Status, string Message) Ask(string api, string url)
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
                var response = http.PostAsync($"{(api == "any" ? anyRoot : root)}/{api}/chat/completions", new StringContent(body.ToJsonString(), Encoding.UTF8, "application/json")).Result;
                string text = response.Content.ReadAsStringAsync().Result;
                return (response.StatusCode, text.StartsWith('{') ? (string?)JsonNode.Parse(text)?["error"]?["message"] ?? text : text);
            }

            // Refused before any connection: every non-public kind, as a literal, a name, an IPv6 or an IPv4-mapped address.
            foreach (var (url, kind) in new[]
                     {
                         ($"{root}/i.png", "loopback"), ($"http://localhost:{port}/i.png", "loopback"), ($"http://[::1]:{port}/i.png", "loopback"),
                         ($"http://[::ffff:127.0.0.1]:{port}/i.png", "loopback"), ("http://10.1.2.3/a.png", "private"), ("http://172.16.0.1/a.png", "private"),
                         ("http://192.168.1.1/a.png", "private"), ("http://169.254.169.254/latest/meta-data", "link-local"), ("http://100.64.0.1/a.png", "carrier-grade NAT"),
                         ("http://224.0.0.1/a.png", "multicast"), ($"http://0.0.0.0:{port}/i.png", "unspecified"), ("http://240.0.0.1/a.png", "reserved"),
                         ("http://[fe80::1]/a.png", "link-local"), ("http://[fd00::1]/a.png", "private"), ("http://[ff02::1]/a.png", "multicast"),
                     })
            {
                var (status, message) = Ask("public", url);
                Check(status == HttpStatusCode.BadRequest && message.Contains(url, StringComparison.Ordinal) && message.Contains($"a {kind} address", StringComparison.Ordinal)
                      && message.Contains("AllowPrivateUrls", StringComparison.Ordinal), $"{url}: refused naming the URL and a {kind} address, not {(int)status}: {message}");
            }

            Check(hits == 0, $"no refused URL was fetched ({hits} requests reached the server)");

            // Allowed: the image downloaded (the request then finds no model: 404), a redirect followed; a failure names its status only.
            var allowed = Ask("any", $"{root}/i.png");
            Check(allowed.Status == HttpStatusCode.NotFound && hits == 1, $"AllowPrivateUrls downloads from a loopback address ({hits} requests): {(int)allowed.Status} {allowed.Message}");
            var redirected = Ask("any", $"{root}/r");
            Check(redirected.Status == HttpStatusCode.NotFound && hits == 2, $"a redirect is followed to the image ({hits} requests): {(int)redirected.Status} {redirected.Message}");
            var gone = Ask("any", $"{root}/gone");
            Check(gone.Status == HttpStatusCode.BadRequest && gone.Message.Contains("HTTP 404", StringComparison.Ordinal) && gone.Message.Contains($"{root}/gone", StringComparison.Ordinal)
                  && !gone.Message.Contains("SECRET-REMOTE-BODY", StringComparison.Ordinal), $"a failed download names the URL and the status, never the remote body: {gone.Message}");
        }
        finally
        {
            foreach (var running in new[] { app, anyApp })
            {
                running.StopAsync().GetAwaiter().GetResult();
                ((IAsyncDisposable)running).DisposeAsync().AsTask().GetAwaiter().GetResult();
            }
        }

        // serve's setting: the flag, or its variable, with --allow-image-urls; the flag alone is a usage error.
        string folder = TempFolder();
        var serve = CommandTable.All.First(c => c.Name == "serve");
        string? saved = Environment.GetEnvironmentVariable(ServeOptions.PrivateImageUrlsVariable);
        ServeSettings Read(params string[] flags)
        {
            var options = new Dictionary<string, List<string>> { ["--config"] = [Path.Combine(folder, "config.json")], ["--cache"] = [folder] };
            using var context = new CommandContext(serve, [], options, [.. flags], new StringWriter(), new StringWriter());
            return ServeOptions.ReadSettings(context);
        }

        try
        {
            Environment.SetEnvironmentVariable(ServeOptions.PrivateImageUrlsVariable, null);
            Check(!Read("--allow-image-urls").AllowPrivateImageUrls && Read("--allow-image-urls", "--allow-private-image-urls").AllowPrivateImageUrls,
                "--allow-private-image-urls turns it on, off by default");
            bool refused = false;
            try
            {
                Read("--allow-private-image-urls");
            }
            catch (UsageException e) when (e.Message.Contains("--allow-image-urls", StringComparison.Ordinal))
            {
                refused = true;
            }

            Check(refused, "--allow-private-image-urls without --allow-image-urls is a usage error naming it");
            Environment.SetEnvironmentVariable(ServeOptions.PrivateImageUrlsVariable, "1");
            Check(Read("--allow-image-urls").AllowPrivateImageUrls && !Read().AllowPrivateImageUrls,
                $"{ServeOptions.PrivateImageUrlsVariable}=1 turns it on with --allow-image-urls (and nothing without)");
        }
        finally
        {
            Environment.SetEnvironmentVariable(ServeOptions.PrivateImageUrlsVariable, saved);
            Directory.Delete(folder, recursive: true);
        }

        Check(Idrak.Cli.Shared.EnvironmentVariables.Find(ServeOptions.PrivateImageUrlsVariable) is { } variable && variable.Commands.Contains("serve"),
            $"{ServeOptions.PrivateImageUrlsVariable} is in the tool's table");
    }

    // A GGUF (version 3, no metadata) with the tensor entries the writer adds; the bytes after them stand for the data.
    private static byte[] HostileGguf(ulong tensorCount, Action<BinaryWriter> tensors)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true))
        {
            writer.Write("GGUF"u8);
            writer.Write(3u);
            writer.Write(tensorCount);
            writer.Write(0UL);
            tensors(writer);
            writer.Write(new byte[64]);
        }

        return stream.ToArray();
    }

    private static void GgufTensorEntry(BinaryWriter writer, string name, uint dimensions, ulong[] sizes, ulong offset)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(name);
        writer.Write((ulong)bytes.Length);
        writer.Write(bytes);
        writer.Write(dimensions);
        foreach (ulong size in sizes)
        {
            writer.Write(size);
        }

        writer.Write(0u);                                                    // F32
        writer.Write(offset);
    }

    private static void GgufHostileHeaders()
    {
        string folder = TempFolder();
        try
        {
            string? Refusal(string name, byte[] bytes)
            {
                string path = Path.Combine(folder, name + ".gguf");
                File.WriteAllBytes(path, bytes);
                try
                {
                    GgufFile.Open(path).Dispose();
                    return null;
                }
                catch (InvalidDataException e)
                {
                    return e.Message;
                }
            }

            string? good = Refusal("good", HostileGguf(2, w => { GgufTensorEntry(w, "a", 1, [2], 0); GgufTensorEntry(w, "b", 1, [2], 32); }));
            Check(good is null, $"a well-formed header with two tensors opens: {good}");

            foreach (uint dimensions in new uint[] { 5u, int.MaxValue, uint.MaxValue })
            {
                string? message = Refusal($"dims-{dimensions}", HostileGguf(1, w => GgufTensorEntry(w, "blk.0.w", dimensions, [], 0)));
                Check(message is not null && message.Contains("'blk.0.w'", StringComparison.Ordinal) && message.Contains($"{dimensions} dimensions", StringComparison.Ordinal)
                      && message.Contains("at most 4", StringComparison.Ordinal), $"{dimensions} dimensions: refused naming the tensor, the count and the maximum, before a shape is sized: {message ?? "opened"}");
            }

            string? twice = Refusal("twice", HostileGguf(2, w => { GgufTensorEntry(w, "blk.0.w", 1, [2], 0); GgufTensorEntry(w, "blk.0.w", 1, [2], 32); }));
            Check(twice is not null && twice.Contains("'blk.0.w'", StringComparison.Ordinal) && twice.Contains("twice", StringComparison.Ordinal),
                $"a duplicate tensor name is refused naming it, not overwritten: {twice ?? "opened"}");

            string? many = Refusal("many", HostileGguf(1UL << 40, w => GgufTensorEntry(w, "a", 1, [2], 0)));
            Check(many is not null && many.Contains("1099511627776 tensors", StringComparison.Ordinal), $"a tensor count larger than the file is refused naming it: {many ?? "opened"}");

            string? longName = Refusal("long-name", HostileGguf(1, w => { w.Write(1UL << 20); w.Write("abc"u8); }));
            Check(longName is not null && longName.Contains("a string of 1048576 bytes", StringComparison.Ordinal), $"a name longer than the file is refused naming its length: {longName ?? "opened"}");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void HipCacheTampered()
    {
        string folder = TempFolder();
        try
        {
            var cache = new HipKernelCache(folder);
            string key = HipKernelCache.Key("source", ["format=5"], "gfx1100", "device", "runtime 1");
            byte[] code = [.. Enumerable.Range(0, 1000).Select(i => (byte)(i * 7))];
            cache.Store(key, "gfx1100", code);
            string path = cache.PathOf(key, "gfx1100")!;
            Check(cache.Load(key, "gfx1100") is { } back && back.SequenceEqual(code) && cache.LastRefusal is null && new FileInfo(path).Length > code.Length,
                "a stored code object comes back, kept with its hash");

            byte[] kept = File.ReadAllBytes(path);
            kept[^1] ^= 0xFF;                                                    // one byte of the code object changed on disk
            File.WriteAllBytes(path, kept);
            Check(cache.Load(key, "gfx1100") is null && cache.LastRefusal is { } changed && changed.Contains("do not match", StringComparison.Ordinal)
                  && changed.Contains(path, StringComparison.Ordinal) && !File.Exists(path), $"a changed code object is refused naming the file, and deleted: {cache.LastRefusal}");

            cache.Store(key, "gfx1100", code);
            File.WriteAllBytes(path, File.ReadAllBytes(path)[..10]);
            Check(cache.Load(key, "gfx1100") is null && cache.LastRefusal is not null && !File.Exists(path), $"a file cut short is refused and deleted: {cache.LastRefusal}");

            File.WriteAllBytes(path, code);                                      // a code object kept without its hash (the earlier format)
            Check(cache.Load(key, "gfx1100") is null && cache.LastRefusal is { } bare && bare.Contains("no content hash", StringComparison.Ordinal) && !File.Exists(path),
                $"a file without a hash is refused and deleted: {cache.LastRefusal}");

            cache.Store(key, "gfx1100", code);
            Check(cache.Load(key, "gfx1100") is { } again && again.SequenceEqual(code) && cache.LastRefusal is null, "stored again, it loads");
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    private static void MatMulBiasShapes(Device device)
    {
        using var x = Tensor.From([1f, 2f, 3f, 4f, 5f, 6f], [2, 3], device);
        using var vector = Tensor.From([1f, 1f, 1f], [3], device);
        using var bias = Tensor.From([10f, 20f], [2], device);
        string? message = null;
        try
        {
            using var y = Tensor.MatMulBias(x, vector, bias);
        }
        catch (ArgumentException e)
        {
            message = e.Message;
        }

        Check(message is not null && message.Contains("MatMul", StringComparison.Ordinal) && message.Contains("[3]", StringComparison.Ordinal),
            $"a rank-1 weight takes MatMul's path and its shape message (not an index error): {message ?? "no error"}");

        using var w = Tensor.From([1f, 0f, 0f, 1f, 1f, 1f], [3, 2], device);
        using var product = Tensor.MatMulBias(x, w, bias);
        float[] got = product.ToArray();
        Check(got.SequenceEqual([1f + 3 + 10, 2 + 3 + 20f, 4 + 6 + 10f, 5 + 6 + 20f]), $"a matrix weight gives x·w + bias: [{string.Join(", ", got)}]");
    }
}
