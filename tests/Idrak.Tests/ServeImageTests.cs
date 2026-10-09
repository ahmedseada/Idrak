// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Net;
using System.Net.Http.Headers;
using System.Net.Sockets;
using System.Text;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Idrak;
using Idrak.AspNetCore;
using Idrak.Cli;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Generation.Abstractions;
using Idrak.Inference;
using Idrak.Models;
using Idrak.Nlp;

// Images in chat over HTTP (plan 11, phase 8) with the tiny Gemma 3 of phase 0, in-process on a loopback port: the
// engine hosting a vision-language chat model (GenerativeModelBuilder.Images), /v1/chat/completions with image_url data
// URLs (and http URLs when allowed), the multipart /v1/chat/upload, /api/chat's image parts, streamed and not, the
// greedy answer equal to transformers' 20 tokens; and `idrak serve` with the same answers as `idrak run --image`,
// grayscale per request and per server, and its limits. Text-only models, bad base64, unsupported formats, http URLs
// without the option and requests over the limits are refused.
internal static partial class Tests
{
    // A 1 x 1 GIF: a real image no registered codec reads.
    private static readonly byte[] TinyGif = Convert.FromBase64String("R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7");

    private static (int[] Tokens, string Text) VlmReference()
    {
        var facts = JsonNode.Parse(File.ReadAllText(TestData("vlm/manifest.json")))!["facts"]!;
        int[] tokens = [.. facts["generate"]!["new_tokens"]!.AsArray().Select(n => (int)n!)];
        return (tokens, BpeTokenizer.Load(VlmModel).Decode(tokens).Trim());
    }

    // The greedy request of phase 0's reference: a system message, then the image and the question.
    private static string VlmChatBody(string imagePart, bool stream, string? model = null, string extra = "") =>
        $$"""{{{(model is null ? "" : $"\"model\":\"{model}\",")}}"messages":[{"role":"system","content":"Read the scan."},{"role":"user","content":[{{imagePart}},{"type":"text","text":"What is in this image?"}]}],"temperature":0,"max_tokens":20,"stream":{{(stream ? "true" : "false")}}{{extra}}}""";

    private static string DataUrlPart(string url) => $$$"""{"type":"image_url","image_url":{"url":"{{{url}}}"}}""";

    // The text of a streamed answer: each chunk's delta content, after checking the framing (data lines, then [DONE]).
    private static string StreamedText(string sse, string what)
    {
        var events = sse.Split("\n\n", StringSplitOptions.RemoveEmptyEntries);
        Check(events.Length > 2 && events.All(e => e.StartsWith("data: ", StringComparison.Ordinal)) && events[^1] == "data: [DONE]", $"{what}: SSE framing: {sse}");
        var text = new StringBuilder();
        foreach (var e in events[..^1])
        {
            var json = JsonNode.Parse(e["data: ".Length..])!;
            Check((string?)json["object"] == "chat.completion.chunk", $"{what}: a chunk: {e}");
            if (json["choices"] is JsonArray { Count: > 0 } choices && (string?)choices[0]!["delta"]?["content"] is { } piece)
            {
                text.Append(piece);
            }
        }

        return text.ToString();
    }

    private static MultipartFormDataContent UploadForm(string png, bool stream, params (string Key, string Value)[] fields)
    {
        var form = new MultipartFormDataContent();
        var file = new ByteArrayContent(File.ReadAllBytes(png));
        file.Headers.ContentType = new MediaTypeHeaderValue("image/png");
        form.Add(file, "image", Path.GetFileName(png));
        form.Add(new StringContent("Read the scan."), "system");
        form.Add(new StringContent("What is in this image?"), "prompt");
        form.Add(new StringContent("0"), "temperature");
        form.Add(new StringContent("20"), "max_tokens");
        form.Add(new StringContent(stream ? "true" : "false"), "stream");
        foreach (var (key, value) in fields)
        {
            form.Add(new StringContent(value), key);
        }

        return form;
    }

    // Records the greedy tokens of every generation while it runs (the default sampler wrapped).
    private static T WithRecordedTokens<T>(List<int> sampled, Func<T> run)
    {
        var library = TokenSamplers.Default(TokenSamplers.DefaultName)!;
        TokenSamplers.Register(TokenSamplers.DefaultName, request => new RecordingSampler(library(request), sampled));
        try
        {
            return run();
        }
        finally
        {
            TokenSamplers.Unregister(TokenSamplers.DefaultName);
        }
    }

    // Idrak.AspNetCore over the engine: a vision-language chat model of one's own (the library's encoder, no CLI).
    private static void AspNetCoreImages(Device device)
    {
        var (tokens, expected) = VlmReference();
        string png = TestData("vlm/image.png");
        string dataUrl = ChatImage.FromFile(png).ToDataUrl();
        var template = JinjaChatTemplate.Load(VlmModel, BpeTokenizer.Load(VlmModel))!;
        PretrainedModel? loaded = null;
        int encoders = 0, disposed = 0;

        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddIdrak().Configure((_, engine) => engine
            .ChatModel("vlm", () =>
            {
                loaded = PretrainedModel.Load(VlmModel, new PretrainedOptions { Device = device });
                return loaded.CreateGenerator();
            }, b => b.Template(template).Images(generator =>
            {
                var vision = loaded!.Vision!;
                var encoder = vision.CreateEncoder(loaded.Device, vision.Preprocessor());
                encoders++;
                return new ChatImages(vision.ImageTokens, ImagePromptFormats.Get((string)loaded.Config["model_type"]!),
                    images => encoder.Encode([.. images.Select(ChatImageDecoder.Decode)])) { Owner = new Disposer(() => { encoder.Dispose(); disposed++; }) };
            }))
            .ChatModel("text", () => { var (m, t) = TinyLanguageModel(device); return new TextGenerator(m, t, 32); }));
        var app = builder.Build();
        app.MapCompletionsApi("/v1", o => o.Images(i => { i.MaxImages = 2; i.AllowUrls = true; }));
        app.MapChatApi("/api", "vlm", o => o.Tools(ToolExecution.Client).Images(i => i.MaxImages = 2));
        app.MapGet("/files/image.png", () => Results.File(File.ReadAllBytes(png), "image/png"));
        app.MapGet("/files/missing.png", () => Results.NotFound());
        app.StartAsync().GetAwaiter().GetResult();
        try
        {
            string url = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            using var http = new HttpClient { BaseAddress = new Uri(url), Timeout = TimeSpan.FromMinutes(5) };
            var engine = app.Services.GetRequiredService<InferenceEngine>();
            Check(engine.Model<IChatModel>("vlm").PartKinds.SetEquals([ChatParts.Text, ChatParts.Image]) && engine.Model<IChatModel>("text").PartKinds.SetEquals(ChatParts.TextOnly),
                "the engine's chat model takes images when given Images and a template that renders them (before it loads)");
            HttpResponseMessage Post(string path, string body) => http.PostAsync(path, new StringContent(body, Encoding.UTF8, "application/json")).Result;
            string Text(HttpResponseMessage response) => response.Content.ReadAsStringAsync().Result;

            // /v1 with a data URL: not streamed, then streamed; the greedy tokens are transformers'.
            var sampled = new List<int>();
            var single = WithRecordedTokens(sampled, () => Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), false, "vlm")));
            string singleText = Text(single);
            var json = JsonNode.Parse(singleText)!;
            string answer = (string?)json["choices"]?[0]?["message"]?["content"] ?? "";
            Check(single.StatusCode == HttpStatusCode.OK && sampled.SequenceEqual(tokens), $"/v1 with an image: tokens {string.Join(" ", sampled)}, expected {string.Join(" ", tokens)}: {singleText}");
            Check(answer.Length == expected.Length && (int)json["usage"]!["prompt_tokens"]! == 38 && (int)json["usage"]!["completion_tokens"]! == 20
                  && (string?)json["choices"]![0]!["finish_reason"] == "length", $"/v1 answer '{answer}', expected '{expected}': {singleText}");
            sampled.Clear();
            string streamed = WithRecordedTokens(sampled, () => StreamedText(Text(Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), true, "vlm"))), "/v1 streamed"));
            Check(streamed == answer && sampled.SequenceEqual(tokens), $"/v1 streamed: '{streamed}', not streamed '{answer}'");

            // Requests at the same time: the engine answers them one after another on its copy, each as alone.
            var together = Enumerable.Range(0, 3).Select(i => http.PostAsync("/v1/chat/completions",
                new StringContent(VlmChatBody(DataUrlPart(dataUrl), i == 1, "vlm"), Encoding.UTF8, "application/json"))).ToArray();
            Task.WaitAll(together);
            Check((string?)JsonNode.Parse(Text(together[0].Result))!["choices"]![0]!["message"]!["content"] == answer
                  && StreamedText(Text(together[1].Result), "concurrent streamed") == answer
                  && (string?)JsonNode.Parse(Text(together[2].Result))!["choices"]![0]!["message"]!["content"] == answer, "three requests at once");

            // An http URL, allowed here (downloaded from this same server); a failed download is a 400.
            var fetched = JsonNode.Parse(Text(Post("/v1/chat/completions", VlmChatBody(DataUrlPart(url + "/files/image.png"), false, "vlm"))))!;
            Check((string?)fetched["choices"]?[0]?["message"]?["content"] == answer, $"an allowed http image URL: {fetched}");
            var missing = Post("/v1/chat/completions", VlmChatBody(DataUrlPart(url + "/files/missing.png"), false, "vlm"));
            Check(missing.StatusCode == HttpStatusCode.BadRequest && Text(missing).Contains("404"), $"a URL that answers 404: {Text(missing)}");

            // The multipart upload: the same answer, both ways.
            var upload = http.PostAsync("/v1/chat/upload", UploadForm(png, false, ("model", "vlm"))).Result;
            var uploaded = JsonNode.Parse(Text(upload))!;
            Check(upload.StatusCode == HttpStatusCode.OK && (string?)uploaded["object"] == "chat.completion" && (string?)uploaded["choices"]![0]!["message"]!["content"] == answer,
                $"upload: {uploaded}");
            var uploadStream = http.PostAsync("/v1/chat/upload", UploadForm(png, true, ("model", "vlm"))).Result;
            Check(uploadStream.Content.Headers.ContentType!.MediaType == "text/event-stream" && StreamedText(Text(uploadStream), "upload streamed") == answer, "upload streamed");
            Check(http.PostAsync("/v1/chat/upload", new StringContent("{}", Encoding.UTF8, "application/json")).Result.StatusCode == HttpStatusCode.UnsupportedMediaType,
                "an upload that is not a form: 415");

            // /api/chat: images in the library's chat JSON parts, and as base64 "images".
            string image64 = Convert.ToBase64String(File.ReadAllBytes(png));
            var parts = JsonNode.Parse(Text(Post("/api/chat", $$$"""
                {"messages":[{"role":"system","content":"Read the scan."},{"role":"user","content":[{"type":"image","media_type":"image/png","data":"{{{image64}}}"},{"type":"text","text":"What is in this image?"}]}],
                 "stream":false,"options":{"temperature":0,"num_predict":20,"repeat_penalty":1}}
                """)))!;
            Check((string?)parts["message"]?["content"] == answer, $"/api/chat with image parts: {parts}");
            var lines = Text(Post("/api/chat", $$$"""
                {"messages":[{"role":"system","content":"Read the scan."},{"role":"user","content":"What is in this image?","images":["{{{image64}}}"]}],
                 "options":{"temperature":0,"num_predict":20,"repeat_penalty":1}}
                """)).Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => JsonNode.Parse(l)!).ToList();
            Check(string.Concat(lines.Select(l => (string?)l["message"]?["content"] ?? "")) == answer && (bool)lines[^1]["done"]!, "/api/chat streamed with images");
            Check(Post("/api/chat", """{"messages":[{"role":"user","content":"x","images":["@@@"]}]}""").StatusCode == HttpStatusCode.BadRequest, "/api/chat: bad base64 is 400");

            // Grayscale per request: the image is read as Pillow's convert("L") gives it (the reference's grey pixels).
            var grey = ChatImageDecoder.Grayscale(ChatImage.FromFile(png));
            AssertClose(ReadNpyFloat32(TestData("vlm/reference/pixel_values-png-gray.npy")), loaded!.Vision!.Preprocessor().Pixels(ChatImageDecoder.Decode(grey)), 1e-5f,
                "a grey request's pixels");
            Check(ChatImageDecoder.Grayscale(grey).Equals(grey) && ChatImageDecoder.FormatOf(grey) == "netpbm", "grey twice is grey");
            var greyAnswer = Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), false, "vlm", ",\"grayscale\":true"));
            Check(greyAnswer.StatusCode == HttpStatusCode.OK, $"grayscale: {Text(greyAnswer)}");

            // Refusals.
            var refused = Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), false, "text"));
            Check(refused.StatusCode == HttpStatusCode.BadRequest && Text(refused).Contains("\\u0027image\\u0027") && Text(refused).Contains("unsupported_content"), $"a text-only model refuses images: {Text(refused)}");
            var bad = Post("/v1/chat/completions", VlmChatBody(DataUrlPart("data:image/png;base64,@@@"), false, "vlm"));
            Check(bad.StatusCode == HttpStatusCode.BadRequest && Text(bad).Contains("image_url"), $"bad base64: {Text(bad)}");
            var gif = Post("/v1/chat/completions", VlmChatBody(DataUrlPart("data:image/gif;base64," + Convert.ToBase64String(TinyGif)), false, "vlm"));
            Check(gif.StatusCode == HttpStatusCode.BadRequest && Text(gif).Contains("PNG, JPEG"), $"an unsupported format: {Text(gif)}");
            var three = Post("/v1/chat/completions", VlmChatBody($"{DataUrlPart(dataUrl)},{DataUrlPart(dataUrl)},{DataUrlPart(dataUrl)}", false, "vlm"));
            Check(three.StatusCode == HttpStatusCode.BadRequest && Text(three).Contains("2 images at most"), $"over the image limit: {Text(three)}");
            var damaged = Post("/v1/chat/completions", VlmChatBody(DataUrlPart("data:image/png;base64," + Convert.ToBase64String(File.ReadAllBytes(png)[..100])), false, "vlm"));
            Check(damaged.StatusCode == HttpStatusCode.BadRequest, $"a damaged PNG: {Text(damaged)}");
            var audio = Post("/v1/chat/completions", """{"model":"vlm","messages":[{"role":"user","content":[{"type":"input_audio","input_audio":{}}]}]}""");
            Check(audio.StatusCode == HttpStatusCode.BadRequest && Text(audio).Contains("\\u0027input_audio\\u0027"), "other content parts are refused");
            Check(encoders == 1, $"one encoder per loaded copy: {encoders}");
        }
        finally
        {
            app.StopAsync().GetAwaiter().GetResult();
            ((IAsyncDisposable)app).DisposeAsync().AsTask().GetAwaiter().GetResult();
        }

        Check(disposed == 1, "the engine disposes the images' owner when it unloads the model");
    }

    // The response to a POST whose headers announce a body of `length` bytes that is never sent.
    private static string RawResponse(int port, string path, string contentType, long length)
    {
        using var client = new TcpClient();
        client.Connect(IPAddress.Loopback, port);
        using var stream = client.GetStream();
        stream.ReadTimeout = 60_000;
        byte[] request = Encoding.ASCII.GetBytes($"POST {path} HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\nContent-Type: {contentType}\r\nContent-Length: {length}\r\nConnection: close\r\n\r\n");
        stream.Write(request);
        var answer = new MemoryStream();
        var buffer = new byte[8192];
        try
        {
            int read;
            while ((read = stream.Read(buffer)) > 0)
            {
                answer.Write(buffer, 0, read);
            }
        }
        catch (IOException)
        {
            // the server closed the connection after answering
        }

        return Encoding.UTF8.GetString(answer.ToArray());
    }

    private sealed class Disposer(Action dispose) : IDisposable
    {
        public void Dispose() => dispose();
    }

    // `idrak serve` with the tiny Gemma 3 (and a text model beside it): the answers of `idrak run --image`, the limits.
    private static void CliServeImages(Device device)
    {
        var (tokens, expected) = VlmReference();
        string png = TestData("vlm/image.png");
        string dataUrl = ChatImage.FromFile(png).ToDataUrl();
        string[] greedy = ["-s", "Read the scan.", "--temperature", "0", "--max-tokens", "20", "-j"];
        var (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "What is in this image?", .. greedy]);
        string cliAnswer = (string)JsonOf(text, "run --image")["text"]!;
        (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "--grayscale", "What is in this image?", .. greedy]);
        string cliGrey = (string)JsonOf(text, "run --image --grayscale")["text"]!;
        Check(code == 0 && cliAnswer.Length == expected.Length, $"run --image: '{cliAnswer}'");

        string cache = TempFolder();
        int port;
        using (var probe = new TcpListener(IPAddress.Loopback, 0))
        {
            probe.Start();
            port = ((IPEndPoint)probe.LocalEndpoint).Port;
        }

        string url = $"http://127.0.0.1:{port}";
        var serveOutput = new StringWriter();
        var serveError = new StringWriter();
        var server = Task.Run(() => CommandLine.Run(["serve", "ocr=" + VlmModel, "text=" + TestData("gguf/tiny-qwen3-q8.gguf"), "-p", port.ToString(), "-d", device.ToString(),
            "--cache", cache, "--max-images", "2", "--max-request-mb", "1", "--keep-alive", "10m"], TextWriter.Synchronized(serveOutput), TextWriter.Synchronized(serveError)));
        using var http = new HttpClient { BaseAddress = new Uri(url), Timeout = TimeSpan.FromMinutes(5) };
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
            Check(serveOutput.ToString().Contains("Images") && serveOutput.ToString().Contains("/v1/chat/upload"), $"the announcement names the upload: {serveOutput}");
            HttpResponseMessage Post(string path, string body) => http.PostAsync(path, new StringContent(body, Encoding.UTF8, "application/json")).Result;
            string Text(HttpResponseMessage response) => response.Content.ReadAsStringAsync().Result;

            var sampled = new List<int>();
            var single = WithRecordedTokens(sampled, () => Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), false, "ocr")));
            var json = JsonNode.Parse(Text(single))!;
            string answer = (string?)json["choices"]?[0]?["message"]?["content"] ?? "";
            Check(single.StatusCode == HttpStatusCode.OK && answer == cliAnswer && answer.Length == expected.Length && sampled.SequenceEqual(tokens),
                $"serve /v1: '{answer}', run --image '{cliAnswer}', tokens {string.Join(" ", sampled)}: {json}");
            Check(StreamedText(Text(Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), true, "ocr"))), "serve /v1 streamed") == answer, "serve /v1 streamed");
            var upload = JsonNode.Parse(Text(http.PostAsync("/v1/chat/upload", UploadForm(png, false, ("model", "ocr"))).Result))!;
            Check((string?)upload["choices"]?[0]?["message"]?["content"] == answer, $"serve upload: {upload}");
            var streamedUpload = http.PostAsync("/v1/chat/upload", UploadForm(png, true, ("model", "ocr"))).Result;
            Check(StreamedText(Text(streamedUpload), "serve upload streamed") == answer, "serve upload streamed");
            var greyUpload = JsonNode.Parse(Text(http.PostAsync("/v1/chat/upload", UploadForm(png, false, ("model", "ocr"), ("grayscale", "true"))).Result))!;
            Check((string?)greyUpload["choices"]?[0]?["message"]?["content"] == cliGrey, $"grayscale=true gives run --grayscale's answer '{cliGrey}': {greyUpload}");
            var chat = JsonNode.Parse(Text(Post("/api/chat", $$$"""
                {"model":"ocr","messages":[{"role":"system","content":"Read the scan."},{"role":"user","content":"What is in this image?","images":["{{{Convert.ToBase64String(File.ReadAllBytes(png))}}}"]}],
                 "stream":false,"options":{"temperature":0,"num_predict":20,"repeat_penalty":1}}
                """)))!;
            Check((string?)chat["message"]?["content"] == answer, $"serve /api/chat with images: {chat}");

            // Refusals: a text model, an http URL without --allow-image-urls, the limits, bad input.
            var textModel = Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), false, "text"));
            Check(textModel.StatusCode == HttpStatusCode.BadRequest && Text(textModel).Contains("\\u0027image\\u0027"), $"a text model refuses images: {Text(textModel)}");
            var textUpload = http.PostAsync("/v1/chat/upload", UploadForm(png, false, ("model", "text"))).Result;
            Check(textUpload.StatusCode == HttpStatusCode.BadRequest, $"a text model refuses an upload: {Text(textUpload)}");
            var remote = Post("/v1/chat/completions", VlmChatBody(DataUrlPart(url + "/ui"), false, "ocr"));
            Check(remote.StatusCode == HttpStatusCode.BadRequest && Text(remote).Contains("does not download"), $"an http URL without the option: {Text(remote)}");
            var three = Post("/v1/chat/completions", VlmChatBody($"{DataUrlPart(dataUrl)},{DataUrlPart(dataUrl)},{DataUrlPart(dataUrl)}", false, "ocr"));
            Check(three.StatusCode == HttpStatusCode.BadRequest, $"three images over --max-images 2: {Text(three)}");
            // Over --max-request-mb (1 MB here): refused from the announced length, before the body is read. (A client still
            // sending a large body may see the connection closed after the 413, so this sends the headers alone.)
            string large = RawResponse(port, "/v1/chat/completions", "application/json", 2 * 1024 * 1024);
            Check(large.StartsWith("HTTP/1.1 413", StringComparison.Ordinal) && large.Contains("--max-request-mb"), $"over --max-request-mb: {large}");
            string largeUpload = RawResponse(port, "/v1/chat/upload", "multipart/form-data; boundary=x", 2 * 1024 * 1024);
            Check(largeUpload.StartsWith("HTTP/1.1 413", StringComparison.Ordinal), $"an upload over --max-request-mb: {largeUpload}");
            var bad = Post("/v1/chat/completions", VlmChatBody(DataUrlPart("data:image/jpeg;base64,not base64!"), false, "ocr"));
            Check(bad.StatusCode == HttpStatusCode.BadRequest, $"bad base64: {Text(bad)}");
            var gifForm = new MultipartFormDataContent { { new ByteArrayContent(TinyGif), "image", "x.gif" }, { new StringContent("x"), "prompt" }, { new StringContent("ocr"), "model" } };
            var gif = http.PostAsync("/v1/chat/upload", gifForm).Result;
            Check(gif.StatusCode == HttpStatusCode.BadRequest && Text(gif).Contains("PNG, JPEG"), $"a GIF upload: {Text(gif)}");
            var noPrompt = http.PostAsync("/v1/chat/upload", new MultipartFormDataContent { { new StringContent("ocr"), "model" } }).Result;
            Check(noPrompt.StatusCode == HttpStatusCode.BadRequest, $"an upload without prompt or image: {Text(noPrompt)}");
            var badFlag = http.PostAsync("/v1/chat/upload", UploadForm(png, false, ("model", "ocr"), ("grayscale", "maybe"))).Result;
            Check(badFlag.StatusCode == HttpStatusCode.BadRequest, $"grayscale=maybe: {Text(badFlag)}");
            var garbage = new StringContent("not a form at all");
            garbage.Headers.ContentType = MediaTypeHeaderValue.Parse("multipart/form-data; boundary=xyz");
            var malformed = http.PostAsync("/v1/chat/upload", garbage).Result;
            Check(malformed.StatusCode == HttpStatusCode.BadRequest, $"a malformed form: {(int)malformed.StatusCode} {Text(malformed)}");
        }
        finally
        {
            var stop = ServeCli("server", "stop", "-p", port.ToString());
            Check(stop.Code == 0, $"server stop: {stop.Output}{stop.Error}");
            Check(server.Wait(TimeSpan.FromSeconds(60)) && server.Result == 0, $"serve exits with 0 after stop: {serveOutput}{serveError}");
        }

        // --grayscale for the whole server: every image grey, as run --grayscale.
        var settings = ServeCli("serve", "--help");
        Check(settings.Output.Contains("--grayscale") && settings.Output.Contains("--allow-image-urls") && settings.Output.Contains("/v1/chat/upload"), "serve's help names the image options");
        using (var probe = new TcpListener(IPAddress.Loopback, 0))
        {
            probe.Start();
            port = ((IPEndPoint)probe.LocalEndpoint).Port;
        }

        url = $"http://127.0.0.1:{port}";
        server = Task.Run(() => CommandLine.Run(["serve", VlmModel, "-p", port.ToString(), "-d", device.ToString(), "--cache", cache, "--grayscale"],
            TextWriter.Synchronized(serveOutput), TextWriter.Synchronized(serveError)));
        using var grey = new HttpClient { BaseAddress = new Uri(url), Timeout = TimeSpan.FromMinutes(5) };
        deadline = DateTime.UtcNow.AddSeconds(60);
        while (true)
        {
            Check(!server.IsCompleted, $"the grey server stopped early: {serveOutput}{serveError}");
            try
            {
                if (grey.GetAsync("/").Result.IsSuccessStatusCode)
                {
                    break;
                }
            }
            catch (AggregateException)
            {
            }

            Check(DateTime.UtcNow < deadline, "the grey server answers within a minute");
            Thread.Sleep(100);
        }

        try
        {
            var answer = JsonNode.Parse(grey.PostAsync("/v1/chat/upload", UploadForm(png, false)).Result.Content.ReadAsStringAsync().Result)!;
            Check((string?)answer["choices"]?[0]?["message"]?["content"] == cliGrey, $"serve --grayscale gives run --grayscale's answer: {answer}");
        }
        finally
        {
            Check(ServeCli("server", "stop", "-p", port.ToString()).Code == 0, "grey server stop");
            Check(server.Wait(TimeSpan.FromSeconds(60)) && server.Result == 0, $"grey serve exits with 0: {serveOutput}{serveError}");
        }
    }
}
