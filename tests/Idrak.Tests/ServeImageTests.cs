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
using Idrak.Gemma3Vision;
using Idrak.Generation.Abstractions;
using Idrak.Inference;
using Idrak.Models;
using Idrak.Models.Abstractions;
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
        RegisterGemma3Vision();
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
                // Everything from the model's vision family (a registration from outside the library).
                var vision = loaded!.Vision!;
                var encoder = vision.CreateEncoder(new VisionEncoderOptions { Device = loaded.Device });
                encoders++;
                return new ChatImages(encoder, vision.PromptFormat, vision.Attention) { Owner = new Disposer(() => { encoder.Dispose(); disposed++; }) };
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
            AssertClose(ReadNpyFloat32(TestData("vlm/reference/pixel_values-png-gray.npy")), ((Gemma3Vision)loaded!.Vision!).Preprocessor().Pixels(ChatImageDecoder.Decode(grey)), 1e-5f,
                "a grey request's pixels");
            Check(ChatImageDecoder.Grayscale(grey).Equals(grey) && ChatImageDecoder.FormatOf(grey) == "netpbm", "grey twice is grey");
            var greyAnswer = Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), false, "vlm", ",\"grayscale\":true"));
            Check(greyAnswer.StatusCode == HttpStatusCode.OK, $"grayscale: {Text(greyAnswer)}");

            // Vision options per request ("vision_options", the family's keys): Gemma 3's pan and scan of a tall page gives
            // transformers' prompt (120 tokens: the page and 3 crops) and tokens, on /v1, the upload and /api/chat.
            var tallFacts = JsonNode.Parse(File.ReadAllText(TestData("vlm-pan-scan/manifest.json")))!["facts"]!["images"]!["tall"]!;
            int[] tallTokens = [.. tallFacts["new_tokens"]!.AsArray().Select(n => (int)n!)];
            string tallPng = TestData("vlm-pan-scan/image-tall.png");
            string panScan = """{"do_pan_and_scan":true,"pan_and_scan_min_crop_size":32}""";
            sampled.Clear();
            var tall = JsonNode.Parse(Text(WithRecordedTokens(sampled, () => Post("/v1/chat/completions",
                VlmChatBody(DataUrlPart(ChatImage.FromFile(tallPng).ToDataUrl()), false, "vlm", $",\"vision_options\":{panScan}")))))!;
            Check((int?)tall["usage"]?["prompt_tokens"] == 120 && sampled.SequenceEqual(tallTokens), $"/v1 with vision_options: {string.Join(" ", sampled)}: {tall}");
            var tallUpload = JsonNode.Parse(Text(http.PostAsync("/v1/chat/upload", UploadForm(tallPng, false, ("model", "vlm"), ("vision_options", panScan))).Result))!;
            Check((int?)tallUpload["usage"]?["prompt_tokens"] == 120 && (string?)tallUpload["choices"]?[0]?["message"]?["content"] == (string?)tall["choices"]![0]!["message"]!["content"],
                $"upload with vision_options: {tallUpload}");
            var tallChat = JsonNode.Parse(Text(Post("/api/chat", $$$"""
                {"messages":[{"role":"system","content":"Read the scan."},{"role":"user","content":"What is in this image?","images":["{{{Convert.ToBase64String(File.ReadAllBytes(tallPng))}}}"]}],
                 "stream":false,"vision_options":{{{panScan}}},"options":{"temperature":0,"num_predict":20,"repeat_penalty":1}}
                """)))!;
            Check((int?)tallChat["prompt_eval_count"] == 120, $"/api/chat with vision_options: {tallChat}");
            var unknownOption = Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), false, "vlm", ",\"vision_options\":{\"tiles\":2}"));
            Check(unknownOption.StatusCode == HttpStatusCode.BadRequest && Text(unknownOption).Contains("tiles") && Text(unknownOption).Contains("pan_and_scan_max_num_crops"),
                $"an option the family does not take: {Text(unknownOption)}");
            var badOptions = Post("/v1/chat/completions", VlmChatBody(DataUrlPart(dataUrl), false, "vlm", ",\"vision_options\":[1]"));
            Check(badOptions.StatusCode == HttpStatusCode.BadRequest && Text(badOptions).Contains("vision_options"), $"vision_options not an object: {Text(badOptions)}");

            // Image transforms per request ("image_transforms"): the reference compare_real.py made with the same pipeline
            // in Pillow (tests/data/vlm/compare/transformed: image.jpg, grey, 64 wide by Lanczos, contrast 1.5; no system
            // line) gives its 20 greedy tokens on /v1 (text, and an array of steps), the upload and /api/chat.
            int[] transformedTokens = [.. ReadNpyInt64(TestData("vlm/compare/transformed/generated_ids.npy")).Select(t => (int)t)];
            string jpgUrl = ChatImage.FromFile(TestData("vlm/image.jpg")).ToDataUrl();
            const string Card = "grayscale,max_width=64,contrast=1.5";
            string TransformBody(string extra) => $$"""{"model":"vlm","messages":[{"role":"user","content":[{{DataUrlPart(jpgUrl)}},{"type":"text","text":"What is in this image?"}]}],"temperature":0,"max_tokens":20{{extra}}}""";
            foreach (string field in new[] { $"\"{Card}\"", """["grayscale",{"name":"max_width","value":64,"resample":"lanczos"},{"name":"contrast","value":1.5}]""" })
            {
                sampled.Clear();
                var transformed = WithRecordedTokens(sampled, () => Post("/v1/chat/completions", TransformBody($",\"image_transforms\":{field}")));
                Check(transformed.StatusCode == HttpStatusCode.OK && sampled.SequenceEqual(transformedTokens), $"/v1 with image_transforms {field}: {string.Join(" ", sampled)}, expected {string.Join(" ", transformedTokens)}: {Text(transformed)}");
            }

            var transformUploadForm = new MultipartFormDataContent();
            var jpgFile = new ByteArrayContent(File.ReadAllBytes(TestData("vlm/image.jpg")));
            jpgFile.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            transformUploadForm.Add(jpgFile, "image", "image.jpg");
            foreach (var (key, value) in new[] { ("prompt", "What is in this image?"), ("temperature", "0"), ("max_tokens", "20"), ("model", "vlm"), ("image_transforms", Card) })
            {
                transformUploadForm.Add(new StringContent(value), key);
            }

            sampled.Clear();
            var transformUpload = WithRecordedTokens(sampled, () => http.PostAsync("/v1/chat/upload", transformUploadForm).Result);
            Check(transformUpload.StatusCode == HttpStatusCode.OK && sampled.SequenceEqual(transformedTokens), $"upload with image_transforms: {Text(transformUpload)}");
            sampled.Clear();
            var transformChat = WithRecordedTokens(sampled, () => Post("/api/chat", $$$"""
                {"messages":[{"role":"user","content":"What is in this image?","images":["{{{Convert.ToBase64String(File.ReadAllBytes(TestData("vlm/image.jpg")))}}}"]}],
                 "stream":false,"image_transforms":"{{{Card}}}","options":{"temperature":0,"num_predict":20,"repeat_penalty":1}}
                """));
            Check(transformChat.StatusCode == HttpStatusCode.OK && sampled.SequenceEqual(transformedTokens), $"/api/chat with image_transforms: {Text(transformChat)}");
            var unknownTransform = Post("/v1/chat/completions", TransformBody(",\"image_transforms\":\"grayscale,blur=2\""));
            Check(unknownTransform.StatusCode == HttpStatusCode.BadRequest && Text(unknownTransform).Contains("image_transforms") && Text(unknownTransform).Contains("max_width")
                  && Text(unknownTransform).Contains("blur"), $"an unknown transform names the registered ones: {Text(unknownTransform)}");
            var badTransformValue = Post("/api/chat", """{"messages":[{"role":"user","content":"x"}],"image_transforms":"contrast=high"}""");
            Check(badTransformValue.StatusCode == HttpStatusCode.BadRequest && Text(badTransformValue).Contains("contrast takes a number"), $"/api/chat: a bad value: {Text(badTransformValue)}");

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
        // Without the Gemma 3 vision family registered, serve refuses the vision model at startup, naming the registry.
        Gemma3VisionNotRegistered(device);
        var (refusedCode, _, refused) = RunIdrakOn(device, null, "serve", VlmModel, "-p", "0");
        Check(refusedCode == 2 && refused.Contains("Vision family 'Gemma3ForConditionalGeneration' is not registered", StringComparison.Ordinal),
            $"serve without the vision family: {refusedCode} {refused}");
        RegisterGemma3Vision();
        var (tokens, expected) = VlmReference();
        string png = TestData("vlm/image.png");
        string dataUrl = ChatImage.FromFile(png).ToDataUrl();
        string[] greedy = ["-s", "Read the scan.", "--temperature", "0", "--max-tokens", "20", "-j"];
        var (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "What is in this image?", .. greedy]);
        string cliAnswer = (string)JsonOf(text, "run --image")["text"]!;
        (code, text, _) = RunIdrakOn(device, null, ["run", VlmModel, "--image", png, "--grayscale", "What is in this image?", .. greedy]);
        string cliGrey = (string)JsonOf(text, "run --image --grayscale")["text"]!;
        Check(code == 0 && cliAnswer.Length == expected.Length, $"run --image: '{cliAnswer}'");

        // --image-transform: the 20 greedy tokens of compare_real.py --image-transform on the same pipeline in Pillow;
        // --grayscale is the grayscale step (left out of the pipeline, the same answer); an alias keeps the pipeline.
        string jpg = TestData("vlm/image.jpg");
        const string Card = "grayscale,max_width=64,contrast=1.5";
        int[] transformedTokens = [.. ReadNpyInt64(TestData("vlm/compare/transformed/generated_ids.npy")).Select(t => (int)t)];
        string[] plainGreedy = ["What is in this image?", "--temperature", "0", "--max-tokens", "20", "-j"];
        var recorded = new List<int>();
        (code, text, var transformError) = WithRecordedTokens(recorded, () => RunIdrakOn(device, null, ["run", VlmModel, "--image", jpg, "--image-transform", Card, .. plainGreedy]));
        var transformedJson = JsonOf(text, "run --image-transform");
        string cliTransformed = (string)transformedJson["text"]!;
        Check(code == 0 && recorded.SequenceEqual(transformedTokens) && (string?)transformedJson["image_transforms"] == Card,
            $"run --image-transform: {string.Join(" ", recorded)}, expected {string.Join(" ", transformedTokens)}: {text} {transformError}");
        recorded.Clear();
        (code, text, _) = WithRecordedTokens(recorded, () => RunIdrakOn(device, null, ["run", VlmModel, "--image", jpg, "--grayscale", "--image-transform", "max_width=64", "--image-transform", "contrast=1.5", .. plainGreedy]));
        Check(code == 0 && recorded.SequenceEqual(transformedTokens) && (string?)JsonOf(text, "--grayscale with transforms")["image_transforms"] == Card,
            $"--grayscale and two --image-transform options: {text}");
        string aliasConfig = Path.Combine(TempFolder(), "config.json");
        (int, string, string) WithConfig(params string[] args)
        {
            var output = new StringWriter();
            var error = new StringWriter();
            int exit = Idrak.Cli.Shared.StandardInput.With(new StringReader(""), () => CommandLine.Run([.. args, "-d", device.ToString(), "-C", aliasConfig], output, error));
            return (exit, output.ToString(), error.ToString());
        }

        (code, text, _) = WithConfig("alias", "set", "card", VlmModel, "--image-transform", Card);
        Check(code == 0 && File.ReadAllText(aliasConfig).Contains($"\"image_transforms\": \"{Card}\"", StringComparison.Ordinal) && text.Contains("--image-transform " + Card, StringComparison.Ordinal),
            $"alias set --image-transform: {text} {File.ReadAllText(aliasConfig)}");
        recorded.Clear();
        (code, text, _) = WithRecordedTokens(recorded, () => WithConfig(["run", "card", "--image", jpg, .. plainGreedy]));
        Check(code == 0 && recorded.SequenceEqual(transformedTokens), $"an alias's image_transforms: {text}");
        (code, text, _) = WithConfig(["run", "card", "--image", jpg, "--image-transform", "none", .. plainGreedy]);
        Check(code == 0 && (string?)JsonOf(text, "--image-transform none")["image_transforms"] == "", $"--image-transform none over the alias: {text}");
        (code, _, transformError) = RunIdrakOn(device, null, "run", VlmModel, "--image", jpg, "--image-transform", "grayscale,blur=2", "hi");
        Check(code == 2 && transformError.Contains("--image-transform", StringComparison.Ordinal) && transformError.Contains("registered: grayscale, max_width", StringComparison.Ordinal),
            $"an unknown transform: {code} {transformError}");

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
        server = Task.Run(() => CommandLine.Run(["serve", VlmModel, "-p", port.ToString(), "-d", device.ToString(), "--cache", cache, "--grayscale", "--image-transform", "max_width=64,contrast=1.5"],
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
            // The server's transforms (--grayscale, then --image-transform) on every image: run --image-transform's answer;
            // a request's own image_transforms replace them (grayscale alone: run --grayscale's answer).
            var form = new MultipartFormDataContent();
            var jpgFile = new ByteArrayContent(File.ReadAllBytes(jpg));
            jpgFile.Headers.ContentType = new MediaTypeHeaderValue("image/jpeg");
            form.Add(jpgFile, "image", "image.jpg");
            foreach (var (key, value) in new[] { ("prompt", "What is in this image?"), ("temperature", "0"), ("max_tokens", "20") })
            {
                form.Add(new StringContent(value), key);
            }

            var transformed = JsonNode.Parse(grey.PostAsync("/v1/chat/upload", form).Result.Content.ReadAsStringAsync().Result)!;
            Check((string?)transformed["choices"]?[0]?["message"]?["content"] == cliTransformed, $"serve --grayscale --image-transform gives run --image-transform's answer: {transformed}");
            Check(serveOutput.ToString().Contains(Card, StringComparison.Ordinal), $"the announcement names the transforms: {serveOutput}");
            var answer = JsonNode.Parse(grey.PostAsync("/v1/chat/upload", UploadForm(png, false, ("image_transforms", "grayscale"))).Result.Content.ReadAsStringAsync().Result)!;
            Check((string?)answer["choices"]?[0]?["message"]?["content"] == cliGrey, $"a request's image_transforms replace the server's: {answer}");
        }
        finally
        {
            Check(ServeCli("server", "stop", "-p", port.ToString()).Code == 0, "grey server stop");
            Check(server.Wait(TimeSpan.FromSeconds(60)) && server.Result == 0, $"grey serve exits with 0: {serveOutput}{serveError}");
        }
    }
}
