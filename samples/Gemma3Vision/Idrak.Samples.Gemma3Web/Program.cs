// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// A web page that reads scans with a Gemma 3 vision-language model (bakrianoo/arabic-legal-documents-ocr-1.0, or any
// Gemma3ForConditionalGeneration folder or Hugging Face id) through Idrak's public API. The library knows no vision
// family: the app registers Gemma 3's (samples/Gemma3Vision/Idrak.Gemma3Vision), loads the model once, and answers
// uploads at POST /api/read, streamed (server-sent events) or as one JSON document, with every figure it measures.
//
//   dotnet run -c Release --project samples/Gemma3Vision/Idrak.Samples.Gemma3Web -- MODEL [options]
//
//   -d, --device NAME     cpu (default), cuda:0, vulkan:0, ...
//   -w, --weights FORMAT  bf16, int8, int4 or a registered packed format (default: as stored); the encoder is float32
//   -k, --kv FORMAT       the KV cache: float32 (default), bfloat16, int8
//   --context N           the context window in tokens (default 8192, at most the model's)
//   --port N              the page's port (default 5080); --host ADDRESS (default 127.0.0.1)
//
// MODEL is a Hugging Face id (downloaded once into Idrak's cache, as idrak pull does; HF_TOKEN for gated ones) or an
// absolute folder path (dotnet run starts a web project in its own folder, so a relative one is read from there).
//
// Then open http://127.0.0.1:5080.

using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;
using Idrak.Abstraction;
using Idrak.Abstraction.Data;
using Idrak.Abstraction.Generation;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Gemma3Vision;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Models.Abstractions;
using Idrak.Nlp;

var positional = new List<string>();
string device = "cpu", weights = "", kv = "float32", host = "127.0.0.1";
int port = 5080, context = 8192;
for (int i = 0; i < args.Length; i++)
{
    switch (args[i])
    {
        case "-d" or "--device": device = args[++i]; break;
        case "-w" or "--weights": weights = args[++i]; break;
        case "-k" or "--kv": kv = args[++i]; break;
        case "--context": context = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--port": port = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
        case "--host": host = args[++i]; break;
        default: positional.Add(args[i]); break;
    }
}

if (positional.Count < 1)
{
    Console.Error.WriteLine("usage: Idrak.Samples.Gemma3Web MODEL [-d DEVICE] [-w FORMAT] [-k FORMAT] [--context N] [--port N] [--host ADDRESS]");
    return 2;
}

string model = positional[0];

// 1. The family: the library registers none, so loading this model without it fails (VisionFamilies names itself).
Gemma3VisionPlugin.Register();

// 2. The model, once: the text decoder with the chosen weights, its vision part as the family read it, and the encoder.
var loadWatch = Stopwatch.StartNew();
var folder = Directory.Exists(model) ? model
    : await HuggingFaceModels.DownloadAsync(model, token: Environment.GetEnvironmentVariable("HF_TOKEN"));
var options = new PretrainedOptions
{
    Device = Device.Parse(device),
    BFloat16 = weights is "bf16" or "bfloat16",
    Int8 = weights == "int8",
    Int4 = weights == "int4",
    PackedFormatName = weights is "" or "bf16" or "bfloat16" or "int8" or "int4" ? null : weights,
};
using var pretrained = PretrainedModel.Load(folder, options);
var vision = pretrained.Vision ?? throw new InvalidOperationException($"{model} has no vision part.");
context = Math.Min(context, pretrained.Spec.MaxPositions);
var cacheFormat = kv.ToLowerInvariant() switch
{
    "bfloat16" or "bf16" => KeyValueFormat.BFloat16,
    "int8" => KeyValueFormat.Int8,
    _ => KeyValueFormat.Float32,
};

// Grey images are made per request (ChatImageDecoder.Grayscale), so one encoder serves both.
using var encoder = new TimedEncoder(vision.CreateEncoder(new VisionEncoderOptions { Device = pretrained.Device }));
var chat = pretrained.CreateChat(cacheFormat, context);
var reader = new ChatGenerator(chat.Generator, chat.Template) { Images = new ChatImages(encoder, vision.PromptFormat, vision.Attention) };
double loadSeconds = loadWatch.Elapsed.TotalSeconds;
Console.WriteLine($"loaded {model} on {pretrained.Device} in {loadSeconds:F1} s; vision: {vision.Describe()}");

var info = new Dictionary<string, object?>
{
    ["model"] = model,
    ["folder"] = folder,
    ["device"] = pretrained.Device.ToString(),
    ["weights"] = weights.Length > 0 ? weights : "as stored",
    ["kv_cache"] = cacheFormat.ToString(),
    ["context"] = context,
    ["vision"] = vision.Describe(),
    ["vision_family"] = vision.Family,
    ["parameters"] = pretrained.Spec.ParameterCount,
    ["load_seconds"] = Math.Round(loadSeconds, 2),
};

// The model answers one request at a time; the others wait their turn.
var gate = new SemaphoreSlim(1, 1);
var json = new JsonSerializerOptions { Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping };

var builder = WebApplication.CreateBuilder();
builder.WebHost.UseUrls($"http://{host}:{port}");
builder.WebHost.ConfigureKestrel(k => k.Limits.MaxRequestBodySize = 64L << 20);
builder.Logging.SetMinimumLevel(LogLevel.Warning);
var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();

app.MapGet("/api/info", () => Results.Json(info, json));

app.MapPost("/api/read", async (HttpContext http, CancellationToken cancel) =>
{
    if (!http.Request.HasFormContentType)
    {
        return Results.BadRequest(new { error = "send multipart/form-data with an 'image' file" });
    }

    var form = await http.Request.ReadFormAsync(cancel);
    if (form.Files.GetFile("image") is not { Length: > 0 } file)
    {
        return Results.BadRequest(new { error = "no image: add a file field named 'image'" });
    }

    string Text(string name, string fallback) => form.TryGetValue(name, out var v) && !string.IsNullOrWhiteSpace(v) ? v.ToString() : fallback;
    float Float(string name, float fallback) => float.TryParse(Text(name, ""), NumberStyles.Float, CultureInfo.InvariantCulture, out var f) ? f : fallback;
    int Int(string name, int fallback) => int.TryParse(Text(name, ""), NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) ? n : fallback;
    bool Bool(string name, bool fallback) => Text(name, "") is { Length: > 0 } b ? b is "true" or "on" or "1" : fallback;

    var system = Text("system", "");
    var prompt = Text("prompt", "Extract the contents of this document.");
    bool stream = Bool("stream", true), grayscale = Bool("grayscale", true);
    var seedText = Text("seed", "");
    var generation = new GenerationOptions
    {
        Temperature = Float("temperature", 0f),
        TopK = Int("top_k", 64),
        TopP = Float("top_p", 0.95f),
        MinP = Float("min_p", 0f),
        RepeatPenalty = Float("repeat_penalty", 1f),
        Seed = seedText.Length > 0 ? int.Parse(seedText, CultureInfo.InvariantCulture) : null,
        NumPredict = Int("max_tokens", 4096),
        NumCtx = context,
        ChunkSize = Math.Max(1, Int("chunk_size", stream ? 1 : 8)),
    };

    // The image: decoded once here for its size and format (EXIF orientation applied, as the library's reader does).
    byte[] bytes;
    using (var memory = new MemoryStream())
    {
        await file.CopyToAsync(memory, cancel);
        bytes = memory.ToArray();
    }

    var image = ChatImage.FromBytes(bytes, file.ContentType);
    var format = ChatImageDecoder.FormatOf(image);
    if (format is null)
    {
        return Results.BadRequest(new { error = $"'{file.FileName}' is not an image the library reads (registered codecs: {string.Join(", ", ImageCodecs.Names)})" });
    }

    var decodeWatch = Stopwatch.StartNew();
    ImageData decoded;
    try
    {
        decoded = ChatImageDecoder.Decode(image);
    }
    catch (Exception ex) when (ex is not OperationCanceledException)
    {
        return Results.BadRequest(new { error = $"'{file.FileName}' could not be decoded: {ex.Message}" });
    }

    double decodeMs = decodeWatch.Elapsed.TotalMilliseconds;
    if (grayscale)
    {
        image = ChatImageDecoder.Grayscale(image);
    }

    var messages = new List<ChatMessage>();
    if (system.Length > 0)
    {
        messages.Add(new ChatMessage("system", system));
    }

    messages.Add(new ChatMessage("user", [image, new ChatText(prompt)]));
    var request = new ChatRequest(messages) { Options = generation };

    var imageInfo = new Dictionary<string, object?>
    {
        ["file"] = file.FileName,
        ["format"] = format,
        ["bytes"] = bytes.Length,
        ["width"] = decoded.Width,
        ["height"] = decoded.Height,
        ["channels"] = decoded.Channels,
        ["grayscale"] = grayscale,
        ["image_tokens"] = encoder.Layout(decoded).Tokens,
        ["decode_ms"] = Math.Round(decodeMs, 1),
    };

    var waitWatch = Stopwatch.StartNew();
    await gate.WaitAsync(cancel);
    try
    {
        double queueMs = waitWatch.Elapsed.TotalMilliseconds;
        ComputeResources.ResetPeakMemoryUsage(pretrained.Device);
        encoder.Reset();
        var total = Stopwatch.StartNew();
        double? firstTokenMs = null;
        int pieces = 0;
        var answer = new StringBuilder();
        ChatChunk? last = null;

        if (stream)
        {
            http.Response.Headers.ContentType = "text/event-stream; charset=utf-8";
            http.Response.Headers.CacheControl = "no-cache";
            http.Response.Headers["X-Accel-Buffering"] = "no";
            await Send(http, "start", new { image = imageInfo, queue_ms = Math.Round(queueMs, 1) }, json, cancel);
        }

        // ChatGenerator.Stream is synchronous: run it off the request thread and hand pieces back as they come.
        var channel = System.Threading.Channels.Channel.CreateUnbounded<ChatChunk>();
        var producer = Task.Run(() =>
        {
            try
            {
                foreach (var chunk in reader.Stream(request, cancel))
                {
                    channel.Writer.TryWrite(chunk);
                }

                channel.Writer.Complete();
            }
            catch (Exception ex)
            {
                channel.Writer.Complete(ex);
            }
        }, cancel);

        try
        {
            await foreach (var chunk in channel.Reader.ReadAllAsync(cancel))
            {
                last = chunk;
                var delta = chunk.Delta.Content;
                if (string.IsNullOrEmpty(delta))
                {
                    continue;
                }

                firstTokenMs ??= total.Elapsed.TotalMilliseconds;
                pieces++;
                answer.Append(delta);
                if (stream)
                {
                    await Send(http, "token", new { text = delta }, json, cancel);
                }
            }

            await producer;
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            if (stream)
            {
                await Send(http, "error", new { error = ex.Message }, json, CancellationToken.None);
                return Results.Empty;
            }

            return Results.Json(new { error = ex.Message }, json, statusCode: 500);
        }

        var memory = ComputeResources.GetMemoryUsage(pretrained.Device);
        var stats = last?.Stats;
        var text = last?.Message?.Content ?? answer.ToString();
        var metrics = new Dictionary<string, object?>
        {
            ["queue_ms"] = Math.Round(queueMs, 1),
            ["image_encode_ms"] = Math.Round(encoder.LastMilliseconds, 1),
            ["time_to_first_token_ms"] = firstTokenMs is { } f ? Math.Round(f, 1) : null,
            ["prompt_tokens"] = stats?.PromptTokens,
            ["prompt_ms"] = stats is null ? null : Math.Round(stats.PromptDuration.TotalMilliseconds, 1),
            ["prompt_tokens_per_second"] = stats is { PromptDuration.TotalSeconds: > 0 } ? Math.Round(stats.PromptTokens / stats.PromptDuration.TotalSeconds, 1) : null,
            ["generated_tokens"] = stats?.GeneratedTokens,
            ["generation_ms"] = stats is null ? null : Math.Round(stats.GenerationDuration.TotalMilliseconds, 1),
            ["tokens_per_second"] = stats is null ? null : Math.Round(stats.TokensPerSecond, 2),
            ["total_ms"] = Math.Round(total.Elapsed.TotalMilliseconds, 1),
            ["library_total_ms"] = stats is null ? null : Math.Round(stats.TotalDuration.TotalMilliseconds, 1),
            ["context_resets"] = stats?.ContextResets,
            ["done_reason"] = last?.DoneReason,
            ["streamed_pieces"] = pieces,
            ["characters"] = text.Length,
            ["device_memory_in_use_mb"] = memory.InUse >> 20,
            ["device_memory_peak_mb"] = memory.Peak >> 20,
            ["device_memory_cached_mb"] = memory.Cached >> 20,
            ["device_memory_limit_mb"] = memory.Limit is { } limit ? limit >> 20 : null,
            ["process_working_set_mb"] = Environment.WorkingSet >> 20,
        };

        if (stream)
        {
            await Send(http, "done", new { text, metrics, image = imageInfo }, json, CancellationToken.None);
            return Results.Empty;
        }

        return Results.Json(new { text, metrics, image = imageInfo, model = info }, json);
    }
    finally
    {
        gate.Release();
    }
});

Console.WriteLine($"open http://{host}:{port}");
await app.RunAsync();
return 0;

static async Task Send(HttpContext http, string name, object data, JsonSerializerOptions json, CancellationToken cancel)
{
    await http.Response.WriteAsync($"event: {name}\ndata: {JsonSerializer.Serialize(data, json)}\n\n", cancel);
    await http.Response.Body.FlushAsync(cancel);
}

/// <summary>The family's encoder, timed: how long the last request's images took to encode.</summary>
internal sealed class TimedEncoder(IVisionEncoder inner) : IVisionEncoder
{
    public double LastMilliseconds { get; private set; }

    public int Width => inner.Width;

    public Device Device => inner.Device;

    public ImageTokenLayout Layout(ImageData image) => inner.Layout(image);

    public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images)
    {
        var watch = Stopwatch.StartNew();
        var features = inner.Encode(images);
        LastMilliseconds += watch.Elapsed.TotalMilliseconds;
        return features;
    }

    public void Reset() => LastMilliseconds = 0;

    public void Dispose() => inner.Dispose();
}
