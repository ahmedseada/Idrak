// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// The legal OCR app's tests: the reader service and the HTTP API in process, with the repository's tiny Gemma 3 fixture
// (random weights: the plumbing, not the text) and generated pages. A plain runner like the repository's: IDRAK_FILTER
// runs the tests whose name contains it, IDRAK_DEVICES the device (its first entry; the CPU by default).
//
//   IDRAK_DEVICES=cpu dotnet run -c Release --project samples/Idrak.Samples.LegalOcr.Tests

using System.Diagnostics;
using System.IO.Compression;
using System.Net.Http.Json;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak;
using Idrak.Data.Abstractions;
using Idrak.Gemma3Vision;
using Idrak.Samples.LegalOcr;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;

Gemma3VisionPlugin.Register();
string device = Environment.GetEnvironmentVariable("IDRAK_DEVICES") is { Length: > 0 } devices ? devices.Split(',')[0].Trim() : "cpu";
string root = Path.Combine(Path.GetTempPath(), $"idrak-legal-tests-{Environment.ProcessId}");
Directory.CreateDirectory(root);
string tinyGemma = Path.Combine(RepositoryRoot(), "tests", "Idrak.Tests", "data", "vlm", "tiny-gemma3");

(string Name, Action Run)[] tests =
[
    ("legal settings: bakrianoo's model is read with its model card's preprocessing and prompt", () =>
    {
        var configuration = new ConfigurationBuilder().AddJsonFile(Path.Combine(RepositoryRoot(), "samples", "Idrak.Samples.LegalOcr", "appsettings.json")).Build();
        var settings = configuration.GetSection("LegalOcr").Get<LegalOcrSettings>()!;
        var bakrianoo = settings.Models.Single(m => m.Id == "bakrianoo");
        Check(bakrianoo.Repo == "bakrianoo/arabic-legal-documents-ocr-1.0", bakrianoo.Repo ?? "no repo");
        Check(ImageTransformPipeline.Parse(bakrianoo.Preprocessing).ToString() == "grayscale,max_width=1024,contrast=1.5", bakrianoo.Preprocessing);
        Check(bakrianoo.Prompt == "Extract details to JSON." && bakrianoo.MaxTokens == 2048, $"{bakrianoo.Prompt}, {bakrianoo.MaxTokens}");
        Check(settings.Weights == "bf16", settings.Weights);
    }),
    ("legal reader: a model loads, reads a page as events (the page as it sees it, its text, then the result), and the next one replaces it", () =>
    {
        using var service = Service(Settings(("a", tinyGemma), ("b", tinyGemma)));
        Check(service.Status.State == ReaderState.Idle, service.Status.Message);
        Load(service, "a");
        var events = Read(service, Page(30, 50), new ReadRequest(Preprocessing: "grayscale,max_width=16", MaxTokens: 5), null);
        Check(events[0].Event == "image" && events[^1].Event == "done", string.Join(" ", events.Select(e => e.Event)));
        var image = events[0].Data;
        Check(image["width"]!.GetValue<int>() == 16 && image["originalWidth"]!.GetValue<int>() == 50 && image["preprocessing"]!.GetValue<string>() == "grayscale,max_width=16",
            image.ToJsonString());
        var done = events[^1].Data;
        string streamed = string.Concat(events.Where(e => e.Event == "token").Select(e => e.Data["text"]!.GetValue<string>()));
        Check(streamed.Trim() == done["text"]!.GetValue<string>(), $"streamed \"{streamed}\", read \"{done["text"]}\"");
        Check(done["generatedTokens"]!.GetValue<int>() is > 0 and <= 5 && done["cer"] is null && done["model"]!.GetValue<string>() == "a", done.ToJsonString());

        Load(service, "b");
        var loaded = service.Models().Where(m => m.Loaded).Select(m => m.Id).ToList();
        Check(loaded.SequenceEqual(["b"]), $"loaded: {string.Join(", ", loaded)}");
    }),
    ("legal reader: a model that cannot be loaded says why; reading before loading is an error event", () =>
    {
        var settings = Settings(("a", tinyGemma)) with
        {
            Models = [new ReaderModel { Id = "a", Name = "a", Folder = tinyGemma }, new ReaderModel { Id = "ours", Name = "ours", Repo = "google/gemma-3-4b-it" }],
        };
        using var service = Service(settings);
        var ours = service.Models().Single(m => m.Id == "ours");
        Check(!ours.Usable && ours.Unusable!.Contains("adapter", StringComparison.Ordinal), ours.Unusable ?? "usable");
        try
        {
            service.StartLoad("ours");
            Check(false, "an unusable model must not start loading");
        }
        catch (ArgumentException ex)
        {
            Check(ex.Message.Contains("adapter", StringComparison.Ordinal), ex.Message);
        }

        var events = Read(service, Page(20, 20), new ReadRequest(), null);
        Check(events.Count == 1 && events[0].Event == "error" && events[0].Data["error"]!.GetValue<string>().Contains("No model is loaded", StringComparison.Ordinal),
            string.Join(" ", events.Select(e => $"{e.Event} {e.Data.ToJsonString()}")));
    }),
    ("legal pages: a ShareGPT file with a zip of images gives the pages with their prompts and expected answers; a page read scores CER", () =>
    {
        string data = EvaluationData(3);
        var settings = Settings(("a", tinyGemma)) with { EvaluationData = data, EvaluationImages = Path.Combine(root, "images.zip") };
        var pages = new EvaluationPages(settings);
        Check(pages.Configured && pages.All.Count == 3, $"{pages.All.Count} pages");
        var first = pages.Get(0)!;
        Check(first.Prompt == "Extract details to JSON." && first.Expected.Contains("محكمة", StringComparison.Ordinal), $"{first.Prompt}: {first.Expected}");
        Check(pages.List().First().Preview.StartsWith("محكمة", StringComparison.Ordinal), pages.List().First().Preview);

        using var service = Service(settings);
        Load(service, "a");
        var done = Read(service, first.Image, new ReadRequest(MaxTokens: 4), first.Expected)[^1];
        Check(done.Data["cer"] is { } cer && cer.GetValue<double>() >= 0 && done.Data["expected"]!.GetValue<string>().Contains("محكمة", StringComparison.Ordinal),
            done.Data.ToJsonString());
    }),
    ("legal api: status, models, pages, a page's image and a read over HTTP, with the JSON the page reads", () =>
    {
        string data = EvaluationData(2);
        var settings = Settings(("a", tinyGemma)) with { EvaluationData = data, EvaluationImages = Path.Combine(root, "images.zip") };
        var builder = WebApplication.CreateSlimBuilder();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Logging.ClearProviders();
        builder.Services.AddSingleton(settings);
        builder.Services.AddSingleton<ReaderService>();
        builder.Services.AddSingleton<EvaluationPages>();
        builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.Converters.Add(new System.Text.Json.Serialization.JsonStringEnumConverter()));
        using var app = builder.Build();
        LegalOcrApi.Map(app);
        app.StartAsync().GetAwaiter().GetResult();
        try
        {
            using var http = new HttpClient { BaseAddress = new Uri(app.Urls.First()) };
            var status = http.GetFromJsonAsync<JsonObject>("/api/status").GetAwaiter().GetResult()!;
            Check(status["state"]!.GetValue<string>() == "Idle" && status["device"] is not null, status.ToJsonString());
            var models = http.GetFromJsonAsync<JsonArray>("/api/models").GetAwaiter().GetResult()!;
            Check(models.Count == 1 && models[0]!["id"]!.GetValue<string>() == "a" && models[0]!["usable"]!.GetValue<bool>(), models.ToJsonString());
            var pageList = http.GetFromJsonAsync<JsonObject>("/api/pages").GetAwaiter().GetResult()!;
            Check(pageList["configured"]!.GetValue<bool>() && pageList["pages"]!.AsArray().Count == 2, pageList.ToJsonString());
            var png = http.GetByteArrayAsync("/api/pages/1/image").GetAwaiter().GetResult();
            Check(png.Length > 8 && png[1] == (byte)'P' && png[2] == (byte)'N', $"{png.Length} bytes");
            Check(http.GetAsync("/api/pages/9/image").GetAwaiter().GetResult().StatusCode == System.Net.HttpStatusCode.NotFound, "a page that is not there");

            var load = http.PostAsync("/api/models/a/load", null).GetAwaiter().GetResult();
            Check(load.StatusCode == System.Net.HttpStatusCode.Accepted, load.StatusCode.ToString());
            var clock = Stopwatch.StartNew();
            while (http.GetFromJsonAsync<JsonObject>("/api/status").GetAwaiter().GetResult()!["state"]!.GetValue<string>() != "Ready" && clock.Elapsed < TimeSpan.FromMinutes(2))
            {
                Thread.Sleep(50);
            }

            string dataUrl = "data:image/png;base64," + Convert.ToBase64String(png);
            using var response = http.PostAsJsonAsync("/api/read", new { image = dataUrl, maxTokens = 3 }).GetAwaiter().GetResult();
            Check(response.Content.Headers.ContentType?.MediaType == "text/event-stream", response.Content.Headers.ContentType?.ToString() ?? "no type");
            string body = response.Content.ReadAsStringAsync().GetAwaiter().GetResult();
            Check(body.Contains("event: image", StringComparison.Ordinal) && body.Contains("event: done", StringComparison.Ordinal), body.Length > 400 ? body[..400] : body);
            var bad = http.PostAsJsonAsync("/api/read", new { prompt = "x" }).GetAwaiter().GetResult();
            Check(bad.StatusCode == System.Net.HttpStatusCode.BadRequest, bad.StatusCode.ToString());
        }
        finally
        {
            app.StopAsync().GetAwaiter().GetResult();
        }
    }),
];

int failed = 0, ran = 0;
foreach (var (name, run) in tests.Where(t => Environment.GetEnvironmentVariable("IDRAK_FILTER") is not { } f || t.Name.Contains(f, StringComparison.OrdinalIgnoreCase)))
{
    ran++;
    var clock = Stopwatch.StartNew();
    try
    {
        run();
        Console.WriteLine($"  PASS {name} ({clock.Elapsed.TotalSeconds:F1} s)");
    }
    catch (Exception ex)
    {
        failed++;
        Console.WriteLine($"  FAIL {name}: {ex.Message}");
    }
}

try
{
    Directory.Delete(root, recursive: true);
}
catch (IOException)
{
}

Console.WriteLine($"{ran - failed} passed, {failed} failed (device {device})");
return failed == 0 ? 0 : 1;

LegalOcrSettings Settings(params (string Id, string Folder)[] models) => new()
{
    Device = device, Weights = "f32", Context = 512,
    Models = [.. models.Select(m => new ReaderModel { Id = m.Id, Name = m.Id, Folder = m.Folder, Preprocessing = "grayscale", MaxTokens = 4 })],
};

ReaderService Service(LegalOcrSettings settings) => new(settings, NullLogger<ReaderService>.Instance);

// Starts loading `id` and waits until the service is ready (or failed).
void Load(ReaderService service, string id)
{
    service.StartLoad(id);
    var clock = Stopwatch.StartNew();
    while (clock.Elapsed < TimeSpan.FromMinutes(2))
    {
        var status = service.Status;
        if (status.State == ReaderState.Ready && status.Model == id)
        {
            return;
        }

        Check(status.State != ReaderState.Failed, status.Message);
        Thread.Sleep(20);
    }

    throw new TimeoutException($"{id} did not load: {service.Status.Message}");
}

// A read's events as (event, data).
List<(string Event, JsonObject Data)> Read(ReaderService service, ChatImage image, ReadRequest request, string? expected)
{
    var events = new List<(string, JsonObject)>();
    var enumerator = service.ReadAsync(image, request, expected, CancellationToken.None).GetAsyncEnumerator();
    try
    {
        while (enumerator.MoveNextAsync().AsTask().GetAwaiter().GetResult())
        {
            var item = enumerator.Current;
            events.Add((item.EventType ?? "message", JsonSerializer.SerializeToNode(item.Data)!.AsObject()));
        }
    }
    finally
    {
        enumerator.DisposeAsync().AsTask().GetAwaiter().GetResult();
    }

    return events;
}

// A generated page: a grey gradient with a darker band, as PNG.
ChatImage Page(int height, int width) => ChatImage.FromBytes(PageBytes(height, width, 0), "image/png");

byte[] PageBytes(int height, int width, int seed)
{
    var pixels = new float[3 * height * width];
    for (int i = 0; i < pixels.Length; i++)
    {
        int y = i / 3 / width;
        pixels[i] = (y % 7 == seed % 7 ? 0.1f : 0.9f) - (i % 11) / 100f;
    }

    return ImageEncoders.Get("png").Encode(new ImageData(pixels, 3, height, width));
}

// A ShareGPT data file of `count` pages (LLaMA-Factory's layout: messages with an <image> marker, an "images" list)
// and their images in images.zip; returns the data file.
string EvaluationData(int count)
{
    string zip = Path.Combine(root, "images.zip");
    File.Delete(zip);
    using (var archive = ZipFile.Open(zip, ZipArchiveMode.Create))
    {
        for (int i = 0; i < count; i++)
        {
            using var stream = archive.CreateEntry($"page-{i}.png").Open();
            stream.Write(PageBytes(24, 32, i));
        }
    }

    var records = new JsonArray();
    for (int i = 0; i < count; i++)
    {
        records.Add(new JsonObject
        {
            ["messages"] = new JsonArray(
                new JsonObject { ["role"] = "user", ["content"] = "<image>Extract details to JSON." },
                new JsonObject { ["role"] = "assistant", ["content"] = $"{{\"court\": \"محكمة النقض\", \"number\": {100 + i}}}" }),
            ["images"] = new JsonArray($"page-{i}.png"),
        });
    }

    string file = Path.Combine(root, "val.json");
    File.WriteAllText(file, records.ToJsonString(), new UTF8Encoding(false));
    return file;
}

static void Check(bool condition, string message)
{
    if (!condition)
    {
        throw new InvalidOperationException(message);
    }
}

static string RepositoryRoot()
{
    for (var folder = new DirectoryInfo(AppContext.BaseDirectory); folder is not null; folder = folder.Parent)
    {
        if (File.Exists(Path.Combine(folder.FullName, "Idrak.slnx")))
        {
            return folder.FullName;
        }
    }

    throw new DirectoryNotFoundException("The repository root (Idrak.slnx) was not found above " + AppContext.BaseDirectory);
}
