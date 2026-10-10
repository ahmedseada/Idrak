// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Arabic legal OCR: scanned Arabic legal documents read by vision-language models in the browser.
//
//   dotnet run -c Release --project samples/Idrak.Samples.LegalOcr
//   then open http://localhost:5090
//
// The models are in appsettings.json (LegalOcr:Models): bakrianoo's arabic-legal-documents-ocr-1.0 (Gemma 3 4B, read
// with its model card's preprocessing and prompt) and the base Gemma 3 4B; the LoRA adapters tuned in the Fine-tune tab
// (LegalOcr:AdaptersFolder) join the switch. A model downloads from Hugging Face the first time it is picked (into the
// library's cache) and loads onto the device; one model is on the device at a time, and fine-tuning takes it whole.
// The evaluation pages (LegalOcr:EvaluationData, LegalOcr:EvaluationImages) are read with their expected answers and
// scored by CER, one at a time or N in a row; the training pages are LegalOcr:TrainingData.

using System.Text.Json.Serialization;
using Idrak.Gemma3Vision;
using Idrak.Samples.LegalOcr;

// The app brings the family it reads with: the library's registries know nothing of it otherwise.
Gemma3VisionPlugin.Register();

var builder = WebApplication.CreateBuilder(args);
var settings = builder.Configuration.GetSection("LegalOcr").Get<LegalOcrSettings>() ?? new LegalOcrSettings();
builder.Services.AddSingleton(settings);
builder.Services.AddSingleton(new AdapterStore(settings, builder.Environment.ContentRootPath));
builder.Services.AddSingleton<ReaderService>();
builder.Services.AddSingleton<TuningService>();
builder.Services.AddSingleton<EvaluationPages>();
builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.Converters.Add(new JsonStringEnumConverter());
    o.SerializerOptions.Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping;
});

var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();
LegalOcrApi.Map(app);
app.Run();

/// <summary>The app's HTTP API (the page in wwwroot uses it; the tests call it in process).</summary>
public static class LegalOcrApi
{
    /// <summary>Maps the endpoints under /api.</summary>
    public static void Map(IEndpointRouteBuilder app)
    {
        var api = app.MapGroup("/api");

        api.MapGet("/status", (ReaderService host) => host.Status);

        api.MapGet("/models", (ReaderService host) => host.Models());

        api.MapPost("/models/{id}/load", IResult (string id, ReaderService host) =>
        {
            try
            {
                host.StartLoad(id);
                return TypedResults.Accepted("/api/status", host.Status);
            }
            catch (ArgumentException ex)
            {
                return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
            }
            catch (InvalidOperationException ex)
            {
                return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status409Conflict);
            }
        });

        api.MapPost("/models/unload", async (ReaderService host) =>
        {
            await host.UnloadAsync();
            return host.Status;
        });

        api.MapPost("/read", IResult (ReadRequest request, ReaderService host, CancellationToken cancellationToken) =>
        {
            if (request.Image is not { Length: > 0 } data)
            {
                return TypedResults.Problem("No image: send it as a data URL or base64 in \"image\".", statusCode: StatusCodes.Status400BadRequest);
            }

            ChatImage image;
            try
            {
                image = data.StartsWith("data:", StringComparison.Ordinal) ? ChatImage.FromDataUrl(data) : ChatImage.FromBytes(Convert.FromBase64String(data));
            }
            catch (FormatException ex)
            {
                return TypedResults.Problem($"The image is not a data URL or base64: {ex.Message}", statusCode: StatusCodes.Status400BadRequest);
            }

            return TypedResults.ServerSentEvents(host.ReadAsync(image, request, null, cancellationToken));
        });

        api.MapGet("/pages", (EvaluationPages pages) =>
        {
            var list = pages.List().ToList();
            return new { source = pages.Source, configured = pages.Configured, error = pages.Error, pages = list };
        });

        api.MapGet("/pages/{index:int}/image", IResult (int index, EvaluationPages pages) =>
            pages.Get(index) is { } page ? TypedResults.Bytes(page.Image.Data, page.Image.MediaType ?? "image/png") : TypedResults.NotFound());

        api.MapPost("/pages/{index:int}/read", IResult (int index, ReadRequest request, EvaluationPages pages, ReaderService host, CancellationToken cancellationToken) =>
            pages.Get(index) is { } page
                ? TypedResults.ServerSentEvents(host.ReadAsync(page.Image, request, page.Expected, cancellationToken))
                : TypedResults.NotFound());

        // The first `count` evaluation pages (0: all) read and scored in a row.
        api.MapPost("/pages/evaluate", IResult (ReadRequest request, int? count, EvaluationPages pages, ReaderService host, CancellationToken cancellationToken) =>
            pages.All is var all && pages.Error is { } error
                ? TypedResults.Problem($"The evaluation pages could not be read: {error}", statusCode: StatusCodes.Status409Conflict)
                : TypedResults.ServerSentEvents(host.EvaluateAsync(all, request, count ?? 0, cancellationToken)));

        // Fine-tuning: the defaults, start, stop, the run's state and its live events.
        api.MapGet("/tuning/defaults", (TuningService tuning) => tuning.Defaults());

        api.MapGet("/tuning", (TuningService tuning) => new { status = tuning.Status, points = tuning.Points(), log = tuning.Log() });

        api.MapGet("/tuning/events", (TuningService tuning, int? run, int? points, int? log, CancellationToken cancellationToken) =>
            TypedResults.ServerSentEvents(tuning.EventsAsync(run ?? -1, points ?? 0, log ?? 0, cancellationToken)));

        api.MapPost("/tuning/start", IResult (TuneRequest request, TuningService tuning) =>
        {
            try
            {
                return TypedResults.Accepted("/api/tuning", tuning.Start(request));
            }
            catch (ArgumentException ex)
            {
                return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
            }
            catch (InvalidOperationException ex)
            {
                return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status409Conflict);
            }
        });

        api.MapPost("/tuning/stop", (TuningService tuning) => tuning.Stop());

        api.MapGet("/adapters", (AdapterStore adapters) => new
        {
            folder = adapters.Folder, adapters = adapters.List().Select(a => new { id = AdapterStore.Prefix + a.Record.Name, folder = a.Folder, record = a.Record }),
        });
    }
}
