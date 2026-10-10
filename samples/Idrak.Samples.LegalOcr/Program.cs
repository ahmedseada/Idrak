// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Arabic legal OCR: scanned Arabic legal documents read by vision-language models in the browser.
//
//   dotnet run -c Release --project samples/Idrak.Samples.LegalOcr
//   then open http://localhost:5090
//
// The models are in appsettings.json (LegalOcr:Models): bakrianoo's arabic-legal-documents-ocr-1.0 (Gemma 3 4B, read
// with its model card's preprocessing and prompt) and our own Gemma 3 LoRA. A model downloads from Hugging Face the
// first time it is picked (into the library's cache) and loads onto the device; one model is on the device at a time.
// The evaluation pages (LegalOcr:EvaluationData, LegalOcr:EvaluationImages) are read with their expected answers and
// scored by CER.

using System.Text.Json.Serialization;
using Idrak.Gemma3Vision;
using Idrak.Samples.LegalOcr;

// The app brings the family it reads with: the library's registries know nothing of it otherwise.
Gemma3VisionPlugin.Register();

var builder = WebApplication.CreateBuilder(args);
var settings = builder.Configuration.GetSection("LegalOcr").Get<LegalOcrSettings>() ?? new LegalOcrSettings();
builder.Services.AddSingleton(settings);
builder.Services.AddSingleton<ReaderService>();
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

        api.MapGet("/pages", (EvaluationPages pages) => new { source = pages.Source, configured = pages.Configured, pages = pages.List() });

        api.MapGet("/pages/{index:int}/image", IResult (int index, EvaluationPages pages) =>
            pages.Get(index) is { } page ? TypedResults.Bytes(page.Image.Data, page.Image.MediaType ?? "image/png") : TypedResults.NotFound());

        api.MapPost("/pages/{index:int}/read", IResult (int index, ReadRequest request, EvaluationPages pages, ReaderService host, CancellationToken cancellationToken) =>
            pages.Get(index) is { } page
                ? TypedResults.ServerSentEvents(host.ReadAsync(page.Image, request, page.Expected, cancellationToken))
                : TypedResults.NotFound());
    }
}
