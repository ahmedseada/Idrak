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
using Idrak.QwenVision;
using Idrak.Samples.LegalOcr;

// The app brings the families it reads with: the library's registries know nothing of them otherwise.
Gemma3VisionPlugin.Register();
QwenVisionPlugin.Register();

var builder = WebApplication.CreateBuilder(args);
var settings = builder.Configuration.GetSection("LegalOcr").Get<LegalOcrSettings>() ?? new LegalOcrSettings();
using var suggestions = settings.ShowSuggestions ? Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Suggestions)) : null;
if (settings.GpuChunkMB is { } chunkMB)
{
    ComputeResources.GpuChunkBytes = chunkMB << 20;
}

if (settings.GpuStackBytes is { } stackBytes)
{
    ComputeResources.GpuStackBytes = stackBytes;
}
builder.Services.AddSingleton(settings);
builder.Services.AddSingleton(new AdapterStore(settings, builder.Environment.ContentRootPath));
builder.Services.AddSingleton<ReaderService>();
builder.Services.AddSingleton<TuningService>();
builder.Services.AddSingleton<SyntheticService>();
builder.Services.AddSingleton<EvaluationPages>();
builder.Services.AddSingleton<Preprocessing>();
// Open streams end when the app stops (see UntilStopped); anything else gets a few seconds, not the default half minute.
builder.Services.Configure<HostOptions>(o => o.ShutdownTimeout = TimeSpan.FromSeconds(5));
builder.Services.ConfigureHttpJsonOptions(o =>
{
    o.SerializerOptions.Converters.Add(new JsonStringEnumConverter());
    o.SerializerOptions.Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping;
    FiniteJsonNumbers.Add(o.SerializerOptions);                             // a diverged loss reaches the page as null
});

var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();
LegalOcrApi.Map(app);

// The model of LegalOcr:LoadAtStart is on the device before the server takes requests (Ctrl+C while it loads exits).
if (settings.LoadAtStart is { Length: > 0 } first)
{
    using var stop = new CancellationTokenSource();
    ConsoleCancelEventHandler cancel = (_, e) =>
    {
        e.Cancel = true;
        stop.Cancel();
    };
    Console.CancelKeyPress += cancel;
    await StartupLoad.RunAsync(app.Services.GetRequiredService<ReaderService>(), first, Console.Out, !Console.IsOutputRedirected, stop.Token);
    Console.CancelKeyPress -= cancel;
    if (stop.IsCancellationRequested)
    {
        return;
    }
}

app.Run();

/// <summary>The app's HTTP API (the page in wwwroot uses it; the tests call it in process).</summary>
public static class LegalOcrApi
{
    // A stream that ends when its request does or when the app stops (Ctrl+C), so an open page (the tuning events, a read)
    // never holds the shutdown until the host's timeout.
    private static async IAsyncEnumerable<T> UntilStopped<T>(IHostApplicationLifetime lifetime, CancellationToken request, Func<CancellationToken, IAsyncEnumerable<T>> source,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken enumeration = default)
    {
        using var both = CancellationTokenSource.CreateLinkedTokenSource(request, lifetime.ApplicationStopping, enumeration);
        await using var items = source(both.Token).GetAsyncEnumerator(both.Token);
        while (true)
        {
            bool more;
            try
            {
                more = await items.MoveNextAsync();
            }
            catch (OperationCanceledException) when (both.IsCancellationRequested)
            {
                more = false;
            }

            if (!more)
            {
                yield break;
            }

            yield return items.Current;
        }
    }

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

        api.MapPost("/read", IResult (ReadRequest request, ReaderService host, IHostApplicationLifetime lifetime, CancellationToken cancellationToken) =>
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

            return TypedResults.ServerSentEvents(UntilStopped(lifetime, cancellationToken, token => host.ReadAsync(image, request, null, token)));
        });

        // The preprocessing editor: the transforms with their inputs and the app's suggested values; a preview without a model.
        api.MapGet("/transforms", (Preprocessing preprocessing) => preprocessing.Transforms());

        // The Read tab's generation settings to start from.
        api.MapGet("/generation", (LegalOcrSettings settings) => settings.Generation);

        // How transformers' generate() reads with a model (the Docker-Preset): its generation_config.json and the defaults.
        api.MapGet("/models/{id}/transformers-generation", IResult (string id, ReaderService host) =>
        {
            try
            {
                return TypedResults.Ok(host.TransformersGeneration(id));
            }
            catch (ArgumentException ex)
            {
                return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status404NotFound);
            }
        });

        api.MapPost("/preview", IResult (PreviewRequest request, Preprocessing preprocessing, EvaluationPages pages) =>
        {
            ChatImage image;
            if (request.Page is { } index)
            {
                if (pages.Get(index) is not { } page)
                {
                    return TypedResults.NotFound();
                }

                image = page.Image;
            }
            else if (request.Image is { Length: > 0 } data)
            {
                try
                {
                    image = data.StartsWith("data:", StringComparison.Ordinal) ? ChatImage.FromDataUrl(data) : ChatImage.FromBytes(Convert.FromBase64String(data));
                }
                catch (FormatException ex)
                {
                    return TypedResults.Problem($"The image is not a data URL or base64: {ex.Message}", statusCode: StatusCodes.Status400BadRequest);
                }
            }
            else
            {
                return TypedResults.Problem("No image: send \"image\" (a data URL or base64) or \"page\" (an evaluation page).", statusCode: StatusCodes.Status400BadRequest);
            }

            try
            {
                return TypedResults.Ok(preprocessing.Preview(image, request.Preprocessing, request.PanAndScan == true));
            }
            catch (Exception ex) when (ex is ArgumentException or InvalidDataException or NotSupportedException)
            {
                return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
            }
        });

        api.MapGet("/pages", (EvaluationPages pages) =>
        {
            var list = pages.List().ToList();
            return new { source = pages.Source, configured = pages.Configured, error = pages.Error, pages = list };
        });

        api.MapGet("/pages/{index:int}/image", IResult (int index, EvaluationPages pages) =>
            pages.Get(index) is { } page ? TypedResults.Bytes(page.Image.Data, page.Image.MediaType ?? "image/png") : TypedResults.NotFound());

        api.MapPost("/pages/{index:int}/read", IResult (int index, ReadRequest request, EvaluationPages pages, ReaderService host, IHostApplicationLifetime lifetime,
            CancellationToken cancellationToken) =>
            pages.Get(index) is { } page
                ? TypedResults.ServerSentEvents(UntilStopped(lifetime, cancellationToken, token => host.ReadAsync(page.Image, ReaderService.ForPage(request, page.Prompt), page.Expected, token)))
                : TypedResults.NotFound());

        // The first `count` evaluation pages (0: all) read and scored in a row, each with its own prompt unless the request gives one.
        api.MapPost("/pages/evaluate", IResult (ReadRequest request, int? count, EvaluationPages pages, ReaderService host, IHostApplicationLifetime lifetime,
            CancellationToken cancellationToken) =>
            pages.All is var all && pages.Error is { } error
                ? TypedResults.Problem($"The evaluation pages could not be read: {error}", statusCode: StatusCodes.Status409Conflict)
                : TypedResults.ServerSentEvents(UntilStopped(lifetime, cancellationToken, token => host.EvaluateAsync(all, request, count ?? 0, token))));

        // Fine-tuning: the defaults, start, stop, the run's state and its live events.
        api.MapGet("/tuning/defaults", (TuningService tuning) => tuning.Defaults());

        api.MapGet("/tuning", (TuningService tuning) => new { status = tuning.Status, points = tuning.Points(), log = tuning.Log() });

        api.MapGet("/tuning/events", (TuningService tuning, int? run, int? points, int? log, IHostApplicationLifetime lifetime, CancellationToken cancellationToken) =>
            TypedResults.ServerSentEvents(UntilStopped(lifetime, cancellationToken, token => tuning.EventsAsync(run ?? -1, points ?? 0, log ?? 0, token))));

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

        // Make training pages: the training records' text drawn as new pages (plan 18).
        api.MapGet("/synthetic/defaults", (SyntheticService synthetic) => synthetic.Defaults());

        api.MapGet("/synthetic", (SyntheticService synthetic) => synthetic.Status);

        api.MapPost("/synthetic/preview", IResult (SyntheticRequest request, SyntheticService synthetic) =>
        {
            try
            {
                return TypedResults.Ok(synthetic.Preview(request));
            }
            catch (ArgumentException ex)
            {
                return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
            }
        });

        api.MapPost("/synthetic/start", IResult (SyntheticRequest request, SyntheticService synthetic) =>
        {
            try
            {
                return TypedResults.Accepted("/api/synthetic", synthetic.Start(request));
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

        api.MapPost("/synthetic/stop", (SyntheticService synthetic) => synthetic.Stop());

        api.MapGet("/adapters", (AdapterStore adapters) => new
        {
            folder = adapters.Folder, adapters = adapters.List().Select(a => new { id = AdapterStore.Prefix + a.Record.Name, folder = a.Folder, record = a.Record }),
        });
    }
}
