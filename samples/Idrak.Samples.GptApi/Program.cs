// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

// Idrak GPT Web API: serves the character-level transformer from the Transformer sample.
//
//   dotnet run -c Release --project samples/Idrak.Samples.GptApi
//   then open http://localhost:5080          the inference UI (index.html)
//             http://localhost:5080/scalar   interactive API reference (Scalar)
//
// Model location: Gpt:ModelPath in appsettings.json (or --Gpt:ModelPath <path>). Point it at a model saved by
// the Transformer sample, or leave the default: when no model exists, one is trained in the background
// and /api/status reports the progress.

using System.Text.Json;
using System.Text.Json.Serialization;
using Microsoft.AspNetCore.Http.HttpResults;
using Idrak.AspNetCore;
using Idrak.Samples.Gpt;
using Idrak.Samples.GptApi;
using Scalar.AspNetCore;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddOpenApi(options => options.AddDocumentTransformer((document, _, _) =>
{
    document.Info.Title = "Idrak GPT API";
    document.Info.Description = "Text generation with a character-level transformer trained and served by Idrak (pure C#, CPU or CUDA).";
    return Task.CompletedTask;
}));
builder.Services.ConfigureHttpJsonOptions(options => options.SerializerOptions.Converters.Add(new JsonStringEnumConverter()));
builder.Services.AddSingleton<GptService>();
builder.Services.AddIdrak()
    .LoadOnFirstUse()
    .AddChatModel("chat", _ => ChatModelFile.Load(builder.Configuration), (_, c) => c.KeepAlive(TimeSpan.FromMinutes(5)));
builder.Services.AddHostedService(services => services.GetRequiredService<GptService>());

var app = builder.Build();
app.UseDefaultFiles();
app.UseStaticFiles();
app.MapOpenApi();
app.MapScalarApiReference(options => options
    .WithTitle("Idrak GPT API")
    .WithTheme(ScalarTheme.DeepSpace));

var api = app.MapGroup("/api").WithTags("GPT");

api.MapGet("/status", (GptService gpt) => gpt.Status)
    .WithName("GetStatus")
    .WithSummary("Service state")
    .WithDescription("Loading, Training (with live progress), Ready or Failed.");

api.MapGet("/model", Results<Ok<ModelInfo>, ProblemHttpResult> (GptService gpt) =>
        gpt.Status.Status == ModelStatus.Ready ? TypedResults.Ok(gpt.Describe()) : NotReady(gpt))
    .WithName("GetModel")
    .WithSummary("Model architecture and training record")
    .WithDescription("Parameter count, layers, heads, width, context length, vocabulary, validation results, weights file and a layer summary.");

api.MapGet("/devices", (GptService gpt) => gpt.Devices())
    .WithName("GetDevices")
    .WithSummary("Available compute devices")
    .WithDescription("The CPU and every CUDA GPU, with memory usage and which one the model is on.");

api.MapPost("/generate", async Task<Results<Ok<GenerationResult>, ProblemHttpResult>> (GenerateRequest request, GptService gpt, CancellationToken cancellationToken) =>
    {
        if (gpt.Status.Status != ModelStatus.Ready)
        {
            return NotReady(gpt);
        }

        try
        {
            return TypedResults.Ok(await gpt.GenerateAsync(request, cancellationToken));
        }
        catch (ArgumentException ex)
        {
            return TypedResults.Problem(ex.Message, statusCode: StatusCodes.Status400BadRequest);
        }
    })
    .WithName("Generate")
    .WithSummary("Generate text")
    .WithDescription("Continues the prompt and returns one or more samples, every generated character with its probability, entropy and top alternatives, and aggregate metrics (latency, throughput, confidence, perplexity, memory, decoding mode). Set device to switch between cpu and cuda; samples to generate a batch; useCache/useGraph to compare decoding modes.");

api.MapPost("/generate/stream", Results<ServerSentEventsResult<object>, ProblemHttpResult> (GenerateRequest request, GptService gpt, CancellationToken cancellationToken) =>
        gpt.Status.Status == ModelStatus.Ready ? TypedResults.ServerSentEvents(gpt.StreamAsync(request, cancellationToken)) : NotReady(gpt))
    .WithName("GenerateStream")
    .WithSummary("Generate text as a live stream (server-sent events)")
    .WithDescription("Same as /api/generate, but emits \"token\" events (each with its sample index) every chunkSize characters as they are produced, then a \"metrics\" event.");

// ---------------------------------------------------------------- the chat API (Idrak.AspNetCore)
// /api/chat, /api/tags, /api/ps and /api/version, served by the inference engine: the chat model loads on the first
// request and stays loaded for 5 minutes after the last one (or the request's keep_alive); tool calls go to the client.
app.MapChatApi("/api", "chat", o => o
        .Tools(ToolExecution.Client)
        .ModelName(app.Configuration["Gpt:ModelName"] ?? "idrak-char-gpt:latest"))
    .WithTags("Chat API");

app.Run();

static ProblemHttpResult NotReady(GptService gpt) =>
    TypedResults.Problem(gpt.Status.Message, statusCode: StatusCodes.Status503ServiceUnavailable, title: $"Model is {gpt.Status.Status}");
