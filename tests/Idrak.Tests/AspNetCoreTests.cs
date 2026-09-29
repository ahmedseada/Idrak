using System.Net;
using System.Net.Http.Json;
using System.Text.Json.Nodes;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Idrak;
using Idrak.AspNetCore;
using Idrak.Data;
using Idrak.Generation;
using Idrak.Inference;
using Idrak.Layers;
using Idrak.Optimizers;
using Idrak.Training;

// Idrak.AspNetCore over a real Kestrel server on a free port (CPU only; the endpoints do not depend on the device).
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] AspNetCore =
    [
        ("aspnetcore: MapPredictor, MapGenerate (JSON and SSE), MapOllamaApi, status, DI predictor, errors", d => { if (d == Device.Cpu) AspNetCoreEndpoints(d); }),
    ];

    private sealed record Row(float A, float B, float C);

    private static void AspNetCoreEndpoints(Device device)
    {
        var split = SmallRegression(120, 4).Split(0.8, seed: 1).StandardizeFeatures().StandardizeTargets();
        var network = Network.Input(3).Seed(3).Linear(8).ReLU().Linear(1);
        var model = network.Build();
        new TrainingRun { Model = model, Loss = Losses.MeanSquaredError, Optimizer = p => new Adam(p, 0.01f), Train = split.Train.Batches(16), Epochs = 2 }.Fit();
        var direct = Predictor.For(model).Input<Row>(r => [r.A, r.B, r.C]).ScaleInputs(split.FeatureScaler!).UnscaleOutputs(split.TargetScaler!).Output(v => v[0]).Build();
        string package = Path.Combine(Path.GetTempPath(), $"api-{Guid.NewGuid():N}.ikm");
        direct.Save(package);

        var builder = WebApplication.CreateSlimBuilder();
        builder.Logging.ClearProviders();
        builder.WebHost.UseUrls("http://127.0.0.1:0");
        builder.Services.AddIdrak()
            .AddPredictor<Row, float>("rows", package, p => p.Input<Row>(r => [r.A, r.B, r.C]).Output(v => v[0]))
            .AddChatModel("chat", _ => { var (m, t) = TinyLanguageModel(device); return new TextGenerator(m, t, 32); },
                (_, c) => c.KeepAlive(TimeSpan.FromMinutes(1)))
            .AddTextModel("text", _ => { var (m, t) = TinyLanguageModel(device); return new TextGenerator(m, t, 32); });
        var app = builder.Build();
        app.MapPredictor<Row, float>("/predict/rows", "rows");
        app.MapGenerate("/generate", "text");
        app.MapOllamaApi("/api", "chat", o => o.Tools(ToolExecution.Client).ModelName("tiny:latest").Version("test-1"));
        app.MapIdrakStatus("/status");
        app.MapGet("/di", async (IPredictor<Row, float> p) => await p.PredictAsync(new Row(1, 6, 11)));
        app.StartAsync().GetAwaiter().GetResult();
        try
        {
            string url = app.Services.GetRequiredService<IServer>().Features.Get<IServerAddressesFeature>()!.Addresses.First();
            using var http = new HttpClient { BaseAddress = new Uri(url) };
            float one = http.PostAsJsonAsync("/predict/rows", new Row(1, 6, 11)).Result.Content.ReadFromJsonAsync<float>().Result;
            Check(one == direct.Predict(new Row(1, 6, 11)), "MapPredictor = direct predictor");
            var many = http.PostAsJsonAsync("/predict/rows/batch", new[] { new Row(1, 6, 11), new Row(9, 12, 17) }).Result.Content.ReadFromJsonAsync<float[]>().Result!;
            Check(many.SequenceEqual(direct.Predict([new Row(1, 6, 11), new Row(9, 12, 17)])), "batch endpoint");
            Check(http.GetStringAsync("/di").Result == one.ToString(System.Globalization.CultureInfo.InvariantCulture), "IPredictor from DI");

            var generated = JsonNode.Parse(http.PostAsJsonAsync("/generate", new { prompt = "abc", options = new { seed = 3, num_predict = 8 } }).Result.Content.ReadAsStringAsync().Result)!;
            Check((string?)generated["done_reason"] == "length" && (int)generated["stats"]!["generated_tokens"]! == 8, $"generate JSON {generated}");
            var sse = http.PostAsJsonAsync("/generate", new { prompt = "abc", stream = true, options = new { seed = 3, num_predict = 8 } }).Result;
            string events = sse.Content.ReadAsStringAsync().Result;
            Check(sse.Content.Headers.ContentType!.MediaType == "text/event-stream" && events.Contains("event: chunk") && events.EndsWith("\n\n") && events.Contains("event: done"), "SSE");
            Check(http.PostAsJsonAsync("/generate", new { prompt = "" }).Result.StatusCode == HttpStatusCode.BadRequest, "empty prompt is 400");

            string body = """{"model":"any","messages":[{"role":"user","content":"hi"}],"keep_alive":"30m","options":{"seed":1,"num_predict":10}}""";
            var chat = http.PostAsync("/api/chat", new StringContent(body)).Result;          // text/plain: accepted like Ollama
            var lines = chat.Content.ReadAsStringAsync().Result.Split('\n', StringSplitOptions.RemoveEmptyEntries).Select(l => JsonNode.Parse(l)!).ToList();
            Check(chat.Content.Headers.ContentType!.MediaType == "application/x-ndjson" && (bool)lines[^1]["done"]! && (string?)lines[^1]["model"] == "any", "NDJSON chat");
            var ps = JsonNode.Parse(http.GetStringAsync("/api/ps").Result)!["models"]![0]!;
            var expires = DateTimeOffset.Parse((string)ps["expires_at"]!);
            Check((string?)ps["name"] == "tiny:latest" && Math.Abs((expires - DateTimeOffset.UtcNow - TimeSpan.FromMinutes(30)).TotalSeconds) < 60, "keep_alive 30m in /api/ps");
            var single = JsonNode.Parse(http.PostAsync("/api/chat", new StringContent(body.Replace("\"keep_alive\":\"30m\"", "\"stream\":false"))).Result.Content.ReadAsStringAsync().Result)!;
            Check((bool)single["done"]! && single["message"]!["role"]!.ToString() == "assistant", "non-streamed chat");
            Check(http.PostAsync("/api/chat", new StringContent("{\"messages\":[")).Result.StatusCode == HttpStatusCode.BadRequest, "bad JSON is 400");
            Check((string?)JsonNode.Parse(http.GetStringAsync("/api/version").Result)!["version"] == "test-1", "version");
            Check((string?)JsonNode.Parse(http.GetStringAsync("/api/tags").Result)!["models"]![0]!["details"]!["family"] == "chat", "tags");
            var status = JsonNode.Parse(http.GetStringAsync("/status").Result)!;
            Check(status["models"]!.AsArray().Count == 3 && status["devices"]!.AsArray().Count >= 1, "status");

            bool threw = false;
            try { app.MapOllamaApi("/x", "chat", _ => { }); } catch (InvalidOperationException) { threw = true; }
            Check(threw, "tool execution must be chosen");
        }
        finally
        {
            app.StopAsync().GetAwaiter().GetResult();
            ((IAsyncDisposable)app).DisposeAsync().AsTask().GetAwaiter().GetResult();
            model.Dispose();
            File.Delete(package);
        }
    }
}
