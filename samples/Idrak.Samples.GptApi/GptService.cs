// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Net.ServerSentEvents;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Idrak.Diagnostics;
using Idrak.Samples.Gpt;

namespace Idrak.Samples.GptApi;

/// <summary>Lifecycle of the served model.</summary>
public enum ModelStatus
{
    /// <summary>Starting up.</summary>
    Loading,

    /// <summary>No saved model was found; one is being trained.</summary>
    Training,

    /// <summary>Ready for inference.</summary>
    Ready,

    /// <summary>Loading or training failed; see the message.</summary>
    Failed,
}

/// <summary>Live progress of background training.</summary>
public sealed record TrainingProgress(
    int Epoch, int Epochs, int Batch, int BatchesPerEpoch, double Loss, double? ValidationLoss, double? ValidationAccuracy,
    double Percent, double ElapsedSeconds, double? RemainingSeconds, string Device);

/// <summary>Service state for the UI and health checks.</summary>
public sealed record StatusResponse(ModelStatus Status, string Message, TrainingProgress? Training);

/// <summary>A compute device the model can run on.</summary>
public sealed record DeviceInfo(string Id, string Name, string Type, bool Available, bool Active, MemoryUsage? Memory);

/// <summary>One row of the model summary.</summary>
public sealed record LayerInfo(string Name, int Depth, long Parameters);

/// <summary>Architecture, training record and location of the served model.</summary>
public sealed record ModelInfo(
    string Name, long Parameters, int Layers, int Heads, int Dim, int Context, int VocabularySize, string Vocabulary,
    int TrainedEpochs, double? ValidationLoss, double? ValidationAccuracy, DateTimeOffset? TrainedAt, string? Corpus,
    string WeightsPath, long WeightsBytes, string ActiveDevice, IReadOnlyList<LayerInfo> Summary);

/// <summary>A text-generation request. Omitted fields use the defaults shown.</summary>
/// <param name="Prompt">Text to continue.</param>
/// <param name="Length">Characters to generate per sample (1-2000).</param>
/// <param name="Temperature">Sampling temperature (0.05-3): lower is more predictable, higher more varied.</param>
/// <param name="TopK">Sample only among the k most likely characters; 0 uses all.</param>
/// <param name="Seed">Random seed for reproducible output; null for random.</param>
/// <param name="Device">"cpu", "cuda", "cuda:N", "vulkan" or "vulkan:N"; null keeps the current device.</param>
/// <param name="Samples">Independent continuations generated together as one batch (1-32).</param>
/// <param name="UseCache">Incremental decoding with a KV cache; false recomputes the whole window every step (for comparison).</param>
/// <param name="UseGraph">Record the decoding step once and replay it as a CUDA graph (GPU only; needs the cache).</param>
/// <param name="ChunkSize">Characters generated between synchronizations when streaming (1-128): larger is faster, smaller is more live.</param>
public sealed record GenerateRequest(
    string Prompt = "the little robot ", int Length = 200, float Temperature = 0.7f, int TopK = 0, int? Seed = null, string? Device = null,
    int Samples = 1, bool UseCache = true, bool UseGraph = true, int ChunkSize = 16);

/// <summary>
/// Owns the model: loads it (or trains one in the background when none is saved), serializes access
/// to it, switches it between devices, and runs generation.
/// </summary>
public sealed class GptService(IConfiguration configuration, ILogger<GptService> logger) : BackgroundService
{
    private readonly SemaphoreSlim _gate = new(1, 1);
    private readonly string _modelPath = Path.GetFullPath(configuration["Gpt:ModelPath"] ?? Path.Combine(AppContext.BaseDirectory, "models", "transformer.weights"));
    private CharGpt? _gpt;
    private volatile StatusResponse _status = new(ModelStatus.Loading, "Starting", null);

    public StatusResponse Status => _status;

    protected override async Task ExecuteAsync(CancellationToken stoppingToken)
    {
        await Task.Yield();
        try
        {
            var device = ParseDevice(configuration["Gpt:Device"]) ?? Device.Default;
            if (File.Exists(_modelPath) && File.Exists(GptConfig.ConfigPath(_modelPath)))
            {
                _gpt = CharGpt.Load(_modelPath, device);
                logger.LogInformation("Loaded {Parameters:N0} parameters from {Path} on {Device}", _gpt.Model.ParameterCount, _modelPath, device);
            }
            else
            {
                _gpt = await Task.Run(() => Train(device, stoppingToken), stoppingToken);
            }

            _status = new StatusResponse(ModelStatus.Ready, $"Ready on {_gpt.Device}", _status.Training);
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            logger.LogError(ex, "Model initialization failed");
            _status = new StatusResponse(ModelStatus.Failed, ex.Message, _status.Training);
        }
    }

    private CharGpt Train(Device device, CancellationToken cancellationToken)
    {
        int epochs = configuration.GetValue("Gpt:TrainEpochs", 2);
        _status = new StatusResponse(ModelStatus.Training, $"No model at {_modelPath}; training one on {device}", null);
        logger.LogInformation("No model at {Path}; training {Epochs} epochs on {Device}", _modelPath, epochs, device);
        string corpus = Grammar.Corpus(sentences: 6000, seed: 1);
        var gpt = CharGpt.Create(new GptConfig(new string([.. corpus.Distinct().Order()])), device);

        // Telemetry turns training events into progress for /api/status.
        var progress = new ProgressHook(device, p => _status = new StatusResponse(ModelStatus.Training, $"Training on {device}", p));
        using (Telemetry.Subscribe(progress))
        {
            var trained = GptTraining.Train(gpt, corpus, epochs, batchSize: 32, "built-in grammar", cancellationToken);
            Directory.CreateDirectory(Path.GetDirectoryName(_modelPath)!);
            gpt.Save(_modelPath, trained);
        }

        logger.LogInformation("Saved trained model to {Path}", _modelPath);
        return gpt;
    }

    /// <summary>The model, or an exception explaining why it is not ready.</summary>
    private CharGpt Model => _gpt ?? throw new InvalidOperationException(_status.Message);

    public ModelInfo Describe()
    {
        var gpt = Model;
        var c = gpt.Config;
        var summary = new List<LayerInfo>();
        void Walk(Abstraction.Module module, int depth)
        {
            bool leaf = !module.Children().Any();
            summary.Add(new LayerInfo(module.DisplayName, depth, leaf ? module.ParameterCount : 0));
            foreach (var child in module.Children())
            {
                Walk(child, depth + 1);
            }
        }

        Walk(gpt.Model, 0);
        return new ModelInfo(gpt.Model.DisplayName, gpt.Model.ParameterCount, c.Layers, c.Heads, c.Dim, c.Context, c.Vocabulary.Length, c.Vocabulary,
            c.TrainedEpochs, c.ValidationLoss, c.ValidationAccuracy, c.TrainedAt, c.Corpus, _modelPath,
            File.Exists(_modelPath) ? new FileInfo(_modelPath).Length : 0, gpt.Device.ToString(), summary);
    }

    public IReadOnlyList<DeviceInfo> Devices()
    {
        var active = _gpt?.Device;
        var devices = new List<DeviceInfo> { Info(Device.Cpu, true) };
        for (int i = 0; i < Device.CudaDeviceCount; i++)
        {
            devices.Add(Info(Device.Cuda(i), true));
        }

        if (Device.CudaDeviceCount == 0)
        {
            devices.Add(new DeviceInfo("cuda", "No NVIDIA GPU detected", "Cuda", false, false, null));
        }

        return devices;

        DeviceInfo Info(Device d, bool available) =>
            new(d.ToString(), d.Name, d.Type.ToString(), available, d == active, ComputeResources.GetMemoryUsage(d));
    }

    /// <summary>Generates the whole text and returns it with per-token details and metrics.</summary>
    public async Task<GenerationResult> GenerateAsync(GenerateRequest request, CancellationToken cancellationToken)
    {
        await _gate.WaitAsync(cancellationToken);
        try
        {
            var gpt = Prepare(request, out var prompt, out var settings);
            return await Task.Run(() => gpt.Generate(prompt, settings, cancellationToken: cancellationToken), cancellationToken);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Streams generation as server-sent events: one "token" event per character, then a "metrics" event
    /// (or an "error" event).
    /// </summary>
    public async IAsyncEnumerable<SseItem<object>> StreamAsync(GenerateRequest request, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var channel = Channel.CreateUnbounded<SseItem<object>>(new UnboundedChannelOptions { SingleReader = true });
        _ = Task.Run(async () =>
        {
            bool entered = false;
            try
            {
                await _gate.WaitAsync(cancellationToken);
                entered = true;
                var gpt = Prepare(request, out var prompt, out var settings);
                var result = gpt.Generate(prompt, settings, token => channel.Writer.TryWrite(new SseItem<object>(token, "token")), cancellationToken);
                channel.Writer.TryWrite(new SseItem<object>(result.Metrics, "metrics"));
            }
            catch (Exception ex) when (ex is not OperationCanceledException)
            {
                channel.Writer.TryWrite(new SseItem<object>(new { error = ex.Message }, "error"));
            }
            catch (OperationCanceledException)
            {
            }
            finally
            {
                if (entered)
                {
                    _gate.Release();
                }

                channel.Writer.TryComplete();
            }
        }, CancellationToken.None);

        await foreach (var item in channel.Reader.ReadAllAsync(cancellationToken))
        {
            yield return item;
        }
    }

    /// <summary>Validates the request, moves the model to the requested device and builds sampling settings.</summary>
    private CharGpt Prepare(GenerateRequest request, out string prompt, out GenerationSettings settings)
    {
        var gpt = Model;
        if (request.Device is { Length: > 0 } requested)
        {
            var device = ParseDevice(requested) ?? throw new ArgumentException($"Unknown device '{requested}'. Use cpu, cuda, cuda:N, vulkan or vulkan:N.");
            if (device != gpt.Device)
            {
                var sw = Stopwatch.StartNew();
                gpt.MoveTo(device);
                logger.LogInformation("Moved the model to {Device} in {Ms} ms", device, sw.ElapsedMilliseconds);
                _status = _status with { Message = $"Ready on {device}" };
            }
        }

        prompt = (request.Prompt ?? "").ToLowerInvariant();
        settings = new GenerationSettings(
            Length: Math.Clamp(request.Length, 1, 2000),
            Temperature: Math.Clamp(request.Temperature, 0.05f, 3f),
            TopK: Math.Clamp(request.TopK, 0, gpt.Config.Vocabulary.Length),
            Seed: request.Seed,
            Samples: Math.Clamp(request.Samples, 1, 32),
            UseCache: request.UseCache,
            UseGraph: request.UseGraph,
            ChunkSize: Math.Clamp(request.ChunkSize, 1, 128));
        return gpt;
    }

    private static Device? ParseDevice(string? text) => text?.Trim().ToLowerInvariant() switch
    {
        null or "" or "auto" => null,
        "cpu" => Device.Cpu,
        "cuda" or "gpu" => Device.IsCudaAvailable ? Device.Cuda() : throw new ArgumentException("CUDA is not available on this machine."),
        "vulkan" => Device.Parse("vulkan:0"),
        var s => Device.Parse(s),                                       // cuda:N, vulkan:N, ... (what --list-devices shows)
    };

    public override void Dispose()
    {
        _gpt?.Dispose();
        _gate.Dispose();
        base.Dispose();
    }

    /// <summary>Converts training telemetry into progress snapshots.</summary>
    private sealed class ProgressHook(Device device, Action<TrainingProgress> report) : ITelemetryHook
    {
        private readonly Stopwatch _clock = Stopwatch.StartNew();
        private int _epochs = 1;
        private double? _validationLoss;
        private double? _validationAccuracy;

        public TelemetryLevel Levels => TelemetryLevel.Training | TelemetryLevel.Batches;

        public void OnTrainingStarted(in TrainingStarted e) => _epochs = e.Epochs;

        public void OnBatchCompleted(in BatchCompleted e)
        {
            if (e.Batch % 5 != 0 && e.Batch != e.BatchesPerEpoch)
            {
                return;
            }

            double done = ((e.Epoch - 1) * (double)e.BatchesPerEpoch + e.Batch) / (_epochs * (double)e.BatchesPerEpoch);
            double elapsed = _clock.Elapsed.TotalSeconds;
            report(new TrainingProgress(e.Epoch, _epochs, e.Batch, e.BatchesPerEpoch, e.Loss, _validationLoss, _validationAccuracy,
                done * 100, elapsed, done > 0 ? elapsed / done - elapsed : null, device.ToString()));
        }

        public void OnEpochCompleted(in EpochCompleted e)
        {
            _validationLoss = e.ValidationLoss;
            _validationAccuracy = e.ValidationMetrics?.GetValueOrDefault("accuracy");
        }
    }
}
