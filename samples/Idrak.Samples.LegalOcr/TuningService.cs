// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Net.ServerSentEvents;
using System.Runtime.CompilerServices;
using Idrak.Models;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

namespace Idrak.Samples.LegalOcr;

/// <summary>Where the fine-tuning job is.</summary>
public enum TuningState
{
    /// <summary>No run yet.</summary>
    Idle,

    /// <summary>Downloading or loading the base model, reading and tokenizing the data, scoring the base model.</summary>
    Preparing,

    /// <summary>Training.</summary>
    Training,

    /// <summary>Stopping after the current step (the adapter so far is then saved).</summary>
    Stopping,

    /// <summary>The run ended; its adapter is in the switch.</summary>
    Finished,

    /// <summary>The run was stopped (the adapter so far is in the switch when a step had been taken).</summary>
    Stopped,

    /// <summary>The run failed (the message says why).</summary>
    Failed,
}

/// <summary>
/// A fine-tuning run's settings from the page: each one null keeps the settings' default (<see cref="TuningDefaults"/>).
/// </summary>
public sealed record TuneRequest
{
    /// <summary>The adapter's name (its folder); a dated one when null.</summary>
    public string? Name { get; init; }

    /// <summary>The switch's model to tune on.</summary>
    public string? Base { get; init; }

    /// <summary>"bf16", "int8" or "int4".</summary>
    public string? Weights { get; init; }

    /// <summary>The training pages' image transforms.</summary>
    public string? Preprocessing { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? Epochs { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public float? LearningRate { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? Rank { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public float? Alpha { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? MaxLength { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? BatchTokens { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? Accumulate { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public bool? TrainProjector { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? CerPages { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? CerEvery { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public bool? CerBefore { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? AnswerTokens { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? MaxPages { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? EvaluateEvery { get; init; }

    /// <summary>See <see cref="TuningDefaults"/>.</summary>
    public int? SaveEvery { get; init; }

    /// <summary>The defaults with this request's values over them.</summary>
    public TuningDefaults Over(TuningDefaults d) => new()
    {
        Base = Base is { Length: > 0 } b ? b : d.Base, Weights = Weights is { Length: > 0 } w ? w : d.Weights, Preprocessing = Preprocessing ?? d.Preprocessing,
        Epochs = Epochs ?? d.Epochs, LearningRate = LearningRate ?? d.LearningRate, Rank = Rank ?? d.Rank, Alpha = Alpha ?? d.Alpha, MaxLength = MaxLength ?? d.MaxLength,
        BatchTokens = BatchTokens ?? d.BatchTokens, Accumulate = Accumulate ?? d.Accumulate, TrainProjector = TrainProjector ?? d.TrainProjector, CerPages = CerPages ?? d.CerPages,
        CerEvery = CerEvery ?? d.CerEvery, CerBefore = CerBefore ?? d.CerBefore, AnswerTokens = AnswerTokens ?? d.AnswerTokens, MaxPages = MaxPages ?? d.MaxPages,
        EvaluateEvery = EvaluateEvery ?? d.EvaluateEvery, SaveEvery = SaveEvery ?? d.SaveEvery,
    };
}

/// <summary>One optimizer step of the run, for the page's chart.</summary>
public sealed record TuningPoint(int Step, int Epoch, float Loss, float LearningRate, float? EvaluationLoss, double? Cer, double Seconds);

/// <summary>The run as the page shows it.</summary>
public sealed record TuningStatus(
    int Run, TuningState State, string Message, string? Name, string? Base, string? Stage, int StageDone, int StageTotal, DownloadInfo? Download,
    int Step, int TotalSteps, int Epoch, int Epochs, float? Loss, float? LearningRate, double? TokensPerSecond, float? EvaluationLoss, float? EvaluationLossBefore,
    double? Cer, double? CerBefore, double Seconds, double? SecondsLeft, string? Folder, int Points, int LogLines, MemoryInfo? Memory, TuningDefaults? Settings);

/// <summary>
/// The Fine-tune tab's job: one run at a time, in the background, with the device to itself (the reader unloads its
/// model and waits). A run loads the base model of the switch (downloading it once), prepares the training data, its
/// images and the evaluation pages through the library's <see cref="ConversationTuning"/>, scores the base model's CER
/// on a few evaluation pages, trains a LoRA adapter (live step, loss, learning rate, speed, evaluation loss, CER, time
/// left and the tuner's log), and writes the adapter into the adapters folder, where the switch lists it. Stopping ends
/// after the current step and saves the adapter so far.
/// </summary>
public sealed class TuningService : IDisposable
{
    private const int LogLimit = 2000;

    private readonly LegalOcrSettings _settings;
    private readonly ReaderService _readers;
    private readonly AdapterStore _adapters;
    private readonly ILogger<TuningService> _logger;
    private readonly Lock _lock = new();
    private readonly List<TuningPoint> _points = [];
    private readonly List<string> _log = [];
    private TuningStatus _status;
    private CancellationTokenSource? _cancel;
    private Task _job = Task.CompletedTask;
    private Stopwatch _clock = new(), _training = new();
    private int _version, _logDropped;

    public TuningService(LegalOcrSettings settings, ReaderService readers, AdapterStore adapters, ILogger<TuningService> logger)
    {
        (_settings, _readers, _adapters, _logger) = (settings, readers, adapters, logger);
        _status = Empty(0, TuningState.Idle, "No run yet.");
    }

    /// <summary>Whether a run is preparing, training or stopping.</summary>
    public bool Running => _status.State is TuningState.Preparing or TuningState.Training or TuningState.Stopping;

    /// <summary>The run's state, with the device's memory.</summary>
    public TuningStatus Status
    {
        get
        {
            var usage = ComputeResources.GetMemoryUsage(_readers.Device);
            lock (_lock)
            {
                return _status with
                {
                    Seconds = Math.Round(_clock.Elapsed.TotalSeconds, 1), Points = _points.Count, LogLines = _logDropped + _log.Count,
                    Memory = new MemoryInfo(usage.InUse, usage.Peak, usage.Limit),
                };
            }
        }
    }

    /// <summary>The run's steps from <paramref name="from"/> on.</summary>
    public IReadOnlyList<TuningPoint> Points(int from = 0)
    {
        lock (_lock)
        {
            return from >= _points.Count ? [] : _points[Math.Max(0, from)..];
        }
    }

    /// <summary>The run's log lines from <paramref name="from"/> on.</summary>
    public IReadOnlyList<string> Log(int from = 0)
    {
        lock (_lock)
        {
            int at = Math.Max(0, from - _logDropped);                         // lines dropped from the front still count
            return at >= _log.Count ? [] : _log[at..];
        }
    }

    /// <summary>What the Fine-tune tab starts from: the defaults, the models it can tune on, the data files and the adapters folder.</summary>
    public object Defaults() => new
    {
        settings = _settings.Tuning,
        bases = _readers.Models().Select(m => new { m.Id, m.Name, m.Usable, m.Unusable }),
        trainingData = _settings.TrainingData,
        trainingDataFound = _settings.TrainingData is { } t && File.Exists(t),
        trainingImages = _settings.TrainingImages ?? _settings.EvaluationImages,
        evaluationData = _settings.EvaluationData,
        evaluationDataFound = _settings.EvaluationData is { } e && File.Exists(e),
        adaptersFolder = _adapters.Folder,
        name = $"legal-{DateTime.Now:yyyyMMdd-HHmm}",
    };

    /// <summary>Starts a run in the background.</summary>
    /// <exception cref="InvalidOperationException">A run is going, or the training data is missing.</exception>
    /// <exception cref="ArgumentException">A setting is not valid (the message says which).</exception>
    public TuningStatus Start(TuneRequest request)
    {
        var settings = request.Over(_settings.Tuning);
        string name = string.IsNullOrWhiteSpace(request.Name) ? $"legal-{DateTime.Now:yyyyMMdd-HHmmss}" : request.Name.Trim();
        string folder = _adapters.PathFor(name);
        Validate(settings);
        var model = _readers.Find(settings.Base);
        if (_settings.TrainingData is not { } data || !File.Exists(data))
        {
            throw new InvalidOperationException($"The training data {_settings.TrainingData ?? "(none)"} was not found: set TrainingData in appsettings.json.");
        }

        lock (_lock)
        {
            if (Running)
            {
                throw new InvalidOperationException("A run is going: stop it first.");
            }

            if (Directory.Exists(folder) && Directory.EnumerateFileSystemEntries(folder).Any())
            {
                throw new InvalidOperationException($"An adapter named {name} exists: give another name.");
            }

            _points.Clear();
            _log.Clear();
            _logDropped = 0;
            _clock = Stopwatch.StartNew();
            _training = new Stopwatch();
            _status = Empty(_status.Run + 1, TuningState.Preparing, "Starting…") with
            {
                Name = name, Base = model.Id, Epochs = settings.Epochs, Settings = settings, Folder = folder,
            };
            _version++;
            _cancel = new CancellationTokenSource();
        }

        var cancel = _cancel.Token;
        _job = Task.Run(() => RunAsync(name, folder, model, settings, cancel));
        return Status;
    }

    /// <summary>Stops the run after its current step (the adapter so far is saved); before training, at once.</summary>
    public TuningStatus Stop()
    {
        lock (_lock)
        {
            if (Running && _cancel is { } cancel)
            {
                cancel.Cancel();
                _status = _status with { State = TuningState.Stopping, Message = _status.State == TuningState.Training ? "Stopping after the current step, then saving…" : "Stopping…" };
                _version++;
            }
        }

        return Status;
    }

    /// <summary>
    /// The run as server-sent events: "status" whenever it changes, "points" (the new steps, from <paramref name="points"/>
    /// on) and "log" (the new lines, from <paramref name="log"/> on); a new run starts its points and log from 0.
    /// </summary>
    public async IAsyncEnumerable<SseItem<object>> EventsAsync(int run, int points, int log, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        int seen = -1;
        while (!cancellationToken.IsCancellationRequested)
        {
            var status = Status;
            if (status.Run != run)
            {
                (run, points, log) = (status.Run, 0, 0);
            }

            if (_version != seen || status.Points > points || status.LogLines > log || status.State is TuningState.Preparing or TuningState.Training or TuningState.Stopping)
            {
                seen = _version;
                yield return new SseItem<object>(status, "status");
                if (Points(points) is { Count: > 0 } newPoints)
                {
                    yield return new SseItem<object>(new { from = points, points = newPoints }, "points");
                    points += newPoints.Count;
                }

                if (Log(log) is { Count: > 0 } lines)
                {
                    yield return new SseItem<object>(new { from = log, lines }, "log");
                    log += lines.Count;
                }
            }

            try
            {
                await Task.Delay(Running ? 500 : 2000, cancellationToken);
            }
            catch (OperationCanceledException)
            {
                yield break;
            }
        }
    }

    /// <summary>Waits for the run to end (the tests).</summary>
    public Task WaitAsync() => _job;

    public void Dispose()
    {
        _cancel?.Cancel();
        try
        {
            _job.Wait(TimeSpan.FromSeconds(30));
        }
        catch (AggregateException)
        {
        }
    }

    // The run: the device taken, the base model loaded, the data prepared, the base scored, training, the adapter recorded.
    private async Task RunAsync(string name, string folder, ReaderModel model, TuningDefaults settings, CancellationToken cancellationToken)
    {
        var previous = MixedPrecision.Default;
        PretrainedModel? pretrained = null;
        ConversationTuning? run = null;
        bool deviceTaken = false;
        try
        {
            Update(s => s with { Stage = null, Message = "Waiting for the reader to give the device…" });
            await _readers.TakeDeviceAsync($"Fine-tuning {name} has the device: reading waits until it ends.", cancellationToken);
            deviceTaken = true;

            // The base: a checkpoint of the switch, or an adapter of ours (its base checkpoint, then the adapter trains on).
            var baseModel = model.Adapter is null ? model : _readers.Find(model.Id) with { Adapter = null };
            string baseFolder = await _readers.FolderAsync(baseModel, (message, download) => Update(s => s with { Message = message, Download = download }), cancellationToken);
            string baseName = baseModel.Folder ?? baseModel.Repo!;
            Update(s => s with { Message = $"Loading {model.Name} ({settings.Weights} weights) onto {_readers.Device}…", Download = null });
            Add($"base: {model.Name} from {baseFolder}");
            pretrained = await Task.Run(() => Load(baseFolder, settings), cancellationToken);
            if (model.Adapter is { } continued)
            {
                Add($"continuing the adapter {continued}: {pretrained.LoadAdapter(continued)} layers");
            }

            // Tensor cores (bfloat16) for the larger products, as idrak tune trains.
            MixedPrecision.Default = MatMulPrecision.BFloat16;
            cancellationToken.ThrowIfCancellationRequested();
            Update(s => s with { Message = "Reading the training data and its images…" });
            var images = model.Adapter is { } a && settings.Preprocessing.Length == 0 ? TuningImages.Read(a) ?? TuningImages.None : TuningImages.Parse(settings.Preprocessing);
            run = ConversationTuning.Prepare(pretrained, [_settings.TrainingData!], new ConversationTuningOptions
            {
                Evaluation = _settings.EvaluationData is { } e && File.Exists(e) ? e : null,
                EvaluationFraction = _settings.EvaluationData is { } f && File.Exists(f) ? 0 : 0.05,
                Images = _settings.TrainingImages ?? _settings.EvaluationImages,
                EvaluationImages = _settings.EvaluationImages,
                Preparation = images,
                VisionParts = settings.TrainProjector ? [VisionTuningParts.Projector] : [],
                Metric = settings.CerPages > 0 ? TuningMetrics.CharacterErrorRate : null,
                MetricSamples = settings.CerPages,
                MetricEvery = settings.CerEvery,
                MaxNewTokens = settings.AnswerTokens,
                MaxConversations = settings.MaxPages,
                Tuning = new FineTuningOptions
                {
                    Rank = settings.Rank, Alpha = settings.Alpha, LearningRate = settings.LearningRate, Epochs = settings.Epochs, MaxLength = settings.MaxLength,
                    BatchTokens = settings.BatchTokens, GradientAccumulation = settings.Accumulate, EvaluateEvery = settings.EvaluateEvery, SaveEvery = settings.SaveEvery,
                },
            }, new Relay<TuningStage>(stage => Update(s => s with { Stage = stage.Name, StageDone = stage.Done, StageTotal = stage.Total })), Add, cancellationToken);
            Update(s => s with { Stage = null });

            if (run.Scorer is { } scorer)
            {
                int scored = 0;
                scorer.Progress = new Relay<TuningAnswer>(answer =>
                {
                    scored = scored % scorer.Count + 1;                    // each scoring counts from 1
                    Update(s => s with { Stage = "scoring CER", StageDone = scored, StageTotal = scorer.Count });
                    Add($"  CER page {scored}/{scorer.Count}: {answer.Score.Value:P2}, {answer.Tokens} tokens");
                });
            }

            if (run.EvaluationSequences is { Count: > 0 })
            {
                Update(s => s with { Message = $"Evaluation loss before training ({run.EvaluationSequences.Count} pages)…" });
                float before = run.EvaluationLoss()!.Value;
                Add($"evaluation loss before training: {before:F4}");
                Update(s => s with { EvaluationLossBefore = before });
            }

            if (run.Scorer is not null && settings.CerBefore)
            {
                Update(s => s with { Message = $"Scoring {model.Name}'s CER on {run.Scorer.Count} evaluation pages before training…", StageDone = 0 });
                var before = run.ScoreAnswers(cancellationToken: cancellationToken)!;
                Add($"before training: CER {before.Score.Value:P2} on {before.Answers.Count} pages, {before.MeanTokens:F0} tokens per answer ({before.Duration.TotalSeconds:F0} s)");
                Update(s => s with { CerBefore = before.Score.Value, Stage = null });
            }

            Directory.CreateDirectory(folder);
            Update(s => s with { State = TuningState.Training, Message = $"Training on {run.TrainingSequences.Count:N0} pages…", Stage = null });
            _training.Start();
            var progress = new Relay<FineTuningProgress>(p =>
            {
                double? cer = p.Answers?.Score.Value;
                double perStep = _training.Elapsed.TotalSeconds / Math.Max(1, p.Step);
                lock (_lock)
                {
                    _points.Add(new TuningPoint(p.Step, p.Epoch, p.Loss, p.LearningRate, p.EvaluationLoss, cer, Math.Round(_training.Elapsed.TotalSeconds, 1)));
                    _status = _status with
                    {
                        Step = p.Step, TotalSteps = p.TotalSteps, Epoch = p.Epoch, Loss = p.Loss, LearningRate = p.LearningRate, TokensPerSecond = Math.Round(p.TokensPerSecond),
                        EvaluationLoss = p.EvaluationLoss ?? _status.EvaluationLoss, Cer = cer ?? _status.Cer, SecondsLeft = Math.Round(perStep * (p.TotalSteps - p.Step)),
                        Stage = _status.State == TuningState.Stopping ? _status.Stage : null,
                        Message = _status.State == TuningState.Stopping ? _status.Message : $"Training: step {p.Step:N0} of {p.TotalSteps:N0}, epoch {p.Epoch} of {settings.Epochs}",
                    };
                    _version++;
                }

                if (p.EvaluationLoss is { } loss)
                {
                    Add($"step {p.Step}: evaluation loss {loss:F4}");
                }

                if (p.Answers is { } answers)
                {
                    Add($"step {p.Step}: CER {answers.Score.Value:P2} on {answers.Answers.Count} pages");
                }
            });

            var result = run.Train(folder, baseName, progress, cancellationToken, line =>
            {
                if (!line.StartsWith("step ", StringComparison.Ordinal) && !line.StartsWith("evaluation batch", StringComparison.Ordinal)
                    && !line.StartsWith("  forward ", StringComparison.Ordinal) && !line.StartsWith("answers at step", StringComparison.Ordinal))
                {
                    Add(line);
                }
            });
            _training.Stop();

            var status = Status;
            AdapterStore.Write(folder, new AdapterRecord
            {
                Name = name, Base = model.Id, BaseModel = baseName, Prompt = Prompt(run.TrainingConversations) ?? model.Prompt,
                Preprocessing = run.Vision?.Images.Pipeline.ToString() ?? settings.Preprocessing, MaxTokens = model.MaxTokens, Created = DateTimeOffset.Now - _clock.Elapsed,
                Stopped = result.Stopped, Steps = result.Last?.Step ?? 0, TotalSteps = result.Last?.TotalSteps ?? 0, Seconds = Math.Round(result.Elapsed.TotalSeconds),
                Loss = result.Last?.Loss, EvaluationLossBefore = status.EvaluationLossBefore, EvaluationLoss = result.EvaluationLosses.Count > 0 ? result.EvaluationLosses[^1] : null,
                CerBefore = status.CerBefore, Cer = result.Answers?.Score.Value, Settings = settings,
            });
            string summary = $"{(result.Stopped ? "Stopped" : "Finished")} after {result.Last?.Step ?? 0:N0} steps in {Elapsed(result.Elapsed)}"
                             + (result.Answers is { } last ? $", CER {last.Score.Value:P2}" : "") + $"; {Prefix(name)} is in the model switch.";
            Add(summary + $" Adapter: {result.Folder}");
            Update(s => s with { State = result.Stopped ? TuningState.Stopped : TuningState.Finished, Message = summary, Stage = null, SecondsLeft = 0 });
            _logger.LogInformation("Tuning {Name}: {Summary}", name, summary);
        }
        catch (OperationCanceledException)
        {
            Add("stopped before training: nothing was saved");
            Update(s => s with { State = TuningState.Stopped, Message = "Stopped before training began: nothing was saved.", Stage = null, Download = null });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Tuning {Name} failed", name);
            bool expected = ex is InvalidOperationException or ArgumentException or InvalidDataException or NotSupportedException or IOException or HttpRequestException
                or ResourceLimitExceededException;
            string message = (expected ? ex.Message : $"{ex.GetType().Name}: {ex.Message}") + ReaderService.AccessHint(model, ex);
            Add("error: " + message);
            Update(s => s with { State = TuningState.Failed, Message = $"The run failed: {message}", Stage = null, Download = null });
        }
        finally
        {
            MixedPrecision.Default = previous;
            run?.Dispose();
            pretrained?.Dispose();
            ComputeResources.ReleaseCachedMemory(_readers.Device);
            if (deviceTaken)
            {
                _readers.ReturnDevice();
            }

            if (Directory.Exists(folder) && !Directory.EnumerateFileSystemEntries(folder).Any())
            {
                Directory.Delete(folder);
            }

            _clock.Stop();
        }
    }

    // The base model's weights onto the device as the run asks (adapters are added by the tuner).
    private PretrainedModel Load(string folder, TuningDefaults settings)
    {
        string weights = settings.Weights.ToLowerInvariant();
        return PretrainedModel.Load(folder, new PretrainedOptions
        {
            Device = _readers.Device,
            BFloat16 = weights is "bf16" or "bfloat16",
            Int8 = weights == "int8",
            Int4 = weights == "int4",
            MaxPositions = Math.Max(_settings.Context, settings.MaxLength + settings.AnswerTokens),
        });
    }

    private static void Validate(TuningDefaults s)
    {
        void Positive(double value, string what)
        {
            if (!(value > 0))
            {
                throw new ArgumentException($"{what} must be above 0 (it is {value}).");
            }
        }

        Positive(s.Epochs, "Epochs");
        Positive(s.LearningRate, "The learning rate");
        Positive(s.Rank, "The rank");
        Positive(s.Alpha, "Alpha");
        Positive(s.MaxLength, "The longest sequence");
        Positive(s.BatchTokens, "Tokens per batch");
        Positive(s.Accumulate, "Accumulation");
        Positive(s.AnswerTokens, "Answer tokens");
        if (s.Weights.ToLowerInvariant() is not ("bf16" or "bfloat16" or "int8" or "int4" or "f32"))
        {
            throw new ArgumentException($"Weights '{s.Weights}': bf16, int8, int4 or f32.");
        }

        if (s.CerPages < 0 || s.MaxPages < 0 || s.CerEvery < 0 || s.EvaluateEvery < 0 || s.SaveEvery < 0)
        {
            throw new ArgumentException("Page counts and step intervals cannot be negative.");
        }

        _ = ImageTransformPipeline.Parse(s.Preprocessing);                   // a bad pipeline is refused before the run
    }

    // The instruction most training pages were sent with (reading with the adapter uses it).
    private static string? Prompt(IReadOnlyList<ChatTranscript> conversations) => conversations.Take(500)
        .Select(c => c.Messages.LastOrDefault(m => m.Role == "user") is { } user ? string.Join("\n", user.Parts.OfType<ChatText>().Select(t => t.Text)).Trim() : "")
        .Where(t => t.Length > 0).GroupBy(t => t, StringComparer.Ordinal).OrderByDescending(g => g.Count()).Select(g => g.Key).FirstOrDefault();

    private static string Prefix(string name) => $"Ours · {name}";

    private static string Elapsed(TimeSpan time) => time.TotalHours >= 1 ? $"{(int)time.TotalHours} h {time.Minutes} min" : time.TotalMinutes >= 1 ? $"{time.Minutes} min {time.Seconds} s" : $"{time.TotalSeconds:F0} s";

    private static TuningStatus Empty(int run, TuningState state, string message) =>
        new(run, state, message, null, null, null, 0, 0, null, 0, 0, 0, 0, null, null, null, null, null, null, null, 0, null, null, 0, 0, null, null);

    private void Update(Func<TuningStatus, TuningStatus> change)
    {
        lock (_lock)
        {
            var next = change(_status);
            // A stop asked for while preparing keeps its state until the run ends.
            _status = _status.State == TuningState.Stopping && next.State is TuningState.Preparing or TuningState.Training ? next with { State = TuningState.Stopping } : next;
            _version++;
        }
    }

    private void Add(string line)
    {
        lock (_lock)
        {
            if (_log.Count == LogLimit)
            {
                _log.RemoveRange(0, LogLimit / 10);                          // the page keeps what it was sent
                _logDropped += LogLimit / 10;
            }

            _log.Add($"{DateTime.Now:HH:mm:ss} {line}");
            _version++;
        }
    }

    // An IProgress that reports on the caller's thread.
    private sealed class Relay<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }
}
