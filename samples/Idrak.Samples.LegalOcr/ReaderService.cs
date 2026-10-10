// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Net.ServerSentEvents;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Idrak.Data;
using Idrak.Data.Abstractions;
using Idrak.Generation;
using Idrak.Models;
using Idrak.Nlp;
using Idrak.Nlp.Abstractions;

namespace Idrak.Samples.LegalOcr;

/// <summary>What the service is doing.</summary>
public enum ReaderState
{
    /// <summary>No model loaded.</summary>
    Idle,

    /// <summary>Fetching a model's files.</summary>
    Downloading,

    /// <summary>Reading a model's weights onto the device.</summary>
    Loading,

    /// <summary>A model is loaded and free.</summary>
    Ready,

    /// <summary>A model is reading a page.</summary>
    Reading,

    /// <summary>The last download or load failed (the message says why).</summary>
    Failed,

    /// <summary>Fine-tuning has the device: reading waits until it ends.</summary>
    Busy,
}

/// <summary>A download in progress: the file being fetched and the bytes so far.</summary>
public sealed record DownloadInfo(string File, long Received, long? Total, double Seconds);

/// <summary>The device's memory as the library measures it.</summary>
public sealed record MemoryInfo(long InUse, long Peak, long? Limit);

/// <summary>The service's state for the page (polled while a model downloads or loads).</summary>
public sealed record ReaderStatus(ReaderState State, string Message, string? Model, string? Loading, DownloadInfo? Download, double? Seconds, string Device,
    MemoryInfo? Memory);

/// <summary>A model of the switch as the page shows it.</summary>
public sealed record ModelInfo(string Id, string Name, string? About, string Source, string? Adapter, bool Usable, string? Unusable, string Preprocessing,
    string Prompt, int MaxTokens, bool Loaded);

/// <summary>A page to read: the image (a data URL or base64), and what replaces the model's own settings.</summary>
/// <param name="Image">The scan as a data URL ("data:image/png;base64,…") or plain base64.</param>
/// <param name="Prompt">The instruction; the model's own when null or empty.</param>
/// <param name="Preprocessing">The image transforms (the library's pipeline syntax, "" for none); the model's own when null.</param>
/// <param name="MaxTokens">The most tokens the answer may take; the model's own when null.</param>
public sealed record ReadRequest(string? Image = null, string? Prompt = null, string? Preprocessing = null, int? MaxTokens = null);

/// <summary>
/// The app's readers: the models of the settings, one on the device at a time (the library's
/// <see cref="ModelHost{TModel}"/> with <see cref="ModelHost{TModel}.MaxLoaded"/> 1), each a library
/// <see cref="ImageReader"/> with the model's own preprocessing and prompt. A model's files download from Hugging Face
/// the first time (the library's <see cref="HuggingFaceModels"/>, into its cache) with the progress in the status. A read
/// streams the page as the model sees it, the answer as it is generated, then its JSON (the library's lenient repair) and
/// timing and, given the expected answer, its CER (the library's metric over the JSON's values).
/// </summary>
public sealed class ReaderService : IDisposable
{
    private readonly LegalOcrSettings _settings;
    private readonly AdapterStore _adapters;
    private readonly ILogger<ReaderService> _logger;
    private readonly ModelHost<LoadedReader> _host;
    private readonly SemaphoreSlim _gate = new(1, 1);                   // one load or read at a time
    private readonly Lock _statusLock = new();
    private readonly Dictionary<string, string> _folders = new(StringComparer.OrdinalIgnoreCase);
    private ReaderStatus _status;
    private string? _loaded;
    private volatile string? _busy;
    private CancellationTokenSource? _loading;

    public ReaderService(LegalOcrSettings settings, AdapterStore adapters, ILogger<ReaderService> logger)
    {
        _settings = settings;
        _adapters = adapters;
        _logger = logger;
        Device = ParseDevice(settings.Device);
        _host = new ModelHost<LoadedReader>(id => LoadedReader.Create(Find(id), _folders[id], settings, Device), defaultKeepAlive: null) { MaxLoaded = 1 };
        _status = new ReaderStatus(ReaderState.Idle, "No model loaded: pick one.", null, null, null, null, Device.ToString(), null);
    }

    /// <summary>The device the models run on.</summary>
    public Device Device { get; }

    /// <summary>The current state, with the device's memory.</summary>
    public ReaderStatus Status
    {
        get
        {
            var usage = ComputeResources.GetMemoryUsage(Device);
            lock (_statusLock)
            {
                return _status with { Memory = new MemoryInfo(usage.InUse, usage.Peak, usage.Limit) };
            }
        }
    }

    /// <summary>The models of the switch: the settings' and the tuned adapters.</summary>
    public IEnumerable<ModelInfo> Models() => _settings.Models.Concat(_adapters.Models()).Select(m =>
    {
        string? unusable = Unusable(m);
        return new ModelInfo(m.Id, m.Name, m.About, m.Folder ?? m.Repo ?? "", m.Adapter, unusable is null, unusable, m.Preprocessing, m.Prompt, m.MaxTokens,
            string.Equals(_loaded, m.Id, StringComparison.OrdinalIgnoreCase));
    });

    /// <summary>Starts loading <paramref name="id"/> in the background (downloading it first when its files are missing); the status reports it.</summary>
    /// <exception cref="ArgumentException">No such model, or it cannot be loaded as configured.</exception>
    /// <exception cref="InvalidOperationException">Fine-tuning has the device.</exception>
    public void StartLoad(string id)
    {
        var model = Find(id);
        if (_busy is { } busy)
        {
            throw new InvalidOperationException(busy);
        }

        if (Unusable(model) is { } reason)
        {
            throw new ArgumentException($"{model.Name}: {reason}");
        }

        _loading?.Cancel();
        var cancel = _loading = new CancellationTokenSource();
        _ = Task.Run(() => LoadAsync(model, cancel.Token));
    }

    /// <summary>Unloads the model and gives its device memory back.</summary>
    public async Task UnloadAsync()
    {
        _loading?.Cancel();
        await _gate.WaitAsync();
        try
        {
            if (_loaded is { } id)
            {
                _host.Unload(id);
                _loaded = null;
            }

            SetStatus(ReaderState.Idle, "No model loaded: pick one.", null);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>
    /// Reads <paramref name="image"/> with the loaded model, as server-sent events: "image" (the page as the model sees
    /// it), "token" (the answer as it is generated), then "done" (the answer, its JSON, timing and, with
    /// <paramref name="expected"/>, the CER), or "error". A read waits for the one before it ("queued").
    /// </summary>
    public async IAsyncEnumerable<SseItem<object>> ReadAsync(ChatImage image, ReadRequest request, string? expected,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var channel = Channel.CreateUnbounded<SseItem<object>>(new UnboundedChannelOptions { SingleReader = true });
        _ = Task.Run(async () =>
        {
            bool entered = false;
            try
            {
                if (!_gate.Wait(0))
                {
                    channel.Writer.TryWrite(new SseItem<object>(new { message = "waiting for the read before this one" }, "queued"));
                    await _gate.WaitAsync(cancellationToken);
                }

                entered = true;
                string id = _loaded ?? throw new InvalidOperationException(_busy ?? "No model is loaded: pick one first.");
                using var lease = _host.Acquire(id);
                Read(lease.Model, image, request, expected, channel.Writer, cancellationToken);
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or InvalidDataException or NotSupportedException or IOException)
            {
                channel.Writer.TryWrite(new SseItem<object>(new { error = ex.Message }, "error"));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Reading failed");
                channel.Writer.TryWrite(new SseItem<object>(new { error = $"{ex.GetType().Name}: {ex.Message}" }, "error"));
            }
            finally
            {
                if (entered)
                {
                    if (_loaded is { } id)
                    {
                        SetStatus(ReaderState.Ready, $"{Find(id).Name} is ready.", id);
                    }

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

    /// <summary>
    /// Reads the first <paramref name="count"/> evaluation pages (all when 0) with the loaded model and scores each, as
    /// server-sent events: "page" (one page's CER, time and answer length, as it is read), then "done" (the mean and
    /// median CER), or "error". Nothing else reads meanwhile.
    /// </summary>
    public async IAsyncEnumerable<SseItem<object>> EvaluateAsync(IReadOnlyList<EvaluationPage> pages, ReadRequest request, int count,
        [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var channel = Channel.CreateUnbounded<SseItem<object>>(new UnboundedChannelOptions { SingleReader = true });
        _ = Task.Run(async () =>
        {
            bool entered = false;
            try
            {
                if (!_gate.Wait(0))
                {
                    channel.Writer.TryWrite(new SseItem<object>(new { message = "waiting for the read before this one" }, "queued"));
                    await _gate.WaitAsync(cancellationToken);
                }

                entered = true;
                string id = _loaded ?? throw new InvalidOperationException(_busy ?? "No model is loaded: pick one first.");
                if (pages.Count == 0)
                {
                    throw new InvalidOperationException("No evaluation pages: set EvaluationData (and EvaluationImages) in appsettings.json.");
                }

                using var lease = _host.Acquire(id);
                var chosen = count > 0 ? pages.Take(count).ToList() : pages;
                var clock = Stopwatch.StartNew();
                var scores = new List<double>(chosen.Count);
                foreach (var page in chosen)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    SetStatus(ReaderState.Reading, $"{lease.Model.Model.Name} is evaluating: page {scores.Count + 1} of {chosen.Count}…", id);
                    var result = ReadPage(lease.Model, page.Image, request, page.Expected, null, cancellationToken);
                    scores.Add(result.Cer ?? 1);
                    double left = clock.Elapsed.TotalSeconds / scores.Count * (chosen.Count - scores.Count);
                    channel.Writer.TryWrite(new SseItem<object>(new
                    {
                        index = page.Index, name = page.Name, cer = result.Cer, seconds = Math.Round(result.Reading.Elapsed.TotalSeconds, 2),
                        tokens = result.Reading.Stats?.GeneratedTokens ?? 0, truncated = result.Reading.Truncated, json = result.Json is not null, repaired = result.Repaired,
                        done = scores.Count, total = chosen.Count, meanCer = scores.Average(), secondsLeft = Math.Round(left),
                    }, "page"));
                }

                var sorted = scores.Order().ToList();
                channel.Writer.TryWrite(new SseItem<object>(new
                {
                    model = id, pages = scores.Count, meanCer = scores.Average(), medianCer = sorted[sorted.Count / 2],
                    seconds = Math.Round(clock.Elapsed.TotalSeconds, 1),
                }, "done"));
            }
            catch (OperationCanceledException)
            {
            }
            catch (Exception ex) when (ex is InvalidOperationException or ArgumentException or InvalidDataException or NotSupportedException or IOException)
            {
                channel.Writer.TryWrite(new SseItem<object>(new { error = ex.Message }, "error"));
            }
            catch (Exception ex)
            {
                _logger.LogError(ex, "Evaluating failed");
                channel.Writer.TryWrite(new SseItem<object>(new { error = $"{ex.GetType().Name}: {ex.Message}" }, "error"));
            }
            finally
            {
                if (entered)
                {
                    if (_loaded is { } id)
                    {
                        SetStatus(ReaderState.Ready, $"{Find(id).Name} is ready.", id);
                    }

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

    /// <summary>
    /// Gives the device to <paramref name="reason"/>'s work (fine-tuning): waits for the read in progress, unloads the
    /// model and refuses loads and reads (with <paramref name="reason"/> as the message) until <see cref="ReturnDevice"/>.
    /// </summary>
    public async Task TakeDeviceAsync(string reason, CancellationToken cancellationToken)
    {
        _loading?.Cancel();
        await _gate.WaitAsync(cancellationToken);
        try
        {
            _busy = reason;
            if (_loaded is { } id)
            {
                _host.Unload(id);
                _loaded = null;
            }

            SetStatus(ReaderState.Busy, reason, null);
        }
        finally
        {
            _gate.Release();
        }
    }

    /// <summary>Gives the device back to reading.</summary>
    public void ReturnDevice()
    {
        _busy = null;
        SetStatus(ReaderState.Idle, "No model loaded: pick one.", null);
    }

    /// <summary>The model of the switch with that id (the settings' or a tuned adapter).</summary>
    /// <exception cref="ArgumentException">No such model.</exception>
    public ReaderModel Find(string id) =>
        _settings.Models.FirstOrDefault(m => string.Equals(m.Id, id, StringComparison.OrdinalIgnoreCase)) ?? _adapters.Find(id)
        ?? throw new ArgumentException($"No model '{id}' (the switch has {string.Join(", ", _settings.Models.Concat(_adapters.Models()).Select(m => m.Id))}).");

    /// <summary>The checkpoint folder of <paramref name="model"/> (its <see cref="ReaderModel.Folder"/>, else its repository downloaded once, with progress).</summary>
    public Task<string> FolderAsync(ReaderModel model, Action<string, DownloadInfo?>? progress, CancellationToken cancellationToken)
    {
        if (model.Folder is { } folder)
        {
            return Task.FromResult(folder);
        }

        var clock = Stopwatch.StartNew();
        progress?.Invoke($"Looking up {model.Repo} on Hugging Face…", null);
        var downloader = new Downloader
        {
            Progress = new Reporter<DownloadProgress>(p => progress?.Invoke($"Downloading {model.Repo}: {p.File}", new DownloadInfo(p.File, p.Received, p.Total, clock.Elapsed.TotalSeconds))),
            Log = line => _logger.LogInformation("{Line}", line),
        };
        return HuggingFaceModels.DownloadAsync(model.Repo!, token: HuggingFace.Token(_settings.HuggingFaceToken), downloader: downloader, cancellationToken: cancellationToken);
    }

    public void Dispose()
    {
        _loading?.Cancel();
        _host.Dispose();
        _gate.Dispose();
    }

    // Why a model cannot be loaded as configured, or null.
    private static string? Unusable(ReaderModel model) =>
        model.Folder is null && model.Repo is null ? "neither Repo nor Folder is set"
        : model.Folder is { } folder && !Directory.Exists(folder) ? $"the folder {folder} does not exist"
        : model.Adapter is { } adapter && !Directory.Exists(adapter) ? $"the adapter folder {adapter} does not exist"
        : null;

    // Downloads (when needed) and loads `model`; the host unloads the model before it.
    private async Task LoadAsync(ReaderModel model, CancellationToken cancellationToken)
    {
        var clock = Stopwatch.StartNew();
        try
        {
            string folder = await FolderAsync(model, (message, download) => SetStatus(ReaderState.Downloading, message, _loaded, model.Id, download), cancellationToken);
            await _gate.WaitAsync(cancellationToken);
            try
            {
                if (_busy is { } busy)
                {
                    throw new InvalidOperationException(busy);
                }

                _folders[model.Id] = folder;
                SetStatus(ReaderState.Loading, $"Loading {model.Name} ({_settings.Weights} weights) onto {Device}…", _loaded, model.Id, seconds: clock.Elapsed.TotalSeconds);
                if (_loaded is { } before && !string.Equals(before, model.Id, StringComparison.OrdinalIgnoreCase))
                {
                    _host.Unload(before);                                       // its memory back before the next loads
                    _loaded = null;
                }

                using (_host.Acquire(model.Id))
                {
                }

                _loaded = model.Id;
                SetStatus(ReaderState.Ready, $"{model.Name} is ready (loaded in {clock.Elapsed.TotalSeconds:F0} s).", model.Id, seconds: clock.Elapsed.TotalSeconds);
                _logger.LogInformation("Loaded {Model} from {Folder} in {Seconds:F1} s", model.Id, folder, clock.Elapsed.TotalSeconds);
            }
            finally
            {
                _gate.Release();
            }
        }
        catch (OperationCanceledException)
        {
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Loading {Model} failed", model.Id);
            SetStatus(ReaderState.Failed, $"{model.Name} could not be loaded: {ex.Message}", _loaded);
        }
    }

    // One read: the page as the model sees it, the answer streamed, then the result.
    private void Read(LoadedReader loaded, ChatImage image, ReadRequest request, string? expected, ChannelWriter<SseItem<object>> output,
        CancellationToken cancellationToken)
    {
        var result = ReadPage(loaded, image, request, expected, output, cancellationToken);
        var reading = result.Reading;
        int generated = reading.Stats?.GeneratedTokens ?? 0;
        double decoding = Math.Max(1e-9, (reading.Elapsed - (reading.FirstText ?? TimeSpan.Zero)).TotalSeconds);
        output.TryWrite(new SseItem<object>(new
        {
            text = reading.Text,
            json = result.Json,
            repaired = result.Json is not null && result.Repaired,
            expected = expected is null ? null : AnswerJson.Show(expected).Json ?? expected,
            cer = result.Cer,
            seconds = Math.Round(reading.Elapsed.TotalSeconds, 2),
            firstTokenSeconds = reading.FirstText is { } f ? Math.Round(f.TotalSeconds, 2) : (double?)null,
            promptTokens = reading.Stats?.PromptTokens,
            generatedTokens = generated,
            tokensPerSecond = generated > 1 ? Math.Round((generated - 1) / decoding, 1) : (double?)null,
            truncated = reading.Truncated,
            model = loaded.Model.Id,
        }, "done"));
    }

    // A page read with the request's settings (the model's own where it gives none): with an output, the page as the model
    // sees it and the answer's text as it is generated go to it; the answer's JSON and, given the expected answer, its CER
    // (the library's metric over the JSON's values).
    private PageResult ReadPage(LoadedReader loaded, ChatImage image, ReadRequest request, string? expected, ChannelWriter<SseItem<object>>? output,
        CancellationToken cancellationToken)
    {
        var model = loaded.Model;
        var pipeline = ImageTransformPipeline.Parse(request.Preprocessing ?? model.Preprocessing);
        string prompt = string.IsNullOrWhiteSpace(request.Prompt) ? model.Prompt : request.Prompt;
        int maxTokens = Math.Clamp(request.MaxTokens ?? model.MaxTokens, 1, Math.Max(1, _settings.Context - 256));
        var read = new ImageReadRequest(image, prompt) { MaxTokens = maxTokens, Transforms = pipeline };
        if (output is not null)
        {
            SetStatus(ReaderState.Reading, $"{model.Name} is reading a page…", model.Id);
            var original = ChatImageDecoder.Decode(image);
            var seen = loaded.Reader.Prepared(read);
            output.TryWrite(new SseItem<object>(new
            {
                image = "data:image/png;base64," + Convert.ToBase64String(ImageEncoders.Get("png").Encode(seen)),
                width = seen.Width, height = seen.Height, originalWidth = original.Width, originalHeight = original.Height,
                preprocessing = pipeline.ToString(), prompt, maxTokens,
            }, "image"));
        }

        var reading = loaded.Reader.Read(read, output is null ? null : text => output.TryWrite(new SseItem<object>(new { text }, "token")), cancellationToken);
        var (json, repaired) = AnswerJson.Show(reading.Text);
        double? cer = expected is null ? null
            : TuningMetrics.Get(TuningMetrics.CharacterErrorRate).Score(AnswerTexts.Of(reading.Text, AnswerTexts.Values), AnswerTexts.Of(expected, AnswerTexts.Values)).Value;
        return new PageResult(reading, json, repaired, cer);
    }

    private sealed record PageResult(ImageReading Reading, string? Json, bool Repaired, double? Cer);

    private void SetStatus(ReaderState state, string message, string? model, string? loading = null, DownloadInfo? download = null, double? seconds = null)
    {
        lock (_statusLock)
        {
            _status = new ReaderStatus(state, message, model, loading, download, seconds, Device.ToString(), null);
        }
    }

    // "auto" → the best device found; else the device named.
    internal static Device ParseDevice(string? text) => text?.Trim().ToLowerInvariant() switch
    {
        null or "" or "auto" => Device.Default,
        "vulkan" => Device.Parse("vulkan:0"),
        var name => Device.Parse(name),
    };

    // An IProgress that reports on the caller's thread (Progress<T> would post to the thread pool, out of order).
    private sealed class Reporter<T>(Action<T> report) : IProgress<T>
    {
        public void Report(T value) => report(value);
    }
}

/// <summary>A loaded model of the app: the checkpoint (its weights as the settings say) and the library reader over it.</summary>
public sealed class LoadedReader : IDisposable
{
    private readonly Device _device;

    private LoadedReader(ReaderModel model, PretrainedModel pretrained, ImageReader reader, Device device)
    {
        Model = model;
        Pretrained = pretrained;
        Reader = reader;
        _device = device;
    }

    /// <summary>The model's settings.</summary>
    public ReaderModel Model { get; }

    /// <summary>The checkpoint.</summary>
    public PretrainedModel Pretrained { get; }

    /// <summary>The reader with the model's preprocessing.</summary>
    public ImageReader Reader { get; }

    /// <summary>Loads <paramref name="model"/> from <paramref name="folder"/> (its adapter merged as the weights are read).</summary>
    public static LoadedReader Create(ReaderModel model, string folder, LegalOcrSettings settings, Device device)
    {
        string weights = settings.Weights.ToLowerInvariant();
        var pretrained = PretrainedModel.Load(folder, new PretrainedOptions
        {
            Device = device,
            BFloat16 = weights is "bf16" or "bfloat16",
            Int8 = weights == "int8",
            Int4 = weights == "int4",
            MaxPositions = settings.Context,
            MergeAdapter = model.Adapter,
        });
        try
        {
            var reader = new ImageReader(pretrained, new ImageReaderOptions
            {
                Transforms = ImageTransformPipeline.Parse(model.Preprocessing),
                VisionOptions = model.Adapter is { } adapter && TuningImages.Read(adapter) is { VisionOptions.Count: > 0 } tuned ? tuned.VisionOptions : null,
                CacheFormat = settings.KeyValueCache.ToLowerInvariant() is "bf16" or "bfloat16" ? KeyValueFormat.BFloat16 : KeyValueFormat.Float32,
                ContextLength = settings.Context,
            });
            return new LoadedReader(model, pretrained, reader, device);
        }
        catch
        {
            pretrained.Dispose();
            throw;
        }
    }

    /// <summary>Disposes the reader and the checkpoint and gives the device's cached memory back.</summary>
    public void Dispose()
    {
        Reader.Dispose();
        Pretrained.Dispose();
        ComputeResources.ReleaseCachedMemory(_device);
    }
}
