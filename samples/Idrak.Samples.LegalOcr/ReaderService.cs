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
    private readonly ILogger<ReaderService> _logger;
    private readonly ModelHost<LoadedReader> _host;
    private readonly SemaphoreSlim _gate = new(1, 1);                   // one load or read at a time
    private readonly Lock _statusLock = new();
    private readonly Dictionary<string, string> _folders = new(StringComparer.OrdinalIgnoreCase);
    private ReaderStatus _status;
    private string? _loaded;
    private CancellationTokenSource? _loading;

    public ReaderService(LegalOcrSettings settings, ILogger<ReaderService> logger)
    {
        _settings = settings;
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

    /// <summary>The models of the switch.</summary>
    public IEnumerable<ModelInfo> Models() => _settings.Models.Select(m =>
    {
        string? unusable = Unusable(m);
        return new ModelInfo(m.Id, m.Name, m.About, m.Folder ?? m.Repo ?? "", m.Adapter, unusable is null, unusable, m.Preprocessing, m.Prompt, m.MaxTokens,
            string.Equals(_loaded, m.Id, StringComparison.OrdinalIgnoreCase));
    });

    /// <summary>Starts loading <paramref name="id"/> in the background (downloading it first when its files are missing); the status reports it.</summary>
    /// <exception cref="ArgumentException">No such model, or it cannot be loaded as configured.</exception>
    public void StartLoad(string id)
    {
        var model = Find(id);
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
                string id = _loaded ?? throw new InvalidOperationException("No model is loaded: pick one first.");
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

    public void Dispose()
    {
        _loading?.Cancel();
        _host.Dispose();
        _gate.Dispose();
    }

    // The model's settings by its id.
    private ReaderModel Find(string id) =>
        _settings.Models.FirstOrDefault(m => string.Equals(m.Id, id, StringComparison.OrdinalIgnoreCase))
        ?? throw new ArgumentException($"No model '{id}' (the settings have {string.Join(", ", _settings.Models.Select(m => m.Id))}).");

    // Why a model cannot be loaded as configured, or null.
    private static string? Unusable(ReaderModel model) =>
        model.Folder is null && model.Repo is null ? "neither Repo nor Folder is set"
        : model.Folder is { } folder && !Directory.Exists(folder) ? $"the folder {folder} does not exist"
        : model.Adapter is { } adapter && !Directory.Exists(adapter) ? $"the adapter folder {adapter} does not exist"
        : model.Id == "ours" && model.Adapter is null ? "no adapter yet: fine-tune one, then set its folder as Adapter"
        : null;

    // Downloads (when needed) and loads `model`; the host unloads the model before it.
    private async Task LoadAsync(ReaderModel model, CancellationToken cancellationToken)
    {
        var clock = Stopwatch.StartNew();
        try
        {
            string folder = model.Folder ?? await DownloadAsync(model, clock, cancellationToken);
            await _gate.WaitAsync(cancellationToken);
            try
            {
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

    // The model's files from Hugging Face into the library's cache (only the missing ones), with live progress.
    private async Task<string> DownloadAsync(ReaderModel model, Stopwatch clock, CancellationToken cancellationToken)
    {
        SetStatus(ReaderState.Downloading, $"Looking up {model.Repo} on Hugging Face…", _loaded, model.Id);
        var downloader = new Downloader
        {
            Progress = new Reporter<DownloadProgress>(p => SetStatus(ReaderState.Downloading, $"Downloading {model.Repo}: {p.File}", _loaded, model.Id,
                new DownloadInfo(p.File, p.Received, p.Total, clock.Elapsed.TotalSeconds))),
            Log = line => _logger.LogInformation("{Line}", line),
        };
        return await HuggingFaceModels.DownloadAsync(model.Repo!, token: HuggingFace.Token(_settings.HuggingFaceToken), downloader: downloader,
            cancellationToken: cancellationToken);
    }

    // One read: the page as the model sees it, the answer streamed, then the result.
    private void Read(LoadedReader loaded, ChatImage image, ReadRequest request, string? expected, ChannelWriter<SseItem<object>> output,
        CancellationToken cancellationToken)
    {
        var model = loaded.Model;
        var pipeline = ImageTransformPipeline.Parse(request.Preprocessing ?? model.Preprocessing);
        string prompt = string.IsNullOrWhiteSpace(request.Prompt) ? model.Prompt : request.Prompt;
        int maxTokens = Math.Clamp(request.MaxTokens ?? model.MaxTokens, 1, Math.Max(1, _settings.Context - 256));
        var read = new ImageReadRequest(image, prompt) { MaxTokens = maxTokens, Transforms = pipeline };
        SetStatus(ReaderState.Reading, $"{model.Name} is reading a page…", model.Id);

        var original = ChatImageDecoder.Decode(image);
        var seen = loaded.Reader.Prepared(read);
        output.TryWrite(new SseItem<object>(new
        {
            image = "data:image/png;base64," + Convert.ToBase64String(ImageEncoders.Get("png").Encode(seen)),
            width = seen.Width, height = seen.Height, originalWidth = original.Width, originalHeight = original.Height,
            preprocessing = pipeline.ToString(), prompt, maxTokens,
        }, "image"));

        var reading = loaded.Reader.Read(read, text => output.TryWrite(new SseItem<object>(new { text }, "token")), cancellationToken);
        var (json, repaired) = AnswerJson.Show(reading.Text);
        double? cer = expected is null ? null
            : TuningMetrics.Get(TuningMetrics.CharacterErrorRate).Score(AnswerTexts.Of(reading.Text, AnswerTexts.Values), AnswerTexts.Of(expected, AnswerTexts.Values)).Value;
        int generated = reading.Stats?.GeneratedTokens ?? 0;
        double decoding = Math.Max(1e-9, (reading.Elapsed - (reading.FirstText ?? TimeSpan.Zero)).TotalSeconds);
        output.TryWrite(new SseItem<object>(new
        {
            text = reading.Text,
            json,
            repaired = json is not null && repaired,
            expected = expected is null ? null : AnswerJson.Show(expected).Json ?? expected,
            cer,
            seconds = Math.Round(reading.Elapsed.TotalSeconds, 2),
            firstTokenSeconds = reading.FirstText is { } f ? Math.Round(f.TotalSeconds, 2) : (double?)null,
            promptTokens = reading.Stats?.PromptTokens,
            generatedTokens = generated,
            tokensPerSecond = generated > 1 ? Math.Round((generated - 1) / decoding, 1) : (double?)null,
            truncated = reading.Truncated,
            model = model.Id,
        }, "done"));
    }

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
