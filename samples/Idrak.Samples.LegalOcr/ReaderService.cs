// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Net.ServerSentEvents;
using System.Runtime.CompilerServices;
using System.Threading.Channels;
using Idrak.Data;
using Idrak.Abstraction.Generation;
using Idrak.Data.Abstractions;
using Idrak.Gemma3Vision;
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

/// <summary>The device's memory as the library measures it: in use by tensors, at its peak, the limit, and freed blocks the pool keeps for reuse.</summary>
public sealed record MemoryInfo(long InUse, long Peak, long? Limit, long Cached = 0);

/// <summary>The service's state for the page (polled while a model downloads or loads).</summary>
public sealed record ReaderStatus(ReaderState State, string Message, string? Model, string? Loading, DownloadInfo? Download, double? Seconds, string Device,
    MemoryInfo? Memory);

/// <summary>A model of the switch as the page shows it.</summary>
public sealed record ModelInfo(string Id, string Name, string? About, string Source, string? Adapter, bool Usable, string? Unusable, string Preprocessing,
    string Prompt, int MaxTokens, bool Loaded, bool PanAndScan, string? FullText);

/// <summary>
/// How transformers' generate() reads with a model: its generation_config.json, with transformers' defaults for what the
/// file leaves out (no sampling, temperature 1, top-k 50, top-p 1, no min-p, repetition penalty 1); what the Docker-Preset
/// reads with, beside the weights the app loaded (the container's are bfloat16).
/// </summary>
public sealed record TransformersGeneration(bool Found, string? File, bool Sample, float Temperature, int TopK, float TopP, float MinP, float RepeatPenalty,
    string Weights);

/// <summary>A page to read: the image (a data URL or base64), and what replaces the model's own settings.</summary>
/// <param name="Image">The scan as a data URL ("data:image/png;base64,…") or plain base64.</param>
/// <param name="Prompt">The instruction; the model's own when null or empty.</param>
/// <param name="Preprocessing">The image transforms (the library's pipeline syntax, "" for none); the model's own when null.</param>
/// <param name="MaxTokens">The most tokens the answer may take; the model's own when null.</param>
/// <param name="PanAndScan">Gemma 3's pan and scan (<c>do_pan_and_scan</c>: an elongated page also read as crops, each its own image); the model's own when null.</param>
/// <param name="RepeatPenalty">Divides recently seen tokens' scores (1 = off); the settings' when null.</param>
/// <param name="PresencePenalty">Subtracted from every recently seen token's score (0 = off); the settings' when null.</param>
/// <param name="FrequencyPenalty">Subtracted once per recent occurrence (0 = off); the settings' when null.</param>
/// <param name="RepeatLastN">How many recent tokens the penalties look at; the settings' when null.</param>
/// <param name="RepeatStop">End the answer when one phrase repeats this many times in a row (0 = off); the settings' when null.</param>
/// <param name="Sample">Sample the words instead of greedy reading; the settings' when null.</param>
/// <param name="Temperature">Sampling's temperature; the settings' when null.</param>
/// <param name="TopK">Sample among the k most likely words (0 = all); the settings' when null.</param>
/// <param name="TopP">Sample among the most likely words holding this probability (1 = off); the settings' when null.</param>
/// <param name="MinP">Drop words less than this fraction as likely as the best (0 = off); the settings' when null.</param>
/// <param name="Seed">The sampling seed (a new one each read when null; the result reports it).</param>
/// <param name="AddSpecialTokens">Encode the prompt with the tokenizer's special tokens added, as a transformers pipeline calling the processor on the rendered template does (Gemma's: a second &lt;bos&gt;); off when null.</param>
public sealed record ReadRequest(string? Image = null, string? Prompt = null, string? Preprocessing = null, int? MaxTokens = null, bool? PanAndScan = null,
    float? RepeatPenalty = null, float? PresencePenalty = null, float? FrequencyPenalty = null, int? RepeatLastN = null, int? RepeatStop = null,
    bool? Sample = null, float? Temperature = null, int? TopK = null, float? TopP = null, float? MinP = null, int? Seed = null, bool? AddSpecialTokens = null);

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
                return _status with { Memory = new MemoryInfo(usage.InUse, usage.Peak, usage.Limit, usage.Cached) };
            }
        }
    }

    /// <summary>The models of the switch: the settings' and the tuned adapters.</summary>
    public IEnumerable<ModelInfo> Models() => _settings.Models.Concat(_adapters.Models()).Select(m =>
    {
        string? unusable = Unusable(m);
        return new ModelInfo(m.Id, m.Name, m.About, m.Folder ?? m.Repo ?? "", m.Adapter, unusable is null, unusable, m.Preprocessing, m.Prompt, m.MaxTokens,
            string.Equals(_loaded, m.Id, StringComparison.OrdinalIgnoreCase), m.PanAndScan, m.FullText);
    });

    /// <summary>Starts loading <paramref name="id"/> in the background (downloading it first when its files are missing); the status reports it.</summary>
    /// <exception cref="ArgumentException">No such model, or it cannot be loaded as configured.</exception>
    /// <exception cref="InvalidOperationException">Fine-tuning has the device.</exception>
    public void StartLoad(string id)
    {
        var model = Loadable(id);
        _loading?.Cancel();
        var cancel = _loading = new CancellationTokenSource();
        _ = Task.Run(() => LoadAsync(model, cancel.Token));
    }

    /// <summary>
    /// Loads <paramref name="id"/> (downloading it first when its files are missing) and returns when it is ready (true)
    /// or could not be loaded (false; the status says why). The status reports it as it goes.
    /// </summary>
    /// <exception cref="ArgumentException">No such model, or it cannot be loaded as configured.</exception>
    /// <exception cref="InvalidOperationException">Fine-tuning has the device.</exception>
    public Task<bool> LoadAsync(string id, CancellationToken cancellationToken = default) => LoadReadyAsync(Loadable(id), cancellationToken);

    private async Task<bool> LoadReadyAsync(ReaderModel model, CancellationToken cancellationToken)
    {
        _loading?.Cancel();
        var cancel = _loading = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken);
        await Task.Run(() => LoadAsync(model, cancel.Token), CancellationToken.None);
        var status = Status;
        return status.State == ReaderState.Ready && string.Equals(status.Model, model.Id, StringComparison.OrdinalIgnoreCase);
    }

    // The model `id` when it can start loading now.
    private ReaderModel Loadable(string id)
    {
        var model = Find(id);
        if (_busy is { } busy)
        {
            throw new InvalidOperationException(busy);
        }

        return Unusable(model) is { } reason ? throw new ArgumentException($"{model.Name}: {reason}") : model;
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
                int batch = Math.Max(1, _settings.ReadBatch);
                for (int at = 0; at < chosen.Count; at += batch)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    var group = chosen.Skip(at).Take(batch).ToList();
                    SetStatus(ReaderState.Reading, group.Count == 1
                        ? $"{lease.Model.Model.Name} is evaluating: page {at + 1} of {chosen.Count}…"
                        : $"{lease.Model.Model.Name} is evaluating: pages {at + 1} to {at + group.Count} of {chosen.Count} together…", id);
                    var results = group.Count == 1
                        ? [ReadPage(lease.Model, group[0].Image, request, group[0].Expected, null, cancellationToken)]
                        : ReadPages(lease.Model, [.. group.Select(p => (p.Image, (string?)p.Expected))], request, cancellationToken);
                    for (int i = 0; i < group.Count; i++)
                    {
                        var (page, result) = (group[i], results[i]);
                        scores.Add(result.Cer ?? 1);
                        double left = clock.Elapsed.TotalSeconds / scores.Count * (chosen.Count - scores.Count);
                        channel.Writer.TryWrite(new SseItem<object>(new
                        {
                            // A batch's pages share its time: each page's share of it.
                            index = page.Index, name = page.Name, cer = result.Cer, seconds = Math.Round(result.Reading.Elapsed.TotalSeconds / group.Count, 2),
                            tokens = result.Reading.Stats?.GeneratedTokens ?? 0, truncated = result.Reading.Truncated,
                            looped = result.Reading.DoneReason == "repetition", json = result.Json is not null, repaired = result.Repaired, seed = result.Seed,
                            done = scores.Count, total = chosen.Count, meanCer = scores.Average(), secondsLeft = Math.Round(left), batch = group.Count,
                        }, "page"));
                    }
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

    /// <summary>
    /// How transformers' generate() would read with the model <paramref name="id"/>: its generation_config.json once its
    /// files are on disk (after a load, or a local folder), else transformers' defaults alone.
    /// </summary>
    public TransformersGeneration TransformersGeneration(string id)
    {
        var model = Find(id);
        string? folder = _folders.TryGetValue(model.Id, out var loaded) ? loaded : model.Folder;
        string? file = folder is null ? null : Path.Combine(folder, "generation_config.json");
        var config = file is not null && File.Exists(file) ? System.Text.Json.Nodes.JsonNode.Parse(File.ReadAllText(file)) as System.Text.Json.Nodes.JsonObject : null;
        T Value<T>(string key, T fallback) => config?[key] is System.Text.Json.Nodes.JsonValue v && v.TryGetValue(out T? x) ? x : fallback;
        string weights = _settings.Weights + (LoadedReader.HeadFormat(_settings) is null ? "" : ", bf16 head");
        return new TransformersGeneration(config is not null, config is null ? null : file, Value("do_sample", false), Value("temperature", 1f), Value("top_k", 50),
            Value("top_p", 1f), Value("min_p", 0f), Value("repetition_penalty", 1f), weights);
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
                SetStatus(ReaderState.Loading, $"Loading {model.Name} ({_settings.Weights} weights{(LoadedReader.HeadFormat(_settings) is null ? "" : ", bf16 head")}) onto {Device}…", _loaded, model.Id, seconds: clock.Elapsed.TotalSeconds);
                if (_loaded is { } before && !string.Equals(before, model.Id, StringComparison.OrdinalIgnoreCase))
                {
                    _host.Unload(before);                                       // its memory back before the next loads
                    _loaded = null;
                }

                double loadSeconds, warmSeconds;
                using (var lease = _host.Acquire(model.Id))
                {
                    loadSeconds = clock.Elapsed.TotalSeconds;
                    SetStatus(ReaderState.Loading, $"Warming up {model.Name}: one short read builds its encoder and measures the device…", _loaded, model.Id,
                        seconds: loadSeconds);
                    warmSeconds = WarmUp(lease.Model, cancellationToken);
                }

                _loaded = model.Id;
                SetStatus(ReaderState.Ready, $"{model.Name} is ready (loaded in {loadSeconds:F0} s, warmed up in {warmSeconds:F1} s).", model.Id,
                    seconds: clock.Elapsed.TotalSeconds);
                _logger.LogInformation("Loaded {Model} from {Folder} in {Seconds:F1} s, warmed up in {Warm:F1} s", model.Id, folder, loadSeconds, warmSeconds);
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
            SetStatus(ReaderState.Failed, $"{model.Name} could not be loaded: {ex.Message}{AccessHint(model, ex)}", _loaded);
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
            doneReason = reading.DoneReason,
            sampled = result.Seed is not null,
            seed = result.Seed,
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
        var read = PageRequest(loaded, image, request);
        var pipeline = read.Transforms!;
        string prompt = read.Prompt;
        int maxTokens = read.MaxTokens;
        bool? pan = PanAndScan(model, request);
        var vision = read.VisionOptions;
        if (output is not null)
        {
            SetStatus(ReaderState.Reading, $"{model.Name} is reading a page…", model.Id);
            // The original size from the file's header (nothing decoded); the page as the model sees it, decoded and
            // transformed once: the read below takes the same pixels (ChatImages.Read).
            var original = ChatImageDecoder.Size(image.Data.Span) ?? DecodedSize(image);
            var seen = loaded.Reader.Prepared(read);
            // The crops Gemma 3's pan and scan reads beside the whole page (as rectangles of the page the model sees).
            var crops = pan == true && loaded.Reader.Vision.Family == Gemma3VisionFamily.Architecture
                ? Gemma3PanAndScan.Default.With(vision).Crops(seen.Height, seen.Width).Select(c => new { top = c.Top, left = c.Left, height = c.Height, width = c.Width }).ToList()
                : null;
            output.TryWrite(new SseItem<object>(new
            {
                image = "data:image/png;base64," + Convert.ToBase64String(ImageEncoders.Get("png").Encode(seen)),
                width = seen.Width, height = seen.Height, originalWidth = original.Width, originalHeight = original.Height,
                preprocessing = pipeline.ToString(), prompt, maxTokens, panAndScan = pan == true, crops,
            }, "image"));
        }

        ImageReading reading;
        try
        {
            reading = loaded.Reader.Read(read, output is null ? null : text => output.TryWrite(new SseItem<object>(new { text }, "token")), cancellationToken);
        }
        catch
        {
            // Nothing of a failed or stopped read stays on the device: the memory pool's freed blocks go back to the driver.
            // After a read that finished they stay in the pool, which the next read reuses (the reader keeps no attention
            // cache; unloading the model, and fine-tuning taking the device, give everything back).
            ComputeResources.ReleaseCachedMemory(Device);
            throw;
        }

        return Scored(reading, read, expected);
    }

    // Pages read together (ImageReader.ReadBatch), each scored as ReadPage scores it; when the device has no room for the
    // batch, the pages are read one by one.
    private IReadOnlyList<PageResult> ReadPages(LoadedReader loaded, IReadOnlyList<(ChatImage Image, string? Expected)> pages, ReadRequest request,
        CancellationToken cancellationToken)
    {
        var reads = pages.Select(p => PageRequest(loaded, p.Image, request)).ToList();
        IReadOnlyList<ImageReading> readings;
        try
        {
            readings = loaded.Reader.ReadBatch(reads, null, cancellationToken);
        }
        catch (ResourceLimitExceededException)
        {
            ComputeResources.ReleaseCachedMemory(Device);
            _logger.LogWarning("No room on the device to read {Pages} pages together; reading them one by one", pages.Count);
            return [.. pages.Select(p => ReadPage(loaded, p.Image, request, p.Expected, null, cancellationToken))];
        }
        catch
        {
            ComputeResources.ReleaseCachedMemory(Device);                  // as ReadPage: nothing of a failed read stays on the device
            throw;
        }

        return [.. readings.Select((reading, i) => Scored(reading, reads[i], pages[i].Expected))];
    }

    // A read's JSON and, given the expected answer, its CER (the library's metric over the JSON's values).
    private static PageResult Scored(ImageReading reading, ImageReadRequest read, string? expected)
    {
        var (json, repaired) = AnswerJson.Show(reading.Text);
        double? cer = expected is null ? null
            : TuningMetrics.Get(TuningMetrics.CharacterErrorRate).Score(AnswerTexts.Of(reading.Text, AnswerTexts.Values), AnswerTexts.Of(expected, AnswerTexts.Values)).Value;
        return new PageResult(reading, json, repaired, cer, read.Sample ? read.Seed : null);
    }

    // Pan and scan as the read asks, else the model's setting (null: an adapter's tuned vision options stand).
    private static bool? PanAndScan(ReaderModel model, ReadRequest request) => request.PanAndScan ?? (model.PanAndScan ? true : null);

    // The vision options of pan and scan on and off, made once (pages read together share them, so their images go
    // through the encoder in one call).
    private static readonly VisionOptions PanOn = VisionOptions.Parse([$"{Gemma3PanAndScan.EnableKey}=true"]);
    private static readonly VisionOptions PanOff = VisionOptions.Parse([$"{Gemma3PanAndScan.EnableKey}=false"]);

    // The read of a page with the request's settings (the model's own where it gives none); a sampled read gets a seed.
    private ImageReadRequest PageRequest(LoadedReader loaded, ChatImage image, ReadRequest request)
    {
        var model = loaded.Model;
        var pipeline = ImageTransformPipeline.Parse(request.Preprocessing ?? model.Preprocessing);
        string prompt = string.IsNullOrWhiteSpace(request.Prompt) ? model.Prompt : request.Prompt;
        int maxTokens = Math.Clamp(request.MaxTokens ?? model.MaxTokens, 1, Math.Max(1, _settings.Context - 256));
        var vision = PanAndScan(model, request) is { } on ? on ? PanOn : PanOff : null;
        var g = _settings.Generation;
        var read = new ImageReadRequest(image, prompt)
        {
            MaxTokens = maxTokens, Transforms = pipeline, VisionOptions = vision,
            RepeatPenalty = Math.Clamp(request.RepeatPenalty ?? g.RepeatPenalty, 0.5f, 3f), PresencePenalty = request.PresencePenalty ?? g.PresencePenalty,
            FrequencyPenalty = request.FrequencyPenalty ?? g.FrequencyPenalty, RepeatLastN = Math.Clamp(request.RepeatLastN ?? g.RepeatLastN, 0, 4096),
            RepeatStop = Math.Clamp(request.RepeatStop ?? g.RepeatStop, 0, 100), AddSpecialTokens = request.AddSpecialTokens ?? false,
        };
        if (request.Sample ?? g.Sample)
        {
            // A seed for every sampled read (the page's, else a new one), shown with the result so a read can be repeated.
            read = read with
            {
                Sample = true, Temperature = request.Temperature ?? g.Temperature, TopK = request.TopK ?? g.TopK, TopP = request.TopP ?? g.TopP,
                MinP = request.MinP ?? g.MinP, Seed = request.Seed ?? Random.Shared.Next(),
            };
        }

        return read;
    }

    // The size of an image whose header gives none: decoded.
    private static (int Width, int Height) DecodedSize(ChatImage image)
    {
        var decoded = ChatImageDecoder.Decode(image);
        return (decoded.Width, decoded.Height);
    }

    // A loaded model's first read pays for what is built on first use (the vision encoder, the device's kernels, the
    // measured choices for these shapes): one short read of a blank page with the model's own preprocessing and prompt,
    // so the first real page is a warm one (rule 76: cold and warm apart). A failure is logged and leaves the model ready.
    private double WarmUp(LoadedReader loaded, CancellationToken cancellationToken)
    {
        var clock = Stopwatch.StartNew();
        try
        {
            const int Side = 64;
            var blank = new ImageData(Enumerable.Repeat(1f, Side * Side).ToArray(), 1, Side, Side);
            var page = ChatImage.FromBytes(ImageEncoders.Get("png").Encode(blank));
            var model = loaded.Model;
            loaded.Reader.Read(new ImageReadRequest(page, model.Prompt)
            {
                MaxTokens = WarmUpTokens, Transforms = ImageTransformPipeline.Parse(model.Preprocessing), RepeatStop = 0,
            }, null, cancellationToken);
        }
        catch (Exception ex) when (ex is not OperationCanceledException)
        {
            _logger.LogWarning("Warming up {Model} failed (its first page reads cold): {Message}", loaded.Model.Id, ex.Message);
            ComputeResources.ReleaseCachedMemory(Device);
        }

        return clock.Elapsed.TotalSeconds;
    }

    // Tokens of the warm-up read: past the first decoding steps (the decoding graph is recorded from the second).
    private const int WarmUpTokens = 4;

    // One page's reading, with its seed when it was sampled.
    private sealed record PageResult(ImageReading Reading, string? Json, bool Repaired, double? Cer, int? Seed);

    private void SetStatus(ReaderState state, string message, string? model, string? loading = null, DownloadInfo? download = null, double? seconds = null)
    {
        lock (_statusLock)
        {
            _status = new ReaderStatus(state, message, model, loading, download, seconds, Device.ToString(), null);
        }
    }

    // What to do when Hugging Face refuses a gated model's files.
    internal static string AccessHint(ReaderModel model, Exception ex) =>
        model.Repo is { } repo && ex is HttpRequestException { StatusCode: System.Net.HttpStatusCode.Unauthorized or System.Net.HttpStatusCode.Forbidden }
            ? $" To use it: accept its license at https://huggingface.co/{repo} with your account, make a read token at https://huggingface.co/settings/tokens, "
              + "and start the app with HF_TOKEN set (or LegalOcr:HuggingFaceToken in appsettings.json)."
            : "";

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

    /// <summary>The head's own format: bfloat16 beside int8 or int4 projections when the settings ask for it, else null (as the projections).</summary>
    public static string? HeadFormat(LegalOcrSettings settings) =>
        settings.Weights.ToLowerInvariant() is "int8" or "int4" && settings.HeadWeights.ToLowerInvariant() is "bf16" or "bfloat16" ? "bfloat16" : null;

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
            HeadFormatName = HeadFormat(settings),
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
                EncoderPrecision = settings.EncoderPrecision.ToLowerInvariant() is "bf16" or "bfloat16" ? MatMulPrecision.BFloat16 : null,
                PrefillPrecision = settings.PrefillPrecision.ToLowerInvariant() is "bf16" or "bfloat16" ? MatMulPrecision.BFloat16 : null,
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
