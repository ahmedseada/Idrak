// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Threading.Channels;
using Idrak.Diagnostics;
using Idrak.Generation;
using Idrak.Layers;

namespace Idrak.Inference;

/// <summary>Thrown when a model's <c>QueueLimit</c> is reached.</summary>
public sealed class InferenceQueueFullException(string message) : InvalidOperationException(message);

/// <summary>What kind of model an engine entry is.</summary>
public enum EngineModelKind
{
    /// <summary>A <see cref="Predictor{TIn, TOut}"/>.</summary>
    Predictor,

    /// <summary>A <see cref="TextGenerator"/>.</summary>
    Text,

    /// <summary>A <see cref="ChatGenerator"/>.</summary>
    Chat,
}

/// <summary>The state of one model in the engine.</summary>
/// <param name="Name">The model's name.</param>
/// <param name="Kind">Predictor, text or chat.</param>
/// <param name="Loaded">Whether its copies are in memory.</param>
/// <param name="Instances">How many copies are loaded when it is loaded.</param>
/// <param name="Running">Requests being processed now.</param>
/// <param name="Queued">Requests waiting (for a batch, a free copy, or the load).</param>
/// <param name="ExpiresAt">When the keep-alive will unload it, if it is idle and a keep-alive is set.</param>
public sealed record ModelStatus(string Name, EngineModelKind Kind, bool Loaded, int Instances, int Running, int Queued, DateTimeOffset? ExpiresAt);

/// <summary>What a loaded model is.</summary>
/// <param name="Name">The model's name.</param>
/// <param name="Kind">Predictor, text or chat.</param>
/// <param name="Parameters">Trainable values in one copy.</param>
/// <param name="Device">Where the copies run.</param>
/// <param name="ContextLength">The context length (text and chat models), or null.</param>
public sealed record ModelDescription(string Name, EngineModelKind Kind, long Parameters, Device Device, int? ContextLength);

/// <summary>Request statistics of one model since the engine started.</summary>
/// <param name="Requests">Completed requests.</param>
/// <param name="Rejected">Requests refused because the queue was full.</param>
/// <param name="Failed">Requests that timed out, were cancelled or threw.</param>
/// <param name="AverageLatency">Mean end-to-end time of the last 1,024 completed requests (queue wait included).</param>
/// <param name="P95Latency">95th percentile of the same.</param>
/// <param name="AverageQueueWait">Mean time the last 1,024 requests waited before running.</param>
/// <param name="Rows">Rows predicted (predictors) or tokens generated (text and chat).</param>
/// <param name="RowsPerSecond">Rows (or tokens) per second of model time.</param>
/// <param name="AverageBatchSize">Rows per model call (predictors; 1 for generation).</param>
public sealed record EngineStats(long Requests, long Rejected, long Failed, TimeSpan AverageLatency, TimeSpan P95Latency, TimeSpan AverageQueueWait,
    long Rows, double RowsPerSecond, double AverageBatchSize);

/// <summary>
/// Hosts named models and serves requests to them from any kind of .NET application: loading (at start or on first
/// use), warm-up, several copies, micro-batching, queue limits, timeouts, keep-alive unloading, statistics and
/// telemetry — each off unless set when the model is added. Predictors run concurrently (they are thread-safe); a text
/// or chat copy runs one generation at a time, because it owns its KV cache.
/// </summary>
/// <example>
/// <code>
/// await using var engine = await InferenceEngine.Create()
///     .Predictor&lt;House, float&gt;("house-price", "models/house-price.ikm", p =&gt; p
///         .Input&lt;House&gt;(h =&gt; [h.Area, h.Beds, h.Baths])
///         .Output(v =&gt; v[0])
///         .Batching(maxBatch: 256, maxWait: TimeSpan.FromMilliseconds(5)))
///     .ChatModel("my-gpt", "models/chat.ikm", "chars", c =&gt; c.Instances(2).KeepAlive(TimeSpan.FromMinutes(5)))
///     .BuildAsync();
///
/// float price = await engine.PredictAsync&lt;House, float&gt;("house-price", house);
/// </code>
/// </example>
public sealed class InferenceEngine : IAsyncDisposable
{
    private readonly Dictionary<string, EngineModel> _models;

    internal InferenceEngine(Dictionary<string, EngineModel> models) => _models = models;

    /// <summary>Starts an engine configuration.</summary>
    public static InferenceEngineBuilder Create() => new();

    /// <summary>The state of every model.</summary>
    public IReadOnlyList<ModelStatus> Models => [.. _models.Values.Select(m => m.Status())];

    /// <summary>The names of the models.</summary>
    public IReadOnlyCollection<string> Names => _models.Keys;

    /// <summary>The kind of the model named <paramref name="name"/>.</summary>
    public EngineModelKind KindOf(string name) => Model(name).Kind;

    /// <summary>Request statistics of <paramref name="name"/>.</summary>
    public EngineStats Stats(string name) => Model(name).Statistics.Snapshot();

    /// <summary>Loads <paramref name="name"/> now (if it is not loaded).</summary>
    public Task LoadAsync(string name, CancellationToken cancellationToken = default) => Model(name).EnsureLoadedAsync(cancellationToken);

    /// <summary>
    /// Changes how long <paramref name="name"/> stays loaded after its last request (null: forever; <see cref="TimeSpan.Zero"/>:
    /// unload after each request), for example from a request's <c>keep_alive</c>. Needs a model the engine can reload.
    /// </summary>
    public void KeepAlive(string name, TimeSpan? idle) => Model(name).SetKeepAlive(idle);

    /// <summary>Unloads <paramref name="name"/> now, freeing its memory, once running requests finish; the next request loads it again.</summary>
    public Task UnloadAsync(string name) => Model(name).UnloadAsync(force: true);

    // ------------------------------------------------------------------ predictors

    /// <summary>A typed handle for the predictor named <paramref name="name"/>.</summary>
    public IPredictor<TIn, TOut> Predictor<TIn, TOut>(string name) => Model<PredictorModel<TIn, TOut>>(name);

    /// <summary>Predicts one input with the predictor named <paramref name="name"/>.</summary>
    public ValueTask<TOut> PredictAsync<TIn, TOut>(string name, TIn input, CancellationToken cancellationToken = default) =>
        Predictor<TIn, TOut>(name).PredictAsync(input, cancellationToken);

    /// <summary>Predicts several inputs as one batch.</summary>
    public ValueTask<IReadOnlyList<TOut>> PredictAsync<TIn, TOut>(string name, IReadOnlyList<TIn> inputs, CancellationToken cancellationToken = default) =>
        Predictor<TIn, TOut>(name).PredictAsync(inputs, cancellationToken);

    /// <summary>Streams predictions for a large input sequence, <paramref name="batchSize"/> rows per model call, in input order.</summary>
    public async IAsyncEnumerable<TOut> PredictManyAsync<TIn, TOut>(string name, IEnumerable<TIn> inputs, int batchSize,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        var model = Model<PredictorModel<TIn, TOut>>(name);
        foreach (var chunk in inputs.Chunk(batchSize))
        {
            foreach (var result in await model.PredictBatchAsync(chunk, cancellationToken).ConfigureAwait(false))
            {
                yield return result;
            }
        }
    }

    /// <summary>Streams predictions for an asynchronous input sequence, <paramref name="batchSize"/> rows per model call.</summary>
    public async IAsyncEnumerable<TOut> PredictManyAsync<TIn, TOut>(string name, IAsyncEnumerable<TIn> inputs, int batchSize,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(batchSize);
        var model = Model<PredictorModel<TIn, TOut>>(name);
        var chunk = new List<TIn>(batchSize);
        await foreach (var input in inputs.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            chunk.Add(input);
            if (chunk.Count == batchSize)
            {
                foreach (var result in await model.PredictBatchAsync([.. chunk], cancellationToken).ConfigureAwait(false))
                {
                    yield return result;
                }

                chunk.Clear();
            }
        }

        if (chunk.Count > 0)
        {
            foreach (var result in await model.PredictBatchAsync([.. chunk], cancellationToken).ConfigureAwait(false))
            {
                yield return result;
            }
        }
    }

    // ------------------------------------------------------------------ text and chat

    /// <summary>Generates the whole continuation of <paramref name="prompt"/> with the text or chat model named <paramref name="name"/>.</summary>
    public async Task<(string Text, string DoneReason, GenerationStats Stats)> GenerateAsync(string name, string prompt, GenerationOptions options,
        CancellationToken cancellationToken = default)
    {
        var text = new System.Text.StringBuilder();
        await foreach (var chunk in StreamAsync(name, prompt, options, cancellationToken).ConfigureAwait(false))
        {
            text.Append(chunk.Text);
            if (chunk.Done)
            {
                return (text.ToString(), chunk.DoneReason!, chunk.Stats!);
            }
        }

        throw new InvalidOperationException("Generation ended without a final chunk.");
    }

    /// <summary>Streams the continuation of <paramref name="prompt"/>.</summary>
    public IAsyncEnumerable<GenerationChunk> StreamAsync(string name, string prompt, GenerationOptions options, CancellationToken cancellationToken = default) =>
        Model<GenerativeModel>(name).StreamTextAsync(prompt, options, cancellationToken);

    /// <summary>Answers a chat request with the chat model named <paramref name="name"/>.</summary>
    public async Task<ChatChunk> ChatAsync(string name, ChatRequest request, CancellationToken cancellationToken = default)
    {
        ChatChunk? last = null;
        await foreach (var chunk in StreamChatAsync(name, request, cancellationToken).ConfigureAwait(false))
        {
            last = chunk;
        }

        return last ?? throw new InvalidOperationException("The chat ended without a final chunk.");
    }

    /// <summary>Streams the answer to a chat request.</summary>
    public IAsyncEnumerable<ChatChunk> StreamChatAsync(string name, ChatRequest request, CancellationToken cancellationToken = default) =>
        Model<GenerativeModel>(name).StreamChatAsync(request, cancellationToken);

    /// <summary>The chat model named <paramref name="name"/> as an <see cref="IChatModel"/> (for <see cref="Generation.Conversation"/>).</summary>
    public IChatModel ChatModel(string name) => Model<GenerativeModel>(name).AsChatModel();

    /// <summary>The tools registered with the chat model named <paramref name="name"/>, or null.</summary>
    public ToolRegistry? ToolsOf(string name) => Model<GenerativeModel>(name).Tools;

    /// <summary>Describes the model named <paramref name="name"/> (loads it if needed).</summary>
    public async Task<ModelDescription> DescribeAsync(string name, CancellationToken cancellationToken = default)
    {
        var model = Model(name);
        await model.EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        return model.Description ?? throw new InvalidOperationException($"'{name}' was unloaded while it was being described.");
    }

    /// <summary>The context length of the text or chat model named <paramref name="name"/> (loads it if needed).</summary>
    public async Task<int> ContextLengthAsync(string name, CancellationToken cancellationToken = default)
    {
        var model = Model<GenerativeModel>(name);
        await model.EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        return model.ContextLength;
    }

    /// <summary>A conversation with the chat model named <paramref name="name"/>; each request uses a free copy of the model.</summary>
    public Conversation Conversation(string name, Func<ConversationBuilder, ConversationBuilder> configure) =>
        configure(Generation.Conversation.For(ChatModel(name))).Build();

    private int _disposed;

    /// <summary>Unloads every model (safe to call more than once).</summary>
    public async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
        {
            return;
        }

        foreach (var model in _models.Values)
        {
            await model.DisposeAsync().ConfigureAwait(false);
        }
    }

    private EngineModel Model(string name) =>
        _models.TryGetValue(name, out var model) ? model : throw new KeyNotFoundException($"The engine has no model named '{name}'. Models: {string.Join(", ", _models.Keys)}.");

    private T Model<T>(string name) where T : EngineModel => Model(name) as T
        ?? throw new InvalidOperationException($"'{name}' is a {Model(name).Kind} model; this call needs a {(typeof(T) == typeof(GenerativeModel) ? "text or chat" : "predictor with matching input and output types")} model.");
}

/// <summary>
/// Configures an <see cref="InferenceEngine"/>. Every model has a name and a source; everything else is set per model
/// (see <see cref="PredictorBuilder{TIn, TOut}"/> and <see cref="GenerativeModelBuilder"/>) and is off unless set.
/// </summary>
public sealed class InferenceEngineBuilder
{
    private readonly List<Func<EngineOptions, EngineModel>> _models = [];
    private readonly HashSet<string> _names = [];
    private readonly EngineOptions _options = new();

    internal InferenceEngineBuilder()
    {
    }

    // ------------------------------------------------------------------ predictors

    /// <summary>A predictor from a package (<see cref="Predictor{TIn, TOut}.Save"/> or <see cref="ModelPackage"/> with an architecture); the stored scalers and settings are restored first.</summary>
    public InferenceEngineBuilder Predictor<TIn, TOut>(string name, string packagePath,
        Func<PredictorBuilder<float[], float[]>, PredictorBuilder<TIn, TOut>> configure)
    {
        var template = PredictorBuilder<float[], float[]>.Template();
        using (var package = ModelPackage.Open(packagePath))
        {
            if (!package.Contains(PackageEntryKind.Architecture, ModelPackage.DefaultModelName))
            {
                throw new InvalidOperationException($"{packagePath} has no architecture; use the Predictor overload with a model factory.");
            }

            Inference.Predictor.Restore(package, template.Settings);
        }

        var typed = configure(template);
        return AddPredictor(name, typed, reloadable: true, () =>
        {
            using var package = ModelPackage.Open(packagePath);
            return package.BuildModel(device: typed.Settings.Device);
        }, ownsModel: true);
    }

    /// <summary>A predictor whose model is made by <paramref name="load"/> (called once per copy, and again after a keep-alive unload).</summary>
    public InferenceEngineBuilder Predictor<TIn, TOut>(string name, Func<Module> load,
        Func<PredictorBuilder<float[], float[]>, PredictorBuilder<TIn, TOut>> configure) =>
        AddPredictor(name, configure(PredictorBuilder<float[], float[]>.Template()), reloadable: true, load, ownsModel: true);

    /// <summary>A predictor for a model you already created (one copy; it is never unloaded or disposed by the engine).</summary>
    public InferenceEngineBuilder Predictor<TIn, TOut>(string name, Module model,
        Func<PredictorBuilder<float[], float[]>, PredictorBuilder<TIn, TOut>> configure) =>
        AddPredictor(name, configure(PredictorBuilder<float[], float[]>.Template()), reloadable: false, () => model, ownsModel: false);

    /// <summary>A predictor built by <paramref name="network"/> with weights from <paramref name="weightsPath"/> (<see cref="Idrak.ModuleFiles.Save(Module, string)"/>).</summary>
    public InferenceEngineBuilder Predictor<TIn, TOut>(string name, NetworkBuilder network, string weightsPath,
        Func<PredictorBuilder<float[], float[]>, PredictorBuilder<TIn, TOut>> configure)
    {
        var typed = configure(PredictorBuilder<float[], float[]>.Template());
        return AddPredictor(name, typed, reloadable: true, () =>
        {
            var model = network.Build();
            if (typed.Settings.Device is { } device)
            {
                model.To(device);
            }

            model.Load(weightsPath);
            return model;
        }, ownsModel: true);
    }

    // ------------------------------------------------------------------ text and chat models

    /// <summary>A text model from a package holding a network (with its architecture) and a tokenizer named <paramref name="tokenizer"/>.</summary>
    public InferenceEngineBuilder TextModel(string name, string packagePath, string tokenizer, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddGenerative(name, EngineModelKind.Text, configure, device => FromPackage(packagePath, tokenizer, device), reloadable: true, owns: true);

    /// <summary>A text model made by <paramref name="load"/> (called once per copy; the engine disposes the model on unload).</summary>
    public InferenceEngineBuilder TextModel(string name, Func<TextGenerator> load, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddGenerative(name, EngineModelKind.Text, configure, _ => load(), reloadable: true, owns: true);

    /// <summary>A text model you already created (one copy; never unloaded or disposed by the engine).</summary>
    public InferenceEngineBuilder TextModel(string name, TextGenerator generator, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddGenerative(name, EngineModelKind.Text, configure, _ => generator, reloadable: false, owns: false);

    /// <summary>A chat model from a package holding a network (with its architecture) and a tokenizer named <paramref name="tokenizer"/>.</summary>
    public InferenceEngineBuilder ChatModel(string name, string packagePath, string tokenizer, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddGenerative(name, EngineModelKind.Chat, configure, device => FromPackage(packagePath, tokenizer, device), reloadable: true, owns: true);

    /// <summary>A chat model whose text generator is made by <paramref name="load"/> (called once per copy).</summary>
    public InferenceEngineBuilder ChatModel(string name, Func<TextGenerator> load, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddGenerative(name, EngineModelKind.Chat, configure, _ => load(), reloadable: true, owns: true);

    /// <summary>A chat model over a text generator you already created (one copy; never unloaded or disposed by the engine).</summary>
    public InferenceEngineBuilder ChatModel(string name, TextGenerator generator, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure = null) =>
        AddGenerative(name, EngineModelKind.Chat, configure, _ => generator, reloadable: false, owns: false);

    // ------------------------------------------------------------------ engine settings

    /// <summary>Loads models on their first request instead of in <see cref="BuildAsync"/>.</summary>
    public InferenceEngineBuilder LoadOnFirstUse()
    {
        _options.LoadOnFirstUse = true;
        return this;
    }

    /// <summary>Publishes <see cref="EngineEvent"/>s (seen by hooks subscribed to <see cref="TelemetryLevel.Engine"/>).</summary>
    public InferenceEngineBuilder Telemetry()
    {
        _options.PublishTelemetry = true;
        return this;
    }

    /// <summary>The clock for keep-alive timers (replaceable in tests).</summary>
    public InferenceEngineBuilder TimeProvider(TimeProvider time)
    {
        _options.Time = time;
        return this;
    }

    /// <summary>Creates the engine and, unless <see cref="LoadOnFirstUse"/> was set, loads every model (warming up those with a warm-up).</summary>
    public async Task<InferenceEngine> BuildAsync(CancellationToken cancellationToken = default)
    {
        var models = _models.Select(create => create(_options)).ToDictionary(m => m.Name);
        var engine = new InferenceEngine(models);
        if (!_options.LoadOnFirstUse)
        {
            try
            {
                foreach (var model in models.Values)
                {
                    await model.EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
                }
            }
            catch
            {
                await engine.DisposeAsync().ConfigureAwait(false);
                throw;
            }
        }

        return engine;
    }

    private InferenceEngineBuilder AddPredictor<TIn, TOut>(string name, PredictorBuilder<TIn, TOut> builder, bool reloadable, Func<Module> load, bool ownsModel)
    {
        var s = builder.Settings;
        CheckName(name);
        Check(name, reloadable, s.Instances ?? 1, s.KeepAliveSet);
        _models.Add(options => new PredictorModel<TIn, TOut>(name, options, builder, load, ownsModel, reloadable));
        return this;
    }

    private InferenceEngineBuilder AddGenerative(string name, EngineModelKind kind, Func<GenerativeModelBuilder, GenerativeModelBuilder>? configure,
        Func<Device?, TextGenerator> load, bool reloadable, bool owns)
    {
        CheckName(name);
        var settings = (configure ?? (b => b))(new GenerativeModelBuilder());
        Check(name, reloadable, settings.InstanceCount ?? 1, settings.KeepAliveSet);
        if (kind == EngineModelKind.Text && (settings.ChatTemplate is not null || settings.ToolRegistry is not null))
        {
            throw new InvalidOperationException($"'{name}' is a text model; Template and Tools apply to chat models.");
        }

        _models.Add(options => new GenerativeModel(name, kind, options, settings, load, owns, reloadable));
        return this;
    }

    private void CheckName(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (!_names.Add(name))
        {
            throw new ArgumentException($"A model named '{name}' was already added.", nameof(name));
        }
    }

    private static void Check(string name, bool reloadable, int instances, bool keepAlive)
    {
        if (!reloadable && (instances > 1 || keepAlive))
        {
            throw new InvalidOperationException($"'{name}' uses a model object you created, which the engine cannot copy or reload; " +
                "Instances and KeepAlive need a package or a factory.");
        }
    }

    private static TextGenerator FromPackage(string path, string tokenizer, Device? device)
    {
        using var package = ModelPackage.Open(path);
        return package.TextGenerator(tokenizer, device: device);
    }
}

/// <summary>Settings for a text or chat model in the engine. Each is off (or the underlying class's own default) unless set.</summary>
public sealed class GenerativeModelBuilder
{
    internal GenerativeModelBuilder()
    {
    }

    internal int? InstanceCount { get; private set; }
    internal TimeSpan? KeepAliveTime { get; private set; }
    internal bool KeepAliveSet { get; private set; }
    internal int? QueueLimitCount { get; private set; }
    internal TimeSpan? TimeoutLimit { get; private set; }
    internal string? WarmUpPrompt { get; private set; }
    internal Device? TargetDevice { get; private set; }
    internal ChatTemplate? ChatTemplate { get; private set; }
    internal ToolRegistry? ToolRegistry { get; private set; }

    /// <summary>Loads <paramref name="count"/> copies, so that many generations run at the same time.</summary>
    public GenerativeModelBuilder Instances(int count)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(count);
        InstanceCount = count;
        return this;
    }

    /// <summary>Unloads the model after it has been idle this long (null keeps it forever, <see cref="TimeSpan.Zero"/> unloads after each request).</summary>
    public GenerativeModelBuilder KeepAlive(TimeSpan? idle)
    {
        KeepAliveTime = idle;
        KeepAliveSet = true;
        return this;
    }

    /// <summary>Rejects new requests while <paramref name="waiting"/> are already waiting for a free copy.</summary>
    public GenerativeModelBuilder QueueLimit(int waiting)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(waiting);
        QueueLimitCount = waiting;
        return this;
    }

    /// <summary>Stops a request that has not finished after <paramref name="limit"/> (queue wait included).</summary>
    public GenerativeModelBuilder Timeout(TimeSpan limit)
    {
        TimeoutLimit = limit;
        return this;
    }

    /// <summary>Generates one token from <paramref name="prompt"/> right after each copy loads.</summary>
    public GenerativeModelBuilder WarmUp(string prompt)
    {
        WarmUpPrompt = prompt;
        return this;
    }

    /// <summary>The device for package sources (factories place their models themselves).</summary>
    public GenerativeModelBuilder Device(Device device)
    {
        TargetDevice = device;
        return this;
    }

    /// <summary>Chat models: the prompt format (the <see cref="ChatGenerator"/> constructor's <c>template</c>).</summary>
    public GenerativeModelBuilder Template(ChatTemplate template)
    {
        ChatTemplate = template;
        return this;
    }

    /// <summary>Chat models: tools that can be run on the server (for example by the ASP.NET Core chat endpoint with server-side execution).</summary>
    public GenerativeModelBuilder Tools(ToolRegistry tools)
    {
        ToolRegistry = tools;
        return this;
    }
}

internal sealed class EngineOptions
{
    public bool LoadOnFirstUse { get; set; }
    public bool PublishTelemetry { get; set; }
    public TimeProvider Time { get; set; } = TimeProvider.System;
}

/// <summary>Counts and latencies of one model's requests.</summary>
internal sealed class EngineStatistics
{
    private readonly Lock _lock = new();
    private readonly double[] _latencies = new double[1024];
    private readonly double[] _waits = new double[1024];
    private long _requests, _rejected, _failed, _rows, _calls, _count;
    private double _modelSeconds;

    public void Completed(TimeSpan latency, TimeSpan wait)
    {
        lock (_lock)
        {
            _latencies[_count % _latencies.Length] = latency.TotalMilliseconds;
            _waits[_count % _waits.Length] = wait.TotalMilliseconds;
            _count++;
            _requests++;
        }
    }

    public void ModelCall(int rows, TimeSpan duration)
    {
        lock (_lock)
        {
            _rows += rows;
            _calls++;
            _modelSeconds += duration.TotalSeconds;
        }
    }

    public void Rejected() => Interlocked.Increment(ref _rejected);

    public void Failed() => Interlocked.Increment(ref _failed);

    public EngineStats Snapshot()
    {
        lock (_lock)
        {
            int n = (int)Math.Min(_count, _latencies.Length);
            var latencies = _latencies.Take(n).Order().ToArray();
            double average = n == 0 ? 0 : latencies.Average();
            double p95 = n == 0 ? 0 : latencies[Math.Min(n - 1, (int)Math.Ceiling(0.95 * n) - 1)];
            double wait = n == 0 ? 0 : _waits.Take(n).Average();
            return new EngineStats(_requests, Interlocked.Read(ref _rejected), Interlocked.Read(ref _failed),
                TimeSpan.FromMilliseconds(average), TimeSpan.FromMilliseconds(p95), TimeSpan.FromMilliseconds(wait),
                _rows, _modelSeconds > 0 ? _rows / _modelSeconds : 0, _calls == 0 ? 0 : (double)_rows / _calls);
        }
    }
}

/// <summary>Loading, copies, queueing, keep-alive and statistics shared by every kind of model.</summary>
internal abstract class EngineModel : IAsyncDisposable
{
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private readonly Lock _stateLock = new();
    private readonly int _instances;
    private readonly bool _shared;
    private readonly bool _reloadable;
    private TimeSpan? _keepAlive;
    private bool _keepAliveSet;
    private readonly int? _queueLimit;
    private readonly TimeSpan? _timeout;
    private Channel<object>? _free;
    private List<object> _loaded = [];
    private int _roundRobin;
    private int _running;
    private int _queued;
    private ITimer? _expiry;
    private DateTimeOffset? _expiresAt;
    private bool _disposed;

    protected EngineModel(string name, EngineModelKind kind, EngineOptions options, int instances, bool shared,
        TimeSpan? keepAlive, bool keepAliveSet, int? queueLimit, TimeSpan? timeout, bool reloadable)
    {
        _reloadable = reloadable;
        Name = name;
        Kind = kind;
        Options = options;
        _instances = instances;
        _shared = shared;
        _keepAlive = keepAlive;
        _keepAliveSet = keepAliveSet;
        _queueLimit = queueLimit;
        _timeout = timeout;
    }

    public string Name { get; }

    public EngineModelKind Kind { get; }

    public EngineOptions Options { get; }

    public EngineStatistics Statistics { get; } = new();

    /// <summary>Set when the first copy loads.</summary>
    public ModelDescription? Description { get; protected set; }

    protected abstract object CreateInstance();

    protected abstract void DisposeInstance(object instance);

    public void SetKeepAlive(TimeSpan? idle)
    {
        if (!_reloadable)
        {
            throw new InvalidOperationException($"'{Name}' uses a model object you created, which the engine cannot reload; its keep-alive cannot be set.");
        }

        lock (_stateLock)
        {
            _keepAlive = idle;
            _keepAliveSet = true;
        }

        ScheduleExpiry();
    }

    public ModelStatus Status()
    {
        lock (_stateLock)
        {
            return new ModelStatus(Name, Kind, _loaded.Count > 0, _instances, _running, _queued, _running == 0 && _queued == 0 ? _expiresAt : null);
        }
    }

    public async Task EnsureLoadedAsync(CancellationToken cancellationToken)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (Volatile.Read(ref _loaded).Count > 0)
        {
            return;
        }

        await _loadLock.WaitAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            if (_loaded.Count > 0)
            {
                return;
            }

            var clock = Stopwatch.StartNew();
            var created = new List<object>(_instances);
            try
            {
                for (int i = 0; i < _instances; i++)
                {
                    created.Add(await Task.Run(CreateInstance, cancellationToken).ConfigureAwait(false));
                }
            }
            catch
            {
                created.ForEach(DisposeInstance);
                throw;
            }

            var free = Channel.CreateUnbounded<object>();
            foreach (var instance in created)
            {
                free.Writer.TryWrite(instance);
            }

            lock (_stateLock)
            {
                _free = free;
                Volatile.Write(ref _loaded, created);
            }

            Publish(EngineEventKind.ModelLoaded, clock.Elapsed, TimeSpan.Zero, 0, null);
        }
        finally
        {
            _loadLock.Release();
        }
    }

    /// <summary>Waits for the model (loading it if needed) and a copy; the lease must be disposed.</summary>
    protected async Task<Lease> AcquireAsync(CancellationToken cancellationToken)
    {
        lock (_stateLock)
        {
            if (_queueLimit is { } limit && _queued >= limit)
            {
                Statistics.Rejected();
                Publish(EngineEventKind.RequestRejected, TimeSpan.Zero, TimeSpan.Zero, 0, "queue full");
                throw new InferenceQueueFullException($"'{Name}' already has {limit} requests waiting.");
            }

            _queued++;
            _expiry?.Dispose();
            _expiry = null;
            _expiresAt = null;
        }

        var start = Stopwatch.GetTimestamp();
        var timeout = _timeout is { } t ? new CancellationTokenSource(t, Options.Time) : null;
        var linked = timeout is null ? null : CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeout.Token);
        var token = linked?.Token ?? cancellationToken;
        bool counted = true;
        try
        {
            await EnsureLoadedAsync(token).ConfigureAwait(false);
            object instance;
            if (_shared)
            {
                var loaded = Volatile.Read(ref _loaded);
                instance = loaded[(int)((uint)Interlocked.Increment(ref _roundRobin) % (uint)loaded.Count)];
            }
            else
            {
                instance = await _free!.Reader.ReadAsync(token).ConfigureAwait(false);
            }

            lock (_stateLock)
            {
                _queued--;
                _running++;
                counted = false;
            }

            return new Lease(this, instance, start, Stopwatch.GetElapsedTime(start), token, timeout, linked);
        }
        catch (Exception ex)
        {
            timeout?.Dispose();
            linked?.Dispose();
            if (counted)
            {
                lock (_stateLock)
                {
                    _queued--;
                }
            }

            Failed(ex, timeout);
            ScheduleExpiry();
            throw Translate(ex, timeout);
        }
    }

    internal void Failed(Exception ex, CancellationTokenSource? timeout)
    {
        Statistics.Failed();
        Publish(EngineEventKind.RequestRejected, TimeSpan.Zero, TimeSpan.Zero, 0, timeout?.IsCancellationRequested == true ? "timeout" : ex.Message);
    }

    internal Exception Translate(Exception ex, CancellationTokenSource? timeout) =>
        ex is OperationCanceledException && timeout?.IsCancellationRequested == true
            ? new TimeoutException($"'{Name}' did not answer within {_timeout!.Value.TotalSeconds:0.###} s.", ex)
            : ex;

    private void Release(object instance)
    {
        lock (_stateLock)
        {
            _running--;
        }

        if (!_shared)
        {
            _free?.Writer.TryWrite(instance);
        }

        ScheduleExpiry();
    }

    private void ScheduleExpiry()
    {
        if (!_keepAliveSet || _keepAlive is not { } idle)
        {
            return;
        }

        lock (_stateLock)
        {
            if (_running > 0 || _queued > 0 || _loaded.Count == 0)
            {
                return;
            }

            if (idle <= TimeSpan.Zero)
            {
                _ = UnloadAsync(force: false);
                return;
            }

            _expiry?.Dispose();
            _expiresAt = Options.Time.GetUtcNow() + idle;
            _expiry = Options.Time.CreateTimer(_ => _ = UnloadAsync(force: false), null, idle, System.Threading.Timeout.InfiniteTimeSpan);
        }
    }

    public async Task UnloadAsync(bool force)
    {
        await _loadLock.WaitAsync().ConfigureAwait(false);
        try
        {
            List<object> instances;
            lock (_stateLock)
            {
                if (_loaded.Count == 0 || (!force && (_running > 0 || _queued > 0)))
                {
                    return;
                }

                _expiry?.Dispose();
                _expiry = null;
                _expiresAt = null;
                instances = _loaded;
                Volatile.Write(ref _loaded, []);
            }

            // Wait for running requests to give their copies back before disposing them.
            while (true)
            {
                lock (_stateLock)
                {
                    if (_running == 0)
                    {
                        break;
                    }
                }

                await Task.Delay(1).ConfigureAwait(false);
            }

            _free?.Writer.TryComplete();
            _free = null;
            instances.ForEach(DisposeInstance);
            Publish(EngineEventKind.ModelUnloaded, TimeSpan.Zero, TimeSpan.Zero, 0, null);
        }
        finally
        {
            _loadLock.Release();
        }
    }

    protected void Publish(EngineEventKind kind, TimeSpan duration, TimeSpan wait, int batch, string? reason)
    {
        if (Options.PublishTelemetry && Idrak.Abstraction.Diagnostics.Telemetry.IsEnabled(TelemetryLevel.Engine))
        {
            Idrak.Abstraction.Diagnostics.Telemetry.Engine(new EngineEvent(kind, Name, duration, wait, batch, reason));
        }
    }

    public virtual async ValueTask DisposeAsync()
    {
        if (_disposed)
        {
            return;
        }

        _disposed = true;
        await UnloadAsync(force: true).ConfigureAwait(false);
    }

    /// <summary>A copy of the model in use by one request.</summary>
    internal sealed class Lease(EngineModel owner, object instance, long start, TimeSpan queueWait, CancellationToken token,
        CancellationTokenSource? timeout, CancellationTokenSource? linked) : IDisposable
    {
        private int _released;

        public object Instance { get; } = instance;

        public CancellationToken Token { get; } = token;

        public TimeSpan QueueWait { get; } = queueWait;

        public void Completed(int batch)
        {
            var latency = Stopwatch.GetElapsedTime(start);
            owner.Statistics.Completed(latency, QueueWait);
            owner.Publish(EngineEventKind.RequestCompleted, latency, QueueWait, batch, null);
        }

        public Exception Fail(Exception ex)
        {
            owner.Failed(ex, timeout);
            return owner.Translate(ex, timeout);
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
            {
                return;
            }

            timeout?.Dispose();
            linked?.Dispose();
            owner.Release(Instance);
        }
    }
}

/// <summary>A predictor in the engine, with optional micro-batching.</summary>
internal sealed class PredictorModel<TIn, TOut> : EngineModel, IPredictor<TIn, TOut>
{
    private readonly PredictorBuilder<TIn, TOut> _builder;
    private readonly Func<Module> _load;
    private readonly bool _ownsModel;
    private readonly (int MaxBatch, TimeSpan MaxWait)? _batching;
    private readonly Channel<Pending>? _pending;
    private readonly Task? _batcher;
    private readonly CancellationTokenSource _stop = new();

    public PredictorModel(string name, EngineOptions options, PredictorBuilder<TIn, TOut> builder, Func<Module> load, bool ownsModel, bool reloadable)
        : base(name, EngineModelKind.Predictor, options, builder.Settings.Instances ?? 1, shared: true,
            builder.Settings.KeepAlive, builder.Settings.KeepAliveSet, builder.Settings.QueueLimit, builder.Settings.Timeout, reloadable)
    {
        _builder = builder;
        _load = load;
        _ownsModel = ownsModel;
        _batching = builder.Settings.Batching;
        if (_batching is not null)
        {
            _pending = Channel.CreateUnbounded<Pending>(new UnboundedChannelOptions { SingleReader = true });
            _batcher = Task.Run(BatchLoopAsync);
        }
    }

    protected override object CreateInstance()
    {
        var predictor = _builder.Create(_load(), _ownsModel);
        Description ??= new ModelDescription(Name, Kind, predictor.Model.ParameterCount, predictor.Device, null);
        return predictor;
    }

    protected override void DisposeInstance(object instance) => ((Predictor<TIn, TOut>)instance).Dispose();

    public async ValueTask<TOut> PredictAsync(TIn input, CancellationToken cancellationToken = default)
    {
        if (_pending is null)
        {
            return (await PredictBatchAsync([input], cancellationToken).ConfigureAwait(false))[0];
        }

        var pending = new Pending(input, new TaskCompletionSource<TOut>(TaskCreationOptions.RunContinuationsAsynchronously), cancellationToken);
        _pending.Writer.TryWrite(pending);
        return await pending.Result.Task.ConfigureAwait(false);
    }

    public async ValueTask<IReadOnlyList<TOut>> PredictAsync(IReadOnlyList<TIn> inputs, CancellationToken cancellationToken = default) =>
        await PredictBatchAsync(inputs, cancellationToken).ConfigureAwait(false);

    public async Task<IReadOnlyList<TOut>> PredictBatchAsync(IReadOnlyList<TIn> inputs, CancellationToken cancellationToken)
    {
        using var lease = await AcquireAsync(cancellationToken).ConfigureAwait(false);
        try
        {
            var predictor = (Predictor<TIn, TOut>)lease.Instance;
            var clock = Stopwatch.StartNew();
            var results = await Task.Run(() => predictor.Predict(inputs), lease.Token).ConfigureAwait(false);
            Statistics.ModelCall(inputs.Count, clock.Elapsed);
            lease.Completed(inputs.Count);
            return results;
        }
        catch (Exception ex)
        {
            throw lease.Fail(ex);
        }
    }

    // Collects requests until maxBatch inputs are waiting or maxWait has passed since the first, then predicts them together.
    private async Task BatchLoopAsync()
    {
        var (maxBatch, maxWait) = _batching!.Value;
        var reader = _pending!.Reader;
        try
        {
            while (await reader.WaitToReadAsync(_stop.Token).ConfigureAwait(false))
            {
                var batch = new List<Pending>(maxBatch);
                if (!reader.TryRead(out var first))
                {
                    continue;
                }

                batch.Add(first);
                using (var window = new CancellationTokenSource(maxWait, Options.Time))
                using (var linked = CancellationTokenSource.CreateLinkedTokenSource(window.Token, _stop.Token))
                {
                    while (batch.Count < maxBatch)
                    {
                        if (reader.TryRead(out var next))
                        {
                            batch.Add(next);
                            continue;
                        }

                        try
                        {
                            if (!await reader.WaitToReadAsync(linked.Token).ConfigureAwait(false))
                            {
                                break;
                            }
                        }
                        catch (OperationCanceledException)
                        {
                            break;
                        }
                    }
                }

                batch.RemoveAll(p => p.Cancellation.IsCancellationRequested && p.Result.TrySetCanceled(p.Cancellation));
                if (batch.Count == 0)
                {
                    continue;
                }

                try
                {
                    var results = await PredictBatchAsync([.. batch.Select(p => p.Input)], CancellationToken.None).ConfigureAwait(false);
                    for (int i = 0; i < batch.Count; i++)
                    {
                        batch[i].Result.TrySetResult(results[i]);
                    }
                }
                catch (Exception ex)
                {
                    batch.ForEach(p => p.Result.TrySetException(ex));
                }
            }
        }
        catch (OperationCanceledException) when (_stop.IsCancellationRequested)
        {
        }

        while (reader.TryRead(out var left))
        {
            left.Result.TrySetCanceled();
        }
    }

    private int _stopped;

    public override async ValueTask DisposeAsync()
    {
        if (Interlocked.Exchange(ref _stopped, 1) != 0)
        {
            return;
        }

        _stop.Cancel();
        _pending?.Writer.TryComplete();
        if (_batcher is not null)
        {
            await _batcher.ConfigureAwait(false);
        }

        await base.DisposeAsync().ConfigureAwait(false);
        _stop.Dispose();
    }

    private sealed record Pending(TIn Input, TaskCompletionSource<TOut> Result, CancellationToken Cancellation);
}

/// <summary>A text or chat model in the engine: each copy runs one generation at a time.</summary>
internal sealed class GenerativeModel : EngineModel
{
    private readonly GenerativeModelBuilder _settings;
    private readonly Func<Device?, TextGenerator> _load;
    private readonly bool _owns;

    public GenerativeModel(string name, EngineModelKind kind, EngineOptions options, GenerativeModelBuilder settings, Func<Device?, TextGenerator> load,
        bool owns, bool reloadable)
        : base(name, kind, options, settings.InstanceCount ?? 1, shared: false, settings.KeepAliveTime, settings.KeepAliveSet,
            settings.QueueLimitCount, settings.TimeoutLimit, reloadable)
    {
        _settings = settings;
        _load = load;
        _owns = owns;
    }

    public ToolRegistry? Tools => _settings.ToolRegistry;

    public int ContextLength { get; private set; }

    protected override object CreateInstance()
    {
        var generator = _load(_settings.TargetDevice);
        ContextLength = generator.ContextLength;
        Description ??= new ModelDescription(Name, Kind, generator.Model.ParameterCount, generator.Device, generator.ContextLength);
        if (_settings.WarmUpPrompt is { } prompt)
        {
            generator.Generate(prompt, new GenerationOptions { NumPredict = 1 });
        }

        return Kind == EngineModelKind.Chat ? new ChatGenerator(generator, _settings.ChatTemplate) : generator;
    }

    protected override void DisposeInstance(object instance)
    {
        if (_owns)
        {
            var generator = instance is ChatGenerator chat ? chat.Generator : (TextGenerator)instance;
            generator.Model.Dispose();
        }
    }

    public async IAsyncEnumerable<GenerationChunk> StreamTextAsync(string prompt, GenerationOptions options,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        using var lease = await AcquireAsync(cancellationToken).ConfigureAwait(false);
        var generator = lease.Instance is ChatGenerator chat ? chat.Generator : (TextGenerator)lease.Instance;
        await foreach (var chunk in Guard(generator.StreamAsync(prompt, options, lease.Token), lease).ConfigureAwait(false))
        {
            if (chunk.Done)
            {
                Statistics.ModelCall(chunk.Stats!.GeneratedTokens, chunk.Stats.GenerationDuration);
                lease.Completed(1);
            }

            yield return chunk;
        }
    }

    public async IAsyncEnumerable<ChatChunk> StreamChatAsync(ChatRequest request,
        [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        if (Kind != EngineModelKind.Chat)
        {
            throw new InvalidOperationException($"'{Name}' is a text model; chat requests need a chat model.");
        }

        using var lease = await AcquireAsync(cancellationToken).ConfigureAwait(false);
        var chat = (ChatGenerator)lease.Instance;
        await foreach (var chunk in Guard(chat.StreamAsync(request, lease.Token), lease).ConfigureAwait(false))
        {
            if (chunk.Done)
            {
                Statistics.ModelCall(chunk.Stats!.GeneratedTokens, chunk.Stats.GenerationDuration);
                lease.Completed(1);
            }

            yield return chunk;
        }
    }

    public IChatModel AsChatModel() => Kind == EngineModelKind.Chat
        ? new EngineChat(this)
        : throw new InvalidOperationException($"'{Name}' is a text model; conversations need a chat model.");

    // Translates a timeout into TimeoutException and counts failures, for streams consumed chunk by chunk.
    private static async IAsyncEnumerable<T> Guard<T>(IAsyncEnumerable<T> source, Lease lease)
    {
        var enumerator = source.GetAsyncEnumerator();
        try
        {
            while (true)
            {
                T item;
                try
                {
                    if (!await enumerator.MoveNextAsync().ConfigureAwait(false))
                    {
                        yield break;
                    }

                    item = enumerator.Current;
                }
                catch (Exception ex)
                {
                    throw lease.Fail(ex);
                }

                yield return item;
            }
        }
        finally
        {
            await enumerator.DisposeAsync().ConfigureAwait(false);
        }
    }

    private sealed class EngineChat(GenerativeModel model) : IChatModel
    {
        public IAsyncEnumerable<ChatChunk> StreamAsync(ChatRequest request, CancellationToken cancellationToken = default) =>
            model.StreamChatAsync(request, cancellationToken);
    }
}
