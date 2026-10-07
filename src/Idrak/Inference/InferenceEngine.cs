// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using Idrak.Layers;

namespace Idrak.Inference;

/// <summary>Thrown when a model's <c>QueueLimit</c> is reached.</summary>
public sealed class InferenceQueueFullException(string message) : InvalidOperationException(message);

/// <summary>The state of one model in the engine.</summary>
/// <param name="Name">The model's name.</param>
/// <param name="Kind">The model's kind (<see cref="EngineModel{TCopy}.Kind"/>: "predictor", "text", "chat", ...).</param>
/// <param name="Loaded">Whether its copies are in memory.</param>
/// <param name="Instances">How many copies are loaded when it is loaded.</param>
/// <param name="Running">Requests being processed now.</param>
/// <param name="Queued">Requests waiting (for a batch, a free copy, or the load).</param>
/// <param name="ExpiresAt">When the keep-alive will unload it, if it is idle and a keep-alive is set.</param>
public sealed record ModelStatus(string Name, string Kind, bool Loaded, int Instances, int Running, int Queued, DateTimeOffset? ExpiresAt);

/// <summary>Request statistics of one model since the engine started.</summary>
/// <param name="Requests">Completed requests.</param>
/// <param name="Rejected">Requests refused because the queue was full.</param>
/// <param name="Failed">Requests that timed out, were cancelled or threw.</param>
/// <param name="AverageLatency">Mean end-to-end time of the last 1,024 completed requests (queue wait included).</param>
/// <param name="P95Latency">95th percentile of the same.</param>
/// <param name="AverageQueueWait">Mean time the last 1,024 requests waited before running.</param>
/// <param name="Rows">Rows predicted (predictors), tokens generated (text and chat), or the units another kind counts.</param>
/// <param name="RowsPerSecond">Rows (or tokens) per second of model time.</param>
/// <param name="AverageBatchSize">Rows per model call (predictors; 1 for generation).</param>
public sealed record EngineStats(long Requests, long Rejected, long Failed, TimeSpan AverageLatency, TimeSpan P95Latency, TimeSpan AverageQueueWait,
    long Rows, double RowsPerSecond, double AverageBatchSize);

/// <summary>
/// Hosts named models of any kind and serves requests to them from any kind of .NET application: loading (at start or
/// on first use), warm-up, several copies, micro-batching, queue limits, timeouts, keep-alive unloading, statistics and
/// telemetry — each off unless set when the model is added. A kind (<see cref="EngineModel{TCopy}"/>) says how to load
/// one copy and answers its requests with the copies the engine lends it; callers reach a model through the contract
/// its kind implements (<see cref="Model{T}"/>): predictors (<see cref="IPredictor{TIn, TOut}"/>, built in), text and
/// chat models (<see cref="ITextModel"/>, <see cref="IChatModel"/>; <see cref="GenerativeModels"/>), and the kinds of
/// domain packages.
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
/// var reply = await engine.Model&lt;IChatModel&gt;("my-gpt").ChatAsync(request);
/// </code>
/// </example>
public sealed class InferenceEngine : IModelCatalog, IAsyncDisposable
{
    private readonly Dictionary<string, HostedModel> _models;

    internal InferenceEngine(Dictionary<string, HostedModel> models) => _models = models;

    /// <summary>Starts an engine configuration.</summary>
    public static InferenceEngineBuilder Create() => new();

    /// <summary>The state of every model.</summary>
    public IReadOnlyList<ModelStatus> Models => [.. _models.Values.Select(m => m.Status())];

    /// <summary>The names of the models.</summary>
    public IReadOnlyCollection<string> Names => _models.Keys;

    /// <summary>The kind of <paramref name="name"/> ("predictor", "text", "chat", ...).</summary>
    public string KindOf(string name) => Hosted(name).Kind;

    /// <summary>Request statistics of <paramref name="name"/>.</summary>
    public EngineStats Stats(string name) => Hosted(name).Statistics.Snapshot();

    /// <summary>Loads <paramref name="name"/> now (if it is not loaded).</summary>
    public Task LoadAsync(string name, CancellationToken cancellationToken = default) => Hosted(name).EnsureLoadedAsync(cancellationToken);

    /// <summary>
    /// Changes how long <paramref name="name"/> stays loaded after its last request (null: forever; <see cref="TimeSpan.Zero"/>:
    /// unload after each request), for example from a request's <c>keep_alive</c>. Needs a model the engine can reload.
    /// </summary>
    public void KeepAlive(string name, TimeSpan? idle) => Hosted(name).SetKeepAlive(idle);

    /// <summary>Unloads <paramref name="name"/> now, freeing its memory, once running requests finish; the next request loads it again.</summary>
    public Task UnloadAsync(string name) => Hosted(name).UnloadAsync(force: true);

    /// <summary>Describes the model named <paramref name="name"/> (loads it if needed).</summary>
    public async Task<ModelDescription> DescribeAsync(string name, CancellationToken cancellationToken = default)
    {
        var model = Hosted(name);
        await model.EnsureLoadedAsync(cancellationToken).ConfigureAwait(false);
        return model.Description ?? throw new InvalidOperationException($"'{name}' was unloaded while it was being described.");
    }

    // ------------------------------------------------------------------ models of any kind

    /// <summary>
    /// The model named <paramref name="name"/> as <typeparamref name="T"/>: a contract its kind implements, such as
    /// <see cref="IPredictor{TIn, TOut}"/>, <see cref="IChatModel"/> or <see cref="ITextModel"/> (or the kind's own class).
    /// Its requests go through the engine (copies, queueing, timeouts, statistics).
    /// </summary>
    /// <exception cref="KeyNotFoundException">The engine has no such model.</exception>
    /// <exception cref="InvalidOperationException">The model's kind does not implement <typeparamref name="T"/>.</exception>
    public T Model<T>(string name)
        where T : class
    {
        var hosted = Hosted(name);
        return hosted.Model as T
            ?? throw new InvalidOperationException($"'{name}' is a {hosted.Kind} model; this call needs a model that is {Display(typeof(T))}.");
    }

    /// <summary>The model named <paramref name="name"/> as <typeparamref name="T"/>, if the engine has it and its kind implements <typeparamref name="T"/>.</summary>
    public bool TryGetModel<T>(string name, [NotNullWhen(true)] out T? model)
        where T : class
    {
        model = _models.TryGetValue(name, out var hosted) ? hosted.Model as T : null;
        return model is not null;
    }

    // ------------------------------------------------------------------ predictors

    /// <summary>A typed handle for the predictor named <paramref name="name"/>.</summary>
    public IPredictor<TIn, TOut> Predictor<TIn, TOut>(string name) => Model<IPredictor<TIn, TOut>>(name);

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
        var model = Predictor<TIn, TOut>(name);
        foreach (var chunk in inputs.Chunk(batchSize))
        {
            foreach (var result in await model.PredictAsync(chunk, cancellationToken).ConfigureAwait(false))
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
        var model = Predictor<TIn, TOut>(name);
        var chunk = new List<TIn>(batchSize);
        await foreach (var input in inputs.WithCancellation(cancellationToken).ConfigureAwait(false))
        {
            chunk.Add(input);
            if (chunk.Count == batchSize)
            {
                foreach (var result in await model.PredictAsync([.. chunk], cancellationToken).ConfigureAwait(false))
                {
                    yield return result;
                }

                chunk.Clear();
            }
        }

        if (chunk.Count > 0)
        {
            foreach (var result in await model.PredictAsync([.. chunk], cancellationToken).ConfigureAwait(false))
            {
                yield return result;
            }
        }
    }

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

    private HostedModel Hosted(string name) =>
        _models.TryGetValue(name, out var model) ? model : throw new KeyNotFoundException($"The engine has no model named '{name}'. Models: {string.Join(", ", _models.Keys)}.");

    // A type as C# writes it: IPredictor<House, Single>.
    private static string Display(Type type) => !type.IsGenericType ? type.Name
        : $"{type.Name[..type.Name.IndexOf('`', StringComparison.Ordinal)]}<{string.Join(", ", type.GetGenericArguments().Select(Display))}>";
}

/// <summary>
/// Configures an <see cref="InferenceEngine"/>. Every model has a name and a kind (<see cref="Add{TCopy}"/>; predictors
/// with <c>Predictor</c>, text and chat models with <see cref="GenerativeModels"/>); everything else is set per model
/// (see <see cref="PredictorBuilder{TIn, TOut}"/> and <see cref="EngineHosting"/>) and is off unless set.
/// </summary>
public sealed class InferenceEngineBuilder
{
    private readonly List<Func<EngineOptions, HostedModel>> _models = [];
    private readonly HashSet<string> _names = [];
    private readonly EngineOptions _options = new();

    internal InferenceEngineBuilder()
    {
    }

    // ------------------------------------------------------------------ models of any kind

    /// <summary>
    /// Hosts <paramref name="model"/> (a model of any kind: a domain package's, or one of your own) as
    /// <paramref name="name"/>. A model object belongs to one engine.
    /// </summary>
    public InferenceEngineBuilder Add<TCopy>(string name, EngineModel<TCopy> model)
        where TCopy : class
    {
        ArgumentNullException.ThrowIfNull(model);
        CheckName(name);
        HostedModel.Check(name, model.Hosting);
        _models.Add(options => HostedModel.Create(name, model, options));
        return this;
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

    /// <summary>The clock for keep-alive timers and batching windows (replaceable in tests).</summary>
    public InferenceEngineBuilder TimeProvider(TimeProvider time)
    {
        _options.Time = time;
        return this;
    }

    /// <summary>Creates the engine and, unless <see cref="LoadOnFirstUse"/> was set, loads every model (warming up those with a warm-up).</summary>
    public async Task<InferenceEngine> BuildAsync(CancellationToken cancellationToken = default)
    {
        var models = new Dictionary<string, HostedModel>(_models.Count);
        try
        {
            foreach (var create in _models)
            {
                var model = create(_options);
                models.Add(model.Name, model);
            }
        }
        catch
        {
            foreach (var model in models.Values)
            {
                await model.DisposeAsync().ConfigureAwait(false);       // stops the batchers of those already hosted
            }

            throw;
        }

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
        if (!reloadable && builder.Settings.KeepAliveSet)
        {
            throw HostedModel.NotReloadable(name);
        }

        return Add(name, new PredictorModel<TIn, TOut>(builder, load, ownsModel, reloadable));
    }

    private void CheckName(string name)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        if (!_names.Add(name))
        {
            throw new ArgumentException($"A model named '{name}' was already added.", nameof(name));
        }
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

    public void ModelCall(long rows, TimeSpan duration)
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
