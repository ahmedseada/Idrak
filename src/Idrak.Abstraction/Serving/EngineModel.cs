// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Serving;

/// <summary>
/// How the inference engine hosts the copies of one model. The engine does the work (loading, queueing, timeouts,
/// keep-alive); the model kind only says what it wants.
/// </summary>
public sealed record EngineHosting
{
    /// <summary>How many copies are loaded (default 1).</summary>
    public int Instances { get; init; } = 1;

    /// <summary>
    /// True when a copy may serve several requests at once (it is thread-safe, as a predictor is): requests take the copies
    /// in turn and never wait for one. False (the default): a copy serves one request at a time (a text generator owns its
    /// KV cache), and requests wait for a free copy.
    /// </summary>
    public bool SharedCopies { get; init; }

    /// <summary>
    /// False when the copy is an object the application created, which the engine can neither copy nor load again: it is
    /// loaded once and never unloaded or disposed, so <see cref="Instances"/> must be 1 and <see cref="KeepAlive"/> null.
    /// Default true.
    /// </summary>
    public bool Reloadable { get; init; } = true;

    /// <summary>Unload the copies after the model has been idle this long (null keeps them; <see cref="TimeSpan.Zero"/> unloads after each request).</summary>
    public TimeSpan? KeepAlive { get; init; }

    /// <summary>Refuse new requests while this many are already waiting (null: no limit).</summary>
    public int? QueueLimit { get; init; }

    /// <summary>Stop a request that has not finished after this long, queue wait included (null: no limit).</summary>
    public TimeSpan? Timeout { get; init; }
}

/// <summary>What a loaded model is (<see cref="EngineModel{TCopy}.Describe"/>).</summary>
/// <param name="Name">The model's name in the engine.</param>
/// <param name="Kind">The model's kind (<see cref="EngineModel{TCopy}.Kind"/>: "predictor", "text", "chat", ...).</param>
/// <param name="Parameters">Trainable values in one copy.</param>
/// <param name="Device">Where the copies run.</param>
/// <param name="ContextLength">The context length (text and chat models), or null.</param>
public sealed record ModelDescription(string Name, string Kind, long Parameters, Device Device, int? ContextLength);

/// <summary>
/// A kind of model the inference engine can host: predictors, text and chat models, and whatever a domain package adds
/// (image generation, transcription, speech, ...). The engine owns the machinery (loading at start or on first use,
/// several copies, queue limits, timeouts, keep-alive unloading, micro-batching, statistics and telemetry); a kind says
/// how to load, describe and unload one copy (<typeparamref name="TCopy"/>) and answers its own requests with the
/// copies the engine lends it through <see cref="Host"/>. A kind exposes its requests by implementing the contract
/// callers ask the engine for (<see cref="IPredictor{TIn, TOut}"/>, <see cref="Generation.IChatModel"/>,
/// <see cref="Generation.ITextModel"/>, ...): <c>engine.Model&lt;IChatModel&gt;("my-gpt")</c>.
/// </summary>
/// <typeparam name="TCopy">One loaded copy of the model (for example a text generator).</typeparam>
/// <example>
/// <code>
/// sealed class EchoModel() : EngineModel&lt;Echo&gt;("echo", new EngineHosting { Instances = 2 })
/// {
///     public override Echo LoadCopy() => new Echo();
///     public override void UnloadCopy(Echo copy) { }
///     public override ModelDescription Describe(Echo copy) => new(Name, Kind, 0, Device.Cpu, null);
///
///     public async Task&lt;string&gt; EchoAsync(string text, CancellationToken token = default)
///     {
///         using var lease = await Host.AcquireAsync(token);
///         lease.Completed(batch: 1, units: text.Length, TimeSpan.Zero);
///         return lease.Copy.Say(text);
///     }
/// }
///
/// var engine = await InferenceEngine.Create().Add("echo", new EchoModel()).BuildAsync();
/// string said = await engine.Model&lt;EchoModel&gt;("echo").EchoAsync("hi");
/// </code>
/// </example>
public abstract class EngineModel<TCopy> : IAsyncDisposable
    where TCopy : class
{
    private IEngineHost<TCopy>? _host;

    /// <summary>A model of <paramref name="kind"/>, hosted as <paramref name="hosting"/> says.</summary>
    /// <param name="kind">The kind's name, shown in the engine's status (for example "chat").</param>
    /// <param name="hosting">How many copies, whether they are shared, keep-alive, queue limit and timeout.</param>
    protected EngineModel(string kind, EngineHosting hosting)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(kind);
        ArgumentNullException.ThrowIfNull(hosting);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hosting.Instances);
        Kind = kind;
        Hosting = hosting;
    }

    /// <summary>The kind's name ("predictor", "text", "chat", ...), as the engine's status reports it.</summary>
    public string Kind { get; }

    /// <summary>How the engine hosts the copies.</summary>
    public EngineHosting Hosting { get; }

    /// <summary>The model's name in the engine (once added to one).</summary>
    public string Name => Host.Name;

    /// <summary>The engine hosting this model: it lends copies to requests and batches them.</summary>
    /// <exception cref="InvalidOperationException">The model has not been added to an engine yet.</exception>
    protected IEngineHost<TCopy> Host =>
        _host ?? throw new InvalidOperationException($"This {Kind} model has not been added to an inference engine yet.");

    /// <summary>Connects the model to the engine that hosts it; the engine calls this once, when it is built.</summary>
    /// <exception cref="InvalidOperationException">The model already belongs to an engine.</exception>
    public void Attach(IEngineHost<TCopy> host)
    {
        ArgumentNullException.ThrowIfNull(host);
        if (Interlocked.CompareExchange(ref _host, host, null) is not null)
        {
            throw new InvalidOperationException($"The {Kind} model '{_host.Name}' already belongs to an inference engine; add a new model object to each engine.");
        }

        OnAttached();
    }

    /// <summary>Called once the model has its <see cref="Host"/>, before any request (for example to create a batcher).</summary>
    protected virtual void OnAttached()
    {
    }

    /// <summary>Loads one copy (the engine calls this on a thread-pool thread, once per copy, and again after an unload).</summary>
    public abstract TCopy LoadCopy();

    /// <summary>Frees one copy the engine unloads (a no-op for objects the application owns).</summary>
    public abstract void UnloadCopy(TCopy copy);

    /// <summary>Describes the model from its first loaded copy.</summary>
    public abstract ModelDescription Describe(TCopy copy);

    /// <summary>
    /// Stops what the model runs besides its copies; the engine calls this when it is disposed, before it unloads the
    /// copies. The default does nothing.
    /// </summary>
    public virtual ValueTask DisposeAsync()
    {
        GC.SuppressFinalize(this);
        return ValueTask.CompletedTask;
    }
}

/// <summary>What the inference engine gives a model it hosts (<see cref="EngineModel{TCopy}.Host"/>). Implemented by the engine.</summary>
/// <typeparam name="TCopy">One loaded copy of the model.</typeparam>
public interface IEngineHost<TCopy>
    where TCopy : class
{
    /// <summary>The model's name in the engine.</summary>
    string Name { get; }

    /// <summary>
    /// Waits for the model (loading it if needed) and a copy, within the model's queue limit and timeout, and lends the
    /// copy to one request (or one batch) until the lease is disposed.
    /// </summary>
    /// <exception cref="TimeoutException">The model's timeout passed while waiting.</exception>
    /// <exception cref="InvalidOperationException">The queue is full (the engine's queue-full exception derives from it).</exception>
    ValueTask<IEngineLease<TCopy>> AcquireAsync(CancellationToken cancellationToken);

    /// <summary>
    /// Micro-batching: requests submitted to the returned batcher are collected until <paramref name="maxBatch"/> are
    /// waiting or <paramref name="maxWait"/> has passed since the first, then answered together by <paramref name="run"/>
    /// (which acquires a copy for the batch). The engine stops the batcher when it is disposed.
    /// </summary>
    IEngineBatcher<TIn, TOut> Batcher<TIn, TOut>(int maxBatch, TimeSpan maxWait, Func<IReadOnlyList<TIn>, CancellationToken, Task<IReadOnlyList<TOut>>> run);
}

/// <summary>A copy of a model lent to one request (or one batch); dispose it to give the copy back.</summary>
/// <typeparam name="TCopy">One loaded copy of the model.</typeparam>
public interface IEngineLease<out TCopy> : IDisposable
{
    /// <summary>The copy.</summary>
    TCopy Copy { get; }

    /// <summary>The request's cancellation, with the model's timeout: pass it to the work done on the copy.</summary>
    CancellationToken Token { get; }

    /// <summary>How long the request waited for the model and a copy.</summary>
    TimeSpan QueueWait { get; }

    /// <summary>Records the request as completed, for the engine's statistics and telemetry.</summary>
    /// <param name="batch">Inputs answered by this call (1 for one generation).</param>
    /// <param name="units">Work done, for the throughput statistics: rows predicted, tokens generated, ...</param>
    /// <param name="modelTime">Time the copy spent on it.</param>
    void Completed(int batch, long units, TimeSpan modelTime);

    /// <summary>Records the request as failed and returns the exception to throw (a <see cref="TimeoutException"/> when the model's timeout cancelled it).</summary>
    Exception Fail(Exception exception);

    /// <summary>A stream produced with the copy, with each exception it throws passed through <see cref="Fail"/>.</summary>
    IAsyncEnumerable<T> Guard<T>(IAsyncEnumerable<T> source);
}

/// <summary>Collects single requests into batches (<see cref="IEngineHost{TCopy}.Batcher"/>).</summary>
/// <typeparam name="TIn">One request's input.</typeparam>
/// <typeparam name="TOut">One request's answer.</typeparam>
public interface IEngineBatcher<in TIn, TOut>
{
    /// <summary>Adds <paramref name="input"/> to the next batch and returns its answer.</summary>
    Task<TOut> SubmitAsync(TIn input, CancellationToken cancellationToken = default);
}
