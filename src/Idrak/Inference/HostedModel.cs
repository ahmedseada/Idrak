// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics;
using System.Threading.Channels;

namespace Idrak.Inference;

/// <summary>
/// The engine's side of one model of any kind (an <see cref="EngineModel{TCopy}"/>): loading, copies, queueing,
/// timeouts, keep-alive, micro-batching, statistics and telemetry. Copies are held as objects; the kind sees them typed
/// through its <see cref="IEngineHost{TCopy}"/> and <see cref="IEngineLease{TCopy}"/>.
/// </summary>
internal sealed class HostedModel : IAsyncDisposable
{
    private readonly SemaphoreSlim _loadLock = new(1, 1);
    private readonly Lock _stateLock = new();
    private readonly EngineHosting _hosting;
    private readonly Func<object> _loadCopy;
    private readonly Action<object> _unloadCopy;
    private readonly Func<object, ModelDescription> _describe;
    private readonly Func<ValueTask> _disposeModel;
    private readonly List<Func<ValueTask>> _batchers = [];
    private TimeSpan? _keepAlive;
    private Channel<object>? _free;
    private List<object> _loaded = [];
    private int _roundRobin;
    private int _running;
    private int _queued;
    private ITimer? _expiry;
    private DateTimeOffset? _expiresAt;
    private bool _disposed;
    private int _disposing;

    private HostedModel(string name, object model, string kind, EngineHosting hosting, EngineOptions options, Func<object> loadCopy,
        Action<object> unloadCopy, Func<object, ModelDescription> describe, Func<ValueTask> disposeModel)
    {
        Name = name;
        Model = model;
        Kind = kind;
        Options = options;
        _hosting = hosting;
        _keepAlive = hosting.KeepAlive;
        _loadCopy = loadCopy;
        _unloadCopy = unloadCopy;
        _describe = describe;
        _disposeModel = disposeModel;
    }

    /// <summary>Hosts <paramref name="model"/> as <paramref name="name"/> and connects the model to its host.</summary>
    public static HostedModel Create<TCopy>(string name, EngineModel<TCopy> model, EngineOptions options)
        where TCopy : class
    {
        var hosted = new HostedModel(name, model, model.Kind, model.Hosting, options, model.LoadCopy, copy => model.UnloadCopy((TCopy)copy),
            copy => model.Describe((TCopy)copy), model.DisposeAsync);
        model.Attach(new Host<TCopy>(hosted));
        return hosted;
    }

    /// <summary>Throws when a model cannot be hosted as it asks.</summary>
    public static void Check(string name, EngineHosting hosting)
    {
        if (!hosting.Reloadable && (hosting.Instances > 1 || hosting.KeepAlive is not null))
        {
            throw NotReloadable(name);
        }
    }

    public static InvalidOperationException NotReloadable(string name) =>
        new($"'{name}' uses a model object you created, which the engine cannot copy or reload; Instances and KeepAlive need a package or a factory.");

    public string Name { get; }

    /// <summary>The kind's object (the <see cref="EngineModel{TCopy}"/>), which callers reach through the contracts it implements.</summary>
    public object Model { get; }

    public string Kind { get; }

    public EngineOptions Options { get; }

    public EngineStatistics Statistics { get; } = new();

    /// <summary>Set when the first copy loads.</summary>
    public ModelDescription? Description { get; private set; }

    public void SetKeepAlive(TimeSpan? idle)
    {
        if (!_hosting.Reloadable)
        {
            throw new InvalidOperationException($"'{Name}' uses a model object you created, which the engine cannot reload; its keep-alive cannot be set.");
        }

        lock (_stateLock)
        {
            _keepAlive = idle;
        }

        ScheduleExpiry();
    }

    public ModelStatus Status()
    {
        lock (_stateLock)
        {
            return new ModelStatus(Name, Kind, _loaded.Count > 0, _hosting.Instances, _running, _queued, _running == 0 && _queued == 0 ? _expiresAt : null);
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
            var created = new List<object>(_hosting.Instances);
            try
            {
                for (int i = 0; i < _hosting.Instances; i++)
                {
                    created.Add(await Task.Run(LoadCopy, cancellationToken).ConfigureAwait(false));
                }
            }
            catch
            {
                created.ForEach(_unloadCopy);
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

    private object LoadCopy()
    {
        var copy = _loadCopy();
        Description ??= _describe(copy);
        return copy;
    }

    /// <summary>Waits for the model (loading it if needed) and a copy; the lease must be disposed.</summary>
    private async ValueTask<IEngineLease<TCopy>> AcquireAsync<TCopy>(CancellationToken cancellationToken)
    {
        lock (_stateLock)
        {
            if (_hosting.QueueLimit is { } limit && _queued >= limit)
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
        var timeout = _hosting.Timeout is { } t ? new CancellationTokenSource(t, Options.Time) : null;
        var linked = timeout is null ? null : CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, timeout.Token);
        var token = linked?.Token ?? cancellationToken;
        bool counted = true;
        try
        {
            await EnsureLoadedAsync(token).ConfigureAwait(false);
            object instance;
            if (_hosting.SharedCopies)
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

            return new Lease<TCopy>(this, instance, start, Stopwatch.GetElapsedTime(start), token, timeout, linked);
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

    private MicroBatcher<TIn, TOut> Batcher<TIn, TOut>(int maxBatch, TimeSpan maxWait, Func<IReadOnlyList<TIn>, CancellationToken, Task<IReadOnlyList<TOut>>> run)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxBatch);
        ArgumentNullException.ThrowIfNull(run);
        var batcher = new MicroBatcher<TIn, TOut>(maxBatch, maxWait, run, Options.Time);
        lock (_stateLock)
        {
            ObjectDisposedException.ThrowIf(_disposing != 0, this);
            _batchers.Add(batcher.StopAsync);
        }

        return batcher;
    }

    private void Failed(Exception ex, CancellationTokenSource? timeout)
    {
        Statistics.Failed();
        Publish(EngineEventKind.RequestRejected, TimeSpan.Zero, TimeSpan.Zero, 0, timeout?.IsCancellationRequested == true ? "timeout" : ex.Message);
    }

    private Exception Translate(Exception ex, CancellationTokenSource? timeout) =>
        ex is OperationCanceledException && timeout?.IsCancellationRequested == true
            ? new TimeoutException($"'{Name}' did not answer within {_hosting.Timeout!.Value.TotalSeconds:0.###} s.", ex)
            : ex;

    private void Release(object instance)
    {
        lock (_stateLock)
        {
            _running--;
        }

        if (!_hosting.SharedCopies)
        {
            _free?.Writer.TryWrite(instance);
        }

        ScheduleExpiry();
    }

    private void ScheduleExpiry()
    {
        lock (_stateLock)
        {
            if (_keepAlive is not { } idle || _running > 0 || _queued > 0 || _loaded.Count == 0)
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

            // Wait for running requests to give their copies back before unloading them.
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
            instances.ForEach(_unloadCopy);
            Publish(EngineEventKind.ModelUnloaded, TimeSpan.Zero, TimeSpan.Zero, 0, null);
        }
        finally
        {
            _loadLock.Release();
        }
    }

    private void Publish(EngineEventKind kind, TimeSpan duration, TimeSpan wait, int batch, string? reason)
    {
        if (Options.PublishTelemetry && Idrak.Abstraction.Diagnostics.Telemetry.IsEnabled(TelemetryLevel.Engine))
        {
            Idrak.Abstraction.Diagnostics.Telemetry.Engine(new EngineEvent(kind, Name, duration, wait, batch, reason));
        }
    }

    /// <summary>Stops the model's batchers, then the model's own work, then unloads every copy (once).</summary>
    public async ValueTask DisposeAsync()
    {
        List<Func<ValueTask>> batchers;
        lock (_stateLock)
        {
            if (_disposing != 0)
            {
                return;
            }

            _disposing = 1;
            batchers = [.. _batchers];
        }

        foreach (var stop in batchers)
        {
            await stop().ConfigureAwait(false);
        }

        await _disposeModel().ConfigureAwait(false);
        _disposed = true;
        await UnloadAsync(force: true).ConfigureAwait(false);
    }

    /// <summary>What the kind sees of the engine.</summary>
    private sealed class Host<TCopy>(HostedModel owner) : IEngineHost<TCopy>
        where TCopy : class
    {
        public string Name => owner.Name;

        public ValueTask<IEngineLease<TCopy>> AcquireAsync(CancellationToken cancellationToken) => owner.AcquireAsync<TCopy>(cancellationToken);

        public IEngineBatcher<TIn, TOut> Batcher<TIn, TOut>(int maxBatch, TimeSpan maxWait, Func<IReadOnlyList<TIn>, CancellationToken, Task<IReadOnlyList<TOut>>> run) =>
            owner.Batcher(maxBatch, maxWait, run);
    }

    /// <summary>A copy of the model in use by one request.</summary>
    private sealed class Lease<TCopy>(HostedModel owner, object instance, long start, TimeSpan queueWait, CancellationToken token,
        CancellationTokenSource? timeout, CancellationTokenSource? linked) : IEngineLease<TCopy>
    {
        private int _released;

        public TCopy Copy { get; } = (TCopy)instance;

        public CancellationToken Token { get; } = token;

        public TimeSpan QueueWait { get; } = queueWait;

        public void Completed(int batch, long units, TimeSpan modelTime)
        {
            owner.Statistics.ModelCall(units, modelTime);
            var latency = Stopwatch.GetElapsedTime(start);
            owner.Statistics.Completed(latency, QueueWait);
            owner.Publish(EngineEventKind.RequestCompleted, latency, QueueWait, batch, null);
        }

        public Exception Fail(Exception exception)
        {
            owner.Failed(exception, timeout);
            return owner.Translate(exception, timeout);
        }

        // Translates a timeout into TimeoutException and counts failures, for streams consumed chunk by chunk.
        public async IAsyncEnumerable<T> Guard<T>(IAsyncEnumerable<T> source)
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
                        throw Fail(ex);
                    }

                    yield return item;
                }
            }
            finally
            {
                await enumerator.DisposeAsync().ConfigureAwait(false);
            }
        }

        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) != 0)
            {
                return;
            }

            timeout?.Dispose();
            linked?.Dispose();
            owner.Release(instance);
        }
    }
}

/// <summary>Collects single requests until <c>maxBatch</c> are waiting or <c>maxWait</c> has passed since the first, then answers them together.</summary>
internal sealed class MicroBatcher<TIn, TOut> : IEngineBatcher<TIn, TOut>
{
    private readonly int _maxBatch;
    private readonly TimeSpan _maxWait;
    private readonly Func<IReadOnlyList<TIn>, CancellationToken, Task<IReadOnlyList<TOut>>> _run;
    private readonly TimeProvider _time;
    private readonly Channel<Pending> _pending = Channel.CreateUnbounded<Pending>(new UnboundedChannelOptions { SingleReader = true });
    private readonly CancellationTokenSource _stop = new();
    private readonly Task _loop;
    private int _stopped;

    public MicroBatcher(int maxBatch, TimeSpan maxWait, Func<IReadOnlyList<TIn>, CancellationToken, Task<IReadOnlyList<TOut>>> run, TimeProvider time)
    {
        _maxBatch = maxBatch;
        _maxWait = maxWait;
        _run = run;
        _time = time;
        _loop = Task.Run(LoopAsync);
    }

    public Task<TOut> SubmitAsync(TIn input, CancellationToken cancellationToken = default)
    {
        var pending = new Pending(input, new TaskCompletionSource<TOut>(TaskCreationOptions.RunContinuationsAsynchronously), cancellationToken);
        if (!_pending.Writer.TryWrite(pending))
        {
            throw new ObjectDisposedException(nameof(InferenceEngine), "The inference engine hosting this model was disposed.");
        }

        return pending.Result.Task;
    }

    public async ValueTask StopAsync()
    {
        if (Interlocked.Exchange(ref _stopped, 1) != 0)
        {
            return;
        }

        _stop.Cancel();
        _pending.Writer.TryComplete();
        await _loop.ConfigureAwait(false);
        _stop.Dispose();
    }

    private async Task LoopAsync()
    {
        var reader = _pending.Reader;
        try
        {
            while (await reader.WaitToReadAsync(_stop.Token).ConfigureAwait(false))
            {
                var batch = new List<Pending>(_maxBatch);
                if (!reader.TryRead(out var first))
                {
                    continue;
                }

                batch.Add(first);
                using (var window = new CancellationTokenSource(_maxWait, _time))
                using (var linked = CancellationTokenSource.CreateLinkedTokenSource(window.Token, _stop.Token))
                {
                    while (batch.Count < _maxBatch)
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
                    var results = await _run([.. batch.Select(p => p.Input)], CancellationToken.None).ConfigureAwait(false);
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

    private sealed record Pending(TIn Input, TaskCompletionSource<TOut> Result, CancellationToken Cancellation);
}
