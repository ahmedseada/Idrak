// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Serving;

/// <summary>A loaded model as reported by <see cref="ModelHost{TModel}.Loaded"/>.</summary>
/// <param name="Name">The name it was loaded under.</param>
/// <param name="LoadedAt">When it was loaded.</param>
/// <param name="ExpiresAt">When it will be unloaded if unused (null = never, or in use).</param>
/// <param name="ActiveUsers">Leases currently held.</param>
public sealed record LoadedModel(string Name, DateTimeOffset LoadedAt, DateTimeOffset? ExpiresAt, int ActiveUsers);

/// <summary>
/// Keeps models loaded between requests and unloads each one when its keep-alive time has passed without use,
/// freeing its device memory. Loading happens on first use (the lease reports how long it took).
/// </summary>
public sealed class ModelHost<TModel> : IDisposable where TModel : class, IDisposable
{
    private readonly Func<string, TModel> _load;
    private readonly TimeSpan? _defaultKeepAlive;
    private readonly TimeProvider _time;
    private readonly Dictionary<string, Entry> _models = [];
    private readonly Dictionary<string, Pending> _loading = [];
    private readonly Lock _lock = new();
    private bool _disposed;
    private readonly ITimer _sweeper;

    /// <summary>Creates the host.</summary>
    /// <param name="load">
    /// Loads a model by name. It runs outside the host's lock (a download or a load never blocks <see cref="Loaded"/>,
    /// <see cref="Unload"/>, the keep-alive sweep or another name); concurrent first requests for one name still load it
    /// once, the later ones waiting for that load alone.
    /// </param>
    /// <param name="defaultKeepAlive">Used when a lease does not specify one (null = forever).</param>
    /// <param name="timeProvider">Clock, replaceable in tests.</param>
    public ModelHost(Func<string, TModel> load, TimeSpan? defaultKeepAlive, TimeProvider? timeProvider = null)
    {
        _load = load;
        _defaultKeepAlive = defaultKeepAlive;
        _time = timeProvider ?? TimeProvider.System;
        _sweeper = _time.CreateTimer(_ => Sweep(), null, TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(1));
    }

    /// <summary>
    /// The most models kept loaded at once (null: no limit). Loading another first unloads idle ones, the longest unused
    /// first, until it fits: large models (one a device) then replace each other instead of filling its memory. A model in
    /// use is never unloaded for it.
    /// </summary>
    public int? MaxLoaded { get; init; }

    /// <summary>Models currently loaded.</summary>
    public IReadOnlyList<LoadedModel> Loaded
    {
        get
        {
            lock (_lock)
            {
                return [.. _models.Select(p => new LoadedModel(p.Key, p.Value.LoadedAt, p.Value.Users > 0 ? null : p.Value.ExpiresAt, p.Value.Users))];
            }
        }
    }

    /// <summary>The names being loaded now (a first <see cref="Acquire(string, TimeSpan?)"/> still running), in no order.</summary>
    public IReadOnlyList<string> Loading
    {
        get
        {
            lock (_lock)
            {
                return [.. _loading.Keys];
            }
        }
    }

    /// <summary>Returns the model <paramref name="name"/>, loading it if needed; it stays loaded for the default keep-alive after the lease ends.</summary>
    public Lease Acquire(string name) => Acquire(name, _defaultKeepAlive);

    /// <summary>
    /// Returns the model <paramref name="name"/>, loading it if needed. Dispose the lease when done; the model then stays
    /// loaded for <paramref name="keepAlive"/> (<see cref="TimeSpan.Zero"/> unloads at once, null keeps it forever).
    /// The load runs on the calling thread without the host's lock held; a second caller for the same name waits for
    /// that load (its lease reports the time it waited), and a failed load fails every caller waiting for it.
    /// </summary>
    public Lease Acquire(string name, TimeSpan? keepAlive)
    {
        Pending? waitFor = null;
        Pending? mine = null;
        lock (_lock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_models.TryGetValue(name, out var entry))
            {
                entry.Users++;
                return new Lease(this, name, entry.Model, TimeSpan.Zero, keepAlive);
            }

            if (_loading.TryGetValue(name, out waitFor))
            {
                waitFor.Waiters++;
            }
            else
            {
                MakeRoomLocked();
                mine = new Pending();
                _loading[name] = mine;
            }
        }

        long start = _time.GetTimestamp();
        if (waitFor is not null)
        {
            // Another caller is loading this name: wait for that load alone (the host's lock stays free).
            var loaded = waitFor.Done.Task.GetAwaiter().GetResult();
            return new Lease(this, name, loaded.Model, _time.GetElapsedTime(start), keepAlive);
        }

        TModel model;
        try
        {
            model = _load(name);
        }
        catch (Exception error)
        {
            lock (_lock)
            {
                _loading.Remove(name);
            }

            mine!.Done.TrySetException(error);
            throw;
        }

        var loadTime = _time.GetElapsedTime(start);
        Entry added;
        lock (_lock)
        {
            _loading.Remove(name);
            if (_disposed)
            {
                model.Dispose();
                var gone = new ObjectDisposedException(GetType().FullName);
                mine!.Done.TrySetException(gone);
                throw gone;
            }

            // The loader and every caller that waited hold it from the start, so nothing unloads it before they have it.
            added = new Entry(model, _time.GetUtcNow()) { Users = 1 + mine!.Waiters };
            _models[name] = added;
        }

        mine.Done.TrySetResult(added);
        return new Lease(this, name, model, loadTime, keepAlive);
    }

    /// <summary>Unloads a model now if nobody is using it; returns whether it was unloaded.</summary>
    public bool Unload(string name)
    {
        lock (_lock)
        {
            if (_models.TryGetValue(name, out var entry) && entry.Users == 0)
            {
                _models.Remove(name);
                entry.Model.Dispose();
                return true;
            }

            return false;
        }
    }

    private void Release(string name, TimeSpan? keepAlive)
    {
        lock (_lock)
        {
            if (!_models.TryGetValue(name, out var entry))
            {
                return;
            }

            entry.Users--;
            entry.LastUsed = _time.GetUtcNow();
            entry.ExpiresAt = keepAlive is { } k ? _time.GetUtcNow() + k : null;
            if (entry.Users == 0 && keepAlive == TimeSpan.Zero)
            {
                _models.Remove(name);
                entry.Model.Dispose();
            }
        }
    }

    // Unloads idle models, the longest unused first, until one more fits under MaxLoaded (the loads under way count).
    // Called under the lock.
    private void MakeRoomLocked()
    {
        if (MaxLoaded is not { } most)
        {
            return;
        }

        foreach (var (name, entry) in _models.Where(p => p.Value.Users == 0).OrderBy(p => p.Value.LastUsed ?? p.Value.LoadedAt).ToList())
        {
            if (_models.Count + _loading.Count < Math.Max(1, most))
            {
                return;
            }

            _models.Remove(name);
            entry.Model.Dispose();
        }
    }

    internal void Sweep()
    {
        lock (_lock)
        {
            var now = _time.GetUtcNow();
            foreach (var (name, entry) in _models.Where(p => p.Value.Users == 0 && p.Value.ExpiresAt <= now).ToList())
            {
                _models.Remove(name);
                entry.Model.Dispose();
            }
        }
    }

    /// <summary>Unloads everything.</summary>
    public void Dispose()
    {
        _sweeper.Dispose();
        lock (_lock)
        {
            _disposed = true;
            foreach (var entry in _models.Values)
            {
                entry.Model.Dispose();
            }

            _models.Clear();
        }
    }

    // A first load under way: the callers that wait for it, and its outcome.
    private sealed class Pending
    {
        public TaskCompletionSource<Entry> Done { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);

        public int Waiters { get; set; }
    }

    private sealed class Entry(TModel model, DateTimeOffset loadedAt)
    {
        public TModel Model { get; } = model;
        public DateTimeOffset LoadedAt { get; } = loadedAt;
        public DateTimeOffset? ExpiresAt { get; set; }
        public DateTimeOffset? LastUsed { get; set; }
        public int Users { get; set; }
    }

    /// <summary>Use of a hosted model; dispose it to start the keep-alive countdown.</summary>
    public sealed class Lease : IDisposable
    {
        private readonly ModelHost<TModel> _host;
        private int _released;

        internal Lease(ModelHost<TModel> host, string name, TModel model, TimeSpan loadDuration, TimeSpan? keepAlive)
        {
            _host = host;
            Name = name;
            Model = model;
            LoadDuration = loadDuration;
            KeepAlive = keepAlive;
        }

        /// <summary>The model name.</summary>
        public string Name { get; }

        /// <summary>The model.</summary>
        public TModel Model { get; }

        /// <summary>Time spent loading for this lease (zero when it was already loaded).</summary>
        public TimeSpan LoadDuration { get; }

        /// <summary>How long the model stays loaded after this lease (null = forever).</summary>
        public TimeSpan? KeepAlive { get; set; }

        /// <inheritdoc />
        public void Dispose()
        {
            if (Interlocked.Exchange(ref _released, 1) == 0)
            {
                _host.Release(Name, KeepAlive);
            }
        }
    }
}
