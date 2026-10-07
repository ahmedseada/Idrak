// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Generation;

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
    private readonly Lock _lock = new();
    private readonly ITimer _sweeper;

    /// <summary>Creates the host.</summary>
    /// <param name="load">Loads a model by name (called under the host's lock, so concurrent first requests load once).</param>
    /// <param name="defaultKeepAlive">Used when a lease does not specify one (null = forever).</param>
    /// <param name="timeProvider">Clock, replaceable in tests.</param>
    public ModelHost(Func<string, TModel> load, TimeSpan? defaultKeepAlive, TimeProvider? timeProvider = null)
    {
        _load = load;
        _defaultKeepAlive = defaultKeepAlive;
        _time = timeProvider ?? TimeProvider.System;
        _sweeper = _time.CreateTimer(_ => Sweep(), null, TimeSpan.FromSeconds(1), TimeSpan.FromSeconds(1));
    }

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

    /// <summary>Returns the model <paramref name="name"/>, loading it if needed; it stays loaded for the default keep-alive after the lease ends.</summary>
    public Lease Acquire(string name) => Acquire(name, _defaultKeepAlive);

    /// <summary>
    /// Returns the model <paramref name="name"/>, loading it if needed. Dispose the lease when done; the model then stays
    /// loaded for <paramref name="keepAlive"/> (<see cref="TimeSpan.Zero"/> unloads at once, null keeps it forever).
    /// </summary>
    public Lease Acquire(string name, TimeSpan? keepAlive)
    {
        lock (_lock)
        {
            TimeSpan loadTime = TimeSpan.Zero;
            if (!_models.TryGetValue(name, out var entry))
            {
                long start = _time.GetTimestamp();
                entry = new Entry(_load(name), _time.GetUtcNow());
                loadTime = _time.GetElapsedTime(start);
                _models[name] = entry;
            }

            entry.Users++;
            return new Lease(this, name, entry.Model, loadTime, keepAlive);
        }
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
            entry.ExpiresAt = keepAlive is { } k ? _time.GetUtcNow() + k : null;
            if (entry.Users == 0 && keepAlive == TimeSpan.Zero)
            {
                _models.Remove(name);
                entry.Model.Dispose();
            }
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
            foreach (var entry in _models.Values)
            {
                entry.Model.Dispose();
            }

            _models.Clear();
        }
    }

    private sealed class Entry(TModel model, DateTimeOffset loadedAt)
    {
        public TModel Model { get; } = model;
        public DateTimeOffset LoadedAt { get; } = loadedAt;
        public DateTimeOffset? ExpiresAt { get; set; }
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
