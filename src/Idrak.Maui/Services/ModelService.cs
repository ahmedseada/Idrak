// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>
/// Models kept loaded through <see cref="ModelHost{TModel}"/>, one host a name, released on memory pressure
/// (<see cref="IModelService"/>).
/// </summary>
public sealed class ModelService : IModelService
{
    private readonly IMemoryPressureSource _pressure;
    private readonly IdrakMauiOptions _options;
    private readonly TimeProvider _time;
    private readonly Dictionary<string, Holder> _hosts = new(StringComparer.Ordinal);
    private readonly Lock _gate = new();

    /// <summary>Models over <paramref name="options"/>' keep-alive and limit, released on <paramref name="pressure"/>'s signals.</summary>
    public ModelService(IMemoryPressureSource pressure, IdrakMauiOptions options, TimeProvider? time = null)
    {
        _pressure = pressure ?? throw new ArgumentNullException(nameof(pressure));
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _time = time ?? TimeProvider.System;
        _pressure.Pressure += OnPressure;
    }

    /// <inheritdoc />
    public event EventHandler<MemoryRelease>? Released;

    /// <inheritdoc />
    public ModelHost<TModel> Host<TModel>(string name, Func<string, TModel> load) where TModel : class, IDisposable
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(name);
        ArgumentNullException.ThrowIfNull(load);
        lock (_gate)
        {
            if (_hosts.TryGetValue(name, out var held))
            {
                return held.Host as ModelHost<TModel> ?? throw new InvalidOperationException(
                    $"The model host '{name}' holds {held.ModelType.Name} models, not {typeof(TModel).Name}; give this one another name.");
            }

            var host = new ModelHost<TModel>(load, _options.ModelKeepAlive, _time) { MaxLoaded = _options.MaxLoadedModels };
            _hosts[name] = new Holder(typeof(TModel), host, () => host.Loaded, host.Unload);
            return host;
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<HostedModel> Loaded
    {
        get
        {
            lock (_gate)
            {
                return [.. _hosts.SelectMany(h => h.Value.Loaded().Select(m => new HostedModel(h.Key, m)))];
            }
        }
    }

    /// <inheritdoc />
    public MemoryRelease Release(MemoryPressureSignal signal)
    {
        ArgumentNullException.ThrowIfNull(signal);
        var unloaded = new List<string>();
        if (signal.Level >= _options.UnloadModelsAt)
        {
            lock (_gate)
            {
                foreach (var (name, holder) in _hosts)
                {
                    foreach (var model in holder.Loaded().Where(m => m.ActiveUsers == 0))
                    {
                        if (holder.Unload(model.Name))
                        {
                            unloaded.Add($"{name}/{model.Name}");
                        }
                    }
                }
            }
        }

        ComputeResources.ReleaseCachedMemory();
        if (signal.Level == MemoryPressureLevel.Critical)
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Aggressive, blocking: true, compacting: true);
        }

        var release = new MemoryRelease(signal, unloaded, CachedMemoryReleased: true);
        Released?.Invoke(this, release);
        return release;
    }

    /// <summary>Stops listening to the signal and unloads every model.</summary>
    public void Dispose()
    {
        _pressure.Pressure -= OnPressure;
        lock (_gate)
        {
            foreach (var holder in _hosts.Values)
            {
                holder.Host.Dispose();
            }

            _hosts.Clear();
        }
    }

    private void OnPressure(object? sender, MemoryPressureSignal signal) => Release(signal);

    private sealed record Holder(Type ModelType, IDisposable Host, Func<IReadOnlyList<LoadedModel>> Loaded, Func<string, bool> Unload);
}
