// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.Concurrent;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>Settings in MAUI's Preferences (the app's own, kept between runs).</summary>
public sealed class MauiPreferencesStore : ISettingsStore
{
    /// <inheritdoc />
    public string? Get(string key) => Preferences.Default.ContainsKey(key) ? Preferences.Default.Get<string?>(key, null) : null;

    /// <inheritdoc />
    public void Set(string key, string? value)
    {
        if (value is null)
        {
            Preferences.Default.Remove(key);
        }
        else
        {
            Preferences.Default.Set(key, value);
        }
    }
}

/// <summary>Secrets in MAUI's SecureStorage (the Android keystore).</summary>
public sealed class MauiSecureStore : ISecretStore
{
    /// <inheritdoc />
    public Task<string?> GetAsync(string key) => SecureStorage.Default.GetAsync(key);

    /// <inheritdoc />
    public Task SetAsync(string key, string? value)
    {
        if (string.IsNullOrEmpty(value))
        {
            SecureStorage.Default.Remove(key);
            return Task.CompletedTask;
        }

        return SecureStorage.Default.SetAsync(key, value);
    }
}

/// <summary>Settings kept in memory only (tests, or an app that keeps none).</summary>
public sealed class MemorySettingsStore : ISettingsStore
{
    private readonly ConcurrentDictionary<string, string> _values = new(StringComparer.Ordinal);

    /// <summary>The keys held.</summary>
    public IReadOnlyCollection<string> Keys => [.. _values.Keys];

    /// <inheritdoc />
    public string? Get(string key) => _values.TryGetValue(key, out var value) ? value : null;

    /// <inheritdoc />
    public void Set(string key, string? value)
    {
        if (value is null)
        {
            _values.TryRemove(key, out _);
        }
        else
        {
            _values[key] = value;
        }
    }
}

/// <summary>Secrets kept in memory only (tests).</summary>
public sealed class MemorySecretStore : ISecretStore
{
    private readonly ConcurrentDictionary<string, string> _values = new(StringComparer.Ordinal);

    /// <inheritdoc />
    public Task<string?> GetAsync(string key) => Task.FromResult(_values.TryGetValue(key, out var value) ? value : null);

    /// <inheritdoc />
    public Task SetAsync(string key, string? value)
    {
        if (string.IsNullOrEmpty(value))
        {
            _values.TryRemove(key, out _);
        }
        else
        {
            _values[key] = value;
        }

        return Task.CompletedTask;
    }
}

/// <summary>A memory-pressure source raised by hand: where the platform has no signal (tests), or a "free memory" button.</summary>
public sealed class ManualMemoryPressure : IMemoryPressureSource
{
    /// <inheritdoc />
    public event EventHandler<MemoryPressureSignal>? Pressure;

    /// <summary>Raises a signal of <paramref name="level"/>.</summary>
    public void Raise(MemoryPressureLevel level, string source = "manual") => Pressure?.Invoke(this, new MemoryPressureSignal(level, source));
}
