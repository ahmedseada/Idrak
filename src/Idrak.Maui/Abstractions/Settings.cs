// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Maui.Abstractions;

/// <summary>
/// Where settings are kept between runs, as text by key: MAUI's Preferences in an app (<c>Idrak.Maui.MauiPreferencesStore</c>),
/// an in-memory one in tests (<c>Idrak.Maui.MemorySettingsStore</c>). Given to <c>UseIdrak</c> through
/// <see cref="IdrakMauiOptions.SettingsStore"/> and registered in the app's services.
/// </summary>
public interface ISettingsStore
{
    /// <summary>The value kept under <paramref name="key"/>, or null when none is.</summary>
    string? Get(string key);

    /// <summary>Keeps <paramref name="value"/> under <paramref name="key"/>; null removes it.</summary>
    void Set(string key, string? value);
}

/// <summary>
/// Where secrets are kept between runs (a Hugging Face token): the phone's secure storage in an app
/// (<c>Idrak.Maui.MauiSecureStore</c>, SecureStorage), an in-memory one in tests. Given to <c>UseIdrak</c> through
/// <see cref="IdrakMauiOptions.SecretStore"/> and registered in the app's services.
/// </summary>
public interface ISecretStore
{
    /// <summary>The secret kept under <paramref name="key"/>, or null when none is.</summary>
    Task<string?> GetAsync(string key);

    /// <summary>Keeps <paramref name="value"/> under <paramref name="key"/>; null or empty removes it.</summary>
    Task SetAsync(string key, string? value);
}
