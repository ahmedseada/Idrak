// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Views;

namespace Idrak.Maui.Navigation;

/// <summary>
/// The app's navigation (<see cref="IAppNavigator"/>): through the host page when one is shown (<see cref="Host"/>, which
/// <see cref="AppHostPage"/> sets), else by pushing pages on the window's navigation page.
/// </summary>
public sealed class AppNavigator(IServiceProvider services) : IAppNavigator
{
    /// <summary>The host page shown, or null.</summary>
    public AppHostPage? Host { get; set; }

    /// <inheritdoc />
    public Task OpenFeatureAsync(string feature)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(feature);
        return Host is { } host ? host.OpenFeatureAsync(feature) : PushAsync(MobileFeatures.Open(feature, services));
    }

    /// <inheritdoc />
    public Task<bool> OpenSectionAsync(string section) => Host is { } host ? host.OpenSectionAsync(section) : Task.FromResult(false);

    /// <inheritdoc />
    public Task OpenLabsAsync() => Host is { } host ? host.OpenLabsAsync() : PushAsync(new LabsPage(services));

    /// <inheritdoc />
    public Task OpenSettingsAsync() => Host is { } host ? host.OpenSettingsAsync() : PushAsync(SettingsPages.Create(services));

    /// <inheritdoc />
    public Task PushAsync(Page page)
    {
        ArgumentNullException.ThrowIfNull(page);
        if (Host is { } host)
        {
            return host.Navigation.PushAsync(page);
        }

        var root = Application.Current?.Windows.FirstOrDefault()?.Page;
        return root?.Navigation.PushAsync(page) ?? Task.CompletedTask;
    }
}
