// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// The library's "terminal" feature: the app's command runner (<see cref="ICommandRunner"/>, registered by the app;
/// Idrak.Mobile's runs the idrak commands) in the app's own process, its output streamed and drawn once a display frame.
/// One view model for the app: a command outlives its page.
/// </summary>
public sealed class TerminalFeature : IMobileFeature
{
    /// <inheritdoc />
    public string Name => MobileFeatures.Terminal;

    /// <inheritdoc />
    public string Summary => "The library's command-line tool in the app's own process: its commands as you type them, their output live; Stop, history, copy, clear, help";

    /// <inheritdoc />
    public string Title => "Terminal";

    /// <inheritdoc />
    public string Glyph => "›";

    /// <inheritdoc />
    public int Order => 85;

    /// <inheritdoc />
    public FeatureGroup Group => FeatureGroup.Device;

    /// <inheritdoc />
    public string Tagline => "The idrak commands";

    /// <inheritdoc />
    public string Icon => "terminal";

    /// <inheritdoc />
    public PhoneNeeds Needs => PhoneNeeds.None;

    /// <inheritdoc />
    public IReadOnlyList<string> Packages { get; } = [];

    /// <inheritdoc />
    public IReadOnlyList<string> ModelFamilies { get; } = [];

    /// <inheritdoc />
    public object CreateViewModel(IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        if (services.GetService<TerminalViewModel>() is { } shared)
        {
            return shared;
        }

        string storage = services.GetService<IdrakStartupReport>()?.AppStorage ?? ComputeResources.CacheFolder;
        return new TerminalViewModel(services.GetService<ICommandRunner>(), services.GetService<ISettingsStore>() ?? new MemorySettingsStore(), storage,
            services.GetService<TerminalLabOptions>());
    }

    /// <inheritdoc />
    public Page CreatePage(IServiceProvider services, object viewModel) =>
        new TerminalPage(viewModel as TerminalViewModel ?? throw new ArgumentException($"The terminal page takes a {nameof(TerminalViewModel)}.", nameof(viewModel)), services);

    /// <inheritdoc />
    public View? CreateView(IServiceProvider services, object viewModel) =>
        viewModel is TerminalViewModel vm ? new TerminalView(vm, services) : throw new ArgumentException($"The {Name} view takes a {nameof(TerminalViewModel)}.", nameof(viewModel));
}
