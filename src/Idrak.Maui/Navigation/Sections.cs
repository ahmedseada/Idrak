// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Views;

namespace Idrak.Maui.Navigation;

/// <summary>The library's sections of the app's navigation, registered as defaults in <see cref="AppSections"/>.</summary>
internal static class LibrarySections
{
    public static IEnumerable<IAppSection> All() =>
    [
        new HomeSection(),
        new FeatureSection(AppSections.Chat, "Chat", "chat", 10, MobileFeatures.Chat, "A model from the store chats on this device"),
        new SegmentedSection(AppSections.Voice, "Voice", "mic", 20, [MobileFeatures.Transcribe, MobileFeatures.Speak, MobileFeatures.AudioProcess],
            "Speech into text, text into speech, a recording through the library's processors"),
        new SegmentedSection(AppSections.Vision, "Vision", "vision", 30, [MobileFeatures.Ocr, MobileFeatures.Image, MobileFeatures.Generate],
            "A photo read by a model; transforms and scan looks on a photo; text drawn as scanned pages"),
        new LabsSection(),
        new SettingsSection(),
    ];
}

/// <summary>Home: the library and the app, the device measured, the labs to try (<see cref="HomeView"/>).</summary>
public sealed class HomeSection : IAppSection
{
    /// <inheritdoc />
    public string Name => AppSections.Home;

    /// <inheritdoc />
    public string Summary => "The library and the app, this device measured, the labs to try";

    /// <inheritdoc />
    public string Title => "Home";

    /// <inheritdoc />
    public string Icon => "home";

    /// <inheritdoc />
    public int Order => 0;

    /// <inheritdoc />
    public SectionPlacement Placement => SectionPlacement.Main;

    /// <inheritdoc />
    public IReadOnlyList<string> Features => [];

    /// <inheritdoc />
    public View CreateView(IServiceProvider services) => new HomeView(services);
}

/// <summary>A section showing one feature's view (the Chat tab).</summary>
/// <param name="name">The section's name.</param>
/// <param name="title">Its label.</param>
/// <param name="icon">Its icon's name.</param>
/// <param name="order">Its place.</param>
/// <param name="feature">The feature it shows.</param>
/// <param name="summary">One line on it.</param>
public sealed class FeatureSection(string name, string title, string icon, int order, string feature, string summary) : IAppSection
{
    /// <inheritdoc />
    public string Name => name;

    /// <inheritdoc />
    public string Summary => summary;

    /// <inheritdoc />
    public string Title => title;

    /// <inheritdoc />
    public string Icon => icon;

    /// <inheritdoc />
    public int Order => order;

    /// <inheritdoc />
    public SectionPlacement Placement => SectionPlacement.Main;

    /// <inheritdoc />
    public IReadOnlyList<string> Features { get; } = [feature];

    /// <inheritdoc />
    public View CreateView(IServiceProvider services) => FeatureViews.Create(services, feature);
}

/// <summary>A section showing several features under a segmented control (Voice: Transcribe, Speak, Studio).</summary>
/// <param name="name">The section's name.</param>
/// <param name="title">Its label and heading.</param>
/// <param name="icon">Its icon's name.</param>
/// <param name="order">Its place.</param>
/// <param name="features">The features it shows, in the control's order (each by its title and icon).</param>
/// <param name="summary">One line on it.</param>
public sealed class SegmentedSection(string name, string title, string icon, int order, IReadOnlyList<string> features, string summary) : IAppSection
{
    /// <inheritdoc />
    public string Name => name;

    /// <inheritdoc />
    public string Summary => summary;

    /// <inheritdoc />
    public string Title => title;

    /// <inheritdoc />
    public string Icon => icon;

    /// <inheritdoc />
    public int Order => order;

    /// <inheritdoc />
    public SectionPlacement Placement => SectionPlacement.Main;

    /// <inheritdoc />
    public IReadOnlyList<string> Features => features;

    /// <inheritdoc />
    public View CreateView(IServiceProvider services) => new SegmentedSectionView(services, this);
}

/// <summary>All labs as a page of the rail (a narrow window shows them as a sheet over Home instead).</summary>
public sealed class LabsSection : IAppSection
{
    /// <inheritdoc />
    public string Name => AppSections.Labs;

    /// <inheritdoc />
    public string Summary => "Every lab, grouped, searchable and filtered by what it needs";

    /// <inheritdoc />
    public string Title => "All labs";

    /// <inheritdoc />
    public string Icon => "grid";

    /// <inheritdoc />
    public int Order => 80;

    /// <inheritdoc />
    public SectionPlacement Placement => SectionPlacement.Foot;

    /// <inheritdoc />
    public IReadOnlyList<string> Features => [];

    /// <inheritdoc />
    public View CreateView(IServiceProvider services) => new LabsView(services, asSheet: false);
}

/// <summary>The settings as a page of the rail (a narrow window opens them from Home's header).</summary>
public sealed class SettingsSection : IAppSection
{
    /// <inheritdoc />
    public string Name => AppSections.Settings;

    /// <inheritdoc />
    public string Summary => "The device, the theme, storage, limits, the token, the features shown";

    /// <inheritdoc />
    public string Title => "Settings";

    /// <inheritdoc />
    public string Icon => "gear";

    /// <inheritdoc />
    public int Order => 90;

    /// <inheritdoc />
    public SectionPlacement Placement => SectionPlacement.Foot;

    /// <inheritdoc />
    public IReadOnlyList<string> Features => [];

    /// <inheritdoc />
    public View CreateView(IServiceProvider services) => SettingsPages.CreateView(services);
}

/// <summary>A feature's view for a section: its own (<see cref="IMobileFeature.CreateView"/>), or a card that opens its page.</summary>
public static class FeatureViews
{
    /// <summary>The view of the feature <paramref name="name"/>, bound to a new view model of it (or the app's shared one).</summary>
    /// <exception cref="NotRegisteredException">No feature has that name.</exception>
    public static View Create(IServiceProvider services, string name)
    {
        ArgumentNullException.ThrowIfNull(services);
        var feature = MobileFeatures.Get(name);
        object viewModel = feature.CreateViewModel(services);
        var view = feature.CreateView(services, viewModel) ?? new FeatureLinkView(feature, services);
        view.BindingContext ??= viewModel;
        return view;
    }
}
