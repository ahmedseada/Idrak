// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using Idrak.Maui;
using Idrak.Maui.Abstractions;

namespace Idrak.Mobile;

/// <summary>One line of the menu: a feature (by name) or the settings (no feature).</summary>
/// <param name="Feature">The feature's name, or null for the settings.</param>
/// <param name="Title">Its title.</param>
/// <param name="Summary">What it does.</param>
/// <param name="Glyph">Its mark.</param>
public sealed record MenuEntry(string? Feature, string Title, string Summary, string Glyph);

/// <summary>The flyout's menu, listed again when the settings are saved (features turned on or off).</summary>
public partial class MenuPage : ContentPage
{
    private readonly IdrakMauiOptions _options;

    /// <summary>The menu of the features <paramref name="options"/> show.</summary>
    public MenuPage(IdrakMauiOptions options, IdrakMauiSettings settings)
    {
        _options = options;
        InitializeComponent();
        BindingContext = this;
        settings.Saved += (_, _) => Dispatcher.Dispatch(Fill);
        Fill();
    }

    /// <summary>Raised when an entry is chosen.</summary>
    public event EventHandler<MenuEntry>? Selected;

    /// <summary>The entries: the features in menu order, then the settings.</summary>
    public ObservableCollection<MenuEntry> Entries { get; } = [];

    private void Fill()
    {
        Entries.Clear();
        foreach (var feature in MobileFeatures.Enabled(_options))
        {
            Entries.Add(new MenuEntry(feature.Name, feature.Title, feature.Summary, feature.Glyph));
        }

        Entries.Add(new MenuEntry(null, "Settings", "Device, theme, storage, limits, token, features", "⚙"));
    }

    private void OnSelectionChanged(object? sender, SelectionChangedEventArgs e)
    {
        if (e.CurrentSelection.FirstOrDefault() is MenuEntry entry)
        {
            Selected?.Invoke(this, entry);
            MenuList.SelectedItem = null;
        }
    }
}
