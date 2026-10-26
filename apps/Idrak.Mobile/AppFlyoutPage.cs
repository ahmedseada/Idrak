// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Views;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Mobile;

/// <summary>
/// The app's navigation (no Shell): a flyout listing the registered features and the settings, each opened in a
/// navigation page as the detail.
/// </summary>
public sealed class AppFlyoutPage : FlyoutPage
{
    private readonly IServiceProvider _services;

    /// <summary>The flyout over <paramref name="menu"/>, opening the first feature.</summary>
    public AppFlyoutPage(IServiceProvider services, MenuPage menu)
    {
        _services = services;
        Flyout = menu;
        FlyoutLayoutBehavior = FlyoutLayoutBehavior.Popover;
        menu.Selected += (_, entry) => Show(entry);
        Detail = new NavigationPage(menu.Entries.Count > 0 && menu.Entries[0].Feature is { } first ? MobileFeatures.Open(first, services) : services.GetRequiredService<SettingsPage>());
    }

    private void Show(MenuEntry entry)
    {
        try
        {
            var page = entry.Feature is { } name ? MobileFeatures.Open(name, _services) : _services.GetRequiredService<SettingsPage>();
            Detail = new NavigationPage(page);
        }
        catch (Exception e)
        {
            Detail = new NavigationPage(new ContentPage { Title = entry.Title, Content = new Label { Text = e.Message, Margin = 16 } });
        }

        IsPresented = false;
    }
}
