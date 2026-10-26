// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui;
using Idrak.Maui.Navigation;
using Idrak.Maui.Theme;

namespace Idrak.Mobile;

/// <summary>
/// The app: the theme the settings choose (the phone's by default), and the library's navigation host as its window's page
/// (Home, Chat, Voice and Vision on the bottom bar of a narrow window, on the side rail of a wide one).
/// </summary>
public partial class App : Application
{
    private readonly IServiceProvider _services;

    /// <summary>The app, its theme applied.</summary>
    public App(IServiceProvider services, IdrakMauiOptions options)
    {
        _services = services;
        InitializeComponent();
        IdrakThemes.Apply(this, options.Theme);
    }

    /// <inheritdoc />
    protected override Window CreateWindow(IActivationState? activationState) => AppHostPage.CreateWindow(_services);
}
