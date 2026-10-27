// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Mobile.WinUI;

/// <summary>The Windows application (WinUI): builds the MAUI app (<see cref="MauiProgram.CreateMauiApp"/>), as Android's MainApplication does.</summary>
public partial class App : MauiWinUIApplication
{
    /// <summary>The WinUI application, its XAML loaded.</summary>
    public App() => InitializeComponent();

    /// <inheritdoc />
    protected override MauiApp CreateMauiApp() => MauiProgram.CreateMauiApp();
}
