// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>The settings page over <see cref="IdrakMauiOptions"/>.</summary>
public partial class SettingsPage : ContentPage
{
    /// <summary>A page bound to <paramref name="viewModel"/>.</summary>
    public SettingsPage(SettingsViewModel viewModel)
    {
        BindingContext = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        InitializeComponent();
    }
}
