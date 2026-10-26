// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>The page of a feature not built yet (<see cref="PlannedFeature"/>): what it will do and need.</summary>
public partial class PlaceholderPage : ContentPage
{
    /// <summary>A page bound to <paramref name="viewModel"/>.</summary>
    public PlaceholderPage(PlaceholderViewModel viewModel)
    {
        ArgumentNullException.ThrowIfNull(viewModel);
        BindingContext = viewModel;
        InitializeComponent();
    }
}
