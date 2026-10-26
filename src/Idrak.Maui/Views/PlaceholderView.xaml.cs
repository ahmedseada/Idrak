// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>A planned lab's view, from its registration alone.</summary>
public partial class PlaceholderView : ContentView
{
    /// <summary>A view bound to <paramref name="viewModel"/>.</summary>
    public PlaceholderView(PlaceholderViewModel viewModel)
    {
        BindingContext = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        InitializeComponent();
    }
}
