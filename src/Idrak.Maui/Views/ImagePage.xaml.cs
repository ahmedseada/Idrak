// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>The Image feature's page, bound to its view model.</summary>
public partial class ImagePage : ContentPage
{
    /// <summary>A page bound to <paramref name="viewModel"/>.</summary>
    public ImagePage(ImageViewModel viewModel)
    {
        ArgumentNullException.ThrowIfNull(viewModel);
        BindingContext = viewModel;
        InitializeComponent();
    }
}
