// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>The Generate lab's fonts: searched, chosen, added (<see cref="GenerateViewModel"/>'s).</summary>
public partial class FontsView : ContentView
{
    /// <summary>A view bound to <paramref name="viewModel"/>.</summary>
    public FontsView(GenerateViewModel viewModel)
    {
        ArgumentNullException.ThrowIfNull(viewModel);
        BindingContext = viewModel;
        InitializeComponent();
    }
}
