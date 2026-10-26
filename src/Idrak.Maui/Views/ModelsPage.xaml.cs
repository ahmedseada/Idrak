// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>The Models feature's page: reads the store each time it appears, and asks before a model is removed.</summary>
public partial class ModelsPage : ContentPage
{
    private readonly ModelsViewModel _viewModel;

    /// <summary>A page bound to <paramref name="viewModel"/>.</summary>
    public ModelsPage(ModelsViewModel viewModel)
    {
        _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        viewModel.Confirm = (title, message) => DisplayAlertAsync(title, message, "Remove", "Keep");
        InitializeComponent();
    }

    /// <inheritdoc />
    protected override async void OnAppearing()
    {
        base.OnAppearing();
        await _viewModel.RefreshAsync();
    }
}
