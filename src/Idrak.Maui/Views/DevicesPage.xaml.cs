// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>The Devices feature's page: reads the devices when it first appears.</summary>
public partial class DevicesPage : ContentPage
{
    private readonly DevicesViewModel _viewModel;
    private bool _read;

    /// <summary>A page bound to <paramref name="viewModel"/>.</summary>
    public DevicesPage(DevicesViewModel viewModel)
    {
        _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        InitializeComponent();
    }

    /// <inheritdoc />
    protected override async void OnAppearing()
    {
        base.OnAppearing();
        if (!_read)
        {
            _read = true;
            await _viewModel.RefreshAsync();
        }
    }
}
