// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// The OCR feature's page: reads the store's readers each time it appears (one may have been pulled meanwhile) and opens
/// the Models page when there is none. A read goes on while another page is open (the view model is the app's one).
/// </summary>
public partial class OcrPage : ContentPage
{
    private readonly OcrViewModel _viewModel;

    /// <summary>A page bound to <paramref name="viewModel"/>; <paramref name="services"/> open the Models page.</summary>
    public OcrPage(OcrViewModel viewModel, IServiceProvider services)
    {
        _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        ArgumentNullException.ThrowIfNull(services);
        BindingContext = viewModel;
        viewModel.OpenModels = () => Navigation.PushAsync(MobileFeatures.Open(MobileFeatures.Models, services));
        InitializeComponent();
    }

    /// <inheritdoc />
    protected override async void OnAppearing()
    {
        base.OnAppearing();
        await _viewModel.RefreshAsync();
    }
}
