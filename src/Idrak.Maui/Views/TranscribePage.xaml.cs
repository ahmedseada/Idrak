// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// The Transcribe feature's page: reads the speech models each time it appears (one may have been pulled meanwhile), opens
/// the Models page when there is none, and stops a recording when it is left (its last segment committed).
/// </summary>
public partial class TranscribePage : ContentPage
{
    private readonly TranscribeViewModel _viewModel;

    /// <summary>A page bound to <paramref name="viewModel"/>; <paramref name="services"/> open the Models page.</summary>
    public TranscribePage(TranscribeViewModel viewModel, IServiceProvider services)
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

    /// <inheritdoc />
    protected override void OnDisappearing()
    {
        _viewModel.Stop();
        base.OnDisappearing();
    }
}
