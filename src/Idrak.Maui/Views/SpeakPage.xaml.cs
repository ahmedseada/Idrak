// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// The Speak feature's page: reads the synthesis models each time it appears (one may have been pulled meanwhile), opens
/// the Models page when there is none, and stops the speech when it is left.
/// </summary>
public partial class SpeakPage : ContentPage
{
    private readonly SpeakViewModel _viewModel;

    /// <summary>A page bound to <paramref name="viewModel"/>; <paramref name="services"/> open the Models page.</summary>
    public SpeakPage(SpeakViewModel viewModel, IServiceProvider services)
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
