// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// The Audio process feature's page: lists the registered processors again each time it appears (a plug-in may have
/// registered one), and stops the microphone and the sound when it is left.
/// </summary>
public partial class AudioProcessPage : ContentPage
{
    private readonly AudioProcessViewModel _viewModel;

    /// <summary>A page bound to <paramref name="viewModel"/>.</summary>
    public AudioProcessPage(AudioProcessViewModel viewModel)
    {
        _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        InitializeComponent();
    }

    /// <inheritdoc />
    protected override void OnAppearing()
    {
        base.OnAppearing();
        _viewModel.RefreshProcessors();
    }

    /// <inheritdoc />
    protected override void OnDisappearing()
    {
        _viewModel.Stop();
        base.OnDisappearing();
    }
}
