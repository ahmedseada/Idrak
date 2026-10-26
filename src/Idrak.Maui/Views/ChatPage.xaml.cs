// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;
using Microsoft.Maui.Controls.PlatformConfiguration;
using Microsoft.Maui.Controls.PlatformConfiguration.AndroidSpecific;

namespace Idrak.Maui.Views;

/// <summary>
/// The Chat feature's page: reads the store's models each time it appears, keeps the composer above the keyboard (the
/// window resizes while the page shows), follows the streaming answer to its end, copies with the clipboard and opens the
/// Models page to pull a model.
/// </summary>
public partial class ChatPage : ContentPage
{
    private readonly ChatViewModel _viewModel;
    private WindowSoftInputModeAdjust? _keyboard;

    /// <summary>A page bound to <paramref name="viewModel"/>; <paramref name="services"/> open the Models page.</summary>
    public ChatPage(ChatViewModel viewModel, IServiceProvider services)
    {
        ArgumentNullException.ThrowIfNull(services);
        _viewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        viewModel.Copy = text => Clipboard.Default.SetTextAsync(text);
        viewModel.OpenModels = () => Navigation.PushAsync(MobileFeatures.Open(MobileFeatures.Models, services));
        InitializeComponent();
    }

    /// <inheritdoc />
    protected override async void OnAppearing()
    {
        base.OnAppearing();
        if (Microsoft.Maui.Controls.Application.Current is { } app)
        {
            var android = app.On<Android>();
            _keyboard ??= android.GetWindowSoftInputModeAdjust();
            android.UseWindowSoftInputModeAdjust(WindowSoftInputModeAdjust.Resize);
        }

        _viewModel.ScrollToEndRequested += OnScrollToEnd;
        await _viewModel.RefreshAsync();
    }

    /// <inheritdoc />
    protected override void OnDisappearing()
    {
        _viewModel.ScrollToEndRequested -= OnScrollToEnd;
        if (Microsoft.Maui.Controls.Application.Current is { } app && _keyboard is { } keyboard)
        {
            app.On<Android>().UseWindowSoftInputModeAdjust(keyboard);
        }

        base.OnDisappearing();
    }

    private void OnScrollToEnd(object? sender, EventArgs e)
    {
        if (_viewModel.Messages.Count > 0)
        {
            MessageList.ScrollTo(_viewModel.Messages.Count - 1, position: ScrollToPosition.End, animate: false);
        }
    }
}
