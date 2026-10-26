// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// Speak's view: reads the synthesis models each time it is shown, opens Models when there is none, and stops the speech
/// when it is hidden. Wide: the measures and the pieces beside the text.
/// </summary>
public partial class SpeakView : AdaptiveView, ITunable
{
    /// <summary>A view bound to <paramref name="viewModel"/>; <paramref name="services"/> open Models.</summary>
    public SpeakView(SpeakViewModel viewModel, IServiceProvider services)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        ArgumentNullException.ThrowIfNull(services);
        BindingContext = viewModel;
        var navigator = Dialogs.Navigator(services);
        viewModel.OpenModels = () => navigator.OpenFeatureAsync(MobileFeatures.Models);
        InitializeComponent();
    }

    /// <summary>The view model.</summary>
    public SpeakViewModel ViewModel { get; }

    /// <inheritdoc />
    public override async void Shown()
    {
        base.Shown();
        await ViewModel.RefreshAsync();
    }

    /// <inheritdoc />
    public override void Hidden()
    {
        ViewModel.Stop();
        base.Hidden();
    }

    /// <inheritdoc />
    public void Tune()
    {
        ViewModel.ToggleSettingsCommand.Execute(null);
        if (ViewModel.ShowSettings && Scroll.Handler is not null)
        {
            _ = Scroll.ScrollToAsync(SettingsPanel, ScrollToPosition.Start, animated: true);
        }
    }

    /// <inheritdoc />
    protected override void Arrange(bool wide)
    {
        Root.ColumnDefinitions = wide ? [new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Star)] : [new ColumnDefinition(GridLength.Star)];
        Place(Right, wide ? 0 : 1, wide ? 1 : 0);
    }

    private void OnTune(object? sender, TappedEventArgs e) => Tune();
}
