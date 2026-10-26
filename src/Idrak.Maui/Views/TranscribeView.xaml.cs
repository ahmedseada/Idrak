// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// Transcribe's view: reads the speech models each time it is shown (one may have been pulled meanwhile), opens Models when
/// there is none, and stops a recording when it is hidden (its last segment committed). Wide: the transcript moves to the
/// right column, the measures and the cleaning flow show.
/// </summary>
public partial class TranscribeView : AdaptiveView, ITunable
{
    /// <summary>A view bound to <paramref name="viewModel"/>; <paramref name="services"/> open Models.</summary>
    public TranscribeView(TranscribeViewModel viewModel, IServiceProvider services)
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
    public TranscribeViewModel ViewModel { get; }

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
        Root.ColumnDefinitions = wide ? [new ColumnDefinition(new GridLength(1.1, GridUnitType.Star)), new ColumnDefinition(GridLength.Star)] : [new ColumnDefinition(GridLength.Star)];
        Right.IsVisible = wide;
        Stats.IsVisible = wide;
        Pipe.IsVisible = wide;
        StepChip.IsVisible = wide;
        Counts.IsVisible = wide;
        TranscriptFoot.IsVisible = wide;
        LiveFoot.IsVisible = !wide;
        Wave.HeightRequest = wide ? 196 : 120;
        TranscriptTitle.SetDynamicResource(StyleProperty, wide ? "IdrakHeading" : "IdrakSectionLabel");
        if (wide)
        {
            TranscriptHost.Content = null;
            Transcript.HeightRequest = -1;
            Right.Content = Transcript;
        }
        else
        {
            Right.Content = null;
            Transcript.HeightRequest = 240;
            TranscriptHost.Content = Transcript;
        }
    }

    private void OnTune(object? sender, TappedEventArgs e) => Tune();
}
