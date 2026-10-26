// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// Studio's view (the audio process lab): lists the registered processors each time it is shown and stops a recording,
/// a playback or monitoring when it is hidden. Wide: the result beside the sound and the pipeline.
/// </summary>
public partial class StudioView : AdaptiveView, ITunable
{
    /// <summary>A view bound to <paramref name="viewModel"/>; <paramref name="services"/> lay it out with the app (none: by its own width).</summary>
    public StudioView(AudioProcessViewModel viewModel, IServiceProvider? services = null)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        InitializeComponent();
    }

    /// <summary>The view model.</summary>
    public AudioProcessViewModel ViewModel { get; }

    /// <inheritdoc />
    public override void Shown()
    {
        base.Shown();
        ViewModel.RefreshProcessors();
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
}
