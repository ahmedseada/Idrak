// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>Image's view: the photos before and after; wide: the steps and the times beside them.</summary>
public partial class ImageView : AdaptiveView, ITunable
{
    /// <summary>A view bound to <paramref name="viewModel"/>; <paramref name="services"/> lay it out with the app (none: by its own width).</summary>
    public ImageView(ImageViewModel viewModel, IServiceProvider? services = null)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        InitializeComponent();
    }

    /// <summary>The view model.</summary>
    public ImageViewModel ViewModel { get; }

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
        Root.ColumnDefinitions = wide ? [new ColumnDefinition(new GridLength(1.2, GridUnitType.Star)), new ColumnDefinition(GridLength.Star)] : [new ColumnDefinition(GridLength.Star)];
        Place(Right, wide ? 0 : 1, wide ? 1 : 0);
    }
}
