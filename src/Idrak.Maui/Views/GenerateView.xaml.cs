// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// Generate's view: the large preview and the controls; wide, the controls on the left and the preview on the right, the
/// grid three a row. The fonts are listed (folders only, on a worker thread) when it is first shown; the fonts page is made
/// once and kept.
/// </summary>
public partial class GenerateView : AdaptiveView, ITunable
{
    private readonly IServiceProvider? _services;
    private FontsPage? _fonts;

    /// <summary>A view bound to <paramref name="viewModel"/>; <paramref name="services"/> lay it out with the app and open the fonts page.</summary>
    public GenerateView(GenerateViewModel viewModel, IServiceProvider? services = null)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        _services = services;
        BindingContext = viewModel;
        InitializeComponent();
    }

    /// <summary>The view model.</summary>
    public GenerateViewModel ViewModel { get; }

    /// <summary>The narrow window's preview height (the wide one's fills the column).</summary>
    public static double NarrowPreviewHeight { get; set; } = 300;

    /// <inheritdoc />
    public override async void Shown()
    {
        base.Shown();
        if (ViewModel.AllFonts.Count == 0)
        {
            await ViewModel.ListFontsAsync();
        }
    }

    /// <inheritdoc />
    public void Tune()
    {
        ViewModel.ToggleSettingsCommand.Execute(null);
        if (ViewModel.ShowSettings && ControlsScroll.Handler is not null)
        {
            _ = ControlsScroll.ScrollToAsync(SettingsPanel, ScrollToPosition.Start, animated: true);
        }
    }

    /// <inheritdoc />
    protected override void Arrange(bool wide)
    {
        Root.ColumnDefinitions = wide ? [new ColumnDefinition(GridLength.Star), new ColumnDefinition(new GridLength(1.3, GridUnitType.Star))] : [new ColumnDefinition(GridLength.Star)];
        Root.RowDefinitions = wide ? [new RowDefinition(GridLength.Star)] : [new RowDefinition(GridLength.Auto), new RowDefinition(GridLength.Star)];
        Place(PreviewCard, 0, wide ? 1 : 0);
        Place(ControlsScroll, wide ? 0 : 1, 0);
        PreviewCard.HeightRequest = wide ? -1 : NarrowPreviewHeight;
        Tiles.Columns = wide ? 3 : 2;
    }

    private async void OnChooseFonts(object? sender, EventArgs e)
    {
        if (_fonts is null)
        {
            _fonts = new FontsPage(ViewModel);
            _fonts.SetValue(HandlerProperties.DisconnectPolicyProperty, HandlerDisconnectPolicy.Manual);   // its list kept when it is closed
        }

        if (_services is not null)
        {
            await Dialogs.Navigator(_services).PushAsync(_fonts);
        }
        else if (Dialogs.PageOf(this) is { } page)
        {
            await page.Navigation.PushAsync(_fonts);
        }
    }
}
