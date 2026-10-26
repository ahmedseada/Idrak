// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.Navigation;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// Home's view: reads the device once and the chat's last exchange each time it is shown; moves the carousel every
/// <see cref="HomeViewModel.SlideInterval"/> while shown (never off screen); two columns on a wide window.
/// </summary>
public partial class HomeView : AdaptiveView
{
    private IDispatcherTimer? _timer;

    /// <summary>Home over the app's services.</summary>
    public HomeView(IServiceProvider services)
        : base(services)
    {
        ArgumentNullException.ThrowIfNull(services);
        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        var status = services.GetService<DeviceStatus>()
            ?? new DeviceStatus(services.GetService<IDeviceInformation>() ?? new DeviceInformation(), services.GetService<IModelStore>());
        ViewModel = new HomeViewModel(options, status, services.GetService<ChatViewModel>(), Dialogs.Navigator(services));
        BindingContext = ViewModel;
        InitializeComponent();
        foreach (double level in (double[])[0.3, 0.6, 0.95, 0.45, 0.8, 0.35])
        {
            TileWave.Push(level);
        }
    }

    /// <summary>Home's view model.</summary>
    public HomeViewModel ViewModel { get; }

    /// <inheritdoc />
    public override async void Shown()
    {
        base.Shown();
        StartCarousel();
        await ViewModel.RefreshAsync();
    }

    /// <inheritdoc />
    public override void Hidden()
    {
        _timer?.Stop();
        base.Hidden();
    }

    /// <inheritdoc />
    protected override void Arrange(bool wide)
    {
        Columns.ColumnDefinitions = wide ? [new ColumnDefinition(GridLength.Star), new ColumnDefinition(new GridLength(340))] : [new ColumnDefinition(GridLength.Star)];
        Right.IsVisible = wide;
        MeasuredTiles.IsVisible = !wide;
        LogoTile.IsVisible = !wide;
        Brand.SetDynamicResource(StyleProperty, wide ? "IdrakHeroTitle" : "IdrakSheetTitle");
        Page.SetDynamicResource(PaddingProperty, wide ? "IdrakWidePagePadding" : "IdrakPagePadding");
        Carousel.HeightRequest = wide ? 286 : 236;
        SpeakLine.IsVisible = wide;
        Bento.ColumnDefinitions = wide
            ? [new ColumnDefinition(new GridLength(1.6, GridUnitType.Star)), new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Star)]
            : [new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Star)];
        Place(ChatTile, 0, 0, wide ? 1 : 3);
        Place(TranscribeTile, wide ? 0 : 1, wide ? 1 : 0);
        Place(OcrTile, wide ? 0 : 1, wide ? 2 : 1);
        Place(SpeakTile, wide ? 0 : 1, wide ? 3 : 2);
        ChatTileGrid.ColumnDefinitions = wide ? [new ColumnDefinition(GridLength.Star)] : [new ColumnDefinition(GridLength.Auto), new ColumnDefinition(GridLength.Star)];
        ChatTileGrid.RowDefinitions = wide ? [new RowDefinition(GridLength.Auto), new RowDefinition(GridLength.Star)] : [new RowDefinition(GridLength.Auto)];
        Place(ChatPreview, wide ? 1 : 0, wide ? 0 : 1);
        ChatTileText.WidthRequest = wide ? -1 : 150;
        foreach (var tile in (Border[])[ChatTile, TranscribeTile, OcrTile, SpeakTile])
        {
            tile.HeightRequest = wide ? 236 : tile == ChatTile ? 118 : 112;
        }
    }

    // The carousel moves by itself while shown (and stays when the interval is zero).
    private void StartCarousel()
    {
        var interval = ViewModel.SlideInterval;
        if (interval <= TimeSpan.Zero || Dispatcher is null)
        {
            return;
        }

        if (_timer is null)
        {
            _timer = Dispatcher.CreateTimer();
            _timer.Tick += (_, _) =>
            {
                if (Motion.CanAnimate(this))
                {
                    ViewModel.NextSlide();
                }
            };
        }

        _timer.Interval = interval;
        _timer.Start();
    }
}
