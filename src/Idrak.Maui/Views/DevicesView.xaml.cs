// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>Devices' view: reads the devices the first time it is shown; one card a row, or two on a wide window.</summary>
public partial class DevicesView : AdaptiveView
{
    private bool _read;

    /// <summary>A view bound to <paramref name="viewModel"/>; <paramref name="services"/> lay it out with the app (none: by its own width).</summary>
    public DevicesView(DevicesViewModel viewModel, IServiceProvider? services = null)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        InitializeComponent();
        Cards.ChildAdded += (_, e) => Size((View)e.Element);
        VirtualCards.ChildAdded += (_, e) => Size((View)e.Element);
    }

    /// <summary>The view model.</summary>
    public DevicesViewModel ViewModel { get; }

    /// <inheritdoc />
    public override async void Shown()
    {
        base.Shown();
        if (!_read)
        {
            _read = true;
            await ViewModel.RefreshAsync();
        }
    }

    /// <inheritdoc />
    protected override void Arrange(bool wide)
    {
        Page.SetDynamicResource(PaddingProperty, wide ? "IdrakWidePagePadding" : "IdrakPagePadding");
        foreach (var child in Cards.Children.Concat(VirtualCards.Children).OfType<View>())
        {
            Size(child);
        }
    }

    // One card a row on a narrow window, two on a wide one.
    private void Size(View card) => FlexLayout.SetBasis(card, new Microsoft.Maui.Layouts.FlexBasis(IsWide ? 0.5f : 1f, isRelative: true));
}
