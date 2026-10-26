// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>Models' view: reads the store each time it is shown, and asks on the page it is on before a model is removed.</summary>
public partial class ModelsView : AdaptiveView
{
    /// <summary>A view bound to <paramref name="viewModel"/>; <paramref name="services"/> lay it out with the app (none: by its own width).</summary>
    public ModelsView(ModelsViewModel viewModel, IServiceProvider? services = null)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        viewModel.Confirm = (title, message) => Dialogs.ConfirmAsync(this, title, message, "Remove", "Keep");
        InitializeComponent();
    }

    /// <summary>The view model.</summary>
    public ModelsViewModel ViewModel { get; }

    /// <inheritdoc />
    public override async void Shown()
    {
        base.Shown();
        await ViewModel.RefreshAsync();
    }

    /// <inheritdoc />
    protected override void Arrange(bool wide) => List.SetDynamicResource(MarginProperty, wide ? "IdrakWidePagePadding" : "IdrakPagePadding");
}
