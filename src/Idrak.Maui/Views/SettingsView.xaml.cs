// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>The settings' view: the groups with their controls, the labs shown, storage; wider margins on a wide window.</summary>
public partial class SettingsView : AdaptiveView
{
    /// <summary>A view bound to <paramref name="viewModel"/>.</summary>
    public SettingsView(SettingsViewModel viewModel, IServiceProvider? services = null)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        InitializeComponent();
    }

    /// <summary>The view model.</summary>
    public SettingsViewModel ViewModel { get; }

    /// <inheritdoc />
    protected override void Arrange(bool wide) => Page.SetDynamicResource(PaddingProperty, wide ? "IdrakWidePagePadding" : "IdrakPagePadding");
}
