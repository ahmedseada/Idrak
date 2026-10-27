// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// The settings' view: the search, the groups with their controls (each setting made with its own control only), the labs
/// shown, storage and the app's restart (asked first); wider margins on a wide window.
/// </summary>
public partial class SettingsView : AdaptiveView
{
    /// <summary>A view bound to <paramref name="viewModel"/>.</summary>
    public SettingsView(SettingsViewModel viewModel, IServiceProvider? services = null)
        : base(services)
    {
        ViewModel = viewModel ?? throw new ArgumentNullException(nameof(viewModel));
        BindingContext = viewModel;
        viewModel.Confirm ??= (title, message) => Dialogs.ConfirmAsync(this, title, message, "Restart", "Not now");
        InitializeComponent();
    }

    /// <summary>The view model.</summary>
    public SettingsViewModel ViewModel { get; }

    /// <inheritdoc />
    protected override void Arrange(bool wide) => Page.SetDynamicResource(PaddingProperty, wide ? "IdrakWidePagePadding" : "IdrakPagePadding");
}

/// <summary>
/// The template of a setting by the control its value takes: <see cref="Segments"/> for a choice of a few,
/// <see cref="Picker"/> for many and for the device, <see cref="Slider"/> for a number with a range, <see cref="Field"/>
/// for the rest. Each template holds only its own control.
/// </summary>
public sealed class SettingTemplates : DataTemplateSelector
{
    /// <summary>A choice of a few.</summary>
    public DataTemplate? Segments { get; set; }

    /// <summary>A choice of many, or the device.</summary>
    public DataTemplate? Picker { get; set; }

    /// <summary>A number with a range.</summary>
    public DataTemplate? Slider { get; set; }

    /// <summary>Text, a secret, or a number without a range.</summary>
    public DataTemplate? Field { get; set; }

    /// <inheritdoc />
    protected override DataTemplate OnSelectTemplate(object item, BindableObject container) => item switch
    {
        SettingEntry { IsSegmented: true } => Segments,
        SettingEntry { IsPicker: true } => Picker,
        SettingEntry { IsSlider: true } => Slider,
        _ => Field,
    } ?? throw new InvalidOperationException("The settings' templates are not all set.");
}
