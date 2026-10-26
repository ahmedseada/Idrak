// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections;
using System.Windows.Input;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>A lab's settings panel: its entries (<see cref="LabSettingEntry"/> or <see cref="TranscribeSettingEntry"/>) and the keep button.</summary>
public partial class LabSettingsList : ContentView
{
    /// <summary>The entries.</summary>
    public static readonly BindableProperty ItemsSourceProperty =
        BindableProperty.Create(nameof(ItemsSource), typeof(IEnumerable), typeof(LabSettingsList), propertyChanged: (b, _, n) => BindableLayout.SetItemsSource(((LabSettingsList)b).Items, n as IEnumerable));

    /// <summary>What the keep button runs.</summary>
    public static readonly BindableProperty KeepCommandProperty =
        BindableProperty.Create(nameof(KeepCommand), typeof(ICommand), typeof(LabSettingsList), propertyChanged: (b, _, n) => ((LabSettingsList)b).Keep.Command = n as ICommand);

    /// <summary>An empty panel.</summary>
    public LabSettingsList()
    {
        InitializeComponent();
        BindableLayout.SetItemTemplateSelector(Items, new EntryTemplates((DataTemplate)Resources["LabSetting"], (DataTemplate)Resources["TranscribeSetting"]));
    }

    /// <summary>The entries.</summary>
    public IEnumerable? ItemsSource
    {
        get => (IEnumerable?)GetValue(ItemsSourceProperty);
        set => SetValue(ItemsSourceProperty, value);
    }

    /// <summary>What the keep button runs.</summary>
    public ICommand? KeepCommand
    {
        get => (ICommand?)GetValue(KeepCommandProperty);
        set => SetValue(KeepCommandProperty, value);
    }

    private sealed class EntryTemplates(DataTemplate lab, DataTemplate transcribe) : DataTemplateSelector
    {
        protected override DataTemplate OnSelectTemplate(object item, BindableObject container) => item is TranscribeSettingEntry ? transcribe : lab;
    }
}
