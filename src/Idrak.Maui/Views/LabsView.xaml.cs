// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace Idrak.Maui.Views;

/// <summary>
/// All labs' view: the sheet over Home on a narrow window (made as the sheet: always the phone's layout), or the
/// rail's page on a wide one (a column a group, the filters). Lists the labs again each time it is shown.
/// </summary>
public partial class LabsView : AdaptiveView
{
    private readonly bool _sheet;
    private readonly DataTemplate _narrowGroup;
    private DataTemplate? _wideGroup;

    /// <summary>All labs over the app's services; <paramref name="asSheet"/> keeps the phone's layout (the sheet).</summary>
    public LabsView(IServiceProvider services, bool asSheet)
        : base(services)
    {
        ArgumentNullException.ThrowIfNull(services);
        _sheet = asSheet;
        var options = services.GetService<IdrakMauiOptions>() ?? new IdrakMauiOptions();
        var status = services.GetService<DeviceStatus>();
        ViewModel = new LabsViewModel(options, status, Dialogs.Navigator(services));
        BindingContext = ViewModel;
        InitializeComponent();
        _narrowGroup = BindableLayout.GetItemTemplate(Groups);
        if (status is not null)
        {
            status.PropertyChanged += (_, _) => ViewModel.UpdateStatus();
        }
    }

    /// <summary>All labs' view model.</summary>
    public LabsViewModel ViewModel { get; }

    /// <inheritdoc />
    public override async void Shown()
    {
        base.Shown();
        ViewModel.Refresh();
        if (ViewModel.Status is { } status)
        {
            await status.EnsureReadAsync();
            ViewModel.UpdateStatus();
        }
    }

    /// <inheritdoc />
    protected override void Arrange(bool wide)
    {
        wide &= !_sheet;
        FilterRow.IsVisible = wide;
        WideCounts.IsVisible = wide;
        NarrowCounts.IsVisible = !wide;
        TitleLabel.SetDynamicResource(StyleProperty, wide ? "IdrakHeroTitle" : "IdrakSheetTitle");
        Page.SetDynamicResource(PaddingProperty, wide ? "IdrakWidePagePadding" : _sheet ? "IdrakNoInset" : "IdrakPagePadding");
        Grid.SetRow(Search, wide ? 0 : 1);
        Grid.SetColumn(Search, wide ? 1 : 0);
        Grid.SetColumnSpan(Search, wide ? 1 : 2);
        Search.WidthRequest = wide ? 360 : -1;
        Groups.Direction = wide ? Microsoft.Maui.Layouts.FlexDirection.Row : Microsoft.Maui.Layouts.FlexDirection.Column;
        Groups.AlignItems = wide ? Microsoft.Maui.Layouts.FlexAlignItems.Start : Microsoft.Maui.Layouts.FlexAlignItems.Stretch;
        BindableLayout.SetItemTemplate(Groups, wide ? _wideGroup ??= WideGroup() : _narrowGroup);
    }

    // A group as a column of the wide page: its label and line, then its labs one under another with their summaries.
    private DataTemplate WideGroup() => new(() =>
    {
        var label = new Label();
        label.SetDynamicResource(StyleProperty, "IdrakSectionLabel");
        label.SetBinding(Label.TextProperty, static (LabGroup g) => g.Title);
        var line = new BoxView { HeightRequest = 1, VerticalOptions = LayoutOptions.Center };
        line.SetDynamicResource(BoxView.ColorProperty, "IdrakLine");
        var head = new Grid { ColumnDefinitions = [new ColumnDefinition(GridLength.Auto), new ColumnDefinition(GridLength.Star)] };
        head.SetDynamicResource(Grid.ColumnSpacingProperty, "IdrakSpacingChip");
        head.Add(label, 0);
        head.Add(line, 1);
        var labs = new VerticalStackLayout();
        BindableLayout.SetItemTemplate(labs, (DataTemplate)Resources["WideLab"]);
        labs.SetBinding(BindableLayout.ItemsSourceProperty, static (LabGroup g) => g);
        var column = new VerticalStackLayout { head, labs };
        column.SetDynamicResource(VerticalStackLayout.SpacingProperty, "IdrakSpacingTile");
        column.SetDynamicResource(PaddingProperty, "IdrakWideGroupPadding");
        FlexLayout.SetBasis(column, new Microsoft.Maui.Layouts.FlexBasis(0.25f, isRelative: true));
        return column;
    });
}
