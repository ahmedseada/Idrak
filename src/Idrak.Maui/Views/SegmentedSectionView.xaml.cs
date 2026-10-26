// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Maui.Abstractions;
using Idrak.Maui.Controls;
using Idrak.Maui.Navigation;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Views;

/// <summary>
/// A section of several labs: the segmented control over the section's enabled features (their titles and icons), each
/// lab's view made on first choice and kept, told when it is shown and hidden; the tune button opens the shown lab's
/// settings.
/// </summary>
public partial class SegmentedSectionView : AdaptiveView
{
    private readonly IServiceProvider _services;
    private readonly Dictionary<string, View> _views = new(StringComparer.OrdinalIgnoreCase);
    private bool _onScreen;

    /// <summary>The section <paramref name="section"/> over the app's services.</summary>
    public SegmentedSectionView(IServiceProvider services, IAppSection section)
        : base(services)
    {
        _services = services ?? throw new ArgumentNullException(nameof(services));
        ArgumentNullException.ThrowIfNull(section);
        var options = services.GetService(typeof(IdrakMauiOptions)) as IdrakMauiOptions ?? new IdrakMauiOptions();
        ViewModel = new SegmentedSectionViewModel(section, options);
        BindingContext = ViewModel;
        InitializeComponent();
        var template = (DataTemplate)Resources["Segment"];
        for (int i = 0; i < ViewModel.Segments.Count; i++)
        {
            SegmentGrid.ColumnDefinitions.Add(new ColumnDefinition(GridLength.Star));
            var segment = (View)template.CreateContent();
            segment.BindingContext = ViewModel.Segments[i];
            SegmentGrid.Add(segment, i);
        }

        ViewModel.SelectionChanged += (_, feature) => ShowView(feature);
        if (ViewModel.Selected is { } first)
        {
            ShowView(first.Name);
        }
    }

    /// <summary>The section's view model.</summary>
    public SegmentedSectionViewModel ViewModel { get; }

    /// <summary>The lab's view shown.</summary>
    public View? Current => Body.Content;

    /// <inheritdoc />
    public override void Shown()
    {
        base.Shown();
        _onScreen = true;
        (Body.Content as IHostedView)?.Shown();
    }

    /// <inheritdoc />
    public override void Hidden()
    {
        _onScreen = false;
        (Body.Content as IHostedView)?.Hidden();
        base.Hidden();
    }

    /// <inheritdoc />
    public override bool ShowFeature(string feature) => ViewModel.Select(feature);

    /// <inheritdoc />
    protected override void Arrange(bool wide)
    {
        // Wide: the heading, then the control beside it (360 wide), then the tune button; narrow: the control under them.
        Header.ColumnDefinitions = wide
            ? [new ColumnDefinition(GridLength.Star), new ColumnDefinition(new GridLength(360)), new ColumnDefinition(GridLength.Auto)]
            : [new ColumnDefinition(GridLength.Star), new ColumnDefinition(GridLength.Auto)];
        Place(SegmentBar, wide ? 0 : 1, wide ? 1 : 0, wide ? 1 : 2);
        Place(Tune, 0, wide ? 2 : 1);
        Header.SetDynamicResource(PaddingProperty, wide ? "IdrakWideSectionTopPadding" : "IdrakSectionTopPadding");
    }

    private void ShowView(string feature)
    {
        if (!_views.TryGetValue(feature, out var view))
        {
            try
            {
                view = FeatureViews.Create(_services, feature);
            }
            catch (Exception e) when (e is not OutOfMemoryException)
            {
                view = new FailedView(feature, e);
            }

            _views[feature] = view;
        }

        if (ReferenceEquals(Body.Content, view))
        {
            return;
        }

        if (_onScreen)
        {
            (Body.Content as IHostedView)?.Hidden();
        }

        Body.Content = view;
        Motion.FadeIn(view);
        if (_onScreen)
        {
            (view as IHostedView)?.Shown();
        }
    }

    private void OnTune(object? sender, TappedEventArgs e) => (Body.Content as ITunable)?.Tune();
}

/// <summary>A lab view whose settings a section's tune button opens.</summary>
internal interface ITunable
{
    /// <summary>Shows (or hides) the lab's settings.</summary>
    void Tune();
}
