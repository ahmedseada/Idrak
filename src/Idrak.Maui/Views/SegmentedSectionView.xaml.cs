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
public partial class SegmentedSectionView : AdaptiveView, IPrebuilt
{
    private readonly IServiceProvider _services;
    private readonly IdrakMauiOptions _options;
    private readonly Dictionary<string, View> _views = new(StringComparer.OrdinalIgnoreCase);
    private View? _current;
    private bool _onScreen;

    /// <summary>The section <paramref name="section"/> over the app's services.</summary>
    public SegmentedSectionView(IServiceProvider services, IAppSection section)
        : base(services)
    {
        _services = services ?? throw new ArgumentNullException(nameof(services));
        ArgumentNullException.ThrowIfNull(section);
        var options = services.GetService(typeof(IdrakMauiOptions)) as IdrakMauiOptions ?? new IdrakMauiOptions();
        _options = options;
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
    public View? Current => _current;

    /// <inheritdoc />
    IEnumerable<Action> IPrebuilt.Parts() =>
        ViewModel.Segments.Where(s => !_views.ContainsKey(s.Name)).Select(s => (Action)(() => ViewOf(s.Name))).ToList();

    /// <inheritdoc />
    public override void Shown()
    {
        base.Shown();
        _onScreen = true;
        (_current as IHostedView)?.Shown();
    }

    /// <inheritdoc />
    public override void Hidden()
    {
        _onScreen = false;
        (_current as IHostedView)?.Hidden();
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
        var view = ViewOf(feature);
        if (ReferenceEquals(_current, view))
        {
            return;
        }

        var previous = _current;
        _current = view;
        view.IsVisible = true;                                                   // the switch first; the lab told after it has drawn
        if (previous is not null)
        {
            previous.IsVisible = false;
        }

        if (_options.SectionFade > TimeSpan.Zero)
        {
            Motion.FadeIn(view, (uint)Math.Min(uint.MaxValue, _options.SectionFade.TotalMilliseconds));
        }

        if (_onScreen)
        {
            (previous as IHostedView)?.Hidden();
            Dispatcher.Dispatch(() =>
            {
                if (_onScreen && ReferenceEquals(_current, view))
                {
                    (view as IHostedView)?.Shown();
                }
            });
        }
    }

    // The lab's view, made once and kept here, hidden until chosen.
    private View ViewOf(string feature)
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

            view.IsVisible = false;
            _views[feature] = view;
            Body.Children.Add(view);
        }

        return view;
    }

    private void OnTune(object? sender, TappedEventArgs e) => (_current as ITunable)?.Tune();
}

/// <summary>A lab view whose settings a section's tune button opens.</summary>
internal interface ITunable
{
    /// <summary>Shows (or hides) the lab's settings.</summary>
    void Tune();
}
