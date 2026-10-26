// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Windows.Input;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.ViewModels;

/// <summary>One segment of a section's control: a feature's title and icon, and whether it is the one shown.</summary>
public sealed class SegmentItem(string name, string title, string icon, ICommand select) : ViewModel
{
    private bool _selected;

    /// <summary>The feature's name.</summary>
    public string Name { get; } = name;

    /// <summary>Its title ("Transcribe").</summary>
    public string Title { get; } = title;

    /// <summary>Its icon's name.</summary>
    public string Icon { get; } = icon;

    /// <summary>Whether it is shown.</summary>
    public bool IsSelected
    {
        get => _selected;
        internal set => SetProperty(ref _selected, value);
    }

    /// <summary>Shows it.</summary>
    public ICommand SelectCommand { get; } = select;
}

/// <summary>A section of several features (Voice, Vision): its heading and the segments of its enabled, registered features.</summary>
public sealed class SegmentedSectionViewModel : ViewModel
{
    /// <summary>The segments of <paramref name="section"/>'s features that <paramref name="options"/> show.</summary>
    public SegmentedSectionViewModel(IAppSection section, IdrakMauiOptions options)
    {
        ArgumentNullException.ThrowIfNull(section);
        ArgumentNullException.ThrowIfNull(options);
        Title = section.Title;
        foreach (string name in section.Features)
        {
            if (options.IsFeatureEnabled(name) && MobileFeatures.Find(name) is { } feature)
            {
                Segments.Add(new SegmentItem(feature.Name, feature.Title, feature.Icon.Length > 0 ? feature.Icon : "grid", new Command(() => Select(feature.Name))));
            }
        }

        if (Segments.Count > 0)
        {
            Segments[0].IsSelected = true;
        }
    }

    /// <summary>The heading ("Voice").</summary>
    public string Title { get; }

    /// <summary>The segments.</summary>
    public ObservableCollection<SegmentItem> Segments { get; } = [];

    /// <summary>The segment shown, or null when the section has none.</summary>
    public SegmentItem? Selected => Segments.FirstOrDefault(s => s.IsSelected);

    /// <summary>Raised when another segment is chosen, with its feature's name.</summary>
    public event EventHandler<string>? SelectionChanged;

    /// <summary>Shows the feature <paramref name="feature"/>'s segment; false when the section has none of that name.</summary>
    public bool Select(string feature)
    {
        if (Segments.FirstOrDefault(s => string.Equals(s.Name, feature, StringComparison.OrdinalIgnoreCase)) is not { } segment)
        {
            return false;
        }

        if (!segment.IsSelected)
        {
            foreach (var s in Segments)
            {
                s.IsSelected = ReferenceEquals(s, segment);
            }

            OnPropertyChanged(nameof(Selected));
            SelectionChanged?.Invoke(this, segment.Name);
        }

        return true;
    }
}
