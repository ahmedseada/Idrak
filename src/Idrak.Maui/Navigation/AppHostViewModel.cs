// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Windows.Input;
using Idrak.Maui.Abstractions;
using Idrak.Maui.ViewModels;

namespace Idrak.Maui.Navigation;

/// <summary>One entry of the bottom bar or the rail: a section, its label and icon, and whether it is the one shown.</summary>
public sealed class NavItem : ViewModel
{
    private bool _selected;

    internal NavItem(IAppSection section, ICommand select)
    {
        Section = section;
        SelectCommand = select;
    }

    /// <summary>The section.</summary>
    public IAppSection Section { get; }

    /// <summary>Its name.</summary>
    public string Name => Section.Name;

    /// <summary>Its label.</summary>
    public string Title => Section.Title;

    /// <summary>Its icon's name.</summary>
    public string Icon => Section.Icon;

    /// <summary>Whether it is the section shown.</summary>
    public bool IsSelected
    {
        get => _selected;
        internal set => SetProperty(ref _selected, value);
    }

    /// <summary>Shows it.</summary>
    public ICommand SelectCommand { get; }
}

/// <summary>
/// The host's navigation (no Shell, no Blazor): the visible sections (<see cref="AppSections.Visible"/>) as the bottom bar's
/// tabs on a narrow window and the rail's entries on a wide one (<see cref="AppLayout"/>), the one shown, and the All labs
/// sheet of a narrow window. The sections are listed again when the settings turn features on or off.
/// </summary>
public sealed class AppHostViewModel : ViewModel
{
    private readonly IdrakMauiOptions _options;
    private bool _sheetOpen;
    private string _selected = AppSections.Home;

    /// <summary>The navigation over the sections <paramref name="options"/> show, laid out by <paramref name="layout"/>.</summary>
    public AppHostViewModel(IdrakMauiOptions options, AppLayout layout)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Layout = layout ?? throw new ArgumentNullException(nameof(layout));
        Layout.PropertyChanged += (_, e) =>
        {
            if (e.PropertyName == nameof(AppLayout.IsWide))
            {
                OnPropertyChanged(nameof(IsWide));
                OnPropertyChanged(nameof(IsNarrow));
                if (IsWide)
                {
                    IsSheetOpen = false;                                       // the sheet is the narrow window's All labs
                }
            }
        };
        CloseSheetCommand = new Command(() => IsSheetOpen = false);
        Refresh();
    }

    /// <summary>The layout measured.</summary>
    public AppLayout Layout { get; }

    /// <summary>Whether the wide layout shows (the rail).</summary>
    public bool IsWide => Layout.IsWide;

    /// <summary>Whether the narrow layout shows (the bottom bar).</summary>
    public bool IsNarrow => Layout.IsNarrow;

    /// <summary>The tabs: the bottom bar's, and the top of the rail.</summary>
    public ObservableCollection<NavItem> Tabs { get; } = [];

    /// <summary>The rail's foot (All labs, Settings).</summary>
    public ObservableCollection<NavItem> Foot { get; } = [];

    /// <summary>The name of the section shown.</summary>
    public string Selected => _selected;

    /// <summary>Whether All labs shows as a sheet over the page (a narrow window).</summary>
    public bool IsSheetOpen
    {
        get => _sheetOpen;
        set => SetProperty(ref _sheetOpen, value);
    }

    /// <summary>Closes the sheet.</summary>
    public ICommand CloseSheetCommand { get; }

    /// <summary>Raised when an entry is tapped, with its section's name (the page shows it).</summary>
    public event EventHandler<string>? SelectRequested;

    /// <summary>Lists the visible sections again (keeping the one shown when it still shows, else Home).</summary>
    public void Refresh()
    {
        Tabs.Clear();
        Foot.Clear();
        foreach (var section in AppSections.Visible(_options))
        {
            var item = new NavItem(section, new Command(() => SelectRequested?.Invoke(this, section.Name)));
            (section.Placement == SectionPlacement.Foot ? Foot : Tabs).Add(item);
        }

        if (Find(_selected) is null)
        {
            _selected = Tabs.FirstOrDefault()?.Name ?? AppSections.Home;
        }

        Mark();
    }

    /// <summary>Whether the sections listed are those the options show now (in order): nothing to list again.</summary>
    public bool ShowsSame() => AppSections.Visible(_options).Select(s => s.Name).SequenceEqual(Tabs.Concat(Foot).Select(i => i.Name), StringComparer.OrdinalIgnoreCase);

    /// <summary>The entry of the section <paramref name="name"/>, or null.</summary>
    public NavItem? Find(string name) => Tabs.Concat(Foot).FirstOrDefault(i => string.Equals(i.Name, name, StringComparison.OrdinalIgnoreCase));

    /// <summary>Marks <paramref name="name"/> as the section shown; false when no visible section has that name.</summary>
    public bool Select(string name)
    {
        if (Find(name) is not { } item)
        {
            return false;
        }

        _selected = item.Name;
        Mark();
        OnPropertyChanged(nameof(Selected));
        return true;
    }

    private void Mark()
    {
        foreach (var item in Tabs.Concat(Foot))
        {
            item.IsSelected = string.Equals(item.Name, _selected, StringComparison.OrdinalIgnoreCase);
        }
    }
}
