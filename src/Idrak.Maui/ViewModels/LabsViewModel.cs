// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Windows.Input;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Navigation;

namespace Idrak.Maui.ViewModels;

/// <summary>The filters of All labs.</summary>
public enum LabFilter
{
    /// <summary>Every lab.</summary>
    All,

    /// <summary>The built ones.</summary>
    Ready,

    /// <summary>The planned ones.</summary>
    Planned,

    /// <summary>Those that need the microphone.</summary>
    Microphone,

    /// <summary>Those that need the camera.</summary>
    Camera,
}

/// <summary>One lab on All labs: a feature (or the settings), its icon and tone, and whether it is built.</summary>
public sealed class LabEntry : ViewModel
{
    private string _detail;

    internal LabEntry(string name, string title, string tagline, string summary, string icon, string tone, bool planned, PhoneNeeds needs, string detail, ICommand open,
        IReadOnlyList<string> keywords)
    {
        Name = name;
        Title = title;
        Tagline = tagline;
        Summary = summary;
        Icon = icon;
        Tone = tone;
        IsPlanned = planned;
        Needs = needs;
        _detail = detail;
        OpenCommand = open;
        Keywords = keywords;
    }

    /// <summary>The feature's name ("settings" for the settings).</summary>
    public string Name { get; }

    /// <summary>Its title.</summary>
    public string Title { get; }

    /// <summary>A few words under the title (the phone's cards).</summary>
    public string Tagline { get; }

    /// <summary>What it does (the tablet's cards).</summary>
    public string Summary { get; }

    /// <summary>Its icon's name.</summary>
    public string Icon { get; }

    /// <summary>Its badge's tone.</summary>
    public string Tone { get; }

    /// <summary>Whether it is planned (the amber "● Planned" and a dashed edge).</summary>
    public bool IsPlanned { get; }

    /// <summary>Whether it is built.</summary>
    public bool IsReady => !IsPlanned;

    /// <summary>"● Ready" or "● Planned".</summary>
    public string StateText => IsPlanned ? "● Planned" : "● Ready";

    /// <summary>What it needs from the phone.</summary>
    public PhoneNeeds Needs { get; }

    /// <summary>A line under the summary (what it needs, or a live fact: "3 pulled · 2.1 GB").</summary>
    public string Detail
    {
        get => _detail;
        internal set => SetProperty(ref _detail, value);
    }

    /// <summary>Opens it.</summary>
    public ICommand OpenCommand { get; }

    /// <summary>More words a search finds it by (settings' titles, model ids).</summary>
    internal IReadOnlyList<string> Keywords { get; set; }

    /// <summary>Whether a search for <paramref name="query"/> finds it.</summary>
    public bool Matches(string query) =>
        query.Length == 0 || Title.Contains(query, StringComparison.CurrentCultureIgnoreCase) || Tagline.Contains(query, StringComparison.CurrentCultureIgnoreCase)
        || Summary.Contains(query, StringComparison.CurrentCultureIgnoreCase) || Name.Contains(query, StringComparison.OrdinalIgnoreCase)
        || Keywords.Any(k => k.Contains(query, StringComparison.CurrentCultureIgnoreCase));

    /// <summary>Whether <paramref name="filter"/> keeps it.</summary>
    public bool Passes(LabFilter filter) => filter switch
    {
        LabFilter.Ready => IsReady,
        LabFilter.Planned => IsPlanned,
        LabFilter.Microphone => (Needs & PhoneNeeds.Microphone) != 0,
        LabFilter.Camera => (Needs & PhoneNeeds.Camera) != 0,
        _ => true,
    };
}

/// <summary>A group of labs ("Voice"), its labs in menu order.</summary>
public sealed class LabGroup(FeatureGroup group, string title, IEnumerable<LabEntry> labs) : ObservableCollection<LabEntry>(labs)
{
    /// <summary>The group.</summary>
    public FeatureGroup Group { get; } = group;

    /// <summary>Its heading.</summary>
    public string Title { get; } = title;
}

/// <summary>A filter chip of All labs.</summary>
public sealed class FilterChip(LabFilter filter, string title, ICommand select) : ViewModel
{
    private bool _selected;

    /// <summary>The filter.</summary>
    public LabFilter Filter { get; } = filter;

    /// <summary>Its label.</summary>
    public string Title { get; } = title;

    /// <summary>Whether it is the filter in effect.</summary>
    public bool IsSelected
    {
        get => _selected;
        internal set => SetProperty(ref _selected, value);
    }

    /// <summary>Chooses it.</summary>
    public ICommand SelectCommand { get; } = select;
}

/// <summary>
/// All labs (the approved design's sheet on a phone, page on a tablet): every enabled feature of
/// <see cref="MobileFeatures"/> grouped by <see cref="IMobileFeature.Group"/> (Language, Voice, Vision, then this device's
/// own: models, devices, tests and the settings), planned ones marked; a search over titles, summaries, the settings and
/// the models in the store; filters by state and by what a lab needs.
/// </summary>
public sealed class LabsViewModel : ViewModel
{
    private static readonly Dictionary<string, string> LibraryTones = new(StringComparer.OrdinalIgnoreCase)
    {
        [MobileFeatures.Chat] = "Indigo", [MobileFeatures.Search] = "Sky", [MobileFeatures.Transcribe] = "Violet", [MobileFeatures.Speak] = "Green",
        [MobileFeatures.AudioProcess] = "Rose", [MobileFeatures.Ocr] = "Amber", [MobileFeatures.Image] = "Sky",
    };

    private readonly IdrakMauiOptions _options;
    private readonly IAppNavigator? _navigator;
    private readonly List<LabEntry> _all = [];
    private IMobileFeature[] _listed = [];
    private string _query = "";
    private LabFilter _filter = LabFilter.All;

    /// <summary>All labs over the features <paramref name="options"/> show; <paramref name="status"/> gives the device's and the store's lines.</summary>
    public LabsViewModel(IdrakMauiOptions options, DeviceStatus? status = null, IAppNavigator? navigator = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        Status = status;
        _navigator = navigator;
        Noun = DeviceWords.Noun;
        foreach (var (filter, title) in new[] { (LabFilter.All, "All"), (LabFilter.Ready, "Ready"), (LabFilter.Planned, "Planned"), (LabFilter.Microphone, "Needs the microphone"), (LabFilter.Camera, "Needs the camera") })
        {
            Filters.Add(new FilterChip(filter, title, new Command(() => Filter = filter)));
        }

        Refresh();
    }

    /// <summary>The device's status, when given.</summary>
    public DeviceStatus? Status { get; }

    /// <summary>"phone", "tablet" or "device".</summary>
    public string Noun { get; }

    /// <summary>The groups shown (empty groups left out).</summary>
    public ObservableCollection<LabGroup> Groups { get; } = [];

    /// <summary>The filter chips.</summary>
    public ObservableCollection<FilterChip> Filters { get; } = [];

    /// <summary>The search.</summary>
    public string Query
    {
        get => _query;
        set
        {
            if (SetProperty(ref _query, (value ?? "").Trim()))
            {
                Show();
            }
        }
    }

    /// <summary>The filter in effect.</summary>
    public LabFilter Filter
    {
        get => _filter;
        set
        {
            if (SetProperty(ref _filter, value))
            {
                Show();
            }
        }
    }

    /// <summary>"9 ready · 2 planned".</summary>
    public string CountsText => $"{_all.Count(l => l.IsReady)} ready · {_all.Count(l => l.IsPlanned)} planned";

    /// <summary>"9 ready · 2 planned · everything runs on this tablet".</summary>
    public string LongCountsText => $"{CountsText} · everything runs on this {Noun}";

    /// <summary>Whether a search or a filter finds nothing.</summary>
    public bool IsEmpty => Groups.Count == 0;

    /// <summary>Every lab listed, before the search and the filter.</summary>
    public IReadOnlyList<LabEntry> All => _all;

    /// <summary>The tone of the feature <paramref name="feature"/>'s badge: the library's own for its labs, else by its group.</summary>
    public static string ToneOf(IMobileFeature feature)
    {
        ArgumentNullException.ThrowIfNull(feature);
        return LibraryTones.TryGetValue(feature.Name, out var tone) ? tone : feature.Group switch
        {
            FeatureGroup.Language => "Indigo",
            FeatureGroup.Voice => "Violet",
            FeatureGroup.Vision => "Amber",
            _ => "Neutral",
        };
    }

    /// <summary>The group's heading ("This tablet" for the device's own).</summary>
    public string GroupTitle(FeatureGroup group) => group switch
    {
        FeatureGroup.Language => "Language",
        FeatureGroup.Voice => "Voice",
        FeatureGroup.Vision => "Vision",
        _ => $"This {Noun}",
    };

    /// <summary>
    /// Lists the enabled features again (the settings may have turned some on or off) and the device's lines; when the same
    /// features are enabled as before, only the device's lines are read again (the cards are kept, not made again).
    /// </summary>
    public void Refresh()
    {
        var enabled = MobileFeatures.Enabled(_options).ToArray();
        if (_all.Count > 0 && enabled.Length == _listed.Length
            && enabled.Zip(_listed).All(p => ReferenceEquals(p.First, p.Second) || p.First.Name == p.Second.Name && p.First.State == p.Second.State && p.First.Title == p.Second.Title))
        {
            UpdateStatus();
            return;
        }

        _listed = enabled;
        _all.Clear();
        foreach (var feature in enabled)
        {
            string name = feature.Name;
            _all.Add(new LabEntry(name, feature.Title, feature.Tagline.Length > 0 ? feature.Tagline : feature.Summary, feature.Summary,
                feature.Icon.Length > 0 ? feature.Icon : "grid", ToneOf(feature), feature.State == FeatureState.Planned, feature.Needs, Detail(feature),
                new Command(async () => await OpenAsync(name)), [.. feature.Packages]));
        }

        _all.Add(new LabEntry(AppSections.Settings, "Settings", "Device, theme, limits", "The device, the theme, storage, memory and thread limits, the token, the labs shown.",
            "gear", "Neutral", planned: false, PhoneNeeds.None, "", new Command(async () => await OpenAsync(AppSections.Settings)),
            [.. IdrakMauiSettings.Definitions.Select(s => s.Title)]));
        UpdateStatus();
        OnPropertyChanged(nameof(CountsText));
        OnPropertyChanged(nameof(LongCountsText));
        Show();
    }

    /// <summary>Takes the device's and the store's latest lines into the Devices and Models cards, and the store's ids into the search.</summary>
    public void UpdateStatus()
    {
        if (Status is null)
        {
            return;
        }

        foreach (var lab in _all)
        {
            if (string.Equals(lab.Name, MobileFeatures.Devices, StringComparison.OrdinalIgnoreCase))
            {
                lab.Detail = Status.DevicesText;
            }
            else if (string.Equals(lab.Name, MobileFeatures.Models, StringComparison.OrdinalIgnoreCase))
            {
                lab.Detail = Status.ModelsText;
                lab.Keywords = [.. Status.ModelIds];
            }
        }
    }

    /// <summary>Opens the lab <paramref name="name"/> (the settings for "settings").</summary>
    public async Task OpenAsync(string name)
    {
        if (_navigator is null)
        {
            return;
        }

        if (string.Equals(name, AppSections.Settings, StringComparison.OrdinalIgnoreCase))
        {
            await _navigator.OpenSettingsAsync().ConfigureAwait(true);
        }
        else
        {
            await _navigator.OpenFeatureAsync(name).ConfigureAwait(true);
        }
    }

    private static string Detail(IMobileFeature feature)
    {
        if (feature.State == FeatureState.Planned)
        {
            return feature.Coming;
        }

        var needs = feature.Needs;
        return (needs & PhoneNeeds.Microphone) != 0 ? "Needs the microphone"
            : (needs & PhoneNeeds.Camera) != 0 ? "Needs the camera"
            : (needs & PhoneNeeds.Model) != 0 ? "Needs a model"
            : "No model needed";
    }

    private void Show()
    {
        foreach (var chip in Filters)
        {
            chip.IsSelected = chip.Filter == _filter;
        }

        Groups.Clear();
        foreach (var group in new[] { FeatureGroup.Language, FeatureGroup.Voice, FeatureGroup.Vision, FeatureGroup.Device })
        {
            var labs = _all.Where(l => GroupOf(l) == group && l.Matches(_query) && l.Passes(_filter)).ToList();
            if (labs.Count > 0)
            {
                Groups.Add(new LabGroup(group, GroupTitle(group), labs));
            }
        }

        OnPropertyChanged(nameof(IsEmpty));
    }

    private static FeatureGroup GroupOf(LabEntry lab) => MobileFeatures.Find(lab.Name)?.Group ?? FeatureGroup.Device;
}
