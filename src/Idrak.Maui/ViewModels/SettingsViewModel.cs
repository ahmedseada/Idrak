// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Windows.Input;
using Idrak.Abstraction.Devices;
using Idrak.Maui.Abstractions;
using Idrak.Maui.Theme;

namespace Idrak.Maui.ViewModels;

/// <summary>One setting on the settings page: its title, help and value, edited as its kind says.</summary>
public sealed class SettingEntry : ViewModel
{
    private string _text;
    private string _error = "";

    internal SettingEntry(IdrakSetting setting, string text, IReadOnlyList<string> choices)
    {
        Setting = setting;
        _text = text;
        Choices = choices;
        ResetCommand = new Command(() => Text = "");
        foreach (string choice in IsSegmented ? choices : [])
        {
            Segments.Add(new ChoiceChip(choice, new Command(() => SelectedChoice = choice)));
        }

        MarkSegments();
    }

    /// <summary>The group it shows in.</summary>
    public string Group => Setting.Group;

    /// <summary>Whether it is a choice of a few (four or fewer): a row of segments.</summary>
    public bool IsSegmented => Setting.Kind == IdrakSettingKind.Choice && Choices.Count is > 0 and <= 4;

    /// <summary>Whether it is a choice of many, or a device: a picker.</summary>
    public bool IsPicker => IsChoice && !IsSegmented;

    /// <summary>Whether it is a number with a range: a slider (and its value, and a reset to the default).</summary>
    public bool IsSlider => Setting.Kind == IdrakSettingKind.Number && Setting.Range is not null;

    /// <summary>Whether it is typed: text, a secret, or a number without a range.</summary>
    public bool IsField => IsEntry && !IsSlider;

    /// <summary>The slider's least value.</summary>
    public double Minimum => Setting.Range?.Minimum ?? 0;

    /// <summary>The slider's most value.</summary>
    public double Maximum => Setting.Range?.Maximum ?? 1;

    /// <summary>The value as a number on the slider (the default's, or the least, when empty); setting it rounds to the step.</summary>
    public double Number
    {
        get => double.TryParse(Text.Length > 0 ? Text : Setting.Placeholder, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out double v)
            ? Math.Clamp(v, Minimum, Maximum) : Minimum;
        set
        {
            double step = Setting.Range?.Step ?? 0;
            double v = step > 0 ? Math.Round(Math.Round(value / step) * step, 6) : value;
            string text = v.ToString("0.###", System.Globalization.CultureInfo.InvariantCulture);
            if (text != (Text.Length > 0 ? Text : null))
            {
                Text = text;
            }
        }
    }

    /// <summary>The value as the slider's label shows it: the number, or "default (…)" when empty.</summary>
    public string NumberText => Text.Length > 0 ? Text : $"default ({Setting.Placeholder})";

    /// <summary>The segments of a choice of a few.</summary>
    public ObservableCollection<ChoiceChip> Segments { get; } = [];

    /// <summary>Empties the value: the library's own.</summary>
    public ICommand ResetCommand { get; }

    private void MarkSegments()
    {
        string? selected = SelectedChoice;
        foreach (var segment in Segments)
        {
            segment.IsSelected = segment.Text == selected;
        }
    }

    /// <summary>The setting.</summary>
    public IdrakSetting Setting { get; }

    /// <summary>Its title.</summary>
    public string Title => Setting.Title + (Setting.NeedsRestart ? " (next start)" : "");

    /// <summary>Its help.</summary>
    public string Summary => Setting.Summary;

    /// <summary>What an empty value means.</summary>
    public string Placeholder => Setting.Placeholder;

    /// <summary>Edited as text (a field).</summary>
    public bool IsEntry => Setting.Kind is IdrakSettingKind.Text or IdrakSettingKind.Number or IdrakSettingKind.Secret;

    /// <summary>Edited as a choice (a picker).</summary>
    public bool IsChoice => !IsEntry;

    /// <summary>Whether the field hides what is typed (the token).</summary>
    public bool IsPassword => Setting.Kind == IdrakSettingKind.Secret;

    /// <summary>The keyboard for the field.</summary>
    public Keyboard Keyboard => Setting.Kind == IdrakSettingKind.Number ? Keyboard.Numeric : Keyboard.Default;

    /// <summary>The values to choose from (a choice).</summary>
    public IReadOnlyList<string> Choices { get; }

    /// <summary>The value as text ("" for the library's own).</summary>
    public string Text
    {
        get => _text;
        set
        {
            if (SetProperty(ref _text, value ?? ""))
            {
                OnPropertyChanged(nameof(SelectedChoice));
                OnPropertyChanged(nameof(Number));
                OnPropertyChanged(nameof(NumberText));
                MarkSegments();
            }
        }
    }

    /// <summary>The value as a choice (the first choice, "default" or "auto", standing for "").</summary>
    public string? SelectedChoice
    {
        get => Text.Length == 0 && Choices.Count > 0 ? Choices[0] : Text;
        set => Text = value is null || Choices.Count > 0 && value == Choices[0] && Setting.Placeholder == value ? "" : value;
    }

    /// <summary>Why the value was refused, or "".</summary>
    public string Error
    {
        get => _error;
        set
        {
            if (SetProperty(ref _error, value ?? ""))
            {
                OnPropertyChanged(nameof(HasError));
            }
        }
    }

    /// <summary>Whether <see cref="Error"/> has something to show.</summary>
    public bool HasError => Error.Length > 0;
}

/// <summary>A feature shown in the menu or not.</summary>
public sealed class FeatureToggle(IMobileFeature feature, bool enabled) : ViewModel
{
    private bool _enabled = enabled;

    /// <summary>The feature's name.</summary>
    public string Name => feature.Name;

    /// <summary>Its title.</summary>
    public string Title => feature.Title;

    /// <summary>What it does.</summary>
    public string Summary => feature.Summary;

    /// <summary>Whether it shows.</summary>
    public bool IsEnabled
    {
        get => _enabled;
        set => SetProperty(ref _enabled, value);
    }
}

/// <summary>
/// The settings page over every option of <see cref="IdrakMauiOptions"/> (<see cref="IdrakMauiSettings.Definitions"/>) and
/// the features shown. Saving checks every value, keeps them (<see cref="IdrakMauiSettings.SaveAsync"/>), applies at once
/// what can be (the limits, the device, the token, the theme, the progress interval, the features) and says which changes
/// wait for the next start.
/// </summary>
public sealed class SettingsViewModel : ViewModel
{
    private readonly IdrakMauiOptions _options;
    private readonly IdrakMauiSettings _settings;
    private readonly IdrakStartupReport _startup;
    private readonly ProgressCenter? _progress;
    private string _message = "";

    /// <summary>A settings page over the app's <paramref name="options"/>.</summary>
    public SettingsViewModel(IdrakMauiOptions options, IdrakMauiSettings settings, IdrakStartupReport startup, ProgressCenter? progress = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        _startup = startup ?? throw new ArgumentNullException(nameof(startup));
        _progress = progress;
        SaveCommand = new Command(async () => await SaveAsync());
        ResetCommand = new Command(async () => await ResetAsync());
        Load(options);
        Message = startup.Problems.Count > 0 ? string.Join("\n", startup.Problems) : "";
    }

    /// <summary>The page's title.</summary>
    public string Title => "Settings";

    /// <summary>Every setting, in order.</summary>
    public ObservableCollection<SettingEntry> Entries { get; } = [];

    /// <summary>Every registered feature, with whether it shows.</summary>
    public ObservableCollection<FeatureToggle> Features { get; } = [];

    /// <summary>The settings by group, in the page's order (General first, the limits last).</summary>
    public ObservableCollection<SettingGroup> Groups { get; } = [];

    /// <summary>The order of the groups on the page.</summary>
    public static IReadOnlyList<string> GroupOrder => Order;

    private static readonly string[] Order = ["General", "Layout and motion", "Chat", "Models in memory", "Storage", "Limits"];

    /// <summary>The app's storage, where the default cache folder is.</summary>
    public string AppStorage => _startup.AppStorage;

    /// <summary>The cache folder in effect.</summary>
    public string CacheFolder => ComputeResources.CacheFolder;

    /// <summary>What the last save said (what waits for the next start, what was refused), or "".</summary>
    public string Message
    {
        get => _message;
        private set
        {
            if (SetProperty(ref _message, value))
            {
                OnPropertyChanged(nameof(HasMessage));
            }
        }
    }

    /// <summary>Whether <see cref="Message"/> has something to show.</summary>
    public bool HasMessage => Message.Length > 0;

    /// <summary>Checks, keeps and applies the values.</summary>
    public ICommand SaveCommand { get; }

    /// <summary>Forgets every saved value: the app's defaults at the next start.</summary>
    public ICommand ResetCommand { get; }

    /// <summary>Checks the values, keeps them and applies what takes effect at once; false when a value was refused.</summary>
    public async Task<bool> SaveAsync()
    {
        var next = _options.Clone();
        bool valid = true;
        foreach (var entry in Entries)
        {
            try
            {
                entry.Setting.Set(next, entry.Text);
                entry.Error = "";
            }
            catch (FormatException e)
            {
                entry.Error = e.Message;
                valid = false;
            }
        }

        if (!valid)
        {
            Message = "Nothing was saved: a value above is not one the setting takes.";
            return false;
        }

        next.DisabledFeatures = new HashSet<string>(Features.Where(f => !f.IsEnabled).Select(f => f.Name), StringComparer.OrdinalIgnoreCase);
        var restart = IdrakMauiSettings.NeedRestart(_options, next);
        var applied = IdrakMauiStartup.Apply(next, _startup.AppStorage, previous: _options);
        var previous = _options.Clone();
        _options.CopyFrom(next);
        if (restart.Count > 0)
        {
            // What waits for the next start stays as it runs now, so the next save compares with what is in effect.
            foreach (var setting in IdrakMauiSettings.Definitions.Where(s => s.NeedsRestart))
            {
                setting.Set(_options, setting.Get(previous));
            }
        }

        await _settings.SaveAsync(next).ConfigureAwait(true);
        if (_progress is not null)
        {
            _progress.Interval = next.ProgressInterval;
        }

        if (Application.Current is { } app)
        {
            IdrakThemes.Apply(app, next.Theme);
        }

        OnPropertyChanged(nameof(CacheFolder));
        Message = string.Join("\n", new[]
        {
            "Saved.",
            restart.Count > 0 ? $"Restart the app for: {string.Join(", ", restart)}." : null,
            applied.Problems.Count > 0 ? string.Join("\n", applied.Problems) : null,
        }.Where(l => l is not null));
        return true;
    }

    /// <summary>Forgets every saved value.</summary>
    public async Task ResetAsync()
    {
        await _settings.ClearAsync().ConfigureAwait(true);
        Message = "The saved settings are forgotten: the app's defaults apply at the next start.";
    }

    private void Load(IdrakMauiOptions options)
    {
        Entries.Clear();
        foreach (var setting in IdrakMauiSettings.Definitions)
        {
            string text = setting.Get(options);
            IReadOnlyList<string> choices = setting.Kind switch
            {
                IdrakSettingKind.Device => DeviceChoices(text),
                IdrakSettingKind.Choice => setting.Choices,
                _ => [],
            };
            Entries.Add(new SettingEntry(setting, setting.Kind == IdrakSettingKind.Device && text == "auto" ? "" : text, choices));
        }

        Groups.Clear();
        foreach (var group in Entries.GroupBy(e => e.Group).OrderBy(g => Array.IndexOf(Order, g.Key) is var i and >= 0 ? i : Order.Length))
        {
            Groups.Add(new SettingGroup(group.Key, group));
        }

        Features.Clear();
        foreach (var feature in MobileFeatures.All)
        {
            Features.Add(new FeatureToggle(feature, options.IsFeatureEnabled(feature.Name)));
        }
    }

    // "auto", "cpu", then every device of every backend (counted, not started), and the current one if none of these.
    private static IReadOnlyList<string> DeviceChoices(string current)
    {
        var names = new List<string> { "auto", "cpu" };
        try
        {
            foreach (var provider in DeviceProviders.All)
            {
                for (int i = 0; i < provider.Count; i++)
                {
                    names.Add($"{provider.Name}:{i}");
                }
            }
        }
        catch (Exception e) when (e is InvalidOperationException or DllNotFoundException or EntryPointNotFoundException)
        {
            // A backend that cannot count its devices lists none.
        }

        if (current.Length > 0 && !names.Contains(current, StringComparer.OrdinalIgnoreCase))
        {
            names.Add(current);
        }

        return names;
    }
}

/// <summary>One choice of a setting shown as segments ("system", "light", "dark").</summary>
public sealed class ChoiceChip(string text, ICommand select) : ViewModel
{
    private bool _selected;

    /// <summary>The choice, as the setting takes it ("automatic").</summary>
    public string Text { get; } = text;

    /// <summary>The choice as its segment shows it, capitalised ("Automatic").</summary>
    public string Label => Text.Length == 0 ? Text : char.ToUpperInvariant(Text[0]) + Text[1..];

    /// <summary>Whether it is the value.</summary>
    public bool IsSelected
    {
        get => _selected;
        internal set => SetProperty(ref _selected, value);
    }

    /// <summary>Chooses it.</summary>
    public ICommand SelectCommand { get; } = select;
}

/// <summary>The settings of one group ("Chat").</summary>
public sealed class SettingGroup(string title, IEnumerable<SettingEntry> entries) : ObservableCollection<SettingEntry>(entries)
{
    /// <summary>Its heading.</summary>
    public string Title { get; } = title;
}
