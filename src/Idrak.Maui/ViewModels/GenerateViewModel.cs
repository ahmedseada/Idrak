// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Windows.Input;
using Idrak.Imaging;
using Idrak.Maui.Abstractions;
using Idrak.Text;

namespace Idrak.Maui.ViewModels;

/// <summary>A font the Generate lab can set a page in: a file of the phone's or one the user added, chosen or not.</summary>
public sealed class FontChoice : ViewModel
{
    private bool _chosen;

    internal FontChoice(string path, long bytes, bool added, bool chosen, Action changed)
    {
        Path = path;
        Bytes = bytes;
        IsAdded = added;
        _chosen = chosen;
        Changed = changed;
    }

    /// <summary>The font file.</summary>
    public string Path { get; }

    /// <summary>Its name (the file's, without the extension).</summary>
    public string Name => System.IO.Path.GetFileNameWithoutExtension(Path);

    /// <summary>Its size.</summary>
    public long Bytes { get; }

    /// <summary>Whether the user added it (else it is the phone's).</summary>
    public bool IsAdded { get; }

    /// <summary>"added · 312 KB" or "phone · 1.2 MB".</summary>
    public string Detail => $"{(IsAdded ? "added" : "phone")} · {ProgressText.Bytes(Bytes)}";

    /// <summary>Whether pages are set in it.</summary>
    public bool IsChosen
    {
        get => _chosen;
        set
        {
            if (SetProperty(ref _chosen, value))
            {
                Changed();
            }
        }
    }

    private Action Changed { get; }
}

/// <summary>A scan effect (a <see cref="ScanProfiles"/> name) as a chip: tapped, it joins the combination or leaves it.</summary>
public sealed class EffectChip(string name, string summary, ICommand toggle) : ViewModel
{
    private bool _selected;

    /// <summary>The profile's name ("marked").</summary>
    public string Name { get; } = name;

    /// <summary>What it does.</summary>
    public string Summary { get; } = summary;

    /// <summary>Whether it is in the combination.</summary>
    public bool IsSelected
    {
        get => _selected;
        internal set => SetProperty(ref _selected, value);
    }

    /// <summary>Adds it to the combination or takes it out.</summary>
    public ICommand ToggleCommand { get; } = toggle;
}

/// <summary>A page layout (a <see cref="PageLayouts"/> name) as a chip: on, pages are drawn in it (one drawn per page).</summary>
public sealed class LayoutChip(string name, ICommand toggle) : ViewModel
{
    private bool _selected;

    /// <summary>The layout's name ("letter").</summary>
    public string Name { get; } = name;

    /// <summary>Whether pages are drawn in it.</summary>
    public bool IsSelected
    {
        get => _selected;
        internal set => SetProperty(ref _selected, value);
    }

    /// <summary>Turns it on or off.</summary>
    public ICommand ToggleCommand { get; } = toggle;
}

/// <summary>One page made: its image (kept at full size), its tile and preview pictures, and how it was made.</summary>
public sealed class PageVariant : ViewModel
{
    private bool _selected;

    internal PageVariant(int index, SyntheticPage page, PhotoPreview tile, PhotoPreview preview, TimeSpan took, ICommand select)
    {
        Index = index;
        Page = page;
        Tile = tile;
        Preview = preview;
        Took = took;
        TileSource = tile.Source();
        SelectCommand = select;
    }

    /// <summary>The variant's number, from 0.</summary>
    public int Index { get; }

    /// <summary>The page as made (the scanned image, the drawn text, the layout and look).</summary>
    public SyntheticPage Page { get; }

    /// <summary>The page with no scan (the same drawing), once asked for; null before.</summary>
    public SyntheticPage? Clean { get; internal set; }

    /// <summary>The clean page's preview, once made.</summary>
    internal PhotoPreview? CleanPreview { get; set; }

    internal PhotoPreview Tile { get; }

    internal PhotoPreview Preview { get; }

    /// <summary>How long it took to make.</summary>
    public TimeSpan Took { get; }

    /// <summary>Its tile in the grid.</summary>
    public ImageSource TileSource { get; }

    /// <summary>Its look ("marked+aged", "none").</summary>
    public string Look => Page.Profile;

    /// <summary>One line on it: "2 · letter · marked+aged · 180 ms".</summary>
    public string Caption => string.Create(CultureInfo.InvariantCulture, $"{Index + 1} · {Page.Layout} · {Page.Profile} · {Photos.Milliseconds(Took)}");

    /// <summary>Whether it is the one in the large preview.</summary>
    public bool IsSelected
    {
        get => _selected;
        internal set => SetProperty(ref _selected, value);
    }

    /// <summary>Shows it in the large preview.</summary>
    public ICommand SelectCommand { get; }
}

/// <summary>
/// The Generate lab (the owner's request: "the lib can generate images from text and add a lot of effects"): text typed or
/// pasted (Arabic and English; the library shapes and orders it) drawn as pages by the library's <see cref="SyntheticPages"/>
/// (what <c>idrak data synth</c> does), in fonts of the phone or added by the user (copied into the app's storage), in the
/// page layouts chosen, then made to look scanned: effects (<see cref="ScanProfiles"/>) as chips that stack into one
/// combination ("aged+handled+photo"), a random combination, or every look drawn one a page; a seed with a shuffle. The
/// pages are made on a worker thread one after another, each shown as it is made, cancellable; the large preview shows the
/// page picked, scanned or clean (the same drawing without its scan, made when first asked). A page is saved to the phone's
/// Pictures, shared, or the whole set written as a data set (records, images and manifest, as the CLI writes them).
/// Every option the CLI exposes is a setting here (<see cref="GenerateSettings"/>).
/// </summary>
public sealed class GenerateViewModel : ViewModel, IDisposable
{
    private readonly IShareService _share;
    private readonly IPictureSaver _pictures;
    private readonly ISettingsStore _settings;
    private readonly ProgressCenter _progress;
    private readonly IFilePicker? _picker;
    private readonly string _appStorage;
    private readonly Dictionary<string, IReadOnlyList<IFontFace>> _faces = new(StringComparer.Ordinal);
    private readonly Lock _facesGate = new();
    private readonly HashSet<string> _chosen = new(StringComparer.Ordinal);
    private readonly List<string> _combination = [];
    private GenerateLabOptions _options;
    private CancellationTokenSource? _stop;
    private PageVariant? _selected;
    private ImageSource? _previewSource;
    private ProgressItem? _work;
    private string _text, _status = "", _error = "", _fontQuery = "", _settingsMessage = "", _fontsText = "", _seedText;
    private bool _busy, _showClean, _everyLook, _showSettings, _listingFonts;

    /// <summary>The lab over the app's services; <paramref name="appStorage"/> is where added fonts and data sets go by default.</summary>
    public GenerateViewModel(IShareService share, ISettingsStore settings, string appStorage, IPictureSaver? pictures = null, ProgressCenter? progress = null,
        GenerateLabOptions? defaults = null, IFilePicker? picker = null)
    {
        _share = share ?? throw new ArgumentNullException(nameof(share));
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        ArgumentException.ThrowIfNullOrWhiteSpace(appStorage);
        _appStorage = appStorage;
        _pictures = pictures ?? new UnavailablePictureSaver();
        _progress = progress ?? new ProgressCenter(TimeSpan.FromMilliseconds(250));
        _picker = picker;
        _options = GenerateSettings.Load(settings, out var skipped, defaults);
        if (skipped.Count > 0)
        {
            _settingsMessage = "A saved setting was not read: " + string.Join("; ", skipped);
        }

        _text = settings.Get(GenerateSettings.TextKey) ?? "";
        _seedText = (_options.Seed ?? 0).ToString(CultureInfo.InvariantCulture);
        _everyLook = settings.Get(GenerateSettings.EveryLookKey) == "1";
        foreach (string path in (settings.Get(GenerateSettings.FontsKey) ?? "").Split('\n', StringSplitOptions.RemoveEmptyEntries))
        {
            _chosen.Add(path);
        }

        foreach (string name in ScanProfiles.Names)
        {
            Effects.Add(new EffectChip(name, Summary(name), new Command(() => ToggleEffect(name))));
        }

        foreach (string look in Split(settings.Get(GenerateSettings.ScanKey)))
        {
            if (Effects.Any(e => e.Name.Equals(look, StringComparison.OrdinalIgnoreCase)))
            {
                _combination.Add(look);
            }
        }

        var keptLayouts = Split(settings.Get(GenerateSettings.LayoutsKey), ',').ToHashSet(StringComparer.OrdinalIgnoreCase);
        var defaultLayouts = Library.Layouts.ToHashSet(StringComparer.OrdinalIgnoreCase);
        foreach (string name in PageLayouts.Names)
        {
            Layouts.Add(new LayoutChip(name, new Command(() => ToggleLayout(name))) { IsSelected = keptLayouts.Count > 0 ? keptLayouts.Contains(name) : defaultLayouts.Contains(name) });
        }

        foreach (var setting in GenerateSettings.Definitions)
        {
            Settings.Add(new LabSettingEntry(setting, setting.Get(_options)));
        }

        MarkEffects();
        GenerateCommand = new Command(async () => await GenerateAsync());
        CancelCommand = new Command(Cancel);
        ShuffleCommand = new Command(Shuffle);
        RandomEffectsCommand = new Command(() => RandomEffects());
        ClearEffectsCommand = new Command(ClearEffects);
        EveryLookCommand = new Command(() => EveryLook = !EveryLook);
        ToggleCleanCommand = new Command(async () => await ShowCleanAsync(!ShowClean));
        SaveCommand = new Command(async () => await SaveToPicturesAsync());
        ShareCommand = new Command(async () => await ShareAsync());
        SaveDatasetCommand = new Command(async () => await SaveDatasetAsync());
        AddFontsCommand = new Command(async () => await AddFontsAsync());
        PickFontsCommand = new Command(async () => await PickFontsForTextAsync());
        ClearFontsCommand = new Command(ClearFonts);
        ApplySettingsCommand = new Command(() => ApplySettings());
        ToggleSettingsCommand = new Command(() => ShowSettings = !ShowSettings);
        UpdateFontsText();
    }

    /// <summary>The page's title.</summary>
    public string Title => "Generate";

    /// <summary>The settings in effect.</summary>
    public GenerateLabOptions Options => _options;

    /// <summary>The text drawn on the pages (kept between runs).</summary>
    public string Text
    {
        get => _text;
        set
        {
            if (SetProperty(ref _text, value ?? ""))
            {
                OnPropertyChanged(nameof(CanGenerate));
            }
        }
    }

    /// <summary>The registered scan effects as chips.</summary>
    public ObservableCollection<EffectChip> Effects { get; } = [];

    /// <summary>The registered page layouts as chips.</summary>
    public ObservableCollection<LayoutChip> Layouts { get; } = [];

    /// <summary>The fonts found (the user's first), filtered by <see cref="FontQuery"/>; listed by <see cref="ListFontsAsync"/>.</summary>
    public ObservableCollection<FontChoice> Fonts { get; } = [];

    /// <summary>Every font found, before the search.</summary>
    public IReadOnlyList<FontChoice> AllFonts { get; private set; } = [];

    /// <summary>The pages made, one a variant, in order.</summary>
    public ObservableCollection<PageVariant> Variants { get; } = [];

    /// <summary>The lab's settings (every option of the CLI's, and the lab's own).</summary>
    public ObservableCollection<LabSettingEntry> Settings { get; } = [];

    /// <summary>The search over the fonts.</summary>
    public string FontQuery
    {
        get => _fontQuery;
        set
        {
            if (SetProperty(ref _fontQuery, (value ?? "").Trim()))
            {
                ShowFonts();
            }
        }
    }

    /// <summary>The fonts chosen, as a line ("Amiri, Cairo"), or what happens when none is.</summary>
    public string FontsText
    {
        get => _fontsText;
        private set => SetProperty(ref _fontsText, value);
    }

    /// <summary>The fonts chosen (their files).</summary>
    public IReadOnlyCollection<string> ChosenFonts => _chosen;

    /// <summary>Whether fonts are being listed.</summary>
    public bool IsListingFonts
    {
        get => _listingFonts;
        private set => SetProperty(ref _listingFonts, value);
    }

    /// <summary>The scan look a page gets: the effects joined with '+' ("aged+handled+photo"), "none" for a clean page; editable as text (a profile written as text too).</summary>
    public string Combination
    {
        get => _combination.Count == 0 ? "none" : string.Join("+", _combination);
        set
        {
            _combination.Clear();
            foreach (string look in Split(value))
            {
                if (!look.Equals("none", StringComparison.OrdinalIgnoreCase))
                {
                    _combination.Add(look);
                }
            }

            MarkEffects();
        }
    }

    /// <summary>Whether every built-in look is drawn, one a page (the CLI's default), instead of the combination.</summary>
    public bool EveryLook
    {
        get => _everyLook;
        set
        {
            if (SetProperty(ref _everyLook, value))
            {
                OnPropertyChanged(nameof(LookText));
                _settings.Set(GenerateSettings.EveryLookKey, value ? "1" : null);
            }
        }
    }

    /// <summary>What the pages will look like, as a line.</summary>
    public string LookText => EveryLook ? $"Every look, one drawn a page ({string.Join(", ", Library.Scans.Distinct(StringComparer.OrdinalIgnoreCase))})"
        : _combination.Count == 0 ? "Clean: the page as drawn" : $"Look: {Combination}";

    /// <summary>The seed as typed (a whole number).</summary>
    public string SeedText
    {
        get => _seedText;
        set => SetProperty(ref _seedText, value ?? "");
    }

    /// <summary>Whether pages are being made or a data set written.</summary>
    public bool IsBusy
    {
        get => _busy;
        private set
        {
            if (SetProperty(ref _busy, value))
            {
                OnPropertyChanged(nameof(IsIdle));
                OnPropertyChanged(nameof(CanGenerate));
            }
        }
    }

    /// <summary>Whether nothing runs.</summary>
    public bool IsIdle => !IsBusy;

    /// <summary>Whether Generate can run now.</summary>
    public bool CanGenerate => !IsBusy && Text.Trim().Length > 0;

    /// <summary>The work running, for its progress bar.</summary>
    public ProgressItem? Work
    {
        get => _work;
        private set
        {
            if (SetProperty(ref _work, value))
            {
                OnPropertyChanged(nameof(HasWork));
            }
        }
    }

    /// <summary>Whether a progress shows.</summary>
    public bool HasWork => Work is not null;

    /// <summary>The page in the large preview, or null.</summary>
    public PageVariant? Selected
    {
        get => _selected;
        private set
        {
            if (SetProperty(ref _selected, value))
            {
                foreach (var variant in Variants)
                {
                    variant.IsSelected = ReferenceEquals(variant, value);
                }

                OnPropertyChanged(nameof(HasPage));
                OnPropertyChanged(nameof(HasNoPage));
                OnPropertyChanged(nameof(PageText));
            }
        }
    }

    /// <summary>Whether a page is shown.</summary>
    public bool HasPage => Selected is not null;

    /// <summary>Whether none is.</summary>
    public bool HasNoPage => Selected is null;

    /// <summary>The page shown: its size, layout, look and what its layout chose.</summary>
    public string PageText => Selected is { } v
        ? string.Create(CultureInfo.InvariantCulture, $"{v.Page.Image.Width:N0} x {v.Page.Image.Height:N0} · {v.Page.Layout} · {(ShowClean ? "clean" : v.Page.Profile)} · {v.Page.Description}")
        : "";

    /// <summary>The large preview's picture: the page scanned, or clean.</summary>
    public ImageSource? PreviewSource
    {
        get => _previewSource;
        private set => SetProperty(ref _previewSource, value);
    }

    /// <summary>Whether the large preview shows the clean page (before its scan).</summary>
    public bool ShowClean
    {
        get => _showClean;
        private set
        {
            if (SetProperty(ref _showClean, value))
            {
                OnPropertyChanged(nameof(CompareText));
                OnPropertyChanged(nameof(PageText));
            }
        }
    }

    /// <summary>The before/after button's words.</summary>
    public string CompareText => ShowClean ? "Show scanned" : "Show clean";

    /// <summary>What happened last.</summary>
    public string Status
    {
        get => _status;
        private set
        {
            if (SetProperty(ref _status, value))
            {
                OnPropertyChanged(nameof(HasStatus));
            }
        }
    }

    /// <summary>Whether <see cref="Status"/> has something to show.</summary>
    public bool HasStatus => Status.Length > 0;

    /// <summary>What went wrong last, or "".</summary>
    public string Error
    {
        get => _error;
        private set
        {
            if (SetProperty(ref _error, value))
            {
                OnPropertyChanged(nameof(HasError));
            }
        }
    }

    /// <summary>Whether <see cref="Error"/> has something to show.</summary>
    public bool HasError => Error.Length > 0;

    /// <summary>Whether the settings show.</summary>
    public bool ShowSettings
    {
        get => _showSettings;
        set
        {
            if (SetProperty(ref _showSettings, value))
            {
                OnPropertyChanged(nameof(SettingsButtonText));
            }
        }
    }

    /// <summary>The settings button's words.</summary>
    public string SettingsButtonText => ShowSettings ? "Hide the settings" : "Settings";

    /// <summary>What the last keep said.</summary>
    public string SettingsMessage
    {
        get => _settingsMessage;
        private set
        {
            if (SetProperty(ref _settingsMessage, value))
            {
                OnPropertyChanged(nameof(HasSettingsMessage));
            }
        }
    }

    /// <summary>Whether <see cref="SettingsMessage"/> has something to show.</summary>
    public bool HasSettingsMessage => SettingsMessage.Length > 0;

    /// <summary>Whether a page can be saved to the phone's Pictures here.</summary>
    public bool CanSaveToPictures => _pictures.CanSave;

    /// <summary>Makes the pages.</summary>
    public ICommand GenerateCommand { get; }

    /// <summary>Stops the pages being made (those made stay) or the data set being written.</summary>
    public ICommand CancelCommand { get; }

    /// <summary>A new seed.</summary>
    public ICommand ShuffleCommand { get; }

    /// <summary>A random combination of effects.</summary>
    public ICommand RandomEffectsCommand { get; }

    /// <summary>No effect: clean pages.</summary>
    public ICommand ClearEffectsCommand { get; }

    /// <summary>Every look one a page, or the combination.</summary>
    public ICommand EveryLookCommand { get; }

    /// <summary>The large preview scanned or clean.</summary>
    public ICommand ToggleCleanCommand { get; }

    /// <summary>Saves the page shown to the phone's Pictures.</summary>
    public ICommand SaveCommand { get; }

    /// <summary>Shares the page shown.</summary>
    public ICommand ShareCommand { get; }

    /// <summary>Writes the text's pages as a data set.</summary>
    public ICommand SaveDatasetCommand { get; }

    /// <summary>Adds font files (copied into the app's storage).</summary>
    public ICommand AddFontsCommand { get; }

    /// <summary>Chooses the fonts that have the text's characters.</summary>
    public ICommand PickFontsCommand { get; }

    /// <summary>Chooses no font (picked for the text at the next Generate).</summary>
    public ICommand ClearFontsCommand { get; }

    /// <summary>Checks and keeps the settings.</summary>
    public ICommand ApplySettingsCommand { get; }

    /// <summary>Shows or hides the settings.</summary>
    public ICommand ToggleSettingsCommand { get; }

    // The library's defaults (the CLI's), without fonts.
    private static SyntheticPageOptions Library => field ??= new SyntheticPageOptions { Style = new PageStyle { Fonts = [] } };

    /// <summary>
    /// Lists the fonts of the phone's folder and the added ones' (the added first), on a worker thread; the ones chosen
    /// before stay chosen. Nothing is read but the folders.
    /// </summary>
    public async Task ListFontsAsync()
    {
        if (IsListingFonts)
        {
            return;
        }

        IsListingFonts = true;
        try
        {
            string added = _options.AddedFontsFolder(_appStorage);
            string? phone = _options.SystemFontsFolder;
            var found = await Task.Run(() => FindFonts(added, phone)).ConfigureAwait(true);
            AllFonts = [.. found.Select(f => new FontChoice(f.Path, f.Bytes, f.Added, _chosen.Contains(f.Path), () => OnChosen(f.Path)))];
            ShowFonts();
            UpdateFontsText();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = $"The fonts could not be listed: {e.Message}";
        }
        finally
        {
            IsListingFonts = false;
        }
    }

    /// <summary>Puts <paramref name="look"/> into the combination, or takes it out.</summary>
    public void ToggleEffect(string look)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(look);
        int at = _combination.FindIndex(l => l.Equals(look, StringComparison.OrdinalIgnoreCase));
        if (at >= 0)
        {
            _combination.RemoveAt(at);
        }
        else
        {
            _combination.Add(look);
        }

        EveryLook = false;
        MarkEffects();
    }

    /// <summary>A random combination of one to three effects (from <paramref name="seed"/>, else a new one); the combination.</summary>
    public string RandomEffects(int? seed = null)
    {
        var random = seed is { } s ? new Random(s) : new Random();
        var names = Effects.Select(e => e.Name).ToList();
        _combination.Clear();
        int count = Math.Min(names.Count, 1 + random.Next(3));
        while (_combination.Count < count)
        {
            string pick = names[random.Next(names.Count)];
            if (!_combination.Contains(pick, StringComparer.OrdinalIgnoreCase))
            {
                _combination.Add(pick);
            }
        }

        EveryLook = false;
        MarkEffects();
        return Combination;
    }

    /// <summary>A new seed.</summary>
    public void Shuffle() => SeedText = Random.Shared.Next().ToString(CultureInfo.InvariantCulture);

    /// <summary>The options the pages are made with now: the settings over the library's defaults, the fonts given.</summary>
    /// <exception cref="ArgumentException">A setting or the look the library refuses (the message says why).</exception>
    public SyntheticPageOptions PageOptions(IReadOnlyList<IFontFace> fonts)
    {
        ArgumentNullException.ThrowIfNull(fonts);
        var defaults = Library with { Style = new PageStyle { Fonts = fonts } };
        var layouts = Layouts.Where(l => l.IsSelected).Select(l => l.Name).ToList();
        return defaults with
        {
            Style = SyntheticPages.Style(defaults.Style, _options.Page),
            Parts = _options.Parts is { } parts ? SyntheticPageParts.Read(parts) : SyntheticPageParts.Empty,
            Variants = _options.Count ?? defaults.Variants,
            Layouts = layouts.Count > 0 ? layouts : defaults.Layouts,
            Scans = EveryLook ? defaults.Scans : [Combination],
            Grey = _options.Grey ?? defaults.Grey,
            Seed = Seed(),
            Prompt = _options.Prompt ?? defaults.Prompt,
            AnswerKey = _options.AnswerKey,
            ImageFormat = _options.ImageFormat ?? defaults.ImageFormat,
            Quality = _options.Quality ?? defaults.Quality,
            Zip = _options.Zip ?? false,
            Workers = _options.Workers ?? 0,
            Format = "text",
        };
    }

    /// <summary>
    /// Makes the pages of the text (<see cref="GenerateLabOptions.Count"/> variants) on a worker thread, one after another,
    /// each shown as it is made; Cancel stops after the page being made (those made stay). With no font chosen, the fonts
    /// that have the text's characters are picked first. True when every page was made.
    /// </summary>
    public async Task<bool> GenerateAsync()
    {
        if (IsBusy)
        {
            return false;
        }

        Error = "";
        string text = Text;
        if (text.Trim().Length == 0)
        {
            Error = "Type or paste the text to draw first.";
            return false;
        }

        _settings.Set(GenerateSettings.TextKey, text);
        KeepLooks();
        using var stop = new CancellationTokenSource();
        _stop = stop;
        IsBusy = true;
        ShowClean = false;
        Variants.Clear();
        Selected = null;
        PreviewSource = null;
        var reporter = _progress.Start("generate pages", _options.Count ?? Library.Variants, ProgressUnit.Items);
        Work = _progress.ItemOf(reporter);
        try
        {
            var fonts = await FontsForAsync(text, stop.Token).ConfigureAwait(true);
            var options = PageOptions(fonts);
            int? side = _options.PreviewPixels;
            int tile = _options.Thumbnail;
            Status = $"Making {options.Variants} page{(options.Variants == 1 ? "" : "s")}: {(EveryLook ? "every look" : Combination)}, seed {options.Seed}…";
            int made = 0;
            var watch = Stopwatch.StartNew();
            await Task.Run(() =>
            {
                for (int v = 0; v < options.Variants; v++)
                {
                    stop.Token.ThrowIfCancellationRequested();
                    long start = Stopwatch.GetTimestamp();
                    var page = SyntheticPages.Make(text, options, record: 0, variant: v, where: "the text");
                    var took = Stopwatch.GetElapsedTime(start);
                    var small = Photos.Preview(page.Image, tile);
                    var large = Photos.Preview(page.Image, side);
                    int index = v;
                    made++;
                    reporter.Report(made);
                    _progress.Dispatch(() => Add(index, page, small, large, took));
                }
            }, stop.Token).ConfigureAwait(true);
            reporter.Finish($"{made} pages");
            Status = string.Create(CultureInfo.InvariantCulture, $"Made {made} page{(made == 1 ? "" : "s")} in {Photos.Milliseconds(watch.Elapsed)}.");
            return true;
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested)
        {
            reporter.Finish("stopped");
            Status = $"Stopped: {Variants.Count} page{(Variants.Count == 1 ? "" : "s")} made.";
            return false;
        }
        catch (Exception e) when (e is ArgumentException or InvalidDataException or NotSupportedException or FileNotFoundException or IOException)
        {
            reporter.Finish("failed");
            Status = "";
            Error = e.Message.Split('\n')[0];
            return false;
        }
        finally
        {
            _stop = null;
            Work = null;
            IsBusy = false;
        }
    }

    /// <summary>Stops what runs (the pages made, or the data set written, so far stay).</summary>
    public void Cancel()
    {
        try
        {
            _stop?.Cancel();
        }
        catch (ObjectDisposedException)
        {
            // It ended meanwhile.
        }
    }

    /// <summary>Shows <paramref name="variant"/> in the large preview (scanned, or clean when that is shown).</summary>
    public async Task SelectAsync(PageVariant variant)
    {
        ArgumentNullException.ThrowIfNull(variant);
        Selected = variant;
        if (ShowClean)
        {
            await ShowCleanAsync(true).ConfigureAwait(true);
        }
        else
        {
            PreviewSource = variant.Preview.Source();
        }
    }

    /// <summary>
    /// Shows the page shown clean (before its scan: the same drawing, made on a worker thread when first asked) or scanned;
    /// false when there is no page.
    /// </summary>
    public async Task<bool> ShowCleanAsync(bool clean)
    {
        if (Selected is not { } variant)
        {
            return false;
        }

        if (!clean)
        {
            ShowClean = false;
            PreviewSource = variant.Preview.Source();
            return true;
        }

        if (variant.CleanPreview is null)
        {
            try
            {
                var fonts = await FontsForAsync(Text, CancellationToken.None).ConfigureAwait(true);
                var options = PageOptions(fonts) with { Scans = ["none"] };
                int? side = _options.PreviewPixels;
                string text = Text;
                var (page, preview) = await Task.Run(() =>
                {
                    var made = SyntheticPages.Make(text, options, record: 0, variant: variant.Index, where: "the text");
                    return (made, Photos.Preview(made.Image, side));
                }).ConfigureAwait(true);
                variant.Clean = page;
                variant.CleanPreview = preview;
            }
            catch (Exception e) when (e is ArgumentException or InvalidDataException or NotSupportedException or IOException)
            {
                Error = e.Message.Split('\n')[0];
                return false;
            }
        }

        if (!ReferenceEquals(Selected, variant))
        {
            return false;
        }

        ShowClean = true;
        PreviewSource = variant.CleanPreview!.Source();
        return true;
    }

    /// <summary>The page shown as a file of the settings' format (encoded on a worker thread): its bytes, name and media type; null when none is shown.</summary>
    public async Task<(byte[] Bytes, string Name, string MediaType)?> EncodeAsync()
    {
        if (Selected is not { } variant)
        {
            Error = "Generate a page first.";
            return null;
        }

        var image = ShowClean && variant.Clean is { } clean ? clean.Image : variant.Page.Image;
        string format = _options.ImageFormat ?? Library.ImageFormat;
        int quality = _options.Quality ?? Library.Quality;
        bool jpeg = format.Equals("jpeg", StringComparison.OrdinalIgnoreCase);
        string extension = jpeg ? ".jpg" : ImageEncoders.Get(format).Extensions.FirstOrDefault() ?? "." + format;
        string name = string.Create(CultureInfo.InvariantCulture, $"page-{Seed()}-{variant.Index + 1}{(ShowClean ? "-clean" : "")}{extension}");
        byte[] bytes = await Task.Run(() => jpeg ? JpegEncoder.Encode(image, quality) : ImageEncoders.Get(format).Encode(image)).ConfigureAwait(true);
        return (bytes, name, jpeg ? "image/jpeg" : format.Equals("png", StringComparison.OrdinalIgnoreCase) ? "image/png" : "application/octet-stream");
    }

    /// <summary>Saves the page shown to the phone's Pictures (the settings' album); where it went, or null (said why).</summary>
    public async Task<string?> SaveToPicturesAsync()
    {
        Error = "";
        if (!_pictures.CanSave)
        {
            Error = $"Not saved to Pictures: {_pictures.Unavailable}.";
            return null;
        }

        try
        {
            if (await EncodeAsync().ConfigureAwait(true) is not { } file)
            {
                return null;
            }

            string where = await _pictures.SaveAsync(file.Bytes, file.Name, file.MediaType, _options.AlbumName).ConfigureAwait(true);
            Status = $"Saved to {where} ({ProgressText.Bytes(file.Bytes.Length)}).";
            return where;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or NotSupportedException or ArgumentException)
        {
            Error = $"The page was not saved: {e.Message}";
            return null;
        }
    }

    /// <summary>Writes the page shown to the cache folder's "images" and offers it to the share sheet; false when there is none or it failed.</summary>
    public async Task<bool> ShareAsync()
    {
        Error = "";
        try
        {
            if (await EncodeAsync().ConfigureAwait(true) is not { } file)
            {
                return false;
            }

            string folder = Path.Combine(ComputeResources.CacheFolder, "images");
            string path = Path.Combine(folder, file.Name);
            await Task.Run(() =>
            {
                Directory.CreateDirectory(folder);
                File.WriteAllBytes(path, file.Bytes);
            }).ConfigureAwait(true);
            await _share.ShareFileAsync(path, "Page", file.MediaType).ConfigureAwait(true);
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The page was not shared: {e.Message}";
            return false;
        }
    }

    /// <summary>
    /// Writes the text's pages as a data set, as <c>idrak data synth</c> writes it: the text as a plain text record, its
    /// pages (the settings' count, look, layouts and seed), the records (.jsonl), the images (a folder, or a zip) and the
    /// manifest, in a folder of its own under the data sets' folder; on a worker thread with its progress, cancellable.
    /// The records' path, or null (said why).
    /// </summary>
    public async Task<string?> SaveDatasetAsync()
    {
        if (IsBusy)
        {
            return null;
        }

        Error = "";
        string text = Text;
        if (text.Trim().Length == 0)
        {
            Error = "Type or paste the text to draw first.";
            return null;
        }

        using var stop = new CancellationTokenSource();
        _stop = stop;
        IsBusy = true;
        var reporter = _progress.Start("data set", _options.Count ?? Library.Variants, ProgressUnit.Items);
        Work = _progress.ItemOf(reporter);
        try
        {
            var fonts = await FontsForAsync(text, stop.Token).ConfigureAwait(true);
            var options = PageOptions(fonts);
            string folder = Path.Combine(_options.DatasetsFolder(_appStorage), DateTime.Now.ToString("yyyyMMdd-HHmmss", CultureInfo.InvariantCulture));
            string input = Path.Combine(folder, "input.txt"), output = Path.Combine(folder, "pages.jsonl");
            var result = await Task.Run(() =>
            {
                Directory.CreateDirectory(folder);
                File.WriteAllText(input, text);
                return SyntheticPages.Write(input, output, options, new Pages(reporter), stop.Token);
            }, stop.Token).ConfigureAwait(true);
            reporter.Finish($"{result.Pages} pages");
            Status = string.Create(CultureInfo.InvariantCulture,
                $"Data set: {result.Pages} pages in {result.Elapsed.TotalSeconds:F1} s: {result.Data}, images in {result.Images}, manifest {result.Manifest}.");
            return result.Data;
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested)
        {
            reporter.Finish("stopped");
            Status = "The data set was stopped.";
            return null;
        }
        catch (Exception e) when (e is ArgumentException or InvalidDataException or NotSupportedException or IOException or UnauthorizedAccessException)
        {
            reporter.Finish("failed");
            Error = $"The data set was not written: {e.Message.Split('\n')[0]}";
            return null;
        }
        finally
        {
            _stop = null;
            Work = null;
            IsBusy = false;
        }
    }

    /// <summary>Adds font files picked by the user: each copied into the added fonts' folder (it must read as a font) and chosen.</summary>
    public async Task<int> AddFontsAsync()
    {
        Error = "";
        IEnumerable<FileResult>? picked;
        try
        {
            picked = await (_picker ?? FilePicker.Default).PickMultipleAsync(new PickOptions { PickerTitle = "Fonts to add (.ttf, .otf, .ttc)" }).ConfigureAwait(true);
        }
        catch (Exception e) when (e is NotImplementedException or NotSupportedException or PermissionException)
        {
            Error = $"Files cannot be picked here: {e.Message}";
            return 0;
        }

        var files = picked?.Where(f => f is not null).ToList() ?? [];
        if (files.Count == 0)
        {
            return 0;
        }

        var streams = new List<(string Name, byte[] Bytes)>();
        foreach (var file in files)
        {
            await using var stream = await file.OpenReadAsync().ConfigureAwait(true);
            using var copy = new MemoryStream();
            await stream.CopyToAsync(copy).ConfigureAwait(true);
            streams.Add((file.FileName, copy.ToArray()));
        }

        return await AddFontsAsync(streams).ConfigureAwait(true);
    }

    /// <summary>Adds fonts given as files' names and bytes: each that reads as a font is copied into the added fonts' folder and chosen; those that do not are named.</summary>
    public async Task<int> AddFontsAsync(IReadOnlyList<(string Name, byte[] Bytes)> files)
    {
        ArgumentNullException.ThrowIfNull(files);
        string folder = _options.AddedFontsFolder(_appStorage);
        var (added, refused) = await Task.Run(() =>
        {
            Directory.CreateDirectory(folder);
            var good = new List<string>();
            var bad = new List<string>();
            foreach (var (name, bytes) in files)
            {
                string safe = Path.GetFileName(name);
                if (!FontFormats.Extensions.Contains(Path.GetExtension(safe).ToLowerInvariant()))
                {
                    bad.Add($"{safe}: not a font file ({string.Join(", ", FontFormats.Extensions)})");
                    continue;
                }

                try
                {
                    var faces = FontFormats.Read(bytes, Path.GetFileNameWithoutExtension(safe));
                    string path = Path.Combine(folder, safe);
                    File.WriteAllBytes(path, bytes);
                    lock (_facesGate)
                    {
                        _faces[path] = faces;
                    }

                    good.Add(path);
                }
                catch (InvalidDataException e)
                {
                    bad.Add($"{safe}: {e.Message}");
                }
            }

            return (good, bad);
        }).ConfigureAwait(true);
        foreach (string path in added)
        {
            _chosen.Add(path);
        }

        KeepFonts();
        await ListFontsAsync().ConfigureAwait(true);
        Status = added.Count > 0 ? $"Added {added.Count} font{(added.Count == 1 ? "" : "s")}: {string.Join(", ", added.Select(Path.GetFileNameWithoutExtension))}." : "";
        Error = refused.Count > 0 ? "Not added: " + string.Join("; ", refused) : "";
        return added.Count;
    }

    /// <summary>Chooses the fonts that have every character of the text (the user's first, up to the setting's count), read on a worker thread.</summary>
    public async Task<int> PickFontsForTextAsync()
    {
        Error = "";
        if (Text.Trim().Length == 0)
        {
            Error = "Type the text first: the fonts are picked by its characters.";
            return 0;
        }

        try
        {
            _chosen.Clear();
            var fonts = await FontsForAsync(Text, CancellationToken.None).ConfigureAwait(true);
            return fonts.Count;
        }
        catch (Exception e) when (e is InvalidDataException or IOException)
        {
            Error = e.Message;
            return 0;
        }
    }

    /// <summary>Chooses no font.</summary>
    public void ClearFonts()
    {
        _chosen.Clear();
        foreach (var font in AllFonts)
        {
            font.IsChosen = false;
        }

        KeepFonts();
        UpdateFontsText();
    }

    /// <summary>Checks every setting and keeps them; false, and nothing kept, when one is refused (each says why).</summary>
    public bool ApplySettings()
    {
        if (LabSettingEntry.Apply(Settings, _options.Clone()) is not { } next)
        {
            SettingsMessage = "Nothing was kept: a value above is not one the setting takes.";
            return false;
        }

        bool folders = next.SystemFonts != _options.SystemFonts || next.FontsFolder != _options.FontsFolder;
        _options = next;
        GenerateSettings.Save(_settings, _options);
        SettingsMessage = "Kept.";
        if (folders)
        {
            _ = ListFontsAsync();
        }

        return true;
    }

    /// <inheritdoc />
    public void Dispose()
    {
        Cancel();
        _stop?.Dispose();
    }

    // On the UI thread: a page made, into the grid; the first into the large preview.
    private void Add(int index, SyntheticPage page, PhotoPreview tile, PhotoPreview preview, TimeSpan took)
    {
        PageVariant? variant = null;
        variant = new PageVariant(index, page, tile, preview, took, new Command(async () => await SelectAsync(variant!)));
        Variants.Add(variant);
        if (Selected is null)
        {
            Selected = variant;
            PreviewSource = preview.Source();
        }
    }

    // The fonts the pages are set in: those chosen (read once, kept), else picked for the text and chosen.
    private async Task<IReadOnlyList<IFontFace>> FontsForAsync(string text, CancellationToken cancellationToken)
    {
        if (AllFonts.Count == 0)
        {
            await ListFontsAsync().ConfigureAwait(true);
        }

        var chosen = AllFonts.Where(f => f.IsChosen).Select(f => f.Path).Concat(_chosen).Distinct(StringComparer.Ordinal).ToList();
        var candidates = AllFonts.Where(f => f.Bytes <= _options.AutoFontBytes || f.IsAdded).Select(f => f.Path).ToList();
        int want = _options.AutoFontCount;
        var (faces, picked) = await Task.Run(() =>
        {
            if (chosen.Count > 0)
            {
                return ((IReadOnlyList<IFontFace>)[.. chosen.SelectMany(Faces)], (IReadOnlyList<string>)[]);
            }

            // The characters a page needs, as the library counts them (marks and format characters ride on their letters).
            var needed = text.EnumerateRunes().Where(r => !Rune.IsWhiteSpace(r) && !Rune.IsControl(r)
                && Rune.GetUnicodeCategory(r) is not (UnicodeCategory.NonSpacingMark or UnicodeCategory.SpacingCombiningMark or UnicodeCategory.EnclosingMark or UnicodeCategory.Format))
                .Select(r => r.Value).ToHashSet();
            var full = new List<string>();
            var covers = new List<(string Path, HashSet<int> Covered)>();
            foreach (string path in candidates)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var read = Faces(path);
                var covered = needed.Where(c => read.Any(f => f.Glyph(c) != 0)).ToHashSet();
                if (covered.Count == needed.Count)
                {
                    full.Add(path);
                    if (full.Count >= want)
                    {
                        break;
                    }
                }
                else if (covered.Count > 0)
                {
                    covers.Add((path, covered));
                }
            }

            // Fonts that have the whole text; else, as the library sets a page across fonts, the fewest that have it between
            // them: each time the one that has the most of what is still missing.
            var take = full;
            if (take.Count == 0)
            {
                var missing = new HashSet<int>(needed);
                while (missing.Count > 0 && take.Count < want && covers.Count > 0)
                {
                    var best = covers.MaxBy(c => c.Covered.Count(missing.Contains));
                    if (best.Covered.Count(missing.Contains) == 0)
                    {
                        break;
                    }

                    take.Add(best.Path);
                    missing.ExceptWith(best.Covered);
                    covers.Remove(best);
                }
            }

            take = [.. candidates.Where(take.Contains)];                         // in the list's order, as a choice made by hand reads them
            return ((IReadOnlyList<IFontFace>)[.. take.SelectMany(Faces)], (IReadOnlyList<string>)take);
        }, cancellationToken).ConfigureAwait(true);
        if (picked.Count > 0)
        {
            foreach (string path in picked)
            {
                _chosen.Add(path);
                if (AllFonts.FirstOrDefault(f => f.Path == path) is { } font)
                {
                    font.IsChosen = true;
                }
            }

            KeepFonts();
            UpdateFontsText();
            Status = $"Picked for the text: {string.Join(", ", picked.Select(Path.GetFileNameWithoutExtension))}.";
        }

        if (faces.Count == 0)
        {
            throw new InvalidDataException(AllFonts.Count == 0
                ? "No font is here: add font files (.ttf, .otf, .ttc), or set the phone's fonts folder in the settings."
                : "No font has the text's characters: add one that has them.");
        }

        return faces;
    }

    // A font file's faces, read once and kept (a face holds its file's bytes).
    private IReadOnlyList<IFontFace> Faces(string path)
    {
        lock (_facesGate)
        {
            if (_faces.TryGetValue(path, out var kept))
            {
                return kept;
            }
        }

        IReadOnlyList<IFontFace> faces;
        try
        {
            faces = FontFormats.Read(path);
        }
        catch (InvalidDataException)
        {
            faces = [];                                                         // a file no format reads sets no page
        }

        lock (_facesGate)
        {
            _faces[path] = faces;
        }

        return faces;
    }

    private static List<(string Path, long Bytes, bool Added)> FindFonts(string added, string? phone)
    {
        var extensions = FontFormats.Extensions.ToHashSet(StringComparer.OrdinalIgnoreCase);
        var found = new List<(string, long, bool)>();
        foreach (var (folder, isAdded) in new[] { (added, true), (phone, false) })
        {
            if (folder is null || !Directory.Exists(folder))
            {
                continue;
            }

            foreach (string file in Directory.EnumerateFiles(folder, "*", SearchOption.AllDirectories).Order(StringComparer.Ordinal))
            {
                if (extensions.Contains(Path.GetExtension(file)))
                {
                    found.Add((file, new FileInfo(file).Length, isAdded));
                }
            }
        }

        return found;
    }

    private void OnChosen(string path)
    {
        if (AllFonts.FirstOrDefault(f => f.Path == path) is { } font)
        {
            if (font.IsChosen)
            {
                _chosen.Add(path);
            }
            else
            {
                _chosen.Remove(path);
            }
        }

        KeepFonts();
        UpdateFontsText();
    }

    private void ShowFonts()
    {
        var shown = _fontQuery.Length == 0 ? AllFonts : [.. AllFonts.Where(f => f.Name.Contains(_fontQuery, StringComparison.OrdinalIgnoreCase))];
        Lists.Replace(Fonts, shown, static (a, b) => ReferenceEquals(a, b));
    }

    private void UpdateFontsText()
    {
        var names = _chosen.Select(Path.GetFileNameWithoutExtension).Order(StringComparer.CurrentCultureIgnoreCase).ToList();
        FontsText = names.Count == 0 ? "No font chosen: Generate picks those that have the text's characters."
            : names.Count <= 4 ? string.Join(", ", names) : $"{string.Join(", ", names.Take(4))} and {names.Count - 4} more";
    }

    private void ClearEffects()
    {
        _combination.Clear();
        EveryLook = false;
        MarkEffects();
    }

    private void ToggleLayout(string name)
    {
        if (Layouts.FirstOrDefault(l => l.Name == name) is { } chip)
        {
            chip.IsSelected = !chip.IsSelected || Layouts.Count(l => l.IsSelected) == 1 && chip.IsSelected;   // one stays on
        }

        KeepLooks();
    }

    private void MarkEffects()
    {
        foreach (var chip in Effects)
        {
            chip.IsSelected = _combination.Contains(chip.Name, StringComparer.OrdinalIgnoreCase);
        }

        OnPropertyChanged(nameof(Combination));
        OnPropertyChanged(nameof(LookText));
        KeepLooks();
    }

    private void KeepLooks()
    {
        _settings.Set(GenerateSettings.ScanKey, _combination.Count == 0 ? null : string.Join("+", _combination));
        _settings.Set(GenerateSettings.LayoutsKey, string.Join(',', Layouts.Where(l => l.IsSelected).Select(l => l.Name)));
    }

    private void KeepFonts() => _settings.Set(GenerateSettings.FontsKey, _chosen.Count == 0 ? null : string.Join('\n', _chosen.Order(StringComparer.Ordinal)));

    // The seed as typed, else the settings' (0).
    private int Seed() => int.TryParse(SeedText, NumberStyles.Integer, CultureInfo.InvariantCulture, out int seed) ? seed : _options.Seed ?? 0;

    private static IEnumerable<string> Split(string? text, char separator = '+') =>
        (text ?? "").Split(separator, StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);

    private static string Summary(string name)
    {
        try
        {
            return ScanProfiles.Get(name).Summary;
        }
        catch (Exception e) when (e is NotSupportedException or NotRegisteredException)
        {
            return "";
        }
    }

    // The data set's progress onto its reporter.
    private sealed class Pages(IProgressReporter reporter) : IProgress<SyntheticPagesProgress>
    {
        public void Report(SyntheticPagesProgress value)
        {
            reporter.Total = value.Total;
            reporter.Report(value.Pages);
        }
    }
}
