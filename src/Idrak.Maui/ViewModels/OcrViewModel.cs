// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using System.Windows.Input;
using Idrak.Maui.Abstractions;
using Idrak.Models.Abstractions;
using Idrak.Text.Generation;
using Microsoft.Maui.ApplicationModel.DataTransfer;
using Microsoft.Maui.Dispatching;
using Microsoft.Maui.Media;

namespace Idrak.Maui.ViewModels;

/// <summary>A vision-language model of the model store, as the OCR page lists it.</summary>
public sealed class VisionModelEntry
{
    internal VisionModelEntry(StoredModel model, string architecture)
    {
        Model = model;
        Architecture = architecture;
    }

    /// <summary>The stored model.</summary>
    public StoredModel Model { get; }

    /// <summary>The architecture its config.json names, a registered vision family's.</summary>
    public string Architecture { get; }

    /// <summary>How the settings keep it: "owner/name@revision".</summary>
    public string Key => $"{Model.Id}@{Model.Revision}";

    /// <summary>Its id.</summary>
    public string Title => Model.Id;

    /// <summary>Its architecture, size and revision.</summary>
    public string Details => $"{Architecture}, {ProgressText.Bytes(Model.Bytes)}, revision {Model.Revision}";

    /// <inheritdoc />
    public override string ToString() => Title;
}

/// <summary>
/// The OCR lab (plan 34, M6): a photo from the camera (its permission asked first) or the gallery, decoded once by the
/// library's own codecs and turned upright; the optional preprocessing pipeline run on it once and the page shown as the
/// model will see it; read by a vision-language model of the store (a complete model whose architecture a registered
/// vision family reads), loaded once through the <see cref="IVisionReaderService"/> after the same measured check as Chat
/// (refused with the limit named when it does not fit); the page handed to the reader as prepared (decoded and
/// transformed once, rule 75); the answer streamed with its tokens per second, the text shown at most once a progress
/// interval whatever the number of tokens; the first read after a load (cold) reported apart from the warm ones (rule 76);
/// Stop keeps what came; the text copied or shared. The instruction, the pipeline and the reading options are settings
/// (<see cref="OcrSettings"/>), the library's own where none is set.
/// </summary>
public sealed class OcrViewModel : ViewModel
{
    private readonly IModelStore _store;
    private readonly IVisionReaderService _readers;
    private readonly IPermissionService _permissions;
    private readonly IShareService _share;
    private readonly IdrakMauiOptions _app;
    private readonly ISettingsStore _settings;
    private readonly IDispatcher? _dispatcher;
    private readonly IClipboard? _clipboard;
    private readonly IMediaPicker? _picker;
    private readonly TimeProvider _time;
    private readonly List<double> _warmRates = [];
    private readonly Lock _pending = new();
    private readonly StringBuilder _pendingText = new();
    private OcrLabOptions _options;
    private VisionModelEntry? _selected;
    private VisionReaderInfo? _info;
    private Task<bool>? _loading;
    private string? _loadingKey;
    private Photo? _photo;
    private ImageData? _prepared;
    private string? _preparedFor;
    private ImageSource? _pageSource;
    private CancellationTokenSource? _cancellation;
    private int _readsSinceLoad;
    private bool _flushQueued;
    private long _lastFlush;
    private int _generated;
    private bool _busy, _reading, _listing, _refused, _showSettings;
    private string _pipeline;
    private string _modelStatus = "", _modelsText = "", _photoText = "", _pageText = "", _text = "", _liveText = "", _coldText = "", _warmText = "";
    private string _status = "", _error = "", _settingsMessage = "";

    /// <summary>The page over the app's services; <paramref name="clipboard"/> and <paramref name="picker"/> are MAUI's own when null.</summary>
    /// <param name="store">The model store.</param>
    /// <param name="readers">Loads the readers.</param>
    /// <param name="permissions">The phone's permissions (the camera).</param>
    /// <param name="share">The share sheet.</param>
    /// <param name="options">The app's options (the device; the progress interval batches the text's updates).</param>
    /// <param name="settings">Where the lab's settings are kept.</param>
    /// <param name="defaults">The app's defaults for the lab's settings (the library's where null or unset).</param>
    /// <param name="dispatcher">The UI thread's dispatcher; null: the app's when there is one, else the reading thread itself.</param>
    /// <param name="clipboard">The clipboard; null: MAUI's.</param>
    /// <param name="picker">The media picker; null: MAUI's.</param>
    /// <param name="time">The clock the batching and the rates read (tests give their own).</param>
    public OcrViewModel(IModelStore store, IVisionReaderService readers, IPermissionService permissions, IShareService share, IdrakMauiOptions options, ISettingsStore settings,
        OcrLabOptions? defaults = null, IDispatcher? dispatcher = null, IClipboard? clipboard = null, IMediaPicker? picker = null, TimeProvider? time = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _readers = readers ?? throw new ArgumentNullException(nameof(readers));
        _permissions = permissions ?? throw new ArgumentNullException(nameof(permissions));
        _share = share ?? throw new ArgumentNullException(nameof(share));
        _app = options ?? throw new ArgumentNullException(nameof(options));
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        _dispatcher = dispatcher;
        _clipboard = clipboard;
        _picker = picker;
        _time = time ?? TimeProvider.System;
        _options = OcrSettings.Load(settings, out var skipped, defaults);
        _pipeline = _options.Pipeline ?? "";
        if (skipped.Count > 0)
        {
            _settingsMessage = "A saved setting was not read: " + string.Join("; ", skipped);
        }

        foreach (var setting in OcrSettings.Definitions)
        {
            Settings.Add(new LabSettingEntry(setting, setting.Get(_options)));
        }

        TakePhotoCommand = new Command(async () => await TakePhotoAsync(camera: true));
        PickPhotoCommand = new Command(async () => await TakePhotoAsync(camera: false));
        PrepareCommand = new Command(async () => await PrepareAsync());
        ReadCommand = new Command(async () => await ReadAsync());
        StopCommand = new Command(Stop);
        CopyCommand = new Command(async () => await CopyAsync());
        ShareCommand = new Command(async () => await ShareAsync());
        OpenPermissionSettingsCommand = new Command(() => _permissions.OpenSettings());
        GoToModelsCommand = new Command(async () => await (OpenModels?.Invoke() ?? Task.CompletedTask));
        RefreshCommand = new Command(async () => await RefreshAsync());
        ApplySettingsCommand = new Command(() => ApplySettings());
        ToggleSettingsCommand = new Command(() => ShowSettings = !ShowSettings);
    }

    /// <summary>The page's title.</summary>
    public string Title => "Read text";

    /// <summary>The settings in effect.</summary>
    public OcrLabOptions Options => _options;

    /// <summary>The store's complete models a registered vision family reads.</summary>
    public ObservableCollection<VisionModelEntry> VisionModels { get; } = [];

    /// <summary>Whether there is a reader to pick.</summary>
    public bool HasVisionModels => VisionModels.Count > 0;

    /// <summary>Whether there is none (the page offers the Models page).</summary>
    public bool HasNoVisionModels => VisionModels.Count == 0;

    /// <summary>The readers as a line: how many, or why none.</summary>
    public string ModelsText
    {
        get => _modelsText;
        private set => SetProperty(ref _modelsText, value);
    }

    /// <summary>The reader picked; picking one keeps it in the settings and loads it (once) on the settings' device.</summary>
    public VisionModelEntry? SelectedModel
    {
        get => _selected;
        set
        {
            if (SetProperty(ref _selected, value) && value is not null)
            {
                _options.Model = value.Key;
                Keep();
                _ = LoadModelAsync();
            }
        }
    }

    /// <summary>The reader loaded, as the service reported it, or null.</summary>
    public VisionReaderInfo? Model
    {
        get => _info;
        private set => SetProperty(ref _info, value);
    }

    /// <summary>The reader's state: loading, or loaded on which device, in which formats, in how long, as measured; or "".</summary>
    public string ModelStatus
    {
        get => _modelStatus;
        private set => SetProperty(ref _modelStatus, value);
    }

    /// <summary>The photo: its name, format, size, orientation and read time; or "".</summary>
    public string PhotoText
    {
        get => _photoText;
        private set => SetProperty(ref _photoText, value);
    }

    /// <summary>The page as the model sees it: its size and the pipeline that made it; or "".</summary>
    public string PageText
    {
        get => _pageText;
        private set => SetProperty(ref _pageText, value);
    }

    /// <summary>The page as the model will see it (the photo through the pipeline), shrunk for the screen and made once; or null.</summary>
    public ImageSource? PageSource
    {
        get => _pageSource;
        private set
        {
            if (SetProperty(ref _pageSource, value))
            {
                OnPropertyChanged(nameof(HasPage));
                OnPropertyChanged(nameof(HasNoPage));
            }
        }
    }

    /// <summary>Whether a page is ready to read.</summary>
    public bool HasPage => PageSource is not null;

    /// <summary>Whether none is.</summary>
    public bool HasNoPage => !HasPage;

    /// <summary>The page as the model will read it (decoded once, through the pipeline once), or null.</summary>
    public ImageData? Prepared => _prepared;

    /// <summary>The photo opened, or null.</summary>
    public Photo? Photo => _photo;

    /// <summary>The preprocessing pipeline's text (kept in the settings when the page is prepared).</summary>
    public string Pipeline
    {
        get => _pipeline;
        set => SetProperty(ref _pipeline, value ?? "");
    }

    /// <summary>The answer so far.</summary>
    public string Text
    {
        get => _text;
        private set
        {
            if (SetProperty(ref _text, value))
            {
                OnPropertyChanged(nameof(HasText));
                OnPropertyChanged(nameof(HasNoText));
            }
        }
    }

    /// <summary>Whether there is text.</summary>
    public bool HasText => Text.Length > 0;

    /// <summary>Whether there is none yet.</summary>
    public bool HasNoText => !HasText;

    /// <summary>The read streaming: its tokens so far and their rate ("38 tokens · 12.4 tokens/s"), then its figures; or "".</summary>
    public string LiveText
    {
        get => _liveText;
        private set => SetProperty(ref _liveText, value);
    }

    /// <summary>The first read after the load (cold), or "".</summary>
    public string ColdText
    {
        get => _coldText;
        private set
        {
            if (SetProperty(ref _coldText, value))
            {
                OnPropertyChanged(nameof(HasCold));
            }
        }
    }

    /// <summary>Whether <see cref="ColdText"/> has something to show.</summary>
    public bool HasCold => ColdText.Length > 0;

    /// <summary>The warm reads since the load (the last and the mean), or "".</summary>
    public string WarmText
    {
        get => _warmText;
        private set
        {
            if (SetProperty(ref _warmText, value))
            {
                OnPropertyChanged(nameof(HasWarm));
            }
        }
    }

    /// <summary>Whether <see cref="WarmText"/> has something to show.</summary>
    public bool HasWarm => WarmText.Length > 0;

    /// <summary>Whether a photo is being read or prepared, or a model loaded.</summary>
    public bool IsBusy
    {
        get => _busy;
        private set
        {
            if (SetProperty(ref _busy, value))
            {
                OnPropertyChanged(nameof(IsIdle));
            }
        }
    }

    /// <summary>Whether nothing runs.</summary>
    public bool IsIdle => !IsBusy && !IsReading;

    /// <summary>Whether a page is being read (the page shows Stop).</summary>
    public bool IsReading
    {
        get => _reading;
        private set
        {
            if (SetProperty(ref _reading, value))
            {
                OnPropertyChanged(nameof(IsIdle));
                OnPropertyChanged(nameof(IsNotReading));
            }
        }
    }

    /// <summary>Whether no page is being read (the page shows Read).</summary>
    public bool IsNotReading => !IsReading;

    /// <summary>What the last action did, or "".</summary>
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

    /// <summary>Why the last action failed, with its cause, or "".</summary>
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

    /// <summary>Whether the camera was refused (the page offers the app's system settings).</summary>
    public bool IsPermissionRefused
    {
        get => _refused;
        private set => SetProperty(ref _refused, value);
    }

    /// <summary>The settings on the page (empty: the library's default, shown in the field).</summary>
    public ObservableCollection<LabSettingEntry> Settings { get; } = [];

    /// <summary>Whether the settings card is open.</summary>
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
    public string SettingsButtonText => ShowSettings ? "Hide settings" : "Settings";

    /// <summary>What the last <see cref="ApplySettings"/> said, or "".</summary>
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

    /// <summary>The instruction sent with the page (the setting's, else the library's).</summary>
    public string Prompt => _options.Prompt ?? OcrSettings.LibraryPrompt;

    /// <summary>The formats the library reads, for the page.</summary>
    public string Formats => $"The library's own codecs read {Photos.Readable()} (not HEIC).";

    /// <summary>How many times the streaming text was handed to the UI thread (each a UI update), for the batching's measure.</summary>
    public int UiUpdates { get; private set; }

    /// <summary>Opens the Models page (the page sets it: a push of the "models" feature).</summary>
    public Func<Task>? OpenModels { get; set; }

    /// <summary>Takes a photo with the camera (the permission asked first).</summary>
    public ICommand TakePhotoCommand { get; }

    /// <summary>Picks a photo from the gallery.</summary>
    public ICommand PickPhotoCommand { get; }

    /// <summary>Runs the pipeline on the photo and shows the page as the model will see it.</summary>
    public ICommand PrepareCommand { get; }

    /// <summary>Reads the page.</summary>
    public ICommand ReadCommand { get; }

    /// <summary>Stops the read (what came is kept).</summary>
    public ICommand StopCommand { get; }

    /// <summary>Copies the text.</summary>
    public ICommand CopyCommand { get; }

    /// <summary>Shares the text as a file.</summary>
    public ICommand ShareCommand { get; }

    /// <summary>Opens the app's page in the system settings (a refused camera is allowed there).</summary>
    public ICommand OpenPermissionSettingsCommand { get; }

    /// <summary>Opens the Models page (<see cref="OpenModels"/>).</summary>
    public ICommand GoToModelsCommand { get; }

    /// <summary>Reads the store's readers again.</summary>
    public ICommand RefreshCommand { get; }

    /// <summary>Checks and keeps the settings.</summary>
    public ICommand ApplySettingsCommand { get; }

    /// <summary>Opens or closes the settings card.</summary>
    public ICommand ToggleSettingsCommand { get; }

    /// <summary>
    /// Reads the store's models on a worker thread and lists the complete ones whose config.json names an architecture a
    /// registered vision family reads; the one kept in the settings is picked again (else the first).
    /// </summary>
    public async Task RefreshAsync()
    {
        if (_listing)
        {
            return;
        }

        _listing = true;
        try
        {
            var (found, others) = await Task.Run(() => ListVisionModels(_store)).ConfigureAwait(true);
            if (Lists.Replace(VisionModels, found, static (a, b) => a.Model == b.Model && a.Architecture == b.Architecture))
            {
                OnPropertyChanged(nameof(HasVisionModels));
                OnPropertyChanged(nameof(HasNoVisionModels));
            }

            var vision = VisionModels;
            var families = VisionFamilies.Names;
            ModelsText = vision.Count > 0 ? $"{vision.Count} vision-language model{(vision.Count == 1 ? "" : "s")} on this phone."
                : families.Count == 0 ? "No vision family is registered: the app registers the families it shows (a vision plug-in's own Register call)."
                : $"No vision-language model on this phone{(others > 0 ? $" among the {others} other model{(others == 1 ? "" : "s")}" : "")}: pull one in Models, "
                  + $"of an architecture registered here ({string.Join(", ", families)}).";
            var keep = vision.FirstOrDefault(m => string.Equals(m.Key, _options.Model, StringComparison.OrdinalIgnoreCase))
                ?? vision.FirstOrDefault(m => _selected is { } current && current.Key == m.Key)
                ?? vision.FirstOrDefault();
            if (keep is not null && (_selected is null || keep.Key != _selected.Key))
            {
                SelectedModel = keep;
            }
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = $"The models could not be read from {_store.Folder}: {e.Message}";
        }
        finally
        {
            _listing = false;
        }
    }

    /// <summary>
    /// Loads the picked reader through the service on the settings' device (once: a later call with the same model and
    /// settings gives the same load), measured first; a model that does not fit, or whose family is not registered, is
    /// refused and named. True when it is loaded.
    /// </summary>
    public Task<bool> LoadModelAsync()
    {
        if (SelectedModel is not { } model)
        {
            Error = NoModel();
            return Task.FromResult(false);
        }

        var request = LoadRequest(model);
        if (_loading is { } running && _loadingKey == request.Key && !(running.IsCompleted && !running.Result))
        {
            return running;
        }

        _loadingKey = request.Key;
        return _loading = LoadAsync(model, request);
    }

    /// <summary>A photo from the camera or the gallery, opened and prepared (<see cref="OpenAsync"/>); false when none came or it failed.</summary>
    public async Task<bool> TakePhotoAsync(bool camera)
    {
        if (!IsIdle)
        {
            return false;
        }

        Error = "";
        IsPermissionRefused = false;
        (ChatImage File, string Name)? taken;
        try
        {
            taken = await Photos.TakeAsync(_picker ?? MediaPicker.Default, _permissions, camera).ConfigureAwait(true);
        }
        catch (PhotoRefusedException e)
        {
            IsPermissionRefused = e.PermissionRefused;
            Error = e.Message;
            return false;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or InvalidDataException)
        {
            Error = $"The photo could not be read: {e.Message}";
            return false;
        }

        return taken is { } photo && await OpenAsync(photo.File, photo.Name).ConfigureAwait(true);
    }

    /// <summary>
    /// Opens <paramref name="file"/>: decoded once by the library's codecs and turned upright on a worker thread (HEIC refused,
    /// naming the formats the library reads), then prepared with the pipeline (<see cref="PrepareAsync"/>). True when it is ready to read.
    /// </summary>
    public async Task<bool> OpenAsync(ChatImage file, string name)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(name);
        if (!IsIdle)
        {
            return false;
        }

        IsBusy = true;
        Error = "";
        Status = $"Reading {name}…";
        try
        {
            _photo = await Task.Run(() => Photos.Decode(file, name)).ConfigureAwait(true);
            _prepared = null;
            _preparedFor = null;
            PageSource = null;
            PhotoText = _photo.Describe();
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException)
        {
            Status = "";
            Error = e.Message;
            return false;
        }
        finally
        {
            IsBusy = false;
        }

        return await PrepareAsync().ConfigureAwait(true);
    }

    /// <summary>
    /// Runs the pipeline's text on the decoded photo once (never decoding it again; the photo itself when the pipeline is
    /// empty) and shows the page as the model will see it, its picture shrunk for the screen and made once; the pipeline is
    /// kept in the settings. Nothing runs when the page is prepared with this pipeline already. True when the page is ready.
    /// </summary>
    public async Task<bool> PrepareAsync()
    {
        if (_photo is not { } photo)
        {
            Error = "Take or pick a photo first.";
            return false;
        }

        ImageTransformPipeline pipeline;
        try
        {
            pipeline = ImageTransformPipeline.Parse(Pipeline);
        }
        catch (ArgumentException e)
        {
            Error = e.Message;
            return false;
        }

        string key = pipeline.ToString();
        if (_prepared is not null && _preparedFor == key)
        {
            return true;                                                        // done once already
        }

        bool wasBusy = IsBusy;
        IsBusy = true;
        try
        {
            int? side = _options.PreviewPixels;
            long start = _time.GetTimestamp();
            var (page, preview) = await Task.Run(() =>
            {
                var prepared = pipeline.IsEmpty ? photo.Pixels : pipeline.Apply(photo.Pixels);
                return (prepared, Photos.Preview(prepared, side));
            }).ConfigureAwait(true);
            var took = _time.GetElapsedTime(start);
            _prepared = page;
            _preparedFor = key;
            _options.Pipeline = pipeline.IsEmpty ? null : key;
            Keep();
            Settings.FirstOrDefault(e => e.Key == "pipeline")?.Text = _options.Pipeline ?? "";
            PageSource = preview.Source();
            PageText = string.Create(CultureInfo.InvariantCulture, $"The page as the model sees it: {page.Width:N0} x {page.Height:N0}, {(page.Channels == 1 ? "grey" : "colour")}")
                + (pipeline.IsEmpty ? ", as decoded" : $", after {key} ({Photos.Milliseconds(took)})");
            Status = $"{photo.Name} is ready to read.";
            return true;
        }
        catch (Exception e) when (e is ArgumentException or InvalidDataException or NotSupportedException)
        {
            Error = $"The pipeline stopped: {e.Message}";
            return false;
        }
        finally
        {
            IsBusy = wasBusy;
        }
    }

    /// <summary>
    /// Reads the page: the reader loaded (measured first), the page prepared (once), handed over as it is and the answer
    /// streamed on a worker thread, the text shown at most once a progress interval; the first read after a load is cold.
    /// True when it ended (or was stopped) with an answer.
    /// </summary>
    public async Task<bool> ReadAsync()
    {
        if (!IsIdle)
        {
            return false;
        }

        Error = "";
        if (SelectedModel is not { } model)
        {
            Error = NoModel();
            return false;
        }

        if (_photo is not { } photo || !await PrepareAsync().ConfigureAwait(true) || _prepared is not { } prepared)
        {
            if (_photo is null)
            {
                Error = "Take or pick a photo first.";
            }

            return false;
        }

        ImageReadRequest request;
        try
        {
            request = _options.ToRequest(photo.File) with { Transforms = ImageTransformPipeline.Parse(_preparedFor) };
        }
        catch (ArgumentException e)
        {
            Error = e.Message;
            return false;
        }

        var load = LoadRequest(model);
        using var cancellation = new CancellationTokenSource();
        _cancellation = cancellation;
        IsReading = true;
        Text = "";
        LiveText = "";
        Status = $"Reading {photo.Name} with {model.Title}…";
        try
        {
            var read = await Task.Run(() => Stream(load, request, prepared, cancellation.Token), CancellationToken.None).ConfigureAwait(true);
            if (read.Info is { } loaded)
            {
                Show(loaded, loaded: true);                                     // memory pressure had taken it: loaded again for this read,
                _readsSinceLoad = 1;                                            // which is the cold one
            }

            Report(read);
            return true;
        }
        catch (Exception e)
        {
            Status = "";
            Error = ChatViewModel.Explain(e, model.Title);
            return false;
        }
        finally
        {
            _cancellation = null;
            IsReading = false;
        }
    }

    /// <summary>Stops the read: what came so far is kept as the text.</summary>
    public void Stop()
    {
        try
        {
            _cancellation?.Cancel();
        }
        catch (ObjectDisposedException)
        {
            // The read ended meanwhile.
        }
    }

    /// <summary>Copies the text to the clipboard; false when there is none.</summary>
    public async Task<bool> CopyAsync()
    {
        if (Text.Length == 0)
        {
            Status = "Nothing to copy yet.";
            return false;
        }

        try
        {
            await (_clipboard ?? Clipboard.Default).SetTextAsync(Text).ConfigureAwait(true);
            Status = $"Copied {Text.Length:N0} characters.";
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The text was not copied: {e.Message}";
            return false;
        }
    }

    /// <summary>Writes the text to a file in the cache folder and offers it to the phone's share sheet; false when there is none.</summary>
    public async Task<bool> ShareAsync()
    {
        if (Text.Length == 0)
        {
            Status = "Nothing to share yet.";
            return false;
        }

        try
        {
            string folder = Path.Combine(ComputeResources.CacheFolder, "ocr");
            Directory.CreateDirectory(folder);
            string stem = Path.GetFileNameWithoutExtension(_photo?.Name ?? "page") is { Length: > 0 } s ? s : "page";
            string path = Path.Combine(folder, $"{stem}-{_time.GetLocalNow():yyyyMMdd-HHmmss}.txt");
            await File.WriteAllTextAsync(path, Text + "\n", new UTF8Encoding(false)).ConfigureAwait(true);
            await _share.ShareFileAsync(path, "Text read", "text/plain").ConfigureAwait(true);
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The text was not shared: {e.Message}";
            return false;
        }
    }

    /// <summary>
    /// Checks every setting on the page and keeps them; false, and nothing kept, when one is refused (each says why). The
    /// pipeline applies at the next prepare, the formats and the context at the next load.
    /// </summary>
    public bool ApplySettings()
    {
        if (LabSettingEntry.Apply(Settings, _options.Clone()) is not { } next)
        {
            SettingsMessage = "Nothing was kept: a value above is not one the setting takes.";
            return false;
        }

        next.Model = _options.Model;
        _options = next;
        Pipeline = next.Pipeline ?? "";
        Keep();
        OnPropertyChanged(nameof(Prompt));
        SettingsMessage = "Kept: the pipeline applies when the page is prepared again, the formats and the context at the next load.";
        return true;
    }

    // The store's complete models a registered vision family reads, and how many others there are.
    internal static (List<VisionModelEntry> Vision, int Others) ListVisionModels(IModelStore store)
    {
        var vision = new List<VisionModelEntry>();
        int others = 0;
        foreach (var model in store.List())
        {
            if (model.Complete && Directory.Exists(model.Folder) && VisionReaderService.VisionArchitecture(model.Folder) is { } architecture)
            {
                vision.Add(new VisionModelEntry(model, architecture));
            }
            else
            {
                others++;
            }
        }

        return (vision, others);
    }

    private VisionLoadRequest LoadRequest(VisionModelEntry model) => _options.ToLoad(model.Model.Id, model.Model.Folder, _app.Device);

    private string NoModel() => VisionModels.Count == 0 ? "No vision-language model to read with: pull one in Models, then come back." : "Pick a reader first.";

    private void Keep() => OcrSettings.Save(_settings, _options);

    private async Task<bool> LoadAsync(VisionModelEntry model, VisionLoadRequest request)
    {
        Error = "";
        Model = null;
        ModelStatus = $"Loading {model.Title} on {request.Device}…";
        IsBusy = true;
        try
        {
            using var lease = await _readers.AcquireAsync(request).ConfigureAwait(true);
            Show(lease.Info, lease.Loaded);
            return true;
        }
        catch (Exception e)
        {
            ModelStatus = $"{model.Title} is not loaded.";
            Error = ChatViewModel.Explain(e, model.Title);
            return false;
        }
        finally
        {
            IsBusy = false;
        }
    }

    private void Show(VisionReaderInfo info, bool loaded)
    {
        Model = info;
        if (loaded)
        {
            _readsSinceLoad = 0;
            _warmRates.Clear();
            ColdText = "";
            WarmText = "";
        }

        ModelStatus = string.Create(CultureInfo.InvariantCulture,
            $"{info.Model} ({info.Architecture}) on {info.Device} · weights {info.Weights} · cache {info.CacheFormat.ToString().ToLowerInvariant()} · {info.Context:N0} tokens · loaded in {info.LoadTime.TotalSeconds:F1} s")
            + (info.Estimate is { } e ? $"\nMeasured before loading: {e}" : "");
    }

    // On a worker thread: the reader acquired, the prepared page handed over, the answer's pieces gathered and handed to
    // the UI thread at most once a progress interval.
    private Reading Stream(VisionLoadRequest load, ImageReadRequest request, ImageData prepared, CancellationToken cancellationToken)
    {
        using var lease = _readers.AcquireAsync(load, cancellationToken).GetAwaiter().GetResult();
        bool cold = lease.Loaded || _readsSinceLoad == 0;
        lock (_pending)
        {
            _pendingText.Clear();
            _flushQueued = false;
            _lastFlush = 0;
            _generated = 0;
        }

        lease.Reader.Keep(request, prepared);
        long start = _time.GetTimestamp(), first = 0;
        int firstGenerated = 0;
        ChatChunk? final = null;
        var content = new StringBuilder();
        bool stopped = false;
        try
        {
            foreach (var chunk in lease.Reader.Stream(request, cancellationToken))
            {
                content.Append(chunk.Delta.Content);
                lock (_pending)
                {
                    _pendingText.Append(chunk.Delta.Content);
                    _generated = Math.Max(_generated, chunk.Generated);
                }

                if (first == 0 && chunk.Delta.Content.Length > 0)
                {
                    (first, firstGenerated) = (_time.GetTimestamp(), chunk.Generated);
                }

                if (chunk.Done)
                {
                    final = chunk;
                }
                else
                {
                    Flush(first, firstGenerated, final: false);
                }
            }
        }
        catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
        {
            stopped = true;
        }

        stopped |= cancellationToken.IsCancellationRequested;
        Flush(first, firstGenerated, final: true);
        _readsSinceLoad++;
        string text = (final?.Message?.Content ?? content.ToString()).Trim();
        return new Reading(text, final?.Stats, final?.Generated ?? _generated, final?.DoneReason, cold, stopped, _time.GetElapsedTime(start), lease.Loaded ? lease.Info : null);
    }

    // Hands what came since the last showing to the UI thread: at once for the first piece and the last, else once the
    // progress interval has passed and the previous showing was taken (a later one carries all that came meanwhile).
    private void Flush(long first, int firstGenerated, bool final)
    {
        long now = _time.GetTimestamp();
        lock (_pending)
        {
            if (!final && (_flushQueued || _lastFlush != 0 && _time.GetElapsedTime(_lastFlush, now) < _app.ProgressInterval))
            {
                return;
            }

            _flushQueued = true;
            _lastFlush = now;
        }

        UiUpdates++;
        Post(() =>
        {
            string text;
            int generated;
            lock (_pending)
            {
                (text, generated) = (_pendingText.ToString(), _generated);
                _pendingText.Clear();
                _flushQueued = false;
            }

            if (text.Length > 0)
            {
                Text += text;
            }

            double seconds = first == 0 ? 0 : _time.GetElapsedTime(first, now).TotalSeconds;
            LiveText = generated == 0 ? "" : string.Create(CultureInfo.InvariantCulture, $"{generated:N0} token{(generated == 1 ? "" : "s")}")
                + (seconds > 0 && generated > firstGenerated ? string.Create(CultureInfo.InvariantCulture, $" · {(generated - firstGenerated) / seconds:F1} tokens/s") : "");
        });
    }

    // After the read: the text as the reader finished it, the live line's last word, the cold or warm line.
    private void Report(Reading read)
    {
        var stats = read.Stats;
        int tokens = stats?.GeneratedTokens ?? read.Generated;
        double rate = stats?.TokensPerSecond ?? 0;
        string speed = string.Create(CultureInfo.InvariantCulture, $"{tokens:N0} token{(tokens == 1 ? "" : "s")}")
            + (rate > 0 ? string.Create(CultureInfo.InvariantCulture, $" · {rate:F1} tokens/s") : "")
            + (stats is { } s ? string.Create(CultureInfo.InvariantCulture, $" · prompt {s.PromptTokens:N0} tokens (the page's included) in {s.PromptDuration.TotalSeconds:F2} s") : "")
            + $" · {Photos.Milliseconds(read.Elapsed)} in all";
        string tail = (read.Cold ? " · cold" : " · warm") + (read.Stopped ? " · stopped" : read.DoneReason is "length" ? " · stopped at the token limit" : read.DoneReason is "repetition" ? " · stopped repeating" : "");
        Post(() =>
        {
            if (!read.Stopped && read.Text.Length > 0)
            {
                Text = read.Text;                                               // the reader's own answer, trimmed
            }

            LiveText = speed + tail;
        });
        Status = read.Stopped ? "Stopped: what came so far is kept." : "Read.";
        if (read.Cold)
        {
            ColdText = "First read after loading (cold: compiling methods and kernels): " + speed;
        }
        else if (rate > 0)
        {
            _warmRates.Add(rate);
            WarmText = string.Create(CultureInfo.InvariantCulture,
                $"Warm: {rate:F1} tokens/s last, {_warmRates.Average():F1} mean over {_warmRates.Count} read{(_warmRates.Count == 1 ? "" : "s")}");
        }
    }

    private void Post(Action action)
    {
        var dispatcher = _dispatcher ?? Application.Current?.Dispatcher;
        if (dispatcher is null || !dispatcher.IsDispatchRequired)
        {
            action();
        }
        else
        {
            dispatcher.Dispatch(action);
        }
    }

    private sealed record Reading(string Text, GenerationStats? Stats, int Generated, string? DoneReason, bool Cold, bool Stopped, TimeSpan Elapsed, VisionReaderInfo? Info);
}
