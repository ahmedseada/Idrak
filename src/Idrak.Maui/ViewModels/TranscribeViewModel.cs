// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows.Input;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;
using Microsoft.Maui.ApplicationModel.DataTransfer;

namespace Idrak.Maui.ViewModels;

/// <summary>A speech model of the model store, as the Transcribe page lists it.</summary>
public sealed class SpeechModelEntry
{
    internal SpeechModelEntry(StoredModel model, string architecture)
    {
        Model = model;
        Architecture = architecture;
    }

    /// <summary>The stored model.</summary>
    public StoredModel Model { get; }

    /// <summary>The architecture its config.json names, a registered <see cref="SpeechFamilies"/> name.</summary>
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

/// <summary>One Transcribe setting on the page: its title, help and value; an empty value is the library's default (its placeholder).</summary>
public sealed class TranscribeSettingEntry : ViewModel
{
    private string _text;
    private string _error = "";

    internal TranscribeSettingEntry(TranscribeSetting setting, string text)
    {
        Setting = setting;
        _text = text;
        Choices = setting.Choices;
    }

    /// <summary>The setting.</summary>
    public TranscribeSetting Setting { get; }

    /// <summary>Its title.</summary>
    public string Title => Setting.Title;

    /// <summary>Its help.</summary>
    public string Summary => Setting.Summary;

    /// <summary>The library's default, shown in an empty field.</summary>
    public string Placeholder => Setting.Placeholder;

    /// <summary>Edited as text (a field).</summary>
    public bool IsEntry => Setting.Kind != TranscribeSettingKind.Choice;

    /// <summary>Edited as a choice (a picker).</summary>
    public bool IsChoice => Setting.Kind == TranscribeSettingKind.Choice;

    /// <summary>The keyboard for the field.</summary>
    public Keyboard Keyboard => Setting.Kind == TranscribeSettingKind.Number ? Keyboard.Numeric : Keyboard.Plain;

    /// <summary>The values to choose from, the first standing for the library's default.</summary>
    public IReadOnlyList<string> Choices { get; }

    /// <summary>The value as text ("" for the library's default).</summary>
    public string Text
    {
        get => _text;
        set
        {
            if (SetProperty(ref _text, value ?? ""))
            {
                OnPropertyChanged(nameof(SelectedChoice));
            }
        }
    }

    /// <summary>The value as a choice (the first choice standing for "").</summary>
    public string? SelectedChoice
    {
        get => Text.Length == 0 && Choices.Count > 0 ? Choices[0] : Text;
        set => Text = value is null || Choices.Count > 0 && value == Choices[0] ? "" : value;
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

/// <summary>
/// The Transcribe page (plan 34, M5): live transcription from the microphone through the library's
/// <see cref="LiveTranscriber"/>, or a file decoded by the library's own codecs (<see cref="AudioCodecs"/>) through the same
/// live path or whole. A speech model is picked from the model store's complete models whose architecture a registered
/// <see cref="SpeechFamilies"/> family reads, and loaded once through the <see cref="IModelService"/> (the host
/// <see cref="ModelHostName"/>) on the settings' device. The microphone permission is asked first
/// (<see cref="IPermissionService"/>; a refusal is shown with how to allow it); the sound is captured through
/// <see cref="AudioCaptures"/> (<see cref="AudioCaptures.ForThisMachine"/> by default: aaudio on Android), cleaned by the
/// optional live pipeline (<see cref="TranscribeLabOptions.Pipeline"/>), metered and transcribed on a worker thread. The page
/// is updated at most once a progress interval (<see cref="ProgressCenter.Interval"/>): each audio block and each update
/// stores its values and only a showing that is due is handed to the UI thread, one at a time; the last values are shown
/// when the run ends. Stopping ends the input, so the last segment is read and committed.
/// </summary>
public sealed class TranscribeViewModel : ViewModel, IDisposable
{
    /// <summary>The <see cref="IModelService"/> host speech models are kept in.</summary>
    public const string ModelHostName = "speech";

    private const long Never = long.MinValue;
    private const string TranscribeTask = "transcribe", TranslateTask = "translate", AutoLanguage = "auto";

    private readonly IModelStore _store;
    private readonly IModelService _models;
    private readonly IPermissionService _permissions;
    private readonly IShareService _share;
    private readonly IdrakMauiOptions _app;
    private readonly ISettingsStore _settings;
    private readonly ProgressCenter _progress;
    private readonly IClipboard? _clipboard;
    private readonly IFilePicker? _picker;
    private readonly TimeProvider _time;
    private readonly LiveState _state = new();
    private readonly Action _publish;
    private readonly Lock _gate = new();
    private readonly Lock _applyGate = new();
    private readonly CancellationTokenSource _abort = new();
    private TranscribeLabOptions _options;
    private SpeechModelEntry? _selected;
    private Task<bool>? _loading;
    private string? _loadingKey;
    private string? _loadedKey;
    private CancellationTokenSource? _stop;
    private LiveSnapshot? _pending;
    private ITimer? _trailing;
    private long _last = Never;
    private long _applied;
    private int _scheduled;
    private volatile bool _live;
    private double _step = 1;
    private bool _running, _recording, _listing, _refused, _showSettings;
    private string _status = "", _error = "", _modelStatus = "", _modelsText = "", _settingsMessage = "";
    private string _committed = "", _tentative = "", _elapsed = "00:00", _heard = "", _behindText = "";
    private string _language = AutoLanguage, _task = TranscribeTask;
    private double _level, _behind, _decibels = double.NegativeInfinity;
    private ProgressItem? _work;

    /// <summary>The page over the app's services; <paramref name="clipboard"/> and <paramref name="picker"/> are MAUI's own when null.</summary>
    public TranscribeViewModel(IModelStore store, IModelService models, IPermissionService permissions, IShareService share, IdrakMauiOptions options, ISettingsStore settings,
        ProgressCenter? progress = null, IClipboard? clipboard = null, IFilePicker? picker = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _models = models ?? throw new ArgumentNullException(nameof(models));
        _permissions = permissions ?? throw new ArgumentNullException(nameof(permissions));
        _share = share ?? throw new ArgumentNullException(nameof(share));
        _app = options ?? throw new ArgumentNullException(nameof(options));
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        _progress = progress ?? new ProgressCenter(options.ProgressInterval);
        _clipboard = clipboard;
        _picker = picker;
        _time = _progress.Time;
        _publish = Publish;
        _options = TranscribeSettings.Load(settings, out var skipped);
        _language = _options.Language ?? AutoLanguage;
        _task = _options.Task ?? TranscribeTask;
        if (skipped.Count > 0)
        {
            _settingsMessage = "A saved setting was not read: " + string.Join("; ", skipped);
        }

        foreach (var setting in TranscribeSettings.Definitions)
        {
            Settings.Add(new TranscribeSettingEntry(setting, setting.Get(_options)));
        }

        RecordCommand = new Command(async () =>
        {
            if (IsRecording)
            {
                Stop();
            }
            else
            {
                await RecordAsync();
            }
        });
        StopCommand = new Command(Stop);
        TranscribeFileCommand = new Command(async () => await PickFileAsync());
        CopyCommand = new Command(async () => await CopyAsync());
        ShareCommand = new Command(async () => await ShareAsync());
        ClearCommand = new Command(Clear);
        OpenPermissionSettingsCommand = new Command(() => _permissions.OpenSettings());
        GoToModelsCommand = new Command(async () =>
        {
            if (OpenModels is { } open)
            {
                await open();
            }
        });
        RefreshCommand = new Command(async () => await RefreshAsync());
        ApplySettingsCommand = new Command(() => ApplySettings());
        ToggleSettingsCommand = new Command(() => ShowSettings = !ShowSettings);
    }

    /// <summary>The page's title.</summary>
    public string Title => "Transcribe";

    /// <summary>The settings in effect (a value changed here applies from the next run; <see cref="ApplySettings"/> checks and keeps the page's).</summary>
    public TranscribeLabOptions Options => _options;

    /// <summary>The speech models the store holds complete, whose architecture a registered family reads.</summary>
    public ObservableCollection<SpeechModelEntry> SpeechModels { get; } = [];

    /// <summary>Whether there is a speech model to pick.</summary>
    public bool HasSpeechModels => SpeechModels.Count > 0;

    /// <summary>Whether there is none (the page offers the Models page).</summary>
    public bool HasNoSpeechModels => SpeechModels.Count == 0;

    /// <summary>The speech models as a line: how many, or why none.</summary>
    public string ModelsText
    {
        get => _modelsText;
        private set => SetProperty(ref _modelsText, value);
    }

    /// <summary>The model picked; picking one keeps it in the settings and loads it (once) on the settings' device.</summary>
    public SpeechModelEntry? SelectedModel
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

    /// <summary>The model's state: loading, loaded on which device in how long, its languages; or "".</summary>
    public string ModelStatus
    {
        get => _modelStatus;
        private set => SetProperty(ref _modelStatus, value);
    }

    /// <summary>The model loading or the file being read whole, with its time or segments, or null.</summary>
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

    /// <summary>Whether <see cref="Work"/> has something to show.</summary>
    public bool HasWork => Work is not null;

    /// <summary>"auto" and the languages the loaded model knows (empty until a model with languages is loaded).</summary>
    public ObservableCollection<string> Languages { get; } = [];

    /// <summary>Whether the loaded model has languages: the language and the task can be chosen.</summary>
    public bool HasLanguages => Languages.Count > 1;

    /// <summary>The language chosen ("auto": detected by the family).</summary>
    public string? SelectedLanguage
    {
        get => _language;
        set
        {
            if (value is not null && SetProperty(ref _language, value))
            {
                _options.Language = value == AutoLanguage ? null : value;
                Keep();
            }
        }
    }

    /// <summary>The tasks: "transcribe" (the spoken language) and "translate" (into the family's target language; a family may refuse it).</summary>
    public IReadOnlyList<string> Tasks { get; } = [TranscribeTask, TranslateTask];

    /// <summary>The task chosen.</summary>
    public string? SelectedTask
    {
        get => _task;
        set
        {
            if (value is not null && SetProperty(ref _task, value))
            {
                _options.Task = value == TranscribeTask ? null : value;
                Keep();
            }
        }
    }

    /// <summary>Whether a recording or a file is being transcribed.</summary>
    public bool IsRunning
    {
        get => _running;
        private set
        {
            if (SetProperty(ref _running, value))
            {
                OnPropertyChanged(nameof(IsIdle));
            }
        }
    }

    /// <summary>Whether nothing is being transcribed.</summary>
    public bool IsIdle => !IsRunning;

    /// <summary>Whether the microphone is being transcribed (the record button stops it).</summary>
    public bool IsRecording
    {
        get => _recording;
        private set
        {
            if (SetProperty(ref _recording, value))
            {
                OnPropertyChanged(nameof(RecordText));
                OnPropertyChanged(nameof(RecordGlyph));
            }
        }
    }

    /// <summary>The record button's words.</summary>
    public string RecordText => IsRecording ? "Stop" : "Record";

    /// <summary>The record button's mark.</summary>
    public string RecordGlyph => IsRecording ? "■" : "●";

    /// <summary>Whether the models are being read.</summary>
    public bool IsListing
    {
        get => _listing;
        private set => SetProperty(ref _listing, value);
    }

    /// <summary>What is going on or what the last run did, or "".</summary>
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

    /// <summary>Whether the microphone was refused (the page offers the app's system settings).</summary>
    public bool IsPermissionRefused
    {
        get => _refused;
        private set => SetProperty(ref _refused, value);
    }

    /// <summary>The level of the latest block heard, from 0 (at <see cref="MeterFloorDecibels"/> or under) to 1 (full scale).</summary>
    public double Level
    {
        get => _level;
        private set => SetProperty(ref _level, value);
    }

    /// <summary>The level of the latest block heard, in decibels of full scale (negative infinity before any).</summary>
    public double LevelDecibels
    {
        get => _decibels;
        private set
        {
            if (SetProperty(ref _decibels, value))
            {
                OnPropertyChanged(nameof(LevelText));
            }
        }
    }

    /// <summary>The level as text ("-23 dB"), or "" before any sound.</summary>
    public string LevelText => double.IsFinite(LevelDecibels) ? string.Create(CultureInfo.InvariantCulture, $"{LevelDecibels:0} dB") : "";

    /// <summary>The level the meter shows as empty, in decibels of full scale (-60 by default).</summary>
    public double MeterFloorDecibels { get; set; } = -60;

    /// <summary>The time since the run started ("01:23").</summary>
    public string ElapsedText
    {
        get => _elapsed;
        private set => SetProperty(ref _elapsed, value);
    }

    /// <summary>The sound read so far ("12.3 s of sound"), or "".</summary>
    public string HeardText
    {
        get => _heard;
        private set => SetProperty(ref _heard, value);
    }

    /// <summary>Seconds of sound that arrived during the last step and wait for the next (<see cref="LiveTranscript.Behind"/>).</summary>
    public double Behind
    {
        get => _behind;
        private set
        {
            if (SetProperty(ref _behind, value))
            {
                OnPropertyChanged(nameof(IsBehind));
            }
        }
    }

    /// <summary>Whether the model runs a step or more behind the speaker.</summary>
    public bool IsBehind => Behind >= _step;

    /// <summary>How far behind the model runs ("2.0 s behind", "keeping up"), or "".</summary>
    public string BehindText
    {
        get => _behindText;
        private set => SetProperty(ref _behindText, value);
    }

    /// <summary>The words final so far: the finished segments (a line each), then the open segment's committed words.</summary>
    public string CommittedText
    {
        get => _committed;
        private set
        {
            if (SetProperty(ref _committed, value))
            {
                OnPropertyChanged(nameof(HasText));
                OnPropertyChanged(nameof(HasNoText));
                OnPropertyChanged(nameof(Text));
                OnPropertyChanged(nameof(TentativeShown));
            }
        }
    }

    /// <summary>The words heard after the committed ones, which the next step may change (shown lighter).</summary>
    public string TentativeText
    {
        get => _tentative;
        private set
        {
            if (SetProperty(ref _tentative, value))
            {
                OnPropertyChanged(nameof(HasText));
                OnPropertyChanged(nameof(HasNoText));
                OnPropertyChanged(nameof(Text));
                OnPropertyChanged(nameof(TentativeShown));
            }
        }
    }

    /// <summary>The tentative words as shown after the committed ones (a space before them), or "".</summary>
    public string TentativeShown => TentativeText.Length == 0 ? "" : (CommittedText.Length > 0 ? " " : "") + TentativeText;

    /// <summary>Whether there is text.</summary>
    public bool HasText => CommittedText.Length > 0 || TentativeText.Length > 0;

    /// <summary>Whether there is none yet.</summary>
    public bool HasNoText => !HasText;

    /// <summary>The whole text as shown: the committed words, then the tentative ones.</summary>
    public string Text => (CommittedText + TentativeShown).Trim();

    /// <summary>The finished segments with their times, in order.</summary>
    public IReadOnlyList<TranscriptSegment> Segments => _state.Segments();

    /// <summary>The settings on the page (empty: the library's default, shown in the field).</summary>
    public ObservableCollection<TranscribeSettingEntry> Settings { get; } = [];

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

    /// <summary>Opens the Models page (the page sets it: a push of the "models" feature).</summary>
    public Func<Task>? OpenModels { get; set; }

    /// <summary>Records from the microphone, or stops the recording running.</summary>
    public ICommand RecordCommand { get; }

    /// <summary>Stops the recording (its last segment is read and committed) or the file.</summary>
    public ICommand StopCommand { get; }

    /// <summary>Picks an audio file and transcribes it.</summary>
    public ICommand TranscribeFileCommand { get; }

    /// <summary>Copies the text.</summary>
    public ICommand CopyCommand { get; }

    /// <summary>Shares the text as a file.</summary>
    public ICommand ShareCommand { get; }

    /// <summary>Clears the text.</summary>
    public ICommand ClearCommand { get; }

    /// <summary>Opens the app's page in the system settings (a refused microphone is allowed there).</summary>
    public ICommand OpenPermissionSettingsCommand { get; }

    /// <summary>Opens the Models page (<see cref="OpenModels"/>).</summary>
    public ICommand GoToModelsCommand { get; }

    /// <summary>Reads the speech models again.</summary>
    public ICommand RefreshCommand { get; }

    /// <summary>Checks and keeps the settings (<see cref="ApplySettings"/>).</summary>
    public ICommand ApplySettingsCommand { get; }

    /// <summary>Opens or closes the settings card.</summary>
    public ICommand ToggleSettingsCommand { get; }

    /// <summary>
    /// Reads the speech models on a worker thread: the store's complete models whose config.json names an architecture a
    /// registered <see cref="SpeechFamilies"/> family reads. The model kept in the settings is picked again (else the first).
    /// </summary>
    public async Task RefreshAsync()
    {
        if (IsListing)
        {
            return;
        }

        IsListing = true;
        try
        {
            var (speech, others) = await Task.Run(() => ListSpeechModels(_store)).ConfigureAwait(true);
            SpeechModels.Clear();
            foreach (var model in speech)
            {
                SpeechModels.Add(model);
            }

            OnPropertyChanged(nameof(HasSpeechModels));
            OnPropertyChanged(nameof(HasNoSpeechModels));
            var families = SpeechFamilies.Names;
            ModelsText = speech.Count > 0 ? $"{speech.Count} speech model{(speech.Count == 1 ? "" : "s")} on this phone."
                : families.Count == 0 ? "No speech family is registered: the app registers the families it shows (SpeechFamilies.Register, a plug-in's own Register call)."
                : $"No speech model on this phone{(others > 0 ? $" among the {others} other model{(others == 1 ? "" : "s")}" : "")}: pull one in Models, "
                  + $"of an architecture registered here ({string.Join(", ", families)}).";
            var keep = speech.FirstOrDefault(m => string.Equals(m.Key, _options.Model, StringComparison.OrdinalIgnoreCase))
                ?? speech.FirstOrDefault(m => _selected is { } current && current.Key == m.Key)
                ?? speech.FirstOrDefault();
            if (keep is not null && !ReferenceEquals(keep, _selected))
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
            IsListing = false;
        }
    }

    /// <summary>
    /// Loads the picked model through the model service on the settings' device (once: a later call with the same model and
    /// device gives the same load), and lists its languages; true when it is loaded.
    /// </summary>
    public Task<bool> LoadModelAsync()
    {
        if (SelectedModel is not { } model)
        {
            Error = NoModel();
            return Task.FromResult(false);
        }

        string wanted = $"{_app.Device}|{model.Model.Folder}";
        if (_loading is { } running && _loadingKey == wanted && !(running.IsCompleted && !running.Result))
        {
            return running;
        }

        _loadingKey = wanted;
        return _loading = LoadAsync(model);
    }

    /// <summary>
    /// Transcribes the microphone until <see cref="Stop"/>: the permission asked first, the settings and the pipeline checked
    /// (a step that needs the whole recording is refused by name), the model loaded, the capture opened and cleaned, then
    /// <see cref="LiveTranscriber"/> on a worker thread; true when the recording ran to its end (its last segment committed).
    /// </summary>
    public async Task<bool> RecordAsync()
    {
        if (IsRunning)
        {
            return false;
        }

        Error = "";
        IsPermissionRefused = false;
        if (SelectedModel is null)
        {
            Error = NoModel();
            return false;
        }

        var permission = await _permissions.RequestAsync(PhoneNeeds.Microphone).ConfigureAwait(true);
        if (!permission.Granted)
        {
            IsPermissionRefused = true;
            Error = permission.Message.Length > 0 ? permission.Message
                : "The app may not use the microphone. Allow it in the app's settings (Permissions), then try again.";
            return false;
        }

        AudioPipeline pipeline;
        LiveTranscriptionOptions live;
        IAudioCapture capture;
        try
        {
            pipeline = _options.ToPipeline();
            live = Checked(_options.ToLive(microphone: true));
            capture = _options.Capture is { Length: > 0 } name ? AudioCaptures.Get(name) : AudioCaptures.ForThisMachine();
            if (!capture.IsAvailable(out string reason))
            {
                throw new NotSupportedException($"The {capture.Name} capture cannot record here: {reason}.");
            }
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            Error = e.Message;
            return false;
        }

        if (!await LoadModelAsync().ConfigureAwait(true) || _loadedKey is not { } key)
        {
            return false;
        }

        IAudioInput input;
        try
        {
            var source = capture.Open(_options.ToCapture());
            try
            {
                input = pipeline.Wrap(source);
            }
            catch
            {
                source.Dispose();
                throw;
            }
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The microphone could not be opened ({capture.Name}): {e.Message}";
            return false;
        }

        return await RunLiveAsync(input, live, key, $"Listening: {input.Description}, {input.SampleRate} Hz", microphone: true).ConfigureAwait(true);
    }

    /// <summary>Stops what runs: the microphone's input ends (its last segment is read and committed), or the file's transcription is cancelled.</summary>
    public void Stop()
    {
        try
        {
            if (_stop is { IsCancellationRequested: false } stop)
            {
                stop.Cancel();
                Status = IsRecording ? "Finishing the last segment…" : "Stopping…";
            }
        }
        catch (ObjectDisposedException)
        {
            // The run ended meanwhile.
        }
    }

    /// <summary>
    /// Transcribes the audio file at <paramref name="path"/>, decoded by the library's codecs: through the live path in blocks
    /// (<see cref="TranscribeFileMode.Live"/>, as <c>idrak transcribe --live FILE</c>) or whole; true when it was read to its end.
    /// </summary>
    /// <param name="path">The file.</param>
    /// <param name="name">Its name as shown (null: the file's name).</param>
    public async Task<bool> TranscribeFileAsync(string path, string? name = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        if (IsRunning)
        {
            return false;
        }

        name ??= Path.GetFileName(path);
        Error = "";
        IsPermissionRefused = false;
        if (SelectedModel is null)
        {
            Error = NoModel();
            return false;
        }

        AudioPipeline pipeline;
        LiveTranscriptionOptions live;
        try
        {
            pipeline = _options.ToPipeline();
            live = Checked(_options.ToLive(microphone: false));
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            Error = e.Message;
            return false;
        }

        if (!await LoadModelAsync().ConfigureAwait(true) || _loadedKey is not { } key)
        {
            return false;
        }

        AudioData audio;
        try
        {
            audio = await Task.Run(() => AudioCodecs.Decode(path)).ConfigureAwait(true);
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException or UnauthorizedAccessException)
        {
            Error = $"{name} could not be read: {e.Message}";
            return false;
        }

        if (_options.FileMode == TranscribeFileMode.Whole)
        {
            return await RunWholeAsync(audio, pipeline, key, name).ConfigureAwait(true);
        }

        IAudioInput input;
        try
        {
            input = pipeline.Wrap(new FileInput(name, audio, _options.FileBlockSeconds ?? TranscribeSettings.DefaultFileBlockSeconds));
        }
        catch (ArgumentException e)
        {
            Error = e.Message;
            return false;
        }

        return await RunLiveAsync(input, live, key, $"Reading {name}: {AudioSeconds(audio.Duration.TotalSeconds)} at {audio.SampleRate} Hz", microphone: false).ConfigureAwait(true);
    }

    /// <summary>Picks an audio file (MAUI's file picker) and transcribes it; false when none was picked or it failed.</summary>
    public async Task<bool> PickFileAsync()
    {
        if (IsRunning)
        {
            return false;
        }

        FileResult? picked;
        try
        {
            picked = await (_picker ?? FilePicker.Default).PickAsync(new PickOptions { PickerTitle = "A recording to transcribe", FileTypes = AudioFileTypes() }).ConfigureAwait(true);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"No file could be picked: {e.Message}";
            return false;
        }

        if (picked is null)
        {
            return false;
        }

        string? copy = null;
        try
        {
            string path = picked.FullPath;
            if (string.IsNullOrEmpty(path) || !File.Exists(path))
            {
                // A file the system gives as a stream only (a content URI): copied once into the cache, removed after.
                string folder = Path.Combine(ComputeResources.CacheFolder, "transcribe");
                Directory.CreateDirectory(folder);
                copy = path = Path.Combine(folder, Path.GetFileName(picked.FileName) is { Length: > 0 } file ? file : "picked-audio");
                await using var from = await picked.OpenReadAsync().ConfigureAwait(true);
                await using var to = File.Create(copy);
                await from.CopyToAsync(to).ConfigureAwait(true);
            }

            return await TranscribeFileAsync(path, picked.FileName).ConfigureAwait(true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            Error = $"{picked.FileName} could not be read: {e.Message}";
            return false;
        }
        finally
        {
            if (copy is not null)
            {
                try
                {
                    File.Delete(copy);
                }
                catch (IOException)
                {
                    // Left for the cache to hold; the next pick of that name replaces it.
                }
            }
        }
    }

    /// <summary>Copies the text to the clipboard; false when there is none.</summary>
    public async Task<bool> CopyAsync()
    {
        string text = Text;
        if (text.Length == 0)
        {
            Status = "Nothing to copy yet.";
            return false;
        }

        try
        {
            await (_clipboard ?? Clipboard.Default).SetTextAsync(text).ConfigureAwait(true);
            Status = $"Copied {text.Length:N0} characters.";
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
        string text = Text;
        if (text.Length == 0)
        {
            Status = "Nothing to share yet.";
            return false;
        }

        try
        {
            string folder = Path.Combine(ComputeResources.CacheFolder, "transcripts");
            Directory.CreateDirectory(folder);
            string path = Path.Combine(folder, $"transcript-{_time.GetLocalNow():yyyyMMdd-HHmmss}.txt");
            await File.WriteAllTextAsync(path, text + "\n", new UTF8Encoding(false)).ConfigureAwait(true);
            await _share.ShareFileAsync(path, "Transcript", "text/plain").ConfigureAwait(true);
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The text was not shared: {e.Message}";
            return false;
        }
    }

    /// <summary>Clears the text and the segments (not while a run goes on).</summary>
    public void Clear()
    {
        if (IsRunning)
        {
            return;
        }

        _state.Clear();
        ApplyNow();
        Status = "";
        Error = "";
    }

    /// <summary>
    /// Checks every setting on the page and the live options they give (<see cref="LiveTranscriptionOptions.Check"/>), then
    /// keeps them in the settings store; false, and nothing kept, when one is refused (each says why). They apply from the
    /// next run.
    /// </summary>
    public bool ApplySettings()
    {
        var next = _options.Clone();
        bool valid = true;
        foreach (var entry in Settings)
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
            SettingsMessage = "Nothing was kept: a value above is not one the setting takes.";
            return false;
        }

        try
        {
            next.ToLive(microphone: true).Check();
            next.ToLive(microphone: false).Check();
        }
        catch (ArgumentException e)
        {
            SettingsMessage = $"Nothing was kept: {e.Message}";
            return false;
        }

        _options = next;
        Keep();
        SettingsMessage = IsRunning ? "Kept: they apply from the next run." : "Kept.";
        return true;
    }

    /// <summary>Cancels what runs at once (its last segment is not read) and stops showing.</summary>
    public void Dispose()
    {
        _abort.Cancel();
        lock (_gate)
        {
            _trailing?.Dispose();
            _trailing = null;
        }
    }

    // The store's complete models whose config.json names a registered speech architecture, and how many others there are.
    internal static (List<SpeechModelEntry> Speech, int Others) ListSpeechModels(IModelStore store)
    {
        var speech = new List<SpeechModelEntry>();
        int others = 0;
        foreach (var model in store.List())
        {
            if (model.Complete && Architecture(model.Folder) is { } architecture && SpeechFamilies.Find(architecture) is not null)
            {
                speech.Add(new SpeechModelEntry(model, architecture));
            }
            else
            {
                others++;
            }
        }

        return (speech, others);
    }

    // config.json's first "architectures" entry (else "architecture"), as SpeechModels.Load reads it; null when there is none.
    private static string? Architecture(string folder)
    {
        string path = Path.Combine(folder, "config.json");
        if (!File.Exists(path))
        {
            return null;
        }

        try
        {
            var config = JsonNode.Parse(File.ReadAllBytes(path)) as JsonObject;
            return config?["architectures"] is JsonArray { Count: > 0 } names && names[0] is JsonValue first && first.TryGetValue(out string? name) ? name
                : config?["architecture"] is JsonValue single && single.TryGetValue(out string? one) ? one
                : null;
        }
        catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static LiveTranscriptionOptions Checked(LiveTranscriptionOptions live)
    {
        live.Check();
        _ = LiveCommitPolicies.Get(live.Policy);
        return live;
    }

    private static FilePickerFileType AudioFileTypes()
    {
        string[] extensions = [.. AudioCodecs.Extensions.Select(e => e.StartsWith('.') ? e : "." + e)];
        return new FilePickerFileType(new Dictionary<DevicePlatform, IEnumerable<string>>
        {
            [DevicePlatform.Android] = ["audio/*"],
            [DevicePlatform.iOS] = ["public.audio"],
            [DevicePlatform.MacCatalyst] = ["public.audio"],
            [DevicePlatform.WinUI] = extensions,
        });
    }

    private static string AudioSeconds(double seconds) => string.Create(CultureInfo.InvariantCulture, $"{seconds:0.0} s");

    // The speech model host: a model is loaded once by its device and folder ("cpu|/path"), kept and released by the service.
    private ModelHost<ISpeechRecognizer> Host => _models.Host<ISpeechRecognizer>(ModelHostName, LoadRecognizer);

    private static ISpeechRecognizer LoadRecognizer(string key)
    {
        int bar = key.IndexOf('|', StringComparison.Ordinal);
        return Idrak.Audio.SpeechModels.Load(key[(bar + 1)..], new SpeechModelOptions { Device = Device.Parse(key[..bar]) });
    }

    private Device ChosenDevice()
    {
        if (IdrakMauiSettings.IsAuto(_app.Device))
        {
            return Device.Default;
        }

        try
        {
            return Device.Parse(_app.Device);
        }
        catch (Exception e) when (e is InvalidOperationException or FormatException or ArgumentException or NotSupportedException)
        {
            return Device.Default;                                             // the start-up named it as a problem already
        }
    }

    private string NoModel() => SpeechModels.Count == 0 ? "No speech model to read with: pull one in Models, then come back." : "Pick a speech model first.";

    private void Keep() => TranscribeSettings.Save(_settings, _options);

    private async Task<bool> LoadAsync(SpeechModelEntry model)
    {
        Error = "";
        ModelStatus = $"Loading {model.Title}…";
        var reporter = _progress.Start($"load {model.Title}", null, ProgressUnit.Elapsed);
        Work = _progress.ItemOf(reporter);
        try
        {
            var (key, languages, device, took) = await Task.Run(() =>
            {
                string at = ChosenDevice().ToString();
                string key = $"{at}|{model.Model.Folder}";
                using var lease = Host.Acquire(key);
                return (key, lease.Model.Languages.Order(StringComparer.Ordinal).ToArray(), lease.Model.Device.ToString(), lease.LoadDuration);
            }).ConfigureAwait(true);
            reporter.Finish($"{model.Title} on {device}");
            _loadedKey = key;
            Languages.Clear();
            if (languages.Length > 0)
            {
                Languages.Add(AutoLanguage);
                foreach (string language in languages)
                {
                    Languages.Add(language);
                }
            }

            OnPropertyChanged(nameof(HasLanguages));
            string wanted = _options.Language ?? AutoLanguage;
            _language = Languages.Contains(wanted) ? wanted : AutoLanguage;
            OnPropertyChanged(nameof(SelectedLanguage));
            ModelStatus = $"{model.Title} on {device}"
                + (took > TimeSpan.Zero ? $", loaded in {ProgressText.Duration(took)}" : "")
                + (languages.Length > 0 ? $"; {languages.Length} language{(languages.Length == 1 ? "" : "s")}" : "");
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            reporter.Finish("not loaded");
            ModelStatus = "";
            Error = $"{model.Title} could not be loaded: {e.Message.Split('\n')[0].Trim()}";
            return false;
        }
    }

    // A run over live sound (the microphone, or a file in blocks): the input metered, transcribed on a worker thread, the
    // page shown at most once an interval, the last values shown at the end.
    private async Task<bool> RunLiveAsync(IAudioInput input, LiveTranscriptionOptions live, string key, string status, bool microphone)
    {
        using var stop = new CancellationTokenSource();
        _stop = stop;
        _step = live.StepSeconds;
        _state.Begin(_time.GetTimestamp());
        _live = true;
        IsRunning = true;
        IsRecording = microphone;
        Status = status;
        Touch(force: true);
        bool done = false;
        try
        {
            await Task.Run(async () =>
            {
                using var lease = Host.Acquire(key);
                await foreach (var update in LiveTranscriber.TranscribeAsync(lease.Model, Metered(input, stop.Token), live, _abort.Token).ConfigureAwait(false))
                {
                    _state.Add(update);
                    Touch(force: false);
                }
            }, CancellationToken.None).ConfigureAwait(true);
            done = true;
            int segments = _state.SegmentCount;
            Status = $"{(microphone ? "Stopped" : "Done")}: {AudioSeconds(_state.HeardSeconds)} of sound, {segments} segment{(segments == 1 ? "" : "s")}.";
        }
        catch (OperationCanceledException) when (_abort.IsCancellationRequested)
        {
            Status = "Stopped.";
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The transcription stopped: {e.Message.Split('\n')[0].Trim()}";
            Status = "";
        }
        finally
        {
            input.Dispose();
            _stop = null;
            EndShowing();
            IsRecording = false;
            IsRunning = false;
        }

        return done;
    }

    // A file read whole: the pipeline applied to the recording, one transcription, its segments committed.
    private async Task<bool> RunWholeAsync(AudioData audio, AudioPipeline pipeline, string key, string name)
    {
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(_abort.Token);
        _stop = stop;
        _state.Begin(_time.GetTimestamp());
        IsRunning = true;
        Status = $"Reading {name} whole: {AudioSeconds(audio.Duration.TotalSeconds)}";
        var reporter = _progress.Start($"transcribe {name}", null, ProgressUnit.Items);
        Work = _progress.ItemOf(reporter);
        var request = _options.ToTranscription();
        try
        {
            var result = await Task.Run(() =>
            {
                using var lease = Host.Acquire(key);
                var cleaned = pipeline.IsEmpty ? audio : pipeline.Apply(audio);
                return lease.Model.Transcribe(cleaned, request, new SegmentProgress(reporter), stop.Token);
            }, CancellationToken.None).ConfigureAwait(true);
            _state.AddWhole(result, audio.Duration.TotalSeconds);
            reporter.Finish($"{name}: {result.Segments.Count} segment(s)");
            Status = $"Done: {AudioSeconds(audio.Duration.TotalSeconds)} of sound read in {ProgressText.Duration(result.Stats.Total)}.";
            return true;
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested)
        {
            reporter.Finish("stopped");
            Status = "Stopped.";
            return false;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            reporter.Finish("failed");
            Error = $"{name} was not transcribed: {e.Message.Split('\n')[0].Trim()}";
            Status = "";
            return false;
        }
        finally
        {
            _stop = null;
            ApplyNow();
            IsRunning = false;
        }
    }

    // The input's blocks as they come, each one's level measured from its samples in place (no copy) and stored; a showing
    // is handed to the UI thread only when one is due.
    private async IAsyncEnumerable<AudioData> Metered(IAudioInput input, [System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken)
    {
        await foreach (var block in input.ReadAsync(cancellationToken).ConfigureAwait(false))
        {
            _state.SetLevel(LevelMeter.Decibels(block.Samples));
            Touch(force: false);
            yield return block;
        }
    }

    // Stores a showing and hands it to the UI thread when one is due: at most once an interval (a forced one at once), one
    // waiting there at a time (a later snapshot replaces it); a showing skipped as too early is made by a timer once due.
    private void Touch(bool force)
    {
        if (!_live)
        {
            return;
        }

        long now = _time.GetTimestamp();
        long last = Interlocked.Read(ref _last);
        var interval = _progress.Interval;
        if (!force && last != Never && _time.GetElapsedTime(last, now) is var since && since < interval)
        {
            lock (_gate)
            {
                _trailing ??= _time.CreateTimer(_ => Trail(), null, interval - since, Timeout.InfiniteTimeSpan);
            }

            return;
        }

        if (Interlocked.CompareExchange(ref _last, now, last) != last && !force)
        {
            return;                                                             // another thread took this showing
        }

        var snapshot = _state.Snapshot(_time, now);
        lock (_gate)
        {
            if (_pending is null || snapshot.Sequence > _pending.Sequence)
            {
                _pending = snapshot;
            }
        }

        if (Interlocked.Exchange(ref _scheduled, 1) == 0)
        {
            _progress.Dispatch(_publish);
        }
    }

    private void Trail()
    {
        lock (_gate)
        {
            _trailing?.Dispose();
            _trailing = null;
        }

        Touch(force: true);
    }

    // On the UI thread: the latest snapshot onto the page.
    private void Publish()
    {
        Volatile.Write(ref _scheduled, 0);
        LiveSnapshot? snapshot;
        lock (_gate)
        {
            snapshot = _pending;
        }

        if (snapshot is not null)
        {
            Apply(snapshot);
        }
    }

    // The run's end: no more showings, the last values shown now (on the UI thread, where the run's continuation is).
    private void EndShowing()
    {
        _live = false;
        lock (_gate)
        {
            _trailing?.Dispose();
            _trailing = null;
        }

        ApplyNow();
    }

    private void ApplyNow() => Apply(_state.Snapshot(_time, _time.GetTimestamp()));

    private void Apply(LiveSnapshot s)
    {
        lock (_applyGate)
        {
            if (s.Sequence <= _applied)
            {
                return;                                                         // an older showing that waited behind a newer one
            }

            _applied = s.Sequence;
            CommittedText = s.Committed;
            TentativeText = s.Tentative;
            LevelDecibels = s.Decibels;
            double floor = MeterFloorDecibels < 0 ? MeterFloorDecibels : -60;
            Level = double.IsFinite(s.Decibels) ? Math.Clamp(1 - s.Decibels / floor, 0, 1) : 0;
            ElapsedText = Clock(s.Elapsed);
            HeardText = s.Heard > 0 ? $"{AudioSeconds(s.Heard)} of sound" : "";
            Behind = s.Behind;
            BehindText = s.Heard <= 0 ? "" : s.Behind >= _step ? $"{AudioSeconds(s.Behind)} behind" : "keeping up";
        }
    }

    private static string Clock(TimeSpan time) => time.ToString(time.TotalHours >= 1 ? @"h\:mm\:ss" : @"mm\:ss", CultureInfo.InvariantCulture);

    // A file as live sound: its blocks in order, as fast as they are read (the live path tried on a recording); a mono
    // recording's own samples are sliced, a multichannel one mixed to one first.
    private sealed class FileInput(string name, AudioData audio, double blockSeconds) : IAudioInput
    {
        public string Description => name;

        public int SampleRate => audio.SampleRate;

        public int Channels => 1;

        public async IAsyncEnumerable<AudioData> ReadAsync([System.Runtime.CompilerServices.EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            float[] mono;
            if (audio.Channels == 1)
            {
                mono = audio.Samples;
            }
            else
            {
                mono = new float[audio.Length];
                audio.ToMono(mono);
            }

            int block = Math.Max(1, (int)Math.Round(blockSeconds * audio.SampleRate));
            for (int start = 0; start < mono.Length && !cancellationToken.IsCancellationRequested; start += block)
            {
                yield return new AudioData(mono.AsSpan(start, Math.Min(block, mono.Length - start)).ToArray(), 1, audio.SampleRate);
                await Task.Yield();
            }
        }

        public void Dispose()
        {
        }
    }

    // A whole file's transcription progress onto its reporter: segments done of the segments in all.
    private sealed class SegmentProgress(IProgressReporter reporter) : IProgress<TranscriptionProgress>
    {
        public void Report(TranscriptionProgress value)
        {
            reporter.Total = value.Segments;
            reporter.Report(value.Segment);
        }
    }

    // What the page shows of a run, at one moment.
    private sealed record LiveSnapshot(long Sequence, string Committed, string Tentative, double Decibels, TimeSpan Elapsed, double Heard, double Behind);

    // The run's text and meter, written by the worker threads (the blocks' reader and the transcription), read by snapshots.
    private sealed class LiveState
    {
        private readonly Lock _gate = new();
        private readonly StringBuilder _finished = new();
        private readonly StringBuilder _current = new();
        private readonly List<TranscriptSegment> _segments = [];
        private string _tentative = "";
        private long _started;
        private long _sequence;
        private double _decibels = double.NegativeInfinity;
        private double _heard;
        private double _behind;

        public int SegmentCount
        {
            get
            {
                lock (_gate)
                {
                    return _segments.Count;
                }
            }
        }

        public double HeardSeconds
        {
            get
            {
                lock (_gate)
                {
                    return _heard;
                }
            }
        }

        public IReadOnlyList<TranscriptSegment> Segments()
        {
            lock (_gate)
            {
                return [.. _segments];
            }
        }

        // A new run: the text so far is kept (a second recording goes on below the first), the open segment and meter start again.
        public void Begin(long now)
        {
            lock (_gate)
            {
                _started = now;
                _current.Clear();
                _tentative = "";
                _decibels = double.NegativeInfinity;
                _heard = 0;
                _behind = 0;
            }
        }

        public void Clear()
        {
            lock (_gate)
            {
                _finished.Clear();
                _current.Clear();
                _segments.Clear();
                _tentative = "";
                _decibels = double.NegativeInfinity;
                _heard = 0;
                _behind = 0;
            }
        }

        public void SetLevel(double decibels)
        {
            lock (_gate)
            {
                _decibels = decibels;
            }
        }

        public void Add(LiveTranscript update)
        {
            lock (_gate)
            {
                if (update.Segment is { } segment)
                {
                    Line(segment);
                    _segments.Add(new TranscriptSegment(update.Start, update.End, segment, 0f, []));
                    _current.Clear();
                    _tentative = "";
                }
                else
                {
                    if (update.Committed.Length > 0)
                    {
                        (_current.Length > 0 ? _current.Append(' ') : _current).Append(update.Committed);
                    }

                    _tentative = update.Tentative;
                }

                _heard = update.End;
                _behind = update.Behind;
            }
        }

        public void AddWhole(Transcription result, double seconds)
        {
            lock (_gate)
            {
                foreach (var segment in result.Segments)
                {
                    string text = segment.Text.Trim();
                    if (text.Length > 0)
                    {
                        Line(text);
                        _segments.Add(segment with { Text = text });
                    }
                }

                if (result.Segments.Count == 0 && result.Text.Trim() is { Length: > 0 } whole)
                {
                    Line(whole);
                    _segments.Add(new TranscriptSegment(0, seconds, whole, 0f, []));
                }

                _heard = seconds;
            }
        }

        public LiveSnapshot Snapshot(TimeProvider time, long now)
        {
            lock (_gate)
            {
                string committed = _current.Length == 0 ? _finished.ToString()
                    : _finished.Length == 0 ? _current.ToString()
                    : string.Concat(_finished.ToString(), "\n", _current.ToString());
                return new LiveSnapshot(++_sequence, committed, _tentative, _decibels, time.GetElapsedTime(_started, now), _heard, _behind);
            }
        }

        private void Line(string text) => (_finished.Length > 0 ? _finished.Append('\n') : _finished).Append(text);
    }
}

/// <summary>The level of a block of samples (the audio pages' meters), in one pass over them where they are.</summary>
public static class LevelMeter
{
    /// <summary>The root mean square of <paramref name="samples"/> in decibels of full scale (-120 for silence; negative infinity when there are none).</summary>
    public static double Decibels(ReadOnlySpan<float> samples)
    {
        if (samples.IsEmpty)
        {
            return double.NegativeInfinity;
        }

        double sum = 0;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && samples.Length >= System.Numerics.Vector<float>.Count)
        {
            var acc = System.Numerics.Vector<float>.Zero;
            for (; i <= samples.Length - System.Numerics.Vector<float>.Count; i += System.Numerics.Vector<float>.Count)
            {
                var v = new System.Numerics.Vector<float>(samples[i..]);
                acc += v * v;
            }

            sum = System.Numerics.Vector.Sum(acc);
        }

        for (; i < samples.Length; i++)
        {
            sum += (double)samples[i] * samples[i];                              // scalar tail
        }

        return 10 * Math.Log10(Math.Max(sum / samples.Length, 1e-12));
    }

    /// <summary>The largest magnitude of <paramref name="samples"/> in decibels of full scale (-120 for silence; negative infinity when there are none).</summary>
    public static double PeakDecibels(ReadOnlySpan<float> samples)
    {
        if (samples.IsEmpty)
        {
            return double.NegativeInfinity;
        }

        float peak = 0;
        int i = 0;
        if (System.Numerics.Vector.IsHardwareAccelerated && samples.Length >= System.Numerics.Vector<float>.Count)
        {
            var max = System.Numerics.Vector<float>.Zero;
            for (; i <= samples.Length - System.Numerics.Vector<float>.Count; i += System.Numerics.Vector<float>.Count)
            {
                max = System.Numerics.Vector.Max(max, System.Numerics.Vector.Abs(new System.Numerics.Vector<float>(samples[i..])));
            }

            for (int lane = 0; lane < System.Numerics.Vector<float>.Count; lane++)
            {
                peak = Math.Max(peak, max[lane]);
            }
        }

        for (; i < samples.Length; i++)
        {
            peak = Math.Max(peak, Math.Abs(samples[i]));                          // scalar tail
        }

        return 20 * Math.Log10(Math.Max(peak, 1e-6));
    }
}
