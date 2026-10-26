// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Windows.Input;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.ViewModels;

/// <summary>A synthesis model of the model store, as the Speak page lists it: with why it cannot speak here, when it cannot.</summary>
public sealed class SynthesisModelEntry
{
    internal SynthesisModelEntry(StoredModel model, string architecture, SpeechTextNeeds? needs, string? reason)
    {
        Model = model;
        Architecture = architecture;
        TextNeeds = needs;
        Reason = reason;
    }

    /// <summary>The stored model.</summary>
    public StoredModel Model { get; }

    /// <summary>The architecture its config.json names, a registered <see cref="SpeechSynthesisFamilies"/> name.</summary>
    public string Architecture { get; }

    /// <summary>What its text front end needs (<see cref="SpeechSynthesisModels.TextNeeds"/>), or null when its files could not be read.</summary>
    public SpeechTextNeeds? TextNeeds { get; }

    /// <summary>Why it cannot speak here (a phonemizer it reads that is not registered: espeak-ng is an outside program), or null.</summary>
    public string? Reason { get; }

    /// <summary>Whether it can speak here.</summary>
    public bool CanSpeak => Reason is null;

    /// <summary>How the settings keep it: "owner/name@revision".</summary>
    public string Key => $"{Model.Id}@{Model.Revision}";

    /// <summary>Its id.</summary>
    public string Title => Model.Id;

    /// <summary>Its architecture, size, revision and what its text needs.</summary>
    public string Details => $"{Architecture}, {ProgressText.Bytes(Model.Bytes)}, revision {Model.Revision}"
        + (TextNeeds is { Phonemes: true } n ? $"; reads phonemes{(n.PhonemeLanguage is { } l ? $" ({l})" : "")}" : "")
        + (TextNeeds is { Romanized: true } ? "; reads romanised text" : "");

    /// <inheritdoc />
    public override string ToString() => CanSpeak ? Title : $"{Title} (cannot speak here)";
}

/// <summary>A voice the Speak page lists: the model's own, a speaker of the model, or an installed voice; with why it cannot speak here, when it cannot.</summary>
public sealed class VoiceEntry
{
    internal VoiceEntry(string name, string title, string details, string? reason = null)
    {
        Name = name;
        Title = title;
        Details = details;
        Reason = reason;
    }

    /// <summary>What <see cref="SpeechSynthesisOptions.Voice"/> takes ("" for the model's own).</summary>
    public string Name { get; }

    /// <summary>Its name on the page.</summary>
    public string Title { get; }

    /// <summary>What it is (a speaker of the model, an installed voice of a kind).</summary>
    public string Details { get; }

    /// <summary>Why it cannot speak here (a converter or phonemizer not registered), or null.</summary>
    public string? Reason { get; }

    /// <summary>Whether it can speak here.</summary>
    public bool CanSpeak => Reason is null;

    /// <inheritdoc />
    public override string ToString() => CanSpeak ? Title : $"{Title} (cannot speak here)";
}

/// <summary>
/// The Speak page (plan 34, M6): text read aloud on the speaker. A synthesis model is picked from the model store's complete
/// models whose architecture a registered <see cref="SpeechSynthesisFamilies"/> family reads; one whose text front end needs
/// what is not registered here (a phonemizer: espeak-ng is an outside program an app does not run) is listed with that
/// reason (<see cref="SpeechSynthesisModels.TextNeeds"/>, <see cref="SpeechTextNeeds.Missing"/>), not hidden. The model is
/// loaded once through the <see cref="IModelService"/> (host <see cref="ModelHostName"/>) on the settings' device, wrapped
/// in a <see cref="VoicedSynthesizer"/> so the installed voices of the voice store speak too (each listed with why not,
/// when it cannot). The text is spoken piece by piece (<see cref="ISpeechSynthesizer.SpeakStream"/>) and played through the
/// <see cref="ISoundPlayer"/> as the first piece is made, the next made while one plays; Stop cuts it at once. The time to
/// the first sound and the real-time factor (the synthesis' own time over the sound's) are shown, the first speech after a
/// load (cold) apart from the later ones (warm). The speech is kept to play again, save (WAV, MP3) and share. The page is
/// updated at most once a progress interval.
/// </summary>
public sealed class SpeakViewModel : ViewModel, IDisposable
{
    /// <summary>The <see cref="IModelService"/> host synthesis models are kept in.</summary>
    public const string ModelHostName = "synthesis";

    private readonly IModelStore _store;
    private readonly IModelService _models;
    private readonly ISoundPlayer _player;
    private readonly IShareService _share;
    private readonly IdrakMauiOptions _app;
    private readonly ISettingsStore _settings;
    private readonly ProgressCenter _progress;
    private readonly TimeProvider _time;
    private readonly UiBatcher _batcher;
    private readonly SpeakState _state = new();
    private readonly CancellationTokenSource _abort = new();
    private SpeakLabOptions _options;
    private SynthesisModelEntry? _selected;
    private VoiceEntry? _voice;
    private Task<bool>? _loading;
    private string? _loadingKey;
    private string? _loadedKey;
    private object? _spokenWith;
    private CancellationTokenSource? _stop;
    private AudioData? _speech;
    private bool _speaking, _playing, _listing, _showSettings;
    private string _text = "", _status = "", _error = "", _modelStatus = "", _modelsText = "", _settingsMessage = "", _progressText = "";
    private string _coldText = "", _warmText = "", _speechText = "", _savedText = "";
    private ProgressItem? _work;

    /// <summary>The page over the app's services.</summary>
    public SpeakViewModel(IModelStore store, IModelService models, ISoundPlayer player, IShareService share, IdrakMauiOptions options, ISettingsStore settings,
        ProgressCenter? progress = null)
    {
        _store = store ?? throw new ArgumentNullException(nameof(store));
        _models = models ?? throw new ArgumentNullException(nameof(models));
        _player = player ?? throw new ArgumentNullException(nameof(player));
        _share = share ?? throw new ArgumentNullException(nameof(share));
        _app = options ?? throw new ArgumentNullException(nameof(options));
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        _progress = progress ?? new ProgressCenter(options.ProgressInterval);
        _time = _progress.Time;
        _batcher = new UiBatcher(_progress, Publish);
        _options = SpeakSettings.Load(settings, out var skipped);
        if (skipped.Count > 0)
        {
            _settingsMessage = "A saved setting was not read: " + string.Join("; ", skipped);
        }

        foreach (var setting in SpeakSettings.Definitions)
        {
            Settings.Add(new LabSettingEntry(setting, setting.Get(_options)));
        }

        SpeakCommand = new Command(async () =>
        {
            if (IsSpeaking)
            {
                Stop();
            }
            else
            {
                await SpeakAsync();
            }
        });
        StopCommand = new Command(Stop);
        PlayAgainCommand = new Command(async () => await PlayAgainAsync());
        SaveWavCommand = new Command(async () => await SaveAsync("wav"));
        SaveMp3Command = new Command(async () => await SaveAsync("mp3"));
        ShareCommand = new Command(async () => await ShareAsync());
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
    public string Title => "Speak";

    /// <summary>The settings in effect (<see cref="ApplySettings"/> checks and keeps the page's).</summary>
    public SpeakLabOptions Options => _options;

    /// <summary>The synthesis models the store holds complete, whose architecture a registered family reads (those that cannot speak here with their reason).</summary>
    public ObservableCollection<SynthesisModelEntry> SynthesisModels { get; } = [];

    /// <summary>Whether there is a synthesis model to pick.</summary>
    public bool HasModels => SynthesisModels.Count > 0;

    /// <summary>Whether there is none (the page offers the Models page).</summary>
    public bool HasNoModels => SynthesisModels.Count == 0;

    /// <summary>The synthesis models as a line: how many (and how many cannot speak here), or why none.</summary>
    public string ModelsText
    {
        get => _modelsText;
        private set => SetProperty(ref _modelsText, value);
    }

    /// <summary>The model picked; picking one keeps it in the settings and loads it (once) when it can speak here.</summary>
    public SynthesisModelEntry? SelectedModel
    {
        get => _selected;
        set
        {
            if (SetProperty(ref _selected, value) && value is not null)
            {
                OnPropertyChanged(nameof(ModelReason));
                OnPropertyChanged(nameof(HasModelReason));
                _options.Model = value.Key;
                Keep();
                if (value.CanSpeak)
                {
                    _ = LoadModelAsync();
                }
                else
                {
                    ModelStatus = "";
                    Voices.Clear();
                    OnPropertyChanged(nameof(HasVoices));
                }
            }
        }
    }

    /// <summary>Why the model picked cannot speak here, or "".</summary>
    public string ModelReason => SelectedModel?.Reason is { } reason ? $"It cannot speak here: {reason}." : "";

    /// <summary>Whether <see cref="ModelReason"/> has something to show.</summary>
    public bool HasModelReason => ModelReason.Length > 0;

    /// <summary>The model's state: loading, loaded on which device in how long, its rate and voices; or "".</summary>
    public string ModelStatus
    {
        get => _modelStatus;
        private set => SetProperty(ref _modelStatus, value);
    }

    /// <summary>The model loading, or null.</summary>
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

    /// <summary>The voices of the loaded model: its own, its speakers, the installed voices (those that cannot speak here with their reason).</summary>
    public ObservableCollection<VoiceEntry> Voices { get; } = [];

    /// <summary>Whether there is a voice to choose besides the model's own.</summary>
    public bool HasVoices => Voices.Count > 1;

    /// <summary>The voice chosen (kept in the settings).</summary>
    public VoiceEntry? SelectedVoice
    {
        get => _voice;
        set
        {
            if (SetProperty(ref _voice, value) && value is not null)
            {
                OnPropertyChanged(nameof(VoiceReason));
                OnPropertyChanged(nameof(HasVoiceReason));
                _options.Voice = value.Name.Length == 0 ? null : value.Name;
                Keep();
            }
        }
    }

    /// <summary>Why the voice chosen cannot speak here, or "".</summary>
    public string VoiceReason => SelectedVoice?.Reason is { } reason ? $"This voice cannot speak here: {reason}." : "";

    /// <summary>Whether <see cref="VoiceReason"/> has something to show.</summary>
    public bool HasVoiceReason => VoiceReason.Length > 0;

    /// <summary>The text to speak.</summary>
    public string Text
    {
        get => _text;
        set => SetProperty(ref _text, value ?? "");
    }

    /// <summary>Whether speech is being made and played (the speak button stops it).</summary>
    public bool IsSpeaking
    {
        get => _speaking;
        private set
        {
            if (SetProperty(ref _speaking, value))
            {
                OnPropertyChanged(nameof(IsIdle));
                OnPropertyChanged(nameof(SpeakText));
                OnPropertyChanged(nameof(SpeakGlyph));
            }
        }
    }

    /// <summary>Whether the speech kept is playing again.</summary>
    public bool IsPlaying
    {
        get => _playing;
        private set
        {
            if (SetProperty(ref _playing, value))
            {
                OnPropertyChanged(nameof(IsIdle));
            }
        }
    }

    /// <summary>Whether nothing is being spoken or played.</summary>
    public bool IsIdle => !IsSpeaking && !IsPlaying;

    /// <summary>The speak button's words.</summary>
    public string SpeakText => IsSpeaking ? "Stop" : "Speak";

    /// <summary>The speak button's mark.</summary>
    public string SpeakGlyph => IsSpeaking ? "■" : "▶";

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

    /// <summary>How far the speech is ("3 pieces, 4.2 s of sound, made in 1.1 s"), or "".</summary>
    public string ProgressLine
    {
        get => _progressText;
        private set => SetProperty(ref _progressText, value);
    }

    /// <summary>The first speech after a load (cold): its time to the first sound and real-time factor, or "".</summary>
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

    /// <summary>The latest speech after the first (warm): its time to the first sound and real-time factor, or "".</summary>
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

    /// <summary>The last time to the first sound measured (cold or warm), or null.</summary>
    public TimeSpan? FirstSound { get; private set; }

    /// <summary>The last real-time factor measured (the synthesis' time over the sound's), or 0.</summary>
    public double RealTimeFactor { get; private set; }

    /// <summary>The speech kept (the last one made, or what was made of it before Stop), or null.</summary>
    public AudioData? Speech
    {
        get => _speech;
        private set
        {
            if (SetProperty(ref _speech, value))
            {
                OnPropertyChanged(nameof(HasSpeech));
                SpeechText = value is null ? "" : $"{AudioText.Seconds(value.Duration.TotalSeconds)} of speech at {value.SampleRate} Hz";
            }
        }
    }

    /// <summary>Whether there is speech to play, save or share.</summary>
    public bool HasSpeech => Speech is not null;

    /// <summary>The speech kept as a line, or "".</summary>
    public string SpeechText
    {
        get => _speechText;
        private set => SetProperty(ref _speechText, value);
    }

    /// <summary>Whether the library has an encoder that writes MP3 (the page shows its button).</summary>
    public bool CanSaveMp3 => AudioEncoders.Names.Contains("mp3", StringComparer.OrdinalIgnoreCase);

    /// <summary>Where the speech was saved last, or "".</summary>
    public string SavedText
    {
        get => _savedText;
        private set => SetProperty(ref _savedText, value);
    }

    /// <summary>The path of the file saved last, or null.</summary>
    public string? SavedPath { get; private set; }

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

    /// <summary>The UI showings made while speaking (at most one a progress interval).</summary>
    public int Showings => _batcher.Dispatches;

    /// <summary>Opens the Models page (the page sets it: a push of the "models" feature).</summary>
    public Func<Task>? OpenModels { get; set; }

    /// <summary>Speaks the text, or stops the speech running.</summary>
    public ICommand SpeakCommand { get; }

    /// <summary>Stops the speech, or the speech playing again, at once.</summary>
    public ICommand StopCommand { get; }

    /// <summary>Plays the speech kept again.</summary>
    public ICommand PlayAgainCommand { get; }

    /// <summary>Saves the speech as WAV.</summary>
    public ICommand SaveWavCommand { get; }

    /// <summary>Saves the speech as MP3.</summary>
    public ICommand SaveMp3Command { get; }

    /// <summary>Saves the speech (in the settings' format) and offers it to the share sheet.</summary>
    public ICommand ShareCommand { get; }

    /// <summary>Opens the Models page (<see cref="OpenModels"/>).</summary>
    public ICommand GoToModelsCommand { get; }

    /// <summary>Reads the models again.</summary>
    public ICommand RefreshCommand { get; }

    /// <summary>Checks and keeps the settings (<see cref="ApplySettings"/>).</summary>
    public ICommand ApplySettingsCommand { get; }

    /// <summary>Opens or closes the settings card.</summary>
    public ICommand ToggleSettingsCommand { get; }

    /// <summary>
    /// Reads the synthesis models on a worker thread: the store's complete models whose config.json names an architecture a
    /// registered family reads, each with what its text needs and why it cannot speak here. The model kept in the settings
    /// is picked again (else the first that can speak).
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
            var (models, others) = await Task.Run(() => ListSynthesisModels(_store)).ConfigureAwait(true);
            SynthesisModels.Clear();
            foreach (var model in models)
            {
                SynthesisModels.Add(model);
            }

            OnPropertyChanged(nameof(HasModels));
            OnPropertyChanged(nameof(HasNoModels));
            var families = SpeechSynthesisFamilies.Names;
            int cannot = models.Count(m => !m.CanSpeak);
            ModelsText = models.Count > 0
                ? $"{models.Count} synthesis model{(models.Count == 1 ? "" : "s")} on this phone{(cannot > 0 ? $", {cannot} of them cannot speak here (their reason below when picked)" : "")}."
                : families.Count == 0 ? "No speech synthesis family is registered: the app registers the families it shows (SpeechSynthesisFamilies.Register, a plug-in's own Register call)."
                : $"No synthesis model on this phone{(others > 0 ? $" among the {others} other model{(others == 1 ? "" : "s")}" : "")}: pull one in Models, "
                  + $"of an architecture registered here ({string.Join(", ", families)}).";
            var keep = models.FirstOrDefault(m => string.Equals(m.Key, _options.Model, StringComparison.OrdinalIgnoreCase))
                ?? models.FirstOrDefault(m => _selected is { } current && current.Key == m.Key)
                ?? models.FirstOrDefault(m => m.CanSpeak)
                ?? models.FirstOrDefault();
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
    /// Loads the picked model through the model service on the settings' device (once: a later call with the same model
    /// and device gives the same load) and lists its voices; true when it is loaded.
    /// </summary>
    public Task<bool> LoadModelAsync()
    {
        if (SelectedModel is not { } model)
        {
            Error = NoModel();
            return Task.FromResult(false);
        }

        if (model.Reason is { } reason)
        {
            Error = $"{model.Title} cannot speak here: {reason}.";
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
    /// Speaks <see cref="Text"/> until it ends or <see cref="Stop"/>: the settings checked, the model loaded, the text spoken
    /// piece by piece and each piece played as soon as it is made; true when it was spoken to its end.
    /// </summary>
    public async Task<bool> SpeakAsync()
    {
        if (IsSpeaking)
        {
            return false;
        }

        Error = "";
        string text = Text.Trim();
        if (text.Length == 0)
        {
            Error = "Type the text to speak first.";
            return false;
        }

        if (SelectedModel is not { } model)
        {
            Error = NoModel();
            return false;
        }

        SpeechSynthesisOptions speech;
        SoundPlayRequest play;
        try
        {
            speech = _options.ToSynthesis();
            play = _options.ToPlayback();
            play.Output.Check();
        }
        catch (ArgumentException e)
        {
            Error = e.Message;
            return false;
        }

        if ((model.Reason ?? model.TextNeeds?.Missing(text)) is { } reason)
        {
            Error = $"{model.Title} cannot speak this here: {reason}.";
            return false;
        }

        if (SelectedVoice?.Reason is { } voiceReason)
        {
            Error = $"Voice {SelectedVoice.Title} cannot speak here: {voiceReason}.";
            return false;
        }

        if (!await LoadModelAsync().ConfigureAwait(true) || _loadedKey is not { } key)
        {
            return false;
        }

        _player.Stop();                                                          // the speech played again, if it plays
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(_abort.Token);
        _stop = stop;
        _state.Begin();
        IsSpeaking = true;
        Status = $"Speaking {text.Length:N0} characters…";
        _batcher.Start();
        _batcher.Touch(force: true);
        bool done = false;
        try
        {
            var (played, cold) = await Task.Run(async () =>
            {
                using var lease = Host.Acquire(key);
                bool first = !ReferenceEquals(lease.Model, Volatile.Read(ref _spokenWith));
                var made = Made(lease.Model.SpeakStream(text, speech, stop.Token), stop.Token);
                var result = await _player.PlayAsync(made, play, stop.Token).ConfigureAwait(false);
                Volatile.Write(ref _spokenWith, lease.Model);
                return (result, first);
            }, CancellationToken.None).ConfigureAwait(true);
            var snapshot = _state.Snapshot();
            if (played.Completed)
            {
                done = true;
                Report(cold, played, snapshot);
                Status = $"Done: {AudioText.Seconds(snapshot.Seconds)} of speech on {played.Output}.";
            }
            else
            {
                Status = $"Stopped after {AudioText.Seconds(played.Seconds)} of speech.";
            }
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested)
        {
            Status = "Stopped.";
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The speech stopped: {e.Message.Split('\n')[0].Trim()}";
            Status = "";
        }
        finally
        {
            _stop = null;
            _batcher.End();
            Publish();
            Speech = _state.Joined() ?? Speech;
            IsSpeaking = false;
        }

        return done;
    }

    /// <summary>Stops what runs at once: the speech being made and played, or the speech playing again.</summary>
    public void Stop()
    {
        try
        {
            if (_stop is { IsCancellationRequested: false } stop)
            {
                Status = "Stopping…";                                           // first: cancelling may end the run at once, which says it stopped
                stop.Cancel();
            }
        }
        catch (ObjectDisposedException)
        {
            // The run ended meanwhile.
        }

        _player.Stop();
    }

    /// <summary>Plays the speech kept again; true when it played to its end.</summary>
    public async Task<bool> PlayAgainAsync()
    {
        if (Speech is not { } speech || IsSpeaking)
        {
            Status = IsSpeaking ? Status : "Nothing to play yet: speak first.";
            return false;
        }

        Error = "";
        IsPlaying = true;
        Status = "Playing the speech…";
        try
        {
            var played = await Task.Run(() => _player.PlayAsync(speech, _options.ToPlayback()), CancellationToken.None).ConfigureAwait(true);
            Status = played.Completed ? $"Played {AudioText.Seconds(played.Seconds)} on {played.Output}." : "Stopped.";
            return played.Completed;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The speech was not played: {e.Message.Split('\n')[0].Trim()}";
            Status = "";
            return false;
        }
        finally
        {
            IsPlaying = false;
        }
    }

    /// <summary>
    /// Saves the speech kept with the encoder <paramref name="format"/> ("wav", "mp3"; null: the settings') in the cache
    /// folder's <c>speech</c> folder; the file's path, or null when there is no speech or it failed.
    /// </summary>
    public async Task<string?> SaveAsync(string? format = null)
    {
        if (Speech is not { } speech)
        {
            Status = "Nothing to save yet: speak first.";
            return null;
        }

        Error = "";
        format ??= _options.SaveFormat ?? "wav";
        try
        {
            var encoder = AudioEncoders.Get(format);
            var encode = encoder.Name.Equals("wav", StringComparison.OrdinalIgnoreCase) ? _options.ToEncode() : new AudioEncodeOptions();
            string folder = Path.Combine(ComputeResources.CacheFolder, "speech");
            string path = Path.Combine(folder, $"speech-{_time.GetLocalNow():yyyyMMdd-HHmmss}{encoder.Extension}");
            await Task.Run(() => AudioEncoders.Save(path, speech, encode)).ConfigureAwait(true);
            SavedPath = path;
            SavedText = $"Saved {Path.GetFileName(path)} ({ProgressText.Bytes(new FileInfo(path).Length)}) in {folder}.";
            return path;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The speech was not saved as {format}: {e.Message.Split('\n')[0].Trim()}";
            return null;
        }
    }

    /// <summary>Saves the speech in the settings' format and offers it to the phone's share sheet; false when there is none.</summary>
    public async Task<bool> ShareAsync()
    {
        if (await SaveAsync().ConfigureAwait(true) is not { } path)
        {
            return false;
        }

        try
        {
            await _share.ShareFileAsync(path, "Speech", ContentType(path)).ConfigureAwait(true);
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The speech was not shared: {e.Message}";
            return false;
        }
    }

    /// <summary>
    /// Checks every setting on the page and the options they give, then keeps them in the settings store; false, and
    /// nothing kept, when one is refused (each says why). The voices are listed again (the voice store may have changed).
    /// </summary>
    public bool ApplySettings()
    {
        if (LabSettingEntry.Apply(Settings, _options.Clone()) is not { } next)
        {
            SettingsMessage = "Nothing was kept: a value above is not one the setting takes.";
            return false;
        }

        try
        {
            next.ToPlayback().Output.Check();
        }
        catch (ArgumentException e)
        {
            SettingsMessage = $"Nothing was kept: {e.Message}";
            return false;
        }

        _options = next;
        Keep();
        SettingsMessage = IsSpeaking ? "Kept: they apply from the next speech." : "Kept.";
        if (_loadedKey is not null && !IsSpeaking)
        {
            _ = ListVoicesAsync();
        }

        return true;
    }

    /// <summary>Stops what runs at once and stops showing.</summary>
    public void Dispose()
    {
        _abort.Cancel();
        _player.Stop();
        _batcher.Dispose();
    }

    /// <summary>
    /// The store's complete models whose config.json names a registered synthesis architecture, each with what its text
    /// needs and why it cannot speak here (none hidden); and how many other models the store holds. Reads files: call it
    /// off the UI thread.
    /// </summary>
    public static (IReadOnlyList<SynthesisModelEntry> Models, int Others) ListSynthesisModels(IModelStore store)
    {
        ArgumentNullException.ThrowIfNull(store);
        var models = new List<SynthesisModelEntry>();
        int others = 0;
        foreach (var model in store.List())
        {
            if (!model.Complete || Architecture(model.Folder) is not { } architecture || SpeechSynthesisFamilies.Find(architecture) is null)
            {
                others++;
                continue;
            }

            SpeechTextNeeds? needs = null;
            string? reason;
            try
            {
                needs = SpeechSynthesisModels.TextNeeds(model.Folder);
                reason = needs.Missing();
            }
            catch (Exception e) when (e is IOException or InvalidDataException or JsonException or NotSupportedException or UnauthorizedAccessException)
            {
                reason = $"its text front end could not be read ({e.Message.Split('\n')[0].Trim()})";
            }

            models.Add(new SynthesisModelEntry(model, architecture, needs, reason));
        }

        return (models, others);
    }

    /// <summary>
    /// The voices <paramref name="model"/> speaks: its own, its speakers, then (through a <see cref="VoicedSynthesizer"/>)
    /// the installed voices of the store <paramref name="speech"/> names, each with why it cannot speak here (none hidden).
    /// Reads files: call it off the UI thread.
    /// </summary>
    public static IReadOnlyList<VoiceEntry> ListVoices(ISpeechSynthesizer model, SpeechSynthesisOptions speech)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(speech);
        var speakers = model is VoicedSynthesizer voiced ? voiced.Synthesizer.Voices : model.Voices;
        var voices = new List<VoiceEntry>
        {
            new("", speakers.Count == 0 ? "The model's own voice" : $"The model's first speaker ({speakers[0]})", "the model as it was trained"),
        };
        foreach (string speaker in speakers)
        {
            voices.Add(new VoiceEntry(speaker, speaker, "a speaker of the model"));
        }

        if (model is not VoicedSynthesizer)
        {
            return voices;                                                       // installed voices speak through a VoicedSynthesizer only
        }

        IReadOnlyList<VoiceProfile> installed;
        string location;
        try
        {
            var store = VoiceStores.Get(speech.VoiceStore);
            (installed, location) = (store.List(), store.Location);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or NotSupportedException or InvalidDataException or JsonException)
        {
            voices.Add(new VoiceEntry("?", "Installed voices", "the voice store could not be read", e.Message.Split('\n')[0].Trim()));
            return voices;
        }

        foreach (var voice in installed)
        {
            voices.Add(new VoiceEntry(voice.Name, voice.Name, $"installed ({voice.Kind}{(voice.Languages.Count > 0 ? ", " + string.Join(", ", voice.Languages) : "")}) in {location}",
                WhyVoiceCannot(voice, speakers, speech)));
        }

        return voices;
    }

    // Why an installed voice cannot speak here with this model, or null.
    private static string? WhyVoiceCannot(VoiceProfile voice, IReadOnlyList<string> speakers, SpeechSynthesisOptions speech)
    {
        switch (voice.Kind)
        {
            case VoiceKinds.Speaker:
                string? speaker = voice.Manifest["speaker"] is JsonValue value && value.TryGetValue(out string? name) ? name : null;
                return speaker is null ? "its voice.json names no speaker"
                    : speakers.Contains(speaker, StringComparer.OrdinalIgnoreCase) ? null
                    : $"it names the speaker '{speaker}', which this model does not have";
            case VoiceKinds.Embedding:
                if (string.Equals(speech.VoiceConversion, "off", StringComparison.OrdinalIgnoreCase))
                {
                    return null;                                                 // spoken in the model's own voice, unconverted
                }

                return voice.Encoder is not { } family ? "its voice.json names no encoder family"
                    : VoiceConverters.Find(family) is null ? $"it is converted by the '{family}' voice converter, which is not registered here (VoiceConverters)"
                    : null;
            case VoiceKinds.Checkpoint:
                if (voice.CheckpointFolder is not { } folder)
                {
                    return "it names no model folder";
                }

                try
                {
                    return SpeechSynthesisModels.TextNeeds(folder).Missing() is { } missing ? $"its model cannot speak here: {missing}" : null;
                }
                catch (Exception e) when (e is IOException or InvalidDataException or JsonException or NotSupportedException or UnauthorizedAccessException)
                {
                    return $"its model cannot be read here ({e.Message.Split('\n')[0].Trim()})";
                }

            default:
                return $"it is of kind '{voice.Kind}', which this library does not speak";
        }
    }

    // config.json's first "architectures" entry (else "model_type"), as SpeechSynthesisModels reads it; null when there is none.
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
                : config?["model_type"] is JsonValue single && single.TryGetValue(out string? one) ? one
                : null;
        }
        catch (Exception e) when (e is JsonException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    private static string ContentType(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    {
        ".wav" => "audio/wav",
        ".mp3" => "audio/mpeg",
        ".flac" => "audio/flac",
        _ => "audio/*",
    };

    // The synthesis host: a model is loaded once by its device and folder ("cpu|/path"), wrapped so installed voices speak.
    private ModelHost<ISpeechSynthesizer> Host => _models.Host<ISpeechSynthesizer>(ModelHostName, LoadSynthesizer);

    private static ISpeechSynthesizer LoadSynthesizer(string key)
    {
        int bar = key.IndexOf('|', StringComparison.Ordinal);
        var synthesizer = SpeechSynthesisModels.Load(key[(bar + 1)..], new SpeechSynthesisLoadOptions { Device = Device.Parse(key[..bar]) });
        return new VoicedSynthesizer(synthesizer);
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

    private string NoModel() => SynthesisModels.Count == 0 ? "No synthesis model to speak with: pull one in Models, then come back." : "Pick a synthesis model first.";

    private void Keep() => SpeakSettings.Save(_settings, _options);

    private async Task<bool> LoadAsync(SynthesisModelEntry model)
    {
        Error = "";
        ModelStatus = $"Loading {model.Title}…";
        var reporter = _progress.Start($"load {model.Title}", null, ProgressUnit.Elapsed);
        Work = _progress.ItemOf(reporter);
        var speech = _options.ToSynthesis();
        try
        {
            var (key, voices, device, rate, took) = await Task.Run(() =>
            {
                string key = $"{ChosenDevice()}|{model.Model.Folder}";
                using var lease = Host.Acquire(key);
                return (key, ListVoices(lease.Model, speech), lease.Model.Device.ToString(), lease.Model.SampleRate, lease.LoadDuration);
            }).ConfigureAwait(true);
            reporter.Finish($"{model.Title} on {device}");
            _loadedKey = key;
            ShowVoices(voices);
            ModelStatus = $"{model.Title} on {device}, {rate} Hz"
                + (took > TimeSpan.Zero ? $", loaded in {ProgressText.Duration(took)}" : "")
                + (voices.Count > 1 ? $"; {voices.Count - 1} voice{(voices.Count == 2 ? "" : "s")} to choose" : "");
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

    private async Task ListVoicesAsync()
    {
        if (_loadedKey is not { } key)
        {
            return;
        }

        var speech = _options.ToSynthesis();
        try
        {
            var voices = await Task.Run(() =>
            {
                using var lease = Host.Acquire(key);
                return ListVoices(lease.Model, speech);
            }).ConfigureAwait(true);
            ShowVoices(voices);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The voices could not be listed: {e.Message.Split('\n')[0].Trim()}";
        }
    }

    private void ShowVoices(IReadOnlyList<VoiceEntry> voices)
    {
        Voices.Clear();
        foreach (var voice in voices)
        {
            Voices.Add(voice);
        }

        OnPropertyChanged(nameof(HasVoices));
        string wanted = _options.Voice ?? "";
        _voice = voices.FirstOrDefault(v => string.Equals(v.Name, wanted, StringComparison.OrdinalIgnoreCase)) ?? voices[0];
        OnPropertyChanged(nameof(SelectedVoice));
        OnPropertyChanged(nameof(VoiceReason));
        OnPropertyChanged(nameof(HasVoiceReason));
    }

    // The speech's pieces as they are made: each kept (to play again and save), its making timed, a showing asked for.
    private async IAsyncEnumerable<AudioData> Made(IAsyncEnumerable<AudioData> speech, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        var pieces = speech.GetAsyncEnumerator(cancellationToken);
        try
        {
            while (true)
            {
                long asked = _time.GetTimestamp();
                if (!await pieces.MoveNextAsync().ConfigureAwait(false))
                {
                    break;
                }

                var piece = pieces.Current;
                _state.Add(piece, _time.GetElapsedTime(asked));
                _batcher.Touch();
                yield return piece;
            }
        }
        finally
        {
            await pieces.DisposeAsync().ConfigureAwait(false);
        }
    }

    // A completed speech's measures: the first after a load is cold, the later ones warm.
    private void Report(bool cold, SoundPlayback played, SpeakSnapshot snapshot)
    {
        double factor = snapshot.Seconds > 0 ? snapshot.Making.TotalSeconds / snapshot.Seconds : 0;
        FirstSound = played.FirstSound;
        RealTimeFactor = factor;
        string line = $"first sound after {(played.FirstSound is { } first ? AudioText.Time(first) : "-")}, real-time factor {AudioText.Factor(factor)}; "
            + $"{snapshot.Pieces} piece{(snapshot.Pieces == 1 ? "" : "s")}, {AudioText.Seconds(snapshot.Seconds)} made in {AudioText.Time(snapshot.Making)}";
        if (cold)
        {
            ColdText = "Cold (the first speech after a load): " + line;
            WarmText = "";
        }
        else
        {
            WarmText = "Warm: " + line;
        }
    }

    // On the UI thread: the speech's progress onto the page.
    private void Publish()
    {
        var s = _state.Snapshot();
        ProgressLine = s.Pieces == 0 ? "" : $"{s.Pieces} piece{(s.Pieces == 1 ? "" : "s")}, {AudioText.Seconds(s.Seconds)} of speech, made in {AudioText.Time(s.Making)}";
    }

    // What a speech has made so far.
    private sealed record SpeakSnapshot(int Pieces, double Seconds, TimeSpan Making);

    // The speech's pieces and measures, written by the making thread, read by showings.
    private sealed class SpeakState
    {
        private readonly Lock _gate = new();
        private readonly List<AudioData> _pieces = [];
        private double _seconds;
        private TimeSpan _making;

        public void Begin()
        {
            lock (_gate)
            {
                _pieces.Clear();
                _seconds = 0;
                _making = TimeSpan.Zero;
            }
        }

        public void Add(AudioData piece, TimeSpan making)
        {
            lock (_gate)
            {
                _pieces.Add(piece);
                _seconds += piece.Duration.TotalSeconds;
                _making += making;
            }
        }

        public SpeakSnapshot Snapshot()
        {
            lock (_gate)
            {
                return new SpeakSnapshot(_pieces.Count, _seconds, _making);
            }
        }

        // The pieces as one recording (each channel copied once), or null when none was made.
        public AudioData? Joined()
        {
            lock (_gate)
            {
                if (_pieces.Count == 0)
                {
                    return null;
                }

                if (_pieces.Count == 1)
                {
                    return _pieces[0];
                }

                var first = _pieces[0];
                var same = _pieces.Where(p => p.SampleRate == first.SampleRate && p.Channels == first.Channels).ToList();
                int length = same.Sum(p => p.Length), channels = first.Channels;
                var samples = GC.AllocateUninitializedArray<float>(channels * length);
                for (int c = 0; c < channels; c++)
                {
                    int at = c * length;
                    foreach (var piece in same)
                    {
                        piece.Channel(c).CopyTo(samples.AsSpan(at, piece.Length));
                        at += piece.Length;
                    }
                }

                return new AudioData(samples, channels, first.SampleRate);
            }
        }
    }
}
