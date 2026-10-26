// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Buffers;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Windows.Input;
using Idrak.Audio;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui.ViewModels;

/// <summary>A registered audio processor as a chip on the Audio process page: tapping it appends its step to the pipeline.</summary>
public sealed class ProcessorChip
{
    internal ProcessorChip(IAudioProcessor processor, Action<ProcessorChip> add)
    {
        AddCommand = new Command(() => add(this));
        Name = processor.Name;
        Label = processor.Label;
        Summary = processor.Summary;
        Live = processor.Live;
        LiveAlternative = processor.LiveAlternative;
        Parameters = [.. processor.Parameters.Select(Describe)];
        Step = StepOf(processor);
    }

    /// <summary>Appends <see cref="Step"/> to the page's pipeline.</summary>
    public ICommand AddCommand { get; }

    /// <summary>Its name in a pipeline ("highpass").</summary>
    public string Name { get; }

    /// <summary>Its name for people ("High-pass filter").</summary>
    public string Label { get; }

    /// <summary>What it does.</summary>
    public string Summary { get; }

    /// <summary>Whether it runs on sound as it comes (a live pipeline, such as monitoring, takes it).</summary>
    public bool Live { get; }

    /// <summary>For a whole-recording processor, the live one to use instead, or null.</summary>
    public string? LiveAlternative { get; }

    /// <summary>"live", or "whole recording" (with its live alternative).</summary>
    public string LiveText => Live ? "live" : LiveAlternative is { } live ? $"whole recording (live: {live})" : "whole recording";

    /// <summary>Its value and options, a line each: the key, label, unit, range, default and help.</summary>
    public IReadOnlyList<string> Parameters { get; }

    /// <summary>The parameters as one text (a line each), or "takes nothing".</summary>
    public string ParametersText => Parameters.Count == 0 ? "takes no value or option" : string.Join("\n", Parameters);

    /// <summary>The step a tap appends: its name, with its value's default (or a value in range when the value must be given).</summary>
    public string Step { get; }

    /// <inheritdoc />
    public override string ToString() => Name;

    private static string Describe(AudioProcessorParameter p)
    {
        string range = p.Range;
        return (p.Key.Length == 0 ? "value" : p.Key) + ": " + p.Label + (p.Unit is { } unit ? $" ({unit})" : "")
            + (range.Length > 0 ? $", {range}" : "") + (p.Choices.Count > 0 ? $", one of {string.Join("/", p.Choices)}" : "")
            + (p.Default is { } d ? $", default {d}" : p.Optional ? ", optional" : p.Key.Length == 0 ? ", required" : "")
            + (p.Help is { } help ? $": {help}" : "");
    }

    // The value, then every option that must be given (each with a value in its range).
    private static string StepOf(IAudioProcessor processor)
    {
        var step = new System.Text.StringBuilder(processor.Name);
        var value = processor.Parameters.FirstOrDefault(p => p.Key.Length == 0);
        if (value is not null && (value.Default is not null || !value.Optional))
        {
            step.Append('=').Append(value.Default ?? Sample(value));
        }

        foreach (var option in processor.Parameters.Where(p => p.Key.Length > 0 && !p.Optional && p.Default is null))
        {
            step.Append(' ').Append(option.Key).Append('=').Append(Sample(option));
        }

        return step.ToString();
    }

    // A value in the parameter's range for one that has no default: its first choice, true, 0 where it takes 0, else its
    // step or least value.
    private static string Sample(AudioProcessorParameter p) =>
        p.Input == AudioProcessorInput.Choice && p.Choices.Count > 0 ? p.Choices[0]
        : p.Input == AudioProcessorInput.Switch ? "true"
        : (p.Min is not { } min || min < 0 || min == 0 && !p.MinExclusive) && (p.Max is not { } max || max >= 0) ? "0"
        : p.Step is { } step && step >= (p.Min ?? double.MinValue) && step <= (p.Max ?? double.MaxValue) ? step.ToString(CultureInfo.InvariantCulture)
        : (p.Min ?? 1).ToString(CultureInfo.InvariantCulture);
}

/// <summary>
/// The Audio process page (plan 34, M6): a recording from the microphone (the permission asked first; the capture
/// <see cref="AudioCaptures.ForThisMachine"/> picks, aaudio on Android) or a file decoded by the library's own codecs
/// (<see cref="AudioCodecs"/>), run through a pipeline written as the library's text (<see cref="AudioPipeline"/>, the
/// registered processors as chips that append a step, checked as it is written), played before and after on the speaker
/// (<see cref="ISoundPlayer"/>: A/B, one at a time), with the pipeline's latency (the shift a recording's apply takes off)
/// and delay (what a live listener waits), the time it took cold (the first call of a pipeline, which compiles the code)
/// and warm, and the level before and after; the result saved and shared. Monitoring (off by default) plays the
/// microphone through the pipeline (live steps only) on the speaker with its delay shown; it feeds back without
/// headphones. The page is updated at most once a progress interval, not once an audio block.
/// </summary>
public sealed class AudioProcessViewModel : ViewModel, IDisposable
{
    private readonly ISoundPlayer _player;
    private readonly IPermissionService _permissions;
    private readonly IShareService _share;
    private readonly ISettingsStore _settings;
    private readonly ProgressCenter _progress;
    private readonly IFilePicker? _picker;
    private readonly TimeProvider _time;
    private readonly UiBatcher _batcher;
    private readonly LiveMeter _meter = new();
    private readonly CancellationTokenSource _abort = new();
    private readonly Dictionary<string, TimeSpan> _cold = new(StringComparer.Ordinal);
    private AudioProcessLabOptions _options;
    private CancellationTokenSource? _stop;
    private AudioData? _original, _processed;
    private string _pipelineText;
    private int _playSequence;
    private bool _recording, _applying, _monitoring, _playing, _refused, _showSettings;
    private string _pipelineCheck = "", _pipelineError = "", _status = "", _error = "", _originalText = "", _processedText = "", _latencyText = "", _coldText = "",
        _warmText = "", _beforeText = "", _afterText = "", _savedText = "", _settingsMessage = "", _playingText = "", _elapsed = "00:00", _monitorText = "", _levelText = "";
    private double _level;

    /// <summary>The page over the app's services; <paramref name="picker"/> is MAUI's own when null.</summary>
    public AudioProcessViewModel(ISoundPlayer player, IPermissionService permissions, IShareService share, IdrakMauiOptions options, ISettingsStore settings,
        ProgressCenter? progress = null, IFilePicker? picker = null)
    {
        _player = player ?? throw new ArgumentNullException(nameof(player));
        _permissions = permissions ?? throw new ArgumentNullException(nameof(permissions));
        _share = share ?? throw new ArgumentNullException(nameof(share));
        ArgumentNullException.ThrowIfNull(options);
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        _progress = progress ?? new ProgressCenter(options.ProgressInterval);
        _picker = picker;
        _time = _progress.Time;
        _batcher = new UiBatcher(_progress, Publish);
        _options = AudioProcessSettings.Load(settings, out var skipped);
        _pipelineText = _options.Pipeline ?? "";
        if (skipped.Count > 0)
        {
            _settingsMessage = "A saved setting was not read: " + string.Join("; ", skipped);
        }

        foreach (var setting in AudioProcessSettings.Definitions)
        {
            Settings.Add(new LabSettingEntry(setting, setting.Get(_options)));
        }

        RefreshProcessors();
        CheckPipeline();
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
        PickFileCommand = new Command(async () => await PickFileAsync());
        AddStepCommand = new Command<ProcessorChip>(AddStep);
        ClearPipelineCommand = new Command(() => PipelineText = "");
        ApplyCommand = new Command(async () => await ApplyAsync());
        PlayOriginalCommand = new Command(async () => await PlayAsync(original: true));
        PlayProcessedCommand = new Command(async () => await PlayAsync(original: false));
        StopCommand = new Command(Stop);
        MonitorCommand = new Command(async () =>
        {
            if (IsMonitoring)
            {
                Stop();
            }
            else
            {
                await MonitorAsync();
            }
        });
        SaveCommand = new Command(async () => await SaveAsync());
        ShareCommand = new Command(async () => await ShareAsync());
        OpenPermissionSettingsCommand = new Command(() => _permissions.OpenSettings());
        ApplySettingsCommand = new Command(() => ApplySettings());
        ToggleSettingsCommand = new Command(() => ShowSettings = !ShowSettings);
    }

    /// <summary>The page's title.</summary>
    public string Title => "Audio process";

    /// <summary>What monitoring warns of: the speaker feeds back into the microphone without headphones.</summary>
    public string MonitorWarning => "Monitoring plays the microphone on the speaker as it is heard: use headphones, or the speaker feeds back into the microphone (a howl).";

    /// <summary>The settings in effect (<see cref="ApplySettings"/> checks and keeps the page's).</summary>
    public AudioProcessLabOptions Options => _options;

    /// <summary>The registered processors as chips, in registration order.</summary>
    public ObservableCollection<ProcessorChip> Processors { get; } = [];

    /// <summary>The pipeline as written (<see cref="AudioPipeline"/> text: steps comma-separated, each a name, a value and options); checked as it changes.</summary>
    public string PipelineText
    {
        get => _pipelineText;
        set
        {
            if (SetProperty(ref _pipelineText, value ?? ""))
            {
                CheckPipeline();
            }
        }
    }

    /// <summary>What the pipeline is ("3 steps, all live"), or "" when it does not read.</summary>
    public string PipelineCheck
    {
        get => _pipelineCheck;
        private set => SetProperty(ref _pipelineCheck, value);
    }

    /// <summary>Why the pipeline does not read (an unknown processor or option, a value out of range), or "".</summary>
    public string PipelineError
    {
        get => _pipelineError;
        private set
        {
            if (SetProperty(ref _pipelineError, value))
            {
                OnPropertyChanged(nameof(HasPipelineError));
            }
        }
    }

    /// <summary>Whether <see cref="PipelineError"/> has something to show.</summary>
    public bool HasPipelineError => PipelineError.Length > 0;

    /// <summary>The sound recorded or picked, or null.</summary>
    public AudioData? Original
    {
        get => _original;
        private set
        {
            if (SetProperty(ref _original, value))
            {
                OnPropertyChanged(nameof(HasOriginal));
            }
        }
    }

    /// <summary>Whether there is a sound to process.</summary>
    public bool HasOriginal => Original is not null;

    /// <summary>The sound as processed by the last apply, or null.</summary>
    public AudioData? Processed
    {
        get => _processed;
        private set
        {
            if (SetProperty(ref _processed, value))
            {
                OnPropertyChanged(nameof(HasProcessed));
            }
        }
    }

    /// <summary>Whether there is a processed sound to play, save or share.</summary>
    public bool HasProcessed => Processed is not null;

    /// <summary>The original as a line ("12.3 s at 16000 Hz, 1 channel, from the microphone"), or "".</summary>
    public string OriginalText
    {
        get => _originalText;
        private set => SetProperty(ref _originalText, value);
    }

    /// <summary>The processed sound as a line, with the pipeline it went through, or "".</summary>
    public string ProcessedText
    {
        get => _processedText;
        private set => SetProperty(ref _processedText, value);
    }

    /// <summary>The pipeline's latency and delay at the sound's rate, or "".</summary>
    public string LatencyText
    {
        get => _latencyText;
        private set => SetProperty(ref _latencyText, value);
    }

    /// <summary>The pipeline's latency of the last apply, in samples of its output (the shift a recording's apply takes off).</summary>
    public int Latency { get; private set; }

    /// <summary>The pipeline's delay of the last apply, in samples of its output (what a live listener waits).</summary>
    public int Delay { get; private set; }

    /// <summary>The first apply of the pipeline (cold: it compiles the code): its time and real-time factor, or "".</summary>
    public string ColdText
    {
        get => _coldText;
        private set => SetProperty(ref _coldText, value);
    }

    /// <summary>The latest warm apply of the pipeline: its time and real-time factor, or "".</summary>
    public string WarmText
    {
        get => _warmText;
        private set => SetProperty(ref _warmText, value);
    }

    /// <summary>The cold time of the last pipeline applied, or null.</summary>
    public TimeSpan? ColdTime { get; private set; }

    /// <summary>The warm time of the last apply, or null.</summary>
    public TimeSpan? WarmTime { get; private set; }

    /// <summary>The original's level ("RMS -23.1 dB, peak -3.0 dB"), or "".</summary>
    public string BeforeText
    {
        get => _beforeText;
        private set => SetProperty(ref _beforeText, value);
    }

    /// <summary>The processed sound's level, or "".</summary>
    public string AfterText
    {
        get => _afterText;
        private set => SetProperty(ref _afterText, value);
    }

    /// <summary>Whether the microphone is being recorded (the record button stops it).</summary>
    public bool IsRecording
    {
        get => _recording;
        private set
        {
            if (SetProperty(ref _recording, value))
            {
                OnPropertyChanged(nameof(RecordText));
                OnPropertyChanged(nameof(RecordGlyph));
                OnPropertyChanged(nameof(IsIdle));
            }
        }
    }

    /// <summary>The record button's words.</summary>
    public string RecordText => IsRecording ? "Stop" : "Record";

    /// <summary>The record button's mark.</summary>
    public string RecordGlyph => IsRecording ? "■" : "●";

    /// <summary>Whether the pipeline is being applied.</summary>
    public bool IsApplying
    {
        get => _applying;
        private set
        {
            if (SetProperty(ref _applying, value))
            {
                OnPropertyChanged(nameof(IsIdle));
            }
        }
    }

    /// <summary>Whether the microphone plays through the pipeline on the speaker.</summary>
    public bool IsMonitoring
    {
        get => _monitoring;
        private set
        {
            if (SetProperty(ref _monitoring, value))
            {
                OnPropertyChanged(nameof(MonitorButtonText));
                OnPropertyChanged(nameof(IsIdle));
            }
        }
    }

    /// <summary>The monitor button's words.</summary>
    public string MonitorButtonText => IsMonitoring ? "Stop monitoring" : "Monitor live";

    /// <summary>Monitoring's state and delay ("delay about 95 ms: the pipeline 3 ms, …"), or "".</summary>
    public string MonitorText
    {
        get => _monitorText;
        private set => SetProperty(ref _monitorText, value);
    }

    /// <summary>The delay monitoring adds, in seconds (the pipeline's, a captured block and the playback buffer), or 0 before it ran.</summary>
    public double MonitorDelaySeconds { get; private set; }

    /// <summary>Whether a recording plays (original or processed).</summary>
    public bool IsPlaying
    {
        get => _playing;
        private set => SetProperty(ref _playing, value);
    }

    /// <summary>What plays ("Playing the original"), or "".</summary>
    public string PlayingText
    {
        get => _playingText;
        private set => SetProperty(ref _playingText, value);
    }

    /// <summary>Whether nothing records, applies or monitors.</summary>
    public bool IsIdle => !IsRecording && !IsApplying && !IsMonitoring;

    /// <summary>The level of the latest block heard (recording or monitoring), from 0 (-60 dB or under) to 1.</summary>
    public double Level
    {
        get => _level;
        private set => SetProperty(ref _level, value);
    }

    /// <summary>The level of the latest block heard as text, or "".</summary>
    public string LevelText
    {
        get => _levelText;
        private set => SetProperty(ref _levelText, value);
    }

    /// <summary>The time recorded or monitored ("00:12").</summary>
    public string ElapsedText
    {
        get => _elapsed;
        private set => SetProperty(ref _elapsed, value);
    }

    /// <summary>What is going on or what the last action did, or "".</summary>
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

    /// <summary>Where the processed sound was saved last, or "".</summary>
    public string SavedText
    {
        get => _savedText;
        private set => SetProperty(ref _savedText, value);
    }

    /// <summary>The path of the file saved last, or null.</summary>
    public string? SavedPath { get; private set; }

    /// <summary>The settings on the page (empty: the default, shown in the field).</summary>
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

    /// <summary>The UI showings made while recording or monitoring (at most one a progress interval).</summary>
    public int Showings => _batcher.Dispatches;

    /// <summary>Records from the microphone, or stops the recording.</summary>
    public ICommand RecordCommand { get; }

    /// <summary>Picks a sound file to process.</summary>
    public ICommand PickFileCommand { get; }

    /// <summary>Appends a chip's step to the pipeline.</summary>
    public ICommand AddStepCommand { get; }

    /// <summary>Empties the pipeline.</summary>
    public ICommand ClearPipelineCommand { get; }

    /// <summary>Applies the pipeline to the sound.</summary>
    public ICommand ApplyCommand { get; }

    /// <summary>Plays the original.</summary>
    public ICommand PlayOriginalCommand { get; }

    /// <summary>Plays the processed sound.</summary>
    public ICommand PlayProcessedCommand { get; }

    /// <summary>Stops what records, monitors or plays.</summary>
    public ICommand StopCommand { get; }

    /// <summary>Starts or stops monitoring.</summary>
    public ICommand MonitorCommand { get; }

    /// <summary>Saves the processed sound.</summary>
    public ICommand SaveCommand { get; }

    /// <summary>Saves the processed sound and offers it to the share sheet.</summary>
    public ICommand ShareCommand { get; }

    /// <summary>Opens the app's page in the system settings (a refused microphone is allowed there).</summary>
    public ICommand OpenPermissionSettingsCommand { get; }

    /// <summary>Checks and keeps the settings (<see cref="ApplySettings"/>).</summary>
    public ICommand ApplySettingsCommand { get; }

    /// <summary>Opens or closes the settings card.</summary>
    public ICommand ToggleSettingsCommand { get; }

    /// <summary>Lists the registered processors again (a plug-in may have registered one since).</summary>
    public void RefreshProcessors()
    {
        Processors.Clear();
        foreach (string name in AudioProcessors.Names)
        {
            if (AudioProcessors.Find(name) is { } processor)
            {
                Processors.Add(new ProcessorChip(processor, AddStep));
            }
        }
    }

    /// <summary>Appends <paramref name="chip"/>'s step to the pipeline (after a comma), and keeps the pipeline.</summary>
    public void AddStep(ProcessorChip? chip)
    {
        if (chip is null)
        {
            return;
        }

        string text = PipelineText.TrimEnd().TrimEnd(',').TrimEnd();
        PipelineText = text.Length == 0 ? chip.Step : $"{text}, {chip.Step}";
        _options.Pipeline = PipelineText.Length == 0 ? null : PipelineText;
        Keep();
    }

    /// <summary>Uses <paramref name="audio"/> as the sound to process (<paramref name="from"/> says where it came from), clearing the last result.</summary>
    public void Use(AudioData audio, string from)
    {
        ArgumentNullException.ThrowIfNull(audio);
        Original = audio;
        Processed = null;
        ProcessedText = AfterText = SavedText = LatencyText = "";
        OriginalText = $"{AudioText.Seconds(audio.Duration.TotalSeconds)} at {audio.SampleRate} Hz, {audio.Channels} channel{(audio.Channels == 1 ? "" : "s")}, {from}";
        BeforeText = LevelOf(audio);
    }

    /// <summary>
    /// Records from the microphone until <see cref="Stop"/> (or the longest recording): the permission asked first, the
    /// capture opened, its blocks kept in pooled buffers, the level shown; true when a recording was made.
    /// </summary>
    public async Task<bool> RecordAsync()
    {
        if (!IsIdle)
        {
            return false;
        }

        if (await MicrophoneAsync().ConfigureAwait(true) is not { } capture)
        {
            return false;
        }

        _player.Stop();                                                          // the speaker would be recorded
        IAudioInput input;
        try
        {
            input = capture.Open(_options.ToCapture());
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The microphone could not be opened ({capture.Name}): {e.Message}";
            return false;
        }

        using var stop = CancellationTokenSource.CreateLinkedTokenSource(_abort.Token);
        _stop = stop;
        double longest = _options.MaxRecordSeconds ?? AudioProcessSettings.DefaultMaxRecordSeconds;
        var recording = new RecordingBuffer(input.Channels, input.SampleRate, (long)Math.Ceiling(longest * input.SampleRate));
        _meter.Begin(_time.GetTimestamp());
        IsRecording = true;
        Status = $"Recording: {input.Description}, {input.SampleRate} Hz";
        _batcher.Start();
        _batcher.Touch(force: true);
        try
        {
            await Task.Run(async () =>
            {
                await foreach (var block in input.ReadAsync(stop.Token).ConfigureAwait(false))
                {
                    bool full = recording.Append(block);
                    _meter.Set(LevelMeter.Decibels(block.Samples), recording.Seconds);
                    _batcher.Touch();
                    if (full)
                    {
                        break;
                    }
                }
            }, CancellationToken.None).ConfigureAwait(true);
        }
        catch (OperationCanceledException) when (stop.IsCancellationRequested)
        {
            // Stopped.
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The recording stopped: {e.Message.Split('\n')[0].Trim()}";
        }
        finally
        {
            input.Dispose();
            _stop = null;
            _batcher.End();
            Publish();
            IsRecording = false;
        }

        using (recording)
        {
            if (recording.Length == 0)
            {
                Status = "Nothing was recorded.";
                return false;
            }

            Use(recording.ToAudio(), $"recorded from {capture.Name}{(recording.Full ? $" (stopped at the longest recording, {AudioText.Seconds(longest)})" : "")}");
        }

        Status = $"Recorded {AudioText.Seconds(Original!.Duration.TotalSeconds)}.";
        return true;
    }

    /// <summary>Uses the sound file at <paramref name="path"/>, decoded by the library's codecs; false when it is not one.</summary>
    /// <param name="path">The file.</param>
    /// <param name="name">Its name as shown (null: the file's name).</param>
    public async Task<bool> OpenFileAsync(string path, string? name = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        name ??= Path.GetFileName(path);
        Error = "";
        try
        {
            var audio = await Task.Run(() => AudioCodecs.Decode(path)).ConfigureAwait(true);
            Use(audio, $"from {name}");
            Status = $"Opened {name}.";
            return true;
        }
        catch (Exception e) when (e is InvalidDataException or NotSupportedException or IOException or UnauthorizedAccessException)
        {
            Error = $"{name} could not be read: {e.Message}";
            return false;
        }
    }

    /// <summary>Picks a sound file (MAUI's file picker) and uses it; false when none was picked or it failed.</summary>
    public async Task<bool> PickFileAsync()
    {
        if (!IsIdle)
        {
            return false;
        }

        FileResult? picked;
        try
        {
            picked = await (_picker ?? FilePicker.Default).PickAsync(new PickOptions { PickerTitle = "A sound to process", FileTypes = AudioFileTypes() }).ConfigureAwait(true);
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
                string folder = Path.Combine(ComputeResources.CacheFolder, "audio-process");
                Directory.CreateDirectory(folder);
                copy = path = Path.Combine(folder, Path.GetFileName(picked.FileName) is { Length: > 0 } file ? file : "picked-audio");
                await using var from = await picked.OpenReadAsync().ConfigureAwait(true);
                await using var to = File.Create(copy);
                await from.CopyToAsync(to).ConfigureAwait(true);
            }

            return await OpenFileAsync(path, picked.FileName).ConfigureAwait(true);
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

    /// <summary>
    /// Applies the pipeline to the sound on a worker thread: its latency and delay read from a stage opened at the sound's
    /// rate and channels, then the sound run through it (whole, or in blocks as live sound goes), timed cold (the pipeline's
    /// first apply) and warm (a second run, unless the settings turn it off); the levels before and after. True when it ran.
    /// </summary>
    public async Task<bool> ApplyAsync()
    {
        if (!IsIdle)
        {
            return false;
        }

        Error = "";
        if (Original is not { } audio)
        {
            Error = "Record or pick a sound first.";
            return false;
        }

        AudioPipeline pipeline;
        try
        {
            pipeline = AudioPipeline.Parse(PipelineText);
        }
        catch (ArgumentException e)
        {
            PipelineError = e.Message;
            Error = $"The pipeline does not read: {e.Message}";
            return false;
        }

        _options.Pipeline = PipelineText.Length == 0 ? null : PipelineText;
        Keep();
        IsApplying = true;
        Status = pipeline.IsEmpty ? "No steps: the sound as it is." : $"Applying {pipeline}…";
        double block = _options.ProcessBlockSeconds ?? 0;
        bool warmRun = _options.WarmRun ?? true;
        try
        {
            var result = await Task.Run(() =>
            {
                int latency = 0, delay = 0, rate = audio.SampleRate;
                if (!pipeline.IsEmpty)
                {
                    using var stage = pipeline.Open(audio.SampleRate, audio.Channels);
                    (latency, delay, rate) = (stage.Latency, stage.Delay, stage.SampleRate);
                }

                var (processed, first) = Run(pipeline, audio, block);
                TimeSpan? again = warmRun && !pipeline.IsEmpty ? Run(pipeline, audio, block).Elapsed : null;
                return (processed, first, again, latency, delay, rate, after: LevelOf(processed));
            }, CancellationToken.None).ConfigureAwait(true);
            string text = pipeline.ToString();
            bool cold = !_cold.ContainsKey(text);
            if (cold)
            {
                _cold[text] = result.first;
            }

            double seconds = audio.Duration.TotalSeconds;
            ColdTime = _cold[text];
            WarmTime = cold ? result.again : result.again ?? result.first;
            ColdText = $"Cold (the pipeline's first apply): {Timing(ColdTime.Value, seconds)}";
            WarmText = WarmTime is { } warm ? $"Warm: {Timing(warm, seconds)}" : "Warm: off in the settings (Time it warm too).";
            (Latency, Delay) = (result.latency, result.delay);
            LatencyText = $"Latency {result.latency} samples ({AudioText.Milliseconds(result.latency * 1000.0 / result.rate)}): the shift an apply takes off, so the result is aligned. "
                + $"Delay {result.delay} samples ({AudioText.Milliseconds(result.delay * 1000.0 / result.rate)}): what a live listener waits.";
            Processed = result.processed;
            AfterText = result.after;
            ProcessedText = $"{AudioText.Seconds(result.processed.Duration.TotalSeconds)} at {result.processed.SampleRate} Hz, {result.processed.Channels} channel{(result.processed.Channels == 1 ? "" : "s")}"
                + (pipeline.IsEmpty ? ", as it was" : $", through {pipeline}{(block > 0 ? $" in blocks of {AudioText.Seconds(block)}" : "")}");
            Status = pipeline.IsEmpty ? "No steps: the processed sound is the original." : $"Applied {pipeline.Steps.Count} step{(pipeline.Steps.Count == 1 ? "" : "s")}.";
            return true;
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException or InvalidDataException or IOException)
        {
            Error = $"The pipeline did not run: {e.Message.Split('\n')[0].Trim()}";      // a step that cannot run at this rate or channel count
            Status = "";
            return false;
        }
        finally
        {
            IsApplying = false;
        }
    }

    /// <summary>Plays the original (<paramref name="original"/>) or the processed sound; another play stops the one before (A/B). True when it played to its end.</summary>
    public async Task<bool> PlayAsync(bool original)
    {
        var audio = original ? Original : Processed;
        string label = original ? "original" : "processed";
        if (audio is null)
        {
            Status = original ? "Record or pick a sound first." : "Apply the pipeline first.";
            return false;
        }

        if (IsRecording || IsMonitoring)
        {
            Status = "Stop the microphone first.";
            return false;
        }

        Error = "";
        int sequence = Interlocked.Increment(ref _playSequence);
        IsPlaying = true;
        PlayingText = $"Playing the {label}…";
        try
        {
            var request = _options.ToPlayback(label);
            var played = await Task.Run(() => _player.PlayAsync(audio, request), CancellationToken.None).ConfigureAwait(true);
            if (sequence == _playSequence)
            {
                Status = played.Completed ? $"Played the {label} ({AudioText.Seconds(played.Seconds)}) on {played.Output}." : $"Stopped the {label}.";
            }

            return played.Completed;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The {label} was not played: {e.Message.Split('\n')[0].Trim()}";
            return false;
        }
        finally
        {
            if (sequence == _playSequence)
            {
                IsPlaying = false;
                PlayingText = "";
            }
        }
    }

    /// <summary>
    /// Plays the microphone through the pipeline on the speaker until <see cref="Stop"/>: the permission asked first, the
    /// pipeline checked live (a whole-recording step refused by name), the capture and the playback opened with short
    /// blocks and buffer; its delay shown. True when it ran.
    /// </summary>
    public async Task<bool> MonitorAsync()
    {
        if (!IsIdle)
        {
            return false;
        }

        Error = "";
        AudioPipeline pipeline;
        try
        {
            pipeline = AudioPipeline.Parse(PipelineText);
            pipeline.ThrowIfNotLive();
        }
        catch (ArgumentException e)
        {
            Error = $"Monitoring runs live steps only: {e.Message}";
            return false;
        }

        SoundPlayRequest request;
        try
        {
            request = _options.ToMonitorPlayback();
            request.Output.Check();
        }
        catch (ArgumentException e)
        {
            Error = e.Message;
            return false;
        }

        if (await MicrophoneAsync().ConfigureAwait(true) is not { } capture)
        {
            return false;
        }

        var captureOptions = _options.ToMonitorCapture();
        IAudioInput input;
        IAudioStage stage;
        try
        {
            input = capture.Open(captureOptions);
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The microphone could not be opened ({capture.Name}): {e.Message}";
            return false;
        }

        try
        {
            stage = pipeline.Open(input.SampleRate, input.Channels);
        }
        catch (ArgumentException e)
        {
            input.Dispose();
            Error = $"The pipeline cannot run on the microphone's sound: {e.Message}";
            return false;
        }

        double pipelineDelay = (double)stage.Delay / stage.SampleRate, total = pipelineDelay + captureOptions.BlockSeconds + request.Output.BufferSeconds;
        MonitorDelaySeconds = total;
        MonitorText = $"Delay about {AudioText.Milliseconds(total * 1000)}: the pipeline {AudioText.Milliseconds(pipelineDelay * 1000)}, a captured block "
            + $"{AudioText.Milliseconds(captureOptions.BlockSeconds * 1000)}, the playback buffer up to {AudioText.Milliseconds(request.Output.BufferSeconds * 1000)} (and the device's own).";
        using var stop = CancellationTokenSource.CreateLinkedTokenSource(_abort.Token);
        _stop = stop;
        _meter.Begin(_time.GetTimestamp());
        IsMonitoring = true;
        Status = $"Monitoring {input.Description} through {(pipeline.IsEmpty ? "no steps" : pipeline.ToString())}";
        _batcher.Start();
        _batcher.Touch(force: true);
        try
        {
            var played = await Task.Run(() => _player.PlayAsync(Monitored(input, stage, stop.Token), request, stop.Token), CancellationToken.None).ConfigureAwait(true);
            Status = $"Monitored {AudioText.Seconds(played.Seconds)} on {played.Output}.";
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"Monitoring stopped: {e.Message.Split('\n')[0].Trim()}";
            Status = "";
            return false;
        }
        finally
        {
            stage.Dispose();
            input.Dispose();
            _stop = null;
            _batcher.End();
            Publish();
            IsMonitoring = false;
        }
    }

    /// <summary>Stops what runs: the recording (what was heard is kept), monitoring, or the sound playing.</summary>
    public void Stop()
    {
        try
        {
            if (_stop is { IsCancellationRequested: false } stop)
            {
                stop.Cancel();
            }
        }
        catch (ObjectDisposedException)
        {
            // It ended meanwhile.
        }

        _player.Stop();
    }

    /// <summary>
    /// Saves the processed sound with the settings' encoder (<paramref name="format"/>: another encoder's name) in the cache
    /// folder's <c>audio-process</c> folder; the file's path, or null when there is none or it failed.
    /// </summary>
    public async Task<string?> SaveAsync(string? format = null)
    {
        if (Processed is not { } processed)
        {
            Status = "Nothing to save yet: apply the pipeline first.";
            return null;
        }

        Error = "";
        format ??= _options.SaveFormat ?? "wav";
        try
        {
            var encoder = AudioEncoders.Get(format);
            var encode = encoder.Name.Equals("wav", StringComparison.OrdinalIgnoreCase) ? _options.ToEncode() : new AudioEncodeOptions();
            string folder = Path.Combine(ComputeResources.CacheFolder, "audio-process");
            string path = Path.Combine(folder, $"processed-{_time.GetLocalNow():yyyyMMdd-HHmmss}{encoder.Extension}");
            await Task.Run(() => AudioEncoders.Save(path, processed, encode)).ConfigureAwait(true);
            SavedPath = path;
            SavedText = $"Saved {Path.GetFileName(path)} ({ProgressText.Bytes(new FileInfo(path).Length)}) in {folder}.";
            return path;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The processed sound was not saved as {format}: {e.Message.Split('\n')[0].Trim()}";
            return null;
        }
    }

    /// <summary>Saves the processed sound and offers it to the phone's share sheet; false when there is none.</summary>
    public async Task<bool> ShareAsync()
    {
        if (await SaveAsync().ConfigureAwait(true) is not { } path)
        {
            return false;
        }

        try
        {
            await _share.ShareFileAsync(path, "Processed sound", Path.GetExtension(path).Equals(".mp3", StringComparison.OrdinalIgnoreCase) ? "audio/mpeg" : "audio/wav")
                .ConfigureAwait(true);
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The processed sound was not shared: {e.Message}";
            return false;
        }
    }

    /// <summary>Checks every setting on the page and the options they give, then keeps them; false, and nothing kept, when one is refused (each says why).</summary>
    public bool ApplySettings()
    {
        if (LabSettingEntry.Apply(Settings, _options.Clone()) is not { } next)
        {
            SettingsMessage = "Nothing was kept: a value above is not one the setting takes.";
            return false;
        }

        try
        {
            next.ToCapture().Check();
            next.ToMonitorCapture().Check();
            next.ToPlayback("check").Output.Check();
            next.ToMonitorPlayback().Output.Check();
        }
        catch (ArgumentException e)
        {
            SettingsMessage = $"Nothing was kept: {e.Message}";
            return false;
        }

        next.Pipeline = PipelineText.Length == 0 ? null : PipelineText;
        _options = next;
        Keep();
        SettingsMessage = IsIdle ? "Kept." : "Kept: they apply from the next run.";
        return true;
    }

    /// <summary>Stops what runs at once and stops showing.</summary>
    public void Dispose()
    {
        _abort.Cancel();
        _player.Stop();
        _batcher.Dispose();
    }

    // The sound through the pipeline, timed: the whole recording at once (Apply), or block by block through one stage
    // (each block a copy, as a capture gives new blocks; what the stage holds given when flushed; its shift taken off).
    internal static (AudioData Audio, TimeSpan Elapsed) Run(AudioPipeline pipeline, AudioData audio, double blockSeconds)
    {
        long start = System.Diagnostics.Stopwatch.GetTimestamp();
        if (pipeline.IsEmpty)
        {
            return (audio, TimeSpan.Zero);
        }

        int block = blockSeconds > 0 ? Math.Max(1, (int)Math.Round(blockSeconds * audio.SampleRate)) : 0;
        if (block == 0)
        {
            var whole = pipeline.Apply(audio);
            return (whole, System.Diagnostics.Stopwatch.GetElapsedTime(start));
        }

        using var stage = pipeline.Open(audio.SampleRate, audio.Channels);
        var parts = new List<AudioData>((audio.Length + block - 1) / block + 1);
        for (int from = 0; from < audio.Length; from += block)
        {
            var part = stage.Process(audio.Slice(from, Math.Min(block, audio.Length - from)));
            if (part.Length > 0)
            {
                parts.Add(part);
            }
        }

        var tail = stage.Flush();
        if (tail.Length > 0)
        {
            parts.Add(tail);
        }

        var joined = Join(parts, stage.Channels, stage.SampleRate, stage.Latency);
        return (joined, System.Diagnostics.Stopwatch.GetElapsedTime(start));
    }

    // The parts in order as one recording (planar), its first `skip` samples (the stages' shift) left out.
    private static AudioData Join(List<AudioData> parts, int channels, int sampleRate, int skip)
    {
        if (parts.Count == 1 && skip == 0)
        {
            return parts[0];
        }

        int length = Math.Max(0, parts.Sum(p => p.Length) - skip);
        var samples = GC.AllocateUninitializedArray<float>(channels * length);
        for (int c = 0; c < channels; c++)
        {
            int at = c * length, left = skip;
            foreach (var part in parts)
            {
                int from = Math.Min(left, part.Length);
                left -= from;
                part.Channel(c)[from..].CopyTo(samples.AsSpan(at, part.Length - from));
                at += part.Length - from;
            }
        }

        return new AudioData(samples, channels, sampleRate);
    }

    private static string LevelOf(AudioData audio) =>
        $"RMS {AudioText.Decibels(LevelMeter.Decibels(audio.Samples))}, peak {AudioText.Decibels(LevelMeter.PeakDecibels(audio.Samples))}";

    private static string Timing(TimeSpan time, double seconds) =>
        $"{AudioText.Time(time)} for {AudioText.Seconds(seconds)} of sound, real-time factor {AudioText.Factor(time.TotalSeconds / Math.Max(1e-9, seconds))}";

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

    // The pipeline text checked against the registry as it is written.
    private void CheckPipeline()
    {
        try
        {
            var pipeline = AudioPipeline.Parse(PipelineText);
            PipelineError = "";
            if (pipeline.IsEmpty)
            {
                PipelineCheck = "No steps yet: tap a processor below, or write one (highpass=80, denoise=1.5, normalize=-20).";
                return;
            }

            var whole = pipeline.Steps.Where(s => !AudioProcessors.Get(s.Name).Live).Select(s => s.Name).ToList();
            PipelineCheck = $"{pipeline.Steps.Count} step{(pipeline.Steps.Count == 1 ? "" : "s")}: "
                + (whole.Count == 0 ? "all live (monitoring takes it)" : $"{string.Join(", ", whole)} need{(whole.Count == 1 ? "s" : "")} the whole recording (monitoring takes live steps only)");
        }
        catch (ArgumentException e)
        {
            PipelineCheck = "";
            PipelineError = e.Message;
        }
    }

    // The microphone, asked for first: the capture of the settings when it can record here, else null with why not.
    private async Task<IAudioCapture?> MicrophoneAsync()
    {
        Error = "";
        IsPermissionRefused = false;
        var permission = await _permissions.RequestAsync(PhoneNeeds.Microphone).ConfigureAwait(true);
        if (!permission.Granted)
        {
            IsPermissionRefused = true;
            Error = permission.Message.Length > 0 ? permission.Message
                : "The app may not use the microphone. Allow it in the app's settings (Permissions), then try again.";
            return null;
        }

        try
        {
            var capture = _options.Capture is { Length: > 0 } name ? AudioCaptures.Get(name) : AudioCaptures.ForThisMachine();
            if (!capture.IsAvailable(out string reason))
            {
                throw new NotSupportedException($"The {capture.Name} capture cannot record here: {reason}.");
            }

            _options.ToCapture().Check();
            return capture;
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            Error = e.Message;
            return null;
        }
    }

    // The microphone's blocks through the stage (in place), each metered where it is; a showing asked for, not made per block.
    private async IAsyncEnumerable<AudioData> Monitored(IAudioInput input, IAudioStage stage, [EnumeratorCancellation] CancellationToken cancellationToken)
    {
        double heard = 0;
        var blocks = input.ReadAsync(cancellationToken).GetAsyncEnumerator(cancellationToken);
        try
        {
            while (true)
            {
                bool more;
                try
                {
                    more = await blocks.MoveNextAsync().ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (cancellationToken.IsCancellationRequested)
                {
                    yield break;
                }

                if (!more)
                {
                    break;
                }

                var block = blocks.Current;
                heard += block.Duration.TotalSeconds;
                var cleaned = stage.Process(block);
                _meter.Set(LevelMeter.Decibels(cleaned.Length > 0 ? cleaned.Samples : block.Samples), heard);
                _batcher.Touch();
                if (cleaned.Length > 0)
                {
                    yield return cleaned;
                }
            }
        }
        finally
        {
            await blocks.DisposeAsync().ConfigureAwait(false);
        }
    }

    private void Keep() => AudioProcessSettings.Save(_settings, _options);

    // On the UI thread: the meter onto the page.
    private void Publish()
    {
        var (decibels, seconds, started) = _meter.Read();
        Level = double.IsFinite(decibels) ? Math.Clamp(1 - decibels / -60, 0, 1) : 0;
        LevelText = double.IsFinite(decibels) ? AudioText.Decibels(decibels) : "";
        ElapsedText = AudioText.Clock(started == 0 ? TimeSpan.Zero : _time.GetElapsedTime(started));
        if (seconds > 0 && IsRecording)
        {
            Status = $"Recording: {AudioText.Seconds(seconds)} heard";
        }
    }

    // The level of the latest block and the seconds heard, written by the reading thread, read by showings.
    private sealed class LiveMeter
    {
        private readonly Lock _gate = new();
        private double _decibels = double.NegativeInfinity, _seconds;
        private long _started;

        public void Begin(long now)
        {
            lock (_gate)
            {
                (_decibels, _seconds, _started) = (double.NegativeInfinity, 0, now);
            }
        }

        public void Set(double decibels, double seconds)
        {
            lock (_gate)
            {
                (_decibels, _seconds) = (decibels, seconds);
            }
        }

        public (double Decibels, double Seconds, long Started) Read()
        {
            lock (_gate)
            {
                return (_decibels, _seconds, _started);
            }
        }
    }

    // A recording as it comes: each channel in a pooled buffer that doubles when full (the old one returned), up to the
    // longest recording; one copy into the recording at the end; the buffers returned when disposed.
    private sealed class RecordingBuffer(int channels, int sampleRate, long maxSamples) : IDisposable
    {
        private readonly float[][] _channels = [.. Enumerable.Range(0, channels).Select(_ => ArrayPool<float>.Shared.Rent(Math.Max(1024, sampleRate)))];
        private readonly int _max = (int)Math.Min(int.MaxValue / Math.Max(1, channels), Math.Max(1, maxSamples));

        public int Length { get; private set; }

        public bool Full => Length >= _max;

        public double Seconds => (double)Length / sampleRate;

        // Appends the block (its channels as the capture's); true when the longest recording is reached.
        public bool Append(AudioData block)
        {
            int count = Math.Min(block.Length, _max - Length);
            if (count <= 0)
            {
                return true;
            }

            if (Length + count > _channels[0].Length)
            {
                int size = (int)Math.Min(_max, Math.Max((long)_channels[0].Length * 2, Length + count));
                for (int c = 0; c < channels; c++)
                {
                    var bigger = ArrayPool<float>.Shared.Rent(size);
                    _channels[c].AsSpan(0, Length).CopyTo(bigger);
                    ArrayPool<float>.Shared.Return(_channels[c]);
                    _channels[c] = bigger;
                }
            }

            for (int c = 0; c < channels; c++)
            {
                block.Channel(Math.Min(c, block.Channels - 1))[..count].CopyTo(_channels[c].AsSpan(Length));
            }

            Length += count;
            return Full;
        }

        public AudioData ToAudio()
        {
            var samples = GC.AllocateUninitializedArray<float>(channels * Length);
            for (int c = 0; c < channels; c++)
            {
                _channels[c].AsSpan(0, Length).CopyTo(samples.AsSpan(c * Length));
            }

            return new AudioData(samples, channels, sampleRate);
        }

        public void Dispose()
        {
            for (int c = 0; c < channels; c++)
            {
                if (_channels[c].Length > 0)
                {
                    ArrayPool<float>.Shared.Return(_channels[c]);
                    _channels[c] = [];
                }
            }
        }
    }
}
