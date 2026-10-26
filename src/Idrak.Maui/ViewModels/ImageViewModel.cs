// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Globalization;
using System.Windows.Input;
using Idrak.Maui.Abstractions;
using Microsoft.Maui.Media;

namespace Idrak.Maui.ViewModels;

/// <summary>A registered image transform as a chip on the Image page: tapping it adds a step of it to the pipeline.</summary>
public sealed class TransformChip
{
    internal TransformChip(IImageTransform transform, Action<TransformChip> add)
    {
        Name = transform.Name;
        Label = transform.Label;
        Summary = transform.Summary;
        Parameters = Describe(transform.Parameters);
        Step = StepOf(transform);
        AddCommand = new Command(() => add(this));
    }

    /// <summary>Its name in a pipeline ("contrast").</summary>
    public string Name { get; }

    /// <summary>Its name for people ("Contrast").</summary>
    public string Label { get; }

    /// <summary>What it does.</summary>
    public string Summary { get; }

    /// <summary>Its inputs as a line ("value: number 0 to 4; resample: lanczos, bicubic"), or "" when it takes none.</summary>
    public string Parameters { get; }

    /// <summary>Whether it takes inputs.</summary>
    public bool HasParameters => Parameters.Length > 0;

    /// <summary>The step a tap adds ("contrast=1.5"): its value from the middle of its range (the first choice), none when the value may be left out.</summary>
    public string Step { get; }

    /// <summary>Adds <see cref="Step"/> to the pipeline.</summary>
    public ICommand AddCommand { get; }

    /// <inheritdoc />
    public override string ToString() => Label;

    // Each input: its name, kind and range (or choices); "?" after one that may be left out.
    internal static string Describe(IReadOnlyList<ImageTransformParameter> parameters) => string.Join("; ", parameters.Select(p =>
    {
        string name = p.Key.Length == 0 ? "value" : p.Key;
        string what = p.Input switch
        {
            ImageTransformInput.Choice => string.Join(", ", p.Choices),
            _ => (p.Input == ImageTransformInput.Integer ? "whole number" : "number")
                 + (p.Min is { } min && p.Max is { } max ? string.Create(CultureInfo.InvariantCulture, $" {min:0.###} to {max:0.###}")
                    : p.Min is { } low ? string.Create(CultureInfo.InvariantCulture, $" from {low:0.###}") : ""),
        };
        return $"{name}{(p.Optional ? "?" : "")}: {what}";
    }));

    // The step a tap adds: the transform's name, with a value when it needs one (the first choice, or the middle of its range
    // on its increments), checked as a pipeline checks it; the name alone when that value is refused.
    internal static string StepOf(IImageTransform transform)
    {
        var value = transform.Parameters.FirstOrDefault(p => p.Key.Length == 0);
        if (value is null || value.Optional)
        {
            return transform.Name;
        }

        string? text = value.Input switch
        {
            ImageTransformInput.Choice => value.Choices.FirstOrDefault(),
            _ when value.Min is { } min && value.Max is { } max => Number(Middle(min, max, value.Step), value.Input == ImageTransformInput.Integer),
            _ when value.Min is { } min => Number(min, value.Input == ImageTransformInput.Integer),
            _ => null,
        };
        if (text is null)
        {
            return transform.Name;
        }

        string step = $"{transform.Name}={text}";
        try
        {
            _ = ImageTransformPipeline.Parse(step);
            return step;
        }
        catch (ArgumentException)
        {
            return transform.Name;
        }

        static double Middle(double min, double max, double? increment)
        {
            double middle = (min + max) / 2;
            return increment is > 0 and var s ? min + Math.Round((middle - min) / s) * s : middle;
        }

        static string Number(double v, bool whole) => whole
            ? ((long)Math.Round(v)).ToString(CultureInfo.InvariantCulture)
            : Math.Round(v, 3).ToString("0.###", CultureInfo.InvariantCulture);
    }
}

/// <summary>How long one step of the last run took.</summary>
/// <param name="Step">The step as written ("contrast=1.5").</param>
/// <param name="Time">Its time.</param>
/// <param name="Cold">Whether it was the first run of its transform in this process (paying for compiling its methods).</param>
public sealed record StepTime(string Step, TimeSpan Time, bool Cold)
{
    /// <summary>"contrast=1.5: 12.3 ms (cold)".</summary>
    public string Text => $"{Step}: {Photos.Milliseconds(Time)}{(Cold ? " (cold)" : "")}";
}

/// <summary>
/// The Image lab (plan 34, M6): a photo from the camera (its permission asked first) or the gallery, its bytes decoded once
/// by the library's own codecs and turned upright by its EXIF orientation; the registered image transforms as chips
/// (name, summary, inputs) that add a step to the pipeline's box, and the scan profiles to draw steps from; the pipeline
/// applied to the decoded photo on a worker thread, step by step with each step's time, the first run of a transform in
/// the process (cold: compiling its methods) reported apart from the warm runs (rule 76); the original and the result shown
/// as pictures shrunk for the screen and encoded once with the library's PNG encoder (never decoded again per frame); the
/// result saved and shared as PNG or JPEG (its quality a setting). Settings are kept between runs
/// (<see cref="ImageSettings"/>).
/// </summary>
public sealed class ImageViewModel : ViewModel
{
    // The transforms run in this process: a run with one not among them is cold (its methods compiled on the way).
    private static readonly HashSet<string> Ran = new(StringComparer.OrdinalIgnoreCase);
    private static readonly Lock RanGate = new();

    private readonly IPermissionService _permissions;
    private readonly IShareService _share;
    private readonly ISettingsStore _settings;
    private readonly IMediaPicker? _picker;
    private readonly TimeProvider _time;
    private readonly List<double> _warm = [];
    private ImageLabOptions _options;
    private Photo? _photo;
    private ImageData? _result;
    private string? _warmFor;
    private (ImageData Image, string Format, string Path)? _saved;
    private PhotoPreview? _original, _shown;
    private ImageSource? _originalSource, _resultSource;
    private string _pipeline;
    private string? _profile;
    private bool _busy, _refused, _showSettings;
    private string _photoText = "", _resultText = "", _coldText = "", _warmText = "", _stepsText = "", _status = "", _error = "", _settingsMessage = "", _profileText = "";

    /// <summary>The page over the app's services; <paramref name="picker"/> is MAUI's media picker when null.</summary>
    public ImageViewModel(IPermissionService permissions, IShareService share, ISettingsStore settings, ImageLabOptions? defaults = null, IMediaPicker? picker = null,
        TimeProvider? time = null)
    {
        _permissions = permissions ?? throw new ArgumentNullException(nameof(permissions));
        _share = share ?? throw new ArgumentNullException(nameof(share));
        _settings = settings ?? throw new ArgumentNullException(nameof(settings));
        _picker = picker;
        _time = time ?? TimeProvider.System;
        _options = ImageSettings.Load(settings, out var skipped, defaults);
        _pipeline = _options.Pipeline ?? "";
        _profile = _options.Profile;
        if (skipped.Count > 0)
        {
            _settingsMessage = "A saved setting was not read: " + string.Join("; ", skipped);
        }

        foreach (string name in ImageTransforms.Names)
        {
            if (ImageTransforms.Find(name) is { } transform)
            {
                Transforms.Add(new TransformChip(transform, chip => AddStep(chip.Step)));
            }
        }

        foreach (string name in ScanProfiles.Names.Order(StringComparer.OrdinalIgnoreCase))
        {
            Profiles.Add(name);
        }

        foreach (var setting in ImageSettings.Definitions)
        {
            Settings.Add(new LabSettingEntry(setting, setting.Get(_options)));
        }

        UpdateProfileText();
        TakePhotoCommand = new Command(async () => await TakePhotoAsync(camera: true));
        PickPhotoCommand = new Command(async () => await TakePhotoAsync(camera: false));
        ApplyCommand = new Command(async () => await ApplyAsync());
        DrawProfileCommand = new Command(() => DrawProfile());
        ClearPipelineCommand = new Command(() => Pipeline = "");
        SaveCommand = new Command(async () => await SaveAsync());
        ShareCommand = new Command(async () => await ShareAsync());
        OpenPermissionSettingsCommand = new Command(() => _permissions.OpenSettings());
        ApplySettingsCommand = new Command(() => ApplySettings());
        ToggleSettingsCommand = new Command(() => ShowSettings = !ShowSettings);
    }

    /// <summary>The page's title.</summary>
    public string Title => "Image";

    /// <summary>The settings in effect.</summary>
    public ImageLabOptions Options => _options;

    /// <summary>The registered image transforms, as chips.</summary>
    public ObservableCollection<TransformChip> Transforms { get; } = [];

    /// <summary>The registered scan profiles.</summary>
    public ObservableCollection<string> Profiles { get; } = [];

    /// <summary>The scan profile picked (kept in the settings), or null.</summary>
    public string? SelectedProfile
    {
        get => _profile;
        set
        {
            if (SetProperty(ref _profile, value))
            {
                _options.Profile = value;
                Settings.FirstOrDefault(e => e.Key == "profile")?.Text = value ?? "";   // the settings card shows the profile picked
                Keep();
                UpdateProfileText();
            }
        }
    }

    /// <summary>What the profile picked makes, or "".</summary>
    public string ProfileText
    {
        get => _profileText;
        private set => SetProperty(ref _profileText, value);
    }

    /// <summary>The pipeline's text (<see cref="ImageTransformPipeline"/>: <c>grayscale,max_width=1024,contrast=1.5</c>).</summary>
    public string Pipeline
    {
        get => _pipeline;
        set => SetProperty(ref _pipeline, value ?? "");
    }

    /// <summary>The photo opened, or null.</summary>
    public Photo? Photo => _photo;

    /// <summary>The last result (the photo through the pipeline), or null.</summary>
    public ImageData? Result => _result;

    /// <summary>The photo's picture on the page (shrunk for the screen, PNG made once), or null.</summary>
    public PhotoPreview? OriginalPreview => _original;

    /// <summary>The result's picture on the page, or null.</summary>
    public PhotoPreview? ResultPreview => _shown;

    /// <summary>The photo's picture for the image control, or null.</summary>
    public ImageSource? OriginalSource
    {
        get => _originalSource;
        private set
        {
            if (SetProperty(ref _originalSource, value))
            {
                OnPropertyChanged(nameof(HasPhoto));
                OnPropertyChanged(nameof(HasNoPhoto));
            }
        }
    }

    /// <summary>The result's picture for the image control, or null.</summary>
    public ImageSource? ResultSource
    {
        get => _resultSource;
        private set
        {
            if (SetProperty(ref _resultSource, value))
            {
                OnPropertyChanged(nameof(HasResult));
            }
        }
    }

    /// <summary>Whether a photo is open.</summary>
    public bool HasPhoto => OriginalSource is not null;

    /// <summary>Whether none is.</summary>
    public bool HasNoPhoto => !HasPhoto;

    /// <summary>Whether there is a result.</summary>
    public bool HasResult => ResultSource is not null;

    /// <summary>The photo: its name, format, size, orientation and read time; or "".</summary>
    public string PhotoText
    {
        get => _photoText;
        private set => SetProperty(ref _photoText, value);
    }

    /// <summary>The result: its size and the pipeline that made it; or "".</summary>
    public string ResultText
    {
        get => _resultText;
        private set => SetProperty(ref _resultText, value);
    }

    /// <summary>The first run of the pipeline's transforms in this process (cold), or "".</summary>
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

    /// <summary>The warm runs: the last and the mean, or "".</summary>
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

    /// <summary>Each step of the last run with its time, a line each, or "".</summary>
    public string StepsText
    {
        get => _stepsText;
        private set => SetProperty(ref _stepsText, value);
    }

    /// <summary>The steps of the last run with their times.</summary>
    public IReadOnlyList<StepTime> Steps { get; private set; } = [];

    /// <summary>Whether a photo is being read, run or saved.</summary>
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

    /// <summary>Whether nothing is running.</summary>
    public bool IsIdle => !IsBusy;

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

    /// <summary>The formats the library reads, for the page ("png (.png), jpeg (.jpg, .jpeg), …").</summary>
    public string Formats => $"The library's own codecs read {Photos.Readable()} (not HEIC).";

    /// <summary>Takes a photo with the camera (the permission asked first).</summary>
    public ICommand TakePhotoCommand { get; }

    /// <summary>Picks a photo from the gallery.</summary>
    public ICommand PickPhotoCommand { get; }

    /// <summary>Runs the pipeline on the photo.</summary>
    public ICommand ApplyCommand { get; }

    /// <summary>Draws the picked profile's steps into the pipeline.</summary>
    public ICommand DrawProfileCommand { get; }

    /// <summary>Empties the pipeline.</summary>
    public ICommand ClearPipelineCommand { get; }

    /// <summary>Saves the result.</summary>
    public ICommand SaveCommand { get; }

    /// <summary>Shares the result.</summary>
    public ICommand ShareCommand { get; }

    /// <summary>Opens the app's page in the system settings (a refused camera is allowed there).</summary>
    public ICommand OpenPermissionSettingsCommand { get; }

    /// <summary>Checks and keeps the settings.</summary>
    public ICommand ApplySettingsCommand { get; }

    /// <summary>Opens or closes the settings card.</summary>
    public ICommand ToggleSettingsCommand { get; }

    /// <summary>A photo from the camera or the gallery, opened (<see cref="OpenAsync"/>); false when none came or it failed.</summary>
    public async Task<bool> TakePhotoAsync(bool camera)
    {
        if (IsBusy)
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
    /// Opens <paramref name="file"/>: decoded once by the library's codecs and turned upright on a worker thread, its picture
    /// made once for the page; a format the library does not read (HEIC) is refused naming the ones it reads. The last result
    /// is dropped. True when it opened.
    /// </summary>
    public async Task<bool> OpenAsync(ChatImage file, string name)
    {
        ArgumentNullException.ThrowIfNull(file);
        ArgumentNullException.ThrowIfNull(name);
        if (IsBusy)
        {
            return false;
        }

        IsBusy = true;
        Error = "";
        Status = $"Reading {name}…";
        try
        {
            int? side = _options.PreviewPixels;
            var resampling = _options.Resampling;
            var (photo, preview) = await Task.Run(() =>
            {
                var read = Photos.Decode(file, name);
                return (read, Photos.Preview(read.Pixels, side, resampling));
            }).ConfigureAwait(true);
            _photo = photo;
            _original = preview;
            _result = null;
            _saved = null;
            _shown = null;
            ResultSource = null;
            ResultText = "";
            StepsText = "";
            Steps = [];
            OriginalSource = preview.Source();
            PhotoText = photo.Describe();
            Status = $"Opened {name}.";
            return true;
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
    }

    /// <summary>Adds <paramref name="step"/> ("contrast=1.5") at the end of the pipeline.</summary>
    public void AddStep(string step)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(step);
        string current = Pipeline.Trim().TrimEnd(',').Trim();
        Pipeline = current.Length == 0 ? step : $"{current},{step}";
    }

    /// <summary>
    /// Draws the picked scan profile's steps for one page into the pipeline (from the settings' seed, else a new one, said);
    /// false when no profile is picked or it names none registered.
    /// </summary>
    public bool DrawProfile()
    {
        Error = "";
        if (SelectedProfile is not { Length: > 0 } name)
        {
            Error = "Pick a scan profile first.";
            return false;
        }

        try
        {
            int seed = _options.Seed ?? Random.Shared.Next();
            var steps = ScanProfiles.Resolve(name).Sample(seed);
            Pipeline = steps.ToString();
            Status = steps.IsEmpty ? $"{name} took no step from seed {seed}: draw again, or set a seed in the settings."
                : $"{name} drew {steps.Steps.Count} step{(steps.Steps.Count == 1 ? "" : "s")} from seed {seed}.";
            return true;
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException)
        {
            Error = e.Message;
            return false;
        }
    }

    /// <summary>
    /// Runs the pipeline on the decoded photo on a worker thread (never decoding it again), step by step with each step's
    /// time; the run is cold when a transform of it had not run in this process, else warm; the result's picture made once.
    /// The pipeline is checked first (an unknown step names the registered ones) and kept in the settings. True when it ran.
    /// </summary>
    public async Task<bool> ApplyAsync()
    {
        if (IsBusy)
        {
            return false;
        }

        Error = "";
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

        _options.Pipeline = pipeline.IsEmpty ? null : pipeline.ToString();
        Keep();
        IsBusy = true;
        Status = pipeline.IsEmpty ? "No step: the photo as decoded." : $"Running {pipeline.Steps.Count} step{(pipeline.Steps.Count == 1 ? "" : "s")}…";
        try
        {
            int? side = _options.PreviewPixels;
            var resampling = _options.Resampling;
            var (result, steps, total, cold, preview) = await Task.Run(() =>
            {
                bool cold;
                lock (RanGate)
                {
                    cold = pipeline.Steps.Any(s => !Ran.Contains(s.Name));
                }

                var image = photo.Pixels;
                var times = new List<StepTime>(pipeline.Steps.Count);
                long start = _time.GetTimestamp();
                foreach (var step in pipeline.Steps)
                {
                    bool first;
                    lock (RanGate)
                    {
                        first = Ran.Add(step.Name);
                    }

                    long at = _time.GetTimestamp();
                    image = ImageTransforms.Get(step.Name).Apply(image, step);
                    times.Add(new StepTime(step.ToString(), _time.GetElapsedTime(at), first));
                }

                var took = _time.GetElapsedTime(start);
                return (image, times, took, cold, Photos.Preview(image, side, resampling));
            }).ConfigureAwait(true);
            _result = result;
            _saved = null;
            _shown = preview;
            ResultSource = preview.Source();
            Steps = steps;
            StepsText = string.Join("\n", steps.Select(s => s.Text));
            ResultText = string.Create(CultureInfo.InvariantCulture, $"{result.Width:N0} x {result.Height:N0}, {(result.Channels == 1 ? "grey" : "colour")}")
                + (pipeline.IsEmpty ? ", as decoded" : $", from {pipeline}");
            Report(total, cold, pipeline);
            Status = $"Done in {Photos.Milliseconds(total)}.";
            return true;
        }
        catch (Exception e) when (e is ArgumentException or InvalidDataException or NotSupportedException or InvalidOperationException)
        {
            Status = "";
            Error = $"The pipeline stopped: {e.Message}";
            return false;
        }
        finally
        {
            IsBusy = false;
        }
    }

    /// <summary>
    /// Writes the result (the photo when nothing ran) to a file of the settings' format in the cache folder's "images",
    /// encoded once at its full size; the path, or null when there is nothing to save or it failed.
    /// </summary>
    public async Task<string?> SaveAsync()
    {
        Error = "";
        var image = _result ?? _photo?.Pixels;
        if (image is null)
        {
            Error = "Nothing to save yet: take or pick a photo.";
            return null;
        }

        string format = _options.Format;
        if (_saved is { } saved && ReferenceEquals(saved.Image, image) && saved.Format == format && File.Exists(saved.Path))
        {
            Status = $"Saved already: {saved.Path}";
            return saved.Path;
        }

        try
        {
            var options = _options.Clone();
            string stem = Path.GetFileNameWithoutExtension(_photo?.Name ?? "image") is { Length: > 0 } s ? s : "image";
            string folder = Path.Combine(ComputeResources.CacheFolder, "images");
            string path = Path.Combine(folder, $"{stem}-{_time.GetLocalNow():yyyyMMdd-HHmmss}{options.Extension}");
            var (bytes, took) = await Task.Run(() =>
            {
                long start = Stopwatch.GetTimestamp();
                byte[] file = options.Encode(image);
                var encoded = Stopwatch.GetElapsedTime(start);
                Directory.CreateDirectory(folder);
                File.WriteAllBytes(path, file);
                return (file.Length, encoded);
            }).ConfigureAwait(true);
            _saved = (image, format, path);
            Status = $"Saved {Path.GetFileName(path)}: {format}, {ProgressText.Bytes(bytes)}, encoded in {Photos.Milliseconds(took)}.";
            return path;
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or ArgumentException or NotSupportedException)
        {
            Error = $"The image was not saved: {e.Message}";
            return null;
        }
    }

    /// <summary>Saves the result (once) and offers it to the phone's share sheet; false when there is nothing to share or it failed.</summary>
    public async Task<bool> ShareAsync()
    {
        if (await SaveAsync().ConfigureAwait(true) is not { } path)
        {
            return false;
        }

        try
        {
            await _share.ShareFileAsync(path, "Image", _options.MediaType).ConfigureAwait(true);
            return true;
        }
        catch (Exception e) when (e is not OutOfMemoryException)
        {
            Error = $"The image was not shared: {e.Message}";
            return false;
        }
    }

    /// <summary>Checks every setting on the page and keeps them; false, and nothing kept, when one is refused (each says why).</summary>
    public bool ApplySettings()
    {
        if (LabSettingEntry.Apply(Settings, _options.Clone()) is not { } next)
        {
            SettingsMessage = "Nothing was kept: a value above is not one the setting takes.";
            return false;
        }

        next.Pipeline = _options.Pipeline;
        _options = next;
        _profile = next.Profile;
        OnPropertyChanged(nameof(SelectedProfile));
        UpdateProfileText();
        Keep();
        SettingsMessage = "Kept.";
        return true;
    }

    private void Keep() => ImageSettings.Save(_settings, _options);

    private void UpdateProfileText()
    {
        try
        {
            ProfileText = _profile is { Length: > 0 } name ? ScanProfiles.Resolve(name).Summary : "";
        }
        catch (Exception e) when (e is NotSupportedException or ArgumentException)
        {
            ProfileText = e.Message;
        }
    }

    // The run's time onto the cold or the warm line.
    private void Report(TimeSpan total, bool cold, ImageTransformPipeline pipeline)
    {
        string what = pipeline.IsEmpty ? "no step" : string.Join(", ", pipeline.Steps.Select(s => s.Name).Distinct(StringComparer.OrdinalIgnoreCase));
        if (cold)
        {
            ColdText = $"Cold (the first run of {what} in this app: compiling its methods): {Photos.Milliseconds(total)}";
        }
        else
        {
            if (_warmFor != pipeline.ToString())
            {
                _warm.Clear();                                                  // another pipeline: its own mean
                _warmFor = pipeline.ToString();
            }

            _warm.Add(total.TotalMilliseconds);
            WarmText = string.Create(CultureInfo.InvariantCulture, $"Warm: {Photos.Milliseconds(total)} last, {_warm.Average():F1} ms mean over {_warm.Count} run{(_warm.Count == 1 ? "" : "s")} of this pipeline");
        }
    }
}
