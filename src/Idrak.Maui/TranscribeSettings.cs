// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>How the Transcribe lab reads a file: through the live path in blocks (as <c>idrak transcribe --live FILE</c>), or whole.</summary>
public enum TranscribeFileMode
{
    /// <summary>Streamed in blocks through the same live transcription as the microphone (<c>LiveTranscriber</c>).</summary>
    Live,

    /// <summary>Transcribed whole, as one recording (<see cref="ISpeechRecognizer.Transcribe"/>).</summary>
    Whole,
}

/// <summary>
/// The Transcribe lab's settings: the speech model, the language and task, every <see cref="LiveTranscriptionOptions"/>
/// value, the cleaning pipeline, the capture and the file mode. A null value is the library's own default
/// (<see cref="LiveTranscriptionOptions"/>, <see cref="TranscriptionOptions"/>, <see cref="AudioCaptureOptions"/>), so the
/// defaults are read from the library, never repeated here. Kept between runs by <see cref="TranscribeSettings"/>.
/// </summary>
public sealed class TranscribeLabOptions
{
    /// <summary>The speech model chosen, as the model store names it ("owner/name@revision"); null: none yet.</summary>
    public string? Model { get; set; }

    /// <summary>The language's code ("en", "ar"); null: detected by the family when it can.</summary>
    public string? Language { get; set; }

    /// <summary>"transcribe" or "translate" (<see cref="TranscriptionOptions.Task"/>); null: the library's ("transcribe").</summary>
    public string? Task { get; set; }

    /// <summary>Beams kept while decoding (<see cref="TranscriptionOptions.Beams"/>); null: the library's (greedy).</summary>
    public int? Beams { get; set; }

    /// <summary>Text given to the model before the first segment (names, spellings); null: none.</summary>
    public string? Prompt { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.StepSeconds"/>; null: the library's.</summary>
    public double? StepSeconds { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.WindowSeconds"/>; null: the library's.</summary>
    public double? WindowSeconds { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.SilenceSeconds"/>; null: the library's.</summary>
    public double? SilenceSeconds { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.SilenceDecibels"/>; null: the library's.</summary>
    public double? SilenceDecibels { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.SilenceFloorDecibels"/>; null: the library's.</summary>
    public double? SilenceFloorDecibels { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.MinSpeechSeconds"/>; null: the library's.</summary>
    public double? MinSpeechSeconds { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.PreRollSeconds"/>; null: the library's.</summary>
    public double? PreRollSeconds { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.Policy"/> (a <see cref="LiveCommitPolicies"/> name); null: the library's.</summary>
    public string? Policy { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.Agreement"/>; null: the library's.</summary>
    public int? Agreement { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.PromptCharacters"/>; null: the library's.</summary>
    public int? PromptCharacters { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.KeepUp"/> for the microphone; null: the library's (on).</summary>
    public bool? KeepUp { get; set; }

    /// <summary>The cleaning pipeline run on the sound before it is transcribed (<see cref="AudioPipeline"/> text, live steps only); null: none.</summary>
    public string? Pipeline { get; set; }

    /// <summary>The capture (an <see cref="AudioCaptures"/> name); null: <see cref="AudioCaptures.ForThisMachine"/> (aaudio on Android).</summary>
    public string? Capture { get; set; }

    /// <summary>The input device (an id the capture lists); null: the system's default.</summary>
    public string? InputDevice { get; set; }

    /// <summary>The rate the microphone is asked for (<see cref="AudioCaptureOptions.SampleRate"/>); null: the library's.</summary>
    public int? SampleRate { get; set; }

    /// <summary>The seconds of sound a captured block holds (<see cref="AudioCaptureOptions.BlockSeconds"/>); null: the library's.</summary>
    public double? BlockSeconds { get; set; }

    /// <summary>How a file is read: through the live path (the default) or whole.</summary>
    public TranscribeFileMode FileMode { get; set; } = TranscribeFileMode.Live;

    /// <summary>The seconds of a file given at a time on the live path; null: <see cref="TranscribeSettings.DefaultFileBlockSeconds"/> (as the CLI's <c>--live FILE</c>).</summary>
    public double? FileBlockSeconds { get; set; }

    /// <summary><see cref="LiveTranscriptionOptions.KeepUp"/> for a file on the live path; null: off (a step every step of sound, as the CLI's <c>--live FILE</c>).</summary>
    public bool? FileKeepUp { get; set; }

    /// <summary>A copy.</summary>
    public TranscribeLabOptions Clone() => (TranscribeLabOptions)MemberwiseClone();

    /// <summary>How each step (or the whole file) is transcribed: the language, task, beams and prompt over the library's defaults.</summary>
    public TranscriptionOptions ToTranscription()
    {
        var library = new TranscriptionOptions();
        return library with
        {
            Language = string.IsNullOrWhiteSpace(Language) ? null : Language,
            Task = string.IsNullOrWhiteSpace(Task) ? library.Task : Task,
            Beams = Beams ?? library.Beams,
            Prompt = string.IsNullOrEmpty(Prompt) ? library.Prompt : Prompt,
            Timestamps = false,                                                 // the live segments carry the live times
            ChunkSeconds = 0,
        };
    }

    /// <summary>
    /// The library's live options with these values over its defaults; <paramref name="microphone"/> picks
    /// <see cref="KeepUp"/> (live sound) or <see cref="FileKeepUp"/> (a file on the live path).
    /// </summary>
    public LiveTranscriptionOptions ToLive(bool microphone = true)
    {
        var library = new LiveTranscriptionOptions();
        return library with
        {
            Transcription = ToTranscription(),
            StepSeconds = StepSeconds ?? library.StepSeconds,
            WindowSeconds = WindowSeconds ?? library.WindowSeconds,
            SilenceSeconds = SilenceSeconds ?? library.SilenceSeconds,
            SilenceDecibels = SilenceDecibels ?? library.SilenceDecibels,
            SilenceFloorDecibels = SilenceFloorDecibels ?? library.SilenceFloorDecibels,
            MinSpeechSeconds = MinSpeechSeconds ?? library.MinSpeechSeconds,
            PreRollSeconds = PreRollSeconds ?? library.PreRollSeconds,
            Policy = string.IsNullOrWhiteSpace(Policy) ? library.Policy : Policy,
            Agreement = Agreement ?? library.Agreement,
            PromptCharacters = PromptCharacters ?? library.PromptCharacters,
            KeepUp = microphone ? KeepUp ?? library.KeepUp : FileKeepUp ?? false,
        };
    }

    /// <summary>How the microphone is opened: the device, rate and block over the library's defaults, one channel.</summary>
    public AudioCaptureOptions ToCapture()
    {
        var library = new AudioCaptureOptions();
        return library with
        {
            Device = string.IsNullOrWhiteSpace(InputDevice) ? null : InputDevice,
            SampleRate = SampleRate ?? library.SampleRate,
            BlockSeconds = BlockSeconds ?? library.BlockSeconds,
            Channels = 1,
        };
    }

    /// <summary>The cleaning pipeline (empty when none), checked against the registered processors and refused when a step needs the whole recording.</summary>
    /// <exception cref="ArgumentException">An unknown processor or option, or a step that is not live (the message names it and its live alternative).</exception>
    public AudioPipeline ToPipeline()
    {
        var pipeline = AudioPipeline.Parse(Pipeline);
        pipeline.ThrowIfNotLive();
        return pipeline;
    }
}

/// <summary>How a Transcribe lab setting is edited.</summary>
public enum TranscribeSettingKind
{
    /// <summary>Free text.</summary>
    Text,

    /// <summary>A number; empty means the library's own value.</summary>
    Number,

    /// <summary>One of <see cref="TranscribeSetting.Choices"/>.</summary>
    Choice,
}

/// <summary>One value of <see cref="TranscribeLabOptions"/> as a setting: its key, title, help, the library's default shown in an empty field.</summary>
public sealed class TranscribeSetting
{
    private readonly Func<TranscribeLabOptions, string> _read;
    private readonly Action<TranscribeLabOptions, string> _write;
    private readonly Func<IReadOnlyList<string>> _choices;

    internal TranscribeSetting(string key, string title, string summary, TranscribeSettingKind kind, string placeholder, Func<TranscribeLabOptions, string> read,
        Action<TranscribeLabOptions, string> write, Func<IReadOnlyList<string>>? choices = null)
    {
        Key = key;
        Title = title;
        Summary = summary;
        Kind = kind;
        Placeholder = placeholder;
        _read = read;
        _write = write;
        _choices = choices ?? (() => []);
    }

    /// <summary>The key in the settings store, after <see cref="TranscribeSettings.Prefix"/> ("step_s").</summary>
    public string Key { get; }

    /// <summary>The title on the page.</summary>
    public string Title { get; }

    /// <summary>What it does, with the library's option it maps to.</summary>
    public string Summary { get; }

    /// <summary>How it is edited.</summary>
    public TranscribeSettingKind Kind { get; }

    /// <summary>What an empty value means (the library's default), shown in an empty field.</summary>
    public string Placeholder { get; }

    /// <summary>The values a <see cref="TranscribeSettingKind.Choice"/> takes, the first standing for "" (the library's default); read when asked (the registries may grow).</summary>
    public IReadOnlyList<string> Choices => _choices();

    /// <summary>The value in <paramref name="options"/> as text ("" for the library's own).</summary>
    public string Get(TranscribeLabOptions options) => _read(options ?? throw new ArgumentNullException(nameof(options)));

    /// <summary>Sets the value in <paramref name="options"/> from text ("" or null for the library's own).</summary>
    /// <exception cref="FormatException">The text is not a value of the setting (the message says what is).</exception>
    public void Set(TranscribeLabOptions options, string? text) => _write(options ?? throw new ArgumentNullException(nameof(options)), text?.Trim() ?? "");
}

/// <summary>
/// The Transcribe lab's settings kept between runs, each under <see cref="Prefix"/> and its key in the app's
/// <see cref="ISettingsStore"/> (MAUI's Preferences): the model, language and task the page chose, and every
/// <see cref="Definitions"/> value.
/// </summary>
public static class TranscribeSettings
{
    /// <summary>What every key starts with.</summary>
    public const string Prefix = IdrakMauiSettings.Prefix + "transcribe.";

    /// <summary>The seconds of a file given at a time on the live path when none is set (the CLI's <c>--live FILE</c>).</summary>
    public const double DefaultFileBlockSeconds = 0.1;

    private const string ModelKey = Prefix + "model", LanguageKey = Prefix + "language", TaskKey = Prefix + "task";

    /// <summary>The settings edited on the page, in its order.</summary>
    public static IReadOnlyList<TranscribeSetting> Definitions { get; } = Define();

    /// <summary>The options saved in <paramref name="store"/> (the library's defaults where nothing is); a saved value that no longer reads is skipped and named.</summary>
    public static TranscribeLabOptions Load(ISettingsStore store, out IReadOnlyList<string> skipped)
    {
        ArgumentNullException.ThrowIfNull(store);
        var options = new TranscribeLabOptions
        {
            Model = Empty(store.Get(ModelKey)),
            Language = Empty(store.Get(LanguageKey)),
            Task = Empty(store.Get(TaskKey)),
        };
        var refused = new List<string>();
        foreach (var setting in Definitions)
        {
            if (store.Get(Prefix + setting.Key) is not { } text)
            {
                continue;
            }

            try
            {
                setting.Set(options, text);
            }
            catch (FormatException e)
            {
                refused.Add($"{setting.Title}: {e.Message}");
            }
        }

        skipped = refused;
        return options;
    }

    /// <summary>Keeps every value of <paramref name="options"/> in <paramref name="store"/> (an empty one removed: the library's own).</summary>
    public static void Save(ISettingsStore store, TranscribeLabOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);
        store.Set(ModelKey, Empty(options.Model));
        store.Set(LanguageKey, Empty(options.Language));
        store.Set(TaskKey, Empty(options.Task));
        foreach (var setting in Definitions)
        {
            store.Set(Prefix + setting.Key, Empty(setting.Get(options)));
        }
    }

    private static string? Empty(string? text) => string.IsNullOrWhiteSpace(text) ? null : text.Trim();

    private static IReadOnlyList<TranscribeSetting> Define()
    {
        var live = new LiveTranscriptionOptions();
        var transcription = new TranscriptionOptions();
        var capture = new AudioCaptureOptions();
        return
        [
            new("step_s", "Step (s)", "Seconds of new sound before the segment is read again: how soon a word shows, at one transcription each (LiveTranscriptionOptions.StepSeconds).",
                TranscribeSettingKind.Number, Show(live.StepSeconds), o => Show(o.StepSeconds), (o, t) => o.StepSeconds = Real(t, above: 0)),
            new("window_s", "Window (s)", "The longest segment: one this long is ended even while the speaker goes on (WindowSeconds).",
                TranscribeSettingKind.Number, Show(live.WindowSeconds), o => Show(o.WindowSeconds), (o, t) => o.WindowSeconds = Real(t, above: 0)),
            new("silence_s", "Pause that ends a segment (s)", "Seconds of quiet after speech that end a segment (SilenceSeconds).",
                TranscribeSettingKind.Number, Show(live.SilenceSeconds), o => Show(o.SilenceSeconds), (o, t) => o.SilenceSeconds = Real(t, above: 0)),
            new("silence_db", "Quiet under the loudest (dB)", "A 20 ms frame is quiet this many decibels under the loudest heard (SilenceDecibels).",
                TranscribeSettingKind.Number, Show(live.SilenceDecibels), o => Show(o.SilenceDecibels), (o, t) => o.SilenceDecibels = Real(t, above: 0)),
            new("silence_floor_db", "Quiet floor (dBFS)", "A frame under this level is quiet whatever was heard before (SilenceFloorDecibels).",
                TranscribeSettingKind.Number, Show(live.SilenceFloorDecibels), o => Show(o.SilenceFloorDecibels), (o, t) => o.SilenceFloorDecibels = Real(t, below: 0)),
            new("min_speech_s", "Least speech (s)", "Seconds of loud frames a segment needs before it is read: a click or a breath is left out (MinSpeechSeconds).",
                TranscribeSettingKind.Number, Show(live.MinSpeechSeconds), o => Show(o.MinSpeechSeconds), (o, t) => o.MinSpeechSeconds = Real(t, atLeast: 0)),
            new("pre_roll_s", "Pre-roll (s)", "Seconds of the quiet before speech kept with it (PreRollSeconds).",
                TranscribeSettingKind.Number, Show(live.PreRollSeconds), o => Show(o.PreRollSeconds), (o, t) => o.PreRollSeconds = Real(t, atLeast: 0)),
            new("policy", "Commit policy", "Which words are final before a segment ends (LiveCommitPolicies).", TranscribeSettingKind.Choice, $"default ({live.Policy})",
                o => o.Policy ?? "", (o, t) => o.Policy = Policy(t), () => ["default", .. LiveCommitPolicies.Names.Order(StringComparer.OrdinalIgnoreCase)]),
            new("agreement", "Agreement", "How many readings in a row a word must keep to be final (Agreement).",
                TranscribeSettingKind.Number, Show(live.Agreement), o => Show(o.Agreement), (o, t) => o.Agreement = Whole(t, 1)),
            new("prompt_chars", "Carried text (characters)", "The characters of the finished text given to the next segment as its prompt; 0: none (PromptCharacters).",
                TranscribeSettingKind.Number, Show(live.PromptCharacters), o => Show(o.PromptCharacters), (o, t) => o.PromptCharacters = Whole(t, 0)),
            new("keep_up", "Keep up", "Read everything waiting before a step, so a slow step is followed by one over all that arrived (KeepUp).",
                TranscribeSettingKind.Choice, "default", o => Show(o.KeepUp), (o, t) => o.KeepUp = Switch(t), () => ["default", "on", "off"]),
            new("beams", "Beams", "Beams kept while decoding; 1 is greedy (TranscriptionOptions.Beams).",
                TranscribeSettingKind.Number, Show(transcription.Beams), o => Show(o.Beams), (o, t) => o.Beams = Whole(t, 1)),
            new("prompt", "Prompt", "Text given to the model before the first segment: names, spellings, style (TranscriptionOptions.Prompt).",
                TranscribeSettingKind.Text, "none", o => o.Prompt ?? "", (o, t) => o.Prompt = t.Length == 0 ? null : t),
            new("pipeline", "Cleaning pipeline", "Audio processors run on the sound as it comes, before it is read: live steps only, such as highpass=80, dehum=50, agc (AudioPipeline).",
                TranscribeSettingKind.Text, "none", o => o.Pipeline ?? "", (o, t) => o.Pipeline = Pipeline(t)),
            new("capture", "Capture", "The system's sound input (AudioCaptures); default: the first that can run here (aaudio on Android).", TranscribeSettingKind.Choice, "default",
                o => o.Capture ?? "", (o, t) => o.Capture = Capture(t), () => ["default", .. AudioCaptures.Names]),
            new("input_device", "Input device", "The input device's id as the capture lists it; empty: the system's default.",
                TranscribeSettingKind.Text, "the system's default", o => o.InputDevice ?? "", (o, t) => o.InputDevice = t.Length == 0 ? null : t),
            new("sample_rate", "Sample rate (Hz)", "The rate the microphone is asked for (AudioCaptureOptions.SampleRate).",
                TranscribeSettingKind.Number, Show(capture.SampleRate), o => Show(o.SampleRate), (o, t) => o.SampleRate = Whole(t, 1000, 384_000)),
            new("block_s", "Capture block (s)", "The seconds of sound each captured block holds: the delay a block adds (AudioCaptureOptions.BlockSeconds).",
                TranscribeSettingKind.Number, Show(capture.BlockSeconds), o => Show(o.BlockSeconds), (o, t) => o.BlockSeconds = Real(t, above: 0)),
            new("file_mode", "A file is read", "live: in blocks through the same live path as the microphone (as idrak transcribe --live FILE); whole: as one recording.",
                TranscribeSettingKind.Choice, "live", o => o.FileMode == TranscribeFileMode.Live ? "" : "whole", (o, t) => o.FileMode = FileMode(t), () => ["live", "whole"]),
            new("file_block_s", "File block (s)", "The seconds of a file given at a time on the live path.",
                TranscribeSettingKind.Number, Show(DefaultFileBlockSeconds), o => Show(o.FileBlockSeconds), (o, t) => o.FileBlockSeconds = Real(t, above: 0)),
            new("file_keep_up", "File keeps up", "On the live path, read all of a file that is waiting before a step (off: a step every step of sound).",
                TranscribeSettingKind.Choice, "default", o => Show(o.FileKeepUp), (o, t) => o.FileKeepUp = Switch(t), () => ["default", "on", "off"]),
        ];
    }

    private static string Show(double? value) => value?.ToString("R", CultureInfo.InvariantCulture) ?? "";

    private static string Show(int? value) => value?.ToString(CultureInfo.InvariantCulture) ?? "";

    private static string Show(bool? value) => value switch { true => "on", false => "off", null => "" };

    // A decimal number, a comma read as the point (a phone's keyboard in many languages gives one).
    private static double? Real(string text, double? above = null, double? atLeast = null, double? below = null)
    {
        if (text.Length == 0)
        {
            return null;
        }

        if (!double.TryParse(text.Replace(',', '.'), NumberStyles.Float, CultureInfo.InvariantCulture, out double value) || !double.IsFinite(value)
            || value <= above || value < atLeast || value >= below)
        {
            string range = above is { } a ? $"above {Show(a)}" : atLeast is { } l ? $"of {Show(l)} or more" : below is { } b ? $"under {Show(b)}" : "";
            throw new FormatException($"'{text}' is not a number {range}.");
        }

        return value;
    }

    private static int? Whole(string text, int minimum, int maximum = int.MaxValue)
    {
        if (text.Length == 0)
        {
            return null;
        }

        return int.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out int value) && value >= minimum && value <= maximum ? value
            : throw new FormatException(maximum == int.MaxValue ? $"'{text}' is not a whole number of {minimum} or more." : $"'{text}' is not a whole number from {minimum} to {maximum}.");
    }

    private static bool? Switch(string text) => text.ToLowerInvariant() switch
    {
        "" or "default" => null,
        "on" or "true" or "1" => true,
        "off" or "false" or "0" => false,
        _ => throw new FormatException($"'{text}' is not on, off or default."),
    };

    private static string? Policy(string text) =>
        text.Length == 0 || text.Equals("default", StringComparison.OrdinalIgnoreCase) ? null
        : LiveCommitPolicies.Find(text) is not null ? text
        : throw new FormatException($"'{text}' is not a commit policy (registered: {string.Join(", ", LiveCommitPolicies.Names)}).");

    private static string? Capture(string text) =>
        text.Length == 0 || text.Equals("default", StringComparison.OrdinalIgnoreCase) ? null
        : AudioCaptures.Find(text) is not null ? text
        : throw new FormatException($"'{text}' is not an audio capture (registered: {string.Join(", ", AudioCaptures.Names)}).");

    private static TranscribeFileMode FileMode(string text) =>
        text.Length == 0 || text.Equals("live", StringComparison.OrdinalIgnoreCase) ? TranscribeFileMode.Live
        : text.Equals("whole", StringComparison.OrdinalIgnoreCase) ? TranscribeFileMode.Whole
        : throw new FormatException($"'{text}' is not live or whole.");

    // The pipeline text, checked as the page's live path takes it: registered processors, live steps only.
    private static string? Pipeline(string text)
    {
        if (text.Length == 0 || text.Equals("none", StringComparison.OrdinalIgnoreCase))
        {
            return null;
        }

        try
        {
            AudioPipeline.Parse(text).ThrowIfNotLive();
            return text;
        }
        catch (Exception e) when (e is ArgumentException or NotSupportedException)
        {
            throw new FormatException(e.Message);
        }
    }
}
