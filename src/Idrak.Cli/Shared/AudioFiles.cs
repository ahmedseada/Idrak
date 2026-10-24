// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using Idrak.Audio;
using Idrak.Audio.Abstractions;

namespace Idrak.Cli.Shared;

/// <summary>
/// Sound files for every command that reads or writes them (<c>transcribe</c>, <c>speak</c>, <c>voice add</c>,
/// <c>data audio</c>, <c>data mel</c>, <c>--audio</c>): the inputs expanded (files, folders, globs, <c>-</c> for standard
/// input), decoded by the registered codecs (<see cref="AudioCodecs"/>: WAV and FLAC built in, MP3 through a plug-in),
/// written by the encoder of the output's extension (<see cref="AudioEncoders"/>), with the <c>-P</c> remedy when a
/// codec or encoder is missing; and the voice store the voice commands and <c>speak</c> use.
/// </summary>
internal static class AudioFiles
{
    /// <summary>The variable whose pipeline <c>idrak transcribe</c> cleans its sound with when <c>--audio-clean</c> is not given.</summary>
    public const string AudioCleanVariable = "IDRAK_AUDIO_CLEAN";

    /// <summary>What <c>--clean</c> asks for: the cleaning stages with their settings, and the pipeline run after them.</summary>
    /// <param name="Stages">The <see cref="AudioCleaners"/> stages, in order.</param>
    /// <param name="Settings">Their settings (<c>--clean-settings</c>).</param>
    /// <param name="Processing">The audio pipeline run after the stages.</param>
    public sealed record CleaningChoice(IReadOnlyList<string> Stages, AudioCleaningOptions Settings, AudioPipeline Processing);

    /// <summary>
    /// --clean VALUE over <paramref name="stages"/>, <paramref name="settings"/> and <paramref name="processing"/> (a
    /// config's, a checkpoint's). A comma list of bare cleaning stage names (<see cref="AudioCleaners"/>: highpass,
    /// denoise, trim, normalize, or a plug-in's) is those stages, as before, with --clean-settings; any other value is an
    /// audio pipeline (<see cref="Pipeline"/>: a value or an option written with '=', or another processor's name), run in
    /// place of the stages; "none" is neither. Without --clean, the given ones (with --clean-settings over the settings).
    /// </summary>
    /// <exception cref="UsageException">The pipeline names an unknown processor or option, or --clean-settings comes with a pipeline.</exception>
    public static CleaningChoice Cleaning(CommandContext context, IReadOnlyList<string> stages, AudioCleaningOptions settings, AudioPipeline processing)
    {
        if (context.Option("--clean") is not { } clean)
        {
            return new CleaningChoice(stages, CleaningSettings(context, settings), processing);
        }

        if (IsStageList(clean))
        {
            return new CleaningChoice([.. clean.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)], CleaningSettings(context, settings), AudioPipeline.Empty);
        }

        if (context.Option("--clean-settings") is not null)
        {
            throw new UsageException("--clean-settings sets the cleaning stages' settings (--clean highpass,denoise,trim,normalize); "
                + $"a pipeline carries its own values and options (--clean \"{clean}\"): give them there, such as \"normalize=-20 peak=0.9\".");
        }

        return new CleaningChoice([], settings, Pipeline("--clean", clean));
    }

    /// <summary>
    /// Whether <paramref name="text"/> is a comma list of bare cleaning stage names (each registered in
    /// <see cref="AudioCleaners"/>, with no '=' and no space): the stages --clean took before pipelines.
    /// </summary>
    public static bool IsStageList(string text)
    {
        var items = text.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
        return items.Length > 0 && items.All(item => item.IndexOfAny(['=', ' ', '\t']) < 0 && AudioCleaners.Find(item) is not null);
    }

    /// <summary>
    /// An audio pipeline given to <paramref name="option"/>: its text (<c>highpass=80, denoise=1.5 quantile=0.2,
    /// normalize=-20</c>; <see cref="AudioPipeline.Parse"/>), a JSON array of steps (<c>["highpass=80", {"name":
    /// "normalize", "value": -20}]</c>), or @FILE holding either; "none" is the empty pipeline.
    /// </summary>
    /// <exception cref="UsageException">An unknown processor or option (naming the registered ones), a bad value, a missing file or bad JSON.</exception>
    public static AudioPipeline Pipeline(string option, string text)
    {
        string given = text;
        try
        {
            if (text.StartsWith('@'))
            {
                string path = text[1..];
                text = File.Exists(path) ? File.ReadAllText(path) : throw new UsageException($"{option}: no pipeline file {path}.");
            }

            string trimmed = text.Trim();
            return trimmed.StartsWith('[') || trimmed.StartsWith('"') || trimmed.StartsWith('{')
                ? AudioPipeline.FromJson(System.Text.Json.Nodes.JsonNode.Parse(trimmed))
                : AudioPipeline.Parse(trimmed);
        }
        catch (System.Text.Json.JsonException e)
        {
            throw new UsageException($"{option}: {given} is not a pipeline's JSON: {e.Message}");
        }
        catch (ArgumentException e)
        {
            throw new UsageException($"{option}: {e.Message} ('idrak audio processors' lists them with their options.)");
        }
    }

    /// <summary>
    /// --audio-clean PIPELINE (<see cref="Pipeline"/>), else, when <paramref name="variable"/> is named and set, its
    /// pipeline; else the empty pipeline. "none" on the command line turns the variable's off.
    /// </summary>
    public static (AudioPipeline Pipeline, string Source) AudioClean(CommandContext context, string? variable = null)
    {
        if (context.Option("--audio-clean") is { } given)
        {
            return (Pipeline("--audio-clean", given), "--audio-clean");
        }

        return variable is not null && Environment.GetEnvironmentVariable(variable) is { Length: > 0 } set
            ? (Pipeline(variable, set), variable)
            : (AudioPipeline.Empty, "--audio-clean");
    }

    /// <summary>--clean-settings F: the stages' settings from a JSON file, or inline JSON, else <paramref name="otherwise"/>.</summary>
    /// <exception cref="UsageException">Neither a file nor a JSON object, or a value out of range.</exception>
    public static AudioCleaningOptions CleaningSettings(CommandContext context, AudioCleaningOptions otherwise)
    {
        if (context.Option("--clean-settings") is not { } settings)
        {
            return otherwise;
        }

        string text = File.Exists(settings) ? File.ReadAllText(settings) : settings;
        AudioCleaningOptions options;
        try
        {
            options = AudioCleaningOptions.FromJson(System.Text.Json.Nodes.JsonNode.Parse(text) as System.Text.Json.Nodes.JsonObject
                ?? throw new UsageException($"--clean-settings: {settings} is not a JSON object."));
            options.Check();
        }
        catch (System.Text.Json.JsonException e)
        {
            throw new UsageException($"--clean-settings: {settings} is neither a file nor JSON: {e.Message}");
        }
        catch (ArgumentException e)
        {
            throw new UsageException($"--clean-settings: {e.Message}");
        }

        return options;
    }

    /// <summary>The argument that stands for standard input.</summary>
    public const string StandardInputName = "-";

    // Sample plug-ins of this repository for formats the library does not decode or encode itself, named in the remedy
    // (none today: WAV, FLAC and MP3 are the library's).
    private static readonly Dictionary<string, string> SamplePlugins = new(StringComparer.OrdinalIgnoreCase);

    /// <summary>
    /// The files <paramref name="inputs"/> name, in the order given: a file as it is, a folder's files a registered codec
    /// reads by extension (all subfolders, sorted), a glob's matches (sorted), and <c>-</c> as itself when
    /// <paramref name="standardInput"/> allows it. A usage error names an input that is not there.
    /// </summary>
    public static IReadOnlyList<string> Expand(IEnumerable<string> inputs, bool standardInput = false)
    {
        var files = new List<string>();
        var seen = new HashSet<string>(StringComparer.Ordinal);
        foreach (string input in inputs)
        {
            IEnumerable<string> found;
            if (input == StandardInputName)
            {
                found = standardInput ? [input] : throw new UsageException("Standard input (-) is not read here; give the files.");
            }
            else if (Directory.Exists(input))
            {
                found = Directory.EnumerateFiles(input, "*", SearchOption.AllDirectories).Where(AudioCodecs.CanDecode).Order(StringComparer.Ordinal);
            }
            else if (input.IndexOfAny(['*', '?']) >= 0)
            {
                string folder = Path.GetDirectoryName(input) is { Length: > 0 } d ? d : ".";
                found = Directory.Exists(folder) ? Directory.EnumerateFiles(folder, Path.GetFileName(input)).Order(StringComparer.Ordinal) : [];
            }
            else
            {
                found = File.Exists(input) ? [input] : throw new UsageException($"Sound file not found: {input}");
            }

            files.AddRange(found.Where(seen.Add));
        }

        return files;
    }

    /// <summary>
    /// The sound of <paramref name="path"/> (or of standard input for <c>-</c>), decoded by the codec that reads it; a
    /// file no registered codec reads is refused with the <c>-P</c> remedy.
    /// </summary>
    public static AudioData Decode(string path)
    {
        try
        {
            if (path == StandardInputName)
            {
                using var input = Console.OpenStandardInput();
                using var bytes = new MemoryStream();
                input.CopyTo(bytes);
                return AudioCodecs.Decode(bytes.GetBuffer().AsSpan(0, (int)bytes.Length));
            }

            return AudioCodecs.Decode(path);
        }
        catch (InvalidDataException e) when (e.Message.Contains("AudioCodecs.Register", StringComparison.Ordinal))
        {
            throw new InvalidDataException(Remedy(e.Message, Path.GetExtension(path)), e);
        }
    }

    /// <summary>
    /// The encoder that writes <paramref name="path"/> (by its extension), checked before any work is done; none is
    /// refused with the <c>-P</c> remedy (naming this repository's plug-in for the extension when it has one).
    /// </summary>
    public static IAudioEncoder EncoderFor(string path)
    {
        try
        {
            return AudioEncoders.For(path);
        }
        catch (NotSupportedException e)
        {
            throw new NotSupportedException(Remedy(e.Message, Path.GetExtension(path)), e);
        }
    }

    /// <summary>Writes <paramref name="audio"/> to <paramref name="path"/> with <paramref name="encoder"/> (the folder is created).</summary>
    public static void Save(string path, AudioData audio, IAudioEncoder encoder, AudioEncodeOptions options)
    {
        if (Path.GetDirectoryName(Path.GetFullPath(path)) is { } folder)
        {
            Directory.CreateDirectory(folder);
        }

        using var stream = File.Create(path);
        encoder.Encode(audio, stream, options);
    }

    /// <summary>The sample format of <c>--bits</c>: 8, 16, 24 or 32-bit integers, or float (32-bit); 16 when not given.</summary>
    public static AudioEncodeOptions Bits(string? text) => text?.ToLowerInvariant() switch
    {
        null or "16" => new AudioEncodeOptions { BitsPerSample = 16 },
        "8" => new AudioEncodeOptions { BitsPerSample = 8 },
        "24" => new AudioEncodeOptions { BitsPerSample = 24 },
        "32" => new AudioEncodeOptions { BitsPerSample = 32 },
        "float" or "f32" or "float32" => new AudioEncodeOptions { BitsPerSample = 32, Float = true },
        _ => throw new UsageException($"--bits takes 8, 16, 24, 32 or float, not '{text}'."),
    };

    /// <summary>The sample format as written in --bits.</summary>
    public static string BitsName(AudioEncodeOptions options) => options.Float ? "float" : options.BitsPerSample.ToString(CultureInfo.InvariantCulture);

    /// <summary>
    /// The voice store of the voice commands and <c>speak</c>: <c>--voices DIR</c>; else, when <c>IDRAK_VOICES</c> is
    /// not set and the cache folder was given (<c>--cache</c> or the config), voices/ under it; else the registered
    /// "folder" store (<see cref="VoiceStores"/>: <c>IDRAK_VOICES</c>, else voices/ under <c>IDRAK_CACHE</c> or
    /// <c>~/.cache/idrak</c>), or the store a plug-in registered over it.
    /// </summary>
    public static IVoiceStore VoiceStore(CommandContext context)
    {
        if (context.Option("--voices") is { } folder)
        {
            return new FolderVoiceStore(folder);
        }

        bool cacheGiven = context.Option("--cache") is not null || context.Config.Get("cache") is not null;
        return cacheGiven && Environment.GetEnvironmentVariable("IDRAK_VOICES") is not { Length: > 0 }
            ? new FolderVoiceStore(Path.Combine(context.CacheFolder, "voices"))
            : VoiceStores.Get();
    }

    /// <summary>A number of seconds for people: "1.50 s".</summary>
    public static string Seconds(double seconds) => string.Create(CultureInfo.InvariantCulture, $"{seconds:0.00} s");

    /// <summary>
    /// How fast sound was made or read against its length, as the trainers print it: "3.6x real time" for 3.6 s of sound a
    /// second ("0.5x" is slower than real time). <paramref name="realTimeFactor"/> is the time taken over the sound's length.
    /// </summary>
    public static string Pace(double realTimeFactor) =>
        realTimeFactor > 0 ? string.Create(CultureInfo.InvariantCulture, $"{1 / realTimeFactor:0.0#}x real time") : "faster than measured";

    /// <summary>A positive number option (seconds, a share, a scale), or <paramref name="fallback"/> when absent.</summary>
    public static double Number(CommandContext context, string option, double fallback, double min = double.NegativeInfinity, double max = double.PositiveInfinity)
    {
        if (context.Option(option) is not { } text)
        {
            return fallback;
        }

        return double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out double value) && double.IsFinite(value) && value >= min && value <= max
            ? value
            : throw new UsageException(min > double.NegativeInfinity || max < double.PositiveInfinity
                ? string.Create(CultureInfo.InvariantCulture, $"{option} needs a number from {min} to {(double.IsPositiveInfinity(max) ? "up" : max)}, not '{text}'.")
                : $"{option} needs a number, not '{text}'.");
    }

    /// <summary>A number option, or null when absent.</summary>
    public static double? OptionalNumber(CommandContext context, string option, double min = double.NegativeInfinity, double max = double.PositiveInfinity) =>
        context.Option(option) is null ? null : Number(context, option, 0, min, max);

    // A registry's refusal of a format with the remedy: load a plug-in that registers it (naming this repository's sample when there is one).
    private static string Remedy(string message, string extension) =>
        Messages.T("{0} Load its plug-in with -P (--plugin).", message)
        + (SamplePlugins.TryGetValue(extension, out var plugin) ? $" For {extension} files: -P {plugin}." : "");
}
