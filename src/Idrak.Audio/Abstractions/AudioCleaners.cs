// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>The settings of the cleaning stages (<see cref="AudioCleaners"/>); each stage reads its own.</summary>
public sealed record AudioCleaningOptions
{
    /// <summary>"highpass": the cutoff, Hz (a second-order Butterworth high-pass: the DC offset and the rumble below it go); 0 leaves the sound as it is.</summary>
    public double HighPassHertz { get; init; } = 60;

    /// <summary>"trim" and "normalize": a frame more than this many dB below the loudest frame is silence.</summary>
    public double SilenceDecibels { get; init; } = 40;

    /// <summary>"trim" and "normalize": the frames the level is measured over, seconds.</summary>
    public double HopSeconds { get; init; } = 0.01;

    /// <summary>"trim": the silence kept on both sides of the speech, seconds.</summary>
    public double TrimPadSeconds { get; init; } = 0.1;

    /// <summary>"normalize": the speech level aimed at, dBFS (the RMS over the frames that are not silence).</summary>
    public double TargetDecibels { get; init; } = -20;

    /// <summary>"normalize": the highest sample allowed after the gain (the gain is lowered to keep under it).</summary>
    public double PeakCeiling { get; init; } = 0.95;

    /// <summary>"denoise": the share of the frames, the quietest ones (the pauses), the noise floor is read from (0 to 1).</summary>
    public double NoiseQuantile { get; init; } = 0.1;

    /// <summary>"denoise": how many times the noise floor is taken off each frequency's magnitude (1: the floor itself; more removes more noise and thins the voice).</summary>
    public double DenoiseStrength { get; init; } = 1.5;

    /// <summary>"denoise": the most a frequency is turned down, dB (at or below 0); a floor keeps the sound natural.</summary>
    public double DenoiseFloorDecibels { get; init; } = -25;

    /// <summary>"denoise": the window the spectrum is taken over, seconds (rounded down to a power of two of samples; the hop a quarter of it).</summary>
    public double DenoiseWindowSeconds { get; init; } = 0.032;

    /// <summary>Settings a plug-in's stage reads (the built-in stages take none).</summary>
    public JsonObject? Settings { get; init; }

    /// <summary>The options as JSON (<see cref="FromJson"/> reads it back).</summary>
    public JsonObject ToJson() => new()
    {
        ["high_pass_hertz"] = HighPassHertz,
        ["silence_decibels"] = SilenceDecibels,
        ["hop_seconds"] = HopSeconds,
        ["trim_pad_seconds"] = TrimPadSeconds,
        ["target_decibels"] = TargetDecibels,
        ["peak_ceiling"] = PeakCeiling,
        ["noise_quantile"] = NoiseQuantile,
        ["denoise_strength"] = DenoiseStrength,
        ["denoise_floor_decibels"] = DenoiseFloorDecibels,
        ["denoise_window_seconds"] = DenoiseWindowSeconds,
        ["settings"] = Settings?.DeepClone(),
    };

    /// <summary>Options from their JSON (<see cref="ToJson"/>); a key missing keeps the default.</summary>
    public static AudioCleaningOptions FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        var d = new AudioCleaningOptions();
        return new AudioCleaningOptions
        {
            HighPassHertz = (double?)json["high_pass_hertz"] ?? d.HighPassHertz,
            SilenceDecibels = (double?)json["silence_decibels"] ?? d.SilenceDecibels,
            HopSeconds = (double?)json["hop_seconds"] ?? d.HopSeconds,
            TrimPadSeconds = (double?)json["trim_pad_seconds"] ?? d.TrimPadSeconds,
            TargetDecibels = (double?)json["target_decibels"] ?? d.TargetDecibels,
            PeakCeiling = (double?)json["peak_ceiling"] ?? d.PeakCeiling,
            NoiseQuantile = (double?)json["noise_quantile"] ?? d.NoiseQuantile,
            DenoiseStrength = (double?)json["denoise_strength"] ?? d.DenoiseStrength,
            DenoiseFloorDecibels = (double?)json["denoise_floor_decibels"] ?? d.DenoiseFloorDecibels,
            DenoiseWindowSeconds = (double?)json["denoise_window_seconds"] ?? d.DenoiseWindowSeconds,
            Settings = json["settings"] is JsonObject settings ? (JsonObject)settings.DeepClone() : null,
        };
    }

    /// <summary>Refuses values out of range.</summary>
    /// <exception cref="ArgumentException">A value is out of range.</exception>
    public void Check()
    {
        if (!(HighPassHertz >= 0) || !(SilenceDecibels > 0) || !(HopSeconds > 0) || !(TrimPadSeconds >= 0) || !(TargetDecibels <= 0) || !(PeakCeiling > 0) || PeakCeiling > 1
            || !(NoiseQuantile >= 0) || NoiseQuantile > 1 || !(DenoiseStrength >= 0) || !(DenoiseFloorDecibels <= 0) || !(DenoiseWindowSeconds > 0))
        {
            throw new ArgumentException("Cleaning options: HighPassHertz, TrimPadSeconds, NoiseQuantile and DenoiseStrength are not negative, SilenceDecibels, HopSeconds, "
                + "PeakCeiling and DenoiseWindowSeconds positive, PeakCeiling and NoiseQuantile at most 1, TargetDecibels and DenoiseFloorDecibels at most 0.");
        }
    }
}

/// <summary>
/// One stage of cleaning a recording before it is used: the trainers run the stages named in
/// <c>SpeechTrainingOptions.Cleaning</c> over every utterance's sound before its features are taken, and
/// <c>idrak data audio --clean</c> writes cleaned files. A stage is deterministic (the same sound gives the same result, so
/// the cleaned features are cached) and knows no model family; one that runs a model (a denoiser, a network that
/// isolates the voice) is a plug-in registered with <see cref="AudioCleaners.Register"/> under its <see cref="IKind.Name"/>
/// ("trim"), reading its own settings from <see cref="AudioCleaningOptions.Settings"/>: their keys are its
/// <see cref="IConfigurableKind.Keys"/> (none for the library's stages), so a setting no stage of a run reads is refused
/// (<see cref="AudioCleaners.CheckSettings"/>).
/// </summary>
public interface IAudioCleaner : IConfigurableKind
{
    /// <summary>The recording cleaned: any channel count, the same rate; the length may change (trimming), the samples are not written in place.</summary>
    AudioData Clean(AudioData audio, AudioCleaningOptions options);
}

/// <summary>
/// The cleaning stages, by name (ignoring case). Built in (Idrak.Audio): "highpass" (a second-order Butterworth high-pass
/// at <see cref="AudioCleaningOptions.HighPassHertz"/>: the DC offset and rumble go), "denoise" (spectral gating: the noise
/// floor read from the quietest frames, the pauses, every frame's magnitude at each frequency turned down by how little
/// it rises above that floor, at most <see cref="AudioCleaningOptions.DenoiseFloorDecibels"/>, through the STFT and back), "trim"
/// (the silence at both ends cut, <see cref="AudioCleaningOptions.TrimPadSeconds"/> kept) and "normalize" (the speech
/// level brought to <see cref="AudioCleaningOptions.TargetDecibels"/>, the peak kept under
/// <see cref="AudioCleaningOptions.PeakCeiling"/>). For a voice recording the order is highpass, denoise, trim,
/// normalize: the floor read before the silence is cut, the level set last.
/// </summary>
public static class AudioCleaners
{
    /// <summary>The DC offset and rumble filtered out.</summary>
    public const string HighPass = "highpass";

    /// <summary>The noise floor gated off.</summary>
    public const string Denoise = "denoise";

    /// <summary>The silence at both ends cut.</summary>
    public const string Trim = "trim";

    /// <summary>The speech level set.</summary>
    public const string Normalize = "normalize";

    private static readonly KindTable<IAudioCleaner> Table = BuiltIn();

    private static KindTable<IAudioCleaner> BuiltIn()
    {
        var table = new KindTable<IAudioCleaner>(nameof(AudioCleaners), "Audio cleaning stage", "add it with AudioCleaners.Register", comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a cleaning stage's result is what later stages, the features and the cache see, so another cannot stand in for it unnoticed");
        table.RegisterDefault(new HighPassCleaner());
        table.RegisterDefault(new DenoiseCleaner());
        table.RegisterDefault(new TrimCleaner());
        table.RegisterDefault(new NormalizeCleaner());
        return table;
    }

    /// <summary>Registers <paramref name="cleaner"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioCleaner cleaner) => Table.Register(cleaner, System.Reflection.Assembly.GetCallingAssembly());

    /// <summary>Removes the app's stage of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Names;

    /// <summary>The stage of that name, or null.</summary>
    public static IAudioCleaner? Find(ReadOnlySpan<char> name) => Table.Find(name);

    /// <summary>The stage of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IAudioCleaner Get(string name) => Table.Get(name);

    /// <summary>Each registered stage as "name: summary", one per line.</summary>
    public static string Describe() => Table.Describe();

    /// <summary>
    /// Checks that every key of <paramref name="options"/>' <see cref="AudioCleaningOptions.Settings"/> is read by one of
    /// <paramref name="stages"/> (one of their <see cref="IConfigurableKind.Keys"/>): a setting none reads is refused,
    /// naming the keys the stages take.
    /// </summary>
    /// <exception cref="NotSupportedException">A stage is not registered.</exception>
    /// <exception cref="ArgumentException">A setting no stage reads.</exception>
    public static void CheckSettings(IEnumerable<string> stages, AudioCleaningOptions options)
    {
        ArgumentNullException.ThrowIfNull(stages);
        ArgumentNullException.ThrowIfNull(options);
        if (options.Settings is not { Count: > 0 } settings)
        {
            return;
        }

        var cleaners = stages.Select(Get).ToList();
        foreach (var (key, _) in settings)
        {
            if (!cleaners.Any(c => c.Keys.Contains(key)))
            {
                var taken = cleaners.SelectMany(c => c.Keys).Distinct(StringComparer.Ordinal).ToList();
                throw new ArgumentException($"No cleaning stage of {string.Join(", ", cleaners.Select(c => c.Name))} reads the setting '{key}'; they read: "
                    + $"{(taken.Count == 0 ? "none" : string.Join(", ", taken))}.", nameof(options));
            }
        }
    }

    /// <summary>The library's stage of that name, or null.</summary>
    public static IAudioCleaner? Default(string name) => Table.Default(name);

    /// <summary>Who registered the stage of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>The recording through the stages named, in order.</summary>
    /// <exception cref="NotSupportedException">A name is not registered.</exception>
    public static AudioData Apply(AudioData audio, IEnumerable<string> stages, AudioCleaningOptions options)
    {
        ArgumentNullException.ThrowIfNull(audio);
        ArgumentNullException.ThrowIfNull(stages);
        ArgumentNullException.ThrowIfNull(options);
        foreach (string name in stages)
        {
            audio = Get(name).Clean(audio, options);
        }

        return audio;
    }
}
