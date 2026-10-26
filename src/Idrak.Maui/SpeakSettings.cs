// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>
/// The Speak lab's settings: the synthesis model and voice the page chose, every <see cref="SpeechSynthesisOptions"/>
/// value, the speaker it plays on and how the speech is saved. A null value is the library's own default
/// (<see cref="SpeechSynthesisOptions"/>, <see cref="VoiceConversionOptions"/>, <see cref="AudioPlaybackOptions"/>), read
/// from the library, never repeated here; the pieces the text is spoken in (<see cref="MaxCharacters"/>) default to the
/// lab's <see cref="SpeakSettings.DefaultMaxCharacters"/>, so the first sentence plays while the next is made. Kept between
/// runs by <see cref="SpeakSettings"/>.
/// </summary>
public sealed class SpeakLabOptions
{
    /// <summary>The synthesis model chosen, as the model store names it ("owner/name@revision"); null: none yet.</summary>
    public string? Model { get; set; }

    /// <summary>The voice chosen (a speaker of the model or an installed voice, <see cref="SpeechSynthesisOptions.Voice"/>); null: the model's own.</summary>
    public string? Voice { get; set; }

    /// <summary>The speaking rate (<see cref="SpeechSynthesisOptions.Rate"/>: 2 twice as fast); null: the library's.</summary>
    public double? Rate { get; set; }

    /// <summary><see cref="SpeechSynthesisOptions.NoiseScale"/>; null: the family's configured value.</summary>
    public double? NoiseScale { get; set; }

    /// <summary><see cref="SpeechSynthesisOptions.DurationNoiseScale"/>; null: the family's configured value.</summary>
    public double? DurationNoiseScale { get; set; }

    /// <summary><see cref="SpeechSynthesisOptions.Seed"/> (and the conversion's); null: the library's.</summary>
    public int? Seed { get; set; }

    /// <summary><see cref="SpeechSynthesisOptions.Vocoder"/> (a <see cref="Vocoders"/> name); null: the model's own.</summary>
    public string? Vocoder { get; set; }

    /// <summary><see cref="SpeechSynthesisOptions.SampleRate"/>; null: the model's rate.</summary>
    public int? SampleRate { get; set; }

    /// <summary>
    /// <see cref="SpeechSynthesisOptions.MaxCharacters"/>: the pieces the text is spoken in, each played as soon as it is
    /// made; null: <see cref="SpeakSettings.DefaultMaxCharacters"/>; 0: the whole text in one piece.
    /// </summary>
    public int? MaxCharacters { get; set; }

    /// <summary><see cref="SpeechSynthesisOptions.DecoderSeconds"/>; null: the model's own setting.</summary>
    public double? DecoderSeconds { get; set; }

    /// <summary><see cref="SpeechSynthesisOptions.VoiceConversion"/> ("auto", "off" or a <see cref="VoiceConverters"/> name); null: the library's ("auto").</summary>
    public string? VoiceConversion { get; set; }

    /// <summary><see cref="VoiceConversionOptions.Tau"/>; null: the library's.</summary>
    public double? Tau { get; set; }

    /// <summary><see cref="SpeechSynthesisOptions.VoiceStore"/> (a <see cref="VoiceStores"/> name); null: the library's ("folder").</summary>
    public string? VoiceStore { get; set; }

    /// <summary>The playback (an <see cref="AudioPlaybacks"/> name); null: <see cref="AudioPlaybacks.ForThisMachine"/> (aaudio on Android).</summary>
    public string? Playback { get; set; }

    /// <summary>The output device (an id the playback lists); null: the system's default.</summary>
    public string? OutputDevice { get; set; }

    /// <summary><see cref="AudioPlaybackOptions.BufferSeconds"/>; null: the library's.</summary>
    public double? BufferSeconds { get; set; }

    /// <summary><see cref="AudioPlaybackOptions.BlockSeconds"/>; null: the library's.</summary>
    public double? BlockSeconds { get; set; }

    /// <summary><see cref="AudioPlaybackOptions.Performance"/>; null: the library's.</summary>
    public AudioPerformance? Performance { get; set; }

    /// <summary>The encoder the speech is saved with (an <see cref="AudioEncoders"/> name: "wav", "mp3"); null: "wav".</summary>
    public string? SaveFormat { get; set; }

    /// <summary>The samples a saved WAV holds ("16", "24", "32", "8", "float"); null: 16-bit.</summary>
    public string? Bits { get; set; }

    /// <summary>A copy.</summary>
    public SpeakLabOptions Clone() => (SpeakLabOptions)MemberwiseClone();

    /// <summary>How the text is spoken: these values over the library's defaults.</summary>
    public SpeechSynthesisOptions ToSynthesis()
    {
        var library = new SpeechSynthesisOptions();
        var conversion = new VoiceConversionOptions();
        return library with
        {
            Voice = string.IsNullOrWhiteSpace(Voice) ? library.Voice : Voice,
            VoiceConversion = string.IsNullOrWhiteSpace(VoiceConversion) ? library.VoiceConversion : VoiceConversion,
            Conversion = Tau is null && Seed is null ? library.Conversion : conversion with { Tau = Tau is { } tau ? (float)tau : conversion.Tau, Seed = Seed ?? conversion.Seed },
            VoiceStore = string.IsNullOrWhiteSpace(VoiceStore) ? library.VoiceStore : VoiceStore,
            Rate = Rate is { } rate ? (float)rate : library.Rate,
            NoiseScale = NoiseScale is { } noise ? (float)noise : library.NoiseScale,
            DurationNoiseScale = DurationNoiseScale is { } durations ? (float)durations : library.DurationNoiseScale,
            Seed = Seed ?? library.Seed,
            Vocoder = string.IsNullOrWhiteSpace(Vocoder) ? library.Vocoder : Vocoder,
            SampleRate = SampleRate ?? library.SampleRate,
            MaxCharacters = MaxCharacters ?? SpeakSettings.DefaultMaxCharacters,
            DecoderSeconds = DecoderSeconds ?? library.DecoderSeconds,
        };
    }

    /// <summary>How the speech is played: the playback and its output options over the library's defaults.</summary>
    public SoundPlayRequest ToPlayback(string label = "speech")
    {
        var library = new AudioPlaybackOptions();
        return new SoundPlayRequest
        {
            Label = label,
            Playback = string.IsNullOrWhiteSpace(Playback) ? null : Playback,
            Output = library with
            {
                Device = string.IsNullOrWhiteSpace(OutputDevice) ? library.Device : OutputDevice,
                BufferSeconds = BufferSeconds ?? library.BufferSeconds,
                BlockSeconds = BlockSeconds ?? library.BlockSeconds,
                Performance = Performance ?? library.Performance,
            },
        };
    }

    /// <summary>The encoder options the speech is saved with.</summary>
    public AudioEncodeOptions ToEncode() => LabSettings.Encode(Bits);
}

/// <summary>
/// The Speak lab's settings kept between runs, each under <see cref="Prefix"/> and its key in the app's
/// <see cref="ISettingsStore"/>: the model and voice the page chose, and every <see cref="Definitions"/> value.
/// </summary>
public static class SpeakSettings
{
    /// <summary>What every key starts with.</summary>
    public const string Prefix = IdrakMauiSettings.Prefix + "speak.";

    /// <summary>
    /// The characters of the pieces the text is spoken in when none is set: a sentence or two, so the first plays while
    /// the next is made (the library's own default, 0, speaks the whole text in one piece).
    /// </summary>
    public const int DefaultMaxCharacters = 160;

    private const string ModelKey = Prefix + "model", VoiceKey = Prefix + "voice";

    /// <summary>The settings edited on the page, in its order.</summary>
    public static IReadOnlyList<LabSetting<SpeakLabOptions>> Definitions { get; } = Define();

    /// <summary>The options saved in <paramref name="store"/> (the library's defaults where nothing is); a saved value that no longer reads is skipped and named.</summary>
    public static SpeakLabOptions Load(ISettingsStore store, out IReadOnlyList<string> skipped)
    {
        ArgumentNullException.ThrowIfNull(store);
        var options = new SpeakLabOptions { Model = LabSettings.Empty(store.Get(ModelKey)), Voice = LabSettings.Empty(store.Get(VoiceKey)) };
        return LabSettings.Load(store, Prefix, Definitions, options, out skipped);
    }

    /// <summary>Keeps every value of <paramref name="options"/> in <paramref name="store"/> (an empty one removed: the library's own).</summary>
    public static void Save(ISettingsStore store, SpeakLabOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);
        store.Set(ModelKey, LabSettings.Empty(options.Model));
        store.Set(VoiceKey, LabSettings.Empty(options.Voice));
        LabSettings.Save(store, Prefix, Definitions, options);
    }

    private static IReadOnlyList<LabSetting<SpeakLabOptions>> Define()
    {
        var speech = new SpeechSynthesisOptions();
        var conversion = new VoiceConversionOptions();
        var output = new AudioPlaybackOptions();
        return
        [
            new("rate", "Speaking rate", "2 speaks twice as fast, 0.5 half as fast (SpeechSynthesisOptions.Rate).",
                LabSettingKind.Number, LabSettings.Show(speech.Rate), o => LabSettings.Show(o.Rate), (o, t) => o.Rate = LabSettings.Real(t, above: 0)),
            new("noise", "Noise", "The noise added to the sound's latents; 0 gives the same sound every time (NoiseScale). Empty: the model's own.",
                LabSettingKind.Number, "the model's", o => LabSettings.Show(o.NoiseScale), (o, t) => o.NoiseScale = LabSettings.Real(t, atLeast: 0)),
            new("duration_noise", "Duration noise", "The noise of the sounds' lengths; 0 gives the same timing every time (DurationNoiseScale). Empty: the model's own.",
                LabSettingKind.Number, "the model's", o => LabSettings.Show(o.DurationNoiseScale), (o, t) => o.DurationNoiseScale = LabSettings.Real(t, atLeast: 0)),
            new("seed", "Seed", "The seed of the noise: the same text and seed give the same sound (Seed).",
                LabSettingKind.Number, LabSettings.Show(speech.Seed), o => LabSettings.Show(o.Seed), (o, t) => o.Seed = LabSettings.Whole(t, int.MinValue)),
            new("pieces", "Piece (characters)", "The text is spoken in sentences joined up to this many characters, each played as soon as it is made; 0: the whole text at once (MaxCharacters).",
                LabSettingKind.Number, LabSettings.Show(DefaultMaxCharacters), o => LabSettings.Show(o.MaxCharacters), (o, t) => o.MaxCharacters = LabSettings.Whole(t, 0)),
            new("decoder_s", "Decoder piece (s)", "The seconds of sound the decoder makes at a time: memory follows the piece, the samples are the same; 0: one pass (DecoderSeconds). Empty: the model's own.",
                LabSettingKind.Number, "the model's", o => LabSettings.Show(o.DecoderSeconds), (o, t) => o.DecoderSeconds = LabSettings.Real(t, atLeast: 0)),
            new("sample_rate", "Sample rate (Hz)", "The speech resampled to this rate (SampleRate). Empty: the model's rate.",
                LabSettingKind.Number, "the model's", o => LabSettings.Show(o.SampleRate), (o, t) => o.SampleRate = LabSettings.Whole(t, 1000, 384_000)),
            new("vocoder", "Vocoder", "A registered vocoder, for a model that makes a spectrogram (Vocoders). Default: the model's own.", LabSettingKind.Choice, "default",
                o => o.Vocoder ?? "", (o, t) => o.Vocoder = LabSettings.Registered(t, Vocoders.Names, "a vocoder"), () => ["default", .. Vocoders.Names]),
            new("voice_conversion", "Voice conversion", "An installed embedding voice: auto converts the speech with the converter it was made with, off speaks in the model's own voice (VoiceConversion).",
                LabSettingKind.Choice, $"default ({speech.VoiceConversion})", o => o.VoiceConversion ?? "", (o, t) => o.VoiceConversion = Conversion(t),
                () => ["default", "auto", "off", .. VoiceConverters.Names]),
            new("tau", "Conversion looseness (tau)", "How loosely a converted voice copies the speech's detail; 0 is deterministic (VoiceConversionOptions.Tau).",
                LabSettingKind.Number, LabSettings.Show(conversion.Tau), o => LabSettings.Show(o.Tau), (o, t) => o.Tau = LabSettings.Real(t, atLeast: 0)),
            new("voice_store", "Voice store", "Where installed voices are looked up (VoiceStores).", LabSettingKind.Choice, $"default ({VoiceStores.FolderName})",
                o => o.VoiceStore ?? "", (o, t) => o.VoiceStore = LabSettings.Registered(t, VoiceStores.Names, "a voice store"), () => ["default", .. VoiceStores.Names]),
            new("playback", "Playback", "The system's sound output (AudioPlaybacks); default: the first that can play here (aaudio on Android).", LabSettingKind.Choice, "default",
                o => o.Playback ?? "", (o, t) => o.Playback = LabSettings.Registered(t, AudioPlaybacks.Names, "an audio playback"), () => ["default", .. AudioPlaybacks.Names]),
            new("output_device", "Output device", "The output device's id as the playback lists it; empty: the system's default.",
                LabSettingKind.Text, "the system's default", o => o.OutputDevice ?? "", (o, t) => o.OutputDevice = t.Length == 0 ? null : t),
            new("buffer_s", "Playback buffer (s)", "The seconds of sound held ahead of the speaker: what a write waits for (AudioPlaybackOptions.BufferSeconds).",
                LabSettingKind.Number, LabSettings.Show(output.BufferSeconds), o => LabSettings.Show(o.BufferSeconds), (o, t) => o.BufferSeconds = LabSettings.Real(t, above: 0)),
            new("block_s", "Playback block (s)", "The seconds of sound handed to the device at a time (AudioPlaybackOptions.BlockSeconds).",
                LabSettingKind.Number, LabSettings.Show(output.BlockSeconds), o => LabSettings.Show(o.BlockSeconds), (o, t) => o.BlockSeconds = LabSettings.Real(t, above: 0)),
            new("performance", "Playback mode", "What the output is tuned for where the system lets it choose (AudioPlaybackOptions.Performance).", LabSettingKind.Choice,
                $"default ({output.Performance})", o => o.Performance?.ToString() ?? "", (o, t) => o.Performance = LabSettings.Performance(t),
                () => ["default", .. Enum.GetNames<AudioPerformance>()]),
            new("format", "Save as", "The encoder the speech is saved with (AudioEncoders: wav, mp3).", LabSettingKind.Choice, "wav",
                o => o.SaveFormat ?? "", (o, t) => o.SaveFormat = LabSettings.Registered(t, AudioEncoders.Names, "an audio encoder"), () => ["wav", .. AudioEncoders.Names.Where(n => n != "wav")]),
            new("bits", "WAV samples", "The samples a saved WAV holds: 16-bit (the default), 24, 32, 8 or 32-bit float.", LabSettingKind.Choice, "16",
                o => o.Bits ?? "", (o, t) => o.Bits = LabSettings.OneOf(t, LabSettings.SampleFormats) is { } bits && bits != "16" ? bits : null, () => LabSettings.SampleFormats),
        ];
    }

    private static string? Conversion(string text) =>
        LabSettings.IsDefault(text) ? null
        : text.Equals("auto", StringComparison.OrdinalIgnoreCase) || text.Equals("off", StringComparison.OrdinalIgnoreCase) ? text.ToLowerInvariant()
        : VoiceConverters.Names.Contains(text, StringComparer.Ordinal) ? text
        : throw new FormatException($"'{text}' is not auto, off or a voice converter (registered: {(VoiceConverters.Names.Count == 0 ? "none" : string.Join(", ", VoiceConverters.Names))}).");
}
