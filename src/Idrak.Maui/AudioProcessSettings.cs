// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio.Abstractions;
using Idrak.Maui.Abstractions;

namespace Idrak.Maui;

/// <summary>
/// The Audio process lab's settings: the pipeline the page holds, how it is applied (whole, or in blocks as live sound
/// goes), the microphone, the speaker, live monitoring and how the result is saved. A null value is the library's own
/// default (<see cref="AudioCaptureOptions"/>, <see cref="AudioPlaybackOptions"/>); the lab's own defaults (the longest
/// recording, monitoring's short blocks and buffer) are <see cref="AudioProcessSettings"/>' constants. Kept between runs by
/// <see cref="AudioProcessSettings"/>.
/// </summary>
public sealed class AudioProcessLabOptions
{
    /// <summary>The pipeline (<see cref="AudioPipeline"/> text) the page holds; null: none yet.</summary>
    public string? Pipeline { get; set; }

    /// <summary>The seconds of a block the sound goes through the pipeline in (as live sound goes; the same samples); null or 0: the whole recording at once (<see cref="AudioPipeline.Apply"/>).</summary>
    public double? ProcessBlockSeconds { get; set; }

    /// <summary>Whether each apply runs a second time to be timed warm (after the first, cold, call compiled the code); null: on.</summary>
    public bool? WarmRun { get; set; }

    /// <summary>The capture (an <see cref="AudioCaptures"/> name); null: <see cref="AudioCaptures.ForThisMachine"/> (aaudio on Android).</summary>
    public string? Capture { get; set; }

    /// <summary>The input device (an id the capture lists); null: the system's default.</summary>
    public string? InputDevice { get; set; }

    /// <summary>The rate the microphone is asked for (<see cref="AudioCaptureOptions.SampleRate"/>); null: the library's.</summary>
    public int? SampleRate { get; set; }

    /// <summary>The channels recorded (<see cref="AudioCaptureOptions.Channels"/>); null: the library's.</summary>
    public int? Channels { get; set; }

    /// <summary>The seconds of sound a captured block holds (<see cref="AudioCaptureOptions.BlockSeconds"/>); null: the library's.</summary>
    public double? CaptureBlockSeconds { get; set; }

    /// <summary>The longest recording, in seconds (it stops itself there); null: <see cref="AudioProcessSettings.DefaultMaxRecordSeconds"/>.</summary>
    public double? MaxRecordSeconds { get; set; }

    /// <summary>The playback (an <see cref="AudioPlaybacks"/> name); null: <see cref="AudioPlaybacks.ForThisMachine"/> (aaudio on Android).</summary>
    public string? Playback { get; set; }

    /// <summary>The output device (an id the playback lists); null: the system's default.</summary>
    public string? OutputDevice { get; set; }

    /// <summary><see cref="AudioPlaybackOptions.BufferSeconds"/> for playing a recording; null: the library's.</summary>
    public double? BufferSeconds { get; set; }

    /// <summary>The captured block while monitoring (what each block adds to the delay); null: <see cref="AudioProcessSettings.DefaultMonitorBlockSeconds"/>.</summary>
    public double? MonitorBlockSeconds { get; set; }

    /// <summary>The playback buffer while monitoring (the most it adds to the delay); null: <see cref="AudioProcessSettings.DefaultMonitorBufferSeconds"/>.</summary>
    public double? MonitorBufferSeconds { get; set; }

    /// <summary>What monitoring's streams are tuned for where the system lets it choose; null: <see cref="AudioPerformance.LowLatency"/>.</summary>
    public AudioPerformance? MonitorPerformance { get; set; }

    /// <summary>The encoder the result is saved with (an <see cref="AudioEncoders"/> name); null: "wav".</summary>
    public string? SaveFormat { get; set; }

    /// <summary>The samples a saved WAV holds ("16", "24", "32", "8", "float"); null: 16-bit.</summary>
    public string? Bits { get; set; }

    /// <summary>A copy.</summary>
    public AudioProcessLabOptions Clone() => (AudioProcessLabOptions)MemberwiseClone();

    /// <summary>The pipeline, checked against the registered processors (empty when none).</summary>
    /// <exception cref="ArgumentException">An unknown processor or option, or a value out of range (the message names what is taken).</exception>
    public AudioPipeline ToPipeline() => AudioPipeline.Parse(Pipeline);

    /// <summary>How the microphone is opened to record: the device, rate, channels and block over the library's defaults.</summary>
    public AudioCaptureOptions ToCapture()
    {
        var library = new AudioCaptureOptions();
        return library with
        {
            Device = string.IsNullOrWhiteSpace(InputDevice) ? library.Device : InputDevice,
            SampleRate = SampleRate ?? library.SampleRate,
            Channels = Channels ?? library.Channels,
            BlockSeconds = CaptureBlockSeconds ?? library.BlockSeconds,
        };
    }

    /// <summary>How the microphone is opened to monitor: as <see cref="ToCapture"/>, with monitoring's block and mode.</summary>
    public AudioCaptureOptions ToMonitorCapture() => ToCapture() with
    {
        BlockSeconds = MonitorBlockSeconds ?? AudioProcessSettings.DefaultMonitorBlockSeconds,
        Performance = MonitorPerformance ?? AudioPerformance.LowLatency,
    };

    /// <summary>How a recording is played (<paramref name="label"/>: "original", "processed").</summary>
    public SoundPlayRequest ToPlayback(string label)
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
            },
        };
    }

    /// <summary>How monitoring plays: a short buffer and blocks, tuned for a short delay.</summary>
    public SoundPlayRequest ToMonitorPlayback()
    {
        var request = ToPlayback("monitor");
        double buffer = MonitorBufferSeconds ?? AudioProcessSettings.DefaultMonitorBufferSeconds;
        return request with
        {
            Output = request.Output with
            {
                BufferSeconds = buffer,
                BlockSeconds = Math.Min(request.Output.BlockSeconds, buffer),
                Performance = MonitorPerformance ?? AudioPerformance.LowLatency,
            },
        };
    }

    /// <summary>The encoder options the result is saved with.</summary>
    public AudioEncodeOptions ToEncode() => LabSettings.Encode(Bits);
}

/// <summary>
/// The Audio process lab's settings kept between runs, each under <see cref="Prefix"/> and its key in the app's
/// <see cref="ISettingsStore"/>: the pipeline the page holds, and every <see cref="Definitions"/> value.
/// </summary>
public static class AudioProcessSettings
{
    /// <summary>What every key starts with.</summary>
    public const string Prefix = IdrakMauiSettings.Prefix + "audio_process.";

    /// <summary>The longest recording when none is set, in seconds (the recording is kept in memory).</summary>
    public const double DefaultMaxRecordSeconds = 300;

    /// <summary>The captured block while monitoring when none is set: each block is a delay.</summary>
    public const double DefaultMonitorBlockSeconds = 0.01;

    /// <summary>The playback buffer while monitoring when none is set: the most it adds to the delay.</summary>
    public const double DefaultMonitorBufferSeconds = 0.08;

    private const string PipelineKey = Prefix + "pipeline";

    /// <summary>The settings edited on the page, in its order.</summary>
    public static IReadOnlyList<LabSetting<AudioProcessLabOptions>> Definitions { get; } = Define();

    /// <summary>The options saved in <paramref name="store"/> (the defaults where nothing is); a saved value that no longer reads is skipped and named.</summary>
    public static AudioProcessLabOptions Load(ISettingsStore store, out IReadOnlyList<string> skipped)
    {
        ArgumentNullException.ThrowIfNull(store);
        var options = new AudioProcessLabOptions { Pipeline = LabSettings.Empty(store.Get(PipelineKey)) };
        return LabSettings.Load(store, Prefix, Definitions, options, out skipped);
    }

    /// <summary>Keeps every value of <paramref name="options"/> in <paramref name="store"/> (an empty one removed).</summary>
    public static void Save(ISettingsStore store, AudioProcessLabOptions options)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(options);
        store.Set(PipelineKey, LabSettings.Empty(options.Pipeline));
        LabSettings.Save(store, Prefix, Definitions, options);
    }

    private static IReadOnlyList<LabSetting<AudioProcessLabOptions>> Define()
    {
        var capture = new AudioCaptureOptions();
        var output = new AudioPlaybackOptions();
        return
        [
            new("process_block_s", "Process in blocks (s)", "The sound goes through the pipeline in blocks of this many seconds, as live sound goes (the same samples); empty or 0: the whole recording at once.",
                LabSettingKind.Number, "whole", o => LabSettings.Show(o.ProcessBlockSeconds), (o, t) => o.ProcessBlockSeconds = LabSettings.Real(t, atLeast: 0)),
            new("warm_run", "Time it warm too", "Run each apply a second time, timed apart from the first (cold) call, which compiles the code.",
                LabSettingKind.Choice, "on", o => LabSettings.Show(o.WarmRun), (o, t) => o.WarmRun = LabSettings.Switch(t), () => ["default", "on", "off"]),
            new("capture", "Capture", "The system's sound input (AudioCaptures); default: the first that can record here (aaudio on Android).", LabSettingKind.Choice, "default",
                o => o.Capture ?? "", (o, t) => o.Capture = LabSettings.Registered(t, AudioCaptures.Names, "an audio capture"), () => ["default", .. AudioCaptures.Names]),
            new("input_device", "Input device", "The input device's id as the capture lists it; empty: the system's default.",
                LabSettingKind.Text, "the system's default", o => o.InputDevice ?? "", (o, t) => o.InputDevice = t.Length == 0 ? null : t),
            new("sample_rate", "Sample rate (Hz)", "The rate the microphone is asked for (AudioCaptureOptions.SampleRate).",
                LabSettingKind.Number, LabSettings.Show(capture.SampleRate), o => LabSettings.Show(o.SampleRate), (o, t) => o.SampleRate = LabSettings.Whole(t, 1000, 384_000)),
            new("channels", "Channels", "The channels recorded (AudioCaptureOptions.Channels).",
                LabSettingKind.Number, LabSettings.Show(capture.Channels), o => LabSettings.Show(o.Channels), (o, t) => o.Channels = LabSettings.Whole(t, 1, 32)),
            new("capture_block_s", "Capture block (s)", "The seconds of sound each captured block holds (AudioCaptureOptions.BlockSeconds).",
                LabSettingKind.Number, LabSettings.Show(capture.BlockSeconds), o => LabSettings.Show(o.CaptureBlockSeconds), (o, t) => o.CaptureBlockSeconds = LabSettings.Real(t, above: 0)),
            new("max_record_s", "Longest recording (s)", "A recording stops itself after this many seconds (it is kept in memory).",
                LabSettingKind.Number, LabSettings.Show(DefaultMaxRecordSeconds), o => LabSettings.Show(o.MaxRecordSeconds), (o, t) => o.MaxRecordSeconds = LabSettings.Real(t, above: 0)),
            new("playback", "Playback", "The system's sound output (AudioPlaybacks); default: the first that can play here (aaudio on Android).", LabSettingKind.Choice, "default",
                o => o.Playback ?? "", (o, t) => o.Playback = LabSettings.Registered(t, AudioPlaybacks.Names, "an audio playback"), () => ["default", .. AudioPlaybacks.Names]),
            new("output_device", "Output device", "The output device's id as the playback lists it; empty: the system's default.",
                LabSettingKind.Text, "the system's default", o => o.OutputDevice ?? "", (o, t) => o.OutputDevice = t.Length == 0 ? null : t),
            new("buffer_s", "Playback buffer (s)", "The seconds of sound held ahead of the speaker when a recording plays (AudioPlaybackOptions.BufferSeconds).",
                LabSettingKind.Number, LabSettings.Show(output.BufferSeconds), o => LabSettings.Show(o.BufferSeconds), (o, t) => o.BufferSeconds = LabSettings.Real(t, above: 0)),
            new("monitor_block_s", "Monitor block (s)", "The seconds of each captured block while monitoring: each block is a delay.",
                LabSettingKind.Number, LabSettings.Show(DefaultMonitorBlockSeconds), o => LabSettings.Show(o.MonitorBlockSeconds), (o, t) => o.MonitorBlockSeconds = LabSettings.Real(t, above: 0)),
            new("monitor_buffer_s", "Monitor buffer (s)", "The playback buffer while monitoring: the most it adds to the delay.",
                LabSettingKind.Number, LabSettings.Show(DefaultMonitorBufferSeconds), o => LabSettings.Show(o.MonitorBufferSeconds), (o, t) => o.MonitorBufferSeconds = LabSettings.Real(t, above: 0)),
            new("monitor_performance", "Monitor mode", "What monitoring's streams are tuned for where the system lets it choose (AudioPerformance).", LabSettingKind.Choice,
                $"default ({AudioPerformance.LowLatency})", o => o.MonitorPerformance?.ToString() ?? "", (o, t) => o.MonitorPerformance = LabSettings.Performance(t),
                () => ["default", .. Enum.GetNames<AudioPerformance>()]),
            new("format", "Save as", "The encoder the processed sound is saved with (AudioEncoders: wav, mp3).", LabSettingKind.Choice, "wav",
                o => o.SaveFormat ?? "", (o, t) => o.SaveFormat = LabSettings.Registered(t, AudioEncoders.Names, "an audio encoder"), () => ["wav", .. AudioEncoders.Names.Where(n => n != "wav")]),
            new("bits", "WAV samples", "The samples a saved WAV holds: 16-bit (the default), 24, 32, 8 or 32-bit float.", LabSettingKind.Choice, "16",
                o => o.Bits ?? "", (o, t) => o.Bits = LabSettings.OneOf(t, LabSettings.SampleFormats) is { } bits && bits != "16" ? bits : null, () => LabSettings.SampleFormats),
        ];
    }
}
