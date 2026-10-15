// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Mp3Audio;

/// <summary>How <see cref="Mp3Encoder"/> writes two channels (and more).</summary>
public enum Mp3ChannelMode
{
    /// <summary>Two channels as left and right, or as mid and side in the frames where that takes fewer bits (joint stereo, no intensity stereo).</summary>
    JointStereo,

    /// <summary>Two channels as left and right in every frame.</summary>
    Stereo,

    /// <summary>One channel: the mean of the input's channels.</summary>
    Mono,
}

/// <summary>How hard <see cref="Mp3Encoder"/> works on each granule: its loops' effort, not the bit rate.</summary>
public enum Mp3Quality
{
    /// <summary>The rate loop only (no scalefactors): the noise spread evenly, the fastest.</summary>
    Fast,

    /// <summary>
    /// The rate loop inside a distortion loop that amplifies the bands whose noise passes the masking threshold (up to 8
    /// passes, stopping after 2 that did not lower it), then the best region split.
    /// </summary>
    Standard,

    /// <summary>Up to 24 distortion passes (stopping after 4 that did not help), more Huffman tables tried per region.</summary>
    High,
}

/// <summary>
/// How <see cref="Mp3Encoder"/> encodes: constant bit rate MPEG-1 Layer III (32, 44.1 and 48 kHz) or MPEG-2 Layer III
/// (16, 22.05 and 24 kHz). Every value has a default; <see cref="FromEnvironment"/> reads them from IDRAK_MP3_*
/// variables (what <see cref="Mp3AudioPlugin.Register"/> uses when it is given none).
/// </summary>
public sealed record Mp3EncodeOptions
{
    /// <summary>The sample rates MP3 Layer III is written at here (MPEG-2.5's 8, 11.025 and 12 kHz are not).</summary>
    public static IReadOnlyList<int> SampleRates { get; } = [16000, 22050, 24000, 32000, 44100, 48000];

    private static readonly int[] Mpeg1Bitrates = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
    private static readonly int[] Mpeg2Bitrates = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];

    /// <summary>
    /// The constant bit rate in kbit/s; null: <see cref="DefaultBitrate"/> for the output's rate and channels. It must be
    /// one of <see cref="Bitrates"/> for the output's rate.
    /// </summary>
    public int? Bitrate { get; init; }

    /// <summary>
    /// The output's sample rate, one of <see cref="SampleRates"/>; null: the input's rate when it is one, else the
    /// nearest one (the sound resampled with <see cref="Resampler.Resample"/>).
    /// </summary>
    public int? SampleRate { get; init; }

    /// <summary>How two channels are written (<see cref="Mp3ChannelMode.JointStereo"/> unless set); one channel is written as one.</summary>
    public Mp3ChannelMode ChannelMode { get; init; } = Mp3ChannelMode.JointStereo;

    /// <summary>The loops' effort (<see cref="Mp3Quality.Standard"/> unless set).</summary>
    public Mp3Quality Quality { get; init; } = Mp3Quality.Standard;

    /// <summary>
    /// The highest frequency kept, in Hz (the spectrum above it is not coded); null: chosen by the bits a sample
    /// gets (none at 2.5 bits a sample and a channel or more, about 17 kHz at 128 kbit/s 44.1 kHz stereo); 0: none.
    /// </summary>
    public int? LowpassHz { get; init; }

    /// <summary>
    /// Frames a worker analyses at a time; the frames of all workers are then quantized in parallel and packed in order.
    /// The bytes are the same for every value; 0 or less encodes in order on one thread.
    /// </summary>
    public int FramesPerTask { get; init; } = 32;

    /// <summary>The bit rates (kbit/s) MP3 has at <paramref name="sampleRate"/>: MPEG-1's from 32 to 320, MPEG-2's from 8 to 160.</summary>
    /// <exception cref="ArgumentException">The rate is not one of <see cref="SampleRates"/>.</exception>
    public static IReadOnlyList<int> Bitrates(int sampleRate) => sampleRate switch
    {
        32000 or 44100 or 48000 => Mpeg1Bitrates,
        16000 or 22050 or 24000 => Mpeg2Bitrates,
        _ => throw new ArgumentException($"MP3 is written at {string.Join(", ", SampleRates)} Hz, not {sampleRate}.", nameof(sampleRate)),
    };

    /// <summary>
    /// The bit rate used when none is set: 64 kbit/s a channel at 32, 44.1 and 48 kHz (128 for stereo), 64 for one
    /// channel and 96 for two at 16, 22.05 and 24 kHz.
    /// </summary>
    public static int DefaultBitrate(int sampleRate, int channels) => sampleRate >= 32000 ? (channels == 1 ? 64 : 128) : (channels == 1 ? 64 : 96);

    /// <summary>
    /// Options from the environment, the defaults where a variable is unset: IDRAK_MP3_BITRATE (kbit/s), IDRAK_MP3_RATE
    /// (Hz), IDRAK_MP3_CHANNELS (joint, stereo or mono), IDRAK_MP3_QUALITY (fast, standard or high), IDRAK_MP3_LOWPASS
    /// (Hz, 0 for none), IDRAK_MP3_FRAMES_PER_TASK.
    /// </summary>
    /// <exception cref="ArgumentException">A variable holds no value it takes; the message names it.</exception>
    public static Mp3EncodeOptions FromEnvironment()
    {
        var options = new Mp3EncodeOptions();
        if (Read("IDRAK_MP3_BITRATE") is { } bitrate)
        {
            options = options with { Bitrate = Number("IDRAK_MP3_BITRATE", bitrate) };
        }

        if (Read("IDRAK_MP3_RATE") is { } rate)
        {
            options = options with { SampleRate = Number("IDRAK_MP3_RATE", rate) };
        }

        if (Read("IDRAK_MP3_CHANNELS") is { } channels)
        {
            options = options with
            {
                ChannelMode = channels.ToLowerInvariant() switch
                {
                    "joint" or "jointstereo" => Mp3ChannelMode.JointStereo,
                    "stereo" => Mp3ChannelMode.Stereo,
                    "mono" => Mp3ChannelMode.Mono,
                    _ => throw new ArgumentException($"IDRAK_MP3_CHANNELS is joint, stereo or mono, not '{channels}'."),
                },
            };
        }

        if (Read("IDRAK_MP3_QUALITY") is { } quality)
        {
            options = options with
            {
                Quality = quality.ToLowerInvariant() switch
                {
                    "fast" => Mp3Quality.Fast,
                    "standard" => Mp3Quality.Standard,
                    "high" => Mp3Quality.High,
                    _ => throw new ArgumentException($"IDRAK_MP3_QUALITY is fast, standard or high, not '{quality}'."),
                },
            };
        }

        if (Read("IDRAK_MP3_LOWPASS") is { } lowpass)
        {
            options = options with { LowpassHz = Number("IDRAK_MP3_LOWPASS", lowpass) };
        }

        if (Read("IDRAK_MP3_FRAMES_PER_TASK") is { } perTask)
        {
            options = options with { FramesPerTask = Number("IDRAK_MP3_FRAMES_PER_TASK", perTask) };
        }

        return options;
    }

    private static string? Read(string name) => Environment.GetEnvironmentVariable(name) is { Length: > 0 } value ? value.Trim() : null;

    private static int Number(string name, string value) => int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int n)
        ? n
        : throw new ArgumentException($"{name} is a whole number, not '{value}'.");
}
