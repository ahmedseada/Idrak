// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio.Abstractions;

/// <summary>How a capture is opened (<see cref="IAudioCapture.Open"/>).</summary>
public sealed record AudioCaptureOptions
{
    /// <summary>The input device: an id <see cref="IAudioCapture.Devices"/> lists, or null for the system's default.</summary>
    public string? Device { get; init; }

    /// <summary>The sample rate asked for (the system converts where the device runs at another).</summary>
    public int SampleRate { get; init; } = 16000;

    /// <summary>The channels asked for.</summary>
    public int Channels { get; init; } = 1;

    /// <summary>The seconds of sound each block holds: the delay a block adds before it is read.</summary>
    public double BlockSeconds { get; init; } = 0.05;

    /// <summary>What the stream is tuned for where the system lets it choose (AAudio).</summary>
    public AudioPerformance Performance { get; init; } = AudioPerformance.Default;

    /// <summary>Asks for the device alone (AAudio's exclusive sharing mode, a shorter path); the system may still share it.</summary>
    public bool Exclusive { get; init; }

    /// <summary>Throws when an option is out of range.</summary>
    /// <exception cref="ArgumentException">The rate, the channels or the block is not positive.</exception>
    public void Check()
    {
        if (SampleRate < 1 || Channels < 1 || !(BlockSeconds > 0) || BlockSeconds * SampleRate < 1)
        {
            throw new ArgumentException($"Audio capture: SampleRate ({SampleRate}), Channels ({Channels}) and BlockSeconds ({BlockSeconds}) are positive, a block at least one sample.");
        }
    }
}

/// <summary>An input device a capture can record from.</summary>
/// <param name="Id">What <see cref="AudioCaptureOptions.Device"/> takes to choose it.</param>
/// <param name="Name">Its name as the system gives it.</param>
/// <param name="IsDefault">Whether it is the one recorded when no device is named.</param>
public sealed record AudioInputDevice(string Id, string Name, bool IsDefault);

/// <summary>
/// A source of live sound (a microphone through the operating system, a stream as it arrives): blocks of
/// <see cref="AudioData"/> in the order they were heard, until the source ends or the reading is cancelled. Disposing it
/// stops the recording.
/// </summary>
public interface IAudioInput : IDisposable
{
    /// <summary>What it records from, for messages ("microphone 0: USB Audio", "standard input").</summary>
    string Description { get; }

    /// <summary>The blocks' sample rate.</summary>
    int SampleRate { get; }

    /// <summary>The blocks' channels.</summary>
    int Channels { get; }

    /// <summary>
    /// The sound as it comes, block after block; ends when the source does, or quietly when
    /// <paramref name="cancellationToken"/> is cancelled (what was heard until then is given first). Read it once.
    /// </summary>
    IAsyncEnumerable<AudioData> ReadAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// A way to record live sound: an operating system's audio interface. It
/// loads what it needs at run time and says why it cannot run where it cannot.
/// </summary>
public interface IAudioCapture
{
    /// <summary>Its name ("winmm", "alsa", "aaudio").</summary>
    string Name { get; }

    /// <summary>Whether it can record on this machine; <paramref name="reason"/> says why not (another operating system, the audio library missing).</summary>
    bool IsAvailable(out string reason);

    /// <summary>The input devices it can record from (empty where it cannot run).</summary>
    IReadOnlyList<AudioInputDevice> Devices();

    /// <summary>Starts recording.</summary>
    /// <exception cref="NotSupportedException">It cannot run on this machine (the reason given).</exception>
    /// <exception cref="InvalidOperationException">The device could not be opened (the system's message given).</exception>
    IAudioInput Open(AudioCaptureOptions options);
}

/// <summary>
/// The audio captures' names (ignoring case). Built in (Idrak.Audio), each loading the operating system's audio library
/// when first used: "winmm" (Windows' waveIn), "alsa" (Linux's libasound) and "aaudio" (Android's libaaudio), each
/// reached directly (<see cref="Idrak.Audio.WinmmCapture"/>, <see cref="Idrak.Audio.AlsaCapture"/>,
/// <see cref="Idrak.Audio.AAudioCapture"/>); <see cref="ForThisMachine"/> picks the first that can run.
/// </summary>
public static class AudioCaptures
{
    /// <summary>Windows' waveIn (winmm.dll).</summary>
    public const string Winmm = "winmm";

    /// <summary>Linux's ALSA (libasound.so.2).</summary>
    public const string Alsa = "alsa";

    /// <summary>Android's AAudio (libaaudio.so, Android 8.0, API 26, and later).</summary>
    public const string Aaudio = "aaudio";

    /// <summary>The first capture that can run on this machine.</summary>
    /// <exception cref="NotSupportedException">
    /// None can; the message gives each one's reason. Always for now: picking among the captures is not on the new design
    /// yet.
    /// </exception>
    public static IAudioCapture ForThisMachine()
    {
        var reasons = new List<string>();
        foreach (var capture in Registered())
        {
            if (capture.IsAvailable(out string reason))
            {
                return capture;
            }

            reasons.Add($"{capture.Name}: {reason}");
        }

        throw new NotSupportedException($"No audio capture can record on this machine ({string.Join("; ", reasons)}); "
            + "or pipe the sound in from a recorder.");
    }

    // Picking the first capture that can run is not on the new design yet.
    private static IEnumerable<IAudioCapture> Registered() =>
        throw new NotSupportedException("The audio captures to pick from were registered, which is not on the new design yet.");
}
