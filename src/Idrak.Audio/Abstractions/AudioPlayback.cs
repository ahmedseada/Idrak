// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// What a stream is tuned for where the system lets it choose (AAudio's performance mode); the other systems take their
/// own default.
/// </summary>
public enum AudioPerformance
{
    /// <summary>The system's default (AAudio: AAUDIO_PERFORMANCE_MODE_NONE).</summary>
    Default,

    /// <summary>Less power, a longer delay (AAudio: AAUDIO_PERFORMANCE_MODE_POWER_SAVING).</summary>
    PowerSaving,

    /// <summary>The shortest delay the device gives (AAudio: AAUDIO_PERFORMANCE_MODE_LOW_LATENCY).</summary>
    LowLatency,
}

/// <summary>How a playback is opened (<see cref="IAudioPlayback.Open"/>).</summary>
public sealed record AudioPlaybackOptions
{
    /// <summary>
    /// The rate a playback opens at when <see cref="SampleRate"/> is not set and the system reports no rate of the
    /// device's own (waveOut, ALSA).
    /// </summary>
    public const int FallbackSampleRate = 48000;

    /// <summary>The channels a playback opens with when <see cref="Channels"/> is not set and the system reports none of the device's own.</summary>
    public const int FallbackChannels = 2;

    /// <summary>The output device: an id <see cref="IAudioPlayback.Devices"/> lists, or null for the system's default.</summary>
    public string? Device { get; init; }

    /// <summary>
    /// The sample rate asked for; null for the device's own where the system reports it (AAudio), else
    /// <see cref="FallbackSampleRate"/>. Sound written at another rate is resampled to the output's (<see cref="Resampler"/>).
    /// </summary>
    public int? SampleRate { get; init; }

    /// <summary>The channels asked for; null for the device's own where the system reports them (AAudio), else <see cref="FallbackChannels"/>.</summary>
    public int? Channels { get; init; }

    /// <summary>The seconds of sound handed to the device at a time (its device buffer holds four of them).</summary>
    public double BlockSeconds { get; init; } = 0.02;

    /// <summary>
    /// The seconds of sound the library holds ahead of the device: a write waits (back-pressure) while this much is
    /// queued and not yet handed on. The delay from a write to the speaker is about this plus the device's buffer.
    /// </summary>
    public double BufferSeconds { get; init; } = 0.5;

    /// <summary>What the stream is tuned for where the system lets it choose (AAudio).</summary>
    public AudioPerformance Performance { get; init; } = AudioPerformance.Default;

    /// <summary>Asks for the device alone (AAudio's exclusive sharing mode, a shorter path); the system may still share it.</summary>
    public bool Exclusive { get; init; }

    /// <summary>How sound at another rate than the output's is resampled (the library's defaults when null).</summary>
    public ResampleOptions? Resample { get; init; }

    /// <summary>Throws when an option is out of range.</summary>
    /// <exception cref="ArgumentException">The rate, the channels, the block or the buffer is not positive, or the buffer holds less than a block.</exception>
    public void Check()
    {
        int rate = SampleRate ?? FallbackSampleRate;
        if (rate < 1 || Channels is < 1 || !(BlockSeconds > 0) || BlockSeconds * rate < 1 || !(BufferSeconds >= BlockSeconds))
        {
            throw new ArgumentException($"Audio playback: SampleRate ({SampleRate}), Channels ({Channels}), BlockSeconds ({BlockSeconds}) and BufferSeconds ({BufferSeconds}) "
                + "are positive, a block at least one sample, the buffer at least a block.");
        }
    }
}

/// <summary>An output device a playback can play on.</summary>
/// <param name="Id">What <see cref="AudioPlaybackOptions.Device"/> takes to choose it.</param>
/// <param name="Name">Its name as the system gives it.</param>
/// <param name="IsDefault">Whether it is the one played on when no device is named.</param>
public sealed record AudioOutputDevice(string Id, string Name, bool IsDefault);

/// <summary>
/// A place sound is played (a speaker through the operating system): blocks of <see cref="AudioData"/> written in order,
/// each converted once to the output's rate and channels (<see cref="Resampler"/> for the rate, the channels' mean or
/// copies for the channels), played one after another without a gap while the writes keep up. Disposing it stops the
/// sound at once; <see cref="DrainAsync"/> first lets everything written play.
/// </summary>
public interface IAudioOutput : IDisposable
{
    /// <summary>What it plays on, for messages ("speaker (default, alsa)").</summary>
    string Description { get; }

    /// <summary>The rate the device plays at (what the system gave, which may differ from the rate asked for).</summary>
    int SampleRate { get; }

    /// <summary>The channels the device plays.</summary>
    int Channels { get; }

    /// <summary>
    /// Queues <paramref name="block"/> after what was written before. Completes once the block is queued; waits while the
    /// output's buffer (<see cref="AudioPlaybackOptions.BufferSeconds"/>) is full, so a writer faster than the sound is
    /// held to its pace. One write at a time: await each before the next. Cancelling it keeps what was queued already.
    /// Sound at another rate is resampled as one stream across blocks of the same rate, so the last few milliseconds of
    /// it wait for the next block of that rate or for <see cref="DrainAsync"/>.
    /// </summary>
    /// <exception cref="InvalidOperationException">A write is already running, or the device failed (the system's message given).</exception>
    /// <exception cref="ObjectDisposedException">The output was disposed.</exception>
    ValueTask WriteAsync(AudioData block, CancellationToken cancellationToken = default);

    /// <summary>
    /// Completes once everything written before it has been played by the device. Cancelling it stops the waiting, not
    /// the sound. Writing again afterwards plays on.
    /// </summary>
    /// <exception cref="InvalidOperationException">A write is running, or the device failed (the system's message given).</exception>
    /// <exception cref="ObjectDisposedException">The output was disposed.</exception>
    Task DrainAsync(CancellationToken cancellationToken = default);
}

/// <summary>
/// A way to play sound: an operating system's audio interface (registered in <see cref="AudioPlaybacks"/>). It loads what
/// it needs at run time and says why it cannot run where it cannot.
/// </summary>
public interface IAudioPlayback
{
    /// <summary>The name it is registered under ("winmm", "alsa", "aaudio").</summary>
    string Name { get; }

    /// <summary>Whether it can play on this machine; <paramref name="reason"/> says why not (another operating system, the audio library missing).</summary>
    bool IsAvailable(out string reason);

    /// <summary>The output devices it can play on (empty where it cannot run).</summary>
    IReadOnlyList<AudioOutputDevice> Devices();

    /// <summary>Opens the device for playing.</summary>
    /// <exception cref="NotSupportedException">It cannot run on this machine (the reason given).</exception>
    /// <exception cref="InvalidOperationException">The device could not be opened (the system's message given).</exception>
    IAudioOutput Open(AudioPlaybackOptions options);
}

/// <summary>
/// The audio playbacks, by name (ignoring case). Built in (Idrak.Audio), each loading the operating system's audio library
/// when first used: "winmm" (Windows' waveOut), "alsa" (Linux's libasound) and "aaudio" (Android's libaaudio). Register
/// another (macOS, a sound server, a network stream) with <see cref="Register"/>; <see cref="ForThisMachine"/> picks the
/// first that can run, an app's before the library's.
/// </summary>
public static class AudioPlaybacks
{
    /// <summary>Windows' waveOut (winmm.dll).</summary>
    public const string Winmm = "winmm";

    /// <summary>Linux's ALSA (libasound.so.2).</summary>
    public const string Alsa = "alsa";

    /// <summary>Android's AAudio (libaaudio.so, Android 8.0, API 26, and later).</summary>
    public const string Aaudio = "aaudio";

    private static readonly SlotTable<string, IAudioPlayback> Table = BuiltIn();

    private static SlotTable<string, IAudioPlayback> BuiltIn()
    {
        var table = new SlotTable<string, IAudioPlayback>(nameof(AudioPlaybacks), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a playback plays on its own system's devices, so another cannot stand in for it");
        table.RegisterDefault(Winmm, new Idrak.Audio.WinmmPlayback());
        table.RegisterDefault(Alsa, new Idrak.Audio.AlsaPlayback());
        table.RegisterDefault(Aaudio, new Idrak.Audio.AAudioPlayback());
        return table;
    }

    /// <summary>Registers <paramref name="playback"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioPlayback playback)
    {
        ArgumentNullException.ThrowIfNull(playback);
        ArgumentException.ThrowIfNullOrWhiteSpace(playback.Name);
        Table.Register(playback.Name, playback, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's playback of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The playback of that name, or null.</summary>
    public static IAudioPlayback? Find(string name) => Table.Find(name);

    /// <summary>The playback of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IAudioPlayback Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(AudioPlaybacks), name, Table.Keys,
            $"Audio playback '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with AudioPlaybacks.Register.");

    /// <summary>The library's playback of that name, or null.</summary>
    public static IAudioPlayback? Default(string name) => Table.Default(name);

    /// <summary>Who registered the playback of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>The first registered playback that can run on this machine, an app's before the library's.</summary>
    /// <exception cref="NotSupportedException">None can; the message gives each one's reason.</exception>
    public static IAudioPlayback ForThisMachine()
    {
        var reasons = new List<string>();
        foreach (string name in Table.Keys.OrderBy(n => Origin(n) == Overrides.Library ? 1 : 0))
        {
            var playback = Table.Find(name)!;
            if (playback.IsAvailable(out string reason))
            {
                return playback;
            }

            reasons.Add($"{name}: {reason}");
        }

        throw new NotSupportedException($"No audio playback can play on this machine ({string.Join("; ", reasons)}). Register one with AudioPlaybacks.Register, "
            + "or write the sound to a file (AudioEncoders).");
    }
}
