// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;

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
/// A way to record live sound: an operating system's audio interface (registered in <see cref="AudioCaptures"/>). It
/// loads what it needs at run time and says why it cannot run where it cannot.
/// </summary>
public interface IAudioCapture
{
    /// <summary>The name it is registered under ("winmm", "alsa").</summary>
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
/// The audio captures, by name (ignoring case). Built in (Idrak.Audio), each loading the operating system's audio library
/// when first used: "winmm" (Windows' waveIn) and "alsa" (Linux's libasound). Register another (macOS, a sound server, a
/// network stream) with <see cref="Register"/>; <see cref="ForThisMachine"/> picks the first that can run, an app's before
/// the library's.
/// </summary>
public static class AudioCaptures
{
    /// <summary>Windows' waveIn (winmm.dll).</summary>
    public const string Winmm = "winmm";

    /// <summary>Linux's ALSA (libasound.so.2).</summary>
    public const string Alsa = "alsa";

    private static readonly SlotTable<string, IAudioCapture> Table = BuiltIn();

    private static SlotTable<string, IAudioCapture> BuiltIn()
    {
        var table = new SlotTable<string, IAudioCapture>(nameof(AudioCaptures), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a capture records from its own system's devices, so another cannot stand in for it");
        table.RegisterDefault(Winmm, new Idrak.Audio.WinmmCapture());
        table.RegisterDefault(Alsa, new Idrak.Audio.AlsaCapture());
        return table;
    }

    /// <summary>Registers <paramref name="capture"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioCapture capture)
    {
        ArgumentNullException.ThrowIfNull(capture);
        ArgumentException.ThrowIfNullOrWhiteSpace(capture.Name);
        Table.Register(capture.Name, capture, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's capture of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The capture of that name, or null.</summary>
    public static IAudioCapture? Find(string name) => Table.Find(name);

    /// <summary>The capture of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IAudioCapture Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(AudioCaptures), name, Table.Keys,
            $"Audio capture '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with AudioCaptures.Register.");

    /// <summary>The library's capture of that name, or null.</summary>
    public static IAudioCapture? Default(string name) => Table.Default(name);

    /// <summary>Who registered the capture of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>The first registered capture that can run on this machine, an app's before the library's.</summary>
    /// <exception cref="NotSupportedException">None can; the message gives each one's reason.</exception>
    public static IAudioCapture ForThisMachine()
    {
        var reasons = new List<string>();
        foreach (string name in Table.Keys.OrderBy(n => Origin(n) == Overrides.Library ? 1 : 0))
        {
            var capture = Table.Find(name)!;
            if (capture.IsAvailable(out string reason))
            {
                return capture;
            }

            reasons.Add($"{name}: {reason}");
        }

        throw new NotSupportedException($"No audio capture can record on this machine ({string.Join("; ", reasons)}). Register one with AudioCaptures.Register, "
            + "or pipe the sound in from a recorder.");
    }
}
