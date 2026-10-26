// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Audio.Abstractions;

namespace Idrak.Maui.Abstractions;

/// <summary>How a sound is played (<see cref="ISoundPlayer.PlayAsync(IAsyncEnumerable{AudioData}, SoundPlayRequest?, CancellationToken)"/>).</summary>
public sealed record SoundPlayRequest
{
    /// <summary>What is played, for the page ("original", "processed", "speech").</summary>
    public string Label { get; init; } = "sound";

    /// <summary>The playback (an <see cref="AudioPlaybacks"/> name); null: <see cref="AudioPlaybacks.ForThisMachine"/> (aaudio on Android).</summary>
    public string? Playback { get; init; }

    /// <summary>How the output is opened (device, rate, block, buffer, performance); the library's defaults when not set.</summary>
    public AudioPlaybackOptions Output { get; init; } = new();
}

/// <summary>What one play did.</summary>
/// <param name="Label">What was played (<see cref="SoundPlayRequest.Label"/>).</param>
/// <param name="Output">The output it played on, as the playback describes it.</param>
/// <param name="SampleRate">The rate the device played at.</param>
/// <param name="Channels">The channels the device played.</param>
/// <param name="Blocks">The blocks written.</param>
/// <param name="Seconds">The seconds of sound written.</param>
/// <param name="FirstSound">From the call to the first block queued on the output (when the device can start playing it); null when none was.</param>
/// <param name="Elapsed">From the call to its end.</param>
/// <param name="Completed">Whether everything was played to its end; false when it was stopped or another sound took the speaker.</param>
public sealed record SoundPlayback(string Label, string Output, int SampleRate, int Channels, int Blocks, double Seconds, TimeSpan? FirstSound, TimeSpan Elapsed,
    bool Completed);

/// <summary>
/// Plays sound on the phone's speaker through the library's <see cref="AudioPlaybacks"/> (<see cref="IAudioOutput"/>): blocks
/// written in order as they come, held to the sound's pace by the output's back-pressure, the next block asked for while
/// the one before is written (so a block that is made, such as a sentence of speech, is made while the last one plays),
/// drained at the end. One sound at a time: a new play stops the one before; <see cref="Stop"/> stops it at once.
/// </summary>
public interface ISoundPlayer
{
    /// <summary>Whether a sound is playing.</summary>
    bool IsPlaying { get; }

    /// <summary>The label of what plays (<see cref="SoundPlayRequest.Label"/>), or null.</summary>
    string? Playing { get; }

    /// <summary>
    /// Plays <paramref name="blocks"/> to their end: the output opened (while the first block is made), each block written
    /// as it comes, the output drained. Completes when the sound has played, or at once when it is stopped (<see cref="Stop"/>,
    /// <paramref name="cancellationToken"/>, or another play), the output then closed and its sound cut.
    /// </summary>
    /// <exception cref="NotSupportedException">No playback can play here, or the one asked for cannot (its reason given).</exception>
    /// <exception cref="InvalidOperationException">The device could not be opened or failed (the system's message given).</exception>
    /// <exception cref="ArgumentException">The output options are out of range.</exception>
    Task<SoundPlayback> PlayAsync(IAsyncEnumerable<AudioData> blocks, SoundPlayRequest? request = null, CancellationToken cancellationToken = default);

    /// <summary>Plays one recording (the output takes it in its own chunks, so a long one is held to its pace like blocks).</summary>
    /// <inheritdoc cref="PlayAsync(IAsyncEnumerable{AudioData}, SoundPlayRequest?, CancellationToken)"/>
    Task<SoundPlayback> PlayAsync(AudioData audio, SoundPlayRequest? request = null, CancellationToken cancellationToken = default);

    /// <summary>Stops what plays at once; false when nothing did.</summary>
    bool Stop();
}
