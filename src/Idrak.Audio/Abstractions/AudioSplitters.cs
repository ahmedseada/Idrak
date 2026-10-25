// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// How a recording is cut into pieces (<see cref="IAudioSplitter"/>, plan 21, S7: <c>idrak data audio --split</c>).
/// Every value is a setting; a splitter reads the ones it uses.
/// </summary>
public sealed record AudioSplitOptions
{
    /// <summary>The frames the sound's level is measured over, seconds ("silence").</summary>
    public double HopSeconds { get; init; } = 0.01;

    /// <summary>A frame whose energy is more than this many dB below the sound's loudest frame is silence ("silence").</summary>
    public float SilenceDecibels { get; init; } = 40f;

    /// <summary>A silence at least this long separates two pieces; shorter pauses stay inside a piece ("silence").</summary>
    public double MinSilenceSeconds { get; init; } = 0.3;

    /// <summary>The silence kept before and after each piece, at most half the pause to its neighbour ("silence").</summary>
    public double PadSeconds { get; init; } = 0.1;

    /// <summary>Pieces shorter than this are left out; 0: none is.</summary>
    public double MinSeconds { get; init; }

    /// <summary>
    /// The longest piece: "silence" cuts a longer one again at its quietest frame in the second half of the limit,
    /// "seconds" cuts the sound into pieces of this length (required there); 0: no limit.
    /// </summary>
    public double MaxSeconds { get; init; }

    /// <summary>The options as JSON (for a manifest or a run's record).</summary>
    public JsonObject ToJson() => new()
    {
        ["hop_seconds"] = HopSeconds, ["silence_db"] = SilenceDecibels, ["min_silence_seconds"] = MinSilenceSeconds, ["pad_seconds"] = PadSeconds,
        ["min_seconds"] = MinSeconds, ["max_seconds"] = MaxSeconds,
    };

    /// <summary>Refuses values out of range.</summary>
    /// <exception cref="ArgumentException">A value is out of range; the message names it.</exception>
    public void Check()
    {
        if (!(HopSeconds > 0) || !(SilenceDecibels > 0) || MinSilenceSeconds < 0 || PadSeconds < 0 || MinSeconds < 0 || MaxSeconds < 0
            || MaxSeconds > 0 && MinSeconds > MaxSeconds)
        {
            throw new ArgumentException(string.Create(CultureInfo.InvariantCulture,
                $"Split options: HopSeconds ({HopSeconds}) and SilenceDecibels ({SilenceDecibels}) are positive, MinSilenceSeconds ({MinSilenceSeconds}), PadSeconds ({PadSeconds}), "
                + $"MinSeconds ({MinSeconds}) and MaxSeconds ({MaxSeconds}) not negative, MinSeconds at most MaxSeconds when it is set."));
        }
    }
}

/// <summary>A piece of a sound: its first sample and its length, in samples of each channel.</summary>
/// <param name="Start">The first sample.</param>
/// <param name="Length">The samples it holds.</param>
public readonly record struct AudioPiece(int Start, int Length)
{
    /// <summary>The sample after its last.</summary>
    public int End => Start + Length;
}

/// <summary>
/// Cuts a recording into pieces (at its pauses, at fixed lengths): what <c>idrak data audio --split</c> and a data
/// preparation step use. Register one with <see cref="AudioSplitters.Register"/>.
/// </summary>
public interface IAudioSplitter
{
    /// <summary>The name it is registered under ("silence").</summary>
    string Name { get; }

    /// <summary>The pieces of <paramref name="audio"/> (any channels, any rate), in order, not overlapping; empty when nothing is kept.</summary>
    /// <exception cref="ArgumentException">An option it needs is missing or out of range.</exception>
    IReadOnlyList<AudioPiece> Split(AudioData audio, AudioSplitOptions options);
}

/// <summary>
/// The splitters, by name (ignoring case). Built in (Idrak.Audio): "silence" (the pieces between pauses: a frame's
/// energy against the loudest frame's, one pass over the samples) and "seconds" (pieces of a fixed length).
/// </summary>
public static class AudioSplitters
{
    /// <summary>The pieces between pauses.</summary>
    public const string Silence = "silence";

    /// <summary>Pieces of a fixed length.</summary>
    public const string Seconds = "seconds";

    private static readonly SlotTable<string, IAudioSplitter> Table = BuiltIn();

    private static SlotTable<string, IAudioSplitter> BuiltIn()
    {
        var table = new SlotTable<string, IAudioSplitter>(nameof(AudioSplitters), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a splitter's cuts are its own, so another cannot stand in for it and give the same pieces");
        table.RegisterDefault(Silence, new SilenceSplitter());
        table.RegisterDefault(Seconds, new SecondsSplitter());
        return table;
    }

    /// <summary>Registers <paramref name="splitter"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioSplitter splitter)
    {
        ArgumentNullException.ThrowIfNull(splitter);
        ArgumentException.ThrowIfNullOrWhiteSpace(splitter.Name);
        Table.Register(splitter.Name, splitter, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's splitter of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The splitter of that name, or null.</summary>
    public static IAudioSplitter? Find(string name) => Table.Find(name);

    /// <summary>The splitter of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static IAudioSplitter Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(AudioSplitters), name, Table.Keys,
            $"Audio splitter '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with AudioSplitters.Register.");

    /// <summary>The library's splitter of that name, or null.</summary>
    public static IAudioSplitter? Default(string name) => Table.Default(name);

    /// <summary>Who registered the splitter of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);
}
