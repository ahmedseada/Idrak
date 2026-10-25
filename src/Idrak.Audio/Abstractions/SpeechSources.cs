// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// Transcribed speech to train on (<c>SpeechTrainer</c>): files of one kind, read by the <see cref="SpeechSources">source
/// kind</see> named <see cref="Kind"/>. Kept in a run's checkpoints, so a resumed run opens the same data.
/// </summary>
/// <param name="Kind">A <see cref="SpeechSources"/> name: "transcripts" (JSONL rows of audio and text) or "audio-folder" (name.wav beside name.txt).</param>
/// <param name="Paths">The files or folders (a .zip stands for a folder).</param>
public sealed record SpeechSource(string Kind, IReadOnlyList<string> Paths)
{
    /// <summary>
    /// Transcripts: the folder or .zip the audio paths are relative to, or null for beside each transcript file (inside
    /// its zip for a transcript in a zip).
    /// </summary>
    public string? Audio { get; init; }

    /// <summary>Utterances shorter than this (seconds) are left out, and counted.</summary>
    public double MinSeconds { get; init; }

    /// <summary>Utterances longer than this (seconds) are left out, and counted; 0 for no limit.</summary>
    public double MaxSeconds { get; init; }

    /// <summary>Leave out (and count) rows whose audio file is missing, instead of refusing the source.</summary>
    public bool SkipMissing { get; init; }

    /// <summary>The source as JSON (<see cref="FromJson"/> reads it back).</summary>
    public JsonObject ToJson()
    {
        var json = new JsonObject { ["kind"] = Kind, ["paths"] = new JsonArray([.. Paths.Select(p => (JsonNode)p)]) };
        if (Audio is not null)
        {
            json["audio"] = Audio;
        }

        if (MinSeconds > 0)
        {
            json["min_seconds"] = MinSeconds;
        }

        if (MaxSeconds > 0)
        {
            json["max_seconds"] = MaxSeconds;
        }

        if (SkipMissing)
        {
            json["skip_missing"] = true;
        }

        return json;
    }

    /// <summary>A source from its JSON (<see cref="ToJson"/>); "path" for a single file is read too.</summary>
    /// <exception cref="InvalidDataException">A key is missing or wrong.</exception>
    public static SpeechSource FromJson(JsonObject json)
    {
        ArgumentNullException.ThrowIfNull(json);
        string kind = (string?)json["kind"] ?? throw new InvalidDataException("A speech source has no \"kind\" (transcripts or audio-folder).");
        var paths = json["paths"] is JsonArray many ? many.Select(p => (string?)p ?? throw new InvalidDataException("A speech source path is not a string.")).ToList()
            : json["path"] is JsonValue one ? [(string)one!] : throw new InvalidDataException($"The {kind} speech source names no \"paths\".");
        return new SpeechSource(kind, paths)
        {
            Audio = (string?)json["audio"],
            MinSeconds = (double?)json["min_seconds"] ?? 0,
            MaxSeconds = (double?)json["max_seconds"] ?? 0,
            SkipMissing = (bool?)json["skip_missing"] ?? false,
        };
    }
}

/// <summary>One transcribed utterance of an opened source.</summary>
/// <param name="Name">Where it comes from, for messages (the transcript file and line, or the audio file).</param>
/// <param name="Audio">The audio file's path, or its entry in a zip as "archive.zip!entry".</param>
/// <param name="Text">What is said.</param>
/// <param name="Seconds">Its length, from the transcript row or the file's header.</param>
public sealed record SpeechUtterance(string Name, string Audio, string Text, double Seconds);

/// <summary>
/// An opened speech source: <see cref="Count"/> utterances, read by index in any order and from several threads at once
/// (the trainer reads them ahead on worker threads).
/// </summary>
public interface ISpeechReader : IDisposable
{
    /// <summary>The utterances kept.</summary>
    int Count { get; }

    /// <summary>What was left out and why (missing files, lengths outside the source's limits), one line each.</summary>
    IReadOnlyList<string> Notes { get; }

    /// <summary>The utterance at <paramref name="index"/>.</summary>
    SpeechUtterance this[int index] { get; }

    /// <summary>The audio file of utterance <paramref name="index"/> as stored (decode it with <see cref="AudioCodecs.Decode(ReadOnlySpan{byte})"/>).</summary>
    byte[] ReadBytes(int index);
}

/// <summary>A kind of speech data: how its files become utterances. Register one with <see cref="SpeechSources.Register"/>.</summary>
public interface ISpeechSource
{
    /// <summary>The kind's name.</summary>
    string Name { get; }

    /// <summary>One line on what it reads.</summary>
    string Summary { get; }

    /// <summary>Opens <paramref name="source"/>'s files.</summary>
    /// <exception cref="FileNotFoundException">A file is missing (and the source does not skip missing files).</exception>
    /// <exception cref="InvalidDataException">A file cannot be read as this kind.</exception>
    ISpeechReader Open(SpeechSource source);
}

/// <summary>
/// The kinds of speech data, by name (ignoring case). Built in (Idrak.Audio): "transcripts" (JSONL rows of
/// <c>{"audio": path, "text": ...}</c>, as <c>idrak data audio</c>'s manifest.jsonl and <c>idrak transcribe -o</c> write
/// them; paths relative to the transcript file, inside its zip, or to <see cref="SpeechSource.Audio"/>) and
/// "audio-folder" (every sound file of a folder or zip with its transcript beside it, <c>name.wav</c> and <c>name.txt</c>).
/// </summary>
public static class SpeechSources
{
    /// <summary>JSONL transcripts.</summary>
    public const string Transcripts = "transcripts";

    /// <summary>A folder of sound files with their texts beside them.</summary>
    public const string AudioFolder = "audio-folder";

    private static readonly SlotTable<string, ISpeechSource> Table = BuiltIn();

    private static SlotTable<string, ISpeechSource> BuiltIn()
    {
        var table = new SlotTable<string, ISpeechSource>(nameof(SpeechSources), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a source kind reads its own files into utterances, so another kind cannot read the same ones");
        table.RegisterDefault(Transcripts, new TranscriptSpeechSource());
        table.RegisterDefault(AudioFolder, new FolderSpeechSource());
        return table;
    }

    /// <summary>Registers <paramref name="source"/> under its name (over the library's of that name, which stays behind it).</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISpeechSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        ArgumentException.ThrowIfNullOrWhiteSpace(source.Name);
        Table.Register(source.Name, source, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's kind of that name (the library's, if any, answers again); false when the app registered none.</summary>
    public static bool Unregister(string name) => Table.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Table.Keys;

    /// <summary>The kind of that name, or null.</summary>
    public static ISpeechSource? Find(string name) => Table.Find(name);

    /// <summary>The kind of that name.</summary>
    /// <exception cref="NotSupportedException">None is registered under that name.</exception>
    public static ISpeechSource Get(string name) => Table.Find(name)
        ?? throw new NotRegisteredException(nameof(SpeechSources), name, Table.Keys,
            $"Speech source '{name}' is not registered (registered: {string.Join(", ", Table.Keys)}); add it with SpeechSources.Register.");

    /// <summary>The library's kind of that name, or null.</summary>
    public static ISpeechSource? Default(string name) => Table.Default(name);

    /// <summary>Who registered the kind of that name: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Table.Origin(name);

    /// <summary>Each kind as "name: summary", one a line.</summary>
    public static string Describe() => string.Join("\n", Table.Values.Select(s => $"{s.Name}: {s.Summary}"));

    /// <summary>Opens <paramref name="source"/> through the kind it names.</summary>
    public static ISpeechReader Open(SpeechSource source)
    {
        ArgumentNullException.ThrowIfNull(source);
        return Get(source.Kind).Open(source);
    }

    /// <summary>
    /// The order of <paramref name="count"/> utterances in pass <paramref name="epoch"/> of a run seeded with
    /// <paramref name="seed"/>: a Fisher-Yates shuffle from the two, the same on every run and on resume, so a run's
    /// position is its pass and how far into the pass it is.
    /// </summary>
    public static int[] Order(int count, int seed, int epoch)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(count);
        var order = new int[count];
        for (int i = 0; i < count; i++)
        {
            order[i] = i;
        }

        var random = new Random(unchecked(seed * 1_000_003 + epoch * 10_007 + 17));
        for (int i = count - 1; i > 0; i--)
        {
            int j = random.Next(i + 1);
            (order[i], order[j]) = (order[j], order[i]);
        }

        return order;
    }
}
