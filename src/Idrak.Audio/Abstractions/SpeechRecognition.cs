// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>
/// How a sound is transcribed (<see cref="ISpeechRecognizer.Transcribe"/>). Every value is a setting; what a family
/// cannot do with one (a task it has no token for, a language it was not trained on) it refuses with a message naming it.
/// </summary>
public sealed record TranscriptionOptions
{
    /// <summary>The language's code ("en", "ar"), or null: detected by the family when it can.</summary>
    public string? Language { get; init; }

    /// <summary>"transcribe" (the spoken language) or "translate" (into the family's target language); a family may refuse one.</summary>
    public string Task { get; init; } = "transcribe";

    /// <summary>Ask the model for timestamps: the segments then carry the times the model gave.</summary>
    public bool Timestamps { get; init; }

    /// <summary>
    /// Apply the family's rules over timestamp tokens while decoding with <see cref="Timestamps"/> (a timestamp opens
    /// and closes each phrase, times never go back, the first timestamp near the start); off, the model's raw choices.
    /// </summary>
    public bool TimestampRules { get; init; } = true;

    /// <summary>Beams kept while decoding (1: greedy).</summary>
    public int Beams { get; init; } = 1;

    /// <summary>The length penalty of a beam search (see <c>BeamOptions.LengthPenalty</c>).</summary>
    public float LengthPenalty { get; init; } = 1f;

    /// <summary>New tokens at most per segment; 0: the family's longest (its positions, less the prompt).</summary>
    public int MaxTokens { get; init; }

    /// <summary>New tokens at least per segment (the end token is not chosen before); 0: none.</summary>
    public int MinTokens { get; init; }

    /// <summary>Seconds of sound decoded at a time; 0: the family's segment length (its features' padding).</summary>
    public double ChunkSeconds { get; init; }

    /// <summary>Seconds each chunk overlaps the one before when <see cref="ChunkSeconds"/> is set.</summary>
    public double OverlapSeconds { get; init; } = 1;

    /// <summary>Give the previous chunk's text to the model as the prompt of the next (where the family has a prompt token).</summary>
    public bool CarryText { get; init; } = true;

    /// <summary>Segments encoded together at most; 0: measured on the device.</summary>
    public int MaxBatch { get; init; }

    /// <summary>The share of the device's free memory a batch of segments may take.</summary>
    public double MemoryFraction { get; init; } = 0.5;

    /// <summary>Sampling temperature; 0: greedy or beam search. Above 0, <see cref="TemperatureFallback"/> gives the ladder tried after it.</summary>
    public float Temperature { get; init; }

    /// <summary>The temperatures tried in turn when a segment's decoding fails the thresholds below (empty: none).</summary>
    public IReadOnlyList<float> TemperatureFallback { get; init; } = [];

    /// <summary>A segment whose text compresses better than this (a repetition loop) is decoded again at the next temperature.</summary>
    public float CompressionRatioThreshold { get; init; } = 2.4f;

    /// <summary>A segment whose mean log-probability is below this is decoded again at the next temperature.</summary>
    public float LogProbThreshold { get; init; } = -1f;

    /// <summary>
    /// A segment the model takes for silence is left out: its no-speech probability (where the family has one, such as
    /// Whisper's &lt;|nospeech|&gt; after the start token) above this while its mean log-probability is under
    /// <see cref="LogProbThreshold"/> (OpenAI Whisper's rule; what it would write there is often invented text). Such a
    /// segment is not decoded again either. Null: no check (the default here; OpenAI's is 0.6).
    /// </summary>
    public float? NoSpeechThreshold { get; init; }

    /// <summary>The samples drawn at a temperature above 0, the likeliest kept (OpenAI Whisper's best_of).</summary>
    public int BestOf { get; init; } = 5;

    /// <summary>The seed of the sampling at temperatures above 0 (the same sound, options and seed give the same text).</summary>
    public int Seed { get; init; }

    /// <summary>Text given to the model before the first segment (names, spellings, style), where the family has a prompt token.</summary>
    public string? Prompt { get; init; }

    /// <summary>Run the encoder in bfloat16 where the device has it (float32 otherwise).</summary>
    public bool BFloat16 { get; init; } = true;

    /// <summary>Apply the family's lists of tokens never chosen (its suppress_tokens); off, the model's raw choices.</summary>
    public bool SuppressTokens { get; init; } = true;
}

/// <summary>One stretch of a transcription.</summary>
/// <param name="Start">Where it starts in the sound, seconds.</param>
/// <param name="End">Where it ends, seconds.</param>
/// <param name="Text">What was said.</param>
/// <param name="AverageLogProb">The mean log-probability of its tokens (how sure the model was).</param>
/// <param name="Tokens">Its tokens as the model gave them (timestamp and special tokens included).</param>
public sealed record TranscriptSegment(double Start, double End, string Text, float AverageLogProb, IReadOnlyList<int> Tokens);

/// <summary>What <see cref="ISpeechRecognizer.Transcribe"/> gives.</summary>
/// <param name="Text">The whole text, the segments' joined.</param>
/// <param name="Language">The language the sound was taken as (given or detected), or null when the family has none.</param>
/// <param name="Segments">The stretches, in order.</param>
/// <param name="Stats">Where the time went.</param>
public sealed record Transcription(string Text, string? Language, IReadOnlyList<TranscriptSegment> Segments, TranscriptionStats Stats);

/// <summary>Where a transcription's time went.</summary>
/// <param name="AudioDuration">The sound's length.</param>
/// <param name="Decode">Decoding the sound's bytes and resampling.</param>
/// <param name="Features">The features.</param>
/// <param name="Encoder">The encoder, every segment.</param>
/// <param name="Decoder">The decoder, every segment.</param>
/// <param name="PeakDeviceBytes">The device's peak memory during the call (0 where the device does not report it).</param>
/// <param name="Segments">Segments encoded.</param>
/// <param name="Batch">Segments encoded together.</param>
public sealed record TranscriptionStats(TimeSpan AudioDuration, TimeSpan Decode, TimeSpan Features, TimeSpan Encoder, TimeSpan Decoder, long PeakDeviceBytes, int Segments, int Batch)
{
    /// <summary>The whole call.</summary>
    public TimeSpan Total => Decode + Features + Encoder + Decoder;
}

/// <summary>Progress of a transcription (<see cref="ISpeechRecognizer.Transcribe"/>'s progress).</summary>
/// <param name="Segment">Segments done.</param>
/// <param name="Segments">Segments in all.</param>
/// <param name="Stage">What is running ("features", "encoder", "decoder").</param>
/// <param name="Elapsed">Time since the call began.</param>
public sealed record TranscriptionProgress(int Segment, int Segments, string Stage, TimeSpan Elapsed);

/// <summary>A speech recognizer: sound in, text out. Made by a family (<see cref="ISpeechFamily.Read"/>) or loaded with <c>SpeechModels.Load</c>.</summary>
public interface ISpeechRecognizer : IDisposable
{
    /// <summary>The device the model runs on.</summary>
    Device Device { get; }

    /// <summary>The language codes the model knows (empty when it has no notion of language).</summary>
    IReadOnlySet<string> Languages { get; }

    /// <summary>Transcribes <paramref name="audio"/> (any rate or channels: mixed to one and resampled to the model's).</summary>
    Transcription Transcribe(AudioData audio, TranscriptionOptions? options = null, IProgress<TranscriptionProgress>? progress = null, CancellationToken cancellationToken = default);

    /// <summary>
    /// Transcribes a stream of sound as it comes: chunks are gathered to <see cref="TranscriptionOptions.ChunkSeconds"/>
    /// (or the model's segment length) and each gives its segments, timed from the stream's start.
    /// </summary>
    IAsyncEnumerable<TranscriptSegment> TranscribeStream(IAsyncEnumerable<AudioData> chunks, TranscriptionOptions? options = null, CancellationToken cancellationToken = default);
}

/// <summary>How a speech model is loaded (<c>SpeechModels.Load</c>).</summary>
public sealed record SpeechModelOptions
{
    /// <summary>The device the model lives on; null for the default.</summary>
    public Device? Device { get; init; }

    /// <summary>How the projection weights are held (as stored, float32, bfloat16).</summary>
    public EncoderWeights Weights { get; init; } = EncoderWeights.AsStored;

    /// <summary>The decoder's key/value cache format.</summary>
    public KeyValueFormat Cache { get; init; } = KeyValueFormat.Float32;

    /// <summary>The architecture to read the checkpoint as, when config.json names none or another.</summary>
    public string? Architecture { get; init; }

    /// <summary>
    /// The language the model reads, for a family whose prompt names one (Whisper's language token): the language of the
    /// labels the speech trainer teaches it, and its transcription default when <see cref="TranscriptionOptions.Language"/>
    /// is not given. Null: the checkpoint's own (Whisper: generation_config.json's "language"), else none (detected when
    /// transcribing; a multilingual Whisper refuses to make labels without one).
    /// </summary>
    public string? Language { get; init; }

    /// <summary>
    /// The task the model is trained for, for a family whose prompt names one (Whisper: "transcribe" or "translate"). Null:
    /// the checkpoint's own (generation_config.json's "task"), else "transcribe".
    /// </summary>
    public string? Task { get; init; }
}

/// <summary>What a speech family reads a checkpoint from (<see cref="ISpeechFamily.Read"/>).</summary>
/// <param name="Architecture">The architecture name the checkpoint was matched by.</param>
/// <param name="Config">The model's config.json.</param>
/// <param name="Path">The path given to load.</param>
/// <param name="Folder">The model folder (where tokenizer.json, preprocessor_config.json and generation_config.json are), or null.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while reading (do not keep it).</param>
/// <param name="Options">How the model is loaded.</param>
/// <param name="Notes">Append anything approximated (shown with the loaded model).</param>
public sealed record SpeechCheckpoint(string Architecture, JsonObject Config, string Path, string? Folder, ITensorStore Tensors, SpeechModelOptions Options, List<string> Notes);

/// <summary>
/// A speech model made by <see cref="ISpeechFamily.Create"/>: what is written into the model folder so the family's
/// <see cref="ISpeechFamily.Read"/> reads it like any checkpoint.
/// </summary>
/// <param name="Config">The config.json.</param>
/// <param name="Tensors">The tensors under the checkpoint's names.</param>
/// <param name="Files">Other files of the model folder by name, with their text.</param>
public sealed record CreatedSpeechModel(JsonObject Config, IReadOnlyList<(string Name, int[] Shape, float[] Values)> Tensors, IReadOnlyDictionary<string, string> Files);

/// <summary>
/// A speech recognition family: how its checkpoints are read into a recognizer. Registered in <see cref="SpeechFamilies"/>
/// under the architecture name its checkpoints give (config.json's "architectures"). The library registers none: a family
/// is a plug-in or the app's (Idrak.Audio.Plugins' Whisper family is one).
/// </summary>
public interface ISpeechFamily
{
    /// <summary>The architecture name it is registered under.</summary>
    string Name { get; }

    /// <summary>The recognizer read from <paramref name="checkpoint"/>.</summary>
    /// <exception cref="InvalidDataException">A tensor or file is missing, or a shape disagrees with the configuration.</exception>
    ISpeechRecognizer Read(SpeechCheckpoint checkpoint);

    /// <summary>
    /// A model that did not exist, from the family's own description of it (<paramref name="spec"/>): its tensors drawn from
    /// <paramref name="seed"/>, its config.json and other files. A family read from checkpoints only keeps this default,
    /// which refuses.
    /// </summary>
    CreatedSpeechModel Create(JsonObject spec, ITokenizer tokenizer, int seed) =>
        throw new NotSupportedException($"Speech family '{Name}' cannot make a model from a description (it reads checkpoints only).");
}

/// <summary>
/// The speech recognition families, by architecture name (config.json's "architectures"). The library registers none: a
/// checkpoint whose name has no family is refused, naming this registry; nothing falls back to another family, since
/// families are not interchangeable. Register one with <see cref="Register"/>.
/// </summary>
public static class SpeechFamilies
{
    private static readonly SlotTable<string, ISpeechFamily> Registry = new(nameof(SpeechFamilies), comparer: StringComparer.Ordinal,
        unguarded: "a family's reading, features, prompt and decoding must agree with one another, so two registrations cannot be mixed");

    /// <summary>Registers <paramref name="family"/> under its <see cref="ISpeechFamily.Name"/>; under a library name it takes that family's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(ISpeechFamily family)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentException.ThrowIfNullOrEmpty(family.Name);
        Registry.Register(family.Name, family, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family <paramref name="name"/> (a library name gets the library's back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered architecture names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The family registered as <paramref name="name"/>, or null.</summary>
    public static ISpeechFamily? Find(string name) => Registry.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names <see cref="Register"/>.</exception>
    public static ISpeechFamily Get(string name) => Registry.Find(name) ?? throw NotRegistered(name);

    /// <summary>The library's family <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static ISpeechFamily? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the family <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>The family that reads a checkpoint of architecture <paramref name="architecture"/> with this config.json: the one registered under that exact name.</summary>
    /// <exception cref="NotSupportedException">None is; the message names this registry.</exception>
    public static ISpeechFamily For(string architecture, JsonObject config)
    {
        ArgumentException.ThrowIfNullOrEmpty(architecture);
        ArgumentNullException.ThrowIfNull(config);
        return Registry.Find(architecture) ?? throw NotRegistered(architecture);
    }

    internal static NotSupportedException NotRegistered(string name) =>
        new($"Speech family '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "add it with SpeechFamilies.Register (a plug-in or the app that brings the family; the library registers none), "
            + "or reference Idrak.Audio.Plugins and call AudioPlugins.RegisterAll().");
}
