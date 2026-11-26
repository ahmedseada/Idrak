// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Formats;

namespace Idrak.Audio.Abstractions;

/// <summary>The kinds of installed voice (<see cref="VoiceProfile.Kind"/>, plan 21, S9).</summary>
public static class VoiceKinds
{
    /// <summary>A name for a speaker a multi-speaker checkpoint holds (route A): voice.json's "speaker" is the checkpoint's voice.</summary>
    public const string Speaker = "speaker";

    /// <summary>A speaker embedding from a few clips of a person (route B): the synthesizer speaks, a voice converter changes the timbre to it.</summary>
    public const string Embedding = "embedding";

    /// <summary>A synthesizer fine-tuned on a person's recordings (route C): voice.json's "checkpoint" is the model folder, relative to the voice's.</summary>
    public const string Checkpoint = "checkpoint";
}

/// <summary>Who agreed to their voice being installed, to what, and when: stored in voice.json, shown with the voice. The library records it and cannot verify it.</summary>
/// <param name="Speaker">The person whose voice it is.</param>
/// <param name="Statement">What they agreed to, in their words.</param>
/// <param name="Agreed">When.</param>
public sealed record VoiceConsent(string Speaker, string Statement, DateTimeOffset Agreed);

/// <summary>One reference clip a voice was made from (voice.json's "clips").</summary>
/// <param name="File">The clip's file name (the path it was read from, without folders).</param>
/// <param name="Sha256">The SHA-256 of the file's bytes (lower-case hex).</param>
/// <param name="Seconds">Its length after trimming, in seconds.</param>
/// <param name="Agreement">The cosine of its own embedding with the voice's (1 for a single clip): a clip far below the others is another speaker or noise.</param>
public sealed record VoiceClip(string File, string Sha256, double Seconds, float Agreement);

/// <summary>
/// An installed voice: its name, kind (<see cref="VoiceKinds"/>), languages, the folder it lives in, and its voice.json
/// as read. The typed properties read the manifest; anything a store or family adds stays in <see cref="Manifest"/>.
/// </summary>
public sealed record VoiceProfile(string Name, string Kind, IReadOnlyList<string> Languages, string Folder, JsonObject Manifest)
{
    /// <summary>The speaker encoder (and converter) family the embedding came from (voice.json's "encoder.family"), or null.</summary>
    public string? Encoder => Manifest["encoder"]?["family"]?.GetValue<string>();

    /// <summary>The converter model folder the embedding was made with (voice.json's "encoder.folder"), or null.</summary>
    public string? EncoderFolder => Manifest["encoder"]?["folder"]?.GetValue<string>();

    /// <summary>The embedding's width (voice.json's "embedding.size"); 0 for a voice without one.</summary>
    public int EmbeddingSize => Manifest["embedding"]?["size"]?.GetValue<int>() ?? 0;

    /// <summary>The consent the voice was installed with, or null when voice.json holds none.</summary>
    public VoiceConsent? Consent => Manifest["consent"] is JsonObject c
        && c["speaker"]?.GetValue<string>() is { } speaker && c["statement"]?.GetValue<string>() is { } statement
        ? new VoiceConsent(speaker, statement, c["agreed"]?.GetValue<DateTimeOffset>() ?? default)
        : null;

    /// <summary>A checkpoint voice's model folder (voice.json's "checkpoint", relative to the voice's folder; "model" when it names none), or null for another kind.</summary>
    public string? CheckpointFolder => Kind == VoiceKinds.Checkpoint ? Path.Combine(Folder, Manifest["checkpoint"]?.GetValue<string>() ?? "model") : null;

    /// <summary>The model a checkpoint voice was trained from (voice.json's "base"), or null.</summary>
    public string? Base => Manifest["base"]?.GetValue<string>();

    /// <summary>The clips the voice was made from (empty for a voice made otherwise).</summary>
    public IReadOnlyList<VoiceClip> Clips => Manifest["clips"] is JsonArray clips
        ? [.. clips.OfType<JsonObject>().Select(c => new VoiceClip(c["file"]?.GetValue<string>() ?? "", c["sha256"]?.GetValue<string>() ?? "",
            c["seconds"]?.GetValue<double>() ?? 0, (float)(c["agreement"]?.GetValue<double>() ?? 1)))]
        : [];
}

/// <summary>
/// Gives a speaker embedding of a person's clips: what a voice converter (or a cloning synthesizer) is conditioned on.
/// The embedding of several clips is the voice's (a family's own combination, the mean of the clips' for OpenVoice).
/// </summary>
public interface ISpeakerEncoder : IDisposable
{
    /// <summary>The family name it is registered under in <see cref="VoiceConverters"/> (stored in voice.json, so a voice is used with the encoder that made it).</summary>
    string Name { get; }

    /// <summary>The rate it reads sound at (clips at another rate are resampled).</summary>
    int SampleRate { get; }

    /// <summary>The embedding's width.</summary>
    int EmbeddingSize { get; }

    /// <summary>The device it runs on.</summary>
    Device Device { get; }

    /// <summary>The model folder it was loaded from (stored in voice.json), or null.</summary>
    string? Folder { get; }

    /// <summary>The embedding of <paramref name="clips"/> (at any rate, any channels: mixed to one): <see cref="EmbeddingSize"/> values.</summary>
    float[] Embed(IReadOnlyList<AudioData> clips, CancellationToken cancellationToken = default);
}

/// <summary>How speech is converted to a target voice (every value a setting).</summary>
public sealed record VoiceConversionOptions
{
    /// <summary>The scale of the noise drawn around the source's latents (OpenVoice's tau): 0 gives a deterministic output, higher a looser copy of the source's detail.</summary>
    public float Tau { get; init; } = 0.3f;

    /// <summary>The seed the noise is drawn from.</summary>
    public int Seed { get; init; }

    /// <summary>The speech's own speaker embedding when known (a synthesizer's voice embedded once); null: embedded from the speech itself.</summary>
    public float[]? Source { get; init; }

    /// <summary>Speech longer than this is converted in pieces cut at the quietest point before each limit (memory stays bounded); 0: in one pass.</summary>
    public double MaxSeconds { get; init; } = 30;

    /// <summary>The seconds of sound the converter's decoder makes in one piece, with its context (the same samples as one pass); 0: each piece in one pass; null: the converter's own setting.</summary>
    public double? DecoderSeconds { get; init; }
}

/// <summary>
/// Changes the timbre of speech to a target speaker's, keeping what is said and how: works on sound, not text, so on
/// any language and any synthesizer's output. Its own <see cref="ISpeakerEncoder.Embed"/> gives the embeddings it takes.
/// </summary>
public interface IVoiceConverter : ISpeakerEncoder
{
    /// <summary>
    /// <paramref name="speech"/> (any rate, mixed to one channel) in the voice of <paramref name="target"/> (an
    /// embedding of <see cref="ISpeakerEncoder.Embed"/>): one channel at <see cref="ISpeakerEncoder.SampleRate"/>.
    /// </summary>
    /// <exception cref="ArgumentException">The target's width is not <see cref="ISpeakerEncoder.EmbeddingSize"/>.</exception>
    AudioData Convert(AudioData speech, ReadOnlySpan<float> target, VoiceConversionOptions? options = null, CancellationToken cancellationToken = default);
}

/// <summary>What a converter family reads a checkpoint from (<see cref="IVoiceConverterFamily.Read"/>).</summary>
/// <param name="Config">The model's config.json.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while reading (do not keep it).</param>
/// <param name="Folder">The model folder.</param>
/// <param name="Notes">Append anything approximated.</param>
public sealed record VoiceConverterCheckpoint(JsonObject Config, ITensorStore Tensors, string Folder, List<string> Notes);

/// <summary>How a voice converter is loaded.</summary>
public sealed record VoiceConverterLoadOptions
{
    /// <summary>The device it runs on; null: <see cref="Device.Default"/>.</summary>
    public Device? Device { get; init; }

    /// <summary>The family to read the folder with; null: the one config.json names ("architectures" or "model_type"), else the first registered that recognises it.</summary>
    public string? Family { get; init; }
}

/// <summary>A voice converter family: how its checkpoints are recognised and read.</summary>
public interface IVoiceConverterFamily
{
    /// <summary>Its name (stored in voice.json).</summary>
    string Name { get; }

    /// <summary>Whether <paramref name="config"/> (a config.json naming no architecture) is one of its checkpoints.</summary>
    bool Recognizes(JsonObject config);

    /// <summary>Reads the checkpoint into a converter on the options' device.</summary>
    /// <exception cref="InvalidDataException">A tensor or setting is missing or disagrees with the configuration.</exception>
    IVoiceConverter Read(VoiceConverterCheckpoint checkpoint, VoiceConverterLoadOptions options);
}

/// <summary>
/// Loads voice converters. The library has no converter family: a family is an application of these contracts, brought
/// by a plug-in (Idrak.Audio.Plugins' OpenVoice family is one); the families were registered and a checkpoint's found
/// among them, which is not on the new design yet.
/// </summary>
public static class VoiceConverters
{
    // The converter families were registered (VoiceConverters.Register) and a checkpoint's found by its name or by
    // recognising its config.json, which is not on the new design yet.
    private static IVoiceConverterFamily ConverterFamilyFor(JsonObject config, string? name) =>
        throw new NotSupportedException($"The voice converter family {(name is null ? "that reads this config.json" : $"'{name}'")}, found among the registered ones, is not on the new design yet.");

    // The checkpoint formats were registered (CheckpointFormats) and the one of a path found among them, which is not on the
    // new design yet.
    private static ICheckpointFormat FormatFor(string path) =>
        throw new NotSupportedException($"{path}: a checkpoint format found by the path is not on the new design yet.");

    /// <summary>
    /// Loads the converter at <paramref name="path"/> (a folder with config.json and a checkpoint a registered
    /// <c>CheckpointFormats</c> format opens) through the options' family, the one config.json names, or the
    /// first registered that recognises config.json.
    /// </summary>
    /// <exception cref="FileNotFoundException">There is no config.json.</exception>
    /// <exception cref="NotSupportedException">No registered family reads it; the message names those registered.</exception>
    public static IVoiceConverter Load(string path, VoiceConverterLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        options ??= new VoiceConverterLoadOptions();
        var format = FormatFor(path);
        string folder = format.Prepare(path);
        string configPath = Path.Combine(folder, "config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"{folder} has no config.json.", configPath);
        }

        var config = JsonNode.Parse(File.ReadAllText(configPath), documentOptions: new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip })?.AsObject()
            ?? throw new InvalidDataException($"{configPath} is not a JSON object.");
        var family = ConverterFamilyFor(config, options.Family);
        using var tensors = format.Open(folder);
        return family.Read(new VoiceConverterCheckpoint(config, tensors, Path.GetFullPath(folder), []), options);
    }
}

/// <summary>How a voice is installed from clips (<see cref="IVoiceStore.Add"/>).</summary>
public sealed record VoiceAddOptions
{
    /// <summary>The speaker's consent; required: a voice without one is refused.</summary>
    public VoiceConsent? Consent { get; init; }

    /// <summary>The languages the voice is meant for (ISO codes, informational: conversion works on any language).</summary>
    public IReadOnlyList<string> Languages { get; init; } = [];

    /// <summary>Copy the clips into the voice's clips/ folder (they hold the person's recordings, so off unless asked).</summary>
    public bool KeepClips { get; init; }

    /// <summary>Cut the silence before and after each clip before embedding it.</summary>
    public bool TrimSilence { get; init; } = true;

    /// <summary>The level, in dB below a clip's peak, under which 10 ms windows at its ends count as silence.</summary>
    public float SilenceDecibels { get; init; } = 40f;

    /// <summary>A clip shorter than this after trimming is refused.</summary>
    public double MinClipSeconds { get; init; } = 1;

    /// <summary>A clip whose agreement (<see cref="VoiceClip.Agreement"/>) is below this is refused, naming it; 0: none is refused (the agreements are still recorded).</summary>
    public float MinAgreement { get; init; }

    /// <summary>Replace a voice of the same name.</summary>
    public bool Overwrite { get; init; }
}

/// <summary>How a trained model folder is installed as a voice of kind <see cref="VoiceKinds.Checkpoint"/> (<see cref="ICheckpointVoiceStore.AddCheckpoint"/>).</summary>
public sealed record VoiceCheckpointOptions
{
    /// <summary>The speaker's consent; required, as for a voice from clips: a model trained on a person's recordings speaks as them.</summary>
    public VoiceConsent? Consent { get; init; }

    /// <summary>The languages the voice speaks (ISO codes; a fine-tuned synthesizer speaks its base model's).</summary>
    public IReadOnlyList<string> Languages { get; init; } = [];

    /// <summary>The model it was trained from (a folder or a hub id), recorded as voice.json's "base".</summary>
    public string? Base { get; init; }

    /// <summary>A speaker of the model to speak with (a multi-speaker checkpoint's), recorded as voice.json's "voice"; null: its own voice.</summary>
    public string? Voice { get; init; }

    /// <summary>What the training did (steps, utterances, seconds, the last losses), recorded as voice.json's "training".</summary>
    public JsonObject? Training { get; init; }

    /// <summary>Files of the model folder not copied (the trainer's state, the discriminators: what only training reads).</summary>
    public IReadOnlyList<string> Exclude { get; init; } = [];

    /// <summary>Replace a voice of the same name.</summary>
    public bool Overwrite { get; init; }
}

/// <summary>
/// Where voices are installed: each a folder holding voice.json (the manifest: kind, languages, encoder, embedding
/// width, clips, consent) and, for an embedding voice, embedding.safetensors.
/// </summary>
public interface IVoiceStore
{
    /// <summary>The name it is registered under.</summary>
    string Name { get; }

    /// <summary>Where it keeps voices (a folder for the built-in store).</summary>
    string Location { get; }

    /// <summary>The installed voices, by name.</summary>
    IReadOnlyList<VoiceProfile> List();

    /// <summary>The voice <paramref name="name"/> (ignoring case), or null.</summary>
    VoiceProfile? Find(string name);

    /// <summary>
    /// Installs the voice <paramref name="name"/> from the sound files <paramref name="clips"/> (any format
    /// <see cref="AudioCodecs"/> decodes): trimmed, embedded by <paramref name="encoder"/>, the clips' agreement
    /// measured, voice.json and embedding.safetensors written.
    /// </summary>
    /// <exception cref="ArgumentException">The name is not a plain name, there is no consent, or a clip is too short or disagrees.</exception>
    /// <exception cref="IOException">A voice of that name exists and <see cref="VoiceAddOptions.Overwrite"/> is off.</exception>
    VoiceProfile Add(string name, IReadOnlyList<string> clips, ISpeakerEncoder encoder, VoiceAddOptions options, CancellationToken cancellationToken = default);

    /// <summary>The embedding of an embedding voice.</summary>
    /// <exception cref="InvalidDataException">The voice has no embedding, or it is not as wide as voice.json says.</exception>
    float[] ReadEmbedding(VoiceProfile voice);

    /// <summary>Installs the voice in <paramref name="folderOrZip"/> (a voice folder or a zip <see cref="Export"/> wrote).</summary>
    /// <exception cref="InvalidDataException">It holds no valid voice.json or no consent.</exception>
    VoiceProfile Import(string folderOrZip, bool overwrite = false);

    /// <summary>Writes the voice <paramref name="name"/> to the zip <paramref name="zip"/>.</summary>
    void Export(string name, string zip);

    /// <summary>Removes the voice <paramref name="name"/>; false when there is none.</summary>
    bool Remove(string name);
}

/// <summary>
/// A voice store that keeps model folders: voices of kind <see cref="VoiceKinds.Checkpoint"/> (a synthesizer fine-tuned on
/// the person's recordings); a store that keeps only embeddings does not implement it.
/// </summary>
public interface ICheckpointVoiceStore : IVoiceStore
{
    /// <summary>
    /// Installs the model folder <paramref name="modelFolder"/> (<c>SpeechSynthesisModels.Load</c> reads it) as the voice
    /// <paramref name="name"/> of kind <see cref="VoiceKinds.Checkpoint"/>: the folder copied into the voice's model/
    /// (without <see cref="VoiceCheckpointOptions.Exclude"/>), voice.json beside it.
    /// </summary>
    /// <exception cref="ArgumentException">The name is not a plain name, or there is no consent.</exception>
    /// <exception cref="FileNotFoundException">The folder holds no config.json.</exception>
    /// <exception cref="IOException">A voice of that name exists and <see cref="VoiceCheckpointOptions.Overwrite"/> is off.</exception>
    VoiceProfile AddCheckpoint(string name, string modelFolder, VoiceCheckpointOptions options);
}

/// <summary>
/// The voice stores' names (ignoring case). Built in (Idrak.Audio): "folder" (<see cref="FolderVoiceStore"/> at
/// <see cref="FolderVoiceStore.DefaultRoot"/>: <c>IDRAK_VOICES</c>, else voices/ under <c>IDRAK_CACHE</c> or
/// <c>~/.cache/idrak</c>), reached directly.
/// </summary>
public static class VoiceStores
{
    /// <summary>The built-in store's name, "folder".</summary>
    public const string FolderName = "folder";
}
