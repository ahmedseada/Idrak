// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
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

/// <summary>A voice converter family: how its checkpoints are recognised and read. Registered in <see cref="VoiceConverters"/>.</summary>
public interface IVoiceConverterFamily
{
    /// <summary>The name it is registered under (stored in voice.json).</summary>
    string Name { get; }

    /// <summary>Whether <paramref name="config"/> (a config.json naming no architecture) is one of its checkpoints.</summary>
    bool Recognizes(JsonObject config);

    /// <summary>Reads the checkpoint into a converter on the options' device.</summary>
    /// <exception cref="InvalidDataException">A tensor or setting is missing or disagrees with the configuration.</exception>
    IVoiceConverter Read(VoiceConverterCheckpoint checkpoint, VoiceConverterLoadOptions options);
}

/// <summary>
/// The voice converter families, by name (ordinal). The library registers none: a family is an application of these
/// contracts, registered by the plug-in that brings it (<c>samples/OpenVoice</c> is one). A checkpoint no family reads is
/// refused naming this registry; nothing falls back to another family.
/// </summary>
public static class VoiceConverters
{
    private static readonly SlotTable<string, IVoiceConverterFamily> Registry = new(nameof(VoiceConverters), comparer: StringComparer.Ordinal,
        unguarded: "a converter's embeddings are its own, so two registrations cannot be mixed");

    /// <summary>Registers <paramref name="family"/> under its <see cref="IVoiceConverterFamily.Name"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IVoiceConverterFamily family)
    {
        ArgumentNullException.ThrowIfNull(family);
        ArgumentException.ThrowIfNullOrEmpty(family.Name);
        Registry.Register(family.Name, family, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's family <paramref name="name"/>; false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The family registered as <paramref name="name"/>, or null.</summary>
    public static IVoiceConverterFamily? Find(string name) => Registry.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names <see cref="Register"/>.</exception>
    public static IVoiceConverterFamily Get(string name) => Registry.Find(name) ?? throw NotRegistered(name);

    /// <summary>The library's family <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IVoiceConverterFamily? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the family <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>
    /// Loads the converter at <paramref name="path"/> (a folder with config.json and a checkpoint a registered
    /// <see cref="CheckpointFormats"/> format opens) through the options' family, the one config.json names, or the
    /// first registered that recognises config.json.
    /// </summary>
    /// <exception cref="FileNotFoundException">There is no config.json.</exception>
    /// <exception cref="NotSupportedException">No registered family reads it; the message names those registered.</exception>
    public static IVoiceConverter Load(string path, VoiceConverterLoadOptions? options = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        options ??= new VoiceConverterLoadOptions();
        var format = CheckpointFormats.For(path);
        string folder = format.Prepare(path);
        string configPath = Path.Combine(folder, "config.json");
        if (!File.Exists(configPath))
        {
            throw new FileNotFoundException($"{folder} has no config.json.", configPath);
        }

        var config = JsonNode.Parse(File.ReadAllText(configPath), documentOptions: new JsonDocumentOptions { CommentHandling = JsonCommentHandling.Skip })?.AsObject()
            ?? throw new InvalidDataException($"{configPath} is not a JSON object.");
        var family = FamilyFor(config, options.Family, folder);
        using var tensors = format.Open(folder);
        return family.Read(new VoiceConverterCheckpoint(config, tensors, Path.GetFullPath(folder), []), options);
    }

    /// <summary>The family that reads <paramref name="config"/>: <paramref name="name"/> when given, the one config.json names, else the first that recognises it.</summary>
    /// <exception cref="NotSupportedException">None does.</exception>
    public static IVoiceConverterFamily FamilyFor(JsonObject config, string? name = null, string? where = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        if (name is not null)
        {
            return Get(name);
        }

        string? named = config["architectures"] is JsonArray { Count: > 0 } a && a[0] is JsonValue v && v.TryGetValue(out string? s) ? s
            : config["model_type"] is JsonValue m && m.TryGetValue(out string? t) ? t : null;
        if (named is not null && Registry.Find(named) is { } family)
        {
            return family;
        }

        foreach (var candidate in Registry.Values)
        {
            if (candidate.Recognizes(config))
            {
                return candidate;
            }
        }

        throw NotRegistered(named ?? $"for {where ?? "this config.json"}");
    }

    internal static NotSupportedException NotRegistered(string name) =>
        new($"Voice converter family '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "register it with VoiceConverters.Register (a plug-in or the app that brings the family; the library registers none).");
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

/// <summary>
/// Where voices are installed: each a folder holding voice.json (the manifest: kind, languages, encoder, embedding
/// width, clips, consent) and, for an embedding voice, embedding.safetensors. Registered in <see cref="VoiceStores"/>.
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
/// The voice stores, by name (ignoring case). Built in (Idrak.Audio): "folder" (<see cref="FolderVoiceStore"/> at
/// <see cref="FolderVoiceStore.DefaultRoot"/>: <c>IDRAK_VOICES</c>, else voices/ under <c>IDRAK_CACHE</c> or
/// <c>~/.cache/idrak</c>). Register another (a shared or remote one) with <see cref="Register"/>.
/// </summary>
public static class VoiceStores
{
    /// <summary>The built-in store's name, "folder".</summary>
    public const string FolderName = "folder";

    private static readonly SlotTable<string, IVoiceStore> Registry = BuiltIn();

    private static SlotTable<string, IVoiceStore> BuiltIn()
    {
        var table = new SlotTable<string, IVoiceStore>(nameof(VoiceStores), comparer: StringComparer.OrdinalIgnoreCase,
            unguarded: "a store's voices live where it keeps them, so two stores cannot stand in for each other");
        table.RegisterDefault(FolderName, new FolderVoiceStore(FolderVoiceStore.DefaultRoot, FolderName));
        return table;
    }

    /// <summary>Registers <paramref name="store"/> under its <see cref="IVoiceStore.Name"/>; under "folder" it takes the built-in's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IVoiceStore store)
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentException.ThrowIfNullOrWhiteSpace(store.Name);
        Registry.Register(store.Name, store, System.Reflection.Assembly.GetCallingAssembly());
    }

    /// <summary>Removes the app's store <paramref name="name"/> (the built-in name gets the built-in back); false when the app registered none.</summary>
    public static bool Unregister(string name) => Registry.Unregister(name);

    /// <summary>The registered names.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The store <paramref name="name"/> (null: "folder").</summary>
    /// <exception cref="NotSupportedException">None is registered under the name.</exception>
    public static IVoiceStore Get(string? name = null) => Registry.Find(name ?? FolderName)
        ?? throw new NotSupportedException($"Voice store '{name}' is not registered (registered: {string.Join(", ", Registry.Keys)}); add it with VoiceStores.Register.");

    /// <summary>The library's store <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IVoiceStore? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the store <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);
}
