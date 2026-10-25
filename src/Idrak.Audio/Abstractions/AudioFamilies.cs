// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;

namespace Idrak.Audio.Abstractions;

/// <summary>What an audio family reads a checkpoint's audio part from (<see cref="IAudioFamily.Read"/>), the audio twin of core's <c>VisionCheckpoint</c>.</summary>
/// <param name="Architecture">The architecture name the checkpoint was matched by.</param>
/// <param name="Config">The model's config.json.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while reading (do not keep it).</param>
/// <param name="Folder">The model folder (where <c>preprocessor_config.json</c> is), or null.</param>
/// <param name="Open">Opens the checkpoint again, for building the encoder later (the caller disposes what it returns).</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record AudioCheckpoint(string Architecture, JsonObject Config, ITensorStore Tensors, string? Folder, Func<ITensorStore> Open, List<string> Notes);

/// <summary>How <see cref="PretrainedAudio.CreateEncoder"/> builds an encoder.</summary>
public sealed record AudioEncoderOptions
{
    /// <summary>Where it runs (<see cref="Device.Default"/> when null; pass the language model's device).</summary>
    public Device? Device { get; init; }

    /// <summary>How the encoder holds its weights: as the checkpoint stores them (the default), float32, or bfloat16.</summary>
    public EncoderWeights Weights { get; init; }

    /// <summary>
    /// Trained tensors by their checkpoint names and in its layout (a training run's kept encoder and projector), read in
    /// place of the checkpoint's; none when null.
    /// </summary>
    public IReadOnlyDictionary<string, (int[] Shape, float[] Values)>? Trained { get; init; }
}

/// <summary>
/// The trainable side of a family's audio part (implemented by its <see cref="IPretrainedAudio"/> when it trains, as a
/// from-scratch family's does): which parts it trains, their parameters in its encoder, the features with gradients and
/// the trained tensors in the checkpoint's names. <c>ChatAudioParts</c> uses it to be a trainable soft-part source.
/// </summary>
public interface IAudioTuningPart
{
    /// <summary>The parts it can train, by the names a training stage uses ("audio-encoder", "audio-projector").</summary>
    IReadOnlyList<string> TrainableParts { get; }

    /// <summary>Makes the parameters of <paramref name="parts"/> in <paramref name="encoder"/> trainable and every other one frozen; returns them.</summary>
    /// <exception cref="ArgumentException">A part is not one it offers, or the encoder is not this family's.</exception>
    IReadOnlyList<Tensor> Train(IAudioLanguageEncoder encoder, IReadOnlyCollection<string> parts);

    /// <summary>Training or evaluation mode of the encoder's layers (dropout).</summary>
    void SetTraining(IAudioLanguageEncoder encoder, bool training);

    /// <summary>
    /// <see cref="IAudioLanguageEncoder.Encode"/> recording gradients to the trained parameters; <paramref name="checkpointing"/>
    /// keeps only each block's output and recomputes the rest in the backward pass.
    /// </summary>
    IReadOnlyList<ImageFeatures> EncodeTraining(IAudioLanguageEncoder encoder, IReadOnlyList<AudioData> audios, bool checkpointing);

    /// <summary>The tensors of <paramref name="parts"/> as they are now in <paramref name="encoder"/>, under the checkpoint's names and layout, on the host.</summary>
    IReadOnlyList<(string Name, int[] Shape, float[] Values)> Export(IAudioLanguageEncoder encoder, IReadOnlyCollection<string> parts);
}

/// <summary>
/// The audio side of an audio-language model: decoded sounds in, each sound's token features out, with as many soft
/// tokens per sound as its family gives it (<see cref="Blocks"/>). It does its family's own feature extraction
/// (<see cref="IPretrainedAudio.Features"/>) on the decoded samples it is given. The mirror of core's <c>IVisionEncoder</c>
/// (the plan's <c>IAudioEncoder</c>; named apart from the sound-file encoder <see cref="IAudioEncoder"/> of
/// <see cref="AudioEncoders"/>, which shares the namespace). One sound may become several blocks of soft tokens (a family
/// that writes a segment per 30 s, say); most give one.
/// </summary>
public interface IAudioLanguageEncoder : IDisposable
{
    /// <summary>The features' width: the language model's (each soft token's embedding).</summary>
    int Width { get; }

    /// <summary>Where the features are made.</summary>
    Device Device { get; }

    /// <summary>The blocks of soft tokens <paramref name="audio"/> becomes, in prompt order, each with its token count, without encoding it.</summary>
    IReadOnlyList<ImageTokenLayout> Blocks(AudioData audio);

    /// <summary>
    /// The features of <paramref name="audios"/>: one <see cref="ImageFeatures"/> per block (the caller disposes them),
    /// the sounds in order and each sound's blocks in the order <see cref="Blocks"/> gives them, each with that block's
    /// layout. Runs without recording gradients. A sound encodes the same alone and in a batch.
    /// </summary>
    IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<AudioData> audios);
}

/// <summary>
/// The audio part of an audio-language checkpoint as its family reads it (<see cref="IAudioFamily.Read"/>): everything
/// the rest of the library needs to put sounds into the prompt, and nothing about how the family builds its encoder.
/// <see cref="AudioLanguage.Attach"/> reads it beside a loaded model; <c>ChatAudioParts</c> turns it into a soft-part
/// source for generation and tuning. A family that lets parts of it train implements <see cref="IAudioTuningPart"/> too.
/// <see cref="PretrainedAudio"/> is a base with the common defaults.
/// </summary>
public interface IPretrainedAudio
{
    /// <summary>The family it was read by: its name in <see cref="AudioFamilies"/> (the architecture name).</summary>
    string Family { get; }

    /// <summary>The width of the audio features: the text decoder's (one embedding per soft token).</summary>
    int Width { get; }

    /// <summary>The checkpoint tensors (by their stored names) the audio part reads: they do not count as unused.</summary>
    IReadOnlySet<string> StoredTensors { get; }

    /// <summary>How sound becomes the encoder's features (from the checkpoint's <c>preprocessor_config.json</c>).</summary>
    IAudioFeatureExtractor Features { get; }

    /// <summary>How the rendered prompt's audio markers expand to the soft tokens, with the family's token ids (<see cref="ImageTokenFormat"/> reused, plan 21 decision 5).</summary>
    IImagePromptFormat PromptFormat { get; }

    /// <summary>How the soft tokens attend in the prompt (a rule of <see cref="ImageAttentionRules"/>, the family's choice).</summary>
    IImageAttentionRule Attention { get; }

    /// <summary>Builds the family's audio encoder from the checkpoint: decoded sounds in, each sound's features out. Dispose it when done.</summary>
    IAudioLanguageEncoder CreateEncoder(AudioEncoderOptions? options = null);

    /// <summary>One line for <c>idrak show</c> and messages: the family's features and soft tokens.</summary>
    string Describe();
}

/// <summary>
/// A base for a family's audio part (<see cref="IPretrainedAudio"/>): a family derives from it and adds what is its own.
/// </summary>
public abstract class PretrainedAudio : IPretrainedAudio
{
    /// <inheritdoc />
    public abstract string Family { get; }

    /// <inheritdoc />
    public abstract int Width { get; }

    /// <inheritdoc />
    public abstract IReadOnlySet<string> StoredTensors { get; }

    /// <inheritdoc />
    public abstract IAudioFeatureExtractor Features { get; }

    /// <inheritdoc />
    public abstract IImagePromptFormat PromptFormat { get; }

    /// <inheritdoc />
    public abstract IImageAttentionRule Attention { get; }

    /// <inheritdoc />
    public abstract IAudioLanguageEncoder CreateEncoder(AudioEncoderOptions? options = null);

    /// <inheritdoc />
    public virtual string Describe() => $"{Family}, {Features.Name} ({Features.Options.Mels} mels at {Features.Options.SampleRate} Hz), {PromptFormat.Name}, audio tokens attend: {Attention.Name}";

    /// <inheritdoc />
    public override string ToString() => Describe();
}

/// <summary>
/// An audio part made by <see cref="ICreatableAudioFamily.Create"/>: what is written into the model folder so the family's
/// <see cref="IAudioFamily.Read"/> reads it like any checkpoint.
/// </summary>
/// <param name="Config">Keys for config.json's top level (the family's audio configuration and soft token ids).</param>
/// <param name="Tensors">The part's tensors under the checkpoint's names, as the checkpoint stores them.</param>
/// <param name="Files">Other files of the model folder by name, with their text (preprocessor_config.json, for example).</param>
public sealed record CreatedAudio(JsonObject Config, IReadOnlyList<(string Name, int[] Shape, float[] Values)> Tensors, IReadOnlyDictionary<string, string> Files);

/// <summary>
/// An audio-language family's audio side: how its checkpoints' audio part is read. Registered in <see cref="AudioFamilies"/>
/// under the architecture name its checkpoints give (config.json's "architectures"), beside the text decoder registered
/// under the same name in core's <c>PretrainedArchitectures</c>.
/// </summary>
public interface IAudioFamily
{
    /// <summary>The architecture name it is registered under.</summary>
    string Name { get; }

    /// <summary>
    /// The checkpoint's audio part (configuration, tensors checked, features, prompt format, attention rule), or null when
    /// this checkpoint of the family has no audio weights (a note says so).
    /// </summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    IPretrainedAudio? Read(AudioCheckpoint checkpoint);
}

/// <summary>
/// An audio-language family that makes audio parts that did not exist, for models made from a spec; a family read from
/// checkpoints only does not implement it.
/// </summary>
public interface ICreatableAudioFamily : IAudioFamily
{
    /// <summary>
    /// An audio part that did not exist, for a model made from a spec: its checkpoint tensors drawn from
    /// <paramref name="seed"/>, the config.json keys that describe it (<see cref="IAudioFamily.Read"/> reads them back) and
    /// its other files. <paramref name="audio"/> is the family's own description of the part: the library does not read it.
    /// </summary>
    /// <param name="audio">The family's description of the part.</param>
    /// <param name="textDim">The text decoder's width (each soft token's embedding).</param>
    /// <param name="tokenizer">The model's tokenizer (the family's audio tokens are looked up in it).</param>
    /// <param name="seed">The seed the weights are drawn from.</param>
    CreatedAudio Create(JsonObject audio, int textDim, ITokenizer tokenizer, int seed);
}

/// <summary>
/// The audio-language families, by architecture name (config.json's "architectures"). The library registers none: a
/// family is an application of these contracts, registered by the app or plug-in that brings it (Idrak.Audio.Plugins'
/// IdrakAudio family is one). A checkpoint is matched to its own family by that exact
/// name; a checkpoint with an audio part (an <c>audio_config</c>) whose name has no family is refused, naming this
/// registry: nothing falls back to another family. Register one with <see cref="Register"/>.
/// </summary>
public static class AudioFamilies
{
    private static readonly SlotTable<string, IAudioFamily> Registry = new(nameof(AudioFamilies), comparer: StringComparer.Ordinal,
        unguarded: "a family's reading, encoder, prompt format and attention rule must agree with one another, so two registrations cannot be mixed");

    /// <summary>Registers <paramref name="family"/> under its <see cref="IAudioFamily.Name"/>; under a library name it takes that family's place until <see cref="Unregister"/>.</summary>
    [MethodImpl(MethodImplOptions.NoInlining)]   // the caller is the registering assembly (its Origin)
    public static void Register(IAudioFamily family)
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
    public static IAudioFamily? Find(string name) => Registry.Find(name);

    /// <summary>The family registered as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names <see cref="Register"/>.</exception>
    public static IAudioFamily Get(string name) => Registry.Find(name) ?? throw NotRegistered(name);

    /// <summary>The library's family <paramref name="name"/>, whatever an app registered over it; null when the library has none.</summary>
    public static IAudioFamily? Default(string name) => Registry.Default(name);

    /// <summary>Who registered the family <paramref name="name"/>: <see cref="Overrides.Library"/> or the app's assembly; null when none is.</summary>
    public static string? Origin(string name) => Registry.Origin(name);

    /// <summary>Whether a config.json describes an audio-language checkpoint: it has an <c>audio_config</c>.</summary>
    public static bool HasAudio(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return config["audio_config"] is JsonObject;
    }

    /// <summary>
    /// The family that reads the audio part of a checkpoint of architecture <paramref name="architecture"/> with this
    /// config.json: the one registered under that exact name, or null for a checkpoint with no audio part.
    /// </summary>
    /// <exception cref="NotSupportedException">The checkpoint has an audio part and no family is registered under its name; the message names this registry.</exception>
    public static IAudioFamily? For(string architecture, JsonObject config)
    {
        ArgumentException.ThrowIfNullOrEmpty(architecture);
        ArgumentNullException.ThrowIfNull(config);
        return Registry.Find(architecture) ?? (HasAudio(config) ? throw NotRegistered(architecture) : null);
    }

    internal static NotRegisteredException NotRegistered(string name) =>
        new(nameof(AudioFamilies), name, Registry.Keys,
            $"Audio family '{name}' is not registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "register it with AudioFamilies.Register (a plug-in or the app that brings the family; the library registers none).");
}
