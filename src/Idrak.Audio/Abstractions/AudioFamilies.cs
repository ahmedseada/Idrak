// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

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

    /// <summary>
    /// Trained tensors by their checkpoint names and in its layout (a training run's kept encoder and projector), read in
    /// place of the checkpoint's; none when null.
    /// </summary>
    public IReadOnlyDictionary<string, (int[] Shape, float[] Values)>? Trained { get; init; }
}

/// <summary>
/// The trainable side of a family's audio part (implemented by its <see cref="IPretrainedAudio"/> when it trains, as a
/// from-scratch family's does): which parts it trains. The training itself is the encoder's
/// (<see cref="ITrainableAudioEncoder"/>: the encoders the family builds implement it, so no part casts an encoder to its
/// own type). <c>ChatAudioParts</c> uses both to be a trainable soft-part source.
/// </summary>
public interface IAudioTuningPart
{
    /// <summary>The parts it can train, by the names a training stage uses ("audio-encoder", "audio-projector").</summary>
    IReadOnlyList<string> TrainableParts { get; }
}

/// <summary>
/// The training side of an encoder a trainable audio family builds (<see cref="IAudioTuningPart"/>): the parameters of
/// its parts, the features with gradients and the trained tensors in the checkpoint's names. Each member works on this
/// encoder's own modules, so a source takes the encoder as this interface once, when it is given the encoder, and calls
/// it every step without a check.
/// </summary>
public interface ITrainableAudioEncoder : IAudioLanguageEncoder
{
    /// <summary>The audio part that built this encoder (its <see cref="IPretrainedAudio.CreateEncoder"/>).</summary>
    IPretrainedAudio Audio { get; }

    /// <summary>Makes the parameters of <paramref name="parts"/> trainable and every other one frozen; returns them.</summary>
    /// <exception cref="ArgumentException">A part is not one the family offers.</exception>
    IReadOnlyList<Tensor> Train(IReadOnlyCollection<string> parts);

    /// <summary>Training or evaluation mode of the encoder's layers (dropout).</summary>
    void SetTraining(bool training);

    /// <summary>
    /// <see cref="IAudioLanguageEncoder.Encode"/> recording gradients to the trained parameters; <paramref name="checkpointing"/>
    /// keeps only each block's output and recomputes the rest in the backward pass.
    /// </summary>
    IReadOnlyList<ImageFeatures> EncodeTraining(IReadOnlyList<AudioData> audios, bool checkpointing);

    /// <summary>The tensors of <paramref name="parts"/> as they are now, under the checkpoint's names and layout, on the host.</summary>
    IReadOnlyList<(string Name, int[] Shape, float[] Values)> Export(IReadOnlyCollection<string> parts);
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
/// An audio-language family's audio side: how its checkpoints' audio part is read, under the architecture name its
/// checkpoints give (config.json's "architectures").
/// </summary>
public interface IAudioFamily
{
    /// <summary>The architecture name its checkpoints give.</summary>
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
/// What the audio-language families share: whether a checkpoint has an audio part. The library has no family: a family is
/// an application of these contracts, brought by an app or a plug-in (Idrak.Audio.Plugins' IdrakAudio family is one).
/// The families were registered by architecture name and a checkpoint's found by its name, which is not on the new design yet.
/// </summary>
public static class AudioFamilies
{
    /// <summary>Whether a config.json describes an audio-language checkpoint: it has an <c>audio_config</c>.</summary>
    public static bool HasAudio(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return config["audio_config"] is JsonObject;
    }
}
