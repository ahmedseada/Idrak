// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Generation;

/// <summary>
/// A source of soft tokens for one kind of message part (<see cref="ChatPart.Kind"/>): how a part of that kind becomes a
/// run of soft tokens in a decoder's prompt. Everything the image path has, with no image in it: the chat template writes
/// the family's marker once per part, <see cref="Format"/> expands each marker to its blocks of soft tokens
/// (<see cref="IImagePromptFormat.Expand"/>: begin, N soft tokens, end), <see cref="Blocks"/> says how many tokens a
/// part takes (without encoding it), <see cref="Encode"/> gives each block's features ([tokens, width], the decoder's
/// width, on its device), whose rows replace the soft tokens' embeddings in the prefill, and <see cref="Attention"/> says
/// how the soft rows attend. The audio part (Idrak.Audio's <c>ChatAudioParts</c>) is the first source; a chat generator
/// and the tuning encoder read any number of sources (<c>ChatGenerator.Parts</c>, <c>ChatTranscriptEncoder.Parts</c>),
/// one per kind. The soft-token types keep their image names for now (plan 21, decision 5: <see cref="ImageTokenLayout"/>,
/// <see cref="ImageFeatures"/>, <see cref="ImageTokenFormat"/>): the prefill replaces runs of a token id by features and
/// knows nothing of what the part is.
/// </summary>
public interface ISoftPartSource
{
    /// <summary>The kind of part it reads (<see cref="ChatParts.Audio"/>, say), as registered in <see cref="ChatParts"/>.</summary>
    string Kind { get; }

    /// <summary>The width of the features: the decoder's (one embedding per soft token).</summary>
    int Width { get; }

    /// <summary>How the rendered prompt's markers of this kind expand to the soft tokens (the family's token ids).</summary>
    IImagePromptFormat Format { get; }

    /// <summary>How the soft tokens attend in the prompt (<see cref="ImageAttentionRules"/>: causal unless the family says otherwise).</summary>
    IImageAttentionRule Attention { get; }

    /// <summary>
    /// The blocks of soft tokens <paramref name="part"/> (one of <see cref="Kind"/>) takes, in prompt order, each with its
    /// token count, without encoding it: one block for most families.
    /// </summary>
    /// <exception cref="ArgumentException">The part is not of this kind, or cannot be read.</exception>
    IReadOnlyList<ImageTokenLayout> Blocks(ChatPart part);

    /// <summary>
    /// The features of <paramref name="parts"/> (each of <see cref="Kind"/>): one <see cref="ImageFeatures"/> per block
    /// (the caller disposes them), the parts in order and each part's blocks in the order <see cref="Blocks"/> gives
    /// them, with that block's layout. Runs without recording gradients; a part encodes the same alone and in a batch.
    /// </summary>
    IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ChatPart> parts);
}

/// <summary>
/// A soft-part source whose layers can be trained (an audio encoder and its projector, made from scratch): training reads
/// its features with gradients (<see cref="EncodeTraining"/>) and trains the parts it offers (<see cref="TrainableParts"/>,
/// the names a pretraining stage's <c>Trained</c> list gives, "audio-encoder" and "audio-projector" for the audio part).
/// The model-part registry (<see cref="ModelParts"/>) opens one beside a loaded model; the training step lays its
/// features out where the soft tokens are, as it does images' features. Dispose it when done.
/// </summary>
public interface ITrainableSoftParts : ISoftPartSource, IDisposable
{
    /// <summary>The parts it can train, by the names a training stage uses.</summary>
    IReadOnlyList<string> TrainableParts { get; }

    /// <summary>
    /// Makes the parameters of <paramref name="parts"/> (of <see cref="TrainableParts"/>) trainable and every other one
    /// frozen, and returns them for the optimizer (none for an empty list: the source runs as a frozen one).
    /// </summary>
    /// <exception cref="ArgumentException">A part is not one it offers; the message names those it does.</exception>
    IReadOnlyList<Tensor> Train(IReadOnlyCollection<string> parts);

    /// <summary>
    /// The parts the last <see cref="Train"/> made trainable (none before it, or after an empty list): what a save
    /// exports (<see cref="Export"/>) beside a run's adapters.
    /// </summary>
    IReadOnlyCollection<string> Trained => [];

    /// <summary>The parameters <see cref="Train"/> made trainable last (none until it is called, or after an empty list).</summary>
    IReadOnlyList<Tensor> Parameters { get; }

    /// <summary>Training or evaluation mode of its layers (dropout).</summary>
    void SetTraining(bool training);

    /// <summary>
    /// <see cref="ISoftPartSource.Encode"/> recording gradients to the trained parameters (<see cref="Train"/>);
    /// <paramref name="checkpointing"/> keeps only each block's output and recomputes the rest in the backward pass.
    /// </summary>
    IReadOnlyList<ImageFeatures> EncodeTraining(IReadOnlyList<ChatPart> parts, bool checkpointing);

    /// <summary>The tensors of <paramref name="parts"/> as they are now, under the checkpoint's names and in its layout, copied to the host.</summary>
    IReadOnlyList<(string Name, int[] Shape, float[] Values)> Export(IReadOnlyCollection<string> parts);
}
