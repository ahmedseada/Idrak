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
