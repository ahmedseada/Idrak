// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers.Abstractions;

namespace Idrak.Layers;

/// <summary>
/// The library's <see cref="ISequenceEncoder"/> built from layers: token ids embedded (<see cref="Words"/>), plus each
/// token's type embedded (<see cref="TokenTypes"/>, optional: type 0 for every token when the caller gives none), plus
/// positions (<see cref="Positions"/>, optional: any module adding them to [batch, time, width], such as
/// <see cref="PositionEmbedding"/>), an optional normalization of the embeddings, the blocks, and an optional final
/// normalization. A padded batch runs its blocks inside a <see cref="PaddedSequences.Use"/> scope, so every
/// <see cref="MultiHeadAttention"/> in them lets no position attend to padding: a sequence gives the same states alone
/// and in the batch. A family plug-in assembles one from its checkpoint's tensors; the encoder takes ownership of the
/// layers.
/// </summary>
public sealed class SequenceEncoder : Module, ISequenceEncoder
{
    private readonly Module[] _blocks;
    private readonly int _maxPositions;
    private PaddedSequences? _training;                                    // a recorded pass's padding, kept for its backward

    /// <summary>Creates the encoder from its layers.</summary>
    /// <param name="words">The token embedding ("words"), [vocabulary, width].</param>
    /// <param name="blocks">The blocks, each [batch, time, width] to the same shape (for example <see cref="TransformerEncoderLayer"/>s, or a family's own).</param>
    /// <param name="positions">Adds the positions to the embeddings ("positions"), or null for none (positions inside the blocks).</param>
    /// <param name="tokenTypes">The token-type embedding ("types"), [types, width], or null for an encoder without token types.</param>
    /// <param name="embeddingNorm">Normalizes the summed embeddings ("embedding_norm"), or null.</param>
    /// <param name="finalNorm">Normalizes the last block's output ("final_norm"), or null.</param>
    /// <param name="maxPositions">The longest sequence; null takes a <see cref="PositionEmbedding"/>'s length, else no limit.</param>
    /// <exception cref="ArgumentException">A token-type embedding of another width than the words', or a non-positive limit.</exception>
    public SequenceEncoder(Embedding words, IEnumerable<Module> blocks, Module? positions = null, Embedding? tokenTypes = null, Module? embeddingNorm = null,
        Module? finalNorm = null, int? maxPositions = null)
    {
        ArgumentNullException.ThrowIfNull(words);
        ArgumentNullException.ThrowIfNull(blocks);
        if (tokenTypes is not null && tokenTypes.Dim != words.Dim)
        {
            throw new ArgumentException($"The token-type embedding is {tokenTypes.Dim} wide; the words' is {words.Dim}.", nameof(tokenTypes));
        }

        if (maxPositions is <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(maxPositions), maxPositions, "The encoder setting maxPositions is positive.");
        }

        Words = words;
        _blocks = [.. blocks];
        Positions = positions;
        TokenTypes = tokenTypes;
        EmbeddingNorm = embeddingNorm;
        FinalNorm = finalNorm;
        _maxPositions = maxPositions ?? (positions as IPositionLimit)?.MaxPositions ?? int.MaxValue;
        words.Name = "words";
        positions?.Name = "positions";
        tokenTypes?.Name = "types";
        embeddingNorm?.Name = "embedding_norm";
        finalNorm?.Name = "final_norm";
        for (int i = 0; i < _blocks.Length; i++)
        {
            _blocks[i].Name ??= $"block{i}";
        }
    }

    /// <summary>The token embedding.</summary>
    public Embedding Words { get; }

    /// <summary>The token-type embedding, or null.</summary>
    public Embedding? TokenTypes { get; }

    /// <summary>The module adding positions to the embeddings, or null.</summary>
    public Module? Positions { get; }

    /// <summary>The embeddings' normalization, or null.</summary>
    public Module? EmbeddingNorm { get; }

    /// <summary>The blocks, in order.</summary>
    public IReadOnlyList<Module> Blocks => _blocks;

    /// <summary>The last block's normalization, or null.</summary>
    public Module? FinalNorm { get; }

    /// <inheritdoc />
    public int Width => Words.Dim;

    /// <inheritdoc />
    public int Vocabulary => Words.Vocabulary;

    /// <inheritdoc />
    public int TypeVocabulary => TokenTypes?.Vocabulary ?? 0;

    /// <inheritdoc />
    public int MaxPositions => _maxPositions;

    /// <inheritdoc />
    public Device Device => WeightsDevice ?? Device.Default;

    /// <inheritdoc />
    public Module Network => this;

    /// <inheritdoc />
    public Tensor Encode(Tensor ids, IReadOnlyList<int>? lengths = null, Tensor? tokenTypes = null)
    {
        ArgumentNullException.ThrowIfNull(ids);
        if (ids.Rank != 2 || ids.Shape[1] == 0)
        {
            throw new ArgumentException($"An encoder takes token ids [batch, time] with at least one position, got {Tensor.FormatShape(ids.Shape)}.", nameof(ids));
        }

        int n = ids.Shape[0], t = ids.Shape[1];
        if (t > MaxPositions)
        {
            throw new ArgumentException($"{t} positions exceed the encoder's {MaxPositions}.", nameof(ids));
        }

        if (tokenTypes is not null && (TokenTypes is null || !tokenTypes.Shape.SequenceEqual(ids.Shape)))
        {
            throw new ArgumentException(TokenTypes is null ? "Token types were given to an encoder without a token-type embedding."
                : $"Token types are [batch, time] like the ids ({Tensor.FormatShape(ids.Shape)}), got {Tensor.FormatShape(tokenTypes.Shape)}.", nameof(tokenTypes));
        }

        PaddedSequences? padding = null;
        if (lengths is not null)
        {
            if (lengths.Count != n)
            {
                throw new ArgumentException($"{lengths.Count} lengths for {n} sequences.", nameof(lengths));
            }

            padding = new PaddedSequences(lengths, t, ids.Device);         // checks each length is 0 to t
            if (!padding.Padded)
            {
                padding.Dispose();
                padding = null;
            }
        }

        try
        {
            var x = Words.Forward(ids);
            if (TokenTypes is not null)
            {
                // Type 0 for every token unless given: its row added to every position (the same sums as looking it up).
                x = tokenTypes is null ? x + TokenTypes.Weight.Narrow(0, 0, 1).Reshape(Width) : x + TokenTypes.Forward(tokenTypes);
            }

            if (Positions is not null)
            {
                x = Positions.Forward(x);
            }

            if (EmbeddingNorm is not null)
            {
                x = EmbeddingNorm.Forward(x);
            }

            using (padding is null ? null : PaddedSequences.Use(padding))
            {
                foreach (var block in _blocks)
                {
                    x = block.Forward(x);
                }
            }

            return FinalNorm?.Forward(x) ?? x;
        }
        finally
        {
            // A recorded pass's backward may read the padding's masks: kept until the next pass (or Dispose).
            _training?.Dispose();
            _training = null;
            if (padding is not null && Autograd.IsEnabled)
            {
                _training = padding;
            }
            else
            {
                padding?.Dispose();
            }
        }
    }

    /// <summary>The states of <paramref name="input"/>, token ids [batch, time] with nothing padded and no token types.</summary>
    protected override Tensor ForwardCore(Tensor input) => Encode(input);

    /// <inheritdoc />
    public override IEnumerable<Module> Children()
    {
        yield return Words;
        if (TokenTypes is not null)
        {
            yield return TokenTypes;
        }

        if (Positions is not null)
        {
            yield return Positions;
        }

        if (EmbeddingNorm is not null)
        {
            yield return EmbeddingNorm;
        }

        foreach (var block in _blocks)
        {
            yield return block;
        }

        if (FinalNorm is not null)
        {
            yield return FinalNorm;
        }
    }

    /// <inheritdoc />
    /// <remarks>Each layer through its own Dispose (attention layers free their masks and key ranges), then the parameters.</remarks>
    public override void Dispose()
    {
        _training?.Dispose();
        _training = null;
        foreach (var child in Children())
        {
            child.Dispose();
        }

        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() =>
        $"SequenceEncoder({Vocabulary} tokens{(TokenTypes is null ? "" : $", {TypeVocabulary} types")}, width {Width}, {_blocks.Length} blocks)";
}
