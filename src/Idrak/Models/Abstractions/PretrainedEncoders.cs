// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Layers;

namespace Idrak.Models.Abstractions;

/// <summary>
/// What an encoder or encoder-decoder family builds its network from (<see cref="IPretrainedEncoderFamily.Build"/>,
/// <see cref="IPretrainedSeq2SeqFamily.Build"/>).
/// </summary>
/// <param name="Architecture">The architecture name the checkpoint was matched by (config.json's "architectures").</param>
/// <param name="Config">The model's config.json.</param>
/// <param name="Tensors">The checkpoint's tensors by their stored names, open while building (do not keep it).</param>
/// <param name="Folder">The model folder, or null.</param>
/// <param name="Device">Where the network is built.</param>
/// <param name="Notes">Append anything approximated (shown in the loaded model's notes).</param>
public sealed record PretrainedCheckpoint(string Architecture, JsonObject Config, ITensorStore Tensors, string? Folder, Device Device, List<string> Notes);

/// <summary>
/// An encoder family (a model of token ids to hidden states, such as a masked-language-model encoder): how its
/// checkpoints become an <see cref="ISequenceEncoder"/> (ids, each sequence's length and optional token types to hidden
/// states [batch, time, width]), with the heads the checkpoint holds and the pooler its sentence vectors use. Idrak's
/// <c>PretrainedEncoderModel.Load</c> reads one. The library has none: a family is a plug-in's.
/// </summary>
public interface IPretrainedEncoderFamily
{
    /// <summary>The architecture name its checkpoints give (config.json's "architectures").</summary>
    string Name { get; }

    /// <summary>
    /// The names its checkpoints call it by, the family's own: a checkpoint's name is matched against them where it comes in
    /// (<see cref="IFamilyProvider.EncoderFamilies"/>, walked in order). <see cref="Name"/> by default.
    /// </summary>
    IReadOnlyList<string> Names => [Name];

    /// <summary>The encoder of the checkpoint (Idrak's <c>SequenceEncoder</c> built from its tensors, or one of the family's own).</summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    ISequenceEncoder Build(PretrainedCheckpoint checkpoint);

    /// <summary>
    /// The heads the checkpoint holds beside the encoder (a classifier, a masked-language-model head; made with
    /// <see cref="EncoderHead"/>'s factories), none by default.
    /// </summary>
    IReadOnlyList<EncoderHead> Heads(PretrainedCheckpoint checkpoint, ISequenceEncoder encoder) => [];

    /// <summary>The pooler (a name one of <see cref="SequencePoolers.All"/> states) the family's sequence vectors use: "first" by default.</summary>
    string Pooler(JsonObject config) => SequencePoolers.First;
}

/// <summary>
/// A text encoder-decoder family: how its checkpoints become an <see cref="ISeq2SeqModel"/> (the source encoded once,
/// the target decoded token by token with cross-attention to it). Idrak's <c>PretrainedSeq2SeqModel.Load</c> reads one and
/// <c>Seq2SeqGenerator</c> generates with it. The library has none: a family is a plug-in's.
/// </summary>
public interface IPretrainedSeq2SeqFamily
{
    /// <summary>The architecture name its checkpoints give (config.json's "architectures").</summary>
    string Name { get; }

    /// <summary>
    /// The names its checkpoints call it by, the family's own: a checkpoint's name is matched against them where it comes in
    /// (<see cref="IFamilyProvider.Seq2SeqFamilies"/>, walked in order). <see cref="Name"/> by default.
    /// </summary>
    IReadOnlyList<string> Names => [Name];

    /// <summary>
    /// The model of the checkpoint: Idrak's <c>Seq2SeqNetwork</c> (an <see cref="ISequenceEncoder"/> and a decoder of
    /// <c>DecoderBlock</c>s with cross-attention) built from its tensors, or one of the family's own.
    /// </summary>
    /// <exception cref="InvalidDataException">A tensor is missing or its shape disagrees with the configuration.</exception>
    ISeq2SeqModel Build(PretrainedCheckpoint checkpoint);

    /// <summary>
    /// Whether the family's sources end with its end token (the model was trained so): generation then appends it when the
    /// tokenizer's encoding does not end with it (a tokenizer read from a SentencePiece model file adds none). False by default.
    /// </summary>
    bool EndsSource(JsonObject config) => false;
}

/// <summary>What a head's factory (<see cref="EncoderHead.Classification"/>, …) makes a head from.</summary>
/// <param name="Width">The encoder's width (the head's input).</param>
/// <param name="Outputs">The head's outputs: the labels of a classifier, the vocabulary of a masked-language-model head.</param>
/// <param name="Settings">The kind's own settings (see its factory).</param>
/// <param name="Device">Where its parameters live (the default device when null).</param>
/// <param name="Seed">The seed of random initial values.</param>
/// <param name="Weights">
/// Reads the head's tensors by their names in the kind ("dense.weight", "out.bias", ...) and shapes in Idrak's layout
/// (a linear layer's weight [in, out]), or null for random values; it returns null for a tensor it does not hold (an
/// optional bias then starts at zero).
/// </param>
public sealed record EncoderHeadRequest(int Width, int Outputs, JsonObject Settings, Device? Device = null, int Seed = 0,
    Func<string, IReadOnlyList<int>, float[]?>? Weights = null)
{
    /// <summary>
    /// The token embedding a head's output layer reads in place, transposed (a tied output: one copy of the table, which
    /// gradients reach as the embedding's), instead of an "out.weight" of its own; null (the default) for none. The
    /// "masked-lm" kind takes it ([outputs, width]); its "out.bias" is still read.
    /// </summary>
    public Embedding? TiedTo { get; init; }
}

/// <summary>
/// The names of the head kinds an encoder's checkpoint may hold (<see cref="EncoderHead.Kind"/>), each made by its own
/// factory: "sequence-classification" (<see cref="EncoderHead.Classification"/>), "masked-lm"
/// (<see cref="EncoderHead.MaskedLanguageModel"/>) and "pooler" (<see cref="EncoderHead.Pooler"/>). Picking a head kind by
/// its name is not on the new design yet.
/// </summary>
public static class EncoderHeads
{
    /// <summary>The sequence classifier: "sequence-classification".</summary>
    public const string SequenceClassification = "sequence-classification";

    /// <summary>The masked-language-model head: "masked-lm".</summary>
    public const string MaskedLanguageModel = "masked-lm";

    /// <summary>The pooled vector's projection (a dense layer and an activation, the vector a classifier reads): "pooler".</summary>
    public const string Pooler = "pooler";
}
