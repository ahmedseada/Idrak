// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Bert;

/// <summary>
/// The BERT family's names: its encoder checkpoints (Hugging Face's <c>BertModel</c>, <c>BertForMaskedLM</c> and
/// <c>BertForSequenceClassification</c>, config.json's "architectures"), each read by a <see cref="BertFamily"/> of that
/// name: word, learned position and token-type embeddings with their LayerNorm, post-norm blocks
/// (<c>TransformerEncoderLayer</c> with <c>NormPlacement.After</c>) of the configuration's activation, and the heads the
/// checkpoint holds (<c>EncoderHead.Pooler</c>: dense + tanh of the first token; <c>EncoderHead.MaskedLanguageModel</c>:
/// the transform, its LayerNorm and the decoder tied to the word embeddings; <c>EncoderHead.Classification</c>: the
/// pooler, then the classifier). Registering the family is not on the new design yet. The tokenizer is the folder's
/// WordPiece (<c>tokenizer.json</c> or <c>vocab.txt</c>).
/// </summary>
public static class BertPlugin
{
    /// <summary>The encoder alone: "BertModel".</summary>
    public const string BertModel = "BertModel";

    /// <summary>The encoder with its masked-language-model head: "BertForMaskedLM".</summary>
    public const string BertForMaskedLM = "BertForMaskedLM";

    /// <summary>The encoder with its pooler and classifier: "BertForSequenceClassification".</summary>
    public const string BertForSequenceClassification = "BertForSequenceClassification";
}
