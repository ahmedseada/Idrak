// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.T5;

/// <summary>
/// The T5 family's names: its encoder-decoder checkpoints (Hugging Face's <c>T5ForConditionalGeneration</c> and
/// <c>MT5ForConditionalGeneration</c>, config.json's "architectures"), read by a <see cref="T5Seq2SeqFamily"/> of that
/// name, so <c>Seq2SeqGenerator</c> generates with them (greedy, sampled or beam search), and its encoder checkpoints
/// (<c>T5EncoderModel</c>, <c>MT5EncoderModel</c>, or the encoder of either encoder-decoder read as one), read by a
/// <see cref="T5EncoderFamily"/> (sentence vectors by the mean of the states). Registering the family is not on the new
/// design yet. The layers are the library's: the shared token embedding (the encoder's, the decoder's and, tied, the
/// head's table), pre-norm blocks with RMS norms (<c>TransformerEncoderLayer</c> with RMS norms, <c>DecoderBlock</c> with
/// cross-attention), attention without score scaling, the library's "bucketed" relative position bias of each stack's
/// first block shared by the others (both ways in the encoder, one way in the decoder), the ReLU feed-forward (T5 v1.0,
/// feed_forward_proj relu) or the gated GELU one (v1.1 and mT5, gated-gelu; gated-silu too), and the head: tied to the
/// shared embedding with the decoder's output scaled by d_model^-0.5 (v1.0), or lm_head.weight unscaled
/// (tie_word_embeddings false). The decoder starts from decoder_start_token_id and ends at eos_token_id. The tokenizer is
/// the folder's Unigram (<c>tokenizer.json</c>, or <c>spiece.model</c>: then the source's closing &lt;/s&gt; is appended by
/// generation). Settings it does not read are refused naming them.
/// </summary>
public static class T5Plugin
{
    /// <summary>T5's encoder-decoder: "T5ForConditionalGeneration".</summary>
    public const string T5ForConditionalGeneration = "T5ForConditionalGeneration";

    /// <summary>mT5's encoder-decoder (the same layers): "MT5ForConditionalGeneration".</summary>
    public const string MT5ForConditionalGeneration = "MT5ForConditionalGeneration";

    /// <summary>T5's encoder alone: "T5EncoderModel".</summary>
    public const string T5EncoderModel = "T5EncoderModel";

    /// <summary>mT5's encoder alone: "MT5EncoderModel".</summary>
    public const string MT5EncoderModel = "MT5EncoderModel";
}
