// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.T5;

/// <summary>
/// The T5 family: its encoder-decoder checkpoints (Hugging Face's <c>T5ForConditionalGeneration</c> and
/// <c>MT5ForConditionalGeneration</c>, config.json's "architectures") registered in <see cref="PretrainedSeq2Seq"/>, so
/// Idrak's <c>PretrainedSeq2SeqModel.Load</c> reads them and <c>Seq2SeqGenerator</c> generates with them (greedy,
/// sampled or beam search), and its encoder checkpoints (<c>T5EncoderModel</c>, <c>MT5EncoderModel</c>, or the encoder
/// of either encoder-decoder read as one) in <see cref="PretrainedEncoders"/> (sentence vectors by the mean of the
/// states). The layers are the library's: the shared token embedding (the encoder's, the decoder's and, tied, the head's
/// table), pre-norm blocks with RMS norms (<c>TransformerEncoderLayer</c> with RMS norms, <c>DecoderBlock</c> with
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

    /// <summary>The encoder-decoder architecture names the plug-in registers in <see cref="PretrainedSeq2Seq"/>, in order.</summary>
    public static IReadOnlyList<string> Seq2SeqArchitectures { get; } = [T5ForConditionalGeneration, MT5ForConditionalGeneration];

    /// <summary>The encoder architecture names the plug-in registers in <see cref="PretrainedEncoders"/>, in order.</summary>
    public static IReadOnlyList<string> EncoderArchitectures { get; } = [T5EncoderModel, MT5EncoderModel];

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("t5"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new T5Registration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family under each of its architecture names (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving <see cref="PretrainedSeq2Seq"/> and <see cref="PretrainedEncoders"/> as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The family under each architecture name, one part each.
    private sealed class T5Registration() : IdrakPlugin("t5")
    {
        protected override void RegisterParts()
        {
            foreach (string architecture in Seq2SeqArchitectures)
            {
                Part(() => PretrainedSeq2Seq.Register(new T5Seq2SeqFamily(architecture)), () => PretrainedSeq2Seq.Unregister(architecture));
            }

            foreach (string architecture in EncoderArchitectures)
            {
                Part(() => PretrainedEncoders.Register(new T5EncoderFamily(architecture)), () => PretrainedEncoders.Unregister(architecture));
            }
        }
    }
}
