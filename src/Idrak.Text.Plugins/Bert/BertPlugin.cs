// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Models.Abstractions;

namespace Idrak.Text.Plugins.Bert;

/// <summary>
/// The BERT family: its encoder checkpoints (Hugging Face's <c>BertModel</c>, <c>BertForMaskedLM</c> and
/// <c>BertForSequenceClassification</c>, config.json's "architectures") registered in <see cref="PretrainedEncoders"/>, so
/// Idrak's <c>PretrainedEncoderModel.Load</c> reads them: word, learned position and token-type embeddings with their
/// LayerNorm, post-norm blocks (<c>TransformerEncoderLayer</c> with <c>NormPlacement.After</c>) of the configuration's
/// activation, and the heads the checkpoint holds (<c>EncoderHeads</c>' "pooler": dense + tanh of the first token;
/// "masked-lm": the transform, its LayerNorm and the decoder tied to the word embeddings; "sequence-classification":
/// the pooler, then the classifier). Sentence vectors go through Idrak.Text's <c>TextEncoder</c> over the model's
/// encoder, with its "mean" or "first" pooler and <c>SpecialTokens</c>. The tokenizer is the folder's WordPiece
/// (<c>tokenizer.json</c> or <c>vocab.txt</c>).
/// </summary>
public static class BertPlugin
{
    /// <summary>The encoder alone: "BertModel".</summary>
    public const string BertModel = "BertModel";

    /// <summary>The encoder with its masked-language-model head: "BertForMaskedLM".</summary>
    public const string BertForMaskedLM = "BertForMaskedLM";

    /// <summary>The encoder with its pooler and classifier: "BertForSequenceClassification".</summary>
    public const string BertForSequenceClassification = "BertForSequenceClassification";

    /// <summary>The architecture names the plug-in registers, in order.</summary>
    public static IReadOnlyList<string> Architectures { get; } = [BertModel, BertForMaskedLM, BertForSequenceClassification];

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("bert"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new BertRegistration();

    /// <summary>Whether the family is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the family under each of <see cref="Architectures"/> (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the family, leaving <see cref="PretrainedEncoders"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // The family under each architecture name, one part each.
    private sealed class BertRegistration() : IdrakPlugin("bert")
    {
        protected override void RegisterParts()
        {
            foreach (string architecture in Architectures)
            {
                Part(() => PretrainedEncoders.Register(new BertFamily(architecture)), () => PretrainedEncoders.Unregister(architecture));
            }
        }
    }
}
