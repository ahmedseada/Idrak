// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Bert;
using Idrak.Text.Plugins.Gemma;
using Idrak.Text.Plugins.Llama;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;
using Idrak.Text.Plugins.T5;

namespace Idrak.Text.Plugins;

/// <summary>
/// The text families this package holds, handed over as one immutable list: each defined by its own class (its spec,
/// its tensor names, the names its checkpoints call it by, <see cref="IPretrainedArchitecture.Names"/>), with its encoder
/// families (BERT's, T5's encoder) and encoder-decoder families (T5's), each stating its names too. This list holds
/// no definition and takes no other family; a checkpoint's own name is matched against the families' own names only where
/// the checkpoint comes in (<see cref="Idrak.Models.PretrainedOptions.Families"/>), by walking it.
/// </summary>
public sealed class BuiltInFamilies : IFamilyProvider
{
    private BuiltInFamilies()
    {
    }

    /// <summary>The package's text families.</summary>
    public static BuiltInFamilies All { get; } = new();

    /// <inheritdoc />
    public IReadOnlyList<IPretrainedArchitecture> Families { get; } = Array.AsReadOnly<IPretrainedArchitecture>(
    [
        QwenArchitectures.Qwen2(), QwenArchitectures.Qwen3(), QwenArchitectures.Qwen2Moe(), QwenArchitectures.Qwen3Moe(),
        LlamaArchitectures.Llama(), MistralArchitectures.Mistral(), MistralArchitectures.Mixtral(),
        GemmaArchitectures.Gemma(), GemmaArchitectures.Gemma2(), GemmaArchitectures.Gemma3(),
    ]);

    /// <inheritdoc />
    public IReadOnlyList<IGgufArchitecture> GgufArchitectures { get; } = Array.AsReadOnly<IGgufArchitecture>(
    [
        .. QwenArchitectures.Gguf(), LlamaArchitectures.Gguf(), GemmaGguf.Gemma(), GemmaGguf.Gemma2(), GemmaGguf.Gemma3(),
    ]);

    /// <inheritdoc />
    public IReadOnlyList<IPretrainedEncoderFamily> EncoderFamilies { get; } = Array.AsReadOnly<IPretrainedEncoderFamily>(
    [
        new BertFamily(BertPlugin.BertModel), new BertFamily(BertPlugin.BertForMaskedLM), new BertFamily(BertPlugin.BertForSequenceClassification),
        new T5EncoderFamily(T5Plugin.T5EncoderModel), new T5EncoderFamily(T5Plugin.MT5EncoderModel),
    ]);

    /// <inheritdoc />
    public IReadOnlyList<IPretrainedSeq2SeqFamily> Seq2SeqFamilies { get; } = Array.AsReadOnly<IPretrainedSeq2SeqFamily>(
    [
        new T5Seq2SeqFamily(T5Plugin.T5ForConditionalGeneration), new T5Seq2SeqFamily(T5Plugin.MT5ForConditionalGeneration),
    ]);

    /// <inheritdoc />
    public IReadOnlyList<IGgufPreTokenizer> GgufPreTokenizers { get; } = Array.AsReadOnly<IGgufPreTokenizer>(
    [
        new GgufPreTokenizer(["qwen2", "qwen35", "megrez", "hunyuan"], QwenArchitectures.SplitPattern),
        new GgufPreTokenizer(["llama3", "llama-bpe", "llama-v3", "smaug-bpe", "falcon3", "pixtral"], LlamaArchitectures.SplitPattern),
    ]);
}
