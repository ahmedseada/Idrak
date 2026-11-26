// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Llama;

namespace Idrak.Text.Plugins.Mistral;

/// <summary>
/// The Mistral family's names and parts: Hugging Face's <c>MistralForCausalLM</c> and <c>MixtralForCausalLM</c>
/// (config.json's "architectures", read by <see cref="MistralArchitectures"/>), the tekken GGUF pre-tokenizer's split
/// pattern (<see cref="MistralArchitectures.TekkenPattern"/>), Mistral's tool-call format "mistral" (<c>[TOOL_CALLS]</c>),
/// and Codestral's fill-in-the-middle format "mistral" (<see cref="FillInMiddle"/>). Its GGUF files are llama.cpp's
/// "llama" (<see cref="LlamaArchitectures.Gguf"/>). Registering them is not on the new design yet.
/// </summary>
public static class MistralPlugin
{
    /// <summary>Mistral: "MistralForCausalLM".</summary>
    public const string MistralForCausalLM = "MistralForCausalLM";

    /// <summary>Mixtral, Mistral's mixture of experts: "MixtralForCausalLM".</summary>
    public const string MixtralForCausalLM = "MixtralForCausalLM";

    /// <summary>The tekken GGUF pre-tokenizer: "tekken".</summary>
    public const string PreTokenizer = "tekken";

    /// <summary>Mistral's tool-call format in <see cref="ToolCallFormats"/>: "mistral".</summary>
    public const string ToolCallFormat = "mistral";

    /// <summary>Mistral's fill-in-the-middle format in <see cref="FillInMiddleFormats"/>: "mistral".</summary>
    public const string FillInMiddleFormat = "mistral";

    /// <summary>
    /// Mistral's fill-in-the-middle format (Codestral), as mistral-common's <c>encode_fim</c> lays it out: suffix first and
    /// no middle token, <c>&lt;s&gt;[SUFFIX]suffix[PREFIX]prefix</c>, so the model continues the prefix; ending on
    /// <c>&lt;/s&gt;</c>. Mistral writes no repository files (<see cref="IFillInMiddlePrompt.Snippets"/> is 0).
    /// </summary>
    public static FillInMiddleLayout FillInMiddle { get; } = FillInMiddleLayout.SuffixPrefixMiddle(FillInMiddleFormat,
        "Mistral's (Codestral's) <s>[SUFFIX] suffix [PREFIX] prefix, suffix first",
        new FillInMiddleTokens("[PREFIX]", "[SUFFIX]", null) { Begin = "<s>", Stop = ["</s>"] });
}

// The family's tool-call format.
internal sealed class MistralToolCallFormat : IToolCallFormat
{
    public string Name => MistralPlugin.ToolCallFormat;

    public bool Detect(ToolCallProbe probe) => probe.Shows(MistralToolCallParser.Marker);

    public IToolCallParser Create(ToolCallContext context) => new MistralToolCallParser(context);
}
