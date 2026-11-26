// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Gemma;
using Idrak.Text.Plugins.Llama;
using Idrak.Text.Plugins.Mistral;
using Idrak.Text.Plugins.Qwen;

namespace Idrak.Text.Plugins;

/// <summary>
/// The text families this package holds, handed over as one immutable list: each defined by its own class (its spec,
/// its tensor names, the names its checkpoints call it by, <see cref="IPretrainedArchitecture.Names"/>). This list holds
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
}
