// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Models.Abstractions;
using Idrak.Text.Plugins.Gemma;
using Idrak.Vision.Plugins.Gemma;
using Idrak.Vision.Plugins.IdrakVision;
using Idrak.Vision.Plugins.Llava;
using Idrak.Vision.Plugins.Qwen;

namespace Idrak.Vision.Plugins;

/// <summary>
/// The vision-language families this package holds, with the text families they build on (Idrak.Text.Plugins'
/// <c>BuiltInFamilies.All</c>), handed over as one immutable list: each vision-language family's text decoder (Qwen-VL's,
/// Gemma 3's with images, LLaVA's, IdrakVision's) and its vision part (<see cref="VisionFamilies"/>), each defined by its
/// own class and stating the names its checkpoints call it by. This list holds no definition and takes no other family; a
/// checkpoint's own name is matched against the families' own names only where the checkpoint comes in
/// (<c>PretrainedOptions.Families</c>), by walking it.
/// </summary>
public sealed class BuiltInVisionFamilies : IFamilyProvider
{
    private static readonly IFamilyProvider Text = Idrak.Text.Plugins.BuiltInFamilies.All;

    private BuiltInVisionFamilies()
    {
    }

    /// <summary>The package's vision-language families and the text families they build on.</summary>
    public static BuiltInVisionFamilies All { get; } = new();

    /// <inheritdoc />
    public IReadOnlyList<IPretrainedArchitecture> Families { get; } = Array.AsReadOnly<IPretrainedArchitecture>(
    [
        .. Text.Families,
        QwenTextDecoder.Architecture(), GemmaArchitectures.Gemma3VisionLanguage(), LlavaText.Architecture(LlavaText.Layouts[0]), IdrakVisionPlugin.TextDecoder(),
    ]);

    /// <inheritdoc />
    public IReadOnlyList<IVisionFamily> VisionFamilies { get; } = Array.AsReadOnly<IVisionFamily>(
    [
        new QwenVisionFamily(QwenVlVariant.Qwen25Vl), new QwenVisionFamily(QwenVlVariant.Qwen2Vl), new Gemma3VisionFamily(), new LlavaVisionFamily(), new IdrakVisionFamily(),
    ]);

    /// <inheritdoc />
    public IReadOnlyList<IGgufArchitecture> GgufArchitectures => Text.GgufArchitectures;

    /// <inheritdoc />
    public IReadOnlyList<IGgufPreTokenizer> GgufPreTokenizers => Text.GgufPreTokenizers;

    /// <inheritdoc />
    public IReadOnlyList<IPretrainedEncoderFamily> EncoderFamilies => Text.EncoderFamilies;

    /// <inheritdoc />
    public IReadOnlyList<IPretrainedSeq2SeqFamily> Seq2SeqFamilies => Text.Seq2SeqFamilies;
}
