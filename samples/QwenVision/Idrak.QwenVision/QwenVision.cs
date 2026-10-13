// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.QwenVision;

/// <summary>
/// The Qwen-VL vision-language families as a registration of Idrak's contracts (plan 16): the library knows no family;
/// this assembly brings Qwen2.5-VL (<c>Qwen2_5_VLForConditionalGeneration</c>) and Qwen2-VL
/// (<c>Qwen2VLForConditionalGeneration</c>). <see cref="Register"/> adds, for each, its vision part
/// (<see cref="QwenVisionFamily"/> in <see cref="VisionFamilies"/>) and its text decoder
/// (<see cref="QwenTextDecoder"/> in <see cref="PretrainedArchitectures"/>): one registration per family, nothing of
/// Qwen in the library. Load it in the command line with <c>idrak -P Idrak.QwenVision.dll ...</c> (or the config's
/// "plugins"), or call <see cref="Register"/> in an app.
/// </summary>
/// <remarks>
/// <para>
/// What the plug-in holds, and the step of plan 16 that builds each part:
/// <list type="bullet">
/// <item><see cref="QwenVisionConfig"/>: <c>vision_config</c> read exactly as transformers reads it (step 0 checks the keys).</item>
/// <item><see cref="QwenImageProcessor"/>: the smart resize within <c>min_pixels</c> and <c>max_pixels</c> to multiples of
/// the patch and merge sizes, and the pixel patches with their grid (step 5).</item>
/// <item><see cref="QwenPositions"/>: the <c>grid</c> image position rule (three rotary axes: time, height, width) and the
/// rope deltas, through the library's position-rule contract (step 1) and its multi-axis rotary positions (step 2).</item>
/// <item><see cref="QwenVisionTower"/>: the vision transformer (2D rotary positions, window and full attention blocks on
/// <c>AttentionSpans</c>, several images of different sizes in one pass by their spans) (steps 5 and 6).</item>
/// <item><see cref="QwenPatchMerger"/>: the merger of each <c>spatial_merge_size</c>² patch group into one text-width
/// token (steps 5 and 6).</item>
/// <item><see cref="QwenImageEncoder"/>: images to features, as <see cref="IVisionEncoder"/> and
/// <see cref="IVisionEncoderStages"/> (step 5).</item>
/// <item><see cref="QwenTextDecoder"/>: the text decoder's registration (Qwen2 style with rotary sections, the old and
/// new tensor names) (steps 2 and 5).</item>
/// <item><see cref="QwenVision"/>: the family's descriptor (prompt format, attention <c>causal</c>, positions <c>grid</c>,
/// the options <c>min_pixels</c> and <c>max_pixels</c>, the tuning parts) (step 5).</item>
/// </list>
/// </para>
/// <para>
/// Order (plan 16): 0 the reference fixtures, 1 and 2 the position rule and the multi-axis rotary positions in the
/// library, 3 training, 4 one registration per family, 5 Qwen2.5-VL, 6 Qwen2-VL, 7 speed and memory on the device,
/// 8 the legal OCR app. CUDA first: Vulkan and HIP take each contract's default.
/// </para>
/// </remarks>
public static class QwenVisionPlugin
{
    /// <summary>The name of the image position rule Qwen-VL registers once the library has the contract (plan 16, step 1).</summary>
    public const string GridPositions = "grid";

    /// <summary>Registers both families: the vision part and the text decoder of each (again: the same registrations).</summary>
    public static void Register()
    {
        foreach (var variant in (QwenVlVariant[])[QwenVlVariant.Qwen25Vl, QwenVlVariant.Qwen2Vl])
        {
            VisionFamilies.Register(new QwenVisionFamily(variant));
            QwenTextDecoder.Register(variant);
        }
    }

    /// <summary>The command line's plug-in entry (<c>-P</c>): <see cref="Register"/>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    // The one failure of this skeleton: what is asked for waits for a step of plan 16.
    internal static NotSupportedException NotBuilt(string what, int step) =>
        new($"Qwen-VL: {what} is not built yet (plan 16, step {step}).");
}

/// <summary>Which Qwen-VL family: the two differ in the vision tower (window attention, norms, activation) and in configuration keys.</summary>
public enum QwenVlVariant
{
    /// <summary>Qwen2.5-VL (<c>Qwen2_5_VLForConditionalGeneration</c>): window attention with full-attention blocks, RMSNorm, SwiGLU.</summary>
    Qwen25Vl,

    /// <summary>Qwen2-VL (<c>Qwen2VLForConditionalGeneration</c>): full attention in every block, LayerNorm, QuickGELU.</summary>
    Qwen2Vl,
}

/// <summary>
/// Reads the vision part of a Qwen-VL checkpoint (<see cref="IVisionFamily"/>): <c>vision_config</c>, the image token
/// ids, and every tower and merger tensor in whichever naming the checkpoint uses (transformers 4.x
/// <c>visual.*</c>; transformers 5 <c>model.visual.*</c>), its shape checked against the configuration.
/// </summary>
public sealed class QwenVisionFamily : IVisionFamily
{
    /// <summary>The architecture name of Qwen2.5-VL checkpoints.</summary>
    public const string Qwen25VlArchitecture = "Qwen2_5_VLForConditionalGeneration";

    /// <summary>The architecture name of Qwen2-VL checkpoints.</summary>
    public const string Qwen2VlArchitecture = "Qwen2VLForConditionalGeneration";

    /// <summary>The family for one variant.</summary>
    public QwenVisionFamily(QwenVlVariant variant) => Variant = variant;

    /// <summary>Which family this reads.</summary>
    public QwenVlVariant Variant { get; }

    /// <inheritdoc />
    public string Name => ArchitectureOf(Variant);

    /// <summary>The architecture name a variant's checkpoints carry.</summary>
    public static string ArchitectureOf(QwenVlVariant variant) => variant == QwenVlVariant.Qwen25Vl ? Qwen25VlArchitecture : Qwen2VlArchitecture;

    /// <inheritdoc />
    public PretrainedVision? Read(VisionCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        throw QwenVisionPlugin.NotBuilt($"reading the vision part of {Name}", Variant == QwenVlVariant.Qwen25Vl ? 5 : 6);
    }

    /// <inheritdoc />
    public override string ToString() => $"{Name} (skeleton)";
}

/// <summary>
/// The vision part of a Qwen-VL checkpoint as Idrak sees it (<see cref="PretrainedVision"/>): the descriptor every
/// consumer reads (the prompt format <c>&lt;|vision_start|&gt;&lt;|image_pad|&gt;…&lt;|vision_end|&gt;</c> with one pad
/// per merged patch group, attention <c>causal</c>, positions <see cref="QwenVisionPlugin.GridPositions"/>, the options
/// <c>min_pixels</c> and <c>max_pixels</c>), the encoder it builds, and, as <see cref="IVisionTuningPart"/>, the parts
/// that train (the merger; the tower when asked).
/// </summary>
public sealed class QwenVision : PretrainedVision, IVisionTuningPart
{
    /// <summary>The vision option keys Qwen-VL takes: the pixel budget of the smart resize.</summary>
    public static readonly IReadOnlyCollection<string> Keys = ["min_pixels", "max_pixels"];

    internal QwenVision(QwenVlVariant variant, QwenVisionConfig config, QwenImageTokens tokens, int textDim)
    {
        (Variant, Config, ImageTokens, TextDim) = (variant, config, tokens, textDim);
        // Qwen2VLProcessor: the template writes <|vision_start|><|image_pad|><|vision_end|>; the processor repeats
        // <|image_pad|> once per merged patch group (grid_h / merge × grid_w / merge) between the two markers it keeps.
        PromptFormat = new ImageTokenFormat("qwen-vl", tokens.ImagePad, tokens.ImagePad);
    }

    /// <summary>Which family.</summary>
    public QwenVlVariant Variant { get; }

    /// <summary>The vision tower's configuration (config.json's <c>vision_config</c>).</summary>
    public QwenVisionConfig Config { get; }

    /// <summary>The ids of the vision markers and the image pad in the text.</summary>
    public QwenImageTokens ImageTokens { get; }

    /// <summary>The text decoder's width: the merger's output, one embedding per merged patch group.</summary>
    public int TextDim { get; }

    /// <inheritdoc />
    public override string Family => QwenVisionFamily.ArchitectureOf(Variant);

    /// <inheritdoc />
    public override int Width => TextDim;

    /// <inheritdoc />
    public override IImagePromptFormat PromptFormat { get; }

    /// <summary>Qwen-VL's image tokens attend causally, like text (the positions carry the image's geometry, not the mask).</summary>
    public override IImageAttentionRule Attention => ImageAttentionRules.Get(ImageAttentionRules.Causal);

    /// <inheritdoc />
    public override IReadOnlyCollection<string> StoredTensors => throw QwenVisionPlugin.NotBuilt("the vision tensor list", 5);

    /// <inheritdoc />
    public override IReadOnlyCollection<string> VisionOptionKeys => Keys;

    /// <inheritdoc />
    public override IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null) => throw QwenVisionPlugin.NotBuilt("the image encoder", 5);

    /// <summary>Qwen-VL lets its merger train, and its tower when asked (the merger alone by default, as the usual LoRA runs do).</summary>
    public IReadOnlyList<string> TrainableParts { get; } = [VisionTuningParts.Projector, VisionTuningParts.Tower];

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Parameters(IVisionEncoder encoder, string part) => throw QwenVisionPlugin.NotBuilt("tuning parts", 5);

    /// <inheritdoc />
    public Tensor Features(IVisionEncoder encoder, Tensor towerOutput) => throw QwenVisionPlugin.NotBuilt("the merger for tuning", 5);

    /// <inheritdoc />
    public Tensor Tower(IVisionEncoder encoder, Tensor pixelValues) => throw QwenVisionPlugin.NotBuilt("the tower for tuning", 5);

    /// <inheritdoc />
    public IReadOnlyDictionary<string, Tensor> Export(IVisionEncoder encoder, string part) => throw QwenVisionPlugin.NotBuilt("exporting trained parts", 5);

    /// <inheritdoc />
    public IReadOnlyCollection<string> Import(IVisionEncoder encoder, IReadOnlyDictionary<string, Tensor> tensors) => throw QwenVisionPlugin.NotBuilt("importing trained parts", 5);

    /// <inheritdoc />
    public override string ToString() => $"{Family}: {Config}, text width {TextDim} (skeleton)";
}

/// <summary>The ids of the vision markers and the image pad token (config.json's <c>vision_start_token_id</c>, <c>vision_end_token_id</c>, <c>image_token_id</c>).</summary>
/// <param name="VisionStart">The id of <c>&lt;|vision_start|&gt;</c>, which opens an image in the prompt.</param>
/// <param name="VisionEnd">The id of <c>&lt;|vision_end|&gt;</c>, which closes it.</param>
/// <param name="ImagePad">The id of <c>&lt;|image_pad|&gt;</c>, repeated once per image token between the markers.</param>
public sealed record QwenImageTokens(int VisionStart, int VisionEnd, int ImagePad)
{
    /// <summary>The ids from config.json, with the Qwen2-VL and Qwen2.5-VL defaults when absent.</summary>
    public static QwenImageTokens FromJson(JsonObject config)
    {
        ArgumentNullException.ThrowIfNull(config);
        return new((int?)config["vision_start_token_id"] ?? 151_652, (int?)config["vision_end_token_id"] ?? 151_653, (int?)config["image_token_id"] ?? 151_655);
    }
}

/// <summary>
/// The vision tower's configuration (<c>vision_config</c>), in the names both families share here. The keys each
/// family writes (Qwen2.5-VL: <c>hidden_size</c>, <c>intermediate_size</c>, <c>out_hidden_size</c>,
/// <c>window_size</c>, <c>fullatt_block_indexes</c>; Qwen2-VL: <c>embed_dim</c>, <c>mlp_ratio</c>, the text
/// <c>hidden_size</c>) are read in <see cref="FromJson"/> once step 0 of plan 16 has checked them against transformers.
/// </summary>
public sealed record QwenVisionConfig
{
    /// <summary>Transformer blocks in the tower (<c>depth</c>).</summary>
    public required int Depth { get; init; }

    /// <summary>The tower's width.</summary>
    public required int Dim { get; init; }

    /// <summary>The tower's feed-forward width.</summary>
    public required int FfDim { get; init; }

    /// <summary>Attention heads (<c>num_heads</c>).</summary>
    public required int Heads { get; init; }

    /// <summary>Input channels (<c>in_channels</c>).</summary>
    public required int Channels { get; init; }

    /// <summary>The patch side in pixels (<c>patch_size</c>).</summary>
    public required int PatchSize { get; init; }

    /// <summary>Patches merged per side into one token (<c>spatial_merge_size</c>): a token is this² patches.</summary>
    public required int MergeSize { get; init; }

    /// <summary>Frames per temporal patch (<c>temporal_patch_size</c>); a still image is repeated to fill one.</summary>
    public required int TemporalPatchSize { get; init; }

    /// <summary>The merger's output width: the text decoder's width.</summary>
    public required int OutDim { get; init; }

    /// <summary>Qwen2.5-VL: the window side in pixels of the window-attention blocks (<c>window_size</c>); 0 for full attention everywhere (Qwen2-VL).</summary>
    public int WindowSize { get; init; }

    /// <summary>Qwen2.5-VL: the blocks that attend over the whole image (<c>fullatt_block_indexes</c>); empty for Qwen2-VL.</summary>
    public IReadOnlyList<int> FullAttentionBlocks { get; init; } = [];

    /// <summary>Patches per side of one merged token, as the position rule and the prompt format count them.</summary>
    public int PatchesPerToken => MergeSize * MergeSize;

    /// <summary>Reads <c>vision_config</c> for one family.</summary>
    public static QwenVisionConfig FromJson(JsonObject? config, QwenVlVariant variant) => throw QwenVisionPlugin.NotBuilt("vision_config (its keys are checked against transformers in step 0)", 0);

    /// <inheritdoc />
    public override string ToString() => $"{Depth} blocks × {Dim} ({Heads} heads), patch {PatchSize}, merge {MergeSize}, out {OutDim}";
}
