// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Plugins;
using Idrak.Imaging;
using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Vision.Plugins.Qwen;

/// <summary>
/// The Qwen-VL vision-language families as a plug-in of Idrak's contracts (plan 16): the library knows no family; this
/// plug-in brings Qwen2.5-VL (<c>Qwen2_5_VLForConditionalGeneration</c>) and Qwen2-VL
/// (<c>Qwen2VLForConditionalGeneration</c>). <see cref="Register"/> adds, for each, its vision part
/// (<see cref="QwenVisionFamily"/> in <see cref="VisionFamilies"/>) and its text decoder
/// (<see cref="QwenTextDecoder"/> in <see cref="PretrainedArchitectures"/>): one registration per family, nothing of
/// Qwen in the library. <see cref="VisionPlugins.RegisterAll"/> registers it with the package's other families (the
/// idrak tool does at startup); an app may call <see cref="Register"/> alone.
/// </summary>
/// <remarks>
/// What the library gives it: the image tokens placed on three rotary axes by the library's
/// <see cref="ImagePositionRules.Grid"/> rule from each token's position in its merged grid
/// (<see cref="IVisionEncoder.AxisPositions"/>), and the decoder's multi-axis rotary embedding
/// (<see cref="RopeSettings.Sections"/>). What is Qwen's, here: <see cref="QwenImageProcessor"/> (the smart resize within
/// the pixel budget, the patches), <see cref="QwenVisionTower"/> (2D rotary positions, window and full attention),
/// <see cref="QwenPatchMerger"/>, <see cref="QwenImageEncoder"/>, the checkpoint's tensors and the text decoder.
/// </remarks>
public static class QwenVisionPlugin
{
    /// <summary>Both variants, in the order they are registered.</summary>
    public static IReadOnlyList<QwenVlVariant> Variants { get; } = [QwenVlVariant.Qwen25Vl, QwenVlVariant.Qwen2Vl];

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("qwen-vl"; listed in <see cref="VisionPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new QwenRegistration();

    /// <summary>Whether the families are registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers both families: the vision part and the text decoder of each (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters both families, leaving <see cref="VisionFamilies"/> and <see cref="PretrainedArchitectures"/> as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    // Each variant's vision part and text decoder, one part each.
    private sealed class QwenRegistration() : IdrakPlugin("qwen-vl")
    {
        protected override void RegisterParts()
        {
            foreach (var variant in Variants)
            {
                string architecture = QwenVisionFamily.ArchitectureOf(variant);
                Part(() => VisionFamilies.Register(new QwenVisionFamily(variant)), () => VisionFamilies.Unregister(architecture));
                Part(() => PretrainedArchitectures.Register(architecture, QwenTextDecoder.Architecture()), () => PretrainedArchitectures.Unregister(architecture));
            }
        }
    }
}

/// <summary>Which Qwen-VL family: the two differ in the vision tower (window attention, norms, activation) and in configuration keys.</summary>
public enum QwenVlVariant
{
    /// <summary>Qwen2.5-VL (<c>Qwen2_5_VLForConditionalGeneration</c>): window attention with full-attention blocks, RMSNorm, SwiGLU.</summary>
    Qwen25Vl,

    /// <summary>Qwen2-VL (<c>Qwen2VLForConditionalGeneration</c>): full attention in every block, LayerNorm, quick-GELU.</summary>
    Qwen2Vl,
}

/// <summary>
/// Reads the vision part of a Qwen-VL checkpoint (<see cref="IVisionFamily"/>): <c>vision_config</c> (transformers 5's
/// keys, or the hub checkpoints' 4.x keys), the image token ids, and every tower and merger tensor under
/// <c>visual.</c> (the hub's naming, which transformers 5 still saves) or <c>model.visual.</c>, each shape checked
/// against the configuration.
/// </summary>
public sealed class QwenVisionFamily : IVisionFamily
{
    /// <summary>The architecture name of Qwen2.5-VL checkpoints.</summary>
    public const string Qwen25VlArchitecture = "Qwen2_5_VLForConditionalGeneration";

    /// <summary>The architecture name of Qwen2-VL checkpoints.</summary>
    public const string Qwen2VlArchitecture = "Qwen2VLForConditionalGeneration";

    private static readonly string[] Prefixes = ["visual.", "model.visual."];

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
        var store = checkpoint.Tensors;
        string? prefix = Prefixes.FirstOrDefault(p => store.Contains(p + "patch_embed.proj.weight"));
        if (prefix is null)
        {
            checkpoint.Notes.Add("The checkpoint has no vision tower: the model reads text only.");
            return null;
        }

        var config = QwenVisionConfig.FromJson(checkpoint.Config["vision_config"] as JsonObject, Variant);
        int textDim = PretrainedArchitectures.Get(checkpoint.Architecture).Spec(checkpoint.Config, []).Dim;
        if (config.OutDim != textDim)
        {
            throw new InvalidDataException($"vision_config: the merger's output width {config.OutDim} is not the text decoder's {textDim}.");
        }

        var tensors = new Dictionary<string, QwenTensor>(StringComparer.Ordinal);
        foreach (var (name, shape) in QwenVision.Expected(config, Variant))
        {
            string stored = prefix + name;
            if (!store.Contains(stored))
            {
                throw new InvalidDataException($"The checkpoint has no '{stored}', which vision_config asks for.");
            }

            var actual = store.ShapeOf(stored);
            if (!actual.SequenceEqual(shape))
            {
                throw new InvalidDataException($"{stored}: shape {Tensor.FormatShape(actual)}, expected {Tensor.FormatShape(shape)} from vision_config.");
            }

            tensors[name] = new QwenTensor(stored, shape);
        }

        return new QwenVision(Variant, config, QwenImageTokens.FromJson(checkpoint.Config), textDim, prefix, tensors, checkpoint.Open, checkpoint.Folder);
    }

    /// <inheritdoc />
    public override string ToString() => Name;
}

/// <summary>A vision tensor: its name in the checkpoint and its shape there.</summary>
/// <param name="Stored">The name in the checkpoint (with its prefix).</param>
/// <param name="Shape">The shape as stored.</param>
public sealed record QwenTensor(string Stored, IReadOnlyList<int> Shape);

/// <summary>
/// The vision part of a Qwen-VL checkpoint as Idrak sees it (<see cref="PretrainedVision"/>): the prompt format
/// (<c>&lt;|vision_start|&gt;&lt;|image_pad|&gt;&lt;|vision_end|&gt;</c> written by the chat template, the pad repeated once
/// per image token), causal attention (the geometry is in the positions, not the mask), the image tokens placed on
/// three rotary axes (<see cref="IVisionEncoder.AxisPositions"/>), the options <c>min_pixels</c> and <c>max_pixels</c>,
/// the encoder it builds, and, as <see cref="IVisionTuningPart"/>, the parts that train: the merger (as
/// <see cref="VisionTuningParts.Projector"/>) and the tower.
/// </summary>
public sealed class QwenVision : PretrainedVision, IVisionTuningPart
{
    private readonly Func<ITensorStore> _open;
    private readonly string? _folder;

    internal QwenVision(QwenVlVariant variant, QwenVisionConfig config, QwenImageTokens tokens, int textDim, string prefix,
        IReadOnlyDictionary<string, QwenTensor> tensors, Func<ITensorStore> open, string? folder)
    {
        (Variant, Config, ImageTokens, TextDim, Prefix, Tensors, _open, _folder) = (variant, config, tokens, textDim, prefix, tensors, open, folder);
        PromptFormat = new ImageTokenFormat("qwen-vl", tokens.ImagePad, tokens.ImagePad);
    }

    /// <summary>The vision option keys Qwen-VL takes: the smart resize's pixel budget.</summary>
    public static IReadOnlyCollection<string> Keys { get; } = [QwenImageProcessor.MinPixelsKey, QwenImageProcessor.MaxPixelsKey];

    /// <summary>Which family.</summary>
    public QwenVlVariant Variant { get; }

    /// <summary>The vision tower's configuration (config.json's <c>vision_config</c>).</summary>
    public QwenVisionConfig Config { get; }

    /// <summary>The ids of the vision markers and the image pad in the text.</summary>
    public QwenImageTokens ImageTokens { get; }

    /// <summary>The text decoder's width: the merger's output, one embedding per image token.</summary>
    public int TextDim { get; }

    /// <summary>Where the vision tensors are in the checkpoint (<c>visual.</c> or <c>model.visual.</c>).</summary>
    public string Prefix { get; }

    /// <summary>Every tower and merger tensor by its name under <see cref="Prefix"/>, with its stored name and shape.</summary>
    public IReadOnlyDictionary<string, QwenTensor> Tensors { get; }

    /// <inheritdoc />
    public override string Family => QwenVisionFamily.ArchitectureOf(Variant);

    /// <inheritdoc />
    public override int Width => TextDim;

    /// <inheritdoc />
    public override IImagePromptFormat PromptFormat { get; }

    /// <summary>Qwen-VL's image tokens attend causally, like text: <see cref="ImageAttentionRules.Causal"/>.</summary>
    public override IImageAttentionRule Attention => ImageAttentionRules.Get(ImageAttentionRules.Causal);

    /// <inheritdoc />
    public override IReadOnlyCollection<string> StoredTensors => [.. Tensors.Values.Select(t => t.Stored)];

    /// <inheritdoc />
    public override IReadOnlyCollection<string> VisionOptionKeys => Keys;

    /// <inheritdoc />
    public override string Describe() => $"{Family}: {Config}, {TextDim}-wide image tokens";

    /// <summary>
    /// The model's image processor: the model folder's <c>preprocessor_config.json</c> (keys it leaves out take
    /// Qwen2VLImageProcessor's defaults), its patch, merge and temporal sizes as vision_config's.
    /// </summary>
    /// <param name="grayscale">Turn images to grayscale first (three equal channels), as some fine-tunes ask.</param>
    public QwenImageProcessor Processor(bool grayscale = false) => QwenImageProcessor.FromFolder(_folder, Config, grayscale);

    /// <inheritdoc />
    /// <remarks>
    /// The parts <see cref="VisionEncoderOptions.TrainedParts"/> names are built with float32 weights, the other as
    /// <see cref="VisionEncoderOptions.Weights"/> says (a bfloat16 checkpoint's projections stay bfloat16 by default).
    /// </remarks>
    public override IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null)
    {
        var weights = options?.Weights ?? EncoderWeights.AsStored;
        EncoderWeights For(string part) => options?.Trains(part) == true ? EncoderWeights.Float32 : weights;
        var device = options?.Device ?? Device.Default;
        var processor = Processor(options?.Grayscale ?? false).With(options?.VisionOptions);
        using var tensors = OpenTensors();
        var tower = QwenVisionTower.FromTensors(Config, Variant, tensors, device, For(VisionTuningParts.Tower));
        try
        {
            var merger = QwenPatchMerger.FromTensors(Config, Variant, tensors, device, For(VisionTuningParts.Projector));
            return new QwenImageEncoder(this, tower, merger, processor, device);
        }
        catch
        {
            tower.Dispose();
            throw;
        }
    }

    /// <summary>Opens the checkpoint's vision tensors under the names of <see cref="Tensors"/> (as stored): dispose it when done.</summary>
    public ITensorStore OpenTensors() => new Models.RenamedTensorStore(_open(), Tensors.ToDictionary(t => t.Key, t => t.Value.Stored, StringComparer.Ordinal));

    /// <summary>The merger (<see cref="VisionTuningParts.Projector"/>) and the tower train; LoRA runs usually train the merger alone or neither.</summary>
    public IReadOnlyList<string> TrainableParts { get; } = [VisionTuningParts.Projector, VisionTuningParts.Tower];

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Parameters(IVisionEncoder encoder, string part)
    {
        ArgumentNullException.ThrowIfNull(part);
        var module = Part(Own(encoder), part);
        if (module.Buffers().Any())
        {
            throw new InvalidOperationException($"Qwen-VL's {Normalized(part)} holds packed (bfloat16) weights in this encoder, which do not train; "
                + "build the encoder with VisionEncoderOptions.TrainedParts naming it (or Weights = EncoderWeights.Float32).");
        }

        return [.. module.Parameters()];
    }

    /// <summary>The image tokens' embeddings [images, tokens, text width] of the tower's output [images, patches, tower width], recording gradients.</summary>
    public Tensor Features(IVisionEncoder encoder, Tensor towerOutput)
    {
        ArgumentNullException.ThrowIfNull(towerOutput);
        var own = Own(encoder);
        if (towerOutput.Device != own.Device)
        {
            throw new ArgumentException($"The tower's output is on {towerOutput.Device}, the encoder on {own.Device}: move it first (gradients do not cross a copy).", nameof(towerOutput));
        }

        return own.Merger.Forward(towerOutput);
    }

    /// <summary>The tower's output [images, patches, tower width] of pixel values [images, grid rows, grid columns, patch values], recording gradients.</summary>
    public Tensor Tower(IVisionEncoder encoder, Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(pixelValues);
        var own = Own(encoder);
        var input = pixelValues.Device == own.Device ? pixelValues : Tensor.From(pixelValues.ToArray(), pixelValues.Shape, own.Device);   // kept: the backward pass reads it
        return own.VisionTower.Forward(input);
    }

    /// <summary>The part's tensors under the checkpoint's names, as stored (Linear weights [out, in]; the patch embedding [dim, channels, temporal, patch, patch]).</summary>
    public IReadOnlyDictionary<string, Tensor> Export(IVisionEncoder encoder, string part)
    {
        _ = Parameters(encoder, part);                                                   // refuses an unknown part and packed weights
        var own = Own(encoder);
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, (tensor, transposed)) in Values(own, Normalized(part)))
        {
            var stored = Tensors[name];
            var values = tensor.ToArray();
            if (transposed)
            {
                values = Transpose(values, tensor.Shape[0], tensor.Shape[1]);           // [in, out] as held → [out, in] as stored
            }

            tensors[stored.Stored] = Tensor.From(values.AsSpan(), stored.Shape.ToArray(), Device.Cpu);
        }

        return tensors;
    }

    /// <summary>Sets the merger's or the tower's tensors from tensors named and laid out as <see cref="Export"/> gives them; returns the names taken.</summary>
    public IReadOnlyCollection<string> Import(IVisionEncoder encoder, IReadOnlyDictionary<string, Tensor> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var own = Own(encoder);
        var taken = new List<string>();
        foreach (string part in TrainableParts)
        {
            var values = Values(own, part).Where(v => tensors.ContainsKey(Tensors[v.Key].Stored)).ToList();
            if (values.Count == 0)
            {
                continue;
            }

            if (Part(own, part).Buffers().Any())
            {
                throw new InvalidOperationException($"Qwen-VL's {part} holds packed (bfloat16) weights in this encoder; build it with VisionEncoderOptions.TrainedParts naming it "
                    + "(or Weights = EncoderWeights.Float32) to load trained values.");
            }

            foreach (var (name, (target, transposed)) in values)
            {
                var stored = Tensors[name];
                var value = tensors[stored.Stored];
                if (!value.Shape.SequenceEqual(stored.Shape.ToArray()))
                {
                    throw new ArgumentException($"{stored.Stored}: shape {Tensor.FormatShape(value.Shape)}, the stored one is {Tensor.FormatShape([.. stored.Shape])}.", nameof(tensors));
                }

                var read = value.ToArray();
                target.Load(transposed ? Transpose(read, target.Shape[1], target.Shape[0]) : read);
                taken.Add(stored.Stored);
            }
        }

        return taken;
    }

    /// <summary>The merger's tensors are the projector's; every other vision tensor the tower's.</summary>
    public IReadOnlyList<string> PartsOf(IEnumerable<string> names)
    {
        ArgumentNullException.ThrowIfNull(names);
        var byStored = Tensors.ToDictionary(t => t.Value.Stored, t => t.Key, StringComparer.Ordinal);
        var parts = new List<string>();
        foreach (string name in names)
        {
            if (byStored.TryGetValue(name, out var own))
            {
                string part = own.StartsWith("merger.", StringComparison.Ordinal) ? VisionTuningParts.Projector : VisionTuningParts.Tower;
                if (!parts.Contains(part))
                {
                    parts.Add(part);
                }
            }
        }

        return parts;
    }

    /// <inheritdoc />
    public override string ToString() => Describe();

    // Every tensor of the vision part by its name under the prefix, with its stored shape, for one variant.
    internal static IEnumerable<(string Name, int[] Shape)> Expected(QwenVisionConfig c, QwenVlVariant variant)
    {
        int d = c.Dim, f = c.FfDim, m = c.MergeSize * c.MergeSize * d;
        bool layerNorm = variant == QwenVlVariant.Qwen2Vl;
        yield return ("patch_embed.proj.weight", [d, c.Channels, c.TemporalPatchSize, c.PatchSize, c.PatchSize]);
        for (int i = 0; i < c.Depth; i++)
        {
            string b = $"blocks.{i}.";
            foreach (string norm in (string[])["norm1", "norm2"])
            {
                yield return ($"{b}{norm}.weight", [d]);
                if (layerNorm)
                {
                    yield return ($"{b}{norm}.bias", [d]);
                }
            }

            yield return ($"{b}attn.qkv.weight", [3 * d, d]);
            yield return ($"{b}attn.qkv.bias", [3 * d]);
            yield return ($"{b}attn.proj.weight", [d, d]);
            yield return ($"{b}attn.proj.bias", [d]);
            if (layerNorm)
            {
                yield return ($"{b}mlp.fc1.weight", [f, d]);
                yield return ($"{b}mlp.fc1.bias", [f]);
                yield return ($"{b}mlp.fc2.weight", [d, f]);
                yield return ($"{b}mlp.fc2.bias", [d]);
            }
            else
            {
                foreach (string gate in (string[])["gate_proj", "up_proj"])
                {
                    yield return ($"{b}mlp.{gate}.weight", [f, d]);
                    yield return ($"{b}mlp.{gate}.bias", [f]);
                }

                yield return ($"{b}mlp.down_proj.weight", [d, f]);
                yield return ($"{b}mlp.down_proj.bias", [d]);
            }
        }

        yield return ("merger.ln_q.weight", [d]);
        if (layerNorm)
        {
            yield return ("merger.ln_q.bias", [d]);
        }

        yield return ("merger.mlp.0.weight", [m, m]);
        yield return ("merger.mlp.0.bias", [m]);
        yield return ("merger.mlp.2.weight", [c.OutDim, m]);
        yield return ("merger.mlp.2.bias", [c.OutDim]);
    }

    private QwenImageEncoder Own(IVisionEncoder encoder) => encoder is QwenImageEncoder own && ReferenceEquals(own.Vision, this)
        ? own
        : throw new ArgumentException("The encoder is not one this vision part made (QwenVision.CreateEncoder).", nameof(encoder));

    private static string Normalized(string part) => part.Trim().ToLowerInvariant();

    private Module Part(QwenImageEncoder own, string part) => Normalized(part) switch
    {
        VisionTuningParts.Projector => own.Merger,
        VisionTuningParts.Tower => own.VisionTower,
        _ => throw new NotSupportedException($"The vision family {Family} does not offer '{part}' for training; it offers: {string.Join(", ", TrainableParts)}."),
    };

    private static Dictionary<string, (Tensor Tensor, bool Transposed)> Values(QwenImageEncoder own, string part) =>
        Normalized(part) == VisionTuningParts.Projector ? own.Merger.Values() : own.VisionTower.Values();

    private static float[] Transpose(float[] values, int rows, int columns)
    {
        var result = new float[values.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < columns; c++)
            {
                result[c * rows + r] = values[r * columns + c];
            }
        }

        return result;
    }
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
/// The vision tower's configuration (<c>vision_config</c>), in names both families share. Qwen2.5-VL's keys:
/// <c>hidden_size</c>, <c>intermediate_size</c>, <c>out_hidden_size</c>, <c>window_size</c>,
/// <c>fullatt_block_indexes</c>; Qwen2-VL's: <c>embed_dim</c>, <c>mlp_ratio</c> and the output width in
/// <c>hidden_size</c>. Both: <c>depth</c>, <c>num_heads</c>, <c>in_channels</c> (the hub's <c>in_chans</c>),
/// <c>patch_size</c>, <c>spatial_merge_size</c>, <c>temporal_patch_size</c>, <c>hidden_act</c>, and the rotary base in
/// <c>rope_parameters.rope_theta</c> (10000 when absent). Absent keys take transformers' defaults.
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

    /// <summary>Input channels.</summary>
    public required int Channels { get; init; }

    /// <summary>The patch side in pixels (<c>patch_size</c>).</summary>
    public required int PatchSize { get; init; }

    /// <summary>Patches merged per side into one token (<c>spatial_merge_size</c>): a token is this² patches.</summary>
    public required int MergeSize { get; init; }

    /// <summary>Frames per temporal patch (<c>temporal_patch_size</c>); a still image is repeated to fill one.</summary>
    public required int TemporalPatchSize { get; init; }

    /// <summary>The merger's output width: the text decoder's width.</summary>
    public required int OutDim { get; init; }

    /// <summary>The feed-forward activation (<c>hidden_act</c>): silu (gated, Qwen2.5-VL) or quick_gelu (Qwen2-VL).</summary>
    public required string Activation { get; init; }

    /// <summary>The base of the tower's 2D rotary frequencies.</summary>
    public float RopeTheta { get; init; } = 10000f;

    /// <summary>Qwen2.5-VL: the window side in pixels of the window-attention blocks (<c>window_size</c>); 0 for full attention everywhere (Qwen2-VL).</summary>
    public int WindowSize { get; init; }

    /// <summary>Qwen2.5-VL: the blocks that attend over the whole image (<c>fullatt_block_indexes</c>); empty for Qwen2-VL.</summary>
    public IReadOnlyList<int> FullAttentionBlocks { get; init; } = [];

    /// <summary>A head's width.</summary>
    public int HeadDim => Dim / Heads;

    /// <summary>The values of one patch: channels × temporal × patch × patch.</summary>
    public int PatchValues => Channels * TemporalPatchSize * PatchSize * PatchSize;

    /// <summary>Merged tokens per window side (window_size / merge / patch), 0 without windows.</summary>
    public int WindowTokens => WindowSize / MergeSize / PatchSize;

    /// <summary>Reads <c>vision_config</c> for one family.</summary>
    /// <exception cref="InvalidDataException">The configuration is missing or its sizes do not fit together.</exception>
    public static QwenVisionConfig FromJson(JsonObject? config, QwenVlVariant variant)
    {
        if (config is null)
        {
            throw new InvalidDataException("config.json has no vision_config.");
        }

        int Int(string key, int otherwise) => (int?)config[key] ?? otherwise;
        float theta = (float?)(config["rope_parameters"] as JsonObject)?["rope_theta"] ?? 10000f;
        int channels = (int?)config["in_channels"] ?? (int?)config["in_chans"] ?? 3;
        QwenVisionConfig result = variant == QwenVlVariant.Qwen25Vl
            ? new()
            {
                Depth = Int("depth", 32), Dim = Int("hidden_size", 3584), FfDim = Int("intermediate_size", 3420), Heads = Int("num_heads", 16), Channels = channels,
                PatchSize = Int("patch_size", 14), MergeSize = Int("spatial_merge_size", 2), TemporalPatchSize = Int("temporal_patch_size", 2),
                OutDim = Int("out_hidden_size", 3584), Activation = (string?)config["hidden_act"] ?? "silu", RopeTheta = theta, WindowSize = Int("window_size", 112),
                FullAttentionBlocks = config["fullatt_block_indexes"] is JsonArray full ? [.. full.Select(i => (int)i!)] : [7, 15, 23, 31],
            }
            : new()
            {
                Depth = Int("depth", 32), Dim = Int("embed_dim", 1280), FfDim = Int("embed_dim", 1280) * Int("mlp_ratio", 4), Heads = Int("num_heads", 16), Channels = channels,
                PatchSize = Int("patch_size", 14), MergeSize = Int("spatial_merge_size", 2), TemporalPatchSize = Int("temporal_patch_size", 2),
                OutDim = Int("hidden_size", 3584), Activation = (string?)config["hidden_act"] ?? "quick_gelu", RopeTheta = theta,
            };
        if (result.Dim % result.Heads != 0 || result.HeadDim % 4 != 0)
        {
            throw new InvalidDataException($"vision_config: width {result.Dim} over {result.Heads} heads must give a head width divisible by 4 (two rotary axes of pairs).");
        }

        if (variant == QwenVlVariant.Qwen25Vl && result.WindowTokens < 1)
        {
            throw new InvalidDataException($"vision_config: window_size {result.WindowSize} is smaller than one merged token ({result.MergeSize} x {result.PatchSize} pixels).");
        }

        string expected = variant == QwenVlVariant.Qwen25Vl ? "silu" : "quick_gelu";
        if (result.Activation != expected)
        {
            throw new NotSupportedException($"vision_config: hidden_act '{result.Activation}' is not supported for {QwenVisionFamily.ArchitectureOf(variant)} (it uses {expected}).");
        }

        return result;
    }

    /// <inheritdoc />
    public override string ToString() => $"{Depth} blocks of {Dim} ({Heads} heads), patch {PatchSize}, merge {MergeSize}, out {OutDim}"
        + (WindowSize > 0 ? $", window {WindowSize} px (full in {string.Join(", ", FullAttentionBlocks)})" : "");
}
