// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Models.Abstractions;

namespace Idrak.Gemma3Vision;

/// <summary>
/// The Gemma 3 vision family as a registration of Idrak's vision contracts (plan 11, "Vision contracts"): the library
/// knows no family; this assembly brings one. <see cref="Register"/> adds Gemma 3's vision part
/// (<see cref="Gemma3VisionFamily"/>, under <c>Gemma3ForConditionalGeneration</c> in <see cref="VisionFamilies"/>) and
/// its attention rule (<see cref="ImageBlocks"/> in <see cref="ImageAttentionRules"/>). The text decoder of
/// <c>Gemma3ForConditionalGeneration</c> is one of the library's text families. Load it in the command line with
/// <c>idrak -P Idrak.Gemma3Vision.dll ...</c> (or the config's "plugins"), or call <see cref="Register"/> in an app.
/// </summary>
public static class Gemma3VisionPlugin
{
    /// <summary>The attention rule Gemma 3's image tokens use: each image's tokens see their whole image, causal elsewhere.</summary>
    public const string ImageBlocks = "image-blocks";

    /// <summary>Registers the Gemma 3 vision family and its attention rule (again: the same registrations).</summary>
    public static void Register()
    {
        ImageAttentionRules.Register(new ImageBlockRule());
        VisionFamilies.Register(new Gemma3VisionFamily());
    }

    /// <summary>The command line's plug-in entry (<c>-P</c>): <see cref="Register"/>.</summary>
    public static void RegisterIdrakPlugin() => Register();

    // Gemma 3's mask (transformers' token_type_ids): a row in an image block sees its whole block, causal elsewhere, within
    // the window on sliding-window layers (KeySpans.ImageBlocks, which the conformance kit checks on every device).
    private sealed class ImageBlockRule : IImageAttentionRule
    {
        public string Name => ImageBlocks;

        public bool Causal => false;

        public KeySpans Spans(int rows, IReadOnlyList<(int Start, int Length)> blocks, int window) => KeySpans.ImageBlocks(rows, blocks, window);

        public override string ToString() => "image blocks seen whole";
    }
}

/// <summary>
/// Reads the vision part of a <c>Gemma3ForConditionalGeneration</c> checkpoint (<see cref="IVisionFamily"/>): SigLIP's
/// configuration, the image token ids (Gemma3Config's defaults when absent), and every encoder and projector tensor in
/// whichever of the three namings the checkpoint uses, its shape checked against the configuration.
/// </summary>
public sealed class Gemma3VisionFamily : IVisionFamily
{
    /// <summary>The architecture name: <c>Gemma3ForConditionalGeneration</c>.</summary>
    public const string Architecture = "Gemma3ForConditionalGeneration";

    // The three namings of the vision part (plan 11): transformers 4.52 to 4.57 rename the modules in memory but save under
    // the old names; version 5 drops the vision model's own level.
    private static readonly (string Name, string Vision, string Projector)[] Layouts =
    [
        ("save_pretrained of transformers 4.x (vision_tower.vision_model.*)", "vision_tower.vision_model.", "multi_modal_projector."),
        ("state dict of transformers 4.52 to 4.57 (model.vision_tower.vision_model.*)", "model.vision_tower.vision_model.", "model.multi_modal_projector."),
        ("save_pretrained of transformers 5 (vision_tower.*)", "vision_tower.", "multi_modal_projector."),
    ];

    /// <inheritdoc />
    public string Name => Architecture;

    /// <inheritdoc />
    public PretrainedVision? Read(VisionCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var store = checkpoint.Tensors;
        var config = checkpoint.Config;
        var layout = Layouts.FirstOrDefault(l => store.Contains(l.Vision + "embeddings.patch_embedding.weight"));
        if (layout.Name is null)
        {
            checkpoint.Notes.Add("The checkpoint has no vision encoder: the model reads text only.");
            return null;
        }

        var encoder = SiglipVisionConfig.FromJson(config["vision_config"] as JsonObject);
        var tokens = new Gemma3ImageTokens((int?)config["boi_token_index"] ?? 255_999, (int?)config["eoi_token_index"] ?? 256_000,
            (int?)config["image_token_index"] ?? 262_144, (int?)config["mm_tokens_per_image"] ?? 256);
        int side = (int)Math.Round(Math.Sqrt(tokens.TokensPerImage));
        if (side == 0 || side * side != tokens.TokensPerImage || encoder.PatchesPerSide % side != 0)
        {
            throw new InvalidDataException($"mm_tokens_per_image {tokens.TokensPerImage} is not a square grid that divides the {encoder.PatchesPerSide} x {encoder.PatchesPerSide} patches.");
        }

        if (encoder.UseHead)
        {
            checkpoint.Notes.Add("The vision encoder's pooling head (vision_use_head) is not used: Gemma 3 projects every patch.");
        }

        // The text decoder's width, as the text family registered under the same name reads it.
        int textDim = PretrainedArchitectures.Get(checkpoint.Architecture).Spec(config, []).Dim, d = encoder.Dim, f = encoder.FfDim;
        var expected = new List<(string Name, int[] Shape)>
        {
            ("vision.embeddings.patch_embedding.weight", [d, encoder.Channels, encoder.PatchSize, encoder.PatchSize]),
            ("vision.embeddings.patch_embedding.bias", [d]),
            ("vision.embeddings.position_embedding.weight", [encoder.Patches, d]),
        };
        for (int i = 0; i < encoder.Layers; i++)
        {
            string l = $"vision.encoder.layers.{i}.";
            foreach (string norm in (string[])["layer_norm1", "layer_norm2"])
            {
                expected.Add(($"{l}{norm}.weight", [d]));
                expected.Add(($"{l}{norm}.bias", [d]));
            }

            foreach (string projection in (string[])["q_proj", "k_proj", "v_proj", "out_proj"])
            {
                expected.Add(($"{l}self_attn.{projection}.weight", [d, d]));
                expected.Add(($"{l}self_attn.{projection}.bias", [d]));
            }

            expected.Add(($"{l}mlp.fc1.weight", [f, d]));
            expected.Add(($"{l}mlp.fc1.bias", [f]));
            expected.Add(($"{l}mlp.fc2.weight", [d, f]));
            expected.Add(($"{l}mlp.fc2.bias", [d]));
        }

        expected.Add(("vision.post_layernorm.weight", [d]));
        expected.Add(("vision.post_layernorm.bias", [d]));
        expected.Add(("projector.mm_soft_emb_norm.weight", [d]));
        expected.Add(("projector.mm_input_projection_weight", [d, textDim]));      // [vision, text], used as x · W

        var tensors = new Dictionary<string, VisionTensor>(StringComparer.Ordinal);
        foreach (var (name, shape) in expected)
        {
            string stored = name.StartsWith("vision.", StringComparison.Ordinal) ? layout.Vision + name["vision.".Length..] : layout.Projector + name["projector.".Length..];
            if (!store.Contains(stored))
            {
                throw new InvalidDataException($"The checkpoint ({layout.Name}) has no '{stored}', which vision_config asks for.");
            }

            var actual = store.ShapeOf(stored);
            if (!actual.SequenceEqual(shape))
            {
                throw new InvalidDataException($"'{stored}' is [{string.Join(", ", actual)}]; vision_config and text_config make it [{string.Join(", ", shape)}].");
            }

            tensors[name] = new VisionTensor(stored, actual);
        }

        return new Gemma3Vision(encoder, tokens, textDim, layout.Name, tensors, checkpoint.Open, checkpoint.Folder);
    }
}

/// <summary>
/// The vision part of a Gemma 3 checkpoint, as data: SigLIP's configuration, the image token ids, and the encoder's and
/// projector's tensors under one naming whatever layout the checkpoint was saved in. <see cref="CreateEncoder(Device?, ImagePreprocessor?, Gemma3PanAndScan?, EncoderWeights)"/>
/// builds the encoder and the projector from it.
/// </summary>
/// <remarks>
/// Tensor names: <c>vision.</c> followed by SigLIP's own names inside its vision model
/// (<c>vision.embeddings.patch_embedding.weight</c> [width, channels, patch, patch], <c>vision.embeddings.position_embedding.weight</c>
/// [patches, width], <c>vision.encoder.layers.N.{layer_norm1, self_attn.{q,k,v,out}_proj, layer_norm2, mlp.fc1, mlp.fc2}.{weight,bias}</c>
/// with Linear weights stored [out, in], <c>vision.post_layernorm.{weight,bias}</c>), and <c>projector.</c> followed by the
/// projector's (<c>projector.mm_soft_emb_norm.weight</c> [width], Gemma's RMSNorm with a gain of 1 + w;
/// <c>projector.mm_input_projection_weight</c> [vision width, text width], used as x · W: it is the transpose of a Linear
/// weight, so it is read as stored).
/// <para>
/// For fine-tuning (<see cref="IVisionTuningPart"/>), Gemma 3 lets its projector and its SigLIP tower train (full weights,
/// saved under the checkpoint's names: the tower's fused q, k, v projection split back into q_proj, k_proj and v_proj),
/// each from an encoder that holds that part in float32 (<see cref="VisionEncoderOptions.TrainedParts"/>,
/// <see cref="EncoderWeights.Float32"/>, or a float32 checkpoint as stored).
/// </para>
/// </remarks>
public sealed class Gemma3Vision : PretrainedVision, IVisionTuningPart
{
    private readonly Func<ITensorStore> _open;
    private readonly string? _folder;

    internal Gemma3Vision(SiglipVisionConfig encoder, Gemma3ImageTokens tokens, int textDim, string layout, IReadOnlyDictionary<string, VisionTensor> tensors,
        Func<ITensorStore> open, string? folder)
    {
        (Encoder, ImageTokens, TextDim, Layout, Tensors, _open, _folder) = (encoder, tokens, textDim, layout, tensors, open, folder);
        // Gemma3Processor: each <start_of_image> the template writes becomes "\n\n" + boi + soft tokens + eoi + "\n\n" (its
        // full_image_sequence); with pan and scan, an image of crops becomes "Here is the original image " + the whole
        // image's block + " and here are some crops to help you see better " + the crops' blocks joined by spaces.
        PromptFormat = new ImageTokenFormat("gemma3", tokens.BeginImage, tokens.ImageToken, tokens.BeginImage, tokens.EndImage, "\n\n", "\n\n")
        {
            Join = blocks => $"Here is the original image {blocks[0]} and here are some crops to help you see better " + string.Join(" ", blocks.Skip(1)),
        };
        PanAndScan = _folder is { } path && File.Exists(Path.Combine(path, "preprocessor_config.json"))
            ? Gemma3PanAndScan.FromConfig(File.ReadAllText(Path.Combine(path, "preprocessor_config.json")))
            : Gemma3PanAndScan.Default;
    }

    /// <inheritdoc />
    public override string Family => Gemma3VisionFamily.Architecture;

    /// <inheritdoc />
    public override int Width => TextDim;

    /// <inheritdoc />
    public override IImagePromptFormat PromptFormat { get; }

    /// <summary>Gemma 3's image tokens attend by <see cref="Gemma3VisionPlugin.ImageBlocks"/> (as registered: an app may take that name's place).</summary>
    public override IImageAttentionRule Attention => ImageAttentionRules.Get(Gemma3VisionPlugin.ImageBlocks);

    /// <inheritdoc />
    public override IReadOnlyCollection<string> StoredTensors => [.. Tensors.Values.Select(t => t.Stored)];

    /// <summary>Gemma 3's vision options: pan and scan's (<see cref="Gemma3PanAndScan.Keys"/>).</summary>
    public override IReadOnlyCollection<string> VisionOptionKeys => Gemma3PanAndScan.Keys;

    /// <summary>
    /// Pan and scan as the model folder's <c>preprocessor_config.json</c> sets it (its null or absent keys:
    /// Gemma3Processor's defaults, off): every encoder's defaults, under the options it is built with and each request's.
    /// </summary>
    public Gemma3PanAndScan PanAndScan { get; }

    /// <summary>The vision encoder's configuration (config.json's <c>vision_config</c>).</summary>
    public SiglipVisionConfig Encoder { get; }

    /// <summary>The ids of the image tokens in the text and how many soft tokens one image becomes.</summary>
    public Gemma3ImageTokens ImageTokens { get; }

    /// <summary>The text decoder's width: the projector's output, one embedding per soft token.</summary>
    public int TextDim { get; }

    /// <summary>
    /// The projector's average pooling: kernel and stride over the grid of patch outputs, so that the
    /// <see cref="SiglipVisionConfig.PatchesPerSide"/>² patches become <see cref="Gemma3ImageTokens.TokensPerImage"/> tokens.
    /// </summary>
    public int PoolSize => Encoder.PatchesPerSide / (int)Math.Round(Math.Sqrt(ImageTokens.TokensPerImage));

    /// <summary>The epsilon of the projector's RMSNorm: the vision encoder's <c>layer_norm_eps</c>, as transformers uses.</summary>
    public float ProjectorNormEpsilon => Encoder.LayerNormEpsilon;

    /// <summary>Which naming the checkpoint uses (for messages).</summary>
    public string Layout { get; }

    /// <summary>Every encoder and projector tensor by its name here (see the remarks), with its name in the checkpoint and its shape.</summary>
    public IReadOnlyDictionary<string, VisionTensor> Tensors { get; }

    /// <summary>
    /// The image preprocessing this model expects: the model folder's <c>preprocessor_config.json</c> (keys it leaves out
    /// take Gemma3ImageProcessor's defaults), or, without one, Gemma 3's processor at the encoder's image size (a square,
    /// bilinear, scaled by 1/255, mean and std 0.5, RGB): this family's decision, nobody else's.
    /// </summary>
    /// <param name="grayscale">Turn images to grayscale first (Pillow's <c>convert("L")</c>, then three equal channels), as some fine-tunes ask.</param>
    public ImagePreprocessor Preprocessor(bool grayscale = false)
    {
        var defaults = new ImagePreprocessor { Height = Encoder.ImageSize, Width = Encoder.ImageSize, Resampling = ImageResampling.Bilinear, Mean = [0.5f, 0.5f, 0.5f], Std = [0.5f, 0.5f, 0.5f] };
        return _folder is { } folder && File.Exists(Path.Combine(folder, "preprocessor_config.json"))
            ? ImagePreprocessor.FromConfig(folder, grayscale, defaults)
            : new ImagePreprocessor { Height = defaults.Height, Width = defaults.Width, Resampling = defaults.Resampling, Mean = defaults.Mean, Std = defaults.Std, Grayscale = grayscale };
    }

    /// <inheritdoc />
    /// <remarks>
    /// The parts <see cref="VisionEncoderOptions.TrainedParts"/> names (projector, tower) are built with float32 weights,
    /// the other as <see cref="VisionEncoderOptions.Weights"/> says: a trained projector over a bfloat16 checkpoint keeps
    /// SigLIP's projections in bfloat16.
    /// </remarks>
    public override IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null)
    {
        var weights = options?.Weights ?? EncoderWeights.AsStored;
        EncoderWeights For(string part) => options?.Trains(part) == true ? EncoderWeights.Float32 : weights;
        return Build(options?.Device, Preprocessor(options?.Grayscale ?? false), PanAndScan.With(options?.VisionOptions), For(VisionTuningParts.Tower),
            For(VisionTuningParts.Projector));
    }

    /// <summary>
    /// Builds the vision encoder and the projector on <paramref name="device"/> from the checkpoint's tensors (read once):
    /// a <see cref="Gemma3ImageEncoder"/> that turns images into the soft tokens' embeddings, computing in float32. Its
    /// projection weights are kept as the checkpoint stores them unless <paramref name="weights"/> says otherwise (a
    /// bfloat16 checkpoint's in bfloat16: 0.83 GB less for gemma-3-4b's 417 M encoder and projector weights, the same
    /// features). Dispose it when done.
    /// </summary>
    /// <param name="device">Where it runs (default <see cref="Device.Default"/>; pass the language model's device).</param>
    /// <param name="preprocessor">How images become pixel values (default <see cref="Preprocessor"/>).</param>
    /// <param name="panAndScan">Pan and scan for every image (default <see cref="PanAndScan"/>); a request's options go over it.</param>
    /// <param name="weights">The projection weights' precision (<see cref="EncoderWeights.Float32"/> forces float32).</param>
    public Gemma3ImageEncoder CreateEncoder(Device? device, ImagePreprocessor? preprocessor = null, Gemma3PanAndScan? panAndScan = null,
        EncoderWeights weights = EncoderWeights.AsStored) => Build(device, preprocessor, panAndScan, weights, weights);

    // The encoder with the tower's and the projector's weights each as asked.
    private Gemma3ImageEncoder Build(Device? device, ImagePreprocessor? preprocessor, Gemma3PanAndScan? panAndScan, EncoderWeights tower, EncoderWeights projectorWeights)
    {
        device ??= Device.Default;
        using var tensors = OpenTensors();
        var encoder = SiglipVisionEncoder.FromTensors(Encoder, tensors, "vision.", device, tower);
        try
        {
            var projector = Gemma3Projector.FromTensors(tensors, Encoder.Dim, TextDim, PoolSize, ProjectorNormEpsilon, "projector.", device, projectorWeights);
            return new Gemma3ImageEncoder(this, encoder, projector, preprocessor ?? Preprocessor(), panAndScan ?? PanAndScan);
        }
        catch
        {
            encoder.Dispose();
            throw;
        }
    }

    /// <summary>
    /// Opens the checkpoint's vision tensors under the names of <see cref="Tensors"/>, as stored (no transposition): read
    /// them one at a time and dispose the store when done.
    /// </summary>
    public ITensorStore OpenTensors() => new RenamedTensorStore(_open(), Tensors.ToDictionary(t => t.Key, t => t.Value.Stored, StringComparer.Ordinal));

    /// <summary>Gemma 3 lets its projector (the soft-embedding norm and the projection) and its SigLIP tower train.</summary>
    public IReadOnlyList<string> TrainableParts { get; } = [VisionTuningParts.Projector, VisionTuningParts.Tower];

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Parameters(IVisionEncoder encoder, string part)
    {
        ArgumentNullException.ThrowIfNull(part);
        var own = Own(encoder);
        Module module = part.Trim().ToLowerInvariant() switch
        {
            VisionTuningParts.Projector => own.Projector,
            VisionTuningParts.Tower => own.Encoder,
            _ => throw new NotSupportedException($"The vision family {Family} does not offer '{part}' for training; it offers: {string.Join(", ", TrainableParts)}."),
        };
        if (module.Buffers().Any())
        {
            throw new InvalidOperationException($"Gemma 3's {part.Trim().ToLowerInvariant()} holds packed (bfloat16) weights in this encoder, which do not train; "
                + "build the encoder with VisionEncoderOptions.Weights = EncoderWeights.Float32 to train it.");
        }

        return [.. module.Parameters()];
    }

    /// <summary>The soft-token embeddings [images, tokens per image, text width] of SigLIP's output [images, patches, vision width] through the projector, recording gradients.</summary>
    public Tensor Features(IVisionEncoder encoder, Tensor towerOutput)
    {
        ArgumentNullException.ThrowIfNull(towerOutput);
        var own = Own(encoder);
        if (towerOutput.Device != own.Device)
        {
            throw new ArgumentException($"The tower's output is on {towerOutput.Device}, the encoder on {own.Device}: move it first (gradients do not cross a copy).", nameof(towerOutput));
        }

        return own.Projector.Forward(towerOutput);
    }

    /// <summary>
    /// SigLIP's output [images, patches, vision width] for pixel values [images, channels, size, size], recording gradients
    /// (copied to the encoder's device first when elsewhere); each encoder layer checkpointed while
    /// <see cref="ActivationMemory.CheckpointsBlocks"/> is on.
    /// </summary>
    public Tensor Tower(IVisionEncoder encoder, Tensor pixelValues)
    {
        ArgumentNullException.ThrowIfNull(pixelValues);
        var own = Own(encoder);
        bool checkpointed = ActivationMemory.CheckpointsBlocks;
        if (pixelValues.Device == own.Device)
        {
            return own.Encoder.ForwardTraining(pixelValues, checkpointed);
        }

        var moved = Tensor.From(pixelValues.ToArray(), pixelValues.Shape, own.Device);         // kept: the backward pass reads it
        return own.Encoder.ForwardTraining(moved, checkpointed);
    }

    /// <summary>
    /// The part's tensors under the checkpoint's names, as the checkpoint stores them. The projector: its two tensors
    /// (<c>…multi_modal_projector.mm_soft_emb_norm.weight</c>, the gain w of 1 + w, and <c>…mm_input_projection_weight</c>,
    /// W of x · W). The tower: every SigLIP tensor (the patch convolution and position embedding, each layer's norms, its
    /// q, k and v projections split from the fused projection the encoder holds, its output projection and MLP with Linear
    /// weights [out, in], and the final norm).
    /// </summary>
    public IReadOnlyDictionary<string, Tensor> Export(IVisionEncoder encoder, string part)
    {
        ArgumentNullException.ThrowIfNull(part);
        var own = Own(encoder);
        _ = Parameters(encoder, part);                                                   // refuses an unknown part and packed weights
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        if (string.Equals(part.Trim(), VisionTuningParts.Projector, StringComparison.OrdinalIgnoreCase))
        {
            tensors[Tensors[NormTensor].Stored] = Host(own.Projector.Gain.ToArray(), [.. own.Projector.Gain.Shape]);
            tensors[Tensors[ProjectionTensor].Stored] = Host(own.Projector.Weight.ToArray(), [.. own.Projector.Weight.Shape]);
            return tensors;
        }

        foreach (var (name, (values, _)) in TowerValues(own.Encoder))
        {
            int[] stored = [.. Tensors[name].Shape];                                     // the convolution's [dim, channels, patch, patch], not the layer's flat rows
            if (values.Length != stored.Aggregate(1, (a, b) => a * b))
            {
                throw new InvalidOperationException($"{Tensors[name].Stored}: {values.Length} values for the stored shape {Tensor.FormatShape(stored)}.");
            }

            tensors[Tensors[name].Stored] = Host(values, stored);
        }

        return tensors;
    }

    /// <summary>
    /// Sets the projector's or the tower's tensors from tensors named and laid out as <see cref="Export"/> gives them
    /// (q, k and v go into the fused projection, each where it lies); returns the names taken.
    /// </summary>
    public IReadOnlyCollection<string> Import(IVisionEncoder encoder, IReadOnlyDictionary<string, Tensor> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var own = Own(encoder);
        var taken = new List<string>();
        foreach (var (name, read) in new (string, Func<Tensor>)[] { (Tensors[NormTensor].Stored, () => own.Projector.Gain), (Tensors[ProjectionTensor].Stored, () => own.Projector.Weight) })
        {
            if (!tensors.TryGetValue(name, out var value))
            {
                continue;
            }

            if (own.Projector.Buffers().Any())
            {
                throw new InvalidOperationException("Gemma 3's projector holds packed (bfloat16) weights in this encoder; build it with VisionEncoderOptions.TrainedParts naming the projector "
                    + "(or Weights = EncoderWeights.Float32) to load trained values.");
            }

            var target = read();
            if (!value.Shape.SequenceEqual(target.Shape))
            {
                throw new ArgumentException($"{name}: shape {Tensor.FormatShape(value.Shape)}, the projector's is {Tensor.FormatShape(target.Shape)}.", nameof(tensors));
            }

            target.Load(value.ToArray());
            taken.Add(name);
        }

        var tower = TowerTensorNames().Where(n => tensors.ContainsKey(Tensors[n].Stored)).ToList();
        if (tower.Count > 0)
        {
            if (own.Encoder.Buffers().Any())
            {
                throw new InvalidOperationException("Gemma 3's tower holds packed (bfloat16) weights in this encoder; build it with VisionEncoderOptions.TrainedParts naming the tower "
                    + "(or Weights = EncoderWeights.Float32) to load trained values.");
            }

            foreach (string name in tower)
            {
                var value = tensors[Tensors[name].Stored];
                int[] expected = [.. Tensors[name].Shape];
                if (!value.Shape.SequenceEqual(expected))
                {
                    throw new ArgumentException($"{Tensors[name].Stored}: shape {Tensor.FormatShape(value.Shape)}, the tower's is {Tensor.FormatShape(expected)}.", nameof(tensors));
                }
            }

            SetTower(own.Encoder, tower.ToDictionary(n => n, n => tensors[Tensors[n].Stored].ToArray(), StringComparer.Ordinal));
            taken.AddRange(tower.Select(n => Tensors[n].Stored));
        }

        return taken;
    }

    /// <summary>The projector's tensors are the projector's; SigLIP's (<c>vision.</c>…) the tower's.</summary>
    public IReadOnlyList<string> PartsOf(IEnumerable<string> names)
    {
        ArgumentNullException.ThrowIfNull(names);
        var projector = new HashSet<string>([Tensors[NormTensor].Stored, Tensors[ProjectionTensor].Stored], StringComparer.Ordinal);
        var tower = new HashSet<string>(TowerTensorNames().Select(n => Tensors[n].Stored), StringComparer.Ordinal);
        var parts = new List<string>();
        foreach (string name in names)
        {
            string? part = projector.Contains(name) ? VisionTuningParts.Projector : tower.Contains(name) ? VisionTuningParts.Tower : null;
            if (part is not null && !parts.Contains(part))
            {
                parts.Add(part);
            }
        }

        return parts;
    }

    // SigLIP's tensors under this part's names (vision.…).
    private IEnumerable<string> TowerTensorNames() => Tensors.Keys.Where(k => k.StartsWith("vision.", StringComparison.Ordinal)).Order(StringComparer.Ordinal);

    // Every SigLIP tensor of the encoder by its name here, in the checkpoint's layout: Linear weights [in, out] transposed
    // to [out, in], the fused [dim, 3·dim] q, k, v projection split into q_proj, k_proj and v_proj.
    private static IEnumerable<(string Name, (float[] Values, int[] Shape))> TowerValues(SiglipVisionEncoder tower)
    {
        int dim = tower.Config.Dim;
        yield return ("vision.embeddings.patch_embedding.weight", (tower.Patches.Weight.ToArray(), [.. tower.Patches.Weight.Shape]));
        yield return ("vision.embeddings.patch_embedding.bias", (tower.Patches.Bias!.ToArray(), [dim]));
        yield return ("vision.embeddings.position_embedding.weight", (tower.PositionEmbedding.ToArray(), [.. tower.PositionEmbedding.Shape]));
        for (int i = 0; i < tower.LayerWeights.Count; i++)
        {
            var w = tower.LayerWeights[i];
            string l = $"vision.encoder.layers.{i}";
            yield return ($"{l}.layer_norm1.weight", (w.Norm1.Gamma.ToArray(), [dim]));
            yield return ($"{l}.layer_norm1.bias", (w.Norm1.Beta.ToArray(), [dim]));
            float[] qkv = w.Qkv.Weight.ToArray(), qkvBias = w.Qkv.Bias!.ToArray();
            string[] parts = ["q_proj", "k_proj", "v_proj"];
            for (int part = 0; part < 3; part++)
            {
                var weight = new float[dim * dim];
                for (int input = 0; input < dim; input++)
                {
                    for (int output = 0; output < dim; output++)
                    {
                        weight[output * dim + input] = qkv[input * 3 * dim + part * dim + output];
                    }
                }

                yield return ($"{l}.self_attn.{parts[part]}.weight", (weight, [dim, dim]));
                yield return ($"{l}.self_attn.{parts[part]}.bias", (qkvBias[(part * dim)..((part + 1) * dim)], [dim]));
            }

            foreach (var (name, layer) in new[] { ("self_attn.out_proj", w.Output), ("mlp.fc1", w.Fc1), ("mlp.fc2", w.Fc2) })
            {
                yield return ($"{l}.{name}.weight", (Transpose(layer.Weight.ToArray(), layer.InFeatures, layer.OutFeatures), [layer.OutFeatures, layer.InFeatures]));
                yield return ($"{l}.{name}.bias", (layer.Bias!.ToArray(), [layer.OutFeatures]));
            }

            yield return ($"{l}.layer_norm2.weight", (w.Norm2.Gamma.ToArray(), [dim]));
            yield return ($"{l}.layer_norm2.bias", (w.Norm2.Beta.ToArray(), [dim]));
        }

        yield return ("vision.post_layernorm.weight", (tower.PostNorm.Gamma.ToArray(), [dim]));
        yield return ("vision.post_layernorm.bias", (tower.PostNorm.Beta.ToArray(), [dim]));
    }

    // Puts SigLIP tensors (names here, the checkpoint's layout) into the encoder: the inverse of TowerValues.
    private static void SetTower(SiglipVisionEncoder tower, Dictionary<string, float[]> values)
    {
        int dim = tower.Config.Dim;
        void Set(string name, Tensor target, Func<float[], float[]>? layout = null)
        {
            if (values.TryGetValue(name, out var v))
            {
                target.Load(layout is null ? v : layout(v));
            }
        }

        Set("vision.embeddings.patch_embedding.weight", tower.Patches.Weight);
        Set("vision.embeddings.patch_embedding.bias", tower.Patches.Bias!);
        Set("vision.embeddings.position_embedding.weight", tower.PositionEmbedding);
        for (int i = 0; i < tower.LayerWeights.Count; i++)
        {
            var w = tower.LayerWeights[i];
            string l = $"vision.encoder.layers.{i}";
            Set($"{l}.layer_norm1.weight", w.Norm1.Gamma);
            Set($"{l}.layer_norm1.bias", w.Norm1.Beta);
            string[] parts = ["q_proj", "k_proj", "v_proj"];
            if (parts.Any(p => values.ContainsKey($"{l}.self_attn.{p}.weight") || values.ContainsKey($"{l}.self_attn.{p}.bias")))
            {
                float[] qkv = w.Qkv.Weight.ToArray(), qkvBias = w.Qkv.Bias!.ToArray();
                for (int part = 0; part < 3; part++)
                {
                    if (values.TryGetValue($"{l}.self_attn.{parts[part]}.weight", out var weight))
                    {
                        for (int input = 0; input < dim; input++)
                        {
                            for (int output = 0; output < dim; output++)
                            {
                                qkv[input * 3 * dim + part * dim + output] = weight[output * dim + input];
                            }
                        }
                    }

                    if (values.TryGetValue($"{l}.self_attn.{parts[part]}.bias", out var bias))
                    {
                        bias.CopyTo(qkvBias, part * dim);
                    }
                }

                w.Qkv.Weight.Load(qkv);
                w.Qkv.Bias!.Load(qkvBias);
            }

            foreach (var (name, layer) in new[] { ("self_attn.out_proj", w.Output), ("mlp.fc1", w.Fc1), ("mlp.fc2", w.Fc2) })
            {
                Set($"{l}.{name}.weight", layer.Weight, v => Transpose(v, layer.OutFeatures, layer.InFeatures));
                Set($"{l}.{name}.bias", layer.Bias!);
            }

            Set($"{l}.layer_norm2.weight", w.Norm2.Gamma);
            Set($"{l}.layer_norm2.bias", w.Norm2.Beta);
        }

        Set("vision.post_layernorm.weight", tower.PostNorm.Gamma);
        Set("vision.post_layernorm.bias", tower.PostNorm.Beta);
    }

    // [rows, columns] row-major to [columns, rows].
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

    private static Tensor Host(float[] values, int[] shape) => Tensor.From(values, shape, Device.Cpu);

    private const string NormTensor = "projector.mm_soft_emb_norm.weight", ProjectionTensor = "projector.mm_input_projection_weight";

    // The encoder, when this vision part built it.
    private Gemma3ImageEncoder Own(IVisionEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        return encoder is Gemma3ImageEncoder own && ReferenceEquals(own.Vision, this) ? own
            : throw new ArgumentException($"{encoder.GetType().Name} is not an encoder of this Gemma 3 vision part (make one with its CreateEncoder).", nameof(encoder));
    }

    /// <inheritdoc />
    public override string Describe() =>
        $"Gemma 3: SigLIP {Encoder.ImageSize} x {Encoder.ImageSize}, {Encoder.Layers} layers of {Encoder.Dim}, {ImageTokens.TokensPerImage} tokens per image, {PanAndScan}";

    // A store showing some tensors of another under other names.
    private sealed class RenamedTensorStore(ITensorStore inner, Dictionary<string, string> names) : ITensorStore
    {
        public IEnumerable<string> Names => names.Keys;

        public bool Contains(string name) => names.ContainsKey(name);

        public int[] ShapeOf(string name) => inner.ShapeOf(Stored(name));

        public float[] Read(string name) => inner.Read(Stored(name));

        public float[] ReadTransposed(string name) => inner.ReadTransposed(Stored(name));

        public WeightFormat? FormatOf(string name) => inner.FormatOf(Stored(name));

        public void Dispose() => inner.Dispose();

        private string Stored(string name) => names.TryGetValue(name, out var stored) ? stored : throw new KeyNotFoundException($"No vision tensor '{name}'.");
    }
}

/// <summary>One tensor of <see cref="Gemma3Vision.Tensors"/>.</summary>
/// <param name="Stored">Its name in the checkpoint.</param>
/// <param name="Shape">Its shape as stored, outermost first.</param>
public sealed record VisionTensor(string Stored, IReadOnlyList<int> Shape);

/// <summary>Gemma 3's image tokens (config.json's top level).</summary>
/// <param name="BeginImage">The token before an image's soft tokens (<c>boi_token_index</c>, <c>&lt;start_of_image&gt;</c>), also the template's marker.</param>
/// <param name="EndImage">The token after them (<c>eoi_token_index</c>, <c>&lt;end_of_image&gt;</c>).</param>
/// <param name="ImageToken">The soft token whose embedding is replaced by an image feature (<c>image_token_index</c>, <c>&lt;image_soft_token&gt;</c>).</param>
/// <param name="TokensPerImage">How many soft tokens one image becomes (<c>mm_tokens_per_image</c>).</param>
public sealed record Gemma3ImageTokens(int BeginImage, int EndImage, int ImageToken, int TokensPerImage);

/// <summary>
/// A SigLIP vision encoder's configuration (a Hugging Face <c>vision_config</c>, <c>siglip_vision_model</c>): patches of
/// <see cref="PatchSize"/> pixels over a square image, a learned position embedding, <see cref="Layers"/> pre-LayerNorm
/// layers with bidirectional attention, a final LayerNorm. Absent keys take transformers' <c>SiglipVisionConfig</c> defaults.
/// </summary>
public sealed record SiglipVisionConfig
{
    /// <summary>The encoder's width (<c>hidden_size</c>).</summary>
    public required int Dim { get; init; }

    /// <summary>The MLP's hidden size (<c>intermediate_size</c>).</summary>
    public required int FfDim { get; init; }

    /// <summary>How many encoder layers (<c>num_hidden_layers</c>).</summary>
    public required int Layers { get; init; }

    /// <summary>Attention heads (<c>num_attention_heads</c>).</summary>
    public required int Heads { get; init; }

    /// <summary>The square image's side in pixels (<c>image_size</c>).</summary>
    public required int ImageSize { get; init; }

    /// <summary>A patch's side in pixels (<c>patch_size</c>), also the convolution's stride.</summary>
    public required int PatchSize { get; init; }

    /// <summary>Color channels (<c>num_channels</c>).</summary>
    public required int Channels { get; init; }

    /// <summary>The LayerNorms' epsilon (<c>layer_norm_eps</c>).</summary>
    public required float LayerNormEpsilon { get; init; }

    /// <summary>The MLP's activation as named in the configuration (<c>hidden_act</c>; "gelu_pytorch_tanh" is GELU with the tanh approximation).</summary>
    public required string Activation { get; init; }

    /// <summary>Whether the model has SigLIP's pooling head (<c>vision_use_head</c>; Gemma 3 has none).</summary>
    public bool UseHead { get; init; }

    /// <summary>Each head's width.</summary>
    public int HeadDim => Dim / Heads;

    /// <summary>Patches along one side of the image.</summary>
    public int PatchesPerSide => ImageSize / PatchSize;

    /// <summary>Patches in the image (the encoder's sequence length).</summary>
    public int Patches => PatchesPerSide * PatchesPerSide;

    /// <summary>Reads a <c>vision_config</c> (transformers' <c>SiglipVisionConfig</c> defaults for absent keys).</summary>
    public static SiglipVisionConfig FromJson(JsonObject? config)
    {
        config ??= new JsonObject();
        var result = new SiglipVisionConfig
        {
            Dim = (int?)config["hidden_size"] ?? 768,
            FfDim = (int?)config["intermediate_size"] ?? 3072,
            Layers = (int?)config["num_hidden_layers"] ?? 12,
            Heads = (int?)config["num_attention_heads"] ?? 12,
            ImageSize = (int?)config["image_size"] ?? 224,
            PatchSize = (int?)config["patch_size"] ?? 16,
            Channels = (int?)config["num_channels"] ?? 3,
            LayerNormEpsilon = (float?)config["layer_norm_eps"] ?? 1e-6f,
            Activation = (string?)config["hidden_act"] ?? "gelu_pytorch_tanh",
            UseHead = (bool?)config["vision_use_head"] ?? false,
        };
        if (result.Dim <= 0 || result.Heads <= 0 || result.Dim % result.Heads != 0)
        {
            throw new InvalidDataException($"vision_config: a width of {result.Dim} does not split into {result.Heads} heads.");
        }

        if (result.PatchSize <= 0 || result.ImageSize % result.PatchSize != 0)
        {
            throw new InvalidDataException($"vision_config: {result.ImageSize}-pixel images do not split into {result.PatchSize}-pixel patches.");
        }

        return result;
    }
}
