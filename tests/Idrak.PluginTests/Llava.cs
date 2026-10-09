// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Data;
using Idrak.Layers;
using Idrak.Models;
using Idrak.Models.Abstractions;

namespace Idrak.PluginTests;

/// <summary>
/// LLaVA (<c>LlavaForConditionalGeneration</c>) as a registration from outside the library, on its public API only (plan
/// 11, "Vision contracts": the second family, proving the contracts general): a CLIP vision tower with a class token and
/// quick-GELU layers, the hidden states of <c>vision_feature_layer</c> (one layer or several, concatenated) with
/// <c>vision_feature_select_strategy</c> ("default" drops the class token), a two-layer MLP projector, a Llama-style text
/// decoder read with the library's text family, image tokens attending causally (the library's causal rule), and
/// LlavaImageProcessor's preprocessing (resize by the shortest edge, bicubic, center crop, CLIP's mean and std).
/// </summary>
public static class LlavaPlugin
{
    /// <summary>The architecture name.</summary>
    public const string Architecture = "LlavaForConditionalGeneration";

    /// <summary>Registers LLaVA's text decoder (<see cref="PretrainedArchitectures"/>) and its vision family (<see cref="VisionFamilies"/>).</summary>
    public static void Register()
    {
        PretrainedArchitectures.Register(Architecture, LlavaText.Architecture(LlavaText.Layouts[0]));
        VisionFamilies.Register(new LlavaVisionFamily());
    }

    /// <summary>Removes both registrations.</summary>
    public static void Unregister()
    {
        PretrainedArchitectures.Unregister(Architecture);
        VisionFamilies.Unregister(Architecture);
    }
}

// LLaVA's text decoder: text_config read by the library's text family of its model_type, under the checkpoint's naming.
internal static class LlavaText
{
    public static readonly (string Text, string Head)[] Layouts = [("language_model.model.", "language_model.lm_head."), ("model.language_model.", "lm_head.")];

    private static readonly Dictionary<string, string> TextFamilies = new(StringComparer.Ordinal)
    {
        ["llama"] = "LlamaForCausalLM", ["mistral"] = "MistralForCausalLM", ["qwen2"] = "Qwen2ForCausalLM",
    };

    // transformers' LlamaConfig defaults, for the keys a sparse text_config leaves out (llava-1.5's names only a few).
    private const string LlamaDefaults = """
        {"vocab_size": 32000, "hidden_size": 4096, "intermediate_size": 11008, "num_hidden_layers": 32, "num_attention_heads": 32,
         "hidden_act": "silu", "max_position_embeddings": 2048, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": false}
        """;

    public static PretrainedArchitecture Architecture((string Text, string Head) layout) => new()
    {
        Spec = (config, notes) =>
        {
            var (family, text) = TextConfig(config);
            return PretrainedArchitectures.Get(family).Spec(text, notes);
        },
        TensorName = name => PretrainedFamilies.LlamaTensorName(name) switch
        {
            null => null,
            var stored when stored.StartsWith("model.", StringComparison.Ordinal) => layout.Text + stored["model.".Length..],
            var stored => layout.Head + stored["lm_head.".Length..],
        },
        ForCheckpoint = names => Architecture(Layouts.FirstOrDefault(l => names.Contains(l.Text + "embed_tokens.weight")) is { Text: not null } found ? found
            : throw new InvalidDataException("This LLaVA checkpoint has no text decoder under a known name.")),
    };

    // The text family's name and its config: text_config, LlamaConfig's defaults for a llama one, and transformers 5's
    // rope_parameters read as rope_theta and rope_scaling.
    public static (string Family, JsonObject Text) TextConfig(JsonObject config)
    {
        var text = config["text_config"] is JsonObject given ? (JsonObject)given.DeepClone() : new JsonObject { ["model_type"] = "llama" };
        string type = (string?)text["model_type"] ?? "llama";
        if (!TextFamilies.TryGetValue(type, out var family))
        {
            throw new NotSupportedException($"LLaVA's text_config model_type '{type}' is not read here ({string.Join(", ", TextFamilies.Keys)}).");
        }

        if (type == "llama")
        {
            foreach (var (key, value) in JsonNode.Parse(LlamaDefaults)!.AsObject())
            {
                if (!text.ContainsKey(key))
                {
                    text[key] = value!.DeepClone();
                }
            }
        }

        if (text["rope_parameters"] is JsonObject rope)
        {
            if (rope["rope_theta"] is { } theta)
            {
                text["rope_theta"] = theta.DeepClone();
            }

            if (((string?)rope["rope_type"] ?? "default") != "default")
            {
                var scaling = (JsonObject)rope.DeepClone();
                scaling.Remove("rope_theta");
                text["rope_scaling"] = scaling;
            }
        }

        return (family, text);
    }
}

/// <summary>A CLIP vision tower's configuration (<c>clip_vision_model</c>; transformers' CLIPVisionConfig defaults for absent keys).</summary>
public sealed record ClipVisionConfig
{
    /// <summary>The tower's width.</summary>
    public int Dim { get; init; } = 768;

    /// <summary>The MLP's hidden size.</summary>
    public int FfDim { get; init; } = 3072;

    /// <summary>Encoder layers.</summary>
    public int Layers { get; init; } = 12;

    /// <summary>Attention heads.</summary>
    public int Heads { get; init; } = 12;

    /// <summary>The square image's side.</summary>
    public int ImageSize { get; init; } = 224;

    /// <summary>A patch's side.</summary>
    public int PatchSize { get; init; } = 32;

    /// <summary>Colour channels.</summary>
    public int Channels { get; init; } = 3;

    /// <summary>The LayerNorms' epsilon.</summary>
    public float LayerNormEpsilon { get; init; } = 1e-5f;

    /// <summary>The MLP's activation (<c>hidden_act</c>).</summary>
    public string Activation { get; init; } = "quick_gelu";

    /// <summary>Patches in the image (the class token is one more position).</summary>
    public int Patches => (ImageSize / PatchSize) * (ImageSize / PatchSize);

    /// <summary>Reads a <c>vision_config</c>.</summary>
    public static ClipVisionConfig FromJson(JsonObject? json)
    {
        json ??= [];
        var d = new ClipVisionConfig();
        return new ClipVisionConfig
        {
            Dim = (int?)json["hidden_size"] ?? d.Dim, FfDim = (int?)json["intermediate_size"] ?? d.FfDim, Layers = (int?)json["num_hidden_layers"] ?? d.Layers,
            Heads = (int?)json["num_attention_heads"] ?? d.Heads, ImageSize = (int?)json["image_size"] ?? d.ImageSize, PatchSize = (int?)json["patch_size"] ?? d.PatchSize,
            Channels = (int?)json["num_channels"] ?? d.Channels, LayerNormEpsilon = (float?)json["layer_norm_eps"] ?? d.LayerNormEpsilon,
            Activation = (string?)json["hidden_act"] ?? d.Activation,
        };
    }

    /// <summary>An activation by its transformers name, as a layer.</summary>
    public static Module ActivationLayer(string name) => name switch
    {
        "quick_gelu" => new QuickGELU(),
        "gelu" => new ExactGELU(),
        "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" => new GELU(),
        _ => throw new NotSupportedException($"Activation '{name}' is not read here."),
    };
}

/// <summary>Reads the vision part of a LLaVA checkpoint.</summary>
public sealed class LlavaVisionFamily : IVisionFamily
{
    // The vision tower and projector namings: transformers 5 drops the vision model's own level; 4.52 to 4.57 rename in memory.
    private static readonly (string Name, string Vision, string Projector)[] Layouts =
    [
        ("save_pretrained of transformers 5 (vision_tower.*)", "vision_tower.", "multi_modal_projector."),
        ("save_pretrained of transformers 4.x (vision_tower.vision_model.*)", "vision_tower.vision_model.", "multi_modal_projector."),
        ("state dict of transformers 4.52 to 4.57 (model.vision_tower.vision_model.*)", "model.vision_tower.vision_model.", "model.multi_modal_projector."),
    ];

    /// <inheritdoc />
    public string Name => LlavaPlugin.Architecture;

    /// <inheritdoc />
    public PretrainedVision? Read(VisionCheckpoint checkpoint)
    {
        ArgumentNullException.ThrowIfNull(checkpoint);
        var config = checkpoint.Config;
        var store = checkpoint.Tensors;
        var layout = Layouts.FirstOrDefault(l => store.Contains(l.Vision + "embeddings.patch_embedding.weight"));
        if (layout.Name is null)
        {
            checkpoint.Notes.Add("The checkpoint has no vision tower: the model reads text only.");
            return null;
        }

        var clip = ClipVisionConfig.FromJson(config["vision_config"] as JsonObject);
        int[] layers = config["vision_feature_layer"] switch
        {
            JsonArray list => [.. list.Select(n => (int)n!)],
            { } one => [(int)one],
            null => [-2],
        };
        int[] hidden = [.. layers.Select(l => l < 0 ? clip.Layers + 1 + l : l)];            // hidden_states has Layers + 1 entries
        if (hidden.Any(l => l < 0 || l > clip.Layers))
        {
            throw new InvalidDataException($"vision_feature_layer {string.Join(", ", layers)} is outside the tower's {clip.Layers} layers.");
        }

        string strategy = (string?)config["vision_feature_select_strategy"] ?? "default";
        if (strategy is not ("default" or "full"))
        {
            throw new NotSupportedException($"vision_feature_select_strategy '{strategy}' is not read here.");
        }

        int textDim = PretrainedArchitectures.Get(checkpoint.Architecture).Spec(config, []).Dim;
        int d = clip.Dim, f = clip.FfDim, input = d * hidden.Length;
        bool bias = (bool?)config["multimodal_projector_bias"] ?? true;
        var expected = new List<(string Name, int[] Shape)>
        {
            ($"{layout.Vision}embeddings.class_embedding", [d]),
            ($"{layout.Vision}embeddings.patch_embedding.weight", [d, clip.Channels, clip.PatchSize, clip.PatchSize]),
            ($"{layout.Vision}embeddings.position_embedding.weight", [clip.Patches + 1, d]),
            ($"{layout.Vision}pre_layrnorm.weight", [d]), ($"{layout.Vision}pre_layrnorm.bias", [d]),
            ($"{layout.Projector}linear_1.weight", [textDim, input]), ($"{layout.Projector}linear_2.weight", [textDim, textDim]),
        };
        if (bias)
        {
            expected.Add(($"{layout.Projector}linear_1.bias", [textDim]));
            expected.Add(($"{layout.Projector}linear_2.bias", [textDim]));
        }

        for (int i = 0; i < clip.Layers; i++)
        {
            string l = $"{layout.Vision}encoder.layers.{i}.";
            foreach (string part in (string[])["layer_norm1", "layer_norm2"])
            {
                expected.Add(($"{l}{part}.weight", [d]));
                expected.Add(($"{l}{part}.bias", [d]));
            }

            foreach (string part in (string[])["q_proj", "k_proj", "v_proj", "out_proj"])
            {
                expected.Add(($"{l}self_attn.{part}.weight", [d, d]));
                expected.Add(($"{l}self_attn.{part}.bias", [d]));
            }

            expected.Add(($"{l}mlp.fc1.weight", [f, d]));
            expected.Add(($"{l}mlp.fc1.bias", [f]));
            expected.Add(($"{l}mlp.fc2.weight", [d, f]));
            expected.Add(($"{l}mlp.fc2.bias", [d]));
        }

        foreach (var (name, shape) in expected)
        {
            if (!store.Contains(name) || !store.ShapeOf(name).SequenceEqual(shape))
            {
                throw new InvalidDataException($"The LLaVA checkpoint ({layout.Name}) has no '{name}' of shape [{string.Join(", ", shape)}].");
            }
        }

        // The whole tower is the vision part's (post_layernorm and layers past the chosen ones are read by nobody).
        var stored = store.Names.Where(n => n.StartsWith(layout.Vision, StringComparison.Ordinal) || n.StartsWith(layout.Projector, StringComparison.Ordinal)).ToList();
        int token = (int?)config["image_token_index"] ?? (int?)config["image_token_id"] ?? 32000;
        return new LlavaVision(clip, hidden, strategy == "full", textDim, token, (string?)config["projector_hidden_act"] ?? "gelu", bias, layout.Vision, layout.Projector,
            stored, checkpoint.Open, checkpoint.Folder);
    }
}

/// <summary>A LLaVA checkpoint's vision part; for fine-tuning (<see cref="IVisionTuningPart"/>) it lets its projector train.</summary>
public sealed class LlavaVision : PretrainedVision, IVisionTuningPart
{
    private readonly Func<ITensorStore> _open;
    private readonly string? _folder;
    private readonly string _vision, _projector;

    internal LlavaVision(ClipVisionConfig clip, int[] hiddenLayers, bool full, int textDim, int imageToken, string projectorActivation, bool projectorBias,
        string vision, string projector, IReadOnlyCollection<string> stored, Func<ITensorStore> open, string? folder)
    {
        (Clip, HiddenLayers, Full, Width, ProjectorActivation, ProjectorBias, _vision, _projector, StoredTensors, _open, _folder) =
            (clip, hiddenLayers, full, textDim, projectorActivation, projectorBias, vision, projector, stored, open, folder);
        // LlavaProcessor: each <image> the template writes becomes as many <image> tokens as the image has features.
        PromptFormat = new ImageTokenFormat("llava", imageToken, imageToken);
        int side = clip.ImageSize / clip.PatchSize;
        Layout = full ? new ImageTokenLayout(clip.Patches + 1) : new ImageTokenLayout(clip.Patches) { Grid = [side, side] };
    }

    /// <summary>The vision tower's configuration.</summary>
    public ClipVisionConfig Clip { get; }

    /// <summary>The hidden states taken (0: the embeddings after pre_layrnorm, k: after k layers), concatenated in this order.</summary>
    public IReadOnlyList<int> HiddenLayers { get; }

    /// <summary>Whether the class token is kept ("full"); "default" drops it.</summary>
    public bool Full { get; }

    /// <summary>Every image's token layout (a fixed size: the tower's patches, plus the class token when kept).</summary>
    public ImageTokenLayout Layout { get; }

    /// <summary>The projector's activation.</summary>
    public string ProjectorActivation { get; }

    /// <summary>Whether the projector's layers have biases.</summary>
    public bool ProjectorBias { get; }

    /// <inheritdoc />
    public override string Family => LlavaPlugin.Architecture;

    /// <inheritdoc />
    public override int Width { get; }

    /// <inheritdoc />
    public override IImagePromptFormat PromptFormat { get; }

    /// <summary>LLaVA's image tokens attend as text does.</summary>
    public override IImageAttentionRule Attention => ImageAttentionRules.Get(ImageAttentionRules.Causal);

    /// <inheritdoc />
    public override IReadOnlyCollection<string> StoredTensors { get; }

    /// <summary>
    /// LlavaImageProcessor's steps: the folder's preprocessor_config.json over the processor's defaults (bicubic, shortest
    /// edge and center crop at the tower's image size, CLIP's mean and std), or those defaults without the file.
    /// </summary>
    public ImagePreprocessor Preprocessor(bool grayscale = false)
    {
        var defaults = new ImagePreprocessor
        {
            Resampling = ImageResampling.Bicubic, ShortestEdge = Clip.ImageSize, CenterCrop = true, CropHeight = Clip.ImageSize, CropWidth = Clip.ImageSize,
            Mean = [0.48145466f, 0.4578275f, 0.40821073f], Std = [0.26862954f, 0.26130258f, 0.27577711f], Grayscale = grayscale,
        };
        return _folder is { } folder && File.Exists(Path.Combine(folder, "preprocessor_config.json")) ? ImagePreprocessor.FromConfig(folder, grayscale, defaults) : defaults;
    }

    /// <summary>LLaVA lets its projector (the two-layer MLP) train; its CLIP tower stays frozen.</summary>
    public IReadOnlyList<string> TrainableParts { get; } = [VisionTuningParts.Projector];

    /// <inheritdoc />
    public IReadOnlyList<Tensor> Parameters(IVisionEncoder encoder, string part)
    {
        ArgumentNullException.ThrowIfNull(part);
        var own = Own(encoder);
        return string.Equals(part.Trim(), VisionTuningParts.Projector, StringComparison.OrdinalIgnoreCase) ? [.. own.Projector.Parameters()]
            : throw new NotSupportedException($"The vision family {Family} does not offer '{part}' for training; it offers: {string.Join(", ", TrainableParts)}.");
    }

    /// <summary>
    /// The features [images, tokens, width] of the selected hidden states [images, patches + 1, width · layers] (as the
    /// encoder's <c>Tower</c> gives them): the class token dropped for "default", then the projector, recording gradients.
    /// </summary>
    public Tensor Features(IVisionEncoder encoder, Tensor towerOutput)
    {
        ArgumentNullException.ThrowIfNull(towerOutput);
        var own = Own(encoder);
        if (towerOutput.Rank != 3 || towerOutput.Shape[1] != Clip.Patches + 1)
        {
            throw new ArgumentException($"LLaVA's tower output is [images, {Clip.Patches + 1}, width], got {Tensor.FormatShape(towerOutput.Shape)}.", nameof(towerOutput));
        }

        return own.Projector.Forward(Full ? towerOutput : towerOutput.Narrow(1, 1, Clip.Patches));
    }

    /// <summary>LLaVA does not offer its tower for training.</summary>
    /// <exception cref="NotSupportedException">Always: the message names the parts offered.</exception>
    public Tensor Tower(IVisionEncoder encoder, Tensor pixelValues) =>
        throw new NotSupportedException($"The vision family {Family} does not offer '{VisionTuningParts.Tower}' for training; it offers: {string.Join(", ", TrainableParts)}.");

    /// <summary>The projector's linear_1 and linear_2 (weight [out, in] and bias) under the checkpoint's names, as stored.</summary>
    public IReadOnlyDictionary<string, Tensor> Export(IVisionEncoder encoder, string part)
    {
        _ = Parameters(encoder, part);                                                   // refuses another part
        var own = Own(encoder);
        var tensors = new Dictionary<string, Tensor>(StringComparer.Ordinal);
        foreach (var (name, layer) in ProjectorLayers(own))
        {
            float[] w = layer.Weight.ToArray();
            int inputs = layer.InFeatures, outputs = layer.OutFeatures;
            var stored = new float[w.Length];
            for (int i = 0; i < inputs; i++)
            {
                for (int o = 0; o < outputs; o++)
                {
                    stored[o * inputs + i] = w[i * outputs + o];
                }
            }

            tensors[$"{_projector}{name}.weight"] = Tensor.From(stored, [outputs, inputs], Device.Cpu);
            if (layer.Bias is { } bias)
            {
                tensors[$"{_projector}{name}.bias"] = Tensor.From(bias.ToArray(), [outputs], Device.Cpu);
            }
        }

        return tensors;
    }

    /// <summary>Sets the projector's layers from tensors named and laid out as <see cref="Export"/> gives them; returns the names taken.</summary>
    public IReadOnlyCollection<string> Import(IVisionEncoder encoder, IReadOnlyDictionary<string, Tensor> tensors)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        var own = Own(encoder);
        var taken = new List<string>();
        foreach (var (name, layer) in ProjectorLayers(own))
        {
            int inputs = layer.InFeatures, outputs = layer.OutFeatures;
            if (tensors.TryGetValue($"{_projector}{name}.weight", out var weight))
            {
                if (!weight.Shape.SequenceEqual([outputs, inputs]))
                {
                    throw new ArgumentException($"{_projector}{name}.weight: shape {Tensor.FormatShape(weight.Shape)}, expected [{outputs}, {inputs}].", nameof(tensors));
                }

                float[] stored = weight.ToArray();
                var values = new float[stored.Length];
                for (int o = 0; o < outputs; o++)
                {
                    for (int i = 0; i < inputs; i++)
                    {
                        values[i * outputs + o] = stored[o * inputs + i];
                    }
                }

                layer.Weight.Load(values);
                taken.Add($"{_projector}{name}.weight");
            }

            if (layer.Bias is { } bias && tensors.TryGetValue($"{_projector}{name}.bias", out var b))
            {
                if (!b.Shape.SequenceEqual(bias.Shape))
                {
                    throw new ArgumentException($"{_projector}{name}.bias: shape {Tensor.FormatShape(b.Shape)}, expected {Tensor.FormatShape(bias.Shape)}.", nameof(tensors));
                }

                bias.Load(b.ToArray());
                taken.Add($"{_projector}{name}.bias");
            }
        }

        return taken;
    }

    private static (string Name, Linear Layer)[] ProjectorLayers(LlavaImageEncoder own) => [("linear_1", own.Projector.First), ("linear_2", own.Projector.Second)];

    private LlavaImageEncoder Own(IVisionEncoder encoder)
    {
        ArgumentNullException.ThrowIfNull(encoder);
        return encoder is LlavaImageEncoder own && ReferenceEquals(own.Vision, this) ? own
            : throw new ArgumentException($"{encoder.GetType().Name} is not an encoder of this LLaVA vision part.", nameof(encoder));
    }

    /// <inheritdoc />
    public override IVisionEncoder CreateEncoder(VisionEncoderOptions? options = null)
    {
        var device = options?.Device ?? Device.Default;
        using var store = _open();
        return new LlavaImageEncoder(this, ClipVisionTower.FromTensors(Clip, HiddenLayers.Max(), store, _vision, device),
            LlavaProjector.FromTensors(store, _projector, Clip.Dim * HiddenLayers.Count, Width, ProjectorActivation, ProjectorBias, device),
            Preprocessor(options?.Grayscale ?? false), device);
    }
}

/// <summary>A CLIP vision tower up to a given layer, giving the hidden states LLaVA selects.</summary>
public sealed class ClipVisionTower : Module
{
    private readonly ClipVisionConfig _config;
    private readonly Conv2d _patches;
    private readonly LayerNorm _preNorm;
    private readonly TransformerEncoderLayer[] _layers;
    private Tensor _class, _positions;

    private ClipVisionTower(ClipVisionConfig config, Conv2d patches, Tensor classEmbedding, Tensor positions, LayerNorm preNorm, TransformerEncoderLayer[] layers)
    {
        (_config, _patches, _class, _positions, _preNorm, _layers) = (config, patches, classEmbedding, positions, preNorm, layers);
    }

    /// <summary>Builds the tower's first <paramref name="layers"/> layers from CLIP's tensors under <paramref name="prefix"/>.</summary>
    public static ClipVisionTower FromTensors(ClipVisionConfig config, int layers, ITensorStore store, string prefix, Device device)
    {
        int d = config.Dim, f = config.FfDim;
        Tensor Read(string name, int[] shape, bool transposed = false) =>
            Tensor.Persistent(transposed ? store.ReadTransposed(prefix + name) : store.Read(prefix + name), shape, device);
        Linear Dense(string name, int inputs, int outputs) => Linear.FromWeights(Read($"{name}.weight", [inputs, outputs], true), Read($"{name}.bias", [outputs]));
        LayerNorm Norm(string name) => LayerNorm.FromWeights(Read($"{name}.weight", [d]), Read($"{name}.bias", [d]), config.LayerNormEpsilon);

        var patches = Conv2d.FromWeights(Read("embeddings.patch_embedding.weight", [d, config.Channels, config.PatchSize, config.PatchSize]), null, config.PatchSize, config.PatchSize);
        var built = new TransformerEncoderLayer[layers];
        for (int i = 0; i < layers; i++)
        {
            string l = $"encoder.layers.{i}";
            // q, k and v side by side in one [d, 3d] projection, as MultiHeadAttention reads them.
            var weight = new float[d * 3 * d];
            var bias = new float[3 * d];
            string[] parts = ["q_proj", "k_proj", "v_proj"];
            for (int p = 0; p < 3; p++)
            {
                var w = store.ReadTransposed($"{prefix}{l}.self_attn.{parts[p]}.weight");
                for (int row = 0; row < d; row++)
                {
                    Array.Copy(w, row * d, weight, row * 3 * d + p * d, d);
                }

                store.Read($"{prefix}{l}.self_attn.{parts[p]}.bias").CopyTo(bias, p * d);
            }

            var qkv = Linear.FromWeights(Tensor.Persistent(weight, [d, 3 * d], device), Tensor.Persistent(bias, [3 * d], device));
            var attention = MultiHeadAttention.FromWeights(qkv, Dense($"{l}.self_attn.out_proj", d, d), config.Heads);
            built[i] = TransformerEncoderLayer.FromLayers(Norm($"{l}.layer_norm1"), attention, Norm($"{l}.layer_norm2"), Dense($"{l}.mlp.fc1", d, f), Dense($"{l}.mlp.fc2", f, d),
                ClipVisionConfig.ActivationLayer(config.Activation));
        }

        return new ClipVisionTower(config, patches, Read("embeddings.class_embedding", [1, 1, d]), Read("embeddings.position_embedding.weight", [config.Patches + 1, d]),
            Norm("pre_layrnorm"), built);
    }

    /// <summary>
    /// The hidden states <paramref name="indices"/> (0: the embeddings after pre_layrnorm; k: after layer k) of pixel values
    /// [images, channels, size, size], concatenated along the width: [images, patches + 1, width · indices].
    /// </summary>
    public Tensor HiddenStates(Tensor pixels, IReadOnlyList<int> indices)
    {
        int n = pixels.Shape[0], d = _config.Dim, p = _config.Patches;
        var patches = _patches.Forward(pixels).Reshape(n, d, p).Permute(0, 2, 1);                // [n, patches, d]
        var classes = Tensor.Concat([.. Enumerable.Repeat(_class, n)], 0);                       // [n, 1, d]
        var x = _preNorm.Forward(Tensor.Concat([classes, patches], 1) + _positions);
        var states = new Dictionary<int, Tensor> { [0] = x };
        for (int i = 0; i < _layers.Length; i++)
        {
            x = _layers[i].Forward(x);
            states[i + 1] = x;
        }

        return indices.Count == 1 ? states[indices[0]] : Tensor.Concat([.. indices.Select(i => states[i])], 2);
    }

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => HiddenStates(input, [_layers.Length]);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_patches, _preNorm, .. _layers];

    /// <inheritdoc />
    public override IEnumerable<Tensor> Parameters() => [.. base.Parameters(), _class, _positions];

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        base.MoveTo(device);
        _class = MoveTensor(_class, device);
        _positions = MoveTensor(_positions, device);
    }
}

/// <summary>LLaVA's projector: linear, activation, linear.</summary>
public sealed class LlavaProjector : Module
{
    private readonly Linear _first, _second;
    private readonly Module _activation;

    private LlavaProjector(Linear first, Module activation, Linear second) => (_first, _activation, _second) = (first, activation, second);

    /// <summary>Builds it from <c>linear_1</c> and <c>linear_2</c> under <paramref name="prefix"/>.</summary>
    public static LlavaProjector FromTensors(ITensorStore store, string prefix, int inputs, int width, string activation, bool bias, Device device)
    {
        Linear Dense(string name, int i, int o) => Linear.FromWeights(Tensor.Persistent(store.ReadTransposed($"{prefix}{name}.weight"), [i, o], device),
            bias ? Tensor.Persistent(store.Read($"{prefix}{name}.bias"), [o], device) : null);
        return new LlavaProjector(Dense("linear_1", inputs, width), ClipVisionConfig.ActivationLayer(activation), Dense("linear_2", width, width));
    }

    /// <summary>linear_1.</summary>
    public Linear First => _first;

    /// <summary>linear_2.</summary>
    public Linear Second => _second;

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => _second.Forward(_activation.Forward(_first.Forward(input)));

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_first, _activation, _second];
}

/// <summary>LLaVA's image side as Idrak's <see cref="IVisionEncoder"/>.</summary>
public sealed class LlavaImageEncoder : IVisionEncoder, IVisionEncoderStages
{
    internal LlavaImageEncoder(LlavaVision vision, ClipVisionTower tower, LlavaProjector projector, ImagePreprocessor preprocessor, Device device) =>
        (Vision, ClipTower, Projector, Preprocessor, Device) = (vision, tower, projector, preprocessor, device);

    /// <summary>The vision part it was built from.</summary>
    public LlavaVision Vision { get; }

    /// <summary>The CLIP tower.</summary>
    public ClipVisionTower ClipTower { get; }

    /// <summary>The projector.</summary>
    public LlavaProjector Projector { get; }

    /// <summary>How images become pixel values.</summary>
    public ImagePreprocessor Preprocessor { get; }

    /// <inheritdoc />
    public int Width => Vision.Width;

    /// <inheritdoc />
    public Device Device { get; }

    /// <inheritdoc />
    public IReadOnlyList<ImageTokenLayout> Blocks(ImageData image, VisionOptions? options = null)
    {
        options?.ThrowIfUnknown(LlavaPlugin.Architecture, []);                          // LLaVA's processor takes none
        return [Vision.Layout];
    }

    /// <inheritdoc />
    public IReadOnlyList<ImageFeatures> Encode(IReadOnlyList<ImageData> images, VisionOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(images);
        options?.ThrowIfUnknown(LlavaPlugin.Architecture, []);
        int size = Vision.Clip.ImageSize, each = 3 * size * size;
        var values = new float[images.Count * each];
        for (int i = 0; i < images.Count; i++)
        {
            var pixels = Preprocessor.Pixels(images[i]);
            if (pixels.Length != each)
            {
                throw new InvalidOperationException($"The preprocessor gives {pixels.Length} pixel values; the tower takes 3 x {size} x {size}.");
            }

            pixels.CopyTo(values, i * each);
        }

        using var input = Tensor.From(values, [images.Count, 3, size, size], Device);
        using var batch = Features(input);
        var tokens = Vision.Layout.Tokens;
        return [.. Enumerable.Range(0, images.Count).Select(i => new ImageFeatures(batch.Narrow(0, i, 1).Reshape(tokens, Width), Vision.Layout))];
    }

    /// <inheritdoc />
    public Tensor PixelValues(ImageData image, VisionOptions? options = null)
    {
        options?.ThrowIfUnknown(LlavaPlugin.Architecture, []);
        using var one = Preprocessor.Process(image, Device.Cpu);
        return Tensor.From(one.ToArray(), [1, .. one.Shape], Device.Cpu);
    }

    /// <inheritdoc />
    public Tensor Tower(Tensor pixelValues)
    {
        using var noGrad = Autograd.NoGrad();
        return ClipTower.HiddenStates(pixelValues, [.. Vision.HiddenLayers]);
    }

    /// <inheritdoc />
    public Tensor Features(Tensor pixelValues)
    {
        using var noGrad = Autograd.NoGrad();
        using var scope = new TensorScope();
        var selected = ClipTower.HiddenStates(pixelValues, [.. Vision.HiddenLayers]);
        if (!Vision.Full)
        {
            selected = selected.Narrow(1, 1, Vision.Clip.Patches);                         // "default": the class token dropped
        }

        return scope.Keep(Projector.Forward(selected));
    }

    /// <inheritdoc />
    public void Dispose()
    {
        ClipTower.Dispose();
        Projector.Dispose();
    }
}
