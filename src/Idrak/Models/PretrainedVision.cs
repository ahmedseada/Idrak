// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Models.Abstractions;

namespace Idrak.Models;

/// <summary>
/// The vision part of a vision-language checkpoint (Gemma 3's <c>Gemma3ForConditionalGeneration</c>), as data: the
/// encoder's configuration, the image token ids, and the encoder's and projector's tensors under one naming whatever
/// layout the checkpoint was saved in, their shapes already checked against the configuration. The loaded model
/// (<see cref="PretrainedModel.Vision"/>) carries it; the encoder and the projector are built from it.
/// </summary>
/// <remarks>
/// Tensor names: <c>vision.</c> followed by SigLIP's own names inside its vision model
/// (<c>vision.embeddings.patch_embedding.weight</c> [width, channels, patch, patch], <c>vision.embeddings.position_embedding.weight</c>
/// [patches, width], <c>vision.encoder.layers.N.{layer_norm1, self_attn.{q,k,v,out}_proj, layer_norm2, mlp.fc1, mlp.fc2}.{weight,bias}</c>
/// with Linear weights stored [out, in], <c>vision.post_layernorm.{weight,bias}</c>), and <c>projector.</c> followed by the
/// projector's (<c>projector.mm_soft_emb_norm.weight</c> [width], Gemma's RMSNorm with a gain of 1 + w;
/// <c>projector.mm_input_projection_weight</c> [vision width, text width], used as x · W: it is the transpose of a Linear
/// weight, so it is read as stored).
/// </remarks>
public sealed record PretrainedVision
{
    /// <summary>The vision encoder's configuration (config.json's <c>vision_config</c>).</summary>
    public required VisionEncoderConfig Encoder { get; init; }

    /// <summary>The ids of the image tokens in the text and how many soft tokens one image becomes.</summary>
    public required ImageTokenIds ImageTokens { get; init; }

    /// <summary>The text decoder's width: the projector's output, one embedding per soft token.</summary>
    public required int TextDim { get; init; }

    /// <summary>
    /// The projector's average pooling: kernel and stride over the grid of patch outputs, so that the
    /// <see cref="VisionEncoderConfig.PatchesPerSide"/>² patches become <see cref="ImageTokenIds.TokensPerImage"/> tokens.
    /// </summary>
    public int PoolSize => Encoder.PatchesPerSide / (int)Math.Round(Math.Sqrt(ImageTokens.TokensPerImage));

    /// <summary>The epsilon of the projector's RMSNorm: the vision encoder's <c>layer_norm_eps</c>, as transformers uses.</summary>
    public float ProjectorNormEpsilon => Encoder.LayerNormEpsilon;

    /// <summary>Which naming the checkpoint uses (for messages).</summary>
    public required string Layout { get; init; }

    /// <summary>Every encoder and projector tensor by its name here (see the remarks), with its name in the checkpoint and its shape.</summary>
    public required IReadOnlyDictionary<string, VisionTensor> Tensors { get; init; }

    // Opens the checkpoint again (set by PretrainedModel.Load).
    internal Func<ITensorStore>? Source { get; init; }

    /// <summary>
    /// Opens the checkpoint's vision tensors under the names of <see cref="Tensors"/>, as stored (no transposition):
    /// read them one at a time and dispose the store when done.
    /// </summary>
    public ITensorStore OpenTensors() =>
        new RenamedTensorStore(Source?.Invoke() ?? throw new InvalidOperationException("This vision part was not read from a checkpoint (PretrainedModel.Load sets where it is)."),
            Tensors.ToDictionary(t => t.Key, t => t.Value.Stored, StringComparer.Ordinal));

    // A store showing some tensors of another under other names.
    private sealed class RenamedTensorStore(ITensorStore inner, Dictionary<string, string> names) : ITensorStore
    {
        public IEnumerable<string> Names => names.Keys;

        public bool Contains(string name) => names.ContainsKey(name);

        public int[] ShapeOf(string name) => inner.ShapeOf(Stored(name));

        public float[] Read(string name) => inner.Read(Stored(name));

        public float[] ReadTransposed(string name) => inner.ReadTransposed(Stored(name));

        public void Dispose() => inner.Dispose();

        private string Stored(string name) => names.TryGetValue(name, out var stored) ? stored : throw new KeyNotFoundException($"No vision tensor '{name}'.");
    }
}

/// <summary>One tensor of <see cref="PretrainedVision.Tensors"/>.</summary>
/// <param name="Stored">Its name in the checkpoint.</param>
/// <param name="Shape">Its shape as stored, outermost first.</param>
public sealed record VisionTensor(string Stored, IReadOnlyList<int> Shape);

/// <summary>The image tokens of a vision-language model (config.json's top level).</summary>
/// <param name="BeginImage">The token before an image's soft tokens (<c>boi_token_index</c>, Gemma 3's <c>&lt;start_of_image&gt;</c>).</param>
/// <param name="EndImage">The token after them (<c>eoi_token_index</c>, <c>&lt;end_of_image&gt;</c>).</param>
/// <param name="ImageToken">The soft token whose embedding is replaced by an image feature (<c>image_token_index</c>, <c>&lt;image_soft_token&gt;</c>).</param>
/// <param name="TokensPerImage">How many soft tokens one image becomes (<c>mm_tokens_per_image</c>).</param>
public sealed record ImageTokenIds(int BeginImage, int EndImage, int ImageToken, int TokensPerImage);

/// <summary>
/// A SigLIP vision encoder's configuration (a Hugging Face <c>vision_config</c>, <c>siglip_vision_model</c>): patches of
/// <see cref="PatchSize"/> pixels over a square image, a learned position embedding, <see cref="Layers"/> pre-LayerNorm
/// layers with bidirectional attention, a final LayerNorm. Absent keys take transformers' defaults.
/// </summary>
public sealed record VisionEncoderConfig
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
    public static VisionEncoderConfig FromJson(JsonObject? config)
    {
        config ??= new JsonObject();
        var result = new VisionEncoderConfig
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
