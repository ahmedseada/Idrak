// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.Gemma3Vision;

/// <summary>
/// SigLIP's vision encoder (the vision tower of Gemma 3 and PaliGemma), without its pooling head: pixel values
/// [images, channels, size, size] become [images, patches, dim]. A convolution with stride = kernel = the patch size
/// (with a bias) cuts the image into patches, row by row; a learned position embedding is added (no class token); then
/// <see cref="SiglipVisionConfig.Layers"/> pre-LayerNorm <see cref="TransformerEncoderLayer"/>s (attention with biases on
/// every projection, scaled by head size^-0.5, every patch seeing every other through <see cref="Tensor.AttentionSpans"/>,
/// so the [patches, patches] scores are never stored; a GELU-tanh MLP) and a final LayerNorm (<c>post_layernorm</c>).
/// </summary>
public sealed class SiglipVisionEncoder : Module
{
    private readonly PatchEmbedding _embedding;
    private readonly TransformerEncoderLayer[] _layers;

    // Embedding, layers and final norm in a Sequential: without autograd it frees each layer's intermediate results as
    // it goes, so a pass holds one layer's worth of activations (4,096 patches x 27 layers would otherwise keep ~10 GB).
    private readonly Sequential _body;

    private SiglipVisionEncoder(SiglipVisionConfig config, Conv2d patches, Tensor positions, TransformerEncoderLayer[] layers, LayerNorm postNorm)
    {
        Config = config;
        _embedding = new PatchEmbedding(config, patches, positions);
        _layers = layers;
        _body = new Sequential([_embedding, .. layers, postNorm]);
    }

    /// <summary>The encoder's configuration.</summary>
    public SiglipVisionConfig Config { get; }

    /// <summary>The encoder layers.</summary>
    public IReadOnlyList<TransformerEncoderLayer> Layers => _layers;

    /// <summary>
    /// Builds the encoder from SigLIP's tensors under <paramref name="prefix"/> (<c>embeddings.patch_embedding.{weight,bias}</c>
    /// [dim, channels, patch, patch], <c>embeddings.position_embedding.weight</c> [patches, dim],
    /// <c>encoder.layers.N.{layer_norm1, self_attn.{q,k,v,out}_proj, layer_norm2, mlp.fc1, mlp.fc2}.{weight,bias}</c> with
    /// linear weights [out, in], <c>post_layernorm.{weight,bias}</c>), as <see cref="Gemma3Vision.OpenTensors"/> gives
    /// them with the prefix <c>vision.</c>. Tensors are read one at a time, in float32.
    /// </summary>
    /// <exception cref="NotSupportedException">An activation other than GELU with the tanh approximation, or a pooling head.</exception>
    public static SiglipVisionEncoder FromTensors(SiglipVisionConfig config, ITensorStore tensors, string prefix = "vision.", Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
        if (config.Activation is not ("gelu_pytorch_tanh" or "gelu_new" or "gelu_fast"))
        {
            throw new NotSupportedException($"vision_config: hidden_act '{config.Activation}' is not supported (the encoder's MLP is GELU with the tanh approximation, gelu_pytorch_tanh).");
        }

        if (config.UseHead)
        {
            throw new NotSupportedException("vision_config: vision_use_head (SigLIP's attention-pooling head) is not supported; vision-language models use the patch outputs.");
        }

        device ??= Device.Default;
        int dim = config.Dim, ff = config.FfDim, p = config.PatchSize;
        var built = new List<IDisposable>();
        try
        {
            Tensor Read(string name, int[] shape, bool transposed = false)
            {
                string full = prefix + name;
                var stored = tensors.ShapeOf(full);
                int[] expected = transposed ? [shape[1], shape[0]] : shape;
                if (!stored.SequenceEqual(expected))
                {
                    throw new InvalidDataException($"{full}: shape {Tensor.FormatShape(stored)}, expected {Tensor.FormatShape(expected)} from vision_config.");
                }

                var tensor = Tensor.Persistent(transposed ? tensors.ReadTransposed(full) : tensors.Read(full), shape, device, requiresGrad: true);
                built.Add(tensor);
                return tensor;
            }

            Linear Dense(string name, int inputs, int outputs) =>
                Linear.FromWeights(Read($"{name}.weight", [inputs, outputs], transposed: true), Read($"{name}.bias", [outputs]));

            LayerNorm Norm(string name) => LayerNorm.FromWeights(Read($"{name}.weight", [dim]), Read($"{name}.bias", [dim]), config.LayerNormEpsilon);

            var patches = Conv2d.FromWeights(Read("embeddings.patch_embedding.weight", [dim, config.Channels, p, p]), Read("embeddings.patch_embedding.bias", [dim]), p, stride: p);
            built.Add(patches);
            var positions = Read("embeddings.position_embedding.weight", [config.Patches, dim]);
            var layers = new TransformerEncoderLayer[config.Layers];
            for (int i = 0; i < layers.Length; i++)
            {
                string l = $"encoder.layers.{i}";

                // q, k and v side by side in one [dim, 3·dim] projection (and one bias), as MultiHeadAttention reads them.
                var qkvWeight = new float[dim * 3 * dim];
                var qkvBias = new float[3 * dim];
                string[] parts = ["q_proj", "k_proj", "v_proj"];
                for (int part = 0; part < 3; part++)
                {
                    string name = $"{prefix}{l}.self_attn.{parts[part]}";
                    if (!tensors.ShapeOf($"{name}.weight").SequenceEqual([dim, dim]) || !tensors.ShapeOf($"{name}.bias").SequenceEqual([dim]))
                    {
                        throw new InvalidDataException($"{name}: shapes {Tensor.FormatShape(tensors.ShapeOf($"{name}.weight"))} and "
                            + $"{Tensor.FormatShape(tensors.ShapeOf($"{name}.bias"))}, expected [{dim}, {dim}] and [{dim}] from vision_config.");
                    }

                    var weight = tensors.ReadTransposed($"{name}.weight");                 // [in, out]
                    for (int row = 0; row < dim; row++)
                    {
                        Array.Copy(weight, row * dim, qkvWeight, row * 3 * dim + part * dim, dim);
                    }

                    tensors.Read($"{name}.bias").CopyTo(qkvBias, part * dim);
                }

                var qkvW = Tensor.Persistent(qkvWeight, [dim, 3 * dim], device, requiresGrad: true);
                built.Add(qkvW);
                var qkvB = Tensor.Persistent(qkvBias, [3 * dim], device, requiresGrad: true);
                built.Add(qkvB);
                var attention = MultiHeadAttention.FromWeights(Linear.FromWeights(qkvW, qkvB), Dense($"{l}.self_attn.out_proj", dim, dim), config.Heads);
                layers[i] = TransformerEncoderLayer.FromLayers(Norm($"{l}.layer_norm1"), attention, Norm($"{l}.layer_norm2"),
                    Dense($"{l}.mlp.fc1", dim, ff), Dense($"{l}.mlp.fc2", ff, dim));
            }

            return new SiglipVisionEncoder(config, patches, positions, layers, Norm("post_layernorm"));
        }
        catch
        {
            foreach (var item in built)
            {
                item.Dispose();
            }

            throw;
        }
    }

    /// <summary>
    /// The patch embeddings before the encoder layers: [images, channels, size, size] (or one image [channels, size,
    /// size]) to [images, patches, dim], each patch's convolution plus its position embedding, patches row by row.
    /// </summary>
    public Tensor Embed(Tensor pixels) => _embedding.Forward(pixels);

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input) => _body.Forward(input);

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_body];

    /// <inheritdoc />
    public override void Dispose()
    {
        foreach (var child in _body.Children())
        {
            child.Dispose();
        }

        GC.SuppressFinalize(this);
    }

    /// <inheritdoc />
    public override string ToString() => $"SiglipVisionEncoder({Config.ImageSize}x{Config.ImageSize}, {Config.Patches} patches, {Config.Layers} layers of {Config.Dim})";
}

/// <summary>
/// Gemma 3's multimodal projector: the encoder's [images, side², visionDim] patch outputs (row-major) are average-pooled
/// over the side x side grid with kernel = stride = <see cref="PoolSize"/>, giving (side / PoolSize)² tokens per image
/// (row-major), each RMS-normalized with a gain of 1 + w (Gemma's RMSNorm, in float32), then projected to the text
/// decoder's width as x · W with W stored [visionDim, textDim] (read as stored, no transposition). The output is the
/// soft tokens' embeddings, not scaled by √textDim.
/// </summary>
public sealed class Gemma3Projector : Module
{
    private readonly RMSNorm _norm;
    private readonly Linear _projection;

    private Gemma3Projector(RMSNorm norm, Linear projection, int poolSize)
    {
        _norm = norm;
        _projection = projection;
        PoolSize = poolSize;
    }

    /// <summary>The pooling's kernel and stride over the grid of patches.</summary>
    public int PoolSize { get; }

    /// <summary>The encoder's width (the input's last dimension).</summary>
    public int VisionDim => _projection.InFeatures;

    /// <summary>The text decoder's width (the output's last dimension).</summary>
    public int TextDim => _projection.OutFeatures;

    /// <summary>
    /// Builds the projector from Gemma 3's tensors (<c>mm_soft_emb_norm.weight</c> [visionDim] and
    /// <c>mm_input_projection_weight</c> [visionDim, textDim]) under <paramref name="prefix"/>, as
    /// <see cref="Gemma3Vision.OpenTensors"/> gives them with the prefix <c>projector.</c>.
    /// </summary>
    public static Gemma3Projector FromTensors(ITensorStore tensors, int visionDim, int textDim, int poolSize, float normEpsilon, string prefix = "projector.", Device? device = null)
    {
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(poolSize);
        device ??= Device.Default;
        string normName = $"{prefix}mm_soft_emb_norm.weight", projectionName = $"{prefix}mm_input_projection_weight";
        if (!tensors.ShapeOf(normName).SequenceEqual([visionDim]) || !tensors.ShapeOf(projectionName).SequenceEqual([visionDim, textDim]))
        {
            throw new InvalidDataException($"{normName} {Tensor.FormatShape(tensors.ShapeOf(normName))} and {projectionName} {Tensor.FormatShape(tensors.ShapeOf(projectionName))}: "
                + $"expected [{visionDim}] and [{visionDim}, {textDim}].");
        }

        var gain = Tensor.Persistent(tensors.Read(normName), [visionDim], device, requiresGrad: true);
        var weight = Tensor.Persistent(tensors.Read(projectionName), [visionDim, textDim], device, requiresGrad: true);   // x · W as stored
        return new Gemma3Projector(RMSNorm.FromWeights(gain, normEpsilon, offset: 1f), Linear.FromWeights(weight), poolSize);
    }

    /// <summary>[images, side², visionDim] to [images, (side / PoolSize)², textDim].</summary>
    protected override Tensor ForwardCore(Tensor input)
    {
        int side = (int)Math.Round(Math.Sqrt(input.Rank == 3 ? input.Shape[1] : 0));
        if (input.Rank != 3 || input.Shape[2] != VisionDim || side * side != input.Shape[1] || side % PoolSize != 0)
        {
            throw new ArgumentException($"The projector takes [images, side², {VisionDim}] with a side divisible by {PoolSize}, got {Tensor.FormatShape(input.Shape)}.");
        }

        int images = input.Shape[0], k = PoolSize, pooled = side / k;
        var x = input;
        if (k > 1)
        {
            // Average pooling as two means over the row-major grid: the k rows of each band, then the k columns of each cell.
            x = x.Reshape(images * pooled, k, side * VisionDim).Mean(1)              // [images·pooled, side·dim]
                .Reshape(images * pooled * pooled, k, VisionDim).Mean(1)             // [images·pooled², dim]
                .Reshape(images, pooled * pooled, VisionDim);
        }

        return _projection.Forward(_norm.Forward(x));
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_norm, _projection];

    /// <inheritdoc />
    public override string ToString() => $"Gemma3Projector(pool {PoolSize}, {VisionDim} -> {TextDim})";
}

// SigLIP's patch embedding: the patch convolution, patches row by row, plus the learned position embedding.
internal sealed class PatchEmbedding(SiglipVisionConfig config, Conv2d patches, Tensor positions) : Module
{
    private Tensor _positions = positions;

    protected override Tensor ForwardCore(Tensor input)
    {
        ArgumentNullException.ThrowIfNull(input);
        if (input.Rank == 3)
        {
            input = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2]);
        }

        int size = config.ImageSize;
        if (input.Rank != 4 || input.Shape[1] != config.Channels || input.Shape[2] != size || input.Shape[3] != size)
        {
            throw new ArgumentException($"The vision encoder takes pixel values [images, {config.Channels}, {size}, {size}], got {Tensor.FormatShape(input.Shape)}.");
        }

        int images = input.Shape[0];
        var x = patches.Forward(input)                                              // [images, dim, side, side]
            .Reshape(images, config.Dim, config.Patches)
            .Permute(0, 2, 1);                                                      // [images, patches, dim]
        return x + _positions;
    }

    public override IEnumerable<Module> Children() => [patches];

    public override IEnumerable<Tensor> Parameters() => [.. patches.Parameters(), _positions];

    protected override void MoveTo(Device device)
    {
        base.MoveTo(device);
        _positions = MoveTensor(_positions, device);
    }

    public override string ToString() => $"PatchEmbedding({config.Patches} patches of {config.PatchSize}x{config.PatchSize})";
}
