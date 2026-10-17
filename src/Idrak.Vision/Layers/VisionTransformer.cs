// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Layers;
using Idrak.Models;

namespace Idrak.Vision;

/// <summary>
/// A vision transformer encoder's shape (SigLIP's, CLIP's, a tower trained from scratch): patches of
/// <see cref="PatchSize"/> pixels over a square image of <see cref="ImageSize"/>, a learned position embedding,
/// <see cref="Layers"/> pre-LayerNorm layers of width <see cref="Dim"/> with <see cref="Heads"/> heads and an MLP of
/// <see cref="FfDim"/> with the named <see cref="Activation"/>, a final LayerNorm. Every value is a setting.
/// </summary>
public sealed record VisionTransformerConfig
{
    /// <summary>The names <see cref="Activation"/> takes: GELU with the tanh approximation (gelu_pytorch_tanh, gelu_new, gelu_fast), exact GELU (gelu), quick_gelu and relu.</summary>
    public static IReadOnlyList<string> Activations { get; } = ["gelu_pytorch_tanh", "gelu_new", "gelu_fast", "gelu", "quick_gelu", "relu"];

    /// <summary>The encoder's width.</summary>
    public int Dim { get; init; } = 768;

    /// <summary>The MLP's hidden size.</summary>
    public int FfDim { get; init; } = 3072;

    /// <summary>Encoder layers.</summary>
    public int Layers { get; init; } = 12;

    /// <summary>Attention heads.</summary>
    public int Heads { get; init; } = 12;

    /// <summary>The square image's side in pixels.</summary>
    public int ImageSize { get; init; } = 224;

    /// <summary>A patch's side in pixels, also the patch convolution's stride.</summary>
    public int PatchSize { get; init; } = 16;

    /// <summary>Color channels (3 for RGB).</summary>
    public int Channels { get; init; } = 3;

    /// <summary>The LayerNorms' epsilon.</summary>
    public float LayerNormEpsilon { get; init; } = 1e-6f;

    /// <summary>The MLP's activation, one of <see cref="Activations"/>.</summary>
    public string Activation { get; init; } = "gelu_pytorch_tanh";

    /// <summary>Each head's width.</summary>
    public int HeadDim => Dim / Heads;

    /// <summary>Patches along one side of the image.</summary>
    public int PatchesPerSide => ImageSize / PatchSize;

    /// <summary>Patches in the image (the encoder's sequence length).</summary>
    public int Patches => PatchesPerSide * PatchesPerSide;

    /// <summary>Checks the sizes fit together and the activation is one the encoder takes.</summary>
    /// <exception cref="InvalidDataException">They do not.</exception>
    public VisionTransformerConfig Checked()
    {
        if (Dim < 1 || FfDim < 1 || Layers < 1 || Heads < 1 || ImageSize < 1 || PatchSize < 1 || Channels < 1)
        {
            throw new InvalidDataException("vision_config: every size must be positive.");
        }

        if (Dim % Heads != 0)
        {
            throw new InvalidDataException($"vision_config: a width of {Dim} does not split into {Heads} heads.");
        }

        if (ImageSize % PatchSize != 0)
        {
            throw new InvalidDataException($"vision_config: {ImageSize}-pixel images do not split into {PatchSize}-pixel patches.");
        }

        _ = ActivationFactory();
        return this;
    }

    /// <summary>A new module of the configured <see cref="Activation"/>.</summary>
    /// <exception cref="InvalidDataException">The name is not one of <see cref="Activations"/>.</exception>
    public Func<Module> ActivationFactory() => Activation switch
    {
        "gelu_pytorch_tanh" or "gelu_new" or "gelu_fast" => () => new GELU(),
        "gelu" => () => new ExactGELU(),
        "quick_gelu" => () => new QuickGELU(),
        "relu" => () => new ReLU(),
        _ => throw new InvalidDataException($"Activation '{Activation}' is not one a vision transformer takes ({string.Join(", ", Activations)})."),
    };
}

/// <summary>One encoder layer's weights as built: its norms, the fused q, k, v projection ([dim, 3·dim], q then k then v), the output projection and the MLP (Linear weights [in, out]).</summary>
public sealed record VisionTransformerLayerWeights(LayerNorm Norm1, Linear Qkv, Linear Output, LayerNorm Norm2, Linear Fc1, Linear Fc2);

/// <summary>
/// A vision transformer encoder (SigLIP's and CLIP's vision tower, a tower trained from scratch), without a pooling head:
/// pixel values [images, channels, size, size] become [images, patches, dim]. A convolution with stride = kernel = the
/// patch size (with a bias) cuts the image into patches, row by row; a learned position embedding is added (no class
/// token); then pre-LayerNorm <see cref="TransformerEncoderLayer"/>s (attention with biases on every projection, scaled by
/// head size^-0.5, every patch seeing every other through <see cref="Tensor.AttentionSpans"/>, so the [patches, patches]
/// scores are never stored; an MLP with the configured activation) and a final LayerNorm (<c>post_layernorm</c>). Built
/// from a checkpoint's tensors under transformers' names (<see cref="FromTensors"/>), and written back under them
/// (<see cref="Tensors"/>, <see cref="Load"/>), so a family saves a trained tower in the checkpoint's own layout.
/// </summary>
public sealed class VisionTransformerEncoder : Module
{
    private readonly PatchEmbedding _embedding;
    private readonly TransformerEncoderLayer[] _layers;

    // Embedding, layers and final norm in a Sequential: without autograd it frees each layer's intermediate results as
    // it goes, so a pass holds one layer's worth of activations (4,096 patches x 27 layers would otherwise keep ~10 GB).
    private readonly Sequential _body;
    private readonly VisionTransformerLayerWeights[] _weights;
    private readonly LayerNorm _postNorm;

    private VisionTransformerEncoder(VisionTransformerConfig config, Conv2d patches, Tensor positions, TransformerEncoderLayer[] layers, VisionTransformerLayerWeights[] weights, LayerNorm postNorm)
    {
        Config = config;
        _embedding = new PatchEmbedding(config, patches, positions);
        _layers = layers;
        _weights = weights;
        _postNorm = postNorm;
        _body = new Sequential([_embedding, .. layers, postNorm]);
    }

    /// <summary>The encoder's configuration.</summary>
    public VisionTransformerConfig Config { get; }

    /// <summary>The encoder layers.</summary>
    public IReadOnlyList<TransformerEncoderLayer> Layers => _layers;

    /// <summary>The patch convolution ([dim, channels, patch, patch] with a bias).</summary>
    public Conv2d Patches => _embedding.Patches;

    /// <summary>The position embedding [patches, dim].</summary>
    public Tensor PositionEmbedding => _embedding.Positions;

    /// <summary>Each layer's weights as built.</summary>
    public IReadOnlyList<VisionTransformerLayerWeights> LayerWeights => _weights;

    /// <summary>The final LayerNorm.</summary>
    public LayerNorm PostNorm => _postNorm;

    /// <summary>
    /// Builds the encoder from the tensors under <paramref name="prefix"/> (<c>embeddings.patch_embedding.{weight,bias}</c>
    /// [dim, channels, patch, patch], <c>embeddings.position_embedding.weight</c> [patches, dim],
    /// <c>encoder.layers.N.{layer_norm1, self_attn.{q,k,v,out}_proj, layer_norm2, mlp.fc1, mlp.fc2}.{weight,bias}</c> with
    /// linear weights [out, in], <c>post_layernorm.{weight,bias}</c>), read one at a time. The projections (attention and
    /// MLP) keep the precision <paramref name="weights"/> asks for (by default as stored: a bfloat16 checkpoint's stay
    /// bfloat16, half the memory, computed in float32: <see cref="StoredWeights"/>); the patch convolution, the position
    /// embedding, the norms and every bias are float32. Every float32 tensor is a parameter.
    /// </summary>
    /// <exception cref="InvalidDataException">A tensor's shape disagrees with the configuration, or the activation is unknown.</exception>
    public static VisionTransformerEncoder FromTensors(VisionTransformerConfig config, ITensorStore tensors, string prefix, Device? device = null, EncoderWeights weights = EncoderWeights.AsStored)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
        ArgumentNullException.ThrowIfNull(prefix);
        config.Checked();
        device ??= Device.Default;
        int dim = config.Dim, ff = config.FfDim, p = config.PatchSize;
        var activation = config.ActivationFactory();
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

            Linear Dense(string name, int inputs, int outputs)
            {
                string full = $"{prefix}{name}.weight";
                if (!StoredWeights.KeepsBFloat16(tensors, weights, full))
                {
                    return Linear.FromWeights(Read($"{name}.weight", [inputs, outputs], transposed: true), Read($"{name}.bias", [outputs]));
                }

                var stored = tensors.ShapeOf(full);
                if (!stored.SequenceEqual([outputs, inputs]))
                {
                    throw new InvalidDataException($"{full}: shape {Tensor.FormatShape(stored)}, expected {Tensor.FormatShape([outputs, inputs])} from vision_config.");
                }

                var layer = StoredWeights.Linear(tensors.ReadTransposed(full), inputs, outputs, Read($"{name}.bias", [outputs]), bfloat16: true, device);
                built.Add(layer);
                return layer;
            }

            LayerNorm Norm(string name) => LayerNorm.FromWeights(Read($"{name}.weight", [dim]), Read($"{name}.bias", [dim]), config.LayerNormEpsilon);

            var patches = Conv2d.FromWeights(Read("embeddings.patch_embedding.weight", [dim, config.Channels, p, p]), Read("embeddings.patch_embedding.bias", [dim]), p, stride: p);
            built.Add(patches);
            var positions = Read("embeddings.position_embedding.weight", [config.Patches, dim]);
            var layers = new TransformerEncoderLayer[config.Layers];
            var layerWeights = new VisionTransformerLayerWeights[config.Layers];
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

                var qkvB = Tensor.Persistent(qkvBias, [3 * dim], device, requiresGrad: true);
                built.Add(qkvB);
                var qkv = StoredWeights.Linear(qkvWeight, dim, 3 * dim, qkvB,
                    StoredWeights.KeepsBFloat16(tensors, weights, parts.Select(part => $"{prefix}{l}.self_attn.{part}.weight")), device);
                built.Add(qkv);
                var output = Dense($"{l}.self_attn.out_proj", dim, dim);
                var attention = MultiHeadAttention.FromWeights(qkv, output, config.Heads);
                var (norm1, norm2) = (Norm($"{l}.layer_norm1"), Norm($"{l}.layer_norm2"));
                var (fc1, fc2) = (Dense($"{l}.mlp.fc1", dim, ff), Dense($"{l}.mlp.fc2", ff, dim));
                layers[i] = TransformerEncoderLayer.FromLayers(norm1, attention, norm2, fc1, fc2, activation());
                layerWeights[i] = new VisionTransformerLayerWeights(norm1, qkv, output, norm2, fc1, fc2);
            }

            return new VisionTransformerEncoder(config, patches, positions, layers, layerWeights, Norm("post_layernorm"));
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

    /// <summary>Every tensor's name and shape under <paramref name="prefix"/>, as a checkpoint of <paramref name="config"/> stores them (the order of <see cref="Tensors"/>).</summary>
    public static IEnumerable<(string Name, int[] Shape)> Expected(VisionTransformerConfig config, string prefix)
    {
        ArgumentNullException.ThrowIfNull(config);
        int d = config.Dim, f = config.FfDim;
        yield return ($"{prefix}embeddings.patch_embedding.weight", [d, config.Channels, config.PatchSize, config.PatchSize]);
        yield return ($"{prefix}embeddings.patch_embedding.bias", [d]);
        yield return ($"{prefix}embeddings.position_embedding.weight", [config.Patches, d]);
        for (int i = 0; i < config.Layers; i++)
        {
            string l = $"{prefix}encoder.layers.{i}";
            foreach (string norm in (string[])["layer_norm1", "layer_norm2"])
            {
                yield return ($"{l}.{norm}.weight", [d]);
                yield return ($"{l}.{norm}.bias", [d]);
            }

            foreach (string proj in (string[])["q_proj", "k_proj", "v_proj", "out_proj"])
            {
                yield return ($"{l}.self_attn.{proj}.weight", [d, d]);
                yield return ($"{l}.self_attn.{proj}.bias", [d]);
            }

            yield return ($"{l}.mlp.fc1.weight", [f, d]);
            yield return ($"{l}.mlp.fc1.bias", [f]);
            yield return ($"{l}.mlp.fc2.weight", [d, f]);
            yield return ($"{l}.mlp.fc2.bias", [d]);
        }

        yield return ($"{prefix}post_layernorm.weight", [d]);
        yield return ($"{prefix}post_layernorm.bias", [d]);
    }

    /// <summary>
    /// The encoder's tensors by their names (prefix + name) in the checkpoint's layout, the inverse of <see cref="FromTensors"/>:
    /// Linear weights [in, out] transposed to [out, in], the fused [dim, 3·dim] q, k, v projection split into q_proj, k_proj
    /// and v_proj. Packed (bfloat16) projections are read back as float32 values.
    /// </summary>
    public IEnumerable<(string Name, int[] Shape, float[] Values)> Tensors(string prefix)
    {
        ArgumentNullException.ThrowIfNull(prefix);
        int dim = Config.Dim;
        yield return ($"{prefix}embeddings.patch_embedding.weight", [dim, Config.Channels, Config.PatchSize, Config.PatchSize], Patches.Weight.ToArray());
        yield return ($"{prefix}embeddings.patch_embedding.bias", [dim], Patches.Bias!.ToArray());
        yield return ($"{prefix}embeddings.position_embedding.weight", [Config.Patches, dim], PositionEmbedding.ToArray());
        for (int i = 0; i < _weights.Length; i++)
        {
            var w = _weights[i];
            string l = $"{prefix}encoder.layers.{i}";
            yield return ($"{l}.layer_norm1.weight", [dim], w.Norm1.Gamma.ToArray());
            yield return ($"{l}.layer_norm1.bias", [dim], w.Norm1.Beta.ToArray());
            yield return ($"{l}.layer_norm2.weight", [dim], w.Norm2.Gamma.ToArray());
            yield return ($"{l}.layer_norm2.bias", [dim], w.Norm2.Beta.ToArray());
            float[] qkv = ((ILinearLayer)w.Qkv).WeightValues(), qkvBias = w.Qkv.Bias!.ToArray();
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

                yield return ($"{l}.self_attn.{parts[part]}.weight", [dim, dim], weight);
                yield return ($"{l}.self_attn.{parts[part]}.bias", [dim], qkvBias[(part * dim)..((part + 1) * dim)]);
            }

            foreach (var (name, layer) in new[] { ("self_attn.out_proj", w.Output), ("mlp.fc1", w.Fc1), ("mlp.fc2", w.Fc2) })
            {
                yield return ($"{l}.{name}.weight", [layer.OutFeatures, layer.InFeatures], HostParallel.Transpose(((ILinearLayer)layer).WeightValues(), layer.InFeatures, layer.OutFeatures));
                yield return ($"{l}.{name}.bias", [layer.OutFeatures], layer.Bias!.ToArray());
            }
        }

        yield return ($"{prefix}post_layernorm.weight", [dim], PostNorm.Gamma.ToArray());
        yield return ($"{prefix}post_layernorm.bias", [dim], PostNorm.Beta.ToArray());
    }

    /// <summary>
    /// Sets the tensors named and laid out as <see cref="Tensors"/> gives them (q, k and v go into the fused projection,
    /// each where it lies; names absent from <paramref name="values"/> are left as they are); returns the names taken.
    /// </summary>
    /// <exception cref="InvalidOperationException">A projection holds packed (bfloat16) weights, which cannot be set.</exception>
    public List<string> Load(string prefix, IReadOnlyDictionary<string, float[]> values)
    {
        ArgumentNullException.ThrowIfNull(prefix);
        ArgumentNullException.ThrowIfNull(values);
        int dim = Config.Dim;
        var taken = new List<string>();
        void Set(string name, Tensor target, Func<float[], float[]>? layout = null)
        {
            if (values.TryGetValue(prefix + name, out var v))
            {
                target.Load(layout is null ? v : layout(v));
                taken.Add(prefix + name);
            }
        }

        Set("embeddings.patch_embedding.weight", Patches.Weight);
        Set("embeddings.patch_embedding.bias", Patches.Bias!);
        Set("embeddings.position_embedding.weight", PositionEmbedding);
        for (int i = 0; i < _weights.Length; i++)
        {
            var w = _weights[i];
            string l = $"encoder.layers.{i}";
            Set($"{l}.layer_norm1.weight", w.Norm1.Gamma);
            Set($"{l}.layer_norm1.bias", w.Norm1.Beta);
            string[] parts = ["q_proj", "k_proj", "v_proj"];
            if (parts.Any(p => values.ContainsKey($"{prefix}{l}.self_attn.{p}.weight") || values.ContainsKey($"{prefix}{l}.self_attn.{p}.bias")))
            {
                float[] qkv = ((ILinearLayer)w.Qkv).WeightValues(), qkvBias = w.Qkv.Bias!.ToArray();
                for (int part = 0; part < 3; part++)
                {
                    if (values.TryGetValue($"{prefix}{l}.self_attn.{parts[part]}.weight", out var weight))
                    {
                        for (int input = 0; input < dim; input++)
                        {
                            for (int output = 0; output < dim; output++)
                            {
                                qkv[input * 3 * dim + part * dim + output] = weight[output * dim + input];
                            }
                        }

                        taken.Add($"{prefix}{l}.self_attn.{parts[part]}.weight");
                    }

                    if (values.TryGetValue($"{prefix}{l}.self_attn.{parts[part]}.bias", out var bias))
                    {
                        bias.CopyTo(qkvBias, part * dim);
                        taken.Add($"{prefix}{l}.self_attn.{parts[part]}.bias");
                    }
                }

                w.Qkv.Weight.Load(qkv);
                w.Qkv.Bias!.Load(qkvBias);
            }

            foreach (var (name, layer) in new[] { ("self_attn.out_proj", w.Output), ("mlp.fc1", w.Fc1), ("mlp.fc2", w.Fc2) })
            {
                Set($"{l}.{name}.weight", layer.Weight, v => HostParallel.Transpose(v, layer.OutFeatures, layer.InFeatures));
                Set($"{l}.{name}.bias", layer.Bias!);
            }

            Set($"{l}.layer_norm2.weight", w.Norm2.Gamma);
            Set($"{l}.layer_norm2.bias", w.Norm2.Beta);
        }

        Set("post_layernorm.weight", PostNorm.Gamma);
        Set("post_layernorm.bias", PostNorm.Beta);
        return taken;
    }

    /// <summary>
    /// The patch embeddings before the encoder layers: [images, channels, size, size] (or one image [channels, size,
    /// size]) to [images, patches, dim], each patch's convolution plus its position embedding, patches row by row.
    /// </summary>
    public Tensor Embed(Tensor pixels) => _embedding.Forward(pixels);

    /// <summary>
    /// The forward pass for training the tower: with <paramref name="checkpointed"/> (or <see cref="ActivationMemory.CheckpointsBlocks"/>
    /// on) and gradients recorded, each encoder layer runs checkpointed (only its output kept, the layer run again in the
    /// backward pass; the same gradients).
    /// </summary>
    public Tensor ForwardTraining(Tensor pixels, bool checkpointed)
    {
        if (!(checkpointed || ActivationMemory.CheckpointsBlocks) || !Autograd.IsEnabled)
        {
            return _body.Forward(pixels);
        }

        var x = _embedding.Forward(pixels);
        foreach (var layer in _layers)
        {
            x = layer.ForwardCheckpointed(x);
        }

        return _postNorm.Forward(x);
    }

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
    public override string ToString() => $"VisionTransformerEncoder({Config.ImageSize}x{Config.ImageSize}, {Config.Patches} patches, {Config.Layers} layers of {Config.Dim})";

    // The patch convolution, patches row by row, plus the learned position embedding.
    private sealed class PatchEmbedding(VisionTransformerConfig config, Conv2d patches, Tensor positions) : Module
    {
        private Tensor _positions = positions;

        public Conv2d Patches => patches;

        public Tensor Positions => _positions;

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
            var x = patches.Forward(input)                                          // [images, dim, side, side]
                .Reshape(images, config.Dim, config.Patches)
                .Permute(0, 2, 1);                                                  // [images, patches, dim]
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
}

/// <summary>The grid of an encoder's patch outputs, row-major, as the projectors read it.</summary>
public static class PatchGrid
{
    /// <summary>
    /// [images, side², dim] average-pooled over the side × side grid with kernel = stride = <paramref name="poolSize"/>:
    /// [images, (side / poolSize)², dim], row-major, as two means (the rows of each band, then the columns of each cell);
    /// the input itself when the pool size is 1.
    /// </summary>
    /// <exception cref="ArgumentException">Not [images, side², dim], or the side is not a multiple of the pool size.</exception>
    public static Tensor AveragePool(Tensor x, int poolSize)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(poolSize);
        int side = (int)Math.Round(Math.Sqrt(x.Rank == 3 ? x.Shape[1] : 0));
        if (x.Rank != 3 || side * side != x.Shape[1] || side % poolSize != 0)
        {
            throw new ArgumentException($"The patch grid is [images, side², dim] with a side divisible by {poolSize}, got {Tensor.FormatShape(x.Shape)}.", nameof(x));
        }

        if (poolSize == 1)
        {
            return x;
        }

        int images = x.Shape[0], dim = x.Shape[2], k = poolSize, pooled = side / k;
        return x.Reshape(images * pooled, k, side * dim).Mean(1)                    // [images·pooled, side·dim]
            .Reshape(images * pooled * pooled, k, dim).Mean(1)                      // [images·pooled², dim]
            .Reshape(images, pooled * pooled, dim);
    }
}
