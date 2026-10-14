// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;
using Idrak.Models.Abstractions;

namespace Idrak.IdrakVision;

/// <summary>
/// IdrakVision's tower: pixel values [images, channels, size, size] become [images, patches, dim]. A convolution with
/// stride = kernel = the patch size (with a bias) cuts the image into patches, row by row; a learned position embedding is
/// added; then pre-LayerNorm <see cref="TransformerEncoderLayer"/>s (attention with biases on every projection, every
/// patch seeing every other; an MLP with the configured activation) and a final LayerNorm. Weights are float32 (the family
/// trains them).
/// </summary>
public sealed class IdrakVisionTower : Module
{
    private readonly TowerEmbedding _embedding;
    private readonly TransformerEncoderLayer[] _layers;
    private readonly Sequential _body;
    private readonly TowerLayerWeights[] _weights;
    private readonly LayerNorm _postNorm;

    private IdrakVisionTower(IdrakVisionConfig config, Conv2d patches, Tensor positions, TransformerEncoderLayer[] layers, TowerLayerWeights[] weights, LayerNorm postNorm)
    {
        Config = config;
        _embedding = new TowerEmbedding(config, patches, positions);
        _layers = layers;
        _weights = weights;
        _postNorm = postNorm;
        _body = new Sequential([_embedding, .. layers, postNorm]);
    }

    /// <summary>The tower's configuration.</summary>
    public IdrakVisionConfig Config { get; }

    internal Conv2d Patches => _embedding.Patches;

    internal Tensor PositionEmbedding => _embedding.Positions;

    internal IReadOnlyList<TowerLayerWeights> LayerWeights => _weights;

    internal LayerNorm PostNorm => _postNorm;

    /// <summary>The forward pass for training: each encoder layer checkpointed when asked and gradients are recorded.</summary>
    internal Tensor ForwardTraining(Tensor pixels, bool checkpointed)
    {
        if (!checkpointed || !Autograd.IsEnabled)
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

    /// <summary>
    /// The tower from its tensors under <paramref name="prefix"/>: <c>embeddings.patch_embedding.{weight,bias}</c>
    /// [dim, channels, patch, patch], <c>embeddings.position_embedding.weight</c> [patches, dim],
    /// <c>encoder.layers.N.{layer_norm1, self_attn.{q,k,v,out}_proj, layer_norm2, mlp.fc1, mlp.fc2}.{weight,bias}</c>
    /// (linear weights [out, in]) and <c>post_layernorm.{weight,bias}</c>.
    /// </summary>
    /// <exception cref="InvalidDataException">A tensor's shape disagrees with the configuration.</exception>
    public static IdrakVisionTower FromTensors(IdrakVisionConfig config, ITensorStore tensors, string prefix, Device device)
    {
        ArgumentNullException.ThrowIfNull(config);
        ArgumentNullException.ThrowIfNull(tensors);
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

            Linear Dense(string name, int inputs, int outputs) => Linear.FromWeights(Read($"{name}.weight", [inputs, outputs], transposed: true), Read($"{name}.bias", [outputs]));

            LayerNorm Norm(string name) => LayerNorm.FromWeights(Read($"{name}.weight", [dim]), Read($"{name}.bias", [dim]), config.LayerNormEpsilon);

            var patches = Conv2d.FromWeights(Read("embeddings.patch_embedding.weight", [dim, config.Channels, p, p]), Read("embeddings.patch_embedding.bias", [dim]), p, stride: p);
            built.Add(patches);
            var positions = Read("embeddings.position_embedding.weight", [config.Patches, dim]);
            var activation = Layers2.Activation(config.Activation);
            var layers = new TransformerEncoderLayer[config.Layers];
            var weights = new TowerLayerWeights[config.Layers];
            for (int i = 0; i < layers.Length; i++)
            {
                string l = $"encoder.layers.{i}";
                var qkvWeight = new float[dim * 3 * dim];
                var qkvBias = new float[3 * dim];
                string[] parts = ["q_proj", "k_proj", "v_proj"];
                for (int part = 0; part < 3; part++)
                {
                    string name = $"{prefix}{l}.self_attn.{parts[part]}";
                    if (!tensors.ShapeOf($"{name}.weight").SequenceEqual([dim, dim]) || !tensors.ShapeOf($"{name}.bias").SequenceEqual([dim]))
                    {
                        throw new InvalidDataException($"{name}: expected [{dim}, {dim}] and [{dim}] from vision_config.");
                    }

                    var weight = tensors.ReadTransposed($"{name}.weight");                 // [in, out]
                    for (int row = 0; row < dim; row++)
                    {
                        Array.Copy(weight, row * dim, qkvWeight, row * 3 * dim + part * dim, dim);
                    }

                    tensors.Read($"{name}.bias").CopyTo(qkvBias, part * dim);
                }

                var qkv = Linear.FromWeights(Tensor.Persistent(qkvWeight, [dim, 3 * dim], device, requiresGrad: true), Tensor.Persistent(qkvBias, [3 * dim], device, requiresGrad: true));
                built.Add(qkv);
                var output = Dense($"{l}.self_attn.out_proj", dim, dim);
                var attention = MultiHeadAttention.FromWeights(qkv, output, config.Heads);
                var (norm1, norm2) = (Norm($"{l}.layer_norm1"), Norm($"{l}.layer_norm2"));
                var (fc1, fc2) = (Dense($"{l}.mlp.fc1", dim, ff), Dense($"{l}.mlp.fc2", ff, dim));
                layers[i] = TransformerEncoderLayer.FromLayers(norm1, attention, norm2, fc1, fc2, activation());
                weights[i] = new TowerLayerWeights(norm1, qkv, output, norm2, fc1, fc2);
            }

            return new IdrakVisionTower(config, patches, positions, layers, weights, Norm("post_layernorm"));
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

    /// <summary>The tower's tensors by their names (prefix + name), as the checkpoint stores them: the inverse of <see cref="FromTensors"/>.</summary>
    internal IEnumerable<(string Name, int[] Shape, float[] Values)> Tensors(string prefix)
    {
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

                yield return ($"{l}.self_attn.{parts[part]}.weight", [dim, dim], weight);
                yield return ($"{l}.self_attn.{parts[part]}.bias", [dim], qkvBias[(part * dim)..((part + 1) * dim)]);
            }

            foreach (var (name, layer) in new[] { ("self_attn.out_proj", w.Output), ("mlp.fc1", w.Fc1), ("mlp.fc2", w.Fc2) })
            {
                yield return ($"{l}.{name}.weight", [layer.OutFeatures, layer.InFeatures], Matrices.Transpose(layer.Weight.ToArray(), layer.InFeatures, layer.OutFeatures));
                yield return ($"{l}.{name}.bias", [layer.OutFeatures], layer.Bias!.ToArray());
            }

            yield return ($"{l}.layer_norm2.weight", [dim], w.Norm2.Gamma.ToArray());
            yield return ($"{l}.layer_norm2.bias", [dim], w.Norm2.Beta.ToArray());
        }

        yield return ($"{prefix}post_layernorm.weight", [dim], PostNorm.Gamma.ToArray());
        yield return ($"{prefix}post_layernorm.bias", [dim], PostNorm.Beta.ToArray());
    }

    /// <summary>Sets tensors named and laid out as <see cref="Tensors"/> gives them; returns the names taken.</summary>
    internal List<string> Load(string prefix, IReadOnlyDictionary<string, float[]> values)
    {
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
                float[] qkv = w.Qkv.Weight.ToArray(), qkvBias = w.Qkv.Bias!.ToArray();
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
                Set($"{l}.{name}.weight", layer.Weight, v => Matrices.Transpose(v, layer.OutFeatures, layer.InFeatures));
                Set($"{l}.{name}.bias", layer.Bias!);
            }

            Set($"{l}.layer_norm2.weight", w.Norm2.Gamma);
            Set($"{l}.layer_norm2.bias", w.Norm2.Beta);
        }

        Set("post_layernorm.weight", PostNorm.Gamma);
        Set("post_layernorm.bias", PostNorm.Beta);
        return taken;
    }

    /// <summary>Every tensor's name and shape under <paramref name="prefix"/>, as a checkpoint of this configuration stores them.</summary>
    internal static IEnumerable<(string Name, int[] Shape)> Expected(IdrakVisionConfig c, string prefix)
    {
        int d = c.Dim, f = c.FfDim;
        yield return ($"{prefix}embeddings.patch_embedding.weight", [d, c.Channels, c.PatchSize, c.PatchSize]);
        yield return ($"{prefix}embeddings.patch_embedding.bias", [d]);
        yield return ($"{prefix}embeddings.position_embedding.weight", [c.Patches, d]);
        for (int i = 0; i < c.Layers; i++)
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
    public override string ToString() => $"IdrakVisionTower({Config.ImageSize}x{Config.ImageSize}, {Config.Patches} patches, {Config.Layers} layers of {Config.Dim})";
}

/// <summary>
/// IdrakVision's projector: the tower's [images, side², dim] patches (row-major) average-pooled over PoolSize x PoolSize
/// cells, then linear layers (with the configured activation between them) to the text decoder's width: the image tokens'
/// embeddings, not scaled.
/// </summary>
public sealed class IdrakVisionProjector : Module
{
    private readonly Linear[] _layers;
    private readonly Func<Module> _activation;
    private readonly Sequential _body;

    private IdrakVisionProjector(IdrakVisionConfig config, Linear[] layers)
    {
        Config = config;
        _layers = layers;
        _activation = Layers2.Activation(config.ProjectorActivation);
        var modules = new List<Module>();
        for (int i = 0; i < layers.Length; i++)
        {
            if (i > 0)
            {
                modules.Add(_activation());
            }

            modules.Add(layers[i]);
        }

        _body = new Sequential(modules);
    }

    /// <summary>The configuration.</summary>
    public IdrakVisionConfig Config { get; }

    /// <summary>The text decoder's width.</summary>
    public int Width => _layers[^1].OutFeatures;

    internal IReadOnlyList<Linear> Layers => _layers;

    /// <summary>The projector from <c>linear_K.{weight,bias}</c> under <paramref name="prefix"/> (weights [out, in]).</summary>
    public static IdrakVisionProjector FromTensors(IdrakVisionConfig config, int width, ITensorStore tensors, string prefix, Device device)
    {
        var layers = new Linear[config.ProjectorDepth];
        for (int k = 0; k < layers.Length; k++)
        {
            int inputs = k == 0 ? config.Dim : width;
            string name = $"{prefix}linear_{k + 1}";
            if (!tensors.ShapeOf($"{name}.weight").SequenceEqual([width, inputs]))
            {
                throw new InvalidDataException($"{name}.weight: shape {Tensor.FormatShape(tensors.ShapeOf($"{name}.weight"))}, expected [{width}, {inputs}].");
            }

            var weight = Tensor.Persistent(tensors.ReadTransposed($"{name}.weight"), [inputs, width], device, requiresGrad: true);
            var bias = config.ProjectorBias ? Tensor.Persistent(tensors.Read($"{name}.bias"), [width], device, requiresGrad: true) : null;
            layers[k] = Linear.FromWeights(weight, bias);
        }

        return new IdrakVisionProjector(config, layers);
    }

    /// <summary>Every tensor's name and shape under <paramref name="prefix"/> for a projector to <paramref name="width"/>.</summary>
    internal static IEnumerable<(string Name, int[] Shape)> Expected(IdrakVisionConfig c, int width, string prefix)
    {
        for (int k = 0; k < c.ProjectorDepth; k++)
        {
            yield return ($"{prefix}linear_{k + 1}.weight", [width, k == 0 ? c.Dim : width]);
            if (c.ProjectorBias)
            {
                yield return ($"{prefix}linear_{k + 1}.bias", [width]);
            }
        }
    }

    internal IEnumerable<(string Name, int[] Shape, float[] Values)> Tensors(string prefix)
    {
        for (int k = 0; k < _layers.Length; k++)
        {
            var layer = _layers[k];
            yield return ($"{prefix}linear_{k + 1}.weight", [layer.OutFeatures, layer.InFeatures], Matrices.Transpose(layer.Weight.ToArray(), layer.InFeatures, layer.OutFeatures));
            if (layer.Bias is { } bias)
            {
                yield return ($"{prefix}linear_{k + 1}.bias", [layer.OutFeatures], bias.ToArray());
            }
        }
    }

    internal List<string> Load(string prefix, IReadOnlyDictionary<string, float[]> values)
    {
        var taken = new List<string>();
        for (int k = 0; k < _layers.Length; k++)
        {
            var layer = _layers[k];
            if (values.TryGetValue($"{prefix}linear_{k + 1}.weight", out var w))
            {
                layer.Weight.Load(Matrices.Transpose(w, layer.OutFeatures, layer.InFeatures));
                taken.Add($"{prefix}linear_{k + 1}.weight");
            }

            if (layer.Bias is { } bias && values.TryGetValue($"{prefix}linear_{k + 1}.bias", out var b))
            {
                bias.Load(b);
                taken.Add($"{prefix}linear_{k + 1}.bias");
            }
        }

        return taken;
    }

    /// <summary>[images, side², dim] to [images, (side / PoolSize)², width].</summary>
    protected override Tensor ForwardCore(Tensor input)
    {
        int side = (int)Math.Round(Math.Sqrt(input.Rank == 3 ? input.Shape[1] : 0));
        int dim = Config.Dim, k = Config.PoolSize;
        if (input.Rank != 3 || input.Shape[2] != dim || side * side != input.Shape[1] || side % k != 0)
        {
            throw new ArgumentException($"The projector takes [images, side², {dim}] with a side divisible by {k}, got {Tensor.FormatShape(input.Shape)}.");
        }

        int images = input.Shape[0], pooled = side / k;
        var x = input;
        if (k > 1)
        {
            // Average pooling as two means over the row-major grid: the k rows of each band, then the k columns of each cell.
            x = x.Reshape(images * pooled, k, side * dim).Mean(1)
                .Reshape(images * pooled * pooled, k, dim).Mean(1)
                .Reshape(images, pooled * pooled, dim);
        }

        return _body.Forward(x);
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_body];

    /// <inheritdoc />
    public override string ToString() => $"IdrakVisionProjector(pool {Config.PoolSize}, {_layers.Length} layers to {Width})";
}

// One encoder layer's weights as built (the fused q, k, v projection [dim, 3·dim]; Linear weights [in, out]).
internal sealed record TowerLayerWeights(LayerNorm Norm1, Linear Qkv, Linear Output, LayerNorm Norm2, Linear Fc1, Linear Fc2);

// The patch convolution, patches row by row, plus the learned position embedding.
internal sealed class TowerEmbedding(IdrakVisionConfig config, Conv2d patches, Tensor positions) : Module
{
    private Tensor _positions = positions;

    public Conv2d Patches => patches;

    public Tensor Positions => _positions;

    protected override Tensor ForwardCore(Tensor input)
    {
        if (input.Rank == 3)
        {
            input = input.Reshape(1, input.Shape[0], input.Shape[1], input.Shape[2]);
        }

        int size = config.ImageSize;
        if (input.Rank != 4 || input.Shape[1] != config.Channels || input.Shape[2] != size || input.Shape[3] != size)
        {
            throw new ArgumentException($"The tower takes pixel values [images, {config.Channels}, {size}, {size}], got {Tensor.FormatShape(input.Shape)}.");
        }

        int images = input.Shape[0];
        var x = patches.Forward(input).Reshape(images, config.Dim, config.Patches).Permute(0, 2, 1);
        return x + _positions;
    }

    public override IEnumerable<Module> Children() => [patches];

    public override IEnumerable<Tensor> Parameters() => [.. patches.Parameters(), _positions];

    protected override void MoveTo(Device device)
    {
        base.MoveTo(device);
        _positions = MoveTensor(_positions, device);
    }
}

internal static class Matrices
{
    // [rows, columns] row-major to [columns, rows].
    public static float[] Transpose(float[] values, int rows, int columns)
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
