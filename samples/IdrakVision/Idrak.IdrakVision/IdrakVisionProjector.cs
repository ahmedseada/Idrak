// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Devices;
using Idrak.Layers;
using Idrak.Models.Abstractions;
using Idrak.Vision;

namespace Idrak.IdrakVision;

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
            yield return ($"{prefix}linear_{k + 1}.weight", [layer.OutFeatures, layer.InFeatures], HostParallel.Transpose(layer.Weight.ToArray(), layer.InFeatures, layer.OutFeatures));
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
                layer.Weight.Load(HostParallel.Transpose(w, layer.OutFeatures, layer.InFeatures));
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

        return _body.Forward(PatchGrid.AveragePool(input, k));
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_body];

    /// <inheritdoc />
    public override string ToString() => $"IdrakVisionProjector(pool {Config.PoolSize}, {_layers.Length} layers to {Width})";
}
