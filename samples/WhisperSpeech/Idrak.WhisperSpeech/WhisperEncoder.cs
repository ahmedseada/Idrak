// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.WhisperSpeech;

/// <summary>
/// A Whisper-style audio encoder: two convolutions over the mel frames (the second halving them) with the exact GELU,
/// the stored positions added, pre-norm transformer layers (every frame seeing every other, through the library's
/// attention paths) and a final layer normalization. Input [batch, mels, frames] (or [mels, frames]), output
/// [batch, frames / 2, dim].
/// </summary>
public sealed class WhisperEncoder : Module
{
    private readonly Conv2d _conv1;
    private readonly Conv2d _conv2;
    private Tensor _positions;
    private readonly TransformerEncoderLayer[] _layers;
    private readonly LayerNorm _norm;

    internal WhisperEncoder(WhisperConfig config, Conv2d conv1, Conv2d conv2, Tensor positions, TransformerEncoderLayer[] layers, LayerNorm norm)
    {
        Config = config;
        _conv1 = conv1;
        _conv2 = conv2;
        _positions = positions;
        _layers = layers;
        _norm = norm;
        conv1.Name = "conv1";
        conv2.Name = "conv2";
        norm.Name = "norm";
    }

    /// <summary>The settings it was built for.</summary>
    public WhisperConfig Config { get; }

    /// <summary>Model width.</summary>
    public int Dim => Config.Dim;

    /// <summary>Reads the encoder from the checkpoint's <c>model.encoder.*</c> tensors.</summary>
    public static WhisperEncoder Read(ITensorStore store, WhisperConfig config, EncoderWeights weights, Device device, string prefix = "model.encoder.")
    {
        ArgumentNullException.ThrowIfNull(store);
        ArgumentNullException.ThrowIfNull(config);
        int d = config.Dim;
        var created = new List<Module>();
        Tensor? positions = null;
        try
        {
            var conv1 = Convolution(store, prefix + "conv1", config.Mels, d, stride: 1, device);
            created.Add(conv1);
            var conv2 = Convolution(store, prefix + "conv2", d, d, stride: 2, device);
            created.Add(conv2);
            positions = WhisperWeights.Table(store, prefix + "embed_positions.weight", config.MaxSourcePositions, d, device);
            var layers = new TransformerEncoderLayer[config.EncoderLayers];
            for (int i = 0; i < layers.Length; i++)
            {
                string layer = $"{prefix}layers.{i}.";
                var attention = MultiHeadAttention.FromWeights(WhisperWeights.PackedQkv(store, layer + "self_attn.", d, weights, device),
                    WhisperWeights.Projection(store, layer + "self_attn.out_proj", d, d, weights, device), config.EncoderHeads);
                layers[i] = TransformerEncoderLayer.FromLayers(WhisperWeights.Norm(store, layer + "self_attn_layer_norm", d, config.NormEpsilon, device), attention,
                    WhisperWeights.Norm(store, layer + "final_layer_norm", d, config.NormEpsilon, device),
                    WhisperWeights.Projection(store, layer + "fc1", d, config.EncoderFfDim, weights, device),
                    WhisperWeights.Projection(store, layer + "fc2", config.EncoderFfDim, d, weights, device),
                    WhisperWeights.Activation(config.Activation));
                created.Add(layers[i]);
            }

            var norm = WhisperWeights.Norm(store, prefix + "layer_norm", d, config.NormEpsilon, device);
            return new WhisperEncoder(config, conv1, conv2, positions, layers, norm);
        }
        catch
        {
            foreach (var module in created)
            {
                module.Dispose();
            }

            positions?.Dispose();
            throw;
        }
    }

    // A Conv1d of kernel 3 and padding 1 stored as [out, in, 3], run as a Conv2d of (1, 3) over [N, C, 1, T].
    private static Conv2d Convolution(ITensorStore store, string name, int inputs, int outputs, int stride, Device device)
    {
        string weightName = name + ".weight";
        if (!store.Contains(weightName))
        {
            throw new InvalidDataException($"The checkpoint has no tensor '{weightName}'.");
        }

        var shape = store.ShapeOf(weightName);
        if (shape.Length != 3 || shape[0] != outputs || shape[1] != inputs)
        {
            throw new InvalidDataException($"The checkpoint's '{weightName}' is {Tensor.FormatShape(shape)}; [{outputs}, {inputs}, kernel] was expected.");
        }

        int kernel = shape[2];
        var weight = Tensor.Persistent(store.Read(weightName), [outputs, inputs * kernel], device, requiresGrad: true);
        Tensor? bias = store.Contains(name + ".bias") ? WhisperWeights.Vector(store, name + ".bias", outputs, device) : null;
        return Conv2d.FromWeights(weight, bias, (1, kernel), (1, stride), (0, kernel / 2));
    }

    /// <inheritdoc />
    public override IEnumerable<Module> Children() => [_conv1, _conv2, .. _layers, _norm];

    /// <inheritdoc />
    public override IEnumerable<Tensor> Buffers() => [_positions, .. base.Buffers()];

    /// <inheritdoc />
    protected override Tensor ForwardCore(Tensor input)
    {
        var features = input.Rank == 2 ? input.Reshape(1, input.Shape[0], input.Shape[1]) : input;
        if (features.Rank != 3 || features.Shape[1] != Config.Mels)
        {
            throw new ArgumentException($"The encoder takes [batch, {Config.Mels} mels, frames], got {Tensor.FormatShape(input.Shape)}.");
        }

        int n = features.Shape[0], frames = features.Shape[2];
        var x = ExactGELU.Apply(_conv1.Forward(features.Reshape(n, Config.Mels, 1, frames)));
        x = ExactGELU.Apply(_conv2.Forward(x));                               // [n, dim, 1, t]
        int t = x.Shape[3];
        if (t > Config.MaxSourcePositions)
        {
            throw new ArgumentException($"{frames} frames give {t} encoder positions; the model holds {Config.MaxSourcePositions} (its segment is {2 * Config.MaxSourcePositions} frames).");
        }

        x = x.Reshape(n, Dim, t).Permute(0, 2, 1) + (t == Config.MaxSourcePositions ? _positions : _positions.Narrow(0, 0, t));
        // With gradients off, each layer in a scope of its own that keeps only its output, the layer before's released: the
        // device holds one layer's attention scores and activations at a time, not every layer's (rule 76).
        bool release = !Autograd.IsEnabled;
        foreach (var layer in _layers)
        {
            var previous = x;
            using (var scope = release ? new TensorScope() : null)
            {
                x = layer.Forward(x);
                scope?.Keep(x);
            }

            if (release && !ReferenceEquals(previous, input) && !ReferenceEquals(previous, features))
            {
                TensorScope.Untrack(previous).Dispose();
            }
        }

        return _norm.Forward(x);
    }

    /// <inheritdoc />
    protected override void MoveTo(Device device)
    {
        base.MoveTo(device);
        _positions = MoveTensor(_positions, device);
    }

    /// <inheritdoc />
    public override void Dispose()
    {
        _positions.Dispose();
        base.Dispose();
    }

    /// <inheritdoc />
    public override string ToString() => $"WhisperEncoder(dim {Dim}, {_layers.Length} layers, {Config.Mels} mels)";
}
