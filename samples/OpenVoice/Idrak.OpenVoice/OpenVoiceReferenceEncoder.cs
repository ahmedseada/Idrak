// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.OpenVoice;

/// <summary>
/// OpenVoice's <c>ReferenceEncoder</c>: a linear spectrogram [T, bins] normalised over the bins (LayerNorm), six
/// weight-normed 3 × 3 convolutions of stride 2 each followed by ReLU (over [1, 1, T, bins] as an image), the frames'
/// channels and bins flattened, a GRU's last state, and a linear projection to the speaker embedding.
/// </summary>
internal sealed class OpenVoiceReferenceEncoder : IDisposable
{
    private readonly LayerNorm? _norm;
    private readonly Conv2d[] _convs;
    private readonly GRU _gru;
    private readonly Linear _proj;

    public OpenVoiceReferenceEncoder(VitsWeights weights, OpenVoiceConfig config)
    {
        int bins = config.Bins;
        if (weights.Contains("ref_enc.layernorm.weight"))
        {
            _norm = weights.Norm("ref_enc.layernorm", bins, 1e-5f);
        }

        var filters = config.ReferenceFilters;
        _convs = new Conv2d[filters.Length];
        int width = bins, inChannels = 1;
        for (int i = 0; i < filters.Length; i++)
        {
            string prefix = $"ref_enc.convs.{i}";
            string g = prefix + ".weight_g", v = prefix + ".weight_v";
            if (!weights.Contains(v))
            {
                g = prefix + ".parametrizations.weight.original0";
                v = prefix + ".parametrizations.weight.original1";
            }

            var shape = weights.Shape(v);
            if (shape is not [var o, var c, 3, 3] || o != filters[i] || c != inChannels)
            {
                throw new InvalidDataException($"'{v}' is {Abstraction.Tensor.FormatShape(shape)}, the reference encoder takes [{filters[i]}, {inChannels}, 3, 3].");
            }

            var weight = Abstraction.Tensor.Persistent(VitsWeights.FoldWeightNorm(weights.Read(g), weights.Read(v), shape), shape, weights.Device);
            _convs[i] = Conv2d.FromWeights(weight, weights.Tensor(prefix + ".bias", filters[i]), (3, 3), (2, 2), (1, 1));
            width = (width - 3 + 2) / 2 + 1;
            inChannels = filters[i];
        }

        var hh = weights.Shape("ref_enc.gru.weight_hh_l0");
        int hidden = hh[1], inputs = inChannels * width;
        var ih = weights.Shape("ref_enc.gru.weight_ih_l0");
        if (ih is not [var gates, var features] || gates != 3 * hidden || features != inputs)
        {
            throw new InvalidDataException($"'ref_enc.gru.weight_ih_l0' is {Abstraction.Tensor.FormatShape(ih)}, the reference encoder gives {inputs} features a frame to a GRU of {hidden}.");
        }

        _gru = new GRU(inputs, hidden, returnSequences: false, bidirectional: false, layers: 1, candidateBias: true, weights.Device);
        _gru.LoadGateWeights(0, false, weights.Read("ref_enc.gru.weight_ih_l0"), weights.Read("ref_enc.gru.weight_hh_l0"),
            weights.Read("ref_enc.gru.bias_ih_l0"), weights.Read("ref_enc.gru.bias_hh_l0"));
        _proj = weights.Linear("ref_enc.proj", hidden, config.GinChannels);
    }

    /// <summary>The embedding [1, gin] of the spectrogram [T, bins].</summary>
    public Tensor Forward(Tensor spectrogram)
    {
        var x = _norm is null ? spectrogram : _norm.Forward(spectrogram);
        x = x.Reshape(1, 1, x.Shape[0], x.Shape[1]);
        foreach (var conv in _convs)
        {
            x = conv.Forward(x).Relu();                                          // [1, C, T', W']
        }

        int frames = x.Shape[2];
        x = x.Permute(0, 2, 1, 3).Reshape(1, frames, x.Shape[1] * x.Shape[3]);   // [1, T', C · W'], each frame's channels outermost
        return _proj.Forward(_gru.Forward(x));
    }

    public void Dispose()
    {
        _norm?.Dispose();
        foreach (var conv in _convs) conv.Dispose();
        _gru.Dispose();
        _proj.Dispose();
    }
}
