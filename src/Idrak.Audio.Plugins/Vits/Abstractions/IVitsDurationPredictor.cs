// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Layers;

namespace Idrak.Audio.Plugins.Vits.Abstractions;

/// <summary>How many frames each token lasts: the log-durations of the hidden states [C, T] (speaking: one utterance, no mask).</summary>
internal interface IVitsDurationPredictor : IDisposable
{
    /// <summary>The log-durations [T] of <paramref name="hidden"/> [C, T], with the speaker's embedding [E, 1] (null for one speaker) and the noise scale (the stochastic predictor draws from <paramref name="random"/>).</summary>
    float[] LogDurations(Tensor hidden, Tensor? speaker, float noiseScale, Random random);

    /// <summary>
    /// The training loss of the tokens' frames <paramref name="durations"/> [B, 1, T] (1 over the padding) given a batch of
    /// hidden states [B, C, T] (detached by the caller, zero over the padding), the noise [B, 2, T] a stochastic predictor
    /// draws (zero over the padding) and the batch's mask (null: one utterance, unpadded): a scalar summed over the tokens.
    /// </summary>
    Tensor Loss(Tensor hidden, Tensor durations, Tensor noise, PaddedSequences? mask);
}
