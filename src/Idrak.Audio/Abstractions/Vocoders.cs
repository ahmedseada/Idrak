// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio.Abstractions;

/// <summary>
/// Turns a spectrogram a model produced into sound. The frames are rows (<c>[frames, mels]</c> log-mel as
/// <c>AudioFeatureExtractors</c> "log-mel" lays them out, with the <see cref="AudioFeatureOptions"/> the vocoder
/// was made with saying the rate, window, hop, mel count and normalisation).
/// </summary>
public interface IVocoder
{
    /// <summary>The vocoder's name, e.g. "griffin-lim".</summary>
    string Name { get; }

    /// <summary>The rate of the sound it makes.</summary>
    int SampleRate { get; }

    /// <summary>The mel bands a frame holds.</summary>
    int Mels { get; }

    /// <summary>The sound of <paramref name="logMel"/> (<c>[frames, mels]</c>): one channel at <see cref="SampleRate"/>, frames × hop samples.</summary>
    AudioData Synthesize(Tensor logMel, Device device);
}

/// <summary>
/// The vocoders' names. Built in (Idrak.Audio): "griffin-lim" (<see cref="GriffinLimVocoder"/>: no weights, the phases
/// recovered by iteration, made for any mel layout), reached directly; a vocoder by another name was registered and
/// picked by it, which is not on the new design yet.
/// </summary>
public static class Vocoders
{
    /// <summary>The library's default vocoder, "griffin-lim".</summary>
    public static string DefaultName => GriffinLimVocoder.DefaultName;
}
