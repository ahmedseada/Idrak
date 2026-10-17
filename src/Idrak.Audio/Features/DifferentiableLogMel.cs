// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>
/// The log-mel spectrogram of sound as tensor operations, so a loss on it reaches the sound's generator (VITS's and
/// HiFi-GAN's mel loss, <c>mel_spectrogram_torch</c>): reflect-padded by (size − hop) / 2 (a gather), the windowed DFT
/// as a convolution with its cosine and sine filters (<see cref="Basis"/>), √(re² + im² + 1e-6), the filterbank
/// (<see cref="MelFilterbank"/>), log of at least 1e-5.
/// </summary>
public static class DifferentiableLogMel
{
    /// <summary>
    /// The DFT filters [2 · bins, size] of a window: the cosines of each bin, then minus its sines, each times the window
    /// (<see cref="Stft.Window"/>), bins being size / 2 + 1; a convolution with them at stride hop is the windowed DFT.
    /// </summary>
    public static float[] Basis(int size, ReadOnlySpan<float> window)
    {
        ArgumentOutOfRangeException.ThrowIfLessThan(size, 2);
        if (window.Length != size)
        {
            throw new ArgumentException($"The window holds {window.Length} values for frames of {size}.", nameof(window));
        }

        int bins = size / 2 + 1;
        var basis = new float[2 * bins * size];
        for (int k = 0; k < bins; k++)
        {
            for (int n = 0; n < size; n++)
            {
                double angle = 2 * Math.PI * n * k / size;
                basis[k * size + n] = (float)(window[n] * Math.Cos(angle));
                basis[(bins + k) * size + n] = (float)(-window[n] * Math.Sin(angle));
            }
        }

        return basis;
    }

    /// <summary>
    /// The log-mel spectra [B · frames, mels] of the sound [B, L] (frames = L / hop when L is whole hops): <paramref name="basis"/>
    /// [2 · bins, size] on the device (<see cref="Basis"/>), <paramref name="melFilters"/> [bins, mels] (<see cref="MelFilterbank.Tensor"/>).
    /// </summary>
    public static Tensor Compute(Tensor sound, Tensor basis, Tensor melFilters, int size, int hop)
    {
        ArgumentNullException.ThrowIfNull(sound);
        ArgumentNullException.ThrowIfNull(basis);
        ArgumentNullException.ThrowIfNull(melFilters);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(hop);
        int bins = size / 2 + 1;
        if (sound.Rank != 2 || basis.Rank != 2 || basis.Shape[0] != 2 * bins || basis.Shape[1] != size || melFilters.Rank != 2 || melFilters.Shape[0] != bins)
        {
            throw new ArgumentException($"The log-mel takes sound [B, L], a basis [{2 * bins}, {size}] and filters [{bins}, mels]; got {Tensor.FormatShape(sound.Shape)}, "
                + $"{Tensor.FormatShape(basis.Shape)}, {Tensor.FormatShape(melFilters.Shape)}.", nameof(sound));
        }

        int batch = sound.Shape[0];
        var padded = sound.Pad(1, (size - hop) / 2, (size - hop) / 2, PaddingMode.Reflect);
        int length = padded.Shape[1], frames = (length - size) / hop + 1;
        var spectrum = padded.Reshape(batch, 1, 1, length).Convolution(basis, null, new ConvGeometry(batch, 1, 1, length, 1, size, 1, hop, 0, 0)).Reshape(batch, 2 * bins, frames);
        var magnitude = (spectrum.Narrow(1, 0, bins).Square() + spectrum.Narrow(1, bins, bins).Square() + 1e-6f).Sqrt();   // [B, bins, frames]
        var byFrame = batch == 1 ? magnitude.Reshape(bins, frames).Transpose() : magnitude.Permute(0, 2, 1).Reshape(batch * frames, bins);
        return byFrame.MatMul(melFilters).Clamp(1e-5f, float.PositiveInfinity).Log();
    }
}
