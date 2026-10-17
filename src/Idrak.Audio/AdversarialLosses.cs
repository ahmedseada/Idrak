// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Audio;

/// <summary>
/// The least-squares GAN losses HiFi-GAN and VITS train with, over the scores and feature maps several discriminators
/// give (<see cref="Layers.HifiGanDiscriminators.Forward"/>): each term is a mean over a discriminator's values, summed
/// over the discriminators (and the layers, for the feature matching).
/// </summary>
public static class AdversarialLosses
{
    /// <summary>The discriminators' loss: Σ_d mean((1 − real_d)²) + mean(made_d²), the made scores of a detached sound.</summary>
    public static Tensor LeastSquaresDiscriminator(IReadOnlyList<Tensor> real, IReadOnlyList<Tensor> made)
    {
        ArgumentNullException.ThrowIfNull(real);
        ArgumentNullException.ThrowIfNull(made);
        if (real.Count == 0 || made.Count != real.Count)
        {
            throw new ArgumentException($"The real and made scores pair up per discriminator: {real.Count} and {made.Count}.", nameof(made));
        }

        Tensor? loss = null;
        for (int d = 0; d < real.Count; d++)
        {
            var term = (real[d] * -1f + 1f).Square().Mean() + made[d].Square().Mean();
            loss = loss is null ? term : loss + term;
        }

        return loss!;
    }

    /// <summary>The generator's adversarial loss: Σ_d mean((1 − made_d)²).</summary>
    public static Tensor LeastSquaresGenerator(IReadOnlyList<Tensor> made)
    {
        ArgumentNullException.ThrowIfNull(made);
        if (made.Count == 0)
        {
            throw new ArgumentException("There are no scores.", nameof(made));
        }

        Tensor? loss = null;
        foreach (var score in made)
        {
            var term = (score * -1f + 1f).Square().Mean();
            loss = loss is null ? term : loss + term;
        }

        return loss!;
    }

    /// <summary>The feature-matching loss: Σ_d Σ_l mean(|real_dl − made_dl|) over each discriminator's layers (the real maps computed without gradients).</summary>
    public static Tensor FeatureMatching(IReadOnlyList<IReadOnlyList<Tensor>> realMaps, IReadOnlyList<IReadOnlyList<Tensor>> madeMaps)
    {
        ArgumentNullException.ThrowIfNull(realMaps);
        ArgumentNullException.ThrowIfNull(madeMaps);
        if (realMaps.Count == 0 || madeMaps.Count != realMaps.Count)
        {
            throw new ArgumentException($"The real and made feature maps pair up per discriminator: {realMaps.Count} and {madeMaps.Count}.", nameof(madeMaps));
        }

        Tensor? loss = null;
        for (int d = 0; d < realMaps.Count; d++)
        {
            if (madeMaps[d].Count != realMaps[d].Count)
            {
                throw new ArgumentException($"Discriminator {d} gave {realMaps[d].Count} real maps and {madeMaps[d].Count} made ones.", nameof(madeMaps));
            }

            for (int l = 0; l < madeMaps[d].Count; l++)
            {
                var difference = (realMaps[d][l] - madeMaps[d][l]).Abs().Mean();
                loss = loss is null ? difference : loss + difference;
            }
        }

        return loss!;
    }
}
