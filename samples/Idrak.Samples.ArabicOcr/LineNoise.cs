// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Samples.ArabicOcr;

/// <summary>
/// Normal noise on the pixels ("noise" in <see cref="Augmentations"/>, option <c>std</c>, <c>p</c>), clamped to [0, 1].
/// The library's augmentation registry has none (core's <c>GaussianNoise</c> is a data loader's sample transform, not an
/// <see cref="IAugmentation"/>), so the app registers its own, through the registry like any other.
/// </summary>
internal sealed class LineNoise(double standardDeviation, double probability) : IAugmentation
{
    public const string Name = "noise";

    string IAugmentation.Name => Name;

    /// <summary>Registers "noise" in <see cref="Augmentations"/> unless something is registered under that name already.</summary>
    public static void Register()
    {
        if (!Augmentations.Names.Contains(Name, StringComparer.OrdinalIgnoreCase))
        {
            Augmentations.Register(Name, o =>
            {
                o.Allow("std", "p");
                return new LineNoise(o.Number("std", 0.05), o.Number("p", 1));
            });
        }
    }

    public AnnotatedImage Apply(AnnotatedImage sample, AugmentationContext context)
    {
        var random = context.Random;
        if (random.NextDouble() >= probability)
        {
            return sample;
        }

        var image = sample.Image;
        var pixels = (float[])image.Pixels.Clone();
        for (int i = 0; i < pixels.Length; i++)
        {
            double u = 1 - random.NextDouble(), v = random.NextDouble();
            double normal = Math.Sqrt(-2 * Math.Log(u)) * Math.Cos(2 * Math.PI * v);
            pixels[i] = Math.Clamp(pixels[i] + (float)(normal * standardDeviation), 0f, 1f);
        }

        return new AnnotatedImage(new ImageData(pixels, image.Channels, image.Height, image.Width));
    }
}
