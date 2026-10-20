// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Text.Abstractions;

/// <summary>The feature cache key of an image prepared for tuning (<see cref="TuningImages"/>).</summary>
public static class FeatureCacheKeyExtensions
{
    extension(FeatureCacheKey)
    {
        /// <summary>The key of <paramref name="image"/> prepared by <paramref name="images"/> and encoded by the encoder described.</summary>
        public static FeatureCacheKey For(ChatImage image, TuningImages images, string family, string checkpoint, string dtype, string stage = FeatureCacheKey.TowerStage)
        {
            ArgumentNullException.ThrowIfNull(image);
            ArgumentNullException.ThrowIfNull(images);
            return new FeatureCacheKey(image.Hash, images.Pipeline.ToString(), images.VisionOptions.ToString(), family, checkpoint, dtype, stage);
        }
    }
}
