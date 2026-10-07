// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

// The image layout of a sample's features, read the way Idrak's image transforms read it.
internal static class ImageShape
{
    /// <summary>A feature shape [..., height, width] as (planes, height, width): the leading dimensions multiplied.</summary>
    public static (int Planes, int Height, int Width) Of(IReadOnlyList<int> shape, int size)
    {
        if (shape.Count < 2)
        {
            throw new ArgumentException($"Image transforms need features shaped [channels, height, width] or [height, width], not [{string.Join(", ", shape)}].");
        }

        int h = shape[^2], w = shape[^1];
        return (size / (h * w), h, w);
    }
}
