// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Data;

/// <summary>
/// Core's augmentations of detection and segmentation samples (<see cref="Augmentations"/>): "flip" (<see cref="RandomFlip"/>)
/// and "shift" (<see cref="RandomShift"/>), registered by <c>LibraryRegistrations</c> when the registry is first used.
/// </summary>
internal static class LibraryAugmentations
{
    internal static void RegisterDefaults()
    {
        Augmentations.Register("flip", o =>
        {
            o.Allow("horizontal", "vertical", "p");
            return new RandomFlip(o.Flag("horizontal", true), o.Flag("vertical", false), o.Number("p", 0.5));
        });
        Augmentations.Register("shift", o =>
        {
            o.Allow("pixels", "fill");
            return new RandomShift(o.Integer("pixels", 0), (float)o.Number("fill", 0));
        });
    }
}
