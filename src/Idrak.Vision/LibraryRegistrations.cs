// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Vision;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the augmentations of detection and
/// segmentation samples beyond core's flip and shift. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls
/// <see cref="RegisterFor"/> once per registry, before that registry is first used. The registries that live here (vision
/// losses, box matchers, vision metrics) register their built-ins themselves.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(Augmentations)] = RegisterAugmentations,
    };

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }

    private static void RegisterAugmentations()
    {
        Augmentations.Register("resized-crop", o =>
        {
            o.Allow("width", "height", "scale", "ratio", "min_visibility");
            return new RandomResizedCrop(o.Integer("width", 0), o.Integer("height", 0), o.Range("scale", (0.08, 1)), o.Range("ratio", (3.0 / 4, 4.0 / 3)),
                o.Number("min_visibility", 0));
        });
        Augmentations.Register("rotation", o =>
        {
            o.Allow("degrees", "fill", "p", "min_visibility");
            return RandomAffine.Rotation(o.Number("degrees", 10), (float)o.Number("fill", 0), o.Number("p", 1), o.Number("min_visibility", 0));
        });
        Augmentations.Register("affine", o =>
        {
            o.Allow("degrees", "translate", "translate_x", "translate_y", "scale", "shear", "fill", "p", "min_visibility");
            var degrees = Symmetric(o.Range("degrees", (0, 0)));
            double translate = o.Number("translate", 0);
            return new RandomAffine(degrees, (o.Number("translate_x", translate), o.Number("translate_y", translate)), o.Range("scale", (1, 1)), Symmetric(o.Range("shear", (0, 0))), (float)o.Number("fill", 0),
                o.Number("p", 1), o.Number("min_visibility", 0));
        });
        Augmentations.Register("color-jitter", o =>
        {
            o.Allow("brightness", "contrast", "saturation", "hue", "p");
            return new ColorJitter(o.Number("brightness", 0), o.Number("contrast", 0), o.Number("saturation", 0), o.Number("hue", 0), o.Number("p", 1));
        });
        Augmentations.Register("cutout", o =>
        {
            o.Allow("p", "scale", "ratio", "value", "count");
            return new Cutout(o.Number("p", 0.5), o.Range("scale", (0.02, 0.33)), o.Range("ratio", (0.3, 3.3)), (float)o.Number("value", 0), o.Integer("count", 1));
        });
        Augmentations.Register("mosaic", o =>
        {
            o.Allow("width", "height", "fill", "p", "min_visibility");
            return new Mosaic(o.Integer("width", 0), o.Integer("height", 0), (float)o.Number("fill", 0.5), o.Number("p", 1), o.Number("min_visibility", 0));
        });
        Augmentations.Register("mixup", o =>
        {
            o.Allow("alpha", "p");
            return new MixUp(o.Number("alpha", 32), o.Number("p", 1));
        });
        Augmentations.Register("resize", o =>
        {
            o.Allow("width", "height", "keep_ratio", "fill");
            int width = o.Integer("width", 0), height = o.Integer("height", 0);
            return new SampleResize(width > 0 ? width : throw new ArgumentException("resize needs width=N."), height > 0 ? height : throw new ArgumentException("resize needs height=N."),
                o.Flag("keep_ratio", false), (float)o.Number("fill", 0.5));
        });
    }

    // A single number d (the range d:d) is the range -d:d, as torchvision reads a single angle.
    private static (double, double) Symmetric((double Low, double High) range) => range.Low == range.High ? (-Math.Abs(range.Low), Math.Abs(range.Low)) : range;
}
