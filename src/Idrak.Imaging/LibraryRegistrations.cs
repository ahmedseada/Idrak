// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Imaging;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the image codecs (png, jpeg, bmp,
/// netpbm), Pillow's image transforms byte for byte, the flip and shift augmentations (Idrak.Vision adds the others) and
/// the "images" sample source. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls <see cref="RegisterFor"/> once
/// per registry, before that registry is first used. The registries that live here (image encoders) register their
/// built-ins themselves.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(ImageTransforms)] = LibraryImageTransforms.RegisterDefaults,   // Pillow's operations, byte for byte
        [typeof(Augmentations)] = LibraryAugmentations.RegisterDefaults,       // flip and shift (Idrak.Vision adds the others)
        [typeof(ImageCodecs)] = RegisterImageCodecs,                            // png, jpeg, bmp, netpbm (asked newest first)
        [typeof(SampleSources)] = () => SampleSources.Register("images", OpenImages),
    };

    private static void RegisterImageCodecs()
    {
        foreach (var codec in (IImageCodec[])[new NetpbmCodec(), new BmpCodec(), new JpegCodec(), new PngCodec()])
        {
            ImageCodecs.Register(codec);
        }
    }

    private static readonly string[] ImageOptions = ["channels", "height", "width"];

    // "images": ImageFolderSource over class folders; options channels, height, width (each the first image's by default).
    private static ImageFolderSource OpenImages(string path, IReadOnlyDictionary<string, string> options)
    {
        foreach (string key in options.Keys)
        {
            if (!ImageOptions.Contains(key, StringComparer.OrdinalIgnoreCase))
            {
                throw new ArgumentException($"The images source has no option '{key}' ({string.Join(", ", ImageOptions)}).");
            }
        }

        int? Number(string name) => options.TryGetValue(name, out var v) ? int.Parse(v, CultureInfo.InvariantCulture) : null;
        return ImageFolderSource.Open(path, Number("channels"), Number("height"), Number("width"));
    }

    private static void RegisterFor(Type registry)
    {
        if (ByRegistry.TryGetValue(registry, out var register))
        {
            register();
        }
    }
}
