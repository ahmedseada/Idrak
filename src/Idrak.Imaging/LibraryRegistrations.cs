// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;

namespace Idrak.Imaging;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction, which cannot name it: the image codecs (png, jpeg, bmp,
/// netpbm) and encoders (png, netpbm), Pillow's image transforms byte for byte, the flip and shift augmentations (Idrak.Vision adds the others) and
/// the "images" sample source, the "opentype" font format and the "coverage" outline rasterizer. <c>LibraryDefaults.Ensure</c> (in Idrak.Abstraction) calls <see cref="RegisterFor"/> once
/// per registry, before that registry is first used.
/// </summary>
internal static class LibraryRegistrations
{
    private static readonly Dictionary<Type, Action> ByRegistry = new()
    {
        [typeof(ImageTransforms)] = () =>
        {
            LibraryImageTransforms.RegisterDefaults();   // Pillow's operations, byte for byte
            ScanTransforms.RegisterDefaults();           // scan degradations (plan 18)
        },
        [typeof(ScanProfiles)] = ScanTransforms.RegisterProfiles,
        [typeof(Augmentations)] = LibraryAugmentations.RegisterDefaults,       // flip and shift (Idrak.Vision adds the others)
        [typeof(ImageCodecs)] = RegisterImageCodecs,
        [typeof(ImageEncoders)] = () =>                                          // png, netpbm
        {
            ImageEncoders.Register(new PngEncoder());
            ImageEncoders.Register(new NetpbmEncoder());
        },                            // png, jpeg, bmp, netpbm (asked newest first)
        [typeof(SampleSources)] = () => SampleSources.Register("images", OpenImages),
        [typeof(FontFormats)] = () => FontFormats.Register(new OpenTypeFormat()),            // TrueType outlines, single fonts and collections
        [typeof(OutlineRasterizers)] = () => OutlineRasterizers.Register(new CoverageRasterizer()),
        [typeof(TextShapers)] = RegisterShapers,                                 // general for every script, arabic for the joining scripts
        [typeof(PageLayouts)] = () =>                                            // letter, plain
        {
            PageLayouts.Register(new LetterLayout());
            PageLayouts.Register(new PlainLayout());
        },
    };

    private static void RegisterImageCodecs()
    {
        foreach (var codec in (IImageCodec[])[new NetpbmCodec(), new BmpCodec(), new JpegCodec(), new PngCodec()])
        {
            ImageCodecs.Register(codec);
        }
    }

    private static void RegisterShapers()
    {
        TextShapers.Register(TextShapers.AnyScript, new Shaping.OpenTypeShaper());
        var arabic = new Shaping.ArabicShaper();
        foreach (string script in Shaping.ArabicShaper.Scripts)
        {
            TextShapers.Register(script, arabic);
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
