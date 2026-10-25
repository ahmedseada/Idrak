# Idrak.Imaging

Images for Idrak (no dependencies beyond Idrak): PNG, JPEG, BMP and Netpbm decoders and encoders written in C#, Pillow's
resampling, filters and enhancements byte for byte, the preprocessors of image models (resize, crop, rescale, normalize,
on the device too), chat image decoding, sample augmentations and image-folder datasets.

## Install

```bash
dotnet add package Idrak.Imaging
```

## What is in the package

| Area | What is there |
|------|---------------|
| Codecs | `PngCodec`, `JpegCodec` (baseline and progressive, decoded to Pillow's RGB), `BmpCodec`, `NetpbmCodec`; `PngEncoder`, `JpegEncoder`, `NetpbmEncoder`; registered in `ImageCodecs` (Idrak.Abstraction) when the registry is first used, so `ImageCodecs.Decode(path)` reads them |
| Transforms | `PillowImageOps` (resize with Pillow's filters, rank filters, blur, blends, point maps), the named transforms of `ImageTransforms` ("grayscale", "max_width", "contrast", ...), and scan degradations ("rotate", "perspective", "noise", "speckle", "fade", "tint", "stain", "lowres", "bleed"; seeded) mixed by the scan profiles of `ScanProfiles` ("clean", "office", "old", "fax", or one written as text) |
| Preprocessing | `ImagePreprocessor`: a model family's resize, crop, rescale and normalization from its preprocessor config, on the host or the device |
| Chat images | `ChatImageDecoder`: the images of a chat request (data URLs, files, bytes) decoded once |
| Fonts | `OpenTypeFormat` ("opentype": .ttf, .otf and .ttc with TrueType or CFF outlines, in C#; registered in `FontFormats`), `CoverageRasterizer` ("coverage": exact area coverage, unhinted; in `OutlineRasterizers`); text shaping in `TextShapers` ("general" for every script, "arabic" for the joining scripts: HarfBuzz's OpenType path, every GSUB and GPOS lookup kind) with the Unicode bidirectional algorithm (`Bidi`, UAX #9 complete with embeddings, isolates and paired brackets, checked against the Unicode conformance files) and `ArabicPresentationForms` (shaping to presentation forms for terminals and fonts without layout tables), glyph for glyph as HarfBuzz on real Arabic fonts (Amiri, Cairo, Reem Kufi, IBM Plex Sans Arabic, Tajawal); `OpenTypeFeatures` (a font's layout features for a script); `TextLayout` (lines: wrapping, direction, alignment, justification, fallback fonts) and `TextRenderer` |
| Pages | the page layouts of `PageLayouts` ("letter": an official letter with a three-place header, emblem, rule, subject, body, signature, stamp and footer; "plain"): a `PageContent` drawn in a `PageStyle`, every choice from the page's seed, with its text in reading order (plan 18) |
| Training | `ImageFolderSource` (class folders, and the "images" source of `SampleSources`), augmentations for the `DataLoader` (`RandomFlip`, `RandomShift`, `RandomRotation`, `GaussianNoise`; "flip" and "shift" in `Augmentations`) |

## Your own scan or paper effect

Every effect a synthetic page gets is an `IImageTransform` in the `ImageTransforms` registry, the library's own
included, so an effect of yours is one class and one registration. It is then usable by name wherever the library's
are: in a scan profile's text (`ScanProfile`, `--scan "coffee=1..3@0.4"`), in `ScanProfiles` as a look of its own, and
in any image pipeline.

```csharp
ImageTransforms.Register(new CoffeeRing());                    // Name "coffee", Keys ["seed"], Check, Apply
ScanProfiles.Register(new ScanProfile("cafe", "coffee=1..3@0.5,blur=0.3..0.8@0.3", "rings from a cup"));
```

From the command line, put the registration in a plug-in (`IdrakPlugin`) of an assembly that declares its package
(`[assembly: IdrakPluginPackage(typeof(MyEffects))]`, `MyEffects : IIdrakPluginPackage`) and load it with `-P`:
`idrak -P my-effects.dll data synth text.txt -o pages.jsonl --fonts DIR --scan cafe --scan mixed`.

## Contracts

Every contract this package serves is shared, in Idrak.Abstraction (`ImageCodecs`, `ImageEncoders`, `ImageTransforms`,
`Augmentations`, `SampleSources`, `ImageData`, `FontFormats`, `OutlineRasterizers`, `TextShapers`, `PageLayouts`, `ScanProfiles`), and this package registers its built-ins into them.
