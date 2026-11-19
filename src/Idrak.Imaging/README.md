# Idrak.Imaging

Images for Idrak (no dependencies beyond Idrak): PNG, JPEG, BMP and Netpbm decoders and encoders written in C#, Pillow's
resampling, filters and enhancements byte for byte, the preprocessors of image models (resize, crop, rescale, normalize,
on the device too), chat image decoding, sample augmentations and image-folder datasets. Fonts read (TrueType, CFF,
Type 1) and written (TrueType), and the pieces documents are drawn and written with: any document page drawn to pixels,
text set in any script for a form field, pages of shapes written as vector PDF (Idrak.Data reads and writes the files).

## Install

```bash
dotnet add package Idrak.Imaging
```

## What is in the package

| Area | What is there |
|------|---------------|
| Codecs | `PngCodec`, `JpegCodec` (baseline and progressive, decoded to Pillow's RGB; a file of one scan decoded a strip of MCU rows at a time, so a large image costs its pixels and a strip), `BmpCodec`, `NetpbmCodec`; `PngEncoder`, `JpegEncoder`, `NetpbmEncoder`; registered in `ImageCodecs` and `ImageEncoders` ("png", "jpeg", "netpbm"; Idrak.Abstraction) when the registry is first used, so `ImageCodecs.Decode(path)` reads them and `ImageEncoders.Save(path, image)` writes them |
| Transforms | `PillowImageOps` (resize with Pillow's filters, rank filters, blur, blends, point maps), the named transforms of `ImageTransforms` ("grayscale", "max_width", "contrast", ...), and scan degradations ("rotate", "perspective", "noise", "speckle", "fade", "tint", "stain", "lowres", "bleed"; seeded) mixed by the scan profiles of `ScanProfiles` ("clean", "office", "old", "fax", or one written as text) |
| Preprocessing | `ImagePreprocessor`: a model family's resize, crop, rescale and normalization from its preprocessor config, on the host or the device |
| Chat images | `ChatImageDecoder`: the images of a chat request (data URLs, files, bytes) decoded once |
| Fonts | `OpenTypeFormat` ("opentype": .ttf, .otf and .ttc with TrueType or CFF outlines, in C#; registered in `FontFormats`), "cff" (a bare CFF program) and "type1" (PFB, PFA, and a PDF's embedded Type 1 program: eexec, subroutines, flex, accents), read by their first bytes with their glyph names and built-in encodings, `TrueTypeWriter` ("truetype" in `FontWriters`: any face as a .ttf, whole or a subset that keeps its glyph ids, with its layout tables and their closure or without them; a legacy font's mapped face is converted once into a complete TrueType font and kept by `FontCache`, `IDRAK_FONT_CACHE`; WOFF, read and written, comes from the `Idrak.Imaging.Plugins` package, a plug-in on these registries), `CoverageRasterizer` ("coverage": exact area coverage, unhinted; in `OutlineRasterizers`); text shaping in `TextShapers` ("general" for every script, "arabic" for the joining scripts: HarfBuzz's OpenType path, every GSUB and GPOS lookup kind) with the Unicode bidirectional algorithm (`Bidi`, UAX #9 complete with embeddings, isolates and paired brackets, checked against the Unicode conformance files) and `ArabicPresentationForms` (shaping to presentation forms for terminals and fonts without layout tables), glyph for glyph as HarfBuzz on real Arabic fonts (Amiri, Cairo, Reem Kufi, IBM Plex Sans Arabic, Tajawal); `OpenTypeFeatures` (a font's layout features for a script); `TextLayout` (lines: wrapping, direction, alignment, justification, fallback fonts) and `TextRenderer` |
| Pages | the page layouts of `PageLayouts` ("letter": an official letter with a three-place header, emblem, rule, subject, body, signature block with a pen-drawn signature, stamp and footer, the emblem, stamp and signature shape profiles drawn by pens; "plain"): a `PageContent` drawn in a `PageStyle`, every choice from the page's seed, with its text in reading order (plan 18) |
| Documents | the page drawer "coverage" (`DocumentPageDrawers`: any `DocumentPage`, whatever format read it, drawn to pixels), the text setter "shaped" (`DocumentTextSetters`: text set into lines of placed glyphs in any script, as a form field's appearance needs), `DocumentShapes` and `DocumentShapeStyle` (a page of shapes and laid-out text written through any `IDocumentPageBuilder` as vector paths and real text); see "Documents" below |
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

## Fonts written

`FontWriters` writes any face (`IFontFace`, whatever read it) as a font file; the built-in writer is "truetype"
(`TrueTypeWriter`, `.ttf`), the format Idrak works from. WOFF, read and written, comes from the `Idrak.Imaging.Plugins`
package, a plug-in on the same registries.

```csharp
IFontFace face = FontFormats.Read("fonts/sans.otf")[0];                   // TrueType or CFF outlines
FontWriters.Save(face, "out/sans.ttf");                                    // whole, with its layout tables
FontWriters.Save(face, "out/sans-subset.ttf", new FontWriteOptions { Glyphs = [0, 36, 37, 38, 39] });
```

- **Whole or a subset:** `FontWriteOptions.Glyphs` keeps those glyphs and the parts of composite glyphs, with
  `KeepLayout` (on by default) every glyph the substitutions reach, and keeps the glyph ids (the others written empty),
  so text already placed by glyph id draws the same; without `KeepLayout` the layout tables are left out (what a PDF
  embedding needs). Cubic outlines become quadratic within `CurveTolerance` font units (1 unless
  `IDRAK_FONT_CURVE_TOLERANCE` sets it); hinting is kept only for a whole face with `KeepHinting`; `Names` replaces name
  strings, the face's own (its copyright and licence) kept. The same face and options write the same bytes. The PDF
  writer embeds its subsets through this writer.
- **Not written:** OpenType with CFF outlines, WOFF2, a variable font's variations (its default instance is written),
  and the `kern`, `BASE`, `JSTF` and `MATH` tables.
- **Legacy fonts converted once:** a face a font map makes complete (`FontMaps`: a legacy layout's characters, the glyphs
  drawn for what it lacks, its ligatures and pairs as layout tables) is written once as a TrueType font and kept by
  `FontCache` in a folder (`IDRAK_FONT_CACHE`; default `fonts` under the cache folder; `0`, `off`, `false` or `no`: not
  kept, mapped and drawn at every read; `--font-maps cache=on|off|DIR` in the `idrak` tool), named by a hash of the
  font, its map and the drawing settings, written only where the drive has room. `FontFormats.Read` then opens it as a
  plain TrueType face: a few milliseconds instead of mapping and drawing again, and the same outlines and pages either way.

## Documents

**Pages drawn to pixels.** The page drawer "coverage" (registered in `DocumentPageDrawers`, used by
`IDocument.DrawPage` and `DocumentPageDrawers.DrawPages`) draws any `DocumentPage`, a scan's image or a vector page,
whatever format read it: fills by their rule, strokes through `Pens` with their caps, joins and dashes, glyph runs
through their faces' outlines (a font the document does not embed taken from `DocumentDrawOptions.Fonts`,
`IDRAK_DOCUMENT_FONTS`, by name and then by the characters it covers, stretched to the document's widths, slanted and
emboldened as its name asks; none found: left out and counted as `missing-font`), images placed by their matrix with
their masks, clips, shadings, tiling patterns, transparency groups, soft masks and the sixteen blend modes, and, when
`DocumentParts.Annotations` is asked, each annotation's appearance. A page is drawn band by band, the bands spread over
the cores, so a large page at a high resolution holds one band of working memory a core. Its numbers are
`DocumentDrawOptions` settings, each refused out of its range naming itself:

| Setting | Variable | Default | What it sets |
|---|---|---|---|
| `Dpi` (or `PixelsPerPoint`) | `IDRAK_DOCUMENT_DPI` | 150 | The resolution |
| `Colour`, `Background`, `Parts`, `Hidden` | | RGB, white, text, images and paths, false | Grey or RGB pixels, the paper, what is drawn, whether a scan's hidden text is drawn |
| `Fonts` | `IDRAK_DOCUMENT_FONTS` | none | A folder of fonts for text whose font is not embedded |
| `BandPixels` | `IDRAK_DOCUMENT_BAND_PIXELS` | 131,072 | About how many pixels a band holds (1 to 67,108,864); `MinBandRows` 8 and `MaxBandRows` 1,024 bound its rows |
| `ShadingSteps` | `IDRAK_DOCUMENT_SHADING_STEPS` | 1,024 | The table axial and radial shadings are drawn through (2 to 1,048,576) |
| `MaxTilePixels` | `IDRAK_DOCUMENT_MAX_TILE_PIXELS` | 1,048,576 | The largest tile a tiling pattern is drawn into (1 to 67,108,864) |
| `FillFlatness` | `IDRAK_DOCUMENT_FILL_FLATNESS` | 1/32 pixel | How closely fills, glyphs and clips follow their curves (0.001 to 16) |
| `StrokeFlatness` | `IDRAK_DOCUMENT_STROKE_FLATNESS` | 0.1 pixel | How closely strokes and their round joins, caps and dots follow their curves (0.001 to 16) |
| `MinSubstituteStretch`, `MaxSubstituteStretch`, `SubstituteSlant`, `SubstituteStroke` | | 0.75, 4/3, 0.21, 0.03 | How a face found for a font that is not embedded is fitted: stretched to the widths, slanted for an italic, stroked for a bold (an app's choice of look, in code) |
| `Drawer`, `Rasterizer`, `Pen` | | "coverage", "coverage", "ideal" | The pieces it draws with, by name, each replaceable |

The drawer's rasterizer fills outlines by the nonzero or even-odd rule (`IOutlineRasterizer.Fill` with an `OutlineFill`;
glyphs keep their own rule, unchanged) and draws the same pixels whatever the bands. The drawing's `Issues` say what was
left out (`missing-font`, `not-drawn`), never thrown.

**Text set for a document.** The text setter "shaped" (`DocumentTextSetters`) sets text into lines of placed glyphs:
each paragraph through the Unicode bidirectional algorithm, cut into runs of one script and direction, each shaped by
its script's shaper in the first face that has its characters, lines broken at spaces within a width. The PDF writer
lays a form field's value out with it, so a field fills right in every script; an app writing text sets it the same way
(Idrak.Data's README shows a page written with it). Nothing in it is for one language or script.

**Pages of shapes written.** `DocumentShapes.Write(page, plan, style)` writes a `ShapePlan`'s areas and strokes, and
`DocumentShapes.Write(page, block, text, x, y, style)` a `TextBlock`'s lines (as `TextLayout` laid them out), through any `IDocumentPageBuilder`:
a synthetic page or a glyph sheet as vector paths and real, searchable text, not only as an image.

## Contracts

Every contract this package serves is shared, in Idrak.Abstraction (`ImageCodecs`, `ImageEncoders`, `ImageTransforms`,
`Augmentations`, `SampleSources`, `ImageData`, `FontFormats`, `FontWriters`, `FontCache`, `OutlineRasterizers`,
`TextShapers`, `PageLayouts`, `ScanProfiles`, `DocumentPageDrawers`, `DocumentTextSetters`), and this package registers its built-ins into them.
