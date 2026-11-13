# Plan 38: handwriting fonts from images (a person's hand, read from a scan, used as a font)

**Not started.**

**Status (2026-11-13, `imaging`):** planned 2026-11-13 from the owner's question "can Idrak extract a font from an image
and use it?", against the code at b96ff2e. **Nothing is built.** No phase starts before the owner names it.

| Phase | What | Lane | State |
|---|---|---|---|
| F0 | The contracts and registries (`IGlyphSource` / `GlyphSources`, `IOutlineTracer` / `OutlineTracers`), the face made from glyphs, its file format (`IFontFormat` "glyphs"), their suites | A | Planned |
| F1 | The tracer: a glyph's pixels to an outline (contours, simplified, curves fitted, one winding) | A | Planned |
| F2 | The glyph sheet: a blank sheet drawn by the shape pipeline, the filled sheet read back (found, straightened, cut, cleaned, traced) | A | Planned |
| F3 | The face: metrics, advances and the shaping data from the glyphs (the legacy-font piece's synthesized tables), missing glyphs drawn | A | Planned |
| F4 | Glyphs from real scans with their transcription: separate characters (digits in cells, unjoined letters) | B | Planned |
| F5 | Joined handwriting: a written word cut into its letters by alignment with its transcription | B | Research, after F4 |
| F6 | The command line: `idrak data font-sheet` and `idrak data font-from`, and `data synth` reading the made fonts | A | Planned |

## Goal

A person's handwriting, taken from images, becomes a font Idrak uses like any other: `idrak data synth --fonts DIR` draws
pages in that hand, so a reader is tuned on writing that looks like the real scans. The easy, reliable route first (a
sheet the writer fills in, box by box), then glyphs taken from real scans whose text is known (a register's score cells),
then joined handwriting (research).

## What exists (found in the source at b96ff2e)

- Line segmentation: `ITextLineSegmenter` / `TextLineSegmenters`, `ProjectionLineSegmenter` (Idrak.Vision), with its
  suite (`TextLineSegmenterSuite`).
- Connected components: `ConnectedComponents` (Idrak.Vision): the separate pieces of ink on a page.
- Scan looks to test against: `ScanProfiles`, the image transforms (Idrak.Imaging).
- Fonts used from memory, not files: `IFontFace`, `IFontFormat` / `FontFormats`, `GlyphOutline`, the shapers reading
  GSUB / GDEF / GPOS through `IFontFace.Table` (Idrak.Abstraction, Idrak.Imaging).
- Being built now (needed here): the shape pipeline (`Shapes`, grids with regions; draws the blank sheet), the
  legacy-font support (a face given Unicode, synthesized forms, lam-alef, marks with anchors, missing digits, letters and
  marks drawn from pen strokes measured from the font). F3 reuses the second wholesale.

What is missing: tracing pixels to outlines; knowing which ink is which character; a face made from traced glyphs and a
way to keep it on disk.

## Decisions

**Taken by the owner (2026-11-13):** plan only, nothing implemented; the glyph sheet and digits from real cells come
first.

**Proposed, for the owner to confirm:**

1. **No font-file writer.** A made font is kept in Idrak's own file, a JSON of outlines and their characters
   ("glyphs" format, `.glyphs.json`), read by a built-in `IFontFormat`, so `--fonts DIR` takes it beside `.ttf` files.
   Writing TrueType (glyf, cmap, hmtx, GSUB...) is several times the work and is not needed to draw pages; it can be a
   later phase if a made font must leave Idrak.
2. **Language-neutral.** The library knows sheets, cells, ink, outlines, characters and joining forms. Which characters
   and forms a sheet asks for is data (a character list per sheet, e.g. an Arabic sheet's letters in their four forms,
   digits, marks); no C# names a script.
3. **Placement:** the contracts in `Idrak.Abstraction.Data` (any package can use them); tracer, face and format in
   Idrak.Imaging beside the fonts; finding glyphs on a page in Idrak.Vision, which already segments; the commands in
   Idrak.Cli.Commands.

## Rules that hold for every phase

CLAUDE.md and [docs/optimization.md](../../docs/optimization.md) apply in full. The ones that shape this plan:

- **Contract plus registry.** `GlyphSources` and `OutlineTracers` are guarded `KindTable` registries (Register,
  Unregister, Names, Get "not registered", Default, Origin, SetPolicy), library defaults through `LibraryDefaults.Ensure`,
  each with a suite in Idrak.Abstraction.Testing; inventory and API dumps regenerated.
- **No platform dependence (rule 70).** Pure C#; no potrace, no outside program; `PortableMath` for any trigonometry.
- **Results proven (rule 71).** Existing fonts and pages are untouched: a page drawn without a made font is
  byte-identical.
- **Memory and speed.** Images walked row by row (rule 72); one pass per contour (rule 73); a sheet's cells traced in
  parallel and checked to pay (rule 79); pooled buffers returned in `finally`.
- **Nothing fixed.** Every threshold (ink level, smoothing, corner angle, simplification tolerance) is a setting with a
  measured or stated default.

## Phases

### F0: contracts

- `IGlyphSource : IConfigurableKind`: from an image (and, where it has one, the image's known text) to labelled glyph
  images: each a character, its joining form, its ink as coverage, and where it sat (baseline, size). Built-ins later:
  "sheet" (F2), "cells" (F4).
- `IOutlineTracer : IConfigurableKind`: coverage to a `GlyphOutline` (font units, y up, one winding), with the tolerance
  it promises. Built-in: "contour" (F1).
- A face made from glyphs (`IFontFace`): characters, outlines, advances, metrics; the shaping tables come from the
  legacy-font piece (F3).
- `IFontFormat` "glyphs": reads and writes the made font's JSON.
- Suites: tracer (a drawn outline rasterized and traced back stays within the promised tolerance; one winding; holes
  kept), glyph source (every glyph labelled once, inside its cell or its component), the "glyphs" format (round trip).

### F1: the tracer ("contour")

Threshold the coverage (setting), find each contour's boundary (one pass, marching squares), keep holes as opposite
contours, simplify (Douglas-Peucker, tolerance a setting), fit quadratic curves where the boundary bends smoothly,
corners where it turns sharply (angle a setting), scale to font units. Tested on outlines from repo fonts: rasterize,
trace, rasterize again, compare pixels.

### F2: the glyph sheet

- **The blank sheet** (`idrak data font-sheet`): drawn by the shape pipeline: a grid of cells, each with a faint guide
  (baseline, x-height) and a small printed label of what to write; corner marks to find the sheet in a scan or a phone
  photo; the character list as data; several pages when the list is long.
- **The filled sheet read back** (source "sheet"): find the corner marks, straighten the page (perspective), cut each
  cell, remove the guide and the label (they are known: drawn by the sheet), clean specks (a setting), trace the ink
  (F1), keep the baseline and size from the guides.
- Tested without a human: a sheet filled by drawing a repo font into the cells, put through scan looks (rotate,
  perspective, noise), read back; every glyph found, labelled right, and its outline close to the font's.

### F3: the face

Metrics from the sheet's guides (baseline, x-height, ascender and descender from the ink); advances and side bearings
from each glyph's ink; joining forms and lam-alef as the sheet labelled them, so the legacy-font piece's synthesized
GSUB / GDEF / GPOS make it shape; marks anchored from the ink as that piece does; whatever the sheet lacks drawn from pen
strokes measured from the written glyphs (the same mechanism). Several fills of the same character are kept as
alternates (OpenType `salt`/`aalt` or a random pick per glyph, a setting), so a made font does not repeat one shape.

### F4: glyphs from real scans with their transcription

For text whose characters do not touch (digits in a register's score cells, unjoined letters): segment lines
(`TextLineSegmenters`), find the ink pieces (`ConnectedComponents`), merge a character's pieces (dots, marks) by
position, align the pieces with the transcription's characters in reading order, keep only alignments that agree
(counts, sizes, order) and report the rest. The owner's register gives each teacher's own 0-9 from the score cells.

### F5: joined handwriting (research)

A written word is one ink piece: cut it into its letters by aligning it with its transcription (joining forms known from
the text, cut points where the stroke is thinnest along the baseline, checked against the forms' expected widths). Built
only after F4, and only kept where its glyphs make better pages than a sheet's.

### F6: the command line

- `idrak data font-sheet --chars FILE -o sheet.pdf|png` (the blank sheet; the character list as data).
- `idrak data font-from IMAGES... --source sheet|cells [--text FILE] -o FONT.glyphs.json` (live progress: images read,
  glyphs found, traced, left out and why).
- `idrak data synth --fonts DIR` reads `.glyphs.json` beside `.ttf` (through the "glyphs" format, nothing else changes).
- `idrak data fonts DIR` shows a made font's coverage like any other.

## Not in this plan

Writing TrueType or OpenType files; a model that generates new handwriting (that is plan 20's kind of work); recognizing
text it is not given (F4 and F5 need the transcription).

## Open questions for the owner

1. The made font's file: Idrak's own JSON (proposed), or TrueType too?
2. Which character list goes on the first sheet (letters in their four forms, both digit sets, marks, punctuation)?
3. The first real scans for F4: the register crops in `test-output\synth\`?
