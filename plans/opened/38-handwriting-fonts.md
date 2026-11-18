# Plan 38: handwriting fonts from images (a person's hand, read from a scan, used as a font)

**Not started.**

**Status (2026-11-18, `imaging`):** planned 2026-11-13 from the owner's question "can Idrak extract a font from an image
and use it?", against the code at b96ff2e; refreshed 2026-11-18 against the code at da19bfc (the pieces it was waiting
on have landed, see "What exists"). **Nothing is built.** Everything PDF moved to [plan 28](28-documents.md), and
font writing (TrueType built in, WOFF from a plug-in) is plan 28's D8. Plan 28 is built and finished first, for any workload,
however long it takes (owner, 2026-11-18): this plan waits until every plan 28 phase is closed. Worked on the branch
`documents` (taken from `migration`). No phase starts before the owner names it.

| Phase | What | Lane | Needs | State |
|---|---|---|---|---|
| F0 | The contracts and registries (`IGlyphSource` / `GlyphSources`, `IOutlineTracer` / `OutlineTracers`), the face made from glyphs (written through plan 28's `FontWriters`), their suites | A | plan 28 D8 | Planned |
| F1 | The tracer: a glyph's pixels to an outline (contours, simplified, curves fitted, one winding) | A | F0 | Planned |
| F2 | The glyph sheet: a blank sheet drawn by the shape pipeline (PNG and PDF), the filled sheet read back from an image or a PDF (found, straightened, cut, cleaned, traced) | A | F1; plan 28 D4, D6, D7 | Planned |
| F3 | The face: metrics, advances and the shaping data from the glyphs (the font maps' synthesized tables), missing glyphs drawn | A | F0 | Planned |
| F4 | Glyphs from real scans with their transcription: separate characters (digits in cells, unjoined letters) | B | F1, F3; plan 28 D4, D7 for PDF scans | Planned |
| F5 | Joined handwriting: a written word cut into its letters by alignment with its transcription | B | F4 | Research, after F4 |
| F6 | The command line: `idrak data font-sheet` and `idrak data font-from` (a `.ttf`, or a `.woff` through the plug-in, out), and `data synth` reading the made fonts | A | F2, F3; plan 28 D6, D8 | Planned |

## Goal

A person's handwriting, taken from images, becomes a font Idrak uses like any other: `idrak data synth --fonts DIR` draws
pages in that hand, so a reader is tuned on writing that looks like the real scans. The easy, reliable route first (a
sheet the writer fills in, box by box), then glyphs taken from real scans whose text is known (digits written in a
table's cells), then joined handwriting (research).

## What exists (found in the source at da19bfc)

- **Line segmentation:** `ITextLineSegmenter` / `TextLineSegmenters`, `ProjectionLineSegmenter` (Idrak.Vision), with its
  suite (`TextLineSegmenterSuite`).
- **Ink on a page:** `ConnectedComponents.Find` (Idrak.Vision; regions with box, area and centre, a label per pixel, four
  or eight neighbours, a minimum area); `Foreground.Extract` and `Foreground.Otsu` (an automatic ink threshold);
  `ContentFrame` (a region cut and fitted into a box).
- **Scan looks to test against:** `ScanProfiles` and the image transforms (Idrak.Imaging), `perspective` among them.
  `Homography` and `Warp` (`src/Idrak.Imaging/ScanTransforms.cs:262, 320`) are private in the internal `ScanImageOps`;
  plan 28's D7 makes them a shared helper, which F2's straightening uses.
- **Fonts used from memory, not files:** `IFontFace`, `IFontFormat` / `FontFormats` (built in: `opentype`, TrueType and
  CFF outlines), `GlyphOutline` (move, line, quadratic, cubic, close; nonzero winding), the shapers reading GSUB / GDEF /
  GPOS through `IFontFace.Table` (Idrak.Abstraction, Idrak.Imaging).
- **Outlines to pixels:** `OutlineRasterizers` "coverage" (exact area coverage), the round trip F1 is tested with.
- **The shape pipeline (landed, 713492b):** `Shapes` with `grid`, `box`, `line`, `polyline`, `polygon`, `circle`,
  `ellipse`, `arc`, `arrow`, `fill`, `hatch` and `text`; every shape's common options include `dash` and `tone` (a faint
  guide), `at`, `size`, `in`, `margin`; `ShapeRegion` (a cell's box for text); `Pens` (`ideal`, `ballpoint`);
  `ShapeProfiles`; drawn as `ShapePath` and `ShapeArea` before rasterizing, so plan 28's D6 can write a sheet as vector
  PDF.
- **Glyph poses (landed, 947fef5):** `GlyphPoses` (`none`, `drift`): each glyph drawn a little differently.
- **Legacy fonts through font maps (landed, 0d77282 and six follow-ups):** `FontMap` (JSON: letters by positional form
  `isol` / `init` / `medi` / `fina`, characters, aliases, ligatures, marks with their place, short-base marks, fallbacks,
  anchors, the characters measured for size); `IFontMapper` / `FontMappers` "opentype" (a face given Unicode and the
  GSUB / GDEF / GPOS made from the map, marks anchored from the ink); `DrawnGlyphs` (digits, marks and composed letters
  drawn from pen strokes measured from the font); letter pairs (`PAIR`), a second style (`PAI2`) and whole words
  (`WORD`), each at a page chance. A glyph is referred to by `#id`, so a face with no `cmap` or `post` table (a made
  face) is mapped like any other. F3 reuses all of it.
- **Images out:** `ImageEncoders` `png` and `netpbm`.
- **Test fonts:** DejaVu (`tests/Idrak.Tests/data/fonts/`) and eight Arabic fonts, hand styles among them
  (`tests/Idrak.Tests/data/fonts/arabic/`), for F1's round trip and F2's filled-by-a-font sheets.
- **The command line:** `idrak data fonts`, `idrak data synth --fonts DIR` (fonts read through `SyntheticPages.Fonts`,
  that is `FontFormats.ReadFolder`, so a registered format is picked up with nothing else changed).

**What is missing:** tracing pixels to outlines; knowing which ink is which character; a face made from traced glyphs;
finding a sheet's corner marks; writing a font file in a common format (plan 28, D8); PDF in and out (plan 28).

## Gaps found in the code (2026-11-18)

| Sev | Gap | Where | What F0-F6 does about it |
|---|---|---|---|
| 🟢 | ~~A made font named `X.glyphs.json` clashes with the font maps beside fonts (`ReadFolder` matches `Path.GetExtension`, `.json`)~~ | `src/Idrak.Abstraction/Data/Fonts.cs:244` | Gone: a made font is a `.ttf`, `.otf`, `.woff` or `.woff2` (owner, 2026-11-18), no Idrak-only file |
| 🟢 | ~~A format decides from 16 bytes; every JSON file starts with `{`~~ | `src/Idrak.Abstraction/Data/Fonts.cs:161` | Gone with it: the common formats have their own signatures (`0x00010000`, `OTTO`, `wOFF`, `wOF2`) |
| 🟡 | `IFontFormat` reads only; a made font must be written | `src/Idrak.Abstraction/Data/Fonts.cs:134` | Plan 28 D8: `IFontWriter` / `FontWriters` ("truetype", "opentype-cff", "woff", "woff2"), and `woff` / `woff2` readers |
| 🟢 | ~~The map beside `X.glyphs.json` would be `X.glyphs.map.json`~~ | `src/Idrak.Abstraction/Data/FontMaps.cs:917` | Gone: `FONT.ttf` and `FONT.map.json`, as for any font; a made font needs no map once written (its layout tables are in the file) |
| 🟡 | `FontMap` is made only from JSON (`Read`, `Parse`), with no builder | `src/Idrak.Abstraction/Data/FontMaps.cs:244-254` | F3 writes the map as JSON bytes and parses them, or adds a builder |
| 🟡 | `salt` / `aalt` (alternate glyphs) are not in the map's keys | `src/Idrak.Abstraction/Data/FontMaps.cs:177` | F3: several fills of a character as a map key, or a random pick per glyph |
| 🟡 | No PDF anywhere in `src/` | — | Plan 28 (D0-D7), finished before this plan |

## Decisions

**Taken by the owner (2026-11-13):** plan only, nothing implemented; the glyph sheet and digits from real cells come
first.

**Taken by the owner (2026-11-18):**

- **PDF is built in, read and written, vector and image pages,** and lives in [plan 28](28-documents.md) (the document
  contract, the reader and writer in `Idrak.Data/Pdf/`, the page drawer in Idrak.Imaging). This plan's PDF needs moved
  there ("Plan 38's needs").
- **Plan 28 first, for any workload.** Plan 28 is built and finished (all its phases, D0-D13) before this plan starts,
  however long it takes, so no phase here waits on PDF or on font writing. **Except, for the session of 2026-11-18**
  (owner: "for this session slot 4 takes what could be finished from phase 38"): a slot with no ready plan 28 piece
  takes this plan's phases that need nothing unmerged — F0, then F1, then F3 (plan 28's decision 23).
- **A made font is exported as TrueType** (`.ttf`), through plan 28's `FontWriters` (D8), or as WOFF through the one
  font plug-in (plan 28, decision 20: one plug-in format, only to prove the path). This replaces the proposed
  Idrak-only JSON file ("glyphs" format): there is none. TrueType is built into Idrak and is what Idrak works from (the
  fastest to open and draw from).

**Proposed, for the owner to confirm:**

1. **The made font is the exported file.** The face F3 makes (outlines, metrics, and the GSUB / GDEF / GPOS its font map
   made) is written straight to `.ttf` (default) or `.woff` (the plug-in), and read back by `FontFormats` like any
   font, so `--fonts DIR`, other software and the web take it with nothing else changed. Several fills of a character
   go in as alternates (`salt` and `aalt` in the written GSUB, and a random pick per glyph in Idrak, a setting); the
   family and style names are what the user gives (`--name`), nothing invented.
2. **Language-neutral.** The library knows sheets, cells, ink, outlines, characters and joining forms. Which characters
   and forms a sheet asks for is data (a character list per sheet, e.g. one script's letters in their four forms,
   digits, marks); no C# names a script.
3. **Placement:** the contracts in `Idrak.Abstraction.Data` (any package can use them); tracer and made face in
   Idrak.Imaging beside the fonts and their writers; finding glyphs on a page in Idrak.Vision, which already segments; the commands in
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
- **Generic examples only.** Test sheets, scans and texts are generic (an ordinary letter, digits in a table, business or
  technology text), never one domain's documents.

## Phases

### F0: contracts

- `IGlyphSource : IConfigurableKind`: from an image (and, where it has one, the image's known text) to labelled glyph
  images: each a character, its joining form, its ink as coverage, and where it sat (baseline, size). Built-ins later:
  "sheet" (F2), "cells" (F4).
- `IOutlineTracer : IConfigurableKind`: coverage to a `GlyphOutline` (font units, y up, one winding), with the tolerance
  it promises. Built-in: "contour" (F1).
- A face made from glyphs (`IFontFace`): characters, outlines, advances, metrics; the shaping tables come from the font
  maps (F3); written through plan 28's `FontWriters` (D8), read back through `FontFormats`.
- Suites: tracer (a drawn outline rasterized and traced back stays within the promised tolerance; one winding; holes
  kept), glyph source (every glyph labelled once, inside its cell or its component), the made face (written as `.ttf`
  and each other format, read back: outlines, advances, characters and shaping kept; a fonts folder holding `.ttf`,
  `.map.json` and made fonts reads each as what it is).

### F1: the tracer ("contour")

Threshold the coverage (setting; `Foreground.Otsu` as the measured default), find each contour's boundary (one pass,
marching squares), keep holes as opposite contours, simplify (Douglas-Peucker, tolerance a setting), fit quadratic
curves where the boundary bends smoothly, corners where it turns sharply (angle a setting), scale to font units. Tested
on outlines from repo fonts: rasterize ("coverage"), trace, rasterize again, compare pixels.

### F2: the glyph sheet

- **The blank sheet** (`idrak data font-sheet`): drawn by the shape pipeline: a `grid` of cells, each with a faint guide
  (baseline, x-height: `line` at a low `tone`, `dash`) and a small printed label of what to write (`text`); corner marks
  to find the sheet in a scan or a phone photo; the character list as data; several pages when the list is long. Saved
  as PNG (`ImageEncoders`) or as a vector PDF through plan 28's writer (D6), the labels as searchable text.
- **The filled sheet read back** (source "sheet"): from an image (`ImageCodecs`) or a PDF (`DocumentFormats`: a scanned
  page's image, plan 28 D4; a tablet-written vector page drawn to pixels, plan 28 D7); find the corner marks, straighten
  the page (perspective, the shared `Homography` / `Warp`), cut each cell, remove the guide and the label (they are
  known: drawn by the sheet), clean specks (a setting), trace the ink (F1), keep the baseline and size from the guides.
- Tested without a human: a sheet filled by drawing a repo font into the cells, put through scan looks (rotate,
  perspective, noise), saved as PNG and as an image PDF, read back; every glyph found, labelled right, and its outline
  close to the font's.

### F3: the face

Metrics from the sheet's guides (baseline, x-height, ascender and descender from the ink); advances and side bearings
from each glyph's ink; joining forms and two-letter ligatures as the sheet labelled them, given to a font map (glyphs
by `#id`) so `FontMappers` "opentype" makes the GSUB / GDEF / GPOS that shape it; marks anchored from the ink as the map
does; whatever the sheet lacks drawn by `DrawnGlyphs` from pen strokes measured from the written glyphs (the map's
`measure`). Several fills of the same character are kept as alternates (`salt` / `aalt` in the written GSUB, and a
random pick per glyph in Idrak, a setting), so a made font does not repeat one shape; `GlyphPoses` still varies each
drawn glyph on top. The face is then written through `FontWriters` (plan 28 D8): `.ttf` by default, or `.woff` through
the plug-in.

### F4: glyphs from real scans with their transcription

For text whose characters do not touch (digits written in a table's cells, unjoined letters): segment lines
(`TextLineSegmenters`), find the ink pieces (`ConnectedComponents`), merge a character's pieces (dots, marks) by
position, align the pieces with the transcription's characters in reading order, keep only alignments that agree
(counts, sizes, order) and report the rest. A scanned table filled in by several writers gives each writer's own 0-9
from its cells. Scans come as images or as PDFs (plan 28).

### F5: joined handwriting (research)

A written word is one ink piece: cut it into its letters by aligning it with its transcription (joining forms known from
the text, cut points where the stroke is thinnest along the baseline, checked against the forms' expected widths). Built
only after F4, and only kept where its glyphs make better pages than a sheet's.

### F6: the command line

- `idrak data font-sheet --chars FILE -o test-output\fonts\sheet.pdf|sheet.png` (the blank sheet; the character list as
  data; PDF through plan 28's writer).
- `idrak data font-from IMAGES|PDFS... --source sheet|cells [--text FILE] --name FAMILY
  -o test-output\fonts\FONT.ttf|FONT.woff` (the format from the extension, through `FontWriters`; `.woff` when the
  plug-in is referenced; live progress: pages read, glyphs found, traced, left out and why).
- `idrak data synth --fonts DIR` reads made fonts like any `.ttf` (nothing else changes); its help lists `.woff` when
  the plug-in is referenced.
- `idrak data fonts DIR` shows a made font's coverage like any other.

## Not in this plan

A model that generates new handwriting (that is plan 20's kind of work); recognizing text it is not given (F4 and F5
need the transcription); writing font files (plan 28's D8) and reading, writing or drawing PDF (plan 28), both
finished first.

## Open questions for the owner

1. ~~The made font's file~~: answered 2026-11-18: exported as TrueType (built in) or WOFF (the one plug-in format)
   through plan 28's D8; no Idrak-only file. Proposal 1 (the exported file is the made font, `.ttf` by default) awaits
   confirmation.
2. Which character list goes on the first sheet (letters in their four forms, both digit sets, marks, punctuation)?
3. The first real scans for F4: which of the owner's crops under `test-output\` (generic ones: digits in a table's
   cells, unjoined letters)?
