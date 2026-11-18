# Plan 38: handwriting fonts from images (a person's hand, read from a scan, used as a font)

**In progress (on the branch `documents`).** Built 2026-11-18 in the one slot the owner gave this plan for that
session: F0, F1 and F3. That slot has stopped (plan 28's decision 23); F2 and F4-F6 wait for plan 28 to finish
(decision 14).

**Status (2026-11-18, `imaging`):** planned 2026-11-13 from the owner's question "can Idrak extract a font from an image
and use it?", against the code at b96ff2e; refreshed 2026-11-18 against the code at da19bfc (the pieces it was waiting
on have landed, see "What exists"). Everything PDF moved to [plan 28](28-documents.md), and
font writing (TrueType built in, WOFF from a plug-in) is plan 28's D8. Plan 28 is built and finished first, for any workload,
however long it takes (owner, 2026-11-18): this plan waits until every plan 28 phase is closed. Worked on the branch
`documents` (taken from `migration`). No phase starts before the owner names it.

| Phase | What | Lane | Needs | State |
|---|---|---|---|---|
| F0 | The contracts and registries (`IGlyphSource` / `GlyphSources`, `IOutlineTracer` / `OutlineTracers`), the face made from glyphs (written through plan 28's `FontWriters`), their suites | A | plan 28 D8 | Built (2026-11-18) |
| F1 | The tracer: a glyph's pixels to an outline (contours, simplified, curves fitted, one winding) | A | F0 | Built (2026-11-18) |
| F2 | The glyph sheet: a blank sheet drawn by the shape pipeline (PNG and PDF), the filled sheet read back from an image or a PDF (found, straightened, cut, cleaned, traced) | A | F1; plan 28 D4, D6, D7 | Planned |
| F3 | The face: metrics, advances and the shaping data from the glyphs (the font maps' synthesized tables), missing glyphs drawn | A | F0 | Built (2026-11-18) |
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
  (owner: "keep the fourth always for plan 38 f0, f1, f3 the other three works on plan 28"): one agent slot is this
  plan's alone and takes F0, then F1, then F3; the other three work on plan 28 (plan 28's decision 23). After F3 that
  slot stops: no further piece from either plan; F2 and F4-F6 wait for plan 28 as decision 14 says.
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

**As built (2026-11-18):**
- **The contracts**, in `Idrak.Abstraction.Data`, language-neutral (no script named; forms are data):
  `IOutlineTracer` (`OutlineTracers.cs`: `GlyphCoverage` in, ink 0 to 1 checked once with its ink box found in the same
  pass; `OutlinePlacement`, the rasterizer's scale and origin; `Tolerance(settings)` in pixels; `Trace` to a
  `GlyphOutline` in font units, y up, one winding, holes kept, `Empty` for no ink) and `IGlyphSource`
  (`GlyphSources.cs`: a `GlyphSourceRequest` of an image, its known text and the settings; `FoundGlyphs` of
  `LabelledGlyph`s, each its text, its form as a string or none, its ink, box, region, baseline and size, checked as
  made, with `Placement(unitsPerEm)` for the tracer; and `GlyphLeftOut`, a character asked for with no glyph and why).
  Both read their options once into settings (`OutlineTracerSettings`, `GlyphSourceSettings`), as pens and poses do.
- **The registries** `OutlineTracers` and `GlyphSources`: guarded `KindTable`s ignoring case (`Register`, `Unregister`,
  `Names`, `Find`, `Get` "not registered" naming the registered ones and "add it with ... Register, or reference a
  package that registers it", `Default`, `Origin`, `SetPolicy`, `Describe`, `AllKeys`); under FallBack a failed call is
  answered by the library's, under Shadow the library's answers and the app's is compared (outlines by their boxes within
  both tolerances, glyphs by labels, forms and boxes). Defaults through `LibraryDefaults.Ensure`; none is built in yet
  ("contour" F1, "sheet" F2, "cells" F4).
- **The face made from glyphs:** `MadeFontFace` and `MadeFontFaceBuilder` in Idrak.Imaging (`MadeFontFace.cs`, beside
  the fonts and the writers, proposal 3): characters, outlines, advances, metrics and the names given (a family at least;
  nothing made up), glyphs with no character for forms and alternates (reached by id), several characters on one glyph,
  glyph 0 given or empty at half an em; its one table `name` (Windows, Unicode, English records of the names given); no
  layout tables (F3's font map gives them). The builder checks every input (units per em 16 to 16,384, the ascender and
  descender, advances and coordinates within 16 bits, contours started by a move, characters once, glyphs added before
  they are mapped) and copies the outlines; the face is read-only, safe from several threads. **Not tested here:** its
  export to `.ttf` through plan 28's `FontWriters` "truetype" (D8, being built now); F3 tests it once that writer is
  merged, with the read back through `FontFormats`.
- **The kit:** `OutlineTracerSuite` (the suite's own shapes, squares, rings with one and two holes, cubic and
  quadratic discs, a dot over a stem, a slanted stroke, a thin bar, random stars with and without a hole, and outlines
  given, filled through `OutlineRasterizers` at several sizes and sub-pixel places: the tolerance finite, no ink to an
  empty outline, filled again within the tolerance both ways, one winding at every pixel centre, holes kept, the same
  outline again and from several threads, an unknown option refused) and `GlyphSourceSuite` (samples of the source's
  kind, as given, padded with paper and blank: glyphs on the image with ink, inside their cell or ink piece, labelled
  once, the known text's characters accounted for or reported, the known glyphs found where they are, none on a blank
  page, the same again and from several threads, an unknown option refused); `Conformance.CheckOutlineTracer` and
  `CheckGlyphSource`. Each suite fails a broken implementation: a tracer that fills holes, traces twice, inks empty
  coverage, leaves y down or takes any option; a source that labels a glyph twice, puts it outside its region, drops a
  character unreported, answers differently each time or takes any option.
- **The plug-in path** (`tests/Idrak.PluginTests/HandwritingPlugin.cs`, public API only): a toy tracer (each row's
  runs of ink as rectangles of whole pixels) and a toy source (ink between blank columns, labelled from the text)
  register, are found by name, find and trace a page's glyphs, pass both suites and unregister; over stand-in library
  ones (`outside-proof`) the app's failure reaches the caller, falls back under FallBack, is compared under Shadow.
- **Tests:** groups `outline tracers` (3), `glyph sources` (3) and `made fonts` (5), generic text and shapes only; the
  kind table's refusal list names both registries. One build (clean, no warning), then one run on the CPU with
  `outline tracers|glyph sources|made fonts|outside plug-in|kind table:|override loop:|public API|abstraction inventory|
  test runner` and the update flags set: 61 tests, 61 passed, none failed. The API dumps (`Idrak.Abstraction`,
  `Idrak.Abstraction.Testing`, `Idrak.Imaging`) and the inventory regenerated through the runner.
- **Waits for:** plan 28's D8 (the TrueType writer) for the made face's export and read back (F3); F1's "contour"
  tracer, F2's "sheet" and F4's "cells" sources, each checked by these suites when it lands.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "outline tracers|glyph sources|made fonts|outside plug-in|kind table:|override loop:|public API|abstraction inventory|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-p38f0.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

### F1: the tracer ("contour")

Threshold the coverage (setting; `Foreground.Otsu` as the measured default), find each contour's boundary (one pass,
marching squares), keep holes as opposite contours, simplify (Douglas-Peucker, tolerance a setting), fit quadratic
curves where the boundary bends smoothly, corners where it turns sharply (angle a setting), scale to font units. Tested
on outlines from repo fonts: rasterize ("coverage"), trace, rasterize again, compare pixels.

**As built (2026-11-18):**
- **The tracer** "contour" (`ContourTracer`, `src/Idrak.Imaging/ContourTracer.cs`, internal; `OutlineTracers.Contour`),
  a library default registered by Idrak.Imaging (one line in `LibraryRegistrations`), pure C#. Its options, read once:
  `trace-threshold` (the ink level: `auto`, Otsu's level of the coverage kept from 0.1 to 0.9, the default; or 0.01 to
  0.99), `trace-simplify` (the simplification tolerance, pixels: 0.5, from 0 to 8), `trace-corner` (degrees: 60, from 0
  to 180), `trace-smoothing` (the length, pixels, over which the boundary's direction is measured either side of a point,
  for its corners and its curves' tangents: 1.5, from 0.5 to 16); a key it does not take or a value out of range is
  refused naming it. The boundary in one pass over the rows (marching squares between pixel centres, two rows of
  coverage at a time, the row above's crossings carried along; each crossing placed between its two pixels by their
  coverage; a cell whose opposite corners are ink joined when its middle is); each contour walked once with the ink on
  its right, so outer contours turn clockwise in font units and holes the other way (one winding). Corners: a turn of at
  least the corner angle within the smoothing length, the sharpest there. Between corners, Douglas-Peucker's splits: a
  piece is a line when its chord holds every boundary point within the tolerance, else a quadratic curve whose control
  point is where the tangents at its ends meet (curves meet smoothly away from corners), held within the tolerance both
  ways, else split at its farthest point; a piece is kept only when no pixel centre beside it changes side. A last check
  of every pixel centre the outline can reach (its winding row by row, curves crossed exactly) splits any piece that
  misplaces one: filled, the outline holds exactly the pixel centres whose coverage reaches the level. Placed in font
  units through the `OutlinePlacement`. Every buffer rented and returned; no state, so glyphs trace from several threads.
- **The tolerance it promises:** the simplification plus the most the level's edge strays from a straight edge's
  half-ink line, `0.1 + 1.5 d + 2 d²` (d the level's distance from 0.5; `auto` at its farthest, 0.4), bounding what was
  measured over every angle on exact coverage (0.086 px at 0.5, 0.24 at 0.4 and 0.6, 0.61 at 0.2 and 0.8, 0.87 at 0.1
  and 0.9, 1.30 at 0.01 and 0.99): 1.52 px by default, 0.60 at the level 0.5. Sharp tips and strokes thinner than a pixel
  move further with a level away from the half. Met: drawn again, the glyphs below differ by 0.08 to 0.16 pixels of ink
  per pixel of edge.
- **The threshold moved:** Otsu's method from Idrak.Vision's `Foreground` to `ImageThresholds.Otsu`
  (`src/Idrak.Abstraction/Data/ImageThresholds.cs`, with `OtsuLeast`, 0.1), the lowest place Idrak.Imaging and
  Idrak.Vision both reach; `Foreground.Otsu` calls it, the code unchanged; proven bit for bit against the method written
  out as loops, and by Vision's groups that use it (foreground, framing, region classifier, line segmentation).
- **The registry:** its docs name "contour"; a name nobody registered is refused with "(Idrak.Imaging registers
  contour)"; `IOutlineTracer` and `OutlineTracers` are on `tests/Idrak.Tests/data/shared-contracts.txt` (decision 10:
  used by Idrak.Imaging alone today, kept in Idrak.Abstraction for the glyph sources of F2 and F4 and the command line of
  F6). The API dump (`Idrak.Abstraction`) and the inventory regenerated through the runner.
- **Tests,** in the group `outline tracers` (five new): the time and allocation per glyph and the points before and after
  simplification, measured; the options, their refusals and the promise; the tracer suite with its defaults and three
  other settings on the suite's shapes and the glyphs of DejaVu, Amiri, Reem Kufi and Cairo; the round trip of those
  glyphs at 24, 48 and 96 px per em under three settings (848,638 pixel centres each inside exactly when its coverage
  reaches the level, 746 on the edge itself left out; one winding; holes kept turning the other way; the coverage drawn
  again within the promise along the outline's length; fewer points than the traced boundary); Otsu's results unchanged.
  One build (clean, no warning), one run on the CPU with `outline tracers|made fonts|kind table:|conformance kit:|public
  API|abstraction inventory|outside plug-in: an outline tracer|override loop:|vision: foreground|vision:
  ContentFrame|vision: RegionClassifier|generic components: line segmentation` and the update flags: 45 tests, 44 passed,
  1 failed (decision 10, the contracts used by Idrak.Imaging alone; fixed by the shared-contracts lines, a data file);
  the same filter again (the inventory flag set): 45 passed, none failed.
- **Measured** (this 4-core x64 container, 96 px per em, 20 glyphs of DejaVu and Amiri: a sheet cell of about 120 pixels):
  the first trace of the process 14.8 ms and 24,000 bytes (compiling); warm 367 µs and 852 bytes a glyph (the outline's
  own arrays 660 bytes; the working buffers are rented and returned); 378 points per outline as traced, 63 simplified
  (6.0x fewer). Over the four fonts: 88 to 28 points a glyph at 24 px per em, 172 to 41 at 48, 332 to 60 at 96.
- **Decided here (the plan did not say):** the keys carry `trace-` (a glyph source's own options sit beside them);
  `auto` keeps Otsu's level from 0.1 to 0.9 (a level near 0 or 1 strays far, and 1 cuts float coverage apart); the
  smoothing setting measures directions and moves no point, so the promise holds; specks are kept (F2 cleans them).
- **Left:** the Idrak.Imaging README's fonts row does not name the tracer yet (it is the row the TrueType writer's piece
  edits; one line for the merge). The warm time was not profiled further; F2 traces a sheet's cells in parallel (rule 79).
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "outline tracers|made fonts|kind table:|conformance kit:|public API|abstraction inventory|outside plug-in: an outline tracer|override loop:|vision: foreground|vision: ContentFrame|vision: RegionClassifier|generic components: line segmentation"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-p38f1.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

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

**As built (2026-11-18):**
- **The contract** (`src/Idrak.Abstraction/Data/FontMakers.cs`, proposal 3's placement): `IFontMaker : IConfigurableKind`
  (`Read` its options once into `FontMakerSettings`; `Make(FontMakerRequest)` to a `MadeFont`) and the guarded registry
  `FontMakers` (Register, Unregister, Names, Find, Get "not registered ... (Idrak.Imaging registers traced)", Default,
  Origin, SetPolicy, Describe, AllKeys; under FallBack a failed make is the library's, under Shadow the two are compared by
  their metrics and each labelled glyph's advance). The request: the glyph source's `FoundGlyphs`, the family (and style,
  more names) the caller gives, the tracer by name (`contour` unless set) with its options, the settings. The result:
  the face, the `FontMap` it made and applied (as data, for F6 to show or keep), the `MadeFontMetrics` with the measure
  stated in words, each `MadeGlyph` (its id, text, form as labelled and which fill), and what was left out with why (the
  source's left-out passed on). **Why this shape:** it is the tracer's and the source's (a configurable kind, settings read
  once, one request record, so a field added later breaks no maker), and it makes a face, not a file: the export is
  already a contract (`FontWriters`, plan 28 D8), so a maker never chooses a format; the labels and the map come back so a
  caller (the suite, F6's progress, an app) knows which glyph is which fill without reading the GSUB.
- **The library's maker** "traced" (`TracedFontMaker`, `src/Idrak.Imaging/TracedFontMaker.cs`, internal; one line in
  `LibraryRegistrations`). Each glyph traced at its baseline and size (`LabelledGlyph.Placement`) and its ink measured as the
  font maps measure it (`InkMeasure`, 256 px per em), on every core (`Parallel.For`; `threads`). Metrics from the ink of the
  glyphs that are not marks (`TracedFontMaker.Heights`): the ascender and descender hold `ink-share` (0.95) of the tops and
  bottoms, the rest outliers; the x-height and cap height are the median tops of the short and the tall glyphs among those
  at least half the median ink height (a dot, a comma or a dash is neither), split by Otsu's level of their tops; the pen is
  the strokes' median thickness. Each glyph placed with `side-bearing` (auto: half the pen) either side of its ink, none for
  `joined-forms` (init:medi:fina) and marks, its advance the bearing past its ink; every outline kept as TrueType gives it
  back (`GlyphEncoder.Whole`, decision 25's rule), so the exported font is the face. The face (`MadeFontFaceBuilder`, now
  with `Heights`: an OS/2 table of version 4 holding the measured x-height and cap height) holds each character's first
  fill under its code and every other glyph by id, and a space when none was written (no ink is found on a sheet; `space`:
  auto, half the plain characters' median advance; U+00A0 shares it). The font map is written as JSON and parsed
  (`FontMap.Parse`): every plain character a letter with its positional forms, the marks placed above or below from where
  their ink sat against half the x-height, the ligatures (a text of several characters) under `ligature-feature` (liga)
  with their components' forms (a ligature's form names the whole's: its first letter init or medi, its last fina or medi;
  or each component's form between '+'; no form: plain letters), the alternates, the anchors (`mark-gap`, `mark-window`)
  and `measure`; the script tags are `scripts` (DFLT, the default script every shaper falls back to: no script named in
  code). `FontMaps.Apply(face, map)` makes the GSUB, GDEF and GPOS through "opentype", anchors the marks from the ink and
  draws the digits and marks the glyphs lack, sized from the written glyphs, as for legacy fonts. A form that is not a
  positional form, a mark with a form, a form without its plain glyph, a ligature whose letters lack their forms, and (with
  `alternates=off`) a further fill are reported left out with why.
- **Alternates:** the font map has a new key, `alternates` ({"#12": ["#40"]}, `FontMapAlternates`), which the mapper
  writes as one alternate substitution (GSUB type 3) listed under `aalt`, `salt` and a private `ALTS`
  (`FontMap.AlternatesFeature`): other software reaches the fills through `salt` and `aalt`; a mark's alternates are marks
  placed as it is, a ligature's ligatures, every alternate a base for marks. **The pick inside Idrak** is the shaper's: a
  feature given the value `random` ("ALTS=random") picks per glyph, itself or one of its alternates, from the run's seed
  (`TextRun.Seed`, new), the glyph and its cluster (SplitMix64, the same on every machine); `TextStyle.Seed` is mixed with
  each paragraph's place into its runs' seeds. A page asks for `ALTS` among the hand's features (`PageDrawing.Hand`, at the
  chance `FontMapDrawing.Alternates`: 1 by default, `IDRAK_FONT_MAP_ALTERNATES`, `--font-maps alternates=`), only where the
  font's own GSUB lists it, picks from its own part of the page's seed and says ", alternates" in its description; the
  glyph poses still vary each glyph on top. The converted-font cache leaves the chance out of its key, as it does the pairs'.
- **Results proven (rule 71):** no repository font has `ALTS` (DejaVu's `salt`/`aalt` and Cairo's `aalt` are stylistic
  alternates, which is why Idrak picks under `ALTS` and not `salt`: picking `salt` at random would change DejaVu's pages),
  so every existing page, shaping and converted font is unchanged; a map without alternates writes the same GSUB bytes (no
  lookup added; a lookup lists under each tag of its feature string, one tag as before); the cache's keys are as before.
  The groups that compare pages, shaping and written fonts passed unchanged (below).
- **Tests:** groups `made fonts` (5 new beside F0's 5) and `font makers` (3), sheets simulated without a human: glyphs of
  DejaVu (letters, punctuation, fi, fl, ffi, second fills of a and e) and of Cairo (17 letters in the forms the test's data
  gives, found through joining contexts; lam-alef alone and after a letter; a fatha and a kasra; second fills of an isolated
  beh and a medial meem) drawn into cells at 96 px per em, kept at 8-bit levels, labelled and made with the defaults.
  Checked: the generic lines (an ordinary letter, a report, a sales report, a team meeting) shape like the source fonts
  glyph for glyph by role (forms and ligatures); the ascender, descender and x-height within 0.02 em of the same measure on
  DejaVu's own outlines (made 0.762, -0.208, 0.560 em; DejaVu's 0.760, -0.209, 0.560; its x 0.547); every advance its ink
  and the bearing (half the pen, 49 units), none for joined forms; a fatha and a kasra their gap (60 units) from their
  base's ink; the digits (Latin, Arabic-Indic) and the marks the sheet lacks drawn, the written marks not; the alternates in
  the GSUB under all three tags, `+salt` taking the second fills, `ALTS=random` picking both fills from the seed (the same
  seed the same, other seeds others), after the joining forms too; a page in the made font picking them and saying so, the
  same seed the same pixels, none at a chance of 0, none in DejaVu; both faces written as TrueType and read back through
  `FontFormats` with every point, advance, character (U+0000-U+FFFF), GSUB, GDEF and GPOS the same and the lines shaped the
  same (with picks too), a fonts folder reading it as any font; the maker's options and refusals; `FontMakerSuite`
  (`Conformance.CheckFontMaker`) passing on "traced" with DejaVu's sheet (defaults, and 2,048 units, a set bearing,
  alternates off, contour at level 0.5) and Cairo's, and failing a maker that drops a character, leaves y down, narrows
  its advances, answers differently each time, drops a fill unreported, or takes any option. One build (clean, no warning),
  one run on the CPU with `made fonts|font makers|outline tracers|glyph sources|fonts write|fonts:|shaping:|pages:|plain:|
  composed pages:|shapes:|glyph poses:|synthetic data:|kind table:|conformance kit:|override loop:|outside plug-in|public
  API|abstraction inventory|every environment variable read in src|the environment table names real commands|no command,
  option or help text names a provider|test runner` and the update flags (the D8 samples made first with
  `make_fixtures.py --only d8`): 146 tests, 145 passed, 1 failed: the suite's "each made glyph sits where its ink sat" on
  the 0.5-level sheet, measured against the ink's box, which held the rasterizer's faint running-sum residue right of a
  glyph; fixed in the suite (the box of the pixels whose coverage reaches half, where a tracer's edge is promised) and in
  the test's sheet (8-bit levels, as a scan). Rebuilt, `made fonts|font makers`: 12 passed, 1 failed (the same check on 'G':
  a curve's control point below its ink; fixed: the outline's true extent, curves' turning points). Rebuilt, rerun: 13
  passed, 0 failed. The API dumps (`Idrak.Abstraction`, `Idrak.Abstraction.Testing`, `Idrak.Imaging`) and the inventory
  regenerated through the runner; `IFontMaker` and `FontMakers` on the shared-contracts list (decision 10: used by
  Idrak.Imaging alone today, for F6's command line and an app's maker next).
- **The checker's verdicts** (`tools/documents/check_font.py` on `test-output/fonts/made/latin-hand.ttf` and
  `joined-hand.ttf`): fontTools opens both and decompiles every table, the OpenType Sanitizer accepts both with no message,
  HarfBuzz shapes the seven lines; and HarfBuzz (uharfbuzz) shapes the three generic Arabic lines with `joined-hand.ttf`
  like Cairo by role (21, 25 and 3 glyphs, lam-alef joined), `salt` giving a beh its second fill.
- **Measured** (this 4-core container while three other agents built: load average 6.5 to 8.5; the owner's machine gives
  the real figures): 132 glyphs at 96 px per em (DejaVu's sheet 67, Cairo's 65). The first make of the process (DejaVu's
  sheet, cold) 154 ms and its completion 107 ms (the map's tables, anchors and drawn glyphs, made on first use), 6.2 MB.
  Warm, both sheets: made in 41 ms on 4 threads (307 µs a glyph) and 45 ms on one (344 µs, 1.12x: little left to spread
  under that load), the same faces both ways; completed in 73 ms (the mapper's serial work); 42 KB managed allocation a
  glyph.
- **Decided here (the plan did not say):** (1) the random pick is under a private `ALTS` besides `salt` and `aalt` (above),
  and not `rand`, which HarfBuzz applies before the joining forms (a picked isolated glyph would never take its forms);
  (2) the pick is in the shaper, before positioning, so the picked fill's advance and anchors are used; (3) a ligature's
  form names the whole's, or each component's between '+'; (4) joined forms and marks take no side bearing; (5) a space is
  added when none was written; (6) scripts default to DFLT; (7) the x-height leaves out glyphs under half the median ink
  height; (8) the measured heights go into the face's OS/2; (9) `MadeGlyph.Form` keeps the label as given (a single
  character's isol is its plain glyph); (10) only the four positional forms a font map shapes are made, others reported.
- **Left:** F6 (`idrak data font-from`: `FontMakers` then `FontWriters.Save`); the face's completion (the mapper measures
  every base's ink again for its anchors, serially, after the maker measured it once) is the larger part of a warm make: a
  candidate to share or spread when the owner's bench shows it matters; the parallel trace measured again on a quiet machine
  (`idrak bench`); the Idrak.Imaging README's fonts row (shared with the font-writer pieces) not edited; `.woff` export when
  the plug-in lands; F2's and F4's sources give real ink to the same maker.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples --only d8
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "made fonts|font makers|outline tracers|glyph sources|fonts write|fonts:|shaping:|pages:|plain:|composed pages:|shapes:|glyph poses:|synthetic data:|kind table:|conformance kit:|override loop:|outside plug-in|public API|abstraction inventory|every environment variable read in src|the environment table names real commands|no command, option or help text names a provider|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-p38f3.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  python -I tools\documents\check_font.py test-output\fonts\made\latin-hand.ttf test-output\fonts\made\joined-hand.ttf -o test-output\fonts\check-made.json
  ```

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
