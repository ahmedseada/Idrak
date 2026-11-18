# Plan 28: documents (a document contract, PDF read and written built in, vector and image pages, Office as a plug-in later)

**In progress (on the branch `documents`).** Built 2026-11-18: D0 (the contract), D1 (the file), and D5's sample
generator with every phase's samples (D1-D13) and the checkers. Paused after D1, resumed the same day by the owner
("resume"): running D3 (text) and D8's built-in part (the TrueType writer, subsetting, the converted-font cache); D8's
plug-in part (WOFF alone in `Idrak.Imaging.Plugins`, decisions 20 and 21) follows once the built-in part is merged.

**Status (2026-11-18, `abstraction`):** planned 2026-10-22 against the code at 99872b8; widened 2026-11-18 against the
code at da19bfc by the owner's decisions 7-9 (PDF read **and written**, vector **and** image pages, finished for good
before plan 38) and 10-15 (forms, annotations, signatures and editing in; fonts written in the common formats; the test
samples in the ignored `test-output/`; ready for any workload, plan 38 waits; worked on the branch `documents`, taken
from `migration`) and 16-19 (only the app's certificates trusted; `idrak data redact`; TrueType built in and used by
Idrak, no JSON font file, WOFF alone in a plug-in (decision 20); two agents at most, a fresh one per piece). The PDF parts of
[plan 38](38-handwriting-fonts.md) moved here (see "Plan 38's needs"). The owner started the work 2026-11-18 ("start
2 agents at max in parallel"); the owner tests later, so the command lists are held.

| Phase | What | Lane | State |
|---|---|---|---|
| D0 | The contract and its registries (`IDocumentFormat`, `DocumentFormats`, `IDocumentTextLayout`, `DocumentTextLayouts`, `IDocumentWriter`, `DocumentWriters`), their suites, a test format from outside proving the plug-in path | A | Built (2026-11-18) |
| D1 | PDF, the file: objects, cross-reference tables and streams, object streams, incremental updates, repair, filters, limits | A | Built (2026-11-18) |
| D2 | PDF encryption: the standard security handler (RC4, AES-128, AES-256), an empty or a given open password | B | Planned |
| D3 | PDF text: pages, the content-stream interpreter, fonts to Unicode, glyph widths, text runs with boxes, the `lines` layout | A | Planned |
| D4 | PDF images and embedded files (a scanned PDF's pages as images) | B | Planned |
| D5 | Fixtures and their generator, READMEs, the owner's runs on real files | A | In progress (the generator, every phase's samples and the checkers built; the READMEs and the owner's runs left) |
| D6 | PDF writing: pages of images, vector paths and text (fonts embedded, searchable), several pages, the same bytes from the same input | B | Planned |
| D7 | PDF vector content read, and any page drawn to pixels (paths, text through the embedded fonts, images, clipping) | A | Planned |
| D8 | Fonts written: TrueType (`.ttf`) built in, with subsetting and legacy fonts converted once (D6 embeds through it; plan 38 exports through it); WOFF (read and write) from a plug-in, the one proof of the plug-in path | B | Planned |
| D9 | Forms: fields read (text, check, radio, choice, push button, signature), filled, created, flattened; appearances generated | A | Planned |
| D10 | Annotations and navigation: every annotation kind read, drawn, created and removed; links, bookmarks (outlines), named destinations, page labels; tagged structure read for reading order | B | Planned |
| D11 | Editing in place and saving: incremental updates, a full rewrite (unused objects dropped), pages merged, split, inserted, removed, moved, rotated; saved encrypted (AES-256) | A | Planned |
| D12 | Signatures: read and validated (CMS, the byte range, the chain against trust the app gives, timestamps, revocation from the file), and signing (PAdES baseline) | B | Planned |
| D13 | Any workload: the documents bench (open, text, draw, write, edit; cold and warm), large and many files, memory bounded, the limits measured | A | Planned |

## Goal

The library reads **and writes** documents through one contract, the way it reads and writes images through
`ImageCodecs` and `ImageEncoders`: a format registered by name, chosen from a file's first bytes, opened page by page,
giving text with its place on the page, vector paths, images, embedded files, form fields, annotations, links,
bookmarks and signatures, and any page drawn to pixels; a writer that makes pages of images, vector paths and text, with
fields, annotations, bookmarks and signatures; and an editor that changes a file in place (a filled form, a new
annotation, a signature, pages merged or split) without losing what was signed before. Fonts are written in the common
formats too (TrueType built in, WOFF from one plug-in), which the writer embeds from and plan 38 exports to. PDF is the library's
default, both ways, for both kinds of
PDF: image PDFs (most scans are a page image, sometimes with a hidden text layer) and vector PDFs (text and drawings,
as made by software or written on a tablet). Office (Word, Excel, PowerPoint) comes later as a plug-in registering into
the same contract, with nothing in the library changed for it. Apps (search and question answering over attachments, a
meeting system's minutes, a searchable PDF made from an OCR'd scan, plan 38's glyph sheet) read and write any
registered format the same way.

## Decisions (taken by the owner, 2026-10-22)

1. **PDF now, as a library default:** the contract and its registry in `Idrak.Abstraction.Data`, the PDF reader in
   `Idrak.Data`, registered through `LibraryDefaults`, as PNG and JPEG are by `Idrak.Imaging`.
2. **Office later, as any plug-in:** it implements the same contract and registers into `DocumentFormats`; this plan
   builds none of it, only a contract it can implement without a change.
3. **Nothing language-specific in the library.** Text order comes from positions on the page and from Unicode's own rules
   through the existing `ReadingOrders` registry (`bidi`, UAX #9), never from code for a language or a script. Tables
   tied to one script (the CJK predefined CMaps, for example) are not built in.
4. **Plan first:** this file is written before any code.

**Taken by the owner (2026-11-18):**

7. **PDF read and written, built in.** The library writes PDF files as well as reading them, with no outside program or
   package; reading and writing share one PDF layer (objects, filters, cross-reference) in `Idrak.Data`.
8. **Vector and image pages, both ways.** Most scanned PDFs are image PDFs (a page image, often with a hidden text
   layer), so images are read and written; vector PDFs (text, lines, curves, fills) are read and written too. A page of
   either kind can be drawn to pixels (D7), so OCR and plan 38 read any PDF the same way.
9. **Finished for good, before plan 38.** Every phase here is built and closed before plan 38 starts, so plan 38 never
   waits on PDF.

**Taken by the owner (2026-11-18, second round):**

10. **Forms, annotations, signatures and editing in place are in this plan** (open question 1 answered "in"): D9-D12.
11. **The writer's image pages default to `Auto`** (open question 3): JPEG for photographs and scans, Flate for line art
    and few levels, measured per image; `Flate` or `Jpeg` fixed by the option.
12. **Fonts are written in the common formats** (narrowed by decision 20 to TrueType built in and WOFF from one plug-in): TrueType (`.ttf`), OpenType with CFF outlines (`.otf`), WOFF and WOFF2
    (and read in all four): D8. The writer embeds font subsets through it, and plan 38 exports a made font through it
    (no Idrak-only font file).
13. **The test samples live in the ignored folder.** The reference reader and renderer from PyPI (open question 2,
    agreed) and the fixture generator write every sample and every expected result under `test-output/` (git ignores
    it), never into the repository: see D5.
14. **Ready for any workload; plan 38 waits.** Size, count and speed are part of "finished": D13 measures and bounds them.
    Plan 38 starts only when every phase here is closed, however long that takes.
15. **The branch:** this plan is worked on the branch `documents`, taken from `migration` (not `main`), until the owner
    says otherwise.

**Taken by the owner (2026-11-18, third round):**

16. **Signatures trust only what the app gives** (open question 4): the certificates passed in `SignatureCheckOptions`;
    the machine's store only when the app asks for it by that option.
17. **Redaction is also a command** (open question 5): `idrak data redact` (D10), beside the library call.
18. **Fonts: TrueType built in, and Idrak uses it; no Idrak-only font file.** The library (`Idrak.Imaging`) reads and
    writes `.ttf` (the fastest of the formats to open and draw from, by estimate; D8's bench measures it). There is no
    JSON font format: TrueType holds everything a made font needs (outlines, metrics, characters, layout tables,
    alternates, names), and a second format would only add upkeep and the `.json` clash with font maps; a person reads a
    font through the reference font tools' text dump (`tools/`). Whatever a font came in as, Idrak works from TrueType
    (the PDF writer's embedded subsets, plan 38's made fonts).
20. **One plug-in font format, only to prove the path** (owner, 2026-11-18: "do one in the plugins for proving
    abstraction works but not all of them"): **WOFF** (read and write) in a plug-in project, registering into
    `FontWriters` and `FontFormats` from outside the library. OpenType CFF writing and WOFF2 are not built (dropped from
    this plan; a later plug-in can add them through the same registries).
21. **The font plug-in project is `Idrak.Imaging.Plugins`** (open question 6): new, beside the other `*.Plugins`
    projects, namespace `Idrak.Imaging.Plugins.Fonts`; it references only the public API.
22. **Every merge is built before it is pushed** (owner, 2026-11-18: "you need to build before push even if merged
    cleanly there still could be errors"): after each cherry-pick onto `documents`, one build of the merged code
    (`dotnet build -c Release tests/Idrak.Tests -nodeReuse:false`, and `tests/Idrak.PluginTests` when the piece touched
    it); a failed build is fixed before the push. A merge with conflicts also runs the test groups the merge touched.
    **Refined the same day** (owner: "if the agent already built and none other projects changed … no rebuild
    needed"): the build is skipped when the merged code is the code the agent built — `git diff <agent's commit> HEAD`
    outside `plans/` is empty, only plan files changed since its start — and done whenever another piece's code landed
    after the agent started.
    Up to four agents may run when pieces with no unmerged dependency are ready; a slot with none waits.
23. **Slot 4 is plan 38's, this session only** (owner, 2026-11-18: "for this session slot 4 takes what could be
    finished from phase 38", then "keep the fourth always for plan 38 f0, f1, f3 the other three works on plan 28"):
    three slots work on plan 28 only; the fourth works on plan 38's phases that need nothing unmerged — F0, then F1,
    then F3 — and on nothing else. When F3 is merged the fourth slot stops for good: it takes no piece from either plan
    ("when those 38 phases finish stop agent 4 and don't give it any pieces from either plans"). Decisions 9 and 14
    hold otherwise.
19. **How the work runs:** at most two agents at a time; each piece goes to a fresh agent, started the moment a slot
    frees with the next piece whose inputs are merged; the owner tests later (the command lists are held and given as
    one list when asked).

**Proposed, for the owner to confirm:**

5. ~~Scope of the first version: text and images, not rendering.~~ Superseded by decisions 7-14: reading covers text,
   images, vector paths, forms, annotations, navigation and signatures, and drawing a page to pixels; writing covers
   images, paths, text, fields, annotations, bookmarks and signatures; editing covers changes in place and page
   operations.
6. **The contract, the PDF reader, writer and editor, and the font writers** ("just implement the contract"): one CLI
   command (`idrak data redact`, decision 17), no dataset rows, no retrieval chunking from documents here, except what
   plan 38 needs (its commands are plan 38's F6). Other users of the contract are listed under "Not in this plan".

   Proposed with decisions 7-9:

   - **A. The writing side is its own contract**, `IDocumentWriter` / `DocumentWriters` (C11-C15), beside the reading
     one, as `ImageEncoders` stands beside `ImageCodecs`: a format may read without writing (an Office plug-in) or write
     without reading.
   - **B. One page description both ways.** The writer takes, and the reader gives back, the same types:
     `DocumentPath` (a path of moves, lines, quadratic and cubic curves, its fill rule, stroke, dash and colour),
     `DocumentImage`, `DocumentTextRun`. So every written page is checked by reading it back exactly.
   - **C. Text is written as glyphs, not characters.** The caller shapes (`TextShapers`, as pages are drawn today) and
     gives glyph ids with their positions and the text they stand for; the writer embeds the font and writes a
     `ToUnicode` map (and `/ActualText` where one glyph stands for several characters, as in a ligature), so written text
     is searchable and copies out in logical order, in every script, with no script named in the writer.
   - **D. Drawing to pixels reuses the library's own pieces:** `FontFormats` (TrueType and CFF outlines),
     `OutlineRasterizers` ("coverage"), `Pens` (a stroke to its outline), `ImageCodecs` for the images, nothing copied.
   - **E. Drawing a page is its own contract**, `IDocumentPageDrawer` / `DocumentPageDrawers` (C16), format-neutral: it
     draws any `DocumentPage` (its paths, glyph runs and images) to an `ImageData`. The library's drawer, "coverage", is in
     `Idrak.Imaging`, beside the rasterizer and the fonts (`Idrak.Data` references only `Idrak`); a PDF page, an Office
     plug-in's page or a page an app built are drawn the same way.

## Rules that hold for every phase

CLAUDE.md and [docs/optimization.md](../../docs/optimization.md) apply in full. The ones that shape this plan:

- **Contract plus registry (plan 10's decision 10).** `DocumentFormats`, `DocumentTextLayouts`, `DocumentWriters`,
  `DocumentPageDrawers` and `FontWriters` are `SlotTable`s with
  `Register`, `Unregister`, `Names`, `Get` ("not registered", naming the registered ones), `Default`, `Origin`,
  `SetPolicy`, guarded as `ImageCodecs` is (an app's format over the library's falls back or is compared in shadow); their
  library defaults come through `LibraryDefaults.Ensure`; the inventory and the API dump are regenerated.
- **No platform dependence (rule 70).** Pure C# and the base library: zlib through `System.IO.Compression`, AES, MD5 and
  SHA-2 through `System.Security.Cryptography`, RC4 written here (the base library has none). No outside program, no
  platform's PDF engine, no NuGet package (rule 0).
- **Parse bytes, not strings (rule 74).** The lexer and parser work on `ReadOnlySpan<byte>`; names and keys are compared
  as bytes (`"Type"u8`); strings are made once, at the edge (a run's text, a metadata value).
- **Memory first.** A document is opened, not loaded: objects are read on demand from a seekable stream through
  `RandomAccess`, a page is read when asked, decoded streams go to pooled buffers returned in `finally` (rules 10, 44),
  and a bounded cache keeps the shared ones (fonts, CMaps) decoded once (rule 75). Opening a 500-page scan reads the
  cross-reference and the page tree only.
- **Untrusted input.** Every PDF is treated as hostile: limits on a stream's decoded size, on objects, on nesting (forms
  within forms, arrays within arrays), cycles detected; no JavaScript run, no outside reference fetched, no file written
  by the reader.
  Each limit is an option (rule "nothing fixed"); passing one is a clear `InvalidDataException` naming the limit and the
  option, never a hang or an out-of-memory.
- **Damaged files are the common case.** A broken cross-reference is rebuilt by scanning the file for objects, and the
  document's information says it was repaired; one bad page or stream fails that page or stream alone.
- **Results proven (rule 71).** Every fixture has its expected text, boxes and images checked exactly; a speed change
  keeps them. Every written file reads back exactly (paths, text, glyphs, raw image pixels), and the same input writes
  the same bytes (no date, no random ID unless given), so a written file is itself a fixture.
- **Writing streams, not builds in memory (rules 10, 44, 77).** A page's content and its images are written to the
  destination as each page is finished; only the cross-reference offsets (a few bytes an object) stay until the end.
  A 500-page image PDF is written with one page's buffers in memory, pooled and returned in `finally`.
- **Spread independent work, then check it paid (rule 79).** An `IDocument` is safe to read from several threads, so
  pages can be read in parallel; measured on few cores too.
- **Tests only through filters,** CPU here (`IDRAK_DEVICES=cpu`); the owner's real documents are the owner's runs.
- **The fix workflow:** fix everything, update the tests, build once, run the phase's filters once with `--no-build`;
  after a merge, build and run before pushing. At most two agents at a time.

## What exists (2026-10-22, 99872b8)

- **No document reading anywhere:** no PDF, Word, Excel or PowerPoint reader in `src/` (searched). `Idrak.Data` reads
  JSON Lines, JSON, CSV, TSV, text, code and Parquet (`DataFileFormats`, `src/Idrak.Data/Abstractions/DataFileFormats.cs`);
  `Idrak.Imaging` reads PNG, JPEG, BMP and Netpbm.
- **The pattern to follow:** `IImageCodec` and `ImageCodecs` (`src/Idrak.Abstraction/Data/ImageCodecs.cs`): a format by
  name, `ReadInfo(header)` deciding from the first 64 KiB, `newestFirst` so an app's format is asked first, a guarded
  wrapper (`GuardedCodec`: shadow comparison, fall back), defaults registered through `LibraryDefaults.Ensure`
  (`src/Idrak.Abstraction/LibraryDefaults.cs`), which already lists `Idrak.Data` and its `LibraryRegistrations`
  (`src/Idrak.Data/LibraryRegistrations.cs`, today only `ModelSources`).
- **Pieces the PDF reader reuses, none copied:**
  - `ImageData` (`src/Idrak.Abstraction/Data/ImageData.cs`): decoded pixels, 8-bit planes; a PDF's raw images decode into it,
    its JPEG images through `ImageCodecs.Get("jpeg")` at run time (no reference to `Idrak.Imaging`).
  - `FontFormats` and `IFontFace` (`src/Idrak.Abstraction/Data/Fonts.cs`; the `opentype` reader in `Idrak.Imaging`): an
    embedded TrueType or OpenType font's `cmap` (`Table("cmap")`) maps its glyphs back to code points when the PDF gives no
    other way, through the registry at run time.
  - `ReadingOrders` (`src/Idrak.Abstraction/Data/ReadingOrders.cs`, default `bidi` over `Idrak.Imaging`'s `Bidi`):
    `FromColumns` turns a line's text in visual (column) order into logical order by Unicode's algorithm.
  - `SlotTable`, `Overrides`, `Comparisons` (shadow runs).
- **Not usable as is:** `PageContent` and `PageStyle` (`src/Idrak.Abstraction/Data/Pages.cs`) describe pages the library
  draws (synthetic pages), not pages it reads.

## What exists for writing and drawing (2026-11-18, da19bfc)

- **Still no PDF anywhere:** no PDF reading or writing in `src/` (searched for "pdf"; only a mention in
  `Idrak.Text/TuningDataReaders.cs` and `Idrak.Imaging/Shaping/Bidi.cs`).
- **The writing pattern to follow:** `IImageEncoder` and `ImageEncoders` (`src/Idrak.Abstraction/Data/ImageEncoders.cs`):
  the writing side of `ImageCodecs`, by name and extension, guarded. Built in: `png` (zlib through `ZLibStream`) and
  `netpbm`. It writes one image, so it cannot hold a document of several pages: hence C11-C15.
- **JPEG:** `JpegEncoder` (`src/Idrak.Imaging/JpegEncoder.cs`, public static) exists but is **not registered** in
  `ImageEncoders` (`src/Idrak.Imaging/LibraryRegistrations.cs:33-34` registers `png` and `netpbm` only). `Idrak.Data`
  references only `Idrak`, so the writer's JPEG pages need a `jpeg` encoder registered (a small piece of D6).
- **Outlines and drawing:** `GlyphOutline` and `OutlineVerb` (move, line, quadratic, cubic, close;
  `src/Idrak.Abstraction/Data/Fonts.cs`); `OutlineRasterizers` "coverage" (`src/Idrak.Imaging/CoverageRasterizer.cs`):
  exact area coverage, a running sum per row, **nonzero only for contours that do not overlap the other way** and **no
  even-odd rule** — PDF paths need both (a gap for D7); `Pens` (`src/Idrak.Abstraction/Data/Pens.cs`, built in `ideal`
  and `ballpoint`): a stroke to its outline, the stroker D7 reuses.
- **Fonts to embed and to draw with:** `OpenTypeFormat` and `CffOutlines` (`src/Idrak.Imaging/`) read TrueType and CFF
  outlines; font maps (`src/Idrak.Abstraction/Data/FontMaps.cs`) give a legacy font Unicode and layout tables, so a mapped
  face is written like any other; `TextShapers` give glyph ids and positions.
- **Pages the library draws, already as vector data:** the shape pipeline (`src/Idrak.Abstraction/Data/Shapes.cs`)
  describes what it draws as `ShapePath` (points, closed, width, dash, tone) and `ShapeArea` (a filled polygon and its
  tone) before `ShapesLayout` rasterizes them, so a page of shapes can be written as PDF paths, not only as an image.
- **Perspective:** `Homography` and `Warp` (`src/Idrak.Imaging/ScanTransforms.cs:262, 320`) are private in the internal
  `ScanImageOps`. D7's drawer (in `Idrak.Imaging`, proposal E) places an image by its matrix with the same resampling,
  so they become a shared internal helper of `Idrak.Imaging` (moved, not copied), which plan 38's sheet straightening
  also uses.
- **The layering this sets:** `Idrak.Data` references only `Idrak`, so the PDF reader and writer reach images, fonts and
  JPEG through the registries at run time (`ImageCodecs`, `ImageEncoders`, `FontFormats`), and drawing a page to pixels
  is its own contract (proposal E) whose library drawer lives in `Idrak.Imaging`, beside the rasterizer and the fonts.

## The contract (`Idrak.Abstraction.Data`, new)

| # | Type | What |
|---|---|---|
| C1 | `IDocumentFormat` | `Name` ("pdf"), `Extensions` (".pdf"), `MediaTypes` ("application/pdf"), `Accepts(ReadOnlySpan<byte> header)`, `Open(Stream seekable, string name, DocumentOptions options) → IDocument` (the stream stays the caller's to dispose after the document) |
| C2 | `IDocument : IDisposable` | `Format`, `Info`, `PageCount`, `ReadPage(int index, DocumentParts parts) → DocumentPage` (`parts`: text, images, both), `Attachments` (embedded files, read on demand); thread-safe |
| C3 | `DocumentInfo` | title, author, subject, keywords, creator, producer, created and modified dates, format version, encrypted, repaired, other entries by key |
| C4 | `DocumentPage` | `Index`, `Width` and `Height` (points, 1/72 inch, as shown after the page's rotation), `Rotation`, `Runs`, `Lines` and `Text` (from the layout), `Images`, `Issues` (what could not be read: glyphs with no code point, an image encoding no codec reads, a damaged stream) |
| C5 | `DocumentTextRun` | `Text`, `Box` (`DocumentBox`: x, y, width, height on the page, top-left origin), `FontSize`, `Font` (name or null), `Hidden` (invisible text, such as a scan's text layer), `Order` (its place in the page's content) |
| C6 | `DocumentImage` | `Box`, `PixelWidth`, `PixelHeight`, `Encoding` (`raw`, `jpeg`, `jpeg2000`, `ccitt`, `jbig2`, as named in the file), `Bytes` (the encoded data, on demand), `Decode() → ImageData` (raw, or through `ImageCodecs` by its encoding; an encoding no registered codec reads is a `NotSupportedException` naming it) |
| C7 | `DocumentAttachment` | `Name`, `MediaType`, `Size`, `Description`, `Read() → Stream` |
| C8 | `DocumentOptions` | `Password` (null: the empty one), `Layout` (a `DocumentTextLayouts` name, default `lines`), `ReadingOrder` (a `ReadingOrders` name, default `bidi`), `IncludeHidden` (default true, marked `Hidden`), `MaxStreamBytes`, `MaxObjects`, `MaxDepth`, `CacheBytes` |
| C9 | `DocumentFormats` (registry) | as `ImageCodecs`: `Register`, `Unregister`, `Names`, `Extensions`, `Get`, `Default`, `Origin`, `SetPolicy`, `CanOpen(path)`, `Detect(header)`, `Open(path, options)`, `Open(Stream, name, options)`, `Open(ReadOnlyMemory<byte>, name, options)`; "not registered" names the formats and the Office plug-in to come |
| C10 | `IDocumentTextLayout` + `DocumentTextLayouts` (registry) | runs → lines and text; `lines` (the library's: runs grouped into lines by baseline and size, each line's direction from its first strong character by Unicode's rule, runs ordered by position in that direction, words split where a gap exceeds a share of the font size, an option; then `ReadingOrders` for logical order), `content` (the runs in content order, joined, no layout) |

Added 2026-11-18 (decisions 7-9, proposals A-E):

| # | Type | What |
|---|---|---|
| C4+ | `DocumentPage` | also `Paths` (the page's vector content, in content order, D7) and, for drawing, each run's `Glyphs` (`DocumentGlyph`: glyph id, position, advance) and `Face` (the run's font as an `IFontFace`, read from the embedded program through `FontFormats`, or null when not embedded); `DocumentParts` gains `Paths` and `Glyphs` (asked for only when wanted: reading text alone stays as cheap as D3 makes it) |
| C6+ | `DocumentImage` | also `Matrix` (how the image is placed: six numbers, for a turned or skewed image) and `Mask` (a stencil or soft mask as grey, when asked) |
| C11 | `DocumentPath` | `Verbs` (`OutlineVerb`: move, line, quadratic, cubic, close) and `Points` in page points, top-left origin; `Fill` (`None`, `NonZero`, `EvenOdd`); `Stroke` (width, or none), `Cap`, `Join`, `MiterLimit`, `Dash` (pattern and phase); `FillColour` and `StrokeColour` (`DocumentColour`: grey, RGB or CMYK as given, with its RGB); `Clip` (the clip this path was drawn under, by index into the page's clips); the same type read (D7) and written (D6) |
| C12 | `IDocumentWriter` | `Name` ("pdf"), `Extensions` (".pdf"), `MediaTypes`, `Create(Stream destination, DocumentWriteOptions options) → IDocumentBuilder` (the stream stays the caller's) |
| C13 | `IDocumentBuilder : IDisposable` | `Info` (title, author, subject, keywords, creator, dates: all optional, none written unless given), `AddPage(width, height) → IDocumentPageBuilder` (points; pages written in order, each flushed when the next starts or on `Finish`), `Attach(name, mediaType, Stream)` (embedded files), `Finish()` (the cross-reference and trailer; a builder disposed without it leaves a file the reader repairs, and says so) |
| C14 | `IDocumentPageBuilder` | `Image(ImageData, DocumentBox, DocumentImageEncoding)` (`Flate`: exact pixels; `Jpeg`: through `ImageEncoders` "jpeg", quality from the options; `Auto`: the options' rule), `Path(DocumentPath)`, `Glyphs(IFontFace face, float size, ReadOnlySpan<DocumentGlyph> glyphs, string text, DocumentTextMode mode)` (mode: `Fill`, `Stroke`, `Hidden`: a scan's searchable text layer), `Save()` / `Restore()` (the graphics state), `Clip(DocumentPath)` |
| C15 | `DocumentWriteOptions` + `DocumentWriters` (registry) | options: `Compression` (Flate level), `ImageEncoding` (`Auto`: JPEG for photographs and scans, Flate for line art and few levels, measured per image; or fixed), `JpegQuality`, `EmbedFonts` (`Subset` default, or `Whole`), `Version`, `Deterministic` (default true: no date or ID unless given), `ObjectStreams` (smaller files, on by default); registry as `ImageEncoders`: `Register`, `Unregister`, `Names`, `Extensions`, `Get`, `Default`, `Origin`, `SetPolicy`, `Create(path, options)`, `Create(Stream, name, options)`; "not registered" names the writers |
| C16 | `IDocumentPageDrawer` + `DocumentPageDrawers` (registry) | `Draw(DocumentPage page, DocumentDrawOptions options) → ImageData`; options: `PixelsPerPoint` (or `Dpi`), `Colour` (grey or RGB), `Background`, `Fonts` (a folder of fonts for runs whose font is not embedded, matched by name, then by the registered fonts' coverage; none: the run is left out and counted in `Issues`), `Parts` (paths, text, images: any of them, so a scan's image can be drawn without its hidden text, or the text alone); the library's drawer, "coverage", in `Idrak.Imaging` |
| C17 | `IDocument.DrawPage(int index, DocumentDrawOptions options) → ImageData` | a shortcut: reads the page with the parts drawing needs and draws it through `DocumentPageDrawers`; a format whose pages cannot be drawn says so (`NotSupportedException` naming it) |

Added 2026-11-18, second round (decisions 10-14):

| # | Type | What |
|---|---|---|
| C18 | `IFontWriter` + `FontWriters` (registry, `Idrak.Abstraction.Data`, beside `FontFormats`) | `Name` ("truetype" built in; "woff" from the plug-in, decisions 18 and 20), `Extensions`, `Write(IFontFace face, Stream destination, FontWriteOptions options)`; options: `Glyphs` (a subset: these glyph ids, composite parts and, when `KeepLayout`, every glyph the layout tables can reach), `KeepLayout`, `Names` (the source face's name records kept, copyright and licence included, with the strings the caller gives over them; nothing invented), `CurveTolerance` (cubic to quadratic for TrueType outlines, in font units), `Hinting` (dropped by default; kept when written whole); registry as `FontFormats`. Built in (`Idrak.Imaging`): "truetype"; the plug-in adds "woff" to both `FontWriters` and `FontFormats` |
| C19 | `DocumentField` | `Name` (full, dotted), `Kind` (text, check box, radio, list, combo, push button, signature), `Value` / `Values`, `Options` (choices, with export values), `Flags` (read-only, required, multi-line, password, comb, ...), `MaxLength`, `Widgets` (page, box, appearance state), `Tooltip`; `IDocument.Fields` (read on demand); XFA forms: their data (`datasets`) read as fields, their layout not drawn (reported in `Issues`) |
| C20 | `DocumentAnnotation` | `Kind` (text note, free text, line, square, circle, polygon, polyline, highlight, underline, squiggly, strike-out, ink, stamp, caret, link, file attachment, popup, redaction, widget, and any other by its name), `Page`, `Box`, `Contents`, `Author`, `Created` / `Modified`, `Colour`, `Opacity`, `Points` (quad points, a line's ends, a polygon's vertices), `Ink` (`DocumentPath`s: strokes written on the page), `Target` (a link's URI or destination), `Replies` (its thread), `Appearance` (drawn by the drawer when asked); `DocumentPage.Annotations` |
| C21 | `DocumentOutline`, `DocumentDestination`, `DocumentPageLabel` | the bookmarks tree (title, destination, open, colour, style, children), destinations (page, a position and zoom, or a name), page labels (`iv`, `A-3`); `IDocument.Outline`, `IDocument.PageLabel(index)`; tagged files: the structure tree's order used by a `tagged` text layout (C10), the reading order the author gave |
| C22 | `DocumentSignature` | `Field`, `Signer` (the certificate), `SignedAt` (claimed, and from a timestamp), `Reason`, `Location`, `ByteRange`, `CoversWholeFile` (or changed after signing, and how: allowed by its permissions or not), `Kind` (CMS, PAdES baseline levels, the legacy PKCS #1 and PKCS #7 SHA-1 forms), `Check(SignatureCheckOptions) → SignatureCheck` (digest, signature, chain against the trust the app gives, timestamp, revocation from the file's DSS, and online only when the app turns it on) |
| C23 | `IDocumentEditor` (from `IDocumentWriter.Edit(IDocument source, Stream destination, DocumentWriteOptions options)`) | `SetField(name, value)`, `AddAnnotation`, `RemoveAnnotation`, `SetInfo`, `AddOutline`, `InsertPages(index, IDocument from, pages)`, `RemovePages`, `MovePage`, `RotatePage`, `AddPage(width, height) → IDocumentPageBuilder` (new content over a page too: a stamp, a watermark), `Sign(field, SigningOptions)`, `Save(DocumentSaveMode)` (`Incremental`: the source's bytes kept, changes appended, earlier signatures stay valid; `Rewrite`: one clean file, unused objects dropped, object streams); a format that cannot edit says so |
| C24 | `IDocumentBuilder` and `IDocumentPageBuilder` widened | `AddField`, `AddAnnotation`, `AddOutline`, `AddLink`, `SetPageLabels`, `CopyPage(IDocument from, int index)` (a page with its resources from another document: merge and split), `Encrypt(DocumentEncryption)` (AES-256, revision 6: user and owner passwords, permissions), `Sign(field, SigningOptions)` at `Finish` |
| C25 | `SigningOptions` | the certificate and its private key (`X509Certificate2`), the chain to embed, the digest (SHA-256 by default), the level (PAdES B-B, B-T with a timestamp authority the app names, B-LT with revocation data embedded), `Reason`, `Location`, the field's appearance (text, an image, or none); nothing fetched unless the app names the authority |

The contract is format-neutral: a Word file has no fixed pages or boxes, so an Office plug-in gives one page per section
(or one for the whole file), runs without boxes (`Box` empty) and `Lines` from its paragraphs. Nothing in C1-C25 names
PDF, and nothing in C18 names one font format.

**Settings:** the options in C8, C15 and C16, with `IDRAK_DOCUMENT_LAYOUT`, `IDRAK_DOCUMENT_READING_ORDER`,
`IDRAK_DOCUMENT_MAX_STREAM_BYTES`, `IDRAK_DOCUMENT_CACHE_BYTES`, `IDRAK_DOCUMENT_IMAGE_ENCODING`,
`IDRAK_DOCUMENT_JPEG_QUALITY`, `IDRAK_DOCUMENT_EMBED_FONTS`, `IDRAK_DOCUMENT_DPI`, `IDRAK_DOCUMENT_FONTS`,
`IDRAK_DOCUMENT_SAVE_MODE`, `IDRAK_DOCUMENT_SIGNATURE_ONLINE` (off by default), `IDRAK_FONT_CURVE_TOLERANCE` and
`IDRAK_FONT_CACHE` read by
the library (names proposed), added to the
CLI's variables table (`src/Idrak.Cli.Commands/Shared/EnvironmentVariables.cs`, which a test checks). Defaults: the stream limit and
the cache from the machine's measured memory, never a fixed size in GB.

---

## Phase D0: the contract and its registries

- C1-C25 in `Idrak.Abstraction.Data`, with XML docs; `DocumentFormats`, `DocumentTextLayouts`, `DocumentWriters`,
  `DocumentPageDrawers` and `FontWriters` with guarded wrappers (shadow: the same page's runs, text and image list
  compared, the same written bytes, the same drawn pixels within a tolerance; fall back per call), `Ensure`d defaults.
  The parts of C19-C25 a later phase fills (forms, annotations, signatures, editing) are in the contract from D0, so
  the API is designed once; a format answers "not supported yet" naming the phase until it lands.
- The `lines` and `content` layouts (in `Idrak.Abstraction`, as they need nothing but the runs and `ReadingOrders`).
- Contract suites in `Idrak.Abstraction.Testing`: `DocumentFormatSuite` (a format's `Accepts` on its own and others'
  headers, page count against pages read, every run's box inside its page, thread-safety: pages read in parallel give the
  same as one by one, limits honoured, a damaged file failing with `InvalidDataException`, never another exception);
  `DocumentWriterSuite` (pages of each kind written then read back through the registered format: paths, glyphs, text
  and raw pixels exact; the same input, the same bytes; a page's memory bounded); `DocumentPageDrawerSuite` (a page of
  known paths, glyphs and images drawn: coverage within its tolerance of the rasterizer's own drawing; parts left out
  when asked; a missing font counted, never thrown); `FontWriterSuite` (a face written and read back through
  `FontFormats`: outlines, advances, characters and shaping kept; a subset keeps its glyphs and their parts).
- **The plug-in path proven:** a toy format and a toy writer in `tests/Idrak.PluginTests` (an outside assembly, public
  API only) register, open and write their files through `DocumentFormats` and `DocumentWriters`, override a name, shadow
  and fall back, as the existing proof families do for their contracts. This is the path Office will take.

**Done when:** the suites pass on the toy format and the toy writer, the inventory and API dump list the new contracts,
and "not registered" names the registered formats, writers, drawers and font writers.

**Tests (CPU here):** one build; `documents contracts|document layouts|outside plug-in|override loop:|public API|
abstraction inventory|conformance kit:`.

**As built (2026-11-18):**
- **The contract**, C1-C25 in `Idrak.Abstraction.Data` (`DocumentPages.cs`, `Documents.cs`, `DocumentTextLayouts.cs`,
  `DocumentWriters.cs`, `DocumentDrawing.cs`, `DocumentForms.cs`, `DocumentSignatures.cs`, `FontWriters.cs`), every
  member documented, nothing naming PDF or a font format. Registries `DocumentFormats`, `DocumentWriters`,
  `FontWriters` (`SlotTable`s) and `DocumentTextLayouts`, `DocumentPageDrawers` (`KindTable`s), each with `Register`,
  `Unregister`, `Names`, `Get` ("not registered" naming the registered ones and what to do), `Default`, `Origin`,
  `SetPolicy`, guarded: formats open the app's and the library's documents each over a view of the caller's stream of its
  own and fall back or compare per call (pages: runs, text, images); writers fall back at `Create` / `Edit` and, in
  shadow, write the same calls to a SHA-256 counter compared byte for byte; drawers compare pixels within two levels;
  font writers keep the app's file until whole under FallBack. Library defaults through `LibraryDefaults.Ensure`; no
  format, writer, drawer or font writer is built in yet (D1, D6, D7, D8).
- **Layouts:** `lines` and `content` in Idrak.Abstraction. `lines` groups runs by baseline within `LineTolerance` of the
  larger size (a superscript stays on its line), takes each line's direction from its first strong character in content
  order (a line with none takes the page's; `DocumentOptions.Direction` can set every line), orders runs by place in that
  direction, puts a space where a gap passes `WordGap` of the size, and turns the columns into logical order through the
  reading order; a run's text is in visual order unless `DocumentTextRun.Logical` (an `/ActualText` span, an Office
  run), which goes in through `ToColumns`. Runs without a place are lines of their own, in content order.
- **Decisions the plan did not settle, taken here:** (1) `IReadingOrder` gains `RightToLeft(text)` (P2/P3), answered by
  `bidi` and `runs` from Idrak.Imaging's bidi classes: the Unicode data stays in one place and the direction rule is
  replaceable with the order (an app's order over `bidi` must answer it). (2) `IDocumentPageDrawer.Draw` and
  `IDocument.DrawPage` return `DocumentDrawing` (the image and its issues), so a missing font is counted, not thrown.
  (3) Annotation kinds are names in `DocumentAnnotationKinds`. (4) A font subset keeps glyph ids (the others written
  empty), as an embedding with `CIDToGIDMap Identity` needs. (5) A builder writes as it goes, so a failure past `Create`
  reaches the caller; an edit under Shadow is the library's alone. (6) Size limits are a quarter (streams) and an eighth
  (cache) of the memory free when a document opens, objects one per KiB of it, nesting 64; all settable.
- **Settings:** `IDRAK_DOCUMENT_LAYOUT`, `_READING_ORDER`, `_MAX_STREAM_BYTES`, `_CACHE_BYTES`, `_IMAGE_ENCODING`,
  `_JPEG_QUALITY`, `_EMBED_FONTS`, `_SAVE_MODE`, `_DPI`, `_FONTS`, `_SIGNATURE_ONLINE` and `IDRAK_FONT_CURVE_TOLERANCE`
  read by the options (a bad value refused naming its variable) and in the CLI's table (no command reads documents yet).
  The table also gained the four `DOTNET_EventPipe*` variables `trace --stacks` sets, which its check found missing.
- **The kit:** `DocumentFormatSuite`, `DocumentWriterSuite`, `DocumentPageDrawerSuite`, `FontWriterSuite` and
  `Conformance.CheckDocumentFormat`, `CheckDocumentWriter`, `CheckDocumentPageDrawer`, `CheckFontWriter`;
  `ReadingOrderSuite` checks `RightToLeft`. Each suite is shown to fail an implementation that breaks it.
- **The plug-in path** (`tests/Idrak.PluginTests/DocumentPlugin.cs`, public API only): a toy format and writer
  register, write by extension, open by first bytes from a file, a stream and memory, pass both suites and unregister;
  over a stand-in library format and writer of the same name (`outside-proof`, a library default for the process) the
  app's failure reaches the caller, falls back under FallBack and is compared under Shadow. A drawer and a font writer of
  the test's own (`test-guard` stand-ins) prove their guards and the two other suites.
- **Tests:** groups `documents contracts` (6) and `document layouts` (5), right-to-left and mixed lines in generic
  text. Run on the CPU with the phase's filter and `every environment variable read in src|the environment table names
  real commands|test runner|generic components: reading order`: 67 tests, 66 passed and the table check failed (the
  `DOTNET_EventPipe*` gap, fixed); rerun with the API and inventory checks without their update flags: 9 passed. The
  API dumps (`Idrak.Abstraction`, `Idrak.Abstraction.Testing`) and the inventory regenerated through the runner.
- **Left for later phases:** every implementation (PDF from D1); the toy's `NotSupportedException`s for forms, links,
  bookmarks, labels, encryption, signing and editing are the toy's own; the library's answer them when D9-D12 land.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents contracts|document layouts|outside plug-in|override loop:|public API|abstraction inventory|conformance kit:|every environment variable read in src|the environment table names real commands|test runner|generic components: reading order"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d0.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

## Phase D1: PDF, the file

In `Idrak.Data` (`src/Idrak.Data/Pdf/`, internal except `PdfFormat`), registered as `pdf`:
- **Lexer and objects** on spans: null, booleans, integers and reals (invariant), literal strings (escapes, octal,
  balanced parentheses, line ends), hex strings, names (`#xx`), arrays, dictionaries, streams (`/Length` direct or
  indirect, a wrong length recovered by `endstream`), indirect references resolved lazily with cycle detection.
- **Cross-reference:** classic tables and trailers, cross-reference streams (`/W`, `/Index`), object streams
  (`/ObjStm`, read once and cached), hybrid files (`/XRefStm`), incremental updates through `/Prev` (the newest object
  wins), the header found within the first KiB (bytes before it allowed).
- **Repair:** when the cross-reference is missing or wrong, a scan for `n g obj` and trailers rebuilds it; `Info.Repaired`.
- **Filters:** `FlateDecode` (`ZLibStream`, raw deflate as a fallback for files without a zlib header), the PNG and TIFF
  predictors, `ASCIIHexDecode`, `ASCII85Decode`, `LZWDecode` (early change), `RunLengthDecode`; chained filters; the image
  encodings (`DCTDecode`, `JPXDecode`, `CCITTFaxDecode`, `JBIG2Decode`) passed through undecoded for D4. Each output
  bounded by `MaxStreamBytes`.
- **Document information:** the `Info` dictionary (PDFDocEncoding and UTF-16 strings, `D:` dates), the version, the page
  count from the page tree (`/Count` checked against the kids, the kids winning).

**Done when:** every fixture's objects, cross-references and information read exactly; repaired files open; a stream
bomb, an object cycle and a deep nesting fail fast with the limit named; the suite passes on `pdf` (pages without text
yet).

**Tests (CPU here):** one build; `documents pdf: file|documents contracts|public API|abstraction inventory`.

**As built (2026-11-18):**
- **The reader** (`src/Idrak.Data/Pdf/`, internal but `PdfFormat`, registered as `pdf` through Idrak.Data's
  `LibraryRegistrations`): `PdfParser` (a ref struct over the file's bytes: whitespace and comments, numbers invariant and
  as damaged writers write them, literal strings with escapes, octal, nested parentheses, CR / CR LF ends and
  continuations, hex strings with an odd last digit, names with `#xx`, arrays, dictionaries, `n g R`; keys compared as
  bytes, the common names made once); `PdfValue` (numbers, booleans and references held without a box); `PdfFile` (the
  header within the first KiB, objects read on demand from a file's handle through `RandomAccess`, from memory in place or
  under one lock from any other stream, each parsed once into a cache within `CacheBytes`, object streams decoded once and
  `/Extends` followed; `/Length` direct or indirect, a wrong one found by `endstream`); `PdfXrefReader` (tables, streams
  with `/W` and `/Index`, hybrids through `/XRefStm`, `/Prev` with the newest entry winning, offsets from the file's start
  or, when that finds no section, from a header after junk; the repair: the file scanned in 1 MiB windows for `n g obj`
  and `trailer`, object streams' headers listed, the catalog found by its type; `Info.Repaired`); `PdfFilters` (Flate
  through `ZLibStream`, raw deflate without a zlib header, data damaged part of the way keeping what decoded; LZW with and
  without early change; ASCIIHex; ASCII85; RunLength; the PNG and TIFF predictors in place at 1 to 16 bits; chains; the
  identity crypt filter; DCT, JPX, CCITT and JBIG2 handed over encoded); `PdfText` (PDFDocEncoding, UTF-16 and UTF-8 with
  their marks, language marks dropped, `D:` dates whole, partial and without the last apostrophe); `PdfDocument` (the page
  tree walked once without recursion, MediaBox, CropBox, Rotate and Resources inherited, the kids counted over `/Count`;
  each page's size as shown: the crop box within the media box, `/UserUnit`, the rotation).
- **Limits and cycles:** `MaxStreamBytes` bounds every stored and decoded stream (each filter's output a pooled buffer
  whose capacity never passes the limit: the flate bomb allocates 127 MiB in all for its 64 MiB limit), `MaxObjects` every
  `/Size`, table, stream, object stream, repair and page tree, `MaxDepth` every nesting (objects, the page tree, filter
  chains, and objects needed to read other objects, 16 levels past it); each an `InvalidDataException` naming the limit and
  `DocumentOptions.<Limit>` (and `IDRAK_DOCUMENT_MAX_STREAM_BYTES` for streams). A cycle (an object needed to read itself,
  a chain of references that comes back, a page-tree node that is its own descendant, a `/Prev` loop, an object stream
  holding itself) names "cycle". Neither is repaired around.
- **Decisions the plan did not settle, taken here:** (1) `hostile-huge-size` is confirmed: a `/Size` (or a table's or
  stream's subsections) claiming more objects than `MaxObjects` is refused at open, before anything is sized from the
  claim (the stream's default `/Index [0 /Size]` lists two billion entries; qpdf, pdfminer.six and pdfium open it by
  ignoring `/Size`). (2) A `/Prev` loop fails (the sample's expectation), it is not read around. (3) `ReadPage` with text,
  images, paths or glyphs asked decodes the page's content streams (checked, then dropped until D3's interpreter), so a
  stream bomb or a reference cycle in `/Contents` trips at "page 0 read" as the samples say; a damaged content stream is a
  `damaged-stream` issue; `DocumentParts.None` reads the size alone. (4) An object not where the cross-reference says
  (or damaged) has the document repaired once, by scanning, and read again; a repair that finds nothing better leaves the
  object's own error. (5) The version is the header's, or the catalog's `/Version` when later. (6) A date without an
  offset is given at offset zero; one that is not a date is kept as written in `Entries`. (7) `Attachments` is empty until
  D4 (the contract's suite reads it on every document); `Fields`, `Outline`, `Signatures` and `PageLabel` answer a
  `NotSupportedException` naming D9, D10 or D12, as D0 decided; an encrypted file one naming D2, and the public-key
  handler naming `Adobe.PubSec`. (8) `IDocumentFormat`, `DocumentFormats`, `IDocument`, `IDocumentPageDrawer` and
  `DocumentPageDrawers`, used by Idrak.Data alone today, are listed in `tests/Idrak.Tests/data/shared-contracts.txt`
  (decision 1 keeps them in Idrak.Abstraction for every format and package). (9) Idrak.Data grants its internals to the
  tests (the object layer is checked object by object). (10) D0's outside plug-in check that nothing detects a PDF header
  now says the toy is not taken for one.
- **Tests:** group `documents pdf: file` (5): the format registered and every sample that is neither encrypted nor hostile
  (43) opening with its page count, version, information, page sizes, rotations and repaired flag; each D1 sample's
  cross-reference sections and every live object in the canonical form (167), read alike from several threads, and every
  listed stream of every sample (176) decoded to its SHA-256, a wrong `/Length` the only one found by `endstream`; the
  nine D1 hostile samples tripping their limit where their JSON says, the 23 encrypted ones refused; `DocumentFormatSuite`
  on `pdf` (the readable samples, the encrypted and cycle and depth ones as invalid, 60 cut or changed files); the lexer,
  text and dates on hand-made bytes, and hand-made files read alike from a file, memory and a plain stream, without a
  cache, repaired when shifted, a bomb and a deep page tree failing. The samples come from `IDRAK_DOCUMENTS_SAMPLES`
  (default `test-output/documents/samples`); missing, the group fails at once naming the command that makes them. First run
  of the phase's filter with `documents contracts|test runner|every environment variable read in src`: 25 tests, 7 failed
  (a table entry's type read without the space before it, so every table was repaired: the repaired flags, a freed object
  back and the `/Prev` loop missed; an internal abstract class outside an Abstractions namespace, made sealed; the
  contracts above), then 1 (the D3 and D4 samples state no information: compared where stated), then with D0's groups
  `document layouts|outside plug-in|override loop:|conformance kit:` added, after D0's check above: 70 passed, 0 failed.
  The API dump (`Idrak.Data`) and the inventory regenerated through the runner.
- **Timings** (this container, 4 cores, Linux; the owner's devices give the real ones): opening and reading the 17 D1
  samples and the twin (44 pages, every part, the information) cold 129-155 ms (8-9 ms a document, mostly compiling),
  warm median 3.0-3.2 ms a pass (175-185 µs a document), 479 KiB allocated a pass; the managed heap at most 16 MiB, the
  process's working set 63 MiB at most. The hostile files fail in 0.2-3 ms, the two bombs in 30-170 ms. The three measured
  limits cost about 2 µs an open.
- **Left for later phases:** text (D3: `hostile-form-recursion`, a D3 sample, trips in its interpreter), images and
  embedded files (D4), encryption (D2), the rest from D6.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: file|documents contracts|document layouts|outside plug-in|override loop:|conformance kit:|public API|abstraction inventory|every environment variable read in src|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d1.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

## Phase D2: PDF encryption

- The standard security handler: revisions 2-4 (RC4 40 to 128 bits, and AES-128 with `/CF` crypt filters, `/StmF`,
  `/StrF`, `/Identity`, `/EncryptMetadata`) and revisions 5-6 (AES-256, the revision 6 hash with SHA-256, -384 and -512).
- The empty user password first, then `DocumentOptions.Password` (as user or owner password); a wrong or missing one is
  a clear error naming the option. The password's bytes are cleared after use (rule 56); the key is held only while the
  document is open.
- Strings and streams decrypted on read (not the cross-reference stream, nor `/Encrypt` itself); the public-key handler
  and others are a `NotSupportedException` naming the handler.

**Done when:** fixtures of each revision and cipher, with an empty and with a given password, read the same text as their
unencrypted twins; a wrong password fails with the message.

**Tests (CPU here):** one build; `documents pdf: encryption|documents pdf: file`.

## Phase D3: PDF text

- **Pages:** the page tree with inherited `Resources`, `MediaBox`, `CropBox` (the visible box) and `Rotate`; boxes reported
  in the rotated page's coordinates, top-left origin.
- **Content interpreter:** the graphics state stack (`q`, `Q`, `cm`), text objects and state (`BT`, `ET`, `Tf`, `Tm`, `Td`,
  `TD`, `T*`, `TL`, `Tc`, `Tw`, `Tz`, `Ts`, `Tr`), showing (`Tj`, `TJ` with its offsets, `'`, `"`), form XObjects (`Do`,
  recursive, within `MaxDepth`), inline images (`BI`, `ID`, `EI`), marked content with `/ActualText` (the text a span
  stands for wins over its glyphs, as the file asks); render mode 3 makes `Hidden` runs. Unknown operators are skipped.
- **Fonts to Unicode**, in this order (each generic, none for a language):
  1. the font's `ToUnicode` CMap (code space ranges, `bfchar`, `bfrange` with arrays, multi-byte codes);
  2. simple fonts: the base encoding (Standard, WinAnsi, MacRoman, PDFDoc, the font's built-in one) and `/Differences`,
     glyph names to code points by the Adobe Glyph List and its `uniXXXX` and `uXXXX` rules (a generated table, from the
     published list, under its license);
  3. the embedded font's own `cmap`, read through `FontFormats` (glyph to code point reversed), for TrueType and
     OpenType programs, and through `CIDToGIDMap` for composite fonts with `Identity-H` / `Identity-V`;
  4. otherwise the glyph has no code point: U+FFFD in the run and a count in the page's `Issues`, so an app knows to send
     that page to OCR.
- **Widths:** `/Widths` with `/FirstChar` and `/MissingWidth`, composite `/W` and `/DW`, vertical metrics for `-V`; the
  standard 14 fonts' widths from their published metrics (a generated table). Each run's box is exact from the text
  matrix, the size, the spacing and the widths.
- **Words and lines:** spaces the file does not hold are inserted between runs whose gap passes a share of the font size
  (`DocumentOptions`, through the layout); the `lines` layout (D0) makes `Lines` and `Text`.

**Done when:** each fixture's text, every run's box (within float rounding), its font size and hidden flag are as written
into the fixture; a page with an unmapped font reports its count; a page in right-to-left text and one mixing directions
come out in logical order through `bidi` with no language code; pages read in parallel give the same as in order.

**Tests (CPU here):** one build; `documents pdf: text|document layouts|documents pdf: file|documents contracts`.

## Phase D4: PDF images and embedded files

- **Images:** image XObjects and inline images with their box from the transformation matrix; `/Width`, `/Height`,
  `/BitsPerComponent` (1, 2, 4, 8, 16), colour spaces `DeviceGray`, `DeviceRGB`, `DeviceCMYK` (to RGB as `ImageData`
  holds it), `Indexed`, `ICCBased` (by its component count; the profile not applied), `CalGray`, `CalRGB`, `Lab`; `/Decode`
  arrays; image masks (stencils) as grey; soft masks dropped (alpha is not kept, as `ImageData` says). Decoded on
  `Decode()` only, into 8-bit planes where the data is 8-bit.
- `DCTDecode` images decode through `ImageCodecs` (`jpeg`); `JPXDecode`, `CCITTFaxDecode`, `JBIG2Decode` are handed over
  encoded, and decode when an app registers a codec of that name.
- **Embedded files:** the `EmbeddedFiles` name tree and file-attachment annotations, as `DocumentAttachment`s read on
  demand.

**Done when:** each fixture's images give their pixels exactly (raw and JPEG against the codec's own decode of the same
bytes), with their boxes; a scan-like page (one image, a hidden text layer) gives both; embedded files read back their
bytes.

**Tests (CPU here):** one build; `documents pdf: images|documents pdf: text|documents contracts`.

## Phase D5: fixtures, docs, the owner's runs

- **The samples live in the ignored folder (decision 13).** `tools/documents/make_fixtures.py` writes every sample and
  its expected results into `test-output/documents/samples/` (git ignores `test-output/`; nothing of it is committed),
  from PyPI packages pinned in `tools/documents/requirements.txt` (run with `-I`, as plan 24's tokenizer tools): the
  sample files, a JSON of what each holds, the reference reader's text and the reference renderer's pages as PNG, and
  the reference font tools' dumps for D8. The tests read them from `IDRAK_DOCUMENTS_SAMPLES` (default
  `test-output/documents/samples`); a group whose samples are missing fails at once, naming the one command that makes
  them, never passes empty. The session makes them in its container before a targeted run; the owner's command list
  makes them once after the pull. Only the fonts already under `tests/Idrak.Tests/data/fonts/` are inputs from the
  repository.
- The samples, each with a JSON of what it holds (text, boxes, images): classic and
  stream cross-references, incremental updates, a broken table, every filter and predictor, every encryption revision,
  simple and composite fonts with and without `ToUnicode`, embedded TrueType, Type 3, right-to-left and mixed text,
  vertical text, hidden text over a scan, each image kind, embedded files; and the hostile ones (a stream bomb, cycles,
  deep nesting). The fonts come from the test data already in the repository. Added for D6 and D7: vector pages (every
  path operator, both fill rules, caps, joins, dashes, clips), each colour space, shadings and patterns, transparency,
  embedded Type 1 and Type 3 fonts, a page drawn by the reference renderer saved beside each as PNG, and image PDFs as
  scanners and phones write them (one JPEG a page, a Flate page, a hidden text layer). Added for D8-D12: each font
  format (and a subset of each test font, dumped by the reference font tools), every field kind filled and empty, every
  annotation kind with and without appearances, outlines, destinations and page labels, tagged files, files edited by
  incremental updates, encrypted with AES-256, and signed (a test certificate made by the generator, never a real one)
  with valid, broken and changed-after-signing signatures.
- A real-files group, `documents real:`, for the owner: every PDF in a folder (`IDRAK_DOCUMENTS_REAL`) opens, every page
  reads within the limits, and a summary (pages, runs, images, issues, time per page cold and warm, rule 76) is written;
  differences against a reference extractor's text are listed, not failed (the reference fixes what the text must say,
  not how fast Idrak may be).
- The README's `Idrak.Data` line and `src/Idrak.Data/README.md` name the document formats; the Office plug-in is named as
  to come.

**Done when:** the generator reproduces every sample byte for byte from its seed and its pinned versions, nothing under
`tests/` or `src/` holds a generated sample, the docs say what is read, written, edited, signed and drawn and what is
not, and the owner's command list is in the as-built note.

**As built, first piece (2026-11-18): the generator and the D1-D4 samples.** `tools/documents/make_fixtures.py` (the
`docfixtures` package beside it, `requirements.txt` pinned with every dependency, `README.md` with the JSON format)
writes 75 samples into `test-output/documents/samples` (D1 25, D2 23, D3 20, D4 10; 10 hostile, each with the limit it
must trip), each with its expected JSON, the reference reader's and renderer's text, the renderer's pages (PNG, 100
DPI) and its images' expected pixels; `manifest.json` lists every file's SHA-256. Every PDF byte is written by the
generator; qpdf (pikepdf) decodes every listed stream to its hash, pdfminer.six is the reference reader, pdfium
(pypdfium2) the reference renderer and the check of every run box. Two runs give the same 396 files; a run takes about a
minute. The owner makes them with `python -I tools\documents\make_fixtures.py -o test-output\documents\samples` after
`python -m pip install -r tools\documents\requirements.txt`. To come in D5: the samples of D6-D13 (each phase adds a
`docfixtures/samples_dN.py`, see the README), the `documents real:` group and the docs.

**As built, second piece (2026-11-18): the D6-D13 samples, the checkers, the large set.** 75 more samples (150 in all,
767 files and the manifest; two runs identical, `--compare` exit 0; about 90 seconds a run), each with its JSON and the
reference tools' results, the README giving every key the tests read: **D6** the writer's inputs (seven PNGs with
what `Auto` measures and must choose) and image PDFs as scanners and phones write them; **D7** eleven vector pages (every
path operator, both fill rules, caps, joins, miter limits, dashes, clips, every colour space and function type, every
shading type, tiling and shading patterns, ca/CA, the 16 blend modes, a soft mask, isolated and knockout groups, a
Type 1 font made here from DejaVu Sans' outlines with fontTools' eexec, Type 3, TrueType, Type1C and OpenType CFF,
images on turned and skewed matrices, turned and cropped pages), each page's paths, clips and shadings by construction
and every path matched exactly against pdfium's own path objects; **D8** WOFF, WOFF2 (with the glyf/loca and hmtx
transforms), CFF from TrueType, TrueType from CFF and subsets with and without layout closure of the four test fonts
whose licences allow it, each with its dump (tables, metrics, cmap, advances, every outline, names, each lookup's
glyphs) and HarfBuzz's shaping, each passing the OpenType Sanitizer and keeping its source's outlines, advances,
characters and shaping; **D9** every field kind filled, empty, under /NeedAppearances, inherited, XFA, flattened (pdfium's
form reading agrees with every field); **D10** every annotation kind with and without appearances, replies, popups,
links, named destinations, bookmarks, page labels, a tagged page, redaction and its applied result (pdfium agrees with
every annotation, link, bookmark, label and the structure tree); **D11** classic and stream incremental updates with
each revision's end, merge inputs whose inherited resource names clash, a stamp, a split, AES-256 with passwords and
permissions, page operations (qpdf performs each and gives the pages the JSON says); **D12** a test PKI made here (root,
intermediate, RSA, EC, revoked and timestamp certificates, an untrusted root, two CRLs, an OCSP response, an in-process
timestamp authority; RSA keys from seeded primes, deterministic ECDSA, a seeded RSA-PSS salt) and 13 signed files
(PAdES B-B with RSA, RSA-PSS and ECDSA, B-T, B-LT, a revoked signer, a broken digest, a wrong chain, changes after
certification allowed and not, two signatures, the two legacy SHA-1 forms), each validated by pyHanko exactly as its
JSON expects (the legacy forms through pyHanko's CMS check, cryptography and pyhanko-certvalidator). **D13**, only with
`--large`, into `test-output/documents/large` (16 seconds, 1,007 files, two runs identical): 3,000 pages of text and images, 1,000 small files, an A0
page with a 418 MB image, forms nested 24 deep, with timing-free checks. Two checkers for what Idrak will write:
`tools/documents/check_written.py` (qpdf, pdfminer.six, pdfium, warnings listed, an expected JSON compared) and
`tools/documents/check_font.py` (fontTools, the Sanitizer, HarfBuzz, a source compared). New pins: pyHanko 0.38.1,
pyhanko-certvalidator 0.33.0, asn1crypto 1.5.1, Brotli 1.2.1, opentype-sanitizer 9.2.0 and their dependencies. The
reference tools' gaps found are in the README (pdfium ignores knockout groups and has no XFA; pdfminer.six cannot
read an embedded Type 1 font's built-in encoding; pyHanko reads neither legacy SHA-1 form). The owner makes the large
set with `python -I tools\documents\make_fixtures.py --only none --large` after the samples.

## Phase D6: PDF writing

In `Idrak.Data` (`src/Idrak.Data/Pdf/`, beside the reader, sharing its objects and filters), registered in
`DocumentWriters` as `pdf`:
- **The file:** objects written on spans (numbers invariant, strings escaped, names `#xx`), streams Flate-compressed
  (`ZLibStream`, level from the options), object streams and a cross-reference stream (or a classic table for
  `Version` 1.4), the trailer, an `ID` only when given (`Deterministic`). Pages written as they finish; the page tree,
  fonts and the cross-reference at `Finish`.
- **Image pages** (decision 8): an `ImageData` placed in a box, grey or RGB, 8-bit, `Flate` (exact pixels, with the PNG
  predictor where it shrinks the stream) or `Jpeg` (`DCTDecode`, through `ImageEncoders` "jpeg"); `Auto` measures the
  image (levels used, edges) and picks per image. A scan written as an image PDF with its OCR text as a `Hidden` layer
  is a searchable PDF. **Registers a `jpeg` encoder** in `ImageEncoders` (`Idrak.Imaging`, over the existing
  `JpegEncoder`), as `png` is.
- **Vector pages:** `DocumentPath`s as path operators (`m l c v y h re`, the fill rules, stroke, caps, joins, dash,
  colours), clipping, the graphics state stack.
- **Text:** a face embedded once per document as a composite font (`Type0`, `Identity-H`, `CIDToGIDMap Identity`) with its
  program (`FontFile2` for TrueType outlines, `FontFile3` / `OpenType` for CFF), `/W` widths from its advances, a
  `ToUnicode` CMap from the runs' text and glyphs, `/ActualText` where a glyph stands for several characters or a
  character for several glyphs. **Subsetting** (`Subset`, the default): only the glyphs used, written through
  `FontWriters` (D8: "truetype" or "opentype-cff" with `Glyphs` set), so a page in one font is not the font's whole size;
  `Whole` embeds the file as read. A face a font map made (legacy fonts) is written as its glyphs, like any face.
- **Pages the library draws:** a helper turns a page of shapes (`ShapePath`, `ShapeArea`) and shaped text into page
  builder calls, so a synthetic page or plan 38's glyph sheet is written as vector PDF, not only as an image.

**Done when:** every page kind written reads back through D1, D3, D4 and D7 exactly (paths, glyphs and text, raw image
pixels; JPEG pages within the codec's own round trip); the same input writes the same bytes on every machine; a
reference reader in `tools/documents/` (from PyPI, run with `-I`) opens every written file without a warning and
extracts the same text; a 500-page image PDF is written with one page's buffers (measured, rule 77).

**Tests (CPU here):** one build; `documents pdf: write|documents writers|documents pdf: file|documents contracts|public API|
abstraction inventory`.

## Phase D7: PDF vector content, and pages drawn to pixels

- **Vector content read:** the content interpreter (D3) also gives paths: construction (`m l c v y h re`), painting
  (`S s f F f* B B* b b* n`), clipping (`W W*`), the line state (`w J j M d`), colours (`g G rg RG k K cs CS sc scn`,
  device, ICC-based by its component count, `Indexed`, `Separation` and `DeviceN` through their tint transform), each
  path transformed by the current matrix into page points, as `DocumentPath`s with their clips. Shading patterns and
  tiling patterns as their own parts (drawn below), transparency (`ca`, `CA`, soft masks) as alpha on each part.
- **Fonts for drawing:** embedded TrueType and CFF programs read through `FontFormats` (the existing `opentype` reader);
  embedded **Type 1** programs (`FontFile`: eexec-decrypted charstrings to outlines) as a new `IFontFormat` "type1" in
  `Idrak.Imaging`; **Type 3** glyphs drawn from their own content streams; fonts not embedded (the standard 14 and
  others) from `DocumentDrawOptions.Fonts`, matched by name and then by coverage, else counted in `Issues`.
- **The drawer ("coverage", `Idrak.Imaging`):** each part in content order onto the page's pixels: fills by their rule,
  strokes through `Pens` (the stroke's outline, then filled), glyph runs through their faces' outlines, images placed by
  their matrix (the shared `Homography` / `Warp` resampling), clips applied, shadings (axial, radial, function-based,
  meshes as triangles), transparency composed. **The rasterizer gains a fill rule** (`NonZero` exact for any overlap,
  and `EvenOdd`) as an option of `IOutlineRasterizer.Fill`, the current behaviour staying the default for glyphs (rule 71:
  every existing page byte-identical).
- **Speed and memory:** a page drawn band by band (rows walked in order, rule 72) so a large page at a high DPI holds one
  band of coverage; pages drawn in parallel when several are asked (rule 79, checked to pay).

**Done when:** each fixture's paths read exactly; each fixture page drawn matches a reference renderer's pixels (from
PyPI in `tools/documents/`, the reference fixing the result, never the speed) within a stated tolerance; every page D6
wrote draws back to the page the library drew as an image, within the tolerance; a scan's page draws its image and,
when asked, its hidden text; existing glyph drawing is byte-identical.

**Tests (CPU here):** one build; `documents pdf: vector|documents drawers|documents pdf: write|documents pdf: images|
fonts|synthetic pages`.

## Phase D8: fonts written (TrueType built in, WOFF from a plug-in)

Two packages (decisions 18 and 20): **built in**, in `Idrak.Imaging` beside `OpenTypeFormat` and `CffOutlines`, the
"truetype" writer, the one Idrak works from; **a plug-in project**, `Idrak.Imaging.Plugins` (new, beside the other
`*.Plugins` projects; namespace `Idrak.Imaging.Plugins.Fonts`; decision 21) with "woff" alone, read and written, registering into the same
`FontWriters` (C18) and `FontFormats` from outside the library: the proof that a font format plugs in. No JSON font
format; no OpenType CFF writer and no WOFF2 (decision 20).
- **"truetype" (`.ttf`, built in):** the tables a face needs, from any `IFontFace`: `head`, `hhea`, `maxp`, `hmtx`, `cmap` (formats
  4 and 12, from the face's characters), `glyf` and `loca` (quadratic outlines; a cubic outline turned into quadratics
  within `CurveTolerance`, measured), `name` (only the strings given), `OS/2` (metrics and ranges from the glyphs),
  `post` (glyph names when the face has them); the face's layout tables (`GSUB`, `GDEF`, `GPOS`) written as they are, so
  a mapped legacy face or plan 38's made face shapes in any software, not only in Idrak; checksums and the table order
  the specification asks for.
- **"woff" (`.woff`, plug-in):** a written font's tables, each zlib-compressed through `ZLibStream` when that is
  smaller; and read back through `FontFormats` like any font. Its test registers it the way an app would and proves
  write, read, override and fall back, as the other plug-in proofs do.
- **Legacy fonts converted once, and Idrak works from the completed font** (owner, 2026-11-18): a face a font map made
  is written as a complete TrueType font: the map's Unicode characters, the glyphs drawn for what the font lacked
  (digits, marks, composed letters), its ligatures, pairs, second style and words, and the GSUB / GDEF / GPOS made from
  the map. Read back, it needs no map (it has its own GSUB, so detection leaves it alone) and shapes the same in Idrak
  and in any other software. A converted-font cache (a folder setting, `IDRAK_FONT_CACHE`, default under the user's
  cache folder, measured room, never a fixed size) keeps each converted font keyed by the source file's hash, the map
  and the drawing settings (`FontMapDrawing`), so a legacy font is mapped and drawn once and then opened as a plain TTF
  (work done once, rule 75; cold and warm loads measured, rule 76). **Gap fixed here:** a page turns on the hand's
  features (`PAIR`, `PAI2`, `WORD`) only when `FontMaps.MappedBy(font)` finds the map
  (`src/Idrak.Imaging/PageLayoutImplementations.cs:208`), so a converted font read later would lose them; the check
  reads the features from the font's own GSUB instead, the same for a mapped face and a converted one (pages drawn
  with a mapped face stay byte-identical, rule 71).
- **Subsetting:** `Glyphs` keeps those glyphs, their composite parts and, with `KeepLayout`, every glyph the layout
  tables can reach (a closure over `GSUB`), with the layout tables rewritten to the kept glyphs; without it the layout
  tables are dropped (what a PDF embedding needs).
- **Results proven:** every test font written in each format reads back through `FontFormats` with the same outlines
  (exact for the same outline kind, within `CurveTolerance` across kinds), advances, characters and shaping (every
  sample line shaped the same before and after); the reference font tools (PyPI, D5) open every written font, and the
  sanitizer browsers use accepts it.

**Tests (CPU here):** one build; `fonts write|fonts woff|fonts|shaping|public API|abstraction inventory`.

## Phase D9: forms

- **Read:** the `AcroForm` tree (inherited values, kids, widgets), every field kind (C19), its value and options, its
  widgets' places and appearance states; XFA forms' data (`datasets`) read as fields.
- **Fill and create:** `SetField` (C23) and `AddField` (C24) for every kind; each widget's appearance stream generated
  (text laid out in the field's font, size, alignment, comb and multi-line rules, through the shapers, so every script
  fills right); `NeedAppearances` not relied on.
- **Flatten:** a field's appearance drawn into its page's content and the field removed (an option per field or all).

**Done when:** every sample's fields read as written; a form filled by Idrak reads back with its values in Idrak and in
the reference reader, and draws (D7) with the values in place; a flattened form keeps its look and has no fields.

**Tests (CPU here):** one build; `documents pdf: forms|documents pdf: write|documents drawers`.

## Phase D10: annotations and navigation

- **Annotations (C20):** every kind read with its points, ink strokes, replies and appearance; drawn by the drawer when
  asked (their appearance streams; one without is drawn from its properties); created (with an appearance made) and
  removed; redaction annotations applied on request (the content under them removed from text, paths and image pixels,
  not only covered).
- **The redaction command** (decision 17): `idrak data redact IN.pdf (--boxes FILE | --text TEXT | --marked) -o OUT.pdf`
  (boxes by page, every match of a text, or the file's own redaction annotations), with live progress (pages done,
  areas removed) and a summary of what was removed per page; the output written as a rewrite (D11), so nothing removed
  survives in an earlier revision. Its help text and README use a generic example.
- **Navigation (C21):** links (URI and destinations), bookmarks read and written, named destinations, page labels.
- **Tagged files:** the structure tree read, and a `tagged` text layout in `DocumentTextLayouts` that orders text as the
  author tagged it (the `lines` layout stays the default for files without tags).

**Done when:** every sample's annotations, outlines, destinations and labels read as written; created ones read back in
Idrak and the reference reader; a redacted area holds no text, path or pixel of what it covered; a tagged sample's text
comes in its tagged order.

**Tests (CPU here):** one build; `documents pdf: annotations|document layouts|documents pdf: write|documents drawers|
cli: data redact|environment variables|help text`.

## Phase D11: editing in place, and saving

- **Incremental updates (C23, `Incremental`):** changed and new objects appended after the source's bytes with a new
  cross-reference section pointing back (`/Prev`), so the source is untouched and earlier signatures stay valid; an
  update on a file with a cross-reference stream keeps a stream.
- **Rewrite (`Rewrite`):** one clean file: only the objects reachable from the root, renumbered, object streams,
  streams recompressed when it pays (measured), duplicate images and fonts written once (by content hash).
- **Pages:** insert from another document (its resources carried and renamed where they clash), remove, move, rotate,
  merge files, split a file; content added over an existing page (a stamp, a watermark, page numbers).
- **Saved encrypted:** AES-256 (revision 6) with a user and an owner password and permissions, through D2's handler;
  removing encryption when the owner password is given.

**Done when:** each edit reads back as made; an incremental update leaves the source's bytes as its prefix; a signed
sample edited incrementally keeps its signature valid (D12); a rewritten file is never larger than its source unless
asked to add, and reads the same; an encrypted file opens with each password and not without.

**Tests (CPU here):** one build; `documents pdf: edit|documents pdf: encryption|documents pdf: write|documents pdf: file`.

## Phase D12: signatures

- **Read and check (C22):** signature fields and their CMS (`SignedData`, detached), the byte range against the file,
  the digest, the signature (RSA, RSA-PSS, ECDSA through `System.Security.Cryptography`), the signer's chain built
  against the trust the app gives (`SignatureCheckOptions`: certificates, or the machine's store when the app asks),
  timestamps (RFC 3161 tokens checked), revocation from the file's DSS (OCSP responses and CRLs), online only when
  `IDRAK_DOCUMENT_SIGNATURE_ONLINE` or the option turns it on; changes after signing found by comparing the signed
  revision with the current one (form filling and annotations allowed by the signature's permissions, or not).
- **Sign (C25):** a signature field (new or empty), its appearance, the byte range reserved and filled, CMS built on the
  DER here (the base library's `System.Formats.Asn1` when the shared framework has it, checked at the phase's start;
  otherwise a small DER writer here: no NuGet package, rule 70), PAdES B-B, B-T (a timestamp from the authority the app
  names) and B-LT (revocation data added to the DSS), always as an incremental update.
- **Untrusted input:** a signature never makes a file trusted on its own; a check reports each step and why it failed.

**Done when:** every sample's signatures check as the generator made them (valid, broken digest, wrong chain, changed
after signing); a file Idrak signs checks valid in Idrak and in the reference tool, and a second signature keeps the
first valid.

**Tests (CPU here):** one build; `documents pdf: signatures|documents pdf: edit|documents pdf: file`.

## Phase D13: any workload

- **The documents bench** (`idrak bench documents`, with the library's settings as variables, `--save` and `--compare`):
  opening, reading text, reading images, drawing a page at chosen DPIs, writing an image PDF and a vector PDF, editing
  incrementally and rewriting, signing and checking; each measured cold and warm apart (rule 76), on the samples and on
  a generated large set (thousands of pages, many small files, one huge image, deep forms).
- **Bounded memory at any size:** a large file read page by page with the cache's measured bound; a large file written
  and drawn with one page (or one band) of buffers; many files read in parallel within `ComputeResources` (rule 79,
  checked to pay); every limit (C8) defaulted from the machine's measured memory and settable.
- **The owner's real files** (D5's `documents real:`) also written back, edited, drawn and timed, the summary under
  `test-output/documents/`.

**Done when:** the bench's numbers are saved for the owner's devices as the baseline; no case grows memory with page
count beyond its cache and one page's buffers; each slow path the bench finds is fixed in the library (CLAUDE.md:
speed and memory first), its results unchanged.

**Tests (CPU here):** one build; `documents|fonts write`; the bench's real numbers come from the owner's devices.

## Plan 38's needs (moved from plan 38, 2026-11-18)

Plan 38 (handwriting fonts from images) takes everything PDF from here and starts only after this plan is finished
(decision 9):
- **The blank glyph sheet as a PDF** (plan 38's F2, `idrak data font-sheet -o sheet.pdf`): several pages, the grid of
  cells, faint guides (baseline, x-height) and corner marks as vector paths (D6's page-of-shapes helper), each cell's
  label as embedded, searchable text; the same sheet as PNG through `ImageEncoders`.
- **A filled sheet read back from a PDF** (plan 38's F2 and F4, `idrak data font-from sheet.pdf`): a scanned or
  photographed sheet saved as an image PDF gives its page images (D4); a sheet filled on a tablet and saved as a vector
  PDF gives its ink as paths (D7), and either kind can be drawn to pixels at a chosen DPI (D7) so the reading code sees
  one kind of input.
- **Shared, not copied:** the `Homography` / `Warp` helper made shared in D7 is the one plan 38 straightens sheets with.
- **A made font exported in a common format** (owner, 2026-11-18): plan 38's face is written through `FontWriters` (D8)
  as `.ttf`, `.otf`, `.woff` or `.woff2`, with the layout tables its font map made, so it works in any software and
  `data synth --fonts` reads it like any other font.
- **Tablet ink:** ink annotations (D10) give a pen's strokes as paths, another way a sheet written on a tablet comes in.

## Not in this plan

- **Office** (Word, Excel, PowerPoint): a later plug-in on C1-C25 (a package such as `Idrak.Data.Plugins`, namespace
  `Idrak.Data.Plugins.Office`, on the base library's `ZipArchive` and `XmlReader`, no outside dependency).
- **Running anything a file carries:** JavaScript, launch actions, and fetching outside references are never done (the
  untrusted-input rule); they are reported in `Issues`.
- **XFA layout** (drawing a dynamic XFA form's own layout: a deprecated format; its data is read, D9), **3D and
  multimedia** content (reported, not played).
- **Users of the contract:** a `pdf` dataset format in `DataFileFormats` (a row per page), document chunking for
  retrieval (`Idrak.Text`), an `idrak` command that reads or writes a document, OCR of pages with issues, a searchable
  PDF made from an OCR run. Each is a small follow-up once D3 and D6 land; plan 38's commands are plan 38's F6.

## Order and parallelism

At most two agents at a time, each in its own worktree; a merge is built and its filters run before the push.

| Step | Lane A | Lane B |
|---|---|---|
| 1 | D0, the contract | D5's generator and the first samples (no library code; into `test-output/`) |
| 2 | D1, the file | D8, fonts written and read (`Idrak.Imaging` only, needs nothing from D1) |
| 3 | D3, text (after D1) | D2, encryption (after D1) |
| 4 | D4, images and embedded files (after D3) | D6, writing (after D1 and D8; its read-back checks grow as D3 and D4 land) |
| 5 | D7, vector content and drawing (after D3, D4 and D6) | D11, editing and saving (after D2 and D6) |
| 6 | D9, forms (after D7 and D11) | D10, annotations and navigation (after D7 and D11) |
| 7 | D12, signatures (after D11) | D5, docs and the owner's runs |
| 8 | D13, any workload (after all) | none |
| then | plan 38 starts (decisions 9 and 14) | |

D2 and D3 touch different files (`Pdf/Security*.cs`, `Pdf/Content*.cs` and `Pdf/Fonts*.cs`); D1 and D8 too (`Pdf/` and
`Idrak.Imaging`'s font files); D4 and D6 too (`Pdf/Images*.cs`, `Pdf/Writer*.cs` and `Idrak.Imaging`'s `jpeg` encoder
registration); D7 and D11 too (`Pdf/Content*.cs` and `Idrak.Imaging`'s drawer, the Type 1 reader and the rasterizer's
fill rule; `Pdf/Edit*.cs`); D9 and D10 too (`Pdf/Forms*.cs`, `Pdf/Annotations*.cs` and `Pdf/Outline*.cs`). The files
both lanes touch (`api/*.txt`, the inventory, the test list, the variables table) are merged one lane at a time and
regenerated, never merged by hand.

## The owner's runs

Each phase's as-built note gives the exact PowerShell list (from `D:\Projects\Idrak`). The pattern:

```powershell
cd D:\Projects\Idrak; git checkout documents; git pull origin documents
dotnet build -c Release tests\Idrak.Tests
# the samples, into the ignored folder (once per pull that changes tools\documents)
python -I tools\documents\make_fixtures.py -o test-output\documents\samples
# D3 onwards: the owner's own PDFs (a folder of real files; nothing is written into it). From D6 each is also written
# back and read again, and from D7 its pages are drawn: the summary, the copies and the drawn pages go to test-output.
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents real:"; $env:IDRAK_DOCUMENTS_REAL = "D:\Documents\pdf-samples"; $env:IDRAK_DOCUMENTS_OUT = "test-output\documents"
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails.txt
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER, Env:IDRAK_DOCUMENTS_REAL, Env:IDRAK_DOCUMENTS_OUT
```

## Risks

- **Real PDFs break the specification** more than they follow it (wrong lengths, broken tables, fonts with no Unicode).
  Answered by the repair path, per-page failures and the owner's real-files group; each new break found becomes a fixture.
- **Text with no code points** (subset fonts with neither `ToUnicode` nor a usable `cmap`): no library can read it as
  text; the page reports it so an app can OCR it.
- **Reading order from positions** is a heuristic for multi-column pages and tables; the layout is a registry, so a better
  one (or an app's) replaces `lines` without touching the reader.
- **Hostile files:** bombs, cycles and deep nesting are covered by limits and fixtures; the reader never runs or fetches
  anything.
- **The API grows** by the contract (C1-C25): each in the API dump with its reason; no version bump, no CHANGELOG
  (CLAUDE.md).
- **Drawing is the largest piece** (shadings, transparency, Type 1 and Type 3 fonts). Answered by the reference
  renderer's pixels per fixture, a stated tolerance, and `Issues` naming what a page used that is not drawn, never a
  silent wrong page.
- **Subsetting a font wrongly** (a composite glyph whose parts are dropped, a CFF subroutine left out) makes text that
  draws in Idrak and breaks elsewhere: every written file is opened by the reference reader (D6), and the subset's glyphs
  are drawn and compared with the whole font's.
- **The rasterizer's fill rule** touches every glyph drawn today: the current behaviour stays the default, and existing
  synthetic pages are checked byte-identical (rule 71).

- **Signatures are security:** a wrong "valid" is worse than no check. Answered by the generator's broken and changed
  samples, every step of a check reported, and nothing trusted that the app did not give as trusted.
- **TrueType writing and subsetting** are exacting: answered by the reference font tools and the browsers' sanitizer on
  every written font (D8).

## Open questions for the owner (2026-11-18)

Answered the same day: 1 (forms, annotations, signatures and editing are in: decision 10), 2 (the reference tools from
PyPI, their samples under `test-output/`: decision 13), 3 (`Auto`: decision 11), 4 (only the app's certificates:
decision 16), 5 (redaction is a command too: decision 17), 6 (the font plug-in project is `Idrak.Imaging.Plugins`, new:
decision 21). None open.

## Runs

| Date | Phase | Machine | What | Result |
|---|---|---|---|---|
