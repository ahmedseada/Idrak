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
| D2 | PDF encryption: the standard security handler (RC4, AES-128, AES-256), an empty or a given open password | B | Built (2026-11-18) |
| D3 | PDF text: pages, the content-stream interpreter, fonts to Unicode, glyph widths, text runs with boxes, the `lines` layout | A | Built (2026-11-18) |
| D4 | PDF images and embedded files (a scanned PDF's pages as images) | B | Built (2026-11-18) |
| D5 | Fixtures and their generator, READMEs, the owner's runs on real files | A | Built (2026-11-19) |
| D6 | PDF writing: pages of images, vector paths and text (fonts embedded, searchable), several pages, the same bytes from the same input | B | Built (2026-11-18) |
| D7 | PDF vector content read, and any page drawn to pixels (paths, text through the embedded fonts, images, clipping) | A | Built (2026-11-18) |
| D8 | Fonts written: TrueType (`.ttf`) built in, with subsetting and legacy fonts converted once (D6 embeds through it; plan 38 exports through it); WOFF (read and write) from a plug-in, the one proof of the plug-in path | B | Built (2026-11-18) |
| D9 | Forms: fields read (text, check, radio, choice, push button, signature), filled, created, flattened; appearances generated | A | Built (2026-11-19) |
| D10 | Annotations and navigation: every annotation kind read, drawn, created and removed; links, bookmarks (outlines), named destinations, page labels; tagged structure read for reading order | B | Built (2026-11-19) |
| D11 | Editing in place and saving: incremental updates, a full rewrite (unused objects dropped), pages merged, split, inserted, removed, moved, rotated; saved encrypted (AES-256) | A | Built (2026-11-18) |
| D12 | Signatures: read and validated (CMS, the byte range, the chain against trust the app gives, timestamps, revocation from the file), and signing (PAdES baseline) | B | Built (2026-11-19) |
| D13 | Any workload: the documents bench (open, text, draw, write, edit; cold and warm), large and many files, memory bounded, the limits measured | A | Built (2026-11-19; the real-files bullet built with D5's `documents real:`) |

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
24. **The converted-font cache is on by default** (owner, 2026-11-18, "1a"): a legacy font is mapped and drawn once,
    kept as a TrueType file and opened from it afterwards (about 3 ms instead of 75-200 ms in this container). The
    drawn and composed glyphs are kept at whole font units (at most half a unit off, what the TrueType format holds);
    that rounding is the new reference for pages drawn from legacy fonts, an accepted exception to rule 71's
    byte-identical pages. `IDRAK_FONT_CACHE=off` (or `--font-maps cache=off`) gives the exact mapped face back.
25. **Cached and mapped faces are identical** (owner, 2026-11-18, "1a" to "how could it be byte identical?"): the
    mapped face rounds its drawn and composed glyphs to whole font units exactly as the TrueType writer does, so a
    legacy font gives the same outlines, and a page the same bytes, with the cache on or off. Pages drawn from legacy
    fonts change once (decision 24's rounding, now in both paths) and never differ between the two paths again
    (rule 71 holds from here). Built as D8c.
26. **The WOFF plug-in waits** (owner, 2026-11-18: "can the plugin be implemented later and the agent takes a real
    work, the plugin is an extra so it can wait"): D8's plug-in part (WOFF in `Idrak.Imaging.Plugins`, decisions 20
    and 21) is deferred to after the real phases; its agent was stopped before committing anything, and its slot took
    D6. D8 counts as built for every phase that depends on it (they need the built-in TrueType writer only).
27. **Real pieces first, the extra only in a gap** (owner, 2026-11-18: "finish the real ready pieces first then do
    the extra only if a real piece waits on a real piece"): a free plan 28 slot always takes a ready real piece; the
    WOFF plug-in (decision 26) takes a plan 28 slot only when no real piece is ready because every remaining one waits
    on another that is still running, and a real piece that becomes ready never waits for it.
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

**As built (2026-11-18):**
- **The handler** (`src/Idrak.Data/Pdf/PdfSecurity.cs`, `PdfCrypt.cs`, internal): the standard security handler,
  revisions 2-6. Revisions 2-4: the file key by Algorithm 2 (40 to 128 bits: revision 2 five bytes, revision 3 its
  `/Length`, revision 4 its stream filter's `/Length` in bytes, or in bits as some writers put it), checked against `/U`
  (Algorithms 4 and 5), the owner password through `/O` (Algorithms 3 and 7); crypt filters from `/V 4` (`/CF` with
  `/CFM` `None`, `V2`, `AESV2`, `AESV3`; `/StmF`, `/StrF`, `/EFF`, `Identity`, a stream's own `/Crypt` filter first in
  its chain), `/EncryptMetadata false` (in the key and for metadata streams). Revisions 5 and 6: AES-256, the file key
  from `/UE` or `/OE` under the hash of Algorithm 2.B (SHA-256, -384 and -512 rounds over AES-128-CBC; revision 5 SHA-256
  alone). RC4 written here; AES, MD5 and SHA-2 through `System.Security.Cryptography`, one MD5 context reused across a
  check's 50 rounds. Passwords: PDFDocEncoding up to 32 bytes (revisions 2-4), SASLprep (RFC 3454 tables B.1 and C.1.2,
  then NFKC through the library's own `UnicodeNormalization`, the same on every platform) and UTF-8 up to 127 bytes
  (5-6); when it differs, the UTF-8 the user typed is tried too (writers that skipped either step).
- **Opening:** the empty user password first (and, with no password given, the empty owner password), then
  `DocumentOptions.Password` as the user's, then the owner's. Neither: an `InvalidDataException` naming
  `DocumentOptions.Password` that says which case it is ("opens only with a password" or "neither the user nor the owner
  password"). Another handler (the public-key `Adobe.PubSec`), another revision, `/V 3` or an unknown `/CFM`: a
  `NotSupportedException` naming it, the message saying the file is encrypted. A damaged encryption dictionary (no `/O`,
  `/U` or `/P`, keys too short, a key length outside 40-128 bits): an `InvalidDataException`. Every buffer holding a
  password or a key (pooled ones returned cleared, stack ones zeroed) is cleared before it is left; the file key is held
  by the open document only and cleared on `Dispose`, with the AES contexts keyed with it. `DocumentInfo.Encrypted` says
  so.
- **Decryption, done once per object:** strings when their object is read at its offset (its key made once per object
  and only when it holds a string; containers copied only where a string changed), cached decrypted with the object;
  objects inside object streams come with their stream; not the encryption dictionary's, a cross-reference stream's
  dictionary or a signature's `/Contents` (for D12). Streams before their filters into a pooled buffer never longer than
  the stored bytes (so `MaxStreamBytes`, checked on the stored length, bounds them too); not cross-reference streams,
  metadata under `/EncryptMetadata false`, or what `Identity` leaves plain. AES data is its IV and whole blocks: a block
  cut short is left out and the padding removed only when it is one (damaged data fails later, as damage). AES contexts
  are pooled per document, the AES-256 ones keyed once. Safe from several threads.
- **Hooks in the D1 files (small, for the merge with D3):** `PdfFile.cs` (the `Security` field and property, `Unlock`,
  one line where `ReadIndirect` returns an object, four lines in `Decode` before the filters, one in `Dispose`),
  `PdfDocument.cs` (`PdfSecurity.Attach` in place of D1's refusal, which is removed; `Encrypted` in the information),
  `PdfBytes.cs` (`PdfWindow.Own`, added), `PdfFormat.cs` (its summary). The contract's docs say what a wrong password
  throws (`IDocumentFormat.Open`, `DocumentOptions.Password`); no public API changed.
- **Decisions the plan did not settle, taken here:** (1) A wrong or missing password is an `InvalidDataException` (the
  contract's documented failure for a file that cannot be opened with the options given, as a limit is), not a new
  exception type: callers that catch it keep working, and the message names the option. (2) No environment variable for
  the password (a secret in the environment; no command reads documents yet). Nothing else in D2 is a policy to set: the
  standard fixes the algorithms. (3) As readers do: a crypt filter `/CF` does not hold is `Identity`; `/Perms` is not
  checked (a reader enforces no permissions, and `/U` or `/O` already proves the key). (4) No registry of security
  handlers: the plan names none, and a whole format can be replaced through `DocumentFormats`.
- **Tests:** group `documents pdf: encryption` (3): every encrypted sample (the 21 of D2 and D11's `encrypted-aes256`)
  opened with its JSON's, its user and its owner password (44 opens; `Grüße-2024` through SASLprep), encrypted and
  opened as user or owner as expected, its cross-reference, every object (420 objects in the canonical form, the
  encryption dictionary's strings plain), every stream (to its SHA-256, the metadata under `/EncryptMetadata false`
  too), its information and pages as its JSON says, the unencrypted twin's objects (but a stream's stored `/Length` and
  revision 5's `/Extensions`), information, pages and text the same, and every object read alike from several threads;
  a wrong password, none and an empty one refused naming the option, a wrong one opening a file whose user password is
  empty, `Adobe.PubSec` refused naming it, the document format suite on the encrypted samples (with the D2 user
  password), 126 cut or changed encrypted files failing only as damaged or as a variant not read; RC4 on the published
  vectors, passwords prepared, `/V 3`, revision 7, a 129-bit key, `/CFM /AESV9` and another handler refused naming
  them, the stream limit on encrypted streams, the key cleared on dispose. `documents pdf: file` updated: the encrypted
  samples open with their password and not without, the format suite refuses those needing one naming the option.
  Run with `documents pdf: file|documents contracts|public API|abstraction inventory|every environment variable read in
  src|test runner`: first 28 tests, 7 failed, all at the samples' loader (the manifest also lists D5's later fonts,
  writer inputs, merge plan and PKI, which have no PDF or no JSON: `Value cannot be null (path2)`; the loader now takes
  the PDF samples only), then 1 (revision 5's `/Extensions` in the twin comparison, set aside), then 28 passed, 0
  failed, also after the MD5 change.
- **Timings** (this container, 4 cores, Linux, load average 14 from two other agents' builds: noisy; the owner's devices
  give the real ones): opening and reading the 22 D2 samples (every page with every part, the information) cold 176 ms
  (125 ms the first document, mostly compiling), warm median 14.2 ms a pass (646 µs a document, 9 ms of it the five
  revision 6 files). A document warm, against the unencrypted twin's 98 µs and 18.9 KiB: RC4 revision 2 157 µs and
  22.8 KiB, RC4 128 bits 209-278 µs and 23-26 KiB, AES-128 206-328 µs and 26-32 KiB, AES-256 revision 5 210 µs and
  28.7 KiB, revision 6 1.3-2.2 ms and 47-62 KiB. The revision 6 hash on its own: 0.66 ms warm (0.71 ms cold) for a user
  check, 1.14 ms for an owner check (the 48 bytes of `/U` in each round), about 9.5 KB allocated a call (the base
  library's one-shot AES context, one a round); revision 5's hash 0.6 µs. A revision 6 file costs two hashes opened by
  the empty password, three by the user's, four by the owner's: the standard's own cost, paid once at open.
- **Left:** a damaged encrypted file whose cross-reference must be rebuilt by scanning and whose catalog lies in an object
  stream does not open (the scan lists object streams before the key is known; it fails as damaged). Writing encrypted
  files is D11's.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: encryption|documents pdf: file|documents contracts|public API|abstraction inventory|every environment variable read in src|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d2.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

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

**As built (2026-11-18):**
- **The reader** (`src/Idrak.Data/Pdf/`, all internal): `PdfContentLexer` (content streams and CMaps as spans: each token
  located, numbers parsed from the bytes, strings and names decoded into the caller's buffer, dictionaries passed over
  balanced, damaged bytes never an error); `PdfContentReader`, the interpreter (`q Q cm`; `BT ET Tf Tm Td TD T* TL Tc Tw
  Tz Ts Tr`; `Tj TJ ' "`; `Do`: forms within `MaxDepth` through their `/Matrix` and `/Resources` or the caller's, images
  taking their place in the content; `BI ID EI` passed over by the image's size when it is unfiltered and known, else to
  an `EI` between whitespace followed by content-like bytes; `BMC BDC EMC` with `/ActualText` inline or in
  `/Properties`; unknown operators passed over; a page's content streams read as one); `PdfFont` (simple: Type 1,
  TrueType, Type 3 by its `/FontMatrix` and `/FontBBox`; composite: Type0 with Identity-H, Identity-V or an embedded
  CMap and the one it uses, `/W /DW /W2 /DW2`, `/CIDToGIDMap`); `PdfCMap` (code spaces of one to four bytes, `bfchar`,
  `bfrange` by a start or an array, `cidchar`, `cidrange`, `notdef`, `/WMode`, kept as ranges); `PdfFontProgram` (the
  embedded TrueType or OpenType program read through `FontFormats` at run time: the face, the best Unicode `cmap`
  reversed to each glyph's lowest code point, the (3,0) and (1,0) subtables); `PdfFontTables` over
  `PdfFontTables.Data.cs`, made by `tools/documents/make_pdf_tables.py` from the venv's pins: the Adobe Glyph List and
  the ITC Zapf Dingbats list (fontTools.agl), the four base encodings (pdfminer.six's latin_enc), the Symbol and
  ZapfDingbats built-in encodings (ISO 32000-2 Annex D.5 and D.6, checked code by code against pdfium: no difference),
  the standard 14 fonts' metrics (pdfminer.six's copy of Adobe's Core 14 AFMs). Stored as the repository's other tables
  are: spans of constants in the assembly's data, names sorted for a binary search (nothing parsed or allocated when the
  reader starts, about 60 KB); the licences quoted in the file and in NOTICE; reruns give the same bytes.
- **Text,** in the plan's order: ToUnicode (a simple font's codes looked up as one byte, then as two, as some writers
  write them); the encoding's glyph names by the Adobe Glyph List's rules (suffix dropped, components joined, the list,
  `uniXXXX...`, `uXXXX[XX]`, the Zapf Dingbats list for that font); the embedded program's own `cmap` (composite: CID to
  glyph through `/CIDToGIDMap`, then the glyph's lowest code point; simple TrueType: code to glyph by 9.6.5.4); else
  U+FFFD, counted in an `unmapped-glyphs` issue naming the fonts. A Type 1 program's own encoding is read from its clear
  text. Render modes 3 and 7 are `Hidden`, 1 and 5 `Stroke`, 2 and 6 `FillStroke`.
- **Widths and boxes:** `/Widths` from `/FirstChar`, past them the standard 14's AFM width by the glyph name's character,
  else `/MissingWidth`; composite `/W /DW`, vertical `/W2 /DW2`. Each glyph is placed through the text matrix, the CTM
  and the page's display map (crop box, `/Rotate`, `/UserUnit`): its advance along the baseline by the font's descent to
  ascent (descriptor, the standard 14's AFM, Type 3's `/FontBBox`), a vertical glyph its width centred by its vertical
  advance; a run's size is the size times the text matrix and CTM's vertical unit; upright runs give `Baseline`.
- **Lines and text** through `DocumentTextLayouts` (the layout the options name): the reader gives the layout its runs as
  pieces in the frame of the page's main text direction (a turned page, turned text and vertical columns read as lines),
  split where a `TJ` offset moves the pen (the layout's `WordGap` decides the space) and around a glyph that stands for
  several characters (a logical piece, so a ligature keeps its characters' order in right-to-left text: D0's layout
  reverses a run by text elements, which would turn DejaVu's lam-alef round); an `/ActualText` span is one logical piece.
  The lines' runs are given back as the page's runs (a layout that makes a line of each piece, as `content` does, has its
  pieces' lines joined back: a line a run).
- **Glyphs and faces** when `DocumentParts.Glyphs` is asked: each run's glyph ids (through `/CIDToGIDMap`, or a simple
  TrueType font's cmap route; the code for a font with no program read), origins and advances on the page; `Face` is the
  embedded TrueType or OpenType program read through `FontFormats` (null for the standard 14, Type 1, a bare CFF and
  Type 3: D7's Type 1 reader adds those). Programs are read only when needed (a glyph with no other text, or glyphs
  asked); fonts are kept in the document's cache by object, read once a page otherwise.
- **Issues:** `unmapped-glyphs`; `missing-font` (a font the page lacks, or none set: read as Helvetica in the standard
  encoding); `not-supported` (a predefined CMap other than Identity, each tied to one script: codes read as two bytes,
  their text from ToUnicode alone); `damaged-stream` (contents, a form or a font that cannot be read).
- **Decisions the plan did not settle, taken here:** (1) a run's `Order` counts runs, images (`Do` of an image, inline
  images) and painted paths and shadings (`S s f F f* B B* b b* sh`, not `n`), as the D4 and D7 samples count them, so a
  run's order does not change when D4 and D7 add their parts. (2) Text set past the crop box or the page keeps its box
  there, as the samples' JSON says (`vector-turned-pages`, `split-source`); D1's format-suite run leaves those two out,
  since the suite wants every run on the page. (3) A font the page lacks is read as Helvetica and counted, not left out.
  (4) `IDocumentTextLayout` and `DocumentTextLayouts`, now used by Idrak.Data alone, are listed in
  `tests/Idrak.Tests/data/shared-contracts.txt` (decision 1), as D1 listed the other document contracts; the inventory is
  regenerated through the runner. (5) D1's sample loader skips the manifest's entries with no document (the D6 inputs,
  the D8 dumps, the test PKI), which the full set of 150 samples has; D1's group passes on the full set. (6) `q` nested
  past 16 times `MaxDepth` is counted, not stored (its `Q` matched), so a stream of `q`s cannot grow memory.
- **Generic sample text:** truetype-simple's second line "Invoice total: 1,250.00 EUR (paid)" is now "Storage total:
  1,250.00 GB (free)" (the same feature: a simple TrueType font with a ToUnicode map, capitals, digits and punctuation).
  `make_fixtures.py --compare` against a run of the old generator: 767 files each, only `d3/truetype-simple.json`, `.pdf`,
  `.text.json` and `.p1.png` differ, and the manifest only in those entries.
- **Tests:** group `documents pdf: text` (5): the timing below; every D3 sample page (20 samples, 31 pages, 109 runs)
  against its JSON (text, box within 0.005 point, size within 0.001, font, hidden, order, `/ActualText`), its lines with
  their direction and its text; `hostile-form-recursion` tripping `MaxDepth` (the limit, its option and the file named)
  and every page with its glyphs read in parallel the same as in order; the options (`IncludeHidden`, `WordGap` 0.2 and
  0.5 on TJ gaps, the `content` layout, glyphs and faces); hand-made bytes (the glyph-name rules, standard font names,
  CMaps, a page with a font it lacks, an inline image holding " EIQ", a `/Properties` span, an unknown operator, a path
  in the order). First run: 41 passed, 11 failed (D1's sample loader on the full set: 9; the two layout contracts now
  used by Idrak.Data alone: 2; a glyph-name expectation of the test); second: 48 passed, 4 failed (the inventory's line,
  the two samples with text past their page in the suite, the test's run-count floor, a Zapf Dingbats width the test
  wrote from memory); after the fixes and the inventory regenerated: `documents pdf: text|documents pdf: file|document
  layouts|documents contracts|outside plug-in|public API|abstraction inventory|every environment variable read in
  src|hidden boxing` 52 passed, 0 failed, and `conformance kit:|override loop:|test runner|generic components: reading
  order|the environment table names real commands` 26 passed.
- **Timings** (this container, 4 cores, another agent building beside it; the owner's devices give the real ones): the
  31 pages of the 20 D3 samples opened from memory and their text read: cold 166-178 ms (the first call, mostly
  compiling), warm median 15-19 ms a pass (486-601 µs a page, each document opened and its fonts read every pass), 65 KiB
  allocated a page; the same pages of documents kept open (fonts read once): 6.1-6.4 ms a pass (196-205 µs a page), 32
  KiB a page. One change measured before and after: the page's glyphs in a pooled buffer and simple fonts' one-character
  texts in a table (94.7 to 65.2 and 48.6 to 31.8 KiB a page, the text the same).
- **Left for later phases:** CFF programs' built-in encodings (StandardEncoding stands in), runs' colours and clips,
  vertical position vectors and faces for Type 1, CFF and Type 3 (D7); images (D4). Most of the 32 KiB a page is D0's
  `lines` layout's own (strings, groups, the reading order), a piece of its own.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: text|documents pdf: file|document layouts|documents contracts|outside plug-in|override loop:|conformance kit:|public API|abstraction inventory|every environment variable read in src|hidden boxing|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d3.txt
  $env:IDRAK_FILTER = "documents pdf: text: reading every D3 sample page"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d3-timing.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```
- **Lines layout memory (2026-11-18):** `lines` allocated, a page: the bidi order's whole UAX #9 twice a line (the
  candidate of `FromColumns` and its check) even for a line it leaves as it is, sort delegates and closures, the groups'
  lists, a sorted copy and LINQ a line for its direction's text, `StringBuilder` chunks, a reversed string a right-to-left
  run. Now its keys are sorted as a span of structs from the shared pool, a line is a slice of them, one pooled buffer
  holds the columns and the direction texts, the page text is made once; `bidi` and `runs` give a line they leave as it
  is back as the same string (`bidi`: no explicit control or Arabic number, no R or AL in a left-to-right line, no L or
  European number in a right-to-left one; `runs`: nothing against the line, no lone surrogate). It allocates what it gives
  back and the texts it asks the order about (a test in `document layouts` holds it there). The same results: the 41 D3
  and D4 sample pages under seven options and 4,000 random pages arranged directly give byte-identical lines and text,
  20,000 random texts the same answers from both orders. Before and after, interleaved: the layout alone, warm, 19,528
  to 1,596 B and 11.2 to 1.7 µs a page, its first pass 25.0 to 2.6 KiB a page (18-20 to 16-17 ms, compiling); a page
  read from documents kept open 31.9 to 9.4 KiB (5.6-5.9 to 2.8-3.0 ms a pass), opened each pass 65.1 to 42.6 KiB
  (13.7-14.2 to 10.1-11.2 ms). Left: a mixed line still runs the whole UAX #9 (4.6-7.6 KB a `FromColumns`, two thirds in
  `Bidi.Levels` and `Bidi.Reorder`), a pooled path through them a piece of its own; the rest of a kept-open page's 9.4
  KiB is the reader's own (its main direction's LINQ and dictionary, the page's font dictionary, operands, the pieces).
  Bidi pooled, 2026-11-19: `Bidi.Levels`, `VisualOrder` and `Reorder` have span forms that write into the caller's
  buffers, their working space (isolate matches, level runs, each isolating run sequence, bracket pairs, the bracket and
  directional status stacks) from the shared pools, no list, closure or LINQ left; the array forms and `Resolve` wrap
  them and allocate what they return. `bidi`, `TextLayout` (L1 alone, in place) and the CLI's `VisualText` use the span
  forms: a mixed line allocates the string it gives back. The same results: both conformance files through both forms,
  20,000 random mixed texts against the old algorithm kept in the test ("bidi:"), equal output hashes of the old and new
  builds on 20,000 more, the 41 D3 and D4 pages' lines and text under seven options byte-identical; a `document layouts`
  test holds a page of mixed lines to what it gives back. Before and after, interleaved, warm: `FromColumns` on a mixed
  line of 20, 80 and 300 characters 4,632, 12,312 and 41,064 to 64, 184 and 624 bytes, 3.1, 8.9 and 27.2 to 2.1, 6.6 and
  21.4 µs; `ToColumns` 2,136-18,672 to 64-624 bytes, 1.4-12.0 to 1.0-10.4 µs; `Levels` and `Reorder` 1,504-13,000 to
  200-1,880 bytes as arrays (their results), 0 as spans. Cold (a process's first line, compiling) the same, 6-7 ms, once
  the two stacks came from the pool: `stackalloc` had the JIT compile two methods fully optimized at the first call (2-3
  ms more). A D3 page 44.2 to 43.2 KiB opened each pass, 10.8 to 9.8 kept open, the time the same within the noise of
  another build running (this container's CPU).
- **Reader memory (2026-11-19):** a page of a document kept open allocated, beyond what it gives back: the content
  reader made again each page (its 13 lists, 4 dictionaries, a set, 32 operands: 2,584 B before any content) and its
  lists grown again (segments, runs, tags, fonts), a font label string each `Tf`, the content stream's over-limit message
  built on every decode, the issues' dictionary and list, the main direction's dictionary and LINQ, a second
  `DocumentTextRun` and a copy of its text for each run the layout is given, boxed enumerators over the layout's lines, a
  parse of each inline `BDC` dictionary for its `/MCID` and a closure for the page's structure. Now the document keeps one
  reader between pages (one thread takes it at a time; a page read beside another makes its own): its working buffers
  are rented from the shared pools for a page and given back at its end (references cleared), its caches emptied; the
  main direction comes from sorted pooled keys; a run that is one piece already is given to the layout as itself (its
  text, box, baseline, size, order, tag and direction equal: the layouts read no other), a whole run's piece shares its
  text; the font label is the resources' key's text; the over-limit message is made when thrown; issues, clips, groups
  and soft masks are made with their first; a flat `/MCID` dictionary (names, numbers, true, false, null) is read by the
  lexer, any other by the parser as before. Numbers were bytes already (`PdfParser.TryNumber`). What is left of the
  reader's own: the content stream's inflater and buffer objects (584 B a page; .NET 10 has no span inflater) and the
  list the layout is given; the layout's lines are its own (a mixed line's UAX #9 still the most). Warm, a page, start to
  now (B): `standard14` (16 runs, 14 fonts) with `lines` 26,200 to 11,328 (8,984 given back), with no layout (glyphs)
  23,856 to 12,928 (12,344); `rtl` 8,856 to 2,920; `vertical` 6,848 to 1,728; `type0-identity` 10,520 to 4,976 (a
  ligature makes pieces); `scan-ocr` (D4's scan with its text layer) 14,128 to 4,752; `searchable-scan` 10,304 to 2,952;
  `tagged` 14,576 to 4,592; `mixed-direction` 50,797 to 39,712 (the bidi order's own); with no layout every page 584 B
  past what it gives back (5,120 to 11,480 before). No D3 or D4 sample holds CJK text: the Type 0 pages stand in. The
  same results: the 123 sample files' pages through every layout and reading order, as text and with every part, glyphs
  alone, images and paths, hidden runs left out (16 ways, 2,864 page reads) hash the same before and after, and each
  build the same twice (the limits fixed: their defaults are measured from free memory, and two hostile samples quote
  them). Interleaved, three processes each (this container, other agents building): kept open 11.5 to 5.0 KiB a page,
  warm 2.28-6.27 to 1.93-2.57 ms a pass, its first pass 11.9-23.0 to 6.3-17.4 ms; opened each pass 45.1 to 39.4 KiB a
  page, warm 9.6-13.2 to 9.3-10.3 ms, cold 182-244 to 198-245 ms (compiling, the same). A test in `documents pdf: text`
  holds five pages (many fonts, right to left, two Type 0, TrueType) to what they give back, the layout's own and the
  reader's own within 1,024 B + 8 a run (seen: 688-832). Tests (`documents pdf: text|file|images|encryption|vector|forms|
  annotations|edit|write|signatures`, `documents drawers|document layouts|documents contracts|outside plug-in|override
  loop:|conformance kit:|cli: data redact|public API|abstraction inventory|hidden boxing|test runner`): first 131 passed,
  1 failed (the new test on `type0-identity`, whose ligature splits its runs: two other Type 0 pages took its place);
  second 131 passed, 1 failed (`type0-no-cmap`'s reader's own 1,256 B: the issues' dictionary and the unmapped fonts'
  sorted set, now a scanned list and one name); third 132 passed, 0 failed.

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

**As built (2026-11-18):**
- **Images** (`src/Idrak.Data/Pdf/PdfImages.cs`, internal): D3's interpreter hands each `Do` of an image XObject (at any
  depth of forms) and each inline image (its entries between `BI` and `ID`, keys spelled out, and its data, copied) to
  the page's images when `DocumentParts.Images` is asked, with the place D3 already counted (runs, images and paths share
  one order). Each `DocumentImage` gets its `Matrix` (the unit square from its top-left corner through the CTM and the
  page's display map: turned, slanted, flipped, mirrored, on a turned page, inside forms), its `Box` (the bounds),
  `PixelWidth`, `PixelHeight` and `Encoding` (`raw`, or the first image filter of its chain: `jpeg`, `jpeg2000`,
  `ccitt`, `jbig2`); an XObject drawn again is read once a page. Nothing is decoded until asked: `Bytes()` is the data
  through every filter before the image encoding (a raw image's samples, a JPEG's file; decrypted), `Decode()` the
  pixels, `Mask()` the mask, each bounded by `MaxStreamBytes` (the pixels' bytes checked before anything is decoded).
- **Raw pixels:** samples of 1, 2, 4, 8 and 16 bits (rows padded to bytes) through tables made once from `/Decode` (16-bit
  samples mapped one at a time); `DeviceGray`, `CalGray`, ICC-based with one component to a grey plane, `DeviceRGB`,
  `CalRGB`, ICC-based with three to RGB planes (profiles and calibrations not applied), 16-bit grey and RGB to values
  (v / 65535); `DeviceCMYK` and ICC-based with four through the library's JPEG rule ((255 - K) - X (255 - K) / 255, its
  MULDIV255); `Indexed` (over grey, RGB, CMYK or Lab; a string or stream table; the index rounded half to even and
  clamped) through its colours; `Lab` through XYZ (its white point scaled to D65) to sRGB; image masks as grey (0 where
  they paint). Rounding as the samples' reference: half to even, the same arithmetic in the same order. Rows walked in
  order, chunks of rows spread over the cores (`HostParallel`, so `ComputeResources` bounds it) for images of 128 Ki
  pixels or more; 8-bit grey with a plain `/Decode` is copied. Data cut short reads as zeros past its end.
- **Encoded images:** `DCTDecode` through `ImageCodecs` "jpeg" (the codec's own decode: the colour space and `/Decode`
  not applied again, as the samples' README says for Adobe CMYK), the JPEG's header read first for its size and the
  limit; `JPXDecode`, `CCITTFaxDecode` and `JBIG2Decode` through a codec registered under `jpeg2000`, `ccitt` or `jbig2`,
  else a `NotRegisteredException` (a `NotSupportedException`) naming the encoding and `ImageCodecs.Register`.
- **Masks** (`Mask()`, the contract's doc now says it): coverage, 255 where the image shows and 0 where it does not, at
  the mask's own size: a soft mask's levels (decoded as an image, its own `/Decode` and encoding), an explicit `/Mask`
  and an image mask 255 where they paint (their `Decode()` grey turned over), a colour key 0 where every component's
  sample lies in its range.
- **Embedded files** (`src/Idrak.Data/Pdf/PdfAttachments.cs`): the catalog's `/Names /EmbeddedFiles` tree (kids and
  leaves in order, a node met again passed over, `MaxDepth` its limit), then the pages' file-attachment annotations, a
  file listed once; `Name` from `/UF`, `/F` (or the platform entries), else the tree's key, `MediaType` the stream's
  `/Subtype`, `Size` its `/Params /Size` (else -1), `Description` `/Desc` (else the annotation's `/Contents`); `Read()`
  decodes and decrypts within `MaxStreamBytes` into a read-only seekable stream that owns the pooled buffer (no second
  copy, returned on dispose). Listed once a document, on the first call; a damaged entry passed over.
- **Issues:** `image-encoding` once a page for each encoding no registered codec reads (counting its images);
  `not-supported` for images in `Separation`, `DeviceN` or `Pattern` (and `Indexed` over them), which `Decode()` refuses
  naming the space (their tint transforms come with D7's functions); `damaged-stream` for an image left out (no readable
  size: its place in the order kept) or one that cannot be decoded (an unknown colour space, bits other than 1-16).
- **Hooks in the files of D1-D3 (small, local):** `PdfContent.cs` (the `_images` field, `ReadImages()` and `Images`, the
  image `Do` and the inline image's three ends passing what they found, `InlineImage` given the resources),
  `PdfDocument.cs` (images into the page, `Attachments`), `PdfFile.cs` (`Filters` internal, for inline images'
  filters). `DocumentPages.cs` (Idrak.Abstraction): the docs of `DocumentImage.Mask` and its constructor's `mask` only.
  No public API changed; the API dumps and the inventory pass as they are.
- **Decisions the plan did not settle, taken here:** (1) `Mask()` is coverage for every kind of mask (above), so a
  drawer needs no kind to use it; an image mask's own `Mask()` is its coverage. (2) An encoded image whose own size is
  not its dictionary's is refused as damaged naming both, so `Decode()` always gives `PixelWidth` x `PixelHeight` (the
  format suite's contract). (3) 16-bit grey and RGB keep their 16 bits as values, the other 16-bit spaces go through
  8-bit levels. (4) An attachment's `Name` is the last part of the path the file gives (control characters replaced),
  so it never leads out of a folder an app writes it to. (5) CCITT's and JBIG2's parameters (`/DecodeParms`,
  `/JBIG2Globals`) are not handed to a codec: `IImageCodec.Decode` takes the bytes alone (a contract question for when an
  app brings such a codec). (6) Reading images alone still runs the text interpreter, so the orders are the same as
  with text.
- **Tests:** group `documents pdf: images` (5): the timing below; every image of every sample whose JSON lists images (D4's
  nine, D1's predictors, D6's scans, D7's and D10's: 75 images, 9 with masks, 6 inline, 12 JPEGs) in order with its
  encoding, size, box and matrix (within 0.01 point), its bytes' SHA-256 (samples or JPEG file), its pixels (the
  interleaved samples' SHA-256, 16 bits big-endian where the depth is 16; Lab within its tolerance 2 against the PNG; a
  JPEG also against the jpeg codec's decode of its bytes) and its mask (soft masks exactly, explicit masks and stencils
  turned over, the colour key against the samples), no image issue; the scan-like pages (`scan-ocr`, D6's
  `searchable-scan`) read with text and images together, their JPEG over the page and their hidden text as D3 checks
  it; every image page read in parallel the same as in order (boxes, matrices, pixel and mask hashes, issues); the
  embedded files of `attachments` and the D10 annotation samples (names, media types, sizes, descriptions, bytes against
  the files beside them, read again, seeking), none in the other samples, refused after the document is closed;
  hand-made files: an RC4 (revision 2) encrypted file's XObject and inline image, an image in a form with its matrix,
  JPEG 2000, CCITT (inline, abbreviated) and JBIG2 (after ASCIIHex) as stored with their issues and refusals, then a
  stand-in `jpeg2000` codec registered and unregistered, the pixel limit named on `Decode()` and `Mask()`, damaged and
  later-phase colour spaces, data cut short, a JPEG whose size is not its dictionary's. Run with `documents pdf:
  images|documents pdf: text|documents pdf: file|documents pdf: encryption|documents contracts|public API|abstraction
  inventory|hidden boxing|test runner`: 38 passed, 0 failed at the first run; after the 16-bit change (one rebuild) 38
  passed, 0 failed. D1's format suite now decodes every sample's images and reads its attachments too.
- **Timings** (this container, 4 cores, load about 5 from other agents' builds; the owner's devices give the real ones):
  the 59 images of the D4-phase samples (2.92 megapixels, most of them the three scan pages): cold 45-55 ms, warm median
  8.4 ms a pass (2.9 ms a megapixel), 1.05 MiB allocated a megapixel (about the decoded pixels themselves); the scan-like
  page's image (826 x 1169 grey JPEG): cold 47-63 ms (the first JPEG decode, mostly compiling), warm median 1.8 ms
  (1.9 ms a megapixel), 0.97 MiB a megapixel (its one byte a pixel), and its page read with text and image listed
  0.37 ms warm. Small raw images 5-33 µs and 1.5-3.7 KiB each (half of it D1's Flate filter), Lab 96 µs for 384 pixels
  (a power a pixel); D6's raw page images 8.2 ms a megapixel, its scanner JPEG pages 2.7. One change measured before and
  after: 16-bit samples mapped one at a time instead of through 65,536-entry tables (16-bit grey 357 µs and 258 KiB to
  9 µs and 1.8 KiB a decode, RGB 934 µs and 771 KiB to 14 µs and 2.7 KiB; the D4 pass 10.9 to 8.4 ms and 1.74 to
  1.05 MiB a megapixel), the pixels the same (the expected hashes).
- **Left:** CCITT's and JBIG2's parameters for a codec (decision 5); images-only reading without the text interpreter
  (decision 6); Lab pixels a power at a time (a table or cache if real files need it); images in Type 3 glyphs,
  patterns and annotation appearances (D7, D10); an image's clip and fill colour for a stencil (D7).
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: images|documents pdf: text|documents pdf: file|documents pdf: encryption|documents contracts|public API|abstraction inventory|hidden boxing|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d4.txt
  $env:IDRAK_FILTER = "documents pdf: images: decoding every D4 sample image"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d4-timing.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

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

**As built, third piece (2026-11-19): the real-files group and the docs.**
- **The group `documents real:`** (`tests/Idrak.Tests.Library/DocumentRealTests.cs`, three tests, CPU only; D13's third
  bullet, the owner's files written back, edited, drawn and timed, is this group): every `*.pdf` under the folder
  `IDRAK_DOCUMENTS_REAL` names (subfolders too, in ordinal order of their paths; nothing written into it). Without the
  variable each test is a SKIP naming it. **Read:** opened (version, encryption, repaired), every page's text through
  `lines` and `tagged`, its images (each decoded, with its mask), paths, clips, shadings and annotations, the fields,
  bookmarks, named destinations, page labels, attachments (each read to its end) and signatures (each checked with
  nothing trusted and nothing fetched: the digest, signature and changes steps are what it proves, "intact" when the
  first two pass), all within the default limits (C8); the reference's differences listed. **Drawn:** every page at the
  drawer's default DPI (150, `IDRAK_DOCUMENT_DPI`) with text, images, paths and annotations, its drawing issues counted.
  **Written back:** as a rewrite and as an incremental update with the information's title changed, each written again
  to a file and read back: the same page count, sizes, turns, text, run and image counts, the title written; the update
  also keeps the source's bytes as its prefix and every signature's digest and signature results. **Timed:** each step
  (open, text, tagged, parts, images, navigation, signatures, draw, rewrite, incremental) cold and warm apart (rule 76),
  with a page's time and managed memory. A file fails only for a crash, a page or part that cannot be read and gives no
  reason, or a written-back file that does not read the same; the message names the file and step, and the run goes on
  to the other files first. **The summary,** `test-output/documents/real/summary.json` and `summary.txt` (written again
  every 5 seconds while a test runs, and at its end): per file its pages, bytes, version, counts (runs, lines,
  characters, images and pixels, paths, annotations by kind, fields by kind, bookmarks, attachments, signatures with
  each step's result), issues by kind with their details and pages, drawing issues, each step's times, the written-back
  sizes and checks, the reference's differences and every reported failure; then the totals.
- **The reference text:** `tools/documents/reference_text.py FOLDER [-o OUT] [--beside] [--password PW] [--timeout S]
  [--memory-mb M]`: pdfminer.six (the pinned reader the samples' reference text comes from, `docfixtures.reference`'s
  `pdfminer_text`) in a child process per file with a time limit (120 s) and on Linux a memory limit (4 GiB), each page's
  text separated by a form feed, into `NAME.reference.txt` at the PDF's path under `OUT` (default
  `test-output/documents/real-reference`, which the group reads) or beside the PDF; progress a file a line with the rate
  and the time left, and `reference.json`. The group compares each page with whitespace collapsed: the same, the same
  but for spaces, the same words in another order, the same words with some as drawn (a word's text elements reversed:
  a right-to-left word as pdfminer.six gives it), or words that differ (the reference's missing, Idrak's extra, the share
  in common). Listed, never failed.
- **The docs:** the root README's `Idrak.Data` line names what documents can do (PDF read, written, edited, signed and
  checked, drawn; forms, annotations, bookmarks, redaction), `Idrak.Imaging`'s names Type 1, TrueType written and pages
  drawn, `Idrak.Data.Plugins` stays the place for Office, to come; `src/Idrak.Data/README.md` gains "Documents": open
  and read, write (a page of an image, a path and text set through the text setter), edit and save (incremental,
  rewrite, merge, overlay, encryption), forms (read, fill, add, flatten), annotations and redaction (with `idrak data
  redact`), sign and check, draw a page, the options and their variables, and what is not supported (from every phase's
  "Left" and "Not in this plan"); `src/Idrak.Imaging/README.md` gains "Fonts written" (TrueType whole and subset, the
  options, what is not written, legacy fonts converted once and kept by `FontCache`, `IDRAK_FONT_CACHE`) and
  "Documents" (the "coverage" drawer and every setting of D7's drawer settings with its variable, the "shaped" text
  setter, `DocumentShapes`), the "cff" and "type1" formats in its font row; `src/Idrak.Imaging.Plugins/README.md`
  already had WOFF; `tools/documents/README.md` gains "Real files". The bench is D13's.
- **Decisions the plan did not settle, taken here:** (1) The group's variables are the tests' own, read in the tests
  as `IDRAK_DOCUMENTS_SAMPLES` is, not in the tool's table: `IDRAK_DOCUMENTS_REAL`, `IDRAK_DOCUMENTS_REAL_REFERENCE`
  (default `test-output/documents/real-reference` when there; a `NAME.reference.txt` beside the PDF comes first),
  `IDRAK_DOCUMENTS_REAL_SAVE` (`pages`, `files`, `all`: the drawn pages as PNG under `real/drawn/` and the written-back
  files under `real/written/`, else nothing kept) and `IDRAK_DOCUMENTS_REAL_WARM` (3 warm passes, 0 to 100); a bad
  value fails naming its variable. (2) A password is never a variable (D2's decision 2): the first line of
  `NAME.password.txt` beside the PDF; a file that needs one and has none (or a wrong one) is listed as skipped. (3) A
  crash is any exception but the contract's `InvalidDataException` and `NotSupportedException` (an aggregate of them
  too); those are listed with their message, and one with no message fails. (4) Cold is the file's first pass of each
  step in the run (the first file's also compiles the code), warm the median of the warm passes; each pass opens the
  document anew outside its clock, so a warm pass reuses no document's cache; memory is
  `GC.GetTotalAllocatedBytes(precise: true)` (drawing spreads over the cores). The timed write-back passes write to a
  stream that only counts, and the check writes once more to a file. (5) The plan's run pattern named an output
  variable (`IDRAK_DOCUMENTS_OUT`): not built; the outputs are `test-output/documents/real/`, as every other document
  group's are fixed under `test-output/documents/`, and the run's own `drawn/` and `written/` start empty. (6) The
  runner stops a test after 300 seconds; a large folder needs `IDRAK_TIMEOUT` (the command below gives two hours).
- **Tests:** one build (no error, no warning); the filter `documents real:|documents contracts|public API|abstraction
  inventory|every environment variable read in src|the environment table names real commands|hidden boxing|test
  runner` on the CPU without `IDRAK_DOCUMENTS_REAL`: 22 passed, 0 failed, 3 skipped (the group, naming the variable);
  with it on a scratch folder of 20 samples (`IDRAK_DOCUMENTS_REAL_SAVE=all`): 25 passed, 0 failed. Reading that
  summary showed four things to tidy, none a failure: counts written "1 pages", an empty totals list printed as nothing,
  an empty reference page counted as one empty word, and the right-to-left page listed as words that differ (now "words
  as drawn"); rebuilt once, the same filter on the folder: 25 passed, 0 failed.
- **The scratch run** (20 samples in five folders: text, right-to-left and tagged letters, two scans, a form,
  annotations, attachments, two vector pages, a repaired file, a 12-page filters file, two signed files, three
  encrypted ones (one opened with the empty password, one with its `NAME.password.txt`, one without), three hostile
  ones; this container, 4 cores, another agent building beside it): 18 opened, 1 skipped (its password not given), 1
  not opened (the page-tree cycle, named); 33 pages, 112 runs (11 hidden), 3 images (2.9 megapixels decoded), 33
  paths, 39 annotations, 15 fields, 5 attachments, 3 signatures, all 3 intact (the chain not trusted, as nothing is);
  10 reported failures, each with its reason (the reference cycle's and the form recursion's page at every step); every
  rewrite and update read the same (18 of 18 each), the updates keeping the source as their prefix and both signed
  files' signatures. A page, cold / warm: open 1.88 / 0.11 ms, text 2.82 / 0.26, tagged 0.45 / 0.25, parts 1.59 / 0.33,
  images 1.22 / 0.38, navigation 0.94 / 0.06, signatures 1.99 / 0.11, draw 8.49 / 5.09 (150 dpi, no fonts folder: 114
  runs in fonts not embedded left out and counted), rewrite 1.33 / 0.24, incremental 0.53 / 0.02. The reference
  (`reference_text.py` on the folder: 18 files with text in 32 s; the file with no password refused, the reference
  cycle past its 30-second limit): 17 files, 32 pages: 27 the same, 1 in another order (the tagged page: pdfminer.six
  groups its text boxes otherwise than `lines` orders the lines), 1 with words as drawn (the right-to-left page), 2 in words (`filters` page 2, a Flate stream without its zlib header pdfminer.six reads nothing
  from; `vector-fonts`, whose Type 1 font's own encoding pdfminer.six gives as `(cid:N)`), 1 not compared (the form
  recursion's page, which Idrak reports).
- **Left:** a run that keeps nothing leaves `real/written/`'s empty folders behind; the summary's times are this
  container's (the owner's devices give the real ones).
- **The owner's run** (held with the others until asked; `D:\Documents\pdf-samples` stands for the owner's folder):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  python -I tools\documents\reference_text.py D:\Documents\pdf-samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents real:|test runner"; $env:IDRAK_DOCUMENTS_REAL = "D:\Documents\pdf-samples"; $env:IDRAK_TIMEOUT = "7200"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-real.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER, Env:IDRAK_DOCUMENTS_REAL, Env:IDRAK_TIMEOUT
  Get-Content test-output\documents\real\summary.txt
  ```
  The summary is `test-output\documents\real\summary.txt` (and `summary.json` beside it); the reference text is in
  `test-output\documents\real-reference\`. `$env:IDRAK_DOCUMENTS_REAL_SAVE = "pages"` before the run also keeps every
  drawn page as PNG under `test-output\documents\real\drawn\`.

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

**As built (2026-11-18):**
- **The writer** (`src/Idrak.Data/Pdf/PdfWriter*.cs`, all internal; `PdfWriter` registered as "pdf" in `DocumentWriters`
  through Idrak.Data's `LibraryRegistrations`, one line): `PdfSyntax` writes on pooled spans (D1's `PdfBuffer`): floats as
  the shortest text that reads back as them, computed values to fixed decimals, never an exponent; names with `#xx`;
  literal strings escaped; text strings ASCII, or UTF-16BE after its mark; any value of D1's object model (`PdfValue`)
  written back as it reads (for D11's editor). `PdfFileWriter`: the header with a binary comment, each object's offset
  kept (eight bytes an object), small objects packed a hundred to a compressed object stream, streams deflated through
  `ZLibStream` at `Compression` (0 stores them), a cross-reference stream (a classic table for `Version` 1.0-1.4), the
  trailer with `/Info` and `/ID` only when there are some; bytes reach the destination in 64 KiB blocks and when each page
  ends (it need not seek). `PdfDocumentBuilder`: objects 1 and 2 kept for the catalog and the page tree, so each page
  object names its parent when written; a page is written when the next starts; the fonts, the page tree (one node), the
  catalog (attached files in a name tree sorted by their bytes), the information and the cross-reference at `Finish`.
  Disposed without it: the page in hand, the page tree and the catalog (standing on their own), a trailer naming the
  catalog, no cross-reference: the reader rebuilds it and says `Repaired` (the fonts, written at the end, are missing:
  counted as `missing-font`). Forms, annotations, links, bookmarks, page labels, copied pages, encryption, signing and
  editing are `NotSupportedException`s naming D9, D10, D11 or D12.
- **Pages** (`PdfPageBuilder`): the content starts by turning the user space over (`1 0 0 -1 0 H cm`), so every point is
  written as the caller gave it and reads back exactly (the reader's matrices compose to the identity); the graphics state
  is tracked so only changes are written (colours `g rg k` / `G RG K`, opacity through an `ExtGState` per pair, `w J j M
  d`, `Tf`, `Tr`), saved and restored with `q`/`Q` (a Restore without a Save refused; Saves left open closed with the
  page). Paths: `m l h`, a rectangle as `re`, cubics as `c`, or `v`/`y` when a control is an end, a quadratic as the cubic
  it raises to; `f f* S B B*`; clips `W n`/`W* n`. A path not starting with a move, a negative width or an all-zero dash is
  refused before anything is written.
- **Images** (`PdfWriterImages`): values to 8-bit levels by the image encoders' rounding (an 8-bit image's planes used as
  they are); `Flate`: the PNG predictor (each row's filter by libpng's smallest sum of signed residuals; kept when it
  lowers the plain rows' sum by a fifth, since on noise the best of five filters is a little smaller by chance), then
  Flate; rows filtered and measured in parallel, the same bytes on any number of threads; `Jpeg`: `DCTDecode`, the bytes
  of `ImageEncoders.Get("jpeg")` at `JpegQuality`. `Auto` at a call takes the options' encoding, and `Auto` there
  measures: the most levels a channel uses (at most `LineArtLevels`: exact), then the share of neighbouring pixel pairs
  (right and below, the largest channel difference) differing by 1 to 16 levels: at or above `SmoothSteps` JPEG, below it
  exact; with no "jpeg" encoder registered Auto stores the image exact. **The jpeg encoder**
  (`src/Idrak.Imaging/JpegImageEncoder.cs`, one line in `LibraryRegistrations.cs`): `JpegEncoder` at
  `ImageEncodeOptions.Quality` (75 unless given), so `ImageEncoders.Save` writes `.jpg` too.
- **Text** (`PdfWriterFont`): a face is one `Type0` font a document (`Identity-H` over a `CIDFontType2` with `CIDToGIDMap
  /Identity`; glyph ids as two bytes); a run is a `TJ` a baseline from `1 0 0 -1 x y Tm`, each glyph moved to its place by
  offsets computed against what a reader adds up (the widths and the size as written; an offset under a quarter of a
  thousandth of a point left out: rounding, not a kern, and never added up); `/W` the advances in thousandths to three
  decimals; the descriptor's box, italic angle and cap height from the face's `head`, `post` and `OS/2`; the program
  through `FontWriters` "truetype" (`Subset`: the glyphs used, ids kept, the others empty, no layout tables; `Whole`:
  every glyph with its layout tables and hinting) as `FontFile2` with `/Length1`, a subset's tag six letters from a hash
  of its name and glyphs. **ToUnicode** from the runs: each glyph the characters it first stood for, found by aligning a
  run's glyphs with its text through the face's own character map (in order or reversed, whichever matches more; with no
  match at all, the text's direction from `ReadingOrders` "bidi"); the characters between two matched glyphs go to the
  glyphs between them (several to one: a ligature). A run whose glyphs do not read back as its text through that map
  (right to left, a ligature's characters, a character drawn by two glyphs, a glyph that stood for other text before) is
  written inside `/Span <</ActualText ...>> BDC ... EMC`. Render modes 0-3 (`Hidden` 3); no script is named.
- **Pages the library draws** (`src/Idrak.Imaging/DocumentShapes.cs`, public, writing through `IDocumentPageBuilder`
  only): `DocumentShapes.Paths(plan, style)` and `Write(page, plan, style)`: a `ShapePlan`'s areas as closed polygons
  filled in their tone, its paths as ideal strokes (the style's stroke width times the path's, dashes in those widths, a
  one-point path a dot of round ends, a tone mixing paper and ink); `Write(page, block, text, x, y, style, mode)`: a
  `TextLayout` block's lines, a run for each stretch of one font, standing for the characters of its glyphs' clusters.
  `DocumentShapeStyle`: scale (points a pixel), stroke width, ink, paper, cap, join.
- **Contract changes (public API, dumps regenerated through the runner):** `ImageEncodeOptions` (`Quality`) and
  `IImageEncoder.Encode(image, options)` (a default member: an encoder with nothing to choose writes what `Encode(image)`
  writes; `ImageEncoders`' guard forwards it), so the JPEG quality reaches the encoder through the registry;
  `DocumentWriteOptions.SmoothSteps` (0.25), `Auto`'s second measure (nothing fixed); `DocumentShapes` and
  `DocumentShapeStyle` (Idrak.Imaging). `DocumentWriters` and the writing contracts (`IDocumentWriter`,
  `IDocumentBuilder`, `IDocumentEditor`), now used by Idrak.Data alone, are listed in
  `tests/Idrak.Tests/data/shared-contracts.txt` (decision 1, as D1 listed the reading ones); the inventory regenerated.
- **Decisions the plan did not settle, taken here:** (1) `Whole` cannot embed "the file as read": `IFontFace` gives no
  file's bytes (and a collection's face could not be embedded as read); it writes the whole face through "truetype"
  (every glyph, layout tables, hinting). Every face, CFF too, is embedded as TrueType (`FontFile2`): decision 20 dropped
  the OpenType CFF writer. (2) `Auto`'s steps rule is an option (`SmoothSteps`), not only `LineArtLevels`: smoothed line art
  (many levels at its edges, flat elsewhere) is stored exact. (3) A path neither filled nor stroked is not written.
  (4) The reading side of a quadratic: PDF has none; the cubic it is written as reads back as the quadratic when its
  controls are two thirds of the way to one point (the test reader does this; D7's reader must, for `DocumentWriterSuite`'s
  quadratic). (5) `tools/documents/check_written.py` decoded every stream with qpdf's `read_bytes`, which refuses JPEG
  (the generator's own JPEG samples failed it too): a JPEG or JPEG 2000 stream is now decoded as an image through Pillow
  (README updated).
- **Tests:** group `documents pdf: write` (9, `tests/Idrak.Tests.Library/DocumentPdfWriteTests.cs`): the writer and the
  jpeg encoder registered, `DocumentWriterSuite` on pdf (defaults; 1.4 with level 0; no object streams at level 9);
  the file's syntax, D1's objects written back, object streams and a cross-reference stream, a classic table, levels 0, 1
  and 9, information in any script, an ID when given, dates and an ID when not deterministic, a bad version refused;
  image pages (grey, RGB, 8-bit planes exact, the predictor on line art only, JPEG the encoder's bytes and the codec's
  pixels, every D6 input measured as its JSON measures it, to four decimals, and stored as its `expectedAuto`); vector
  pages (every operator, nested clips, order); text (runs, glyph ids and places to a thousandth of a point, boxes to a
  hundredth, `/ActualText` on the two runs that need it, hidden and stroked runs, one font a face, the subset's ids,
  outlines and empty glyphs, `/W`, a CFF face, FS_Hand_Style mapped, whole embedding); a searchable scan and the shapes
  helper; the same bytes twice and from four threads, attached files, an unfinished file repaired, the later phases
  refused; the 500-page measure; the sample for a person to open. **Read-back:** text through D3's reader; images,
  paths, clips and attached files through D1's objects, filters and content lexer
  (`tests/Idrak.Tests.Library/DocumentPdfWriteReadBack.cs`), as D4 and D7 will read them: once they land their own parts
  are taken (the read-back fills a part only when the reader gives none) and D4's `Decode()` replaces the pixels read here.
  Runs: the targeted filter (below) first 71 tests, 64 passed, 7 failed: `Math.Abs` of the signed byte -128 in the
  predictor's sums overflowed (5 tests), the test's own expected text of `1e20f` and its search for paint operators after
  a space (2), and the inventory (the writing contracts now used by Idrak.Data alone, listed); then the write group with
  the inventory and API checks: 15 tests, 14 passed, the unfinished file not found by the repair (D1 finds a catalog only
  among objects standing on their own: the dispose path no longer packs the page tree and catalog); then 15 passed;
  the full filter again after the last changes: 71 passed, 0 failed. The API dumps (`Idrak.Abstraction`, `Idrak.Imaging`)
  and the inventory regenerated through the runner and checked again without the flags (passed). No new environment
  variable (the options D0 named are read). The plan's filter names `documents writers`, a group that does not exist (the
  writer suite runs inside `documents pdf: write` and `documents contracts`).
- **The checker's verdicts** (`tools/documents/check_written.py test-output/documents/written --expected auto`, 17 files:
  every page kind, the information, the file forms, JPEG and Auto pages, the searchable scan, the shapes page, a mixed
  file with attachments, the sample): all 17 pass: qpdf finds no damage and no warning and decodes every stream (113),
  pdfminer.six and pdfium open each with no warning, and the 13 files with an expected JSON extract exactly its text in
  both. Two findings: pdfminer.six ignores `/ActualText`, so the mapped legacy face's run shows `(cid:N)` for the glyphs
  its shaper added that stand for no character (Idrak and pdfium read it); pdfium gives right-to-left runs in the
  glyphs' order. The unfinished file (`test-output/documents/written-unfinished/`, damaged by design): Idrak's reader
  repairs it (3 pages); pdfium opens it and pdfminer.six reads it (warning of the missing font); qpdf does not, as it does
  not recover objects inside object streams without their cross-reference stream.
- **Timings and memory** (this container, 4 cores, other agents building beside it: noisy; the owner's devices give the
  real ones; each run written to a stream that only counts, the measuring's own collections left out of the time):
  - 500 pages of line art, A4 at 100 dpi (827 x 1169 grey, Auto: Flate with the predictor): cold 4.2 s, warm 4.0 s (8.0
    ms a page; 6.8 and 6.7 s, 13.3 ms a page, in the noisier last run), 1.9 MiB written; **6 KiB allocated a page**; the
    managed memory after a full collection every 100 pages at most 34 MiB (the test's own two source images, 7.4 MiB,
    included), the heap with its garbage at most 35 MiB, and no growth from page 20 to 500 (5.5 MiB less).
  - 500 pages of a scan-like page the same size (Auto: JPEG at 90): cold 22.4 s, warm 18.7 s (37 ms a page; 22.6 and
    26.1 s, 52 ms a page, in the last run), 172 MiB written; 6,199 KiB allocated a page, **6,193 KiB of them the JPEG
    encoder's** (alone on the page: 40.6 ms and 6,193 KiB a call, warm): its working arrays (a `short[64]` a block, the
    sample planes, the scan's list) are made at every call, so the heap with that garbage reached 759 MiB before the GC
    took it, 183 MiB after a full collection at page 20 and 34 MiB at the end (149 MiB less: nothing kept per page).
  The writer's own buffers (the levels, the predictor's rows, the deflate output, the content stream, the object stream)
  are pooled, returned in `finally` or kept one of each for the next page: one page's buffers.
- **Left for later phases:** images' read-back through D4 and paths' through D7 (their readers replace the test's
  read-back); forms, annotations, links, bookmarks, page labels, copied pages, encryption, signing and editing (D9-D12);
  pooling `JpegEncoder`'s working arrays and its speed (a piece of its own, in Idrak.Imaging); pruning a subset's widths
  to a `/DW`; a deeper page tree for very large documents (one node holds every page now).
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: write|documents pdf: text|documents pdf: file|documents contracts|document layouts|outside plug-in: a document format|override loop:|conformance kit:|cli vision: ImageEncoders|fonts write|public API|abstraction inventory|cli health: every environment variable read in src|cli polish: the environment table names real commands|hidden boxing|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d6.txt
  $env:IDRAK_FILTER = "documents pdf: write: a 500-page"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d6-timing.txt
  $env:IDRAK_FILTER = "documents pdf: write: a sample document"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  python -I tools\documents\check_written.py test-output\documents\written -o test-output\documents\written-check.json --expected auto
  ```
  The sample run writes `test-output\documents\written\sample.pdf` (a vector page with text, a photograph, a searchable
  scan and a page of shapes) to open.

**JPEG encoder memory (2026-11-18):** the writer's pages found `JpegEncoder` allocating about 6 MB a page (an 827 x 1,169
grey page: a 64-entry array per block, the planes copied, padded sample planes, a scan list, a growing `MemoryStream`).
Now an 8-bit image's planes are read in place and floats rounded a row at a time (`ImagePreprocessor.Byte`, `ToBytes`'
rounding); colour conversion, downsampling, DCT and quantization run an MCU row at a time on all cores into pooled
coefficient buffers (4 x `ComputeResources.MaxCpuThreads` MCU rows at a time with the standard tables, each chunk written
before the next; the whole image with optimized tables, whose statistics need every block first); the entropy coder
writes 32 bits at a time into one pooled buffer copied once into the file; the hot loops are compiled optimized at once
(a process's first float page ran 130 ms of unoptimized code, 54 ms optimized). The same files byte for byte: 147 cases
recorded from 0e13af7 before the change (`data/image-transforms/jpeg-encoder.sha256`, `--bench-jpeg cases`) and Pillow's
51. `--bench-jpeg`, 0e13af7 against the change, three alternated rounds on a loaded 4-thread container, medians, warm
(KiB allocated a call; the file): grey page q75 from 8 bits 26.6 → 7.1 ms (5,462 → 157 KiB; 136), from floats
15.5 → 8.0 (5,464 → 157), optimized 18.8 → 8.7 (5,206 → 146; 122); colour 1,600 x 1,200 q90 4:2:0 79.0 → 23.6
(26,162 → 949; 938), optimized 90.7 → 32.0 (26,109 → 876; 859), 4:4:4 134.7 → 46.4 (41,049 → 1,907; 1,886); colour page
1,654 x 2,339 from floats 133.3 → 52.6 (52,116 → 1,171; 1,150). Cold (the process's first encode, the grey page):
74.8 → 38.4 ms. Not done: the DCT and colour conversion in vectors; the entropy coding (serial) overlapped with the next
chunk's DCT. Tests: `image transforms|image data: 8-bit|jpeg:|synthetic data|scans:|cli vision: ImageEncoders|naming:|
public API|test runner`, 42 passed.

**JPEG round trip (2026-11-18):** the `jpeg=Q` transform (`PillowImageOps.JpegRoundTrip`, the scan look of every
profile) encoded with optimized tables only to decode the file at once: a statistics pass and the whole image's
coefficients for tables that change the file, never its pixels, then the entropy coding both ways. While the "jpeg"
codec is the library's, `JpegEncoder.RoundTrip` now writes each MCU row's quantized coefficients straight into the
decoder's coefficient arrays (natural order, the divisors in that order; the decoder's frame made from the same SOF0 and
DQT bodies the file holds, `JpegDecoder.ForFrame`) and takes its pixel stage's image: nothing entropy coded, written or
read back, and only the decoder's coefficients held (before: also the encoder's, 2.0 MB for the grey page and 3.0 MB for
the colour one, its output buffer and the decoder's copy of the file). An app's "jpeg" codec still decodes a real file,
written with the standard tables. The same pixels: 979 round trips hashed with ad4859a's build before the change (grey
and colour, 8-bit and floats; 1 x 1 to 250 x 131 at every MCU edge at qualities 1 to 100 and every subsampling; the
scan-like page grey and colour and a 1,001 x 701 pattern at 35 to 95; the transform's pipelines and three profiles'
samples) and Pillow's 51 decoded files; the encoder's files unchanged (its 147 and Pillow's 51 byte for byte); a new
test compares the round trip with its file decoded. Three alternated rounds on a loaded 4-thread container, medians,
warm (KiB allocated a call, the pixels being 944 grey and 2,832 colour): grey page 827 x 1,169 q75 11.2 → 3.5 ms (1,105
→ 953), q40 8.5 → 3.6, q92 17.0 → 3.7 (1,252 → 952); colour page 827 x 1,169 q75 4:2:0 15.5 → 6.6 (3,001 → 2,850), from
floats 18.8 → 10.0, q90 4:4:4 25.5 → 9.8; colour page 1,654 x 2,339 q90 from floats 85.7 → 34.4 (12,459 → 11,359). Cold
(the process's first round trip): grey 111.7 → 82.0 ms, colour 133.3 → 101.8. A grey page through a profile's sample:
`clean` (jpeg=86 alone) 13.1 → 4.5 ms, `pale` (fade, uneven, rotate, noise, jpeg=78) 27.6 → 19.7, `office` (rotate,
blur, lowres, jpeg=54) 42.1 → 39.4. Tests: `image transforms|image data: 8-bit|jpeg:|synthetic data|scans:|cli vision:
ImageEncoders|public API|test runner`, 40 passed.

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

**As built (2026-11-18):**
- **Vector content read** (`src/Idrak.Data/Pdf/PdfContentVector.cs`, a part of D3's `PdfContentReader`; on when a read
  asks for paths, glyphs or images): paths built in page points through the current matrix (`m l c v y h re`), a cubic
  whose controls lie two thirds of the way to one point read as the quadratic it raises (D6's decision 4); painting and
  order as D3 counts them (`S s f F f* B B* b b* sh`; `n` paints nothing); clips `W W*`, a form's `/BBox`, and text
  clips (render modes 4-7: the runs between `BT` and `ET` named by `ClipRun`/`ClipRunCount`); the line state `w J j M
  d`; colours in every space (`PdfColours.cs`: device, ICC by its component count, `CalGray`, `CalRGB`, `Lab`,
  `Indexed`, `Separation` and `DeviceN` through their tint transforms); functions of types 0, 2, 3 and 4
  (`PdfFunctions.cs`; the calculator compiled once, a 100-value stack, a million-step budget); `ExtGState` (`LW LC LJ
  ML D ca CA BM SMask`); shadings of types 1-7 (`PdfShadings.cs`; patches cut into 16 x 16 cells of two triangles);
  tiling patterns (the cell as a page of its own; an uncoloured one in the fill colour) and shading patterns;
  transparency groups (a form's `/Group`) and soft masks (luminosity or alpha, the mask's content as a page of its own,
  its backdrop). Images (D4) now take their opacity, a stencil's colour, blend, group and soft mask; a `Separation` or
  `DeviceN` image decodes through its tint transform (D4 counted it as not supported).
- **Fonts for drawing:** `CffFontFormat` ("cff", a bare CFF program; CID-keyed glyphs by `cidNNNNN` names) and
  `Type1FontFormat` ("type1": PFB, PFA or the PDF's `FontFile` with `/Length1-3`, eexec and `lenIV`, `Subrs`, flex and
  hint replacement, `seac` accents, `div`) in `src/Idrak.Imaging/PostScriptFonts.cs`, registered in `FontFormats`
  with no file extension (read by their first bytes); each gives a `post` table of its glyph names and a `cmap` of its
  built-in encoding (`PostScriptFontTables.cs`: the CFF standard strings and encoding, made from fontTools 4.68, credited
  in `NOTICE`). `CffOutlines` reads names, charset, encoding and widths. Type 3 fonts (`PdfType3.cs`): each glyph's
  filled paths from its `/CharProcs` stream through `/FontMatrix`, read once when first drawn. `PdfFont` finds a simple
  font's glyph by its name (`/Differences`, the built-in encoding) for Type 1, bare CFF and Type 3 programs; a vertical
  run's glyphs move by their position vector (`/DW2`). Fonts not embedded come from `DocumentDrawOptions.Fonts` (the
  existing `IDRAK_DOCUMENT_FONTS`), matched by name and then by the characters they cover; a substitute's glyphs are
  stretched to the document's widths (0.75 to 4/3), slanted 0.21 for an `Italic` or `Oblique` name and emboldened by a
  stroke of 3% of the size for a `Bold` one; with no folder or no match the run is left out and counted (`missing-font`).
- **The drawer** ("coverage", `src/Idrak.Imaging/DocumentDrawer.cs`, `DocumentScene.cs`, `DocumentPaints.cs`,
  `DocumentStroker.cs`; one line in `LibraryRegistrations.cs`): the page's parts become a scene once (outlines in
  pixels, in content order), then drawn band by band (`BandRows`: 128 Ki pixels a band, 8 to 1,024 rows), bands in
  parallel, on premultiplied float RGBA; 8-bit planes out, grey by 0.299/0.587/0.114. Fills by their rule; strokes
  through `Pens` and an offset outline (`DocumentStroker`: butt, round and square caps; miter, round and bevel joins
  with the miter limit; dashes; a closed path as two loops); glyph runs through their faces' outlines and `GlyphMatrix`
  (stroked and clipping render modes too); images by their homography (`PerspectiveWarp`, the scan helpers' `Homography`
  and `Warp` moved out of `ScanImageOps` unchanged): nearest when magnified, the area's mean when shrunk; clips as
  coverage masks cached per band; shadings (axial and radial through a 1,024-step lookup, function-based a pixel at a
  time, meshes as triangles by their pixel centres, `Extend`, `Background`); tilings (an upright cell snapped to whole
  pixels as pdfium does, the others mapped back into a tile at the page's resolution); the 16 blend modes (separable,
  and hue, saturation, colour and luminosity); groups (isolated, knockout; a non-isolated, non-knockout opaque `Normal`
  group drawn straight through) and soft masks drawn per band from their own scenes. `DocumentDrawing.DrawPages` draws
  several pages in parallel (`ComputeResources.ParallelOptions`; `parallel: false` one after another). **The
  rasterizer's fill rule** (`CoverageRasterizer.Fill(..., OutlineFill)`): `NonZero` and `EvenOdd` accumulate signed
  coverage in the box clipped to the pixels (edges left of it folded onto its left side) and fold it as AGG does;
  `OutlineFill.Glyph`, the default, runs the glyph code as it was (not a line of it changed).
- **Contract changes (public API, dumps regenerated through the runner):** `OutlineFill` and
  `IOutlineRasterizer.Fill(outline, scale, x, y, coverage, width, height, rule)` (a default member: `Glyph` forwards to
  the old `Fill`, any other rule `NotSupportedException`; `OutlineRasterizers`' guard forwards both); `DocumentPath`
  (`FillPattern`, `StrokePattern`, `Blend`, `Group`, `SoftMask`, `ClipRun`, `ClipRunCount`), `DocumentTextRun`
  (`StrokeColour`, `StrokeWidth`, `GlyphMatrix`, `Blend`, `Group`, `SoftMask`), `DocumentImage` (`Alpha`, `Stencil`,
  `Blend`, `Group`, `SoftMask`), `DocumentPage` (`Shadings`, `Groups`, `SoftMasks`); new in `DocumentPaint.cs`:
  `DocumentBlend`, `DocumentShadingKind`, `DocumentShading`, `DocumentPattern` (a shading, or a tiling's cell, box,
  steps and matrix), `DocumentGroup`, `DocumentSoftMask`; `DocumentDrawing.DrawPages`; `CoverageRasterizer.Fill` with
  the rule (Idrak.Imaging). `IDocument`, `IDocumentPageDrawer` and `DocumentPageDrawers`, now used by Idrak.Data and
  Idrak.Imaging, left `tests/Idrak.Tests/data/shared-contracts.txt`; the inventory regenerated. No new environment
  variable.
- **The tolerance and why:** pdfium (`pypdfium2`, the generator's reference) and the library are two rasterizers whose
  edges differ by up to half a pixel, so both drawings are first averaged 3 x 3, then the share of pixels with a channel
  off by more than 32 levels is held to **2%**, or **8%** on a page with a run in a standard font not embedded (DejaVu
  Sans in every style here, pdfium's own substitutes there: the glyphs differ in shape, and 8% is about the share such
  text covers). Where pdfium departs from the colour rules the samples' JSON states (D4's), the JSON wins: those parts
  are left out of the comparison by their boxes, each with its reason, and their colours checked against the JSON
  (`vector-colour-spaces` paths 3, 7, 11, 14, 15, 17: pdfium's own CMYK to RGB, and a `DeviceN` with a one-input
  stitching tint transform painted white; `colour-spaces` images 3, 5, 6, 11: pdfium's CMYK and Lab conversions and an
  index past the highest not clamped; `predictors` image 7: pdfium misreads the TIFF predictor on 4-bit samples, whose
  pixels D1 and D4 check exactly). pdfium sizes a page up where the contract rounds (583.3 points to 584 pixels): the
  shared region is compared when the sizes differ by one. The knockout group and every shading are also probed at their
  JSON's points within its tolerance.
- **How each sample scored** (53 pages of the D3, D4 and D7 samples at their references' resolution; the share off by
  more than 32 levels): at 2%: `truetype-simple` 0.23, `type0-identity` 0.27, `type0-no-tounicode` 0.16 and 0.40,
  `type0-no-cmap` 1.46, `cidtogidmap-stream` 0.17, `type3` 0.01, `rtl` 0.54, `mixed-direction` 0.50, `vertical` 0.24 and
  0.08, `searchable-scan` 0.06 and 0.07; at 8%: `standard14` 5.16, `predictors` 4.16, `colour-spaces` 4.24, `bit-depths`
  3.82, `decode-arrays` 3.37, `masks` 2.86, `text-state` 2.81, `inline-images` 2.68, `encodings` 1.84, `dct` 1.80,
  `image-placement` 1.72 and 0.36, `render-modes` 1.21, `text-matrix` 0.71, `actual-text` 0.69, `page-tree` 0.40, 0.47
  and 0.40, `rotated-pages` 0.55, 0.41, 0.57, 0.54, 0.15 and 0.20, `cropbox` 0.55 and 0.50, `scan-ocr` 0.33,
  `attachments` 0.23, `vector-paths` 0.48, `vector-fill-rules` 0.18, `vector-strokes` 0.89, `vector-clipping` 2.07,
  `vector-colour-spaces` 0.18, `vector-shadings` 3.76, `vector-patterns` 3.87, `vector-transparency` 3.02,
  `vector-fonts` 1.04, `vector-images` 3.21, `vector-turned-pages` 0.66 and 1.30. Pages the pdf writer wrote draw back
  as the library draws the same parts in memory with no pixel off by more than 8 levels (the vector page, the page of
  shapes), and an image page as its image, exactly.
- **Tests** (`tests/Idrak.Tests.Library/DocumentPdfVectorTests.cs`): group `documents pdf: vector` (3): every D7 sample
  page's 111 paths, 13 clips, 7 shadings, groups and the 21 runs' and 8 images' paint as their JSON says (points within
  0.002 point); `vector-fonts`' four kinds of face, its 23 Type 1 glyphs' outlines and widths as fontTools reads them;
  D6's quadratic read back, functions of every type on known values, hand-made pages (every operator, a dash, nested
  clips, tint transforms). Group `documents drawers` (3): the drawer registered and `DocumentPageDrawerSuite` on it;
  every sample page against pdfium (scores above, each drawn page saved to `test-output/documents/drawn/`); written
  pages drawn back; a scan with and without its hidden text; a missing font counted; the fill rules on hand-made
  outlines; a large page with bounded memory; pages in parallel giving the same pixels as one by one; the timings.
  D6's read-back (`DocumentPdfWriteReadBack.cs`) now takes the reader's own parts (its path and image reading removed).
  **Runs:** the targeted filter (below) with the API and inventory flags: 177 tests, 172 passed, 5 failed: the inventory
  (three new abstract classes outside `Idrak.Abstraction`: made sealed classes holding a delegate), the shared-contracts
  list (the three lines above), the vector test's floor of 150 paths (the samples have 111), the drawer comparing only
  11 pages (pdfium's sizes rounded up: the shared region compared) and the fill-rule test's own expectation (a square
  wound against the one it overlaps empties the overlap under nonzero); then `documents pdf: vector|documents
  drawers|abstraction inventory`: 9 passed, 1 failed, five pages past 2% (`predictors` 7.16%, `standard14` 6.69%,
  `colour-spaces` 10.03%, `vector-colour-spaces` 6.76%, `vector-patterns` 16.56%): upright tilings snapped to whole
  pixels as pdfium does, substitutes stretched, slanted and emboldened, the standard-font pages held to 8% and pdfium's
  departures left out as above; then 10 passed; then the whole filter again without the flags (the API and the
  inventory checked): **177 passed, 0 failed**. Existing glyph drawing: the glyph code is unchanged and every synthetic
  page group in the filter passes.
- **Timings and memory** (this container, 4 cores, other agents building beside it: noisy; measured apart cold and
  warm, managed allocation per call; the owner's devices give the real ones):
  - a vector page (`vector-transparency`): 100 dpi (556 x 417) cold 10.6 ms and 0.85 MiB, warm 8.9 ms and 0.79 MiB
    (0.66 MiB the pixels); 300 dpi (1667 x 1250) cold 65.4 ms and 18.1 MiB, warm 46.0 ms and 6.10 MiB (5.96 MiB the
    pixels).
  - a text page (`truetype-simple`): 100 dpi cold 2.9 ms and 1.17 MiB, warm 1.9 ms and 0.77 MiB (0.70 the pixels);
    300 dpi cold 11.8 ms and 6.73 MiB, warm 8.9 ms and 6.34 MiB (6.26 the pixels).
  - a scan page (`scan-ocr`, A4): 100 dpi cold 15.7 ms and 4.04 MiB, warm 15.3 ms and 3.77 MiB (2.76 the pixels);
    300 dpi (2479 x 3508) cold 104 ms and 25.9 MiB, warm 127 ms (noise: slower than cold) and 25.9 MiB (24.9 the pixels).
  - a large page (`vector-transparency` at 1200 dpi, 6667 x 5000): 674 ms, 95.9 MiB allocated of which 95.4 MiB the
    pixels returned, 0.5 MiB besides (the bands); the GC's peak heap 329 MiB with 211 MiB of the test process before it.
  - the 11 D7 samples at 150 dpi, warm: one after another 170 ms, in parallel 162 ms: little gain, since each page's
    bands already take every core; the pixels are the same either way.
- **Left for later:** a non-isolated group with another blend or opacity is drawn as an isolated one; a stroke under a
  matrix that is not uniform has its width scaled by the root of its determinant (its pen not sheared); a `d0` Type 3
  glyph's own colours and images and every glyph's strokes are not kept; a TrueType font's Mac standard glyph names
  (`post` version 1, and version 2's indices below 258) are not read when a font's glyph is found by name; a CFF
  Expert encoding reads as the standard one; a tiling cell's `/BBox` is not a clip; a pixel at an edge is the AGG
  rule's approximation; annotations are not drawn (D10); no command of the CLI draws a page yet (D13's bench). The
  drawer's numbers became settings: see "Drawer settings, 2026-11-19" at the end of this phase.
- **Decisions the plan did not settle, taken here:** (1) The fill rule is a default member of `IOutlineRasterizer`
  with its own enum, so every rasterizer keeps working and the glyph path is not touched. (2) Vector paint (patterns,
  shadings, groups, soft masks, blend modes) is new public data on the page, not a drawer's private reading of the PDF,
  so any format and any drawer can carry it. (3) The inventory's rule against new abstract classes holds: the scene's
  items, paints and PDF functions are sealed classes holding a delegate. (4) Upright tilings are snapped to whole pixels
  as pdfium draws them (no seams, and it matches); others are mapped exactly. (5) Where the reference departs from the
  samples' JSON, the JSON (the plan's rules) wins and the part is left out of the pixel comparison with its reason.
  (6) The shared warp is named `PerspectiveWarp` (`ImageWarp` is Idrak.Vision's). (7) The Type 1 and CFF formats claim
  no extension: they are read from PDFs and by their first bytes, not chosen by a file's name.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: vector|documents drawers|documents pdf: write|documents pdf: images|documents pdf: text|documents pdf: file|documents pdf: encryption|documents contracts|document layouts|conformance kit:|fonts:|fonts write|fonts woff|shapes:|pages:|plain:|glyph poses:|scans:|synthetic data:|hidden boxing|override loop:|outside plug-in|public API|abstraction inventory|every environment variable read in src|the environment table names real commands|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d7.txt
  $env:IDRAK_FILTER = "documents drawers: the fill rules"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d7-timing.txt
  $env:IDRAK_FILTER = "documents drawers: the library's coverage drawer"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d7-scores.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```
  The last run draws every D3, D4 and D7 sample page into `test-output\documents\drawn\` (for example
  `vector-transparency.p1.png`, beside pdfium's `test-output\documents\samples\d7\vector-transparency.p1.png`) and
  writes each page's score.
- **Drawer settings, 2026-11-19:** the drawer's numbers are `DocumentDrawOptions` settings (a second part of the
  record, `src/Idrak.Abstraction/Data/DocumentDrawSettings.cs`), each defaulting to the value the drawer always had and
  refused out of range by an `ArgumentOutOfRangeException` naming the setting and its range:
  - `BandPixels` (`IDRAK_DOCUMENT_BAND_PIXELS`): about how many pixels a band holds, 131,072 (128 Ki), 1 to 67,108,864;
    `MinBandRows` 8 and `MaxBandRows` 1,024 bound its rows (1 to 65,536 each; drawing refuses a minimum above the
    maximum, and a band past what one array holds).
  - `ShadingSteps` (`IDRAK_DOCUMENT_SHADING_STEPS`): the axial and radial shadings' table, 1,024 steps, 2 to 1,048,576.
  - `MaxTilePixels` (`IDRAK_DOCUMENT_MAX_TILE_PIXELS`): the largest tile, 1,048,576 pixels (1 Mi), 1 to 67,108,864.
  - A face found for a font that is not embedded: `MinSubstituteStretch` 0.75 (more than 0, at most 1) and
    `MaxSubstituteStretch` 4/3 (1 to 100), `SubstituteSlant` 0.21 (0 to 1; 0: not slanted), `SubstituteStroke` 0.03 of
    the size (0 to 0.25; 0: not emboldened). No variables: they are an app's choice of look, made in code.
  - `FillFlatness` (`IDRAK_DOCUMENT_FILL_FLATNESS`): 1/32 of a pixel, 0.001 to 16, for fills, glyphs and clips;
    `StrokeFlatness` (`IDRAK_DOCUMENT_STROKE_FLATNESS`): 0.1 pixel, 0.001 to 16, for a stroke's flattened curves and its
    round joins, caps and dots.

  The five variables are read through `DocumentSettings` and listed in the CLI's environment table (and so in its help).
  The scene reads the numbers once a draw into its builder's fields (nothing per pixel or per glyph, no new allocation);
  `BandRows(width, pixels, minRows, maxRows)`, `ShadingPaint.Of(..., steps)`, `TilePaint(..., maxTilePixels)` and
  `DocumentStroker.Stroke(..., flatness, ...)` take them as arguments. The fill flatness reaches the rasterizer through a
  new default member of the contract, `IOutlineRasterizer.Fill(..., rule, flatness)`: a rasterizer that does not
  implement it fills by its own measure, `OutlineRasterizers`' guard forwards it to an app's, and `CoverageRasterizer`
  implements it (its other fills keep 1/32). The band budget stays 131,072 rather than "auto": the runtime reports no
  cache size in a portable way (rule 70), and the band governs the memory a core holds and the spread over the cores.
  - **The defaults draw the same, proven:** every sample PDF (d1 to d12, up to six pages each) drawn at 150 dpi RGB with
    the fonts folder, 72 dpi grey, 300 dpi with annotations and 100 dpi without fonts, each drawing's 8-bit planes and
    issues hashed (SHA-256) at the start (d18fd2f) and after: 668 drawings (the 12 hostile and password samples refused
    alike), every hash the same.
  - **The bands changed the drawing** (measured on the drawer as it was, the settings changing no byte): the rule fill
    works each band's rows in the band's own frame, so one-row bands differ from the default bands at 930 of 80.9 M
    values (the 47 D3, D4 and D7 pages at 150 dpi) and one band for the whole page at 156, by one level each.
  - **Tests:** `documents drawers: the drawer's settings` (`tests/Idrak.Tests.Library/DocumentDrawSettingsTests.cs`): the
    defaults and the band rows the drawer always took (every width to 40,000), each variable read, a bad value refused
    naming its variable and value, each setting out of range refused naming itself, crossed row bounds refused; on the
    samples at their reference's dpi, the coarse value changes the drawing (`vector-shadings` 2 steps: 23,308 values,
    6.44% off pdfium; `vector-patterns` a 1-pixel tile: 223,157, 29.24%, and 400 pixels: 23,716, 7.57%; `vector-paths`
    fills within 4 pixels: 5,092, 0.90%; `vector-strokes` strokes within 4 pixels: 4,728, 1.31%; `standard14` not
    slanted 30,342, not stroked 25,602, not stretched 59,598) and the fine one stays within the page's tolerance and
    probes (256 steps: the same bytes; fills within 1/128: 0.48%; strokes within 0.02: 0.90%); the rasterizer's default
    flatness is its fill as it was for every rule, 4 pixels covers a circle less, a flatness not above 0 is refused; an
    app's rasterizer over a library one gets the flatness through the guard, one without the overload fills by its own.
    **Run:** the filter `documents drawers|documents pdf: vector|documents pdf: forms|documents pdf: write|documents
    contracts|outside plug-in|override loop:|kind table:|conformance kit:|fonts:|outline tracers:|pages:|shapes:|glyph
    poses:|public API|abstraction inventory|every environment variable read in src|the environment table names real
    commands|hidden boxing|test runner` with the API and inventory flags: 150 tests, 149 passed, 1 failed (the test's own
    expectation that 256 shading steps change the drawing: they draw the same bytes, so a fine value is now held to the
    tolerance only); then that test, the API and the inventory without the flags: 7 passed. The API dumps gained the
    settings, their variables and the rasterizer's overload; the inventory did not change.
  - **Bands never change the drawing** (a second commit, so it can be taken or left on its own): `CoverageRasterizer`'s
    nonzero and even-odd fills work each row in the outline's own frame (y's fraction kept in the points, its whole
    pixels moved to the row index) and find each row's stretch of an edge from the edge's ends (`LineRows`) instead of
    carrying it from the row above, so a band gets exactly the rows the whole target would; the glyph rule's code is
    untouched. Now one-row bands and one band for the whole page draw the 47 D3, D4 and D7 pages exactly as the default
    bands do (0 of 80.9 M values). The price, measured against the drawings before: 1,392 of 87.1 M values (0.0016%)
    move by one level, in 49 of the 94 drawings of those pages at 150 dpi RGB and 72 dpi grey, none by more; of the 668
    hashed drawings, 316 change by such values. No old drawing can be kept while bands stop mattering, since the old
    one depended on where the default bands fell. Test `documents drawers: bands never change the drawing`: outlines
    filled into bands of one, seven and 64 rows (y with and without a fraction, nonzero and even-odd) equal the whole
    fill row for row; every D3, D4 and D7 sample page drawn with one-row, seven-row and one band equals the default
    bands' drawing and issues. **Run:** the same filter, no flags: 151 passed, 0 failed (every sample still within its
    tolerance against pdfium).

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

**As built, the built-in part (2026-11-18): the "truetype" writer, subsets, legacy fonts converted once, the gap fixed.**
- **The writer** (`src/Idrak.Imaging/TrueTypeWriter.cs`, `TrueTypeGlyphs.cs`, `LayoutClosure.cs`; public `TrueTypeWriter`,
  registered as "truetype" through Idrak.Imaging's `LibraryRegistrations`): any `IFontFace` as a `.ttf`. The library's own
  TrueType faces (and a mapped face's own glyphs) keep their glyph records as they are, composites included, so their
  outlines read back the same points and their left side bearings stay the font's; any other outline is written as
  quadratic contours (an on-curve point exactly midway between two controls left implied, a contour's closing segment its
  own), a cubic curve turned into the fewest quadratics within `CurveTolerance` (the method of fontTools' cu2qu) and every
  point rounded half up to whole units. cmap formats 4 (delta or glyph-array segments, whichever is smaller per run) and 12
  from the face's characters (the library's faces list their own; any other face is asked every code point); name: the
  face's records kept, the options' strings over them; OS/2: the face's, its Unicode ranges (the specification's blocks),
  code pages (FontForge's rule, as fontTools'), first and last character, average width and `usMaxContext` measured from
  what is written, upgraded to version 4 when older, made from the face when it has none; post 2.0 with the face's names
  (a glyph past them named by its character or id) or 3.0; GDEF, GPOS and GSUB as they are; hinting (cvt, fpgm, prep,
  gasp, the glyphs' programs, maxp's limits) kept only for a whole face with `KeepHinting`. The directory by tag, the data
  in the order the specification recommends, checksums and `checkSumAdjustment`; the same face and options write the same
  bytes (head's dates are the face's).
- **Subsets:** glyph 0, the glyphs asked for, with `KeepLayout` the GSUB closure as fontTools' subsetter computes it (the
  lookups of the features every script's language systems list, feature variations included, applied until nothing is
  added; contextual rules only where their whole context can be there, each nested lookup on the glyphs its place can
  hold), then the parts of composites; ids kept, the others written empty with their advances, cmap holding the kept
  glyphs' characters; without `KeepLayout` GDEF, GPOS and GSUB are left out.
- **Legacy fonts converted once** (`FontCache` in Idrak.Abstraction, beside `FontMaps`): `FontFormats.Read(path)` and
  `Read(memory)` hand every face a map made to the cache, which writes it once through `FontWriters` ("truetype",
  `FontCache.Writer`) with its layout tables and hinting, keeps it in `FontCache.Folder` and gives the converted face back,
  on the first read and every later one; a font no map fits is read exactly as before (the same list, the same faces).
  The folder: `IDRAK_FONT_CACHE` (a folder; 0, off, false or no turn the cache off and a legacy font is mapped and drawn at
  every read, as before), default `fonts` in `ComputeResources.CacheFolder` (`~/.cache/idrak`, the local application data
  where there is no profile); `--font-maps cache=on|off|DIR` for `data synth` and `data fonts`; in the CLI's variables
  table. A kept file is named by a SHA-256 of the font file, the face's index, the map (its name and the SHA-256 of its
  JSON), the mapper, the writer and the builds of the assemblies that made it, the curve tolerance and hinting, and the
  drawing settings (`FontMapDrawing`, a page's chances of the hand's features aside); written to a temporary file then
  moved into place, only where the drive has the room; a damaged one is made again. Read back it is a plain TrueType face
  with its own GSUB (the maps' detection leaves it alone), still named by `FontMaps.MappedBy` (so `data fonts` names the
  map). Counted by `FontCache.Hits`, `Converted` and `Failed` (a face the writer refuses is used as mapped).
- **The gap fixed:** `PageDrawing.Asked` reads the hand's features (`PAIR`, `PAI2`, `WORD`) from the font's own GSUB
  feature list (`LayoutTable.HasFeature`), not from `FontMaps.MappedBy`: a mapped face and its converted font ask the same.
  Proven unchanged (rule 71) by the decision itself: for every legacy face mapped by every library map that fits it and
  every one of the eight hands, the GSUB's answer is the map's rule's, and the drawing code that consumes it is unchanged.
- **Tests:** group `fonts write` (5): every test font's face (14: TrueType, CFF, a collection's two, the variable fonts'
  default instance, the three legacy fonts as mapped) written and read back with the same metrics, names, glyph count,
  characters (every code point), advances and outlines (5,640 glyphs the same points; 408 cubic glyphs within 0.485 units at
  tolerance 1; 320 other glyphs, drawn and composed ones and CFF's straight ones, within 0.006), the font's own left side
  bearings, and the shaping of 79 generic lines with and without the hand's features; the eight subsets of the D8 samples
  keep exactly fontTools' glyphs; `FontWriterSuite` on TrueType, CFF and a mapped face; names, hinting, a face of no tables
  (a cubic glyph, a character past the basic plane), the file's structure; the cache (converted, found, another setting,
  a damaged file, off, the command's setting); the hand's features (32 faces and hands asked the same; MCS-HAND's page from
  its converted font has the mapped face's text, description and pixels). Run once on the CPU with
  `fonts write|fonts:|shaping:|pages:|plain:|shapes:|glyph poses:|scans:|synthetic data:|documents contracts|document
  layouts|conformance kit:|override loop:|outside plug-in|public API|abstraction inventory|every environment variable read
  in src|the environment table names real commands|no command, option or help text names a provider|test runner` and the
  update flags: 136 passed, 0 failed; the API dumps (`Idrak.Abstraction`, `Idrak.Imaging`) and the inventory regenerated
  through the runner and checked again without the flags (passed). The checker then found two faults, fixed: the variable
  fonts' GDEF 1.3 still named an item variation store, which the Sanitizer refuses without fvar (the default instance is
  written, so GDEF's store and GSUB's and GPOS's feature variations are unlinked); and mcs-diwany-jaly-s-u's emdash keeps
  an hmtx bearing of 4 against an xMin of 0 (head's "lsb at x=0" flag clear), which the written font had moved (copied
  records now keep their font's bearing; a check added). Rebuilt, rerun: `fonts write|conformance kit:|documents contracts`
  19 passed, `fonts:` 16 passed.
- **The checker's verdicts** (`tools/documents/check_font.py` on the 26 fonts the tests write into
  `test-output/fonts/written`: every test font whole, the eight subsets, the three converted legacy fonts, the face of no
  tables): fontTools opens each and decompiles every table, the OpenType Sanitizer (opentype-sanitizer 9.2.0) accepts each
  (its only warning: arbfonts-diwany-thuluth's own GSUB lists its tags out of order). The nine whole fonts with a source of
  one font compared with it: outlines exactly (TrueType) or within 0.34 units by bounds, area and centroid (the CFF font),
  every advance and character, and HarfBuzz's shaping of the seven sample lines kept.
- **Timings** (this container, 4 cores; the owner's devices give the real ones): a legacy font mapped and its glyphs and
  tables made at every read 74-203 ms; converted once 94-267 ms (cold: mapped, drawn, written, read back, kept); read from
  the cache 2.7-6.9 ms (warm, median of five), for FS_Hand_Style, MCS-HAND and mohareb-hand.
- **Decisions the plan did not settle, taken here:** (1) The cache hooks into both `FontFormats.Read` overloads and gives
  the converted face cold and warm alike, so what is drawn never depends on the cache's state; pages from a legacy font read
  through them now draw from the converted TrueType: its own glyphs exactly, the drawn and composed ones at whole units
  (half a unit at most; MCS-HAND's sample page came out pixel for pixel the same); `IDRAK_FONT_CACHE=off` gives back the
  mapped face exactly. (2) Glyph 0 is always kept in a subset, as fontTools and the specification do. (3) A subset's layout
  tables are written as they are, not rewritten to the kept glyphs: with the ids kept they stay valid (pruning them for
  size is left). (4) Contours keep their direction (a CFF font's stay counter-clockwise; the fill rule is nonzero either
  way). (5) A variable font is written as its default instance (no fvar, gvar, HVAR, MVAR, avar, STAT). (6) kern, BASE,
  JSTF and MATH are not written (the shapers read GDEF, GPOS and GSUB). (7) One edge of the gap fix: a map whose hand
  feature has ligatures none of which resolve in a font used to turn the feature on (the description said ", pairs" with
  nothing joined); now it does not. No repository font hits it. (8) The cache has counters, not a telemetry event yet (a gap
  for rule 78); kept files of an older build are not pruned; the MAUI app has no font-map settings to carry it to.
- **Left:** the plug-in part (OpenType CFF writing, WOFF and WOFF2 read and written) waits for open question 6, the
  project's name.
- **The owner's run** (held with the others until asked; the legacy fonts' tests convert into the cache folder):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "fonts write|fonts:|shaping:|pages:|plain:|shapes:|glyph poses:|scans:|synthetic data:|documents contracts|document layouts|conformance kit:|override loop:|outside plug-in|public API|abstraction inventory|every environment variable read in src|the environment table names real commands|no command, option or help text names a provider|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d8.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  python -I tools\documents\check_font.py (Get-ChildItem test-output\fonts\written\*.ttf).FullName -o test-output\fonts\check-written.json
  ```

**As built, D8c (2026-11-18): a legacy font's mapped face is its converted font (decision 25).**
- **Measured first** (at 58694b2, the three legacy fonts with the cache off and on): the mapped faces and their converted
  fonts differed in 98 of 923 glyphs (FS_Hand_Style 38, MCS-HAND 29, mohareb-hand 31), and only in how a contour closes:
  TrueType gives each contour its own segment back to its start, which the drawn polygons (digits, marks, strokes, dots,
  bars) did not carry. No point, advance or layout table differed: the drawing already puts every drawn point at whole
  units (rounded half to even as it draws), and a composed letter's only fractional points are its font's implied
  midpoints, which TrueType implies again exactly. The "half a unit at most" of decision 24 did not happen in these fonts,
  and 72 generic pages (below) were already the same both ways.
- **The change:** the writer's point step moves out of `GlyphEncoder.Encode` into `GlyphEncoder.Points`
  (`TrueTypeGlyphs.cs`), the one place an outline becomes TrueType points (whole units rounded half up, a cubic curve made
  quadratic within the tolerance, an exact midpoint between two controls implied, the closing segment the contour's own);
  the writer encodes them, and `GlyphEncoder.Whole` reads them back through `TrueTypeContour.ToOutline`, the reader's
  contour-to-outline routine moved out of `OpenTypeFace`. The mapped face keeps every drawn and composed glyph as `Whole`
  gives it, once a glyph where the glyphs are drawn (`MappedFontFace.DrawGlyphs`; the tolerance of `FontWriteOptions`, as
  the cache writes with); the font's own glyphs are untouched. Both moved routines are the same code as before (compared
  text for text), so the two faces cannot drift: the mapped face's drawn glyphs are what the writer writes, read the way
  every TrueType glyph is read.
- **The proof** (group `fonts write`, new test "cached and mapped faces are identical"): every font under `fonts/arabic` a
  map fits (FS_Hand_Style, MCS-HAND, mohareb-hand), read with the cache off and on, and FS_Hand_Style by the glyph-names
  map too (every other library map that applies, written as the cache writes): 4 faces, 1,238 glyphs with the same steps,
  points and advances, every character and GSUB, GDEF and GPOS byte for byte; 36 pages (two layouts, three generic texts
  with drawn digits, marks and composed letters, the hand's features off and on) byte for byte the same image, text and
  description. The other checks are now exact: every test font written whole has its mapped faces' 162 drawn and composed
  glyphs back with the same points (they were within 0.006 with CFF's straight ones), the font writer contract passes on a
  mapped face with the same points (it was bounds within the tolerance), the cache test's drawn glyphs and the hand's
  features page are the same steps, points and bytes.
- **References that changed: none.** No reference file holds a page or an outline drawn from a legacy font (the synthetic
  pages draw DejaVu and features.ttf, the shaping references the Unicode fonts). Measured before and after the change on
  the same 72 pages (the three fonts, two layouts, three texts, the hand's features off and on, the cache off and on): 0
  pixels changed, the largest difference 0; the converted fonts the cache keeps are byte for byte as before (the same
  SHA-256). So decision 24's rounding changed no page, and none changes now.
- **Speed:** the step runs once a glyph, when the face draws its glyphs (rule 75), nothing per page. Reading a legacy font
  with the cache off and drawing every glyph, before and after (warm, median of five, this container): FS_Hand_Style
  57.6-59.5 and 58.9-60.7 ms, MCS-HAND 122-127 and 124-125 ms, mohareb-hand 115.5-115.7 and 115.4-118.2 ms, within the
  noise.
- **Tests:** one build; `fonts write|fonts:|shaping:|pages:|shapes: (its five tests)|glyph poses:|glyph sources:|outline
  tracers:|scans:|synthetic data:|documents contracts|document layouts|conformance kit:|cli arabic:|public API:|abstraction
  inventory:|test runner:` on the CPU: 111 passed, 2 failed. "cli arabic: every message ... has both languages": two
  command summaries had grown (`trace`'s "or samples its stacks (--stacks)", `profile`'s "or one prompt pass") without
  their Arabic entries; the catalog's two entries now match them. "fonts write: subsets ...": the D8 samples were not
  made in this checkout; made with `tools/documents/make_fixtures.py --only d8`. Rebuilt once, rerun with the whole
  `fonts write` group: 7 passed, 0 failed. No public API changed (the new types are internal).
- **Decisions the plan did not settle, taken here:** (1) The drawing keeps its own rounding (half to even, as it draws):
  its points are whole already, which the writer keeps, so the two agree; changing it to half up would move dots and
  strokes at exact halves and change pages for no gain in identity. One rule everywhere is a small change if wanted.
  (2) A legacy font with CFF outlines (none in the repository): its drawn and composed glyphs are the same both ways, but
  its own glyphs stay cubic in the mapped face and become quadratic in the converted font; making them the same means the
  mapped face converting its own glyphs too (left). (3) A drawn glyph TrueType cannot hold (a point past 16 bits) is kept
  as drawn: the writer then refuses the face and the cache gives the mapped face, so the two paths still agree.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples --only d8
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "fonts write|fonts:|shaping:|pages:|shapes: the contract|shapes: a grid|shapes: blanked|shapes: the same seed|shapes: letter and plain|glyph poses:|glyph sources:|outline tracers:|scans:|synthetic data:|documents contracts|document layouts|conformance kit:|cli arabic:|public API:|abstraction inventory:|test runner:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d8c.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

**As built, the plug-in part (2026-11-18): WOFF read and written from `Idrak.Imaging.Plugins` (decisions 20, 21, 27).**
- **The project** `src/Idrak.Imaging.Plugins` (new, beside the other `*.Plugins`, in the solution and `build/IdrakFromSource.targets`,
  with its README; it references Idrak.Imaging alone and uses the public API only: no assembly grants it internals, it
  declares no library registrations, `LibraryDefaults` does not list it): `ImagingPlugins` (`Package`, `All`, `RegisterAll`,
  `UnregisterAll`; `[assembly: IdrakPluginPackage]`) and, in `Idrak.Imaging.Plugins.Fonts`, `WoffPlugin` ("woff"), which
  registers a `WoffFormat` into `FontFormats` and a `WoffWriter` into `FontWriters`, both with the package as their origin.
- **Read** (`WoffFormat`, ".woff", by its `wOFF` signature): `Decode` checks the whole directory before allocating or
  inflating anything (the header's length is the file's, reserved 0, tables in tag order and never twice, each on a 4-byte
  boundary, after the directory, inside the file, storing no more than its length and no fewer bytes than zlib needs for
  it, 1,032 to 1 at most, no two blocks overlapping, metadata and private data inside the file, the font inside within
  `MaxFontBytes` and exactly the size the header gives), then unpacks the sfnt: its directory by tag, its tables in the
  WOFF's order, each inflated through `ZLibStream` into its place and stopped one byte past its declared length. The
  unpacked font goes to the first other registered format that accepts it (the library's "opentype", through
  `FontFormats`), so the faces are that font's own. Every failure is an `InvalidDataException` naming the file and the cause.
- **Write** (`WoffWriter`, ".woff"): the face written by `SfntWriter` ("truetype", through `FontWriters`, every
  `FontWriteOptions` passed on), then `Encode`: each table zlib-compressed at `CompressionLevel` and kept so only when
  smaller, the directory by tag, the data in the sfnt's own order, every table zero-padded to 4 bytes, the header's version
  head's fontRevision (as fontTools), no metadata or private data; written to the destination once. The same face writes
  the same bytes, and the file unpacks to exactly the truetype writer's file.
- **Settings:** `WoffFormat.MaxFontBytes` (a quarter of the memory free when a file is read, within
  `ComputeResources.CpuMemoryLimit`, as `DocumentOptions.MaxStreamBytes`; `IDRAK_WOFF_MAX_BYTES`), `WoffWriter.CompressionLevel`
  (9; `IDRAK_WOFF_COMPRESSION`, 0 to 9, 0 stores every table), `WoffWriter.SfntWriter`; the two variables in the tool's table.
- **Who sees it:** the idrak tool registers the package at startup as its fifth (`IDRAK_PLUGIN_PACKAGES` short name
  `imaging`, after reinforcement), so `data fonts` (its folder walk now takes the registered formats' extensions) and
  `data synth --fonts` (`FontFormats.ReadFolder`) read `.woff`; both commands' help and messages list the registered
  formats' extensions (".ttf, .otf, .ttc, .woff" in the tool, the first three without the package); `plugins list` and
  `formats` gained the font formats and font writers. Referenced by `Idrak.Cli.Commands` (the tool, and the Mobile app's
  terminal through it) and `tests/Idrak.Tests` (its tests, the API guard, the inventory). `IdrakPlugins`' refusal and
  the help page `idrak help plugins` name it.
- **Tests:** group `fonts woff` (6): the four D8 samples (fontTools' WOFF of the test fonts) read with every table byte for
  byte the source's, the dumps' glyph count, table lengths, advances and characters, the source faces' outlines and every
  character exactly (2,521 glyphs, 1,108 characters of the dumps), and their 20 sample lines shaped as from the source and
  as HarfBuzz shapes them (all 20); the 14 test faces of the TrueType tests written as WOFF unpack to the truetype writer's
  bytes and read back with the same tables, glyphs, outlines, every character and the shaping of 79 lines (451,944 bytes
  for 909,476 of TrueType, 49.7%; 168 tables compressed, 11 stored), the layout checked table by table; `FontWriterSuite`
  on TrueType, CFF and mapped faces, `Save`, `ReadFolder`, `Encode` of a TrueType and a CFF file, the levels and the
  settings; the registration from outside (once, the package's origin, no library default, unregistered as found, `Load`
  as `-P` does) and the truetype writer and opentype reader it builds on overridden by an app (used; its failure reaching
  the caller, falling back under FallBack, compared under Shadow); 19 damaged and hostile files refused (0.07 ms each),
  a 16 MiB bomb past a 1 MiB limit refused before inflating (4,744 bytes allocated) and one lying about its length stopped
  at it (35,920 bytes); `data fonts` on a `.woff` and a folder of them, the help with and without the package. The
  `plugins: packages` guards take the fifth package. Run once on the CPU with
  `fonts woff|fonts write|fonts:|plugins: packages|outside plug-in:|conformance kit:|public API|IdrakFromSource|abstraction
  inventory|every environment variable read in src|the environment table names real commands|cli arabic:|cli polish: help
  lists|cli health: plugins list|cli health: help topics|synthetic data: idrak data synth|no command, option or help text
  names a provider|test runner` and the update flags: 90 passed, 2 failed. "fonts woff: idrak data fonts ...": the
  repository's Amiri has no Arabic-Indic digits, so the test now gives it only the generic lines it holds. "cli polish:
  help lists ...": `idrak trace`'s first help line was 121 characters (its summary had grown in D8c); "printed" became
  "shown", the Arabic catalog entry with it. Rebuilt once; `fonts woff|cli polish: help lists|cli arabic: every
  message|public API|abstraction inventory` without the flags: 14 passed, 0 failed. `api/Idrak.Imaging.Plugins.txt` new
  and the inventory regenerated through the runner.
- **The checker's verdicts** (`tools/documents/check_font.py` on the 14 WOFF files the tests write into
  `test-output/fonts/woff`): fontTools opens each and decompiles every table, the OpenType Sanitizer accepts each (its only
  warning, as for the TrueType files: arbfonts-diwany-thuluth's own GSUB lists its tags out of order). The nine with a
  source of one font compared with it: outlines exactly (TrueType) or within 0.34 units by bounds, area and centroid (the
  CFF font), every advance and character, and HarfBuzz's shaping of the seven sample lines kept.
- **Decisions the plan did not settle, taken here:** (1) A face read from a WOFF is the face its font's reader gives
  (`Format` "opentype"), not wrapped: a wrapper would lose the library's own glyph records (kept by the truetype writer)
  and make it ask every code point. (2) The font inside goes to the first other registered format that accepts it, and the
  sfnt writer is a setting, so a CFF writer or an app's reader later plugs in unchanged. (3) The checks are as strict as the
  Sanitizer's (exact length and font size, sorted directory, no overlap); a table's checksum is carried, not verified
  (zlib's Adler-32 checks a compressed table, the font's reader the rest). (4) The data keep the sfnt's order and the
  decoder keeps the WOFF's, so writing and reading round-trip byte for byte. (5) Level 9 by default (WOFF is for
  delivery); the limit's default as the documents'. (6) The package is the tool's fifth, registered last; the Mobile app's
  font picker still lists .ttf, .otf and .ttc (left). (7) Tables are inflated one after another (fonts are small; spreading
  them waits for a measurement, rule 79).
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "fonts woff|fonts write|fonts:|plugins: packages|outside plug-in:|conformance kit:|public API|IdrakFromSource|abstraction inventory|every environment variable read in src|the environment table names real commands|cli arabic:|cli polish: help lists|cli health: plugins list|cli health: help topics|synthetic data: idrak data synth|no command, option or help text names a provider|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d8b.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  python -I tools\documents\check_font.py (Get-ChildItem test-output\fonts\woff\*.woff).FullName -o test-output\fonts\check-woff.json
  ```

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

**As built (2026-11-19):**
- **Reading** (`src/Idrak.Data/Pdf/PdfForms.cs`, internal): `IDocument.Fields` reads the catalog's AcroForm once (kept by the
  document, safe from several threads): the tree from `/Fields` walked without recursion (a node met twice passed over,
  nesting within `MaxDepth`, named in its error), each node inheriting `/FT /Ff /V /DV /DA /Q /MaxLen /Opt`; a node whose
  kids have a `/T` holds fields, any other is a field whose widgets are itself and its kids without one. Each field: kind,
  value and values (a check box's or radio group's on state; a list's export values, several for a multi-select one), the
  choices (export and text; a radio group's its buttons' on states, through `/Opt` when the states are indices), the flags
  as the contract names them by kind, `MaxLen`, alignment, the default appearance's font, size and colour (the form's own
  only for text and choice fields), the tooltip, the default value, `/MK`'s background, border, a push button's caption
  and a button's mark; each widget's page (`/P`, else the page whose `/Annots` lists it), its box as shown, its `/AS` and
  on state. An XFA form's `datasets` packet (from the `/XFA` array or the whole XDP) is read as XML with no DTD and nothing
  fetched: each leaf's value by its path (a repeated name indexed from its second); the values no AcroForm twin holds (its
  name without the `[0]`s) become text fields of their own, with no widget, and every page read says `not-supported`: the
  XFA layout is not drawn.
- **Widgets as annotations** (D10 adds the other kinds): `ReadPage` with `DocumentParts.Annotations` gives each widget of the
  page's `/Annots` with its box and, unless its flags hide it, its appearance (the `/AP /N` stream of its state) read as
  content of the page through D3's reader (`PdfContentReader.ReadForm`: the stream's box through its matrix onto the
  widget's rectangle, 12.5.5); `PdfDocument.Page`'s content reading moved into `Content`, shared by pages and appearances.
  **The drawer** (`DocumentDrawer.cs`) draws each annotation's appearance over the page when the options' parts ask for
  annotations (the contract said so; nothing drew them).
- **Appearances** (`PdfFormsAppearance.cs`): a text box (one line centred on its capitals, several lines wrapped from the
  top, comb cells, a password's asterisks, aligned), a list (its choices from `/TI`, the chosen rows highlighted), a
  drop-down list's choice, a push button's caption, a check box's or radio button's mark drawn as a path (check, circle,
  cross, diamond, square, star: no font needed) on and off, a signature field's box; the background and border from `/MK`
  as plain content, the text inside `/Tx BMC ... EMC`. A size of 0 fits the text: one line to its box's height and width,
  several from `MaxFontSize` down by half points until they fit. Text goes in the field's own simple font when it has a code
  for every character (written with its codes and widths; the standard fonts from their AFM tables); else it is set in
  faces through the new **text setter** contract and embedded through D6's font path (a composite subset with its
  ToUnicode and `/ActualText`), so every script fills right with no script named: the faces of
  `DocumentFormOptions.Faces`, then the fonts folder's (`DocumentFormOptions.Fonts`, the drawing's `IDRAK_DOCUMENT_FONTS`),
  and the field's embedded program, those named as its font first, the first that shows every character leading. No face
  showing the text: an `InvalidOperationException` naming the options. A value set over a widget's own appearance keeps
  what that appearance draws outside its `/Tx` section (12.7.4.3: its box and border; its resources merged).
- **Filling** (`PdfFormsEdit.cs`, `PdfFormEditor`; `IDocumentEditor.SetField`): the value checked (a text's length in
  characters; a button's on states, `Off` or ""; a choice's export value or text, several only for a multi-select list,
  any text in an editable drop-down; push buttons and signature fields refused), `/V` (and a list's `/I`) set, each widget's
  appearance made (a check box or radio button only its `/AS` switched when it has an appearance for the state, else both
  made). **Adding** (`AddField`, the builder's and the editor's; `PdfNewField`): a terminal field from the contract's
  `DocumentField` (`/FT /T /Ff /V /DV /TU /MaxLen /Q /DA /Opt /I`), its one widget in its own object or several as kids
  (`/Rect /P /F /AP /AS /MK`; a button's on state the widget's, else the choice of its index, else "Yes"), parents made for
  a dotted name's start; the form's resources list the fonts the default appearances name (a standard font object, or the
  embedded face's font). The builder writes a field's appearances when it is added and each widget with its page (which
  lists it in `/Annots`): a widget on a page written already, a name taken, a field under a field with a value and a page
  never added are refused. **Flattening** (`IDocumentEditor.Flatten`, new; names, a parent's taking its children, or none
  for every field): each visible widget's appearance as it shows now (a value set's, else its state's, made when the form
  has none) drawn over its page as a form XObject (`/Fw1`, ... after D11's prefix) placed by 12.5.5, the widgets left out
  of their pages' `/Annots`, the fields and the parents left empty out of the tree.
- **Saving:** an incremental update writes the changed fields and widgets again under their numbers and the new objects
  after; a rewrite writes them where its copier reaches them (`PdfEditCopier.Source.Replaced` and `Dropped`, new). The
  catalog's form entry is written again when the tree changes (fields flattened or added): the source's entries kept but
  `/XFA` (the AcroForm is what is shown), none when no field is left.
- **Hooks in the existing files (small, local):** `PdfDocument.cs` (`Fields`, `Form`, `HasXfa`, `Options`, `PageNumber`,
  the annotations and the XFA issue in `Page`, `Content`), `PdfContent.cs` (`ReadForm`; `Draw`'s form part moved into
  `DrawForm` unchanged), `PdfFonts.cs` (`Composite`), `PdfEditor.cs` (`SetField`, `AddField`, `Flatten`, `Pages`, the form
  editor's two calls in each save mode, the catalog's form entry), `PdfEditPages.cs` (`PdfAnnotsEdit`; a page's `/Annots`
  edited; form XObjects in an overlay's resources; `Fw` among the prefix's names), `PdfEditCopy.cs` (replaced and dropped
  objects), `PdfWriterPage.cs` (`Form`, `Annotations`, `ToUserSpace`), `PdfWriter.cs` (`AddField`, widgets with their
  pages, the form at `Finish`), Idrak.Imaging's `DocumentDrawer.cs` and `LibraryRegistrations.cs` (one line).
- **Contract changes (public API; `api/Idrak.Abstraction.txt` and the inventory regenerated through the runner):**
  `DocumentField` gains `DefaultValue`, `Alignment`, `Font`, `FontSize`, `TextColour`, `Background`, `Border`, `Caption`,
  `Mark`; `DocumentWidget.OnState`; `DocumentFieldAlignment`, `DocumentFieldMark`; `DocumentFormOptions` (`Fonts`,
  `Faces`, `MinFontSize` 4, `MaxFontSize` 12, `Padding` 2, `LineSpacing` 1.15, `BorderWidth` 1, `Highlight`, `TextSetter`)
  and `DocumentWriteOptions.Forms`; `IDocumentEditor.Flatten` (a default member: an editor that does not flatten says so);
  the text setter contract `IDocumentTextSetter`, `DocumentTextSetters` (a `KindTable`, guarded: fall back, or the lines
  compared in shadow), `DocumentTextSetting`, `DocumentSetLine`, `DocumentSetRun`, its library setter "shaped" in
  Idrak.Imaging over `TextLayout`. No new environment variable (the fonts folder is the drawing's).
- **Decisions the plan did not settle, taken here:** (1) Laying a field's text out needs a script per run, and the script
  data lives in Idrak.Imaging, which Idrak.Data does not reference: a new contract (the text setter) carries it, as the
  page drawer carries drawing; an app can replace it. (2) A field's own simple font is used when it has every character
  (no embedding, as every filler does); else faces given, embedded; a field's `/DA` is left as it is when its text needs a
  face. (3) Widget annotations are read and drawn in D9 (forms need them drawn); D10 adds the other kinds to the same
  reading. (4) The form's default appearance is a field's only for text and choice fields (a signature field has none).
  (5) Check and radio marks are paths, not ZapfDingbats glyphs, so they draw with no font; `/MK /CA` still names the
  character for viewers that make their own. (6) A field held inside another object (not an object of its own) is refused
  for filling. (7) Flattening an XFA form's fields drops the `/XFA` packets with the tree. (8) The builder adds a widget only
  to the page in hand or one to come: a written page is not written again.
- **Tests:** group `documents pdf: forms` (6, `tests/Idrak.Tests.Library/DocumentPdfFormsTests.cs`): every D9 sample's
  fields as its JSON says (6 samples, 45 fields, 52 widgets; inherited values, flags and appearances; the XFA data beside
  its twins and the page issue), each widget's appearance read as an annotation with the text its JSON says, and the fields
  read alike from 8 threads; forms-empty filled with forms-filled's values (incremental, its prefix kept; and a rewrite)
  reading back as forms-filled's JSON says, the same bytes twice, values a field does not take refused; Latin, Arabic and
  Cyrillic filled and added in a face given (values and appearances read back in logical order, embedded subsets), a text
  no font shows refused; the builder's every kind (15 fields, one on two pages, under a dotted parent) read back as added,
  a value set over a builder's appearance keeping its background once, the refusals, the editor adding a form to merge-a
  (both modes), a tree past `MaxDepth` refused naming it; flattening (all and by name, a form asking for its appearances, a
  field on two pages, the XFA form); the timings. D6's and D11's refusals of D9 replaced. **Runs:** the brief's filter with
  `documents pdf: images|documents pdf: encryption|document layouts|outside plug-in|override loop:|every environment
  variable read in src|the environment table names real commands|hidden boxing` and the update flags: 118 tests, 113
  passed, 5 failed, all in the new group: a signature field given the form's `/DA` (decision 4); a multi-line value's line
  feed counted as a character the font must encode; a simple font's resource written `/Helv3 0 R` (no space before the
  reference: the reader found no `/Helv`, so every appearance in a document's own font read back empty, found with a scratch
  file through pikepdf). Rebuilt (a line-end literal my edit broke needed one more build); then `documents pdf: forms|public
  API|abstraction inventory`: 11 passed, 1 failed: forms-empty filled drew 9.81% off pdfium's drawing of forms-filled. The
  sample's producer draws each text box's background inside its `/Tx` section with no `/MK`, so any new appearance (pdfium's
  too) shows the box without it: pdfium's own two drawings differ by 10.36%. The test now compares with pdfium's drawing of
  forms-need-appearances (the same values, appearances pdfium makes): 3.37%; and the `/Tx` splice was added. Rebuilt once:
  12 passed; then the whole filter without the flags: **118 passed, 0 failed**. Drawing scores: forms-filled with its widgets
  0.84% off pdfium's; the three-script form drawn with its widgets against it flattened 0.000%; forms-filled flattened
  against its twin drawn alike 0.000%, against pdfium's twin 0.65%.
- **The checker's verdicts** (`tools/documents/check_written.py test-output/documents/forms --expected auto`, 13 files:
  forms filled both ways, in three scripts, built, merge-a with a form both ways, flattened forms): all 13 pass: qpdf finds no
  damage, problem or warning and decodes every stream; pdfminer.six and pdfium open each with no warning; the 7 with an
  expected JSON extract exactly its text in both. **pdfium's form reading** (the generator's `pdfium_fields` on the same
  files, each against the `NAME.fields.json` the tests write): every field Idrak filled, added or kept read with the same
  kind, value, checked state and choices (18 widgets of the built form, 14 of each filled one), none in the flattened files.
  As D6 found, pdfium gives a right-to-left run in its glyphs' order and pdfminer.six ignores `/ActualText`: on the flattened
  three-script page they show the Arabic value in visual order, Idrak's reader in logical order.
- **Timings** (this container, 4 cores, other agents building beside it; a process running the timings alone, so cold is
  the first call; the owner's devices give the real ones): reading every field of forms-filled (12 fields, 14 widgets, the
  document opened each time) cold 68 ms (179 KiB), warm 0.69-0.71 ms (131 KiB); filling one field, saved incrementally cold
  62-64 ms (1,226 KiB: the first appearance and font), warm 1.02-1.03 ms (192 KiB); flattening the page's 12 fields, saved
  incrementally, cold 16 ms (195 KiB), warm 0.95-0.97 ms (189 KiB).
- **Left:** `/MK /R` (a widget's rotation) and `/BS` dash styles are not drawn; a rich-text value (`/RV`) is read as its
  plain `/V`; a field held inside another object cannot be filled; a field's `/DA` is not changed when its text needs a face
  (a viewer that makes its own appearance uses its font); calculation order (`/CO`) and actions are not run (never: no
  JavaScript); the XFA template is not read (only its data); reading a form allocates about 130 KiB for 14 widgets (closures,
  records, each widget's page frame read), a pass of its own if forms grow large; D6's and D11's outputs were not compared
  byte for byte against the base commit (their same-bytes checks pass, and no new path runs without a form edit).
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: forms|documents pdf: edit|documents pdf: write|documents pdf: vector|documents drawers|documents pdf: text|documents pdf: file|documents pdf: images|documents pdf: encryption|documents contracts|document layouts|outside plug-in|override loop:|conformance kit:|public API|abstraction inventory|every environment variable read in src|the environment table names real commands|hidden boxing|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d9.txt
  $env:IDRAK_FILTER = "documents pdf: forms: timings"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d9-timing.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  python -I tools\documents\check_written.py test-output\documents\forms -o test-output\documents\forms-check.json --expected auto
  ```
  The forms the tests write are in `test-output\documents\forms\` (for example `forms-empty-filled-incremental.pdf`, a
  form filled by Idrak, and `forms-three-scripts.pdf`, filled in three scripts) to open.

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

**As built (2026-11-19):**
- **The contract (public API, dumps regenerated through the runner):** `DocumentAnnotation` gains `InteriorColour`,
  `BorderWidth`, `LineEndings`, `Flags` (`DocumentAnnotationFlags`), `InReplyTo`, `ReplyType`, `State`, `StateModel`,
  `Popup`, `Parent`, `Open`, `OverlayText`; `DocumentDestination` gains `Fit` (`DocumentDestinationFit`: position, page,
  width, height, box, bounds and their width and height), `Right`, `Bottom`; `DocumentTextRun.Tag` (`DocumentTag`: the
  element's type, its place among the elements, the marked content's place in the structure's order; an artifact);
  `IDocument.Destinations` (the named destinations) and `IDocumentEditor.Redact(index, areas)`, both default members (an
  implementer of the contract keeps compiling; the registry's wrappers forward the first); `DocumentRedaction` (a box and
  its fill, black unless set, null unfilled), `DocumentRedactionResult` (what was removed, counted); `DocumentRedactions`
  (`Find`: every place a text stands in a page's runs, from its glyphs; `Marked`: a page's redaction marks as areas);
  `DocumentAnnotationLooks.Look` (an annotation without an appearance drawn from its properties: rectangle, ellipse, line
  and its endings, polygon, polyline, ink, a highlight multiplied, underline, strike-out, wavy line, a note's icon, caret,
  a stamp's and an attachment's frame, a link's border; none for free text, popups, widgets and kinds not known), shared by
  the drawer and the writer so a created annotation looks as one without an appearance draws; the `tagged` layout
  (`DocumentTextLayouts.Tagged`, `DocumentTaggedLayout.cs`): each structure element's runs laid out by `lines`, the
  elements in the structure's order, untagged runs after, artifacts out; a page with no tagged run reads as `lines`.
- **Read** (`src/Idrak.Data/Pdf/PdfAnnotations.cs`, `PdfOutline.cs`, `PdfAnnotationsContent.cs`): every annotation of a
  page's `/Annots` in order (a form's widgets among them), every field the samples state; the box, quads, a line's ends,
  vertices and ink strokes through the page's display map; a link's `/Dest` or action (a web address or a place; any
  other action not taken), named ones resolved and kept by name; a file attachment's file; the replies to an annotation
  under it (and each reply still in the page's list); the appearance read as D9 reads a widget's (`PdfForms.AppearanceStream`
  of `/AS`, `Placement`, `PdfDocument.Content` through `ReadForm`, now taking `/CA` as a group's opacity), none for a
  hidden annotation; an annotation without `/NM` known by its object (`#12`) or place. Bookmarks (open and closed, colours,
  styles, nesting, a cycle ended), destinations in the page's points as shown, named destinations (the name tree and the
  catalog's `/Dests`), page labels (the number tree), each read once a document; the structure tree read once (elements
  and `/MCID`s, marked-content references, `/Pg` inherited), each run tagged by the innermost `BDC` with an `/MCID` (inline
  or named in `/Properties`) or `/Artifact` it stands in.
- **Drawn** (`DocumentDrawer.cs`): with `DocumentParts.Annotations` asked, each annotation's appearance, or its look
  from its properties, over the page in the page's order (D9's widget appearances included, its hidden-text option kept);
  hidden, not-viewed and popup annotations not drawn, one drawn from its text alone (free text without an appearance)
  counted `not-drawn`.
- **Written** (`PdfAnnotationsWrite.cs`, `PdfAnnotationsEdit.cs`): `AddAnnotation` and `AddLink` on a page builder, the
  look drawn at once into a form of the page's user space (paths, images, glyph runs; a highlight under `/BM /Multiply`),
  the dictionary written at the document's end (every page numbered then: destinations to later pages), its points in
  user space, threads and popups by id, a file attachment's file embedded; one `/Annots` with D9's widgets. `AddOutline`
  and `SetPageLabels` on the builder (`/Outlines` with counts as the format asks, `/PageLabels`). The editor:
  `AddAnnotation`, `RemoveAnnotation` (by the id it was read with; its popup with it), `AddOutline` (the source's bookmarks
  kept, followed to where their pages stand now), both save modes; page changes ride on `PdfEditPages.Write` through
  `PdfPageEdits` (the page's dictionary changed before the copier writes it, new objects named by references the copier
  writes as they are), beside D9's `PdfAnnotsEdit`.
- **Redaction** (`PdfRedaction.cs`): the page's content interpreted as read and written again operator by operator: a
  text operator keeps the glyphs whose boxes do not overlap an area (by a hundredth of the glyph each way) and moves past
  the removed ones by `TJ` offsets of their advances, so what stays stands where it stood (a span's `/ActualText` over a
  removed glyph dropped, its named properties too); a path inside an area is not painted (its clip kept), one crossing an
  area painted under even-odd clips leaving out each area it crosses, a shading likewise; an image (XObject or inline)
  under an area drawn from a copy whose pixels with their centre inside take the area's fill (black when unfilled), the
  original left out of the resources when no longer drawn; a form under an area drawn from a redacted copy of its content
  (within `MaxDepth`); the annotations over an area removed (a popup with its annotation); each filled area painted after
  the content. Saved as a rewrite only (an incremental save refused naming `Rewrite`), so nothing removed survives.
- **The command** `idrak data redact IN.pdf (--boxes FILE | --text TEXT | --marked) -o OUT.pdf` (`DataRedactCommand.cs`):
  the three sources together, `--ignore-case`, `--fill black|white|none|#rrggbb`, `--password`; a progress line (pages,
  areas removed) and a writing stage; a line per page of what was removed and a total, `--json` the same; written to
  `OUT.partial` then moved (nothing left when refused); help and README with generic examples; in the variables table
  (`DocumentCommands`), its Arabic summary in the catalog.
- **Merged over D9 and D12** (onto `d18fd2f`): D9 read widgets alone (`PdfForms.Widgets`): replaced by the one reader,
  which reuses D9's appearance helpers, so widgets read and draw as before (D9's group passes) and every other kind joins
  them; `ReadForm` takes the opacity; D9's unused refusals in `PdfDocument` and the editor removed; a signed save counts
  annotation and bookmark edits as edits.
- **Tests:** group `documents pdf: annotations` (7, `tests/Idrak.Tests.Library/DocumentPdfAnnotationsTests.cs`): every
  annotation of the D10 samples (49) as its JSON says, field by field, replies under their note; navigation's bookmarks,
  named destinations and labels; the tagged page's order and tags (the artifact out; `lines` the default; untagged files
  the same through both); every kind created by the builder with links, bookmarks and labels, and the editor's adds,
  removals and bookmark (incremental: the source its prefix; rewrite) read back; redaction by the sample's marks (16
  glyphs, the path inside, the crossing one clipped with an even-odd clip leaving out the area, the image's 81 pixels
  exactly as `.redacted.png`, the three marks; no removed word in the file's bytes, any decoded stream or any object,
  as text or hex; drawn against pdfium's `redaction-applied`), by a text match and a box; drawing against pdfium; timings.
  Group `cli: data redact` (2, `tests/Idrak.Tests/CliRedactTests.cs`). D6's and D11's refusal checks now check D10's misuse.
  Run (the filter below, with the update flags): 138 tests, 133 passed, 5 failed, all new: the reading test's floor of 50
  (the samples hold 49; now every one is counted); the editor's rewrite lost a reply (finding its `/IRT` copied the
  source's annotation while the reply's object was open: references and images are now found first); redaction dropped a
  name's `/` and a string's `(` from copied operators (the lexer's start is past them), losing the third line; a link with
  no border counted not drawn (its look is now empty). Rebuilt once: `documents pdf: annotations|cli: data redact` 8
  passed, 1 failed (the CLI test parsed one line of indented `--json`); rebuilt: 2 passed; then `public API|abstraction
  inventory` without the flags and the timing runs: 10 passed, 0 failed. `api/Idrak.Abstraction.txt` and the inventory
  regenerated through the runner.
- **Drawn against pdfium** (100 dpi, the share of pixels off by more than 32 levels after a 3 x 3 mean; standard-font
  pages held to 8%): `annotations` 0.45% with its annotations (3.82% without them), `annotations-no-appearance` 1.57%
  (2.80% without: the looks from properties land where pdfium's own do), `navigation` 0.74% (its links have no border);
  the redacted page against pdfium's `redaction-applied` 1.21%.
- **The checker's verdicts** (`tools/documents/check_written.py --expected auto`): `test-output/documents/annotated`, 3
  files (the builder's every kind, the editor's update and rewrite), and `test-output/documents/redacted`, 8 files (the
  library's and the command's redactions by marks, text, boxes, and a 500-page file before and after): all 11 pass, qpdf
  without damage or warning, pdfminer.six and pdfium extracting exactly the expected text (no removed word). pdfium lists
  every created annotation with its subtype, both links (the place link to page 2) and the bookmarks.
- **Timings** (this container, 4 cores; the owner's devices give the real ones): the 52 annotations of the six D10
  samples, opened each pass, read with their appearances (as D9 reads widgets): warm median 3.7 ms a pass, 543 KiB; with
  every part of their pages: cold 33.7 ms, warm 5.6 ms, 745 KiB; `redaction`'s page redacted by its marks and saved as a
  rewrite: cold 74.2 ms (compiling), warm 1.85 ms, 101 KiB; 500 pages (142 KB) redacted through the command by a text on
  every page: 0.14 s (0.28 ms a page) after the group's other runs, 168 KB written.
- **Decisions the plan did not settle, taken here:** (1) Replies are listed on the page and under the annotation they
  reply to. (2) An appearance is read with text, paths and images whatever was asked (D9's rule for widgets, now for all),
  none for a hidden annotation. (3) A glyph is removed when its box overlaps an area by a hundredth of the glyph both ways
  (a neighbour that only touches stays); a pixel by its centre (the samples' rule). (4) Annotations over an area are
  removed, widgets and popups only with their annotation; an unfilled area's pixels take black. (5) Redacting turns a
  page's images under an area into Flate RGB or grey copies (masks not carried) and keeps fonts' glyph programs (no text
  left in them). (6) New bookmarks and destinations name pages of the document as saved; the editor writes the bookmark
  tree again (source items kept). (7) `--password` is an option of the command, not a variable (D2's decision 2).
- **Left:** a reply whose parent is removed by redaction can copy the parent back into a rewrite (a reference to it); text
  matched within one run only (a word split across text operators is not found); images in patterns, soft masks and
  Type 3 glyphs are not redacted; a redacted image loses its soft mask; free text without an appearance is not drawn
  (its font is the document's); the builder's copied page's new annotations reach earlier pages only.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: annotations|document layouts|documents pdf: edit|documents pdf: write|documents pdf: vector|documents pdf: text|documents pdf: file|documents pdf: images|documents pdf: forms|documents pdf: signatures|documents drawers|documents contracts|conformance kit:|outside plug-in|override loop:|cli: data redact|cli polish: help lists|cli polish: the environment table names real commands|cli arabic:|public API|abstraction inventory|every environment variable read in src|hidden boxing|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d10.txt
  $env:IDRAK_FILTER = "documents pdf: annotations: timings|documents pdf: annotations: annotations draw|documents pdf: annotations: redaction applied|cli: data redact: the redaction sample"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d10-timing.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  dotnet run -c Release --no-build --project src\Idrak.Cli -- data redact test-output\documents\samples\d10\redaction.pdf --marked -o test-output\documents\redacted\redaction-marked.pdf
  python -I tools\documents\check_written.py test-output\documents\annotated -o test-output\documents\annotated-check.json --expected auto
  python -I tools\documents\check_written.py test-output\documents\redacted -o test-output\documents\redacted-check.json --expected auto
  ```
  The redaction run writes `test-output\documents\redacted\redaction-marked.pdf` (the sample's three marks applied) to
  open beside `test-output\documents\samples\d10\redaction.pdf`.

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

**As built (2026-11-18):**
- **The editor** (`src/Idrak.Data/Pdf/PdfEditor.cs`, `PdfEditPages.cs`, `PdfEditCopy.cs`, `PdfEditFile.cs`, all internal;
  `IDocumentWriter.Edit` on `pdf`): `SetInfo`, `InsertPages` (pages of any document the library's pdf format opened),
  `RemovePages`, `MovePage`, `RotatePage` (to a multiple of 90), `AddPage`, `Overlay` (content over a page through D6's
  page-builder calls, in the page's frame as shown: its display map turned back, so a stamp stands upright on a turned,
  cropped or scaled page; written as a stream after the page's own, which a stream holding `q` opens before them; its
  resources added to the page's own, named after a prefix `Ov1`, `Ov2`, ... that none of the page's names starts with
  when D6's `/Im1`, `/F1`, `/GS1` would clash), `Encrypt`, `Save`. Edits are recorded and written by `Save` (the
  documents pages come from stay open until then; an image given to a page is read then). `SetField` and `AddField`
  answer a `NotSupportedException` naming D9, `AddAnnotation`, `RemoveAnnotation` and `AddOutline` D10, `Sign` D12.
- **Incremental** (`DocumentSaveMode.Incremental`): the source's bytes copied as they are (a line end added when they
  end without one), then what changed under its own number and generation (a page turned or drawn over; the page tree
  when pages change: one flat node under the root's number, keeping what the root passes on, a page under another node
  written with what it inherited), what is new under numbers from the source's `/Size`, and one cross-reference section:
  a classic table after a table or a hybrid, a stream (small objects in a new object stream) after a stream; `/Prev` the
  source's last section; the trailer's `/Root`, `/Info`, `/Encrypt` and `/ID` (the first kept, the second the options'
  `Id`, else kept). A source whose cross-reference was rebuilt by scanning gets a section listing every object and no
  `/Prev`. An encrypted source's new objects are encrypted with its own handler and key (RC4, AES-128, AES-256). What
  earlier signatures cover is the prefix, untouched.
- **Rewrite** (`DocumentSaveMode.Rewrite`): the catalog 1, the page tree 2 (one node, as D6), the pages 3 on (each with
  what it inherits written on it), then what the catalog, the pages and the information reach, copied depth first
  (`PdfEditCopier`): what an object refers to is numbered before it, so one whose content and references' numbers match
  one already written (SHA-256) is written once: streams, arrays, fonts, font descriptors, encodings and graphics states
  (pages, annotations, fields, bookmarks and structure elements stay their own). A reference to a page the result does
  not hold is null, one to a page-tree node the new root. Streams are copied as stored (decrypted, encrypted again for
  the result), deflated at `Compression` when stored plain or when their zlib header says a faster level than the
  options' (no predictor), kept so only when smaller. Object streams and a cross-reference stream (a classic table before
  1.5); the catalog's `/Version` dropped (the header states the output's); a chain past 48 objects deep numbered and
  written after (no deep recursion on long bookmark lists).
- **The builder (C24):** `CopyPage` (a page of another PDF document with what it uses, copied once per source document
  through the same copier, duplicates written once; a reference to a page of it not copied yet numbered, and written
  null at `Finish` if the page never comes; content over it as the editor's `Overlay`), `Encrypt` (before the first page
  or attached file).
- **Saved encrypted** (`PdfSecurityWrite.cs`; `PdfSecurity` made `partial`): AES-256, revision 6, by Algorithms 8, 9 and
  10: a fresh 32-byte file key and salts; `/U` `/UE` from the user password, `/O` `/OE` from the owner password (none: 32
  random bytes, so the permissions cannot be lifted), `/P` from `DocumentPermissions` (the reserved bits set as the
  standard asks), `/Perms`; passwords by SASLprep and UTF-8 up to 127 bytes, as D2 reads them. Strings are encrypted in
  objects that stand on their own (`PdfSyntax.Encrypting`), not in object streams; streams by `/StmF`, embedded files by
  `/EFF`, metadata plain under `/EncryptMetadata false`; the cross-reference stream and the encryption dictionary plain;
  AES with a fresh IV and its padding. Before PDF 2.0 the catalog states Adobe's extension level 8; an encrypted file
  always has an `/ID`. A file keeps its own encryption when edited (any handler D2 reads); `Encrypt(null)` removes it by
  a rewrite when the owner password opened it; changed or removed with the user password, or by an update, is refused
  naming why. Every buffer that held a password or an intermediate key is cleared; the file key on dispose.
- **Hooks in the existing files (small, local):** `PdfWriterSyntax.cs` (`PdfSyntax`: the string cipher, `Literal`,
  `HexString` and `Text` encrypting while it is set; `PdfFileWriter`: `partial`, generations, the offset shift,
  `Security`, `PlainNumber`, `EncryptNumber`, streams encrypted in `EndStream` by a `PdfCryptUse`, no object of another
  generation or the encryption dictionary packed, `/Encrypt` and a second ID in the trailer), `PdfWriterPage.cs` (its
  fonts and images through two delegates, not the builder; an optional frame and resource-name prefix; `Close`; its
  resources readable), `PdfWriterFonts.cs` (`Number` settable; CIDSystemInfo's strings through `Literal`), `PdfWriter.cs`
  (`Edit`, `CopyPage`, `Encrypt`, the information writer shared, the catalog's extension), `PdfDocument.cs`
  (`StatedVersion`, `PageObject`, `Frame`: added at the end, `Page` untouched), `PdfSecurity.cs` (`partial`). D6's output
  is byte for byte as before (its group's same-bytes checks pass); its three D11 refusals moved into this group.
- **Public API:** `DocumentFormats.Unwrap(IDocument)`: the registry's wrapper of a file it opened (`Open(path)`) hides the
  format's own document, which a writer of that format reads to edit it or copy its pages (an Office plug-in's writer
  will meet the same); `api/Idrak.Abstraction.txt` and the inventory (`IDocumentEditor` now implemented by `PdfEditor`)
  regenerated through the runner and checked again without the flags.
- **Decisions the plan did not settle, taken here:** (1) Encryption is random (file key, salts, IVs), so an encrypted
  output is the one output that does not repeat byte for byte; every other edit writes the same bytes from the same
  input. (2) Changing or removing encryption needs the owner password; otherwise the editor, like D2's reader, enforces
  no permission. (3) An incremental update cannot set or remove encryption (its prefix stays as it is): refused naming
  `Rewrite`. (4) A rewrite of a source the reader repaired may be larger: it gets the cross-reference it lacked
  (`missing-xref`, 1,043 to 1,225 bytes); every other sample came out smaller. (5) Recompression is decided by the zlib
  header's level class against `Compression` (level 6 against producers' default: copied as stored), then measured.
  (6) `RotatePage` sets the rotation (the contract's "to"); the sample's relative steps are turned into it by the test.
  (7) The registry's file wrapper is looked through by the public `Unwrap` (above), not an internal grant
  (Idrak.Abstraction grants its internals to the tests alone). (8) A page inserted twice shares its content and
  annotations with its first copy. (9) `Save(mode)` takes the mode; `DocumentWriteOptions.SaveMode`
  (`IDRAK_DOCUMENT_SAVE_MODE`, already in the CLI's table) is what an app passes when it does not choose (tested).
  (10) The editor's page builders get fonts and images through delegates, not an interface (no new contract outside an
  Abstractions namespace).
- **Tests:** group `documents pdf: edit` (9, `tests/Idrak.Tests.Library/DocumentPdfEditTests.cs`): the two incremental
  samples (each revision ends with `%%EOF` where its JSON says and its prefix opens alone; the sections are its revisions,
  newest first; the newest reads as `final`: pages, rotations, title, text, the field's value through the objects, each
  page's annotations, the bookmark); an incremental update of four samples (classic tables, streams over object streams,
  inherited resources), a light edit (the information, a page turned, a stamp) and a heavy one (pages added, moved and
  removed): the source's bytes its prefix, one more section of the source's kind with `/Prev` its last, reading as
  edited, the same as a rewrite of the same edits, the same bytes twice, and the options' save mode choosing; merge-a and
  merge-b merged by `CopyPage` and by `InsertPages` (both modes) as merge-plan says, each page's `/F1`, `/Im1` and `/GS1`
  its own source's; the stamp sample's word at its place and size, translucent, with an image, over merge-a's pages
  (both modes): named `/Ov1F1`, `/Ov1Im1`, `/Ov1GS1` beside the page's own, the page's text, image and state kept;
  split-source split by copying and by removing (text, rotations, smaller than the whole); merge-b copied from two copies
  of its file holding its image and font once; page-operations' five cases in both modes, and content over a page turned
  0, 90, 180 and 270 degrees reading upright where it was put; a rewrite of every readable sample (90) reading the same
  (information, pages, sizes, rotations, text, runs with their boxes, images' data, issues), no larger than its source
  but the repaired one, the same bytes twice; encryption: the builder's AES-256 in three file forms (object streams, a
  1.4 classic table, none packed), encrypted-aes256 updated with its encryption kept, rewritten as it is, decrypted by
  its owner, refused by its user, encrypted anew (a password through SASLprep), merge-b encrypted by a rewrite, each
  opened by each password and refused without and with a wrong one, `/P`, the permissions and `/Perms` checked, no plain
  string or stream in the file; every encrypted sample of D2 and D11 (22) rewritten and updated with its own handler
  (RC4 40 and 128 bits, AES-128, AES-256 revisions 5 and 6, crypt filters, metadata left plain) reading the same; the 13
  D12 signed samples updated (every byte of every byte range unchanged, the signature dictionaries the same, the stamp
  read); the refusals; the timings below. D6's group lost its three D11 refusals (encryption, copied pages, editing),
  which now work.
  Runs (the brief's filter with `document layouts|outside plug-in|override loop:|hidden boxing` added): three builds to
  compile (a property named as `PdfDocument`'s own `Version` method; then two errors the first build never reached: a
  dereference the null-state analysis could not prove, a parameter named as a local of D6's `Finish`); the first run, with
  the update flags, 105 tests: 101 passed, 4 failed: the inventory (an internal interface for the page builders' fonts
  and images, outside an Abstractions namespace: two delegates now), merge-plan (no PDF, so the samples' loader does not
  list it: its JSON read beside merge-a's), `missing-xref`'s rewrite larger than its source (decision 4), and the
  decrypted file's title looked for in bytes the rewrite had packed and compressed (that file now saved unpacked and
  stored). Rebuilt once; the three and the inventory and API checks: 9 passed; the full filter without the flags: 105
  passed, 0 failed. Then the update's untouched pages passed over (timings below): rebuilt once, 105 passed, 0 failed, and
  the 79 files the edit and write groups write without encryption the same bytes before and after (SHA-256).
- **The checker's verdicts** (`tools/documents/check_written.py`): `test-output/documents/edited`, 62 files (updates and
  rewrites of the D11 samples, merges, stamps, splits, page operations, the 13 signed samples updated, the decrypted
  file, clean rewrites of the D6 and D11 samples): qpdf finds no damage, problem or warning and decodes every stream;
  pdfminer.six and pdfium open each without a warning; the 29 with an expected JSON on upright pages extract exactly its
  text in both. The other 14 differ only on turned pages (`/Rotate` 90 or 270): pdfium gives a turned page's two lines in
  the other order and pdfminer.six splits its words into letters, as both read the source samples' turned pages
  (`split-source` checked the same way fails on its page 4 alike). `test-output/documents/edited-encrypted`, 5 files (the
  builder's three forms, encrypted-aes256 updated, merge-b encrypted by a rewrite): with the user password and with the
  owner password all three tools open each, qpdf without damage or warning, the texts as expected; qpdf reads revision 6
  with AESV3 for strings and streams, `/P` and the permissions as set, and which password matched (user, owner); without a
  password all three refuse each. The one note: pdfminer.six says encrypted-aes256-updated does not allow text extraction,
  as it says of its source (its permissions deny copying, by design).
- **Timings and memory** (this container, 4 cores, load average 8-12 from other agents' builds: noisy; the owner's
  devices give the real ones; each written to a stream that only counts): on D13's `pages-3000` (3,978,312 bytes, 3,000
  pages, in memory), a one-field incremental update (`SetInfo`): 345 bytes appended, cold 25.8-41.5 ms (the first call,
  mostly compiling), warm 1.7 ms, 284 KiB allocated; a rewrite: 3,964,581 bytes, cold 68-78 ms, warm 35-43 ms (12-14 µs a
  page), 2.8 MiB allocated; merged with `huge-image` page by page by `CopyPage` (3,001 pages, 7,065,870 bytes in):
  7,054,478 bytes, cold 55-87 ms, warm 60-63 ms (20-21 µs a page), 5.4 MiB allocated; merged with a copy of itself (6,000
  pages, every stream and font the same): 4,015,337 bytes (the second copy's streams and fonts written once), warm 76-103
  ms, 10.4 MiB allocated; the process's peak working set 138-172 MiB, the managed heap 21-28 MiB after. One change
  measured before and after: the update passes over the pages it does not touch without reading them (no frame read, no
  set of pages met, no page list for a source referred to as it is): warm 4.3 to 1.7 ms and 615 to 284 KiB, the same
  bytes. The 90 readable samples rewritten: 1,499,316 bytes to 1,036,680 (69.1%).
- **Left for later phases:** fields, annotations, bookmarks and signing in the editor (D9, D10, D12; refused naming
  them); objects an update no longer uses are left in the file, not freed; a deeper page tree for very large documents
  (a rewrite writes one node, as D6); plain Flate streams deflated again only by their level class (a pass that would
  decode and deflate every one is measurable, not taken); the builder's `CopyPage` copies streams as stored (no
  recompression, duplicates once); an editor holds a small record per page from its start.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  python -I tools\documents\make_fixtures.py --only none --large
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: edit|documents pdf: encryption|documents pdf: write|documents pdf: file|documents pdf: text|documents pdf: images|documents contracts|document layouts|outside plug-in|override loop:|conformance kit:|public API|abstraction inventory|every environment variable read in src|the environment table names real commands|hidden boxing|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d11.txt
  $env:IDRAK_FILTER = "documents pdf: edit: timings"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d11-timing.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  python -I tools\documents\check_written.py test-output\documents\edited -o test-output\documents\edited-check.json --expected auto
  python -I tools\documents\check_written.py test-output\documents\edited-encrypted -o test-output\documents\edited-encrypted-check.json --expected auto --password reader-2026
  ```

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

**As built (2026-11-19):**
- **The DER, on the base library:** `System.Formats.Asn1` is in the shared framework (`Microsoft.NETCore.App.Ref/10.0`
  holds `System.Formats.Asn1.dll`; `System.Security.Cryptography.Pkcs` is not there, so `SignedCms` and
  `Rfc3161TimestampToken` are not used): every structure is read with `AsnReader` (BER taken, as signers write it) and
  written with `AsnWriter` (DER), no DER code of its own and no package (`src/Idrak.Data/Pdf/Cms*.cs`, internal):
  `CmsCertificate` (X.509 read here the same on every platform: the signed part, names, validity, key, basic constraints,
  key usage, extended key usage, key identifiers, OCSP and list addresses; the `X509Certificate2` the contract hands out made
  when asked), `CmsSignedData` (ContentInfo, SignedData, signers by issuer and serial or key identifier, signed attributes
  kept as signed, unsigned ones, certificates and lists), `CmsVerifier` (message digest; signature over the signed
  attributes with the content type and the ESS signing-certificate v1 or v2 they bind, which PAdES asks for), `CmsWriter`
  (SignedData written: content type, message digest and signing-certificate-v2 signed in DER order, the timestamp
  unsigned), `CmsCrl` and `CmsOcsp` (lists and basic responses read, signatures checked by the issuer or a responder it
  authorized, a status at a time; delta, indirect and unread critical extensions refused by name; OCSP requests written),
  `CmsTimestamp` (RFC 3161 tokens and TSTInfo read, the imprint, the token's own signature and the timeStamping purpose
  checked; requests written, responses read against their nonce and digest), `CmsPath` (paths built and checked here: each
  link's signature, validity at the time, CA, path length, key usage, the first certificate's usage, critical extensions
  not read fail the path by name), `CmsKeys` (RSA verified on `BigInteger`, PKCS #1 v1.5 with or without the DigestInfo's
  NULL and PSS with any digest, mask and salt; ECDSA through the base library's `ECDsa`; signing with the app's key as the
  base library holds it).
- **Read and checked** (`PdfSignatures.cs`, `PdfSignatureCheck.cs`, `PdfSignatureChanges.cs`): `IDocument.Signatures`
  lists every signature field of the form whose value is a signature dictionary, in the order made (by where their signed
  bytes end), found once a document: field (full name), kind (`ETSI.CAdES.detached` B-B, B-T with a signature timestamp,
  B-LT when the DSS holds a VRI entry for it, or revocation data and no VRI; B-LTA when a document timestamp follows;
  `adbe.pkcs7.detached` CMS; the two legacy forms; `ETSI.RFC3161` document timestamps as Other, checked by their own
  rules), `KindName` (the SubFilter), signer, `SignedAt` (/M, else the CMS signing time), `TimestampedAt` (genTime, not yet
  checked), reason, location, the byte range and the changes. The byte range is checked against the file: two pairs from
  byte 0, its hole exactly this dictionary's own `/Contents` hex string (its digits are the string read) inside its object,
  ending at the file's end or after a revision's `%%EOF`. `Check` reports six steps in order, each with why: **digest** (the
  byte range read in 1 MiB windows and hashed; `adbe.pkcs7.sha1` the carried SHA-1 too; `adbe.x509.rsa_sha1` the digest its
  PKCS #1 signature opens to, so a signature that opens to another digest is a digest that differs), **signature**,
  **chain** (only the certificates in `SignatureCheckOptions.Trusted`, and the machine's roots with `UseMachineStore`;
  certificates the signature, the DSS, `/Cert` and the timestamp carry are only links), **timestamp** (none passes saying
  so; else its imprint over the signature's value, its token, its authority's chain and revocation), **revocation** (every
  certificate of the path but the anchor, from the DSS, VRI entries, the CMS's own lists and Adobe's revocation attribute;
  online through `SignatureServices` only with `Online` or `IDRAK_DOCUMENT_SIGNATURE_ONLINE`, the responder first, then the
  lists), **changes**. The chain's time is `At`, else the valid timestamp's time, else now.
- **Changes after signing:** the signed revision opened alone (the file's bytes to the signature's end, with the
  document's options and password), every object a later section writes or frees compared with what it was (numbers by
  value, references, containers member by member, streams by their dictionaries and decoded bytes; written again as it was
  is no change), and each that differs placed by walking the current document from the trailer (the DSS, information and
  metadata; the form, its fields and widgets with their names, appearances and signature dictionaries; pages and their
  annotations; pages' content and resources; the rest), the walk stopping once all are placed; then classified as
  validation data, form filling (values and appearances, new signature fields and their widgets, the form's `/Fields`,
  `/SigFlags`, `/DR`), annotations, or other (anything else; a field the signature's FieldMDP lock names). Allowed unless
  other or past the DocMDP permission (pyHanko's rule): the signature's own (`/Reference` DocMDP), else its field's `/Lock`,
  else the certification's when it came first. New objects nothing reaches change nothing.
- **Signing** (`PdfSigning.cs`; `IDocumentEditor.Sign`, `IDocumentBuilder.Sign`): each signature an incremental update of
  its own on what came before: an empty signature field of the form (its widget given the appearance) or a new one (no
  period in its name) on `SigningOptions.Page`, in `Box` (points from the page's top-left as shown, so it stands upright on
  a turned page; invisible without one), added to the page's `/Annots` and the form's `/Fields` with `/SigFlags 3` (a form
  made as an object of its own when there is none); the appearance: text in the standard Helvetica (WinAnsi), an image
  fitted, or both; the signature dictionary (`ETSI.CAdES.detached`, `/M`, reason, location) never packed, its byte range
  and `/Contents` reserved (from the certificates and the key, plus 16 KiB for a timestamp), filled, the bytes hashed
  around the hole (the source's once, each attempt from a copy of that state), the CMS written and timestamped, written into
  the hole; when a token does not fit, made once more with the room it needed. B-T asks the authority the app names through
  `SignatureServices` (nothing else is asked); B-LT then adds one more update: the DSS (existing items kept, the same bytes
  not written twice) with the chain's and the authority's certificates, the revocation data the app gives
  (`SigningOptions.Revocation`) or, with `Online`, fetches, and a VRI entry. An editor with no other change signs over the
  source as it is; with other changes (or a rewrite) the edited file is written to a temporary file first, opened and
  signed onto the destination; several signatures follow each other. A file keeps its encryption (strings and streams by
  its own key, `/Contents` never). The builder signs at `Finish`: `Sign` is called before the first page (as `Encrypt`),
  from then on the file goes to a temporary file (deleted when closed), which is signed onto the destination at the end
  (disposed unfinished, it is copied as far as it got, unsigned).
- **Contract (public API; dumps regenerated through the runner):** `SignatureServices` (a `KindTable` by address scheme;
  the library's "http" and "https" in Idrak.Data over the base library's `HttpClient`, a request bounded by its timeout
  and `MaxResponseBytes`, measured like `MaxStreamBytes`), `ISignatureService`, `SignatureServiceKind`,
  `SignatureServiceRequest`; `SigningOptions.RsaPadding`, `SigningTime`, `Revocation`, `Online`, `Timeout`;
  `SignatureCheckOptions.RequireRevocation`, `Timeout`. `ISignatureService` and `SignatureServices`, used by Idrak.Data
  alone today, are in `tests/Idrak.Tests/data/shared-contracts.txt`; the inventory regenerated. No new environment
  variable (`IDRAK_DOCUMENT_SIGNATURE_ONLINE`, D0's, now read by both options).
- **Hooks in the existing files (small, local):** `PdfDocument.cs` (`Signatures` read once, `Options`, `Disposed`),
  `PdfEditor.cs` (`Sign`, `SaveSigned`, `_destination` settable for the temporary file), `PdfWriter.cs` (the builder's
  output through `PdfSpillStream`, `Sign`, `Finish` and `Dispose`; `Date` internal), `LibraryRegistrations.cs` (http,
  https), `DocumentSignatures.cs` (the options). The D6 and D11 refusals of `Sign` now refuse a call without options.
- **Decisions the plan did not settle, taken here:** (1) Paths are built and checked here, not by `X509Chain`, whose
  answers differ by platform (OpenSSL, CryptoAPI, Security.framework, Android), which would fetch what the file lacks and
  cannot take the file's lists and responses offline. (2) RSA is verified on `BigInteger`: some platforms' policies refuse
  SHA-1 signatures (the legacy forms) and the base library takes no PSS salt length but the digest's. (3) A certificate with
  no revocation data passes the revocation step as "not checked" (soft-fail, as pyHanko's default and the samples' JSON),
  failing only with `RequireRevocation`; what is found revoked at the time always fails. The revoked sample's JSON says
  "untrusted" for its chain (pyHanko's verdict on the whole path): here the chain step passes and the revocation step
  fails, and the test reads it so. (4) The network goes through a registry of its own, unguarded (a service answers for an
  authority: another asked in its place would answer for itself, so nothing is compared or fallen back to); the tests stand
  in a timestamp authority under "idrak-test" and the PKI's addresses over "http". (5) B-LT needs revocation data given or
  `Online`, refused at `Sign` otherwise; signing digests are SHA-256, -384 or -512 (SHA-1 is read only); RSA signs with
  PKCS #1 v1.5 unless `RsaPadding` is PSS. (6) `/M` is `SigningTime`, else now: a signed file does not repeat byte for byte
  (ECDSA and PSS are random anyway). (7) An appearance's text is WinAnsi in the standard Helvetica (other characters `?`);
  any script goes through `AppearanceImage` until D9's appearances are shaped. (8) The certification's permission also
  bounds later approval signatures (stricter than pyHanko, which reads each signature's own).
- **Tests:** group `documents pdf: signatures` (6, `tests/Idrak.Tests.Library/DocumentPdfSignaturesTests.cs`): the timing
  below (first, so its cold is cold); every D12 sample (13 files, 14 signatures) read and checked as its JSON says (kind,
  field, signer by SHA-256, `/M`, the timestamp's time, reason, location, byte range, coverage, changes; each step and its
  detail against `results`, at the JSON's validation time and at the timestamp's, with the JSON's trust alone) and as the
  pyHanko verdicts it records (intact, valid, docmdpOk), alike from four threads; the reasons each step gives (a changed
  byte, the untrusted root named, the revoked signer's date and keyCompromise, the page's content changed under permission
  2), nothing trusted without trust given, the intermediate or the signer itself as the anchor, a time past expiry, required
  revocation (B-LT from the file alone passes it), a disposed document; fetching only when turned on (nothing asked when
  off; the OCSP responder and the root's list asked when on, through an app's "http" over the library's; the variable
  turning both options on; a scheme no service serves refused naming them); signing (B-B with RSA, RSA-PSS and ECDSA; B-T
  and B-LT with RSA and ECDSA; forms-empty's own signature field; a second signature on page 2 keeping the first valid as
  form filling; the builder's; after other edits; after a rewrite; D11's AES-256 sample by its owner password), each
  checking valid (B-LT with revocation required), the source's bytes its prefix, one timestamp asked for B-T and B-LT and
  nothing of "http"; misuse refused (no options, no key, SHA-1, B-T without an authority, B-LT without data, a text field,
  a signed field, a dotted name, a page out of range, the builder after its first page, an unregistered scheme, a token for
  another request). D6's and D11's `Sign` refusals updated. Run with `documents pdf: signatures|documents pdf: edit|documents
  pdf: encryption|documents pdf: write|documents pdf: file|documents contracts|document layouts|outside plug-in|override
  loop:|kind table:|conformance kit:|public API|abstraction inventory|every environment variable read in src|the environment
  table names real commands|hidden boxing|test runner` and the update flags, one build (two compile errors at the first:
  a collection expression typed `int` against a byte span, a pattern variable read past an `||`; fixed, rebuilt once): 104
  passed, 0 failed. Then one change (a form the signing makes is an object of its own, so a later signature writes the form
  alone again), rebuilt once; `documents pdf: signatures|public API|abstraction inventory` without the flags: 12 passed, 0
  failed.
- **The reference validator's verdicts** (`tools/documents/check_signed.py`, new: pyHanko with the test root alone, at the
  validation time, nothing fetched; B-LT again with the file's DSS alone, revocation required, at the timestamp's time):
  the 12 files the tests sign into `test-output/documents/signed` and `signed-encrypted` (13 signatures) are all as their
  JSON says: every signature intact, valid and trusted; B-B and B-T over the entire file; the two B-LT entire-revision with
  the DSS update a change of level none, trusted from the DSS alone; the first of `signed-twice` entire-revision at form
  filling, allowed; the timestamps valid and trusted; the encrypted file read with its user password. `check_written.py`
  on the 11 plain files: qpdf, pdfminer.six and pdfium open each without a warning.
- **Timings** (this container, 4 cores, other agents building beside it; the owner's devices give the real ones): the 13
  D12 samples' 14 signatures listed (opened, read, byte ranges, changes: the five that do not cover their file open their signed
  revision and compare it) cold 104 ms, warm 9.7 ms a pass (692 µs and 70.7 KiB a signature); checked (every step, the DSS read once a
  document) cold 46.5 ms, warm 12.3 ms (880 µs and 27.8 KiB a signature). A two-page document of 4,720 bytes signed at
  B-B: 9,933 bytes added, cold 53.5 ms (the process's first signature), warm 1.30 ms, 103 KiB allocated; at B-LT (a
  timestamp from the stand-in authority, then the DSS update): 47,643 bytes added, cold 14.9 ms, warm 2.82 ms, 296 KiB.
- **Left:** document timestamps written (B-LTA); appearance text in any script (D9's shaped appearances); a list's or a
  response's signature verified once a check, not once a certificate it is asked about; delta and indirect lists, name
  constraints and policy constraints (a critical one fails the path, named); DSA and EdDSA keys; freshness beyond a
  list's or a response's next update.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents pdf: signatures|documents pdf: edit|documents pdf: encryption|documents pdf: write|documents pdf: file|documents contracts|document layouts|outside plug-in|override loop:|kind table:|conformance kit:|public API|abstraction inventory|every environment variable read in src|the environment table names real commands|hidden boxing|test runner"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d12.txt
  $env:IDRAK_FILTER = "documents pdf: signatures: timings"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d12-timing.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  python -I tools\documents\check_signed.py test-output\documents\signed -o test-output\documents\signed-check.json
  python -I tools\documents\check_signed.py test-output\documents\signed-encrypted -o test-output\documents\signed-encrypted-check.json --password reader-2026
  ```

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

**As built (2026-11-19):**
- **The bench** (`src/Idrak.Cli.Commands/Commands/Measure/BenchDocumentsCommand.cs`, `DocumentBench.cs`): `idrak bench
  documents` (`b documents`), a command of two words beside `idrak bench` (bench kinds have no registry: the command's own
  pattern, as `cluster bench` and `tuning show`), through the public document API only (any registered format, writer and
  drawer). Cases, in order: `open`; `text-lines` and `text-content` (every page's text through the `lines` layout and
  with none); `text-parallel` (the same over `ComputeResources.ParallelOptions`: the files at once, or one file's pages);
  `images` (every image decoded); `draw` (a case a resolution); `write-image` (each file's images written as an image PDF,
  `Auto`); `write-vector` (each file's paths and text written back, each under its clips; text whose font the file does
  not embed in `--font`'s face, else left out and counted); `edit-incremental`, `edit-rewrite` (the information changed);
  `fill` (every text field); `sign` (PAdES B-B with the sample generator's test certificate, at its first day plus one, so
  a file signs to the same bytes); `check` (every signature `sign` made, against the generator's root). Each case runs
  once cold (its first run on a set; the first set's also compiles the case's code, its JIT time and methods recorded)
  and `--repeat` times warm, each run after a full collection: the time (files refused during it left out), the managed
  bytes allocated on every thread (`GC.GetTotalAllocatedBytes(precise)`, the sampler's own allocations taken off) and the
  process's working set at its highest (`Environment.WorkingSet` sampled every 10 ms on a thread of the run's own).
  Options and variables (the option winning): `--cases` (`IDRAK_BENCH_DOCUMENTS_CASES`), `--sets`
  (`IDRAK_BENCH_DOCUMENTS_SETS`: `samples` (its manifest's hostile samples left out), `large` (each member of the large set
  a set of its own: `deep-forms`, `huge-image`, `pages-3000`, `many-small`), or folders and files), `--dpi`
  (`IDRAK_BENCH_DOCUMENTS_DPI`, 150,300), `-n, --repeat` (`IDRAK_BENCH_DOCUMENTS_REPEAT`, 3), `--draw-files` and
  `--draw-pages` (`IDRAK_BENCH_DOCUMENTS_DRAW_FILES`, `_DRAW_PAGES`: 50 files spread evenly, 10 pages each), `--font`
  (`IDRAK_BENCH_DOCUMENTS_FONT`), `--certificate`, `--key`, `--chain`, `--trust`; `--short` (6 files spread evenly, 2
  pages each, one warm run, one resolution) and `--hash` (every run's output hashed: text, pixels, written bytes; a run
  that differs from the cold one is flagged, and `--compare` says which hashed cases changed against the saved ones:
  rule 71 as a tool). A file no format opens (a password, damage) is refused once and left out of every run.
  `--save NAME` and `--compare NAME` are `idrak bench`'s own (its `Compare`, `LoadSaved` and `SavedPath`, now shared):
  three results a case and set (time, allocation, working set), each with its cold value. The saved results record every
  variable the tool's table names for the command (the `IDRAK_DOCUMENT_*` ones with D7's drawer settings, the bench's,
  the CPU's) as the environment gives them, and the limits they come to here. Live progress: the case, set and run, the
  units done of the total, the rate and the time left, and the run of all. Help and the CLI README with generic examples.
- **The limits (C8), measured and settable:** `MaxStreamBytes` (a quarter of the memory free when a document opens, at
  most the largest array), `CacheBytes` (an eighth) and `MaxObjects` (one a KiB) from the CPU device's available memory
  within `ComputeResources.CpuMemoryLimit`, read once at open; `MaxDepth` is a count of levels, not a size (64 by
  default, `DefaultMaxDepth`; each level is also checked against the stack the reading thread has left), at most
  `MostNesting` (2^20, so the reader's counts of nested parts stay in an `int`). Each is set by its option and its
  variable: `IDRAK_DOCUMENT_MAX_OBJECTS` and `IDRAK_DOCUMENT_MAX_DEPTH` new (the stream's and the cache's existed). Two
  faults fixed: `CacheBytes = 0` read back as the measured size (0 meant unset), so no option could turn the cache off
  (D1's "read without a cache" read with it); now 0 keeps nothing, by option or variable (`IDRAK_DOCUMENT_CACHE_BYTES=0`),
  unset is measured. And a limit's message printed its measured value, so two hostile samples gave other issue text on
  each run (found by the reader's agent): the eight messages now name the limit, its option and its variable, never the
  measured number (`PdfFile`, `PdfBytes`, `PdfImages`, `PdfEditCopy`).
- **Memory bounded at any size** (group `documents workload:`, 4, `tests/Idrak.Tests.Library/DocumentWorkloadTests.cs`):
  the limits within a memory limit (64 and 192 MiB: exactly a quarter, an eighth, one a KiB) and from the free memory
  (between two readings), each by option and variable, bad values refused naming the variable, `MaxObjects=1000` tripping
  on `pages-3000` naming its variable, the deep forms (8, 16, 24 levels) read within 64 and 24 refused under 20;
  `pages-3000` read page by page (text and images, every page's text against the generator's hash) under the measured
  cache, a 2 MiB one and none: the managed memory after every page, a full collection every 250 pages, grows at most the
  cache's held bytes and one page (1.09, 1.08 and 0.00 MiB from page 20 to 3,000; the cache holds 1.64 MiB: estimates
  that hold the real memory); written back page by page (images and text in a test face): -4.0 MiB from page 20 to 3,000,
  30 KiB allocated a page; 600 pages drawn: 0.76 MiB; the huge image's page drawn: 21.9 MiB besides the decoded image and
  the page's pixels (1.2 GiB more before the decoder's fix below); 1,000 small files read in parallel within
  `MaxCpuThreads` (4 at once; 2 when set to 2), the same text, 2.68x one by one; `pages-3000`'s pages in parallel 3.54x
  (rule 79: it pays, here).
- **Slow paths the bench found, fixed, the outputs proven the same** (`--hash` runs before and after compared:
  `images` and `write-image` on the samples, `huge-image` and `pages-3000`; `write-vector` and `write-image` on the samples,
  `pages-3000` and `many-small`; every hashed case the same, and the `jpeg:` group's Pillow-exact decodes passing):
  - **The JPEG decoder held the image two and a half times** (`src/Idrak.Imaging/JpegCodec.cs`): every coefficient
    (2 bytes a sample) and full component planes before the output, all rented from the shared pool, which kept them.
    Now the image is made a strip of MCU rows at a time (enough rows for two 16-row bands a thread): each component's
    window of sample rows after the row above (the fancy upsampling's context), the output rows the strip completes
    converted at once; a sequential file whose one scan holds every component (no other scan after it) also decodes its
    coefficients a strip at a time. Progressive and multi-scan files keep their coefficients and convert by strips. The
    huge image (139.5 Mpx): cold 1,555 to 411 MiB allocated, the process's working set after it 1,140 to 516 MiB, warm
    decode the same (230-250 ms); drawing its page at 36 dpi: 1.36 GiB to 427 MiB.
  - **Writing an image held it twice more** (`src/Idrak.Data/Pdf/PdfWriterImages.cs`): `Auto` copied the image whole,
    interleaved, to measure it, and Flate filtered the whole copy into a second one. Now the measures are made a band of
    rows at a time (no copy; the same measures, checked against the whole-image ones on every D6 writer sample), and a
    Flate image holds one copy, the stream the deflater takes in one piece, filtered band by band. The huge image
    (Flate by `Auto`): cold 1,432 to 945 MiB, working set 1.27 GiB to 936 MiB, 5.5 to 5.0 s, the same 11,553,322 bytes.
  - **The writer matched every text run's glyphs to its characters** (`PdfWriterFonts.Map`): lists, a sort and a string a
    glyph, though only glyphs not met before take a text. A run of known glyphs is no longer matched (the same map, the
    same answer). `write-vector` on `pages-3000` (39,000 runs): 1,061 to 84.9 MiB allocated, 1,242 to 788 ms.
- **Found in the other agents' files, reported, not fixed:** the drawer keeps a band-sized float coverage mask for every
  clip of a band (`DocumentScene.ClipMask`), and the reader gives each form's `/BBox` as a clip of its own
  (`PdfContent`): `deep-forms`' page drawing one form 1,000 times allocates about 372 MiB a draw at 150 dpi and raises the
  working set by 490 MiB (its other three pages 7.8 MiB together). Masks over a clip's own box, identical clips shared, or
  a mask dropped after the last part that uses it would each bound it. **Fixed 2026-11-19:** see "Clip memory,
  2026-11-19" at the end of this phase (that page: about 380 to 3.7 MiB a draw at 150 dpi).
- **Decisions the plan did not settle, taken here:** (1) The bench is a command, `bench documents`, not a kind of
  `bench`'s positional (a model named `documents` stays possible). (2) The large set's members are sets of their own, so
  each workload is timed apart; the samples are one set. (3) Drawing takes at most 50 files of a set and 10 pages of a
  file by default (every page of 1,000 small files at 300 dpi is minutes, not a workload). (4) `MaxDepth` stays a count
  (64), settable, not a share of the memory: nesting costs stack, which each level checks. (5) A fully streamed Flate
  image (a band, not one copy) is not built: .NET's zlib-ng writes other bytes when its input comes in other pieces
  (measured: pieces of 32 KiB multiples from 64 KiB match one piece at levels 4 to 9, level 1 never), so the written files
  would change (rule 71); the owner decides (leftovers 28.24).
- **Tests:** group `documents bench:` (2, `tests/Idrak.Tests/DocumentBenchTests.cs`): every case on the samples in the
  short mode with `--hash` (cold and warm apart, the same output every run, the variables and measured limits recorded,
  `IDRAK_DOCUMENT_CACHE_BYTES=64m` read as such), `fill` on the form samples, the text output; `--save` and `--compare`
  (each case's three results, cold beside warm, the outputs the same), the variables under their options, usage errors.
  `cli measure:`'s help count 10 to 11; D6's image test checks the band-wise measures. Runs (each with `--no-build`, CPU):
  the brief's filter with `cli measure:`, 120 tests, 2 failed (the option `CacheBytes = 0` read as measured; the huge
  page allocating 1.64 GB: the decoder) and fixed; after moving onto `6153b50` and `6765477` and the fixes above, with
  `jpeg|image transforms|image data:|image preprocessing:` added for the decoder: 140 passed, 0 failed (3 `documents
  real:` skipped without a folder), and again on the final code (after the writer's map): 140 passed, 0 failed. Two builds
  failed once each (a local named twice in nested scopes; spans in a tuple), each fixed and rebuilt once. The API dump
  (`DocumentOptions`' four constants) regenerated through the runner; the inventory unchanged; both pass without the
  flags.
- **Timings** (this container, 4 cores, other agents building beside it: noisy; the owner's devices give the real
  baseline; `test-output/bench-cache/bench/d13-final.json`, the whole run 75 s): the samples (100 files opened, 13
  refused for a password or an unsupported handler; 175 pages): open warm 11 ms (113 µs a file), cold 93 ms (80 ms
  compiling); text-lines 44 ms (252 µs a page, 4.2 MiB), text-content 41 ms, text-parallel 22 ms; images 86 ms (80 images,
  10.5 Mpx); 79 pages drawn at 150 dpi 380 ms (4.8 ms a page), at 300 dpi 1.29 s (16 ms); write-image 250 ms,
  write-vector 117 ms; incremental save 74 µs a file, rewrite 203 µs; 7 forms filled 1.2 ms each; signing 0.87 ms a file,
  checking 1.2 ms a signature. `pages-3000`: opened 6.4 ms (10.3 MiB: the page tree), text 67 µs a page (57 without a
  layout, 21 over 4 threads), images 54 µs a page, write-image 158 µs a page, write-vector 265 µs a page (85 MiB),
  incremental save 8 ms with the open, rewrite 37 ms, signed 16 ms, checked 8 ms. `many-small` (1,000 files): open 11 µs
  a file, text 59 µs (19 over 4 threads), write-vector 534 µs a file (228 MiB: a font subset a file), signing 557 µs,
  checking 651 µs. `huge-image`: decoded 237 ms (400 MiB, the planes), drawn at 300 dpi 1.45 s (800 MiB: the planes and
  the page), written by `Auto` (Flate) 4.9 s. The process's working set at most 1.7 GiB (the huge image's runs).
- **Left:** the drawer's clip masks (above; fixed, "Clip memory, 2026-11-19" below); a fully streamed Flate image (decision 5, leftovers 28.24); `write-vector`
  of many small files spends most of its 228 KiB a file on each file's font subset and its first runs' matching.
- **The owner's real files** (the plan's third bullet): built with D5's `documents real:` group (6765477), which writes
  each file back, edits, draws and times it.
- **The owner's run** (held with the others until asked):
  ```powershell
  cd D:\Projects\Idrak; git checkout documents; git pull origin documents
  dotnet build -c Release tests\Idrak.Tests
  python -I tools\documents\make_fixtures.py -o test-output\documents\samples --large
  dotnet run -c Release --no-build --project src\Idrak.Cli -- bench documents --font tests\Idrak.Tests\data\fonts\dejavu-sans-subset.ttf --cache test-output\bench-cache --save owner -O test-output\bench-documents.txt
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents|fonts write|cli measure:|cli polish:|cli arabic:|public API|abstraction inventory|every environment variable read in src|the environment table names real commands|hidden boxing|test runner|jpeg|image transforms|image data:|image preprocessing:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-d13.txt
  $env:IDRAK_FILTER = "documents workload:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\d13-workload.txt
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

- **Clip memory, 2026-11-19** (D13's deep-forms finding, fixed in `src/Idrak.Imaging/DocumentScene.cs` alone; the
  reader unchanged: a form's `/BBox` already reads as a plain `m l l l h` path).
  - **Where the bytes went:** 355 of the 358 MiB a warm draw of deep-forms' fourth page allocated at 150 dpi were
    `float[]` (the runtime's allocation ticks): each band (875 x 149 pixels) rented a band-sized mask for every clip one
    of its parts used and kept it to the band's end, about 260 at once, far more than the shared pool holds.
  - **The change:** a band works a clip's coverage once over the clip's own box (`ClipCover`, `ClipBox`): the library's
    rasterizer works each row and column in the outline's own frame, so a fill over the box's columns and any of its rows
    is that box of the band-wide fill, value for value; within the parent's box, times the parent's values. An upright
    rectangle (a `re`, a form's box at a right angle: four lines exactly level and upright) fills only the rows that
    differ, its first, its last and one between (every row between gets the same values), each under its parent's row.
    A part's colours are made and composed within its clip's box alone (a part wholly outside it still renders, on an
    empty box, so what it decodes and reports is as before). Each clip goes back to the pool after the last part that
    needs it, directly or under a clip made within it (the release points worked out once a scene), so a band holds the
    clips in use: a few a nesting level. An app's rasterizer keeps the band-wide masks (with the same release), so its
    clips are filled as they always were. No setting: nothing here is a threshold, and no drawing changes.
  - **Numbers** (this container, 4 cores; the owner's devices give the real ones). Deep-forms' fourth page, three
    processes each, before and after interleaved, the page read each draw: 150 dpi cold 176-190 to 79-86 ms, 473-481 to
    21.7 MiB, the working set +469-478 to +16 MiB; warm 45-50 to 19-25 ms, 378-383 to 3.7 MiB a draw (1.56 MiB the
    pixels), the working set +10-55 to +2.6 MiB. 300 dpi cold 164-180 to 92-97 ms, 255-368 to 26.4 MiB, +249-362 to +21
    MiB; warm 64-75 to 32-37 ms, 229-265 to 8.4 MiB a draw. The bench's `draw` (`--hash`, `--save`, `--compare`):
    deep-forms' four pages at 150 dpi warm 51.2 to 14.4 ms and 345 to 8.3 MiB, cold 130 to 15.6 ms and 477 to 8.6 MiB;
    at 300 dpi 85.2 to 49.9 ms and 242 to 27.1 MiB; the samples, the huge image, `pages-3000` and `many-small` the same
    within the noise (the samples at 150 dpi 470 to 458 ms, at 300 dpi 1,370 to 1,380 ms). Large clips on a letter page
    of 300 blocks, each case alone in its process (150 dpi, warm): one page clip 11.3-11.6 to 10.5-10.9 ms (no clip
    10.3-11.2 to 9.0-9.1); a large upright rectangle a block 38-46 to 13-15 ms; a large turned one 42-43 to 38-40 ms; a
    small rectangle a block under a page clip 26-27 to 12-16 ms (300 dpi: 72 to 33-36, 73-75 to 59-69, 55 to 31-32).
  - **The same drawings, proven:** 1,341 drawings hashed (planes and issues, SHA-256) with the build before and after:
    every sample PDF (102 files drawn, 177 pages) and the large set (deep-forms' four pages, `pages-3000`'s first two,
    the huge image's page, every 25th of the 1,000 small files), at 150 dpi RGB with the fonts folder, 72 dpi grey, 300
    dpi with annotations, 150 dpi in one-row bands, 100 dpi without fonts and with hidden text, and 150 dpi through an
    app's rasterizer (the band-wide path), the reader's limits fixed by their variables (measured from the free memory,
    they decide whether the hostile flate bomb is drawn): every hash the same, the same 12 refused; the tests' crafted
    page and clip pages (49 drawings in seven settings, bands of one, seven and all rows among them) the same; the
    bench's 10 hashed `draw` cases the same.
  - **Found, left as it was (the owner decides; fixed since: "Text clips in every band, 2026-11-19" below):** a clip of
    text whose runs were not read clips nothing of its own and answers its parent's coverage only the first time a band
    asks, nothing after, so a part or a clip under it is cut by the parent or not by where the bands fall: the crafted
    page differs between the default bands and one band at 3,150 values, in both builds. Answering the parent every time
    would change such drawings, so it is not done here.
  - **Tests** (`tests/Idrak.Tests.Library/DocumentDrawClipTests.cs`, group `documents drawers`, 2 new): pages of 11,
    101 and 1,001 clips as the pdf writer writes them (siblings under a page clip; upright, turned, three nested)
    render every band within `ClipRenderBytes` (64 KiB) past the pools whatever the count (seen 2.2, 2.2 and 3.7 KiB),
    each clip adds at most `ClipSceneBytes` (1 KiB) to a draw (seen 474-551 B), and deep-forms' page, when the large set
    is there, within `DeepFormsDrawBytes` (8 MiB) beyond its pixels (seen 1.06 MiB); a clip's coverage over its own box
    equals the band-wide one value for value (13 clips alone and 13 chains, bands of 1, 7, 64 and 83 rows: 4,214
    coverages, a rectangle's rows shared 185 times), and a crafted page of every kind of clip draws the same bytes and
    issues through an app's rasterizer at 150 dpi, 2 pixels a point in grey, one-row bands and one band. **Run:** the
    filter `documents drawers|documents pdf: vector|documents pdf: forms|documents pdf: annotations|documents
    workload:|documents bench:|documents contracts|outline tracers:|fonts:|pages:|shapes:|glyph poses:|public API|
    abstraction inventory|every environment variable read in src|the environment table names real commands|hidden
    boxing|test runner`: 114 passed, 0 failed (one build, no error or warning). No public API, inventory or variable
    changed.
- **Text clips in every band, 2026-11-19** (the clip-memory note's finding, fixed in
  `src/Idrak.Imaging/DocumentScene.cs` alone):
  - **The cause:** a band cached a clip of text whose runs were not read (its font neither embedded nor found, its runs
    left out by `IncludeHidden = false`, or runs the page does not have) as clipping nothing once it had answered the
    clip it was made under the first time, so in each band only the first part to ask (directly, or through a clip made
    under it) was cut by that clip, and where the bands fell decided which.
  - **The fix, the drawer's documented rule kept:** such a clip clips nothing of its own (its glyphs are not known); a
    text clip whose runs were read but hold no glyphs (spaces) still lets nothing through, as PDF has it. The scene puts
    every part and clip made under it under the clip it was made under, once a scene, and drops it, so no band asks for
    it. A page without one is left as it is; the clips' own boxes, a rectangle's shared rows and the early release are
    unchanged (deep-forms' fourth page at 150 dpi: 3.75 MiB a draw either way, warm 18.6-20.3 to 19.4-22.9 ms, noise).
  - **Drawings:** the 1,341 drawings of the clip-memory note (every sample and the large set, six settings, the reader's
    limits pinned) hash the same before and after, the same 12 refused: no sample has an unread text clip under another
    clip with a part under it. Changed: the tests' crafted page (default bands and one band differed at 3,150 values at
    150 dpi, one-row and seven-row bands from the default at 1,000 at 2 pixels a point) and a new page of text clips (up
    to 9,984 values between bands); each now draws as its lifted page (each unread text clip's parts and clips under the
    clip it was made under), which drew the same before and after, in every band and through an app's rasterizer too.
    Neither has a pdfium reference (pages made in memory).
  - **Tests** (`tests/Idrak.Tests.Library/DocumentDrawClipTests.cs`, group `documents drawers`, 1 new): the crafted
    pages' test draws the crafted page and a page of text clips (left out or in a font not there, at the top, nested
    under and over rectangles, a turned one, a circle, an even-odd frame and each other, under a text clip whose glyphs
    were read and one of spaces, in a soft mask) with default, one-row and seven-row bands and one band, through the
    library's rasterizer and an app's, each the same bytes and issues as its lifted page; the case that failed is a
    named regression (`documents drawers: a text clip whose runs were not read answers the clip it was made under every
    time, in every band`). **Run:** the filter `documents drawers|documents pdf: vector|documents pdf: forms|documents
    pdf: annotations|documents workload:|documents contracts|outline tracers:|fonts:|pages:|shapes:|glyph poses:|public
    API|abstraction inventory|hidden boxing|test runner`: 111 passed, 0 failed (one build, no error or warning). No
    public API, inventory or variable changed.
- **Bench runs comparable, 2026-11-19** (`BenchDocumentsCommand.cs`, `DocumentBench.cs`; `documents real:`'s
  `DocumentRealTests.cs`; no library change).
  - **The false alarm's cause:** not the memory. A capped case (draw's 50 files, `--short`'s 6) spread its cap over the
    files not refused yet, so the refusals one run learned moved the next run's spread. The samples drawn at 150 dpi
    alone (`--cases draw`): the cold run took 50 files spread over 113 and refused 5 (a password, an unsupported
    handler), drawing 45; the first warm run spread over the other 108 and refused 5 more (the `-user` encrypted
    samples), the last drew 49 (89 pages): other files in every run, "OUTPUT DIFFERS BETWEEN RUNS" on every build. With
    `open` first (the whole bench) the refusals were known before a capped case's cold run, so it showed when a capped
    case was its set's first.
  - **The change:** a cap spreads over every file of the set, refused or not, the refused ones then left out, so every
    run of every case at one cap takes the same files (the samples drawn: 45 files, 85 pages, refused in the cold run
    alone). The limits measured from the memory (`MaxStreamBytes`, `MaxObjects`, `CacheBytes`) are resolved once when
    the bench starts (`DocumentBench.Pin`, the variables winning) and every run of every case opens with them; the saved
    `settings.limits` are those values (they were read again for the record, not the ones used). With `--hash` each
    file's output is hashed too (`file_outputs`, the cold run's; `limit_files`, those that met a limit: refused by one,
    or a page's issue naming one): a case whose runs differ names each file and its hash or refusal in every run
    (`differing_files`; at most 20 lines on screen). `--compare` notes the limits other than the saved ones (each, now
    and saved), names the files of a changed case and says whether the other limits explain it (every changed file met a
    limit in one of the two runs, in the case or refused when its set's files were opened); a changed case is listed as
    changed either way. Results saved before compare as before, their files not named. `documents real:` reads every
    file, pass and read-back under limits pinned once a run (its write-back check compares two opens), and its summary
    names them.
  - **The run** (this container, 4 cores; `--hash`, every case, the samples and the large set, `--save`, then
    `--compare`): 70 cases, 67 the same output every run and the same as saved; both runs under the same limits (a
    stream 2.00 GiB, a cache 1.25 GiB, 10,502,019 objects, nesting 64). The samples' draw cases the same every run (150
    dpi: 85 pages, 45 files, warm 427-433 ms; 300 dpi: 1,315-1,325 ms).
  - **Found by the files' hashes, left as it is (the owner decides):** `edit-incremental`, `edit-rewrite` and `sign` on
    the samples differ in every run, in the AES-encrypted samples opened with the empty password alone (6, 7 and 6
    files: `d2/r4-aes-128-*`, `d2/r5-aes-256-empty`, `d2/r6-aes-256-*`; the RC4 and plain files repeat): each save
    encrypts with a fresh random IV (ISO 32000-2; D11's decision 1: "an encrypted output is the one output that does not
    repeat byte for byte"). Not a fault; the bench names them in every run and every comparison. Hashing what such a
    file reads back to, not its bytes, would make those cases repeat; not done, since it changes what the bench checks.
  - **Tests** (group `documents bench:`, 2 new, 4 in all): the pinned options measure nothing when a file opens (their
    text names nothing "measured") and every run reads under the bench's (a 64-byte stream limit pinned there stops
    `d3/encodings.pdf`'s 205-byte content, naming the limit; the measured one reads it); the saved limits are the
    variables' (300m, 2,000,000, 0); the samples drawn at 36 dpi (`--cases draw`, the cold run refusing the encrypted
    files) give the same output every run, file by file; a file whose output differs between runs (made in the test) is
    named with its case and each run's hash or refusal. `--compare`: a save under a 1 GiB stream limit against 2 GiB
    notes that limit alone, the outputs the same; a file refused under 64 bytes and read under 1 GiB is a change the
    other limits explain; a deliberately altered saved output is a change naming its file, under the same limits ("read
    under the same limits") and under other ones ("NOT explained by the other limits"). The short-mode test also checks
    every case's files' hashes. **Run:** the filter `documents bench:|documents real:|documents workload:|documents
    contracts|cli measure:|cli polish:|cli arabic:|public API|abstraction inventory|every environment variable read in
    src|the environment table names real commands|hidden boxing|test runner`: 53 passed, 0 failed, 3 skipped (`documents
    real:` without a folder). The build failed once on three nullable errors in the new tests (result tuples starting as
    `default`), fixed together and rebuilt once: no error or warning. No public API, inventory or variable changed.

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

## Planned follow-ups, built only when the owner says

- **`data synth` writes PDFs** (**built 2026-11-19**, as-built note below; owner, 2026-11-18: "add it to plan but don't
  implement it unless I tell you"): today `idrak data synth` writes each page as an image (JPEG by default, any
  registered `ImageEncoders` format) with its label in the `.jsonl`. With D6's writer it could also write PDFs:
  `--format pdf`, the drawn page (scan look and all) as an image page with the label's text as a hidden, searchable
  layer placed on the drawn runs (a scan-like PDF); `--format pdf-vector`, the page written before any scan look as real
  text and shapes (`DocumentShapes`, the page's fonts embedded as subsets: a PDF as software makes it); the records
  point at the PDF and its page. Through `DocumentWriters` only (any registered writer), the options carried to the
  command and its variables. **Built last, without waiting for the owner** (owner, 2026-11-18: "keep it in the plan but
  it's the last to implement … if the plan completed then start the pdf output directly don't wait for me"): it starts
  on its own the moment every other plan 28 phase and every queued fix is merged and the slots are free, and on nothing
  earlier.

  **As built (2026-11-19):**
  - **Turned glyphs, a contract change** (`src/Idrak.Abstraction/Data/DocumentWriters.cs`):
    `IDocumentPageBuilder.Glyphs(face, glyphs, matrices, text, mode, colour)`, a default member: one matrix for all the
    glyphs or one each, each mapping the glyph's em onto the page as `DocumentTextRun.GlyphMatrix` reads it back;
    upright matrices at one size go to the call with a size (the same bytes), any other is a `NotSupportedException`
    from a writer without it; a wrong count, a flat or an unfinite matrix an `ArgumentException`; the shadow builder
    forwards it. The pdf writer (`PdfWriterPage.cs`) gives each turned glyph its own text matrix (the matrix over its
    size, the size its scale along the baseline) and, when the run's glyphs do not read back as its text, each glyph its
    own `/ActualText` span with its share of the text (from the run's alignment, `PdfWriterFont.Shares`), so the reader
    makes each glyph a run of its own drawn through its own matrix (a span across turned glyphs would read as one run
    drawn through the first's).
  - **`ShapePlan.Texts` written** (the gap D6 left): `DocumentShapes.Write(page, plan, style, lettering, pageText,
    rightToLeft)` writes the areas and paths, then each text as the page draws it: `ShapeDrawing.Place`, now shared by
    the drawing and the writer (the same numbers: the image unchanged), says the text, lays it out, fits it and turns
    each glyph onto the path at its middle; the writer gives each glyph (spaces too, at the middle of their advance) its
    origin and its em turned onto the path, in reading order (by cluster), a call a stretch of one face, the characters
    of their clusters as text. `DocumentShapeStyle.Left` and `Top` put a page cut from a larger drawing at its corner;
    `TextBlock.Text` keeps the text a block was laid out from. Posed glyphs (`pose=`) are written through their poses'
    matrices (moved, height scaled, slanted, turned) in their ink's share.
  - **The page as drawn** (`PageStyle.Vectors`, `RenderedPage.Vectors`, `IPageVectors` in `Pages.cs`; `PageRecord` in
    `src/Idrak.Imaging/PageVectors.cs`): letter, plain and shapes record what they draw when asked, nothing copied: each
    text block with its glyph painter and first line number (its reading order: drawn order, or the shapes layout's
    group, index, part), each shape plan with its stroke width and lettering (the stamp's plans turned, at the page's
    place), each rule, each in its layer, and at the end the colours `Compose` used (the paper, the ink as the paper
    darkened, the second ink; grey on a grey page). `Write` writes the paper (a filled rectangle unless white), the
    ink's parts in drawing order, then the second ink's (the page lays it over the ink); `WriteText` the page's own text
    alone in reading order (none of the ornaments'). A pen's wobble and pressure and a stamp's uneven ink are written as
    the ideal stroke and the full ink. Asked for nothing, the layouts draw as before: the colours are drawn into locals
    in the order `Compose` took them, every record call is skipped.
  - **`data synth`** (`SyntheticPages`, `src/Idrak.Text/SyntheticDocuments.cs`, `DataSynthCommand.cs`):
    `SyntheticPageOptions.Output` (`Images`, `ScannedDocuments`, `VectorDocuments`), `Writer` ("pdf", any
    `DocumentWriters` name, its first extension naming the files), `PagesPerDocument` (1: the plan named no other),
    `Dpi` (0: the drawer's `IDRAK_DOCUMENT_DPI`, else 150, since one resolution maps pixels and points both ways; the
    default page, 1240 pixels, is A4's width) and `WriteOptions` (the writer's variables unless given). A scan-like page
    is its image (the look and all; Auto, the writer's default, keeps most text pages exact: their paper is flat) with
    the label's runs hidden on the glyphs they were drawn with (a cut page's by the cut's corner, a look that changes
    the size scaled with it); a vector page is the page before any look (no look is drawn for it: its manifest says
    `none`). A page alone is its own document, written whole where it is made, beside the others, as an image is encoded
    there; several pages a file are written in order by one writer (a file named after its first page, finished when it
    holds its pages) while the next chunk is made, the pages waiting as 8-bit planes. Each record names its document as
    its image and the page's index in it, from 0, in `"pages"` (beside `"images"`); its manifest line `"image"` and
    `"page_index"` (`"page"` is the layout's description). Options: `--format pdf|pdf-vector` (claimed over the common
    `--format`), `--writer`, `--pdf-pages`, `--pdf-images auto|flate|jpeg`, `--pdf-fonts subset|whole`, `--dpi`,
    `--quality` (in a PDF, over `IDRAK_DOCUMENT_JPEG_QUALITY`); each refused out of range naming itself, an unregistered
    writer naming the registered ones, the document options without `--format pdf*` and `--pdf-images` on vector pages
    refused; a layout that keeps no vectors fails naming the library's that do. `IDRAK_DOCUMENT_IMAGE_ENCODING`,
    `_JPEG_QUALITY`, `_EMBED_FONTS` and `_DPI` list `data synth` in the environment table (and so in its help); the CLI,
    Imaging, Data and Text READMEs say it.
  - **Results proven:** the image output byte for byte: three fixed-seed runs (jpeg with every layout, emblems and
    stamps; png grey and colour, cut pages; a zip of shapes pages) with the build of 7b65d25 and with this one: the 21
    files the same SHA-256, the zip's 6 entries the same bytes (only the entries' times differ, as between any two
    runs); a test runs jpeg and png with the vectors kept and without (the same images, records, manifests). The same
    seed writes the same PDF bytes twice, on one worker or many and into a zip; a page alone and the same page through
    the ordered writer give the same bytes.
  - **Tests** (`tests/Idrak.Tests/SyntheticPdfTests.cs`, group `synthetic data pdf:`, 6; and one each in `documents pdf:
    write` and `shapes:`): scan-like pages on letter, plain and shapes and two seeds (the page's size, the image exact
    or the JPEG round trip, the hidden glyphs read back in order where written, each run inside its drawn box within a
    pixel, cut pages, the text where the ink is, every record's file and page opened); vector pages (every glyph read
    back in order with its text, the label's runs the label, the stamps' words written, subset fonts, the page drawn
    back within 2% of the page before its look: 0.00 to 0.01% measured); several pages a file, the same bytes, a zip;
    memory over 300 pages; image output unchanged; the CLI's options and refusals; turned glyphs written and read back;
    every emblem and stamp profile's texts read back as the plan says them and drawn back within 1% of the page drawing.
    **Runs:** the filter below with the API and inventory flags: 147 tests, 145 passed, 2 failed: the shared-contracts
    list (the writing contracts now used by Idrak.Text too: their four lines removed) and the scan test's own comparison
    (a drawn page's table of a byte's value is k / 255, the reader's k x (1 / 255): compared as 8-bit levels); then
    those, the inventory and the API without the flags: 9 passed. Then two speed changes (a chunk written while the next
    is made; a page alone written where it is made), each built once and run with `synthetic data|public API`: 17
    passed, twice.
  - **The reference tools** (`tools/documents/check_written.py --expected auto`, the expected text each page's label, on
    15 files in `test-output/synth-pdf/`: six scan-like, six vector, a vector file of six pages, two scan-like files of
    three, from generic letters in DejaVu Sans): qpdf finds no damage and decodes every stream, pdfminer.six and pdfium
    open every file with no warning; 11 match the label exactly in both. Of the 24 pages, pdfium gives the label exactly
    on all 12 scan-like pages and pdfminer.six on 11 (one header in its own order); on vector pages both give the label
    in order with the ornaments' words beside it (a ring's name, a monogram's initials, a stamp's word), or exactly
    where a page has none; pdfminer.six breaks a ring's name into pieces (its layout of turned glyphs), pdfium reads it
    whole. No word of a label is missing in either.
  - **Timings and memory** (this container, 4 cores; the owner's devices give the real ones): the writer alone, 620 x
    877 pages into one 300-page document, after a full collection every page: scan-like 66 KiB a page, cold 32-36 ms a
    page (the first 9), warm 16-18 ms and 113 KiB allocated a page, the managed memory flat (0.3 MiB over 300 pages);
    vector 2.7 KiB a page, cold 0.5-1.0 ms, warm 0.44-0.77 ms and 115 KiB a page, flat. Whole runs at A4 150 dpi, 90
    pages, under a 900 MiB heap limit (each run's peak resident about 940 MiB, the same for jpeg: the chunks' drawing):
    jpeg images 5.7 s, scan-like PDFs a page a file 9.6 s (12.5 s before a page alone was written where it was made), 90
    pages in one file 10.7 s, `--pdf-images jpeg` 6.4 s; vector PDFs 0.9-1.0 s (no look, no image). Without the limit a
    90-page file's peak was 1.9 GiB against 1.4 GiB a page a file: garbage the GC had not taken, not memory kept.
  - **Left for later:** a right-to-left ring text in a reference reader (pdfium gives turned runs in the glyphs' order);
    an app's layout keeps no vectors unless it fills `RenderedPage.Vectors`; a stamp's uneven ink and a pen's wobble in
    vector pages; the JPEG pages could take `--quality` alone without Auto's measure (`--pdf-images jpeg` does).
  - **The owner's run** (held with the others until asked):
    ```powershell
    cd D:\Projects\Idrak; git checkout documents; git pull origin documents
    dotnet build -c Release tests\Idrak.Tests
    $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "synthetic data|pages:|composed pages:|shapes:|glyph poses:|scans:|fonts:|documents pdf: write|documents pdf: text|documents drawers|documents contracts|kind table:|conformance kit:|cli polish:|cli arabic:|public API|abstraction inventory|every environment variable read in src|the environment table names real commands|hidden boxing|test runner"
    dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-synth-pdf.txt
    $env:IDRAK_FILTER = "synthetic data pdf: memory"
    dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress | Out-File -Encoding utf8 test-output\synth-pdf-memory.txt
    Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
    New-Item -ItemType Directory -Force test-output\synth-pdf\fonts
    Copy-Item tests\Idrak.Tests\data\fonts\dejavu-sans-subset.ttf test-output\synth-pdf\fonts\
    dotnet run -c Release --no-build --project src\Idrak.Cli -- data text -n 6 -l en --seed 1 -o test-output\synth-pdf\letters.txt
    dotnet run -c Release --no-build --project src\Idrak.Cli -- data synth test-output\synth-pdf\letters.txt -o test-output\synth-pdf\scan\pages.jsonl --fonts test-output\synth-pdf\fonts --format pdf --layout letter --layout plain --scan office --scan none --page "emblem=1,stamp=1" --seed 7
    dotnet run -c Release --no-build --project src\Idrak.Cli -- data synth test-output\synth-pdf\letters.txt -o test-output\synth-pdf\vector\pages.jsonl --fonts test-output\synth-pdf\fonts --format pdf-vector --pdf-pages 6 --layout letter --layout shapes --page "emblem=1,stamp=1" --seed 7
    ```
    The two runs write `test-output\synth-pdf\scan\pages\*.pdf` (a scan-like PDF a page: select the text on the page
    image) and `test-output\synth-pdf\vector\pages\*.pdf` (three files of six vector pages: the emblems' and stamps'
    words are text too), each with `pages.jsonl` naming the file and page of every record.

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
# The owner's own PDFs (a folder of real files; nothing is written into it), read, drawn, written back and timed: the
# summary goes to test-output\documents\real (D5's third piece); the reference text first, from the same folder.
python -I tools\documents\reference_text.py D:\Documents\pdf-samples
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents real:"; $env:IDRAK_DOCUMENTS_REAL = "D:\Documents\pdf-samples"; $env:IDRAK_TIMEOUT = "7200"
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-real.txt
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER, Env:IDRAK_DOCUMENTS_REAL, Env:IDRAK_TIMEOUT
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
