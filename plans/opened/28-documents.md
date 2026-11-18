# Plan 28: documents (a document contract, PDF read and written built in, vector and image pages, Office as a plug-in later)

**Not started.**

**Status (2026-11-18, `abstraction`):** planned 2026-10-22 against the code at 99872b8; widened 2026-11-18 against the
code at da19bfc by the owner's decisions 7-9 (PDF read **and written**, vector **and** image pages, finished for good
before plan 38). The PDF parts of [plan 38](38-handwriting-fonts.md) moved here (see "Plan 38's needs"). **Nothing is
built.** No phase starts before the owner names it.

| Phase | What | Lane | State |
|---|---|---|---|
| D0 | The contract and its registries (`IDocumentFormat`, `DocumentFormats`, `IDocumentTextLayout`, `DocumentTextLayouts`, `IDocumentWriter`, `DocumentWriters`), their suites, a test format from outside proving the plug-in path | A | Planned |
| D1 | PDF, the file: objects, cross-reference tables and streams, object streams, incremental updates, repair, filters, limits | A | Planned |
| D2 | PDF encryption: the standard security handler (RC4, AES-128, AES-256), an empty or a given open password | B | Planned |
| D3 | PDF text: pages, the content-stream interpreter, fonts to Unicode, glyph widths, text runs with boxes, the `lines` layout | A | Planned |
| D4 | PDF images and embedded files (a scanned PDF's pages as images) | B | Planned |
| D5 | Fixtures and their generator, READMEs, the owner's runs on real files | A | Planned |
| D6 | PDF writing: pages of images, vector paths and text (fonts embedded, searchable), several pages, the same bytes from the same input | B | Planned |
| D7 | PDF vector content read, and any page drawn to pixels (paths, text through the embedded fonts, images, clipping) | A | Planned |

## Goal

The library reads **and writes** documents through one contract, the way it reads and writes images through
`ImageCodecs` and `ImageEncoders`: a format registered by name, chosen from a file's first bytes, opened page by page,
giving text with its place on the page, vector paths, images and embedded files, and any page drawn to pixels; and a
writer that makes pages of images, vector paths and text. PDF is the library's default, both ways, for both kinds of
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

**Proposed, for the owner to confirm:**

5. ~~Scope of the first version: text and images, not rendering.~~ Superseded by decisions 7-9: reading covers text,
   images, vector paths and drawing a page to pixels; writing covers images, paths and text. Still out: forms,
   annotations, signatures and editing a file in place (see "Not in this plan", and the open questions).
6. **The contract, the PDF reader and the PDF writer** ("just implement the contract"): no CLI command, no dataset rows,
   no retrieval chunking from documents here, except what plan 38 needs (its commands are plan 38's F6). Other users of
   the contract are listed under "Not in this plan".

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

- **Contract plus registry (decision 10).** `DocumentFormats`, `DocumentTextLayouts` and `DocumentWriters` are `SlotTable`s with
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

The contract is format-neutral: a Word file has no fixed pages or boxes, so an Office plug-in gives one page per section
(or one for the whole file), runs without boxes (`Box` empty) and `Lines` from its paragraphs. Nothing in C1-C17 names
PDF.

**Settings:** the options in C8, C15 and C16, with `IDRAK_DOCUMENT_LAYOUT`, `IDRAK_DOCUMENT_READING_ORDER`,
`IDRAK_DOCUMENT_MAX_STREAM_BYTES`, `IDRAK_DOCUMENT_CACHE_BYTES`, `IDRAK_DOCUMENT_IMAGE_ENCODING`,
`IDRAK_DOCUMENT_JPEG_QUALITY`, `IDRAK_DOCUMENT_EMBED_FONTS`, `IDRAK_DOCUMENT_DPI` and `IDRAK_DOCUMENT_FONTS` read by the
library (names proposed), added to the
CLI's variables table (`src/Idrak.Cli.Commands/Shared/EnvironmentVariables.cs`, which a test checks). Defaults: the stream limit and
the cache from the machine's measured memory, never a fixed size in GB.

---

## Phase D0: the contract and its registries

- C1-C17 in `Idrak.Abstraction.Data`, with XML docs; `DocumentFormats`, `DocumentTextLayouts`, `DocumentWriters` and
  `DocumentPageDrawers` with guarded wrappers (shadow: the same page's runs, text and image list compared, the same
  written bytes, the same drawn pixels within a tolerance; fall back per call), `Ensure`d defaults.
- The `lines` and `content` layouts (in `Idrak.Abstraction`, as they need nothing but the runs and `ReadingOrders`).
- Contract suites in `Idrak.Abstraction.Testing`: `DocumentFormatSuite` (a format's `Accepts` on its own and others'
  headers, page count against pages read, every run's box inside its page, thread-safety: pages read in parallel give the
  same as one by one, limits honoured, a damaged file failing with `InvalidDataException`, never another exception);
  `DocumentWriterSuite` (pages of each kind written then read back through the registered format: paths, glyphs, text
  and raw pixels exact; the same input, the same bytes; a page's memory bounded); `DocumentPageDrawerSuite` (a page of
  known paths, glyphs and images drawn: coverage within its tolerance of the rasterizer's own drawing; parts left out
  when asked; a missing font counted, never thrown).
- **The plug-in path proven:** a toy format and a toy writer in `tests/Idrak.PluginTests` (an outside assembly, public
  API only) register, open and write their files through `DocumentFormats` and `DocumentWriters`, override a name, shadow
  and fall back, as the existing proof families do for their contracts. This is the path Office will take.

**Done when:** the suites pass on the toy format and the toy writer, the inventory and API dump list the new contracts,
and "not registered" names the registered formats, writers and drawers.

**Tests (CPU here):** one build; `documents contracts|document layouts|outside plug-in|override loop:|public API|
abstraction inventory|conformance kit:`.

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

- `tools/documents/make_fixtures.py` writes the fixtures into `tests/Idrak.Tests/data/documents/` from PyPI packages
  (run with `-I`, as Plan 24's tokenizer tools), each with a JSON of what it holds (text, boxes, images): classic and
  stream cross-references, incremental updates, a broken table, every filter and predictor, every encryption revision,
  simple and composite fonts with and without `ToUnicode`, embedded TrueType, Type 3, right-to-left and mixed text,
  vertical text, hidden text over a scan, each image kind, embedded files; and the hostile ones (a stream bomb, cycles,
  deep nesting). The fonts come from the test data already in the repository. Added for D6 and D7: vector pages (every
  path operator, both fill rules, caps, joins, dashes, clips), each colour space, shadings and patterns, transparency,
  embedded Type 1 and Type 3 fonts, a page drawn by the reference renderer saved beside each as PNG, and image PDFs as
  scanners and phones write them (one JPEG a page, a Flate page, a hidden text layer).
- A real-files group, `documents real:`, for the owner: every PDF in a folder (`IDRAK_DOCUMENTS_REAL`) opens, every page
  reads within the limits, and a summary (pages, runs, images, issues, time per page cold and warm, rule 76) is written;
  differences against a reference extractor's text are listed, not failed (the reference fixes what the text must say,
  not how fast Idrak may be).
- The README's `Idrak.Data` line and `src/Idrak.Data/README.md` name the document formats; the Office plug-in is named as
  to come.

**Done when:** the generator reproduces every fixture byte for byte from its seed, the docs say what is read, written
and drawn and what is not, and the owner's command list is in the as-built note.

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
  character for several glyphs. **Subsetting** (`Subset`, the default): only the glyphs used, with `glyf`, `loca`, `hmtx`
  and `cmap` rebuilt (and `CFF` charstrings kept by index), so a page in one font is not the font's whole size; `Whole`
  embeds the file as read. A face a font map made (legacy fonts) is written as its glyphs, like any face.
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

## Not in this plan

- **Office** (Word, Excel, PowerPoint): a later plug-in on C1-C17 (a package such as `Idrak.Data.Plugins`, namespace
  `Idrak.Data.Plugins.Office`, on the base library's `ZipArchive` and `XmlReader`, no outside dependency).
- **Forms** (field values), **annotations** (comments, highlights), **signatures** (validation), **editing** an existing
  PDF in place (an incremental update written over a file read): open question 5 asks whether "finished for good"
  includes them.
- **Users of the contract:** a `pdf` dataset format in `DataFileFormats` (a row per page), document chunking for
  retrieval (`Idrak.Text`), an `idrak` command that reads or writes a document, OCR of pages with issues, a searchable
  PDF made from an OCR run. Each is a small follow-up once D3 and D6 land; plan 38's commands are plan 38's F6.

## Order and parallelism

At most two agents at a time, each in its own worktree; a merge is built and its filters run before the push.

| Step | Lane A | Lane B |
|---|---|---|
| 1 | D0, the contract | none |
| 2 | D1, the file | D5's generator and the first fixtures (no library code) |
| 3 | D3, text (after D1) | D2, encryption (after D1) |
| 4 | D4, images and embedded files (after D3) | D6, writing (after D1; its read-back checks grow as D3 and D4 land) |
| 5 | D7, vector content and drawing (after D3, D4 and D6) | D5, docs and the owner's runs |
| then | plan 38 starts (decision 9) | |

D2 and D3 touch different files (`Pdf/Security*.cs`, `Pdf/Content*.cs` and `Pdf/Fonts*.cs`); D4 and D6 too
(`Pdf/Images*.cs`, `Pdf/Writer*.cs` and `Idrak.Imaging`'s `jpeg` encoder registration); D7 touches `Pdf/Content*.cs`
after D3 is merged, and `Idrak.Imaging` (the drawer, the Type 1 reader, the rasterizer's fill rule). The files both
lanes touch (`api/*.txt`, the inventory, the test list, the variables table) are merged one lane at a time and
regenerated, never merged by hand.

## The owner's runs

Each phase's as-built note gives the exact PowerShell list (from `D:\Projects\Idrak`). The pattern:

```powershell
cd D:\Projects\Idrak; git checkout main; git pull origin main
dotnet build -c Release tests\Idrak.Tests
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
- **The API grows** by the contract (C1-C17): each in the API dump with its reason; no version bump, no CHANGELOG
  (CLAUDE.md).
- **Drawing is the largest piece** (shadings, transparency, Type 1 and Type 3 fonts). Answered by the reference
  renderer's pixels per fixture, a stated tolerance, and `Issues` naming what a page used that is not drawn, never a
  silent wrong page.
- **Subsetting a font wrongly** (a composite glyph whose parts are dropped, a CFF subroutine left out) makes text that
  draws in Idrak and breaks elsewhere: every written file is opened by the reference reader (D6), and the subset's glyphs
  are drawn and compared with the whole font's.
- **The rasterizer's fill rule** touches every glyph drawn today: the current behaviour stays the default, and existing
  synthetic pages are checked byte-identical (rule 71).

## Open questions for the owner (2026-11-18)

1. "Finished for good": are forms (field values), annotations (comments, highlights), signatures (validation) and
   editing a PDF in place in this plan, or a later one?
2. The reference tools for D6 and D7's checks: a reader and a renderer from PyPI in `tools/documents/` (run with `-I`,
   as plan 24's tools), used only to make expected results, never by the library. Agreed?
3. The writer's default for image pages: `Auto` (JPEG for photographs and scans, Flate for line art, measured per image),
   or always `Flate` (exact pixels, larger files)?

## Runs

| Date | Phase | Machine | What | Result |
|---|---|---|---|---|
