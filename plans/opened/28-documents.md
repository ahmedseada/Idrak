# Plan 28: documents (a document contract, PDF built in, Office as a plug-in later)

**Not started.**

**Status (2026-10-22, `abstraction`):** planned 2026-10-22 from the owner's decisions, against the code at 99872b8 (every
"Comes from" below was found in the source). **Nothing is built.** No phase starts before the owner names it.

| Phase | What | Lane | State |
|---|---|---|---|
| D0 | The contract and its registries (`IDocumentFormat`, `DocumentFormats`, `IDocumentTextLayout`, `DocumentTextLayouts`), their suites, a test format from outside proving the plug-in path | A | Planned |
| D1 | PDF, the file: objects, cross-reference tables and streams, object streams, incremental updates, repair, filters, limits | A | Planned |
| D2 | PDF encryption: the standard security handler (RC4, AES-128, AES-256), an empty or a given open password | B | Planned |
| D3 | PDF text: pages, the content-stream interpreter, fonts to Unicode, glyph widths, text runs with boxes, the `lines` layout | A | Planned |
| D4 | PDF images and embedded files | B | Planned |
| D5 | Fixtures and their generator, READMEs, the owner's runs on real files | A | Planned |

## Goal

The library reads documents through one contract, the way it reads images through `ImageCodecs`: a format registered
by name, chosen from a file's first bytes, opened page by page, giving text with its place on the page, images and
embedded files. PDF is the library's default. Office (Word, Excel, PowerPoint) comes later as a plug-in registering into
the same contract, with nothing in the library changed for it. Apps (search and question answering over attachments, a
meeting system's minutes) read any registered format the same way.

## Decisions (taken by the owner, 2026-10-22)

1. **PDF now, as a library default:** the contract and its registry in `Idrak.Abstraction.Data`, the PDF reader in
   `Idrak.Data`, registered through `LibraryDefaults`, as PNG and JPEG are by `Idrak.Imaging`.
2. **Office later, as any plug-in:** it implements the same contract and registers into `DocumentFormats`; this plan
   builds none of it, only a contract it can implement without a change.
3. **Nothing language-specific in the library.** Text order comes from positions on the page and from Unicode's own rules
   through the existing `ReadingOrders` registry (`bidi`, UAX #9), never from code for a language or a script. Tables
   tied to one script (the CJK predefined CMaps, for example) are not built in.
4. **Plan first:** this file is written before any code.

**Proposed, for the owner to confirm:**

5. **Scope of the first version: text and images, not rendering.** Text with its box, font size and order; the images of
   each page (a scanned PDF's pages come out as images, ready for OCR); embedded files; the document's information; files
   encrypted with an empty or a given open password. Not drawing a page's vector content to pixels (several times the
   work), not forms, annotations or signatures (see "Not in this plan").
6. **Only the contract and the PDF reader** ("just implement the contract"): no CLI command, no dataset rows, no retrieval
   chunking from documents here. Those are users of the contract, listed under "Not in this plan".

## Rules that hold for every phase

CLAUDE.md and [docs/optimization.md](../../docs/optimization.md) apply in full. The ones that shape this plan:

- **Contract plus registry (decision 10).** `DocumentFormats` and `DocumentTextLayouts` are `SlotTable`s with
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
  within forms, arrays within arrays), cycles detected; no JavaScript run, no outside reference fetched, no file written.
  Each limit is an option (rule "nothing fixed"); passing one is a clear `InvalidDataException` naming the limit and the
  option, never a hang or an out-of-memory.
- **Damaged files are the common case.** A broken cross-reference is rebuilt by scanning the file for objects, and the
  document's information says it was repaired; one bad page or stream fails that page or stream alone.
- **Results proven (rule 71).** Every fixture has its expected text, boxes and images checked exactly; a speed change
  keeps them.
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

The contract is format-neutral: a Word file has no fixed pages or boxes, so an Office plug-in gives one page per section
(or one for the whole file), runs without boxes (`Box` empty) and `Lines` from its paragraphs. Nothing in C1-C10 names
PDF.

**Settings:** the options in C8, with `IDRAK_DOCUMENT_LAYOUT`, `IDRAK_DOCUMENT_READING_ORDER`,
`IDRAK_DOCUMENT_MAX_STREAM_BYTES` and `IDRAK_DOCUMENT_CACHE_BYTES` read by the library (names proposed), added to the
CLI's variables table (`src/Idrak.Cli.Commands/Shared/EnvironmentVariables.cs`, which a test checks). Defaults: the stream limit and
the cache from the machine's measured memory, never a fixed size in GB.

---

## Phase D0: the contract and its registries

- C1-C10 in `Idrak.Abstraction.Data`, with XML docs; `DocumentFormats` and `DocumentTextLayouts` with guarded wrappers
  (shadow: the same page's runs, text and image list compared; fall back per call), `Ensure`d defaults.
- The `lines` and `content` layouts (in `Idrak.Abstraction`, as they need nothing but the runs and `ReadingOrders`).
- A contract suite in `Idrak.Abstraction.Testing` (`DocumentFormatSuite`): a format's `Accepts` on its own and others'
  headers, page count against pages read, every run's box inside its page, thread-safety (pages read in parallel give the
  same as one by one), limits honoured, a damaged file failing with `InvalidDataException`, never another exception.
- **The plug-in path proven:** a toy format in `tests/Idrak.PluginTests` (an outside assembly, public API only) registers,
  opens its files through `DocumentFormats.Open`, overrides a name, shadows and falls back, as the existing proof families
  do for their contracts. This is the path Office will take.

**Done when:** the suites pass on the toy format, the inventory and API dump list the new contracts, and "not registered"
names the registered formats.

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
  deep nesting). The fonts come from the test data already in the repository.
- A real-files group, `documents real:`, for the owner: every PDF in a folder (`IDRAK_DOCUMENTS_REAL`) opens, every page
  reads within the limits, and a summary (pages, runs, images, issues, time per page cold and warm, rule 76) is written;
  differences against a reference extractor's text are listed, not failed (the reference fixes what the text must say,
  not how fast Idrak may be).
- The README's `Idrak.Data` line and `src/Idrak.Data/README.md` name the document formats; the Office plug-in is named as
  to come.

**Done when:** the generator reproduces every fixture byte for byte from its seed, the docs say what is read and what is
not, and the owner's command list is in the as-built note.

## Not in this plan

- **Office** (Word, Excel, PowerPoint): a later plug-in on C1-C10 (a package such as `Idrak.Data.Plugins`, namespace
  `Idrak.Data.Plugins.Office`, on the base library's `ZipArchive` and `XmlReader`, no outside dependency).
- **Drawing pages to pixels** (vector content, fonts, shading); until then a scanned page's image is what OCR reads.
- **Forms** (field values), **annotations** (comments, highlights), **signatures** (validation), **writing or editing** PDFs.
- **Users of the contract:** a `pdf` dataset format in `DataFileFormats` (a row per page), document chunking for
  retrieval (`Idrak.Text`), an `idrak` command that reads a document, OCR of pages with issues. Each is a small follow-up
  once D3 lands.

## Order and parallelism

At most two agents at a time, each in its own worktree; a merge is built and its filters run before the push.

| Step | Lane A | Lane B |
|---|---|---|
| 1 | D0, the contract | none |
| 2 | D1, the file | D5's generator and the first fixtures (no library code) |
| 3 | D3, text (after D1) | D2, encryption (after D1) |
| 4 | D5, docs and the owner's runs | D4, images and embedded files (after D3) |

D2 and D3 touch different files (`Pdf/Security*.cs`, `Pdf/Content*.cs` and `Pdf/Fonts*.cs`); the files both lanes touch
(`api/*.txt`, the inventory, the test list, the variables table) are merged one lane at a time and regenerated, never
merged by hand.

## The owner's runs

Each phase's as-built note gives the exact PowerShell list (from `D:\Projects\Idrak`). The pattern:

```powershell
# once, after a phase is merged
dotnet build -c Release tests\Idrak.Tests

# D3 onwards: the owner's own PDFs (a folder of real files; nothing is written into it)
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents real:"; $env:IDRAK_DOCUMENTS_REAL = "D:\Documents\pdf-samples"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER, Env:IDRAK_DOCUMENTS_REAL
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
- **The API grows** by the contract (C1-C10): each in the API dump with its reason; no version bump, no CHANGELOG
  (CLAUDE.md).

## Runs

| Date | Phase | Machine | What | Result |
|---|---|---|---|---|
