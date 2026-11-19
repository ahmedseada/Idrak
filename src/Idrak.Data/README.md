# Idrak.Data

Datasets for Idrak (no dependencies beyond Idrak): read JSON Lines, JSON, CSV, text and Parquet files, archives and compressed files; download and cache from URLs, Hugging Face, GitHub, Kaggle and Zenodo; filter, map, shuffle, deduplicate, split, mix and turn rows into chat transcripts for fine-tuning, or columns into training samples (`TableSamples`).

And documents: PDF read, written, edited, signed and drawn, in C# with no outside program or package; forms,
annotations, bookmarks and redaction (below). Office formats (Word, Excel, PowerPoint) are to come as a plug-in in
`Idrak.Data.Plugins`, on the same contract.

## Documents

A document format is registered by name in `DocumentFormats`, as image codecs are in `ImageCodecs`, and found from a
file's first bytes; a writer in `DocumentWriters`. This package registers `pdf` for both. A file is opened, not loaded:
its objects are read when a page or part asks for them, a page at a time. Every type below is in
`Idrak.Abstraction.Data` (a global using when implicit usings are on).

### Open and read

```csharp
using var document = DocumentFormats.Open("quarterly-report.pdf");
Console.WriteLine($"{document.PageCount} pages, PDF {document.Info.Version}, \"{document.Info.Title}\"");

var page = document.ReadPage(0, DocumentParts.Text | DocumentParts.Images);
Console.WriteLine(page.Text);                                   // the page's lines in reading order
foreach (var line in page.Lines)
    Console.WriteLine($"{line.Box}: {line.Text}");              // each line's box, in points from the top-left corner
foreach (var image in page.Images)
    ImageEncoders.Save($"figure-{image.Order}.png", image.Decode());
foreach (var issue in page.Issues)
    Console.WriteLine($"{issue.Kind}: {issue.Detail}");         // unmapped-glyphs: text with no code points, a page to OCR
```

- **The file:** PDF 1.0 to 2.0; classic and stream cross-references, object streams, hybrid files, incremental updates;
  a broken cross-reference rebuilt by scanning the file (`Info.Repaired`); every standard filter and predictor.
- **Encryption:** the standard security handler, revisions 2 to 6 (RC4 40 to 128 bits, AES-128, AES-256), opened with
  the empty password or `DocumentOptions.Password` (the user's or the owner's); a wrong or missing one is an
  `InvalidDataException` naming the option.
- **Text:** runs with their box, size, font, colour and whether they are hidden (a scan's text layer), from simple,
  composite, Type 3 and the standard 14 fonts, through `ToUnicode`, the encodings, the Adobe Glyph List and the embedded
  font's own `cmap`; `/ActualText` wins over its glyphs. A page's runs become lines and text through a layout of
  `DocumentTextLayouts`: `lines` (the default: by position, each line's direction by Unicode's rule, then the reading
  order, `bidi` by default), `tagged` (the order a tagged file's structure gives) or `content` (the runs in content order).
  `DocumentParts.Glyphs` adds each run's glyph ids, places and face.
- **Images:** raw samples of 1 to 16 bits in grey, RGB, CMYK, indexed, Lab, ICC-based (by its component count),
  separation and DeviceN colours, with their masks (`Mask()`); JPEG through the registered "jpeg" codec; JPEG 2000, CCITT
  and JBIG2 handed over encoded, decoded when an app registers a codec of that name in `ImageCodecs`.
- **Vector content:** `DocumentParts.Paths` gives the paths (fill rules, strokes, dashes, colours, clips), shadings,
  patterns, transparency groups and soft masks.
- **The document's parts:** `Fields` (forms), `Outline` (bookmarks), `Destinations` (named), `PageLabel(index)`,
  `Attachments` (embedded files, each read when asked), `Signatures`; a page's `Annotations` with
  `DocumentParts.Annotations`.
- **Untrusted input:** every file is treated as hostile. A decoded stream, the objects and the nesting have limits
  (`DocumentOptions.MaxStreamBytes`, `MaxObjects`, `MaxDepth`), cycles are found; a file past one fails with an
  `InvalidDataException` naming the limit, never a hang. One damaged page or stream fails that page or stream alone.

```csharp
using var document = DocumentFormats.Open("manual.pdf", new DocumentOptions { Layout = DocumentTextLayouts.Tagged });
foreach (var bookmark in document.Outline)
    Console.WriteLine($"{bookmark.Title}: page {bookmark.Destination?.Page + 1}");
foreach (var attachment in document.Attachments)
{
    using var content = attachment.Read();
    using var file = File.Create(attachment.Name);          // the name's last part only: it never leads out of a folder
    content.CopyTo(file);
}
```

### Write

A document is written page by page (each page goes to the destination when the next starts); images, vector paths and
text with its fonts embedded as subsets, so text can be searched and copied in every script.

```csharp
IFontFace face = FontFormats.Read("fonts/sans.ttf")[0];
using var builder = DocumentWriters.Create("out/newsletter.pdf");        // the writer from the extension
builder.Info = new DocumentInfo { Title = "Company newsletter", Author = "Communications team" };

var page = builder.AddPage(595, 842);                                     // A4, in points (1/72 inch)
page.Image(ImageCodecs.Decode("photos/new-office.jpg"), new DocumentBox(56, 120, 483, 300), DocumentImageEncoding.Auto);
page.Path(new DocumentPath([OutlineVerb.Move, OutlineVerb.Line], [56, 100, 539, 100])
{
    Fill = DocumentFill.None, Stroke = 1, StrokeColour = DocumentColour.Grey(0.4f),
});

// Text is written as glyphs: the library's text setter shapes it (every script), the writer embeds the font.
float baseline = 80;
foreach (var line in DocumentTextSetters.Get(DocumentTextSetters.Shaped).Set("Our new office opens in May.", new DocumentTextSetting([face], 18) { Width = 483 }))
{
    foreach (var run in line.Runs)
    {
        var glyphs = run.Glyphs.ToArray();                                // placed from the line's start on its baseline
        for (int i = 0; i < glyphs.Length; i++)
            glyphs[i] = glyphs[i] with { X = glyphs[i].X + 56, Y = glyphs[i].Y + baseline };
        page.Glyphs(run.Face, run.Size, glyphs, run.Text, DocumentTextMode.Fill, DocumentColour.Black);
    }

    baseline += 18 * 1.3f;
}

builder.AddOutline(new DocumentOutline("Our new office") { Destination = new DocumentDestination(0) });
builder.Finish();
```

- **Image pages:** `DocumentImageEncoding.Auto` measures each image: photographs and scans as JPEG, line art and images
  of few levels exact (Flate, with the PNG predictor where it helps); `Flate` or `Jpeg` fixes it. A scan with its OCR
  text written in `DocumentTextMode.Hidden` is a searchable PDF.
- **Pages the library draws:** `DocumentShapes` (Idrak.Imaging) writes a page of shapes and laid-out text as vector
  paths and real text.
- **Also:** attached files (`Attach`), form fields (`AddField`), annotations and links on a page (`AddAnnotation`,
  `AddLink`), bookmarks, page labels, pages copied from other documents (`CopyPage`: merge and split), encryption
  (`Encrypt`, AES-256) and a signature at `Finish` (`Sign`).
- **The same bytes from the same input** (`DocumentWriteOptions.Deterministic`, on by default: no date or identifier
  unless given), on every machine; an encrypted file is the exception (its key and salts are random).

### Edit and save

An editor changes a document and saves it either as an **incremental update** (the source's bytes kept as they are, the
changes appended: earlier signatures stay valid) or as a **rewrite** (one clean file: only what the document reaches,
renumbered, object streams, duplicate streams and fonts written once).

```csharp
using var source = DocumentFormats.Open("newsletter.pdf");
using var appendix = DocumentFormats.Open("appendix.pdf");
using var output = File.Create("newsletter-complete.pdf");
using (var editor = DocumentWriters.Get("pdf").Edit(source, output, new DocumentWriteOptions()))
{
    editor.SetInfo(source.Info with { Title = "Company newsletter, spring issue" });
    editor.InsertPages(editor.PageCount, appendix, 0, 1);            // merge: two of the appendix's pages at the end
    editor.RotatePage(1, 90);
    editor.Overlay(0).Path(new DocumentPath(                          // new content over a page: a stamp, a band
        [OutlineVerb.Move, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Line, OutlineVerb.Close],
        [0, 0, 595, 0, 595, 24, 0, 24]) { FillColour = DocumentColour.Rgb(0.9f, 0.85f, 0.2f).WithAlpha(0.4f) });
    editor.Save(DocumentSaveMode.Incremental);
}
```

- **Pages:** `InsertPages` (from any document the pdf format opened, its resources carried and renamed where they
  clash), `RemovePages`, `MovePage`, `RotatePage`, `AddPage`, `Overlay`; a builder's `CopyPage` splits a file.
- **Encrypted:** an edited file keeps its own encryption; `Encrypt(new DocumentEncryption { UserPassword = ...,
  OwnerPassword = ..., Permissions = DocumentPermissions.Print })` saves it with AES-256 (revision 6) by a rewrite, and
  `Encrypt(null)` removes it when the owner password opened the source.

### Forms

```csharp
foreach (var field in document.Fields)
    Console.WriteLine($"{field.Name} ({field.Kind}): {field.Value}");

using (var editor = DocumentWriters.Get("pdf").Edit(document, output, new DocumentWriteOptions()))
{
    editor.SetField("contact.name", "Sam Taylor");
    editor.SetField("subscribe", "Yes");                                 // a check box's on state
    editor.AddField(new DocumentField("contact.email", DocumentFieldKind.Text)
    {
        Widgets = [new DocumentWidget(0, new DocumentBox(72, 640, 220, 20))], Tooltip = "Your e-mail address",
    });
    editor.Save(DocumentSaveMode.Incremental);
}
```

- **Read:** every field kind (text, check box, radio, list, combo, push button, signature) with its value, choices,
  flags, widgets, inherited values; an XFA form's data read as fields.
- **Filled and added** with appearances made, so a reader shows the values without making its own: text laid out in
  the field's font, or, when it lacks a character, in faces given (`DocumentFormOptions.Faces`, the fonts folder) through
  the text setter, embedded; check and radio marks drawn as paths.
- **Flattened** (`Flatten(names)`, none for every field): each field's appearance drawn into its page and the field
  removed.

### Annotations, links and redaction

```csharp
var page = document.ReadPage(0, DocumentParts.Annotations);
foreach (var annotation in page.Annotations)
    Console.WriteLine($"{annotation.Kind} at {annotation.Box}: {annotation.Contents ?? annotation.Uri}");

editor.AddAnnotation(new DocumentAnnotation(DocumentAnnotationKinds.Note, 0, new DocumentBox(520, 72, 20, 20))
{
    Contents = "Figures checked against the March report", Author = "Review team",
});
editor.RemoveAnnotation(page.Annotations[0]);
```

Every annotation kind is read (its points, ink strokes, replies, popup and appearance), drawn, added (with an appearance
made from its properties) and removed; links, bookmarks, named destinations and page labels are read and written.
**Redaction** removes for real what lies under areas: the glyphs from the text, the paths and the image pixels, and the
annotations over them; the areas come from a text match (`DocumentRedactions.Find`, on a page read with its glyphs) or
the file's own redaction marks (`DocumentRedactions.Marked`), and the file is saved as a rewrite, so nothing removed
survives in an earlier revision:

```csharp
using (var editor = DocumentWriters.Get("pdf").Edit(source, output, new DocumentWriteOptions()))
{
    for (int i = 0; i < source.PageCount; i++)
    {
        var areas = DocumentRedactions.Find(source.ReadPage(i, DocumentParts.Text | DocumentParts.Glyphs), "Project Falcon");
        if (areas.Count > 0)
            editor.Redact(i, areas);
    }

    editor.Save(DocumentSaveMode.Rewrite);
}
```

The `idrak` tool does the same: `idrak data redact report.pdf --text "Project Falcon" -o report-shared.pdf`.

### Sign and check

```csharp
using System.Security.Cryptography.X509Certificates;

var root = X509CertificateLoader.LoadCertificateFromFile("company-root.cer");
foreach (var signature in document.Signatures)
{
    var check = signature.Check(new SignatureCheckOptions { Trusted = [root] });
    Console.WriteLine($"{signature.Field} ({signature.Kind}): {(check.Valid ? "valid" : "not valid")}");
    foreach (var step in check.Steps.Where(s => !s.Passed))
        Console.WriteLine($"  {step.Name}: {step.Detail}");
}

using var signer = X509Certificate2.CreateFromPemFile("signer.pem", "signer.key.pem");
using (var editor = DocumentWriters.Get("pdf").Edit(source, output, new DocumentWriteOptions()))
{
    editor.Sign("Approval", new SigningOptions(signer)
    {
        Reason = "Approved for release", Location = "Head office", Page = 0, Box = new DocumentBox(360, 740, 180, 48),
        AppearanceText = "Approved", Level = DocumentSignatureLevel.BaselineT, TimestampAuthority = new Uri("https://timestamps.example.com/"),
    });
    editor.Save(DocumentSaveMode.Incremental);                           // a signature is always an incremental update
}
```

- **Checked in six steps,** each with why it failed: the digest of the signed bytes, the signature (RSA PKCS #1 v1.5 and
  PSS, ECDSA), the chain against the certificates the app trusts (`Trusted`; the machine's store only with
  `UseMachineStore`), the timestamp, revocation (from the file's own data, or fetched only when `Online` or
  `IDRAK_DOCUMENT_SIGNATURE_ONLINE` turns it on), and the changes made after signing (allowed by the signature's
  permissions or not). A signature never makes a file trusted on its own.
- **Signed** as PAdES B-B, B-T (a timestamp from the authority the app names) and B-LT (revocation data added), SHA-256,
  -384 or -512, with a visible appearance (text, an image) or none; a second signature keeps the first valid. The
  network goes through `SignatureServices` ("http" and "https" built in, an app's in their place).

### Draw a page

```csharp
var drawing = document.DrawPage(0, new DocumentDrawOptions { Dpi = 200, Parts = DocumentParts.Text | DocumentParts.Images | DocumentParts.Paths | DocumentParts.Annotations });
ImageEncoders.Save("page-1.png", drawing.Image);
foreach (var issue in drawing.Issues)
    Console.WriteLine($"{issue.Kind}: {issue.Detail}");                  // missing-font: text whose font is not embedded or found
var pages = DocumentPageDrawers.DrawPages(document, [0, 1, 2], new DocumentDrawOptions());   // several pages in parallel
```

Any page is drawn to pixels, a scan's image or a vector page alike, through `DocumentPageDrawers` (the library's drawer,
"coverage", is in Idrak.Imaging: its settings are in that package's README). Fonts a file does not embed come from a
folder (`DocumentDrawOptions.Fonts`, `IDRAK_DOCUMENT_FONTS`); without one such text is left out and counted.

### Options and variables

| Options | Setting | Variable | Default |
|---|---|---|---|
| `DocumentOptions` (reading) | `Password` | none (a secret stays out of the environment) | the empty password |
| | `Layout` | `IDRAK_DOCUMENT_LAYOUT` | `lines` |
| | `ReadingOrder` | `IDRAK_DOCUMENT_READING_ORDER` | `bidi` |
| | `MaxStreamBytes` | `IDRAK_DOCUMENT_MAX_STREAM_BYTES` | measured from the free memory when a document opens |
| | `CacheBytes` | `IDRAK_DOCUMENT_CACHE_BYTES` | measured from the free memory when a document opens (0: nothing kept) |
| | `MaxObjects` | `IDRAK_DOCUMENT_MAX_OBJECTS` | measured from the free memory when a document opens |
| | `MaxDepth` | `IDRAK_DOCUMENT_MAX_DEPTH` | 64 levels (each also checked against the stack the reading thread has left) |
| | `IncludeHidden`, `Direction`, `WordGap`, `LineTolerance` | | true; each line's own; 0.2; 0.5 |
| `DocumentWriteOptions` (writing, editing) | `ImageEncoding` | `IDRAK_DOCUMENT_IMAGE_ENCODING` | `auto` |
| | `JpegQuality` | `IDRAK_DOCUMENT_JPEG_QUALITY` | 90 |
| | `EmbedFonts` | `IDRAK_DOCUMENT_EMBED_FONTS` | `subset` |
| | `SaveMode` (when the app does not choose) | `IDRAK_DOCUMENT_SAVE_MODE` | `incremental` |
| | `Compression`, `LineArtLevels`, `SmoothSteps`, `Version`, `Deterministic`, `ObjectStreams`, `Forms` | | 6; 32; 0.25; the writer's; true; true; `DocumentFormOptions` |
| `DocumentDrawOptions` (drawing) | `Dpi`, `Fonts` and the drawer's settings | `IDRAK_DOCUMENT_DPI`, `IDRAK_DOCUMENT_FONTS`, ... | 150 dpi; none (Idrak.Imaging's README) |
| `SignatureCheckOptions`, `SigningOptions` | `Online` | `IDRAK_DOCUMENT_SIGNATURE_ONLINE` | off |
| `FontWriteOptions` (fonts embedded) | `CurveTolerance` | `IDRAK_FONT_CURVE_TOLERANCE` | 1 font unit |

Each variable is in the `idrak` tool's table (`idrak env`); a bad value is refused naming its variable. A limit's
message names the limit, its option and its variable, never a measured value, so a file gives the same messages on
every run. The memory a document holds is its cache (within `CacheBytes`) and the page being read; a file is written
and a page drawn with one page's (or one band's) buffers; a large JPEG is decoded a strip of rows at a time, and an
image written is measured a band at a time and stored with one copy of it (the stream the deflater takes).

`idrak bench documents` measures all of it on any folders of PDF files (by default the sample generator's samples and
large set): opening, text with the `lines` layout and with none, text over the cores, images, pages drawn at chosen
resolutions, image and vector PDFs written, incremental and rewritten saves, forms filled, signing and checking; each
case cold and warm apart (the time, the managed bytes allocated, the process's working set at its highest), the
settings above recorded, `--save` and `--compare` as `idrak bench`'s (the `idrak` tool's README).

### What is not supported

- **Office formats** (Word, Excel, PowerPoint): to come as a plug-in in `Idrak.Data.Plugins`, on this contract.
- **Nothing a file carries is run:** JavaScript, launch and other actions, calculations of a form's fields; no outside
  reference is fetched (revocation data only when the app turns `Online` on). 3D and multimedia content is not played.
- **XFA forms:** their data is read as fields; their own layout is not read or drawn (the page says `not-supported`).
- **Encryption:** the public-key handler (`Adobe.PubSec`) and other handlers are refused naming the handler; a damaged
  encrypted file whose cross-reference must be rebuilt and whose catalog lies in an object stream does not open.
- **Text:** predefined CMaps other than Identity (tables tied to one script) are not built in: such codes are read as two
  bytes and their text comes from `ToUnicode` alone (`not-supported`). A glyph with no code point by any route is
  U+FFFD and counted in `unmapped-glyphs`.
- **Images:** JPEG 2000, CCITT and JBIG2 are not decoded built in (register a codec), and the parameters of CCITT and
  JBIG2 are not handed to one; ICC profiles and calibrations are not applied.
- **Drawing:** a non-isolated transparency group with another blend mode or opacity is drawn as an isolated one; a
  stroke under a matrix that is not uniform takes the root of its determinant as its scale; a Type 3 glyph's own colours
  and images are not kept; a tiling cell's box is not a clip; free text annotations without an appearance, a widget's
  rotation and dashed borders are not drawn.
- **Forms:** rich text values are read as plain text; a field held inside another object cannot be filled.
- **Redaction:** a text match finds a word within one run (not one split across text operators); images inside
  patterns, soft masks and Type 3 glyphs are not redacted, and a redacted image loses its soft mask.
- **Writing and editing:** fonts are embedded as TrueType (a subset, or whole), never as OpenType CFF; every page sits
  under one page-tree node; objects an incremental update no longer uses stay in the file.
- **Signatures:** document timestamps (B-LTA) are not written; a signature's appearance text is set in the standard
  Helvetica's WinAnsi encoding (other characters show as `?`: give `AppearanceImage` for any script); delta and
  indirect revocation lists, name and policy constraints (a critical one fails the chain, named), DSA and EdDSA keys
  are not read.
- **Not built on the contract yet:** a `pdf` dataset format (a row per page), document chunking for retrieval, OCR of
  pages whose text has issues.

## Contracts

The plug-in points only this package uses are in `Idrak.Data.Abstractions` (add the `using` line to name them):
`DataFileFormats` and `IDataFileFormat` (file formats), `DatasetSources` and `IDatasetSource` (with `DatasetSpec`,
`ReadOptions`, `IDatasetRows`), `ParquetCodecs` and `IParquetCodec`, `ITextNormalizer`. The registries register the
built-ins themselves, as the library defaults of their slots: an app's registration under a built-in name overrides it
and `Unregister` brings it back. A Parquet codec or a dataset source an app overrides throws to the caller when it fails,
reported with a hint (`SetPolicy` opts it into `FallBack` to the library's, or `Shadow`); a file format is used as registered (its rows are read lazily). Shared
contracts (sample sources, `IDownloader`, model sources, and the documents' `DocumentFormats`, `DocumentWriters`,
`DocumentTextLayouts` and `SignatureServices`) are in Idrak.Abstraction; this package registers its `pdf` format and
writer and the "http" and "https" signature services into them.

## Folders skipped

Reading a folder or an archive skips the folders `ReadOptions.SkippedFolders` names: by default
(`ReadOptions.DefaultSkippedFolders`) only `.git` and the editors' state (`.vs`, `.idea`). No language's build output or
dependency folders are skipped unless named (`bin`, `obj`, `node_modules`, `dist`, `__pycache__`, …): a caller that knows
them adds them. The `idrak` tool's data commands add every registered coding language's (Idrak.Text's `CodingLanguages`;
the tool registers Idrak.Text.Plugins' at startup), so they read code folders as before. A recipe takes them as
`"skip_folders": [...]`, `DatasetSpec.Open` and `GitHub.Repository` as an argument. The table from file extensions to
language names that labels the `code` format's rows (`DataFiles.LanguageOf`) is data only.

## Install

```bash
dotnet add package Idrak.Data
```

This is a library (for your code). The command-line tool `idrak` inspects, downloads and builds datasets with no code needed (`idrak data`); it is a separate package:

```bash
dotnet tool install -g Idrak.Cli
idrak help data
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

## Documentation

Guides, samples and the full API overview: https://github.com/ahmedseada/Idrak

License: Apache 2.0.
