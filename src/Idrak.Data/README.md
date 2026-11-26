# Idrak.Data

Datasets for Idrak (no dependencies beyond Idrak and Idrak.Imaging): read JSON Lines, JSON, CSV, text and Parquet files, archives and compressed files; download and cache from URLs, Hugging Face, GitHub, Kaggle and Zenodo; filter, map, shuffle, deduplicate, split, mix and turn rows into chat transcripts for fine-tuning, or columns into training samples (`TableSamples`).

And documents: PDF read, written, edited, signed and drawn, in C# with no outside program or package; forms,
annotations, bookmarks and redaction (below). Office formats (Word, Excel, PowerPoint) are to come as a plug-in in
`Idrak.Data.Plugins`, on the same contract.

## Documents

A document format is registered by name in `DocumentFormats` and found from a file's first bytes; a writer in
`DocumentWriters`. This package registers `pdf` for both. A file is opened, not loaded:
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
  `DocumentOptions.Password`, tried as the owner's then the user's password, or the empty password; a wrong or missing
  one is an `InvalidDataException` naming the option, but where the user password is empty a wrong one opens the file
  as its user and every page read says so (`wrong-password`). `Info.Security` says how a file is encrypted (the
  handler, its revision, the key's length, the ciphers, the permissions; never a key, salt or IV) and whether the owner
  password opened it (`OwnerAccess`: only then may its encryption be changed or removed). A damaged encrypted file is
  repaired too, its key found before its object streams are listed.
- **Text:** runs with their box, size, font, colour and whether they are hidden (a scan's text layer), from simple,
  composite, Type 3 and the standard 14 fonts, through `ToUnicode`, the encodings, the Adobe Glyph List and the embedded
  font's own `cmap`; `/ActualText` wins over its glyphs. A page's runs become lines and text through a layout of
  `DocumentTextLayouts`: `lines` (the default: by position, each line's direction by Unicode's rule, then the reading
  order, `bidi` by default), `tagged` (the order a tagged file's structure gives) or `content` (the runs in content order).
  `DocumentParts.Glyphs` adds each run's glyph ids, places and face.
- **Images:** raw samples of 1 to 16 bits in grey, RGB, CMYK, indexed, Lab, ICC-based (by its component count),
  separation and DeviceN colours, with their masks (`Mask()`); JPEG through the library's JPEG codec (Idrak.Imaging's
  `JpegCodec`, reached directly); JPEG 2000, CCITT and JBIG2 handed over encoded (the library has no decoder for them).
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
page.Image(new JpegCodec().Decode(File.ReadAllBytes("photos/new-office.jpg")), new DocumentBox(56, 120, 483, 300), DocumentImageEncoding.Auto);
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
  paths and real text, a shape's text along its path glyph by glyph.
- **Turned glyphs:** `IDocumentPageBuilder.Glyphs(face, glyphs, matrices, text, mode, colour)` draws each glyph through
  its own matrix (text round a ring, a turned or slanted line); the pdf writer gives each turned glyph its own text
  matrix (and its own `/ActualText` when its characters are not what its font's map gives), so it reads back as a run
  of its own, drawn as written. A writer that writes upright glyphs only refuses turned ones (the default member).
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
  flags, widgets (each one's turn), border style, inherited values, and a rich text value as styled runs
  (`DocumentField.RichText`: bold, italic, size, colour, family, underline, strike-out); an XFA form's data read as
  fields, and its template as the form's design (`IDocument.FormTemplate`: sections, fields with captions, kinds,
  choices and bindings, each linked to the data's value and the field holding it).
- **Filled and added** with appearances made, so a reader shows the values without making its own: text laid out in
  the field's font, or, when it lacks a character, in faces given (`DocumentFormOptions.Faces`, the fonts folder) through
  the text setter, embedded (the field's `/DA` then names that face's font, added to the form's resources); rich text
  (`SetRichText`) in each run's font, size and colour; turned (`/MK /R`) and bordered (solid, dashed, beveled, inset,
  underlined) as each widget says; check and radio marks drawn as paths. A field held inside another object (written
  in its parent's `/Kids`, or in an object stream) is filled too: moved to an object of its own.
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
**Redaction** removes for real what lies under areas: the glyphs from the text, the paths and the image pixels (an
image's soft mask kept, redacted under the same areas), the same inside forms, soft masks' groups, tiling patterns' cells
and Type 3 glyphs, and the annotations over them with their popups and replies (every reference to them left out of the
file: other pages' lists, form fields, the structure tree, actions); the areas come from a text match
(`DocumentRedactions.Find`, on a page read with its glyphs: a word split across text operators, fonts or kerning is
found whole) or the file's own redaction marks (`DocumentRedactions.Marked`), and the file is saved as a rewrite, so
nothing removed survives in an earlier revision:

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
  permissions or not). A signature never makes a file trusted on its own. RSA is verified and paths are built by
  Idrak itself, the same on every platform, and tested against public known-bad cases (Project Wycheproof's RSA and
  ECDSA vectors, x509-limbo's path cases, the sample generator's bad paths): a path with an RSA key under 2048 bits
  (`MinimumRsaKeyBits`) or longer than 12 certificates (`MaxChainLength`) fails the chain step, naming it. A path's
  name constraints (directory names, mailboxes, DNS names, URIs, IPv4 and IPv6 addresses, the trusted certificate's too)
  and its certificate policies (RFC 5280's policy processing: mappings, inhibit-any-policy, explicit policies) are
  evaluated whether marked critical or not; the check's result names the policies the path is valid for (`Policies`),
  and the app may ask for policies (`InitialPolicies`, `RequireExplicitPolicy`, `InhibitPolicyMapping`,
  `InhibitAnyPolicy`). Revocation reads delta lists with their complete ones, indirect lists (from a CRL issuer the
  certificate's distribution point names, its own path checked to the same trusted certificate) and a list's scope;
  data past its next update is stale (fetched again online, else "not checked (stale)", with a grace the app may give,
  `FreshnessGrace`); each list's and response's signature is verified once a check.
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
| `SignatureCheckOptions` | `MinimumRsaKeyBits` (0: any size) | `IDRAK_DOCUMENT_SIGNATURE_RSA_BITS` | 2048 bits (pyHanko's default) |
| | `MaxChainLength` (certificates, the signer's and the trusted one's included) | `IDRAK_DOCUMENT_SIGNATURE_CHAIN` | 12 |
| | `InitialPolicies` (policy identifiers a signer's path must be valid for) | `IDRAK_DOCUMENT_SIGNATURE_POLICIES` | any policy |
| | `RequireExplicitPolicy`, `InhibitPolicyMapping`, `InhibitAnyPolicy` (RFC 5280's initial flags) | `IDRAK_DOCUMENT_SIGNATURE_EXPLICIT_POLICY`, `IDRAK_DOCUMENT_SIGNATURE_INHIBIT_MAPPING`, `IDRAK_DOCUMENT_SIGNATURE_INHIBIT_ANY_POLICY` | off |
| | `FreshnessGrace` (how long past its next update revocation data stays fresh) | `IDRAK_DOCUMENT_SIGNATURE_GRACE` | 0 |
| | `MaxNameConstraintChecks` (name comparisons a path's constraints may ask for) | `IDRAK_DOCUMENT_SIGNATURE_NAME_CHECKS` | 1,048,576 (OpenSSL's bound) |
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
- **XFA forms:** their data is read as fields and their template as the form's design; their own layout is not laid out
  or drawn (the page says `not-supported`) and their scripts are never run (each listed in the template's issues).
- **Encryption:** the public-key handler (`Adobe.PubSec`) and other handlers are refused naming the handler; a damaged
  encrypted file whose cross-reference must be rebuilt and whose catalog lies in an object stream does not open.
- **Text:** predefined CMaps other than Identity (tables tied to one script) are not built in: such codes are read as two
  bytes and their text comes from `ToUnicode` alone (`not-supported`). A glyph with no code point by any route is
  U+FFFD and counted in `unmapped-glyphs`.
- **Images:** JPEG 2000, CCITT and JBIG2 are not decoded (the library has no decoder for them; `Bytes()` gives them as
  stored); ICC profiles and calibrations are not applied.
- **Drawing:** text painted with a pattern is drawn in a solid colour (black for a coloured pattern); free text
  annotations without an appearance are not drawn.
- **Forms:** a rich text's paragraph alignment is the field's, and a line of several styled runs is laid out with its
  runs in order (each run shaped in its own direction: a right-to-left line of several styles is not reordered across
  them); calculation order and actions are never run.
- **Redaction:** a tiling pattern turned on the page is redacted wherever a copy of its cell lies under an area (an
  upright one only where no copy shows that part clear of the areas); a Type 3 glyph no longer drawn on a page stays in
  the page's copy of its font when a form there takes the page's resources as its own.
- **Writing and editing:** fonts are embedded as TrueType (a subset, or whole), never as OpenType CFF; every page sits
  under one page-tree node; objects an incremental update no longer uses stay in the file.
- **Signatures:** document timestamps (B-LTA) are not written; a signature's appearance text is set in the standard
  Helvetica's WinAnsi encoding (other characters show as `?`: give `AppearanceImage` for any script); B-LT signing
  fetches a certificate's complete list, not its delta. Known limits (the known-bad tests print a `LIMIT:` line for each
  case they meet):
  - **name constraints of the forms otherName, x400Address, ediPartyName and registeredID** are not compared: a
    constraint on one fails a path whose certificates hold a name of that form, naming it (as RFC 5280 asks of a form a
    validator does not process); directory names are compared with RFC 4518's insignificant spaces and the Unicode
    simple case folding, not its full string preparation (NFKC), and DNS names in ASCII as written;
  - **policy qualifiers** (a CPS address, a user notice) are not read, and the trusted certificate's own policy
    extensions are not processed (it is the path's anchor, RFC 5280 6.1), while its name constraints are applied;
  - **revocation lists:** a distribution point's reasons field is not read (a list's own `onlySomeReasons` is, and lists
    covering some reasons each count when they cover every reason between them); an indirect list's issuer's
    certificate is found among the signature's and the document's certificates, never fetched;
  - **DSA, EdDSA and ML-DSA keys** are not read (RSA and ECDSA are): a path or a signature made with one fails;
  - **ECDSA's curves are the platform's:** the DER signature, the ranges of r and s and the digest's length are read by
    Idrak, the curve arithmetic by the base library's `ECDsa`: P-256, P-384 and P-521 on every platform, secp256k1 and
    the Brainpool curves only where the platform's `ECDsa` takes them.
  - By design, not limits: a trusted certificate's own dates are not checked (RFC 5280 6.1 and pyHanko alike), a list
    issued after the time a check is made at is used and a certificate revoked after that time is good then (PAdES
    long-term validation), the rules for how a CA writes certificates and lists (key identifiers present, serial
    numbers' form, which extensions are critical, the CRL number) are not refused, and name and policy constraints are
    applied whether marked critical or not (pyHanko alike; the CA/Browser Forum lets a CA mark name constraints
    non-critical).
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
