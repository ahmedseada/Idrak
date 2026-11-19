# tools/documents: the PDF and font samples of plan 28

`make_fixtures.py` writes plan 28's samples (phase D5, decision 13) and what a reader, a drawer, an editor or a
signature check must get from each: 150 samples for phases D1-D4 and D6-D12 (the file, encryption, text, images and
embedded files; writing, vector content and drawing, fonts, forms, annotations and navigation, editing, signatures),
10 of them hostile; with `--large`, the D13 set (thousands of pages, many small files, a huge image page, deep forms)
in a folder of its own. Nothing it writes is committed: everything goes into the folder given by `-o` (default
`test-output/documents/samples`, which git ignores), and the tests read it from `IDRAK_DOCUMENTS_SAMPLES`.

```bash
# Linux (a virtual environment outside the repository is best)
python3 -m venv ~/venvs/idrak-docs && ~/venvs/idrak-docs/bin/python -m pip install -r tools/documents/requirements.txt
~/venvs/idrak-docs/bin/python -I tools/documents/make_fixtures.py -o test-output/documents/samples
~/venvs/idrak-docs/bin/python -I tools/documents/make_fixtures.py --only none --large   # the D13 set alone
```

```powershell
# Windows PowerShell, from D:\Projects\Idrak
python -m pip install -r tools\documents\requirements.txt
python -I tools\documents\make_fixtures.py -o test-output\documents\samples
python -I tools\documents\make_fixtures.py --only none --large --large-out test-output\documents\large
```

- `-o DIR` the output folder. It is emptied first; a non-empty folder without a `manifest.json` is refused (never a
  folder the tool did not write).
- `--only d1,d7` some phases only (`d1 d2 d3 d4 d6 d7 d8 d9 d10 d11 d12`; `none` for no phase); `--no-reference` the
  samples and their JSON without the reference tools' runs; `--seed S` another seed (default `2026`).
- `--large` also writes the D13 set into `--large-out` (default `test-output/documents/large`, emptied first, with its
  own `manifest.json`); it is off by default so the normal run stays quick.
- `--compare A B` compares two output folders: the manifests byte for byte and every file's SHA-256 (exit code 1 on a
  difference); it works on two large folders too.

The run prints each phase as it is made, then each sample as it is written with the time spent, the rate and the time
left, then the counts and the reference tools' notes. The default run takes about 90 seconds here (70 of them the
reference tools, most of that the hostile files' probes); `--large` adds about 16 seconds.

Three checkers for files Idrak writes, sharing the generator's reference runs:

- `check_written.py FILES|DIR -o REPORT.json [--expected auto|FILE.json] [--password PW] [--render DIR] [--dpi N]`
  (D6 onwards): qpdf (pikepdf) opens each PDF, runs its syntax check and decodes every stream (a JPEG or JPEG 2000
  image, which qpdf hands over encoded, as the image it is, through Pillow); pdfminer.six reads
  every page's text; pdfium counts, reads and (with `--render`) draws the pages. Every warning, problem and stream
  error is listed; with an expected JSON (this README's format; `auto` takes `NAME.json` beside `NAME.pdf`), the page
  count, the page sizes and each page's text (whitespace collapsed) are compared. Exit code 1 when a file fails.
- `check_font.py FONT... -o REPORT.json [--source SRC] [--tolerance U] [--mapping MAP.json] [--line TEXT]... [--dump DIR]`
  (D8): fontTools opens each font and decompiles every table, the OpenType Sanitizer (`ots-sanitize`, the checker
  browsers use) accepts it or says why not, HarfBuzz shapes the lines; with `--source`, every outline (exact for the
  same outline kind; across TrueType and CFF each glyph's exact bounds, area and centroid within the tolerance),
  advance, character and the shaping are compared with the source (a subset through `--mapping`: a JSON list of
  source glyph ids, or a D8 sample's JSON; its shaping on the lines it covers). Exit code 1 when a font fails.
- `check_signed.py FILES|DIR -o REPORT.json [--password PW] [--pki DIR]` (D12): pyHanko validates every signature of
  each PDF Idrak signed with the test root alone as trusted, at the JSON's validation time, nothing fetched (and a B-LT
  signature again with the file's DSS alone, revocation required, at its timestamp's time), and compares what it finds
  with `NAME.json` beside `NAME.pdf` (each signature's field, kind, validity, coverage, timestamp, revocation and
  changes, as the tests write it; the PKI is the folder it names, or `--pki`). Exit code 1 when a signature differs.

The PDF reader's font tables (D3) are made here too: `make_pdf_tables.py [OUT_FILE]` writes
`src/Idrak.Data/Pdf/PdfFontTables.Data.cs` (committed: the library reads it, nothing is generated at build) from the pinned
fontTools (the Adobe Glyph List and the ITC Zapf Dingbats list) and pdfminer.six (the base encodings and the standard 14
fonts' AFM metrics), with the Symbol and ZapfDingbats built-in encodings of ISO 32000-2 Annex D checked code by code
against pdfium; the licence notices go into the file. Reruns give the same bytes.

## Real files

The test group `documents real:` (`tests/Idrak.Tests.Library/DocumentRealTests.cs`, plan 28's D5 with D13's real-files
clause) reads a folder of real PDFs: every `*.pdf` under the folder `IDRAK_DOCUMENTS_REAL` names (its subfolders too, in
the order of their paths; nothing is written into it). Without the variable its three tests are SKIP lines. For each
file it:

- **reads** it within the default limits: opened (version, encryption, repaired), every page's text through the `lines`
  and the `tagged` layouts, its images (each decoded, with its mask), paths, clips, shadings and annotations, the
  fields, bookmarks, named destinations, page labels, attachments (each read to its end) and signatures (each checked,
  nothing trusted and nothing fetched, so the digest, signature and changes steps are what it proves);
- **draws** every page at the drawer's default DPI (text, images, paths and annotations);
- **writes it back** as a rewrite and as an incremental update with its title changed, each read back: the same page
  count, sizes, turns, text, runs and images, the title written; the update also keeps the source's bytes as its prefix
  and every signature's digest and signature as they checked;
- **times** each step (open, text, tagged, parts, images, navigation, signatures, draw, rewrite, incremental) cold (the
  file's first pass of the step in the run; the first file's also compiles the code) and warm (the median of the warm
  passes) apart, with the time and the managed memory a page.

A file fails the group only for a crash (an exception the contract does not document), a page or part that cannot be
read and says nothing why, or a written-back file that does not read the same; the failure names the file and the step.
A damaged, hostile or unsupported file reports its reason (an `InvalidDataException` or `NotSupportedException` with its
message) and the run goes on; an encrypted file that needs a password and has none is listed as skipped.

The summary, `test-output/documents/real/summary.json` and `summary.txt` (written again every few seconds while the run
goes, so a long run can be followed in it): per file its pages, bytes, version, the counts (runs, lines, characters,
images and pixels decoded, paths, annotations by kind, fields by kind, bookmarks, attachments, signatures and each one's
check), the issues by kind with their details and pages, what drawing left out, every step's times cold and warm with a
page's time and memory, the written-back files' sizes and checks, the reference's differences and every reported
failure; then the totals.

| Variable (the tests' own) | Default | What it sets |
|---|---|---|
| `IDRAK_DOCUMENTS_REAL` | none: the group is skipped | The folder of PDFs |
| `IDRAK_DOCUMENTS_REAL_REFERENCE` | `test-output/documents/real-reference` when it is there | A folder of reference text: `NAME.reference.txt` at the PDF's path under it (a `NAME.reference.txt` beside the PDF comes first) |
| `IDRAK_DOCUMENTS_REAL_SAVE` | none | `pages`, `files` or both (`all`): the drawn pages as PNG in `test-output/documents/real/drawn/` and the written-back files in `.../written/` (the folder's paths kept), else nothing is kept |
| `IDRAK_DOCUMENTS_REAL_WARM` | 3 | The warm passes of each step, 0 to 100 (0: cold alone, for a large folder) |

A password is never a variable: the first line of `NAME.password.txt` beside `NAME.pdf` opens it. The run takes as long
as the folder needs (each step once cold and three times warm), longer than the runner's 300 seconds a test for a large
folder: give it `IDRAK_TIMEOUT` (seconds).

**The reference text.** `reference_text.py FOLDER [-o OUT] [--beside] [--password PW] [--timeout S] [--memory-mb M]`
writes pdfminer.six's text of every PDF in the folder (the reader the samples' reference text comes from, pinned in
`requirements.txt`) as `NAME.reference.txt` at the PDF's path under `OUT` (default
`test-output/documents/real-reference`, the folder the group reads), or beside each PDF with `--beside`: each page's text
in order, the pages separated by a form feed. Each file is read in a child process with a time limit (120 s) and, on
Linux, a memory limit (4 GiB); a file it cannot read is listed with its error and gets no text (an older one is
removed). It prints each file with the rate and the time left and writes `reference.json` into `OUT`. The group compares
each page, whitespace collapsed: the same, the same but for spaces, the same words in another order, the same words with
some of them as drawn (a right-to-left word's characters in the order they are drawn, which is how pdfminer.six gives
them, where Idrak gives logical order), or words that differ (the reference's words Idrak lacks and Idrak's extra ones,
and the share in common). The differences are listed, never failed: the reference fixes what the text must say, not how
fast Idrak may be. pdfminer.six also ignores `/ActualText` and cannot read an embedded Type 1 font's own encoding
(`(cid:N)` in its text), so such pages list differences that are the reference's.

```powershell
# Windows PowerShell, from D:\Projects\Idrak (the folder D:\Documents\pdf-samples is an example)
python -I tools\documents\reference_text.py D:\Documents\pdf-samples
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "documents real:"; $env:IDRAK_DOCUMENTS_REAL = "D:\Documents\pdf-samples"; $env:IDRAK_TIMEOUT = "7200"
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-real.txt
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER, Env:IDRAK_DOCUMENTS_REAL, Env:IDRAK_TIMEOUT
Get-Content test-output\documents\real\summary.txt
```

## Reproducible

Every byte comes from the seed and the pinned versions (`requirements.txt`): no dates, no random identifiers (the file
`/ID` is the MD5 of the seed and the sample's name; AES vectors, the revision 5-6 salts and keys and the padding of `/U`
come from SHA-256 of the seed, the sample and a counter), fonts subset with their timestamps kept, ICC profiles with
their dates cleared. Flate streams and PNGs use the interpreter's zlib, whose version the manifest records with the
packages'. Two runs into two folders give the same files, the manifest included (checked: 767 files and the manifest
identical; the large set: 1,007 files and its manifest identical).

What the specifications leave random, and how it is fixed here (D12, `docfixtures/pki.py`):

- **RSA keys:** primes drawn from `random.Random` seeded by SHA-256 of the seed and the key's label, tested by
  Miller-Rabin with seeded bases (the key is then the same on every run and machine with the same Python); the **EC
  key:** its scalar is SHA-256 of the seed and its label.
- **Certificates and CRLs:** fixed serials and times, signed with RSA PKCS #1 v1.5 (no randomness in it).
- **The OCSP response and the timestamp tokens:** written with asn1crypto (cryptography's OCSP builder stamps the
  current time), fixed `producedAt`, `genTime` and serials; the timestamp request's nonce is SHA-256 of the seed.
- **Signatures:** RSA PKCS #1 v1.5 needs nothing; RSA-PSS takes a 32-byte salt from the seed (EMSA-PSS encoded here,
  the private exponent applied, then checked by cryptography); ECDSA is deterministic (RFC 6979, cryptography's
  `deterministic_signing`). Signing times are fixed (`/M` 2026-03-02T09:30:00Z, the timestamp 5 seconds later).
- **CMS:** asn1crypto writes SET OF in DER order (the signed attributes, the certificates), so the bytes repeat.
- **Fonts (D8):** fontTools with timestamps kept, WOFF through zlib, WOFF2 through Brotli (both pinned).

## The tools, and why

- **Every PDF byte is written here** (`docfixtures/pdf.py`, `filters.py`, `security.py`, `document.py`), not by a PDF
  library: the cross-reference kinds, object streams, hybrid files, incremental updates, damaged tables, wrong lengths,
  each filter and predictor, the encryption, the vector content, forms, annotations and signatures are exactly what
  each sample says, which no writer library lets one choose.
- **qpdf** (`pikepdf`) checks every non-hostile file (`check_pdf_syntax`) and decodes every stream the JSON lists, whose
  bytes must hash as the generator wrote them: the filters, predictors and the encryption are proven by an independent
  reader, not by the generator's own decoders. D11's merges, splits, page operations and the stamp are performed by
  qpdf as the reference, and its permissions are read back by it.
- **pdfminer.six** is the reference reader: its text per page. It also supplies the encodings' tables (`latin_enc`) and
  the standard 14 fonts' AFM widths the samples are laid out with.
- **pdfium** (`pypdfium2`) is the reference renderer: each page as PNG at 100 DPI, and its text with each character's
  loose box, against which every run box of the JSON is checked (`reference.pdfium.boxes`); its path objects are
  matched against every expected path (`reference.pdfium.paths`), and its form fields, annotations, links, bookmarks,
  page labels and structure tree against D9's and D10's expectations.
- **fontTools** subsets the embedded fonts, gives glyph names their text (the Adobe Glyph List rules), writes the Type 1
  font of D7 (`t1Lib`, eexec) and every D8 font (WOFF, WOFF2, CFF from TrueType, TrueType from CFF through cu2qu,
  subsets) and dumps them; **HarfBuzz** (`uharfbuzz`) shapes the composite fonts' text (GSUB only, so every glyph
  advances by its width) and D8's sample lines; **python-bidi** checks the visual order of every right-to-left and mixed
  line; **cryptography** gives AES (RC4 is written here), the test PKI's keys, certificates and CRLs and the
  signatures; **Pillow** writes the JPEGs and the scanned and photographed pages.
- **pyHanko** and **pyhanko-certvalidator** (D12) are the reference signature validator: each signed sample is
  validated with the test root alone as trusted (decision 16), at a fixed time, nothing fetched, and must give what its
  JSON expects. pyHanko pulls in aiohttp and its helpers, oscrypto, uritools, tzlocal (tzdata on Windows), certifi
  and idna (pinned with it; nothing is fetched).
- **asn1crypto** (pyHanko's own dependency) writes the CMS SignedData, the OCSP response and the RFC 3161 timestamp
  tokens.
- **Brotli** (`brotli`) is the compression fontTools writes and reads WOFF2 with.
- **The OpenType Sanitizer** (`opentype-sanitizer`, `ots-sanitize`, the font checker browsers use) checks every D8
  font and every font `check_font.py` is given. Its wheels are `py2.py3-none` for Windows and Linux, so Python 3.13
  installs them.
- **Fonts:** only the repository's test fonts with their licences beside them: the DejaVu Sans subset and the fonts
  made from it (`dejavu-sans-subset.ttf`, `dejavu-sans-cff.otf`, `features.ttf`; `LICENSE-DejaVu.txt`) and Amiri
  (`arabic/amiri.ttf`, `OFL-amiri.txt`). D7's Type 1 font is made here from DejaVu Sans' outlines.

## The samples

| Folder | Samples |
|---|---|
| `d1/` | `xref-table`, `xref-stream` (PNG Up predictor), `object-streams` (`/Extends`), `hybrid` (`/XRefStm`), `incremental` (two updates through `/Prev`: a stream replaced, a page added, the title changed, an object freed), `incremental-stream` (an update as a cross-reference stream replacing an object held in an object stream), `header-junk` (590 bytes before `%PDF-`, offsets from the header) and `header-junk-absolute` (offsets from the file's start), `broken-xref` (every offset 11 bytes short: repair), `missing-xref` (no table, no startxref: repair), `wrong-length` (`/Length` short, long, indirect), `filters` (one page a filter: Flate with and without the zlib header, ASCIIHex with an odd last digit, ASCII85 with `z`, LZW with and without early change, RunLength, three chains, none), `predictors` (PNG 10-15 at 1, 8 and 16 bits, TIFF 2 at 4, 8 and 16 bits, LZW with Paeth), `info` and `info-partial` (PDFDocEncoding, UTF-16BE, UTF-8, escapes, octal, a line continuation, hex strings, `D:` dates whole, partial and odd), `page-tree` (inherited MediaBox, Resources and Rotate; a wrong `/Count`), and nine hostile files (below) |
| `d2/` | `plain` (the unencrypted twin), then for each of `r2-rc4-40`, `r3-rc4-128`, `r4-rc4-128`, `r4-aes-128`, `r4-aes-128-metadata-plain` (`/EncryptMetadata false`), `r4-aes-128-identity-strings` (`/StrF /Identity`), `r4-aes-128-xref-stream`, `r5-aes-256`, `r6-aes-256`, `r6-aes-256-object-streams` an `-empty` (empty user password) and a `-user` (`user-pass-1`) file, owner password `owner-pass-1`; `r6-aes-256-unicode` (`Grüße-2024`); `unsupported-handler` (`/Adobe.PubSec`) |
| `d3/` | `standard14` (not embedded, AFM widths), `encodings` (MacRoman, Standard, WinAnsi, `/Differences` with `uni2192`, `f_f`, `a.sc`, `Euro`, `uni00410042`, `u1F4C8` and the unmappable `g17`), `truetype-simple` (with and without ToUnicode), `type0-identity`, `type0-no-tounicode`, `type0-no-cmap`, `cidtogidmap-stream`, `type3`, `text-state` (TJ, Tc, Tw, Tz, TL, T*, `'`, `"`, Ts), `render-modes` (0-7), `actual-text`, `rtl` (Amiri), `mixed-direction` (DejaVu), `vertical` (Identity-V), `rotated-pages` (0, 90, 180, 270, text turned by Tm and cm), `cropbox`, `text-matrix` (sizes from Tm and the CTM, a form XObject), `hostile-form-recursion` |
| `d4/` | `colour-spaces` (Gray, RGB, CMYK, Indexed over RGB, grey and CMYK, ICCBased N 3 and 1, CalGray, CalRGB, Lab), `bit-depths` (1, 2, 4, 8, 16 bits, rows ending mid-byte), `decode-arrays`, `masks` (stencils, soft masks, an explicit mask, a colour key), `dct` (grey, 4:2:0, progressive 4:4:4, Adobe CMYK), `inline-images` (abbreviations, a named colour space, data holding ` EIQ`), `image-placement` (turned, slanted, flipped, mirrored, on a turned page), `scan-ocr` (an A4 grey JPEG with a hidden OCR text layer, D3 and D4), `attachments` (an EmbeddedFiles name tree with Kids and Limits, a file-attachment annotation) |

| `d6/` | `inputs` (no PDF: seven PNGs the writer is given, photographs, a scanned page, line art, a screenshot, two levels, each with its measures and the `Auto` encoding it must get), `scanner-jpeg-pages` (A4 grey 150 dpi, Letter RGB 200 dpi, A5 grey 100 dpi, one JPEG a page), `flate-page` (two-level and line-art pages, Flate with PNG predictors, a cross-reference stream), `phone-photo` (a photo page, and the same on a page turned by /Rotate 90), `searchable-scan` (two scanned pages with a hidden text layer in a composite font, object streams) |
| `d7/` | `vector-paths` (m l c v y h re; S s f F f* B B* b b*), `vector-fill-rules` (stars, nested squares both ways, overlaps, a self-crossing curve, each nonzero and even-odd), `vector-strokes` (caps, joins, miter limits 1.5 and 10, dashes with phases, round dots, widths 0, 0.5, 3, a non-uniform CTM), `vector-clipping` (W, W*, nested, after Q, a form's BBox, an image under a clip, text as a clip), `vector-colour-spaces` (Gray, RGB, CMYK, ICCBased RGB and grey, Indexed over RGB and CMYK, CalGray, CalRGB, Lab, Separation through type 2, 0 and 4 functions, DeviceN through type 4 and 3; strokes in Lab and Separation), `vector-shadings` (axial with a stitching function, radial, function-based, free-form, lattice, Coons, tensor), `vector-patterns` (coloured and uncoloured tiling, a turned one, a shading pattern, a pattern stroke), `vector-transparency` (ca, CA, the 16 blend modes, a luminosity soft mask, an isolated knockout group and a plain group), `vector-fonts` (an embedded Type 1 made here, Type 3, TrueType, Type1C, OpenType CFF under CIDFontType0; outlined, filled-and-outlined and turned text), `vector-images` (turned, skewed, mirrored, under ca, with a soft mask, a stencil), `vector-turned-pages` (/Rotate 90 and a CropBox) |
| `d8/` | for `dejavu-sans-subset`, `dejavu-sans-cff`, `features` and `amiri`: `NAME.source` (the repository font's dump and shaping; no font file), `NAME.woff`, `NAME.woff2` (TrueType: glyf and loca transformed), `dejavu-sans-subset.hmtx.woff2` (hmtx transformed too), `dejavu-sans-subset.cff.otf` and `features.cff.otf` (TrueType outlines as CFF), `dejavu-sans-cff.truetype.ttf` (CFF as TrueType, cu2qu within 1 unit), `NAME.subset.ttf|otf` (a fixed glyph list, layout dropped) and `NAME.subset-layout.ttf|otf` (layout kept with its closure) |
| `d9/` | `forms-filled` (every field kind with appearances), `forms-empty`, `forms-need-appearances` (/NeedAppearances, no appearance for text, choice, push button and signature fields), `forms-inherited` (FT, Ff, DA, V inherited; one field with two widgets on two pages), `forms-xfa` (XFA template and datasets with AcroForm twins), `forms-flattened` (forms-filled with its appearances drawn into the page and no fields) |
| `d10/` | `annotations` and `annotations-no-appearance` (every kind: markup over words, a note with a popup and replies, free text, line, square with an open popup, circle, polygon, polyline, ink, stamp, caret, file attachment, links), `navigation` (links to explicit and named destinations, a bookmarks tree, page labels i ii 1 2 3 A-1 A-2 A Cover), `tagged` (structure order differs from content order, an artifact), `redaction` and `redaction-applied` (text, paths and an image under three redaction areas; the result of applying them) |
| `d11/` | `edit-incremental-table` (three classic updates: a field filled and a note added, a page added and the title changed, a page turned and the note freed), `edit-incremental-stream` (two updates as cross-reference streams over object streams), `merge-a`, `merge-b` and `stamp` (resources inherited from the page tree under the same names /F1 /Im1 /GS1), `merge-plan` (no PDF: what the merge and the stamp must give), `split-source`, `encrypted-aes256` (revision 6, user and owner passwords, permissions), `page-operations` (insert, remove, move, rotate and a combination, with the pages each gives) |
| `d12/` | `pki` (no PDF: the test PKI's files), `pades-bb`, `pades-bb-pss`, `pades-bb-ecdsa`, `pades-bt`, `pades-blt`, `pades-revoked`, `broken-digest`, `wrong-chain`, `changed-allowed`, `changed-not-allowed`, `two-signatures` (CMS then PAdES ECDSA), `legacy-pkcs7-sha1`, `legacy-x509-sha1` |
| `d13/` (with `--large`, in its own folder) | `pages-3000` (3,000 Letter pages of text with a shared image and 300 page images, object streams), `many-small/` (1,000 one-page files; `many-small.json` lists them), `huge-image` (an A0 page holding one 9,933 by 14,043 RGB JPEG, 418 MB decoded), `deep-forms` (form XObjects nested 8, 16 and 24 deep; one form drawn 1,000 times) |

Text in the samples is ordinary business and technology text in English and Arabic (reports, meetings, orders, a
newsletter); names are made-up generic ones (Example Organisation, Alex Morgan) and every web address is under
example.com.

## The files

Beside each `NAME.pdf` in its phase's folder:

| File | What |
|---|---|
| `NAME.json` | what the file holds and what a reader must get (below) |
| `NAME.text.json` | the reference tools' text: `pdfminer` (pdfminer.six, a string a page) and `pdfium` (a string a page) |
| `NAME.pN.png` | page N as pdfium draws it, RGB, at `reference.dpi` (100) |
| `NAME.pN.imK.png` | the expected pixels of page N's K-th image in content order (`.smask.png`, `.mask.png`: its soft mask and explicit mask) |
| `NAME.<file name>` | the bytes of an embedded file (`attachments` only) |
| `NAME.pN.imK.redacted.png` | `redaction`: the image's expected pixels once the redactions are applied |
| `NAME.dump.json`, `NAME.shaping.json` | D8: the font's dump and HarfBuzz's shaping of the sample lines (below); `NAME` is the font's file name (`amiri.woff2.dump.json`) or `NAME.source` for the repository font |
| `inputs.NAME.png` | D6: an image the writer is given |
| `pki.*` | D12: `pki.LABEL.cer` (DER) and `.pem`, `pki.LABEL.key.pem` (unencrypted PKCS #8, throwaway test keys), `pki.root.crl`, `pki.intermediate.crl`, `pki.signer.ocsp` |

`manifest.json` lists every sample (`name`, `folder`, `file`, `kind`, `phases`, `hostile`, `expected`: its JSON,
`files`: the files beside it, `size`, `sha256`), every file written (`path`, `size`, `sha256`; the manifest itself
excepted), the seed, `renderDpi` and `versions` (Python, zlib, qpdf, pdfium, HarfBuzz and every package).

### NAME.json

| Key | What |
|---|---|
| `schema`, `name`, `file`, `kind`, `phases`, `description`, `size`, `sha256` | the sample |
| `hostile` | null, or what a reader must do with a hostile file: `limit` (the option it trips: `MaxStreamBytes`, `MaxDepth`, `MaxObjects`, or `cycle` for a cycle detected), `options` (the values to open it with in the test), `error` (`InvalidDataException`), `trips` (when: on open or reading page 0), and the sizes (`decodedBytes`, `depth`, `claimedObjects`). A hostile file's JSON has only these keys and `reference.outcomes` |
| `openError` | a file that must not open: the exception and what it names (`unsupported-handler`: `NotSupportedException`, `Adobe.PubSec`) |
| `open` | `password` (null: none; `""`: the empty one), `encrypted`, `repaired` (true: the cross-reference must be rebuilt; null: either is right), `version` (the header's), `pageCount` |
| `encryption`, `twin` | encrypted files: `filter`, `V`, `R`, `lengthBits`, `cipher`, `cryptFilter`, `stringFilter`, `encryptMetadata`, `userPassword`, `ownerPassword`, `permissions` (`/P`), `wrongPassword` (fails), `encryptDictionary` (its object number); `twin` names the plain file with the same objects and text |
| `info` | `title`, `author`, `subject`, `keywords`, `creator`, `producer` (decoded text), `created`, `modified` (ISO 8601, with the offset when the date has one: `D:2024` is `2024-01-01T00:00:00`, `Z` is `+00:00`), `other` (any other key: text, a name without its slash), `raw` (the dates as written) |
| `structure` | `headerOffset` (bytes before `%PDF-`), `offsetsFrom` (`file` or `header`), `damage`, `revisions` (oldest first: `kind` `table` / `stream` / `objstm` / `hybrid`, `offset` of its cross-reference, `objects` written, `objectStreams`, `compressed` objects, `free` objects), `revisionEnds`, `freed` ([number, next generation]), `countWritten` |
| `streams` | every stream object as the newest revision has it: `object`, `revision`, `type`, `filters`, `storedLength` (bytes in the file, encrypted if the file is), `lengthWritten` (the `/Length` as written: may be wrong, or `indirect`), `decodedLength` and `decodedSha256` (decrypted and through every filter except the image encodings `DCTDecode`, `JPXDecode`, `CCITTFaxDecode`, `JBIG2Decode`, which a reader hands over as they are), `note` |
| `objects` | D1 and D2 samples: every live object of the newest revision (and the encryption dictionary) in a canonical form: numbers and booleans as they are, names `"/Name"`, strings `{"string": hex}` (decrypted), references `{"ref": [n, g]}`, dictionaries `{"dict": {...}}`, arrays as lists, streams `{"stream": dictionary with its /Length, "decodedLength", "decodedSha256"}` |
| `pages` | each page (below) |
| `attachments` | `name` (its key in the name tree), `fileName`, `mediaType`, `size`, `sha256`, `description`, `source` (`EmbeddedFiles` or `annotation`), `page` (the annotation's page), `file` (its bytes beside the sample) |
| `metadataSha256`, `unicodeRule` | D2: the XMP metadata stream's bytes; `type0-no-tounicode`: the rule its text follows |
| `files` | the files beside the sample |
| `reference` | what the reference tools did (below) |

A page: `index`, `width` and `height` (points, as shown: after the CropBox and `/Rotate`), `rotation`, `mediaBox`,
`cropBox`, `text` (the lines joined by `\n`, logical order), `lines` (`text` and `direction`: `ltr`, `rtl` or `ttb` for
a vertical column), `runs`, `images`, `issues` (`unmappedGlyphs`: glyphs whose text is U+FFFD).

**Geometry.** Points; the page as shown (after its CropBox and `/Rotate`), origin at the top left, y down. A **run** is
one text-showing operator (`Tj`, `'`, `"`, or a whole `TJ`), or one marked-content span with `/ActualText`. Its keys:

| Key | What |
|---|---|
| `order` | its place among the page's runs and images, in content order |
| `text` | the text of its glyphs in content order (each glyph's text as the font maps it; a span's `/ActualText`). In a right-to-left run the glyphs come in visual order, so the characters come reversed, **but a glyph that stands for several characters keeps them in logical order** (DejaVu Sans's lam-alef ligature in `mixed-direction` reads `لا` inside a reversed run); the line's `text` is logical |
| `box` | `[x, y, width, height]`: the bounding box of its glyphs' boxes. A glyph's box spans its advance along the baseline and the font's descent to ascent across it (FontDescriptor `/Ascent` and `/Descent`; the standard 14: the AFM's Ascender and Descender, or the FontBBox's y range where the AFM has none; Type 3: its FontBBox), raised by the rise; a vertical glyph spans its vertical advance (`/DW2`) down the column and its width centred across it. Mapped through the text matrix, the CTM, the crop and the rotation; a turned run's box is the box around its turned glyphs. Leading and trailing spaces count |
| `origin`, `end` | the start of the first glyph's baseline and the point after the last glyph's advance |
| `fontSize` | the size times the length the text matrix and the CTM give a unit vertical vector |
| `font`, `fontResource`, `fontKind` | the `/BaseFont` (subset tag included; null for Type 3), the resource name, and `standard14`, `truetype`, `type0` or `type3` |
| `ascent`, `descent`, `vertical` | the values the box used (thousandths of the size) |
| `hidden`, `renderMode` | hidden for render modes 3 (invisible) and 7 (clip only) |
| `angle` | the baseline's angle as shown, degrees, y down (a vertical column: its direction less 90) |
| `actualText`, `glyphText` | a span: true, and its glyphs' own text |
| `unmappedGlyphs` | its glyphs with no text |

Word gaps: where the text has a space the file holds no space character (`text-state`'s TJ line), the gap is at least
0.3 of the font size; kerning offsets are at most 0.06 of it, so a layout's word-gap share between those bounds gives the
expected text.

**Text without ToUnicode:** simple fonts by their encoding and `/Differences` (glyph names by the Adobe Glyph List
rules); composite fonts through the embedded program's cmap reversed (the lowest code point mapping to the glyph, after
`CIDToGIDMap`); otherwise U+FFFD, counted in `issues.unmappedGlyphs`.

An **image**: `resource` (null inline), `label`, `object` (null inline), `inline` (with `inlineDictionary`),
`pixelWidth`, `pixelHeight`, `bitsPerComponent`, `colourSpace` (a name, `["ICCBased", N]` or `["Indexed", base,
hival]`), `components`, `decode`, `imageMask`, `encoding` (`raw` or `jpeg`), `filters`, `predictor`, `samplesLength`
and `samplesSha256` (the decoded samples, before colour conversion), `jpegSha256`, `ctm` (the matrix it is drawn with,
in the file's user space), `box` (the placed unit square's bounding box as shown), `order`, `softMask`, `mask` (an
image of their own) or `colourKeyMask`, and `pixels`: `file`, `channels` (1 grey, 3 RGB), `depth` (8, or 16 for 16-bit
grey and RGB), `sha256` (of the interleaved samples) and `tolerance` (levels; 2 for Lab, else 0). The rules
(`docfixtures/images.py`): `/Decode` maps each sample; grey, RGB, CalGray, CalRGB and ICCBased are read by their
component count (calibration and profiles not applied); CMYK goes to RGB as `(255 - K) - X * (255 - K) / 255` with
Pillow's rounding (the library's JPEG rule); Indexed takes the clamped index's colour; Lab goes through XYZ (scaled from
its white point to D65) to sRGB; a stencil is grey, 0 where it paints and 255 where it does not; a JPEG's pixels are
the JPEG codec's own decode of the stream's bytes (Pillow, libjpeg-turbo, which reads Adobe CMYK as inverted, so the
`/Decode [1 0 ...]` written for other readers is not applied again).

`reference` (non-hostile): `dpi`, `text`, `pages`; `qpdf` (`opened`, `pages`, `encrypted`, `problems`,
`streamsChecked`, `streamMismatches`: empty when every listed stream decodes to its hash); `pdfminer` (`opened`,
`error`, `pages`); `pdfium` (`opened`, `error`, `pageCount`, `rendered`, `pageErrors`, `boxes`: per page the runs
matched to pdfium's characters and the largest difference of the left and right edges, `largestDifferenceX`, and of
the top and bottom edges, `largestDifferenceY`, in points). Hostile files and `unsupported-handler`: `outcomes`, each
tool's `result` (`opened`, `error` with its message, `crashed` with the return code, `timeout`) in a child process
with 30 seconds and, on Linux, 2 GiB of address space.

### The JSON added for D6-D13

Every sample's JSON keeps the keys above (`schema` 1); a sample with no main file (`inputs`, `merge-plan`, `pki`,
`many-small`, the D8 `NAME.source` dumps) has `file`, `size` and `sha256` null. What each phase adds:

**Vector content (D6, D7).** A page of a vector sample also lists `paths`, `clips` and `shadings`, and its runs and
images give `fillColour`, `strokeColour` (runs), `clip`, `alpha` (`fill` ca, `stroke` CA), `blend` and `group`.

| Key | What |
|---|---|
| a path | `order` (shared with the runs and images), `operators` (m l c v y h re as written), `paint` (S s f F f* B B* b b*), `verbs` (move, line, cubic, close: `v` and `y` are cubics with the current or the end point doubled, `re` a move, three lines and a close) and `points` (each verb's points, the page as shown, top-left origin), `fill` (nonzero, evenodd or null), `stroke` (null, or `width` in page points: the line width times the square root of the CTM's determinant, `lineWidth` as written, `cap` butt/round/square, `join` miter/round/bevel, `miterLimit`, `dash` with `pattern` and `phase` in page points and `patternWritten` and `phaseWritten`), `fillColour` and `strokeColour`, `clip`, `alpha`, `blend`, `softMask` (null or `type`, `group`, `backdrop`), `group`, `ctm`, `bbox` |
| a colour | `space` (DeviceGray, DeviceRGB, DeviceCMYK, ICCBased, Indexed, CalGray, CalRGB, Lab, Separation, DeviceN), `components` as given, `rgb` (0-255) and `tolerance` (levels: Lab 2, Separation and DeviceN 1, else 0); a pattern: `space` Pattern, `pattern` (its resource), `patternType`, `paintType`, `shadingType`, and for an uncoloured tiling its `components`, `rgb` and `tolerance`. RGB as `ImageData` holds it: grey and RGB round(c * 255); CMYK through 8 bits and (255 - K) - X * (255 - K) / 255 (D4's rule); ICCBased, CalGray, CalRGB by component count; Lab through XYZ (D65) to sRGB; Indexed through its lookup; Separation and DeviceN through their tint transform into the alternate space |
| a clip | `index`, `rule`, `operators` (`["BBox"]` for a form's BBox, `["Tr 7"]` for text with `textRun` naming the run), `verbs`, `points`, `parent` (the clip it intersects, or null), `bbox`, `form` |
| a shading (`sh`) | `order`, `resource`, `shadingType`, `colourSpace`, `object`, `clip`, `alpha`, `blend`, `softMask`, `group`, `ctm`, `samples` (points whose colour is exact: `point` in the page as shown, `components`, `rgb`, `tolerance`) |
| `patterns` (a page) | each pattern resource: `patternType`, `paintType`, `bbox`, `xStep`, `yStep`, `matrix`, `object`; a shading pattern its `shadingType` |
| `groups` (a page) | each transparency group: `index`, `isolated`, `knockout`, `form`, `object`, `probes` (`point`, `rgb`, `tolerance`: the colour the group must give there) |
| `fontPrograms` | `vector-fonts`: `type1` (`object`, `lengths` Length1-3, `fontName`, `encoding`, `unitsPerEm` 1000, `glyphs`: each name's `width` and `outline` as fontTools reads the program: moveTo, lineTo, curveTo, closePath with their points), `type1c`, `opentypeCff` (its glyphs in font units) |

`inputs` (D6): `images` (`name`, `file`, `kind`, `width`, `height`, `channels`, `pixelsSha256` of the interleaved
samples, `distinctColours`, `smallSteps` and `largeSteps` (the share of neighbouring pixel pairs, right and down,
whose largest channel difference is 1-16, and over 64), `expectedAuto` Jpeg or Flate, `why`) and `text` (lines to write).

**Fonts (D8).** `font`: `source` and `licence` (repository paths), `lines` (the sample lines), `sourcePath`, `format`
(truetype, opentype-cff, woff, woff2), `derivation` (source, whole, converted, subset), `outlines`, `transforms`,
`tolerance` (font units, across outline kinds) and `toleranceWhy`, `glyphs` (a subset: `requested` glyph ids and
`requestedNames`, `characters`, `kept`: the source glyph id of each glyph the subset holds, in its order, and
`keptNames`), `layout` (dropped or closure); `dump` and `shaping` name the files beside. The dump: `format`, `sha256`,
`size`, `sfntVersion`, `flavor`, `tables` (`tag`, `length`), `unitsPerEm`, `glyphCount`, `glyphOrder`, `head`, `hhea`,
`os2`, `post`, `cmap` (`subtables` [platform, encoding, format], `best`: code point to glyph id), `hmtx` ([advance, left
side bearing] per glyph), `names` ([nameID, platformID, platEncID, langID, text]), `glyphs` (one a line: `name`,
`kind` simple/composite/empty/cff, `contours` ([x, y, on-curve] per point, composites resolved), `components`
(`glyph`, `transform` [a b c d dx dy]), `bounds`, or `outline` for CFF), `layout` (`GSUB`, `GPOS`: `scripts`,
`features`, `lookups` with `index`, `type` (extension resolved), `flag`, `markFilteringSet`, `subtables` and `glyphs`:
every glyph id a lookup refers to; `GDEF`: `glyphClasses`). The shaping: `lines` (`text`, `direction`, `script`,
`glyphs`: `gid`, `name`, `cluster`, `xAdvance`, `yAdvance`, `xOffset`, `yOffset`). `reference`: `fontTools`
(`opened`, `tables`), `ots` (`passed`, `messages`), `kept` (the comparison with the source: `sameOutlineKind`,
`comparedBy`, `tolerance`, `largestOutlineDistance`, the differences, `kept`).

**Forms (D9).** `fields` (each: `name` full and dotted, `partialName`, `kind` text/checkbox/radio/list/combo/pushbutton/
signature, `value` (a check box's or radio group's on state, null when off or empty), `values` (multi-select),
`defaultValue`, `options` (`export`, `display`), `selected`, `flags` (names) and `flagsValue`, `maxLength`,
`alignment`, `font`, `fontSize`, `tooltip`, `inherited` (keys taken from an ancestor), `widgets` (`page`, `box`,
`rect`, `state`, `onState`, `hasAppearance`, `appearanceText`); a push button's `action` and `caption`; a signature
field's `signed`) and `form` (`needAppearances`, `appearances`, `xfa`: `packets`, `data` (path to value),
`datasetsSha256`, `issue`; `flattenedFrom`). `reference.fields` (pdfium's widgets and the `problems` found),
`reference.flatten`.

**Annotations and navigation (D10).** `annotations` (each: `id` its /NM, `kind`, `subtype`, `page`, `box`, `rect`,
`contents`, `author`, `created`, `modified`, `colour` and `interiorColour` (`components`, `rgb`), `opacity`, `flags`,
`points` (quad points as four corners each, upper left, upper right, lower left, lower right; a line's ends; vertices),
`ink` (each stroke's `verbs` and `points`), `target` (`uri`; or `page`, `fit` and `values`; or `name` and `source`),
`popup`, `parent`, `inReplyTo`, `replyType`, `state`, `stateModel`, `icon`, `borderWidth`, `lineEndings`,
`hasAppearance`, `attachment`). `outline` (`title`, `target`, `colour`, `italic`, `bold`, `open`, `children`),
`outlineVisibleCount`, `namedDestinations` (name to `page`, `fit`, `values`, `source`), `pageLabels` (each page's
label) and `pageLabelRanges`. `tagged` (`structure`: elements with `type`, `mcids`, `page`, `text`; `text` in tagged
order; `contentOrderText`; `artifacts`). `redaction` (`areas`: `id`, `page`, `rect`, `box`, `points`,
`interiorColour`, `overlayText`; `rules`; `afterApplying`: `lines`, `removedText`, `paths` (`removed`, `clipped`,
`kept`: path orders), `images` (`order`, `pixels`, `sha256`, `changedPixels`), `sample`).

**Editing (D11).** `revisions` (each revision's `kind`, `offset`, `objects`, `objectStreams`, `compressed`, `free`,
`end` and `changes`) with `structure.revisionEnds`; `final` (what the newest revision reads: `pageCount`,
`rotations`, `title`, `fields`, `annotations`, `outline`, `pageText`; `pages` is then empty); `resources` (merge
inputs); `merge` and `stamp` (`merge-plan`: the `result` pages with their `from`, `page`, `font`, `text`;
`clashingNames`; `rule`); `split` (`parts`: `pages`, `text`, `rotations`); `encryption` with `permissionsAllowed`,
`permissionsDenied` and `openWith`; `operations` (each case: `name`, `steps` (`op` insert with `at`, `from` and
`pages`; remove with `pages`; move with `from` and `to`; rotate with `page` and `by`), applied in order, indices into
the document as it stands; `result`: each page's `source` "sample:index", `text`, `rotation`).

**Signatures (D12).** `signatures` (each: `field`, `subFilter`, `kind` (PAdES B-B, B-T, B-LT, CMS, legacy PKCS #7
SHA-1, legacy PKCS #1 SHA-1), `signer` (`subject`, `issuer`, `serial`, `certificate`, `sha256`, `key`), `chain`,
`byteRange`, `contentsReserved`, `cmsLength`, `revision`, `signedRevisionEnd`, `coversWholeFile`, `digestAlgorithm`,
`signatureAlgorithm`, `claimedSigningTime`, `cmsSigningTime`, `timestamp` (`time`, `authority`, `policy`,
`certificate`), `reason`, `location`, `docMdp`, `certification`, `trust` (the only certificates a check is given as
trusted, decision 16), `validationTime`, `results`: `digest` and `signature` valid/invalid, `chain` trusted/untrusted,
`timestamp` valid or null, `revocation` good/revoked/not-checked, `modifications` none/allowed/not-allowed,
`modificationLevel` none/form-filling/annotations/other); `dss` (`certs`, `crls`, `ocsps`, `vri`); `damage`
(`broken-digest`: the byte changed). `pki` (the PKI sample): `parties`, `crls`, `ocsp`, `signingTime`,
`validationTime`, `timestampPolicy`. `reference.validator`: pyHanko's verdict per signature and the `problems`.

**The large set (D13).** `checks`: `pageCount`, `pageSize`, `textPages` ([index, text] the reference reads),
`pageTextSha256` (every page's text, lines joined by newlines, UTF-8), `images`, `imageCount`, `image` (the huge
page's), `depths` and `formDraws` (deep forms), `files` (many-small: each `file`, `pageCount`, `text`, `size`,
`sha256`). `reference`: the light check (qpdf and pdfium open and count, pdfium draws the first page, pdfminer.six
reads the listed pages).

## What the reference tools say

Every non-hostile file opens in qpdf, and every listed stream decodes to its hash. pdfium draws every page it can load
and pdfminer.six reads every file, except where noted:

- **Run boxes against pdfium:** on every horizontal run whose text pdfium gives the same, the left and right edges
  agree to 0.03 point or better (standard 14, embedded TrueType and composite widths, Tc, Tw, Tz, TJ, rise, crop and
  page rotation), except Amiri's first letters, which overhang their advance by 0.2 point in pdfium's boxes. The top and
  bottom agree on embedded fonts and differ by 0.227 of the size on the standard 14: pdfium draws those with its own
  substitute fonts and takes their extents, not the AFM's Ascender and Descender, which the JSON uses. Slanted text, text
  turned 30 degrees and vertical columns differ by up to 2 points (pdfium's loose boxes are not the turned glyph boxes).
  Type 3 runs and composite fonts without ToUnicode are not compared (pdfium's text differs, below).
- **pdfium reverses the characters of a ligature in right-to-left text** (it gives `ال` for the lam-alef glyph whose
  ToUnicode is `لا`); pdfminer.six keeps them, as the JSON does.
- **Type 3:** pdfium gives the code (`F`) where the glyph name (`checkmark`) gives `✓`; pdfminer.six follows the names.
- **No ToUnicode in a composite font:** pdfminer.six reverses the embedded cmap as the JSON does; pdfium gives other
  characters.
- **`/Differences` names:** pdfium gives the code for a name that is not one character (`f_f`, `uni00410042`, `g17`)
  and drops `u1F4C8`; pdfminer.six follows the Adobe Glyph List rules as the JSON does, except that it gives the base
  encoding's character (U+201E) for `g17`, which names nothing (the JSON: U+FFFD, one unmapped glyph).
- **`/ActualText`:** pdfium honours it; pdfminer.six does not.
- **`/StrF /Identity`** (`r4-aes-128-identity-strings-*`): qpdf reads them; pdfium refuses (it needs `/StmF` and
  `/StrF` to be the same filter) and pdfminer.six does not support the Identity crypt filter.
- **Flate without the zlib header:** qpdf decodes it; pdfium and pdfminer.six read no text from that page.
- **`page-tree`:** pdfium trusts the root's wrong `/Count` (5) and cannot load pages 4 and 5.
- **`header-junk-absolute`:** qpdf takes offsets after junk to count from the header and so rebuilds the table.
- **`wrong-length`:** pdfminer.six cuts the first stream at its short `/Length`; qpdf and pdfium recover it.

For the D6-D13 samples (every non-hostile file opens in qpdf with no problem, and every listed stream decodes to its
hash):

- **D6:** pdfminer.six and pdfium read every page; the hidden text layer's boxes agree with pdfium's to 0.031 point.
- **D7:** every expected path is matched to one of pdfium's path objects in content order with no point difference,
  and pdfium agrees on every fill rule, stroke (width, cap, join, dash) and grey or RGB colour. pdfium does not honour
  **knockout groups**: it blends the knockout group's two circles where they overlap ((134, 146, 184)) as it does the
  plain group's; the group's `probes` give what a knockout group must draw there ((140, 191, 242)). pdfminer.six warns
  on the two-colourant DeviceN ("only 1, 3 and 4 components") and reads the embedded **Type 1** font's text as
  `(cid:N)` (it does not take the program's built-in encoding); pdfium reads it. pdfium gives `F` for the Type 3
  check mark (as in D3). The title runs' tops and bottoms differ by 0.227 of the size (the standard 14, as in D3).
- **D8:** every written font opens in fontTools and passes the OpenType Sanitizer; WOFF, WOFF2 and the subsets keep
  every outline, advance and character of their source exactly and shape the sample lines the same (subsets: their
  glyphs); TrueType written as CFF keeps every glyph's bounds, area and centroid within 0.0001 unit, CFF written as
  TrueType within 1.06 units (tolerance 1.5). A contour of one point (DejaVu's `uni044B` holds one; it draws nothing)
  is left out of the CFF, which cannot hold it.
- **D9:** pdfium's form environment reads every widget's full name, kind, value, flags, check state and options as
  the JSON says (the inherited ones included). Under /NeedAppearances it makes the text and choice fields'
  appearances itself. The flattened page differs from pdfium's drawing of `forms-filled` only at glyph edges (up to
  84 levels on about 1% of the pixels): pdfium anti-aliases text in a widget's appearance differently from page text.
  The pypdfium2 wheel has no XFA: it warns ("compiled without XFA support", printed by the run) and reads the AcroForm
  twins.
- **D10:** pdfium lists every annotation with its subtype, rect, contents, quad points, vertices and ink strokes,
  every link's URI and destination page (named destinations resolved), the bookmarks with their open counts, every page
  label and the structure tree as the JSON says. It does not draw popups. pdfminer.six reads `redaction-applied`
  without the removed words.
- **D11:** qpdf performs every split, page operation, merge and the stamp (renaming the stamp's clashing resources)
  and pdfminer.six reads the pages the JSON says; qpdf reads `encrypted-aes256`'s permissions, opens it with either
  password and refuses it without. pdfminer.six splits a turned page's words into letters, so pages are matched by
  their letters there.
- **D12:** pyHanko validates every PAdES and CMS signature as its JSON says (digest, signature, chain against the test
  root alone, timestamp, coverage, DocMDP and the modification level; the revoked signer's revocation date and reason
  from the DSS). pyHanko does not read `adbe.pkcs7.sha1` or `adbe.x509.rsa_sha1` ("not a recognized SubFilter"): for
  those the SignedData is checked by pyHanko's `validate_sig_integrity` with the encapsulated digest compared, the
  PKCS #1 signature by cryptography, and the chain by pyhanko-certvalidator. pyHanko 0.38 has no PAdES-LT entry
  point any more (its AdES validation replaced it): B-LT is checked with a validation context made from the file's
  DSS alone, revocation required (hard-fail), at the timestamp's time.
- **D13:** qpdf and pdfium open every large file and count its pages, pdfium draws the first page, pdfminer.six reads
  the listed pages exactly; each of the 1,000 small files opens with one page and reads its text.

## Hostile files

Each must fail fast with an `InvalidDataException` naming the limit (or the cycle), opened with the `options` its JSON
gives. What the references do (Linux, the child limits above):

| File | Limit | qpdf | pdfminer.six | pdfium |
|---|---|---|---|---|
| `d1/hostile-flate-bomb` (1 MB to 1 GiB) | `MaxStreamBytes` | MemoryError | MemoryError | crashed |
| `d1/hostile-chained-bomb` (Flate twice, 2.4 KB to 1 GiB) | `MaxStreamBytes` | MemoryError | MemoryError | crashed |
| `d1/hostile-reference-cycle` | cycle | opened (empty contents) | timeout | opened |
| `d1/hostile-page-tree-cycle` | cycle | loop detected | opened (1 page) | opened (2 pages) |
| `d1/hostile-prev-loop` | cycle | opened | RecursionError | opened |
| `d1/hostile-deep-arrays` (100,000 deep) | `MaxDepth` | no /Root | opened | opened |
| `d1/hostile-deep-dicts` (100,000 deep) | `MaxDepth` | no /Root | opened | opened |
| `d1/hostile-huge-size` (`/Size` two billion) | `MaxObjects` | opened | opened | opened |
| `d1/hostile-objstm-self` | cycle | no /Root | RecursionError | opened |
| `d3/hostile-form-recursion` | `MaxDepth` | opened | opened | opened |

## Adding samples

Each phase has its module, `docfixtures/samples_dN.py`, whose `build(seed)` returns `Sample`s (`docfixtures/common.py`:
the bytes or none, the expected JSON, the files beside it, the reference kind and any further checks); `make_fixtures.py`
lists the phases in `PHASES`. The writer, filters, security handler, fonts and font programs (`fontprograms.py`),
images, vector content (`vector.py`), the font dumps (`fontdump.py`), the PKI and CMS (`pki.py`, `cms.py`) and the
reference runs are shared. Keep each new expected value made by construction and checked by a reference tool, every
byte seeded, and the text generic.
