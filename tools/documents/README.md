# tools/documents: the PDF samples of plan 28

`make_fixtures.py` writes plan 28's PDF samples (phase D5, decision 13) and what a reader must get from each: 75 files
for phases D1-D4 (the file, encryption, text, images and embedded files), 10 of them hostile. Nothing it writes is
committed: everything goes into the folder given by `-o` (default `test-output/documents/samples`, which git ignores),
and the tests read it from `IDRAK_DOCUMENTS_SAMPLES`.

```bash
# Linux (a virtual environment outside the repository is best)
python3 -m venv ~/venvs/idrak-docs && ~/venvs/idrak-docs/bin/python -m pip install -r tools/documents/requirements.txt
~/venvs/idrak-docs/bin/python -I tools/documents/make_fixtures.py -o test-output/documents/samples
```

```powershell
# Windows PowerShell, from D:\Projects\Idrak
python -m pip install -r tools\documents\requirements.txt
python -I tools\documents\make_fixtures.py -o test-output\documents\samples
```

- `-o DIR` the output folder. It is emptied first; a non-empty folder without a `manifest.json` is refused (never a
  folder the tool did not write).
- `--only d1,d3` some phases only; `--no-reference` the samples and their JSON without the reference tools' runs;
  `--seed S` another seed (default `2026`).
- `--compare A B` compares two output folders: the manifests byte for byte and every file's SHA-256 (exit code 1 on a
  difference).

The run prints each sample as it is written with the time spent, the rate and the time left, then the counts and the
reference tools' notes. A whole run takes about a minute (most of it the hostile files' probes).

## Reproducible

Every byte comes from the seed and the pinned versions (`requirements.txt`): no dates, no random identifiers (the file
`/ID` is the MD5 of the seed and the sample's name; AES vectors, the revision 5-6 salts and keys and the padding of `/U`
come from SHA-256 of the seed, the sample and a counter), fonts subset with their timestamps kept, ICC profiles with
their dates cleared. Flate streams and PNGs use the interpreter's zlib, whose version the manifest records with the
packages'. Two runs into two folders give the same files, the manifest included (checked: 395 files and the manifest
identical).

## The tools, and why

- **Every PDF byte is written here** (`docfixtures/pdf.py`, `filters.py`, `security.py`), not by a PDF library: the
  cross-reference kinds, object streams, hybrid files, incremental updates, damaged tables, wrong lengths, each filter
  and predictor and the encryption are exactly what each sample says, which no writer library lets one choose.
- **qpdf** (`pikepdf`) checks every non-hostile file (`check_pdf_syntax`) and decodes every stream the JSON lists, whose
  bytes must hash as the generator wrote them: the filters, predictors and the encryption are proven by an independent
  reader, not by the generator's own decoders.
- **pdfminer.six** is the reference reader: its text per page. It also supplies the encodings' tables (`latin_enc`) and
  the standard 14 fonts' AFM widths the samples are laid out with.
- **pdfium** (`pypdfium2`) is the reference renderer: each page as PNG at 100 DPI, and its text with each character's
  loose box, against which every run box of the JSON is checked (`reference.pdfium.boxes`).
- **fontTools** subsets the embedded fonts and gives glyph names their text (the Adobe Glyph List rules);
  **HarfBuzz** (`uharfbuzz`) shapes the composite fonts' text (GSUB only, so every glyph advances by its width);
  **python-bidi** checks the visual order of every right-to-left and mixed line; **cryptography** gives AES (RC4 is
  written here); **Pillow** writes the JPEGs and the scanned page.
- **Fonts:** only the repository's test fonts with their licences beside them: the DejaVu Sans subset
  (`tests/Idrak.Tests/data/fonts/dejavu-sans-subset.ttf`, `LICENSE-DejaVu.txt`) and Amiri (`arabic/amiri.ttf`,
  `OFL-amiri.txt`).

## The samples

| Folder | Samples |
|---|---|
| `d1/` | `xref-table`, `xref-stream` (PNG Up predictor), `object-streams` (`/Extends`), `hybrid` (`/XRefStm`), `incremental` (two updates through `/Prev`: a stream replaced, a page added, the title changed, an object freed), `incremental-stream` (an update as a cross-reference stream replacing an object held in an object stream), `header-junk` (590 bytes before `%PDF-`, offsets from the header) and `header-junk-absolute` (offsets from the file's start), `broken-xref` (every offset 11 bytes short: repair), `missing-xref` (no table, no startxref: repair), `wrong-length` (`/Length` short, long, indirect), `filters` (one page a filter: Flate with and without the zlib header, ASCIIHex with an odd last digit, ASCII85 with `z`, LZW with and without early change, RunLength, three chains, none), `predictors` (PNG 10-15 at 1, 8 and 16 bits, TIFF 2 at 4, 8 and 16 bits, LZW with Paeth), `info` and `info-partial` (PDFDocEncoding, UTF-16BE, UTF-8, escapes, octal, a line continuation, hex strings, `D:` dates whole, partial and odd), `page-tree` (inherited MediaBox, Resources and Rotate; a wrong `/Count`), and nine hostile files (below) |
| `d2/` | `plain` (the unencrypted twin), then for each of `r2-rc4-40`, `r3-rc4-128`, `r4-rc4-128`, `r4-aes-128`, `r4-aes-128-metadata-plain` (`/EncryptMetadata false`), `r4-aes-128-identity-strings` (`/StrF /Identity`), `r4-aes-128-xref-stream`, `r5-aes-256`, `r6-aes-256`, `r6-aes-256-object-streams` an `-empty` (empty user password) and a `-user` (`user-pass-1`) file, owner password `owner-pass-1`; `r6-aes-256-unicode` (`Grüße-2024`); `unsupported-handler` (`/Adobe.PubSec`) |
| `d3/` | `standard14` (not embedded, AFM widths), `encodings` (MacRoman, Standard, WinAnsi, `/Differences` with `uni2192`, `f_f`, `a.sc`, `Euro`, `uni00410042`, `u1F4C8` and the unmappable `g17`), `truetype-simple` (with and without ToUnicode), `type0-identity`, `type0-no-tounicode`, `type0-no-cmap`, `cidtogidmap-stream`, `type3`, `text-state` (TJ, Tc, Tw, Tz, TL, T*, `'`, `"`, Ts), `render-modes` (0-7), `actual-text`, `rtl` (Amiri), `mixed-direction` (DejaVu), `vertical` (Identity-V), `rotated-pages` (0, 90, 180, 270, text turned by Tm and cm), `cropbox`, `text-matrix` (sizes from Tm and the CTM, a form XObject), `hostile-form-recursion` |
| `d4/` | `colour-spaces` (Gray, RGB, CMYK, Indexed over RGB, grey and CMYK, ICCBased N 3 and 1, CalGray, CalRGB, Lab), `bit-depths` (1, 2, 4, 8, 16 bits, rows ending mid-byte), `decode-arrays`, `masks` (stencils, soft masks, an explicit mask, a colour key), `dct` (grey, 4:2:0, progressive 4:4:4, Adobe CMYK), `inline-images` (abbreviations, a named colour space, data holding ` EIQ`), `image-placement` (turned, slanted, flipped, mirrored, on a turned page), `scan-ocr` (an A4 grey JPEG with a hidden OCR text layer, D3 and D4), `attachments` (an EmbeddedFiles name tree with Kids and Limits, a file-attachment annotation) |

Text in the samples is ordinary business and technology text in English and Arabic (reports, meetings, orders, a
newsletter).

## The files

Beside each `NAME.pdf` in its phase's folder:

| File | What |
|---|---|
| `NAME.json` | what the file holds and what a reader must get (below) |
| `NAME.text.json` | the reference tools' text: `pdfminer` (pdfminer.six, a string a page) and `pdfium` (a string a page) |
| `NAME.pN.png` | page N as pdfium draws it, RGB, at `reference.dpi` (100) |
| `NAME.pN.imK.png` | the expected pixels of page N's K-th image in content order (`.smask.png`, `.mask.png`: its soft mask and explicit mask) |
| `NAME.<file name>` | the bytes of an embedded file (`attachments` only) |

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

## Adding the samples of later phases (D6-D13)

The samples for writing (D6), vector content and drawing (D7), fonts in the common formats with the font tools' dumps
(D8), forms (D9), annotations and navigation (D10), editing (D11) and signatures with a generated test certificate
(D12), and the large sets of D13, come with those phases. Each phase adds a module `docfixtures/samples_dN.py` whose
`build(seed)` returns `Sample`s (`docfixtures/common.py`: the bytes, the expected JSON, the files beside it), and one
line in `make_fixtures.py` (`PHASES` and `builders`); the writer, filters, security handler, fonts, images and the
reference runs are shared. Keep each new expected value made by construction and checked by a reference tool, every
byte seeded, and the text generic.
