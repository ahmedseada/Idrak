# 18. Synthetic document pages: tuning data from text

Status: phases 1 and 2 done 2026-10-13 (fonts: reader and rasterizer, tests "fonts:"; shaping: bidi, itemizer, the general and Arabic shapers, every GSUB and GPOS lookup kind, tests "shaping:", glyph for glyph as HarfBuzz 14 on 33 strings); 2b: CFF outlines (Type 2 charstrings, subroutines; CID-keyed read, no fixture yet); 3: pages (`TextLayout`: wrapping, bidi lines, alignment and justification, fallback fonts; `PageLayouts` "letter" and "plain", every choice from the page's seed and the style's ranges and chances; tests "pages:"). Planned 2026-10-13. Owner's choice: the library's own font engine (no dependency). Arabic is the main goal; the
engine serves any language (owner, 2026-10-13).

## Why

A Qwen2.5-VL 3B adapter tuned one epoch on bakrianoo's pages (clean, digitally made: "digital, high quality" in their
metadata) reads clean pages well and invents text on old, pale or small scans (a 1426H letter came back as 1440s dates and
"وزارة العدل"); the Read tab's CER over the JSON values is about 50 % on the first 20 evaluation pages. Its training pages
have no header, footer or subject block. The owner's idea, kept in the library for anyone with a good text source: render
the same texts as pages with many fonts, sizes, layouts (header, subject, body, footer, signature) and scan qualities
(rotation, blur, noise, faded ink, JPEG, low resolution), and tune on those. The answer of each page is its text, so the
pages need no transcription.

## Where it lives (decision 10, CLAUDE.md)

| Piece | Package | Why |
|---|---|---|
| Contracts and registries: `IFontFace` / `FontFormats` (a font file read into glyphs and metrics), `ITextShaper` / `TextShapers` (text to positioned glyphs), `IPageLayout` / `PageLayouts` (a page's blocks placed) | Idrak.Abstraction | used by Imaging (implements), Nlp (builds data), Vision and apps (now or later) |
| The OpenType reader (TrueType outlines first), the rasterizer, the shaper (bidi, Arabic joining, GSUB, GPOS), page composition, scan degradations as `ImageTransforms` | Idrak.Imaging | drawing is image work; no dependency beyond Idrak |
| The data builder: records (ShareGPT / JSON Lines with JSON answers, or plain text) to rendered variants, written as ShareGPT plus an images zip that `ConversationTuning` reads | Idrak.Nlp | owns the tuning data formats; references Imaging |
| Front ends: `idrak synth` (CLI) and a "Make training pages" panel on the Legal OCR app's Fine-tune tab | Idrak.Cli, the app | |

## Phases

Any language: the font reader and the rasterizer know nothing of scripts. Text is split into runs by Unicode script and
direction; each run goes to the shaper registered for its script (`TextShapers`): a general one for scripts without
reordering or joining (Latin, Cyrillic, Greek, Armenian, Georgian, Hebrew, CJK, ...: cmap, then the font's GSUB and GPOS for
that script), the Arabic one for the Arabic script (Arabic, Persian, Urdu, ...; Syriac and N'Ko join the same way), and later
others (Indic syllables and reordering, Thai, Khmer, Myanmar) as their own entries. A script with no shaper registered falls
back to the general one, which is how a font without layout tables is drawn anyway. CFF outlines (most .otf, the CJK Noto
fonts) come after TrueType outlines.

1. **Fonts.** OpenType reader in C#: the table directory (and TrueType collections), `head`, `hhea`, `maxp`, `hmtx`,
   `cmap` (formats 4 and 12), `loca` / `glyf` with composite glyphs, `GDEF`. Glyph outlines flattened (quadratic
   curves) and filled by an anti-aliased scanline rasterizer (nonzero winding, exact area coverage per pixel), no
   hinting. Checked against FreeType (Pillow) renders of the same glyphs: ink masks agree within a small tolerance.
2. **Shaping.** Script runs (a Unicode script table made by a tool from the Unicode data, as .NET has none); a reduced Unicode bidirectional algorithm (paragraph direction, strong and weak types, numbers kept left
   to right inside right-to-left text, neutrals, reordering of a line); Arabic joining from the Unicode joining types;
   GSUB lookups 1, 2, 4, 5, 6, 7 (and 8) with lookup flags and mark filtering, the Arabic features in order (ccmp,
   isol, fina, medi, init, rlig, calt, liga); GPOS 1, 2, 3, 4, 5, 6, 7, 8, 9 (kerning, cursive attachment, marks).
   Checked glyph for glyph against HarfBuzz (uharfbuzz) on fixed strings in several languages: glyph ids, advances and
   offsets equal.
2b. **CFF outlines** (Type 2 charstrings), so .otf fonts and the CJK fonts draw; **more shapers** (Indic, Thai, ...) as
   their own registered entries.
3. **Pages.** A page layout of blocks: header (two or three columns: the state and ministry lines, the number and date
   fields), subject line, body (wrapped right to left, justified or not, numbered items), signature block, footer; each
   block with its own font, size and spacing drawn from ranges; the page's text in reading order comes back with the
   image, so the answer always matches what is drawn.
4. **Scans.** Degradations as registered `ImageTransforms` steps with ranges sampled per page: rotation (fraction of a
   degree to a few degrees), perspective, blur, noise, speckle, faded ink, background tint and stains, low resolution
   (down then up), JPEG, bleed-through. A profile names a mix and how often each step is taken.
5. **Data.** `SyntheticPages` (Idrak.Nlp): from records with JSON answers (the full_text field and the others), N
   variants each, seeded and repeatable; header, subject and footer text drawn from the record's fields or from lists
   given, written into the answer too; output ShareGPT + images zip + a manifest of how each page was made (seed, font,
   size, steps). Mixed with the real pages at a chosen share.
6. **Front ends.** `idrak synth` and the app's panel (fonts folder, variants per record, profile, preview of a few pages,
   progress, then "Tune on these").
7. **Proof.** Tune the 3B on real + synthetic pages; the Read tab's CER over values on the same 20 pages, before and after.

## Tests (owner rule: targeted, CPU)

- Fixtures under `tests/Idrak.Tests/data/fonts`: a subset of DejaVu Sans (Arabic, Latin, digits, its GSUB and GPOS;
  the Bitstream Vera / DejaVu license allows it), HarfBuzz reference shapings and FreeType reference masks made by
  `tools/fonts/make_font_references.py` (reruns give the same bytes).
- Every phase's group in the test program ("fonts:", "shaping:", "pages:", "scans:", "synthetic data:").
