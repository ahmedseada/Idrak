# Plug-in 1. Idrak.HarfBuzz: HarfBuzz as a second shaper

Status: planned 2026-10-14, not started (owner's decision: the seven Arabic fonts in use are already checked against
HarfBuzz; build this when a new font or script needs checking, or an app wants HarfBuzz to shape).

## Why

Idrak shapes text with its own C# shaper (plan 18, phase 2), which is meant to give HarfBuzz's glyphs, clusters, advances
and offsets. Checking that today takes Python: `idrak data fonts DIR --write-lines lines.txt`, then
`python tools/fonts/harfbuzz_reference.py DIR lines.txt -o hb.json` (pip install uharfbuzz), then
`idrak data fonts DIR --harfbuzz hb.json`. The check found four real shaping bugs on Google Fonts' Amiri, Cairo and
Reem Kufi (fixed 2026-10-14, now glyph for glyph on 200 lines in seven fonts), so it is worth having in one step and
without Python.

HarfBuzz cannot be in the library: it is native code (a binary per platform and architecture through HarfBuzzSharp),
which rule 70 and decision 10 keep out, and the library already has a shaper of its own. Speed is not the reason: HarfBuzz
is fast and the interop cost per run is small. A plug-in keeps the library free of it and lets whoever wants it opt in.

It is a plug-in of its own, not part of an Arabic package: the Arabic shaper is script support every joining script needs
(Arabic, Persian, Urdu, Syriac, N'Ko) and stays in the library, and HarfBuzz checks every script, not Arabic alone.

## Where it lives

| Piece | Where | Why |
|---|---|---|
| `Idrak.HarfBuzz` (the plug-in) | `samples/HarfBuzz/Idrak.HarfBuzz` | an outside plug-in like `samples/Gemma3Vision`: references Idrak and Idrak.Imaging as packages, and HarfBuzzSharp (a pinned version) |
| Named shapers: a way to pick a shaper by its name, beside the by-script table | `TextShapers` (Idrak.Abstraction) | the only library change: today a shaper is found by script ("Arab", "*"); a second shaper for the same script needs a name to be chosen by (see phase 2) |
| `--against NAME` | `idrak data fonts` (Idrak.Cli) | compares the library's shaping with a named shaper in process, no files |
| `Shaper` setting | `TextStyle` (Imaging), `SyntheticPageOptions`, `idrak data synth --shaper`, the Legal OCR app's `Synthetic.Shaper` | nothing fixed: pages may be shaped by a chosen shaper |
| Tests | `samples/HarfBuzz/Idrak.HarfBuzz.Tests` (its own runner and filter, like the app tests) | the main test project must not reference a native dependency |

## Phases

1. **The shaper.** `HarfBuzzShaper : ITextShaper` (`Name` "harfbuzz"):
   - Faces: a HarfBuzz face built from `IFontFace.Table(tag)` (HarfBuzzSharp's table callback), so every face Idrak reads
     works, a collection's fonts and variable fonts at their default instance included. A face and a font are made once
     per `IFontFace` (a `ConditionalWeakTable`, rule 75) and are safe to share between threads: synthetic pages shape on
     every processor.
   - A run: the text as UTF-16 (clusters come back as UTF-16 indices, as Idrak's), direction from `TextRun.RightToLeft`,
     the script from its ISO 15924 code, the language from its BCP 47 tag, `TextRun.Clusters` to HarfBuzz's cluster
     level, features as `+tag` / `-tag`.
   - The answer: `ShapedGlyph`s in font units (the font's scale left at units per em), as the library's shaper gives them.
   - `HarfBuzzPlugin.Register()` and `RegisterIdrakPlugin()` (for `-P`) register it by name only: loading the plug-in
     changes nothing until a shaper is asked for by name.
   Done when: on `tests/Idrak.Tests/data/fonts/arabic` (Amiri, Cairo, Reem Kufi subsets) the plug-in gives the same
   output as `arabic/shaping.json` on all 100 lines in each font (or the differences are explained by HarfBuzz versions,
   see risks).

2. **Named shapers in the library.** `TextShapers` gains a second table, by the shaper's own name:
   `TextShapers.RegisterNamed(ITextShaper)`, `TextShapers.Named(name)` (throws "not registered" naming what is), and
   `TextShapers.Shape(..., shaper: name)`. The library registers its own as "general" and "arabic". The inventory test and
   the API dump are updated. Done when: a test registers a stub by name, shapes with it by name, and the by-script table
   is unchanged.

3. **Front ends.**
   - `idrak data fonts DIR --against harfbuzz -P Idrak.HarfBuzz.dll`: every sample line shaped by the library and by
     the named shaper, compared glyph for glyph, the same report as `--harfbuzz FILE` (which stays, for references made
     elsewhere). With no `-P`, the message names the plug-in to load.
   - `TextStyle.Shaper` (null: by script, as today), `SyntheticPageOptions` passes it on, `idrak data synth --shaper
     NAME`, the Legal OCR app's `Synthetic.Shaper` setting with a plug-in path setting to load it.
   Done when: the CLI compares the seven fonts in one command, and a synthetic page made with `--shaper harfbuzz` has the
   same text and, for the checked fonts, the same pixels as one made by the library's shaper.

4. **Tests and the owner's run.** The plug-in's test runner (CPU only, no device): the shaper against `arabic/shaping.json`,
   the face bridge on a collection and a variable font, the named registry, the CLI comparison. The owner runs it on
   Windows with the seven Google Fonts families and gets "200 of 200 lines identical" for each.

## Risks

- **HarfBuzz versions.** HarfBuzzSharp bundles its own HarfBuzz (it trails the newest release); the fixtures were made with
  HarfBuzz 14 through uharfbuzz. Shaping rarely changes between versions, but a difference in an edge case would show as a
  mismatch that is neither shaper's bug. The plug-in reports the HarfBuzz version it runs, and the fixtures' version is
  written beside them; a mismatch is first checked against the version.
- **Platforms.** HarfBuzzSharp ships native assets for Windows, Linux and macOS (x64, ARM64); Android and iOS need their
  own native asset packages. The plug-in states the platforms it has; the library is unaffected anywhere.
- **Size.** The native library adds about 1–3 MB per platform to an app that references the plug-in, none to one that
  does not.
- **Trimming and AOT.** HarfBuzzSharp is P/Invoke; the plug-in is checked under trimming warnings before it ships.

## Not in this plan

- Moving the Arabic shaper, Arabic random text or the Hijri slot out of the library: they are script and language
  support with no dependency, and stay.
- An `Idrak.Nlp.Arabic` package (normalization of alef and yeh forms, tashkeel removal for scoring, reading numbers):
  worth a plan of its own once there is Arabic language logic to put in it; it would not reference HarfBuzzSharp.
