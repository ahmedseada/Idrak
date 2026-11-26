# Idrak.Imaging.Plugins

Image and font formats the library does not ship, one namespace each (`Idrak.Imaging.Plugins.<Kind>`), written on the
library's public API only. It builds on Idrak.Imaging (and through it Idrak), whose OpenType reader and TrueType writer
it reaches directly. Nothing is registered: a format's reader and writer are made and called directly.

```csharp
using Idrak.Imaging.Plugins.Fonts;

var faces = new WoffFormat().Read(File.ReadAllBytes("fonts/report.woff"), "report");    // the font inside, by its flavor
using var file = File.Create("out/report.woff");
new WoffWriter().Write(faces[0], file, new FontWriteOptions());                         // written as TrueType, then packed
```

## The formats

| Namespace | Classes | Read more |
|---|---|---|
| `Idrak.Imaging.Plugins.Fonts` | `WoffFormat` ("woff", `.woff`) and `WoffWriter` ("woff", `.woff`) | `Fonts/WoffFormat.cs`, `Fonts/WoffWriter.cs` |

**WOFF** (`Idrak.Imaging.Plugins.Fonts`, plan 28's D8, decisions 20 and 21): WOFF 1.0 read and written, a font format
written from outside the library on its public API.

- **Read** (`WoffFormat`, "woff"): a file starting with `wOFF` is unpacked (`Decode`) into the sfnt font its encoder
  held, each table inflated through `ZLibStream` (or copied when stored) and checked against its length, then read as
  the header's flavor says by the library's OpenType reader (`OpenTypeFormat`, reached directly; another flavor is
  refused): the faces are that font's, the same glyphs, outlines, advances, characters and tables (their `Format` names
  that reader). A WOFF file is untrusted: every offset and length is checked before anything is inflated (the file's
  length, the directory in tag order, each table aligned, inside the file and overlapping no other block, no more than
  zlib can give from its stored bytes), the font inside may take at most `MaxFontBytes` (a quarter of the memory free
  when the file is read, within `ComputeResources.CpuMemoryLimit`; `IDRAK_WOFF_MAX_BYTES`, bytes or with k, m or g),
  and a table inflates to exactly the length it declares, never more. A damaged or hostile file is an
  `InvalidDataException` naming the cause.
- **Write** (`WoffWriter`, "woff"): the face written as an sfnt by the library's TrueType writer (`TrueTypeWriter`,
  reached directly; an `SfntWriter` naming another writer stops: a font writer by name is not on the new design yet),
  with the options given (subsets, layout tables, names, curve tolerance, hinting), then each table zlib-compressed at
  `CompressionLevel` (9 unless `IDRAK_WOFF_COMPRESSION` sets 0 to 9; 0 stores every table) when that is smaller, padded
  to 4 bytes, the directory by tag and the data in the font's own order. The same face and options write the same
  bytes, and the file unpacks to exactly the sfnt written first. `Encode` packs a TrueType or OpenType file directly.
- Not here: WOFF 2.0 (Brotli and the glyf, loca and hmtx transforms), WOFF metadata and private data (read past, never
  written) and OpenType CFF writing (decision 20).

License: Apache 2.0.
