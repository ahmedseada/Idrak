# Idrak.Imaging.Plugins

Image and font formats the library does not ship, as plug-ins: each registers into the library's registries from
outside it, one namespace each (`Idrak.Imaging.Plugins.<Kind>`), written on the library's public API only. It builds on
Idrak.Imaging (and through it Idrak), whose readers and writers it reaches through the registries at run time.

```csharp
using Idrak.Imaging.Plugins;

ImagingPlugins.RegisterAll();      // every plug-in of the package; calling it again changes nothing

var faces = FontFormats.Read("fonts/report.woff");             // read by its wOFF signature
FontWriters.Save(faces[0], "out/report.woff");                 // written by its extension
```

- `ImagingPlugins.RegisterAll()` registers every plug-in, `ImagingPlugins.UnregisterAll()` takes them back (the
  registries are left as they were found), `ImagingPlugins.All` lists them.
- One plug-in alone: its namespace's static `<Format>Plugin` class, with `Register()` and `Unregister()`.
- Each is an `IIdrakPlugin` (`Idrak.Abstraction.Plugins`), recorded by name in `IdrakPlugins`; a plug-in registered
  later under the same name takes its place. None is a library default: the registries name this package as their
  origin.
- The `idrak` tool registers this package at startup with the others (`IDRAK_PLUGIN_PACKAGES`, short name `imaging`),
  so `idrak data fonts` and `idrak data synth --fonts` read `.woff` files and list `.woff` in their help;
  `ImagingPlugins.Package` is the package (an `IIdrakPluginPackage`: its name and `All`); the assembly declares it with
  `[assembly: IdrakPluginPackage]`, which `-P` and `IdrakPlugins.Load` read when a host loads
  `Idrak.Imaging.Plugins.dll` itself.

## The plug-ins

| Namespace | Plug-in | Registers | Read more |
|---|---|---|---|
| `Idrak.Imaging.Plugins.Fonts` | `WoffPlugin` ("woff") | `FontFormats`: `woff` (`WoffFormat`, `.woff`); `FontWriters`: `woff` (`WoffWriter`, `.woff`) | `Fonts/WoffFormat.cs`, `Fonts/WoffWriter.cs` |

**WOFF** (`Idrak.Imaging.Plugins.Fonts`, plan 28's D8, decisions 20 and 21): WOFF 1.0 read and written, the proof that
a font format plugs into `FontFormats` and `FontWriters` from outside the library.

- **Read** (`WoffFormat`, "woff"): a file starting with `wOFF` is unpacked (`Decode`) into the sfnt font its encoder
  held, each table inflated through `ZLibStream` (or copied when stored) and checked against its length, then read by
  the first other registered font format that accepts it (the library's "opentype"): the faces are that font's, the
  same glyphs, outlines, advances, characters and tables (their `Format` names that reader). A WOFF file is untrusted:
  every offset and length is checked before anything is inflated (the file's length, the directory in tag order, each
  table aligned, inside the file and overlapping no other block, no more than zlib can give from its stored bytes), the
  font inside may take at most `MaxFontBytes` (a quarter of the memory free when the file is read, within
  `ComputeResources.CpuMemoryLimit`; `IDRAK_WOFF_MAX_BYTES`, bytes or with k, m or g), and a table inflates to exactly
  the length it declares, never more. A damaged or hostile file is an `InvalidDataException` naming the cause.
- **Write** (`WoffWriter`, "woff"): the face written as an sfnt by `SfntWriter` (the library's "truetype" unless set;
  through `FontWriters`, with the options given: subsets, layout tables, names, curve tolerance, hinting), then each
  table zlib-compressed at `CompressionLevel` (9 unless `IDRAK_WOFF_COMPRESSION` sets 0 to 9; 0 stores every table)
  when that is smaller, padded to 4 bytes, the directory by tag and the data in the font's own order. The same face and
  options write the same bytes, and the file unpacks to exactly the sfnt written first. `Encode` packs a TrueType or
  OpenType file directly.
- Not here: WOFF 2.0 (Brotli and the glyf, loca and hmtx transforms), WOFF metadata and private data (read past, never
  written) and OpenType CFF writing (decision 20; a later plug-in adds them through the same registries).

License: Apache 2.0.
