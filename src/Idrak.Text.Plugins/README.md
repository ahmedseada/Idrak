# Idrak.Text.Plugins

The text model families: decoders, encoders and encoder-decoders, with their GGUF mappings, tool-call formats and RoPE
scalings. The library (`Idrak`, `Idrak.Text`) keeps only generic defaults and fallbacks; the families live here, one
namespace each (`Idrak.Text.Plugins.<Family>`), written on the library's public API only. It builds on Idrak.Text.

```csharp
using Idrak.Text.Plugins;

TextPlugins.RegisterAll();      // every family of the package; calling it again changes nothing
```

- `TextPlugins.RegisterAll()` registers every family, `TextPlugins.UnregisterAll()` takes them back (the registries
  are left as they were found), `TextPlugins.All` lists them.
- One family alone: its namespace's static `<Family>Plugin` class, with `Register()` and `Unregister()`.
- Each family is an `IIdrakPlugin` (`Idrak.Abstraction.Plugins`), recorded by name in `IdrakPlugins`; a plug-in
  registered later under the same name takes its place.
- The `idrak` tool registers this package at startup, before any `-P` plug-in (which can still take a family's place);
  `RegisterIdrakPlugin()` is the `-P` entry, for a host that loads `Idrak.Text.Plugins.dll` itself.

A model whose family is not registered is refused with "not registered", naming this package.

License: Apache 2.0.
