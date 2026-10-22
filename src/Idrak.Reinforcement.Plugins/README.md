# Idrak.Reinforcement.Plugins

The reinforcement-learning environment families: environments, and the rewards they score with, that more than one app
uses. The library (`Idrak.Reinforcement`, `Idrak.Abstraction`'s reinforcement contracts) keeps only the general
algorithms; the environments live here, one namespace each (`Idrak.Reinforcement.Plugins.<Family>`), written on the
library's public API only. It builds on Idrak.Reinforcement and Idrak.Text (and through them Idrak).

```csharp
using Idrak.Reinforcement.Plugins;

ReinforcementPlugins.RegisterAll();      // every family of the package; calling it again changes nothing
```

- `ReinforcementPlugins.RegisterAll()` registers every family, `ReinforcementPlugins.UnregisterAll()` takes them back
  (the registries are left as they were found), `ReinforcementPlugins.All` lists them.
- One family alone: its namespace's static `<Family>Plugin` class, with `Register()` and `Unregister()`.
- Each family is an `IIdrakPlugin` (`Idrak.Abstraction.Plugins`), recorded by name in `IdrakPlugins`; a plug-in
  registered later under the same name takes its place.
- The `idrak` tool registers this package at startup, before any `-P` plug-in (which can still take a family's place);
  `RegisterIdrakPlugin()` is the `-P` entry, for a host that loads `Idrak.Reinforcement.Plugins.dll` itself.

## The families

| Namespace | Plug-in | Registers | Read more |
|---|---|---|---|

None yet.

An environment whose kind is not registered is refused with "not registered", naming this package.

License: Apache 2.0.
