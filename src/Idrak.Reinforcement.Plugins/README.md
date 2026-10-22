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
| `Idrak.Reinforcement.Plugins.Coding` | `CodingPlugin` ("coding") | `EnvironmentKinds`: `coding`; `RewardFunctions`: `tests-passed`, `verify-fraction`, `rounds`, `tool-errors`; `CommandSandboxes`: `container` | `CodingEnvironmentKind`, `CodingVerifier` |

**Coding** (`Idrak.Reinforcement.Plugins.Coding`): the coding task layer (`CodingTask` with its `task.json` layout,
`CodingSuite`, `CodingWorkspace`, `CodingVerifier` and `CodingOutcome`; an app evaluates a model on a suite with them,
running Idrak.Text's `CodingAgent` loop), the `coding` environment (an episode is a task drawn from its seed; setup once per
task; the assistant's tool calls run through `CodingTools`; the hidden tests' verification is the reward; options
`suite`, `reward`, `rounds`, `time`, `sandbox`, the limits, `workspaces` (`IDRAK_RL_WORKSPACES`), `keep`
(`IDRAK_RL_KEEP_WORKSPACES`), `refuse-test-edits`), its rewards, and the `container` sandbox: Docker or Podman
(`IDRAK_SANDBOX_PROGRAM`, else the first on the PATH), the image `IDRAK_SANDBOX_IMAGE` (none assumed), the network off unless
`IDRAK_SANDBOX_NETWORK`; `IDRAK_SANDBOX=container` makes the coding tools use it.

An environment whose kind is not registered is refused with "not registered", naming this package.

License: Apache 2.0.
