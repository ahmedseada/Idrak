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
| `Idrak.Reinforcement.Plugins.CSharp` | `CSharpCodingPlugin` ("csharp"; requires "coding" and Idrak.Text.Plugins' "csharp-language") | `EnvironmentKinds`: `csharp`; `RewardFunctions`: `csharp-tests`, `csharp-build` | `CSharpEnvironmentVariant`, `CSharpRewards` |

**Coding** (`Idrak.Reinforcement.Plugins.Coding`): the coding task layer (`CodingTask` with its `task.json` layout,
`CodingSuite`, `CodingWorkspace`, `CodingVerifier` and `CodingOutcome`; an app evaluates a model on a suite with them,
running Idrak.Text's `CodingAgent` loop), the `coding` environment (an episode is a task drawn from its seed; setup once per
task; the assistant's tool calls run through `CodingTools`; the hidden tests' verification is the reward; options
`suite`, `reward`, `rounds`, `time`, `sandbox`, the limits, `workspaces` (`IDRAK_RL_WORKSPACES`), `keep`
(`IDRAK_RL_KEEP_WORKSPACES`), `refuse-test-edits`), its rewards, and the `container` sandbox: Docker or Podman
(`IDRAK_SANDBOX_PROGRAM`, else the first on the PATH), the image `IDRAK_SANDBOX_IMAGE` (none assumed), the network off unless
`IDRAK_SANDBOX_NETWORK`; `IDRAK_SANDBOX=container` makes the coding tools use it.
The task layer and the environment name no programming language: build output, dependency folders (linked, not
copied) and git excludes come from the coding languages (`CodingLanguages`, Idrak.Text; option `languages`: all, detected,
none or names, `IDRAK_CODING_LANGUAGES`), and the option `protect` (globs) adds files whose edits count as test edits. A
variant for one language (`CodingEnvironmentVariant`, `Idrak.Reinforcement.Plugins.Coding.Abstractions`) is the coding
environment under its own name with that language's defaults, its own options and its own info.

**C#** (`Idrak.Reinforcement.Plugins.CSharp`): the `csharp` environment, the coding environment on C# projects. A task
that names no commands restores (`dotnet restore`), builds (`dotnet build --no-restore`) and tests (`dotnet test
--no-build`); every verification build writes its errors and warnings to logs (`-fileLoggerParameters` errorsonly and
warningsonly) and every test run a TRX results file, read into the info (`CSharpInfo`) for the rewards `csharp-tests`
(the share of the tests passed: seven of ten is 0.7) and `csharp-build` (1 when it builds, less `error` 0.1 per error and
`warning` 0 per warning, at least `floor` −1). Options: `packages` (one package folder every episode restores from,
`NUGET_PACKAGES`; `IDRAK_CSHARP_PACKAGES`), `node-reuse` (false: `MSBUILDDISABLENODEREUSE=1`, so parallel episodes leave
no build nodes running), `build-server` (true: the compiler server is shared; measured 1.4 s a build against 7.8 s
without it); the protected files (`protect`, `IDRAK_CSHARP_PROTECT`, by default `Directory.Build.props`,
`Directory.Build.targets`, `Directory.Packages.props`, `global.json` and the `*Tests` / `*Test` projects) and the reward
(`csharp-tests`) are its defaults for the coding options. Sandbox: `process` with `dotnet` allowed, or `container` with an
image of your own that holds the .NET SDK (`IDRAK_SANDBOX_IMAGE`).

An environment whose kind is not registered is refused with "not registered", naming this package.

License: Apache 2.0.
