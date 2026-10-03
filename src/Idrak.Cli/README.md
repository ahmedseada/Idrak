# idrak

One command-line tool for Idrak (plans/idrak-cli.md). This is the skeleton: the command dispatcher, the common
options, help, the config file, `--plugin` loading and JSON output, with `idrak version` as the only command. The
command groups are empty files under `Commands/`, filled in as the plan is built.

```
dotnet run -c Release --project src/Idrak.Cli -- help
dotnet run -c Release --project src/Idrak.Cli -- version --json
```

## Adding a command

A command derives from `Command` (name, one-line summary, usage, its value options and flags, `Run(CommandContext)`
returning an exit code) and is listed in its group's `All` (for example `Commands/HealthCommands.cs`). Multi-word
names ("cache info") are matched before shorter ones. `CommandContext` gives the parsed arguments, the common options
(`Device`, `CacheFolder`, `Json`, `Quiet`, `Verbose`), the config file and output helpers (`Write`, `Detail`, `Table`,
`WriteJson`, `Error`). A usage error is a `UsageException` (exit code 2); any other exception prints its message
(with the stack under `IDRAK_TRACE=1`) and exits with 1.

The assembly is named `Idrak.Cli`, not `idrak`: assembly names ignore case, so `idrak` would clash with the Idrak
library. The installed command is still `idrak` (`ToolCommandName`).

## Design commands

| Command | What it does |
|---|---|
| `idrak suggest DATA` (`sg`) | Reads a table, image folder, text, chat or preference rows; writes `network.json` (builder JSON), `train.json`, `prep.json`, or `tune.json` with `--base`; `--search N` measures N candidates on the device, `--explain` prints the rules |
| `idrak explain network.json` (`x`) | Layers, output shapes, parameters, FLOPs per sample, inference and training memory against the device |
| `idrak viz network.json` | The network as text, Mermaid (`--format mermaid`) or SVG (`--format svg`) |

```
idrak sg houses.csv -t SalePrice -e -o ./run
idrak x ./run/network.json
idrak viz ./run/network.json --format mermaid
```
