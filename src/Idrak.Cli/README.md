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
