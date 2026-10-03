# idrak

One command-line tool for Idrak (plans/idrak-cli.md). This is the skeleton: the command dispatcher, the common
options, help, the config file, `--plugin` loading and JSON output, with `idrak version` as the only command. The
command groups are empty files under `Commands/`, filled in as the plan is built.

```
dotnet run -c Release --project src/Idrak.Cli -- help
dotnet run -c Release --project src/Idrak.Cli -- version --json
```

## Setup and health

```
idrak doctor                     # what works on this machine and what to fix (--android, --network, --fix)
idrak dev -j                     # every device as JSON
idrak report --tests --readme    # the README's tested-on rows for this machine
idrak env --all                  # every environment variable Idrak reads; idrak help env without values
idrak help topics                # concept pages: devices, formats, models, precision, plugins, config, env, exit codes
eval "$(idrak completion bash)"  # completion (zsh, fish, pwsh too)
```

Also `version`, `init`, `cache info/clear`, `config get/set/unset/list`, `plugins list`, `formats`, `update`,
`login`/`logout` and `setup android`; `idrak help COMMAND` shows each one's options, examples and environment
variables.

## Shared helpers

Every command takes `@file` (arguments from a file, one per line), `--log FILE`, `-O/--output FILE`,
`--format text|json|csv|md`, `--color auto|always|never`, `--plain`, `--offline`, `--threads N`, `--seed N` and
`--timeout DURATION` besides the common options; `CommandContext` has an accessor for each. `Shared/Terminal.cs`
(colour, questions with `--yes` and `--dry-run`, Ctrl+C), `Shared/Progress.cs` (progress lines),
`Shared/EnvironmentVariables.cs` (the variable table behind `idrak env` and each command's help) and
`Shared/HelpTopics.cs` are there for every group.

## Adding a command

A command derives from `Command` (name, one-line summary, usage, its value options and flags, `Run(CommandContext)`
returning an exit code) and is listed in its group's `All` (for example `Commands/HealthCommands.cs`). Multi-word
names ("cache info") are matched before shorter ones. `CommandContext` gives the parsed arguments, the common options
(`Device`, `CacheFolder`, `Json`, `Quiet`, `Verbose`), the config file and output helpers (`Write`, `Detail`, `Table`,
`WriteJson`, `Error`). A usage error is a `UsageException` (exit code 2); any other exception prints its message
(with the stack under `IDRAK_TRACE=1`) and exits with 1.

The assembly is named `Idrak.Cli`, not `idrak`: assembly names ignore case, so `idrak` would clash with the Idrak
library. The installed command is still `idrak` (`ToolCommandName`).
