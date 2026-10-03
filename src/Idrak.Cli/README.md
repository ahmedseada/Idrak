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

## Train and data

```
idrak tune init -b Qwen/Qwen3-0.6B --data chats.jsonl      # a commented tune.json
idrak tune --config tune.json                               # idrak-tune's train with those settings
idrak tune train qwen chats.jsonl -o adapters/chat -w int4  # or every option on the command line
idrak train network.json --data houses.csv -t price -o houses.ikm
idrak train cnn.json --data ./shapes -d vulkan:0            # a folder of class folders of images
idrak predict houses.ikm -i new-houses.csv -o priced.csv
idrak runs list && idrak runs show NAME && idrak runs compare A B
idrak resume NAME --epochs 20 && idrak package --model RUN_FOLDER -o model.ikm
idrak data preview train.parquet && idrak data validate chats.jsonl --as chat && idrak data stats chats.jsonl -m qwen
idrak data convert alpaca.json chats.jsonl --as chat && idrak data split chats.jsonl -t label
```

`idrak tune` and `idrak data` run the code of idrak-tune and idrak-data (`Commands/Train/TuneTool.cs`,
`Commands/Data/DataTool.cs`); those tools compile the same files as thin forwarders. `idrak train` writes a run folder
under `CACHE/runs` (run.json, network.json, scalers, `last.ikw`/`best.ikw`, `log.jsonl` in the library's telemetry
format) that `runs`, `resume` and `package` read.
