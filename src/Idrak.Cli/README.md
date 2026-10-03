# idrak

One command-line tool for Idrak (plans/idrak-cli.md). This is the skeleton: the command dispatcher, the common
options, help, the config file, `--plugin` loading and JSON output, with `idrak version` as the only command. The
command groups are empty files under `Commands/`, filled in as the plan is built.

```
dotnet run -c Release --project src/Idrak.Cli -- help
dotnet run -c Release --project src/Idrak.Cli -- version --json
```

## Serve

| Command | What it does |
|---|---|
| `idrak serve MODEL [MODEL...]` (`s`) | The chat API (`/api/chat`, `/api/tags`, `/api/ps`, `/api/version`) and the OpenAI-style API (`/v1/models`, `/v1/chat/completions`, `/v1/completions`) on one port, streaming and tool calls; models load on first use and unload after `--keep-alive`; `--host`, `--port`, `--api-key`, `--cors`, `--max-concurrency`, `--metrics`, `--log-requests` |
| `idrak ui MODEL` | The same server with its web chat page (`/ui`) opened in the browser |
| `idrak server ps` (`ps`), `server stop`, `server load MODEL`, `server unload MODEL` | A running server's models (state, memory, context, last use) and control |
| `idrak server keys add/list/rm` | API keys for serve, stored as hashes in the config |
| `idrak api PATH [JSON]` | Calls an endpoint of a running server |
| `idrak ping [URL]` | Checks a server (any compatible one): which API answers, models, latency |
| `idrak mcp serve TOOLS.dll` | Serves an assembly's `[Tool]` methods over MCP (standard input/output) |

```
idrak s qwen phi -p 8080 --api-key $KEY        # two models on one port
idrak ps
idrak api /v1/chat/completions '{"model":"qwen","messages":[{"role":"user","content":"Hi"}]}'
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
