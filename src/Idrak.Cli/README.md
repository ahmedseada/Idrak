# idrak

One command-line tool for Idrak (plans/idrak-cli.md). This is the skeleton: the command dispatcher, the common
options, help, the config file, `--plugin` loading and JSON output, with `idrak version` as the only command. The
command groups are empty files under `Commands/`, filled in as the plan is built.

```
dotnet run -c Release --project src/Idrak.Cli -- help
dotnet run -c Release --project src/Idrak.Cli -- version --json
```

## Measure and retrieval

| Command | What it does |
|---|---|
| `idrak bench [MODEL]` (`b`) | Prompt and generation tokens per second, GFLOP/s and memory of a model; without one, the kernel benchmarks (`--kernels matmul,gemv,attention`, `--small`); `--devices all`, `--matrix` (weight x KV formats), `--save NAME`, `--compare NAME` |
| `idrak eval MODEL SET.jsonl` | Answer metrics (number, exact, contains, F1) on held-out conversations |
| `idrak perplexity MODEL FILE` | Perplexity of a text, to compare weight formats and fine-tunes |
| `idrak profile MODEL` | Time per layer, operation and kernel of one decoding step |
| `idrak check MODEL --reference FILE` | Token ids, chat templates, logits and greedy output against a transformers reference |
| `idrak tuning show` | The kernel choices kept in the CPU, CUDA and Vulkan tuning caches; `--reset` |
| `idrak rag index DIR -o INDEX` | Chunks a folder's text files into a BM25 index, hybrid with `-m MODEL` |
| `idrak rag search "query" --index INDEX` | The best passages with scores |
| `idrak rag ask "question" --index INDEX -m MODEL` | An answer citing the passages |
| `idrak rag eval --index INDEX --questions FILE` | Hit rate and MRR on questions with known documents or passages |

## Adding a command

A command derives from `Command` (name, one-line summary, usage, its value options and flags, `Run(CommandContext)`
returning an exit code) and is listed in its group's `All` (for example `Commands/HealthCommands.cs`). Multi-word
names ("cache info") are matched before shorter ones. `CommandContext` gives the parsed arguments, the common options
(`Device`, `CacheFolder`, `Json`, `Quiet`, `Verbose`), the config file and output helpers (`Write`, `Detail`, `Table`,
`WriteJson`, `Error`). A usage error is a `UsageException` (exit code 2); any other exception prints its message
(with the stack under `IDRAK_TRACE=1`) and exits with 1.

The assembly is named `Idrak.Cli`, not `idrak`: assembly names ignore case, so `idrak` would clash with the Idrak
library. The installed command is still `idrak` (`ToolCommandName`).
