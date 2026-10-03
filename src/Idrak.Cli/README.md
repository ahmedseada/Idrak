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

## Model commands

| Command | What it does |
|---|---|
| `idrak pull MODEL` | Downloads a Hugging Face model (`owner/name`) or a GGUF file (`owner/name:Q4_K_M`, `owner/name/FILE.gguf`, a URL) into the cache; resumes interrupted downloads; `--dry-run` |
| `idrak list` (`ls`) | Cached models with sizes, formats and last use; `--all`, `--sort` |
| `idrak rm MODEL...` | Removes cached models (`-y`, `--dry-run`) |
| `idrak show MODEL` | Family, parameters, layers, context, vocabulary, RoPE, windows, chat template and tool-call format, files, license |
| `idrak search QUERY` | Hub models Idrak can load (registered families and GGUF architectures) |
| `idrak alias set NAME MODEL [-w F] [-k F]` / `alias list` / `alias rm NAME` | Short names for models and their formats, in the config's "aliases" |
| `idrak memory MODEL` | Memory per weight and KV format at a context length, against each device's memory |
| `idrak quantize MODEL -w F` | Size and perplexity of the model packed in a registered format; `-o DIR` writes it |
| `idrak merge MODEL ADAPTER -o DIR` | Merges a LoRA or DoRA adapter into the base weights |
| `idrak inspect FILE` (`i`) | Tensors and metadata of a GGUF, safetensors or `.ikm` file |
| `idrak verify MODEL` | Checks a model's files; `--read` reads every tensor, `--hub` compares hashes |
| `idrak convert IN OUT` | GGUF or Hugging Face folder to a Hugging Face folder (bf16, f16, f32) |
| `idrak diff A B` | Which tensors differ between two checkpoints, and by how much |
| `idrak families` | Supported model families and what each supports |

The cache layout (downloads/huggingface/models/OWNER/NAME/COMMIT, downloads/urls, gguf/) is the library's
(`ModelSource`, `Downloader`); `Shared/ModelCache.cs` reads it.
