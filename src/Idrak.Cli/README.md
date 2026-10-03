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

## Run models

`chat` (`c`), `run` (`r`), `batch`, `compare`, `complete`, `embed`, `tokenize`, `template`, `agent`, `tools list` and
`tools test` (`Commands/Run/`). Every one takes a Hugging Face id, a model folder, a `.gguf` file or an alias, with
`-w, --weights` and `-k, --kv`; the generation settings (`-s, --system`, `--temperature`, `--top-k`, `--top-p`,
`--max-tokens`, `--seed`, `--think`, `--no-think`, `--tools FILE.dll`) are shared through `Shared/GenerationSettings.cs`.

```
idrak chat Qwen/Qwen3-0.6B -s "Answer briefly." --history talk.json
cat notes.txt | idrak run qwen "Summarize in three bullets"
idrak batch qwen -i prompts.jsonl -o answers.jsonl
idrak tokenize qwen --chat "Hello" --count
idrak agent qwen "Add a test for Parse" --workspace ./src
```

Chat reads its input through `Shared/StandardInput.cs`, so the tests drive a conversation (with slash commands)
in-process.
## Developer commands

| Command | What it does |
|---|---|
| `idrak new console\|webapi\|rag\|plugin NAME` | A ready-to-build project referencing the Idrak packages (`--source DIR`: the projects of a checkout); the plug-in registers a packed weight format and a network step and has tests on the public API only |
| `idrak test [--filter TEXT] [-d NAME]` | Runs tests/Idrak.Tests of the checkout around the current folder |
| `idrak onnx import FILE.onnx` | Rebuilds the model from Idrak layers and saves a model package (.ikm) |
| `idrak onnx export MODEL.ikm\|network.json` | Writes an .onnx file |
| `idrak onnx check FILE.onnx` | Compares Idrak's import with a round trip, ONNX Runtime (when Idrak.Onnx.Runtime is loaded) and reference outputs |
| `idrak kernels dump [ptx\|spirv\|hip]` | The generated GPU kernels, for debugging |
| `idrak trace [-o FILE.jsonl] -- COMMAND ...` | Runs a command with telemetry printed live or written to JSON Lines |
| `idrak demo xor\|spirals\|shapes\|gpt` | Trains a small sample in seconds on the device, with the speed |
| `idrak shell` | An interactive prompt for idrak commands, with history and completion |

The templates live in `Templates/KIND/` as `*.template` files, embedded in the tool; `__NAME__`, `__NAMESPACE__`,
`__FORMAT__` and `__REF(Package)__` are filled in by `idrak new`.
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
