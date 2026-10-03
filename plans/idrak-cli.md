# idrak: one command-line tool

Today Idrak has two separate tools (`idrak-tune`, `idrak-data`), and the most common actions (chat with a model,
serve it, check the devices, benchmark) need `dotnet run` on a sample or the test runner. This plan puts everything
behind one `dotnet tool` named `idrak`, with subcommands. Nothing here is built yet.

Rules that apply to every command:

- No dependencies beyond the Idrak packages (the argument parsing is written in the tool, as in `idrak-tune`); the
  tool stays AOT-compatible where the libraries it uses are.
- No provider names in commands, options or output: the serving API is "the chat API" (its routes are the ones common
  local-model clients use) and "the OpenAI-style API" is named by its wire format only (see plans/plug-in.md, "Noted
  for later").
- Card-agnostic: nothing in the tool picks behaviour by a card's or vendor's name; it reports what devices report.
- Every command accepts the common options below, prints human-readable text by default and JSON with `--json`, and
  exits with 0 on success, 1 on a failed operation, 2 on a usage error.

## Common options

| Option | Meaning |
|---|---|
| `--device NAME` | `cpu`, `cuda:0`, `vulkan:1`, `hip:0` (`Device.Parse`); default `Device.Default` |
| `--plugin PATH` | Load an assembly before the command runs (repeatable); it registers its formats, families, tool-call formats and so on from a module initializer or an `IIdrakPlugin.Register()` (plug-in gap 26) |
| `--json` | Machine-readable output |
| `--quiet` / `--verbose` | Less or more output |
| `--cache DIR` | The cache folder (models, tuning, kernels); default `IDRAK_CACHE` or `~/.cache/idrak` |
| `--help`, `-h` | Help for the command; `idrak help COMMAND` |

A config file (`~/.idrak/config.json`, or `IDRAK_CONFIG`) gives defaults for these (device, cache, plug-ins, model
aliases, a Hugging Face token); options on the command line win.

## Commands

Priority: 1 = first release, 2 = next, 3 = later.

### Setup and health

| Command | What it does | Priority |
|---|---|---|
| `idrak doctor` | Checks the .NET runtime, every backend (CUDA driver, Vulkan loader and ICDs, HIP runtime and hipRTC), each device's limits, the caches, and the environment; prints what is missing with how to fix it. `--android`: the Turnip ICD (`VK_ICD_FILENAMES`), `/dev/kgsl-3d0`, `DOTNET_GCHeapHardLimit` | 1 |
| `idrak devices` | Every device: backend, name, memory, compute units, subgroup size, matrix units, the measured kernel width, whether a plain run uses it (today's `--list-devices`, as a tool) | 1 |
| `idrak cache info` / `clear [models\|tuning\|kernels]` | Sizes and paths of the caches; clears one or all | 2 |
| `idrak config get/set/list` | Reads and writes the config file | 2 |
| `idrak report` | One file with the machine, devices, drivers, the test list and benchmark results (what is collected by hand into zip files today), for the README's tested-on tables | 1 |
| `idrak plugins list` | Everything registered: packed formats, KV formats, model families, RoPE scalings, tool-call formats, checkpoint and dataset formats, graph ops, devices | 2 |
| `idrak version` | Versions of the tool, the libraries, the runtime and the drivers | 1 |

### Run models

| Command | What it does | Priority |
|---|---|---|
| `idrak chat MODEL` | Interactive chat with a Hugging Face id, a folder or a GGUF file (`--int8`, `--int4`, `--kv FORMAT`, `--system TEXT`, `--tools`, `--think`); streaming, with tokens per second | 1 |
| `idrak run MODEL "prompt"` | One answer and exit; reads the prompt from standard input when piped (`cat file \| idrak run m "summarize"`) | 1 |
| `idrak complete MODEL "text"` | Raw completion without the chat template | 2 |
| `idrak embed MODEL FILE` | Embeddings of lines or files to JSON or `.npy` | 2 |
| `idrak tokenize MODEL "text"` | Tokens and ids; `--chat` renders the chat template first | 2 |
| `idrak template MODEL` | The chat template, the detected tool-call format and a rendered sample conversation | 2 |

### Serve

| Command | What it does | Priority |
|---|---|---|
| `idrak serve MODEL [MODEL...]` | The chat API (streaming, tool calls) and the OpenAI-style API on one port; several models loaded on demand with an idle unload timeout; `--host`, `--port`, `--api-key` | 1 |
| `idrak ui MODEL` | A small web chat page in the browser over the same server | 3 |
| `idrak mcp serve PLUGIN.dll` | Serves an assembly's tools over MCP | 3 |

### Models

| Command | What it does | Priority |
|---|---|---|
| `idrak pull MODEL` | Downloads a Hugging Face model or a GGUF file into the cache | 1 |
| `idrak list` | Cached models with sizes and formats | 1 |
| `idrak show MODEL` | Family, parameters, layers, context, vocabulary, RoPE scaling, windows, files | 2 |
| `idrak rm MODEL` | Removes a cached model | 1 |
| `idrak convert IN OUT` | Hugging Face folder, GGUF and `.ikm` conversions | 3 |
| `idrak quantize MODEL --int8\|--int4\|--bf16\|--format NAME` | Packs the weights (any registered packed format) and reports the size and a quick quality check | 2 |
| `idrak merge MODEL ADAPTER` | Merges a LoRA or DoRA adapter into the base weights | 2 |
| `idrak inspect FILE` | Tensor names, shapes, types and metadata of a GGUF or safetensors file | 2 |
| `idrak diff A B` | Which tensors differ between two checkpoints, and by how much | 3 |

### Train and fine-tune

| Command | What it does | Priority |
|---|---|---|
| `idrak tune ...` | Today's `idrak-tune`, as a subcommand (LoRA, QLoRA, DPO/ORPO and DoRA as they land) | 2 |
| `idrak train SPEC.json --data FILE` | Trains a network from a builder JSON and a CSV or (later) a dataset loader | 3 |
| `idrak resume RUN` | Continues a run from its checkpoint | 3 |
| `idrak runs list/show/compare` | Training runs from their JSON Lines logs: loss curves, best epoch, settings | 3 |

### Data

| Command | What it does | Priority |
|---|---|---|
| `idrak data ...` | Today's `idrak-data`, as a subcommand | 2 |
| `idrak data preview FILE` | First rows, columns and types (Parquet, JSON Lines, CSV) | 2 |
| `idrak data dedupe/split` | Removes duplicates; train and test split | 3 |
| `idrak data stats FILE --model MODEL` | Token counts per row with a model's tokenizer, length histogram | 3 |

### Retrieval

| Command | What it does | Priority |
|---|---|---|
| `idrak rag index DIR --out INDEX` | Chunks, embeds and builds the hybrid index | 3 |
| `idrak rag ask "question" --index INDEX --model MODEL` | Answers with cited passages | 3 |
| `idrak rag search "query" --index INDEX` | Top passages without generation | 3 |

### Measure

| Command | What it does | Priority |
|---|---|---|
| `idrak bench [MODEL]` | Tokens per second (prompt and generation), GFLOP/s, memory on the chosen device; `--save NAME`, `--compare NAME` | 1 |
| `idrak eval MODEL SET.jsonl` | Answer metrics (accuracy, exact match; more as plug-in gap 16 lands) | 2 |
| `idrak profile MODEL` | Time per layer and kernel of one decoding step | 3 |
| `idrak check MODEL --reference FILE` | Compares ids, template and logits with a transformers reference (the Chat sample's check) | 3 |

### Developers

| Command | What it does | Priority |
|---|---|---|
| `idrak new console\|webapi\|rag\|plugin NAME` | Project templates, including a plug-in starter that registers a format and has a test | 2 |
| `idrak onnx import/export/check` | ONNX conversions with an output check | 3 |
| `idrak kernels dump [ptx\|spirv\|hip]` | The generated kernels, for debugging | 3 |

## Structure

- Project `src/Idrak.Cli`, assembly and tool command `idrak`, package `Idrak.Cli`; references Idrak,
  Idrak.LanguageModels, Idrak.Datasets, Idrak.AspNetCore (serve) and Idrak.Onnx (onnx) as needed.
- `Program.cs` dispatches on the first argument through `CommandTable`; each group of commands lives in its own file
  under `Commands/` and exposes its commands to the table, so groups are built independently.
- A command is a small class: name, one-line summary, usage text, and `Run(CommandContext)` returning the exit code.
  `CommandContext` holds the parsed options (common and the command's own), the positional arguments, the config,
  the device, and output helpers (`Write`, `Table`, `Json`, `Error`).
- `idrak-tune` and `idrak-data` keep working; their code moves behind `idrak tune` and `idrak data`, and the old tool
  names become thin forwarders for a release.
- Tests: `tests/Idrak.Tests` gains a CLI group that runs commands in-process (no network: models from the test
  fixtures) and checks exit codes and JSON output.

## Done when (first release)

`doctor`, `devices`, `version`, `report`, `chat`, `run`, `serve`, `pull`, `list`, `rm` and `bench` work with every
common option on CPU and Vulkan (tested here) with JSON output covered by tests; `--plugin` loads an assembly built
from `tests/Idrak.PluginTests`; the README and `installation/` describe `dotnet tool install` and the commands.
