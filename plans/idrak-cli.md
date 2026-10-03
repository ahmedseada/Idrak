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

### Short forms (as Angular's CLI has them)

Every option has its long form; the common ones also have a one-letter short form, and the most used commands have a
short alias (as `ng g c` for `ng generate component`). Help lists both (`-d, --device NAME`), and both work everywhere.

Common options:

| Short | Long |
|---|---|
| `-h` | `--help` |
| `-d` | `--device NAME` |
| `-j` | `--json` |
| `-q` | `--quiet` |
| `-v` | `--verbose` |
| `-P` | `--plugin PATH` |
| `-C` | `--config FILE` |
| (none) | `--cache DIR` (rarely typed) |

`idrak --version` (and `idrak -V`) print the same as `idrak version`.

Command aliases:

| Alias | Command |
|---|---|
| `idrak c` | `chat` |
| `idrak r` | `run` |
| `idrak s` | `serve` |
| `idrak b` | `bench` |
| `idrak ls` | `list` |
| `idrak dev` | `devices` |
| `idrak doc` | `doctor` |
| `idrak sg` | `suggest` |
| `idrak ps` | `server ps` (models loaded in a running server) |
| `idrak i` | `inspect` |
| `idrak x` | `explain` |

Short forms of commands' own options, kept the same across commands where the meaning is the same:

| Short | Long | Commands |
|---|---|---|
| `-m` | `--model NAME` | where a model is an option rather than the first argument |
| `-s` | `--system TEXT` | chat, run |
| `-t` | `--target COL` | suggest, data |
| `-o` | `--out PATH` | suggest, convert, quantize, report, embed |
| `-p` | `--port N` | serve, ui |
| `-H` | `--host NAME` | serve, ui |
| `-f` | `--force` | pull, rm, convert |
| `-n` | `--search N` / `--repeat N` | suggest / bench |
| `-y` | `--yes` (no confirmation) | rm, cache clear, data dedupe |
| `-e` | `--explain` | suggest, doctor |
| `-b` | `--base MODEL` | suggest, tune, distill |
| `-k` | `--kv FORMAT` | chat, run, serve, bench |
| `-w` | `--weights FORMAT` (int8, int4, bf16 or any registered packed format) | chat, run, serve, bench, quantize |
| `-i` | `--input FILE` | run, batch, predict, embed |

Rules: short forms are case-sensitive (`-v` verbose, `-V` version); one letter per short form, not combined
(`-q -j`, not `-qj`), so a value never hides inside a group; `-d=vulkan:0` and `-d vulkan:0` both work, as for long
forms; a command may not reuse a common short form for another meaning, and a test checks that no two options of a
command share a short form. The skeleton on the `cli` branch knows only `-h` so far: the dispatcher gains a short form
for each option and an alias list for commands.

A config file (`~/.idrak/config.json`, or `IDRAK_CONFIG`) gives defaults for these (device, cache, plug-ins, model
aliases, a Hugging Face token); options on the command line win.

## Commands

Priority: 1 = first release, 2 = next, 3 = later. Every command below is in scope for the build; the priority orders
the work and decides what the first release must have. Short forms and aliases are listed above.

### Setup and health

| Command | What it does | Priority |
|---|---|---|
| `idrak doctor` | Checks the .NET runtime, every backend (CUDA driver, Vulkan loader and ICDs, HIP runtime and hipRTC), each device's limits, the caches, disk space and the environment; prints what is missing with how to fix it. `--android`: the Turnip ICD (`VK_ICD_FILENAMES`), `/dev/kgsl-3d0`, `DOTNET_GCHeapHardLimit`. `--explain`: why each check matters | 1 |
| `idrak devices` | Every device: backend, name, memory, compute units, subgroup size, matrix units, the measured kernel width, whether a plain run uses it (today's `--list-devices`) | 1 |
| `idrak version` (`--version`, `-V`) | Versions of the tool, the libraries, the runtime and the drivers | 1 |
| `idrak report` | One file (Markdown and JSON) with the machine, devices, drivers, the test list and benchmark results, for the README's tested-on tables; `--tests`, `--bench`, `--out FILE`, `--zip` | 1 |
| `idrak init` | Writes a config file by asking a few questions (default device, cache folder, model aliases), or with `--yes` from what `doctor` finds | 2 |
| `idrak env` | Every environment variable Idrak reads, its value, default and meaning (see "Environment variables"); `--all`, `--json`; also `idrak help env` | 1 |
| `idrak cache info` / `cache clear [models\|tuning\|kernels\|all]` | Sizes and paths of the caches; clears one or all (`--yes` skips the question) | 2 |
| `idrak config get/set/unset/list` | Reads and writes the config file; `--profile NAME` keeps several named profiles (e.g. phone, desktop) | 2 |
| `idrak plugins list` | Everything registered: packed and KV formats, model families, RoPE scalings, tool-call formats, checkpoint and dataset formats, graph ops and layer types, ONNX ops, devices; `--plugin` shows what an assembly adds | 2 |
| `idrak completion bash\|zsh\|fish\|pwsh` | Prints a shell completion script for commands, options and cached model names | 2 |
| `idrak update` | Tells whether a newer tool version exists and the command to install it (no self-replacement) | 3 |

### Run models

| Command | What it does | Priority |
|---|---|---|
| `idrak chat MODEL` | Interactive chat with a Hugging Face id, a folder, a GGUF file or an alias; streaming with tokens per second; `--weights`, `--kv`, `--system`, `--tools FILE.dll`, `--think`, `--temperature`, `--top-k`, `--top-p`, `--max-tokens`, `--seed`, `--history FILE` (save and resume) | 1 |
| Chat slash commands | Inside `chat`: `/help`, `/system TEXT`, `/reset`, `/save FILE`, `/load FILE`, `/stats` (speed, memory, context used), `/think on\|off`, `/tools`, `/set temperature 0.7`, `/copy` (last answer), `/retry`, `/exit` | 1 |
| `idrak run MODEL "prompt"` | One answer and exit; reads the prompt from standard input when piped (`cat file \| idrak run m "summarize"`); `--input FILE`; `--json` gives the text, token counts and speed | 1 |
| `idrak batch MODEL --input prompts.jsonl --out answers.jsonl` | Answers many prompts with batched generation, resumable after an interruption | 2 |
| `idrak compare MODEL_A MODEL_B "prompt"` | The same prompt on two models (or two settings of one), answers side by side with speed | 3 |
| `idrak complete MODEL "text"` | Raw completion without the chat template | 2 |
| `idrak embed MODEL --input FILE` | Embeddings of lines or files to JSON or `.npy` | 2 |
| `idrak tokenize MODEL "text"` | Tokens and ids; `--chat` renders the chat template first; `--count` prints only the number | 2 |
| `idrak template MODEL` | The chat template, the detected tool-call format and a rendered sample conversation with a tool call | 2 |
| `idrak agent MODEL` | The coding agent (read, search, edit, run commands in a folder), from the CodingAgent sample; `--yes` to allow commands without asking | 3 |
| `idrak tools list/test TOOLS.dll` | The tools an assembly registers and a call of each with sample arguments | 3 |

### Serve

| Command | What it does | Priority |
|---|---|---|
| `idrak serve MODEL [MODEL...]` | The chat API (streaming, tool calls) and the OpenAI-style API on one port; several models loaded on demand with an idle unload timeout; `--host`, `--port`, `--api-key`, `--cors ORIGIN`, `--max-concurrency N`, `--keep-alive DURATION` | 1 |
| `idrak server ps` (`idrak ps`) | Models loaded in a running server, their memory and last use | 2 |
| `idrak server stop` / `server load MODEL` / `server unload MODEL` | Controls a running server through its API | 3 |
| `idrak api PATH [JSON]` | Calls a running server's endpoint from the command line (for scripts and checks) | 3 |
| `idrak ui MODEL` | A small web chat page in the browser over the same server | 3 |
| `idrak mcp serve TOOLS.dll` | Serves an assembly's tools over MCP | 3 |

### Models

| Command | What it does | Priority |
|---|---|---|
| `idrak pull MODEL` | Downloads a Hugging Face model or a GGUF file into the cache, with progress and resume | 1 |
| `idrak list` (`ls`) | Cached models with sizes, formats and last use | 1 |
| `idrak rm MODEL` | Removes a cached model (`--yes`) | 1 |
| `idrak show MODEL` | Family, parameters, layers, context, vocabulary, RoPE scaling, windows, chat template and tool-call format, files, license | 2 |
| `idrak search QUERY` | Searches the Hugging Face hub for models the library can load (families it knows, file formats it reads) | 2 |
| `idrak alias set NAME MODEL [--weights F --kv F]` / `alias list` / `alias rm` | Short names for models and their settings, stored in the config | 2 |
| `idrak memory MODEL` | Memory needed per weight format and KV format at a context length, against each device's memory: what fits where | 2 |
| `idrak quantize MODEL --weights int8\|int4\|bf16\|NAME --out DIR` | Packs the weights (any registered packed format), reports the size and a quick quality check (perplexity on a short text) | 2 |
| `idrak merge MODEL ADAPTER --out DIR` | Merges a LoRA or DoRA adapter into the base weights | 2 |
| `idrak inspect FILE` (`i`) | Tensor names, shapes, types and metadata of a GGUF, safetensors or `.ikm` file | 2 |
| `idrak verify MODEL` | Checks a cached model's files (sizes, hashes where the hub gives them, readable tensors) | 3 |
| `idrak convert IN OUT` | Hugging Face folder, GGUF and `.ikm` conversions | 3 |
| `idrak diff A B` | Which tensors differ between two checkpoints, and by how much | 3 |

### Train and fine-tune

| Command | What it does | Priority |
|---|---|---|
| `idrak tune ...` | Today's `idrak-tune` as a subcommand: LoRA, QLoRA, DoRA, DPO/ORPO/SimPO, optimizers and schedules | 2 |
| `idrak tune init` | Writes a commented `tune.json` for a model and dataset (from `suggest` when available) | 2 |
| `idrak train SPEC.json --data FILE` | Trains a network from a builder JSON and a CSV, image folder or (later) a dataset loader; writes a model package (`.ikm`) | 2 |
| `idrak resume RUN` | Continues a run from its checkpoint | 3 |
| `idrak runs list/show/compare` | Training runs from their JSON Lines logs: loss curves (text plot), best epoch, settings, time | 3 |
| `idrak predict MODEL.ikm --input FILE` | Runs a trained package on new rows (CSV or JSON Lines) and writes predictions | 2 |
| `idrak package --model DIR --out MODEL.ikm` | Bundles a network, its scalers and tokenizer into one model package | 3 |
| `idrak distill --teacher A --student B --data FILE` | The teacher pattern (plans/plug-in.md, "Noted for later"), when the library has it | 3 |

### Data

| Command | What it does | Priority |
|---|---|---|
| `idrak data ...` | Today's `idrak-data` as a subcommand | 2 |
| `idrak data preview FILE` | First rows, columns and types (Parquet, JSON Lines, CSV) | 2 |
| `idrak data validate FILE --as chat\|preference\|table` | Checks rows against the shape a command expects, with the first bad rows shown | 2 |
| `idrak data stats FILE --model MODEL` | Token counts per row with a model's tokenizer, length histogram, how many rows exceed a context length | 2 |
| `idrak data convert IN OUT` | Between CSV, JSON Lines and Parquet, and from common chat layouts to the chat rows the tools read | 3 |
| `idrak data dedupe FILE` / `data split FILE` | Removes duplicates (exact and near); train, validation and test split with a seed | 3 |
| `idrak data sample FILE -n N` | A random or stratified sample of rows | 3 |
| `idrak data mix RECIPE.json` | Assembles a training set from several sources with weights (the dataset recipes) | 3 |

### Retrieval

| Command | What it does | Priority |
|---|---|---|
| `idrak rag index DIR --out INDEX` | Chunks, embeds and builds the hybrid index (BM25 and vectors) | 3 |
| `idrak rag ask "question" --index INDEX --model MODEL` | Answers with cited passages | 3 |
| `idrak rag search "query" --index INDEX` | Top passages without generation, with scores | 3 |
| `idrak rag eval --index INDEX --questions FILE` | Retrieval quality (hit rate, MRR) on question/passage pairs | 3 |

### Measure

| Command | What it does | Priority |
|---|---|---|
| `idrak bench [MODEL]` (`b`) | Tokens per second (prompt and generation), GFLOP/s, memory on the chosen device; `--save NAME`, `--compare NAME`; without a model, the kernel benchmarks (`--kernels matmul,gemv,attention`) | 1 |
| `idrak eval MODEL SET.jsonl` | Answer metrics (accuracy, exact match; more as plug-in gap 16 lands) | 2 |
| `idrak perplexity MODEL FILE` | Perplexity of a text, to compare weight formats and fine-tunes | 2 |
| `idrak profile MODEL` | Time per layer and kernel of one decoding step | 3 |
| `idrak check MODEL --reference FILE` | Compares ids, template and logits with a transformers reference (the Chat sample's check) | 3 |
| `idrak tuning show` | The kernel choices measured on this device (from the tuning caches), and `--reset` | 3 |

### Design a model

| Command | What it does | Priority |
|---|---|---|
| `idrak suggest DATA` (`sg`) | Reads the data, picks the task and writes a network and training setup: `network.json` (the builder's JSON, so `Network.FromJson` and `idrak train` read it), `train.json` and `prep.json`; `--search N` tries N candidates briefly on the chosen device and keeps the best; `--explain` gives the rule behind each choice | 2 |
| `idrak explain network.json` (`x`) | Layers, output shapes, parameters, FLOPs per sample and memory for training and inference on the chosen device | 2 |
| `idrak viz network.json` | The network as a text diagram, or Mermaid / SVG (`--format`) | 3 |

How `suggest` decides, in three steps:

1. Read the data: file type (CSV, JSON Lines, Parquet, an image folder, chat rows), columns and value types, missing
   values, row count, duplicates, class balance, image sizes and channels, text lengths in words and, with a model,
   in tokens.
2. Pick the task: a numeric target is regression, a target with few distinct values is classification, a folder of
   class folders is image classification, text plus a label is text classification, chat rows are a chat fine-tune,
   chosen/rejected pairs are preference training (DPO). `--task` overrides the guess.
3. Propose a setup from rules sized by the data (rows, features, classes, sequence length, memory of the chosen
   device), then optionally measure: `--search N` trains N variants (width, depth, dropout, learning rate) for a few
   epochs on the device and keeps the best on held-out data.

Options: `--target COL` (`-t`), `--text COL`, `--task regression|classify|image|sequence|chat|preference`,
`--budget small|medium|large` or `--max-params N`, `--search N` (`-n`), `--explain` (`-e`), `--base MODEL` (`-b`),
`--out DIR` (`-o`), `--assist MODEL`.

Rules stay deterministic and explainable; the search is opt-in. Limits are reported rather than hidden (few rows:
strong regularization and a warning; unbalanced classes: class weights or resampling). `--assist MODEL` may let a
local chat model explain the choices in plain words; it never changes them.

### Developers

| Command | What it does | Priority |
|---|---|---|
| `idrak new console\|webapi\|rag\|plugin NAME` | Project templates; the plug-in starter registers a format and has a test that runs it without internal access | 2 |
| `idrak test [--filter TEXT] [--device NAME]` | Runs the library's test runner from a source checkout (the commands the README and installation guides give) | 3 |
| `idrak onnx import/export/check` | ONNX conversions with an output check against ONNX Runtime when installed | 3 |
| `idrak kernels dump [ptx\|spirv\|hip]` | The generated kernels, for debugging | 3 |
| `idrak trace COMMAND ...` | Runs a command with telemetry printed live (layers, batches, kernels) or written to JSON Lines | 3 |

## Helpers every command shares

- Progress: a progress line for downloads, loading, training and long benchmarks (rate, ETA); none when output is
  not a terminal or with `--quiet`.
- Colour only on a terminal, never with `NO_COLOR` set or `--json`.
- `--dry-run` on commands that delete, overwrite or download (rm, cache clear, pull, convert, quantize, merge): says
  what would happen.
- `--yes` (`-y`) skips confirmation questions; without a terminal a question is an error that names `--yes`.
- `--log FILE` writes the verbose output to a file while the terminal stays normal.
- `@file` reads arguments from a file, one per line (long option lists in scripts).
- Model arguments accept a Hugging Face id, a folder, a GGUF file, a `.ikm` package or an alias, everywhere.
- Ctrl+C stops cleanly: generation stops after the current token, downloads keep their partial file to resume,
  servers drain open requests.
- Errors say what to do next (the exact command or option), and `IDRAK_TRACE=1` adds the stack.

## Environment variables

`idrak env` lists every environment variable the libraries and the tool read, with its current value, its default
and what it does; `--all` includes unset ones, `--json` gives them as data, and `idrak help env` (also
`idrak help environment`) prints the same reference without values. Each command's help ends with the variables that
affect it. The list is generated from one table in the tool (the source of truth), and a test fails when a variable
read by the libraries (a string literal passed to `Environment.GetEnvironmentVariable`) is missing from it.

| Group | Variables |
|---|---|
| Devices and backends | `IDRAK_DISABLE_CUDA`, `IDRAK_DISABLE_VULKAN`, `IDRAK_DISABLE_HIP`, `IDRAK_VULKAN_DEFAULT`, `IDRAK_HIP_DEFAULT`, `IDRAK_CUDA_DEBUG`, `IDRAK_POWER_SOURCE` |
| Tuning and caches | `IDRAK_CACHE`, `IDRAK_AUTOTUNE`, `IDRAK_TUNING_CACHE`, `IDRAK_TUNE_LOG`, `IDRAK_CPU_TUNING_FILE`, `IDRAK_VULKAN_TUNING_CACHE`, `IDRAK_HIP_KERNEL_CACHE` |
| Precision and memory | `IDRAK_MATMUL`, `IDRAK_FP8_DELAYED`, `IDRAK_OFFLOAD` |
| CPU | `IDRAK_CPU_KCHUNK`, `IDRAK_CPU_PARALLEL_ELEMENTS`, `IDRAK_CPU_PARALLEL_FLOPS` |
| Vulkan | `IDRAK_VULKAN_KERNELS`, `IDRAK_VULKAN_MATRIX`, `IDRAK_VULKAN_WIDTH`, `IDRAK_VULKAN_WIDTH_PROBE`, `IDRAK_VULKAN_SUBGROUPS`, `IDRAK_VULKAN_SUBGROUP_SIZE`, `IDRAK_VULKAN_STORAGE`, `IDRAK_VULKAN_STAGING`, `IDRAK_VULKAN_STAGING_BYTES`, `IDRAK_VULKAN_PAGE_BYTES`, `IDRAK_VULKAN_MAX_STORAGE_BYTES`, `IDRAK_VULKAN_MAX_ALLOCATIONS`, `IDRAK_VULKAN_PUSH_DESCRIPTORS`, `IDRAK_VULKAN_BATCH_COMMANDS`, `IDRAK_VULKAN_IN_FLIGHT`; the Vulkan loader's `VK_ICD_FILENAMES` (which driver; needed for Turnip on a phone) and `VK_INSTANCE_LAYERS` (validation layers) |
| HIP | `IDRAK_HIP_KERNELS`, `HIP_PATH`, `ROCM_PATH` |
| Model and data sources | `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, `HF_TOKEN_PATH`, `HF_HOME`, `HF_HUB_CACHE`, `HF_ENDPOINT`, `GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_API_URL`, `KAGGLE_USERNAME`, `KAGGLE_KEY`, `KAGGLE_CONFIG_DIR`, `ZENODO_TOKEN`, and the local model store's folder variable the GGUF source reads |
| The tool | `IDRAK_CONFIG`, `IDRAK_TRACE`, `NO_COLOR` |
| Tests and diagnostics | `IDRAK_DEVICES`, `IDRAK_FILTER`, `IDRAK_TIMEOUT`, `IDRAK_SPIRV_VAL` |
| .NET | `DOTNET_GCHeapHardLimit` (needed on Android), `DOTNET_CLI_TELEMETRY_OPTOUT`, `DOTNET_NOLOGO` |

Tokens and keys are shown as set or not set, never their values.

## Examples

```
idrak doctor                                   # what works on this machine and what to fix
idrak doc --android                            # the phone checks (Turnip, KGSL, heap limit)
idrak dev -j                                   # devices as JSON
idrak pull Qwen/Qwen3-0.6B
idrak alias set qwen Qwen/Qwen3-0.6B -w int8 -k int8
idrak c qwen -d vulkan:0 -s "Answer briefly."  # chat on the Vulkan GPU
cat notes.txt | idrak r qwen "Summarize in three bullets"
idrak batch qwen -i prompts.jsonl -o answers.jsonl
idrak s qwen phi -p 8080 --api-key $KEY        # serve two models on one port
idrak memory Qwen/Qwen3-8B --kv int8           # what fits on which device
idrak quantize Qwen/Qwen3-0.6B -w int4 -o ./qwen-int4
idrak sg houses.csv -t SalePrice -e            # design a network for a CSV, with the reasons
idrak sg ./shapes -n 6 -d vulkan:0             # try 6 CNN variants on the GPU, keep the best
idrak sg reviews.jsonl -t label --text review -b Qwen/Qwen3-0.6B
idrak sg chats.jsonl -b Qwen/Qwen3-0.6B -o ./run   # LoRA setup sized to the GPU
idrak x ./run/network.json                     # parameters, FLOPs, memory
idrak train ./run/network.json --data houses.csv -o houses.ikm
idrak predict houses.ikm -i new-houses.csv
idrak tune --config ./run/tune.json
idrak data validate chats.jsonl --as chat
idrak data stats chats.jsonl -m qwen
idrak rag index ./docs -o docs.idx && idrak rag ask "How do I reset?" --index docs.idx -m qwen
idrak b qwen --save laptop && idrak b qwen --compare laptop
idrak report --tests --bench --zip -o phone-report.zip
idrak new plugin MyFormat
```

What `idrak suggest houses.csv --target SalePrice` prints (illustrative):

```
Data      1,460 rows · 79 columns (36 numeric, 43 categories) · 6.6% missing
Task      regression (target is a number; 663 distinct values)
Prep      standard-scale numbers · one-hot 43 categories (→ 288 features) · fill missing with medians
          log-transform the target (skew 1.9) · 0 duplicates · 80/20 split, seed 1
Network   MLP 288 → 128 → 64 → 1, ReLU, dropout 0.1 (≈ 45k parameters; small for 1,460 rows)
Training  AdamW 1e-3, weight decay 1e-4 · batch 64 · up to 300 epochs, early stop after 20 · mean squared error
Wrote     network.json · train.json · prep.json
Next      idrak train network.json --data houses.csv --config train.json
```

What `idrak sg ./shapes --search 6 -d vulkan:0` prints (illustrative):

```
Data      3,000 images in 5 class folders · 28×28 grayscale · balanced (±4%)
Task      image classification, 5 classes
Prep      scale to [0, 1] · augment: flips, ±10° rotations, ±2 px shifts (small set)
Network   [conv 16 → batch norm → ReLU → pool] × 2 → global average pool → linear 5 (≈ 6k parameters)
Search    6 candidates × 5 epochs on vulkan:0 (≈ 40 s): width 16/32 · depth 2/3 · dropout 0/0.2
          best: width 32, depth 2, dropout 0.2 → validation accuracy 97.1% (first guess 95.4%)
Training  AdamW 3e-3 · cosine with 2 warm-up epochs · batch 128 · 40 epochs · cross-entropy
```

What `idrak sg chats.jsonl -b Qwen/Qwen3-0.6B -d cuda:0` prints (illustrative):

```
Data      4,200 conversations (chat rows) · 312 tokens median, 1,980 longest
Task      chat fine-tune (loss on the assistant's tokens only)
Memory    cuda:0 has 16 GB → bfloat16 base with LoRA fits; QLoRA not needed
Setup     LoRA rank 16, alpha 32 on q/k/v/o and gate/up/down · packing at 2,048 tokens
          AdamW 2e-4 · cosine with 3% warm-up · 2 epochs · ≈ 23 min at the measured 17k tokens/s
Wrote     tune.json → idrak tune --config tune.json
```

## Structure

- Project `src/Idrak.Cli`, assembly `Idrak.Cli` (an assembly named `idrak` would clash with the Idrak library, as
  assembly names ignore case), tool command `idrak`, package `Idrak.Cli`; references Idrak, Idrak.LanguageModels,
  Idrak.Datasets, Idrak.AspNetCore (serve) and Idrak.Onnx (onnx) as needed. The skeleton is on the `cli` branch.
- `Program.cs` calls `CommandLine.Run`, which dispatches through `CommandTable`; each group of commands lives in its
  own file under `Commands/` (`HealthCommands`, `RunCommands`, `ServeCommands`, `ModelCommands`, `TrainCommands`,
  `DataCommands`, `RetrievalCommands`, `MeasureCommands`, `DeveloperCommands`, `DesignCommands`), so groups are built
  independently.
- A command is a class deriving from `Command`: name (several words for subcommands), aliases, one-line summary, usage
  text, its value options and flags with their short forms, and `Run(CommandContext)` returning the exit code.
  `CommandContext` holds the parsed options, the positional arguments, the config, the device, and output helpers
  (`Write`, `Detail`, `Table`, `WriteJson`, `Error`).
- Shared helpers live in `src/Idrak.Cli/Shared/` (model resolution and loading with `--weights`/`--kv`, progress,
  terminal detection, confirmation questions); a group adds helpers in new files there rather than editing another
  group's.
- `idrak-tune` and `idrak-data` keep working; their code moves behind `idrak tune` and `idrak data`, and the old tool
  names become thin forwarders for a release.
- Tests: `tests/Idrak.Tests` gains CLI groups that run commands in-process through `CommandLine.Run` with captured
  output (no network: models from the test fixtures in `tests/Idrak.Tests/data`), checking exit codes, text and JSON
  output; a test checks every command's help, that no two options of a command share a short form, and that no
  command, option or output names a provider.

## Who builds what

The build runs as several agents in parallel on branch `idrak-cli` (from `architecture`), one per group, each in its
own copy of the repository; the parent session merges them. Each agent owns its group file under `Commands/`, may add
files under `Shared/` and tests of its own, and touches nothing else except the README rows and CHANGELOG line of its
commands.

| Agent | Owns | Commands |
|---|---|---|
| 1 Health and foundation | `HealthCommands`, the dispatcher's short forms, aliases, `--version`, completion, `@file`, `--log`, progress and terminal helpers | doctor, devices, version, report, init, env, cache, config, plugins, completion, update |
| 2 Run | `RunCommands`, `Shared/` model loading | chat (with slash commands), run, batch, compare, complete, embed, tokenize, template, agent, tools |
| 3 Serve | `ServeCommands` | serve, server ps/stop/load/unload, api, ui, mcp serve |
| 4 Models | `ModelCommands` | pull, list, rm, show, search, alias, memory, quantize, merge, inspect, verify, convert, diff |
| 5 Train and data | `TrainCommands`, `DataCommands` | tune (+ init), train, resume, runs, predict, package, distill (stub until the library has it), data and its subcommands |
| 6 Retrieval and measure | `RetrievalCommands`, `MeasureCommands` | rag index/ask/search/eval, bench, eval, perplexity, profile, check, tuning show |
| 7 Design | `DesignCommands` | suggest (rules and `--search`), explain, viz |
| 8 Developers | `DeveloperCommands` | new (templates), test, onnx, kernels dump, trace |

Improvements found while building (a missing option, a helper several commands need, a command that would make the
tool more useful) are added to this plan and built, not only noted.

Where a command needs something the library lacks (for example `distill`, structured output, the OpenAI-style API),
the agent builds the command around what exists, marks the missing part in its help and in this plan, and does not
change the library beyond small, tested additions.

## Done when

First release: `doctor`, `devices`, `version`, `report`, `chat`, `run`, `serve`, `pull`, `list`, `rm` and `bench` work
with every common option and short form on CPU and Vulkan (tested here), with JSON output covered by tests;
`--plugin` loads an assembly built from `tests/Idrak.PluginTests`; the README and `installation/` describe
`dotnet tool install` and the commands.

Complete: every command above exists, is tested in-process, has help with examples, and is listed in the README;
priorities 2 and 3 may leave a documented gap where the library lacks a feature.
