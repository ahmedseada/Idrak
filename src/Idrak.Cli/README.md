# idrak: the command-line tool

`idrak` puts Idrak behind one command: check what works on a machine, chat with a language model or serve it, manage
the model cache, benchmark, fine-tune and train, prepare data, build retrieval indexes and design networks. It needs
nothing beyond the Idrak packages, reports what the devices report (nothing is chosen by a card's or vendor's name),
prints text by default and one JSON document with `--json`, and exits with 0 on success, 1 when the operation fails
and 2 on a usage error. The design and the full command list are in [plans/idrak-cli.md](../../plans/idrak-cli.md).

## Install

```bash
dotnet tool install -g Idrak.Cli          # the idrak command, from the package feed
idrak doctor                              # what works on this machine and what to fix
dotnet tool update -g Idrak.Cli           # later: the newest version (idrak update says whether there is one)
```

From a source checkout, without installing:

```bash
dotnet run -c Release --project src/Idrak.Cli -- help
```

Or pack the checkout and install that package (a local feed is a folder):

```bash
dotnet pack src/Idrak.Cli -c Release -o ./packages
dotnet tool install -g Idrak.Cli --add-source ./packages --prerelease
```

The tool targets .NET 10. On an Android phone (Termux with Ubuntu under proot, see
[installation/android-termux.md](../../installation/android-termux.md)), `idrak setup android` adds the settings the
phone needs and runs `idrak doctor --android`.

## First steps

```bash
idrak doctor                                   # the runtime, every backend, each device, the caches, disk, environment
idrak dev                                      # the devices: memory, compute units, lanes, kernel width, matrix units
idrak pull Qwen/Qwen3-0.6B                     # into the cache, with progress and resume
idrak alias set qwen Qwen/Qwen3-0.6B -w int8 -k int8
idrak c qwen -d vulkan:0 -s "Answer briefly."  # chat on the Vulkan GPU
cat notes.txt | idrak r qwen "Summarize in three bullets"
idrak s qwen -p 8080                           # the chat API and the OpenAI-style API on one port
idrak b qwen --save laptop                     # tokens per second, kept to compare with later
```

`idrak help` lists every command by group, `idrak help COMMAND` (or `idrak COMMAND --help`) shows one command's
arguments, options, examples, limits and the environment variables that affect it, and `idrak help topics` lists the
concept pages (devices, formats, models, precision, plugins, config, env, exit codes).

## How every command works

Model arguments are the same everywhere: a Hugging Face id (`owner/name`), a model folder, a `.gguf` file, a pulled
GGUF file (`owner/name:TAG`, `owner/name/FILE.gguf`), a `.ikm` package where the command reads one, or an alias from
the config. A model in the cache is used without asking the network (and its last use shows in `idrak list`); one
that is not is downloaded into the cache with a progress line, or, with `--offline`, is an error that names
`idrak pull`. `-w, --weights` picks the weight format (int8, int4, bf16 or any registered packed format) and `-k,
--kv` the KV cache format.

Common options (every command):

| Option | Meaning |
|---|---|
| `-d, --device NAME` | `cpu`, `cuda:0`, `vulkan:1`, `hip:0`; default: the config's device, else the default device |
| `-P, --plugin PATH` | Load an assembly that registers formats, families, tool-call formats, ... (repeatable) |
| `-j, --json` | One JSON document (also `--format json`) |
| `-q, --quiet` / `-v, --verbose` | Less or more output |
| `-C, --config FILE` | The config file (default `IDRAK_CONFIG` or `~/.idrak/config.json`) |
| `--cache DIR` | The cache folder: models, tuning, kernels, runs, saved benchmarks (default `IDRAK_CACHE` or `~/.cache/idrak`) |
| `-O, --output FILE` | Write the command's main output to a file |
| `--format text\|json\|csv\|md` | The output format of commands that print tables |
| `--offline` | Use only what is cached; a download is an error naming the missing file |
| `--timeout DURATION` | Give up after a time (`30s`, `5m`, `1h30m`): downloads, server calls, long runs |
| `--seed N` | One seed for sampling, shuffling and initialization |
| `--threads N` | CPU threads |
| `--color auto\|always\|never`, `--plain` | Colour control (`NO_COLOR` still wins); no Unicode or progress characters |
| `--log FILE` | Also write every line, verbose ones included, to a file |
| `@FILE` | Read arguments from a file, one per line |
| `--lang en\|ar` | The language of the messages (default: the config's `lang`, `IDRAK_LANG`, else English); see "Arabic messages" |
| `--lang-render auto\|visual\|visual-right\|logical` | How Arabic reaches the terminal (default auto) |
| `-h, --help` | Help; `idrak -V` and `idrak --version` print the versions |

Short forms are one letter, case-sensitive and not combined (`-q -j`, not `-qj`); the same letter means the same thing
in every command (`-m` a model, `-o` an output path, `-i` an input file, `-y` yes, `-f` force, `-n` a count, `-w` and
`-k` the formats, `-s` a system prompt, `-t` a target column, `-p` and `-H` a port and host, `-b` a base model, `-e`
explain). Commands that delete, overwrite or download ask first on a terminal, take `-y, --yes` to skip the question
(without a terminal a question is an error naming `--yes`) and `--dry-run` to say what would happen. Ctrl+C stops
cleanly: generation after the current token, training after the current batch with its checkpoints kept, downloads
with their partial files kept for resuming.

The config file holds defaults for these options (`device`, `cache`, `plugins`), model aliases and named profiles
(`idrak config set device cpu --profile phone`); options on the command line win. `idrak init` writes one.

## The commands

### Setup and health

What works on this machine, and the tool's own settings.

| Command | What it does |
|---|---|
| `idrak doctor` (`doc`) | What works on this machine and what to fix: runtime, backends, devices, caches, disk, environment |
| `idrak devices` (`dev`) | Every device: backend, name, memory, compute units, subgroup size, matrix units, kernel width |
| `idrak version` | Versions of the tool, the libraries, the .NET runtime and the device drivers |
| `idrak report` | A report (Markdown and JSON) of the machine, devices, drivers, tests and benchmarks |
| `idrak env` (`environment`) | Every environment variable Idrak reads: value, where it comes from, default and meaning |
| `idrak env set` | Save environment variables for every idrak run, in any terminal (asks for them, with suggested values) |
| `idrak env unset` | Remove saved environment variables |
| `idrak env reset` | Remove every saved environment variable: all back to their defaults (asks first) |
| `idrak init` | Writes the config file: default device, cache folder, model aliases |
| `idrak cache info` | Sizes and paths of the caches (models, tuning, kernels, downloads) |
| `idrak cache clear` | Clears one part of the cache (models, tuning, kernels) or all of it |
| `idrak config get` | Prints a config value (dotted keys, e.g. aliases.qwen.model) |
| `idrak config set` | Sets a config value (JSON values such as ["a.dll"] or 4 are kept as JSON) |
| `idrak config unset` | Removes a config value |
| `idrak config list` | Every config value (or a profile's), and the file's path |
| `idrak plugins list` | Everything registered: formats, families, RoPE scalings, tool-call formats, ops, devices |
| `idrak formats` | Every registered weight, KV cache, checkpoint, dataset and tool-call format |
| `idrak completion` | Prints a shell completion script (bash, zsh, fish, pwsh) |
| `idrak update` | Whether a newer version of the tool exists, and the command to install it |
| `idrak login` | Stores a token for the Hugging Face hub, GitHub or Kaggle where the library reads it |
| `idrak logout` | Removes a stored token (Hugging Face hub, GitHub or Kaggle) |
| `idrak setup android` | The Android (Termux) setup steps that are safe to automate, then the phone checks |

```bash
idrak doctor                     # failed checks print how to fix them; --fix prints (and with -y runs) the safe fixes
idrak doc --android              # the phone checks: the Turnip driver file, /dev/kgsl-3d0, the .NET heap limit
idrak dev -j                     # every device as JSON
idrak report --tests --bench --zip -o phone-report.zip
idrak report --readme            # the README's tested-on rows for this machine, ready to paste
idrak env --all                  # every environment variable Idrak reads; idrak help env without values
eval "$(idrak completion bash)"  # completion for commands, options, devices, formats and cached models
```

### Run models

Chat, one-off answers, batches and the tools a model may call. Generation uses a 4096-token window unless
`--context` says otherwise; `--temperature 0` is greedy decoding.

| Command | What it does |
|---|---|
| `idrak chat` (`c`) | Interactive chat with a model, streamed with tokens per second |
| `idrak run` (`r`) | One answer to a prompt (argument, --input file or piped), then exit |
| `idrak batch` | Answers many prompts (JSON Lines) with batched generation, resumable |
| `idrak compare` | The same prompt on two models (or two settings of one), answers side by side with speed |
| `idrak complete` | Raw completion of a text, without the chat template |
| `idrak embed` | Embeddings of lines or files to JSON or .npy |
| `idrak tokenize` | Tokens and ids of a text (--chat renders the chat template first, --count only counts) |
| `idrak template` | The chat template, its tool-call format and a rendered sample conversation with a tool call |
| `idrak agent` | The coding agent: reads, searches, edits files and runs commands in a folder |
| `idrak tools list` | The tools an assembly registers: names, descriptions and parameters |
| `idrak tools test` | Calls each tool of an assembly with sample arguments and shows the results |

```bash
idrak chat Qwen/Qwen3-0.6B -s "Answer briefly." --history talk.json   # /help lists the slash commands
cat notes.txt | idrak run qwen "Summarize in three bullets"
idrak run qwen "Extract the name and age: Sara is 31." --schema person.json --json
idrak batch qwen -i prompts.jsonl -o answers.jsonl                   # resumable after an interruption
idrak compare qwen qwen --weights-b int4 "Explain recursion" --seed 1
idrak tokenize qwen --chat "Hello" --count
idrak agent qwen "Add a test for Parse" --workspace ./src
```

### Serve

Several models on one port over the chat API (the routes common local-model clients use) and the OpenAI-style
API, loaded on first use and unloaded when idle. The port is `-p/--port`, else `IDRAK_PORT`, else the config's
`serve.port`, else 7317, Idrak's own (so it does not collide with another local model server). Each running server
records itself in the cache folder, so `ps`, `api`, `ping` and `server ...` find it without `--port` (they use
`--port` or `IDRAK_PORT` when given, then the server started last, then `serve.port` and 7317).

Clients: give them `http://127.0.0.1:7317` as the chat API's address and `http://127.0.0.1:7317/v1` as the
OpenAI-style API's base URL (any API key unless the server needs one). Clients that expect the chat API on port 11434
and cannot be told another work unchanged with `idrak serve MODEL -p 11434` (or `idrak config set serve.port 11434`),
as long as no other server uses that port. Checked with an API client's AI request (OpenAI-compatible, base URL
`http://127.0.0.1:7317/v1`, model `Qwen/Qwen3-0.6B`): English and Arabic answers in 2.5 to 3 s each. A
client that appends its own route to the address it was given (`.../api/chat/chat/completions`) is answered too, and
any 404 prints the right addresses in the server's window.

| Command | What it does |
|---|---|
| `idrak serve` (`s`) | Serve models over the chat API and the OpenAI-style API on one port |
| `idrak ui` | Serve models and open a small web chat page in the browser |
| `idrak server ps` (`ps`) | Models of a running server: loaded or not, memory, context, last use |
| `idrak server stop` | Stop a running server; open requests finish first |
| `idrak server load` | Load a model in a running server now, instead of on its first request |
| `idrak server unload` | Unload a model from a running server, freeing its memory |
| `idrak server keys add` | Make an API key for serve (printed once; the config keeps only its hash) |
| `idrak server keys list` | The API keys serve accepts (names and dates; the keys are not stored) |
| `idrak server keys rm` | Remove an API key (servers started afterwards no longer accept it) |
| `idrak api` | Call an endpoint of a running server (for scripts and checks) |
| `idrak ping` | Check a running server (Idrak or any compatible one): reachable, APIs, models, latency |
| `idrak mcp serve` | Serve the tools of an assembly over MCP (standard input/output) |

```bash
idrak s qwen phi -p 8080 --api-key $KEY        # two models on one port
idrak ui qwen                                  # the same server, with its web chat page opened
idrak ps
idrak api /v1/chat/completions '{"model":"qwen","messages":[{"role":"user","content":"Hi"}]}'
idrak ping http://192.168.1.20:8080 --timeout 5s
```

### Models

The model cache: what `pull` downloads is what loading reads (Hugging Face models under
`downloads/huggingface/models`, GGUF files beside them or under `downloads/urls`, prepared GGUF folders under `gguf`).

| Command | What it does |
|---|---|
| `idrak pull` | Download a Hugging Face model or a GGUF file into the cache, with progress and resume |
| `idrak list` (`ls`) | Cached models with sizes, formats and last use |
| `idrak rm` | Remove a cached model |
| `idrak show` | Family, parameters, layers, context, vocabulary, RoPE, windows, chat template, files and license of a model |
| `idrak search` | Search the Hugging Face hub for models Idrak can load |
| `idrak alias set` | Give a model (and its weight and KV formats) a short name, kept in the config |
| `idrak alias list` (`alias`, `alias ls`) | The model aliases in the config |
| `idrak alias rm` | Remove a model alias from the config |
| `idrak memory` | Memory per weight and KV format at a context length, against each device's memory |
| `idrak quantize` | Pack a model's weights (int8, int4, bf16 or any registered format): size and a quick perplexity check |
| `idrak merge` | Merge a LoRA or DoRA adapter into the base weights |
| `idrak inspect` (`i`) | Tensor names, shapes, types and metadata of a GGUF, safetensors or .ikm file |
| `idrak verify` | Check a cached model's files: sizes, hashes where the hub gives them, readable tensors |
| `idrak convert` | Convert between GGUF and Hugging Face folders (safetensors bf16, f16, f32) |
| `idrak diff` | Which tensors differ between two checkpoints, and by how much |
| `idrak families` | Supported model families and what each supports (windows, soft-capping, RoPE scalings, experts, GGUF) |

```bash
idrak pull Qwen/Qwen3-0.6B-GGUF:Q8_0           # one GGUF file of a repository
idrak ls --sort size
idrak show Qwen/Qwen3-0.6B
idrak memory Qwen/Qwen3-8B --kv int8           # what fits on which device
idrak quantize Qwen/Qwen3-0.6B -w int4 -o ./qwen-int4
idrak merge Qwen/Qwen3-0.6B ./run/adapter -o ./qwen-tuned
idrak rm Qwen/Qwen3-0.6B --dry-run
```

### Train

Fine-tuning language models (`tune`) and training networks from the builder's JSON
(`train`), with run folders under `CACHE/runs` that `runs`, `resume` and `package` read.

| Command | What it does |
|---|---|
| `idrak tune` | Fine-tune language models (LoRA, QLoRA, DoRA, DPO/ORPO/SimPO); evaluate, chat, export, download, info |
| `idrak tune init` | Write a commented tune.json (model, data, adapter and training settings) for idrak tune --config |
| `idrak train` | Train a network from a builder JSON on a CSV or an image folder; writes a model package (.ikm) |
| `idrak resume` | Continue a training run from its last checkpoint (the remaining epochs, or --epochs N more) |
| `idrak runs list` (`runs`) | Training runs (from their JSON Lines logs): epochs, best epoch and loss, time, status |
| `idrak runs show` | One training run: settings, loss curves (text plot), the epochs, best epoch and time |
| `idrak runs compare` | Training runs side by side: settings, best epoch and loss, time, and their validation curves |
| `idrak predict` | Run a model package (.ikm) on new rows (CSV, JSON Lines, Parquet) or images and write the predictions |
| `idrak package` | Bundle a network, its weights, scalers and tokenizer from a folder into one model package (.ikm) |
| `idrak distill` | Distil a teacher model into a student: its token probabilities (on the fly, or precomputed as top-k logits with `--precompute`) or its written answers (`--generate`) |

```bash
idrak tune init -b Qwen/Qwen3-0.6B --data chats.jsonl && idrak tune --config tune.json
idrak tune train qwen chats.jsonl -o adapters/chat -w int4
idrak train network.json --data houses.csv -t price -o houses.ikm
idrak train cnn.json --data ./shapes -d vulkan:0    # a folder of class folders of images
idrak predict houses.ikm -i new-houses.csv -o priced.csv
idrak runs list && idrak runs show NAME && idrak resume NAME --epochs 20
idrak distill --teacher Qwen/Qwen3-8B --teacher-weights int4 --student Qwen/Qwen3-0.6B --data chats.jsonl --alpha 0.8 -o adapters/distilled
```

#### Fine-tuning with `idrak tune`

`idrak tune` trains LoRA / QLoRA / DoRA adapters for any model the library loads (Hugging Face ids, folders, .gguf
files, aliases) on any data `idrak data` reads, then evaluates, chats with and exports them. Nothing in it is specific
to a model family or an application: a project that needs a tuned model runs it with its data instead of writing its
own tuner. Its commands are `train`, `evaluate`, `chat`, `export`, `download` and `info`; `idrak help tune` lists
every option.

```bash
idrak tune train owner/model "data.csv?user={question}&assistant={answer}" -o adapters/qa --eval-fraction 0.02
idrak tune evaluate adapters/qa "data.csv?user={question}&assistant={answer}" --eval-fraction 0.02
idrak tune chat adapters/qa "a message"
idrak tune export adapters/qa -o merged
```

- **Data.** Conversations train the assistant's turns (OpenAI / Hugging Face messages, ShareGPT, Alpaca, question /
  answer pairs are recognized); text rows train every token. Columns map into a conversation with templates
  (`"data.csv?user={question}&assistant={answer}"`, the spec syntax under Data below), and `-s, --system` adds an
  instruction to conversations without one. `--eval-fraction` holds out part of the data; `evaluate` with the same
  data and fraction scores that part.
- **Fixed answers.** When every answer is one of a known list (labels, yes / no, multiple choice), `evaluate --choices
  a,b,c` (or `--choices auto`: the distinct answers in the data) scores each as the model's answer after the prompt and
  takes the most likely, generating nothing (the library's `AnswerScorer`: each prompt runs once for all its answers).
  It prints accuracy and recall per answer, the base model and the adapter side by side; `-o F.jsonl` writes each
  conversation's answer and every probability, for an application's own reports. `--samples 0` scores all rows.
- **Preference training.** `--loss dpo`, `--loss orpo` or `--loss simpo` trains on preference rows instead of
  conversations: TRL's layouts with a prompt and a chosen and a rejected answer (message lists or strings, or chosen and
  rejected as whole conversations that share their prompt). Only the answers are trained. DPO's reference is the model
  with its adapter disabled, so no second copy is loaded. `--beta` sets DPO's beta (0.1), ORPO's lambda (0.1) or
  SimPO's beta (2), `--margin` SimPO's target margin (1).
- **Optimizer and schedule.** `--optimizer adamw|adam|adamw8bit|sgd` (default adamw; `--weight-decay`, `--momentum`
  for SGD), `--schedule cosine|linear|constant|wsd` (default cosine; `--warmup` fraction of the steps, `--min-lr`,
  `--decay` for the decay part of warm-up/stable/decay).
- **Adapter type.** `--adapter-type dora` trains DoRA adapters (a magnitude per output on top of LoRA), saved in the
  PEFT format with `use_dora`; `lora` is the default. (`--adapter DIR` names an existing adapter folder.)
- **Long conversations.** A conversation longer than `--max-length` keeps its whole answer: the user message before it
  is shortened (its start kept) rather than the answer cut.
- **Progress.** The steps done, still to do and in total, the epoch and loss, elapsed time and ETA on the progress
  line; notable events (graph recording, FP8 checks, evaluation losses) print above it. Ctrl+C stops after the current
  step and saves the adapter so far.
- **Output.** The adapter in the PEFT format, and `idrak-tuning.json`: the base model as named on the command line,
  the system prompt and the maximum length. Any program can then load the folder (`TuningManifest.Read(folder)
  .LoadModel(folder, device)`), and every `idrak tune` command accepts the adapter folder in place of the model.
- **Speed.** Sequence packing, graph replay of the training step where the backend records graphs, fused LoRA products
  on matrix units (a frozen bfloat16 or 4-bit base read as stored in both directions), the optimizer over every adapter
  matrix in three passes; `--fp8` for the frozen base's forward products (checked against bfloat16 first), `-w int4` /
  `-w int8` for QLoRA. `train --profile` times a few steps per kernel instead of training.
- **Memory.** Results no backward step reads are released during the forward pass. When a step runs out of device memory
  the tuner steps down, each step at a small cost, and says so: feed-forward activations recomputed in the backward pass
  (`--recompute`), then activations held as bfloat16 between the passes (`--bf16-activations`), then activation
  checkpointing (`--checkpointing`, a third more compute). The flags start a run at that step.
- **Settings files.** `idrak tune init` writes a commented `tune.json` and `idrak suggest --base` one sized to the data
  and the device; `idrak tune --config tune.json` runs it. Its keys are the long options without the dashes, plus
  `command`, `model` and `data`, and an optional `"format": "idrak-tune/1"`; options on the command line win.

### Data

Datasets: `idrak data` reads sources from files, Hugging Face, GitHub, Kaggle, Zenodo and URLs (below); the
subcommands work on local files.

| Command | What it does |
|---|---|
| `idrak data` | Datasets: show, count, download, build (and preview, validate, stats, convert, dedupe, split, sample, mix) |
| `idrak data preview` | First rows, columns and their types of a data file (Parquet, JSON Lines, CSV) |
| `idrak data validate` | Check rows against the shape a command expects (chat, preference or table), with the first bad rows |
| `idrak data stats` | Row lengths in characters, words and tokens, a length histogram, and the rows over a context length |
| `idrak data convert` | Convert rows between CSV, JSON Lines and JSON (Parquet is read), and chat layouts to chat rows |
| `idrak data dedupe` | Remove duplicate rows (exact, or near: text compared without case, spacing and punctuation) |
| `idrak data split` | Split rows into train, validation and test files with a seed (stratified by a column with -t) |
| `idrak data sample` | A random (or, with -t, stratified) sample of rows |
| `idrak data mix` | Assemble a training set from several sources with weights (a dataset recipe) |

```bash
idrak data show "hf:owner/qa-set?config=main"
idrak data preview train.parquet
idrak data validate chats.jsonl --as chat
idrak data stats chats.jsonl -m qwen
idrak data convert alpaca.json chats.jsonl --as chat
idrak data split chats.jsonl -t label --seed 2
```

#### Data sources and recipes

`idrak data show`, `count`, `download`, `build` and `cache` read sources from files, Hugging Face, GitHub, Kaggle,
Zenodo and URLs and write JSON Lines (one JSON object per line): conversations as `{"messages": [...], "tools": [...]}`
and plain text as `{"text": ...}`, the layout Idrak's fine-tuning, Hugging Face's `datasets` and most training tools
read. A spec is a source with options after `?` (joined with `&`):

| Spec | Source |
|---|---|
| `hf:owner/name?config=main` | A Hugging Face dataset (`config`, `split`, `files`, `max_files`, `revision`) |
| `github:owner/repo[@ref][?files=src/**/*.cs]` | A repository's files as documents |
| `github:owner/repo?files=data/*.jsonl` | Data files in a repository |
| `github:owner/repo?release=latest&asset=*.csv` | Release assets |
| `kaggle:owner/dataset`, `zenodo:123456` | Kaggle and Zenodo datasets |
| `https://host/file.jsonl.gz`, a local file or folder | URLs and local data (JSON Lines, JSON, CSV, text, code, Parquet; compressed or archived) |

Options for any source: `take`, `skip`, `weight`, `columns=a,b`, `text=lines|paragraphs|document`, `documents=true`,
and conversations from columns: `user=...&assistant=...&system=...` (e.g. `user={question}&assistant={answer}`).

```bash
idrak data count "hf:owner/qa-set?config=main" "hf:owner/qa-set?config=main&split=test"
idrak data build "hf:owner/qa-set?config=main&user={question}&assistant={answer}" -o qa.jsonl --eval-fraction 0.02
idrak data show "github:owner/repo?files=src/**/*.cs"
idrak data build recipe.json -o train.jsonl
```

A recipe (also read by `idrak tune` and `idrak data mix`) mixes sources:

```json
{
  "sources": [
    {"source": "hf:owner/qa-set", "config": "main", "user": "{question}", "assistant": "{answer}", "weight": 1, "take": 2000},
    {"source": "hf:owner/instructions", "weight": 2, "take": 4000},
    "my-examples.jsonl?weight=0.5"
  ],
  "system": "You are a helpful assistant.",
  "seed": 1, "min_chars": 20, "eval_fraction": 0.02
}
```

Credentials come from the environment (or `idrak login`): `HF_TOKEN` (or `huggingface-cli login`), `GITHUB_TOKEN`,
`KAGGLE_USERNAME` and `KAGGLE_KEY` (or `~/.kaggle/kaggle.json`), `ZENODO_TOKEN`. Downloads are cached under the
cache folder (`--cache`, the config, `IDRAK_CACHE` or `~/.cache/idrak`), in `downloads/` laid out by source:

```
downloads/huggingface/datasets/<owner>/<name>/<commit>/<path in the repository>
downloads/huggingface/datasets/<owner>/<name>/parquet/<config>/<split>/00000.parquet   (the Hub's Parquet copy)
downloads/github/<owner>/<repo>/<commit>.tar.gz                                     (repository snapshots)
downloads/github/<owner>/<repo>/<commit>/<path>                                     (single files)
downloads/github/<owner>/<repo>/releases/<tag>/<asset>
downloads/kaggle/<owner>/<dataset>/<latest | vN>/<dataset>.zip
downloads/zenodo/<record>/<file>
downloads/urls/<host>/<path>
```

Branches and tags are resolved to their commit first, so new commits are downloaded again rather than read stale.
`idrak data cache` shows the size per source; `idrak data cache --clear` empties it; `download --refresh` fetches
again. The same features are available from code in the `Idrak.Datasets` library (`Dataset`, `HuggingFace`,
`GitHub`, `Kaggle`, `Zenodo`, `ChatRows`, `DatasetSpec`, `DatasetRecipe`).

### Retrieval

A hybrid search index (BM25, and vectors from a model with `-m`) over a folder of text files.

| Command | What it does |
|---|---|
| `idrak rag index` | Chunk a folder's text files and build a search index (BM25, and vectors with --model) |
| `idrak rag search` | The best passages of an index for a query, with scores (no generation) |
| `idrak rag ask` | Answer a question from an index's passages with a chat model, citing them |
| `idrak rag eval` | Retrieval quality of an index (hit rate, MRR) on question/passage pairs |

```bash
idrak rag index ./docs -o docs.idx && idrak rag ask "How do I reset?" --index docs.idx -m qwen
idrak rag search "install on Android" --index docs.idx --top 10 --format md
idrak rag eval --index docs.idx --questions questions.jsonl
```

### Measure

Speed and quality on the chosen device.

| Command | What it does |
|---|---|
| `idrak bench` (`b`) | Speed of a model (tokens per second, GFLOP/s, memory) or of the kernels on a device |
| `idrak eval` | Answer metrics of a chat model on held-out conversations (accuracy, exact match, F1) |
| `idrak perplexity` | Perplexity of a text under a model (compare weight formats and fine-tunes) |
| `idrak profile` | Time per layer, operation and kernel of one decoding step |
| `idrak check` | Compare token ids, chat templates, logits and greedy output with a transformers reference |
| `idrak tuning show` | The kernel choices measured on this machine (the tuning caches); --reset clears them |

```bash
idrak b qwen --save laptop && idrak b qwen --compare laptop
idrak b qwen --devices all                     # one table across every device
idrak b --kernels matmul,gemv -d cuda:0 -j     # the kernel benchmarks, without a model
idrak eval qwen held-out.jsonl --limit 100
idrak perplexity qwen wiki.txt -w int4
```

### Design

`suggest` reads the data, picks the task and writes a network (`network.json`, the builder's JSON), a training
setup (`train.json`) and the data preparation (`prep.json`), or a LoRA setup (`tune.json`) with `--base`, from rules
it explains with `--explain`; `--search N` measures N candidates on the device.

| Command | What it does |
|---|---|
| `idrak suggest` (`sg`) | Design a network and its training setup for a data set |
| `idrak explain` (`x`) | Layers, shapes, parameters, FLOPs and memory of a network.json |
| `idrak viz` | Draw a network.json as text, Mermaid or SVG |

```bash
idrak sg houses.csv -t SalePrice -e -o ./run   # design a network for a CSV, with the reasons
idrak sg ./shapes -n 6 -d vulkan:0             # try 6 CNN variants on the GPU, keep the best
idrak x ./run/network.json                     # parameters, FLOPs, memory
idrak viz ./run/network.json --as mermaid
idrak train ./run/network.json --data houses.csv --config ./run/train.json
```

### Developers

Project templates, the library's tests, ONNX, the generated kernels and telemetry.

| Command | What it does |
|---|---|
| `idrak new` | Writes a new project: console, webapi, rag or plugin |
| `idrak test` | Runs the library's tests from a source checkout |
| `idrak onnx import` | Imports an ONNX model into Idrak layers and saves a model package (.ikm) |
| `idrak onnx export` | Exports a model package (.ikm) or a network JSON to ONNX |
| `idrak onnx check` | Checks an ONNX model: Idrak's import against a round trip and against ONNX Runtime |
| `idrak kernels dump` | Writes the generated GPU kernels (PTX, SPIR-V, HIP source) for debugging |
| `idrak trace` | Runs an idrak command with telemetry printed live or written to JSON Lines |
| `idrak demo` | Trains a small built-in sample in seconds on the device, with the speed |
| `idrak shell` | An interactive prompt for idrak commands, with history and completion |

```bash
idrak new plugin MyFormat
idrak test --filter "cli" -d cpu
idrak onnx check model.onnx -d vulkan:0
idrak trace -o train.jsonl --levels training,batches -- train network.json --data houses.csv
idrak demo shapes -d vulkan:0
```

## Arabic messages

`--lang ar` (or `idrak config set lang ar`, or `IDRAK_LANG=ar`) prints the tool's messages in Arabic: `idrak help`
(titles, groups, every command's summary, the common options; a command's own option text stays English), usage
errors, questions, `doctor`, `devices` and the chat's own lines. JSON, CSV and Markdown output are never translated.

Most terminals (Windows' console host, Windows Terminal, VS Code's terminal, xterm, kitty, Alacritty, WezTerm, iTerm2)
draw characters left to right in the order they arrive, so Arabic written as stored shows reversed with its letters
apart. For them the tool shapes Arabic (each letter in its joined form, lam-alef as one ligature) and reorders each
line by the Unicode Bidirectional Algorithm: Arabic reads right to left, English words, numbers and paths inside it
keep their order, and table columns stay aligned. Terminals that do this themselves (GNOME Terminal and other VTE
terminals, Konsole, mlterm, mintty in Git Bash, macOS Terminal) get the text as it is, and so do files and pipes.
`--lang-render` overrides the choice (also the config's `lang-render` and `IDRAK_LANG_RENDER`): `visual` always
reorders, `visual-right` also ends right-to-left lines at the terminal's right edge, `logical` never reorders.

```bash
idrak help --lang ar
idrak doctor --lang ar
idrak devices --lang ar --lang-render visual     # a terminal the tool does not recognize
```

Words in reverse order or letters apart: try `--lang-render visual` (or `logical` if it looked right before). Boxes
instead of letters: the terminal's font has no Arabic; choose one that has (Cascadia Code, Courier New, DejaVu Sans
Mono; the Windows console host's default Consolas has none). `idrak help arabic` has the details.

## Output for scripts

Text goes to the standard output, progress lines and questions to the error output (and only on a terminal), so
pipes stay clean. `--json` prints one document per command, `--format csv` or `--format md` prints the tables (and
name-value results such as `show`, `version`, `doctor` or `eval`) as CSV or Markdown, and `-O FILE` writes the main
output to a file:

```bash
idrak dev -j | jq '.devices[].device'
idrak ls --format csv -O models.csv
idrak b qwen --format md >> results.md
```

Exit codes: 0 success, 1 the operation failed (a failed check, a model that is not cached, a server that does not
answer), 2 a usage error (an unknown option, a missing argument, a question with no terminal to ask on). Errors say
what to do next; `IDRAK_TRACE=1` adds the stack.

## Environment

`idrak env` lists every environment variable the libraries and the tool read, with its value, default and meaning
(`--all` includes unset ones; tokens and keys show only as set). `idrak help env` prints the same reference without
values, and every command's help ends with the variables that affect it.

Setting a variable differs in every shell (`$env:NAME="1"`, `set NAME=1`, `export NAME=1`, `set -x NAME 1`), so
`idrak env set` saves them in the config instead, and every idrak run sets them before it starts, in any terminal.
The programs the tool starts get them too, so `idrak test` runs the tests with them:

```
idrak env set                          # asks: a variable (number, name or part of it; ? lists all), its value
                                       # (Enter takes the suggestion, - removes it), then another or done
idrak env set IDRAK_CUDA_DEBUG 1       # or directly
idrak env set IDRAK_DEVICES cuda:0 --user   # also in the Windows user environment, for programs started without idrak
idrak env                              # the values, and whether each is saved or from the terminal
idrak env unset IDRAK_CUDA_DEBUG
idrak env reset                        # every saved one back to its default (asks; -y, --dry-run, --all-profiles)
```

A variable set in the terminal wins over the saved value; `--profile NAME` saves one for a profile only; tokens are
stored by `idrak login`, not here. The most used:

| Variable | Meaning |
|---|---|
| `IDRAK_CACHE` | The cache folder (`--cache` wins) |
| `IDRAK_CONFIG`, `IDRAK_PROFILE` | The config file and the profile in use |
| `IDRAK_DISABLE_CUDA`, `IDRAK_DISABLE_VULKAN`, `IDRAK_DISABLE_HIP` | Backends not looked for |
| `IDRAK_VULKAN_DEFAULT` | A Vulkan GPU may be the default device |
| `IDRAK_MATMUL`, `IDRAK_OFFLOAD` | Matrix product precision; spilling to system memory |
| `HF_TOKEN`, `HF_ENDPOINT`, `HF_HOME` | The Hugging Face token, hub address and folder |
| `IDRAK_API_KEY` | The key `serve` requires and the client commands send |
| `IDRAK_PORT` | The port `serve` and `ui` listen on and the client commands call (`--port` wins; default 7317) |
| `IDRAK_TRACE`, `NO_COLOR` | Error stacks; no colour |
| `IDRAK_LANG`, `IDRAK_LANG_RENDER` | The language of the messages; how Arabic reaches the terminal |
| `VK_ICD_FILENAMES`, `DOTNET_GCHeapHardLimit` | The Vulkan driver file and the heap limit a phone needs |

## Inside the tool

- `Program.cs` calls `CommandLine.Run`, which matches the longest run of leading words to a command ("cache info"
  before "cache"), parses the options (long and short forms, `--name=value`, `--` ends the options) and runs it.
  `CommandTable` lists the groups (one file each under `Commands/`, in the plan's order, which `idrak help` follows).
- A command derives from `Command`: its name (several words for subcommands), aliases, a one-line summary, its usage
  text (the usage line, then "Arguments:", "Options:", "Examples:" and limits or gaps; `Help` adds the common options
  and the environment section), its value options, flags and short forms, and `Run(CommandContext)` returning an exit
  code. A usage error is a `UsageException` (exit code 2); any other exception prints its message and exits with 1.
- `CommandContext` holds the parsed arguments, the common options (`Device`, `CacheFolder`, `Json`, `Format`,
  `Offline`, `Seed`, `Threads`, `Timeout` and `TimeoutToken`, `Plain`, `Quiet`, `Verbose`), the config file and the
  output helpers (`Write`, `Detail`, `Table`, `Fields`, `WriteJson`, `Error`, `Log`).
- `Shared/` holds what several groups use: model arguments and loading (`Models`, `ModelCache`), the network
  (`Http`: downloads with `--offline`, `--timeout`, `--cache` and progress), `Progress` (progress lines and the
  training callback), `Terminal` (colour, questions, `--dry-run`, Ctrl+C), `Units` (sizes, counts, times),
  `EnvironmentVariables` (the variable table), `HelpTopics`, `GenerationSettings`, `ImageFiles`, `RowFiles` and the
  design rules.
- Messages: write a message as `Messages.T("No file {0}.", path)`; the English text is the key, and its Arabic
  translation goes in `Shared/Messages.Arabic.cs` (a test fails for a message without one, and for an entry no longer
  used). `Messages.Begin` sets up the run's language and, for Arabic on a terminal that does not reorder text itself,
  wraps the output in `VisualWriter` (`Shared/VisualText.cs`, with `Bidi.cs` for the Unicode Bidirectional Algorithm,
  `ArabicShaping.cs` and `UnicodeTables.cs`, generated by tools/unicode/make_tables.py).
- The assembly is named `Idrak.Cli`, not `idrak` (assembly names ignore case, so it would clash with the Idrak
  library); the installed command is still `idrak`. `idrak tune` and `idrak data` keep their own argument parsing
  (`Commands/Train/TuneTool.cs`, `Commands/Data/DataTool.cs`, writing through `Shared/ToolConsole.cs`); `TuneCommand`
  and `DataCommand` turn idrak's options into theirs.
- Tests: `tests/Idrak.Tests` has a CLI group per command group (`Cli*Tests.cs`), all run in-process through
  `CommandLine.Run` with captured output and no network (the hub is a local stand-in; models are the fixtures in
  `tests/Idrak.Tests/data`). They check exit codes, text and JSON, every command's help layout, short forms, the
  environment table against the sources, and that no command, option or help text names a provider:
  `IDRAK_FILTER="cli " IDRAK_DEVICES=cpu dotnet run -c Release --project tests/Idrak.Tests`.
