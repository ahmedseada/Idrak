# idrak: one command-line tool

Idrak had two separate tools (`idrak-tune`, `idrak-data`), and the most common actions (chat with a model, serve
it, check the devices, benchmark) needed `dotnet run` on a sample or the test runner. This plan puts everything behind
one `dotnet tool` named `idrak`, with subcommands; the two old tools are removed (folded into `idrak tune` and `idrak
data`, see "Old tools removed" below). Built on branch `idrak-cli` by eight groups in parallel, then
polished across the groups ("Polish after the merge" below); the tool's guide is src/Idrak.Cli/README.md.

Rules that apply to every command:

- No dependencies beyond the Idrak packages (the argument parsing is written in the tool); the
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
| `-n` | a count: `--search N` / `--repeat N` / `--rows N` | suggest / bench / data preview, data sample |
| `-y` | `--yes` (no confirmation) | rm, cache clear, data dedupe |
| `-e` | `--explain` | suggest, doctor |
| `-b` | `--base MODEL` | suggest, tune, distill |
| `-k` | `--kv FORMAT` | chat, run, serve, bench |
| `-w` | `--weights FORMAT` (int8, int4, bf16 or any registered packed format) | chat, run, serve, bench, quantize |
| `-i` | `--input FILE` | run, batch, predict, embed |
| `-O` | `--output FILE` | every command that prints a result |

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

Built with the group beyond the rows above (found while building):

- `report --source DIR` (the checkout `--tests` runs from; otherwise found from the current folder upwards) and
  `report --readme` printing both README tables' rows; `doctor --rc FILE` and `setup android --rc FILE` (the start-up
  file the safe fixes add lines to, each line once); `plugins list --added` (only what the plug-ins added; their names
  are marked "+" otherwise); `update --prerelease`; `env NAME...` (only those variables); `idrak environment` as an
  alias of `env` (so `help environment` works); `init --profile NAME` and `--dry-run`.
- `idrak completion --complete -- WORDS...`: the candidates the scripts ask for, so completion follows the commands,
  options, device names, weight and KV formats, profiles, cached models and aliases of this install (plug-ins too).
- A group word alone (`idrak cache`, `idrak config`) lists its subcommands instead of "unknown command".
- `idrak login github` keeps the token in the config ("tokens.github") and the tool passes it on as `GITHUB_TOKEN`
  (the library reads only that variable); config listings show any key holding a token or key only as "set".
- The library gains a read-only `Idrak.Diagnostics.DeviceListing` (`Backends`, `All()`, `Describe(device)`: memory,
  compute units, lanes, kernel width, matrix units, driver, hardware kind) that `devices`, `doctor`, `version` and
  `report` read; the test runner's `--list-devices` can move onto it.

Gaps (Health): `report --bench` times the float32 products on each device (GFLOP/s); model speeds come from `idrak
bench` once the Measure group lands. `report --tests` runs the test runner of a source checkout as a child process (no
test list ships with the tool). `devices` shows no compute units for Vulkan devices (core Vulkan reports none) and no
hardware kind for CUDA devices (not read yet). `doctor` checks the Vulkan driver files on Linux only (Windows keeps
them in the registry).

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

Built (branch `cli-run`): every command above, with the second pass's `chat --file`/`--mcp` and `run --schema`.
Added while building (all built and tested):

- `--no-think` beside `--think` (chat, run, batch, compare, agent; `/think default` returns to the model's own mode);
  `--temperature 0` means greedy decoding.
- `chat`: `/file FILE` (a file's text joins the next message), `/system none`, a line ending with `\` continues on the
  next, `//text` sends a message that starts with `/`; Ctrl+C stops the answer, not the chat; `/copy` uses the
  terminal's clipboard escape (OSC 52) and prints the answer elsewhere.
- `run --mcp SERVER` as in chat; `run` writes the figures to the error output with `-v` so pipes stay clean.
- `batch --batch-size N`; prompts may be strings, `{"prompt", "id"}` or `{"messages"}`; a torn last line from an
  interruption is dropped and answered again.
- `compare --weights-b F`, `--kv-b F` (two settings of one model), `--width N` (0: one after the other); one seed for both.
- `complete --stop TEXT`; `template --source`, `--think`; `embed --whole`, `--max-length N`, `--batch-size N`.
- `agent --workspace DIR`, `--read-only`, `--rounds N`, a task argument (or tasks read line by line), and `-y, --yes`;
  without `--yes` each command is asked about.
- `tools test --tool NAME`, `--args JSON`, `--timeout S`; tool assemblies also export static `Tool` properties or fields.
- Shared helpers for serve and bench: `Shared/GenerationSettings.cs` (sampling options, `/set`), `Shared/ToolAssemblies.cs`,
  `Shared/StructuredOutput.cs`, `Shared/StandardInput.cs` (injectable input: tests drive chat in-process), and
  `Models.CacheLayout`/`CreateChat`/`CreateGenerator`/`ContextLength` (generation uses a 4096-token window unless
  `--context` says otherwise, since the KV cache is sized by it); model paths that do not exist are a usage error
  instead of a download attempt.

Gaps (in the commands' help too):

- `run --schema`: the library has no JSON-schema constrained sampling (plug-in gap 13). The schema is given to the
  model as an instruction and the answer is checked afterwards (types, required properties, items, enums), exit 1 when
  it does not match.
- `embed`: no embedding heads for decoder models trained as embedders (last-token pooling, instruction prefixes); the
  command mean-pools the last hidden states (through the library's `TextEncoder`) and scales to unit length.
- `/stats` memory: the library reports no device memory in use; the process and managed memory are shown.
- `agent`: `CodingAgent` has no approval hook, so the command runs the coding tools through its own tool loop with
  `ToolRegistry.RequireApproval` on `run_command` (no task verification, which only suite runs need).

### Serve

| Command | What it does | Priority |
|---|---|---|
| `idrak serve MODEL [MODEL...]` | The chat API (streaming, tool calls) and the OpenAI-style API on one port; several models loaded on demand with an idle unload timeout; `--host`, `--port`, `--api-key`, `--cors ORIGIN`, `--max-concurrency N`, `--keep-alive DURATION` | 1 |
| `idrak server ps` (`idrak ps`) | Models loaded in a running server, their memory and last use | 2 |
| `idrak server stop` / `server load MODEL` / `server unload MODEL` | Controls a running server through its API | 3 |
| `idrak api PATH [JSON]` | Calls a running server's endpoint from the command line (for scripts and checks) | 3 |
| `idrak ui MODEL` | A small web chat page in the browser over the same server | 3 |
| `idrak mcp serve TOOLS.dll` | Serves an assembly's tools over MCP | 3 |

Built (branch `cli-serve`; also `serve --metrics`, `--log-requests`/`--log-content`, `server keys add/list/rm` and
`ping` from the second pass below). Added while building, and built:

- `NAME=MODEL` on serve and ui names a served model; a file or folder is named after itself, an alias keeps its name.
- The default port is 7317, Idrak's own (decided 2026-10-03: 11434 is another local server's default); `-p/--port`, then `IDRAK_PORT`, then the config's `serve.port` choose another, and `-p 0`
  picks a free one. `-p 11434` (or `serve.port` 11434) gives drop-in use for clients that expect that port. 7317 lies
  in IANA's registered range 7300-7359 (a financial exchange's own network protocol), which no tool of a developer
  machine uses by default, so it was kept. Each running server records itself in `CACHE/servers/PORT.json`, so `ps`,
  `api`, `ping` and `server ...` find it without `--port`; they take `--port` or `IDRAK_PORT` first, then the server
  started last, then `serve.port` and 7317.
- `/ui` is served by every server (not only `ui`); `ui --no-browser` prints the address instead of opening it.
- Control endpoints the server commands use: `GET /idrak/ps`, `POST /idrak/load`, `/idrak/unload`, `/idrak/stop`,
  `GET /idrak/status`; `api --method NAME`; `mcp serve --list`.
- `IDRAK_API_KEY` is the default of `--api-key` for serve and for the client commands (in the environment table).
- Library: the OpenAI-style API is `MapCompletionsApi` in Idrak.AspNetCore (`CompletionsApiOptions` with
  `ChatModel(name, IChatModel)` and `Embeddings(name, IEmbedder)`, and `CompletionsTranslation`), neutrally named.

Gaps (Serve):

- `/v1/embeddings` answers 404 from `idrak serve`: the library has no embedder over a pretrained language model (only
  `TextEncoder` over a trained package), so nothing can be registered yet; the endpoint works when an `IEmbedder` is
  given (`CompletionsApiOptions.Embeddings`).
- The chat API's `/api/generate`, `/api/show`, `/api/embed` and `/api/pull` are not provided by Idrak.AspNetCore.
- Tool calls go to the client (`ToolExecution.Client`); serving with server-side tools waits for `--tools FILE.dll`
  loading shared with chat.
- `ps` reports a model's memory as the device memory it took when it loaded (the engine does not expose a model's size).
- `mcp serve` speaks MCP over standard input/output only; HTTP would need the MCP ASP.NET Core package.
- `--timeout` and `--offline` (common options of the second pass): done in the polish; `ping` waits `--timeout` (else
  10 s) per call, `api` and `server ...` give up when `--timeout` runs out.

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

Built (agent 4), with what building them added: `pull` takes `owner/name`, `owner/name:TAG` and `owner/name/FILE.gguf`
(one GGUF file of a repository, by quantization tag or name), a `.gguf` URL, `--file`, `--revision`, `--token`,
`--force` and `--dry-run`; it downloads through `ModelSource.DownloadAsync` (the files loading reads, into the folder
loading looks in) and prepares a pulled GGUF file's description in the cache. `list` takes a filter, `--sort
used|name|size` and `--all` (folders prepared from GGUF files, and Hugging Face's own cache, read-only), shows
interrupted downloads as "partial", and records last use in `models-last-use.json` in the cache (the commands that
read a cached model update it). `rm` takes several names and `owner/name@REVISION`, removes a GGUF file's prepared
folders with it and the empty folders left, and refuses Hugging Face's own cache. `memory` takes several formats
(`-w int8,int4`), `--batch N` and `--memory SIZE`; bytes per weight are measured by packing a sample in each
registered format, KV bytes from each registered KV layout's row width. `verify` takes `--read` (every tensor read,
all values finite) and `--hub` (sizes and SHA-256 against the hub). `inspect` takes `--filter`, `--limit`,
`--no-tensors`; `diff` takes `--tolerance`, `--filter`, `--limit`. `idrak families` (second pass) is built here.
`--offline` (a common option from the foundation) is honoured: pull reports a cached model and fails otherwise, search
fails, every other model command reads only the cache. A model the cache lacks is an error naming `idrak pull`.

Gaps (Models): a GPU's total memory is not reported by the library (only a configured `GpuMemoryLimit`), so `memory`
shows "?" for GPUs unless `--memory SIZE` is given; there is no loader for pre-packed language-model weights, so
`quantize --out` writes the packed module (`Module.Save`) with the config and tokenizer, and chat/run/serve pack at
load time with `-w`; the library has no GGUF writer and loads no language model from an `.ikm` package, so `convert`
writes Hugging Face folders only (from GGUF or another folder, in bf16, f16 or f32).

Improvement for the shared model loading (Shared/Models.cs, agent 2): `Models.Resolve` should download into
`--cache` (a downloader on `CommandContext.CacheFolder`, as `ModelCache.Downloader` makes) and prepare GGUF files there,
and call `ModelCache.Touch` so `idrak list` shows the last use by chat, run and serve. Done in the polish (below).

### Train and fine-tune

| Command | What it does | Priority |
|---|---|---|
| `idrak tune ...` | The former `idrak-tune` as a subcommand: LoRA, QLoRA, DoRA, DPO/ORPO/SimPO, optimizers and schedules | 2 |
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
| `idrak data ...` | The former `idrak-data` as a subcommand | 2 |
| `idrak data preview FILE` | First rows, columns and types (Parquet, JSON Lines, CSV) | 2 |
| `idrak data validate FILE --as chat\|preference\|table` | Checks rows against the shape a command expects, with the first bad rows shown | 2 |
| `idrak data stats FILE --model MODEL` | Token counts per row with a model's tokenizer, length histogram, how many rows exceed a context length | 2 |
| `idrak data convert IN OUT` | Between CSV, JSON Lines and Parquet, and from common chat layouts to the chat rows the tools read | 3 |
| `idrak data dedupe FILE` / `data split FILE` | Removes duplicates (exact and near); train, validation and test split with a seed | 3 |
| `idrak data sample FILE -n N` | A random or stratified sample of rows | 3 |
| `idrak data mix RECIPE.json` | Assembles a training set from several sources with weights (the dataset recipes) | 3 |

Train and data, as built (agent 5): `idrak tune` runs idrak-tune's code (`Commands/Train/TuneTool.cs`) and `idrak data`
idrak-data's (`Commands/Data/DataTool.cs`); the two old tools compiled the same files as forwarders until they were
removed (below). Added while building:

- `idrak tune`: `-b`/`--base MODEL` (the model as an option), `-w`/`--weights int8|int4|bf16`, `-k`, `-o`, `-s`, model
  aliases, and a tune.json given with `-C`/`--config` (keys are the options without dashes plus `command`, `model`,
  `data`; the command line wins; config files may hold comments). `--json` gives the output lines in one document.
- `idrak train`: `-t`/`--target` (repeatable), `--ignore`, `--task`, `--epochs`, `--patience`, `--lr`, `--batch`,
  `--optimizer`, `--weight-decay`, `--validation`, `--seed`, `--no-scale`, `--run DIR`, and a train.json through
  `--config` (as `idrak suggest` writes it). A class column may hold names; CSV rows can feed image or sequence inputs
  when their count matches. Runs go to `CACHE/runs/TIME-NAME` (run.json, network.json, scalers, `last.ikw`/`best.ikw`,
  `log.jsonl` in the library's telemetry format).
- `idrak runs` alone is `runs list`; `runs list FOLDER` reads any folder of runs or telemetry logs.
- `idrak predict --top N` (the N most likely classes) and `-o` (rows with the predictions, by extension).
- `idrak package --checkpoint best|last|FILE`.
- `idrak data dedupe --columns`, `--near`, `--dry-run`, and in-place rewriting only with `-y`; `data split --validation
  --test -t` (stratified); `data convert --as chat|preference|text -s SYSTEM -f`; `data stats --context --column`.

Gaps (library): the optimizer state and learning-rate schedule are not checkpointed, so `resume` continues from the
weights with a fresh optimizer; there is no image decoding in the library (the tool reads 8-bit PNG, BMP and Netpbm itself, with one decoder shared with suggest);
writing Parquet is not in the library (`data convert` reads it, writes JSON Lines, JSON, CSV or TSV); `distill` waits for
the teacher pattern and only explains the workaround (`idrak batch`, then `idrak tune train`); `tune init` writes the
library's defaults until `idrak suggest` can size them; idrak tune loads base weights as int8, int4 or bf16 only (not
other registered packed formats).

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

Built (Retrieval and measure), with what was added while building and the gaps found:

- `bench`: `--prompt-tokens N`, `--tokens N`, `--small` (small kernel shapes for slow devices and tests), `-n`/`--repeat`
  (median of the runs), `--devices all|LIST` and `--matrix` (one table across devices, and across weight x KV
  formats; `-w`/`-k` take comma-separated lists there; a run that fails is reported and the others go on). Saved
  results are `bench/NAME.json` under the cache folder; `--compare` matches by run and measurement name and marks
  changes over 3% better or worse. The kernel benchmarks use the public tensor API on any device (matrix products in
  float32 and mixed bfloat16, one-row products in every packed format or `-w`, an attention layer's decoding step
  per KV format); the test runner's `--bench-vulkan`/`--bench-gemv` time backend internals the tool cannot reach, so
  they stay in the test runner. GFLOP/s of a model is 2 x parameters per token (attention not counted).
- `eval`: `--metric`, `--max-tokens`, `--batch`, `--limit`, `--think`/`--no-think`, `-o` (answers as JSON Lines).
- `perplexity`: `--window N` (non-overlapping windows), `--max-tokens N`.
- `profile`: `--prompt-tokens N`, `--top N`; each top-level layer timed with the device drained, the inner layers and
  operations from telemetry, kernels from the GPU profiler where the device times them (CUDA).
- `check`: exit code 1 when anything differs (the sample returned 2, which the tool keeps for usage errors).
- `tuning show`: `--backend cpu|cuda|vulkan`; also reads the CPU's measured cut-overs.
- `rag index`: `-m MODEL` adds vectors (hybrid, fused with k 60), `--chunk`, `--overlap`, `--sentences`,
  `--extensions`, `--max-tokens`, `-f`/`--force`; one index file holds the library's index plus the vectors and the
  name of the model that made them. `rag search`/`rag eval`: `--top`; `rag eval` reads `{"question", "document"
  and/or "passage"}` rows. `rag ask`: `--top`, `-s`, `--max-tokens`, `--temperature`, `--think`/`--no-think`, streams
  the answer and lists the cited passages.
- Gaps: the library has no embedding-model loader (a bi-encoder from a Hugging Face embedding model), so `rag index
  -m` embeds with a chat model's averaged hidden states, a rough complement to BM25; `RetrievalIndex.Save` does not
  keep the vectors of a non-`TextEncoder` embedder (the tool stores them itself); eval metrics beyond
  number/exact/contains/F1 wait for plug-in gap 16. `--format csv|md` and `--output` (second pass, now applied) apply once the
  Health agent's common options land.

### Design a model

| Command | What it does | Priority |
|---|---|---|
| `idrak suggest DATA` (`sg`) | Reads the data, picks the task and writes a network and training setup: `network.json` (the builder's JSON, so `Network.FromJson` and `idrak train` read it), `train.json` and `prep.json`; `--search N` tries N candidates briefly on the chosen device and keeps the best; `--explain` gives the rule behind each choice | 2 |
| `idrak explain network.json` (`x`) | Layers, output shapes, parameters, FLOPs per sample and memory for training and inference on the chosen device | 2 |
| `idrak viz network.json` | The network as a text diagram, or Mermaid / SVG (`--as`; `--format` is the common table format) | 3 |

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

Built (Design group): the files `suggest` writes are documented in code. `network.json` is the builder's own JSON
(`idrak-network/1`). `train.json` (`idrak-train/1`): task, target, optimizer, learningRate, weightDecay, batchSize,
epochs, earlyStopping, schedule, warmupEpochs, loss (`mse`, `cross-entropy`, `token-cross-entropy`), metric,
validationFraction, seed. `prep.json` (`idrak-prep/1`): kind `table` (scaled numbers with their fill, mean and std;
one-hot values), `text` (a word vocabulary and length), `images` (channels, height, width, classes, augmentation),
`language-model` (a character vocabulary and window) or `chat-mapping` (with `--base`), with the target, split and
balance; `Shared/DataPreparation.cs` applies it to the data, so `idrak train` can reuse it. `tune.json`
(`idrak-tune/1`) holds `idrak tune` long option names as keys (model, data, loss, rank, alpha, targets, lr, epochs,
max-length, batch-tokens, bf16 or int4, ...). Chat rows without `--base` get a small character-level language model
from scratch (with a warning); preference rows need `--base`.

Additions made while building: `--max-params` takes `50k`, `2M`; `suggest` takes `-w`, `-k`, `--context` for the
`--assist` model; `explain` takes `--batch N` (else the batch of a `train.json` next to the network, else 32) and a
folder holding `network.json`; `viz` prints an SVG with a white background that any viewer shows.

Gaps (Design): images are decoded by the library (`ImageCodecs`, `ImageFolderSource`: PNG, BMP, PGM/PPM, and any
registered codec); the tool still reads JPEG headers itself, so JPEG folders are profiled, but trained by `--search` and
`train` only when a JPEG codec is registered. The "augment" list of prep.json maps onto the library's transforms
(`RandomFlip`, `RandomRotation`, `RandomShift`) but is not applied yet, to keep the search and training results as they
were; applying it is a later step. The library does not
report a GPU's memory size (only a configured `ComputeResources.GpuMemoryLimit`), so memory checks on a GPU say the
size is unknown unless a limit is set.

### Developers

| Command | What it does | Priority |
|---|---|---|
| `idrak new console\|webapi\|rag\|plugin NAME` | Project templates; the plug-in starter registers a format and has a test that runs it without internal access | 2 |
| `idrak test [--filter TEXT] [--device NAME]` | Runs the library's test runner from a source checkout (the commands the README and installation guides give) | 3 |
| `idrak onnx import/export/check` | ONNX conversions with an output check against ONNX Runtime when installed | 3 |
| `idrak kernels dump [ptx\|spirv\|hip]` | The generated kernels, for debugging | 3 |
| `idrak trace COMMAND ...` | Runs a command with telemetry printed live (layers, batches, kernels) or written to JSON Lines | 3 |

Built on the `cli-dev` branch, with these additions found while building: `new --source DIR` (project references to a
source checkout instead of packages, for working against unreleased changes), `-o/--out` and `-f/--force` on `new`;
`onnx export` also takes the network builder's JSON (exported with initialized weights) and `--shape`; `onnx check`
also compares with a reference file of another framework's outputs (`FILE.onnx.expected.json`, as
tools/pytorch writes it, or `--expected FILE`) and takes `--batch` and `--tolerance`; `kernels dump` writes each HIP
kernel's parameters next to the source; `trace` takes `--levels` (training, batches, gradients, layers, operations,
inference, tools, engine) and `--sync` (true GPU timings), and with `--json` wraps the traced command's document in
its own; `demo` and `shell` from the second pass are in this group.

Gaps (Developers):

- `onnx check` against ONNX Runtime needs the Idrak.Onnx.Runtime assembly (and Microsoft.ML.OnnxRuntime with its
  native library) next to the tool or loaded with `--plugin`: the tool does not depend on it, so a plain install
  reports the comparison as unavailable and still runs the round-trip and reference checks.
- `kernels dump` writes the kernels in their default shapes (Vulkan at its widest width, cooperative-matrix products
  in 16 x 16 x 16, PTX for the default kernel shapes); the library has no public way to ask a device which variants
  it built, so per-device dumps wait for that. The tool reads the generators through `InternalsVisibleTo` (Idrak
  grants it to Idrak.Cli for this command only).
- `new` refers to packages of the tool's own version; a local prerelease build asks for `VERSION-*`, which resolves
  only where such packages are published (a local feed), so `--source` is the way to use unreleased code.
- `test` runs the runner as a child process (`dotnet run`), so it needs the .NET SDK, not only the runtime.
- `shell` completes command names and options; completing model names and file paths waits for the shared
  completion helper (`idrak completion`).

### More from a second pass

Found while planning the build; each belongs to the group in brackets and is built with it.

| Command or option | What it does | Group | Priority |
|---|---|---|---|
| `idrak help topics` and `idrak help TOPIC` | Concept pages without leaving the terminal: devices, formats (weights, KV caches, checkpoints, datasets), models (supported families and features), precision, plugins, config, env, exit codes | Health | 1 |
| `idrak formats` | Every registered weight, KV cache, checkpoint, dataset and tool-call format, built in or from a plug-in | Health | 2 |
| `idrak families` | Supported model families with what each supports (windows, soft-capping, RoPE scalings, experts, GGUF) | Models | 2 |
| `idrak doctor --fix` | Prints the exact commands that fix what failed (and runs safe ones with `--yes`) | Health | 2 |
| `idrak doctor --network` | Checks the hub and proxy reachability and tokens (set or not) | Health | 2 |
| `idrak login hf\|github\|kaggle` / `logout` | Stores a token where the library already looks for it (never printed) | Health | 2 |
| `idrak demo xor\|spirals\|shapes\|gpt` | Runs a built-in sample in seconds to show the library works on this device, with the device and speed | Developers | 2 |
| `idrak report --readme` | Prints the README tested-on table rows for this machine, ready to paste | Health | 1 |
| `idrak bench --devices all` / `--matrix` | The same benchmark on every device, or across weight and KV formats, in one table | Measure | 2 |
| `idrak serve --metrics` | A metrics endpoint (requests, tokens per second, queue, memory) in the plain text format monitoring tools read | Serve | 2 |
| `idrak serve --log-requests FILE` | One JSON line per request (time, model, tokens, speed; no prompt text unless `--log-content`) | Serve | 2 |
| `idrak server keys add/list/rm` | API keys for `serve`, stored hashed in the config | Serve | 3 |
| `idrak ping URL` | Checks a running server (any compatible one): reachable, models, latency | Serve | 2 |
| `idrak chat --mcp SERVER` | Uses an MCP server's tools in the chat | Run | 3 |
| `idrak chat --file FILE` | Adds a text file's content to the conversation (several allowed) | Run | 2 |
| `idrak run --schema schema.json` | Structured output when the library has a JSON-schema logits processor (plug-in gap 13); until then a clear message | Run | 3 |
| `idrak shell` | An interactive prompt for idrak commands with history and completion | Developers | 3 |
| `idrak setup android` | Prints (or with `--yes`, runs) the Android steps of installation/android-termux.md that are safe to automate, then `doctor --android` | Health | 3 |

Common options added by the second pass (built by the Health agent with the foundation, used by every group):

| Option | Meaning |
|---|---|
| `--offline` | Use only what is cached; any download is an error naming the missing file |
| `--threads N` | CPU threads for the CPU backend and host work |
| `--seed N` | One seed for sampling, shuffling and initialization, for reproducible runs |
| `--format text\|json\|csv\|md` | Output format for commands that print tables (`--json` stays the short way to JSON) |
| `--output FILE` (`-O`) | Write the command's main output to a file |
| `--color auto\|always\|never` | Colour control (`NO_COLOR` still wins) |
| `--plain` | No Unicode box or progress characters (screen readers, old terminals, logs) |
| `--timeout DURATION` | Give up after a time (downloads, server calls, long runs) |

For the other groups: `CommandContext` has `Offline`, `Threads` (already applied to `ComputeResources.MaxCpuThreads`),
`Seed`, `Format` (`Json` is true for `--format json`; `Table` prints CSV or Markdown tables; `Write` lines are left out
of CSV), `Output` (the `-O` file, `ConsoleOutput` stays the console), `ColorMode`, `Plain`, `Timeout` and
`TimeoutToken` (cancelled when the time runs out), and `Log`. `Shared/Terminal.cs` has `Paint` (colour rules),
`Confirm`/`Ask`/`AskSecret` (`--yes`; without a terminal an error naming it), `DryRun`, `ConfirmFlags` and
`ConfirmShortForms` (`--yes`, `-y`, `--dry-run`) and `Interrupt` (Ctrl+C: the first press cancels a token);
`Shared/Progress.cs` has `ProgressLine` (rate and time left on the error output, only on a terminal);
`Shared/EnvironmentVariables.cs` is the variable table (`HelpSection(command)` ends each command's help; add the
command names a variable affects there); `Shared/HelpTopics.cs` the concept pages; `Command.BeforePlugins` runs
before `--plugin` assemblies load.

Later, not in this build: messages in Arabic (`--lang ar`) given the library's name and audience; a plug-in marketplace
listing; remote devices (run a command on another machine's `idrak serve`).

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
| The tool | `IDRAK_CONFIG`, `IDRAK_TRACE`, `NO_COLOR`, `IDRAK_API_KEY` (serve and the server commands) |
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
- `idrak-tune` and `idrak-data`: their code moves behind `idrak tune` and `idrak data`. Planned as thin forwarders
  for a release; removed instead before any release (see "Old tools removed").
- Tests: `tests/Idrak.Tests` gains CLI groups that run commands in-process through `CommandLine.Run` with captured
  output (no network: models from the test fixtures in `tests/Idrak.Tests/data`), checking exit codes, text and JSON
  output; a test checks every command's help, that no two options of a command share a short form, and that no
  command, option or output names a provider.

## Old tools removed

Done (branch `cleanup-old-tools`): the forwarders `idrak-tune` (src/Idrak.FineTuning.Cli, package
`Idrak.FineTuning.Cli`) and `idrak-data` (src/Idrak.Datasets.Cli, package `Idrak.Datasets.Cli`) are deleted, with no
deprecation period: nothing was released or shared yet (versions 0.y.z). Their READMEs moved into
src/Idrak.Cli/README.md (fine-tuning features, the data spec syntax, recipes, credentials and the cache layout).
`TuneTool` and `DataTool` serve only idrak: their usage text is idrak's own (no string replacements in `TuneCommand`
and `DataCommand`), the `RunTool` entry points and idrak-data's exit code 2 for a failed data command are gone (idrak
exits with 1, a usage error with 2), and `ToolConsole` lost its standalone console with the live status bar (the
`IToolHost` is required). The tune.json format stays `idrak-tune/1`: it names the kind of file, alongside
`idrak-train/1`, `idrak-prep/1` and `idrak-network/1`, not the tool. `idrak tune init` now writes it and `idrak tune
--config` accepts it (it had rejected the key, so a tune.json from `idrak suggest --base` did not run) and refuses
another file's format. The pack loops in the workflows and tools/publish.ps1 pack every project under src/, so they
needed no change.

## Polish after the merge

Built after the eight groups were merged (branch `cli-polish`), so the tool reads as one:

- Environment help: every command's help ends with the section generated from `Shared/EnvironmentVariables.cs`; the
  hand-written "Environment:" lines are gone. The table names the commands each variable affects in groups (devices,
  models, remote data, caches, logins, server clients, the network): the data subcommands read local files only, so
  the hub tokens no longer show there; `demo`, `onnx`, `convert`, `diff`, `inspect`, `suggest`, `rag search` and `rag
  eval` were added where they read models or run on a device, `setup android` to the Vulkan driver variables, `cache
  info`/`clear` to the tuning caches. A test checks that every name in the table is a command.
- One help layout: the summary, `Usage:` and aliases, the command's text with "Arguments:" and "Options:" sections
  (short forms shown as `-x, --long`, descriptions in one column), "Examples:", then limits and gaps, the common
  options and the environment. `idrak help` lists the commands by plan group (setup and health, run models, serve,
  models, train, data, retrieval, measure, design, developers) with their aliases. Help is reflowed to 118 columns. A
  usage error prints the usage line, not the whole help. A test checks the layout of every command's help and that
  every option a command accepts is described; options that did nothing were dropped (`--kv` for embed and
  perplexity, `--adapter` for memory and suggest).
- Common options everywhere, through `Shared/Http.cs`: every download honours `--offline` (a refused request names the
  missing file; the library still falls back to a copy it downloaded before), `--timeout` (each request and the body
  being read) and `--cache`, and draws a progress line per file; `idrak tune` and `idrak data` get the same through
  `Shared/ToolHost.cs`. Model resolution
  looks in the cache first (no network; a cached hub model is used as it is, `idrak pull` updates it), prepares GGUF
  files under `--cache`, records the last use for `idrak list`, and names `idrak pull` for a repository's GGUF file
  (`owner/name:TAG`) it does not have. `--timeout`: `ping` per call, `api` and `server ...` for the whole call
  (streamed answers too), `run`, `train` and `mcp serve` stop as at Ctrl+C (`Terminal.Interrupt`), and a command
  that gives up says it was the time limit. `--format csv|md` covers the commands that printed name-value results as
  text (`show`, `version`, `plugins list`, `doctor`, `eval`, `rag eval`) through `CommandContext.Fields`, and every
  table's headers are in sentence case. `--seed` reaches `suggest` (split, initialization, search, the files it
  writes, the `--assist` model), `demo`, `rag ask`, `profile` and `onnx check`.
- Progress and confirmation: model loading (elapsed time), training (epoch and batch, as a trainer callback), model
  and kernel benchmarks and `eval` draw a `ProgressLine`, which steps aside for output lines (`Erase`); a progress
  line no longer overflows on its first draw. `data dedupe` asks before overwriting in place, as `rm` and `cache
  clear` do; a question is refused with `--json` (the output is one document) and names `--dry-run` where the command
  has it.
- Shared code instead of copies: `Shared/Units.cs` (sizes, counts, durations, ages; five formatters before), one
  image decoder (`ImageFiles`, train and predict now read BMP and resize as suggest does), `suggest --base` finds its
  model as the model commands do (`ModelCache.TryLocate`, `ModelFacts`), `Models.Load` takes a device (bench's copy
  removed), confirmation through `Terminal.Confirm` only, colour through `Terminal.UseColour`, `CommandContext.Offline`
  and `Seed` instead of per-group readers.
- Tests: the "cli polish" group (help layout and grouping, the environment table, `pull` then `run` from the same
  `--cache` offline, `--offline` naming a missing file, `--timeout` on `ping` and `api`, `--format` on the table
  commands, `--seed`, progress lines).
- Docs: src/Idrak.Cli/README.md as the tool's guide, an "idrak: the command-line tool" section in the main README,
  `idrak doctor` and `idrak setup android` in installation/README.md.

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
