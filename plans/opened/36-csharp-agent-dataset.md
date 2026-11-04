# Plan 36: `idrak data code`, a tool that turns a C# repository into training, tuning and reinforcement data

**Status (2026-11-04):** planned, nothing built. Rewritten on the owner's decisions of 2026-11-04: the tool lives in
Idrak; the samples repository (`ahmedseada/Rl-Samples-`) only stores its output; Craft is out of scope (an app on
Idrak, not part of it); the teacher is any model or none (no Claude Code, no Anthropic client, no provider named in the
library); the reported bugs (phase G-1 below) are fixed first. The earlier text (2026-11-01, a generator outside Idrak
on the Anthropic SDK, with Craft) is superseded; what it reused is kept in "What exists today".

## Goal

One tool, `idrak data code`, that reads any C# repository or project (Idrak itself first) and writes, under
`test-output\`, the data Idrak's trainers already read:

| Output | Rows | Read by |
|---|---|---|
| Pretraining text | one row per source file (`{"text", "path", "language"}`), deduplicated and split | `idrak pretrain`, `idrak tune` (text) |
| RL tasks | task folders (`task.json`, `workspace/`, `verify/`, `solution/`) proven to fail before and pass after | `idrak reinforce --env csharp`, `agent-run`, `agent-check` |
| Fine-tuning transcripts | chat JSONL with `reasoning_content` and `tool_calls` (the agent-demo layout) | `idrak tune` |
| Preference pairs | `{"prompt", "chosen", "rejected"}` from a passing and a failing transcript of one task | `idrak tune` DPO / ORPO / SimPO |
| Explanation rows | question and answer on a contract or registry, checked against `api/*.txt` | `idrak tune` |

The first three come with no model; the last three need one (any `IChatModel`: local, served or scripted). The end use
is Track 3, `idrak new --agent`, a coding agent tuned on this data that writes an Idrak project from a description.

## What exists today (reused, not rebuilt)

| Need | Comes from |
|---|---|
| Reading a repo as rows | Idrak.Data's `code` format (one row per file: `text`, `path`, `language`), `DatasetSpec.Open`, `GitHub.Repository`, `ReadOptions.SkippedFolders` |
| Filter, dedupe, split, mix, validate | `DatasetRecipe`, `idrak data dedupe|split|mix|validate|stats` |
| Task layout and check | `CodingTask` (`task.json`: prompt, language, base, setup, verify, tags, solution), `CodingSuite`, `CodingVerifier` (`CodingTaskCheck`: start fails, solution passes), `CodingRun.ToJson` |
| C# RL environment | `Idrak.Reinforcement.Plugins.CSharp`: env `csharp`, rewards `csharp-tests`, `csharp-build`, protected files, a shared package folder |
| The agent and its tools | `CodingAgent(IChatModel, AgentOptions)`, `CodingTools` (`read_file`, `write_file`, `edit_file`, `list_files`, `search`, `run_command`), `CodingAgentRun.ToJson` (the transcript) |
| A teacher over any model | `TeacherData.Generate(ChatGenerator, rows, options)` with `reasoning_content`; `ChatRows` for chat, text and preference rows |
| The agent commands | `agent`, `agent-run`, `agent-check`, `agent-train` in `samples/Idrak.Samples.CodingAgent/Program.cs` (a sample today; this plan moves `agent-check` and `agent-run` into `idrak`, phase G3) |
| Training | `idrak tune` (LoRA, QLoRA, DoRA, DPO, ORPO, SimPO), `idrak reinforce` (GRPO, PPO, REINFORCE), `idrak distill`, `idrak pretrain` |
| What to convert in Idrak | about 2,170 named test entries (`("group: what it checks", d => ...)`), 25 sample and app projects, `Idrak.Samples.Override`, the plug-in packages, every `*.Abstractions` namespace (the list under "Verification and splits") |

Missing: a task-family contract and registry, the extractor that applies families to a repo, local git history as a
source, a transcript writer that rejects by verification, a suite and run manifest, the `idrak data code` command, the
agent commands in `idrak`, and `idrak new --agent`.

## The shape

### Packages and contracts (abstraction always; nothing family-specific)

| Where | What |
|---|---|
| `Idrak.Data.Abstractions` | `ITaskFamily` (`IKind`): given a `CodeSource` (the repo's files, its history when there is one, the language), yields `TaskDraft`s (prompt, the workspace files to write, the hidden verify files, the setup and verify commands, the solution files, tags). `TaskFamilies`, a `SlotTable` registry; an unregistered name fails with "not registered". `IRepositoryHistory`: commits with message, author date, diff and files (the default runs `git` as a process through the registered command sandbox; an app can register a libgit-backed one). `IRowFilter` for dedupe and quality filters on rows (the library's: exact and near dedupe, size, binary, generated-file detection). |
| `Idrak.Data` | `CodeExtractor`: walks a repo once (rule 73: memory in order; rule 74: bytes, not strings, for detection), applies the chosen families, writes rows and task folders under the output root with a manifest (`manifest.json`: families, counts asked, kept, rejected and why, time). Registers the mechanical families: `file-text` (pretraining rows), `method-body` (a method's body removed, its signature and doc kept; the hidden check is the project's existing tests that reach it), `commit-revert` (a commit's diff reverted, its message the prompt; the check is the tests the commit touched), `test-subject` (the code a named test reaches removed or broken; the check is that test), `doc-to-method` and `test-name-to-body` (weak SFT pairs, tagged `weak`). Which tests a change reaches comes from the test's own `("group: ...")` entry and the files its diff touches, never from running the whole suite. |
| `Idrak.Reinforcement.Plugins.Coding` | `CodingVerifier` proves each draft (`agent-check` semantics: fails on the workspace, passes with the solution). Only a proven draft becomes a task folder. |
| `Idrak.Text` | `TranscriptWriter`: for each task, runs `CodingAgent` over the given `IChatModel` in a fresh workspace copy, verifies with `CodingVerifier`, keeps a passing run as a chat row (`CodingAgentRun.ToJson`), keeps failing runs of a kept task as `rejected` for preference pairs, with `attempts`, `temperature`, `seed` as options. `PromptRewriter`: a model turns a commit message or a test name into a task prompt (kept only when the task still verifies). `ExplainWriter`: questions over `api/*.txt` and the `*.Abstractions` sources, answers checked against the dump (an answer that names a member not in the dump is rejected). All three take any `IChatModel`; a scripted model (as `CSharpCodingPluginTests` uses) makes them testable on CPU. |
| `Idrak.Cli.Commands` | `idrak data code <repo-or-project> -o test-output\code-data [--families a,b] [--model M --adapter A] [--attempts N] [--max-tasks N] [--eval-by group] [--sandbox container]`; `idrak agent-check SUITE`, `idrak agent-run SUITE --model M -o OUT` (moved from the sample). Live progress: family, file or task, kept and rejected counts, rate, time left. |

### Settings (nothing fixed that someone may need to adjust)

Every flag has an option on `CodeExtractorOptions` / `TranscriptOptions` and an `IDRAK_CODE_*` variable: the families,
the skipped folders (the registered languages' build folders by default), the largest file, the dedupe threshold, the
split seed and the eval fraction or group, the model and adapter, attempts per task, temperature and seed, the time
limit per task, the sandbox (`IDRAK_SANDBOX`), concurrency (`ComputeResources`), `--max-tasks` and `--max-time` as
budgets, the output formats (JSON Lines, Parquet). Defaults are measured where they can be (concurrency from the cores,
the largest file from the format's rule); none is the only way.

### Verification and splits

- A task folder exists only after `CodingVerifier` proved it; a transcript only after the hidden tests passed on its
  result; an explanation only after its names were found in the dump. Rejections are counted by reason in the manifest.
- Splits are by group, never by random row: a test group, a contract, a project, a commit's date range. The eval set
  is checked for leakage against the training set (near-duplicate text) before the manifest says "done".
- Idrak's own conversion: light tasks reference Idrak by a local package feed (`dotnet pack` to `test-output\packages`,
  no version bump, nothing published); heavy tasks use the repository at a fixed commit with the real test by name
  (`IDRAK_DEVICES=cpu IDRAK_FILTER=<name>`), built once per commit and shared. GPU-only, bench, window, model-download
  and owner-machine tests are left out with the reason in the manifest. Every `*.Abstractions` namespace found at the
  run's commit must have a kept task or a written reason, or the run fails.

**Every abstraction namespace, none left out** (the owner, 2026-11-01: "every package has its own abstraction
namespace, make sure to include all"). G5 does not use a hand-written list: it finds every namespace ending in
`.Abstraction` or `.Abstractions`, or under `Idrak.Abstraction.*`, at the commit the run reads, and records each public
interface, abstract class and `SlotTable` registry in it. Found at `main` 58e5c2c (files declaring each namespace):

| Repository | Namespaces |
|---|---|
| Idrak, the shared contracts (`Idrak.Abstraction`) | `Idrak.Abstraction` (46), `.Cluster` (17), `.Data` (15), `.Devices` (25), `.Devices.Cpu` (21), `.Diagnostics` (8), `.Formats` (9), `.Generation` (40), `.Modules` (3), `.Operations` (4), `.Plugins` (5), `.Reinforcement` (11), `.Retrieval` (5), `.Serving` (5), `.Text` (2), `.Training` (6), `.Testing` (52), `.Testing.Devices` (4) |
| Idrak, each package's own | `Idrak.Models.Abstractions` (17), `Idrak.Layers.Abstractions` (11), `Idrak.Training.Abstractions` (3), `Idrak.Datasets.Abstractions` (4), `Idrak.Onnx.Abstractions` (3), `Idrak.Gpu.Abstractions` (11), `Idrak.Text.Abstractions` (15), `Idrak.Text.Retrieval.Abstractions` (1), `Idrak.Text.Reinforcement.Abstractions` (1), `Idrak.Data.Abstractions` (7), `Idrak.Vision.Abstractions` (11), `Idrak.Audio.Abstractions` (24), `Idrak.Audio.Plugins.Vits.Abstractions` (1), `Idrak.Cluster.Abstractions` (1), `Idrak.Networking.Abstractions` (6), `Idrak.Mcp.Abstractions` (1), `Idrak.Reinforcement.Abstractions` (2), `Idrak.Reinforcement.Plugins.Coding.Abstractions` (1), `Idrak.Maui.Abstractions` (16) |

Packages with no abstraction namespace of their own are covered as **implementations**: each plug-in package gives
"implement contract X the way this package does, register it, use it" tasks, checked against the contract. `Idrak.Network`
declares 2 public interfaces outside an `.Abstractions` namespace; G5 catalogs them and notes them for the owner.

### Track 3: `idrak new --agent`

`idrak new --agent "<what to build>" -o test-output\gen\api [--model M --adapter A] [--dry-run]`: plans the project
(packages, contracts, registries), starts from the nearest template, then `CodingAgent` writes, builds and tests it
until green or a round limit, with live progress. Templates become `ProjectTemplates`, a `SlotTable` registry (the
four built in; plug-ins add theirs); an unknown template fails with "not registered". Measured on held-out project
descriptions: builds, tests pass, uses the registries rather than hard-coded types.

## Bugs reported by users (2026-11-04)

Reported by the owner's friends using the library in an app and in another session; each checked against `main`
on 2026-11-04 (line numbers of that day). "Read" means the fault is visible in the source as described; "reviewer only"
means the report is plausible from the source but not yet reproduced here. Fixed first, as phase G-1, by Idrak's rules
(the fix workflow, targeted tests, nothing family-specific).

| # | Where | Issue | Checked | Fix shape |
|---|---|---|---|---|
| B1 | `src/Idrak.Cli.Commands/Commands/Serve/ServeOptions.cs:75`, `:196`; `src/Idrak.Cli/Commands/Serve/ServeHost.cs:488` | An empty `--api-key ""` or `IDRAK_API_KEY=` sets `ApiKey` to `""`, so `NeedsKey` is true and the server looks protected, yet an empty bearer compares equal to the empty key and **every request is accepted** | Read | An empty key is a configuration error at startup (fail loudly, name the flag and the variable), and `Authorize` never accepts an empty `given` |
| B2 | `src/Idrak.Text/Generation/CodingTools.cs:753-773` (`WriteFile`), `:822-833` (`ExecuteAsync`), `:837-843` (`Resolve`) | The agent's `write_file` and `edit_file` never ask; the allowed programs are every registered coding language's (`CodingLanguages`, so `node`, `npx`, `dotnet`, `git`), which run any code; `Resolve` checks the textual path only, so a symlink inside the workspace reaches outside it; the default sandbox is `process` (a plain process) | Reviewer only (the allow list, the path check and the default sandbox are read; the symlink escape not reproduced) | `Resolve` resolves links (`ResolveLinkTarget` / `FileSystemInfo.LinkTarget`) and refuses a target outside the root; the allowed programs become an option with a conservative default per language and an `IDRAK_*` variable; the sandbox default stays a setting, documented as unsafe for untrusted models; an optional confirmation hook on writes (a contract in `Idrak.Text.Abstractions`) for apps that want to ask |
| B3 | `src/Idrak.Cli/Commands/Serve/ServeHost.cs:185-199`, `:246-250` | `serve` has no Origin check: `/idrak/stop`, `/idrak/load`, `/idrak/unload` are plain POSTs, so any web page a user opens can call them on a local server (CORS only governs what the page may read, not what reaches the server; without `--cors` a simple cross-site POST still lands) | Reviewer only (the endpoints and the CORS setup are read; a cross-site POST not reproduced) | Control endpoints require a key always, or an `Origin`/`Sec-Fetch-Site` check that refuses a browser origin not in `--cors`; a setting for the policy |
| B4 | `src/Idrak.Gpu/Cuda/CudaBackend.cs:191-197` (`TensorCoreKernels`), `:226-232`, `:2092` (`LaunchGrid`) | First-use race: `_tensorCoreTried` is check-then-set without a lock when reached through `TensorCoreFunction` (only `TensorCoresUnavailable` takes `lock (_signatures)`), two threads can load the module twice; `_signatures` is written under that lock but read in `LaunchGrid` without it, so a launch during the first tensor-core load reads a dictionary being resized | Read | One lock (or `Lazy<T>`) around the tensor-core load for every entry point; `_signatures` becomes a frozen or copy-on-write map once a module is loaded, so launches read without a lock |
| B5 | `src/Idrak.Abstraction/Tensors/TensorScope.cs:116-119` | `Dispose` sets `_disposed = true` before the ordering check throws, so after the throw the scope is dead but still `t_current`: the thread stays on it, and later tensors are added to a disposed scope and leak | Read | Check the order first, set `_disposed` only on the path that really disposes; a test that disposes out of order and asserts the current scope and a later tensor's owner |
| B6 | `src/Idrak.Text/Jinja.cs:1087` vs `:1107` | `caller()` is set on the call block's scope (`callScope`), but a macro's body runs in a scope chained to its definition scope (`MakeMacro`), so `caller` is undefined inside the macro: `{% call %}` blocks in chat templates fail | Read | Pass `caller` into the macro invocation (a per-call binding in `MakeMacro`'s local scope, as Jinja does), with a template test using `{% call m() %}...{% endcall %}` |
| B7 | `src/Idrak.Abstraction/Tensors/Tensor.cs:620`, `:660` | `Backward` clears each node's `_backward` after use (`:660`), so a second `Backward` through a reused activation skips those nodes (`:620`) and silently drops their gradients; PyTorch throws ("Trying to backward through the graph a second time") | Reviewer only (the clearing and the skip are read; the drop not reproduced) | A node whose `_backward` was consumed but whose inputs still require grad throws with the cause named, unless a `retain_graph`-style option is set; a test with two backward passes through one activation |

### The full risk review (2026-11-04)

The same friends' full review (70 risks, 7 areas, 17 re-read by its reviewer; `main` 4374037, read only, nothing built
or run) was received on 2026-11-04. Its first seven items are B1-B7 above: S1 = B1; S2, S3, S4, S7 = B2; S8 = B3;
G2, G3 = B4; C5 = B5; L1 = B6; C4 = B7. The rest are listed here with the review's ID and severity, its check
("verified": its reviewer re-read the code; "reported": one reviewer's read), mine where I re-read the source on
2026-11-04, and where each goes: **G-1** (a code fix with a targeted test, in this plan), **decision** (a design or
process choice that is the owner's, not taken by this plan; where it conflicts with a standing rule in `CLAUDE.md`, the
rule holds until the owner changes it), or **noted** (a scope limit or a known gap, recorded, nothing to do here).

**Security (S5-S14)**

| ID | Sev. | Risk | Where | Check | Goes to |
|---|---|---|---|---|---|
| S5 | Medium | `-y` approves every command; tools loaded with `--tools DLL` are added after the approval rule, so they skip it | `src/Idrak.Cli.Commands/Commands/Run/AgentCommand.cs:102-121` | reported; mine: read, both hold | G-1: extra tools go through the same approval rule; `--yes` stays a setting, documented |
| S6 | Medium | Some "read-only" git subcommands can write or start programs through options (`git log --output`, `git diff --ext-diff`, pagers, `-c` settings) | `src/Idrak.Text/Generation/CodingTools.cs:37`, `:827-830` | reported (per git's docs, not tested); mine: the list read | G-1: refuse option forms that write or spawn, run git with `--no-pager` and `-c core.pager=` and `GIT_EXTERNAL_DIFF` cleared, as a setting |
| S9 | Medium | `serve -H 0.0.0.0` with no key starts without a warning; MCP and the dashboard refuse the same setup | `src/Idrak.Cli/Commands/Serve/ServeHost.cs:181` | reported; mine: no warning in the source | G-1: warn (or refuse, a setting) on a public bind with no key, as MCP does |
| S10 | Medium | `--allow-image-urls` fetches any URL, loopback and link-local included (SSRF), and echoes the remote reply in errors | `src/Idrak.AspNetCore/ImageInputs.cs:76-100` | reported | G-1: resolve and refuse loopback, link-local and private ranges unless a setting allows them; errors name the URL, not the reply |
| S11 | Medium | Chat-template rendering has no step or output limit: a hostile template in a downloaded model can hang the process or exhaust memory | `src/Idrak.Text/Jinja.cs` | reported | G-1: a step budget and an output cap in the renderer, settings with defaults, a loud failure naming the template |
| S12 | Low | The coding-agent sample runs model commands with no approval | `samples/Idrak.Samples.CodingAgent/Program.cs:337` | reported | G-1: the sample asks, or says in its first line that it does not |
| S13 | Low | Hostile GGUF: the dimension count is not checked before the shape array is allocated; duplicate tensor names overwrite silently (denial of service only) | `Gguf.cs:583` | reported; mine: read, `new long[dims]` before any bound | G-1: bound `dims` by the format's maximum, refuse duplicates, by the allocation-failure rule (88-90) |
| S14 | Low | Cached HIP code objects load with no content hash | `HipKernelCache.cs` | reported | G-1: hash the cached object as the CUDA cache does, or note why not |

**CI and tests (T1-T11)**

| ID | Sev. | Risk | Where | Check | Goes to |
|---|---|---|---|---|---|
| T1 | High | CI red on `main` since 2026-10-26 (25 of 192 runs green): the build passes, the CPU test step fails | `ci.yml` | verified | decision: the owner finds the failing group (the owner's run) and whether a green CI gates merges |
| T2 | High | 85K lines of GPU code never run in CI; GPU tests are the owner's by hand | `ci.yml` | verified | decision: a self-hosted GPU runner (the tests rule in `CLAUDE.md` holds) |
| T3 | Medium | Missing hardware counts as PASS: the Vulkan tests print "skipped" and return, though the runner has a SKIP state | `tests/Idrak.Tests.Library/VulkanTests.cs:101-108` | verified | G-1: report SKIP through the runner's state |
| T4 | Medium | About 170 tests guarded by a CPU-only check pass with no work on other devices | `tests/` | reported | G-1 with T3: a guarded test on another device is a SKIP, never a PASS |
| T5 | Medium | `Idrak.Maui` (30K lines) and its tests are not in the solution and never run in CI | `Idrak.slnx` | verified; mine: no `Maui` entry in the solution | decision: add it to the solution (needs the MAUI workloads on CI) or record it as built apart |
| T6 | Medium | Most model families are tested on tiny random-weight models; real-checkpoint checks exist for T5 and BERT only, owner-run | `T5Tests.cs:991`, `BertTests.cs:599` | reported | noted: the owner's runs; the generator's eval (G5) adds real runs on the tuned model |
| T7 | Low | "Bidi passes Unicode's suites" has no committed test (48 paragraphs against python-bidi are) | `Bidi.cs:88` | reported | G-1: commit the suite run as a filtered test, or soften the claim (README row below) |
| T8 | Low | The MP3 encoder's fixtures come from itself, so its tests catch regressions, not wrong output | `AudioMp3EncodeTests.cs:19` | reported | noted: an external fixture (a reference encoder's output) when the owner has one |
| T9 | Low | About 65 timing-based assertions, mostly cluster tests, may flake on busy runners | `tests/` (cluster groups) | reported | noted |
| T10 | Low | The custom runner is single-threaded, no IDE or JUnit output, one hang ends the run; shared static registries can make tests order-dependent | `TestRunner.cs:174-401` | reported | decision: the runner is the owner's design |
| T11 | Low | Finite-difference gradient checks are loose (step 1e-2, tolerance 2e-2); PyTorch fixtures at 1e-4 make up for it | `BasicTests.cs:470-505` | reported | noted |

**GPU backends (G1, G4-G11)**

| ID | Sev. | Risk | Where | Check | Goes to |
|---|---|---|---|---|---|
| G1 | High | HIP has never run its kernels on a real AMD GPU; every GCN kernel is proven in the C# interpreter only | `plans/opened/35-gcn.md` | reported | noted: plan 35's own status; the owner's hardware |
| G4 | Medium | Per-thread launch settings (shared memory, cooperative) leak into the next launch when the argument check throws before they are reset | `src/Idrak.Gpu/Cuda/CudaBackend.cs:2092-2098` | verified; mine: read, the throw precedes the reset | G-1 with B4: reset in `finally` or before the check |
| G5 | Medium | AMD has no matrix-core kernels (MFMA, WMMA); gfx90a, MI300 and RDNA4 fall back to the CPU; RDNA is wave64 only | `GcnTargets.Gfx9.cs:31` | reported | noted: plan 35's scope |
| G6 | Low | GPU resources are never released: no module unload, stream destroy or context release; each backend leaks a stream and its modules | `CudaBackend.cs:272` | reported | G-1: dispose paths, with the device-memory rule (76) test |
| G7 | Low | Three tuning implementations of uneven rigour: Vulkan times without pairing, "off" accepted for CUDA only | `VulkanBackend.KernelTuning.cs:316`, `TuningCache.cs:233` | reported | G-1: one tuning contract's behaviour across backends (paired timing, "off" everywhere) |
| G8 | Low | Naive fallbacks remain: a 16×16 matmul, Vulkan's batched tile, a Vulkan sum in one workgroup (TODO) | `VulkanKernels.cs:489` | reported | noted: speed-first rule, measured when the owner runs the benches |
| G9 | Low | PTX built from string templates with hand-numbered registers and no IR: mistakes show only when the driver compiles | `PtxKernels.cs:757` | reported | decision |
| G10 | Low | AMD architecture found by scanning a 16 KiB buffer for "gfx"; tuning-cache merges across processes can lose entries | `HipDeviceLimits.cs:186` | reported | G-1: the cache merge; the detection stays until the owner has the hardware |
| G11 | Low | The CUDA allocator weaves capture, chunks, offload and lookups together, hard to change safely | `CudaBackend.cs:497-620` | reported | noted |

**Core: tensors, autograd, training, loading (C1-C3, C6-C14)**

| ID | Sev. | Risk | Where | Check | Goes to |
|---|---|---|---|---|---|
| C1 | High | Tensors are float32 only: int8, int4 and bf16 are bit-packed into float storage, token indices travel as floats (exact below 2^24) | `Backend.cs:1540`, `AdamW8Bit.cs` | reported | decision: dtypes are a library-wide design choice |
| C2 | High | No strides or offsets: Permute, Transpose and Narrow copy; MatMul broadcasting is limited | `Tensor.Advanced.cs:318-366` | verified | decision, with C1 |
| C3 | Medium | Sizes are `int`: a tensor tops out at 2^31 elements and GPU index math is 32-bit | `Tensor.cs` | reported | decision, with C1 |
| C6 | Low | `MatMulBias` reads the weight's second dimension before its rank check: a rank-1 weight fails with an index error | `src/Idrak.Abstraction/Tensors/Tensor.Advanced.cs:69` | reported; mine: read, `w._shape[1]` precedes `w.Rank != 2` | G-1: check the rank first, with a test |
| C7 | Low | Some backward behaviour is chosen by matching operation names as strings | `Tensor.cs` ~450 | reported | noted |
| C8 | Medium | The generic Trainer has no out-of-memory recovery and its checkpoint saves weights only; both live in text fine-tuning or the cluster-stop path | `Trainer.cs:626`, `TrainerCallbacks.cs:120` | reported | decision: move both into the generic trainer (README row below) |
| C9 | Low | Cluster-simulation code (faults, late joins, replays) is woven into the core training loop | `Trainer.cs:380-470` | reported | noted |
| C10 | Low | Mixed precision is a switch for matrix products only, not a full autocast | `MixedPrecision.cs` | reported | noted, with C1 |
| C11 | Low | Adam uses the epsilon-hat form, so results differ slightly from PyTorch's | `CpuBackend.cs:1067` | reported | decision: match PyTorch or document the difference |
| C12 | Low | The optimizer step count is saved as a float, exact to about 16.7M steps | `Optimizer.cs:183` | reported | G-1: save it as an integer, read both |
| C13 | Low | The CPU matrix product never splits k: deterministic, likely slower than BLAS at large k (a guess) | `CpuTuning.cs:151` | reported | noted: measured when the owner runs the benches (rule 71) |
| C14 | Low | GGUF reads 15 quantization types (no IQ1-IQ3, TQ, MXFP4); safetensors has no FP8 or int8 | `Gguf.cs:188`, `SafeTensors.cs:525` | reported | noted (README row below) |

**LLM stack and serving (L2-L7)**

| ID | Sev. | Risk | Where | Check | Goes to |
|---|---|---|---|---|---|
| L2 | Low | Jinja gaps: `recursive` accepted then ignored; no `loop.cycle`, `include`, `import` or `with` | `Jinja.cs:425` | reported | G-1 with B6: refuse `recursive` loudly rather than ignore it; the rest noted |
| L3 | Medium | The OpenAI API is partial: `n>1` refused, `logprobs` always null, `response_format` and `json_schema` silently ignored, an object `tool_choice` likely fails | `CompletionsApi.cs:717`, `:757` | reported | G-1: a silently ignored field is refused with its name (nothing silent); the rest decision (README row below) |
| L4 | Medium | MoE waits for the host in every layer, refuses graph capture, gathers with float indices | `MixtureOfExperts.cs:155` | reported | noted: speed-first; with C1 for the indices |
| L5 | Low | "QLoRA" here is a frozen int4 or int8 base, not NF4 with double quantization | `FineTuning.cs:1070` | reported | decision: implement NF4 or rename |
| L6 | Low | The command allow-list check lives in two places that can drift | `CodingTools.cs:822`, `ProcessCommandSandbox.cs:194` | reported; mine: read, both exist | G-1 with B2: one check, the sandbox's, called by the tools |
| L7 | Low | Very large files: `FineTuning.cs` 3,220 lines, `TuneTool.cs` 1,610, `Jinja.cs` 1,250 | those files | verified | noted |

**Media stacks (M1-M5)**: no hinting, variable fonts (CFF2 refused) or Indic shaper (M1); the first alternate always
taken in OpenType alternate substitution (M2); MP3 constant bit rate and long blocks only (M3); FLAC MD5 not checked
(M4); JPEG 12-bit, arithmetic, lossless and hierarchical refused (M5). All reported, all scope limits: noted; M2 and M4
are small G-1 items if the owner wants them (a selector option for alternates; the MD5 check as a `ReadOptions` setting).

**Process and scale (P1-P8)**

| ID | Sev. | Risk | Where | Check | Goes to |
|---|---|---|---|---|---|
| P1 | High | 594K lines in 5 weeks, 96% of 1,444 commits by Claude, one owner reviewing: review is the bottleneck | `git log` | verified | decision |
| P2 | High | Over-abstraction: 5,045 public members in `Idrak.Abstraction`, 145 of 351 interfaces with one implementation, 370 public members on `Backend` | `api/`, the inventory | reported | decision: conflicts with the standing rule "Abstraction always"; the rule holds until the owner changes it |
| P3 | Medium | 303 `IDRAK_*` variables and 42 per-thread static fields make behaviour hard to reason about and test | `src/` | reported | decision: conflicts with "Nothing fixed that someone may need to adjust"; the rule holds |
| P4 | Medium | Comments carry hundreds of plan and rule references ("plan 35", "rule 86", "G7a"), some already stale | `CudaBackend.cs:34`, `TuningCache.cs:216` | reported | decision |
| P5 | Medium | A snapshot that did not compile was committed to `main` | commit 922fcc19 | verified | noted; with T1 |
| P6 | Low | Only the owner runs the full suite or any GPU test, so no agent sees the whole picture | `CLAUDE.md` | verified | decision: the owner's rule; it holds |
| P7 | Low | 8 empty "reserved" packages are packed and published with every release | `src/` | verified; mine: 8 READMEs say "Reserved, not implemented" | decision: keep publishing them or mark them not packable |
| P8 | Low | 17 leftover `slot/*` branches on origin | `git branch -r` | verified; mine: 17 on origin | decision: the owner deletes them (a push to origin is the owner's) |

**README claims the review calls overstated** (for the owner: fix the code or the sentence): strides, slices and views
(C2); every ggml quantization (C14); Trainer OOM recovery and bit-for-bit resume (C8); an OpenAI-compatible `/v1` API
(L3); a coding agent with a sandbox (B2, S4); bidi passing Unicode's official tests (T7); the test count (824 in the
README, 1,563 in a plan, about 2,170 named checks counted by the review).

### Phase G-1, planned (2026-11-04): the bugs first

The owner (2026-11-04): "plan the bugs first." G-1 runs before G0; nothing of the dataset tool starts until G-1 is
merged. Rules that hold: the fix workflow in `CLAUDE.md` (every error of a piece fixed, then its tests updated, one
build of the changed projects, one filtered CPU run with `--no-build`, merge and push to `main`); at most two agents at a
time; each piece is one agent's, with its own filter, so two pieces never touch the same file; a slot takes the next
piece whose inputs are merged. Every behaviour a fix adds is a setting (an option, and an `IDRAK_*` variable where the
library already reads one); every failure message names the cause; nothing family- or card-specific.

**Order.** Pieces are ordered by what the dataset tool and the trained agent will run on, then by severity, then by how
close two pieces' files are (so one agent takes a file's bugs together):

| Wave | Piece | Bugs | Files | The fix | Test (group, `IDRAK_FILTER`) |
|---|---|---|---|---|---|
| 1 | **1a: the coding tools' boundary** | B2 (S2, S3, S4, S7), S6, L6 | `src/Idrak.Text/Generation/CodingTools.cs` (`Resolve`, `ExecuteAsync`, `Files`), `src/Idrak.Text/Generation/ProcessCommandSandbox.cs`, `src/Idrak.Text/Abstractions` (a write-confirmation contract), `src/Idrak.Text.Plugins/CSharp/CSharpLanguagePlugin.cs`, `src/Idrak.Text.Plugins/TypeScript/TypeScriptLanguagePlugin.cs` | `Resolve` resolves links (`FileSystemInfo.ResolveLinkTarget(true)` on the deepest existing part) and refuses a target outside the root; the folder walk refuses to follow a link that leaves the root and keeps a visited set. One allow-list check: the sandbox's (`ProcessCommandSandbox:194`), called by `ExecuteAsync`; the tools' copy removed. Git: `--no-pager`, `-c core.pager=`, `GIT_EXTERNAL_DIFF`, `GIT_PAGER` and `PAGER` cleared, option forms that write or spawn (`--output`, `--ext-diff`, `-c`, `--exec-path`, `!` aliases) refused by name. The allowed programs become `CodingToolOptions.Commands` with a conservative default per language (build and test runners; `node`, `npx`, `ng` only when the option names them) and `IDRAK_AGENT_COMMANDS`. `write_file` and `edit_file` go through a write policy contract (`IWritePolicy` in `Idrak.Text.Abstractions`, a `SlotTable` registry, default "allow", `ask` registered by the CLI). The sandbox default stays `process` as a setting, with the option's doc and the first line of `idrak agent` saying it is a plain process and how to pick `container` | `coding tools:`, `command sandboxes:` (`IDRAK_FILTER="coding tools|command sandboxes"`): a symlink out of the workspace refused for read, write, edit and list; a git `--output` refused; the allow list refused in one place; the write policy's `ask` called once per write |
| 1 | **1b: `idrak serve`'s doors** | B1 (S1), B3 (S8), S9, L3's silent fields | `src/Idrak.Cli.Commands/Commands/Serve/ServeOptions.cs` (`:196`, `NeedsKey`), `src/Idrak.Cli/Commands/Serve/ServeHost.cs` (`Authorize`, the control endpoints, the bind), `CompletionsApi.cs` (`ReadBody`, the ignored fields) | An empty `--api-key` or `IDRAK_API_KEY` fails at startup naming the flag and the variable; `Authorize` never accepts an empty `given`. Control endpoints (`/idrak/load`, `/idrak/unload`, `/idrak/stop`) need a key whenever one is set, and refuse a request whose `Origin` or `Sec-Fetch-Site` shows a browser origin not in `--cors` (a setting, `--control-origins`, default: none); a JSON body needs a JSON `Content-Type`. A public bind (`0.0.0.0`, `*`, `::`) with no key warns in the first line, or refuses under `--require-key` (`IDRAK_SERVE_REQUIRE_KEY`). A request field the API does not implement (`response_format`, `json_schema`, `logprobs`, an object `tool_choice`) is refused with its name, never ignored | `cli serve:` (`IDRAK_FILTER="cli serve"`): empty key refused at startup; `Bearer` with an empty value is 401; a cross-origin POST to `/idrak/stop` is 403 and the server stays up; a public bind without a key prints the warning; `response_format` answers 400 naming the field |
| 2 | **2a: device and scope state** | B4 (G2, G3), G4, B5 (C5), G6 | `src/Idrak.Gpu/Cuda/CudaBackend.cs` (`TensorCoreKernels`, `_signatures`, `LaunchGrid`, disposal), `src/Idrak.Abstraction/Tensors/TensorScope.cs` | Tensor-core loading under one lock for every entry (`TensorCoreFunction`, `TensorCoresUnavailable`), the tried flag set after the load ends; `_signatures` replaced by a frozen map rebuilt on each module load (reads lock-free, writes under the lock); `LaunchGrid` resets `t_sharedBytes` and `t_cooperative` before the argument check (or in `finally`). `TensorScope.Dispose` checks the order first and sets `_disposed` only when it disposes. Disposal of a backend unloads its modules, destroys its stream and releases its context, with the device-memory rule's check (rule 76) | A new `tensor scope:` group (`IDRAK_FILTER="tensor scope"`): dispose out of order throws, the current scope is unchanged and a later tensor lands in the live scope. The CUDA pieces are the owner's GPU run (`IDRAK_DEVICES=cuda IDRAK_FILTER="cuda launch|tensor cores"`, new groups): two threads' first use loads once and both launch; a wrong argument count throws and the next launch has no leaked shared bytes; device bytes after dispose equal the bytes before load |
| 2 | **2b: templates and the agent command** | B6 (L1), S11, L2's `recursive`, S5, S12 | `src/Idrak.Text/Jinja.cs` (`CallBlockNode`, `MakeMacro`, the executor), `src/Idrak.Cli.Commands/Commands/Run/AgentCommand.cs:102-121`, `samples/Idrak.Samples.CodingAgent/Program.cs:337` | `caller` bound in the macro's local scope per invocation; a step budget (`JinjaOptions.MaxSteps`, `IDRAK_TEMPLATE_MAX_STEPS`) and an output cap (`MaxOutputChars`) that fail naming the template; `recursive` refused with its name rather than ignored. The approval rule applies to every tool, extra assemblies included (`--tools` added before `RequireApproval`), and `--yes` stays a setting documented in the first line. The sample asks before a command, or prints in its first line that it does not (its own `--yes`) | `chat template:` (`IDRAK_FILTER="chat template"`): `{% call m() %}...{% endcall %}` renders; a looping template stops at the budget with the template's name in the message; `recursive` is refused. `cli run:` (`IDRAK_FILTER="cli run"`): an extra tool is asked for without `--yes`. The sample's own test project with its filter |
| 3 | **3a: inputs that can be hostile** | S10, S13, S14, C6 | `src/Idrak.AspNetCore/ImageInputs.cs:76-100`, `Gguf.cs:583`, `src/Idrak.Gpu/Hip/HipKernelCache.cs`, `src/Idrak.Abstraction/Tensors/Tensor.Advanced.cs:69` | Image URLs resolved first and refused for loopback, link-local, private and multicast ranges unless `--allow-private-image-urls` (`IDRAK_SERVE_PRIVATE_IMAGE_URLS`); an error names the URL and the status, never the body. GGUF: `dims` bounded by the format's maximum before the array exists, a duplicate name refused with the name (rules 88-90: the failure diagnosed by its type). HIP: the cached code object stored with its content hash and refused when it differs. `MatMulBias` checks the rank before reading a dimension | `cli images:` (`IDRAK_FILTER="cli images"`): a loopback image URL is refused with the URL in the message. `cli models:` (`IDRAK_FILTER="cli models"`): a GGUF with a huge dimension count and one with a duplicate name both fail naming the cause. A new `hip cache:` test (CPU, the cache is files). `MatMulBias` with a rank-1 weight throws the rank message, in the tensor group that holds its other tests |
| 3 | **3b: a second backward, the runner's SKIP, the step count** | B7 (C4), T3, T4, C12, G7, G10 | `src/Idrak.Abstraction/Tensors/Tensor.cs:620,660`, `tests/Idrak.Tests.Library/VulkanTests.cs:101-108`, `tests/Idrak.Tests.Library/TestRunner.cs`, `Optimizer.cs:183`, `VulkanBackend.KernelTuning.cs:316`, `TuningCache.cs:233`, `HipDeviceLimits.cs:186` | A node whose step was consumed but whose inputs still require grad throws naming the tensor, unless `Backward(retainGraph: true)`. Missing hardware throws `SkippedException` (the runner's SKIP), never prints and returns; the CPU-only guard in the ~170 device-parameterised tests becomes the same SKIP. The optimizer step saved as an integer, read as either. Tuning: paired timing in Vulkan, "off" accepted by every backend, the cross-process cache merge keeping every entry | A new `autograd:` group (`IDRAK_FILTER="autograd"`): two backward passes through one activation throw; `retainGraph` keeps both. `vulkan:` on CPU (`IDRAK_FILTER="vulkan"`) reports SKIP counts, not PASS. `cluster:` for the step count (`IDRAK_FILTER="cluster"`, the state tests only, by name). The tuning pieces are the owner's GPU runs |
| 4 | **4: the README's seven claims and the small media items** | T7, M2, M4, the README rows | `README.md`, `Bidi.cs:88` and its test, `LayoutEngine.cs:388`, `FlacCodec.cs:28-33` | Each README claim either made true by a piece above (the sandbox, the API's refusals) or rewritten to what the code does (strides, quantizations, the trainer, the test count from the runner's own count). Bidi: the official suite committed as a filtered test (`bidi:`), or the claim softened. Alternates: a selector option (`AlternateIndex`); FLAC: `ReadOptions.VerifyMd5` | `bidi:` (`IDRAK_FILTER="bidi"`), `audio cli:` for FLAC, the text-layout group for alternates |

**Not in G-1** (decisions in the table above, left to the owner): C1, C2, C3 (dtypes, strides, sizes), T1, T2, T5
(CI, the GPU runner, Maui in the solution), C8, C11, L5 (the trainer's recovery, Adam's form, NF4), P1-P8 (process),
G1, G5, G9 (AMD hardware and the PTX builder). The owner says which, if any, join a later phase.

**Done when** every piece's filtered CPU run passes with `--no-build`, every failure message in those tests names the
cause, the public API dump (`api/Idrak.txt`) and the inventory test pass for the new contract and options, and the owner
has run the two GPU filters of 2a and the tuning pieces of 3b.

**The owner's command list, after G-1 is merged** (one pull, one build, then every run with `--no-build`):

```powershell
cd D:\Projects\Idrak; git checkout main; git pull origin main
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES="cpu"; $env:IDRAK_FILTER="coding tools|command sandboxes|cli serve|tensor scope|chat template|cli run|cli images|cli models|hip cache|autograd|vulkan|bidi"; dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-g1-cpu.txt
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="cuda launch|tensor cores|tuning"; dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\fails-g1-gpu.txt
```

G-1 is done when each row has a fix with a targeted test (`IDRAK_DEVICES=cpu` with the group's `IDRAK_FILTER`), the
failure messages name the cause, and the owner has the GPU run command for B4.

## Phases

At most two agents at a time; a slot takes the next piece whose inputs are merged. Each piece has its own test group and
filter, CPU only; the model runs are the owner's.

| Phase | What | Done when | Needs | Test (`IDRAK_FILTER`) |
|---|---|---|---|---|
| G-1 | The reported bugs (the section above), four waves | Every piece's filtered run passes; the owner has the GPU commands | - | per piece, above |
| G0 | Contracts: `ITaskFamily`, `TaskFamilies`, `IRepositoryHistory`, `IRowFilter` in `Idrak.Data.Abstractions`; `CodeSource`, `TaskDraft`; the API dump regenerated; the inventory test | The registries exist with the "not registered" failure; `api/Idrak.txt` and the inventory pass | G-1 | `data code contracts` (new group) |
| G1 | `CodeExtractor` with `file-text`, `method-body`, `test-subject`; the manifest; the `idrak data code` command without a model | On Idrak itself: rows for every source file, proven task folders for a chosen test group, a manifest with reasons; a run on a small fixture repo in the tests | G0 | `data code` (new group) |
| G2 | `commit-revert` on `IRepositoryHistory` (the git process default), the weak pair families, dedupe and quality filters, group splits and the leakage check | A fixture repo with three commits gives three proven tasks; the eval split shares no near-duplicate with train | G0 (parallel with G1, other files) | `data code history`, `data code splits` |
| G3 | `agent-check` and `agent-run` as `idrak` commands (from the sample, which then calls them); `CodingVerifier` as the proof in G1's pipeline | The sample's three demo tasks check and run through `idrak`; the sample test project passes with its filter | G1 | `cli agent` (new group), the sample's filter |
| G4 | `TranscriptWriter`, `PromptRewriter`, `ExplainWriter` in Idrak.Text on any `IChatModel`; `--model` on `idrak data code`; preference pairs | With a scripted model: a passing run becomes a chat row, a failing one a `rejected` row, a pair is written; an explanation naming an unknown member is rejected | G3 | `data code transcripts` (new group) |
| G5 | Idrak's own conversion at volume: every convertible test entry, every `*.Abstractions` namespace, every sample, app and plug-in; the namespace coverage check; the manifest's report under `test-output\` | The run on `main` ends with every namespace covered or explained; counts in the report | G2, G4 | the owner's run: `idrak data code . -o test-output\idrak-data` |
| G6 | The owner's model runs: a local model as the teacher (`--model`), then `idrak tune` on the transcripts, `idrak reinforce --env csharp` on the task sets, `agent-run` on the eval before and after; the output copied to the samples repository | The tuned adapter beats the base on the held-out eval | G5 | the owner's commands, given with G5 |
| G7 | Track 3: `ProjectTemplates`, `idrak new --agent`, its tests on a scripted model | CPU tests pass; the owner's run on the tuned adapter writes projects that build | G3 for the command; G6 for the model | `cli new agent` (new group) |

## Decisions for the owner

1. **Taken (2026-11-04):** the tool lives in Idrak; the samples repository stores its output; Craft is out; any model
   or none; the bugs first.
2. **The local teacher model** for G6 and the base G6 tunes: the owner's choice at run time (`--model`, `--adapter`); the
   benches' coding model is the recommendation, nothing in code.
3. **Volume:** `--max-tasks` and `--max-time` bound a run; the first run on Idrak takes every convertible entry once
   with up to three transcript attempts each; the owner raises it after the manifest shows the kept rate.
4. **The git default:** `IRepositoryHistory`'s default runs `git` through the command sandbox (a process); whether a
   pure-C# reader of the pack files is wanted later is open.
5. **Storage in the samples repository:** plain files or Git LFS for large outputs, open.

## Rules that hold for every phase

- `CLAUDE.md` and `docs/optimization.md` hold for every line (spans over strings on the walk, bytes parsed not strings,
  pooled buffers returned in `finally`, work done once, memory that does not grow, the library's telemetry, independent
  work spread over the cores).
- No family-specific code in Idrak: the tool takes any model; no provider, no Claude Code, no Anthropic client.
- Tests: targeted only (`IDRAK_DEVICES=cpu IDRAK_FILTER=<group>`); model, GPU and training runs are the owner's, given as
  exact PowerShell commands, one list per change.
- Nothing runs or is written outside a task's sandbox and the output folder under `test-output\`.
