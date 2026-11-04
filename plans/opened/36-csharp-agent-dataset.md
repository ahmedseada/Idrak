# Plan 36: a C# agent dataset made with Claude, and a project generator on it

**Status (2026-11-04):** planned, nothing built. Decisions taken by the owner on 2026-11-04 (they override the
text below where it differs; the body is kept as written until the owner asks for the rewrite):

- **The tool lives in Idrak**, not in the samples repository: the no-model extractor in Idrak.Data (a task-family
  contract and `SlotTable` registry in `Idrak.Data.Abstractions`, a data command that writes under `test-output\`),
  the model-driven transcript step in Idrak.Text beside `CodingAgent` and `TeacherData`. `ahmedseada/Rl-Samples-` only
  stores what the tool writes out.
- **Craft is out of scope.** It was an app on Idrak, not part of it: no Craft clone, plug-ins, tests or namespaces in
  any track; the namespace coverage check covers Idrak's own abstraction namespaces only.
- **Any model, or none.** The teacher is any `IChatModel` (local or served) or no model at all for the mechanical
  families (every file as a pretraining row, a test's subject removed, a commit reverted with its message as the
  prompt, a method body dropped). No Claude Code, no Anthropic client, no provider named in the library; the
  `RLS_*` settings become library options and `IDRAK_*` variables.
- **Bugs reported by users are part of this plan** (the section "Bugs reported by users" below, phase G-1): they are in
  the paths the tool and the trained agent run on (`CodingTools`, `serve`, the device and autograd core).

**Status (2026-11-01):** planned, nothing built. Written from the owner's request (2026-11-01): "using claude api key and
sdk ... create as many samples as possible ... 1. samples for fine-tuning a C# model, agent; 2. convert all test cases to
reusable RL and fine-tune samples from Idrak and Craft, also the samples and apps folders and plug-in projects, so the
models learn how to use Idrak's abstractions to turn them into projects and how to provide plug-ins and non-default
implementations; 3. an Idrak C# project generator using the Idrak CLI agent trained on that dataset. Plan first."
Every "comes from" below was found in the source at Idrak `main` 58e5c2c and Craft `main` 6dbb736.

## Goal

Three outcomes, in this order:

1. **Track 1, a general C# coding-agent dataset**: thousands of verified C# tasks and agent transcripts, in the formats
   Idrak already reads, for `idrak tune` (SFT on LoRA) and `idrak reinforce --env csharp` (RL on hidden tests).
2. **Track 2, an Idrak-and-Craft dataset**: every test, sample, app and plug-in project of both repositories turned into
   reusable tasks, so a model learns Idrak's contracts, `SlotTable` registries, plug-ins and non-default
   implementations, and how to turn them into a project.
3. **Track 3, a project generator**: `idrak new --agent "<what to build>"`, the CLI's coding agent (on the model tuned
   on tracks 1 and 2) writing a whole Idrak project from a description, built and tested before it is handed over.

Claude (through the Anthropic SDK) is the teacher that writes tasks and transcripts; Idrak's own tools check them,
train on them and run the result. Nothing in Idrak's library depends on Claude.

## What exists today (reused, not rebuilt)

| Need | Comes from |
|---|---|
| Agent tool set | `Idrak.Text/Generation/CodingTools.cs`: `read_file`, `write_file`, `edit_file`, `list_files`, `search`, `run_command` |
| Agent loop | Idrak.Text's `CodingAgent` (`idrak agent`, `samples/Idrak.Samples.CodingAgent`) |
| SFT row layout | `samples/Idrak.Samples.CodingAgent/data/agent-demo-*.jsonl`: `messages` with `reasoning_content` and `tool_calls` (220 rows, Python) |
| RL task layout | `task.json` (`prompt`, `language`, `base`, `setup`, `verify`, `tags`) with `workspace/`, `verify/tests/`, `solution/` |
| Task checking | `agent-check <suite>`: verification fails on the starting files and passes with `solution/` |
| C# RL environment | `Idrak.Reinforcement.Plugins.CSharp`: env `csharp`, rewards `csharp-tests`, `csharp-build`; protected files; shared package folder |
| Training | `idrak tune` (LoRA, QLoRA, DoRA, DPO/ORPO/SimPO), `idrak reinforce` (GRPO, PPO, REINFORCE), `idrak distill` |
| Data tools | `idrak data validate`, `dedupe`, `stats`, `split`, `mix` |
| Project templates | `src/Idrak.Cli.Commands/Templates`: `console`, `webapi`, `rag`, `plugin` (`idrak new`) |
| Abstraction map | `plans/10-abstraction-inventory.md`: 503 contracts (interfaces, abstract classes, registries); `api/*.txt`, Craft's `api/*.txt` |
| Tests to convert | Idrak: 1,563 named test entries (`("group: what it checks", d => ...)`) in `tests/`, `samples/`; Craft: `tests/Craft.Tests` (83 files) |
| Projects to convert | Idrak: 25 sample and app projects (`samples/`, `apps/`), `Idrak.Samples.Override` (non-default implementations), the plug-in packages; Craft: 18 plug-ins, `samples/Craft.Plugins.Sample`, `templates/craft-plugin` |

Missing: any Claude client (none in Idrak; Craft's `Craft.Plugins.ClaudeCode` drives the Claude Code program, not the
API), a C# task suite larger than the three demo tasks, C# transcripts (the demo rows are Python), and an agent mode
for `idrak new`.

## The shape

### Where things live

- **`ahmedseada/Rl-Samples-`** (empty today; the owner, 2026-11-01: the samples live here): the generator and the dataset. It is a tool and data, not library code.
  - `src/RlSamples.Generator`: a C# console app on the official Anthropic C# SDK (`Anthropic` package) and on Idrak's
    packages (it reuses `CodingTools`, `CodingTask`, `CodingVerifier` to run and check, so a transcript's tools behave
    exactly as the student's will).
  - `src/RlSamples.Extract`: reads the Idrak and Craft clones (no API, no cost) into a catalog.
  - `data/`: the output, Parquet or JSON Lines by `idrak data`'s formats; large files by Git LFS (decision 4).
- **Idrak, `main`**: only what the library lacks (phase G6): the project generator, and any gap the generator finds in
  `CodingTools`, the `csharp` environment or the data commands, fixed in Idrak by Idrak's rules.
- **Craft**: nothing in this plan. A Craft plug-in that calls the generator can follow after Craft's usability phases,
  when the owner says.

### The generator (`RlSamples.Generator`)

- **Model choice is a setting**, never fixed: `RLS_MODEL` (default `claude-opus-5-5`), `RLS_WORKER_MODEL` for cheap bulk
  steps (paraphrase, tags, review; default the same model until the owner picks, decision 2), `RLS_EFFORT` per step.
  Adaptive thinking; streaming for long replies; refusals checked (`stop_reason`) and logged, never trained on.
- **Cost first**: one frozen, cached system prefix per step (the task rules, Idrak's coding rules from
  `docs/optimization.md`, the relevant `api/*.txt` excerpt, the six tool schemas), volatile content after the cache
  breakpoint; `usage.cache_read_input_tokens` logged, so a broken cache shows. One-shot steps (task specs, Q&A, reviews)
  go through **Message Batches** (half price, results keyed by `custom_id`). Agent transcripts are multi-turn and run
  online, with caching.
- **Every row verified before it is kept**: a task folder only when `agent-check` passes (fails before, passes with the
  solution); a transcript only when the task's hidden tests pass after it (rejection sampling); failing transcripts of
  the same task kept apart as DPO pairs.
- **Resumable and counted**: a run folder under `test-output\` with a manifest (rows asked, kept, rejected and why,
  tokens, dollars, time left); Ctrl+C keeps what is done; `--max-dollars` stops a run at a budget.
- **The key** (the owner, 2026-11-02): `RLS_ANTHROPIC_API_KEY`, read by the generator and given to the SDK's client,
  not `ANTHROPIC_API_KEY` (which Claude Code also reads: set, it bills every `claude` run to the key and refuses Remote
  Control). The generator never takes a key on the command line, never writes one to a file or a log, and fails with
  the variable's name when it is unset.
- **Every setting has an `RLS_*` variable** and a flag: model, effort, concurrency, budget, the source clones' paths,
  the sandbox (`IDRAK_SANDBOX`, `container` with an SDK image recommended), the split seed.

### The datasets

| Set | Rows | Format | For |
|---|---|---|---|
| T1-tasks | C# task folders (bug fix, implement, refactor, add tests, make faster by `optimization.md`, async, spans, LINQ to loops, ...) | suite folders | `reinforce --env csharp`, `agent-run` |
| T1-sft | Claude's verified transcripts on T1-tasks, in the agent's own tools | chat JSONL (agent-demo layout) | `tune` |
| T1-pref | a passing and a failing transcript of one task | preference JSONL | `tune` DPO/ORPO |
| T2-tests | each test entry turned into a task (below) | suite folders | `reinforce`, `agent-run` |
| T2-contracts | per contract: write a non-default implementation, register it in its `SlotTable`, prove it is used, prove an unregistered name fails with "not registered" | suite folders + transcripts | both |
| T2-projects | per sample, app and plug-in: "build this from the description" with the project's own build and tests as the check | suite folders + transcripts | both |
| T2-explain | questions on the contracts and registries answered from the source, each answer checked against the API dump | chat JSONL | `tune` |
| Eval | held out **by group** (a test group, a contract, a project), never by random row | the same | `agent-run` before and after |

### Track 2: how a test becomes a task

From each test entry (its group, its text, the method it runs, the code it reaches):

- **Light tasks** (most): a small project referencing Idrak by a local package feed (`dotnet pack` to
  `test-output\packages`, nothing published, no version bump). The workspace has the contract and a stub; the hidden
  check is the test's assertions moved into the project. Builds in seconds.
- **Heavy tasks** (few, the valuable ones): the repository itself at a fixed commit, the tested code removed or broken,
  the check the real test by name (`IDRAK_DEVICES=cpu IDRAK_FILTER=<name>`, `CRAFT_FILTER=<name>`). Built once per
  commit and shared (`IDRAK_CSHARP_PACKAGES`, the build server on).
- **Left out**, with the reason written in the catalog: GPU-only tests, benches, window and Android tests, tests that
  download a model, and tests that need the owner's machine.

Claude writes each conversion (the stub, the moved assertions, the prompt in the owner's words); `agent-check` decides
whether it is kept.

### Track 2: every abstraction namespace, none left out (the owner, 2026-11-01)

The owner: "every package has its own abstraction namespace, make sure to include all". The contracts are not only in
`Idrak.Abstraction`: each package keeps its own in `{package}.Abstractions` (Idrak's rule). G0 does not use a
hand-written list: it finds every namespace ending in `.Abstraction`, `.Abstractions` or under `Idrak.Abstraction.*`
in both repositories' sources, at the commit the run reads, and records each public interface, abstract class and
`SlotTable` registry in it. **G4 fails when any namespace has no kept task** (a contract task, an implementation task
or a written reason), and when a namespace found at the run's commit is missing from the catalog.

Found at Idrak 58e5c2c and Craft 6dbb736 (files declaring each namespace):

| Repository | Namespaces |
|---|---|
| Idrak, the shared contracts (`Idrak.Abstraction`) | `Idrak.Abstraction` (46), `.Cluster` (17), `.Data` (15), `.Devices` (25), `.Devices.Cpu` (21), `.Diagnostics` (8), `.Formats` (9), `.Generation` (40), `.Modules` (3), `.Operations` (4), `.Plugins` (5), `.Reinforcement` (11), `.Retrieval` (5), `.Serving` (5), `.Text` (2), `.Training` (6), `.Testing` (52), `.Testing.Devices` (4) |
| Idrak, each package's own | `Idrak.Models.Abstractions` (17), `Idrak.Layers.Abstractions` (11), `Idrak.Training.Abstractions` (3), `Idrak.Datasets.Abstractions` (4), `Idrak.Onnx.Abstractions` (3), `Idrak.Gpu.Abstractions` (11), `Idrak.Text.Abstractions` (15), `Idrak.Text.Retrieval.Abstractions` (1), `Idrak.Text.Reinforcement.Abstractions` (1), `Idrak.Data.Abstractions` (7), `Idrak.Vision.Abstractions` (11), `Idrak.Audio.Abstractions` (24), `Idrak.Audio.Plugins.Vits.Abstractions` (1), `Idrak.Cluster.Abstractions` (1), `Idrak.Networking.Abstractions` (6), `Idrak.Mcp.Abstractions` (1), `Idrak.Reinforcement.Abstractions` (2), `Idrak.Reinforcement.Plugins.Coding.Abstractions` (1), `Idrak.Maui.Abstractions` (16) |
| Craft | `Craft.Abstractions` (53), `Craft.Core.Abstractions` (1), `Craft.Editor.Abstractions` (4), `Craft.Plugins.Learning.Abstractions` (1) |

Packages with no abstraction namespace of their own are covered from the other side, as **implementations**: each
plug-in package (`Idrak.Text.Plugins`, `Idrak.Vision.Plugins`, `Idrak.Audio.Plugins`, `Idrak.Data.Plugins`,
`Idrak.Diffusion.Plugins`, `Idrak.Video.Plugins`, `Idrak.Reinforcement.Plugins`, `Idrak.HarfBuzz`, `Idrak.CoreML`,
`Idrak.Onnx.Runtime`, `Idrak.AspNetCore`, `Idrak.Extensions`, the 18 Craft plug-ins and `Craft.Plugins.Sample`) gives
"implement contract X the way this package does, register it, use it" tasks, checked against the contract it
implements. Two packages declare public interfaces outside an `.Abstractions` namespace (`Idrak.Network`: 2,
`Craft.Maui`: 7); G0 catalogs them too, and they go in the inventory note for the owner to decide whether they move.

Each contract gets, as it applies: a non-default implementation registered in its `SlotTable` and shown to be used; the
"not registered" failure for an unknown name; a plug-in package (`IIdrakPlugin`, `[assembly: IdrakPluginPackage]`,
`Register`/`Unregister` leaving the registry as found) that ships it; and a project that uses it end to end.

### Track 3: the project generator (`idrak new --agent`)

- `idrak new --agent "a REST API that classifies sentiment with a plug-in tokenizer" -o test-output\gen\api` plans the
  project (which packages, contracts and registries), starts from the nearest template, then the coding agent writes,
  builds and tests it until green or a round limit, showing live progress (step, round, build and test results, time).
- Templates become a registry (`ProjectTemplates`, a `SlotTable`): the four built in, plug-ins add their own (`-P`); an
  unknown template fails with "not registered".
- The model is any model argument (`--adapter` for the tuned LoRA); no family named in the command.
- Every file is written in the output folder only; `--dry-run` lists the files; commands run in the registered sandbox.
- Measured by an eval set of descriptions (from T2-projects' held-out projects): builds, tests pass, uses the
  registries rather than hard-coded types.

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

G-1 is done when each row has a fix with a targeted test (`IDRAK_DEVICES=cpu` with the group's `IDRAK_FILTER`), the
failure messages name the cause, and the owner has the GPU run command for B4.

## Phases

At most two agents at a time in Idrak (Idrak's limit; the Rl-Samples work counts against it while this plan runs).

| Phase | What | Done when | Needs |
|---|---|---|---|
| G-1 | The user-reported bugs B1-B7 and every "G-1" row of the full review, each with its targeted test | Every row fixed and tested; the owner has the GPU command for B4 | - |
| G0 | `RlSamples.Extract`: the catalog of every test entry, contract, registry, sample, app and plug-in of both repos, with what each needs (CPU, GPU, a model, a window) | The catalog's counts match the inventory and the test lists; no API call | - |
| G1 | `RlSamples.Generator`'s core: the client, cached prefixes, batches, the run manifest, the budget, the verifier on `CodingVerifier` | A 20-task pilot runs end to end on the owner's machine with its cost measured | G0 |
| G2 | Track 1 at volume: T1-tasks, T1-sft, T1-pref | Targets of decision 3 reached; every row verified | G1, pilot cost approved |
| G3 | Track 2: T2-tests, T2-contracts, T2-projects, T2-explain | Every catalog entry is a kept task or a written reason | G1 (parallel with G2) |
| G4 | Quality: `idrak data validate`, `dedupe` (near), `stats`, group splits, a leakage check of eval against train, the namespace coverage check | A report under `test-output\`; every abstraction namespace covered | G2, G3 |
| G5 | The owner's runs: `idrak tune` on T1-sft + T2-sft, then `idrak reinforce --env csharp` on the task sets, `agent-run` on the eval before and after | The tuned adapter beats the base on the eval | G4 |
| G6 | Track 3 in Idrak: `ProjectTemplates` registry, `idrak new --agent`, its tests (a scripted model, as `CSharpCodingPluginTests` does) | CPU tests pass; the owner's run on the tuned adapter writes projects that build | G5 for the model; the command itself can start after G1 |

## Decisions for the owner

1. **Where the plan and the code live.** Recommended: this plan in Idrak (`plans/opened`), the generator and data in
   `Rl-Samples-` on `main`; Track 3 in Idrak. Both CLAUDE.md files
   say `main` only, while this session was given `claude/idrak-csharp-finetuning-qpp0k8`: recommended `main`.
2. **Models.** Recommended: `claude-opus-5-5` for task specs and agent transcripts; the owner says whether bulk steps
   (paraphrase, tagging) may use `claude-sonnet-5-5` or `claude-haiku-5-5`.
3. **Volume and budget.** Recommended first targets: 2,000 T1 tasks, 10,000 T1 transcripts, 3,000 pairs; every
   convertible Track 2 entry once, with up to three transcripts each. Rough cost before the pilot: about $0.20 a
   verified transcript on Opus 5.5 with caching (a 10-turn run, mostly cache reads), so about $2,000 for 10,000, plus
   rejected runs; G1's pilot replaces this guess with a measured number before G2 starts.
4. **Data storage** (decided by the owner, 2026-11-01: "the target samples will live in the samples repo"): every
   generated sample (task folders, transcripts, pairs, eval sets) lives in `ahmedseada/Rl-Samples-`; Git LFS or plain
   files there is still open for large files.
5. **The student model**: which base G5 tunes (the bench's Qwen-Coder is the recommendation, the owner's runs decide).
6. **Terms of use**: Anthropic's terms limit using Claude's outputs to train models; the owner checks that this use
   (a model for the owner's own C# agent and generator) is allowed under the account's terms before G2.

## Rules that hold for every phase

- Idrak's `CLAUDE.md` and `docs/optimization.md` hold for any Idrak code (G6) and for the generator's hot paths
  (catalog walk, transcript recording): spans over strings, pooled buffers, work done once, memory that does not grow.
- No family-specific code in Idrak: the generator command takes any model; Claude is only the generator's teacher.
- Tests: targeted only (`IDRAK_DEVICES=cpu IDRAK_FILTER=<group>`); the training and model runs are the owner's, given as
  exact PowerShell commands.
- Nothing runs or is written outside a task's sandbox and output folder.
