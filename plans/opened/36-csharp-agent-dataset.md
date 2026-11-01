# Plan 36: a C# agent dataset made with Claude, and a project generator on it

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

- **`ahmedseada/Rl-Samples-`** (empty today): the generator and the dataset. It is a tool and data, not library code.
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

## Phases

At most two agents at a time in Idrak (Idrak's limit; the Rl-Samples work counts against it while this plan runs).

| Phase | What | Done when | Needs |
|---|---|---|---|
| G0 | `RlSamples.Extract`: the catalog of every test entry, contract, registry, sample, app and plug-in of both repos, with what each needs (CPU, GPU, a model, a window) | The catalog's counts match the inventory and the test lists; no API call | - |
| G1 | `RlSamples.Generator`'s core: the client, cached prefixes, batches, the run manifest, the budget, the verifier on `CodingVerifier` | A 20-task pilot runs end to end on the owner's machine with its cost measured | G0 |
| G2 | Track 1 at volume: T1-tasks, T1-sft, T1-pref | Targets of decision 3 reached; every row verified | G1, pilot cost approved |
| G3 | Track 2: T2-tests, T2-contracts, T2-projects, T2-explain | Every catalog entry is a kept task or a written reason | G1 (parallel with G2) |
| G4 | Quality: `idrak data validate`, `dedupe` (near), `stats`, group splits, a leakage check of eval against train | A report under `test-output\` | G2, G3 |
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
4. **Data storage**: Git LFS in `Rl-Samples-`, or the data kept out of git (a release asset, a Hugging Face dataset).
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
