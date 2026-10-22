# Plan 27: reinforcement learning (`Idrak.Reinforcement`, `Idrak.Reinforcement.Plugins`)

**Status (2026-10-22, `abstraction`):** planned 2026-10-21 from the owner's decisions, against the code at acfc83b (every
"Comes from" below was found in the source). The owner answered the open questions and started the plan on 2026-10-22
("go with recommendation", at most two agents): R0 and R1 run side by side.

| Phase | What | Lane | State |
|---|---|---|---|
| R0 | What moves out of `Idrak.Text` into the library: optimizers and schedules by name as registries, the answer metrics into `TuningMetrics`, `LogSigmoid` into core | A | Done 2026-10-22; CPU pass |
| R1 | The contracts (`Idrak.Abstraction.Reinforcement`) and the two package skeletons (`Idrak.Reinforcement`, `Idrak.Reinforcement.Plugins`: registration, the CLI) | B | Done 2026-10-22; CPU pass |
| R2 | `Idrak.Reinforcement`, the general part: rollouts, advantages, policy and value losses, the on-policy trainer, a module policy, checkpoints, progress, settings | A | Done 2026-10-22; CPU pass |
| R3 | The language-model policy in `Idrak.Text`: a fine-tuning session, sampled tokens recorded, the `prompts` environment, the text rewards, on-policy distillation from a teacher (alone or beside a reward) | B | Built 2026-10-22; CPU pass (rollouts on a cluster done in R3b; see "R3b as built") |
| R3b | R3's leftover: rollouts on a cluster (the language-model policy's episodes spread over the ranks, gathered in rank order; the policy refused a cluster session before) | B | Done 2026-10-22; CPU pass |
| R3c | R3b's leftover: `idrak reinforce` under `idrak cluster sim` (the command on every rank, the adapter and manifest written once by rank 0) | A | Done 2026-10-22; CPU pass (on-policy distillation on a cluster not taken, refused with the reason; see "R3c as built") |
| R4 | `Idrak.Reinforcement.Plugins.Coding`: the coding task layer moves out of `Idrak.Text`, the coding environment, its rewards, the command sandbox (`process`, `container`) | B | Done 2026-10-22; CPU pass |
| R4b | No programming language in the library: coding languages as a registry (`CodingLanguages`), C# and TypeScript/Angular as `Idrak.Text.Plugins` profiles; `Idrak.Reinforcement.Plugins.CSharp`: agentic coding on C# projects (dotnet defaults, rewards from the build's errors and the tests' results, a shared package cache) | A | Done 2026-10-22; CPU pass |
| R5 | `idrak reinforce`, the settings carried into the CLI and samples, the CodingAgent sample on the library, a small demo suite | A | Done 2026-10-22; CPU pass (left: the demo suite through R4b's `--env csharp` after its merge; see "R5 as built") |
| R6 | (optional, the owner decides) `Idrak.Reinforcement.Plugins.Classic`: bandit, CartPole, Pendulum, MountainCar; off-policy (replay buffer, DQN) | A or B | Queued (the owner, 2026-10-22): starts after R5, when the main plan is finished |

## Goal

Reinforcement learning as a general part of the library: an agent acts in an environment, is rewarded, and its policy is
updated from what it did. The library holds the contracts and the algorithms, card-agnostic and with no model family or
environment in it; environments that several apps use are plug-ins in `Idrak.Reinforcement.Plugins`, one namespace per
environment family, as `Idrak.Vision.Plugins` holds the vision families. **Agentic coding is the first application**,
not the package's purpose: a language model learns to solve coding tasks (a project, a prompt, hidden tests) with the
coding agent's tools, rewarded by the tests.

## Decisions (taken by the owner, 2026-10-21)

1. **A separate package, `Idrak.Reinforcement`,** not joined to fine-tuning in `Idrak.Text`.
2. **General reinforcement learning,** not only language models; agentic coding is the first application on top.
3. **`Idrak.Reinforcement.Plugins`** for environments used by more than one app, laid out like `Idrak.Vision.Plugins`:
   one namespace per family (`Idrak.Reinforcement.Plugins.Coding`), a static `…Plugin` class each, a root class
   `ReinforcementPlugins` with `All`, `RegisterAll()`, `UnregisterAll()`, `RegisterIdrakPlugin()`.
4. **What should move out of `Idrak.Text` moves** in this plan (the owner, 2026-10-21); see "What moves out of
   `Idrak.Text`".
5. **Plan first:** this file is written before any code.

**Proposed in the same session (2026-10-21), confirmed by the owner 2026-10-22** ("go with recommendation"):

6. **`Idrak.Reinforcement` does not reference `Idrak.Text`.** A user training a game agent or a vision policy must not pull in
   chat templates, retrieval and the coding agent. The contracts live in `Idrak.Abstraction.Reinforcement`; `Idrak.Text`
   implements the language-model policy against them and never references `Idrak.Reinforcement` either. The plug-in
   package references both (as `Idrak.Vision.Plugins` references `Idrak.Vision` and `Idrak.Text.Plugins`).
7. **DPO, ORPO and SimPO stay in `Idrak.Text`.** They are losses on fixed preference pairs over `FineTuner`, built and
   tested; moving them is churn. `PreferencePair` stays with them.
8. **The coding agent's loop stays in `Idrak.Text`** (a chat model with tools is inference, used by `idrak agent` with no
   training); its task layer (tasks, suites, workspaces, verification) is the coding environment and moves (M5).

## Rules that hold for every phase

CLAUDE.md and [docs/optimization.md](../docs/optimization.md) apply in full. The ones that shape this plan:

- **Contract plus registry (decision 10).** Every new behaviour is a contract and a `SlotTable` registry (environments,
  rewards, advantage estimators, policy losses, sandboxes), its defaults registered from the package's static
  registrations; the inventory and the API dump are regenerated after each public change. A contract any package can use
  lives in `Idrak.Abstraction`; one only `Idrak.Reinforcement` can use lives in `Idrak.Reinforcement.Abstractions`. The
  plug-in package counts as an app when the inventory decides where a contract lives (Plan 26, decision 11).
- **No environment, family or task in the library.** An environment kind that is not registered fails loudly with "not
  registered", naming `Idrak.Reinforcement.Plugins`; nothing falls back to another environment.
- **Plug-ins use the public API only;** no `InternalsVisibleTo` for either new package.
- **Results unchanged, proven (rule 71).** R0 and R3 change code that runs today (optimizer and schedule names, the answer
  metrics, `FineTuner.Train`, the coding agent): every existing test of it passes unchanged, and `FineTuner.Train`'s losses
  on the `fine-tuning:` cases are the same bit for bit before and after it becomes a loop over a session.
- **Reproducible.** A seeded run replays the same episodes, rewards and advantages on every device: seeds per episode
  drawn from the run's seed, sums in a fixed order (rule 86: a group's mean and deviation, a batch's loss), no order from
  thread timing (parallel episodes are collected, then sorted by episode index).
- **Speed and memory first, measured at run time, card-agnostic.** Rollouts and updates share one device: the KV cache
  of generation and the training activations come from the same pool and do not grow with steps (rule 77); the full
  `[tokens, vocabulary]` logits are never stored (`FineTuner`'s chunked head, `LossChunkRows`); environment work (tools,
  builds, tests) runs on the cores while the device generates other episodes' turns (rule 79, checked that it paid).
  Group size, rollout batch and update batch default to what the device's free memory measures; each is a setting.
- **Nothing fixed.** Every algorithm constant (clip, KL weight, group size, γ, λ, normalization), every limit (rounds,
  time, memory of a command) and every placement has an option, and an `IDRAK_RL_*` or `IDRAK_SANDBOX_*` variable where
  the library reads one, carried into `src/Idrak.Cli/Shared/EnvironmentVariables.cs` (a test checks the table).
- **Long work shows live progress** through `IProgressReporter` (Plan 26, G4): episodes, mean reward, pass rate, tokens a
  second, time left.
- **Tests only through filters,** CPU here (`IDRAK_DEVICES=cpu`); GPU tests, real models and training runs are the
  owner's, with the exact commands in each phase.
- **The fix workflow:** fix everything, update the tests, build once, run the phase's filters once with `--no-build`;
  after a merge, build and run before pushing. At most two agents at a time.

## What exists (2026-10-21, acfc83b)

**Fine-tuning (`src/Idrak.Text`):**
- `FineTuner.Train` (`FineTuning.cs:1119`, sequences; `:1152`, preference pairs) runs everything in one call through
  `TrainCore` (`:1182`): adapters (`LoRA`, `DoRA`; `ILinearAdapter`, `src/Idrak.Abstraction/Modules/LinearAdapters.cs`),
  optimizer and schedule, FP8 copies of frozen weights, checkpoints and resume, the cluster (`FineTuningOptions.Cluster`),
  evaluation, generated answers scored on rank 0 (`:1364`). It takes a fixed list of `TrainingSequence(int[] Tokens,
  bool[] Trained)` (`Abstractions/Distillation.cs:14`); there is no way to train a step, generate, and train the next.
- `FineTuningLoss` / `FineTuningLossInput` (`FineTuningLosses.cs:20`, `:29`): a loss from the trained tokens'
  log-probabilities, with `ReferenceLogProbabilities` (the base model, adapters disabled, no second model loaded),
  `TeacherDivergence`, `SequenceSums`, `SequenceMeans`. No per-sequence values (advantages) and no behaviour
  log-probabilities (those of the policy that sampled).
- `FineTuningLosses`: token cross-entropy, distillation, DPO, ORPO, SimPO, and the public helper `LogSigmoid`
  (`FineTuningLosses.cs:373`).
- `FineTuningOptimizers` (`adamw`, `adam`, `adamw8bit`, `sgd`) and `FineTuningSchedules` (`cosine`, `linear`, `constant`,
  `wsd`) (`FineTuningLosses.cs:394`, `:464`): **closed name lists, not registries**, used by `Idrak.Text`'s pretraining and
  by `Idrak.Audio`'s `SpeechTrainer` and `AdversarialSynthesisTrainer` and the CLI's `TuneTool`.

**Generation:** `ChatGenerator.ChatBatch` / `StreamBatch` (`Generation/ChatGenerator.cs:248`, `:289`) generate many replies
at once; `ChatChunk` (`src/Idrak.Abstraction/Generation/ChatModel.cs:57`) carries text, message and stats, **not the
token ids sampled** nor their log-probabilities. Re-encoding a reply's text can give other ids than the ones sampled.

**Scoring:** `TuningMetrics` (`src/Idrak.Abstraction/Generation/TuningMetrics.cs:85`, a `SlotTable`: `cer`, `wer`);
`AnswerMetric` (`src/Idrak.Text/Evaluation.cs:17`: `Auto`, `Number`, `Exact`, `Contains`, `F1`) is **a closed enum with its
own scorer** (`ChatEvaluation.Score`, `:130`), used by `ChatEvaluation`, the CLI's `TuneTool` and `EvalCommand`, and
`GgufTests`. `AnswerScorer` (`AnswerScorer.cs:37`) gives log p(answer | prompt) for a list of answers (a judge's yes/no).

**The coding agent (`src/Idrak.Text/Generation`):**
- `CodingAgent` (`CodingAgent.cs:148`): the loop (`RunAsync(AgentTask, CodingTools)`, `:273`, uses only the task's
  `Prompt`) and the task layer in the same class: `AgentTask` with `LoadSuite` / `Load` (`:27`; `task.json`, `workspace/`,
  `verify/` hidden tests, `solution/`), `PrepareWorkspace` (`:359`), `CommitStart` (`:413`), setup, `VerifyAsync` (`:250`),
  `CheckTaskAsync` (`:215`), `AgentOutcome` (`:87`), `AgentRun` (`:119`) with its JSON transcript.
- `CodingTools` (`CodingTools.cs`): list, read, search, edit, write, `run_command` through `ExecuteAsync` (`:360`), which
  starts a plain `Process` (`:400`, `:429`) limited by an allow-list of programs, read-only git, a time limit and an
  output cap; **no memory limit, no network limit, no isolation** beyond the workspace path.
- Users: `src/Idrak.Cli/Commands/Run/AgentCommand.cs` (`idrak agent`), `samples/Idrak.Samples.CodingAgent` (`agent`,
  `agent-run` with `--attempts`, `agent-check`; `data/agent-demo-*.jsonl` transcripts for `idrak tune`), the tests in
  `tests/Idrak.Tests/CodingToolsTests.cs` (`coding agent:`, `coding tools`). No task suite is checked in.

**Generic training pieces (`src/Idrak.Abstraction/Training`):** `Optimizer`, `Adam`, `AdamW`, `Sgd`,
`LearningRateScheduler` and four schedules; `Module`, autograd and the tensor operations in core. **No reinforcement
learning anywhere:** no environment, policy, reward, advantage or rollout type.

## The packages and their references

| Package | References | Holds |
|---|---|---|
| `Idrak.Abstraction` (namespace `Idrak.Abstraction.Reinforcement`) | (as today) | The contracts and registries R-C1 to R-C7; the sandbox contract R-C8 in `Idrak.Abstraction` too (any package runs commands) |
| `Idrak.Reinforcement` (new) | `Idrak` | The general algorithms (R2), registered into the contracts; a module policy for policies built from `Module`s; the trainer |
| `Idrak.Text` | (as today) | The language-model policy, the `prompts` environment, the text rewards, the `process` sandbox (R3), on the contracts alone; **no reference to `Idrak.Reinforcement`** |
| `Idrak.Reinforcement.Plugins` (new) | `Idrak.Reinforcement`, `Idrak.Text` | Environment families, one namespace each (R4, R6) |
| `Idrak.Cli` | the above | `idrak reinforce`; registers `Idrak.Reinforcement.Plugins` at startup with the other three plug-in packages |

The dependency runs one way: plug-ins on the packages, never back; `Idrak.Text` and `Idrak.Reinforcement` meet only in
`Idrak.Abstraction`'s registries.

## What moves out of `Idrak.Text`

| # | What | From | To | Why | Users changed | Phase |
|---|---|---|---|---|---|---|
| M1 | Optimizers by name | `FineTuningOptimizers` (closed list, `FineTuningLosses.cs:464`) | **new** registry `Optimizers` (`Idrak.Abstraction.Training`): `adamw`, `adam`, `adamw8bit`, `sgd`, an app's own by name | `Idrak.Reinforcement` (no reference to `Idrak.Text`) needs them; a closed list breaks "abstraction always" | `FineTuner`, `Pretraining`, `PipelinePretraining`, Audio's two trainers, `TuneTool` | R0 |
| M2 | Learning-rate schedules by name | `FineTuningSchedules` (closed list, `:394`) | **new** registry `LearningRateSchedules` (`Idrak.Abstraction.Training`): `cosine`, `linear`, `constant`, `wsd` | the same | the same | R0 |
| M3 | The answer metrics | `AnswerMetric` + `ChatEvaluation.Score` (`Evaluation.cs:17`, `:130`) | `TuningMetrics` (exists): `number`, `exact`, `contains`, `f1` beside `cer`, `wer`; `auto` stays `ChatEvaluation`'s choice of a name | rewards and evaluation then read one registry; a closed enum is a fixed list | `ChatEvaluation`, `TuneTool`, `EvalCommand`, `GgufTests` | R0 |
| M4 | `LogSigmoid` | `FineTuningLosses.LogSigmoid` | core `Losses` (`src/Idrak/Losses.cs`), `FineTuningLosses` calling it | generic tensor math; policy and preference losses both use it | DPO, SimPO, the policy losses | R0 |
| M5 | The coding task layer | `AgentTask`, `LoadSuite`, `PrepareWorkspace`, `CommitStart`, setup, hidden files, `VerifyAsync`, `CheckTaskAsync`, `AgentOutcome`, the task part of `AgentRun` (`CodingAgent.cs`) | `Idrak.Reinforcement.Plugins.Coding`: `CodingTask`, `CodingSuite`, `CodingWorkspace`, `CodingOutcome`, `CodingEnvironment` | a task, its workspace and its verification are the coding environment; two copies (evaluation and training) would break "no copies" | `idrak agent` (only the loop), the CodingAgent sample, `coding agent:` tests | R4 |

**Stays in `Idrak.Text`:** `FineTuner`, its adapters, losses DPO / ORPO / SimPO / distillation, `PreferencePair`,
`ChatTranscriptEncoder`, `TuningAnswerScorer`, `AnswerScorer`, `CodingTools`, and `CodingAgent`'s loop
(`RunAsync(string prompt, CodingTools tools, …)`, `AgentOptions`, `DefaultSystemPrompt`, the stream callbacks), which
returns the conversation (`CodingAgentRun`: messages, tools, rounds, calls, errors, tokens, times) with no task in it.
Names keep a family-free, `System.*`-free form; the moved types keep their JSON layouts (`task.json`, the run's JSON line).

## The contracts (`Idrak.Abstraction.Reinforcement`, new)

| # | Contract | Registry | Defaults (package) | Used by |
|---|---|---|---|---|
| R-C1 | `IEnvironment<TObservation, TAction>`: `Reset(seed) → TObservation`, `StepAsync(TAction, ct) → EnvironmentStep<TObservation>` (observation, reward, terminated, truncated, info), `ObservationSpace`, `ActionSpace` (`Space`: `Discrete(n)`, `Box(shape, low, high)`, `Text`, `Conversation`) | — | — | the trainer, every environment |
| R-C2 | `IEnvironmentKind`: `Name`, `Summary`, `Keys` (options), the observation and action types, `Create(options, seed)` | `EnvironmentKinds` (by name, ignoring case) | `prompts` (`Idrak.Text`); `coding` and the classic ones are plug-ins | the trainer, `idrak reinforce --env` |
| R-C3 | `IPolicy<TObservation, TAction>`: `ActAsync(observations, sampling, ct) → (actions, ActionRecords)` (what the update needs: token ids, behaviour log-probabilities, values), `Update(RolloutBatch, IPolicyLoss) → UpdateStats`, `Save`, `Load` | — | `ModulePolicy` (`Idrak.Reinforcement`), `LanguageModelPolicy` (`Idrak.Text`) | the trainer |
| R-C4 | `IRewardFunction`: `Name`, `Summary`, `Keys`, `ScoreAsync(IReadOnlyList<RewardInput>, options, ct) → float[]` (`RewardInput`: prompt, answer, reference, info) | `RewardFunctions` | `metric`, `regex`, `json`, `length`, `judge`, `sum` (`Idrak.Text`); `tests-passed`, `verify-fraction`, `rounds`, `tool-errors` (the coding plug-in) | environments that score text |
| R-C5 | `IAdvantageEstimator`: rewards, values, ends and group ids → advantages and returns, in a fixed order | `AdvantageEstimators` | `gae` (γ, λ), `group` (GRPO: centred on the group's mean, scaled by its deviation or not), `leave-one-out` (RLOO), `returns` (REINFORCE, with or without a baseline) (`Idrak.Reinforcement`) | the trainer |
| R-C6 | `IPolicyLoss`: `PolicyLossInput` (log-probabilities of the actions taken, behaviour and reference log-probabilities, advantages, values and returns, counts per sequence, step sizes) → one-element loss; `Keys` | `PolicyLosses` | `reinforce`, `ppo` (clipped ratio, value loss, entropy), `grpo` (clipped ratio, KL to the reference by the k1 or k3 estimator, normalized per sequence, per token or by a constant) (`Idrak.Reinforcement`) | `ModulePolicy`, `LanguageModelPolicy` through `FineTuningLosses.Policy` |
| R-C7 | `RolloutBatch`, `Trajectory`, `ActionRecord`: the data passed between them (pooled arrays, returned in `finally`) | — | — | all of the above |
| R-C8 | `ICommandSandbox` (`Idrak.Abstraction`): `Name`, `RunAsync(command, directory, CommandLimits, ct) → CommandResult` (`CommandLimits`: time, output, memory, processes, network allowed); a sandbox that cannot enforce a limit it is asked for **refuses**, naming one that can | `CommandSandboxes` | `process` (`Idrak.Text`, the code of `CodingTools.ExecuteAsync`); `container` (the coding plug-in) | `CodingTools`, the coding environment |

R-C8 is used by one library package (`Idrak.Text`; the plug-in counts as an app): it gets a line in
`tests/Idrak.Tests/data/shared-contracts.txt` ("any package that runs an outside command, inference or training, runs it
the same way and under the same limits").

## The plug-ins (`Idrak.Reinforcement.Plugins`)

| # | Namespace | Plug-in | Registers | Comes from | Phase |
|---|---|---|---|---|---|
| 1 | `Idrak.Reinforcement.Plugins.Coding` | `CodingPlugin` ("coding") | `EnvironmentKinds`: `coding`; `RewardFunctions`: `tests-passed`, `verify-fraction`, `rounds`, `tool-errors`; `CommandSandboxes`: `container` | M5 (`src/Idrak.Text/Generation/CodingAgent.cs`), new | R4 |
| 2 | `Idrak.Reinforcement.Plugins.CSharp` | `CSharpCodingPlugin` ("csharp") | `EnvironmentKinds`: `csharp`; `RewardFunctions`: `csharp-tests`, `csharp-build`; requires "coding" and "csharp-language" | new (the owner, 2026-10-22) | R4b |

R4b also adds two profiles to `Idrak.Text.Plugins` (not reinforcement plug-ins, as `idrak agent` uses them without
training): `Idrak.Text.Plugins.CSharp` ("csharp-language"), `Idrak.Text.Plugins.TypeScript` ("typescript-language",
profiles `typescript` and `angular`), from the library's coding agent, and `Idrak.Text.Plugins.Python` ("python-language",
ignored folders only), from `Idrak.Data`'s code reader.
| 3 | `Idrak.Reinforcement.Plugins.Classic` | `ClassicControlPlugin` ("classic") | `EnvironmentKinds`: `bandit`, `cartpole`, `pendulum`, `mountain-car` | new | R6 (optional) |

`container` is a plug-in, not a library default, because it runs an outside program (Docker or Podman), as the espeak-ng
phonemizer is a plug-in (Plan 26, #18).

---

## Phase R0: what moves out of `Idrak.Text` (M1-M4)

- `Optimizers` and `LearningRateSchedules` in `Idrak.Abstraction.Training`, each a `SlotTable` with `Register`,
  `Unregister`, `Names`, `Find`, `Get` ("not registered", naming the registered ones), `Default`, `Origin`, a policy, a
  contract suite and an inventory line; the four optimizers and four schedules registered as library defaults with the
  same parameters and the same results. `FineTuningOptimizers` and `FineTuningSchedules` are removed; every user reads the
  registries (`FineTuner`, `Pretraining`, `PipelinePretraining`, `SpeechTrainer`, `AdversarialSynthesisTrainer`,
  `TuneTool`); `FineTuningOptions.Optimizer` and `Scheduler` keep their delegate form.
- `number`, `exact`, `contains`, `f1` as `ITuningMetric`s beside `cer` and `wer` (higher is better, the same
  normalization as `ChatEvaluation`'s today, on spans: rule 74); `AnswerMetric` is removed, `ChatEvaluation.Run` and
  `Score` take a metric name (`auto` picks `number` for gsm8k-style references, else `f1`, as now); `idrak tune` and
  `idrak eval`'s `--metric` list the registry.
- `LogSigmoid` in core `Losses`; DPO and SimPO call it.

**Done when:** every test of the moved code passes unchanged (results unchanged, rule 71: the same losses and scores on
the existing cases); the inventory, API and naming guards pass; nothing names `FineTuningOptimizers`,
`FineTuningSchedules` or `AnswerMetric`.

**Tests (CPU here):** one build; `fine-tuning:|pretrain:|evaluation:|tuning metrics|audio train|audio tts train|
cli tune|cli eval|gguf:|optimizers|schedules|public API|abstraction inventory|cli health: every environment variable|
naming:`.

**R0 as built (2026-10-22).** M1-M4 in one change, results unchanged.
- **M1, optimizers by name:** `IOptimizerKind` (`Name`, `Summary`, `Create(parameters, OptimizerSettings)`) and the
  registry **`OptimizerKinds`** in `Idrak.Abstraction.Training` (`Register`, `Unregister`, `Names`, `Find`, `Get`,
  `Default`, `Origin`, `Describe`, and `Factory(name, settings)`, which gives the `Func<IReadOnlyList<Tensor>, Optimizer>`
  that `FineTuningOptions.Optimizer` and `HostOptimizer` take and fails on an unknown name at once). `OptimizerSettings`:
  learning rate, weight decay (default 0), momentum (default 0.9) and `Beta1`, `Beta2`, `Epsilon`, null for the kind's own
  (Adam's 0.9, 0.999, 1e-8), so the old `Create(name, rate, decay, momentum)` and Pretraining's betas are both one call.
  adamw, adam and sgd are registered by Idrak.Abstraction itself; adamw8bit (in core) by core's `LibraryRegistrations`
  through `LibraryDefaults.Ensure`, called between adam and sgd so the names keep their old order (adamw, adam, adamw8bit,
  sgd). **Deviation: the registry is `OptimizerKinds`, not `Optimizers`:** a class `Optimizers` would be hidden by the
  namespace `Idrak.Optimizers` in all code under `Idrak.*` (C# finds the namespace first), so every user, R2's included,
  would have to spell `Idrak.Abstraction.Training.Optimizers`; the repository met the same with `Idrak.Network`
  (`LibraryDefaults`' note). The contract is named to match (`IOptimizerKind`, as R-C2's `IEnvironmentKind`).
- **M2, schedules by name:** `ILearningRateSchedule` (`Name`, `Summary`, `Create(LearningRateScheduleSettings)` returning
  the `Func<Optimizer, int, LearningRateScheduler>` that `FineTuningOptions.Scheduler` takes, so wsd still refuses a decay
  share outside [0, 1] when made) and **`LearningRateSchedules`** (the same members, `Factory(name, settings)`, the names
  as constants `Cosine`, `Linear`, `Constant`, `WarmupStableDecay`). `LearningRateScheduleSettings(WarmupFraction = 0.03,
  MinLearningRate = 0, DecayFraction = 0.2)` are the old `Create`'s parameters and defaults; the four schedules are the old
  code, moved. The fine-tuner's default schedule is the registry's "cosine".
- Both registries are **unguarded** (`unguarded:` with the reason; no `SetPolicy`): an optimizer holds its moments from the
  first step to the last and a checkpoint saves them under its own names, and a schedule sets the rate at every step from
  where it is, so neither can fall back half-way (as `AttentionPositionBiases`). Contract suites in the testing kit:
  `OptimizerKindSuite` (the parameters given, the rate asked for, the first step and the steps lower a squared loss, a
  parameter the loss does not reach is left alone, the same case gives the same values, a reference agrees) and
  `LearningRateScheduleSuite` (the optimizer and its rate as the base, every rate finite, not negative and not above the
  optimizer's, the same rates again, a reference agrees), with `Conformance.CheckOptimizerKind` and
  `CheckLearningRateSchedule`, which compare with the library's of that name. `FineTuningOptimizers` and
  `FineTuningSchedules` are removed; their users read the registries: `FineTuner`, `Pretraining` (its own switch for the
  Adam kinds' betas became one `Factory` call with the betas in the settings), `PipelinePretraining`, `SpeechTrainer`,
  `AdversarialSynthesisTrainer` (Adam and AdamW still take the adversarial betas and epsilon, the others the speech
  trainer's, as before), `TuneTool`, the ArabicOcr sample's `train`. An app's kind or schedule under a library name is now
  used by these trainers; the CLI's `tune` keeps its default AdamW and cosine when none is named, as before.
- `tests/Idrak.PluginTests/Lion.cs` adds `LionKind`: registered into `OptimizerKinds` from the outside assembly, made by name
  ("LION", any case) it makes the reference first step, passes the optimizer suite and unregisters.
- **M3, the answer metrics:** "number", "exact", "contains" and "f1" are `ITuningMetric`s in `TuningMetrics` after "cer"
  and "wer" (constants `NumberMatch`, `ExactMatch`, `ContainsMatch`, `WordF1`), higher is better, each answer out of 1, no
  options (an option is refused naming the metric). The code is `ChatEvaluation`'s, moved unchanged
  (`TuningMetricImplementations.cs`: the span-based normalization in a stack or pooled buffer, F1 counted by span through
  an alternate lookup, the source-generated patterns), so the scores are the same; no Unicode normalization was added
  (the error rates keep theirs). `TuningMetrics.FinalNumber` and `HasFinalAnswer` replace `ChatEvaluation.FinalNumber`.
  The enum is removed: `ChatEvaluation.Run` and `Score` take a metric name, `ChatEvaluation.Auto` ("auto", any case) stays
  its own choice (number when the first reference holds "#### n", else f1), `EvaluationReport.Metric` is the name chosen.
  `idrak eval --metric` takes auto or any registered metric and lists the registry (help and error); with an error rate it
  shows the answers that are not 0 under `-v`. `idrak tune --metric` lists the registry; auto and the four answer metrics
  score held-out rows (`ChatEvaluation`) and any other registered metric generated text on conversation files, as before.
- **M4:** `Losses.LogSigmoid` in core (the same code); DPO, conservative DPO, ORPO and SimPO call it; the
  `FineTuningLosses` member is removed.
- No new environment variable. `api/Idrak.Abstraction.txt`, `Idrak.Abstraction.Testing.txt`, `Idrak.Text.txt`,
  `Idrak.txt` and `plans/10-abstraction-inventory.md` regenerated. The tests whose code moved keep their expectations
  (`fine-tuning options:` schedules' rates, `evaluation:` scores, preference formulas and log-sigmoid); only the calls
  changed. One expectation changed with the cause: `vision tuning contracts: cer and wer` registered an app metric "exact"
  and expected nothing after `Unregister`; the library has an "exact" now, so the test checks that the library's comes
  back. New group (`TrainingRegistryTests.cs`): `optimizers:` (each name makes its constructor's updates bit for bit, betas
  too, on 5,000 elements so adamw8bit's 8-bit moments run; every kind passes the suite; an app's sgd without momentum takes
  the library's place and fails the suite on the reference alone; unknown names refused naming the registered ones),
  `schedules:` (the four formulas' rates, the suite, an app's constant, wsd's decay share, unknown names) and `tuning
  metrics:` (21 known answers against their scores, options refused, corpus sums, auto, unknown names).
- **The run (CPU, one build):** `fine-tuning options:|evaluation:|optimizers|schedules:|tuning metrics:|vision tuning
  contracts: cer and wer|pretrain:|distillation: the fine-tuning loss|qwen vl: a tuning example|vision tuning
  trainer|audio train|audio tts train|cli measure: eval|cli train data|cli vision tuning|a Lion optimizer|public
  API|abstraction inventory|naming:|cli health: every environment variable|override loop:|conformance kit:`: **87 passed,
  0 failed**; the ArabicOcr sample's `ocr train`: 2 passed, 0 failed. (The plan's `cli tune`, `cli eval`, `gguf:` and
  `fine-tuning:` name no test of the changed code as such: `cli eval` is `cli measure: eval`, the tuning CLI's are `cli
  train data` and `cli vision tuning`, the GGUF file's metric test is `evaluation:`, and `fine-tuning:`'s tests use none of
  the moved members.)
- **The owner's commands** (repo at `D:\Projects\Idrak`):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "fine-tuning options:|optimizers|schedules:|tuning metrics:|pretrain:|distillation: the fine-tuning loss|vision tuning trainer|audio train|audio tts train|a Lion optimizer|conformance kit:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

## Phase R1: the contracts and the two package skeletons

- `Idrak.Abstraction.Reinforcement`: R-C1 to R-C7 (R-C8 goes with R4), each registry with its suite in
  `Idrak.Abstraction.Testing` (an environment's reset and step rules, a seeded replay, a reward's batch against its single
  form, an estimator's fixed order), the "not registered" messages naming `Idrak.Reinforcement` or
  `Idrak.Reinforcement.Plugins`, and inventory lines.
- `src/Idrak.Reinforcement` (references `Idrak`): project, README, `LibraryRegistrations` (empty until R2), listed in
  `LibraryDefaults`' package list and `DynamicDependency`, `Idrak.slnx`, `build/IdrakFromSource.targets`, the API guard.
- `src/Idrak.Reinforcement.Plugins` (references `Idrak.Reinforcement`, `Idrak.Text`): project, README in
  `Idrak.Vision.Plugins`' layout, `ReinforcementPlugins` (`Package`, `All` — empty until R4 —, `RegisterAll()`,
  `UnregisterAll()`, `RegisterIdrakPlugin()`), not in `LibraryDefaults`; no internals either way (a test).
- The CLI references both and registers the plug-in package at startup with the other three
  (`CommandLine.PluginPackages`; `IDRAK_PLUGIN_PACKAGES` takes `reinforcement`), before `-P` plug-ins.
- The README's package table gets the two rows.

**Done when:** both packages build empty, the contract suites pass on test fakes, `plugins: packages` covers the fourth
package (register twice, unregister as found, `-P` on it, the CLI's order), and the inventory and API dump list them.

**Tests (CPU here):** one build; `reinforcement contracts|plugins:|public API|abstraction inventory|naming:|override loop:|
cli health|conformance kit:`.

**R1 as built (2026-10-22).** As planned, with these specifics:
- **The contracts** (`src/Idrak.Abstraction/Reinforcement/`, namespace `Idrak.Abstraction.Reinforcement`, full XML docs):
  - `Space` (`Spaces.cs`): `SpaceKind` `Discrete`, `Box`, `Text`, `Conversation`; `Space.Discrete(n)`, `Space.Box(shape, low,
    high)` (one bound for all elements or one per element), `Space.Text`, `Space.Conversation`; `Count`, `Shape`, `Size`, `Low`,
    `High`, `Contains(int)`, `Contains(ReadOnlySpan<float>)`, `ContainsValue(object?)` (named apart so a `float[]` argument
    is never ambiguous between the span and the object form), equality by value, `ToString` ("discrete(4)", "box([3], -1 to 1)").
  - R-C1 (`Environments.cs`): `EnvironmentStep<TObservation>` (observation, reward, terminated, truncated, info as
    `IReadOnlyDictionary<string, string>?`, `Done`); a non-generic `IEnvironment : IDisposable` (the two spaces) under the
    generic `IEnvironment<TObservation, TAction>` (`Reset(int seed)`, `ValueTask<EnvironmentStep<T>> StepAsync(action, ct)`).
    The non-generic base exists so a kind can create environments without the registry being generic; `ValueTask`
    because simulations complete synchronously and are stepped very often, while a coding environment awaits. The rules
    (no step before a reset or after an episode ended: `InvalidOperationException`; seeded replay; independent instances;
    cancellation) are written on the interface and checked by the suite.
  - R-C2: `IEnvironmentKind` (`Name`, `Summary`, `Keys`, `ObservationType`, `ActionType`, `IEnvironment Create(options,
    seed)`) and `EnvironmentKinds` (unguarded, reason: a kind creates environments that keep an episode's state between
    steps, so no single call can fall back; no `SetPolicy`), with `Create<TObservation, TAction>(name, options, seed)`, which
    refuses other types naming both.
  - R-C3 (`Policies.cs`): `IPolicy<TObservation, TAction> : IDisposable` with `ActAsync(observations, ActionSampling, ct) →
    ValueTask<PolicyActions<TAction>>` (actions plus `ActionRecord`s), `Update(RolloutBatch, IPolicyLoss, episodes?,
    lossOptions?) → UpdateStats`, `Save`, `Load`. Refined from the plan: `Update` takes the mini-batch as indices into the
    batch (null for all), so a trainer splits a batch without copying it, and the loss's options, as every registry here
    takes options per call. `ActionSampling` (`Greedy`, `Temperature` > 0, `Seed`), `UpdateStats` (loss, episodes, units,
    metrics by name).
  - R-C4 (`Rewards.cs`): `RewardInput(Prompt, Answer, Reference?, Info?)`, `IRewardFunction` (`ScoreAsync(inputs, options,
    ct) → ValueTask<float[]>`), `RewardFunctions` (guarded, shadow compared within 1e-5).
  - R-C5 (`Advantages.cs`): `AdvantageInput`, a `readonly ref struct` over spans (rewards and values per step, lengths,
    truncation flags, final values and groups per episode; checked when made), and `IAdvantageEstimator` (`UsesValues`,
    `Estimate(input, Span<float> advantages, Span<float> returns, options)`): the outputs are written into the caller's
    (pooled) arrays rather than returned, so an iteration allocates nothing; episode ends are per episode (terminated or
    truncated, with the final value a truncated one bootstraps from) rather than a flag per step. `AdvantageEstimators`
    (guarded: the shadowed app writes into pooled arrays of its own and is compared within 1e-5).
  - R-C6 (`PolicyLosses.cs`): `PolicyLossInput` (log-probabilities [units] recorded, counts per sequence, advantages per
    unit, behaviour log-probabilities (null: on-policy, the current ones detached), reference, values and returns
    (together or not at all), entropies, `StepUnits`, `StepSequences`), `IPolicyLoss` (`Loss(input, options) → Tensor`),
    `PolicyLosses` (guarded; a shadowed app's loss is computed under `Autograd.NoGrad`, so only the library's is recorded).
  - R-C7 (`Rollouts.cs`): `ActionRecord` (ids sampled, behaviour log-probability per unit, value or NaN, context ids),
    `Trajectory<TObservation, TAction>` (index, seed, group (its index unless given), one observation more than the
    steps, actions, records, finite rewards, ends, final value, info; `Return` summed in step order), and
    `RolloutBatch<TObservation, TAction> : IDisposable`: episodes sorted by number (duplicates refused), rewards, values,
    advantages, returns, final values, lengths, groups, offsets and truncation in arrays rented from `ArrayPool` and
    returned by `Dispose` (twice harmless; reading after it throws `ObjectDisposedException`); `Input` and `Estimate`.
  - Each registry: `Register` (`[MethodImpl(NoInlining)]`, the calling assembly as origin), `Unregister`, `Names`, `Find`,
    `Get`, `Default`, `Origin`, `Describe`, `SetPolicy` on the guarded three, names ignoring case, a static constructor
    calling `LibraryDefaults.Ensure`; no defaults yet. The "not registered" messages: environment kinds name
    `Idrak.Reinforcement.Plugins` and `ReinforcementPlugins.RegisterAll()`; rewards name `Idrak.Text` and
    `Idrak.Reinforcement.Plugins`; estimators and losses name `Idrak.Reinforcement`.
- **The contract suites** (`src/Idrak.Abstraction.Testing/Contracts/`): `EnvironmentSuite<TObservation, TAction>` (on an
  `IEnvironmentKind`, with actions chosen by the test: types, options refused unless in `Keys`, no step before a reset or
  after an end, observations and actions in their spaces, finite rewards, a seeded replay in the same environment, a new
  one and two at once, cancellation, `Dispose` twice), `RewardFunctionSuite` (on the test's `RewardInput`s: a finite score
  each, none for none, batch equals one by one, the same again and from two calls, options, cancellation, reference),
  `AdvantageEstimatorSuite` (bit for bit again and from four threads, every step written, wrong lengths and a missing critic
  refused, options, reference), `PolicyLossSuite` (one finite element, the same again, the gradient reaches the
  log-probabilities and agrees with central differences where the one-sided differences agree, so a clip's kinks are left
  out; on-policy cases keep the drawn log-probabilities as the behaviour of a perturbed input; options, reference;
  `NeedsReference`, `NeedsValues`). Entry points `Conformance.CheckEnvironmentKind`, `CheckRewardFunction`,
  `CheckAdvantageEstimator`, `CheckPolicyLoss` (the reference defaults to the registry's library default of that name).
- **The packages:** `src/Idrak.Reinforcement` (references `Idrak` only; README; internal `LibraryRegistrations` with an
  empty map; `InternalsVisibleTo Idrak.Tests`), listed in `LibraryDefaults`' assemblies and `DynamicDependency`,
  `Idrak.slnx`, `build/IdrakFromSource.targets`, the API guard and the inventory. `src/Idrak.Reinforcement.Plugins`
  (references `Idrak.Reinforcement` and `Idrak.Text`, no internals either way; README in `Idrak.Vision.Plugins`' layout,
  families table empty; `ReinforcementPlugins` with `Package`, `All` (empty), `RegisterAll`, `UnregisterAll`,
  `RegisterIdrakPlugin`), not in `LibraryDefaults`. The README package tables (root and `src/Idrak/README.md`), the
  Abstraction and testing-kit READMEs, `IdrakPlugins.Get`'s message, the CLI's README and `help plugins` text name them.
- **The CLI:** references both; `CommandLine.PluginPackages` gains `("reinforcement", …)` after audio;
  `IDRAK_PLUGIN_PACKAGES` defaults to `text,vision,audio,reinforcement`. Beyond the plan: `plugins list` (the tool's
  registry snapshot, `Registries.cs`) lists `environments`, `rewards`, `advantages` and `policylosses`, so the fourth
  package's registrations are covered by the "register twice, unregister as found" check as the others' are.
- **Tests:** `reinforcement contracts` (7, `tests/Idrak.Tests/ReinforcementContractTests.cs`): spaces and data types; each
  suite passed by a fake (a seeded corridor, a length-budget reward, discounted returns with and without a baseline, a
  PPO-style clipped loss with a k3 KL and a value term) and failed by one broken on purpose (shared random draws, a
  batch-relative reward, a noisy estimator, a detached loss, a shifted reference); every registry's message, origin,
  case-insensitive lookup and unregister; Throw, FallBack and Shadow over a library default registered by the test (once
  per process, named `reinforcement-test-guarded`); `Idrak.Reinforcement` referencing `Idrak` alone. `plugins: packages`
  covers the fourth package (register twice, unregister as found, `-P` on its DLL, the tool's order, the variable, no
  internals, `Idrak.Reinforcement` in `LibraryDefaults` and the plug-in package not, the `EnvironmentKinds` and
  `RewardFunctions` messages). The inventory, API guard and IdrakFromSource lists include both packages; `api/
  Idrak.Reinforcement.txt` (empty surface) and `api/Idrak.Reinforcement.Plugins.txt` are new. No line in
  `shared-contracts.txt` was needed: the inventory places the contracts in `Idrak.Abstraction` (the registries there use
  them).
- **Run (CPU, the agent):** one build (no warnings); `IDRAK_UPDATE_API=1` (`public API`) and `IDRAK_UPDATE_INVENTORY=1`
  (`abstraction inventory: plans`) regenerated the dumps; then `reinforcement contracts|plugins:|public API|abstraction
  inventory|naming:|override loop:|cli health|conformance kit:`: 90 passed, 0 failed. R1 has no GPU code: there are no
  CUDA filters for it.
- **The owner's commands** (from `D:\Projects\Idrak`):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "reinforcement contracts|plugins:|public API|abstraction inventory|naming:|override loop:|cli health|conformance kit:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

## Phase R2: `Idrak.Reinforcement`, the general part

- **Rollouts:** `RolloutCollector` runs N environments at once (`IDRAK_RL_ENVIRONMENTS`, default from the cores), asks the
  policy for a batch of actions, steps the environments in parallel (rule 79, measured: one thread per environment only
  where it paid), and records trajectories into pooled buffers that do not grow between iterations (rule 77). Episodes
  are numbered and sorted by number before use, so thread timing changes nothing.
- **Advantages (R-C5) and losses (R-C6):** the defaults listed above, each with its paper and its formula in the doc
  comment, written as tensor operations (autograd), sums in a fixed order. KL estimators k1 and k3 as a `grpo` and `ppo`
  option, not a fixed choice.
- **The trainer:** `ReinforcementTrainer.Train(environmentKind, policy, ReinforcementOptions, progress, ct)`: collect,
  estimate, then `Epochs` passes of `MiniBatch`-sized updates (`policy.Update`), clip of the gradient norm, a learning-rate
  schedule from M2; evaluation episodes every `EvaluateEvery` iterations (greedy, a seed of their own); checkpoints with
  resume (policy, optimizer, the run's random state, the iteration), written through the library's checkpoint helpers;
  stops cleanly on cancellation between iterations.
- **`ModulePolicy`:** a policy over a `Module` (categorical over `Discrete`, Gaussian over `Box`), with an optional value
  head; `Update` is autograd plus an optimizer from M1. It is the general policy for non-text environments and for the
  tests.
- **Options (`ReinforcementOptions`):** algorithm (a `PolicyLosses` name), estimator, group size, γ, λ, clip, KL weight
  and estimator, entropy weight, value weight, epochs, mini-batch, environments in parallel, iterations, seed,
  evaluate and save intervals, resume folder; each with an `IDRAK_RL_*` variable the CLI reads (R5).
- **Telemetry:** the library's counters (episodes, steps, reward mean and spread, pass rate, KL, clip fraction, update
  time, rollout time), free when nobody listens (rule 78); progress through `IProgressReporter`.

**Done when:** on test fakes (a two-armed bandit and a short corridor written in `tests/Idrak.Tests`, not shipped),
REINFORCE, PPO and GRPO raise the mean reward to the optimum within a seeded number of iterations; a seeded run gives
the same rewards, advantages and weights twice and on 1 and 4 threads; a resumed run matches an uninterrupted one bit
for bit; memory held after iteration 2 does not grow by iteration 20.

**Tests (CPU here):** one build; `reinforcement:|reinforcement contracts|public API|abstraction inventory|
cli health: every environment variable`.

**R2 as built (2026-10-22).** `src/Idrak.Reinforcement` (references `Idrak` only), no model family, environment or language.
- **Estimators** (`Estimators.cs`, registered as `AdvantageEstimators` library defaults by `LibraryRegistrations`, in this order):
  `gae` (γ 0.99, λ 0.95, `normalize`; TD errors summed backwards, the truncated episode bootstrapped from its final value,
  returns A + V), `group` (an episode's return less its group's mean, over the sample deviation + ε unless `scale=false`;
  ε 1e-4; a group of one, or equal returns with ε 0, gives 0), `leave-one-out` (R_i − mean of the others), `returns` (G_t
  discounted, `baseline` none, mean or value, `normalize`). Group members are found by sorting (group, episode) keys in a
  pooled array, so each group's statistics are computed once, in episode order; every sum is in double, in a fixed order,
  rounded once. The group estimators write each step's reward-to-go as its return. Papers and formulas in the XML docs.
- **Losses** (`Losses.cs`, `PolicyLosses` defaults): `reinforce` (−A log p, `entropy`, `value` 0.5 when values are given,
  `normalize`, `constant`), `ppo` and `grpo` (one class, `ClippedRatioLoss`: min(ρA, clip(ρ, 1 − ε, 1 + ε_high)A), `clip`
  0.2, `clip-high` (DAPO's decoupled upper clip), `kl` with `kl-estimator` k1, k2 or k3, `entropy`; ppo adds `value` 0.5 and
  normalizes per token, grpo per sequence; both take `normalize` token, sequence or constant with `constant`). Each unit's
  term is weighted by the step's normalization (a weight tensor) and summed by one reduction. **Choice: grpo's `kl` defaults to
  0** (DAPO, TRL's GRPOTrainer), the paper's 0.04 is `kl=0.04`; a β above 0 without reference log-probabilities is refused
  naming `kl`, so nothing needs a reference unless asked.
- **`RolloutCollector<TObservation, TAction>`:** all of a rollout's episodes at once, one environment per episode slot (made
  once, reused by every later rollout and the evaluations); each round one `ActAsync` for every episode under way, in slot
  order, with a seed derived from the sampling seed and the round; then the environments stepped sequentially or through
  `Parallel.ForEachAsync` (`Environments` at once). With `ParallelSteps` null it times the first rounds both ways (a warm-up
  round, then two sequential and two parallel, per step) and keeps the faster (rule 79). Scratch lists per slot are cleared
  and reused; a finished episode is copied out at its exact size. A truncated (or `maxSteps`-cut) episode with values gets the
  value of its last observation from one more batched `ActAsync`. **Deviation from the plan's wording:** "N environments at
  once" became "every episode of the rollout at once, N stepped at once": if the policy's batch were N wide its draws would
  depend on N, and the default N comes from the cores, so a seed would replay differently on another machine; this way the
  results do not depend on N or on the threads (tested), at the cost of an environment per episode slot.
- **`ReinforcementTrainer<TObservation, TAction>`** (and a static `ReinforcementTrainer.Train` by kind or kind name):
  collect, estimate (`AdvantageEstimators`; the algorithm's own when none is named: grpo group, ppo gae or returns when the
  policy has no critic, reinforce returns), `Epochs` (ppo 4, else 1) passes of `MiniBatch`-episode updates, each pass
  shuffled from its own seed; evaluation every `EvaluateEvery` iterations (greedy, `EvaluationEpisodes` from seeds of their
  own, the same each time); checkpoints `iteration-NNNNNN/` (the policy's `Save` folder and `state.json`: iteration, seed,
  updates, algorithm, environment), written into a `.writing` folder and moved whole, every `SaveEvery` iterations and at
  the end or a stop; `Resume` takes a checkpoint or the folder of them (its latest), loads the policy and replays the
  schedule's steps; a cancelled token stops between iterations (an iteration under way is dropped before its updates) with a
  checkpoint. **The run's random state is its seed and where it is:** every draw derives from SplitMix64 over (seed, stream,
  iteration, group or round or pass), so a resume needs no generator state. Episodes of a group share their environment's
  seed. `OnIteration` and `OnBatch` hooks; `IterationSummary` (returns, mean, spread, min, max, loss, updates, rate,
  evaluation, times, the policy's metrics averaged and the share that terminated); `ReinforcementResult`.
- **Contract added (in `Idrak.Reinforcement.Abstractions`, not in R1's):** `IScheduledPolicy` (`Optimizer`,
  `MaxGradientNorm`, `Device`). The trainer clips and schedules (a `LearningRateSchedules` schedule over the run's updates,
  `ScheduleSettings` with no warm-up by default) only a policy that exposes its optimizer; `IPolicy` has no way to reach one,
  and R3's language-model policy schedules through its fine-tuning session (it cannot implement this one, as `Idrak.Text`
  does not reference `Idrak.Reinforcement`). **No R1 contract was changed.**
- **`ModulePolicy`:** `ModulePolicy.Categorical` and `.Gaussian` make a `ModulePolicy<int>` or `<float[]>` over float
  observations (a box): an actor (logits, or means with a learned state-independent log std per element), an optional critic
  [batch, 1], an optional frozen reference (a module of the actor's shape that takes its starting weights, for a KL).
  Acting is one forward under `NoGrad`, draws row by row from the call's seed (the behaviour log-probability is that of the
  distribution sampled, at its temperature; a Gaussian action is clipped to the box and its density taken there, on acting
  and on updating). `Update` gathers the mini-batch's units into pooled arrays, recomputes log-probabilities, entropies and
  values with autograd, the reference's under `NoGrad`, calls the loss, clips (`ClipGradientNorm`) and steps an optimizer made
  through `OptimizerKinds` (`ModulePolicyOptions`: adam, 3e-4); metrics: k3 KL to the behaviour, clip fraction, entropy, KL
  to the reference, gradient norm. `Save`/`Load` a folder (.ikw weights, the optimizer state as safetensors, a JSON kind
  check). `ModulePolicy.Perceptron` builds a small tanh network.
- **Settings:** `ReinforcementOptions.FromEnvironment` and `ModulePolicyOptions.FromEnvironment`; 33 variables, all in
  `src/Idrak.Cli/Shared/EnvironmentVariables.cs` (group "Reinforcement learning", command `test` until R5 adds `reinforce`):
  `IDRAK_RL_ALGORITHM`, `_ESTIMATOR`, `_GROUP`, `_EPISODES`, `_ENVIRONMENTS`, `_PARALLEL_STEPS`, `_ITERATIONS`, `_EPOCHS`,
  `_MINIBATCH`, `_GAMMA`, `_LAMBDA`, `_GROUP_SCALE`, `_NORMALIZE_ADVANTAGES`, `_CLIP`, `_KL`, `_KL_ESTIMATOR`, `_ENTROPY`,
  `_VALUE`, `_NORMALIZE`, `_MAX_GRAD_NORM`, `_SCHEDULE`, `_WARMUP`, `_TEMPERATURE`, `_MAX_STEPS`, `_SEED`,
  `_EVALUATE_EVERY`, `_EVALUATION_EPISODES`, `_SAVE_EVERY`, `_CHECKPOINTS`, `_RESUME`, `_LOG`, `_OPTIMIZER`,
  `_LEARNING_RATE`. `IDRAK_RL_LOG` (`Log`) appends a JSON line per iteration.
- **Progress and telemetry:** `IProgressReporter` (an iteration a unit; detail: reward mean ± spread, evaluation,
  episodes a second); the library's telemetry: `TrainingStarted`, a `BatchCompleted` per update (with the gradient norm),
  an `EpochCompleted` per iteration (metrics: episodes, steps, reward mean, spread, min, max, kl, clip-fraction, entropy,
  gradient-norm, terminated, rollout and update seconds; the evaluation as its validation metric), `TrainingCompleted`,
  each built only when a hook listens (rule 78). **Not done:** a "pass rate" counter: what passes is the environment's
  (R3's and R4's rewards and info), so the general trainer reports the share that terminated instead.
- **Inventory:** with `Idrak.Reinforcement` the only library user of R1's contracts in this tree, decision 10 would move
  them into it; `tests/Idrak.Tests/data/shared-contracts.txt` gained lines for `IEnvironment`, `IEnvironment<,>`,
  `IEnvironmentKind`, `EnvironmentKinds`, `IPolicy<,>`, `IAdvantageEstimator`, `AdvantageEstimators`, `IPolicyLoss`,
  `PolicyLosses` and `IProgressReporter` (the first library user of the interface type itself). **After R3's merge**, the
  lines of those `Idrak.Text` uses too become stale and the inventory test asks for their removal (the list only shrinks).
  `api/Idrak.Reinforcement.txt` and `plans/10-abstraction-inventory.md` regenerated.
- **Tests** (`tests/Idrak.Tests/ReinforcementTests.cs`, group `reinforcement:`, 11; fakes: a two-armed bandit with common
  draws per group, a 5-cell corridor (optimum 0.97), a one-element box target): every estimator and loss passes R1's suites
  (with option sets: gae with γ, λ and normalize, group without scale, returns with a mean baseline; reinforce by a constant,
  ppo with values, entropy, a k3 KL and clip-high, grpo with k1 and k3 KLs and per token) and matches a direct computation
  on fixed inputs (GAE summed forwards, not by the recursion); the defaults, order and messages; REINFORCE, PPO and GRPO take
  the bandit to the best arm at p ≥ 0.95 (iterations: reinforce 6, ppo 1, grpo 6 of 100) and the corridor to its optimum
  greedy return (2, 2, 2 of 150); a Gaussian policy's first ratio is 1 and its mean reaches 1 ± 0.1 (19 of 150); a seeded
  PPO run gives the same rewards, advantages and weights bit for bit twice, on 1 and on 4 threads, sequential, in parallel
  and measured, and GRPO in groups of 4 sequential and parallel; a run stopped after 3 of 6 iterations and resumed into
  a policy of other starting weights matches the uninterrupted one bit for bit (rewards, advantages, weights, the linear
  schedule's rate); over iterations 2 to 20 device memory in use stays 3,888 bytes, the managed heap 17,141 → 17,128 KB, and
  16 environments are made once; the collector's order, seeds, groups, cut and bootstrapped episodes, reuse and measuring; a
  cancelled run's checkpoint; progress, telemetry and the JSON log; the variables.
- **The run (CPU, this agent):** one build (no warnings), `IDRAK_UPDATE_API=1` (`public API`) and `IDRAK_UPDATE_INVENTORY=1`
  (`abstraction inventory: plans`), then `reinforcement:|reinforcement contracts|public API|abstraction inventory|cli
  health: every environment variable|cli polish: the environment table|naming:|override loop:|conformance kit:|plugins:
  packages`: 48 passed, 3 failed, all three in the tests' own setup (the bandit's PPO reached the target in its first
  iteration, so first and last iteration were the same one; the "stopped" run had 3 iterations, so its linear schedule
  differed from the 6-iteration run's, where a stop is a cancellation of the same run; the environment count included the
  one the test makes to read the spaces); fixed in the tests, one rebuild, the three rerun: 3 passed. Total 51 of 51.
- **The owner's commands** (from `D:\Projects\Idrak`; R2 is CPU code, there are no CUDA filters for it):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "reinforcement:|reinforcement contracts|public API|abstraction inventory|cli health: every environment variable|cli polish: the environment table|naming:|override loop:|plugins: packages"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```

## Phase R3: the language-model policy in `Idrak.Text`

- **`FineTuningSession`** (public): opened once with a model and `FineTuningOptions` (adapters, optimizer, schedule, FP8,
  cluster, device memory), it holds what `TrainCore` holds today across calls: `Step(sequences, loss) → step stats`,
  `LogProbabilities(sequences, adapters: on | off)` (no gradient; for the behaviour and reference values), `Generator(…)`
  (a `ChatGenerator` on the live adapters), `Save`, `Checkpoint` and resume, `Dispose`. `FineTuner.Train` becomes a loop
  over a session, its results unchanged bit for bit (rule 71). The generation KV cache and the training activations
  share the device pool; the cache is released before an update and the activations before a rollout, measured so the
  device's peak stays one of the two, not both (rule 77).
- **Sampled tokens recorded:** `GenerationOptions.ReturnTokens` (off by default, so serving pays nothing) makes each reply's
  last `ChatChunk` carry the prompt ids it rendered and the reply ids sampled (a new optional record `ChatTokens`), and
  their log-probabilities when asked. A trajectory is trained on exactly the ids sampled, never on a re-encoding of the
  text.
- **Multi-turn episodes:** each assistant turn's prompt and reply ids are recorded. When the chat template renders turn n+1's
  prompt as turn n's prompt plus its reply (checked at run time on the episode's ids), the episode trains as one sequence
  with only the assistant tokens marked trained; otherwise each turn trains on its own prompt (a template that drops
  earlier reasoning, for example). The choice is automatic and a setting (`IDRAK_RL_TURNS`: `auto`, `episode`, `turn`).
- **`FineTuningLossInput`** gains per-sequence values (`SequenceValues`, the advantages) and `BehaviorLogProbabilities`;
  `FineTuningLosses.Policy(IPolicyLoss)` turns a registered policy loss into a `FineTuningLoss`.
- **`LanguageModelPolicy : IPolicy<Conversation, ChatMessage>`:** acts through `ChatBatch` on the session's generator
  (`ReturnTokens` on); updates through `session.Step` with the policy loss; the reference is the base model with the
  adapters off (as DPO's); the behaviour log-probabilities are recomputed once per rollout batch by the training forward
  (so the first ratio is exactly 1), or taken from the sampler's (a setting: `IDRAK_RL_BEHAVIOR`: `recompute`, `sampler`).
  Rollouts on a cluster are spread over the ranks and gathered in rank order.
- **The `prompts` environment** (`EnvironmentKinds`, registered by `Idrak.Text`): chat prompts with references and
  optional tools, from Idrak.Data's chat-transcript readers; one turn, or several when the prompt offers tools; rewarded
  by a `RewardFunctions` expression with weights (`--reward exact` or `exact:1,length:0.1`).
- **Text rewards (R-C4)** registered by `Idrak.Text`: `metric` (any `TuningMetrics` name, after M3), `regex` (the answer
  matches a pattern: a final-answer tag, a format), `json` (the answer parses, through the library's `JsonRepairs`),
  `length` (a penalty past a token budget), `judge` (a judge model's probability of "yes" through `AnswerScorer`, any model
  the library loads), `sum` (weighted).
- **On-policy distillation (the owner, 2026-10-22)** (Agarwal et al. 2024, "On-Policy Distillation of Language Models:
  Learning from Self-Generated Mistakes", arXiv:2306.13649). The student generates its own episodes in any environment
  (`prompts`, or R4's multi-turn `coding` episodes with tool calls), and the teacher's next-token distributions are
  computed on exactly the ids the student sampled, so the student learns where its own outputs go wrong, not only on
  answers someone else wrote (`idrak distill` today).
  - **The teacher** is a live `DistillationTeacher` (`DistillationTeachers.FromModel`, exists): frozen, no gradients, in
    its own precision (int8 or int4 allowed), on the device the app loads it on, so teacher and student can sit on one
    device or on two (measured, never assumed). A precomputed teacher file (`FromFile`) cannot cover new samples and is
    refused with that reason. Teacher and student share the vocabulary (`DistillationTeacher.Check`, the fingerprint);
    otherwise refused, naming `idrak distill --generate` (answers written by the teacher) as the way across tokenizers.
  - **Only the assistant's tokens train:** prompts, tool calls' results and the environment's messages never do, so an
    agentic-coding episode is distilled turn by turn as the multi-turn rule above lays it out.
  - **Divergences** as a registry, `TeacherDivergences` (`Idrak.Text.Abstractions`): `forward` (KL(q ‖ p), today's
    `FineTuningLossInput.TeacherDivergence`), `reverse` (KL(p ‖ q), mode-seeking, the usual on-policy choice), `jsd`
    (generalized Jensen-Shannon with β); `TeacherDivergence` takes the kind. Each costs one more pass of the student's and
    the teacher's heads over the trained rows; the `[tokens, vocabulary]` logits are never stored.
  - **Two uses.** (1) Pure distillation: `FineTuningLosses.OnPolicyDistillation(divergence, temperature, β)`, no reward
    and no advantage; a share λ of each step's batches comes from the dataset's own answers instead of the student's
    samples (the paper's λ, a setting). (2) With reinforcement learning: a teacher term added to the policy loss with a
    weight, a dense per-token signal beside the environment's sparse reward. For agentic coding, where most early
    episodes fail the tests and give no signal, the teacher's per-token guidance is what moves the student; the tests'
    reward takes over as the student improves (the weight can follow a schedule from M2). The KL to the reference in
    `grpo` and `ppo` stays as it is; the teacher term is separate.
  - **Settings:** the teacher (model and device), the divergence, β, the temperature, λ, the teacher weight; variables
    `IDRAK_RL_TEACHER_WEIGHT`, `IDRAK_RL_DIVERGENCE`, `IDRAK_RL_DISTILL_MIX` (names proposed, as R5's).

**Done when:** `FineTuner.Train`'s tests pass unchanged with the same losses; on a tiny decoder made in the test (no
download), GRPO on a `prompts` task with a `metric:exact` reward raises the pass rate within a seeded number of
iterations; the trained ids equal the sampled ids; a multi-turn episode picks `episode` on a prefix-stable template and
`turn` on one that drops reasoning; peak device memory over 20 iterations does not grow (CPU here; CUDA the owner's).
On-policy distillation: with the teacher equal to the student every divergence is 0 and the weights do not move; with a
stronger tiny teacher trained in the test, the student's reverse KL on its own samples falls within a seeded number of
iterations, alone and as a term beside a reward; `forward`, `reverse` and `jsd` match a direct computation on fixed
logits; a multi-turn episode trains its assistant tokens only; a vocabulary mismatch and a teacher file are refused with
their messages.

**Tests (CPU here):** one build; `reinforcement text:|distillation:|fine-tuning:|conversation tuning:|chat templates:|generation|
coding agent:|evaluation:|tuning metrics|reinforcement:|public API|abstraction inventory|cli health: every environment
variable`. **Owner (CUDA):** the same `reinforcement text:|fine-tuning:` on `cuda`, and the GRPO run of R5 on a small real
model.

**R3 as built (2026-10-22).** Items 1-7 built in one change; `Idrak.Text` still references no `Idrak.Reinforcement` (the policy
meets its trainer and losses only through `Idrak.Abstraction.Reinforcement`). R2's losses and estimators were written beside
this phase, so the policy is proven here with test-only fakes (a clipped-ratio loss registered from the test into
`PolicyLosses`, a group-normalized advantage and an update loop written in the test); **the end-to-end run with R2's real
trainer and `grpo` is R4's first item, after both merge.**
- **`FineTuningSession`** (`src/Idrak.Text/FineTuningSession.cs`, public): opened with a model, `FineTuningOptions` and the
  schedule's `totalSteps` (an optional sample of sequences decides the image side and the FP8 check; without one FP8 is checked
  on the first step's sequences). It holds what `TrainCore` held: adapters (resumed from `ResumeFrom`), trained parameters, FP8
  copies, the optimizer, the cluster's sync, the schedule and the step runner with its recorded graphs. Members: `Step(sequences,
  loss?) → FineTuningStepStats` (one optimizer step over all the sequences' batches, packed as `Packing` says; null loss: the
  options' own, the fused cross-entropy when none), `LogProbabilities(sequences, adapters: on | off)` (no gradient, the network in
  evaluation mode for the call, the batches a step would make), `Generator(context, cache format)` (a `ChatGenerator` on the live
  adapters, `KeepCache` off; the session releases every generator's cache before a step and its log-probabilities), `Save`,
  `Checkpoint` (adapters, optimizer state, steps: the stop-checkpoint format, so `ResumeFrom` reads it), `Restore` (into a session
  that has not passed the checkpoint; the schedule moves on to it), `Steps`, `TotalSteps`, `ResumedSteps`, `LearningRate`,
  `Settings`, `Dispose` (the reverse order of `TrainCore`'s usings). **`FineTuner.Train` is a loop over a session**: `TrainCore`
  opens one (its internal constructor runs `TrainCore`'s old set-up in the same order), makes the epochs' batches from its own
  seeded `Random` as before, then `Begin` (schedule and runner with the run's batch sizes) and `Run`/`Skip` per step; the stop
  checkpoint is `session.Checkpoint`. The step runner takes its loss of one's own per step (`StepRunner.Custom`; graphs only for
  the fused loss, as before), and a session's loss of one's own drops the cached reference values of sequences it no longer trains
  (`CustomLoss.Keep`), so a run that brings new sequences each step holds one step's.
- **Results unchanged, proven (rule 71):** a scratch program (not checked in) ran `FineTuner.Train` on seven fixed cases with the
  DLLs built before the change and with the new ones, the test project's own tiny model writer and sequences called by
  reflection: the packed fused loss, gradient accumulation, a loss delegate, padded batches with evaluations, DPO pairs with
  evaluations, distillation from a teacher model, DoRA with SGD and a linear schedule. **Every step's loss, every evaluation loss
  and a hash of every trained adapter value are the same bits** (and a second run of the old build gave the same bits, so the
  comparison means something). In the suite, `reinforcement text: FineTuner.Train is a loop over a session` checks that Train's
  losses step by step and its adapters equal, bit for bit, a session stepped by hand (the fused loss, `Loss` set, a loss given per
  step; one step per epoch so the batches are the same draws). The existing `fine-tuning:`, `fine-tuning options:`,
  `distillation:`, `conversation tuning:`, `vision tuning` and `pretrain:` tests pass with unchanged expectations.
- **Sampled tokens:** `GenerationOptions.ReturnTokens` and `ReturnLogProbabilities` (off by default: serving copies nothing);
  `ChatTokens(Prompt, Reply, LogProbabilities?)` (`Idrak.Abstraction.Generation`) on the last `ChatChunk`, `GenerationChunk` and
  `BatchChunk` (`Tokens`, an init property, so every constructor stays as it was). The core generator records the prompt as fed
  (after its cut to the window), each sampled id and log(probability) as the sampler drew it (after temperature and filters), and
  cuts the reply at the token that completes the stop sequence (the shortest run of ids whose decoding holds the stop, by
  bisection; a decoded length alone is not monotone when a byte-level reply ends inside a character), on the single and the
  batched paths and their fall-backs; `ChatBatch` now reads `StreamBatch` (the same texts as `GenerateBatch`), `StreamBatch` (both
  paths) and `Stream` pass the ids on. A test checks the ids and log-probabilities against a recording sampler on `Chat`,
  `Stream`, `ChatBatch` and `StreamBatch`, the stop cut, and the same texts with the option off.
- **Multi-turn episodes** (`EpisodeTurns`, `Idrak.Text.Reinforcement`): `PrefixStable(records)` on the ids, `Lay(records, mode)`
  → `EpisodeLayout(Mode, Sequences)`: `episode` (the last prompt and reply, every reply's ids trained) when stable, `turn` (each
  reply on its own prompt) otherwise; `auto` picks; `episode` on unstable ids is refused naming `turn`/`auto`;
  `IDRAK_RL_TURNS`. Tested on ChatML ids (stable) and Qwen3's template, which drops earlier reasoning (not stable), and on a
  two-turn run of the policy on the tiny model, whose layout follows what its ids say and whose trained ids are the replies'.
- **`FineTuningLossInput`**: `SequenceValues`, `TokenValues`, `BehaviorLogProbabilities` / `HasBehavior`, from the sequences
  (`TrainingSequence` gained the init properties `Value`, `TokenValues`, `BehaviorLogProbabilities`, `ReferenceLogProbabilities`,
  not part of its equality) or given to the public constructor (new optional parameters); `ReferenceLogProbabilities` is taken
  from the sequences when every one carries it (no reference pass then). **`FineTuningLosses.Policy(IPolicyLoss, options,
  reference = true)`** adapts a registered policy loss (units = trained tokens, counts per sequence, step sizes). Deviation: the
  adapter takes `reference` as a flag because R1's `IPolicyLoss` does not say whether a loss reads the reference (see "For R2").
- **`LanguageModelPolicy : IPolicy<ChatObservation, ChatMessage>`** (`src/Idrak.Text/Reinforcement/LanguageModelPolicy.cs`). R1
  defined no conversation type, so **`ChatObservation(Messages, Tools?, Think?)`** is new in `Idrak.Abstraction.Generation`
  (value equality, so the environment suite's replays compare conversations; R4's coding environment uses it too). It acts through
  `ChatBatch` on the session's generator with `ReturnTokens` on, sampling at the call's temperature and seed with no top-k, top-p,
  min-p or penalty (greedy: top-k 1), records `ActionRecord(reply ids, sampler log-probabilities, NaN, prompt ids)` and refuses a
  reply whose window slid. `Update` lays out the whole rollout batch on its first update (one per batch): the sequences, each
  token's advantage (its turn's step), the reference (`LogProbabilities`, adapters off) and the behaviour (recomputed by the
  training forward before any step on the batch, or the sampler's: `IDRAK_RL_BEHAVIOR` `recompute` | `sampler`), then
  `session.Step` with the policy loss (and the teacher term). `Lay(batch)` shows what it trains; `Save`/`Load` are
  `Checkpoint`/`Restore`. Options: `MaxNewTokens`, `ContextLength`, `Generation`, `Think`, `Turns`, `Behavior`, `Reference`,
  `TeacherWeight`, `Divergence`, `Beta`, `TeacherTemperature`. Measured on CPU: the first update's |log p − behaviour| is exactly 0
  (ratios exactly 1).
- **The `prompts` environment** (`PromptsEnvironmentKind`, registered by `Idrak.Text`'s `LibraryRegistrations`): tasks from
  `data` (a chat-transcript JSON-lines file, the last assistant message the reference) or given in code (`PromptTask`), one drawn
  per episode from its seed; options `reward` (a reward or metric name, or `exact:1,length:0.1`), `rounds`, and each term's
  options as `term.key`; a kind made with an `IToolRegistry` runs a reply's calls for up to `rounds` replies (the last one
  truncated when it still calls tools); one made with a tokenizer gives each answer's token count in the info (`tokens`).
- **Text rewards** (`TextRewards`, registered by `Idrak.Text`): `metric` (any `TuningMetrics` name, `exact` by default; 1 − value
  for an error rate; refuses an input without a reference), `regex` (`pattern`, `whole`, `ignore_case`; a matcher per thread),
  `json` (1 as written, `repaired` 0.5 after `JsonRepairs`' repair, 0 none), `length` (−`penalty` per `unit` past `budget`; tokens
  from the info, or words, characters), `judge` (`JudgeReward`: P(yes) against no through `AnswerScorer.Choose`; the library's
  loads `model` on `device` once per process, `Release` frees it; an app's is made with its scorer; **the question and the two
  words are required options**, so no language is assumed), `sum` (`terms`, weighted, in order). All pass the reward suite.
- **On-policy distillation:** registry **`TeacherDivergences`** (`Idrak.Text.Abstractions`, `ITeacherDivergence`): `forward`,
  `reverse`, `jsd` (β in (0, 1), as TRL's GKD), each its value and its gradient with respect to the softened logits, run chunk by
  chunk through the existing kernels: core's `Tensor.TokenDivergences` / `TokenDivergenceRows` gained an optional
  `TokenDivergenceRule` (null keeps the old forward path, so today's distillation is unchanged; the library's `forward` by name
  takes that path, checked bit for bit). `FineTuningLossInput.TeacherDivergence(temperature, divergence, beta)` beside the old
  overload; **`FineTuningLosses.OnPolicyDistillation(divergence = reverse, temperature, beta)`** (T² · mean) and
  `FineTuningLosses.Add(first, second, weight)` (the teacher term beside a policy loss); **`DistillationTeachers.CheckOnPolicy`**:
  a teacher file refused with its reason (it cannot cover new samples), another vocabulary refused naming `idrak distill
  --generate`; **`OnPolicyDistiller.Train(session, data, options)`**: the student samples, a share λ (`Mix`,
  `IDRAK_RL_DISTILL_MIX`) of each step's prompts train on the data's answers instead; only assistant tokens train (the replies'
  ids, the encoder's assistant spans). Beside a reward: `LanguageModelPolicyOptions.TeacherWeight` (`IDRAK_RL_TEACHER_WEIGHT`),
  `Divergence` (`IDRAK_RL_DIVERGENCE`). Tested: forward, reverse and jsd against a double-precision computation (values, and the
  hidden gradient against central differences of the divergence itself); a teacher equal to the student gives divergences under
  1e-6 and moves the weights by under 1e-6 (SGD, so a tiny gradient is not scaled up as Adam would); a stronger tiny teacher
  (tuned in the test) lowers the student's reverse KL on its own samples (a Monte Carlo estimate on 16 samples) below 0.7 of its
  start alone, with λ = 0.5, and as a term beside a reward.
- **Variables** (in `EnvironmentVariables.cs`, group "Reinforcement learning", commands `tune`, `distill` until R5 adds
  `reinforce`): `IDRAK_RL_TURNS`, `IDRAK_RL_BEHAVIOR`, `IDRAK_RL_TEACHER_WEIGHT`, `IDRAK_RL_DIVERGENCE`, `IDRAK_RL_DISTILL_MIX`;
  each option reads its variable when set, a value given in code wins.
- **Inventory:** R1's contracts are now used by `Idrak.Text` (and, after R2, by `Idrak.Reinforcement`); alone in this branch the
  inventory would move them to `Idrak.Text`, so nine lines in `tests/Idrak.Tests/data/shared-contracts.txt` keep them in
  `Idrak.Abstraction` (decision 6). **At the merge with R2, the inventory test names the lines that became stale (several packages
  use them then); delete those** and regenerate the inventory. `api/Idrak.Abstraction.txt`, `Idrak.Text.txt`, `Idrak.txt` and
  `plans/10-abstraction-inventory.md` regenerated.
- **For R2 (no R1 contract changed here):** `IPolicyLoss` could say whether a loss reads the reference (a `NeedsReference`
  member, as the suite already has), so `FineTuningLosses.Policy` and the policy would skip the reference pass on their own;
  `PolicyLossInput` has no per-sequence values, which is why the policy repeats a turn's advantage over its tokens.
- **Left:** rollouts on a cluster spread over the ranks (the policy refuses a session with a cluster, naming it); the CUDA
  measurement that the device's peak holds the cache or the activations, not both (the CPU test checks that held memory does
  not grow over 20 iterations); `idrak reinforce` and `idrak distill --on-policy` are R5's.
- **Tests:** `tests/Idrak.Tests/ReinforcementTextTests.cs`, group `reinforcement text:` (9). The run (CPU, one build plus one
  rebuild for the failures): `reinforcement text:|distillation:|fine-tuning:|fine-tuning options:|conversation tuning:|chat
  templates:|generation|coding agent:|evaluation:|tuning metrics:|vision tuning|reinforcement contracts|public API|abstraction
  inventory|cli health: every environment variable|naming:|pretrain:|scoring:|decoding|cli polish|plugins: packages`: 174 passed,
  4 failed. Fixed: the inventory placement (above), the environment suite's replays (`ChatObservation`'s value equality), the GRPO
  test's bar (its pass rate rose from 0.06 to about 0.6 with one epoch a rollout; now two epochs and a bar of a rise of 0.3 to at
  least 0.6). Rerun `reinforcement text:|abstraction inventory|public API`: all pass (the inventory regenerated once more for the
  shared lines). Not mine: `t5 real:` (caught by `generation`) needs the owner's real references (`make_tiny_t5.py --real`).
- **The owner's commands** (repo at `D:\Projects\Idrak`):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "reinforcement text:|fine-tuning:|fine-tuning options:|distillation:|conversation tuning:|generation:|decoding|scoring:|pretrain:|abstraction inventory|public API|cli health: every environment variable"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "reinforcement text:|fine-tuning:|fine-tuning options:|distillation:|conversation tuning:|generation:|decoding|scoring:|pretrain:|vision tuning"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```
  On CUDA, the GRPO test's "first ratios exactly 1" says whether a recomputed forward gives the step's values bit for bit on
  the card; a failure there is a finding (the behaviour pass and the step pack the sequences differently), not a setting.

**R3b as built (2026-10-22).** R3's leftover: the language-model policy's rollouts on a cluster. `Idrak.Text` still references
no `Idrak.Reinforcement`; the two meet in one new contract in `Idrak.Abstraction.Reinforcement`.
- **Why shards (the design's root).** The sampler draws each token from a counter of (seed, step, **row of the batch**)
  (`TokenSampler`, the same on CPU and GPU), so a reply depends on which other episodes share its `ActAsync` call and at which
  row. Spreading a rollout's episodes over W ranks therefore cannot replay a one-process run that acts on all of them in one
  call. Per-row seeds would mean a new sampling kernel on every backend (CPU, CUDA PTX, Vulkan); not taken. Instead the
  collector cuts a rollout into **shards**: a shard's episodes under way act together in one call a round, from a seed of the
  shard's own (`Seeds.Derive(seed, sampling, round, shard)`; shard 0's is the round's old seed). A shard replays alike wherever
  it is played, so **a seeded run with the same shard count gives the same episodes on any number of ranks**, one process
  included.
- **The split (`RolloutSpread`, `Idrak.Reinforcement`, public):** every rank computes it from the episodes' numbers, the
  shard count and the rank count alone (no message). `Shards(numbers, S, spread)`: the episodes ranked by number, then
  `blocks` (the default: runs in number order, sizes within one, so a group's episodes, which share a prompt, act together)
  or `interleaved` (episode k in shard k mod S); duplicates refused. `Rank(s, S, W) = ⌊s·W/S⌋`: runs of shards, shard r on
  rank r when S = W, S/W shards a rank when S is a multiple. Settings: `ReinforcementOptions.Shards` (`IDRAK_RL_SHARDS`, 0 =
  one per rank, so **one process keeps one shard and acts exactly as before**: R2's, R3's, R4's and R5's tests pass
  unchanged) and `ReinforcementOptions.Spread` (`IDRAK_RL_SPREAD`), passed to the collector by the trainer; the collector's
  constructor takes both (`shards`, `spread`, optional).
- **Contract added (R-C3's neighbour, nothing changed in R1's):** `IRolloutRanks<TObservation, TAction>` (`Rank`, `WorldSize`,
  `Gather(mine, ct)`: every rank's episodes in rank order, a collective). The collector looks for it on its policy; a policy
  that is not one plays every episode in its process. In the inventory it is used by `Idrak.Reinforcement` and `Idrak.Text`,
  so it stays in `Idrak.Abstraction` with no shared-contracts line.
- **`RolloutCollector` (minimal hook):** this rank's slots are the episodes of its shards; their environments are made for
  those slots alone, lazily, each with its slot's creation seed (the same on whichever rank plays it; `Created` counts what
  was made, `Played` the last rollout's share); each round one `ActAsync` per shard of this rank, in shard order, then this
  rank's environments stepped together (sequentially or in parallel as measured, unchanged); the final values of cut
  episodes per shard too. After the rollout the policy gathers, and the batch is sorted by episode number, so every rank
  holds the same batch. **The environment side:** each rank resets and steps only its own episodes' environments (the
  `prompts` environment; the `coding` environment's workspaces, made per episode, are per rank too).
- **`ReinforcementTrainer` (minimal hook):** on a cluster every rank runs a trainer with the same settings; rank 0 alone
  writes a checkpoint's folder, its `state.json` and the move into place, and the JSON log; every rank calls the policy's
  `Save` (a collective: the session's rank 0 writes the adapters and the optimizer state). Nothing else changed.
- **`LanguageModelPolicy`:** the refusal is gone. It implements `IRolloutRanks` over its session's cluster (`Rank`,
  `WorldSize` from the session's synchronisation; 0 and 1 in one process). `Gather` sends this rank's episodes as bytes
  (`RolloutExchange`, internal: text messages and tool lists written once each by reference and shared again as they were,
  tool calls, reasoning, the ids, log-probabilities, values and contexts as sampled, rewards, ends, info; a part that is not
  text is refused) and keeps its own episodes as the objects it played. The exchange is an all-gather of each rank's count,
  then of the values padded to the longest rank's, straight from host memory where the collectives take host values (the
  library's and the simulator's), else through two device buffers made for the call: bytes travel three to a value as an
  integer below 2^24, so no copy or wire changes them. **The update stays the session's data-parallel step** (every rank
  lays out the same batch and calls `Step` with the same sequences; each rank trains its share of the step's batches and the
  gradients are all-reduced, as `FineTuner` does). **The behaviour and reference passes are spread too** (rule 75: every
  rank computing all of them is W times the work): each rank computes a contiguous run of the batch's sequences, cut where
  the positions before it reach r/W of all positions, and the runs are gathered (`LanguageModelPolicyOptions.SpreadPasses`,
  `IDRAK_RL_SPREAD_PASSES`, on by default).
- **What is bit-identical across 1, 2 and 4 ranks (CPU, tested, 4 shards):** the first rollout's episodes whole (numbers,
  seeds, groups, prompt and reply ids, replies, the sampler's log-probabilities), every iteration's rewards and advantages,
  and in the test's four iterations every later rollout's ids and replies; on every rank of one run the adapters are the same
  bits; spreading the passes or running them whole gives the same adapters bit for bit, and every update's ratios are exactly
  1 on every rank (a sequence's forward on the CPU does not depend on what it is packed with). **What is not, and why:** the
  adapters after an update differ from one process's by float rounding (largest gap 6.0e-8 after four updates), because a
  step's gradient is summed per rank over its batches and then across the ranks in the collective's order, where one process
  accumulates all batches in one order; so later rollouts sample from weights a rounding apart and their sampler
  log-probabilities differ by up to 2.9e-6. Their ids stayed the same here; a later token can differ where the rounding meets
  a near-tie of the sampler (then the runs part, each valid). Without a fixed shard count (the default, one per rank), runs on
  different rank counts draw differently by design.
- **Measured (rule 79), in the test on the simulated cluster (4 cores, the tiny decoder, 16 episodes of 6 tokens, 4 shards,
  iterations 2 to 4, one run):** rollout / update seconds: one process 0.09 / 0.25; 2 ranks 0.04 / 0.13; 4 ranks 0.03 / 0.11;
  4 ranks with the passes run whole on every rank 0.04 / 0.19 (so spreading them paid); interleaved, iteration 2: one process
  0.02 / 0.08, 4 ranks 0.01 / 0.04. The baseline is one process with the same 4 shards (four calls a round, needed for equal
  episodes); one process with its default single shard was not timed here. `blocks` stays the default for the reason above
  (a group's rows in one call; on a model without per-row starts rows of one length decode together), which is reasoning,
  not a measurement: on this model the two splits cost about the same.
- **Tests** (`tests/Idrak.Tests/ReinforcementClusterTests.cs`, group `reinforcement cluster:`, 2, in-process clusters only, no
  child process): the split (blocks, interleaved, sizes, duplicates, shards to ranks), the settings and their variables (a bad
  split refused naming `IDRAK_RL_SPREAD`), the exchange on 3 ranks (uneven and empty byte lists, floats as bits: -0, the
  smallest subnormal, the largest), a two-step text episode's round trip (tools, a tool call, reasoning, info, a context and
  none; a tool list and messages shared as they were), an image refused; then R3's tiny decoder warmed up once and
  checkpointed, the `prompts` environment (two prompts, reward `metric` exact), grpo (group advantages, a k3 KL of 0.01, one
  epoch) under R2's trainer for 4 iterations of 16 episodes in groups of 4, on one process and on 2 and 4 ranks (blocks),
  4 ranks with the passes whole, and interleaved on one process and 4 ranks (2 iterations): the checks above, every rank
  playing its share each iteration (16, 8 or 4 episodes), the 2-rank run's checkpoint written once by rank 0 with no
  `.writing` left and its log one line an iteration, and each rank's device memory held after iteration 2 not growing by the
  last. Variables in `EnvironmentVariables.cs` (Reinforcement): `IDRAK_RL_SHARDS`, `IDRAK_RL_SPREAD`, `IDRAK_RL_SPREAD_PASSES`.
  `api/Idrak.Abstraction.txt`, `Idrak.Reinforcement.txt`, `Idrak.Text.txt` and `plans/10-abstraction-inventory.md`
  regenerated.
- **The run (CPU, this agent):** one build of `tests/Idrak.Tests` (no warnings); `IDRAK_UPDATE_API=1` (`public API`) and
  `IDRAK_UPDATE_INVENTORY=1` (`abstraction inventory: plans`); then `reinforcement cluster:|reinforcement:|reinforcement
  contracts|public API|abstraction inventory|cli health: every environment variable|cli polish: the environment table|naming:`:
  31 passed, 0 failed; `reinforcement text:|fine-tuning:|fine-tuning options:|cluster: 2 ranks equal one process|cli
  reinforce|plugins: coding`: 43 passed, 0 failed. Total 74 of 74, no rebuild.
- **Left:** `idrak reinforce` on a cluster: `idrak cluster sim` runs pretrain, tune, train and resume only, and the command
  saves the adapter on every rank; the library side is ready (the command's session already takes the rank's collectives as
  tune does). A real multi-process cluster (`idrak cluster` over TCP) was not run here; the exchange takes the collectives'
  device path there when they do not take host values.
- **The owner's commands** (repo at `D:\Projects\Idrak`; CUDA runs the same in-process cluster carved from one card):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "reinforcement cluster:|reinforcement:|reinforcement contracts|reinforcement text:|fine-tuning:|fine-tuning options:|cluster: 2 ranks equal one process|cli reinforce|plugins: coding|public API|abstraction inventory|cli health: every environment variable|cli polish: the environment table|naming:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "reinforcement cluster:|reinforcement text:|cluster: 2 ranks equal one process"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  ```
  On CUDA the ratio and spread-passes bit checks are skipped (CPU only); the first rollout's bit-identity across rank counts
  and the adapters within 1e-4 are checked there, and a miss is a finding (a kernel whose result depends on the batch it runs
  in), not a setting.

**R3c as built (2026-10-22).** R3b's leftover: `idrak reinforce` under `idrak cluster sim`. Almost all of it is in the CLI;
the library gained one method.
- **`idrak cluster sim --topology FILE -- reinforce MODEL ... -o DIR`** (`ClusterSimCommand`): `reinforce` joins the commands
  a simulated cluster runs (pretrain, tune, train, resume, reinforce; the usage error lists them), with `-o`/`--out` as the
  folder rank 0 writes (a run without it is refused naming `-o DIR`). Every rank runs the command on a thread, exactly as
  `tune` runs there: `TuneTool.Session` (R5's hook) already trained on the rank's device with its collectives
  (`FineTuningOptions.Cluster`, `ClusterOptions` from the rank) and wrote the manifest on rank 0 only; the policy then
  implements R3b's `IRolloutRanks` over the session, so each rank plays its shards' episodes and R3b's trainer writes the
  checkpoints and the JSON log on rank 0 alone. Like `tune`'s, reinforce's `-o` is an adapter folder, not a run folder: no
  event log is put in it (`RunFolder` is null for both; `--event-log FILE` still writes one). `--placement pipeline` is
  refused as for tune; the dashboard does not steer it (the RL trainer has no stop-with-checkpoint hook: the existing
  message names it); `--profile` runs it as any command (not tested here).
- **`idrak reinforce`** (`ReinforceCommand`): `--shards N` (`ReinforcementOptions.Shards`, over `IDRAK_RL_SHARDS`) and
  `--spread blocks|interleaved` (over `IDRAK_RL_SPREAD`; a bad name is a usage error naming it); `ClusterJoin.Join` first,
  as tune and distill do, so `--mode localhost` (each rank a process of the tool, joined from `IDRAK_RANK` and
  `IDRAK_CLUSTER_TOPOLOGY`) and `--cluster FILE`, `--node`, `--coordinator`, `--world-size` (a real cluster; `--rank` stays the
  adapter's, `IDRAK_RANK` names the process's) reach it too. The final adapter is written by the new collective save, so
  rank 0 alone creates `--out` and writes it; the other ranks take part and write nothing. Progress: rank 0's progress line
  and text lines; on another rank the progress reporter is the quiet one (its output is the null writer under cluster sim,
  and its error text, kept to name a failing rank, holds no progress). `--resume` works on every rank from the checkpoint
  rank 0 wrote (each rank reads it; the check that one exists runs on each).
- **Library (one method, `src/Idrak.Text/FineTuningSession.cs`, public, `api/Idrak.Text.txt` regenerated):**
  `FineTuningSession.SaveOnce(folder)`: every rank calls it at the same point (a collective: `Gather` makes a sharded run's
  weights whole while they are read, as `Checkpoint` does), rank 0 alone creates the folder and calls `Save`, then a
  barrier; in one process it creates the folder and saves; returns whether this rank wrote. `Save` itself is unchanged
  (`Checkpoint` calls it inside its own gather). Nothing in `Idrak.Reinforcement` changed; the inventory is unchanged.
- **`idrak distill --on-policy` on a cluster: not taken, and now refused (why).** The CLI path would be cheap (the same
  session hook and `SaveOnce`), but the library's `OnPolicyDistiller` is not cluster-aware: every rank would sample every
  step's prompts (W times the generation, rule 75) and step on them without an exchange, so the ranks' data-parallel steps
  agree only where generation is bit-identical across ranks, and where it is not, ranks that lay out different batches can
  call the collectives a different number of times (a hang, not an error). Doing it right is R3b's design applied to the
  distiller (shards with seeds of their own, each rank sampling its shards, the sampled ids exchanged), a library change not
  taken here. So: `cluster sim -- distill` is a usage error naming why (its other ways are teacher passes of their own, not
  asked here); `distill --on-policy` on any cluster rank (`--cluster FILE` too: before, it ran with every rank sampling
  everything and every rank writing the adapter, untested) and `reinforce --algorithm distill` on a cluster are usage errors
  pointing to `reinforce` with a policy loss and `--teacher` (the teacher's term beside the reward, whose episodes are
  spread). `distill --on-policy` in one process now saves through `SaveOnce` (the same files).
- **Variables:** none new (`--shards` and `--spread` are R3b's `IDRAK_RL_SHARDS`, `IDRAK_RL_SPREAD`); `reinforce` joins the
  commands of the Cluster rows (`ClusterRun` in `EnvironmentVariables.cs`), since it now reads `IDRAK_CLUSTER_*` and
  `IDRAK_RANK` through `ClusterJoin`. Help: `cluster sim`'s usage (the command list, reinforce's folder, shards and spread, an
  example), `reinforce`'s (the two options, the cluster options, a paragraph on clusters, an example), `distill`'s
  `--on-policy` line, `idrak help reinforcement`, the CLI README's Train examples.
- **Test** (`tests/Idrak.Tests/CliReinforceTests.cs`, `cli reinforce: cluster sim runs a seeded grpo on 2 ranks ...`, in-process
  ranks only, no child process): R5's tiny decoder and prompts file, grpo with groups of 4, 8 episodes, reward `cer` (a
  spread, so the updates move the adapters; checked), seed 3. One process with `IDRAK_RL_SHARDS=4` against `cluster sim` on
  2 CPU ranks with `--shards 4` (so the option is proven: without it the cluster would cut 2 shards and draw other
  episodes): both exit 0, every rank ends (the virtual-clock line names both), rank 0's iteration lines appear once each;
  the cluster's `--out` holds exactly the entries one process's holds (the adapter's files with `adapter_model.safetensors`,
  `idrak-tuning.json`, `checkpoints/` with `iteration-000002` alone; no event log, nothing `.writing`); the log one line an
  iteration; the first iteration's episode count and mean, spread, min and max reward equal one process's; the adapters' tensors the
  same names and within R3b's 1e-4 (largest gap measured 4.4e-6 after 2 iterations and after the resume; R3b's 6.0e-8 was
  from warmed adapters at a smaller step). `--resume` to 3 iterations on the cluster and in one process: iteration 3 alone,
  the log's third line, `iteration-000003`, the adapters again within 1e-4. `SaveOnce` on an in-process 2-rank cluster with
  a folder per rank: rank 0 wrote, rank 1's folder does not exist. Usage errors: no `-o`, `cluster sim -- distill`,
  `reinforce --algorithm distill` on the cluster, `--spread diagonal`.
- **The run (CPU, this agent):** a build of `tests/Idrak.Tests` with one compile error (a nullable tuple's default in the
  test), fixed, one rebuild with no warning; `IDRAK_UPDATE_API=1` (`public API`: `Idrak.Text.txt` gained `SaveOnce`) and
  `IDRAK_UPDATE_INVENTORY=1` (`abstraction inventory: plans`: unchanged); then once `cli reinforce|cli cluster|cli tune|cli
  distill|cli train data|reinforcement cluster:|cluster: 2 ranks equal one process|cli health|cli polish|public API|
  abstraction inventory|naming:`: **54 passed, 0 failed** (no rerun).
- **Left:** (1) `cluster sim --mode localhost -- reinforce` and `reinforce --cluster FILE` are wired through `ClusterJoin` as
  tune's are, not run here (child processes loading a model; the owner's run below). (2) A stop (Ctrl+C, `--timeout`) is
  each rank's own token, checked between iterations; ranks are not made to agree on it (tune's runs share the same gap), so a
  stop landing between two ranks' checks can leave one rank waiting in the next rollout's exchange until the collectives
  time out. Agreeing would be a stop flag in the rollout exchange or the trainer (library, not built). (3) The dashboard does
  not steer reinforce (no stop-with-checkpoint in the RL trainer). (4) On-policy distillation on a cluster (above). (5) A
  real multi-machine run, as R3b left it.
- **The owner's commands** (repo at `D:\Projects\Idrak`; the CUDA cluster is carved from one card, two ranks on it):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "cli reinforce|cli distill|cli cluster|cli train data|reinforcement cluster:|cluster: 2 ranks equal one process|cli health|cli polish|public API|abstraction inventory|naming:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "cli reinforce: cluster sim|cli reinforce: a seeded|reinforcement cluster:|cluster: 2 ranks equal one process"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  # a real small model on a simulated cluster of two ranks carved from the card, against one process with the same 4 shards
  New-Item -ItemType Directory -Force runs | Out-Null
  1..32 | ForEach-Object { '{"question": "What is ' + $_ + ' plus ' + $_ + '? Answer with the number.", "answer": "' + (2 * $_) + '"}' } | Set-Content runs\sums.jsonl
  dotnet run -c Release --no-build --project src\Idrak.Cli -- cluster init -y --nodes 1 --devices cuda:0,cuda:0 -o runs\two-cuda.json
  dotnet run -c Release --no-build --project src\Idrak.Cli -- cluster sim --topology runs\two-cuda.json -- reinforce Qwen/Qwen3-0.6B --env prompts --data runs\sums.jsonl --reward number --algorithm grpo --group 8 --episodes 16 --shards 4 --iterations 5 --max-new 64 --no-think --seed 1 -o runs\rl-cluster
  dotnet run -c Release --no-build --project src\Idrak.Cli -- reinforce Qwen/Qwen3-0.6B --env prompts --data runs\sums.jsonl --reward number --algorithm grpo --group 8 --episodes 16 --shards 4 --iterations 5 --max-new 64 --no-think --seed 1 -d cuda:0 -o runs\rl-one
  dotnet run -c Release --no-build --project src\Idrak.Cli -- cluster sim --topology runs\two-cuda.json -- reinforce Qwen/Qwen3-0.6B --env prompts --data runs\sums.jsonl --reward number --algorithm grpo --group 8 --episodes 16 --shards 4 --iterations 6 --max-new 64 --no-think --seed 1 -o runs\rl-cluster --resume
  # the same on localhost processes (each rank a process of the tool)
  dotnet run -c Release --no-build --project src\Idrak.Cli -- cluster sim --topology runs\two-cuda.json --mode localhost -- reinforce Qwen/Qwen3-0.6B --env prompts --data runs\sums.jsonl --reward number --algorithm grpo --group 8 --episodes 16 --shards 4 --iterations 2 --max-new 64 --no-think --seed 1 -o runs\rl-localhost
  ```
  What to look at: the first iteration's reward line equal in `rl-cluster` and `rl-one` (the first rollout is the same on
  any rank count with the same shards; a miss on the card is a finding: a kernel whose result depends on its batch), later
  lines close; `runs\rl-cluster` holding one adapter, `idrak-tuning.json` and `checkpoints\`; the resumed run starting at
  iteration 6; the localhost run ending on both processes.

## Phase R4: `Idrak.Reinforcement.Plugins.Coding`

- **First, R2 and R3 together** (they were built side by side, so neither could run the other): GRPO and PPO from R2's
  `ReinforcementTrainer` driving R3's `LanguageModelPolicy` on the `prompts` environment with a `metric:exact` reward, on
  the tiny decoder of R3's tests, raise the pass rate within a seeded number of iterations; on-policy distillation as a
  teacher term beside `grpo`; R3's test-local fakes stay only where they test something R2's code does not.
- **M5:** the task layer moves out of `CodingAgent` into the plug-in as `CodingTask` (the `task.json` layout unchanged),
  `CodingSuite` (`LoadSuite`), `CodingWorkspace` (copy of base, workspace and hidden files; the starting commit;
  setup), `CodingVerifier` (`VerifyAsync`, `CheckTaskAsync`) and `CodingOutcome`; `CodingAgent` keeps the loop on a
  prompt and returns `CodingAgentRun`. `idrak agent` and the sample use the loop; the sample's `agent-run` and
  `agent-check` use the plug-in (R5). The `coding agent:` tests keep their names and counts.
- **R-C8, the sandbox:** `ICommandSandbox` and `CommandSandboxes` in `Idrak.Abstraction`; `CodingTools.ExecuteAsync`'s
  process code becomes the `process` sandbox (registered by `Idrak.Text`): the allow-list, the time limit, the output
  cap and the process-tree kill as today, plus a memory limit on the process tree where .NET reports its working set (on
  every platform, rule 70); it refuses `network: off`, naming `container`. `CodingTools` runs every command through the
  sandbox its options name (`CodingToolOptions.Sandbox`, `IDRAK_SANDBOX`).
- **`container`** (the plug-in): runs each command in a Docker or Podman container (the program found on the PATH, or
  `IDRAK_SANDBOX_PROGRAM`), the workspace mounted, the image a setting (`IDRAK_SANDBOX_IMAGE`), network off unless
  allowed (`IDRAK_SANDBOX_NETWORK`), memory and CPU limits from `CommandLimits`; refuses loudly when no program is found.
- **`CodingEnvironment : IEnvironment<Conversation, ChatMessage>`** (`EnvironmentKinds`: `coding`; options: suite, filter,
  rounds, time limit, sandbox, limits): `Reset` picks a task by the episode's seed and lays out a fresh workspace from a
  prepared copy (setup such as `dotnet restore` runs once per task and is reused, rule 75; a setup that fails marks the
  task broken and leaves it out, reported); `StepAsync` runs the assistant turn's tool calls through `CodingTools` and
  returns their results as the next observation; the episode ends when the assistant answers without a tool call or the
  budget runs out; the hidden tests are copied in (overwriting any agent copy, as today) and verification gives the reward.
- **Coding rewards:** `tests-passed` (1 when every verify command passes), `verify-fraction` (the share that pass),
  `rounds` and `tool-errors` (penalties per round or failed call); weights from the reward expression. Files the agent
  changed under the hidden tests' paths are reported in the episode's info and can be refused (`refuse-test-edits`).
- **Episodes in parallel:** many workspaces at once; while the device generates some episodes' turns, the others' tools,
  builds and tests run on the cores (rule 79; the number of workspaces is measured from the cores and the free disk, and
  a setting, `IDRAK_RL_WORKSPACES`). Workspaces are deleted after their episode unless kept for inspection
  (`IDRAK_RL_KEEP_WORKSPACES`).
- **Agentic coding with a teacher:** the coding environment runs under R3's on-policy distillation as under any
  algorithm: the student works the tasks with the tools, the teacher grades every token of the student's own turns, and
  the tests' reward is added when a weight is given. No coding-specific distillation code: it is R3's, on R4's episodes.

**Done when:** the moved `coding agent:` tests pass unchanged; `plugins: coding` passes on a task suite written in the test
(a C# project and a hidden test; a scripted fake chat model that solves it on some seeds): a seeded episode replays the same,
the reward follows verification, a broken setup is left out and reported, a test edit is refused when asked; on-policy distillation runs on the suite with a fake
teacher and trains the assistant's tokens only; `process`
enforces its time, output and memory limits and refuses `network: off`; `container` refuses when no program is found
(running it is the owner's).

**Tests (CPU here):** one build; `plugins: coding|coding agent:|coding tools|command sandboxes|reinforcement contracts|
plugins:|public API|abstraction inventory|cli health: every environment variable`. **Owner:** `container` on a machine with
Docker or Podman, and the coding run of R5.

**R4 as built (2026-10-22).** Items 0-7 built; `Idrak.Text` still references no `Idrak.Reinforcement`, and the plug-in uses the
public API only.
- **R2 and R3 together (item 0).** R2's `ReinforcementTrainer` drives R3's `LanguageModelPolicy` with no change to the trainer:
  the policy is not an `IScheduledPolicy`, so the trainer leaves its rate and clip to the fine-tuning session, as R2 meant. On
  R3's tiny decoder and the `prompts` environment (reward `metric`, `metric.metric=exact`), `grpo` (group advantages, a k3 KL
  of 0.01, two epochs a rollout, 40 iterations) and `ppo` (returns less their mean, two epochs of two mini-batches, no KL, 24
  iterations) raise the pass rate. The first update's ratios are exactly 1, the trained ids are the sampled ids, and the
  device memory held does not grow. On-policy distillation runs as a teacher term beside `grpo` under the trainer (the
  student's reverse KL on its own samples falls below 0.7 of its start). **Contract change (R-C6):** `IPolicyLoss.NeedsReference(options)`,
  a default interface member that answers true, so R1's fakes and any app loss still compile and are given the reference. R2's
  `reinforce` answers false; `ppo` and `grpo` answer true when their option `kl` is weighted. The guarded slot answers for
  the app's loss or the library's. This removes the flag cleanly: `FineTuningLosses.Policy(loss, options)` lost its
  `reference` parameter and `LanguageModelPolicyOptions.Reference` is gone. The policy computes the reference pass only for
  a loss that reads it, and adds it later on the same batch if a later loss does (the adapters off do not move). R3's
  test-local `TestClippedLoss`, `GroupAdvantages` and its two update loops are removed: the adapter and multi-turn tests use
  R2's `grpo`, and the pass-rate and teacher tests use R2's trainer. Kept: `RecordingLoss`, a library loss under a test name
  that records the first gap and whether a reference came (R2 records neither), and the two-turn rollout of the multi-turn
  test (its subject is the layout). **The bars were measured:** a scratch program (not checked in) ran both algorithms on
  seeds 1-8. The five-iteration mean of the sampled pass rate went from 0.35-0.45 to 0.57-0.69 for grpo and from 0.31-0.42
  to 0.54-0.69 for ppo. The test's bar, a rise of more than 0.1 to at least 0.5, held on all 16 runs. R3's single-run bar
  (0.3 to 0.6 over 3 iterations) failed under the trainer's draws, at 0.36 to 0.50. The exact probability of "yes" is not
  a usable measure: one ppo run that passed half its episodes had moved to another spelling that the metric's
  normalization accepts.
- **M5, the task layer in the plug-in** (`src/Idrak.Reinforcement.Plugins/Coding/`): `CodingTask` (the `task.json` layout and
  `Load` unchanged), `CodingSuite.Load(folder, filter)`, `CodingWorkspace` (`Prepare`, `Overlay`, `CommitStart`, `Clone`,
  `Delete`, `HiddenFiles`, `TestEdits`), `CodingVerifier` (`SetupAsync`, `VerifyAsync`, `CheckTaskAsync` → `CodingTaskCheck`,
  and `RunAsync(agent, task, folder)`, the old `CodingAgent.RunAsync(task, folder)`; options `CodingVerifierOptions`: `Tools`,
  `Timeout`, the old `AgentOptions.VerifyTimeout`; `OnStatus` moved from the agent), `CodingVerification` (passed, commands,
  passed commands, output, `Fraction`), `CodingOutcome` (the old `AgentOutcome`'s names), and `CodingRun` (task, agent run,
  outcome, verification output; `ToJson` writes the old run's JSON line, key for key). `CodingAgent` keeps the loop:
  `RunAsync(prompt, tools)` → `CodingAgentRun` (messages, tools, think, `End`: `CodingAgentEnd` Answered, OutOfBudget, Error;
  `Note`, rounds, calls, errors, tokens, times, workspace; `ToJson`, `AddFigures`), `Options`, and
  `RunToolCallsAsync(registry, reply, previousCalls)` → `CodingToolTurn`: the loop's way of running one reply's calls (the
  repeated-call note, "Error: …" messages, the error count). It is public, so the environment steps exactly as the loop
  does with no copy. `AgentTask`, `AgentOutcome` and `AgentRun` are gone. `idrak agent` needed no change. The sample references
  `Idrak.Reinforcement.Plugins` and calls `ReinforcementPlugins.RegisterAll()` (so `IDRAK_SANDBOX=container` works there).
  `agent-run` uses `CodingVerifier.RunAsync`, `agent-check` uses `CheckTaskAsync`, and the live `agent --out` line goes
  through `CodingRun`, so it reads as before. The `coding agent:` tests keep their names and count (3), with the calls
  changed. The language defaults of the task layer (node_modules, the build-output folders, the git excludes) moved with it
  into the plug-in, which is not the library; R4b takes the rest.
- **R-C8, the sandbox** (`src/Idrak.Abstraction/CommandSandboxes.cs`, namespace `Idrak.Abstraction`, so `CommandResult`
  moved out of `Idrak.Text.Generation` with no new using for its callers): `ICommandSandbox` (`Name`, `Summary`,
  `Check(limits)` → null or the reason naming a sandbox that can, `RunAsync(words, directory, limits, ct)` → `CommandResult`,
  throwing `NotSupportedException` for limits it cannot keep). `CommandLimits`: time, output cap, memory, processes, CPUs,
  network, the programs allowed, variables, and the root the command may see. `CommandResult` gained `OutOfMemory`. The
  registry `CommandSandboxes` is **unguarded**, with this reason: a sandbox is a safety boundary, and falling back would run
  a model's command under other limits than the ones asked for. Constants `Process`, `Container` and `EnvironmentVariable`;
  `DefaultName` reads `IDRAK_SANDBOX`. One line in `shared-contracts.txt` (`CommandSandboxes`). The inventory asked for an
  `ICommandSandbox` line first, then refused it as stale, because Abstraction's own registry uses the interface. It also
  asked to remove R3's `IRewardFunction` line, now stale. **`process`** (`ProcessCommandSandbox`, Idrak.Text, registered by
  its `LibraryRegistrations`) is `ExecuteAsync`'s process code moved: the programs allowed, found on the PATH (the public
  `FindProgram`, which `container` reuses), standard input closed, the time limit with a process-tree kill, the output cap.
  It adds a memory limit on the started process's working set (`Process.WorkingSet64` every `PollInterval`, 50 ms; only
  when a limit is set). **Deviation:** the processes the command starts in turn are not counted, because .NET gives no
  portable way to list a process's children; `container` limits the whole tree. `process` refuses `network: off`, a process
  limit and a CPU limit, naming `container`. `CodingTools` keeps its own checks in their old order and messages (folder,
  shell operators, the allow-list, git's subcommands), then hands the words to the sandbox its options name. The options are
  `CodingToolOptions.Sandbox` (`IDRAK_SANDBOX`, default process), `MaxMemoryBytes`, `MaxProcesses`, `MaxCpus`, `Network` and
  `Environment`, or a sandbox given to its constructor. The tools check the sandbox when they are made, and it gets the
  allow-list as `CommandLimits.Programs`. The non-interactive variables stay in `CodingTools`, passed as the limits'
  variables, so no sandbox names a language. Results unchanged: `coding tools: run_command` passes as before.
- **`container`** (`ContainerSandbox`, the plug-in): `run --rm --name idrak-sandbox-<guid>`, `--network none` unless both the
  command and `IDRAK_SANDBOX_NETWORK` (on, or a network's name) allow it, the root bind-mounted at `/workspace` (`--mount`
  type=bind, so Windows drive letters are safe) with `--workdir` its sub-folder, `--memory`/`--memory-swap`, `--cpus`,
  `--pids-limit`, `--user`, `--env` (sorted) and `ExtraArguments`, then the image and the command. The program is Docker,
  then Podman, found on the PATH, or `IDRAK_SANDBOX_PROGRAM`. The image comes from `IDRAK_SANDBOX_IMAGE` and none is
  assumed. The program runs through the library's `process` sandbox (its time limit and output cap); a container still
  running after a timeout or a cancellation is removed (`rm --force`). It refuses with the variable's name when no program
  is found or no image is set. Tested: the refusals and the exact command lines. Running it is the owner's.
- **`coding` environment** (`CodingEnvironmentKind`, `CodingEnvironment : IEnvironment<ChatObservation, ChatMessage>`; R3's
  observation type, as the plan's `Conversation`). Options: `suite`, `filter`, `reward` (default `tests-passed`; term
  options as term.key), `rounds` (40), `time` (1200 s), `system`, `think`, `sandbox`, `memory`, `processes`, `cpus`,
  `network`, `command-timeout`, `verify-timeout`, `workspaces`, `keep`, `work`, `refuse-test-edits`. Environments of a kind
  with equal options share one run: the tasks, the tool definitions (one set, so replays compare equal), the prepared copies,
  the broken tasks and the workspace gate. The last one disposed deletes the copies (unless kept). `Reset(seed)` orders
  the tasks by the seed (`Random(seed).Shuffle`) and plays the first whose setup passed. The first draw of a task lays it
  out and runs its setup once, then commits its start for `git diff`; every episode clones that copy, `obj` included, so a
  restore is reused. A failed setup leaves the task out, reported in `BrokenTasks` and the info's `broken_tasks`. Each
  step runs the reply's calls through `CodingTools` with `CodingAgent.RunToolCallsAsync`. The episode ends on a reply
  without calls (terminated), or on the rounds or the time (truncated, after the last reply's calls ran, as the loop does).
  At the end: the test edits are found (files at the hidden paths, or in the hidden files' folders, that differ from the
  prepared copy), the hidden tests are copied in and verified, and the reward expression is scored over the info (`task`,
  `rounds`, `tool_calls`, `tool_errors`, `verify_passed`, `verify_commands`, `verify_passed_commands`, `test_edits`,
  `outcome`, `refused`, `broken_tasks`; `CodingInfo`). The episode's workspace is then deleted unless kept. **Deviation:**
  `refuse-test-edits` is an environment option, not a reward: a refused episode skips verification and counts as failed
  (`refused=test-edits`), so every reward stays a function of the info. A seeded episode replays the same steps, rewards
  and info; the observations replay exactly where the tools' output is deterministic (`run_command` reports seconds), and
  only while no episode reaches the wall-clock limit.
- **Coding rewards** (`CodingRewards`, in `RewardFunctions` by the plug-in): `tests-passed`; `verify-fraction` (commands run
  in order and stop at the first failure, as verification always did, so it is the share passed before it); `rounds`
  (−`penalty` 0.01 per reply past `free`); `tool-errors` (−`penalty` 0.05 per failed call). An answer without the
  environment's info is refused, naming it. All four pass the reward suite.
- **Episodes in parallel:** R2's collector steps one environment per episode and resets them one after another, so a gate at
  `Reset` would deadlock. **Deviation:** `IDRAK_RL_WORKSPACES` (`workspaces`) is the most episodes whose tools or
  verification run at once (a semaphore), not the number of workspaces that exist. Its default is the CPU threads, fewer when
  the free disk holds fewer than two copies of the largest prepared workspace per slot. A workspace exists only while its
  episode runs. `IDRAK_RL_KEEP_WORKSPACES` (`keep`) keeps them all. **Measured (rule 79), on 4 cores:** 4 episodes, each a
  `dotnet build` and run of the test's project, took 4.7 s and 5.1 s one at a time and 1.6 s and 1.3 s four at once (two
  runs), with the same rewards, info and conversations.
- **Agentic coding with a teacher:** R3's policy with `TeacherWeight` 1 (a teacher equal to the student) under R2's trainer
  with `grpo` runs on the coding environment with no coding-specific code; in every episode the trained ids are the replies'
  ids. A multi-turn batch from the environment (a tool call, its result, an answer) trains the two replies' ids alone.
- **The plug-in:** `CodingPlugin` ("coding") in `ReinforcementPlugins.All` registers the environment, the four rewards and
  `container` as `Part`s; Unregister leaves the registries as found. A README table row; the Abstraction README has a row for
  the sandboxes. The CLI's `plugins list` shows `sandboxes`. New variables in `EnvironmentVariables.cs`: `IDRAK_SANDBOX`,
  `IDRAK_SANDBOX_PROGRAM`, `IDRAK_SANDBOX_IMAGE`, `IDRAK_SANDBOX_NETWORK` (group "Command sandbox": `agent`, `test`), and
  `IDRAK_RL_WORKSPACES`, `IDRAK_RL_KEEP_WORKSPACES` (Reinforcement). `api/Idrak.Abstraction.txt`, `Idrak.Text.txt`,
  `Idrak.Reinforcement.Plugins.txt` and `plans/10-abstraction-inventory.md` were regenerated.
- **Tests:** `tests/Idrak.Tests/CodingPluginTests.cs`, 6 tests. `command sandboxes:` (2): process's limits (programs, a 1 ms
  time limit, the output cap, a 1-byte memory limit with a 1 ms poll stops `dotnet --info`, a roomy one does not), the
  refusals, cancellation, and the tools through a recording sandbox; container's refusals and command lines. `plugins:
  coding` (4): registration and the reward suites; a C# suite written by the test (hello and hi: a net10.0 project, setup
  `dotnet restore`, verification `dotnet build --no-restore` then running the hidden `tests/Check.cs`; broken: a setup that
  fails) driven by scripted replies; the environment suite on a quick suite (`git grep`) and the parallel measurement; the
  distillation runs on the tiny decoder. `reinforcement text:` now has 10 tests (ppo added). `reinforcement contracts`'
  package check now expects the family "coding".
- **The run (CPU, this agent):** one build of `tests/Idrak.Tests` (a first pass stopped on one compile error, a name clash in
  the policy, fixed before anything ran) and of `samples/Idrak.Samples.CodingAgent` (no warnings); `IDRAK_UPDATE_API=1`
  (`public API`) and `IDRAK_UPDATE_INVENTORY=1` (`abstraction inventory: plans`). Then
  `plugins: coding|command sandboxes|coding agent:|coding tools|reinforcement text:`: 21 passed, 1 failed (the grpo bar,
  above). Then `reinforcement:|reinforcement contracts|plugins:|cli run: tools|public API|abstraction inventory|cli health:
  every environment variable|cli polish: the environment table|naming:|override loop:|conformance kit:|fine-tuning:|distillation:`:
  117 passed, 2 failed (the inventory's two stale lines; R1's "no family yet" check). The fixes took three rebuilds. The
  first gave the bars five-iteration means and 40 iterations; on its rerun grpo passed and ppo failed the bar of the time.
  The seed probe above set the measured bars and 24 ppo iterations (second rebuild), and `reinforcement text: R2's trainer`
  then passed 2 of 2. The third rebuild fixed the two stale checks, and `abstraction inventory|reinforcement contracts`
  passed 11 of 11. On the final code every test of both lists passes (137 different tests; the second list repeats the 4
  `plugins: coding` ones).
- **Left:** `container` has not been run (no Docker or Podman here), and the language-model tests have not run on CUDA: both
  are the owner's. The process sandbox's memory limit counts the started process only. No language moved into the library;
  R4b removes the ones still there (the tools' commands, ignored folders and prompt, and the task layer's node_modules and
  build-output folders, now in the plug-in). `idrak reinforce --env coding` and the coding options in the CLI are R5's.
- **The owner's commands** (repo at `D:\Projects\Idrak`):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "plugins: coding|command sandboxes|coding agent:|coding tools|reinforcement text:|reinforcement:|reinforcement contracts|plugins:|cli run: tools|public API|abstraction inventory|cli health: every environment variable|cli polish: the environment table|naming:|override loop:|conformance kit:|fine-tuning:|distillation:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "reinforcement text:|plugins: coding: R3's on-policy distillation"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  dotnet build -c Release samples\Idrak.Samples.CodingAgent
  # container, on a machine with Docker or Podman, with an image of your own that holds the .NET SDK, on a task suite:
  $env:IDRAK_SANDBOX = "container"; $env:IDRAK_SANDBOX_IMAGE = "<your image with the .NET SDK>"
  dotnet run -c Release --no-build --project samples\Idrak.Samples.CodingAgent -- agent-check <suite folder>
  Remove-Item Env:IDRAK_SANDBOX, Env:IDRAK_SANDBOX_IMAGE
  ```
  On CUDA the pass-rate bars were measured on the CPU only. A miss there is a finding to report (the card's draws differ).

## Phase R4b: no programming language in the library; `Idrak.Reinforcement.Plugins.CSharp` (the owner, 2026-10-22)

The owner's rule: nothing language-specific in the library. R4b first takes every programming language out of the
library's coding agent (part 1), then adds agentic coding on C# projects as a plug-in of its own on top (part 2), so
nothing tied to one programming language sits in the library or in the general `coding` plug-in (which stays driven by
each task's own commands).

### Part 1: coding languages as a registry, out of the library

**What names a language in the library today (found in the source at 2a5df3e):**
- `src/Idrak.Text/Generation/CodingTools.cs`: `CodingToolOptions.Commands` (`:27`: `dotnet`, `npm`, `npx`, `ng`, `node`,
  `tsc`, `git`), `IgnoredDirectories` (`:48`: `bin`, `obj`, `node_modules`, `dist`, `.angular` beside `.git`, `.vs`,
  `.idea`, `coverage`, `out`), and the tools' descriptions (`:78`, `:109-110`: "dotnet build, dotnet test, npm test, npx
  tsc --noEmit, ng build").
- `src/Idrak.Text/Generation/CodingAgent.cs`: `DefaultSystemPrompt`'s build and test lines for C#, TypeScript and Angular
  (`:160`); the doc examples (`:19`, `:23`); and in the task layer that M5 moves to the coding plug-in in R4: the suite
  skipping `node_modules` (`:35`), `node_modules` linked instead of copied (`:356-375`), the build-output folders not
  copied (`:392`) and the start commit's git excludes (`:423`).
- The CLI's `idrak agent` (`src/Idrak.Cli/Commands/Run/AgentCommand.cs:101`) reads `CodingAgent.DefaultSystemPrompt`.

**The contract** (`Idrak.Text.Abstractions`, since only `Idrak.Text` uses it in the library; plug-ins count as apps):
`ICodingLanguage`: `Name` ("csharp"), `Summary`, `Detect(workspace)` (from the files at the workspace's top, e.g. a
`.csproj`, a `package.json`), `Commands` (the programs `run_command` may start for it), `PromptLines` (how to build,
check and test), `ToolExamples` (for the tools' descriptions), `IgnoredDirectories`, `LinkedDirectories` (dependency
folders linked into a workspace, not copied), `GitExcludes`, `DefaultSetup` and `DefaultVerify` (for tasks that name
none). Registry `CodingLanguages` (a `SlotTable`, unguarded: a profile is read once per run; with a reason).
- **The library registers no language.** What stays in the library is language-free: the agent loop, the tools, the
  read-only `git` subcommands, `.git` as an ignored folder, and a prompt with no build or test commands in it.
- **The agent composes its defaults from the registered languages:** the prompt's generic part plus each language's
  lines, the allowed commands as `git` plus each language's, the ignored and linked folders as the union. Which
  languages: every registered one (the default, so the tool behaves as today), the ones detected in the workspace, or a
  list (`AgentOptions.Languages`, `IDRAK_CODING_LANGUAGES`: `all`, `detected` or names).
- **A workspace whose language is detected by no registered profile** (a `.csproj` with no `csharp` registered) gets a
  clear notice naming the plug-in to register, not a silent agent that cannot build; a task's explicit commands still
  run (they are the task's, not the library's).

**The profiles move to `Idrak.Text.Plugins`** (not to the reinforcement plug-ins: `idrak agent` uses them with no
training): `Idrak.Text.Plugins.CSharp` (plug-in "csharp-language", profile `csharp`: `dotnet`; `bin`, `obj`, `.vs`;
build and test lines) and `Idrak.Text.Plugins.TypeScript` (plug-in "typescript-language", profiles `typescript`: `npm`,
`npx`, `node`, `tsc`, `node_modules` linked, `dist`; and `angular`: `ng`, `.angular`). `TextPlugins.All` lists them, so
the CLI (which registers `Idrak.Text.Plugins` at startup) and the CodingAgent sample (which calls
`TextPlugins.RegisterAll()`) keep today's behaviour.

**Results unchanged, proven (rule 71):** with `Idrak.Text.Plugins` registered and `Languages: all`, the composed system
prompt, the allowed commands and the tools' descriptions are byte for byte today's, and the ignored folders today's plus
the `python` profile's two (below) (a test compares
them with the current strings, kept in the test only); with no language registered, they name no language (a test scans
them for every profile's commands and folders).

**`Idrak.Data`'s code format (the owner, 2026-10-22: "go with recommendation"):** `src/Idrak.Data/DataFiles.cs` holds
the `code` data format's table from file extensions to language names (`:21-22`, about forty languages, as labels on
rows) and the folders its folder reader skips (`:31`: `bin`, `obj`, `node_modules`, `.git`, `.vs`, `.idea`, `dist`,
`.angular`, `__pycache__`, `.venv`).
- **The extension table stays** as data: a generic list of many languages used as row labels, like a media-type map, with
  no behaviour for any of them.
- **The skip list becomes a setting,** `ReadOptions.SkippedFolders`, whose library default names no language: `.git` and
  the editors' state folders (`.vs`, `.idea`). The language folders come from the registered coding languages' ignored
  folders, passed in by the callers that know them (the CLI's data commands and `Idrak.Text`'s readers of code data);
  `Idrak.Data` reads no registry of `Idrak.Text`, which it does not reference.
- So the CLI reads code folders as today, a `python` profile joins `Idrak.Text.Plugins.Python` ("python-language":
  ignored folders `__pycache__`, `.venv`; no commands, no prompt lines). The coding agent's ignored folders gain those two
  (the one change to the agent's defaults, reported and tested); its prompt and commands stay byte for byte today's.
- A direct user of `Idrak.Data` with no setting now reads build-output folders it used to skip; the setting and its
  default are in the API dump and `src/Idrak.Data/README.md`.

### Part 2: `Idrak.Reinforcement.Plugins.CSharp`

It builds on `coding` and on the `csharp` profile and copies neither: `CSharpCodingPlugin` requires `CodingPlugin` and
the `csharp` language plug-in (`Requires`, as Gemma 3 vision requires the Gemma text plug-in).
- **The `csharp` environment** (`EnvironmentKinds`): the `coding` environment with the `csharp` profile's defaults for
  what a task does not say (setup `dotnet restore`; verification `dotnet build --no-restore`, then `dotnet test
  --no-build` with a results file) and the profile's prompt lines and commands; `language: "csharp"`.
- **Rewards read from machine-readable output, not console text:**
  - `csharp-tests`: the share of tests that passed, from `dotnet test`'s TRX results file (XML through the base
    library's `XmlReader`), so a task with ten tests that passes seven gets 0.7, not 0; tests that did not run count as
    failed.
  - `csharp-build`: 0 when the build fails, else 1, less a weight per error (and optionally per warning), from the
    build's error and warning log (`-fileLoggerParameters` with `errorsonly` / `warningsonly`), whose
    `file(line,col): error CSnnnn: text` lines are a stable format.
  - Both combine with the general ones (`rounds`, `tool-errors`) and with R3's teacher term by weights.
- **Speed (rules 75, 79):** one shared package folder for every episode (`NUGET_PACKAGES`, filled by each task's setup
  once), so a workspace restores from disk; build nodes not left behind by parallel episodes (`MSBUILDDISABLENODEREUSE`)
  and the build server per workspace a setting, measured; the number of builds at once measured from the cores (the
  workspaces setting of R4).
- **Guards against gaming the tests:** the hidden tests are copied in after the agent (R4); edits to the test projects
  and to the build's shared files (`Directory.Build.props`, `Directory.Packages.props`, the test `.csproj`) are reported
  and can be refused (a glob list, a setting).
- **Sandbox:** `process` with `dotnet` on the allow-list (as today), or `container` with an image holding the .NET SDK,
  the image the owner's setting (`IDRAK_SANDBOX_IMAGE`), never a fixed one.
- **Settings:** the default commands, the reward weights, the protected globs, the package folder; variables
  `IDRAK_CSHARP_PACKAGES`, `IDRAK_CSHARP_PROTECT` (names proposed).

**Done when (part 1):** the coding agent's prompt, commands, folders and tool descriptions are today's byte for byte
with `Idrak.Text.Plugins` registered, and name no language with none registered; `detected` picks `csharp` for a
`.csproj` workspace and `typescript` for a `package.json` one; an unregistered detected language gives its notice;
`plugins: csharp-language`, `plugins: typescript-language` and `plugins: python-language` register and unregister as
found; `coding agent:` and `coding tools` pass with the same expectations; the CLI's code-data reading skips the same
folders as today, and `Idrak.Data` alone skips only `.git`, `.vs` and `.idea`.

**Done when (part 2):** `plugins: csharp` passes on a tiny C# solution written by the test (a library project and a test project;
it runs the real `dotnet`, as the CodingAgent tests do): a task with no commands gets the C# defaults; a fake model's
partly right answer gets the share of tests it passes; a build with errors gets `csharp-build`'s penalty per error; an
edit to the test project is refused when asked; two episodes in parallel share the package folder and leave no build
nodes running; unregistering `csharp` leaves `coding` as it found it.

**Tests (CPU here):** one build; `coding languages|plugins: csharp|plugins: typescript|plugins: python|data files|cli data|plugins: coding|coding agent:|
coding tools|cli agent|reinforcement contracts|plugins:|public API|abstraction inventory|cli health: every environment
variable`. **Owner:** `idrak agent` on a C# and on a TypeScript project (unchanged behaviour), and the C# demo suite run of
R5.

**R4b as built (2026-10-22).** Both parts built; the library registers no coding language, `Idrak.Data` still references
no `Idrak.Text`, and the plug-ins use the public API only.
- **The contract** (`src/Idrak.Text/Abstractions/CodingLanguages.cs`, `Idrak.Text.Abstractions`): `ICodingLanguage` (`Name`,
  `Summary`, `Detect(workspace, depth)`, `Commands`, `PromptLines` as "Name: commands", `ToolExamples` (`CodingToolExamples`:
  folders, file globs, search patterns and globs, commands, command lines), `IgnoredDirectories`, `LinkedDirectories`,
  `GitExcludes`, `Environment`, `DefaultSetup`, `DefaultVerify`), `CodingLanguage` (a profile as data; `Detect` looks for its
  `Markers` within the depth), and the registry `CodingLanguages` (a `SlotTable`, unguarded: a profile is read once, when the
  tools are made, so no call could fall back). Choices: `all` (the default), `detected`, `none` (added: the language-free
  defaults alone), or names in their order; `IDRAK_CODING_LANGUAGES`. `Merge` joins lists keeping every list's own order (an
  item new to the result goes right after the one before it in its list). `Notice` and `FindFile` are public helpers.
- **The library composes its defaults** (`CodingTools`, `CodingAgent`): `CodingToolOptions.Languages` (all by default) and
  `DetectionDepth` (2); `Commands` and `IgnoredDirectories` became nullable (null: composed). The programs are the languages'
  then `git`; the ignored folders `.git`, the languages' ignored and linked ones, then `.idea`, `coverage`, `out`
  (`CodingToolOptions.LanguageFreeIgnoredDirectories`); the commands' variables `CI`, `NO_COLOR`, `FORCE_COLOR` and the
  languages'; the tools' examples merged in order, a sentence without them when there are none. `CodingTools` exposes the
  resolved `Languages`, `Commands`, `IgnoredDirectories` and `LanguageNotice` (found on first read). `CodingAgent.SystemPrompt(languages)`
  composes the prompt: the language-free text, and the line on checking the work with each language's "Name: commands" in
  parentheses, cut at 110 columns (a way kept whole when it fits, else broken after its "Name:"). `DefaultSystemPrompt` is
  now a static property (every registered language), no longer a const; `RunAsync` uses the tools' languages.
- **Results unchanged, proven:** with Idrak.Text.Plugins' languages and `all`, the prompt, the programs (`dotnet, npm, npx, ng,
  node, tsc, git`, in the old order), every tool description and the variables are the old ones byte for byte (the test keeps
  the old strings); the ignored folders are the old ten plus `__pycache__` and `.venv`. With none registered, a scan for every
  profile's programs, folders, examples, prompt labels, variables and names finds none.
- **The profiles** (`Idrak.Text.Plugins`, in `TextPlugins.All` after DeepSeek): `CSharp` ("csharp-language": `csharp`, markers
  `*.sln`, `*.slnx`, `*.csproj`; `dotnet`; `bin`, `obj`, `.vs`; the DOTNET_ variables; setup `dotnet restore`, verify `dotnet
  build --no-restore`, `dotnet test --no-build`), `TypeScript` ("typescript-language": `typescript`, `package.json` or
  `tsconfig.json`, `npm, npx, node, tsc`, `node_modules` linked, `dist`; `angular`, `angular.json`, `npm, npx, ng, node, tsc`,
  `.angular`), `Python` ("python-language": `__pycache__`, `.venv`, nothing else). A README section.
- **The notice:** a workspace none of the chosen languages detects, holding source files (named by Idrak.Data's extension table,
  now readable as `DataFiles.LanguageOf`), gets "Note: … detects this workspace, whose files are csharp, xml: the agent gets
  no build or test commands for them. Register a coding-language plug-in that provides 'csharp' (Idrak.Text.Plugins: …)";
  `idrak agent` prints it on the error output. A task's own commands still run.
- **The coding plug-in's task layer** names no language: `CodingWorkspace.Prepare`, `Overlay`, `CommitStart`, `HiddenFiles`,
  `TestEdits` take the languages (optional; every registered one by default) for what is not copied, what is linked and the
  git excludes; new `ProtectedEdits(start, workspace, globs)`. `CodingSuite.Load` skips `.git` and the languages' folders
  (it skipped `node_modules` only). The `coding` environment gained the options `languages` (resolved once per run; `detected`
  looks in the tasks' folders; the tools get the names, so every episode offers the same definitions) and `protect` (globs
  whose edits count as test edits).
- **`Idrak.Data`:** `ReadOptions.SkippedFolders` (default `ReadOptions.DefaultSkippedFolders`: `.git`, `.vs`, `.idea`), read by
  folder and archive reading; carried by `DatasetSpec.Open`/`Download` (an argument), `DatasetRecipe.SkippedFolders` (JSON
  `"skip_folders"`) and `GitHub.Repository(…, options)` (the github source passes the spec's). The CLI's data commands pass
  `CodeFolders.Skipped` (Idrak.Data's three and every registered language's ignored and linked folders: exactly the old list):
  `data show`, `count`, `download`, the default build, `data recipe`, and `RowFiles`/`DataProfile` (folders and archives). The
  extension table stays as data. README section.
- **Part 2, `Idrak.Reinforcement.Plugins.CSharp`:** `CSharpCodingPlugin` ("csharp", in `ReinforcementPlugins.All`) `Requires`
  `CodingPlugin` and `CSharpLanguagePlugin` (so the package now references Idrak.Text.Plugins, as Idrak.Vision.Plugins does) and
  registers the `csharp` kind and the rewards `csharp-tests`, `csharp-build`. The kind is `CodingEnvironmentKind(new
  CSharpEnvironmentVariant())`: a new contract, `CodingEnvironmentVariant` (`Idrak.Reinforcement.Plugins.Coding.Abstractions`),
  lets a language variant set the coding options' defaults (`reward` csharp-tests, `languages` csharp, `protect`), add
  options (`packages`, `node-reuse`, `build-server`), variables, the task's defaults (the language's setup and verify for what a
  task does not name) and info before the reward, with no copy of the environment. Verification's `dotnet build` gets `-fl1
  -flp1:errorsonly -fl2 -flp2:warningsonly` (logs `msbuild1.log`, `msbuild2.log`) and `dotnet test` `--logger trx
  --results-directory .idrak-results`; stale ones are deleted before verification (an agent cannot plant a passing TRX);
  after it, `CSharpResults` reads the TRX `Counters` with `XmlReader` and counts MSBuild's `error`/`warning` lines once each
  into the info (`CSharpInfo`). `csharp-tests` = passed / total (0 when none ran); `csharp-build` = (built ? 1 : 0) − `error`
  (0.1) × errors − `warning` (0) × warnings, at least `floor` (−1). `packages` / `IDRAK_CSHARP_PACKAGES` sets `NUGET_PACKAGES`
  (default: none, the machine's NuGet folder, already shared); `node-reuse` false sets `MSBUILDDISABLENODEREUSE=1`;
  `build-server` false sets `UseSharedCompilation=false`. `protect` defaults to `IDRAK_CSHARP_PROTECT` or
  `**/Directory.Build.props`, `**/Directory.Build.targets`, `**/Directory.Packages.props`, `**/global.json`, `**/*Tests/**`,
  `**/*Test/**`, `**/*Tests.csproj`, `**/*Test.csproj`. **Measured** (a scratch solution of a library and an xunit project, 4
  cores, warm, three builds each): the shared compiler server 1.3-1.5 s a build, `UseSharedCompilation=false` 7.7-7.9 s, so it
  stays on by default; node reuse on 1.32-1.48 s, off 1.37-1.43 s (no difference), so it is off by default (no nodes left
  behind by parallel episodes). README row and section. New variables in `EnvironmentVariables.cs`: `IDRAK_CODING_LANGUAGES`,
  `IDRAK_CSHARP_PACKAGES`, `IDRAK_CSHARP_PROTECT`, and `NUGET_PACKAGES`, `MSBUILDDISABLENODEREUSE`, `UseSharedCompilation` (set
  for the csharp environment's commands). `idrak agent --languages LIST`; `plugins list` shows `languages`.
- **Deviations, with reasons.** `AgentOptions.Languages` is `CodingToolOptions.Languages`, reached through `AgentOptions.Tools`:
  one setting the tools and the prompt share, not two that could disagree. The contract has `Environment` beside the plan's
  fields: the tools' non-interactive variables named languages (DOTNET_*, npm_config_*, NG_CLI_ANALYTICS). `Detect` takes a
  depth (`DetectionDepth`, 2) because a solution's projects sit below its top. The library keeps `.git`, `.idea`, `coverage`
  and `out` (no language's), and `.vs` went to the C# profile as planned, so Idrak.Data's default (`.git`, `.vs`, `.idea`,
  the owner's) and the agent's language-free list differ in `.vs`. The Angular profile lists `npm, npx, node, tsc` too (its
  toolchain), which with the order-keeping merge gives the old program order exactly. The coding workspace's git excludes are
  the old five lines with `dist/` before `.angular/`. The composed prompt ends its lines with `\n` on every platform; the old
  const took its source file's line endings, so on a checkout with CRLF line endings the prompt changes from CRLF to LF (rule
  70). The build log uses the file loggers' default names because the coding tools refuse a `;` in a command (no shell), which
  `-flp:logfile=…;errorsonly` needs. The rewards read the environment's info (the files are parsed at the episode's end,
  before the workspace is deleted), as every R4 reward does. The leftover-node check reads `/proc` (Linux); elsewhere the test
  checks that `MSBUILDDISABLENODEREUSE=1` reaches the commands.
- **Tests** (`tests/Idrak.Tests/CodingLanguageTests.cs`, 7; `CSharpCodingPluginTests.cs`, 2): `coding languages:` (3: unchanged
  byte for byte, none names a language, detection, names, the notice, the variable, the merge), `plugins: csharp-language`,
  `plugins: typescript-language`, `plugins: python-language` (register once, unregister as found), `data files:` (Idrak.Data
  alone skips three folders in folders and archives; the CLI's lists and `idrak data count` skip the old ten), `plugins:
  csharp:` (2: registration over coding and the language, the rewards' suite, TRX and log reading, the variables; and a
  solution of a library and an xunit project built and tested by the real dotnet: the C# defaults, 3 of 4 tests (0.75), two
  build errors (−0.2), the protected edits refused, two episodes in parallel with one package folder and no node left). Changed
  expectations: `datasets:` (an archive and the GitHub snapshot now read `node_modules` and `bin` unless told), R1's
  `reinforcement contracts` package check (the plug-in package's references and families). The C# test downloads the xunit
  packages once into the kept folder `%TEMP%/idrak-csharp-test-packages` (network needed the first time).
- **The run (CPU, this agent):** one build of `tests/Idrak.Tests` (no warning), `IDRAK_UPDATE_API=1` (`public API`) and
  `IDRAK_UPDATE_INVENTORY=1` (`abstraction inventory: plans`): `api/Idrak.Data.txt`, `Idrak.Text.txt`, `Idrak.Text.Plugins.txt`,
  `Idrak.Reinforcement.Plugins.txt` and `plans/10-abstraction-inventory.md` regenerated. Then `coding languages|plugins: csharp|
  plugins: typescript|plugins: python|plugins: coding|coding agent:|coding tools|command sandboxes|data files|cli train data:
  data|datasets: JSON Lines|datasets: Hugging Face|cli run: tools|plugins:|reinforcement contracts|public API|abstraction
  inventory|cli health: every environment variable|cli health: plugins list|cli polish: the environment table|naming:|
  override loop:`: 95 passed, 1 failed (`datasets: Hugging Face`: its GitHub pattern line, which the new default reaches, a
  test expectation). One rebuild of the test project fixed it; the filter then passed 1 of 1.
- **Left:** `idrak tune` (`TuneTool`) and `idrak distill` (`DistillCommand`), R5's files, read their sources with Idrak.Data's
  default now (only `.git`, `.vs`, `.idea` skipped in a folder or an archive given as data); each needs `CodeFolders.Skipped`
  passed (`spec.Open(downloads, CodeFolders.Skipped)`, `recipe with { SkippedFolders = … }`). Idrak.Text has no reader of code
  folders to change. `CodingVerifier` (agent-run, agent-check) applies no language defaults: a task run there still names its
  commands. The owner's: `idrak agent` on a C# and a TypeScript project, and the C# demo suite of R5.
- **The owner's commands** (repo at `D:\Projects\Idrak`; no GPU code changed):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "coding languages|plugins: csharp|plugins: typescript|plugins: python|plugins: coding|coding agent:|coding tools|command sandboxes|data files|cli train data: data|datasets: JSON Lines|datasets: Hugging Face|cli run: tools|plugins:|reinforcement contracts|public API|abstraction inventory|cli health: every environment variable|cli health: plugins list|cli polish: the environment table|naming:|override loop:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  # idrak agent on a C# and on a TypeScript project (the same programs, prompt and folders as before), and detection:
  dotnet run -c Release --no-build --project src\Idrak.Cli -- agent <model> "Build the project and run its tests" --workspace <a C# project> --yes
  dotnet run -c Release --no-build --project src\Idrak.Cli -- agent <model> "Build the project and run its tests" --workspace <a TypeScript project> --yes --languages detected
  ```

## Phase R5: `idrak reinforce`, the samples, a demo suite

- **`idrak reinforce MODEL`** (Train group, beside `tune`): `--env prompts --data FILE --reward exact` or `--env coding
  --suite DIR`; `--algorithm` (a `PolicyLosses` name), `--estimator`, `--group`, `--kl`, `--clip`, `--epochs`,
  `--iterations`, `--environments`, `--sandbox`, `--teacher MODEL` with `--teacher-device`, `--teacher-weight`,
  `--divergence`, `--algorithm distill` (on-policy distillation alone), LoRA and device options as `tune` has them (shared, not copied),
  `--out DIR` (adapters and checkpoints; `--resume`), `--list` (environments, rewards, algorithms, estimators, sandboxes,
  each from its registry). The live progress line through `IProgressReporter`; a JSON line per iteration with
  `IDRAK_RL_LOG`.
- **Every new variable** in `src/Idrak.Cli/Shared/EnvironmentVariables.cs`: `IDRAK_RL_ALGORITHM`, `IDRAK_RL_ESTIMATOR`,
  `IDRAK_RL_GROUP`, `IDRAK_RL_KL`, `IDRAK_RL_CLIP`, `IDRAK_RL_ENVIRONMENTS`, `IDRAK_RL_WORKSPACES`,
  `IDRAK_RL_KEEP_WORKSPACES`, `IDRAK_RL_TURNS`, `IDRAK_RL_BEHAVIOR`, `IDRAK_RL_LOG`, `IDRAK_SANDBOX`,
  `IDRAK_SANDBOX_PROGRAM`, `IDRAK_SANDBOX_IMAGE`, `IDRAK_SANDBOX_NETWORK`, and R3's `IDRAK_RL_TEACHER_WEIGHT`,
  `IDRAK_RL_DIVERGENCE`, `IDRAK_RL_DISTILL_MIX` (names as proposed, the owner 2026-10-22).
- **`idrak distill --on-policy`:** the existing command gains on-policy distillation over its chat data (the `prompts`
  environment), with `--divergence`, `--beta` and `--mix` (λ); the training is R3's, not a copy.
- **`samples/Idrak.Samples.CodingAgent`:** `agent-run` and `agent-check` on the plug-in's suite and verifier, and a new
  `agent-train` that calls the library's trainer (no trainer of its own); its README.
- **A demo suite** under `samples/Idrak.Samples.CodingAgent/suite/`: a few small C# tasks (a workspace, hidden tests, a
  solution), each passing `agent-check`, run through the `csharp` environment of R4b, for the owner's first real run
  (this part of R5 waits for R4b's merge).

**Done when:** `cli reinforce` runs a seeded GRPO iteration on the test's tiny model (`prompts`) and on the test's coding
suite with the fake model, `--list` names every registry's entries, `--resume` continues, the variables table test
passes, and `agent-check` passes on the demo suite (CPU here).

**Tests (CPU here):** one build; `cli reinforce|cli health|cli tune|plugins: coding|reinforcement text:|public API|
abstraction inventory`; the CodingAgent sample built.

**R5 as built (2026-10-22).** Everything in the CLI and the sample; no library code changed, so no API dump or inventory
change (the `public API` and `abstraction inventory` guards pass as they were).
- **`idrak reinforce MODEL`** (`src/Idrak.Cli/Commands/Train/ReinforceCommand.cs`, Train group after `tune init`): R2's
  `ReinforcementTrainer<ChatObservation, ChatMessage>` over R3's `LanguageModelPolicy` on a `FineTuningSession` of the loaded
  model. `--env NAME` is any `EnvironmentKinds` kind that observes conversations and acts with replies (another kind is
  refused naming both types; an unregistered one with the registry's "not registered" message, exit 2). `prompts`: the
  library's kind made over `--data` read **as idrak tune reads data** (files, folders, hf: specs, column templates,
  question/answer rows normalized to conversations, `--system`, `--max-rows`), with the model's tokenizer for the length
  reward; `coding` (or a plug-in's): the registered kind with its options. Environment options: `--reward` (`IDRAK_RL_REWARD`),
  `--suite`, `--rounds`, `--sandbox` (checked in `CommandSandboxes`), `--no-think`, `--data` for a kind that reads one file,
  and `--env-option KEY=VALUE` for any key (term options with a dot); every key is checked against the kind's `Keys` before
  the model loads. Run options: `--algorithm` (a `PolicyLosses` name, or `distill`), `--estimator`, `--group`, `--episodes`,
  `--kl`, `--clip`, `--epochs`, `--minibatch`, `--iterations`, `--environments`, `--temperature`, `--save-every`,
  `--evaluate-every`, `--seed` (common); each over `ReinforcementOptions.FromEnvironment()`, so every `IDRAK_RL_*` variable
  applies and the command line wins (a bad variable is a usage error naming it). Policy: `--max-new` (`IDRAK_RL_MAX_NEW`,
  256), `--teacher MODEL` (`IDRAK_RL_TEACHER`) with `--teacher-device` (`IDRAK_RL_TEACHER_DEVICE`), `--teacher-weights`,
  `--teacher-weight` (1 by default when a teacher is given, else `IDRAK_RL_TEACHER_WEIGHT`; a weight without a teacher is
  refused), `--divergence` (checked in `TeacherDivergences`), `--beta`; `IDRAK_RL_TURNS` and `IDRAK_RL_BEHAVIOR` through
  the policy's own defaults. `--out DIR`: the adapter (PEFT) and `idrak-tuning.json` in DIR (so `idrak tune chat DIR` and
  `--adapter DIR` load it), checkpoints in `DIR/checkpoints` (or `IDRAK_RL_CHECKPOINTS`); `--resume` goes on from the
  latest there (`IDRAK_RL_RESUME` names another; nothing to resume is a usage error). Ctrl+C (or `--timeout`) stops after
  the current iteration, with a checkpoint and the adapter saved. Progress: a `ProgressLine` (the CLI's `IProgressReporter`)
  handed to the trainer as its `Progress`, a text line per iteration (reward mean ± spread, min to max, loss, updates,
  evaluation, rollout and update seconds), the trainer's JSON line per iteration with `IDRAK_RL_LOG`, and with `--json` one
  document (iterations, cancelled, checkpoint, each iteration's returns and loss, the output lines). `--list` prints every
  registry's entries with their summaries and option keys (environments, rewards, algorithms with `distill`, estimators,
  sandboxes, divergences), in text or JSON.
- **Shared with tune, not copied:** the session's options are `TuneTool.SessionValueOptions` / `SessionFlags` (LoRA rank,
  alpha, targets, type, an adapter to go on with; optimizer, schedule, warm-up, lengths, batch tokens, accumulation; context,
  matmul, GPU memory; checkpointing, recompute, bf16 activations, packing, graphs, FP8, CPU optimizer, offload) and
  `TuneArguments` (weights `-w`, device, seed) builds tune's arguments, which `TuneTool.Parse` reads as `idrak tune` does.
  `TuneTool.Session` is a new hook: `train` then loads the model with the tool's device, weights and process settings
  (put back afterwards), hands it with the parsed `FineTuningOptions` to the command's work, and writes the manifest on
  success; `TuneTool.ReadChats` reads the data after the model as conversations. `idrak distill` now takes its session
  options from the same list (it gains `--momentum`, `--decay`, `--adapter`, `--matmul`, `--gpu-memory`, `--no-graphs`, `--fp8`,
  `--cpu-optimizer`) and builds its arguments with `TuneArguments` (its messages unchanged).
- **The schedule:** without `--schedule`, the session's schedule is the run's (`IDRAK_RL_SCHEDULE`, constant; warm-up
  `IDRAK_RL_WARMUP`, 0), not tune's cosine with 3 % warm-up; the session's steps are iterations × epochs × mini-batches (the
  trainer's defaults: ppo 4 epochs, else 1). R2's `MaxGradientNorm` and schedule apply only to an `IScheduledPolicy`; the
  language-model policy keeps the session's (as R4 noted).
- **`idrak distill --on-policy`:** R3's `OnPolicyDistiller.Train` on a session of the student with the teacher
  (`DistillationTeachers.FromModel`), over the prompts of `--data` read as tune reads them; `--divergence`
  (`IDRAK_RL_DIVERGENCE`), `--beta`, `--mix` (λ, `IDRAK_RL_DISTILL_MIX`), `--steps` (default: `--epochs` passes over the
  prompts), `--batch-size` (prompts a step), `--max-new` (256), `--temperature` (the divergence's softening); the student
  samples at `IDRAK_RL_TEMPERATURE`. A stored-logits teacher reaches the library's refusal (it cannot cover new samples). The
  same helper (`DistillCommand.TrainOnPolicy`) runs `idrak reinforce --algorithm distill`, its steps the iterations and
  its prompts a step the episodes.
- **Variables** (`EnvironmentVariables.cs`): `ReinforcementRun` is now `["reinforce"]`; new `OnPolicyRun` (`reinforce`,
  `distill`) for `IDRAK_RL_TURNS`, `_BEHAVIOR`, `_TEACHER_WEIGHT`, `_DIVERGENCE`, `_DISTILL_MIX`, `_TEMPERATURE` (R3 had named
  `tune`, `distill`); `reinforce` joins the sandbox rows; new rows `IDRAK_RL_ENV`, `IDRAK_RL_REWARD`, `IDRAK_RL_MAX_NEW`,
  `IDRAK_RL_TEACHER`, `IDRAK_RL_TEACHER_DEVICE`. Help: the command's usage, a topic page `idrak help reinforcement` (`rl`)
  listing the registries (it registers the tool's plug-in packages first, as every command does, so `coding` and
  `container` show), the CLI README's Train table.
- **The sample** (`samples/Idrak.Samples.CodingAgent`): `agent-train <model> <suite> --out DIR` runs R2's trainer in the
  plug-in's `coding` environment over R3's policy (no trainer of its own): `ReinforcementOptions.FromEnvironment()` with
  `--algorithm`, `--iterations`, `--group`, `--episodes`, `--temperature`, `--resume` over it, `--lr`, `--rank`, `--max-new`
  (1024), `--rounds`, `--filter`, `--work`, `--adapter` (trained further, not merged); checkpoints in `DIR/checkpoints`, the
  adapter in DIR. `agent-run` and `agent-check` were already on the plug-in (R4). New `README.md` (none existed). The csproj
  keeps `suite/` out of its compile.
- **The demo suite** (`samples/Idrak.Samples.CodingAgent/suite/csharp/`): `sum-values` (a sum that skips the last value),
  `reverse-words` (a method to implement), `palindrome` (case and punctuation ignored). Each is a console project
  (`net10.0`, no packages); the workspace's `tests/Checks.cs` is a quick run the agent can build, the hidden
  `verify/tests/Checks.cs` replaces it with the checks (exit 0 when all pass; in a sub-folder, so only edits under `tests/`
  count as test edits); `setup` is `dotnet restore -nodeReuse:false`, `verify` is `dotnet build --no-restore
  -nodeReuse:false` then `dotnet bin/Debug/net10.0/<Name>.dll`, explicit, so it runs on `coding` today. **`agent-check`:
  3/3 tasks check out** (each 5 s; 17 s for the suite on 4 cores).
- **Deviations:** (1) the CLI's coding test runs grpo with the tests' tiny decoder on R4's quick suite (git grep), not with
  R4's scripted model: a script cannot learn (`Update` is not one), and the command trains a real model's adapters; the
  scripted episodes stay R4's. (2) `--algorithm distill` runs on `prompts` only (R3's distiller); in another environment
  the teacher goes beside a policy loss (`--teacher` with grpo), since no registered loss is zero on the reward. (3)
  `agent-train` was built but not run here (it needs a real model).
- **Tests** (`tests/Idrak.Tests/CliReinforceTests.cs`, 4): `cli reinforce:` `--list` against every registry, and 17 usage
  errors that exit 2 before a model loads (variables included); a seeded grpo iteration on `prompts` over the tiny decoder
  (the command line over `IDRAK_RL_ITERATIONS`/`_GROUP`, `IDRAK_RL_LOG` lines, the adapter, manifest and checkpoint, the same
  returns and loss from the same seed, `--resume` going on at iteration 2); grpo on a coding suite (each episode verified and
  rewarded 0 or 1, adapter and checkpoint, workspaces deleted); `cli distill: --on-policy` (two steps with a mix and jsd, the
  adapter, `reinforce --algorithm distill`, its usage errors).
- **The run (CPU, this agent):** one build of `tests/Idrak.Tests` (after one compile fix: `string.Create` over a non-literal
  concatenation) and of the sample (after one: a name clash in its top-level code). `cli reinforce|cli distill|cli train
  data|cli health|cli polish|public API|abstraction inventory|naming:`: 40 passed, 2 failed (`reinforce`'s summary line was
  wider than the help's 120 characters; the coding run hit tune's "needs data" check, made before the session hook was set).
  `plugins: coding|reinforcement:|coding agent:|reinforcement contracts|cli run: tools`: 26 passed; `reinforcement text:`: 10
  passed. Fixed, one rebuild, the failing filters rerun: 4 of 4 passed; then the topic page registered the plug-ins (one
  rebuild), and `cli health: help topics|cli health: no command|cli reinforce: --list|cli polish: help lists`: 4 of 4.
- **Left:** the demo suite through R4b's `--env csharp` after its merge (its tasks name their commands, so `csharp` takes
  them as they are); the owner's real run below; on CUDA the CLI's tiny-model runs.
- **The owner's commands** (repo at `D:\Projects\Idrak`):
  ```powershell
  dotnet build -c Release tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "cli reinforce|cli distill|cli train data|cli health|cli polish|plugins: coding|reinforcement text:|reinforcement:|reinforcement contracts|coding agent:|cli run: tools|public API|abstraction inventory|naming:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  $env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "cli reinforce: a seeded|cli reinforce: grpo|cli distill|cli train data: distill|reinforcement text:"
  dotnet run -c Release --no-build --project tests\Idrak.Tests
  Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
  dotnet build -c Release samples\Idrak.Samples.CodingAgent
  dotnet run -c Release --no-build --project samples\Idrak.Samples.CodingAgent -- agent-check samples\Idrak.Samples.CodingAgent\suite
  # the first real runs: the CLI, then the sample's agent-train and an evaluation with its adapter
  dotnet run -c Release --project src\Idrak.Cli -- reinforce Qwen/Qwen3-0.6B --env coding --suite samples\Idrak.Samples.CodingAgent\suite --algorithm grpo --group 8 --iterations 20 --out runs\rl-demo
  dotnet run -c Release --no-build --project samples\Idrak.Samples.CodingAgent -- agent-train Qwen/Qwen3-0.6B samples\Idrak.Samples.CodingAgent\suite --out runs\agent-rl --algorithm grpo --group 8 --iterations 20 --cuda
  dotnet run -c Release --no-build --project samples\Idrak.Samples.CodingAgent -- agent-run Qwen/Qwen3-0.6B samples\Idrak.Samples.CodingAgent\suite --adapter runs\agent-rl --out runs\after.jsonl --cuda
  ```

## Phase R6 (optional): classic environments and off-policy learning

For the owner to start or drop: it proves the general part on standard tasks, and the coding application does not need it.

- `Idrak.Reinforcement.Plugins.Classic`: `bandit`, `cartpole`, `pendulum`, `mountain-car`, on Gymnasium's equations, limits
  and termination rules, matched step by step against Gymnasium by a reference script in `tools/reinforcement/` (from
  PyPI, run with `-I`), as Plan 24's tokenizer fixtures are.
- Off-policy in `Idrak.Reinforcement`: a replay buffer (pooled, fixed size) and DQN (a target network, ε-greedy as a
  schedule), registered as an algorithm.

**Done when:** the environments match Gymnasium's trajectories on fixed seeds; PPO solves CartPole and DQN solves the
bandit and CartPole within seeded step counts on the CPU.

**Tests (CPU here):** `plugins: classic|reinforcement:|public API|abstraction inventory`.

## Order and parallelism

At most two agents at a time, each in its own worktree; a merge is built and its filters run before the push.

| Step | Lane A | Lane B |
|---|---|---|
| 1 | R0, the moves (they touch `FineTuner`, Audio's trainers, the CLI's tune and eval) | R1, the contracts and skeletons (new files, the CLI's plug-in registration; none of R0's files) |
| 2 | R2, the general part (after R1) | R3, the language-model policy (after R0 and R1) |
| 3 | none (R6 waits for the main plan) | R4, the coding plug-in (after R2 and R3) |
| 4 | R4b, the C# coding plug-in (after R4) | R5, the CLI and samples (after R4; its demo suite after R4b) |
| 5 | R6, classic environments and off-policy (after R5: the owner, 2026-10-22) | none |

Files both lanes touch (`api/*.txt`, the abstraction inventory, `tests/Idrak.Tests/Program.cs`'s test list,
`EnvironmentVariables.cs`) are merged one lane at a time; the generated dumps are regenerated, never merged by hand. A
step is a dependency order, not a barrier.

## Questions for the owner (answered 2026-10-22: "go with recommendation")

1. Decisions 6-8 (dependency direction, DPO stays, the agent loop stays): **confirmed**.
2. M5 moves the coding task layer into the plug-in, so evaluating a model on a coding suite (`agent-run`) needs
   `Idrak.Reinforcement.Plugins`: **moves** (no second copy of the task layer).
3. The coding environment's default sandbox: **`process`** (works everywhere with no outside program, rule 70; network
   allowed, as `CodingTools` today); `container` (isolated, network off, Docker or Podman) is the opt-in plug-in sandbox.
4. R6: **queued after the main plan** (the owner, 2026-10-22): it starts once R5 is merged.
5. The variable names in R5: **as proposed**.

## The owner's runs

**Collected for later (the owner, 2026-10-22: "I will test all later"):** phases go on as their inputs merge, without
waiting for the owner's runs; each merged phase adds its commands here, and the owner runs the list at the end.

```powershell
# once, after the last merge you want to test
dotnet build -c Release tests\Idrak.Tests

# R3c (merged): CPU, CUDA, then idrak reinforce under idrak cluster sim on one card split in two, against one process
# (the first iteration's reward line must match: a difference on the card is a finding), resume, and localhost mode
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "cli reinforce|cli distill|cli cluster|cli train data|reinforcement cluster:|cluster: 2 ranks equal one process|cli health|cli polish|public API|abstraction inventory|naming:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "cli reinforce: cluster sim|cli reinforce: a seeded|reinforcement cluster:|cluster: 2 ranks equal one process"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
New-Item -ItemType Directory -Force runs | Out-Null
1..32 | ForEach-Object { '{"question": "What is ' + $_ + ' plus ' + $_ + '? Answer with the number.", "answer": "' + (2 * $_) + '"}' } | Set-Content runs\sums.jsonl
dotnet run -c Release --no-build --project src\Idrak.Cli -- cluster init -y --nodes 1 --devices cuda:0,cuda:0 -o runs\two-cuda.json
dotnet run -c Release --no-build --project src\Idrak.Cli -- cluster sim --topology runs\two-cuda.json -- reinforce Qwen/Qwen3-0.6B --env prompts --data runs\sums.jsonl --reward number --algorithm grpo --group 8 --episodes 16 --shards 4 --iterations 5 --max-new 64 --no-think --seed 1 -o runs\rl-cluster
dotnet run -c Release --no-build --project src\Idrak.Cli -- reinforce Qwen/Qwen3-0.6B --env prompts --data runs\sums.jsonl --reward number --algorithm grpo --group 8 --episodes 16 --shards 4 --iterations 5 --max-new 64 --no-think --seed 1 -d cuda:0 -o runs\rl-one
dotnet run -c Release --no-build --project src\Idrak.Cli -- cluster sim --topology runs\two-cuda.json -- reinforce Qwen/Qwen3-0.6B --env prompts --data runs\sums.jsonl --reward number --algorithm grpo --group 8 --episodes 16 --shards 4 --iterations 6 --max-new 64 --no-think --seed 1 -o runs\rl-cluster --resume
dotnet run -c Release --no-build --project src\Idrak.Cli -- cluster sim --topology runs\two-cuda.json --mode localhost -- reinforce Qwen/Qwen3-0.6B --env prompts --data runs\sums.jsonl --reward number --algorithm grpo --group 8 --episodes 16 --shards 4 --iterations 2 --max-new 64 --no-think --seed 1 -o runs\rl-localhost

# R3b (merged): CPU, then CUDA (the simulated cluster is carved from one card; the first rollout's bit-identity and the
# 1e-4 adapter tolerance are checked there: a miss is a finding)
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "reinforcement cluster:|reinforcement:|reinforcement contracts|reinforcement text:|fine-tuning:|fine-tuning options:|cluster: 2 ranks equal one process|cli reinforce|plugins: coding|public API|abstraction inventory|cli health: every environment variable|cli polish: the environment table|naming:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "reinforcement cluster:|reinforcement text:|cluster: 2 ranks equal one process"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER

# R4b (merged 24d6e50, with f3864ba: tune, distill and reinforce skip the languages' build folders as before): no GPU
# code; CPU filters, then the agent on a C# and on a TypeScript project of your own (unchanged behaviour)
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "coding languages|plugins: csharp|plugins: typescript|plugins: python|plugins: coding|coding agent:|coding tools|command sandboxes|data files|cli train data|datasets: JSON Lines|datasets: Hugging Face|cli run: tools|plugins:|reinforcement contracts|public API|abstraction inventory|cli health|cli polish|naming:|override loop:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
dotnet run -c Release --no-build --project src\Idrak.Cli -- agent <model> "Build the project and run its tests" --workspace <a C# project> --yes
dotnet run -c Release --no-build --project src\Idrak.Cli -- agent <model> "Build the project and run its tests" --workspace <a TypeScript project> --yes --languages detected
dotnet run -c Release --project src\Idrak.Cli -- reinforce Qwen/Qwen3-0.6B --env csharp --suite samples\Idrak.Samples.CodingAgent\suite --algorithm grpo --group 8 --iterations 20 --out runs\rl-csharp

# R5 (merged): CPU, CUDA, the demo suite's check, then the first real runs (they download a small model)
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "cli reinforce|cli distill|cli train data|cli health|cli polish|plugins: coding|reinforcement text:|reinforcement:|reinforcement contracts|coding agent:|cli run: tools|public API|abstraction inventory|naming:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "cli reinforce: a seeded|cli reinforce: grpo|cli distill|cli train data: distill|reinforcement text:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
dotnet build -c Release samples\Idrak.Samples.CodingAgent
dotnet run -c Release --no-build --project samples\Idrak.Samples.CodingAgent -- agent-check samples\Idrak.Samples.CodingAgent\suite
dotnet run -c Release --project src\Idrak.Cli -- reinforce Qwen/Qwen3-0.6B --env coding --suite samples\Idrak.Samples.CodingAgent\suite --algorithm grpo --group 8 --iterations 20 --out runs\rl-demo
dotnet run -c Release --no-build --project samples\Idrak.Samples.CodingAgent -- agent-train Qwen/Qwen3-0.6B samples\Idrak.Samples.CodingAgent\suite --out runs\agent-rl --algorithm grpo --group 8 --iterations 20 --cuda
dotnet run -c Release --no-build --project samples\Idrak.Samples.CodingAgent -- agent-run Qwen/Qwen3-0.6B samples\Idrak.Samples.CodingAgent\suite --adapter runs\agent-rl --out runs\after.jsonl --cuda

# R4 (merged): CPU, CUDA for the language-model tests (the pass-rate bars were measured on the CPU only: a miss on the
# card is a finding), the sample, and `container` on a machine with Docker or Podman and an image of your own with the .NET SDK
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "plugins: coding|command sandboxes|coding agent:|coding tools|reinforcement text:|reinforcement:|reinforcement contracts|plugins:|cli run: tools|public API|abstraction inventory|cli health: every environment variable|cli polish: the environment table|naming:|override loop:|conformance kit:|fine-tuning:|distillation:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "reinforcement text:|plugins: coding: R3's on-policy distillation"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
dotnet build -c Release samples\Idrak.Samples.CodingAgent
$env:IDRAK_SANDBOX = "container"; $env:IDRAK_SANDBOX_IMAGE = "<your image with the .NET SDK>"
dotnet run -c Release --no-build --project samples\Idrak.Samples.CodingAgent -- agent-check samples\Idrak.Samples.CodingAgent\suite
Remove-Item Env:IDRAK_SANDBOX, Env:IDRAK_SANDBOX_IMAGE

# R3 (merged 9cbb9d4): CPU, then CUDA (watch "the first update's ratios are exactly 1": a failure on the card means the
# recomputed forward and the training step pack sequences differently there, a finding to report)
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "reinforcement text:|fine-tuning:|fine-tuning options:|distillation:|conversation tuning:|generation:|decoding|scoring:|pretrain:|abstraction inventory|public API|cli health: every environment variable"
dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "reinforcement text:|fine-tuning:|fine-tuning options:|distillation:|conversation tuning:|generation:|decoding|scoring:|pretrain:|vision tuning"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER

# R2 (merged ba2f720): no GPU code; its CPU filters on the owner's machine
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "reinforcement:|reinforcement contracts|public API|abstraction inventory|cli health: every environment variable|cli polish: the environment table|naming:|override loop:|plugins: packages"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER

# R1 (merged 6dd4e01): no GPU code; its CPU filters on the owner's machine
$env:IDRAK_DEVICES = "cpu"; $env:IDRAK_FILTER = "reinforcement contracts|plugins:|public API|abstraction inventory|naming:|override loop:|cli health|conformance kit:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER

# R0 (merged d1e1d6f) and row 32 (e9a09ef)
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "fine-tuning options:|optimizers|schedules:|tuning metrics:|pretrain:|distillation: the fine-tuning loss|vision tuning trainer|audio train|audio tts train|a Lion optimizer|conformance kit:|operations:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
```


Each phase's as-built note gives the exact PowerShell list (from `D:\Projects\Idrak`): one
`dotnet build -c Release tests\Idrak.Tests`, the phase's CPU filters, then its CUDA filters with `--no-build`. The pattern:

```powershell
# once, after a phase is merged
dotnet build -c Release tests\Idrak.Tests

# R3: the language-model policy on CUDA
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_FILTER = "reinforcement text:|fine-tuning:"
dotnet run -c Release --no-build --project tests\Idrak.Tests
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER

# R5: a first real run on the demo suite (adapters into runs\rl-demo)
dotnet run -c Release --project src\Idrak.Cli -- reinforce Qwen/Qwen3-0.6B --env csharp --suite samples\Idrak.Samples.CodingAgent\suite --algorithm grpo --group 8 --iterations 20 --out runs\rl-demo
```

## Risks

- **`FineTuner` split into a session (R3)** is the largest change to working code: `TrainCore` holds adapters, FP8
  copies, recorded graphs, the cluster and checkpoints in one call. Proven by the existing tests and losses bit for bit;
  recorded graphs (`CudaGraphs`) may need a graph per shape class when rollout lengths vary (measured, not assumed).
- **Re-tokenization drift and template drift:** training on re-encoded text or re-rendered turns trains on what the model
  did not sample. Avoided by `ReturnTokens` and the per-turn check (R3).
- **Reward hacking in coding:** an agent can edit tests, hard-code outputs or stop early. The hidden tests are copied in after
  the agent; test edits are reported and can be refused; budgets are penalized; the transcripts are kept for review.
- **Outside commands from a model:** a policy's commands run on the machine. `process` limits programs, time, output and
  memory but not the network; `container` isolates. The run's report names the sandbox and its limits.
- **Device memory with generation and training on one device:** checked by the non-growth tests (R2, R3) and the owner's
  CUDA runs; the defaults come from the device's measured free memory, never from a card.
- **The API grows** (new contracts, the session, `ReturnTokens`) and breaks for consumers of `FineTuningOptimizers`,
  `FineTuningSchedules`, `AnswerMetric` and `AgentTask`: each in the API dump with its reason; no version bump, no
  CHANGELOG (CLAUDE.md).

## Runs

| Date | Phase | Machine | What | Result |
|---|---|---|---|---|
