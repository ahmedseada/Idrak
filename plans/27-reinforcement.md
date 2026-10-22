# Plan 27: reinforcement learning (`Idrak.Reinforcement`, `Idrak.Reinforcement.Plugins`)

**Status (2026-10-22, `abstraction`):** planned 2026-10-21 from the owner's decisions, against the code at acfc83b (every
"Comes from" below was found in the source). The owner answered the open questions and started the plan on 2026-10-22
("go with recommendation", at most two agents): R0 and R1 run side by side.

| Phase | What | Lane | State |
|---|---|---|---|
| R0 | What moves out of `Idrak.Text` into the library: optimizers and schedules by name as registries, the answer metrics into `TuningMetrics`, `LogSigmoid` into core | A | In progress (2026-10-22) |
| R1 | The contracts (`Idrak.Abstraction.Reinforcement`) and the two package skeletons (`Idrak.Reinforcement`, `Idrak.Reinforcement.Plugins`: registration, the CLI) | B | In progress (2026-10-22) |
| R2 | `Idrak.Reinforcement`, the general part: rollouts, advantages, policy and value losses, the on-policy trainer, a module policy, checkpoints, progress, settings | A | Planned |
| R3 | The language-model policy in `Idrak.Text`: a fine-tuning session, sampled tokens recorded, the `prompts` environment, the text rewards, on-policy distillation from a teacher (alone or beside a reward) | B | Planned |
| R4 | `Idrak.Reinforcement.Plugins.Coding`: the coding task layer moves out of `Idrak.Text`, the coding environment, its rewards, the command sandbox (`process`, `container`) | B | Planned |
| R5 | `idrak reinforce`, the settings carried into the CLI and samples, the CodingAgent sample on the library, a small demo suite | A | Planned |
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
| 2 | `Idrak.Reinforcement.Plugins.Classic` | `ClassicControlPlugin` ("classic") | `EnvironmentKinds`: `bandit`, `cartpole`, `pendulum`, `mountain-car` | new | R6 (optional) |

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

## Phase R4: `Idrak.Reinforcement.Plugins.Coding`

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
  solution), each passing `agent-check`, for the owner's first real run.

**Done when:** `cli reinforce` runs a seeded GRPO iteration on the test's tiny model (`prompts`) and on the test's coding
suite with the fake model, `--list` names every registry's entries, `--resume` continues, the variables table test
passes, and `agent-check` passes on the demo suite (CPU here).

**Tests (CPU here):** one build; `cli reinforce|cli health|cli tune|plugins: coding|reinforcement text:|public API|
abstraction inventory`; the CodingAgent sample built.

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
| 4 | R5, the CLI and samples (after R4) | none |
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
dotnet run -c Release --project src\Idrak.Cli -- reinforce Qwen/Qwen3-0.6B --env coding --suite samples\Idrak.Samples.CodingAgent\suite --algorithm grpo --group 8 --iterations 20 --out runs\rl-demo
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
