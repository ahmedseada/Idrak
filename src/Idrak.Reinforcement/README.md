# Idrak.Reinforcement

Reinforcement learning for Idrak (no dependencies beyond Idrak): an agent acts in an environment, is rewarded, and its
policy is updated from what it did. This package holds the general algorithms, card-agnostic and with no environment,
task or model family in it; it does not reference Idrak.Text, so a game agent or a vision policy pulls in no chat
templates or coding agent.

## Contracts

The contracts are in `Idrak.Abstraction.Reinforcement` (add the `using` line to name them), so every package reaches
them: `IEnvironment<TObservation, TAction>` and `Space` (an episode's reset and steps, its observation and action
spaces), `IEnvironmentKind` and `EnvironmentKinds` (environments by name), `IPolicy<TObservation, TAction>` (acting and
updating) and `IRolloutRanks<TObservation, TAction>` (a policy on every rank of a cluster, which gathers the ranks' episodes), `IRewardFunction` and `RewardFunctions` (scores of answers), `IAdvantageEstimator` and `AdvantageEstimators`
(advantages and returns in a fixed summation order), `IPolicyLoss` and `PolicyLosses` (the algorithms' losses), and the
data passed between them: `Trajectory`, `ActionRecord`, `RolloutBatch` (pooled arrays, returned on `Dispose`).

This package registers its estimators and losses into those registries as library defaults when it ships with the
application: an app's registration under one of their names overrides it, `Unregister` brings it back, and `SetPolicy`
opts the slot into `FallBack` to the library's or `Shadow`. Environments live in plug-ins
([Idrak.Reinforcement.Plugins](../Idrak.Reinforcement.Plugins/README.md)) and apps; a kind that is not registered is
refused with "not registered". The contract suites (an environment's reset and step rules and seeded replay, a reward's
batch against its single form, an estimator's fixed order, a loss's gradient) are in `Idrak.Abstraction.Testing`
(`Conformance.CheckEnvironmentKind`, `CheckRewardFunction`, `CheckAdvantageEstimator`, `CheckPolicyLoss`).

## What it holds

- **Advantage estimators** (`AdvantageEstimators`, registered as library defaults): `gae` (generalized advantage
  estimation, Schulman et al. 2015, arXiv:1506.02438: γ, λ, a critic's values), `group` (GRPO's group-relative
  advantages, Shao et al. 2024, arXiv:2402.03300: an episode's return less its group's mean, over the group's deviation
  or not), `leave-one-out` (RLOO, Ahmadian et al. 2024, arXiv:2402.14740) and `returns` (REINFORCE's discounted returns,
  less a baseline or not). Every sum runs in a fixed order, so a batch gives the same advantages bit for bit anywhere.
- **Policy losses** (`PolicyLosses`, the algorithms): `reinforce`, `ppo` (a clipped ratio, a value loss, an entropy bonus;
  Schulman et al. 2017, arXiv:1707.06347) and `grpo` (a clipped ratio and a KL penalty to the reference by the k1, k2 or k3
  estimator, normalized per sequence, per token or by a constant), as tensor operations with autograd.
- **`RolloutCollector`**: plays a rollout's episodes at once, an environment each (made once and reused), the policy asked
  for every episode's action in one call, the environments stepped in parallel only where it measured faster; scratch
  buffers reused, episodes numbered and sorted by number, so threads change nothing. On a cluster (a policy that is an
  `IRolloutRanks`: the language-model policy on a fine-tuning session opened with a cluster) the episodes are cut into
  shards (`RolloutSpread`: `IDRAK_RL_SHARDS`, one per rank by default; `IDRAK_RL_SPREAD` blocks or interleaved), each rank
  plays its shards' episodes in environments of its own, and the policy gathers every rank's in rank order; a seeded run
  with the same shard count gives the same episodes on any number of ranks.
- **`ReinforcementTrainer`**: collect, estimate, `Epochs` passes of mini-batch updates; the gradient clip and a schedule from
  `LearningRateSchedules` for a policy that exposes its optimizer (`IScheduledPolicy`); greedy evaluation episodes with
  seeds of their own; checkpoints (the policy with its optimizer, the iteration, the updates, the seed) and a resume that
  continues bit for bit; a clean stop between iterations; a control point at each iteration (`Steering`, from
  `RunSteerings.For`: the cluster dashboard's or a control file's pause, resume and stop, a stop agreed by every rank);
  progress through `IProgressReporter`, the library's telemetry, and a JSON line per iteration (with its pass rate,
  `PassReturn`). Every draw derives from the run's seed and where the run is.
- **`ModulePolicy`**: a policy over a `Module`: categorical over discrete actions or Gaussian over a box, a critic or not,
  a frozen reference or not; its updates are autograd and an optimizer made through `OptimizerKinds`.
- **`PolicyNetworks`**: a module policy or DQN's Q-network and target built from settings for an environment kind's spaces
  (`ModulePolicyOptions`' hidden widths, activation and value head), what `idrak reinforce --env cartpole` trains.
- **Settings**: `ReinforcementOptions` and `ModulePolicyOptions`, each option with a default and an `IDRAK_RL_*` variable
  (`FromEnvironment`; listed by `idrak help env`).
- **Off-policy learning** (`Idrak.Reinforcement.Abstractions` holds its contract): `ReplayBuffer` (a ring of the last N
  transitions in arrays rented once, sampled from a seed, saved and loaded bit for bit), `QLosses` (`IQLoss`, the
  Q-learning losses by name: `dqn`, Mnih et al. 2015, a Huber loss against a target network, double Q-learning (van
  Hasselt et al. 2016) as its option `double`) and `OffPolicyTrainer` (DQN on a `Module` Q-network: ε-greedy rounds over
  several environments, updates on mini-batches from the buffer, a target network copied or Polyak-averaged, a schedule
  from `LearningRateSchedules`, greedy evaluations, checkpoints that resume bit for bit with the episodes under way
  replayed from their seeds and actions, progress, telemetry and a JSON line per report). Its settings are
  `OffPolicyOptions` (`IDRAK_RL_DQN_*` and the shared `IDRAK_RL_SEED`, `IDRAK_RL_GAMMA` and the others it names).

## Install

```bash
dotnet add package Idrak.Reinforcement
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

License: Apache 2.0.
