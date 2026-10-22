# Idrak.Reinforcement

Reinforcement learning for Idrak (no dependencies beyond Idrak): an agent acts in an environment, is rewarded, and its
policy is updated from what it did. This package holds the general algorithms, card-agnostic and with no environment,
task or model family in it; it does not reference Idrak.Text, so a game agent or a vision policy pulls in no chat
templates or coding agent.

## Contracts

The contracts are in `Idrak.Abstraction.Reinforcement` (add the `using` line to name them), so every package reaches
them: `IEnvironment<TObservation, TAction>` and `Space` (an episode's reset and steps, its observation and action
spaces), `IEnvironmentKind` and `EnvironmentKinds` (environments by name), `IPolicy<TObservation, TAction>` (acting and
updating), `IRewardFunction` and `RewardFunctions` (scores of answers), `IAdvantageEstimator` and `AdvantageEstimators`
(advantages and returns in a fixed summation order), `IPolicyLoss` and `PolicyLosses` (the algorithms' losses), and the
data passed between them: `Trajectory`, `ActionRecord`, `RolloutBatch` (pooled arrays, returned on `Dispose`).

This package registers its estimators and losses into those registries as library defaults when it ships with the
application: an app's registration under one of their names overrides it, `Unregister` brings it back, and `SetPolicy`
opts the slot into `FallBack` to the library's or `Shadow`. Environments live in plug-ins
([Idrak.Reinforcement.Plugins](../Idrak.Reinforcement.Plugins/README.md)) and apps; a kind that is not registered is
refused with "not registered". The contract suites (an environment's reset and step rules and seeded replay, a reward's
batch against its single form, an estimator's fixed order, a loss's gradient) are in `Idrak.Abstraction.Testing`
(`Conformance.CheckEnvironmentKind`, `CheckRewardFunction`, `CheckAdvantageEstimator`, `CheckPolicyLoss`).

## Install

```bash
dotnet add package Idrak.Reinforcement
```

Part of [Idrak](https://www.nuget.org/packages/Idrak), a self-contained deep-learning library for .NET.

License: Apache 2.0.
