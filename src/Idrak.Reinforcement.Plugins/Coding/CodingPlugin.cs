// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// Agentic coding as a plug-in: the coding environment (<see cref="CodingEnvironmentKind"/>, "coding" in
/// <c>EnvironmentKinds</c>), its rewards (<see cref="CodingRewards"/>: tests-passed, verify-fraction, rounds and
/// tool-errors in <c>RewardFunctions</c>), the container sandbox (<see cref="ContainerSandbox"/>, "container" in
/// <c>CommandSandboxes</c>) and the coding tasks' proof (<see cref="CodingTaskVerifier"/>, "coding" in
/// Idrak.Data's <c>TaskVerifiers</c>, which proves the drafts of <c>idrak data code</c>). The task layer it builds on (<see cref="CodingTask"/>, <see cref="CodingSuite"/>,
/// <see cref="CodingWorkspace"/>, <see cref="CodingVerifier"/>) needs no registration: an app evaluates a model on a task
/// suite with it directly. <c>ReinforcementPlugins.RegisterAll</c> registers the plug-in with the package's other
/// families (the idrak tool does at startup); an app may call <c>Register</c> alone.
/// </summary>
public static class CodingPlugin
{
}
