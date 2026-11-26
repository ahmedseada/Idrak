// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// Agentic coding as a plug-in: the coding environment (<see cref="CodingEnvironmentKind"/>, "coding"), its rewards
/// (<see cref="CodingRewards"/>: tests-passed, verify-fraction, rounds and tool-errors), the container sandbox
/// (<see cref="ContainerSandbox"/>, "container") and the coding tasks' proof (<see cref="CodingTaskVerifier"/>,
/// "coding", which proves the drafts of <c>idrak data code</c>). The task layer it builds on (<see cref="CodingTask"/>,
/// <see cref="CodingSuite"/>, <see cref="CodingWorkspace"/>, <see cref="CodingVerifier"/>) is used directly: an app
/// evaluates a model on a task suite with it. Picking any of them by its name is not on the new design yet.
/// </summary>
public static class CodingPlugin
{
}
