// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Reinforcement;

namespace Idrak.Reinforcement.Plugins.Coding;

/// <summary>
/// Agentic coding as a plug-in: the coding environment (<see cref="CodingEnvironmentKind"/>, "coding" in
/// <see cref="EnvironmentKinds"/>), its rewards (<see cref="CodingRewards"/>: tests-passed, verify-fraction, rounds and
/// tool-errors in <see cref="RewardFunctions"/>) and the container sandbox (<see cref="ContainerSandbox"/>, "container" in
/// <see cref="CommandSandboxes"/>). The task layer it builds on (<see cref="CodingTask"/>, <see cref="CodingSuite"/>,
/// <see cref="CodingWorkspace"/>, <see cref="CodingVerifier"/>) needs no registration: an app evaluates a model on a task
/// suite with it directly. <see cref="ReinforcementPlugins.RegisterAll"/> registers the plug-in with the package's other
/// families (the idrak tool does at startup); an app may call <see cref="Register"/> alone.
/// </summary>
public static class CodingPlugin
{
    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("coding"; listed in <see cref="ReinforcementPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new CodingRegistration();

    /// <summary>Whether the plug-in is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the environment, the rewards and the sandbox (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters them, leaving <see cref="EnvironmentKinds"/>, <see cref="RewardFunctions"/> and <see cref="CommandSandboxes"/> as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class CodingRegistration() : IdrakPlugin("coding")
    {
        protected override void RegisterParts()
        {
            Part(() => EnvironmentKinds.Register(new CodingEnvironmentKind()), () => EnvironmentKinds.Unregister(CodingEnvironmentKind.KindName));
            foreach (var reward in CodingRewards.All)
            {
                Part(() => RewardFunctions.Register(reward), () => RewardFunctions.Unregister(reward.Name));
            }

            Part(() => CommandSandboxes.Register(new ContainerSandbox()), () => CommandSandboxes.Unregister(CommandSandboxes.Container));
        }
    }
}
