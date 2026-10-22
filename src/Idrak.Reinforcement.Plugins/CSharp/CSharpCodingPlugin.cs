// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;
using Idrak.Abstraction.Reinforcement;
using Idrak.Reinforcement.Plugins.Coding;
using Idrak.Text.Plugins.CSharp;

namespace Idrak.Reinforcement.Plugins.CSharp;

/// <summary>
/// Agentic coding on C# projects as a plug-in: the <c>csharp</c> environment (<see cref="CSharpEnvironmentVariant"/>, the
/// coding environment with dotnet's defaults, in <see cref="EnvironmentKinds"/>) and its rewards (<see cref="CSharpRewards"/>:
/// csharp-tests and csharp-build in <see cref="RewardFunctions"/>). It builds on the coding plug-in
/// (<see cref="CodingPlugin"/>) and the C# coding language of Idrak.Text.Plugins (<see cref="CSharpLanguagePlugin"/>) and
/// copies neither: registering it registers them when they are missing, and unregistering it takes back only what it
/// registered.
/// </summary>
public static class CSharpCodingPlugin
{
    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("csharp"; listed in <see cref="ReinforcementPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new CSharpCodingRegistration();

    /// <summary>Whether the plug-in is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the environment and the rewards, and the coding plug-in and the C# language when missing (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters them, leaving <see cref="EnvironmentKinds"/>, <see cref="RewardFunctions"/> and the plug-ins it needs as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class CSharpCodingRegistration() : IdrakPlugin("csharp")
    {
        protected override void RegisterParts()
        {
            Requires(CodingPlugin.Plugin);
            Requires(CSharpLanguagePlugin.Plugin);
            Part(() => EnvironmentKinds.Register(new CodingEnvironmentKind(new CSharpEnvironmentVariant())), () => EnvironmentKinds.Unregister(CSharpEnvironmentVariant.KindName));
            foreach (var reward in CSharpRewards.All)
            {
                Part(() => RewardFunctions.Register(reward), () => RewardFunctions.Unregister(reward.Name));
            }
        }
    }
}
