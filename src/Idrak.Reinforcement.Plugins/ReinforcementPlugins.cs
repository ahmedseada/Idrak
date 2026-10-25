// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

[assembly: IdrakPluginPackage(typeof(Idrak.Reinforcement.Plugins.ReinforcementPluginsPackage))]

namespace Idrak.Reinforcement.Plugins;

/// <summary>
/// The reinforcement-learning environment families of Idrak.Reinforcement.Plugins: environments, and the rewards they
/// score with, that more than one app uses. Each family is a plug-in in a namespace of its own
/// (<c>Idrak.Reinforcement.Plugins.Family</c>), whose static <c>FamilyPlugin</c> class registers it alone
/// (<c>Register()</c>, <c>Unregister()</c>; see <see cref="IIdrakPlugin"/>). <see cref="RegisterAll"/> registers every
/// one of them through <see cref="IdrakPlugins"/>; no environment is created until a trainer asks for its kind, so
/// registering costs only the names.
/// </summary>
public static class ReinforcementPlugins
{
    /// <summary>
    /// The package, as <c>-P</c> finds it too (<c>[assembly: IdrakPluginPackage]</c>): its name ("Idrak.Reinforcement.Plugins", as the
    /// library's "not registered" messages and the registries' origins name it) and <see cref="All"/>.
    /// </summary>
    public static IIdrakPluginPackage Package { get; } = new ReinforcementPluginsPackage();

    /// <summary>
    /// Every family plug-in of the package, in the order <see cref="RegisterAll"/> registers them: agentic coding
    /// (<see cref="Coding.CodingPlugin"/>), agentic coding on C# projects (<see cref="CSharp.CSharpCodingPlugin"/>), then
    /// the classic control environments (<see cref="Classic.ClassicControlPlugin"/>).
    /// </summary>
    public static IReadOnlyList<IIdrakPlugin> All { get; } = [Coding.CodingPlugin.Plugin, CSharp.CSharpCodingPlugin.Plugin, Classic.ClassicControlPlugin.Plugin];

    /// <summary>Registers every family of the package (<see cref="IdrakPlugins.RegisterAll(IIdrakPluginPackage)"/>); once: calling it again changes nothing.</summary>
    public static void RegisterAll() => IdrakPlugins.RegisterAll(Package);

    /// <summary>Unregisters every family of the package, the last first, leaving the registries as <see cref="RegisterAll"/> found them.</summary>
    public static void UnregisterAll() => IdrakPlugins.UnregisterAll(Package);
}

/// <summary>
/// The package <c>-P</c> and <see cref="IdrakPlugins.Load"/> find through this assembly's
/// <see cref="IdrakPluginPackageAttribute"/>: <see cref="ReinforcementPlugins"/>' name and plug-ins.
/// </summary>
internal sealed class ReinforcementPluginsPackage : IIdrakPluginPackage
{
    /// <inheritdoc/>
    public string Name => "Idrak.Reinforcement.Plugins";

    /// <inheritdoc/>
    public IReadOnlyList<IIdrakPlugin> All => ReinforcementPlugins.All;
}
