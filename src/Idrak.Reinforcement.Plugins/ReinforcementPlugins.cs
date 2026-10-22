// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

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
    /// <summary>The package's name, as the library's "not registered" messages name it: "Idrak.Reinforcement.Plugins".</summary>
    public const string Package = "Idrak.Reinforcement.Plugins";

    /// <summary>
    /// Every family plug-in of the package, in the order <see cref="RegisterAll"/> registers them: agentic coding
    /// (<see cref="Coding.CodingPlugin"/>), then agentic coding on C# projects (<see cref="CSharp.CSharpCodingPlugin"/>).
    /// </summary>
    public static IReadOnlyList<IIdrakPlugin> All { get; } = [Coding.CodingPlugin.Plugin, CSharp.CSharpCodingPlugin.Plugin];

    /// <summary>Registers every family of the package (<see cref="IdrakPlugins.RegisterAll"/>); once: calling it again changes nothing.</summary>
    public static void RegisterAll() => IdrakPlugins.RegisterAll(All);

    /// <summary>Unregisters every family of the package, the last first, leaving the registries as <see cref="RegisterAll"/> found them.</summary>
    public static void UnregisterAll() => IdrakPlugins.UnregisterAll(All);

    /// <summary>The entry the idrak tool's <c>-P</c> (<c>--plugin</c>) calls when it loads this assembly: <see cref="RegisterAll"/>.</summary>
    public static void RegisterIdrakPlugin() => RegisterAll();
}
