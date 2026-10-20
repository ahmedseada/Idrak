// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

namespace Idrak.Text.Plugins;

/// <summary>
/// The text model families (decoders, encoders, encoder-decoders, their GGUF mappings, tool-call formats and RoPE
/// scalings) of Idrak.Text.Plugins. Each family is a plug-in in a namespace of its own
/// (<c>Idrak.Text.Plugins.Family</c>), whose static <c>FamilyPlugin</c> class registers it alone (<c>Register()</c>,
/// <c>Unregister()</c>; see <see cref="IIdrakPlugin"/>). <see cref="RegisterAll"/> registers every one of them through
/// <see cref="IdrakPlugins"/>; no model is loaded until one asks for its family, so registering costs only the names.
/// </summary>
public static class TextPlugins
{
    /// <summary>The package's name, as the library's "not registered" messages name it: "Idrak.Text.Plugins".</summary>
    public const string Package = "Idrak.Text.Plugins";

    /// <summary>Every family plug-in of the package, in the order <see cref="RegisterAll"/> registers them: BERT (<see cref="Bert.BertPlugin"/>).</summary>
    public static IReadOnlyList<IIdrakPlugin> All { get; } = [Bert.BertPlugin.Plugin];

    /// <summary>Registers every family of the package (<see cref="IdrakPlugins.RegisterAll"/>); once: calling it again changes nothing.</summary>
    public static void RegisterAll() => IdrakPlugins.RegisterAll(All);

    /// <summary>Unregisters every family of the package, the last first, leaving the registries as <see cref="RegisterAll"/> found them.</summary>
    public static void UnregisterAll() => IdrakPlugins.UnregisterAll(All);

    /// <summary>The entry the idrak tool's <c>-P</c> (<c>--plugin</c>) calls when it loads this assembly: <see cref="RegisterAll"/>.</summary>
    public static void RegisterIdrakPlugin() => RegisterAll();
}
