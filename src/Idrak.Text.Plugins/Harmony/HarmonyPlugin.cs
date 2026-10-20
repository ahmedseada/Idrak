// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

namespace Idrak.Text.Plugins.Harmony;

/// <summary>
/// The harmony message format of GPT-OSS models: its tool-call format "harmony" in <see cref="ToolCallFormats"/>
/// (reasoning in the analysis channel, calls as messages to <c>functions.name</c>; <see cref="HarmonyToolCallParser"/>).
/// </summary>
public static class HarmonyPlugin
{
    /// <summary>The tool-call format in <see cref="ToolCallFormats"/>: "harmony".</summary>
    public const string ToolCallFormat = "harmony";

    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("harmony"; listed in <see cref="TextPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new HarmonyRegistration();

    /// <summary>Whether the format is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers the format (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters the format, leaving <see cref="ToolCallFormats"/> as <see cref="Register"/> found it.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class HarmonyRegistration() : IdrakPlugin("harmony")
    {
        protected override void RegisterParts() =>
            Part(() => ToolCallFormats.Register(ToolCallFormat, p => p.Shows("to=functions."), _ => new HarmonyToolCallParser()), () => ToolCallFormats.Unregister(ToolCallFormat));
    }
}
