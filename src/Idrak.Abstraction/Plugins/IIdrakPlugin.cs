// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Plugins;

/// <summary>
/// A plug-in: one model family (or another named set of registrations) that lives outside the library and registers its
/// parts (architectures, vision or speech families, tool-call formats, ...) into the library's registries through their
/// public <c>Register</c> methods. Known by name in <see cref="IdrakPlugins"/>.
/// <para>
/// The convention of the plug-in packages (<c>Idrak.Text.Plugins</c>, <c>Idrak.Vision.Plugins</c>,
/// <c>Idrak.Audio.Plugins</c>): each family lives in a namespace of its own (<c>Idrak.Audio.Plugins.Whisper</c>) with a
/// public static class named for it (<c>WhisperPlugin</c>) that holds the family's plug-in (<c>Plugin</c>, usually an
/// <see cref="IdrakPlugin"/>) and has <c>Register()</c> (<c>IdrakPlugins.Register(Plugin)</c>) and <c>Unregister()</c>
/// (<c>IdrakPlugins.Unregister(Plugin)</c>); the package's root class holds the package (a static <c>Package</c>, an
/// <see cref="IIdrakPluginPackage"/> that the assembly also declares with <see cref="IdrakPluginPackageAttribute"/> for
/// hosts that load it themselves), lists every family's plug-in (<c>All</c>) and registers them all with
/// <c>RegisterAll()</c>. Plug-ins use the library's public API only.
/// </para>
/// </summary>
public interface IIdrakPlugin
{
    /// <summary>The plug-in's name, unique among plug-ins (a plug-in registered under a taken name replaces the earlier one).</summary>
    string Name { get; }

    /// <summary>Whether its parts are registered now.</summary>
    bool IsRegistered { get; }

    /// <summary>Registers its parts into the library's registries; once: a second call while registered does nothing.</summary>
    void Register();

    /// <summary>Removes what <see cref="Register"/> registered, leaving the registries as it found them; nothing when not registered.</summary>
    void Unregister();
}
