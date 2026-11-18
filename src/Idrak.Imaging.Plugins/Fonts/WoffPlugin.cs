// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

namespace Idrak.Imaging.Plugins.Fonts;

/// <summary>
/// WOFF 1.0 fonts as a plug-in ("woff"): <see cref="WoffFormat"/> in <see cref="FontFormats"/> (a .woff file read by its
/// <c>wOFF</c> signature, its font inside handed to the library's reader) and <see cref="WoffWriter"/> in
/// <see cref="FontWriters"/> (a face written as TrueType through the library's "truetype" writer, then each table
/// compressed). Never a library default: the library reads and writes TrueType, and this plug-in, registered from
/// outside it, adds a format to both registries (plan 28, decision 20). <see cref="ImagingPlugins.RegisterAll"/>
/// registers it with the package's other plug-ins (the idrak tool does at startup); an app may call <see cref="Register"/>
/// alone. Each registration makes a <see cref="WoffFormat"/> and a <see cref="WoffWriter"/> with their defaults, which
/// read <see cref="WoffFormat.MaxFontBytesVariable"/> and <see cref="WoffWriter.CompressionLevelVariable"/> then; an app
/// that wants other settings registers its own instances with <see cref="FontFormats.Register"/> and
/// <see cref="FontWriters.Register"/>.
/// </summary>
public static class WoffPlugin
{
    /// <summary>The plug-in, as <see cref="IdrakPlugins"/> records it ("woff"; listed in <see cref="ImagingPlugins.All"/>).</summary>
    public static IIdrakPlugin Plugin { get; } = new WoffRegistration();

    /// <summary>Whether the plug-in is registered.</summary>
    public static bool IsRegistered => Plugin.IsRegistered;

    /// <summary>Registers "woff" in <see cref="FontFormats"/> and <see cref="FontWriters"/> (through <see cref="IdrakPlugins"/>); once: calling it again changes nothing.</summary>
    public static void Register() => IdrakPlugins.Register(Plugin);

    /// <summary>Unregisters them, leaving both registries as <see cref="Register"/> found them.</summary>
    public static void Unregister() => IdrakPlugins.Unregister(Plugin);

    private sealed class WoffRegistration() : IdrakPlugin(WoffFormat.FormatName)
    {
        protected override void RegisterParts()
        {
            var format = new WoffFormat();
            var writer = new WoffWriter();
            Part(() => FontFormats.Register(format), () => FontFormats.Unregister(WoffFormat.FormatName));
            Part(() => FontWriters.Register(writer), () => FontWriters.Unregister(WoffWriter.WriterName));
        }
    }
}
