// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

[assembly: IdrakPluginPackage(typeof(Idrak.Imaging.Plugins.ImagingPluginsPackage))]

namespace Idrak.Imaging.Plugins;

/// <summary>
/// The image and font formats of Idrak.Imaging.Plugins: formats the library does not ship, registered into its
/// registries from outside it (plan 28, decisions 20 and 21: WOFF, the proof that a font format plugs into
/// <see cref="FontFormats"/> and <see cref="FontWriters"/>). Each kind is a plug-in in a namespace of its own
/// (<c>Idrak.Imaging.Plugins.Fonts</c>), whose static <c>FormatPlugin</c> class registers it alone (<c>Register()</c>,
/// <c>Unregister()</c>; see <see cref="IIdrakPlugin"/>). <see cref="RegisterAll"/> registers every one of them through
/// <see cref="IdrakPlugins"/>; nothing is read or written until a file asks for its format, so registering costs only the
/// names.
/// </summary>
public static class ImagingPlugins
{
    /// <summary>
    /// The package, as <c>-P</c> finds it too (<c>[assembly: IdrakPluginPackage]</c>): its name ("Idrak.Imaging.Plugins", as the
    /// registries' origins name it) and <see cref="All"/>.
    /// </summary>
    public static IIdrakPluginPackage Package { get; } = new ImagingPluginsPackage();

    /// <summary>Every plug-in of the package, in the order <see cref="RegisterAll"/> registers them: WOFF fonts (<see cref="Fonts.WoffPlugin"/>).</summary>
    public static IReadOnlyList<IIdrakPlugin> All { get; } = [Fonts.WoffPlugin.Plugin];

    /// <summary>Registers every plug-in of the package (<see cref="IdrakPlugins.RegisterAll(IIdrakPluginPackage)"/>); once: calling it again changes nothing.</summary>
    public static void RegisterAll() => IdrakPlugins.RegisterAll(Package);

    /// <summary>Unregisters every plug-in of the package, the last first, leaving the registries as <see cref="RegisterAll"/> found them.</summary>
    public static void UnregisterAll() => IdrakPlugins.UnregisterAll(Package);
}

/// <summary>
/// The package <c>-P</c> and <see cref="IdrakPlugins.Load"/> find through this assembly's
/// <see cref="IdrakPluginPackageAttribute"/>: <see cref="ImagingPlugins"/>' name and plug-ins.
/// </summary>
internal sealed class ImagingPluginsPackage : IIdrakPluginPackage
{
    /// <inheritdoc/>
    public string Name => "Idrak.Imaging.Plugins";

    /// <inheritdoc/>
    public IReadOnlyList<IIdrakPlugin> All => ImagingPlugins.All;
}
