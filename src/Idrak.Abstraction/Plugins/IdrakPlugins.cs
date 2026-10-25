// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction.Plugins;

/// <summary>
/// The registered plug-ins (<see cref="IIdrakPlugin"/>), by name: what a plug-in package's <c>RegisterAll()</c> and a
/// family's <c>Register()</c> go through. <see cref="Register"/> registers a plug-in's parts and records it; a plug-in
/// registered under a taken name replaces the earlier one (which is unregistered first), so an outside plug-in loaded
/// after a package (the CLI's <c>-P</c>, through <see cref="Load"/>) can take a family's place. The library registers none: model families live in
/// the plug-in packages (<c>Idrak.Text.Plugins</c>, <c>Idrak.Vision.Plugins</c>, <c>Idrak.Audio.Plugins</c>; environment
/// families in <c>Idrak.Reinforcement.Plugins</c>) and in apps.
/// </summary>
public static class IdrakPlugins
{
    // Register and Unregister run whole under it (unregister the earlier, register the new, record it).
    private static readonly Lock Gate = new();

    private static readonly SlotTable<string, IIdrakPlugin> Registry = new(nameof(IdrakPlugins), comparer: StringComparer.Ordinal,
        unguarded: "a plug-in is not called per use: it registers its parts into the library's registries, which guard them");

    // Each assembly's plug-in package, read from its attribute and made once (null inside when it declares none).
    private static readonly ConditionalWeakTable<Assembly, StrongBox<IIdrakPluginPackage?>> Packages = new();

    /// <summary>
    /// Registers <paramref name="plugin"/>'s parts and records it under its <see cref="IIdrakPlugin.Name"/>. Once: the same
    /// plug-in again does nothing. Another plug-in recorded under that name is unregistered first (and registered again
    /// when this one fails, so a failed registration leaves the registries as they were).
    /// </summary>
    public static void Register(IIdrakPlugin plugin)
    {
        ArgumentNullException.ThrowIfNull(plugin);
        ArgumentException.ThrowIfNullOrWhiteSpace(plugin.Name);
        lock (Gate)
        {
            var earlier = Registry.Find(plugin.Name);
            if (ReferenceEquals(earlier, plugin))
            {
                plugin.Register();                                       // nothing when registered; again after its own Unregister
                return;
            }

            earlier?.Unregister();
            try
            {
                plugin.Register();
            }
            catch
            {
                earlier?.Register();
                throw;
            }

            // Recorded as its own assembly's, also when another plug-in's part registers it (Requires).
            Overrides.AsRegisteredBy(plugin.GetType().Assembly, () => Registry.Register(plugin.Name, plugin, implementation: plugin));
        }
    }

    /// <summary>Registers each of <paramref name="plugins"/> in order (<see cref="Register"/>): a plug-in package's <c>RegisterAll()</c>.</summary>
    public static void RegisterAll(IEnumerable<IIdrakPlugin> plugins)
    {
        ArgumentNullException.ThrowIfNull(plugins);
        foreach (var plugin in plugins)
        {
            Register(plugin);
        }
    }

    /// <summary>Registers every plug-in of <paramref name="package"/>, in order (<see cref="Register"/>): a plug-in package's <c>RegisterAll()</c>.</summary>
    public static void RegisterAll(IIdrakPluginPackage package)
    {
        ArgumentNullException.ThrowIfNull(package);
        RegisterAll(package.All);
    }

    /// <summary>
    /// Unregisters the plug-in recorded as <paramref name="name"/> (its parts taken back) and forgets it; false when none is.
    /// </summary>
    public static bool Unregister(string name)
    {
        ArgumentNullException.ThrowIfNull(name);
        lock (Gate)
        {
            if (Registry.Find(name) is not { } plugin)
            {
                return false;
            }

            plugin.Unregister();
            Registry.Unregister(name);
            return true;
        }
    }

    /// <summary>
    /// Unregisters <paramref name="plugin"/> (its parts taken back), and forgets it when it is the one recorded under its
    /// name; another plug-in recorded under that name stays. False when it was not registered.
    /// </summary>
    public static bool Unregister(IIdrakPlugin plugin)
    {
        ArgumentNullException.ThrowIfNull(plugin);
        lock (Gate)
        {
            bool registered = plugin.IsRegistered;
            plugin.Unregister();
            if (ReferenceEquals(Registry.Find(plugin.Name), plugin))
            {
                Registry.Unregister(plugin.Name);
            }

            return registered;
        }
    }

    /// <summary>Unregisters each of <paramref name="plugins"/>, the last first (<see cref="Unregister(IIdrakPlugin)"/>): a plug-in package's <c>UnregisterAll()</c>.</summary>
    public static void UnregisterAll(IReadOnlyList<IIdrakPlugin> plugins)
    {
        ArgumentNullException.ThrowIfNull(plugins);
        for (int i = plugins.Count - 1; i >= 0; i--)
        {
            Unregister(plugins[i]);
        }
    }

    /// <summary>Unregisters every plug-in of <paramref name="package"/>, the last first: a plug-in package's <c>UnregisterAll()</c>.</summary>
    public static void UnregisterAll(IIdrakPluginPackage package)
    {
        ArgumentNullException.ThrowIfNull(package);
        UnregisterAll(package.All);
    }

    /// <summary>
    /// The plug-in package <paramref name="assembly"/> declares (<see cref="IdrakPluginPackageAttribute"/>), its attribute
    /// read and the package made once per assembly; null when it declares none.
    /// </summary>
    /// <exception cref="InvalidOperationException">The attribute names a type that is not an <see cref="IIdrakPluginPackage"/>.</exception>
    public static IIdrakPluginPackage? PackageOf(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        return Packages.GetValue(assembly, static a => new(Create(a))).Value;

        static IIdrakPluginPackage? Create(Assembly assembly) => assembly.GetCustomAttribute<IdrakPluginPackageAttribute>() is not { } declared ? null
            : Activator.CreateInstance(declared.Package) as IIdrakPluginPackage
              ?? throw new InvalidOperationException($"{assembly.GetName().Name}: [assembly: IdrakPluginPackage(typeof({declared.Package.FullName}))] names a type that is not an "
                                                     + $"{nameof(IIdrakPluginPackage)}.");
    }

    /// <summary>
    /// Registers what <paramref name="assembly"/> declares, as a host that loads the assembly itself does (the idrak tool's
    /// <c>-P</c>): for a library assembly loaded after the application started (<see cref="LibraryRegistrationsAttribute"/>),
    /// its built-ins, as the registries' library defaults; then every plug-in of its package (<see cref="PackageOf"/>,
    /// <see cref="RegisterAll(IIdrakPluginPackage)"/>). Once: loading it again registers nothing twice.
    /// </summary>
    /// <returns>False when the assembly declares neither a plug-in package nor library registrations.</returns>
    public static bool Load(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        bool library = LibraryDefaults.Include(assembly);
        if (PackageOf(assembly) is not { } package)
        {
            return library;
        }

        RegisterAll(package);
        return true;
    }

    /// <summary>The recorded plug-in names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The plug-in recorded as <paramref name="name"/>, or null.</summary>
    public static IIdrakPlugin? Find(string name) => Registry.Find(name);

    /// <summary>The plug-in recorded as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names the plug-in packages.</exception>
    public static IIdrakPlugin Get(string name) => Registry.Find(name)
        ?? throw new NotRegisteredException(nameof(IdrakPlugins), name, Registry.Keys,
            $"No plug-in '{name}' is registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
            + "add it with IdrakPlugins.Register, or reference Idrak.Text.Plugins, Idrak.Vision.Plugins, Idrak.Audio.Plugins or "
            + "Idrak.Reinforcement.Plugins and call its RegisterAll() (TextPlugins, VisionPlugins, AudioPlugins, ReinforcementPlugins).");

    /// <summary>The assembly that brings the plug-in <paramref name="name"/>; null when none is recorded.</summary>
    public static string? Origin(string name) => Registry.Origin(name);
}
