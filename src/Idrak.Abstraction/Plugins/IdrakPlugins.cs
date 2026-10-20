// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Abstraction.Plugins;

/// <summary>
/// The registered plug-ins (<see cref="IIdrakPlugin"/>), by name: what a plug-in package's <c>RegisterAll()</c> and a
/// family's <c>Register()</c> go through. <see cref="Register"/> registers a plug-in's parts and records it; a plug-in
/// registered under a taken name replaces the earlier one (which is unregistered first), so an outside plug-in loaded
/// after a package (the CLI's <c>-P</c>) can take a family's place. The library registers none: model families live in
/// the plug-in packages (<c>Idrak.Text.Plugins</c>, <c>Idrak.Vision.Plugins</c>, <c>Idrak.Audio.Plugins</c>) and in apps.
/// </summary>
public static class IdrakPlugins
{
    // Register and Unregister run whole under it (unregister the earlier, register the new, record it).
    private static readonly Lock Gate = new();

    private static readonly SlotTable<string, IIdrakPlugin> Registry = new(nameof(IdrakPlugins), comparer: StringComparer.Ordinal,
        unguarded: "a plug-in is not called per use: it registers its parts into the library's registries, which guard them");

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

            Registry.Register(plugin.Name, plugin, plugin.GetType().Assembly, plugin);
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

    /// <summary>The recorded plug-in names, in the order they were registered.</summary>
    public static IReadOnlyCollection<string> Names => Registry.Keys;

    /// <summary>The plug-in recorded as <paramref name="name"/>, or null.</summary>
    public static IIdrakPlugin? Find(string name) => Registry.Find(name);

    /// <summary>The plug-in recorded as <paramref name="name"/>.</summary>
    /// <exception cref="NotSupportedException">None is; the message names the plug-in packages.</exception>
    public static IIdrakPlugin Get(string name) => Registry.Find(name)
        ?? throw new NotSupportedException($"No plug-in '{name}' is registered (registered: {(Registry.Keys.Count == 0 ? "none" : string.Join(", ", Registry.Keys))}); "
                                           + "add it with IdrakPlugins.Register, or reference Idrak.Text.Plugins, Idrak.Vision.Plugins or Idrak.Audio.Plugins "
                                           + "and call its RegisterAll() (TextPlugins, VisionPlugins, AudioPlugins).");

    /// <summary>The assembly that brings the plug-in <paramref name="name"/>; null when none is recorded.</summary>
    public static string? Origin(string name) => Registry.Origin(name);
}
