// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Reflection;
using System.Runtime.CompilerServices;

namespace Idrak.Abstraction;

/// <summary>
/// One registry entry an app registered (see <see cref="Overrides.Report"/>).
/// </summary>
/// <param name="Registry">The registry, for example "RopeScalings".</param>
/// <param name="Name">The entry's name in it.</param>
/// <param name="Implementation">The app's implementation: its type, or the method of a delegate.</param>
/// <param name="Origin">The assembly the app's implementation comes from.</param>
/// <param name="ReplacesDefault">Whether it shadows a library default (false: a name only the app registered).</param>
/// <param name="Guarded">Whether its policy applies (false: the registry's entries cannot fall back, and the app's is used as it is).</param>
/// <param name="Policy">The slot's policy.</param>
/// <param name="ShadowRate">The share of calls compared under <see cref="SlotPolicy.Shadow"/>.</param>
/// <param name="FallBacks">Calls that fell back to the library default so far.</param>
/// <param name="Compared">Calls compared under <see cref="SlotPolicy.Shadow"/> so far.</param>
/// <param name="Differed">Compared calls whose outputs differed, or where the app's implementation threw.</param>
public sealed record SlotOverride(string Registry, string Name, string Implementation, string Origin, bool ReplacesDefault, bool Guarded,
    SlotPolicy Policy, double ShadowRate, long FallBacks, long Compared, long Differed)
{
    /// <summary>What the policy means for this entry, in a few words: "fall back", "throw", "shadow 1%", or why there is none.</summary>
    public string PolicyText => !ReplacesDefault ? "none (no library default)" : !Guarded ? "none (used as it is)" : Policy switch
    {
        SlotPolicy.Shadow => $"shadow {ShadowRate.ToString("0.##%", System.Globalization.CultureInfo.InvariantCulture)}",
        SlotPolicy.Throw => "throw",
        _ => "fall back",
    };

    /// <inheritdoc />
    public override string ToString() =>
        $"{Registry}/{Name}: {Implementation} ({Origin}), {(ReplacesDefault ? "replaces the library default" : "added")}, policy {PolicyText}";
}

/// <summary>
/// The override loop's entry point: what an app has registered over the library's defaults, in every registry
/// (<see cref="Report"/>), and how the library marks its own built-ins as defaults (<see cref="AsLibraryDefaults"/>).
/// </summary>
/// <example>
/// <code>
/// RopeScalings.Register("yarn", MyYarn);                                  // the app's version; the library's stays behind it
/// RopeScalings.SetPolicy("yarn", SlotPolicy.Shadow, shadowRate: 0.05);   // the default answers, 5% of calls compared
/// foreach (var o in Overrides.Report()) Console.WriteLine(o);             // at startup: every override, origin and policy
/// </code>
/// </example>
public static class Overrides
{
    /// <summary>The origin of a library default.</summary>
    public const string Library = "library";

    private static readonly Lock Gate = new();
    private static readonly List<Func<IReadOnlyList<SlotOverride>>> Tables = [];

    [ThreadStatic]
    private static int t_defaults;

    /// <summary>Whether registrations on this thread are the library's built-ins (inside <see cref="AsLibraryDefaults"/>).</summary>
    internal static bool RegisteringDefaults => t_defaults > 0;

    /// <summary>
    /// Every entry an app has registered in the registries in use (a registry nobody has used holds no app entries), by
    /// registry: what it replaces, the assembly it comes from, its policy and what happened so far. Print it at startup to
    /// see which of the library's defaults the app runs without.
    /// </summary>
    public static IReadOnlyList<SlotOverride> Report()
    {
        Func<IReadOnlyList<SlotOverride>>[] tables;
        lock (Gate)
        {
            tables = [.. Tables];
        }

        return [.. tables.SelectMany(t => t()).OrderBy(o => o.Registry, StringComparer.Ordinal)];
    }

    /// <summary>
    /// Runs <paramref name="registrations"/> with every registration it makes on this thread taken as a library default,
    /// not an app override. For the library's packages, which register their built-ins this way (on first use of a
    /// registry); an app has no reason to call it.
    /// </summary>
    public static void AsLibraryDefaults(Action registrations)
    {
        ArgumentNullException.ThrowIfNull(registrations);
        t_defaults++;
        try
        {
            registrations();
        }
        finally
        {
            t_defaults--;
        }
    }

    internal static void Track(Func<IReadOnlyList<SlotOverride>> table)
    {
        lock (Gate)
        {
            Tables.Add(table);
        }
    }

    // The implementation's name and the assembly it comes from: an object's type, a delegate's method (its declaring type,
    // outside the compiler's closure classes, and the method unless the compiler named it), or a name given as it is.
    internal static (string Implementation, string Origin) Describe(object? implementation)
    {
        switch (implementation)
        {
            case null:
                return ("null", "unknown");
            case string given:
                return (given, "unknown");
            case Delegate d:
                var method = d.Method;
                var type = method.DeclaringType;
                while (type is not null && type.IsDefined(typeof(CompilerGeneratedAttribute), inherit: false) && type.DeclaringType is not null)
                {
                    type = type.DeclaringType;
                }

                string name = type?.FullName ?? "?";
                if (!method.Name.Contains('<', StringComparison.Ordinal))
                {
                    name += "." + method.Name;
                }

                return (name, AssemblyName(method.Module.Assembly));
            default:
                var t = implementation.GetType();
                return (t.FullName ?? t.Name, AssemblyName(t.Assembly));
        }
    }

    private static string AssemblyName(Assembly assembly) => assembly.GetName().Name ?? "unknown";
}
