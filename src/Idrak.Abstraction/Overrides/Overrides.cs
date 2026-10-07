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
/// <param name="Failures">Calls of the app's implementation that threw so far.</param>
/// <param name="FallBacks">Of those, the calls that fell back to the library default.</param>
/// <param name="Compared">Calls compared under <see cref="SlotPolicy.Shadow"/> so far.</param>
/// <param name="Differed">Compared calls whose outputs differed, or where the app's implementation threw.</param>
public sealed record SlotOverride(string Registry, string Name, string Implementation, string Origin, bool ReplacesDefault, bool Guarded,
    SlotPolicy Policy, double ShadowRate, long Failures, long FallBacks, long Compared, long Differed)
{
    /// <summary>What the policy means for this entry, in a few words: "fall back", "throw", "shadow 1%", or why there is none.</summary>
    public string PolicyText => !ReplacesDefault ? "none (no library default)" : !Guarded ? "none (used as it is)" : Policy switch
    {
        SlotPolicy.Shadow => $"shadow {ShadowRate.ToString("0.##%", System.Globalization.CultureInfo.InvariantCulture)}",
        SlotPolicy.FallBack => "fall back",
        _ => "throw",
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
/// RopeScalings.SetPolicy("yarn", SlotPolicy.FallBack);                   // a call that throws is answered by the library's
/// RopeScalings.SetPolicy("yarn", SlotPolicy.Shadow, shadowRate: 0.05);   // or: the default answers, 5% of calls compared
/// foreach (var o in Overrides.Report()) Console.WriteLine(o);             // at startup: every override, origin and policy
/// </code>
/// </example>
public static class Overrides
{
    /// <summary>The origin of a library default.</summary>
    public const string Library = "library";

    /// <summary>
    /// The environment variable that sets slots' first policy (before any <c>SetPolicy</c>): a policy for every slot
    /// (<c>throw</c>, <c>fallback</c>, <c>shadow</c>, or <c>shadow:RATE</c> with RATE from 0 to 1), and/or
    /// <c>Registry/name=policy</c> entries for single slots, separated by commas (<c>RopeScalings/yarn=fallback</c>).
    /// </summary>
    public const string PolicyVariable = "IDRAK_OVERRIDE_POLICY";

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

    /// <summary>
    /// The policy and shadow rate <see cref="PolicyVariable"/> set to <paramref name="setting"/> gives the slot
    /// <paramref name="name"/> of <paramref name="registry"/>: its own entry, else the policy for every slot, else
    /// <see cref="SlotPolicy.Throw"/>. Names compare ignoring case; entries it cannot read are ignored.
    /// </summary>
    internal static (SlotPolicy Policy, double ShadowRate) PolicyFrom(string? setting, string registry, string name)
    {
        (SlotPolicy, double)? every = null, mine = null;
        foreach (string part in (setting ?? "").Split(',', StringSplitOptions.TrimEntries | StringSplitOptions.RemoveEmptyEntries))
        {
            int equals = part.IndexOf('=', StringComparison.Ordinal);
            var policy = Parse(equals < 0 ? part : part[(equals + 1)..]);
            if (policy is null)
            {
                continue;
            }

            if (equals < 0)
            {
                every = policy;
            }
            else if (string.Equals(part[..equals].Trim(), $"{registry}/{name}", StringComparison.OrdinalIgnoreCase))
            {
                mine = policy;
            }
        }

        return mine ?? every ?? (SlotPolicy.Throw, Slot.DefaultShadowRate);

        static (SlotPolicy, double)? Parse(string text)
        {
            string[] pieces = text.Trim().Split(':', 2);
            double rate = Slot.DefaultShadowRate;
            if (pieces.Length == 2 && !(double.TryParse(pieces[1], System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out rate)
                                        && rate is >= 0 and <= 1))
            {
                return null;
            }

            return pieces[0].ToLowerInvariant() switch
            {
                "throw" => (SlotPolicy.Throw, rate),
                "fallback" or "fall-back" => (SlotPolicy.FallBack, rate),
                "shadow" => (SlotPolicy.Shadow, rate),
                _ => null,
            };
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
