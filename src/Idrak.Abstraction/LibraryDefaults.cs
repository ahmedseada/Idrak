// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Abstraction;

/// <summary>
/// The built-ins that ship in the Idrak assembly but are found through registries here (its GPU devices, its sample
/// sources, ...). Idrak.Abstraction cannot name them; every such registry calls <see cref="Ensure"/> before its first use,
/// which has Idrak register them, first and once, when the application ships Idrak. An application that references
/// Idrak.Abstraction alone gets the defaults that live here and whatever it registers itself.
/// </summary>
internal static class LibraryDefaults
{
    private static readonly Lock Gate = new();

    private static bool _done;

    /// <summary>Has Idrak register its built-ins, unless done already (on this or another thread, which this then waits for).</summary>
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.LibraryRegistrations", "Idrak")]
    public static void Ensure()
    {
        lock (Gate)   // re-entrant: a registration that touches another registry comes back here and returns
        {
            if (_done)
            {
                return;
            }

            _done = true;
            Type? registrations;
            try
            {
                registrations = Type.GetType("Idrak.LibraryRegistrations, Idrak", throwOnError: false);
            }
            catch (FileLoadException)
            {
                return;
            }

            registrations?.GetMethod("RegisterAll", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static)?.Invoke(null, null);
        }
    }
}
