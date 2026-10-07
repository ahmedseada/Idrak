// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Abstraction;

/// <summary>
/// The built-ins that ship in the first-party assemblies (Idrak, its domain packages) but are found through registries
/// here (GPU devices, sample sources, layer types, ...). Idrak.Abstraction cannot name them; every such registry calls
/// <see cref="Ensure"/> before its first use, which has each assembly the application ships register them, once. An application that references
/// Idrak.Abstraction alone gets the defaults that live here and whatever it registers itself.
/// </summary>
internal static class LibraryDefaults
{
    private static readonly Lock Gate = new();

    private static bool _done;

    // The first-party assemblies that register built-ins, in order (core first: the others build on it). Each may define
    // an internal static class `<assembly>.LibraryRegistrations` with a static `RegisterAll()`; one that is missing
    // (not shipped with the application) is skipped. Planned packages are listed so they work the day they appear.
    private static readonly string[] Assemblies =
        ["Idrak", "Idrak.Gpu", "Idrak.Data", "Idrak.Nlp", "Idrak.LanguageModels", "Idrak.Vision", "Idrak.Diffusion", "Idrak.Audio"];

    /// <summary>Has the first-party assemblies register their built-ins, unless done already (on this or another thread, which this then waits for).</summary>
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.LibraryRegistrations", "Idrak")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Gpu.LibraryRegistrations", "Idrak.Gpu")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Data.LibraryRegistrations", "Idrak.Data")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Nlp.LibraryRegistrations", "Idrak.Nlp")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.LanguageModels.LibraryRegistrations", "Idrak.LanguageModels")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Vision.LibraryRegistrations", "Idrak.Vision")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Diffusion.LibraryRegistrations", "Idrak.Diffusion")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Audio.LibraryRegistrations", "Idrak.Audio")]
    [UnconditionalSuppressMessage("Trimming", "IL2057", Justification = "The type names are the constant first-party names listed above, kept by the DynamicDependency attributes.")]
    public static void Ensure()
    {
        lock (Gate)   // re-entrant: a registration that touches another registry comes back here and returns
        {
            if (_done)
            {
                return;
            }

            _done = true;
            foreach (string assembly in Assemblies)
            {
                Type? registrations;
                try
                {
                    registrations = Type.GetType($"{assembly}.LibraryRegistrations, {assembly}", throwOnError: false);
                }
                catch (Exception e) when (e is FileLoadException or FileNotFoundException or BadImageFormatException)
                {
                    continue;
                }

                registrations?.GetMethod("RegisterAll", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public)
                    ?.Invoke(null, null);
            }
        }
    }
}
