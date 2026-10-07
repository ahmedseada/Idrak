// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Abstraction;

/// <summary>
/// The built-ins that ship in the first-party assemblies (Idrak, its domain packages) but are found through registries
/// here (GPU devices, sample sources, layer types, ...). Idrak.Abstraction cannot name them; every such registry calls
/// <see cref="Ensure"/> with its own type before its first use, which has each assembly the application ships register
/// that registry's built-ins, once. Only the registries an application uses pay for their built-ins: training a network
/// never builds the GGUF tables or the ONNX translators. An application that references Idrak.Abstraction alone gets the
/// defaults that live here and whatever it registers itself.
/// </summary>
internal static class LibraryDefaults
{
    private static readonly Lock Gate = new();

    // The registries whose built-ins are registered (or being registered: a registration may use another registry).
    private static readonly HashSet<Type> Done = [];

    // The RegisterFor methods of the first-party assemblies the application ships, found on the first Ensure.
    private static List<System.Reflection.MethodInfo>? _registrations;

    // The first-party assemblies that register built-ins, in order (core first: the others build on it). Each may define
    // an internal static class `<assembly>.LibraryRegistrations` with a static `RegisterFor(Type registry)`; one that is
    // missing (not shipped with the application) is skipped. Planned packages are listed so they work the day they appear.
    private static readonly string[] Assemblies =
        ["Idrak", "Idrak.Gpu", "Idrak.Data", "Idrak.Nlp", "Idrak.Vision", "Idrak.Diffusion", "Idrak.Audio"];

    /// <summary>
    /// Has the first-party assemblies register their built-ins of <paramref name="registry"/>, unless done already (on
    /// this or another thread, which this then waits for).
    /// </summary>
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.LibraryRegistrations", "Idrak")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Gpu.LibraryRegistrations", "Idrak.Gpu")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Data.LibraryRegistrations", "Idrak.Data")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Nlp.LibraryRegistrations", "Idrak.Nlp")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Vision.LibraryRegistrations", "Idrak.Vision")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Diffusion.LibraryRegistrations", "Idrak.Diffusion")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.NonPublicMethods, "Idrak.Audio.LibraryRegistrations", "Idrak.Audio")]
    public static void Ensure(Type registry)
    {
        lock (Gate)   // re-entrant: a registration that touches another registry comes back here
        {
            if (!Done.Add(registry))
            {
                return;
            }

            _registrations ??= Find();
            object[] arguments = [registry];
            Overrides.AsLibraryDefaults(() =>   // the built-ins are the slots' library defaults, not overrides
            {
                foreach (var register in _registrations)
                {
                    register.Invoke(null, arguments);
                }
            });
        }
    }

    [UnconditionalSuppressMessage("Trimming", "IL2057", Justification = "The type names are the constant first-party names listed above, kept by the DynamicDependency attributes on Ensure.")]
    private static List<System.Reflection.MethodInfo> Find()
    {
        var found = new List<System.Reflection.MethodInfo>();
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

            if (registrations?.GetMethod("RegisterFor", System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.Public,
                    [typeof(Type)]) is { } register)
            {
                found.Add(register);
            }
        }

        return found;
    }
}
