// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;
using System.Reflection;

namespace Idrak.Abstraction;

/// <summary>
/// The built-ins that ship in the first-party assemblies (Idrak, its domain packages) but are found through registries
/// here (GPU devices, sample sources, layer types, ...). Idrak.Abstraction cannot name them; every such registry calls
/// <see cref="Ensure"/> with its own type before its first use, which has each assembly the application ships register
/// that registry's built-ins, once, through the <see cref="ILibraryRegistrations"/> its
/// <see cref="LibraryRegistrationsAttribute"/> names. Only the registries an application uses pay for their built-ins:
/// training a network never builds the GGUF tables or the ONNX translators. An application that references
/// Idrak.Abstraction alone gets the defaults that live here and whatever it registers itself.
/// </summary>
internal static class LibraryDefaults
{
    private static readonly Lock Gate = new();

    // The registries whose built-ins are registered (or being registered: a registration may use another registry).
    private static readonly HashSet<Type> Done = [];

    // The registrations of the first-party assemblies the application ships, in Assemblies' order, found on the first
    // Ensure (each assembly's attribute read once, its implementation made once); then those of assemblies a host loaded
    // later (Include).
    private static List<(Assembly Assembly, ILibraryRegistrations Registrations)>? _registrations;

    // The first-party assemblies that register built-ins, in order (core first: the others build on it). Each names its
    // ILibraryRegistrations with [assembly: LibraryRegistrations(typeof(...))]; one that is missing (not shipped with the
    // application) is skipped. Planned packages are listed so they work the day they appear.
    private static readonly string[] Assemblies =
        ["Idrak", "Idrak.Gpu", "Idrak.Imaging", "Idrak.Data", "Idrak.Text", "Idrak.Vision", "Idrak.Diffusion", "Idrak.Audio", "Idrak.Reinforcement",
            "Idrak.Cluster", "Idrak.Network", "Idrak.Onnx.Runtime", "Idrak.Mcp"];

    /// <summary>
    /// Has the first-party assemblies register their built-ins of <paramref name="registry"/>, unless done already (on
    /// this or another thread, which this then waits for).
    /// </summary>
    // The attributes keep each assembly's implementation in a trimmed or AOT application that ships the assembly: the
    // type its LibraryRegistrations attribute names (a test checks that the two agree).
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.LibraryRegistrations", "Idrak")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Gpu.LibraryRegistrations", "Idrak.Gpu")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Imaging.LibraryRegistrations", "Idrak.Imaging")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Data.LibraryRegistrations", "Idrak.Data")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Text.LibraryRegistrations", "Idrak.Text")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Vision.LibraryRegistrations", "Idrak.Vision")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Diffusion.LibraryRegistrations", "Idrak.Diffusion")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Audio.LibraryRegistrations", "Idrak.Audio")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Reinforcement.LibraryRegistrations", "Idrak.Reinforcement")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Cluster.LibraryRegistrations", "Idrak.Cluster")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Networking.LibraryRegistrations", "Idrak.Network")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Onnx.Runtime.LibraryRegistrations", "Idrak.Onnx.Runtime")]
    [DynamicDependency(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor, "Idrak.Mcp.LibraryRegistrations", "Idrak.Mcp")]
    // An application ships only the packages it references (a phone app takes a few), so the trimmer cannot resolve the
    // others' assemblies (IL2035). That is the intent: an assembly not shipped has nothing to keep, and Find skips it.
    [UnconditionalSuppressMessage("Trimming", "IL2035", Justification = "Each package is optional: one the application does not ship has no registrations to keep.")]
    public static void Ensure(Type registry)
    {
        lock (Gate)   // re-entrant: a registration that touches another registry comes back here
        {
            if (!Done.Add(registry))
            {
                return;
            }

            var registrations = _registrations ??= Find();
            Overrides.AsLibraryDefaults(() =>   // the built-ins are the slots' library defaults, not overrides
            {
                for (int i = 0; i < registrations.Count; i++)
                {
                    registrations[i].Registrations.RegisterFor(registry);
                }
            });
        }
    }

    /// <summary>
    /// Adds the built-ins of <paramref name="assembly"/>, a library assembly a host loaded after the application started
    /// (the tool's <c>-P</c>): into every registry in use already, now, and into the others on their first use, after
    /// the application's own. False when the assembly declares no <see cref="LibraryRegistrationsAttribute"/>; true, and
    /// nothing registered again, when it is one of the application's already.
    /// </summary>
    /// <exception cref="InvalidOperationException">The attribute names a type that is not an <see cref="ILibraryRegistrations"/>.</exception>
    public static bool Include(Assembly assembly)
    {
        ArgumentNullException.ThrowIfNull(assembly);
        if (assembly.GetCustomAttribute<LibraryRegistrationsAttribute>() is not { } declared)
        {
            return false;
        }

        lock (Gate)
        {
            var registrations = _registrations ??= Find();
            foreach (var (known, _) in registrations)
            {
                if (known == assembly)
                {
                    return true;
                }
            }

            var added = Create(assembly, declared);
            registrations.Add((assembly, added));
            Type[] inUse = [.. Done];
            Overrides.AsLibraryDefaults(() =>
            {
                foreach (var registry in inUse)
                {
                    added.RegisterFor(registry);
                }
            });
            return true;
        }
    }

    private static List<(Assembly, ILibraryRegistrations)> Find()
    {
        var found = new List<(Assembly, ILibraryRegistrations)>(Assemblies.Length);
        foreach (string name in Assemblies)
        {
            Assembly assembly;
            try
            {
                assembly = Assembly.Load(new AssemblyName(name));
            }
            catch (Exception e) when (e is FileLoadException or FileNotFoundException or BadImageFormatException)
            {
                continue;
            }

            if (assembly.GetCustomAttribute<LibraryRegistrationsAttribute>() is { } declared)
            {
                found.Add((assembly, Create(assembly, declared)));
            }
        }

        return found;
    }

    private static ILibraryRegistrations Create(Assembly assembly, LibraryRegistrationsAttribute declared) =>
        Activator.CreateInstance(declared.Type) as ILibraryRegistrations
        ?? throw new InvalidOperationException($"{assembly.GetName().Name}: [assembly: LibraryRegistrations(typeof({declared.Type.FullName}))] names a type that is not an "
                                               + $"{nameof(ILibraryRegistrations)}.");
}
