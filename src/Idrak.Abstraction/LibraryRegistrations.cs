// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Abstraction;

/// <summary>
/// What a library assembly adds to the registries of Idrak.Abstraction, which cannot name it (the GPU devices, the
/// sample sources of a domain package, ...). The assembly names its implementation with
/// <see cref="LibraryRegistrationsAttribute"/>; Idrak.Abstraction reads that attribute once per assembly, creates the
/// implementation once, and calls <see cref="RegisterFor"/> once per registry, before that registry is first used, so an
/// application pays only for the registries it uses. What it registers there are the slots' library defaults, not an
/// app's overrides.
/// </summary>
public interface ILibraryRegistrations
{
    /// <summary>Registers this assembly's built-ins of <paramref name="registry"/> (the registry's type); nothing for a registry it adds nothing to.</summary>
    void RegisterFor(Type registry);
}

/// <summary>
/// Names an assembly's <see cref="ILibraryRegistrations"/>: <c>[assembly: LibraryRegistrations(typeof(LibraryRegistrations))]</c>.
/// The type needs a public parameterless constructor.
/// </summary>
/// <param name="type">The implementation of <see cref="ILibraryRegistrations"/>.</param>
[AttributeUsage(AttributeTargets.Assembly, AllowMultiple = false)]
public sealed class LibraryRegistrationsAttribute([DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)] Type type) : Attribute
{
    /// <summary>The implementation of <see cref="ILibraryRegistrations"/>.</summary>
    [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)]
    public Type Type { get; } = type;
}
