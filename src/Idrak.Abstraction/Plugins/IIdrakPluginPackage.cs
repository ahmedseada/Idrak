// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Diagnostics.CodeAnalysis;

namespace Idrak.Abstraction.Plugins;

/// <summary>
/// A plug-in package: an assembly's plug-ins (<see cref="IIdrakPlugin"/>), in the order they register. The assembly
/// declares it with <see cref="IdrakPluginPackageAttribute"/>, which is how a host that loads the assembly itself finds
/// it (the idrak tool's <c>-P</c>, <see cref="IdrakPlugins.Load"/>); code that references the package reaches the same
/// plug-ins through its root class's static <c>Package</c> (<c>TextPlugins.Package</c>) and registers them all with
/// <see cref="IdrakPlugins.RegisterAll(IIdrakPluginPackage)"/>.
/// </summary>
public interface IIdrakPluginPackage
{
    /// <summary>The package's name, as the library's "not registered" messages and the registries' origins name it: its assembly's ("Idrak.Text.Plugins").</summary>
    string Name { get; }

    /// <summary>Every plug-in of the package, in the order they register (a plug-in another builds on first).</summary>
    IReadOnlyList<IIdrakPlugin> All { get; }
}

/// <summary>
/// Names an assembly's plug-in package: <c>[assembly: IdrakPluginPackage(typeof(MyPackage))]</c>, where
/// <c>MyPackage : IIdrakPluginPackage</c> has a public parameterless constructor. What the idrak tool's <c>-P</c> reads
/// (once per assembly) to register the assembly's plug-ins.
/// </summary>
/// <param name="package">The implementation of <see cref="IIdrakPluginPackage"/>.</param>
[AttributeUsage(AttributeTargets.Assembly, AllowMultiple = false)]
public sealed class IdrakPluginPackageAttribute([DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)] Type package) : Attribute
{
    /// <summary>The implementation of <see cref="IIdrakPluginPackage"/>.</summary>
    [DynamicallyAccessedMembers(DynamicallyAccessedMemberTypes.PublicParameterlessConstructor)]
    public Type Package { get; } = package;
}
