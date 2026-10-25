// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Abstraction.Plugins;

[assembly: IdrakPluginPackage(typeof(Idrak.PluginTests.PluginTestsPackage))]

namespace Idrak.PluginTests;

/// <summary>
/// This assembly as a plug-in package, the way an outside package declares one: what <c>idrak -P Idrak.PluginTests.dll</c>
/// registers, the image models' plug-in (<see cref="ImageFamilyPlugin"/>) and the sound format's (<see cref="AudioPlugin"/>).
/// </summary>
public sealed class PluginTestsPackage : IIdrakPluginPackage
{
    /// <inheritdoc/>
    public string Name => "Idrak.PluginTests";

    /// <inheritdoc/>
    public IReadOnlyList<IIdrakPlugin> All { get; } = [ImageFamilyPlugin.Plugin, AudioPlugin.Plugin];
}
