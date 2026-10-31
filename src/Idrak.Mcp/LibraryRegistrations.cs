// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

[assembly: LibraryRegistrations(typeof(Idrak.Mcp.LibraryRegistrations))]

namespace Idrak.Mcp;

/// <summary>
/// What this assembly adds to the registries of Idrak.Abstraction: the "mcp" profiler probe (what MCP served and called
/// during a run). <c>LibraryDefaults.Ensure</c> calls <see cref="RegisterFor"/> once per registry, before that registry is
/// first used, when the application ships Idrak.Mcp.
/// </summary>
internal sealed class LibraryRegistrations : ILibraryRegistrations
{
    public void RegisterFor(Type registry)
    {
        if (registry == typeof(ProfileProbes))
        {
            ProfileProbes.Register(new Diagnostics.McpProbe());
        }
    }
}
