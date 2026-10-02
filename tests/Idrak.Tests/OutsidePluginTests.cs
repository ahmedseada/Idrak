// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak;

// Plug-ins from an assembly without internal access (tests/Idrak.PluginTests): run here on every device like the
// other groups.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] OutsidePluginGroup = [.. Idrak.PluginTests.PluginTests.All];
}
