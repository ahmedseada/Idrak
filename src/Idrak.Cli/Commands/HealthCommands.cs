// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Commands;

/// <summary>The doctor, devices, version, report, cache, config, plugins commands (plans/idrak-cli.md).</summary>
internal static class HealthCommands
{
    public static IReadOnlyList<Command> All { get; } = [new VersionCommand()];
}
