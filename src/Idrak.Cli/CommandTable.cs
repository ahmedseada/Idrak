// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands;

namespace Idrak.Cli;

/// <summary>Every command, gathered from the groups under Commands/ (each group adds its own; this list names the groups).</summary>
internal static class CommandTable
{
    public static IReadOnlyList<Command> All { get; } =
    [
        .. HealthCommands.All,
        .. RunCommands.All,
        .. ServeCommands.All,
        .. ModelCommands.All,
        .. TrainCommands.All,
        .. DataCommands.All,
        .. RetrievalCommands.All,
        .. MeasureCommands.All,
        .. DesignCommands.All,
        .. DeveloperCommands.All,
    ];
}
