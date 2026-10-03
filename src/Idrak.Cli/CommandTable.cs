// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands;

namespace Idrak.Cli;

/// <summary>
/// Every command, gathered from the groups under Commands/ (each group adds its own; this list names the groups, with
/// the titles and in the order of plans/idrak-cli.md, which is how <c>idrak help</c> lists them).
/// </summary>
internal static class CommandTable
{
    /// <summary>The groups: a title and its commands, in the plan's order.</summary>
    public static IReadOnlyList<(string Title, IReadOnlyList<Command> Commands)> Groups { get; } =
    [
        ("Setup and health", HealthCommands.All),
        ("Run models", RunCommands.All),
        ("Serve", ServeCommands.All),
        ("Models", ModelCommands.All),
        ("Train", TrainCommands.All),
        ("Data", DataCommands.All),
        ("Retrieval", RetrievalCommands.All),
        ("Measure", MeasureCommands.All),
        ("Design", DesignCommands.All),
        ("Developers", DeveloperCommands.All),
    ];

    public static IReadOnlyList<Command> All { get; } = [.. Groups.SelectMany(g => g.Commands)];
}
