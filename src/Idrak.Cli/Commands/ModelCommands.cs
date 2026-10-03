// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

namespace Idrak.Cli.Commands;

/// <summary>
/// The model commands (plans/idrak-cli.md, "Models"): pull, list, rm, show, search, alias, memory, quantize, merge,
/// inspect, verify, convert, diff, families. Each is a class under Commands/Models/; the cache layout is in
/// Shared/ModelCache.cs.
/// </summary>
internal static class ModelCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new PullCommand(),
        new ListCommand(),
        new RmCommand(),
        new ShowCommand(),
        new SearchCommand(),
        new AliasSetCommand(),
        new AliasListCommand(),
        new AliasRmCommand(),
        new MemoryCommand(),
        new QuantizeCommand(),
        new MergeCommand(),
        new InspectCommand(),
        new VerifyCommand(),
        new ConvertCommand(),
        new DiffCommand(),
        new FamiliesCommand(),
    ];
}
