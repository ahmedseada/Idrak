// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Design;

namespace Idrak.Cli.Commands;

/// <summary>The suggest, explain and viz commands (plans/idrak-cli.md, "Design a model").</summary>
internal static class DesignCommands
{
    public static IReadOnlyList<Command> All { get; } = [new SuggestCommand(), new ExplainCommand(), new VizCommand()];
}
