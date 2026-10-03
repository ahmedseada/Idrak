// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Run;

namespace Idrak.Cli.Commands;

/// <summary>
/// The commands that run models (plans/idrak-cli.md, "Run models"): chat, run, batch, compare, complete, embed,
/// tokenize, template, agent, tools list and tools test. Each lives in its own file under Commands/Run/.
/// </summary>
internal static class RunCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new ChatCommand(),
        new RunCommand(),
        new BatchCommand(),
        new CompareCommand(),
        new CompleteCommand(),
        new EmbedCommand(),
        new TokenizeCommand(),
        new TemplateCommand(),
        new AgentCommand(),
        new ToolsListCommand(),
        new ToolsTestCommand(),
    ];
}
