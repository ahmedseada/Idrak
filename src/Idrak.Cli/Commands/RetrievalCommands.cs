// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Retrieval;

namespace Idrak.Cli.Commands;

/// <summary>The rag index, rag ask, rag search and rag eval commands (plans/idrak-cli.md, "Retrieval").</summary>
internal static class RetrievalCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new RagIndexCommand(),
        new RagSearchCommand(),
        new RagAskCommand(),
        new RagEvalCommand(),
    ];
}
