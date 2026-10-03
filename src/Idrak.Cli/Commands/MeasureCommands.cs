// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Measure;

namespace Idrak.Cli.Commands;

/// <summary>The bench, eval, perplexity, profile, check and tuning show commands (plans/idrak-cli.md, "Measure").</summary>
internal static class MeasureCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new BenchCommand(),
        new EvalCommand(),
        new PerplexityCommand(),
        new ProfileCommand(),
        new CheckCommand(),
        new TuningShowCommand(),
    ];
}
