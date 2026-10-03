// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Train;

namespace Idrak.Cli.Commands;

/// <summary>The tune, train, resume, runs, predict, package and distill commands (plans/idrak-cli.md).</summary>
internal static class TrainCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new TuneCommand(),
        new TuneInitCommand(),
        new TrainCommand(),
        new ResumeCommand(),
        new RunsListCommand(),
        new RunsShowCommand(),
        new RunsCompareCommand(),
        new PredictCommand(),
        new PackageCommand(),
        new DistillCommand(),
    ];
}
