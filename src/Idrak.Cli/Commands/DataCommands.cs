// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Data;

namespace Idrak.Cli.Commands;

/// <summary>The data command (show, count, download, build, cache: <see cref="Data.DataTool"/>) and its subcommands (plans/idrak-cli.md).</summary>
internal static class DataCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new DataCommand(),
        new DataPreviewCommand(),
        new DataValidateCommand(),
        new DataStatsCommand(),
        new DataConvertCommand(),
        new DataDedupeCommand(),
        new DataSplitCommand(),
        new DataSampleCommand(),
        new DataMixCommand(),
    ];
}
