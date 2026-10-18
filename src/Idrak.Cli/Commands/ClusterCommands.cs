// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Cluster;

namespace Idrak.Cli.Commands;

/// <summary>The cluster commands (plan 23): init, inspect, sim, start, bench, replay, calibrate and ui, and the transports listing.</summary>
internal static class ClusterCommands
{
    public static IReadOnlyList<Command> All { get; } =
    [
        new ClusterInitCommand(),
        new ClusterInspectCommand(),
        new ClusterSimCommand(),
        new ClusterStartCommand(),
        new ClusterBenchCommand(),
        new ClusterReplayCommand(),
        new ClusterCalibrateCommand(),
        new TransportsCommand(),
        new ClusterUiCommand(),
    ];
}
