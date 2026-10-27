// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Cluster;
using Idrak.Cli.Commands.Serve;

namespace Idrak.Cli;

/// <summary>
/// The tool's servers (<see cref="ICommandServers"/>), on ASP.NET Core: serve's and ui's (<see cref="ServeHost"/>) and the
/// cluster dashboard (<see cref="DashboardServer"/>). Registered by the Program before the first command, and by the tests.
/// </summary>
internal sealed class AspNetCoreServers : ICommandServers
{
    /// <inheritdoc />
    public int Serve(CommandContext context, ServeSettings settings, IReadOnlyList<ServedModel> models, Action<string>? started) =>
        new ServeHost(context, settings, models).Run(started);

    /// <inheritdoc />
    public int ServeMcp(CommandContext context, ServeSettings settings, IReadOnlyList<ServedModel> models, IReadOnlyList<ITool> tools) =>
        new ServeHost(context, settings, models).RunMcp(tools);

    /// <inheritdoc />
    public IClusterDashboard OpenDashboard(CommandContext context, string? topology) => Dashboard.Open(context, topology);
}
