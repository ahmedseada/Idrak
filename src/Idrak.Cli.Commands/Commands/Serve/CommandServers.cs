// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Commands.Cluster;
using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Serve;

/// <summary>
/// The servers the commands start: serve's and ui's HTTP server (serve --mcp's protocol on standard input and output with
/// it) and the cluster dashboard. They need ASP.NET Core, which is the idrak tool's (src/Idrak.Cli registers them at
/// startup), not this library's: a host that registers none (an app) refuses those commands by what they need
/// (<see cref="CommandNeeds"/>) before they ask for a server, and <see cref="CommandServers.Get"/> says so if one does.
/// </summary>
internal interface ICommandServers
{
    /// <summary>Starts the HTTP server over <paramref name="models"/>, waits for Ctrl+C, the host's stop or <c>idrak server stop</c>; the exit code.</summary>
    int Serve(CommandContext context, ServeSettings settings, IReadOnlyList<ServedModel> models, Action<string>? started);

    /// <summary>Serves <paramref name="models"/> (and <paramref name="tools"/>) over MCP on standard input and output until the client leaves; the exit code.</summary>
    int ServeMcp(CommandContext context, ServeSettings settings, IReadOnlyList<ServedModel> models, IReadOnlyList<ITool> tools);

    /// <summary>Starts the cluster dashboard with the command's settings, prints its address and opens it unless <c>--no-browser</c>.</summary>
    IClusterDashboard OpenDashboard(CommandContext context, string? topology);
}

/// <summary>A running cluster dashboard: the runs it shows and steers; disposing it stops it.</summary>
internal interface IClusterDashboard : IDisposable
{
    /// <summary>The runs it shows (a command attaches its own).</summary>
    DashboardRuns Runs { get; }
}

/// <summary>The servers of the process (<see cref="ICommandServers"/>): the idrak tool's, registered at its start; none in another host.</summary>
internal static class CommandServers
{
    private static volatile ICommandServers? _servers;

    /// <summary>Gives the process's servers (the tool's Program, before the first command).</summary>
    public static void Register(ICommandServers servers) => _servers = servers ?? throw new ArgumentNullException(nameof(servers));

    /// <summary>The process's servers.</summary>
    /// <exception cref="CommandUnavailableException">The host registered none: the command is refused, named.</exception>
    public static ICommandServers Get(CommandContext context)
    {
        ArgumentNullException.ThrowIfNull(context);
        return _servers ?? throw new CommandUnavailableException(context.Host.Refusal(context.Command.Name, CommandNeeds.NetworkListener, Messages.T("no server is registered here"))
                                                                 ?? Messages.T("idrak {0} is not available in {1}: {2}", context.Command.Name, context.Host.Name, Messages.T("no server is registered here")));
    }
}
