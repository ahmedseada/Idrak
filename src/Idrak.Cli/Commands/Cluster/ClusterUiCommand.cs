// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using Idrak.Cli.Shared;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// <c>idrak cluster ui [--topology FILE] [--port N] [--host H] [--no-browser]</c> (plan 23, phase C9): the cluster dashboard
/// in a browser, as <c>idrak ui</c> opens the chat page. It sets up a topology (the builder's questions, or a form), shows
/// its drawings and predictions, starts simulated runs and steers them, and follows a real run's folder read-only.
/// </summary>
internal sealed class ClusterUiCommand : Command
{
    public override string Name => "cluster ui";

    public override string Summary => "Opens the cluster dashboard: set up a topology, start and steer simulated runs, follow real ones";

    public override string Usage => """
        [--topology FILE] [--port N] [--host H] [--no-browser] [--follow RUN] [--unit UNIT]

        Serves the cluster dashboard and opens it in the browser. Set up: the topology builder's questions (the ones idrak
        cluster init asks, each with what it means and this machine's suggestion) or a form of nodes, devices and caps;
        the topology drawn as a network, a ring, a link matrix, the all-reduce's phases and each card's memory, with the
        all-reduce time each algorithm would take, played on the virtual clock. Run: a training command started as a
        simulated cluster (in this process, or as processes of this machine), paused at a step boundary, resumed,
        stopped with a checkpoint, and faults injected at the next step (in-process; each written to the event log so
        --seed and --step replay the run as steered); its ranks, losses, links, memory and events shown as they come.
        A reinforce run is paused and stopped at an iteration boundary instead (every rank at the same one; the stop
        writes its checkpoint and the adapter so far) and shows its iterations, rewards and pass rate from its JSON log.
        A real cluster start run is followed from its run folder (rank 0's log.jsonl and the ranks' lines), its
        controls refused. It listens on 127.0.0.1; with --host another address, requests from other machines need an
        API key (idrak server keys add, or IDRAK_API_KEY; sent as Authorization: Bearer KEY or X-API-Key). The
        library's telemetry is taken only while a page listens. Stop with Ctrl+C.

        Options:
              --topology FILE  the topology file the page opens and saves (idrak cluster init writes one)
              --port N         the port (IDRAK_CLUSTER_UI_PORT; 0, the default: one the system picks, printed)
              --host H         the address to listen on (IDRAK_CLUSTER_UI_HOST; 127.0.0.1)
              --no-browser     print the page's address instead of opening it
              --follow RUN     follow a run folder from the start (a cluster start run's, read-only)
              --unit UNIT      the unit sizes are shown in: B, KB, MB, GB or TB (powers of 1,024); default: per
                               value, the largest unit it fills

        Examples:
          idrak cluster ui --topology topology.json
          idrak cluster ui --port 8090 --no-browser
          idrak cluster ui --follow run --topology lan.json
        """;

    public override IReadOnlyCollection<string> ValueOptions => ["--topology", "--port", "--host", "--follow", "--unit"];

    public override IReadOnlyCollection<string> Flags => ["--no-browser"];

    public override int Run(CommandContext context)
    {
        using var interrupt = new Interrupt(context);
        using var dashboard = Dashboard.Open(context, context.Option("--topology"));
        context.Write("Stop with Ctrl+C.");
        context.Output.Flush();
        interrupt.Token.WaitHandle.WaitOne();
        context.Write("dashboard stopped");
        return ExitCodes.Ok;
    }
}
