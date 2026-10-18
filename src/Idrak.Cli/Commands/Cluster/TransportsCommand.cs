// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// <c>idrak transports</c> (plan 23, phase C8): every registered transport, who registered it, its capabilities, and
/// whether it can join two ranks on this machine in each setting a cluster runs in (two ranks of one process, two
/// processes here, two machines), with the reason it gives.
/// </summary>
internal sealed class TransportsCommand : Command
{
    // Names the plan reserves that a plug-in provides (no library default).
    private static readonly string[] PlugInOnly = ["rdma", "udp"];

    public override string Name => "transports";

    public override string Summary => "Every registered transport, its capabilities, and whether it is available here and why";

    public override string Usage => """
        [NAME]

        Lists the transports registered here (the library's, and any a plug-in adds with -P), each with who registered
        it, its capabilities (its own estimate of bandwidth and latency until a bench measures them, whether it copies,
        moves device memory directly, is reliable, crosses processes and machines) and whether it can join two ranks on
        this machine in each setting a cluster runs in: two ranks of one process (a simulation, or GPUs of one
        process), two processes here (idrak cluster start, cluster sim --mode localhost) and two machines (a LAN; a key
        in IDRAK_CLUSTER_KEY is needed off this machine), with the reason each gives. The devices of the one-process
        setting are the first two this machine lists (the same one twice when it lists one). Names a plug-in provides
        (rdma, udp) are listed when nothing registers them. -j gives the reasons per setting as JSON.

        Arguments:
          NAME  only this transport

        Examples:
          idrak transports
          idrak transports tcp -j
          idrak transports -P MyRdma.dll
        """;

    public override int Run(CommandContext context)
    {
        string? only = context.Positional.Count > 0 ? context.Positional[0] : null;
        var providers = Transports.All.Where(p => only is null || string.Equals(p.Name, only, StringComparison.OrdinalIgnoreCase)).ToList();
        if (only is not null && providers.Count == 0)
        {
            context.Error($"Transport '{only}' is not registered (registered: {string.Join(", ", Transports.Names)}); a plug-in adds one (-P PLUGIN.dll).");
            return ExitCodes.Failed;
        }

        ClusterOptions environment;
        try
        {
            environment = ClusterOptions.FromEnvironment();
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);
        }

        var settings = Settings(environment);
        var inv = CultureInfo.InvariantCulture;
        var json = new JsonArray();
        var rows = new List<IReadOnlyList<string>>();
        var reasons = new List<string>();
        foreach (var provider in providers)
        {
            var c = provider.Capabilities;
            string capabilities = string.Join(", ", new[]
            {
                c.ZeroCopy ? "zero-copy" : null, c.GpuDirect ? "device memory" : null, c.Reliable ? "reliable" : "unreliable",
                c.CrossProcess ? "processes" : null, c.CrossMachine ? "machines" : null,
            }.OfType<string>());
            var available = new JsonObject();
            var cells = new List<string>();
            foreach (var (setting, context0) in settings)
            {
                TransportAvailability availability;
                try
                {
                    availability = provider.Availability(context0, 1);
                }
                catch (Exception e) when (e is not OutOfMemoryException)
                {
                    availability = new TransportAvailability(provider.Name, false, $"could not tell: {e.Message}");
                }

                cells.Add(availability.Available ? "yes" : "no");
                available[setting] = new JsonObject { ["available"] = availability.Available, ["reason"] = availability.Reason };
                reasons.Add($"  {provider.Name}, {setting}: {(availability.Available ? "yes" : "no")} ({availability.Reason})");
            }

            string origin = Transports.Origin(provider.Name) ?? "?";
            rows.Add([provider.Name, origin, Estimate(c), capabilities, .. cells]);
            json.Add(new JsonObject
            {
                ["name"] = provider.Name,
                ["origin"] = origin,
                ["summary"] = provider.Summary,
                ["carrier"] = provider.Carrier,
                ["bandwidth_estimate"] = c.BandwidthBytesPerSecond,
                ["latency_estimate_seconds"] = c.Latency.TotalSeconds,
                ["zero_copy"] = c.ZeroCopy,
                ["gpu_direct"] = c.GpuDirect,
                ["reliable"] = c.Reliable,
                ["cross_process"] = c.CrossProcess,
                ["cross_machine"] = c.CrossMachine,
                ["available"] = available,
            });
        }

        context.Table(["transport", "from", "estimate", "capabilities", .. settings.Select(s => s.Name)], rows);
        if (!context.Json)
        {
            context.Write("");
            context.Write("why, per setting (two ranks of the first devices this machine lists):");
            foreach (string reason in reasons)
            {
                context.Write(reason);
            }
        }

        var missing = only is null ? PlugInOnly.Where(n => Transports.Find(n) is null).ToList() : [];
        if (missing.Count > 0)
        {
            context.Write($"{string.Join(", ", missing)}: not registered here (native libraries the library does not ship); a plug-in registers them (-P PLUGIN.dll).");
        }

        context.WriteJson(new JsonObject
        {
            ["transports"] = json,
            ["settings"] = new JsonArray([.. settings.Select(s => (JsonNode)s.Name)]),
            ["not_registered"] = new JsonArray([.. missing.Select(m => (JsonNode)m)]),
        });
        return ExitCodes.Ok;

        string Estimate(TransportCapabilities c) => c.BandwidthBytesPerSecond > 0
            ? string.Create(inv, $"{c.BandwidthBytesPerSecond / (1 << 20):F0} MiB/s, {c.Latency.TotalMicroseconds:F0} us")
            : "measured by a bench";
    }

    // The three settings, each a context of rank 0 of two ranks on this machine's first devices.
    private static List<(string Name, ClusterContext Context)> Settings(ClusterOptions environment)
    {
        var devices = MachineFacts.Gather().Devices.Select(d => d.Device).ToList();
        string first = devices.FirstOrDefault(d => !d.StartsWith("cpu", StringComparison.OrdinalIgnoreCase)) ?? "cpu";
        string second = devices.FirstOrDefault(d => d != first && !d.StartsWith("cpu", StringComparison.OrdinalIgnoreCase)) ?? first;
        string intra = first == second ? ClusterTopology.SameDevice : "pcie";
        var here = ClusterTopology.Create([new ClusterNodeSpec(Environment.MachineName, "127.0.0.1:29500", [new(first, null), new(second, null)])], first == second ? null : intra, null);
        var apart = ClusterTopology.Create(
            [new ClusterNodeSpec(Environment.MachineName, "192.0.2.1:29500", [new(first, null)]), new ClusterNodeSpec("other", "192.0.2.2:29500", [new(first, null)])],
            null, "ethernet", [new ClusterLinkOverride(Environment.MachineName, "other", "direct-link")]);
        var inProcess = environment with { Simulation = environment.Simulation with { Mode = SimulationMode.InProcess } };
        var processes = environment with { Coordinator = "127.0.0.1:29500", Simulation = environment.Simulation with { Mode = SimulationMode.Localhost } };
        var machines = environment with { Coordinator = "192.0.2.1:29500", Simulation = environment.Simulation with { Mode = SimulationMode.Localhost } };
        return
        [
            ("one process", new ClusterContext(here, 0, inProcess)),
            ("processes here", new ClusterContext(here, 0, processes)),
            ("two machines", new ClusterContext(apart, 0, machines)),
        ];
    }
}
