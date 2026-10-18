// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// What the dashboard's set-up endpoints answer (plan 23, phase C9), as JSON: this machine's facts, the topology builder's
/// questions (the same <see cref="TopologyBuilder"/> <c>idrak cluster init</c> asks), a topology from the builder's answers,
/// and a topology's inspection (<see cref="TopologyInspection"/>: links, every pair, the predicted all-reduce per algorithm
/// and per phase). Pure functions of their inputs, so the tests call them as the endpoints do.
/// </summary>
internal static class DashboardTopology
{
    /// <summary>The all-reduce whose time is predicted when the page asks for no size (cluster inspect's default).</summary>
    public const string DefaultPredictBytes = "100MB";

    /// <summary>This machine's facts, the registered names the page offers, and the page's settings.</summary>
    public static JsonObject Machine(MachineFacts facts, DashboardSettings settings) => new()
    {
        ["host"] = facts.HostName,
        ["address"] = facts.Address,
        ["adapter"] = facts.AdapterName,
        ["adapter_bits_per_second"] = facts.AdapterBitsPerSecond,
        ["devices"] = new JsonArray([.. facts.Devices.Select(d => (JsonNode)new JsonObject
        {
            ["device"] = d.Device, ["type"] = d.Type.ToString(), ["memory"] = d.MemoryBytes, ["name"] = d.Name, ["card"] = d.CardKey,
        })]),
        ["links"] = new JsonArray([.. LinkProfiles.Names.Order(StringComparer.Ordinal).Select(n => LinkProfiles.Find(n)).Where(p => p is not null).Select(p => (JsonNode)new JsonObject
        {
            ["name"] = p!.Name, ["bandwidth"] = p.BandwidthBytesPerSecond, ["latency_us"] = p.Latency.TotalMicroseconds,
        })]),
        ["failure_models"] = new JsonArray([.. FailureModels.Names.Order(StringComparer.Ordinal).Select(n => (JsonNode)n)]),
        ["collectives"] = new JsonArray([.. Collectives.Names.Order(StringComparer.Ordinal).Select(n => (JsonNode)n)]),
        ["fault_kinds"] = new JsonArray([.. Enum.GetNames<FaultKind>().Select(n => (JsonNode)n)]),
        ["topology"] = settings.Topology is { } path ? Path.GetFullPath(path) : null,
        ["lang"] = settings.Language,
        ["unit"] = settings.Unit,
        ["predict_bytes"] = ClusterTopology.ParseMemory(DefaultPredictBytes),
        ["follow"] = settings.Follow is { } follow ? Path.GetFullPath(follow) : null,
    };

    /// <summary>
    /// One step of the builder (POST /api/topology/questions): the answers so far (by key) are taken in order, then
    /// <c>answer</c> (when the body has it; empty: the question's default) answers the next question. Out: the next question
    /// (null when every one is answered), the answers taken, why the answer was refused (or null), and when done the file.
    /// </summary>
    /// <exception cref="ArgumentException">An answer given in advance the builder refuses, naming its key.</exception>
    public static JsonObject Ask(MachineFacts facts, JsonObject body)
    {
        var builder = new TopologyBuilder(facts, Answers(body["answers"]));
        string? refused = null;
        if (body.ContainsKey("answer"))
        {
            if (builder.Next is null)
            {
                throw new ArgumentException("Every question is answered: there is nothing to answer.");
            }

            refused = builder.Answer(body["answer"]?.GetValue<string>());
        }

        var next = builder.Next;
        return new JsonObject
        {
            ["question"] = next is null ? null : new JsonObject
            {
                ["key"] = next.Key, ["prompt"] = next.Prompt, ["description"] = next.Description, ["default"] = next.Default, ["suggestion"] = next.Suggestion,
            },
            ["answers"] = new JsonObject([.. builder.Answers.Select(a => KeyValuePair.Create(a.Key, (JsonNode?)a.Value))]),
            ["refused"] = refused,
            ["done"] = next is null,
            ["json"] = next is null ? builder.ToJson() : null,
        };
    }

    /// <summary>
    /// The topology the answers describe, every question left open given its default (as <c>cluster init --yes</c>), and
    /// the commented file <c>cluster init</c> writes for it.
    /// </summary>
    /// <exception cref="ArgumentException">An answer the builder refuses, or an open question with no default; the message names its key.</exception>
    public static (ClusterTopology Topology, string Json) Build(MachineFacts facts, JsonNode? answers)
    {
        var builder = new TopologyBuilder(facts, Answers(answers));
        try
        {
            builder.AnswerRest();
        }
        catch (InvalidOperationException e)
        {
            throw new ArgumentException(e.Message, e);
        }

        return (builder.Build(), builder.ToJson());
    }

    /// <summary>A topology as the builder's answers, so the page's form shows a file it did not write.</summary>
    public static JsonObject AnswersOf(ClusterTopology topology)
    {
        var inv = CultureInfo.InvariantCulture;
        var answers = new JsonObject { ["nodes"] = topology.Nodes.Count.ToString(inv) };
        for (int i = 0; i < topology.Nodes.Count; i++)
        {
            var node = topology.Nodes[i];
            answers[$"node.{i + 1}.name"] = node.Name;
            if (i == 0)
            {
                answers["node.1.address"] = node.Address ?? "";
            }

            answers[$"node.{i + 1}.devices"] = string.Join(",", node.Devices.Select(d => d.Device));
            for (int j = 0; j < node.Devices.Count; j++)
            {
                answers[$"node.{i + 1}.device.{j + 1}.memory"] = node.Devices[j].MemoryBytes is { } bytes ? ClusterTopology.FormatMemory(bytes) : "";
            }
        }

        if (topology.Nodes.Any(n => n.Devices.Select(d => d.Device).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1))
        {
            answers["intra-node"] = topology.IntraNode ?? "pcie";
        }

        if (topology.Nodes.Count > 1)
        {
            answers["inter-node"] = topology.InterNode ?? "ethernet-1g";
            answers["overrides"] = string.Join(";", topology.Overrides.Select(o => $"{o.A},{o.B}={o.Link}"));
        }

        answers["seed"] = topology.Simulation.Seed.ToString(inv);
        answers["failures"] = topology.Simulation.Failures;
        return answers;
    }

    /// <summary>
    /// The inspection the page draws (GET or POST /api/topology/inspect): the nodes with their ranks and caps (and each
    /// device's memory where it is on this machine), every pair's link, each kind of link with its figures, the predicted
    /// all-reduce of <paramref name="bytes"/> per algorithm with its phases, the default algorithm and the transport per pair.
    /// </summary>
    public static JsonObject Inspect(ClusterTopology topology, MachineFacts facts, long bytes, string? path)
    {
        var options = ClusterOptions.FromEnvironment() with { Simulation = new SimulationOptions { Mode = SimulationMode.InProcess } };
        int? here = topology.NodeOf(facts.HostName);
        long? Physical(string device) => facts.Devices.FirstOrDefault(d => d.Device.Equals(device, StringComparison.OrdinalIgnoreCase))?.MemoryBytes;
        var nodes = new JsonArray();
        for (int n = 0; n < topology.Nodes.Count; n++)
        {
            var node = topology.Nodes[n];
            nodes.Add(new JsonObject
            {
                ["name"] = node.Name,
                ["address"] = node.Address,
                ["here"] = here == n,
                ["coordinator"] = n == 0,
                ["ranks"] = new JsonArray([.. topology.Ranks(n).Select(r => (JsonNode)r)]),
                ["devices"] = new JsonArray([.. node.Devices.Select(d => (JsonNode)new JsonObject
                {
                    ["device"] = d.Device, ["cap"] = d.MemoryBytes, ["physical"] = Physical(d.Device),
                    ["card"] = facts.Devices.FirstOrDefault(f => f.Device.Equals(d.Device, StringComparison.OrdinalIgnoreCase))?.CardKey ?? d.Device,
                })]),
            });
        }

        var ranks = new JsonArray([.. Enumerable.Range(0, topology.WorldSize).Select(r =>
        {
            var (node, local) = topology.Place(r);
            var device = topology.Nodes[node].Devices[local];
            return (JsonNode)new JsonObject
            {
                ["rank"] = r, ["node"] = topology.Nodes[node].Name, ["node_index"] = node, ["local"] = local, ["device"] = device.Device,
                ["cap"] = device.MemoryBytes, ["physical"] = Physical(device.Device),
            };
        })]);

        var result = new JsonObject
        {
            ["path"] = path is null ? null : Path.GetFullPath(path),
            ["world"] = topology.WorldSize,
            ["node_here"] = here is { } h ? topology.Nodes[h].Name : null,
            ["nodes"] = nodes,
            ["ranks"] = ranks,
            ["intra_node"] = topology.IntraNode,
            ["inter_node"] = topology.InterNode,
            ["overrides"] = new JsonArray([.. topology.Overrides.Select(o => (JsonNode)new JsonObject { ["a"] = o.A, ["b"] = o.B, ["link"] = o.Link })]),
            ["seed"] = topology.Simulation.Seed,
            ["failures"] = topology.Simulation.Failures,
            ["bytes"] = bytes,
        };

        try
        {
            result["pairs"] = new JsonArray([.. TopologyInspection.Pairs(topology).Select(p => (JsonNode)new JsonObject
            {
                ["a"] = p.A, ["b"] = p.B, ["link"] = p.Link, ["bandwidth"] = p.BandwidthBytesPerSecond, ["latency_us"] = p.Latency.TotalMicroseconds,
                ["kind"] = Kebab(p.Kind.ToString()),
            })]);
            result["links"] = new JsonArray([.. TopologyInspection.Links(topology).Select(l => (JsonNode)new JsonObject
            {
                ["between"] = l.Between, ["a"] = l.FirstRank, ["b"] = l.SecondRank, ["link"] = l.Link, ["bandwidth"] = l.BandwidthBytesPerSecond,
                ["latency_us"] = l.Latency.TotalMicroseconds, ["source"] = l.Source,
            })]);
        }
        catch (Exception e) when (e is NotSupportedException or InvalidOperationException or ResourceLimitExceededException)
        {
            result["links_error"] = $"A link could not be resolved here: {e.Message}";
        }

        string collective = options.CollectiveFor(topology);
        result["default_collective"] = collective;
        var predictions = new JsonArray();
        foreach (string algorithm in TopologyInspection.Playable)
        {
            var entry = new JsonObject { ["algorithm"] = algorithm, ["default"] = algorithm == collective };
            try
            {
                entry["seconds"] = TopologyInspection.PredictAllReduce(topology, algorithm, bytes).TotalSeconds;
                entry["phases"] = new JsonArray([.. TopologyInspection.PredictPhases(topology, algorithm, bytes).Select(p => (JsonNode)new JsonObject
                {
                    ["name"] = p.Name, ["where"] = p.Where, ["links"] = new JsonArray([.. p.Links.Select(l => (JsonNode)l)]), ["seconds"] = p.Time.TotalSeconds,
                })]);
            }
            catch (Exception e) when (e is InvalidOperationException or NotSupportedException)
            {
                entry["seconds"] = null;
                entry["error"] = e.Message;
            }

            predictions.Add(entry);
        }

        result["predictions"] = predictions;
        try
        {
            var plan = TransportPlan.Make(topology, options);
            var transports = new JsonArray();
            for (int a = 0; a < topology.WorldSize; a++)
            {
                for (int b = a + 1; b < topology.WorldSize; b++)
                {
                    var (transport, reason) = plan[a, b];
                    transports.Add(new JsonObject { ["a"] = a, ["b"] = b, ["transport"] = transport, ["reason"] = reason });
                }
            }

            result["transports"] = transports;
        }
        catch (NotSupportedException e)
        {
            result["transports_error"] = e.Message;
        }

        return result;
    }

    // The answers object of a request body, by key (values as text).
    private static Dictionary<string, string> Answers(JsonNode? node)
    {
        var answers = new Dictionary<string, string>(StringComparer.Ordinal);
        if (node is null)
        {
            return answers;
        }

        if (node is not JsonObject given)
        {
            throw new ArgumentException("\"answers\" is an object of the builder's keys and their answers.");
        }

        foreach (var (key, value) in given)
        {
            if (value is not null)
            {
                answers[key] = value.GetValueKind() == System.Text.Json.JsonValueKind.String ? value.GetValue<string>() : value.ToJsonString();
            }
        }

        return answers;
    }

    private static string Kebab(string name) => string.Concat(name.Select((c, i) => char.IsUpper(c) ? (i > 0 ? "-" : "") + char.ToLowerInvariant(c) : c.ToString()));
}
