// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Cli.Shared;
using Idrak.Cluster;

namespace Idrak.Cli.Commands.Cluster;

/// <summary>
/// <c>idrak cluster init</c> (plan 23): builds a topology file one question at a time, each with what it means and a
/// suggestion from the device or adapter it concerns on this machine; with <c>--yes</c> the options and the suggestions
/// answer instead.
/// </summary>
internal sealed class ClusterInitCommand : Command
{
    public override string Name => "cluster init";

    public override string Summary => "Builds a cluster topology file, asking for each input with a suggestion from this machine";

    public override string Usage => """
        [-o topology.json] [--yes] [--dry-run] [--nodes N] [--names A,B] [--address HOST:PORT]
        [--devices LIST;LIST] [--memory SIZES;SIZES] [--intra LINK] [--inter LINK]
        [--override A,B=LINK;...] [--override-address HOST,HOST;...] [--seed N] [--failures MODEL]

        Asks how many machines (nodes) the cluster has, their names (host names: a process finds its node by its
        host's name), the first one's address (where rank 0 listens), each machine's devices (one rank each; a device
        named twice is two ranks on one card) and each device's memory cap, the links inside and between machines, and
        the seed and failure model of simulated runs; a pair with a link of its own (a cable) is asked each end's address
        on it when the cable has a subnet of its own. Each question says what the input means, with a suggestion from
        this machine: its devices and their memory, its host name, its address, the speed its network adapter reports
        and its adapters off the LAN. An option answers its question in advance.

        Options:
          -o, --out FILE        the file to write (topology.json)
          -y, --yes             no questions: the options, then the suggestions
              --dry-run         print the file instead of writing it
              --nodes N         how many machines
              --names A,B       their names, in order
              --address H:P     the first machine's address
              --devices L;L     each machine's devices, machines separated by ';' (cuda:0,cuda:0;cuda:0)
              --memory S;S      each device's cap, as --devices (8GB,8GB;6GB; empty for the device's own)
              --intra LINK      the link between different devices of one machine (pcie, nvlink-900g)
              --inter LINK      the link between machines (ethernet-1g, direct-link-10g)
              --override O      pairs with a link of their own (desk,mini=direct-link-10g;...)
              --override-address A  each override's two ends' addresses on its own link, overrides separated by ';'
                                (10.0.0.1,10.0.0.2; a side left empty takes its node's address)
              --seed N          the seed of simulated runs
              --failures M      the failure model of simulated runs

        Sizes take B, KB, MB, GB, TB, all powers of 1,024 (GiB the same). A link name may end in its rate in bits a
        second (ethernet-2.5g, wifi-300m).

        Examples:
          idrak cluster init
          idrak cluster init -y --nodes 3 --names desk,mini,old --devices "cuda:0,cuda:0;cuda:0;cuda:0" --memory "8GB,8GB;8GB;6GB"
          idrak cluster init -y --nodes 2 --names desk,mini --override desk,mini=direct-link-10g --override-address 10.0.0.1,10.0.0.2
          idrak cluster inspect topology.json
        """;

    public override IReadOnlyCollection<string> ValueOptions =>
        ["--out", "--nodes", "--names", "--address", "--devices", "--memory", "--intra", "--inter", "--override", "--override-address", "--failures"];

    public override IReadOnlyCollection<string> Flags => Terminal.ConfirmFlags;

    public override IReadOnlyDictionary<string, string> ShortForms { get; } = new Dictionary<string, string> { ["-o"] = "--out", ["-y"] = "--yes" };

    public override int Run(CommandContext context)
    {
        string output = context.Option("--out") ?? "topology.json";
        try
        {
            return Build(context, output);
        }
        catch (ArgumentException e)
        {
            throw new UsageException(e.Message);                    // an option's answer the builder refused, naming it
        }
    }

    private static int Build(CommandContext context, string output)
    {
        var builder = new TopologyBuilder(MachineFacts.Gather(), Given(context));
        if (context.Flag("--yes"))
        {
            try
            {
                builder.AnswerRest();
            }
            catch (InvalidOperationException e)
            {
                throw new UsageException(e.Message);
            }
        }
        else
        {
            while (builder.Next is { } question)
            {
                // In colour: what the input means dim, the suggestion green, a refused answer red; Ask colours the rest.
                var console = context.ConsoleOutput;
                console.WriteLine();
                foreach (string line in question.Description.Split('\n'))
                {
                    console.WriteLine("  " + Terminal.Paint(context, line, Colour.Dim, console));
                }

                if (question.Suggestion is { } suggestion)
                {
                    console.WriteLine("  " + Terminal.Paint(context, "suggestion: " + suggestion, Colour.Green, console));
                }

                string option = Option(question.Key);
                string answer = Terminal.Ask(context, question.Prompt, question.Default ?? "", requireTerminal: $"--yes with {option}");
                if (builder.Answer(answer) is { } problem)
                {
                    console.WriteLine("  " + Terminal.Paint(context, problem, Colour.Red, console));
                }
            }
        }

        var topology = builder.Build();
        string json = builder.ToJson();
        var result = new JsonObject { ["path"] = Path.GetFullPath(output), ["nodes"] = topology.Nodes.Count, ["ranks"] = topology.WorldSize, ["written"] = false };
        if (Terminal.DryRun(context))
        {
            context.Write(json.TrimEnd());
            context.WriteJson(result);
            return ExitCodes.Ok;
        }

        if (File.Exists(output) && !context.Flag("--yes") && !Terminal.Confirm(context, $"{output} exists; replace it?"))
        {
            context.Error($"{output} exists; give another -o, or --yes to replace it.");
            return ExitCodes.Failed;
        }

        File.WriteAllText(output, json);
        result["written"] = true;
        context.Write($"wrote {output}: {topology.Nodes.Count} node(s), {topology.WorldSize} rank(s); next: idrak cluster inspect {output}");
        context.WriteJson(result);
        return ExitCodes.Ok;
    }

    // The options as the builder's keys.
    private static Dictionary<string, string> Given(CommandContext context)
    {
        var given = new Dictionary<string, string>(StringComparer.Ordinal);
        void Put(string key, string? value)
        {
            if (value is not null)
            {
                given[key] = value;
            }
        }

        Put("nodes", context.Option("--nodes"));
        Put("node.1.address", context.Option("--address"));
        Put("intra-node", context.Option("--intra"));
        Put("inter-node", context.Option("--inter"));
        Put("overrides", context.Option("--override"));
        Put("seed", context.Option("--seed"));
        Put("failures", context.Option("--failures"));
        if (context.Option("--names") is { } names)
        {
            var list = names.Split(',', StringSplitOptions.TrimEntries);
            for (int i = 0; i < list.Length; i++)
            {
                given[$"node.{i + 1}.name"] = list[i];
            }
        }

        if (context.Option("--devices") is { } devices)
        {
            var nodes = devices.Split(';', StringSplitOptions.TrimEntries);
            for (int i = 0; i < nodes.Length; i++)
            {
                given[$"node.{i + 1}.devices"] = nodes[i];
            }
        }

        if (context.Option("--override-address") is { } cables)
        {
            var list = cables.Split(';', StringSplitOptions.TrimEntries);
            for (int i = 0; i < list.Length; i++)
            {
                given[$"override.{i + 1}.address"] = list[i];
            }
        }

        if (context.Option("--memory") is { } memory)
        {
            var nodes = memory.Split(';', StringSplitOptions.TrimEntries);
            for (int i = 0; i < nodes.Length; i++)
            {
                var sizes = nodes[i].Split(',', StringSplitOptions.TrimEntries);
                for (int j = 0; j < sizes.Length; j++)
                {
                    given[$"node.{i + 1}.device.{j + 1}.memory"] = sizes[j];
                }
            }
        }

        return given;
    }

    // The option that answers a question key.
    private static string Option(string key) => key switch
    {
        "nodes" => "--nodes",
        "node.1.address" => "--address",
        "intra-node" => "--intra",
        "inter-node" => "--inter",
        "overrides" => "--override",
        _ when key.StartsWith("override.", StringComparison.Ordinal) => "--override-address",
        "seed" => "--seed",
        "failures" => "--failures",
        _ when key.EndsWith(".name", StringComparison.Ordinal) => "--names",
        _ when key.EndsWith(".devices", StringComparison.Ordinal) => "--devices",
        _ => "--memory",
    };
}
