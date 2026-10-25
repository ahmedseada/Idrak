// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Net.NetworkInformation;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;

namespace Idrak.Cluster;

/// <summary>A device found on this machine, as the topology builder offers it.</summary>
/// <param name="Device">Its name ("cuda:0", "cpu").</param>
/// <param name="Type">Its type.</param>
/// <param name="MemoryBytes">Its memory, or null when unknown.</param>
/// <param name="Name">What it is, for the listing.</param>
/// <param name="Card">The listed device that drives the same card ("cuda:0" for vulkan:0 on that card), or null when it
/// is its own: ranks on one card share its memory whichever backend each names.</param>
public sealed record MachineDevice(string Device, DeviceType Type, long? MemoryBytes, string Name, string? Card = null)
{
    /// <summary>The card this device is: <see cref="Card"/>, or the device itself.</summary>
    public string CardKey => Card ?? Device;
}

/// <summary>What the topology builder suggests from: this machine's name, network address and adapter, and devices.</summary>
/// <param name="HostName">This machine's host name.</param>
/// <param name="Address">Its network address on the LAN, or null.</param>
/// <param name="AdapterName">The network adapter that address is on, or null.</param>
/// <param name="AdapterBitsPerSecond">The speed the adapter reports, or null when it does not.</param>
/// <param name="Devices">The devices found here, the CPU first.</param>
public sealed record MachineFacts(string HostName, string? Address, string? AdapterName, double? AdapterBitsPerSecond, IReadOnlyList<MachineDevice> Devices)
{
    /// <summary>The facts of this machine: its host name, its LAN address and adapter, and every device it lists (each started to read its memory).</summary>
    public static MachineFacts Gather()
    {
        string? address = null, adapter = null;
        double? speed = null;
        try
        {
            foreach (var nic in NetworkInterface.GetAllNetworkInterfaces())
            {
                if (nic.OperationalStatus != OperationalStatus.Up || nic.NetworkInterfaceType is NetworkInterfaceType.Loopback or NetworkInterfaceType.Tunnel)
                {
                    continue;
                }

                var ipv4 = nic.GetIPProperties().UnicastAddresses.Select(a => a.Address)
                    .FirstOrDefault(a => a.AddressFamily == AddressFamily.InterNetwork && !a.ToString().StartsWith("169.254.", StringComparison.Ordinal));
                if (ipv4 is not null)
                {
                    (address, adapter, speed) = (ipv4.ToString(), nic.Name, nic.Speed > 0 ? nic.Speed : null);
                    break;
                }
            }
        }
        catch (Exception e) when (e is NetworkInformationException or PlatformNotSupportedException)
        {
            // No adapters to read here: the questions ask without a suggestion.
        }

        var devices = new List<MachineDevice> { new("cpu", DeviceType.Cpu, GC.GetGCMemoryInfo().TotalAvailableMemoryBytes, Device.Cpu.Name) };
        foreach (var device in Device.Available.Where(d => d.Type != DeviceType.Cpu))
        {
            try
            {
                devices.Add(new MachineDevice(device.ToString(), device.Type, device.Backend.Hardware.MemoryBytes, device.Name));
            }
            catch (Exception e)
            {
                devices.Add(new MachineDevice(device.ToString(), device.Type, null, $"could not start: {e.Message}"));
            }
        }

        // A device a backend does not list because another backend drives its card (vulkan:0 beside cuda:0) is still a
        // device a rank can name: it is listed with its twin's memory (the same UUID), without starting a backend for it.
        foreach (var provider in DeviceProviders.All.Where(p => p.Name != SimulatedDevices.Kind))
        {
            for (int i = 0; i < provider.Count; i++)
            {
                if (provider.Listed(i) || provider.TypeOf(i) == DeviceType.Cpu)
                {
                    continue;
                }

                var uuid = provider.DeviceUuid(i);
                var twin = uuid is null ? null : Device.Available.FirstOrDefault(d => d.Type != DeviceType.Cpu && DeviceProviders.Find(d.Kind)?.DeviceUuid(d.Ordinal) == uuid);
                var listed = twin is null ? null : devices.FirstOrDefault(d => d.Device == twin.ToString());
                devices.Add(new MachineDevice($"{provider.Name}:{i}", provider.TypeOf(i), listed?.MemoryBytes,
                    twin is null ? $"{provider.Display} device {i}" : $"{twin.Name} (the card {twin} drives)", listed?.Device));
            }
        }

        return new MachineFacts(Environment.MachineName, address, adapter, speed, devices);
    }
}

/// <summary>One question of the topology builder: what it asks, why it matters, and what this machine suggests.</summary>
/// <param name="Key">Its key ("nodes", "node.2.devices", "inter-node"): the name an app or the command line answers it by.</param>
/// <param name="Prompt">The question.</param>
/// <param name="Description">What the input means and why it matters, in a few lines.</param>
/// <param name="Default">What an empty answer gives, or null when an answer is needed.</param>
/// <param name="Suggestion">What this machine suggests and why, or null.</param>
public sealed record TopologyQuestion(string Key, string Prompt, string Description, string? Default, string? Suggestion);

/// <summary>
/// Builds a topology file one question at a time (plan 23, <c>idrak cluster init</c>): how many machines and their names,
/// the first one's address (where rank 0 listens), each machine's devices and each device's memory cap, the links, the
/// simulation's seed and failure model. Every question carries a description and a default or suggestion derived from
/// the device or adapter it concerns on this machine, never from a table; any question can be answered in advance by
/// its key (an app's own form, the command's options), and <see cref="AnswerRest"/> takes the defaults for the rest.
/// </summary>
public sealed class TopologyBuilder
{
    private const int DefaultPort = 29500;

    private readonly MachineFacts _facts;
    private readonly Dictionary<string, string> _answers = new(StringComparer.Ordinal);
    private readonly IReadOnlyDictionary<string, string> _given;

    /// <summary>A builder suggesting from <paramref name="facts"/>, with <paramref name="given"/> answered in advance (by key).</summary>
    public TopologyBuilder(MachineFacts facts, IReadOnlyDictionary<string, string>? given = null)
    {
        _facts = facts ?? throw new ArgumentNullException(nameof(facts));
        _given = given ?? new Dictionary<string, string>();
        foreach (string key in _given.Keys.Where(k => !KnownKey(k)))
        {
            throw new ArgumentException($"'{key}' is not a question of the topology builder (nodes, node.N.name, node.1.address, node.N.devices, node.N.device.M.memory, intra-node, inter-node, overrides, override.N.address, seed, failures).");
        }

        TakeGiven();
    }

    /// <summary>The answers so far, by key.</summary>
    public IReadOnlyDictionary<string, string> Answers => _answers;

    /// <summary>The question to answer next, or null when every question is answered.</summary>
    public TopologyQuestion? Next => Question(NextKey());

    /// <summary>Answers <see cref="Next"/> (blank: its default); returns why the answer is refused, or null when taken.</summary>
    public string? Answer(string? text)
    {
        var question = Next ?? throw new InvalidOperationException("Every question is answered.");
        string value = string.IsNullOrWhiteSpace(text) ? question.Default ?? "" : text.Trim();
        if (string.IsNullOrWhiteSpace(text) && question.Default is null)
        {
            return "this needs an answer";
        }

        if (Validate(question.Key, value) is { } problem)
        {
            return problem;
        }

        _answers[question.Key] = value;
        TakeGiven();
        return null;
    }

    /// <summary>Takes every remaining question's default (no terminal: <c>--yes</c>).</summary>
    /// <exception cref="InvalidOperationException">A question has no default; the message names its key.</exception>
    public void AnswerRest()
    {
        while (Next is { } question)
        {
            if (question.Default is null)
            {
                throw new InvalidOperationException($"'{question.Key}' ({question.Prompt}) has no default: give it.");
            }

            if (Answer(null) is { } problem)
            {
                throw new InvalidOperationException($"'{question.Key}': {problem}");
            }
        }
    }

    /// <summary>The topology the answers describe.</summary>
    /// <exception cref="InvalidOperationException">A question is still open.</exception>
    public ClusterTopology Build()
    {
        if (Next is { } open)
        {
            throw new InvalidOperationException($"'{open.Key}' is not answered yet.");
        }

        int nodes = Nodes;
        var specs = new List<ClusterNodeSpec>();
        for (int i = 1; i <= nodes; i++)
        {
            var devices = Devices(i).Select((d, j) => new ClusterDeviceSpec(d, _answers[$"node.{i}.device.{j + 1}.memory"] is { Length: > 0 } m ? ClusterTopology.ParseMemory(m) : null)).ToList();
            specs.Add(new ClusterNodeSpec(_answers[$"node.{i}.name"], i == 1 && _answers["node.1.address"] is { Length: > 0 } a ? a : null, devices));
        }

        var overrides = Overrides(_answers.GetValueOrDefault("overrides") ?? "");
        for (int i = 0; i < overrides.Count; i++)
        {
            if (_answers.GetValueOrDefault($"override.{i + 1}.address") is { Length: > 0 } cable && CableAddresses(cable) is var (first, second))
            {
                overrides[i] = overrides[i] with { AddressA = first, AddressB = second };
            }
        }

        return ClusterTopology.Create(specs, _answers.GetValueOrDefault("intra-node") is { Length: > 0 } intra ? intra : null,
            _answers.GetValueOrDefault("inter-node") is { Length: > 0 } inter ? inter : null, overrides,
            new ClusterSimulationSpec(int.Parse(_answers["seed"], CultureInfo.InvariantCulture), _answers["failures"]));
    }

    /// <summary>The topology file, a comment on every key (comments and trailing commas are allowed where it is read).</summary>
    public string ToJson()
    {
        var topology = Build();
        var w = new StringBuilder();
        w.Append("{\n");
        w.Append($"  // idrak cluster topology (plan 23). Written by idrak cluster init on {DateTime.Now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture)}; edit freely.\n");
        w.Append("  // Comments and trailing commas are allowed. Sizes take B, KB, MB, GB, TB (KiB, MiB, GiB, TiB the same): powers of 1,024.\n");
        w.Append($"  \"format\": {Q(ClusterTopology.Format)},\n");
        w.Append("  \"nodes\": [\n");
        int rank = 0;
        for (int i = 0; i < topology.Nodes.Count; i++)
        {
            var node = topology.Nodes[i];
            w.Append("    {\n");
            w.Append($"      \"name\": {Q(node.Name)},   // the machine's host name: a process finds its node by it\n");
            if (node.Address is { } address)
            {
                w.Append($"      \"address\": {Q(address)},   // rank 0 listens here; the other machines connect to it\n");
            }

            if (i == 0)
            {
                w.Append("      // device: the physical device the rank's device is carved from, on the machine that runs it (cuda:0, vulkan:0, cpu)\n");
                w.Append("      // memory: a hard cap per rank; past it an allocation fails, or spills to pinned host memory where the device can\n");
            }

            w.Append("      \"gpus\": [\n");
            foreach (var device in node.Devices)
            {
                string memory = device.MemoryBytes is { } bytes ? $", \"memory\": {Q(ClusterTopology.FormatMemory(bytes))}" : "";
                w.Append($"        {{ \"device\": {Q(device.Device)}{memory} }},   // rank {rank++}\n");
            }

            w.Append("      ]\n");
            w.Append(i + 1 < topology.Nodes.Count ? "    },\n" : "    }\n");
        }

        w.Append("  ],\n");
        w.Append("  \"links\": {\n");
        if (topology.IntraNode is { } intraNode)
        {
            w.Append($"    \"intra-node\": {Q(intraNode)},   // different devices of one machine (ranks sharing a device use same-device by themselves)\n");
        }

        if (topology.InterNode is { } interNode)
        {
            w.Append($"    \"inter-node\": {Q(interNode)},   // between machines; a rate suffix sets the bandwidth (ethernet-2.5g)\n");
        }

        w.Append("    \"overrides\": [");
        w.Append(string.Join(", ", topology.Overrides.Select(o => $"{{ \"between\": [{Q(o.A)}, {Q(o.B)}], \"type\": {Q(o.Link)}"
            + (o.AddressA is not null || o.AddressB is not null ? $", \"address\": [{(o.AddressA is { } a ? Q(a) : "null")}, {(o.AddressB is { } b ? Q(b) : "null")}]" : "") + " }")));
        w.Append("]   // pairs of machines with a link of their own (a cable that skips the switch; \"address\": each end's address on it)\n");
        w.Append("  },\n");
        w.Append("  \"simulation\": {\n");
        w.Append($"    \"seed\": {topology.Simulation.Seed.ToString(CultureInfo.InvariantCulture)},   // fixes jitter and faults: a simulated run replays bit for bit\n");
        w.Append($"    \"failures\": {Q(topology.Simulation.Failures)}   // {string.Join(" | ", FailureModels.Names)}\n");
        w.Append("  }\n");
        w.Append($"  // {topology.WorldSize} ranks in this order. The world size is never typed again: idrak cluster start finds this machine's node by\n");
        w.Append("  // its host name and starts its ranks; IDRAK_RANK and IDRAK_WORLD_SIZE are read only without a file.\n");
        w.Append("}\n");
        string json = w.ToString();
        _ = ClusterTopology.Parse(json);                               // what is written reads back
        return json;
    }

    private int Nodes => _answers.TryGetValue("nodes", out var n) ? int.Parse(n, CultureInfo.InvariantCulture) : 0;

    private string[] Devices(int node) => _answers.TryGetValue($"node.{node}.devices", out var list)
        ? list.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries)
        : [];

    // Every key the answers so far make due, in order; the first unanswered one is next.
    private IEnumerable<string> Keys()
    {
        yield return "nodes";
        int nodes = Nodes;
        for (int i = 1; i <= nodes; i++)
        {
            yield return $"node.{i}.name";
            if (i == 1)
            {
                yield return "node.1.address";
            }
        }

        for (int i = 1; i <= nodes; i++)
        {
            yield return $"node.{i}.devices";
            for (int j = 1; j <= Devices(i).Length; j++)
            {
                yield return $"node.{i}.device.{j}.memory";
            }
        }

        if (nodes > 0 && Enumerable.Range(1, nodes).All(i => _answers.ContainsKey($"node.{i}.devices")))
        {
            if (Enumerable.Range(1, nodes).Any(i => Devices(i).Distinct(StringComparer.OrdinalIgnoreCase).Count() > 1))
            {
                yield return "intra-node";
            }

            if (nodes > 1)
            {
                yield return "inter-node";
                yield return "overrides";
                if (_answers.TryGetValue("overrides", out string? given))
                {
                    int count = Overrides(given).Count;
                    for (int i = 1; i <= count; i++)
                    {
                        yield return $"override.{i}.address";
                    }
                }
            }

            yield return "seed";
            yield return "failures";
        }
    }

    private string? NextKey() => Keys().FirstOrDefault(k => !_answers.ContainsKey(k));

    private void TakeGiven()
    {
        while (NextKey() is { } key && _given.TryGetValue(key, out var value))
        {
            if (Validate(key, value.Trim()) is { } problem)
            {
                throw new ArgumentException($"'{key}={value}': {problem}");
            }

            _answers[key] = value.Trim();
        }
    }

    private static bool KnownKey(string key)
    {
        string[] parts = key.Split('.');
        return key is "nodes" or "intra-node" or "inter-node" or "overrides" or "seed" or "failures" or "node.1.address"
               || (parts.Length == 3 && parts[0] == "override" && int.TryParse(parts[1], out _) && parts[2] == "address")
               || (parts.Length == 3 && parts[0] == "node" && int.TryParse(parts[1], out _) && parts[2] is "name" or "devices")
               || (parts.Length == 5 && parts[0] == "node" && int.TryParse(parts[1], out _) && parts[2] == "device" && int.TryParse(parts[3], out _) && parts[4] == "memory");
    }

    private TopologyQuestion? Question(string? key)
    {
        if (key is null)
        {
            return null;
        }

        string[] parts = key.Split('.');
        string first = _facts.Devices.FirstOrDefault(d => d.Type != DeviceType.Cpu)?.Device ?? "cpu";
        if (parts.Length == 3 && parts[0] == "override")
        {
            var o = Overrides(_answers["overrides"])[int.Parse(parts[1], CultureInfo.InvariantCulture) - 1];
            var others = OtherAdapters();
            return new(key, $"Addresses of {o.A} and {o.B} on their own link ({o.A}'s,{o.B}'s; empty for their node addresses)?",
                "A cable on a subnet of its own reaches each machine at an address the LAN does not use; the direct-link transport\n" +
                "dials it. Give each end's address on the cable (a host; a port is ignored), or leave one side empty for its node's.",
                "",
                others.Count > 0
                    ? $"this machine's adapters off the LAN: {string.Join(", ", others.Select(a => $"{a.Address} ({a.Adapter})"))}"
                    : "none found off the LAN here: the nodes' own addresses");
        }

        switch (key)
        {
            case "nodes":
                return new(key, "How many machines (nodes) does the cluster have?",
                    "A node is one machine with its own memory and devices. Links inside a node are fast; links between nodes are the\n" +
                    "slow part a cluster pays for. A simulated cluster runs every node on this machine.",
                    "1", "1 (this machine); more when imitating a LAN or a rental");
            case "node.1.address":
                return new(key, "Address of node 1, where rank 0 listens (host:port; empty for none)?",
                    "The other machines connect here first; with it in the file nobody types --coordinator. A simulated run does not use it.",
                    _facts.Address is { } a ? $"{a}:{DefaultPort}" : "",
                    _facts.Address is { } b ? $"{b}:{DefaultPort} (this machine's address on {_facts.AdapterName})" : "none found: give host:port, or leave it empty");
            case "intra-node":
                return new(key, "Link inside a node, between its different devices?",
                    $"Profiles: {Profiles("pcie", "nvlink", "shared-memory")}. Ranks sharing one device are joined by same-device by\n" +
                    "themselves (a copy inside one memory, measured on the device).",
                    _facts.Devices.Any(d => d.Type != DeviceType.Cpu) ? "pcie" : "shared-memory",
                    _facts.Devices.Any(d => d.Type != DeviceType.Cpu)
                        ? "pcie (two cards through the host, measured here); nvlink-RATE where the cards are joined by NVLink"
                        : "shared-memory (CPU ranks of one machine, measured here)");
            case "inter-node":
                string suggested = Ethernet(_facts.AdapterBitsPerSecond);
                return new(key, "Link between nodes?",
                    $"Profiles: {Profiles("ethernet-1g", "ethernet-2.5g", "ethernet-10g", "ethernet-100g", "direct-link", "infiniband", "roce", "thunderbolt", "usb4", "wifi")}.\n" +
                    "A name without a rate needs one (infiniband-200g, wifi-300m). A home LAN is usually ethernet-1g; a pod infiniband.",
                    suggested,
                    _facts.AdapterBitsPerSecond is { } bits
                        ? string.Create(CultureInfo.InvariantCulture, $"{suggested} (adapter \"{_facts.AdapterName}\" reports {bits / 1e9:0.###} Gb/s)")
                        : $"{suggested} (no adapter speed reported here)");
            case "overrides":
                return new(key, "Pairs of nodes with a link of their own (a,b=profile; separated by ';', empty for none)?",
                    "For a cable between two machines that skips the switch, or a faster card on one pair (desk,mini=direct-link-10g).",
                    "", "none, unless two machines have a link of their own");
            case "seed":
                return new(key, "Seed for simulated runs?", "Fixes jitter and faults, so a simulated run replays bit for bit.", "1", null);
            case "failures":
                return new(key, "Failure model for simulated runs?",
                    $"Models: {string.Join(", ", FailureModels.Names)}. Overridable per run.", "none", "none for a first run");
        }

        int node = int.Parse(parts[1], CultureInfo.InvariantCulture);
        if (parts[2] == "name")
        {
            return new(key, $"Name of node {node}?",
                "A process finds its own node by the machine's host name, so name each node after its host (case does not matter).\n" +
                "Names appear in logs, telemetry and overrides.",
                node == 1 ? _facts.HostName : $"node{node}", node == 1 ? $"{_facts.HostName} (this machine's host name)" : null);
        }

        if (parts[2] == "devices")
        {
            return new(key, $"Devices of node {_answers[$"node.{node}.name"]}, as a comma list?",
                "Each entry is one rank: the physical device on this machine its device is carved from (cuda:0, vulkan:0, cpu).\n" +
                "Naming one device twice makes two ranks on it; they share its compute and each has its own cap.",
                first, $"{first} ({string.Join("; ", _facts.Devices.Select(d => d.MemoryBytes is { } m ? string.Create(CultureInfo.InvariantCulture, $"{d.Device} {m / (1 << 20):N0} MiB") : d.Device))})");
        }

        int index = int.Parse(parts[3], CultureInfo.InvariantCulture);
        string device = Devices(node)[index - 1];
        var found = _facts.Devices.FirstOrDefault(d => d.Device.Equals(device, StringComparison.OrdinalIgnoreCase));
        string CardOf(string d) => _facts.Devices.FirstOrDefault(f => f.Device.Equals(d, StringComparison.OrdinalIgnoreCase))?.CardKey ?? d;
        int sharing = Devices(node).Count(d => CardOf(d).Equals(CardOf(device), StringComparison.OrdinalIgnoreCase));
        string? split = found?.MemoryBytes is { } total ? ClusterTopology.FormatMemory(total / sharing / (1 << 20) * (1 << 20)) : null;
        return new(key, $"Memory cap of {_answers[$"node.{node}.name"]}/{device}{(sharing > 1 ? $" (rank {index} of the node)" : "")}?",
            "A hard limit for that rank's device: past it an allocation fails, or spills to pinned host memory where the device can.\n" +
            "A cap above the physical memory is allowed. Use the real card's size when imitating a machine you own; empty for its own.",
            split ?? "",
            split is null ? $"{device} is not on this machine: give the size of the card on that machine" :
            sharing > 1 ? $"{split} ({(found?.Card is { } card ? $"{device} is {card}'s card; its" : $"{device}'s")} memory split between the {sharing} ranks on it)" : $"{split} ({device}'s memory)");
    }

    private string? Validate(string key, string value)
    {
        string[] parts = key.Split('.');
        try
        {
            if (parts.Length == 3 && parts[0] == "override")
            {
                return value.Length == 0 || CableAddresses(value) is not null ? null : "two addresses separated by ',' (a side left empty takes its node's address)";
            }

            switch (key)
            {
                case "nodes":
                    return int.TryParse(value, NumberStyles.None, CultureInfo.InvariantCulture, out int n) && n is >= 1 and <= 4096 ? null : "a number of machines from 1";
                case "node.1.address":
                    return value.Length == 0 || (value.LastIndexOf(':') is var colon and > 0 && int.TryParse(value.AsSpan(colon + 1), NumberStyles.None, CultureInfo.InvariantCulture, out int port) && port is >= 1 and <= 65535)
                        ? null : "host:port (port 1 to 65535), or empty";
                case "intra-node" or "inter-node":
                    LinkProfiles.Resolve(value);
                    return null;
                case "overrides":
                    _ = Overrides(value).Select(o => (Node(o.A), Node(o.B), LinkProfiles.Resolve(o.Link))).ToList();
                    return null;
                case "seed":
                    return int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out _) ? null : "a whole number";
                case "failures":
                    return FailureModels.Find(value) is null ? $"not a failure model ({string.Join(", ", FailureModels.Names)})" : null;
            }

            if (parts[2] == "name")
            {
                int node = int.Parse(parts[1], CultureInfo.InvariantCulture);
                bool taken = Enumerable.Range(1, node - 1).Any(i => _answers.GetValueOrDefault($"node.{i}.name")?.Equals(value, StringComparison.OrdinalIgnoreCase) == true);
                return value.Length == 0 || !char.IsAsciiLetterOrDigit(value[0]) || value.Any(c => !(char.IsAsciiLetterOrDigit(c) || c is '-' or '_' or '.'))
                    ? "letters, digits, '-', '_' and '.', starting with a letter or digit"
                    : taken ? $"another node is named {value}" : null;
            }

            if (parts[2] == "devices")
            {
                var entries = value.Split(',', StringSplitOptions.TrimEntries);
                return entries.Length == 0 || entries.Any(e => !DeviceName(e))
                    ? "device names separated by commas (cuda:0, vulkan:0, cpu)"
                    : entries.Any(e => e.StartsWith(SimulatedDevices.Kind + ":", StringComparison.OrdinalIgnoreCase) || e.Equals(SimulatedDevices.Kind, StringComparison.OrdinalIgnoreCase))
                        ? "name physical devices: a carved sim device cannot be carved again"
                        : null;
            }

            if (value.Length > 0)
            {
                // A bare number is refused: 4048 would be 4,048 bytes, never what a cap means.
                return value.All(char.IsAsciiDigit) ? $"a size with its unit ({value}MB, {value}GB; B, KB, MB, GB, TB, powers of 1,024)"
                    : ClusterTopology.ParseMemory(value) > 0 ? null : "a size above 0";
            }

            return null;
        }
        catch (Exception e) when (e is FormatException or NotSupportedException or InvalidDataException or ArgumentException)
        {
            return e.Message;
        }
    }

    private string Node(string name) =>
        Enumerable.Range(1, Nodes).Any(i => _answers.GetValueOrDefault($"node.{i}.name")?.Equals(name, StringComparison.OrdinalIgnoreCase) == true)
            ? name
            : throw new ArgumentException($"no node is named {name}");

    // "a,b" (either side may be empty: its node's address) as the two ends' addresses, or null when it is not that.
    private static (string? First, string? Second)? CableAddresses(string text)
    {
        var parts = text.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length != 2 || parts.All(p => p.Length == 0) || parts.Any(p => p.Any(char.IsWhiteSpace)))
        {
            return null;
        }

        return (parts[0].Length > 0 ? parts[0] : null, parts[1].Length > 0 ? parts[1] : null);
    }

    // This machine's IPv4 addresses on adapters other than the LAN's (a cable's own subnet), for the cable question's suggestion.
    private List<(string Address, string Adapter)> OtherAdapters()
    {
        var found = new List<(string, string)>();
        try
        {
            foreach (var nic in NetworkInterface.GetAllNetworkInterfaces())
            {
                if (nic.OperationalStatus != OperationalStatus.Up || nic.NetworkInterfaceType is NetworkInterfaceType.Loopback or NetworkInterfaceType.Tunnel)
                {
                    continue;
                }

                foreach (var address in nic.GetIPProperties().UnicastAddresses.Select(a => a.Address).Where(a => a.AddressFamily == AddressFamily.InterNetwork))
                {
                    string text = address.ToString();
                    if (text != _facts.Address && !text.StartsWith("169.254.", StringComparison.Ordinal))
                    {
                        found.Add((text, nic.Name));
                    }
                }
            }
        }
        catch (Exception e) when (e is NetworkInformationException or PlatformNotSupportedException)
        {
            // No adapters to read here: no suggestion.
        }

        return found;
    }

    private static List<ClusterLinkOverride> Overrides(string text)
    {
        var list = new List<ClusterLinkOverride>();
        foreach (string entry in text.Split(';', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries))
        {
            int equals = entry.IndexOf('=');
            string[] pair = equals > 0 ? entry[..equals].Split(',', StringSplitOptions.TrimEntries) : [];
            if (pair.Length != 2 || equals == entry.Length - 1)
            {
                throw new ArgumentException($"'{entry}' is not a,b=profile");
            }

            list.Add(new ClusterLinkOverride(pair[0], pair[1], entry[(equals + 1)..].Trim()));
        }

        return list;
    }

    private static bool DeviceName(string text)
    {
        int colon = text.IndexOf(':');
        string kind = colon < 0 ? text : text[..colon];
        return kind.Length > 0 && kind.All(char.IsAsciiLetterOrDigit)
               && (colon < 0 || int.TryParse(text.AsSpan(colon + 1), NumberStyles.None, CultureInfo.InvariantCulture, out _));
    }

    // The Ethernet profile nearest the adapter's reported speed (no speed: the 1 Gb/s of most home networks).
    private static string Ethernet(double? bitsPerSecond)
    {
        if (bitsPerSecond is not { } bits)
        {
            return "ethernet-1g";
        }

        var (name, _) = new[] { ("ethernet-1g", 1e9), ("ethernet-2.5g", 2.5e9), ("ethernet-10g", 10e9), ("ethernet-100g", 100e9) }
            .MinBy(p => Math.Abs(Math.Log(p.Item2) - Math.Log(bits)));
        return name;
    }

    private static string Profiles(params string[] names) => string.Join(" · ", names.Select(n =>
    {
        var profile = LinkProfiles.Find(n);
        return profile is null ? n
            : profile.BandwidthBytesPerSecond > 0 ? string.Create(CultureInfo.InvariantCulture, $"{n} {profile.BandwidthBytesPerSecond / 1e6:0.#} MB/s")
            : profile is MeasuredLink ? $"{n} measured here" : $"{n} (give a rate)";
    }));

    // A JSON string, non-ASCII names kept readable.
    private static string Q(string text) => "\"" + JsonEncodedText.Encode(text, System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping).Value + "\"";
}
