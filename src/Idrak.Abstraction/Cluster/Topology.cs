// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Globalization;
using System.Text;
using System.Text.Json;

namespace Idrak.Abstraction.Cluster;

/// <summary>One device of a node: the physical device it is (or is carved from), and its memory cap.</summary>
/// <param name="Device">The device on the machine that runs the rank ("cuda:0", "vulkan:0", "cpu").</param>
/// <param name="MemoryBytes">The hard cap on its memory, or null for the device's own.</param>
public sealed record ClusterDeviceSpec(string Device, long? MemoryBytes);

/// <summary>One machine of the cluster.</summary>
/// <param name="Name">Its name: the machine's host name, so a process finds its own node.</param>
/// <param name="Address">"host:port" where it can be reached, or null; the first node's is the coordinator rank 0 listens on.</param>
/// <param name="Devices">Its devices, one rank each, in rank order.</param>
public sealed record ClusterNodeSpec(string Name, string? Address, IReadOnlyList<ClusterDeviceSpec> Devices);

/// <summary>A pair of nodes joined by a link of their own.</summary>
/// <param name="A">One node's name.</param>
/// <param name="B">The other's.</param>
/// <param name="Link">The link profile's name (a rate suffix allowed).</param>
public sealed record ClusterLinkOverride(string A, string B, string Link);

/// <summary>What a simulated run of the topology draws from.</summary>
/// <param name="Seed">The seed of jitter and faults.</param>
/// <param name="Failures">The failure model's name.</param>
public sealed record ClusterSimulationSpec(int Seed, string Failures);

/// <summary>
/// A cluster: its nodes (machines), each with its devices (one rank each), and the links between them (plan 23). Ranks are
/// numbered from 0 in file order: the first node's devices, then the next node's. The world size is the device count,
/// never typed again.
/// </summary>
public interface IClusterTopology
{
    /// <summary>The nodes, in order.</summary>
    IReadOnlyList<ClusterNodeSpec> Nodes { get; }

    /// <summary>The number of ranks: every node's devices.</summary>
    int WorldSize { get; }

    /// <summary>The simulation's seed and failure model.</summary>
    ClusterSimulationSpec Simulation { get; }

    /// <summary>
    /// The link between two ranks: <c>same-device</c> when they share a device of one node, the node's intra-node link when
    /// they are different devices of one node, the pair's override or the inter-node link when they are on different nodes.
    /// </summary>
    ILinkProfile Link(int rankA, int rankB);

    /// <summary>The node <paramref name="rank"/> is on and its device's index in that node.</summary>
    (int Node, int Local) Place(int rank);

    /// <summary>The ranks node <paramref name="node"/> holds, in order.</summary>
    IReadOnlyList<int> Ranks(int node);

    /// <summary>The node named <paramref name="hostName"/> (ignoring case), or null.</summary>
    int? NodeOf(string hostName);
}

/// <summary>
/// The topology file (<c>idrak cluster init</c> writes it): JSON with comments and trailing commas allowed. Memory sizes
/// take B, KB/KiB, MB/MiB, GB/GiB, TB/TiB, all powers of 1,024 (decision 7), so "16GB" is a whole card's memory.
/// </summary>
public sealed class ClusterTopology : IClusterTopology
{
    /// <summary>The format this version writes and reads.</summary>
    public const string Format = "idrak-topology/1";

    /// <summary>The link two ranks on one device of one node use.</summary>
    public const string SameDevice = "same-device";

    private readonly int[] _rankNode;
    private readonly int[] _rankLocal;
    private readonly int[][] _nodeRanks;
    private readonly Dictionary<(int, int), ILinkProfile> _overrides = [];
    private readonly ILinkProfile? _intra;
    private readonly ILinkProfile? _inter;
    private readonly ILinkProfile? _same;
    private readonly Dictionary<string, int> _byName = new(StringComparer.OrdinalIgnoreCase);

    private ClusterTopology(IReadOnlyList<ClusterNodeSpec> nodes, string? intraNode, string? interNode, IReadOnlyList<ClusterLinkOverride> overrides,
        ClusterSimulationSpec simulation)
    {
        if (nodes.Count == 0)
        {
            throw new InvalidDataException("A topology needs at least one node.");
        }

        Nodes = nodes;
        IntraNode = intraNode;
        InterNode = interNode;
        Overrides = overrides;
        Simulation = simulation;

        int world = 0;
        bool repeatsDevice = false, mixesDevices = false;
        for (int n = 0; n < nodes.Count; n++)
        {
            var node = nodes[n];
            if (string.IsNullOrWhiteSpace(node.Name))
            {
                throw new InvalidDataException($"Node {n + 1} has no name.");
            }

            if (!_byName.TryAdd(node.Name, n))
            {
                throw new InvalidDataException($"Two nodes are named '{node.Name}' (names are host names, compared ignoring case).");
            }

            if (node.Devices.Count == 0)
            {
                throw new InvalidDataException($"Node '{node.Name}' has no devices (\"gpus\"); each device is one rank.");
            }

            var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (var device in node.Devices)
            {
                if (string.IsNullOrWhiteSpace(device.Device))
                {
                    throw new InvalidDataException($"Node '{node.Name}' has a device with no name (\"device\": \"cuda:0\", \"vulkan:0\" or \"cpu\").");
                }

                if (device.MemoryBytes is <= 0)
                {
                    throw new InvalidDataException($"Node '{node.Name}', device {device.Device}: the memory cap must be more than 0.");
                }

                if (!seen.Add(device.Device))
                {
                    repeatsDevice = true;
                }
            }

            mixesDevices |= seen.Count > 1;
            world = checked(world + node.Devices.Count);
        }

        WorldSize = world;
        _rankNode = new int[world];
        _rankLocal = new int[world];
        _nodeRanks = new int[nodes.Count][];
        for (int n = 0, rank = 0; n < nodes.Count; n++)
        {
            _nodeRanks[n] = new int[nodes[n].Devices.Count];
            for (int local = 0; local < nodes[n].Devices.Count; local++, rank++)
            {
                _rankNode[rank] = n;
                _rankLocal[rank] = local;
                _nodeRanks[n][local] = rank;
            }
        }

        if (mixesDevices)
        {
            _intra = intraNode is null
                ? throw new InvalidDataException("A node has more than one device, so the topology needs \"links\": { \"intra-node\": … } (nvlink, pcie, shared-memory).")
                : LinkProfiles.Resolve(intraNode);
        }
        else if (intraNode is not null)
        {
            _intra = LinkProfiles.Resolve(intraNode);
        }

        if (nodes.Count > 1)
        {
            _inter = interNode is null
                ? throw new InvalidDataException("The topology has more than one node, so it needs \"links\": { \"inter-node\": … } (ethernet-1g, direct-link, infiniband, …).")
                : LinkProfiles.Resolve(interNode);
        }
        else if (interNode is not null)
        {
            _inter = LinkProfiles.Resolve(interNode);
        }

        if (repeatsDevice)
        {
            _same = LinkProfiles.Resolve(SameDevice);
        }

        foreach (var o in overrides)
        {
            if (!_byName.TryGetValue(o.A, out int a) || !_byName.TryGetValue(o.B, out int b))
            {
                throw new InvalidDataException($"The link override between '{o.A}' and '{o.B}' names a node the topology does not have (nodes: {string.Join(", ", nodes.Select(x => x.Name))}).");
            }

            if (a == b)
            {
                throw new InvalidDataException($"The link override names node '{o.A}' twice; an override joins two different nodes.");
            }

            if (!_overrides.TryAdd((Math.Min(a, b), Math.Max(a, b)), LinkProfiles.Resolve(o.Link)))
            {
                throw new InvalidDataException($"Two link overrides join '{o.A}' and '{o.B}'.");
            }
        }

        if (string.IsNullOrWhiteSpace(simulation.Failures))
        {
            throw new InvalidDataException("The simulation's failure model has no name (\"none\" for none).");
        }
    }

    /// <inheritdoc />
    public IReadOnlyList<ClusterNodeSpec> Nodes { get; }

    /// <inheritdoc />
    public int WorldSize { get; }

    /// <inheritdoc />
    public ClusterSimulationSpec Simulation { get; }

    /// <summary>The intra-node link's name as given, or null.</summary>
    public string? IntraNode { get; }

    /// <summary>The inter-node link's name as given, or null.</summary>
    public string? InterNode { get; }

    /// <summary>The pairs of nodes with a link of their own.</summary>
    public IReadOnlyList<ClusterLinkOverride> Overrides { get; }

    /// <summary>The first node's address, where rank 0 listens, or null when the file gives none.</summary>
    public string? Coordinator => Nodes[0].Address;

    /// <summary>A topology from its parts, checked as a file is (link names resolved through <see cref="LinkProfiles"/>).</summary>
    /// <exception cref="InvalidDataException">A part is missing or inconsistent; the message names it.</exception>
    /// <exception cref="NotSupportedException">A link name is not registered.</exception>
    public static ClusterTopology Create(IReadOnlyList<ClusterNodeSpec> nodes, string? intraNode, string? interNode,
        IReadOnlyList<ClusterLinkOverride>? overrides = null, ClusterSimulationSpec? simulation = null)
    {
        ArgumentNullException.ThrowIfNull(nodes);
        return new(nodes, intraNode, interNode, overrides ?? [], simulation ?? new ClusterSimulationSpec(1, "none"));
    }

    /// <inheritdoc />
    public ILinkProfile Link(int rankA, int rankB)
    {
        CheckRank(rankA);
        CheckRank(rankB);
        if (rankA == rankB)
        {
            throw new ArgumentException($"Rank {rankA} has no link to itself.");
        }

        int a = _rankNode[rankA], b = _rankNode[rankB];
        if (a == b)
        {
            var devices = Nodes[a].Devices;
            return string.Equals(devices[_rankLocal[rankA]].Device, devices[_rankLocal[rankB]].Device, StringComparison.OrdinalIgnoreCase)
                ? _same!
                : _intra!;
        }

        return _overrides.TryGetValue((Math.Min(a, b), Math.Max(a, b)), out var own) ? own : _inter!;
    }

    /// <inheritdoc />
    public (int Node, int Local) Place(int rank)
    {
        CheckRank(rank);
        return (_rankNode[rank], _rankLocal[rank]);
    }

    /// <inheritdoc />
    public IReadOnlyList<int> Ranks(int node)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(node);
        ArgumentOutOfRangeException.ThrowIfGreaterThanOrEqual(node, Nodes.Count);
        return _nodeRanks[node];
    }

    /// <inheritdoc />
    public int? NodeOf(string hostName) => _byName.TryGetValue(hostName, out int n) ? n : null;

    private void CheckRank(int rank)
    {
        if ((uint)rank >= (uint)WorldSize)
        {
            throw new ArgumentOutOfRangeException(nameof(rank), rank, $"The topology has ranks 0 to {WorldSize - 1}.");
        }
    }

    /// <summary>Reads the topology file at <paramref name="path"/>.</summary>
    /// <exception cref="InvalidDataException">The file is not a topology; the message names the file and the key.</exception>
    public static ClusterTopology Load(string path)
    {
        try
        {
            return Parse(File.ReadAllBytes(path));
        }
        catch (InvalidDataException e)
        {
            throw new InvalidDataException($"{path}: {e.Message}", e);
        }
    }

    /// <summary>Reads a topology from JSON text.</summary>
    /// <exception cref="InvalidDataException">It is not a topology; the message names the key.</exception>
    public static ClusterTopology Parse(string json) => Parse(Encoding.UTF8.GetBytes(json));

    /// <summary>Reads a topology from UTF-8 JSON.</summary>
    /// <exception cref="InvalidDataException">It is not a topology; the message names the key.</exception>
    public static ClusterTopology Parse(ReadOnlySpan<byte> utf8Json)
    {
        JsonDocument document;
        try
        {
            var reader = new Utf8JsonReader(utf8Json, new JsonReaderOptions { CommentHandling = JsonCommentHandling.Skip, AllowTrailingCommas = true });
            document = JsonDocument.ParseValue(ref reader);
        }
        catch (JsonException e)
        {
            throw new InvalidDataException($"not JSON: {e.Message}", e);
        }

        using (document)
        {
            var root = document.RootElement;
            Expect(root, JsonValueKind.Object, "the file");
            Keys(root, "the file", "format", "nodes", "links", "simulation");
            if (root.TryGetProperty("format", out var format))
            {
                CheckFormat(Text(format, "format"));
            }

            if (!root.TryGetProperty("nodes", out var nodesElement))
            {
                throw new InvalidDataException("no \"nodes\": a topology lists its machines, each with its devices.");
            }

            Expect(nodesElement, JsonValueKind.Array, "\"nodes\"");
            var nodes = new List<ClusterNodeSpec>();
            foreach (var node in nodesElement.EnumerateArray())
            {
                string where = $"node {nodes.Count + 1}";
                Expect(node, JsonValueKind.Object, where);
                Keys(node, where, "name", "address", "gpus");
                string name = node.TryGetProperty("name", out var n) ? Text(n, where + " \"name\"") : throw new InvalidDataException($"{where} has no \"name\".");
                string? address = node.TryGetProperty("address", out var addr) ? Text(addr, $"node '{name}' \"address\"") : null;
                if (!node.TryGetProperty("gpus", out var gpus))
                {
                    throw new InvalidDataException($"node '{name}' has no \"gpus\": each entry is one rank's device.");
                }

                Expect(gpus, JsonValueKind.Array, $"node '{name}' \"gpus\"");
                var devices = new List<ClusterDeviceSpec>();
                foreach (var gpu in gpus.EnumerateArray())
                {
                    string at = $"node '{name}' device {devices.Count + 1}";
                    Expect(gpu, JsonValueKind.Object, at);
                    Keys(gpu, at, "device", "memory");
                    string device = gpu.TryGetProperty("device", out var d) ? Text(d, at + " \"device\"") : throw new InvalidDataException($"{at} has no \"device\".");
                    long? memory = null;
                    if (gpu.TryGetProperty("memory", out var m))
                    {
                        try
                        {
                            memory = ParseMemory(Text(m, at + " \"memory\""));
                        }
                        catch (FormatException e)
                        {
                            throw new InvalidDataException($"{at}: {e.Message}", e);
                        }
                    }

                    devices.Add(new ClusterDeviceSpec(device, memory));
                }

                nodes.Add(new ClusterNodeSpec(name, address, devices));
            }

            string? intra = null, inter = null;
            var overrides = new List<ClusterLinkOverride>();
            if (root.TryGetProperty("links", out var links))
            {
                Expect(links, JsonValueKind.Object, "\"links\"");
                Keys(links, "\"links\"", "intra-node", "inter-node", "overrides");
                intra = links.TryGetProperty("intra-node", out var i) ? Text(i, "\"intra-node\"") : null;
                inter = links.TryGetProperty("inter-node", out var o) ? Text(o, "\"inter-node\"") : null;
                if (links.TryGetProperty("overrides", out var list))
                {
                    Expect(list, JsonValueKind.Array, "\"overrides\"");
                    foreach (var entry in list.EnumerateArray())
                    {
                        string at = $"override {overrides.Count + 1}";
                        Expect(entry, JsonValueKind.Object, at);
                        Keys(entry, at, "between", "type");
                        if (!entry.TryGetProperty("between", out var between) || between.ValueKind != JsonValueKind.Array || between.GetArrayLength() != 2)
                        {
                            throw new InvalidDataException($"{at} needs \"between\": [two node names].");
                        }

                        string type = entry.TryGetProperty("type", out var t) ? Text(t, at + " \"type\"") : throw new InvalidDataException($"{at} has no \"type\" (a link profile).");
                        overrides.Add(new ClusterLinkOverride(Text(between[0], at + " \"between\""), Text(between[1], at + " \"between\""), type));
                    }
                }
            }

            var simulation = new ClusterSimulationSpec(1, "none");
            if (root.TryGetProperty("simulation", out var sim))
            {
                Expect(sim, JsonValueKind.Object, "\"simulation\"");
                Keys(sim, "\"simulation\"", "seed", "failures");
                int seed = 1;
                if (sim.TryGetProperty("seed", out var s) && (s.ValueKind != JsonValueKind.Number || !s.TryGetInt32(out seed)))
                {
                    throw new InvalidDataException("\"seed\" is not a whole number.");
                }

                simulation = new ClusterSimulationSpec(seed, sim.TryGetProperty("failures", out var f) ? Text(f, "\"failures\"") : "none");
            }

            return new ClusterTopology(nodes, intra, inter, overrides, simulation);
        }
    }

    /// <summary>The topology as JSON, in the file's layout (no comments; <c>idrak cluster init</c> adds them).</summary>
    public string ToJson()
    {
        var buffer = new System.Buffers.ArrayBufferWriter<byte>();
        using (var w = new Utf8JsonWriter(buffer, new JsonWriterOptions { Indented = true }))
        {
            w.WriteStartObject();
            w.WriteString("format", Format);
            w.WriteStartArray("nodes");
            foreach (var node in Nodes)
            {
                w.WriteStartObject();
                w.WriteString("name", node.Name);
                if (node.Address is { } address)
                {
                    w.WriteString("address", address);
                }

                w.WriteStartArray("gpus");
                foreach (var device in node.Devices)
                {
                    w.WriteStartObject();
                    w.WriteString("device", device.Device);
                    if (device.MemoryBytes is { } bytes)
                    {
                        w.WriteString("memory", FormatMemory(bytes));
                    }

                    w.WriteEndObject();
                }

                w.WriteEndArray();
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteStartObject("links");
            if (IntraNode is { } intra)
            {
                w.WriteString("intra-node", intra);
            }

            if (InterNode is { } inter)
            {
                w.WriteString("inter-node", inter);
            }

            w.WriteStartArray("overrides");
            foreach (var o in Overrides)
            {
                w.WriteStartObject();
                w.WriteStartArray("between");
                w.WriteStringValue(o.A);
                w.WriteStringValue(o.B);
                w.WriteEndArray();
                w.WriteString("type", o.Link);
                w.WriteEndObject();
            }

            w.WriteEndArray();
            w.WriteEndObject();
            w.WriteStartObject("simulation");
            w.WriteNumber("seed", Simulation.Seed);
            w.WriteString("failures", Simulation.Failures);
            w.WriteEndObject();
            w.WriteEndObject();
        }

        return Encoding.UTF8.GetString(buffer.WrittenSpan) + "\n";
    }

    /// <summary>
    /// Reads a memory size: a number (decimals allowed) and a unit of B, KB/KiB, MB/MiB, GB/GiB or TB/TiB, all powers of
    /// 1,024 (decision 7); a bare number is bytes. Case is ignored.
    /// </summary>
    /// <exception cref="FormatException">It is not a size; the message says what is accepted.</exception>
    public static long ParseMemory(ReadOnlySpan<char> text)
    {
        var trimmed = text.Trim();
        int split = trimmed.Length;
        while (split > 0 && char.IsAsciiLetter(trimmed[split - 1]))
        {
            split--;
        }

        var unit = trimmed[split..];
        var number = trimmed[..split].TrimEnd();
        int power = unit.Length == 0 ? 0 : char.ToUpperInvariant(unit[0]) switch
        {
            'B' when unit.Length == 1 => 0,
            'K' => 1,
            'M' => 2,
            'G' => 3,
            'T' => 4,
            _ => -1,
        };
        bool unitOk = unit.Length switch
        {
            0 => true,
            1 => power >= 0,
            2 => power > 0 && char.ToUpperInvariant(unit[1]) == 'B',
            3 => power > 0 && char.ToUpperInvariant(unit[1]) == 'I' && char.ToUpperInvariant(unit[2]) == 'B',
            _ => false,
        };
        if (!unitOk || power < 0 || !double.TryParse(number, NumberStyles.AllowDecimalPoint, CultureInfo.InvariantCulture, out double count) || !(count >= 0))
        {
            throw new FormatException($"'{text.ToString()}' is not a memory size (a number and B, KB, MB, GB or TB; KiB, MiB, GiB and TiB mean the same, powers of 1,024).");
        }

        double bytes = Math.Round(count * Math.Pow(1024, power));
        return bytes < long.MaxValue ? (long)bytes : throw new FormatException($"'{text.ToString()}' is too large a memory size.");
    }

    /// <summary>A memory size as the file writes it: the largest unit that divides it exactly ("8GB", "512MB", "100B").</summary>
    public static string FormatMemory(long bytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        ReadOnlySpan<string> units = ["B", "KB", "MB", "GB", "TB"];
        int power = 0;
        long value = bytes;
        while (power < units.Length - 1 && value != 0 && value % 1024 == 0)
        {
            value /= 1024;
            power++;
        }

        return value.ToString(CultureInfo.InvariantCulture) + units[power];
    }

    private static void CheckFormat(string format)
    {
        const string prefix = "idrak-topology/";
        if (!format.StartsWith(prefix, StringComparison.Ordinal) || !int.TryParse(format.AsSpan(prefix.Length), NumberStyles.None, CultureInfo.InvariantCulture, out int version))
        {
            throw new InvalidDataException($"\"format\" is '{format}', not '{Format}'.");
        }

        if (version > 1)
        {
            throw new InvalidDataException($"\"format\" is '{format}', written by a newer Idrak; this one reads '{Format}'.");
        }
    }

    private static void Expect(JsonElement element, JsonValueKind kind, string where)
    {
        if (element.ValueKind != kind)
        {
            throw new InvalidDataException($"{where} is {element.ValueKind.ToString().ToLowerInvariant()}, not {(kind == JsonValueKind.Array ? "a list" : "an object")}.");
        }
    }

    private static string Text(JsonElement element, string where) =>
        element.ValueKind == JsonValueKind.String && element.GetString() is { Length: > 0 } text
            ? text
            : throw new InvalidDataException($"{where} is not a text value.");

    private static void Keys(JsonElement element, string where, params ReadOnlySpan<string> allowed)
    {
        foreach (var property in element.EnumerateObject())
        {
            bool known = false;
            foreach (string key in allowed)
            {
                known |= property.NameEquals(key);
            }

            if (!known)
            {
                throw new InvalidDataException($"{where} has an unknown key \"{property.Name}\" (allowed: {string.Join(", ", allowed.ToArray())}).");
            }
        }
    }
}
