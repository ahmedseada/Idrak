// Copyright (c) 2026 Ahmed Seada
// Licensed under the Apache License, Version 2.0. See LICENSE in the repository root.

using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;
using Idrak.Abstraction.Cluster;
using Idrak.Abstraction.Testing;

// Plan 23, phase C0: the cluster contracts in Idrak.Abstraction (transports, collectives, link profiles, topology,
// selectors, failure models), their options, the wire format, the network telemetry, the kit's four suites (checked on
// the test's reference implementations and on broken twins), and the PyTorch DDP reference fixture.
internal static partial class Tests
{
    private static readonly (string Name, Action<Device> Run)[] ClusterGroup =
    [
        ("cluster: wire header round trip, every field and byte; fewer bytes than a header wait for more; a wrong magic and a newer version refused naming both versions; the keyed hello verified, a wrong key or rank refused, the key never written", ClusterWireFormat),
        ("cluster: options from environment, every variable read (sizes with units, shares, flags, choices), a bad value naming its variable and what it takes, the key never in ToString", ClusterOptionsFromEnvironment),
        ("cluster: the five registries refuse an unregistered name with the Register hint, take an app's entry and give it back with Unregister", ClusterRegistries),
        ("cluster: link profiles resolve a rate suffix in bits a second (direct-link-10g); the kit's LinkProfileSuite passes a profile and fails one whose time shrinks with the bytes", ClusterLinkProfiles),
        ("cluster: topology file parses (comments, trailing commas, memory in GB and GiB as powers of 1,024), ranks in file order, same-device for a shared card, the intra-node and inter-node links and an override, the first node's address as coordinator; refusals name the key; the kit's TopologySuite passes and the JSON reads back", ClusterTopologyFile),
        ("cluster: the kit's TransportSuite passes on the reference in-memory transport and fails one that corrupts messages", ClusterTransportSuite),
        ("cluster: the kit's CollectivesSuite passes on the reference naive collectives over buffers on the device, and fails collectives that do not reduce", ClusterCollectivesSuite),
        ("cluster: the network telemetry events reach the console, the metrics recorder and the JSON lines log, at the network level only", ClusterTelemetry),
        ("cluster: ddp fixture agrees with single (losses, step 1 gradients, weights and moments within 1e-5; bf16 within its measured tolerance; every file as the manifest hashes it)", ClusterDdpFixture),
    ];

    private static string ClusterData(string relative) => TestData(Path.Combine("cluster", relative));

    // The link profiles the topology tests name, registered over whatever the library has and removed after.
    private static IDisposable ClusterTestProfiles()
    {
        LinkProfile[] profiles =
        [
            new(ClusterTopology.SameDevice, 400e9, TimeSpan.FromMicroseconds(8)),
            new("pcie", 25e9, TimeSpan.FromMicroseconds(5)),
            new("ethernet-1g", 125e6, TimeSpan.FromMicroseconds(80)) { Jitter = TimeSpan.FromMicroseconds(20) },
            new("direct-link", 125e6, TimeSpan.FromMicroseconds(40)),
        ];
        foreach (var p in profiles)
        {
            LinkProfiles.Register(p);
        }

        return new Undo(() =>
        {
            foreach (var p in profiles)
            {
                LinkProfiles.Unregister(p.Name);
            }
        });
    }

    private sealed class Undo(Action undo) : IDisposable
    {
        public void Dispose() => undo();
    }

    private static void ClusterWireFormat(Device device)
    {
        if (!FirstRun(nameof(ClusterWireFormat)))
        {
            return;
        }

        var header = new WireHeader(WireMessageKind.Chunk, 3, -7, 123_456u, 9u, 11u, WireElement.BFloat16, ReduceOp.Max, 0xA5A5, 65_537u);
        var bytes = new byte[WireFormat.HeaderBytes + 4];
        WireFormat.WriteHeader(bytes, header);
        Check(bytes.AsSpan(0, 4).SequenceEqual("IDRK"u8) && bytes[4] == 1 && bytes[5] == 0, $"the header starts with IDRK and version 1: {Convert.ToHexString(bytes, 0, 8)}");
        Check(WireFormat.TryReadHeader(bytes, out var back) && back == header, $"the header reads back field for field: {back}");
        Check(!WireFormat.TryReadHeader(bytes.AsSpan(0, WireFormat.HeaderBytes - 1), out _), "fewer bytes than a header wait for more");
        var wrong = (byte[])bytes.Clone();
        wrong[0] = (byte)'X';
        ThrowsNaming<InvalidDataException>(() => WireFormat.TryReadHeader(wrong, out _), "a wrong magic", "IDRK");
        var newer = (byte[])bytes.Clone();
        newer[4] = 2;
        ThrowsNaming<InvalidDataException>(() => WireFormat.TryReadHeader(newer, out _), "a newer version", "version 2", "version 1");

        var key = ClusterKey.FromText("a long shared secret of the owner");
        Span<byte> nonce = stackalloc byte[WireFormat.NonceBytes];
        WireFormat.NewNonce(nonce);
        Span<byte> mac = stackalloc byte[WireFormat.MacBytes];
        WireFormat.SignHello(key, nonce, 2, 4, mac);
        Check(WireFormat.VerifyHello(key, nonce, 2, 4, mac), "the hello verifies under its key");
        Check(!WireFormat.VerifyHello(ClusterKey.FromText("another secret of sixteen+"), nonce, 2, 4, mac), "another key is refused");
        Check(!WireFormat.VerifyHello(key, nonce, 3, 4, mac) && !WireFormat.VerifyHello(key, nonce, 2, 5, mac), "the hello is bound to its rank and world size");
        Check(key.ToString() == "(secret)", "the key is never written");
        ThrowsNaming<ArgumentException>(() => ClusterKey.FromText("short"), "a short key", "16");
    }

    private static void ClusterOptionsFromEnvironment(Device device)
    {
        if (!FirstRun(nameof(ClusterOptionsFromEnvironment)))
        {
            return;
        }

        // A value for every variable that differs from its default.
        var good = new Dictionary<string, string>
        {
            ["IDRAK_CLUSTER_TOPOLOGY"] = "lan.json", ["IDRAK_CLUSTER_NODE"] = "desk", ["IDRAK_RANK"] = "2", ["IDRAK_WORLD_SIZE"] = "4",
            ["IDRAK_COORDINATOR"] = "192.168.1.10:29500", ["IDRAK_CLUSTER_TRANSPORT"] = "tcp", ["IDRAK_CLUSTER_SELECTOR"] = "mine",
            ["IDRAK_CLUSTER_COLLECTIVE"] = "ring", ["IDRAK_CLUSTER_KEY"] = "a long shared secret of the owner", ["IDRAK_CLUSTER_PORT"] = "29501",
            ["IDRAK_CLUSTER_TIMEOUT"] = "2.5", ["IDRAK_CLUSTER_RETRIES"] = "5", ["IDRAK_CLUSTER_CHUNK_BYTES"] = "4MB", ["IDRAK_CLUSTER_IN_FLIGHT"] = "4",
            ["IDRAK_CLUSTER_SOCKET_BUFFER"] = "512KiB", ["IDRAK_CLUSTER_BUCKET_BYTES"] = "25MB", ["IDRAK_CLUSTER_WIRE"] = "bf16", ["IDRAK_CLUSTER_TOPK"] = "0.01",
            ["IDRAK_CLUSTER_ERROR_FEEDBACK"] = "0", ["IDRAK_CLUSTER_LOCAL_SGD"] = "4", ["IDRAK_CLUSTER_SYNC_WEIGHTS"] = "10", ["IDRAK_CLUSTER_REBALANCE_EVERY"] = "50",
            ["IDRAK_CLUSTER_REBALANCE_STEPS"] = "5", ["IDRAK_CLUSTER_REBALANCE_TOLERANCE"] = "0.25", ["IDRAK_CLUSTER_GRADIENT_SYNC"] = "mine", ["IDRAK_CLUSTER_START"] = "check",
            ["IDRAK_CLUSTER_SHARD"] = "parameters", ["IDRAK_CLUSTER_PLACEMENT"] = "pipeline", ["IDRAK_CLUSTER_MICROBATCHES"] = "8", ["IDRAK_CLUSTER_SCHEDULE"] = "1f1b",
            ["IDRAK_CLUSTER_SUGGEST_SHARE"] = "0.3", ["IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE"] = "0.002", ["IDRAK_CLUSTER_PROFILES_FILE"] = "profiles.txt",
            ["IDRAK_CLUSTER_SEED"] = "7", ["IDRAK_CLUSTER_MODE"] = "localhost", ["IDRAK_CLUSTER_THROTTLE"] = "off", ["IDRAK_CLUSTER_FAILURES"] = "mixed",
            ["IDRAK_CLUSTER_REPLAY_STEP"] = "12", ["IDRAK_CLUSTER_CLOCK"] = "simulated",
            // C3: the carving device, the event log's file and whether it records transfers.
            ["IDRAK_CLUSTER_CARVE_FROM"] = "cpu", ["IDRAK_CLUSTER_EVENT_LOG"] = "run.events", ["IDRAK_CLUSTER_LOG_TRANSFERS"] = "0",
            // C9: the control file a launcher steers its processes through, and how often it is read while paused.
            ["IDRAK_CLUSTER_CONTROL"] = "run.control", ["IDRAK_CLUSTER_CONTROL_POLL"] = "0.5",
            // C4: overlap of the gradient buckets with the backward pass, and the buckets' compression by name.
            ["IDRAK_CLUSTER_OVERLAP"] = "0", ["IDRAK_CLUSTER_COMPRESSION"] = "top-k", ["IDRAK_CLUSTER_MESSAGE_COST"] = "0.00005",
            // C5: the rank's device, the run's name, the launcher, the rendezvous's retry wait, the shared-memory rings.
            ["IDRAK_CLUSTER_DEVICE"] = "cuda:1", ["IDRAK_CLUSTER_RUN"] = "run-7", ["IDRAK_CLUSTER_PARENT"] = "4242", ["IDRAK_CLUSTER_RETRY_DELAY"] = "0.25",
            ["IDRAK_CLUSTER_SHM_BYTES"] = "1MB", ["IDRAK_CLUSTER_SHM_FOLDER"] = "rings",
            // C6: receive credit in bytes, background staging, device collectives, a sharded layer's gather size, the shared-memory doorbell.
            ["IDRAK_CLUSTER_RECEIVE_BYTES"] = "1MB", ["IDRAK_CLUSTER_ASYNC_STAGING"] = "0", ["IDRAK_CLUSTER_DEVICE_COLLECTIVES"] = "0", ["IDRAK_CLUSTER_DEVICE_ACTIVATIONS"] = "0", ["IDRAK_CLUSTER_DIRECT_WAKE"] = "0", ["IDRAK_CLUSTER_OWN_TENSORS"] = "0", ["IDRAK_CLUSTER_STREAMED_SAVE"] = "0",
            ["IDRAK_CLUSTER_GATHER_BYTES"] = "64MB", ["IDRAK_CLUSTER_SHM_SPIN"] = "0.0001", ["IDRAK_CLUSTER_SHM_CHECK"] = "2",
            // C7: a pipeline's stages and how they are cut.
            ["IDRAK_CLUSTER_STAGES"] = "2", ["IDRAK_CLUSTER_PIPELINE_BALANCE"] = "memory", ["IDRAK_CLUSTER_PIPELINE_CUTS"] = "3,6",
        };
        Check(good.Keys.Order().SequenceEqual(ClusterOptions.Variables.Order()), $"every variable has a test value: {string.Join(", ", ClusterOptions.Variables.Except(good.Keys))}");
        var defaults = new ClusterOptions();
        Check(ClusterOptions.From(_ => null, null) == defaults, "no variable leaves the defaults");
        foreach (var (name, value) in good)
        {
            var one = ClusterOptions.From(n => n == name ? value : null, null);
            Check(one != defaults, $"{name}={value} is read");
        }

        var all = ClusterOptions.From(n => good.GetValueOrDefault(n), null);
        Check(all is { RebalanceSteps: 5, RebalanceTolerance: 0.25, GradientSync: "mine", GradientSyncName: "mine", StartingState: StartingState.Check, Overlap: false, Compression: "top-k" } && all.MessageCost == TimeSpan.FromSeconds(0.00005),
            $"the data-parallel settings parsed: {all}");
        Check(defaults is { GradientSyncName: "all-reduce", CompressionName: "none", Overlap: true } && new ClusterOptions { LocalSgdEvery = 4 }.GradientSyncName == "local-sgd"
              && new ClusterOptions { Wire = WireType.BFloat16 }.CompressionName == "bf16" && new ClusterOptions { TopK = 0.1 }.CompressionName == "top-k",
            "unset, the synchronisation follows IDRAK_CLUSTER_LOCAL_SGD and the compression IDRAK_CLUSTER_TOPK and IDRAK_CLUSTER_WIRE");
        Check(all is { Device: "cuda:1", RunId: "run-7", Parent: 4242, SharedMemoryBytes: 1L << 20, SharedMemoryFolder: "rings" } && all.RetryDelay == TimeSpan.FromSeconds(0.25),
            $"the C5 settings parsed: {all}");
        Check(all is { ReceiveBytes: 1L << 20, AsyncStaging: false, DeviceCollectives: false, DeviceActivations: false, DirectWake: false, OwnTensorsOnly: false, StreamedSave: false, GatherBytes: 64L << 20, SharedMemorySpin: { } spin } && spin == TimeSpan.FromSeconds(0.0001)
              && all.SharedMemoryCheck == TimeSpan.FromSeconds(2) && new ClusterOptions { Shard = ShardMode.Optimizer }.GradientSyncName == "sharded",
            $"the C6 settings parsed, and a shard picks the sharded synchronisation: {all}");
        Check(all is { Rank: 2, WorldSize: 4, Wire: WireType.BFloat16, Shard: ShardMode.Parameters, ChunkBytes: 4L << 20, SocketBufferBytes: 512L << 10, BucketBytes: 25L << 20,
                  TopK: 0.01, ErrorFeedback: false, LocalSgdEvery: 4, Simulation: { Seed: 7, Mode: SimulationMode.Localhost, Throttle: false, Failures: "mixed", ReplayStep: 12, Clock: ThroughputClock.Simulated,
                      CarveFrom: "cpu", EventLog: "run.events", LogTransfers: false } }
              && all.Timeout == TimeSpan.FromSeconds(2.5) && all.Key is not null,
            $"every value parsed: {all}");
        Check(!all.ToString().Contains("owner", StringComparison.Ordinal) && all.ToString().Contains("(secret)", StringComparison.Ordinal), "the key never in ToString");
        foreach (var (name, value, expected) in new[]
                 {
                     ("IDRAK_RANK", "-1", "0 or more"), ("IDRAK_CLUSTER_WIRE", "fp8", "f32 or bf16"), ("IDRAK_CLUSTER_CHUNK_BYTES", "4QB", "size"),
                     ("IDRAK_CLUSTER_KEY", "short", "16"), ("IDRAK_CLUSTER_TOPK", "2", "0 to 1"), ("IDRAK_CLUSTER_MODE", "cloud", "in-process or localhost"),
                     ("IDRAK_CLUSTER_PORT", "70000", "0 to 65535"), ("IDRAK_CLUSTER_SHARD", "3", "none, optimizer"),
                     ("IDRAK_CLUSTER_START", "copy", "broadcast or check"), ("IDRAK_CLUSTER_REBALANCE_TOLERANCE", "-1", "0 or more"),
                 })
        {
            ThrowsNaming<ArgumentException>(() => ClusterOptions.From(n => n == name ? value : null, null), $"{name}={value}", name, expected);
        }
    }

    private sealed class TestTransportProvider : ITransportProvider
    {
        public string Name => "test-transport";

        public string Summary => "a test's";

        public TransportCapabilities Capabilities => default;

        public TransportAvailability Availability(ClusterContext context, int peer) => new(Name, false, "a test's");

        public ITransport Connect(ClusterContext context, CancellationToken cancellationToken = default) => throw new NotSupportedException();
    }

    private sealed class TestCollectivesProvider : ICollectivesProvider
    {
        public string Name => "test-collectives";

        public string Summary => "a test's";

        public ICollectives Create(ITransport transport, Device device, ClusterContext context) => new NaiveReferenceCollectives(transport, device);
    }

    private sealed class TestSelector : ITransportSelector
    {
        public string Name => "test-selector";

        public string Summary => "a test's";

        public (string Transport, string Reason) Choose(int rankA, int rankB, IClusterTopology topology, IReadOnlyList<TransportAvailability> available) => ("memory", "a test's");
    }

    private sealed class TestFailures : IFailureModel
    {
        public string Name => "test-failures";

        public string Summary => "a test's";

        public IReadOnlyList<Fault> Draw(long step, IClusterTopology topology, Random seeded) => [];
    }

    private static void ClusterRegistries(Device device)
    {
        if (!FirstRun(nameof(ClusterRegistries)))
        {
            return;
        }

        void Registry<T>(string registry, Func<string, T> get, Func<string, T?> find, Action<T> register, Func<string, bool> unregister, Func<string, string?> origin, T entry, string name)
            where T : class
        {
            ThrowsNaming<NotSupportedException>(() => get("no-such-" + registry), $"{registry}.Get of an unknown name", "is not registered", registry + ".Register");
            register(entry);
            Check(ReferenceEquals(find(name), entry) && ReferenceEquals(get(name.ToUpperInvariant()), entry), $"{registry}: the app's entry is found, ignoring case");
            Check(origin(name) is { } o && o != Overrides.Library, $"{registry}: the entry's origin is the app: {origin(name)}");
            Check(unregister(name) && find(name) is null, $"{registry}: Unregister removes the app's entry");
        }

        Registry("Transports", Transports.Get, Transports.Find, Transports.Register, Transports.Unregister, Transports.Origin, (ITransportProvider)new TestTransportProvider(), "test-transport");
        Registry("Collectives", Collectives.Get, Collectives.Find, Collectives.Register, Collectives.Unregister, Collectives.Origin, (ICollectivesProvider)new TestCollectivesProvider(), "test-collectives");
        Registry("TransportSelectors", TransportSelectors.Get, TransportSelectors.Find, TransportSelectors.Register, TransportSelectors.Unregister, TransportSelectors.Origin, (ITransportSelector)new TestSelector(), "test-selector");
        Registry("FailureModels", FailureModels.Get, FailureModels.Find, FailureModels.Register, FailureModels.Unregister, FailureModels.Origin, (IFailureModel)new TestFailures(), "test-failures");
        Registry("LinkProfiles", LinkProfiles.Resolve, LinkProfiles.Find, LinkProfiles.Register, LinkProfiles.Unregister, LinkProfiles.Origin, (ILinkProfile)new LinkProfile("test-link", 1e9, TimeSpan.FromMicroseconds(3)), "test-link");
        ThrowsNaming<NotSupportedException>(() => LinkProfiles.Resolve("no-such-link-10g"), "a rate suffix on an unknown name", "not registered", "-10g");
    }

    // A profile whose time shrinks as the bytes grow: the suite must fail it.
    private sealed class ShrinkingProfile : ILinkProfile
    {
        public string Name => "shrinking";

        public double BandwidthBytesPerSecond => 1e9;

        public TimeSpan Latency => TimeSpan.FromMilliseconds(1);

        public TimeSpan Jitter => TimeSpan.Zero;

        public double LossRate => 0;

        public double FailureRate => 0;

        public TimeSpan Predict(long bytes, Random seeded) => bytes == 0 ? Latency : TimeSpan.FromTicks(Math.Max(1, Latency.Ticks - (bytes / 1000)));

        public ILinkProfile With(double? bandwidthBytesPerSecond = null, TimeSpan? latency = null, TimeSpan? jitter = null, double? lossRate = null, double? failureRate = null, string? name = null) => this;
    }

    private static void ClusterLinkProfiles(Device device)
    {
        if (!FirstRun(nameof(ClusterLinkProfiles)))
        {
            return;
        }

        Check(LinkProfiles.TryParseRate("10g", out double ten) && ten == 1.25e9 && LinkProfiles.TryParseRate("2.5G", out double twoHalf) && twoHalf == 312.5e6
              && LinkProfiles.TryParseRate("300m", out double wifi) && wifi == 37.5e6, "rates in bits a second, powers of 1,000");
        Check(!LinkProfiles.TryParseRate("10x", out _) && !LinkProfiles.TryParseRate("g", out _) && !LinkProfiles.TryParseRate("-1g", out _), "a rate that is not one is refused");
        using (ClusterTestProfiles())
        {
            var resolved = LinkProfiles.Resolve("direct-link-10g");
            Check(resolved is { Name: "direct-link-10g", BandwidthBytesPerSecond: 1.25e9 } && resolved.Latency == TimeSpan.FromMicroseconds(40),
                $"direct-link-10g is direct-link at 10 Gb/s, its latency kept: {resolved.Name} {resolved.BandwidthBytesPerSecond} {resolved.Latency}");
            Check(ReferenceEquals(LinkProfiles.Resolve("ethernet-1g"), LinkProfiles.Find("ethernet-1g")), "a registered name resolves to itself");
            Conformance.CheckLinkProfile(LinkProfiles.Resolve("ethernet-1g")).ThrowIfFailed();
        }

        Conformance.CheckLinkProfile(new LinkProfile("unmeasured", 0, TimeSpan.FromMicroseconds(10))).ThrowIfFailed();
        var broken = Conformance.CheckLinkProfile(new ShrinkingProfile());
        Check(!broken.Passed && broken.Failures.Any(f => f.Check == "the time grows with the bytes"), $"the suite fails a shrinking profile:\n{broken}");
        ThrowsNaming<ArgumentException>(() => _ = new LinkProfile("bad", -1, TimeSpan.Zero), "a negative bandwidth", "bad");
        ThrowsNaming<InvalidOperationException>(() => new LinkProfile("unknown-rate", 0, TimeSpan.Zero).Predict(10, new Random(1)), "a profile without a rate", "unknown-rate-10g");
    }

    private const string ClusterLanJson = """
        {
          // three machines over the home switch; the first one's card split in two
          "format": "idrak-topology/1",
          "nodes": [
            { "name": "desk", "address": "192.168.1.10:29500",
              "gpus": [ { "device": "cuda:0", "memory": "8GB" }, { "device": "cuda:0", "memory": "8GiB" }, ] },
            { "name": "mini", "gpus": [ { "device": "cuda:0", "memory": "8 GB" } ] },
            { "name": "old", "gpus": [ { "device": "cuda:0", "memory": "6GB" } ] },
          ],
          "links": {
            "intra-node": "pcie",
            "inter-node": "ethernet-1g",   /* the switch */
            "overrides": [ { "between": ["desk", "mini"], "type": "direct-link-10g" } ]
          },
          "simulation": { "seed": 7, "failures": "none" }
        }
        """;

    private static void ClusterTopologyFile(Device device)
    {
        if (!FirstRun(nameof(ClusterTopologyFile)))
        {
            return;
        }

        Check(ClusterTopology.ParseMemory("8GB") == 8L << 30 && ClusterTopology.ParseMemory("8GiB") == 8L << 30 && ClusterTopology.ParseMemory("1.5gb") == 3L << 29
              && ClusterTopology.ParseMemory("512MiB") == 512L << 20 && ClusterTopology.ParseMemory("100") == 100 && ClusterTopology.ParseMemory(" 16 GB ") == 16L << 30
              && ClusterTopology.ParseMemory("1TB") == 1L << 40 && ClusterTopology.ParseMemory("64K") == 64L << 10, "memory sizes, powers of 1,024 whatever the spelling");
        foreach (string bad in new[] { "8XB", "GB", "-1GB", "8GiBs", "" })
        {
            ThrowsNaming<FormatException>(() => ClusterTopology.ParseMemory(bad), $"memory '{bad}'", "not a memory size");
        }

        Check(ClusterTopology.FormatMemory(8L << 30) == "8GB" && ClusterTopology.FormatMemory(512L << 20) == "512MB" && ClusterTopology.FormatMemory(100) == "100B"
              && ClusterTopology.FormatMemory(1536L << 20) == "1536MB" && ClusterTopology.FormatMemory(0) == "0B", "sizes written in the largest exact unit");

        using (ClusterTestProfiles())
        {
            var lan = ClusterTopology.Parse(ClusterLanJson);
            Check(lan.WorldSize == 4 && lan.Nodes.Count == 3 && lan.Place(1) == (0, 1) && lan.Place(2) == (1, 0) && lan.Place(3) == (2, 0) && lan.Ranks(0).SequenceEqual([0, 1]),
                "ranks in file order: desk 0 and 1, mini 2, old 3");
            Check(lan.Nodes[0].Devices.All(d => d.MemoryBytes == 8L << 30) && lan.Nodes[1].Devices[0].MemoryBytes == 8L << 30 && lan.Nodes[2].Devices[0].MemoryBytes == 6L << 30,
                "8GB, 8GiB and 8 GB are one size; 6GB");
            Check(lan.Link(0, 1).Name == ClusterTopology.SameDevice && lan.Link(1, 0).Name == ClusterTopology.SameDevice, "two ranks on desk's card share same-device");
            Check(lan.Link(0, 2) is { Name: "direct-link-10g", BandwidthBytesPerSecond: 1.25e9 } && lan.Link(1, 2).Name == "direct-link-10g", "desk and mini use their override");
            Check(lan.Link(2, 3).Name == "ethernet-1g" && lan.Link(0, 3).Name == "ethernet-1g", "the others use the inter-node link");
            Check(lan.Coordinator == "192.168.1.10:29500" && lan.NodeOf("DESK") == 0 && lan.NodeOf("Old") == 2 && lan.NodeOf("laptop") is null, "the first node's address coordinates; nodes found by host name");
            Check(lan.Simulation == new ClusterSimulationSpec(7, "none"), "the simulation block");
            Conformance.CheckTopology(lan).ThrowIfFailed();

            var one = ClusterTopology.Parse("""{ "nodes": [ { "name": "solo", "gpus": [ { "device": "cpu" } ] } ] }""");
            Check(one.WorldSize == 1 && one.Coordinator is null && one.Simulation == new ClusterSimulationSpec(1, "none"), "one device needs no links");
            var mixed = ClusterTopology.Create([new ClusterNodeSpec("box", null, [new ClusterDeviceSpec("cuda:0", null), new ClusterDeviceSpec("cuda:1", null), new ClusterDeviceSpec("cpu", 4L << 30)])], "pcie", null);
            Check(mixed.Link(0, 1).Name == "pcie" && mixed.Link(1, 2).Name == "pcie", "different devices of one node use the intra-node link");
            Conformance.CheckTopology(mixed).ThrowIfFailed();

            string Swap(string from, string to) => ClusterLanJson.Replace(from, to, StringComparison.Ordinal);
            foreach (var (json, what, expected) in new[]
                     {
                         (Swap("\"gpus\": [ { \"device\": \"cuda:0\", \"memory\": \"6GB\" } ]", "\"gpu\": []"), "an unknown key", "unknown key \"gpu\""),
                         (Swap("\"name\": \"old\"", "\"name\": \"MINI\""), "two nodes named alike", "named 'MINI'"),
                         (Swap("\"inter-node\": \"ethernet-1g\",", ""), "two nodes without an inter-node link", "inter-node"),
                         (Swap("\"between\": [\"desk\", \"mini\"]", "\"between\": [\"desk\", \"laptop\"]"), "an override naming no node", "laptop"),
                         (Swap("idrak-topology/1", "idrak-topology/2"), "a newer format", "newer"),
                         (Swap("\"memory\": \"6GB\"", "\"memory\": \"6QB\""), "a bad memory size", "not a memory size"),
                         (Swap("\"nodes\"", "\"machines\""), "no nodes", "unknown key \"machines\""),
                     })
            {
                ThrowsNaming<InvalidDataException>(() => ClusterTopology.Parse(json), what, expected);
            }

            ThrowsNaming<InvalidDataException>(() => ClusterTopology.Create([new ClusterNodeSpec("box", null, [new ClusterDeviceSpec("cuda:0", null), new ClusterDeviceSpec("cuda:1", null)])], null, null),
                "two devices of a node without an intra-node link", "intra-node");
            ThrowsNaming<NotSupportedException>(() => ClusterTopology.Parse(Swap("\"inter-node\": \"ethernet-1g\"", "\"inter-node\": \"warp-drive\"")), "an unregistered link", "warp-drive", "not registered");
            ThrowsNaming<ArgumentOutOfRangeException>(() => lan.Place(4), "a rank outside the group", "0 to 3");
            ThrowsNaming<ArgumentException>(() => lan.Link(2, 2), "a rank's link to itself", "itself");

            string path = Path.Combine(Path.GetTempPath(), $"idrak-topology-{Guid.NewGuid():N}.json");
            File.WriteAllText(path, lan.ToJson());
            try
            {
                var loaded = ClusterTopology.Load(path);
                Check(loaded.ToJson() == lan.ToJson() && loaded.WorldSize == 4, "the written file loads back the same");
                File.WriteAllText(path, Swap("\"seed\": 7", "\"seed\": \"seven\""));
                ThrowsNaming<InvalidDataException>(() => ClusterTopology.Load(path), "a bad seed in a file", path, "seed");
            }
            finally
            {
                File.Delete(path);
            }
        }
    }

    private static void ClusterTransportSuite(Device device)
    {
        if (!FirstRun(nameof(ClusterTransportSuite)))
        {
            return;
        }

        var options = new ContractCheckOptions { RandomCases = 2 };
        Conformance.CheckTransport(new TransportUnderTest("memory", (w, t) => MemoryTransport.Connect(w, t)), options).ThrowIfFailed();
        var broken = Conformance.CheckTransport(new TransportUnderTest("corrupting", (w, t) => MemoryTransport.Connect(w, t, corrupt: true)), options);
        Check(!broken.Passed && broken.Failures.Any(f => f.Check == "every message arrives whole and in order, under its tag"), $"the suite fails a corrupting transport:\n{broken}");
    }

    private static void ClusterCollectivesSuite(Device device)
    {
        Conformance.CheckCollectives(new CollectivesUnderTest("naive-reference", (w, t) => NaiveReferenceCollectives.Connect(w, t, device)),
            new ContractCheckOptions { RandomCases = 2 }).ThrowIfFailed();
        if (FirstRun(nameof(ClusterCollectivesSuite)))
        {
            var broken = Conformance.CheckCollectives(new CollectivesUnderTest("not-reducing", (w, t) => NaiveReferenceCollectives.Connect(w, t, device, skipReduce: true)),
                new ContractCheckOptions { RandomCases = 0 });
            Check(!broken.Passed && broken.Failures.Any(f => f.Check.StartsWith("all-reduce", StringComparison.Ordinal)), $"the suite fails collectives that do not reduce:\n{broken}");
        }
    }

    private static void ClusterTelemetry(Device device)
    {
        if (!FirstRun(nameof(ClusterTelemetry)))
        {
            return;
        }

        Check((TelemetryLevel.All & TelemetryLevel.Network) != 0, "All includes Network");
        var console = new StringWriter();
        var metrics = new MetricsRecorder(TelemetryLevel.Network);
        var training = new MetricsRecorder(TelemetryLevel.Training);
        string path = Path.Combine(Path.GetTempPath(), $"idrak-network-{Guid.NewGuid():N}.jsonl");
        try
        {
            using (Telemetry.Subscribe(training))
            {
                Check(!Telemetry.IsEnabled(TelemetryLevel.Network), "nothing listens at the network level");
                using (Telemetry.Subscribe(new ConsoleLogger(TelemetryLevel.Network, output: console)))
                using (Telemetry.Subscribe(metrics))
                {
                    var json = new JsonLinesLogger(path, TelemetryLevel.Network);
                    using (Telemetry.Subscribe(json))
                    {
                        Check(Telemetry.IsEnabled(TelemetryLevel.Network), "the network level is on while a hook asks for it");
                        Telemetry.Transfer(new TransferCompleted("sim", 0, 1, 4, 4096, TimeSpan.FromMilliseconds(0.5), TimeSpan.FromMilliseconds(33)));
                        Telemetry.Collective(new CollectiveCompleted("ring", "all-reduce", 0, 4, 1 << 20, 1 << 20, TimeSpan.FromMilliseconds(4), TimeSpan.FromMilliseconds(17),
                            TimeSpan.FromMilliseconds(2), 3));
                        Telemetry.TransportChosen(new TransportChosen(0, 1, "tcp", "fastest available"));
                        Telemetry.RankState(new RankStateChanged(3, RankState.Slow, 12, "blocked others 40% of the step"));
                        Telemetry.Fault(new FaultInjected(7, 3, FaultKind.SlowWorker, [3], TimeSpan.FromMilliseconds(250), 2.5));
                        Telemetry.Calibrated(new ProfileCalibrated("ethernet-1g", 125e6, 112e6, TimeSpan.FromMicroseconds(80), TimeSpan.FromMicroseconds(95), 1.2e-4, 40));
                    }

                    json.Dispose();
                }
            }

            string text = console.ToString();
            foreach (string expected in new[] { "Transfer sim 0->1", "simulated 33.000 ms", "Collective ring all-reduce rank 0/4", "slowest rank 3", "Ranks 0 and 1 use tcp",
                         "Rank 3 slow at step 12", "replay with --seed 7 --step 3", "Calibrated ethernet-1g: 125.0 -> 112.0 MB/s" })
            {
                Check(text.Contains(expected, StringComparison.Ordinal), $"the console prints '{expected}':\n{text}");
            }

            Check(metrics.Collectives is [{ Collective: "ring", SlowestRank: 3 }] && training.Collectives.Count == 0, "the metrics recorder keeps collectives at the network level only");
            var events = File.ReadAllLines(path).Select(l => (string)JsonNode.Parse(l)!["event"]!).ToArray();
            Check(events.SequenceEqual(["transfer", "collective", "transport_chosen", "rank_state", "fault", "profile_calibrated"]), $"the JSON lines: {string.Join(", ", events)}");
            var fault = JsonNode.Parse(File.ReadAllLines(path)[4])!;
            Check((int)fault["seed"]! == 7 && (long)fault["step"]! == 3 && (string)fault["kind"]! == "SlowWorker" && fault["ranks"]!.AsArray().Count == 1, $"a fault's replay key: {fault.ToJsonString()}");
        }
        finally
        {
            File.Delete(path);
        }
    }

    private static void ClusterDdpFixture(Device device)
    {
        if (!FirstRun(nameof(ClusterDdpFixture)))
        {
            return;
        }

        // A stale fixture fails here first: every file as the manifest hashes it.
        var manifest = JsonNode.Parse(File.ReadAllText(ClusterData("manifest.json")))!;
        foreach (var (file, entry) in manifest["files"]!.AsObject())
        {
            string hash = Convert.ToHexStringLower(SHA256.HashData(File.ReadAllBytes(ClusterData(file))));
            Check(hash == (string)entry!["sha256"]!, $"{file} differs from the manifest: rerun tools/cluster/make_tiny_ddp.py");
        }

        var ddp = JsonNode.Parse(File.ReadAllText(ClusterData("ddp/ddp.json")))!;
        var single = JsonNode.Parse(File.ReadAllText(ClusterData("single/single.json")))!;
        var windows = ddp["rank_windows"]!.AsArray().SelectMany(w => w!.AsArray().Select(i => (int)i!)).Order().ToArray();
        Check(windows.SequenceEqual(Enumerable.Range(0, 8)), $"the two ranks take the eight windows once: {string.Join(",", windows)}");
        foreach (string run in new[] { "sgd", "adamw", "adamw_folded" })
        {
            var losses = ddp[run]!["losses"]!.AsArray().Select(v => (double)v!).ToArray();
            var reference = single[run]!["losses"]!.AsArray().Select(v => (double)v!).ToArray();
            Check(losses.Length == 3 && losses.Zip(reference).All(p => Math.Abs(p.First - p.Second) <= 1e-5), $"{run}: ddp losses {string.Join(", ", losses)} against single {string.Join(", ", reference)}");
            if (ddp[run]!["rank_losses"] is JsonArray ranks)
            {
                var mean = Enumerable.Range(0, 3).Select(s => ranks.Average(r => (double)r!.AsArray()[s]!)).ToArray();
                Check(mean.Zip(losses).All(p => Math.Abs(p.First - p.Second) <= 1e-6), $"{run}: the losses are the ranks' mean");
            }
        }

        static double Largest(Dictionary<string, (int[] Shape, float[] Values)> a, Dictionary<string, (int[] Shape, float[] Values)> b, string what)
        {
            Check(a.Keys.Order().SequenceEqual(b.Keys.Order()), $"{what}: the tensors' names differ");
            double largest = 0;
            foreach (var (name, (shape, values)) in a)
            {
                Check(shape.SequenceEqual(b[name].Shape), $"{what}: {name}'s shape differs");
                for (int i = 0; i < values.Length; i++)
                {
                    largest = Math.Max(largest, Math.Abs(values[i] - b[name].Values[i]));
                }
            }

            return largest;
        }

        foreach (string file in new[] { "grads-step1.safetensors", "weights-step3.safetensors", "adamw/weights-step3.safetensors", "adamw/state-step3.safetensors", "adamw-folded/weights-step3.safetensors" })
        {
            double d = Largest(ReadTensors(ClusterData("ddp/" + file)), ReadTensors(ClusterData("single/" + file)), file);
            Check(d <= 1e-5, $"{file}: ddp and single differ by {d:G3}");
        }

        // The bf16 wire moves the results, by about what its 8 bits of mantissa allow (the README's measured tolerance).
        double grads = Largest(ReadTensors(ClusterData("bf16/grads-step1.safetensors")), ReadTensors(ClusterData("ddp/grads-step1.safetensors")), "bf16 grads");
        double weights = Largest(ReadTensors(ClusterData("bf16/weights-step3.safetensors")), ReadTensors(ClusterData("ddp/weights-step3.safetensors")), "bf16 weights");
        Check(grads is > 0 and <= 1e-3 && weights is > 0 and <= 5e-4, $"bf16 against ddp: gradients {grads:G3}, weights {weights:G3}");
        Check(File.ReadAllText(ClusterData("README.md"), Encoding.UTF8).Contains("How the C# tests read it", StringComparison.Ordinal), "the fixture's README says how the tests read it");
    }
}
