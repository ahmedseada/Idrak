# Plan 23: Idrak.Cluster and Idrak.Network (a cluster simulated on one GPU; real transports measured on real links)

**Status:** planned 2026-10-18 from the owner's spec (plan 23, draft 2) against `abstraction` at fec56e1; nothing built.
**Supersedes plan 6** (`6-network.md`): that plan put the simulator inside `Idrak.Abstraction` as a `"local"` transport and
built data parallel straight on TCP. This plan builds the simulator first, as its own package native to Idrak, and every
distributed feature is proven on it before any real link is used. Plan 6 stays in the repository as history only; nothing
below depends on it.

**Goal.** Simulate multi-GPU and multi-machine clusters on **one local GPU** so every piece of distributed functionality
and architecture (transports, collectives, topology-aware reduction, sharding, pipelines, failure handling, resume) is
verified locally, deterministically, with a virtual clock that **predicts** the step time of clusters the owner does not
own. Real hardware (the owner's home LAN of three machines; rented boxes) is used **only to measure speed and calibrate
the profiles**. Same topology file, same launcher, same training code on every target; only the transports change.

**Rules that hold for every phase** (`CLAUDE.md`, `docs/optimization.md`):

- **Card-agnostic.** Nothing in code, comments, tests or docs names a card, a vendor path, a link speed or a memory size
  as a fact of the library. Every number of that kind is **declared by the user in a topology file** (a fake GPU's cap,
  a link's bandwidth) or **measured at run time** (the physical device's memory, a link's real throughput). There are
  no preset topologies: `idrak cluster init` builds the file interactively from the devices it finds and the answers
  the user gives, so no card, link or size is ever a library resource.
- **Abstraction (decision 10 of plan 10).** Every new behaviour is a contract plus a `SlotTable` registry. A contract two
  packages use lives in `Idrak.Abstraction` (`Idrak.Abstraction.Cluster`); one only `Idrak.Network` uses lives in
  `Idrak.Network.Abstractions`; one only `Idrak.Cluster` uses lives in `Idrak.Cluster.Abstractions`. `Idrak.Cluster`
  references `Idrak.Abstraction` and implements its contracts directly (the owner's call, 2026-10-18: no duplicate
  types and no bridge; when the simulator is ever extracted into its own repository, the contracts and types it uses
  move with it). The inventory test and the public API guard (`api/*.txt`) are regenerated when a public surface
  changes; `api/Idrak.Network.txt` and `api/Idrak.Cluster.txt` join the guard.
- **No native libraries** (plans/README.md, "The rule every plan keeps"). Sockets, memory-mapped files, named semaphores
  and the CUDA driver's peer copies are what .NET and the installed driver give; NCCL, MPI and RDMA verbs are prebuilt
  native libraries, so `rdma` is a **registered name whose library default is absent** (it fails with "not registered"
  and the `-P` hint) until a plug-in provides it.
- **Speed and memory first.** Bytes on the wire are parsed from `ReadOnlySpan<byte>` with `BinaryPrimitives` out of
  rented buffers returned in `finally` (rules 7-14, 25, 74); gradients are reduced **in place, slice by slice**, never
  copied whole (rule 77: device memory does not grow across steps); every wait on a link is a telemetry number (78),
  cold and warm apart (76); ranks, chunks and both ring directions run at once and the first step reports what the
  spread paid (79); every sum is made in an order fixed by the algorithm and the sizes (86), never by timing, so every
  rank holds bit-identical values, a rerun gives bit-identical values, and the simulator and a real transport running
  the same algorithm agree bit for bit (71). Two algorithms (naive and ring) order a sum differently, so they agree
  within float rounding, exactly on whole numbers (found while writing C0's suite; the plan had claimed bit equality).
- **Nothing fixed.** Every size, chunk, bucket, timeout, retry count, port, address, wire type, sync interval, placement
  and the transport choice is a library option (`ClusterOptions`, `SimulationOptions`) with an
  environment variable, listed in `src/Idrak.Cli/Shared/EnvironmentVariables.cs` under a new group `"Cluster"` (the
  existing test that scans sources for names read and fails when one is unlisted covers them).
- **Tests.** Only `IDRAK_DEVICES=cpu IDRAK_FILTER="cluster:"` (and the `"cluster: cli"` subset) here; multi-process tests
  start ranks as child processes over loopback; the owner runs CUDA, Vulkan, the three-machine LAN and the rental from
  the command lists at the end of each phase. Never the full suite.
- **The fix workflow.** Each phase: fix every error first, update the tests the fixes touch, build once (only the
  projects that changed), run the phase's filter once with `--no-build`, commit and push on `abstraction`, hand the owner
  the command list. At most two agents at a time.
- **References measure, they never limit** (the owner, 2026-10-18). The PyTorch DDP fixture and any other outside
  reference exist only to check a result Idrak also produces; they bound nothing Idrak builds. Where Idrak.Cluster or
  Idrak.Network goes beyond what the reference tool can do (the virtual clock, fake devices with caps, seeded replay,
  the hierarchical reduction's predicted win, calibration, a transport or a collective PyTorch has no equal of), the
  feature is designed from this plan and proven by Idrak's own tests (the one-process equivalence, the conformance
  suites, the simulator's determinism), and no reference's absence is a reason to leave it out or shrink it.
- **Dated runs** (simulated predictions, LAN measurements, rental measurements, calibration deltas) are appended to the
  "Runs" section at the end of this file, newest last.

## What Idrak has, and what is missing

| Part | Today (at fec56e1) | Missing |
|---|---|---|
| Devices as slots | `DeviceProvider` (`src/Idrak.Abstraction/Devices/DeviceProviders.cs`: `Kind`, `Count`, `Create(ordinal)`, `DeviceUuid`) in the unguarded `DeviceProviders` table; `Device.Get`/`Parse`; `Idrak.Gpu.LibraryRegistrations.RegisterFor` registers CUDA, Vulkan and HIP through `LibraryDefaults.Ensure` (`LibraryDefaults.cs:29-30` lists the first-party assemblies) | a **fake device**: one more backend instance of a physical device's kind with its own memory cap |
| Memory caps | `MemoryAccountant(Func<long?> limit, string deviceName)` (`Backend.cs:2422`) accounts every backend's bytes against a limit function; `ComputeResources.GpuMemoryLimit` (`ComputeResources.cs:70`, `IDRAK_GPU_MEMORY_LIMIT_MB`) is the one limit today, the same for every GPU; `OffloadToHostMemory` (`:131`, `IDRAK_OFFLOAD`) spills new tensors to pinned host memory at the limit | a **per-device** limit (a fake GPU's hard cap) and offload per device |
| Copies between devices | `Backend.Upload`/`Download`/`DownloadRange`/`Copy`, `CreateHostStaging(slots, slotFloats)` with pinned slots where the device has them | a device-to-device copy (today: down to the host, then up); the CUDA peer path |
| Gradient buffers and the step | `Optimizer.ZeroGrad`/`Step`/`ClipAndStep(maxNorm)` (`src/Idrak.Abstraction/Training/Optimizer.cs`), `FusedAdamW`; `HostOptimizer` already moves gradients to the CPU in staged pieces | any exchange of gradients between ranks; the hook is the gap between the last `Backward` and `ClipAndStep` |
| The fine-tuning and pretraining steps | `FineTuner.StepRunner.RunStep` (`src/Idrak.Nlp/FineTuning.cs:1602` calls `Update` :1708, which calls `ClipAndStep` :1719; the queued `Backward` at :1752); `Pretrainer.Train` (`Pretraining.cs`: seeded source order, `SaveCheckpoint`, resume); `Trainer.Fit` (`src/Idrak/Training/Trainer.cs:206-209`: `Backward` then `Optimizer.Step`) | a rank's share of the draw, rank 0's checkpoint, resume at another world size, the all-reduce in the gap |
| Telemetry | `TelemetryLevel` flags (`Diagnostics/Telemetry.cs:11`), `IsEnabled` (one static read and an AND), `BatchCompleted(DataTime, ComputeTime)`, `HostFallback(Device, Operation)` (`Events.cs:262`), `TelemetryJson` into the run's `log.jsonl`, `idrak trace` and `idrak profile` | a `Network` level and events for a transfer, a collective and its wait, with simulated and wall time |
| Persisted measurements | `CpuTuning` keeps measured cut-overs in `cpu/tuning.tsv` under `IDRAK_CACHE` (or `~/.cache/idrak`), off with `IDRAK_TUNING_CACHE=0` (`Devices/Cpu/CpuTuning.cs:58-76`) | the calibrated link profiles, kept the same way |
| Conformance kit | `ContractSuite<T>` (`src/Idrak.Abstraction.Testing/ContractSuite.cs`: `FixedCases`, `RandomCase`, `Run`), `Conformance.Check`, `Stress.Run` (memory watched on a device), `Regression.Replay`; sixteen suites in `Contracts/` | suites for a transport, a collective, a link profile and a topology |
| The CLI | `Command` subclasses in `*Commands.All` lists; `train`, `pretrain`, `tune`, `resume`, `distill`, `trace`, `profile`, `bench`, `devices` exist; `cluster` and `transports` are free names | `idrak cluster …`, `idrak transports`, `--cluster` on the training commands |
| Anything distributed | nothing: the only "Network" types are the neural-network builder (`src/Idrak/Layers/NetworkBuilder.cs`) | everything below |

## Package layout and dependencies

```
Idrak.Abstraction                            (contracts + registries: Idrak.Abstraction.Cluster; the telemetry events)
   ├── Idrak.Cluster                         (the simulator: the "sim" transport and "sim" devices, link profiles, the virtual clock,
   │                                          seeded faults and replay, the localhost launcher and throttle, calibration, the "cluster" probe)
   ├── Idrak.Network                         (every real transport, the collectives, the selector, the training hooks, the "network" probe)
   │        └── Idrak.Cli                    (idrak cluster …, idrak transports, --cluster on train/pretrain/tune; references both)
   └── Idrak.Abstraction.Testing             (TransportSuite, CollectivesSuite, LinkProfileSuite, TopologySuite)
Idrak.Nlp, Idrak (core Training)             (take an ICollectives through their options; with none they run as today)
```

- **`src/Idrak.Cluster`** references `Idrak.Abstraction` only, as `Idrak.Gpu` does, and implements its contracts
  directly: `SimTransport : ITransport`, `SimulatedDeviceProvider : DeviceProvider` (kind `"sim"`), the built-in
  `ILinkProfile`s and `IFailureModel`s, the topology builder, the `"cluster"` `IProfileProbe`, and it publishes `Telemetry`
  events itself. It holds the virtual clock, the link models, the fault injector, the replayable event log, the
  in-process cluster, the localhost process launcher with its throttle, and the calibration arithmetic. **It is native
  to Idrak and uses everything the library already has** (the owner's call, 2026-10-18): `Device` and `DeviceProvider`
  for the fake devices, `Backend` and `MemoryAccountant` for their caps, `Storage` and `CreateHostStaging` where a
  simulated transfer carries device memory, `ComputeResources` for its limits, `SlotTable` for its registries,
  `Telemetry` and the `IProfileProbe` contract for its numbers, the tuning-cache format for its calibrated profiles,
  `SyncProgress` and the CLI's progress line for its live output, the conformance kit for its tests; nothing is
  reimplemented here to keep the package detachable. **Extraction later** (working name then: Mithal or Sarab): the
  package moves as a whole and `Idrak.Abstraction.Cluster`'s contracts and whatever Abstraction types it uses move or
  are copied with it on that day, not before. `Idrak.Cluster.LibraryRegistrations.RegisterFor` registers its
  implementations; `"Idrak.Cluster"` joins `LibraryDefaults.Assemblies` and the `DynamicDependency` list.
- **`src/Idrak.Network`** references `Idrak.Abstraction` only. It holds every real transport, the collectives (naive,
  ring, hierarchical) over any `ITransport` (the simulator's included), the transport selector, the `"network"` probe
  and the training hooks' implementation. `Idrak.Network.LibraryRegistrations.RegisterFor` registers all of it;
  `"Idrak.Network"` joins `LibraryDefaults.Assemblies` and its `DynamicDependency` list, so an app that ships the
  package has the transports the first time `Transports` is used, and one that does not gets
  `Transport 'tcp' is not registered (registered: sim); add it with Transports.Register` and the CLI's `-P` hint.
- **The two packages do not reference each other.** The simulator needs no collective (it carries bytes); the
  collectives need no simulator (they take any `ITransport`). The tests and the CLI reference both, which is where a
  simulated run of a real collective happens. Either package can be shipped, or extracted, without the other.
- **`Idrak` does not reference either** (unlike `Idrak.Gpu`): the trainers take the contracts from Abstraction; apps
  that want a cluster add the packages they need. `Idrak.Cli` references both.

## Contracts (`src/Idrak.Abstraction/Cluster/`, namespace `Idrak.Abstraction.Cluster`)

```csharp
public enum ReduceOp { Sum, Max, Min }
public enum WireType { Float32, BFloat16 }         // how values travel; sums are always made in float32, in a fixed order

/// Point-to-point bytes between ranks. Sync core over spans; the async shell only awaits the socket.
public interface ITransport : IDisposable
{
    string Name { get; }                             // "sim", "shared-memory", "pcie-host", "nvlink", "direct-link", "tcp", "udp", "rdma"
    TransportCapabilities Capabilities { get; }      // see below
    int Rank { get; }
    int WorldSize { get; }
    void Send(int to, ReadOnlySpan<byte> bytes, int tag);
    int Receive(int from, Span<byte> bytes, int tag);         // blocks up to ClusterOptions.Timeout; names the rank on timeout
    ValueTask SendAsync(int to, ReadOnlyMemory<byte> bytes, int tag, CancellationToken ct = default);
    ValueTask<int> ReceiveAsync(int from, Memory<byte> bytes, int tag, CancellationToken ct = default);
    void Barrier();
}

public readonly record struct TransportCapabilities(
    double BandwidthBytesPerSecond,   // the transport's own estimate, replaced by the measured value once the bench ran
    TimeSpan Latency,
    bool ZeroCopy,                    // the bytes go from the caller's buffer without a copy
    bool GpuDirect,                   // device memory is sent without a host round trip (peer copies)
    bool Reliable,                    // delivery is guaranteed (false: udp)
    bool CrossProcess, bool CrossMachine);

/// Collectives over device buffers; built on ITransport by Idrak.Network; the registry names the algorithm.
public interface ICollectives : IDisposable
{
    string Name { get; }                             // "naive", "ring", "hierarchical"
    int Rank { get; }
    int WorldSize { get; }
    Device Device { get; }                           // this rank's device
    void AllReduce(Storage buffer, int count, ReduceOp op, WireType wire = WireType.Float32);   // in place; every rank ends bit-identical
    void AllGather(Storage input, int count, Storage output);                                   // output: world · count, rank order
    void ReduceScatter(Storage input, int count, Storage output, ReduceOp op);                  // input: world · count; output: count
    void Broadcast(Storage buffer, int count, int root);
    void Barrier();
}

public interface ILinkProfile
{
    string Name { get; }                             // "same-device", "nvlink", "pcie", "infiniband", "roce", "thunderbolt", "direct-link", "shared-memory", "ethernet-1g", "ethernet-2.5g", "ethernet-10g", "ethernet-100g", "wifi"
    double BandwidthBytesPerSecond { get; }
    TimeSpan Latency { get; }
    TimeSpan Jitter { get; }                         // the width of a seeded uniform draw added to each transfer
    double LossRate { get; }                         // per message; a lost message is resent after Latency
    double FailureRate { get; }                      // per step; drawn by the failure model
    TimeSpan Predict(long bytes, Random seeded);     // bytes / bandwidth + latency + jitter(seeded)
    ILinkProfile With(double? bandwidth = null, TimeSpan? latency = null, ...);   // calibration writes a corrected copy
}

public interface IClusterTopology
{
    IReadOnlyList<ClusterNodeSpec> Nodes { get; }                 // name (the host's), optional address, the GPUs (device string, declared cap)
    ILinkProfile Link(int rankA, int rankB);                      // intra-node, inter-node, or an override for the pair
    int WorldSize { get; }                                        // the device count over every node: never typed again
    (int Node, int Local) Place(int rank);                        // rank → (node index, device index in that node)
    int? NodeOf(string hostName);                                 // the node whose name is this host's, so a process finds itself
    IReadOnlyList<int> Ranks(int node);                           // the ranks a node holds, in file order
}
public sealed record ClusterNodeSpec(string Name, string? Address, IReadOnlyList<ClusterDeviceSpec> Devices);   // Address: "host:port" of the first node is the coordinator
public sealed record ClusterDeviceSpec(string Device, long? MemoryBytes);       // "cuda:0", "vulkan:0", "cpu"; null: the physical device's own

public interface ITransportSelector
{
    string Name { get; }                             // "fastest" (library default), "forced"
    string Choose(int rankA, int rankB, IClusterTopology topology, IReadOnlyList<ITransportAvailability> available);   // a Transports name; logged
}

public interface IFailureModel
{
    string Name { get; }                             // "none" (library default), "slow-worker", "dropped-worker", "link-loss", "partition", "late-join", "mixed"
    IEnumerable<Fault> Draw(int step, IClusterTopology topology, Random seeded);   // every fault carries (seed, step, kind, rank(s), duration)
}

// Every registry copies RandomTextSources (src/Idrak.Nlp/Abstractions/RandomTextData.cs): a SlotTable named after the
// class, OrdinalIgnoreCase, RegisterDefault inside LibraryDefaults.Ensure, [NoInlining] Register(T) with the calling
// assembly as Origin, Unregister, Names, Find, Get (the "is not registered (registered: …); add one with X.Register" message), Default, Origin, Describe.
public static class Transports { }          // unguarded SlotTable<string, ITransportProvider> (Availability per pair, Connect a rank); "sim" from Idrak.Cluster, the real ones from Idrak.Network
public static class Collectives { }         // unguarded SlotTable<string, ICollectivesProvider> (Create a rank's collectives over a transport); "naive", "ring", "hierarchical"
public static class LinkProfiles { }        // unguarded (a profile is data, no call to fall back from): an app's or a calibrated profile takes the built-in's place; Resolve reads a rate suffix ("direct-link-10g")
public static class TransportSelectors { }  // unguarded
public static class FailureModels { }       // unguarded
```

**`ClusterOptions`** (a `sealed record` with `init` defaults; `FromEnvironment()` over an internal
`From(Func<string, string?> variable, ClusterOptions? defaults)` as `ProfileOptions` does at `Profiling.cs:164`, so the
variable reader is a parameter and the parsing is tested without setting variables; a bad value names the variable, the
value and what was expected):

| Option | Variable | Default |
|---|---|---|
| `Rank`, `WorldSize` | `IDRAK_RANK`, `IDRAK_WORLD_SIZE` | derived from the topology file (world size: its device count; rank: this host's node when that node has one worker); the variables are read only without a file, or to check against it |
| `Node` (which node of the file this machine is) | `IDRAK_CLUSTER_NODE` | the node whose name is this host's name |
| `Coordinator` (`host:port` rank 0 listens on) | `IDRAK_COORDINATOR` | the first node's `address` in the file; else none |
| `Topology` (the file) | `IDRAK_CLUSTER_TOPOLOGY` | none: one node, the devices given |
| `Transport` (a `Transports` name, or `auto` for the selector) | `IDRAK_CLUSTER_TRANSPORT` | `auto` |
| `Selector` | `IDRAK_CLUSTER_SELECTOR` | `fastest` |
| `Collective` | `IDRAK_CLUSTER_COLLECTIVE` | `hierarchical` when the topology has more than one node, else `ring` |
| `Key` (the shared secret of the HMAC handshake; never logged, never an option on the command line) | `IDRAK_CLUSTER_KEY` (secret) | none: nothing listens off loopback without one |
| `ListenPort` (0: the OS picks, told to the coordinator) | `IDRAK_CLUSTER_PORT` | 0 |
| `Timeout`, `Retries` | `IDRAK_CLUSTER_TIMEOUT` (s), `IDRAK_CLUSTER_RETRIES` | 120, 3 |
| `ChunkBytes` (null: measured at first use and kept in the tuning cache), `InFlight`, `SocketBufferBytes` | `IDRAK_CLUSTER_CHUNK_BYTES`, `IDRAK_CLUSTER_IN_FLIGHT`, `IDRAK_CLUSTER_SOCKET_BUFFER` | measured, 2, the OS default |
| `BucketBytes` (gradient buckets reduced during backward; null: measured against the step) | `IDRAK_CLUSTER_BUCKET_BYTES` | measured (the spec's ~25 MB is the first probe, not a constant) |
| `Wire` (`f32`, `bf16`), `TopK` (0: off), `ErrorFeedback`, `LocalSgdEvery` (0: every step) | `IDRAK_CLUSTER_WIRE`, `IDRAK_CLUSTER_TOPK`, `IDRAK_CLUSTER_ERROR_FEEDBACK`, `IDRAK_CLUSTER_LOCAL_SGD` | `f32`, 0, on, 0 |
| `SyncWeightsEvery` (mixed device kinds round differently) | `IDRAK_CLUSTER_SYNC_WEIGHTS` | 0 (never) |
| `Shard` (`none`, `optimizer`, `gradients`, `parameters`) | `IDRAK_CLUSTER_SHARD` | `none` |
| `Placement` (`single`, `pipeline`), `MicroBatches`, `Schedule` | `IDRAK_CLUSTER_PLACEMENT`, `IDRAK_CLUSTER_MICROBATCHES`, `IDRAK_CLUSTER_SCHEDULE` | `single`, accumulation, `gpipe` |
| `SimulationOptions.Seed`, `.Mode` (`in-process`, `localhost`), `.Throttle` (apply the profile to localhost), `.FailureModel`, `.ReplayStep` | `IDRAK_CLUSTER_SEED`, `IDRAK_CLUSTER_MODE`, `IDRAK_CLUSTER_THROTTLE`, `IDRAK_CLUSTER_FAILURES`, `IDRAK_CLUSTER_REPLAY_STEP` | 1, `in-process`, on, `none`, none |
| `SimulationOptions.ClockForThroughput` (`wall`, `simulated`) | `IDRAK_CLUSTER_CLOCK` | `wall` |
| The calibration cache: one text file per link under `{cache}/cluster/`, header `idrak-cluster-profiles 1`, then `link bandwidth latency jitter residual samples` lines, ignored and rewritten on a header mismatch (the CUDA `TuningCache` format, `src/Idrak.Gpu/Cuda/TuningCache.cs`, with pure `Serialize`/`Parse` so tests run without a link); `idrak tuning show` lists it | `IDRAK_CACHE`, `IDRAK_TUNING_CACHE` (existing); `IDRAK_CLUSTER_PROFILES_FILE` overrides the file | `~/.cache/idrak/cluster/` |

**Telemetry (the owner's note: integrated wherever it helps, in both packages).** The layers already exist
(`TelemetryLevel` flags, `ITelemetryHook` with default no-op `On…` methods, `Telemetry.IsEnabled` as one static read and
an AND, `ConsoleLogger`, `MetricsRecorder`, `ChannelTelemetry`, `JsonLinesLogger`, `TelemetryJson`,
`Telemetry.Configure()`; `src/Idrak.Abstraction/Diagnostics/`). This plan adds one level and six events, and every
layer carries them:

- `TelemetryLevel.Network = 1 << 11` (the next free flag; `All` includes it; `trace --levels network`).
- Events (`Events.cs`, readonly record structs, all at `Network`), published only when `IsEnabled` (rule 78: free when
  nobody listens; the simulator's counters are kept regardless, they are the virtual clock's own state):
  `TransferCompleted(Transport, From, To, Tag, Bytes, Wall, Simulated)`;
  `CollectiveCompleted(Collective, Operation, Rank, WorldSize, BytesSent, BytesReceived, Wall, Simulated, Blocked, SlowestRank)`;
  `TransportChosen(RankA, RankB, Transport, Reason)` once per pair at start;
  `RankStateChanged(Rank, State, Step, Detail)` (joined, slow, dropped, partitioned, rejoined, finished);
  `FaultInjected(Seed, Step, Kind, Ranks, Duration)` (the replay key, so a crash's log names what to replay);
  `ProfileCalibrated(Link, BandwidthBefore, BandwidthAfter, LatencyBefore, LatencyAfter, Residual, Samples)`.
  `Telemetry.Transfer`, `.Collective`, `.TransportChosen`, `.RankState`, `.Fault`, `.Calibrated` publish them;
  `ITelemetryHook` gains the six `On…` defaults.
- **`Idrak.Cluster` publishes directly:** the simulator calls `Telemetry.Transfer`, `.Fault` and `.RankState` behind
  `IsEnabled(Network)` as every other package does, so a listener costs nothing when none is subscribed; its own
  counters (per-rank clock, per-link busy time, transfers and bytes, faults) are the virtual clock's state and are kept
  regardless, since the prediction needs them.
- **Where it shows up:** `ConsoleLogger` prints one line per collective at `Network` (bytes, wall, simulated, blocked,
  the slowest rank) and one per fault and rank-state change; `MetricsRecorder` keeps per-step network time, bytes and
  blocked time beside the losses; `ChannelTelemetry` streams them to an app's UI; `JsonLinesLogger` and
  `TelemetryJson.Write` put them in the run's `log.jsonl`, so `runs show` charts the network share of each step and
  `runs compare` lines up a simulated run against a real one; `idrak trace --levels network` prints them live;
  `idrak profile`'s probes above read the same events; the training progress line shows the network share of the step,
  the slowest rank, and the predicted time beside the wall time; `TrainingStarted.Device` names the rank's device.
- **Localhost and real clusters:** every rank keeps its own telemetry; rank 0's `log.jsonl` also receives each rank's
  per-step summary (bytes, blocked, slowest) as a small message in the barrier that follows the step, so one file
  describes the cluster; `idrak cluster calibrate --from RUN` reads the `TransferCompleted` records of that file (or a
  bench's JSON) and `ProfileCalibrated` records what it wrote.
- **Timing on devices:** a transfer's wall time is taken after `SynchronizeForTiming` as the kernel events are, so a
  queued device copy is not counted as free; `BatchCompleted` keeps its meaning (data and compute time) and is not
  reused for network time.

**Profiling (the owner's note: the profiler is integrated wherever it helps).** The profiler already exists
(`Profiler.Run(name, device, work, ProfileOptions)` → `ProfileReport`, probes in the `ProfileProbes` registry, writers
in `ProfileWriters`, `idrak profile`; `src/Idrak.Abstraction/Diagnostics/Profiling.cs`). This plan adds two probes,
each registered by its package's `LibraryRegistrations.RegisterFor(typeof(ProfileProbes))`, so every profile of a
distributed step shows network time next to kernel time without a new tool:

- `"network"` probe (`src/Idrak.Network/Diagnostics/NetworkProbe.cs`, not intrusive): per transport and per pair, the
  transfers, bytes, wall time, time blocked waiting for the slowest rank, and the collective count per step; rows named
  `transport/pair` and `collective/operation`, so `--top` ranks the pairs that cost the most.
- `"cluster"` probe (`src/Idrak.Cluster/Diagnostics/ClusterProbe.cs`, registered by `Idrak.Cluster.LibraryRegistrations`, not intrusive; only when the `sim` transport is active): the virtual
  clock per rank, each link's busy share, the critical path of the step, the faults drawn; the predicted step time as a
  row beside the measured one.

Where the profiler is used, in order of value: `idrak cluster sim … --profile` (a profile of the simulated step, cold
and warm apart, kernel rows from the existing probes beside the two new ones); `idrak profile MODEL --cluster
FILE` (the inference profile across a simulated pipeline: a token's hops as rows); `idrak cluster bench` writes its
rows through `ProfileWriters` (table, markdown, csv, json) so `calibrate --from` reads a bench's JSON as readily as a
run's `log.jsonl`; the collectives' `ChunkBytes` and `BucketBytes` measurement at first use runs under
`Profiler.Run` so `idrak tuning show` lists what was chosen and why; the "hierarchical beats ring" and "overlap shortens
the step" tests assert on `ProfileReport` rows, the same numbers the owner sees. The `"cluster"` probe lives in
`Idrak.Cluster` beside the counters it reads; the `"network"` probe in `Idrak.Network` beside the transports it times.

## The topology file (one format for every target)

JSON, read with the `System.Text.Json` the library already uses (decision 2: the spec showed YAML; the repository has
no YAML reader and no dependencies, so the same keys in JSON, comments and trailing commas allowed). As
`idrak cluster init` writes it (comments shortened here):

```json
{
  "format": "idrak-topology/1",
  "nodes": [
    { "name": "desk", "address": "192.168.1.10:29500",            // the host's name; the first node's address is the coordinator
      "gpus": [ { "device": "cuda:0", "memory": "8GB" }, { "device": "cuda:0", "memory": "8GB" } ] },   // ranks 0 and 1, one card
    { "name": "mini", "gpus": [ { "device": "cuda:0", "memory": "8GB" } ] },                              // rank 2
    { "name": "old",  "gpus": [ { "device": "cuda:0", "memory": "6GB" } ] }                               // rank 3
  ],
  "links": {
    "intra-node": "same-device",   // same-device | nvlink | pcie | shared-memory
    "inter-node": "ethernet-1g",   // infiniband | roce | ethernet-1g|2.5g|10g|100g | thunderbolt | direct-link | wifi
    "overrides": [ { "between": ["desk", "mini"], "type": "direct-link-10g" } ]
  },
  "simulation": { "seed": 1, "failures": "none" }
}
```

- `memory` is the fake GPU's **hard cap**, declared by the user; absent: the physical device's own. A cap above the
  physical device's memory spills to pinned host memory (`OffloadToHostMemory`, per device), so sharding and offload
  paths run for real on one card.
- `device` names the physical device the fake one is carved from on **this** machine (`cuda:0`, `vulkan:0`, `cpu`), so
  one file runs mixed CUDA, Vulkan and CPU workers in one simulated cluster.
- Link names are `LinkProfiles` entries; a suffix (`-10g`) overrides the bandwidth. `idrak cluster inspect FILE`
  prints the parsed topology, every pair's link, and the transport the selector would pick on this machine.
- **No presets** (the owner, 2026-10-18). The file is written by the **topology builder**, `idrak cluster init`: an
  interactive command in the style of `tune init` and `pretrain init`, through the CLI's `Terminal.Ask`, that asks one
  question per input with a one-line description of what the input means and why it matters, offers a default found
  on this machine where one exists, and writes the JSON with a comment per key (comments and trailing commas are
  allowed when the file is read). The questions, in order: how many machines and a name for each; for each machine,
  which physical device each fake GPU is carved from (default: the devices `Device.Available` lists here, with their
  measured memory shown) and its memory cap (default: the physical device's own; a cap above it is allowed and the
  description says it will spill to host memory); the link inside a machine (the `LinkProfiles` names with each one's
  bandwidth and latency printed beside it); the link between machines (the same list; the description says which the
  home LAN and a rental usually have); any pair with a different link (an override); the seed and the failure model
  for simulated runs. The first node is also asked its address (default: this machine's LAN address and port 29500),
  which is the coordinator every other machine connects to, and each node's name defaults to a host name, this
  machine's for the first. Every answer has a matching option (`--nodes 3`, `--devices cuda:0,cpu`, `--intra pcie`,
  `--inter ethernet-1g`, `--override a,b=direct-link-10g`, `--memory a:0=8GB`) so `--yes` writes the file without a
  terminal, as the house rule for prompts requires; `--dry-run` prints the file instead of writing it; `-j` prints it
  as JSON. The builder lives in `Idrak.Cluster` (`TopologyBuilder`: the questions as data, each with its name,
  description, default and validation, so an app can present them in its own UI) and the CLI only asks them.
- **The file is the source of rank and world size** (the owner, 2026-10-18: nothing the builder already counted is
  typed again). The world size is the file's device count. A process finds its node by this host's name (the builder
  suggests host names as node names) or by `--node NAME` / `IDRAK_CLUSTER_NODE`; when that node has one worker, its
  rank follows and nothing more is needed. When a node has several workers, `idrak cluster start` (below) starts one
  process per worker on that machine and gives each its rank; a hand-started process on such a node passes `--rank`
  or `IDRAK_RANK`. The coordinator is the first node's address when the file gives addresses (an optional `address`
  per node), else `--coordinator` / `IDRAK_COORDINATOR`. `IDRAK_RANK` and `IDRAK_WORLD_SIZE` remain for a run with
  no file (bare `--coordinator`, every rank on its own device) and for launchers that set them; a value that
  contradicts the file stops the run before it starts, naming both.
- **The built-in profiles carry no hardware figure** (C1, card-agnostic rule): a name that states its rate carries
  exactly that rate (`ethernet-1g` is 1 Gb/s, `direct-link-10g` 10 Gb/s); a link this machine can measure
  (`same-device`, `shared-memory`, `pcie` through host staging, `nvlink` where peers exist) is measured at first use and
  kept in the tuning cache; any other (`infiniband`, `roce`, `wifi`) has no rate until the topology gives one in its
  name (`infiniband-200g`) or a bench measures it, and `Predict` refuses to guess (the C0 contract already does).
- **The reader picks `same-device` itself**: two ranks naming one device of one node are joined by it, so
  `"intra-node"` is needed only where a node has different devices (decided while writing C0).
- **Two workers on one card** (the owner's question, 2026-10-18) are joined by the `same-device` link: a copy between
  them is a device-to-device copy inside one memory, at the card's own memory bandwidth, faster than NVLink or PCIe.
  Its bandwidth and latency are **measured on the device at first use** (one timed device copy, kept in the tuning
  cache) rather than declared, so the profile holds no card figure. The builder picks it by itself when a node names
  one physical device more than once. `nvlink` is two cards with peer access; `pcie` is two cards without it, or the
  host-staged path; `shared-memory` is two processes on one machine.
- **Every question suggests from the device it concerns** (the owner, 2026-10-18). The builder's default and the
  one-line suggestion beside each answer are derived from what it finds, never from a table: a worker's memory cap
  defaults to the physical device's measured memory, split evenly when the device is named more than once, with the
  driver's reserve subtracted; the link inside a node defaults to `same-device` when workers share a card, `nvlink`
  when the driver reports peer access between the named cards, `pcie` otherwise, and `shared-memory` for CPU workers;
  the link between nodes defaults to the profile nearest the speed the network adapter reports
  (`NetworkInterface.Speed`), or to `ethernet-1g` when none is up, and the suggestion names the adapter; the seed
  suggests the last run's seed when a run folder is given; the failure model suggests `none` for a first run and
  `mixed` once a run has passed. Where a profile was calibrated before, the calibrated figure is shown as the
  suggestion with its date. The suggestions go through the same `ISettingAdvisor` the trainer uses (decision 4), so
  `idrak suggest topology.json` prints them again for an existing file.

## Idrak.Cluster: the simulator (`src/Idrak.Cluster/`, on `Idrak.Abstraction`)

| File | What |
|---|---|
| `VirtualClock.cs` | Per-rank simulated time. A transfer from A to B started at A's time *t* over link *L* completes at `max(t, L.free) + L.Latency + bytes / L.Bandwidth + jitter(seeded)`; the link's `free` moves to that completion (one transfer at a time per direction: contention is modelled); B's clock becomes `max(B.now, completion)` when B receives. A collective's simulated duration is the critical path through those updates, so a run **predicts** the step time of a cluster that is not there, independent of wall time, and the result is deterministic for a seed. |
| `LinkModel.cs` | The arithmetic of `ILinkProfile.Predict` and the throttle: a token bucket that paces a real pipe (localhost mode) to the profile's bandwidth and adds its latency, so a localhost socket imitates any slower link. |
| `SimTransport.cs` | `ITransport` named `sim`: bytes by rank and tag over `Channel<ReadOnlyMemory<byte>>` per directed edge, bounded to the in-flight count so a rank that races ahead waits as it would on a socket; every transfer is charged on the virtual clock and published as `TransferCompleted` with its simulated time. Buffers are rented from `ArrayPool<byte>` and returned by the receiver. |
| `SimulatedDeviceProvider.cs` | `DeviceProvider` of kind `sim` (below, "Fake devices"). |
| `Profiles.cs`, `Failures.cs` | the built-in `ILinkProfile`s and `IFailureModel`s, registered as library defaults. |
| `TopologyBuilder.cs` | the builder's questions as data (name, description, default found on this machine, validation) and the writer of the commented JSON; the CLI's `cluster init` asks them, an app can show them in its own UI. |
| `InProcessCluster.cs` | N ranks as tasks on their own threads (`Parallel.For` with `MaxDegreeOfParallelism = worldSize`), one `SimTransport` per rank over one edge set, one clock; `Run(rank => work)`. The deterministic unit-test mode. |
| `LocalhostCluster.cs` | One process per rank: starts `dotnet …` children with `IDRAK_RANK`, `IDRAK_CLUSTER_NODE`, the topology file and the key set, a port the OS picks reported back on stdout as bytes, waits with the timeout, kills the children in `finally`. The **real** transport code runs; the throttle (optional) applies the profile. `idrak cluster start` is the same launcher restricted to this host's node, without the throttle. |
| `Faults.cs` | The seeded fault injector: every fault (slow worker: its clock and its sends are stretched by a factor; dropped worker: its pipe closes; link loss: messages dropped at `LossRate`; partition: a set of links closed for a span; late join: a rank that connects N steps late) is drawn from `Random(seed)` at a step and written to the event log with `(seed, step, kind, ranks, duration)`. |
| `EventLog.cs` | A compact binary log (bytes, `BinaryPrimitives`) of transfers, faults and clock values, enough to replay a run bit for bit from `(seed, step)`: `Replay(seed, step)` reconstructs the fault schedule and the clock state at that step. |
| `Calibration.cs` | Given measured transfers `(bytes, seconds)` per link and the profile that predicted them, fits bandwidth and latency by least squares over `seconds = latency + bytes / bandwidth`, reports the residual, and returns the corrected profile values. |

**Fake devices** (`src/Idrak.Cluster/SimulatedDeviceProvider.cs`): a `DeviceProvider` of kind `"sim"` registered into
`DeviceProviders`. `sim:N` is the N-th device of the active topology: one more backend instance of the physical
device's kind (`DeviceProvider.Create(ordinal, limit)`, an overload the three GPU providers and the CPU provider gain: the
backend's `MemoryAccountant` takes the fake device's cap as its limit function, and offload is per device), its own
pool, its own streams, so two fake GPUs on one card contend for compute as two real ones would share a bus, and each
hits its own hard cap. `DeviceUuid` reports the physical device, so a run that wants real parallelism (two physical
cards) is told when two ranks share one. The simulator carries bytes between ranks and hands out devices; a
`SimTransport` given a `Storage` stages it through the device's own `CreateHostStaging` like any other transport, and the
collectives in `Idrak.Network` do the reduction arithmetic on whichever device the rank holds.

## Idrak.Network: transports, collectives, training hooks

**Transports** (`src/Idrak.Network/Transports/`), each an `ITransport` registered by name with its conformance test:

| Name | Scope | How, with .NET and the driver only |
|---|---|---|
| `sim` | one process | **in `Idrak.Cluster`** (above); listed here because the selector and the collectives treat it as any other transport |
| `shared-memory` | processes on one machine | `MemoryMappedFile` ring buffers, one per directed pair, a named semaphore or an event per direction; the fastest cross-process path, zero-copy on the receive side |
| `pcie-host` | GPUs in one machine, one process | device → pinned host staging (`CreateHostStaging`) → device; the path consumer boards actually have; the default `Backend.CopyTo` |
| `nvlink` (alias `p2p`) | GPUs in one machine, one process | `cuMemcpyPeerAsync` when `cuDeviceCanAccessPeer` says yes (`cuCtxEnablePeerAccess` once per pair, remembered), a `CudaBackend.CopyTo` override; refused peers fall back to `pcie-host` and the telemetry says so once per pair |
| `direct-link` | two machines, one cable (crossover Ethernet, Thunderbolt or USB4 networking) | TCP straight to the peer's address from the file's override: no coordinator hop, no switch; the profile differs, the socket code is `tcp`'s |
| `tcp` | any machines | the baseline: rendezvous at the coordinator, HMAC-SHA256 hello under `Key`, one connection to the ring successor, Nagle off, the socket buffers as set |
| `udp` | any machines | optional, later: control messages (barrier, heartbeats) and loss-tolerant gradient chunks with sequence numbers; `Reliable = false` so collectives that need delivery refuse it |
| `rdma` | InfiniBand, RoCE | a registered name with no library default (native verbs); a plug-in later |

**The wire format** (`WireFormat`, public in Abstraction so a plug-in transport speaks it): a 36-byte little-endian
header (`TryReadHeader(ReadOnlySpan<byte>, out WireHeader)`, `WriteHeader(Span<byte>, in WireHeader)`): magic `IDRK`,
version (u16), kind (u16: hello, welcome, chunk, ack, barrier, heartbeat, bye, error), rank, sequence, tag, chunk index,
chunk count, element type (f32, bf16), op, payload bytes. The hello carries an HMAC over (nonce ‖ rank ‖ world) under
`Key`; nothing is accepted from a peer whose hello failed; an unknown version ends the connection naming both versions.

**Transport selection** (`FastestSelector`): for each pair of ranks, every registered transport reports whether it is
available between them on this machine (`ITransportAvailability`: same process, same machine, peer access, a route to
the address); the selector takes the first available in the order of the table above (fastest first), logs the choice
per pair at `TelemetryLevel.Network`, and `ClusterOptions.Transport` forces one. `idrak transports` prints the table for
this machine; `idrak cluster inspect` prints the choice per pair.

**Collectives** (`src/Idrak.Network/Collectives/`), each an `ICollectives` over any `ITransport`:

- `naive`: every rank sends to rank 0, rank 0 sums in rank order and broadcasts. The first implementation and the
  reference every other collective is compared with (within float rounding; exactly on whole numbers).
- `ring`: reduce-scatter then all-gather over the ring; slice *j*'s sum is formed at rank (*j* - 1) mod N by adding
  ranks *j*, *j*+1, … in that order whatever the timing (rule 86), chunks of `ChunkBytes` with `InFlight` outstanding
  per direction; the receive side sums in float32 straight from the rented receive buffer into the staging slot.
- `hierarchical`: reduce-scatter inside each node over the intra-node transport, exchange only the reduced shards
  between nodes over the inter-node transport (one rank per node, a ring across nodes), all-gather inside each node.
  The simulator **must show** its simulated time beating `ring`'s on a two-level topology; the test asserts it.

Device buffers: a collective takes a `Storage` on any device and stages chunks through `CreateHostStaging` (pinned where
the device has it); on the CPU the buffer is the storage; with `GpuDirect` transports the chunk stays on the device and
`Binary(Add)` sums it there. The staging is created once per group and reused every step (rule 77). `WireType.BFloat16`
packs on the device (`PackBFloat16`), sums in float32 on arrival, sends finished slices back as float32; `TopK` with
error feedback keeps the residual per parameter in a buffer allocated once.

**Training hooks** (`Idrak.Nlp`, `Idrak` core): `FineTuningOptions.Cluster`, `PretrainingOptions.Cluster`,
`Trainer.Cluster` (an `ICollectives?`; null: today's behaviour, no change in any existing test). In `RunStep`, after the
last micro-batch's `Backward` and before `Update`: `cluster?.AllReduceGradients(optimizer.Parameters, options)`; the
gradient storages are addressed as (parameter, offset, count) pieces in parameter order, so the ring's slices need no
packed copy. The sum is token-weighted (one float per rank travels in the first chunk) so the result is the gradient one
process would compute on every rank's rows. Data is sharded by **global draw position** (rank *r* takes draws
*k·W + r*), so a checkpoint resumes at any world size; rank 0 writes checkpoints, a barrier follows; the reported loss
is the token-weighted mean so every rank makes the same early-stopping decision; dropout seeds combine with the rank.

## Phases, in order

### C0: the reference, the contracts, the registries

**Files.** `src/Idrak.Abstraction/Cluster/` (the contracts above, `ClusterOptions`, `WireFormat`, `Topology` reader,
the five registries, the telemetry events), `src/Idrak.Abstraction.Testing/Contracts/TransportSuite.cs`,
`CollectivesSuite.cs`, `LinkProfileSuite.cs`, `TopologySuite.cs`; `tools/cluster/make_tiny_ddp.py` writing
`tests/Idrak.Tests/data/cluster/`.

**The reference fixture** (the addition that proves correctness against something outside Idrak): PyTorch
`DistributedDataParallel` with the `gloo` backend, two CPU processes over plan 19's tiny model and tokens
(`tests/Idrak.Tests/data/pretrain/tiny-llama/`), rank *r* taking windows *r*, *r*+2, …; SGD for 3 steps, then AdamW;
`losses`, `grads-step1.safetensors`, `weights-step3.safetensors`; a `single/` run of the global batch in one process that
must agree with `ddp/` within 1e-5 (the script asserts it); `bf16/` with gradients rounded to bfloat16 before the sum, so
the looser tolerance of the bf16 wire is measured, not guessed; `manifest.json` with SHA-256 and versions; a README.
Reruns write the same bytes.

**Suites.** `TransportSuite` (every message size including 0, 1, 7, 65,537 bytes and a size above `ChunkBytes`; tags;
both directions at once; a receive from a rank that never sends times out naming it), `CollectivesSuite` (every
operation against the sum in rank order on 2, 3, 4 and 8 ranks, sizes that are not multiples of the world size, exact
on whole numbers and within float rounding otherwise, identical bytes on every rank and on a rerun), `LinkProfileSuite` (`Predict` is monotonic in bytes, jitter is reproducible for a seed, `With` keeps the
other fields), `TopologySuite` (every rank placed once, every pair has a link, overrides win, a file the builder
wrote reads back to the same topology).

**Tests** (`"cluster: "`): "cluster: ddp fixture agrees with single"; "cluster: wire header round trip, refuses a wrong
magic and a newer version"; "cluster: options from environment, the key never in ToString"; "cluster: every registry
describes its defaults and refuses an unregistered name with the Register hint"; "cluster: topology file parses";
the kit's suites checked on a reference in-memory transport and naive collectives written in the test project, and on
broken twins they must fail; inventory and API dumps regenerated (Abstraction +1 namespace). The builder's questions
and `inspect` move to C1, where `Idrak.Cluster` holds the builder.

**Done when** the fixture exists with its README, two script runs write the same bytes, and the filter passes on the CPU.

**C0 as built (2026-10-18).** In `src/Idrak.Abstraction/Cluster/`: `ITransport` with `ITransportProvider`,
`TransportCapabilities`, `TransportAvailability` and `ClusterContext`; `ICollectives` with `ICollectivesProvider`,
`ReduceOp`, `WireType`; `ILinkProfile` with the data record `LinkProfile` and `LinkProfiles.Resolve` (a registered name,
or one with a rate suffix in bits a second); `IClusterTopology` with the reader and writer `ClusterTopology` (JSON with
comments and trailing commas, unknown keys refused by name, memory sizes per decision 7, `same-device` picked by the
reader for ranks sharing a device, the first node's address as `Coordinator`); `ITransportSelector`; `IFailureModel` with
`Fault` and `FaultKind`; the five unguarded registries (`Transports`, `Collectives`, `LinkProfiles`,
`TransportSelectors`, `FailureModels`); `ClusterOptions` and `SimulationOptions` with 35 variables (`ClusterKey` keeps
the secret out of every `ToString`); `WireFormat` (36-byte header, the HMAC hello). `LibraryDefaults` lists
`Idrak.Cluster` and `Idrak.Network`. Telemetry: `TelemetryLevel.Network` (in `All` and in `trace`'s default and
`--levels network`), the six events, their publishers and hook methods, carried by `ConsoleLogger`, `MetricsRecorder`
(`Collectives`), `ChannelTelemetry`, `JsonLinesLogger` and `TelemetryJson`. Kit: `TransportSuite`,
`CollectivesSuite`, `LinkProfileSuite`, `TopologySuite`, `TransportUnderTest`, `CollectivesUnderTest` and
`Conformance.CheckTransport|CheckCollectives|CheckLinkProfile|CheckTopology`. Tests: `tests/Idrak.Tests/ClusterTests.cs`
(nine `"cluster: "` tests) and `ClusterReference.cs` (an in-memory transport and naive collectives, each with a broken
twin the suites fail). The variables are in the CLI's table under "Cluster" (listed for `test` until C2 gives the
training commands their part). The DDP fixture: `tools/cluster/make_tiny_ddp.py` and `tests/Idrak.Tests/data/cluster/`
(torch 2.14.1, transformers 5.19.0; ddp against single within 1e-5: SGD weights 1.2e-7, AdamW 2.3e-6, folded AdamW
6.7e-6; bf16 against ddp: gradients 5.9e-4, weights 1.6e-4; three runs wrote identical bytes). One build of the test
project; `IDRAK_DEVICES=cpu` with the filter `cluster:|public API|abstraction inventory|cli health: …|cli polish: …|cli
dev: trace|profiler:`: 22 passed, 0 failed (the transport suite 2.4 s, the collectives suite 1.6 s). API dumps of
Idrak.Abstraction and Idrak.Abstraction.Testing and the inventory regenerated. Left for C1: the `sim` transport and
devices, the built-in profiles and failure models, `naive` and `ring` in Idrak.Network, the builder and `cluster
init|inspect|sim`.

### C1: Idrak.Cluster in-process, the `sim` transport, `naive` and `ring`

**Files.** `src/Idrak.Cluster/` (the table above, except `LocalhostCluster` and `Calibration`),
`src/Idrak.Cluster/Idrak.Cluster.csproj` and `src/Idrak.Network/Idrak.Network.csproj` (each references Abstraction
only; `IsAotCompatible`), a `LibraryRegistrations.cs` in each, `Idrak.Cluster`: `SimTransport.cs`,
`SimulatedDeviceProvider.cs`, `Profiles.cs`, `Failures.cs`, `TopologyBuilder.cs`; `Idrak.Network`:
`Collectives/NaiveCollectives.cs`, `RingCollectives.cs`, `Collectives/ChunkPlanner.cs`, `Selection/FastestSelector.cs`;
the `Create(ordinal, limit)` overload in `DeviceProvider` and the four providers.

**The new-package checklist** (both packages, from the survey of how the others are wired): a line each in
`Idrak.slnx`; `_IdrakSourcePackage` in `build/IdrakFromSource.targets`; `ApiAssemblyNames` in
`tests/Idrak.Tests/ApiSurfaceTests.cs` and the dumps `api/Idrak.Cluster.txt`, `api/Idrak.Network.txt`;
`LibraryAssemblyNames` in `InventoryTests.cs`; `LibraryDefaults.Assemblies` and a `[DynamicDependency]` for each; a
`ProjectReference` in `tests/Idrak.Tests/Idrak.Tests.csproj` and in `Idrak.Cli.csproj`; a `README.md` per package
listing its contracts (packed with it); `Idrak.Abstraction.Cluster` is not a global using (its short names, `Fault`,
`Transports`, would reach every project; the packages that use it say `using Idrak.Abstraction.Cluster;`); every contract only `Idrak.Network` uses today but any package may use later
(`ICollectives`, `ITransportSelector`) justified in `tests/Idrak.Tests/data/shared-contracts.txt` (`ITransport`,
`ILinkProfile`, `IClusterTopology` and `IFailureModel` have two users, Cluster and Network, so decision 10 places them in
Abstraction on its own). The csproj of each: `IsAotCompatible`, `GenerateDocumentationFile`, `InternalsVisibleTo
Idrak.Tests`, one `ProjectReference` (Abstraction), and a test asserts that neither references the other or anything
else ("cluster: Idrak.Cluster and Idrak.Network reference Idrak.Abstraction only").

**Tests** (`"cluster: "`): "cluster: sim ring all-reduce equals naive within float rounding, exactly on whole numbers, identical on every rank and rerun" (2, 3, 4, 8 ranks;
sizes 1, 7, 1,023, 65,537; sum, max, min); "cluster: sim all-gather, reduce-scatter, broadcast, barrier";
"cluster: a rank that never arrives times out naming it"; "cluster: the virtual clock charges bytes / bandwidth +
latency and is deterministic for a seed" (two runs, equal logs); "cluster: a bounded pipe makes a fast rank wait";
"cluster: a fake device hits its cap and spills to the host" (`sim:0` with a cap of a few MiB over the CPU: an allocate
past the cap offloads, `GetMemoryUsage().Offloaded` grows, `InUse` does not pass the cap); "cluster: two fake devices
on one physical device report one UUID"; "cluster: chunk size measured once and kept" (the cache file appears; a forced
`ChunkBytes` skips it); `TransportSuite` on `sim`; `CollectivesSuite` on `naive` and `ring` over `sim`; "cluster: sim
telemetry reports bytes, wall and simulated time per collective, and nothing is published when no hook listens"
(the counters checked against `Telemetry.IsEnabled` false); "cluster: TransportChosen is published once per pair";
"cluster: ConsoleLogger, MetricsRecorder, ChannelTelemetry and JsonLinesLogger carry the network events"; `Stress.Run` on `ring`: device memory does not grow
across 100 steps (rule 77); "cluster: the network and cluster probes report bytes, blocked time and the predicted step
through Profiler.Run" (the probes registered, `ProfileReport` rows named per pair, the `cluster` probe skipped and
listed when `sim` is not active).

**Done when** the filter passes on the CPU after one build of Idrak.Abstraction, Idrak.Cluster, Idrak.Network and the
tests; the owner's `cuda` and `vulkan` runs of the same filter pass with fake devices carved from the GPU.

**C1 as built (2026-10-18).** `src/Idrak.Cluster/`: `SimulatedDevices` (kind `sim`; `Carve(physical, cap, slot,
offload)` gives a device of the physical one's type and kernels on a backend instance of its own with its own memory
accountant capped at `cap`; the same carving gives the same device, a carved device is never listed or carved again,
and it reports its card's UUID), `VirtualClock` (network time only: bytes / bandwidth + latency, one transfer at a time
each way, jitter and loss drawn from a SplitMix64 hash of seed, ends, tag and message number, a lost message sent
again), `SimTransport.cs` (the `sim` provider: a hub per coordinator name, bounded queues of `InFlight` messages per
receiver and tag, timeouts and a left rank named at once), `Profiles.cs` (`BuiltInLinks`: the ethernet rates from their
names, the bases at bandwidth 0 until a rate suffix, `same-device`, `shared-memory` and `pcie` measured here once per
device with `IDRAK_CLUSTER_PROBE_BYTES` (default 32 MiB, at most an eighth of the free memory); the `none` failure
model), `InProcessCluster` (a thread per rank on carved devices, `ClusterRunResult` with every rank's error),
`TopologyBuilder` (`MachineFacts.Gather()`, one `TopologyQuestion` per input with a description, a default and the
suggestion's reason; answers by key; a commented file), `TopologyInspection` (the links as this machine resolves them;
`PredictAllReduce` plays ring or naive on the clock), the `cluster` probe. `src/Idrak.Network/` (root namespace
`Idrak.Networking`, below): `Abstractions/HostCollectives.cs` (one pooled host buffer per call, chunks of
`ChunkBytes`, chunk-interleaved exchange so a ring never waits in a circle, `CollectiveCompleted` only while a hook
listens), `Collectives/Algorithms.cs` (`naive`, `ring`), `WireValues.cs` (float32 and bfloat16 with a Vector256 loop
and a scalar tail), `Providers.cs` (the providers and the `fastest` selector), the `network` probe. Devices:
`DeviceProvider.TypeOf(ordinal)` and `CreateSeparate(ordinal, SeparateBackendOptions)` (name, memory limit, offload)
in place of the planned `Create(ordinal, limit)`; `Device.CreateSeparateBackend`; CPU, CUDA (its own stream, module
and accountant on the retained primary context; offload per instance), Vulkan and HIP implement it. CLI: `idrak
cluster init` and `idrak cluster inspect` (group "Cluster"). Tests: 14 in `ClusterSimTests.cs`, 36 in the C0+C1
filter, passing on the CPU.

Changes from the plan, kept: **the namespace is `Idrak.Networking`** (the package and assembly stay `Idrak.Network`):
a namespace `Idrak.Network` hides the `Idrak.Network` builder from every file in an `Idrak.*` namespace that
references the package (the CLI broke on it); `LibraryDefaults.Assemblies` pairs each assembly with its root namespace,
and the package's own contracts go in `Idrak.Networking.Abstractions`. The contracts only one package uses so far
(`ITransportProvider`, `Transports`, `ITransportSelector`, `TransportSelectors`, `IFailureModel`, `FailureModels`) are
justified in `shared-contracts.txt` (both packages use them by C5). A carved device's cap is hard on the CPU, Vulkan
and HIP (an allocation past it fails naming the device); on CUDA it spills to host memory when offload is on (the
owner's CUDA run checks it). HIP carving is compiled, not yet run. Moved: `idrak cluster sim` to C2 (it runs training);
the measured chunk size and its cache to C5 (in C1 `ChunkBytes` is a setting, unset means one message per piece);
`TransportChosen` to C3 (the in-process cluster uses `sim` alone until the selector chooses across pipes); the
`slow`, `drop`, `loss`, `partition` and `late` failure models to C3; `hierarchical` and its default to C3 (C1's default
collective is `ring`).

### C2: data parallel on the simulator

**Files.** `src/Idrak.Nlp/FineTuning.cs` (the hook in `RunStep`), `Pretraining.cs` (the rank's share of the draw, rank
0's checkpoint and the barrier, resume at another world size), `src/Idrak/Training/Trainer.cs` (`Cluster`, the hook
before `Optimizer.Step`), `src/Idrak.Network/Training/GradientReduction.cs` (`AllReduceGradients`: the pieces, the
token weights, the starting-state broadcast or hash check, `SyncWeightsEvery`), `HostOptimizer` reducing on the pieces it
already downloads.

**Batch sizes by measured device speed** (the spec's heterogeneous cards): each rank times its first warm steps
(rule 76) and all-gathers tokens per second; the batch tokens per rank are then split in proportion, re-measured every
`IDRAK_CLUSTER_REBALANCE_EVERY` steps (default: the first 3 warm steps only); the token-weighted sum makes the result the
same as equal splits.

**Tests** (`"cluster: "`): "cluster: data parallel 2 ranks on sim matches the ddp fixture" (grads 1e-4, losses and
weights 1e-5, SGD and AdamW; the fixture's own ddp-against-single gap for folded AdamW is already 6.7e-6, so that
comparison is against `ddp/adamw-folded/` and its tolerance is set from that measured gap, not widened by guess); "cluster: 2 ranks equal one process with the global batch" (rule 71, Idrak against Idrak);
"cluster: 3 ranks with unequal token counts give the token-weighted gradient"; "cluster: bf16 wire within the fixture's
tolerance"; "cluster: rank 0 alone writes the checkpoint; a 2-rank run resumes as 3 ranks giving the 3-rank run's
losses"; "cluster: mixed kinds (cpu and a fake device of every listed kind) sync weights every K steps"; "cluster:
batches split by measured speed give the one-process losses"; "cluster: HostOptimizer reduces on its downloaded pieces";
"cluster: the Spirals sample's Trainer with a cluster". Fine-tuning and pretraining filters unchanged with `Cluster` null.

**Done when** every test passes on the CPU and the owner's `cuda` run of `"cluster:"` passes with two fake GPUs.

**C2 as built (2026-10-18).** The hook is a contract in `src/Idrak.Abstraction/Cluster/GradientSync.cs`:
`IGradientSync` (`Start`, `Reduce(tokens, loss, normalized)`, `Stepped(step)`, `Shares`, `Sum`, `Broadcast`, `Barrier`),
`IGradientSyncProvider` and the unguarded registry `GradientSyncs` (`Create(collectives, optimizer, options)`), named by
`ClusterOptions.GradientSync`; `IHostGradients` (an optimizer that holds the gradients on the host); `DrawShares.Owners`
(the rank of each of a step's draws); `GradientSyncResult`; `StartingState`. Idrak.Network implements and registers
`"all-reduce"` in `src/Idrak.Network/Training/GradientReduction.cs` (from `LibraryRegistrations`); `HostCollectives` gained
`AllReduceHostValues` (a bucket reduced straight from the host buffer). `Trainer.Cluster`, `FineTuningOptions.Cluster` and
`PretrainingOptions.Cluster` (each an `ICollectives?`, with a `ClusterOptions?` beside it; null reads the environment) call
it after the last backward pass and before the update; Idrak and Idrak.Nlp reference neither Idrak.Network nor
Idrak.Cluster. The reduction: the gradients as (parameter, offset, count) pieces in parameter order, a bucket of
`BucketBytes` at a time through one pooled host buffer (returned every step), a piece of a parameter larger than a bucket
written back through one device staging buffer made once (on the CPU straight into its memory); the header (this rank's
tokens, its loss and one presence flag per parameter) rides in the first bucket on a float32 wire, the gradients weighted
by the rank's tokens and the first bucket divided by the step's tokens once known, every later bucket weighted before it
travels; a rank without rows adds zeros and gets the gradients other ranks had; a step no rank had rows in returns 0
tokens (the Trainer's ranks of unequal shards end an epoch together that way). `Start` sends rank 0's values (or with
`StartingState.Check` compares every parameter's SHA-256 across ranks and names the rank and parameter that differ);
`Stepped` sends rank 0's weights every `SyncWeightsEvery` steps and re-splits by speed: each rank's tokens per second
over the warm steps (the first, cold step aside, rule 76) summed into one list, the shares in proportion unless the
ranks are within `RebalanceTolerance`. Data: the Trainer shards a `DataLoader` by sample (`DataLoader.Shard`, rank r the
epoch's samples k·W + r, so batch k of every rank is one process's batch k of W times the size; a shuffling loader needs
a seed), another batch source by batch; the fine-tuner's step is `GradientAccumulation` batches per rank (W ranks train
as one process with W times the accumulation); the pretrainer's step is the options' step at every world size (every rank
draws it alike on the background draw, each trains its share), so a checkpoint resumes at any world size. Rank 0 writes
checkpoints, adapters, the Trainer's `Checkpoint` and `CsvLog` files (`TrainerContext.Rank`/`WorldSize`) and publishes the
training telemetry, then a barrier; losses, held-out losses (each rank its share, the sums added) and early stopping are
over every rank's rows; the Trainer sends rank 0's batch-norm statistics at each epoch's end; `Dropout.SeedsForRank` mixes
the dropout seeds with the rank (rank 0 draws what one process would). `HostOptimizer` implements `IHostGradients`: the
sync has it download (its pinned pieces), reduces on its host copies, and the update clips and steps from them.
`idrak cluster sim --topology FILE [--collective] [--wire] [--seed] -- pretrain … -o RUN` (`ClusterSimCommand`, group
"Cluster", Arabic summary) runs `pretrain` once per rank on carved devices, rank 0 writing the run, and prints each rank's
network time on the virtual clock beside the wall time. Four settings joined `ClusterOptions` and the CLI's table:
`RebalanceSteps` (`IDRAK_CLUSTER_REBALANCE_STEPS`, 3: the plan's first 3 warm steps; 0 keeps equal splits),
`RebalanceTolerance` (`IDRAK_CLUSTER_REBALANCE_TOLERANCE`, 0.1), `GradientSync` (`IDRAK_CLUSTER_GRADIENT_SYNC`,
`all-reduce`), `StartingState` (`IDRAK_CLUSTER_START`, `broadcast`); `BucketBytes` unset means 16 MiB buckets until C4
measures it. Tests: 13 in `tests/Idrak.Tests/ClusterTrainingTests.cs` (`ClusterTrainingGroup`); with C0 and C1, 36
`"cluster: "` tests pass on the CPU, and the fine-tuning, pretraining, Trainer, callback, data loader and HostOptimizer
filters pass unchanged with no cluster. Measured against the DDP fixture (two ranks, the Trainer on its eight windows):
step 1 gradients 5.8e-7 apart, SGD weights after three steps 1.2e-7, folded AdamW weights 2.2e-5; over a bf16 wire against
`bf16/`: gradients 1.2e-4, losses 1.4e-6, weights 4.9e-5 (inside its measured 5.9e-4, 1.5e-4, 1.6e-4).

Changes from the plan, kept: **a registry and a contract instead of an `AllReduceGradients` method**, so C4's bucketed,
local-SGD and top-k synchronisations register beside `all-reduce` and a run names one; `ClusterOptions` travels beside
the `ICollectives` (the contract gives no options). **On a bfloat16 wire the header goes first on its own in float32**
(a token count past 256 would be rounded in bf16), one more small collective a step; the float32 wire keeps it in the
first chunk as planned. **A loss of one's own (DPO, ORPO, SimPO, distillation) is not token-weighted**: every rank knows
the whole step's draws, so it divides by the step's token and sequence counts over every rank and the ranks' gradients
are summed (`normalized`), which is exact for losses normalized by sequences as well as by tokens. **The fine-tuner's
global step is W times one rank's** (resume at another world size is the pretrainer's, which has checkpoints of its
state); **the pretrainer's step is the same at every world size** (the options describe the step, each rank trains about
1/W of it). Every rank reads every sequence of a step (on the background draw, overlapping the step) and keeps its
share: a reader that tells a sequence's length without reading it would spare that and is left for a source slow to read.
Rank 0 alone publishes the training telemetry (in-process ranks share one process's telemetry); the fine-tuner's held-out
loss of one's own runs on every rank (they agree), and generated answers are scored on rank 0. The AdamW tolerance of the
fixture test is the fixture's own measured growth (`vs_single`: step 1 gradients 1.3e-7 apart gave folded AdamW weights
6.7e-6 apart) applied to the run's measured gradient gap (bound 2.9e-5, never below 1e-5); SGD, the losses and each
rank's losses are held to 1e-5. The Spirals test builds the sample's data, model and settings in the test (the sample's
`Program.cs` is top-level code; the sample is unchanged). The mixed-kinds test carves the CPU and one device of every other
kind listed; on a machine with the CPU alone it runs two CPU ranks (the owner's CUDA and Vulkan runs mix kinds). Moved:
`cluster sim` for `tune` and `train` (with `--cluster` on the training commands) to C8; `--mode localhost` to C5;
`--failures` and `--profile` on `cluster sim` to C3 and C8.

### C3: topology-aware all-reduce, the selector across mixed pipes, failures and replay

**Files.** `Collectives/HierarchicalCollectives.cs`, `Selection/` (availability per transport), `src/Idrak.Cluster/Faults.cs`,
`EventLog.cs`, `Replay.cs`.

**Tests** (`"cluster: "`): "cluster: hierarchical equals ring within float rounding, every rank and rerun identical"; "cluster: hierarchical beats ring in
simulated time on a two-level topology" (a file the test builds: two machines of four fake devices, `nvlink` inside,
`ethernet-1g` between; the ratio is
printed and must be above 1); "cluster: the selector picks per pair in the fastest-first order and logs it; a forced
transport wins"; "cluster: a slow worker shows as Blocked on every other rank and SlowestRank names it"; "cluster: a
dropped worker ends the step on every rank naming it"; "cluster: link loss is resent and the step completes"; "cluster:
a partition times out naming both sides"; "cluster: a late join connects at step N"; "cluster: every fault and rank-state change is a telemetry
record with its seed and step, and runs show charts the network share"; "cluster: rank 0's log holds every rank's
per-step summary in localhost mode"; "cluster: replay --seed S --step N
reproduces the fault schedule and the clock bit for bit" (two replays, equal logs).

**Done when** the simulator shows the hierarchical win and every fault replays.

### C4: overlap, buckets, compression, local SGD

**Files.** `Training/GradientBuckets.cs` (a hook per parameter at the end of its backward; buckets of `BucketBytes`
reduced while the backward continues; not inside a replayed graph: a recorded step reduces after the replay and the
telemetry shows the lost overlap), `Compression/` (`bf16`, `top-k` with error feedback), `LocalSgd.cs` (sync every *k*
steps, the average of the weights, not the gradients).

**Tests**: "cluster: bucketed overlap equals the unbucketed step"; "cluster: overlap shortens the simulated step on a
slow link" (the simulator's number, asserted); "cluster: top-k with error feedback converges on the Spirals sample within
the full-gradient loss after N steps" (the tolerance measured on the CPU and written into the test name); "cluster:
local SGD every k steps equals one process at k = 1 and runs k times fewer collectives".

### C5: real transports on localhost, then the home LAN, then a rental

**Files.** `Transports/SharedMemoryTransport.cs`, `TcpTransport.cs`, `Rendezvous.cs`, `DirectLinkTransport.cs`,
`PciHostTransport.cs`, `NvlinkTransport.cs` (`CudaBackend.CopyTo` peer override), `src/Idrak.Cluster/LocalhostCluster.cs`,
`Bench/LinkBench.cs` (bandwidth and latency per pair, cold and warm apart).

**Tests** (`"cluster: "`): `TransportSuite` and `CollectivesSuite` on `shared-memory` and `tcp` over loopback with
child processes (2 and 3 ranks); "cluster: tcp rendezvous retries then fails naming the rank and address"; "cluster: a
hello with a wrong key is refused; nothing listens off loopback without a key"; "cluster: localhost mode with the
throttle reproduces the profile's bandwidth within 20%" (the token bucket measured); "cluster: the localhost run and the
in-process run give bit-identical weights" (rule 71 across modes); "cluster: CopyTo default equals Download then Upload
on every device"; "cluster: nvlink falls back to pcie-host where peer access is refused and the telemetry says so once".

**The owner's runs** (speed only; correctness is already proven): the three home machines over `direct-link` and `tcp`,
then a rental with `pcie-host` and `nvlink`; `idrak cluster bench` on each pair; the numbers go into "Runs".

### C6: sharding (ZeRO-style)

`Shard = optimizer` (each rank keeps the moments of its slice; reduce-scatter, update the slice, all-gather the weights),
then `gradients`, then `parameters` (each rank holds its slice of the weights and gathers a layer's weights just before
its forward and backward, freeing them after). The fake GPUs' hard caps are what make this testable on one card: a
model that does not fit one fake GPU's cap fits across four.

**Tests**: "cluster: sharded optimizer state equals the unsharded weights after 3 steps"; "cluster: a model over the
cap of one fake GPU trains across four with `Shard = parameters`, each rank holding its share" (`InUse` per rank against
the share); "cluster: a sharded run resumes at the same world size".

### C7: pipeline parallelism

`Placement = pipeline`: decoder blocks cut so stages hold equal parameter bytes (measured from the checkpoint's tensor
sizes) or by each node's declared memory; `PipelineBoundary : Module` whose forward sends the activation and whose
backward receives the gradient (an `Autograd.Function` with a `Receive` in its backward); GPipe with `MicroBatches`
micro-batches in flight; 1F1B only if the telemetry's bubble number justifies it. Inference: the hidden state passes
through the stages; each keeps its own KV cache.

**Tests**: "cluster: pipeline 2 stages on sim gives the one-device logits, losses and gradients of the tiny model";
"cluster: micro-batches 1, 2 and 4 give the same loss"; "cluster: a plan covers every block once, embeddings first,
head last"; "cluster: a rank holds only its stage's tensors"; "cluster: chat through a 2-stage pipeline answers as one
device does".

### C8: the command line and calibration

- `idrak cluster init [-o topology.json] [--yes with the options above] [--dry-run]` (the interactive builder)
- `idrak cluster inspect FILE` (the parsed topology, which node this host is, the links, the transport per pair)
- `idrak cluster sim --topology FILE [--mode in-process|localhost] [--failures MODEL] [--seed S] -- <train args>`
  (every rank of every node on this machine, simulated)
- `idrak cluster start --topology FILE [--node NAME] -- <train command and args>`: runs this machine's part of a real
  cluster job. It finds this host's node in the file, starts one process per worker of that node with its rank,
  world size, coordinator and device set, streams their progress as one live line per rank, and ends them all when
  one fails. The same on every machine, so a three-machine job is three identical command lines.
- `idrak cluster replay --seed S --step N --topology …` (reproduces the fault schedule and the clock; reruns the step)
- `idrak cluster bench --topology … [--bytes N] [--rounds N]` (bandwidth and latency per pair, cold and warm; the input
  calibration needs)
- `idrak cluster calibrate --from RUN|bench.json` (reads the transfer records, fits each link's bandwidth and latency,
  writes the per-link profile files under `{cache}/cluster/` with the residual; `--dry-run` prints the deltas)
- `idrak transports` (every registered transport, available or not on this machine, with its capabilities)
- `--profile` on `cluster sim` and `--cluster FILE` on `profile` (the two probes above; `-j` for the JSON the
  calibration reads); `idrak tuning show` lists the calibrated profiles and the measured chunk and bucket sizes
- `--cluster FILE` on `train`, `pretrain`, `tune`, `resume`, `distill` for a hand-started single process: the rank and
  world size come from the file as above; `--rank N` names the rank only when this node has several workers (the
  words every cluster tool uses, decision 3), `--world-size N` only without a file. Each command owns its options: on
  `tune` and `distill`, `--rank` stays the LoRA rank as today, nothing is renamed, and a hand-started rank on a
  multi-worker node passes `IDRAK_RANK` instead; `cluster start` covers both without it. The key only
  from `IDRAK_CLUSTER_KEY`.

**Tests** (`"cluster: cli …"`): one per command through `CommandLine.Run`; "cluster: cli sim --mode localhost over a
file the test writes with cluster init --yes writes one run folder from rank 0"; "cluster: cli calibrate corrects a profile from a
run whose log it reads" and from a bench's JSON; "cluster: cli cluster init answers through the test input write the file, --yes with options writes the same, and
without a terminal the usage error names the option to add"; "cluster: cli cluster sim --profile prints kernel rows beside the network
and cluster probes' rows, the predicted step beside the measured"; "cluster: cli profile --cluster shows a token's hops
through a simulated pipeline"; "cluster: cli --world-size without --coordinator is a usage error naming it"; "cluster: cli start finds this host's node, starts one process per worker with its rank, and stops all when one
fails"; "cluster: cli pretrain --cluster on a one-worker node needs no rank; on a two-worker node it names the
missing --rank"; "cluster: cli tune --cluster with IDRAK_RANK keeps --rank as the adapter's rank"; "cluster: cli a
world size that contradicts the file is refused naming both"; help within 120
columns; the four `cli` health filters and `cli arabic` still pass.

## Memory and speed, where every byte goes

| Where | What is held | Bound by | Measured with |
|---|---|---|---|
| Gradient reduction | the gradients themselves, reduced in place slice by slice; `InFlight + 1` staging slots per direction of `ChunkBytes`, allocated once per group (rule 77) | `ChunkBytes` (measured or set) × `InFlight` | `GetMemoryUsage().InUse/Peak` per rank; `CollectiveCompleted.Bytes` |
| Buckets during backward | one bucket in flight at a time beyond the slots | `BucketBytes` | `CollectiveCompleted.Blocked` against the step |
| bf16, top-k | one packed chunk per in-flight chunk; the error-feedback residual per parameter, allocated once | the same, plus one float per trained value | `Peak` |
| Fake devices | each its own pool under its own cap; past the cap, pinned host memory | the declared cap | `Offloaded` per fake device |
| Sharding | 1/W of the moments, gradients or weights per rank | `Shard` | `InUse` per rank against the share |
| Pipeline | a stage's blocks and `MicroBatches` activations | `MicroBatches`, the stage's cap | each rank's `Peak` |
| Simulator | pipe buffers rented and returned; the event log in blocks flushed to disk | the in-flight count | the test's `Stress.Run` |

Speed: every transfer and collective reports wall and simulated time and the time blocked on the slowest rank (rule
78), cold and warm apart (76); the virtual clock makes a slow link's cost a number before the link exists; the
calibration keeps the prediction honest against every measured run.

## Order and parallelism

C0 → C1 → C2 and C3 in parallel (two agents: C2 is `Idrak.Nlp` and core over `sim`; C3 is `Idrak.Network` and
`Idrak.Cluster` only) → C4 → C5 (localhost tests here; the LAN and rental runs are the owner's) → C6 → C7 → C8 (the
training options can land with C2; `cluster init|inspect|sim` with C1; `cluster start` and `bench` with C5;
`calibrate` after C5's bench). Each phase is one
commit series on `abstraction` with its "as built" note appended here. Every step: the filter green on the simulator
with mixed CPU, CUDA and Vulkan fake workers before the next starts.

## Decisions (for the owner)

1. **Decided by the owner (2026-10-18): `Idrak.Cluster` references `Idrak.Abstraction`** and implements the contracts
   itself; no bridge, no duplicate types; on extraction the contracts and used types move with it.
2. **Decided (2026-10-18): the topology file is JSON** with the spec's keys; no YAML reader.
3. **Decided (the owner, 2026-10-18): `--rank N` for this process's rank and `--world-size N` for the world size**,
   with `--coordinator HOST:PORT`: the words anyone from another cluster tool already knows, matching `IDRAK_RANK`,
   `IDRAK_WORLD_SIZE` and `IDRAK_COORDINATOR`. An option's meaning is scoped by its command: on `tune` and `distill`
   `--rank` stays the LoRA rank, nothing existing is renamed, and those two read the cluster rank from `IDRAK_RANK`
   when `--cluster` is given. "Node" stays the topology file's word for a machine; a rank is one worker on one
   device, numbered from zero across the nodes in file order.
4. **Decided (2026-10-18): the wire type defaults to float32, and the library measures and suggests.** A `"cluster"`
   `ISettingAdvisor` registered in `SettingAdvisors` by `Idrak.Network`: during the first warm steps it measures the
   network share of the step (`CollectiveCompleted.Blocked` and `Wall` against the step), and on a sample slice the
   loss of precision bf16 would cause (the slice reduced both ways, the largest relative difference); when the share
   is above `IDRAK_CLUSTER_SUGGEST_SHARE` (default a fifth of the step) and the bf16 error is within
   `IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE` (default the fixture's measured tolerance), it suggests `--wire bf16`; by the
   same measurements it suggests `--local-sgd K`, gradient accumulation, `--bucket-bytes`, `--topk` or the
   hierarchical collective when the topology has more than one node and `ring` is set. Each suggestion is a
   `SettingSuggestion` (setting, value, reason with the numbers, device, source) through `Telemetry.Suggestion`, shown
   once on the progress line, listed by `idrak suggest`, and never applied on its own.
5. **Decided (2026-10-18): `udp` stays out of C5**; the name stays registered with no library default.
6. **Decided (2026-10-18): no presets.** The topology file is built by `idrak cluster init`, an interactive builder
   with a description for every input and a default found on this machine, with options for every answer so it also
   runs without a terminal.
7. **Decided (2026-10-18): memory sizes in the topology file accept `B`, `KB`/`KiB`, `MB`/`MiB`, `GB`/`GiB`, `TB`/`TiB`,
   all as powers of 1,024** (device memory is counted that way, and `"16GB"` must mean the whole card); `inspect` prints
   MiB beside the declared value, as `devices` does.

## Runs

Dated, newest last: date, commit, target (sim | localhost | LAN pair | rental), topology, what was measured, predicted
step time, measured step time, the calibration delta written.

| Date | Commit | Target | Topology | Measured | Predicted | Delta |
|---|---|---|---|---|---|---|
| | | | | | | |
