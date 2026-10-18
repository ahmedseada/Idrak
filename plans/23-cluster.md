# Plan 23: Idrak.Cluster and Idrak.Network (a cluster simulated on one GPU; real transports measured on real links)

**Status (2026-10-18, `abstraction`):** planned 2026-10-18 from the owner's spec (plan 23, draft 2) against fec56e1.
Every phase is built and merged: C0-C7, C8a, C8b, C8c (the setting advisor, decision 4) and C9. Each phase's "as built" note below has the details; the owner's GPU runs are listed per phase.

| Phase | What | State |
|---|---|---|
| C0 | Contracts, registries, telemetry, conformance kit, PyTorch DDP fixture | Done (f5b9e62); CPU, CUDA, Vulkan pass |
| C1 | In-process simulator, carved devices, virtual clock, `naive`/`ring`, `cluster init`/`inspect` | Done (75a472a); CPU, CUDA, Vulkan pass |
| C2 | Data parallel on the simulator, `cluster sim` | Done (12f1ba8); CPU, CUDA, Vulkan pass |
| C3 | Hierarchical all-reduce, transport selection, faults, replay | Done (92401af); CPU, CUDA, Vulkan pass |
| C4 | Overlap, measured buckets, compression, local SGD, faults in training | Done (dcb68ec); CPU, CUDA, Vulkan pass |
| C5 | Real transports, localhost mode, `cluster start`, `cluster bench` | Done (dcb68ec); CPU, CUDA, Vulkan pass |
| C6 | ZeRO sharding, flow control, GPU staging, shared-memory wake-up | Done (cb514b4); CPU, CUDA, Vulkan pass (the 262,148 B fixed in C8b) |
| C7 | Pipeline parallelism | Done (00a0d8f); CPU, CUDA, Vulkan pass |
| C8a | `replay`, `calibrate`, `transports`, `--profile`, `--cluster` on the training commands | Done (7c41b8d, c3dbe13); CPU, CUDA, Vulkan pass |
| C8b | Own-tensor loads and saves, device-to-device activations, direct wake, train/tune stops | Done (e016c9f); CPU, CUDA, Vulkan pass |
| C8c | The `"cluster"` setting advisor (decision 4) | Done (a36bfd1); CPU passes, owner's GPU run pending |
| C9 | The dashboard (`idrak cluster ui`) | Done (8e4b84c); CPU, CUDA pass |

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

**C3 as built (2026-10-18).** `src/Idrak.Network/Collectives/HierarchicalCollectives.cs`: `hierarchical`, registered
beside `naive` and `ring`. An all-reduce reduce-scatters inside each node (a ring of the node's ranks over its link),
runs one ring across the nodes per column of values with one rank per node (the owner of that column's slice in each
node), then all-gathers inside; columns are cut at every node's slice boundary, so nodes of different sizes work. Each
sum is made in an order fixed by the topology and the sizes (inside the node in ring order, then across nodes in ring
order), so every rank and every rerun hold identical bytes, within float rounding of ring, exact on whole numbers.
Reduce-scatter, all-gather and broadcast are two-level too (cross-node pieces cut where the destination nodes' slices
begin, then delivered inside the node; all-gather the same backwards; broadcast a chain of one rank per node, then a
chain inside each). The ring steps moved into `HostCollectives` as `RingScatter`/`RingGather` over any list of ranks, and
`ring` runs on them with the same operations in the same order. `ClusterOptions.CollectiveFor(topology)` makes
`hierarchical` the default with more than one node, `ring` on one. `TopologyInspection.PredictAllReduce` plays `ring`,
`naive` and `hierarchical` message for message (the algorithms' pieces, tags, message numbers and chunks), so a
prediction equals a run's simulated time even with jitter or loss in the profiles; `idrak cluster inspect` prints all
three with the default marked and the transport per pair (grouped; per pair in `-j`). Selection: `TransportPlan`
(`src/Idrak.Abstraction/Cluster/Selection.cs`) asks every registered transport for its availability per pair, lets the
selector choose (a forced `ClusterOptions.Transport` wins for every pair it can join, and is refused naming the pair and
its reason where it cannot), and publishes `TransportChosen` once per pair; `InProcessCluster` starts through it.
Failures (`src/Idrak.Cluster/Failures.cs`): `none`, `slow-worker`, `dropped-worker`, `link-loss`, `partition`,
`late-join`, `mixed` (`BuiltInFailures`), drawing with `FailureSettings` (`IDRAK_CLUSTER_FAULT_RATE`, `_SLOW_FACTOR`,
`_FAULT_DURATION`, `_LOSS_SHARE`, `_LATE_STEPS`). The injector (`Faults.cs`) draws step s's faults once, from a
generator seeded with (seed, s) alone, logs each and publishes `FaultInjected`; a rank applies them as it begins the step
(`ClusterRank.BeginStep(step)`), each state change logged and published as `RankStateChanged` with its seed and step; the
virtual clock asks it, per transfer, what the sender's active faults do (a slow worker's transfers stretched by its
factor, a lossy link's extra loss with each lost message sent again, a partition's hold). A dropped worker breaks the
group at once: every rank's step ends naming it. A partition for the rest of the run, or for the timeout or longer, fails
at once on every rank with a `TimeoutException` naming both sides (the virtual clock knows the message would never
arrive in time); a shorter one holds the messages across and heals (`Partitioned`, then `Rejoined`). A late joiner sits
out its first steps while the others run them as a group of their own over the same clock (`GroupTopology`, the
collectives behind `ClusterRank.Collectives` switching groups); when the group changes its ranks meet at a barrier, so
the joiner's clock joins theirs. On a simulated transport `CollectiveCompleted.Blocked` is the virtual clock's wait and
`SlowestRank` the origin of the critical path (`ITransport.SimulatedCriticalRank`), so a slow worker is named on every
rank, the same on every run. `EventLog.cs`: a binary log (`BinaryPrimitives`) of the faults, each rank's clock as it
begins every step and as it finishes, its state changes and its transfers (`IDRAK_CLUSTER_LOG_TRANSFERS`), kept per rank
in pooled 64 KiB blocks and joined rank by rank, so a run of a seed writes the same bytes every time; on
`ClusterRunResult.Log`, and in the file `IDRAK_CLUSTER_EVENT_LOG` names. `Replay.cs`: `ClusterReplay.Run` reruns the work
with a seed through a step (`SimulationOptions.ReplayStep`: every rank sits out the later steps) and
`ClusterReplay.Schedule` gives the fault schedule without running. Tests: 11 in `ClusterTopologyTests.cs`
(`ClusterTopologyGroup`); the C3 filter with the C0 and C1 tests, the API guard, the inventory, the CLI health, polish,
Arabic and trace checks and the profiler: 48 passed, 0 failed on the CPU. Hierarchical against ring, 1 MiB a rank on two
machines of four fake devices (`nvlink-900g` inside, `ethernet-1g` between): ring 14.680 ms, hierarchical 2.111 ms of
simulated network time, ratio 6.95.

Changes from the plan, kept: **`TransportPlan` is in Idrak.Abstraction**, not `Idrak.Network/Selection/`: the in-process
cluster (Idrak.Cluster) and C5's launchers (Idrak.Network) both start groups and the packages do not reference each
other; `FastestSelector` moved to `src/Idrak.Network/Selection/FastestSelector.cs`, and `ITransportSelector` and
`TransportSelectors` left `shared-contracts.txt` (Abstraction uses them now). **Steps are `ClusterRank.BeginStep`**: faults
are drawn and applied at step boundaries, so work that begins no step runs without faults (C2's training loops call it at
the start of each step). **Fault semantics fixed**: a duration is simulated time from the start of the step, on the
affected rank's (for a link, the sender's) clock, zero for the rest of the run; a late join is drawn at step 0 only;
`mixed` draws the kinds a run survives (slow worker, lossy link, late join), a drop and a partition only by name.
**`RankStateChanged` gained `Seed`** (optional, last), written by the console and JSON-lines loggers, so every fault and
state change carries the replay key. **New settings**: `SimulationOptions.CarveFrom` (`IDRAK_CLUSTER_CARVE_FROM`: every
fake device carved from one device here, so a topology written for other machines, the test's `cuda:0` to `cuda:3` per
machine included, runs on this one), `.EventLog` and `.LogTransfers`; the five failure settings; all in the CLI's table.
The "beats ring" test names distinct devices per machine because the reader joins ranks naming one device by
`same-device`, which would replace the `nvlink-900g` link. **Contention stays per pair of ranks and direction** (as in
C1): a node's network adapter shared by its ranks is not modelled, because a shared lane charged from several threads
would depend on their order and break the bit-for-bit replay; the 6.95 above is the win with an adapter per rank pair
(were the four rings across the nodes to share one adapter per machine, the same arithmetic gives about 1.75). An event-ordered model of shared
adapters moves to C5, where the LAN measures them. The event log is joined at the end of a run (in memory until then,
`IDRAK_CLUSTER_LOG_TRANSFERS=0` keeps it to clocks, faults and states), not flushed while it runs: flushing moves to C8
with `cluster sim`'s long runs. Moved: "rank 0's log holds every rank's per-step summary in localhost mode" to C5 (the
localhost cluster); "runs show charts the network share" to C8 (`cluster sim` runs training through C2's hooks, and
`runs show` reads its log); the `idrak cluster replay` command to C8 (it reruns a training step; C3 gives the
`ClusterReplay` API); a group over several transports at once (a routed transport) to C5, with the real transports; until
then an in-process run whose plan mixes transports is refused naming them.

**After merging C2 and C3 (2026-10-18).** Open: training on the simulator does not yet call
`ClusterRank.BeginStep(step)`, so `idrak cluster sim` runs without injected faults. A rank that sits a step out (a late
join, a replay's end) changes the group the gradients are summed over and the token weighting, so the trainers'
step loop needs to take part in that, not just the launcher; it lands with C4, which reworks the same step. **Closed by
C4** (below): the Trainer, the fine-tuner and the pretrainer begin every step through `IGradientSync.BeginStep`, so
`idrak cluster sim` (now with `--failures` and `--seed`) runs with the faults injected, and the Trainer, sharing a `DataLoader`'s steps among the ranks that take part, is exact
under a late join too (another batch source refuses a rank that sits out, at once and naming why).

### C4: overlap, buckets, compression, local SGD

**Files.** `Training/GradientBuckets.cs` (a hook per parameter at the end of its backward; buckets of `BucketBytes`
reduced while the backward continues; not inside a replayed graph: a recorded step reduces after the replay and the
telemetry shows the lost overlap), `Compression/` (`bf16`, `top-k` with error feedback), `LocalSgd.cs` (sync every *k*
steps, the average of the weights, not the gradients).

**Tests**: "cluster: bucketed overlap equals the unbucketed step"; "cluster: overlap shortens the simulated step on a
slow link" (the simulator's number, asserted); "cluster: top-k with error feedback converges on the Spirals sample within
the full-gradient loss after N steps" (the tolerance measured on the CPU and written into the test name); "cluster:
local SGD every k steps equals one process at k = 1 and runs k times fewer collectives".

**C4 as built (2026-10-18; merged with C5 at 378a7c1).** Overlap: `Autograd.OnGradientReady(Action<Tensor, bool>)` (Idrak.Abstraction) calls a
listener on the training thread for each parameter right after the last step of the backward pass that reads it (the
second argument false inside a pass nested in another's, a checkpointed block's); the backward loop pays for it only
while someone listens. `IGradientSync` gained `BeginStep(step)`, `Overlap(tokens, normalized)` (a scope the trainers put
around the step's last `Backward()`) and `OverlapLost(reason)`, as default members. `src/Idrak.Network/Training/`
`GradientBuckets.cs` holds `BucketLayout` (pieces packed in reverse parameter order, the order a backward pass finishes
them), `OverlapWorker` (a thread per rank reducing queued buckets in bucket order) and `BucketSizing`;
`GradientReduction.cs` ("all-reduce") sends each bucket whose parameters are all finished while the backward pass goes
on, through one host buffer of the gradients allocated once, and the rest, then a header (tokens, loss, a presence flag
per parameter, a changed flag per bucket) at `Reduce`; without overlap the header goes first and each bucket is reduced
and written back through one pooled buffer. Every bucket is this rank's values times its tokens, summed, then divided by
the step's tokens, in both modes, so overlap on and off give the same bits (rule 71, every world size and algorithm). A
gradient that changes after its bucket went (a weight shared into a checkpointed block) marks the bucket; every rank
learns it from the header and sums it again, whole and uncompressed. Unset `BucketBytes` is measured once on the first
step (16 MiB buckets as the first probe): the backward pass's time, the latency (a tiny collective timed after another
that takes what is still queued on the links) and the slowest bucket's time, the largest of each over the ranks, choose
the bucket count with the least predicted wait after the backward pass (at most 256, a bucket kept to what the link
moves in a hundred latencies). The latency counts each collective's per-message cost beyond its bytes: the wire headers
(36 bytes a message, 2(W-1) messages a ring) and the transport's own work per message, which on a real transport is in
the timed latency and on the simulator (whose built-in links carry no latency) is the wall time of the tiny probe here,
agreed over the ranks and kept for the process (`ClusterOptions.MessageCost`, `IDRAK_CLUSTER_MESSAGE_COST` in seconds,
pins it); without it the sizing chose 2 KiB buckets (256 of them) for 520 KiB of gradients on the simulator, with it 52
KiB (10). On a simulated transport the backward pass is not timed (taken as long as the gradients' transfer), so the
choice comes from the virtual clock and the kept per-message cost alone and a seed's runs in a process stay the same bit
for bit (a replay in another process pins the cost with the setting). A recorded
fine-tuning step (a CUDA graph runs no hook) reduces after the replay and says so (`OverlapLost`). Telemetry: one new
event, `GradientsSynced` (sync, compression, rank, step, buckets, how many went during the backward pass, bucket bytes,
bytes, backward, network and exposed time, the lost overlap's reason), carried by the console, `MetricsRecorder`
(`Synchronisations`), the channel, JSON lines and `TelemetryJson`; on a simulated transport "exposed" is predicted from
the backward pass's measured time and each bucket's simulated time. Compression: `IGradientCompression`,
`IGradientCompressor` and the registry `GradientCompressions` in `Idrak.Networking.Abstractions` (only Idrak.Network uses
them) with "none", "bf16" (`Compression/Compressions.cs`) and "top-k" (`Compression/TopK.cs`: per bucket, in blocks of
2^24 values so an index travels exactly as float32, the k largest by magnitude found by a selection pass, ties to the
lower index, sent as (index, value) pairs every rank all-gathers and sums in rank order; error feedback keeps the rest in
a residual of one float per trained value, allocated once per parameter); `ClusterOptions.Compression`
(`IDRAK_CLUSTER_COMPRESSION`; unset: top-k with `TopK`, bf16 with `Wire=bf16`, else none). Local SGD: "local-sgd"
(`Training/LocalSgd.cs`) beside "all-reduce": each rank steps on its own gradients; every `LocalSgdEvery` steps the
weights are averaged, each rank weighted by its rows since the last average; `GradientSync` unset picks it when
`LocalSgdEvery` is set (`ClusterOptions.GradientSyncName`). The shared part of both (`Abstractions/SyncCore.cs`): the
start, the weight sync, the speed split, and the step groups. Step groups: `IStepGroup` and `IHostValueCollectives`
(Idrak.Abstraction: Idrak.Cluster implements them, Idrak.Network uses them); the in-process rank's collectives implement
both; when a step's group changed, the ranks take the trained state of a rank that trained the step before (the
synchronisation's bucket size, the weights, and the optimizer's state through `SaveState`/`LoadState` in two broadcasts),
the shares go even over the group and the speed measurement starts again; a rank that sits a step out runs no collective.
Trainers: each step begins through `BeginStep`; a rank that sits out keeps its draws, schedule and step count alike and
trains nothing; the fine-tuner and the pretrainer share each step's draws over the ranks that take part (so a late join
still trains every draw, token-weighted: the run equals the run without the fault), and the step's group's first rank
writes its checkpoints. The Trainer does the same for a `DataLoader` over a sample source (`DataLoader.StepShares`: every
rank draws each step as one process with batches W times as large and keeps sample i of the step when i mod (ranks taking
part) is its place; with every rank taking part, the samples C2's `Shard` gave it, in the same order), so every rank runs
the same steps an epoch, a late join across an epoch's end trains every sample and equals the run without the fault, and
no rank waits on another; another batch source (a stream, a source of one's own) keeps C2's sharding by epoch, and a rank
of it that sits a step out (other than past a replay's last step, where every rank does) fails at once naming why, which
ends every other rank at once (its pipes close) instead of a stall. `idrak cluster sim` gained `--failures MODEL` and `--seed S` (registered value options; `--seed`
was read in C2 but not declared) and lists the faults injected with their replay key (text and `-j`). New settings:
`Overlap` (`IDRAK_CLUSTER_OVERLAP`, 1), `Compression` (`IDRAK_CLUSTER_COMPRESSION`) and `MessageCost`
(`IDRAK_CLUSTER_MESSAGE_COST`), in `ClusterOptions.Variables` and
the CLI's table; `BucketBytes`, `LocalSgdEvery` and `GradientSync` described anew. Tests: 11 in
`tests/Idrak.Tests/ClusterOverlapTests.cs` (`ClusterOverlapGroup`, before C5's `ClusterTransportGroup`). After merging C5
(378a7c1), one build and one run of `cluster:|public API|abstraction inventory|cli health: …|cli polish: …|cli arabic:
every message|cli dev: trace|profiler:|fine-tuning:|fine-tuning options:|pretrain:|cli pretrain:|training:|callbacks:|
offloading: HostOptimizer` on the CPU: 139 passed, 0 failed, no child process left. Measured on the CPU (4 cores):
overlap on `ethernet-1g`, 2 ranks, 520 KiB of gradients in 9 buckets: exposed network time 3.52 ms a step against 4.26
ms without (17.4% shorter; the backward pass measured 1.7 ms); the measured bucket size there 52 KiB (10 buckets); top-k 10% with error feedback on the Spirals
sample after 20 epochs: training loss 0.3703 against the full gradient's 0.3536 (gap 0.0167, the test holds it to 0.02),
1,810 KiB sent against 8,951 KiB (without error feedback 0.3680, so error feedback did not help on this small run); local
SGD: 24 weight averages at k = 1, 6 at k = 4 over 24 steps; slow worker (x8) in the pretrainer: ranks 0 and 1 blocked
0.32 ms of simulated time against 0.05 ms without.

Changes from the plan, kept: **the hook is a listener on autograd, not a module hook**: a parameter is finished when the
last step of the pass that reads it has run, which a module cannot see (a tied weight, a weight read by two layers).
**Every bucket is weighted by the rank's tokens and divided by the step's after the sum** (C2 weighted later buckets by
tokens / total before they travelled): overlap sends buckets before the step's tokens are known, and the same arithmetic
in both modes is what makes them equal bit for bit. **The header is its own small collective** (C2 put it in the first
bucket on a float32 wire): one more latency a step, the price of identical buckets in both modes. **Bucket order is
reversed parameter order** (C2: parameter order). **Compression's contract lives in Idrak.Networking.Abstractions** and
the `Wire` setting stays (it implies "bf16"). **`HostCollectives` decides "simulated" by the transport's clock having
moved**, not by the call's own simulated time: a tiny header that took no simulated time had its blocked time taken from
the wall clock. **Overlap is off for the HostOptimizer** (it downloads the gradients itself after the backward pass) and
behind a plug-in's device-only collectives; the telemetry names why. **Local SGD still sums a two-float header every
step** (the step's loss over every rank, and the epoch's end for ranks of unequal shares); "k times fewer collectives"
counts the weight averages. **The Trainer draws a `DataLoader`'s epoch step by step under a cluster** (C2 sharded it by epoch): exact under a changing
group, without the prefetch of the next batch (a step's share is known only as the step begins); other batch sources keep
the epoch's sharding and refuse a rank that sits out. After a join the Trainer's batch-norm statistics are rank 0's again
at the epoch's end (C2's broadcast), not at the join. A rank's own loss list holds the steps it trained.
**Out-of-process ranks (C5's `LocalhostCluster`, `cluster start` children, tcp and shared memory) take part in
`BeginStep` as a fixed group**: their collectives are not an `IStepGroup`, so the synchronisation lets every rank take part
until `SimulationOptions.ReplayStep` (`IDRAK_CLUSTER_REPLAY_STEP`, inherited by the children), past which every rank sits
out alike, as `ClusterRank.BeginStep` already did; faults stay the simulated transport's (C3), so `cluster sim --mode
localhost --failures …` is refused naming why. Overlap, compression, local SGD and the measured bucket size run over the
real transports unchanged (the worker thread uses the transport while the training thread does not), and C5's per-step
summaries are sent to rank 0 from `SyncCore.Stepped` for both synchronisations, by the ranks that took part in the step. Not run here: the CUDA check that a
replayed fine-tuning step names the lost overlap (`SupportsGraphs` only) and the GPU runs of the whole filter (the
owner's). Open: overlap with device-side staging copies (`IHostStaging`) for GPU gradients (C5 landed without it; here a
finished bucket is downloaded on the training thread, so on a GPU each launch waits for the device; for C6, which reworks
the gradient pieces), the `--profile` of a step's buckets (C8), and
persisting the measured bucket size in the tuning cache (C8, with `idrak tuning show`).

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

**C5 as built (2026-10-18).** Merged C2 (12f1ba8) first, so C5 builds on the training hooks. `src/Idrak.Network/Transports/`:
`SharedMemoryTransport.cs` (`shared-memory`: one memory-mapped ring per sending and receiving rank, `SharedRing`, a file of
`IDRAK_CLUSTER_SHM_BYTES` (4 MiB) after a 128-byte header under a folder named after the run in `IDRAK_CLUSTER_SHM_FOLDER`
(the temporary folder); the two positions only grow, each on its own cache line, written after the bytes they cover; a
message larger than the ring streams through it; one reader thread per rank spins, yields, then sleeps a millisecond while
idle and checks the writers' processes once a second), `TcpTransport.cs` (`tcp` and `direct-link`: a socket per pair, Nagle
off, `IDRAK_CLUSTER_SOCKET_BUFFER`, a reader thread per socket, frames of up to 64 KiB written in one call), `Rendezvous.cs`
(rank 0 listens at the coordinator; every other rank listens on `IDRAK_CLUSTER_PORT`, dials rank 0 with
`IDRAK_CLUSTER_RETRIES` attempts, the wait doubling from `IDRAK_CLUSTER_RETRY_DELAY`, gets a welcome with every rank's
address, dials the ranks below and takes the ranks above; the hello is a challenge each way: the acceptor's nonce, the
HMAC of it and a nonce of the connector's, the HMAC of that back, so a recorded hello is worth nothing; a failed hello gets
an error frame naming why, never the key; without a key nothing listens or dials beyond loopback),
`PciHostTransport.cs` and `NvlinkTransport.cs` (`pcie-host`, `nvlink` and its other name `p2p`: ranks of one process,
frames handed over by a call, device memory handed over through `IDeviceTransport` and copied by the receiver: through
host memory, or `Backend.CopyTo` where the devices reach each other; a refused peer copies through the host and publishes
`TransportChosen(pcie-host, reason)` once per pair). Their shared bases are in `Idrak.Networking.Abstractions`
(`FramedTransport`: frames of `WireFormat`, one unbounded queue per sending rank and tag fed by the readers so a send never
waits on a peer that is itself sending, receives with the timeout, a barrier of frames to rank 0 and back, a bye on
dispose, a carrier closed without one ending every wait on that rank at once naming it; `LocalTransport`;
`IDeviceTransport`). `Bench/LinkBench.cs`: every pair one at a time (a barrier before each), half the round trip of an
empty message and a message of `--bytes` with its answer, the first round apart from the median of the warm ones; rank 0
gets the rows. `src/Idrak.Cluster/LocalhostCluster.cs` (`Run`: every rank of a file on this machine, or one node's ranks;
each process given `IDRAK_RANK`, `IDRAK_WORLD_SIZE`, `IDRAK_CLUSTER_TOPOLOGY`, `IDRAK_CLUSTER_NODE`, `IDRAK_CLUSTER_DEVICE`,
`IDRAK_COORDINATOR`, `IDRAK_CLUSTER_RUN`, `IDRAK_CLUSTER_PARENT`, `IDRAK_CLUSTER_MODE=localhost`, the key inherited; lines
streamed; the first rank to fail ends the others; every process left is killed in `finally` and when this process exits;
`Join`: a process as one rank, its device carved with its entry's cap when it declares one, the plan's transports, the
throttle, the collectives; a rank whose launcher is gone ends itself) and `LinkModel.cs` (`TokenBucket`,
`ThrottledTransport`: in a localhost simulation each pair whose link declares a rate leaves at that rate and is held for
its latency, the arrival time travelling in 8 bytes ahead of the payload). `Backend.CopyTo(source, destination, count)` and
`CopiesDirectlyTo` (default: a copy inside the backend, else down and up through one pooled host buffer); CUDA overrides
them (`CudaBackend.Peers.cs`: a device copy between two backends of one card, `cuMemcpyPeerAsync` where
`cuDeviceCanAccessPeer` says yes, peer access enabled once per pair and remembered, else the default). CLI: `idrak cluster
start --topology FILE [--node NAME] -- COMMAND` (this host's node, or `--node`; `idrak …` after `--` runs this tool; one
live line per rank in a terminal: its device, state, time, lines and their rate, and its last line, the command's own
progress and time left; "rank N: …" lines otherwise), `idrak cluster bench --topology FILE [--bytes N] [--rounds N] [--node
NAME | --localhost]` (each machine starts its node's ranks, which run the bench; rank 0's machine prints the table, `-j` the
rows), `idrak cluster sim --mode localhost` (C2's deferral: every rank a process of this tool, rank 0's output shown, one
line per rank), and `pretrain` (and `resume`) join the cluster they were started for (`ClusterSession.JoinStarted`), so
`cluster start -- idrak pretrain …` needs nothing more. Rank 0's `log.jsonl` holds every rank's per-step summary where the
ranks are processes of their own: `HostCollectives` keeps the step's totals (bytes each way, wall, blocked, the peer waited
on longest) when its transport crosses processes, the all-reduce sync's `Stepped` sends them to rank 0, which publishes a
`CollectiveCompleted` per rank with the operation `step-summary`, and pretrain's log writes those. Six settings joined
`ClusterOptions` and the CLI's table: `Device` (`IDRAK_CLUSTER_DEVICE`), `RunId` (`IDRAK_CLUSTER_RUN`), `Parent`
(`IDRAK_CLUSTER_PARENT`), `RetryDelay` (`IDRAK_CLUSTER_RETRY_DELAY`, 1 s), `SharedMemoryBytes` (`IDRAK_CLUSTER_SHM_BYTES`),
`SharedMemoryFolder` (`IDRAK_CLUSTER_SHM_FOLDER`). Tests: 12 in `tests/Idrak.Tests/ClusterTransportTests.cs`
(`ClusterTransportGroup`; child processes are the test program again with `--cluster-child`, or the CLI); the C5 filter with
C0 to C3, the API guard, the inventory, the CLI health, polish, Arabic and trace checks and the profiler: 73 passed, 0 failed
on the CPU. The throttle, `ethernet-200m` (25.0 MB/s) between two processes over tcp, 4 MiB messages, three runs: warm 24.02 to 24.37 MB/s
(at most 3.9% under), cold 14.5 to 22.9 MB/s (the first round pays for connecting and compiling, rule 76). A rank that
leaves removes every ring it used, its peers' included, so a rank that died leaves nothing behind once one peer leaves.

Changes from the plan, kept: **one group over several transports runs** (C3 refused it in-process): `TransportPlan.Connect`
connects each carrier once for the ranks its pairs join and routes every message to its pair's (`RoutedTransport` and
`SubsetTopology` in `src/Idrak.Abstraction/Cluster/Routing.cs`); a carrier over a part of the group sees those ranks as a
group of their own and a run name of its own; `ITransportProvider.Carrier` (a default member: the name) lets `direct-link`
ride on the `tcp` mesh, so a file with a cable for one pair and the LAN for the rest connects one set of sockets; a network
carrier whose ranks leave out rank 0 is refused (its rendezvous needs the coordinator), which a topology of two or more
nodes never makes. `InProcessCluster` runs a mixed plan the same way. **The launcher picks the coordinator's port**, not rank
0 reporting it on its output: in a mixed group rank 0 meets its node's shared-memory peers before it listens, so every rank
must know the port before any starts (a loopback port the system gives free; ranks of one process given port 0 still share
the system's choice). **Whether every rank is on this machine is told by the coordinator being on loopback**, not by
`IDRAK_CLUSTER_MODE`: a process of `cluster start` is also one process per rank (the mode says that), and its pairs between
machines need the key and may take a cable; a localhost simulation (coordinator on loopback) needs no key, takes no cable
and is throttled. `cluster start` sets `IDRAK_CLUSTER_THROTTLE=0`. **Receives are buffered**: every carrier's reader drains
into per-rank, per-tag queues (unbounded), so a ring's sends never wait on each other in a circle whatever the message and
socket-buffer sizes; `InFlight` still bounds the simulator only. **Shared memory is not zero-copy on the receive side**: the
reader copies a frame out of the ring into a pooled buffer (one copy) so the ring frees as soon as it is read; files and
polling instead of named maps and semaphores, which .NET does not offer on every system (rule 70). **The transports carry
bytes; the collectives still stage device buffers through the host**: `IDeviceTransport` and `Backend.CopyTo` give the device
path, and keeping a `GpuDirect` chunk on the device with the sum made there moves to C6, where sharding moves device memory
between ranks. **The localhost-against-in-process test is on real training**: `cluster sim --mode localhost` and in-process
`cluster sim` run `pretrain` 3 steps on two CPU ranks and the checkpoints' tensors are bit-identical; the collectives'
results of three groups of child processes (2 ranks on shared-memory, 3 forced onto tcp, 3 on two nodes routed over both)
are also bit-identical to the in-process run's. `TransportSuite` and `CollectivesSuite` run in one process on the real
transports (2 to 8 ranks, rings smaller than the messages) and on a routed group; in child processes the same exchange and
every collective run per rank. **Not done, with the reason**: the measured `ChunkBytes` and its cache (moved here from C1)
moves to C8 with `calibrate` and `idrak tuning show`, which read the bench's numbers C5 now produces; one network adapter
shared by a machine's ranks stays out of the virtual clock (C3's reason holds: a lane charged from several rank threads
depends on their order and breaks bit-for-bit replay; an event-ordered clock stepping every rank in virtual-time order is
a rewrite of `VirtualClock` and `SimTransport`), and moves to C8, where `calibrate` can fit a per-machine adapter share from
benches of concurrent pairs; `cluster bench` measures one pair at a time. A cable on a subnet of its own needs the node's
address in the file to be the cable's (an override has no address field yet: C8). `idrak transports` is C8's. Tests skipped
on this CPU-only machine: the copies between two physical GPUs (`cluster: nvlink falls back …` prints the skip); the
owner's CUDA run checks two backends of one card copying device to device, and peer access where a machine has two cards.

### C6: sharding (ZeRO-style)

`Shard = optimizer` (each rank keeps the moments of its slice; reduce-scatter, update the slice, all-gather the weights),
then `gradients`, then `parameters` (each rank holds its slice of the weights and gathers a layer's weights just before
its forward and backward, freeing them after). The fake GPUs' hard caps are what make this testable on one card: a
model that does not fit one fake GPU's cap fits across four.

**Tests**: "cluster: sharded optimizer state equals the unsharded weights after 3 steps"; "cluster: a model over the
cap of one fake GPU trains across four with `Shard = parameters`, each rank holding its share" (`InUse` per rank against
the share); "cluster: a sharded run resumes at the same world size".

**C6 as built (2026-10-18).** Sharding is a gradient synchronisation of its own, `"sharded"` in `GradientSyncs`
(`src/Idrak.Network/Training/Sharding.cs`, registered beside `all-reduce` and `local-sgd`), picked when `Shard` is set
(`ClusterOptions.GradientSyncName`; `all-reduce` and `local-sgd` refuse a shard naming the setting). It is the C4
reduction itself (`GradientReduction`, now partial): the same buckets, overlap, measured bucket size, header and telemetry.
Each trained parameter has one owner (`ShardOwners.Assign`: largest first to the rank owning the fewest values, the same
on every rank); each bucket's pieces are grouped by owner (`BucketLayout` with owners, `Counts`, `Piece`) and
reduce-scattered (`IHostValueCollectives.TryReduceScatter(values, counts, op, wire)` and `TryAllGather(values, counts)`,
new default members, in `HostCollectives` for every algorithm: ring around its ring with each rank's own count, naive and
hierarchical through their all-reduce and broadcasts; `HostValues` falls back through the device), so a rank receives the
token-weighted sum of its own parameters' gradients only. `IGradientSync` gained default members the trainers call:
`Updates`/`Update(maxNorm, readNorm)` (the global norm is every rank's squares summed, the owned gradients clipped, the
optimizer stepped: only owned parameters have gradients, so only their moments exist, as any optimizer makes a
parameter's state when it first steps it; then each bucket's weights all-gathered), `Gather()` (whole weights for an
evaluation, a checkpoint, a copy of the best weights; what is written meanwhile into a rank's own goes to its master),
`SaveOptimizerState()`/`LoadOptimizerState()` (every rank's share joined for rank 0's checkpoint, which is the format one
process writes, so it resumes sharded or not; a resumed rank keeps its share) and `End()` (the whole weights again after
the last step). `Optimizer.StateParameter(name)` (the parameter a saved entry belongs to; `GroupedOptimizer` maps its
groups) tells the shares apart. Modes: `optimizer` drops the others' gradients once reduced; `gradients` drops them as soon
as their bucket was taken during the backward pass (a gradient changed after it went, a weight shared into a checkpointed
block, fails naming the parameter and the fix); `parameters` keeps a master copy of the owned weights, gives every trained
parameter's memory back (`Backend.Evict`, its recomputation a broadcast from the owner) and gathers a layer's weights
just before its forward (`ForwardHooks`, a new thread-scoped listener in Idrak.Abstraction entered by `Module.Forward` and
by Idrak's fused paths: `Linear.ForwardMany`, the fused convolution with its norm, a tied head's table) and when the
backward pass reads them (the storage's restore in `Tensor.Backward`), giving them back after; a module whose trained
parameters fit `GatherBytes` (unset: each rank's share) or has no children is gathered whole, a larger one only its own;
the overlap worker is drained before each gather so the two never share the collectives at once. A changed group (a late
joiner) moves each share: every rank says whether it trained the step before and as which rank, the old owners are
recomputed, and the weights (owners' masters) and each moving parameter's optimizer entries go from the old owner to the
new; the step count comes from the holder. Trainers: the `Trainer` updates through `Update`, gathers for validation and
the best weights, ends the run; the fine-tuner's `Update` and the pretrainer's runner take the sync, record no CUDA graph
when sharded, gather around evaluations, answers and saves; the pretrainer joins the optimizer state for its checkpoints
and loads a resumed state after the sync exists (it writes no cancelled-step checkpoint when sharded: the state is every
rank's). New settings, in `ClusterOptions.Variables` and the CLI's table: `GatherBytes` (`IDRAK_CLUSTER_GATHER_BYTES`).
Tests: `tests/Idrak.Tests/ClusterShardingTests.cs` (`ClusterShardingGroup`, 8 tests after `ClusterTransportGroup`).
Measured on the CPU (4 cores, shared with another agent): sharded `optimizer`, `gradients` and `parameters` runs equal the
unsharded run bit for bit with naive (3 ranks, Adam, ten buckets sent during the backward pass), and each rank's optimizer
holds its own parameters' moments only; ring with AdamW clipped to 0.5 on 2 ranks, gap 0. A model of 462,083 values
(1.76 MiB of weights, 7.05 MiB of training state with Adam) on fake GPUs capped at 4 MiB: unsharded on one it fails
("would exceed its memory limit"), with `Shard = parameters` on four each rank's `InUse` at every epoch's end is exactly its
share (three times its owned values: master and two moments; 1.50, 1.50, 1.50 and 0.79 MiB), the same every epoch (rule
77), peaks 3.26 MiB and 2.55 MiB, and the weights after 3 epochs are the unsharded run's bit for bit. The pretrainer
(tiny Llama, AdamW, 2 ranks) resumes `optimizer` and `parameters` runs at the same world size with the uninterrupted run's
losses and weights, its checkpoint holding every parameter's moments; a late joiner (rank 2 at step 7) ends `optimizer` and
`parameters` runs with the run without the fault's weights.

Changes from the plan, kept: **shares are whole parameters, not flat slices**: the user's optimizer is used unchanged
(any optimizer, its state per parameter by `StateParameter`), the results equal the unsharded run's bit for bit where the
sums are ordered alike, and a checkpoint is one process's format; the cost is balance (a parameter larger than a share
sits on one rank) and a reduce-scatter whose pieces are as uneven as a bucket's owners (never more bytes than the
all-reduce). **A gathered unit is a module whose parameters fit `GatherBytes`**, not a fixed "layer": a decoder block is
gathered whole because its forward reads its norms' gains in fused passes. **`Gather()` gives every rank the whole weights**
for evaluation and checkpoints (the evaluations run unequal shares, which per-layer gathers cannot); a model whose whole
weights do not fit one device once cannot be evaluated or saved this way yet, nor built (each rank builds the whole model
before the first step evicts it): a sharded load and a streamed save are left for C8. **Every rank must run the step's
forward and backward under `Shard = parameters`** (the gathers are collectives): a rank with no rows in a step would leave
the others waiting until the timeout; the trainers' steps give every rank rows when a step has at least a draw per rank.
A forward on another thread than the training thread (none in the trainers today) is not gathered. Sharding refuses the
HostOptimizer (its state is on the host already), local SGD and top-k (bf16 travels).

**C4's open item, closed: background staging of a finished bucket.** On a device with pinned staging slots each bucket
the backward pass finishes is queued to a slot of its own (`IHostStaging.Download(source, offset, slot, slotOffset,
floats)` and `PlacesCopies`, new default members; CUDA implements both, its `Wait` makes the context current for the
reducing thread) behind the backward pass's work, and the reducing thread waits for it, copies it out and reduces it, so the
training thread never waits for the device; one slot per bucket (pinned memory the size of the gradients, as the held
buffer is), made once per layout. `AsyncStaging` (`IDRAK_CLUSTER_ASYNC_STAGING`, on) turns it off; the values are the same.
Measured with a staging whose copies complete 2 ms after they are queued (the test's, so the path runs without a GPU): 18
buckets a step for 4 steps, the training thread spent 1.87 ms queuing them, the reducing thread waited 5.42 ms for them;
the weights bit for bit the synchronous copy's. The CPU has no staging and copies on the training thread as before.

**C5's open items, closed: receive flow control.** `FramedTransport` gives each peer and tag a credit: a sender has at most
`InFlight` messages and `ReceiveBytes` (`IDRAK_CLUSTER_RECEIVE_BYTES`, unset: messages only) bytes the peer has not taken;
the receiver acknowledges each message as a receive takes it (an `Ack` frame carrying its bytes), a blocking send beyond
its credit waits (timeout naming the rank and the settings), an awaited send awaits without a thread. A message is always
let go when nothing of its peer and tag is outstanding, so a message larger than the budget travels alone and ranks that
all send before they receive never wait in a circle. Measured: 120 messages of 16 KiB to a receiver taking one a
millisecond held at most 16 KiB with 2 in flight and a 24 KiB budget on tcp, shared-memory and pcie-host, against 1.9 MiB
without a bound; three ranks each sending 1 MiB to every other under two tags (budget 4 KiB, one in flight) before
receiving all complete. **Device collectives.** Over a transport that moves device memory (`IDeviceTransport`: pcie-host,
nvlink in one process; `PostDevice`/`WaitTaken` hand a piece on without waiting, so a ring step posts, takes, then waits)
ring keeps every piece of a float32 sum, an all-gather, a reduce-scatter and a broadcast on the device: the piece handed
device to device, what arrived plus the rank's own added by the device's `Copy2D` (accumulate), the order the host path
sums in, bit for bit; two device buffers made on first need and kept. Elsewhere (a maximum, bf16, naive, hierarchical,
socket transports) chunks are staged through the host; `CollectiveCompleted.Staging` ("device", "host", or null for host
values) says which, carried by the console and JSON-lines loggers. `DeviceCollectives` (`IDRAK_CLUSTER_DEVICE_COLLECTIVES`,
on) turns it off. Chunking (`ChunkBytes`) applies to the host path only: a device piece goes whole.

**The shared-memory doorbell (the owner's Windows measurements, 2026-10-18).** The reader slept a millisecond between looks
after 5 ms of spinning, and so did a writer facing a full ring; on Windows a sleep is a timer tick (15.6 ms), longer than
the 5 ms spin, so once idle both sides stayed in the sleep regime: each message waited about half a tick (the bench's warm
latency 7,961 us), a 16 MB message stalled once per ring refill (warm 100 MiB/s, below the cold round's 199.5, which ran
while both sides still spun), and training through `cluster start` ran at 9.5k tok/s against 92k in-process. Now an idle
side raises a flag in the ring's header and blocks on a loopback datagram socket (`Doorbell`), which the other side rings
only when the flag is up (each side writes its position or flag, a full fence, then reads the other's, so no wake is
lost); it waits at most `SharedMemoryCheck` (`IDRAK_CLUSTER_SHM_CHECK`, 1 s: how soon a writer that died is named) and spins
first for `SharedMemorySpin` (`IDRAK_CLUSTER_SHM_SPIN`; unset: the cost of one cross-thread wake through the doorbell,
measured once per process, the spin-then-block break-even). Loopback datagrams work alike on Windows, Linux and macOS
(rule 70). Measured here (Linux, where a sleep of a millisecond takes 1.1 ms, so the old loop never stuck): a message
after an idle gap between two processes 334 to 418 us median against a timer tick of 1,090 us (the test holds it under half
a tick); `cluster bench --localhost`, 16 MB, before 1,913 to 3,159 MiB/s warm, after 1,894 to 2,525 MiB/s (latency 12 to 73
us before, 13 to 132 us after: the old 5 ms spin kept a core busy after every message, which bought those microseconds;
`IDRAK_CLUSTER_SHM_SPIN=0.001` gives 13 to 20 us); pretraining the tiny model 200 steps on two CPU ranks, mean of the
progress lines, before: in-process 36.6k to 40.9k tok/s, `cluster sim --mode localhost` 20.8k to 29.2k, `cluster start`
15.5k to 20.0k; after: 38.4k to 40.3k, 26.3k to 31.2k, 23.9k to 29.8k (the processes' remaining gap to in-process is two
processes' CPU kernels on four cores). Why `cluster start` was slower than `cluster sim --mode localhost` on Windows could
not be reproduced here: on Linux the two now run alike; the difference the owner saw is the sleep regime above (a
throttled localhost run's paced arrivals change when each side goes idle), which the doorbell removes; the owner's Windows
runs are to confirm it. Not changed: a blocked receive still wakes through the thread pool (the channel's continuation),
part of the 334 us; an inbox that wakes its waiter directly is left for C8, with the bench's calibration.

**Merged with C7 (2026-10-18, onto 00a0d8f).** Sharding and pipeline placement are refused together, at the start of
the run (`PipelineStage.Place`, the entry every pipeline run takes, before anything moves), naming
`IDRAK_CLUSTER_PLACEMENT=pipeline` and `IDRAK_CLUSTER_SHARD` and why: a pipeline already splits the parameters by stage, a
stage clips by the norm over every stage and rank 0's checkpoint gathers each stage's whole weights and optimizer state,
where a sharded synchronisation over a stage's replicas would keep each replica's share and clip over the replicas alone,
and the parameters shard's gathers would run between the stage-to-stage sends. Making them work together (each stage's
replicas sharding that stage through C7's subgroups, the clipping norm summed over stages and shares, the checkpoint
joining shares per stage) is left for after C8. A plain pipeline run is untouched by the gathers: `ForwardHooks` has no
listener unless `Shard = parameters` installs one, so `Module.Forward` and the tied table's entry are one thread-static
read, and `PipelineBoundary` holds no parameters. One build and one run of the C6 filter with `generation:|decoder:` added:
178 passed, 0 failed on the CPU (C7's pipeline tests included; the shared-memory test measured 508 us median after an idle
gap that run, 51 ms worst, the machine shared with another agent's build).

Open, named with their phase: the measured `ChunkBytes` (C8, as C5 said); a sharded load (each rank building only its
share) and a streamed checkpoint save for weights that never fit one device whole (C8); sharding with top-k (after C8's
advisor shows a need). Not run here: everything on a GPU (the CUDA staging, peer copies and the device path on nvlink, the
CUDA carve that spills instead of failing), and the shared-memory measurements on Windows (the owner's runs).

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

**C7 as built (2026-10-18).** Contracts (`src/Idrak.Abstraction/Cluster/Pipeline.cs`): `PipelinePlan` (a model's
top-level modules cut into contiguous stages, every module once, each stage at least one, embeddings first and the head
last by construction; `Balance` finds the cut whose largest stage, over its share, is least, by a dynamic programme over the
modules, ties to the earliest cut so every rank finds the same plan; `FromStarts` takes cuts by hand; `For` picks by the
options), `PipelineModule`, `PipelineStageRange`, `PipelineLayout` (stages of world / stages ranks each, stage by stage in
rank order: ranks 0 to R-1 are stage 0, so a stage's replicas sit together and only activations cross between stages),
`PipelineBalance`, `IPipelineSchedule` with `PipelineWork` and the unguarded registry `PipelineSchedules`, and
`ClusterGroups` (`Transport`: some ranks of a connected transport as a group of their own, a barrier on a tag below every
collective's; `Topology`: the C5 `SubsetTopology`). `ICollectives` gained two default members, `Transport` (for messages of
one's own on tags 0 and up) and `Context`; `HostCollectives` and the simulator's `SimStepper` give them. Idrak.Network
registers `gpipe` (`src/Idrak.Network/Training/PipelineSchedules.cs`). Idrak: `PipelineBoundary : Module, ICachedModule`
(`src/Idrak/Layers/PipelineBoundary.cs`): at a stage's end its forward is an `Autograd.Function` that sends the activation
and returns the stage's root, whose backward receives the gradient; at a stage's start the mirror (receive forward, send the
gradient in backward); one function per micro-batch, tags per micro-batch, float32 through one pooled host buffer over any
transport; without autograd the first stage's boundary runs generation (below). `PipelineStage`
(`src/Idrak/Training/PipelineStage.cs`): `PlanFor` measures any `Sequential`'s top-level modules (the distinct storages each
holds or reads, a tied head counting the table it reads) and cuts them; `Place` does that on every rank alike, moves this
rank's stage (and the table its tied head reads) to its device, releases every other module's tensors but never one the
stage reads (`keepOthers` keeps them where they were loaded, for the rank that writes whole checkpoints), cuts a block's
`NextNorm` hand-off when the next norm is another stage's, and with more ranks than stages makes the stage's replica group
(`ClusterGroups`, the collective `CollectiveFor` names). `Step(M, input, loss, options)` runs the schedule's order (GPipe's
nests each micro-batch's `TensorScope` so its activations go right after its backward pass; a schedule that does not nest
keeps one scope a step), sums a tied table's gradient over the stages holding it in stage order (the same bits on each),
gives every stage the last stage's loss, and publishes `PipelineStepped`; `Evaluate` (forward only), `SumStages`,
`ClipAndStep` (the norm over the whole pipeline, a shared table counted by its first holder, `Owns`), `Serve` and
`StopServing`. Telemetry: one event, `PipelineStepped` (schedule, rank, stage, stages, replica, micro-batches, step, wall,
blocked, bubble, the schedule's ideal bubble, bytes each way) at `Network`, carried by the console, `MetricsRecorder`
(`PipelineSteps`), the channel, JSON lines and `TelemetryJson`. Inference: rank 0's `PipelineStage.Network` is its modules
and the boundary, so a `TextGenerator`/`ChatGenerator` over it runs the whole pipeline; each cached step sends a 32-byte
header (the context's batch, capacity, length before the step, last-position-only, sliding steps, cache format, rows'
starts) and the hidden state; each later stage keeps its own `DecodingContext` in step (made for the header's shape, reset
or cut back with it, row starts mirrored), a middle stage passes every position on and the last applies last-position-only
and sends the logits back to the first; a recorded decode step is refused through a boundary (`TextGenerator`), a custom
cache layout and a cross-attention memory are refused by name. Training: **the pretrainer** (`src/Idrak.Nlp/PipelinePretraining.cs`,
`Pretrainer` is now partial): with `ClusterOptions.Placement` pipeline it places the model, draws every step alike on every
rank, gives each replica its share (`DrawShares` over the replica group), runs the step's micro-batches (`MicroBatches`, or
the pretrainer's own batches, packed where the model packs), each micro-batch's chunked token loss over the pipeline's
trained tokens on the last stage (the head run inside the loss), sums each stage over its replicas (`GradientSyncs` on the
replica group), clips by the pipeline's norm, and writes whole checkpoints from rank 0 (every stage's replica 0 sends its
parameters and optimizer state; the optimizer's under `pipeline.<stage>.`, the cut in the state file). Settings:
`Stages` (`IDRAK_CLUSTER_STAGES`, one per rank), `PipelineBalance` (`IDRAK_CLUSTER_PIPELINE_BALANCE`, auto),
`PipelineCuts` (`IDRAK_CLUSTER_PIPELINE_CUTS`); `IDRAK_CLUSTER_PLACEMENT` is now checked (single or pipeline); all in
`ClusterOptions.Variables` and the CLI's table. Tests: 5 in `tests/Idrak.Tests/ClusterPipelineTests.cs`
(`ClusterPipelineGroup`, after `ClusterTransportGroup`). One build of the test project, then on the CPU the filter
`cluster:|public API|abstraction inventory|cli health: …|cli polish: …|cli arabic: every message|cli dev: trace|profiler:|fine-tuning:|fine-tuning
options:|pretrain:|cli pretrain:|training:|callbacks:|generation:|decoder:`: 168 passed, 0 failed. Measured on the CPU (4
cores, two ranks carved from it): the tiny model's two stages hold 102,656 and 102,784 B of its 139,904 B (73% each: the
tied head's stage holds the embedding table too), `InUse` per rank 102,660 and 102,788 B; two stages against one device:
logits equal, gradients within 1.0e-7; the pretrainer at 1, 2 and 4 micro-batches, with clipping at 0.5, and as 2 stages × 2
replicas: losses and the gathered checkpoint's weights within 1e-5 of one process. **The bubble** (blocked over wall, the
cold step aside), stage 0 and stage 1: M = 1: 60% and 18% (GPipe's arithmetic 50%); M = 2: 53% and 15% (33%); M = 4: 64% and
6% (20%). On this tiny model a micro-batch's compute is about its messages' host staging, so more micro-batches do not
shrink the measured bubble here; the first stage's wait is its backward pass waiting on the last stage.

Changes from the plan, kept: **1F1B is not built**: the measured bubble does not justify it, and non-interleaved 1F1B has
GPipe's bubble by the same arithmetic ((S-1)/(M+S-1)); it only bounds the activations held to S micro-batches, which GPipe's
nested scopes already free one micro-batch at a time as its backward runs. An interleaved schedule (virtual stages) is what
shrinks the bubble; `IPipelineSchedule` takes one as a plug-in (`InFlight` reports what it holds), left for when a GPU run's
bubble asks for it. **Stages are cut by the bytes of the tensors the model was built with**, not the checkpoint file's sizes:
the same for float32 checkpoints, and what the device will hold for packed or bfloat16 builds. **Each rank loads the whole
model on the host, then keeps its stage** (`Place`): a build that reads only a stage's tensors needs every family's builder
to build a range of modules, which a family's own `Build` does not offer; the device never holds the whole model, the host
does once per process while placing (rank 0 keeps the other stages' modules on the host for whole checkpoints). **Pipeline
with data parallel is built** (stages × replicas, `IDRAK_CLUSTER_STAGES`), not refused. **The pretrainer first**: the
fine-tuner and the `Trainer` refuse pipeline placement naming the pretrainer (their adapters, recorded steps and data paths
are single placement's); the pretrainer refuses conversation records (images and parts), a teacher, the host optimizer and
training the vision part under pipeline placement, by name; a checkpoint resumes only with the placement and cut it was
written with (its optimizer state is per stage), refused naming why otherwise. A late join (a step group that changes) is
refused: a pipeline needs every stage in every step; a replay's end sits every rank out alike. Packed batches are the
pretrainer's default batches; with `MicroBatches` set the step's sequences are shared into that many padded micro-batches.
Activations cross through host memory (as the collectives' buffers do since C5); device-to-device copies through
`IDeviceTransport` move to C8 with the profile of a token's hops. Deferred: `idrak profile MODEL --cluster FILE` (C8, as
planned), `--placement pipeline` on `cluster sim` and `pretrain --cluster` (C8's command line), the GPU runs (the owner's).

**After merging C7 (2026-10-18), open for C8: a stage's load.** Under pipeline placement each rank loads the whole
model on the host and keeps only its stage, and rank 0 keeps every stage on the host to write whole checkpoints.
Pipeline exists for models that do not fit one device, so this is a memory-heavy path the library fixes (CLAUDE.md):
a rank reads only its stage's tensors from the checkpoint (the safetensors header names each tensor's offset; the
model contracts map tensor names to blocks, generically, no family named), and each rank writes its own stage's
shard of a checkpoint that a reader joins, so no rank ever holds the whole model. Activations between stages also
move device to device where the transport can (C6's device-side chunks) instead of through host memory.

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

**C8 in two parts (2026-10-18).** What C2-C7 handed to C8 is split so two agents can run beside C9 in turn:
**C8a, the command line and calibration** (the list above; plus `cluster sim` for `tune` and `train`, `--placement
pipeline`/`--stages` on `cluster sim` and `pretrain --cluster`, `--profile` of a step's buckets, `runs show` charting the
network share, the event log flushed while a run goes, an address on a link override for a cable on its own subnet, the
measured `ChunkBytes` and bucket size kept in the tuning cache and shown by `idrak tuning show`, and `calibrate` fitting
one network adapter shared by a machine's ranks where a bench shows it); **C8b, memory and speed** (a pipeline stage and
a parameter shard each load only their own tensors from the checkpoint and write only their own shard of it, joined by
the reader, so no rank holds the whole model; activations between stages moved device to device where the transport
can; a receive inbox that wakes its waiter directly; then sharding with a pipeline, and top-k with sharding, if the
measured numbers justify them).

**C8a as built (2026-10-18).** Commands (group "Cluster", each with its Arabic summary): `idrak cluster replay --topology
FILE --seed S --step N [--failures M] [--from LOG] [-- COMMAND …]` (`ClusterReplayCommand`, over C3's `ClusterReplay`): the
fault schedule through step N; with a training command after `--` the run again through step N (`ReplayStep`, every rank
sitting out the later steps) and each rank's clock as it began step N; `--from` a run's event log compares the faults and
every rank's clock through N bit for bit and names the first that differs (exit 1), and a seed, model or world size other
than the log's is a usage error naming both. `idrak cluster calibrate --from RUN|BENCH.json [--topology] [--dry-run]
[--shared-tolerance F]` (`ClusterCalibrateCommand` over `Idrak.Cluster/Calibration.cs`, `LinkCalibration`): transfers from a
run's `log.jsonl` (its `transfer` records over real transports) or a bench's JSON (per pair an empty message at the warm
latency and the message's round trip less one latency), each on the link the topology gives its pair, fitted per link by
least squares over `seconds = latency + bytes / bandwidth` (a negative intercept refits through the origin) with the
residual, kept by `ClusterCache` (below) and published as `ProfileCalibrated`. `idrak transports [NAME]`
(`TransportsCommand`): every registered transport with who registered it, its estimate and capabilities, and its
availability with the reason in three settings on this machine (two ranks of one process on the first two devices listed,
two processes here, two machines with the key from the environment); `rdma` and `udp` are named as plug-in only. `cluster
bench --concurrent` (`LinkBench.RunConcurrent`): between every two nodes of two or more ranks, as many pairs at once as the
smaller holds, their bytes together over the slowest pair's time, cold and warm; its rows ride in the JSON's
`"concurrent"`. `cluster sim` takes `tune`, `train` and `resume` beside `pretrain` (the folder rank 0 writes is named per
command: `-o RUN`, `-o DIR`, `--run DIR`, `RUN`), `--placement single|pipeline` and `--stages N` (pipeline with pretrain or
resume only, stages that do not divide the ranks named), `--event-log FILE`, and `--profile` (in-process only).

`--profile` runs the training under `Profiler.Run` with the non-intrusive probes and `kernels` (skipped where the device
does not time kernels), one warm run by default (`IDRAK_PROFILE_WARM`); the cold run trains into the run folder, each later
one into `RUN.profile-N`, removed after; the report's table (JSON with `-j`) has the network probe's new "gradient
buckets" (per rank: buckets a step, the share sent during the backward pass, the bucket bytes, network and exposed time a
step) and "pipeline stages" tables, and the cluster probe's new rows `steps`, `measured step` (the run's wall over rank 0's
steps) and `predicted step` (the same with rank 0's network wall time replaced by the slowest rank's virtual-clock time);
the command prints both. `idrak profile MODEL --cluster FILE [--stages N] [--steps N]`: every rank loads the model on the
host, `PipelineStage.Place` keeps its stage, rank 0 times decoding steps and the other stages `Serve`; the rows are a
token's hops (`rank a -> rank b`, the stages, messages, bytes, wall and simulated time per token) from the
`TransferCompleted` of the timed steps, with the token's wall and virtual-clock time. `idrak tuning show` gained the
backend `cluster`: the calibrated links (bandwidth, latency, residual, transfers, date, a shared adapter) and the measured
chunk and bucket sizes with the figures that chose them (`--reset` deletes them). `idrak runs show` charts "network %" a step
(rank 0's `GradientsSynced` exposed time over the step's batch time, or a pipeline's first stage's blocked over wall) with
the mean and the worst step, and `-j` carries `"network"`; pretrain's run log now holds every rank's `gradient_sync` and
`pipeline_step` and, unless `IDRAK_CLUSTER_LOG_TRANSFERS=0`, the transfers over real transports a calibration reads.

`--cluster FILE` on `train`, `pretrain`, `tune`, `resume` and `distill` (`ClusterJoin`, called first in each command; with
none of its options the C5 environment join stays): the world size is the file's device count; this host's node is found by
its host name or `--node`; a node of one worker gives the rank; a node of several needs it: `--rank N` (or `IDRAK_RANK`) on
train, pretrain and resume, `IDRAK_RANK` on tune and distill, whose `--rank` stays the adapter's (the usage error says so);
`--world-size` or `IDRAK_WORLD_SIZE` that contradicts the file is refused naming both; without a file `--coordinator H:P`,
`--world-size N` and the rank join a cluster of one node per rank on this process's device (`LocalhostCluster.Join(ClusterTopology,
ClusterOptions, string?)`, new), and `--world-size` without a coordinator names `--coordinator`. pretrain also takes
`--placement` and `--stages` (and reads them under `cluster sim` too). The ranks of `train` and `tune` train on the rank's
device with its collectives (`Trainer.Cluster`, `FineTuningOptions.Cluster`); rank 0 alone writes their run folder, package,
log and adapter manifest; `tune` no longer sets the process-wide default device while it is a rank (the ranks of a
simulation share the process).

Library. `ClusterCache` (`src/Idrak.Abstraction/Cluster/ClusterCache.cs`, with the records `CalibratedLink` and
`ClusterTuning`): one file per calibrated link under `{cache}/cluster/link-NAME.txt` (header `idrak-cluster-profiles 1`, then
`link bandwidth latency jitter residual samples adapter date source`) or every link in `IDRAK_CLUSTER_PROFILES_FILE`, and
the measured sizes in `{cache}/cluster/tuning.txt` (header `idrak-cluster-tuning 1`, `kind key bytes date detail`) or
`IDRAK_CLUSTER_TUNING_FILE` (`ClusterOptions.TuningFile`); an empty name keeps none, `IDRAK_TUNING_CACHE=0` neither; pure
`Serialize`/`Parse`, a header mismatch read as empty and written anew, writes beside the file and moved over it, a file read
once per version. `LinkProfiles.Resolve` returns the calibration in the library profile's place (an app's registration stays
above both; `Find` still gives the registered profile), so every topology naming the link uses it and `cluster inspect`
says "calibrated DATE (N transfers, residual)". The measured `ChunkBytes` (`HostCollectives.Chunks.cs`): with `ChunkBytes`
unset over a real transport, the first collective agrees on the size every rank kept for the transport (one all-reduce of
two floats; powers of two only), else measures: an all-reduce of `ChunkProbeBytes` (`IDRAK_CLUSTER_CHUNK_PROBE`, 4 MiB) per
candidate (powers of 4 from 64 KiB, then the whole probe in one message), the median of three warm calls, the slowest
rank's (a max over the ranks, so every rank chooses alike), ties to the larger; kept with every candidate's time. The
measured bucket size is kept the same way after the first step's measurement over a real transport (key: synchronisation,
algorithm, ranks, device kind, gradient values, compression) and a later run of that shape uses it from its first step when
every rank kept the same (`SyncCore.Started`, a new hook after the starting state). The event log is flushed while a run
goes (`SimulationOptions.EventLogFlushSteps`, `IDRAK_CLUSTER_EVENT_LOG_FLUSH`, 1): each rank appends its records since its
last flush (and the faults drawn) to the file's journal as it begins every N steps, on its own thread, and gives the blocks
back to the pool; `EventLog.Load`/`Parse` read a journal (a chunk being written left out) as well as a finished log, and
the run's end writes the finished log over it, the same bytes as a run kept in memory. `cluster sim` puts the event log in
the run folder (`cluster-events.idrklog`) unless `IDRAK_CLUSTER_EVENT_LOG` or `--event-log` names one, and names it with the
replay command. An override may carry `"address": [A's, B's]` (each end's address on its own link; null for a node's own):
`ClusterLinkOverride` gained `AddressA`/`AddressB`, `IClusterTopology.AddressOf(rank, from)` (a default member; the subset
and group topologies forward it) gives where `from` reaches `rank`, `direct-link`'s availability and the rendezvous dial it
(a bare host, or host:port whose port is ignored: the port comes from the rendezvous); `cluster init` asks
`override.N.address` for each override (a suggestion from this machine's adapters off the LAN; `--override-address
A,B;…`), and `cluster inspect` prints each override's ends. A shared adapter: `LinkProfile.AdapterBytesPerSecond`; a
topology gives each pair of nodes on such a link the adapter's rate over the ranks of the larger node, never more than the
link's own.

Changes from the plan, kept: **a shared network adapter is a static share**, not an event-ordered lane: a link whose
calibration found one adapter (a bench's concurrent pairs moving less than their count times one pair, beyond
`--shared-tolerance`, 0.25) gives each pair of ranks between those machines the adapter's rate over the larger node's ranks,
in the topology itself, so the virtual clock and every prediction stay bit for bit whatever order the rank threads charge;
it is exact when every rank of a machine crosses at once (the hierarchical collective's columns, a pipeline's replicas) and
pessimistic for a flat ring, which crosses with one pair a machine; an event-ordered clock sharing the adapter only among
transfers that overlap in virtual time is left open (a rewrite of `VirtualClock` and `SimTransport`). **On a virtual clock
an unset `ChunkBytes` stays one message a piece** (chunking buys a real link its pipelining; on the simulator it would only
make `PredictAllReduce` and a replay depend on a wall-time measurement), and the chunk is not measured under
`Profiler.Run`: the cache's detail line holds every candidate's time instead. **A measured bucket is reused over real
transports only** (on the clock it is measured each run, cheap and the same). **`calibrate --from RUN` needs the run's
`--topology`** (a bench's JSON names its own): a run folder does not record its topology. **A run's log reads transfers
with their arrival time** (the real transports time a frame's bytes, not a round trip), so a fit from a run's log gives the
bandwidth and a latency near zero; a bench gives both. **The fitted jitter is not changed** (the profile's own is kept):
the residual is recorded, and a jitter drawn into every simulated transfer would move every prediction. **`--profile` runs
the training twice by default** (cold and one warm run) and **profile `--cluster` loads the whole model on every rank's host
side** before placing its stage (C8b's stage loads fix that). The tests run against a profiles file nobody writes
(`tests/Idrak.Tests/Program.cs` sets `IDRAK_CLUSTER_PROFILES_FILE` when unset), so a machine's calibrations never move
their figures. New settings, in `ClusterOptions.Variables` and the CLI's table: `ChunkProbeBytes`
(`IDRAK_CLUSTER_CHUNK_PROBE`), `TuningFile` (`IDRAK_CLUSTER_TUNING_FILE`), `SimulationOptions.EventLogFlushSteps`
(`IDRAK_CLUSTER_EVENT_LOG_FLUSH`); the cluster variables now list the commands that read them (replay, calibrate, inspect,
transports, train, tune, resume, distill, profile, tuning show). Tests: 12 in `tests/Idrak.Tests/ClusterCliTests.cs`
(`ClusterCliGroup`, after `ClusterPipelineGroup`); C2's "another command is a usage error" now uses `chat`; the options test has values for the new variables. Fixed on the way:
`cluster sim` declared the common `--seed` again (the CLI health check refused it; `cluster replay` reads it as a common
option too), and `idrak env set` asked "port" took IDRAK_CLUSTER_PORT and IDRAK_CLUSTER_TRANSPORT as matches beside
IDRAK_PORT (it now takes IDRAK_ + the answer first). One build of the test project, the API dumps regenerated
(Abstraction, Cluster, Network; the inventory unchanged), then on the CPU the filter `cluster:|public API|abstraction
inventory|cli health|cli polish|cli arabic: every message|cli dev: trace|cli pretrain:|cli measure|cli train data|profiler:|
pretrain:|fine-tuning:|training:|callbacks:|tuning show`: 189 passed, 0 failed, no process left.
Not run here: the GPU runs (the owner's), and the LAN or rental benches and calibrations. Deferred to C8b as planned:
stage and shard loads of their own tensors, device-to-device activations, the waking inbox, sharding with a pipeline and
top-k with sharding.

**Merged with C9 (2026-10-18, onto f262009).** Both kept. `cluster sim` takes every C8a option (tune, train and resume,
`--placement`/`--stages`, `--event-log`, `--profile`) and C9's `--dashboard`/`--port`/`--no-browser` with its run control;
C9's `RunRanks` is the one place ranks run (`Simulate` wraps it, with a run control and the dashboard's output when given),
`CheckTrainer(inner, steered)` checks the command and the folder rank 0 writes for the command line, the dashboard's
`POST /api/runs` and `Prepare` alike, and `RunFolder` names each command's folder (pretrain `-o`, train `--run`, resume
`RUN`). **A run the dashboard steers is pretrain's or the resume of one**: a stop ends every rank as it begins the next
step and only the pretrainer answers it with that step's checkpoint, so `--dashboard` (and a run started from the page)
with tune or train is a usage error saying so; `--profile` with `--dashboard` is refused too (a profile trains more than
once). **One event log name and format**: `RUN/cluster-events.idrklog` (`ClusterSimCommand.EventLogFile`), written by
`cluster sim` and by the dashboard's runs (C9 had `cluster-events.bin`), flushed as the run goes (C8a's journal, the live
faults' mark kept in the fault records), read by `EventLog.Load` while the run goes or after. **`cluster replay --from`
replays a steered run** (C9's deferral "`cluster replay` taking the event log", closed): the log's live faults join the
seed's schedule in the comparison and a rerun injects them again at their steps (`ClusterRunControl` over them, as
`ClusterReplay.Run(…, steered)` does); the dashboard's fault answer names it, and C9's live-fault test now also replays the
steered run through the CLI and gets "matches". The `--seed` and `env set` fixes both phases made are kept once (C9's
`env set` match). `Tests.All`: … `ClusterPipelineGroup`, `ClusterCliGroup`, `ClusterDashboardGroup`, …; the options test
lists both phases' variables; the CLI's `ClusterRun` lists both phases' commands, "cluster ui" included, beside C9's
`ClusterUi`. API dumps and the inventory regenerated after the build. Found on the way: `RunFolder` now skips a leading
"idrak" and reads another command's `-o` (cluster start's `RUN/ranks.jsonl` had stopped being written), and C9's live-fault
test steers a 400-step run (40 steps could finish between the test seeing two steps past the fault and asking the stop:
"the run is finished: there is nothing to stop"); the replays use the same count. One build after the merge, the dumps and
inventory regenerated, then on the CPU the filter `cluster:|public API|abstraction inventory|cli health|cli polish|cli
arabic: every message|cli dev: trace|cli pretrain:|cli measure|cli train data|cli serve|profiler:|pretrain:|fine-tuning:|
training:|callbacks:|tuning show`: 204 passed, 0 failed, no process left. Open: the dashboard steering train and tune (a
stop answered with a checkpoint by the Trainer and the fine-tuner), with C8b's stage-own checkpoint.

**Owner's GPU run after C6 + C7 (2026-10-18), open for C8b.** "cluster: a model whose training state is over the cap
of one fake GPU trains across four with Shard = parameters" fails on CUDA and on Vulkan, passes on the CPU: every rank
holds 1,835,012 B at each epoch's end against its share of 1,572,864 B, constant over the epochs (no growth), so
262,148 B above the share = one 256 x 256 weight of the test's model (262,144 B) and one float. On the GPU backends a
gathered unit's weight, or a backend scratch sized to the largest operand, stays in use at the epoch's end where the
CPU gives it back. C8b first makes the test name what is held (the device's live storages by size and owner on
failure), so the owner's next GPU run says which, then fixes it in the library; the test's bound stays as it is.

**C8b as built (2026-10-18).** Memory and speed. **The 262,148 B (closed, from the code; the owner's GPU run to confirm):**
`GradientReduction.Write` copied a piece of a parameter cut across buckets through a device scratch (`_scratch`,
`Math.Max(n, BucketValues)` floats, made once and kept for the run) on every non-CPU device; the CPU wrote host memory
in place. With `BucketBytes` 256 KiB the 256 x 256 weights straddle buckets, so every GPU rank kept 65,536 floats
(262,144 B) beside its share; the float is the `Trainer`'s epoch sums (`Tensor.Zeros([1 + metrics])`), which the CPU
holds too (the CPU now prints 1,572,868 B = share + 4 per rank, the listing below). Fixed by a device contract:
`Backend.UploadRange(source, destination, offset)` (default: a pooled block of the piece's size taken and given back at
once, then `Copy2D`; CPU, CUDA (`cuMemcpyHtoD(Async)` at the offset, through the pinned staging ring) and Vulkan (mapped
memory at the offset, else the staging buffer copied to the offset) override it), and the scratch is gone. The test
names what is held on failure: `StorageTracking` (`Track()`, or `IDRAK_TRACK_STORAGES=1` for a whole process; one
static read per storage made when off) records each storage with the method that made it, `Backend.LiveStorages(name)`
(a virtual on the device contract: the tracked storages alive and in memory, largest first; a device may add its own
blocks) and `StorageTracking.Describe`; "cluster: a model whose training state is over the cap…" prints each rank's
listing at the last epoch's end ("7 storage(s), 1,572,868 B; 4 x 262,144 B Optimizer.CreateState; 2 x 262,144 B
ShardedWeights..ctor; 4 B Trainer.Fit") and every epoch's in its failure message, parameters and gradients by index and
shape. The bound is unchanged. **Stage 0 holding 102,404 B against its 102,656 B on CUDA** (the coordinator's second GPU
item): nothing between 94819b0 (passed) and 8e4b84c changes where a stage's tensors live (that range is C9: the
dashboard, the control gate, the no-rows overlap); 102,404 = 102,656 - 256 + 4 (the boundary's token), and 256 B is one
64-float tensor, e.g. `ShardModel`'s first weight [2, 32], which the C9-era tests before the pipeline group (the no-rows
reduction on carved slots 0-2) can leave to the finalizer. So the likely cause is a release by a finalizer inside the
measured window on carved slot 0, not a stage tensor missing. The test now collects (full collection and finalizers)
before it measures, tracks storages from there, checks each of the stage's tensors is on its device and in memory, and on
failure names each one that is not (its module path, shape, bytes, and where: released, evicted, on another device) and
what was made or released on the device meanwhile; IDRAK_TRACK_STORAGES=1 names what earlier tests left too. Not certain
without the owner's run (no GPU here).

**A stage and a shard load only their own tensors (closed).** `DeferredTensors` (Idrak.Abstraction): a device "deferred"
(`DeferredProvider`, registered by `DeviceProviders` itself, by name only: never listed, never the default) whose storages
hold no memory; each tensor's values come from a source (`SetSource`) read when it moves to a real device
(`Module.MoveTensor`: one tensor at a time); a build on it hands out placeholders (`Build`, `Placeholder`: one array per
length per thread, never written) so **any family's builder runs unchanged** and an upload of a placeholder keeps nothing,
while values a build computes (rotary tables) are kept; inside `StayDeferred` a move gives a tensor on the device whose
memory is given back at once (`Backend.Evict`) and read on its first `Restore`. `PretrainedOptions.Deferred`
(`PretrainedModel.Load`): every weight is checked against the checkpoint's header and answered with a placeholder
(`CheckpointWeights.Deferred`), then given its place in the file through the save mapping (`PretrainedModel.StoredTensors()`:
each weight's checkpoint name, stored shape and transposition, the names the architecture contract gives, so no family
is named); a parameter the mapping does not cover fails naming its module ("load the model whole"); packed weights and a
merged adapter are refused (a deferred load reads float32 weights). `PretrainedModel.BytesRead` (`ITensorStore.BytesRead`,
counted by `SafeTensorsReader` per tensor byte range), `IsDeferred`, `Place()`. Pipeline: `PipelineStage.Place` works as
before on a deferred network: the stage's modules are read as they move, the others released (or kept by rank 0 with no
memory). Shard = parameters: the pretrainer moves the deferred network inside `StayDeferred` (nothing read), every rank
reads what no shard trains, and `ShardedWeights` reads a rank's own parameters as its share is made (the others'
recomputation replaced by the broadcast from their owner, `Backend.SetRecompute`); `SyncCore.Start`'s broadcast and the
starting-state check skip evicted tensors (every rank's are read from the same file). `ClusterOptions.OwnTensorsOnly`
(`IDRAK_CLUSTER_OWN_TENSORS`, on); `pretrain` loads deferred under a pipeline or a parameters shard. Measured (tiny model,
2 ranks carved from the CPU): a pipeline rank read 0 B at the load and 102,656 / 102,784 B when placed (73% each of the
file's 139,904 B: the tied table is read by both stages), in use 102,660 / 102,788 B, the deferred device keeping 2,048 B
(the unplaced modules' rotary tables), weights bit for bit one process's; a parameters shard read 70,016 / 69,888 B
(50% each, together the file once) against 139,904 B each loaded whole, the losses and checkpoint byte for byte the
whole-load run's.

**Checkpoints: streamed through rank 0 (chosen over a sharded format with an index).** Each rank answers rank 0's
requests a tensor at a time (`StreamedCheckpoint`: the owner's optimizer entries' names and shapes for the header, then
each weight and entry as rank 0 writes it), and rank 0 writes the one-process files (`model.safetensors`,
`optimizer.safetensors`): the joined checkpoints byte for byte, so every reader and resume is unchanged, and no rank holds
another's share or more than one tensor of it. Why not shards with an index: the ranks of a cluster do not share a file
system (separate machines; the tests give each rank its own folder), so shards would be scattered over machines, and the
index's readers (`SafeTensorsReader` reads one already) would need every part gathered anyway; the bytes cross the network
once either way. `PretrainedModel.SaveHuggingFace(…, values)` takes each weight's values from a delegate (rank 0's own, or
the owner's over the transport); the pipeline's writer (`SavePipelinedStreamed`, the cut and state in `WritePipelineState`)
and the sharded one (`SaveShardedStreamed`; `IGradientSync.Owner`, `Local`, `SaveOwnOptimizerState`, new default members)
replace the gathers: a sharded save no longer calls `Gather()` or joins the state on every rank. A pipeline stage resumes
reading only its own optimizer entries. `ClusterOptions.StreamedSave` (`IDRAK_CLUSTER_STREAMED_SAVE`, on; off: C6's and
C7's joined saves). Left: **evaluation under Shard = parameters still gathers the whole weights** (`Gather()`): a decoder
block's inference forward hands its sum to the next block's norm (`NextNorm`, a gain outside its own module) and the ranks'
evaluation shares run unequal forwards, so per-layer gathers cannot serve it yet; `End()` gives every rank the whole model
after the last step (the shard test's peak, 612,736 B, is that and the activations, equal to the whole-load run's), and an
optimizer's `SaveState` copies a stage's or a share's whole state to the host before it is streamed.

**Activations device to device (closed).** `IDeviceTransport` moved to `Idrak.Abstraction.Cluster` (Idrak's pipeline uses
it now; Idrak.Network implements it as before). `PipelineBoundary` (new `deviceActivations` parameter): over a transport
that hands device memory over, the shape still travels as bytes and the values are posted (`PostDevice`, the storage
held by a reference until taken) and copied by the receiver (`ReceiveDevice`); `Flush` waits for them at a step's or an
evaluation's end. `PipelineStepped.Staging` ("device" or "host") in the console and JSON lines. `ClusterOptions.DeviceActivations`
(`IDRAK_CLUSTER_DEVICE_ACTIVATIONS`, on). Measured: 2 stages over pcie-host, logits and every gradient bit-identical to host
staging, 32,832 B a step either way.

**The receive inbox wakes its waiter directly (closed; no measurable gain here).** `FramedTransport`'s queues are an
`InboxQueue` of its own: a blocked receive waits on the queue's monitor and the thread that queues a message pulses it;
`ClusterOptions.DirectWake` (`IDRAK_CLUSTER_DIRECT_WAKE`, on; off: the channel's wake through the thread pool, kept to
measure against). Measured on this machine (4 cores, shared), a message after an idle gap: between two processes over
shared memory 265 to 279 us median through the pool against 260 to 307 us directly; between two threads over pcie-host
60.2 to 69.5 us against 53.0 to 67.8 us. The wait is the OS waking a descheduled thread (the doorbell's, then the
receiver's); the pool hop is a small part of it. Not changed: `LocalTransport`'s device handoffs still wake through a
channel.

**A stop under pipeline placement writes its checkpoint (closed).** The pipelined pretrainer answers
`ClusterRunStoppedException` at a step boundary with the last finished step's checkpoint (every stage's part streamed to
rank 0) and rethrows. Measured: stopped after 4 steps, the checkpoint byte for byte the uninterrupted 4-step run's; resumed
(each stage reading its own part) to 7 steps, losses and weights the 7-step run's. **The Trainer and the fine-tuner answer
a stop too** (the coordinator's addition): `Trainer.StopFolder` (`checkpoint-STEP`: `model.ikw`, `optimizer.safetensors`,
`train-state.json`; a sharded run's weights gathered and state joined) and `Trainer.ResumeFrom` (weights and optimizer
state loaded, the finished steps drawn alike and not trained; a cluster run needs a `DataLoader` over a sample source);
the fine-tuner writes `checkpoint-STEP` in its output folder (the adapters, `optimizer.safetensors`, `tune-state.json`) and
`FineTuningOptions.ResumeFrom` goes on from it (adapters loaded into the ones just made, a trained vision part or part
source refused). Tested: stopped at step 3 and resumed, the Trainer (unsharded and Shard = parameters) and the fine-tuner
(unsharded and Shard = optimizer) end with the uninterrupted run's weights bit for bit.

**Sharding with a pipeline, top-k with sharding: kept refused.** The numbers do not ask for them yet: a stage of the tiny
model holds 73% of it with 2 stages (the tied table on both), a stage's replicas only exist from 4 ranks, and the stage's
gathers would run between the stage-to-stage sends; a sharded reduce-scatter sends no more bytes than the all-reduce, and
top-k's gain (C4: under half the bytes at 10%) is for a network-bound step, which no run here is (the owner's LAN and rental
runs and C8a's advisor are what would show the need). Both refusals stay, naming the settings.

Tests: `tests/Idrak.Tests/ClusterMemoryTests.cs` (`ClusterMemoryGroup`, 7 tests after `ClusterDashboardGroup`); updated:
the over-the-cap sharding test (the listing), "a rank holds only its stage's tensors" (collect first, per-tensor
residency), the options test (the four variables). One build of the test project, the API dumps (Idrak.Abstraction,
Idrak, Idrak.Nlp, Idrak.Network) and the inventory regenerated, then on the CPU the filter `cluster:|public API|abstraction
inventory|cli health|cli polish|cli arabic: every message|profiler:|pretrain:|cli pretrain:|training:|fine-tuning:|callbacks:|offloading:|generation:|decoder:|pretrained:|minimal:|minimum backend:`:
216 passed, 1 failed (the options test, which lacked the new variables), fixed and rerun. Not run here: everything on a
GPU (the UploadRange overrides on CUDA and Vulkan, the two GPU items above, pcie-host and nvlink between GPUs).

**Merged with C8a (2026-10-18, onto 7c41b8d).** Conflict only in `tests/Idrak.Tests/Program.cs` (both group lists kept:
`ClusterCliGroup`, then `ClusterMemoryGroup` after `ClusterDashboardGroup`). The command line now steers `tune` and `train`
too: C8a's usage error in `ClusterSimCommand.CheckTrainer` for a steered tune or train is lifted (every trainer it takes
writes a checkpoint on a stop); `tune` answers a stop with "stopped as asked before step N" and the checkpoint, and takes
`--resume-from DIR` (`FineTuningOptions.ResumeFrom`); `train` sets `Trainer.StopFolder` to its run folder under a cluster and
answers the stop with exit 0, and `idrak resume RUN` of a train run goes on from the newest stop checkpoint when it is newer
than `last.ikw` (`TrainSession.Run(…, stopped)`, `Trainer.ResumeFrom`); an image-folder run keeps its epoch resume. The test
"cluster: a stopped Trainer and a stopped fine-tuner…" checks `CheckTrainer` takes both. One build of the merged tree, the
dumps and the inventory regenerated (no change), then the same filter on the CPU: 229 passed, 0 failed, no process left.

**C8c as built (2026-10-18): the "cluster" setting advisor (decision 4).** Through the existing contract and registry:
`ClusterSettingAdvisor` (`src/Idrak.Network/Training/ClusterSettingAdvisor.cs`, name "cluster") is registered in
`SettingAdvisors` by Idrak.Network's `LibraryRegistrations`; its subject is a new record in Idrak.Abstraction,
`ClusterStepMeasure` (`src/Idrak.Abstraction/Cluster/ClusterAdvice.cs`: the run's `ClusterOptions`, synchronisation,
collective, compression, ranks, nodes, warm steps measured, a step's time, the exposed and the whole network time, the
backward pass, gradient bytes, bucket bytes and count, whether the times are the virtual clock's, bf16's measured error and
the slice's values; `Share` and `Compute` derived), so an app's own "cluster" advisor gets the same figures; the contract
itself is unchanged. The measurement (`GradientReduction.Advice.cs`, all-reduce and sharded; local SGD measures nothing)
runs while `ClusterOptions.Suggest` (`IDRAK_CLUSTER_SUGGEST`; unset: while someone listens at `TelemetryLevel.Suggestions`)
and only on the first warm steps (`SuggestSteps`, `IDRAK_CLUSTER_SUGGEST_STEPS`, 3; the first, cold step left out, rule 76):
each step's time from one step's end to the next (`SyncCore.StepTimed`, a new hook at the start of `Stepped`) with the wall
time of its reduction replaced by the network time the step waited for (on a simulated transport the virtual clock's
predicted exposure, so a simulated step is the cluster's; on a real one the two are the same wall time), the network time
and the backward pass; on the last of those steps the first bucket's own values (at most `SuggestSliceValues`,
`IDRAK_CLUSTER_SUGGEST_SLICE`, 65,536), weighted as the step weights them, are copied as the bucket is reduced (during the
backward pass when it goes then); as that step ends the slice is reduced twice, over a float32 wire (with every rank's four
sums beside it, so the figures are means over the ranks) and over a bfloat16 wire, the error is the largest difference over
the largest float32 value, rank 0 asks the advisors (`SettingAdvisors.Publish`) and the measurement ends. Off, a step costs
one field read; nothing a run computes changes (the slice is a copy, reduced after the step's own reduction). The ranks
decide alike: ranks of one process take one decision for the run (the first rank's, kept per run options instance, the one
every rank's context holds), so a listener subscribed while ranks start cannot split them; ranks in processes of their own
agree in the bucket size's agreement collective (a fifth float) or, where there is none, in one float of their own over the
real transport. A group that changes (a fault) ends the measurement on every member; each suggestion is published once a
run (the published set is the run's, shared by a pretraining run's stages).

The suggestions (each a `SettingSuggestion` with the setting, the value, a reason with the numbers and how to set it, the
device and "cluster"; none applied). Nothing at or under `IDRAK_CLUSTER_SUGGEST_SHARE` (the share: exposed network time over
the step). Above it: **`ClusterOptions.Collective` = hierarchical** where ring runs over more than one node and a node holds
several ranks; **`ClusterOptions.Wire` = bf16** where the buckets travel uncompressed and the slice's error is within
`IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE`; then, where the share with bf16's half the bytes (where suggested) is still above
the threshold T, **`GradientAccumulation`** and **`ClusterOptions.LocalSgdEvery`** (all-reduce runs not on local SGD) by
A = ceil(E'(1-T)/(T·C)) (E' the exposed time, halved with bf16; C the step's compute), and **`ClusterOptions.TopK`** with the
share T·C/((1-T)·E·W) (top-k all-gathers (index, value) pairs: about share x ranks the bytes of a ring; all-reduce runs
without compression); **`ClusterOptions.BucketBytes`** where a set bucket size makes one bucket of every gradient while
overlap is on: B = ceil(N / max(N - K, N/256)) buckets (N the network time, K the backward pass, 2 to 256). **The default bf16
tolerance is bfloat16's own rounding step, 2^-7 (0.0078125), the owner's choice** (decision 4): **Idrak's bf16 wire loses more
than the fixture's DDP hook**, which sums in float32, because Idrak's ring rounds both each contribution it sends and the
reduced sum it gathers. On the Spirals model the slice lost 1.47e-3 (batches of 32), 2.17e-3 (the CLI's regression model) and
4.58e-3 (batches of 128) of its largest value, all accepted; the fixture measured the advisor's way (`bf16/` against `ddp/`,
each tensor a slice) lost 2.71e-3 at worst (`model.embed_tokens.weight`; 1.33e-3 of the whole model's largest gradient, the
README's figure, a measure a slice cannot take), which as a default would have refused the last.

Command line. `SuggestionLine` (`src/Idrak.Cli/Commands/Cluster/ClusterSuggestions.cs`) shows each suggestion once a
command (per output, so a profile's cold and warm runs show it once) as `suggested: SETTING = VALUE (cluster on DEVICE):
REASON`, taking the progress lines drawn on the error output off the screen first (`ProgressLine.EraseOn`, new); it is
subscribed by `cluster sim`'s `RunRanks` (so `cluster sim`, `cluster replay` and the dashboard's runs always measure) and by
`ClusterJoin` for every rank of `--cluster` and `cluster start` processes. The run's log keeps them (`setting_suggestion`
records): pretrain's `TelemetryAppender` now listens at `Suggestions`, `train` under a cluster appends them (`SuggestionLog`);
tune writes no run log, so its suggestions are shown only. `idrak suggest RUN` (a run folder with run.json and log.jsonl, its
log.jsonl, or a name from `idrak runs list`; DATA otherwise as before) lists them, `-j` as JSON. `cluster sim` gained
`--local-sgd K`, `--bucket-bytes SIZE` and `--topk F` (each suggestion's command-line form; passed to localhost processes
as their variables). The dashboard's telemetry hook passes `setting_suggestion` records and the page draws them on the event
timeline as "suggestion" events (blue, with the advisor's name), a followed run's from its log. New settings, in
`ClusterOptions.Variables` and the CLI's table: `Suggest` (`IDRAK_CLUSTER_SUGGEST`), `SuggestSteps`
(`IDRAK_CLUSTER_SUGGEST_STEPS`), `SuggestSliceValues` (`IDRAK_CLUSTER_SUGGEST_SLICE`); `IDRAK_CLUSTER_SUGGEST_SHARE` and
`IDRAK_CLUSTER_SUGGEST_BF16_TOLERANCE` (read since C0) are now used.

Changes from the plan, kept: **the share is the sync's own per-step timing, not `CollectiveCompleted.Blocked` and `Wall`**:
those are timed only while someone listens at `Network` (rule 78), while the sync times its reductions every step anyway;
the share is the exposed time (what the step waited for after its backward pass), the time `--wire bf16` and the rest can
save. **The slice's two collectives are on the virtual clock** (they are traffic), so the setting is part of a run's replay
key: see the merge below. **The suggestions name `ClusterOptions` and `GradientAccumulation`**, with
the command-line flag and variable in the reason.

Tests: 6 in `tests/Idrak.Tests/ClusterAdvisorTests.cs` (`ClusterAdvisorGroup`, after `ClusterDashboardGroup`); the options test
has values for the three new variables. Measured on the CPU (4 cores), the Spirals classifier (three layers of 256, 520 KiB of
gradients, batches of 32 a rank), two ranks: ethernet-1g between two nodes, the network 72% of a 5.36 ms step (3.88 ms exposed,
4.26 ms network, backward 0.82 ms): bf16 (error 1.47e-3 within the test's 2^-7 and the default 2.71e-3), accumulation 6,
local SGD 6, top-k 0.048; same-device 0.7%: nothing; a tolerance of 1e-9: no bf16, accumulation 12, local SGD 12, top-k
0.042; ring over two nodes of two ranks, 77%: hierarchical; one pinned 64 MiB bucket, 83% (backward 0.54 ms): 261 KiB
(2 buckets); rank 0's collectives 566 unset without a listener, 566 off, 568 on; the weights and losses bit for bit on and
off. `cluster sim -- train` of a regression network (2-512-512-1) on ethernet-1g: 78%, bf16 (2.17e-3), accumulation 8,
local SGD 8, top-k 0.035, each shown once and listed by `idrak suggest RUN`. One build, the API dump (Abstraction) and the
inventory regenerated, then on the CPU the filter `cluster:|public API|abstraction inventory|cli health|cli polish|cli
arabic: every message|profiler:|pretrain:|fine-tuning:|training:|cli design: suggest|image reader: suggested`: 184 passed,
0 failed, no process left. Not run here: the GPU runs (the owner's).

**C8c merged with the main line (2b51430: C8b, the status table, the ClusterProbe fix), and the advisor's setting in the
event log.** Both kept: C8b's `Backend.UploadRange` write path (its scratch gone) and stops beside the advisor's hooks in
`GradientReduction` and `Sharding.Began`; `Tests.All` … `ClusterDashboardGroup`, `ClusterMemoryGroup`, `ClusterAdvisorGroup` …;
the options test and the CLI's table list both phases' variables. A replay no longer depends on anyone remembering the
advisor's setting: **an in-process run resolves it once as it starts** (`InProcessCluster.Run`: `Suggest` set, or whether
someone listens at `Suggestions` then; every rank takes it, with its warm steps and slice, from its context, so ranks can no
longer split on a listener subscribed while they start), **the event log's header keeps it** (format version 2: a byte for
on, the warm steps and the slice's values after the failure model's name, in the finished log and the journal alike; a
version 1 log predates the advisor and reads as off; `EventLog.Suggest`, `SuggestSteps`, `SuggestSliceValues`, and
`EventLog.AdviceOf(options)`), and **a replay from a log takes it** (`ClusterReplay.Run` with the steered run's log, `idrak
cluster replay --from`); a replay without a log keeps the setting it is given. The dashboard test's `Suggest = true` is
gone; "cluster: a replay stops at a step bit for bit…" now also runs the pretrainer with a listener and one warm step (its
slice collectives at the end of step 1), replays it through step 2 from its log with nobody listening and the setting unset,
clocks bit for bit, and checks a replay without the log differs at step 2. The event log's version message now names both
versions it reads. The owner set the bf16 tolerance's default to 2^-7 (above); the slow-link test uses the default. One build
of the merged tree, the API dumps (Abstraction, Cluster) and the inventory regenerated, then on the CPU the filter
`cluster:|public API|abstraction inventory|cli health|cli polish|cli arabic: every message|profiler:|pretrain:|fine-tuning:|
training:|cli design: suggest|image reader: suggested|offloading:`: 199 passed, 0 failed, no process left.

### C9: the dashboard (set up, control and monitor in a browser)

Beside C8, once C6 and C7 have landed (two agents). Decided by the owner (decision 8): simulated runs are set up,
started and controlled from the page; every run is monitored there, real `cluster start` runs across machines
read-only.

**Commands.** `idrak cluster ui [--topology FILE] [--port N] [--host H] [--no-browser]` opens the dashboard on its own
(set up a topology, then start runs from the page), as `idrak ui` opens the chat page; `--dashboard [--port N]` on
`idrak cluster sim` opens it on a run started from the terminal. The port defaults to one the system picks, printed with
the URL (`IDRAK_CLUSTER_UI_PORT` sets it); the host defaults to `127.0.0.1` (`IDRAK_CLUSTER_UI_HOST`); off loopback an
API key is required (the `idrak server keys` store and `Authorization: Bearer` / `X-API-Key`, as `serve`), CORS off.

**Server.** ASP.NET Core minimal APIs in `Idrak.Cli` (which already references it for `serve`), built the way
`ServeHost` builds its app; live data as server-sent events (`Results.Stream`, flushed per event) from a
`ChannelTelemetry` subscribed only while a page listens (rule 78: nothing measured for nobody). Endpoints:
`GET /api/machine` (the builder's `MachineFacts`); `POST /api/topology/questions` (the builder's next question with its
description, default and suggestion, an answer in, the refusal or the next question out: the same `TopologyBuilder`
`cluster init` asks with); `POST /api/topology` (validate, save; the commented JSON `cluster init` writes);
`GET /api/topology/inspect` (`TopologyInspection`: ranks, links, predicted all-reduce per algorithm, the transport per
pair); `POST /api/runs` (start `cluster sim`, in-process or localhost, with its training command and options);
`POST /api/runs/{id}/pause|resume|stop`; `POST /api/runs/{id}/faults` (inject now); `GET /api/runs` and
`GET /api/runs/{id}/events` (SSE). Every long action reports progress (what, how far, how fast, time left).

**Control through the library, not the page.** `Idrak.Cluster` gains a run controller: pause and resume take effect at
the next step boundary (the trainers' `BeginStep` waits while paused), stop cancels after the current step and the
trainer writes its checkpoint, and faults asked for live are queued to a `"live"` failure model registered in
`FailureModels` that hands them to the next step and composes with the run's own model. Every live fault is written to
the event log with its step, so `--seed S --step N` replays a run steered from the page bit for bit. A localhost run
takes pause, resume and stop through its processes' environment channel (the launcher's), not faults (C5: faults
belong to the simulated transport).

**Monitoring.** Per rank: state, step, loss, tokens a second, network time against blocked time (the virtual clock's
for a simulated run), memory in use against its cap; per link: bytes and time; the fault and rank-state timeline; the
collectives as they run (`CollectiveCompleted`, `GradientsSynced`), the hierarchical/ring choice and the bubble number
(C7) when a pipeline runs. A real run is followed read-only from its run folder: rank 0's `log.jsonl` (C5 writes every
rank's per-step summary there) and the launcher's rank lines; its controls are disabled with the reason shown.

**The page.** One self-contained HTML/CSS/JS page (as `ChatPage`): no CDN or external URL, charts drawn by the page
itself, works offline, light and dark; the topology drawn as nodes, ranks and links, updating as the form changes;
sizes in the `--unit` the user picks (powers of 1,024, decision 7); the CLI's language (Arabic, right to left) where
the CLI's messages are Arabic.

**Tests** (`"cluster: ui …"`, through an in-memory test server): the builder API asks `cluster init`'s questions and
writes the file `cluster init --yes` writes for the same answers; a sim run started from the API pauses at a step
boundary, resumes, and stops with a checkpoint, the losses equal to an uninterrupted run's; a fault injected live is in
the event log and a replay reproduces it bit for bit; the event stream carries every rank's step, collective and fault
events in order, and nothing is measured while no page listens; the dashboard binds to `127.0.0.1` by default and
refuses an off-loopback request without a key; the page names no external URL and every endpoint it calls exists; a
run folder written by `cluster start` is followed read-only and its controls are refused naming why.

**C9 as built (2026-10-18).** Control through the library. `src/Idrak.Abstraction/Cluster/RunControl.cs`:
`ClusterRunStoppedException` (an `OperationCanceledException` naming the step no rank began), `ClusterControlCommand` and
`ClusterControlFile` (the control channel of a run of processes: one word, run, pause or stop, replaced whole; rank 0's
report in the `.state` file beside it); `SimulationOptions.Control` (`IDRAK_CLUSTER_CONTROL`) and `.ControlPoll`
(`IDRAK_CLUSTER_CONTROL_POLL`, 0.2 s). `src/Idrak.Cluster/RunControl.cs`: `ClusterRunControl` (`Pause` returns the step
boundary every rank waits at, the highest step begun plus one; `Resume`; `Stop` lets the highest step begun finish and
stops every rank as it begins the next; `Inject` checks a fault as the injector does and queues it; `State`, `LastStep`,
`PauseAt`, `StopAfter`; `Changed` and, only while `Observed`, `RankStepped` with each rank's wall time between steps and
its device's memory), `ClusterRunState`, `ClusterRunChange`, `ClusterRankStep`; `LiveFailures`, registered in
`FailureModels` as `"live"` (queued faults stamped with the next step drawn, or faults scheduled at their own steps);
`LocalhostControl` (pause, resume and stop through the control file; `LocalhostLaunch.Control` gives every rank its path).
`InProcessCluster.Run(topology, options, work, control)` finds the control from the simulated transport's world (by the
run's rendezvous name); every rank's `SimStepper.BeginStep` passes the control's gate first, so pauses and stops happen
at one step boundary on every rank. The injector draws the run's own model from (seed, step) as before and then the live
model's faults, which take nothing from the generator, so the two compose; each live fault is logged with the high bit of
its kind byte set (`EventLog.LiveFaults`; the format's version is unchanged and older logs read as before) and published
as `FaultInjected`. `ClusterReplay.Run(…, EventLog steered)` injects the steered run's live faults again at their steps.
`TopologyInspection.Pairs` (`TopologyPair`, `TopologyPairKind`: same device, inside a node, between nodes, an override)
and `PredictPhases` (`AllReducePhase`: each phase of ring, naive and hierarchical played on the same clock as
`PredictAllReduce`, its links slowest first and what it adds to the slowest rank's clock, so the phases add up to the
prediction). Idrak.Network: `SyncCore.BeginStep` of a fixed group (processes) with a control file has rank 0 read it and
broadcast one float, so every rank runs, waits (rank 0 broadcasting again every `ControlPoll`, so no wait reaches the
timeout) or stops at the same boundary. Idrak.Nlp: the pretrainer answers a cancellation thrown as a step begins with
the last finished step's checkpoint (rank 0), as it answers a cancelled step. CLI: `idrak cluster ui [--topology FILE]
[--port N] [--host H] [--no-browser] [--follow RUN] [--unit UNIT]` (`ClusterUiCommand`, Arabic summary) and
`cluster sim --dashboard [--port N] [--no-browser]` (in-process: steered by a `ClusterRunControl`; localhost: by a
`LocalhostControl`); `pretrain` answers `ClusterRunStoppedException` with "stopped as asked", the checkpoint and exit 0;
`cluster start` writes `RUN/ranks.jsonl` (the start, every rank line, each rank's end) when its command names a run
folder. `src/Idrak.Cli/Commands/Cluster/Dashboard/`: `DashboardServer.cs` (Kestrel through `WebApplication.CreateSlimBuilder`
as `ServeHost`; the thirteen routes of the plan's endpoint list in `DashboardServer.Routes`; `DashboardSettings` with
`IDRAK_CLUSTER_UI_PORT` (0: the system picks, printed), `IDRAK_CLUSTER_UI_HOST` (127.0.0.1), `IDRAK_CLUSTER_UI_HISTORY`
(20,000 records kept per run for a page that connects late) and `IDRAK_CLUSTER_UI_POLL` (0.5 s between reads of a run
folder while a page listens)), `DashboardRuns.cs` (a run's records, the telemetry hook, the folder tail),
`DashboardTopology.cs` (machine facts, the builder's step, answers to and from a topology, the inspection as JSON) and
`dashboard.html` (the page). Every record is a JSON object with an `"event"` key, the telemetry ones as `TelemetryJson`
writes a run's log.jsonl, so the page reads a live in-process run, a localhost run and a followed folder alike. The page:
the header (the topology file, the run's state, Start, Pause at next step, Resume, Stop and checkpoint, Inject fault, each
disabled with the server's reason as its title and a strip naming it), the fault strip (kind, rank, factor, duration,
"inject at next step", the replay key), Set up (the builder's questions one at a time, or the node cards and links form,
each node's suggestion from the machine's devices; the topology drawn as the network bus, the ring, the link matrix,
the default all-reduce's phases and each card's memory split into its ranks' caps, all from the inspection; the
predicted all-reduce per algorithm of a size the page sets), Run (start a run in-process or as processes, or follow a
folder; KPI tiles with the time left; a card per rank with its state, loss, steps a second, memory against its cap, and
network against blocked time; the loss drawn by the page; the links table; rank 0's output or the ranks' lines) and
Events (newest first, each fault with its replay key); light and dark by `prefers-color-scheme`, IBM Plex when installed
and system fonts otherwise, no address outside the server; right to left with its labels in Arabic when the tool's
language is. Tests: 7 in `tests/Idrak.Tests/ClusterDashboardTests.cs` (the seventh, the sharded stop, after the merge below) (`ClusterDashboardGroup`, after
`ClusterPipelineGroup`), each through the real server on 127.0.0.1 and a port the system picks. Measured on the CPU (two
ranks carved from it, the tiny model): a run started paused from the API, resumed, paused at a step boundary (every rank
waiting before the same step), resumed and stopped after 19 steps wrote checkpoint-19 with the losses and weights of an
uninterrupted 19-step `cluster sim` run, bit for bit; a slow worker (x8) injected live on three ranks at step 7 was replayed
from the run's event log through step 8 with every rank's clock equal (rank 1 at 0.000270511 s against 0.000195342 s
unsteered). One build of the test project, the API dumps (Idrak.Abstraction, Idrak.Cluster) and the inventory regenerated,
then on the CPU the filter `cluster:|public API|abstraction inventory|cli health: …|cli polish: …|cli arabic: every
message|cli dev: trace|cli serve|profiler:|pretrain:|cli pretrain:|training:`: 126 passed, 0 failed, no process left.

Changes from the plan, kept: **the page is an embedded resource** (`dashboard.html`, `LogicalName` `Dashboard/dashboard.html`),
not a C# raw string like `ChatPage`: a page this size is edited as HTML, and it is still one file served from memory
with no external address. **Stop is a step boundary, not the cancellation token**: a token is seen by each rank at its own
point in a step, and a rank that cancels mid-step leaves the others waiting in a collective; the gate throws
`ClusterRunStoppedException` on every rank as it begins the next step, rank 0's pretrainer writes that step's checkpoint,
and `pretrain` ends with exit 0 ("stopped as asked"). **A steered in-process run needs the "sim" transport** (its steps
are where ranks wait); a forced other transport is refused naming why. **A replay of a steered run takes its event log**:
live faults are not drawn from the seed, so `--seed S --step N` alone cannot know them; the dashboard writes each run's
event log to `RUN/cluster-events.bin` (unless `IDRAK_CLUSTER_EVENT_LOG` names another file), and C8's `cluster replay`
takes it. **One in-process run at a time** (409 otherwise): the library's telemetry is the process's, so two in-process
runs' records could not be told apart. **The telemetry is summed before it is sent**: a transfer per message would flood
the page, so transfers are added up per pair and the collectives per rank, and sent as a `links` and a `collectives`
record together with each step's batch record. **Per-rank tokens a second are steps a second**: rank 0 alone publishes the
training telemetry (C2), so no rank's token count is published; a rank's card shows its steps a second from the control
gate's wall time, and the run's loss (the same token-weighted mean on every rank). **Access is decided per request**:
this machine's requests need no key whatever the address listened on, another machine's need a valid key (and none is
taken when no key is set, the refusal naming `idrak server keys add`); beyond the plan, a request from another site's page
(its `Origin`) and, on a loopback dashboard, one naming another host (DNS rebinding) are refused, since CORS off alone
does not stop a page from posting to 127.0.0.1. The test checks the per-request rule as a function (a test cannot make a
request from another machine) and the `Origin` and host refusals over HTTP. **The tests use a loopback Kestrel server**,
not an in-memory test server: the test project does not reference the test host, and the real listener is what the
owner runs. **The inspection gained the per-phase breakdown** (`PredictPhases`) and the pair matrix (`Pairs`) the owner's
five views need, played on the same virtual clock as the totals.

**After merging C6 (2026-10-18).** The stop of a sharded run: every rank stops at the same step boundary (the gate), and
every rank then takes part in its checkpoint as a saving step does in C6 (the weights gathered, the optimizer state
joined, rank 0 writes, a barrier), so the checkpoint holds every rank's share; a stop of an unsharded run is rank 0's
alone as before. The pretrainer now catches `ClusterRunStoppedException` (not any cancellation) at the step boundary.
"cluster: ui a sharded run (Shard = parameters) pauses … stop …" checks it against an uninterrupted sharded run: the
weights, the optimizer state and the losses byte for byte. Pause needs nothing more for a sharded run (ranks wait at the
gate, between steps). **Fixed, found while testing C9 (C4's code):** a rank with no rows in a step ran no backward pass,
so it never opened the overlap and sent the header before its buckets while the ranks that overlapped sent their buckets
first; with `IDRAK_CLUSTER_BUCKET_BYTES` set (more than one bucket), three ranks and two rows a step failed at once ("a
21848-byte message from rank 1 (tag -11) does not fit rank 2's 36-byte buffer"), and a measured size of several
buckets would have failed alike from the second step. The bucket layout never depended on rows; the order of the
collectives did. Every rank now opens the overlap in every step it takes part in, rows or not: the fine-tuner's step
without rows and its recorded step (`Synced`, the lost overlap's reason given after it opens) and the `Trainer`'s step
without a batch; `IGradientSync.Overlap`'s documentation states the rule. "cluster: a rank with no rows in a step takes part
in the bucketed reduction as the others do" (3 ranks, buckets of 4 KiB, two sequences a step) equals one process. Two
"cli health" checks the wider filter runs failed already on the main line and are fixed: `cluster sim` declared
`--seed`, a common option, as its own (C4); and `env set`, asked, found no variable for "port" once several names
held it (`IDRAK_CLUSTER_PORT`, now `IDRAK_CLUSTER_UI_PORT`): of several matches, the one named `IDRAK_` and the part
is taken. After the merge, one build, the dumps and the inventory regenerated, then on the CPU the filter `cluster:|public
API|abstraction inventory|cli health|cli polish|cli arabic: every message|cli dev: trace|cli serve|profiler:|pretrain:|cli
pretrain:|training:|fine-tuning:|callbacks:|offloading: HostOptimizer`: 175 passed, 0 failed, no process left. A sharded
run (parameters) paused at step 11 and stopped after 28 steps wrote the uninterrupted run's checkpoint.

Deferred, with the phase: a stop under pipeline placement writes no checkpoint (its checkpoint gathers every stage in a
collective the stopping ranks no longer run; C8, with the stage-own checkpoint); the fine-tuner and the `Trainer` do not
yet answer a stop with a checkpoint (`cluster sim` runs `pretrain` only; C8 brings `tune` and `train`); `idrak cluster
replay --events FILE` (C8, with the replay command); messages from the server (errors, run details) stay English on the
Arabic page; the owner's GPU run of the filter and a browser check of the page.

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
`calibrate` after C5's bench) beside C9 (the dashboard, two agents). Each phase is one
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
   once on the progress line, listed by `idrak suggest`, and never applied on its own. **Built in C8c (2026-10-18)**,
   see "C8c as built". **The default bf16 tolerance is bfloat16's own rounding step, 2^-7 (0.0078125), decided by the
   owner (2026-10-18)**: Idrak's bf16 ring rounds both the contributions it sends and the reduced sum it gathers, so a
   slice loses up to about two half-steps of its largest value (measured 1.47e-3, 2.17e-3 and 4.58e-3, all accepted); the
   fixture's DDP hook sums in float32 and lost 2.71e-3 in the same measure, which would have refused some of them.
5. **Decided (2026-10-18): `udp` stays out of C5**; the name stays registered with no library default.
6. **Decided (2026-10-18): no presets.** The topology file is built by `idrak cluster init`, an interactive builder
   with a description for every input and a default found on this machine, with options for every answer so it also
   runs without a terminal.
7. **Decided (2026-10-18): memory sizes in the topology file accept `B`, `KB`/`KiB`, `MB`/`MiB`, `GB`/`GiB`, `TB`/`TiB`,
   all as powers of 1,024** (device memory is counted that way, and `"16GB"` must mean the whole card); `inspect` shows
   each size in the largest unit it fills, or the one `--unit` names; a size without a unit is refused by the builder.
8. **Decided by the owner (2026-10-18): a web dashboard (C9)** sets up topologies, starts, pauses, stops and injects
   faults into simulated runs, and monitors every run, real `cluster start` runs across machines read-only. Opened by
   `idrak cluster ui` and by `--dashboard` on `idrak cluster sim`; built next, beside C8.

## Runs

Dated, newest last: date, commit, target (sim | localhost | LAN pair | rental), topology, what was measured, predicted
step time, measured step time, the calibration delta written.

| Date | Commit | Target | Topology | Measured | Predicted | Delta |
|---|---|---|---|---|---|---|
| | | | | | | |
