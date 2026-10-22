# Plan 6: Idrak.Network (several devices and machines on one job; several machines serving one API)

**Done (2026-10-22).**

**Superseded (2026-10-18) by [plan 23](23-cluster.md), Idrak.Cluster and Idrak.Network.** Kept as history only:
nothing is built from this file, and plan 23 does not depend on it. Where plan 23 reuses an idea from here (the fixed
summation order, the wire header, the HMAC hello, the rank's share of the draw) it restates it in its own terms.

**Status:** sketched earlier; detailed 2026-10-15 against the code at `abstraction` 82d84937; not started; superseded. Idrak runs
every model on one device: several GPUs in one machine can each run their own model (`Device.Cuda(n)`), but one
training job or one model cannot use more than one device, and the serving packages (`Idrak.AspNetCore`, the inference
engine) run on one machine.

**Goal.** A new package, `Idrak.Network`, that lets **several devices or machines work on one job** (distributed
compute, part 1) and **several machines serve one API** (cluster serving, part 2), built on the device, optimizer,
training-loop, telemetry and serving contracts the library already has, with every policy a setting and nothing tuned
to a card, a link or a memory size.

- **Part 1, distributed compute:** data parallel first (each rank takes its share of every batch, gradients are
  all-reduced over the link between the backward pass and the optimizer step, so every kind of training joins through
  one hook); then models too big for one device (decoder blocks placed across devices and machines, activations crossing
  the boundary, micro-batches); tensor parallel only where a measurement shows it pays.
- **Part 2, cluster serving:** nodes, placement, routing and health in front of the existing ASP.NET Core endpoints, and
  a model split across machines (part 1's placement) served as one node.

**Rules that hold for every phase** (`CLAUDE.md`, `docs/optimization.md`):

- **Card-agnostic.** No phase assumes a device, a vendor, a link speed or a memory size. Chunk sizes, in-flight counts
  and the number of pipeline micro-batches are measured on the running devices or derived from what the backend reports
  (`Backend.CreateHostStaging` slot sizes, `AvailableMemory`); no card, GB budget or Gb/s figure appears in code,
  comments, tests or docs. The traffic table below is arithmetic over link speeds, not a tuning.
- **Abstraction (decision 10).** Everything new is a contract with a `SlotTable` registry. Contracts two packages use
  (`Idrak.Nlp` trainers and `Idrak.Network`; `Idrak.AspNetCore` and `Idrak.Network`) live in `Idrak.Abstraction`;
  package-only ones in `Idrak.Network.Abstractions`. The inventory test and the public API guard (`api/*.txt`) are
  regenerated when they change; `api/Idrak.Network.txt` is added to the guard's package list.
- **Speed and memory first.** The wire format is bytes parsed with `BinaryPrimitives` and rented buffers returned in
  `finally` (rules 7-14, 25, 74); gradients are reduced in place in chunks, never copied whole (77: device memory does
  not grow across steps); every rank's wait on the link is measured by the library's telemetry (78), cold and warm apart
  (76); independent ranks and chunks are spread and the spread is checked (79); results are proven unchanged against a
  single-process run (71); sums are made in a fixed order (86) so every rank holds identical values.
- **No native libraries** (the rule the sketch kept, and plans/README.md's "rule every plan keeps"): NCCL, MPI and RDMA
  verbs are prebuilt native libraries, so the package does not use them. Machines talk through .NET's own sockets
  (`System.Net.Sockets`, TCP); GPUs in one machine copy through the driver each backend already drives (CUDA's peer
  copies where `cuDeviceCanAccessPeer`, host staging otherwise); faster transports come later as plug-ins registered
  into the same slot table.
- **Nothing fixed.** Every size, chunk, bucket, timeout, retry count, port, address, wire type, sync interval and
  placement is a library option (`NetworkOptions`, `PlacementOptions`, `ClusterOptions`) with an `IDRAK_NETWORK_*`,
  `IDRAK_PLACEMENT*` or `IDRAK_CLUSTER_*` environment variable, listed in `src/Idrak.Cli/Shared/EnvironmentVariables.cs`
  (a test scans the sources for names read and fails when one is missing there).
- **Tests.** Targeted CPU runs only (`IDRAK_DEVICES=cpu IDRAK_FILTER="network:"`); multi-process tests start their ranks
  as child processes over loopback; the owner runs CUDA and Vulkan and the two-machine run from the command lists at the
  end of each phase. Never the full suite.
- **The fix workflow.** Every phase: fix all errors first, update the tests the fixes touch, build once (only the
  projects that changed), run the phase's filter once with `--no-build`, commit and push on `abstraction`, hand the owner
  the GPU command list. At most two agents at a time.

## What Idrak has, and what is missing

Every row names the code the phases build on (file: line at 82d84937).

| Part | Today |
|---|---|
| A device as a slot | supported: `DeviceProvider` (`src/Idrak.Abstraction/Devices/DeviceProviders.cs:13`: `Kind`, `Count`, `Create(ordinal)` :31, `DeviceUuid` :52 so one GPU reached by two drivers is recognized), `DeviceProviders` as an unguarded `SlotTable` (:60-64: "a device never falls back to another"), `Device.Get`/`Parse` (`Device.cs:97`, :118), `Device.Backend` started on first use (:161), `Device.InUse` (:145). `Idrak.Gpu` registers CUDA, Vulkan and HIP from `LibraryRegistrations.RegisterFor` (`src/Idrak.Gpu/LibraryRegistrations.cs:23-35`) through `LibraryDefaults.Ensure` (`src/Idrak.Abstraction/LibraryDefaults.cs:44`), whose assembly list (:29-30) names `Idrak.Diffusion` and `Idrak.Audio` before they exist |
| What a backend must offer | supported: the minimum backend of plan 7 (`Backend.cs`: `Allocate` :482, `Return` :576, `Upload` :629, `Download` :632, `DownloadRange` :635, `Copy` :645, `Synchronize` :1672, `GetMemoryUsage` :623, `ReleaseCachedMemory` :626, `Capabilities` :485, `Hardware` :491); every other operation has a host fallback through `HostCall` (`Backend.Operations.g.cs`, generated by tools/operations/generate.py). Pinned staging for background copies: `CreateHostStaging(slots, slotFloats)` :570 and `IHostStaging` (`IHostStaging.cs:12`: `Download`, `Upload`, `Wait` per slot). **Missing**: a copy from one device's storage to another device's (today: `Download` to the host, then `Upload`); a device has no notion of a peer |
| Gradient buffers and the step | supported: `Optimizer.ZeroGrad` (`src/Idrak.Abstraction/Training/Optimizer.cs:30`), `Step` :47, `ClipAndStep(maxNorm)` :55, `GradientsZeroed` :118, `ClipGradientNormOnDevice` :171 (`SumSquares` + `ClipFactor` + `GroupScaleShift`, no host read), `SaveState`/`LoadState` :71/:80. `Adam.ClipAndStep` (`Adam.cs:77`) runs `FusedAdamW` (`Backend.cs:1632`; :102 in Adam) over every parameter in a few device passes and zeroes the gradients. `HostOptimizer` (`src/Idrak/Optimizers/HostOptimizer.cs:16`) already moves every gradient to the CPU in staged pieces (`Pipelined`, :95-140) and updates there. **Missing**: any exchange of gradients between devices or processes; the hook for one is the gap between `Backward` and `ClipAndStep` |
| The fine-tuning step | supported: `FineTuner.StepRunner` (`src/Idrak.Nlp/FineTuning.cs:1179`), `RunStep` :1584 (`ZeroGrad` :1593, one `Backward` per micro-batch :1612, `Update` :1624 which calls `ClipAndStep` :1579), the out-of-memory ladder :1253-1300, `GradientAccumulation` (:712, :1668). Recorded steps: `TrainingGraph.Record` :2348 records the forward and backward pass only (comment :782), replays through `Backend.ReplayGraph` :2479, and is kept only while the optimizer keeps the same gradient buffers (:1441-1453); `_graphsAllowed` :1196-1200 excludes `HostOptimizer` and offloading |
| The pretraining loop | supported: `Pretrainer.Train` (`src/Idrak.Nlp/Pretraining.cs:330`): stages, `SourceState` with `Pass`/`Next`/`Drawn` (:739-743) drawn in a seeded order (:787), the optimizer state loaded on resume (:401), `StepRunner.Run` per step (:457), a checkpoint on cancellation (:463), `Telemetry.BatchCompleted` per step (:495), `SaveCheckpoint` (:501, :800: a model folder, `optimizer.safetensors`, `pretrain-state.json` with every source's position), the device's peak bytes in the progress record (:505), `LatestCheckpoint` (:858). **Missing**: a rank's share of the draw, a checkpoint written by one rank only, a resume at another world size |
| Other training loops | `Trainer.Fit` (`src/Idrak/Training/Trainer.cs:206-209`: `Backward` then `Optimizer.Step`), the GPT sample's hand-written loop (`samples/Idrak.Samples.GptTraining/Program.cs:302-307`: `ZeroGrad`, `Backward`, `ClipGradientNorm`, `Step`). The same hook fits both |
| Model placement | supported for one device: `PretrainedOptions.Device` (`src/Idrak/Models/PretrainedModel.cs:106`), `PretrainedModel.Device` :244, `Load(folder, options)` :258, `Module.To(device)` moving every parameter (`src/Idrak.Abstraction/Modules/Module.cs:199`, `MoveTo` :209), `Tensor.To` (`Tensor.cs:279`). `DecoderBlock` (`src/Idrak/Layers/Decoder.cs:835`) is the unit a pipeline would place. **Missing**: a placement that puts blocks on different devices, and a module boundary whose forward sends and whose backward receives |
| Memory pool and telemetry | supported: `MemoryUsage` (InUse, Cached, Limit, Offloaded, Peak, Largest; `ComputeResources.cs:15`), `ComputeResources.GetMemoryUsage` :178, `ResetPeakMemoryUsage` :181, `GpuMemoryLimit` :69, `IDRAK_OFFLOAD` :85, `IDRAK_GPU_CHUNK_MB` :93, `MemoryAccountant` (`Backend.cs:2328`), `AvailableMemory` :2192, `FreeMemory` :2199. Telemetry: `TelemetryLevel` flags (`Diagnostics/Telemetry.cs:11`), `IsEnabled` :202 (one static read and an AND), `BatchCompleted(DataTime, ComputeTime)` (`Events.cs:42`), `DeviceFailed` :242, `HostFallback` :253. **Missing**: an event for a collective (bytes, duration, the wait on the slowest rank) |
| Serving on one machine | supported: `InferenceEngine` through `AddIdrak` (`src/Idrak.AspNetCore/IdrakServices.cs:19`), `IdrakBuilder` :39, `MapChatApi` (`IdrakEndpoints.cs:192`), `MapGenerate` :119, `MapPredictor` :103, `MapIdrakStatus` :247 (loaded models, queues, latency); `EngineHosting` (`src/Idrak.Abstraction/Serving/EngineModel.cs:10`: `Instances`, `SharedCopies`, `KeepAlive`, `QueueLimit`, `Timeout`), `ModelDescription(Name, Kind, Parameters, Device, ContextLength)` :45, `EngineModel<TCopy>` :77, `IModelCatalog` (`IModelCatalog.cs:14`), `ModelHost` leases (`ModelHost.cs:17`); `idrak serve` with `-H/--host`, `-p/--port`, `--api-key`, `IDRAK_PORT` (`src/Idrak.Cli/Commands/ServeCommands.cs:38-60`, `Serve/ServeHost.cs:92`); the KV cache kept between turns (`TextGenerator.KeepCache`, named at `src/Idrak.Nlp/Generation/ImageReader.cs:28-32`); KV formats as a registry (`KeyValueLayouts`, `src/Idrak.Abstraction/Generation/KeyValueLayouts.cs:158`). **Missing**: nodes, a gateway, routing, affinity, health across machines |
| Plug-in loading | supported: `-P`/`--plugin` common option (`src/Idrak.Cli/CommandContext.cs:23-30`, loaded :316), the "is not registered" message with the `-P` hint. `LibraryDefaults.Assemblies` (:29-30) lists first-party assemblies whose `LibraryRegistrations.RegisterFor` registers built-ins |
| Anything distributed | **missing**: the only "Network" types are `NetworkBuilder` and `INetworkBuilder` (`src/Idrak/Layers/NetworkBuilder.cs`), the neural-network builder, unrelated |

## Separate package or part of the library?

**Decision (this plan's): `src/Idrak.Network` is a separate package, on `Idrak.Abstraction` alone; its contracts live in
`Idrak.Abstraction`; transports are slot-table registrations.**

- **Why a package.** Sockets, a listening port, a shared-secret handshake and a wire protocol are a dependency and an
  attack surface the library must not carry into every app that trains a classifier (plans/README.md, "The rule every
  plan keeps"; CLAUDE.md, abstraction always). The GPU devices set the precedent: `Idrak.Gpu` holds the drivers and
  references only `Idrak.Abstraction` (decision 11); `Idrak.Network` does the same for the link.
- **Why the contracts are in Abstraction.** `Idrak.Nlp`'s `StepRunner`, core's `Trainer`, `Idrak.AspNetCore`'s
  gateway and `Idrak.Network` all use them: more than one library package (decision 10), so `Idrak.Abstraction.Network`
  (`ICollective`, `ProcessGroup`, `ITransport`, `Transports`, `PlacementPlan`, `Placements`) and
  `Idrak.Abstraction.Serving` (`INodeRegistry`, `NodeRegistries`, `IRequestRouter`, `RequestRouters`). A trainer takes a
  `ProcessGroup` as an option and never names a transport; with no group it runs exactly as today.
- **Why a library-default transport in Abstraction.** The `"local"` transport (N ranks in one process, no sockets) is
  the simulator every test runs on the CPU and the path for several GPUs in one machine (N4). It needs nothing beyond
  `Backend.Upload`/`Download`, so it lives in Abstraction as its default implementation, as the CPU device does
  (plan 10, decision 2). `"tcp"` is registered by `Idrak.Network`; a vendor library later (NCCL-like) is a plug-in that
  registers `"nccl"` into the same table and is loaded with `-P`.
- **`LibraryDefaults`.** `"Idrak.Network"` is added to `LibraryDefaults.Assemblies` and its `DynamicDependency` list
  (`LibraryDefaults.cs:29-43`), the way `Idrak.Diffusion` and `Idrak.Audio` are reserved: an app that ships
  `Idrak.Network` gets `"tcp"` registered the first time `Transports` is used; one that does not gets
  `"Transport 'tcp' is not registered (registered: local); add it with Transports.Register."` and the CLI's `-P` hint.
- **Does `Idrak` depend on it?** No (unlike `Idrak.Gpu`). `Idrak.Cli` references it so `idrak pretrain --nodes 2`
  works; apps add `Idrak.Network` when they want a link. The serving gateway lives in `Idrak.AspNetCore` (which
  references `Idrak.Network` for the node registry's TCP health checks) rather than a fourth package: one bridge over
  the engine, as decision 8 keeps it. Decision 5 below asks the owner to confirm.

## Where each part lives (decisions 8 and 10 of plan 10)

| Part | Package and namespace | Why |
|---|---|---|
| `ICollective`, `ProcessGroup`, `NetworkOptions`, `ITransport`, `Transports` (registry; `"local"` the library default), `WireFormat` (bytes), `CollectiveCompleted` event and `TelemetryLevel.Network` | `Idrak.Abstraction.Network`, `Idrak.Abstraction.Diagnostics` | used by Nlp, core, Network and AspNetCore |
| `PlacementPlan`, `PlacementOptions`, `IPlacement`, `Placements` (registry; `"single"`, `"pipeline"`), `PipelineBoundary` (the module whose forward sends and backward receives) | `Idrak.Abstraction.Network` and `Idrak.Abstraction.Modules` | core's `Load` and the trainers read the plan |
| `Backend.CopyTo(Storage, Backend, Storage, int)` with a host-staged default; CUDA's peer override | `Idrak.Abstraction.Devices` (`Backend.cs`, by hand beside `Upload`/`Download`), `Idrak.Gpu.Cuda` | a minimum backend needs nothing new (plan 7) |
| The TCP transport, rendezvous, the HMAC handshake, the ring all-reduce over sockets, the bandwidth bench | `Idrak.Network` (`Transport/`, `Collectives/`), registered by `Idrak.Network.LibraryRegistrations` | the dependency the library must not have |
| `FineTuningOptions.Group`, `PretrainingOptions.Group`, the all-reduce in `RunStep`, the rank's share of the draw, rank 0's checkpoints, `TrainingRun.DataParallel(group)`, `Trainer.Group` | `Idrak.Nlp`, `Idrak` (core `Training`) | the loops are theirs |
| `INodeRegistry`, `ClusterNode`, `NodeRegistries` (registry; `"static"`, `"file"`), `IRequestRouter`, `RequestRouters` (`"least-loaded"`, `"affinity"`), `ClusterOptions` | `Idrak.Abstraction.Serving` | AspNetCore and Network use them |
| The gateway: `AddCluster`, routing in `MapChatApi`/`MapCompletionsApi`/`MapPredictor`, streaming pass-through, `/cluster` status | `Idrak.AspNetCore` | the one HTTP bridge (decision 8) |
| `--node`, `--nodes`, `--coordinator`, `--transport` on `tune`, `train`, `pretrain`, `resume`, `distill`; `idrak cluster serve|nodes|ping|bench` | `Idrak.Cli` (`Commands/Train`, new `Commands/Cluster`) | the surface |
| References and fixtures | `tools/network/make_tiny_ddp.py`; `tests/Idrak.Tests/data/network/` | as plans 11, 12 and 19 did |
| Tests | `tests/Idrak.Tests/NetworkTests.cs` (`"network:"`), `NetworkCliTests.cs` (`"network: cli ..."`), `tests/Idrak.Tests/NetworkRank.cs` (the child-process entry the multi-process tests start) | one group prefix |

## Conventions every phase follows (read before coding)

- **A registry** copies `RandomTextSources` (`src/Idrak.Nlp/Abstractions/RandomTextData.cs:31`) and `DeviceProviders`:
  a `SlotTable<string, T>` named after the class, `comparer: StringComparer.OrdinalIgnoreCase`, library defaults through
  `RegisterDefault` inside `Overrides.AsLibraryDefaults` or `LibraryDefaults.Ensure(typeof(X))`, `[MethodImpl(NoInlining)]
  Register(T)` with `Assembly.GetCallingAssembly()`, `Unregister`, `Names`, `Find`, `Get`, `Default`, `Origin`,
  `Describe`. `Transports`, `NodeRegistries` and `Placements` are `unguarded:` ("a transport holds sockets and ranks'
  state; a group never falls back to another transport"); `RequestRouters` is guarded (`Throw` by default; two routers
  can answer the same call).
- **A contract suite** per registry in `src/Idrak.Abstraction.Testing/Contracts/`: `TransportSuite` (every collective
  against the plain sum on 2, 3, 4 and 8 ranks, odd sizes, empty buffers, a timeout), `PlacementSuite` (a plan covers
  every block exactly once), `RequestRouterSuite` (every request goes to a healthy node; affinity keeps a session).
- **Tests** are `(string Name, Action<Device> Run)[] NetworkGroup` entries added to `Tests.All`
  (`tests/Idrak.Tests/Program.cs:234`), names starting `"network: "`. Multi-process tests start ranks with
  `Process.Start(dotnet, "Idrak.Tests.dll --network-rank R …")` over loopback with a port the test picks (0: the OS
  chooses, reported back on stdout as bytes), wait with the group's timeout, and kill the children in `finally`.
- **Fixtures** come from `tools/network/make_tiny_ddp.py` (imports `dump_json`, `sha256` from `tools/vlm/make_tiny.py`),
  reruns write the same bytes, a `manifest.json` and a hand-written README; tolerances as plan 19: losses within 1e-4,
  gradients within 1e-4 of the largest, weights within 1e-5 (float32 on the wire); bfloat16 on the wire has its own
  looser tolerance written in the README from the fixture's own bfloat16 run.
- **A command** is an `internal sealed class XCommand : Command` (`src/Idrak.Cli/Command.cs:7`) in a `*Commands.All`
  list, with `Name`, `Summary` (an Arabic entry in `Messages.Arabic.cs`), `Usage` (Arguments, Options, Examples), 
  `ValueOptions`, `Flags`, `ShortForms`, `UsageException` for usage errors, `ProgressLine`, `ExitCodes`; new variables go
  in `EnvironmentVariables.All` under a new group `"Network"`; the tests "cli polish", "cli health", "cli arabic" check it.
- **Regenerate** after a public change: the inventory with
  `IDRAK_UPDATE_INVENTORY=1 IDRAK_DEVICES=cpu IDRAK_FILTER=inventory dotnet run -c Release --project tests/Idrak.Tests`
  and the API dumps with `IDRAK_UPDATE_API=1 IDRAK_DEVICES=cpu IDRAK_FILTER="public API" dotnet run -c Release --project tests/Idrak.Tests`.
- **Bytes on the wire.** Every message is parsed by a sync method over `ReadOnlySpan<byte>` (rules 8, 11, 25) from a
  rented buffer (rule 10); the async shell only awaits `Socket.ReceiveAsync(Memory<byte>)` and moves leftovers to the
  front (rule 9). Numbers are little-endian through `BinaryPrimitives`; nothing is formatted or parsed as text on the
  data path. `InvariantCulture` for the few text fields (addresses in messages to the user).
- **Telemetry.** `Telemetry.Collective(in CollectiveCompleted e)` at `TelemetryLevel.Network` per collective (group,
  operation, bytes sent and received, duration, the wait before the first byte arrived, the slowest rank), published
  only when `IsEnabled`; `BatchCompleted.DataTime` is not reused for it. The CLI's `log.jsonl` gets the records through
  `TelemetryJson.Write`, so `runs show` can chart the network share of each step.

## The rule it keeps

No native libraries. GPUs in one machine: the CUDA driver API already has peer-to-peer copies (`cuDeviceCanAccessPeer`,
`cuCtxEnablePeerAccess`, `cuMemcpyPeerAsync`) over PCIe or NVLink; collectives are the library's own kernels and copies;
where peer access is refused (many consumer cards) the copy goes through pinned host memory, slower, and the telemetry
says so once. Machines: .NET's own sockets, collectives written in C#. Faster transports (NCCL, InfiniBand/RDMA) come
later as plug-ins behind `ITransport`, like the NPU add-ons in the hardware plans.

---

# Part 1: distributed compute

## Phase N0: the reference and the simulator

**The reference.** `tools/network/make_tiny_ddp.py` writes `tests/Idrak.Tests/data/network/`:

- It reuses plan 19's fixture model and data (`tests/Idrak.Tests/data/pretrain/tiny-llama/`, `tokens.bin`), so no new
  model: 8 windows of 33 tokens (window i at i·stride as the pretrain fixture fixes it), every position trained.
- `ddp/`: PyTorch `DistributedDataParallel` with the `gloo` backend, 2 processes on the CPU (`torch.multiprocessing`,
  `MASTER_ADDR=127.0.0.1`, a free port), rank r taking windows r, r+2, r+4, r+6 (4 per rank), the mean token
  cross-entropy, plain SGD at 0.2 for 3 steps, then a second run with AdamW (lr 1e-3, betas 0.9/0.95, eps 1e-8, weight
  decay 0.1, no clipping): `losses` per rank per step (the DDP loss is each rank's own; the fixture also records their
  mean), `grads-step1.safetensors` (the averaged gradients every rank holds after step 1), `weights-step3.safetensors`
  for both optimizers, and `adamw/state-step3.safetensors`.
- `single/`: the same 3 steps in one process on all 8 windows at once (batch 8). Equal window lengths make DDP's mean
  of per-rank means equal the global token mean, so `single/` and `ddp/` must agree with each other within 1e-5: the
  script asserts it, which proves the reference before Idrak is compared with it.
- `bf16/`: the SGD run again with the gradients rounded to bfloat16 before averaging and summed in float32 (what N3's
  bfloat16 wire option does), so the looser tolerance for that option is measured, not guessed.
- `manifest.json` (versions, sizes, SHA-256), `README.md` (what the script did, tolerances, how to regenerate).

**The simulator.** `"local"` transport in `Idrak.Abstraction.Network` (`LocalTransport`): N ranks in one process, each
an `ICollective` sharing one in-memory exchange (`Channel<ReadOnlyMemory<float>>` per directed edge of the ring, bounded
to the in-flight count, so a rank that races ahead waits as it would on a socket). Every collective runs the same ring
algorithm the TCP transport runs (N2), over the same `WireFormat`, so the summation order, chunking and timeouts are
tested on the CPU without a network; only the bytes' carrier differs. `ProcessGroup.StartLocal(worldSize, devices)` makes
the N groups and runs each rank's work on its own thread (`Parallel.For` over ranks with `MaxDegreeOfParallelism =
worldSize`; rule 79).

**Tests** (`"network: "`, `NetworkTests.cs`): "network: ddp fixture agrees with single" (the script's own assertion
rerun on the saved files: a guard against a stale fixture); "network: local all-reduce equals the plain sum" (2, 3, 4, 8
ranks; sizes 1, 7, 1023, 65,537; sum, max; float32 bit-exact against a sum made in the ring's order on one thread);
"network: local broadcast, all-gather, barrier"; "network: a rank that never arrives times out naming the rank".

**Done when** the fixture folder exists with its README, two script runs write the same bytes, and the simulator's
collectives pass on the CPU.

## Phase N1: contracts, registries, the wire format, settings

**Contracts** (`src/Idrak.Abstraction/Network/`):

```csharp
public enum ReduceOp { Sum, Max, Min }
public enum WireType { Float32, BFloat16 }   // how values travel; sums are always made in float32

public interface ICollective : IDisposable
{
    int Rank { get; }
    int WorldSize { get; }
    void AllReduce(Storage buffer, int count, ReduceOp op, WireType wire = WireType.Float32);   // in place; every rank ends with identical values
    void Broadcast(Storage buffer, int count, int root);
    void AllGather(Storage input, int count, Storage output);                                   // output holds world · count, in rank order
    void ReduceScatter(Storage input, int count, Storage output, ReduceOp op);                  // input world · count, output count
    void Send(Storage buffer, int count, int to, int tag);
    void Receive(Storage buffer, int count, int from, int tag);
    void Barrier();
}

public sealed record NetworkOptions
{
    public string Transport { get; init; } = "local";       // a Transports name; IDRAK_NETWORK_TRANSPORT
    public int Rank { get; init; }                           // IDRAK_NETWORK_NODE
    public int WorldSize { get; init; } = 1;                 // IDRAK_NETWORK_NODES
    public string? Coordinator { get; init; }                // "host:port" rank 0 listens on; IDRAK_NETWORK_COORDINATOR
    public int ListenPort { get; init; }                     // this rank's port for ring connections (0: the OS picks, told to the coordinator); IDRAK_NETWORK_PORT
    public string? Key { get; init; }                        // the shared secret of the HMAC handshake; IDRAK_NETWORK_KEY (never logged)
    public TimeSpan Timeout { get; init; } = TimeSpan.FromSeconds(120);    // any wait for a peer; IDRAK_NETWORK_TIMEOUT (seconds)
    public int Retries { get; init; } = 3;                   // connection attempts at rendezvous; IDRAK_NETWORK_RETRIES
    public int? ChunkBytes { get; init; }                    // ring chunk (null: measured at first use); IDRAK_NETWORK_CHUNK_BYTES
    public int InFlight { get; init; } = 2;                  // chunks in flight per direction; IDRAK_NETWORK_IN_FLIGHT
    public int? SocketBufferBytes { get; init; }             // SO_SNDBUF/SO_RCVBUF (null: the OS default); IDRAK_NETWORK_SOCKET_BUFFER
    public IReadOnlyList<Device>? Devices { get; init; }     // local transport: one device per rank
}

public sealed class ProcessGroup : IDisposable
{
    public static ProcessGroup Start(NetworkOptions options);                       // Transports.Get(options.Transport).Connect(options)
    public static ProcessGroup[] StartLocal(int worldSize, IReadOnlyList<Device>? devices = null);
    public int Rank { get; }
    public int WorldSize { get; }
    public Device Device { get; }                                                   // this rank's device
    public ICollective Collective { get; }
    public ITransportInfo Info { get; }                                             // name, peers, measured bandwidth
    public bool IsRoot => Rank == 0;
    public void AllReduceGradients(IEnumerable<Tensor> parameters, WireType wire, ReadOnlySpan<float> weights = default);   // N3's hook, usable from a hand-written loop
}

public interface ITransport
{
    string Name { get; }
    string Summary { get; }
    ICollective Connect(NetworkOptions options, CancellationToken cancellationToken = default);
}
public static class Transports   // the registry: "local" the library default (unguarded)
```

**The wire format** (`WireFormat`, `Idrak.Abstraction.Network`, public so a plug-in transport speaks it): a 32-byte
little-endian header parsed by `WireFormat.TryReadHeader(ReadOnlySpan<byte>, out WireHeader)` and written by
`WriteHeader(Span<byte>, in WireHeader)`: magic `IDRK` (4), version (u16 = 1), kind (u16: hello, welcome, chunk, ack,
barrier, bye, error), rank (i32), sequence (u32), tag (i32), chunk index (u32), chunk count (u32), element type (u8:
f32, bf16), op (u8), payload bytes (u32); the payload follows. A hello carries a 32-byte HMAC-SHA256 of
(nonce ‖ rank ‖ world) under `Key`; the coordinator's welcome carries every rank's address and port in rank order (so
the ring's neighbours are known without a second round trip). Nothing is accepted from a peer whose hello failed. A
`version` the receiver does not know ends the connection with a clear message naming both versions.

**Settings and variables.** Every option above, plus the N3, N5 and N7 ones, is read from its variable by
`NetworkOptions.FromEnvironment()` (as `ComputeResources` reads `IDRAK_OFFLOAD` at `ComputeResources.cs:85`), and each
is listed in `EnvironmentVariables.All` with group `"Network"`, `IDRAK_NETWORK_KEY` marked `Secret: true`.

**Telemetry.** `TelemetryLevel.Network` (a new flag), `CollectiveCompleted(string Group, string Operation, int Rank, int
WorldSize, long BytesSent, long BytesReceived, TimeSpan Duration, TimeSpan Wait, int SlowestRank)`, and
`Telemetry.Collective(in CollectiveCompleted e)`.

**Tests** (`"network: "`): "network: wire header round trip (every field, bytes)"; "network: wire header refuses a
wrong magic and a newer version"; "network: a hello with a wrong key is refused" (the simulator exercises the handshake
too); "network: options from environment" (every variable read, the key never in `ToString`); "network: Transports
registry describes local and refuses an unregistered name with the Register hint"; the `TransportSuite` on `"local"`.
Inventory and API regenerated (Abstraction +1 namespace).

**Done when** the tests pass on the CPU and `IDRAK_FILTER="network:"` is the only run.

**Risks.** The header is a public contract from day one: keep `version` honoured on both sides so a plug-in transport
written against version 1 keeps working.

## Phase N2: the TCP transport

**Files.** `src/Idrak.Network/Idrak.Network.csproj` (references `Idrak.Abstraction` only; `IsAotCompatible`),
`LibraryRegistrations.cs` (`RegisterFor(typeof(Transports))` → `Transports.RegisterDefault("tcp", new TcpTransport())`),
`Transport/TcpTransport.cs`, `Transport/Rendezvous.cs`, `Transport/RingConnections.cs`, `Collectives/RingCollective.cs`,
`Collectives/ChunkPlanner.cs`, `Bench/Bandwidth.cs`.

**Rendezvous.** Rank 0 listens on `Coordinator`; every other rank connects, sends its hello (HMAC), learns every rank's
address from the welcome, and opens one connection to its ring successor (rank + 1 mod N) while accepting one from its
predecessor; `Retries` attempts `Timeout` apart. Nagle off (`NoDelay`), the socket buffers as set or the OS default,
`Timeout` applied to every receive.

**Ring all-reduce in a fixed order.** The gradient view is cut into N equal slices (the last padded in the plan, not in
memory); reduce-scatter: in step s (0 ≤ s < N-1) rank r sends slice (r - s) mod N to its successor and adds the slice it
receives from its predecessor into its own; so slice j's full sum is formed at rank (j - 1) mod N by adding ranks
j, j+1, …, in that order, whatever the timing (rule 86); all-gather then copies each finished slice around the ring, so
every rank ends with bit-identical values. Within a slice, chunks of `ChunkBytes` flow with `InFlight` chunks
outstanding per direction so both directions stay busy; the receive side sums in float32 straight from the rented
receive buffer into the device-bound staging slot (no second copy).

**Device buffers.** The collective takes a `Storage` on any device: it downloads each chunk through
`Backend.CreateHostStaging(slots, slotFloats)` (:570; pinned where the device has it, `DownloadRange` :635 otherwise),
sends, receives, sums on the host and uploads the finished slice once. The staging is created once per group and reused
every step (rule 77); its slot count is `InFlight + 1` per direction. On the CPU the buffer is the storage itself.

**bfloat16 on the wire.** `WireType.BFloat16`: the chunk is packed on the device with `PackBFloat16` (`Backend.cs:1744`,
half the bytes come down), sent as bfloat16 words, summed in float32 on arrival (`BFloat16Bits` unpacking, vectorized
with a scalar tail), and the finished slice goes back as float32. Partial sums travel as bfloat16 too (rounded once per
hop; the fixture's `bf16/` run measures the effect for N = 2; the README records the tolerance).

**Chunk size measured at first use.** With `ChunkBytes` null, the first all-reduce of a group tries three sizes derived
from the staging slot (the slot, a half, a quarter) over the first slice, keeps the fastest median of three and stores
it in the tuning cache under `network/tcp/<peer count>-<link id>` (`IDRAK_CACHE`, `IDRAK_TUNING_CACHE=0` off), as the
kernel widths are kept. No byte count is written into code.

**Bandwidth bench.** `Bandwidth.Measure(group, bytes, rounds)`: all-reduce bytes per second per transport and world
size, cold and warm apart (rule 76); reported through `CollectiveCompleted` and printed by `idrak cluster bench` (N8).

**Tests** (`"network: "`): "network: tcp ring all-reduce over loopback, 2 and 3 processes" (child processes, every
collective against the plain sum, sizes that are not multiples of the world size); "network: tcp bfloat16 wire sums in
float32 and matches the fixture's bf16 run"; "network: tcp rendezvous retries then fails naming the rank and address";
"network: tcp timeout ends the step on every rank naming the slow rank"; "network: tcp chunk size is measured once and
kept" (the cache file appears; a forced `ChunkBytes` skips the measurement); "network: tcp telemetry reports bytes and
wait per collective"; `TransportSuite` on `"tcp"` over loopback; "network: tcp is not registered without Idrak.Network"
(a test app over Abstraction alone). API dump `api/Idrak.Network.txt` added.

**Done when** the loopback tests pass on the CPU with `--no-build` after one build of Idrak.Network and the tests.

**Risks.** Loopback hides latency: the chunk-size measurement must rerun per link (the cache key includes the peers'
addresses). Windows Defender may prompt for the listening port: the README of the package says so.

## Phase N3: data-parallel training

**The hook.** `FineTuningOptions.Group` and `PretrainingOptions.Group` (`ProcessGroup?`, null today's behaviour). In
`RunStep` (`FineTuning.cs:1584`) after the last micro-batch's `Backward` (:1612) and before `Update` (:1624):

```csharp
group?.AllReduceGradients(optimizer.Parameters, options.GradientWire, weights: [trainedTokens]);
```

`AllReduceGradients` packs the trained parameters' gradient storages into one logical buffer in parameter order (no
copy: the ring's slices are addressed as (parameter, offset, count) pieces, as `HostOptimizer.Pipelined` addresses its
pieces), appends one float per rank holding this rank's trained-token count, and all-reduces the sum; then every rank
scales its gradients by (own count ÷ total count) before the sum, which is folded into the first chunk's pass, so the
result is the token-weighted mean: the gradient one process would compute on every rank's rows at once. With equal
counts it is DDP's mean. Clipping then runs on identical gradients (`ClipAndStep` :1579), `FusedAdamW` makes identical
updates, and the weights stay identical without being sent.

**Graphs.** A recorded step holds the forward and backward only (:782, `TrainingGraph.Record` :2348) and writes the
same gradient buffers every replay (checked at :1441-1453), so the all-reduce sits between `graph.Run` (:1460) and
`Update` (:1462) unchanged; `_graphsAllowed` (:1196) gains no new exclusion. Overlapping buckets with the backward pass
(a hook per parameter) cannot happen inside a replayed graph: the first version reduces after the backward in both
paths and reports the wait; bucketed overlap for ordinary steps is a later speed item measured against this baseline.

**HostOptimizer.** Its gradients already come down in staged pieces (`HostOptimizer.cs:95-140`); with a group, the
reduction runs on each downloaded piece before `Step(i)` (`ProcessGroup.Collective.AllReduce` on the CPU copy), so
nothing is downloaded twice. `GradientAccumulation` (:1668): the reduction happens once per optimizer step, after the
last micro-batch.

**Data sharded by rank.** `Pretrainer`: every rank draws the same seeded order (`SourceState.Order`, :787) and
`Draw` (:660) hands rank r the items at global draw positions k·W + r (a `Stride`/`Offset` on `SourceState`), so
`Pass`/`Next`/`Drawn` stay global and a checkpoint resumes at any world size (the next draw is the next global position).
`FineTuner.MakeBatches`: batch b goes to rank b mod W. `Trainer`/`TrainingRun` (`DataLoader`): a rank sampler with the
same stride. The GPT sample's loop: `group.AllReduceGradients(model.Parameters(), wire)` before `optimizer.Step()`.

**Starting state.** Rank 0 broadcasts every trained parameter at the start (a created model's random init differs per
process), or every rank loads the same checkpoint and the group compares a hash of the parameters (`SumSquares` per
parameter, all-gathered; a mismatch names the rank and the parameter).

**Mixed device kinds.** A CPU rank and a GPU rank round differently in `FusedAdamW`, so identical gradients give
slowly drifting weights. `NetworkOptions.SyncWeightsEvery` (`IDRAK_NETWORK_SYNC_WEIGHTS`, default 0: never) has rank 0
broadcast the weights every K steps; the hash check runs at each checkpoint and reports the largest difference per
parameter at `TelemetryLevel.Network`. A homogeneous group needs neither; the owner's loopback proof uses K.

**Checkpoints and resume.** Rank 0 writes `checkpoint-STEP/` (`SaveCheckpoint` :800) and the run folder's log; the other
ranks write nothing (a `Barrier` after the save so nobody starts the next step on a half-written folder). `Resume` on
every rank reads the same checkpoint (a shared path or a copy); the model and optimizer state are then identical by
construction and the positions are global, so a 2-rank run can resume as 3 ranks.

**Metrics and loss.** The reported loss is the token-weighted mean across ranks (one scalar all-reduce in the same
message as the counts); evaluation losses likewise, so every rank sees the same number and the same early-stopping
decision. Layers with running state (`BatchNorm`): statistics averaged at each epoch end (an option `SyncBatchNorm`,
default end of epoch); dropout masks differ per rank on purpose (the seed is combined with the rank).

**Telemetry and progress.** `BatchCompleted` per step as today; `CollectiveCompleted` per reduction; the progress line
shows the network share of the step and the slowest rank; `TrainingStarted.Device` names the rank's device.

**Tests** (`"network: "`): "network: data parallel 2 ranks (local) matches the ddp fixture" (step-1 averaged gradients
within 1e-4, 3 losses and `weights-step3` within 1e-5, SGD and AdamW); "network: data parallel equals one process with
the global batch" (the same on Idrak alone: a 1-rank run over 8 windows against a 2-rank run, within float rounding;
rule 71); "network: 3 ranks with unequal token counts give the token-weighted gradient"; "network: bfloat16 wire within
the fixture's bf16 tolerance"; "network: rank 0 alone writes the checkpoint and a 2-rank run resumes as 3 ranks giving
the 3-rank run's losses"; "network: a graph-replayed step reduces after the replay" (where `SupportsGraphs`; skipped on
the CPU with the reason printed); "network: HostOptimizer reduces on the downloaded pieces"; "network: mixed kinds sync
weights every K steps" (the simulator with one rank on the CPU and one on each listed device); "network: the GPT sample's
loop with AllReduceGradients" (a tiny `Network` built in the test); "network: TrainingRun.DataParallel on the Spirals
sample's network". Fine-tuning and pretraining filters still pass with `Group` null (no behaviour change).

**Done when** every test above passes on the CPU with `--no-build` after one build of Idrak, Idrak.Nlp, Idrak.Network and
the tests, and the owner's CUDA filter `"network:"` passes.

**Risks.** The token-weighted mean differs from DDP's mean when counts differ: documented, and the fixture uses equal
counts. A rank that dies stops the job: the timeout names it and the run resumes from the last checkpoint; elastic
restarts are not in this plan.

## Phase N4: several GPUs in one machine

**One process, one thread per device.** `ProcessGroup.StartLocal(worldSize, devices)` with `Devices = [cuda:0, cuda:1]`
(or any mix `Device.Available` lists; a device given twice is refused naming it; two names that resolve to one GPU
through `DeviceProvider.DeviceUuid` (:52), such as `cuda:0` and `vulkan:0` on one card, are refused too unless
`NetworkOptions.AllowSharedDevice`). Each rank's model copy lives on its device; the step runs on the rank's thread; the
local transport carries chunks between devices.

**Device-to-device copies.** `Backend.CopyTo(Storage source, Backend target, Storage destination, int count)` (new
virtual in `Backend.cs` beside `Upload`/`Download`; default: `DownloadRange` into a pooled host buffer then
`target.Upload`, so every backend works from day one, plan 7's pattern). `CudaBackend` overrides it with
`cuMemcpyPeerAsync` when `cuDeviceCanAccessPeer` says yes (`cuCtxEnablePeerAccess` once per pair, remembered), through
the host otherwise; the chosen path is reported once per pair at `TelemetryLevel.Devices` as the host fallbacks are
(`HostFallback`, `Events.cs:253`). Vulkan and HIP keep the default until measured.

**The local ring.** With `CopyTo`, the reduce-scatter's receive-and-add runs on the receiving device: the chunk arrives
in a device-side slot (one per in-flight chunk, allocated once), `Binary(Add)` sums it into the slice (fixed order as
N2), no host round trip when peer copies exist. When a backend has no peer path the host-staged default keeps the same
algorithm.

**Combined with N3.** A machine with several GPUs in a multi-machine job: ranks are (machine, device); the group reduces
inside the machine first (local), then across machines (TCP) with one rank per machine, then broadcasts inside: a
`"hierarchical"` composition in `Idrak.Network` over the two transports, chosen when `Devices` has more than one entry
and `Coordinator` is set.

**Tests** (`"network: "`): "network: local 2 devices give one device's losses with the global batch" (on every device
listed; on a machine with one GPU the second rank is the CPU and the test checks losses, not speed); "network: CopyTo
default equals Download then Upload on every device"; "network: a device named twice is refused; one GPU under two
providers is refused naming the UUID"; "network: hierarchical reduce equals flat reduce" (simulator). The owner's run:
`"network:"` on `cuda`; a two-GPU machine when one is available, recorded in the README's "Tested on" table.

**Done when** two devices in one process give the one-device losses with the global batch, and the step time with two
GPUs is measured and recorded where two exist.

**Risks.** Peer access refused on consumer cards: the host path is slower and the telemetry says so; nothing in the
library assumes either.

## Phase N5: models too big for one device

**Placement contract** (`Idrak.Abstraction.Network`):

```csharp
public sealed record PlacementPlan(IReadOnlyList<PlacementStage> Stages)   // each stage: the block range and its rank/device
{
    public PlacementStage StageOf(int block);
    public int Count => Stages.Count;
}
public sealed record PlacementStage(int Rank, Device Device, int FirstBlock, int BlockCount, bool HoldsEmbeddings, bool HoldsHead);
public sealed record PlacementOptions
{
    public string Kind { get; init; } = "single";       // Placements name: "single", "pipeline"; IDRAK_PLACEMENT
    public int? MicroBatches { get; init; }             // pipeline: micro-batches in flight (null: GradientAccumulation); IDRAK_PIPELINE_MICROBATCHES
    public IReadOnlyList<int>? Cuts { get; init; }      // explicit block cuts; null: balanced by measured parameter bytes
    public string Schedule { get; init; } = "gpipe";    // "gpipe" now, "1f1b" later; IDRAK_PIPELINE_SCHEDULE
}
public interface IPlacement { string Name; string Summary; PlacementPlan Plan(DecoderSpec spec, IReadOnlyList<(string Name, long Bytes)> blocks, ProcessGroup group, PlacementOptions options); }
public static class Placements   // registry: "single" (everything on group.Device), "pipeline"
```

**Loading.** `PretrainedOptions.Placement` and `PretrainedOptions.Group` (`PretrainedModel.cs:105`): `Load` asks
`Placements.Get(kind).Plan(...)` and builds each rank's stage only (its blocks' tensors read, the others skipped, so a
rank holds a fraction of the weights: the point of the phase), with `PipelineBoundary` modules between stages. The
`"pipeline"` plan cuts the block list so stages hold equal parameter bytes (measured from the checkpoint's tensor sizes,
not assumed), the embeddings on the first stage, the head on the last. `model.Device` is the rank's device.

**Activations across the boundary.** `PipelineBoundary : Module` whose forward on stage s sends the activation
`[rows, positions, dim]` to rank s+1 (`Send` with a tag per micro-batch) and whose backward receives the gradient back
(`Autograd.Function` with a `Receive` in its backward), so `Tensor.Backward` drives the whole pipeline with no other
change to the decoder. Inference: a token's hidden state passes through the stages in turn; each stage keeps its own
KV cache (`KeyValueLayouts` unchanged); the last stage samples and sends the token id back to the first (one small
message per token).

**Training schedule.** GPipe: `MicroBatches` forward passes flow through the stages, then their backward passes in
reverse; each stage holds the activations of the micro-batches in flight, so the stage's memory is the per-micro-batch
activation times the count (the ladder of `StepRunner` applies per stage: recompute, bfloat16 activations,
checkpointing, each measured on that stage's device). `MicroBatches` defaults to `GradientAccumulation`, so a run that
already accumulates pays no extra memory. The optimizer step is local to each stage (no gradient exchange: a stage owns
its parameters); with data parallel on top (N3), ranks that hold the same stage all-reduce within their stage group.

**Sharded optimizer state (ZeRO-1 style).** With data parallel, each rank keeps the Adam moments of a slice of the
parameters (`Optimizer.Parameters` partitioned by rank in parameter order), reduce-scatters the gradients, updates its
slice, all-gathers the updated weights: an option `NetworkOptions.ShardOptimizerState` (`IDRAK_NETWORK_SHARD_STATE`),
cutting the moments' memory by the world size at the cost of one all-gather of the weights per step; the memory saved
and the time paid are both in the telemetry.

**Tests** (`"network: "`): "network: pipeline 2 stages (local, CPU) gives the one-device logits and losses of the tiny
Llama" (plan 19's fixture; within 1e-5); "network: pipeline backward gives the one-device gradients" (every parameter,
1e-4); "network: micro-batches 1, 2 and 4 give the same loss"; "network: a plan covers every block once with the
embeddings first and the head last" (`PlacementSuite`); "network: chat through a 2-stage pipeline answers as one device
does" (greedy, same tokens); "network: sharded optimizer state equals the unsharded weights after 3 steps";
"network: a rank holds only its stage's tensors" (`GetMemoryUsage().InUse` of each rank against the stage's bytes).

**Done when** a model split in two stages loads with each rank holding its stage's bytes, chats and fine-tunes with LoRA
to the one-device result on the CPU; the owner's run splits a model that does not fit one of the machine's devices
across that device and the CPU (slow, but the proof that the split is correct), and across two GPUs when two exist.

**Risks.** The per-token message in inference adds a round trip per stage per token: pipeline serving pays latency for
capacity, said in the honest section. GPipe's bubble is 1 - MicroBatches ÷ (MicroBatches + Stages - 1) of the step: the
telemetry shows the idle time per stage, and 1F1B comes only if that number justifies it.

## Phase N6: tensor parallel (only where it pays)

**What.** Attention heads and feed-forward columns split across ranks with an all-reduce per layer (two per block:
after attention and after the feed-forward), `Placements` name `"tensor"`, `PlacementOptions.TensorRanks`.

**The measurement that decides it.** `idrak cluster bench --placement pipeline` against `--placement tensor` on the
same two devices and the same model: tokens per second decoding and per training step, cold and warm apart. Tensor
parallel stays in only if it is at least 10% faster than the pipeline on the devices measured (a setting of the bench,
not of the library); the expectation, to be confirmed: over TCP it loses (an all-reduce of the hidden state per layer
per token), over peer-copying GPUs in one machine it may win. The result goes into this plan's "as built" section with
the numbers; nothing is merged before the measurement exists.

**Tests** (`"network: "`): "network: tensor parallel 2 ranks gives the one-device logits" (local, CPU); the bench prints
both placements.

**Done when** it beats the pipeline on a measured setup; otherwise it stays out and the measurement is recorded.

---

# Part 2: cluster serving

## Phase N7: cluster serving

**Contracts** (`Idrak.Abstraction.Serving`):

```csharp
public sealed record ClusterNode(string Name, Uri Address, IReadOnlyList<string> Devices, IReadOnlyList<string> Models, double Load, bool Healthy, DateTimeOffset Seen);
public interface INodeRegistry : IDisposable
{
    string Name { get; }
    IReadOnlyList<ClusterNode> Nodes { get; }
    event Action<ClusterNode>? Changed;                     // health or load changed
    ValueTask RefreshAsync(CancellationToken cancellationToken = default);
}
public static class NodeRegistries    // registry: "static" (a fixed list), "file" (a JSON file re-read when it changes); IDRAK_CLUSTER_NODES
public interface IRequestRouter { string Name; ClusterNode Choose(IReadOnlyList<ClusterNode> healthy, RouteKey key); }
public readonly record struct RouteKey(string Model, string? Session, ulong PrefixHash);
public static class RequestRouters    // registry: "least-loaded", "affinity" (consistent hashing on Session, else PrefixHash, so a chat's next turn lands where its KV cache is); IDRAK_CLUSTER_ROUTER
public sealed record ClusterOptions
{
    public string Registry { get; init; } = "static";
    public string Router { get; init; } = "affinity";
    public TimeSpan HealthEvery { get; init; } = TimeSpan.FromSeconds(5);     // IDRAK_CLUSTER_HEALTH_SECONDS
    public int FailuresBeforeUnhealthy { get; init; } = 3;                    // IDRAK_CLUSTER_FAILURES
    public TimeSpan RequestTimeout { get; init; } = TimeSpan.FromMinutes(5);  // IDRAK_CLUSTER_TIMEOUT
    public int MaxInFlightPerNode { get; init; }                              // 0: the node's own queue limit; IDRAK_CLUSTER_IN_FLIGHT
    public IReadOnlyDictionary<string, int> Replicas { get; init; }           // model name -> copies wanted across nodes
}
```

**The gateway** (`Idrak.AspNetCore`): `builder.Services.AddIdrak().AddCluster(c => c.Node("gpu-a", "http://10.0.0.5:8080").Node("gpu-b", "http://10.0.0.6:8080"))`
(the sketch's API) registers the node registry and router; `MapChatApi`, `MapCompletionsApi`, `MapGenerate` and
`MapPredictor` route to a node when a cluster is configured and serve locally otherwise: the request body is forwarded as
bytes (not re-serialized), the response streamed through as chunks arrive (`HttpClient` with
`HttpCompletionOption.ResponseHeadersRead`, a rented buffer per stream), a node that fails mid-stream is marked after
`FailuresBeforeUnhealthy` and the next request goes elsewhere. Health reads each node's `MapIdrakStatus` (:247) every
`HealthEvery`: loaded models, queue depth, latency, memory (`MemoryUsage`), which is the `Load`. Placement: `Replicas`
says how many nodes should hold each model; the gateway asks nodes to load (`server load` already exists:
`ServerCommands.cs:242`) on the least-loaded nodes that have the device memory the model's `ModelDescription` reports.
`/cluster` (`MapClusterStatus`) lists nodes, models and routing counts; `Telemetry.Engine` gets one `EngineEvent` per
routed request with the node named in `Model` as `model@node`.

**Affinity.** The chat API's session id (or, without one, a hash of the first 64 tokens' bytes of the rendered prompt)
is the `RouteKey`; the "affinity" router sends it to the same healthy node while it stays healthy, so a second turn
hits the node that holds the kept KV cache (`TextGenerator.KeepCache`); the telemetry counts affinity hits.

**One model across machines (joins part 1).** A pipeline group (N5) registers as one node: rank 0's HTTP host is the
node's address, its `ModelDescription.Device` lists every stage's device, and the gateway routes to it like any node.

**Tests** (`"network: "`): "network: cluster gateway in front of two in-process nodes serves a chat on both"
(`WebApplicationFactory` as `AspNetCoreTests.cs` uses it; two engines, the gateway over them); "network: a stopped node
is skipped after N failures and comes back"; "network: affinity keeps a session on one node and reports the hits";
"network: least-loaded picks the shorter queue"; "network: replicas place a model on two nodes"; "network: streaming
passes through chunk by chunk" (the client sees the first token before the last); `RequestRouterSuite`; "network: the
file registry re-reads on change".

**Done when** a gateway in front of two nodes serves a model on both and keeps serving when one stops, with throughput
with two nodes measured against one in the owner's run (two processes on one machine first, then two machines).

**Risks.** The gateway is one more hop of latency per request: measured and shown in `/cluster`. Authentication: nodes
accept the gateway's `--api-key` as today; nothing listens without one when `--host` is not loopback (the existing serve
rule applies to `cluster serve`).

## Phase N8: the command line and the owner's proof run

**Options on the training commands** (`tune`, `train`, `pretrain`, `resume`, `distill`; `TuneCommand.cs`,
`TrainingCommands.cs`, `PretrainCommand.cs`, `DistillCommand.cs`): `--nodes N` (world size), `--node N` (this
process's rank), `--coordinator HOST:PORT`, `--transport NAME` (default `tcp` when `--coordinator` is set, else
`local`), `--devices cuda:0,cuda:1` (the local transport's one device per rank, ranks in one process), `--wire f32|bf16`,
`--sync-weights N`, `--placement single|pipeline|tensor`, `--micro-batches N`. Not `--rank`: `tune` and `distill`
already use `--rank` for the LoRA rank (`TuneTool.cs:53`, `DistillCommand.cs:30`), and the conventions give an option
one meaning across commands. The key comes from `IDRAK_NETWORK_KEY` only (never an option: it would show in process
lists). Each option maps to `NetworkOptions`/`PlacementOptions`, the environment variables fill in what is not given.

**`idrak cluster` subcommands** (`Commands/ClusterCommands.cs`, `Commands/Cluster/`): `cluster serve MODEL... --nodes
FILE|LIST [-H] [-p] [--router NAME] [--replicas MODEL=N]` (the gateway; the serve options of `ServeCommand.OptionsHelp`
apply), `cluster nodes [--nodes FILE]` (the table, `-j` JSON), `cluster ping HOST:PORT` (health of one node), `cluster
bench --coordinator HOST:PORT --node N --nodes N [--bytes N] [--rounds N] [--placement pipeline|tensor MODEL]` (all-reduce
bandwidth per transport and world size, cold and warm; the N6 decision's numbers). Arabic summaries, `EnvironmentVariables`
group `"Network"`, help within 120 columns.

**Tests** (`"network: cli ..."`): "network: cli pretrain --nodes 2 --node 0/1 over loopback writes one run folder from
rank 0" (two child processes through `CommandLine.Run`'s process form); "network: cli tune with --devices cpu,cpu runs
two ranks in one process"; "network: cli cluster serve routes to two nodes"; "network: cli cluster bench prints bytes per
second"; "network: cli --nodes without --coordinator is a usage error naming it"; "network: cli refuses cuda:0 twice";
`help` for each passes the polish tests.

**The owner's proof run** (from `D:\Projects\Idrak`, one GPU). What is **not** a valid proof: two ranks on `cuda:0`
(two contexts on one card share its compute and memory: nothing is parallel), or `cuda:0` with `vulkan:0` on the same
card (the same GPU under two drivers; the group refuses it by UUID). What the owner can run:

1. The simulator on the CPU, then the GPU filters:

```powershell
cd D:\Projects\Idrak
git pull
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES="cpu"; $env:IDRAK_FILTER="network:"; dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES="cuda"; $env:IDRAK_FILTER="network:"; dotnet run -c Release --no-build --project tests\Idrak.Tests
$env:IDRAK_DEVICES="vulkan"; $env:IDRAK_FILTER="network:"; dotnet run -c Release --no-build --project tests\Idrak.Tests
```

2. Two processes on one machine over loopback TCP, rank 0 on `cuda:0` and rank 1 on the CPU, with plan 19's A5 model
and tokens (`work\model`, `work\tokens.bin`). The step waits for the CPU rank, so this proves correctness and the
telemetry, not speed: the two-rank losses must equal a one-process run with twice the batch tokens, and
`runs compare` shows it. Two PowerShell windows (or `Start-Process` for the second):

```powershell
$env:IDRAK_NETWORK_KEY = "a long shared secret of your own"
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\model --tokens work\tokens.bin --steps 50 --batch-tokens 4096 --lr 6e-4 --seed 1 -o work\single -d cuda:0
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\model --tokens work\tokens.bin --steps 50 --batch-tokens 2048 --lr 6e-4 --seed 1 -o work\dp --nodes 2 --node 0 --coordinator 127.0.0.1:29500 --sync-weights 10 -d cuda:0
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\model --tokens work\tokens.bin --steps 50 --batch-tokens 2048 --lr 6e-4 --seed 1 -o work\dp-rank1 --nodes 2 --node 1 --coordinator 127.0.0.1:29500 --sync-weights 10 -d cpu
dotnet run -c Release --project src\Idrak.Cli -- runs compare work\single work\dp
dotnet run -c Release --project src\Idrak.Cli -- cluster bench --coordinator 127.0.0.1:29500 --nodes 2 --node 0 --bytes 67108864
dotnet run -c Release --project src\Idrak.Cli -- cluster bench --coordinator 127.0.0.1:29500 --nodes 2 --node 1 --bytes 67108864
```

(`--sync-weights 10` because a CPU rank and a CUDA rank round the AdamW update differently; a run of two GPUs or two
machines with the same device kind leaves it at 0.) The same for a LoRA tune:
`tune train work\model work\chats.jsonl -o work\adapters --nodes 2 --node 0 --coordinator 127.0.0.1:29500 -d cuda:0`
and `--node 1 -d cpu`.

3. The pipeline on one machine, the model split between `cuda:0` and the CPU (slow; proves the split):

```powershell
dotnet run -c Release --project src\Idrak.Cli -- run work\model "..." --placement pipeline --devices cuda:0,cpu
dotnet run -c Release --project src\Idrak.Cli -- cluster bench --placement pipeline work\model --devices cuda:0,cpu
```

4. Two machines when a second PC is on the LAN (`192.168.1.10` stands for the first PC's address; the same command on
each PC with its own `--node`; the key in `IDRAK_NETWORK_KEY` on both):

```powershell
dotnet run -c Release --project src\Idrak.Cli -- pretrain work\model --tokens work\tokens.bin --steps 200 --batch-tokens 4096 -o work\lan --nodes 2 --node 0 --coordinator 192.168.1.10:29500 --wire bf16 -d cuda:0
dotnet run -c Release --project src\Idrak.Cli -- cluster bench --coordinator 192.168.1.10:29500 --nodes 2 --node 0 --bytes 268435456
```

and the gateway over two serving PCs:

```powershell
dotnet run -c Release --project src\Idrak.Cli -- serve work\model -H 0.0.0.0 -p 8080 --api-key KEY
dotnet run -c Release --project src\Idrak.Cli -- cluster serve work\model --nodes http://192.168.1.10:8080,http://192.168.1.11:8080 -p 7317 --api-key KEY
dotnet run -c Release --project src\Idrak.Cli -- cluster nodes --nodes http://192.168.1.10:8080,http://192.168.1.11:8080
```

The measured numbers (bytes per second per link, step time with 1 and 2 ranks, tokens per second with 1 and 2 nodes)
go into the README's "Tested on" table and this plan's "as built" section.

**Done when** the CLI tests pass on the CPU, the four `cli` health filters still pass, and the owner's steps 1 to 3
run; step 4 when a second machine exists.

---

## Memory and speed, where every byte goes

| Where | What is held | Bound by | Measured with |
|---|---|---|---|
| Gradient reduction (N2, N3) | the gradients themselves, reduced in place slice by slice: no second copy of the gradients; `InFlight + 1` staging slots per direction of `ChunkBytes` each, allocated once per group and reused every step (rule 77) | `ChunkBytes` (measured at first use or set) × `InFlight` | `GetMemoryUsage().InUse/Peak` per rank, `CollectiveCompleted.Bytes` |
| bfloat16 on the wire | one packed chunk per in-flight chunk (half the floats' bytes) on the device | the same two settings | the same |
| HostOptimizer with a group | nothing new: the reduction runs on the pieces it already downloads | `HostOptimizer`'s 16 MiB slots (host side, not a card property) | `CollectiveCompleted.Duration` against the step |
| Pipeline (N5) | each stage holds its blocks' weights (a fraction of the model) and the activations of `MicroBatches` micro-batches; the ladder (recompute, bfloat16 activations, checkpointing) applies per stage on its own device | `MicroBatches`, the stage's `AvailableMemory` | each rank's `Peak`, the progress record's `DeviceBytes` |
| Sharded optimizer state | the moments of 1/W of the parameters per rank | `ShardOptimizerState` | `Peak` before and after |
| Gateway (N7) | one rented buffer per streamed response, returned in `finally`; no body kept whole | `MaxInFlightPerNode` | `/cluster` and `EngineEvent` |

Speed: every collective's duration and its wait on the slowest rank are in the telemetry (rule 78) and in the run log,
so "how much of each step is the network" is a number the owner reads, not a guess; cold (first step: handshake, chunk
measurement, staging allocation) and warm steps are reported apart (rule 76). Independent work is spread (ranks on
threads in the local transport, chunks in flight on sockets, both directions of the ring busy) and the first step
reports what the spread paid (rule 79). Nothing above names a card or a GB figure; the one byte count in the design, the
staging slot, is the backend's own report or the measured chunk.

**Traffic per step (arithmetic, not tuning).** Each rank sends and receives 2·(N-1)/N of the trained values per step:

| What is trained | Trained values (about) | bfloat16 per step | at 0.11 GB/s (1 Gb/s) | at 1.1 GB/s (10 Gb/s) |
|---|---|---|---|---|
| A classifier or regressor (`Network`, a few million weights) | ~2 M | ~4 MB | ~0.04 s | ~0.004 s |
| LoRA rank 16 on a 0.5B model | ~9 M | ~18 MB | ~0.16 s | ~0.02 s |
| A GPT trained from scratch, 100 M weights | ~100 M | ~200 MB | ~1.8 s | ~0.18 s |
| LoRA rank 16 on a 7B model | ~40 M | ~80 MB | ~0.7 s | ~0.07 s |
| Every weight of a 0.5B model | ~500 M | ~1 GB | ~9 s | ~0.9 s |

Where the transfer is a large share of the step, `GradientAccumulation` (already in `FineTuningOptions` and
`PretrainingOptions`) exchanges every K micro-batches instead of every one; the choice is the user's and the telemetry
shows its effect.

## Settings and environment variables

| Option | Variable | Default |
|---|---|---|
| `NetworkOptions.Transport` | `IDRAK_NETWORK_TRANSPORT` | `local` (`tcp` when a coordinator is set) |
| `Rank`, `WorldSize`, `Coordinator`, `ListenPort` | `IDRAK_NETWORK_NODE`, `IDRAK_NETWORK_NODES`, `IDRAK_NETWORK_COORDINATOR`, `IDRAK_NETWORK_PORT` | 0, 1, none, 0 (the OS picks) |
| `Key` | `IDRAK_NETWORK_KEY` (secret) | none: nothing listens without it |
| `Timeout`, `Retries` | `IDRAK_NETWORK_TIMEOUT`, `IDRAK_NETWORK_RETRIES` | 120 s, 3 |
| `ChunkBytes`, `InFlight`, `SocketBufferBytes` | `IDRAK_NETWORK_CHUNK_BYTES`, `IDRAK_NETWORK_IN_FLIGHT`, `IDRAK_NETWORK_SOCKET_BUFFER` | measured, 2, the OS default |
| `GradientWire` | `IDRAK_NETWORK_WIRE` (`f32`, `bf16`) | decision 3 |
| `SyncWeightsEvery`, `ShardOptimizerState`, `SyncBatchNorm` | `IDRAK_NETWORK_SYNC_WEIGHTS`, `IDRAK_NETWORK_SHARD_STATE`, `IDRAK_NETWORK_SYNC_BATCHNORM` | 0, off, epoch |
| `PlacementOptions.Kind`, `MicroBatches`, `Schedule` | `IDRAK_PLACEMENT`, `IDRAK_PIPELINE_MICROBATCHES`, `IDRAK_PIPELINE_SCHEDULE` | `single`, accumulation, `gpipe` |
| `ClusterOptions.Registry`, `Router`, `HealthEvery`, `FailuresBeforeUnhealthy`, `RequestTimeout`, `MaxInFlightPerNode` | `IDRAK_CLUSTER_NODES`, `IDRAK_CLUSTER_ROUTER`, `IDRAK_CLUSTER_HEALTH_SECONDS`, `IDRAK_CLUSTER_FAILURES`, `IDRAK_CLUSTER_TIMEOUT`, `IDRAK_CLUSTER_IN_FLIGHT` | `static`, `affinity`, 5, 3, 300, the node's own |
| The tuning cache for the measured chunk | `IDRAK_CACHE`, `IDRAK_TUNING_CACHE` (existing) | `~/.cache/idrak/network/` |

## Order and parallelism

N0 → N1 → N2 and N3 in parallel (two agents: N2 is `Idrak.Network` only, N3 is `Idrak.Nlp` and core on the local
transport; N3's TCP tests run once N2 lands) → N4 → N5 → N8 (the training options can land with N3; `cluster` with N7)
→ N7 (AspNetCore only, can run beside N4 and N5) → the owner's proof → N6 last, and only on its measurement. Each phase
is one commit series on `abstraction` with its "as built" section appended to this plan (what was built, filter names
and counts, what is left, the owner's commands), as plans 11 to 13 and 19 did.

Effort (rough): N0 to N3 four to six weeks; N4 two to three; N5 four to six; N7 three to five; N8 one to two; N6 three
to four if measured worthwhile.

## What distributed training buys and costs at the owner's scale, honestly

- **With one GPU, part 1 buys no speed.** A second rank on the CPU slows the step to the CPU's pace; the loopback runs
  above prove correctness, the telemetry and the resume, nothing more. The data-parallel path pays only with a second
  GPU in the machine (N4) or a second machine (N2, N3), and across machines only when the gradient traffic is a small
  share of the step: LoRA and small models on any link, full training of larger models on 10 Gb/s or with accumulation.
- **The pipeline (N5) buys capacity, not speed.** A model too big for one device runs; each token pays a hop per stage
  and a training step pays the bubble. A stage on the CPU makes the whole model run at the CPU's pace. Its value on the
  owner's machine is a model that otherwise does not load; its speed value waits for a second GPU.
- **Part 2 buys throughput and availability with two serving machines**, and nothing with one; the gateway on one
  machine is a hop of latency.
- **What is certain to be delivered by the plan** is the machinery proven at tiny scale against PyTorch DDP and against
  Idrak's own single-process runs, with every policy a setting and every cost a telemetry number, so that the day a
  second device or machine exists, the owner's run is one command line away and the measurement decides the rest.

## Decisions (for the owner)

1. **One package or two for serving?** The gateway in `Idrak.AspNetCore` (one bridge, decision 8; it gains a reference
   to `Idrak.Network`) or a separate `Idrak.Network.Serving`. **Recommended: in `Idrak.AspNetCore`.**
2. **Option names on the training commands.** `--node`/`--nodes`/`--coordinator` (since `--rank` is the LoRA rank in
   `tune` and `distill`), or `--rank`/`--world` with the LoRA option renamed. **Recommended: `--node`/`--nodes`;** the
   LoRA option keeps its meaning.
3. **The wire type's default.** bfloat16 on TCP (half the traffic; the fixture's `bf16/` run gives its tolerance) and
   float32 on the local transport, or float32 everywhere until measured. **Recommended: float32 everywhere by default,
   `--wire bf16` opt-in;** the proof runs compare both and the default flips in the "as built" section if the bf16
   losses stay within the fixture's tolerance on the owner's run.
4. **Mixed device kinds.** Allow a CPU rank beside a GPU rank (the only two-rank proof on one GPU) with periodic weight
   sync, or refuse mixed kinds. **Recommended: allow, with `--sync-weights` and the hash check at each checkpoint;** the
   slow rank is visible in the telemetry.
5. **Does `Idrak` depend on `Idrak.Network`** (as it does on `Idrak.Gpu`), so every app has `"tcp"`? **Recommended:
   no;** the CLI and apps that want a link reference it; `Transports.Get("tcp")` without it says "not registered" with
   the `-P` hint, as a family does.
6. **Tensor parallel's gate.** The 10% margin over the pipeline on the measured devices, or any gain. **Recommended:
   10%;** below it the extra code path is not worth keeping.
7. **The first transport beyond TCP.** None planned; a vendor collective library as a `samples/` plug-in only if a
   measured run shows TCP or peer copies as the limit. **Recommended: wait for the measurement.**
