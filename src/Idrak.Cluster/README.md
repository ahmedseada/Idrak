# Idrak.Cluster

A cluster of GPUs and machines simulated on one machine (plan 23), so distributed training is proven locally before any
real link is used, and a run predicts the network time of a cluster you do not own.

```bash
dotnet add package Idrak.Cluster   # the simulator, on Idrak.Abstraction
dotnet add package Idrak.Network   # the collective algorithms and the real transports
```

## What is in the package

| Part | What it does |
|------|--------------|
| `SimulatedDevices` (`sim:N`) | A device carved from a physical one (`Device.CreateSeparateBackend`): its own memory pool and a hard cap, the physical device's type and kernels. Past the cap an allocation fails, or spills to pinned host memory where the device can (CUDA) |
| The `sim` transport | Ranks that are threads of one process, joined by bounded queues; every transfer charged on a `VirtualClock` from the topology's link profiles |
| `VirtualClock` | Network time per rank: bytes / bandwidth + latency + jitter, one transfer at a time per direction, loss sent again; deterministic for a seed (jitter and loss hashed from the message, never from thread timing) |
| `InProcessCluster` | Runs work on every rank of a topology, each on its own thread and carved device (or all carved from `SimulationOptions.CarveFrom`), over the transport the selector plans per pair, with the collectives `ClusterOptions.CollectiveFor` names (hierarchical across nodes, else ring) |
| Built-in link profiles | Rates only where the name states them (`ethernet-1g`), measured here where the link can be reached (`same-device`, `shared-memory`, `pcie`), and otherwise none until the topology gives one (`infiniband-200g`) |
| `TopologyBuilder` | The questions of `idrak cluster init` as data: each with a description and a default derived from this machine's devices and adapter, answerable in advance by key, written as commented JSON |
| The `cluster` profiler probe | A run's simulated network time per rank and per link |
| Failure models (`BuiltInFailures`, `FailureSettings`) | `none`, `slow-worker`, `dropped-worker`, `link-loss`, `partition`, `late-join` and `mixed`, drawn step by step from (seed, step) alone; each fault and rank-state change is a telemetry record with its seed and step |
| `ClusterRank.BeginStep` | A step of a simulated run: its faults drawn and applied (a dropped worker ends the step on every rank naming it, a partition that never heals times out naming both sides), a late joiner sitting out while the others run as a group |
| `EventLog` | A compact binary log of the faults, each rank's clock at every step, its state changes and its transfers, the same bytes on every run of a seed |
| `ClusterReplay` | Reruns a run through a step with its seed: the same fault schedule and the same clock, bit for bit |
| `LocalhostCluster` | One process per rank: `Run` starts every rank of a topology on this machine (a localhost simulation over the real transports) or one node's ranks (`idrak cluster start`), each told its rank, world size, node, device, coordinator and run in its environment; streams their lines, ends them all when one fails, and kills any left. `Join` makes a process one rank: its device, its transport routed per pair, its collectives |
| The throttle (`IDRAK_CLUSTER_THROTTLE`) | In a localhost simulation, each pair whose link declares a rate is paced by a token bucket to that bandwidth and held for its latency, so loopback imitates the cluster's links |
| Rank 0's measured choices (`ClusterOptions.ShareTuning`, `IDRAK_CLUSTER_SHARE_TUNING`) | Every rank runs rank 0's measured kernel choices (variants, splits, the workgroup width, the attention path) through the backends' `IMeasuredChoicesProvider`s: in one process each rank's backend follows rank 0's live from before the ranks start, between processes rank 0 sends its choices as the ranks join; a rank on another device or driver keeps its own and says so (`MeasuredChoicesShared`) |
| Each rank's measured memory (`ClusterMemory`, `ClusterContext.RankMemory`) | As the ranks join, each measures the free memory of its device (what it reports new tensors could take, within a carved device's cap) and reports it to every rank: in one process before the ranks' threads start, between processes right after they connect. A pipeline's stages are then cut so each fits its rank (`IDRAK_CLUSTER_PIPELINE_BALANCE=auto`: by the measured memory when it differs by more than `IDRAK_CLUSTER_PIPELINE_MEMORY_TOLERANCE`, 0.1 by default, else equal bytes); `IDRAK_CLUSTER_STAGE_LAYERS` (`--stage-layers 3,5,2`) sets each stage's layers by hand and overrides the measurement. With `IDRAK_CLUSTER_OWN_TENSORS` (on) each rank reads only its stage's tensors from the checkpoint, so it starts at once and holds only its part. Each rank's measured memory, its stage's layers and why ("sized by measured memory", "set by the user") are a `PipelineStagePlaced` telemetry record |
| `TopologyInspection.PredictAllReduce` | Plays `ring`, `naive` or `hierarchical` on the virtual clock, message for message, without moving data |

Nothing names a card, a link speed or a memory size: every such figure is declared in the topology file or measured at
run time.
