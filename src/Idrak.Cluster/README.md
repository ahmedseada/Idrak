# Idrak.Cluster

A cluster of GPUs and machines simulated on one machine (plan 23), so distributed training is proven locally before any
real link is used, and a run predicts the network time of a cluster you do not own.

```bash
dotnet add package Idrak.Cluster   # the simulator, on Idrak.Abstraction
dotnet add package Idrak.Network   # the collective algorithms (and, later, the real transports)
```

## What is in the package

| Part | What it does |
|------|--------------|
| `SimulatedDevices` (`sim:N`) | A device carved from a physical one (`Device.CreateSeparateBackend`): its own memory pool and a hard cap, the physical device's type and kernels. Past the cap an allocation fails, or spills to pinned host memory where the device can (CUDA) |
| The `sim` transport | Ranks that are threads of one process, joined by bounded queues; every transfer charged on a `VirtualClock` from the topology's link profiles |
| `VirtualClock` | Network time per rank: bytes / bandwidth + latency + jitter, one transfer at a time per direction, loss sent again; deterministic for a seed (jitter and loss hashed from the message, never from thread timing) |
| `InProcessCluster` | Runs work on every rank of a topology, each on its own thread and carved device, with the collectives `ClusterOptions.Collective` names |
| Built-in link profiles | Rates only where the name states them (`ethernet-1g`), measured here where the link can be reached (`same-device`, `shared-memory`, `pcie`), and otherwise none until the topology gives one (`infiniband-200g`) |
| `TopologyBuilder` | The questions of `idrak cluster init` as data: each with a description and a default derived from this machine's devices and adapter, answerable in advance by key, written as commented JSON |
| The `cluster` profiler probe | A run's simulated network time per rank and per link |

Nothing names a card, a link speed or a memory size: every such figure is declared in the topology file or measured at
run time.
