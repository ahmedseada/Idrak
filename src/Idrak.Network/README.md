# Idrak.Network

The collective algorithms of an Idrak cluster (plan 23), over any transport registered in `Transports` (the
simulator's `sim` included), and the transport selector. The real transports (shared memory, host-staged and peer
copies, a direct cable, TCP) join in phase C5.

| Part | What it does |
|------|--------------|
| `naive` | Every rank to rank 0, combined in rank order and sent back: the reference the others agree with |
| `ring` | Reduce-scatter then all-gather around the ranks: each sends and receives 2(N-1)/N of the values |
| `hierarchical` | Reduce-scatter inside each node over its link, a ring across the nodes on the shards (one rank per node), all-gather inside: the slow links carry only the shards; the default across several nodes |
| `fastest` | For a pair of ranks, the first transport available between them, fastest first; `TransportPlan` asks every transport per pair and publishes each choice once |
| `all-reduce` (`GradientSyncs`) | Data-parallel training: each step's gradients summed across the ranks, token-weighted, a bucket at a time, each bucket sent while the backward pass goes on (`IDRAK_CLUSTER_OVERLAP`; the bucket size measured on the first step unless `IDRAK_CLUSTER_BUCKET_BYTES` sets it); the starting values sent or checked, weights re-sent every `SyncWeightsEvery` steps, a step's rows split by the ranks' measured speeds; reduced on the host where a `HostOptimizer` holds them; a rank that joins late takes the weights and optimizer state |
| `local-sgd` (`GradientSyncs`) | Each rank steps on its own gradients; the weights are averaged every `IDRAK_CLUSTER_LOCAL_SGD` steps, weighted by each rank's rows |
| `GradientCompressions`: `none`, `bf16`, `top-k` | How a gradient bucket travels (`IDRAK_CLUSTER_COMPRESSION`): every value as float32 or bfloat16, or each rank's largest share as index and value pairs with the rest kept for the next step (error feedback); `IGradientCompression` and `IGradientCompressor` in `Idrak.Networking.Abstractions` take a plug-in's |
| The `network` profiler probe | Collectives and transfers: wall time, bytes, time blocked on peers, the pairs that cost most |

Every sum is made in an order fixed by the algorithm and the sizes, so every rank ends with identical bytes and a rerun
gives them again. A bfloat16 wire halves the traffic; sums are made in float32. Values are staged through one pooled
host buffer per call, so device memory does not grow with use.
