# Idrak.Network

The real transports and the collective algorithms of an Idrak cluster (plan 23): every transport registered in
`Transports` speaks the same wire format (`WireFormat`: a 36-byte header, a keyed hello first), and the collectives run
over any of them (the simulator's `sim` included), or over a group routed per pair across several.

| Part | What it does |
|------|--------------|
| `shared-memory` | The ranks of one machine: a memory-mapped ring per direction (a file, so it works alike everywhere), a message larger than the ring streaming through it; an idle side blocks on a loopback doorbell the other rings (`IDRAK_CLUSTER_SHM_SPIN`, `IDRAK_CLUSTER_SHM_CHECK`); a hello with the HMAC of a nonce under the cluster key first |
| `tcp` | Any machines: rank 0 listens at the coordinator, every rank proves the key in a challenge each way, then the ranks connect pairwise; Nagle off; a reader per socket; without a key nothing listens beyond loopback |
| `direct-link` | A cable between two machines (a topology override of `direct-link`, `thunderbolt` or `usb4`): the same sockets as tcp, dialled at the peer's address in the file |
| `pcie-host`, `nvlink` (`p2p`) | GPUs of one process: device memory handed over (`IDeviceTransport`) and copied through host memory, or device to device where the driver grants peer access (`Backend.CopyTo`); refused peers fall back, said once per pair; ring collectives over them keep each piece on the device and sum it there (`IDRAK_CLUSTER_DEVICE_COLLECTIVES`) |
| `LinkBench` | Bandwidth and latency of every pair, the first round apart from the warm ones (`idrak cluster bench`) |
| `naive` | Every rank to rank 0, combined in rank order and sent back: the reference the others agree with |
| `ring` | Reduce-scatter then all-gather around the ranks: each sends and receives 2(N-1)/N of the values |
| `hierarchical` | Reduce-scatter inside each node over its link, a ring across the nodes on the shards (one rank per node), all-gather inside: the slow links carry only the shards; the default across several nodes |
| `fastest` | For a pair of ranks, the first transport available between them, fastest first; `TransportPlan` asks every transport per pair and publishes each choice once |
| `all-reduce` (`GradientSyncs`) | Data-parallel training: each step's gradients summed across the ranks, token-weighted, a bucket at a time, each bucket sent while the backward pass goes on (`IDRAK_CLUSTER_OVERLAP`; the bucket size measured on the first step unless `IDRAK_CLUSTER_BUCKET_BYTES` sets it); the starting values sent or checked, weights re-sent every `SyncWeightsEvery` steps, a step's rows split by the ranks' measured speeds; reduced on the host where a `HostOptimizer` holds them; a rank that joins late takes the weights and optimizer state |
| `local-sgd` (`GradientSyncs`) | Each rank steps on its own gradients; the weights are averaged every `IDRAK_CLUSTER_LOCAL_SGD` steps, weighted by each rank's rows |
| `sharded` (`GradientSyncs`) | ZeRO-style data parallel (`IDRAK_CLUSTER_SHARD`): each rank owns a share of the trained parameters; their gradients reduce-scattered to it bucket by bucket, their optimizer state kept there, the weights gathered after the update (`optimizer`, `gradients`) or kept on their owner and gathered layer by layer before each forward and backward (`parameters`, `IDRAK_CLUSTER_GATHER_BYTES`); a late joiner takes its share, checkpoints join every rank's |
| `gpipe` (`PipelineSchedules`) | Pipeline placement: each stage runs every micro-batch's forward pass, then every backward pass in reverse (`IDRAK_CLUSTER_SCHEDULE`; Idrak's `PipelineStage` runs it) |
| `GradientCompressions`: `none`, `bf16`, `top-k` | How a gradient bucket travels (`IDRAK_CLUSTER_COMPRESSION`): every value as float32 or bfloat16, or each rank's largest share as index and value pairs with the rest kept for the next step (error feedback); `IGradientCompression` and `IGradientCompressor` in `Idrak.Networking.Abstractions` take a plug-in's |
| The `network` profiler probe | Collectives and transfers: wall time, bytes, time blocked on peers, the pairs that cost most |

Every sum is made in an order fixed by the algorithm and the sizes, so every rank ends with identical bytes and a rerun
gives them again. A bfloat16 wire halves the traffic; sums are made in float32. Values are staged through one pooled
host buffer per call, so device memory does not grow with use.
