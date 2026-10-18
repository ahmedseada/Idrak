# Idrak.Network

The real transports and the collective algorithms of an Idrak cluster (plan 23): every transport registered in
`Transports` speaks the same wire format (`WireFormat`: a 36-byte header, a keyed hello first), and the collectives run
over any of them (the simulator's `sim` included), or over a group routed per pair across several.

| Part | What it does |
|------|--------------|
| `shared-memory` | The ranks of one machine: a memory-mapped ring per direction (a file, so it works alike everywhere), a message larger than the ring streaming through it; a hello with the HMAC of a nonce under the cluster key first |
| `tcp` | Any machines: rank 0 listens at the coordinator, every rank proves the key in a challenge each way, then the ranks connect pairwise; Nagle off; a reader per socket; without a key nothing listens beyond loopback |
| `direct-link` | A cable between two machines (a topology override of `direct-link`, `thunderbolt` or `usb4`): the same sockets as tcp, dialled at the peer's address in the file |
| `pcie-host`, `nvlink` (`p2p`) | GPUs of one process: device memory handed over (`IDeviceTransport`) and copied through host memory, or device to device where the driver grants peer access (`Backend.CopyTo`); refused peers fall back, said once per pair |
| `LinkBench` | Bandwidth and latency of every pair, the first round apart from the warm ones (`idrak cluster bench`) |
| `naive` | Every rank to rank 0, combined in rank order and sent back: the reference the others agree with |
| `ring` | Reduce-scatter then all-gather around the ranks: each sends and receives 2(N-1)/N of the values |
| `hierarchical` | Reduce-scatter inside each node over its link, a ring across the nodes on the shards (one rank per node), all-gather inside: the slow links carry only the shards; the default across several nodes |
| `fastest` | For a pair of ranks, the first transport available between them, fastest first; `TransportPlan` asks every transport per pair and publishes each choice once |
| `all-reduce` (`GradientSyncs`) | Data-parallel training: each step's gradients summed across the ranks after the backward pass, token-weighted, a bucket at a time; the starting values sent or checked, weights re-sent every `SyncWeightsEvery` steps, a step's rows split by the ranks' measured speeds; reduced on the host where a `HostOptimizer` holds them |
| The `network` profiler probe | Collectives and transfers: wall time, bytes, time blocked on peers, the pairs that cost most |

Every sum is made in an order fixed by the algorithm and the sizes, so every rank ends with identical bytes and a rerun
gives them again. A bfloat16 wire halves the traffic; sums are made in float32. Values are staged through one pooled
host buffer per call, so device memory does not grow with use.
