# Idrak.Network

The collective algorithms of an Idrak cluster (plan 23), over any transport registered in `Transports` (the
simulator's `sim` included), and the transport selector. The real transports (shared memory, host-staged and peer
copies, a direct cable, TCP) join in phase C5.

| Part | What it does |
|------|--------------|
| `naive` | Every rank to rank 0, combined in rank order and sent back: the reference the others agree with |
| `ring` | Reduce-scatter then all-gather around the ranks: each sends and receives 2(N-1)/N of the values |
| `fastest` | For a pair of ranks, the first transport available between them, fastest first |
| The `network` profiler probe | Collectives and transfers: wall time, bytes, time blocked on peers, the pairs that cost most |

Every sum is made in an order fixed by the algorithm and the sizes, so every rank ends with identical bytes and a rerun
gives them again. A bfloat16 wire halves the traffic; sums are made in float32. Values are staged through one pooled
host buffer per call, so device memory does not grow with use.
