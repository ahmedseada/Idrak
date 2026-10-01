# Idrak.Network plan (project and plan)

**Status:** not started. Idrak runs every model on one device: several GPUs in one machine can each run their own
model (`Device.Cuda(n)`), but one training job or one model cannot use more than one GPU, and the serving packages
(`Idrak.AspNetCore`, the inference engine) run on one machine. Goal: a new package, `Idrak.Network`, that lets
**several GPUs or machines work on one job** (distributed compute, part 1) and **several machines serve one API**
(cluster serving, part 2).

## The project

```
src/Idrak.Network/
  Idrak.Network.csproj        depends on Idrak (and Microsoft.AspNetCore.App for part 2), nothing native
  Transport/                  how bytes move: same-GPU-machine peer copies, TCP between machines
  Collectives/                all-reduce, all-gather, reduce-scatter, broadcast, send/receive, barrier
  Training/                   data parallel, sharded optimizer state, pipeline and tensor parallel
  Cluster/                    nodes, membership, health, model placement, request routing (part 2)
tests/Idrak.Tests/NetworkTests.cs   multi-process tests on one machine (CPU and GPU)
samples/Idrak.Samples.Distributed/  data-parallel fine-tuning and a two-node chat cluster
```

### The rule it keeps

No native libraries: NCCL, MPI and RDMA verbs are prebuilt native libraries, so the core package does not use them.

- **GPUs in one machine:** the CUDA driver API already has peer-to-peer copies (`cuDeviceCanAccessPeer`,
  `cuCtxEnablePeerAccess`, `cuMemcpyPeerAsync`) over PCIe or NVLink. Collectives are our own kernels and copies.
- **Machines:** .NET's own sockets (TCP; optionally QUIC), with collectives written in C#.
- **Faster transports** (NCCL, InfiniBand/RDMA) could come later as optional add-on packages behind the same
  `ITransport` interface, like the NPU add-ons in the hardware plans.

### Public API (sketch)

```csharp
// Part 1: one job on several devices
using var group = await ProcessGroup.StartAsync(new NetworkOptions
{
    Devices = [Device.Cuda(0), Device.Cuda(1)],          // GPUs in this machine, or
    // Peers = ["10.0.0.5:29500", "10.0.0.6:29500"], Rank = 0,   // machines (one process per GPU)
});

var run = new TrainingRun { Model = model, ... }.DataParallel(group);   // gradients averaged across devices
run.Fit();                                                               // same history on every rank

using var big = PretrainedModel.Load("Qwen/Qwen2.5-7B", new PretrainedOptions
{
    Placement = Placement.Pipeline(group),                // layers split across GPUs (model too big for one)
});

// Part 2: several machines, one API
builder.Services.AddIdrak()
    .AddCluster(c => c.Node("gpu-a", "http://10.0.0.5:8080").Node("gpu-b", "http://10.0.0.6:8080"))
    .AddChatModel("chat", "Qwen/Qwen3-0.6B", m => m.Replicas(2));         // placed on two nodes
app.MapOllamaApi("/api", "chat");                         // the gateway routes each request to a node
```

### Building blocks every part uses

| Piece | What it does |
|---|---|
| `ITransport` | send/receive tensors between ranks; peer copies on one machine, TCP between machines |
| `ProcessGroup` | ranks, world size, device per rank; start, join, shut down; timeouts |
| Collectives | ring all-reduce (bandwidth-optimal), all-gather, reduce-scatter, broadcast; overlap with compute |
| `Placement` | where each layer or replica lives (one device, data parallel, pipeline stages, tensor shards) |
| Rendezvous | how ranks find each other: a fixed list, an environment variable, or the first node as coordinator |
| Security | a shared secret or mutual TLS between nodes; nothing listens without it |

## Part 1: distributed compute

### Phase 1. Several GPUs in one machine, data parallel
- Peer-to-peer copies between GPUs; a ring all-reduce of gradients (bucketed, started while the backward pass is
  still running, as PyTorch DDP does).
- `TrainingRun.DataParallel(group)` and the fine-tuner (`FineTuner`): each GPU gets its share of every batch;
  gradients are averaged before the optimizer step; one rank writes checkpoints.
- **Done when** two GPUs give the same losses and weights as one GPU with twice the batch (within float rounding),
  and the step time drops measurably.

### Phase 2. Several machines, data parallel
- TCP transport and rendezvous; the same collectives over sockets; gradients sent as bfloat16 to halve the traffic.
- **Done when** a LoRA fine-tune runs on two machines with the same losses as one machine, and the tests run the
  same thing as two local processes (so CI covers it without two machines).

### Phase 3. Models too big for one GPU
- **Pipeline parallel:** the decoder's layers split across GPUs; micro-batches keep every stage busy during
  training; for inference, each token passes through the stages in turn.
- **Sharded optimizer state** (ZeRO-1 style): each rank keeps a slice of the Adam state, cutting its memory by the
  number of ranks.
- **Done when** a 7B model too large for one 8 GB GPU loads and chats across two GPUs, and fine-tunes with LoRA.

### Phase 4. Tensor parallel (only where it pays)
- Attention heads and feed-forward columns split across GPUs, with an all-reduce per layer. Worth it only on fast
  links (NVLink, or GPUs in one machine); over PCIe or a network, pipeline parallel is usually faster.
- **Done when** it beats pipeline parallel on a measured setup; otherwise it stays out.

## Part 2: cluster serving

### Phase 5. Nodes and placement
- A node is any machine running Idrak's ASP.NET Core host. A gateway knows the nodes, checks their health
  (`MapIdrakStatus` already reports loaded models, queues and latency), and places models (replicas per model,
  GPU memory per node).
- **Done when** a gateway in front of two nodes serves a model on both and keeps serving when one node stops.

### Phase 6. Routing
- Requests go to the least-loaded replica; streaming responses pass straight through.
- **Prompt-cache affinity:** a chat's next turn goes to the node that already holds its KV cache (the kept cache
  reuses shared prompt prefixes), so a second turn does not recompute the whole conversation.
- **Done when** throughput with two nodes is close to twice one node's, and chat turns hit their cached prefix.

### Phase 7. One model across machines (joins part 1)
- Pipeline-parallel models (phase 3) served behind the gateway, for models no single machine can hold.

## How it is tested

- **Every collective against a reference:** all-reduce results equal the plain sum, on the CPU and on GPUs, for odd
  sizes and many ranks (2, 3, 4, 8).
- **Multi-process on one machine:** the tests start ranks as separate processes over loopback TCP, so CI covers
  the network code without extra hardware.
- **Multi-GPU and multi-machine runs by hand,** recorded in the README's "Tested on" table like the GPUs.
- **Benchmarks:** `--bench-network` reports all-reduce bandwidth (GB/s) per transport and the scaling of a training
  step with 1, 2 and 4 GPUs.

## Hardware to test on

| Setup | Why |
|---|---|
| One PC with two NVIDIA GPUs (PCIe) | phase 1 and 3; the most common multi-GPU setup |
| Two PCs on gigabit or 10-gigabit Ethernet | phase 2 and part 2; shows when the network is the limit |
| A cloud machine with NVLink GPUs | phase 4 (tensor parallel) |

## Risks

- **Bandwidth:** gigabit Ethernet moves about 0.1 GB/s, so data-parallel training of anything but adapters (LoRA)
  is network-bound across machines; the plan targets LoRA and bfloat16 gradients first.
- **Consumer GPUs and peer access:** many GeForce cards do not allow direct peer-to-peer copies; the fallback goes
  through host memory and is slower.
- **Failure handling:** a rank that dies stops a training job; checkpoints and a clear error come first, elastic
  restarts later.
- **Security:** anything that listens on the network needs authentication from the start.

## Effort (rough)

Part 1: phase 1 3–4 weeks, phase 2 2–3 weeks, phase 3 4–6 weeks, phase 4 3–4 weeks (only if measured worthwhile).
Part 2: phases 5–6 3–5 weeks; phase 7 follows phase 3.

## Priority

Independent of the new GPU backends (plans 3–5) and useful to NVIDIA users now; it can start once plans 1 and 2 are
done, alongside the shared backend work.
