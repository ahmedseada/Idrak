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
// Part 1: one job on several machines (phase 1) or several GPUs (phase 2)
using var group = await ProcessGroup.StartAsync(new NetworkOptions
{
    Coordinator = "192.168.1.10:29500", Rank = 0, WorldSize = 2, Key = sharedKey,   // machines on a LAN, or
    // Devices = [Device.Cuda(0), Device.Cuda(1)],                                  // GPUs in this machine
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

**The first goal: training over a local network, for every kind of training.** Several machines on one LAN (each
with its own GPU, or CPU) train one model together: each takes its share of every batch, and their gradients are
averaged over TCP, sent as bfloat16 to halve the traffic. It works for anything Idrak trains (a network built with
`Network` or `Sequential`, a GPT trained from scratch, a fine-tuned language model), not only LoRA. Everything else in
this plan builds on it.

### Phase 1. Several machines over TCP (local-network training, first)

**Where it plugs in: the gradient exchange sits between the backward pass and the optimizer step**, which every
kind of training in Idrak has. So one mechanism covers all of them:

| Kind of training | How it joins the group |
|---|---|
| `Trainer` / `TrainingRun` (any model: tabular, images, sequences, classifiers) | `trainer.DataParallel(group)` / `run.DataParallel(group)`; callbacks, metrics, early stopping and checkpoints keep working |
| A GPT or other model trained from scratch (the GptTraining sample's loop) | the same, or `group.AllReduceGradients(model)` before `optimizer.Step()` in a hand-written loop |
| Hand-written loops | `group.AllReduceGradients(model)` (or a `DistributedOptimizer` wrapping any optimizer) |
| Language-model fine-tuning: full weights, LoRA, QLoRA (`FineTuner`, `idrak-tune train`) | `FineTuningOptions.Group`; only the trained values travel (adapters for LoRA, every weight otherwise) |

**How a run looks**

```powershell
# on every machine: the same command, its own rank
idrak-tune train Qwen/Qwen2.5-0.5B chats.jsonl --out adapters/chat --cuda `
    --coordinator 192.168.1.10:29500 --node 0 --nodes 2      # --node 1 on the second PC; key from IDRAK_NETWORK_KEY
```

```csharp
using var group = await ProcessGroup.StartAsync(new NetworkOptions
{
    Coordinator = "192.168.1.10:29500", Rank = rank, WorldSize = 2, Key = sharedKey,
});

// any model with the Trainer or TrainingRun
var history = new TrainingRun { Model = model, Loss = Losses.CrossEntropy, ... }.DataParallel(group).Fit();

// or a hand-written loop (a GPT trained from scratch)
loss.Backward();
group.AllReduceGradients(model);        // averaged across machines, bfloat16 on the wire
optimizer.Step();
```

(`--node` / `--nodes`, not `--rank`: `idrak-tune` already uses `--rank` for the LoRA rank.)

**What gets built**

| Piece | Design |
|---|---|
| Rendezvous | rank 0 listens on the coordinator address; the others connect, check the shared key (an HMAC handshake, nothing accepted without it), learn their neighbours, and open the ring connections |
| Transport | `System.Net.Sockets`: one TCP connection to each ring neighbour, Nagle off, large socket buffers, sends and receives in chunks so both directions stay busy |
| All-reduce | ring reduce-scatter then all-gather (each machine sends and receives 2·(N−1)/N of the gradients, whatever N is) |
| bfloat16 on the wire | gradients converted to bfloat16 on the GPU before they are read back, sent as bfloat16, added up in float32 on arrival; the averaged result goes back to the GPU as float32 |
| Overlap | gradients grouped in buckets (about 25 MB); a bucket's all-reduce starts as soon as the backward pass has written it, while later layers are still computing; uploads and downloads go through the staging ring |
| Data | every rank reads a different slice of each batch (the same seed everywhere), so the global batch is the per-machine batch × N; `DataLoader`, the datasets library and the fine-tuning batches each get a per-rank slice |
| Same model on every machine | rank 0 broadcasts the starting weights (or every rank loads the same checkpoint and the group checks a hash), so random initialization cannot differ |
| Optimizers | any optimizer (SGD, Adam, AdamW, 8-bit AdamW, grouped): each machine runs the same step on the same averaged gradients, so the weights stay identical without being sent again |
| Gradient clipping | the norm is taken after averaging, so every machine clips by the same amount |
| Layers with running state | BatchNorm running statistics averaged across machines at the end of each epoch (or synchronized per batch as an option); dropout masks differ per machine on purpose |
| Metrics and early stopping | losses and metrics summed across machines before they are reported, so every machine sees the same validation result and stops at the same epoch |
| Checkpoints and logs | rank 0 writes checkpoints and the loss log; every rank can resume from them |
| Failures | a send or receive that waits longer than a timeout ends the step on every rank with a clear message naming the rank; the run resumes from the last checkpoint |
| Mixed machines | allowed (different GPUs, or a CPU rank); every step waits for the slowest machine, and the log shows each rank's step time so the slow one is visible |

**How much traffic (estimates)**

Each step moves about the gradient size per machine (2·(N−1)/N of it). In bfloat16 that is 2 bytes per trained value:

| What is trained | Trained values (about) | bfloat16 per step | 1 Gb/s Ethernet (~0.11 GB/s) | 10 Gb/s (~1.1 GB/s) |
|---|---|---|---|---|
| A classifier or regressor (`Network`, a few million weights) | ~2 M | ~4 MB | ~0.04 s | ~0.004 s |
| LoRA rank 16, a 0.5B model | ~9 M | ~18 MB | ~0.16 s | ~0.02 s |
| A GPT trained from scratch, 100 M weights | ~100 M | ~200 MB | ~1.8 s | ~0.18 s |
| LoRA rank 16, a 7B model | ~40 M | ~80 MB | ~0.7 s | ~0.07 s |
| Every weight of a 0.5B model | ~500 M | ~1 GB | ~9 s | ~0.9 s |

Every kind of training runs over any network; the network decides how much of each step is spent waiting:

- **Small and medium models, and LoRA:** gradient traffic is a fraction of a typical step on gigabit Ethernet, and
  overlap hides most of it.
- **Large models trained in full** (a 100 M-weight GPT from scratch, full fine-tuning): exchange gradients every few
  steps instead of every step (gradient accumulation, added to the `Trainer` where it is missing; `FineTuningOptions`
  already has it), or use 10 Gb/s or faster. The plan reports, per run, how much of each step was spent waiting on the
  network, so the choice is visible.
- Float32 on the wire stays available as an option for comparison.

**Done when**, for each kind of training (a `TrainingRun` model such as the HousePrices or Spirals sample, the
GptTraining sample trained from scratch, and a LoRA and a full fine-tune with `idrak-tune`):
- two PCs on a LAN get the same losses as one PC with twice the batch, within bfloat16 rounding of the gradients
  (both runs logged and compared with the existing `compare` tool), and the weights stay identical on both PCs;
- the step time with two machines is measured on 1 Gb/s (and 2.5 or 10 Gb/s if available) and recorded in the README;
- the tests run the same thing as two and three local processes over loopback, so CI covers it without a LAN.

### Phase 2. Several GPUs in one machine
- The same collectives with the CUDA driver's peer-to-peer copies instead of TCP (or through host memory where the
  GPUs cannot reach each other directly, as on many GeForce cards).
- Combined with phase 1: several GPUs in each of several machines (reduce inside the machine first, then across).
- **Done when** two GPUs in one PC give the same losses as one GPU with twice the batch, and the step time drops.

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
| Two or three PCs on gigabit Ethernet (1 Gb/s; 2.5 or 10 Gb/s if available) | phase 1, the first goal, and part 2; shows when the network is the limit |
| One PC with two NVIDIA GPUs (PCIe) | phases 2 and 3 |
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

Part 1: phase 1 (local-network training) 4–6 weeks, phase 2 (GPUs in one machine) 2–3 weeks, phase 3 4–6 weeks, phase 4 3–4 weeks (only if measured worthwhile).
Part 2: phases 5–6 3–5 weeks; phase 7 follows phase 3.

## Priority

Independent of the new GPU backends (plans 3–5) and useful to NVIDIA users now; it can start once plans 1 and 2 are
done, alongside the shared backend work.
