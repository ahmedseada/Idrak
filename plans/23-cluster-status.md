# Plan 23 status: Idrak.Cluster and Idrak.Network

The whole plan as it stands on 2026-10-18, phase by phase: what each phase delivered, how it was tested, what was
measured, and what is still open. The full design, the contracts and every phase's "as built" note are in
[`23-cluster.md`](23-cluster.md); this file is the summary.

Branch `abstraction`, head `7c41b8d` when this was written.

## At a glance

| Phase | What | Commit | CPU | CUDA | Vulkan |
|---|---|---|---|---|---|
| C0 | Contracts, registries, telemetry, conformance kit, PyTorch DDP fixture | f5b9e62 | ✅ | ✅ | ✅ |
| C1 | In-process simulator, carved fake devices, virtual clock, `naive`/`ring`, `cluster init`/`inspect` | 75a472a, 3e925fc | ✅ | ✅ | ✅ |
| C2 | Data-parallel training on the simulator, `cluster sim` | 12f1ba8, 0d94327 | ✅ | ✅ | ✅ |
| C3 | Hierarchical all-reduce, transport selection, faults, event log, replay | 92401af | ✅ | ✅ | ✅ |
| C4 | Overlap, measured buckets, compression, local SGD, faults inside training | dcb68ec | ✅ | ✅ | ✅ |
| C5 | Real transports, localhost mode, `cluster start`, `cluster bench` | dcb68ec, 11e3cb8 | ✅ | ✅ | ✅ |
| C6 | ZeRO sharding, flow control, GPU staging, GPU-side all-reduce, shared-memory wake-up | cb514b4, 0a2a96f | ✅ | ⚠️ 1 item | ⚠️ 1 item |
| C7 | Pipeline parallelism | 00a0d8f | ✅ | ⚠️ 1 item | ✅ |
| C8a | `replay`, `calibrate`, `transports`, `--profile`, `--cluster` on the training commands | 7c41b8d | ✅ | pending | pending |
| C8b | Memory and speed: own-tensor loads and saves, GPU-to-GPU activations, direct wake | in progress | | | |
| C9 | Web dashboard: set up, control, monitor | 8e4b84c, f262009 | ✅ | ✅ | pending |

⚠️ = the two GPU-only memory items under "Open", both assigned to C8b. "Pending" = not yet run on the owner's
machine at that commit.

## Packages

- **`Idrak.Abstraction.Cluster`** (namespace in Idrak.Abstraction): the contracts and registries every package uses:
  transports, collectives, link profiles, topology, selectors, failure models, gradient syncs, pipeline schedules,
  run control, the wire format and `ClusterOptions` with its environment variables.
- **`Idrak.Cluster`**: the simulator. Carved fake devices with hard memory caps, the `sim` transport on a
  deterministic virtual clock, built-in link profiles, failure models, the event log and replay, the in-process and
  localhost clusters, the topology builder and inspection, calibration, the run controller, the `"cluster"` probe.
- **`Idrak.Network`** (assembly `Idrak.Network`, namespace `Idrak.Networking`, because a namespace `Idrak.Network`
  would hide the `Idrak.Network` builder class): the real transports, the collectives (`naive`, `ring`,
  `hierarchical`), gradient syncs (`all-reduce`, `local-sgd`, `sharded`), compression, buckets, the GPipe schedule,
  the link bench, the `"network"` probe.
- Both packages reference `Idrak.Abstraction` only (tested). Idrak core and Idrak.Nlp reach them through contracts.

## Decisions

1. `Idrak.Cluster` references `Idrak.Abstraction` and implements its contracts: no bridge, no duplicate types.
2. The topology file is JSON (`idrak-topology/1`), comments and trailing commas allowed.
3. `--rank N`, `--world-size N`, `--coordinator HOST:PORT`, matching `IDRAK_RANK`, `IDRAK_WORLD_SIZE`,
   `IDRAK_COORDINATOR`. Each command owns its options: on `tune` and `distill`, `--rank` stays the LoRA rank and the
   cluster rank comes from `IDRAK_RANK`. "Node" is a machine; a rank is one worker on one device.
4. The wire defaults to float32; the library is to measure and suggest bf16, local SGD and the rest
   (**not built yet**, see Open).
5. `udp` stays out of C5 (registered name, no library default).
6. No presets: `idrak cluster init` asks every question with a description and a suggestion from this machine.
7. Memory sizes `B`, `KB`/`KiB`, `MB`/`MiB`, `GB`/`GiB`, `TB`/`TiB`, all powers of 1,024; a size without a unit is
   refused.
8. A web dashboard (C9): simulated runs set up and controlled from the page, every run monitored, real runs
   read-only.

## Phases

### C0: the reference, the contracts, the registries ✅

- Contracts and registries in `src/Idrak.Abstraction/Cluster/`; the 36-byte wire header and the HMAC-SHA256 hello.
- Telemetry level `Network` with its events (transfer, collective, transport chosen, rank state, fault, calibrated).
- Conformance kit: `TransportSuite`, `CollectivesSuite`, `LinkProfileSuite`, `TopologySuite`.
- PyTorch DDP fixture (`tools/cluster/make_tiny_ddp.py`, `tests/Idrak.Tests/data/cluster/`): a measurement to agree
  with, never a limit on what Idrak builds.

### C1: the in-process simulator ✅

- Fake devices carved from a real one (`SimulatedDevices.Carve`), each its own backend with a hard memory cap, on
  CPU, CUDA (spills to host when offload is on), Vulkan and HIP (HIP compiled, not run).
- `sim` transport on a virtual clock: bytes / bandwidth + latency, per-direction contention, seeded jitter and loss.
- Built-in links: rates only from names (`ethernet-1g`, `infiniband-200g`, …); `same-device`, `shared-memory`,
  `pcie` measured on the machine. Nothing tuned for a card.
- `naive` and `ring` collectives (fixed summation order: identical on every rank and rerun).
- `idrak cluster init` (interactive, coloured, suggestions from the machine, card twins such as `vulkan:0` beside
  `cuda:0`, sizes need a unit) and `idrak cluster inspect` (ranks, links, predicted all-reduce, `--unit`).

### C2: data parallel on the simulator ✅

- `Cluster` hook on `Trainer`, `FineTuningOptions`, `PretrainingOptions`; gradient sync as a contract plus registry.
- Token-weighted gradients (equal to one process), data sharded by global draw, rank 0 checkpoints, resume at another
  world size, batches split by measured device speed, `HostOptimizer` reducing its own pieces.
- `idrak cluster sim` (pretrain at first; tune, train and resume since C8a).
- Measured against the DDP fixture: SGD weights 1.2e-7; AdamW (folded) 2.2e-5 CPU, 2.5e-5 Vulkan, 2.8e-5 CUDA,
  bound 2.9e-5 derived from the fixture's own measured growth.

### C3: topology-aware all-reduce, selection, faults, replay ✅

- `hierarchical` collective (the default across nodes): 6.95× faster than ring on two machines of four devices
  (nvlink inside, 1 Gb Ethernet between): 2.1 ms against 14.7 ms simulated.
- Transport plan chosen per pair, fastest first, published once per pair.
- Failure models: slow worker, dropped worker, link loss, partition, late join, mixed; drawn from (seed, step).
- Event log and replay: `--seed S --step N` reproduces faults and clocks bit for bit.

### C4: overlap, buckets, compression, local SGD ✅

- Gradients reduced in buckets while the backward pass runs; overlap on and off bit-identical.
- Bucket size measured once (with each message's fixed cost); exposed network time 12–62% shorter on 1 Gb Ethernet
  (CUDA 12–24%, Vulkan 48–62%).
- Compression registry: `none`, `bf16`, `top-k` with error feedback (10% top-k: loss 0.370 against 0.354, 1.8 MB sent
  against 8.9 MB on the Spirals sample).
- Local SGD every k steps (k = 1 equals one process).
- The trainers take part in each step's group: faults reach training; late joiners take the trained state.

### C5: real transports ✅

- `shared-memory`, `tcp`, `direct-link`, `pcie-host`, `nvlink` (alias `p2p`); one group can mix transports.
- `LocalhostCluster` (one process per rank) with a token-bucket throttle (within 10% of the profile).
- `idrak cluster start` (this machine's ranks, live line per rank, all stopped when one fails) and
  `idrak cluster bench` (latency and bandwidth per pair, cold and warm apart).
- Localhost and in-process runs give bit-identical weights.

### C6: sharding (ZeRO-style) and transport memory ✅ (one GPU item open)

- `Shard = optimizer | gradients | parameters`: results bit-identical to unsharded; a model over one fake GPU's cap
  trains across four, each rank holding its share (CPU).
- Receive flow control: a slow receiver holds at most 16 KiB instead of about 1.9 MiB; no all-send deadlock.
- Gradient buckets staged from the GPU in the background; ring all-reduce kept on the GPU over pcie-host and nvlink.
- Shared memory wakes on a doorbell, not a sleep. On the owner's Windows machine: latency 7,961 µs → 236 µs,
  16 MB messages 100 MiB/s → 7,726 MiB/s.

### C7: pipeline parallelism ✅ (one CUDA item open)

- Stages cut by measured bytes, declared memory or by hand; `PipelineBoundary` sends activations forward and
  gradients back; GPipe with micro-batches; pipeline plus data-parallel replicas.
- Two stages give one device's logits exactly; 1, 2 and 4 micro-batches give one process's losses within 1e-5.
- Chat through stages, each with its own KV cache. 1F1B not built: the measured bubble did not justify it.
- Training through the pretrainer; sharding with a pipeline refused for now, naming both settings.

### C8a: the command line and calibration ✅

- `idrak cluster replay` (fault schedule and clocks; reruns; `--from` a run's event log, steered runs included).
- `idrak cluster calibrate --from RUN|bench.json` (fitted bandwidth and latency per link, kept in the cache and used
  by every later topology; a shared machine adapter fitted as a static share).
- `idrak transports`, `cluster bench --concurrent`, `--profile` on `cluster sim`, `profile --cluster`,
  `tuning show --backend cluster`, network share in `runs show`.
- `--cluster FILE` on `train`, `pretrain`, `tune`, `resume`, `distill` (decision 3's rules).
- Measured chunk and bucket sizes kept in the cache; event log flushed while a run goes
  (`RUN/cluster-events.idrklog`); an `address` on link overrides for a cable on its own subnet.

### C8b: memory and speed 🔄 (in progress)

- A pipeline stage and a parameter shard load only their own tensors and save only their own part; no rank holds the
  whole model; the one-process checkpoint format still loads.
- Activations between stages GPU to GPU where the transport allows.
- A receive inbox that wakes its waiter directly.
- A pipeline stop writes its checkpoint; the Trainer and fine-tuner answer a dashboard stop with a checkpoint, so the
  dashboard can steer `train` and `tune`.
- The two GPU memory items (below), each test naming what is held.
- Sharding with a pipeline and top-k with sharding, if the measured numbers justify them.

### C9: the dashboard ✅

- `idrak cluster ui [--topology FILE]` and `idrak cluster sim … --dashboard`; `cluster ui --follow RUN` for a real
  run, read-only.
- Set up: the builder's questions as a form, five topology views (network, ring, link matrix, all-reduce plan,
  memory), predicted times.
- Control: start, pause and resume at a step boundary, stop with a checkpoint (sharded runs join every share), inject
  a fault live (replayable).
- Monitor: per rank loss, steps a second, memory against cap, network against blocked time; links; event timeline.
- 127.0.0.1 by default, a key off this machine, other sites' pages and other host names refused; one self-contained
  page, no external address, light and dark, Arabic right to left.
- Mock-up: https://claude.ai/artifact/X6kUofVQnZ9jEAiP2fP9Ka

## Commands

| Command | What it does | Phase |
|---|---|---|
| `idrak cluster init` | Builds a topology file, asking every input with a suggestion | C1 |
| `idrak cluster inspect FILE` | Ranks, links, predicted all-reduce, transport per pair | C1, C3, C8a |
| `idrak cluster sim --topology FILE -- pretrain\|tune\|train\|resume …` | Every rank on this machine, simulated (in-process or `--mode localhost`), faults, `--profile` | C2, C4, C5, C8a |
| `idrak cluster start --topology FILE -- COMMAND` | This machine's ranks of a real cluster run | C5 |
| `idrak cluster bench --topology FILE` | Latency and bandwidth per pair | C5, C8a |
| `idrak cluster replay --topology FILE --step N` | Reproduces faults and clocks; reruns; checks a run's log | C8a |
| `idrak cluster calibrate --from RUN\|bench.json` | Fits link profiles into the cache | C8a |
| `idrak cluster ui` | The dashboard | C9 |
| `idrak transports` | Every registered transport and whether it works here | C8a |
| `idrak profile MODEL --cluster FILE` | A token's hops through a simulated pipeline | C8a |
| `idrak tuning show --backend cluster` | Calibrated links, measured chunk and bucket sizes | C8a |
| `--cluster FILE` on `train`, `pretrain`, `tune`, `resume`, `distill` | A hand-started rank of a cluster run | C8a |

## Settings

Every behaviour is a setting (`ClusterOptions`, each with its environment variable; `idrak env` lists them):

- Run: `IDRAK_CLUSTER_TOPOLOGY`, `_NODE`, `_DEVICE`, `_MODE`, `_RUN`, `_TIMEOUT`, `_KEY`, `_PORT`, `_RETRIES`,
  `_RETRY_DELAY`, `IDRAK_RANK`, `IDRAK_WORLD_SIZE`, `IDRAK_COORDINATOR`.
- Transport: `_TRANSPORT`, `_SELECTOR`, `_IN_FLIGHT`, `_RECEIVE_BYTES`, `_SOCKET_BUFFER`, `_SHM_BYTES`, `_SHM_FOLDER`,
  `_SHM_SPIN`, `_SHM_CHECK`, `_THROTTLE`, `_DEVICE_COLLECTIVES`.
- Collectives and gradients: `_COLLECTIVE`, `_WIRE`, `_CHUNK_BYTES`, `_CHUNK_PROBE`, `_BUCKET_BYTES`, `_OVERLAP`,
  `_ASYNC_STAGING`, `_MESSAGE_COST`, `_COMPRESSION`, `_TOPK`, `_ERROR_FEEDBACK`, `_GRADIENT_SYNC`, `_LOCAL_SGD`,
  `_SYNC_WEIGHTS`, `_START`, `_REBALANCE_EVERY`, `_REBALANCE_STEPS`, `_REBALANCE_TOLERANCE`.
- Sharding and pipeline: `_SHARD`, `_GATHER_BYTES`, `_PLACEMENT`, `_STAGES`, `_MICROBATCHES`, `_PIPELINE_BALANCE`,
  `_PIPELINE_CUTS`, `_SCHEDULE`.
- Simulation: `_CLOCK`, `_SEED`, `_FAILURES` (and the failure-model settings), `_CARVE_FROM`, `_EVENT_LOG`,
  `_EVENT_LOG_FLUSH`, `_LOG_TRANSFERS`, `_REPLAY_STEP`, `_PROBE_BYTES`.
- Calibration and cache: `_PROFILES_FILE`, `_TUNING_FILE`.
- Dashboard: `_CONTROL`, `_CONTROL_POLL`, `_UI_PORT`, `_UI_HOST`, `_UI_HISTORY`, `_UI_POLL`.

(All `IDRAK_CLUSTER_…` unless written in full.)

## Open

| Item | Where | State |
|---|---|---|
| Sharding (`parameters`) holds 262,148 B above each rank's share on CUDA and Vulkan (one 256×256 weight and a float), constant, not growing | C8b | Test to name what is held, then fix |
| A CUDA pipeline stage holds 252 B less than its own tensors (passed before C9 merged) | C8b | Same: test to name it, then fix |
| Stage and shard loads of their own tensors; streamed or sharded checkpoints | C8b | In progress |
| Dashboard steering `train` and `tune` (needs their stop-with-checkpoint) | C8b | In progress |
| Sharding with a pipeline; top-k with sharding | C8b | If measurements justify, else stay refused |
| **The `"cluster"` setting advisor (decision 4): measure the network share and bf16's error, suggest `--wire bf16`, local SGD, bucket size, top-k, hierarchical** | **not assigned** | **Not built: no phase implemented it; `IDRAK_CLUSTER_SUGGEST_SHARE` is read but nothing suggests** |
| A shared machine adapter modelled event by event (now a static share, exact when a machine's ranks cross at once) | after C8 | Needs a virtual-clock rewrite |
| AdamW against the DDP fixture on CUDA: 2.81e-5 against a 2.91e-5 bound | watch | Passes with little margin |
| The dashboard's own server messages on the Arabic page | after C9 | English for now |
| `udp` and `rdma` transports | plug-ins | Names registered, no library default |

## The owner's runs still to do

- The final GPU run after C8b: `cluster:` on CUDA and Vulkan, and the training, generation and checkpoint filters on
  CUDA.
- A browser check of the dashboard (`idrak cluster ui --topology C:\idrak-test\card.json`).
- The real LAN: `cluster bench` then `cluster start` on the three home machines (SEADA-LEGION with two ranks on the
  RTX 5070 Ti, mini, old) over 1 Gb Ethernet; the numbers go into the plan's "Runs" table, then `cluster calibrate`.
