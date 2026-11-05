# Plan 37: tensor layout, views, element types and 64-bit sizes (memory first)

**Status (2026-11-05):** W0-W2 (weight placement) and V0, V1, V1b (layouts, CPU views read in place) built, `main`
6eff13c; the rest planned; work stopped by the owner until resumed. Q0, K0, W3 and the wider T0, T2, T3 added
2026-11-05; the network stages N0-N3 added the same day. Written from the owner's request (2026-11-04): "add it as separate plan
that's look like a legit good plan and memory is the main purpose and goal so it's good to have", after the risk
review's items C1, C2 and C3 (plan 36, "The full risk review"). Every "today" below was read in the source at `main`
82b0435.

## Goal

Less memory and fewer copies on every path, with every existing result unchanged (rule 71). In order of payoff:

1. **Views (C2):** transpose, permute, narrow, slice, expand and select share their input's storage instead of copying
   it. A kernel that needs a contiguous input asks for one, and only then is a copy made.
2. **Element types (C1):** a tensor declares its element type. bfloat16 and float16 hold half the bytes of float32;
   int8 and int4 weights are real types, not bits packed into float storage; token ids and indices are exact int32 or
   int64, not floats that go wrong above 2^24.
3. **Experts on demand (R0, R1):** a mixture-of-experts model routes on the device and keeps only its often-used
   experts in device memory, the rest in host memory or the file, brought in when the router picks them, so a model
   larger than the device's memory runs.
4. **Quantized weights kept packed (Q0) and smaller caches (T2, K0):** a low-bit model takes about its file's size in
   memory, not float32's, because its weights stay packed and are expanded inside the kernel, tile by tile; the KV
   cache can be int8 or fp8, and requests that share a prompt share its cache.
5. **Element and block types as registered data (T0, T3):** a type is a registry entry (its bits per value, block
   size, scales, conversions and kernels), so int2, ternary, fp8 and ggml's block formats plug in without changing
   tensors.
6. **Training across devices and machines (N0-N3):** fewer bytes on the network (int8/fp8 gradients with error
   feedback, bf16/fp8 and packed weights when sharded), pipeline stages sized by each machine's measured memory, each
   rank mapping only its shard, and tensor parallelism for a layer too large for one device.
7. **64-bit sizes (C3):** a tensor and a storage may hold more than 2^31 elements; index arithmetic that can overflow
   is 64-bit.

## What it gains, and what it does not

The weight-placement stages (W0-W2, R0, R1) turn "fails or crawls" into "runs at a usable speed" for a model larger than
the device; mixture-of-experts models gain the most. The stages after them (views, element types, 64-bit sizes) lower
the memory of every run.

| Case | Today | After | Why |
|---|---|---|---|
| A dense model larger than the device | Fails, or crawls with `IDRAK_OFFLOAD=1` | Runs near Ollama's speed (W1) | Only the activations cross the bus, not the weights every token |
| A mixture-of-experts model larger than the device | Fails or crawls | Near full speed (R0, R1) | A few experts run per token; the used ones stay on the device, the cold ones run on the CPU |
| Starting a large model | Loads everything first | Starts at once (W2) | The file is mapped and read as layers are first used |
| Another program takes device memory during a run | Can fail | Keeps running, a little slower (W2) | Layers move to the CPU instead of failing |
| Any model, any run | Copies on every transpose and slice; float32 everywhere | Less memory and fewer copies (V, T) | Views share storage; bf16/f16 halve the bytes; int types replace bits packed in floats |
| A 4-bit or lower GGUF model | Types without a packed format expanded to float32 at load: up to 8× the file's size in memory (16× for 2-bit) | About the file's size (Q0) | Weights stay packed; the product expands each block per tile, as llama.cpp does |
| A long context, or many users on `idrak serve` | The KV cache in float32, one per request | 2-4× less cache (T2), prompts' caches shared (K0) | An int8 or fp8 cache; a paged cache whose pages a shared prompt prefix reuses |
| A CPU layer of a split model (W1) | Its weights copied from the file into memory | No second copy (W3) | The CPU reads the file's mapping, shared with the system's page cache |
| Training over a network | Gradients in float32 or bf16 (top-k optional); sharded weights gathered in float32 | 2-8× fewer bytes on the wire (N0, N1) | The network is far slower than device memory, so bytes on the wire decide the step's speed |
| A cluster of unequal machines | Pipeline stages split evenly | Stages sized by each machine's measured memory (N2) | W1's `auto` sizing, across machines |
| A layer too large for one device | Cannot train or run it | Runs split across devices (N3) | Tensor parallelism splits the layer's products |

What it does not change: a model that already fits the device runs as today in W1 and W2 (no gain, no loss); a model far
larger than the device is still bound by the CPU's and the system memory's speed (Ollama has the same limit); lazy start
does not lower a dense model's memory once it runs, since every layer serves every token; training keeps its own
staging (W1 aims at inference). The numbers come from W0 and the owner's runs, never from an estimate.

## Today

| What | Where | Effect |
|---|---|---|
| Shape only, no strides or offset | `Tensor._shape` (`int[]`), `Tensor.Size` (`int`), `src/Idrak.Abstraction/Tensors/Tensor.cs` | Every tensor is one contiguous block from element 0 |
| `Permute`, `Transpose`, `Narrow` copy | `Tensor.Advanced.cs` ~276, ~354, ~369: `Empty(outShape)` then `Backend.Permute` | A new block per call; attention's head transposes and every slice cost memory and a pass over it |
| `Reshape` is the only view | `Tensor.Operations.cs` ~127 | Shares storage when contiguous |
| One element type, float32 | `Storage` (`Backend.cs` ~200): `Length` (`int`) floats | bf16, f16, int8, int4 kept as bits inside floats (`Tensor.Quantized.cs`, `PackedFormat`, `AdamW8Bit`) |
| Indices as floats | gathers, embeddings, MoE routing | Exact only below 2^24 |
| Low-bit GGUF types expanded at load | `GgufFile` (plan 36's C14 types IQ1-IQ3, TQ1_0, TQ2_0, MXFP4, and any type with no packed format) | Dequantized to float32 when the model loads: memory up to 8-16× the file's size |
| KV cache in the compute type, one per request | the decoder's `DecodingContext` | The largest memory of a long context or of many concurrent requests |
| Cluster sharding without tensor parallelism | `Idrak.Cluster`, `ClusterOptions.Shard` (`none`, `optimizer`, `gradients`, `parameters`: ZeRO 1-3's ideas) and `Placement` (`single`, `pipeline`); gradient compressions `none`, `bf16`, `top-k` | No split of a single layer across devices; pipeline stages split evenly; no int8/fp8 gradients; sharded weights gathered in float32 |
| `int` sizes | `Storage.Length`, `Tensor.Size`, GPU index math | At most 2^31 elements per tensor |
| About 150 virtual backend operations | `src/Idrak.Abstraction/Devices/Backend*.cs` | Each implemented on CPU, CUDA, Vulkan, HIP |

## The shape

### Contracts (abstraction always; nothing family- or card-specific)

| Where | What |
|---|---|
| `Idrak.Abstraction` (`Tensors`) | `TensorLayout`: shape, strides (in elements), offset; `IsContiguous`; built once per tensor, read on hot paths as spans. `ElementType`: an `IKind` with its name, its size in bits, whether it is a float, and its conversions; a `SlotTable` registry `ElementTypes` with the built-ins float32, bfloat16, float16, int64, int32, int8, uint8, int4 and bool. An unknown type fails with "not registered". |
| `Idrak.Abstraction` (`Devices`) | `Storage` gains its element type and a 64-bit length in elements and in bytes. `BackendCapabilities` gains, per operation: the element types it takes and whether it takes strided input. The dispatcher (plan 9's operations as data) reads them: a strided input to an operation that needs contiguous input is made contiguous once, by the backend's own copy, and the telemetry counts it. |
| Each backend | Strided reads in the kernels that gain the most (element-wise, reductions, matrix products' operands, attention's Q/K/V), contiguous-only elsewhere; per-type kernels where a type is native to the card (measured at run time: tensor cores, bf16 support), conversions elsewhere. |

### Settings (nothing fixed that someone may need to adjust)

- `ComputeResources` / run options: the default float type for new tensors (`IDRAK_DEFAULT_DTYPE`, default float32, so
  nothing changes until asked); the type for weights loaded from a checkpoint (keep the file's type, or a chosen one);
  whether views are used (`IDRAK_VIEWS`, default on once stage V passes; off reproduces today's copies for a check).
- `idrak tune`, `idrak pretrain`, `idrak serve`, `idrak run`: `--dtype` for weights and activations, carried to the apps'
  settings and the benches' variables.

## Stages

At most two agents at a time (Idrak's limit). Every stage keeps every existing test's output unchanged with the new
behaviour on and off (rule 71); every stage measures memory and time before and after, cold and warm apart (rule 75),
on the benches, with the device's own reported limits (rule 84), never one card's numbers.

| Stage | What | Done when | Test (`IDRAK_FILTER`) |
|---|---|---|---|
| W0 | **First (the owner, 2026-11-04: "fails when offloading is disabled and when enabled it becomes very very slow, like it's only running on CPU and system memory"; Ollama runs the same model).** Today `IDRAK_OFFLOAD=1` (`ComputeResources.OffloadToHostMemory`, `src/Idrak.Gpu/Cuda/CudaBackend.Offload.cs`) puts tensors that do not fit in pinned system memory mapped into the GPU, and stages a layer's offloaded weights to the GPU before it computes; for generation that moves every offloaded weight across the bus on every token, and a kernel reading mapped memory directly re-reads it across the bus. W0 measures before anything changes: telemetry of the bytes crossing the bus per token, layers staged per token and kernels reading mapped memory, printed by `idrak run`; the owner runs the failing model once with it, to confirm the cause | The cause shown by the owner's run, in numbers | `offload telemetry` (new) on `cpu` and `vulkan` |
| W1 | **A device and CPU split, as Ollama does, sized at run time:** a contract plus registry `WeightPlacements` (`Idrak.Abstraction`) places whole layers: on the device as far as the measured free memory (or the user's limit) allows, the rest **computed by the CPU from system memory**, so only the activations cross the bus, once per layer boundary, and the CPU layers' KV cache stays in system memory. No weight crosses the bus per token. Placements: `device`, `split` (this), `stream` (today's staging with prefetch, kept as a measured alternative where the bus is faster than the CPU), `auto` (picks by measuring the bus and the CPU at load). Settings: `IDRAK_WEIGHT_PLACEMENT`, `IDRAK_DEVICE_LAYERS` (a count, or `auto`), `--device-layers` on `idrak run`, `serve`, `tune`. Without any setting, a model that does not fit starts in `auto` instead of failing | Results equal the all-device run (rule 71); a model larger than the device runs at a speed near Ollama's on the owner's card, measured cold and warm | `weight placement` (new) on `cpu` and `vulkan`; `cuda` and `hip` by the owner |
| W2 | **Lazy start and growing:** the model file is mapped and each layer's weights are read on first use, so a model of any size starts at once; layers move between the device and the CPU as the measured free memory changes (more on the device when memory frees, fewer under pressure, never a failure for want of room). R1 uses the same contract for single experts | Start time independent of the model's size; a run continues when another program takes device memory | `weight placement` |
| V0 | `TensorLayout` in `Tensor`: strides and offset carried, every tensor still contiguous; `Contiguous()` (a no-op today); the capabilities fields | All CPU groups unchanged; the API dump regenerated | `tensor layout` (new) |
| V1 | Views on CPU: `Transpose`, `Permute`, `Narrow`, `Select`, `Expand`, slicing return views; the CPU kernels read strides where it pays (element-wise, reductions, matmul operands), `Contiguous()` before the rest; autograd through views (a view's gradient lands in the base's gradient at the view's place) | Every CPU group unchanged; peak memory measured lower on the attention and decoding benches | `tensor views` (new), `gradient`, the decoding and attention groups |
| V2 | Views on CUDA, then Vulkan, then HIP: the same kernels strided, the dispatcher's contiguous fallback elsewhere | The owner's GPU runs unchanged and lower in memory | the owner's GPU filters |
| T0 | **Element and block types** (the owner, 2026-11-05): `ElementType` and its registry `ElementTypes` describe both per-element types (float32, bf16, f16, fp8 E4M3/E5M2, int64, int32, int8, uint8, int4, int2, bool) and **block formats** (fractional bits per value, a block size, scales and zero points, as ggml's Q4_K, IQ2 and MXFP4 and the safetensors block-scaled FP8): each entry carries its conversions and names its kernels, so a new format is registered, never coded into `Tensor`. A type a backend has no kernel for converts through the dispatcher, counted. Types are data, not C# generics: GPU kernels are written per type anyway and rule 89 forbids type tests on a value-type `T` in hot paths; generic math is used only inside single CPU kernels where the JIT specializes it. `Storage` typed; float32 the only type in use | All groups unchanged | `element types` (new) |
| T1 | int64 and int32 indices: embeddings, gathers, scatters, MoE routing, token ids, top-k; the float-index paths removed | Indices exact above 2^24 (a test past it); MoE no longer gathers with floats | `element types`, the MoE and decoding groups |
| Q0 | **Quantized weights kept packed** (the owner, 2026-11-05; the largest memory gain for low-bit models): a GGUF or block-scaled safetensors weight stays in its file's format in memory, and the matrix product expands each block per tile inside the kernel (as llama.cpp does), on CPU first, then Vulkan, CUDA and HIP; every registered block format of T0 (the existing 15 GGUF types, plan 36's C14 types, safetensors FP8 and I8 with their scales). Setting: `IDRAK_KEEP_PACKED` (default on once Q0 passes; off expands at load as today, for checking) | Results equal today's expand-at-load path bit for bit (the same block arithmetic, rule 71); a 4-bit model's weight memory measured near its file's size; tokens per second measured, cold and warm apart | `quant types`, `gguf:`, `pretrained` on `cpu`; the product kernels on `vulkan`; `cuda`, `hip` by the owner |
| T2 | bfloat16 and float16 storage for weights and activations, compute in the card's native type where it has one, float32 accumulation; the mixed-precision switch (C10) becomes this. **fp8** storage and compute where the card supports it (measured at run time), a conversion elsewhere. **KV cache in int8 or fp8** (the owner, 2026-11-05): a setting `IDRAK_KV_CACHE_TYPE` (`auto` keeps the compute type; `int8`, `fp8`), the scales per head and block, attention reading it directly | Memory halved for bf16 runs on the benches, the KV cache 2-4× smaller; results within the tolerances the reference gives (Hugging Face / PyTorch / llama.cpp at the same types) | the training and inference groups with `IDRAK_DEFAULT_DTYPE=bf16`; `KV-cache decoding` with each cache type |
| T3 | int8 and int4 as real types: the packed formats, `AdamW8Bit`'s state and the quantized loaders move onto them; the bit-packing in float storage removed. **int2, ternary (BitNet b1.58), fp8 and ggml's block formats** join as registered packed formats through T0 (the owner, 2026-11-05): storage formats expanded in the kernel (Q0's path), not types anything computes in | Same results as the packed form, bit for bit; each new format against its reference (gguf-py, ml_dtypes) | quantization groups, `quant types` |
| K0 | **A paged KV cache with shared prompt prefixes** (the owner, 2026-11-05): the cache in fixed pages from a pool; requests whose prompts start alike share those pages (copy on write when they diverge), as vLLM does; `idrak serve` uses it for concurrent requests. Settings: `IDRAK_KV_PAGE_TOKENS`, `IDRAK_KV_SHARE_PREFIX` | Results equal the unpaged cache bit for bit; memory of N requests sharing a prompt measured near one prompt's cache plus their own tokens | `KV-cache decoding`, `cli serve` on `cpu`; the attention kernels on `vulkan` |
| W3 | **CPU layers read the file's mapping** (the owner, 2026-11-05): a split model's CPU layers (W1) and lazily loaded layers (W2) read their weights from the checkpoint's file mapping in place where the stored format is the one computed in (or Q0 keeps it packed), so the weights are not copied a second time into memory; the operating system's page cache holds them once | Results unchanged; the process's memory measured lower by the CPU layers' bytes | `weight placement`, `lazy load` on `cpu` |
| R0 | MoE routing on the device (moved here from plan 36's L4 by the owner, 2026-11-04): plan 36's G-1b adds the routing and the row-offset expert product as backend operations with CPU kernels, GPUs falling back to host routing (counted). R0 adds the device kernels: top-k over the router's logits, per-expert counts, an exclusive scan into offsets, the permutation, and the expert product reading its rows through the offsets, on Vulkan, CUDA and HIP, with int32 indices from T1 and a fixed summation order (rule 86); graph capture then allowed for MoE. No host round trip per layer | The MoE groups unchanged on every device; host syncs per layer zero on the GPU benches; the owner's GPU runs | the MoE groups on `vulkan` here, `cuda` and `hip` by the owner |
| R1 | Expert offloading (the owner, 2026-11-04, after "only few are active and other on demand", as Ollama / llama.cpp do with `--cpu-moe`): today every expert of a `MixtureOfExperts` layer (`src/Idrak/Layers/MixtureOfExperts.cs`) stays in device memory, so a large MoE model needs its full size on the device. A contract plus registry `ExpertPlacements` (`Idrak.Abstraction`) chooses where experts live: `resident` (today's), `offload` (cold experts in host memory, pinned where the backend can, or read from the checkpoint's file mapping; the hot ones in a device cache) and `auto` (resident when the model fits the measured free device memory, else `offload`). The device cache's size comes from the measured free memory or a user limit (`ComputeResources`); which experts stay is learned from the routing counts at run time (most used first), never fixed per model. While a layer computes, the next layer's chosen experts are copied in on the backend's copy stream (independent work spread, rule 79). A cold expert either is copied in or runs on the CPU (a setting, as llama.cpp's `--cpu-moe`). Settings: `IDRAK_MOE_EXPERTS` (`resident`, `offload`, `auto`), `IDRAK_MOE_DEVICE_EXPERTS_MB`, `IDRAK_MOE_COLD` (`copy`, `cpu`), and `--moe-experts` on `idrak run`, `serve`, `tune` | Results equal the resident run bit for bit on every placement (rule 71); a model whose experts exceed the device's free memory runs; device memory measured lower and the time per token measured, cold and warm apart (rule 75) | `moe placement` (new) on `cpu` and `vulkan` here; `cuda` and `hip` by the owner |
| N0 | **int8 and fp8 gradient compression with error feedback** (the owner, 2026-11-05): new entries in the existing gradient-compression registry (beside `none`, `bf16`, `top-k`), with each rank keeping the rounding error it did not send and adding it to the next step's gradient, so nothing drifts; the scales per bucket; the sum in a fixed order (rule 86). Settings through the existing `IDRAK_CLUSTER_*` compression option | The simulated cluster's loss curve within the fixture's tolerance of the float32 run over N steps (as the `bf16 wire` test does); bytes on the wire measured 4× fewer than float32 | `cluster` compression groups on `cpu` |
| N1 | **Sharded weights sent smaller** (after Q0 and T2): with `IDRAK_CLUSTER_SHARD=parameters`, a layer's weights are gathered in their stored type (bf16, fp8) or, for a frozen packed base (QLoRA-style fine-tuning across machines), in their packed format, expanded on the receiving rank; activations between pipeline stages in bf16 where the run computes in it | Results equal the float32 gather where the stored type is float32, and within the type's reference tolerance otherwise; gather bytes measured 2-8× fewer | `cluster` sharding and pipeline groups on `cpu` |
| N2 | **Pipeline stages sized by each machine's measured memory, and each rank mapping only its shard** (W1's `auto` and W2/W3 across machines): a rank reports its measured free memory; the stages are cut so each fits (a user's per-rank count overrides); each rank maps the checkpoint and touches only its shard's or stage's pages (the existing per-shard reads, through W3's mapping), so a rank starts fast and holds only its part | A cluster of unequal simulated ranks places stages that fit each; results equal the even split; each rank's memory measured at its part's size | `cluster` pipeline and topology groups on `cpu` |
| N3 | **Tensor parallelism**: a layer's matrix products split across devices or machines (column-split then row-split, as Megatron does for attention and the MLP), with the all-gather and reduce-scatter through `Idrak.Network`'s collectives, so a layer too large for one device trains and runs; a placement `tensor` beside `single` and `pipeline`, combinable with data parallelism. Card-agnostic: the split count from the measured devices or a user setting | Results equal the single-device run within the fixed-order sum (rule 86); the simulated cluster and the owner's multi-GPU runs | `cluster` groups on `cpu` (simulated ranks), across `vulkan` for the device path; `cuda`/`hip` multi-GPU by the owner |
| S0 | 64-bit lengths: `Storage` and `Tensor.Size` as `long`; index math 64-bit where it can overflow, 32-bit where the size is proven small (measured: no slowdown on the benches) | A tensor above 2^31 elements on CPU; the benches unchanged in speed | `large tensors` (new; CPU, memory permitting, skipped with the reason otherwise) |

**Where each device is tested (the owner, 2026-11-04):** the sessions test on `cpu` and on `vulkan` through Mesa's
software driver (llvmpipe, installed in the cloud container; Idrak's Vulkan group passes on it), so every Vulkan kernel
of this plan is proven before it lands. The owner tests `cuda` and `hip` on real cards, from the exact commands each
piece hands over. llvmpipe proves results, never speed: speed is measured only on the owner's devices.

W comes first: it turns today's failure or crawl on a model larger than the device into a run at a usable speed. Then
V: views pay off with float32 alone. Then T0, T1 (exact indices are a correctness fix), then Q0 (the largest memory
gain for low-bit models, built on T0's block formats), then T2 with the KV cache types, T3 with int2 and the other
packed formats, K0 and W3. N0 and N2 need neither types nor packing and may run as soon as the owner resumes; N1
follows Q0 and T2; N3 after them, the largest of the network stages. S0 last: it touches every kernel's index math and gains only for the largest tensors.

## Decisions for the owner

1. **Public API breadth.** `Tensor.Size` and `Storage.Length` becoming `long` (S0) is a breaking change for apps that
   read them. Recommended: add `LongSize` / `LongLength` in S0 and keep the `int` ones (checked, failing past 2^31)
   until the owner removes them.
2. **The default type.** Recommended: float32 stays the default; bf16 is chosen per run (`--dtype`) until the owner's
   runs show bf16 results the owner accepts on the owner's models.
3. **Order against plan 36.** Recommended: plan 37 starts after plan 36's G-1b; it may run beside plan 36's tool phases
   (G1 onward), since they share no files, two agents at most in all.

## Rules that hold for every stage

- `CLAUDE.md` and `docs/optimization.md`: spans over strings, memory walked in order (rule 72), work done once, device
  memory that does not grow (rule 76), the declared block and the device's reported limits for every kernel (rules
  80-87), no type or null test on a value-type `T` on hot paths (rules 88-90).
- Results proven unchanged (rule 71) at every stage, with the new behaviour on and off.
- Card-agnostic: what a card supports (bf16, tensor cores) is measured at run time, never assumed.
- Tests: targeted only (`IDRAK_DEVICES=cpu IDRAK_FILTER=<group>`); GPU runs and benches are the owner's, given as exact
  PowerShell commands.
