# Plan 37: tensor layout, views, element types and 64-bit sizes (memory first)

**Status (2026-11-04):** planned, nothing built. Written from the owner's request (2026-11-04): "add it as separate plan
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
4. **64-bit sizes (C3):** a tensor and a storage may hold more than 2^31 elements; index arithmetic that can overflow
   is 64-bit.

## Today

| What | Where | Effect |
|---|---|---|
| Shape only, no strides or offset | `Tensor._shape` (`int[]`), `Tensor.Size` (`int`), `src/Idrak.Abstraction/Tensors/Tensor.cs` | Every tensor is one contiguous block from element 0 |
| `Permute`, `Transpose`, `Narrow` copy | `Tensor.Advanced.cs` ~276, ~354, ~369: `Empty(outShape)` then `Backend.Permute` | A new block per call; attention's head transposes and every slice cost memory and a pass over it |
| `Reshape` is the only view | `Tensor.Operations.cs` ~127 | Shares storage when contiguous |
| One element type, float32 | `Storage` (`Backend.cs` ~200): `Length` (`int`) floats | bf16, f16, int8, int4 kept as bits inside floats (`Tensor.Quantized.cs`, `PackedFormat`, `AdamW8Bit`) |
| Indices as floats | gathers, embeddings, MoE routing | Exact only below 2^24 |
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
| T0 | `ElementType` and the registry; `Storage` typed; float32 the only type in use; conversions on every backend | All groups unchanged | `element types` (new) |
| T1 | int64 and int32 indices: embeddings, gathers, scatters, MoE routing, token ids, top-k; the float-index paths removed | Indices exact above 2^24 (a test past it); MoE no longer gathers with floats | `element types`, the MoE and decoding groups |
| T2 | bfloat16 and float16 storage for weights and activations, compute in the card's native type where it has one, float32 accumulation; the mixed-precision switch (C10) becomes this | Memory halved for bf16 runs on the benches; results within the tolerances the reference gives (Hugging Face / PyTorch in bf16) | the training and inference groups with `IDRAK_DEFAULT_DTYPE=bf16` |
| T3 | int8 and int4 as real types: the packed formats, `AdamW8Bit`'s state and the quantized loaders move onto them; the bit-packing in float storage removed | Same results as the packed form, bit for bit | quantization groups |
| R0 | MoE routing on the device (moved here from plan 36's L4 by the owner, 2026-11-04): plan 36's G-1b adds the routing and the row-offset expert product as backend operations with CPU kernels, GPUs falling back to host routing (counted). R0 adds the device kernels: top-k over the router's logits, per-expert counts, an exclusive scan into offsets, the permutation, and the expert product reading its rows through the offsets, on Vulkan, CUDA and HIP, with int32 indices from T1 and a fixed summation order (rule 86); graph capture then allowed for MoE. No host round trip per layer | The MoE groups unchanged on every device; host syncs per layer zero on the GPU benches; the owner's GPU runs | the MoE groups on `vulkan` here, `cuda` and `hip` by the owner |
| R1 | Expert offloading (the owner, 2026-11-04, after "only few are active and other on demand", as Ollama / llama.cpp do with `--cpu-moe`): today every expert of a `MixtureOfExperts` layer (`src/Idrak/Layers/MixtureOfExperts.cs`) stays in device memory, so a large MoE model needs its full size on the device. A contract plus registry `ExpertPlacements` (`Idrak.Abstraction`) chooses where experts live: `resident` (today's), `offload` (cold experts in host memory, pinned where the backend can, or read from the checkpoint's file mapping; the hot ones in a device cache) and `auto` (resident when the model fits the measured free device memory, else `offload`). The device cache's size comes from the measured free memory or a user limit (`ComputeResources`); which experts stay is learned from the routing counts at run time (most used first), never fixed per model. While a layer computes, the next layer's chosen experts are copied in on the backend's copy stream (independent work spread, rule 79). A cold expert either is copied in or runs on the CPU (a setting, as llama.cpp's `--cpu-moe`). Settings: `IDRAK_MOE_EXPERTS` (`resident`, `offload`, `auto`), `IDRAK_MOE_DEVICE_EXPERTS_MB`, `IDRAK_MOE_COLD` (`copy`, `cpu`), and `--moe-experts` on `idrak run`, `serve`, `tune` | Results equal the resident run bit for bit on every placement (rule 71); a model whose experts exceed the device's free memory runs; device memory measured lower and the time per token measured, cold and warm apart (rule 75) | `moe placement` (new) on `cpu` and `vulkan` here; `cuda` and `hip` by the owner |
| S0 | 64-bit lengths: `Storage` and `Tensor.Size` as `long`; index math 64-bit where it can overflow, 32-bit where the size is proven small (measured: no slowdown on the benches) | A tensor above 2^31 elements on CPU; the benches unchanged in speed | `large tensors` (new; CPU, memory permitting, skipped with the reason otherwise) |

**Where each device is tested (the owner, 2026-11-04):** the sessions test on `cpu` and on `vulkan` through Mesa's
software driver (llvmpipe, installed in the cloud container; Idrak's Vulkan group passes on it), so every Vulkan kernel
of this plan is proven before it lands. The owner tests `cuda` and `hip` on real cards, from the exact commands each
piece hands over. llvmpipe proves results, never speed: speed is measured only on the owner's devices.

W comes first: it turns today's failure or crawl on a model larger than the device into a run at a usable speed. Then
V: views pay off with float32 alone. T1 before T2: exact indices are a correctness fix. S0 last: it touches
every kernel's index math and gains only for the largest tensors.

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
