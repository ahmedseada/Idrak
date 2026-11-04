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
3. **64-bit sizes (C3):** a tensor and a storage may hold more than 2^31 elements; index arithmetic that can overflow
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
| V0 | `TensorLayout` in `Tensor`: strides and offset carried, every tensor still contiguous; `Contiguous()` (a no-op today); the capabilities fields | All CPU groups unchanged; the API dump regenerated | `tensor layout` (new) |
| V1 | Views on CPU: `Transpose`, `Permute`, `Narrow`, `Select`, `Expand`, slicing return views; the CPU kernels read strides where it pays (element-wise, reductions, matmul operands), `Contiguous()` before the rest; autograd through views (a view's gradient lands in the base's gradient at the view's place) | Every CPU group unchanged; peak memory measured lower on the attention and decoding benches | `tensor views` (new), `gradient`, the decoding and attention groups |
| V2 | Views on CUDA, then Vulkan, then HIP: the same kernels strided, the dispatcher's contiguous fallback elsewhere | The owner's GPU runs unchanged and lower in memory | the owner's GPU filters |
| T0 | `ElementType` and the registry; `Storage` typed; float32 the only type in use; conversions on every backend | All groups unchanged | `element types` (new) |
| T1 | int64 and int32 indices: embeddings, gathers, scatters, MoE routing, token ids, top-k; the float-index paths removed | Indices exact above 2^24 (a test past it); MoE no longer gathers with floats | `element types`, the MoE and decoding groups |
| T2 | bfloat16 and float16 storage for weights and activations, compute in the card's native type where it has one, float32 accumulation; the mixed-precision switch (C10) becomes this | Memory halved for bf16 runs on the benches; results within the tolerances the reference gives (Hugging Face / PyTorch in bf16) | the training and inference groups with `IDRAK_DEFAULT_DTYPE=bf16` |
| T3 | int8 and int4 as real types: the packed formats, `AdamW8Bit`'s state and the quantized loaders move onto them; the bit-packing in float storage removed | Same results as the packed form, bit for bit | quantization groups |
| S0 | 64-bit lengths: `Storage` and `Tensor.Size` as `long`; index math 64-bit where it can overflow, 32-bit where the size is proven small (measured: no slowdown on the benches) | A tensor above 2^31 elements on CPU; the benches unchanged in speed | `large tensors` (new; CPU, memory permitting, skipped with the reason otherwise) |

V comes first: views pay off with float32 alone. T1 before T2: exact indices are a correctness fix. S0 last: it touches
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
