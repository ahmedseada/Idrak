# Plan 9: operations as data (one dispatcher, kernels per device)

**Status:** planned (branch `abstraction`, 2026-10-05). Nothing built yet.

## Why

Every device today is a subclass of the internal `Backend`: 10 abstract and 126 virtual methods, one per operation
(`Fill`, `Unary`, `Binary`, `Im2Col`, `Permute`, the decoding kernels, ...). A virtual method's default body is the
host fallback; a backend overrides the ones it runs itself (CUDA 130 overrides, Vulkan 139, HIP 34). Earlier work
already took every device check out of the core (no `is CudaBackend` outside `src/Idrak/Backends`), so the seam is
clean. What it cannot do:

| Today | Example of what fails |
|---|---|
| An operation is a method on `Backend` | adding one means editing `Backend` and deciding, in each backend, whether to override it |
| Kernels belong to the backend that ships them | a faster CUDA kernel for `Softmax`, or a Vulkan kernel for an `Autograd.Function`, cannot come from another package |
| The fallback chain is implicit | which path an operation took (device kernel, composed operations, host) is visible only through host-call counters |
| `Backend` is internal | a device cannot be added from outside the library (plan 7, item 12c; plans/README.md, "Not supported yet") |
| Plug-in formats run generic code | packed weights and KV cache layouts of one's own get the dequantize or composed path (plans/plug-in.md, items 22 and the KV row) |
| Profiling, tracing and graph capture are written per backend | each backend counts its own calls and records its own graphs |

## The idea

An operation becomes a descriptor, and a kernel becomes an entry in a table keyed by the operation and the device it
runs on. One dispatcher picks the kernel for each call and walks one explicit chain when there is none:

```
call site (Tensor, layers)        Ops.Softmax(x, y, rows, cols)
        │
dispatcher                        resolve (Softmax, cuda:0) once, cache the slot
        │
kernel table for the device       1. registered device kernel whose requirement holds (capabilities, format)
                                  2. composed kernel (other operations, on the same device)
                                  3. host kernel (copy to the host, run the CPU code, copy back)
```

- **Operation descriptor** (`Op`): a name, the storages it reads and writes, a small attribute struct (shapes,
  scalars, enums), the CPU reference kernel, and optionally a composed form.
- **Kernel key**: the operation, the device kind (`cuda`, `vulkan`, ...) and, where it matters, a format (int8 or
  bfloat16 weights, a KV layout name).
- **Requirement**: a predicate over the device's capabilities (matrix units, subgroup size, compute capability as
  reported, never a card name), so several kernels for one operation can coexist and the best one that applies wins.
- **Resolution is cached per (operation, device)**: a call is an array index and a delegate call, the same cost as
  today's virtual call. The table changes only when something registers (at start-up, or when a plug-in loads).

## Phases

Each phase ends with every test passing on every device we have (CPU, CUDA, Vulkan on lavapipe, plus the 5070 Ti and
one other real GPU before merging), and the decoding benchmarks within 2% of the previous phase.

| # | Phase | Done when |
|---|---|---|
| 0 | **Inventory**: a generated catalogue of the 136 operations (inputs, outputs, attributes, which backends run each natively, which fall back). A test regenerates it and fails when it drifts | `plans/9-operations-catalogue.md` exists and the test keeps it current |
| 1 | **Descriptors and the dispatcher, internal**: `Op`, kernel keys, requirements, the table and the cached resolution. The existing virtual methods are registered as kernels by an adapter, so nothing changes in behaviour | every call site goes through the dispatcher; 434 of 434 on every device; a micro-benchmark of a million tiny dispatches within 1% of the virtual call |
| 2 | **The chain made visible**: per device and operation, which kernel ran (device, composed, host); `idrak kernels -d vulkan:0` lists it; profiling, tracing and host-call counting move into the dispatcher | the host-call counters are read from the dispatcher; `idrak kernels` output is tested |
| 3 | **Backends register kernels directly**, one at a time (minimal, HIP, Vulkan, CUDA, CPU): overrides become registrations, and `Backend` shrinks to memory, copies, synchronization, capabilities and graph hooks | `Backend` has about 15 members; no operation is a virtual method any more |
| 4 | **Public device API (item 12c)**: public `DeviceProvider`, storage, the kernel table and `Kernels.Register`. A conformance kit (the operation tests, runnable against any registered device) and a device written outside the library in `tests/Idrak.PluginTests` (a plain-loop reference device) prove it | an outside assembly registers a device that passes the conformance kit; README and plans/7 updated |
| 5 | **Kernels for plug-ins**: `Autograd.Function`, `GraphOps`, `PackedWeight` and `KeyValueLayout` implementations can register device kernels, so a plug-in format gets fast paths instead of the generic one | a packed format and a KV layout from the plug-in tests run a registered device kernel (checked through `idrak kernels`) |

After phase 5, a separate plan: a lazy graph built on the descriptors (trace a step, fuse element-wise chains, plan
memory, replay on any device), which would unify CUDA graphs, Vulkan's recorded steps (and the choice in
plans/README.md, "Future improvements") and ONNX export.

## What it is not

- Not a new tensor API: call sites keep their shape, only what is behind them changes.
- Not tuned to any device: requirements read reported capabilities and measurements, as every rule in plans/README.md
  says; a test fails if a requirement names a vendor or a card.
- Not a rewrite of the kernels: PTX, SPIR-V and the CPU code stay; only how they are found changes.

## Risks

| Risk | Answer |
|---|---|
| Dispatch costs more than a virtual call | resolution cached in a per-device slot array; phase 1 measures it and does not merge above 1% |
| Graph capture (CUDA and Vulkan) depends on calls going straight to the backend | the dispatcher calls the same kernels in the same order; the graph tests run in every phase |
| A large diff across 400+ files | phases are separate merges; phase 1 keeps the old methods behind the adapter, so it can land without touching the backends |
| The public API freezes too early | phase 4 marks it preview in 0.y.z; the operation set is versioned, and an unknown operation takes the fallback chain |

## Open questions

1. Attribute structs per operation, or one generic bag? (Proposed: a struct per operation; no allocation per call.)
2. Should the composed form be written once per operation (in the descriptor) or per device? (Proposed: once, in the
   descriptor, since composed means "other operations, on any device".)
3. Name of the public surface: `Idrak.Devices` namespace, or keep `Idrak.Backends`? (Proposed: `Idrak.Devices`.)
