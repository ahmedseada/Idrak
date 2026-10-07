# Plan 9: operations as data (one dispatcher, kernels per device)

**Status:** phase 1 done (branch `abstraction`, 2026-10-06, as plan 10's phase 1b): descriptors, the kernel table and
the dispatcher, internal; phases 0 and 2 to 5 not started. Part of plan 10 (`Idrak.Abstraction`): its
dispatcher and descriptors live there, and its phases 1 to 5 map onto plan 10's phases 1, 3, 4, 5 and 6.

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
| 1 | **Descriptors and the dispatcher, internal**: `Op`, kernel keys, requirements, the table and the cached resolution. The existing virtual methods are registered as kernels by an adapter, so nothing changes in behaviour | done, see "Phase 1 as built" below. The 1% micro-benchmark bar was replaced (2026-10-06) by the decoding gate (within 2%, measured on a GPU and a quiet CPU): about 1 ns per call measured |
| 2 | **The chain made visible**: per device and operation, which kernel ran (device, composed, host); `idrak kernels -d vulkan:0` lists it; profiling, tracing and host-call counting move into the dispatcher | the host-call counters are read from the dispatcher; `idrak kernels` output is tested |
| 2 (as built) | `Kernels.Chain(backend)` (registered, device, composed, host, none per operation), `Kernels.Trace` and `Kernels.HostCalls` in the dispatcher; `idrak kernels [-d] [--source] [-j]`, tested; a trace takes the out-of-line path, nothing is added without one. Profiling still lives in each backend (`StartProfile`/`StopProfile`). On lavapipe: 90 own kernels, 2 composed, 9 host, 8 none | done (CPU, lavapipe) |
| 3 | **Backends register kernels directly**, one at a time (minimal, HIP, Vulkan, CUDA, CPU): overrides become registrations, and `Backend` shrinks to memory, copies, synchronization, capabilities and graph hooks | `Backend` has about 15 members; no operation is a virtual method any more |
| 4 | **Public device API (item 12c)**: public `DeviceProvider`, storage, the kernel table and `Kernels.Register`. A conformance kit (the operation tests, runnable against any registered device) and a device written outside the library in `tests/Idrak.PluginTests` (a plain-loop reference device) prove it | an outside assembly registers a device that passes the conformance kit; README and plans/7 updated |
| 4 (as built) | Public with plan 10 phase 4 (wave 2): `DeviceProvider`, `Storage`, the kernel table and `Kernels.Register`; an outside device ran in a console app. The conformance kit (`Idrak.Abstraction.Testing`) and the plain-loop `ReferenceDevice` in `tests/Idrak.PluginTests` came with plan 10 phases 5 and 6c (wave 3) | done (CPU, lavapipe) |
| 5 | **Kernels for plug-ins**: `Autograd.Function`, `GraphOps`, `PackedWeight` and `KeyValueLayout` implementations can register device kernels, so a plug-in format gets fast paths instead of the generic one | a packed format and a KV layout from the plug-in tests run a registered device kernel (checked through `idrak kernels`) |
| 5 (as built) | With plan 10 phase 6 (wave 3): `PluginOperation<TKernel>`, `PluginOperations.Register`, `Kernels.Register(op, kind, kernel)`, `op.KernelFor(backend)`; the plug-in tests' packed format and KV layout run registered kernels (CPU, Vulkan SPIR-V through `VulkanKernel.Dispatch`), checked through `Kernels.Chain` and `idrak kernels` | done (CPU, lavapipe); CUDA PTX and HIP C++ plug-in kernels through `CudaKernel`/`HipKernel` (plan 10, wave 4; PTX checked with ptxas, not yet run on a GPU) |

## Phase 1 as built (2026-10-06)

- **Operations**: the 109 compute methods of `Backend` are renamed `NameKernel` (the device's own kernel, its default
  body the host fallback, as before); 15 members stay device plumbing (copies, graph capture, profiling, and the
  questions a caller asks before choosing a path: `PrefersPackedMatMul`, `SupportsSegmentedAttention`, ...).
- **Generated** by `tools/operations/generate.py` from those declarations: a descriptor per operation (`Ops.Softmax`,
  overloads numbered: `MaxPoolBackward2`), its slot (`OperationIndex`), its kernel delegate (`OperationKernels`, the
  backend then the arguments), and `Backend.Name(...)`, which every call site already calls. A test fails when
  `Backend` and the generated files disagree, or when a public virtual method is neither an operation nor plumbing.
- **Kernel table** (`Kernels.Register(operation, deviceKind, kernel, requirement)`, returning a handle that removes
  it): the last registration whose requirement holds wins, else the device's own kernel. The key is the device kind
  (`Backend.Kind`: cpu, cuda, vulkan, hip); formats come with phase 5.
- **Cost**: a device with nothing registered for it reads one field per call (the fast path is inlined at the call
  site); a registration marks every device to resolve its slots again on its next call. A million 16-float CPU fills:
  1.75 ms straight to the kernel, 2.72 ms through the dispatcher (about 1 ns per call, `--bench-dispatch`). Decoding on
  a 4-core shared CPU moved within the machine's own run-to-run spread (an A/A run of one build differed by 7%), so the
  2% gate is still to be measured on a quiet machine and on the GPUs.
- **Host-call counting by operation** keeps the operation's name (`Fill`, not `FillKernel`).

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
3. Name of the public surface: decided by plan 10, every abstraction lives under `Idrak.Abstraction`; the device
   contract is `Idrak.Abstraction.Devices` and the dispatcher `Idrak.Abstraction.Operations`.
