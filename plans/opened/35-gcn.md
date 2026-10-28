# Plan 35: GCN kernels for HIP (AMD's PTX: written in C#, assembled by Idrak, no hipRTC, no SDK)

**Status:** planned 2026-10-28 against `main` a2a6eb1. Built 2026-10-28: G0b (goldens, 0841e86), the SDK removed (e88c9d9), G0a (the code object writer and the gfx9 target, e85efbd), G0c (the first encoder forms, `fill_f32` in `GcnKernels`, loading and launching through `hipModuleLaunchKernel`'s `extra` buffer, the `gcn gate:` tests; 8e49053). Next: **the owner's gate run** on the gfx90c, the driver's runtime alone; G1 waits on it. **Scope (the owner, 2026-10-28):** option B of
leftover 8.11, with a name and a structure equivalent to CUDA's, not a generic one. Idrak writes AMD's kernels in AMD's
own assembly, GCN, in C#, as it writes NVIDIA's in PTX. Idrak's GCN assembler turns them into a code object, and the
HIP runtime that ships with the graphics driver loads it. **No SDK, like CUDA (the owner, 2026-10-28):** the library
depends on no part of the HIP SDK or ROCm's compilers, as it depends on no CUDA Toolkit. hipRTC, its HIP C++ kernels
and the HIP C++ plug-in kernel are removed (done 2026-10-28, ahead of G2b); GCN is HIP's only kernel path, as PTX is CUDA's.

## The name: GCN

NVIDIA's kernels are PTX, so `PtxKernels`. AMD's are **GCN**: the instruction set, the LLVM target and the code
object's triple are all `amdgcn` (`amdgcn-amd-amdhsa--gfx90c`) for every AMD GPU HIP runs on: GCN 5, CDNA and RDNA.
The owner's gfx90c is GCN 5. The device API stays HIP (AMD's CUDA), and devices stay `hip:N`.

| NVIDIA (exists) | AMD (this plan) | What it is |
|---|---|---|
| `cuda:N` | `hip:N` (exists) | The device |
| `Cuda/CudaDriver.cs` | `Hip/HipRuntime.cs` (exists) | The driver's API, loaded at run time |
| `Cuda/CudaBackend*.cs` | `Hip/HipBackend*.cs` (exists) | The backend: operations to kernels |
| `Cuda/CudaDeviceLimits.cs` | `Hip/HipDeviceLimits.cs` (exists) | What the device reports |
| `Cuda/PtxKernels*.cs`: PTX text written in C# | **`Hip/GcnKernels*.cs`**: GCN assembly text written in C# | The kernels |
| `ptxas`, inside NVIDIA's driver | **`Hip/GcnAssembler*.cs`**, inside Idrak (AMD's driver has none) | Text to machine code: registers, waits, hazards, encoding |
| a cubin made by the driver | **`Hip/GcnCodeObject.cs`** | The ELF code object `hipModuleLoadData` loads |
| `.target sm_XX` (from compute capability) | **`Hip/GcnTargets.cs`**: `.amdgcn_target gfxXXX` (from the reported `gcnArchName`) | The instruction-set family |
| `Cuda/CudaKernel.cs`: a plug-in's PTX text | **`Hip/GcnKernel.cs`**: a plug-in's GCN text (replaces `HipKernel`, whose HIP C++ needed hipRTC) | Plug-ins' kernels |
| `Cuda/TuningCache.cs`, `CudaMeasuredChoices` | `Hip/HipKernelCache.cs` (exists), **`HipMeasuredChoices`** | Kept code and measured choices |

### GCN text as PTX text is written

PTX is written with virtual registers (`%f1`, `%r2`), and `ptxas` gives them hardware registers. GCN kernels are
written the same way: virtual registers (`%v1` for a vector register, one value per lane; `%s1` for a scalar register,
one value per wave), labels and branches, and the instructions of the GCN reference with their real mnemonics.
`GcnAssembler` does what `ptxas` does: it allocates registers, inserts the waits the memory counters need, fills the
ISA's hazards and encodes the bytes. It then prints the same kernel with physical registers, which is valid input
for LLVM's `llvm-mc`. That printed text is the test oracle. Every kernel assembled by Idrak equals, byte for byte, what
`llvm-mc` makes of its printed text (checked in the cloud container, which has LLVM 18 with AMD targets; LLVM is never
a dependency, rule 70). Checked 2026-10-28: `llvm-mc` and `ld.lld` made a loadable gfx90c code object (flags `0x132`,
gfx90c and xnack) from a hand-written `fill_f32`.

### Where the kernels come from

- **Written in GCN, as PTX is,** for the paths that decide speed: element-wise kernels, row norms and softmaxes,
  the few-row products decoding runs on (int8, int4, bfloat16, float32; leftover 8.10), decode attention, and the
  fused packed decoding kernels.
- **Lowered to GCN from the Vulkan catalog** (`GcnKernels.Spirv.cs`) for everything else: about 90 kernels already
  written once in `KernelBuilder` (125 operations), whose SPIR-V is turned into the same GCN text. This brings
  training, convolution, pooling, CTC, sampling, rotary and the rest to the device without a second hand-written copy.
- A hand-written kernel replaces a lowered one only when it is measured faster and its results are proven equal
  (rule 71), the same bar CUDA's kernels meet.

## Contracts, registries and settings

| Contract | Registry (`SlotTable`) | Built in | Namespace |
|---|---|---|---|
| `IGcnTarget`: the families it handles (from the reported version and feature flags), wave sizes, register files, encodings, wait counters, hazards, the ELF machine value | `GcnTargets` | `gfx9` (GCN 5, CDNA); later `gfx10`, `gfx11`, `gfx12` | `Idrak.Gpu.Abstractions` |
| `IGeneratedKernelDispatcher`: launch a catalog kernel by name with storages and a push block | (one per backend) | Vulkan, HIP | `Idrak.Gpu.Abstractions` |

A family that is not registered fails with "not registered", naming the reported target. The device then runs every
operation through the host fallback, and `idrak devices` and `idrak doctor` say so and why.

| Setting | Variable | Default |
|---|---|---|
| Code object format | `IDRAK_HIP_CODE_OBJECT` (exists) | As today: the newest the runtime loads |
| Wave size where a family runs both | `IDRAK_HIP_WAVE` = `auto`, `32`, `64` | `auto`: measured |
| Occupancy the allocator aims for | `IDRAK_HIP_OCCUPANCY` = `auto` or waves per SIMD | `auto`: measured |
| Kernels on or off, the cache | `IDRAK_HIP_KERNELS`, `IDRAK_HIP_KERNEL_CACHE` (exist) | As today; the cache key is the GCN text's hash, the target, the device and the runtime and library versions (no compiler version: Idrak is the compiler) |

Each setting is carried into the HIP options beside the others, the apps' settings and the benches' variables.

## Card-agnostic

- The family comes from the version the device reports (as PTX's target comes from compute capability). It is the
  instruction set, never a card, vendor id or memory size, and it makes no tuning decision.
- The ELF header needs a machine value per target (gfx90c has its own), and each family has a name. The
  instruction-set tables, `Hip/GcnTargets*.cs` (family names, encodings, `GcnTargets.MachineValues.cs`), hold these as
  format data and are the only files where `HipNoVendorNames` allows target literals in code. Vendor ids and names stay
  forbidden there too, and no tuning decision branches on a target name.
- Block size, wave size, occupancy, k splits and the wave-operation form (DPP, `ds_swizzle`, `ds_bpermute`, LDS) are
  measured on the device (rule 84). Matrix units (MFMA, WMMA) are chosen by a probe kernel that runs and checks its
  result, never by name.

## Rules this plan is held to (read 2026-10-28)

- **0, 70:** .NET only, no outside program at run time; LLVM is a development oracle, never called by the library or
  the tests.
- **1, 11, 74:** the assembler parses GCN text as UTF-8 bytes over spans, and tables are looked up by span
  (`SearchValues`, alternate lookups, rules 15 and 17): no string per token.
- **44, 45, 55:** pooled buffers returned in `finally`; no `stackalloc` sized by a kernel.
- **71:** every kernel proven against the CPU backend. Every lowered kernel's results are equal to Vulkan's, within the
  tolerance the device-against-CPU test already uses.
- **75, 76:** each kernel is assembled once per device and kept in the cache. The cold assembly is measured apart from
  the warm load and reported by the telemetry (rule 78).
- **77:** the existing pool. Nothing in this plan grows device memory.
- **80-87, for GCN:** a register per loaded value (the allocator never merges two loads' registers within a pass);
  grids that fill every compute unit; few launches (fused decoding kernels); coalesced reads; choices by the reported
  limits; per-call work done once at a kernel's start; sums in a fixed tree, never atomics; each kernel declares its
  block (`.amdhsa` metadata `max_flat_workgroup_size`), which bounds the registers the allocator may use.
- **88-90:** the assembler's hot paths do no generic range search over value types and no type or null test on a
  value-type `T`. An allocation test that fails is diagnosed by type, step and `DOTNET_TieredCompilation=0`.
- **Owner's rules:** contracts plus `SlotTable`s; the inventory and `api/Idrak.txt` regenerated where they change;
  no version bump, no CHANGELOG; `main` only; at most two agents; CPU-filtered test runs only here.

## Implementation phases

Two slots, **A** and **B**, never more. A wave is a dependency order, not a barrier: when a slot finishes, it takes
the next piece whose inputs are merged. Each piece follows the owner's fix workflow: write, update tests, build only
the changed projects once, run its CPU filter once with `--no-build`, then merge and push to `main`.

| Phase | Slot | What | Files | Needs | CPU filter here |
|---|---|---|---|---|---|
| **G0a** | A | **Code object writer.** ELF64 shared object: `.text`, the 64-byte kernel descriptors, `.dynsym`/`.dynstr`/`.hash`/`.dynamic`, the `NT_AMDGPU_METADATA` note in MessagePack (written by hand). `GcnTargets` with `gfx9` and the machine-values file | `GcnCodeObject.cs`, `GcnTargets*.cs`, `Abstractions/IGcnTarget.cs` | none | `gcn code object` |
| **G0b** | B | **Goldens.** `tools/gcn-golden` (development only): `llvm-mc` and `ld.lld` turn checked-in GCN text into bytes and code objects, kept under `tests/Idrak.Tests.Library/Gcn/Golden/`. The `HipNoVendorNames` exemption | `tools/gcn-golden/*`, golden files, `HipTests.cs` | none | `hip: no vendor` |
| **G0c** | A | **The gate.** `fill_f32` in GCN with physical registers, encoded by the few forms it needs, written by G0a, equal to its golden, loaded and run by a new HIP test | `GcnKernels.cs` (one kernel), `HipTests.cs` | G0a, G0b | `gcn code object` |
| — | owner | **The gate run**: the HIP test on the gfx90c with the HIP SDK uninstalled or off `PATH` (the driver alone). If the driver's runtime refuses the object (it may need `amd_comgr`), the plan stops and returns to the owner | | G0c | |
| **G1a** | A | **Assembler: parse and encode.** GCN text as UTF-8 bytes; every gfx9 form the kernels use (SOP1/2/K/C/P, SMEM, VOP1/2/3/C, VOP3P, DS, GLOBAL), operands, literals, modifiers, labels and branches; each form against its golden | `GcnAssembler.Parse.cs`, `GcnAssembler.Encode.cs`, `GcnTargets.Gfx9.cs` | gate passed | `gcn assembler` |
| **G1b** | B | **The GCN interpreter** (tests only): decodes the bytes and runs them on the CPU (waves of 64, EXEC, VCC, SCC, SGPRs and VGPRs, LDS, barriers between a block's waves, global memory, the kernel arguments). Written from the ISA reference, apart from the encoder | `tests/Idrak.Tests.Library/Gcn/GcnInterpreter*.cs` | gate passed | `gcn interpreter` |
| **G2a** | A | **Assembler: the `ptxas` part.** Virtual registers; liveness over the branches (a register written under a narrowed EXEC stays live from before, for its other lanes); allocation of VGPRs and SGPRs within the declared block's budget; `s_waitcnt` for vmcnt, lgkmcnt and expcnt; gfx9 hazards (`s_nop`); the descriptor's register counts and segment sizes from use; the physical-register printer | `GcnAssembler.Registers.cs`, `.Waits.cs`, `.Hazards.cs`, `.Print.cs` | G1a | `gcn assembler` |
| **G2b** | B | **GCN the only path.** The SDK's half is **done early (the owner, 2026-10-28: "delete now so no agent uses it wrongly")**: `HipRtc.cs`, the HIP C++ kernels (`HipKernels*.cs`), their operation overrides and `HipKernel` (C++ plug-ins) deleted, with every message, test and doc that named hipRTC or the SDK; the runtime is looked for only where the driver (Windows) or ROCm (Linux) puts it, never in `HIP_PATH`; every kernel operation on hip:0 takes the host fallback and says why. Left for G2b: `HipBackend.Kernels.cs` loading GCN code objects through the kernel cache; `idrak kernels -d hip:0` and `idrak kernels dump gcn` (the GCN text, virtual and physical) | `HipBackend.Kernels.cs`, `HipKernelCache.cs`, CLI | gate passed | `hip` |
| **G3a** | A | **Hand-written GCN: element-wise.** fill, affine, axpy, multiply-add, add/sub/mul/max/min, row-vector add, the activations (ReLU, sigmoid, tanh, GELU both, SiLU, square, abs, exp, log, sqrt, sin, cos, sign), with LLVM's math sequences read from clang's output and proven against the CPU. All run in the interpreter here | `GcnKernels.Pointwise.cs` | G2a, G1b | `gcn kernels` |
| **G3b** | B | **Hand-written GCN: rows.** RMS norm (with and without gain), layer norm (fused, training, gradients), softmax, scaled masked softmax, log-softmax top-k: reductions in a fixed LDS tree, `ds_bpermute` and DPP as measured candidates | `GcnKernels.Rows.cs` | G2a, G1b | `gcn kernels` |
| — | owner | **First device run**: the HIP group on the driver alone; cold assembly and warm load timed | | G3a, G3b | |
| **G4a** | A | **SPIR-V to GCN.** A reader for the subset `KernelBuilder` emits (anything else refused by name); uniformity (scalar or vector per value and branch); structured ifs and loops on EXEC; shared memory, barriers, `SubgroupShuffleXor`; the math functions. Output: the same GCN text G2a assembles | `GcnKernels.Spirv*.cs` | G2a | `gcn spirv` |
| **G4b** | B | **One operation layer for Vulkan and HIP.** The thin Vulkan operation bodies move to a shared base over `IGeneratedKernelDispatcher`. HIP's dispatcher maps bindings to 64-bit pointers and the push block to scalars in the kernel arguments, and passes grid sizes as explicit arguments (no hidden arguments). Vulkan unchanged by proof | `Abstractions/IGeneratedKernelDispatcher.cs`, `Vulkan/VulkanBackend.*.cs`, `HipBackend.Generated.cs` | G2b | `vulkan` CPU-side groups, `hip` |
| — | owner | **Vulkan unchanged**: the Vulkan groups give the same counts and results as before G4b | | G4b | |
| **G5a, G5b** | A, B | **Parity by kernel group**, split between the slots: A takes products, attention (spans, tiled, decoding), sampling and penalties, rotary, the fused decoding kernels; B takes training (backward passes, AdamW, 8-bit Adam, norms' gradients), convolution and pooling, CTC, resampling, shapes. Each group is lowered through G4a, run in the interpreter against the CPU, then wired. Done when HIP overrides every member Vulkan does | `GcnKernels.Spirv*.cs`, `HipBackend.*.cs` | G4a, G4b | `gcn kernels`, `hip` |
| — | owner | **The whole list on hip:0** with `gcn`, the device-against-CPU test included; `idrak kernels -d hip:0` shows no host fallback where Vulkan has a kernel | | G5a, G5b | |
| **G6a** | A | **Hand-written decoding products**: few-row int8, int4, bfloat16 and float32 products per row count with k splits (leftover 8.10); each replaces its lowered kernel only when faster and proven equal | `GcnKernels.Products.cs` | G5a | `gcn kernels` |
| **G6b** | B | **Hand-written decode attention and fused packed kernels**, under the same bar | `GcnKernels.Attention.cs`, `GcnKernels.Decoding.cs` | G5a | `gcn kernels` |
| — | owner | **The benches**: gemv and decode on hip:0 against Vulkan on the same GPU, and against plan 8's hipRTC numbers (2026-10-26) | | G6a, G6b | |
| **G7a** | A | **Measured choices**: block size, k splits, occupancy and wave-operation form, measured per device, driver and build; `HipMeasuredChoices` in `MeasuredChoiceProviders` | `HipBackend.Tuning.cs` | G6a | `hip` |
| **G7b** | B | **`GcnKernel` for plug-ins**: a plug-in's GCN text, assembled once a device and cached, launched through `IKernelLauncher<GcnKernel>`, its arguments checked against what the text declares (as `CudaKernel` does for PTX). The public API dump regenerated | `GcnKernel.cs`, `HipBackend.Plugins.cs`, `api/Idrak.txt` | G2b | `hip`, `public API` |
| **G8a, G8b** | A, B | **The other families**: `gfx10` (RDNA 1-2: wave32/64, `s_waitcnt_vscnt`, its hazards) and `gfx11` (RDNA 3: its encodings, `s_delay_alu`), then `gfx12` (RDNA 4: split wait counters) and CDNA's gfx9 variants (gfx90a and later: AGPRs, packed float32). Each comes with its goldens and an interpreter mode | `GcnTargets.Gfx10.cs` … | G5 | `gcn assembler`, `gcn interpreter` |
| **G9** | A | **Matrix units**: MFMA (CDNA) and WMMA (RDNA 3-4) in hand-written GCN and lowered from `KernelBuilder.Matrix`, each chosen by its probe | `GcnKernels.Matrix.cs` | G8 | `gcn kernels` |

**Done when:**

- **G0:** the gfx90c, with the driver's runtime alone, runs a code object Idrak wrote.
- **G1-G3:** every gfx9 form equals its golden; every hand-written kernel passes the interpreter here and the HIP group
  on hip:0 on the driver alone.
- **G4-G5:** Vulkan's results unchanged; HIP overrides every member Vulkan does; the whole list passes on hip:0.
- **G6-G7:** faster than plan 8's hipRTC numbers on every bench row, the gap to Vulkan stated per row with its cause;
  measured choices kept and reused.
- **G8-G9:** each family's goldens and interpreter pass here; hardware runs are the owner's, when a card is at hand.

## The owner's commands (the form every phase hands over)

After each piece is merged, one list: the pull, one build, then the runs with `--no-build`. The gate (G0c), for one:

```powershell
cd D:\Projects\Idrak; git checkout main; git pull origin main
dotnet build -c Release tests\Idrak.Tests
$env:IDRAK_DEVICES="hip"; $env:IDRAK_FILTER="gcn"; dotnet run -c Release --no-build --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\gcn-gate-fails.txt
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
```

For the gate, the HIP SDK is uninstalled, or its folder is off `PATH` and `HIP_PATH` unset for that shell, so only the
driver's files can be found. The phase's hand-over
gives the exact lines.

## Decisions for the owner

1. **Hand-written plus lowered (recommended):** hand-written GCN for the hot paths and the rest lowered from the
   Vulkan catalog. The other option is every kernel hand-written in GCN, as PTX is: the same structure as CUDA, about
   125 kernels written a second time, and G5 several times longer.
2. **The shared operation layer (G4b, recommended):** one copy of Vulkan's operation bodies, proven unchanged by the
   owner's Vulkan runs. The other option is HIP's own thin copies: no Vulkan change, two copies to keep in step.
3. **The order of G8's families**, by the cards the owner can reach.

## Risks

- **The gate:** the driver's runtime may need SDK pieces (`amd_comgr`) to load any code object. G0 settles that before
  anything else; if it does, the plan returns to the owner, since the SDK is not an option.
- **Until G3 and G5:** with hipRTC removed (2026-10-28), operations without a GCN kernel yet run on the host on hip:0,
  slower than the hipRTC kernels were on a machine with the SDK, until G3 and G5 bring them back.
- **Hazards and waits:** a mistake hangs or silently corrupts. Each family's rules come from AMD's ISA reference and
  are compared with clang's output for the same kernels; the interpreter checks the waits, and only hardware checks
  the hazards.
- **One reading of the ISA:** the encoder is checked against `llvm-mc`; the interpreter is written apart from it.
  What both misread shows only on hardware.
- **Hardware at hand:** one gfx90c (gfx9, wave64, no matrix units). RDNA, CDNA and the matrix units stay unproven until
  the owner runs them.

## Out of this plan

hipGraph capture (8.4), pinned staging and offloading (8.7), the backend API points (8.8). Leftovers this plan closes:
8.2 and 8.6 (G5), 8.5 (G9), 8.10 (G6), 8.11 (G0-G5).

## Effort (rough, two slots)

G0: one to two days. G1-G2: about two weeks. G3: a week. G4: one to two weeks. G5: about two weeks. G6-G7: a week plus
the owner's runs. G8: about a week per family. G9: a week once a card with matrix units is at hand.
