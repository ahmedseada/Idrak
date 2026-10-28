# Plan 35: Idrak's own AMDGPU code (HIP kernels with no hipRTC, no SDK)

**Status:** planned 2026-10-28 against `main` 7525c69. Nothing built. **Scope (the owner, 2026-10-28): option B of
leftover 8.11.** Idrak writes the AMDGPU machine code and the code object itself, in C#, as it writes PTX for CUDA and
SPIR-V for Vulkan. The HIP runtime that ships with the graphics driver (or with ROCm) then loads it with
`hipModuleLoadData`. Leftover 8.11's other option (SPIR-V finished by the runtime) is not part of this plan.

## Why

Every HIP kernel today is HIP C++ compiled by hipRTC (`HipKernels*.cs`, `HipRtc.cs`). hipRTC ships with ROCm and the
HIP SDK, not with the graphics driver. On a machine with the driver alone (the owner's Windows laptop with the
Adrenalin driver, for one), HIP runs memory and copies on the GPU and **every kernel on the host**
(`HipBackend.Kernels.cs`: "no hipRTC: the HIP SDK or ROCm is needed"). The README's promise, "Every GPU. No native
dependencies.", holds for CUDA (PTX finished by the driver) and Vulkan (SPIR-V finished by the driver). It does not
hold for HIP, because AMD's runtime loads only finished machine code. If Idrak writes that machine code itself, HIP
needs only the driver, and the kernels become Idrak's to measure and change.

The second gain is coverage. HIP overrides 48 `Backend` members against CUDA's 135 and Vulkan's 125 (surveyed
2026-10-28). Training, convolution, pooling, sampling, rotary, CTC and the fused decoding path all take the host
fallback. Idrak already generates about 90 Vulkan kernels covering 125 operations, through `KernelBuilder` →
SPIR-V. With SPIR-V as the compiler's input, HIP gets that whole catalog, and every kernel added for Vulkan later
reaches HIP with no second copy.

## What exists (surveyed 2026-10-28)

| Piece | Where | Kept, changed or replaced |
|---|---|---|
| Runtime loading, devices, limits, the target string (`gcnArchName` with its feature flags), stream, allocator, copies | `Hip/HipRuntime.cs`, `HipDeviceLimits.cs`, `HipBackend.cs` | Kept |
| Code object loading (`hipModuleLoadData`), code object formats 4/5/6 tried in order (`IDRAK_HIP_CODE_OBJECT`) | `HipBackend.Kernels.cs`, `HipRtc.CodeObjectFormats` | Kept; the format list now also drives Idrak's writer |
| Kernel cache on disk (`IDRAK_HIP_KERNEL_CACHE`), keyed by source, options, target, device and versions | `HipKernelCache.cs` | Kept; key gains the compiler's name and the catalog's hash |
| HIP C++ kernels compiled by hipRTC | `HipKernels*.cs`, `HipRtc.cs` | Kept as the `hiprtc` compiler (a setting), no longer the only path |
| Plug-ins' HIP C++ kernels (`HipKernel`) | `HipKernel.cs`, `HipBackend.Plugins.cs` | Unchanged: C++ source needs hipRTC |
| Kernels written in C#, emitted as SPIR-V | `Vulkan/SpirV/KernelBuilder*.cs`, `Vulkan/VulkanKernels*.cs` | Unchanged; their SPIR-V becomes the AMDGPU compiler's input |
| Vulkan operations (fit check, variant choice, `Grid(name, n, storages, push)`) | `Vulkan/VulkanBackend.*.cs` | Shared with HIP (phase A3) |
| A test forbidding GPU target names in `src/Idrak.Gpu/Hip` | `tests/Idrak.Tests.Library/HipTests.cs` (`HipNoVendorNames`) | Kept; one data file exempted (decision 1) |
| LLVM 18 in the cloud container (`llvm-mc`, `llvm-objdump`, `llvm-readelf`, `clang`, `ld.lld`), with AMDGPU targets | the container | A development oracle only, never a dependency (rule 70) |

## The shape

```
SPIR-V words (KernelBuilder, unchanged)
  → SpirvReader          the subset KernelBuilder emits; anything else refused, naming the op
  → Machine IR           SSA values, structured regions (if, loop, merge) as SPIR-V gives them
  → Uniformity           which values are the same in every lane (scalar registers, SALU) and which vary (VGPR, VALU)
  → Lowering             EXEC masks for divergent ifs and loops; math sequences; memory; barriers; wave operations
  → Selection            instructions of one ISA family (IAmdgpuTarget)
  → Registers            linear scan over SSA, VGPR and SGPR apart, within the occupancy budget
  → Waits and hazards    s_waitcnt (or each family's counters), the ISA's documented hazards
  → Encoding             bytes per family, from the ISA's tables
  → Code object          ELF64 shared object: .text, kernel descriptors, dynamic symbols, the AMDGPU metadata note
  → hipModuleLoadData    the existing loader, cache and format fallback
```

Everything lives in `src/Idrak.Gpu/Amdgpu/`, internal to Idrak.Gpu:

```
src/Idrak.Gpu/Amdgpu/
  SpirvReader.cs           SPIR-V words → Machine IR (KernelBuilder's subset)
  MachineIr*.cs            values, blocks, regions; a printer for `idrak kernels dump`
  Uniformity.cs            scalar or vector, per value and per branch
  Lowering*.cs             control flow (EXEC), math, memory, LDS, barriers, wave operations
  Registers.cs             allocation, occupancy, the budget a target gives
  Waits.cs, Hazards.cs     counters and hazards, both driven by IAmdgpuTarget
  Targets/                 IAmdgpuTarget for each family: Gfx9 (GCN 5, CDNA), Gfx10 (RDNA 1-2), Gfx11 (RDNA 3), Gfx12 (RDNA 4)
  Encoding/                the families' instruction formats (SOP*, VOP*, DS, FLAT/GLOBAL, SMEM, VOP3P, MFMA/WMMA)
  CodeObject.cs            the ELF writer, the kernel descriptor, the metadata note (MessagePack, written by hand)
  AmdgpuCompiler.cs        the pipeline above; IHipKernelCompiler "idrak"
```

### Contracts and registries (the abstraction rule)

| Contract | Registry (`SlotTable`) | Built in | Lives in |
|---|---|---|---|
| `IAmdgpuTarget`: whether it handles a reported ISA version (major, minor, stepping and feature flags), wave sizes, register files, encodings, wait counters, hazards, the ELF machine value | `AmdgpuTargets` | `gfx9`, then `gfx10`, `gfx11`, `gfx12` (phase A6) | `Idrak.Gpu.Abstractions` (only Idrak.Gpu can use it) |
| `IHipKernelCompiler`: turns the kernel catalog into a code object for a device, or says why it cannot | `HipKernelCompilers` | `idrak` (this plan), `hiprtc` (today's path, moved behind the contract) | `Idrak.Gpu.Abstractions` |
| `IGeneratedKernelDispatcher`: launch a catalog kernel by name with storages and a push block | (none: one per backend) | Vulkan (descriptor sets, push constants), HIP (kernel arguments) | `Idrak.Gpu.Abstractions` |

An ISA family that is not registered fails with "not registered", naming the reported target. The device then runs
`hiprtc` when that compiler is present, or the host fallback, and `idrak devices` and `idrak doctor` say which and why.

### Settings (nothing fixed)

| Setting | Variable | Default |
|---|---|---|
| Which compiler builds HIP kernels | `IDRAK_HIP_COMPILER` = `auto`, `idrak`, `hiprtc` | `auto`: `idrak` when a registered target handles the device, else `hiprtc`, else the host |
| Code object format | `IDRAK_HIP_CODE_OBJECT` (exists) | The newest format the runtime loads, tried in order as today |
| Wave size on families that run both | `IDRAK_HIP_WAVE` = `auto`, `32`, `64` | `auto`: measured per kernel group (phase A7) |
| Occupancy target (waves per SIMD the allocator aims for) | `IDRAK_HIP_OCCUPANCY` = `auto` or a number | `auto`: measured (phase A7) |
| Kernels on or off, the cache | `IDRAK_HIP_KERNELS`, `IDRAK_HIP_KERNEL_CACHE` (exist) | As today |

The same settings are carried into `ComputeResources` or the HIP options where the library keeps the others, and into
the apps' settings and the benches' variables.

### Card-agnostic

- The ISA family is chosen from the version the device reports in its target string: the instruction set, like PTX's
  target from compute capability. It is never a card, a vendor id or a memory size, and it never makes a tuning
  decision.
- Wave size, block size, occupancy, k splits and the wave-operation lowering (DPP, `ds_swizzle`, `ds_bpermute` or LDS)
  are candidates measured on the device (rule 84), kept with the HIP measured choices per device, driver and build.
- Matrix units (MFMA, WMMA) are chosen by a probe: a one-instruction kernel is assembled, loaded and run, and its
  result is checked. They are never chosen by target name.
- Sums keep the fixed tree each Vulkan kernel already has (rule 86). The compiler never reorders a reduction.

## Phases

| Phase | What | Depends on | Agent |
|---|---|---|---|
| A0 | **The gate: the driver alone loads Idrak's code object.** `CodeObject.cs` (ELF, kernel descriptor, metadata note) and a hand-assembled gfx9 `fill_f32` kernel; its bytes checked against `llvm-mc`'s, and its ELF against `llvm-readelf` and a clang-built object of the same kernel. A HIP test loads and runs it. **The owner's run:** with the HIP SDK hidden (driver runtime only), the test passes on the gfx90c. If the runtime refuses it (it may need `amd_comgr` to read metadata; unknown until run), the plan stops here and goes back to the owner | none | A |
| A1 | **gfx9 encoder and target.** `IAmdgpuTarget` and `AmdgpuTargets`; the gfx9 instruction formats the lowering needs (SOP1/2/K/C/P, SMEM, VOP1/2/3/C, VOP3P, DS, GLOBAL); golden bytes for every form, generated once by `tools/amdgpu-golden` with `llvm-mc` and checked in, so the tests are pure C# | A0 | A |
| A2 | **The compiler middle, element-wise first.** `SpirvReader`, Machine IR, uniformity, structured control flow on EXEC, the math sequences (exp, log, rcp, rsq, sqrt, division, sin/cos, tanh, pow, erf: each as LLVM lowers it, read from clang's output for the same expression and proven against the CPU), registers, waits, gfx9 hazards. The element-wise and pointwise catalog compiles. **The AMDGPU interpreter** (tests only, `tests/Idrak.Tests.Library/Amdgpu/`) runs the emitted code of the subset Idrak emits on the CPU (waves, EXEC, SGPR/VGPR, LDS, barriers, global memory), so every compiled kernel is checked against the CPU backend in this container | A1 | A |
| A3 | **One operation layer for Vulkan and HIP.** The thin Vulkan operation bodies move to a shared base over `IGeneratedKernelDispatcher` (decision 2). Vulkan's results are proven unchanged by the owner's Vulkan runs (rule 71). HIP's dispatcher maps bindings to 64-bit pointers and the push block to scalars in the kernel argument segment, and grid sizes to explicit arguments (no hidden arguments, no dispatch pointer). `IHipKernelCompiler` with `idrak` and `hiprtc` behind it | A2 | B |
| A4 | **The whole catalog on gfx9.** Rows and reductions (LDS, barriers, `SubgroupShuffleXor` through `ds_bpermute`, DPP as a measured candidate), products, attention (spans, tiled, decoding), sampling and penalties, rotary, the fused packed decoding kernels, training (backward passes, AdamW, 8-bit Adam, norms' gradients), convolution and pooling, CTC, resampling. Each group compiles, passes the interpreter against the CPU, then the owner's hip:0 run. The HIP overrides reach Vulkan's 125 | A3 | A, B (by kernel group) |
| A5 | **Speed on the owner's card.** The gemv and decode benches on hip:0: `idrak` against `hiprtc` against Vulkan on the same GPU (leftover 8.10). Profiles drive the compiler's next candidates (scheduling, `s_clause` and load batching, VGPR budget, DPP). Rules 80-87 apply to the emitted code | A4 | A |
| A6 | **The other families.** `gfx10` (RDNA 1-2: wave32/64, `s_waitcnt_vscnt`, its hazards), `gfx11` (RDNA 3: VOPD optional, `s_delay_alu`, its encodings), `gfx12` (RDNA 4: split wait counters, new encodings), and CDNA's gfx9 variants (gfx90a and later: AGPRs, packed float32, the 942 memory model). Each: golden bytes from `llvm-mc`, the interpreter, then a run on that hardware when the owner can reach one | A4 | A, B |
| A7 | **Measured choices.** Wave size, occupancy, block size and k splits tuned per device by measurement, kept like Vulkan's and CUDA's (`MeasuredChoiceProviders` gains HIP) | A5 | A |
| A8 | **Matrix units.** `KernelBuilder.Matrix` (cooperative matrices) lowered to MFMA (CDNA) and WMMA (RDNA 3-4), each chosen by its probe. Needs a card with them | A6 | B |

**Done when:**

- **A0:** the owner's gfx90c, with the driver's runtime alone, loads and runs a code object Idrak wrote.
- **A1, A2:** golden encodings and the interpreter pass here (`IDRAK_DEVICES=cpu`, filter `amdgpu`); the pointwise HIP
  tests pass on the owner's hip:0 with `IDRAK_HIP_COMPILER=idrak` and the SDK hidden.
- **A3:** Vulkan's filtered tests give the same counts as before on the owner's machine; HIP's give the same with each
  compiler.
- **A4:** the whole test list on hip:0 with `idrak`, the device-against-CPU test included, and no host fallback for
  any operation Vulkan runs on the device (`idrak kernels -d hip:0`).
- **A5:** `idrak` measured equal to or faster than `hiprtc` on every bench row; the gap to Vulkan stated per row, with
  the cause where it remains.
- **A6-A8:** each family's goldens and interpreter pass here; each hardware run is the owner's, when a card is at hand.

## Testing

- **Encodings:** checked-in golden bytes per instruction form and family, made by `tools/amdgpu-golden` from `llvm-mc`
  (a development tool, not run by tests); the tests compare in C#, so they run on Windows, Linux and Android alike.
- **Code objects:** structure tests in C#. During development, the same objects are read by `llvm-readelf --notes`
  and `llvm-objdump -d`, and compared with clang's object for the same kernel.
- **Semantics:** the AMDGPU interpreter runs each compiled catalog kernel against the CPU backend. It is written from
  the ISA reference, apart from the encoder, so the two are not checked by the same reading. What both may misread is
  caught only on hardware.
- **Hardware:** the owner's HIP runs with `IDRAK_HIP_COMPILER=idrak`, with the SDK present and hidden, and the benches
  against `hiprtc` and Vulkan.
- Test group: `amdgpu` (new); the HIP tests keep `hip`.

## Decisions for the owner

1. **The ELF needs target names.** A code object's `e_flags` carry an `EF_AMDGPU_MACH_*` value per target (gfx90c has
   its own number). Proposed: one data file, `Amdgpu/Targets/MachineValues.cs`, maps reported targets to these values
   as format data (as PTX names its `.target`), and is exempted from `HipNoVendorNames`. No decision reads it.
2. **Share the Vulkan operations (A3)** through one base over `IGeneratedKernelDispatcher` (recommended: one copy of
   about 125 operation bodies, proven unchanged on Vulkan by the owner's runs), or give HIP its own thin copies (no
   Vulkan change, two copies to keep in step).
3. **hipRTC after A4:** kept as `IDRAK_HIP_COMPILER=hiprtc` and for plug-ins' `HipKernel` (recommended), or the
   library's HIP C++ kernels removed, leaving hipRTC for plug-ins only.
4. **Family order after gfx9 (A6):** by the cards the owner can reach for runs.

## Risks

- **The runtime may not load a code object without the SDK's pieces** (`amd_comgr` reads metadata in some runtime
  versions). A0 settles this before anything else is built.
- **Size.** It is a compiler back end: selection, registers, waits, hazards and four encoding families. The SPIR-V
  input is closed (KernelBuilder emits a known subset of about 150 ops and 23 math functions), which bounds it.
  Anything outside that subset is refused by name, never guessed.
- **Hazards and waits** are where wrong code hangs or corrupts a GPU silently. They follow the ISA references per
  family, are checked by the interpreter, and are compared against clang's output for the same kernels.
- **Precision.** Each math sequence follows LLVM's lowering for the same expression and is proven against the CPU
  (rule 71). Where Vulkan's tolerance is looser, HIP keeps the CPU's.
- **Hardware at hand:** one gfx90c (gfx9, wave64, no matrix units). RDNA, CDNA and matrix units stay unproven on
  hardware until the owner runs them.

## Out of this plan

hipGraph capture (8.4), pinned staging and offloading (8.7), the backend API points (8.8), and plug-ins writing
AMDGPU code themselves. Leftovers 8.2, 8.5, 8.6, 8.10 and 8.11 are closed by this plan's phases (A4, A8, A4, A5, A0-A4).

## Effort (rough)

A0: a day. A1-A2: about two weeks of agent work. A3: three to four days. A4: about two weeks, by kernel group. A5 and
A7: a week, gated by the owner's runs. A6: about a week per family, each gated by hardware. A8: a week once a card
with matrix units is at hand.

## Rules that apply

`docs/optimization.md` rules 70-79 (no platform dependence, results proven unchanged, device memory that does not
grow, cold and warm measured apart) and 80-87 (independent loads, a full device, few launches, coalesced reads, the
reported limits, per-call work once, a fixed summation order, the declared block) apply to the compiler's output
as they do to PTX. Rules 88-90 apply to the compiler's own hot paths (a kernel is compiled once a device and cached,
but the cold start is measured, rule 76).
