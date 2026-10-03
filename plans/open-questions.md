# Open questions and checks for the maintainer

## Decided (2026-10-03)

- Merging into `main`: not yet.
- Public backend API (12c): stays in the plan, not started.
- LinkedIn series: dropped for now.
- Native Linux GPU drivers: the maintainer tests them from a live USB when one is at hand.
- `idrak serve` gets a default port of its own (not 11434, which another local server uses); `-p/--port`, the config
  and an environment variable choose another.
- Arabic messages: only if the console can show them correctly (right to left, joined letters); otherwise dropped.
  Result: the tool shapes and reorders Arabic itself for terminals that do not (plans/idrak-cli.md, "Arabic
  messages"), checked against the Unicode conformance suite and an independent implementation; `--lang ar` built.
  The look in real Windows terminals is hardware check 7 below.
- Started: mixture of experts, fast sliding-window kernels, dataset loaders, the teacher pattern (distillation),
  provider-neutral serving names, Arabic output (render check first).


Decisions and hardware checks that wait for the maintainer, collected while work continued. Each item says what is
needed, the options, and the recommended one. Nothing here blocks the work in progress.

## Decisions

| # | Question | Options | Recommended |
|---|---|---|---|
| 1 | When to merge `architecture` (and later `idrak-cli`) into `main` | a) after the CLI lands and real-GPU checks pass · b) now · c) per feature | a |
| 2 | Public backend API (item 12c) | a) start now, using the HIP backend's list of awkward contract points (plans/8-hip.md) · b) keep parked until HIP runs on real AMD hardware | b |
| 3 | Provider-neutral names for the serving API (plans/plug-in.md, "Noted for later") | a) rename with obsolete forwarders in the next release · b) keep the old names longer | a |
| 4 | SPIR-V memory model for the cooperative-matrix kernels: SPIRV-Tools 2025.2 and later reject cooperative matrices without the Vulkan memory model; the drivers tested accept the kernels as they are | a) move the matrix kernels to the Vulkan memory model as its own task, tested on real GPUs · b) leave as is until a driver refuses them | a, after the RTX checks below |
| 5 | README tagline | a) "Deep learning in pure .NET. Every GPU. Zero dependencies." · b) another | a |
| 6 | LinkedIn series | a) post 2 next (GPU code without the CUDA toolkit) · b) post 4 next (an LLM on a phone) ; English only or with Arabic | a, English only |
| 7 | Native Linux GPU drivers (NVIDIA's Linux driver, RADV, ANV, NVK) | a) live USB on a laptop · b) a cloud GPU instance (a T4 also covers compute 7.5) · c) keep "WSL2 only" | c for now; a when convenient |
| 9 | `idrak serve`'s default port: 11434, the one common local-model clients connect to by default (so they work without settings), or a port of Idrak's own | a) keep 11434 for drop-in use · b) an Idrak port, with `-p 11434` documented for those clients | a |
| 8 | Mixture of experts (Mixtral, Qwen-MoE): planned in plans/plug-in.md, not built | a) next after the CLI · b) later | a |

## Hardware checks to run

| # | Machine | What | Command |
|---|---|---|---|
| 1 | RTX 5070 Ti | Single-pass bfloat16 matrix units under MixedPrecision: correctness and speed against float32 | `$env:IDRAK_DEVICES="vulkan:0"; dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan matmul` and `$env:IDRAK_FILTER="matrix units"; dotnet run -c Release --project tests/Idrak.Tests` |
| 2 | Any CUDA machine | The recorded fine-tuning step with a custom optimizer (could not run without CUDA) | `$env:IDRAK_FILTER="fine-tuning options"; dotnet run -c Release --project tests/Idrak.Tests` |
| 3 | RTX 5070 Ti, laptops | The full lists after the plug-in work (315+ tests), CUDA and Vulkan | the per-device blocks in installation/windows.md |
| 4 | An AMD discrete GPU with ROCm, or Windows with the HIP SDK | The HIP backend's first run | plans/8-hip.md, "Commands to validate" |
| 5 | Phone | The width probe and the chat after the merged kernels | installation/android-termux.md, steps 8 and 9 |
| 6 | Any machine | The `idrak` tool once the CLI lands: `idrak doctor`, `idrak devices`, `idrak c MODEL`, `idrak s MODEL`, `idrak b MODEL` | plans/idrak-cli.md, "Examples" |
| 7 | Windows (Windows Terminal, a PowerShell or cmd window), Git Bash, a Linux terminal | Arabic messages read right to left with joined letters; the device table aligned; `logical` reversed where `auto` is right (or the other way round in terminals that reorder text themselves) | `idrak help --lang ar`, `idrak doctor --lang ar`, `idrak devices --lang ar`, then the same with `--lang-render logical` (plans/idrak-cli.md, "Arabic messages") |
