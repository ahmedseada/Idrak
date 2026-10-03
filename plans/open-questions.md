# Open questions and checks for the maintainer

## Decided (2026-10-03)

- Merging into `main`: not yet.
- Public backend API (12c): stays in the plan, not started.
- LinkedIn series: dropped for now.
- Native Linux GPU drivers: the maintainer tests them from a live USB when one is at hand.
- `idrak serve` gets a default port of its own (not 11434, which another local server uses); `-p/--port`, the config
  and an environment variable choose another. Done: 7317, then `-p`, `IDRAK_PORT`, `serve.port` (plans/idrak-cli.md).
- Arabic messages: only if the console can show them correctly (right to left, joined letters); otherwise dropped.
- Branches: idrak-cli was fast-forwarded into `architecture` and retired; all work continues on `architecture`.
- Round 3, all built and merged (now on architecture): mixture of experts, fast sliding-window and soft-cap kernels
  (CPU and Vulkan run; CUDA written, hardware check 9), dataset loaders, the teacher pattern (`idrak distill`),
  provider-neutral serving names, Arabic output. Arabic: the tool shapes and reorders it itself for terminals that
  do not (plans/idrak-cli.md, "Arabic messages"), checked against the Unicode conformance suite and an independent
  implementation; `--lang ar` built. The look in real Windows terminals is hardware check 11 below.


Decisions and hardware checks that wait for the maintainer, collected while work continued. Each item says what is
needed, the options, and the recommended one. Nothing here blocks the work in progress.

## Decisions

| # | Question | Options | Recommended |
|---|---|---|---|
| 1 | When to merge `architecture` (which now holds the CLI) into `main` | a) after the CLI lands and real-GPU checks pass · b) now · c) per feature | a |
| 2 | Public backend API (item 12c) | a) start now, using the HIP backend's list of awkward contract points (plans/8-hip.md) · b) keep parked until HIP runs on real AMD hardware | b |
| 3 | Provider-neutral names for the serving API (plans/plug-in.md, "Noted for later") | a) rename with obsolete forwarders in the next release · b) keep the old names longer | a, done |
| 4 | SPIR-V memory model for the cooperative-matrix kernels: SPIRV-Tools 2025.2 and later reject cooperative matrices without the Vulkan memory model; the drivers tested accept the kernels as they are | a) move the matrix kernels to the Vulkan memory model as its own task, tested on real GPUs · b) leave as is until a driver refuses them | a, after the RTX checks below |
| 5 | README tagline | a) "Deep learning in pure .NET. Every GPU. Zero dependencies." · b) another | a |
| 6 | LinkedIn series | a) post 2 next (GPU code without the CUDA toolkit) · b) post 4 next (an LLM on a phone) ; English only or with Arabic | a, English only |
| 7 | Native Linux GPU drivers (NVIDIA's Linux driver, RADV, ANV, NVK) | a) live USB on a laptop · b) a cloud GPU instance (a T4 also covers compute 7.5) · c) keep "WSL2 only" | c for now; a when convenient |
| 8 | Mixture of experts (Mixtral, Qwen-MoE): built (branch feature-moe, plans/plug-in.md); grouped expert kernels remain | a) next after the CLI · b) later | done |
| 9 | `idrak serve`'s default port: 11434, the one common local-model clients connect to by default (so they work without settings), or a port of Idrak's own | a) keep 11434 for drop-in use · b) an Idrak port, with `-p 11434` documented for those clients | a |

## Hardware checks to run

| # | Machine | What | Command |
|---|---|---|---|
| 1 | RTX 5070 Ti | Single-pass bfloat16 matrix units under MixedPrecision: correctness and speed against float32 | `$env:IDRAK_DEVICES="vulkan:0"; dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan matmul` and `$env:IDRAK_FILTER="matrix units"; dotnet run -c Release --project tests/Idrak.Tests` |
| 2 | Any CUDA machine | The recorded fine-tuning step with a custom optimizer (could not run without CUDA) | `$env:IDRAK_FILTER="fine-tuning options"; dotnet run -c Release --project tests/Idrak.Tests` |
| 3 | RTX 5070 Ti, laptops | The full lists after the plug-in work (315+ tests), CUDA and Vulkan | the per-device blocks in installation/windows.md |
| 4 | An AMD discrete GPU with ROCm, or Windows with the HIP SDK | The HIP backend's first run | plans/8-hip.md, "Commands to validate" |
| 5 | Phone | The width probe and the chat after the merged kernels | installation/android-termux.md, steps 8 and 9 |
| 6 | RTX 5070 Ti, laptops | Mixture of experts on CUDA (composed operations, never run on CUDA here) and a real model's speed | `$env:IDRAK_DEVICES="cuda:0"; $env:IDRAK_FILTER="experts"; dotnet run -c Release --project tests/Idrak.Tests`, then `idrak c Qwen/Qwen1.5-MoE-A2.7B-Chat -w int4`, then `idrak b` on it |
| 7 | Any machine | The `idrak` tool once the CLI lands: `idrak doctor`, `idrak devices`, `idrak c MODEL`, `idrak s MODEL`, `idrak b MODEL` | plans/idrak-cli.md, "Examples" |
| 8 | Honor tablet (later) | CPU tests and, if its GPU has a Vulkan driver reachable from Termux (Mali through Mesa's Panfrost/PanVK, or Adreno through Turnip), the Vulkan list and a chat; first check the SoC and GPU with the phone guide's step 6 | installation/android-termux.md, steps 1-9 |
| 9 | RTX 5070 Ti, a laptop with CUDA | The windowed and soft-capped CUDA kernels (decoding, `attention_flash_*`, the float32 gradients; never run): correctness, then the windowed decoder's speed against the composed path | `$env:IDRAK_DEVICES="cuda:0"; $env:IDRAK_FILTER="window kernels"; dotnet run -c Release --project tests/Idrak.Tests`, then `$env:IDRAK_FILTER="sliding windows"` and `"decoding"`; a failure there: `$env:IDRAK_CUDA_DEBUG="1"` names the kernel, `$env:IDRAK_WINDOW_KERNELS="0"` confirms the composed path still passes |
| 10 | RTX 5070 Ti | The windowed kernels on Vulkan, and their speed | `$env:IDRAK_DEVICES="vulkan:0"; $env:IDRAK_FILTER="window kernels"; dotnet run -c Release --project tests/Idrak.Tests`, then `dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan window` |
| 11 | Windows (Windows Terminal, a PowerShell or cmd window), Git Bash, a Linux terminal | Arabic messages read right to left with joined letters; the device table aligned; `logical` reversed where `auto` is right (or the other way round in terminals that reorder text themselves) | `idrak help --lang ar`, `idrak doctor --lang ar`, `idrak devices --lang ar`, then the same with `--lang-render logical` (plans/idrak-cli.md, "Arabic messages") |
