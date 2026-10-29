# HIP (AMD GPUs: the graphics driver on Windows, ROCm's runtime on Linux)

The HIP backend (`hip:0`, `hip:1`, …; [plan 8](../plans/done/8-hip.md)) needs no SDK, as CUDA needs no CUDA Toolkit:
only the HIP runtime the graphics driver installs (Windows) or ROCm's runtime library (Linux). Its kernels are Idrak's
own GCN code, written in C# and assembled by Idrak ([plan 35](../plans/opened/35-gcn.md)); until they land, memory,
uploads, downloads, copies and fills run on the GPU and every other operation on the CPU through the host fallback. On
a machine without HIP nothing changes: the device list shows `hip:-` with the reason, and every other backend works as
before.

What the backend needs at run time (nothing at build time):

| | Linux | Windows |
|---|---|---|
| Runtime (memory, copies, launches) | `libamdhip64.so` from ROCm's runtime | `amdhip64_6.dll` (or `amdhip64_7.dll`, `amdhip64.dll`), installed by the AMD graphics driver |
| Where it is looked for | the loader's search path, then `$ROCM_PATH/lib`, then `/opt/rocm/lib` | the DLL search path (the driver's runtime); no SDK folder |

## 1. Install

### Linux (ROCm's runtime)

Install ROCm's runtime for your distribution and GPU from AMD's ROCm installation guide (the `amdgpu-install` package,
or the distribution's ROCm packages), then give your user access to the GPU and log in again:

```bash
sudo usermod -aG render,video "$USER"
rocminfo | grep -E "Name:|Compute Unit|Wavefront"          # the GPU, its gfx target, compute units, wavefront size
ls /opt/rocm/lib/libamdhip64.so*
```

ROCm supports a list of GPUs per release; a GPU outside it reports no devices.

### Windows (the graphics driver)

Install a current AMD Adrenalin or PRO graphics driver; it installs the HIP runtime. Nothing else is needed.

## 2. Devices

```bash
dotnet run -c Release --project tests/Idrak.Tests -- --list-devices
```

Expected: a `hip:0` row with the GPU's name, memory, compute units, gfx target, wavefront width and HIP version,
"yes" in the plain-run column, and the note "not the default device (IDRAK_HIP_DEFAULT=1 to prefer it)". A `hip:-` row
gives the reason when no device was found (runtime missing, no supported GPU, no permission).

## 3. Tests

Linux:

```bash
IDRAK_DEVICES=hip:0 IDRAK_FILTER=hip dotnet run -c Release --project tests/Idrak.Tests -- --progress --failures-only --failures-file test-output/hip-fails.txt
```

Windows (PowerShell):

```powershell
$env:IDRAK_DEVICES="hip:0"; $env:IDRAK_FILTER="hip"; dotnet run -c Release --project tests\Idrak.Tests -- --progress --failures-only --failures-file test-output\hip-fails.txt
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
```

The HIP device test prints "hip:0: kernels loaded" or "unavailable (reason)"; until the GCN kernels land it says they
are not built yet, and checks memory, copies and the host fallback against the CPU. On a failure, send the failures
file and the run again with `IDRAK_TRACE=1`.

## 4. Profiling, verbose output, benches and telemetry

`hip:0` takes the same diagnostics as the other GPUs:

- **Profiler.** The backend times its kernels (`IKernelProfiler`) as CUDA's does: while a profile runs, each GCN launch
  is timed on the GPU between two HIP events (created once and reused), by the kernel's name, or by a label with its
  floating-point operations for products and attention. `idrak profile MODEL -d hip:0` lists the kernels of one decoding
  step, and `--profile` with `IDRAK_PROFILE_PROBES=kernels` adds them to any command's report. Until the GCN kernels
  land, the operations take the host fallback, so the table shows no kernel and the report says why. A runtime without
  the event functions times nothing, and the telemetry says which functions it lacks.
- **Verbose.** `--verbose` (or `IDRAK_VERBOSE=1`) prints, on the error stream, the first host fallback of each
  operation with the reason the kernels are unavailable, and each device failure once: a code object the runtime refused
  (each format tried with its error), a launch it refused (the kernel and its grid), a profile that cannot time.
- **Telemetry.** The same events reach every hook at `TelemetryLevel.Devices` (`DeviceFailed`, `HostFallback`), as on
  Vulkan.
- **Bench.** `--bench-hip [launch|copies|fill]` in the test runner times the launch cost of `fill_f32` (its code object
  loaded, cold and again; a launch over one float, queued, with a wait, and on the device; a launch over a large buffer),
  uploads, downloads and copies on the device, and fill bandwidth (`fill_f32` beside the runtime's own fill), each first
  call apart from the warm ones. The large buffer is a sixteenth of the memory the GPU has free (at most 256 MB), or
  `IDRAK_BENCH_HIP_MB`; `IDRAK_BENCH_HIP_REPEATS` sets the warm calls (default 20). Without a HIP device it prints one
  line saying why. `idrak bench -d hip:0` and `idrak report --bench -d hip:0` run their benchmarks on `hip:0` as on any
  device.

Windows (PowerShell), every output under `test-output\`:

```powershell
cd D:\Projects\Idrak
dotnet build -c Release tests\Idrak.Tests
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-hip > test-output\bench-hip.txt
dotnet run -c Release --no-build --project src\Idrak.Cli -- profile Qwen/Qwen3-0.6B -d hip:0 > test-output\profile-hip.txt
$env:IDRAK_PROFILE_OUT="test-output\profile-hip-run.txt"; $env:IDRAK_PROFILE_PROBES="kernels,operations"
dotnet run -c Release --no-build --project src\Idrak.Cli -- run Qwen/Qwen3-0.6B "Hello" -d hip:0 --profile --verbose 2> test-output\verbose-hip.txt
Remove-Item Env:IDRAK_PROFILE_OUT, Env:IDRAK_PROFILE_PROBES
dotnet run -c Release --no-build --project src\Idrak.Cli -- bench -d hip:0 > test-output\kernel-bench-hip.txt
```

## Settings

| Variable | Meaning |
|----------|---------|
| `IDRAK_HIP_DEFAULT=1` | Prefer a HIP device as `Device.Default` (after CUDA, before Vulkan). |
| `IDRAK_DISABLE_HIP=1` | Do not look for HIP devices. |
| `IDRAK_HIP_RUNTIME` | A HIP runtime (a file name or path) tried before the graphics driver's. |
| `IDRAK_HIP_WAIT=block` | Waits use the runtime's blocking synchronize instead of polling. |
| `IDRAK_HIP_KERNEL_CACHE` | `0` keeps no assembled kernels; a folder keeps them there (default `IDRAK_CACHE/hip/kernels`, `~/.cache/idrak/hip/kernels`). |
| `IDRAK_HIP_CODE_OBJECT` | The code object version (5 or 4) Idrak writes the GCN kernels in, when the runtime needs one. |
