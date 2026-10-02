# HIP (ROCm on Linux, the HIP SDK on Windows)

**Untested on real hardware.** The HIP backend (`hip:0`, `hip:1`, …; [plan 8](../plans/8-hip.md)) has not run on an AMD
GPU yet: these are the steps a first run should follow, and what to send back. On a machine without HIP nothing
changes: the device list shows `hip:-` with the reason, and every other backend works as before.

What the backend needs at run time (nothing at build time):

| | Linux | Windows |
|---|---|---|
| Runtime (memory, copies, launches) | `libamdhip64.so` from ROCm | `amdhip64_6.dll` (or `amdhip64_7.dll`, `amdhip64.dll`) from the HIP SDK; the graphics driver also installs one |
| Compiler (the kernels) | `libhiprtc.so` from ROCm | `hiprtcMMmm.dll` (for example `hiprtc0602.dll`) from the HIP SDK |
| Where it is looked for | the loader's search path, then `$ROCM_PATH/lib`, then `/opt/rocm/lib` | the DLL search path, then `%HIP_PATH%\bin` |

Without the compiler, HIP devices still work: memory and copies run on the GPU and every other operation on the CPU
(the HIP device test prints why the kernels are unavailable).

## 1. Install

### Linux (ROCm)

Install ROCm for your distribution and GPU from AMD's ROCm installation guide (the `amdgpu-install` package with the
`rocm` use case, or the distribution's ROCm packages), then give your user access to the GPU and log in again:

```bash
sudo usermod -aG render,video "$USER"
rocminfo | grep -E "Name:|Compute Unit|Wavefront"          # the GPU, its gfx target, compute units, wavefront size
ls /opt/rocm/lib/libamdhip64.so* /opt/rocm/lib/libhiprtc.so*
```

ROCm supports a list of GPUs per release; a GPU outside it reports no devices.

### Windows (HIP SDK)

Install the AMD HIP SDK for Windows (it sets `HIP_PATH`), with a current Adrenalin or PRO driver, then open a new
PowerShell:

```powershell
$env:HIP_PATH
Get-ChildItem "$env:HIP_PATH\bin\amdhip64*.dll", "$env:HIP_PATH\bin\hiprtc*.dll"
```

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
IDRAK_DEVICES=hip:0 IDRAK_FILTER=hip dotnet run -c Release --project tests/Idrak.Tests   # the HIP tests first
IDRAK_DEVICES=hip:0 dotnet run -c Release --project tests/Idrak.Tests > hip0.txt         # the whole list on the GPU
IDRAK_DEVICES=hip:0 IDRAK_HIP_KERNELS=0 dotnet run -c Release --project tests/Idrak.Tests > hip0-nokernels.txt
```

Windows (PowerShell):

```powershell
$env:IDRAK_DEVICES="hip:0"; $env:IDRAK_FILTER="hip"; dotnet run -c Release --project tests/Idrak.Tests; Remove-Item Env:IDRAK_FILTER
dotnet run -c Release --project tests/Idrak.Tests > hip0.txt
$env:IDRAK_HIP_KERNELS="0"; dotnet run -c Release --project tests/Idrak.Tests > hip0-nokernels.txt
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_HIP_KERNELS
```

The line "hip:0: kernels compiled, N threads per block" (or "cached" on later runs, or "unavailable (reason)") in the
HIP device test says whether the kernels loaded. The second run without kernels separates a kernel fault from a
runtime fault. Send back the device list, both result files and, on a failure, the run again with `IDRAK_TRACE=1`.

## 4. Chat with a model

```bash
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8 --device hip:0
```

Most operations still run on the CPU through the host fallback, so expect this to be slower than the CPU alone; it
shows that a model runs end to end on the device.

## Settings

| Variable | Meaning |
|----------|---------|
| `IDRAK_HIP_DEFAULT=1` | Prefer a HIP device as `Device.Default` (after CUDA, before Vulkan). |
| `IDRAK_DISABLE_HIP=1` | Do not look for HIP devices. |
| `IDRAK_HIP_KERNELS=0` | Run no HIP kernels: every operation but memory and copies on the host fallback (diagnostics). |
| `IDRAK_HIP_KERNEL_CACHE` | `0` keeps no compiled kernels; a folder keeps them there (default `IDRAK_CACHE/hip/kernels`, `~/.cache/idrak/hip/kernels`). |
