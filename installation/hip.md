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

## Settings

| Variable | Meaning |
|----------|---------|
| `IDRAK_HIP_DEFAULT=1` | Prefer a HIP device as `Device.Default` (after CUDA, before Vulkan). |
| `IDRAK_DISABLE_HIP=1` | Do not look for HIP devices. |
| `IDRAK_HIP_RUNTIME` | A HIP runtime (a file name or path) tried before the graphics driver's. |
| `IDRAK_HIP_WAIT=block` | Waits use the runtime's blocking synchronize instead of polling. |
| `IDRAK_HIP_KERNEL_CACHE` | `0` keeps no assembled kernels; a folder keeps them there (default `IDRAK_CACHE/hip/kernels`, `~/.cache/idrak/hip/kernels`). |
| `IDRAK_HIP_CODE_OBJECT` | The code object version (5 or 4) Idrak writes the GCN kernels in, when the runtime needs one. |
