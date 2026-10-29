# HIP (AMD GPUs: the graphics driver on Windows, ROCm's runtime on Linux)

The HIP backend (`hip:0`, `hip:1`, …; [plan 8](../plans/done/8-hip.md)) needs no SDK, as CUDA needs no CUDA Toolkit:
only the HIP runtime the graphics driver installs (Windows) or ROCm's runtime library (Linux). Its kernels are Idrak's
own GCN code, written in C# and assembled by Idrak ([plan 35](../plans/opened/35-gcn.md)). Memory, uploads, downloads,
copies and fills run on the GPU, and so do the row operations (RMS norm with and without gain, layer norm fused, its
training forward and gradients, softmax, log-softmax, the scaled masked softmax, log-softmax top-k) and the element-wise
ones (the activations, add, sub, mul, max, min, affine, axpy, multiply-add, the row-vector add), the matrix products
(float32 in every transposition with beta, batches and a bias; few rows for decoding; int8, int4, bfloat16 and float16
weights; their expansions and bfloat16 packing), attention (decoding and the tiled form over float32, bfloat16 and int8
key/value caches, attention over key ranges with grouped heads and its strided form, the int8 cache's scores and
weighted values) and the shape operations (the cache writes, the decoder mask, the gathers, permutes, the adding
strided copy) on Idrak's GCN kernels; every other operation (training's other kernels, ...) still runs on the CPU
through the host fallback until its kernel lands. On a machine without HIP nothing changes: the device list shows `hip:-` with the
reason, and every other backend works as before.

### The kernels and their cache

On the first operation that needs them, Idrak assembles every GCN kernel for the target the GPU reports (its
`gcnArchName`, such as `gfx90c`) into one code object, keeps it in the kernel cache (`IDRAK_CACHE/hip/kernels`, or
`~/.cache/idrak/hip/kernels`) and loads it once; later runs read it from the cache. The cache key hashes every kernel's
text, the target, the device, the code object format and the runtime, driver and library versions: there is no compiler
version, because Idrak is the compiler, so a new library build assembles again. A cached file the runtime refuses, or
whose metadata does not read, is deleted and assembled again. The load is reported once to the telemetry
(`KernelsLoaded`: from the cache or assembled, the time, the kernels and the bytes), so `--verbose` shows the cold
assembly apart from the warm load.

The kernels are unavailable, and every operation takes the host fallback, when `IDRAK_HIP_KERNELS=0` turns them off,
when the GPU reports a target no registered GCN family handles (the reason names it: "not registered"; gfx9 is built
in, later families come with plan 35's phase G8), or when the runtime refuses the code object in every format. The
reason shows in `idrak kernels -d hip:0`, `idrak devices`, the host fallbacks' `--verbose` lines and the HIP device test.

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

The HIP device test prints "hip:0: kernels loaded" or "unavailable (reason)", and compares every operation against the
CPU: the row, element-wise, product, attention and shape operations on the GPU's kernels (none of them may take the host
fallback while the kernels are loaded), the rest through the host fallback. A run whose kernels did not load is reported as a skip, not a pass. On a
failure, send the failures file and the run again with `IDRAK_TRACE=1`.

## 4. Profiling, verbose output, benches and telemetry

`hip:0` takes the same diagnostics as the other GPUs:

- **Profiler.** The backend times its kernels (`IKernelProfiler`) as CUDA's does: while a profile runs, each GCN launch
  is timed on the GPU between two HIP events (created once and reused), by the kernel's name, or by a label with its
  floating-point operations: the row kernels as "kernel rows x cols", the attention kernels as "kernel rows x positions x
  head" (the positions the cache's capacity or the key rows: an upper bound) and the element-wise and shape ones as
  "kernel n", each with
  its floating-point operations, so the report gives their rate. `idrak profile MODEL -d hip:0` lists the kernels of one
  decoding step, and `--profile` with `IDRAK_PROFILE_PROBES=kernels` adds them to any command's report. Operations still
  on the host fallback show in its table, not the kernels'. A runtime without the event functions times nothing, and the
  telemetry says which functions it lacks.
- **Verbose.** `--verbose` (or `IDRAK_VERBOSE=1`) prints, on the error stream, the kernel set's load (from the cache or
  assembled, its time and size), the first host fallback of each operation (with the reason when the kernels are
  unavailable), and each device failure once: a code object the runtime refused
  (each format tried with its error), a launch it refused (the kernel and its grid), a profile that cannot time.
- **Telemetry.** The same events reach every hook at `TelemetryLevel.Devices` (`KernelsLoaded`, `DeviceFailed`,
  `HostFallback`), as on Vulkan.
- **Bench.** `--bench-hip [launch|copies|fill|rows|pointwise|products|attention|shapes]` in the test runner times the launch cost of `fill_f32`
  (its code object loaded, cold and again; a launch over one float, queued, with a wait, and on the device; a launch
  over a large buffer), uploads, downloads and copies on the device, fill bandwidth (`fill_f32` beside the runtime's own
  fill), the kernel set (assembled cold, then read from a cache warm), each row and element-wise kernel, the products
  (float32, few-row, packed few-row with the k splits each took, the expansions and bfloat16 packing), attention
  (decoding over each cache format, the tiled form, key ranges, the int8 cache's products) and the shape operations (its
  first call, then warm: GB/s and GFLOP/s), each first call apart from the warm ones, each section ending with the host
  fallbacks it took (none expected). The large buffer is a sixteenth of the memory the GPU has free (at most 256 MB), or
  `IDRAK_BENCH_HIP_MB`; `IDRAK_BENCH_HIP_REPEATS` sets the warm calls (default 20). Without a HIP device it prints one
  line saying why. `idrak bench -d hip:0` and `idrak report --bench -d hip:0` run their benchmarks on `hip:0` as on any
  device.

Windows (PowerShell), every output under `test-output\`:

```powershell
cd D:\Projects\Idrak
dotnet build -c Release tests\Idrak.Tests
dotnet run -c Release --no-build --project tests\Idrak.Tests -- --bench-hip > test-output\bench-hip.txt
dotnet run -c Release --no-build --project src\Idrak.Cli -- kernels -d hip:0 > test-output\kernels-hip.txt
dotnet run -c Release --no-build --project src\Idrak.Cli -- kernels dump gcn -o test-output\kernels
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
| `IDRAK_HIP_KERNELS=0` | HIP's GCN kernels stay off: every operation through the host fallback (to compare or isolate a kernel). |
| `IDRAK_HIP_KERNEL_CACHE` | `0` keeps no assembled kernel set (assembled at every start); a folder keeps it there (default `IDRAK_CACHE/hip/kernels`, `~/.cache/idrak/hip/kernels`). |
| `IDRAK_HIP_POINTWISE_BLOCKS` | Blocks of 256 an element-wise launch may take per compute unit the GPU reports (default 8); each kernel's grid-stride loop takes the rest. |
| `IDRAK_HIP_GEMV_SPLITS` | The k splits of the packed few-row products (decoding through int8, int4, bfloat16 and float16 weights): a positive count; by default about two blocks a compute unit the GPU reports, a power of two, at most min(64, k / 64). The splits' sums are added in split order by a second kernel. |
| `IDRAK_HIP_ATTENTION_SPLIT` | Key positions one split of an attention row takes (rounded up to a multiple of 256; default 512; `0` turns splits off): few rows over many keys split them over blocks (at most 64 splits, 4,096 blocks), merged in split order. |
| `IDRAK_HIP_CODE_OBJECT` | The code object version (5 or 4) Idrak writes the GCN kernels in, when the runtime needs one. |
