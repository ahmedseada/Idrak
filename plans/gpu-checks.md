# GPU checks for plans 12 and 13

Everything in plans 12 and 13 passes the targeted CPU tests (here and on the owner's Windows machine). These are the
GPU runs nobody has done yet: every kernel written for CUDA and Vulkan in these plans was compiled and, where a host
tool could, assembled or validated, but never run on a GPU. Run from `D:\Projects\Idrak` in PowerShell.

## Correctness, every device

```powershell
cd D:\Projects\Idrak
git pull origin abstraction
dotnet build -c Release tests/Idrak.Tests
foreach ($d in "cuda", "vulkan") {
    $env:IDRAK_DEVICES = $d
    foreach ($f in "conformance kit", "attention spans", "vision tuning", "image prefill", "vision kernels", "conv", "vulkan cnn",
                   "vision layers", "vision sequence", "vision detection", "vision augment", "image famil",
                   "outside plug-in", "cli vision:", "fine-tuning", "packed", "checkpoint") {
        $env:IDRAK_FILTER = $f
        Write-Host "== $d / $f"
        dotnet run -c Release --no-build --project tests/Idrak.Tests
    }
}
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_FILTER
```

"vulkan cnn" names Vulkan tests only; on CUDA it finds none. "vision tuning graph" prints "recorded once, replayed for
two batches" on a device that records graphs.

## Speed (the measured choices should never pick the slower path)

```powershell
foreach ($d in "cuda", "vulkan") {
    $env:IDRAK_DEVICES = $d
    dotnet run -c Release --no-build --project tests/Idrak.Tests -- --bench-spans train
    dotnet run -c Release --no-build --project tests/Idrak.Tests -- --bench-conv
    $env:IDRAK_MATMUL = "bf16"; dotnet run -c Release --no-build --project tests/Idrak.Tests -- --bench-conv; Remove-Item Env:IDRAK_MATMUL
}
$env:IDRAK_DEVICES = "cuda"; $env:IDRAK_TUNE_LOG = "1"; dotnet run -c Release --no-build --project tests/Idrak.Tests -- --bench-conv
Remove-Item Env:IDRAK_DEVICES, Env:IDRAK_TUNE_LOG
```

`IDRAK_SPAN_TOKENS` changes the span benchmark's size (default 4,096); `IDRAK_ATTENTION_PATH=spans|composed` forces a
path.

## The real model (plan 12, phase 5)

The commands are in `plans/12-vision-tuning.md`, "Phase 5: proof on the real model".

## What to look for

- Any FAIL: the test names the operation and the device; `IDRAK_CUDA_DEBUG=1` names a failing CUDA kernel, and
  `IDRAK_RETRY_ON_HOST=1` shows whether the host path still passes.
- `idrak kernels -d cuda:0` (and `vulkan:0`): the convolution, pooling, resampling, CTC and span-attention operations
  listed as the device's own, not "host".
- The benchmarks: the measured choice within a few percent of the fastest column.
