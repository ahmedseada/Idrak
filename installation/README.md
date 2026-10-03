# Installation

How Idrak was installed, built and run on each machine it was tested on, step by step, from a clean device to the
tests and an interactive chat with a language model. The results of these runs are in the main README ("Tested on
architectures").

| Guide | Machine | Backends |
|-------|---------|----------|
| [Windows](windows.md) | Windows 10/11 desktops and laptops (NVIDIA, Intel and AMD GPUs) | CPU, CUDA, Vulkan |
| [Linux and WSL2](linux.md) | Ubuntu on x64, Ubuntu under WSL2 | CPU, CUDA, Vulkan (lavapipe under WSL2) |
| [Android phone](android-termux.md) | Snapdragon phone, Termux + Ubuntu (proot), Mesa Turnip | CPU (ARM64 NEON), Vulkan (Adreno) |
| [HIP](hip.md) (untested on real hardware) | Linux with ROCm, Windows with the HIP SDK, an AMD GPU | HIP |

Every guide ends with the same three checks:

1. `--list-devices`: which devices Idrak sees and which a plain run uses.
2. The test suite on each device (`IDRAK_DEVICES=...`): every test should pass.
3. The chat sample with a small model (Qwen3-0.6B, int8 weights), on the CPU and on the GPU.

## The idrak tool

The command-line tool ([src/Idrak.Cli](../src/Idrak.Cli/README.md)) does these checks in a few commands, from a
checkout (`dotnet run -c Release --project src/Idrak.Cli -- doctor`) or installed (`dotnet tool install -g Idrak.Cli`):

```bash
idrak doctor                         # the .NET runtime, each backend (driver libraries, Vulkan driver files), each
                                     # device's limits, the caches, disk space and the environment, with the fix for each problem
idrak devices                        # what --list-devices shows: every device and whether a plain run uses it
idrak test -d cpu                    # the test suite of the checkout, on one device (or several: -d cpu,vulkan:0)
idrak chat Qwen/Qwen3-0.6B -w int8 -d vulkan:0
idrak report --tests --readme        # the README's tested-on rows for this machine
```

On the phone, `idrak setup android` adds the settings the guide sets by hand (the Turnip driver file in
`VK_ICD_FILENAMES`, `DOTNET_GCHeapHardLimit`) to `~/.bashrc` once each (`--yes`; without it, it prints them) and then
runs `idrak doctor --android`, which also checks the GPU device node.

## Requirements common to all machines

- .NET 10 SDK (`dotnet --version` prints 10.x).
- git.
- About 3 GB of free disk: the SDK, the build output and the model download (Qwen3-0.6B is 1.4 GB).
- Network access to github.com, huggingface.co and the .NET package feed for the first build and the model download.

## Environment variables used in these guides

| Variable | Meaning |
|----------|---------|
| `IDRAK_DEVICES` | Devices the test runner uses, comma-separated (`cpu`, `cuda:0`, `vulkan:0`, ...). |
| `IDRAK_FILTER` | Run only tests whose names contain this text. |
| `IDRAK_VULKAN_DEFAULT=1` | Prefer a Vulkan device as `Device.Default` when no CUDA device is present. |
| `IDRAK_HIP_DEFAULT=1` | Prefer a HIP device as `Device.Default` (after CUDA, before Vulkan); see [hip.md](hip.md) for the other HIP settings. |
| `IDRAK_VULKAN_WIDTH` | Force the Vulkan kernels' workgroup width instead of the measured one (diagnostics only). |
| `IDRAK_AUTOTUNE=0` | Use the formulas only, no on-device measuring (diagnostics only). |
| `VK_ICD_FILENAMES` | Which Vulkan driver the loader uses (needed on the phone for Turnip). |
| `DOTNET_GCHeapHardLimit` | Caps the .NET heap reservation; needed on Android, whose address space is too small for the default. |
