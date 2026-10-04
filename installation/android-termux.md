# Android phone (Termux, Ubuntu in proot, Mesa Turnip)

How Idrak was built and run on an Android phone, on its CPU and on its Adreno GPU, ending with an interactive chat
with Qwen3-0.6B. No root access is needed.

Tested on a Snapdragon 8+ Gen 1 (SM8475) phone: Adreno 730 GPU, 8 ARM64 cores, 11 GB of memory shared by the CPU and
the GPU. Results: the CPU list 245 of 245, the Vulkan list 247 of 247 on the GPU, chat at 5-6.4 tokens/s on the CPU
and 14.4-17.4 tokens/s on the GPU.

## How the pieces fit

| Layer | What it is | Why it is needed |
|-------|------------|------------------|
| Termux | A Linux terminal app for Android | Gives a shell and a package manager without root |
| proot-distro + Ubuntu | A full Ubuntu (ARM64) running inside Termux, without root | .NET's official ARM64 Linux builds run there (glibc) |
| .NET 10 SDK | Builds and runs Idrak | Idrak is a .NET 10 library |
| Mesa Turnip | An open-source Vulkan driver for Adreno GPUs | Qualcomm's own Vulkan driver cannot be loaded from the Ubuntu side; Turnip talks to the GPU's kernel driver (KGSL) directly |

## 1. Install Termux

Install Termux from F-Droid (https://f-droid.org/packages/com.termux/) or from its GitHub releases
(https://github.com/termux/termux-app/releases). The Google Play version is outdated.

Open Termux and update it:

```bash
pkg update && pkg upgrade -y
```

Allow Termux to write to the phone's storage (to copy result files out later):

```bash
termux-setup-storage
```

Accept the Android permission prompt.

## 2. Install Ubuntu inside Termux

```bash
pkg install -y proot-distro
proot-distro install ubuntu
proot-distro login ubuntu
```

The prompt changes to `root@localhost`. Every later step runs inside Ubuntu unless it says "in Termux". To come back
later: open Termux, then `proot-distro login ubuntu`.

Tip: paste one command (or one block joined with `&&`) at a time. Lines pasted while a command is still starting can
be lost.

## 3. Base packages and time zone

Some packages ask for a time zone interactively and hang the install; set it first:

```bash
export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime
apt update && apt install -y tzdata git curl ca-certificates libicu-dev
```

## 4. .NET 10 SDK

```bash
apt install -y dotnet-sdk-10.0
dotnet --info | head -20
```

(Microsoft's `dotnet-install.sh` script also works, but on this phone its download stalled; the Ubuntu package
installed at once.)

### The heap limit (required)

.NET reserves a very large range of virtual addresses for its heap at start-up (256 GiB on a machine like this), and
Android's address space for apps is too small for that, so every `dotnet` command fails before doing anything.
Capping the heap fixes it:

```bash
echo 'export DOTNET_GCHeapHardLimit=0x100000000' >> ~/.bashrc     # 4 GiB
source ~/.bashrc
dotnet build-server shutdown
```

If `dotnet` still fails to start, use 2 GiB instead (`0x80000000`).

## 5. Get Idrak and run it on the CPU

```bash
git clone https://github.com/ahmedseada/Idrak.git ~/Idrak
cd ~/Idrak
dotnet run -c Release --project tests/Idrak.Tests -- --list-devices
IDRAK_DEVICES=cpu dotnet run -c Release --project tests/Idrak.Tests > ~/phone-tests.txt 2>&1
tail -3 ~/phone-tests.txt
```

The first build takes several minutes on a phone. The last line should read `N passed, 0 failed`. The CPU backend uses
the ARM64 NEON instructions (with DotProd and RDM when the CPU has them).

### Copying result files to the phone

The Ubuntu files live inside Termux's private storage. From inside Ubuntu, the phone's internal storage is at
`/sdcard`:

```bash
mkdir -p /sdcard/IdrakOutput && cp ~/phone-tests.txt /sdcard/IdrakOutput/
```

The file then shows in the phone's file manager under Internal storage > IdrakOutput.

## 6. Check the GPU

In Termux (not Ubuntu), the GPU's details:

```bash
pkg install -y vulkan-tools vulkan-loader-android
for p in ro.soc.manufacturer ro.soc.model ro.hardware.vulkan; do echo "$p = $(getprop $p)"; done
cat /sys/class/kgsl/kgsl-3d0/gpu_model
vulkaninfo --summary
```

This shows Qualcomm's own driver (Vulkan 1.1 on this phone). Programs in the Ubuntu side cannot use it (it is built
for Android's libraries, not glibc).

Inside Ubuntu, check that the GPU's kernel device is visible:

```bash
ls -la /dev/kgsl-3d0
```

If it is listed, Turnip can drive the GPU from Ubuntu.

## 7. Build the Turnip Vulkan driver

Ubuntu's own `mesa-vulkan-drivers` package includes Turnip, but built only for the Linux DRM interface (msm), not for
Android's KGSL interface, so it finds no GPU here. Build Turnip from Mesa's sources with KGSL support, into
`/opt/turnip` (about 20-40 minutes on the phone):

```bash
export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC
apt install -y tzdata git build-essential meson ninja-build pkg-config python3-mako python3-yaml python3-packaging \
  glslang-tools libdrm-dev libexpat1-dev libzstd-dev zlib1g-dev libelf-dev bison flex wayland-protocols libwayland-dev \
  libx11-dev libxext-dev libxrandr-dev libxshmfence-dev libxcb-randr0-dev libxcb-dri3-dev libxcb-present-dev \
  libxcb-shm0-dev vulkan-tools

git clone --depth 1 https://gitlab.freedesktop.org/mesa/mesa.git ~/mesa
cd ~/mesa
meson setup build -Dvulkan-drivers=freedreno -Dfreedreno-kmds=kgsl -Dgallium-drivers= -Dplatforms= \
  -Dglx=disabled -Degl=disabled -Dgbm=disabled -Dllvm=disabled -Dbuildtype=release -Dprefix=/opt/turnip
ninja -C build install
```

The option that matters is `-Dfreedreno-kmds=kgsl`.

Point the Vulkan loader at the new driver, now and in every later session:

```bash
export VK_ICD_FILENAMES=$(ls /opt/turnip/share/vulkan/icd.d/freedreno_icd*.json)
echo "export VK_ICD_FILENAMES=$VK_ICD_FILENAMES" >> ~/.bashrc
vulkaninfo --summary | grep -E "deviceName|driverName|apiVersion"
```

Expected: `deviceName = Turnip Adreno (TM) 7xx`, `driverName = turnip Mesa driver`, `apiVersion = 1.4.x`.
(Turnip may name the GPU by a neighbouring model; it named this Adreno 730 "725".)

## 8. Run Idrak on the GPU

```bash
cd ~/Idrak && git pull -q && source ~/.bashrc
dotnet run -c Release --project tests/Idrak.Tests -- --list-devices
IDRAK_DEVICES=vulkan:0 dotnet run -c Release --project tests/Idrak.Tests > ~/phone-gpu-tests.txt 2>&1
tail -1 ~/phone-gpu-tests.txt
dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan > ~/phone-gpu-bench.txt 2>&1
cp ~/phone-gpu-tests.txt ~/phone-gpu-bench.txt /sdcard/IdrakOutput/
```

`--list-devices` should show `vulkan:0` as the Turnip Adreno.

When the backend opens the device it measures its kernels' workgroup width (the formula from the GPU's limits gave
1024 on this Adreno, which ran far slower than narrower widths) and stores the choice in
`~/.cache/idrak/vulkan/tuning.tsv`, so the first start takes a few seconds longer. To see what it chose:

```bash
grep width3 ~/.cache/idrak/vulkan/tuning.tsv
```

(It chose 128 on this phone.)

## 9. Chat with a language model

Clear any diagnostic overrides from earlier experiments, keep the heap limit, and start the chat:

```bash
cd ~/Idrak && git pull -q && source ~/.bashrc
unset IDRAK_VULKAN_WIDTH IDRAK_VULKAN_SUBGROUPS IDRAK_AUTOTUNE IDRAK_VULKAN_STORAGE
export DOTNET_GCHeapHardLimit=0x100000000

# on the GPU
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8 --vulkan

# on the CPU, for comparison
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8 --device cpu
```

The model (1.4 GB) is downloaded once to `~/.cache/idrak/downloads`. Loading prints the device, for example
`Loaded Qwen3ForCausalLM ... in 17.6 s on vulkan:0, int8 weights`. Type a message and press Enter; each answer ends
with its speed, for example `[136 tokens, 17.4 tok/s]`. An empty line quits.

To keep a transcript and copy it out:

```bash
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8 --vulkan 2>&1 | tee ~/chat-log.txt
cp ~/chat-log.txt /sdcard/IdrakOutput/
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Every `dotnet` command fails at start with a GC or memory reservation error | The default heap reservation does not fit Android's address space | `export DOTNET_GCHeapHardLimit=0x100000000` (step 4) |
| `apt install` stops at a time-zone question | tzdata asks interactively | `export DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC` before installing (step 3) |
| `dotnet-install.sh` stays at 0 KB/s | Download stalls | `apt install -y dotnet-sdk-10.0` (step 4) |
| `vulkaninfo` in Ubuntu shows only llvmpipe or no GPU | Turnip was built without KGSL, or `VK_ICD_FILENAMES` is not set | Build with `-Dfreedreno-kmds=kgsl` and `source ~/.bashrc` (step 7) |
| The chat says `on cpu` | The sample uses the default device | Add `--vulkan` (or `--device vulkan:0`) |
| `VK_ERROR_DEVICE_LOST` during a long run | A GPU job ran long enough for the driver's watchdog to reset the GPU | Update Idrak (`git pull`): the measured width avoids the slow wide kernels; as a workaround `IDRAK_VULKAN_WIDTH=256` |
| Speeds drop over a long session | The phone heats up and lowers its clocks, more so while charging | Let it cool between runs |
