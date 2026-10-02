# Linux and WSL2

Used on Ubuntu 26.04 under WSL2 on the RTX 5070 Ti desktop, and in the development container (x64, lavapipe).

## 1. GPU drivers

- Native Linux with NVIDIA: the distribution's NVIDIA driver package (it provides CUDA and Vulkan).
- WSL2: install the NVIDIA driver on Windows only; WSL2 receives CUDA from it. WSL2 offers Vulkan only through
  lavapipe (Mesa's CPU implementation), which is still useful to run the Vulkan kernels.

```bash
nvidia-smi                                                  # shows the RTX card
sudo apt-get install -y vulkan-tools mesa-vulkan-drivers    # Vulkan loader, lavapipe and vulkaninfo
vulkaninfo --summary
```

## 2. .NET 10 SDK and git

```bash
sudo apt-get update
sudo apt-get install -y dotnet-sdk-10.0 git
dotnet --version
```

When the package is not available, use Microsoft's install script
(`curl -sSL https://dot.net/v1/dotnet-install.sh | bash -s -- --channel 10.0`, then add `~/.dotnet` to `PATH`).

## 3. Get the code

Under WSL2 clone into the Linux file system (`~`), not `/mnt/c` or `/mnt/d`: builds there are much faster.

```bash
git clone https://github.com/ahmedseada/Idrak.git ~/Idrak
cd ~/Idrak
git checkout architecture
```

## 4. Devices and tests

```bash
dotnet run -c Release --project tests/Idrak.Tests -- --list-devices
dotnet run -c Release --project tests/Idrak.Tests                       # every device marked "yes"
IDRAK_DEVICES=vulkan:0 dotnet run -c Release --project tests/Idrak.Tests # lavapipe or a real GPU
dotnet run -c Release --project tests/Idrak.Tests -- --bench-gemv
dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan
```

## 5. Chat with a model

```bash
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8 --vulkan
```
