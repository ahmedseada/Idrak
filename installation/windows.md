# Windows

Used on an RTX 5070 Ti desktop and on two laptops (RTX 3060 with AMD Radeon graphics, RTX 5050 with Intel UHD
graphics). All commands are for PowerShell.

## 1. Drivers

- NVIDIA: a current Game Ready or Studio driver (tested with 581.29 and 610.88). It brings both CUDA and Vulkan; no
  CUDA toolkit is needed (Idrak loads the driver's own libraries).
- Intel and AMD integrated graphics: the driver from Windows Update or the vendor is enough; it includes Vulkan.

Check: `nvidia-smi` lists the NVIDIA card.

## 2. .NET 10 SDK and git

```powershell
winget install Microsoft.DotNet.SDK.10
winget install Git.Git
```

Open a new PowerShell window, then `dotnet --version` should print 10.x.

## 3. Get the code

```powershell
cd D:\Projects
git clone https://github.com/ahmedseada/Idrak.git
cd Idrak
git checkout architecture
```

## 4. Devices

```powershell
dotnet run -c Release --project tests/Idrak.Tests -- --list-devices
```

Each GPU appears once per backend (`cuda:0`, `vulkan:0`, `vulkan:1`, ...). A GPU that CUDA drives is tested through
Vulkan only when named.

## 5. Tests

```powershell
# every device marked "yes" in the list
dotnet run -c Release --project tests/Idrak.Tests

# one device at a time, results saved to files
mkdir -Force results | Out-Null
foreach ($d in "cpu","cuda:0","vulkan:0","vulkan:1") {
  $env:IDRAK_DEVICES=$d; $n=$d -replace ':',''
  dotnet run -c Release --project tests/Idrak.Tests *> "results\$n-tests.txt"
}
Remove-Item Env:IDRAK_DEVICES

# benchmarks
dotnet run -c Release --project tests/Idrak.Tests -- --bench-gemv   *> results\bench-gemv.txt
dotnet run -c Release --project tests/Idrak.Tests -- --bench-vulkan *> results\bench-vulkan.txt
```

The last line of each tests file reads `N passed, 0 failed`.

The first benchmark run on a new build measures kernel choices and stores them (in
`%USERPROFILE%\.cache\idrak`); timings settle from the second run.

## 6. Chat with a model

```powershell
# the default device (CUDA when present)
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8

# a named device
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8 --vulkan
dotnet run -c Release --project samples/Idrak.Samples.Chat -- chat Qwen/Qwen3-0.6B --int8 --device vulkan:1
```

The model is downloaded once from Hugging Face. The first line after loading names the device (`on cuda:0`,
`on vulkan:0`). Type a message and press Enter; an empty line quits.
