# macOS (Apple silicon)

Tested on 2026-10-03 on a MacBook Pro with an Apple M4 Max (14 cores: 10 performance, 4 efficiency; 36 GB; macOS
27.2) with .NET 10.0.12. Results: README, "Tested on architectures".

## What runs

- The CPU backend (ARM64 NEON with DotProd and RDM; the tiled product kernel NEON 8x8): every test, the `idrak` tool,
  and language models such as Qwen3-0.6B.
- No GPU backend: CUDA, Vulkan and HIP are not used on macOS (`idrak doctor` lists them as information). Apple GPUs
  need a Metal backend, which is planned.

## One script: install, test, measure, collect

Paste into Terminal. No administrator password is needed (.NET installs into `~/.dotnet`). It takes about 30 to 60
minutes and downloads about 2.5 GB (the .NET SDK, the source, a 0.6B model); every step writes a log, and everything
ends up in `~/idrak-mac-results.zip`.

```bash
cd ~ && R=~/idrak-mac-results && rm -rf "$R" && mkdir -p "$R" && caffeinate -dimsu -w $$ &
R=~/idrak-mac-results; step(){ n="$1"; shift; echo "=== $(date '+%H:%M:%S') $n: $*" | tee -a "$R/00-steps.log"; "$@" > "$R/$n.log" 2>&1; echo "    exit $?" | tee -a "$R/00-steps.log"; }
step 01-system bash -c 'sw_vers; uname -m; sysctl -n machdep.cpu.brand_string; sysctl hw.memsize hw.ncpu hw.perflevel0.physicalcpu hw.perflevel1.physicalcpu hw.l1dcachesize hw.l2cachesize 2>/dev/null'
step 02-dotnet-install bash -c 'curl -sSL https://dot.net/v1/dotnet-install.sh -o /tmp/dotnet-install.sh && bash /tmp/dotnet-install.sh --channel 10.0 --install-dir ~/.dotnet'
export DOTNET_ROOT=~/.dotnet PATH=~/.dotnet:~/.dotnet/tools:$PATH DOTNET_CLI_TELEMETRY_OPTOUT=1 DOTNET_NOLOGO=1
step 03-dotnet-info dotnet --info
step 04-source bash -c 'rm -rf ~/Idrak-main ~/idrak-src.zip && curl -sSL -o ~/idrak-src.zip https://github.com/ahmedseada/Idrak/archive/refs/heads/main.zip && unzip -q ~/idrak-src.zip -d ~'
cd ~/Idrak-main
step 05-build dotnet build -c Release
step 06-tests-cpu env IDRAK_DEVICES=cpu dotnet run -c Release --no-build --project tests/Idrak.Tests
step 07-bench-cpu dotnet run -c Release --no-build --project tests/Idrak.Tests -- --bench-cpu
step 08-bench-text dotnet run -c Release --no-build --project tests/Idrak.Tests -- --bench-text
step 09-tool-install bash -c 'rm -rf pkg ~/idrak-tool && dotnet pack src/Idrak.Cli -c Release -o pkg && dotnet tool install Idrak.Cli --tool-path ~/idrak-tool --add-source pkg --prerelease'
I=~/idrak-tool/idrak
step 10-idrak-version $I version
step 11-idrak-doctor $I doctor
step 12-idrak-devices $I devices
step 13-idrak-bench-kernels $I bench --kernels matmul,gemv -d cpu
step 14-idrak-bench-model $I bench Qwen/Qwen3-0.6B -d cpu
step 15-idrak-bench-int8 $I bench Qwen/Qwen3-0.6B -d cpu -w int8
step 16-idrak-run $I run Qwen/Qwen3-0.6B "Explain in three sentences why the sky is blue." -d cpu
cd ~ && rm -f idrak-mac-results.zip && zip -qr idrak-mac-results.zip idrak-mac-results && echo "DONE: ~/idrak-mac-results.zip"
```

With git instead of the zip download (the Command Line Tools install git on first use):

```bash
git clone https://github.com/ahmedseada/Idrak.git ~/Idrak && cd ~/Idrak
```

## The web chat page

```bash
~/idrak-tool/idrak ui Qwen/Qwen3-0.6B -d cpu -w int8      # opens http://127.0.0.1:7317/ui
```

## Results on the M4 Max

| Measurement | Value |
|---|---|
| Tests (CPU) | 425 of 426; the failure was a test comparing a temp path with the working directory's (`/var` is a link to `/private/var` on macOS), fixed in the test |
| Float32 product 1024³ / 2048³ | 477 / 448 GFLOP/s (mixed bfloat16: 456 / 511) |
| Qwen3-0.6B, stored weights | prompt 378 tokens/s, generation 23.5 tokens/s |
| Qwen3-0.6B, int8 weights | prompt 356 tokens/s, generation 37.8 tokens/s, 887 MiB |

Found on this machine and fixed: float32 products of 5 to 7 rows ran 4 to 7 times slower than 4 or 8 rows (they now
split the columns over the threads, as 1 to 4 rows did), and `idrak doctor` warned about backends macOS does not use.
