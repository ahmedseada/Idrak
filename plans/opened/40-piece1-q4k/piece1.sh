#!/usr/bin/env bash
# Adad piece 1 (q4_k's prompt product, control, A and B): one cycle's runs, one at a time.
#   piece1.sh bench N      the kernel bench, every shape and candidate        -> cycleN/kernels.txt, choices.txt, measured.txt, compiled.txt
#   piece1.sh profile N    Nsight Compute over each lane's stored variant once a shape (adad profile, choices of cycleN)
#                          -> cycleN/ncu.ncu-rep, ncu-details.csv, ncu-launches.txt
#   piece1.sh trace N      dotnet-trace of the bench's host process (both models, m = 512, every product), read by idrak trace
#                          -> cycleN/trace.nettrace, trace.txt, trace-focus-*.txt
set -u
top=D:/Projects/idrak/test-output/adad/piece1-q4k
wt=D:/Projects/idrak/test-output/worktrees/adad
adad="$wt/src/Idrak.Adad.Host/bin/Release/net10.0/adad.dll"
ncu="/c/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/ncu.bat"
what=$1; n=$2; out=$top/${3:-cycle$n}
mkdir -p "$out"
case $what in
  bench)
    dotnet "$adad" kernels --out "$out" > "$out/bench-stdout.txt" 2> "$out/bench-stderr.txt" < /dev/null ;;
  profile)
    # The kernel profile: Nsight Systems' CUDA trace of each lane's stored variant, 3 runs a shape (Nsight Compute's
    # counters need a permission this machine does not give: ERR_NVGPUCTRPERM), and each kernel's SASS from the
    # toolkit's ptxas (sm_120) and cuobjdump: registers, spills, the main loop's instruction mix.
    nsys="/c/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe"
    "$nsys" profile -t cuda --force-overwrite true -o "$out/nsys" dotnet "$adad" profile --choices "$out/choices.txt" --repeat 3       > "$out/nsys-stdout.txt" 2> "$out/nsys-stderr.txt" < /dev/null
    # nsys does not pass the program's output through: the launch lines (the trace's order) from the same run, untraced.
    dotnet "$adad" profile --choices "$out/choices.txt" --repeat 3 > "$out/nsys-launches.txt" 2>&1 < /dev/null
    "$nsys" stats --report cuda_gpu_trace --format csv --force-export true --output "$out/nsys" "$out/nsys.nsys-rep" > "$out/nsys-stats.txt" 2>&1 < /dev/null
    rm -rf "$out/ptx"; dotnet "$adad" ptx --out "$out/ptx" > "$out/ptx-written.txt" 2>&1 < /dev/null
    for f in "$out"/ptx/*.ptx; do
      b=${f%.ptx}
      ptxas -arch=sm_120 -v -o "$b.cubin" "$f" > "$b.ptxas.txt" 2>&1 < /dev/null
      cuobjdump -sass "$b.cubin" > "$b.sass" 2>&1 < /dev/null
    done
    python "$top/sass.py" "$out/ptx" "$out/sass-mix.txt" > /dev/null
    python "$top/kernels-trace.py" "$out" > "$out/kernel-trace.txt" ;;
  trace)
    dotnet-trace collect --providers Microsoft-DotNETCore-SampleProfiler,Microsoft-Windows-DotNETRuntime:0x1:5 -o "$out/trace.nettrace" \
      -- dotnet "$adad" kernels --m 512 --samples 5 --out "$out/trace-run" > "$out/trace-collect.txt" 2>&1 < /dev/null
    idrak trace --read "$out/trace.nettrace" --top 30 > "$out/trace.txt" 2>&1 < /dev/null
    idrak trace --read "$out/trace.nettrace" --timeline --by-thread > "$out/trace-timeline.txt" 2>&1 < /dev/null
    for f in SuperLane.Run CudaDevice.Launch CudaDevice.LoadModule Tuner.Measure; do
      idrak trace --read "$out/trace.nettrace" --focus "$f" --top 20 > "$out/trace-focus-${f//./-}.txt" 2>&1 < /dev/null
    done ;;
esac
echo "$what $n done (exit $?)"
