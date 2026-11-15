#!/bin/sh
# The proof re-benched on committed code (cf77ceeb, CLI 0.4.1-dev.2611151417/1418), each type measured on its own:
# CUDA: a fresh tuning folder per type shared by both routes (IDRAK_TUNING_CACHE), runs in the order adad fill, name fill,
# adad warm, name warm; CPU: a fresh CPU tuning file per type (IDRAK_CPU_TUNING_FILE; the CPU's lane choices are measured
# anew in every process), two runs per route in the order adad, name, adad, name. Every run's full log is kept (its
# "measured lane:" lines are the lanes it chose); one summary line a run in results.txt.
cd D:/Projects/idrak || exit 1
out=test-output/rebench
results=$out/results.txt
: > "$results"
line() {
  awk -v d="$1" -v w="$2" -v r="$3" -v p="$4" '$1=="prompt"&&$3=="tokens/s"{pr=$2} $1=="generation"&&$3=="tokens/s"{g=$2} $1=="memory"&&$2=="peak"{mem=$3} $1=="weights"&&$2=="memory"{wm=$3} END{printf "%-5s %-8s %-5s %-6s prompt %8s  generation %7s  weights %6s MiB  peak %6s MiB\n", d, w, r, p, pr, g, wm, mem}' "$5"
}
for w in bf16 f16 int8 q8_0 q4_0 iq4_nl q4_k q6_k iq2_xxs iq3_xxs; do
  tune=$PWD/$out/tuning-cuda-$w
  rm -rf "$tune"
  for pass in fill warm; do
    for r in adad name; do
      log=$out/cuda-$w-$r-$pass.txt
      IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$tune idrak bench Qwen/Qwen3-0.6B -w $w -d cuda --prompt-tokens 512 --tokens 128 -n 3 > "$log" 2>&1
      line cuda $w $r $pass "$log" | tee -a "$results"
    done
  done
done
for w in bf16 int8 q8_0 q4_0 q4_k; do
  cpufile=$PWD/$out/tuning-cpu-$w.tsv
  rm -f "$cpufile"
  for pass in run1 run2; do
    for r in adad name; do
      log=$out/cpu-$w-$r-$pass.txt
      IDRAK_ROUTES=$r IDRAK_CPU_TUNING_FILE=$cpufile idrak bench Qwen/Qwen3-0.6B -w $w -d cpu --prompt-tokens 128 --tokens 32 -n 2 > "$log" 2>&1
      line cpu $w $r $pass "$log" | tee -a "$results"
    done
  done
done
echo "rebench done" | tee -a "$results"
