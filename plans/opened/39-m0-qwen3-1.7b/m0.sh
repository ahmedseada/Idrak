#!/bin/sh
# M0 (plan 39, decision 10): the Qwen/Qwen3-1.7B baseline of today's Idrak, IDRAK_ROUTES=name (set on every run),
# measured on commit 4802653c with the CLI built from it (0.4.1-dev.2611191613, package 0.4.1-dev.2611191614).
# Modelled on plans/opened/39-rebench/rebench.sh, but one model run a call of this script:
#   m0.sh bench cuda TYPE fill|warm        fill: a fresh tuning folder tuning-cuda-TYPE (IDRAK_TUNING_CACHE); warm reads it
#   m0.sh bench cpu TYPE run1|run2         run1: a fresh CPU tuning file tuning-cpu-TYPE.tsv (IDRAK_CPU_TUNING_FILE)
#   m0.sh profile cuda TYPE prompt|decode [LABEL]  the record's profile commands, with the type's filled tuning folder
#   m0.sh profile cpu TYPE prompt [LABEL]          the record's CPU prompt-pass profile, with the type's CPU tuning file
#   m0.sh trace cuda|cpu TYPE              idrak trace --stacks of a bench run; the .nettrace kept beside its report
# CUDA bench: --prompt-tokens 512 --tokens 128 -n 3; CPU bench: --prompt-tokens 128 --tokens 32 -n 2 (the re-bench's).
# Every run's full log is kept; one summary line a bench run in results.txt (with the run's exit code).
# Profiles: a profile's first run on a bench-filled folder measures the shapes the bench never ran (the head over every
# prompt row) inside its timed pass (profile-cuda-bf16-prompt-first.txt: head 847.7 ms against a 3.97 ms kernel), so each
# profile runs twice, as the record did ("run twice on a folder and the second read"): LABEL prompt-first / decode-first,
# then prompt / decode, the one read. Around each profile and trace the tuning files are copied and diffed (lines added
# or changed), so what each run measured is on file.
set -u
cd D:/Projects/idrak || exit 1
model=Qwen/Qwen3-1.7B
out=D:/Projects/idrak/test-output/adad/m0-qwen3-1.7b
results=$out/results.txt
export IDRAK_ROUTES=name
kind=$1; dev=$2; w=$3; pass=${4:-}; label=${5:-$pass}
line() {
  awk -v d="$1" -v w="$2" -v p="$3" -v x="$4" '$1=="prompt"&&$3=="tokens/s"{pr=$2} $1=="generation"&&$3=="tokens/s"{g=$2} $1=="memory"&&$2=="peak"{mem=$3} $1=="weights"&&$2=="memory"{wm=$3} END{printf "%-5s %-8s name  %-6s prompt %8s  generation %7s  weights %6s MiB  peak %6s MiB  exit %s\n", d, w, p, pr, g, wm, mem, x}' "$5"
}
tfile() { if [ "$dev" = cuda ]; then ls "$out"/tuning-cuda-"$w"/*.txt 2>/dev/null | head -1; else echo "$out/tuning-cpu-$w.tsv"; fi; }
snap() { f=$(tfile); if [ -f "$f" ]; then cp "$f" "$out/.snap"; else : > "$out/.snap"; fi; }
sums() { f=$(tfile); echo "tuning file: $f"; if [ -f "$f" ]; then echo "tuning lines added or changed by this run:"; diff "$out/.snap" "$f" | grep '^[<>]' || echo "  (none)"; fi; rm -f "$out/.snap"; }
echo "start $(date '+%F %T')  $kind $dev $w $pass  IDRAK_ROUTES=$IDRAK_ROUTES"
case "$kind" in
bench)
  log=$out/$dev-$w-$pass.txt
  if [ "$dev" = cuda ]; then
    tune=$out/tuning-cuda-$w
    [ "$pass" = fill ] && rm -rf "$tune"
    snap
    IDRAK_TUNING_CACHE=$tune idrak bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 3 > "$log" 2>&1
  else
    cpufile=$out/tuning-cpu-$w.tsv
    [ "$pass" = run1 ] && rm -f "$cpufile"
    snap
    IDRAK_CPU_TUNING_FILE=$cpufile idrak bench $model -w "$w" -d cpu --prompt-tokens 128 --tokens 32 -n 2 > "$log" 2>&1
  fi
  code=$?
  cat "$log"
  line "$dev" "$w" "$pass" "$code" "$log" | tee -a "$results"
  sums
  ;;
profile)
  log=$out/profile-$dev-$w-$label.txt
  snap
  if [ "$dev" = cuda ] && [ "$pass" = prompt ]; then
    IDRAK_TUNING_CACHE=$out/tuning-cuda-$w idrak profile $model -w "$w" -d cuda --pass prompt --prompt-tokens 512 --top 30 > "$log" 2>&1
  elif [ "$dev" = cuda ]; then
    IDRAK_TUNING_CACHE=$out/tuning-cuda-$w idrak profile $model -w "$w" -d cuda --top 25 > "$log" 2>&1
  else
    IDRAK_CPU_TUNING_FILE=$out/tuning-cpu-$w.tsv idrak profile $model -w "$w" -d cpu --pass prompt --prompt-tokens 128 --top 20 > "$log" 2>&1
  fi
  code=$?
  cat "$log"
  echo "exit $code"
  sums
  ;;
trace)
  log=$out/trace-$dev-$w.txt
  nt=$out/trace-$dev-$w.nettrace
  snap
  if [ "$dev" = cuda ]; then
    IDRAK_TUNING_CACHE=$out/tuning-cuda-$w idrak trace --stacks -o "$nt" --top 20 -- bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 1 > "$log" 2>&1
  else
    IDRAK_CPU_TUNING_FILE=$out/tuning-cpu-$w.tsv idrak trace --stacks -o "$nt" --top 20 -- bench $model -w "$w" -d cpu --prompt-tokens 128 --tokens 8 -n 1 > "$log" 2>&1
  fi
  code=$?
  cat "$log"
  echo "exit $code"
  sums
  ;;
*)
  echo "usage: m0.sh bench|profile|trace cuda|cpu TYPE [fill|warm|run1|run2|prompt|decode]"; exit 2
  ;;
esac
echo "end $(date '+%F %T')"
