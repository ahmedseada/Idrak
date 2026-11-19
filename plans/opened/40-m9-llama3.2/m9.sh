#!/bin/sh
# M9: another family, meta-llama/Llama-3.2-1B-Instruct and -3B-Instruct, with M6's protocol (m6-proof-1.7b/m6.sh), measuring
# only. The M6 build's CLI 0.4.1-dev.2611191810 (commit de1f8a96; no code changed since). One model run a call; SIZE is
# 1b or 3b, the outputs in m9-llama3.2/SIZE/:
#   m9.sh SIZE cuda TYPE ROUTE fill|warm   a fresh tuning folder tuning-cuda-TYPE shared by both routes, removed before the
#                                          type's first run (adad fill); runs adad fill, name fill, adad warm, name warm:
#                                          idrak bench MODEL -w TYPE -d cuda --prompt-tokens 512 --tokens 128 -n 3
#   m9.sh SIZE cpu TYPE ROUTE run1|run2    a fresh CPU tuning file tuning-cpu-TYPE.tsv, removed before the type's first run
#                                          (adad run1); runs adad, name, adad, name: --prompt-tokens 128 --tokens 32 -n 2
#   m9.sh SIZE profile cuda|cpu TYPE ROUTE [LABEL]  the prompt pass on the type's filled folder or file (CUDA
#                                          --prompt-tokens 512 --top 30; CPU --prompt-tokens 128 --top 20)
#   m9.sh SIZE dprofile cuda TYPE ROUTE [LABEL]     a decoding step's profile (--top 25; the median of 5 steps)
#   m9.sh SIZE trace cuda|cpu TYPE ROUTE   idrak trace --stacks of a bench run (CUDA --prompt-tokens 512 --tokens 128 -n 1, CPU
#                                          --prompt-tokens 128 --tokens 8 -n 1) on the filled folder or file
#   m9.sh SIZE complete TYPE ROUTE         the greedy continuation (--temperature 0 --max-tokens 48) of ../prompt.txt (M7's
#                                          generic text) on CUDA, on the type's filled folder -> complete-TYPE-ROUTE.txt
# Every run's full log is kept (its "measured lane:" lines are the lanes it chose); one summary line a bench run in
# SIZE/results.txt. The route is set on every run (IDRAK_ROUTES). Each profile, trace and continuation copies its tuning file
# before and diffs it after into SIZE/tuning-added.txt.
set -u
cd D:/Projects/idrak || exit 1
size=$1; shift
case "$size" in
1b) model=meta-llama/Llama-3.2-1B-Instruct ;;
3b) model=meta-llama/Llama-3.2-3B-Instruct ;;
*) echo "size: 1b or 3b"; exit 2 ;;
esac
top=D:/Projects/idrak/test-output/adad/m9-llama3.2
out=$top/$size
results=$out/results.txt
kind=$1
tfile() { if [ "$1" = cuda ]; then ls "$out/tuning-cuda-$2"/*.txt 2>/dev/null | head -1; else echo "$out/tuning-cpu-$2.tsv"; fi; }
snap() { f=$(tfile "$1" "$2"); if [ -f "$f" ]; then cp "$f" "$out/.snap"; else : > "$out/.snap"; fi; }
sums() {
  f=$(tfile "$1" "$2")
  { echo "== $3 ($(date '+%T')): lines added or changed in $(basename "$(dirname "$f")")/$(basename "$f")"; diff "$out/.snap" "$f" | grep '^[<>]' || echo "  (none)"; } | tee -a "$out/tuning-added.txt"
  rm -f "$out/.snap"
}
line() {
  awk -v d="$1" -v w="$2" -v r="$3" -v p="$4" -v x="$5" '$1=="prompt"&&$3=="tokens/s"{pr=$2} $1=="generation"&&$3=="tokens/s"{g=$2} $1=="memory"&&$2=="peak"{mem=$3} $1=="weights"&&$2=="memory"{wm=$3} END{printf "%-5s %-8s %-5s %-6s prompt %8s  generation %7s  weights %6s MiB  peak %6s MiB  exit %s\n", d, w, r, p, pr, g, wm, mem, x}' "$6"
}
case "$kind" in
cuda)
  w=$2; r=$3; pass=$4
  tune=$out/tuning-cuda-$w; log=$out/cuda-$w-$r-$pass.txt
  [ "$r" = adad ] && [ "$pass" = fill ] && rm -rf "$tune"
  echo "start $(date '+%F %T')  $size cuda $w $r $pass"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$tune idrak bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 3 > "$log" 2>&1
  code=$?
  cat "$log"; line cuda "$w" "$r" "$pass" "$code" "$log" | tee -a "$results"
  ;;
cpu)
  w=$2; r=$3; pass=$4
  cpufile=$out/tuning-cpu-$w.tsv; log=$out/cpu-$w-$r-$pass.txt
  [ "$r" = adad ] && [ "$pass" = run1 ] && rm -f "$cpufile"
  echo "start $(date '+%F %T')  $size cpu $w $r $pass"
  IDRAK_ROUTES=$r IDRAK_CPU_TUNING_FILE=$cpufile idrak bench $model -w "$w" -d cpu --prompt-tokens 128 --tokens 32 -n 2 > "$log" 2>&1
  code=$?
  cat "$log"; line cpu "$w" "$r" "$pass" "$code" "$log" | tee -a "$results"
  ;;
profile|dprofile)
  dev=$2; w=$3; r=$4; label=${5:-}
  kindname=prompt; [ "$kind" = dprofile ] && kindname=decode
  log=$out/profile-$dev-$w-$r-$kindname${label:+-$label}.txt
  echo "start $(date '+%F %T')  $size $kind $dev $w $r $label"
  snap "$dev" "$w"
  if [ "$dev" = cuda ] && [ "$kind" = profile ]; then
    IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-cuda-$w idrak profile $model -w "$w" -d cuda --pass prompt --prompt-tokens 512 --top 30 > "$log" 2>&1
  elif [ "$dev" = cuda ]; then
    IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-cuda-$w idrak profile $model -w "$w" -d cuda --top 25 > "$log" 2>&1
  else
    IDRAK_ROUTES=$r IDRAK_CPU_TUNING_FILE=$out/tuning-cpu-$w.tsv idrak profile $model -w "$w" -d cpu --pass prompt --prompt-tokens 128 --top 20 > "$log" 2>&1
  fi
  echo "exit $?" >> "$log"; cat "$log"
  sums "$dev" "$w" "$(basename "$log" .txt)"
  ;;
trace)
  dev=$2; w=$3; r=$4
  log=$out/trace-$dev-$w-$r.txt; nt=$out/trace-$dev-$w-$r.nettrace
  echo "start $(date '+%F %T')  $size trace $dev $w $r"
  snap "$dev" "$w"
  if [ "$dev" = cuda ]; then
    IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-cuda-$w idrak trace --stacks -o "$nt" --top 20 -- bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 1 > "$log" 2>&1
  else
    IDRAK_ROUTES=$r IDRAK_CPU_TUNING_FILE=$out/tuning-cpu-$w.tsv idrak trace --stacks -o "$nt" --top 20 -- bench $model -w "$w" -d cpu --prompt-tokens 128 --tokens 8 -n 1 > "$log" 2>&1
  fi
  echo "exit $?" >> "$log"; cat "$log"
  sums "$dev" "$w" "$(basename "$log" .txt)"
  ;;
complete)
  w=$2; r=$3; log=$out/complete-$w-$r.txt
  echo "start $(date '+%F %T')  $size complete $w $r"
  snap cuda "$w"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-cuda-$w idrak complete $model -w "$w" -d cuda -i "$top/prompt.txt" --temperature 0 --max-tokens 48 > "$log" 2> "$out/complete-$w-$r.err"
  echo "exit $?" >> "$out/complete-$w-$r.err"; cat "$log"; echo; tail -n 3 "$out/complete-$w-$r.err"
  sums cuda "$w" "$(basename "$log" .txt)"
  ;;
*)
  echo "usage: m9.sh 1b|3b cuda|cpu TYPE ROUTE PASS | profile|dprofile|trace DEVICE TYPE ROUTE [LABEL] | complete TYPE ROUTE"; exit 2
  ;;
esac
echo "end $(date '+%F %T')"
