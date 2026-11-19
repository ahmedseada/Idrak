#!/bin/sh
# M6: the proof's adad route against name on Qwen/Qwen3-1.7B, the re-bench's protocol (plans/opened/39-rebench/rebench.sh)
# on the larger model, measuring only. Commit de1f8a96, CLI 0.4.1-dev.2611191810 (package 0.4.1-dev.2611191811). One model
# run a call of this script:
#   m6.sh cuda TYPE ROUTE fill|warm   a fresh tuning folder tuning-cuda-TYPE shared by both routes, removed before the
#                                     type's first run (adad fill); runs in the order adad fill, name fill, adad warm,
#                                     name warm: idrak bench Qwen/Qwen3-1.7B -w TYPE -d cuda --prompt-tokens 512 --tokens 128 -n 3
#   m6.sh cpu TYPE ROUTE run1|run2    a fresh CPU tuning file tuning-cpu-TYPE.tsv, removed before the type's first run
#                                     (adad run1); runs adad, name, adad, name: --prompt-tokens 128 --tokens 32 -n 2
#   m6.sh profile cuda|cpu TYPE ROUTE [LABEL]  the prompt pass on the type's filled folder or file (CUDA --prompt-tokens 512
#                                     --top 30; CPU --prompt-tokens 128 --top 20)
#   m6.sh dprofile cuda TYPE ROUTE [LABEL]     one decoding step's profile (--top 25)
#   m6.sh trace cuda|cpu TYPE ROUTE   idrak trace --stacks of a bench run (CUDA --prompt-tokens 512 --tokens 128 -n 1, CPU
#                                     --prompt-tokens 128 --tokens 8 -n 1) on the filled folder or file
#   m6.sh small TYPE adad|name        the check at Qwen/Qwen3-0.6B: a fill bench of the route on a fresh folder
#                                     check-0.6b/tuning-cuda-TYPE (adad first, the folder removed before it)
#   m6.sh smallprofile TYPE ROUTE     the route's CUDA prompt profile at 0.6B on that folder (--prompt-tokens 512 --top 30)
# Every run's full log is kept (its "measured lane:" lines are the lanes it chose); one summary line a bench run in
# results.txt. The route is set on every run (IDRAK_ROUTES). Each profile and trace run (but the CUDA adad prompt
# profiles, made before this was added) writes the lines it stored in its tuning file to tuning-added.txt.
set -u
cd D:/Projects/idrak || exit 1
model=Qwen/Qwen3-1.7B
out=D:/Projects/idrak/test-output/adad/m6-proof-1.7b
results=$out/results.txt
kind=$1
# The tuning file a profile or trace reads; its lines are copied before the run and diffed after it, the diff (the lines
# the run stored, or "(none)") appended to tuning-added.txt with the run's name.
tfile() { if [ "$1" = cuda ]; then ls "$out/tuning-cuda-$2"/*.txt 2>/dev/null | head -1; else echo "$out/tuning-cpu-$2.tsv"; fi; }
snap() { f=$(tfile "$1" "$2"); if [ -f "$f" ]; then cp "$f" "$out/.snap"; else : > "$out/.snap"; fi; }
sums() {
  f=$(tfile "$1" "$2")
  { echo "== $3 ($(date '+%T')): lines added or changed in $f"; diff "$out/.snap" "$f" | grep '^[<>]' || echo "  (none)"; } | tee -a "$out/tuning-added.txt"
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
  echo "start $(date '+%F %T')  cuda $w $r $pass"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$tune idrak bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 3 > "$log" 2>&1
  code=$?
  cat "$log"; line cuda "$w" "$r" "$pass" "$code" "$log" | tee -a "$results"
  ;;
cpu)
  w=$2; r=$3; pass=$4
  cpufile=$out/tuning-cpu-$w.tsv; log=$out/cpu-$w-$r-$pass.txt
  [ "$r" = adad ] && [ "$pass" = run1 ] && rm -f "$cpufile"
  echo "start $(date '+%F %T')  cpu $w $r $pass"
  IDRAK_ROUTES=$r IDRAK_CPU_TUNING_FILE=$cpufile idrak bench $model -w "$w" -d cpu --prompt-tokens 128 --tokens 32 -n 2 > "$log" 2>&1
  code=$?
  cat "$log"; line cpu "$w" "$r" "$pass" "$code" "$log" | tee -a "$results"
  ;;
profile|dprofile)
  dev=$2; w=$3; r=$4; label=${5:-}
  kindname=prompt; [ "$kind" = dprofile ] && kindname=decode
  log=$out/profile-$dev-$w-$r-$kindname${label:+-$label}.txt
  echo "start $(date '+%F %T')  $kind $dev $w $r $label"
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
  echo "start $(date '+%F %T')  trace $dev $w $r"
  snap "$dev" "$w"
  if [ "$dev" = cuda ]; then
    IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-cuda-$w idrak trace --stacks -o "$nt" --top 20 -- bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 1 > "$log" 2>&1
  else
    IDRAK_ROUTES=$r IDRAK_CPU_TUNING_FILE=$out/tuning-cpu-$w.tsv idrak trace --stacks -o "$nt" --top 20 -- bench $model -w "$w" -d cpu --prompt-tokens 128 --tokens 8 -n 1 > "$log" 2>&1
  fi
  echo "exit $?" >> "$log"; cat "$log"
  sums "$dev" "$w" "$(basename "$log" .txt)"
  ;;
small|smallprofile)
  # The check on Qwen/Qwen3-0.6B with this build (the re-bench at 0.6B has no profiles): a fresh folder
  # check-0.6b/tuning-cuda-TYPE (removed before the type's adad fill), filled by a bench of each route, then each route's
  # prompt profile on it. Logs and the tuning diff in check-0.6b/.
  w=$2; r=$3; sm=Qwen/Qwen3-0.6B; so=$out/check-0.6b; mkdir -p "$so"; tune=$so/tuning-cuda-$w
  if [ "$kind" = small ]; then
    log=$so/cuda-$w-$r-fill.txt
    [ "$r" = adad ] && rm -rf "$tune"
    echo "start $(date '+%F %T')  0.6B cuda $w $r fill"
    IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$tune idrak bench $sm -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 3 > "$log" 2>&1
    code=$?
    cat "$log"; line cuda06 "$w" "$r" fill "$code" "$log" | tee -a "$so/results.txt"
  else
    log=$so/profile-cuda-$w-$r-prompt.txt
    echo "start $(date '+%F %T')  0.6B profile cuda $w $r"
    f=$(ls "$tune"/*.txt | head -1); cp "$f" "$out/.snap"
    IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$tune idrak profile $sm -w "$w" -d cuda --pass prompt --prompt-tokens 512 --top 30 > "$log" 2>&1
    echo "exit $?" >> "$log"; cat "$log"
    { echo "== 0.6B $(basename "$log" .txt) ($(date '+%T')): lines added or changed in $f"; diff "$out/.snap" "$f" | grep '^[<>]' || echo "  (none)"; } | tee -a "$so/tuning-added.txt"
    rm -f "$out/.snap"
  fi
  ;;
*)
  echo "usage: m6.sh cuda|cpu TYPE ROUTE PASS | profile|dprofile|trace DEVICE TYPE ROUTE | small|smallprofile TYPE ROUTE"; exit 2
  ;;
esac
echo "end $(date '+%F %T')"
