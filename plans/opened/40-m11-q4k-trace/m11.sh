#!/bin/sh
# M11: q4_k's prompt on CUDA traced before B, measuring only. The M6 build's CLI 0.4.1-dev.2611191810 (commit de1f8a96; no
# source changed since). One model run a call; MODEL is qwen3-1.7b or llama-3b, ROUTE adad or name, the outputs in MODEL/:
#   m11.sh MODEL setup              tuning-q4_k-adad and tuning-q4_k-name: copies of the warm q4_k folder (Qwen3-1.7B: M6's
#                                   m6-proof-1.7b/tuning-cuda-q4_k; Llama 3B: M9's m9-llama3.2/3b/tuning-cuda-q4_k), one a
#                                   route, both filled by M6's or M9's adad and name runs
#   m11.sh MODEL trace ROUTE        idrak trace --stacks of a prompt-heavy bench: --prompt-tokens 512 --tokens 1 -n 10 (a
#                                   warm-up and 10 timed prompt passes, one generated token each) -> trace-ROUTE.txt
#   m11.sh MODEL profile ROUTE      the prompt profile (--pass prompt --prompt-tokens 512 --top 40) -> profile-ROUTE-prompt.txt
#   m11.sh MODEL bench ROUTE        a warm bench at the record's CUDA sizes (--prompt-tokens 512 --tokens 128 -n 3)
#                                   -> bench-ROUTE.txt (weights memory, memory in use, peak)
# Every run copies its folder's tuning file before and diffs it after into MODEL/tuning-added.txt.
set -u
cd D:/Projects/idrak || exit 1
m=$1; shift
case "$m" in
qwen3-1.7b) model=Qwen/Qwen3-1.7B; src=D:/Projects/idrak/test-output/adad/m6-proof-1.7b/tuning-cuda-q4_k ;;
llama-3b) model=meta-llama/Llama-3.2-3B-Instruct; src=D:/Projects/idrak/test-output/adad/m9-llama3.2/3b/tuning-cuda-q4_k ;;
*) echo "model: qwen3-1.7b or llama-3b"; exit 2 ;;
esac
out=D:/Projects/idrak/test-output/adad/m11-q4k-trace/$m
kind=$1
tfile() { ls "$out/tuning-q4_k-$1"/*.txt | head -1; }
snap() { cp "$(tfile "$1")" "$out/.snap"; }
sums() {
  { echo "== $2 ($(date '+%T')): lines added or changed in tuning-q4_k-$1"; diff "$out/.snap" "$(tfile "$1")" | grep '^[<>]' || echo "  (none)"; } | tee -a "$out/tuning-added.txt"
  rm -f "$out/.snap"
}
case "$kind" in
setup)
  for r in adad name; do rm -rf "$out/tuning-q4_k-$r"; mkdir -p "$out/tuning-q4_k-$r"; cp "$src"/*.txt "$out/tuning-q4_k-$r/"; cmp "$(ls "$src"/*.txt | head -1)" "$(tfile "$r")" && echo "tuning-q4_k-$r: a byte-equal copy of $src"; done
  ;;
trace)
  r=$2; log=$out/trace-$r.txt; nt=$out/trace-$r.nettrace
  echo "start $(date '+%F %T')  $m trace $r"
  snap "$r"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-q4_k-$r idrak trace --stacks -o "$nt" --top 25 -- bench $model -w q4_k -d cuda --prompt-tokens 512 --tokens 1 -n 10 < /dev/null > "$log" 2>&1
  echo "exit $?" >> "$log"; cat "$log"
  sums "$r" "$(basename "$log" .txt)"
  ;;
profile)
  r=$2; log=$out/profile-$r-prompt.txt
  echo "start $(date '+%F %T')  $m profile $r"
  snap "$r"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-q4_k-$r idrak profile $model -w q4_k -d cuda --pass prompt --prompt-tokens 512 --top 40 < /dev/null > "$log" 2>&1
  echo "exit $?" >> "$log"; cat "$log"
  sums "$r" "$(basename "$log" .txt)"
  ;;
bench)
  r=$2; log=$out/bench-$r.txt
  echo "start $(date '+%F %T')  $m bench $r"
  snap "$r"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-q4_k-$r idrak bench $model -w q4_k -d cuda --prompt-tokens 512 --tokens 128 -n 3 < /dev/null > "$log" 2>&1
  echo "exit $?" >> "$log"; cat "$log"
  sums "$r" "$(basename "$log" .txt)"
  ;;
*)
  echo "usage: m11.sh MODEL setup | trace ROUTE | profile ROUTE | bench ROUTE"; exit 2
  ;;
esac
echo "end $(date '+%F %T')"
