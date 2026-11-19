#!/bin/sh
# M10: B's quality cost, measured with the existing lane 3 (StoredMeasured 3, Adad's block route: x rounded into q8_0
# blocks of 32 values, the weight codes exact, int32 sums scaled into float32; kernels blocks_quantize_q8_0 and
# matmul_blocks_*) forced on the prompt products. Measuring only. The M6 build's CLI 0.4.1-dev.2611191810 (commit
# de1f8a96; no source changed since). One model run a call; MODEL is qwen3-1.7b or llama-3b, the outputs in MODEL/:
#   m10.sh MODEL ppl LABEL TYPE ROUTE RUN   idrak perplexity on text.txt (--window 512 --max-tokens 6656: 13 windows of
#                                           exactly 512 tokens) on CUDA, IDRAK_ROUTES=ROUTE, IDRAK_TUNING_CACHE=tuning-LABEL
#                                           -> ppl-LABEL-RUN.txt and a line in results.txt
#   m10.sh MODEL force TYPE                 tuning-TYPE-lane3: a copy of tuning-TYPE-native with every prompt-shaped
#                                           StoredMeasured entry ("ProductRoute 2000 512 N K C 0 0", but the head's N = the
#                                           vocabulary) set to 3 and every 512-row group ("ProductRoute 9700 512 ...") set
#                                           to 0, so each product runs its own lane; the edit written as force-TYPE.diff
#   m10.sh MODEL control TYPE               tuning-TYPE-ctl: a copy of tuning-TYPE-native (the side bench's native control)
#   m10.sh MODEL bench LABEL TYPE PASS      idrak bench MODEL -w TYPE -d cuda --prompt-tokens 512 --tokens 128 -n 3 (adad) on
#                                           tuning-LABEL -> bench-LABEL-PASS.txt, a line in results.txt
#   m10.sh MODEL profile LABEL TYPE         the adad prompt profile (--pass prompt --prompt-tokens 512 --top 30) on
#                                           tuning-LABEL -> profile-LABEL-prompt.txt
# Every run copies its folder's tuning file before and diffs it after into MODEL/tuning-added.txt.
set -u
cd D:/Projects/idrak || exit 1
m=$1; shift
case "$m" in
qwen3-1.7b) model=Qwen/Qwen3-1.7B; vocab=151936 ;;
llama-3b) model=meta-llama/Llama-3.2-3B-Instruct; vocab=128256 ;;
*) echo "model: qwen3-1.7b or llama-3b"; exit 2 ;;
esac
top=D:/Projects/idrak/test-output/adad/m10-lane3-quality
out=$top/$m
results=$out/results.txt
kind=$1
tfile() { ls "$out/tuning-$1"/*.txt 2>/dev/null | head -1; }
snap() { f=$(tfile "$1"); if [ -f "$f" ]; then cp "$f" "$out/.snap"; else : > "$out/.snap"; fi; }
sums() {
  f=$(tfile "$1")
  { echo "== $2 ($(date '+%T')): lines added or changed in tuning-$1"; if [ -f "$f" ]; then diff "$out/.snap" "$f" | grep '^[<>]' || echo "  (none)"; else echo "  (no tuning file)"; fi; } | tee -a "$out/tuning-added.txt"
  rm -f "$out/.snap"
}
case "$kind" in
ppl)
  l=$2; w=$3; r=$4; run=$5; log=$out/ppl-$l-$run.txt
  echo "start $(date '+%F %T')  $m ppl $l $w $r $run"
  snap "$l"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-$l idrak perplexity $model "$top/text.txt" -w "$w" -d cuda --window 512 --max-tokens 6656 < /dev/null > "$log" 2>&1
  code=$?
  cat "$log"
  awk -v l="$l" -v r="$r" -v run="$run" -v w="$w" -v x="$code" '/perplexity [0-9]/ {for (i = 1; i <= NF; i++) if ($i == "perplexity") p = $(i + 1)} /tokens scored in/ {s = $1; win = $5; for (i = 1; i <= NF; i++) { if ($i == "log-likelihood") nll = $(i + 1); if ($i == "s" && $(i - 1) ~ /^[0-9.]+$/) sec = $(i - 1) }} END {printf "ppl   %-14s %-8s %-5s %-5s perplexity %s  mean-nll %s  scored %s  windows %s  seconds %s  exit %s\n", l, w, r, run, p, nll, s, win, sec, x}' "$log" | tee -a "$results"
  sums "$l" "$(basename "$log" .txt)"
  ;;
force)
  w=$2; src=$out/tuning-$w-native; dst=$out/tuning-$w-lane3
  rm -rf "$dst"; mkdir -p "$dst"; cp "$src"/*.txt "$dst/"
  f=$(tfile "$w-lane3"); o=$(tfile "$w-native")
  awk -v vocab="$vocab" '
    $1 == "ProductRoute" && $2 == 2000 && $3 == 512 && $4 != vocab && $9 == "=" { $10 = 3 }
    $1 == "ProductRoute" && $2 == 9700 && $3 == 512 && $(NF - 1) == "=" { $NF = 0 }
    { print }' "$o" > "$f"
  diff -u "$o" "$f" > "$out/force-$w.diff"
  echo "force-$w: $(grep -c '^+[^+]' "$out/force-$w.diff") lines changed"; grep '^[-+][^-+]' "$out/force-$w.diff"
  ;;
control)
  w=$2; src=$out/tuning-$w-native; dst=$out/tuning-$w-ctl
  rm -rf "$dst"; mkdir -p "$dst"; cp "$src"/*.txt "$dst/"; cmp "$(tfile "$w-native")" "$(tfile "$w-ctl")" && echo "tuning-$w-ctl: a byte-equal copy of tuning-$w-native"
  ;;
bench)
  l=$2; w=$3; pass=$4; log=$out/bench-$l-$pass.txt
  echo "start $(date '+%F %T')  $m bench $l $w $pass"
  snap "$l"
  IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=$out/tuning-$l idrak bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 3 < /dev/null > "$log" 2>&1
  code=$?
  cat "$log"
  awk -v l="$l" -v p2="$pass" -v x="$code" '$1=="prompt"&&$3=="tokens/s"{pr=$2} $1=="generation"&&$3=="tokens/s"{g=$2} $1=="memory"&&$2=="peak"{mem=$3} $1=="weights"&&$2=="memory"{wm=$3} END{printf "bench %-14s %-5s prompt %8s  generation %7s  weights %6s MiB  peak %6s MiB  exit %s\n", l, p2, pr, g, wm, mem, x}' "$log" | tee -a "$results"
  sums "$l" "$(basename "$log" .txt)"
  ;;
profile)
  l=$2; w=$3; log=$out/profile-$l-prompt.txt
  echo "start $(date '+%F %T')  $m profile $l $w"
  snap "$l"
  IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=$out/tuning-$l idrak profile $model -w "$w" -d cuda --pass prompt --prompt-tokens 512 --top 30 < /dev/null > "$log" 2>&1
  echo "exit $?" >> "$log"; cat "$log"
  sums "$l" "$(basename "$log" .txt)"
  ;;
*)
  echo "usage: m10.sh MODEL ppl LABEL TYPE ROUTE RUN | force TYPE | control TYPE | bench LABEL TYPE PASS | profile LABEL TYPE"; exit 2
  ;;
esac
echo "end $(date '+%F %T')"
