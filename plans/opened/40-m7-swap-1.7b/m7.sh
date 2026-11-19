#!/bin/sh
# M7: the lane swap on Qwen/Qwen3-1.7B, measuring only. Commit de1f8a96's CLI 0.4.1-dev.2611191810 (the M6 build: the M6
# tuning folders name its commit, so it reads them as filled). Every folder here is a copy of an M6 folder
# (test-output/adad/m6-proof-1.7b/tuning-cuda-TYPE); the M6 folders are never written. One model run a call:
#   m7.sh setup                          the copies: tuning-control-TYPE (untouched) for q4_0, q8_0, int8, q4_k, iq2_xxs;
#                                        tuning-swap-TYPE (q4_0, q8_0, int8) and tuning-swap-TYPE-A, -B (q4_k, iq2_xxs)
#                                        edited as below; each edit written as swap-TYPE[-A|-B].diff (original, edited)
#     q4_0, q8_0: every "ProductRoute 2000 512 ..." entry at lane 2 set to 6 (the tensor rows), and the gate/up group
#                 "9700 512 2048 6144 6144 0 0" set to 1
#     int8:       q/o "2000 512 2048 2048 7" 2 -> 6, down "2000 512 2048 6144 7" 5 -> 6; the groups stay 1
#     q4_k, iq2_xxs A: every "ProductRoute 3000 512 ..." entry set to 4 (the tensor rows over rows widened for the
#                 call), both 512-row groups set to 1;  B: the same with 3 (the tensor rows decoding as each tile loads)
#     The candidate-count field of every key is kept (the key still matches); each lane set is among the key's candidates
#     (StoredMeasured: 6 when TensorRowsSource exists, as at 0.6B with the same counts; StoredSuper: q4_k's five {0..4},
#     iq2_xxs's four {0,1,3,4}; groups {0,1}).
#   m7.sh bench LABEL TYPE ROUTE PASS    idrak bench Qwen/Qwen3-1.7B -w TYPE -d cuda --prompt-tokens 512 --tokens 128 -n 3
#                                        on tuning-LABEL -> LABEL-ROUTE-PASS.txt, a summary line in results.txt
#   m7.sh profile LABEL TYPE              the adad prompt profile (--pass prompt --prompt-tokens 512 --top 30) on
#                                        tuning-LABEL -> profile-LABEL-prompt.txt
#   m7.sh complete LABEL TYPE ROUTE       the greedy continuation (--temperature 0 --max-tokens 48) of prompt.txt (475
#                                        tokens: the 512 class, the bench's prompt lanes) -> complete-LABEL-ROUTE.txt
# Every run copies its folder's tuning file before and diffs it after into tuning-added.txt (the lines it stored or
# changed, or "(none)").
set -u
cd D:/Projects/idrak || exit 1
model=Qwen/Qwen3-1.7B
m6=D:/Projects/idrak/test-output/adad/m6-proof-1.7b
out=D:/Projects/idrak/test-output/adad/m7-swap
results=$out/results.txt
kind=$1
tfile() { ls "$out/tuning-$1"/*.txt | head -1; }
snap() { cp "$(tfile "$1")" "$out/.snap"; }
sums() {
  f=$(tfile "$1")
  { echo "== $2 ($(date '+%T')): lines added or changed in tuning-$1"; diff "$out/.snap" "$f" | grep '^[<>]' || echo "  (none)"; } | tee -a "$out/tuning-added.txt"
  rm -f "$out/.snap"
}
line() {
  awk -v l="$1" -v r="$2" -v p="$3" -v x="$4" '$1=="prompt"&&$3=="tokens/s"{pr=$2} $1=="generation"&&$3=="tokens/s"{g=$2} $1=="memory"&&$2=="peak"{mem=$3} $1=="weights"&&$2=="memory"{wm=$3} END{printf "%-22s %-5s %-5s prompt %8s  generation %7s  weights %6s MiB  peak %6s MiB  exit %s\n", l, r, p, pr, g, wm, mem, x}' "$5"
}
copy() { mkdir -p "$out/tuning-$2"; cp "$m6/tuning-cuda-$1"/*.txt "$out/tuning-$2/"; }
edit() {
  # edit TYPE LABEL SED-SCRIPT: the copy edited in place, the diff written
  f=$(tfile "$2"); o=$(ls "$m6/tuning-cuda-$1"/*.txt | head -1)
  sed -E -i "$3" "$f"
  diff -u "$o" "$f" > "$out/$2.diff"
  echo "$2: $(grep -c '^+[^+]' "$out/$2.diff") lines changed"; grep '^[-+][^-+]' "$out/$2.diff"
}
case "$kind" in
setup)
  for w in q4_0 q8_0 int8 q4_k iq2_xxs; do copy "$w" "control-$w"; done
  for w in q4_0 q8_0; do
    copy "$w" "swap-$w"
    edit "$w" "swap-$w" 's/^(ProductRoute 2000 512 [0-9]+ [0-9]+ [0-9]+ 0 0) = 2$/\1 = 6/; s/^(ProductRoute 9700 512 2048 6144 6144 0 0) = 0$/\1 = 1/'
  done
  copy int8 swap-int8
  edit int8 swap-int8 's/^(ProductRoute 2000 512 2048 2048 7 0 0) = 2$/\1 = 6/; s/^(ProductRoute 2000 512 2048 6144 7 0 0) = 5$/\1 = 6/'
  for w in q4_k iq2_xxs; do
    copy "$w" "swap-$w-A"
    edit "$w" "swap-$w-A" 's/^(ProductRoute 3000 512 [0-9]+ [0-9]+ [0-9]+ 0 0) = [0-9]+$/\1 = 4/; s/^(ProductRoute 9700 512 [0-9 ]+) = 0$/\1 = 1/'
    copy "$w" "swap-$w-B"
    edit "$w" "swap-$w-B" 's/^(ProductRoute 3000 512 [0-9]+ [0-9]+ [0-9]+ 0 0) = [0-9]+$/\1 = 3/; s/^(ProductRoute 9700 512 [0-9 ]+) = 0$/\1 = 1/'
  done
  ;;
bench)
  l=$2; w=$3; r=$4; pass=$5; log=$out/$l-$r-$pass.txt
  echo "start $(date '+%F %T')  bench $l $w $r $pass"
  snap "$l"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-$l idrak bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 3 > "$log" 2>&1
  code=$?
  cat "$log"; line "$l" "$r" "$pass" "$code" "$log" | tee -a "$results"
  sums "$l" "$(basename "$log" .txt)"
  ;;
profile)
  l=$2; w=$3; log=$out/profile-$l-prompt.txt
  echo "start $(date '+%F %T')  profile $l $w"
  snap "$l"
  IDRAK_ROUTES=adad IDRAK_TUNING_CACHE=$out/tuning-$l idrak profile $model -w "$w" -d cuda --pass prompt --prompt-tokens 512 --top 30 > "$log" 2>&1
  echo "exit $?" >> "$log"; cat "$log"
  sums "$l" "$(basename "$log" .txt)"
  ;;
complete)
  l=$2; w=$3; r=$4; log=$out/complete-$l-$r.txt
  echo "start $(date '+%F %T')  complete $l $w $r"
  snap "$l"
  IDRAK_ROUTES=$r IDRAK_TUNING_CACHE=$out/tuning-$l idrak complete $model -w "$w" -d cuda -i "$out/prompt.txt" --temperature 0 --max-tokens 48 > "$log" 2> "$out/complete-$l-$r.err"
  echo "exit $?" >> "$out/complete-$l-$r.err"; cat "$log"; echo; tail -n 3 "$out/complete-$l-$r.err"
  sums "$l" "$(basename "$log" .txt)"
  ;;
*)
  echo "usage: m7.sh setup | bench LABEL TYPE ROUTE PASS | profile LABEL TYPE | complete LABEL TYPE ROUTE"; exit 2
  ;;
esac
echo "end $(date '+%F %T')"
