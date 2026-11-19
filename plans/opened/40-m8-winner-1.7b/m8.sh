#!/bin/sh
# M8: the winner everywhere on Qwen/Qwen3-1.7B, measuring only, with M7's method (m7-swap/m7.sh). The winner is the
# native kind (CUDA lane 2 of StoredMeasured, lane 1 of StoredSuper); it is forced on every prompt product (the 512-row
# entries) of one type per family, every 512-row group launch off (0), so each product runs the native kind on its own.
# Decode entries are left as they are. The M6 build's CLI 0.4.1-dev.2611191810 (commit de1f8a96: the M6 tuning folders name
# it, so it reads them as filled). Every folder here is a copy of an M6 folder (m6-proof-1.7b/tuning-cuda-TYPE); the M6
# folders are never written. One model run a call:
#   m8.sh setup                          tuning-control-TYPE (untouched) for bf16, int8, q4_0, q4_k, iq2_xxs; tuning-force-TYPE
#                                        for bf16, int8, q4_0, iq2_xxs: every "ProductRoute 2000 512 ..." entry set to 2
#                                        (StoredMeasured), or every "ProductRoute 3000 512 ..." entry set to 1 (StoredSuper),
#                                        and every "ProductRoute 9700 512 ..." group set to 0; each edit written as
#                                        force-TYPE.diff. q4_k is already so in M6 (every 3000 entry 1, both groups 0): its
#                                        edit changes nothing and it runs only as a control. Each key's candidate-count field
#                                        is kept; lane 2 is among the 2000 keys' candidates (bf16's five and q4_0's six already
#                                        hold 2, int8's seven are every lane 0-6), lane 1 always among StoredSuper's.
#   m8.sh bench LABEL TYPE ROUTE PASS    idrak bench Qwen/Qwen3-1.7B -w TYPE -d cuda --prompt-tokens 512 --tokens 128 -n 3
#                                        on tuning-LABEL -> LABEL-ROUTE-PASS.txt, a summary line in results.txt
#   m8.sh profile LABEL TYPE [REPEAT]     the adad prompt profile (--pass prompt --prompt-tokens 512 --top 30) on
#                                        tuning-LABEL -> profile-LABEL-prompt.txt (a repeat: profile-LABEL-prompt-REPEAT.txt)
#   m8.sh complete LABEL TYPE ROUTE       the greedy continuation (--temperature 0 --max-tokens 48) of prompt.txt (M7's, 475
#                                        tokens: the 512 class, the bench's prompt lanes) -> complete-LABEL-ROUTE.txt
# Every run copies its folder's tuning file before and diffs it after into tuning-added.txt (the lines it stored or
# changed, or "(none)").
set -u
cd D:/Projects/idrak || exit 1
model=Qwen/Qwen3-1.7B
m6=D:/Projects/idrak/test-output/adad/m6-proof-1.7b
out=D:/Projects/idrak/test-output/adad/m8-winner
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
force() {
  # force TYPE: the copy tuning-force-TYPE edited in place, the diff written (empty when nothing changed)
  w=$1; l=force-$w; copy "$w" "$l"; f=$(tfile "$l"); o=$(ls "$m6/tuning-cuda-$w"/*.txt | head -1)
  sed -E -i 's/^(ProductRoute 2000 512 [0-9]+ [0-9]+ [0-9]+ 0 0) = [0-9]+$/\1 = 2/; s/^(ProductRoute 3000 512 [0-9]+ [0-9]+ [0-9]+ 0 0) = [0-9]+$/\1 = 1/; s/^(ProductRoute 9700 512 [0-9 ]+) = [0-9]+$/\1 = 0/' "$f"
  diff -u "$o" "$f" > "$out/$l.diff"
  echo "$l: $(grep -c '^+[^+]' "$out/$l.diff") lines changed"; grep '^[-+][^-+]' "$out/$l.diff"
}
case "$kind" in
setup)
  for w in bf16 int8 q4_0 q4_k iq2_xxs; do copy "$w" "control-$w"; done
  for w in bf16 int8 q4_0 q4_k iq2_xxs; do force "$w"; done
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
  l=$2; w=$3; label=${4:-}; log=$out/profile-$l-prompt${label:+-$label}.txt
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
  echo "usage: m8.sh setup | bench LABEL TYPE ROUTE PASS | profile LABEL TYPE | complete LABEL TYPE ROUTE"; exit 2
  ;;
esac
echo "end $(date '+%F %T')"
