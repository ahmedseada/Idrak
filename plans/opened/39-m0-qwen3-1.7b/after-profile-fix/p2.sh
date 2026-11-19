#!/bin/sh
# Plan 39, piece 2 of phase 1: `idrak profile` fixed (M0-1, M0-2, M0-3), re-measured against M0 (IDRAK_ROUTES=name, set
# on every run). CLI 0.4.1-dev.2611191709, built from 45a32809 with this piece's change (the profile times the network's
# own cached pass, a hook before each layer); the interim CLI 0.4.1-dev.2611191703 (the profile's own per-layer loop in
# the network's precision) made the files named -2611191703 and those in interim-2611191703/. One model run a call:
#   p2.sh check cuda bf16               a bench on bench-check/tuning-cuda-bf16, a copy of M0's folder: its first run
#                                       measured anew (the tuning file's library line names the commit, 4802653c in M0's
#                                       files, so a build of another commit measures its choices again), the next is warm
#   p2.sh bench cuda TYPE fill          a fresh folder tuning-cuda-TYPE filled by a bench of this build (M0's folders
#                                       cannot be read by it, above): the folder the profiles start from holds only what
#                                       the bench stored (copies of M0's folders with M0's profile lines taken out were
#                                       made first and are replaced by these fills)
#   p2.sh profile cuda TYPE prompt|decode LABEL   M0's profile commands
#   p2.sh profile cpu TYPE prompt LABEL M0's CPU prompt profile, with a copy of M0's CPU tuning file (its identity names
#                                       no commit, so it is read as filled)
# LABEL: first (the first profile on the bench-filled folder) or read (the second). Every run's lines added to its tuning
# file are written to tuning-added.txt (diff before and after), with the run's name.
set -u
cd D:/Projects/idrak || exit 1
model=Qwen/Qwen3-1.7B
out=D:/Projects/idrak/test-output/adad/p2-profile
export IDRAK_ROUTES=name
kind=$1; dev=$2; w=$3; pass=${4:-}; label=${5:-}
tfile() { if [ "$dev" = cuda ]; then ls "$1"/*.txt 2>/dev/null | head -1; else echo "$1"; fi; }
snap() { f=$(tfile "$1"); if [ -f "$f" ]; then cp "$f" "$out/.snap"; else : > "$out/.snap"; fi; }
sums() {
  f=$(tfile "$1")
  { echo "== $2 ($(date '+%T')): lines added or changed in $f"; diff "$out/.snap" "$f" | grep '^[<>]' || echo "  (none)"; } | tee -a "$out/tuning-added.txt"
  rm -f "$out/.snap"
}
echo "start $(date '+%F %T')  $kind $dev $w $pass $label  IDRAK_ROUTES=$IDRAK_ROUTES"
case "$kind" in
bench)
  tune=$out/tuning-cuda-$w; log=$out/cuda-$w-$pass.txt
  [ "$pass" = fill ] && rm -rf "$tune"
  snap "$tune"
  IDRAK_TUNING_CACHE=$tune idrak bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 3 > "$log" 2>&1
  echo "exit $?" >> "$log"; cat "$log"; sums "$tune" "bench cuda $w $pass"
  ;;
check)
  tune=$out/bench-check/tuning-cuda-$w; log=$out/bench-check/cuda-$w-warm.txt
  snap "$tune"
  IDRAK_TUNING_CACHE=$tune idrak bench $model -w "$w" -d cuda --prompt-tokens 512 --tokens 128 -n 3 > "$log" 2>&1
  echo "exit $?" >> "$log"; cat "$log"; sums "$tune" "bench check cuda $w warm"
  ;;
profile)
  log=$out/profile-$dev-$w-$pass-$label.txt
  if [ "$dev" = cuda ]; then
    tune=$out/tuning-cuda-$w; snap "$tune"
    if [ "$pass" = prompt ]; then
      IDRAK_TUNING_CACHE=$tune idrak profile $model -w "$w" -d cuda --pass prompt --prompt-tokens 512 --top 30 > "$log" 2>&1
    else
      IDRAK_TUNING_CACHE=$tune idrak profile $model -w "$w" -d cuda --top 25 > "$log" 2>&1
    fi
  else
    tune=$out/tuning-cpu-$w.tsv; snap "$tune"
    IDRAK_CPU_TUNING_FILE=$tune idrak profile $model -w "$w" -d cpu --pass prompt --prompt-tokens 128 --top 20 > "$log" 2>&1
  fi
  echo "exit $?" >> "$log"; cat "$log"; sums "$tune" "profile $dev $w $pass $label"
  ;;
*)
  echo "usage: p2.sh bench|check|profile ..."; exit 2
  ;;
esac
echo "end $(date '+%F %T')"
