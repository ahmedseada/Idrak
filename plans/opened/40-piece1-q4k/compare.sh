#!/usr/bin/env bash
# compare.sh A B: two kernel benches' rows side by side (cycleA/kernels.txt, cycleB/kernels.txt): each product, m and lane's
# warm µs in both, the change, each run's own spread (min-max over its samples), and whether the change lies outside the
# larger spread; then the measured choice per shape in both and whether it moved.
top=D:/Projects/idrak/test-output/adad/piece1-q4k
a=$top/$1/kernels.txt; b=$top/$2/kernels.txt
rows() {
  awk -F'|' '
    /^## / { split($0, h, ", m = "); model = substr(h[1], 4); m = h[2] + 0 }
    /^\| (q|k|v|o|gate|up|down) \|/ {
      gsub(/ /, "", $2); lane = $5; gsub(/^ +| +$/, "", lane); sub(/ \(chosen\)/, "", lane); chosen = ($5 ~ /chosen/) ? 1 : 0
      split($7, w, " "); warm = w[1]; mm = w[2]; gsub(/[()]/, "", mm); split(mm, r, "-")
      spread = $8; gsub(/[ %]/, "", spread)
      printf "%s %s %d %s %s %s %s %d %s\n", model, $2, m, lane, warm, r[1], r[2], chosen, spread
    }' "$1"
}
join -j1 <(rows "$a" | awk '{print $1"/"$2"/"$3"/"$4, $5, $9, $8}' | sort) <(rows "$b" | awk '{print $1"/"$2"/"$3"/"$4, $5, $9, $8}' | sort) |
  awk -v A="$1" -v B="$2" 'BEGIN { printf "%-34s %10s %10s %8s %7s %7s %s\n", "model/product/m/lane", "cycle" A " us", "cycle" B " us", "change", "spreadA", "spreadB", "outside spread" }
    { change = ($5 - $2) / $2 * 100; s = ($3 > $6 ? $3 : $6); out = (change > s || -change > s) ? "yes" : "no"
      printf "%-34s %10.1f %10.1f %7.1f%% %6.1f%% %6.1f%% %s%s\n", $1, $2, $5, change, $3, $6, out, ($4 != $7 ? "  (chosen moved)" : "") }'
