#!/bin/sh
# M8: how much of the winner's time (the native kind on every prompt product) is conversion, per family, from lanes.txt.
# The winner's products = the native kernel + the weight's per-call conversion (native_pack_w / super_widen / grid_widen)
# + x packing + the parts' sums. Printed: each step's ms, the conversion's share of the winner's products and of the pass,
# and the share of conversion, x packing and sums together. The rows: the forced profiles (bf16, int8, q4_0, iq2_xxs: its
# repeat, the first being an outlier) and q4_k's control (already the winner everywhere). Usage: shares.sh lanes.txt
awk '
  { L[$1] = $0; for (i = 2; i < NF; i += 2) V[$1, $i] = $(i + 1) }
  END {
    printf "%-10s %-28s %7s %8s %7s %6s %6s %9s %11s %9s %13s\n", "family", "profile", "pass", "kernel", "conv.", "x pack", "sums", "products", "conv/prod", "conv/pass", "all extra/prod"
    n = split("16-bit-float:force-bf16 plain-integer:force-int8 32-value-blocks:force-q4_0 super-blocks:control-q4_k grids:force-iq2_xxs-prompt-rerun", r, " ")
    for (i = 1; i <= n; i++) {
      split(r[i], p, ":"); k = p[2]
      nat = V[k, "native"]; w = V[k, "widen"]; x = V[k, "packx"]; s = V[k, "parts"]; pass = V[k, "pass"]; prod = nat + w + x + s
      printf "%-10s %-28s %7.2f %8.3f %7.3f %6.3f %6.3f %9.3f %10.1f%% %8.1f%% %12.1f%%\n", p[1], k, pass, nat, w, x, s, prod, w / prod * 100, w / pass * 100, (w + x + s) / prod * 100
    }
  }' "$1"
