#!/bin/sh
# M7: each variant's prompt (tok/s) against the same session's controls, from results.txt; and the controls against M6's
# warm runs (m6-proof-1.7b/results.txt), the spread between sessions. Usage: compare.sh M7_RESULTS M6_RESULTS
awk '
  FNR == 1 { file++ }
  file == 1 { P[$1, $2, $3] = $5 }
  file == 2 && $1 == "cuda" && $4 == "warm" { M6[$2, $3] = $6 }
  END {
    printf "%-15s %8s %8s %13s %12s %10s %10s %13s\n", "variant", "fill", "warm", "control adad", "control name", "warm/adad", "warm/name", "fill vs warm"
    n = split("q4_0:swap-q4_0 q8_0:swap-q8_0 int8:swap-int8 q4_k:swap-q4_k-A q4_k:swap-q4_k-B iq2_xxs:swap-iq2_xxs-A iq2_xxs:swap-iq2_xxs-B", v, " ")
    for (i = 1; i <= n; i++) {
      split(v[i], p, ":"); t = p[1]; l = p[2]; c = "control-" t
      fw = P[l, "adad", "fill"]; ww = P[l, "adad", "warm"]; ca = P[c, "adad", "warm"]; cn = P[c, "name", "warm"]
      printf "%-15s %8d %8d %13d %12d %10.3f %10.3f %+12.1f%%\n", l, fw, ww, ca, cn, ww / ca, ww / cn, (fw / ww - 1) * 100
    }
    printf "\n%-10s %12s %10s %8s %12s %10s %8s\n", "type", "M7 adad", "M6 adad", "change", "M7 name", "M6 name", "change"
    m = split("q4_0 q8_0 int8 q4_k iq2_xxs", t2, " ")
    for (i = 1; i <= m; i++) {
      t = t2[i]; c = "control-" t
      printf "%-10s %12d %10d %+7.1f%% %12d %10d %+7.1f%%\n", t, P[c, "adad", "warm"], M6[t, "adad"], (P[c, "adad", "warm"] / M6[t, "adad"] - 1) * 100, P[c, "name", "warm"], M6[t, "name"], (P[c, "name", "warm"] / M6[t, "name"] - 1) * 100
    }
  }' "$1" "$2"
