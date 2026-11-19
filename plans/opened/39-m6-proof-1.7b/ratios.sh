#!/bin/sh
# M6: adad against name from results.txt (Qwen3-1.7B) beside the same ratio from the 0.6B re-bench
# (plans/opened/39-rebench/results.txt). CUDA: the warm runs. CPU: the mean of run1 and run2 of each route.
# move = the 1.7B ratio over the 0.6B ratio, minus one. adadW, nameW: weights memory at 1.7B (MiB).
# Usage: ratios.sh M6_RESULTS REBENCH_RESULTS > ratios.txt
awk '
  FNR == 1 { file++ }
  {
    dev = $1; w = $2; r = $3; pass = $4; p = $6; g = $8; wm = $10
    if (dev == "cuda" && pass != "warm") next
    key = dev " " w
    if (file == 1 && !(key in seen)) { seen[key] = 1; order[++n] = key }
    P[file, key, r] += p; G[file, key, r] += g; C[file, key, r]++
    if (file == 1) W[key, r] = wm
  }
  END {
    printf "%-12s %9s %9s %7s %7s %7s | %7s %7s %7s %7s %7s | %7s %7s\n", "device type", "adad pr", "name pr", "ratio", "0.6B", "move", "adad g", "name g", "ratio", "0.6B", "move", "adadW", "nameW"
    for (i = 1; i <= n; i++) {
      k = order[i]
      ap = P[1, k, "adad"] / C[1, k, "adad"]; np = P[1, k, "name"] / C[1, k, "name"]
      ag = G[1, k, "adad"] / C[1, k, "adad"]; ng = G[1, k, "name"] / C[1, k, "name"]
      sp = (P[2, k, "adad"] / C[2, k, "adad"]) / (P[2, k, "name"] / C[2, k, "name"])
      sg = (G[2, k, "adad"] / C[2, k, "adad"]) / (G[2, k, "name"] / C[2, k, "name"])
      printf "%-12s %9.1f %9.1f %7.3f %7.3f %+6.1f%% | %7.2f %7.2f %7.3f %7.3f %+6.1f%% | %7s %7s\n", k, ap, np, ap / np, sp, (ap / np / sp - 1) * 100, ag, ng, ag / ng, sg, (ag / ng / sg - 1) * 100, W[k, "adad"], W[k, "name"]
    }
  }' "$1" "$2"
