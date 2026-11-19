#!/bin/sh
# M9: adad against name on Llama 3.2 1B and 3B (1b/results.txt, 3b/results.txt), beside the same ratios on Qwen3-0.6B
# (the re-bench, plans/opened/39-rebench/results.txt) and Qwen3-1.7B (M6, m6-proof-1.7b/results.txt). CUDA: the warm
# runs; CPU: the mean of run1 and run2 of each route. Usage: ratios.sh LLAMA_1B LLAMA_3B QWEN_0.6B QWEN_1.7B
awk '
  FNR == 1 { file++ }
  {
    dev = $1; w = $2; r = $3; pass = $4; p = $6; g = $8
    if (dev == "cuda" && pass != "warm") next
    if (dev != "cuda" && dev != "cpu") next
    key = dev " " w
    if (file == 1 && !(key in seen)) { seen[key] = 1; order[++n] = key }
    P[file, key, r] += p; G[file, key, r] += g; C[file, key, r]++
  }
  function ratio(f, k, which) {
    if (C[f, k, "adad"] == 0 || C[f, k, "name"] == 0) return -1
    if (which == "p") return (P[f, k, "adad"] / C[f, k, "adad"]) / (P[f, k, "name"] / C[f, k, "name"])
    return (G[f, k, "adad"] / C[f, k, "adad"]) / (G[f, k, "name"] / C[f, k, "name"])
  }
  function show(x) { return x < 0 ? "      -" : sprintf("%7.3f", x) }
  END {
    printf "%-12s %9s %9s %9s %9s | prompt adad/name: %7s %7s %7s %7s | generation adad/name: %7s %7s %7s %7s\n", "device type", "1B adad", "1B name", "3B adad", "3B name", "Q0.6B", "Q1.7B", "L1B", "L3B", "Q0.6B", "Q1.7B", "L1B", "L3B"
    for (i = 1; i <= n; i++) {
      k = order[i]
      printf "%-12s %9.1f %9.1f %9.1f %9.1f | %18s %s %s %s %s | %22s %s %s %s %s\n", k, P[1, k, "adad"] / C[1, k, "adad"], P[1, k, "name"] / C[1, k, "name"], P[2, k, "adad"] / C[2, k, "adad"], P[2, k, "name"] / C[2, k, "name"], "", show(ratio(3, k, "p")), show(ratio(4, k, "p")), show(ratio(1, k, "p")), show(ratio(2, k, "p")), "", show(ratio(3, k, "g")), show(ratio(4, k, "g")), show(ratio(1, k, "g")), show(ratio(2, k, "g"))
    }
    printf "\ngeneration tok/s (CUDA warm; CPU mean of two runs)\n%-12s %9s %9s %9s %9s\n", "device type", "1B adad", "1B name", "3B adad", "3B name"
    for (i = 1; i <= n; i++) {
      k = order[i]
      printf "%-12s %9.2f %9.2f %9.2f %9.2f\n", k, G[1, k, "adad"] / C[1, k, "adad"], G[1, k, "name"] / C[1, k, "name"], G[2, k, "adad"] / C[2, k, "adad"], G[2, k, "name"] / C[2, k, "name"]
    }
  }' "$1" "$2" "$3" "$4"
