#!/bin/sh
# M6: each CUDA type's fill run against its warm run, per route, from results.txt: the prompt's and generation's
# difference (fill over warm, minus one) and the peak's difference (fill minus warm, MiB). Usage: spread.sh results.txt
awk '$1 == "cuda" { k = $2 " " $3; P[k, $4] = $6; G[k, $4] = $8; M[k, $4] = $13; if (!(k in s)) { s[k] = 1; o[++n] = k } }
  END { printf "%-14s %8s %8s %9s\n", "type route", "prompt", "gen", "peak MiB";
        for (i = 1; i <= n; i++) { k = o[i]; printf "%-14s %+7.1f%% %+7.1f%% %+9d\n", k, (P[k, "fill"] / P[k, "warm"] - 1) * 100, (G[k, "fill"] / G[k, "warm"] - 1) * 100, M[k, "fill"] - M[k, "warm"] } }' "$1"
