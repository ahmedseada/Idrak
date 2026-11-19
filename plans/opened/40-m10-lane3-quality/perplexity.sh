#!/bin/sh
# M10: the perplexity tables, from MODEL/results.txt (the "ppl" lines). For each weight type: the native lane (the
# measured choice; fill and warm), lane 3 forced on the prompt products (run1, run2), name (run1, run2), and bf16 under
# adad and name (fill/warm, run1/run2); then the differences of the first run of each kind (perplexity and its relative
# change, mean negative log-likelihood). Usage: perplexity.sh qwen3-1.7b/results.txt llama-3b/results.txt
for f in "$@"; do
  echo "## $(dirname "$f")"
  awk '
    $1 == "ppl" { key = $2 "/" $5; P[key] = $7; N[key] = $9; S[key] = $11; W[key] = $13 }
    function d(a, b) { return sprintf("%+.3f (%+.2f%%)", P[a] - P[b], (P[a] / P[b] - 1) * 100) }
    END {
      printf "%-28s %10s %10s %8s %8s\n", "kind / run", "perplexity", "mean nll", "scored", "windows"
      n = split("q4_0-native/fill q4_0-native/warm q4_0-lane3/run1 q4_0-lane3/run2 q4_0-name/run1 q4_0-name/run2 int8-native/fill int8-native/warm int8-lane3/run1 int8-lane3/run2 int8-name/run1 int8-name/run2 bf16-adad/fill bf16-adad/warm bf16-name/run1 bf16-name/run2", k, " ")
      for (i = 1; i <= n; i++) printf "%-28s %10s %10s %8s %8s\n", k[i], P[k[i]], N[k[i]], S[k[i]], W[k[i]]
      printf "\ndifferences (perplexity, relative):\n"
      split("q4_0 int8", t, " ")
      for (j = 1; j <= 2; j++) {
        w = t[j]; nat = w "-native/fill"; l3 = w "-lane3/run1"; nm = w "-name/run1"
        printf "%-5s lane 3 - native %s   lane 3 - name %s   native - name %s\n", w, d(l3, nat), d(l3, nm), d(nat, nm)
        printf "%-5s against bf16 (adad): native %s   lane 3 %s   name %s\n", w, d(nat, "bf16-adad/fill"), d(l3, "bf16-adad/fill"), d(nm, "bf16-adad/fill")
        printf "%-5s against bf16 (name): native %s   lane 3 %s   name %s\n", w, d(nat, "bf16-name/run1"), d(l3, "bf16-name/run1"), d(nm, "bf16-name/run1")
      }
      printf "bf16  adad - name %s\n", d("bf16-adad/fill", "bf16-name/run1")
    }' "$f"
  echo
done
