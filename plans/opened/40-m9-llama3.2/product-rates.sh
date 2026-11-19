#!/bin/sh
# M9: each product kernel of a CUDA prompt profile with its work and rate (M6's product-rates.sh, with the layer count and
# the shapes as arguments). Work = 2 m n k for each call, from the kernel's label: "MxNxK" (one product) or
# "multi MxN1+N2[+N3]xK" / "_multi MxN1+N2[+N3]xK" (a group over one input). The native kind's kernels carry no shape: all
# of them together are given the model's prompt products' work (the head left out) minus the labelled kernels' (one line,
# "native kind (all)"). m = 512.
# Usage: product-rates.sh LAYERS HIDDEN Q KV INTER FILE...  (q: hidden -> Q, k and v: hidden -> KV, o: Q -> hidden,
# gate and up: hidden -> INTER, down: INTER -> hidden)
L=$1; h=$2; q=$3; kv=$4; f=$5; shift 5
for file in "$@"; do
  awk -v L="$L" -v h="$h" -v q="$q" -v kv="$kv" -v inter="$f" -v file="$(basename "$file" .txt | sed 's/^profile-cuda-//; s/-prompt//')" -v dir="$(basename "$(dirname "$file")")" '
    function work(label,   p, a, n, i, outs, s) {
      n = split(label, p, "x"); if (n != 3) return -1
      s = 0; outs = split(p[2], a, "+"); for (i = 1; i <= outs; i++) s += a[i]
      return 2 * p[1] * s * p[3]
    }
    BEGIN { m = 512; total = L * 2 * m * (h * q + h * kv * 2 + q * h + h * inter * 2 + inter * h) }
    /^Kernels \(/ { inside = 1; next }
    inside && /^exit/ { inside = 0 }
    inside && NF >= 4 && $(NF) == "%" && ($1 ~ /^native_mma_|^gemm_tc_/) {
      ms = $(NF - 2); calls = $(NF - 3); label = $(NF - 4)
      if ($1 ~ /^native_mma_/) { natms += ms; next }
      w = work(label) * calls; known += w
      printf "%-4s %-22s %-58s %4d calls %7.3f ms %8.2f GFLOP %6.1f TFLOP/s\n", dir, file, substr($0, 1, index($0, label) + length(label) - 1), calls, ms, w / 1e9, w / ms / 1e9
    }
    END {
      if (natms > 0) printf "%-4s %-22s %-58s %4s       %7.3f ms %8.2f GFLOP %6.1f TFLOP/s\n", dir, file, "native kind (all)", "", natms, (total - known) / 1e9, (total - known) / natms / 1e9
    }' "$file"
done
