#!/bin/sh
# M6: each product kernel of a CUDA prompt profile with its work and rate. Work = 2 m n k for each call, from the
# kernel's label: "MxNxK" (one product), "multi MxN1+N2[+N3]xK" or "_multi MxN1+N2[+N3]xK" (a group over one input). The
# native kind's kernels carry no shape: their work is the model's prompt products (the head left out) minus the labelled
# ones; when there are two native kernels (int8 at 1.7B), the one reading int8 codes raw ("_ri8") is the down product
# (its lane: "512x6144->2048 ... int8 codes read raw") and the other the o product (2048->2048).
# Usage: product-rates.sh HIDDEN INTERMEDIATE FILE...   (Qwen3: q hidden->2048 at 0.6B or hidden->hidden at 1.7B, k and v
# hidden->1024, o q->hidden, gate and up hidden->INTERMEDIATE, down INTERMEDIATE->hidden; 28 layers; m = 512)
h=$1; f=$2; shift 2
for file in "$@"; do
  awk -v h="$h" -v inter="$f" -v file="$(basename "$file")" -v dir="$(basename "$(dirname "$file")")" '
    function work(label,   p, a, n, i, outs, s) {
      # label "MxNxK" or "MxN1+N2+N3xK"
      n = split(label, p, "x"); if (n != 3) return -1
      s = 0; outs = split(p[2], a, "+"); for (i = 1; i <= outs; i++) s += a[i]
      return 2 * p[1] * s * p[3]
    }
    BEGIN {
      L = 28; m = 512; qn = (h == 1024 ? 2048 : h)
      total = L * 2 * m * (h * qn + h * 1024 * 2 + qn * h + h * inter * 2 + inter * h)
      down = L * 2 * m * inter * h; o = L * 2 * m * qn * h
    }
    /^Kernels \(/ { inside = 1; next }
    inside && /^exit/ { inside = 0 }
    inside && NF >= 4 && $(NF) == "%" && ($1 ~ /^native_mma_|^gemm_tc_/) {
      ms = $(NF - 2); calls = $(NF - 3); label = $(NF - 4)
      if ($1 ~ /^native_mma_/) { nat[++nn] = $1; natms[nn] = ms; next }
      w = work(label) * calls; known += w
      printf "%-22s %-60s %4d calls %7.3f ms %8.2f GFLOP %6.1f TFLOP/s\n", dir "/" file, substr($0, 1, index($0, label) + length(label) - 1), calls, ms, w / 1e9, w / ms / 1e9
    }
    END {
      for (i = 1; i <= nn; i++) {
        if (nn == 1) w = total - known
        else if (nat[i] ~ /_ri8/) w = down
        else w = o
        printf "%-22s %-60s %4s       %7.3f ms %8.2f GFLOP %6.1f TFLOP/s\n", dir "/" file, nat[i] " (native kind)", "", natms[i], w / 1e9, w / natms[i] / 1e9
      }
    }' "$file"
done
